"""LiveKit voice worker using GPT-Live client delegation and mentatd."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import aiohttp
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    AudioConfig,
    BackgroundAudioPlayer,
    JobContext,
    WorkerOptions,
    get_job_context,
)
from livekit.agents.voice import room_io
from livekit.plugins import dtln, silero
from livekit.plugins.openai.realtime import GPTLiveDelegation, GPTLiveModel

from request import (
    END_CONVERSATION_TOOL,
    PRIVATE_CONTEXT_ENV,
    DelegationRunner,
    EndingPolicy,
    PrivateContext,
    consult_envelope,
    load_private_context,
    recent_turns,
    run_close_sequence,
    split_persona,
    turn_request,
    with_private_context,
)
from stream import CommentaryChunker, ToolResult, TurnDone, TurnError, TurnFailure, TurnStream

logger = logging.getLogger("mentat.voice")

DEFAULT_MENTAT_URL = "http://127.0.0.1:8484"
GPT_LIVE_VOICE = "meridian"
TIMEOUT = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
HERE = Path(__file__).parent
PERSONA_PATH = HERE / "persona.md"
EARCON_PATH = HERE / "assets" / "earcon.wav"
CONSULT_FAILED = (
    "Mentat could not be reached, so there is no answer to that question. "
    "Tell Josh briefly and offer to try again."
)


def load_persona(path: Path = PERSONA_PATH) -> tuple[str, str]:
    """Read worker instructions and the backend voice card."""
    return split_persona(path.read_text())


class FrontAgent(Agent):
    """The GPT-Live voice that delegates every backend request."""

    def __init__(
        self,
        *,
        instructions: str,
        voice_card: str,
        room_name: str,
        mentat_url: str,
        background: BackgroundAudioPlayer,
        ending_policy: EndingPolicy,
        ending_changed: Callable[[], None],
    ) -> None:
        super().__init__(instructions=instructions)
        self._voice_card = voice_card
        self._room_name = room_name
        self._mentat_url = mentat_url
        self._background = background
        self._ending_policy = ending_policy
        self._ending_changed = ending_changed
        self._delegations = DelegationRunner(self._run_delegation, self._on_delegation_error)

    async def on_enter(self) -> None:
        self.duplex_session.on("delegation_created", self._on_delegation_created)

    async def on_exit(self) -> None:
        await self._delegations.close()

    def _on_delegation_created(self, delegation: GPTLiveDelegation) -> None:
        """A plugin read-loop callback that must hand work to an asyncio task."""
        logger.info(
            "delegation %s created: %r", delegation.id, delegation.pending_transcript[:200]
        )
        self._ending_policy.delegation_started()
        self._ending_changed()
        self._background.play(AudioConfig(str(EARCON_PATH)))
        self._delegations.start(delegation)

    def _on_delegation_error(self, delegation_id: str, error: BaseException) -> None:
        logger.exception(
            "delegation failed unexpectedly: %s (%s)",
            delegation_id,
            error,
            exc_info=(type(error), error, error.__traceback__),
        )

    async def _run_delegation(self, delegation: GPTLiveDelegation) -> None:
        try:
            await self._stream_backend(delegation)
        except asyncio.CancelledError:
            raise
        except (TurnError, aiohttp.ClientError, TimeoutError) as error:
            logger.warning("delegation failed: %s", error)
            self.duplex_session.append_commentary(
                CONSULT_FAILED,
                delegation_id=delegation.id,
            )

    async def _stream_backend(self, delegation: GPTLiveDelegation) -> None:
        envelope = consult_envelope(
            self._voice_card,
            summary="",
            last_turns=recent_turns(self.chat_ctx.items),
            question=delegation.pending_transcript,
        )
        turn = TurnStream()
        chunker = CommentaryChunker()
        saw_text = False
        end_tool_succeeded = False
        async with aiohttp.ClientSession(timeout=TIMEOUT) as http:
            async with http.post(
                f"{self._mentat_url}/v1/conversation",
                json=turn_request(self._room_name, envelope),
            ) as response:
                if response.status != 200:
                    raise TurnError(f"daemon answered HTTP {response.status}")
                done_seen = False
                async for data in response.content.iter_any():
                    for item in turn.feed(data):
                        if isinstance(item, str):
                            saw_text = True
                            for chunk in chunker.feed(item):
                                self.duplex_session.append_commentary(
                                    chunk,
                                    delegation_id=delegation.id,
                                )
                        elif isinstance(item, ToolResult):
                            logger.info(
                                "delegation %s: tool %s%s",
                                delegation.id,
                                item.name,
                                " failed" if item.is_error else "",
                            )
                            if item.name == END_CONVERSATION_TOOL and not item.is_error:
                                end_tool_succeeded = True
                            self._ending_policy.tool_result_seen(item.name, item.is_error)
                            self._ending_changed()
                        elif isinstance(item, TurnFailure):
                            raise TurnError(item.message)
                        elif isinstance(item, TurnDone):
                            for chunk in chunker.flush():
                                saw_text = True
                                self.duplex_session.append_commentary(
                                    chunk,
                                    delegation_id=delegation.id,
                                )
                            done_seen = True
                            break
                    if done_seen:
                        break
        if not turn.done:
            raise TurnError("stream ended without done")
        if not saw_text and not end_tool_succeeded:
            raise TurnError("turn produced no commentary")
        self._ending_policy.turn_done(time.monotonic())
        logger.info(
            "delegation %s done: text=%s end_tool=%s; %s",
            delegation.id,
            saw_text,
            end_tool_succeeded,
            self._ending_policy.describe(),
        )
        self._ending_changed()


def log_turn_metrics(session: AgentSession) -> None:
    """Log the duration reported by the GPT-Live realtime model."""

    @session.on("metrics_collected")
    def _on_metrics(event: Any) -> None:
        metrics = getattr(event, "metrics", event)
        duration = getattr(metrics, "session_duration", None)
        if duration is not None:
            logger.info("session duration %.3fs", duration)


def prewarm(proc: agents.JobProcess) -> None:
    """Load process-shared voice state before a room is assigned."""
    proc.userdata["vad"] = silero.VAD.load()
    private = load_private_context(os.environ.get(PRIVATE_CONTEXT_ENV))
    proc.userdata["private"] = private
    logger.info(
        "private context: about=%d words, pronunciations=%d",
        len(private.about.split()),
        len(private.pronunciations),
    )


async def entrypoint(ctx: JobContext) -> None:
    """Serve one room until GPT-Live or the close policy ends it."""
    await ctx.connect()
    instructions, voice_card = load_persona()
    private: PrivateContext = ctx.proc.userdata["private"]
    instructions = with_private_context(instructions, private)
    ending_policy = EndingPolicy()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        llm=GPTLiveModel(voice=GPT_LIVE_VOICE, delegation="client"),
    )
    log_turn_metrics(session)

    background = BackgroundAudioPlayer()
    await background.start(room=ctx.room, agent_session=session)

    timer_task: asyncio.Task[None] | None = None

    async def _wait_for_deadline(seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        if ending_policy.elapsed(time.monotonic()) == "close":
            logger.info("closing: window elapsed")
            session.shutdown()
        else:
            logger.info("window elapsed without close; %s", ending_policy.describe())

    def _rearm_timer() -> None:
        nonlocal timer_task
        if timer_task is not None:
            timer_task.cancel()
            timer_task = None
        remaining = ending_policy.remaining(time.monotonic())
        if remaining is not None:
            logger.info("close window armed: %.1f s left", remaining)
            timer_task = asyncio.create_task(_wait_for_deadline(remaining))

    cleanup_task: asyncio.Task[None] | None = None

    async def _delete_room() -> None:
        await ctx.delete_room()

    async def _shutdown_job() -> None:
        get_job_context().shutdown()

    @session.on("close")
    def _on_close(_: Any) -> None:
        nonlocal cleanup_task, timer_task
        if timer_task is not None:
            timer_task.cancel()
            timer_task = None
        if cleanup_task is None:
            logger.info("voice session closing")
            cleanup_task = asyncio.create_task(
                run_close_sequence(
                    background.aclose,
                    _delete_room,
                    _shutdown_job,
                    logger.error,
                )
            )

    async def _await_cleanup() -> None:
        if cleanup_task is not None:
            await cleanup_task

    ctx.add_shutdown_callback(_await_cleanup)

    @session.on("agent_state_changed")
    def _on_agent_state(event: Any) -> None:
        old_state = getattr(event, "old_state", None)
        new_state = getattr(event, "new_state", None)
        if new_state == "listening":
            ending_policy.agent_listening(time.monotonic())
        elif new_state in {"thinking", "speaking"}:
            ending_policy.agent_busy()
        if new_state == "speaking" and old_state != "speaking":
            ending_policy.agent_speaking()
        elif old_state == "speaking" and new_state != "speaking":
            ending_policy.agent_quiet(time.monotonic())
        logger.info("agent %s -> %s; %s", old_state, new_state, ending_policy.describe())
        _rearm_timer()

    agent = FrontAgent(
        instructions=instructions,
        voice_card=voice_card,
        room_name=ctx.room.name,
        mentat_url=os.environ.get("MENTAT_URL", DEFAULT_MENTAT_URL),
        background=background,
        ending_policy=ending_policy,
        ending_changed=_rearm_timer,
    )

    @session.on("user_state_changed")
    def _on_user_state(event: Any) -> None:
        if event.new_state == "speaking":
            ending_policy.user_spoke()
        elif event.new_state in {"listening", "away"}:
            ending_policy.user_quiet()
        logger.info(
            "user %s -> %s; %s",
            getattr(event, "old_state", None),
            event.new_state,
            ending_policy.describe(),
        )
        _rearm_timer()

    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=dtln.noise_suppression(),
            ),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            host="127.0.0.1",
            port=int(os.environ.get("MENTAT_VOICE_HTTP_PORT", "8482")),
        )
    )
