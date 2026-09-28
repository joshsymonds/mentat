"""LiveKit voice worker streaming mentatd responses through Flux and Sonic."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import AsyncGenerator, Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    inference,
)
from livekit.agents.voice import room_io
from livekit.plugins import dtln, silero

from request import (
    END_CONVERSATION_TOOL,
    PRIVATE_CONTEXT_ENV,
    EndingPolicy,
    PrivateContext,
    call_context,
    consult_envelope,
    load_private_context,
    recent_turns,
    run_close_sequence,
    split_persona,
    turn_request,
    with_private_context,
)
from stream import ToolResult, ToolStart, TurnDone, TurnError, TurnFailure, TurnStream

logger = logging.getLogger("mentat.voice")

DEFAULT_MENTAT_URL = "http://127.0.0.1:8484"
TIMEOUT = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
CONSULT_FAILED = "I couldn't complete that request with Mentat. Please try again."
HERE = Path(__file__).parent
PERSONA_PATH = HERE / "persona.md"
EARCON_PATH = HERE / "assets" / "earcon.wav"
TTS_VOICE = '47c38ca4-5f35-497b-b1a3-415245fb35e1'
SEND_SMS_TOOL = "mcp__mentat__send_sms"


def is_sms_request(text: str) -> bool:
    """Recognize an explicit request to send a text before backend speech starts."""
    return bool(
        re.search(r"\b(?:send|sending)\b.{0,60}\b(?:text|sms|message)\b", text, re.I)
        or re.search(
            r"^\s*(?:(?:please|can you|could you|would you)\s+)?"
            r"(?:text|message)\s+(?!from\b|about\b|that\b|is\b|was\b)"
            r"(?:to\s+)?[\w+][\w.'+-]*",
            text,
            re.I,
        )
        or re.search(r"\b(?:want|like|need)\s+to\s+(?:text|message)\b", text, re.I)
    )


def is_sms_followup(text: str) -> bool:
    """Recognize short consent or correction turns for an outstanding text."""
    return bool(
        re.search(
            r"^\s*(?:yes|yeah|yep|sure|okay|ok|send it|go ahead|that's right|correct)\b",
            text,
            re.I,
        )
        or re.search(
            r"\b(?:actually|change|correct|edit|replace|revise|update|instead|make it|meant)\b",
            text,
            re.I,
        )
        or re.search(r"\b(?:wrong|not what i meant)\b", text, re.I)
    )


def is_sms_decline(text: str) -> bool:
    """Recognize an explicit cancellation of the pending text."""
    return bool(
        re.search(r"^\s*(?:no|nope)(?:\s*[.!?]|\s*$)", text, re.I)
        or re.search(
            r"^\s*(?:no|nope)\s*,?\s*(?:don't send|do not send|cancel)\b", text, re.I
        )
        or re.search(r"\b(?:cancel|never mind|forget it)\b", text, re.I)
    )


class SmsCommentaryBuffer:
    """Append a complete SMS delegation as one unit, after any send result."""

    def __init__(self, append: Callable[[str], None]) -> None:
        self._append = append
        self._parts: list[str] = []
        self._send_sms_seen = False
        self._send_sms_succeeded = False
        self._finished = False

    def add(self, text: str) -> None:
        self._parts.append(text)

    def tool_result(self, name: str, *, is_error: bool) -> None:
        if name == SEND_SMS_TOOL:
            self._send_sms_seen = True
            self._send_sms_succeeded = not is_error

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self._send_sms_seen and not self._send_sms_succeeded:
            self._append("I couldn't send that text. Please try again.")
            return
        text = "".join(self._parts)
        if text:
            self._append(text)


def load_persona(path: Path = PERSONA_PATH) -> tuple[str, str]:
    """Read worker instructions and the backend voice card."""
    return split_persona(path.read_text())


def write_turn_marker(room_name: str, turn_id: str) -> None:
    """Write the opt-in private marker for a backend voice turn."""
    marker_path = os.environ.get("MENTAT_EVAL_DELEGATION_LOG")
    if not marker_path:
        return
    marker = json.dumps(
        {"room": room_name, "id": turn_id, "created_at": time.time()},
        separators=(",", ":"),
        allow_nan=False,
    )
    try:
        with Path(marker_path).open("a", encoding="utf-8") as marker_file:
            marker_file.write(marker + "\n")
    except OSError:
        logger.exception("failed to write eval delegation marker")


class FrontAgent(Agent):
    """A text-only front that streams every user turn through mentatd."""

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
        self._sms_consent = False

    async def on_user_turn_completed(self, chat_ctx: Any, new_message: Any) -> None:
        """Ask mentatd for this turn and speak its streamed text verbatim."""
        question = str(getattr(new_message, "text_content", "")).strip()
        if not question:
            return
        turn_id = uuid4().hex
        self._ending_policy.delegation_started()
        self._ending_changed()
        self._background.play(AudioConfig(str(EARCON_PATH)))
        backend_text = self._backend_text(question, turn_id, chat_ctx)
        speech_queue: asyncio.Queue[str | None] | None = None
        speech_handle: Any = None
        speech_completion: asyncio.Task[None] | None = None

        async def speech_source(
            queue: asyncio.Queue[str | None],
        ) -> AsyncGenerator[str, None]:
            while (text := await queue.get()) is not None:
                yield text

        def start_speech() -> tuple[
            asyncio.Queue[str | None], Any, asyncio.Task[None]
        ]:
            queue: asyncio.Queue[str | None] = asyncio.Queue()
            handle = self.session.say(speech_source(queue), allow_interruptions=True)
            completion = asyncio.create_task(handle.wait_for_playout())
            return queue, handle, completion

        async def finish_speech() -> bool:
            nonlocal speech_queue, speech_handle, speech_completion
            if speech_queue is None or speech_handle is None or speech_completion is None:
                return False
            queue, handle, completion = speech_queue, speech_handle, speech_completion
            speech_queue = None
            speech_handle = None
            speech_completion = None
            await queue.put(None)
            try:
                await completion
            except asyncio.CancelledError:
                if handle.interrupted:
                    logger.info("delegation %s interrupted", turn_id)
                    return True
                handle.interrupt()
                raise
            if handle.interrupted:
                logger.info("delegation %s interrupted", turn_id)
                return True
            speech_exception = handle.exception()
            if speech_exception is not None:
                raise speech_exception
            return False

        backend_iterator = backend_text.__aiter__()
        backend_next: asyncio.Task[str | None] | None = None
        try:
            while True:
                backend_next = asyncio.create_task(backend_iterator.__anext__())
                if speech_completion is None:
                    await backend_next
                else:
                    done, _ = await asyncio.wait(
                        (backend_next, speech_completion),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if speech_completion in done:
                        if speech_handle is not None and speech_handle.interrupted:
                            backend_next.cancel()
                            await asyncio.gather(backend_next, return_exceptions=True)
                            backend_next = None
                            logger.info("delegation %s interrupted", turn_id)
                            return
                        await speech_completion
                        raise RuntimeError("speech ended before the backend segment boundary")
                try:
                    text = backend_next.result()
                except StopAsyncIteration:
                    backend_next = None
                    break
                backend_next = None
                if text is None:
                    if await finish_speech():
                        return
                    continue
                if speech_queue is None:
                    speech_queue, speech_handle, speech_completion = start_speech()
                await speech_queue.put(text)
            if await finish_speech():
                return
        finally:
            if backend_next is not None:
                backend_next.cancel()
                await asyncio.gather(backend_next, return_exceptions=True)
            if speech_handle is not None:
                speech_handle.interrupt()
                if speech_completion is not None:
                    await asyncio.gather(speech_completion, return_exceptions=True)
            await backend_text.aclose()

    async def _backend_text(
        self, question: str, turn_id: str, chat_ctx: Any
    ) -> AsyncGenerator[str | None, None]:
        sms_mode = is_sms_request(question)
        sms_declined = self._sms_consent and is_sms_decline(question)
        if self._sms_consent and (is_sms_followup(question) or sms_declined):
            sms_mode = True
        elif self._sms_consent and not sms_mode:
            self._sms_consent = False
        if is_sms_request(question):
            self._sms_consent = True
        if sms_declined:
            self._sms_consent = False

        envelope = consult_envelope(
            self._voice_card,
            summary="",
            last_turns=recent_turns(chat_ctx.items),
            question=question,
        )
        write_turn_marker(self._room_name, turn_id)
        turn = TurnStream()
        pending_sms_text: list[str] = []
        sms_buffer = SmsCommentaryBuffer(pending_sms_text.append) if sms_mode else None
        commentary_logged = False
        commentary_tail = ""
        saw_text = False
        end_tool_succeeded = False

        def log_first_commentary() -> None:
            nonlocal commentary_logged
            if not commentary_logged:
                logger.info("delegation %s first commentary", turn_id)
                commentary_logged = True

        try:
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
                                if sms_buffer is not None:
                                    sms_buffer.add(item)
                                else:
                                    commentary_tail = (commentary_tail + item)[-16:]
                                    log_first_commentary()
                                    yield item
                            elif isinstance(item, ToolStart):
                                if sms_buffer is None and commentary_tail:
                                    ending = commentary_tail.rstrip()
                                    if not re.search(r"[.!?…][\"'”’)]*$", ending):
                                        yield ". "
                                    elif ending == commentary_tail:
                                        yield " "
                                    yield None
                                    commentary_tail = ""
                            elif isinstance(item, ToolResult):
                                if sms_buffer is not None:
                                    sms_buffer.tool_result(item.name, is_error=item.is_error)
                                if item.name == SEND_SMS_TOOL:
                                    self._sms_consent = False
                                logger.info(
                                    "delegation %s: tool %s%s",
                                    turn_id,
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
                                if sms_buffer is not None:
                                    sms_buffer.finish()
                                    for text in pending_sms_text:
                                        if text:
                                            log_first_commentary()
                                            yield text
                                done_seen = True
                                break
                        if done_seen:
                            break
            if not turn.done:
                raise TurnError("stream ended without done")
            if not saw_text and not end_tool_succeeded:
                raise TurnError("turn produced no commentary")
        except asyncio.CancelledError:
            raise
        except (TurnError, aiohttp.ClientError, TimeoutError) as error:
            logger.warning("delegation %s failed: %s", turn_id, error)
            yield CONSULT_FAILED
            return
        self._ending_policy.turn_done(time.monotonic())
        logger.info(
            "delegation %s done: text=%s end_tool=%s; %s",
            turn_id,
            saw_text,
            end_tool_succeeded,
            self._ending_policy.describe(),
        )
        self._ending_changed()

def log_turn_metrics(session: AgentSession) -> None:
    """Log the duration reported by the voice session."""

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
        "private context: about=%d words, pronunciations=%d, places=%d",
        len(private.about.split()),
        len(private.pronunciations),
        len(private.places),
    )


async def entrypoint(ctx: JobContext) -> None:
    """Serve one room until mentatd or the close policy ends it."""
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=inference.STT("deepgram/flux-general"),
        tts=inference.TTS("cartesia/sonic-3.6", voice=TTS_VOICE),
        turn_handling={
            "turn_detection": "stt",
            "endpointing": {"min_delay": 0.5},
        },
    )
    voice_room_io = room_io.RoomIO(
        agent_session=session,
        room=ctx.room,
        options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=dtln.noise_suppression(),
            ),
        ),
    )
    voice_room_io_closed = False

    async def _close_voice_room_io() -> None:
        nonlocal voice_room_io_closed
        if not voice_room_io_closed:
            try:
                await voice_room_io.aclose()
            except Exception:
                logger.exception("voice RoomIO cleanup failed")
            else:
                voice_room_io_closed = True

    ctx.add_shutdown_callback(_close_voice_room_io)
    await voice_room_io.start()
    await ctx.connect()
    instructions, voice_card = load_persona()
    private: PrivateContext = ctx.proc.userdata["private"]
    instructions = with_private_context(instructions, private)
    # The caller's token carries call context, which the voice agent keeps for
    # the call before the first turn is handled.
    caller = await ctx.wait_for_participant()
    context = call_context(caller.attributes, private.places, datetime.now().astimezone())
    logger.info("%s (attributes: %s)", context, sorted(caller.attributes))
    instructions += "\n\n" + context
    ending_policy = EndingPolicy()

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

    async def _close_audio() -> None:
        await _close_voice_room_io()
        await background.aclose()

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
                    _close_audio,
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

    await session.start(agent=agent)


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            host="127.0.0.1",
            port=int(os.environ.get("MENTAT_VOICE_HTTP_PORT", "8482")),
        )
    )
