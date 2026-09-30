"""LiveKit voice worker streaming mentatd responses through direct providers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import wave
from collections.abc import AsyncGenerator, AsyncIterable, Callable
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
    tts,
)
from livekit.agents.voice import room_io
from livekit.plugins import dtln, elevenlabs, openai, silero, turn_detector

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
TURN_CONTINUATION_WINDOW = 3.0
CONSULT_FAILED = "I couldn't complete that request with Mentat. Please try again."
HERE = Path(__file__).parent
PERSONA_PATH = HERE / "persona.md"
EARCON_PATH = HERE / "assets" / "earcon.wav"
TTS_VOICE = "21m00Tcm4TlvDq8ikWAM"
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
    """Append a complete SMS delegation after its result, retaining tool boundaries."""

    def __init__(self, append: Callable[[str | None], None]) -> None:
        self._append = append
        self._parts: list[str] = []
        self._blocks: list[str] = []
        self._send_sms_seen = False
        self._send_sms_succeeded = False
        self._finished = False

    def add(self, text: str) -> None:
        self._parts.append(text)

    def tool_boundary(self) -> None:
        if self._parts:
            self._blocks.append("".join(self._parts))
            self._parts.clear()

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
        if self._parts:
            self._blocks.append("".join(self._parts))
        if len(self._blocks) < 2:
            text = "".join(self._blocks)
            if text:
                self._append(text)
            return
        for index, block in enumerate(self._blocks):
            text = block
            if index < len(self._blocks) - 1:
                ending = text.rstrip().rstrip("\\\"'”’")
                if ending and ending[-1] not in ".!?…":
                    text = text.rstrip() + "."
            if text:
                self._append(text)
                if index < len(self._blocks) - 1:
                    self._append(None)


def load_persona(path: Path = PERSONA_PATH) -> tuple[str, str]:
    """Read worker instructions and the backend voice card."""
    return split_persona(path.read_text())


class InputAudioRecorder:
    """Keep STT input private and publish it only after a committed transcript."""

    def __init__(self, room_name: str, output_dir: str | Path | None = None) -> None:
        self._output_dir = Path(output_dir) if output_dir else None
        self._room_name = re.sub(r"[^A-Za-z0-9._-]+", "_", room_name).strip("._-") or "room"
        self._frames: list[tuple[bytes, int, int]] = []
        self._turn_number = 0

    def begin(self) -> None:
        if self._output_dir is not None:
            self._frames.clear()

    def capture(self, frame: Any) -> None:
        if self._output_dir is None:
            return
        try:
            self._frames.append(
                (bytes(frame.data), int(frame.sample_rate), int(frame.num_channels))
            )
        except Exception:
            logger.exception("failed to capture eval STT input")

    def commit(self, transcript: str) -> None:
        if self._output_dir is None:
            return
        frames, self._frames = self._frames, []
        if not frames:
            return
        number = self._turn_number + 1
        audio_path: Path | None = None
        transcript_path: Path | None = None
        audio_fd: int | None = None
        transcript_fd: int | None = None
        audio_created = False
        transcript_created = False
        try:
            sample_rate, channels = frames[0][1:]
            if any(rate != sample_rate or count != channels for _, rate, count in frames):
                raise ValueError("STT audio format changed within a committed turn")
            self._output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            while True:
                stem = f"{self._room_name}-turn-{number:03d}"
                audio_path = self._output_dir / f"{stem}.wav"
                transcript_path = self._output_dir / f"{stem}.txt"
                try:
                    audio_fd = os.open(
                        audio_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    audio_created = True
                except FileExistsError:
                    number += 1
                    continue
                try:
                    transcript_fd = os.open(
                        transcript_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    transcript_created = True
                except FileExistsError:
                    os.close(audio_fd)
                    audio_fd = None
                    audio_path.unlink()
                    audio_created = False
                    number += 1
                    continue
                break
            self._turn_number = number
            with os.fdopen(audio_fd, "wb") as audio_raw:
                audio_fd = None
                with wave.open(audio_raw, "wb") as audio_file:
                    audio_file.setnchannels(channels)
                    audio_file.setsampwidth(2)
                    audio_file.setframerate(sample_rate)
                    audio_file.writeframes(b"".join(data for data, _, _ in frames))
            with os.fdopen(transcript_fd, "w", encoding="utf-8") as transcript_file:
                transcript_fd = None
                transcript_file.write(transcript)
        except Exception:
            logger.exception("failed to write private eval STT input")
            for descriptor in (audio_fd, transcript_fd):
                if descriptor is not None:
                    try:
                        os.close(descriptor)
                    except OSError:
                        logger.exception("failed to close incomplete eval evidence")
            for path, created in (
                (audio_path, audio_created),
                (transcript_path, transcript_created),
            ):
                if path is not None and created:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        logger.exception("failed to remove incomplete eval STT evidence")


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
        input_audio_dir = os.environ.get("MENTAT_VOICE_INPUT_RECORD_DIR")
        self._input_audio = (
            InputAudioRecorder(room_name, input_audio_dir) if input_audio_dir else None
        )
        self._sms_consent = False
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_text = ""
        self._turn_started_at: float | None = None
        self._closed = False

    def stt_node(
        self, audio: AsyncIterable[Any], model_settings: Any
    ) -> AsyncIterable[Any]:
        """Record only the exact input frames handed to STT, when opted in."""
        if self._input_audio is None:
            return super().stt_node(audio, model_settings)

        self._input_audio.begin()

        async def recorded_audio() -> AsyncGenerator[Any, None]:
            async for frame in audio:
                self._input_audio.capture(frame)
                yield frame

        return super().stt_node(recorded_audio(), model_settings)

    async def on_user_turn_completed(self, chat_ctx: Any, new_message: Any) -> None:
        """Start backend work immediately, merging only a short continuation."""
        question = str(getattr(new_message, "text_content", "")).strip()
        if not question or self._closed:
            return
        input_audio = getattr(self, "_input_audio", None)
        if input_audio is not None:
            input_audio.commit(question)
        turn_started_at = time.monotonic()
        task = self._turn_task
        if task is not None and not task.done():
            started_at = self._turn_started_at
            if (
                started_at is not None
                and turn_started_at - started_at <= TURN_CONTINUATION_WINDOW
            ):
                question = f"{self._turn_text} {question}"
                turn_started_at = started_at
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        if self._closed:
            return
        self._turn_text = question
        self._turn_started_at = turn_started_at
        self._turn_task = asyncio.create_task(self._run_turn(question, chat_ctx))

    async def aclose(self) -> None:
        """Cancel and drain active backend and speech work before session close."""
        self._closed = True
        task = self._turn_task
        self._turn_task = None
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _run_turn(self, question: str, chat_ctx: Any) -> None:
        try:
            await self._speak_turn(question, chat_ctx)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("delegation turn failed")

    async def _speak_turn(self, question: str, chat_ctx: Any) -> None:
        """Ask mentatd for this turn and speak its streamed text verbatim."""
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
                    try:
                        await backend_next
                    except StopAsyncIteration:
                        backend_next = None
                        break
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
        pending_sms_text: list[str | None] = []
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
                                if sms_buffer is not None:
                                    sms_buffer.tool_boundary()
                                elif commentary_tail:
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
                                        if text is None:
                                            yield None
                                        elif text:
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
        stt=openai.STT(
            model="gpt-live-transcribe",
            api_key=os.environ["OPENAI_API_KEY"],
            vad=ctx.proc.userdata["vad"],
        ),
        tts=tts.StreamAdapter(
            tts=elevenlabs.TTS(
                model="eleven_v4_turbo",
                api_key=os.environ["ELEVENLABS_API_KEY"],
                voice_id=os.environ.get("MENTAT_VOICE_TTS_VOICE", TTS_VOICE),
            )
        ),
        turn_handling={
            "turn_detection": turn_detector.MultilingualModel(),
            "endpointing": {"min_delay": 0.5, "max_delay": 3.0},
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

    agent: FrontAgent | None = None

    async def _close_audio() -> None:
        if agent is not None:
            await agent.aclose()
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
