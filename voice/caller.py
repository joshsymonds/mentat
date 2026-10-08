"""Scripted LiveKit caller: python caller.py ROOM 'question@delay::answer-regex' ..."""

import asyncio
import math
import os
import re
import struct
import sys
import time
import wave
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

RATE = 24000
FRAME_SAMPLES = RATE // 100
MAX_ANSWER_SECONDS = 30
MAX_CAPTURE_SECONDS = 120
ANSWER_START_TIMEOUT_SECONDS = 30
ANSWER_END_SILENCE_SECONDS = 20
ANSWER_RMS_THRESHOLD = 200
END_DRAIN_SECONDS = 0.1
READINESS_TIMEOUT_SECONDS = 15
MICROPHONE_SETTLE_SECONDS = 3.0
ECHO_GAIN_DB = -30
ECHO_DELAY_SECONDS = 0.150
ECHO_GAIN = 10 ** (ECHO_GAIN_DB / 20)


@dataclass(frozen=True)
class TTSResponse:
    pcm: bytes
    http_status: int | None
    response_bytes: int


TRANSCRIPTION_MAX_ATTEMPTS = 4
_TRANSCRIPTION_RETRY_DELAYS = (0.5, 1.0, 2.0)


class TranscriptionError(RuntimeError):
    """A batch transcription provider rejected one captured clip with HTTP 4xx."""

    def __init__(self, status: int, code: str | None = None):
        if not 400 <= status < 500:
            raise ValueError("Transcription failures must be HTTP 4xx responses")
        self.status = status
        self.code = code
        label = f"; {code}" if code else ""
        super().__init__(f"Transcription rejected clip (HTTP {status}{label})")


class TranscriptionCapacityError(TranscriptionError):
    """Scribe rejected a request because provider concurrency was exhausted."""

    def __init__(self, status: int = 429):
        super().__init__(status, "concurrent_limit_exceeded")
        self.capacity_failure = {
            "source": "audio transcription",
            "cause": "ElevenLabs concurrent_limit_exceeded",
        }


class TranscriptionTimestampError(TranscriptionError):
    """Scribe returned text without usable per-word timestamps."""

    def __init__(self, text: str):
        RuntimeError.__init__(self, "Transcription rejected reply clip (missing word timestamps)")
        self.text = text
        self.status = None
        self.code = "missing_word_timestamps"


async def _transcription_retry_sleep(delay: float) -> None:
    await asyncio.sleep(delay)


async def wait_for_microphone_ready(
    publication: Any,
    *,
    deadline: float,
    monotonic: Any = time.monotonic,
    sleep: Any = asyncio.sleep,
) -> None:
    """Wait for microphone subscription and its fixed settle under one deadline."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("caller microphone subscription exceeded its deadline")
    try:
        await asyncio.wait_for(publication.wait_for_subscription(), timeout=remaining)
    except TimeoutError as error:
        raise TimeoutError("caller microphone subscription exceeded its deadline") from error

    remaining = deadline - monotonic()
    if remaining < MICROPHONE_SETTLE_SECONDS:
        raise TimeoutError("caller microphone subscription settle exceeded its deadline")
    try:
        await asyncio.wait_for(sleep(MICROPHONE_SETTLE_SECONDS), timeout=remaining)
    except TimeoutError as error:
        raise TimeoutError("caller microphone subscription settle exceeded its deadline") from error
    if monotonic() > deadline:
        raise TimeoutError("caller microphone subscription settle exceeded its deadline")


@dataclass(frozen=True)
class ScriptStep:
    delay: float
    line: str
    answer_pattern: str
    language: str = "en"


def parse_step_language(value: str) -> str:
    """Read an optional private scripted-caller language marker."""
    if not isinstance(value, str) or "::" not in value:
        raise ValueError("script step must be LINE@DELAY::ANSWER_REGEX")
    speech = value.rsplit("::", 1)[0]
    line = speech.rpartition("@")[0]
    match = re.match(r"^\[\[([a-z]{2,3})\]\]", line)
    if match is None:
        return "en"
    language = match.group(1)
    if language not in {"en", "es"}:
        raise ValueError("script language must be en or es")
    return language


def parse_step(value: str) -> tuple[float, str, str]:
    """Parse LINE@DELAY::ANSWER_REGEX, keeping regex syntax literal."""
    if "::" not in value:
        raise ValueError("script step must be LINE@DELAY::ANSWER_REGEX")
    speech, pattern = value.rsplit("::", 1)
    line, separator, delay_text = speech.rpartition("@")
    if not separator or not line or not pattern:
        raise ValueError("script step must be LINE@DELAY::ANSWER_REGEX")
    language = parse_step_language(value)
    if language != "en":
        line = line[len(f"[[{language}]]"):]
    try:
        delay = float(delay_text)
    except ValueError as exc:
        raise ValueError("script delay must be a non-negative number") from exc
    if not delay >= 0 or not delay < float("inf"):
        raise ValueError("script delay must be a non-negative number")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid answer regex: {exc}") from exc
    return delay, line, pattern


def is_agent_audio_track(
    participant_kind: Any,
    track_kind: Any,
    publication_source: Any,
    agent_kind: Any,
    audio_kind: Any,
    microphone_source: Any,
) -> bool:
    """Select only microphone audio tracks published by LiveKit agent participants."""
    return (
        participant_kind == agent_kind
        and track_kind == audio_kind
        and publication_source == microphone_source
    )


def first_matching_latency(
    segments: list[dict[str, Any]], pattern: str, speech_end: float, capture_started: float
) -> float | None:
    """Return seconds from caller speech end to the start of the segment holding the first match."""
    try:
        matcher = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid answer regex: {exc}") from exc
    texts = [str(segment.get("text", "")) for segment in segments]
    match = matcher.search(" ".join(texts))
    if match is None or not segments:
        return None
    offset = 0
    for segment, text in zip(segments, texts, strict=True):
        if match.start() <= offset + len(text):
            return capture_started + float(segment["start"]) - speech_end
        offset += len(text) + 1
    return None


def _segments_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text", "")) for segment in segments).strip()


def _format_segments(segments: list[dict[str, Any]]) -> str:
    return " ".join(
        f"[{float(segment['start']):.2f}-{float(segment['end']):.2f}] {str(segment.get('text', '')).strip()}"
        for segment in segments
    )


def _frame_has_voice(frame: Any) -> bool:
    data = bytes(frame.data)
    if not data or len(data) % 2:
        return False
    samples = [sample[0] for sample in struct.iter_unpack("<h", data)]
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    return rms >= ANSWER_RMS_THRESHOLD


def _frame_duration(frame: Any) -> float:
    return frame.samples_per_channel / frame.sample_rate


def _mono_rate_samples(frame: Any) -> list[int]:
    if frame.sample_rate == RATE:
        factor = 1
    elif frame.sample_rate == 2 * RATE:
        factor = 2
    else:
        raise ValueError("agent audio must be 24 kHz or 48 kHz")
    channels = frame.num_channels
    data = bytes(frame.data)
    interleaved = struct.unpack(f"<{len(data) // 2}h", data)
    mono = [
        sum(interleaved[offset : offset + channels]) // channels
        for offset in range(0, len(interleaved), channels)
    ]
    return [
        sum(mono[offset : offset + factor]) // len(mono[offset : offset + factor])
        for offset in range(0, len(mono), factor)
    ]


class EchoResidual:
    """Mix an attenuated copy of agent audio into mic PCM, heard ECHO_DELAY_SECONDS earlier."""

    def __init__(self) -> None:
        self._heard: deque[tuple[float, list[int]]] = deque()

    def feed(self, frame: Any, heard_at: float) -> None:
        self._heard.append((heard_at, _mono_rate_samples(frame)))

    def mix(self, mic_pcm: bytes, started_at: float) -> bytes:
        oldest_needed = started_at - ECHO_DELAY_SECONDS
        while self._heard and self._heard[0][0] + len(self._heard[0][1]) / RATE < oldest_needed:
            self._heard.popleft()
        mic = struct.unpack(f"<{len(mic_pcm) // 2}h", mic_pcm)
        mixed = []
        for index, sample in enumerate(mic):
            echo = self._echo_at(started_at + index / RATE - ECHO_DELAY_SECONDS)
            mixed.append(max(-32768, min(32767, sample + round(ECHO_GAIN * echo))))
        return struct.pack(f"<{len(mixed)}h", *mixed)

    def _echo_at(self, when: float) -> int:
        for heard_at, samples in self._heard:
            offset = round((when - heard_at) * RATE)
            if 0 <= offset < len(samples):
                return samples[offset]
        return 0


async def _wait_any(events: list[asyncio.Event]) -> None:
    waits = [asyncio.create_task(event.wait()) for event in events]
    try:
        await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for wait in waits:
            wait.cancel()
        await asyncio.gather(*waits, return_exceptions=True)


async def _capture_answer(
    track_queue: asyncio.Queue[Any],
    ended: asyncio.Event | None = None,
    stop: asyncio.Event | None = None,
    on_voice: Callable[[float], None] | None = None,
    echo: EchoResidual | None = None,
) -> tuple[bytes, int, int, float]:
    from livekit import rtc

    stops = [event for event in (ended, stop) if event is not None]
    if stops and not track_queue.empty():
        track = track_queue.get_nowait()
    elif not stops:
        track = await asyncio.wait_for(track_queue.get(), timeout=15)
    else:
        track_task = asyncio.create_task(track_queue.get())
        ended_task = asyncio.create_task(_wait_any(stops))
        done, pending = await asyncio.wait(
            (track_task, ended_task), timeout=15, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if track_task not in done:
            raise RuntimeError("agent audio track produced no frames")
        track = track_task.result()
    capture_started: float | None = None
    audio_stream = rtc.AudioStream(track)
    frames: list[bytes] = []
    last_frame: Any = None
    loop = asyncio.get_running_loop()
    try:
        no_frame_deadline = loop.time() + min(ANSWER_START_TIMEOUT_SECONDS, MAX_CAPTURE_SECONDS)
        capture_started_at = 0.0
        capture_deadline = 0.0
        answer_start_deadline = 0.0
        answer_window_deadline = 0.0
        last_voice_at = 0.0
        captured_seconds = 0.0
        heard_voice = False
        trailing_silence = 0.0
        while True:
            if stop is not None and stop.is_set():
                break
            if capture_started is None:
                deadline = no_frame_deadline
            elif not heard_voice:
                deadline = min(capture_deadline, answer_start_deadline)
            else:
                idle_completion_deadline = max(
                    answer_window_deadline,
                    last_voice_at + ANSWER_END_SILENCE_SECONDS,
                )
                elapsed = max(captured_seconds, loop.time() - capture_started_at)
                quiet = max(trailing_silence, loop.time() - last_voice_at)
                if elapsed >= MAX_ANSWER_SECONDS and quiet >= ANSWER_END_SILENCE_SECONDS:
                    break
                deadline = min(capture_deadline, idle_completion_deadline)
            remaining = max(0.0, deadline - loop.time())
            try:
                if not stops:
                    event = await asyncio.wait_for(audio_stream.__anext__(), timeout=remaining)
                else:
                    next_task = asyncio.Task(audio_stream.__anext__(), eager_start=True)
                    if any(event.is_set() for event in stops):
                        try:
                            event = await asyncio.wait_for(next_task, timeout=min(remaining, END_DRAIN_SECONDS))
                        except TimeoutError:
                            break
                    else:
                        ended_task = asyncio.create_task(_wait_any(stops))
                        done, pending = await asyncio.wait(
                            (next_task, ended_task),
                            timeout=remaining,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if next_task in done:
                            ended_task.cancel()
                            await asyncio.gather(ended_task, return_exceptions=True)
                            event = next_task.result()
                        elif ended_task in done:
                            try:
                                event = await asyncio.wait_for(next_task, timeout=END_DRAIN_SECONDS)
                            except TimeoutError:
                                next_task.cancel()
                                await asyncio.gather(next_task, return_exceptions=True)
                                break
                        else:
                            for task in pending:
                                task.cancel()
                            await asyncio.gather(*pending, return_exceptions=True)
                            if capture_started is None:
                                break
                            if loop.time() >= capture_deadline:
                                raise RuntimeError("agent audio capture exceeded its deadline")
                            if not heard_voice:
                                raise RuntimeError("agent audio response did not start before its deadline")
                            break
            except TimeoutError as error:
                if capture_started is None:
                    break
                if loop.time() >= capture_deadline:
                    raise RuntimeError("agent audio capture exceeded its deadline") from error
                if not heard_voice:
                    raise RuntimeError("agent audio response did not start before its deadline") from error
                break
            except StopAsyncIteration:
                break
            last_frame = event.frame
            if echo is not None:
                echo.feed(last_frame, time.monotonic())
            if capture_started is None:
                capture_started = time.monotonic()
                capture_started_at = loop.time()
                capture_deadline = capture_started_at + MAX_CAPTURE_SECONDS
                answer_start_deadline = capture_started_at + ANSWER_START_TIMEOUT_SECONDS
                answer_window_deadline = capture_started_at + MAX_ANSWER_SECONDS
            frame_seconds = _frame_duration(last_frame)
            captured_seconds += frame_seconds
            frames.append(bytes(last_frame.data))
            if _frame_has_voice(last_frame):
                if not heard_voice and on_voice is not None:
                    on_voice(time.monotonic())
                heard_voice = True
                last_voice_at = loop.time()
                trailing_silence = 0.0
            elif heard_voice:
                trailing_silence += frame_seconds
    finally:
        await audio_stream.aclose()
        if ended is None or not ended.is_set():
            track_queue.put_nowait(track)
    if not frames or capture_started is None or last_frame is None:
        raise RuntimeError("agent audio track produced no frames")
    if not heard_voice:
        raise RuntimeError("agent audio track produced no speech")
    return b"".join(frames), last_frame.sample_rate, last_frame.num_channels, capture_started


class ContinuousCapture:
    """Capture agent audio from before caller speech through the answer window."""

    def __init__(
        self,
        track_queue: asyncio.Queue[Any],
        ended: asyncio.Event | None = None,
        echo: EchoResidual | None = None,
    ) -> None:
        self._track_queue = track_queue
        self._ended = ended
        self._echo = echo
        self._task: asyncio.Task[tuple[bytes, int, int, float]] | None = None
        self._stop = asyncio.Event()
        self._voice: asyncio.Future[float] | None = None

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("continuous capture already started")
        voice = asyncio.get_running_loop().create_future()
        self._voice = voice

        def on_voice(onset: float) -> None:
            if not voice.done():
                voice.set_result(onset)

        def settle_voice(task: asyncio.Task[tuple[bytes, int, int, float]]) -> None:
            if voice.done():
                return
            if task.cancelled():
                voice.cancel()
                return
            error = task.exception() or RuntimeError("agent audio track produced no speech")
            voice.set_exception(error)
            voice.exception()

        self._task = asyncio.create_task(
            _capture_answer(self._track_queue, self._ended, self._stop, on_voice, self._echo)
        )
        self._task.add_done_callback(settle_voice)
        await asyncio.sleep(0)

    async def wait_for_voice(self) -> float:
        if self._voice is None:
            raise RuntimeError("continuous capture has not started")
        return await asyncio.shield(self._voice)

    def stop(self) -> None:
        if self._task is None:
            raise RuntimeError("continuous capture has not started")
        self._stop.set()

    async def result(self) -> tuple[bytes, int, int, float]:
        if self._task is None:
            raise RuntimeError("continuous capture has not started")
        return await self._task


async def _speech_end_after_playout(source: Any) -> float:
    await source.wait_for_playout()
    return time.monotonic()


async def _tts(http: Any, text: str, *, language: str = "en") -> TTSResponse:
    request = {
        "model": "gpt-4o-mini-tts",
        "voice": "ash",
        "input": text,
        "response_format": "pcm",
    }
    if language == "es":
        request["instructions"] = "Speak clearly in Spanish."
    elif language != "en":
        raise ValueError("script language must be en or es")
    async with http.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json=request,
    ) as response:
        pcm = await response.read()
        response.raise_for_status()
        return TTSResponse(pcm, response.status, len(pcm))


async def _transcribe(http: Any, pcm: bytes, sample_rate: int, channels: int) -> list[dict[str, Any]]:
    if len(pcm) * 10 < sample_rate * channels * 2:
        return []

    import aiohttp
    from aiohttp import FormData

    import io

    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    wav_data = wav_bytes.getvalue()

    for attempt in range(TRANSCRIPTION_MAX_ATTEMPTS):
        form = FormData()
        form.add_field("file", wav_data, filename="answer.wav", content_type="audio/wav")
        form.add_field("model_id", "scribe_v2")
        form.add_field("tag_audio_events", "false")
        async with http.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
            data=form,
        ) as response:
            status = getattr(response, "status", None)
            if status == 429:
                try:
                    error_payload = await response.json()
                except (aiohttp.ContentTypeError, ValueError, TypeError):
                    error_payload = None
                detail = error_payload.get("detail") if isinstance(error_payload, dict) else None
                code = detail.get("code") if isinstance(detail, dict) else None
                if attempt + 1 < TRANSCRIPTION_MAX_ATTEMPTS:
                    await _transcription_retry_sleep(_TRANSCRIPTION_RETRY_DELAYS[attempt])
                    continue
                if code == "concurrent_limit_exceeded":
                    raise TranscriptionCapacityError(status)
                raise TranscriptionError(status, code if isinstance(code, str) else None)
            if isinstance(status, int) and 400 <= status < 500:
                try:
                    error_payload = await response.json()
                except (aiohttp.ContentTypeError, ValueError, TypeError):
                    error_payload = None
                detail = error_payload.get("detail") if isinstance(error_payload, dict) else None
                code = detail.get("code") if isinstance(detail, dict) else None
                raise TranscriptionError(status, code if isinstance(code, str) else None)
            response.raise_for_status()
            result = await response.json()

        if not isinstance(result, dict):
            raise RuntimeError("Scribe returned an invalid transcription response")
        text = result.get("text")
        words = result.get("words")
        if not isinstance(text, str):
            if isinstance(words, list):
                text = " ".join(
                    word["text"].strip()
                    for word in words
                    if isinstance(word, dict)
                    and isinstance(word.get("text"), str)
                    and word["text"].strip()
                    and word.get("type", "word") == "word"
                )
            else:
                text = ""
        text = text.strip()
        if not text:
            return []

        timestamped_words = []
        if isinstance(words, list):
            for word in words:
                if (
                    not isinstance(word, dict)
                    or not isinstance(word.get("text"), str)
                    or not word["text"].strip()
                    or word.get("type", "word") != "word"
                ):
                    continue
                start = word.get("start")
                end = word.get("end")
                if (
                    isinstance(start, (int, float))
                    and not isinstance(start, bool)
                    and math.isfinite(start)
                    and isinstance(end, (int, float))
                    and not isinstance(end, bool)
                    and math.isfinite(end)
                    and 0 <= start <= end
                ):
                    timestamped_words.append({
                        "start": float(start),
                        "end": float(end),
                        "text": word["text"].strip(),
                    })
        if not timestamped_words:
            raise TranscriptionTimestampError(text)
        return timestamped_words
    raise RuntimeError("Scribe transcription retry loop ended unexpectedly")


async def run(room_name: str, raw_steps: list[str]) -> None:
    import aiohttp
    from livekit import api, rtc

    steps = [
        ScriptStep(*parse_step(step), language=parse_step_language(step))
        for step in raw_steps
    ]
    async with aiohttp.ClientSession() as http:
        audio = [
            (
                step,
                await _tts(http, step.line, language="es")
                if step.language == "es"
                else await _tts(http, step.line),
            )
            for step in steps
        ]
        token = (
            api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity("scripted-caller")
            .with_grants(api.VideoGrants(room_join=True, room=room_name))
            .to_jwt()
        )
        room = rtc.Room()
        answer_tracks: asyncio.Queue[Any] = asyncio.Queue()
        capture_end: asyncio.Event | None = None

        @room.on("track_subscribed")
        def _track_subscribed(track: Any, publication: Any, participant: Any) -> None:
            if is_agent_audio_track(
                participant.kind,
                track.kind,
                publication.source,
                rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
                rtc.TrackKind.KIND_AUDIO,
                rtc.TrackSource.SOURCE_MICROPHONE,
            ):
                answer_tracks.put_nowait(track)

        @room.on("track_unsubscribed")
        def _track_unsubscribed(track: Any, publication: Any, participant: Any) -> None:
            if capture_end is not None and is_agent_audio_track(
                participant.kind,
                track.kind,
                publication.source,
                rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
                rtc.TrackKind.KIND_AUDIO,
                rtc.TrackSource.SOURCE_MICROPHONE,
            ):
                capture_end.set()

        @room.on("disconnected")
        def _disconnected(*_args: Any) -> None:
            if capture_end is not None:
                capture_end.set()

        await room.connect(os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"), token)
        try:
            source = rtc.AudioSource(RATE, 1)
            track = rtc.LocalAudioTrack.create_audio_track("mic", source)
            readiness_deadline = time.monotonic() + READINESS_TIMEOUT_SECONDS
            publication = await room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )
            await wait_for_microphone_ready(publication, deadline=readiness_deadline)
            print("caller connected", flush=True)

            async def push(pcm: bytes) -> None:
                for offset in range(0, len(pcm), FRAME_SAMPLES * 2):
                    chunk = pcm[offset : offset + FRAME_SAMPLES * 2].ljust(FRAME_SAMPLES * 2, b"\0")
                    await source.capture_frame(rtc.AudioFrame(chunk, RATE, 1, FRAME_SAMPLES))

            silence = bytes(FRAME_SAMPLES * 2)

            async def quiet(seconds: float) -> None:
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    await source.capture_frame(rtc.AudioFrame(silence, RATE, 1, FRAME_SAMPLES))

            for step, speech in audio:
                pcm = speech.pcm if isinstance(speech, TTSResponse) else speech
                await quiet(step.delay)
                print(f"say: {step.line}", flush=True)
                capture_end = asyncio.Event()
                capture = ContinuousCapture(answer_tracks, capture_end)
                await capture.start()
                await push(pcm)
                speech_end = await _speech_end_after_playout(source)
                answer_pcm, sample_rate, channels, capture_started = await capture.result()
                try:
                    segments = await _transcribe(http, answer_pcm, sample_rate, channels)
                except TranscriptionError as error:
                    print(f"transcription failure: {error}", flush=True)
                    continue
                transcript = _segments_text(segments)
                latency = first_matching_latency(segments, step.answer_pattern, speech_end, capture_started)
                print(f"transcript: {transcript}", flush=True)
                print(f"segments: {_format_segments(segments)}", flush=True)
                if latency is None:
                    print("first matching answer word: no regex match", flush=True)
                else:
                    print(f"first matching answer word: {latency:.2f}s after speech end", flush=True)
        finally:
            await room.disconnect()


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: caller.py ROOM 'LINE@DELAY::ANSWER_REGEX' [...]")
    asyncio.run(run(sys.argv[1], sys.argv[2:]))


if __name__ == "__main__":
    main()
