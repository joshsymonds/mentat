"""Scripted LiveKit caller: python caller.py ROOM 'question@delay::answer-regex' ..."""

import asyncio
import math
import os
import re
import struct
import sys
import time
import wave
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


@dataclass(frozen=True)
class TTSResponse:
    pcm: bytes
    http_status: int | None
    response_bytes: int


class WhisperTranscriptionError(RuntimeError):
    """Whisper rejected one captured clip with an HTTP 4xx response."""

    def __init__(self, status: int):
        if not 400 <= status < 500:
            raise ValueError("Whisper transcription failures must be HTTP 4xx responses")
        self.status = status
        super().__init__(f"Whisper transcription rejected clip (HTTP {status})")


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
    """Return seconds from caller speech end to the first matching segment start."""
    try:
        matcher = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid answer regex: {exc}") from exc
    for segment in segments:
        if matcher.search(str(segment.get("text", ""))):
            return capture_started + float(segment["start"]) - speech_end
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


async def _capture_answer(
    track_queue: asyncio.Queue[Any], ended: asyncio.Event | None = None
) -> tuple[bytes, int, int, float]:
    from livekit import rtc

    if ended is None:
        track = await asyncio.wait_for(track_queue.get(), timeout=15)
    else:
        track_task = asyncio.create_task(track_queue.get())
        ended_task = asyncio.create_task(ended.wait())
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
                if ended is None:
                    event = await asyncio.wait_for(audio_stream.__anext__(), timeout=remaining)
                else:
                    next_task = asyncio.create_task(audio_stream.__anext__())
                    if ended.is_set():
                        try:
                            event = await asyncio.wait_for(next_task, timeout=min(remaining, END_DRAIN_SECONDS))
                        except TimeoutError:
                            break
                    else:
                        ended_task = asyncio.create_task(ended.wait())
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
        self, track_queue: asyncio.Queue[Any], ended: asyncio.Event | None = None
    ) -> None:
        self._track_queue = track_queue
        self._ended = ended
        self._task: asyncio.Task[tuple[bytes, int, int, float]] | None = None

    async def start(self) -> None:
        if self._task is not None:
            raise RuntimeError("continuous capture already started")
        self._task = asyncio.create_task(_capture_answer(self._track_queue, self._ended))
        await asyncio.sleep(0)

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

    from aiohttp import FormData

    import io

    wav_bytes = io.BytesIO()
    with wave.open(wav_bytes, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    form = FormData()
    form.add_field("file", wav_bytes.getvalue(), filename="answer.wav", content_type="audio/wav")
    form.add_field("model", "whisper-1")
    form.add_field("response_format", "verbose_json")
    form.add_field("timestamp_granularities[]", "segment")
    async with http.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        data=form,
    ) as response:
        status = getattr(response, "status", None)
        if isinstance(status, int) and 400 <= status < 500:
            raise WhisperTranscriptionError(status)
        response.raise_for_status()
        result = await response.json()
    return result.get("segments", [])


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
                except WhisperTranscriptionError as error:
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
