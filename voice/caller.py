"""Scripted LiveKit caller: python caller.py ROOM 'question@delay::answer-regex' ..."""

import asyncio
import os
import re
import sys
import time
import wave
from dataclasses import dataclass
from typing import Any

RATE = 24000
FRAME_SAMPLES = RATE // 100
MAX_ANSWER_SECONDS = 30
LISTENING_TIMEOUT_SECONDS = 15
LISTENING_SETTLE_SECONDS = 0.3
LISTENING_POLL_INTERVAL_SECONDS = 0.05
AGENT_STATE_ATTRIBUTE = "lk.agent.state"


class WhisperTranscriptionError(RuntimeError):
    """Whisper rejected one captured clip with an HTTP 4xx response."""

    def __init__(self, status: int):
        if not 400 <= status < 500:
            raise ValueError("Whisper transcription failures must be HTTP 4xx responses")
        self.status = status
        super().__init__(f"Whisper transcription rejected clip (HTTP {status})")


async def wait_for_listening(
    publication: Any,
    participants: Any,
    agent_kind: Any,
    *,
    deadline: float,
    monotonic: Any = time.monotonic,
    sleep: Any = asyncio.sleep,
) -> None:
    """Wait for microphone subscription and agent listening state under one deadline."""
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("caller microphone subscription exceeded its deadline")
    try:
        await asyncio.wait_for(publication.wait_for_subscription(), timeout=remaining)
    except TimeoutError as error:
        raise TimeoutError("caller microphone subscription exceeded its deadline") from error

    while not any(
        participant.kind == agent_kind
        and participant.attributes.get(AGENT_STATE_ATTRIBUTE) == "listening"
        for participant in participants()
    ):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("agent listening state did not arrive before deadline")
        await sleep(min(LISTENING_POLL_INTERVAL_SECONDS, remaining))

    remaining = deadline - monotonic()
    if remaining < LISTENING_SETTLE_SECONDS:
        raise TimeoutError("agent listening settle exceeded its deadline")
    try:
        await asyncio.wait_for(sleep(LISTENING_SETTLE_SECONDS), timeout=remaining)
    except TimeoutError as error:
        raise TimeoutError("agent listening settle exceeded its deadline") from error
    if monotonic() > deadline:
        raise TimeoutError("agent listening settle exceeded its deadline")


@dataclass(frozen=True)
class ScriptStep:
    delay: float
    line: str
    answer_pattern: str


def parse_step(value: str) -> tuple[float, str, str]:
    """Parse LINE@DELAY::ANSWER_REGEX, keeping regex syntax literal."""
    if "::" not in value:
        raise ValueError("script step must be LINE@DELAY::ANSWER_REGEX")
    speech, pattern = value.rsplit("::", 1)
    line, separator, delay_text = speech.rpartition("@")
    if not separator or not line or not pattern:
        raise ValueError("script step must be LINE@DELAY::ANSWER_REGEX")
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
        no_frame_deadline = loop.time() + MAX_ANSWER_SECONDS
        while True:
            remaining = (
                max(0.0, no_frame_deadline - loop.time())
                if capture_started is None
                else max(0.0, capture_deadline - loop.time())
            )
            try:
                if ended is None:
                    event = await asyncio.wait_for(audio_stream.__anext__(), timeout=remaining)
                else:
                    next_task = asyncio.create_task(audio_stream.__anext__())
                    ended_task = asyncio.create_task(ended.wait())
                    done, pending = await asyncio.wait(
                        (next_task, ended_task),
                        timeout=remaining,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    if next_task not in done:
                        break
                    event = next_task.result()
            except (TimeoutError, StopAsyncIteration):
                break
            last_frame = event.frame
            if capture_started is None:
                capture_started = time.monotonic()
                capture_deadline = loop.time() + MAX_ANSWER_SECONDS
            frames.append(bytes(last_frame.data))
    finally:
        await audio_stream.aclose()
        if ended is None or not ended.is_set():
            track_queue.put_nowait(track)
    if not frames or capture_started is None or last_frame is None:
        raise RuntimeError("agent audio track produced no frames")
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


async def _tts(http: Any, text: str) -> bytes:
    async with http.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": "gpt-4o-mini-tts", "voice": "ash", "input": text, "response_format": "pcm"},
    ) as response:
        response.raise_for_status()
        return await response.read()


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

    steps = [ScriptStep(*parse_step(step)) for step in raw_steps]
    async with aiohttp.ClientSession() as http:
        audio = [(step, await _tts(http, step.line)) for step in steps]
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
            readiness_deadline = time.monotonic() + LISTENING_TIMEOUT_SECONDS
            publication = await room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )
            await wait_for_listening(
                publication,
                lambda: room.remote_participants.values(),
                rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
                deadline=readiness_deadline,
            )
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

            for step, pcm in audio:
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
