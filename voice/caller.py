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


async def _capture_answer(track_queue: asyncio.Queue[Any]) -> tuple[bytes, int, int, float]:
    from livekit import rtc

    track = await asyncio.wait_for(track_queue.get(), timeout=15)
    capture_started: float | None = None
    audio_stream = rtc.AudioStream(track)
    frames: list[bytes] = []
    try:
        async for event in audio_stream:
            frame = event.frame
            if capture_started is None:
                capture_started = time.monotonic()
            raw = bytes(frame.data)
            frames.append(raw)
            elapsed = time.monotonic() - capture_started
            if elapsed >= MAX_ANSWER_SECONDS:
                break
    finally:
        await audio_stream.aclose()
        track_queue.put_nowait(track)
    if not frames or capture_started is None:
        raise RuntimeError("agent audio track produced no frames")
    return b"".join(frames), frame.sample_rate, frame.num_channels, capture_started


async def _tts(http: Any, text: str) -> bytes:
    async with http.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        json={"model": "gpt-4o-mini-tts", "voice": "ash", "input": text, "response_format": "pcm"},
    ) as response:
        response.raise_for_status()
        return await response.read()


async def _transcribe(http: Any, pcm: bytes, sample_rate: int, channels: int) -> list[dict[str, Any]]:
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

        await room.connect(os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880"), token)
        try:
            source = rtc.AudioSource(RATE, 1)
            track = rtc.LocalAudioTrack.create_audio_track("mic", source)
            await room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
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

            while not room.remote_participants:
                await quiet(0.2)
            for step, pcm in audio:
                await quiet(step.delay)
                print(f"say: {step.line}", flush=True)
                await push(pcm)
                speech_end = time.monotonic()
                answer_pcm, sample_rate, channels, capture_started = await _capture_answer(answer_tracks)
                segments = await _transcribe(http, answer_pcm, sample_rate, channels)
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
