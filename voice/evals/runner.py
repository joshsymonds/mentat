"""Remote scripted caller capture for isolated voice evaluations."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import caller

RATE = caller.RATE
FRAME_SAMPLES = caller.FRAME_SAMPLES
PARTICIPANT_DEADLINE_SECONDS = 30.0
ANSWER_CAPTURE_DEADLINE_SECONDS = 45.0
ROOM_DELETE_DEADLINE_SECONDS = 60.0
REMOTE_OPERATION_DEADLINE_SECONDS = 30.0
ROOM_POLL_INTERVAL_SECONDS = 0.25


@dataclass(frozen=True)
class CaptureDependencies:
    """Live services and caller primitives, injectable for offline tests."""

    api: Any
    rtc: Any
    http: Any
    tts: Callable[..., Any]
    transcribe: Callable[..., Any]
    capture_factory: Callable[..., Any]
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Any] = asyncio.sleep


def _dependencies(http: Any) -> CaptureDependencies:
    from livekit import api, rtc

    return CaptureDependencies(
        api=api,
        rtc=rtc,
        http=http,
        tts=caller._tts,
        transcribe=caller._transcribe,
        capture_factory=caller.ContinuousCapture,
    )


async def _with_deadline(awaitable: Any, timeout: float, label: str) -> Any:
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError as error:
        raise RuntimeError(f"{label} exceeded its deadline") from error


def _finite_timestamp(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} timestamp is missing or invalid")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} timestamp is missing or invalid")
    return result


def _segment_start(segments: list[dict[str, Any]]) -> float:
    starts = []
    for segment in segments:
        value = segment.get("start")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError("transcription segment has no valid start timestamp")
        start = float(value)
        if not math.isfinite(start) or start < 0:
            raise RuntimeError("transcription segment has no valid start timestamp")
        starts.append(start)
    if not starts:
        raise RuntimeError("transcription returned no segments")
    return min(starts)


def _trace_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()


async def _room_exists(
    api_client: Any, api: Any, room_name: str, *, timeout: float = REMOTE_OPERATION_DEADLINE_SECONDS
) -> bool:
    response = await _with_deadline(
        api_client.room.list_rooms(api.ListRoomsRequest(names=[room_name])),
        timeout,
        "LiveKit room listing",
    )
    return any(room.name == room_name for room in response.rooms)


async def _wait_for_room_deletion(
    api_client: Any,
    api: Any,
    room_name: str,
    *,
    deadline: float,
    poll_interval: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], Any],
) -> float:
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError("room deletion was not observed before deadline")
        exists = await _room_exists(
            api_client,
            api,
            room_name,
            timeout=min(REMOTE_OPERATION_DEADLINE_SECONDS, remaining),
        )
        if not exists:
            break
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError("room deletion was not observed before deadline")
        try:
            await asyncio.wait_for(
                sleep(min(poll_interval, remaining)),
                timeout=min(REMOTE_OPERATION_DEADLINE_SECONDS, remaining),
            )
        except TimeoutError as error:
            raise RuntimeError("room deletion was not observed before deadline") from error
    return _finite_timestamp(monotonic(), "room deletion")


async def capture_script(
    room_name: str,
    raw_steps: list[str],
    *,
    dependencies: CaptureDependencies,
    room_delete_deadline: float = ROOM_DELETE_DEADLINE_SECONDS,
    poll_interval: float = ROOM_POLL_INTERVAL_SECONDS,
    room_close_after: int | None = None,
) -> list[dict[str, Any]]:
    """Capture every script turn and enforce its expected room-close policy."""
    if not raw_steps:
        raise ValueError("at least one scripted line is required")
    if not math.isfinite(room_delete_deadline) or room_delete_deadline <= 0:
        raise ValueError("room deletion deadline must be finite and positive")
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise ValueError("room poll interval must be finite and positive")
    if room_close_after is not None and (
        isinstance(room_close_after, bool)
        or not isinstance(room_close_after, int)
        or not 1 <= room_close_after <= len(raw_steps)
    ):
        raise ValueError("room_close_after must name a scripted turn or be None")
    steps = [caller.ScriptStep(*caller.parse_step(value)) for value in raw_steps]

    api_key = os.environ["LIVEKIT_API_KEY"]
    api_secret = os.environ["LIVEKIT_API_SECRET"]
    token = (
        dependencies.api.AccessToken(api_key, api_secret)
        .with_identity("scripted-caller")
        .with_grants(dependencies.api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )
    room = dependencies.rtc.Room()
    answer_tracks: asyncio.Queue[Any] = asyncio.Queue()

    @room.on("track_subscribed")
    def track_subscribed(track: Any, publication: Any, participant: Any) -> None:
        if caller.is_agent_audio_track(
            participant.kind,
            track.kind,
            publication.source,
            dependencies.rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
            dependencies.rtc.TrackKind.KIND_AUDIO,
            dependencies.rtc.TrackSource.SOURCE_MICROPHONE,
        ):
            answer_tracks.put_nowait(track)

    livekit_url = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
    api_client = dependencies.api.LiveKitAPI(livekit_url, api_key, api_secret)
    traces: list[dict[str, Any]] = []
    deletion_task: asyncio.Task[float] | None = None
    try:
        await _with_deadline(
            room.connect(livekit_url, token),
            REMOTE_OPERATION_DEADLINE_SECONDS,
            "LiveKit room connection",
        )
        source = dependencies.rtc.AudioSource(RATE, 1)
        local_track = dependencies.rtc.LocalAudioTrack.create_audio_track("mic", source)
        await _with_deadline(
            room.local_participant.publish_track(
                local_track,
                dependencies.rtc.TrackPublishOptions(
                    source=dependencies.rtc.TrackSource.SOURCE_MICROPHONE
                ),
            ),
            REMOTE_OPERATION_DEADLINE_SECONDS,
            "LiveKit microphone publication",
        )

        participant_deadline = dependencies.monotonic() + PARTICIPANT_DEADLINE_SECONDS
        while not room.remote_participants:
            if dependencies.monotonic() >= participant_deadline:
                raise RuntimeError("voice worker did not join the selected room before deadline")
            await _with_deadline(
                source.capture_frame(
                    dependencies.rtc.AudioFrame(bytes(FRAME_SAMPLES * 2), RATE, 1, FRAME_SAMPLES)
                ),
                REMOTE_OPERATION_DEADLINE_SECONDS,
                "LiveKit participant wait audio",
            )
            await _with_deadline(
                dependencies.sleep(0.2),
                REMOTE_OPERATION_DEADLINE_SECONDS,
                "voice worker join polling",
            )

        async def push(pcm: bytes) -> None:
            for offset in range(0, len(pcm), FRAME_SAMPLES * 2):
                chunk = pcm[offset : offset + FRAME_SAMPLES * 2].ljust(FRAME_SAMPLES * 2, b"\0")
                await _with_deadline(
                    source.capture_frame(
                        dependencies.rtc.AudioFrame(chunk, RATE, 1, FRAME_SAMPLES)
                    ),
                    REMOTE_OPERATION_DEADLINE_SECONDS,
                    "LiveKit speech audio playout",
                )

        silence = bytes(FRAME_SAMPLES * 2)

        async def quiet(seconds: float) -> None:
            deadline = dependencies.monotonic() + seconds
            while dependencies.monotonic() < deadline:
                await _with_deadline(
                    source.capture_frame(
                        dependencies.rtc.AudioFrame(silence, RATE, 1, FRAME_SAMPLES)
                    ),
                    REMOTE_OPERATION_DEADLINE_SECONDS,
                    "LiveKit silence playout",
                )
                await _with_deadline(
                    dependencies.sleep(min(0.01, max(0.0, deadline - dependencies.monotonic()))),
                    REMOTE_OPERATION_DEADLINE_SECONDS,
                    "silence timing",
                )

        for index, step in enumerate(steps):
            await quiet(step.delay)
            pcm = await _with_deadline(
                dependencies.tts(dependencies.http, step.line),
                REMOTE_OPERATION_DEADLINE_SECONDS,
                "scripted speech synthesis",
            )
            capture = dependencies.capture_factory(answer_tracks)
            await _with_deadline(
                capture.start(),
                REMOTE_OPERATION_DEADLINE_SECONDS,
                "continuous answer capture start",
            )
            await push(pcm)
            speech_end = _finite_timestamp(
                await _with_deadline(
                    caller._speech_end_after_playout(source),
                    REMOTE_OPERATION_DEADLINE_SECONDS,
                    "speech playout completion",
                ),
                "speech_end",
            )
            if room_close_after == index + 1:
                deletion_task = asyncio.create_task(
                    _wait_for_room_deletion(
                        api_client,
                        dependencies.api,
                        room_name,
                        deadline=speech_end + room_delete_deadline,
                        poll_interval=poll_interval,
                        monotonic=dependencies.monotonic,
                        sleep=dependencies.sleep,
                    )
                )
            try:
                answer_pcm, sample_rate, channels, capture_started = await asyncio.wait_for(
                    capture.result(), timeout=ANSWER_CAPTURE_DEADLINE_SECONDS
                )
            except TimeoutError as error:
                raise RuntimeError("agent audio capture exceeded its deadline") from error
            if not answer_pcm:
                raise RuntimeError("agent audio capture returned no frames")
            capture_started = _finite_timestamp(capture_started, "capture start")
            segments = await _with_deadline(
                dependencies.transcribe(dependencies.http, answer_pcm, sample_rate, channels),
                REMOTE_OPERATION_DEADLINE_SECONDS,
                "answer transcription",
            )
            transcript = _trace_text(segments)
            if not transcript:
                raise RuntimeError("transcription returned an empty transcript")
            if not re.search(step.answer_pattern, transcript):
                raise RuntimeError(f"transcript did not match expected answer: {transcript}")
            first_audio = capture_started + _segment_start(segments)
            if first_audio < speech_end:
                raise RuntimeError("first audio timestamp precedes speech end")
            trace = {
                "turn": index + 1,
                "room": room_name,
                "line": step.line,
                "transcript": transcript,
                "speech_end": speech_end,
                "first_audio": first_audio,
                "room_deleted": None,
            }
            traces.append(trace)

            if room_close_after == index + 1:
                if deletion_task is None:
                    raise RuntimeError("room deletion observer was not started")
                trace["room_deleted"] = await deletion_task
                if trace["room_deleted"] < speech_end:
                    raise RuntimeError("room deletion timestamp precedes speech end")
                if index < len(steps) - 1:
                    raise RuntimeError("room was deleted before all scripted lines were captured")
            elif not await _room_exists(api_client, dependencies.api, room_name):
                raise RuntimeError("room was deleted before all scripted lines were captured")
        return traces
    finally:
        if deletion_task is not None:
            if not deletion_task.done():
                deletion_task.cancel()
                try:
                    await deletion_task
                except asyncio.CancelledError:
                    pass
            elif not deletion_task.cancelled():
                deletion_task.exception()
        try:
            await _with_deadline(
                room.disconnect(),
                REMOTE_OPERATION_DEADLINE_SECONDS,
                "LiveKit room disconnect",
            )
        finally:
            close = getattr(api_client, "aclose", None)
            if close is not None:
                await _with_deadline(
                    close(),
                    REMOTE_OPERATION_DEADLINE_SECONDS,
                    "LiveKit API client close",
                )


async def run_remote_capture(
    room_name: str, raw_steps: list[str], *, room_close_after: int | None = None
) -> list[dict[str, Any]]:
    import aiohttp

    async with aiohttp.ClientSession() as http:
        return await capture_script(
            room_name,
            raw_steps,
            dependencies=_dependencies(http),
            room_close_after=room_close_after,
        )


def _parse_arguments(argv: list[str]) -> tuple[str, list[str], int | None]:
    import argparse

    parser = argparse.ArgumentParser(
        description="Capture scripted caller turns from a remote voice room"
    )
    parser.add_argument("room")
    parser.add_argument(
        "--room-close-after",
        default="last",
        metavar="TURN|none|last",
        help="require room deletion after TURN, or require it to stay open with none (default: last)",
    )
    parser.add_argument("steps", nargs="+", metavar="LINE@DELAY::ANSWER_REGEX")
    arguments = parser.parse_args(argv)
    if arguments.room_close_after == "last":
        room_close_after = len(arguments.steps)
    elif arguments.room_close_after == "none":
        room_close_after = None
    else:
        try:
            room_close_after = int(arguments.room_close_after)
        except ValueError:
            parser.error("--room-close-after must be a turn number, 'none', or 'last'")
    return arguments.room, arguments.steps, room_close_after


def main(argv: list[str] | None = None) -> int:
    room, steps, room_close_after = _parse_arguments(sys.argv[1:] if argv is None else argv)
    traces = asyncio.run(
        run_remote_capture(room, steps, room_close_after=room_close_after)
    )
    print(json.dumps({"turns": traces}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
