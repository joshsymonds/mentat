"""Remote scripted caller capture for isolated voice evaluations."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import caller
from evals.dev_stack import DevStack, _redact_diagnostics
from evals.report import NO_ANSWER_FAILURE, score_observations
from evals.scenarios import SCENARIOS

RATE = caller.RATE
FRAME_SAMPLES = caller.FRAME_SAMPLES
PARTICIPANT_DEADLINE_SECONDS = 30.0
ANSWER_CAPTURE_DEADLINE_SECONDS = 45.0
ROOM_DELETE_DEADLINE_SECONDS = 60.0
REMOTE_OPERATION_DEADLINE_SECONDS = 30.0
ROOM_POLL_INTERVAL_SECONDS = 0.25
TOKEN_REQUEST_DEADLINE_SECONDS = 10.0
FAKE_PHONE_LOG = "evals/phone.jsonl"
SMS_CONFIRMATION_PATTERN = re.compile(
    r"\b(?:should i|would you like|do you want|shall i|want me to|say yes)\b",
    re.IGNORECASE,
)
PHONE_TOOL_KINDS = {
    "send_sms": "sms",
    "open_on_phone": "open",
    "list_conversations": "conversations",
    "read_conversation": "messages",
    "search_messages": "search",
    "find_places": "location",
    "navigate_to": "navigate",
    "dial": "dial",
    "set_alarm": "alarm",
    "set_timer": "timer",
    "get_location": "location",
    "get_current_location": "location",
}


class DeadlineExceeded(RuntimeError):
    """A bounded capture operation did not complete before its deadline."""


class PartialCaptureFailure(RuntimeError):
    """A scripted turn failed after earlier turns were captured completely."""

    def __init__(
        self,
        turns: list[dict[str, Any]],
        turn: int,
        message: str,
        speech_started_at: float | None = None,
    ):
        super().__init__(message)
        self.turns = turns
        self.failure = {"turn": turn, "message": message}
        if speech_started_at is not None:
            self.failure["speech_started_at"] = speech_started_at


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
        raise DeadlineExceeded(f"{label} exceeded its deadline") from error


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
        if not isinstance(segment, dict):
            raise RuntimeError("transcription segment is not an object")
        start = _finite_timestamp(segment.get("start"), "transcription segment start")
        end = _finite_timestamp(segment.get("end"), "transcription segment end")
        if start < 0 or end < start:
            raise RuntimeError("transcription segment has invalid timestamp bounds")
        if not isinstance(segment.get("text"), str):
            raise RuntimeError("transcription segment has no text")
        starts.append(start)
    if not starts:
        raise RuntimeError("transcription returned no segments")
    return min(starts)


PCM_WINDOW_SECONDS = 0.020
PCM_RMS_THRESHOLD = 200


def _first_audio_after(
    answer_pcm: bytes,
    sample_rate: int,
    channels: int,
    capture_started: float,
    speech_end: float,
) -> tuple[float, bool]:
    """Find sustained agent speech from signed 16-bit little-endian PCM."""
    capture_started = _finite_timestamp(capture_started, "capture start")
    speech_end = _finite_timestamp(speech_end, "speech_end")
    if (
        not isinstance(answer_pcm, bytes)
        or isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate < 50
        or sample_rate % 50 != 0
        or isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels < 1
    ):
        raise RuntimeError(NO_ANSWER_FAILURE)

    window_frames = sample_rate // 50
    window_values = window_frames * channels
    window_bytes = window_values * 2
    sample_frame_bytes = channels * 2
    if (
        window_frames < 1
        or not answer_pcm
        or len(answer_pcm) % sample_frame_bytes != 0
    ):
        raise RuntimeError(NO_ANSWER_FAILURE)

    rms_values = []
    window_count = len(answer_pcm) // window_bytes
    for window_index in range(window_count):
        offset = window_index * window_bytes
        window = answer_pcm[offset : offset + window_bytes]
        values = struct.unpack(f"<{window_values}h", window)
        rms_values.append(math.sqrt(sum(value * value for value in values) / len(values)))

    first_audio = None
    overlap = False
    run_start = None
    for index, rms in enumerate(rms_values):
        if rms >= PCM_RMS_THRESHOLD:
            if run_start is None:
                run_start = index
            continue
        if run_start is not None and index - run_start >= 2:
            run_end = index
            run_start_time = capture_started + run_start * PCM_WINDOW_SECONDS
            overlap = overlap or run_start_time < speech_end
            for qualifying_index in range(run_start, run_end):
                window_start = capture_started + qualifying_index * PCM_WINDOW_SECONDS
                if window_start >= speech_end:
                    first_audio = window_start if first_audio is None else min(first_audio, window_start)
            run_start = None
        else:
            run_start = None
    if run_start is not None and len(rms_values) - run_start >= 2:
        run_end = len(rms_values)
        run_start_time = capture_started + run_start * PCM_WINDOW_SECONDS
        overlap = overlap or run_start_time < speech_end
        for qualifying_index in range(run_start, run_end):
            window_start = capture_started + qualifying_index * PCM_WINDOW_SECONDS
            if window_start >= speech_end:
                first_audio = window_start if first_audio is None else min(first_audio, window_start)

    if first_audio is None:
        raise RuntimeError(NO_ANSWER_FAILURE)
    return first_audio, overlap


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
    token = os.environ.get("MENTAT_VOICE_TOKEN", "")
    token_parts = token.split(".")
    if len(token_parts) != 3 or any(not part for part in token_parts):
        raise RuntimeError("endpoint-issued voice token is missing or invalid")
    livekit_url = os.environ.get("LIVEKIT_URL", "")
    parsed_livekit_url = urlsplit(livekit_url)
    if parsed_livekit_url.scheme not in ("ws", "wss") or parsed_livekit_url.hostname is None:
        raise RuntimeError("endpoint-issued LiveKit URL is missing or invalid")
    room = dependencies.rtc.Room()
    answer_tracks: asyncio.Queue[Any] = asyncio.Queue()
    capture_end: asyncio.Event | None = None

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

    @room.on("track_unsubscribed")
    def track_unsubscribed(track: Any, publication: Any, participant: Any) -> None:
        if capture_end is not None and caller.is_agent_audio_track(
            participant.kind,
            track.kind,
            publication.source,
            dependencies.rtc.ParticipantKind.PARTICIPANT_KIND_AGENT,
            dependencies.rtc.TrackKind.KIND_AUDIO,
            dependencies.rtc.TrackSource.SOURCE_MICROPHONE,
        ):
            capture_end.set()

    @room.on("disconnected")
    def disconnected(*_args: Any) -> None:
        if capture_end is not None:
            capture_end.set()

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
            try:
                pcm = await _with_deadline(
                    dependencies.tts(dependencies.http, step.line),
                    REMOTE_OPERATION_DEADLINE_SECONDS,
                    "scripted speech synthesis",
                )
            except DeadlineExceeded as error:
                if str(error) == "scripted speech synthesis exceeded its deadline":
                    raise PartialCaptureFailure(traces, index + 1, str(error)) from error
                raise
            capture_end = asyncio.Event()
            capture = dependencies.capture_factory(answer_tracks, capture_end)
            await _with_deadline(
                capture.start(),
                REMOTE_OPERATION_DEADLINE_SECONDS,
                "continuous answer capture start",
            )
            speech_started_at = time.time()
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
            capture_started = _finite_timestamp(capture_started, "capture start")
            try:
                first_audio, overlap = _first_audio_after(
                    answer_pcm, sample_rate, channels, capture_started, speech_end
                )
            except RuntimeError as error:
                if str(error) == NO_ANSWER_FAILURE:
                    raise PartialCaptureFailure(
                        traces,
                        index + 1,
                        str(error),
                        speech_started_at=speech_started_at,
                    ) from error
                raise
            segments = await _with_deadline(
                dependencies.transcribe(dependencies.http, answer_pcm, sample_rate, channels),
                REMOTE_OPERATION_DEADLINE_SECONDS,
                "answer transcription",
            )
            transcript = _trace_text(segments)
            if not transcript:
                raise RuntimeError("transcription returned an empty transcript")
            trace = {
                "turn": index + 1,
                "room": room_name,
                "line": step.line,
                "transcript": transcript,
                "speech_started_at": speech_started_at,
                "speech_end": speech_end,
                "capture_started": capture_started,
                "first_audio": first_audio,
                "overlap": overlap,
                "segments": segments,
                "room_deleted": None,
            }
            traces.append(trace)

            if room_close_after == index + 1:
                if deletion_task is None:
                    raise RuntimeError("room deletion observer was not started")
                try:
                    trace["room_deleted"] = await deletion_task
                except RuntimeError as error:
                    if str(error) == "room deletion was not observed before deadline":
                        raise PartialCaptureFailure(
                            traces,
                            index + 1,
                            str(error),
                        ) from error
                    raise
                if trace["room_deleted"] < speech_end:
                    raise RuntimeError("room deletion timestamp precedes speech end")
                if index < len(steps) - 1:
                    raise PartialCaptureFailure(
                        traces,
                        index + 2,
                        "room was deleted before all scripted lines were captured",
                    )
            elif not await _room_exists(api_client, dependencies.api, room_name):
                if index < len(steps) - 1:
                    raise PartialCaptureFailure(
                        traces,
                        index + 2,
                        "room was deleted before all scripted lines were captured",
                    )
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


def _json_lines(text: str, label: str) -> list[dict[str, Any]]:
    if not isinstance(text, str):
        raise RuntimeError(f"{label} log is missing or invalid")
    if not text.strip():
        if label == "fake phone":
            return []
        raise RuntimeError(f"{label} log is missing or empty")
    records = []
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"{label} log contains an empty line at {index}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"{label} log contains malformed JSON at line {index}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"{label} log line {index} is not an object")
        records.append(value)
    return records


def _eval_delegations(marker_log: str, room_name: str) -> list[dict[str, Any]]:
    records = _json_lines(marker_log, "delegation marker")
    delegations = []
    seen_ids = set()
    for line_number, marker in enumerate(records, 1):
        if set(marker) != {"room", "id", "created_at"}:
            raise RuntimeError(f"delegation marker line {line_number} has an invalid shape")
        marker_room = marker.get("room")
        if not isinstance(marker_room, str) or not marker_room:
            raise RuntimeError(f"delegation marker line {line_number} has an invalid room")
        delegation_id = marker.get("id")
        if not isinstance(delegation_id, str) or not delegation_id:
            raise RuntimeError(f"delegation marker line {line_number} has an invalid id")
        created_at = _finite_timestamp(marker.get("created_at"), "delegation creation")
        if marker_room != room_name:
            continue
        if delegation_id in seen_ids:
            raise RuntimeError("delegation marker file has a missing or repeated delegation id")
        if delegations and created_at <= delegations[-1]["created_at"]:
            raise RuntimeError("delegation marker timestamps are ambiguous or out of order")
        seen_ids.add(delegation_id)
        delegations.append({"id": delegation_id, "created_at": created_at})
    if not delegations:
        raise RuntimeError("delegation marker file has no records for the current room")
    return delegations


def _attribute_model_calls(
    traces: list[dict[str, Any]],
    marker_log: str,
    room_name: str,
    recorded_turns: list[dict[str, Any]],
    *,
    partial_capture: bool = False,
    failure_started_at: float | None = None,
    failure_turn: int | None = None,
) -> list[dict[str, Any]]:
    all_delegations = _eval_delegations(marker_log, room_name)
    starts = []
    for index, trace in enumerate(traces, 1):
        if not isinstance(trace, dict) or trace.get("turn") != index:
            raise RuntimeError(f"captured turn {index} has invalid turn identity")
        started_at = _finite_timestamp(trace.get("speech_started_at"), "caller speech start")
        if starts and started_at <= starts[-1]:
            raise RuntimeError("captured caller speech start timestamps are ambiguous or out of order")
        starts.append(started_at)
        trace["model_calls"] = []
    if failure_started_at is not None:
        failure_started_at = _finite_timestamp(
            failure_started_at, "failed caller speech start"
        )
        if starts and failure_started_at <= starts[-1]:
            raise RuntimeError("partial failure speech start is ambiguous or out of order")

    start_boundary = starts[0] if starts else failure_started_at
    delegations = [
        marker
        for marker in all_delegations
        if start_boundary is not None and marker["created_at"] >= start_boundary
    ]
    if len(delegations) != len(recorded_turns):
        raise RuntimeError("delegation and SDK result counts differ")

    for delegation, recorded in zip(delegations, recorded_turns, strict=True):
        created_at = delegation["created_at"]
        if created_at in starts:
            raise RuntimeError("ambiguous delegation timestamp coincides with caller speech start")
        model_calls = recorded.get("model_calls") if isinstance(recorded, dict) else None
        if not isinstance(model_calls, list):
            raise RuntimeError("recorded SDK result has invalid model-call evidence")
        if not starts:
            if partial_capture:
                continue
            raise RuntimeError("no captured turn precedes delegation")
        if partial_capture:
            turn_index = next(
                (
                    index
                    for index in range(len(starts) - 1, -1, -1)
                    if starts[index] < created_at
                    and (
                        (index + 1 < len(starts) and created_at < starts[index + 1])
                        or (index + 1 == len(starts) and (
                            failure_started_at is None or created_at < failure_started_at
                        ))
                    )
                ),
                None,
            )
            if turn_index is None:
                if failure_started_at is not None and created_at >= failure_started_at:
                    phone_tools = recorded.get("phone_tools", [])
                    if phone_tools:
                        if (
                            isinstance(failure_turn, bool)
                            or not isinstance(failure_turn, int)
                            or failure_turn != len(starts) + 1
                        ):
                            raise RuntimeError("failed-turn phone tools have no valid turn attribution")
                        for tool in phone_tools:
                            if not isinstance(tool, dict):
                                raise RuntimeError("recorded SDK result has invalid phone-tool evidence")
                            tool["turn"] = failure_turn
                    continue
                raise RuntimeError("no captured turn precedes delegation")
        else:
            turn_index = next(
                (index for index in range(len(starts) - 1, -1, -1) if starts[index] < created_at),
                None,
            )
            if turn_index is None:
                raise RuntimeError("no captured turn precedes delegation")
        traces[turn_index]["model_calls"].extend(model_calls)
        for tool in recorded.get("phone_tools", []):
            tool["turn"] = turn_index + 1
    return traces


def _recorded_turns(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        current.append(message)
        if message.get("type") != "result":
            continue
        calls = []
        tools = []
        for raw in current:
            if raw.get("type") == "stream_event":
                event = raw.get("event")
                if isinstance(event, dict) and event.get("type") == "message_start":
                    started = event.get("message")
                    if not isinstance(started, dict):
                        raise RuntimeError("recorded message_start has no message object")
                    model = started.get("model")
                    message_id = started.get("id")
                    if not isinstance(model, str) or not model or not isinstance(message_id, str) or not message_id:
                        raise RuntimeError("recorded message_start has no finite model-call evidence")
                    calls.append({"id": message_id, "model": model})
            if raw.get("type") == "assistant":
                assistant_message = raw.get("message")
                blocks = assistant_message.get("content") if isinstance(assistant_message, dict) else None
                if not isinstance(blocks, list):
                    raise RuntimeError("recorded assistant message has no content blocks")
                for block in blocks:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    name = block.get("name")
                    arguments = block.get("input")
                    if not isinstance(name, str) or not isinstance(arguments, dict):
                        raise RuntimeError("recorded tool use has invalid name or arguments")
                    tool_kind = PHONE_TOOL_KINDS.get(name.rsplit("__", 1)[-1])
                    if tool_kind is not None:
                        tools.append({"turn": len(turns) + 1, "name": name, "kind": tool_kind, "input": arguments})
        if not calls:
            raise RuntimeError("recorded assistant result has no message_start model-call evidence")
        turns.append({"model_calls": calls, "phone_tools": tools})
        current = []
    if any(
        item.get("type") in ("assistant", "user")
        or (item.get("type") == "stream_event" and isinstance(item.get("event"), dict)
            and item["event"].get("type") == "message_start")
        for item in current
    ):
        raise RuntimeError("recorded SDK turn is missing its top-level result boundary")
    return turns


def _phone_commands(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}
    for entry in entries:
        event = entry.get("event")
        payload = entry.get("command") if event == "command" else entry.get("result")
        if event not in ("command", "result") or not isinstance(payload, dict):
            raise RuntimeError("fake phone log contains an invalid event")
        command_id = payload.get("id")
        if not isinstance(command_id, str) or not command_id:
            raise RuntimeError("fake phone log event has no command id")
        if event == "command":
            if command_id in results or any(command.get("id") == command_id for command in commands):
                raise RuntimeError("fake phone log repeats a command id")
            if not isinstance(payload.get("kind"), str):
                raise RuntimeError("fake phone command has no kind")
            commands.append(dict(payload))
        else:
            if command_id in results:
                raise RuntimeError("fake phone log repeats a command result")
            results[command_id] = dict(payload)
    command_ids = {command["id"] for command in commands}
    if command_ids != set(results):
        raise RuntimeError("fake phone log has a missing or unmatched command result")
    if any(result.get("status") != "ok" for result in results.values()):
        raise RuntimeError("fake phone reported a failed command")
    return commands


def _match_phone_tools(
    commands: list[dict[str, Any]], recorded_turns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    tools = [tool for turn in recorded_turns for tool in turn["phone_tools"]]
    if len(commands) != len(tools):
        raise RuntimeError(
            f"fake phone observed {len(commands)} commands but SDK recording has {len(tools)} phone tool uses"
        )
    matched = []
    for index, (command, tool) in enumerate(zip(commands, tools, strict=True), 1):
        if command["kind"] != tool["kind"]:
            raise RuntimeError(f"fake phone command {index} does not match its recorded phone tool")
        arguments = tool["input"]
        for key, value in command.items():
            if key in ("id", "kind", "expires_at"):
                continue
            if arguments.get(key) != value:
                raise RuntimeError(f"fake phone command {index} does not match its recorded phone tool arguments")
        matched.append({**command, "turn": tool["turn"]})
    return matched


def _scenario_steps(scenario: Any) -> list[str]:
    caller_lines = getattr(scenario, "caller_lines", None)
    expectations = getattr(scenario, "turns", None)
    if not isinstance(caller_lines, tuple) or not isinstance(expectations, tuple) or len(caller_lines) != len(expectations):
        raise RuntimeError("scenario scripts and turn expectations are missing or mismatched")
    raw_steps = []
    for line, expectation in zip(caller_lines, expectations, strict=True):
        patterns = getattr(expectation, "answer_patterns", None)
        if not isinstance(line, str) or not line or not isinstance(patterns, tuple) or not patterns:
            raise RuntimeError("scenario has a missing caller line or answer pattern")
        raw_step = f"{line}@0::.*"
        try:
            caller.parse_step(raw_step)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"scenario caller line cannot be scripted: {line!r}") from error
        raw_steps.append(raw_step)
    return raw_steps


def _turn_kind(scenario: Any, turn_index: int) -> str:
    if not getattr(scenario, "commands", ()):
        return "search"
    if getattr(scenario, "place_query", None) is not None and turn_index == 1:
        return "search"
    return "action"


def _confirmation_time(trace: dict[str, Any]) -> float | None:
    segments = trace.get("segments")
    capture_started = _finite_timestamp(trace.get("capture_started"), "capture start")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("SMS confirmation has no transcript segment timestamps")
    transcript_parts = []
    ranges = []
    cursor = 0
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise RuntimeError("SMS confirmation has an invalid transcript segment")
        if cursor:
            transcript_parts.append(" ")
            cursor += 1
        start = cursor
        transcript_parts.append(segment["text"])
        cursor += len(segment["text"])
        end = _finite_timestamp(segment.get("end"), "transcription segment end")
        ranges.append((start, cursor, end))
    match = SMS_CONFIRMATION_PATTERN.search("".join(transcript_parts))
    if match is None:
        return None
    for start, end, segment_end in ranges:
        if start < match.end() <= end:
            return capture_started + segment_end
    raise RuntimeError("SMS confirmation prompt has no matching transcript segment timestamp")


def _completed_stdout(result: Any, label: str) -> str:
    if getattr(result, "returncode", None) != 0:
        raise RuntimeError(f"{label} failed: {getattr(result, 'stderr', '')}")
    stdout = getattr(result, "stdout", None)
    if not isinstance(stdout, str):
        raise RuntimeError(f"{label} returned no output")
    return stdout


def _capture_envelope(text: str, expected_turns: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError("remote scripted capture returned malformed JSON") from error
    if not isinstance(payload, dict) or set(payload) not in ({"turns"}, {"turns", "failure"}):
        raise RuntimeError("remote scripted capture returned an invalid envelope")
    traces = payload.get("turns")
    if not isinstance(traces, list) or len(traces) > expected_turns:
        raise RuntimeError("remote scripted capture returned missing or invalid turns")
    if "failure" not in payload:
        if len(traces) != expected_turns or not traces:
            raise RuntimeError("remote scripted capture returned missing or incomplete turns")
        return traces, None
    failure = payload["failure"]
    allowed_failure_fields = (
        {"turn", "message"},
        {"turn", "message", "speech_started_at"},
    )
    same_turn_hangup_timeout = (
        isinstance(failure, dict)
        and failure.get("message") == "room deletion was not observed before deadline"
        and isinstance(failure.get("turn"), int)
        and not isinstance(failure.get("turn"), bool)
        and failure["turn"] == len(traces)
        and bool(traces)
        and isinstance(traces[-1], dict)
        and traces[-1].get("turn") == failure["turn"]
        and traces[-1].get("room_deleted") is None
    )
    if (
        not isinstance(failure, dict)
        or set(failure) not in allowed_failure_fields
        or isinstance(failure.get("turn"), bool)
        or not isinstance(failure.get("turn"), int)
        or not (
            same_turn_hangup_timeout
            or (
                failure.get("message") != "room deletion was not observed before deadline"
                and failure["turn"] == len(traces) + 1
            )
        )
        or failure["turn"] > expected_turns
        or failure.get("message") not in {
            "scripted speech synthesis exceeded its deadline",
            NO_ANSWER_FAILURE,
            "room was deleted before all scripted lines were captured",
            "room deletion was not observed before deadline",
        }
        or (
            failure.get("message") == NO_ANSWER_FAILURE
        ) != ("speech_started_at" in failure)
        or (
            failure.get("message") == "room was deleted before all scripted lines were captured"
            and not traces
        )
    ):
        raise RuntimeError("remote scripted capture returned invalid partial failure metadata")
    if "speech_started_at" in failure:
        failure_started_at = _finite_timestamp(
            failure["speech_started_at"], "failed caller speech start"
        )
        if traces and not isinstance(traces[-1], dict):
            raise RuntimeError("remote scripted capture returned invalid partial turn metadata")
        if traces and failure_started_at <= _finite_timestamp(
            traces[-1].get("speech_started_at"), "caller speech start"
        ):
            raise RuntimeError("partial failure speech start is ambiguous or out of order")
    return traces, failure


def _voice_token(base_url: str) -> dict[str, Any]:
    parsed = urlsplit(base_url)
    try:
        is_loopback = parsed.hostname is not None and ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        is_loopback = False
    if (
        parsed.scheme != "http"
        or not is_loopback
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("dev voice-token endpoint must be an HTTP loopback origin")
    request = Request(
        f"{base_url.rstrip('/')}/v1/voice/token",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=TOKEN_REQUEST_DEADLINE_SECONDS) as response:
            if response.status != 200:
                raise RuntimeError(f"voice-token endpoint returned HTTP {response.status}")
            grant = json.loads(response.read())
    except (OSError, TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("dev voice-token request failed") from error
    if not isinstance(grant, dict):
        raise RuntimeError("voice-token response is not an object")
    token = grant.get("token")
    room = grant.get("room")
    url = grant.get("url")
    expires_at = grant.get("expires_at")
    token_parts = token.split(".") if isinstance(token, str) else []
    if (
        len(token_parts) != 3
        or any(not part for part in token_parts)
        or not isinstance(room, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", room) is None
    ):
        raise RuntimeError("voice-token response is missing a valid token or room")
    if not isinstance(url, str):
        raise RuntimeError("voice-token response is missing a valid LiveKit URL")
    try:
        livekit_url = urlsplit(url)
    except ValueError as error:
        raise RuntimeError("voice-token response has an invalid LiveKit URL") from error
    if livekit_url.scheme not in ("ws", "wss") or livekit_url.hostname is None:
        raise RuntimeError("voice-token response is missing a valid LiveKit URL")
    if not isinstance(expires_at, str):
        raise RuntimeError("voice-token response is missing token expiry")
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("voice-token response has an invalid expiry") from error
    if expiry.tzinfo is None:
        raise RuntimeError("voice-token response expiry has no timezone")
    return grant


def observe_scenario(scenario: Any, stack: Any) -> dict[str, Any]:
    """Run and strictly evaluate one scenario against the isolated DevStack."""
    raw_steps = _scenario_steps(scenario)
    close_after = getattr(scenario, "room_close_after", None)
    if close_after is not None and (
        isinstance(close_after, bool)
        or not isinstance(close_after, int)
        or not 1 <= close_after <= len(raw_steps)
    ):
        raise RuntimeError("scenario has an invalid room-close turn")
    base_url = getattr(stack, "base_url", None)
    if not isinstance(base_url, str) or not base_url:
        raise RuntimeError("dev stack has no base URL")
    grant = _voice_token(base_url)
    room = grant["room"]
    stack.start_worker(room)
    close_arg = "none" if close_after is None else str(close_after)
    capture = stack.run_voice(
        ["evals/runner.py", "--fake-phone", "--room-close-after", close_arg, room, *raw_steps],
        token=grant["token"],
        livekit_url=grant["url"],
    )
    capture_text = _completed_stdout(capture, "remote scripted capture")
    traces, capture_failure = _capture_envelope(capture_text, len(scenario.turns))

    phone_log_path = "voice/" + FAKE_PHONE_LOG
    phone_text = _completed_stdout(stack.run_remote(["sudo", "cat", phone_log_path]), "fake phone log read")
    phone_commands = _phone_commands(_json_lines(phone_text, "fake phone"))
    sdk_recording_applicable = not (
        not traces
        and capture_failure is not None
        and capture_failure["message"] == "scripted speech synthesis exceeded its deadline"
        and "speech_started_at" not in capture_failure
    )
    if sdk_recording_applicable:
        marker_log = _completed_stdout(
            stack.run_remote(["sudo", "cat", "voice/evals/delegations.jsonl"]),
            "delegation marker log read",
        )
        session_id = "voice-" + room
        record_path = "records/" + quote(session_id, safe="") + ".jsonl"
        record_text = _completed_stdout(
            stack.run_remote(["sudo", "cat", record_path]), "daemon SDK recording read"
        )
        recorded_turns = _recorded_turns(_json_lines(record_text, "SDK recording"))
        traces = _attribute_model_calls(
            traces,
            marker_log,
            room,
            recorded_turns,
            partial_capture=capture_failure is not None,
            failure_started_at=(
                capture_failure.get("speech_started_at")
                if capture_failure is not None
                else None
            ),
            failure_turn=(capture_failure.get("turn") if capture_failure is not None else None),
        )
    else:
        recorded_turns = []
    phone_commands = _match_phone_tools(phone_commands, recorded_turns)

    room_deleted_turns = []
    for index, (trace, expected) in enumerate(
        zip(traces, scenario.turns), 1
    ):
        if not isinstance(trace, dict) or trace.get("turn") != index or trace.get("room") != room:
            raise RuntimeError(f"captured turn {index} has invalid turn or room identity")
        if not isinstance(trace.get("transcript"), str) or not trace["transcript"].strip():
            raise RuntimeError(f"captured turn {index} has no complete transcript")
        for field in ("speech_started_at", "speech_end", "first_audio", "capture_started"):
            _finite_timestamp(trace.get(field), field)
        if not isinstance(trace.get("overlap"), bool):
            raise RuntimeError(f"captured turn {index} has no valid overlap observation")
        segments = trace.get("segments")
        if not isinstance(segments, list) or not segments:
            raise RuntimeError(f"captured turn {index} has no transcript segment evidence")
        for segment in segments:
            _segment_start([segment])
        trace["kind"] = _turn_kind(scenario, index)
        trace["expect_confirmation"] = getattr(expected, "sms_recipient", None) is not None
        trace["expect_hangup"] = close_after == index
        trace["confirmation"] = (
            _confirmation_time(trace) if trace["expect_confirmation"] else None
        )
        if trace.get("room_deleted") is not None:
            _finite_timestamp(trace["room_deleted"], "room deletion")
            room_deleted_turns.append(index)
    if len(room_deleted_turns) > 1:
        raise RuntimeError("capture observed room deletion more than once")
    room_closed_after = room_deleted_turns[0] if room_deleted_turns else None
    from evals.scenarios import evaluate_scenario_failures, evaluate_scenario_prefix

    transcripts = [trace["transcript"] for trace in traces]
    if capture_failure is None:
        failures = evaluate_scenario_failures(
            scenario,
            transcripts,
            phone_commands,
            room_closed_after,
        )
    else:
        failures = evaluate_scenario_prefix(
            scenario,
            transcripts,
            phone_commands,
            room_closed_after,
            failed_turn=(capture_failure["turn"] if capture_failure["turn"] > len(traces) else None),
        )
    product_failures = [
        {"turn": failure.turn, "message": failure.message}
        for failure in failures
    ]
    observation = {
        "room": room,
        "turns": traces,
        "phone_commands": phone_commands,
        "room_closed_after": room_closed_after,
    }
    if capture_failure is not None:
        observation["failure"] = capture_failure
        observation["product_failures"] = product_failures
    elif product_failures:
        observation["product_failures"] = product_failures
    return observation


async def _run_remote_capture_with_fake_phone(
    room_name: str, raw_steps: list[str], *, room_close_after: int | None
) -> list[dict[str, Any]]:
    base_url = os.environ.get("MENTAT_URL", "")
    log_path = Path(__file__).resolve().parents[1] / FAKE_PHONE_LOG
    log_path.unlink(missing_ok=True)
    phone_script = (
        "import asyncio, sys; from phone import run_fake_phone; "
        "asyncio.run(run_fake_phone(sys.argv[1], sys.argv[2], 'success'))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", phone_script, base_url, str(log_path)],
        cwd=Path(__file__).resolve().parents[1],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        await asyncio.sleep(0.2)
        if process.poll() is not None:
            raise RuntimeError("fake phone process exited before scripted capture")
        return await run_remote_capture(room_name, raw_steps, room_close_after=room_close_after)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)


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
    parser.add_argument("--fake-phone", action="store_true", help="run the loopback-only fake phone during capture")
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


def _run_local_eval(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python3 -m voice.evals.runner eval",
        description="Run repeated voice evaluations against one isolated DevStack",
    )
    parser.add_argument("--live", action="store_true", help="explicitly opt in to the live isolated stack")
    parser.add_argument("--runs", type=int, default=10, help="observations per scenario (default: 10)")
    parser.add_argument("--list", action="store_true", help="list the contracted scenarios without running them")
    arguments = parser.parse_args(argv)
    if arguments.list:
        if arguments.live or arguments.runs != 10:
            print("eval --list cannot be combined with --live or --runs", file=sys.stderr)
            return 2
        for scenario in SCENARIOS:
            print(scenario.name)
        return 0
    if not arguments.live:
        print("eval requires --live; no DevStack was started", file=sys.stderr)
        return 2
    if arguments.runs <= 0:
        print("eval --runs must be a positive integer", file=sys.stderr)
        return 2

    observations: dict[str, Any] = {"cases": []}
    capture_failures: list[tuple[int, int, str]] = []
    checkout = Path(__file__).resolve().parents[2]
    try:
        with DevStack(checkout=checkout, opt_in=True) as stack:
            for scenario_index, scenario in enumerate(SCENARIOS):
                runs = []
                for run_index in range(arguments.runs):
                    try:
                        runs.append(observe_scenario(scenario, stack))
                    except Exception as error:
                        message = _redact_diagnostics(
                            f"{scenario.name} run {run_index + 1}: {error}"
                        )
                        capture_failures.append((scenario_index, run_index, message))
                        runs.append({"failure": _redact_diagnostics(str(error))})
                observations["cases"].append({
                    "name": scenario.name,
                    "runs": runs,
                })
    except Exception as error:
        message = _redact_diagnostics(f"DevStack setup failed: {error}")
        observations["cases"] = [
            {"name": scenario.name, "runs": []}
            for scenario in SCENARIOS
        ]
        capture_failures.append((0, 0, message))

    report = score_observations(observations, required_runs=arguments.runs)
    for scenario_index, _run_index, message in capture_failures:
        report["failures"].append(message)
        case = report["cases"][scenario_index]
        case["failures"].append(message)
    report["passed"] = not report["failures"]
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "eval":
        return _run_local_eval(arguments[1:])
    room, steps, room_close_after = _parse_arguments(arguments)
    capture = _run_remote_capture_with_fake_phone if "--fake-phone" in arguments else run_remote_capture
    try:
        traces = asyncio.run(capture(room, steps, room_close_after=room_close_after))
    except PartialCaptureFailure as error:
        envelope = {"turns": error.turns, "failure": error.failure}
        print(json.dumps(envelope, separators=(",", ":")), flush=True)
        return 0
    print(json.dumps({"turns": traces}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
