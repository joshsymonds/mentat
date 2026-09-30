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
import tempfile
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import caller
from evals.dev_stack import DevStack, RemoteCommandError, _redact_diagnostics
from evals.report import NO_ANSWER_FAILURE, score_observations
from evals.scenarios import (
    SCENARIOS,
    SMS_CONFIRMATION_PATTERN,
    _SPOKEN_SMS_RECIPIENT,
    _sms_body_tokens,
    _spoken_sms_body,
    _spoken_sms_correction_body,
    _uncertain_without_alice_attribution,
)

RATE = caller.RATE
FRAME_SAMPLES = caller.FRAME_SAMPLES
PARTICIPANT_DEADLINE_SECONDS = 30.0
ANSWER_CAPTURE_DEADLINE_SECONDS = 45.0
ANSWER_TRANSCRIPTION_DEADLINE_SECONDS = 30.0
ROOM_DELETE_DEADLINE_SECONDS = 60.0
REMOTE_OPERATION_DEADLINE_SECONDS = 30.0
ROOM_POLL_INTERVAL_SECONDS = 0.25
TOKEN_REQUEST_DEADLINE_SECONDS = 10.0
FAKE_PHONE_LOG = "evals/phone.jsonl"
SMS_AUDIO_SCENARIOS = frozenset({"sms-say-back-yes", "sms-correction-new-yes"})
DEFAULT_VOICE_MODEL = "chatgpt/sol-fast"


def _retain_sms_audio(
    evidence_dir: Path,
    scenario: str,
    room_name: str,
    turn: int,
    transcript: str,
    pcm: bytes,
    sample_rate: int,
    channels: int,
) -> None:
    """Atomically retain one completed SMS turn outside the staged repository."""
    if scenario not in SMS_AUDIO_SCENARIOS:
        raise ValueError("audio retention is restricted to the SMS say-back scenarios")
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", room_name) is None:
        raise RuntimeError("SMS audio retention received an invalid room name")
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % (channels * 2):
        raise RuntimeError("SMS audio retention received invalid PCM")

    evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    if evidence_dir.is_symlink() or not evidence_dir.is_dir():
        raise RuntimeError("SMS audio evidence directory is not a private directory")
    if evidence_dir.stat().st_mode & 0o077:
        raise RuntimeError("SMS audio evidence directory is not a private directory")
    audio_dir = evidence_dir / "sms-audio"
    audio_dir.mkdir(mode=0o700, exist_ok=True)
    if audio_dir.is_symlink() or not audio_dir.is_dir():
        raise RuntimeError("SMS audio evidence directory is not a private directory")
    os.chmod(audio_dir, 0o700)

    filename = f"{room_name}-turn-{turn:03d}.wav"
    destination = audio_dir / filename
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(dir=audio_dir, prefix=".audio-", suffix=".tmp", delete=False) as temporary:
            temporary_name = temporary.name
        with wave.open(temporary_name, "wb") as output:
            output.setnchannels(channels)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm)
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    metadata_path = audio_dir / "transcripts.jsonl"
    metadata = json.dumps({
        "scenario": scenario,
        "room": room_name,
        "turn": turn,
        "transcript": transcript,
        "filename": filename,
    }, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    descriptor = os.open(
        metadata_path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    with os.fdopen(descriptor, "a", encoding="utf-8") as output:
        os.fchmod(output.fileno(), 0o600)
        output.write(metadata)
        output.flush()
        os.fsync(output.fileno())
SUPPORTED_VOICE_MODELS = frozenset({DEFAULT_VOICE_MODEL, "claude-opus-5-5", "claude-sonnet-5-5"})
SCRIPTED_TTS_TIMEOUT_PATTERN = re.compile(
    r"scripted speech synthesis for line ([1-9][0-9]*) exceeded its deadline"
)
WHISPER_HTTP_FAILURE_PATTERN = re.compile(
    r"Whisper transcription rejected utterance [1-9][0-9]* \(HTTP 4[0-9]{2}\)"
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


def _scripted_tts_timeout_line(message: Any) -> int | None:
    if not isinstance(message, str):
        return None
    match = SCRIPTED_TTS_TIMEOUT_PATTERN.fullmatch(message)
    return int(match.group(1)) if match is not None else None


def _is_scripted_tts_timeout(message: Any) -> bool:
    return _scripted_tts_timeout_line(message) is not None


def _is_whisper_http_failure(message: Any) -> bool:
    return isinstance(message, str) and WHISPER_HTTP_FAILURE_PATTERN.fullmatch(message) is not None


class PartialCaptureFailure(RuntimeError):
    """A scripted turn failed after earlier turns were captured completely."""

    def __init__(
        self,
        turns: list[dict[str, Any]],
        turn: int,
        message: str,
        speech_started_at: float | None = None,
        segments: list[dict[str, Any]] | None = None,
    ):
        super().__init__(message)
        self.turns = turns
        self.failure = {"turn": turn, "message": message}
        if speech_started_at is not None:
            self.failure["speech_started_at"] = speech_started_at
        if segments is not None:
            self.failure["segments"] = segments


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
    wall_time: Callable[[], float] = time.time
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
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise RuntimeError(f"{label} timestamp is missing or invalid") from error
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


def _pcm_utterances(
    answer_pcm: bytes,
    sample_rate: int,
    channels: int,
) -> list[tuple[float, float, bytes]]:
    """Split signed 16-bit PCM at sustained speech and 300 ms quiet boundaries."""
    if (
        not isinstance(answer_pcm, bytes)
        or isinstance(sample_rate, bool)
        or not isinstance(sample_rate, int)
        or sample_rate < 50
        or sample_rate % 50 != 0
        or isinstance(channels, bool)
        or not isinstance(channels, int)
        or channels < 1
        or not answer_pcm
        or len(answer_pcm) % (channels * 2) != 0
    ):
        raise RuntimeError(NO_ANSWER_FAILURE)

    window_frames = sample_rate // 50
    window_values = window_frames * channels
    window_bytes = window_values * 2
    window_count = len(answer_pcm) // window_bytes
    rms_values = []
    for window_index in range(window_count):
        offset = window_index * window_bytes
        values = struct.unpack(f"<{window_values}h", answer_pcm[offset : offset + window_bytes])
        rms_values.append(math.sqrt(sum(value * value for value in values) / len(values)))

    ranges = []
    utterance_start = None
    speech_run_start = None
    quiet_run_start = None
    quiet_windows = round(0.300 / PCM_WINDOW_SECONDS)
    for index, rms in enumerate(rms_values):
        if rms >= PCM_RMS_THRESHOLD:
            if speech_run_start is None:
                speech_run_start = index
            quiet_run_start = None
            continue

        if utterance_start is None:
            if speech_run_start is not None and index - speech_run_start >= 2:
                utterance_start = speech_run_start
                quiet_run_start = index
            speech_run_start = None
            continue

        if quiet_run_start is None:
            quiet_run_start = index
        if index - quiet_run_start + 1 >= quiet_windows:
            ranges.append((utterance_start, quiet_run_start))
            utterance_start = None
            speech_run_start = None
            quiet_run_start = None

    if utterance_start is None and speech_run_start is not None and window_count - speech_run_start >= 2:
        utterance_start = speech_run_start
    if utterance_start is not None:
        ranges.append((utterance_start, window_count))

    return [
        (
            start * PCM_WINDOW_SECONDS,
            end * PCM_WINDOW_SECONDS,
            answer_pcm[start * window_bytes : end * window_bytes],
        )
        for start, end in ranges
    ]


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
    retain_sms_audio_dir: Path | None = None,
    retain_sms_audio_scenario: str | None = None,
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
    if (retain_sms_audio_dir is None) != (retain_sms_audio_scenario is None):
        raise ValueError("SMS audio retention directory and scenario must be provided together")
    if retain_sms_audio_scenario is not None and retain_sms_audio_scenario not in SMS_AUDIO_SCENARIOS:
        raise ValueError("audio retention is restricted to the SMS say-back scenarios")
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

    synthesized: list[bytes] = []
    for index, step in enumerate(steps):
        try:
            pcm = await _with_deadline(
                dependencies.tts(dependencies.http, step.line),
                REMOTE_OPERATION_DEADLINE_SECONDS,
                "scripted speech synthesis",
            )
        except DeadlineExceeded as error:
            if str(error) != "scripted speech synthesis exceeded its deadline":
                raise
            try:
                pcm = await _with_deadline(
                    dependencies.tts(dependencies.http, step.line),
                    REMOTE_OPERATION_DEADLINE_SECONDS,
                    "scripted speech synthesis",
                )
            except DeadlineExceeded as retry_error:
                if str(retry_error) != "scripted speech synthesis exceeded its deadline":
                    raise
                raise PartialCaptureFailure(
                    [],
                    1,
                    f"scripted speech synthesis for line {index + 1} exceeded its deadline",
                ) from retry_error
        synthesized.append(pcm)

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
        publication = await _with_deadline(
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
        try:
            await caller.wait_for_microphone_ready(
                publication,
                deadline=participant_deadline,
                monotonic=dependencies.monotonic,
                sleep=dependencies.sleep,
            )
        except TimeoutError as error:
            raise DeadlineExceeded(str(error)) from error

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

        for index, (step, pcm) in enumerate(zip(steps, synthesized, strict=True)):
            await quiet(step.delay)
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
            paired_monotonic = _finite_timestamp(
                dependencies.monotonic(), "paired monotonic speech end"
            )
            paired_wall_time = _finite_timestamp(
                dependencies.wall_time(), "paired wall speech end"
            )
            speech_end_wall = _finite_timestamp(
                paired_wall_time + speech_end - paired_monotonic,
                "speech_end_wall",
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
            utterances = _pcm_utterances(answer_pcm, sample_rate, channels)
            raw_segments = []
            segments = []
            transcription_failure: tuple[caller.WhisperTranscriptionError, int] | None = None
            if (
                not math.isfinite(ANSWER_TRANSCRIPTION_DEADLINE_SECONDS)
                or ANSWER_TRANSCRIPTION_DEADLINE_SECONDS <= 0
            ):
                raise RuntimeError("answer transcription deadline must be finite and positive")
            transcription_deadline = (
                dependencies.monotonic() + ANSWER_TRANSCRIPTION_DEADLINE_SECONDS
            )
            try:
                transcription_tasks = [
                    asyncio.create_task(
                        dependencies.transcribe(
                            dependencies.http, utterance_pcm, sample_rate, channels
                        )
                    )
                    for _utterance_start, _utterance_end, utterance_pcm in utterances
                ]
                remaining = transcription_deadline - dependencies.monotonic()
                if remaining > 0:
                    done, pending = await asyncio.wait(
                        transcription_tasks,
                        timeout=remaining,
                        return_when=asyncio.ALL_COMPLETED,
                    )
                else:
                    done, pending = set(), set(transcription_tasks)
                if pending:
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)

                for utterance_ordinal, (utterance, task) in enumerate(
                    zip(utterances, transcription_tasks, strict=True), start=1
                ):
                    if task not in done:
                        break
                    utterance_start, utterance_end, _utterance_pcm = utterance
                    try:
                        utterance_segments = task.result()
                    except caller.WhisperTranscriptionError as error:
                        if transcription_failure is None:
                            transcription_failure = (error, utterance_ordinal)
                        continue
                    if not isinstance(utterance_segments, list):
                        raise RuntimeError("transcription returned invalid segments")
                    raw_segments.extend(utterance_segments)
                    texts = []
                    for segment in utterance_segments:
                        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
                            raise RuntimeError("transcription segment has no text")
                        text = segment["text"].strip()
                        if text:
                            texts.append(text)
                    text = " ".join(texts)
                    if text:
                        segments.append({
                            "start": utterance_start,
                            "end": utterance_end,
                            "text": text,
                        })
            finally:
                if retain_sms_audio_dir is not None and retain_sms_audio_scenario is not None:
                    _retain_sms_audio(
                        retain_sms_audio_dir,
                        retain_sms_audio_scenario,
                        room_name,
                        index + 1,
                        _trace_text(segments),
                        answer_pcm,
                        sample_rate,
                        channels,
                    )
            if transcription_failure is not None:
                error, utterance_ordinal = transcription_failure
                raise PartialCaptureFailure(
                    traces,
                    index + 1,
                    f"Whisper transcription rejected utterance {utterance_ordinal} "
                    f"(HTTP {error.status})",
                    speech_started_at=speech_started_at,
                ) from error
            if pending:
                raise PartialCaptureFailure(
                    traces,
                    index + 1,
                    "answer transcription exceeded its deadline",
                    speech_started_at=speech_started_at,
                    segments=raw_segments or None,
                )
            if not segments:
                raise PartialCaptureFailure(
                    traces,
                    index + 1,
                    NO_ANSWER_FAILURE,
                    speech_started_at=speech_started_at,
                    segments=raw_segments or None,
                )
            transcript = _trace_text(segments)
            trace = {
                "turn": index + 1,
                "room": room_name,
                "line": step.line,
                "transcript": transcript,
                "speech_started_at": speech_started_at,
                "speech_end": speech_end,
                "speech_end_wall": speech_end_wall,
                "capture_started": capture_started,
                "first_audio": first_audio,
                "overlap": overlap,
                "segments": segments,
                "raw_segments": raw_segments,
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
        if label in {"fake phone", "delegation marker"}:
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


def _requested_voice_model() -> str:
    model = os.environ.get("MENTAT_VOICE_MODEL", DEFAULT_VOICE_MODEL)
    if model not in SUPPORTED_VOICE_MODELS:
        raise RuntimeError(f"unsupported requested voice model: {model!r}")
    return model


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
    unattributed_model_calls: list[dict[str, Any]] | None = None,
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
    aligned_turns = []
    for recorded in recorded_turns:
        if not isinstance(recorded, dict):
            raise RuntimeError("recorded SDK result has invalid model-call evidence")
        skipped_results = recorded.get("skipped_execution_errors_before", 0)
        if (
            isinstance(skipped_results, bool)
            or not isinstance(skipped_results, int)
            or skipped_results < 0
        ):
            raise RuntimeError("recorded SDK result has invalid skipped-result evidence")
        aligned_turns.extend([None] * skipped_results)
        aligned_turns.append(recorded)
    if len(delegations) != len(aligned_turns):
        raise RuntimeError("delegation and SDK result counts differ")

    for delegation, recorded in zip(delegations, aligned_turns, strict=True):
        created_at = delegation["created_at"]
        if recorded is None:
            recorded = {"model_calls": [], "phone_tools": []}
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
                    if model_calls or recorded.get("phone_tools", []):
                        if (
                            isinstance(failure_turn, bool)
                            or not isinstance(failure_turn, int)
                            or failure_turn != len(starts) + 1
                        ):
                            raise RuntimeError("failed-turn SDK evidence has no valid turn attribution")
                    if model_calls:
                        if unattributed_model_calls is None:
                            raise RuntimeError("failed-turn model calls have no provenance destination")
                        for call in model_calls:
                            if not isinstance(call, dict):
                                raise RuntimeError("recorded SDK result has invalid model-call evidence")
                            unattributed_model_calls.append({**call, "turn": failure_turn})
                    for tool in recorded.get("phone_tools", []):
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
    skipped_since_success = 0
    skipped_total = 0
    for message_index, message in enumerate(messages):
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
                    if not isinstance(message_id, str) or not message_id:
                        raise RuntimeError("recorded message_start has no finite model-call evidence")
                    usage = started.get("usage")
                    calls.append({
                        "id": message_id,
                        "model": model if isinstance(model, str) and model else None,
                        "service_tier": usage.get("service_tier") if isinstance(usage, dict) else None,
                    })
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
                        tools.append({
                            "turn": len(turns) + skipped_total + 1,
                            "name": name,
                            "kind": tool_kind,
                            "input": arguments,
                        })
        if not calls:
            later_model_call_result = False
            later_result_has_start = False
            if (
                message.get("subtype") == "error_during_execution"
                and message.get("is_error") is True
            ):
                for later in messages[message_index + 1:]:
                    if later.get("type") == "stream_event":
                        event = later.get("event")
                        if isinstance(event, dict) and event.get("type") == "message_start":
                            later_result_has_start = True
                    if later.get("type") == "result":
                        if later_result_has_start:
                            later_model_call_result = True
                            break
                        later_result_has_start = False
            if later_model_call_result:
                skipped_since_success += 1
                skipped_total += 1
                current = []
                continue
            raise RuntimeError("recorded assistant result has no message_start model-call evidence")
        result_usage = message.get("usage")
        result_usage = result_usage if isinstance(result_usage, dict) else {}
        for call in calls:
            call["speed"] = result_usage.get("speed")
            call["result_service_tier"] = result_usage.get("service_tier")
            if "fast_mode_state" in message:
                call["fast_mode_state"] = message["fast_mode_state"]
        recorded = {"model_calls": calls, "phone_tools": tools}
        if skipped_since_success:
            recorded["skipped_execution_errors_before"] = skipped_since_success
            skipped_since_success = 0
        turns.append(recorded)
        current = []
    if any(
        item.get("type") in ("assistant", "user")
        or (item.get("type") == "stream_event" and isinstance(item.get("event"), dict)
            and item["event"].get("type") == "message_start")
        for item in current
    ):
        raise RuntimeError("recorded SDK turn is missing its top-level result boundary")
    return turns


def _model_call_provenance(
    call: dict[str, Any], requested_model: str
) -> dict[str, Any]:
    evidence = {
        "id": call.get("id"),
        "requested_model": requested_model,
        "observed_model": call.get("model"),
        "service_tier": call.get("service_tier"),
        "result_service_tier": call.get("result_service_tier"),
        "speed": call.get("speed"),
        "fast_mode_state": call.get("fast_mode_state"),
    }
    failures = []
    observed_model = evidence["observed_model"]
    model_matches = observed_model == requested_model
    if requested_model == "chatgpt/sol-fast":
        model_matches = model_matches or observed_model == "gpt-6-sol"
    elif requested_model in ("claude-opus-5-5", "claude-sonnet-5-5") and isinstance(observed_model, str):
        model_matches = model_matches or re.fullmatch(
            rf"{re.escape(requested_model)}-[0-9]{{8}}", observed_model
        ) is not None
    if not isinstance(observed_model, str) or not observed_model:
        failures.append(f"observed model is unproven (requested {requested_model})")
    elif not model_matches:
        failures.append(
            f"requested model {requested_model} but observed {observed_model}"
        )
    if evidence["result_service_tier"] != "standard":
        failures.append(
            f"result usage service tier is unproven or nonstandard: "
            f"{evidence['result_service_tier']!r}"
        )
    if requested_model == "claude-opus-5-5":
        if evidence["service_tier"] != "standard":
            failures.append(
                f"message_start service tier is unproven or nonstandard: "
                f"{evidence['service_tier']!r}"
            )
        if evidence["speed"] != "standard":
            failures.append(
                f"result usage speed is unproven or nonstandard: {evidence['speed']!r}"
            )
        if evidence["fast_mode_state"] != "off":
            failures.append(
                f"fast mode is unproven or enabled: {evidence['fast_mode_state']!r}"
            )
    elif requested_model == "claude-sonnet-5-5":
        if evidence["service_tier"] != "standard":
            failures.append(
                f"message_start service tier is unproven or nonstandard: "
                f"{evidence['service_tier']!r}"
            )
        if evidence["speed"] != "standard":
            failures.append(
                f"result usage speed is unproven or nonstandard: {evidence['speed']!r}"
            )
        if "fast_mode_state" in call and evidence["fast_mode_state"] != "off":
            failures.append(
                f"fast mode is unproven or enabled: {evidence['fast_mode_state']!r}"
            )
    elif evidence["service_tier"] not in (None, "standard"):
        failures.append(
            f"message_start service tier is nonstandard: {evidence['service_tier']!r}"
        )
    evidence["failures"] = failures
    return evidence


def _add_model_provenance(
    report: dict[str, Any], observations: dict[str, Any], requested_model: str
) -> None:
    observation_cases = observations.get("cases")
    if not isinstance(observation_cases, list):
        return
    for case_index, case_report in enumerate(report.get("cases", [])):
        if case_index >= len(observation_cases):
            continue
        case_observation = observation_cases[case_index]
        if not isinstance(case_observation, dict):
            continue
        runs = case_observation.get("runs")
        if not isinstance(runs, list):
            continue
        for run_index, run in enumerate(runs):
            if not isinstance(run, dict):
                continue
            run_turns = run.get("turns")
            if not isinstance(run_turns, list):
                run_turns = []
            for turn_index, turn in enumerate(run_turns, 1):
                if not isinstance(turn, dict):
                    continue
                calls = turn.get("model_calls")
                calls = calls if isinstance(calls, list) else []
                proven_calls = [
                    _model_call_provenance(call, requested_model)
                    for call in calls
                    if isinstance(call, dict)
                ]
                provenance_failures = [
                    failure
                    for call in proven_calls
                    for failure in call["failures"]
                ]
                if len(proven_calls) != len(calls):
                    provenance_failures.append("model-call evidence is invalid")
                if not proven_calls:
                    provenance_failures.append("model-call evidence is missing")
                provenance = {
                    "requested_model": requested_model,
                    "model_calls": proven_calls,
                    "verified": not provenance_failures,
                }
                case_turn = next(
                    (
                        item for item in case_report.get("turns", [])
                        if item.get("run") == run_index + 1
                        and item.get("turn") == turn_index
                    ),
                    None,
                )
                if case_turn is not None:
                    case_turn["model_provenance"] = provenance
                for failure in dict.fromkeys(provenance_failures):
                    message = (
                        f"{case_report['name']} run {run_index + 1} turn {turn_index}: "
                        f"model provenance failure: {failure}"
                    )
                    case_report["failures"].append(message)
                    report["failures"].append(message)
            extra_calls = run.get("unattributed_model_calls", [])
            if not isinstance(extra_calls, list):
                extra_calls = [{"turn": 1, "invalid": True}]
            extra_provenance = []
            for call in extra_calls:
                turn_number = call.get("turn") if isinstance(call, dict) else None
                if (
                    isinstance(turn_number, bool)
                    or not isinstance(turn_number, int)
                    or turn_number <= 0
                    or not isinstance(call, dict)
                ):
                    evidence = {"turn": turn_number, "failures": ["model-call evidence is invalid"]}
                else:
                    evidence = {
                        "turn": turn_number,
                        **_model_call_provenance(call, requested_model),
                    }
                extra_provenance.append(evidence)
                for failure in evidence["failures"]:
                    message = (
                        f"{case_report['name']} run {run_index + 1} turn {evidence['turn']}: "
                        f"model provenance failure: {failure}"
                    )
                    case_report["failures"].append(message)
                    report["failures"].append(message)
            if extra_provenance:
                case_report.setdefault("unattributed_model_calls", []).append({
                    "run": run_index + 1,
                    "model_calls": extra_provenance,
                })
    report["passed"] = not report["failures"]


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
            received_at = entry.get("received_at")
            try:
                received_at = _finite_timestamp(received_at, "fake phone command receipt")
            except RuntimeError as error:
                raise RuntimeError(
                    "fake phone command receipt timestamp is missing or invalid"
                ) from error
            commands.append({**payload, "received_at": received_at})
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
            if key in ("id", "kind", "expires_at", "received_at"):
                continue
            if arguments.get(key) != value:
                raise RuntimeError(f"fake phone command {index} does not match its recorded phone tool arguments")
        matched.append({**command, "turn": tool["turn"]})
    return matched


def _answer_time(
    trace: dict[str, Any],
    expectation: Any,
    *,
    scenario_name: str | None = None,
    turn_index: int | None = None,
) -> float | None:
    capture_started = _finite_timestamp(trace.get("capture_started"), "capture start")
    speech_end = _finite_timestamp(trace.get("speech_end"), "speech end")
    segments = trace.get("segments")
    patterns = getattr(expectation, "answer_patterns", None)
    reject_patterns = getattr(expectation, "reject_patterns", ())
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("answer segment timestamp evidence is missing")
    if not isinstance(patterns, tuple) or not patterns or not isinstance(reject_patterns, tuple):
        raise RuntimeError("answer expectation patterns are missing or invalid")

    transcript_parts = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise RuntimeError("answer segment timestamp evidence is malformed")
        if index:
            transcript_parts.append(" ")
        transcript_parts.append(segment["text"])
        segment_end = _finite_timestamp(segment.get("end"), "answer segment timestamp")
        segment_start = _finite_timestamp(segment.get("start"), "answer segment timestamp")
        if segment_start < 0 or segment_end < segment_start:
            raise RuntimeError("answer segment timestamp bounds are invalid")
        text = "".join(transcript_parts)
        try:
            matches = all(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
            rejected = any(re.search(pattern, text, re.IGNORECASE) for pattern in reject_patterns)
        except (TypeError, re.error) as error:
            raise RuntimeError("answer expectation contains an invalid pattern") from error
        if not matches or rejected or segment_start < speech_end - capture_started:
            continue
        if (
            scenario_name == "alice-keck-context-chain"
            and turn_index == 1
            and _uncertain_without_alice_attribution(text)
        ):
            continue
        sms_recipient = getattr(expectation, "sms_recipient", None)
        sms_body = getattr(expectation, "sms_body", None)
        if sms_recipient is not None or sms_body is not None:
            if not isinstance(sms_recipient, str) or not isinstance(sms_body, str):
                raise RuntimeError("answer expectation has an incomplete SMS say-back")
            try:
                spoken_body = (
                    _spoken_sms_correction_body(
                        text,
                        scenario_name or "scenario",
                        turn_index or 1,
                    )
                    if scenario_name == "sms-correction-new-yes"
                    and turn_index == 2
                    and _SPOKEN_SMS_RECIPIENT.search(text) is None
                    else _spoken_sms_body(
                        text,
                        sms_recipient,
                        sms_body,
                        scenario_name or "scenario",
                        turn_index or 1,
                    )
                )
            except AssertionError:
                continue
            if _sms_body_tokens(spoken_body) != _sms_body_tokens(sms_body):
                continue
        return capture_started + segment_start
    return None


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
    if getattr(scenario, "place_query", None) is not None and turn_index == 1:
        return "search"
    if any(
        isinstance(command, dict) and command.get("turn") == turn_index
        for command in getattr(scenario, "commands", ())
    ):
        return "action"
    return "search"


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
        stderr = _redact_diagnostics(getattr(result, "stderr", ""))
        raise RuntimeError(f"{label} failed: {stderr}")
    stdout = getattr(result, "stdout", None)
    if not isinstance(stdout, str):
        raise RuntimeError(f"{label} returned no output")
    return stdout


def _remote_artifact_text(
    stack: Any, path: str, label: str, *, allow_missing: bool = False
) -> str | None:
    try:
        result = stack.run_remote(["sudo", "cat", path])
    except subprocess.CalledProcessError as error:
        stderr = _redact_diagnostics(error.stderr).strip()
        if (
            allow_missing
            and error.returncode == 1
            and stderr == f"cat: {path}: No such file or directory"
        ):
            return None
        detail = stderr or f"remote command exited with status {error.returncode}"
        raise RuntimeError(f"{label} at {path} read failed: {detail}") from error
    return _completed_stdout(result, f"{label} at {path} read")


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
        {"turn", "message", "speech_started_at", "segments"},
    )
    preflight_tts_line = _scripted_tts_timeout_line(
        failure.get("message") if isinstance(failure, dict) else None
    )
    preflight_tts_failure = (
        preflight_tts_line is not None and preflight_tts_line <= expected_turns
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
                and (
                    (preflight_tts_failure and not traces and failure["turn"] == 1)
                    or (
                        not preflight_tts_failure
                        and failure["turn"] == len(traces) + 1
                    )
                )
            )
        )
        or failure["turn"] > expected_turns
        or not (
            preflight_tts_failure
            or failure.get("message") in {
                "answer transcription exceeded its deadline",
                NO_ANSWER_FAILURE,
                "room was deleted before all scripted lines were captured",
                "room deletion was not observed before deadline",
            }
            or _is_whisper_http_failure(failure.get("message"))
        )
        or (
            failure.get("message") in {
                NO_ANSWER_FAILURE,
                "answer transcription exceeded its deadline",
            }
            or _is_whisper_http_failure(failure.get("message"))
        ) != ("speech_started_at" in failure)
        or (
            failure.get("message") == "room was deleted before all scripted lines were captured"
            and not traces
        )
    ):
        raise RuntimeError("remote scripted capture returned invalid partial failure metadata")
    if "segments" in failure and (
        failure.get("message") not in {
            NO_ANSWER_FAILURE,
            "answer transcription exceeded its deadline",
        }
        or not isinstance(failure["segments"], list)
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


def _is_preflight_tts_capture_failure(
    traces: list[dict[str, Any]], failure: dict[str, Any] | None
) -> bool:
    return (
        not traces
        and isinstance(failure, dict)
        and failure.get("turn") == 1
        and _is_scripted_tts_timeout(failure.get("message"))
        and "speech_started_at" not in failure
    )


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


def _capture_command(scenario: Any, room: str, raw_steps: list[str]) -> list[str]:
    close_after = getattr(scenario, "room_close_after", None)
    close_arg = "none" if close_after is None else str(close_after)
    command = ["evals/runner.py", "--fake-phone", "--room-close-after", close_arg]
    if getattr(scenario, "name", None) in SMS_AUDIO_SCENARIOS:
        command.extend(("--retain-sms-audio", scenario.name))
    command.extend((room, *raw_steps))
    return command


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
    command = _capture_command(scenario, room, raw_steps)
    try:
        capture = stack.run_voice(
            command,
            token=grant["token"],
            livekit_url=grant["url"],
        )
    except RemoteCommandError as error:
        if error.returncode != 1 or not isinstance(error.output, str):
            raise
        try:
            traces, capture_failure = _capture_envelope(error.output, len(scenario.turns))
        except RuntimeError:
            raise error
        if not _is_preflight_tts_capture_failure(traces, capture_failure):
            raise
        capture_returncode = error.returncode
    else:
        capture_text = getattr(capture, "stdout", None)
        if not isinstance(capture_text, str):
            raise RuntimeError("remote scripted capture returned no output")
        traces, capture_failure = _capture_envelope(capture_text, len(scenario.turns))
        capture_returncode = getattr(capture, "returncode", None)

    if capture_returncode != 0 and not _is_preflight_tts_capture_failure(
        traces, capture_failure
    ):
        raise RuntimeError(
            f"remote scripted capture failed: {getattr(capture, 'stderr', '')}"
        )

    phone_log_path = "voice/" + FAKE_PHONE_LOG
    phone_text = _remote_artifact_text(stack, phone_log_path, "fake phone log")
    if phone_text is None:
        raise RuntimeError(f"fake phone log at {phone_log_path} is missing")
    phone_commands = _phone_commands(_json_lines(phone_text, "fake phone"))
    sdk_recording_applicable = not _is_preflight_tts_capture_failure(
        traces, capture_failure
    )
    if sdk_recording_applicable:
        marker_path = "voice/evals/delegations.jsonl"
        marker_log = _remote_artifact_text(stack, marker_path, "delegation marker log")
        if marker_log is None:
            raise RuntimeError(f"delegation marker log at {marker_path} is missing")
        session_id = "voice-" + room
        record_path = "records/" + quote(session_id, safe="") + ".jsonl"
        record_text = _remote_artifact_text(
            stack, record_path, "daemon SDK recording", allow_missing=True
        )
        recorded_turns = (
            []
            if record_text is None
            else _recorded_turns(_json_lines(record_text, "SDK recording"))
        )
        unattributed_model_calls: list[dict[str, Any]] = []
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
            unattributed_model_calls=unattributed_model_calls,
        )
    else:
        unattributed_model_calls = []
        recorded_turns = []
    phone_commands = _match_phone_tools(phone_commands, recorded_turns)
    # Phone receipt and paired speech-end timestamps use the same epoch clock.
    command_receipts: dict[int, float] = {}
    for command in phone_commands:
        turn = command["turn"]
        received_at = command["received_at"]
        command_receipts[turn] = min(command_receipts.get(turn, received_at), received_at)

    room_deleted_turns = []
    for index, (trace, expected) in enumerate(
        zip(traces, scenario.turns), 1
    ):
        if not isinstance(trace, dict) or trace.get("turn") != index or trace.get("room") != room:
            raise RuntimeError(f"captured turn {index} has invalid turn or room identity")
        if not isinstance(trace.get("transcript"), str) or not trace["transcript"].strip():
            raise RuntimeError(f"captured turn {index} has no complete transcript")
        for field in (
            "speech_started_at",
            "speech_end",
            "speech_end_wall",
            "first_audio",
            "capture_started",
        ):
            _finite_timestamp(trace.get(field), field)
        if not isinstance(trace.get("overlap"), bool):
            raise RuntimeError(f"captured turn {index} has no valid overlap observation")
        segments = trace.get("segments")
        if not isinstance(segments, list) or not segments:
            raise RuntimeError(f"captured turn {index} has no transcript segment evidence")
        for segment in segments:
            _segment_start([segment])
        raw_segments = trace.get("raw_segments")
        if raw_segments is not None and (
            not isinstance(raw_segments, list) or not raw_segments
        ):
            raise RuntimeError(f"captured turn {index} has invalid raw transcript segment evidence")
        trace["kind"] = _turn_kind(scenario, index)
        trace["command_received_at"] = command_receipts.get(index)
        trace["answer_at"] = _answer_time(
            trace,
            expected,
            scenario_name=scenario.name,
            turn_index=index,
        )
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
    early_close = (
        capture_failure is not None
        and capture_failure["message"]
        == "room was deleted before all scripted lines were captured"
    )
    early_close_after = capture_failure["turn"] - 1 if early_close else None
    room_closed_after = (
        early_close_after
        if early_close
        else room_deleted_turns[0] if room_deleted_turns else None
    )
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
            None if early_close else room_closed_after,
            failed_turn=(
                None
                if early_close
                else capture_failure["turn"]
                if capture_failure["turn"] > len(traces)
                else None
            ),
        )
    product_failures = [
        {"turn": failure.turn, "message": failure.message}
        for failure in failures
    ]
    if early_close:
        follow_ups = len(raw_steps) - early_close_after
        product_failures.append({
            "turn": early_close_after,
            "message": (
                f"call ended after turn {early_close_after} with "
                f"{follow_ups} follow-ups remaining"
            ),
        })
    observation = {
        "room": room,
        "turns": traces,
        "phone_commands": phone_commands,
        "room_closed_after": room_closed_after,
    }
    if unattributed_model_calls:
        observation["unattributed_model_calls"] = unattributed_model_calls
    if capture_failure is not None and not early_close:
        observation["failure"] = capture_failure
        observation["product_failures"] = product_failures
    elif product_failures:
        observation["product_failures"] = product_failures
    return observation


async def _run_remote_capture_with_fake_phone(
    room_name: str,
    raw_steps: list[str],
    *,
    room_close_after: int | None,
    retain_sms_audio_dir: Path | None = None,
    retain_sms_audio_scenario: str | None = None,
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
        return await run_remote_capture(
            room_name,
            raw_steps,
            room_close_after=room_close_after,
            retain_sms_audio_dir=retain_sms_audio_dir,
            retain_sms_audio_scenario=retain_sms_audio_scenario,
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 5)
            except subprocess.TimeoutExpired:
                process.kill()
                await asyncio.to_thread(process.wait)


async def run_remote_capture(
    room_name: str,
    raw_steps: list[str],
    *,
    room_close_after: int | None = None,
    retain_sms_audio_dir: Path | None = None,
    retain_sms_audio_scenario: str | None = None,
) -> list[dict[str, Any]]:
    import aiohttp

    async with aiohttp.ClientSession() as http:
        return await capture_script(
            room_name,
            raw_steps,
            dependencies=_dependencies(http),
            room_close_after=room_close_after,
            retain_sms_audio_dir=retain_sms_audio_dir,
            retain_sms_audio_scenario=retain_sms_audio_scenario,
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
    try:
        requested_model = _requested_voice_model()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
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
    report["requested_model"] = requested_model
    _add_model_provenance(report, observations, requested_model)
    report["passed"] = not report["failures"]
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "eval":
        return _run_local_eval(arguments[1:])
    retain_sms_audio_scenario = None
    if "--retain-sms-audio" in arguments:
        flag_index = arguments.index("--retain-sms-audio")
        if flag_index + 1 >= len(arguments) or arguments.count("--retain-sms-audio") != 1:
            raise RuntimeError("--retain-sms-audio requires one SMS scenario name")
        retain_sms_audio_scenario = arguments[flag_index + 1]
        if retain_sms_audio_scenario not in SMS_AUDIO_SCENARIOS:
            raise RuntimeError("audio retention is restricted to the SMS say-back scenarios")
        arguments = arguments[:flag_index] + arguments[flag_index + 2:]
        evidence_path = os.environ.get("MENTAT_EVAL_RETAINED_EVIDENCE_DIR")
        if not evidence_path:
            raise RuntimeError("private retained-evidence directory is unavailable")
        retain_sms_audio_dir = Path(evidence_path)
    else:
        retain_sms_audio_dir = None
    room, steps, room_close_after = _parse_arguments(arguments)
    capture = _run_remote_capture_with_fake_phone if "--fake-phone" in arguments else run_remote_capture
    capture_options = {"room_close_after": room_close_after}
    if retain_sms_audio_scenario is not None:
        capture_options.update({
            "retain_sms_audio_dir": retain_sms_audio_dir,
            "retain_sms_audio_scenario": retain_sms_audio_scenario,
        })
    try:
        traces = asyncio.run(capture(room, steps, **capture_options))
    except PartialCaptureFailure as error:
        envelope = {"turns": error.turns, "failure": error.failure}
        print(json.dumps(envelope, separators=(",", ":")), flush=True)
        return int(
            not error.turns
            and error.failure.get("turn") == 1
            and _is_scripted_tts_timeout(error.failure.get("message"))
        )
    print(json.dumps({"turns": traces}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
