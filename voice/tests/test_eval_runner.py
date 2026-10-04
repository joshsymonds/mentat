"""Offline tests for remote scripted caller capture."""

import asyncio
import hashlib
import json
import os
import struct
import sys
import time
import unittest
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import runner
from evals.runner import CaptureDependencies, capture_script
from evals.scenarios import SCENARIOS


TEST_VOICE_ENV = {
    "LIVEKIT_API_KEY": "key",
    "LIVEKIT_API_SECRET": "secret",
    "MENTAT_VOICE_TOKEN": "test-issued.header.signature",
    "LIVEKIT_URL": "wss://test-livekit.invalid",
}


def pcm_windows(*levels, sample_rate=24000, channels=1):
    """Encode PCM levels as 20 ms signed Int16 windows."""
    samples_per_window = sample_rate // 50 * channels
    values = [level for level in levels for _ in range(samples_per_window)]
    return struct.pack(f"<{len(values)}h", *values)


RENDERED_PCM_TEXT = {}


@contextmanager
def _run_context(stack):
    yield stack


def _batch_for(stack):
    class Batch:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def run(self, _run_id):
            return _run_context(stack)

    return Batch()


def rendered_pcm(text, seconds_per_character=0.2):
    """Return deterministic mono PCM with a realistic duration for a scripted line."""
    samples = int(len(text) * seconds_per_character * runner.RATE)
    marker = int.from_bytes(hashlib.sha256(text.encode()).digest()[:2], "little") % 30000 + 1
    pcm = struct.pack("<h", marker) * samples
    RENDERED_PCM_TEXT[pcm] = text
    return pcm


def rendered_aware_transcriber(transcribe):
    async def wrapped(http, pcm, sample_rate, channels):
        if pcm in RENDERED_PCM_TEXT:
            return [{"start": 0.0, "end": 1.0, "text": RENDERED_PCM_TEXT[pcm]}]
        return await transcribe(http, pcm, sample_rate, channels)

    return wrapped


class Clock:
    def __init__(self):
        self.now = 10.0
        self.epoch = 1_700_000_000.0

    def monotonic(self):
        return self.now

    def wall_time(self):
        return self.epoch + (self.now - 10.0)

    async def sleep(self, seconds):
        self.now += seconds


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_phone_finishes_inside_the_caller_process_group(self):
        events = []

        class PhoneProcess:
            def __init__(self):
                self.returncode = None

            def poll(self):
                return self.returncode

            def terminate(self):
                events.append("terminate-phone")
                self.returncode = 0

            def kill(self):
                events.append("kill-phone")
                self.returncode = -9

            def wait(self, timeout=None):
                events.append(("wait-phone", timeout))
                return self.returncode

        process = PhoneProcess()

        async def capture(*_args, **_kwargs):
            events.append("capture-complete")
            return [{"turn": 1}]

        async def sleep(_seconds):
            events.append("phone-started")

        with (
            patch.object(runner.subprocess, "Popen", return_value=process) as popen,
            patch.object(runner, "run_remote_capture", capture),
            patch.object(runner.asyncio, "sleep", sleep),
        ):
            result = await runner._run_remote_capture_with_fake_phone(
                "private-room", ["Question"], room_close_after=None
            )

        self.assertEqual(result, [{"turn": 1}])
        self.assertNotIn("start_new_session", popen.call_args.kwargs)
        self.assertLess(events.index("capture-complete"), events.index("terminate-phone"))
        self.assertIn(("wait-phone", 5), events)

    async def test_capture_deadline_covers_extended_answer_window(self):
        self.assertGreaterEqual(
            runner.ANSWER_CAPTURE_DEADLINE_SECONDS,
            runner.caller.MAX_CAPTURE_SECONDS + 30.0,
        )

    async def test_multi_turn_trace_captures_before_speech_and_observes_room_deletion(self):
        clock = Clock()
        events = []
        room_presence = [True, False]

        class Source:
            async def capture_frame(self, frame):
                events.append("speech-frame")
                clock.now += 0.1

            async def wait_for_playout(self):
                events.append("playout")
                clock.now += 0.2

        class Room:
            remote_participants = {
                "agent": SimpleNamespace()
            }
            local_participant = SimpleNamespace(
                publish_track=lambda *_args, **_kwargs: asyncio.sleep(
                    0,
                    result=SimpleNamespace(wait_for_subscription=lambda: asyncio.sleep(0)),
                )
            )

            def on(self, event):
                self.callback_event = event
                return lambda callback: callback

            async def connect(self, *_args, **_kwargs):
                events.append("connect")

            async def disconnect(self):
                events.append("disconnect")

        class AccessToken:
            def __init__(self, *_args):
                pass

            def with_identity(self, _identity):
                return self

            def with_grants(self, _grants):
                return self

            def to_jwt(self):
                return "fake-token"

        class LiveKitAPI:
            def __init__(self, *_args, **_kwargs):
                self.room = self
                self.calls = 0

            async def list_rooms(self, _request):
                clock.now += 0.2
                present = room_presence[min(self.calls, len(room_presence) - 1)]
                self.calls += 1
                return SimpleNamespace(rooms=[SimpleNamespace(name="android-selected-room")] if present else [])

            async def aclose(self):
                pass

        class ContinuousCapture:
            def __init__(self, _queue, _ended=None):
                pass

            async def start(self):
                events.append("capture-start")

            async def result(self):
                events.append("capture-result")
                clock.now += 0.05
                return pcm_windows(0, 0, 0, 0, 0, 600, 600, *([0] * 18)), 24000, 1, clock.monotonic() + 0.1

        class AudioSource(Source):
            def __init__(self, *_args):
                pass

        class AudioFrame:
            def __init__(self, *_args):
                pass

        class LocalAudioTrack:
            @staticmethod
            def create_audio_track(*_args):
                return object()

        class RoomServiceRequest:
            def __init__(self, **kwargs):
                self.names = kwargs["names"]

        api = SimpleNamespace(
            AccessToken=AccessToken,
            VideoGrants=lambda **_kwargs: object(),
            LiveKitAPI=LiveKitAPI,
            ListRoomsRequest=RoomServiceRequest,
        )
        rtc = SimpleNamespace(
            Room=Room,
            AudioSource=AudioSource,
            AudioFrame=AudioFrame,
            LocalAudioTrack=LocalAudioTrack,
            TrackPublishOptions=lambda **_kwargs: object(),
            TrackSource=SimpleNamespace(SOURCE_MICROPHONE=1),
            ParticipantKind=SimpleNamespace(PARTICIPANT_KIND_AGENT=1),
        )

        async def tts(_http, text):
            events.append(f"tts:{text}")
            return rendered_pcm(text)

        async def transcribe(_http, pcm, _rate, _channels):
            self.assertEqual(pcm, pcm_windows(600, 600))
            return [{"start": 0.0, "end": 23.0, "text": "answer"}]

        dependencies = CaptureDependencies(
            api=api,
            rtc=rtc,
            http=object(),
            tts=tts,
            transcribe=rendered_aware_transcriber(transcribe),
            capture_factory=ContinuousCapture,
            monotonic=clock.monotonic,
            wall_time=clock.wall_time,
            sleep=clock.sleep,
        )

        async def speech_end(source):
            await source.wait_for_playout()
            return clock.monotonic()

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner.caller, "_speech_end_after_playout", speech_end),
        ):
            traces = await capture_script(
                "android-selected-room",
                ["First question@0::answer", "Second question@0::answer"],
                dependencies=dependencies,
                room_delete_deadline=1,
                poll_interval=0.1,
                room_close_after=2,
            )

        self.assertEqual(len(traces), 2)
        self.assertEqual([item["transcript"] for item in traces], ["answer", "answer"])
        for item in traces:
            self.assertAlmostEqual(
                item["speech_end_wall"],
                clock.epoch + (item["speech_end"] - 10.0),
            )
        self.assertTrue(all(item["speech_end"] < item["first_audio"] for item in traces))
        self.assertIsNone(traces[0]["room_deleted"])
        self.assertGreaterEqual(traces[1]["room_deleted"], traces[1]["speech_end"])
        self.assertEqual(traces[1]["room"], "android-selected-room")
        self.assertLess(events.index("capture-start"), events.index("speech-frame"))
        self.assertLess(events.index("playout"), events.index("capture-result"))
        self.assertEqual(events[-1], "disconnect")

    async def test_first_line_waits_for_subscription_and_fixed_settle_without_agent_metadata(self):
        dependencies = self.dependencies_for_failure("other")
        subscribed = asyncio.Event()
        wait_started = asyncio.Event()
        settle_started = asyncio.Event()
        release_settle = asyncio.Event()
        settle_durations = []
        events = []
        dependencies.rtc.Room.remote_participants = {"agent": SimpleNamespace()}

        class Publication:
            async def wait_for_subscription(self):
                wait_started.set()
                await subscribed.wait()
                events.append("subscription-ready")

        class Source:
            async def capture_frame(self, _frame):
                events.append("speech-frame")

            async def wait_for_playout(self):
                pass

        publication = Publication()
        dependencies.rtc.Room.local_participant = SimpleNamespace(
            publish_track=lambda *_args, **_kwargs: asyncio.sleep(0, result=publication)
        )
        dependencies.rtc.AudioSource = lambda *_args: Source()

        async def settle_sleep(seconds):
            settle_durations.append(seconds)
            settle_started.set()
            await release_settle.wait()

        dependencies = CaptureDependencies(**{
            **dependencies.__dict__, "sleep": settle_sleep,
        })

        async def speech_end(source):
            await source.wait_for_playout()
            return time.monotonic()

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner.caller, "_speech_end_after_playout", speech_end),
        ):
            task = asyncio.create_task(
                capture_script(
                    "android-selected-room",
                    ["First question@0::answer"],
                    dependencies=dependencies,
                    room_close_after=None,
                )
            )
            try:
                await asyncio.wait_for(wait_started.wait(), timeout=0.05)
                self.assertFalse(task.done())
                self.assertEqual(events, [])
                subscribed.set()
                await asyncio.wait_for(settle_started.wait(), timeout=0.05)
                self.assertEqual(events, ["subscription-ready"])
                self.assertEqual(settle_durations, [3.0])
                self.assertFalse(task.done())
                release_settle.set()
                await task
            finally:
                release_settle.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)

        samples = len(rendered_pcm("First question")) // 2
        self.assertEqual(events[0], "subscription-ready")
        self.assertEqual(events[1:], ["speech-frame"] * (
            (samples + runner.FRAME_SAMPLES - 1) // runner.FRAME_SAMPLES
        ))

    async def test_missing_worker_subscription_fails_within_participant_deadline(self):
        dependencies = self.dependencies_for_failure("other")

        class Publication:
            async def wait_for_subscription(self):
                await asyncio.Event().wait()

        publication = Publication()
        dependencies.rtc.Room.local_participant = SimpleNamespace(
            publish_track=lambda *_args, **_kwargs: asyncio.sleep(0, result=publication)
        )

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner, "PARTICIPANT_DEADLINE_SECONDS", 0.01),
            self.assertRaisesRegex(
                runner.DeadlineExceeded,
                "caller microphone subscription exceeded its deadline",
            ),
        ):
            await capture_script(
                "android-selected-room",
                ["First question@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )

    async def test_settle_deadline_fails_by_name_before_scripted_frames_without_agent_metadata(self):
        dependencies = self.dependencies_for_failure("other")
        dependencies.rtc.Room.remote_participants = {"agent": SimpleNamespace()}
        events = []

        class Source:
            async def capture_frame(self, _frame):
                events.append("scripted-frame")

            async def wait_for_playout(self):
                pass

        dependencies.rtc.AudioSource = lambda *_args: Source()
        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner, "PARTICIPANT_DEADLINE_SECONDS", 0.2),
            self.assertRaisesRegex(
                runner.DeadlineExceeded,
                "caller microphone subscription settle exceeded its deadline",
            ),
        ):
            await capture_script(
                "android-selected-room",
                ["First question@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )
        self.assertEqual(events, [])

    async def test_injected_endpoint_token_and_url_reach_room_connect_without_local_mint(self):
        dependencies = self.dependencies_for_failure("other")
        token = "endpoint-issued.header.signature"
        livekit_url = "wss://issued-livekit.invalid"
        connections = []

        async def connect(room, url, issued_token):
            connections.append((url, issued_token))

        with (
            patch.object(dependencies.rtc.Room, "connect", connect),
            patch.object(
                dependencies.api,
                "AccessToken",
                side_effect=AssertionError("caller must not mint a replacement token"),
            ),
            patch.dict(
                os.environ,
                {
                    "LIVEKIT_API_KEY": "key",
                    "LIVEKIT_API_SECRET": "secret",
                    "MENTAT_VOICE_TOKEN": token,
                    "LIVEKIT_URL": livekit_url,
                },
            ),
        ):
            await capture_script(
                "android-selected-room",
                ["Question@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )

        self.assertEqual(connections, [(livekit_url, token)])

    async def test_three_turn_chain_keeps_open_room_without_deletion_wait(self):
        dependencies = self.dependencies_for_failure("alice")
        steps = [
            r"Why is this garden named for Alice Keck?@0::\bAlice Keck Park\b",
            r"Okay, who was she?@0::W\. M\. Keck",
            r"Okay, what was the source of her wealth?@0::Superior Oil",
        ]
        with patch.dict(os.environ, TEST_VOICE_ENV):
            traces = await capture_script(
                "android-selected-room",
                steps,
                dependencies=dependencies,
                room_delete_deadline=0.05,
                poll_interval=0.01,
                room_close_after=None,
            )

        self.assertEqual(len(traces), 3)
        self.assertEqual([trace["turn"] for trace in traces], [1, 2, 3])
        self.assertTrue(all(trace["room_deleted"] is None for trace in traces))
        self.assertEqual(dependencies.api.LiveKitAPI.listings, 3)

    async def test_cli_parses_open_room_expectation(self):
        self.assertEqual(
            runner._parse_arguments(
                [
                    "android-selected-room",
                    "--room-close-after",
                    "none",
                    "First@0::answer",
                    "Second@0::answer",
                ]
            ),
            (
                "android-selected-room",
                ["First@0::answer", "Second@0::answer"],
                None,
            ),
        )

    async def test_playout_and_room_listing_deadlines_fail_closed(self):
        dependencies = self.dependencies_for_failure("answer")

        async def stuck_playout(_source):
            await asyncio.sleep(1)
            return time.monotonic()

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner, "REMOTE_OPERATION_DEADLINE_SECONDS", 0.01),
            patch.object(runner.caller, "_speech_end_after_playout", stuck_playout),
            self.assertRaisesRegex(RuntimeError, "speech playout completion exceeded its deadline"),
        ):
            await capture_script(
                "android-selected-room",
                ["Question@0::wrong"],
                dependencies=dependencies,
                room_close_after=1,
            )

        dependencies = self.dependencies_for_failure("room-list")
        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner, "REMOTE_OPERATION_DEADLINE_SECONDS", 0.01),
            self.assertRaisesRegex(RuntimeError, "LiveKit room listing exceeded its deadline"),
        ):
            await capture_script(
                "android-selected-room",
                ["Question@0::answer"],
                dependencies=dependencies,
                room_close_after=1,
                room_delete_deadline=0.05,
            )

    async def test_room_disconnect_ends_the_active_capture_window(self):
        dependencies = self.dependencies_for_failure("room-disconnect")
        with patch.dict(os.environ, TEST_VOICE_ENV):
            traces = await capture_script(
                "android-selected-room",
                ["Question@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )
        self.assertEqual(len(traces), 1)

    def test_pcm_onset_uses_sustained_post_playout_windows_and_preserves_overlap(self):
        post_playout = pcm_windows(0, 0, 600, 600, 600, 600)
        self.assertEqual(
            runner._first_audio_after(post_playout, 24000, 1, 10.0, 10.04),
            (10.04, False),
        )
        self.assertEqual(
            runner._first_audio_after(pcm_windows(0, 0, 200, 200), 24000, 1, 10.0, 10.04),
            (10.04, False),
        )

        overlapping = pcm_windows(0, 600, 600, 600, 600, 0)
        self.assertEqual(
            runner._first_audio_after(overlapping, 24000, 1, 10.0, 10.06),
            (10.06, True),
        )

    async def test_capture_transcribes_pcm_utterances_when_asr_timestamps_collapse(self):
        dependencies = self.dependencies_for_failure("other")
        windows = [0] * (23 * 50)
        for onset, level in ((50, 400), (500, 500), (1000, 600)):
            windows[onset : onset + 5] = [level] * 5
        pcm = pcm_windows(*windows)
        calls = []
        responses = {400: "OK.", 500: "Timer set for 5 minutes.", 600: "OK."}

        class LongCapture:
            async def start(self):
                pass

            async def result(self):
                return pcm, 24000, 1, 100.0

        async def transcribe(_http, audio, _rate, _channels):
            calls.append(audio)
            level = struct.unpack_from("<h", audio)[0]
            return [{"start": 0.0, "end": 23.0, "text": responses[level]}]

        dependencies = CaptureDependencies(
            **{
                **dependencies.__dict__,
                "capture_factory": lambda *_args: LongCapture(),
                "transcribe": rendered_aware_transcriber(transcribe),
            }
        )

        async def speech_end(_source):
            return 110.0

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner.caller, "_speech_end_after_playout", speech_end),
        ):
            traces = await capture_script(
                "android-selected-room",
                [r"Set a timer@0::timer set for 5 minutes"],
                dependencies=dependencies,
                room_close_after=None,
            )

        trace = traces[0]
        self.assertEqual(len(calls), 3)
        self.assertEqual(trace["transcript"], "OK. Timer set for 5 minutes. OK.")
        self.assertEqual(
            [(segment["start"], segment["end"]) for segment in trace["segments"]],
            [(1.0, 1.1), (10.0, 10.1), (20.0, 20.1)],
        )
        self.assertEqual([segment["start"] for segment in trace["raw_segments"]], [0.0] * 3)
        self.assertEqual(
            runner._answer_time(
                trace,
                SimpleNamespace(answer_patterns=(r"timer set for 5 minutes",), reject_patterns=()),
            ),
            110.0,
        )

    async def test_all_caller_lines_are_synthesized_before_room_connect_and_timeout_retries_once(self):
        dependencies = self.dependencies_for_failure("other")
        events = []
        attempts = {}
        original_room = dependencies.rtc.Room

        class Room(original_room):
            async def connect(self, *_args, **_kwargs):
                events.append("connect")

        async def tts(_http, text):
            events.append(f"tts:{text}")
            attempts[text] = attempts.get(text, 0) + 1
            if text == "First" and attempts[text] == 1:
                raise runner.DeadlineExceeded("scripted speech synthesis exceeded its deadline")
            return rendered_pcm(text)

        dependencies = CaptureDependencies(
            **{**dependencies.__dict__, "rtc": SimpleNamespace(**{
                **dependencies.rtc.__dict__, "Room": Room,
            }), "tts": tts}
        )
        with patch.dict(os.environ, TEST_VOICE_ENV):
            await capture_script(
                "android-selected-room",
                ["First@0::answer", "Second@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )

        self.assertEqual(attempts, {"First": 2, "Second": 1})
        self.assertLess(max(i for i, value in enumerate(events) if value.startswith("tts:")), events.index("connect"))

    async def test_truncated_render_retries_and_retains_each_attempt_privately(self):
        import json
        import tempfile

        dependencies = self.dependencies_for_failure("other")
        renders = [
            runner.caller.TTSResponse(b"\x01\x00" * 12_000, 206, 24_000),
            runner.caller.TTSResponse(b"\x02\x00" * 24_000, 200, 48_000),
        ]
        pushed = []

        class Source:
            async def capture_frame(self, frame):
                pushed.append(frame.pcm)

            async def wait_for_playout(self):
                return None

        dependencies.rtc.AudioFrame = lambda pcm, *_args: SimpleNamespace(pcm=pcm)
        dependencies.rtc.AudioSource = lambda *_args: Source()
        attempts = 0

        async def tts(*_args):
            nonlocal attempts
            response = renders[attempts]
            attempts += 1
            return response

        async def transcribe(_http, pcm, _rate, _channels):
            text = "Exact caller words" if pcm == renders[1].pcm else "Exact caller"
            return [{"start": 0.0, "end": 1.0, "text": text}]

        dependencies = CaptureDependencies(**{
            **dependencies.__dict__,
            "tts": tts,
            "transcribe": transcribe,
        })
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, TEST_VOICE_ENV):
            traces = await capture_script(
                "android-selected-room", ["Exact caller words@0::answer"],
                dependencies=dependencies, room_close_after=None,
                retain_caller_audio_dir=Path(temporary),
            )
            audio_dir = Path(temporary) / "caller-audio"
            first = "android-selected-room-turn-001-attempt-01"
            second = "android-selected-room-turn-001-attempt-02"
            first_record = json.loads((audio_dir / f"{first}.json").read_text())
            second_record = json.loads((audio_dir / f"{second}.json").read_text())
            self.assertEqual((audio_dir / f"{first}-rendered.pcm").read_bytes(), renders[0].pcm)
            self.assertEqual((audio_dir / f"{second}-rendered.pcm").read_bytes(), renders[1].pcm)
            self.assertEqual(first_record["content_check_passed"], False)
            self.assertEqual(second_record["content_check_passed"], True)
            self.assertEqual(first_record["tts_http_status"], 206)
            self.assertEqual(first_record["tts_response_bytes"], 24_000)
            self.assertEqual(second_record["tts_http_status"], 200)
            self.assertEqual(second_record["tts_response_bytes"], 48_000)
            self.assertEqual(b"".join(pushed), renders[1].pcm)
            self.assertEqual(traces[0]["tts_retry_count"], 1)
            self.assertEqual(attempts, 2)

    async def test_three_truncated_renders_fail_with_named_content_infrastructure_error(self):
        import json
        import tempfile

        dependencies = self.dependencies_for_failure("other")
        attempts = 0

        async def tts(*_args):
            nonlocal attempts
            attempts += 1
            pcm = bytes([attempts, 0]) * 12_000
            return runner.caller.TTSResponse(pcm, 200 + attempts, len(pcm))

        async def transcribe(*_args):
            return [{"start": 0.0, "end": 1.0, "text": "Exact caller"}]

        dependencies = CaptureDependencies(**{
            **dependencies.__dict__, "tts": tts, "transcribe": transcribe,
        })
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.dict(os.environ, TEST_VOICE_ENV),
                self.assertRaises(runner.PartialCaptureFailure) as caught,
            ):
                await capture_script(
                    "retry-room", ["Exact caller words@0::answer"],
                    dependencies=dependencies, room_close_after=None,
                    retain_caller_audio_dir=Path(temporary),
                )
            audio_dir = Path(temporary) / "caller-audio"
            retained = []
            for attempt in range(1, 4):
                stem = f"retry-room-turn-001-attempt-{attempt:02d}"
                record = json.loads((audio_dir / f"{stem}.json").read_text())
                pcm = (audio_dir / f"{stem}-rendered.pcm").read_bytes()
                self.assertEqual(record["attempt"], attempt)
                self.assertFalse(record["content_check_passed"])
                self.assertEqual(record["tts_http_status"], 200 + attempt)
                self.assertEqual(record["tts_response_bytes"], 24_000)
                self.assertEqual(record["pushed_frames"], 0)
                self.assertEqual(record["pushed_samples"], 0)
                retained.append(pcm)
            self.assertEqual(len(set(retained)), 3)

        self.assertEqual(attempts, 3)
        self.assertEqual(caught.exception.failure["message"],
                         "scripted speech content verification failed for line 1")
        self.assertEqual(caught.exception.failure["retry_count"], 2)

    async def test_rendered_script_content_is_transcribed_before_connect_and_checks_sms_body(self):
        line = "Text +1-202-555-0142: I will be there at six."
        pcm = bytes(runner.RATE * 2)  # Same one-second rendering in both cases.
        connections = []
        dependencies = self.dependencies_for_failure("other")
        dependencies.rtc.Room.connect = lambda *_args, **_kwargs: asyncio.sleep(
            0, result=connections.append(True)
        )
        transcripts = iter((
            "Text plus one two zero two five five five zero one four two. I will be there at.",
            "answer",
        ))

        async def transcribe(_http, _audio, _rate, _channels):
            return [{"start": 0.0, "end": 1.0, "text": next(transcripts)}]

        dependencies = CaptureDependencies(**{
            **dependencies.__dict__,
            "tts": lambda *_args: asyncio.sleep(0, result=pcm),
            "transcribe": rendered_aware_transcriber(transcribe),
        })
        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            self.assertRaises(runner.PartialCaptureFailure) as caught,
        ):
            await capture_script(
                "android-selected-room", [f"{line}@0::answer"],
                dependencies=dependencies, room_close_after=None,
            )
        self.assertEqual(caught.exception.failure, {
            "turn": 1,
            "message": "scripted speech content verification failed for line 1",
            "line": 1,
            "retry_count": 2,
        })
        self.assertTrue(runner._is_preflight_tts_capture_failure([], caught.exception.failure))
        self.assertEqual(connections, [])

        transcripts = iter((
            "Text plus 1 202 555 0142: I will be there at 6.",
            "answer",
        ))
        dependencies = CaptureDependencies(**{
            **dependencies.__dict__,
            "transcribe": rendered_aware_transcriber(transcribe),
        })
        with patch.dict(os.environ, TEST_VOICE_ENV):
            await capture_script(
                "android-selected-room", [f"{line}@0::answer"],
                dependencies=dependencies, room_close_after=None,
            )
        self.assertEqual(connections, [True])

        transcripts = iter((
            "Text plus one two zero two five five five zero one four two. "
            "I will be there at six.",
            "answer",
        ))
        with patch.dict(os.environ, TEST_VOICE_ENV):
            await capture_script(
                "android-selected-room", [f"{line}@0::answer"],
                dependencies=dependencies, room_close_after=None,
            )
        self.assertEqual(connections, [True, True])

        transcripts = iter((
            "Text plus one 202 five five five zero 1 four 2. I will be there at six.",
            "answer",
        ))
        with patch.dict(os.environ, TEST_VOICE_ENV):
            await capture_script(
                "android-selected-room", [f"{line}@0::answer"],
                dependencies=dependencies, room_close_after=None,
            )
        self.assertEqual(connections, [True, True, True])

        transcripts = iter((
            "Text plus 1 two zero two five five five zero one four two. "
            "I will be there at six.",
            "answer",
        ))
        with patch.dict(os.environ, TEST_VOICE_ENV):
            await capture_script(
                "android-selected-room", [f"{line}@0::answer"],
                dependencies=dependencies, room_close_after=None,
            )
        self.assertEqual(connections, [True, True, True, True])

    def test_rendered_content_accepts_evaluation_transcription_variants(self):
        sms_line = "Text +1-202-555-0142: I will be there at six."
        accepted = (
            (
                "Why is this garden named for Alice Keck?",
                "Why is this garden named for Alice Kek?",
            ),
            (
                "Why is this garden named for Alice Keck?",
                "Why is this garden named for Alice Keck?",
            ),
            (
                "Set a timer for five minutes.",
                "Set a timer for five-minute.",
            ),
            (
                "Set a timer for five minutes.",
                "Start a timer for five minutes.",
            ),
            (
                "Set a timer for five minutes.",
                "Set a timer for five minutes.",
            ),
            (
                "Find Alice Keck Park Memorial Garden in Santa Barbara.",
                "Find Alice Kek Park Memorial Garden in Santa Barbara.",
            ),
            (
                "Find Alice Keck Park Memorial Garden in Santa Barbara.",
                "Find Alice Keck Park Memorial Garden in Santa Barbara.",
            ),
            (
                "Navigate to Alice Keck Park Memorial Garden.",
                "Navigate to Alice Keck Park Memorial Gardens.",
            ),
            (
                "Navigate to Alice Keck Park Memorial Garden.",
                "Navigate to Alice Keck Park Memorial Garden.",
            ),
            (
                sms_line,
                "Tax plus one two zero two five five five dash oh one four two. "
                "I will be there at six.",
            ),
            (
                sms_line,
                "Plus one two zero two five five five dash oh one four two. "
                "I will be there at six.",
            ),
            (sms_line, sms_line),
            (
                sms_line,
                "Text +12025550142. I will be there at 6:00.",
            ),
        )
        for expected, transcript in accepted:
            with self.subTest(expected=expected, transcript=transcript):
                self.assertTrue(
                    runner._rendered_content_matches(
                        expected, [{"text": transcript}]
                    )
                )

        self.assertFalse(
            runner._rendered_content_matches(
                sms_line,
                [{
                    "text": (
                        "Text plus one two zero two five five five zero one four two."
                    )
                }],
            )
        )
        for expected, unrelated in (
            ("Yes.", "No."),
            ("Set a timer for five minutes.", "Set a timer for five."),
            ("Set a timer for five minutes.", "Bananas pancakes oranges grapes apples."),
            (
                "Why is this garden named for Alice Keck?",
                "Bananas pancakes oranges grapes apples.",
            ),
            (
                sms_line,
                "Text +1-202-555-0142: I will be where at six.",
            ),
        ):
            with self.subTest(expected=expected, unrelated=unrelated):
                self.assertFalse(
                    runner._rendered_content_matches(
                        expected, [{"text": unrelated}]
                    )
                )

    async def test_naturally_short_yes_and_timer_renderings_pass_content_check(self):
        for line, transcript, duration in (
            ("Yes.", "Yes.", 0.18),
            ("Set a timer for five minutes.", "Set a timer for 5 minutes.", 0.45),
            ("Set an alarm for 7 a.m.", "Set an alarm for 7 AM.", 0.45),
        ):
            with self.subTest(line=line):
                dependencies = self.dependencies_for_failure("other")
                pcm = bytes(int(duration * runner.RATE) * 2)
                transcriptions = iter((transcript, "answer"))

                async def transcribe(_http, _audio, _rate, _channels):
                    return [{"start": 0.0, "end": duration, "text": next(transcriptions)}]

                dependencies = CaptureDependencies(**{
                    **dependencies.__dict__,
                    "tts": lambda *_args: asyncio.sleep(0, result=pcm),
                    "transcribe": rendered_aware_transcriber(transcribe),
                })
                with patch.dict(os.environ, TEST_VOICE_ENV):
                    traces = await capture_script(
                        "android-selected-room", [f"{line}@0::answer"],
                        dependencies=dependencies, room_close_after=None,
                    )
                self.assertEqual(len(traces), 1)

    async def test_push_preserves_rendered_sample_count_for_full_and_partial_frames(self):
        dependencies = self.dependencies_for_failure("other")
        rendered_samples = 23_041
        rendered = bytes(rendered_samples * 2)
        RENDERED_PCM_TEXT[rendered] = "Question"
        pushed = []
        dependencies.rtc.AudioFrame = lambda pcm, rate, channels, samples: SimpleNamespace(
            pcm=pcm, sample_rate=rate, channels=channels, samples=samples
        )
        dependencies.rtc.AudioSource = lambda *_args: SimpleNamespace(
            capture_frame=lambda frame: asyncio.sleep(0, result=pushed.append(frame)),
            wait_for_playout=lambda: asyncio.sleep(0),
        )
        dependencies = CaptureDependencies(**{
            **dependencies.__dict__,
            "tts": lambda *_args: asyncio.sleep(0, result=rendered),
        })
        with patch.dict(os.environ, TEST_VOICE_ENV):
            await capture_script(
                "android-selected-room", ["Question@0::answer"],
                dependencies=dependencies, room_close_after=None,
            )
        self.assertEqual(b"".join(frame.pcm for frame in pushed), rendered)
        self.assertEqual(sum(frame.samples for frame in pushed), rendered_samples)
        self.assertEqual(
            [frame.samples for frame in pushed],
            [runner.FRAME_SAMPLES] * (rendered_samples // runner.FRAME_SAMPLES)
            + [rendered_samples % runner.FRAME_SAMPLES],
        )

    async def test_odd_render_sample_mismatch_fails_with_line_name(self):
        dependencies = self.dependencies_for_failure("other")
        dependencies = CaptureDependencies(**{
            **dependencies.__dict__,
            "tts": lambda *_args: asyncio.sleep(0, result=bytes([0])),
        })
        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            self.assertRaises(runner.PartialCaptureFailure) as caught,
        ):
            await capture_script(
                "android-selected-room", ["Question@0::answer"],
                dependencies=dependencies, room_close_after=None,
            )
        self.assertEqual(caught.exception.failure, {
            "turn": 1,
            "message": "scripted speech sample count mismatch for line 1",
            "line": 1,
            "retry_count": 0,
        })

    async def test_preflight_tts_failures_on_every_line_use_strict_turn_one_envelopes(self):
        import io
        import json
        from contextlib import redirect_stdout

        from evals.report import score_observations

        lines = [f"Line {index}@0::answer" for index in range(1, 4)]
        for failed_line in range(1, 4):
            with self.subTest(failed_line=failed_line):
                dependencies = self.dependencies_for_failure("other")
                attempts = {}

                async def tts(_http, text):
                    line_number = int(text.split()[1])
                    attempts[line_number] = attempts.get(line_number, 0) + 1
                    if line_number == failed_line:
                        raise runner.DeadlineExceeded(
                            "scripted speech synthesis exceeded its deadline"
                        )
                    return rendered_pcm(text)

                dependencies = CaptureDependencies(
                    **{**dependencies.__dict__, "tts": tts}
                )

                async def remote_capture(room, steps, *, room_close_after):
                    return await capture_script(
                        room,
                        steps,
                        dependencies=dependencies,
                        room_close_after=room_close_after,
                    )

                output = io.StringIO()
                with (
                    patch.dict(os.environ, TEST_VOICE_ENV),
                    patch.object(runner, "run_remote_capture", remote_capture),
                    redirect_stdout(output),
                ):
                    exit_code = await asyncio.to_thread(
                        runner.main, ["android-selected-room", *lines]
                    )

                self.assertEqual(exit_code, 1)
                captured_stdout = output.getvalue()
                envelope = json.loads(captured_stdout)
                self.assertEqual(envelope["turns"], [])
                self.assertEqual(envelope["failure"], {
                    "turn": 1,
                    "message": (
                        f"scripted speech synthesis for line {failed_line} "
                        "exceeded its deadline"
                    ),
                    "line": failed_line,
                    "retry_count": 1,
                })
                parsed = runner._capture_envelope(captured_stdout, len(lines))
                self.assertEqual(parsed, ([], envelope["failure"]))
                self.assertEqual(attempts, {
                    line: (2 if line == failed_line else 1)
                    for line in range(1, failed_line + 1)
                })
                report = score_observations({
                    "cases": [{"name": "preflight", "runs": [{
                        "turns": parsed[0],
                        "failure": parsed[1],
                        "product_failures": [],
                    }]}],
                }, required_runs=1)
                self.assertFalse(report["passed"])
                self.assertEqual(report["cases"][0]["capture_failures"], [{
                    "run": 1,
                    "turn": 1,
                    "message": envelope["failure"]["message"],
                    "line": failed_line,
                    "retry_count": 1,
                }])

    async def test_exhausted_preflight_tts_timeout_is_partial_failure_before_join(self):
        dependencies = self.dependencies_for_failure("other")
        connects = []
        original_room = dependencies.rtc.Room

        class Room(original_room):
            async def connect(self, *_args, **_kwargs):
                connects.append(True)

        async def tts(*_args):
            raise runner.DeadlineExceeded("scripted speech synthesis exceeded its deadline")

        dependencies = CaptureDependencies(
            **{**dependencies.__dict__, "rtc": SimpleNamespace(**{
                **dependencies.rtc.__dict__, "Room": Room,
            }), "tts": tts}
        )
        with patch.dict(os.environ, TEST_VOICE_ENV):
            with self.assertRaises(runner.PartialCaptureFailure) as caught:
                await capture_script(
                    "android-selected-room",
                    ["Question@0::answer"],
                    dependencies=dependencies,
                    room_close_after=None,
                )

        self.assertEqual(caught.exception.failure, {
            "turn": 1,
            "message": "scripted speech synthesis for line 1 exceeded its deadline",
            "line": 1,
            "retry_count": 1,
        })
        self.assertEqual(connects, [])

    async def test_concurrent_utterance_transcriptions_preserve_capture_order(self):
        dependencies = self.dependencies_for_failure("other")
        levels = [0] * 55
        for onset, level in zip((0, 20, 40), (600, 700, 800), strict=True):
            levels[onset : onset + 5] = [level] * 5
        pcm = pcm_windows(*levels)

        class LongCapture:
            async def start(self):
                pass

            async def result(self):
                return pcm, 24000, 1, time.monotonic() + 0.1

        utterance_indexes = {
            utterance_pcm: index
            for index, (_start, _end, utterance_pcm) in enumerate(
                runner._pcm_utterances(pcm, 24000, 1)
            )
        }
        delays = {0: 0.04, 1: 0.001, 2: 0.02}
        async def transcribe(_http, utterance_pcm, _rate, _channels):
            index = utterance_indexes[utterance_pcm]
            await asyncio.sleep(delays[index])
            return [{"start": 0.0, "end": 0.2, "text": f"part-{index}"}]

        dependencies = CaptureDependencies(
            **{**dependencies.__dict__, "capture_factory": lambda *_args: LongCapture(), "transcribe": rendered_aware_transcriber(transcribe)}
        )
        async def speech_end(_source):
            return time.monotonic()

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner.caller, "_speech_end_after_playout", speech_end),
        ):
            traces = await capture_script(
                "android-selected-room", ["Question@0::answer"],
                dependencies=dependencies, room_close_after=None,
            )

        self.assertEqual(traces[0]["transcript"], "part-0 part-1 part-2")

    async def test_whisper_4xx_preserves_completed_turns_and_allows_later_run(self):
        import json

        dependencies = self.dependencies_for_failure("other")
        levels = [0] * 65
        for onset, level in zip((0, 20, 40), (600, 700, 800), strict=True):
            levels[onset : onset + 5] = [level] * 5
        pcm = pcm_windows(*levels)
        utterance_indexes = {
            utterance_pcm: index
            for index, (_start, _end, utterance_pcm) in enumerate(
                runner._pcm_utterances(pcm, 24000, 1)
            )
        }
        self.assertEqual(len(utterance_indexes), 3)
        fail = True
        attempts = 0

        class Capture:
            async def start(self):
                pass

            async def result(self):
                return pcm, 24000, 1, time.monotonic() + 0.1

        async def transcribe(_http, utterance_pcm, _rate, _channels):
            nonlocal attempts
            attempts += 1
            if fail and attempts == 5 and utterance_indexes[utterance_pcm] == 1:
                raise runner.caller.WhisperTranscriptionError(429)
            return [{"start": 0.0, "end": 0.2, "text": "completed"}]

        dependencies = CaptureDependencies(**{
            **dependencies.__dict__,
            "capture_factory": lambda *_args: Capture(),
            "transcribe": rendered_aware_transcriber(transcribe),
        })
        speech_end = lambda *_args: asyncio.sleep(0, result=time.monotonic() - 1)
        with patch.dict(os.environ, TEST_VOICE_ENV), patch.object(
            runner.caller, "_speech_end_after_playout", speech_end
        ):
            with self.assertRaises(runner.PartialCaptureFailure) as caught:
                await capture_script(
                    "android-selected-room",
                    ["First question@0::answer", "Rejected question@0::answer"],
                    dependencies=dependencies,
                    room_close_after=None,
                )

            self.assertEqual(len(caught.exception.turns), 1)
            self.assertEqual(
                caught.exception.turns[0]["transcript"],
                "completed completed completed",
            )
            self.assertEqual(caught.exception.failure["turn"], 2)
            self.assertEqual(
                caught.exception.failure["message"],
                "Whisper transcription rejected utterance 2 (HTTP 429)",
            )
            envelope = json.dumps({
                "turns": caught.exception.turns,
                "failure": caught.exception.failure,
            })
            self.assertEqual(
                runner._capture_envelope(envelope, 2),
                (caught.exception.turns, caught.exception.failure),
            )

            fail = False
            traces = await capture_script(
                "android-selected-room",
                ["First question@0::answer", "Later question@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )
            self.assertEqual(len(traces), 2)

    async def test_non_4xx_transcription_errors_still_surface(self):
        dependencies = self.dependencies_for_failure("other")
        pcm = pcm_windows(0, 0, 600, 600, *([0] * 18))

        class Capture:
            async def start(self):
                pass

            async def result(self):
                return pcm, 24000, 1, time.monotonic() + 0.1

        async def transcribe(*_args):
            raise RuntimeError("Whisper HTTP 500")

        dependencies = CaptureDependencies(**{
            **dependencies.__dict__,
            "capture_factory": lambda *_args: Capture(),
            "transcribe": rendered_aware_transcriber(transcribe),
        })
        with patch.dict(os.environ, TEST_VOICE_ENV), patch.object(
            runner.caller,
            "_speech_end_after_playout",
            lambda *_args: asyncio.sleep(0, result=time.monotonic() - 1),
        ):
            with self.assertRaisesRegex(RuntimeError, "Whisper HTTP 500"):
                await capture_script(
                    "android-selected-room",
                    ["Question@0::answer"],
                    dependencies=dependencies,
                    room_close_after=None,
                )

    async def test_asr_deadline_is_named_partial_failure_with_turn_start_and_prefix(self):
        import json
        import tempfile
        import wave

        dependencies = self.dependencies_for_failure("other")
        levels = [0] * 55
        for onset, level in zip((0, 20, 40), (600, 700, 800), strict=True):
            levels[onset : onset + 5] = [level] * 5
        pcm = pcm_windows(*levels)

        class LongCapture:
            async def start(self):
                pass

            async def result(self):
                return pcm, 24000, 1, time.monotonic() + 0.1

        utterance_indexes = {
            utterance_pcm: index
            for index, (_start, _end, utterance_pcm) in enumerate(
                runner._pcm_utterances(pcm, 24000, 1)
            )
        }
        async def transcribe(_http, utterance_pcm, _rate, _channels):
            if utterance_indexes[utterance_pcm] == 1:
                await asyncio.sleep(1)
            else:
                await asyncio.sleep(0)
            return [{"start": 0.0, "end": 0.2, "text": "completed"}]

        dependencies = CaptureDependencies(
            **{**dependencies.__dict__, "capture_factory": lambda *_args: LongCapture(), "transcribe": rendered_aware_transcriber(transcribe)}
        )
        async def speech_end(_source):
            return time.monotonic()

        with tempfile.TemporaryDirectory() as temporary:
            evidence_dir = Path(temporary) / "retained-evidence"
            with (
                patch.dict(os.environ, TEST_VOICE_ENV),
                patch.object(runner, "ANSWER_TRANSCRIPTION_DEADLINE_SECONDS", 0.1),
                patch.object(runner.caller, "_speech_end_after_playout", speech_end),
            ):
                with self.assertRaises(runner.PartialCaptureFailure) as caught:
                    await capture_script(
                        "android-selected-room", ["Question@0::answer"],
                        dependencies=dependencies,
                        room_close_after=None,
                        retain_sms_audio_dir=evidence_dir,
                        retain_sms_audio_scenario="sms-say-back-yes",
                    )

            audio_dir = evidence_dir / "sms-audio"
            audio_path = audio_dir / "android-selected-room-turn-001.wav"
            with wave.open(str(audio_path), "rb") as retained:
                self.assertEqual(retained.getnchannels(), 1)
                self.assertEqual(retained.getsampwidth(), 2)
                self.assertEqual(retained.getframerate(), 24000)
                self.assertEqual(retained.readframes(retained.getnframes()), pcm)
            metadata = json.loads((audio_dir / "transcripts.jsonl").read_text())
            self.assertEqual(metadata["filename"], audio_path.name)
            self.assertEqual(metadata["transcript"], "completed")

        self.assertEqual(caught.exception.failure["turn"], 1)
        self.assertEqual(caught.exception.failure["message"], "answer transcription exceeded its deadline")
        self.assertIsInstance(caught.exception.failure["speech_started_at"], float)
        self.assertEqual(caught.exception.failure["segments"], [{"start": 0.0, "end": 0.2, "text": "completed"}])
        self.assertEqual(
            runner._capture_envelope(json.dumps({
                "turns": caught.exception.turns,
                "failure": caught.exception.failure,
            }), 1),
            ([], caught.exception.failure),
        )

    async def test_utterance_transcriptions_share_one_per_turn_deadline(self):
        dependencies = self.dependencies_for_failure("other")
        windows = [0] * 55
        for onset in (0, 20, 40):
            windows[onset : onset + 5] = [600] * 5
        pcm = pcm_windows(*windows)
        calls = []

        class LongCapture:
            async def start(self):
                pass

            async def result(self):
                return pcm, 24000, 1, time.monotonic() + 0.1

        async def transcribe(*_args):
            calls.append(1)
            await asyncio.sleep(0.1)
            return [{"start": 0.0, "end": 0.2, "text": "answer"}]

        dependencies = CaptureDependencies(
            **{
                **dependencies.__dict__,
                "capture_factory": lambda *_args: LongCapture(),
                "transcribe": rendered_aware_transcriber(transcribe),
            }
        )

        async def speech_end(_source):
            return time.monotonic()

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner, "ANSWER_TRANSCRIPTION_DEADLINE_SECONDS", 0.2),
            patch.object(runner.caller, "_speech_end_after_playout", speech_end),
        ):
            traces = await capture_script(
                "android-selected-room",
                ["Question@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(traces[0]["transcript"], "answer answer answer")

    def test_pcm_utterances_require_two_onset_windows_and_300ms_quiet(self):
        levels = [200, 200, *([0] * 14), 600, 600, *([0] * 15)]
        levels.extend([0] * 7)
        levels.extend([700, 700, *([0] * 15)])
        utterances = runner._pcm_utterances(pcm_windows(*levels), 24000, 1)

        self.assertEqual([(start, end) for start, end, _pcm in utterances], [(0.0, 0.36), (0.8, 0.84)])
        self.assertEqual(utterances[0][2], pcm_windows(*levels[:18]))
        with self.assertRaisesRegex(RuntimeError, "no-answer"):
            runner._pcm_utterances(bytes([0]), 24000, 1)

    def test_pcm_onset_ignores_a_partial_trailing_window(self):
        trailing_half_window = struct.pack("<240h", *([0] * 240))
        pcm = pcm_windows(0, 0, 600, 600) + trailing_half_window
        self.assertEqual(
            runner._first_audio_after(pcm, 24000, 1, 10.0, 10.04),
            (10.04, False),
        )

    def test_pcm_onset_rejects_short_noise_silence_and_malformed_metadata_as_no_answer(self):
        cases = (
            (pcm_windows(0, 0, 600, 0, 0), 24000, 1),
            (pcm_windows(0, 0, 199, 199), 24000, 1),
            (pcm_windows(0, 0, 0, 0), 24000, 1),
            (bytes([0]), 24000, 1),
            (pcm_windows(0, 600), 0, 1),
            (pcm_windows(0, 600), 24000, 0),
            (pcm_windows(0, 600), 24000, 2),
            (pcm_windows(600, 600, channels=2) + bytes([0, 0]), 24000, 2),
        )
        for pcm, sample_rate, channels in cases:
            with self.subTest(sample_rate=sample_rate, channels=channels, pcm=pcm[:8]):
                with self.assertRaisesRegex(RuntimeError, "no-answer"):
                    runner._first_audio_after(pcm, sample_rate, channels, 10.0, 10.0)

    async def test_capture_records_early_audio_overlap_and_full_trace(self):
        dependencies = self.dependencies_for_failure("other")

        class EarlyCapture:
            def __init__(self, _queue, _ended=None):
                pass

            async def start(self):
                pass

            async def result(self):
                return pcm_windows(600, 600, *([0] * 58), 600, 600), 24000, 1, 99.0

        transcriptions = iter(("Okay, I heard you.", "The timer is set for five minutes."))

        async def transcribe(*_args):
            return [{"start": 0.0, "end": 23.0, "text": next(transcriptions)}]

        dependencies = CaptureDependencies(
            **{
                **dependencies.__dict__,
                "capture_factory": EarlyCapture,
                "transcribe": rendered_aware_transcriber(transcribe),
            }
        )

        async def speech_end(_source):
            return 100.0

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner.caller, "_speech_end_after_playout", speech_end),
        ):
            traces = await capture_script(
                "android-selected-room",
                [r"Set a timer@0::timer is set"],
                dependencies=dependencies,
                room_close_after=None,
            )

        self.assertEqual(len(traces), 1)
        trace = traces[0]
        self.assertEqual(
            trace["transcript"],
            "Okay, I heard you. The timer is set for five minutes.",
        )
        self.assertEqual([segment["start"] for segment in trace["segments"]], [0.0, 1.2])
        self.assertEqual(trace["segments"][1]["start"], 1.2)
        self.assertEqual(trace["raw_segments"][0]["start"], 0.0)
        self.assertEqual(trace["first_audio"], 100.2)
        self.assertEqual(trace["speech_end"], 100.0)
        self.assertEqual(trace["raw_segments"][1]["end"], 23.0)
        self.assertTrue(trace["overlap"])
        from evals.report import score_observations

        scored = score_observations({"cases": [{
            "name": "rounded whisper end",
            "runs": [{"turns": [{
                "kind": "search",
                "speech_end": trace["speech_end"],
                "first_audio": trace["first_audio"],
                "answer_at": trace["first_audio"],
                "capture_started": trace["capture_started"],
                "segments": trace["segments"],
                "command_received_at": None,
                "overlap": trace["overlap"],
                "expect_confirmation": False,
                "confirmation": None,
                "expect_hangup": False,
                "room_deleted": None,
                "model_calls": [],
                "raw_segments": trace["raw_segments"],
            }]}],
        }]}, required_runs=1)
        self.assertEqual(len(scored["cases"][0]["turns"]), 1)
        self.assertEqual(scored["cases"][0]["turns"][0]["segments"][1]["end"], 23.0)

    async def test_raw_asr_timestamps_do_not_define_pcm_transcript_timing(self):
        dependencies = self.dependencies_for_failure("other")
        raw_segments = [
            {"start": 0.0, "end": 23.0, "text": "I am still listening."},
        ]

        async def transcribe(*_args):
            return raw_segments

        dependencies = CaptureDependencies(
            **{**dependencies.__dict__, "transcribe": rendered_aware_transcriber(transcribe)}
        )
        with patch.dict(os.environ, TEST_VOICE_ENV):
            traces = await capture_script(
                "android-selected-room",
                ["Question@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )

        trace = traces[0]
        self.assertEqual(trace["transcript"], "I am still listening.")
        self.assertEqual(trace["segments"], [{"start": 0.1, "end": 0.14, "text": "I am still listening."}])
        self.assertEqual(trace["raw_segments"], raw_segments)
        self.assertIsNone(
            runner._answer_time(
                trace,
                SimpleNamespace(answer_patterns=(r"Should I send it",)),
            )
        )
        self.assertIsNone(runner._confirmation_time(trace))
        from evals.scenarios import evaluate_scenario_failures

        failures = evaluate_scenario_failures(
            SCENARIOS[0],
            [trace["transcript"]],
            [{"turn": 1, "kind": "timer", "seconds": 300}],
            1,
        )
        self.assertTrue(any("missing answer pattern" in failure.message for failure in failures))

    async def test_all_silent_transcript_segments_keep_named_no_answer_and_raw_evidence(self):
        dependencies = self.dependencies_for_failure("other")
        raw_segments = [
            {"start": 0.0, "end": 0.08, "text": "  "},
        ]

        async def transcribe(*_args):
            return raw_segments

        dependencies = CaptureDependencies(
            **{**dependencies.__dict__, "transcribe": rendered_aware_transcriber(transcribe)}
        )
        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            self.assertRaises(runner.PartialCaptureFailure) as caught,
        ):
            await capture_script(
                "android-selected-room",
                ["Question@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )

        self.assertEqual(caught.exception.failure["message"], runner.NO_ANSWER_FAILURE)
        self.assertEqual(caught.exception.failure["segments"], raw_segments)

    async def test_capture_silence_and_malformed_pcm_emit_named_no_answer_failures(self):
        for pcm, sample_rate in (
            (pcm_windows(0, 0, 0, 0), 24000),
            (pcm_windows(0, 600), 0),
        ):
            with self.subTest(sample_rate=sample_rate):
                dependencies = self.dependencies_for_failure("other")

                class Capture:
                    async def start(self):
                        pass

                    async def result(self):
                        return pcm, sample_rate, 1, 99.0

                dependencies = CaptureDependencies(
                    **{
                        **dependencies.__dict__,
                        "capture_factory": lambda *_args: Capture(),
                    }
                )

                async def speech_end(_source):
                    return 100.0

                with (
                    patch.dict(os.environ, TEST_VOICE_ENV),
                    patch.object(runner.caller, "_speech_end_after_playout", speech_end),
                    self.assertRaises(runner.PartialCaptureFailure) as caught,
                ):
                    await capture_script(
                        "android-selected-room",
                        ["Question@0::.*"],
                        dependencies=dependencies,
                        room_close_after=None,
                    )
                self.assertEqual(caught.exception.failure["message"], runner.NO_ANSWER_FAILURE)
                self.assertEqual(caught.exception.failure["turn"], 1)
                self.assertEqual(caught.exception.turns, [])

    async def test_capture_fails_when_no_agent_audio_follows_speech_end(self):
        dependencies = self.dependencies_for_failure("other")

        class EarlyOnlyCapture:
            def __init__(self, _queue, _ended=None):
                pass

            async def start(self):
                pass

            async def result(self):
                return pcm_windows(600, 600, 0, 0, 0, 0), 24000, 1, 99.0

        async def transcribe(*_args):
            return [{"start": 0.2, "end": 0.4, "text": "Okay, I heard you."}]

        dependencies = CaptureDependencies(
            **{
                **dependencies.__dict__,
                "capture_factory": EarlyOnlyCapture,
                "transcribe": rendered_aware_transcriber(transcribe),
            }
        )

        async def speech_end(_source):
            return 100.0

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner.caller, "_speech_end_after_playout", speech_end),
            self.assertRaisesRegex(RuntimeError, "no-answer"),
        ):
            await capture_script(
                "android-selected-room",
                [r"Question@0::.*"],
                dependencies=dependencies,
                room_close_after=None,
            )

    async def test_wrong_answer_is_rejected_only_after_complete_capture(self):
        from evals.scenarios import evaluate_scenario

        dependencies = self.dependencies_for_failure("answer")
        with patch.dict(os.environ, TEST_VOICE_ENV):
            traces = await capture_script(
                "android-selected-room",
                [r"Set a timer@0::five minutes"],
                dependencies=dependencies,
                room_close_after=None,
            )

        self.assertEqual(traces[0]["transcript"], "wrong response")
        self.assertEqual(len(traces[0]["segments"]), 1)
        with self.assertRaisesRegex(AssertionError, "missing answer pattern"):
            evaluate_scenario(SCENARIOS[0], [traces[0]["transcript"]], [], None)

    async def test_partial_capture_preserves_turn_one_and_names_turn_two_failure(self):
        cases = ("early deletion", "no post-playout audio", "tts timeout")
        for failure in cases:
            with self.subTest(failure=failure):
                dependencies = self.dependencies_for_failure("other")
                if failure == "early deletion":
                    calls = 0
                    original = dependencies.api.LiveKitAPI.list_rooms

                    async def list_rooms(api_client, request):
                        nonlocal calls
                        calls += 1
                        if calls == 1:
                            return SimpleNamespace(rooms=[])
                        return await original(api_client, request)

                    dependencies.api.LiveKitAPI.list_rooms = list_rooms
                elif failure == "no post-playout audio":
                    speech_ends = iter((0.0, 1000.0))

                    async def transcribe(*_args):
                        return [{"start": 0.1, "end": 0.5, "text": "answer"}]

                    class Capture:
                        starts = iter((10.0, 20.0))

                        def __init__(self, _queue, _ended=None):
                            pass

                        async def start(self):
                            pass

                        async def result(self):
                            return pcm_windows(0, 0, 0, 0, 0, 600, 600, *([0] * 18)), 24000, 1, next(self.starts)

                    dependencies = CaptureDependencies(
                        **{
                            **dependencies.__dict__,
                            "transcribe": rendered_aware_transcriber(transcribe),
                            "capture_factory": Capture,
                        }
                    )

                    async def speech_end(_source):
                        return next(speech_ends)

                    with patch.object(
                        runner.caller, "_speech_end_after_playout", speech_end
                    ):
                        await self.assert_partial_failure(dependencies)
                    continue
                else:
                    calls = 0

                    async def tts(_http, text):
                        nonlocal calls
                        calls += 1
                        if calls >= 2:
                            await asyncio.sleep(1)
                        return rendered_pcm(text)

                    dependencies = CaptureDependencies(
                        **{**dependencies.__dict__, "tts": tts}
                    )

                with (
                    patch.dict(os.environ, TEST_VOICE_ENV),
                    patch.object(runner, "REMOTE_OPERATION_DEADLINE_SECONDS", 0.01),
                ):
                    if failure == "tts timeout":
                        with self.assertRaises(runner.PartialCaptureFailure) as caught:
                            await capture_script(
                                "android-selected-room",
                                ["First@0::answer", "Second@0::answer"],
                                dependencies=dependencies,
                                room_close_after=None,
                            )
                        self.assertEqual(caught.exception.failure["turn"], 1)
                        self.assertEqual(
                            caught.exception.failure["message"],
                            "scripted speech synthesis for line 2 exceeded its deadline",
                        )
                        self.assertEqual(caught.exception.turns, [])
                    else:
                        await self.assert_partial_failure(dependencies)

    async def test_first_turn_tts_and_no_audio_failures_keep_named_zero_turn_envelopes(self):
        dependencies = self.dependencies_for_failure("other")

        async def stuck_tts(_http, _text):
            await asyncio.sleep(1)

        dependencies = CaptureDependencies(**{**dependencies.__dict__, "tts": stuck_tts})
        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner, "REMOTE_OPERATION_DEADLINE_SECONDS", 0.01),
            self.assertRaises(runner.PartialCaptureFailure) as caught,
        ):
            await capture_script(
                "android-selected-room",
                ["First@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )
        self.assertEqual(caught.exception.failure["turn"], 1)
        self.assertEqual(caught.exception.turns, [])

        dependencies = self.dependencies_for_failure("other")

        class Capture:
            async def start(self):
                pass

            async def result(self):
                return pcm_windows(0, 0, 0, 0, 0, 600, 600, *([0] * 18)), 24000, 1, time.monotonic() + 0.1

        async def future_speech_end(_source):
            return time.monotonic() + 1000.0

        dependencies = CaptureDependencies(
            **{
                **dependencies.__dict__,
                "capture_factory": lambda *_args: Capture(),
            }
        )
        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner.caller, "_speech_end_after_playout", future_speech_end),
            self.assertRaises(runner.PartialCaptureFailure) as caught,
        ):
            await capture_script(
                "android-selected-room",
                ["First@0::answer"],
                dependencies=dependencies,
                room_close_after=None,
            )
        self.assertEqual(caught.exception.failure["turn"], 1)
        self.assertEqual(caught.exception.turns, [])
        self.assertEqual(
            caught.exception.failure["message"],
            runner.NO_ANSWER_FAILURE,
        )

    async def assert_partial_failure(self, dependencies):
        with patch.dict(os.environ, TEST_VOICE_ENV):
            with self.assertRaises(runner.PartialCaptureFailure) as caught:
                await capture_script(
                    "android-selected-room",
                    ["First@0::answer", "Second@0::answer"],
                    dependencies=dependencies,
                    room_close_after=None,
                )
        self.assertEqual(caught.exception.failure["turn"], 2)
        self.assertTrue(caught.exception.failure["message"])
        self.assertEqual(len(caught.exception.turns), 1)
        self.assertEqual(caught.exception.turns[0]["turn"], 1)
        self.assertEqual(caught.exception.turns[0]["transcript"], "answer")

    async def test_expected_hangup_timeout_preserves_the_completed_same_turn(self):
        dependencies = self.dependencies_for_failure("other")

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            self.assertRaises(runner.PartialCaptureFailure) as caught,
        ):
            await capture_script(
                "android-selected-room",
                ["Set a timer for five minutes.@0::five minutes"],
                dependencies=dependencies,
                room_delete_deadline=0.02,
                poll_interval=0.005,
                room_close_after=1,
            )

        self.assertEqual(caught.exception.failure, {
            "turn": 1,
            "message": "room deletion was not observed before deadline",
        })
        self.assertEqual(len(caught.exception.turns), 1)
        trace = caught.exception.turns[0]
        self.assertEqual(trace["turn"], 1)
        self.assertEqual(trace["transcript"], "answer")
        self.assertGreater(trace["first_audio"], trace["speech_end"])
        self.assertIsNone(trace["room_deleted"])

    async def test_missing_audio_transcript_and_room_deletion_fail_closed(self):
        for failure, expected in (
            ("audio", "no frames"),
            ("capture", "exceeded its deadline"),
            ("transcript", "no-answer"),
            ("room", "room deletion was not observed"),
        ):
            with self.subTest(failure=failure):
                dependencies = self.dependencies_for_failure(failure)
                with self.assertRaisesRegex(RuntimeError, expected):
                    with (
                        patch.dict(os.environ, TEST_VOICE_ENV),
                        patch.object(runner, "ANSWER_CAPTURE_DEADLINE_SECONDS", 0.01),
                    ):
                        await capture_script(
                            "android-selected-room",
                            ["Question@0::answer"],
                            dependencies=dependencies,
                            room_delete_deadline=0.05,
                            poll_interval=0.01,
                            room_close_after=1,
                        )

    def test_missing_or_invalid_segment_timestamps_fail_closed(self):
        for segments in (
            [{"start": 0.1, "text": "answer"}],
            [{"start": float("nan"), "end": 0.2, "text": "answer"}],
            [{"start": 0.2, "end": 0.1, "text": "answer"}],
        ):
            with self.subTest(segments=segments), self.assertRaises(RuntimeError):
                runner._segment_start(segments)

    def test_sms_audio_retention_does_not_chmod_the_shared_parent(self):
        import stat
        import tempfile
        import wave

        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "retained-evidence"
            evidence.mkdir(mode=0o700)
            pcm = pcm_windows(600, 600)
            chmod = os.chmod

            def deny_parent_chmod(path, mode, *args, **kwargs):
                if Path(path) == evidence:
                    raise PermissionError("shared parent is owned by another writer")
                chmod(path, mode, *args, **kwargs)

            with patch.object(runner.os, "chmod", side_effect=deny_parent_chmod):
                runner._retain_sms_audio(
                    evidence,
                    "sms-say-back-yes",
                    "private-audio-room",
                    1,
                    "synthetic answer",
                    pcm,
                    24000,
                    1,
                )

            audio_dir = evidence / "sms-audio"
            audio_path = audio_dir / "private-audio-room-turn-001.wav"
            self.assertEqual(stat.S_IMODE(evidence.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(audio_dir.stat().st_mode), 0o700)
            with wave.open(str(audio_path), "rb") as retained:
                self.assertEqual(retained.readframes(retained.getnframes()), pcm)

    async def test_sms_audio_retention_writes_complete_wav_and_private_transcript_metadata(self):
        import json
        import stat
        import tempfile
        import wave

        dependencies = self.dependencies_for_failure("other")
        pcm = pcm_windows(0, 0, 600, 600, 0, 0, 700, 700, sample_rate=48000, channels=2)

        class StereoCapture:
            async def start(self):
                pass

            async def result(self):
                return pcm, 48000, 2, time.monotonic() + 0.1

        dependencies = CaptureDependencies(
            **{**dependencies.__dict__, "capture_factory": lambda *_args: StereoCapture()}
        )

        with tempfile.TemporaryDirectory() as temporary:
            for scenario, turn_count in (
                ("sms-say-back-yes", 2),
                ("sms-correction-new-yes", 3),
            ):
                with self.subTest(scenario=scenario):
                    evidence = Path(temporary) / scenario
                    steps = [f"SMS line {turn}@0::answer" for turn in range(1, turn_count + 1)]
                    with (
                        patch.dict(os.environ, TEST_VOICE_ENV),
                        patch.object(
                            runner.caller,
                            "_speech_end_after_playout",
                            lambda *_args: asyncio.sleep(0, result=time.monotonic() - 1),
                        ),
                    ):
                        traces = await capture_script(
                            "android-selected-room",
                            steps,
                            dependencies=dependencies,
                            room_close_after=None,
                            retain_sms_audio_dir=evidence,
                            retain_sms_audio_scenario=scenario,
                        )

                    audio_dir = evidence / "sms-audio"
                    metadata = audio_dir / "transcripts.jsonl"
                    self.assertEqual(stat.S_IMODE(evidence.stat().st_mode), 0o700)
                    self.assertEqual(stat.S_IMODE(audio_dir.stat().st_mode), 0o700)
                    self.assertEqual(stat.S_IMODE(metadata.stat().st_mode), 0o600)
                    records = [json.loads(line) for line in metadata.read_text().splitlines()]
                    self.assertEqual(len(records), turn_count)
                    for turn, (trace, record) in enumerate(zip(traces, records, strict=True), 1):
                        audio = audio_dir / record["filename"]
                        self.assertEqual(stat.S_IMODE(audio.stat().st_mode), 0o600)
                        with wave.open(str(audio), "rb") as retained:
                            self.assertEqual(retained.getnchannels(), 2)
                            self.assertEqual(retained.getsampwidth(), 2)
                            self.assertEqual(retained.getframerate(), 48000)
                            self.assertEqual(retained.readframes(retained.getnframes()), pcm)
                        self.assertEqual(record, {
                            "scenario": scenario,
                            "room": "android-selected-room",
                            "turn": turn,
                            "transcript": trace["transcript"],
                            "filename": audio.name,
                        })
                        self.assertNotIn("audio_filename", trace)

    async def test_sms_audio_survives_transcription_errors_and_empty_or_invalid_results(self):
        import json
        import tempfile
        import wave

        pcm = pcm_windows(0, 0, 600, 600, *([0] * 18))
        for failure in ("exception", "empty", "invalid"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                dependencies = self.dependencies_for_failure("other")

                async def transcribe(*_args):
                    if failure == "exception":
                        raise RuntimeError("transcription unavailable")
                    if failure == "empty":
                        return []
                    return [{"start": 0.1, "end": 0.5}]

                class Capture:
                    async def start(self):
                        pass

                    async def result(self):
                        return pcm, 24000, 1, time.monotonic() + 0.1

                dependencies = CaptureDependencies(
                    **{
                        **dependencies.__dict__,
                        "capture_factory": lambda *_args: Capture(),
                        "transcribe": rendered_aware_transcriber(transcribe),
                    }
                )
                evidence_dir = Path(temporary) / "retained-evidence"
                with (
                    patch.dict(os.environ, TEST_VOICE_ENV),
                    patch.object(
                        runner.caller,
                        "_speech_end_after_playout",
                        lambda *_args: asyncio.sleep(0, result=time.monotonic() - 1),
                    ),
                ):
                    with self.assertRaises((RuntimeError, runner.PartialCaptureFailure)):
                        await capture_script(
                            "android-selected-room",
                            ["Question@0::answer"],
                            dependencies=dependencies,
                            room_close_after=None,
                            retain_sms_audio_dir=evidence_dir,
                            retain_sms_audio_scenario="sms-say-back-yes",
                        )

                audio_dir = evidence_dir / "sms-audio"
                audio_path = audio_dir / "android-selected-room-turn-001.wav"
                with wave.open(str(audio_path), "rb") as retained:
                    self.assertEqual(retained.readframes(retained.getnframes()), pcm)
                metadata = json.loads((audio_dir / "transcripts.jsonl").read_text())
                self.assertEqual(metadata["filename"], audio_path.name)
                self.assertEqual(metadata["transcript"], "")

    async def test_partial_sms_capture_keeps_audio_for_completed_turns(self):
        import json
        import tempfile

        dependencies = self.dependencies_for_failure("other")
        valid_pcm = pcm_windows(0, 0, 600, 600, *([0] * 18))

        class Capture:
            calls = 0

            async def start(self):
                pass

            async def result(self):
                type(self).calls += 1
                if self.calls == 2:
                    return pcm_windows(*([0] * 24)), 24000, 1, time.monotonic() + 0.1
                return valid_pcm, 24000, 1, time.monotonic() + 0.1

        dependencies = CaptureDependencies(
            **{**dependencies.__dict__, "capture_factory": lambda *_args: Capture()}
        )
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "retained-evidence"
            with (
                patch.dict(os.environ, TEST_VOICE_ENV),
                patch.object(
                    runner.caller,
                    "_speech_end_after_playout",
                    lambda *_args: asyncio.sleep(0, result=time.monotonic() - 1),
                ),
            ):
                with self.assertRaises(runner.PartialCaptureFailure) as caught:
                    await capture_script(
                        "android-selected-room",
                        ["Initial message@0::answer", "Confirmation@0::answer"],
                        dependencies=dependencies,
                        room_close_after=None,
                        retain_sms_audio_dir=evidence,
                        retain_sms_audio_scenario="sms-say-back-yes",
                    )

            self.assertEqual(len(caught.exception.turns), 1)
            audio_dir = evidence / "sms-audio"
            self.assertEqual(
                {path.name for path in audio_dir.iterdir()},
                {"android-selected-room-turn-001.wav", "transcripts.jsonl"},
            )
            record = json.loads((audio_dir / "transcripts.jsonl").read_text())
            self.assertEqual(record["turn"], 1)
            self.assertEqual(record["transcript"], caught.exception.turns[0]["transcript"])

    @staticmethod
    def dependencies_for_failure(failure):
        class Room:
            remote_participants = {
                "agent": SimpleNamespace()
            }
            callbacks = {}
            local_participant = SimpleNamespace(
                publish_track=lambda *_args, **_kwargs: asyncio.sleep(
                    0,
                    result=SimpleNamespace(wait_for_subscription=lambda: asyncio.sleep(0)),
                )
            )

            def on(self, event):
                def register(callback):
                    self.callbacks[event] = callback
                    return callback

                return register

            async def connect(self, *_args, **_kwargs):
                pass

            async def disconnect(self):
                pass

        class API:
            listings = 0

            def __init__(self, *_args, **_kwargs):
                self.room = self

            async def list_rooms(self, _request):
                type(self).listings += 1
                if failure == "room-list":
                    await asyncio.sleep(1)
                return SimpleNamespace(rooms=[SimpleNamespace(name="android-selected-room")])

            async def aclose(self):
                pass

        class Token:
            def __init__(self, *_args):
                pass

            def with_identity(self, _value):
                return self

            def with_grants(self, _value):
                return self

            def to_jwt(self):
                return "token"

        rtc = SimpleNamespace(
            Room=Room,
            AudioSource=lambda *_args: SimpleNamespace(
                capture_frame=lambda *_a: asyncio.sleep(0),
                wait_for_playout=lambda: asyncio.sleep(0),
            ),
            AudioFrame=lambda *_args: object(),
            LocalAudioTrack=SimpleNamespace(create_audio_track=lambda *_args: object()),
            TrackPublishOptions=lambda **_kwargs: object(),
            TrackSource=SimpleNamespace(SOURCE_MICROPHONE=1),
            ParticipantKind=SimpleNamespace(PARTICIPANT_KIND_AGENT=1),
            TrackKind=SimpleNamespace(KIND_AUDIO=1),
        )

        class Capture:
            def __init__(self, _queue, ended=None):
                self.ended = ended

            async def start(self):
                pass

            async def result(self):
                if failure == "room-disconnect":
                    Room.callbacks["disconnected"]()
                    if self.ended is None or not self.ended.is_set():
                        raise AssertionError("room disconnect did not end the capture")
                if failure == "audio":
                    raise RuntimeError("agent audio track produced no frames")
                if failure == "capture":
                    await asyncio.sleep(1)
                return pcm_windows(0, 0, 0, 0, 0, 600, 600, *([0] * 18)), 24000, 1, time.monotonic() + 0.1

        alice_responses = iter((
            "Alice Keck Park in Santa Barbara was donated by Alice Keck.",
            "W. M. Keck was her father.",
            "Her family's wealth came from Superior Oil.",
        ))

        async def transcribe(_http, pcm, _rate, _channels):
            if pcm in RENDERED_PCM_TEXT:
                text = RENDERED_PCM_TEXT[pcm]
                return [{"start": 0.0, "end": 1.0, "text": text}]
            if failure == "transcript":
                return []
            if failure == "alice":
                text = next(alice_responses)
            else:
                text = "wrong response" if failure == "answer" else "answer"
            return [{"start": 0.1, "end": 0.5, "text": text}]

        return CaptureDependencies(
            api=SimpleNamespace(
                AccessToken=Token,
                VideoGrants=lambda **_kwargs: object(),
                LiveKitAPI=API,
                ListRoomsRequest=lambda **kwargs: SimpleNamespace(**kwargs),
            ),
            rtc=rtc,
            http=object(),
            tts=lambda _http, text: asyncio.sleep(0, result=rendered_pcm(text)),
            transcribe=rendered_aware_transcriber(transcribe),
            capture_factory=Capture,
            monotonic=time.monotonic,
            sleep=asyncio.sleep,
        )

    async def test_caller_audio_retention_records_exact_render_and_actual_push(self):
        import json
        import stat
        import tempfile

        dependencies = self.dependencies_for_failure("other")
        rendered = b"\x01\x00" * 63_365
        RENDERED_PCM_TEXT[rendered] = "Exact caller words"
        pushed = []

        class Source:
            async def capture_frame(self, frame):
                pushed.append(frame)

            async def wait_for_playout(self):
                return None

        dependencies.rtc.AudioFrame = lambda pcm, rate, channels, samples: SimpleNamespace(
            pcm=pcm, sample_rate=rate, channels=channels, samples=samples
        )
        dependencies.rtc.AudioSource = lambda *_args: Source()
        dependencies = CaptureDependencies(**{
            **dependencies.__dict__,
            "tts": lambda *_args: asyncio.sleep(
                0, result=runner.caller.TTSResponse(rendered, 200, len(rendered))
            ),
        })
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, TEST_VOICE_ENV):
            evidence = Path(temporary)
            traces = await capture_script(
                "android-selected-room", ["Exact caller words@0::answer"],
                dependencies=dependencies, room_close_after=None,
                retain_caller_audio_dir=evidence,
            )
            audio_dir = evidence / "caller-audio"
            stem = "android-selected-room-turn-001-attempt-01"
            metadata_path = audio_dir / f"{stem}.json"
            record = json.loads(metadata_path.read_text())
            self.assertEqual((audio_dir / f"{stem}-rendered.pcm").read_bytes(), rendered)
            self.assertEqual(
                (audio_dir / f"{stem}-pushed.pcm").read_bytes(),
                b"".join(frame.pcm for frame in pushed),
            )
            self.assertEqual(record, {
                "turn": 1,
                "line": "Exact caller words",
                "attempt": 1,
                "content_check_passed": True,
                "tts_http_status": 200,
                "tts_response_bytes": len(rendered),
                "pushed_frames": len(pushed),
                "pushed_samples": 63_365,
                "rendered_filename": f"{stem}-rendered.pcm",
                "pushed_filename": f"{stem}-pushed.pcm",
            })
            self.assertEqual(stat.S_IMODE(evidence.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(audio_dir.stat().st_mode), 0o700)
            self.assertEqual(record["pushed_frames"], len(pushed))
            self.assertEqual(record["pushed_samples"], sum(frame.samples for frame in pushed))
            for path in audio_dir.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(traces[0]["line"], "Exact caller words")
            self.assertEqual(traces[0]["tts_retry_count"], 0)

    async def test_short_preflight_render_is_retained_with_zero_push_metrics(self):
        import json
        import tempfile

        dependencies = self.dependencies_for_failure("other")
        dependencies = CaptureDependencies(**{
            **dependencies.__dict__,
            "tts": lambda *_args: asyncio.sleep(
                0, result=runner.caller.TTSResponse(b"\x01\x00", 200, 2)
            ),
        })
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.dict(os.environ, TEST_VOICE_ENV),
                self.assertRaises(runner.PartialCaptureFailure),
            ):
                await capture_script(
                    "private-room", ["Long scripted line@0::answer"],
                    dependencies=dependencies, room_close_after=None,
                    retain_caller_audio_dir=Path(temporary),
                )

            audio_dir = Path(temporary) / "caller-audio"
            stem = "private-room-turn-001-attempt-01"
            record = json.loads((audio_dir / f"{stem}.json").read_text())
            self.assertEqual((audio_dir / f"{stem}-rendered.pcm").read_bytes(), b"\x01\x00")
            self.assertEqual((audio_dir / f"{stem}-pushed.pcm").read_bytes(), b"")
            self.assertEqual(record["line"], "Long scripted line")
            self.assertFalse(record["content_check_passed"])
            self.assertEqual(record["tts_http_status"], 200)
            self.assertEqual(record["tts_response_bytes"], 2)
            self.assertEqual(record["pushed_frames"], 0)
            self.assertEqual(record["pushed_samples"], 0)


class LocalEvalTests(unittest.TestCase):
    def test_expected_hangup_timeout_emits_failed_report_and_nonzero_exit(self):
        import io
        import json
        from contextlib import redirect_stdout

        class Stack:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def run(self, _run_id):
                return _run_context(self)

        observation = {
            "turns": [{
                "turn": 1,
                "kind": "action",
                "speech_end": 101.0,
                "speech_end_wall": 1_700_000_001.0,
                "first_audio": 102.0,
                "capture_started": 101.5,
                "segments": [{"start": 0.5, "end": 0.8, "text": "Timer set."}],
                "overlap": False,
                "confirmation": None,
                "room_deleted": None,
                "expect_confirmation": False,
                "expect_hangup": True,
                "model_calls": [{
            "id": "m1", "model": "claude-opus-5", "service_tier": None,
            "speed": None, "result_service_tier": None, "fast_mode_state": None,
        }],
            }],
            "failure": {
                "turn": 1,
                "message": "room deletion was not observed before deadline",
            },
            "product_failures": [],
            "phone_commands": [{"id": "fake-timer", "turn": 1, "kind": "timer"}],
        }
        output = io.StringIO()
        with (
            patch.object(runner, "DevStack", Stack),
            patch.object(runner, "SCENARIOS", (SCENARIOS[0],)),
            patch.object(runner, "observe_scenario", return_value=observation),
            redirect_stdout(output),
        ):
            result = runner._run_local_eval(["--live", "--runs", "1"])

        report = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertFalse(report["passed"])
        failures = " ".join(report["failures"])
        self.assertIn("room deletion was not observed before deadline", failures)
        self.assertIn("expected room_deleted observation is missing", failures)

    def test_stack_setup_failure_emits_structured_failed_report(self):
        import io
        import json
        from contextlib import redirect_stdout
        from evals.dev_stack import RemoteCommandError

        diagnostic = "DISTINCTIVE_REMOTE_SETUP_FAILURE"

        class FailingStack:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                raise RemoteCommandError(
                    1,
                    ["ssh", "ultraviolet", "sudo", "bash"],
                    output="",
                    stderr=f"remote setup failed: {diagnostic}",
                )

            def __exit__(self, *_args):
                raise AssertionError("failed __enter__ must clean up its own stack")

        output = io.StringIO()
        with patch.object(runner, "DevStack", FailingStack), redirect_stdout(output):
            result = runner._run_local_eval(["--live", "--runs", "1"])

        report = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertFalse(report["passed"])
        self.assertIn(diagnostic, " ".join(report["failures"]))
        self.assertIn(diagnostic, " ".join(report["cases"][0]["failures"]))

    def test_eval_json_scrubs_assignment_json_and_header_credentials(self):
        import io
        import json
        from contextlib import redirect_stdout

        secrets = (
            "bare-assignment-secret",
            "double-quoted-secret",
            "single-quoted-secret",
            "json-secret",
            "single-json-secret",
            "colon-secret",
            "export-secret",
            "patchbay-header-secret",
            "authorization-header-secret",
            "api-header-secret",
        )
        diagnostic = "DISTINCTIVE-CAUSE\n" + "\n".join((
            f"API_KEY={secrets[0]}",
            f'LIVEKIT_API_SECRET="{secrets[1]}"',
            f"LIVEKIT_AUTH_TOKEN='{secrets[2]}'",
            f'{{"LIVEKIT_API_SECRET": "{secrets[3]}"}}',
            f"{{'LIVEKIT_API_KEY': '{secrets[4]}'}}",
            f"VOICE_TOKEN: {secrets[5]}",
            f"export DEVICE_PASSWORD={secrets[6]}",
            f"X-Patchbay-Key: {secrets[7]}",
            f"Authorization: Bearer {secrets[8]}",
            f"x-api-key: {secrets[9]}",
        ))

        class Stack:
            def __init__(self, **_kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def run(self, _run_id):
                return _run_context(self)

        output = io.StringIO()
        with (
            patch.object(runner, "DevStack", Stack),
            patch.object(runner, "observe_scenario", side_effect=RuntimeError(diagnostic)),
            redirect_stdout(output),
        ):
            result = runner._run_local_eval(["--live", "--runs", "1"])

        serialized = output.getvalue()
        report = json.loads(serialized)
        self.assertEqual(result, 1)
        self.assertIn("DISTINCTIVE-CAUSE", serialized)
        self.assertFalse(report["passed"])
        for secret in secrets:
            self.assertNotIn(secret, serialized)


class ScenarioObservationTests(unittest.TestCase):
    def test_direct_turn_accepts_missing_optional_sdk_record(self):
        import json
        import subprocess
        from subprocess import CompletedProcess

        expectation = type(SCENARIOS[0].turns[0])(("forty-two",))
        scenario = type(SCENARIOS[0])(
            name="direct-answer",
            caller_lines=("What is the answer?",),
            turns=(expectation,),
            commands=(),
            room_close_after=None,
        )
        room = "direct-answer-room"
        trace = {
            "turn": 1,
            "room": room,
            "line": scenario.caller_lines[0],
            "transcript": "The answer is forty-two.",
            "speech_started_at": 100.0,
            "speech_end": 101.0,
            "speech_end_wall": 1_700_000_001.0,
            "first_audio": 102.0,
            "overlap": False,
            "capture_started": 101.5,
            "segments": [{"start": 0.2, "end": 0.8, "text": "The answer is forty-two."}],
            "room_deleted": None,
        }
        record_path = f"records/voice-{room}.jsonl"

        class Stack:
            base_url = "http://127.0.0.1:8485"

            def start_worker(self, _room):
                pass

            def run_voice(self, command, *, token, livekit_url):
                return CompletedProcess(command, 0, json.dumps({"turns": [trace]}), "")

            def run_remote(self, command):
                path = command[-1]
                if path == "voice/evals/phone.jsonl":
                    return CompletedProcess(command, 0, "", "")
                if path == "voice/evals/delegations.jsonl":
                    return CompletedProcess(command, 0, "", "")
                if path == record_path:
                    raise subprocess.CalledProcessError(
                        1,
                        ["ssh", "ultraviolet", "bash", "-s"],
                        output="",
                        stderr=f"cat: {record_path}: No such file or directory\n",
                    )
                raise AssertionError(f"unexpected remote command {command!r}")

        with patch.object(
            runner,
            "_voice_token",
            return_value={"token": "a.b.c", "room": room, "url": "wss://livekit.invalid"},
        ):
            observation = runner.observe_scenario(scenario, Stack())

        self.assertEqual(observation["turns"][0]["transcript"], "The answer is forty-two.")
        self.assertEqual(observation["turns"][0]["model_calls"], [])
        self.assertEqual(observation["phone_commands"], [])

    def test_remote_artifact_failures_name_path_and_scrub_stderr(self):
        import json
        import subprocess
        from subprocess import CompletedProcess

        expectation = type(SCENARIOS[0].turns[0])(("forty-two",))
        scenario = type(SCENARIOS[0])(
            name="direct-answer",
            caller_lines=("What is the answer?",),
            turns=(expectation,),
            commands=(),
            room_close_after=None,
        )
        room = "artifact-error-room"
        trace = {
            "turn": 1,
            "room": room,
            "line": scenario.caller_lines[0],
            "transcript": "The answer is forty-two.",
            "speech_started_at": 100.0,
            "speech_end": 101.0,
            "speech_end_wall": 1_700_000_001.0,
            "first_audio": 102.0,
            "overlap": False,
            "capture_started": 101.5,
            "segments": [{"start": 0.2, "end": 0.8, "text": "The answer is forty-two."}],
            "room_deleted": None,
        }
        paths = (
            "voice/evals/phone.jsonl",
            "voice/evals/delegations.jsonl",
            f"records/voice-{room}.jsonl",
        )
        for failed_path in paths:
            class Stack:
                base_url = "http://127.0.0.1:8485"

                def start_worker(self, _room):
                    pass

                def run_voice(self, command, *, token, livekit_url):
                    return CompletedProcess(command, 0, json.dumps({"turns": [trace]}), "")

                def run_remote(self, command):
                    path = command[-1]
                    if path == failed_path:
                        raise subprocess.CalledProcessError(
                            255,
                            ["ssh", "ultraviolet", "bash", "-s"],
                            output="",
                            stderr="Permission denied; Authorization: Bearer secret-value\\n",
                        )
                    if path == "voice/evals/phone.jsonl":
                        return CompletedProcess(command, 0, "", "")
                    if path == "voice/evals/delegations.jsonl":
                        return CompletedProcess(command, 0, "", "")
                    if path == f"records/voice-{room}.jsonl":
                        return CompletedProcess(command, 0, "", "")
                    raise AssertionError(f"unexpected remote command {command!r}")

            with patch.object(
                runner,
                "_voice_token",
                return_value={"token": "a.b.c", "room": room, "url": "wss://livekit.invalid"},
            ), self.subTest(path=failed_path):
                with self.assertRaises(RuntimeError) as raised:
                    runner.observe_scenario(scenario, Stack())
            self.assertIn(failed_path, str(raised.exception))
            self.assertNotIsInstance(raised.exception, subprocess.CalledProcessError)
            self.assertNotIn("secret-value", str(raised.exception))

    def test_missing_phone_and_marker_artifacts_fail_closed(self):
        import subprocess

        for path, label in (
            ("voice/evals/phone.jsonl", "fake phone log"),
            ("voice/evals/delegations.jsonl", "delegation marker log"),
        ):
            class Stack:
                def run_remote(self, command):
                    raise subprocess.CalledProcessError(
                        1,
                        ["ssh", "ultraviolet", "bash", "-s"],
                        output="",
                        stderr=f"cat: {path}: No such file or directory\\n",
                    )

            with self.subTest(path=path), self.assertRaises(RuntimeError) as raised:
                runner._remote_artifact_text(Stack(), path, label)
            self.assertIn(path, str(raised.exception))

    def test_turn_kinds_cover_search_action_and_search_only_scenarios(self):
        self.assertEqual(
            [runner._turn_kind(SCENARIOS[2], index) for index in (1, 2)],
            ["search", "action"],
        )
        self.assertEqual(
            [runner._turn_kind(SCENARIOS[5], index) for index in (1, 2, 3)],
            ["search", "search", "search"],
        )
        self.assertEqual(
            [runner._turn_kind(SCENARIOS[0], 1), runner._turn_kind(SCENARIOS[3], 1)],
            ["action", "search"],
        )

    def test_partial_observation_keeps_completed_wrong_answer_as_product_failure(self):
        import json
        from subprocess import CompletedProcess

        scenario = SCENARIOS[2]
        grant = {
            "token": "header.payload.signature",
            "room": "partial-product-room",
            "url": "wss://livekit.invalid",
        }
        trace = {
            "turn": 1,
            "room": grant["room"],
            "line": scenario.caller_lines[0],
            "transcript": "Wrong park.",
            "speech_started_at": 100.0,
            "speech_end": 101.0,
            "speech_end_wall": 1_700_000_001.0,
            "first_audio": 102.0,
            "overlap": False,
            "capture_started": 101.5,
            "segments": [{"start": 0.2, "end": 0.5, "text": "Wrong park."}],
            "room_deleted": None,
        }
        capture = {
            "turns": [trace],
            "failure": {
                "turn": 2,
                "message": "answer transcription exceeded its deadline",
                "speech_started_at": 200.0,
                "segments": [{"start": 0.0, "end": 0.2, "text": "partial"}],
            },
        }
        record = "".join(
            json.dumps(message) + "\n"
            for message in (
                {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m1", "model": "claude-opus-5"}}},
                {"type": "result", "session_id": "voice-partial-product-room"},
                {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m2", "model": "claude-opus-5"}}},
                {"type": "result", "session_id": "voice-partial-product-room"},
            )
        )
        delegation_markers = (
            '{"room":"partial-product-room","id":"d1","created_at":100.5}\n'
            '{"room":"partial-product-room","id":"d2","created_at":200.5}\n'
        )

        class Stack:
            base_url = "http://127.0.0.1:8485"

            def start_worker(self, _room):
                pass

            def run_voice(self, command, *, token, livekit_url):
                return CompletedProcess(command, 0, json.dumps(capture), "")

            def run_remote(self, command):
                if command == ["sudo", "cat", "voice/evals/phone.jsonl"]:
                    output = ""
                elif command == ["sudo", "cat", "voice/evals/delegations.jsonl"]:
                    output = delegation_markers
                elif command == ["sudo", "cat", "records/voice-partial-product-room.jsonl"]:
                    output = record
                else:
                    raise AssertionError(f"unexpected remote command {command!r}")
                return CompletedProcess(command, 0, output, "")

        with patch.object(runner, "_voice_token", return_value=grant):
            observation = runner.observe_scenario(scenario, Stack())

        self.assertEqual(observation["turns"][0]["model_calls"], [{
            "id": "m1", "model": "claude-opus-5", "service_tier": None,
            "speed": None, "result_service_tier": None,
        }])
        self.assertEqual(observation["failure"], capture["failure"])
        product_failures = observation["product_failures"]
        self.assertTrue(any(failure["turn"] == 1 for failure in product_failures))
        self.assertTrue(any("missing answer pattern" in failure["message"] for failure in product_failures))

    def test_complete_wrong_place_and_missing_sms_confirmation_keep_full_evidence(self):
        import json
        from subprocess import CompletedProcess

        def observe_fixture(scenario, transcripts, phone_payloads, tool_payloads):
            room = "complete-product-room"
            traces = []
            for index, transcript in enumerate(transcripts, 1):
                started = float(index * 100)
                traces.append({
                    "turn": index,
                    "room": room,
                    "line": scenario.caller_lines[index - 1],
                    "transcript": transcript,
                    "speech_started_at": started,
                    "speech_end": started + 1.0,
                    "speech_end_wall": started + 1.0,
                    "first_audio": started + 2.0,
                    "overlap": False,
                    "capture_started": started + 1.5,
                    "segments": [{"start": 0.2, "end": 0.6, "text": transcript}],
                    "room_deleted": started + 3.0 if index == len(transcripts) else None,
                })
            phone_log = []
            for index, payload in enumerate(phone_payloads, 1):
                command = {"id": f"phone-{index}", **payload}
                phone_log.extend((
                    {"event": "command", "command": command, "received_at": float(index * 100 + 1)},
                    {"event": "result", "result": {"id": command["id"], "status": "ok", "detail": "Fake phone completed"}},
                ))
            record = []
            marker_log = []
            for index, payload in enumerate(tool_payloads, 1):
                record.extend((
                    {"type": "stream_event", "event": {"type": "message_start", "message": {"id": f"m{index}", "model": "claude-opus-5"}}},
                ))
                if payload is not None:
                    name, arguments = payload
                    record.append({"type": "assistant", "message": {"content": [{"type": "tool_use", "id": f"t{index}", "name": f"mcp__mentat__{name}", "input": arguments}]}})
                record.append({"type": "result", "session_id": "voice-" + room})
                marker_log.append({"room": room, "id": f"d{index}", "created_at": index * 100 + 0.5})
            capture = json.dumps({"turns": traces})
            encoded_record = "".join(json.dumps(item) + "\n" for item in record)
            encoded_phone = "".join(json.dumps(item) + "\n" for item in phone_log)
            encoded_markers = "".join(json.dumps(item) + "\n" for item in marker_log)

            class Stack:
                base_url = "http://127.0.0.1:8485"

                def start_worker(self, _room):
                    pass

                def run_voice(self, command, *, token, livekit_url):
                    return CompletedProcess(command, 0, capture, "")

                def run_remote(self, command):
                    outputs = {
                        "voice/evals/phone.jsonl": encoded_phone,
                        "voice/evals/delegations.jsonl": encoded_markers,
                        f"records/voice-{room}.jsonl": encoded_record,
                    }
                    key = command[-1]
                    if key not in outputs:
                        raise AssertionError(f"unexpected remote command {command!r}")
                    return CompletedProcess(command, 0, outputs[key], "")

            with patch.object(
                runner, "_voice_token", return_value={"token": "a.b.c", "room": room, "url": "wss://livekit.invalid"}
            ):
                return runner.observe_scenario(scenario, Stack())

        place = SCENARIOS[2]
        wrong_place = observe_fixture(
            place,
            ["Wrong Park is in Santa Barbara.", "Taking you to Wrong Park."],
            [
                {"kind": "location", "query": place.place_query},
                {"kind": "navigate", "name": "Wrong Park", "address": "1 Main St", "place_id": "wrong", "lat": 34.4, "lng": -119.7},
            ],
            [
                ("find_places", {"query": place.place_query}),
                ("navigate_to", {"name": "Wrong Park", "address": "1 Main St", "place_id": "wrong", "lat": 34.4, "lng": -119.7}),
            ],
        )
        self.assertEqual(len(wrong_place["turns"]), 2)
        self.assertEqual([len(turn["model_calls"]) for turn in wrong_place["turns"]], [1, 1])
        self.assertEqual([command["kind"] for command in wrong_place["phone_commands"]], ["location", "navigate"])
        self.assertTrue(any(item["turn"] == 2 and "unexpected place name" in item["message"] for item in wrong_place["product_failures"]))
        from evals.report import score_observations

        place_report = score_observations(
            {"cases": [{"name": place.name, "runs": [wrong_place]}]},
            required_runs=1,
        )
        self.assertFalse(place_report["passed"])
        self.assertEqual(len(place_report["cases"][0]["turns"]), 2)
        self.assertEqual(place_report["cases"][0]["turns"][1]["model_call_count"], 1)
        self.assertEqual(place_report["cases"][0]["turns"][0]["latency_seconds"]["first_audio"], 1.0)
        self.assertTrue(any("unexpected place name" in failure for failure in place_report["failures"]))

        sms = SCENARIOS[3]
        missing_prompt = observe_fixture(
            sms,
            ["I can text +1-202-555-0142: I will be there at six.", "Sent that message."],
            [],
            [None, None],
        )
        self.assertEqual(len(missing_prompt["turns"]), 2)
        self.assertIsNone(missing_prompt["turns"][0]["confirmation"])
        self.assertTrue(any(item["turn"] == 1 and "say-back" in item["message"] for item in missing_prompt["product_failures"]))
        sms_report = score_observations(
            {"cases": [{"name": sms.name, "runs": [missing_prompt]}]},
            required_runs=1,
        )
        self.assertFalse(sms_report["passed"])
        self.assertEqual(len(sms_report["cases"][0]["turns"]), 2)
        self.assertEqual(sms_report["cases"][0]["turns"][0]["model_call_count"], 1)
        sms_failures = " ".join(sms_report["failures"])
        self.assertIn("say-back", sms_failures)
        self.assertIn("confirmation observation is missing", sms_failures)

    def test_spanish_caller_rendering_matches_accented_transcription(self):
        self.assertTrue(
            runner._rendered_content_matches(
                "¿Cuál es la capital de Francia?",
                [{"text": "Cuál es la capital de Francia."}],
            )
        )
        self.assertFalse(
            runner._rendered_content_matches(
                "¿Cuál es la capital de Francia?",
                [{"text": "Cuál es la capital de España."}],
            )
        )

    def test_every_scenario_script_fits_the_single_room_capture_format(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.name):
                steps = runner._scenario_steps(scenario)
                self.assertEqual(len(steps), len(scenario.caller_lines))
                self.assertEqual(
                    [runner.caller.parse_step(step)[1] for step in steps],
                    list(scenario.caller_lines),
                )
                if scenario.name == "spanish-language-switch":
                    self.assertEqual(
                        [runner.caller.parse_step_language(step) for step in steps],
                        ["en", "es", "en", "en", "es"],
                    )

    def test_spanish_interpreter_trace_requires_stt_modes_voices_reply_ids_and_no_phone_commands(self):
        scenario = SimpleNamespace(
            name="spanish-interpreter",
            caller_lines=(
                "Please interpret for a Spanish-speaking gardener.",
                "La planta necesita agua.",
                "The soil is dry.",
                "Pon un temporizador de cinco minutos.",
                "Dijo: deja de traducir.",
                "I'm done interpreting.",
            ),
            caller_languages=("en", "es", "en", "es", "es", "en"),
            reply_languages=("es", "en", "es", "en", "en", "en"),
            voice_mode_expectations=("es", "en"),
        )
        entries = [
            {"room": "interpreter-room", "event": "mode", "mode": "interpreter", "language": "es", "voice_id": "spanish-library", "created_at": 101.0, "lookup_ms": 24.5, "selection": "resolved"},
            {"room": "interpreter-room", "event": "mode", "mode": "normal", "language": "en", "voice_id": "english-default", "created_at": 106.0, "lookup_ms": 0.0, "selection": "default"},
            *[
                {"room": "interpreter-room", "event": "speech", "reply": index, "turn_id": f"turn-{index}", "language": language, "voice_id": voice, "created_at": timestamp}
                for index, (language, voice, timestamp) in enumerate(
                    zip(
                        scenario.reply_languages,
                        ("spanish-library", "english-default", "spanish-library", "english-default", "english-default", "english-default"),
                        (102.0, 103.0, 104.0, 105.0, 105.5, 107.0),
                        strict=True,
                    ),
                    1,
                )
            ],
        ]
        phone_commands = []

        failures, lookup_ms = runner._spanish_interpreter_evidence_failures(
            scenario, "interpreter-room", entries, list(scenario.caller_lines), phone_commands
        )
        self.assertEqual(failures, [])
        self.assertEqual(lookup_ms, 24.5)

        invalid_evidence = (
            ([*entries[:3], *entries[4:]], list(scenario.caller_lines), phone_commands, "reply 2"),
            (entries, [*scenario.caller_lines[:3], "Set a five minute timer.", *scenario.caller_lines[4:]], phone_commands, "turn 4"),
            ([*entries[:1], *entries[2:]], list(scenario.caller_lines), phone_commands, "mode transitions"),
            ([{**entry, "mode": "conversation"} if entry.get("event") == "mode" and entry.get("language") == "es" else entry for entry in entries], list(scenario.caller_lines), phone_commands, "interpreter/es"),
            ([*entries, {**entries[0], "created_at": 108.0}], list(scenario.caller_lines), phone_commands, "mode transitions"),
            ([*entries[:1], {**entries[1], "created_at": 105.25}, *entries[2:]], list(scenario.caller_lines), phone_commands, "normal/en transition"),
            ([*entries[:2], {**entries[2], "language": "en"}, *entries[3:]], list(scenario.caller_lines), phone_commands, "reply 1"),
            ([*entries[:2], {**entries[2], "voice_id": "english-default"}, *entries[3:]], list(scenario.caller_lines), phone_commands, "reply 1"),
            ([*entries[:3], {**entries[3], "voice_id": "spanish-library"}, *entries[4:]], list(scenario.caller_lines), phone_commands, "reply 2"),
            ([*entries[:2], *entries[3:]], list(scenario.caller_lines), phone_commands, "reply 1 is missing"),
            ([*entries, {**entries[2], "reply": 2, "created_at": 108.0}], list(scenario.caller_lines), phone_commands, "duplicate speech reply order"),
            ([*entries[:3], entries[4], entries[3], *entries[5:]], list(scenario.caller_lines), phone_commands, "out of order"),
            ([*entries[:2], {**entries[2], "turn_id": ""}, *entries[3:]], list(scenario.caller_lines), phone_commands, "turn id"),
            ([*entries[:2], *entries[2:4], {**entries[4], "turn_id": "turn-2"}, *entries[5:]], list(scenario.caller_lines), phone_commands, "turn id is shared"),
            (entries, list(scenario.caller_lines), [{"kind": "timer", "turn": 4}], "command(s)"),
        )
        for changed_entries, sidecars, commands, expected in invalid_evidence:
            with self.subTest(expected=expected):
                failures, _ = runner._spanish_interpreter_evidence_failures(
                    scenario, "interpreter-room", changed_entries, sidecars, commands
                )
                self.assertTrue(any(expected in failure.lower() for failure in failures), failures)

    def test_spanish_interpreter_requires_final_english_segment_after_normal_transition(self):
        scenario = SimpleNamespace(
            caller_lines=(
                "Please interpret for a Spanish-speaking gardener.",
                "La planta necesita agua.",
                "The soil is dry.",
                "Pon un temporizador de cinco minutos.",
                "Dijo: deja de traducir.",
                "I'm done interpreting.",
            ),
            caller_languages=("en", "es", "en", "es", "es", "en"),
            reply_languages=("es", "en", "es", "en", "en", "en"),
            voice_mode_expectations=("es", "en"),
        )
        entries = [
            {"room": "interpreter-room", "event": "mode", "mode": "interpreter", "language": "es", "voice_id": "spanish-library", "created_at": 101.0, "lookup_ms": 24.5, "selection": "resolved"},
            {"room": "interpreter-room", "event": "mode", "mode": "normal", "language": "en", "voice_id": "english-default", "created_at": 106.0, "lookup_ms": 0.0, "selection": "default"},
            *[
                {"room": "interpreter-room", "event": "speech", "reply": index, "turn_id": f"turn-{index}", "language": language, "voice_id": voice, "created_at": timestamp}
                for index, (language, voice, timestamp) in enumerate(
                    zip(
                        scenario.reply_languages[:5],
                        ("spanish-library", "english-default", "spanish-library", "english-default", "english-default"),
                        (102.0, 103.0, 104.0, 105.0, 105.5),
                        strict=True,
                    ),
                    1,
                )
            ],
            {"room": "interpreter-room", "event": "speech", "reply": 6, "turn_id": "turn-6", "language": "en", "voice_id": "english-default", "created_at": 105.75},
            {"room": "interpreter-room", "event": "speech", "reply": 6, "turn_id": "turn-6", "language": "en", "voice_id": "english-default", "created_at": 107.0},
        ]

        failures, lookup_ms = runner._spanish_interpreter_evidence_failures(
            scenario,
            "interpreter-room",
            entries,
            list(scenario.caller_lines),
            [],
        )

        self.assertEqual(failures, [])
        self.assertEqual(lookup_ms, 24.5)

        invalid_entries = (
            ([*entries[:-1]], "post-normal"),
            ([*entries[:-1], {**entries[-1], "language": "es"}], "wrong language"),
            ([*entries[:-1], {**entries[-1], "voice_id": "spanish-library"}], "wrong voice"),
            ([entries[0], {**entries[1], "created_at": 105.25}, *entries[2:]], "normal/en transition"),
        )
        for changed_entries, expected in invalid_entries:
            with self.subTest(expected=expected):
                failures, _ = runner._spanish_interpreter_evidence_failures(
                    scenario,
                    "interpreter-room",
                    changed_entries,
                    list(scenario.caller_lines),
                    [],
                )
                self.assertTrue(any(expected in failure.lower() for failure in failures), failures)

    def test_spanish_interpreter_allows_default_voice_ack_before_translation_in_reply_group(self):
        scenario = SimpleNamespace(
            caller_lines=(
                "Please interpret for a Spanish-speaking gardener.",
                "La planta necesita agua.",
                "The soil is dry.",
                "Pon un temporizador de cinco minutos.",
                "Dijo: deja de traducir.",
                "I'm done interpreting.",
            ),
            caller_languages=("en", "es", "en", "es", "es", "en"),
            reply_languages=("es", "en", "es", "en", "en", "en"),
            voice_mode_expectations=("es", "en"),
        )
        entries = [
            {"room": "interpreter-room", "event": "mode", "mode": "interpreter", "language": "es", "voice_id": "spanish-library", "created_at": 101.0, "lookup_ms": 24.5, "selection": "resolved"},
            {"room": "interpreter-room", "event": "mode", "mode": "normal", "language": "en", "voice_id": "english-default", "created_at": 106.0, "lookup_ms": 0.0, "selection": "default"},
            {"room": "interpreter-room", "event": "speech", "reply": 1, "turn_id": "turn-1", "language": "en", "voice_id": "english-default", "created_at": 100.5},
            {"room": "interpreter-room", "event": "speech", "reply": 1, "turn_id": "turn-1", "language": "es", "voice_id": "spanish-library", "created_at": 101.5},
            *[
                {"room": "interpreter-room", "event": "speech", "reply": index, "turn_id": f"turn-{index}", "language": language, "voice_id": voice, "created_at": timestamp}
                for index, (language, voice, timestamp) in enumerate(
                    zip(
                        scenario.reply_languages[1:],
                        ("english-default", "spanish-library", "english-default", "english-default", "english-default"),
                        (102.5, 103.5, 104.5, 105.5, 107.0),
                        strict=True,
                    ),
                    2,
                )
            ],
        ]

        failures, lookup_ms = runner._spanish_interpreter_evidence_failures(
            scenario,
            "interpreter-room",
            entries,
            list(scenario.caller_lines),
            [],
        )

        self.assertEqual(failures, [])
        self.assertEqual(lookup_ms, 24.5)

        invalid_entries = (
            ([*entries[:2], {**entries[2], "voice_id": "spanish-library"}, *entries[3:]], "wrong voice"),
            (
                [
                    *entries[:5],
                    {
                        **entries[3],
                        "language": "en",
                        "voice_id": "english-default",
                        "created_at": 101.75,
                    },
                    *entries[5:],
                ],
                "wrong language",
            ),
            ([*entries[:3], *entries[4:]], "no Spanish speech after interpreter mode"),
        )
        for changed_entries, expected in invalid_entries:
            with self.subTest(expected=expected):
                failures, _ = runner._spanish_interpreter_evidence_failures(
                    scenario,
                    "interpreter-room",
                    changed_entries,
                    list(scenario.caller_lines),
                    [],
                )
                self.assertTrue(any(expected in failure for failure in failures), failures)

    def test_spanish_interpreter_allows_tagged_english_meta_after_mode_before_spanish_translation(self):
        scenario = SimpleNamespace(
            caller_lines=(
                "Please interpret for a Spanish-speaking gardener.",
                "La planta necesita agua.",
                "The soil is dry.",
                "Pon un temporizador de cinco minutos.",
                "Dijo: deja de traducir.",
                "I'm done interpreting.",
            ),
            caller_languages=("en", "es", "en", "es", "es", "en"),
            reply_languages=("es", "en", "es", "en", "en", "en"),
            voice_mode_expectations=("es", "en"),
        )
        entries = [
            {"room": "interpreter-room", "event": "mode", "mode": "interpreter", "language": "es", "voice_id": "spanish-library", "created_at": 101.0, "lookup_ms": 24.5, "selection": "resolved"},
            {"room": "interpreter-room", "event": "mode", "mode": "normal", "language": "en", "voice_id": "english-default", "created_at": 106.0, "lookup_ms": 0.0, "selection": "default"},
            {"room": "interpreter-room", "event": "speech", "reply": 1, "turn_id": "turn-1", "language": "en", "voice_id": "english-default", "created_at": 100.5},
            {"room": "interpreter-room", "event": "speech", "reply": 1, "turn_id": "turn-1", "language": "en", "voice_id": "english-default", "created_at": 101.25},
            {"room": "interpreter-room", "event": "speech", "reply": 1, "turn_id": "turn-1", "language": "es", "voice_id": "spanish-library", "created_at": 101.5},
            *[
                {"room": "interpreter-room", "event": "speech", "reply": index, "turn_id": f"turn-{index}", "language": language, "voice_id": voice, "created_at": timestamp}
                for index, (language, voice, timestamp) in enumerate(
                    zip(
                        scenario.reply_languages[1:],
                        ("english-default", "spanish-library", "english-default", "english-default", "english-default"),
                        (102.0, 103.0, 104.0, 105.0, 107.0),
                        strict=True,
                    ),
                    2,
                )
            ],
        ]

        failures, _ = runner._spanish_interpreter_evidence_failures(
            scenario, "interpreter-room", entries, list(scenario.caller_lines), []
        )

        self.assertEqual(failures, [])

        invalid_entries = (
            ([*entries[:3], {**entries[3], "voice_id": "spanish-library"}, *entries[4:]], "wrong voice"),
            ([*entries[:4], *entries[5:]], "no Spanish speech after interpreter mode"),
        )
        for changed_entries, expected in invalid_entries:
            with self.subTest(expected=expected):
                failures, _ = runner._spanish_interpreter_evidence_failures(
                    scenario, "interpreter-room", changed_entries, list(scenario.caller_lines), []
                )
                self.assertTrue(any(expected in failure for failure in failures), failures)

    def test_spanish_interpreter_observation_reads_committed_stt_and_fake_phone_log(self):
        import json
        import subprocess
        from subprocess import CompletedProcess

        scenario = SimpleNamespace(
            name="spanish-interpreter",
            caller_lines=(
                "Please interpret for a Spanish-speaking gardener.",
                "La planta necesita agua.",
                "The soil is dry.",
                "Pon un temporizador de cinco minutos.",
                "Dijo: deja de traducir.",
                "I'm done interpreting.",
            ),
            caller_languages=("en", "es", "en", "es", "es", "en"),
            reply_languages=("es", "en", "es", "en", "en", "en"),
            voice_mode_expectations=("es", "en"),
            turns=tuple(SimpleNamespace(answer_patterns=("ok",), reject_patterns=(), sms_recipient=None, sms_body=None) for _ in range(6)),
            room_close_after=6,
            commands=(),
        )
        room = "spanish-interpreter-room"
        traces = [
            {
                "turn": index,
                "room": room,
                "line": line,
                "transcript": "Ok.",
                "speech_started_at": 100.0 + index,
                "speech_end": 100.3 + index,
                "speech_end_wall": 1_700_000_000.3 + index,
                "first_audio": 101.0 + index,
                "capture_started": 100.0 + index,
                "overlap": False,
                "segments": [{"start": 0.6, "end": 0.8, "text": "Ok."}],
                "room_deleted": 108.0 if index == 6 else None,
            }
            for index, line in enumerate(scenario.caller_lines, 1)
        ]
        voice_entries = [
            {"room": room, "event": "mode", "mode": "interpreter", "language": "es", "voice_id": "spanish-library", "created_at": 101.0, "lookup_ms": 24.5, "selection": "resolved"},
            {"room": room, "event": "mode", "mode": "normal", "language": "en", "voice_id": "english-default", "created_at": 106.0, "lookup_ms": 0.0, "selection": "default"},
            *[
                {"room": room, "event": "speech", "reply": index, "turn_id": f"turn-{index}", "language": language, "voice_id": voice, "created_at": timestamp}
                for index, (language, voice, timestamp) in enumerate(
                    zip(
                        scenario.reply_languages,
                        ("spanish-library", "english-default", "spanish-library", "english-default", "english-default", "english-default"),
                        (102.0, 103.0, 104.0, 105.0, 105.5, 107.0),
                        strict=True,
                    ),
                    1,
                )
            ],
        ]

        class Stack:
            base_url = "http://127.0.0.1:8485"

            def start_worker(self, _room):
                pass

            def run_voice(self, command, *, token, livekit_url):
                return CompletedProcess(command, 0, json.dumps({"turns": traces}), "")

            def run_remote(self, command):
                path = command[-1]
                if path == "voice/evals/phone.jsonl":
                    return CompletedProcess(command, 0, "", "")
                if path == "voice/evals/voice-modes.jsonl":
                    return CompletedProcess(command, 0, "".join(json.dumps(row) + "\n" for row in voice_entries), "")
                if path == "voice/evals/delegations.jsonl":
                    return CompletedProcess(command, 0, "", "")
                if path == f"records/voice-{room}.jsonl":
                    raise subprocess.CalledProcessError(1, command, output="", stderr=f"cat: {path}: No such file or directory")
                prefix = "voice/evals/retained-evidence/input-audio/"
                if path.startswith(prefix) and path.endswith(".txt"):
                    index = int(path.rsplit("-", 1)[1][:-4])
                    return CompletedProcess(command, 0, scenario.caller_lines[index - 1], "")
                raise AssertionError(f"unexpected remote artifact {path!r}")

        with patch.object(
            runner,
            "_voice_token",
            return_value={"token": "a.b.c", "room": room, "url": "wss://livekit.invalid"},
        ):
            observation = runner.observe_scenario(scenario, Stack())

        self.assertNotIn("product_failures", observation)
        self.assertEqual(observation["phone_commands"], [])
        self.assertEqual(observation["first_spanish_lookup_ms"], 24.5)
        self.assertEqual(len(observation["turns"]), 6)

    def test_spanish_language_evidence_accepts_only_mode_correct_reply_segments(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-language-switch")
        records = [
            {"room": "switch-room", "event": "mode", "mode": "conversation", "language": "es", "voice_id": "spanish-library", "created_at": 102.0, "lookup_ms": 24.5, "selection": "resolved"},
            {"room": "switch-room", "event": "mode", "mode": "normal", "language": "en", "voice_id": "english-default", "created_at": 104.0, "lookup_ms": 0.0, "selection": "default"},
            {"room": "switch-room", "event": "mode", "mode": "conversation", "language": "es", "voice_id": "spanish-library", "created_at": 105.0, "lookup_ms": 0.0, "selection": "reused"},
            {"room": "switch-room", "event": "speech", "reply": 1, "turn_id": "turn-1", "language": "en", "voice_id": "english-default", "created_at": 101.0},
            {"room": "switch-room", "event": "speech", "reply": 1, "turn_id": "turn-1", "language": "es", "voice_id": "spanish-library", "created_at": 103.0},
            *[
                {"room": "switch-room", "event": "speech", "reply": index, "turn_id": f"turn-{index}", "language": language, "voice_id": voice_id, "created_at": timestamp}
                for index, language, voice_id, timestamp in (
                    (2, "es", "spanish-library", 103.5),
                    (3, "en", "english-default", 104.5),
                    (4, "es", "spanish-library", 105.5),
                    (5, "es", "spanish-library", 106.5),
                )
            ],
        ]
        failures, first_lookup_ms = runner._spanish_switch_evidence_failures(
            scenario, "switch-room", records, list(scenario.caller_lines)
        )
        self.assertEqual(failures, [])
        self.assertEqual(first_lookup_ms, 24.5)

        invalid_evidence = (
            ([*records[:4], {**records[4], "language": "en", "voice_id": "english-default"}, *records[5:]], "speech reply order 1"),
            ([*records[:4], {**records[4], "voice_id": "wrong-spanish-voice"}, *records[5:]], "speech reply order 1"),
            ([*records[:4], {**records[4], "voice_id": ""}, *records[5:]], "no synthesized voice"),
            ([*records[:5], {**records[5], "turn_id": "turn-1"}, *records[6:]], "turn id is shared"),
            ([*records[:4], {**records[4], "turn_id": "turn-other"}, *records[5:]], "turn id"),
            ([*records[:4], {**records[4], "turn_id": ""}, *records[5:]], "turn id"),
            ([*records[:4], {**records[4], "created_at": float("nan")}, *records[5:]], "timestamp"),
            ([*records, {**records[4], "reply": 1, "created_at": 103.2}], "duplicate speech reply order"),
            ([*records, {"room": "switch-room", "event": "speech", "reply": 6, "turn_id": "turn-6", "language": "xx", "voice_id": "wrong-voice", "created_at": float("nan")}], "unexpected speech reply order 6"),
            ([*records[:5], records[6], records[5], *records[7:]], "speech reply groups are out of order"),
        )
        for changed_records, expected in invalid_evidence:
            with self.subTest(expected=expected):
                failures, _ = runner._spanish_switch_evidence_failures(
                    scenario, "switch-room", changed_records, list(scenario.caller_lines)
                )
                self.assertTrue(any(expected in failure.lower() for failure in failures), failures)

    def test_spanish_language_evidence_binds_stt_modes_and_synthesized_voices(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-language-switch")
        records = [
            {"room": "switch-room", "event": "mode", "mode": "conversation", "language": "es", "voice_id": "spanish-library", "created_at": 101.0, "lookup_ms": 24.5, "selection": "resolved"},
            {"room": "switch-room", "event": "mode", "mode": "normal", "language": "en", "voice_id": "english-default", "created_at": 104.0, "lookup_ms": 0.0, "selection": "default"},
            {"room": "switch-room", "event": "mode", "mode": "conversation", "language": "es", "voice_id": "spanish-library", "created_at": 105.0, "lookup_ms": 0.0, "selection": "reused"},
            *[
                {"room": "switch-room", "event": "speech", "reply": index, "turn_id": f"turn-{index}", "language": language, "voice_id": voice_id, "created_at": created_at}
                for index, (language, voice_id, created_at) in enumerate(zip(
                    ("es", "es", "en", "es", "es"),
                    ("spanish-library", "spanish-library", "english-default", "spanish-library", "spanish-library"),
                    (102.5, 103.0, 104.5, 105.5, 106.0),
                    strict=True,
                ), 1)
            ],
        ]
        transcript_sidecars = list(scenario.caller_lines)

        failures, first_lookup_ms = runner._spanish_switch_evidence_failures(
            scenario, "switch-room", records, transcript_sidecars
        )
        self.assertEqual(failures, [])
        self.assertEqual(first_lookup_ms, 24.5)

        invalid_evidence = (
            ([*records[:-1], {**records[-1], "voice_id": "different-spanish-voice"}], transcript_sidecars, "reused"),
            (records, transcript_sidecars[:-1] + ["Cambia a español."], "turn 5"),
            ([record for record in records if record.get("event") != "mode" or record["language"] != "en"], transcript_sidecars, "mode transition"),
            ([{**record, "reply": 6} if record.get("event") == "speech" and record["reply"] == 5 else record for record in records], transcript_sidecars, "reply order"),
            ([{key: value for key, value in record.items() if key != "turn_id"} if record.get("event") == "speech" and record["reply"] == 5 else record for record in records], transcript_sidecars, "turn id"),
            ([{**record, "selection": "resolved"} if record.get("event") == "mode" and record["language"] == "es" and record["created_at"] == 105.0 else record for record in records], transcript_sidecars, "second spanish switch"),
        )
        for changed_records, changed_sidecars, expected in invalid_evidence:
            with self.subTest(expected=expected):
                failures, _ = runner._spanish_switch_evidence_failures(
                    scenario, "switch-room", changed_records, changed_sidecars
                )
                self.assertTrue(any(expected in failure.lower() for failure in failures), failures)

    def test_spanish_switch_observation_loads_private_sidecars_and_voice_trace(self):
        import json
        import subprocess
        from subprocess import CompletedProcess

        scenario = next(s for s in SCENARIOS if s.name == "spanish-language-switch")
        room = "spanish-switch-room"
        replies = (
            "Claro, continuaremos en español.",
            "La capital de Francia es París.",
            "Sure, we're back to English.",
            "Claro, hablaremos español otra vez.",
            "El cielo es azul.",
        )
        traces = [
            {
                "turn": index,
                "room": room,
                "line": line,
                "transcript": reply,
                "speech_started_at": 100.0 + index,
                "speech_end": 100.5 + index,
                "speech_end_wall": 1_700_000_000.5 + index,
                "first_audio": 101.0 + index,
                "capture_started": 100.8 + index,
                "overlap": False,
                "segments": [{"start": 0.2, "end": 0.6, "text": reply}],
                "room_deleted": 106.0 if index == 5 else None,
            }
            for index, (line, reply) in enumerate(
                zip(scenario.caller_lines, replies, strict=True), 1
            )
        ]
        languages = ("es", "es", "en", "es", "es")
        voices = ("spanish-library", "spanish-library", "english-default", "spanish-library", "spanish-library")
        evidence = [
            {"room": room, "event": "mode", "mode": "conversation", "language": "es", "voice_id": "spanish-library", "created_at": 102.0, "lookup_ms": 24.5, "selection": "resolved"},
            {"room": room, "event": "mode", "mode": "normal", "language": "en", "voice_id": "english-default", "created_at": 104.0, "lookup_ms": 0.0, "selection": "default"},
            {"room": room, "event": "mode", "mode": "conversation", "language": "es", "voice_id": "spanish-library", "created_at": 105.0, "lookup_ms": 0.0, "selection": "reused"},
            *[
                {"room": room, "event": "speech", "reply": index, "turn_id": f"turn-{index}", "language": language, "voice_id": voice, "created_at": timestamp}
                for index, (language, voice, timestamp) in enumerate(
                    zip(languages, voices, (102.5, 103.0, 104.5, 105.5, 106.0), strict=True), 1
                )
            ],
        ]
        requested_paths = []
        record_path = f"records/voice-{room}.jsonl"

        class Stack:
            base_url = "http://127.0.0.1:8485"

            def start_worker(self, _room):
                pass

            def run_voice(self, command, *, token, livekit_url):
                return CompletedProcess(command, 0, json.dumps({"turns": traces}), "")

            def run_remote(self, command):
                path = command[-1]
                requested_paths.append(path)
                if path == "voice/evals/phone.jsonl":
                    return CompletedProcess(command, 0, "", "")
                if path == "voice/evals/voice-modes.jsonl":
                    return CompletedProcess(command, 0, "".join(json.dumps(row) + "\n" for row in evidence), "")
                if path == "voice/evals/delegations.jsonl":
                    return CompletedProcess(command, 0, "", "")
                if path == record_path:
                    raise subprocess.CalledProcessError(
                        1, command, output="", stderr=f"cat: {record_path}: No such file or directory\n"
                    )
                prefix = "voice/evals/retained-evidence/input-audio/"
                if path.startswith(prefix) and path.endswith(".txt"):
                    index = int(path.rsplit("-", 1)[1][:-4])
                    return CompletedProcess(command, 0, scenario.caller_lines[index - 1], "")
                raise AssertionError(f"unexpected remote artifact {path!r}")

        with patch.object(
            runner,
            "_voice_token",
            return_value={"token": "a.b.c", "room": room, "url": "wss://livekit.invalid"},
        ):
            observation = runner.observe_scenario(scenario, Stack())

        self.assertNotIn("product_failures", observation)
        self.assertEqual(observation["first_spanish_lookup_ms"], 24.5)
        self.assertIn("voice/evals/voice-modes.jsonl", requested_paths)
        self.assertEqual(
            sum(path.endswith(".txt") for path in requested_paths),
            len(scenario.caller_lines),
        )

    def test_same_turn_hangup_timeout_retains_sdk_and_fake_phone_evidence(self):
        import json
        from subprocess import CompletedProcess

        scenario = SCENARIOS[0]
        room = "timer-hangup-timeout"
        capture = {
            "turns": [{
                "turn": 1,
                "room": room,
                "line": scenario.caller_lines[0],
                "transcript": "Your timer is set for five minutes.",
                "speech_started_at": 100.0,
                "speech_end": 101.0,
                "speech_end_wall": 1_700_000_001.0,
                "first_audio": 101.8,
                "capture_started": 101.4,
                "overlap": False,
                "segments": [{"start": 0.2, "end": 0.6, "text": "Your timer is set for five minutes."}],
                "room_deleted": None,
            }],
            "failure": {
                "turn": 1,
                "message": "room deletion was not observed before deadline",
            },
        }
        phone_log = "".join(json.dumps(entry) + "\n" for entry in (
            {"event": "command", "received_at": 1_700_000_001.5, "command": {
                "id": "fake-timer-command",
                "kind": "timer",
                "seconds": 300,
                "expires_at": "2026-09-26T21:00:00Z",
            }},
            {"event": "result", "result": {
                "id": "fake-timer-command",
                "status": "ok",
                "detail": "Fake phone completed timer",
            }},
        ))
        record = "".join(json.dumps(message) + "\n" for message in (
            {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m1", "model": "claude-opus-5"}}},
            {"type": "assistant", "message": {"content": [{
                "type": "tool_use",
                "id": "timer-tool-1",
                "name": "mcp__mentat__set_timer",
                "input": {"seconds": 300},
            }]}},
            {"type": "result", "session_id": "voice-" + room},
        ))
        delegation_markers = json.dumps({"room": room, "id": "d1", "created_at": 100.5}) + "\n"
        grant = {"token": "header.payload.signature", "room": room, "url": "wss://livekit.invalid"}

        class Stack:
            base_url = "http://127.0.0.1:8485"

            def start_worker(self, _room):
                pass

            def run_voice(self, command, *, token, livekit_url):
                return CompletedProcess(command, 0, json.dumps(capture), "")

            def run_remote(self, command):
                if command == ["sudo", "cat", "voice/evals/phone.jsonl"]:
                    output = phone_log
                elif command == ["sudo", "cat", "voice/evals/delegations.jsonl"]:
                    output = delegation_markers
                elif command == ["sudo", "cat", f"records/voice-{room}.jsonl"]:
                    output = record
                else:
                    raise AssertionError(f"unexpected remote command {command!r}")
                return CompletedProcess(command, 0, output, "")

        with patch.object(runner, "_voice_token", return_value=grant):
            observation = runner.observe_scenario(scenario, Stack())

        self.assertEqual(observation["failure"], capture["failure"])
        self.assertEqual(observation["turns"][0]["transcript"], "Your timer is set for five minutes.")
        self.assertEqual(observation["turns"][0]["model_calls"], [{
            "id": "m1", "model": "claude-opus-5", "service_tier": None,
            "speed": None, "result_service_tier": None,
        }])
        self.assertEqual(observation["phone_commands"], [{
            "id": "fake-timer-command",
            "kind": "timer",
            "seconds": 300,
            "expires_at": "2026-09-26T21:00:00Z",
            "received_at": 1_700_000_001.5,
            "turn": 1,
        }])
        self.assertEqual(observation["product_failures"], [])
        from evals.report import score_observations

        report = score_observations({"cases": [{"name": scenario.name, "runs": [observation]}]}, required_runs=1)
        self.assertFalse(report["passed"])
        failures = " ".join(report["failures"])
        self.assertIn("turn 1: capture failed: room deletion was not observed before deadline", failures)
        self.assertNotIn("invalid partial", failures)
        self.assertEqual(report["cases"][0]["turns"][0]["model_call_count"], 1)

    def test_partial_capture_envelope_is_strict_and_complete_envelopes_stay_unchanged(self):
        import json

        trace = {"turn": 1, "speech_started_at": 1_699_999_999.0}
        partial = json.dumps({
            "turns": [trace],
            "failure": {
                "turn": 2,
                "message": runner.NO_ANSWER_FAILURE,
                "speech_started_at": 1_700_000_000.0,
            },
        })
        self.assertEqual(runner._capture_envelope(partial, 2), (
            [trace],
            {
                "turn": 2,
                "message": runner.NO_ANSWER_FAILURE,
                "speech_started_at": 1_700_000_000.0,
            },
        ))
        first_turn_failure = json.dumps({
            "turns": [],
            "failure": {
                "turn": 1,
                "message": "scripted speech synthesis for line 1 exceeded its deadline",
            },
        })
        self.assertEqual(
            runner._capture_envelope(first_turn_failure, 2),
            ([], {"turn": 1, "message": "scripted speech synthesis for line 1 exceeded its deadline"}),
        )
        started_failure = json.dumps({
            "turns": [],
            "failure": {
                "turn": 1,
                "message": runner.NO_ANSWER_FAILURE,
                "speech_started_at": 1_700_000_000.0,
            },
        })
        self.assertEqual(
            runner._capture_envelope(started_failure, 2)[1]["speech_started_at"],
            1_700_000_000.0,
        )
        same_turn_timeout = json.dumps({
            "turns": [{"turn": 1, "room_deleted": None}],
            "failure": {
                "turn": 1,
                "message": "room deletion was not observed before deadline",
            },
        })
        self.assertEqual(runner._capture_envelope(same_turn_timeout, 1), (
            [{"turn": 1, "room_deleted": None}],
            {"turn": 1, "message": "room deletion was not observed before deadline"},
        ))
        complete = json.dumps({"turns": [trace, {"turn": 2}]})
        self.assertEqual(runner._capture_envelope(complete, 2), ([trace, {"turn": 2}], None))
        for malformed in (
            {"turns": [trace], "failure": {"turn": True, "message": "room was deleted before all scripted lines were captured"}},
            {"turns": [trace], "failure": {"turn": 3, "message": "room was deleted before all scripted lines were captured"}},
            {"turns": [trace], "failure": {"turn": 2, "message": "unrecognized failure"}},
            {"turns": [trace], "failure": {"turn": 2, "message": runner.NO_ANSWER_FAILURE}},
            {"turns": [trace], "failure": {"turn": 2, "message": "scripted speech synthesis exceeded its deadline", "speech_started_at": 1_700_000_000.0}},
            {"turns": [], "failure": {"turn": 1, "message": "room was deleted before all scripted lines were captured"}},
            {"turns": [{"turn": 1, "room_deleted": 4.0}], "failure": {"turn": 1, "message": "room deletion was not observed before deadline"}},
            {"turns": [{"turn": 1, "room_deleted": None}], "failure": {"turn": 1, "message": "room deletion was not observed before deadline", "speech_started_at": 1_700_000_000.0}},
            {"turns": [trace], "failure": {"turn": 2, "message": "room deletion was not observed before deadline"}},
            {"turns": [trace], "failure": None},
            {"turns": [trace], "extra": "not allowed"},
        ):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(
                RuntimeError, "invalid"
            ):
                runner._capture_envelope(json.dumps(malformed), 2)

    def test_remote_cli_emits_partial_capture_envelope_with_zero_completed_turns(self):
        import io
        import json
        from contextlib import redirect_stdout

        async def fail_before_first_turn(_room, _steps, *, room_close_after):
            self.assertEqual(room_close_after, 1)
            raise runner.PartialCaptureFailure(
                [], 1, "scripted speech synthesis for line 1 exceeded its deadline"
            )

        output = io.StringIO()
        with patch.object(runner, "run_remote_capture", fail_before_first_turn), redirect_stdout(output):
            result = runner.main(["android-selected-room", "Question@0::answer"])

        self.assertEqual(result, 1)
        self.assertEqual(json.loads(output.getvalue()), {
            "turns": [],
            "failure": {
                "turn": 1,
                "message": "scripted speech synthesis for line 1 exceeded its deadline",
            },
        })

    def test_missing_or_malformed_evidence_fails_closed(self):
        class Response:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def read(self):
                return b'{"token":"present","url":"wss://livekit.invalid","expires_at":"2026-09-26T22:00:00Z"}'

        with (
            patch.object(runner, "urlopen", return_value=Response()),
            self.assertRaisesRegex(RuntimeError, "missing a valid token or room"),
        ):
            runner._voice_token("http://127.0.0.1:8485")
        with self.assertRaisesRegex(RuntimeError, "recording log is missing or empty"):
            runner._json_lines("", "SDK recording")
        with self.assertRaisesRegex(RuntimeError, "malformed JSON"):
            runner._json_lines("{broken}\n", "SDK recording")
        with self.assertRaisesRegex(RuntimeError, "missing or unmatched command result"):
            runner._phone_commands([
                {"event": "command", "command": {"id": "c1", "kind": "sms"}, "received_at": 1.0},
            ])
        self.assertEqual(runner._recorded_turns([{"type": "system", "subtype": "init"}]), [])
        with self.assertRaisesRegex(RuntimeError, "no transcript segment timestamps"):
            runner._confirmation_time({"capture_started": 5.0, "segments": []})

    def test_run15_confirmation_timestamp_uses_post_readback_request(self):
        from evals import scenarios

        self.assertIs(runner.SMS_CONFIRMATION_PATTERN, scenarios.SMS_CONFIRMATION_PATTERN)
        cases = (
            (
                "Just to confirm, I'm texting 202-  555-0142 saying I will be there at 6.",
                "Say the word, and I'll send it.",
            ),
            (
                "OK, so text him, plus 1, 202-555-0142, saying, I will be there at 6.",
                "Sound right?",
            ),
            (
                "Got it, texting plus one, 202-555-0142. I will be there at 7.",
                "Good to send.",
            ),
        )
        for readback, question in cases:
            with self.subTest(question=question):
                self.assertEqual(
                    runner._confirmation_time({
                        "capture_started": 10.0,
                        "segments": [
                            {"start": 0.0, "end": 0.3, "text": readback},
                            {"start": 0.4, "end": 0.8, "text": question},
                        ],
                    }),
                    10.8,
                )

    def test_run17_confirmation_timing_recognizes_live_prompt_forms_from_shared_pattern(self):
        from evals import scenarios

        self.assertIs(runner.SMS_CONFIRMATION_PATTERN, scenarios.SMS_CONFIRMATION_PATTERN)
        prompts = (
            "Just say when and I'll send it.",
            "Say when and I'll send it.",
            "Sound good?",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    runner._confirmation_time({
                        "capture_started": 10.0,
                        "segments": [
                            {"start": 0.0, "end": 0.3, "text": "Texting 202-555-0142, I will be there at six."},
                            {"start": 0.4, "end": 0.8, "text": prompt},
                        ],
                    }),
                    10.8,
                )

    def test_run17_answer_timing_requires_live_sms_readback_and_confirmation(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        cases = (
            [
                "Texting 202-555-0142, I will be there at six.",
                "Just say when and I'll send it.",
            ],
            [
                "Texting plus 1, 2, 0, 2, 5, 5, 5, 0, 1, 4, 2, I will be there at six.",
                "Say when and I'll send it.",
            ],
            [
                "Texting plus 1202.",
                "5-5-5.",
                "0142. I will be there at six.",
                "Sound good?",
            ],
        )
        for texts in cases:
            with self.subTest(texts=texts):
                self.assertEqual(
                    runner._answer_time(
                        {
                            "capture_started": 10.0,
                            "speech_end": 10.0,
                            "segments": [
                                {
                                    "start": index * 0.2,
                                    "end": index * 0.2 + 0.1,
                                    "text": text,
                                }
                                for index, text in enumerate(texts)
                            ],
                        },
                        scenario.turns[0],
                        scenario_name=scenario.name,
                        turn_index=1,
                    ),
                    10.0 + (len(texts) - 1) * 0.2,
                )

    def test_sms_readback_observed_message_prefixes_receive_answer_timestamps(self):
        cases = (
            (
                "sms-say-back-yes",
                1,
                "I'll text +1-202-555-0142. The exact message is. I will be there at six. Should I send it?",
            ),
            (
                "sms-say-back-yes",
                1,
                "I'll text +1-202-555-0142. Message. I will be there at six. Should I send it?",
            ),
            (
                "sms-say-back-yes",
                1,
                "I'll text +1-202-555-0142. The message reads... I will be there at six. Should I send it?",
            ),
            (
                "sms-correction-new-yes",
                2,
                "I'll text +1-202-555-0142. The exact message is, I will be there at 7. Should I send it?",
            ),
        )
        for scenario_name, turn_index, transcript in cases:
            with self.subTest(transcript=transcript):
                scenario = next(s for s in SCENARIOS if s.name == scenario_name)
                self.assertEqual(
                    runner._answer_time(
                        {
                            "capture_started": 10.0,
                            "speech_end": 10.0,
                            "segments": [{"start": 0.25, "end": 2.0, "text": transcript}],
                        },
                        scenario.turns[turn_index - 1],
                        scenario_name=scenario.name,
                        turn_index=turn_index,
                    ),
                    10.25,
                )

    def test_sms_body_content_connectors_receive_runner_answer_timestamps(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        for body_clause in (
            "The message would say, I will be there at six.",
            "The exact message is. I will be there at six.",
            "Message. I will be there at 6.",
            "The message reads... I will be there at six.",
            "Draft: I will be there at 6.",
            "The words I plan to send are, I will be there at six.",
            "Here is what I will send: I will be there at six.",
            "For your approval, I will be there at six.",
        ):
            transcript = (
                f"I'll text +1-202-555-0142. {body_clause} Should I send it?"
            )
            with self.subTest(body_clause=body_clause):
                self.assertEqual(
                    runner._answer_time(
                        {
                            "capture_started": 10.0,
                            "speech_end": 10.0,
                            "segments": [{"start": 0.25, "end": 2.0, "text": transcript}],
                        },
                        scenario.turns[0],
                        scenario_name=scenario.name,
                        turn_index=1,
                    ),
                    10.25,
                )

        for invalid_body in (
            "The message would say, I will be there at seven.",
            "The message would say, Please tell Alice I will be there at six.",
            "Correction. I will be there at seven.",
            "Please include this too. I will be there at six.",
            "Send pizza instead. I will be there at six.",
            "The message would say,",
        ):
            transcript = (
                f"I'll text +1-202-555-0142. {invalid_body} Should I send it?"
            )
            with self.subTest(invalid_body=invalid_body):
                self.assertIsNone(
                    runner._answer_time(
                        {
                            "capture_started": 10.0,
                            "speech_end": 10.0,
                            "segments": [{"start": 0.25, "end": 2.0, "text": transcript}],
                        },
                        scenario.turns[0],
                        scenario_name=scenario.name,
                        turn_index=1,
                    )
                )

    def test_correction_preamble_is_not_the_exact_body_answer(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        self.assertIsNone(runner._answer_time(
            {
                "capture_started": 10.0,
                "speech_end": 10.0,
                "segments": [{
                    "start": 0.25,
                    "end": 2.0,
                    "text": "I'll text +1-202-555-0142. Correction. I will be there at seven. Should I send it?",
                }],
            },
            scenario.turns[1],
            scenario_name=scenario.name,
            turn_index=2,
        ))

    def test_run17_exact_sms_correction_segments_get_answer_and_confirmation_timestamps(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        trace = {
            "capture_started": 10.0,
            "speech_end": 10.0,
            "segments": [
                {"start": 0.0, "end": 0.8399999737739563, "text": " Okay, hang on."},
                {
                    "start": 0.0,
                    "end": 6.559999942779541,
                    "text": " OK, so to plus 1, 2, 0, 2, 5, 5, 5, 0, 1, 4, 2, I'll say.",
                },
                {"start": 0.0, "end": 2.0, "text": " I will be there at six."},
                {
                    "start": 0.0,
                    "end": 1.600000023841858,
                    "text": " just say when and I'll send it.",
                },
            ],
        }
        self.assertEqual(
            runner._answer_time(
                trace,
                scenario.turns[0],
                scenario_name=scenario.name,
                turn_index=1,
            ),
            10.0,
        )
        self.assertEqual(runner._confirmation_time(trace), 11.600000023841858)

    def test_run17_live_sms_yes_or_no_and_same_number_have_matching_timestamps(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        initial = {
            "capture_started": 20.0,
            "speech_end": 20.0,
            "segments": [
                {"start": 0.1, "end": 4.0, "text": " I've got the number as plus-one-two-zero-two-five-five-five."},
                {"start": 0.2, "end": 1.0, "text": " 0142."},
                {"start": 0.3, "end": 2.0, "text": " And the message is, I will be there at six."},
                {"start": 0.7, "end": 0.8, "text": " Yes or no?"},
                {"start": 0.8, "end": 0.8, "text": " Should I send it?"},
            ],
        }
        correction = {
            "capture_started": 30.0,
            "speech_end": 30.0,
            "segments": [
                {"start": 0.1, "end": 1.0, "text": " Okay."},
                {"start": 0.2, "end": 3.2, "text": " I've updated it to say I will be there at 7."},
                {"start": 0.7, "end": 1.44, "text": " Same number, yes or no?"},
            ],
        }
        self.assertEqual(
            runner._answer_time(initial, scenario.turns[0], scenario_name=scenario.name, turn_index=1),
            20.7,
        )
        self.assertEqual(runner._confirmation_time(initial), 20.8)
        self.assertEqual(
            runner._answer_time(correction, scenario.turns[1], scenario_name=scenario.name, turn_index=2),
            30.7,
        )
        self.assertEqual(runner._confirmation_time(correction), 31.44)

    def test_alice_oil_heiress_funding_answer_gets_live_answer_timestamp(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        answer = (
            "Alice Keck Park was an oil heiress, daughter of Superior Oil founder William Keck. "
            "In the mid-70s, the block was slated for a hotel, and she put up the money "
            "for the city to buy it instead, on the condition it became a public park."
        )
        trace = {
            "capture_started": 40.0,
            "speech_end": 40.0,
            "segments": [
                {"start": 0.2, "end": 8.4, "text": answer},
            ],
        }
        self.assertEqual(
            runner._answer_time(
                trace,
                scenario.turns[0],
                scenario_name=scenario.name,
                turn_index=1,
            ),
            40.2,
        )
        trace["segments"][0]["text"] += " However, John Smith put up the money for the city to buy it instead."
        self.assertIsNone(
            runner._answer_time(
                trace,
                scenario.turns[0],
                scenario_name=scenario.name,
                turn_index=1,
            )
        )

    def test_alice_live_father_answers_receive_answer_timestamps(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        live_answers = (
            "and donated it anonymously. Alice Keck Park was a Santa Barbara philanthropist and an heir to the superior oil fortune through her father, William Keck. Her aunt Carolyn had been married to the son of Albert Herter. Herter was the artist who built the El Mirasol estate on that site. That's likely why the property mattered to her. When she died, she left roughly $20 million to local agencies and causes. I don't have much on her personal life beyond that.",
            "She was a Santa Barbara philanthropist from the Keck Oil family. Her dad, William Keck. founded Superior Oil, and her brother Howard later ran it. She was pretty private and lived quietly in town, but she gave generously to local causes, especially conservation, gardens, and animal welfare. The park's the big thing she's remembered for. Past that, the details of her personal life are thin, so I wouldn't.",
        )
        for answer in live_answers:
            trace = {
                "capture_started": 60.0,
                "speech_end": 60.0,
                "segments": [{"start": 0.4, "end": 12.0, "text": answer}],
            }
            with self.subTest(answer=answer):
                self.assertEqual(
                    runner._answer_time(
                        trace,
                        scenario.turns[1],
                        scenario_name=scenario.name,
                        turn_index=2,
                    ),
                    60.4,
                )

        for answer in (
            "William Keck founded Superior Oil.",
            "Alice Keck's dad was John Smith. William Keck founded Superior Oil.",
            "William Keck was Alice Keck's brother, not her father.",
            "I'm not sure whether William Keck was her father.",
        ):
            trace = {
                "capture_started": 60.0,
                "speech_end": 60.0,
                "segments": [{"start": 0.4, "end": 12.0, "text": answer}],
            }
            with self.subTest(answer=answer):
                self.assertIsNone(
                    runner._answer_time(
                        trace,
                        scenario.turns[1],
                        scenario_name=scenario.name,
                        turn_index=2,
                    )
                )

    def test_run17_live_alice_answer_timestamps_match_sentence_break_and_full_name(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        first = {
            "capture_started": 40.0,
            "speech_end": 40.0,
            "segments": [
                {"start": 0.1, "end": 0.84, "text": " Looking it up."},
                {"start": 0.3, "end": 2.0, "text": " Alice Keck Park."},
                {"start": 0.8, "end": 5.2, "text": " anonymously bought the land and gave it to Santa Barbara for a public garden in 1975."},
            ],
        }
        second = {
            "capture_started": 50.0,
            "speech_end": 50.0,
            "segments": [
                {"start": 0.1, "end": 0.5, "text": " Checking."},
                {"start": 0.5, "end": 5.86, "text": " Alice Keck Park was a Santa Barbara philanthropist and daughter of William Myron Keck, who founded"},
                {"start": 0.7, "end": 6.86, "text": " Superior Oil."},
            ],
        }
        self.assertEqual(
            runner._answer_time(first, scenario.turns[0], scenario_name=scenario.name, turn_index=1),
            40.8,
        )
        self.assertEqual(
            runner._answer_time(second, scenario.turns[1], scenario_name=scenario.name, turn_index=2),
            50.5,
        )

    def test_confirmation_timestamp_recognizes_all_approved_prompt_literals(self):
        prompts = (
            "Should I send it?",
            "Shall I send it?",
            "Want me to send it?",
            "Would you like me to send it?",
            "Say yes to send it.",
            "Say send to confirm.",
            "Say the word and I'll send it.",
            "Confirm, and I'll send it.",
        )
        for prompt in prompts:
            with self.subTest(prompt=prompt):
                self.assertEqual(
                    runner._confirmation_time({
                        "capture_started": 10.0,
                        "segments": [
                            {"start": 0.0, "end": 0.3, "text": "Please confirm."},
                            {"start": 0.4, "end": 0.8, "text": prompt},
                        ],
                    }),
                    10.8,
                )

    def test_later_preflight_tts_failure_accepts_nonzero_capture_without_sdk_record(self):
        import json
        from dataclasses import replace
        from subprocess import CompletedProcess

        from evals.dev_stack import RemoteCommandError
        from evals.report import score_observations

        two_line = SCENARIOS[2]
        three_line = replace(
            two_line,
            name="three-line-preflight",
            caller_lines=two_line.caller_lines + ("Third line.",),
            turns=two_line.turns + (two_line.turns[-1],),
            commands=(),
            room_close_after=None,
        )
        scenarios = (SCENARIOS[0], two_line, three_line)
        grant = {
            "token": "header.payload.signature",
            "room": "room-preflight",
            "url": "wss://livekit.invalid",
            "expires_at": "2026-09-26T22:00:00Z",
        }

        class Stack:
            base_url = "http://127.0.0.1:8485"

            def __init__(self, payload):
                self.payload = payload
                self.remote_commands = []

            def start_worker(self, _room):
                pass

            def run_voice(self, command, **_kwargs):
                raise RemoteCommandError(
                    1,
                    command,
                    output=json.dumps(self.payload),
                    stderr="intentional preflight failure",
                )

            def run_remote(self, command):
                self.remote_commands.append(command)
                if command == ["sudo", "cat", "voice/evals/phone.jsonl"]:
                    return CompletedProcess(command, 0, "", "")
                raise AssertionError(f"SDK recording should not be read: {command!r}")

        for failed_line, scenario in enumerate(scenarios, 1):
            with self.subTest(failed_line=failed_line):
                failure = {
                    "turn": 1,
                    "message": (
                        f"scripted speech synthesis for line {failed_line} "
                        "exceeded its deadline"
                    ),
                }
                stack = Stack({"turns": [], "failure": failure})
                with patch.object(runner, "_voice_token", return_value=grant):
                    observation = runner.observe_scenario(scenario, stack)

                self.assertEqual(observation["failure"], failure)
                self.assertNotIn(
                    ["sudo", "cat", "voice/evals/delegations.jsonl"],
                    stack.remote_commands,
                )
                self.assertFalse(any("records/" in " ".join(command) for command in stack.remote_commands))
                report = score_observations({
                    "cases": [{"name": scenario.name, "runs": [observation]}],
                }, required_runs=1)
                self.assertFalse(report["passed"])
                self.assertEqual(report["cases"][0]["capture_failures"], [{
                    "run": 1,
                    "turn": 1,
                    "message": failure["message"],
                }])

    def test_render_content_envelope_is_reported_as_eval_infrastructure_failure(self):
        import io
        import json
        from contextlib import redirect_stdout
        from subprocess import CompletedProcess

        from evals.dev_stack import RemoteCommandError

        scenario = SCENARIOS[0]
        grant = {
            "token": "header.payload.signature",
            "room": "room-short-render",
            "url": "wss://livekit.invalid",
            "expires_at": "2026-09-26T22:00:00Z",
        }

        class Stack:
            base_url = "http://127.0.0.1:8485"

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def run(self, _run_id):
                return _run_context(self)

            def start_worker(self, _room):
                pass

            def run_voice(self, command, **_kwargs):
                raise RemoteCommandError(
                    1,
                    command,
                    output=json.dumps({
                        "turns": [],
                        "failure": {
                            "turn": 1,
                            "message": (
                                "scripted speech content verification failed for line 1"
                            ),
                        },
                    }),
                    stderr="short synthetic render",
                )

            def run_remote(self, command):
                if command == ["sudo", "cat", "voice/evals/phone.jsonl"]:
                    return CompletedProcess(command, 0, "", "")
                raise AssertionError(f"unexpected remote artifact read: {command!r}")

        output = io.StringIO()
        with (
            patch.object(runner, "DevStack", return_value=Stack()),
            patch.object(runner, "SCENARIOS", (scenario,)),
            patch.object(runner, "_requested_voice_model", return_value="test-model"),
            patch.object(runner, "_voice_token", return_value=grant),
            redirect_stdout(output),
        ):
            exit_code = runner._run_local_eval(["--live", "--runs", "1"])

        report = json.loads(output.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertFalse(report["passed"])
        self.assertTrue(any(
            "eval infrastructure failure" in failure.lower()
            and "line 1" in failure
            for failure in report["failures"]
        ))
        self.assertFalse(any(
            "content verification failed" in product_failure.get("message", "")
            for case in report["cases"]
            for product_failure in case.get("product_failures", [])
        ))

    def test_first_turn_tts_failure_keeps_named_observation_without_sdk_record(self):
        import json
        from subprocess import CompletedProcess

        grant = {
            "token": "header.payload.signature",
            "room": "room-zero-turn",
            "url": "wss://livekit.invalid",
            "expires_at": "2026-09-26T22:00:00Z",
        }

        class Stack:
            base_url = "http://127.0.0.1:8485"

            def __init__(self):
                self.remote_commands = []

            def start_worker(self, _room):
                pass

            def run_voice(self, _command, **_kwargs):
                return CompletedProcess(
                    [],
                    1,
                    json.dumps({
                        "turns": [],
                        "failure": {
                            "turn": 1,
                            "message": "scripted speech synthesis for line 1 exceeded its deadline",
                        },
                    }),
                    "",
                )

            def run_remote(self, command):
                self.remote_commands.append(command)
                if command == ["sudo", "cat", "voice/evals/phone.jsonl"]:
                    return CompletedProcess(command, 0, "", "")
                if command == ["sudo", "cat", "voice/evals/delegations.jsonl"]:
                    return CompletedProcess(command, 0, "", "")
                raise AssertionError(f"SDK recording should not be read: {command!r}")

        stack = Stack()
        with patch.object(runner, "_voice_token", return_value=grant):
            observation = runner.observe_scenario(SCENARIOS[0], stack)

        self.assertEqual(observation["turns"], [])
        self.assertEqual(observation["failure"]["turn"], 1)
        self.assertEqual(
            observation["failure"]["message"],
            "scripted speech synthesis for line 1 exceeded its deadline",
        )
        self.assertNotIn(
            ["sudo", "cat", "records/voice-room-zero-turn.jsonl"],
            stack.remote_commands,
        )

    def test_backend_results_follow_delegation_timing_not_transcript_text(self):
        traces = [
            {
                "turn": 1,
                "speech_started_at": 100.0,
                "speech_end": 101.0,
                "transcript": "What is this garden named for?",
            },
            {
                "turn": 2,
                "speech_started_at": 102.0,
                "speech_end": 103.0,
                "line": "Okay, who was she?",
                "transcript": "Okay who was she?",
            },
        ]
        delegation_log = (
            '{"room":"different-room","id":"old","created_at":1.0}\n'
            '{"room":"android-eval-room","id":"delegation-2","created_at":102.25}\n'
        )
        recorded_turns = [{"model_calls": [{"id": "model-call-1", "model": "claude-opus-5"}]}]

        attributed = runner._attribute_model_calls(
            traces, delegation_log, "android-eval-room", recorded_turns
        )

        self.assertEqual([len(trace["model_calls"]) for trace in attributed], [0, 1])
        self.assertEqual(traces[1]["line"], "Okay, who was she?")
        self.assertEqual(traces[1]["transcript"], "Okay who was she?")

    def test_fake_phone_receipt_timestamp_is_required_finite_and_preserved(self):
        command = {"id": "phone-1", "kind": "sms", "to": "+1-202-555-0142"}
        log = [
            {"event": "command", "command": command, "received_at": 12.0},
            {"event": "result", "result": {"id": "phone-1", "status": "ok"}},
        ]
        parsed = runner._phone_commands(log)
        self.assertEqual(parsed, [{**command, "received_at": 12.0}])
        self.assertEqual(
            runner._match_phone_tools(parsed, [{"phone_tools": [{
                "turn": 2,
                "kind": "sms",
                "input": {"to": "+1-202-555-0142"},
            }]}]),
            [{**command, "received_at": 12.0, "turn": 2}],
        )
        with self.assertRaisesRegex(RuntimeError, "does not match its recorded phone tool"):
            runner._match_phone_tools(parsed, [{"phone_tools": [{
                "turn": 2,
                "kind": "navigate",
                "input": {"to": "+1-202-555-0142"},
            }]}])
        with self.assertRaisesRegex(RuntimeError, "receipt timestamp"):
            runner._phone_commands([
                {"event": "command", "command": command},
                log[1],
            ])
        for invalid in (None, True, float("nan"), "12"):
            malformed = [dict(log[0], received_at=invalid), log[1]]
            with self.subTest(received_at=invalid), self.assertRaisesRegex(RuntimeError, "receipt timestamp"):
                runner._phone_commands(malformed)

    def test_answer_timestamp_is_first_segment_completing_all_expected_patterns(self):
        expectation = SCENARIOS[3].turns[0]
        trace = {
            "capture_started": 100.0,
            "speech_end": 100.4,
            "segments": [
                {"start": 0.1, "end": 0.4, "text": "Text +1-202-555-0142: I will be there at six."},
                {"start": 0.5, "end": 0.9, "text": "Should I send it?"},
                {"start": 1.0, "end": 1.4, "text": "Anything else?"},
            ],
        }
        self.assertEqual(
            runner._answer_time(
                trace, expectation, scenario_name=SCENARIOS[3].name, turn_index=1
            ),
            100.5,
        )
        self.assertIsNone(runner._answer_time({
            "capture_started": 100.0,
            "speech_end": 100.0,
            "segments": [{"start": 0.1, "end": 0.4, "text": "I will text that."}],
        }, expectation, scenario_name=SCENARIOS[3].name, turn_index=1))
        with self.assertRaisesRegex(RuntimeError, "answer segment timestamp"):
            runner._answer_time({
                "capture_started": 100.0,
                "speech_end": 100.0,
                "segments": [{"start": 0.1, "end": float("nan"), "text": "Text +1-202-555-0142"}],
            }, expectation, scenario_name=SCENARIOS[3].name, turn_index=1)

    def test_answer_timestamp_uses_first_post_speech_end_utterance_with_prior_overlap(self):
        trace = {
            "capture_started": 100.0,
            "speech_end": 101.0,
            "segments": [
                {"start": 0.8, "end": 1.0, "text": "Timer set for"},
                {"start": 1.2, "end": 1.4, "text": "5 minutes."},
            ],
        }
        self.assertEqual(
            runner._answer_time(
                trace,
                SimpleNamespace(
                    answer_patterns=(r"Timer set for", r"5 minutes"),
                    reject_patterns=(),
                ),
            ),
            101.2,
        )

    def test_answer_timestamp_skips_unverified_preamble_and_sms_confirmation_filler(self):
        alice_scenario = SCENARIOS[5]
        alice_trace = {
            "capture_started": 100.0,
            "speech_end": 100.0,
            "segments": [
                {
                    "start": 1.0,
                    "end": 5.0,
                    "text": "I am not sure whether Alice Keck gave the land to the city.",
                },
                {
                    "start": 12.0,
                    "end": 17.0,
                    "text": "I checked. Alice Keck Park purchased the land and gave it to the city.",
                },
            ],
        }
        self.assertEqual(
            runner._answer_time(
                alice_trace,
                alice_scenario.turns[0],
                scenario_name=alice_scenario.name,
                turn_index=1,
            ),
            112.0,
        )

        sms_scenario = SCENARIOS[3]
        sms_trace = {
            "capture_started": 100.0,
            "speech_end": 100.0,
            "segments": [
                {"start": 1.0, "end": 2.0, "text": "Should I send it?"},
                {
                    "start": 12.0,
                    "end": 15.0,
                    "text": "Text +1-202-555-0142: I will be there at six. Should I send it?",
                },
            ],
        }
        self.assertEqual(
            runner._answer_time(
                sms_trace,
                sms_scenario.turns[0],
                scenario_name=sms_scenario.name,
                turn_index=1,
            ),
            112.0,
        )

    def test_delegation_jsonl_is_room_scoped_and_rejects_malformed_or_local_duplicates(self):
        records = (
            '{"room":"room-a","id":"same","created_at":10.0}\n'
            '{"room":"room-b","id":"same","created_at":11.0}\n'
        )
        self.assertEqual(
            runner._eval_delegations(records, "room-a"),
            [{"id": "same", "created_at": 10.0}],
        )
        with self.assertRaisesRegex(RuntimeError, "repeated delegation id"):
            runner._eval_delegations(
                records + '{"room":"room-a","id":"same","created_at":12.0}\n',
                "room-a",
            )
        with self.assertRaisesRegex(RuntimeError, "malformed JSON"):
            runner._eval_delegations('{broken}\n', "room-a")

    def test_partial_capture_attributes_calls_using_wall_clock_turn_starts_only(self):
        traces = [{
            "turn": 1,
            "speech_started_at": 1_700_000_000.0,
            "first_audio": 190_000.0,
        }]
        voice_log = (
            '{"room":"room-a","id":"d1","created_at":1700000000.5}\n'
            '{"room":"room-a","id":"d2","created_at":1700000002.5}\n'
        )
        recorded_turns = [
            {"model_calls": [{
            "id": "m1", "model": "claude-opus-5", "service_tier": None,
            "speed": None, "result_service_tier": None, "fast_mode_state": None,
        }], "phone_tools": []},
            {"model_calls": [{"id": "m2", "model": "claude-opus-5"}], "phone_tools": []},
        ]

        post_failure_calls = []
        runner._attribute_model_calls(
            traces,
            voice_log,
            "room-a",
            recorded_turns,
            partial_capture=True,
            failure_started_at=1_700_000_002.0,
            failure_turn=2,
            unattributed_model_calls=post_failure_calls,
        )

        self.assertEqual(post_failure_calls, [{
            "id": "m2", "model": "claude-opus-5", "turn": 2,
        }])
        self.assertEqual(traces[0]["model_calls"], [{
            "id": "m1", "model": "claude-opus-5", "service_tier": None,
            "speed": None, "result_service_tier": None, "fast_mode_state": None,
        }])

    def test_partial_failed_turn_keeps_tool_attribution_without_backend_metric(self):
        traces = [{
            "turn": 1,
            "speech_started_at": 1_700_000_000.0,
            "first_audio": 190_000.0,
        }]
        voice_log = (
            '{"room":"room-a","id":"d1","created_at":1700000000.5}\n'
            '{"room":"room-a","id":"d2","created_at":1700000002.5}\n'
        )
        recorded_turns = [
            {"model_calls": [{
            "id": "m1", "model": "claude-opus-5", "service_tier": None,
            "speed": None, "result_service_tier": None, "fast_mode_state": None,
        }], "phone_tools": []},
            {
                "model_calls": [{"id": "m2", "model": "claude-opus-5"}],
                "phone_tools": [{"turn": 2, "kind": "navigate"}],
            },
        ]

        post_failure_calls = []
        runner._attribute_model_calls(
            traces,
            voice_log,
            "room-a",
            recorded_turns,
            partial_capture=True,
            failure_started_at=1_700_000_002.0,
            failure_turn=2,
            unattributed_model_calls=post_failure_calls,
        )

        self.assertEqual(post_failure_calls[0]["turn"], 2)
        self.assertEqual(recorded_turns[1]["phone_tools"][0]["turn"], 2)
        self.assertEqual(traces[0]["model_calls"], [{
            "id": "m1", "model": "claude-opus-5", "service_tier": None,
            "speed": None, "result_service_tier": None, "fast_mode_state": None,
        }])
        self.assertEqual(recorded_turns[1]["phone_tools"][0]["turn"], 2)

    def test_unmatched_or_ambiguous_backend_evidence_fails_closed(self):
        traces = [
            {"turn": 1, "speech_started_at": 100.0, "speech_end": 101.0},
            {"turn": 2, "speech_started_at": 102.0, "speech_end": 103.0},
        ]
        call = {"model_calls": [{"id": "model-call-1", "model": "claude-opus-5"}]}

        with self.assertRaisesRegex(RuntimeError, "delegation and SDK result counts differ"):
            runner._attribute_model_calls(
                traces[:1],
                "",
                "room-a",
                [call],
            )
        with self.assertRaisesRegex(RuntimeError, "delegation and SDK result counts differ"):
            runner._attribute_model_calls(
                traces[:1],
                '{"room":"room-a","id":"d1","created_at":100.5}\n',
                "room-a",
                [],
            )
        with self.assertRaisesRegex(RuntimeError, "delegation and SDK result counts differ"):
            runner._attribute_model_calls(
                traces,
                '{"room":"room-a","id":"d1","created_at":99.0}\n',
                "room-a",
                [call],
            )
        with self.assertRaisesRegex(RuntimeError, "delegation and SDK result counts differ"):
            runner._attribute_model_calls(
                traces,
                '{"room":"room-a","id":"d1","created_at":100.5}\n',
                "room-a",
                [call, call],
            )
        with self.assertRaisesRegex(RuntimeError, "ambiguous delegation timestamp"):
            runner._attribute_model_calls(
                traces,
                '{"room":"room-a","id":"d1","created_at":102.0}\n',
                "room-a",
                [call],
            )

    def test_only_the_two_sms_scenarios_request_private_audio_retention(self):
        for scenario in SCENARIOS:
            command = runner._capture_command(scenario, "safe-room", ["Question@0::answer"])
            self.assertIn("--retain-caller-audio", command)
            if scenario.name in {"sms-say-back-yes", "sms-correction-new-yes"}:
                with self.subTest(scenario=scenario.name):
                    self.assertEqual(
                        command[command.index("--retain-sms-audio") + 1],
                        scenario.name,
                    )
            else:
                with self.subTest(scenario=scenario.name):
                    self.assertNotIn("--retain-sms-audio", command)

    def test_cli_passes_private_retention_only_for_named_sms_scenario(self):
        import io
        import json
        import tempfile
        from contextlib import redirect_stdout

        calls = []

        async def capture(room, steps, **options):
            calls.append((room, steps, options))
            return []

        with tempfile.TemporaryDirectory() as temporary:
            output = io.StringIO()
            with (
                patch.dict(os.environ, {
                    **TEST_VOICE_ENV,
                    "MENTAT_EVAL_RETAINED_EVIDENCE_DIR": temporary,
                }),
                patch.object(runner, "run_remote_capture", capture),
                redirect_stdout(output),
            ):
                result = runner.main([
                    "--retain-caller-audio", "--retain-sms-audio", "sms-say-back-yes",
                    "android-selected-room", "Question@0::answer",
                ])

        self.assertEqual(result, 0)
        self.assertEqual(calls, [(
            "android-selected-room",
            ["Question@0::answer"],
            {
                "room_close_after": 1,
                "retain_sms_audio_dir": Path(temporary),
                "retain_sms_audio_scenario": "sms-say-back-yes",
                "retain_caller_audio_dir": Path(temporary),
            },
        )])
        self.assertNotIn("filename", json.loads(output.getvalue()))

    def test_one_scenario_posts_token_runs_fake_capture_and_observes_turn_evidence(self):
        import http.server
        import json
        import threading
        from subprocess import CompletedProcess

        token_requests = []
        grant = {
            "token": "header.payload.signature",
            "room": "android-eval-room",
            "url": "wss://livekit.invalid",
            "expires_at": "2026-09-26T22:00:00Z",
        }

        class TokenHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                token_requests.append((self.command, self.path, self.rfile.read(int(self.headers["Content-Length"]))))
                payload = json.dumps(grant).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = http.server.HTTPServer(("127.0.0.1", 0), TokenHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        assistant_turns = [
            {
                "turn": 1,
                "room": grant["room"],
                "line": "Text +1-202-555-0142: I will be there at six.",
                "transcript": "Text +1-202-555-0142: I will be there at six. Should I send it?",
                "speech_started_at": 99.0,
                "speech_end": 99.5,
                "speech_end_wall": 1_700_000_000.5,
                "first_audio": 100.2,
                "overlap": False,
                "capture_started": 100.0,
                "segments": [
                    {"start": 0.2, "end": 0.5, "text": "Text +1-202-555-0142: I will be there at six."},
                    {"start": 0.6, "end": 1.0, "text": "Should I send it?"},
                ],
                "room_deleted": None,
            },
            {
                "turn": 2,
                "room": grant["room"],
                "line": "Yes.",
                "transcript": "Message sent to +1-202-555-0142.",
                "speech_started_at": 200.0,
                "speech_end": 201.0,
                "speech_end_wall": 1_700_000_001.0,
                "first_audio": 201.2,
                "overlap": False,
                "capture_started": 201.0,
                "segments": [{"start": 0.2, "end": 0.6, "text": "Message sent to +1-202-555-0142."}],
                "room_deleted": 203.0,
            },
        ]
        phone_log = "".join(
            json.dumps(entry) + "\n"
            for entry in (
                {
                    "event": "command",
                    "received_at": 1_700_000_002.0,
                    "command": {
                        "id": "phone-id-independent-of-tool-id",
                        "kind": "sms",
                        "to": "+1-202-555-0142",
                        "body": "I will be there at six.",
                        "expires_at": "2026-09-26T21:00:00Z",
                    },
                },
                {
                    "event": "result",
                    "result": {
                        "id": "phone-id-independent-of-tool-id",
                        "status": "ok",
                        "detail": "Fake phone completed sms",
                    },
                },
            )
        )
        record = [
            {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m1", "model": "claude-opus-5"}}},
            {"type": "result", "session_id": "voice-android-eval-room"},
            {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m2", "model": "claude-opus-5"}}},
            {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m3", "model": "claude-opus-5"}}},
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "sdk-tool-id-does-not-match-phone-id",
                            "name": "mcp__mentat__send_sms",
                            "input": {"to": "+1-202-555-0142", "body": "I will be there at six.", "send": True},
                        }
                    ]
                },
            },
            {"type": "result", "session_id": "voice-android-eval-room"},
        ]
        record_text = "".join(json.dumps(message) + "\n" for message in record)
        delegation_markers = (
            '{"room":"old-room","id":"previous","created_at":1.0}\n'
            '{"room":"android-eval-room","id":"d1","created_at":99.25}\n'
            '{"room":"android-eval-room","id":"d2","created_at":200.25}\n'
        )

        class Stack:
            base_url = f"http://127.0.0.1:{server.server_port}"

            def __init__(self):
                self.worker_rooms = []
                self.voice_commands = []
                self.voice_grants = []
                self.remote_commands = []

            def start_worker(self, room):
                self.worker_rooms.append(room)

            def run_voice(self, command, *, token, livekit_url):
                self.voice_commands.append(command)
                self.voice_grants.append((token, livekit_url))
                if command[0] == "evals/runner.py":
                    return CompletedProcess(command, 0, json.dumps({"turns": assistant_turns}), "")
                raise AssertionError(f"unexpected remote voice command {command!r}")

            def run_remote(self, command):
                self.remote_commands.append(command)
                if command == ["sudo", "cat", "voice/evals/phone.jsonl"]:
                    return CompletedProcess(command, 0, phone_log, "")
                if command == ["sudo", "cat", "voice/evals/delegations.jsonl"]:
                    return CompletedProcess(command, 0, delegation_markers, "")
                if command == ["sudo", "cat", "records/voice-android-eval-room.jsonl"]:
                    return CompletedProcess(command, 0, record_text, "")
                raise AssertionError(f"unexpected remote command {command!r}")

        stack = Stack()
        try:
            observation = runner.observe_scenario(SCENARIOS[3], stack)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(token_requests, [("POST", "/v1/voice/token", b"{}")])
        self.assertEqual(stack.worker_rooms, [grant["room"]])
        self.assertEqual(stack.voice_grants, [(grant["token"], grant["url"])])
        self.assertNotIn(grant["token"], repr(observation))
        self.assertEqual(stack.voice_commands[0][0], "evals/runner.py")
        self.assertIn(grant["room"], stack.voice_commands[0])
        self.assertIn("--fake-phone", stack.voice_commands[0])
        self.assertEqual(observation["turns"][0]["confirmation"], 101.0)
        self.assertNotEqual(observation["turns"][0]["confirmation"], observation["turns"][0]["first_audio"])
        self.assertEqual(observation["turns"][0]["kind"], "search")
        self.assertIsNone(observation["turns"][0]["command_received_at"])
        self.assertEqual(observation["turns"][0]["answer_at"], 100.6)
        self.assertEqual(observation["turns"][1]["kind"], "action")
        self.assertEqual(observation["turns"][1]["command_received_at"], 1_700_000_002.0)
        self.assertEqual(observation["turns"][1]["answer_at"], 201.2)
        self.assertEqual([len(turn["model_calls"]) for turn in observation["turns"]], [1, 2])
        self.assertEqual(observation["phone_commands"][0]["turn"], 2)
        self.assertEqual(observation["phone_commands"][0]["id"], "phone-id-independent-of-tool-id")
        self.assertEqual(observation["room_closed_after"], 2)


class LocalEvalCliTests(unittest.TestCase):
    def test_concurrent_batch_schedules_full_cartesian_product_with_indexed_timing(self):
        import io
        import json
        import threading
        from contextlib import contextmanager, redirect_stdout

        scenarios = (SimpleNamespace(name="first", turns=(), commands=(), place_query=None),
                     SimpleNamespace(name="second", turns=(), commands=(), place_query=None))
        lifecycle = []
        entered = []
        active = 0
        maximum_active = 0
        lock = threading.Lock()
        first_pair = threading.Barrier(2)
        completed = []
        captured = {}
        original_score = runner.score_observations

        def capture_score(observations, *, required_runs):
            captured["observations"] = observations
            return original_score(observations, required_runs=required_runs)

        class Batch:
            @contextmanager
            def run(self, run_id):
                entered.append(run_id)
                yield SimpleNamespace(base_url="http://127.0.0.1:8485", run_id=run_id)

        @contextmanager
        def dev_stack(**kwargs):
            lifecycle.append(("enter", kwargs["opt_in"]))
            try:
                yield Batch()
            finally:
                lifecycle.append(("exit",))

        def observe(_scenario, stack):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            if stack.run_id in ("case-1-run-1", "case-1-run-2"):
                first_pair.wait(timeout=2)
            time.sleep(0.06 if stack.run_id == "case-1-run-1" else 0.005)
            with lock:
                active -= 1
                completed.append(stack.run_id)
            return {"run_id": stack.run_id, "turns": []}

        output = io.StringIO()
        with (
            patch.object(runner, "SCENARIOS", scenarios),
            patch.object(runner, "DevStack", side_effect=dev_stack),
            patch.object(runner, "observe_scenario", side_effect=observe),
            patch.object(runner, "score_observations", side_effect=capture_score),
            redirect_stdout(output),
        ):
            result = runner._run_local_eval(["--live", "--runs", "2", "--concurrency", "2"])

        report = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertEqual(lifecycle, [("enter", True), ("exit",)])
        self.assertEqual(set(entered), {
            "case-1-run-1", "case-1-run-2", "case-2-run-1", "case-2-run-2",
        })
        self.assertEqual(maximum_active, 2)
        self.assertLess(completed.index("case-1-run-2"), completed.index("case-1-run-1"))
        observations = captured["observations"]
        self.assertEqual(
            [[run["run_id"] for run in case["runs"]] for case in observations["cases"]],
            [["case-1-run-1", "case-1-run-2"], ["case-2-run-1", "case-2-run-2"]],
        )
        self.assertEqual(report["batch_timing"]["concurrency_cap"], 2)
        self.assertLessEqual(report["batch_timing"]["started_at"], report["batch_timing"]["ended_at"])
        self.assertGreaterEqual(report["batch_timing"]["wall_seconds"], 0)
        for case in report["cases"]:
            timings = case["run_timings"]
            self.assertEqual([item["run"] for item in timings], [1, 2])
            for timing in timings:
                self.assertLessEqual(timing["started_at"], timing["ended_at"])
                self.assertGreaterEqual(timing["concurrency"], 1)
                self.assertLessEqual(timing["concurrency"], 2)

    def test_concurrency_default_and_positive_override_validation(self):
        self.assertEqual(runner.DEFAULT_CONCURRENCY, 16)
        self.assertEqual(runner._parse_local_eval_arguments(["--live", "--runs", "1"])["concurrency"], 16)
        self.assertEqual(runner._parse_local_eval_arguments(
            ["--live", "--runs", "1", "--concurrency", "3"]
        )["concurrency"], 3)
        for invalid in ("0", "-1", "not-a-number"):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                runner._parse_local_eval_arguments(["--live", "--concurrency", invalid])

    def test_harness_error_stops_queued_launches_and_keeps_original_positions(self):
        import io
        import json
        import threading
        from contextlib import contextmanager, redirect_stdout

        scenario = SimpleNamespace(name="abort", turns=(), commands=(), place_query=None)
        entered = []
        second_started = threading.Event()
        stack_exited = threading.Event()
        captured = {}
        original_score = runner.score_observations

        def capture_score(observations, *, required_runs):
            captured["observations"] = observations
            return original_score(observations, required_runs=required_runs)

        class Batch:
            @contextmanager
            def run(self, run_id):
                entered.append(run_id)
                yield SimpleNamespace(base_url="http://127.0.0.1:8485", run_id=run_id)

        @contextmanager
        def dev_stack(**_kwargs):
            try:
                yield Batch()
            finally:
                stack_exited.set()

        def observe(_scenario, stack):
            if stack.run_id.endswith("run-1"):
                self.assertTrue(second_started.wait(timeout=2))
                raise RuntimeError("harness exploded")
            second_started.set()
            self.assertTrue(stack_exited.wait(timeout=2))
            return {"run_id": stack.run_id, "turns": []}

        output = io.StringIO()
        with (
            patch.object(runner, "SCENARIOS", (scenario,)),
            patch.object(runner, "DevStack", side_effect=dev_stack),
            patch.object(runner, "observe_scenario", side_effect=observe),
            patch.object(runner, "score_observations", side_effect=capture_score),
            redirect_stdout(output),
        ):
            runner._run_local_eval(["--live", "--runs", "4", "--concurrency", "2"])

        report = json.loads(output.getvalue())
        self.assertEqual(set(entered), {"case-1-run-1", "case-1-run-2"})
        runs = captured["observations"]["cases"][0]["runs"]
        self.assertEqual([run.get("run_id") for run in runs], [None, "case-1-run-2", None, None])
        self.assertIn("harness exploded", runs[0]["failure"])
        self.assertEqual([run["failure"] for run in runs[2:]], [
            "aborted before launch: case-1-run-1 failed",
            "aborted before launch: case-1-run-1 failed",
        ])

    def test_sigterm_stops_queued_launches_when_dev_stack_overrides_signal_handler(self):
        import concurrent.futures
        import io
        import json
        import signal
        import threading
        from contextlib import redirect_stdout

        scenario = SimpleNamespace(name="signal", turns=(), commands=(), place_query=None)
        launch_calls = []
        launch_entered = threading.Event()
        release_enter = threading.Event()
        batch_exited = threading.Event()
        handlers = {}
        real_as_completed = concurrent.futures.as_completed
        captured = {}
        original_score = runner.score_observations

        def capture_score(observations, *, required_runs):
            captured["observations"] = observations
            return original_score(observations, required_runs=required_runs)

        def fake_signal(signum, handler):
            if callable(handler):
                handlers[signum] = handler
            return signal.SIG_DFL

        test_case = self

        class Batch:
            def run(self, run_id):
                launch_calls.append(run_id)

                class Run:
                    def __enter__(self):
                        launch_entered.set()
                        if run_id.endswith("run-1"):
                            test_case.assertTrue(release_enter.wait(timeout=2))
                        return SimpleNamespace(run_id=run_id)

                    def __exit__(self, *_args):
                        return False

                return Run()

        @contextmanager
        def dev_stack(**_kwargs):
            # DevStack installs its own signal handler after the runner's handler.
            def dev_stack_handler(_signum, _frame):
                raise KeyboardInterrupt("received signal 15")

            handlers[signal.SIGTERM] = dev_stack_handler
            try:
                yield Batch()
            finally:
                batch_exited.set()
                release_enter.set()

        def observe(_scenario, stack):
            self.assertTrue(batch_exited.is_set())
            return {"run_id": stack.run_id, "turns": []}

        def interrupted_as_completed(futures):
            self.assertTrue(launch_entered.wait(timeout=2))
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            yield from real_as_completed(futures)

        output = io.StringIO()
        with (
            patch.object(runner, "SCENARIOS", (scenario,)),
            patch.object(runner, "DevStack", side_effect=dev_stack),
            patch.object(runner, "observe_scenario", side_effect=observe),
            patch.object(runner, "score_observations", side_effect=capture_score),
            patch.object(runner.signal, "signal", side_effect=fake_signal),
            patch.object(runner.concurrent.futures, "as_completed", side_effect=interrupted_as_completed),
            redirect_stdout(output),
        ):
            runner._run_local_eval(["--live", "--runs", "4", "--concurrency", "2"])

        self.assertEqual(launch_calls, ["case-1-run-1"])
        runs = captured["observations"]["cases"][0]["runs"]
        self.assertEqual(runs[0]["run_id"], "case-1-run-1")
        self.assertTrue(all("aborted before launch" in run["failure"] for run in runs[1:]))

    def test_list_prints_all_scenarios_without_starting_dev_stack(self):
        output = []
        with patch.object(runner, "DevStack", create=True) as dev_stack, patch(
            "builtins.print", side_effect=lambda *args, **_kwargs: output.append(args[0])
        ):
            result = runner.main(["eval", "--list"])

        self.assertEqual(result, 0)
        self.assertEqual(output, [scenario.name for scenario in SCENARIOS])
        dev_stack.assert_not_called()

    def test_live_eval_repeats_every_scenario_and_prints_strict_report(self):
        from contextlib import contextmanager
        import json

        lifecycle = []
        calls = []

        @contextmanager
        def dev_stack(**kwargs):
            lifecycle.append(("enter", kwargs["opt_in"]))
            try:
                yield _batch_for(SimpleNamespace(base_url="http://127.0.0.1:8485"))
            finally:
                lifecycle.append(("exit",))

        def observe(scenario, _stack):
            calls.append(scenario.name)
            turns = []
            for index, expectation in enumerate(scenario.turns, 1):
                turns.append({
                    "kind": runner._turn_kind(scenario, index),
                    "speech_end": 10.0,
                    "speech_end_wall": 10.0,
                    "first_audio": 11.0,
                    "capture_started": 10.25,
                    "segments": [{"start": 0.75, "end": 1.0, "text": "Synthetic assistant response."}],
                    "command_received_at": 11.0 if runner._turn_kind(scenario, index) == "action" else None,
                    "answer_at": 11.0,
                    "overlap": False,
                    "expect_confirmation": expectation.sms_recipient is not None,
                    "confirmation": 12.0 if expectation.sms_recipient is not None else None,
                    "expect_hangup": scenario.room_close_after == index,
                    "room_deleted": 13.0 if scenario.room_close_after == index else None,
                    "model_calls": [{
                        "id": f"synthetic-{index}",
                        "model": "claude-opus-5-5",
                        "service_tier": "standard",
                        "result_service_tier": "standard",
                        "speed": "standard",
                        "fast_mode_state": "off",
                    }],
                })
            observation = {"turns": turns}
            if scenario.name == "spanish-language-switch":
                observation["first_spanish_lookup_ms"] = 24.5
            if scenario.name == "spanish-interpreter":
                observation["phone_commands"] = []
            return observation

        output = []
        with patch.dict(os.environ, {"MENTAT_VOICE_MODEL": "claude-opus-5-5"}), patch.object(
            runner, "DevStack", side_effect=dev_stack, create=True
        ), patch.object(
            runner, "observe_scenario", side_effect=observe
        ), patch("builtins.print", side_effect=lambda *args, **_kwargs: output.append(args[0])):
            result = runner.main(["eval", "--live", "--runs", "2"])

        self.assertEqual(result, 0)
        self.assertEqual(
            sorted(calls),
            sorted(scenario.name for scenario in SCENARIOS for _ in range(2)),
        )
        self.assertEqual(lifecycle, [("enter", True), ("exit",)])
        report = json.loads(output[0])
        self.assertTrue(report["passed"])
        self.assertEqual(
            [turn["kind"] for turn in report["cases"][2]["turns"][:2]],
            ["search", "action"],
        )
        self.assertEqual(
            [turn["kind"] for turn in report["cases"][5]["turns"][:3]],
            ["search", "search", "search"],
        )
        self.assertTrue(all(case["run_count"] == 2 for case in report["cases"]))
        self.assertTrue(all(gate["run_count"] == 2 for case in report["cases"] for gate in case["gates"]))
        self.assertIn("latency_seconds", report["cases"][0]["turns"][0])
        self.assertIn("model_call_count", report["cases"][0]["turns"][0])
        self.assertEqual(report["cases"][0]["turns"][0]["latency_seconds"]["room_deleted"], 3.0)
        self.assertEqual(
            report["cases"][6]["first_spanish_lookup_ms"], [24.5, 24.5]
        )
        self.assertEqual(report["cases"][7]["phone_commands"], [[], []])

    def test_concurrent_identical_phone_receipts_keep_times_after_reverse_completion(self):
        import io
        import json
        import threading
        from contextlib import contextmanager, redirect_stdout

        scenario = SimpleNamespace(
            name="isolated-phone",
            turns=(SimpleNamespace(answer_patterns=("timer set",), sms_recipient=None),),
            commands=({"turn": 1, "kind": "timer", "seconds": 300},),
            place_query=None,
            room_close_after=None,
        )
        run_ids = []
        completed = []
        first_run_receipts = []
        first_receipt_pending = threading.Event()
        second_run_exited = threading.Event()
        second_finished_with_first_pending = []
        captured = {}
        original_score = runner.score_observations

        class Batch:
            @contextmanager
            def run(self, run_id):
                run_ids.append(run_id)
                try:
                    yield SimpleNamespace(run_id=run_id)
                finally:
                    completed.append(run_id)
                    if run_id.endswith("run-2"):
                        second_finished_with_first_pending.append(
                            first_receipt_pending.is_set() and not first_run_receipts
                        )
                        second_run_exited.set()

        @contextmanager
        def batch_context(**_kwargs):
            yield Batch()

        def observe(_scenario, stack):
            run_one = stack.run_id.endswith("run-1")
            if run_one:
                first_receipt_pending.set()
                self.assertTrue(second_run_exited.wait(timeout=2))
                base = 10.0
                receipt_at = 10.5
            else:
                base = 20.0
                receipt_at = 20.5
            if run_one:
                first_run_receipts.append(receipt_at)
            return {
                "turns": [{
                    "kind": "action",
                    "speech_end": base,
                    "speech_end_wall": base,
                    "first_audio": base + 0.5,
                    "capture_started": base + 0.1,
                    "segments": [{"start": 0.2, "end": 0.4, "text": "Timer set for five minutes."}],
                    "command_received_at": receipt_at,
                    "answer_at": base + 1.0,
                    "overlap": False,
                    "expect_confirmation": False,
                    "confirmation": None,
                    "expect_hangup": False,
                    "room_deleted": None,
                    "model_calls": [{
                        "id": stack.run_id,
                        "model": "gpt-6-sol",
                        "service_tier": None,
                        "result_service_tier": "standard",
                        "speed": "standard",
                        "fast_mode_state": "off",
                    }],
                }],
                "phone_commands": [{
                    "id": "same-command-id",
                    "kind": "timer",
                    "turn": 1,
                    "received_at": receipt_at,
                }],
            }

        def capture_score(observations, *, required_runs):
            captured["observations"] = observations
            captured["required_runs"] = required_runs
            return original_score(observations, required_runs=required_runs)

        output = io.StringIO()
        with (
            patch.object(runner, "SCENARIOS", (scenario,)),
            patch.object(runner, "DevStack", side_effect=batch_context),
            patch.object(runner, "observe_scenario", side_effect=observe),
            patch.object(runner, "score_observations", side_effect=capture_score),
            patch.object(runner, "_requested_voice_model", return_value="chatgpt/sol-fast"),
            redirect_stdout(output),
        ):
            result = runner._run_local_eval(["--live", "--runs", "2", "--concurrency", "2"])

        report = json.loads(output.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(set(run_ids), {"case-1-run-1", "case-1-run-2"})
        self.assertEqual(completed, ["case-1-run-2", "case-1-run-1"])
        self.assertEqual(second_finished_with_first_pending, [True])
        runs = captured["observations"]["cases"][0]["runs"]
        self.assertEqual(
            [[receipt["id"], receipt["received_at"]] for run in runs
             for receipt in run["phone_commands"]],
            [["same-command-id", 10.5], ["same-command-id", 20.5]],
        )
        self.assertEqual(captured["required_runs"], 2)
        self.assertTrue(report["passed"])

    def test_live_eval_retains_interpreter_phone_commands_and_fails_on_timer(self):
        from contextlib import contextmanager
        import json

        scenario = SimpleNamespace(
            name="spanish-interpreter",
            turns=tuple(
                SimpleNamespace(sms_recipient=None, sms_body=None)
                for _ in range(6)
            ),
            room_close_after=6,
            commands=(),
            place_query=None,
        )
        lifecycle = []
        run_commands = [
            [],
            [{"id": "fake-timer", "kind": "timer", "turn": 4}],
        ]
        observations = []

        @contextmanager
        def dev_stack(**kwargs):
            lifecycle.append(("enter", kwargs["opt_in"]))
            try:
                yield _batch_for(SimpleNamespace(base_url="http://127.0.0.1:8485"))
            finally:
                lifecycle.append(("exit",))

        def observe(_scenario, _stack):
            index = len(observations)
            observations.append(index)
            phone_commands = run_commands[index]
            product_failures = []
            return {
                "turns": [
                    {
                        "kind": "search",
                        "speech_end": float(turn * 10),
                        "speech_end_wall": float(turn * 10),
                        "first_audio": float(turn * 10 + 0.5),
                        "capture_started": float(turn * 10 + 0.1),
                        "segments": [{"start": 0.2, "end": 0.4, "text": "Synthetic reply."}],
                        "command_received_at": None,
                        "answer_at": float(turn * 10 + 1),
                        "overlap": False,
                        "expect_confirmation": False,
                        "confirmation": None,
                        "expect_hangup": turn == 6,
                        "room_deleted": float(turn * 10 + 2) if turn == 6 else None,
                        "model_calls": [],
                    }
                    for turn in range(1, 7)
                ],
                "phone_commands": phone_commands,
                "product_failures": product_failures,
            }

        output = []
        with patch.dict(os.environ, {"MENTAT_VOICE_MODEL": "claude-opus-5-5"}), patch.object(
            runner, "SCENARIOS", [scenario]
        ), patch.object(
            runner, "DevStack", side_effect=dev_stack, create=True
        ), patch.object(
            runner, "observe_scenario", side_effect=observe
        ), patch(
            "builtins.print", side_effect=lambda *args, **_kwargs: output.append(args[0])
        ):
            result = runner.main(["eval", "--live", "--runs", "2", "--concurrency", "1"])

        report = json.loads(output[0])
        self.assertEqual(result, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["cases"][0]["phone_commands"],
            [[], [{"id": "fake-timer", "kind": "timer", "turn": 4}]],
        )
        self.assertTrue(any("fake phone command log is not empty" in failure for failure in report["failures"]))
        self.assertEqual(lifecycle, [("enter", True), ("exit",)])

    def test_live_eval_requires_opt_in_and_aborts_after_harness_failure(self):
        from contextlib import contextmanager
        import json

        calls = []
        lifecycle = []

        @contextmanager
        def dev_stack(**kwargs):
            lifecycle.append(("enter", kwargs["opt_in"]))
            try:
                yield _batch_for(SimpleNamespace(base_url="http://127.0.0.1:8485"))
            finally:
                lifecycle.append(("exit",))

        output = []
        with patch.object(runner, "DevStack", side_effect=AssertionError("stack started"), create=True), patch(
            "builtins.print", side_effect=lambda *args, **_kwargs: output.append(args[0])
        ):
            result = runner.main(["eval", "--runs", "1"])
        self.assertNotEqual(result, 0)
        self.assertEqual(output, ["eval requires --live; no DevStack was started"])

        def observe(scenario, _stack):
            calls.append(scenario.name)
            if scenario is SCENARIOS[0]:
                raise RuntimeError("deliberate scenario failure")
            return {"turns": [
                {
                    "kind": runner._turn_kind(scenario, index),
                    "speech_end": 10.0,
                    "speech_end_wall": 10.0,
                    "first_audio": 11.0,
                    "capture_started": 10.25,
                    "segments": [{"start": 0.75, "end": 1.0, "text": "Synthetic assistant response."}],
                    "command_received_at": 11.0 if runner._turn_kind(scenario, index) == "action" else None,
                    "answer_at": 11.0,
                    "overlap": False,
                    "expect_confirmation": expectation.sms_recipient is not None,
                    "confirmation": 12.0 if expectation.sms_recipient is not None else None,
                    "expect_hangup": scenario.room_close_after == index,
                    "room_deleted": 13.0 if scenario.room_close_after == index else None,
                    "model_calls": [],
                }
                for index, expectation in enumerate(scenario.turns, 1)
            ]}

        output.clear()
        with patch.object(runner, "DevStack", side_effect=dev_stack, create=True), patch.object(
            runner, "observe_scenario", side_effect=observe
        ), patch("builtins.print", side_effect=lambda *args, **_kwargs: output.append(args[0])):
            result = runner.main(["eval", "--live", "--runs", "1", "--concurrency", "1"])
        self.assertEqual(result, 1)
        self.assertEqual(calls, [SCENARIOS[0].name])
        self.assertEqual(lifecycle, [("enter", True), ("exit",)])
        report = json.loads(output[0])
        self.assertFalse(report["passed"])
        self.assertTrue(any("deliberate scenario failure" in failure for failure in report["failures"]))
    def test_complete_product_failure_exits_nonzero_with_turn_metrics(self):
        import io
        import json
        from contextlib import contextmanager, redirect_stdout

        scenario = SCENARIOS[2]
        observations = {
            "turns": [
                {
                    "turn": 1,
                    "kind": "search",
                    "speech_end": 10.0,
                    "speech_end_wall": 10.0,
                    "first_audio": 11.0,
                    "capture_started": 10.25,
                    "segments": [{"start": 0.75, "end": 1.0, "text": "Synthetic search response."}],
                    "command_received_at": None,
                    "answer_at": 11.0,
                    "overlap": False,
                    "expect_confirmation": False,
                    "confirmation": None,
                    "expect_hangup": False,
                    "room_deleted": None,
                    "model_calls": [{
            "id": "m1", "model": "claude-opus-5", "service_tier": None,
            "speed": None, "result_service_tier": None, "fast_mode_state": None,
        }],
                },
                {
                    "turn": 2,
                    "kind": "action",
                    "speech_end": 20.0,
                    "speech_end_wall": 20.0,
                    "first_audio": 21.5,
                    "capture_started": 20.25,
                    "segments": [{"start": 1.0, "end": 1.2, "text": "Synthetic action response."}],
                    "command_received_at": 21.5,
                    "answer_at": 21.5,
                    "overlap": False,
                    "expect_confirmation": False,
                    "confirmation": None,
                    "expect_hangup": True,
                    "room_deleted": 23.0,
                    "model_calls": [{"id": "m2", "model": "claude-opus-5"}],
                },
            ],
            "phone_commands": [{"id": "navigate-1", "kind": "navigate", "turn": 2}],
            "room_closed_after": 2,
            "product_failures": [{
                "turn": 2,
                "message": "place-search-navigation: navigation selected an unexpected place name 'Wrong Park'",
            }],
        }

        @contextmanager
        def dev_stack(**_kwargs):
            yield _batch_for(SimpleNamespace(base_url="http://127.0.0.1:8485"))

        output = io.StringIO()
        with (
            patch.object(runner, "DevStack", side_effect=dev_stack),
            patch.object(runner, "SCENARIOS", (scenario,)),
            patch.object(runner, "observe_scenario", return_value=observations),
            redirect_stdout(output),
        ):
            result = runner._run_local_eval(["--live", "--runs", "1"])

        report = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["cases"][0]["turns"]), 2)
        self.assertEqual(report["cases"][0]["turns"][1]["model_call_count"], 1)
        self.assertEqual(report["cases"][0]["turns"][1]["latency_seconds"]["first_audio"], 1.5)
        self.assertTrue(any("unexpected place name" in failure for failure in report["failures"]))

    def test_partial_case_is_scored_and_fails_eval_with_named_turn(self):
        import io
        import json
        from contextlib import contextmanager, redirect_stdout

        @contextmanager
        def dev_stack(**_kwargs):
            yield _batch_for(SimpleNamespace(base_url="http://127.0.0.1:8485"))

        def observe(scenario, _stack):
            turns = []
            for index, expectation in enumerate(scenario.turns, 1):
                turns.append({
                    "kind": runner._turn_kind(scenario, index),
                    "speech_end": 10.0,
                    "speech_end_wall": 10.0,
                    "first_audio": 11.0,
                    "capture_started": 10.25,
                    "segments": [{"start": 0.75, "end": 1.0, "text": "Synthetic assistant response."}],
                    "command_received_at": 11.0 if runner._turn_kind(scenario, index) == "action" else None,
                    "answer_at": 11.0,
                    "overlap": False,
                    "expect_confirmation": expectation.sms_recipient is not None,
                    "confirmation": 12.0 if expectation.sms_recipient is not None else None,
                    "expect_hangup": scenario.room_close_after == index,
                    "room_deleted": 13.0 if scenario.room_close_after == index else None,
                    "model_calls": [{"model": "claude-opus-5"}],
                })
            if scenario is SCENARIOS[2]:
                return {
                    "turns": turns[:1],
                    "room_closed_after": 1,
                    "product_failures": [{
                        "turn": 1,
                        "message": "call ended after turn 1 with 1 follow-ups remaining",
                    }],
                }
            return {"turns": turns}

        output = io.StringIO()
        with (
            patch.object(runner, "DevStack", side_effect=dev_stack),
            patch.object(runner, "observe_scenario", side_effect=observe),
            redirect_stdout(output),
        ):
            result = runner.main(["eval", "--live", "--runs", "1"])

        report = json.loads(output.getvalue())
        self.assertEqual(result, 1)
        self.assertFalse(report["passed"])
        case_report = report["cases"][2]
        self.assertEqual(len(case_report["turns"]), 1)
        self.assertEqual(case_report["turns"][0]["model_call_count"], 1)
        self.assertEqual(case_report["turns"][0]["latency_seconds"]["first_audio"], 1.0)
        failures = " ".join(case_report["failures"])
        self.assertIn("place-search-navigation run 1 turn 1: product failure", failures)
        self.assertIn("call ended after turn 1 with 1 follow-ups remaining", failures)
        self.assertNotIn("capture failed", failures)

    def test_room_closing_before_followups_is_a_product_failure_with_completed_evidence(self):
        import json
        from subprocess import CompletedProcess

        from evals.report import score_observations

        scenario = SimpleNamespace(
            name="three-turn-call",
            caller_lines=("Set a timer.", "What time is it?", "Thanks."),
            turns=tuple(
                SimpleNamespace(
                    answer_patterns=(r"Timer set",) if index == 1 else (r"answer",),
                    reject_patterns=(),
                    sms_recipient=None,
                    sms_body=None,
                )
                for index in range(3)
            ),
            commands=({"turn": 1, "kind": "timer", "seconds": 300},),
            room_close_after=3,
            place_query=None,
        )
        room = "early-close-three-turn"
        capture = {
            "turns": [{
                "turn": 1,
                "room": room,
                "line": scenario.caller_lines[0],
                "transcript": "Timer set for five minutes.",
                "speech_started_at": 100.0,
                "speech_end": 101.0,
                "speech_end_wall": 1_700_000_001.0,
                "first_audio": 102.0,
                "capture_started": 101.5,
                "overlap": False,
                "segments": [{"start": 0.2, "end": 0.6, "text": "Timer set for five minutes."}],
                "room_deleted": None,
            }],
            "failure": {
                "turn": 2,
                "message": "room was deleted before all scripted lines were captured",
            },
        }
        phone_log = "".join(json.dumps(entry) + "\n" for entry in (
            {"event": "command", "received_at": 1_700_000_001.5, "command": {
                "id": "timer-command", "kind": "timer", "seconds": 300,
            }},
            {"event": "result", "result": {"id": "timer-command", "status": "ok"}},
        ))
        record = "".join(json.dumps(message) + "\n" for message in (
            {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "m1", "model": "claude-opus-5"}}},
            {"type": "assistant", "message": {"content": [{
                "type": "tool_use", "id": "tool1", "name": "mcp__mentat__set_timer",
                "input": {"seconds": 300},
            }]}},
            {"type": "result", "session_id": "voice-" + room},
        ))
        delegation_markers = json.dumps({"room": room, "id": "d1", "created_at": 100.5}) + "\n"
        grant = {"token": "header.payload.signature", "room": room, "url": "wss://livekit.invalid"}

        class Stack:
            base_url = "http://127.0.0.1:8485"

            def start_worker(self, _room):
                pass

            def run_voice(self, _command, **_kwargs):
                return CompletedProcess([], 0, json.dumps(capture), "")

            def run_remote(self, command):
                outputs = {
                    "voice/evals/phone.jsonl": phone_log,
                    "voice/evals/delegations.jsonl": delegation_markers,
                    f"records/voice-{room}.jsonl": record,
                }
                return CompletedProcess(command, 0, outputs[command[-1]], "")

        with patch.object(runner, "_voice_token", return_value=grant):
            observation = runner.observe_scenario(scenario, Stack())

        self.assertEqual(len(observation["turns"]), 1)
        self.assertEqual(observation["turns"][0]["model_calls"], [{
            "id": "m1", "model": "claude-opus-5", "service_tier": None,
            "speed": None, "result_service_tier": None,
        }])
        self.assertEqual(observation["phone_commands"][0]["turn"], 1)
        self.assertEqual(observation["room_closed_after"], 1)
        self.assertNotIn("failure", observation)
        self.assertTrue(any(
            failure["turn"] == 1
            and "call ended after turn 1 with 2 follow-ups remaining" in failure["message"]
            for failure in observation["product_failures"]
        ))

        report = score_observations(
            {"cases": [{"name": scenario.name, "runs": [observation]}]},
            required_runs=1,
        )
        self.assertFalse(report["passed"])
        failures = " ".join(report["failures"])
        self.assertIn("product failure", failures)
        self.assertIn("call ended after turn 1 with 2 follow-ups remaining", failures)
        self.assertNotIn("capture failed", failures)
        self.assertEqual(report["cases"][0]["turns"][0]["latency_seconds"]["first_audio"], 1.0)
        self.assertEqual(report["cases"][0]["turns"][0]["model_call_count"], 1)


    def test_cancelled_delegation_pairs_with_skipped_execution_error_result(self):
        messages = []
        for message_id in ("msg-1", "msg-2"):
            messages.extend([
                {
                    "type": "stream_event",
                    "event": {
                        "type": "message_start",
                        "message": {"id": message_id, "model": "claude-opus-5"},
                    },
                },
                {"type": "result"},
            ])
        messages.extend([
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": "Synthetic cancelled delegation.",
            },
            {
                "type": "stream_event",
                "event": {
                    "type": "message_start",
                    "message": {"id": "msg-4", "model": "claude-opus-5"},
                },
            },
            {"type": "result"},
        ])
        recorded_turns = runner._recorded_turns(messages)
        traces = [
            {"turn": index, "speech_started_at": float(100 + 2 * (index - 1))}
            for index in range(1, 5)
        ]
        markers = "".join(
            json.dumps({
                "room": "room-a",
                "id": f"d{index}",
                "created_at": float(100 + 2 * (index - 1)) + 0.5,
            }) + "\n"
            for index in range(1, 5)
        )

        runner._attribute_model_calls(traces, markers, "room-a", recorded_turns)

        self.assertEqual(
            [[call["id"] for call in trace["model_calls"]] for trace in traces],
            [["msg-1"], ["msg-2"], [], ["msg-4"]],
        )
        with self.assertRaisesRegex(RuntimeError, "delegation and SDK result counts differ"):
            runner._attribute_model_calls(
                traces,
                markers + json.dumps({
                    "room": "room-a", "id": "d5", "created_at": 108.5,
                }) + "\n",
                "room-a",
                recorded_turns,
            )

    def test_recorded_turns_skip_aborted_call_free_result_and_keep_completed_turns(self):
        messages = [
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": "Synthetic aborted turn.",
            },
        ]
        for index in range(1, 4):
            messages.extend([
                {
                    "type": "stream_event",
                    "event": {
                        "type": "message_start",
                        "message": {
                            "id": f"msg-{index}",
                            "model": "claude-opus-5-5",
                            "usage": {"service_tier": "standard"},
                        },
                    },
                },
                {
                    "type": "result",
                    "usage": {"speed": "standard", "service_tier": "standard"},
                },
            ])

        recorded = runner._recorded_turns(messages)

        self.assertEqual(
            [turn["model_calls"][0]["id"] for turn in recorded],
            ["msg-1", "msg-2", "msg-3"],
        )

    def test_recorded_turns_still_reject_successful_result_without_model_start(self):
        with self.assertRaisesRegex(RuntimeError, "no message_start model-call evidence"):
            runner._recorded_turns([{"type": "result", "usage": {"service_tier": "standard"}}])
        with self.assertRaisesRegex(RuntimeError, "no message_start model-call evidence"):
            runner._recorded_turns([{
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": False,
            }])

    def test_recorded_turns_reject_trailing_call_free_execution_error(self):
        with self.assertRaisesRegex(RuntimeError, "no message_start model-call evidence"):
            runner._recorded_turns([{
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
            }])

    def test_recorded_turns_capture_model_and_standard_speed_service_evidence(self):
        messages = [
            {
                "type": "stream_event",
                "event": {
                    "type": "message_start",
                    "message": {
                        "id": "msg-proof",
                        "model": "claude-opus-5",
                        "usage": {"service_tier": "standard"},
                    },
                },
            },
            {
                "type": "result",
                "usage": {"speed": "standard", "service_tier": "standard"},
                "fast_mode_state": "off",
            },
        ]

        recorded = runner._recorded_turns(messages)

        self.assertEqual(recorded[0]["model_calls"], [{
            "id": "msg-proof",
            "model": "claude-opus-5",
            "service_tier": "standard",
            "speed": "standard",
            "result_service_tier": "standard",
            "fast_mode_state": "off",
        }])

    def test_recorded_sonnet_call_does_not_invent_fast_mode_evidence(self):
        messages = [
            {
                "type": "stream_event",
                "event": {
                    "type": "message_start",
                    "message": {
                        "id": "msg-sonnet",
                        "model": "claude-sonnet-5-5",
                        "usage": {"service_tier": "standard"},
                    },
                },
            },
            {
                "type": "result",
                "usage": {"service_tier": "standard", "speed": "standard"},
            },
        ]

        call = runner._recorded_turns(messages)[0]["model_calls"][0]
        provenance = runner._model_call_provenance(call, "claude-sonnet-5-5")

        self.assertNotIn("fast_mode_state", call)
        self.assertEqual(provenance["failures"], [])

    def test_sol_recorded_shape_proves_service_from_result_usage(self):
        messages = [
            {
                "type": "stream_event",
                "event": {
                    "type": "message_start",
                    "message": {
                        "id": "msg-sol",
                        "model": "gpt-6-sol",
                        "usage": {"input_tokens": 32},
                    },
                },
            },
            {
                "type": "result",
                "usage": {"service_tier": "standard", "speed": "standard"},
                "fast_mode_state": "off",
            },
        ]

        call = runner._recorded_turns(messages)[0]["model_calls"][0]
        provenance = runner._model_call_provenance(call, "chatgpt/sol-fast")

        self.assertIsNone(call["service_tier"])
        self.assertEqual(call["result_service_tier"], "standard")
        self.assertEqual(provenance["failures"], [])

    def test_printed_report_includes_per_request_provenance_and_fails_unproven_usage(self):
        import io
        import json
        from contextlib import redirect_stdout

        class Stack:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def run(self, _run_id):
                return _run_context(self)

        scenario = SimpleNamespace(name="synthetic-provenance")
        base_turn = {
            "kind": "search",
            "speech_end": 10.0,
            "speech_end_wall": 10.0,
            "first_audio": 11.0,
            "answer_at": 11.0,
            "command_received_at": None,
            "capture_started": 10.25,
            "segments": [{"start": 0.75, "end": 1.0, "text": "Synthetic provenance response."}],
            "overlap": False,
            "expect_confirmation": False,
            "confirmation": None,
            "expect_hangup": False,
            "room_deleted": None,
        }
        observations = (
            ({
                "id": "msg-ok",
                "model": "claude-opus-5-5",
                "service_tier": "standard",
                "result_service_tier": "standard",
                "speed": "standard",
                "fast_mode_state": "off",
            }, True),
            ({
                "id": "msg-unproven",
                "model": "claude-opus-5-5",
                "service_tier": "standard",
                "result_service_tier": "standard",
                "fast_mode_state": "off",
            }, False),
            ({
                "id": "msg-no-model",
                "model": None,
                "service_tier": "standard",
                "result_service_tier": "standard",
                "speed": "standard",
                "fast_mode_state": "off",
            }, False),
        )
        for call, expected_pass in observations:
            with self.subTest(call=call):
                output = io.StringIO()
                with (
                    patch.dict(os.environ, {"MENTAT_VOICE_MODEL": "claude-opus-5-5"}),
                    patch.object(runner, "SCENARIOS", [scenario]),
                    patch.object(runner, "DevStack", return_value=Stack()),
                    patch.object(runner, "observe_scenario", return_value={
                        "turns": [{**base_turn, "model_calls": [call]}],
                        "unattributed_model_calls": [{"turn": 2, **call}],
                    }),
                    redirect_stdout(output),
                ):
                    result = runner.main(["eval", "--live", "--runs", "1"])

                report = json.loads(output.getvalue())
                self.assertEqual(report["requested_model"], "claude-opus-5-5")
                self.assertEqual(report["passed"], expected_pass)
                self.assertEqual(result, 0 if expected_pass else 1)
                provenance = report["cases"][0]["turns"][0]["model_provenance"]
                self.assertEqual(provenance["requested_model"], "claude-opus-5-5")
                self.assertEqual(provenance["model_calls"][0]["observed_model"], call.get("model"))
                self.assertEqual(provenance["model_calls"][0]["speed"], call.get("speed"))
                if call.get("model") is None:
                    self.assertIn("observed model is unproven", " ".join(report["failures"]))
                self.assertEqual(provenance["model_calls"][0]["service_tier"], "standard")
                self.assertEqual(provenance["verified"], expected_pass)
                extra = report["cases"][0]["unattributed_model_calls"][0]["model_calls"][0]
                self.assertEqual(extra["turn"], 2)
                self.assertEqual(extra["observed_model"], call.get("model"))

    def test_model_provenance_report_fails_mismatched_or_unproven_requests(self):
        requested = "claude-opus-5-5"
        wrong_model = runner._model_call_provenance({
            "id": "msg-wrong",
            "model": "claude-sonnet-5",
            "service_tier": "standard",
            "speed": "standard",
            "result_service_tier": "standard",
            "fast_mode_state": "off",
        }, requested)
        unproven_speed = runner._model_call_provenance({
            "id": "msg-unknown",
            "model": "claude-opus-5-5",
            "service_tier": "standard",
            "result_service_tier": "standard",
            "fast_mode_state": "off",
        }, requested)
        fast_mode = runner._model_call_provenance({
            "id": "msg-fast",
            "model": "claude-opus-5-5",
            "service_tier": "standard",
            "speed": "fast",
            "result_service_tier": "standard",
            "fast_mode_state": "on",
        }, requested)
        priority_tier = runner._model_call_provenance({
            "id": "msg-priority",
            "model": "claude-opus-5-5",
            "service_tier": "priority",
            "speed": "standard",
            "result_service_tier": "priority",
            "fast_mode_state": "off",
        }, requested)

        self.assertIn("observed claude-sonnet-5", " ".join(wrong_model["failures"]))
        self.assertIn("speed", " ".join(unproven_speed["failures"]))
        self.assertIn("standard", " ".join(fast_mode["failures"]))
        self.assertIn("service tier", " ".join(priority_tier["failures"]))
        dated_alias = runner._model_call_provenance({
            "id": "msg-snapshot",
            "model": "claude-opus-5-5-20260901",
            "service_tier": "standard",
            "speed": "standard",
            "result_service_tier": "standard",
            "fast_mode_state": "off",
        }, requested)
        self.assertEqual(dated_alias["failures"], [])


    def test_requested_model_comes_from_a_safe_configured_arm(self):
        for model in ("chatgpt/sol-fast", "claude-opus-5-5", "claude-sonnet-5-5"):
            with self.subTest(model=model), patch.dict(os.environ, {"MENTAT_VOICE_MODEL": model}):
                self.assertEqual(runner._requested_voice_model(), model)
        for model in ("claude-opus-5", "claude-sonnet-5"):
            with self.subTest(model=model), patch.dict(os.environ, {"MENTAT_VOICE_MODEL": model}):
                with self.assertRaisesRegex(RuntimeError, "unsupported requested voice model"):
                    runner._requested_voice_model()

    def test_sonnet_model_provenance_requires_observed_model_and_standard_tier_and_speed(self):
        requested = "claude-sonnet-5-5"
        exact = runner._model_call_provenance({
            "id": "msg-sonnet",
            "model": requested,
            "service_tier": "standard",
            "result_service_tier": "standard",
            "speed": "standard",
            "fast_mode_state": "off",
        }, requested)
        dated_alias = runner._model_call_provenance({
            "id": "msg-sonnet-snapshot",
            "model": "claude-sonnet-5-5-20260915",
            "service_tier": "standard",
            "result_service_tier": "standard",
            "speed": "standard",
        }, requested)
        mismatch = runner._model_call_provenance({
            "id": "msg-other-model",
            "model": "claude-sonnet-5",
            "service_tier": "standard",
            "result_service_tier": "standard",
            "speed": "standard",
        }, requested)
        missing_speed = runner._model_call_provenance({
            "id": "msg-no-speed",
            "model": requested,
            "service_tier": "standard",
            "result_service_tier": "standard",
        }, requested)
        missing_tier = runner._model_call_provenance({
            "id": "msg-no-tier",
            "model": requested,
            "service_tier": "standard",
            "speed": "standard",
        }, requested)
        missing_start_tier = runner._model_call_provenance({
            "id": "msg-no-start-tier",
            "model": requested,
            "result_service_tier": "standard",
            "speed": "standard",
        }, requested)
        fast_mode_on = runner._model_call_provenance({
            "id": "msg-fast-mode",
            "model": requested,
            "service_tier": "standard",
            "result_service_tier": "standard",
            "speed": "standard",
            "fast_mode_state": "on",
        }, requested)

        self.assertEqual(exact["failures"], [])
        self.assertEqual(dated_alias["failures"], [])
        self.assertIn("observed claude-sonnet-5", " ".join(mismatch["failures"]))
        self.assertIn("speed", " ".join(missing_speed["failures"]))
        self.assertIn("service tier", " ".join(missing_tier["failures"]))
        self.assertIn("service tier", " ".join(missing_start_tier["failures"]))
        self.assertIn("fast mode", " ".join(fast_mode_on["failures"]))

    def test_partial_capture_keeps_post_failure_backend_request_provenance(self):
        traces = [{
            "turn": 1,
            "speech_started_at": 100.0,
            "speech_end": 101.0,
        }]
        recorded_turns = [
            {"model_calls": [{"id": "msg-1", "model": "gpt-6-sol"}], "phone_tools": []},
            {"model_calls": [{"id": "msg-2", "model": "gpt-6-sol"}], "phone_tools": []},
        ]
        post_failure_calls = []

        runner._attribute_model_calls(
            traces,
            '{"room":"room-a","id":"d1","created_at":100.5}\n'
            '{"room":"room-a","id":"d2","created_at":101.5}\n',
            "room-a",
            recorded_turns,
            partial_capture=True,
            failure_started_at=101.0,
            failure_turn=2,
            unattributed_model_calls=post_failure_calls,
        )

        self.assertEqual(
            post_failure_calls,
            [{"turn": 2, "id": "msg-2", "model": "gpt-6-sol"}],
        )

    def test_sol_service_tier_uses_result_usage_while_opus_requires_start_proof(self):
        sol = runner._model_call_provenance({
            "id": "msg-sol",
            "model": "gpt-6-sol",
            "service_tier": None,
            "result_service_tier": "standard",
            "speed": "standard",
            "fast_mode_state": "off",
        }, "chatgpt/sol-fast")
        opus_without_start_tier = runner._model_call_provenance({
            "id": "msg-opus",
            "model": "claude-opus-5-5-20260901",
            "service_tier": None,
            "result_service_tier": "standard",
            "speed": "standard",
            "fast_mode_state": "off",
        }, "claude-opus-5-5")

        self.assertEqual(sol["failures"], [])
        self.assertIn("service tier", " ".join(opus_without_start_tier["failures"]))


if __name__ == "__main__":
    unittest.main()
