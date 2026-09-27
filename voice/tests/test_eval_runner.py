"""Offline tests for remote scripted caller capture."""

import asyncio
import os
import sys
import time
import unittest
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


class Clock:
    def __init__(self):
        self.now = 10.0

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds


class RunnerTests(unittest.IsolatedAsyncioTestCase):
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
            remote_participants = {"agent": object()}
            local_participant = SimpleNamespace(
                publish_track=lambda *_args, **_kwargs: asyncio.sleep(0)
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
                return b"audio", 24000, 1, clock.monotonic() + 0.1

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
        )

        async def tts(_http, text):
            events.append(f"tts:{text}")
            return b"\x00\x00"

        async def transcribe(_http, pcm, _rate, _channels):
            self.assertEqual(pcm, b"audio")
            return [{"start": 0.1, "end": 0.4, "text": "answer"}]

        dependencies = CaptureDependencies(
            api=api,
            rtc=rtc,
            http=object(),
            tts=tts,
            transcribe=transcribe,
            capture_factory=ContinuousCapture,
            monotonic=clock.monotonic,
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
        self.assertTrue(all(item["speech_end"] < item["first_audio"] for item in traces))
        self.assertIsNone(traces[0]["room_deleted"])
        self.assertGreaterEqual(traces[1]["room_deleted"], traces[1]["speech_end"])
        self.assertEqual(traces[1]["room"], "android-selected-room")
        self.assertLess(events.index("capture-start"), events.index("speech-frame"))
        self.assertLess(events.index("playout"), events.index("capture-result"))
        self.assertEqual(events[-1], "disconnect")

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

    async def test_capture_records_early_audio_overlap_and_full_trace(self):
        dependencies = self.dependencies_for_failure("other")

        class EarlyCapture:
            def __init__(self, _queue, _ended=None):
                pass

            async def start(self):
                pass

            async def result(self):
                return b"audio", 24000, 1, 99.0

        async def transcribe(*_args):
            return [
                {"start": 0.2, "end": 0.4, "text": "Okay, I heard you."},
                {"start": 1.2, "end": 1.6, "text": "The timer is set for five minutes."},
            ]

        dependencies = CaptureDependencies(
            **{
                **dependencies.__dict__,
                "capture_factory": EarlyCapture,
                "transcribe": transcribe,
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
        self.assertEqual(trace["transcript"], "Okay, I heard you. The timer is set for five minutes.")
        self.assertEqual(trace["segments"][0]["start"], 0.2)
        self.assertEqual(trace["first_audio"], 100.2)
        self.assertEqual(trace["speech_end"], 100.0)
        self.assertTrue(trace["overlap"])

    async def test_capture_fails_when_no_agent_audio_follows_speech_end(self):
        dependencies = self.dependencies_for_failure("other")

        class EarlyOnlyCapture:
            def __init__(self, _queue, _ended=None):
                pass

            async def start(self):
                pass

            async def result(self):
                return b"audio", 24000, 1, 99.0

        async def transcribe(*_args):
            return [{"start": 0.2, "end": 0.4, "text": "Okay, I heard you."}]

        dependencies = CaptureDependencies(
            **{
                **dependencies.__dict__,
                "capture_factory": EarlyOnlyCapture,
                "transcribe": transcribe,
            }
        )

        async def speech_end(_source):
            return 100.0

        with (
            patch.dict(os.environ, TEST_VOICE_ENV),
            patch.object(runner.caller, "_speech_end_after_playout", speech_end),
            self.assertRaisesRegex(RuntimeError, "no agent audio after speech end"),
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

    async def test_missing_audio_transcript_and_room_deletion_fail_closed(self):
        for failure, expected in (
            ("audio", "no frames"),
            ("capture", "exceeded its deadline"),
            ("transcript", "empty transcript"),
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

    @staticmethod
    def dependencies_for_failure(failure):
        class Room:
            remote_participants = {"agent": object()}
            callbacks = {}
            local_participant = SimpleNamespace(
                publish_track=lambda *_args, **_kwargs: asyncio.sleep(0)
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
                return b"audio", 24000, 1, time.monotonic() + 0.1

        alice_responses = iter((
            "Alice Keck Park in Santa Barbara was donated by Alice Keck.",
            "W. M. Keck was her father.",
            "Her family's wealth came from Superior Oil.",
        ))

        async def transcribe(*_args):
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
            tts=lambda *_args: asyncio.sleep(0, result=b"\x00\x00"),
            transcribe=transcribe,
            capture_factory=Capture,
            monotonic=time.monotonic,
            sleep=asyncio.sleep,
        )


class LocalEvalTests(unittest.TestCase):
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
            ["action", "action"],
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
                {"event": "command", "command": {"id": "c1", "kind": "sms"}},
            ])
        self.assertEqual(runner._recorded_turns([{"type": "system", "subtype": "init"}]), [])
        with self.assertRaisesRegex(RuntimeError, "no transcript segment timestamps"):
            runner._confirmation_time({"capture_started": 5.0, "segments": []})

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
            'INFO mentat.voice: eval-delegation {"id":"old","created_at":1.0}\n'
            '2026-09-26T22:00:00 INFO mentat.voice: '
            'eval-delegation {"id":"delegation-2","created_at":102.25}\n'
        )
        recorded_turns = [{"model_calls": [{"id": "model-call-1", "model": "claude-opus-5"}]}]

        attributed = runner._attribute_model_calls(traces, delegation_log, recorded_turns)

        self.assertEqual([len(trace["model_calls"]) for trace in attributed], [0, 1])
        self.assertEqual(traces[1]["line"], "Okay, who was she?")
        self.assertEqual(traces[1]["transcript"], "Okay who was she?")

    def test_unmatched_or_ambiguous_backend_evidence_fails_closed(self):
        traces = [
            {"turn": 1, "speech_started_at": 100.0, "speech_end": 101.0},
            {"turn": 2, "speech_started_at": 102.0, "speech_end": 103.0},
        ]
        call = {"model_calls": [{"id": "model-call-1", "model": "claude-opus-5"}]}

        with self.assertRaisesRegex(RuntimeError, "delegation and SDK result counts differ"):
            runner._attribute_model_calls(
                traces,
                'eval-delegation {"id":"d1","created_at":99.0}\n',
                [call],
            )
        with self.assertRaisesRegex(RuntimeError, "delegation and SDK result counts differ"):
            runner._attribute_model_calls(
                traces,
                'eval-delegation {"id":"d1","created_at":100.5}\n',
                [call, call],
            )
        with self.assertRaisesRegex(RuntimeError, "ambiguous delegation timestamp"):
            runner._attribute_model_calls(
                traces,
                'eval-delegation {"id":"d1","created_at":102.0}\n',
                [call],
            )

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
        voice_log = (
            'INFO mentat.voice: eval-delegation {"id":"previous","created_at":1.0}\n'
            'INFO mentat.voice: eval-delegation {"id":"d1","created_at":99.25}\n'
            'INFO mentat.voice: eval-delegation {"id":"d2","created_at":200.25}\n'
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
                if command == ["sudo", "cat", "voice.log"]:
                    return CompletedProcess(command, 0, voice_log, "")
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
        self.assertEqual([len(turn["model_calls"]) for turn in observation["turns"]], [1, 2])
        self.assertEqual(observation["phone_commands"][0]["turn"], 2)
        self.assertEqual(observation["phone_commands"][0]["id"], "phone-id-independent-of-tool-id")
        self.assertEqual(observation["room_closed_after"], 2)


class LocalEvalCliTests(unittest.TestCase):
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
                yield SimpleNamespace(base_url="http://127.0.0.1:8485")
            finally:
                lifecycle.append(("exit",))

        def observe(scenario, _stack):
            calls.append(scenario.name)
            turns = []
            for index, expectation in enumerate(scenario.turns, 1):
                turns.append({
                    "kind": "search" if not scenario.commands or (scenario.place_query and index == 1) else "action",
                    "speech_end": 10.0,
                    "first_audio": 11.0,
                    "overlap": False,
                    "expect_confirmation": expectation.sms_recipient is not None,
                    "confirmation": 12.0 if expectation.sms_recipient is not None else None,
                    "expect_hangup": scenario.room_close_after == index,
                    "room_deleted": 13.0 if scenario.room_close_after == index else None,
                    "model_calls": [],
                })
            return {"turns": turns}

        output = []
        with patch.object(runner, "DevStack", side_effect=dev_stack, create=True), patch.object(
            runner, "observe_scenario", side_effect=observe
        ), patch("builtins.print", side_effect=lambda *args, **_kwargs: output.append(args[0])):
            result = runner.main(["eval", "--live", "--runs", "2"])

        self.assertEqual(result, 0)
        self.assertEqual(calls, [scenario.name for scenario in SCENARIOS for _ in range(2)])
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

    def test_live_eval_requires_opt_in_and_continues_after_case_failure(self):
        from contextlib import contextmanager
        import json

        calls = []
        lifecycle = []

        @contextmanager
        def dev_stack(**kwargs):
            lifecycle.append(("enter", kwargs["opt_in"]))
            try:
                yield SimpleNamespace(base_url="http://127.0.0.1:8485")
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
                    "kind": "search" if not scenario.commands or (scenario.place_query and index == 1) else "action",
                    "speech_end": 10.0,
                    "first_audio": 11.0,
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
            result = runner.main(["eval", "--live", "--runs", "1"])
        self.assertEqual(result, 1)
        self.assertEqual(calls, [scenario.name for scenario in SCENARIOS])
        self.assertEqual(lifecycle, [("enter", True), ("exit",)])
        report = json.loads(output[0])
        self.assertFalse(report["passed"])
        self.assertTrue(any("deliberate scenario failure" in failure for failure in report["failures"]))


if __name__ == "__main__":
    unittest.main()
