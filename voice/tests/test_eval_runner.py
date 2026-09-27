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
            def __init__(self, _queue):
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
            patch.dict(os.environ, {"LIVEKIT_API_KEY": "key", "LIVEKIT_API_SECRET": "secret"}),
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

    async def test_three_turn_chain_keeps_open_room_without_deletion_wait(self):
        dependencies = self.dependencies_for_failure("alice")
        steps = [
            r"Why is this garden named for Alice Keck?@0::\bAlice Keck Park\b",
            r"Okay, who was she?@0::W\. M\. Keck",
            r"Okay, what was the source of her wealth?@0::Superior Oil",
        ]
        with patch.dict(os.environ, {"LIVEKIT_API_KEY": "key", "LIVEKIT_API_SECRET": "secret"}):
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
            patch.dict(os.environ, {"LIVEKIT_API_KEY": "key", "LIVEKIT_API_SECRET": "secret"}),
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
            patch.dict(os.environ, {"LIVEKIT_API_KEY": "key", "LIVEKIT_API_SECRET": "secret"}),
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

    async def test_missing_audio_transcript_and_room_deletion_fail_closed(self):
        for failure, expected in (
            ("audio", "no frames"),
            ("capture", "exceeded its deadline"),
            ("transcript", "empty transcript"),
            ("answer", "did not match expected answer"),
            ("room", "room deletion was not observed"),
        ):
            with self.subTest(failure=failure):
                dependencies = self.dependencies_for_failure(failure)
                with self.assertRaisesRegex(RuntimeError, expected):
                    with (
                        patch.dict(os.environ, {"LIVEKIT_API_KEY": "key", "LIVEKIT_API_SECRET": "secret"}),
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

    @staticmethod
    def dependencies_for_failure(failure):
        class Room:
            remote_participants = {"agent": object()}
            local_participant = SimpleNamespace(
                publish_track=lambda *_args, **_kwargs: asyncio.sleep(0)
            )

            def on(self, _event):
                return lambda callback: callback

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
            def __init__(self, _queue):
                pass

            async def start(self):
                pass

            async def result(self):
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


class ScenarioObservationTests(unittest.TestCase):
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
        with self.assertRaisesRegex(RuntimeError, "no completed turns"):
            runner._recorded_turns([{"type": "system", "subtype": "init"}])
        with self.assertRaisesRegex(RuntimeError, "no transcript segment timestamps"):
            runner._confirmation_time({"capture_started": 5.0, "segments": []})

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
                "speech_end": 99.5,
                "first_audio": 100.2,
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
                "speech_end": 201.0,
                "first_audio": 201.2,
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

        class Stack:
            base_url = f"http://127.0.0.1:{server.server_port}"

            def __init__(self):
                self.worker_rooms = []
                self.voice_commands = []
                self.remote_commands = []

            def start_worker(self, room):
                self.worker_rooms.append(room)

            def run_voice(self, command):
                self.voice_commands.append(command)
                if command[0] == "evals/runner.py":
                    return CompletedProcess(command, 0, json.dumps({"turns": assistant_turns}), "")
                raise AssertionError(f"unexpected remote voice command {command!r}")

            def run_remote(self, command):
                self.remote_commands.append(command)
                if command == ["sudo", "cat", "evals/phone.jsonl"]:
                    return CompletedProcess(command, 0, phone_log, "")
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
        self.assertEqual(stack.voice_commands[0][0], "evals/runner.py")
        self.assertIn(grant["room"], stack.voice_commands[0])
        self.assertIn("--fake-phone", stack.voice_commands[0])
        self.assertEqual(observation["turns"][0]["confirmation"], 101.0)
        self.assertNotEqual(observation["turns"][0]["confirmation"], observation["turns"][0]["first_audio"])
        self.assertEqual([len(turn["model_calls"]) for turn in observation["turns"]], [1, 2])
        self.assertEqual(observation["phone_commands"][0]["turn"], 2)
        self.assertEqual(observation["phone_commands"][0]["id"], "phone-id-independent-of-tool-id")
        self.assertEqual(observation["room_closed_after"], 2)


if __name__ == "__main__":
    unittest.main()
