import asyncio
import os
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import caller
from caller import _capture_answer, first_matching_latency, is_agent_audio_track, parse_step


class ParseStepTests(unittest.TestCase):
    def test_parses_delay_line_and_answer_regex(self):
        self.assertEqual(
            parse_step(r"Who wrote Pride and Prejudice?@1.5::Jane\s+Austen"),
            (1.5, "Who wrote Pride and Prejudice?", r"Jane\s+Austen"),
        )

    def test_accepts_post_deploy_calls_a_through_d(self):
        calls = [
            r"What is the latest released version of livekit-agents on PyPI?@1::[0-9]+\.[0-9]+\.[0-9]+",
            r"What is the current Bitcoin price in USD according to CoinGecko?@1::(?i)(?:\$|USD\s*)[0-9,]+(?:\.[0-9]+)?|[0-9,]+(?:\.[0-9]+)?\s*USD",
            r"Who wrote Pride and Prejudice?@1::(?i)Jane\s+Austen",
            r"What was the latest Formula 1 Grand Prix, and who won it?@1::(?i)\b(?:won|winner|unsure|uncertain)\b",
        ]
        self.assertEqual([parse_step(call)[1] for call in calls], [
            "What is the latest released version of livekit-agents on PyPI?",
            "What is the current Bitcoin price in USD according to CoinGecko?",
            "Who wrote Pride and Prejudice?",
            "What was the latest Formula 1 Grand Prix, and who won it?",
        ])

    def test_rejects_missing_regex_or_invalid_delay(self):
        with self.assertRaises(ValueError):
            parse_step("Question@1")
        with self.assertRaises(ValueError):
            parse_step("Question@soon::answer")


class MatchingLatencyTests(unittest.TestCase):
    def test_latency_uses_first_matching_segment_start(self):
        segments = [
            {"start": 0.2, "text": "Let me check."},
            {"start": 1.4, "text": "Jane Austen wrote it."},
            {"start": 2.0, "text": "Jane Austen."},
        ]
        self.assertAlmostEqual(first_matching_latency(segments, r"Jane\s+Austen", 10.0, 10.05), 1.45)

    def test_returns_none_when_no_segment_matches(self):
        self.assertIsNone(first_matching_latency([{"start": 0.2, "text": "I am unsure."}], r"Jane", 10.0, 10.05))

    def test_rejects_invalid_regex(self):
        with self.assertRaises(ValueError):
            first_matching_latency([], "[", 0.0, 0.0)


class CaptureTests(unittest.IsolatedAsyncioTestCase):
    def fake_rtc(self, frames, before_first_frame=None):
        class AudioStream:
            def __init__(self, track):
                self.frames = iter(frames)
                self.first_frame = True

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    frame = next(self.frames)
                    if self.first_frame and before_first_frame is not None:
                        before_first_frame()
                    self.first_frame = False
                    return SimpleNamespace(frame=frame)
                except StopIteration:
                    raise StopAsyncIteration

            async def aclose(self):
                return None

        return SimpleNamespace(AudioStream=AudioStream)

    @staticmethod
    def frame(silent):
        sample = b"\0\0" if silent else b"\0\1"
        return SimpleNamespace(
            data=sample * 100,
            samples_per_channel=100,
            sample_rate=1000,
            num_channels=1,
        )

    async def test_capture_keeps_search_answer_after_acknowledgment_pause(self):
        answer = self.frame(False)
        answer.data = b"\x01\0" * 100
        frames = [self.frame(False), *[self.frame(True) for _ in range(13)], answer]
        queue = asyncio.Queue()
        queue.put_nowait(object())
        with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=self.fake_rtc(frames))}):
            pcm, _, _, _ = await _capture_answer(queue)
        self.assertEqual(len(pcm), len(frames) * 200)
        self.assertTrue(pcm.endswith(answer.data))

    async def test_latency_origin_includes_wait_for_first_audio_frame(self):
        class Clock:
            now = 100.0

            def monotonic(self):
                return self.now

            def advance(self, seconds):
                self.now += seconds

        clock = Clock()
        queue = asyncio.Queue()
        queue.put_nowait(object())
        with patch.object(caller, "time", clock):
            with patch.dict(
                sys.modules,
                {"livekit": SimpleNamespace(rtc=self.fake_rtc([self.frame(False)], lambda: clock.advance(8.1)))},
            ):
                _, _, _, capture_started = await _capture_answer(queue)
        self.assertEqual(capture_started, 108.1)
        self.assertAlmostEqual(
            first_matching_latency([{"start": 0.0, "text": "Answer"}], "Answer", 100.0, capture_started),
            8.1,
        )

    async def test_latency_origin_is_after_delayed_track_subscription(self):
        frames = [self.frame(False)]

        class DelayedQueue(asyncio.Queue):
            async def get(self):
                await asyncio.sleep(0)
                caller.time.monotonic()
                return object()

        queue = DelayedQueue()
        clock = Mock(side_effect=[100.0, 108.1, 108.2])
        with patch.object(caller, "time", SimpleNamespace(monotonic=clock)):
            with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=self.fake_rtc(frames))}):
                _, _, _, capture_started = await _capture_answer(queue)
        self.assertEqual(capture_started, 108.1)
        self.assertAlmostEqual(
            first_matching_latency([{"start": 0.0, "text": "Answer"}], "Answer", 100.0, capture_started),
            8.1,
        )


class TranscribeTests(unittest.IsolatedAsyncioTestCase):
    async def test_tts_returns_exact_pcm_and_http_response_metadata(self):
        class Response:
            status = 206

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

            async def read(self):
                return b"\x01\x00\x02\x00"

        http = Mock()
        http.post.return_value = Response()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = await caller._tts(http, "synthetic line")

        self.assertEqual(result.pcm, b"\x01\x00\x02\x00")
        self.assertEqual(result.http_status, 206)
        self.assertEqual(result.response_bytes, 4)

    async def test_skips_40ms_pcm_before_wav_form_or_upload(self):
        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

            async def json(self):
                return {"segments": []}

        http = Mock()
        http.post.return_value = Response()
        form_data = Mock()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"aiohttp": SimpleNamespace(FormData=form_data)}):
                segments = await caller._transcribe(http, b"\0\0" * 960, 24000, 1)
        self.assertEqual(segments, [])
        form_data.assert_not_called()
        http.post.assert_not_called()

    async def test_uploads_valid_duration_pcm_for_transcription(self):
        class FormData:
            def add_field(self, *_args, **_kwargs):
                return None

        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

            async def json(self):
                return {"segments": [{"text": "answer"}]}

        http = Mock()
        http.post.return_value = Response()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"aiohttp": SimpleNamespace(FormData=FormData)}):
                segments = await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)
        self.assertEqual(segments, [{"text": "answer"}])
        http.post.assert_called_once()

    async def test_whisper_4xx_is_a_named_clip_failure_but_server_errors_surface(self):
        class FormData:
            def add_field(self, *_args, **_kwargs):
                return None

        for status in (400, 429):
            class Response:
                def __init__(self):
                    self.status = status

                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_args):
                    return None

                def raise_for_status(self):
                    raise RuntimeError(f"HTTP {status}")

            http = Mock()
            http.post.return_value = Response()
            with self.subTest(status=status), patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with patch.dict(sys.modules, {"aiohttp": SimpleNamespace(FormData=FormData)}):
                    with self.assertRaisesRegex(RuntimeError, rf"Whisper transcription rejected clip \(HTTP {status}\)"):
                        await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)

        class ServerErrorResponse:
            status = 500

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def raise_for_status(self):
                raise RuntimeError("HTTP 500")

        http = Mock()
        http.post.return_value = ServerErrorResponse()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"aiohttp": SimpleNamespace(FormData=FormData)}):
                with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                    await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)


class ContinuousCaptureTests(unittest.IsolatedAsyncioTestCase):
    async def test_capture_starts_before_speech_and_preserves_early_frames(self):
        early = CaptureTests.frame(False)
        later = CaptureTests.frame(False)
        queue = asyncio.Queue()
        queue.put_nowait(object())
        capture = caller.ContinuousCapture(queue)
        speech_started = False
        first_frame_before_speech = []
        with patch.dict(
            sys.modules,
            {
                "livekit": SimpleNamespace(
                    rtc=CaptureTests().fake_rtc([early, later], lambda: first_frame_before_speech.append(not speech_started))
                )
            },
        ):
            await capture.start()
            speech_started = True
            pcm, _, _, capture_started = await capture.result()
        self.assertEqual(first_frame_before_speech, [True])
        self.assertEqual(pcm, early.data + later.data)
        self.assertIsInstance(capture_started, float)

    async def test_long_answer_finishes_on_silence_and_persistent_track_serves_next_turn(self):
        voice = CaptureTests.frame(False)
        silence = CaptureTests.frame(True)
        track = SimpleNamespace(turn=0)
        answers = [
            [voice] * 2 + [silence] * 50 + [voice] * 310 + [silence] * 210,
            [voice] * 2 + [silence] * 210,
        ]

        class PersistentTrackStream:
            def __init__(self, _track):
                self.frames = iter(answers[track.turn])
                track.turn += 1

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    frame = next(self.frames)
                except StopIteration:
                    await asyncio.Future()
                await asyncio.sleep(caller._frame_duration(frame) / 100)
                return SimpleNamespace(frame=frame)

            async def aclose(self):
                return None

        queue = asyncio.Queue()
        queue.put_nowait(track)
        with patch.object(caller, "MAX_ANSWER_SECONDS", 0.3):
            with patch.object(caller, "MAX_CAPTURE_SECONDS", 3):
                with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=PersistentTrackStream))}):
                    first, _, _, _ = await asyncio.wait_for(_capture_answer(queue), timeout=4)
                    self.assertIs(queue.get_nowait(), track)
                    queue.put_nowait(track)
                    second, _, _, _ = await asyncio.wait_for(_capture_answer(queue), timeout=4)
        first_voice_and_gap = b"".join(frame.data for frame in answers[0][:362])
        second_voice = b"".join(frame.data for frame in answers[1][:2])
        self.assertTrue(first.startswith(first_voice_and_gap))
        self.assertTrue(second.startswith(second_voice))
        trailing_silence_bytes = int(caller.ANSWER_END_SILENCE_SECONDS * 1000 * 2)
        self.assertGreaterEqual(len(first), len(first_voice_and_gap) + trailing_silence_bytes)
        self.assertGreaterEqual(len(second), len(second_voice) + trailing_silence_bytes)
        self.assertEqual(track.turn, 2)

    async def test_capture_waits_through_simulated_15_second_ack_search_gap(self):
        acknowledgement = CaptureTests.frame(False)
        answer = CaptureTests.frame(False)
        answer.data = b"\x00\x02" * 100
        silence = CaptureTests.frame(True)
        clock = SimpleNamespace(seconds=0.0)
        frames = [acknowledgement] * 2 + [silence] * 150 + [answer] * 2 + [silence] * 210

        class TimedPersistentTrack:
            def __init__(self, _track):
                self.frames = iter(frames)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    frame = next(self.frames)
                except StopIteration:
                    await asyncio.Future()
                clock.seconds += caller._frame_duration(frame)
                return SimpleNamespace(frame=frame)

            async def aclose(self):
                return None

        queue = asyncio.Queue()
        queue.put_nowait(object())
        with patch.object(caller, "MAX_ANSWER_SECONDS", 1):
            with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=TimedPersistentTrack))}):
                pcm, _, _, _ = await asyncio.wait_for(_capture_answer(queue), timeout=2)
        answer_pcm = answer.data * 2
        self.assertIn(answer_pcm, pcm)
        self.assertGreaterEqual(clock.seconds, 35.0)

    async def test_capture_fails_when_stream_hangs_at_finite_window(self):
        frame = CaptureTests.frame(False)

        class HangingAfterFrameStream:
            def __init__(self, track):
                self.frames = iter([frame])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return SimpleNamespace(frame=next(self.frames))
                except StopIteration:
                    await asyncio.Future()

            async def aclose(self):
                return None

        queue = asyncio.Queue()
        queue.put_nowait(object())
        with patch.object(caller, "MAX_CAPTURE_SECONDS", 0.01):
            with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=HangingAfterFrameStream))}):
                with self.assertRaisesRegex(RuntimeError, "agent audio capture exceeded its deadline"):
                    await asyncio.wait_for(_capture_answer(queue), timeout=0.2)

    async def test_capture_extends_past_initial_window_for_recent_speech(self):
        first = CaptureTests.frame(False)
        final = CaptureTests.frame(False)

        class DelayedFinalSpeech:
            def __init__(self, track):
                self.frames = iter([first, final])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    frame = next(self.frames)
                except StopIteration:
                    await asyncio.Future()
                if frame is final:
                    await asyncio.sleep(0.08)
                return SimpleNamespace(frame=frame)

            async def aclose(self):
                return None

        queue = asyncio.Queue()
        queue.put_nowait(object())
        with patch.object(caller, "MAX_ANSWER_SECONDS", 0.05):
            with patch.object(caller, "MAX_CAPTURE_SECONDS", 1):
                with patch.object(caller, "ANSWER_END_SILENCE_SECONDS", 0.1):
                    with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=DelayedFinalSpeech))}):
                        pcm, _, _, _ = await asyncio.wait_for(_capture_answer(queue), timeout=0.5)
        self.assertEqual(pcm, first.data + final.data)

    async def test_capture_finishes_after_silence_when_persistent_track_stops_emitting_frames(self):
        frame = CaptureTests.frame(False)

        class SilentPersistentTrack:
            def __init__(self, track):
                self.emitted = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self.emitted:
                    self.emitted = True
                    return SimpleNamespace(frame=frame)
                await asyncio.Future()

            async def aclose(self):
                return None

        queue = asyncio.Queue()
        track = object()
        queue.put_nowait(track)
        with patch.object(caller, "MAX_ANSWER_SECONDS", 0.05):
            with patch.object(caller, "ANSWER_END_SILENCE_SECONDS", 0.01):
                with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=SilentPersistentTrack))}):
                    pcm, _, _, _ = await asyncio.wait_for(_capture_answer(queue), timeout=0.2)
        self.assertEqual(pcm, frame.data)
        self.assertIs(queue.get_nowait(), track)

    async def test_capture_returns_frames_promptly_when_agent_audio_ends(self):
        frame = CaptureTests.frame(False)
        ended = asyncio.Event()
        frame_arrived = asyncio.Event()

        class HangingAfterFrameStream:
            def __init__(self, track):
                self.frames = iter([frame])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    event = SimpleNamespace(frame=next(self.frames))
                    frame_arrived.set()
                    return event
                except StopIteration:
                    await asyncio.Future()

            async def aclose(self):
                return None

        queue = asyncio.Queue()
        queue.put_nowait(object())
        with patch.object(caller, "MAX_ANSWER_SECONDS", 5):
            with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=HangingAfterFrameStream))}):
                capture = asyncio.create_task(_capture_answer(queue, ended))
                await asyncio.wait_for(frame_arrived.wait(), timeout=0.2)
                ended.set()
                pcm, _, _, _ = await asyncio.wait_for(capture, timeout=0.2)
        self.assertEqual(pcm, frame.data)

    async def test_capture_drains_buffered_frames_after_agent_audio_ends(self):
        first = CaptureTests.frame(False)
        trailing = CaptureTests.frame(False)
        ended = asyncio.Event()
        read_started = asyncio.Event()
        trailing_ready = asyncio.Event()

        class BufferedTrailingStream:
            def __init__(self, track):
                self.frames = iter([first])
                self.trailing_delivered = False

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return SimpleNamespace(frame=next(self.frames))
                except StopIteration:
                    if not self.trailing_delivered:
                        read_started.set()
                        await trailing_ready.wait()
                        self.trailing_delivered = True
                        return SimpleNamespace(frame=trailing)
                    await asyncio.Future()

            async def aclose(self):
                return None

        queue = asyncio.Queue()
        queue.put_nowait(object())
        with patch.object(caller, "MAX_ANSWER_SECONDS", 5):
            with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=BufferedTrailingStream))}):
                capture = asyncio.create_task(_capture_answer(queue, ended))
                await asyncio.wait_for(read_started.wait(), timeout=0.2)
                ended.set()
                await asyncio.sleep(0.01)
                # The frame is already buffered in the stream; let its pending read deliver it.
                trailing_ready.set()
                pcm, _, _, _ = await asyncio.wait_for(capture, timeout=0.2)
        self.assertEqual(pcm, first.data + trailing.data)

    async def test_capture_fails_finitely_when_no_frames_arrive(self):
        class HangingStream:
            def __init__(self, track):
                pass

            def __aiter__(self):
                return self

            async def __anext__(self):
                await asyncio.Future()

            async def aclose(self):
                return None

        queue = asyncio.Queue()
        queue.put_nowait(object())
        with patch.object(caller, "ANSWER_START_TIMEOUT_SECONDS", 0.01):
            with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=HangingStream))}):
                with self.assertRaisesRegex(RuntimeError, "no frames"):
                    await asyncio.wait_for(_capture_answer(queue), timeout=0.2)

    async def test_speech_end_timestamp_follows_playout_completion(self):
        class Clock:
            now = 1.0

            def monotonic(self):
                return self.now

        clock = Clock()

        class Source:
            async def wait_for_playout(self):
                clock.now = 4.0

        with patch.object(caller, "time", clock):
            speech_end = await caller._speech_end_after_playout(Source())
        self.assertEqual(speech_end, 4.0)


class CallerSubscriptionReadinessTests(unittest.IsolatedAsyncioTestCase):
    async def run_caller(self, subscription_timeout=1.0, steps=None, transcribe=None):
        subscribed = asyncio.Event()
        speech_started = asyncio.Event()
        output = []
        room = None
        participant = SimpleNamespace()

        class Publication:
            async def wait_for_subscription(self):
                await subscribed.wait()

        class AudioSource:
            def __init__(self, *_args):
                pass

            async def capture_frame(self, frame):
                if frame.data != b"\0\0" * frame.samples_per_channel:
                    speech_started.set()

            async def wait_for_playout(self):
                return None

        class LocalParticipant:
            async def publish_track(self, *_args, **_kwargs):
                return Publication()

        class Room:
            def __init__(self):
                self.local_participant = LocalParticipant()
                self.remote_participants = {"worker": participant}
                self.handlers = {}
                self.disconnected = False

            def on(self, event):
                return lambda callback: self.handlers.setdefault(event, callback)

            async def connect(self, *_args):
                return None

            async def disconnect(self):
                self.disconnected = True

        room = Room()

        class ClientSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

        class AccessToken:
            def __init__(self, *_args):
                pass

            def with_identity(self, *_args):
                return self

            def with_grants(self, *_args):
                return self

            def to_jwt(self):
                return "token"

        rtc = SimpleNamespace(
            Room=lambda: room,
            AudioSource=AudioSource,
            LocalAudioTrack=SimpleNamespace(create_audio_track=lambda *_args: object()),
            TrackPublishOptions=lambda **_kwargs: object(),
            AudioFrame=lambda data, rate, channels, samples: SimpleNamespace(
                data=data, sample_rate=rate, num_channels=channels, samples_per_channel=samples
            ),
            ParticipantKind=SimpleNamespace(PARTICIPANT_KIND_AGENT="agent"),
            TrackKind=SimpleNamespace(KIND_AUDIO="audio"),
            TrackSource=SimpleNamespace(SOURCE_MICROPHONE="microphone"),
        )
        api = SimpleNamespace(
            AccessToken=AccessToken,
            VideoGrants=lambda **_kwargs: object(),
        )

        class Capture:
            def __init__(self, *_args):
                pass

            async def start(self):
                return None

            async def result(self):
                return b"answer", 24000, 1, 0.0

        async def tts(_http, _text):
            return b"\1\0" * caller.FRAME_SAMPLES

        async def default_transcribe(*_args):
            return []

        patchers = [
            patch.object(caller, "ContinuousCapture", Capture),
            patch.object(caller, "_tts", tts),
            patch.object(caller, "_transcribe", transcribe or default_transcribe),
            patch.object(caller, "READINESS_TIMEOUT_SECONDS", subscription_timeout),
            patch("builtins.print", side_effect=lambda *args, **_kwargs: output.append(" ".join(map(str, args)))),
            patch.dict(os.environ, {"LIVEKIT_API_KEY": "key", "LIVEKIT_API_SECRET": "secret"}),
            patch.dict(sys.modules, {
                "aiohttp": SimpleNamespace(ClientSession=ClientSession),
                "livekit": SimpleNamespace(api=api, rtc=rtc),
            }),
        ]
        for patcher in patchers:
            patcher.start()
        task = asyncio.create_task(caller.run(
            "room", steps or ["First line@0::answer"]
        ))

        def cleanup():
            for patcher in reversed(patchers):
                patcher.stop()

        return task, subscribed, speech_started, output, room, participant, cleanup

    async def test_subscription_and_fixed_settle_gate_first_line_without_agent_metadata(self):
        task, subscribed, speech_started, output, room, _participant, cleanup = await self.run_caller(4.0)
        try:
            await asyncio.sleep(0.02)
            self.assertFalse(speech_started.is_set())
            self.assertFalse(any(line.startswith("say:") for line in output))
            subscribed.set()
            subscribed_at = time.monotonic()
            await asyncio.sleep(0.02)
            self.assertFalse(speech_started.is_set())
            self.assertFalse(any(line.startswith("say:") for line in output))
            await asyncio.wait_for(task, timeout=4)
            self.assertGreaterEqual(time.monotonic() - subscribed_at, 2.9)
            self.assertTrue(speech_started.is_set())
            self.assertTrue(any(line == "say: First line" for line in output))
            self.assertTrue(room.disconnected)
        finally:
            cleanup()

    async def test_rejected_clip_does_not_stop_later_scripted_lines(self):
        calls = 0

        async def transcribe(*_args):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise caller.WhisperTranscriptionError(400)
            return [{"start": 0.0, "end": 0.2, "text": "answer"}]

        task, subscribed, _speech_started, output, _room, _participant, cleanup = await self.run_caller(
            subscription_timeout=4.0,
            steps=["Rejected question@0::answer", "Later question@0::answer"],
            transcribe=transcribe,
        )
        subscribed.set()
        try:
            await asyncio.wait_for(task, timeout=4)
            self.assertEqual(calls, 2)
            self.assertIn("transcription failure: Whisper transcription rejected clip (HTTP 400)", output)
            self.assertIn("say: Later question", output)
        finally:
            cleanup()

    async def test_settle_timeout_is_named_and_never_speaks(self):
        task, subscribed, speech_started, output, room, _participant, cleanup = await self.run_caller(2.9)
        subscribed.set()
        try:
            with self.assertRaisesRegex(TimeoutError, "caller microphone subscription settle exceeded its deadline"):
                await asyncio.wait_for(task, timeout=1)
            self.assertFalse(speech_started.is_set())
            self.assertFalse(any(line.startswith("say:") for line in output))
            self.assertTrue(room.disconnected)
        finally:
            cleanup()

    async def test_subscription_timeout_never_speaks_and_disconnects(self):
        task, _subscribed, speech_started, output, room, _participant, cleanup = await self.run_caller(0.02)
        try:
            with self.assertRaisesRegex(TimeoutError, "caller microphone subscription exceeded its deadline"):
                await asyncio.wait_for(task, timeout=1)
            self.assertFalse(speech_started.is_set())
            self.assertFalse(any(line.startswith("say:") for line in output))
            self.assertTrue(room.disconnected)
        finally:
            cleanup()

class PublisherFilteringTests(unittest.TestCase):
    def test_agent_audio_selects_only_microphone_source(self):
        publishers = [
            ("agent", "audio", 0, "background audio"),
            ("agent", "audio", 2, "spoken answer"),
            ("standard", "audio", 2, "browser microphone"),
            ("agent", "video", 2, "agent video"),
        ]
        selected = [
            track
            for publisher_kind, track_kind, source, track in publishers
            if is_agent_audio_track(publisher_kind, track_kind, source, "agent", "audio", 2)
        ]
        self.assertEqual(selected, ["spoken answer"])


class CleanupDocumentationTests(unittest.TestCase):
    def test_cleanup_reads_pid_remotely_and_restarts_after_all_exit_paths(self):
        readme = Path(__file__).resolve().parent.parent.joinpath("README.md").read_text()
        trap_source = readme.split("```sh\nDEV_DIR=\n", 1)[1].split(
            "\nssh ultraviolet sudo systemctl stop", 1
        )[0]
        script = r'''ssh() {
  printf '%s\n' "$*" >> "$CALL_LOG"
  case " $* " in *" sh -s "*) cat > "$REMOTE_SCRIPT";; esac
}
''' + trap_source + r'''
DEV_DIR="$TEST_DEV_DIR"
case "$EXIT_PATH" in
  success) exit 0 ;;
  failure) exit 7 ;;
  INT) kill -INT "$$" ;;
  TERM) kill -TERM "$$" ;;
esac
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev_dir = root / "dev"
            dev_dir.mkdir()
            (dev_dir / "agent.pid").write_text("456\n")
            for exit_path, expected_code in (("success", 0), ("failure", 7), ("INT", 130), ("TERM", 143)):
                call_log = root / "calls"
                remote_script = root / "remote-script"
                environment = os.environ | {
                    "CALL_LOG": str(call_log),
                    "REMOTE_SCRIPT": str(remote_script),
                    "TEST_DEV_DIR": str(dev_dir),
                    "EXIT_PATH": exit_path,
                }
                result = subprocess.run(
                    ["bash", "-c", script], capture_output=True, text=True, env=environment, timeout=5
                )
                self.assertEqual(result.returncode, expected_code, result.stderr)
                calls = call_log.read_text()
                self.assertIn("systemctl start mentat-voice", calls)
                remote = remote_script.read_text()
                self.assertIn('kill "$(cat "$DEV_DIR/agent.pid")"', remote)


if __name__ == "__main__":
    unittest.main()
