import asyncio
import os
import struct
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

    def test_script_language_marker_is_removed_from_the_spoken_line(self):
        raw_step = "[[es]]¿Cuál es la capital de Francia?@0::.*"
        self.assertEqual(
            parse_step(raw_step),
            (0.0, "¿Cuál es la capital de Francia?", ".*"),
        )
        self.assertEqual(caller.parse_step_language(raw_step), "es")
        self.assertEqual(caller.parse_step_language("English@0::.*"), "en")


class MatchingLatencyTests(unittest.TestCase):
    def test_latency_uses_first_matching_segment_start(self):
        segments = [
            {"start": 0.2, "text": "Let me check."},
            {"start": 1.4, "text": "Jane Austen wrote it."},
            {"start": 2.0, "text": "Jane Austen."},
        ]
        self.assertAlmostEqual(first_matching_latency(segments, r"Jane\s+Austen", 10.0, 10.05), 1.45)

    def test_latency_matches_phrases_across_word_segments(self):
        segments = [
            {"start": 0.2, "text": "Jane"},
            {"start": 0.6, "text": "Austen"},
            {"start": 1.0, "text": "wrote"},
        ]
        self.assertAlmostEqual(first_matching_latency(segments, r"Jane\s+Austen", 10.0, 10.05), 0.25)
        self.assertAlmostEqual(first_matching_latency(segments, r"wrote", 10.0, 10.05), 1.05)

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

    async def test_capture_feeds_every_agent_frame_to_echo_residual(self):
        frames = [self.frame(False), self.frame(False)]
        echo = Mock()
        queue = asyncio.Queue()
        queue.put_nowait(object())
        with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=self.fake_rtc(frames))}):
            await _capture_answer(queue, echo=echo)
        self.assertEqual([call.args[0] for call in echo.feed.call_args_list], frames)


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

    async def test_spanish_tts_requests_a_clear_spanish_rendering(self):
        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

            async def read(self):
                return b"\\x01\\x00\\x02\\x00"

        http = Mock()
        http.post.return_value = Response()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            await caller._tts(http, "¿Cuál es la capital de Francia?", language="es")

        self.assertEqual(
            http.post.call_args.kwargs["json"],
            {
                "model": "gpt-4o-mini-tts",
                "voice": "ash",
                "input": "¿Cuál es la capital de Francia?",
                "instructions": "Speak clearly in Spanish.",
                "response_format": "pcm",
            },
        )

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

    async def test_uploads_scribe_batch_request_and_maps_phrase_timestamps(self):
        class FormData:
            def __init__(self):
                self.fields = []

            def add_field(self, name, value, **kwargs):
                self.fields.append((name, value, kwargs))

        form = FormData()

        class Response:
            status = 200

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

            async def json(self):
                return {
                    "text": "+1 800 555 0142",
                    "words": [
                        {"text": "+1", "start": 0.2, "end": 0.4, "type": "word"},
                        {"text": "800", "start": 0.5, "end": 0.8, "type": "word"},
                        {"text": "555", "start": 0.9, "end": 1.1, "type": "word"},
                        {"text": "0142", "start": 1.2, "end": 1.6, "type": "word"},
                    ],
                }

        http = Mock()
        http.post.return_value = Response()
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"aiohttp": SimpleNamespace(FormData=lambda: form)}):
                segments = await caller._transcribe(http, b"\0\0" * 24000, 24000, 1)

        self.assertEqual(
            segments,
            [
                {"start": 0.2, "end": 0.4, "text": "+1"},
                {"start": 0.5, "end": 0.8, "text": "800"},
                {"start": 0.9, "end": 1.1, "text": "555"},
                {"start": 1.2, "end": 1.6, "text": "0142"},
            ],
        )
        self.assertEqual(http.post.call_args.args[0], "https://api.elevenlabs.io/v1/speech-to-text")
        self.assertEqual(http.post.call_args.kwargs["headers"], {"xi-api-key": "test-key"})
        self.assertEqual([field[0] for field in form.fields], ["file", "model_id", "tag_audio_events"])
        self.assertEqual(form.fields[0][2], {"filename": "answer.wav", "content_type": "audio/wav"})
        self.assertEqual(form.fields[1][1], "scribe_v2")
        self.assertEqual(form.fields[2][1], "false")

    async def test_text_without_word_timestamps_fails_by_name_and_keeps_text(self):
        class FormData:
            def add_field(self, *_args, **_kwargs):
                return None

        class Response:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            def raise_for_status(self):
                return None

            async def json(self):
                return {"text": "Got it.", "words": [{"text": "Got"}, {"text": "it."}]}

        http = Mock()
        http.post.return_value = Response()
        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"aiohttp": SimpleNamespace(FormData=FormData)}):
                with self.assertRaises(caller.TranscriptionTimestampError) as caught:
                    await caller._transcribe(http, b"\0\0" * 7200, 24000, 1)

        self.assertEqual(caught.exception.text, "Got it.")
        self.assertIn("word timestamps", str(caught.exception))

    async def test_recovers_after_bounded_429_retry(self):
        class FormData:
            def add_field(self, *_args, **_kwargs):
                return None

        class Response:
            def __init__(self, status, payload):
                self.status = status
                self.payload = payload

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def json(self):
                return self.payload

            def raise_for_status(self):
                return None

        http = Mock()
        http.post.side_effect = [
            Response(429, {
                "detail": {
                    "code": "rate_limit_exceeded",
                    "status": "too_many_requests",
                }
            }),
            Response(200, {
                "text": "The answer is six.",
                "words": [{"text": "The", "start": 0.0, "end": 0.1, "type": "word"}],
            }),
        ]
        delays = []

        async def sleep(delay):
            delays.append(delay)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"aiohttp": SimpleNamespace(FormData=FormData)}):
                with patch.object(caller, "_transcription_retry_sleep", sleep):
                    segments = await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)

        self.assertEqual(http.post.call_count, 2)
        self.assertEqual(delays, [0.5])
        self.assertEqual(
            segments,
            [{"start": 0.0, "end": 0.1, "text": "The"}],
        )

    async def test_retries_429_and_uses_typed_concurrency_error_after_exhaustion(self):
        class FormData:
            def add_field(self, *_args, **_kwargs):
                return None

        class Response:
            status = 429

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def json(self):
                return {
                    "detail": {
                        "type": "rate_limit_error",
                        "code": "concurrent_limit_exceeded",
                        "message": (
                            "Too many concurrent requests. Your current subscription is associated with a maximum of "
                            "12 concurrent requests (running in parallel). This is done such that a single user does "
                            "not overwhelm our systems and affect other users negatively. Please upgrade your "
                            "subscription or contact sales if you want to increase this limit."
                        ),
                        "status": "too_many_concurrent_requests",
                        "request_id": "6987ad41e236a010fe8eaa67d8756296",
                        "docs_url": "https://elevenlabs.io/docs/eleven-api/resources/errors#rate-limiting-and-concurrency",
                    }
                }

            def raise_for_status(self):
                raise RuntimeError("unexpected status handling")

        http = Mock()
        http.post.return_value = Response()
        delays = []

        async def sleep(delay):
            delays.append(delay)

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {"aiohttp": SimpleNamespace(FormData=FormData)}):
                with patch.object(caller, "_transcription_retry_sleep", sleep, create=True):
                    with self.assertRaises(caller.TranscriptionCapacityError) as raised:
                        await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)

        self.assertEqual(http.post.call_count, caller.TRANSCRIPTION_MAX_ATTEMPTS)
        self.assertEqual(len(delays), caller.TRANSCRIPTION_MAX_ATTEMPTS - 1)
        self.assertLessEqual(sum(delays), 30)
        self.assertEqual(raised.exception.status, 429)
        self.assertEqual(
            raised.exception.capacity_failure,
            {"source": "audio transcription", "cause": "ElevenLabs concurrent_limit_exceeded"},
        )

    async def test_other_429_and_4xx_are_named_and_server_transport_failures_surface(self):
        class FormData:
            def add_field(self, *_args, **_kwargs):
                return None

        class ContentTypeError(Exception):
            pass

        class Response:
            def __init__(self, status, detail=None):
                self.status = status
                self.detail = detail

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return None

            async def json(self):
                return {
                    "detail": {
                        "code": self.detail,
                        "status": "too_many_requests" if self.status == 429 else "invalid_request",
                    }
                } if self.detail else {}

            def raise_for_status(self):
                raise RuntimeError(f"HTTP {self.status}")

        class NonJsonResponse(Response):
            async def json(self):
                raise ContentTypeError("error body is not JSON")

        async def no_wait(_delay):
            return None

        with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test-key"}):
            with patch.dict(sys.modules, {
                "aiohttp": SimpleNamespace(FormData=FormData, ContentTypeError=ContentTypeError)
            }):
                with patch.object(caller, "_transcription_retry_sleep", no_wait, create=True):
                    http = Mock()
                    http.post.return_value = Response(429, "rate_limit_exceeded")
                    with self.assertRaises(caller.TranscriptionError) as raised:
                        await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)
                    self.assertNotIsInstance(raised.exception, caller.TranscriptionCapacityError)
                    self.assertEqual(raised.exception.code, "rate_limit_exceeded")
                    self.assertEqual(http.post.call_count, caller.TRANSCRIPTION_MAX_ATTEMPTS)

                    http = Mock()
                    http.post.return_value = Response(400, "invalid_api_key")
                    with self.assertRaises(caller.TranscriptionError) as bad_request:
                        await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)
                    self.assertEqual(bad_request.exception.code, "invalid_api_key")
                    self.assertEqual(http.post.call_count, 1)

                    http = Mock()
                    http.post.return_value = NonJsonResponse(429)
                    with self.assertRaises(caller.TranscriptionError) as non_json_rate_limit:
                        await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)
                    self.assertEqual(non_json_rate_limit.exception.code, None)
                    self.assertEqual(http.post.call_count, caller.TRANSCRIPTION_MAX_ATTEMPTS)

                    http = Mock()
                    http.post.return_value = NonJsonResponse(400)
                    with self.assertRaises(caller.TranscriptionError) as non_json_bad_request:
                        await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)
                    self.assertEqual(non_json_bad_request.exception.status, 400)
                    self.assertEqual(http.post.call_count, 1)

                    http = Mock()
                    http.post.return_value = Response(500)
                    with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                        await caller._transcribe(http, b"\0\0" * 2400, 24000, 1)

                    http = Mock()
                    http.post.side_effect = OSError("transport failed")
                    with self.assertRaisesRegex(OSError, "transport failed"):
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

    async def test_continuous_capture_feeds_every_agent_frame_to_echo_residual(self):
        frames = [CaptureTests.frame(False), CaptureTests.frame(False)]
        echo = Mock()
        queue = asyncio.Queue()
        queue.put_nowait(object())
        capture = caller.ContinuousCapture(queue, echo=echo)
        with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=CaptureTests().fake_rtc(frames))}):
            await capture.start()
            await capture.result()
        self.assertEqual([call.args[0] for call in echo.feed.call_args_list], frames)

    @staticmethod
    def live_stream(frames, tracks=None, pace=0.0, hang=True):
        class LiveStream:
            def __init__(self, track):
                if tracks is not None:
                    tracks.append(track)
                self.frames = iter(frames)

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    frame = next(self.frames)
                except StopIteration:
                    if not hang:
                        raise StopAsyncIteration from None
                    await asyncio.Future()
                await asyncio.sleep(pace)
                return SimpleNamespace(frame=frame)

            async def aclose(self):
                return None

        return SimpleNamespace(rtc=SimpleNamespace(AudioStream=LiveStream))

    async def test_wait_for_voice_returns_onset_of_first_voiced_frame(self):
        class Clock:
            now = 100.0

            def monotonic(self):
                return self.now

        clock = Clock()
        silent, voiced = CaptureTests.frame(True), CaptureTests.frame(False)

        class SteppedStream:
            def __init__(self, track):
                self.frames = iter([silent, silent, voiced])

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    frame = next(self.frames)
                except StopIteration:
                    raise StopAsyncIteration from None
                clock.now += 1.0
                return SimpleNamespace(frame=frame)

            async def aclose(self):
                return None

        queue = asyncio.Queue()
        queue.put_nowait(object())
        capture = caller.ContinuousCapture(queue)
        with patch.object(caller, "time", clock):
            with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=SteppedStream))}):
                await capture.start()
                onset = await asyncio.wait_for(capture.wait_for_voice(), timeout=1)
                _, _, _, capture_started = await capture.result()
        self.assertEqual(capture_started, 101.0)
        self.assertEqual(onset, 103.0)

    async def test_wait_for_voice_raises_when_capture_ends_without_speech(self):
        queue = asyncio.Queue()
        queue.put_nowait(object())
        capture = caller.ContinuousCapture(queue)
        stream = self.live_stream([CaptureTests.frame(True)], hang=False)
        with patch.dict(sys.modules, {"livekit": stream}):
            await capture.start()
            with self.assertRaisesRegex(RuntimeError, "produced no speech"):
                await asyncio.wait_for(capture.wait_for_voice(), timeout=1)

    async def test_wait_for_voice_requires_start(self):
        capture = caller.ContinuousCapture(asyncio.Queue())
        with self.assertRaisesRegex(RuntimeError, "not started"):
            await capture.wait_for_voice()

    async def test_stop_requires_start(self):
        capture = caller.ContinuousCapture(asyncio.Queue())
        with self.assertRaisesRegex(RuntimeError, "not started"):
            capture.stop()

    async def test_stop_ends_ongoing_reply_with_frames_so_far_and_returns_track(self):
        voiced = CaptureTests.frame(False)
        track = object()
        queue = asyncio.Queue()
        queue.put_nowait(track)
        capture = caller.ContinuousCapture(queue)
        stream = self.live_stream([voiced] * 3)
        with patch.object(caller, "MAX_ANSWER_SECONDS", 5):
            with patch.dict(sys.modules, {"livekit": stream}):
                await capture.start()
                await asyncio.wait_for(capture.wait_for_voice(), timeout=1)
                await asyncio.sleep(0.01)
                self.assertTrue(queue.empty())
                capture.stop()
                pcm, _, _, _ = await asyncio.wait_for(capture.result(), timeout=0.5)
        self.assertEqual(pcm, voiced.data * 3)
        self.assertIs(queue.get_nowait(), track)

    async def test_stop_ends_capture_while_agent_audio_keeps_flowing(self):
        voiced = CaptureTests.frame(False)
        track = object()
        queue = asyncio.Queue()
        queue.put_nowait(track)
        capture = caller.ContinuousCapture(queue)
        stream = self.live_stream([voiced] * 100000, pace=0.001)
        with patch.object(caller, "MAX_ANSWER_SECONDS", 30):
            with patch.dict(sys.modules, {"livekit": stream}):
                await capture.start()
                await asyncio.wait_for(capture.wait_for_voice(), timeout=1)
                capture.stop()
                pcm, _, _, _ = await asyncio.wait_for(capture.result(), timeout=0.5)
        self.assertTrue(pcm)
        self.assertLess(len(pcm), len(voiced.data) * 100000)
        self.assertIs(queue.get_nowait(), track)

    async def test_capture_after_stop_receives_the_same_track(self):
        voiced = CaptureTests.frame(False)
        track = object()
        queue = asyncio.Queue()
        queue.put_nowait(track)
        streamed = []
        stream = self.live_stream([voiced] * 2, tracks=streamed)
        with patch.object(caller, "MAX_ANSWER_SECONDS", 5):
            with patch.dict(sys.modules, {"livekit": stream}):
                first = caller.ContinuousCapture(queue)
                await first.start()
                await asyncio.wait_for(first.wait_for_voice(), timeout=1)
                first.stop()
                await asyncio.wait_for(first.result(), timeout=0.5)
                second = caller.ContinuousCapture(queue)
                await second.start()
                await asyncio.wait_for(second.wait_for_voice(), timeout=1)
                second.stop()
                await asyncio.wait_for(second.result(), timeout=0.5)
        self.assertEqual(streamed, [track, track])

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
                raise caller.TranscriptionError(400)
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
            self.assertIn("transcription failure: Transcription rejected clip (HTTP 400)", output)
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


def echo_frame(samples, sample_rate=caller.RATE, channels=1):
    return SimpleNamespace(
        # rtc.AudioFrame.data is an int16 memoryview, not bytes.
        data=memoryview(struct.pack(f"<{len(samples)}h", *samples)).cast("h"),
        samples_per_channel=len(samples) // channels,
        sample_rate=sample_rate,
        num_channels=channels,
    )


def mic_pcm(samples):
    return struct.pack(f"<{len(samples)}h", *samples)


def mic_samples(pcm):
    return list(struct.unpack(f"<{len(pcm) // 2}h", pcm))


class EchoResidualTests(unittest.TestCase):
    def test_mixes_agent_audio_heard_150ms_earlier_attenuated_by_30db(self):
        echo = caller.EchoResidual()
        echo.feed(echo_frame([10000] * 2400), heard_at=0.0)
        out = mic_samples(echo.mix(mic_pcm([0] * 240), started_at=0.15))
        self.assertEqual(out, [316] * 240)

    def test_mic_gets_the_agent_sample_heard_exactly_150ms_earlier(self):
        echo = caller.EchoResidual()
        agent = [30 * index for index in range(1000)]
        echo.feed(echo_frame(agent), heard_at=0.5)
        out = mic_samples(echo.mix(mic_pcm([0] * 240), started_at=0.65))
        self.assertEqual(out, [round(sample * 10 ** (-30 / 20)) for sample in agent[:240]])

    def test_mic_is_unchanged_before_the_echo_delay_elapses(self):
        echo = caller.EchoResidual()
        echo.feed(echo_frame([10000] * 2400), heard_at=0.0)
        self.assertEqual(mic_samples(echo.mix(mic_pcm([0] * 240), started_at=0.0)), [0] * 240)

    def test_agent_audio_heard_while_nothing_is_mixed_does_not_extend_the_delay(self):
        echo = caller.EchoResidual()
        echo.feed(echo_frame([10000] * 2400), heard_at=0.0)
        echo.feed(echo_frame([20000] * 2400), heard_at=1.0)
        out = mic_samples(echo.mix(mic_pcm([0] * 240), started_at=1.15))
        self.assertEqual(out, [632] * 240)

    def test_48khz_stereo_agent_audio_is_mixed_as_mono_24khz(self):
        echo = caller.EchoResidual()
        stereo = [value for index in range(480) for value in (60 * (index // 2), 0)]
        echo.feed(echo_frame(stereo, sample_rate=48000, channels=2), heard_at=0.5)
        out = mic_samples(echo.mix(mic_pcm([0] * 240), started_at=0.65))
        self.assertEqual(out, [round(30 * index * 10 ** (-30 / 20)) for index in range(240)])

    def test_48khz_mono_agent_audio_is_decimated_to_24khz(self):
        echo = caller.EchoResidual()
        agent = [30 * (index // 2) for index in range(480)]
        echo.feed(echo_frame(agent, sample_rate=48000), heard_at=0.5)
        out = mic_samples(echo.mix(mic_pcm([0] * 240), started_at=0.65))
        self.assertEqual(out, [round(30 * index * 10 ** (-30 / 20)) for index in range(240)])

    def test_rejects_agent_audio_at_an_unsupported_sample_rate(self):
        echo = caller.EchoResidual()
        with self.assertRaises(ValueError):
            echo.feed(echo_frame([10000] * 4410, sample_rate=44100), heard_at=0.0)

    def test_mixed_output_clips_to_int16(self):
        echo = caller.EchoResidual()
        echo.feed(echo_frame([32767] * 2400), heard_at=0.0)
        echo.feed(echo_frame([-32767] * 2400), heard_at=0.1)
        high = mic_samples(echo.mix(mic_pcm([32760] * 240), started_at=0.15))
        low = mic_samples(echo.mix(mic_pcm([-32760] * 240), started_at=0.25))
        self.assertEqual(high, [32767] * 240)
        self.assertEqual(low, [-32768] * 240)

    def test_mic_is_returned_unchanged_without_agent_audio(self):
        echo = caller.EchoResidual()
        pcm = mic_pcm([1, -2, 3] * 80)
        self.assertEqual(echo.mix(pcm, started_at=5.0), pcm)


if __name__ == "__main__":
    unittest.main()
