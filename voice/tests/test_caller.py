import asyncio
import os
import subprocess
import sys
import tempfile
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

    async def test_capture_returns_frames_when_stream_hangs_at_finite_window(self):
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
        with patch.object(caller, "MAX_ANSWER_SECONDS", 0.01):
            with patch.dict(sys.modules, {"livekit": SimpleNamespace(rtc=SimpleNamespace(AudioStream=HangingAfterFrameStream))}):
                pcm, rate, channels, _ = await asyncio.wait_for(_capture_answer(queue), timeout=0.2)
        self.assertEqual(pcm, frame.data)
        self.assertEqual((rate, channels), (1000, 1))

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
        with patch.object(caller, "MAX_ANSWER_SECONDS", 0.01):
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
