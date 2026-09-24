import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from caller import first_matching_latency, is_agent_audio_track, parse_step


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


class PublisherFilteringTests(unittest.TestCase):
    def test_mixed_audio_publishers_selects_only_agent_audio(self):
        publishers = [
            ("agent", "audio", "answer audio"),
            ("standard", "audio", "browser microphone"),
            ("agent", "video", "agent video"),
        ]
        selected = [
            track
            for publisher_kind, track_kind, track in publishers
            if is_agent_audio_track(publisher_kind, track_kind, "agent", "audio")
        ]
        self.assertEqual(selected, ["answer audio"])


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
