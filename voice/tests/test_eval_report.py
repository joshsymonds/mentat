import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.report import nearest_rank, score_observations


ROOT = Path(__file__).resolve().parents[2]


def turn(start=0.0, first_audio=2.0, confirmation=2.5, room_deleted=5.0, calls=None):
    return {
        "speech_end": start,
        "first_audio": first_audio,
        "confirmation": confirmation,
        "room_deleted": room_deleted,
        "expect_confirmation": True,
        "expect_hangup": True,
        "model_calls": ["claude-sonnet-4-5" for _ in range(calls if calls is not None else 1)],
    }


def case(name="send message", kind="action", latency=2.0):
    return {
        "name": name,
        "kind": kind,
        "runs": [
            {"turns": [turn(first_audio=latency, confirmation=latency + 0.5)]}
            for _ in range(10)
        ],
    }


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_uses_ceiling_rank(self):
        self.assertEqual(nearest_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 100], 0.90), 9)

    def test_empty_sample_is_missing(self):
        with self.assertRaises(ValueError):
            nearest_rank([], 0.5)


class ScoringTests(unittest.TestCase):
    def test_reports_per_turn_latencies_and_backend_call_counts(self):
        observation = {"cases": [case()]}
        observation["cases"][0]["runs"][0]["turns"][0]["model_calls"] = [
            {"model": "haiku"}, {"model": "sonnet"}
        ]
        result = score_observations(observation)
        report = result["cases"][0]
        self.assertTrue(result["passed"])
        self.assertEqual(report["turns"][0]["latency_seconds"], {
            "first_audio": 2.0,
            "confirmation": 2.5,
            "room_deleted": 5.0,
        })
        self.assertEqual(report["turns"][0]["model_call_count"], 2)
        self.assertEqual(report["gates"][0]["turn"], 1)
        self.assertEqual(report["gates"][0]["first_audio_p50_seconds"], 2.0)
        self.assertEqual(report["gates"][0]["first_audio_p95_seconds"], 2.0)

    def test_action_gate_fails_and_names_the_case(self):
        result = score_observations({"cases": [case(latency=5.1)]})
        self.assertFalse(result["passed"])
        self.assertIn("send message", " ".join(result["failures"]))

    def test_action_p95_gate_catches_one_slow_run(self):
        observation = {"cases": [case()]}
        observation["cases"][0]["runs"][9]["turns"][0]["first_audio"] = 5.1
        result = score_observations(observation)
        self.assertFalse(result["passed"])
        self.assertIn("p95", " ".join(result["failures"]))

    def test_action_gates_are_calculated_per_turn_across_runs(self):
        observation = {"cases": [case()]}
        for run in observation["cases"][0]["runs"]:
            run["turns"].append(turn())
        observation["cases"][0]["runs"][0]["turns"][0]["first_audio"] = 5.1

        result = score_observations(observation)

        self.assertFalse(result["passed"])
        self.assertIn("turn 1", " ".join(result["failures"]))
        self.assertNotIn("turn 2", " ".join(result["failures"]))

    def test_search_uses_ten_second_median_gate(self):
        result = score_observations({"cases": [case("web search", "search", 10.1)]})
        self.assertFalse(result["passed"])
        self.assertIn("web search", " ".join(result["failures"]))

    def test_missing_turn_observation_is_a_named_failure(self):
        observation = {"cases": [case()]}
        del observation["cases"][0]["runs"][0]["turns"][0]["room_deleted"]
        result = score_observations(observation)
        self.assertFalse(result["passed"])
        self.assertIn("send message", " ".join(result["failures"]))
        self.assertIn("room_deleted", " ".join(result["failures"]))

    def test_missing_expectation_flags_fail_closed(self):
        for flag in ("expect_confirmation", "expect_hangup"):
            with self.subTest(flag=flag):
                observation = {"cases": [case()]}
                del observation["cases"][0]["runs"][0]["turns"][0][flag]
                result = score_observations(observation)
                self.assertFalse(result["passed"])
                self.assertIn(flag, " ".join(result["failures"]))

    def test_missing_model_calls_fail_closed(self):
        observation = {"cases": [case()]}
        del observation["cases"][0]["runs"][0]["turns"][0]["model_calls"]
        result = score_observations(observation)
        self.assertFalse(result["passed"])
        self.assertIn("send message", " ".join(result["failures"]))
        self.assertIn("model_calls", " ".join(result["failures"]))

    def test_late_confirmation_and_expected_hangup_fail_finite_deadlines(self):
        observation = {"cases": [case()]}
        for run in observation["cases"][0]["runs"]:
            run["turns"][0]["confirmation"] = 1000.0
            run["turns"][0]["room_deleted"] = 2000.0

        result = score_observations(observation)

        self.assertFalse(result["passed"])
        failures = " ".join(result["failures"])
        self.assertIn("confirmation", failures)
        self.assertIn("room_deleted", failures)

    def test_alice_chain_turns_can_stay_open_without_confirmation(self):
        turn_without_end = {
            "speech_end": 0.0,
            "first_audio": 2.0,
            "model_calls": [],
            "expect_confirmation": False,
            "expect_hangup": False,
        }
        observation = {
            "cases": [{
                "name": "Alice Keck chain",
                "kind": "search",
                "runs": [{"turns": [dict(turn_without_end), dict(turn_without_end)]} for _ in range(10)],
            }]
        }

        result = score_observations(observation)

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(len(result["cases"][0]["turns"]), 20)

    def test_invalid_timestamps_fail_closed(self):
        observation = {"cases": [case()]}
        observation["cases"][0]["runs"][0]["turns"][0]["first_audio"] = -1
        result = score_observations(observation)
        self.assertFalse(result["passed"])
        self.assertIn("send message", " ".join(result["failures"]))

    def test_cli_emits_report_and_nonzero_for_failed_case(self):
        observation = {"cases": [case(latency=5.1)]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "observations.json"
            path.write_text(json.dumps(observation))
            result = subprocess.run(
                [sys.executable, str(ROOT / "voice" / "evals" / "report.py"), str(path)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("send message", result.stdout)
        self.assertIn('"passed": false', result.stdout)


if __name__ == "__main__":
    unittest.main()
