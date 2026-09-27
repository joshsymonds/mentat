import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.report import nearest_rank, score_observations
from evals.runner import NO_ANSWER_FAILURE


ROOT = Path(__file__).resolve().parents[2]


def turn(start=0.0, first_audio=2.0, confirmation=2.5, room_deleted=5.0, calls=None, kind="action"):
    return {
        "kind": kind,
        "speech_end": start,
        "first_audio": first_audio,
        "overlap": False,
        "confirmation": confirmation,
        "room_deleted": room_deleted,
        "expect_confirmation": True,
        "expect_hangup": True,
        "model_calls": ["claude-sonnet-4-5" for _ in range(calls if calls is not None else 1)],
    }


def case(name="send message", kind="action", latency=2.0):
    return {
        "name": name,
        "runs": [
            {"turns": [turn(first_audio=latency, confirmation=latency + 0.5, kind=kind)]}
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
    def test_explicit_run_count_gates_run_and_per_turn_samples(self):
        observation = {"cases": [case()]}
        observation["cases"][0]["runs"] = observation["cases"][0]["runs"][:2]
        for run in observation["cases"][0]["runs"]:
            run["turns"].append(turn())

        result = score_observations(observation, required_runs=2)

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(result["cases"][0]["run_count"], 2)
        self.assertEqual([gate["run_count"] for gate in result["cases"][0]["gates"]], [2, 2])

    def test_explicit_run_count_keeps_percentile_gate_strict(self):
        observation = {"cases": [case()]}
        observation["cases"][0]["runs"] = observation["cases"][0]["runs"][:2]
        observation["cases"][0]["runs"][1]["turns"][0]["first_audio"] = 5.1

        result = score_observations(observation, required_runs=2)

        self.assertFalse(result["passed"])
        self.assertIn("p95", " ".join(result["failures"]))
        self.assertEqual(result["cases"][0]["gates"][0]["run_count"], 2)

    def test_explicit_run_count_fails_when_case_or_turn_count_differs(self):
        observation = {"cases": [case()]}
        observation["cases"][0]["runs"] = observation["cases"][0]["runs"][:2]
        for run in observation["cases"][0]["runs"]:
            run["turns"].append(turn())
        del observation["cases"][0]["runs"][1]["turns"][1]

        result = score_observations(observation, required_runs=2)

        self.assertFalse(result["passed"])
        failures = " ".join(result["failures"])
        self.assertIn("expected 2 latency observations, found 1", failures)

    def test_invalid_required_run_counts_fail_closed(self):
        observation = {"cases": [case()]}
        for required_runs in (0, -1, True, 1.5, "2", None):
            with self.subTest(required_runs=required_runs):
                result = score_observations(observation, required_runs=required_runs)
                self.assertFalse(result["passed"])
                self.assertIn("required_runs", " ".join(result["failures"]))

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

    def test_mixed_case_applies_search_and_action_gates_to_each_turn(self):
        observation = {
            "cases": [{
                "name": "place search and navigation",
                "runs": [{
                    "turns": [
                        turn(first_audio=9.0, kind="search"),
                        turn(first_audio=6.0, kind="action"),
                    ]
                } for _ in range(10)],
            }]
        }

        result = score_observations(observation)

        self.assertFalse(result["passed"])
        self.assertEqual(
            [turn["kind"] for turn in result["cases"][0]["turns"][:2]],
            ["search", "action"],
        )
        gates = result["cases"][0]["gates"]
        self.assertEqual([gate["kind"] for gate in gates], ["search", "action"])
        self.assertEqual(gates[0]["first_audio_p50_seconds"], 9.0)
        self.assertEqual(gates[1]["first_audio_p50_seconds"], 6.0)
        self.assertEqual(gates[1]["first_audio_p95_seconds"], 6.0)
        failures = " ".join(result["failures"])
        self.assertIn("turn 2", failures)
        self.assertIn("p95", failures)
        self.assertNotIn("turn 1", failures)

    def test_complete_product_failure_scores_captured_turn_and_fails_report(self):
        captured = turn(first_audio=2.25, confirmation=None, room_deleted=5.0, calls=2)
        captured["turn"] = 1
        captured["expect_confirmation"] = False
        observation = {
            "cases": [{
                "name": "place-search-navigation",
                "runs": [{
                    "turns": [captured],
                    "product_failures": [{
                        "turn": 1,
                        "message": "place-search-navigation: turn 1 missing answer pattern 'Alice Keck Park'",
                    }],
                    "phone_commands": [{"id": "place-command", "turn": 1, "kind": "location"}],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        self.assertEqual(len(result["cases"][0]["turns"]), 1)
        self.assertEqual(result["cases"][0]["turns"][0]["model_call_count"], 2)
        self.assertEqual(result["cases"][0]["turns"][0]["latency_seconds"]["first_audio"], 2.25)
        self.assertIn("product failure", " ".join(result["cases"][0]["failures"]))
        self.assertIn("missing answer pattern", " ".join(result["failures"]))

    def test_complete_product_failure_metadata_fails_closed(self):
        captured = turn()
        captured["turn"] = 1
        observation = {
            "cases": [{
                "name": "place-search-navigation",
                "runs": [{"turns": [captured], "product_failures": [{"turn": True, "message": "forged"}]}],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        self.assertIn("invalid complete product failure metadata", " ".join(result["failures"]))

    def test_partial_capture_scores_completed_turns_and_names_failed_turn(self):
        observation = {
            "cases": [{
                "name": "send message",
                "runs": [{
                    "turns": [turn()],
                    "failure": {
                        "turn": 2,
                        "message": "room was deleted before all scripted lines were captured",
                    },
                    "product_failures": [
                        {"turn": 1, "message": "turn 1 missing answer pattern 'Alice Keck Park'"}
                    ],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        report = result["cases"][0]
        self.assertEqual(len(report["turns"]), 1)
        self.assertEqual(report["turns"][0]["latency_seconds"]["first_audio"], 2.0)
        self.assertEqual(report["turns"][0]["model_call_count"], 1)
        self.assertEqual(report["gates"][0]["run_count"], 1)
        failures = " ".join(result["failures"])
        self.assertIn("send message run 1 turn 2", failures)
        self.assertIn("room was deleted before all scripted lines were captured", failures)
        self.assertIn("send message run 1 turn 1", failures)
        self.assertIn("missing answer pattern 'Alice Keck Park'", failures)
        self.assertEqual(report["turns"][0]["model_call_count"], 1)
        self.assertEqual(report["turns"][0]["latency_seconds"]["first_audio"], 2.0)

    def test_same_turn_hangup_timeout_scores_answer_and_keeps_fake_phone_evidence(self):
        captured = turn(first_audio=2.25, confirmation=None, room_deleted=None, calls=2)
        captured["turn"] = 1
        captured["expect_confirmation"] = False
        observation = {
            "cases": [{
                "name": "timer-300-seconds",
                "runs": [{
                    "turns": [captured],
                    "failure": {
                        "turn": 1,
                        "message": "room deletion was not observed before deadline",
                    },
                    "product_failures": [],
                    "phone_commands": [{
                        "id": "fake-timer-1",
                        "turn": 1,
                        "kind": "timer",
                    }],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        report = result["cases"][0]
        self.assertEqual(len(report["turns"]), 1)
        self.assertEqual(report["turns"][0]["latency_seconds"]["first_audio"], 2.25)
        self.assertIsNone(report["turns"][0]["latency_seconds"]["room_deleted"])
        self.assertEqual(report["turns"][0]["model_call_count"], 2)
        self.assertEqual(report["gates"][0]["run_count"], 1)
        failures = " ".join(report["failures"])
        self.assertIn("timer-300-seconds run 1 turn 1: capture failed", failures)
        self.assertIn("room deletion was not observed before deadline", failures)
        self.assertIn("expected room_deleted observation is missing", failures)
        self.assertEqual(observation["cases"][0]["runs"][0]["phone_commands"][0]["id"], "fake-timer-1")

    def test_valid_completed_prefix_reports_only_the_later_capture_failure(self):
        trace = turn(first_audio=2.0, confirmation=None, room_deleted=None, kind="search")
        trace["expect_confirmation"] = False
        trace["expect_hangup"] = False
        observation = {
            "cases": [{
                "name": "place-search-navigation",
                "runs": [{
                    "turns": [trace],
                    "failure": {
                        "turn": 2,
                        "message": "scripted speech synthesis exceeded its deadline",
                    },
                    "product_failures": [],
                    "phone_commands": [{
                        "id": "phone-command-1",
                        "turn": 1,
                        "kind": "location",
                    }],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        failures = " ".join(result["failures"])
        self.assertIn("place-search-navigation run 1 turn 2: capture failed", failures)
        self.assertNotIn("product failure", failures)
        self.assertEqual(result["cases"][0]["turns"][0]["latency_seconds"]["first_audio"], 2.0)

    def test_partial_failure_turn_product_error_requires_observed_phone_command(self):
        trace = turn(first_audio=2.0, confirmation=None, room_deleted=None, kind="search")
        trace["expect_confirmation"] = False
        trace["expect_hangup"] = False
        observation = {
            "cases": [{
                "name": "place-search-navigation",
                "runs": [{
                    "turns": [trace],
                    "failure": {
                        "turn": 2,
                        "message": "scripted speech synthesis exceeded its deadline",
                    },
                    "product_failures": [{
                        "turn": 2,
                        "message": "place-search-navigation: navigation selected an unexpected place name 'Wrong Park'",
                    }],
                    "phone_commands": [{
                        "id": "phone-command-2",
                        "turn": 2,
                        "kind": "navigate",
                    }],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        failures = " ".join(result["failures"])
        self.assertIn("place-search-navigation run 1 turn 2: product failure", failures)
        self.assertIn("place-search-navigation run 1 turn 2: capture failed", failures)
        self.assertNotIn("invalid partial product failure metadata", failures)
        self.assertEqual(result["cases"][0]["turns"][0]["model_call_count"], 1)

    def test_failed_turn_product_error_without_real_phone_evidence_fails_closed(self):
        trace = turn(first_audio=2.0, confirmation=None, room_deleted=None, kind="search")
        trace["expect_confirmation"] = False
        trace["expect_hangup"] = False
        observation = {
            "cases": [{
                "name": "place-search-navigation",
                "runs": [{
                    "turns": [trace],
                    "failure": {
                        "turn": 2,
                        "message": "scripted speech synthesis exceeded its deadline",
                    },
                    "product_failures": [{"turn": 2, "message": "forged"}],
                    "phone_commands": [],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        self.assertIn("invalid partial product failure metadata", " ".join(result["failures"]))

    def test_first_turn_capture_failure_is_named_with_no_completed_turns(self):
        observation = {
            "cases": [{
                "name": "timer",
                "runs": [{
                    "turns": [],
                    "failure": {
                        "turn": 1,
                        "message": NO_ANSWER_FAILURE,
                        "speech_started_at": 1_700_000_000.0,
                    },
                    "product_failures": [],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        failures = " ".join(result["failures"])
        self.assertIn("timer run 1 turn 1", failures)
        self.assertIn(NO_ANSWER_FAILURE, failures)

    def test_first_turn_pcm_no_answer_failure_is_named_in_cli_report(self):
        observation = {
            "cases": [{
                "name": "timer-no-answer",
                "runs": [{
                    "turns": [],
                    "failure": {
                        "turn": 1,
                        "message": NO_ANSWER_FAILURE,
                        "speech_started_at": 1_700_000_000.0,
                    },
                    "product_failures": [],
                }],
            }]
        }
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

        self.assertEqual(result.returncode, 1)
        self.assertIn("timer-no-answer run 1 turn 1: capture failed", result.stdout)
        self.assertIn(NO_ANSWER_FAILURE, result.stdout)
        self.assertNotIn("invalid partial capture failure metadata", result.stdout)

    def test_malformed_partial_capture_failure_metadata_fails_closed(self):
        malformed_failures = (
            {"turn": "2", "message": "room was deleted before all scripted lines were captured"},
            {"turn": 2, "message": NO_ANSWER_FAILURE},
            {"turn": 2, "message": "room deletion was not observed before deadline"},
            {
                "turn": 2,
                "message": "scripted speech synthesis exceeded its deadline",
                "speech_started_at": 100.0,
            },
        )
        for failure in malformed_failures:
            with self.subTest(failure=failure):
                observation = {
                    "cases": [{
                        "name": "send message",
                        "runs": [{"turns": [turn()], "failure": failure}],
                    }]
                }

                result = score_observations(observation, required_runs=1)

                self.assertFalse(result["passed"])
                self.assertIn("invalid partial capture failure", " ".join(result["failures"]))

    def test_same_turn_hangup_timeout_requires_the_exact_unclosed_hangup_turn(self):
        invalid_turns = (
            {**turn(room_deleted=None), "turn": 1, "expect_hangup": False},
            {**turn(room_deleted=5.0), "turn": 1},
            {**turn(room_deleted=None), "turn": 2},
        )
        for invalid_turn in invalid_turns:
            with self.subTest(turn=invalid_turn):
                result = score_observations({
                    "cases": [{
                        "name": "timer",
                        "runs": [{
                            "turns": [invalid_turn],
                            "failure": {
                                "turn": 1,
                                "message": "room deletion was not observed before deadline",
                            },
                            "product_failures": [],
                        }],
                    }]
                }, required_runs=1)
                self.assertFalse(result["passed"])
                self.assertIn(
                    "invalid partial capture failure metadata",
                    " ".join(result["failures"]),
                )

    def test_missing_per_turn_kind_fails_closed(self):
        observation = {"cases": [case()]}
        del observation["cases"][0]["runs"][0]["turns"][0]["kind"]

        result = score_observations(observation)

        self.assertFalse(result["passed"])
        self.assertIn("missing kind", " ".join(result["failures"]))

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
            "overlap": False,
            "model_calls": [],
            "expect_confirmation": False,
            "expect_hangup": False,
        }
        observation = {
            "cases": [{
                "name": "Alice Keck chain",
                "runs": [{
                    "turns": [
                        {**turn_without_end, "kind": "search"},
                        {**turn_without_end, "kind": "search"},
                    ]
                } for _ in range(10)],
            }]
        }

        result = score_observations(observation)

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(len(result["cases"][0]["turns"]), 20)

    def test_early_audio_is_reported_as_overlap_and_latency_uses_post_playout_audio(self):
        observation = {"cases": [case()]}
        for run in observation["cases"][0]["runs"]:
            run["turns"][0]["speech_end"] = 2.0
            run["turns"][0]["first_audio"] = 2.2
            run["turns"][0]["overlap"] = True

        result = score_observations(observation)

        self.assertTrue(result["passed"], result["failures"])
        first_turn = result["cases"][0]["turns"][0]
        self.assertAlmostEqual(first_turn["latency_seconds"]["first_audio"], 0.2)
        self.assertTrue(first_turn["overlap"])
        self.assertAlmostEqual(
            result["cases"][0]["gates"][0]["first_audio_p50_seconds"], 0.2
        )

    def test_first_audio_before_speech_end_is_not_a_valid_latency(self):
        observation = {"cases": [case()]}
        observation["cases"][0]["runs"][0]["turns"][0]["speech_end"] = 2.0
        observation["cases"][0]["runs"][0]["turns"][0]["first_audio"] = 1.5
        observation["cases"][0]["runs"][0]["turns"][0]["overlap"] = True

        result = score_observations(observation)

        self.assertFalse(result["passed"])
        self.assertIn("first_audio precedes speech_end", " ".join(result["failures"]))

    def test_missing_or_invalid_timestamps_fail_closed(self):
        for invalid in (None, float("nan"), -1):
            with self.subTest(invalid=invalid):
                observation = {"cases": [case()]}
                if invalid is None:
                    del observation["cases"][0]["runs"][0]["turns"][0]["first_audio"]
                else:
                    observation["cases"][0]["runs"][0]["turns"][0]["first_audio"] = invalid
                result = score_observations(observation)
                self.assertFalse(result["passed"])
                self.assertIn("first_audio", " ".join(result["failures"]))

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
