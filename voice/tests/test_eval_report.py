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
        "speech_end_wall": start,
        "first_audio": first_audio,
        "capture_started": start + 0.5,
        "segments": [{"start": 1.0, "end": 1.2, "text": "assistant response"}],
        "command_received_at": first_audio,
        "answer_at": first_audio,
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
        observation["cases"][0]["runs"][1]["turns"][0]["command_received_at"] = 5.1

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

    def test_action_and_answer_gates_use_outcome_timestamps_not_first_audio(self):
        action = turn(first_audio=2.0, confirmation=None, room_deleted=None)
        action.update({
            "turn": 1,
            "expect_confirmation": False,
            "expect_hangup": False,
            "command_received_at": 12.0,
            "answer_at": 2.0,
        })
        search = turn(first_audio=2.0, confirmation=None, room_deleted=None, kind="search")
        search.update({
            "turn": 1,
            "expect_confirmation": False,
            "expect_hangup": False,
            "command_received_at": None,
            "answer_at": 12.0,
        })
        observations = {"cases": [
            {"name": "slow phone receipt", "runs": [{"turns": [dict(action)]} for _ in range(10)]},
            {"name": "slow matching answer", "runs": [{"turns": [dict(search)]} for _ in range(10)]},
        ]}

        result = score_observations(observations)

        self.assertFalse(result["passed"])
        actions, searches = result["cases"]
        self.assertEqual(actions["turns"][0]["latency_seconds"], {
            "command_receipt": 12.0,
            "answer": 2.0,
            "first_audio": 2.0,
            "first_speech": 1.5,
            "confirmation": None,
            "room_deleted": None,
        })
        self.assertEqual(actions["gates"][0]["command_receipt_p50_seconds"], 12.0)
        self.assertEqual(searches["gates"][0]["answer_p50_seconds"], 12.0)
        self.assertEqual(searches["gates"][0]["first_audio_p50_seconds"], 2.0)
        self.assertIn("command-receipt p50 12s", " ".join(actions["failures"]))
        self.assertIn("answer p50 12s", " ".join(searches["failures"]))

    def test_action_receipt_uses_wall_speech_end_and_gates_without_matching_answer(self):
        timely = turn(first_audio=12.0, confirmation=None, room_deleted=None)
        timely.update({
            "turn": 1,
            "speech_end": 10.0,
            "speech_end_wall": 1_700_000_000.0,
            "command_received_at": 1_700_000_002.0,
            "answer_at": None,
            "expect_confirmation": False,
            "expect_hangup": False,
        })
        slow = dict(timely)
        slow["command_received_at"] = 1_700_000_006.0
        observation = {"cases": [
            {"name": "timely receipt", "runs": [{"turns": [dict(timely)]} for _ in range(10)]},
            {"name": "slow receipt", "runs": [{"turns": [dict(slow)]} for _ in range(10)]},
        ]}

        result = score_observations(observation)

        self.assertFalse(result["passed"])
        timely_report, slow_report = result["cases"]
        self.assertEqual(timely_report["gates"][0]["command_receipt_p50_seconds"], 2.0)
        self.assertEqual(slow_report["gates"][0]["command_receipt_p50_seconds"], 6.0)
        self.assertIn("missing answer_at timestamp", " ".join(timely_report["failures"]))
        self.assertFalse(any("command-receipt" in failure for failure in timely_report["failures"]))
        self.assertTrue(any("command-receipt p50 6s" in failure for failure in slow_report["failures"]))

    def test_search_command_receipt_uses_wall_clock_without_changing_answer_gate(self):
        with_command = turn(start=100.0, first_audio=102.0, confirmation=None, room_deleted=None, kind="search")
        with_command.update({
            "speech_end_wall": 1_700_000_000.0,
            "command_received_at": 1_700_000_002.0,
            "answer_at": 112.0,
            "expect_confirmation": False,
            "expect_hangup": False,
        })
        without_command = dict(with_command)
        without_command.update({
            "command_received_at": None,
            "answer_at": 109.0,
        })
        result = score_observations({"cases": [
            {"name": "search with location command", "runs": [{"turns": [with_command]}]},
            {"name": "search without command", "runs": [{"turns": [without_command]}]},
        ]}, required_runs=1)

        commanded, uncommanded = result["cases"]
        self.assertEqual(commanded["turns"][0]["latency_seconds"]["command_receipt"], 2.0)
        self.assertEqual(commanded["gates"][0]["answer_p50_seconds"], 12.0)
        self.assertTrue(any("answer p50 12s" in failure for failure in commanded["failures"]))
        self.assertIsNone(uncommanded["turns"][0]["latency_seconds"]["command_receipt"])
        self.assertEqual(uncommanded["gates"][0]["answer_p50_seconds"], 9.0)
        self.assertEqual(uncommanded["failures"], [])

    def test_missing_or_malformed_command_receipt_and_answer_timestamps_fail_closed(self):
        for field, value in (("command_received_at", None), ("command_received_at", float("nan")),
                             ("answer_at", None), ("answer_at", float("inf")),
                             ("speech_end_wall", None)):
            with self.subTest(field=field, value=value):
                observation_turn = turn()
                observation_turn.update({
                    "turn": 1,
                    "command_received_at": 2.5,
                    "answer_at": 2.0,
                })
                observation_turn[field] = value
                result = score_observations({"cases": [{
                    "name": "strict timing", "runs": [{"turns": [observation_turn]}],
                }]}, required_runs=1)
                self.assertFalse(result["passed"])
                expected_name = (
                    "command receipt"
                    if field == "command_received_at"
                    else "speech_end_wall"
                    if field == "speech_end_wall"
                    else "answer_at"
                )
                self.assertIn(expected_name, " ".join(result["failures"]))

    def test_reports_per_turn_latencies_and_backend_call_counts(self):
        observation = {"cases": [case()]}
        observation["cases"][0]["runs"][0]["turns"][0]["model_calls"] = [
            {"model": "haiku"}, {"model": "sonnet"}
        ]
        result = score_observations(observation)
        report = result["cases"][0]
        self.assertTrue(result["passed"])
        self.assertEqual(report["turns"][0]["latency_seconds"], {
            "command_receipt": 2.0,
            "answer": 2.0,
            "first_audio": 2.0,
            "first_speech": 1.5,
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

    def test_scored_turn_preserves_judge_questions_verdicts_probabilities_and_status(self):
        judge = {
            "context": "",
            "questions": [{
                "id": "place_lookup",
                "question": "Did the assistant identify the requested place?",
                "verdict": True,
                "probability": 0.93,
            }],
            "unavailable": None,
        }
        scored_turn = turn(kind="search")
        scored_turn["judge"] = judge

        result = score_observations({
            "cases": [{"name": "place lookup", "runs": [{"turns": [scored_turn]}]}],
        }, required_runs=1)

        self.assertEqual(result["cases"][0]["turns"][0]["judge"], judge)

    def test_later_judge_unavailable_keeps_prior_verdict_raw_evidence_and_phone_commands(self):
        prior_judge = {
            "context": "",
            "questions": [{
                "id": "sms_recipient",
                "question": "Did the assistant repeat the recipient correctly?",
                "verdict": True,
                "probability": 0.91,
            }],
            "unavailable": None,
        }
        unavailable_judge = {
            "context": "verified recipient: 5550100",
            "questions": [{
                "id": "sms_body",
                "question": "Did the assistant read back the message correctly?",
                "verdict": None,
                "probability": None,
            }],
            "unavailable": "judge transport failed after retries",
        }
        first = turn(first_audio=2.0, confirmation=None, room_deleted=None)
        first.update({"turn": 1, "kind": "search", "judge": prior_judge,
                      "raw_segments": [{"start": 0.0, "end": 0.3, "text": "I have Alex."}],
                      "expect_confirmation": False, "expect_hangup": False})
        second = turn(first_audio=2.0, confirmation=None, room_deleted=None)
        second.update({"turn": 2, "kind": "search", "judge": unavailable_judge,
                       "raw_segments": [{"start": 0.0, "end": 0.4, "text": "What should I say?"}],
                       "expect_confirmation": False, "expect_hangup": False})
        phone_commands = [{
            "id": "fake-sms-1", "turn": 1, "kind": "sms",
            "recipient": "5550100", "body": "Meet at six.",
        }]
        observation = {"cases": [{"name": "sms say-back", "runs": [{
            "turns": [first, second],
            "product_failures": [{"turn": 2, "message": "judge unavailable: transport failed"}],
            "phone_commands": phone_commands,
        }]}]}

        result = score_observations(observation, required_runs=1)

        report = result["cases"][0]
        self.assertFalse(result["passed"])
        self.assertIn("product failure", " ".join(report["failures"]))
        self.assertEqual([item["judge"] for item in report["turns"]], [prior_judge, unavailable_judge])
        self.assertEqual([item["segments"] for item in report["turns"]], [
            first["raw_segments"], second["raw_segments"],
        ])
        self.assertEqual(report["phone_commands"], [phone_commands])

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

    def test_product_failure_report_preserves_raw_transcript_segments(self):
        raw_segments = [
            {"start": 0.0, "end": 0.1, "text": "Okay."},
            {"start": 0.1, "end": 0.3, "text": "The timer is set."},
        ]
        failed_turn = turn()
        failed_turn["turn"] = 1
        failed_turn["raw_segments"] = raw_segments
        observation = {
            "cases": [{
                "name": "timer",
                "runs": [{
                    "turns": [failed_turn],
                    "product_failures": [{"turn": 1, "message": "missing answer"}],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        self.assertEqual(result["cases"][0]["turns"][0]["segments"], raw_segments)

    def test_partial_capture_report_preserves_raw_failed_turn_segments(self):
        raw_segments = [
            {"start": 0.0, "end": 0.1, "text": "Okay."},
            {"start": 0.1, "end": 0.3, "text": "The timer is set."},
        ]
        observation = {
            "cases": [{
                "name": "timer",
                "runs": [{
                    "turns": [],
                    "failure": {
                        "turn": 1,
                        "message": NO_ANSWER_FAILURE,
                        "speech_started_at": 1_700_000_000.0,
                        "segments": raw_segments,
                    },
                    "product_failures": [],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        self.assertEqual(
            result["cases"][0]["capture_failures"],
            [{"run": 1, "turn": 1, "message": NO_ANSWER_FAILURE, "segments": raw_segments}],
        )
        for segment in raw_segments:
            self.assertIn(segment, result["cases"][0]["capture_failures"][0]["segments"])

    def test_asr_deadline_failure_keeps_completed_turn_and_partial_transcript_evidence(self):
        completed = turn()
        completed["speech_started_at"] = 1_700_000_000.0
        partial_segments = [{"start": 0.0, "end": 0.2, "text": "finished phrase"}]
        observation = {
            "cases": [{
                "name": "send message",
                "runs": [{
                    "turns": [completed],
                    "failure": {
                        "turn": 2,
                        "message": "answer transcription exceeded its deadline",
                        "speech_started_at": 1_700_000_010.0,
                        "segments": partial_segments,
                    },
                    "product_failures": [{"turn": 1, "message": "completed product evidence"}],
                    "phone_commands": [{"id": "sms-1", "turn": 1, "kind": "sms"}],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        report = result["cases"][0]
        self.assertEqual(len(report["turns"]), 1)
        self.assertEqual(report["capture_failures"], [{
            "run": 1,
            "turn": 2,
            "message": "answer transcription exceeded its deadline",
            "segments": partial_segments,
        }])
        failures = " ".join(report["failures"])
        self.assertIn("completed product evidence", failures)
        self.assertIn("answer transcription exceeded its deadline", failures)

    def test_partial_capture_scores_completed_turns_and_names_failed_turn(self):
        completed = turn()
        completed["speech_started_at"] = 1_700_000_000.0
        observation = {
            "cases": [{
                "name": "send message",
                "runs": [{
                    "turns": [completed],
                    "failure": {
                        "turn": 2,
                        "message": "answer transcription exceeded its deadline",
                        "speech_started_at": 1_700_000_001.0,
                        "segments": [{"start": 0.0, "end": 0.2, "text": "partial"}],
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
        self.assertIn("answer transcription exceeded its deadline", failures)
        self.assertIn("first-speech evidence missing capture_started timestamp", failures)
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
        trace["speech_started_at"] = 1_700_000_000.0
        trace["expect_confirmation"] = False
        trace["expect_hangup"] = False
        observation = {
            "cases": [{
                "name": "place-search-navigation",
                "runs": [{
                    "turns": [trace],
                    "failure": {
                        "turn": 2,
                        "message": "answer transcription exceeded its deadline",
                        "speech_started_at": 1_700_000_001.0,
                        "segments": [{"start": 0.0, "end": 0.2, "text": "partial"}],
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
        trace["speech_started_at"] = 1_700_000_000.0
        trace["expect_confirmation"] = False
        trace["expect_hangup"] = False
        observation = {
            "cases": [{
                "name": "place-search-navigation",
                "runs": [{
                    "turns": [trace],
                    "failure": {
                        "turn": 2,
                        "message": "answer transcription exceeded its deadline",
                        "speech_started_at": 1_700_000_001.0,
                        "segments": [{"start": 0.0, "end": 0.2, "text": "partial"}],
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
        trace["speech_started_at"] = 1_700_000_000.0
        trace["expect_confirmation"] = False
        trace["expect_hangup"] = False
        observation = {
            "cases": [{
                "name": "place-search-navigation",
                "runs": [{
                    "turns": [trace],
                    "failure": {
                        "turn": 2,
                        "message": "answer transcription exceeded its deadline",
                        "speech_started_at": 1_700_000_001.0,
                        "segments": [{"start": 0.0, "end": 0.2, "text": "partial"}],
                    },
                    "product_failures": [{"turn": 2, "message": "forged"}],
                    "phone_commands": [],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        self.assertIn("invalid partial product failure metadata", " ".join(result["failures"]))

    def test_report_includes_retry_counts_for_completed_and_failed_lines(self):
        completed = turn()
        completed.update({"script_line": 1, "tts_retry_count": 2})
        message = "scripted speech content verification failed for line 2"
        result = score_observations({
            "cases": [{
                "name": "caller-lines",
                "runs": [
                    {"turns": [completed]},
                    {
                        "turns": [],
                        "failure": {
                            "turn": 1,
                            "message": message,
                            "line": 2,
                            "retry_count": 1,
                        },
                        "product_failures": [],
                    },
                ],
            }],
        }, required_runs=2)

        self.assertEqual(result["cases"][0]["line_retry_counts"], [
            {"run": 1, "line": 1, "retry_count": 2},
            {"run": 2, "line": 2, "retry_count": 1},
        ])
        self.assertIn("eval infrastructure failure", " ".join(result["failures"]))

    def test_preflight_render_content_failure_is_line_indexed_infrastructure_failure(self):
        message = "scripted speech content verification failed for line 2"
        result = score_observations({
            "cases": [{
                "name": "sms-say-back-yes",
                "runs": [{
                    "turns": [],
                    "failure": {"turn": 1, "message": message},
                    "product_failures": [],
                }],
            }]
        }, required_runs=1)

        self.assertFalse(result["passed"])
        self.assertEqual(result["cases"][0]["capture_failures"], [{
            "run": 1, "turn": 1, "message": message,
        }])
        failures = " ".join(result["failures"])
        self.assertIn("eval infrastructure failure", failures)
        self.assertNotIn("invalid partial capture failure metadata", failures)
        self.assertNotIn("product failure", failures)

    def test_preflight_sample_and_content_failures_are_infrastructure_failures(self):
        messages = (
            "scripted speech sample count mismatch for line 1",
            "scripted speech content verification failed for line 1",
        )
        for message in messages:
            with self.subTest(message=message):
                result = score_observations({
                    "cases": [{
                        "name": "sms-say-back-yes",
                        "runs": [{
                            "turns": [],
                            "failure": {"turn": 1, "message": message},
                            "product_failures": [],
                        }],
                    }]
                }, required_runs=1)

                self.assertFalse(result["passed"])
                case_report = result["cases"][0]
                self.assertEqual(case_report["capture_failures"], [{
                    "run": 1,
                    "turn": 1,
                    "message": message,
                }])
                failures = " ".join(result["failures"])
                self.assertIn("eval infrastructure failure", failures)
                self.assertNotIn("invalid partial capture failure metadata", failures)
                self.assertNotIn("product failure", failures)

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
                "message": "scripted speech synthesis for line 2 exceeded its deadline",
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
        observation["cases"][0]["runs"][9]["turns"][0]["command_received_at"] = 5.1
        result = score_observations(observation)
        self.assertFalse(result["passed"])
        self.assertIn("p95", " ".join(result["failures"]))

    def test_action_gates_are_calculated_per_turn_across_runs(self):
        observation = {"cases": [case()]}
        for run in observation["cases"][0]["runs"]:
            run["turns"].append(turn())
        observation["cases"][0]["runs"][0]["turns"][0]["command_received_at"] = 5.1

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
            "capture_started": 0.5,
            "segments": [{"start": 1.0, "end": 1.2, "text": "assistant response"}],
            "command_received_at": None,
            "answer_at": 2.0,
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

    def test_first_audio_is_diagnostic_and_does_not_gate_the_turn(self):
        observation = {"cases": [case()]}
        observation["cases"][0]["runs"][0]["turns"][0]["speech_end"] = 2.0
        observation["cases"][0]["runs"][0]["turns"][0]["first_audio"] = 1.5
        observation["cases"][0]["runs"][0]["turns"][0]["command_received_at"] = 2.5
        observation["cases"][0]["runs"][0]["turns"][0]["answer_at"] = 2.5
        observation["cases"][0]["runs"][0]["turns"][0]["overlap"] = True

        result = score_observations(observation)

        self.assertTrue(result["passed"], result["failures"])
        turn_report = result["cases"][0]["turns"][0]
        self.assertEqual(turn_report["latency_seconds"]["first_audio"], -0.5)
        self.assertEqual(result["cases"][0]["gates"][0]["command_receipt_p50_seconds"], 2.0)

    def test_first_speech_gate_uses_first_nonempty_segment_after_capture_started(self):
        observation = {"cases": [case()]}
        expected_latency = 1.75
        for run in observation["cases"][0]["runs"]:
            scored_turn = run["turns"][0]
            scored_turn["first_audio"] = scored_turn["speech_end"] + 0.1
            scored_turn["capture_started"] = scored_turn["speech_end"] + 0.25
            scored_turn["segments"] = [
                {"start": 0.1, "end": 0.2, "text": "  "},
                {"start": 1.5, "end": 1.8, "text": "First transcribed response."},
                {"start": 2.0, "end": 2.2, "text": "Later response."},
            ]

        result = score_observations(observation)

        self.assertTrue(result["passed"], result["failures"])
        self.assertEqual(
            result["cases"][0]["turns"][0]["latency_seconds"]["first_audio"],
            0.1,
        )
        self.assertEqual(
            result["cases"][0]["turns"][0]["latency_seconds"]["first_speech"],
            expected_latency,
        )
        self.assertEqual(
            result["cases"][0]["gates"][0]["first_speech_p50_seconds"],
            expected_latency,
        )

    def test_first_speech_p50_over_two_seconds_fails(self):
        observation = {"cases": [case()]}
        for run in observation["cases"][0]["runs"]:
            run["turns"][0]["capture_started"] = 0.0
            run["turns"][0]["segments"] = [
                {"start": 2.01, "end": 2.2, "text": "Late response."},
            ]

        result = score_observations(observation)

        self.assertFalse(result["passed"])
        self.assertIn("first-speech p50 2.01s", " ".join(result["failures"]))

    def test_missing_first_speech_evidence_fails_closed(self):
        for missing in ("segments", "capture_started"):
            with self.subTest(missing=missing):
                observation = {"cases": [case()]}
                for run in observation["cases"][0]["runs"]:
                    del run["turns"][0][missing]

                result = score_observations(observation)

                self.assertFalse(result["passed"])
                self.assertIn("first-speech", " ".join(result["failures"]))
                case_report = result["cases"][0]
                self.assertEqual(len(case_report["turns"]), 10)
                self.assertEqual(case_report["gates"][0]["command_receipt_p50_seconds"], 2.0)
                self.assertIsNone(case_report["gates"][0]["first_speech_p50_seconds"])

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

    def test_early_room_close_is_reported_as_product_failure_not_capture_failure(self):
        captured = turn(first_audio=2.25, confirmation=None, room_deleted=None, calls=2)
        captured.update({
            "turn": 1,
            "expect_confirmation": False,
            "expect_hangup": False,
            "command_received_at": 2.5,
            "answer_at": 2.25,
        })
        observation = {
            "cases": [{
                "name": "three-turn-call",
                "runs": [{
                    "turns": [captured],
                    "phone_commands": [{"id": "timer-1", "turn": 1, "kind": "timer"}],
                    "room_closed_after": 1,
                    "product_failures": [{
                        "turn": 1,
                        "message": "call ended after turn 1 with 2 follow-ups remaining",
                    }],
                }],
            }]
        }

        result = score_observations(observation, required_runs=1)

        self.assertFalse(result["passed"])
        report = result["cases"][0]
        self.assertEqual(len(report["turns"]), 1)
        self.assertEqual(report["turns"][0]["model_call_count"], 2)
        self.assertEqual(report["turns"][0]["latency_seconds"]["first_audio"], 2.25)
        failures = " ".join(result["failures"])
        self.assertIn("three-turn-call run 1 turn 1: product failure", failures)
        self.assertIn("call ended after turn 1 with 2 follow-ups remaining", failures)
        self.assertNotIn("capture failed", failures)

    def test_report_counts_only_named_capacity_evidence_per_run_and_batch(self):
        observation = {"cases": [case(name="capacity", kind="action")]}
        runs = observation["cases"][0]["runs"]
        runs[0]["capacity_failures"] = [
            {"source": "candidate daemon start", "cause": "Address already in use"},
            {"source": "candidate daemon start", "cause": "Address already in use"},
        ]
        runs[1]["capacity_failures"] = [
            {"source": "room dispatch", "cause": "HTTP 429: capacity limit"},
        ]
        runs[2]["failure"] = "scenario failed: provider capacity limit wording"

        result = score_observations(observation)

        report = result["cases"][0]
        self.assertEqual(result["capacity_failure_count"], 2)
        self.assertEqual(report["capacity_failure_count"], 2)
        self.assertEqual(
            report["run_capacity_failures"],
            [
                {"run": 1, "count": 1, "causes": [
                    {"source": "candidate daemon start", "cause": "Address already in use"}
                ]},
                {"run": 2, "count": 1, "causes": [
                    {"source": "room dispatch", "cause": "HTTP 429: capacity limit"}
                ]},
                *[{"run": run_index, "count": 0, "causes": []} for run_index in range(3, 11)],
            ],
        )

    def test_clean_report_emits_zero_capacity_count(self):
        result = score_observations({"cases": [case(name="clean")]})

        self.assertEqual(result["capacity_failure_count"], 0)
        self.assertEqual(result["cases"][0]["capacity_failure_count"], 0)
        self.assertEqual(
            result["cases"][0]["run_capacity_failures"],
            [
                {"run": run_index, "count": 0, "causes": []}
                for run_index in range(1, 11)
            ],
        )

    def test_report_preserves_batch_and_per_run_timing_without_changing_scores(self):
        batch_timing = {
            "started_at": 1_700_000_000.0,
            "ended_at": 1_700_000_009.0,
            "wall_seconds": 9.0,
            "concurrency_cap": 4,
        }
        run_timings = [
            {
                "started_at": 1_700_000_003.0,
                "ended_at": 1_700_000_007.0,
                "concurrency": 2,
            },
            {
                "started_at": 1_700_000_001.0,
                "ended_at": 1_700_000_005.0,
                "concurrency": 4,
            },
        ]
        observation = {
            "batch_timing": batch_timing,
            "cases": [{
                "name": "timed case",
                "runs": [
                    {"turns": [turn()], "timing": run_timings[0]},
                    {"turns": [turn()], "timing": run_timings[1]},
                ] + [{"turns": [turn()]} for _ in range(8)],
            }],
        }
        expected_scores = score_observations({"cases": observation["cases"]})

        result = score_observations(observation)

        self.assertEqual(result["batch_timing"], batch_timing)
        self.assertEqual(
            result["cases"][0]["run_timings"],
            [
                {"run": 1, **run_timings[0]},
                {"run": 2, **run_timings[1]},
            ],
        )
        self.assertEqual(result["passed"], expected_scores["passed"])
        self.assertEqual(result["failures"], expected_scores["failures"])
        self.assertEqual(result["cases"][0]["turns"], expected_scores["cases"][0]["turns"])
        self.assertEqual(result["cases"][0]["gates"], expected_scores["cases"][0]["gates"])


if __name__ == "__main__":
    unittest.main()
