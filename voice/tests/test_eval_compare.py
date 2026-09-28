import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.compare import compare_reports
from evals.scenarios import SCENARIOS


ROOT = Path(__file__).resolve().parents[2]
SOL_MODEL = "chatgpt/sol-fast"
OPUS_MODEL = "claude-opus-5-5"
SONNET_MODEL = "claude-sonnet-5-5"


def report(
    latencies,
    *,
    requested_model=SOL_MODEL,
    passed=True,
    failures=None,
    case_failures=None,
    run_count=None,
):
    observed_runs = len(latencies) if run_count is None else run_count
    observed_model = {
        SOL_MODEL: "gpt-6-sol",
        OPUS_MODEL: OPUS_MODEL,
        SONNET_MODEL: SONNET_MODEL,
    }[requested_model]
    has_standard_speed = requested_model in (OPUS_MODEL, SONNET_MODEL)
    is_opus = requested_model == OPUS_MODEL
    calls = []
    for index in range(len(latencies)):
        calls.append({
            "id": f"call-{index + 1}",
            "requested_model": requested_model,
            "observed_model": observed_model,
            "service_tier": "standard" if has_standard_speed else None,
            "result_service_tier": "standard",
            "speed": "standard" if has_standard_speed else None,
            "fast_mode_state": "off" if is_opus else None,
            "failures": [],
        })

    cases = []
    for scenario in SCENARIOS:
        action_turns = {command["turn"] for command in scenario.commands}
        turns = []
        gates = []
        for turn_number in range(1, len(scenario.turns) + 1):
            kind = "action" if turn_number in action_turns else "search"
            metric = "command_receipt" if kind == "action" else "answer"
            observations = [latency for latency in latencies]
            ordered = sorted(observations)
            p50 = ordered[(len(ordered) + 1) // 2 - 1] if ordered else None
            p95 = ordered[(len(ordered) * 95 + 99) // 100 - 1] if ordered else None
            gate = {
                "turn": turn_number,
                "kind": kind,
                "run_count": len(observations),
                f"{metric}_p50_seconds": p50,
                f"{metric}_p95_seconds": p95,
                "first_speech_p50_seconds": p50,
                "first_speech_p95_seconds": p95,
            }
            gates.append(gate)
            for index, latency in enumerate(latencies):
                turns.append({
                    "run": index + 1,
                    "turn": turn_number,
                    "kind": kind,
                    "model_call_count": 1,
                    "model_provenance": {
                        "requested_model": requested_model,
                        "verified": True,
                        "model_calls": [{**calls[index]}],
                    },
                    "latency_seconds": {
                        "first_speech": latency,
                        "command_receipt": latency if kind == "action" else None,
                        "answer": latency if kind == "search" else None,
                    },
                })
        cases.append({
            "name": scenario.name,
            "run_count": observed_runs,
            "turns": turns,
            "capture_failures": [],
            "gates": gates,
            "failures": case_failures if not cases and case_failures else [],
        })
    return {
        "passed": passed,
        "failures": failures or [],
        "requested_model": requested_model,
        "cases": cases,
    }


def opus_report(latencies, **kwargs):
    return report(latencies, requested_model=OPUS_MODEL, **kwargs)


def sonnet_report(latencies, **kwargs):
    return report(latencies, requested_model=SONNET_MODEL, **kwargs)


class CompareTests(unittest.TestCase):
    def test_latency_gate_failures_do_not_disqualify_and_correctness_is_summarized(self):
        sol = report([1.0] * 10, passed=False, failures=[
            "timer-300-seconds turn 1: first-speech p50 3s exceeds 2s",
            "timer-300-seconds turn 1: command-receipt p50 3s exceeds 2s",
        ])
        sol["cases"][0]["failures"] = [
            "timer-300-seconds turn 1: answer p50 3s exceeds 2s",
        ]

        result = compare_reports(sol, opus_report([2.0] * 10))

        self.assertTrue(result["candidates"]["sol"]["eligible"])
        self.assertTrue(result["candidates"]["sol"]["correctness"]["passed"])
        self.assertEqual(result["candidates"]["sol"]["correctness"]["failure_count"], 0)
        self.assertIsNone(result["winner"])

    def test_two_and_three_arm_results_include_per_turn_latency_table(self):
        two_arm = compare_reports(report([1.0] * 10), opus_report([2.0] * 10))
        three_arm = compare_reports(
            report([1.0] * 10), opus_report([2.0] * 10), sonnet_report([0.5] * 10)
        )

        for result, expected_fastest in ((two_arm, "sol"), (three_arm, "sonnet")):
            self.assertIsNone(result["winner"])
            self.assertEqual(result["fastest_eligible"], [expected_fastest])
            self.assertEqual(result["decision"], "informational_fastest_eligible")
            for candidate in result["candidates"].values():
                table = candidate["turn_metrics"]
                self.assertTrue(table)
                self.assertTrue(all("scenario" in row and "turn" in row for row in table))
                self.assertTrue(all("first_speech_p50_seconds" in row for row in table))
                self.assertTrue(all("first_speech_p95_seconds" in row for row in table))
                self.assertTrue(all("command_receipt_p50_seconds" in row for row in table))
                self.assertTrue(all("command_receipt_p95_seconds" in row for row in table))
                self.assertTrue(all("answer_p50_seconds" in row for row in table))
                self.assertTrue(all("answer_p95_seconds" in row for row in table))
                self.assertIn("correctness", candidate)

    def test_omitting_contracted_scenarios_makes_that_arm_ineligible(self):
        sol = report([1.0] * 10)
        opus = opus_report([2.0] * 10)
        opus["cases"] = opus["cases"][:1]

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertEqual(result["fastest_eligible"], ["sol"])
        self.assertTrue(result["candidates"]["sol"]["eligible"])
        self.assertFalse(result["candidates"]["opus"]["eligible"])

    def test_omitting_a_contracted_scenario_turn_blocks_both_arms(self):
        sol = report([1.0] * 10)
        opus = opus_report([2.0] * 10)
        for arm in (sol, opus):
            case = next(
                case for case in arm["cases"]
                if case["name"] == "place-search-navigation"
            )
            case["turns"] = [turn for turn in case["turns"] if turn["turn"] == 1]
            case["gates"] = [gate for gate in case["gates"] if gate["turn"] == 1]

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertFalse(result["candidates"]["opus"]["eligible"])

    def test_missing_model_provenance_blocks_both_arms(self):
        sol = report([1.0] * 10)
        opus = opus_report([2.0] * 10)
        for arm in (sol, opus):
            del arm["requested_model"]
            for case in arm["cases"]:
                for turn in case["turns"]:
                    del turn["model_provenance"]

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertFalse(result["candidates"]["opus"]["eligible"])

    def test_wrong_report_model_and_unverified_turn_provenance_block_arm(self):
        sol = report([1.0] * 10)
        sol["requested_model"] = "claude-opus-5-5"
        sol["cases"][0]["turns"][0]["model_provenance"] = {
            "requested_model": "chatgpt/sol-fast",
            "model_calls": [],
            "verified": False,
        }
        opus = opus_report([2.0] * 10)

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertTrue(result["candidates"]["opus"]["eligible"])
        self.assertTrue(any("requested_model" in reason for reason in result["candidates"]["sol"]["reasons"]))
        self.assertTrue(any("not verified" in reason for reason in result["candidates"]["sol"]["reasons"]))

    def test_observed_model_must_match_the_selected_arm(self):
        sol = report([1.0] * 10)
        sol["cases"][0]["turns"][0]["model_provenance"]["model_calls"][0]["observed_model"] = OPUS_MODEL

        result = compare_reports(sol, opus_report([2.0] * 10))

        self.assertIsNone(result["winner"])
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertTrue(any("observed model" in reason for reason in result["candidates"]["sol"]["reasons"]))

    def test_opus_fast_speed_or_fast_mode_disqualifies_that_arm(self):
        sol = report([1.5] * 10)
        opus = opus_report([1.0] * 10)
        opus_call = opus["cases"][0]["turns"][0]["model_provenance"]["model_calls"][0]
        opus_call["speed"] = "fast"
        opus_call["fast_mode_state"] = "on"

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertFalse(result["candidates"]["opus"]["eligible"])
        self.assertTrue(any("speed" in reason for reason in result["candidates"]["opus"]["reasons"]))
        self.assertTrue(any("fast mode" in reason for reason in result["candidates"]["opus"]["reasons"]))

    def test_fastest_eligible_action_receipt_p50_is_informational(self):
        sol = report([1.0] * 10)
        opus = opus_report([1.5] * 10)

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertTrue(result["candidates"]["sol"]["eligible"])
        self.assertTrue(result["candidates"]["opus"]["eligible"])
        self.assertEqual(result["candidates"]["sol"]["action_receipt_p50_seconds"], 1.0)
        self.assertEqual(result["candidates"]["opus"]["action_receipt_p50_seconds"], 1.5)
        self.assertEqual(result["fastest_eligible"], ["sol"])
        self.assertEqual(result["decision"], "informational_fastest_eligible")

    def test_within_ten_percent_has_no_sol_tie_preference(self):
        sol = report([1.0] * 10)
        opus = opus_report([1.1] * 10)

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertEqual(result["fastest_eligible"], ["sol"])
        self.assertEqual(result["decision"], "informational_fastest_eligible")

    def test_missing_latency_observations_disqualify_without_becoming_latency_only(self):
        sol = report([1.0] * 10)
        turn = sol["cases"][0]["turns"][0]
        turn["latency_seconds"]["first_speech"] = None
        turn["latency_seconds"]["command_receipt"] = None
        sol["cases"][0]["gates"][0]["run_count"] = 9
        sol["cases"][0]["gates"][0]["command_receipt_p50_seconds"] = 1.0
        sol["cases"][0]["gates"][0]["command_receipt_p95_seconds"] = 1.0

        result = compare_reports(sol, opus_report([2.0] * 10))

        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertTrue(any("observations were missing" in reason for reason in result["candidates"]["sol"]["reasons"]))

    def test_exact_latency_tie_reports_both_arms_without_sol_preference(self):
        result = compare_reports(report([1.0] * 10), opus_report([1.0] * 10))

        self.assertIsNone(result["winner"])
        self.assertEqual(result["fastest_eligible"], ["sol", "opus"])

    def test_reports_with_fewer_than_ten_runs_are_ineligible(self):
        sol = report([1.0, 1.0])
        opus = opus_report([2.0, 2.0])

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertFalse(result["candidates"]["opus"]["eligible"])
        self.assertIn("10 runs", " ".join(result["candidates"]["sol"]["reasons"]))

    def test_candidate_is_ineligible_for_missing_runs_or_report_failures(self):
        sol = report([1.0] * 9, run_count=10)
        opus = opus_report([2.0] * 10, passed=False, failures=["provenance: model did not match"])

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertFalse(result["candidates"]["opus"]["eligible"])
        self.assertIn("observed", " ".join(result["candidates"]["sol"]["reasons"]))
        self.assertFalse(result["candidates"]["sol"]["correctness"]["passed"])
        self.assertIn(
            "one or more runs were not observed",
            " ".join(result["candidates"]["sol"]["correctness"]["failures"]),
        )
        self.assertIn("provenance", " ".join(result["candidates"]["opus"]["reasons"]))
        self.assertFalse(result["candidates"]["opus"]["correctness"]["passed"])
        self.assertEqual(result["decision"], "no_eligible_candidate")
        self.assertIn("no candidate is eligible", result["message"].lower())

    def test_capture_failures_make_a_candidate_ineligible_even_if_report_claims_passed(self):
        sol = report([1.0] * 10)
        sol["cases"][0]["capture_failures"] = [{"run": 1, "turn": 1, "message": "capture failed"}]

        result = compare_reports(sol, opus_report([2.0] * 10))

        self.assertIsNone(result["winner"])
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertFalse(result["candidates"]["sol"]["correctness"]["passed"])
        self.assertIn(
            "one or more capture failures",
            " ".join(result["candidates"]["sol"]["correctness"]["failures"]),
        )
        self.assertIn("capture failures", " ".join(result["candidates"]["sol"]["reasons"]))

    def test_latency_case_gate_failures_do_not_disqualify_a_candidate(self):
        sol = report([1.0] * 10, case_failures=[
            "timer-300-seconds turn 1: command-receipt p95 4s exceeds 3s"
        ])
        opus = opus_report([2.0] * 10)

        result = compare_reports(sol, opus)

        self.assertIsNone(result["winner"])
        self.assertTrue(result["candidates"]["sol"]["eligible"])
        self.assertTrue(result["candidates"]["opus"]["eligible"])

    def test_rejects_unmatched_malformed_action_gate(self):
        sol = report([1.0] * 10)
        sol["cases"][0]["gates"].append({
            "turn": 99,
            "kind": "action",
            "run_count": "malformed",
            "command_receipt_p50_seconds": "malformed",
        })

        with self.assertRaises(ValueError):
            compare_reports(sol, opus_report([2.0] * 10))

    def test_missing_or_malformed_action_metrics_are_rejected(self):
        for bad_value in ("1.2", True, float("nan"), float("inf"), -0.1):
            with self.subTest(value=bad_value):
                sol = report([1.0] * 10)
                sol["cases"][0]["turns"][0]["latency_seconds"]["command_receipt"] = bad_value
                with self.assertRaises(ValueError):
                    compare_reports(sol, opus_report([2.0] * 10))

        missing = report([1.0] * 10)
        del missing["cases"][0]["turns"][0]["latency_seconds"]["command_receipt"]
        with self.assertRaises(ValueError):
            compare_reports(missing, opus_report([2.0] * 10))

    def test_fastest_eligible_arm_is_informational_in_three_arm_comparison(self):
        result = compare_reports(
            report([1.0] * 10),
            opus_report([1.4] * 10),
            sonnet_report([0.8] * 10),
        )

        self.assertIsNone(result["winner"])
        self.assertEqual(result["fastest_eligible"], ["sonnet"])
        self.assertEqual(result["decision"], "informational_fastest_eligible")
        self.assertTrue(result["candidates"]["sonnet"]["eligible"])

    def test_slightly_slower_sol_does_not_replace_fastest_sonnet(self):
        result = compare_reports(
            report([1.05] * 10),
            opus_report([1.4] * 10),
            sonnet_report([1.0] * 10),
        )

        self.assertIsNone(result["winner"])
        self.assertEqual(result["fastest_eligible"], ["sonnet"])
        self.assertEqual(result["decision"], "informational_fastest_eligible")

    def test_sonnet_fastest_and_eligible_with_dated_model_alias(self):
        sonnet = sonnet_report([0.8] * 10)
        sonnet["cases"][0]["turns"][0]["model_provenance"]["model_calls"][0][
            "observed_model"
        ] = "claude-sonnet-5-5-20260928"
        sonnet["cases"][0]["turns"][0]["model_provenance"]["model_calls"][0].pop(
            "fast_mode_state"
        )

        result = compare_reports(report([1.0] * 10), opus_report([1.4] * 10), sonnet)

        self.assertIsNone(result["winner"])
        self.assertTrue(result["candidates"]["sonnet"]["eligible"])

    def test_sonnet_unproven_standard_speed_makes_arm_ineligible(self):
        sonnet = sonnet_report([0.8] * 10)
        call = sonnet["cases"][0]["turns"][0]["model_provenance"]["model_calls"][0]
        call["service_tier"] = None
        call["speed"] = None

        result = compare_reports(report([1.0] * 10), opus_report([1.4] * 10), sonnet)

        self.assertIsNone(result["winner"])
        self.assertFalse(result["candidates"]["sonnet"]["eligible"])
        self.assertTrue(any("standard" in reason for reason in result["candidates"]["sonnet"]["reasons"]))

    def test_no_eligible_arm_in_three_arm_comparison_has_no_winner(self):
        result = compare_reports(
            report([1.0] * 9),
            opus_report([2.0] * 9),
            sonnet_report([0.8] * 9),
        )

        self.assertIsNone(result["winner"])
        self.assertEqual(result["decision"], "no_eligible_candidate")
        self.assertFalse(any(candidate["eligible"] for candidate in result["candidates"].values()))

    def test_cli_reads_three_json_reports_and_prints_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("sol.json", "opus.json", "sonnet.json")]
            reports = [report([1.0] * 10), opus_report([1.4] * 10), sonnet_report([0.8] * 10)]
            for path, candidate_report in zip(paths, reports, strict=True):
                path.write_text(json.dumps(candidate_report), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(ROOT / "voice" / "evals" / "compare.py"), *(str(path) for path in paths)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertIsNone(result["winner"])
        self.assertEqual(result["fastest_eligible"], ["sonnet"])

    def test_cli_rejects_report_counts_other_than_two_or_three(self):
        compare_script = str(ROOT / "voice" / "evals" / "compare.py")
        for arguments in ([], ["one.json"], ["one.json", "two.json", "three.json", "four.json"]):
            with self.subTest(arguments=arguments):
                completed = subprocess.run(
                    [sys.executable, compare_script, *arguments],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn("usage:", completed.stderr)

    def test_cli_reads_two_json_reports_and_prints_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            sol_path = Path(directory) / "sol.json"
            opus_path = Path(directory) / "opus.json"
            sol_path.write_text(json.dumps(report([1.0] * 10)), encoding="utf-8")
            opus_path.write_text(json.dumps(opus_report([1.1] * 10)), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(ROOT / "voice" / "evals" / "compare.py"), str(sol_path), str(opus_path)],
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertIsNone(result["winner"])
        self.assertEqual(result["fastest_eligible"], ["sol"])
        self.assertEqual(result["decision"], "informational_fastest_eligible")


if __name__ == "__main__":
    unittest.main()
