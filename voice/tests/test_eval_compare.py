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
    observed_model = "gpt-6-sol" if requested_model == SOL_MODEL else OPUS_MODEL
    is_opus = requested_model == OPUS_MODEL
    calls = []
    for index in range(len(latencies)):
        calls.append({
            "id": f"call-{index + 1}",
            "requested_model": requested_model,
            "observed_model": observed_model,
            "service_tier": "standard" if is_opus else None,
            "result_service_tier": "standard",
            "speed": "standard" if is_opus else None,
            "fast_mode_state": "off" if is_opus else None,
            "failures": [],
        })

    cases = []
    for scenario in SCENARIOS:
        turns = []
        gates = []
        for turn_number in range(1, len(scenario.turns) + 1):
            p50 = sorted(latencies)[(len(latencies) + 1) // 2 - 1] if latencies else None
            gates.append({
                "turn": turn_number,
                "kind": "action",
                "run_count": len(latencies),
                "command_receipt_p50_seconds": p50,
            })
            for index, latency in enumerate(latencies):
                turns.append({
                    "run": index + 1,
                    "turn": turn_number,
                    "kind": "action",
                    "model_call_count": 1,
                    "model_provenance": {
                        "requested_model": requested_model,
                        "verified": True,
                        "model_calls": [{**calls[index]}],
                    },
                    "latency_seconds": {"command_receipt": latency},
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


class CompareTests(unittest.TestCase):
    def test_omitting_contracted_scenarios_makes_that_arm_ineligible(self):
        sol = report([1.0] * 10)
        opus = opus_report([2.0] * 10)
        opus["cases"] = opus["cases"][:1]

        result = compare_reports(sol, opus)

        self.assertEqual(result["winner"], "sol")
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

        self.assertEqual(result["winner"], "opus")
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertTrue(result["candidates"]["opus"]["eligible"])
        self.assertTrue(any("requested_model" in reason for reason in result["candidates"]["sol"]["reasons"]))
        self.assertTrue(any("not verified" in reason for reason in result["candidates"]["sol"]["reasons"]))

    def test_observed_model_must_match_the_selected_arm(self):
        sol = report([1.0] * 10)
        sol["cases"][0]["turns"][0]["model_provenance"]["model_calls"][0]["observed_model"] = OPUS_MODEL

        result = compare_reports(sol, opus_report([2.0] * 10))

        self.assertEqual(result["winner"], "opus")
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertTrue(any("observed model" in reason for reason in result["candidates"]["sol"]["reasons"]))

    def test_opus_fast_speed_or_fast_mode_disqualifies_that_arm(self):
        sol = report([1.5] * 10)
        opus = opus_report([1.0] * 10)
        opus_call = opus["cases"][0]["turns"][0]["model_provenance"]["model_calls"][0]
        opus_call["speed"] = "fast"
        opus_call["fast_mode_state"] = "on"

        result = compare_reports(sol, opus)

        self.assertEqual(result["winner"], "sol")
        self.assertFalse(result["candidates"]["opus"]["eligible"])
        self.assertTrue(any("speed" in reason for reason in result["candidates"]["opus"]["reasons"]))
        self.assertTrue(any("fast mode" in reason for reason in result["candidates"]["opus"]["reasons"]))

    def test_faster_eligible_action_receipt_p50_wins(self):
        sol = report([1.0] * 10)
        opus = opus_report([1.5] * 10)

        result = compare_reports(sol, opus)

        self.assertEqual(result["winner"], "sol")
        self.assertTrue(result["candidates"]["sol"]["eligible"])
        self.assertTrue(result["candidates"]["opus"]["eligible"])
        self.assertEqual(result["candidates"]["sol"]["action_receipt_p50_seconds"], 1.0)
        self.assertEqual(result["candidates"]["opus"]["action_receipt_p50_seconds"], 1.5)

    def test_within_ten_percent_tie_favors_sol(self):
        sol = report([1.0] * 10)
        opus = opus_report([1.1] * 10)

        result = compare_reports(sol, opus)

        self.assertEqual(result["winner"], "sol")
        self.assertEqual(result["decision"], "within_10_percent_tie")

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
        self.assertIn("provenance", " ".join(result["candidates"]["opus"]["reasons"]))
        self.assertEqual(result["decision"], "no_eligible_candidate")
        self.assertIn("no winner can be deployed", result["message"].lower())

    def test_capture_failures_make_a_candidate_ineligible_even_if_report_claims_passed(self):
        sol = report([1.0] * 10)
        sol["cases"][0]["capture_failures"] = [{"run": 1, "turn": 1, "message": "capture failed"}]

        result = compare_reports(sol, opus_report([2.0] * 10))

        self.assertEqual(result["winner"], "opus")
        self.assertFalse(result["candidates"]["sol"]["eligible"])
        self.assertIn("capture failures", " ".join(result["candidates"]["sol"]["reasons"]))

    def test_case_gate_failures_make_a_candidate_ineligible(self):
        sol = report([1.0] * 10, case_failures=["command-receipt p95 exceeded limit"])
        opus = opus_report([2.0] * 10)

        result = compare_reports(sol, opus)

        self.assertEqual(result["winner"], "opus")
        self.assertFalse(result["candidates"]["sol"]["eligible"])
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
        self.assertEqual(result["winner"], "sol")
        self.assertEqual(result["decision"], "within_10_percent_tie")


if __name__ == "__main__":
    unittest.main()
