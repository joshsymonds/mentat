"""Offline comparison of scored Sol and Opus voice-evaluation reports."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.scenarios import SCENARIOS


RUNS_REQUIRED = 10
EXPECTED_MODELS = {
    "sol": "chatgpt/sol-fast",
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
}
SCENARIO_TURN_COUNTS = {scenario.name: len(scenario.turns) for scenario in SCENARIOS}


def _positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_latency(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{label} must be a finite non-negative number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite non-negative number") from error
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return numeric


def _p50(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot calculate p50 without action receipt observations")
    ordered = sorted(values)
    return ordered[math.ceil(0.5 * len(ordered)) - 1]


def _string_failures(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a list of strings")
    return value


def _model_matches(requested_model: str, observed_model: Any) -> bool:
    if requested_model == "chatgpt/sol-fast":
        return observed_model in (requested_model, "gpt-6-sol")
    if requested_model in ("claude-opus-5-5", "claude-sonnet-5-5"):
        return (
            observed_model == requested_model
            or isinstance(observed_model, str)
            and re.fullmatch(re.escape(requested_model) + r"-[0-9]{8}", observed_model) is not None
        )
    return False


def _model_call_failures(call: Any, requested_model: str, label: str) -> list[str]:
    if not isinstance(call, dict):
        return [f"{label}: model-call provenance is malformed"]

    failures: list[str] = []
    required_fields = {
        "id", "requested_model", "observed_model", "service_tier",
        "result_service_tier", "speed", "failures",
    }
    if requested_model != "claude-sonnet-5-5":
        required_fields.add("fast_mode_state")
    if not required_fields.issubset(call):
        failures.append(f"{label}: model-call provenance is incomplete")
    if not isinstance(call.get("id"), str) or not call["id"]:
        failures.append(f"{label}: model-call id is unproven")
    if call.get("requested_model") != requested_model:
        failures.append(f"{label}: requested model does not match this arm")
    if not _model_matches(requested_model, call.get("observed_model")):
        failures.append(f"{label}: observed model does not match this arm")
    if call.get("result_service_tier") != "standard":
        failures.append(f"{label}: result service tier is unproven or nonstandard")

    if requested_model in ("claude-opus-5-5", "claude-sonnet-5-5"):
        if call.get("service_tier") != "standard":
            failures.append(f"{label}: message service tier is unproven or nonstandard")
        if call.get("speed") != "standard":
            failures.append(f"{label}: speed is unproven or nonstandard")
        if requested_model == "claude-opus-5-5" and call.get("fast_mode_state") != "off":
            failures.append(f"{label}: fast mode is unproven or enabled")
        if (
            requested_model == "claude-sonnet-5-5"
            and call.get("fast_mode_state") not in (None, "off")
        ):
            failures.append(f"{label}: fast mode is enabled")
    elif call.get("service_tier") not in (None, "standard"):
        failures.append(f"{label}: message service tier is nonstandard")

    call_failures = call.get("failures")
    if not isinstance(call_failures, list) or any(
        not isinstance(failure, str) for failure in call_failures
    ):
        failures.append(f"{label}: model-call failure evidence is malformed")
    elif call_failures:
        failures.extend(f"{label}: {failure}" for failure in call_failures)
    return failures


def _turn_provenance_failures(
    turn: dict[str, Any], requested_model: str, label: str
) -> list[str]:
    provenance = turn.get("model_provenance")
    if not isinstance(provenance, dict):
        return [f"{label}: model provenance is missing"]

    failures: list[str] = []
    if provenance.get("requested_model") != requested_model:
        failures.append(f"{label}: provenance requested model does not match this arm")
    if provenance.get("verified") is not True:
        failures.append(f"{label}: model provenance is not verified")
    calls = provenance.get("model_calls")
    if not isinstance(calls, list) or not calls:
        failures.append(f"{label}: model-call provenance is missing")
        return failures
    call_count = turn.get("model_call_count")
    if (
        not isinstance(call_count, int)
        or isinstance(call_count, bool)
        or call_count != len(calls)
    ):
        failures.append(f"{label}: model-call provenance count does not match the turn")
    for call_index, call in enumerate(calls, start=1):
        failures.extend(
            _model_call_failures(call, requested_model, f"{label} call {call_index}")
        )
    return failures


def _unattributed_provenance_failures(
    case: dict[str, Any], requested_model: str, expected_runs: int, label: str
) -> list[str]:
    groups = case.get("unattributed_model_calls", [])
    if not isinstance(groups, list):
        return [f"{label}: unattributed model-call provenance is malformed"]

    failures: list[str] = []
    for group_index, group in enumerate(groups, start=1):
        group_label = f"{label} unattributed group {group_index}"
        if not isinstance(group, dict):
            failures.append(f"{group_label}: provenance is malformed")
            continue
        run_number = group.get("run")
        if (
            not isinstance(run_number, int)
            or isinstance(run_number, bool)
            or not 1 <= run_number <= expected_runs
        ):
            failures.append(f"{group_label}: run is invalid")
        calls = group.get("model_calls")
        if not isinstance(calls, list) or not calls:
            failures.append(f"{group_label}: model-call evidence is missing")
            continue
        for call_index, call in enumerate(calls, start=1):
            call_label = f"{group_label} call {call_index}"
            turn_number = call.get("turn") if isinstance(call, dict) else None
            if (
                not isinstance(turn_number, int)
                or isinstance(turn_number, bool)
                or turn_number <= 0
            ):
                failures.append(f"{call_label}: turn is invalid")
            failures.extend(_model_call_failures(call, requested_model, call_label))
    return failures


def _score_candidate(report: Any, candidate: str) -> tuple[dict[str, Any], set[tuple[str, int]]]:
    if not isinstance(report, dict):
        raise ValueError(f"{candidate} report must be a JSON object")
    if not isinstance(report.get("passed"), bool):
        raise ValueError(f"{candidate} report passed must be a boolean")

    reasons: list[str] = []
    requested_model = EXPECTED_MODELS[candidate]
    if report.get("requested_model") != requested_model:
        reasons.append(f"requested_model must be {requested_model}")
    failures = _string_failures(report.get("failures"), f"{candidate} report failures")
    if not report["passed"]:
        reasons.append("report passed is false")
    reasons.extend(f"report failure: {failure}" for failure in failures)

    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{candidate} report cases must be a non-empty list")

    all_receipts: list[float] = []
    action_signatures: set[tuple[str, int]] = set()
    seen_case_names: set[str] = set()

    for case_index, case in enumerate(cases, start=1):
        label = f"{candidate} case {case_index}"
        if not isinstance(case, dict):
            raise ValueError(f"{label} must be an object")
        name = case.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} name must be a non-empty string")
        if name in seen_case_names:
            raise ValueError(f"{candidate} report contains duplicate case {name!r}")
        seen_case_names.add(name)
        if name not in SCENARIO_TURN_COUNTS:
            reasons.append(f"{name}: scenario is not in the contracted eval suite")

        expected_runs = _positive_integer(case.get("run_count"), f"{label} run_count")
        if expected_runs != RUNS_REQUIRED:
            reasons.append(
                f"{name}: expected {RUNS_REQUIRED} runs, found {expected_runs}"
            )
        case_failures = _string_failures(case.get("failures"), f"{label} failures")
        reasons.extend(f"{name}: {failure}" for failure in case_failures)
        capture_failures = case.get("capture_failures")
        if not isinstance(capture_failures, list):
            raise ValueError(f"{label} capture_failures must be a list")
        if capture_failures:
            reasons.append(f"{name}: one or more capture failures were reported")
        turns = case.get("turns")
        if not isinstance(turns, list):
            raise ValueError(f"{label} turns must be a list")
        gates = case.get("gates")
        if not isinstance(gates, list):
            raise ValueError(f"{label} gates must be a list")

        records_by_turn: dict[int, dict[int, str]] = {}
        turn_records: dict[tuple[int, int], dict[str, Any]] = {}
        action_turns: set[int] = set()
        for turn_index, turn in enumerate(turns, start=1):
            turn_label = f"{label} turn record {turn_index}"
            if not isinstance(turn, dict):
                raise ValueError(f"{turn_label} must be an object")
            run_number = _positive_integer(turn.get("run"), f"{turn_label} run")
            turn_number = _positive_integer(turn.get("turn"), f"{turn_label} turn")
            if run_number > expected_runs:
                raise ValueError(f"{turn_label} run exceeds case run_count")
            kind = turn.get("kind")
            if kind not in ("action", "search"):
                raise ValueError(f"{turn_label} kind must be action or search")
            if run_number in records_by_turn.setdefault(turn_number, {}):
                raise ValueError(f"{turn_label} duplicates run {run_number}, turn {turn_number}")
            records_by_turn[turn_number][run_number] = kind
            turn_records[(run_number, turn_number)] = turn
            if kind == "action":
                action_turns.add(turn_number)

        if not turns:
            reasons.append(f"{name}: no run observations")
        observed_runs = {
            run_number
            for run_records in records_by_turn.values()
            for run_number in run_records
        }
        if observed_runs != set(range(1, expected_runs + 1)):
            reasons.append(f"{name}: one or more runs were not observed")
        expected_turn_count = SCENARIO_TURN_COUNTS.get(name)
        if expected_turn_count is not None:
            expected_turns = set(range(1, expected_turn_count + 1))
            if set(records_by_turn) != expected_turns:
                reasons.append(f"{name}: scenario turn observations are incomplete")
        for (run_number, turn_number), turn_record in turn_records.items():
            reasons.extend(
                _turn_provenance_failures(
                    turn_record,
                    requested_model,
                    f"{name} run {run_number} turn {turn_number}",
                )
            )
        reasons.extend(
            _unattributed_provenance_failures(
                case, requested_model, expected_runs, name
            )
        )

        for turn_number, run_records in records_by_turn.items():
            kinds = set(run_records.values())
            if len(kinds) != 1:
                raise ValueError(f"{name} turn {turn_number} kind differs across runs")
            if set(run_records) != set(range(1, expected_runs + 1)):
                reasons.append(f"{name} turn {turn_number}: not every run was observed")

        gates_by_turn: dict[int, dict[str, Any]] = {}
        for gate_index, gate in enumerate(gates, start=1):
            gate_label = f"{label} gate {gate_index}"
            if not isinstance(gate, dict):
                raise ValueError(f"{gate_label} must be an object")
            gate_turn = _positive_integer(gate.get("turn"), f"{gate_label} turn")
            if gate_turn in gates_by_turn:
                raise ValueError(f"{label} contains duplicate gate for turn {gate_turn}")
            gates_by_turn[gate_turn] = gate
            observed_kinds = records_by_turn.get(gate_turn)
            if observed_kinds is None:
                raise ValueError(f"{label} references unobserved turn {gate_turn}")
            if gate.get("kind") not in set(observed_kinds.values()):
                raise ValueError(f"{label} kind does not match observed turn {gate_turn}")

        for turn_number in sorted(action_turns):
            action_signatures.add((name, turn_number))
            observed = records_by_turn[turn_number]
            receipt_values: list[float] = []
            missing_receipts = 0
            for run_number, turn_kind in observed.items():
                if turn_kind != "action":
                    continue
                turn_record = turn_records[(run_number, turn_number)]
                latency = turn_record.get("latency_seconds")
                if not isinstance(latency, dict):
                    raise ValueError(f"{name} run {run_number} turn {turn_number} latency_seconds must be an object")
                if "command_receipt" not in latency:
                    raise ValueError(f"{name} run {run_number} turn {turn_number} is missing command_receipt")
                value = latency["command_receipt"]
                if value is None:
                    missing_receipts += 1
                else:
                    receipt_values.append(
                        _finite_latency(value, f"{name} run {run_number} turn {turn_number} command_receipt")
                    )

            gate = gates_by_turn.get(turn_number)
            if gate is None:
                raise ValueError(f"{name} turn {turn_number} is missing its action gate")
            if gate.get("kind") != "action":
                raise ValueError(f"{name} turn {turn_number} gate kind must be action")
            gate_run_count = gate.get("run_count")
            if not isinstance(gate_run_count, int) or isinstance(gate_run_count, bool) or gate_run_count < 0:
                raise ValueError(f"{name} turn {turn_number} gate run_count must be a non-negative integer")
            if gate_run_count != len(receipt_values):
                reasons.append(
                    f"{name} turn {turn_number}: action gate observed {gate_run_count} of "
                    f"{len(receipt_values)} command receipts"
                )
            gate_p50 = gate.get("command_receipt_p50_seconds")
            if receipt_values:
                gate_p50_value = _finite_latency(
                    gate_p50, f"{name} turn {turn_number} command_receipt_p50_seconds"
                )
                if gate_run_count == len(receipt_values) and gate_p50_value != _p50(receipt_values):
                    raise ValueError(f"{name} turn {turn_number} action gate p50 does not match its observations")
            elif gate_p50 is not None:
                raise ValueError(f"{name} turn {turn_number} action gate p50 has no observations")
            if missing_receipts:
                reasons.append(
                    f"{name} turn {turn_number}: {missing_receipts} command receipts were not observed"
                )
            all_receipts.extend(receipt_values)

    missing_scenarios = set(SCENARIO_TURN_COUNTS) - seen_case_names
    if missing_scenarios:
        reasons.append(
            "missing contracted scenarios: " + ", ".join(sorted(missing_scenarios))
        )
    if not action_signatures:
        raise ValueError(f"{candidate} report has no action command-receipt metrics")

    result = {
        "eligible": not reasons and len(all_receipts) > 0,
        "action_receipt_p50_seconds": _p50(all_receipts) if all_receipts else None,
        "reasons": reasons,
    }
    if not all_receipts and not reasons:
        result["eligible"] = False
        result["reasons"].append("no action command-receipt observations")
    return result, action_signatures


def compare_reports(
    sol_report: Any, opus_report: Any, sonnet_report: Any | None = None
) -> dict[str, Any]:
    """Compare two or three scored reports and select an eligible winner."""
    scored = {
        "sol": _score_candidate(sol_report, "sol"),
        "opus": _score_candidate(opus_report, "opus"),
    }
    if sonnet_report is not None:
        scored["sonnet"] = _score_candidate(sonnet_report, "sonnet")

    eligible_signatures = [
        (name, signatures)
        for name, (candidate, signatures) in scored.items()
        if candidate["eligible"]
    ]
    if eligible_signatures:
        expected_signatures = eligible_signatures[0][1]
        for name, signatures in eligible_signatures[1:]:
            if signatures != expected_signatures:
                first_name = eligible_signatures[0][0]
                raise ValueError(
                    f"{first_name.title()} and {name.title()} reports do not contain "
                    "the same action cases and turns"
                )

    candidates = {name: result[0] for name, result in scored.items()}
    eligible = [
        name for name, candidate in candidates.items() if candidate["eligible"]
    ]
    if not eligible:
        return {
            "winner": None,
            "decision": "no_eligible_candidate",
            "message": "No winner can be deployed because no candidate is eligible.",
            "candidates": candidates,
        }
    if len(eligible) == 1:
        winner = eligible[0]
        return {
            "winner": winner,
            "decision": "only_eligible_candidate",
            "message": f"{winner} is the only eligible candidate.",
            "candidates": candidates,
        }

    fastest = min(
        eligible, key=lambda name: candidates[name]["action_receipt_p50_seconds"]
    )
    fastest_p50 = candidates[fastest]["action_receipt_p50_seconds"]
    sol_eligible = candidates["sol"]["eligible"]
    sol_p50 = candidates["sol"]["action_receipt_p50_seconds"]
    if sol_eligible and sol_p50 <= fastest_p50 * 1.1:
        winner = "sol"
        decision = (
            "within_10_percent_tie"
            if fastest != "sol"
            or any(
                name != "sol"
                and candidates[name]["action_receipt_p50_seconds"] <= sol_p50 * 1.1
                for name in eligible
            )
            else "faster_action_receipt_p50"
        )
        message = (
            "Action command-receipt p50s are within 10%; Sol wins the tie-break."
            if decision == "within_10_percent_tie"
            else "Sol has the faster action command-receipt p50."
        )
    else:
        winner = fastest
        decision = "faster_action_receipt_p50"
        label = {"sol": "Sol", "opus": "Opus", "sonnet": "Sonnet"}[winner]
        message = f"{label} has the fastest action command-receipt p50."
    return {"winner": winner, "decision": decision, "message": message, "candidates": candidates}


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) not in (2, 3):
        print(
            "usage: compare.py SOL_REPORT.json OPUS_REPORT.json [SONNET_REPORT.json]",
            file=sys.stderr,
        )
        return 2
    try:
        reports = [
            json.loads(Path(argument).read_text(encoding="utf-8"))
            for argument in arguments
        ]
        result = compare_reports(*reports)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
