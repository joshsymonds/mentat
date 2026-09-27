"""Offline scoring for scripted voice-evaluation observations."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ACTION_P50_LIMIT_SECONDS = 3.5
ACTION_P95_LIMIT_SECONDS = 5.0
SEARCH_P50_LIMIT_SECONDS = 10.0
CONFIRMATION_DEADLINE_SECONDS = 30.0
HANGUP_DEADLINE_SECONDS = 60.0
RUNS_REQUIRED = 10
NO_ANSWER_FAILURE = (
    "no-answer: captured PCM is silent, malformed, or has no qualifying post-playout onset"
)
PARTIAL_CAPTURE_MESSAGES = {
    "answer transcription exceeded its deadline",
    NO_ANSWER_FAILURE,
    "room deletion was not observed before deadline",
}
SCRIPTED_TTS_TIMEOUT_PATTERN = re.compile(
    r"scripted speech synthesis for line ([1-9][0-9]*) exceeded its deadline"
)


def _is_scripted_tts_timeout(message: Any) -> bool:
    return (
        isinstance(message, str)
        and SCRIPTED_TTS_TIMEOUT_PATTERN.fullmatch(message) is not None
    )


def nearest_rank(values: list[float], percentile: float) -> float:
    """Return the nearest-rank percentile, with rank ceil(p * n)."""
    if not values:
        raise ValueError("nearest-rank requires at least one value")
    if not isinstance(percentile, (int, float)) or isinstance(percentile, bool):
        raise ValueError("percentile must be a number in (0, 1]")
    if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in values):
        raise ValueError("values must be finite numbers")
    try:
        percentile = float(percentile)
        ordered = [float(value) for value in values]
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError("percentile and values must be finite numbers") from error
    if not math.isfinite(percentile) or not 0 < percentile <= 1:
        raise ValueError("percentile must be a number in (0, 1]")
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("values must be finite numbers")
    ordered.sort()
    rank = math.ceil(percentile * len(ordered))
    return ordered[rank - 1]


def _timestamp(turn: dict[str, Any], field: str) -> float:
    value = turn.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"missing or invalid {field} timestamp")
    try:
        timestamp = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"missing or invalid {field} timestamp") from error
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError(f"missing or invalid {field} timestamp")
    return timestamp


def _transcript_segments(segments: Any) -> list[dict[str, Any]]:
    if not isinstance(segments, list) or not segments:
        raise ValueError("missing or invalid transcript segment evidence")
    evidence = []
    for segment in segments:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("invalid transcript segment evidence")
        start = _timestamp(segment, "start")
        end = _timestamp(segment, "end")
        if end < start:
            raise ValueError("invalid transcript segment bounds")
        evidence.append({"start": start, "end": end, "text": segment["text"]})
    return evidence


def _score_turn(
    turn: Any, name: str, run_index: int, turn_index: int
) -> tuple[dict[str, Any], list[str]]:
    label = f"{name} run {run_index + 1} turn {turn_index + 1}"
    if not isinstance(turn, dict):
        raise ValueError(f"{label}: turn observation must be an object")
    kind = turn.get("kind")
    if kind not in ("action", "search"):
        raise ValueError(f"missing kind or invalid turn kind {kind!r}")
    speech_end = _timestamp(turn, "speech_end")
    first_audio = _timestamp(turn, "first_audio")
    problems: list[str] = []
    answer_at = None
    if "answer_at" not in turn or turn["answer_at"] is None:
        problems.append(f"{label}: missing answer_at timestamp")
    else:
        try:
            answer_at = _timestamp(turn, "answer_at")
        except ValueError:
            problems.append(f"{label}: missing or invalid answer_at timestamp")
    command_received_at = None
    command_speech_end = None
    if "command_received_at" not in turn:
        problems.append(f"{label}: missing command receipt timestamp declaration")
    elif turn["command_received_at"] is None:
        if kind == "action":
            problems.append(f"{label}: missing command receipt timestamp")
    else:
        try:
            command_received_at = _timestamp(turn, "command_received_at")
        except ValueError:
            problems.append(f"{label}: missing or invalid command receipt timestamp")
    if kind == "action" or command_received_at is not None:
        try:
            command_speech_end = _timestamp(turn, "speech_end_wall")
        except ValueError:
            problems.append(f"{label}: missing or invalid speech_end_wall timestamp")
    if "overlap" not in turn or not isinstance(turn["overlap"], bool):
        raise ValueError(f"{label}: missing or invalid overlap observation")
    overlap = turn["overlap"]
    if "expect_confirmation" not in turn:
        raise ValueError(f"{label}: missing expect_confirmation declaration")
    if "expect_hangup" not in turn:
        raise ValueError(f"{label}: missing expect_hangup declaration")
    expect_confirmation = turn["expect_confirmation"]
    expect_hangup = turn["expect_hangup"]
    if not isinstance(expect_confirmation, bool):
        raise ValueError(f"{label}: expect_confirmation must be a boolean")
    if not isinstance(expect_hangup, bool):
        raise ValueError(f"{label}: expect_hangup must be a boolean")

    calls = turn.get("model_calls")
    if not isinstance(calls, list):
        raise ValueError(f"{label}: missing or invalid model_calls observations")

    event_times: dict[str, float | None] = {}
    for field, expected, deadline in (
        ("confirmation", expect_confirmation, CONFIRMATION_DEADLINE_SECONDS),
        ("room_deleted", expect_hangup, HANGUP_DEADLINE_SECONDS),
    ):
        if field not in turn or turn[field] is None:
            event_times[field] = None
            if expected:
                problems.append(f"{label}: expected {field} observation is missing")
            continue
        event_time = _timestamp(turn, field)
        if event_time < speech_end:
            problems.append(f"{label}: {field} precedes speech_end")
        latency = event_time - speech_end
        event_times[field] = event_time
        if expected and latency > deadline:
            problems.append(f"{label}: {field} latency {latency:g}s exceeds {deadline:g}s deadline")
        if field == "room_deleted" and not expected:
            problems.append(f"{label}: room deleted when hang-up was not expected")

    answer_latency = None
    if answer_at is not None:
        answer_latency = answer_at - speech_end
        if answer_latency < 0:
            problems.append(f"{label}: answer precedes speech_end")
    command_latency = None
    if command_received_at is not None and command_speech_end is not None:
        command_latency = command_received_at - command_speech_end
        if command_latency < 0:
            problems.append(f"{label}: command receipt precedes speech_end_wall")
    first_audio_latency = first_audio - speech_end
    report = {
        "run": run_index + 1,
        "turn": turn_index + 1,
        "kind": kind,
        "expect_hangup": expect_hangup,
        "overlap": overlap,
        "latency_seconds": {
            "command_receipt": command_latency,
            "answer": answer_latency,
            "first_audio": first_audio_latency,
            "confirmation": (
                None if event_times["confirmation"] is None
                else event_times["confirmation"] - speech_end
            ),
            "room_deleted": (
                None if event_times["room_deleted"] is None
                else event_times["room_deleted"] - speech_end
            ),
        },
        "model_call_count": len(calls),
    }
    if "raw_segments" in turn:
        report["segments"] = _transcript_segments(turn["raw_segments"])
    return report, problems


def score_observations(
    observations: Any, required_runs: int = RUNS_REQUIRED
) -> dict[str, Any]:
    """Score scenario cases and fail closed for every missing or invalid datum."""
    if (
        not isinstance(required_runs, int)
        or isinstance(required_runs, bool)
        or required_runs <= 0
    ):
        return {
            "passed": False,
            "failures": ["input: required_runs must be a positive integer"],
            "cases": [],
        }
    failures: list[str] = []
    reports: list[dict[str, Any]] = []
    if not isinstance(observations, dict) or not isinstance(observations.get("cases"), list):
        return {"passed": False, "failures": ["input: expected an object containing a cases list"], "cases": []}
    if not observations["cases"]:
        return {"passed": False, "failures": ["input: cases list must not be empty"], "cases": []}

    for case_index, scenario in enumerate(observations["cases"]):
        if not isinstance(scenario, dict):
            failures.append(f"case {case_index + 1}: scenario must be an object")
            continue
        name = scenario.get("name")
        if not isinstance(name, str) or not name.strip():
            name = f"case {case_index + 1}"
            failures.append(f"{name}: missing scenario name")
        runs = scenario.get("runs")
        if not isinstance(runs, list):
            message = f"{name}: missing or invalid runs"
            failures.append(message)
            reports.append({"name": name, "turns": [], "gates": [], "failures": [message]})
            continue
        case_failures: list[str] = []
        if len(runs) != required_runs:
            case_failures.append(f"{name}: expected {required_runs} runs, found {len(runs)}")
        turns: list[dict[str, Any]] = []
        capture_failures: list[dict[str, Any]] = []
        for run_index, run in enumerate(runs):
            if not isinstance(run, dict) or not isinstance(run.get("turns"), list):
                case_failures.append(f"{name} run {run_index + 1}: missing turn observations")
                continue
            valid_failure = False
            if "failure" in run:
                failure = run["failure"]
                preflight_tts_failure = (
                    isinstance(failure, dict)
                    and _is_scripted_tts_timeout(failure.get("message"))
                )
                failure_fields = (
                    {"turn", "message"},
                    {"turn", "message", "speech_started_at"},
                    {"turn", "message", "speech_started_at", "segments"},
                )
                try:
                    valid_failure = (
                        isinstance(failure, dict)
                        and set(failure) in failure_fields
                        and not isinstance(failure.get("turn"), bool)
                        and isinstance(failure.get("turn"), int)
                        and (
                            (
                                failure.get("message")
                                == "room deletion was not observed before deadline"
                                and failure["turn"] == len(run["turns"])
                                and bool(run["turns"])
                                and isinstance(run["turns"][-1], dict)
                                and run["turns"][-1].get("turn") == failure["turn"]
                                and run["turns"][-1].get("expect_hangup") is True
                                and run["turns"][-1].get("room_deleted") is None
                            )
                            or (
                                failure.get("message")
                                != "room deletion was not observed before deadline"
                                and (
                                    (preflight_tts_failure and not run["turns"] and failure["turn"] == 1)
                                    or (
                                        not preflight_tts_failure
                                        and failure["turn"] == len(run["turns"]) + 1
                                    )
                                )
                            )
                        )
                        and (
                            preflight_tts_failure
                            or failure.get("message") in PARTIAL_CAPTURE_MESSAGES
                        )
                        and (
                            failure.get("message") in {
                                NO_ANSWER_FAILURE,
                                "answer transcription exceeded its deadline",
                            }
                        ) == ("speech_started_at" in failure)
                        and (
                            "segments" not in failure
                            or failure.get("message") in {
                                NO_ANSWER_FAILURE,
                                "answer transcription exceeded its deadline",
                            }
                        )
                    )
                    if valid_failure and "segments" in failure:
                        _transcript_segments(failure["segments"])
                    if valid_failure and "speech_started_at" in failure:
                        failure_start = _timestamp(failure, "speech_started_at")
                        if run["turns"]:
                            previous_turn = run["turns"][-1]
                            if not isinstance(previous_turn, dict):
                                valid_failure = False
                            elif failure_start <= _timestamp(
                                previous_turn, "speech_started_at"
                            ):
                                valid_failure = False
                except (ValueError, TypeError):
                    valid_failure = False
                if not valid_failure:
                    case_failures.append(
                        f"{name} run {run_index + 1}: invalid partial capture failure metadata"
                    )
                    continue
                case_failures.append(
                    f"{name} run {run_index + 1} turn {failure['turn']}: "
                    f"capture failed: {failure['message']}"
                )
                capture_failure = {
                    "run": run_index + 1,
                    "turn": failure["turn"],
                    "message": failure["message"],
                }
                if "segments" in failure:
                    capture_failure["segments"] = _transcript_segments(failure["segments"])
                capture_failures.append(capture_failure)
                product_failures = run.get("product_failures")
                if not isinstance(product_failures, list):
                    case_failures.append(
                        f"{name} run {run_index + 1}: invalid partial product failure metadata"
                    )
                else:
                    for product_failure in product_failures:
                        product_turn = (
                            product_failure.get("turn")
                            if isinstance(product_failure, dict)
                            else None
                        )
                        failed_turn_command = False
                        if product_turn == failure["turn"]:
                            commands = run.get("phone_commands")
                            failed_turn_command = (
                                isinstance(commands, list)
                                and any(
                                    isinstance(command, dict)
                                    and isinstance(command.get("id"), str)
                                    and bool(command["id"])
                                    and isinstance(command.get("kind"), str)
                                    and bool(command["kind"])
                                    and isinstance(command.get("turn"), int)
                                    and not isinstance(command["turn"], bool)
                                    and command["turn"] == product_turn
                                    for command in commands
                                )
                            )
                        valid_product_failure = (
                            isinstance(product_failure, dict)
                            and set(product_failure) == {"turn", "message"}
                            and isinstance(product_turn, int)
                            and not isinstance(product_turn, bool)
                            and (
                                1 <= product_turn <= len(run["turns"])
                                or failed_turn_command
                            )
                            and isinstance(product_failure.get("message"), str)
                            and bool(product_failure["message"].strip())
                        )
                        if not valid_product_failure:
                            case_failures.append(
                                f"{name} run {run_index + 1}: invalid partial product failure metadata"
                            )
                            continue
                        case_failures.append(
                            f"{name} run {run_index + 1} turn {product_failure['turn']}: "
                            f"product failure: {product_failure['message']}"
                        )
            elif "product_failures" in run:
                product_failures = run["product_failures"]
                if not isinstance(product_failures, list):
                    case_failures.append(
                        f"{name} run {run_index + 1}: invalid complete product failure metadata"
                    )
                else:
                    for product_failure in product_failures:
                        product_turn = (
                            product_failure.get("turn")
                            if isinstance(product_failure, dict)
                            else None
                        )
                        valid_product_failure = (
                            isinstance(product_failure, dict)
                            and set(product_failure) == {"turn", "message"}
                            and isinstance(product_turn, int)
                            and not isinstance(product_turn, bool)
                            and 1 <= product_turn <= len(run["turns"])
                            and isinstance(product_failure.get("message"), str)
                            and bool(product_failure["message"].strip())
                        )
                        if not valid_product_failure:
                            case_failures.append(
                                f"{name} run {run_index + 1}: invalid complete product failure metadata"
                            )
                            continue
                        case_failures.append(
                            f"{name} run {run_index + 1} turn {product_turn}: "
                            f"product failure: {product_failure['message']}"
                        )
            if not run["turns"]:
                if not valid_failure:
                    case_failures.append(f"{name} run {run_index + 1}: missing turn observations")
                continue
            for turn_index, raw_turn in enumerate(run["turns"]):
                try:
                    report, problems = _score_turn(raw_turn, name, run_index, turn_index)
                    turns.append(report)
                    case_failures.extend(problems)
                except ValueError as error:
                    case_failures.append(f"{name} run {run_index + 1} turn {turn_index + 1}: {error}")

        by_turn: dict[int, list[dict[str, Any]]] = {}
        for turn in turns:
            by_turn.setdefault(turn["turn"], []).append(turn)
        gates: list[dict[str, Any]] = []
        for turn_index, turn_reports in sorted(by_turn.items()):
            kinds = {turn["kind"] for turn in turn_reports}
            if len(kinds) != 1:
                case_failures.append(f"{name} turn {turn_index}: kind differs across runs")
                continue
            kind = next(iter(kinds))
            metric = "command_receipt" if kind == "action" else "answer"
            latencies = [
                turn["latency_seconds"][metric]
                for turn in turn_reports
                if isinstance(turn["latency_seconds"][metric], (int, float))
                and not isinstance(turn["latency_seconds"][metric], bool)
            ]
            p50 = nearest_rank(latencies, 0.50) if latencies else None
            p95 = nearest_rank(latencies, 0.95) if latencies else None
            audio_latencies = [turn["latency_seconds"]["first_audio"] for turn in turn_reports]
            gate = {
                "turn": turn_index,
                "kind": kind,
                "run_count": len(latencies),
                f"{metric}_p50_seconds": p50,
                f"{metric}_p95_seconds": p95,
                "first_audio_p50_seconds": nearest_rank(audio_latencies, 0.50),
                "first_audio_p95_seconds": nearest_rank(audio_latencies, 0.95),
            }
            gates.append(gate)
            if kind == "action":
                if p50 is not None and p50 > ACTION_P50_LIMIT_SECONDS:
                    case_failures.append(
                        f"{name} turn {turn_index}: command-receipt p50 {p50:g}s "
                        f"exceeds {ACTION_P50_LIMIT_SECONDS:g}s"
                    )
                if p95 is not None and p95 > ACTION_P95_LIMIT_SECONDS:
                    case_failures.append(
                        f"{name} turn {turn_index}: command-receipt p95 {p95:g}s "
                        f"exceeds {ACTION_P95_LIMIT_SECONDS:g}s"
                    )
            elif p50 is not None and p50 > SEARCH_P50_LIMIT_SECONDS:
                case_failures.append(
                    f"{name} turn {turn_index}: answer p50 {p50:g}s "
                    f"exceeds {SEARCH_P50_LIMIT_SECONDS:g}s"
                )
            if len(latencies) != required_runs:
                case_failures.append(
                    f"{name} turn {turn_index}: expected {required_runs} latency observations, "
                    f"found {len(latencies)}"
                )

        report = {
            "name": name,
            "run_count": len(runs),
            "turns": turns,
            "gates": gates,
            "capture_failures": capture_failures,
            "failures": case_failures,
        }
        reports.append(report)
        failures.extend(case_failures)

    return {"passed": not failures, "failures": failures, "cases": reports}


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: report.py OBSERVATIONS.json", file=sys.stderr)
        return 2
    try:
        observations = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(json.dumps({"passed": False, "failures": [f"input: {error}"], "cases": []}))
        return 2
    report = score_observations(observations)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
