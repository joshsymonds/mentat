"""Qualify semantic-judge questions against a retained scripted reply corpus."""

from __future__ import annotations

import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping

from .judge import Judge, JudgeUnavailable, JevJudge, Verdict
from .scenarios import SCENARIOS, ScenarioTurn, scenario_turns


FIXTURE_PATH = Path(__file__).with_name("judge_fixtures.json")
_RUN_COUNT = 3
_CORRECT_THRESHOLD = 0.95

JudgeFactory = Callable[[Mapping[str, object], int], Judge]


def load_fixtures(path: Path = FIXTURE_PATH) -> list[dict[str, object]]:
    """Load and validate the corpus, deriving questions only from live builders."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read qualification corpus: {error}") from None
    if not isinstance(payload, list) or not payload:
        raise ValueError("qualification corpus must be a non-empty JSON array")

    runtime_turns: dict[tuple[str, int], ScenarioTurn] = {
        (scenario.name, runtime_turn.turn): runtime_turn
        for scenario in SCENARIOS
        for runtime_turn in scenario_turns(scenario)
    }
    fixtures: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, raw_fixture in enumerate(payload):
        if not isinstance(raw_fixture, dict):
            raise ValueError(f"fixture {index} must be an object")
        fixture = dict(raw_fixture)
        fixture_id = fixture.get("id")
        scenario_name = fixture.get("scenario")
        turn = fixture.get("turn")
        reply = fixture.get("reply")
        expected = fixture.get("expected")
        source = fixture.get("source")
        if not isinstance(fixture_id, str) or not fixture_id.strip() or fixture_id in seen_ids:
            raise ValueError(f"fixture {index} must have a unique non-empty id")
        seen_ids.add(fixture_id)
        if "inputs" in fixture or "context" in fixture:
            raise ValueError(f"fixture {fixture_id} must derive inputs and context from its scenario turn")
        if not isinstance(scenario_name, str) or isinstance(turn, bool) or not isinstance(turn, int):
            raise ValueError(f"fixture {fixture_id} must name a scenario and one-based turn")
        runtime = runtime_turns.get((scenario_name, turn))
        if runtime is None:
            raise ValueError(f"fixture {fixture_id} names an unknown scenario turn")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError(f"fixture {fixture_id} must define a non-empty reply")
        if not isinstance(expected, bool):
            raise ValueError(f"fixture {fixture_id} expected must be a boolean")
        if not isinstance(source, Mapping):
            raise ValueError(f"fixture {fixture_id} must include source provenance")
        _validate_source(fixture_id, expected, source)
        runtime_turn = runtime
        families = tuple(spec.family for spec in runtime_turn.expectation.question_specs)
        fixture["family"] = "+".join(families)
        fixture["inputs"] = {
            key: value
            for spec in runtime_turn.expectation.question_specs
            for key, value in spec.inputs.items()
        }
        fixture["context"] = runtime_turn.context
        fixture["questions"] = dict(runtime_turn.questions)
        fixtures.append(fixture)

    _validate_corpus_coverage(fixtures, runtime_turns)
    return fixtures


def _validate_source(fixture_id: str, expected: bool, source: Mapping[str, object]) -> None:
    kind = source.get("kind")
    if expected:
        if kind == "trace_segment_join":
            if not all(isinstance(source.get(field), str) and source[field] for field in ("artifact", "case")):
                raise ValueError(f"fixture {fixture_id} needs trace artifact and case provenance")
            if not _positive_int(source.get("run")) or not _positive_int(source.get("turn")):
                raise ValueError(f"fixture {fixture_id} needs positive trace run and turn provenance")
            segments = source.get("segments")
            if not isinstance(segments, list) or not segments or any(
                isinstance(segment, bool) or not isinstance(segment, int) or segment < 0
                for segment in segments
            ):
                raise ValueError(f"fixture {fixture_id} needs non-negative trace segment indices")
        elif kind == "transcript":
            if not isinstance(source.get("artifact"), str) or not source["artifact"]:
                raise ValueError(f"fixture {fixture_id} needs transcript provenance")
            if not _positive_int(source.get("line")):
                raise ValueError(f"fixture {fixture_id} needs a positive transcript line number")
        elif kind == "record_segment":
            if not isinstance(source.get("artifact"), str) or not source["artifact"]:
                raise ValueError(f"fixture {fixture_id} needs retained record provenance")
            if not _positive_int(source.get("line")):
                raise ValueError(f"fixture {fixture_id} needs a positive retained record line number")
            block = source.get("block")
            if isinstance(block, bool) or not isinstance(block, int) or block < 0:
                raise ValueError(f"fixture {fixture_id} needs a non-negative retained text block index")
        elif kind == "scripted":
            # A correct shape no retained eval reply exhibits (DL14); scored apart
            # from the retained replies so it never pads their rate.
            if not isinstance(source.get("reason"), str) or not source["reason"].strip():
                raise ValueError(f"scripted fixture {fixture_id} must give its reason")
        else:
            raise ValueError(f"correct fixture {fixture_id} must cite retained scripted source")
    elif kind != "scenario_corruption" or not isinstance(source.get("based_on"), str) or not isinstance(source.get("change"), str):
        raise ValueError(f"wrong fixture {fixture_id} must cite an explicit scenario corruption")


def _is_scripted_correct(fixture: Mapping[str, object]) -> bool:
    source = fixture.get("source")
    return fixture.get("expected") is True and isinstance(source, Mapping) and source.get("kind") == "scripted"


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_corpus_coverage(
    fixtures: list[dict[str, object]],
    runtime_turns: Mapping[tuple[str, int], ScenarioTurn],
) -> None:
    by_turn: dict[tuple[str, int], list[dict[str, object]]] = {}
    for fixture in fixtures:
        key = (str(fixture["scenario"]), int(fixture["turn"]))
        by_turn.setdefault(key, []).append(fixture)
    for scenario_name, turn in runtime_turns:
        fixtures_for_turn = by_turn.get((scenario_name, turn), [])
        if not any(fixture["expected"] is True for fixture in fixtures_for_turn):
            raise ValueError(f"qualification corpus has no correct fixture for {scenario_name} turn {turn}")
        if not any(fixture["expected"] is False for fixture in fixtures_for_turn):
            raise ValueError(f"qualification corpus has no wrong fixture for {scenario_name} turn {turn}")
    if set(by_turn) != set(runtime_turns):
        raise ValueError("qualification corpus must contain only registered scenario turns")


def qualify_fixtures(
    fixtures: list[dict[str, object]],
    *,
    judge_factory: JudgeFactory,
    runs: int = _RUN_COUNT,
) -> dict[str, object]:
    """Run every fixture independently in each concurrent, uncached sweep."""
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError("runs must be a positive integer")
    if not fixtures:
        raise ValueError("at least one fixture is required")

    jobs: dict[Future[dict[str, object]], tuple[int, int]] = {}
    evidence_by_run: dict[int, list[dict[str, object]]] = {run: [] for run in range(1, runs + 1)}
    with ThreadPoolExecutor(max_workers=len(fixtures) * runs) as pool:
        for fixture_index, fixture in enumerate(fixtures):
            for run_number in range(1, runs + 1):
                future = pool.submit(_evaluate_fixture, fixture, run_number, judge_factory)
                jobs[future] = (run_number, fixture_index)
        for future in as_completed(jobs):
            run_number, _fixture_index = jobs[future]
            evidence_by_run[run_number].append(future.result())

    run_results = []
    for run_number in range(1, runs + 1):
        fixture_order = {str(fixture["id"]): index for index, fixture in enumerate(fixtures)}
        evidence = sorted(evidence_by_run[run_number], key=lambda item: fixture_order[str(item["id"])])
        counts, family_counts = _score_run(fixtures, evidence)
        correct_rate = counts["correct"] / counts["correct_total"]
        scripted_rate = (
            counts["scripted_correct"] / counts["scripted_total"] if counts["scripted_total"] else 1.0
        )
        passed = (
            correct_rate >= _CORRECT_THRESHOLD
            and scripted_rate >= _CORRECT_THRESHOLD
            and counts["wrong"] == counts["wrong_total"]
            and counts["unavailable"] == 0
        )
        run_results.append({
            "run": run_number,
            "status": "pass" if passed else "fail",
            "correct_rate": correct_rate,
            "scripted_correct_rate": scripted_rate,
            "counts": counts,
            "family_counts": family_counts,
            "fixtures": evidence,
        })
    return {
        "run_count": runs,
        "thresholds": {
            "correct_fixtures": _CORRECT_THRESHOLD,
            "wrong_fixtures": "all must be judged no",
            "unavailable_per_run": 0,
        },
        "status": "pass" if all(run["status"] == "pass" for run in run_results) else "fail",
        "runs": run_results,
    }


def _evaluate_fixture(
    fixture: dict[str, object],
    run_number: int,
    judge_factory: JudgeFactory,
) -> dict[str, object]:
    fixture_id = str(fixture["id"])
    questions = fixture["questions"]
    if not isinstance(questions, Mapping):
        raise ValueError(f"fixture {fixture_id} has no built question mapping")
    context = str(fixture.get("context", ""))
    try:
        judge = judge_factory(fixture, run_number)
        answers = judge.evaluate(str(fixture["reply"]), questions, context=context)
        if not isinstance(answers, Mapping) or set(answers) != set(questions):
            raise JudgeUnavailable("judge returned incomplete question evidence")
        verdicts: dict[str, dict[str, object]] = {}
        for question_id, question in questions.items():
            verdict = answers[question_id]
            if not isinstance(verdict, Verdict) or verdict.question != question:
                raise JudgeUnavailable("judge returned malformed question evidence")
            probability = verdict.probability
            if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise JudgeUnavailable("judge returned an invalid probability")
            verdicts[question_id] = {
                "question": question,
                "verdict": verdict.verdict,
                "probability": float(probability),
            }
        observation = all(item["verdict"] is True for item in verdicts.values())
        return {
            "id": fixture_id,
            "family": fixture["family"],
            "expected": fixture["expected"],
            "status": "pass" if observation is fixture["expected"] else "fail",
            "observation": "yes" if observation else "no",
            "questions": dict(questions),
            "verdicts": verdicts,
            "probabilities": {key: item["probability"] for key, item in verdicts.items()},
        }
    except JudgeUnavailable as error:
        return {
            "id": fixture_id,
            "family": fixture["family"],
            "expected": fixture["expected"],
            "status": "unavailable",
            "observation": "unavailable",
            "questions": dict(questions),
            "verdicts": {},
            "probabilities": {},
            "error": str(error),
        }


def _score_run(
    fixtures: list[dict[str, object]],
    evidence: list[dict[str, object]],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    by_id = {str(item["id"]): item for item in evidence}
    counts = {
        "correct": 0,
        "correct_total": sum(
            1 for fixture in fixtures if fixture["expected"] is True and not _is_scripted_correct(fixture)
        ),
        "scripted_correct": 0,
        "scripted_total": sum(1 for fixture in fixtures if _is_scripted_correct(fixture)),
        "wrong": 0,
        "wrong_total": sum(1 for fixture in fixtures if fixture["expected"] is False),
        "passed": 0,
        "failed": 0,
        "unavailable": 0,
    }
    family_counts: dict[str, dict[str, int]] = {}
    for fixture in fixtures:
        result = by_id[str(fixture["id"])]
        family = str(fixture["family"])
        family_count = family_counts.setdefault(family, {
            "correct": 0, "correct_total": 0, "wrong": 0, "wrong_total": 0,
            "passed": 0, "failed": 0, "unavailable": 0,
        })
        expected = fixture["expected"] is True
        family_count["correct_total" if expected else "wrong_total"] += 1
        if result["status"] == "unavailable":
            counts["unavailable"] += 1
            family_count["unavailable"] += 1
            continue
        if result["status"] == "pass":
            counts["passed"] += 1
            family_count["passed"] += 1
            if _is_scripted_correct(fixture):
                counts["scripted_correct"] += 1
            else:
                counts["correct" if expected else "wrong"] += 1
            family_count["correct" if expected else "wrong"] += 1
        else:
            counts["failed"] += 1
            family_count["failed"] += 1
    return counts, family_counts


def qualification_exit_code(result: Mapping[str, object]) -> int:
    """Return failure unless every required run satisfies all qualification bars."""
    return 0 if result.get("status") == "pass" else 1


def _live_judge_factory(_fixture: Mapping[str, object], _run_number: int) -> Judge:
    """Create one fresh cache scope for each fixture in each qualification run."""
    return JevJudge()


def main() -> int:
    fixtures = load_fixtures()
    result = qualify_fixtures(fixtures, judge_factory=_live_judge_factory, runs=_RUN_COUNT)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return qualification_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
