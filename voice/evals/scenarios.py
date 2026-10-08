"""Scenario corpus and strict offline evaluator for recorded voice turns."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from evals.judge import Judge


@dataclass(frozen=True)
class QuestionSpec:
    """One qualified question family and its scenario-specific semantic facts."""

    family: str
    inputs: Mapping[str, Any]


@dataclass(frozen=True)
class TurnExpectation:
    """Qualified judge questions required for one spoken assistant turn."""

    question_specs: tuple[QuestionSpec, ...]


@dataclass(frozen=True)
class Scenario:
    """One scripted voice interaction and its observable expectations."""

    name: str
    caller_lines: tuple[str, ...]
    turns: tuple[TurnExpectation, ...]
    commands: tuple[Mapping[str, Any], ...]
    room_close_after: int | None
    place_query: str | None = None
    selected_place_pattern: str | None = None
    caller_languages: tuple[str, ...] = ()
    reply_languages: tuple[str, ...] = ()
    voice_mode_expectations: tuple[str, ...] = ()
    preconnect_first_line: bool = False
    barge_in_after: tuple[float | None, ...] = ()
    exact_caller_stt: bool = False


@dataclass(frozen=True)
class ScenarioTurn:
    """Runtime-derived questions and context for one one-based scenario turn."""

    turn: int
    expectation: TurnExpectation
    questions: Mapping[str, str]
    context: str


def _question(family: str, **inputs: Any) -> QuestionSpec:
    return QuestionSpec(family, inputs)


SCENARIOS = (
    Scenario(
        name="timer-300-seconds",
        caller_lines=("Set a timer for five minutes.",),
        turns=(TurnExpectation((
            _question("action_ack", action="timer", expected="five minutes"),
            _question("timer_duration", seconds=300),
        )),),
        commands=({"turn": 1, "kind": "timer", "seconds": 300},),
        room_close_after=1,
    ),
    Scenario(
        name="equivalent-alarm",
        caller_lines=("Set an alarm for 7 a.m.",),
        turns=(TurnExpectation((
            _question("action_ack", action="alarm", expected="7 a.m."),
            _question("alarm_time", hour=7, minute=0),
        )),),
        commands=({"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},),
        room_close_after=1,
    ),
    Scenario(
        name="place-search-navigation",
        caller_lines=(
            "Find Alice Keck Park Memorial Garden in Santa Barbara.",
            "Navigate to Alice Keck Park Memorial Garden.",
        ),
        turns=(
            TurnExpectation((_question(
                "place_lookup", place_name="Alice Keck Park Memorial Garden",
                locality="Santa Barbara",
            ),)),
            TurnExpectation((_question(
                "action_ack", action="navigate", expected="Alice Keck Park Memorial Garden",
            ),)),
        ),
        commands=(
            {"turn": 1, "kind": "location"},
            {"turn": 2, "kind": "navigate"},
        ),
        room_close_after=2,
        place_query="Alice Keck Park Memorial Garden",
        selected_place_pattern=r"(?i)^Alice Keck Park Memorial Gardens?$",
    ),
    Scenario(
        name="sms-say-back-yes",
        caller_lines=("Text +1-202-555-0142: I will be there at six.", "Yes."),
        turns=(
            TurnExpectation((_question(
                "sms_confirmation", recipient="+1-202-555-0142",
                body="I will be there at six.",
            ),)),
            TurnExpectation((_question("sms_send_ack", recipient="+1-202-555-0142"),)),
        ),
        commands=({"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."},),
        room_close_after=2,
    ),
    Scenario(
        name="sms-correction-new-yes",
        caller_lines=(
            "Text +1-202-555-0142: I will be there at six.",
            "Correction: I will be there at seven.",
            "Yes.",
        ),
        turns=(
            TurnExpectation((_question(
                "sms_confirmation", recipient="+1-202-555-0142",
                body="I will be there at six.",
            ),)),
            TurnExpectation((_question(
                "sms_confirmation", recipient="+1-202-555-0142",
                body="I will be there at seven.",
            ),)),
            TurnExpectation((_question("sms_send_ack", recipient="+1-202-555-0142"),)),
        ),
        commands=({"turn": 3, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at seven."},),
        room_close_after=3,
    ),
    Scenario(
        name="alice-keck-context-chain",
        caller_lines=(
            "Why is this garden named for Alice Keck?",
            "Okay, who was she?",
            "Okay, what was the source of her wealth?",
        ),
        turns=(
            TurnExpectation((_question(
                "alice_donor", donor="Alice Keck", action="bought and donated Alice Keck Park Memorial Garden",
            ),)),
            TurnExpectation((_question(
                "alice_father", person="Alice Keck", father="William M. Keck",
            ),)),
            TurnExpectation((_question(
                "alice_wealth", person="Alice Keck", source="the Superior Oil family fortune",
            ),)),
        ),
        commands=(),
        room_close_after=None,
    ),
    Scenario(
        name="spanish-language-switch",
        caller_lines=(
            "Can we continue in Spanish, please?",
            "¿Cuál es la capital de Francia?",
            "Let's switch back to English, please.",
            "Can we speak Spanish again, please?",
            "¿De qué color es el cielo en un día despejado?",
        ),
        turns=tuple(
            TurnExpectation((_question("spanish_switch", turn=turn),))
            for turn in range(1, 6)
        ),
        commands=(),
        room_close_after=5,
        caller_languages=("en", "es", "en", "en", "es"),
        reply_languages=("es", "es", "en", "es", "es"),
        voice_mode_expectations=("es", "en", "es"),
    ),
    Scenario(
        name="spanish-interpreter",
        caller_lines=(
            "Please interpret for the Spanish-speaking gardener and tell them I'm ready.",
            "La tierra está demasiado seca para plantar tomates.",
            "Water the seedlings every morning before the sun gets strong.",
            "Ahora programa un temporizador de cinco minutos para regar las plantas.",
            "Mientras hablábamos del trabajo, el jardinero dijo: «deja de traducir».",
            "Please stop interpreting and speak to me in English.",
        ),
        turns=tuple(
            TurnExpectation((_question("interpreter_turn", turn=turn),))
            for turn in range(1, 7)
        ),
        commands=(),
        room_close_after=None,
        caller_languages=("en", "es", "en", "es", "es", "en"),
        reply_languages=("es", "en", "es", "en", "en", "en"),
        voice_mode_expectations=("es", "en"),
    ),
    Scenario(
        name="phone-first-line",
        caller_lines=("Hey, what's the capital of Australia?",),
        turns=(TurnExpectation((_question(
            "claims",
            required=["Canberra is the capital of Australia"],
            rejected=["Sydney is the capital of Australia"],
        ),)),),
        commands=(),
        room_close_after=None,
        preconnect_first_line=True,
        exact_caller_stt=True,
    ),
    Scenario(
        name="barge-in-long-reply",
        caller_lines=(
            "Tell me a long story about a lighthouse keeper and her cat.",
            "Stop. What's the capital of Japan?",
        ),
        turns=(
            TurnExpectation((_question("story_start"),)),
            TurnExpectation((_question(
                "barge_in_answer",
                answer="say Tokyo",
                abandoned="the story about a lighthouse keeper and her cat",
            ),)),
        ),
        commands=(),
        room_close_after=None,
        barge_in_after=(None, 8.0),
        exact_caller_stt=True,
    ),
)


def build_turn_questions(expectation: TurnExpectation) -> dict[str, str]:
    """Build the full question map for one declared turn expectation."""
    from evals.questions import build_questions

    if not isinstance(expectation, TurnExpectation) or not expectation.question_specs:
        raise ValueError("turn expectation must declare at least one question family")
    questions: dict[str, str] = {}
    for spec in expectation.question_specs:
        built = build_questions(spec.family, spec.inputs)
        duplicate_ids = questions.keys() & built.keys()
        if duplicate_ids:
            raise ValueError(f"duplicate judge question ids: {sorted(duplicate_ids)}")
        questions.update(built)
    return questions


def _derive_turn(
    scenario: Scenario,
    turn_number: int,
    expectation: TurnExpectation,
    accepted_sms: Mapping[int, tuple[str, str]],
) -> tuple[TurnExpectation, str]:
    sms = _expectation_sms(expectation)
    context = ""
    if sms is not None and scenario.name == "sms-correction-new-yes":
        prior = [accepted_sms[index] for index in sorted(accepted_sms) if index < turn_number]
        if prior and prior[-1][0] == sms[0]:
            context = f"Previously verified SMS recipient: {sms[0]}."
            expectation = TurnExpectation(tuple(
                QuestionSpec(spec.family, {**spec.inputs, "verified_recipient": sms[0]})
                if spec.family in {"sms_say_back", "sms_confirmation"}
                and "verified_recipient" not in spec.inputs
                else spec
                for spec in expectation.question_specs
            ))
    return expectation, context


def scenario_turns(scenario: Scenario) -> tuple[ScenarioTurn, ...]:
    """Return the runtime question/context derivation for every scenario turn."""
    accepted_sms: dict[int, tuple[str, str]] = {}
    result = []
    for turn_number, expectation in enumerate(scenario.turns, 1):
        derived, context = _derive_turn(scenario, turn_number, expectation, accepted_sms)
        result.append(ScenarioTurn(
            turn=turn_number,
            expectation=derived,
            questions=build_turn_questions(derived),
            context=context,
        ))
        sms = _expectation_sms(expectation)
        if sms is not None:
            accepted_sms[turn_number] = sms
    return tuple(result)


def judge_turn(
    expectation: TurnExpectation,
    reply: str,
    judge: Judge,
    *,
    context: str = "",
) -> dict[str, Any]:
    """Judge a complete or accumulated reply and return frozen report evidence."""
    from evals.judge import JudgeUnavailable, Verdict

    if context and any(
        spec.family in {"sms_say_back", "sms_confirmation"}
        and "verified_recipient" not in spec.inputs
        for spec in expectation.question_specs
    ):
        expectation = TurnExpectation(tuple(
            QuestionSpec(spec.family, {**spec.inputs, "verified_recipient": spec.inputs["recipient"]})
            if spec.family in {"sms_say_back", "sms_confirmation"}
            and "verified_recipient" not in spec.inputs
            else spec
            for spec in expectation.question_specs
        ))
    questions = build_turn_questions(expectation)
    records = [
        {"id": question_id, "question": question, "verdict": None, "probability": None}
        for question_id, question in questions.items()
    ]
    try:
        verdicts = judge.evaluate(reply, questions, context=context)
        if not isinstance(verdicts, dict) or set(verdicts) != set(questions):
            raise ValueError("judge returned an incomplete result")
        for record in records:
            verdict = verdicts[record["id"]]
            if (
                not isinstance(verdict, Verdict)
                or verdict.question != record["question"]
                or not math.isfinite(verdict.probability)
                or not 0.0 <= verdict.probability <= 1.0
            ):
                raise ValueError("judge returned a malformed verdict")
            record["verdict"] = verdict.verdict
            record["probability"] = verdict.probability
    except JudgeUnavailable as error:
        return {"context": context, "questions": records, "unavailable": str(error)}
    except (TypeError, ValueError, AttributeError, OverflowError):
        return {
            "context": context,
            "questions": records,
            "unavailable": "judge returned a malformed response",
        }
    return {"context": context, "questions": records, "unavailable": None}


@dataclass(frozen=True)
class ScenarioFailure:
    """A product-level scenario failure attributed to its assistant turn."""

    turn: int
    message: str


def _sms_value_matches(field: str, actual: Any, expected: Any) -> bool:
    """Compare SMS payload values without accepting a changed recipient or message."""
    if field == "to":
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        actual_digits = re.sub(r"\D", "", actual)
        expected_digits = re.sub(r"\D", "", expected)
        national_number = (
            expected_digits[1:]
            if expected.startswith("+1") and expected_digits.startswith("1")
            else ""
        )
        return actual_digits == expected_digits or (
            bool(national_number)
            and actual_digits == national_number
            and not actual.lstrip().startswith("+")
        )
    if field == "body":
        if not isinstance(actual, str) or not isinstance(expected, str):
            return False
        return _sms_body_tokens(actual) == _sms_body_tokens(expected)
    return actual == expected


def _sms_body_tokens(body: str) -> tuple[str, ...]:
    """Normalize only accepted spoken forms without changing message meaning."""
    normalized = re.sub(r"\bi'll\b", "i will", body, flags=re.IGNORECASE)
    normalized = re.sub(r"\bsix\b", "6", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bseven\b", "7", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(\d{1,2}):00\b", r"\1", normalized)
    normalized = re.sub(r"\b(\d{1,2})\s+o['’]?\s?clock\b", r"\1", normalized, flags=re.IGNORECASE)
    return tuple(re.findall(r"[a-z0-9]+", normalized.lower()))


_NUMBER_WORD_VALUES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90,
}


def _spoken_number(value: str) -> int | None:
    """Normalize spoken cardinal numbers in exact caller transcription evidence."""
    if value.isdigit():
        try:
            return int(value)
        except ValueError:
            return None
    parts = re.split(r"[- ]", value.lower())
    numbers = [_NUMBER_WORD_VALUES.get(part) for part in parts]
    if any(number is None for number in numbers):
        return None
    return sum(number for number in numbers if number is not None)


def _expectation_sms(expectation: TurnExpectation) -> tuple[str, str] | None:
    for spec in expectation.question_specs:
        if spec.family in {"sms_say_back", "sms_confirmation"}:
            recipient, body = spec.inputs.get("recipient"), spec.inputs.get("body")
            if isinstance(recipient, str) and isinstance(body, str):
                return recipient, body
    return None


def _scenario_failures(
    scenario: Scenario,
    turns: list[str],
    phone_commands: list[Mapping[str, Any]],
    room_closed_after: int | None,
    *,
    judge: Judge,
    judge_evidence: list[dict[str, Any]] | None,
    complete: bool,
    failed_turn: int | None = None,
) -> list[ScenarioFailure]:
    failures: list[ScenarioFailure] = []

    def fail(turn: int, message: str) -> None:
        failures.append(ScenarioFailure(turn=turn, message=message))

    def require(condition: bool, turn: int, message: str) -> bool:
        if not condition:
            fail(turn, message)
        return condition

    if len(turns) > len(scenario.turns):
        fail(len(scenario.turns) + 1, f"{scenario.name}: expected at most {len(scenario.turns)} recorded turns, got {len(turns)}")
        return failures
    if complete and len(turns) != len(scenario.turns):
        fail(max(1, len(turns) + 1), f"{scenario.name}: expected {len(scenario.turns)} recorded turns, got {len(turns)}")

    accepted_sms: dict[int, tuple[str, str]] = {}
    for turn_number, (text, expectation) in enumerate(zip(turns, scenario.turns), 1):
        if not isinstance(text, str):
            fail(turn_number, f"{scenario.name}: turn {turn_number} has no transcript text")
            continue
        sms = _expectation_sms(expectation)
        expectation, context = _derive_turn(scenario, turn_number, expectation, accepted_sms)
        evidence = judge_turn(expectation, text, judge, context=context)
        if judge_evidence is not None:
            judge_evidence.append(evidence)
        if evidence["unavailable"] is not None:
            fail(turn_number, f"{scenario.name}: turn {turn_number} judge unavailable: {evidence['unavailable']}")
        elif any(question["verdict"] is not True for question in evidence["questions"]):
            failed_questions = [question["id"] for question in evidence["questions"] if question["verdict"] is not True]
            fail(turn_number, f"{scenario.name}: turn {turn_number} did not satisfy judge questions {failed_questions!r}")
        elif sms is not None:
            accepted_sms[turn_number] = sms

    def check_command(index: int, expected: Mapping[str, Any], actual: Mapping[str, Any], command_turn: int) -> None:
        for field, value in expected.items():
            matches = (
                _sms_value_matches(field, actual.get(field), value)
                if expected.get("kind") == "sms" and field in ("to", "body")
                else field in actual and actual[field] == value
            )
            require(field in actual and matches, command_turn,
                    f"{scenario.name}: command {index} expected {field}={value!r}, got {actual.get(field)!r}")
        if actual.get("kind") == "sms":
            command_turn = actual.get("turn")
            if not require(
                isinstance(command_turn, int) and not isinstance(command_turn, bool),
                _command_failure_turn([expected], [actual], len(turns)),
                f"{scenario.name}: SMS command {index} has no integer turn",
            ):
                return
            earlier = [turn for turn in accepted_sms if turn < command_turn]
            if not require(bool(earlier), command_turn,
                           f"{scenario.name}: SMS command {index} preceded its message say-back"):
                return
            recipient, body = accepted_sms[max(earlier)]
            require(
                _sms_value_matches("to", actual.get("to"), recipient)
                and _sms_value_matches("body", actual.get("body"), body),
                command_turn,
                f"{scenario.name}: SMS command {index} did not match the latest confirmed message",
            )
        if scenario.place_query is not None and actual.get("kind") == "navigate":
            name = actual.get("name")
            require(
                isinstance(name, str) and re.search(scenario.selected_place_pattern or r"(?!)", name),
                command_turn, f"{scenario.name}: navigation selected an unexpected place name {name!r}",
            )
            require(
                isinstance(actual.get("address"), str) and bool(actual["address"].strip()),
                command_turn, f"{scenario.name}: navigation command has no result address",
            )
            require(
                isinstance(actual.get("place_id"), str) and bool(actual["place_id"].strip()),
                command_turn, f"{scenario.name}: navigation command has no returned place id",
            )
            require(
                isinstance(actual.get("lat"), (int, float)) and not isinstance(actual.get("lat"), bool)
                and math.isfinite(actual["lat"])
                and isinstance(actual.get("lng"), (int, float)) and not isinstance(actual.get("lng"), bool)
                and math.isfinite(actual["lng"]),
                command_turn, f"{scenario.name}: navigation command has invalid coordinates",
            )

    if complete:
        expected_commands = scenario.commands
        if not require(len(phone_commands) == len(expected_commands),
                       _command_failure_turn(list(expected_commands), phone_commands, len(turns)),
                       f"{scenario.name}: expected {len(expected_commands)} phone commands, got {len(phone_commands)}"):
            return failures
        for index, (expected, actual) in enumerate(zip(expected_commands, phone_commands, strict=True), 1):
            command_turn = _command_failure_turn([expected], [actual], len(turns))
            if not isinstance(actual, Mapping):
                fail(command_turn, f"{scenario.name}: command {index} is not an object")
                continue
            check_command(index, expected, actual, command_turn)
    else:
        observed_through = len(turns)
        if failed_turn is not None:
            if isinstance(failed_turn, bool) or not isinstance(failed_turn, int) or failed_turn != len(turns) + 1 or failed_turn > len(scenario.turns):
                fail(max(1, len(turns) + 1), f"{scenario.name}: invalid partial failure turn")
                return failures
            observed_through = failed_turn
        expected_by_turn: dict[int, list[Mapping[str, Any]]] = {}
        actual_by_turn: dict[int, list[Mapping[str, Any]]] = {}
        for expected in scenario.commands:
            command_turn = expected.get("turn")
            if isinstance(command_turn, int) and command_turn <= observed_through:
                expected_by_turn.setdefault(command_turn, []).append(expected)
        for command in phone_commands:
            if not isinstance(command, Mapping):
                fail(max(1, observed_through), f"{scenario.name}: fake phone command is not an object")
                continue
            command_turn = command.get("turn")
            if isinstance(command_turn, bool) or not isinstance(command_turn, int) or command_turn < 1:
                fail(max(1, observed_through), f"{scenario.name}: phone command has no valid turn")
                continue
            if command_turn > observed_through:
                fail(command_turn, f"{scenario.name}: phone command occurred after the failed turn")
                continue
            actual_by_turn.setdefault(command_turn, []).append(command)
        for command_turn in range(1, observed_through + 1):
            expected_turn_commands = expected_by_turn.get(command_turn, [])
            actual_turn_commands = actual_by_turn.get(command_turn, [])
            if command_turn == failed_turn and not actual_turn_commands:
                continue
            require(len(expected_turn_commands) == len(actual_turn_commands), command_turn,
                    f"{scenario.name}: expected {len(expected_turn_commands)} phone commands, got {len(actual_turn_commands)}")
            for index, (expected, actual) in enumerate(zip(expected_turn_commands, actual_turn_commands), 1):
                check_command(index, expected, actual, command_turn)

    if complete:
        close_turn = room_closed_after if room_closed_after is not None else scenario.room_close_after or max(1, len(turns))
        require(room_closed_after == scenario.room_close_after, close_turn,
                f"{scenario.name}: expected room close after turn {scenario.room_close_after!r}, got {room_closed_after!r}")
    elif room_closed_after is not None and room_closed_after != scenario.room_close_after:
        fail(room_closed_after, f"{scenario.name}: room closed after unexpected turn {room_closed_after}")
    return failures


def _command_failure_turn(expected_commands: list[Mapping[str, Any]], phone_commands: list[Mapping[str, Any]], completed_turns: int) -> int:
    for command in phone_commands:
        turn = command.get("turn") if isinstance(command, Mapping) else None
        if isinstance(turn, int) and not isinstance(turn, bool) and turn > 0:
            return turn
    for command in expected_commands:
        turn = command.get("turn")
        if isinstance(turn, int) and not isinstance(turn, bool) and turn > 0:
            return turn
    return max(1, completed_turns)


def evaluate_scenario_failures(
    scenario: Scenario,
    turns: list[str],
    phone_commands: list[Mapping[str, Any]],
    room_closed_after: int | None,
    *,
    judge: Judge,
    judge_evidence: list[dict[str, Any]] | None = None,
) -> list[ScenarioFailure]:
    """Return all truth failures for a complete captured scenario."""
    if not isinstance(turns, list) or not isinstance(phone_commands, list):
        raise ValueError("scenario turns and phone commands must be lists")
    if judge_evidence is not None:
        judge_evidence.clear()
    return _scenario_failures(scenario, turns, phone_commands, room_closed_after,
                              judge=judge, judge_evidence=judge_evidence, complete=True)


def evaluate_scenario_prefix(
    scenario: Scenario,
    turns: list[str],
    phone_commands: list[Mapping[str, Any]],
    room_closed_after: int | None,
    *,
    judge: Judge,
    failed_turn: int | None = None,
    judge_evidence: list[dict[str, Any]] | None = None,
) -> list[ScenarioFailure]:
    """Return scenario truth failures for a complete, contiguous turn prefix."""
    if not isinstance(turns, list) or not isinstance(phone_commands, list):
        raise ValueError("scenario prefix turns and phone commands must be lists")
    if judge_evidence is not None:
        judge_evidence.clear()
    return _scenario_failures(scenario, turns, phone_commands, room_closed_after,
                              judge=judge, judge_evidence=judge_evidence,
                              complete=False, failed_turn=failed_turn)


def evaluate_scenario(
    scenario: Scenario,
    turns: list[str],
    phone_commands: list[Mapping[str, Any]],
    room_closed_after: int | None,
    *,
    judge: Judge,
) -> None:
    """Assert judge decisions, exact phone commands, and room close match a scenario."""
    failures = evaluate_scenario_failures(scenario, turns, phone_commands, room_closed_after, judge=judge)
    if failures:
        raise AssertionError(failures[0].message)
