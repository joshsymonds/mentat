"""Scenario corpus and strict offline evaluator for recorded voice turns."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TurnExpectation:
    """Required spoken content for one assistant turn."""

    answer_patterns: tuple[str, ...]
    reject_patterns: tuple[str, ...] = ()
    sms_recipient: str | None = None
    sms_body: str | None = None


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


SCENARIOS = (
    Scenario(
        name="timer-300-seconds",
        caller_lines=("Set a timer for five minutes.",),
        turns=(
            TurnExpectation(
                (
                    r"\b(?:five|5)[ -]minutes?\b",
                    r"\b(?:done|set|started|running|counting down|on the clock)\b",
                )
            ),
        ),
        commands=({"turn": 1, "kind": "timer", "seconds": 300},),
        room_close_after=1,
    ),
    Scenario(
        name="equivalent-alarm",
        caller_lines=("Set an alarm for 7 a.m.",),
        turns=(
            TurnExpectation(
                (
                    r"\b7(?:(?:\s*:\s*00)(?:\s*a\.?\s*m\.?)?|\s*a\.?\s*m\.?|\s*o['’]?clock)(?!\s*p\.?\s*m\.?)\b",
                    r"\balarm\b",
                    r"\b(?:done|set|started)\b",
                ),
                reject_patterns=(r"\b7(?::00)?\s*p\.?\s*m\.?(?![A-Za-z])",),
            ),
        ),
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
            TurnExpectation((r"Alice Keck Park", r"Santa Barbara")),
            TurnExpectation((r"Alice Keck Park", r"(?:navigat|directions|route|taking you)")),
        ),
        commands=(
            {"turn": 1, "kind": "location"},
            {"turn": 2, "kind": "navigate"},
        ),
        room_close_after=2,
        place_query="Alice Keck Park Memorial Garden",
        selected_place_pattern=r"(?i)Alice Keck Park(?: Memorial)? Garden",
    ),
    Scenario(
        name="sms-say-back-yes",
        caller_lines=("Text +1-202-555-0142: I will be there at six.", "Yes."),
        turns=(
            TurnExpectation(
                (r"\b(?:text|send)\b", r"\b(?:should I|would you like|want me to|say yes)\b"),
                sms_recipient="+1-202-555-0142",
                sms_body="I will be there at six.",
            ),
            TurnExpectation((r"\b(?:sent|texted)\b",)),
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
            TurnExpectation(
                (r"\b(?:text|send)\b", r"\b(?:should I|would you like|want me to|say yes)\b"),
                sms_recipient="+1-202-555-0142",
                sms_body="I will be there at six.",
            ),
            TurnExpectation(
                (r"\b(?:text|send)\b", r"\b(?:should I|would you like|want me to|say yes)\b"),
                sms_recipient="+1-202-555-0142",
                sms_body="I will be there at seven.",
            ),
            TurnExpectation((r"\b(?:sent|texted)\b",)),
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
            TurnExpectation(
                (
                    r"\bAlice Keck Park\b",
                    r"\bAlice Keck\b(?: Park)?\s+(?:donated|gave|gifted)\b|"
                    r"\bAlice Keck Park\b.{0,8}\bwho\s+(?:bought|donated|gave|gifted)\b|"
                    r"\b(?:donated|given|gifted)\b.{0,80}\bby Alice Keck\b|"
                    r"\bcity\b.{0,80}\b(?:got|received)\b.{0,80}\banonymous gift\b.{0,100}\bdedicated\b",
                ),
                reject_patterns=(
                    r"\b(?:not sure|don't know|do not know|unclear|can't say|cannot say)\b.{0,100}\b(?:donat|gave|gift)\w*\b",
                    r"\b(?:not|never|did not|didn't|was not|wasn't)\b.{0,100}\b(?:donat|gave|gift)\w*\b",
                    r"\b(?:donated|given|gifted)\b.{0,40}\bby (?:the )?city\b|"
                    r"\b(?:the )?city\b.{0,40}\b(?:donated|gave|gifted)\b",
                    r"\b(?:someone else|somebody else|another person)\b.{0,80}\b(?:donated|gave|gifted)\b|"
                    r"\b(?:donated|gave|gifted)\b.{0,80}\b(?:someone else|somebody else|another person)\b",
                ),
            ),
            TurnExpectation(
                (r"W\.?\s*M\.?\s*Keck", r"(?:father|daughter)"),
                reject_patterns=(
                    r"\b(?:not|never|is not|isn't|was not|wasn't)\b.{0,80}\b(?:father|daughter|son|child|related)\b",
                ),
            ),
            TurnExpectation(
                (
                    r"Superior Oil",
                    r"(?:family|father).{0,80}(?:oil|fortune)|(?:oil|fortune).{0,80}(?:family|father)",
                ),
                reject_patterns=(
                    r"\b(?:can't say|cannot say|don't know|do not know|not sure|may have)\b.{0,100}\b(?:inher|wealth|fortune|Superior Oil)\b",
                    r"\b(?:not|never|did not|didn't|was not|wasn't)\b.{0,100}\b(?:inher|wealth|fortune|Superior Oil)\b",
                ),
            ),
        ),
        commands=(),
        room_close_after=None,
    ),
)


def _require(condition: bool, message: str) -> None:
    """Raise reliably even when Python optimization disables ``assert``."""
    if not condition:
        raise AssertionError(message)


def _spoken_sms_body(text: str, recipient: str, scenario_name: str, turn: int) -> str:
    """Extract the complete say-back between the recipient and confirmation prompt."""
    pattern = re.compile(
        r"\b(?:text|send)\s+(?P<recipient>[+\d().\s-]*\d)\s*[:,.]?\s+"
        r"(?P<body>.+?)(?=\s+(?:should i|would you like|do you want|shall i|want me to)\b)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    spoken_recipient = re.sub(r"\D", "", match.group("recipient")) if match else ""
    expected_recipient = re.sub(r"\D", "", recipient)
    national_number = expected_recipient[1:] if recipient.startswith("+1") and expected_recipient.startswith("1") else ""
    _require(
        match is not None
        and spoken_recipient in (expected_recipient, national_number),
        f"{scenario_name}: turn {turn} has no complete {recipient} message say-back",
    )
    tail = text[match.end():]
    _require(
        re.search(r"\b(?:actually|correction|instead|rather|i meant|make that)\b", tail, re.IGNORECASE) is None,
        f"{scenario_name}: turn {turn} contradicts its SMS say-back after the confirmation prompt",
    )
    return " ".join(match.group("body").split()).strip(' "“”')


def _sms_body_tokens(body: str) -> tuple[str, ...]:
    """Compare spoken renderings while preserving every message word and value."""
    normalized = re.sub(r"\bi'll\b", "i will", body, flags=re.IGNORECASE)
    normalized = re.sub(r"\bsix\b", "6", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bseven\b", "7", normalized, flags=re.IGNORECASE)
    return tuple(re.findall(r"[a-z0-9]+", normalized.lower()))


def _sms_value_matches(field: str, actual: Any, expected: Any) -> bool:
    """Compare SMS payload values without accepting a changed recipient or message."""
    if field == "to":
        return (
            isinstance(actual, str)
            and isinstance(expected, str)
            and re.sub(r"\D", "", actual) == re.sub(r"\D", "", expected)
        )
    if field == "body":
        return (
            isinstance(actual, str)
            and isinstance(expected, str)
            and _sms_body_tokens(actual) == _sms_body_tokens(expected)
        )
    return actual == expected


@dataclass(frozen=True)
class ScenarioFailure:
    """A product-level scenario failure attributed to its assistant turn."""

    turn: int
    message: str


def _scenario_failures(
    scenario: Scenario,
    turns: list[str],
    phone_commands: list[Mapping[str, Any]],
    room_closed_after: int | None,
    *,
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
        fail(
            len(scenario.turns) + 1,
            f"{scenario.name}: expected at most {len(scenario.turns)} recorded turns, got {len(turns)}",
        )
        return failures
    if complete and len(turns) != len(scenario.turns):
        fail(
            max(1, len(turns) + 1),
            f"{scenario.name}: expected {len(scenario.turns)} recorded turns, got {len(turns)}",
        )

    spoken_sms: dict[int, tuple[str, str]] = {}
    for turn_number, (text, expectation) in enumerate(
        zip(turns, scenario.turns, strict=False), 1
    ):
        if not isinstance(text, str):
            fail(turn_number, f"{scenario.name}: turn {turn_number} has no transcript text")
            continue
        for pattern in expectation.answer_patterns:
            require(
                re.search(pattern, text, re.IGNORECASE) is not None,
                turn_number,
                f"{scenario.name}: turn {turn_number} missing answer pattern {pattern!r}; got {text!r}",
            )
        for pattern in expectation.reject_patterns:
            require(
                re.search(pattern, text, re.IGNORECASE) is None,
                turn_number,
                f"{scenario.name}: turn {turn_number} contains an uncertain non-answer matching {pattern!r}",
            )
        if expectation.sms_recipient is not None or expectation.sms_body is not None:
            if not require(
                expectation.sms_recipient is not None and expectation.sms_body is not None,
                turn_number,
                f"{scenario.name}: turn {turn_number} must define both SMS recipient and body",
            ):
                continue
            try:
                body = _spoken_sms_body(text, expectation.sms_recipient, scenario.name, turn_number)
            except AssertionError as error:
                fail(turn_number, str(error))
            else:
                if require(
                    _sms_body_tokens(body) == _sms_body_tokens(expectation.sms_body),
                    turn_number,
                    f"{scenario.name}: turn {turn_number} said back {body!r}, expected {expectation.sms_body!r}",
                ):
                    spoken_sms[turn_number] = (expectation.sms_recipient, expectation.sms_body)

    def check_command(
        index: int,
        expected: Mapping[str, Any],
        actual: Mapping[str, Any],
        command_turn: int,
    ) -> None:
        for field, value in expected.items():
            matches = (
                _sms_value_matches(field, actual.get(field), value)
                if expected.get("kind") == "sms" and field in ("to", "body")
                else field in actual and actual[field] == value
            )
            require(
                field in actual and matches,
                command_turn,
                f"{scenario.name}: command {index} expected {field}={value!r}, "
                f"got {actual.get(field)!r}",
            )
        if actual.get("kind") == "sms":
            command_turn = actual.get("turn")
            if not require(
                isinstance(command_turn, int) and not isinstance(command_turn, bool),
                _command_failure_turn([expected], [actual], len(turns)),
                f"{scenario.name}: SMS command {index} has no integer turn",
            ):
                return
            earlier_confirmations = [turn for turn in spoken_sms if turn < command_turn]
            if not require(
                bool(earlier_confirmations),
                command_turn,
                f"{scenario.name}: SMS command {index} preceded its message say-back",
            ):
                return
            confirmed_recipient, confirmed_body = spoken_sms[max(earlier_confirmations)]
            require(
                _sms_value_matches("to", actual.get("to"), confirmed_recipient)
                and _sms_value_matches("body", actual.get("body"), confirmed_body),
                command_turn,
                f"{scenario.name}: SMS command {index} did not match the latest confirmed message",
            )

        if scenario.place_query is not None and actual.get("kind") == "navigate":
            name = actual.get("name")
            require(
                isinstance(name, str) and re.search(scenario.selected_place_pattern or r"(?!)", name),
                command_turn,
                f"{scenario.name}: navigation selected an unexpected place name {name!r}",
            )
            require(
                bool(turns) and isinstance(name, str) and name in turns[0],
                command_turn,
                f"{scenario.name}: navigation target {name!r} was not among the spoken search results",
            )
            require(
                isinstance(actual.get("address"), str) and bool(actual["address"].strip()),
                command_turn,
                f"{scenario.name}: navigation command has no result address",
            )
            require(
                isinstance(actual.get("place_id"), str) and bool(actual["place_id"].strip()),
                command_turn,
                f"{scenario.name}: navigation command has no returned place id",
            )
            require(
                isinstance(actual.get("lat"), (int, float))
                and not isinstance(actual.get("lat"), bool)
                and math.isfinite(actual["lat"])
                and isinstance(actual.get("lng"), (int, float))
                and not isinstance(actual.get("lng"), bool)
                and math.isfinite(actual["lng"]),
                command_turn,
                f"{scenario.name}: navigation command has invalid coordinates",
            )

    if complete:
        expected_commands = scenario.commands
        if not require(
            len(phone_commands) == len(expected_commands),
            _command_failure_turn(list(expected_commands), phone_commands, len(turns)),
            f"{scenario.name}: expected {len(expected_commands)} phone commands, got {len(phone_commands)}",
        ):
            return failures
        for index, (expected, actual) in enumerate(
            zip(expected_commands, phone_commands, strict=True), 1
        ):
            command_turn = _command_failure_turn([expected], [actual], len(turns))
            if not isinstance(actual, Mapping):
                fail(command_turn, f"{scenario.name}: command {index} is not an object")
                continue
            check_command(index, expected, actual, command_turn)
    else:
        observed_through = len(turns)
        if failed_turn is not None:
            if (
                isinstance(failed_turn, bool)
                or not isinstance(failed_turn, int)
                or failed_turn != len(turns) + 1
                or failed_turn > len(scenario.turns)
            ):
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
            require(
                len(expected_turn_commands) == len(actual_turn_commands),
                command_turn,
                f"{scenario.name}: expected {len(expected_turn_commands)} phone commands, "
                f"got {len(actual_turn_commands)}",
            )
            for index, (expected, actual) in enumerate(
                zip(expected_turn_commands, actual_turn_commands), 1
            ):
                check_command(index, expected, actual, command_turn)

    if complete:
        close_turn = (
            room_closed_after
            if room_closed_after is not None
            else scenario.room_close_after or max(1, len(turns))
        )
        require(
            room_closed_after == scenario.room_close_after,
            close_turn,
            f"{scenario.name}: expected room close after turn {scenario.room_close_after!r}, "
            f"got {room_closed_after!r}",
        )
    elif room_closed_after is not None and room_closed_after != scenario.room_close_after:
        fail(
            room_closed_after,
            f"{scenario.name}: room closed after unexpected turn {room_closed_after}",
        )
    return failures


def _command_failure_turn(
    expected_commands: list[Mapping[str, Any]],
    phone_commands: list[Mapping[str, Any]],
    completed_turns: int,
) -> int:
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
) -> list[ScenarioFailure]:
    """Return all truth failures for a complete captured scenario."""
    if not isinstance(turns, list) or not isinstance(phone_commands, list):
        raise ValueError("scenario turns and phone commands must be lists")
    return _scenario_failures(
        scenario,
        turns,
        phone_commands,
        room_closed_after,
        complete=True,
    )


def evaluate_scenario_prefix(
    scenario: Scenario,
    turns: list[str],
    phone_commands: list[Mapping[str, Any]],
    room_closed_after: int | None,
    *,
    failed_turn: int | None = None,
) -> list[ScenarioFailure]:
    """Return scenario truth failures for a complete, contiguous turn prefix."""
    if not isinstance(turns, list) or not isinstance(phone_commands, list):
        raise ValueError("scenario prefix turns and phone commands must be lists")
    return _scenario_failures(
        scenario,
        turns,
        phone_commands,
        room_closed_after,
        complete=False,
        failed_turn=failed_turn,
    )


def evaluate_scenario(
    scenario: Scenario,
    turns: list[str],
    phone_commands: list[Mapping[str, Any]],
    room_closed_after: int | None,
) -> None:
    """Assert recorded answers, phone commands, and room close match a scenario."""
    failures = evaluate_scenario_failures(
        scenario, turns, phone_commands, room_closed_after
    )
    if failures:
        raise AssertionError(failures[0].message)
