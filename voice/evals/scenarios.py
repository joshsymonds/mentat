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
        caller_lines=("Set a timer for 300 seconds.",),
        turns=(TurnExpectation((r"\b300[- ]second timer\b", r"\bset\b")),),
        commands=({"turn": 1, "kind": "timer", "seconds": 300},),
        room_close_after=1,
    ),
    Scenario(
        name="equivalent-alarm",
        caller_lines=("Set an alarm for 7 a.m.",),
        turns=(TurnExpectation((r"\b7(?::00| o'clock|\s*a\.m\.)\b", r"\balarm\b", r"\bset\b")),),
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
        caller_lines=("Text Alice: I will be there at six.", "Yes."),
        turns=(
            TurnExpectation(
                (r"\bAlice\b", r"I will be there at six\.", r"\b(?:send|text)\b", r"\b(?:should I|want me to|say yes)\b"),
                sms_recipient="Alice",
                sms_body="I will be there at six.",
            ),
            TurnExpectation((r"\bAlice\b", r"\b(?:sent|texted)\b")),
        ),
        commands=({"turn": 2, "kind": "sms", "to": "Alice", "body": "I will be there at six."},),
        room_close_after=2,
    ),
    Scenario(
        name="sms-correction-new-yes",
        caller_lines=(
            "Text Alice: I will be there at six.",
            "Correction: I will be there at seven.",
            "Yes.",
        ),
        turns=(
            TurnExpectation(
                (r"\bAlice\b", r"I will be there at six\.", r"\b(?:send|text)\b", r"\b(?:should I|want me to|say yes)\b"),
                sms_recipient="Alice",
                sms_body="I will be there at six.",
            ),
            TurnExpectation(
                (r"\bAlice\b", r"I will be there at seven\.", r"\b(?:send|text)\b", r"\b(?:should I|want me to|say yes)\b"),
                sms_recipient="Alice",
                sms_body="I will be there at seven.",
            ),
            TurnExpectation((r"\bAlice\b", r"\b(?:sent|texted)\b")),
        ),
        commands=({"turn": 3, "kind": "sms", "to": "Alice", "body": "I will be there at seven."},),
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
                (r"Alice Keck Park", r"Santa Barbara", r"\b(?:donated|gave|gifted)\b"),
                reject_patterns=(
                    r"\b(?:not sure|don't know|do not know|unclear|can't say|cannot say)\b.{0,100}\b(?:donat|gave|gift)\w*\b",
                ),
            ),
            TurnExpectation((r"W\.?\s*M\.?\s*Keck", r"(?:father|daughter)")),
            TurnExpectation(
                (
                    r"Superior Oil",
                    r"(?:family|father).{0,80}(?:oil|fortune)|(?:oil|fortune).{0,80}(?:family|father)",
                ),
                reject_patterns=(
                    r"\b(?:can't say|cannot say|don't know|do not know|not sure|may have)\b.{0,100}\b(?:inher|wealth|fortune|Superior Oil)\b",
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
        r"\b(?:text|send)\s+" + re.escape(recipient) + r"\s*:\s*"
        r"(?P<body>.+?)(?=\s+(?:should i|would you like|do you want|shall i)\b)",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    _require(
        match is not None,
        f"{scenario_name}: turn {turn} has no complete {recipient} message say-back",
    )
    tail = text[match.end():]
    _require(
        re.search(r"\b(?:actually|correction|instead|rather|i meant|make that)\b", tail, re.IGNORECASE) is None,
        f"{scenario_name}: turn {turn} contradicts its SMS say-back after the confirmation prompt",
    )
    return " ".join(match.group("body").split()).strip(' "“”')


def evaluate_scenario(
    scenario: Scenario,
    turns: list[str],
    phone_commands: list[Mapping[str, Any]],
    room_closed_after: int | None,
) -> None:
    """Assert recorded answers, phone commands, and room close match a scenario.

    ``turn`` in each command is one-based and identifies the assistant turn
    that issued it. Every expected answer pattern must occur in its turn;
    unexpected, missing, duplicate, mistimed, and incorrectly parameterized
    commands fail the evaluation.
    """
    _require(
        len(turns) == len(scenario.turns),
        f"{scenario.name}: expected {len(scenario.turns)} recorded turns, got {len(turns)}",
    )
    spoken_sms: dict[int, tuple[str, str]] = {}
    for turn_number, (text, expectation) in enumerate(zip(turns, scenario.turns, strict=True), 1):
        for pattern in expectation.answer_patterns:
            _require(
                re.search(pattern, text, re.IGNORECASE) is not None,
                f"{scenario.name}: turn {turn_number} missing answer pattern {pattern!r}; got {text!r}",
            )
        for pattern in expectation.reject_patterns:
            _require(
                re.search(pattern, text, re.IGNORECASE) is None,
                f"{scenario.name}: turn {turn_number} contains an uncertain non-answer matching {pattern!r}",
            )
        if expectation.sms_recipient is not None or expectation.sms_body is not None:
            _require(
                expectation.sms_recipient is not None and expectation.sms_body is not None,
                f"{scenario.name}: turn {turn_number} must define both SMS recipient and body",
            )
            body = _spoken_sms_body(text, expectation.sms_recipient, scenario.name, turn_number)
            _require(
                body == expectation.sms_body,
                f"{scenario.name}: turn {turn_number} said back {body!r}, expected {expectation.sms_body!r}",
            )
            spoken_sms[turn_number] = (expectation.sms_recipient, body)

    expected_commands = scenario.commands
    _require(
        len(phone_commands) == len(expected_commands),
        f"{scenario.name}: expected {len(expected_commands)} phone commands, got {len(phone_commands)}",
    )
    for index, (expected, actual) in enumerate(zip(expected_commands, phone_commands, strict=True), 1):
        for field, value in expected.items():
            _require(
                field in actual and actual[field] == value,
                f"{scenario.name}: command {index} expected {field}={value!r}, "
                f"got {actual.get(field)!r}",
            )
        if actual.get("kind") == "sms":
            command_turn = actual.get("turn")
            _require(
                isinstance(command_turn, int),
                f"{scenario.name}: SMS command {index} has no integer turn",
            )
            earlier_confirmations = [turn for turn in spoken_sms if turn < command_turn]
            _require(
                bool(earlier_confirmations),
                f"{scenario.name}: SMS command {index} preceded its message say-back",
            )
            confirmed_recipient, confirmed_body = spoken_sms[max(earlier_confirmations)]
            _require(
                actual.get("to") == confirmed_recipient and actual.get("body") == confirmed_body,
                f"{scenario.name}: SMS command {index} did not match the latest confirmed message",
            )

        if scenario.place_query is not None and actual.get("kind") == "navigate":
            name = actual.get("name")
            _require(
                isinstance(name, str) and re.search(scenario.selected_place_pattern or r"(?!)", name),
                f"{scenario.name}: navigation selected an unexpected place name {name!r}",
            )
            _require(
                name in turns[0],
                f"{scenario.name}: navigation target {name!r} was not among the spoken search results",
            )
            _require(
                isinstance(actual.get("address"), str) and bool(actual["address"].strip()),
                f"{scenario.name}: navigation command has no result address",
            )
            _require(
                isinstance(actual.get("place_id"), str) and bool(actual["place_id"].strip()),
                f"{scenario.name}: navigation command has no returned place id",
            )
            _require(
                isinstance(actual.get("lat"), (int, float))
                and not isinstance(actual.get("lat"), bool)
                and math.isfinite(actual["lat"])
                and isinstance(actual.get("lng"), (int, float))
                and not isinstance(actual.get("lng"), bool)
                and math.isfinite(actual["lng"]),
                f"{scenario.name}: navigation command has invalid coordinates",
            )

    _require(
        room_closed_after == scenario.room_close_after,
        f"{scenario.name}: expected room close after turn {scenario.room_close_after!r}, "
        f"got {room_closed_after!r}",
    )
