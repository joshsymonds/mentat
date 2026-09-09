"""What the eval decides, once a response exists: schema, verdict, report.

The front's whole job is one judgment made hundreds of times a day — answer
this myself, or send it to Mentat — and that judgment is a prompt, not code, so
the only way to know whether an edit to persona.md helped is to measure it. The
scenarios are the measurement's fixed half; this module is the ruler.

Deliberately stdlib-only and free of livekit, so the schema and the scoring run
in `just test-voice` offline next to voice/request.py. The online half — the
persona, the real ask_mentat schema, the calls to Luna — is run.py, which
imports this and never the other way round.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The consult tool and the phone tools the front is given. Calls to any other
#: name are hallucinated tools and score as invalid.
TOOL_NAME = "ask_mentat"
PHONE_TOOL_NAMES = (
    "find_places",
    "navigate_to",
    "dial",
    "send_text",
    "set_alarm",
    "set_timer",
    "open_link",
)

#: The decisions the eval measures, plus the verdict for a response that is
#: neither: an empty turn, an unknown or mixed tool call, or a failed request.
CONSULT = "consult"
DIRECT = "direct"
INVALID = "invalid"

#: Scenarios in this category are run and reported but not scored: their
#: correct behaviour depends on session state the runner does not reproduce
#: (a consult already in flight), so a pass or fail here would be noise.
SKIP_CATEGORY = "skip"

#: Argument names accepted by `expect_args`, keyed by the expected tool.
TOOL_ARGUMENT_KEYS = {
    TOOL_NAME: frozenset(("question", "effort", "model")),
    "find_places": frozenset(("query", "locality")),
    "navigate_to": frozenset(("choice",)),
    "dial": frozenset(("number",)),
    "send_text": frozenset(("number", "body")),
    "set_alarm": frozenset(("hour", "minute", "label")),
    "set_timer": frozenset(("minutes", "seconds", "label")),
    "open_link": frozenset(("url",)),
}

#: How many turns of prior conversation a scenario may carry. The same window
#: request.CONSULT_WINDOW_TURNS gives a real consult — a scenario with more
#: history than the front ever sees would measure something else.
HISTORY_TURNS = 2

_ROLES = ("user", "assistant")
_REQUIRED_FIELDS = ("id", "category", "utterance", "expect", "note")
_OPTIONAL_FIELDS = ("history", "expect_args")


class ScenarioError(ValueError):
    """A scenario file that cannot be trusted to measure anything."""


@dataclass(frozen=True)
class Scenario:
    """One labelled utterance and the decision the front should make on it."""

    id: str
    category: str
    utterance: str
    expect: str
    note: str
    history: tuple[tuple[str, str], ...] = ()
    expect_args: Mapping[str, str] = field(default_factory=dict)

    @property
    def scored(self) -> bool:
        return self.category != SKIP_CATEGORY


@dataclass(frozen=True)
class ToolCall:
    """One tool call the model asked for, with its arguments already decoded."""

    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True)
class Response:
    """What one turn produced: text, tool calls, or the error that stopped it."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class Result:
    """One scenario's verdict."""

    scenario: Scenario
    response: Response
    got: str
    hit: bool | None

    @property
    def scored(self) -> bool:
        return self.hit is not None


def parse_scenario(raw: Any, where: str) -> Scenario:
    """One scenario line, validated into a Scenario or rejected by name.

    Strict about unknown keys and about every field's shape: the file is
    hand-written and grows, and a misspelled key that silently drops its
    meaning — an `expected` that never scores, an `expect_arg` that never
    checks — would leave the eval reporting on a scenario it is not running.
    `where` names the line, because "some scenario is wrong" is not a report
    anyone can act on.
    """
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}: scenario is {type(raw).__name__}, not an object")

    known = (*_REQUIRED_FIELDS, *_OPTIONAL_FIELDS)
    for key in raw:
        if key not in known:
            raise ScenarioError(f"{where}: unknown field {key!r}")
    for key in _REQUIRED_FIELDS:
        if key not in raw:
            raise ScenarioError(f"{where}: missing field {key!r}")

    text_fields = {
        key: _text(raw[key], f"{where}: {key}")
        for key in ("id", "category", "utterance", "note")
    }

    expect = raw["expect"]
    if expect not in (CONSULT, DIRECT, *PHONE_TOOL_NAMES):
        allowed = (CONSULT, DIRECT, *PHONE_TOOL_NAMES)
        raise ScenarioError(f"{where}: expect is {expect!r}, not one of {allowed}")

    return Scenario(
        **text_fields,
        expect=expect,
        history=_history(raw.get("history", []), where),
        expect_args=_expect_args(raw.get("expect_args", {}), expect, where),
    )


def load_scenarios(path: Path) -> list[Scenario]:
    """Every scenario in a JSON-lines file, in file order.

    Blank lines are separators, so the set can be grouped by category as it
    grows. Duplicate ids are rejected: the id is how a miss is reported and how
    a fix is checked, so two scenarios sharing one is a file that cannot be
    read.
    """
    scenarios: list[Scenario] = []
    seen: dict[str, str] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        where = f"line {number}"
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as err:
            raise ScenarioError(f"{where}: {err}") from err
        scenario = parse_scenario(raw, where)
        if scenario.id in seen:
            raise ScenarioError(
                f"{where}: duplicate id {scenario.id!r}, already used on {seen[scenario.id]}"
            )
        seen[scenario.id] = where
        scenarios.append(scenario)
    if not scenarios:
        raise ScenarioError(f"{path}: no scenarios")
    return scenarios


def classify(response: Response) -> str:
    """Classify text or exactly one known tool call.

    A holding line spoken alongside one tool call does not change the tool's
    classification. Multiple calls are invalid even when one is the expected
    tool: the response took more than one action and cannot satisfy a single
    scenario label.
    """
    if response.error:
        return INVALID
    if response.tool_calls:
        if len(response.tool_calls) != 1:
            return INVALID
        name = response.tool_calls[0].name
        if name == TOOL_NAME:
            return CONSULT
        return name if name in PHONE_TOOL_NAMES else INVALID
    return DIRECT if response.text.strip() else INVALID


def tool_arguments(response: Response, tool_name: str) -> Mapping[str, Any]:
    """The arguments for one named tool, or nothing if it was not called."""
    for call in response.tool_calls:
        if call.name == tool_name:
            return call.arguments
    return {}


def judge(scenario: Scenario, response: Response) -> Result:
    """One scenario's verdict, or no verdict at all for an unscored category.

    A scenario naming `expect_args` demands both the consult and those
    arguments: the escalation cases exist to measure the effort and model the
    front chose, and a consult at the default knobs is the miss they are
    looking for.
    """
    got = classify(response)
    if not scenario.scored:
        return Result(scenario=scenario, response=response, got=got, hit=None)
    hit = got == scenario.expect
    if hit and scenario.expect_args:
        expected_tool = TOOL_NAME if scenario.expect == CONSULT else scenario.expect
        arguments = tool_arguments(response, expected_tool)
        hit = all(
            str(arguments.get(key, "")) == value
            for key, value in scenario.expect_args.items()
        )
    return Result(scenario=scenario, response=response, got=got, hit=hit)


def accuracy(results: Iterable[Result]) -> tuple[int, int, float]:
    """Hits, scored total, and the fraction — 0.0 rather than a crash on none."""
    scored = [result for result in results if result.scored]
    hits = sum(1 for result in scored if result.hit)
    return hits, len(scored), (hits / len(scored) if scored else 0.0)


def per_category(results: Iterable[Result]) -> dict[str, tuple[int, int]]:
    """Hits and totals by category, in the order the categories first appear."""
    counts: dict[str, tuple[int, int]] = {}
    for result in results:
        if not result.scored:
            continue
        hits, total = counts.get(result.scenario.category, (0, 0))
        counts[result.scenario.category] = (hits + (1 if result.hit else 0), total + 1)
    return counts


def exit_code(results: Sequence[Result], min_accuracy: float) -> int:
    """0 if the run cleared the caller's bar, 1 otherwise.

    A run that scored nothing fails whatever the bar: an eval that measured no
    scenarios has not demonstrated anything to pass.
    """
    _, total, fraction = accuracy(results)
    if not total:
        return 1
    return 0 if fraction >= min_accuracy else 1


def report(results: Sequence[Result]) -> str:
    """The whole run as plain text: the score, the misses, and the unscored.

    Every miss carries what the model actually did — the text it answered with,
    or the arguments it consulted with — because the next move after a miss is
    editing persona.md, and that edit is made against the wording the model
    produced, not against a tally.
    """
    hits, total, fraction = accuracy(results)
    unscored = [result for result in results if not result.scored]

    lines = [
        f"voice eval — {len(results)} scenarios, {total} scored, {len(unscored)} unscored",
        "",
        f"overall  {hits}/{total}  {_percent(fraction)}",
        "",
    ]

    counts = per_category(results)
    width = max((len(name) for name in counts), default=0)
    for name, (category_hits, category_total) in counts.items():
        lines.append(
            f"  {name:<{width}}  {category_hits}/{category_total}  "
            f"{_percent(category_hits / category_total)}"
        )

    misses = [result for result in results if result.hit is False]
    lines.extend(["", f"misses ({len(misses)})"])
    for result in misses:
        lines.append(
            f"  [{result.scenario.id}] {result.scenario.category}: "
            f"expected {_expected(result.scenario)}, got {result.got}"
        )
        lines.append(f"      {_observed(result.response)}")

    if unscored:
        lines.extend(["", "unscored"])
        for result in unscored:
            lines.append(
                f"  [{result.scenario.id}] {result.scenario.category}: got {result.got}"
            )
            lines.append(f"      {_observed(result.response)}")

    return "\n".join(lines)


def _text(value: Any, where: str) -> str:
    """One required string field, rejected if it is blank or not a string."""
    if not isinstance(value, str) or not value.strip():
        raise ScenarioError(f"{where}: expected a non-empty string, got {value!r}")
    return value


def _history(value: Any, where: str) -> tuple[tuple[str, str], ...]:
    """The prior turns, as (role, text) pairs the runner can replay."""
    if not isinstance(value, list):
        raise ScenarioError(f"{where}: history is {type(value).__name__}, not a list")
    if len(value) > HISTORY_TURNS:
        raise ScenarioError(
            f"{where}: history has {len(value)} turns, more than the {HISTORY_TURNS} "
            "the front carries"
        )
    turns: list[tuple[str, str]] = []
    for index, turn in enumerate(value):
        at = f"{where}: history[{index}]"
        if not isinstance(turn, list) or len(turn) != 2:
            raise ScenarioError(f"{at}: expected a [role, text] pair, got {turn!r}")
        role, text = turn
        if role not in _ROLES:
            raise ScenarioError(f"{at}: role is {role!r}, not one of {_ROLES}")
        turns.append((role, _text(text, f"{at}: text")))
    return tuple(turns)


def _expect_args(value: Any, expect: str, where: str) -> Mapping[str, str]:
    """The tool arguments a scenario pins, checked against what the tool takes."""
    if not isinstance(value, dict):
        raise ScenarioError(
            f"{where}: expect_args is {type(value).__name__}, not an object"
        )
    if value and expect == DIRECT:
        raise ScenarioError(
            f"{where}: expect_args on a {expect!r} scenario, which calls no tool"
        )
    if not value:
        return {}
    allowed = TOOL_ARGUMENT_KEYS[TOOL_NAME if expect == CONSULT else expect]
    for key, argument in value.items():
        if key not in allowed:
            raise ScenarioError(
                f"{where}: expect_args key {key!r} is not accepted by {expect!r}"
            )
        _text(argument, f"{where}: expect_args[{key!r}]")
    return dict(value)


def _expected(scenario: Scenario) -> str:
    """The expectation as one phrase, arguments included when it pins them."""
    if not scenario.expect_args:
        return scenario.expect
    pinned = " ".join(f"{key}={value}" for key, value in scenario.expect_args.items())
    return f"{scenario.expect} {pinned}"


def _observed(response: Response) -> str:
    """What the model actually produced, on one line."""
    if response.error:
        return f"error: {response.error}"
    if response.tool_calls:
        return " ".join(_call(call) for call in response.tool_calls)
    return f'text: "{response.text.strip()}"' if response.text.strip() else "(nothing)"


def _call(call: ToolCall) -> str:
    arguments = ", ".join(f"{key}={value}" for key, value in call.arguments.items())
    return f"{call.name}({arguments})"


def _percent(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"
