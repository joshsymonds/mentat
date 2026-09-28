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
    spoken_place_pattern: str | None = None


_ALICE_DONOR_CLAIM = (
    r"\bAlice (?:Bertha )?Keck(?: Park)?\s+(?:bought|purchased|donated|gave|gifted)\b|"
    r"\bAlice (?:Bertha )?Keck(?: Park)?\b.{0,100}\bshe\s+(?:bought|purchased|donated|gave|gifted)\b|"
    r"\bAlice (?:Bertha )?Keck Park\b.{0,8}\bwho\s+(?:bought|purchased|donated|gave|gifted)\b|"
    r"\bAlice (?:Bertha )?Keck Park\b[.!?]\s+(?:anonymously\s+)?(?:bought|purchased|donated|gave|gifted)\b|"
    r"\bAlice (?:Bertha )?Keck Park,\s+a (?:local|Santa Barbara)\s+philanthropist[,.]?\s*who\s+(?:bought|purchased|donated|gave|gifted)\b|"
    r"\bAlice (?:Bertha )?Keck Park\s+was\s+a Santa Barbara philanthropist\s+who\s+(?:bought|purchased|donated|gave|gifted)\b|"
    r"\b(?:bought|purchased|donated|gave|gifted|given)\b.{0,80}\b(?:by|from)\s+Alice Keck\b|"
    r"\b(?:gift|donation)\b.{0,80}\b(?:by|from)\s+Alice Keck\b"
)
SMS_CONFIRMATION_PATTERN = re.compile(
    r"\b(?:should i|would you like|do you want|shall i|want me to|say yes|say send|"
    r"say the word|yes\s+or\s+no|(?:just\s+)?say when(?:\s+and i'll send it)?|"
    r"confirm,\s+and i'll send it|sound right|sound good|good to send)\b",
    re.IGNORECASE,
)
_UNCERTAIN_DONOR_CLAIM = (
    r"\b(?:not sure|don't know|do not know|unclear|can't say|cannot say)\b"
    r".{0,40}?\b(?:whether|if)\b.{0,100}?\b(?:bought|purchased|donat\w*|gave|gift\w*)\b"
)
_TRANSFER_ACTION = r"(?:bought|purchased|paid(?:\s+for)?|donat\w*|gave|gift\w*|given)"
_PARK_LAND_OBJECT = (
    r"(?:Alice (?:Bertha )?Keck Park|(?:(?:the|that|this|her|his|their)\s+)?"
    r"(?:land|property|park|garden|site)|it)"
)
_OTHER_ROLE = r"(?:the\s+)?(?:her\s+)?(?:husband|wife|father|mother|daughter|son|family|city|someone else|somebody else|another person)"
_NAMED_OTHER_ACTOR = (
    r"(?-i:(?!(?:Alice|Halis)\s+(?:Bertha\s+)?Keck\b)(?!Keck\s+Park\b)"
    r"(?:[A-Z][A-Za-z'’-]+|[A-Z]\.)(?:\s+(?:[A-Z][A-Za-z'’-]+|[A-Z]\.|de|van|von|da)){1,5})"
)
_OTHER_ACTOR = rf"(?:{_OTHER_ROLE}|{_NAMED_OTHER_ACTOR})"
_ACTION_ACKNOWLEDGMENT = (
    r"\b(?:done|set|start(?:\s+it)?|started|starting|setting|running|sent|sending|texted|cent|scent|"
    r"navigat\w*|directions|route|taking you|sending you there|on the clock)\b"
)
_SMS_SENT_ACKNOWLEDGMENT = r"\b(?:done|sent|sense|sending|texted|cent|scent)\b"
_DIGIT_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}
_DIGIT_WORD_PATTERN = "|".join(_DIGIT_WORDS)
_SPOKEN_SMS_RECIPIENT = re.compile(
    rf"(?P<recipient>(?:\+|plus(?:[\s-]+))?"
    rf"(?:(?:{_DIGIT_WORD_PATTERN}|\d)[\s().,/\-]*){{10,15}})"
    r"\s*[:,.]?\s+",
    re.IGNORECASE,
)
_NON_ALICE_PARK_DONOR = (
    r"(?:"
    rf"\b{_OTHER_ROLE}\b\s*,?\s*who\s+\b{_TRANSFER_ACTION}\b"
    rf"(?:\s+\w+){{0,3}}\s+\b{_PARK_LAND_OBJECT}\b|"
    rf"\b{_OTHER_ROLE}\b(?:\s+\w+){{0,4}}\s+\b{_TRANSFER_ACTION}\b"
    rf"(?:\s+\w+){{0,3}}\s+\b{_PARK_LAND_OBJECT}\b|"
    rf"\b{_NAMED_OTHER_ACTOR}\b(?:\s*,\s*|\s+)(?:(?:was\s+the\s+one\s+)?who\s+)?\b{_TRANSFER_ACTION}\b"
    rf"(?:\s+\w+){{0,3}}\s+\b{_PARK_LAND_OBJECT}\b|"
    rf"\b{_PARK_LAND_OBJECT}\b[^\n.!?;]{{0,50}}\b{_TRANSFER_ACTION}\b"
    rf"(?:\s+\w+){{0,3}}\s+\b(?:by|from)\s+{_OTHER_ACTOR}\b|"
    rf"\b{_OTHER_ACTOR}\b[^\n.!?;]{{0,30}}\b(?:buyer|purchaser|payer|donor|giver)\b"
    rf"[^\n.!?;]{{0,30}}\bof\s+{_PARK_LAND_OBJECT}\b|"
    rf"\b(?:buyer|purchaser|payer|donor|giver)\b[^\n.!?;]{{0,30}}\bof\s+{_PARK_LAND_OBJECT}\b"
    rf"[^\n.!?;]{{0,30}}\b(?:was|is)\s+{_OTHER_ACTOR}\b|"
    rf"\b{_PARK_LAND_OBJECT}\b[^\n.!?;]{{0,30}}\b(?:buyer|purchaser|payer|donor|giver)\b"
    rf"[^\n.!?;]{{0,30}}\b(?:was|is)\s+{_OTHER_ACTOR}\b|"
    rf"\b{_PARK_LAND_OBJECT}\b[^\n.!?;]{{0,40}}\b(?:gift|donation)\b[^\n.!?;]{{0,40}}"
    rf"\b(?:by|from)\s+{_OTHER_ACTOR}\b)"
)


def _uncertain_without_alice_attribution(text: str) -> bool:
    uncertainties = list(re.finditer(_UNCERTAIN_DONOR_CLAIM, text, re.IGNORECASE))
    if not uncertainties:
        return False
    last_uncertainty = uncertainties[-1]
    later = text[last_uncertainty.end():]
    if re.search(_ALICE_DONOR_CLAIM, later, re.IGNORECASE):
        return False
    alice_named = re.search(r"\bAlice Keck(?: Park)?\b", text[:last_uncertainty.end()], re.IGNORECASE)
    later_pronoun_claim = re.search(
        r"\bshe\s+(?:bought|purchased|donated|gave|gifted)\b", later, re.IGNORECASE
    )
    return alice_named is None or later_pronoun_claim is None


SCENARIOS = (
    Scenario(
        name="timer-300-seconds",
        caller_lines=("Set a timer for five minutes.",),
        turns=(
            TurnExpectation((_ACTION_ACKNOWLEDGMENT,)),
        ),
        commands=({"turn": 1, "kind": "timer", "seconds": 300},),
        room_close_after=1,
    ),
    Scenario(
        name="equivalent-alarm",
        caller_lines=("Set an alarm for 7 a.m.",),
        turns=(
            TurnExpectation((_ACTION_ACKNOWLEDGMENT,)),
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
            TurnExpectation((r"\b(?:Alice|Halis) Keck Park(?: Memorial Gardens?)?\b", r"Santa Barbara")),

            TurnExpectation((_ACTION_ACKNOWLEDGMENT,)),
        ),
        commands=(
            {"turn": 1, "kind": "location"},
            {"turn": 2, "kind": "navigate"},
        ),
        room_close_after=2,
        place_query="Alice Keck Park Memorial Garden",
        selected_place_pattern=r"(?i)^Alice Keck Park Memorial Gardens?$",
        spoken_place_pattern=r"(?i)\b(?:Alice|Halis) Keck Park(?: Memorial Gardens?)?\b",
    ),
    Scenario(
        name="sms-say-back-yes",
        caller_lines=("Text +1-202-555-0142: I will be there at six.", "Yes."),
        turns=(
            TurnExpectation(
                (r"\b(?:text|texting|send|message is|it says)\b", SMS_CONFIRMATION_PATTERN.pattern),
                sms_recipient="+1-202-555-0142",
                sms_body="I will be there at six.",
            ),
            TurnExpectation((_SMS_SENT_ACKNOWLEDGMENT,)),
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
                (r"\b(?:text|texting|send|message is|it says)\b", SMS_CONFIRMATION_PATTERN.pattern),
                sms_recipient="+1-202-555-0142",
                sms_body="I will be there at six.",
            ),
            TurnExpectation(
                (r"\b(?:text|texting|send|updated it to say|same number)\b", SMS_CONFIRMATION_PATTERN.pattern),
                sms_recipient="+1-202-555-0142",
                sms_body="I will be there at seven.",
            ),
            TurnExpectation((_SMS_SENT_ACKNOWLEDGMENT,)),
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
                    r"\bAlice (?:Bertha )?Keck(?: Park)?\b",
                    _ALICE_DONOR_CLAIM,
                ),
                reject_patterns=(
                    r"\b(?:not(?!\s+sure)|never|did not|didn't|was not|wasn't)\b.{0,100}\b(?:bought|purchased|donat\w*|gave|gift\w*)\b",
                    _NON_ALICE_PARK_DONOR,
                ),
            ),
            TurnExpectation(
                (r"(?:W\.?\s*M\.?\s*Keck|William\s+(?:M\.?|Myron)\s*Keck)", r"(?:father|daughter)"),
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
    """Extract the complete message say-back before or after its recipient."""
    match = _SPOKEN_SMS_RECIPIENT.search(text)
    prompt = (
        SMS_CONFIRMATION_PATTERN.search(text[match.end():])
        if match is not None
        else None
    )
    spoken_recipient = (
        re.sub(
            r"\b(?:" + _DIGIT_WORD_PATTERN + r")\b",
            lambda digit: _DIGIT_WORDS[digit.group().lower()],
            match.group("recipient").lower(),
        )
        if match
        else ""
    )
    spoken_recipient = re.sub(r"\D", "", spoken_recipient)
    expected_recipient = re.sub(r"\D", "", recipient)
    national_number = expected_recipient[1:] if recipient.startswith("+1") and expected_recipient.startswith("1") else ""
    _require(
        match is not None
        and prompt is not None
        and spoken_recipient in (expected_recipient, national_number),
        f"{scenario_name}: turn {turn} has no complete {recipient} message say-back",
    )

    before_recipient = text[:match.start()]
    prompt_start = match.end() + prompt.start()
    after_recipient = text[match.end():prompt_start]
    if after_recipient.strip(" \t,.:;–—-\"“”"):
        body = after_recipient
    else:
        body = before_recipient
        body = re.sub(r"^\s*(?:oh|okay|ok|sure)[,:]?\s+", "", body, flags=re.IGNORECASE)
        body = re.sub(
            r"(?:,?\s+)(?:ready\s+)?to\s+(?:text|send)\s+to\s*$",
            "",
            body,
            flags=re.IGNORECASE,
        )
    body = " ".join(body.split()).strip(' \t,.:;–—-"“”')
    body = re.sub(r"^i'll say[,:.]?\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"^saying\b\s*,?\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"^(?:and\s+)?the message is[,:]?\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(r"^it says[,:.]?\s*", "", body, flags=re.IGNORECASE)
    body = re.sub(
        r"^(?:the exact message is|the message reads|message)(?:\.{3}|[,:.]|\s+)\s*",
        "",
        body,
        flags=re.IGNORECASE,
    )
    _require(bool(body), f"{scenario_name}: turn {turn} has no complete SMS body say-back")

    prompt_end = match.end() + prompt.end()
    tail = text[prompt_end:]
    _require(
        re.search(r"\b(?:actually|correction|instead|rather|i meant|make that)\b", tail, re.IGNORECASE) is None,
        f"{scenario_name}: turn {turn} contradicts its SMS say-back after the confirmation prompt",
    )
    return body


def _spoken_sms_correction_body(text: str, scenario_name: str, turn: int) -> str:
    """Extract the revised body from a same-recipient correction read-back."""
    prompt = SMS_CONFIRMATION_PATTERN.search(text)
    _require(prompt is not None, f"{scenario_name}: turn {turn} has no SMS confirmation")
    before_prompt = text[:prompt.start()]
    _require(
        re.search(r"\bsame number\b", before_prompt, re.IGNORECASE) is not None,
        f"{scenario_name}: turn {turn} has no confirmed SMS recipient",
    )
    updated = re.search(r"\bupdated it to say\b", before_prompt, re.IGNORECASE)
    same_number = re.search(r"\bsame number\b", before_prompt, re.IGNORECASE)
    _require(
        updated is not None and same_number is not None and updated.end() <= same_number.start(),
        f"{scenario_name}: turn {turn} has no complete corrected SMS body",
    )
    body = before_prompt[updated.end():same_number.start()].strip(' \t,.:;–—-"“”')
    _require(bool(body), f"{scenario_name}: turn {turn} has no complete corrected SMS body")
    tail = text[prompt.end():]
    _require(
        re.search(r"\b(?:actually|correction|instead|rather|i meant|make that)\b", tail, re.IGNORECASE) is None,
        f"{scenario_name}: turn {turn} contradicts its corrected SMS body after confirmation",
    )
    return body


def _sms_body_tokens(body: str) -> tuple[str, ...]:
    """Compare spoken renderings while preserving every message word and value."""
    normalized = re.sub(r"\bi'll\b", "i will", body, flags=re.IGNORECASE)
    normalized = re.sub(r"\bsix\b", "6", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bseven\b", "7", normalized, flags=re.IGNORECASE)
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
_DURATION_UNIT_PATTERN = r"(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)"
_DURATION_NUMBER_TOKEN_PATTERN = (
    r"(?:\d+|" + "|".join(_NUMBER_WORD_VALUES) + r"|hundred|thousand|million|billion|trillion)"
)
_NUMBER_WORD_PATTERN = (
    r"(?:\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"
)
_TIMER_FILLER_PATTERN = re.compile(
    r"\b(?:starting that now|sure|okay|alright),?\s+one\s+sec(?:ond)?\b(?=$|[.!?,;])",
    re.IGNORECASE,
)


def _without_timer_fillers(text: str) -> str:
    """Remove the brief live-observed acknowledgments before checking durations."""
    return _TIMER_FILLER_PATTERN.sub(" ", text)


def _spoken_number(value: str) -> int | None:
    if value.isdigit():
        try:
            return int(value)
        except ValueError:
            return None
    parts = re.split(r"[- ]", value.lower())
    total = 0
    for part in parts:
        number = _NUMBER_WORD_VALUES.get(part)
        if number is None:
            return None
        total += number
    return total


def _spoken_durations(text: str) -> list[int]:
    text = _without_timer_fillers(text)
    durations = []
    pattern = re.compile(
        rf"\b(?P<value>{_NUMBER_WORD_PATTERN})[\s-]*"
        rf"(?P<unit>{_DURATION_UNIT_PATTERN})\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        value = _spoken_number(match.group("value"))
        if value is None:
            continue
        unit = match.group("unit").lower()
        multiplier = 3600 if unit.startswith(("hour", "hr")) else 60 if unit.startswith(("minute", "min")) else 1
        durations.append(value * multiplier)
    return durations


def _spoken_alarm_times(text: str) -> list[tuple[int, int] | None]:
    times = []
    clock_time = (
        rf"(?P<hour>{_NUMBER_WORD_PATTERN})"
        rf"(?:\s*:\s*(?P<colon_minute>\d{{1,2}})|"
        rf"[\s-]+(?P<word_minute>{_NUMBER_WORD_PATTERN}))?"
        r"\s*(?P<meridiem>a\.?m\.?|p\.?m\.?|o['’]?clock)?\b"
    )
    patterns = (
        re.compile(
            rf"\balarm\s+(?:for|at)\s+{clock_time}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<hour>{_NUMBER_WORD_PATTERN})"
            rf"(?:\s*:\s*(?P<colon_minute>\d{{1,2}})|"
            rf"[\s-]+(?P<word_minute>{_NUMBER_WORD_PATTERN}))?"
            r"\s*(?P<meridiem>a\.?m\.?|p\.?m\.?|o['’]?clock)\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?P<hour>{_NUMBER_WORD_PATTERN})\s*:\s*"
            r"(?P<colon_minute>\d{1,2})\b",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            hour = _spoken_number(match.group("hour"))
            minute = (
                int(match.group("colon_minute"))
                if match.group("colon_minute") is not None
                else _spoken_number(match.groupdict().get("word_minute"))
                if match.groupdict().get("word_minute") is not None
                else 0
            )
            if hour is None or hour > 24 or minute is None or minute > 59:
                times.append(None)
                continue
            meridiem = (match.groupdict().get("meridiem") or "").lower().replace(".", "")
            if meridiem.startswith("p") and hour < 12:
                hour += 12
            elif meridiem.startswith("a") and hour == 12:
                hour = 0
            times.append((hour, minute))
    return times


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
        if scenario.name == "alice-keck-context-chain" and turn_number == 1:
            require(
                not _uncertain_without_alice_attribution(text),
                turn_number,
                f"{scenario.name}: turn {turn_number} has uncertainty without a later Alice attribution",
            )
        if expectation.sms_recipient is not None or expectation.sms_body is not None:
            if not require(
                expectation.sms_recipient is not None and expectation.sms_body is not None,
                turn_number,
                f"{scenario.name}: turn {turn_number} must define both SMS recipient and body",
            ):
                continue
            inherited_correction = (
                scenario.name == "sms-correction-new-yes"
                and turn_number > 1
                and bool(spoken_sms)
                and _SPOKEN_SMS_RECIPIENT.search(text) is None
                and re.search(r"\bsame number\b", text, re.IGNORECASE) is not None
            )
            try:
                body = (
                    _spoken_sms_correction_body(text, scenario.name, turn_number)
                    if inherited_correction
                    else _spoken_sms_body(text, expectation.sms_recipient, scenario.name, turn_number)
                )
            except AssertionError as error:
                fail(turn_number, str(error))
            else:
                prior_recipient_matches = (
                    not inherited_correction
                    or spoken_sms[max(spoken_sms)][0] == expectation.sms_recipient
                )
                if require(
                    prior_recipient_matches
                    and _sms_body_tokens(body) == _sms_body_tokens(expectation.sms_body),
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
        if actual.get("kind") in ("timer", "alarm"):
            command_turn = actual.get("turn")
            if isinstance(command_turn, int) and not isinstance(command_turn, bool) and 1 <= command_turn <= len(turns):
                spoken = turns[command_turn - 1]
                if actual.get("kind") == "timer":
                    timer_text = _without_timer_fillers(spoken)
                    durations = _spoken_durations(spoken)
                    spoken_units = re.findall(
                        rf"\b{_DURATION_NUMBER_TOKEN_PATTERN}[\s-]*{_DURATION_UNIT_PATTERN}\b",
                        timer_text,
                        re.IGNORECASE,
                    )
                    require(
                        len(durations) == len(spoken_units)
                        and all(duration == actual.get("seconds") for duration in durations),
                        command_turn,
                        f"{scenario.name}: spoken timer duration {durations!r} did not match fake phone seconds {actual.get('seconds')!r}",
                    )
                else:
                    times = _spoken_alarm_times(spoken)
                    require(
                        all(time == (actual.get("hour"), actual.get("minute")) for time in times),
                        command_turn,
                        f"{scenario.name}: spoken alarm time {times!r} did not match fake phone time {(actual.get('hour'), actual.get('minute'))!r}",
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
                bool(turns)
                and isinstance(name, str)
                and re.search(
                    scenario.spoken_place_pattern or re.escape(name),
                    turns[0],
                    re.IGNORECASE,
                ) is not None,
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
