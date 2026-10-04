"""Pure semantic yes/no questions for voice-eval reply-text families.

The fixture dispatcher accepts these family/input shapes:

* ``action_ack``: ``action`` and ``expected`` strings.
* ``timer_duration``: positive integer ``seconds``.
* ``alarm_time``: integer ``hour`` (0–23) and ``minute`` (0–59).
* ``place_lookup``: ``place_name`` and ``locality`` strings.
* ``sms_say_back``: ``recipient`` and ``body`` strings; optionally
  ``verified_recipient`` for an already-confirmed recipient inherited by a
  correction.
* ``sms_send_ack``: ``recipient`` string.
* ``sms_confirmation``: the same inputs as ``sms_say_back``.
* ``claims``: ``required`` and ``rejected`` lists of semantic claim strings.
* ``alice_donor``: ``donor`` and ``action`` strings.
* ``alice_father``: ``person`` and ``father`` strings.
* ``alice_wealth``: ``person`` and ``source`` strings.
* ``spanish_switch``: ``turn`` integer from 1 through 5.
* ``interpreter_turn``: ``turn`` integer from 1 through 6.

Question wording describes facts, never text-matching patterns. Each returned
question is phrased so a ``yes`` means the reply satisfies that criterion.
"""

from __future__ import annotations

import re
from collections.abc import Mapping


def _text(inputs: Mapping[str, object], name: str) -> str:
    value = inputs.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _turn(inputs: Mapping[str, object], maximum: int) -> int:
    value = inputs.get("turn")
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"turn must be an integer from 1 through {maximum}")
    return value


# Naming what the action is not keeps the judge from accepting a neighbouring
# action ("your alarm is set for five minutes" for a timer).
_ACTION_CONTRASTS = {
    "timer": "an alarm or a reminder",
    "alarm": "a timer or a reminder",
    "navigate": "only a description of the place without starting directions",
}


def _build_action_ack(inputs: Mapping[str, object]) -> dict[str, str]:
    action = _text(inputs, "action")
    expected = _text(inputs, "expected")
    contrast = _ACTION_CONTRASTS.get(action)
    if contrast is None:
        raise ValueError(f"action must be one of {sorted(_ACTION_CONTRASTS)}")
    if action == "alarm":
        question = (
            f"Does the reply say that an alarm, not {contrast}, was successfully set "
            f"with this expected result: {expected}?"
        )
    elif action == "navigate":
        question = (
            f"Does the reply say that directions to {expected} were started, rather than only "
            "describing the place?"
        )
    else:
        question = (
            f"Does the reply say that a {action}, not {contrast}, was successfully set or "
            f"started with this expected result: {expected}?"
        )
    return {"action_completed": question}


def _build_timer_duration(inputs: Mapping[str, object]) -> dict[str, str]:
    seconds = inputs.get("seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds <= 0:
        raise ValueError("seconds must be a positive integer")
    minutes, remaining_seconds = divmod(seconds, 60)
    if remaining_seconds == 0 and minutes:
        unit = "minute" if minutes == 1 else "minutes"
        duration = f"{seconds} seconds ({minutes} {unit})"
    else:
        duration = f"{seconds} seconds"
    return {
        "timer_duration": (
            f"Does the reply state that the timer duration is exactly {duration}, accepting "
            "equivalent spoken number and time-unit phrasing but rejecting any changed "
            "duration or additional timer length?"
        )
    }


def _build_alarm_time(inputs: Mapping[str, object]) -> dict[str, str]:
    hour = inputs.get("hour")
    minute = inputs.get("minute")
    if isinstance(hour, bool) or not isinstance(hour, int) or not 0 <= hour <= 23:
        raise ValueError("hour must be an integer from 0 through 23")
    if isinstance(minute, bool) or not isinstance(minute, int) or not 0 <= minute <= 59:
        raise ValueError("minute must be an integer from 0 through 59")
    hour12 = hour % 12 or 12
    meridiem = "a.m." if hour < 12 else "p.m."
    if minute == 0:
        equivalent = f"{_small_number(hour12)} o'clock in the {'morning' if hour < 12 else 'afternoon or evening'}"
    else:
        equivalent = f"{_small_number(hour12)} {_small_number(minute)} in the {'morning' if hour < 12 else 'afternoon or evening'}"
    expected = f"{hour12}:{minute:02d} {meridiem} ({equivalent})"
    return {
        "alarm_time": (
            f"Does the reply state that the alarm is set for exactly {expected}, accepting "
            "equivalent spoken clock-time phrasing but rejecting a different time or a.m./p.m.?"
        )
    }


def _small_number(value: int) -> str:
    ones_to_nineteen = (
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen",
    )
    tens = ("", "", "twenty", "thirty", "forty", "fifty")
    if value < len(ones_to_nineteen):
        return ones_to_nineteen[value]
    ten, one = divmod(value, 10)
    return tens[ten] if one == 0 else f"{tens[ten]}-{ones_to_nineteen[one]}"


def _build_place_lookup(inputs: Mapping[str, object]) -> dict[str, str]:
    place_name = _text(inputs, "place_name")
    locality = _text(inputs, "locality")
    return {
        "place_named": (
            f"Does the reply identify the requested place as {place_name} and correctly "
            f"associate it with {locality}?"
        )
    }


# Scribe writes numbers as numerals or as spoken digit words; without the
# examples the judge accepted "eight zero zero" where 202 was expected.
_DIGIT_WORDS = "Punctuation does not matter, and spoken digit words count as digits (two zero two is 202, eight zero zero is 800)"


def _spoken_recipient(recipient: str) -> str:
    """Name a NANP number digit by digit; the judge compares those reliably."""
    digits = re.sub(r"\D", "", recipient)
    if recipient.startswith("+1") and len(digits) == 11:
        return f"exactly {' '.join(digits[1:])} with or without a leading +1 or 1"
    return f"exactly {' '.join(digits)}"


def _build_sms_say_back(inputs: Mapping[str, object]) -> dict[str, str]:
    recipient = _text(inputs, "recipient")
    body = _text(inputs, "body")
    verified_recipient = inputs.get("verified_recipient")
    number = _spoken_recipient(recipient)
    if verified_recipient is not None:
        if not isinstance(verified_recipient, str) or not verified_recipient.strip():
            raise ValueError("verified_recipient must be a non-empty string")
        if verified_recipient.strip() != recipient:
            raise ValueError("verified_recipient must match the expected recipient")
        recipient_question = (
            f"{_DIGIT_WORDS}: does the reply say the text will go to the previously verified "
            f"number, {number}, either by giving those digits or by calling it the same number?"
        )
    else:
        recipient_question = (
            f"{_DIGIT_WORDS}, and words like the same number are not digits: does the reply "
            f"give the digits of the phone number it will text, and are they {number}?"
        )
    return {
        "sms_recipient": recipient_question,
        "sms_body": (
            "A time may be said differently (six, 6, 6:00, six o'clock) but must be the same "
            "time, and no other detail may change: does the reply read back the message body "
            f"with the same meaning as {body!r}?"
        ),
    }


def _build_sms_confirmation(inputs: Mapping[str, object]) -> dict[str, str]:
    questions = _build_sms_say_back(inputs)
    questions["sms_confirmation"] = (
        "Does the reply explicitly ask the caller to confirm or authorize the message "
        "before sending it, rather than announce that it has already been sent?"
    )
    return questions


def _build_sms_send_ack(inputs: Mapping[str, object]) -> dict[str, str]:
    recipient = _text(inputs, "recipient")
    return {
        "sms_sent": (
            f"Does the reply clearly confirm that the text message to {recipient} was sent, "
            "rather than merely drafted, offered, or still awaiting confirmation?"
        )
    }


def _build_claims(inputs: Mapping[str, object]) -> dict[str, str]:
    required = inputs.get("required")
    rejected = inputs.get("rejected")
    if not isinstance(required, list) or not isinstance(rejected, list):
        raise ValueError("required and rejected must be lists of semantic claims")
    required_claims = [_claim(value, "required") for value in required]
    rejected_claims = [_claim(value, "rejected") for value in rejected]
    if not required_claims and not rejected_claims:
        raise ValueError("at least one required or rejected claim is needed")
    questions = {
        f"required_{index}": f"Does the reply accurately state this required fact: {claim}?"
        for index, claim in enumerate(required_claims, 1)
    }
    questions.update(
        {
            f"rejected_{index}": f"Does the reply avoid claiming this incorrect or prohibited fact: {claim}?"
            for index, claim in enumerate(rejected_claims, 1)
        }
    )
    return questions


def _claim(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"each {field} claim must be a non-empty string")
    return value.strip()


def _build_alice_donor(inputs: Mapping[str, object]) -> dict[str, str]:
    donor = _text(inputs, "donor")
    action = _text(inputs, "action")
    return {
        "alice_donor": (
            f"Does the reply accurately identify {donor} as the person who {action}, "
            "without unresolved uncertainty or attributing that contribution to someone else?"
        )
    }


def _build_alice_father(inputs: Mapping[str, object]) -> dict[str, str]:
    person = _text(inputs, "person")
    father = _text(inputs, "father")
    return {
        "alice_father": (
            f"Does the reply correctly state that {father} was {person}'s father, and avoid "
            "expressing uncertainty or denying that relationship?"
        )
    }


def _build_alice_wealth(inputs: Mapping[str, object]) -> dict[str, str]:
    person = _text(inputs, "person")
    source = _text(inputs, "source")
    return {
        "alice_wealth": (
            f"Does the reply accurately explain that {person}'s wealth came from {source}, "
            "without denying or expressing uncertainty about that source?"
        )
    }


_SPANISH_SWITCH_QUESTIONS = {
    1: "Does the reply acknowledge the request and switch to responding in Spanish?",
    2: "Does the reply answer that the capital of France is Paris, in Spanish?",
    3: "Is the reply itself written in English, and does it acknowledge switching back to English?",
    4: "Does the reply acknowledge the request and switch to responding in Spanish again?",
    5: "Does the reply answer that a clear daytime sky is blue, in Spanish?",
}


def _build_spanish_switch(inputs: Mapping[str, object]) -> dict[str, str]:
    turn = _turn(inputs, len(_SPANISH_SWITCH_QUESTIONS))
    return {"spanish_switch": _SPANISH_SWITCH_QUESTIONS[turn]}


_INTERPRETER_TURN_QUESTIONS = {
    1: (
        "Does the reply tell the Spanish-speaking gardener in Spanish that the caller is "
        "ready, without adding a different message?"
    ),
    2: (
        "Does the reply translate the gardener's statement into English: the soil or ground "
        "is too dry to plant tomatoes?"
    ),
    3: (
        "Does the reply translate the caller's instruction into Spanish: water the seedlings "
        "every morning before the sun becomes strong or intense?"
    ),
    4: (
        "Does the reply translate into English that the gardener is asking to set a five-minute "
        "timer to water the plants, without claiming that a timer was actually set?"
    ),
    5: (
        "Does the reply relay in English that the gardener said ‘stop translating’, presenting "
        "it as quoted speech from the gardener rather than as an instruction to stop?"
    ),
    6: (
        "Does the reply acknowledge the caller's request to stop interpreting and return to "
        "speaking English?"
    ),
}


def _build_interpreter_turn(inputs: Mapping[str, object]) -> dict[str, str]:
    turn = _turn(inputs, len(_INTERPRETER_TURN_QUESTIONS))
    return {"interpreter_turn": _INTERPRETER_TURN_QUESTIONS[turn]}


_BUILDERS = {
    "action_ack": _build_action_ack,
    "timer_duration": _build_timer_duration,
    "alarm_time": _build_alarm_time,
    "place_lookup": _build_place_lookup,
    "sms_say_back": _build_sms_say_back,
    "sms_send_ack": _build_sms_send_ack,
    "sms_confirmation": _build_sms_confirmation,
    "claims": _build_claims,
    "alice_donor": _build_alice_donor,
    "alice_father": _build_alice_father,
    "alice_wealth": _build_alice_wealth,
    "spanish_switch": _build_spanish_switch,
    "interpreter_turn": _build_interpreter_turn,
}


def build_questions(family: str, inputs: Mapping[str, object]) -> dict[str, str]:
    """Build yes/no questions for one fixture family from semantic facts."""
    if not isinstance(family, str) or family not in _BUILDERS:
        raise ValueError(f"unsupported question family: {family!r}")
    if not isinstance(inputs, Mapping):
        raise ValueError("inputs must be a mapping of semantic facts")
    return _BUILDERS[family](inputs)
