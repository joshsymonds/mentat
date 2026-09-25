"""Pure request, delegation, and close-policy helpers for the voice front."""

from __future__ import annotations

import asyncio
import json
import math
import tomllib
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SESSION_PREFIX = "voice-"
TURN_META = {"surface": "voice", "user": "josh"}
TURN_EFFORT = "low"
TURN_MODEL = "sonnet"

CONSULT_FRAMING = (
    "Your answer is streamed to the voice as it arrives and spoken in its own words. "
    "Keep facts, outcomes and uncertainty intact, in concise spoken prose."
)
CONSULT_SMS_RULE = (
    "Before sending a text, state the recipient and the full message and wait for a yes "
    "in a later turn. A yes authorizes exactly that message once; then call send_sms with send=true."
)
CONSULT_ENDING_RULE = (
    "End the call as soon as Josh's intent is complete. When an action is confirmed or a "
    "question is answered and nothing is left open, confirm it in a few words, with no "
    "offer of more help, then call end_conversation with reason done in the same turn. "
    "When Josh signs off, say a short goodbye, then call end_conversation with reason "
    "signoff. Keep the call open only when you asked Josh something you need answered. "
    "Load end_conversation in the same tool search as any other tool the turn needs. "
    "To end the call, say the closing words, then call end_conversation, and say nothing after."
)
CONSULT_WINDOW_TURNS = 2
CONSULT_TURN_CHARS = 500
VOICE_CARD_MARKER = "---VOICE-CARD---"
PRIVATE_CONTEXT_ENV = "MENTAT_VOICE_PRIVATE"
CLOSE_TAIL_S = 1.0
CLOSE_UNSPOKEN_S = 4.0
IDLE_S = 30.0
END_CONVERSATION_TOOL = "mcp__mentat__end_conversation"
# the startup user item that, with the persona's opening policy, makes the voice speak first
CALL_OPENED = "(Josh just opened the call.)"


class EndingPolicy:
    """Close state machine for idle and backend-requested call endings.

    A successful ``end_conversation`` closes the call as soon as the goodbye has
    been spoken: a short tail after the voice goes quiet, or a short wait when no
    goodbye starts at all. Only a newer delegation revokes it; the caller talking
    over the goodbye ("bye!") does not.
    """

    def __init__(self) -> None:
        self._deadline_kind: str | None = None
        self._window: float | None = None
        self._armed_at: float | None = None
        self._end_tool_succeeded = False
        self._user_speaking = False
        self._agent_speaking = False

    def describe(self) -> str:
        """One log-friendly line of the state the close decision reads."""
        return (
            f"deadline={self._deadline_kind} end_tool={self._end_tool_succeeded} "
            f"user_speaking={self._user_speaking} agent_speaking={self._agent_speaking}"
        )

    @property
    def deadline(self) -> float | None:
        return self._window

    def remaining(self, now: float) -> float | None:
        """Seconds until the active window elapses, or None when none is armed."""
        if self._window is None or self._armed_at is None:
            return None
        return max(0.0, self._window - (now - self._armed_at))

    def _arm(self, kind: str, window: float, now: float) -> None:
        self._deadline_kind = kind
        self._window = window
        self._armed_at = now

    def _clear_deadline(self) -> None:
        self._deadline_kind = None
        self._window = None
        self._armed_at = None

    def tool_result_seen(self, name: str, is_error: bool) -> None:
        """Remember whether the successful call-ending tool was observed."""
        if name == END_CONVERSATION_TOOL:
            self._end_tool_succeeded = not is_error

    def delegation_started(self) -> None:
        """Cancel an ending when a newer backend delegation begins."""
        self._clear_deadline()
        self._end_tool_succeeded = False

    def turn_done(self, now: float) -> None:
        """Arm the close after a successful ending tool and clean turn."""
        if not self._end_tool_succeeded:
            return
        if self._agent_speaking:
            self._arm("tool", CLOSE_TAIL_S, now)
        else:
            self._arm("tool", CLOSE_UNSPOKEN_S, now)

    def agent_speaking(self) -> None:
        """Hold the close while the assistant is speaking."""
        self._agent_speaking = True

    def agent_quiet(self, now: float) -> None:
        """Start the short tail once the goodbye has been spoken."""
        self._agent_speaking = False
        if self._deadline_kind == "tool":
            self._arm("tool", CLOSE_TAIL_S, now)

    def user_spoke(self) -> None:
        """Cancel the idle close; a requested ending stands."""
        self._user_speaking = True
        if self._deadline_kind == "idle":
            self._clear_deadline()

    def user_quiet(self) -> None:
        self._user_speaking = False

    def agent_listening(self, now: float) -> None:
        """Arm the existing thirty-second idle close window."""
        if not self._user_speaking and self._deadline_kind is None:
            self._arm("idle", IDLE_S, now)

    def agent_busy(self) -> None:
        """Cancel idle closure while the assistant is working."""
        if self._deadline_kind == "idle":
            self._clear_deadline()

    def elapsed(self, now: float) -> str | None:
        """Return ``close`` once the active window has elapsed."""
        if self._agent_speaking:
            return None
        if self._window is None or self._armed_at is None:
            return None
        if now - self._armed_at < self._window:
            return None
        self._clear_deadline()
        return "close"


class DelegationRunner:
    """Own one cancellable backend delegation task at a time."""

    def __init__(
        self,
        run: Callable[[Any], Awaitable[None]],
        on_error: Callable[[str, BaseException], None],
    ) -> None:
        self._run = run
        self._on_error = on_error
        self._task: asyncio.Task[None] | None = None

    def start(self, delegation: Any) -> asyncio.Task[None]:
        """Start a delegation, cancelling the previous one first."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        task = asyncio.create_task(self._run(delegation), name=f"delegation:{delegation.id}")
        task.add_done_callback(
            lambda completed, delegation_id=str(delegation.id): self._task_done(
                delegation_id, completed
            )
        )
        self._task = task
        return task

    def _task_done(self, delegation_id: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            error = task.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            self._on_error(delegation_id, error)

    async def close(self) -> None:
        """Cancel and drain the in-flight delegation, if any."""
        task = self._task
        self._task = None
        if task is None:
            return
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def run_close_sequence(
    close_player: Callable[[], Awaitable[None]],
    delete_room: Callable[[], Awaitable[None]],
    shutdown_job: Callable[[], Awaitable[None]],
    log: Callable[[str], None],
) -> None:
    """Close audio, delete the room, and always shut down the job."""
    try:
        await close_player()
    except Exception as error:
        log(f"close: audio cleanup failed: {error}")
    try:
        await delete_room()
    except Exception as error:
        log(f"close: room deletion failed: {error}")
    finally:
        await shutdown_job()


@dataclass(frozen=True)
class Place:
    """A named spot Josh is at when his phone is within ``radius_m`` of it."""

    lat: float
    lng: float
    radius_m: float


@dataclass(frozen=True)
class PrivateContext:
    """Private deployment context kept outside the public repository."""

    about: str = ""
    pronunciations: Mapping[str, str] = field(default_factory=dict)
    places: Mapping[str, Place] = field(default_factory=dict)


def parse_private_context(text: str) -> PrivateContext:
    """Parse the strict private-context TOML document."""
    data = tomllib.loads(text)
    unknown = set(data) - {"about", "pronunciations", "places"}
    if unknown:
        raise ValueError(f"private context has unknown keys: {sorted(unknown)}")
    about = data.get("about", "")
    pronunciations = data.get("pronunciations", {})
    if not isinstance(about, str):
        raise ValueError("private context: about must be a string")
    if not isinstance(pronunciations, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in pronunciations.items()
    ):
        raise ValueError("private context: pronunciations must map strings to strings")
    return PrivateContext(
        about=about.strip(),
        pronunciations=dict(pronunciations),
        places=_parse_places(data.get("places", {})),
    )


def _parse_places(raw: Any) -> dict[str, Place]:
    if not isinstance(raw, dict):
        raise ValueError("private context: places must be a table of places")
    places: dict[str, Place] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict) or set(spec) != {"lat", "lng", "radius_m"}:
            raise ValueError(f"private context: place {name!r} needs exactly lat, lng, radius_m")
        values = [spec["lat"], spec["lng"], spec["radius_m"]]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            raise ValueError(f"private context: place {name!r} coordinates must be numbers")
        lat, lng, radius = (float(v) for v in values)
        if not (-90 <= lat <= 90 and -180 <= lng <= 180 and radius > 0):
            raise ValueError(f"private context: place {name!r} is out of range")
        places[name] = Place(lat=lat, lng=lng, radius_m=radius)
    return places


def load_private_context(path: str | None) -> PrivateContext:
    """Load private context from ``path`` or return an empty context."""
    if not path:
        return PrivateContext()
    return parse_private_context(Path(path).read_text())


def with_private_context(instructions: str, private: PrivateContext) -> str:
    """Append private facts and deterministic pronunciation instructions."""
    rendered = instructions
    if private.about:
        rendered += "\n\n" + private.about
    pronunciation_lines = [
        f"Say {word} as {respelling}."
        for word, respelling in sorted(private.pronunciations.items())
    ]
    if pronunciation_lines:
        rendered += "\n" + "\n".join(pronunciation_lines)
    return rendered


def recent_turns(
    items: Iterable[Any], count: int = CONSULT_WINDOW_TURNS
) -> list[tuple[str, str]]:
    """Return the latest text-bearing message turns, oldest first, without the opening cue."""
    turns = [
        (str(item.role), str(item.text_content))
        for item in items
        if getattr(item, "type", None) == "message"
        and getattr(item, "text_content", None)
        and item.text_content != CALL_OPENED
    ]
    return turns[-count:]


def turn_request(
    room_name: str,
    text: str,
    effort: str = TURN_EFFORT,
    model: str = TURN_MODEL,
) -> dict[str, Any]:
    """Build one voice request for the conversation API."""
    return {
        "session_id": SESSION_PREFIX + room_name,
        "text": text,
        "meta": TURN_META,
        "effort": effort,
        "model": model,
    }


def split_persona(text: str) -> tuple[str, str]:
    """Split persona instructions from the consult voice card."""
    instructions, marker, voice_card = text.partition(VOICE_CARD_MARKER)
    if not marker:
        raise ValueError(f"persona text has no {VOICE_CARD_MARKER} line")
    return instructions.strip(), voice_card.strip()


def consult_envelope(
    persona_card: str,
    summary: str,
    last_turns: Iterable[tuple[str, str]],
    question: str,
) -> str:
    """Build a bounded backend prompt with voice, action, and ending rules."""
    sections = [
        CONSULT_FRAMING,
        persona_card,
        "Backend rules:\n" + CONSULT_SMS_RULE + "\n" + CONSULT_ENDING_RULE,
    ]
    if summary.strip():
        sections.append("Conversation so far:\n" + summary)
    window = "\n".join(f"{role}: {_capped(text)}" for role, text in last_turns)
    if window:
        sections.append(window)
    sections.append("Question:\n" + question)
    return "\n\n".join(sections)


def _capped(text: str) -> str:
    """Cap one carried turn with an explicit truncation marker."""
    if len(text) <= CONSULT_TURN_CHARS:
        return text
    return text[:CONSULT_TURN_CHARS] + "…"


# participant attributes mentatd stamps on the phone's token from its call context
ATTR_TIME_ZONE = "mentat.time_zone"
ATTR_LOCATION = "mentat.location"
ATTR_DRIVING = "mentat.driving"
EARTH_RADIUS_M = 6_371_000.0


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    if 0 <= hour < 5:
        return "the middle of the night"
    return "night"


def _distance_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def nearest_place(attributes: Mapping[str, str], places: Mapping[str, Place]) -> str | None:
    """The named place the phone's reported location falls inside, nearest first."""
    try:
        location = json.loads(attributes[ATTR_LOCATION])
        lat, lng = float(location["lat"]), float(location["lng"])
        accuracy = max(0.0, float(location.get("accuracy_m", 0.0)))
    except (KeyError, TypeError, ValueError):
        return None
    best: tuple[float, str] | None = None
    for name, place in places.items():
        distance = _distance_m(lat, lng, place.lat, place.lng)
        # a fuzzy fix may still count, but never by more than the place's own radius
        if distance - min(accuracy, place.radius_m) <= place.radius_m:
            if best is None or distance < best[0]:
                best = (distance, name)
    return best[1] if best else None


def call_timezone(attributes: Mapping[str, str], home: tzinfo) -> tzinfo:
    """The phone's zone when it reported a real one, else the worker's own."""
    name = attributes.get(ATTR_TIME_ZONE)
    if name:
        try:
            return ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError):
            pass
    return home


def call_context(
    attributes: Mapping[str, str],
    places: Mapping[str, Place],
    now: datetime,
) -> str:
    """One instruction paragraph describing Josh's situation as the call opens.

    ``now`` is aware and in the worker's zone, which stands in for home time.
    """
    home = now.tzinfo
    if home is None:
        raise ValueError("call_context needs an aware datetime")
    zone = call_timezone(attributes, home)
    local = now.astimezone(zone)
    clock = local.strftime("%I:%M %p").lstrip("0").lower()
    lines = [f"It's {local.strftime('%A')} {_part_of_day(local.hour)}, {clock}."]
    if local.utcoffset() != now.astimezone(home).utcoffset():
        city = str(zone).rsplit("/", 1)[-1].replace("_", " ")
        lines.append(f"Josh's phone is on {city} time, not home time, so he's probably traveling.")
    if (place := nearest_place(attributes, places)) is not None:
        lines.append(f"Josh is at {place}.")
    if attributes.get(ATTR_DRIVING) == "true":
        lines.append("Josh is driving.")
    return "Call context at the start of this call: " + " ".join(lines)

