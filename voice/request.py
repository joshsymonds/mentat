"""Pure request, delegation, and close-policy helpers for the voice front."""

from __future__ import annotations

import asyncio
import tomllib
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    "To end the call, say the closing words, then call end_conversation with a reason, "
    "and say nothing after."
)
CONSULT_WINDOW_TURNS = 2
CONSULT_TURN_CHARS = 500
VOICE_CARD_MARKER = "---VOICE-CARD---"
PRIVATE_CONTEXT_ENV = "MENTAT_VOICE_PRIVATE"
CLOSE_QUIET_S = 6.0
IDLE_S = 30.0
END_CONVERSATION_TOOL = "mcp__mentat__end_conversation"


class EndingPolicy:
    """Close state machine for idle and backend-requested call endings."""

    def __init__(self) -> None:
        self._deadline_kind: str | None = None
        self._armed_at: float | None = None
        self._end_tool_succeeded = False
        self._user_speaking = False

    @property
    def deadline(self) -> float | None:
        if self._deadline_kind == "tool":
            return CLOSE_QUIET_S
        if self._deadline_kind == "idle":
            return IDLE_S
        return None

    def _clear_deadline(self) -> bool:
        had_deadline = self._deadline_kind is not None
        self._deadline_kind = None
        self._armed_at = None
        return had_deadline

    def tool_result_seen(self, name: str, is_error: bool) -> None:
        """Remember whether the successful call-ending tool was observed."""
        if name == END_CONVERSATION_TOOL:
            self._end_tool_succeeded = not is_error

    def turn_done(self, now: float) -> None:
        """Arm the close window after a successful ending tool and clean turn."""
        if self._end_tool_succeeded and not self._user_speaking:
            self._deadline_kind = "tool"
            self._armed_at = now
        return None

    def assistant_activity(self, now: float) -> None:
        """Reset the close clock when assistant speech starts or stops."""
        if self._deadline_kind == "tool":
            self._armed_at = now
        return None

    def user_spoke(self) -> str | None:
        """Revoke a pending close when the caller speaks again."""
        self._user_speaking = True
        self._end_tool_succeeded = False
        return "cancel" if self._clear_deadline() else None

    def user_quiet(self) -> None:
        self._user_speaking = False

    def agent_listening(self, now: float) -> None:
        """Arm the existing thirty-second idle close window."""
        if not self._user_speaking and self._deadline_kind is None:
            self._deadline_kind = "idle"
            self._armed_at = now
        return None

    def agent_busy(self) -> str | None:
        """Cancel idle closure while the assistant is working."""
        if self._deadline_kind == "idle":
            return "cancel" if self._clear_deadline() else None
        return None

    def elapsed(self, now: float) -> str | None:
        """Return ``close`` once the active silence window has elapsed."""
        if self._deadline_kind is None or self._armed_at is None:
            return None
        window = CLOSE_QUIET_S if self._deadline_kind == "tool" else IDLE_S
        if now - self._armed_at < window:
            return None
        self._clear_deadline()
        return "close"


class DelegationRunner:
    """Own one cancellable backend delegation task at a time."""

    def __init__(self, run: Callable[[str], Awaitable[None]]) -> None:
        self._run = run
        self._task: asyncio.Task[None] | None = None

    def start(self, delegation_id: str) -> asyncio.Task[None]:
        """Start a delegation, cancelling the previous one first."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = asyncio.create_task(self._run(delegation_id), name=f"delegation:{delegation_id}")
        return self._task

    async def close(self) -> None:
        """Cancel and drain the in-flight delegation, if any."""
        task = self._task
        self._task = None
        if task is not None and not task.done():
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
class PrivateContext:
    """Private deployment context kept outside the public repository."""

    about: str = ""
    keyterms: tuple[str, ...] = ()
    pronunciations: Mapping[str, str] = field(default_factory=dict)


def parse_private_context(text: str) -> PrivateContext:
    """Parse the strict private-context TOML document."""
    data = tomllib.loads(text)
    unknown = set(data) - {"about", "keyterms", "pronunciations"}
    if unknown:
        raise ValueError(f"private context has unknown keys: {sorted(unknown)}")
    about = data.get("about", "")
    keyterms = data.get("keyterms", [])
    pronunciations = data.get("pronunciations", {})
    if not isinstance(about, str):
        raise ValueError("private context: about must be a string")
    if not isinstance(keyterms, list) or not all(isinstance(k, str) for k in keyterms):
        raise ValueError("private context: keyterms must be a list of strings")
    if not isinstance(pronunciations, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in pronunciations.items()
    ):
        raise ValueError("private context: pronunciations must map strings to strings")
    return PrivateContext(
        about=about.strip(),
        keyterms=tuple(keyterms),
        pronunciations=dict(pronunciations),
    )


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
    """Return the latest text-bearing message turns, oldest first."""
    turns = [
        (str(item.role), str(item.text_content))
        for item in items
        if getattr(item, "type", None) == "message"
        and getattr(item, "text_content", None)
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
