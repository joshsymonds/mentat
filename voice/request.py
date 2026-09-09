"""Pure logic behind the voice front: what a room sends mentat, and what it logs.

A plain turn is the room's own utterance. A consult is the front voice asking
the deep brain a question mid-conversation and speaking the answer verbatim,
which needs an envelope around it and a check on whether the caller kept
talking during the wait. The last section is the turn-latency line the journal
gets, whose decisions — which speech counts as a turn at all, and how a turn's
two halves of numbers are joined — are the same kind of pure judgment.

No livekit import: the chat-context readers are structural — they read .type,
.role and .text_content off whatever items they are handed — so a real livekit
ChatMessage and a plain test double both work, and voice/tests runs offline
next to voice/stream.py. Everything agent.py does beyond this is glue.
"""

from __future__ import annotations

import tomllib
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: mentat session ids are namespaced so a daemon shared with other surfaces
#: can never collide with a LiveKit room name.
SESSION_PREFIX = "voice-"

# Voice turns are latency-bound: low effort, identified surface, and a fast
# model — the daemon's default is the deepest one, whose latency (and usage
# limits) don't suit a caller waiting in silence. The user is part of the
# turn's authority context (mentat policy is per-turn).
TURN_META = {"surface": "voice", "user": "josh"}
TURN_EFFORT = "low"
TURN_MODEL = "sonnet"

#: Governs the consulted reply's form: mentatd answers in its own register by
#: default, but a consult's answer is spoken by the front voice, unedited.
CONSULT_FRAMING = (
    "Your reply will be read aloud verbatim to the user as a continuation of "
    "this conversation — match this voice and style."
)

#: How many message turns of conversation a consult envelope carries. Two is
#: the question plus what it was about; more re-feeds mentatd its own memory,
#: which it already holds — one persistent session per room.
CONSULT_WINDOW_TURNS = 2

#: Per-turn character cap for the conversation window in a consult envelope.
#: The envelope must stay bounded: mentatd keeps one persistent session per
#: room and already remembers the earlier consults, so an unbounded window
#: re-feeds it its own history at a token cost that grows with the call.
CONSULT_TURN_CHARS = 500

#: Splits persona.md into the front's own instructions and the voice card a
#: consult envelope carries. A pinned literal on its own line: persona.md is
#: written by hand and the split has to survive editing around it.
VOICE_CARD_MARKER = "---VOICE-CARD---"

#: Environment variable naming the private-context file (see PrivateContext).
PRIVATE_CONTEXT_ENV = "MENTAT_VOICE_PRIVATE"

#: Seconds of silence after a completed answer before the front closes.
DONE_GRACE_S = 6.0

#: Seconds of total silence while listening before the front closes.
IDLE_S = 30.0


class EndingPolicy:
    """Pure state machine for model-requested and silent conversation ends.

    The front's event handlers call these methods as state changes happen. A
    deadline is a duration, not a wall-clock timestamp, so the entrypoint can
    replace one timer task whenever a new event changes it.
    """

    def __init__(self) -> None:
        self._deadline: float | None = None
        self._deadline_kind: str | None = None
        self._requested_reason: str | None = None
        self._consult_in_flight = False

    @property
    def deadline(self) -> float | None:
        """The current silence deadline, or ``None`` when no timer is armed."""
        return self._deadline

    @property
    def consult_in_flight(self) -> bool:
        """Whether Mentat is currently working on a consult."""
        return self._consult_in_flight

    @property
    def pending_reason(self) -> str | None:
        """The model-requested end waiting for its speech to finish."""
        return self._requested_reason

    def _clear_deadline(self) -> bool:
        had_deadline = self._deadline is not None
        self._deadline = None
        self._deadline_kind = None
        return had_deadline

    def end_requested(self, reason: str) -> None:
        """Record a model-requested sign-off or completed-answer close."""
        if self._consult_in_flight or reason not in {"signoff", "done"}:
            return None
        self._requested_reason = reason
        self._clear_deadline()
        return None

    def playout_finished(self, delivered: bool) -> str | None:
        """Resolve an end request once its associated speech is terminal."""
        reason = self._requested_reason
        self._requested_reason = None
        if not delivered:
            return None
        if reason == "signoff":
            return "close"
        if reason == "done":
            self._deadline = DONE_GRACE_S
            self._deadline_kind = "done"
        return None

    def consult_answered(self, text: str, delivered: bool) -> None:
        """Arm the short close window for a delivered, non-question answer."""
        if delivered and not text.strip().endswith("?"):
            self._deadline = DONE_GRACE_S
            self._deadline_kind = "done"
        else:
            self._clear_deadline()
        return None

    def consult_started(self) -> str | None:
        """Suppress idle closure while Mentat is working."""
        self._consult_in_flight = True
        self._requested_reason = None
        return "cancel" if self._clear_deadline() else None

    def consult_finished(self) -> None:
        """Resume the idle close window after a consult finishes."""
        self._consult_in_flight = False
        if self._deadline is None:
            self._deadline = IDLE_S
            self._deadline_kind = "idle"
        return None

    def user_spoke(self) -> str | None:
        """Cancel any pending close when Josh starts another utterance."""
        had_request = self._requested_reason is not None
        self._requested_reason = None
        return "cancel" if self._clear_deadline() or had_request else None

    def agent_listening(self) -> None:
        """Arm idle closure when the agent is listening for a new turn."""
        if not self._consult_in_flight and self._deadline is None:
            self._deadline = IDLE_S
            self._deadline_kind = "idle"
        return None

    def agent_busy(self) -> str | None:
        """Stop idle closure while the agent is speaking or thinking."""
        if self._deadline_kind == "idle":
            return "cancel" if self._clear_deadline() else None
        return None

    def elapsed(self, seconds: float) -> str | None:
        """Close when an armed deadline has elapsed outside a consult."""
        if self._consult_in_flight or self._deadline is None:
            return None
        if seconds < self._deadline:
            return None
        self._clear_deadline()
        return "close"


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
    """What the front knows about Josh that must never sit in the repository.

    The repository is public, so persona.md carries the voice and nothing
    about the person. This file — a TOML document deployed as a secret,
    never a store path — carries the rest: a paragraph of who he is for the
    instructions, the names that speech recognition should expect, and the
    words the synthesizer says wrong from the spelling alone.
    """

    about: str = ""
    keyterms: tuple[str, ...] = ()
    pronunciations: Mapping[str, str] = field(default_factory=dict)


def parse_private_context(text: str) -> PrivateContext:
    """The private-context TOML: `about`, `keyterms`, `[pronunciations]`."""
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
    """The private context at `path`, or an empty one when no path is set.

    Empty is a supported deployment (a dev room, CI, a fresh host): the voice
    still works, it just knows no one. A path that is set but unreadable is
    not — that is a broken deploy and raises.
    """
    if not path:
        return PrivateContext()
    return parse_private_context(Path(path).read_text())


def with_private_context(instructions: str, private: PrivateContext) -> str:
    """The front's instructions with the private paragraph folded in."""
    if not private.about:
        return instructions
    return f"{instructions}\n\n{private.about}"


def count_user_messages(items: Iterable[Any]) -> int:
    """How many user messages the chat context holds right now."""
    return sum(
        1
        for item in items
        if getattr(item, "type", None) == "message" and item.role == "user"
    )


def conversation_advanced(items: Iterable[Any], users_at_dispatch: int) -> bool:
    """Did the user say something new since the consult was dispatched?

    Only user messages count. Comparing raw context length would fire on every
    consult: the framework appends the function-call and function-output items
    itself, and the front's spoken front-matter lands as an assistant message
    while the consult is still in flight. A caller who kept talking through the
    wait, though, has moved the conversation on — and the answer, arriving
    afterwards, has to be reoriented before it is spoken.
    """
    return count_user_messages(items) > users_at_dispatch


def without_last_user_message(items: Iterable[Any]) -> list[Any]:
    """The chat items with the most recent user message removed.

    What a not_for_me call erases: the line that turned out not to be for
    the front. Only that one — the reply it would have gotten never exists,
    since the tool stops the response before anything is generated.
    """
    kept = list(items)
    for index in range(len(kept) - 1, -1, -1):
        item = kept[index]
        if getattr(item, "type", None) == "message" and item.role == "user":
            del kept[index]
            break
    return kept


def recent_turns(
    items: Iterable[Any], count: int = CONSULT_WINDOW_TURNS
) -> list[tuple[str, str]]:
    """The last few (role, text) message turns, oldest first.

    The window a consult sends with its question. Oldest first because it is
    read as conversation; reversed, the exchange would read backwards to the
    consulted model. Messages carrying no text are skipped rather than sent as
    a bare "user:" line with nothing after it.
    """
    turns = [
        (str(item.role), str(item.text_content))
        for item in items
        if getattr(item, "type", None) == "message" and item.text_content
    ]
    return turns[-count:]


def turn_request(
    room_name: str,
    text: str,
    effort: str = TURN_EFFORT,
    model: str = TURN_MODEL,
) -> dict[str, Any]:
    """The POST body for one /v1/conversation turn from this room.

    effort and model default to the fast pairing every plain voice turn wants;
    a consult escalates them per call. Values are passed through unvalidated —
    the daemon owns that vocabulary, and duplicating it here would only turn a
    clear daemon rejection into a local one that drifts out of date.
    """
    return {
        "session_id": SESSION_PREFIX + room_name,
        "text": text,
        "meta": TURN_META,
        "effort": effort,
        "model": model,
    }


def split_persona(text: str) -> tuple[str, str]:
    """persona.md's two halves: the front's instructions and its voice card.

    Both halves describe the same voice, which is why they share a file: the
    card is what mentatd is told to sound like when its answer is read out in
    that voice, and a card that drifts from the instructions above it would
    make the consulted answers sound like someone else.
    """
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
    """The consult text: what the front voice sends the deep brain.

    mentatd's reply is spoken verbatim, so the envelope leads with that
    constraint and the persona it must sound like, then gives just enough
    conversation for the question to make sense.
    """
    sections = [CONSULT_FRAMING, persona_card]
    # An empty heading is noise the consulted model would have to interpret,
    # so the label leaves with its section.
    if summary.strip():
        sections.append("Conversation so far:\n" + summary)
    window = "\n".join(f"{role}: {_capped(text)}" for role, text in last_turns)
    if window:
        sections.append(window)
    sections.append("Question:\n" + question)
    return "\n\n".join(sections)


def _capped(text: str) -> str:
    """One turn's text, cut at the cap with a visible truncation marker."""
    if len(text) <= CONSULT_TURN_CHARS:
        return text
    return text[:CONSULT_TURN_CHARS] + "…"


#: The per-turn latency fields the journal gets, in the order a turn passes
#: through them: the caller stops talking, the turn is endpointed, the
#: transcript lands, the front's first token arrives, the first audio frame
#: comes back, and speech begins. Named in the line so a journal reader can
#: recover each stage without knowing this code.
LATENCY_FIELDS = (
    ("endpoint", "end_of_turn_delay"),
    ("transcript", "transcription_delay"),
    ("llm_ttft", "llm_node_ttft"),
    ("tts_ttfb", "tts_node_ttfb"),
    ("e2e", "e2e_latency"),
)

#: Marks an assistant message as a reply the pipeline generated itself. Only
#: that path measures time to first token; speech pushed out through
#: session.say — a consulted answer — carries the speech and playback numbers
#: say() does measure, which is why the presence of metrics cannot be the test
#: and this key is.
PIPELINE_REPLY_KEY = "llm_node_ttft"


def turn_latency(
    pending: Mapping[str, Any], role: str, metrics: Mapping[str, Any]
) -> tuple[str | None, Mapping[str, Any]]:
    """One chat item's effect on the journal: a line to log, and what to hold.

    A turn's numbers arrive split across two messages — the endpoint and
    transcription delays are measured on the caller's turn, the model and
    speech latencies on the reply that answers it — so the caller's half is
    held until a reply lands, and the two are reported as one line rather than
    two the reader has to pair up by eye.

    Speech the pipeline did not generate is no turn and yields no line, and it
    must not spend the held half either: a consulted answer is spoken in the
    gap between the question and the reply that half belongs to.
    """
    if role == "user":
        return None, metrics
    if role != "assistant" or PIPELINE_REPLY_KEY not in metrics:
        return None, pending
    joined = {**pending, **metrics}
    line = " ".join(
        f"{name}={_seconds(joined.get(key))}" for name, key in LATENCY_FIELDS
    )
    return line, {}


def _seconds(value: float | None) -> str:
    """One latency field, or a mark that this turn never measured it."""
    return "?" if value is None else f"{value:.3f}s"
