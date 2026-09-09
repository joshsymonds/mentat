"""Pure translation of mentat NDJSON wire bytes into speakable chunks.

No livekit or aiohttp imports — this module is the testable core of the voice
agent (voice/tests/test_stream.py runs it offline with stdlib unittest). The
wire contract is pinned by the daemon's golden tests (test/wire.test.ts).

Only text is spoken. Everything else on the wire — tool activity, thinking,
kinds newer than this adapter — is silent, because the wait a turn spends on
tools is already covered: the front voice says its own holding line before
dispatching the consult, and a pad plays under it until the answer lands
(voice/agent.py).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping


class TurnError(Exception):
    """The turn failed: a terminal error line or an is_error done."""


class LineSplitter:
    """Splits a chunked byte stream into complete NDJSON lines.

    Buffering happens at the byte level so a UTF-8 sequence split across
    chunks survives; '\\n' is a single byte in UTF-8, so splitting before
    decoding is safe. An incomplete tail (connection cut mid-line) is never
    emitted.
    """

    _buffer: bytes

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += chunk
        *complete, self._buffer = self._buffer.split(b"\n")
        return [decoded for raw in complete if (decoded := raw.decode().strip())]


class TurnStream:
    """Accumulates one turn's wire bytes into the text a voice pipeline speaks.

    State is turn-scoped: construct one per turn, never reuse across turns.
    """

    done: bool
    """True once the turn's done event arrived without an error."""

    _splitter: LineSplitter

    def __init__(self) -> None:
        self._splitter = LineSplitter()
        self.done = False

    def feed(self, data: bytes) -> list[str]:
        """Returns the chunks to speak, in order, for these bytes.

        Raises TurnError on the terminal error line and on an is_error done;
        anything already collected from this chunk is dropped with it, since
        the turn is over.
        """
        chunks: list[str] = []
        for line in self._splitter.feed(data):
            spoken = self._consume(line)
            if spoken is not None:
                chunks.append(spoken)
        return chunks

    def _consume(self, line: str) -> str | None:
        try:
            event = json.loads(line)
        except ValueError as err:
            raise TurnError(f"malformed wire line: {line[:120]}") from err
        if not isinstance(event, dict):
            raise TurnError(f"malformed wire line: {line[:120]}")

        kind = event.get("kind")
        if kind == "text_delta":
            # omitempty: the daemon drops the text key when the delta is empty.
            text = event.get("text", "")
            if not text:
                return None
            return str(text)
        if kind == "error":
            raise TurnError(str(event.get("message", "unknown daemon error")))
        if kind == "done":
            done = event.get("done", {})
            if isinstance(done, dict) and done.get("is_error"):
                raise TurnError(str(done.get("text", "turn failed")))
            self.done = True
            return None
        # tool_start, tool_result, thinking_delta, thinking, and kinds newer
        # than this adapter: nothing to say, and forward compatibility means a
        # newer daemon must not break the turn.
        return None


class Respeller:
    """Rewrites whole words on their way to the TTS, and only there.

    A name the voice mispronounces is fixed by handing the synthesizer a
    respelling ("Symonds" -> "Sigh-monds") while the caption keeps the real
    spelling — the transcript is produced from the model's text, not from
    what the TTS was given, so the substitution never shows on a screen.

    Text arrives in model-sized chunks that split words anywhere, so a
    trailing run of word characters is held back until the next chunk or
    flush decides where the word ends. Matching is whole-word and
    case-sensitive: an entry for "Symonds" leaves "symonds" and "Symondson"
    alone.
    """

    _pattern: re.Pattern[str] | None
    _tail: str

    def __init__(self, respellings: Mapping[str, str]) -> None:
        self._map = dict(respellings)
        self._pattern = (
            re.compile(
                r"\b("
                + "|".join(re.escape(w) for w in sorted(self._map, key=len, reverse=True))
                + r")\b"
            )
            if self._map
            else None
        )
        self._tail = ""

    def feed(self, chunk: str) -> str:
        """The text safe to speak so far, respelled; the rest waits."""
        text = self._tail + chunk
        held = re.search(r"\w+\Z", text)
        if held:
            ready, self._tail = text[: held.start()], text[held.start() :]
        else:
            ready, self._tail = text, ""
        return self._respell(ready)

    def flush(self) -> str:
        """Whatever was held back, now that the text has ended."""
        out, self._tail = self._respell(self._tail), ""
        return out

    def _respell(self, text: str) -> str:
        if self._pattern is None or not text:
            return text
        return self._pattern.sub(lambda m: self._map[m.group(1)], text)
