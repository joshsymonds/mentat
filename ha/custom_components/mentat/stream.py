"""Pure translation of mentat NDJSON wire lines into chat_log delta dicts.

No Home Assistant imports — this module is the testable core of the
integration (ha/tests/test_stream.py runs it offline with stdlib unittest).
The wire contract is pinned by the daemon's golden tests (test/wire.test.ts).
"""

from __future__ import annotations

import json


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


def wire_line_to_delta(line: str) -> dict[str, str] | None:
    """Maps one wire line to an AssistantContentDeltaDict-shaped dict.

    Returns None for lines that carry nothing to stream (tool events, a
    successful done, kinds newer than this component). Raises TurnError for
    the terminal error line and for a done with is_error set.
    """
    try:
        event = json.loads(line)
    except ValueError as err:
        raise TurnError(f"malformed wire line: {line[:120]}") from err
    if not isinstance(event, dict):
        raise TurnError(f"malformed wire line: {line[:120]}")

    kind = event.get("kind")
    if kind == "text_delta":
        text = event.get("text", "")
        return {"content": text} if text else None
    if kind == "thinking_delta":
        text = event.get("text", "")
        return {"thinking_content": text} if text else None
    if kind == "error":
        raise TurnError(str(event.get("message", "unknown daemon error")))
    if kind == "done":
        done = event.get("done", {})
        if isinstance(done, dict) and done.get("is_error"):
            raise TurnError(str(done.get("text", "turn failed")))
        return None
    return None
