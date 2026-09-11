"""Pure translation of mentat NDJSON bytes into commentary and tool events."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass

COMMENTARY_MAX_BYTES = 480


class TurnError(Exception):
    """The turn failed: a terminal error or an error ``done`` event."""


@dataclass(frozen=True)
class ToolResult:
    name: str
    is_error: bool


@dataclass(frozen=True)
class ToolStart:
    name: str


class LineSplitter:
    """Split a chunked UTF-8 byte stream into complete NDJSON lines."""

    def __init__(self) -> None:
        self._buffer = b""

    def feed(self, chunk: bytes) -> list[str]:
        self._buffer += chunk
        *complete, self._buffer = self._buffer.split(b"\n")
        return [decoded for raw in complete if (decoded := raw.decode().strip())]


class TurnStream:
    """Translate one daemon turn into text and tool items in arrival order."""

    def __init__(self) -> None:
        self._splitter = LineSplitter()
        self.done = False

    def feed(self, data: bytes) -> list[str | ToolResult | ToolStart]:
        """Return items for complete lines in ``data``."""
        items: list[str | ToolResult | ToolStart] = []
        for line in self._splitter.feed(data):
            item = self._consume(line)
            if item is not None:
                items.append(item)
        return items

    def _consume(self, line: str) -> str | ToolResult | ToolStart | None:
        try:
            event = json.loads(line)
        except ValueError as err:
            raise TurnError(f"malformed wire line: {line[:120]}") from err
        if not isinstance(event, Mapping):
            raise TurnError(f"malformed wire line: {line[:120]}")

        kind = event.get("kind")
        if kind == "text_delta":
            text = event.get("text", "")
            return str(text) if text else None
        if kind == "tool_start":
            return ToolStart(str(event.get("tool", "")))
        if kind == "tool_result":
            return ToolResult(str(event.get("tool", "")), bool(event.get("is_error", False)))
        if kind == "error":
            raise TurnError(str(event.get("message", "unknown daemon error")))
        if kind == "done":
            done = event.get("done", {})
            if isinstance(done, Mapping) and done.get("is_error"):
                raise TurnError(str(done.get("text", "turn failed")))
            self.done = True
        return None


class CommentaryChunker:
    """Emit commentary at sentence boundaries while respecting a byte cap."""

    def __init__(self, max_bytes: int = COMMENTARY_MAX_BYTES) -> None:
        self._max_bytes = max_bytes
        self._buffer = ""

    def feed(self, text: str) -> list[str]:
        """Add text and emit complete sentences or required cap splits."""
        if not text:
            return []
        self._buffer += text
        output: list[str] = []
        while self._buffer:
            boundary = next(
                (
                    index
                    for index, char in enumerate(self._buffer)
                    if char in ".?!\n"
                ),
                None,
            )
            if boundary is not None:
                end = boundary + 1
                output.extend(self._split_piece(self._buffer[:end]))
                self._buffer = self._buffer[end:]
                continue
            if len(self._buffer.encode("utf-8")) <= self._max_bytes:
                break
            piece, self._buffer = self._take_piece(self._buffer)
            output.append(piece)
        return output

    def flush(self) -> list[str]:
        """Emit all remaining text, splitting oversized pieces safely."""
        if not self._buffer or not self._buffer.strip():
            self._buffer = ""
            return []
        output = self._split_piece(self._buffer)
        self._buffer = ""
        return output

    def _split_piece(self, text: str) -> list[str]:
        output: list[str] = []
        remaining = text
        while remaining:
            if len(remaining.encode("utf-8")) <= self._max_bytes:
                output.append(remaining)
                break
            piece, remaining = self._take_piece(remaining)
            output.append(piece)
        return output

    def _take_piece(self, text: str) -> tuple[str, str]:
        prefix = self._prefix_under_cap(text)
        cut = max(
            (index + 1 for index, char in enumerate(prefix) if index > 0 and char.isspace()),
            default=len(prefix),
        )
        return text[:cut], text[cut:]

    def _prefix_under_cap(self, text: str) -> str:
        used = 0
        end = 0
        for index, char in enumerate(text):
            size = len(char.encode("utf-8"))
            if used + size > self._max_bytes:
                break
            used += size
            end = index + 1
        return text[:end]
