"""Offline tests for mentat NDJSON streams and commentary chunking."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stream import (
    CommentaryChunker,
    LineSplitter,
    ToolResult,
    ToolStart,
    TurnDone,
    TurnError,
    TurnFailure,
    TurnStream,
)

DONE_OK = b'{"kind":"done","done":{"text":"Hi.","is_error":false}}\n'
DONE_ERR = b'{"kind":"done","done":{"text":"failed","is_error":true}}\n'


class CommentaryChunkerTest(unittest.TestCase):
    def test_emits_complete_sentences_and_holds_trailing_text(self):
        chunker = CommentaryChunker()
        self.assertEqual(chunker.feed("Hello there. Next"), ["Hello there."])
        self.assertEqual(chunker.feed(" sentence!"), [" Next sentence!"])
        self.assertEqual(chunker.flush(), [])

    def test_newline_is_a_boundary(self):
        chunker = CommentaryChunker()
        self.assertEqual(chunker.feed("First line\nSecond"), ["First line\n"])
        self.assertEqual(chunker.flush(), ["Second"])

    def test_long_sentence_splits_at_whitespace_without_losing_text(self):
        text = ("word " * 140) + "finished."
        chunks = CommentaryChunker().feed(text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk.encode()) <= 480 for chunk in chunks))

    def test_long_word_splits_at_character_boundaries(self):
        text = "あ" * 300
        chunker = CommentaryChunker()
        chunks = chunker.feed(text)
        chunks += chunker.flush()
        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk.encode()) <= 480 for chunk in chunks))

    def test_dense_symbols_respect_utf8_cap(self):
        text = "§" * 500 + "!"
        chunks = CommentaryChunker().feed(text)
        self.assertEqual("".join(chunks), text)
        self.assertTrue(all(len(chunk.encode()) <= 480 for chunk in chunks))

    def test_flush_emits_trailing_partial_and_empty_input_is_silent(self):
        chunker = CommentaryChunker()
        self.assertEqual(chunker.feed(""), [])
        self.assertEqual(chunker.flush(), [])
        self.assertEqual(chunker.feed("still waiting"), [])
        self.assertEqual(chunker.flush(), ["still waiting"])


class TurnStreamTest(unittest.TestCase):
    def test_text_deltas_are_spoken_in_order(self):
        stream = TurnStream()
        self.assertEqual(
            stream.feed(
                b'{"kind":"text_delta","text":"Hello"}\n'
                b'{"kind":"text_delta","text":" there"}\n'
            ),
            ["Hello", " there"],
        )

    def test_tool_events_are_surfaced_alongside_text(self):
        stream = TurnStream()
        self.assertEqual(
            stream.feed(
                b'{"kind":"tool_start","tool":"Read"}\n'
                b'{"kind":"text_delta","text":"Checking."}\n'
                b'{"kind":"tool_result","tool":"Read","content":"ok"}\n'
            ),
            [ToolStart("Read"), "Checking.", ToolResult("Read", False)],
        )

    def test_end_conversation_tool_result_pins_success_and_error(self):
        ok = TurnStream()
        self.assertEqual(
            ok.feed(
                b'{"kind":"tool_result","tool":"mcp__mentat__end_conversation",'
                b'"content":"ok"}\n'
            ),
            [ToolResult("mcp__mentat__end_conversation", False)],
        )
        error = TurnStream()
        self.assertEqual(
            error.feed(
                b'{"kind":"tool_result","tool":"mcp__mentat__end_conversation",'
                b'"is_error":true,"content":"failed"}\n'
            ),
            [ToolResult("mcp__mentat__end_conversation", True)],
        )

    def test_thinking_and_unknown_events_remain_silent(self):
        stream = TurnStream()
        self.assertEqual(
            stream.feed(
                b'{"kind":"thinking_delta","text":"hmm"}\n'
                b'{"kind":"thinking","tokens":42}\n'
                b'{"kind":"sparkle","text":"hi"}\n'
            ),
            [],
        )

    def test_empty_text_delta_is_skipped(self):
        self.assertEqual(TurnStream().feed(b'{"kind":"text_delta"}\n'), [])

    def test_clean_done_completes_the_turn(self):
        stream = TurnStream()
        self.assertEqual(
            stream.feed(b'{"kind":"text_delta","text":"partial"}\n' + DONE_OK),
            ["partial", TurnDone()],
        )
        self.assertTrue(stream.done)

    def test_error_done_and_error_line_return_terminal_failures(self):
        self.assertEqual(TurnStream().feed(DONE_ERR), [TurnFailure("failed")])
        self.assertEqual(
            TurnStream().feed(b'{"kind":"error","message":"backend died"}\n'),
            [TurnFailure("backend died")],
        )

    def test_text_before_error_in_one_feed_is_preserved(self):
        stream = TurnStream()
        self.assertEqual(
            stream.feed(
                b'{"kind":"text_delta","text":"partial"}\n'
                b'{"kind":"error","message":"backend died"}\n'
            ),
            ["partial", TurnFailure("backend died")],
        )

    def test_malformed_lines_raise(self):
        with self.assertRaises(TurnError):
            TurnStream().feed(b"{not json\n")
        with self.assertRaises(TurnError):
            TurnStream().feed(b"42\n")


class LineSplitterTest(unittest.TestCase):
    def test_partial_lines_and_multibyte_data_are_held(self):
        splitter = LineSplitter()
        encoded = '{"text":"héllo"}\n'.encode()
        cut = encoded.index("é".encode()) + 1
        self.assertEqual(splitter.feed(encoded[:cut]), [])
        self.assertEqual(splitter.feed(encoded[cut:]), ['{"text":"héllo"}'])

    def test_blank_lines_are_skipped_and_incomplete_tail_is_held(self):
        splitter = LineSplitter()
        self.assertEqual(splitter.feed(b'\n{"a":1}\n{"trunc'), ['{"a":1}'])


if __name__ == "__main__":
    unittest.main()
