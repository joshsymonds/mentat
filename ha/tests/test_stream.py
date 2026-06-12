"""Tests for the pure mentat-wire → chat_log-delta translation.

The wire lines here mirror the NDJSON contract pinned by the daemon's golden
tests (test/wire.test.ts) — the wire format is ours, so these literals are the
same bytes those goldens pin, not hand-guessed protocol shapes.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "custom_components" / "mentat"))

from stream import LineSplitter, TurnError, wire_line_to_delta


class WireLineToDeltaTest(unittest.TestCase):
    def test_text_delta_becomes_content(self):
        self.assertEqual(
            wire_line_to_delta('{"kind":"text_delta","text":"Hello"}'),
            {"content": "Hello"},
        )

    def test_text_delta_with_omitted_text_is_skipped(self):
        # omitempty: the daemon drops the text key when the delta is empty.
        self.assertIsNone(wire_line_to_delta('{"kind":"text_delta"}'))

    def test_thinking_delta_becomes_thinking_content(self):
        self.assertEqual(
            wire_line_to_delta('{"kind":"thinking_delta","text":"hmm"}'),
            {"thinking_content": "hmm"},
        )

    def test_non_text_events_are_skipped(self):
        self.assertIsNone(wire_line_to_delta('{"kind":"thinking","tokens":42}'))
        self.assertIsNone(wire_line_to_delta('{"kind":"tool_start","tool":"Read"}'))
        self.assertIsNone(
            wire_line_to_delta('{"kind":"tool_result","tool":"Read","content":"ok"}')
        )

    def test_unknown_kind_is_skipped(self):
        # Forward compatibility: a daemon newer than this component must not
        # break the turn.
        self.assertIsNone(wire_line_to_delta('{"kind":"sparkle","text":"hi"}'))

    def test_successful_done_is_skipped(self):
        line = (
            '{"kind":"done","done":{"text":"Hi.","is_error":false,'
            '"session_id":"abc","cost_usd":0.01,"input_tokens":1,"output_tokens":2,'
            '"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}'
        )
        self.assertIsNone(wire_line_to_delta(line))

    def test_error_done_raises(self):
        line = (
            '{"kind":"done","done":{"text":"turn exploded","is_error":true,'
            '"session_id":"abc","cost_usd":0,"input_tokens":0,"output_tokens":0,'
            '"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}'
        )
        with self.assertRaises(TurnError) as ctx:
            wire_line_to_delta(line)
        self.assertIn("turn exploded", str(ctx.exception))

    def test_error_line_raises(self):
        with self.assertRaises(TurnError) as ctx:
            wire_line_to_delta('{"kind":"error","message":"backend died"}')
        self.assertIn("backend died", str(ctx.exception))

    def test_malformed_json_raises(self):
        with self.assertRaises(TurnError):
            wire_line_to_delta("{not json")

    def test_non_object_line_raises(self):
        with self.assertRaises(TurnError):
            wire_line_to_delta("42")


class LineSplitterTest(unittest.TestCase):
    def test_two_lines_in_one_chunk(self):
        splitter = LineSplitter()
        self.assertEqual(splitter.feed(b'{"a":1}\n{"b":2}\n'), ['{"a":1}', '{"b":2}'])

    def test_partial_line_across_chunks(self):
        splitter = LineSplitter()
        self.assertEqual(splitter.feed(b'{"kind":"tex'), [])
        self.assertEqual(splitter.feed(b't_delta"}\n'), ['{"kind":"text_delta"}'])

    def test_multibyte_utf8_split_across_chunks(self):
        encoded = '{"text":"héllo"}\n'.encode()
        # Split inside the two-byte é sequence.
        cut = encoded.index("é".encode()) + 1
        splitter = LineSplitter()
        self.assertEqual(splitter.feed(encoded[:cut]), [])
        self.assertEqual(splitter.feed(encoded[cut:]), ['{"text":"héllo"}'])

    def test_blank_lines_are_skipped(self):
        splitter = LineSplitter()
        self.assertEqual(splitter.feed(b'\n{"a":1}\n\n'), ['{"a":1}'])

    def test_incomplete_tail_is_never_emitted(self):
        splitter = LineSplitter()
        self.assertEqual(splitter.feed(b'{"a":1}\n{"trunc'), ['{"a":1}'])


if __name__ == "__main__":
    unittest.main()
