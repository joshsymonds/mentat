"""Tests for the pure mentat-wire → speakable-chunk translation.

The wire lines here mirror the NDJSON contract pinned by the daemon's golden
tests (test/wire.test.ts) — the wire format is ours, so these literals are the
same bytes those goldens pin, not hand-guessed protocol shapes.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stream import LineSplitter, Respeller, TurnError, TurnStream

DONE_OK = (
    b'{"kind":"done","done":{"text":"Hi.","is_error":false,'
    b'"stop_reason":"end_turn","session_id":"abc","cost_usd":0.01,'
    b'"input_tokens":1,"output_tokens":2,'
    b'"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}\n'
)
DONE_ERR = (
    b'{"kind":"done","done":{"text":"turn exploded","is_error":true,'
    b'"session_id":"abc","cost_usd":0,"input_tokens":0,"output_tokens":0,'
    b'"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}\n'
)


class TurnStreamTextTest(unittest.TestCase):
    def test_text_deltas_are_spoken_in_order(self):
        stream = TurnStream()
        self.assertEqual(
            stream.feed(b'{"kind":"text_delta","text":"Hello"}\n'
                        b'{"kind":"text_delta","text":" there"}\n'),
            ["Hello", " there"],
        )

    def test_text_delta_with_omitted_text_is_skipped(self):
        # omitempty: the daemon drops the text key when the delta is empty.
        stream = TurnStream()
        self.assertEqual(stream.feed(b'{"kind":"text_delta"}\n'), [])

    def test_non_speakable_kinds_yield_nothing(self):
        stream = TurnStream()
        self.assertEqual(
            stream.feed(
                b'{"kind":"thinking_delta","text":"hmm"}\n'
                b'{"kind":"thinking","tokens":42}\n'
                b'{"kind":"tool_result","tool":"Read","content":"ok"}\n'
            ),
            [],
        )

    def test_tool_start_yields_nothing(self):
        # Tool activity is silent. The wait a turn spends on tools is covered
        # by the front voice's own holding line, not by anything said here.
        stream = TurnStream()
        self.assertEqual(stream.feed(b'{"kind":"tool_start","tool":"Read"}\n'), [])

    def test_repeated_tool_starts_stay_silent(self):
        stream = TurnStream()
        self.assertEqual(stream.feed(b'{"kind":"tool_start","tool":"Read"}\n'), [])
        self.assertEqual(stream.feed(b'{"kind":"tool_start","tool":"Grep"}\n'), [])

    def test_tool_start_before_text_yields_only_the_text(self):
        stream = TurnStream()
        self.assertEqual(
            stream.feed(
                b'{"kind":"tool_start","tool":"Read"}\n'
                b'{"kind":"text_delta","text":"Tuesday."}\n'
            ),
            ["Tuesday."],
        )

    def test_unknown_kind_yields_nothing(self):
        # Forward compatibility: a daemon newer than this adapter must not
        # break the turn.
        stream = TurnStream()
        self.assertEqual(stream.feed(b'{"kind":"sparkle","text":"hi"}\n'), [])


class TurnStreamTerminalTest(unittest.TestCase):
    def test_clean_done_completes_the_turn(self):
        stream = TurnStream()
        self.assertFalse(stream.done)
        self.assertEqual(stream.feed(DONE_OK), [])
        self.assertTrue(stream.done)

    def test_error_done_raises(self):
        stream = TurnStream()
        with self.assertRaises(TurnError) as ctx:
            stream.feed(DONE_ERR)
        self.assertIn("turn exploded", str(ctx.exception))
        self.assertFalse(stream.done)

    def test_error_line_raises(self):
        stream = TurnStream()
        with self.assertRaises(TurnError) as ctx:
            stream.feed(b'{"kind":"error","message":"backend died"}\n')
        self.assertIn("backend died", str(ctx.exception))

    def test_text_before_an_error_in_the_same_chunk_is_never_surfaced(self):
        # The turn is over the moment the error line lands, so the deltas
        # collected alongside it in that chunk die with it — the caller speaks
        # the apology, not the front half of a sentence the daemon abandoned.
        delta = b'{"kind":"text_delta","text":"Everything is fi"}\n'
        # Alone that delta is speech, so the drop below is a real one and not a
        # line the adapter was going to swallow anyway.
        self.assertEqual(TurnStream().feed(delta), ["Everything is fi"])

        stream = TurnStream()
        surfaced = None
        with self.assertRaises(TurnError) as ctx:
            surfaced = stream.feed(
                delta + b'{"kind":"error","message":"backend died"}\n'
            )
        self.assertIn("backend died", str(ctx.exception))
        self.assertIsNone(surfaced)

    def test_malformed_json_raises(self):
        stream = TurnStream()
        with self.assertRaises(TurnError):
            stream.feed(b"{not json\n")

    def test_non_object_line_raises(self):
        stream = TurnStream()
        with self.assertRaises(TurnError):
            stream.feed(b"42\n")


class TurnStreamBytesTest(unittest.TestCase):
    def test_multibyte_utf8_split_across_chunks(self):
        encoded = '{"kind":"text_delta","text":"héllo"}\n'.encode()
        # Split inside the two-byte é sequence.
        cut = encoded.index("é".encode()) + 1
        stream = TurnStream()
        self.assertEqual(stream.feed(encoded[:cut]), [])
        self.assertEqual(stream.feed(encoded[cut:]), ["héllo"])

    def test_partial_trailing_line_is_held_until_its_newline(self):
        stream = TurnStream()
        self.assertEqual(stream.feed(b'{"kind":"text_delta","text":"par'), [])
        self.assertEqual(stream.feed(b'tial"}\n'), ["partial"])

    def test_blank_lines_are_skipped(self):
        stream = TurnStream()
        self.assertEqual(
            stream.feed(b'\n{"kind":"text_delta","text":"hi"}\n\n'), ["hi"]
        )


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


class RespellerTest(unittest.TestCase):
    def setUp(self):
        self.respeller = Respeller({"Symonds": "Sigh-monds"})

    def test_whole_word_is_respelled_and_the_rest_left_alone(self):
        out = self.respeller.feed("Josh Symonds is here. ") + self.respeller.flush()
        self.assertEqual(out, "Josh Sigh-monds is here. ")

    def test_a_word_split_across_chunks_is_still_one_word(self):
        # The model streams tokens, and a name can land as "Sym" + "onds".
        out = self.respeller.feed("Josh Sym")
        out += self.respeller.feed("onds called.")
        out += self.respeller.flush()
        self.assertEqual(out, "Josh Sigh-monds called.")

    def test_a_trailing_word_is_held_until_flush(self):
        # Nothing after "Symonds" yet, so it cannot be known to have ended.
        self.assertEqual(self.respeller.feed("Call Symonds"), "Call ")
        self.assertEqual(self.respeller.flush(), "Sigh-monds")

    def test_matching_is_whole_word_and_case_sensitive(self):
        out = self.respeller.feed("symonds and Symondson stay; ") + self.respeller.flush()
        self.assertEqual(out, "symonds and Symondson stay; ")

    def test_markup_passes_through_untouched(self):
        text = '<expr type="expression" label="joking"/> Symonds, really? [laughter]'
        out = self.respeller.feed(text) + self.respeller.flush()
        self.assertEqual(out, '<expr type="expression" label="joking"/> Sigh-monds, really? [laughter]')

    def test_empty_map_is_a_passthrough(self):
        plain = Respeller({})
        self.assertEqual(plain.feed("Hello there") + plain.flush(), "Hello there")
