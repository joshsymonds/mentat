"""Tests for the pure layer under the voice front: requests, consults, metrics.

The chat-context stand-ins here mirror livekit's ChatContext.items: message
items carry type/role/text_content, and function-call items carry neither role
nor text. Reading them structurally is what keeps this testable offline —
livekit is not importable without the flake's voice-env.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from request import (
    CONSULT_TURN_CHARS,
    CONSULT_WINDOW_TURNS,
    TURN_EFFORT,
    TURN_META,
    TURN_MODEL,
    VOICE_CARD_MARKER,
    PrivateContext,
    consult_envelope,
    conversation_advanced,
    count_user_messages,
    load_private_context,
    parse_private_context,
    recent_turns,
    split_persona,
    turn_latency,
    turn_request,
    with_private_context,
)


class Message:
    """Stand-in for livekit.agents.llm.ChatMessage."""

    type = "message"

    def __init__(self, role, text):
        self.role = role
        self.text_content = text


class FunctionCall:
    """Stand-in for a non-message chat item: no role, no text."""

    type = "function_call"


class RecentTurnsTest(unittest.TestCase):
    """The conversation window a consult carries with its question."""

    def test_returns_the_last_two_turns_in_order(self):
        # Oldest first: the window is read as conversation, and reversing it
        # would make the exchange read backwards to the consulted model.
        self.assertEqual(
            recent_turns(
                [
                    Message("user", "morning"),
                    Message("assistant", "morning yourself"),
                    Message("user", "what's on today?"),
                ]
            ),
            [("assistant", "morning yourself"), ("user", "what's on today?")],
        )

    def test_ignores_non_message_items(self):
        # The framework interleaves function-call and function-output items,
        # which carry neither role nor text; reading them must not raise.
        self.assertEqual(
            recent_turns([Message("user", "asked"), FunctionCall()]),
            [("user", "asked")],
        )

    def test_fewer_than_two_messages_returns_what_there_is(self):
        self.assertEqual(recent_turns([Message("user", "hi")]), [("user", "hi")])

    def test_empty_context_returns_nothing(self):
        self.assertEqual(recent_turns([]), [])

    def test_textless_messages_are_skipped(self):
        # text_content is None when a message holds no text part, and an empty
        # one contributes a bare "user:" line the consulted model would have
        # to interpret.
        self.assertEqual(
            recent_turns(
                [
                    Message("user", "real question"),
                    Message("assistant", None),
                    Message("assistant", ""),
                ]
            ),
            [("user", "real question")],
        )

    def test_window_size_is_adjustable(self):
        items = [
            Message("user", "one"),
            Message("assistant", "two"),
            Message("user", "three"),
        ]
        self.assertEqual(recent_turns(items, count=1), [("user", "three")])
        self.assertEqual(len(recent_turns(items, count=3)), 3)

    def test_default_window_is_the_pinned_size(self):
        items = [Message("user", str(i)) for i in range(5)]
        self.assertEqual(len(recent_turns(items)), CONSULT_WINDOW_TURNS)


class TurnRequestTest(unittest.TestCase):
    def test_body_matches_the_conversation_api(self):
        self.assertEqual(
            turn_request("kitchen", "what's on today?"),
            {
                "session_id": "voice-kitchen",
                "text": "what's on today?",
                "meta": {"surface": "voice", "user": "josh"},
                "effort": "low",
                "model": "sonnet",
            },
        )

    def test_session_id_namespaces_the_room(self):
        # A daemon shared with other surfaces must never collide with a room.
        self.assertEqual(turn_request("office", "hi")["session_id"], "voice-office")

    def test_voice_turns_are_low_effort_and_identified(self):
        # Authority is per-turn in mentat policy: the surface and user ride
        # along with every request. The model rides along too: the daemon's
        # default is the deepest model, whose latency and usage limits don't
        # suit a caller waiting in silence (first live test hit both).
        self.assertEqual(TURN_EFFORT, "low")
        self.assertEqual(TURN_META, {"surface": "voice", "user": "josh"})
        self.assertEqual(TURN_MODEL, "sonnet")

    def test_effort_and_model_are_passed_through(self):
        # A consult escalates both knobs; the tool call carries them, so the
        # request builder takes them as arguments rather than reading globals.
        body = turn_request("kitchen", "why?", effort="high", model="fable")
        self.assertEqual(body["effort"], "high")
        self.assertEqual(body["model"], "fable")

    def test_unknown_effort_and_model_are_not_validated_here(self):
        # This layer stays dumb: the daemon owns the vocabulary, and rejecting
        # a value here would only turn a clear daemon error into a local one.
        body = turn_request("kitchen", "why?", effort="medium", model="opus")
        self.assertEqual(body["effort"], "medium")
        self.assertEqual(body["model"], "opus")


class SplitPersonaTest(unittest.TestCase):
    """persona.md's two halves: the front's instructions and its voice card."""

    def test_splits_on_the_marker_and_strips_both_halves(self):
        text = f"# The voice\n\nBe warm.\n\n{VOICE_CARD_MARKER}\n\nSound warm.\n"
        self.assertEqual(
            split_persona(text), ("# The voice\n\nBe warm.", "Sound warm.")
        )

    def test_a_missing_marker_is_an_error(self):
        # The half that only fires at deploy: a persona edited past its marker
        # would otherwise hand mentatd the whole file as a voice card.
        with self.assertRaises(ValueError):
            split_persona("# The voice\n\nBe warm.\n")

    def test_the_marker_is_pinned(self):
        # persona.md is hand-written around this literal; changing one without
        # the other splits the file in the wrong place, or not at all.
        self.assertEqual(VOICE_CARD_MARKER, "---VOICE-CARD---")


class ConsultEnvelopeTest(unittest.TestCase):
    PERSONA = "You are Luna, dry and quick."
    QUESTION = "What did the doctor say about the results?"

    def test_framing_sentence_comes_first_verbatim(self):
        # The reply is spoken as-is in the front's voice, so the envelope's
        # first instruction is the one that governs style.
        envelope = consult_envelope(self.PERSONA, "", [], self.QUESTION)
        self.assertTrue(
            envelope.startswith(
                "Your reply will be read aloud verbatim to the user as a "
                "continuation of this conversation — match this voice and style."
            ),
            envelope,
        )

    def test_sections_appear_in_order(self):
        envelope = consult_envelope(
            self.PERSONA,
            "Josh is planning a trip.",
            [("user", "where should I go?"), ("assistant", "somewhere warm")],
            self.QUESTION,
        )
        positions = [
            envelope.index("read aloud verbatim"),
            envelope.index(self.PERSONA),
            envelope.index("Josh is planning a trip."),
            envelope.index("user: where should I go?"),
            envelope.index(self.QUESTION),
        ]
        self.assertEqual(positions, sorted(positions), envelope)

    def test_turns_render_one_per_line_as_role_text(self):
        envelope = consult_envelope(
            self.PERSONA,
            "",
            [("user", "first"), ("assistant", "second"), ("user", "third")],
            self.QUESTION,
        )
        lines = envelope.splitlines()
        self.assertIn("user: first", lines)
        self.assertIn("assistant: second", lines)
        self.assertIn("user: third", lines)

    def test_summary_section_is_omitted_when_empty(self):
        with_summary = consult_envelope(
            self.PERSONA, "Josh is planning a trip.", [], self.QUESTION
        )
        without = consult_envelope(self.PERSONA, "", [], self.QUESTION)
        blank = consult_envelope(self.PERSONA, "   \n ", [], self.QUESTION)
        self.assertIn("Josh is planning a trip.", with_summary)
        # The label goes with the section: an empty heading is noise the
        # consulted model would have to interpret.
        self.assertIn("Conversation so far:", with_summary)
        self.assertNotIn("Conversation so far:", without)
        self.assertEqual(without, blank)

    def test_empty_last_turns_leaves_no_blank_hole(self):
        envelope = consult_envelope(self.PERSONA, "A summary.", [], self.QUESTION)
        self.assertIn(self.QUESTION, envelope)
        self.assertNotIn("\n\n\n", envelope)

    def test_long_turns_are_truncated_at_the_cap(self):
        envelope = consult_envelope(
            self.PERSONA, "", [("user", "x" * (CONSULT_TURN_CHARS + 50))], "q?"
        )
        self.assertIn("user: " + "x" * CONSULT_TURN_CHARS + "…", envelope)
        self.assertNotIn("x" * (CONSULT_TURN_CHARS + 1), envelope)

    def test_turns_at_the_cap_are_left_alone(self):
        envelope = consult_envelope(
            self.PERSONA, "", [("user", "x" * CONSULT_TURN_CHARS)], "q?"
        )
        self.assertIn("user: " + "x" * CONSULT_TURN_CHARS + "\n", envelope + "\n")
        self.assertNotIn("…", envelope)

    def test_question_carries_a_label(self):
        envelope = consult_envelope(self.PERSONA, "", [], self.QUESTION)
        self.assertIn("Question:\n" + self.QUESTION, envelope)

    def test_the_cap_is_pinned(self):
        # Bounded on purpose: mentatd's session already remembers the prior
        # consults, so a growing window would re-feed it its own history.
        self.assertEqual(CONSULT_TURN_CHARS, 500)


class ReorientationTest(unittest.TestCase):
    def test_counts_only_user_messages(self):
        self.assertEqual(
            count_user_messages(
                [
                    Message("user", "one"),
                    Message("assistant", "reply"),
                    FunctionCall(),
                    Message("user", "two"),
                ]
            ),
            2,
        )

    def test_counts_nothing_in_an_empty_context(self):
        self.assertEqual(count_user_messages([]), 0)

    def test_tool_and_assistant_items_do_not_advance_the_conversation(self):
        # The framework appends the function call and the front's own spoken
        # front-matter during the consult, so raw length would always grow.
        baseline_items = [Message("user", "what did the doctor say?")]
        at_dispatch = count_user_messages(baseline_items)
        during = [
            *baseline_items,
            FunctionCall(),
            Message("assistant", "let me check on that"),
        ]
        self.assertFalse(conversation_advanced(during, at_dispatch))

    def test_a_new_user_message_advances_the_conversation(self):
        baseline_items = [Message("user", "what did the doctor say?")]
        at_dispatch = count_user_messages(baseline_items)
        during = [
            *baseline_items,
            FunctionCall(),
            Message("assistant", "let me check on that"),
            Message("user", "actually, what time is it?"),
        ]
        self.assertTrue(conversation_advanced(during, at_dispatch))

    def test_an_unchanged_context_has_not_advanced(self):
        items = [Message("user", "asked"), Message("assistant", "answered")]
        self.assertFalse(conversation_advanced(items, count_user_messages(items)))


class TurnLatencyTest(unittest.TestCase):
    """One chat item's effect on the journal: what to log, what to hold.

    The metric shapes below are the ones livekit-agents 1.6.10 actually
    stamps: the caller's half on the user message, and two different assistant
    halves depending on which path produced the speech.
    """

    #: The caller's turn, measured by the pipeline as it endpoints and
    #: transcribes.
    USER = {
        "end_of_turn_delay": 0.14,
        "transcription_delay": 0.22,
        "stopped_speaking_at": 1000.0,
    }
    #: A consulted answer, shaped as session.say stamps it (agent_activity
    #: _tts_task_impl): real speech and playback numbers, and no llm_node_ttft,
    #: because no llm ran to produce it.
    SAY = {
        "tts_node_ttfb": 0.31,
        "started_speaking_at": 1002.0,
        "stopped_speaking_at": 1004.0,
        "playback_latency": 0.09,
    }
    #: A reply the pipeline generated itself — the only path that measures
    #: time to first token.
    REPLY = {
        "llm_node_ttft": 0.85,
        "tts_node_ttfb": 0.30,
        "started_speaking_at": 1005.0,
        "stopped_speaking_at": 1006.0,
        "e2e_latency": 1.41,
    }

    def test_the_callers_half_is_held_rather_than_logged(self):
        # Half a turn's numbers are not a turn: they wait for the reply that
        # completes them.
        line, pending = turn_latency({}, "user", self.USER)
        self.assertIsNone(line)
        self.assertEqual(pending, self.USER)

    def test_spoken_for_speech_logs_nothing_and_keeps_the_held_half(self):
        # A consulted answer goes out through session.say, whose item is not
        # metric-free — which is why emptiness cannot be the test. The held
        # half belongs to the reply that actually answers the turn.
        line, pending = turn_latency(self.USER, "assistant", self.SAY)
        self.assertIsNone(line)
        self.assertEqual(pending, self.USER)

    def test_a_pipeline_reply_joins_both_halves_into_one_line(self):
        line, _ = turn_latency(self.USER, "assistant", self.REPLY)
        self.assertEqual(
            line,
            "endpoint=0.140s transcript=0.220s llm_ttft=0.850s "
            "tts_ttfb=0.300s e2e=1.410s",
        )

    def test_a_consult_does_not_eat_the_next_replys_user_half(self):
        # The sequence a consult really produces: the question, the answer
        # spoken on its behalf, then the front's own next pipeline reply.
        _, pending = turn_latency({}, "user", self.USER)
        _, pending = turn_latency(pending, "assistant", self.SAY)
        line, _ = turn_latency(pending, "assistant", self.REPLY)
        self.assertIn("endpoint=0.140s", line)
        self.assertIn("transcript=0.220s", line)

    def test_the_held_half_is_dropped_once_it_has_been_logged(self):
        # Otherwise the next turn inherits the last caller's numbers and every
        # line after the first reads as a turn that never happened.
        _, pending = turn_latency(self.USER, "assistant", self.REPLY)
        self.assertEqual(pending, {})

    def test_stages_that_went_unmeasured_are_marked_not_omitted(self):
        # A reply with no user half still logs: the fields are positional in
        # the line, and a silently missing one would shift the reader's eye.
        line, _ = turn_latency({}, "assistant", {"llm_node_ttft": 0.5})
        self.assertEqual(
            line, "endpoint=? transcript=? llm_ttft=0.500s tts_ttfb=? e2e=?"
        )

    def test_only_an_assistant_message_can_close_a_turn(self):
        # ChatMessage.role is also "system" and "developer". Reply metrics are
        # read off the assistant message and nowhere else, so even an item
        # carrying them under another role neither logs nor spends the held
        # half — it is the role, not just the key, that says a turn ended.
        line, pending = turn_latency(self.USER, "system", self.REPLY)
        self.assertIsNone(line)
        self.assertEqual(pending, self.USER)


if __name__ == "__main__":
    unittest.main()


class PrivateContextTest(unittest.TestCase):
    def test_full_document_parses(self):
        ctx = parse_private_context(
            'about = """\nWho he is: a person.\n"""\n'
            'keyterms = ["Symonds", "Rosalind"]\n'
            "[pronunciations]\n"
            'Symonds = "Sigh-monds"\n'
        )
        self.assertEqual(ctx.about, "Who he is: a person.")
        self.assertEqual(ctx.keyterms, ("Symonds", "Rosalind"))
        self.assertEqual(ctx.pronunciations, {"Symonds": "Sigh-monds"})

    def test_missing_sections_default_to_empty(self):
        ctx = parse_private_context('keyterms = ["Rosalind"]\n')
        self.assertEqual(ctx.about, "")
        self.assertEqual(ctx.pronunciations, {})

    def test_unknown_keys_and_wrong_types_are_refused(self):
        # A typo must not silently become a file that carries nothing.
        with self.assertRaises(ValueError):
            parse_private_context('abuot = "x"\n')
        with self.assertRaises(ValueError):
            parse_private_context('keyterms = "Rosalind"\n')
        with self.assertRaises(ValueError):
            parse_private_context("[pronunciations]\nSymonds = 3\n")

    def test_no_path_means_an_empty_context(self):
        self.assertEqual(load_private_context(None), PrivateContext())
        self.assertEqual(load_private_context(""), PrivateContext())

    def test_about_is_folded_into_the_instructions(self):
        base = "# The voice\n\nBe warm."
        self.assertEqual(with_private_context(base, PrivateContext()), base)
        self.assertEqual(
            with_private_context(base, PrivateContext(about="Who he is: Josh.")),
            "# The voice\n\nBe warm.\n\nWho he is: Josh.",
        )
