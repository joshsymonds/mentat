"""Offline coverage for semantic reply-question builders."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.questions import build_questions


class QuestionBuilderTests(unittest.TestCase):
    def test_dispatcher_covers_each_supported_fixture_family(self):
        fixtures = {
            "action_ack": {"action": "timer", "expected": "five-minute timer"},
            "timer_duration": {"seconds": 300},
            "alarm_time": {"hour": 7, "minute": 0},
            "place_lookup": {"place_name": "Alice Keck Park Memorial Garden", "locality": "Santa Barbara"},
            "sms_say_back": {
                "recipient": "+1-202-555-0142",
                "body": "I will be there at 6:00.",
            },
            "sms_send_ack": {"recipient": "+1-202-555-0142"},
            "sms_confirmation": {"recipient": "+1-202-555-0142", "body": "I will be there at six."},
            "claims": {
                "required": ["the park is in Santa Barbara"],
                "rejected": ["Alice Keck bought the entire park"],
            },
            "alice_donor": {"donor": "Alice Keck Park", "action": "funded the city's purchase of the land"},
            "alice_father": {"person": "Alice Keck", "father": "William Myron Keck"},
            "alice_wealth": {"person": "Alice Keck", "source": "family wealth from Superior Oil"},
            "spanish_switch": {"turn": 1},
            "interpreter_turn": {"turn": 1},
            "story_start": {},
            "barge_in_answer": {
                "answer": "say Tokyo",
                "abandoned": "the story about a lighthouse keeper and her cat",
            },
        }
        for family, inputs in fixtures.items():
            with self.subTest(family=family):
                questions = build_questions(family, inputs)
                self.assertTrue(questions)
                self.assertTrue(all(isinstance(key, str) and isinstance(value, str) for key, value in questions.items()))
                self.assertTrue(all(question.endswith("?") for question in questions.values()))

    def test_timer_and_alarm_questions_preserve_expected_time_values(self):
        timer = build_questions("timer_duration", {"seconds": 300})["timer_duration"]
        self.assertIn("300 seconds", timer)
        self.assertIn("5 minutes", timer)
        alarm = build_questions("alarm_time", {"hour": 7, "minute": 0})["alarm_time"]
        self.assertIn("7:00 a.m.", alarm)
        self.assertIn("seven o'clock in the morning", alarm)

    def test_action_acknowledgement_and_place_lookup_name_expected_facts(self):
        alarm = build_questions("action_ack", {"action": "alarm", "expected": "7:00 a.m."})
        self.assertEqual(
            alarm["action_completed"],
            "Does the reply say that an alarm, not a timer or a reminder, was successfully set "
            "with this expected result: 7:00 a.m.?",
        )
        navigation = build_questions(
            "action_ack", {"action": "navigate", "expected": "Alice Keck Park Memorial Garden"},
        )
        self.assertEqual(
            navigation["action_completed"],
            "Does the reply say that directions to Alice Keck Park Memorial Garden were started, "
            "rather than only describing the place?",
        )
        timer = build_questions("action_ack", {"action": "timer", "expected": "a five-minute timer"})
        self.assertEqual(
            timer["action_completed"],
            "Does the reply say that a timer, not an alarm or a reminder, was successfully set or "
            "started with this expected result: a five-minute timer?",
        )
        with self.assertRaises(ValueError):
            build_questions("action_ack", {"action": "dance", "expected": "a dance"})

        place = build_questions("place_lookup", {"place_name": "Alice Keck Park Memorial Garden", "locality": "Santa Barbara"})
        self.assertIn("Alice Keck Park Memorial Garden", place["place_named"])
        self.assertIn("Santa Barbara", place["place_named"])

    def test_sms_questions_pin_recipient_body_time_equivalence_and_verified_inheritance(self):
        direct = build_questions(
            "sms_say_back",
            {"recipient": "+1-202-555-0142", "body": "I will be there at 6:00."},
        )
        self.assertEqual(set(direct), {"sms_recipient", "sms_body"})
        recipient = direct["sms_recipient"]
        self.assertIn("exactly 2 0 2 5 5 5 0 1 4 2 with or without a leading +1 or 1", recipient)
        self.assertIn("eight zero zero is 800", recipient)
        self.assertIn("words like the same number are not digits", recipient)
        self.assertNotIn("previously verified", recipient)
        body = direct["sms_body"]
        self.assertIn("I will be there at 6:00.", body)
        self.assertIn("six, 6, 6:00, six o'clock", body)
        self.assertIn("must be the same time", body)
        confirmation = build_questions(
            "sms_confirmation", {"recipient": "+1-202-555-0142", "body": "I will be there at six."},
        )
        self.assertEqual(set(confirmation), {"sms_recipient", "sms_body", "sms_confirmation"})
        self.assertIn("explicitly ask", confirmation["sms_confirmation"])
        self.assertIn("before sending", confirmation["sms_confirmation"])

        inherited = build_questions(
            "sms_say_back",
            {
                "recipient": "+1-202-555-0142",
                "body": "I will be there at seven.",
                "verified_recipient": "+1-202-555-0142",
            },
        )["sms_recipient"]
        self.assertIn("previously verified number, exactly 2 0 2 5 5 5 0 1 4 2", inherited)
        self.assertIn("spoken digit words count as digits", inherited)
        self.assertIn("calling it the same number", inherited)
        with self.assertRaises(ValueError):
            build_questions(
                "sms_say_back",
                {
                    "recipient": "+1-202-555-0142",
                    "body": "I will be there at seven.",
                    "verified_recipient": "+1-202-555-0199",
                },
            )

        sent = build_questions("sms_send_ack", {"recipient": "+1-202-555-0142"})
        self.assertIn("+1-202-555-0142", sent["sms_sent"])

    def test_generic_required_and_rejected_claims_have_satisfaction_polarity(self):
        questions = build_questions(
            "claims",
            {"required": ["the park is in Santa Barbara"], "rejected": ["Alice Keck bought the entire park"]},
        )
        self.assertIn("Does the reply accurately state", questions["required_1"])
        self.assertIn("the park is in Santa Barbara", questions["required_1"])
        self.assertIn("Does the reply avoid claiming this incorrect or prohibited fact", questions["rejected_1"])
        self.assertIn("Alice Keck bought the entire park", questions["rejected_1"])

    def test_alice_facts_are_explicit_expected_claims(self):
        donor = build_questions("alice_donor", {"donor": "Alice Keck Park", "action": "funded the city's purchase of the land"})
        father = build_questions("alice_father", {"person": "Alice Keck", "father": "William Myron Keck"})
        wealth = build_questions("alice_wealth", {"person": "Alice Keck", "source": "family wealth from Superior Oil"})
        self.assertIn("Alice Keck Park", donor["alice_donor"])
        self.assertIn("funded the city's purchase of the land", donor["alice_donor"])
        self.assertIn("without unresolved uncertainty", donor["alice_donor"])
        self.assertIn("William Myron Keck", father["alice_father"])
        self.assertIn("Alice Keck", father["alice_father"])
        self.assertIn("family wealth from Superior Oil", wealth["alice_wealth"])
        self.assertIn("Alice Keck", wealth["alice_wealth"])

    def test_all_spanish_switch_and_interpreter_turns_have_expected_semantics(self):
        switch_facts = {
            1: ("switch to responding in Spanish", "acknowledge the request"),
            2: ("Paris", "Spanish"),
            3: ("itself written in English", "acknowledge switching back to English"),
            4: ("switch to responding in Spanish again", "acknowledge the request"),
            5: ("blue", "Spanish"),
        }
        for turn, facts in switch_facts.items():
            with self.subTest(switch_turn=turn):
                question = build_questions("spanish_switch", {"turn": turn})["spanish_switch"]
                for fact in facts:
                    self.assertIn(fact, question)

        interpreter_facts = {
            1: ("ready", "Spanish"),
            2: ("soil", "too dry", "plant tomatoes"),
            3: ("water", "every morning", "before the sun", "strong"),
            4: ("timer", "five-minute", "water the plants"),
            5: ("gardener", "quoted speech", "rather than as an instruction to stop"),
            6: ("stop interpreting", "English"),
        }
        for turn, facts in interpreter_facts.items():
            with self.subTest(interpreter_turn=turn):
                question = build_questions("interpreter_turn", {"turn": turn})["interpreter_turn"]
                for fact in facts:
                    self.assertIn(fact, question)

    def test_barge_in_families_build_exact_questions(self):
        self.assertEqual(
            build_questions("story_start", {}),
            {"story_start": "Does the reply start telling a story?"},
        )
        self.assertEqual(
            build_questions(
                "barge_in_answer",
                {
                    "answer": "say Tokyo",
                    "abandoned": "the story about a lighthouse keeper and her cat",
                },
            ),
            {
                "barge_in_answer": "Does the reply say Tokyo?",
                "barge_in_dropped": (
                    "Does the reply stop telling the story about a lighthouse keeper and her "
                    "cat, counting as stopping even if it offers to return to the story later, "
                    "and not counting as stopping if it keeps telling the story?"
                ),
            },
        )

    def test_barge_in_families_reject_missing_or_blank_inputs(self):
        with self.assertRaises(ValueError):
            build_questions("story_start", {"subject": "a lighthouse keeper"})
        answer = "say Tokyo"
        abandoned = "the story about a lighthouse keeper and her cat"
        for inputs in ({}, {"answer": answer}, {"abandoned": abandoned}, {"answer": " ", "abandoned": abandoned}, {"answer": answer, "abandoned": ""}):
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                build_questions("barge_in_answer", inputs)

    def test_dispatcher_rejects_unknown_family_and_invalid_turn(self):
        with self.assertRaises(ValueError):
            build_questions("unknown", {})
        with self.assertRaises(ValueError):
            build_questions("spanish_switch", {"turn": 6})
        with self.assertRaises(ValueError):
            build_questions("interpreter_turn", {"turn": 7})


if __name__ == "__main__":
    unittest.main()
