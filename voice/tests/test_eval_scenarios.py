"""Offline acceptance checks for the scripted voice scenario evaluator."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from scenarios import SCENARIOS, evaluate_scenario


class ScenarioCorpusTests(unittest.TestCase):
    def test_corpus_covers_required_scenarios_and_exact_alice_prompts(self):
        by_name = {scenario.name: scenario for scenario in SCENARIOS}
        self.assertEqual(len(by_name), len(SCENARIOS), "duplicate scenario names")
        self.assertEqual(
            set(by_name),
            {
                "timer-300-seconds",
                "equivalent-alarm",
                "place-search-navigation",
                "sms-say-back-yes",
                "sms-correction-new-yes",
                "alice-keck-context-chain",
            },
        )
        for scenario in SCENARIOS:
            self.assertEqual(
                len(scenario.caller_lines),
                len(scenario.turns),
                f"{scenario.name} must define one caller line per assistant turn",
            )
        alice = by_name["alice-keck-context-chain"]
        self.assertEqual(
            alice.caller_lines,
            (
                "Why is this garden named for Alice Keck?",
                "Okay, who was she?",
                "Okay, what was the source of her wealth?",
            ),
        )
        place = by_name["place-search-navigation"]
        self.assertEqual(place.place_query, "Alice Keck Park Memorial Garden")
        self.assertTrue(place.selected_place_pattern)
        sms_number = "+1-202-555-0142"
        for name in ("sms-say-back-yes", "sms-correction-new-yes"):
            sms = by_name[name]
            self.assertIn(sms_number, sms.caller_lines[0])
            self.assertEqual(sms.commands[0]["to"], sms_number)

    def test_accepts_equivalent_spoken_sms_and_sends_exact_dictated_text(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        evaluate_scenario(
            scenario,
            turns=[
                "I can text +1-202-555-0142: I'll be there at 6! Should I send it?",
                "Sent that message.",
            ],
            phone_commands=[
                {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
            ],
            room_closed_after=2,
        )

    def test_accepts_naturally_formatted_spoken_phone_number(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        evaluate_scenario(
            scenario,
            turns=[
                "I can text +1 (202) 555-0142: I'll be there at 6. Should I send it?",
                "Sent that message.",
            ],
            phone_commands=[
                {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
            ],
            room_closed_after=2,
        )

    def test_rejects_different_spoken_phone_number(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                turns=[
                    "I can text +1 (202) 555-0199: I'll be there at 6. Should I send it?",
                    "Sent that message.",
                ],
                phone_commands=[
                    {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
                ],
                room_closed_after=2,
            )

    def test_accepts_correction_after_new_yes_and_keeps_exact_sent_body(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        evaluate_scenario(
            scenario,
            turns=[
                "I can text +1-202-555-0142: I'll be there at 6! Should I send it?",
                "I can text +1-202-555-0142: I will be there at 7. Would you like me to send it?",
                "Sent the corrected message.",
            ],
            phone_commands=[
                {"turn": 3, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at seven."}
            ],
            room_closed_after=3,
        )

    def test_accepts_correct_sms_say_back_and_single_authorized_send(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        evaluate_scenario(
            scenario,
            turns=[
                "I can text +1-202-555-0142: I will be there at six. Should I send it?",
                "Sent Alice that message.",
            ],
            phone_commands=[
                {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
            ],
            room_closed_after=2,
        )

    def test_accepts_correction_only_after_new_yes_and_sends_corrected_text_once(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        evaluate_scenario(
            scenario,
            turns=[
                "I can text +1-202-555-0142: I will be there at six. Should I send it?",
                "I can text +1-202-555-0142: I will be there at seven. Should I send it?",
                "Sent Alice the corrected message.",
            ],
            phone_commands=[
                {"turn": 3, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at seven."}
            ],
            room_closed_after=3,
        )

    def test_five_minute_timer_maps_to_300_seconds(self):
        scenario = next(s for s in SCENARIOS if s.name == "timer-300-seconds")
        self.assertEqual(scenario.caller_lines, ("Set a timer for five minutes.",))
        for answer in (
            "A five-minute timer is set.",
            "Timer set for 5 minutes.",
            "I set a timer for five minutes.",
        ):
            with self.subTest(answer=answer):
                evaluate_scenario(
                    scenario,
                    turns=[answer],
                    phone_commands=[{"turn": 1, "kind": "timer", "seconds": 300}],
                    room_closed_after=1,
                )
        with self.assertRaisesRegex(AssertionError, "missing answer pattern"):
            evaluate_scenario(
                scenario,
                turns=["Timer set for 10 minutes."],
                phone_commands=[{"turn": 1, "kind": "timer", "seconds": 300}],
                room_closed_after=1,
            )

    def test_accepts_timer_and_equivalent_alarm_commands(self):
        for name, answer, command in (
            ("timer-300-seconds", "A five-minute timer is set.", {"turn": 1, "kind": "timer", "seconds": 300}),
            ("equivalent-alarm", "Your 7:00 alarm is set.", {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0}),
        ):
            with self.subTest(name=name):
                evaluate_scenario(
                    next(s for s in SCENARIOS if s.name == name),
                    turns=[answer],
                    phone_commands=[command],
                    room_closed_after=1,
                )

    def test_rejects_changed_spoken_time_even_when_sent_payload_matches(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        with self.assertRaisesRegex(AssertionError, "said back"):
            evaluate_scenario(
                scenario,
                turns=[
                    "I can text +1-202-555-0142: I'll be there at 7. Should I send it?",
                    "Sent that message.",
                ],
                phone_commands=[
                    {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
                ],
                room_closed_after=2,
            )

    def test_rejects_wrong_sms_recipient(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        with self.assertRaisesRegex(AssertionError, "expected to="):
            evaluate_scenario(
                scenario,
                turns=[
                    "I can text +1-202-555-0142: I will be there at six. Should I send it?",
                    "Sent that message.",
                ],
                phone_commands=[
                    {"turn": 2, "kind": "sms", "to": "+1-202-555-0199", "body": "I will be there at six."}
                ],
                room_closed_after=2,
            )

    def test_rejects_sms_sent_before_later_turn_yes(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                turns=[
                    "I can text +1-202-555-0142: I will be there at six. Should I send it?",
                    "Sent Alice that message.",
                ],
                phone_commands=[
                    {"turn": 1, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
                ],
                room_closed_after=2,
            )

    def test_rejects_contradictory_sms_sayback_even_if_expected_body_is_present(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        with self.assertRaisesRegex(AssertionError, "contradicts"):
            evaluate_scenario(
                scenario,
                turns=[
                    "I can text +1-202-555-0142: I will be there at six. Should I send it? Actually, I will be there at seven.",
                    "Sent Alice that message.",
                ],
                phone_commands=[
                    {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
                ],
                room_closed_after=2,
            )

    def test_rejects_wrong_sms_payload_without_duplicate_command(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        with self.assertRaisesRegex(AssertionError, "command 1 expected body='I will be there at seven.'"):
            evaluate_scenario(
                scenario,
                turns=[
                    "I can text +1-202-555-0142: I will be there at six. Should I send it?",
                    "I can text +1-202-555-0142: I will be there at seven. Should I send it?",
                    "Sent Alice the corrected message.",
                ],
                phone_commands=[
                    {"turn": 3, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
                ],
                room_closed_after=3,
            )

    def test_rejects_wrong_navigation_without_other_mismatches(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        with self.assertRaisesRegex(AssertionError, "navigation selected an unexpected place name"):
            evaluate_scenario(
                scenario,
                turns=[
                    "Alice Keck Park Memorial Garden is nearby in Santa Barbara. Would you like directions?",
                    "Navigating to Alice Keck Park Memorial Garden in Santa Barbara now.",
                ],
                phone_commands=[
                    {"turn": 1, "kind": "location"},
                    {
                        "turn": 2,
                        "kind": "navigate",
                        "name": "Another Garden",
                        "address": "10 Garden Street",
                        "place_id": "live-result-id",
                        "lat": 34.42,
                        "lng": -119.70,
                    }
                ],
                room_closed_after=2,
            )

    def test_rejects_missing_answer_and_early_close(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                turns=["Alice Keck Park Memorial Garden is nearby. Would you like directions?", ""],
                phone_commands=[
                    {"turn": 1, "kind": "location"},
                    {
                        "turn": 2,
                        "kind": "navigate",
                        "name": "Alice Keck Park Memorial Garden",
                        "address": "1 Garden Road",
                        "place_id": "live-result-id",
                        "lat": 34.42,
                        "lng": -119.70,
                    }
                ],
                room_closed_after=1,
            )

    def test_rejects_missing_hangup_and_hangup_when_chain_must_stay_open(self):
        timer = next(s for s in SCENARIOS if s.name == "timer-300-seconds")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                timer,
                turns=["A 300-second timer is set."],
                phone_commands=[{"turn": 1, "kind": "timer", "seconds": 300}],
                room_closed_after=None,
            )
        chain = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                chain,
                turns=[
                    "Alice Keck donated Alice Keck Park Memorial Garden in Santa Barbara.",
                    "Alice Keck was the daughter of W. M. Keck, founder of Superior Oil.",
                    "She inherited wealth from her father's Superior Oil fortune.",
                ],
                phone_commands=[],
                room_closed_after=3,
            )

    def test_rejects_echo_only_alice_answers(self):
        chain = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                chain,
                turns=[
                    "Why is this garden named for Alice Keck?",
                    "Alice Keck was the daughter of W. M. Keck and an American philanthropist.",
                    "Okay, what was the source of her wealth?",
                ],
                phone_commands=[],
                room_closed_after=None,
            )

    def test_rejects_uncertain_repetition_as_alice_keck_answers(self):
        chain = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        cases = (
            [
                "Alice Keck Park Memorial Garden is in Santa Barbara. I'm not sure whether Alice Keck donated it or gave it.",
                "Alice Keck was the daughter of W. M. Keck, founder of Superior Oil.",
                "Her wealth came from her father's Superior Oil fortune, which she inherited.",
            ],
            [
                "Alice Keck donated Alice Keck Park Memorial Garden in Santa Barbara.",
                "Alice Keck was the daughter of W. M. Keck, founder of Superior Oil.",
                "I can't say; she may have inherited her wealth from the W. M. Keck family's Superior Oil fortune.",
            ],
        )
        for turns in cases:
            with self.subTest(turns=turns), self.assertRaisesRegex(AssertionError, "non-answer|uncertain"):
                evaluate_scenario(chain, turns=turns, phone_commands=[], room_closed_after=None)

    def test_accepts_dynamic_place_selection_and_correct_alice_fact_chain(self):
        place = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        evaluate_scenario(
            place,
            turns=[
                "Alice Keck Park Memorial Garden is a nearby Santa Barbara result. Which result should I navigate to?",
                "Navigating to Alice Keck Park Memorial Garden now.",
            ],
            phone_commands=[
                {"turn": 1, "kind": "location"},
                {
                    "turn": 2,
                    "kind": "navigate",
                    "name": "Alice Keck Park Memorial Garden",
                    "address": "1 Garden Road",
                    "place_id": "live-result-id",
                    "lat": 34.42,
                    "lng": -119.70,
                }
            ],
            room_closed_after=2,
        )
        chain = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        evaluate_scenario(
            chain,
            turns=[
                "Alice Keck donated Alice Keck Park Memorial Garden in Santa Barbara.",
                "Alice Keck was the daughter of W. M. Keck, founder of Superior Oil.",
                "Her wealth came from her father's Superior Oil fortune, which she inherited.",
            ],
            phone_commands=[],
            room_closed_after=None,
        )

    def test_place_search_requires_location_before_navigation(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        turns = [
            "Alice Keck Park Memorial Garden is a nearby Santa Barbara result. Which result should I navigate to?",
            "Navigating to Alice Keck Park Memorial Garden now.",
        ]
        navigate = {
            "turn": 2,
            "kind": "navigate",
            "name": "Alice Keck Park Memorial Garden",
            "address": "1 Garden Road",
            "place_id": "live-result-id",
            "lat": 34.42,
            "lng": -119.70,
        }
        evaluate_scenario(
            scenario,
            turns=turns,
            phone_commands=[{"turn": 1, "kind": "location"}, navigate],
            room_closed_after=2,
        )
        invalid_sequences = (
            (navigate,),
            (navigate, {"turn": 1, "kind": "location"}),
            ({"turn": 1, "kind": "location"}, {"turn": 1, "kind": "location"}, navigate),
            ({"turn": 2, "kind": "location"}, navigate),
        )
        for commands in invalid_sequences:
            with self.subTest(commands=commands), self.assertRaises(AssertionError):
                evaluate_scenario(
                    scenario,
                    turns=turns,
                    phone_commands=list(commands),
                    room_closed_after=2,
                )


if __name__ == "__main__":
    unittest.main()
