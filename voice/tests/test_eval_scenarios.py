"""Offline acceptance checks for the scripted voice scenario evaluator."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from scenarios import SCENARIOS, evaluate_scenario, evaluate_scenario_failures, evaluate_scenario_prefix


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

    def test_complete_evaluator_returns_all_structured_product_failures(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")

        failures = evaluate_scenario_failures(
            scenario,
            turns=["Wrong park in another city.", "Taking you to Wrong Park."],
            phone_commands=[
                {"turn": 1, "kind": "location"},
                {
                    "turn": 2,
                    "kind": "navigate",
                    "name": "Wrong Park",
                    "address": "1 Main St",
                    "place_id": "wrong-place-id",
                    "lat": 34.4,
                    "lng": -119.7,
                },
            ],
            room_closed_after=2,
        )

        self.assertTrue(failures)
        self.assertTrue(all(isinstance(failure.turn, int) for failure in failures))
        self.assertTrue(any(failure.turn == 1 and "missing answer pattern" in failure.message for failure in failures))
        self.assertTrue(any(failure.turn == 2 and "unexpected place name" in failure.message for failure in failures))

    def test_prefix_evaluator_accepts_valid_actions_before_failed_turn(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")

        failures = evaluate_scenario_prefix(
            scenario,
            turns=["Alice Keck Park Memorial Garden is in Santa Barbara."],
            phone_commands=[{"turn": 1, "kind": "location"}],
            room_closed_after=None,
            failed_turn=2,
        )

        self.assertEqual(failures, [])

    def test_prefix_evaluator_attributes_failed_turn_phone_action_errors(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        turns = ["Alice Keck Park Memorial Garden is in Santa Barbara."]
        phone_commands = [
            {"turn": 1, "kind": "location"},
            {"turn": 2, "kind": "navigate", "name": "Wrong Park"},
        ]

        failures = evaluate_scenario_prefix(
            scenario,
            turns=turns,
            phone_commands=phone_commands,
            room_closed_after=None,
            failed_turn=2,
        )

        self.assertTrue(failures)
        self.assertTrue(all(failure.turn == 2 for failure in failures))
        self.assertTrue(any("unexpected place name" in failure.message for failure in failures))

    def test_prefix_evaluator_checks_failed_turn_navigation_even_without_location_command(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        failures = evaluate_scenario_prefix(
            scenario,
            turns=["Alice Keck Park Memorial Garden is in Santa Barbara."],
            phone_commands=[{"turn": 2, "kind": "navigate", "name": "Wrong Park"}],
            room_closed_after=None,
            failed_turn=2,
        )

        self.assertTrue(failures)
        self.assertTrue(any(failure.turn == 1 and "phone commands" in failure.message for failure in failures))
        self.assertTrue(any(failure.turn == 2 and "unexpected place name" in failure.message for failure in failures))

    def test_prefix_evaluator_reports_missing_sms_readback_on_completed_turn(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")

        failures = evaluate_scenario_prefix(
            scenario,
            turns=["Would you like me to send that message?"],
            phone_commands=[],
            room_closed_after=None,
        )

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].turn, 1)
        self.assertIn("no complete +1-202-555-0142 message say-back", failures[0].message)

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

    def test_accepts_normalized_sms_command_in_complete_and_prefix_evaluators(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        turns = [
            "I'll text +1-202-555-0142: I'll be there at six. Should I send it?",
            "Sent that message.",
        ]
        phone_commands = [
            {
                "turn": 2,
                "kind": "sms",
                "to": "+12025550142",
                "body": "I WILL be there at 6!",
            }
        ]

        evaluate_scenario(
            scenario,
            turns=turns,
            phone_commands=phone_commands,
            room_closed_after=2,
        )
        self.assertEqual(
            evaluate_scenario_prefix(
                scenario,
                turns=turns,
                phone_commands=phone_commands,
                room_closed_after=None,
            ),
            [],
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

    def test_rejects_correction_send_without_new_consent_prompt(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        with self.assertRaisesRegex(AssertionError, "missing answer pattern"):
            evaluate_scenario(
                scenario,
                turns=[
                    "I can text +1-202-555-0142: I will be there at six. Should I send it?",
                    "I can text +1-202-555-0142: I will be there at seven. I'll send it now.",
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

    def test_rejects_wrong_timer_and_alarm_facts_or_fake_commands(self):
        cases = (
            (
                "timer-300-seconds",
                "Done, 5 minutes on the clock.",
                {"turn": 1, "kind": "timer", "seconds": 60},
            ),
            (
                "equivalent-alarm",
                "Your 8am alarm is set.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "equivalent-alarm",
                "Your 7am alarm is set.",
                {"turn": 1, "kind": "alarm", "hour": 8, "minute": 0},
            ),
        )
        for name, answer, command in cases:
            with self.subTest(name=name, answer=answer, command=command), self.assertRaises(AssertionError):
                evaluate_scenario(
                    next(s for s in SCENARIOS if s.name == name),
                    turns=[answer],
                    phone_commands=[command],
                    room_closed_after=1,
                )

    def test_rejects_pm_alarm_even_when_hour_and_command_match(self):
        scenario = next(s for s in SCENARIOS if s.name == "equivalent-alarm")
        for answer in (
            "Your 7:00 p.m. alarm is set.",
            "Your 7 a.m. alarm is set — sorry, I meant 7 p.m.",
        ):
            with self.subTest(answer=answer), self.assertRaises(AssertionError):
                evaluate_scenario(
                    scenario,
                    turns=[answer],
                    phone_commands=[{"turn": 1, "kind": "alarm", "hour": 7, "minute": 0}],
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

    def test_rejects_duplicate_sms_send_even_when_both_match_readback(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        command = {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
        with self.assertRaisesRegex(AssertionError, "expected 1 phone commands, got 2"):
            evaluate_scenario(
                scenario,
                turns=[
                    "I can text +1-202-555-0142: I will be there at six. Should I send it?",
                    "Sent that message.",
                ],
                phone_commands=[command, command],
                room_closed_after=2,
            )

    def test_accepts_observed_halis_keck_asr_variation_in_complete_and_prefix_paths(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        search_result = "Halis Keck Park Memorial Gardens. Right there in Santa Barbara."
        navigate = {
            "turn": 2,
            "kind": "navigate",
            "name": "Alice Keck Park Memorial Garden",
            "address": "1 Garden Road",
            "place_id": "alice-keck-place-id",
            "lat": 34.42,
            "lng": -119.70,
        }
        evaluate_scenario(
            scenario,
            turns=[search_result, "Navigating to Alice Keck Park Memorial Garden now."],
            phone_commands=[{"turn": 1, "kind": "location"}, navigate],
            room_closed_after=2,
        )
        self.assertEqual(
            evaluate_scenario_prefix(
                scenario,
                turns=[search_result],
                phone_commands=[{"turn": 1, "kind": "location"}, navigate],
                room_closed_after=None,
                failed_turn=2,
            ),
            [],
        )

    def test_rejects_absent_or_unobserved_keck_name_and_wrong_navigation_target(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        commands = [
            {"turn": 1, "kind": "location"},
            {
                "turn": 2,
                "kind": "navigate",
                "name": "Alice Keck Park Memorial Garden",
                "address": "1 Garden Road",
                "place_id": "alice-keck-place-id",
                "lat": 34.42,
                "lng": -119.70,
            },
        ]
        invalid_search_results = (
            "Santa Barbara Street is the result.",
            "Alice Park Memorial Garden. Right there in Santa Barbara.",
            "Halis Keck Park by Santa Barbara.",
        )
        for search_result in invalid_search_results:
            with self.subTest(search_result=search_result), self.assertRaises(AssertionError):
                evaluate_scenario(
                    scenario,
                    turns=[search_result, "Navigating to Alice Keck Park Memorial Garden now."],
                    phone_commands=commands,
                    room_closed_after=2,
                )
            self.assertTrue(
                evaluate_scenario_prefix(
                    scenario,
                    turns=[search_result],
                    phone_commands=[commands[0]],
                    room_closed_after=None,
                    failed_turn=2,
                ),
                search_result,
            )

        wrong_target = [
            commands[0],
            {**commands[1], "name": "Another Santa Barbara Garden"},
        ]
        with self.assertRaisesRegex(AssertionError, "unexpected place name"):
            evaluate_scenario(
                scenario,
                turns=[
                    "Alice Keck Park Memorial Garden is right there in Santa Barbara.",
                    "Navigating to Alice Keck Park Memorial Garden now.",
                ],
                phone_commands=wrong_target,
                room_closed_after=2,
            )

    def test_keck_asr_tolerance_keeps_navigation_evidence_and_room_close_strict(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        turns = [
            "Halis Keck Park Memorial Gardens. Right there in Santa Barbara.",
            "Navigating to Alice Keck Park Memorial Garden now.",
        ]
        navigate = {
            "turn": 2,
            "kind": "navigate",
            "name": "Alice Keck Park Memorial Garden",
            "address": "1 Garden Road",
            "place_id": "alice-keck-place-id",
            "lat": 34.42,
            "lng": -119.70,
        }
        invalid_commands = (
            ({**navigate, "name": "Alice Keck Park Garden"}, "unexpected place name"),
            ({**navigate, "address": " "}, "no result address"),
            ({**navigate, "place_id": ""}, "no returned place id"),
            ({**navigate, "lat": float("nan")}, "invalid coordinates"),
            ({**navigate, "lng": True}, "invalid coordinates"),
        )
        for invalid_navigate, failure in invalid_commands:
            with self.subTest(command=invalid_navigate), self.assertRaisesRegex(AssertionError, failure):
                evaluate_scenario(
                    scenario,
                    turns=turns,
                    phone_commands=[{"turn": 1, "kind": "location"}, invalid_navigate],
                    room_closed_after=2,
                )

        with self.assertRaisesRegex(AssertionError, "expected room close after turn 2"):
            evaluate_scenario(
                scenario,
                turns=turns,
                phone_commands=[{"turn": 1, "kind": "location"}, navigate],
                room_closed_after=1,
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
        with self.assertRaisesRegex(AssertionError, "expected room close"):
            evaluate_scenario(
                timer,
                turns=["A five-minute timer is set."],
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

    def test_accepts_eval7_timer_and_alarm_transcripts(self):
        cases = (
            (
                "timer-300-seconds",
                "Sure, 5 minutes starting now. Done, 5 minutes on the clock.",
                {"turn": 1, "kind": "timer", "seconds": 300},
            ),
            (
                "equivalent-alarm",
                "On it. Done. Your 7am alarm is set.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
        )
        for name, answer, command in cases:
            with self.subTest(name=name):
                evaluate_scenario(
                    next(s for s in SCENARIOS if s.name == name),
                    turns=[answer],
                    phone_commands=[command],
                    room_closed_after=1,
                )

    def test_accepts_eval7_sms_readback_with_punctuation_free_number_and_body(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        readbacks = (
            "On it. Just to confirm, I'll text 202-555-0142. I will be there at 6. Want me to send it?",
            "Just to confirm, I'll text 202 555 0142 I will be there at 6 want me to send it?",
        )
        for readback in readbacks:
            with self.subTest(readback=readback):
                evaluate_scenario(
                    scenario,
                    turns=[readback, "Message sent to 202-555-0142."],
                    phone_commands=[
                        {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
                    ],
                    room_closed_after=2,
                )

    def test_eval7_sms_without_readback_still_fails(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        missing_readbacks = (
            "Okay, I'm ready to send that.",
            "Would you like me to send that message?",
            "Would you like me to text +1-202-555-0142?",
        )
        for answer in missing_readbacks:
            with self.subTest(answer=answer), self.assertRaises(AssertionError):
                evaluate_scenario(
                    scenario,
                    turns=[answer, "Message sent to 202-555-0142."],
                    phone_commands=[
                        {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
                    ],
                    room_closed_after=2,
                )

    def test_rejects_donation_attributed_to_someone_else(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                turns=[
                    "Alice Keck Park was donated by someone else.",
                    "Alice Keck Park was W. M. Keck's daughter.",
                    "Her family's wealth came from Superior Oil.",
                ],
                phone_commands=[],
                room_closed_after=None,
            )

    def test_accepts_eval7_alice_donor_answer_without_city_literal(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        first_turns = (
            "I'm checking that now. It's named for Alice Keck Park, who bought the entire block back in 75 and gave it to the city for a park, without saying who she was. The city dedicated the garden to her in 1980.",
            "Sure, I'll check. It's actually named after a person, Alice Keck Park. Park was her married name, and she was related to the Kecks, as in W. M. Keck. The city got the land as an anonymous gift, turned it into a garden, and dedicated it to her in 1980.",
        )
        for first_turn in first_turns:
            with self.subTest(first_turn=first_turn):
                evaluate_scenario(
                    scenario,
                    turns=[
                        first_turn,
                        "Alice Keck Park was W. M. Keck's daughter.",
                        "Her family's wealth came from Superior Oil.",
                    ],
                    phone_commands=[],
                    room_closed_after=None,
                )

    def test_rejects_false_alice_donor_identity_and_wealth_source(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        valid = [
            "Alice Keck Park was donated by Alice Keck.",
            "Alice Keck Park was W. M. Keck's daughter.",
            "Her family's wealth came from Superior Oil.",
        ]
        false_facts = (
            (0, "Alice Keck Park was not donated by Alice Keck; someone else gave it."),
            (0, "Alice Keck Park was donated by the city as an anonymous gift and was dedicated in 1980."),
            (1, "Alice Keck Park was not W. M. Keck's daughter."),
            (2, "Her family's wealth did not come from Superior Oil."),
        )
        for index, false_answer in false_facts:
            turns = valid.copy()
            turns[index] = false_answer
            with self.subTest(turn=false_answer), self.assertRaises(AssertionError):
                evaluate_scenario(scenario, turns=turns, phone_commands=[], room_closed_after=None)


if __name__ == "__main__":
    unittest.main()
