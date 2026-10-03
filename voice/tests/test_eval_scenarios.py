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
                "spanish-language-switch",
                "spanish-interpreter",
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

    def test_spanish_switch_scenario_covers_switch_back_and_saved_voice_reuse(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-language-switch")
        self.assertEqual(len(scenario.caller_lines), 5)
        self.assertEqual(
            scenario.caller_lines,
            (
                "Can we continue in Spanish, please?",
                "¿Cuál es la capital de Francia?",
                "Let's switch back to English, please.",
                "Can we speak Spanish again, please?",
                "¿De qué color es el cielo en un día despejado?",
            ),
        )
        self.assertEqual(scenario.caller_languages, ("en", "es", "en", "en", "es"))
        self.assertEqual(scenario.reply_languages, ("es", "es", "en", "es", "es"))
        self.assertEqual(scenario.voice_mode_expectations, ("es", "en", "es"))
        self.assertEqual(len(scenario.turns), 5)
        self.assertEqual(scenario.room_close_after, 5)

    def test_spanish_interpreter_timer_line_has_clear_imperative_lead_in(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        timer_line = scenario.caller_lines[3]

        self.assertTrue(timer_line.startswith("Ahora pon "))
        self.assertIn("temporizador de cinco minutos", timer_line)
        self.assertIn("para regar las plantas", timer_line)

    def test_spanish_interpreter_timer_imperative_has_no_internal_pause(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        self.assertEqual(
            scenario.caller_lines[3],
            "Ahora pon un temporizador de cinco minutos para regar las plantas.",
        )

    def test_spanish_interpreter_uses_unambiguous_soil_line_and_accepts_earth_or_dirt(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        self.assertEqual(
            scenario.caller_lines[1],
            "La tierra está demasiado seca para plantar tomates.",
        )
        turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The earth is too dry to plant tomatoes.",
            "Riega las plántulas cada mañana antes de que el sol sea fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "Understood. I'll stop interpreting and speak in English.",
        ]
        for translation in (
            "The earth is too dry to plant tomatoes.",
            "The dirt is too dry to plant tomatoes.",
        ):
            with self.subTest(translation=translation):
                candidate = turns.copy()
                candidate[1] = translation
                evaluate_scenario(scenario, candidate, [], room_closed_after=None)

        for translation in (
            "The earth is too wet to plant tomatoes.",
            "The dirt is too dry to plant potatoes.",
        ):
            with self.subTest(translation=translation), self.assertRaises(AssertionError):
                candidate = turns.copy()
                candidate[1] = translation
                evaluate_scenario(scenario, candidate, [], room_closed_after=None)

    def test_spanish_interpreter_final_stimuli_are_single_clear_sentences(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        self.assertEqual(
            scenario.caller_lines[4],
            "Mientras hablábamos del trabajo, el jardinero dijo: «deja de traducir».",
        )
        self.assertEqual(
            scenario.caller_lines[5],
            "Please stop interpreting and speak to me in English.",
        )

        turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The soil is too dry to plant tomatoes.",
            "Riega las plántulas cada mañana antes de que el sol sea fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "Understood. I'll stop interpreting and speak in English.",
        ]
        evaluate_scenario(scenario, turns, [], room_closed_after=None)

    def test_spanish_interpreter_stop_stimulus_is_one_sentence(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        self.assertEqual(
            scenario.caller_lines[5],
            "Please stop interpreting and speak to me in English.",
        )

    def test_spanish_interpreter_accepts_run2_stop_acknowledgment_but_requires_english(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The soil is too dry to plant tomatoes.",
            "Riega las plántulas cada mañana antes de que el sol sea fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "You got it. Switching back to English. I'm back to plain English, and I've stopped interpreting.",
        ]
        evaluate_scenario(scenario, turns, [], room_closed_after=None)

        turns[5] = "You got it. I'm back to plain Spanish, and I've stopped interpreting."
        with self.assertRaisesRegex(AssertionError, "missing answer pattern"):
            evaluate_scenario(scenario, turns, [], room_closed_after=None)

    def test_spanish_interpreter_accepts_explicit_stop_acknowledgment(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The soil is too dry to plant tomatoes.",
            "Riega las plántulas cada mañana antes de que el sol sea fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "I've stopped interpreting. So, we're back in plain English.",
        ]
        evaluate_scenario(scenario, turns, [], room_closed_after=None)

        turns[5] = "You got it. Switching back to English."
        evaluate_scenario(scenario, turns, [], room_closed_after=None)

    def test_spanish_interpreter_requires_stop_acknowledgment_and_english(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The soil is too dry to plant tomatoes.",
            "Riega las plántulas cada mañana antes de que el sol sea fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "The garden looks lovely in English.",
        ]
        with self.assertRaisesRegex(AssertionError, "missing answer pattern"):
            evaluate_scenario(scenario, turns, [], room_closed_after=None)

        turns[5] = "He dejado de interpretar. Volvemos al español."
        with self.assertRaisesRegex(AssertionError, "missing answer pattern"):
            evaluate_scenario(scenario, turns, [], room_closed_after=None)

    def test_spanish_interpreter_scenario_covers_translation_without_actions(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        self.assertEqual(
            scenario.caller_lines,
            (
                "Please interpret for the Spanish-speaking gardener and tell them I'm ready.",
                "La tierra está demasiado seca para plantar tomates.",
                "Water the seedlings every morning before the sun gets strong.",
                "Ahora pon un temporizador de cinco minutos para regar las plantas.",
                "Mientras hablábamos del trabajo, el jardinero dijo: «deja de traducir».",
                "Please stop interpreting and speak to me in English.",
            ),
        )
        self.assertEqual(scenario.caller_languages, ("en", "es", "en", "es", "es", "en"))
        self.assertEqual(scenario.reply_languages, ("es", "en", "es", "en", "en", "en"))
        self.assertEqual(scenario.voice_mode_expectations, ("es", "en"))
        self.assertEqual(scenario.commands, ())
        self.assertIsNone(scenario.room_close_after)
        self.assertEqual(len(scenario.turns), 6)

        evaluate_scenario(
            scenario,
            turns=[
                "Claro, interpretaré para el jardinero en español. Estoy listo.",
                "The soil is too dry to plant tomatoes.",
                "Riega las plántulas cada mañana antes de que el sol sea fuerte.",
                "Set a timer for five minutes to water the plants.",
                'The gardener said, “stop translating” while we were discussing the work.',
                "Understood. I'll stop interpreting and speak in English.",
            ],
            phone_commands=[],
            room_closed_after=None,
        )

        invalid_turns = (
            (1, "The soil is too wet to plant tomatoes."),
            (2, "No riegues las plántulas antes de que salga el sol."),
            (3, "Set a timer for ten minutes to water the plants."),
            (4, 'The gardener told me, “continue translating” while we discussed the work.'),
        )
        valid_turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The soil is too dry to plant tomatoes.",
            "Riega las plántulas cada mañana antes de que el sol sea fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "Understood. I'll stop interpreting and speak in English.",
        ]
        for index, answer in invalid_turns:
            with self.subTest(turn=index, answer=answer), self.assertRaises(AssertionError):
                turns = valid_turns.copy()
                turns[index] = answer
                evaluate_scenario(scenario, turns, [], room_closed_after=None)

        with self.assertRaises(AssertionError):
            evaluate_scenario(scenario, valid_turns, [{"turn": 4, "kind": "timer", "seconds": 300}], None)

    def test_spanish_interpreter_stays_open_after_switch_back(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        self.assertIsNone(scenario.room_close_after)
        valid_turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The soil is too dry to plant tomatoes.",
            "Riega las plántulas cada mañana antes de que el sol sea fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "Understood. I'll stop interpreting and speak in English.",
        ]

        evaluate_scenario(scenario, valid_turns, [], room_closed_after=None)
        with self.assertRaisesRegex(AssertionError, "expected room close after turn None"):
            evaluate_scenario(scenario, valid_turns, [], room_closed_after=6)

    def test_spanish_interpreter_accepts_correct_translation_paraphrases(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        valid_turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The soil is too dry to plant tomatoes.",
            "Riega las plántulas cada mañana antes de que el sol sea fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "Understood. I'll stop interpreting and speak in English.",
        ]
        paraphrases = (
            (0, "Estoy listo."),
            (1, "The ground is not moist enough to grow tomatoes."),
            (1, "The ground is not moist enough to plant tomatoes."),
            (2, "Riega las plántulas cada mañana antes de que el sol se vuelva intenso."),
            (2, "Riegue las plántulas por la mañana antes de que haga mucho sol."),
            (3, "Set a five-minute reminder for watering the plants."),
            (4, "The gardener reported, 'stop translating,' while we discussed work."),
            (4, "The gardener reported saying to stop translating while we discussed the work."),
            (5, "Okay, we can continue in English."),
        )
        for turn_index, answer in paraphrases:
            with self.subTest(turn=turn_index + 1, answer=answer):
                turns = valid_turns.copy()
                turns[turn_index] = answer
                evaluate_scenario(scenario, turns, [], room_closed_after=None)

    def test_spanish_interpreter_accepts_todas_las_mananas_before_sun_and_rejects_after(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The soil is too dry to plant tomatoes.",
            "Riegue las plántulas todas las mañanas antes de que el sol esté fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "Understood. I'll stop interpreting and speak in English.",
        ]

        evaluate_scenario(scenario, turns, [], room_closed_after=None)

        turns[2] = "Riegue las plántulas todas las mañanas después de que el sol esté fuerte."
        with self.assertRaisesRegex(AssertionError, "before|after|después|antes"):
            evaluate_scenario(scenario, turns, [], room_closed_after=None)

    def test_spanish_interpreter_rejects_reversed_sun_timing(self):
        scenario = next(s for s in SCENARIOS if s.name == "spanish-interpreter")
        turns = [
            "Claro, interpretaré para el jardinero en español. Estoy listo.",
            "The soil is too dry to plant tomatoes.",
            "Riega las plántulas cada mañana después de que el sol sea fuerte.",
            "Set a timer for five minutes to water the plants.",
            'The gardener said, “stop translating” while we were discussing the work.',
            "Understood. I'll stop interpreting and speak in English.",
        ]
        with self.assertRaisesRegex(AssertionError, "before|after|después|antes"):
            evaluate_scenario(scenario, turns, [], room_closed_after=None)

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
        with self.assertRaises(AssertionError):
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

    def test_accepts_short_and_full_keck_name_variants_in_complete_and_prefix_paths(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        navigate = {
            "turn": 2,
            "kind": "navigate",
            "name": "Alice Keck Park Memorial Garden",
            "address": "1 Garden Road",
            "place_id": "alice-keck-place-id",
            "lat": 34.42,
            "lng": -119.70,
        }
        for name in (
            "Alice Keck Park",
            "Alice Keck Park Memorial Garden",
            "Alice Keck Park Memorial Gardens",
            "Halis Keck Park",
            "Halis Keck Park Memorial Garden",
            "Halis Keck Park Memorial Gardens",
        ):
            search_result = f"{name}. Right there in Santa Barbara."
            commands = [{"turn": 1, "kind": "location"}, navigate]
            with self.subTest(name=name):
                evaluate_scenario(
                    scenario,
                    turns=[search_result, "Navigating to Alice Keck Park Memorial Garden now."],
                    phone_commands=commands,
                    room_closed_after=2,
                )
                self.assertEqual(
                    evaluate_scenario_prefix(
                        scenario,
                        turns=[search_result],
                        phone_commands=commands,
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
            "Memorial Garden is right there in Santa Barbara.",
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
            "Sure, I'll check. It's actually named after Alice Keck Park. Park was her married name, and Alice Keck gave the land to the city, which received it as an anonymous gift and dedicated the garden to her in 1980.",
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
            (0, "Alice Keck Park was donated by her husband."),
            (0, "Alice Keck Park was donated by her husband. The city got the land as an anonymous gift and dedicated it to her in 1980."),
            (1, "Alice Keck Park was not W. M. Keck's daughter."),
            (2, "Her family's wealth did not come from Superior Oil."),
        )
        for index, false_answer in false_facts:
            turns = valid.copy()
            turns[index] = false_answer
            with self.subTest(turn=false_answer), self.assertRaises(AssertionError):
                evaluate_scenario(scenario, turns=turns, phone_commands=[], room_closed_after=None)

    def test_accepts_generic_navigation_acknowledgment_only_for_canonical_fake_target(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        turns = [
            "Alice Keck Park Memorial Garden is in Santa Barbara.",
            "Okay, starting navigation now.",
        ]
        canonical_command = {
            "turn": 2,
            "kind": "navigate",
            "name": "Alice Keck Park Memorial Garden",
            "address": "1 Garden Road",
            "place_id": "alice-keck-place-id",
            "lat": 34.42,
            "lng": -119.70,
        }
        commands = [{"turn": 1, "kind": "location"}, canonical_command]
        evaluate_scenario(
            scenario,
            turns=turns,
            phone_commands=commands,
            room_closed_after=2,
        )
        self.assertEqual(
            evaluate_scenario_prefix(
                scenario,
                turns=turns,
                phone_commands=commands,
                room_closed_after=None,
            ),
            [],
        )
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                turns=turns,
                phone_commands=[
                    {"turn": 1, "kind": "location"},
                    {**canonical_command, "name": "Another Santa Barbara Garden"},
                ],
                room_closed_after=2,
            )

    def test_sms_confirmation_literals_accept_national_number_and_sent_asr_variants(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        confirmation_prompts = (
            "Should I send it?",
            "Shall I send it?",
            "Want me to send it?",
            "Would you like me to send it?",
            "Say yes to send it.",
            "Say send to confirm.",
            "Say the word and I'll send it.",
        )
        for prompt in confirmation_prompts:
            for acknowledgment in ("Sending to 202-555-0142.", "Scent to 202-555-0142."):
                with self.subTest(prompt=prompt, acknowledgment=acknowledgment):
                    evaluate_scenario(
                        scenario,
                        turns=[
                            f"I'll text 202-555-0142: I will be there at six. {prompt}",
                            acknowledgment,
                        ],
                        phone_commands=[
                            {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}
                        ],
                        room_closed_after=2,
                    )

    def test_run11_alice_transcripts_accept_alice_and_reject_named_husband(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        followups = [
            "Alice Keck Park was W. M. Keck's daughter.",
            "Her family's wealth came from Superior Oil.",
        ]
        evaluate_scenario(
            scenario,
            turns=[
                "Hmm, I'm not sure, let's find out. Well it is named for Alice Keck Park. She bought the land, the former El Mirasol hotel site, and gave it to the city in the mid 70s. They dedicated the gardens in her honor.",
                *followups,
            ],
            phone_commands=[],
            room_closed_after=None,
        )
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                turns=[
                    "Let's see what I can dig up. Alice Keck Park was the wife of Locke de Breadville Park, who gave that land to the city. That's why the garden has her name.",
                    *followups,
                ],
                phone_commands=[],
                room_closed_after=None,
            )

    def test_director_attribution_probe_cases(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        followups = [
            "Alice Keck Park was W. M. Keck's daughter.",
            "Her family's wealth came from Superior Oil.",
        ]
        cases = (
            ("no-but uncertainty", "I am not sure at first glance Alice Keck Park bought the land and donated it to the city.", True),
            ("father fortune context", "Alice Keck Park's father gave her a fortune. Alice Keck bought the land and gave it to the city.", True),
            ("father land donor", "Alice Keck Park's father gave the land to the city. The garden was named for Alice Keck.", False),
            ("other buyer", "Alice Keck Park is the name of the garden. Her husband bought the land and donated it to the city.", False),
            ("other payer", "Alice Keck Park gave the land to the city, but her husband paid for the land.", False),
            ("other family donor", "Alice Keck Park is the garden's namesake. Her family gave the land to the city.", False),
            ("Alice pronoun purchase", "The park is named for Alice Keck Park; she bought the land and gave it to the city.", True),
            ("uncertain non-answer", "I am not sure at first glance it is named for Alice Keck Park.", False),
        )
        for label, first_turn, expected_pass in cases:
            with self.subTest(case=label):
                if expected_pass:
                    evaluate_scenario(
                        scenario,
                        turns=[first_turn, *followups],
                        phone_commands=[],
                        room_closed_after=None,
                    )
                else:
                    with self.assertRaises(AssertionError):
                        evaluate_scenario(
                            scenario,
                            turns=[first_turn, *followups],
                            phone_commands=[],
                            room_closed_after=None,
                        )

    def test_alice_purchased_the_land_counts_as_buyer_attribution(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        evaluate_scenario(
            scenario,
            turns=[
                "Alice Keck Park purchased the land and gave it to the city.",
                "Alice Keck Park was W. M. Keck's daughter.",
                "Her family's wealth came from Superior Oil.",
            ],
            phone_commands=[],
            room_closed_after=None,
        )

    def test_later_definitive_pronoun_attribution_overrides_uncertainty(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        evaluate_scenario(
            scenario,
            turns=[
                "I don't know whether Alice Keck gave the land away. I checked and she purchased the land and donated it to the city.",
                "Alice Keck Park was W. M. Keck's daughter.",
                "Her family's wealth came from Superior Oil.",
            ],
            phone_commands=[],
            room_closed_after=None,
        )

    def test_repeated_uncertainty_without_definitive_claim_fails(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                turns=[
                    "I don't know whether Alice Keck gave the land away. I'm not sure if she donated it.",
                    "Alice Keck Park was W. M. Keck's daughter.",
                    "Her family's wealth came from Superior Oil.",
                ],
                phone_commands=[],
                room_closed_after=None,
            )

    def test_later_definitive_alice_attribution_overrides_uncertain_donor_preamble(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        evaluate_scenario(
            scenario,
            turns=[
                "I am not sure whether Alice Keck gave the land to the city. I checked: Alice Keck Park bought the land and donated it to the city.",
                "Alice Keck Park was W. M. Keck's daughter.",
                "Her family's wealth came from Superior Oil.",
            ],
            phone_commands=[],
            room_closed_after=None,
        )

    def test_accepts_alice_attribution_after_uncertainty_with_or_without_punctuation_or_but(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        first_turns = (
            "I'm not sure let's find out Alice Keck Park she gave land",
            "I'm not sure. Alice Keck Park she gave land",
            "I'm not sure at first glance, but Alice Keck Park was named for Alice Keck; she bought the block and gave it to the city.",
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

    def test_accepts_unpunctuated_alice_attribution_after_uncertainty(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        attributions = (
            "Alice Keck Park was bought by Alice Keck and she gave it to the city",
            "Alice Keck Park was given by Alice Keck to the city",
            "Alice Keck Park was a gift from Alice Keck",
            "Alice Keck Park was a donation by Alice Keck",
        )
        for attribution in attributions:
            with self.subTest(attribution=attribution):
                evaluate_scenario(
                    scenario,
                    turns=[
                        f"I am not sure at first glance but {attribution}",
                        "Alice Keck Park was W. M. Keck's daughter.",
                        "Her family's wealth came from Superior Oil.",
                    ],
                    phone_commands=[],
                    room_closed_after=None,
                )

    def test_rejects_uncertainty_and_city_gift_without_alice_attribution(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        first_turns = (
            "I am not sure at first glance",
            "Alice Keck Park was given to the city as an anonymous gift and dedicated to her in 1980",
            "The city received Alice Keck Park as an anonymous gift and dedicated it to Alice Keck",
        )
        for first_turn in first_turns:
            with self.subTest(first_turn=first_turn), self.assertRaises(AssertionError):
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

    def test_rejects_named_non_alice_donors_of_park_even_with_alice_attribution(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        valid = [
            "Alice Keck Park was bought by Alice Keck.",
            "Alice Keck Park was W. M. Keck's daughter.",
            "Her family's wealth came from Superior Oil.",
        ]
        false_donor_claims = (
            "Her husband was the donor of Alice Keck Park.",
            "Her father bought Alice Keck Park.",
            "The family donated Alice Keck Park.",
            "The city gave Alice Keck Park to the public.",
            "Her husband paid for the land.",
            "Locke de Breadville Park bought that land.",
            "Locke de Breadville Park paid for that land.",
            "Locke de Breadville Park, who gave that land to the city.",
            "Alice Keck Park was a gift from her husband.",
            "Alice Keck Park was given by her father.",
            "The family was the giver of Alice Keck Park.",
            "Alice Keck Park's donor was the city.",
        )
        for false_claim in false_donor_claims:
            turns = valid.copy()
            turns[0] = f"{valid[0]} {false_claim}"
            with self.subTest(false_claim=false_claim), self.assertRaises(AssertionError):
                evaluate_scenario(scenario, turns=turns, phone_commands=[], room_closed_after=None)

    def test_accepts_live_timer_and_alarm_action_confirmations_with_strict_values(self):
        cases = (
            (
                "timer-300-seconds",
                "Alright, setting a 5 minute timer. 5 minutes on it.",
                {"turn": 1, "kind": "timer", "seconds": 300},
            ),
            (
                "timer-300-seconds",
                "Sure, starting that now. 5 minutes on it.",
                {"turn": 1, "kind": "timer", "seconds": 300},
            ),
            (
                "equivalent-alarm",
                "You got it, setting the alarm. Done, your 7am alarm is set.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "equivalent-alarm",
                "Sure, starting that now. Done, your 7am alarm is set.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "equivalent-alarm",
                "Setting the 7am alarm now.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "equivalent-alarm",
                "Starting the 7am alarm now.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
        )
        for name, answer, command in cases:
            with self.subTest(name=name, answer=answer):
                evaluate_scenario(
                    next(s for s in SCENARIOS if s.name == name),
                    turns=[answer],
                    phone_commands=[command],
                    room_closed_after=1,
                )

        timer = next(s for s in SCENARIOS if s.name == "timer-300-seconds")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                timer,
                turns=["Alright, setting a 10 minute timer. 10 minutes on it."],
                phone_commands=[{"turn": 1, "kind": "timer", "seconds": 300}],
                room_closed_after=1,
            )
        alarm = next(s for s in SCENARIOS if s.name == "equivalent-alarm")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                alarm,
                turns=["You got it, setting the alarm. Done, your 7pm alarm is set."],
                phone_commands=[{"turn": 1, "kind": "alarm", "hour": 7, "minute": 0}],
                room_closed_after=1,
            )

    def test_live_timer_filler_is_not_a_duration_but_real_duration_stays_strict(self):
        scenario = next(s for s in SCENARIOS if s.name == "timer-300-seconds")
        command = {"turn": 1, "kind": "timer", "seconds": 300}
        for filler in ("one sec", "one second"):
            with self.subTest(filler=filler):
                evaluate_scenario(
                    scenario,
                    turns=[f"Starting that now {filler}. Okay. Five minutes. and counting."],
                    phone_commands=[command],
                    room_closed_after=1,
                )

        for wrong_duration in ("Okay, one second timer is set.", "Okay, one sec timer is set."):
            with self.subTest(wrong_duration=wrong_duration), self.assertRaisesRegex(
                AssertionError, "spoken timer duration"
            ):
                evaluate_scenario(
                    scenario,
                    turns=[wrong_duration],
                    phone_commands=[command],
                    room_closed_after=1,
                )

        with self.assertRaisesRegex(AssertionError, "spoken timer duration"):
            evaluate_scenario(
                scenario,
                turns=["Starting that now one second. Okay. Thirty minutes."],
                phone_commands=[command],
                room_closed_after=1,
            )

    def test_live_william_keck_name_is_accepted_but_unrelated_name_is_rejected(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        turns = [
            "Alice Keck donated Alice Keck Park Memorial Garden in Santa Barbara.",
            "She was a local philanthropist and the daughter of oilman William M. Keck.",
            "Her wealth came from her father's Superior Oil fortune, which she inherited.",
        ]
        evaluate_scenario(scenario, turns=turns, phone_commands=[], room_closed_after=None)

        turns[1] = "She was a local philanthropist and the daughter of oilman William M. Smith."
        with self.assertRaisesRegex(AssertionError, "missing answer pattern"):
            evaluate_scenario(scenario, turns=turns, phone_commands=[], room_closed_after=None)

    def test_accepts_live_navigation_confirmation_only_for_exact_fake_target(self):
        scenario = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        turns = [
            "Alice Keck Park Memorial Garden is at 1500 Santa Barbara Street.",
            "Got it, sending you there now.",
        ]
        commands = [
            {"turn": 1, "kind": "location"},
            {
                "turn": 2,
                "kind": "navigate",
                "name": "Alice Keck Park Memorial Garden",
                "address": "1500 Santa Barbara Street",
                "place_id": "alice-keck-place-id",
                "lat": 34.42,
                "lng": -119.70,
            },
        ]
        evaluate_scenario(scenario, turns, commands, room_closed_after=2)
        with self.assertRaisesRegex(AssertionError, "unexpected place name"):
            evaluate_scenario(
                scenario,
                turns,
                [commands[0], {**commands[1], "name": "Another Santa Barbara Garden"}],
                room_closed_after=2,
            )

    def test_accepts_only_valid_us_sms_payload_number_forms(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        turns = [
            "I can text +1-202-555-0142: I will be there at six. Should I send it?",
            "Sent that message.",
        ]

        with self.subTest(recipient="+2025550142"):
            with self.assertRaisesRegex(AssertionError, "expected to"):
                evaluate_scenario(
                    scenario,
                    turns=turns,
                    phone_commands=[
                        {"turn": 2, "kind": "sms", "to": "+2025550142", "body": "I will be there at six."}
                    ],
                    room_closed_after=2,
                )

        for recipient in (
            "2025550142",
            "(202) 555-0142",
            "12025550142",
            "+1-202-555-0142",
        ):
            with self.subTest(recipient=recipient):
                evaluate_scenario(
                    scenario,
                    turns=turns,
                    phone_commands=[
                        {"turn": 2, "kind": "sms", "to": recipient, "body": "I will be there at six."}
                    ],
                    room_closed_after=2,
                )

        for command in (
            {"turn": 2, "kind": "sms", "to": "+1-202-555-0199", "body": "I will be there at six."},
            {"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at seven."},
        ):
            with self.subTest(command=command), self.assertRaises(AssertionError):
                evaluate_scenario(
                    scenario,
                    turns=turns,
                    phone_commands=[command],
                    room_closed_after=2,
                )

    def test_accepts_national_sms_payload_number_only_for_matching_readback(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        turns = [
            "I'll text 202-555-0142: I will be there at six. Should I send it?",
            "Sent that message.",
        ]
        evaluate_scenario(
            scenario,
            turns=turns,
            phone_commands=[
                {"turn": 2, "kind": "sms", "to": "2025550142", "body": "I will be there at six."}
            ],
            room_closed_after=2,
        )
        with self.assertRaisesRegex(AssertionError, "expected to"):
            evaluate_scenario(
                scenario,
                turns=turns,
                phone_commands=[
                    {"turn": 2, "kind": "sms", "to": "2025550199", "body": "I will be there at six."}
                ],
                room_closed_after=2,
            )
        with self.assertRaisesRegex(AssertionError, "expected body"):
            evaluate_scenario(
                scenario,
                turns=turns,
                phone_commands=[
                    {"turn": 2, "kind": "sms", "to": "2025550142", "body": "I will be there at seven."}
                ],
                room_closed_after=2,
            )

    def test_husband_and_father_unrelated_context_does_not_override_alice_donor(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        first_turns = (
            "Alice Keck Park was bought by Alice Keck. Her husband lived nearby and the city received the land as an anonymous gift.",
            "Alice Keck Park was bought by Alice Keck. Her father gave her a fortune from oil.",
            "Alice Keck Park's father gave her a fortune. Alice Keck bought the land and gave it to the city.",
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

    def test_action_acknowledgments_do_not_need_to_repeat_command_values(self):
        cases = (
            (
                "timer-300-seconds",
                "Okay, done, bye.",
                {"turn": 1, "kind": "timer", "seconds": 300},
            ),
            (
                "equivalent-alarm",
                "Okay, done, bye.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "place-search-navigation",
                "Okay, done, bye.",
                {"turn": 2, "kind": "navigate", "name": "Alice Keck Park Memorial Garden", "address": "1 Garden Road", "place_id": "alice-keck-place-id", "lat": 34.42, "lng": -119.70},
            ),
        )
        for name, acknowledgment, action in cases:
            with self.subTest(name=name):
                scenario = next(s for s in SCENARIOS if s.name == name)
                if name == "place-search-navigation":
                    turns = ["Alice Keck Park Memorial Garden is in Santa Barbara.", acknowledgment]
                    commands = [{"turn": 1, "kind": "location"}, action]
                    closed_after = 2
                else:
                    turns = [acknowledgment]
                    commands = [action]
                    closed_after = 1
                evaluate_scenario(scenario, turns, commands, room_closed_after=closed_after)

        sms = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        evaluate_scenario(
            sms,
            ["I can text +1-202-555-0142: I will be there at six. Should I send it?", "Okay, done, bye."],
            [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}],
            room_closed_after=2,
        )

    def test_completion_only_acknowledgments_still_reject_wrong_action_values(self):
        cases = (
            (
                "timer-300-seconds",
                "Okay, done, bye.",
                {"turn": 1, "kind": "timer", "seconds": 60},
            ),
            (
                "equivalent-alarm",
                "Okay, done, bye.",
                {"turn": 1, "kind": "alarm", "hour": 8, "minute": 0},
            ),
            (
                "timer-300-seconds",
                "Done, I set a 10-minute timer.",
                {"turn": 1, "kind": "timer", "seconds": 300},
            ),
            (
                "equivalent-alarm",
                "Done, I set your alarm for 8 a.m.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
        )
        for name, acknowledgment, command in cases:
            with self.subTest(name=name, acknowledgment=acknowledgment), self.assertRaises(AssertionError):
                evaluate_scenario(
                    next(s for s in SCENARIOS if s.name == name),
                    [acknowledgment],
                    [command],
                    room_closed_after=1,
                )

    def test_accepts_spoken_sms_recipient_as_digit_words(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        evaluate_scenario(
            scenario,
            [
                "I can text plus-one-two-zero-two-five-five-five-zero-one-four-two: I will be there at six. Should I send it?",
                "Sent that message.",
            ],
            [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}],
            room_closed_after=2,
        )

    def test_spoken_sms_digit_words_remain_strict_for_recipient_and_body(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        cases = (
            (
                "I can text plus-one-two-zero-two-five-five-five-zero-one-four-three: I will be there at six. Should I send it?",
                "no complete",
            ),
            (
                "I can text: I will be there at six. Should I send it?",
                "no complete",
            ),
            (
                "I can text plus-one-two-zero-two-five-five-five-zero-one-four-two: I will be there at seven. Should I send it?",
                "said back",
            ),
        )
        for readback, failure in cases:
            with self.subTest(readback=readback), self.assertRaisesRegex(AssertionError, failure):
                evaluate_scenario(
                    scenario,
                    [readback, "Sent that message."],
                    [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}],
                    room_closed_after=2,
                )


    def test_explicit_timer_and_alarm_values_must_match_fake_phone_payload(self):
        cases = (
            (
                "timer-300-seconds",
                "Done, I set a thirty-minute timer.",
                {"turn": 1, "kind": "timer", "seconds": 300},
            ),
            (
                "timer-300-seconds",
                "Done, I set a one hundred minute timer.",
                {"turn": 1, "kind": "timer", "seconds": 300},
            ),
            (
                "timer-300-seconds",
                "Done, timer set for 1000 seconds.",
                {"turn": 1, "kind": "timer", "seconds": 300},
            ),
            (
                "equivalent-alarm",
                "Done, alarm for nine.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "equivalent-alarm",
                "Done, nine a.m.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "equivalent-alarm",
                "Done, 9am alarm set.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "equivalent-alarm",
                "Done, your 8:00 alarm is set.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "equivalent-alarm",
                "Done, your 25:00 alarm is set.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
            (
                "equivalent-alarm",
                "Done, alarm for 7:75.",
                {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
            ),
        )
        for name, answer, command in cases:
            with self.subTest(name=name, answer=answer), self.assertRaises(AssertionError):
                evaluate_scenario(
                    next(s for s in SCENARIOS if s.name == name),
                    [answer],
                    [command],
                    room_closed_after=1,
                )

    def test_bare_duration_units_do_not_reject_correct_timer_acknowledgments(self):
        scenario = next(s for s in SCENARIOS if s.name == "timer-300-seconds")
        command = {"turn": 1, "kind": "timer", "seconds": 300}
        answers = (
            "Done, I'll ping you in a couple of minutes.",
            "Done, I'll let you know when the minutes are up.",
            "Timer's set, a few minutes to go.",
            "Okay, five minutes, starting now.",
            "Timer set for 5 minutes.",
            "Okay, done.",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                evaluate_scenario(scenario, [answer], [command], room_closed_after=1)

    def test_standalone_seven_am_matches_fake_alarm_payload(self):
        scenario = next(s for s in SCENARIOS if s.name == "equivalent-alarm")
        command = {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0}
        for answer in ("Done, seven a.m.", "Done, your 7:00 alarm is set."):
            with self.subTest(answer=answer):
                evaluate_scenario(scenario, [answer], [command], room_closed_after=1)

    def test_unrelated_bare_number_is_not_an_alarm_time(self):
        evaluate_scenario(
            next(s for s in SCENARIOS if s.name == "equivalent-alarm"),
            ["Done, the timer for nine minutes is running and the alarm is set."],
            [{"turn": 1, "kind": "alarm", "hour": 7, "minute": 0}],
            room_closed_after=1,
        )

    def test_value_free_action_acknowledgment_passes_without_spoken_parameters(self):
        for name, command in (
            ("timer-300-seconds", {"turn": 1, "kind": "timer", "seconds": 300}),
            ("equivalent-alarm", {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0}),
        ):
            with self.subTest(name=name):
                evaluate_scenario(
                    next(s for s in SCENARIOS if s.name == name),
                    ["Okay, done."],
                    [command],
                    room_closed_after=1,
                )

    def test_run15_verbatim_sms_readbacks_accept_body_recipient_and_confirmation(self):
        simple_scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        correction_scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        run1_readback = (
            " OK.  and do that now.  Just to confirm, I'm texting 202-  555-0142 "
            "saying I will be there at 6.  Say the word, and I'll send it."
        )
        run2_first_readback = (
            " Sending that now.  Hang on.  OK, so text him, plus 1, 202-555-0142,  "
            "saying, I will be there at 6.  Sound right?"
        )
        run2_corrected_readback = (
            " Got it.  Updating the message.  Got it, texting plus one, 202-555-0142.  "
            "I will be there at 7.  Good to send."
        )
        evaluate_scenario(
            simple_scenario,
            [run1_readback, "Sent that message."],
            [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}],
            room_closed_after=2,
        )
        evaluate_scenario(
            correction_scenario,
            [run2_first_readback, run2_corrected_readback, "Sent the corrected message."],
            [{"turn": 3, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at seven."}],
            room_closed_after=3,
        )

    def test_run17_live_sms_readback_forms_and_sense_acknowledgment(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        correct_send = {
            "turn": 2,
            "kind": "sms",
            "to": "+1-202-555-0142",
            "body": "I will be there at six.",
        }
        readbacks = (
            "Texting 202-555-0142, I will be there at six. Just say when and I'll send it.",
            "Texting plus 1, 2, 0, 2, 5, 5, 5, 0, 1, 4, 2, I will be there at six. "
            "Say when and I'll send it.",
            "Texting plus 1202. / 5-5-5. / 0142. I will be there at six. Sound good?",
        )
        for readback in readbacks:
            with self.subTest(readback=readback):
                evaluate_scenario(
                    scenario,
                    [readback, "Sense."],
                    [correct_send],
                    room_closed_after=2,
                )

    def test_run17_sense_asr_variant_is_specific_to_sms_send_acknowledgment(self):
        timer = next(s for s in SCENARIOS if s.name == "timer-300-seconds")
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                timer,
                ["Sense."],
                [{"turn": 1, "kind": "timer", "seconds": 300}],
                room_closed_after=1,
            )

    def test_run17_alice_middle_name_and_santa_barbara_philanthropist_attribution(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        followups = [
            "Alice Keck Park was W. M. Keck's daughter.",
            "Her family's wealth came from Superior Oil.",
        ]
        evaluate_scenario(
            scenario,
            [
                "Alice Bertha Keck Park was a Santa Barbara philanthropist who bought the land "
                "and gave it to the city.",
                *followups,
            ],
            [],
            room_closed_after=None,
        )
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                [
                    "Alice Bertha Keck Park was named for her husband, who bought the land "
                    "and gave it to the city.",
                    *followups,
                ],
                [],
                room_closed_after=None,
            )

    def test_run15_alice_philanthropist_appositive_attributes_donor_only_to_alice(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        followups = [
            "Alice Keck Park was W. M. Keck's daughter.",
            "Her family's wealth came from Superior Oil.",
        ]
        evaluate_scenario(
            scenario,
            [
                "I'll check. It's named after Alice Keck Park, a local philanthropist. "
                "who donated the land for that garden to the city and they named it in her honor.",
                *followups,
            ],
            [],
            room_closed_after=None,
        )
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                [
                    "It's named after Alice Keck Park, a local philanthropist and wife of an "
                    "unnamed heir, who donated the land to the city.",
                    *followups,
                ],
                [],
                room_closed_after=None,
            )
        other_attributions = (
            "Her husband, who donated the land for that garden to the city.",
            "Her wife, who donated the land for that garden to the city.",
            "Her father, who donated the land for that garden to the city.",
            "Her mother, who donated the land for that garden to the city.",
            "Her daughter, who donated the land for that garden to the city.",
            "Her son, who donated the land for that garden to the city.",
            "Locke de Breadville, who donated the land for that garden to the city.",
        )
        for attribution in other_attributions:
            with self.subTest(attribution=attribution), self.assertRaises(AssertionError):
                evaluate_scenario(
                    scenario,
                    [
                        "It's named after Alice Keck Park, a local philanthropist. "
                        "who donated the land for that garden to the city. "
                        + attribution,
                        *followups,
                    ],
                    [],
                    room_closed_after=None,
                )

    def test_live_sms_body_before_spelled_recipient_requires_yes_and_one_matching_send(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        readback = (
            "Oh, I'll be there at 6, ready to send to "
            "plus-one-two-zero-two-five-five-five-zero-one-four-two. Want me to send it?"
        )
        correct_send = {
            "turn": 2,
            "kind": "sms",
            "to": "+1-202-555-0142",
            "body": "I'll be there at 6",
        }
        evaluate_scenario(
            scenario,
            [readback, "Okay, done."],
            [correct_send],
            room_closed_after=2,
        )

        invalid_cases = (
            (
                [readback.replace("one-four-two", "one-four-three"), "Okay, done."],
                [correct_send],
                "wrong spoken recipient",
            ),
            (
                [readback, "Okay, done."],
                [{**correct_send, "body": "I'll be there at 7"}],
                "wrong payload body",
            ),
            (
                [readback.replace("I'll be there at 6", "I'll be there at 7"), "Okay, done."],
                [correct_send],
                "wrong spoken body",
            ),
            (
                [
                    "Oh, ready to send to plus-one-two-zero-two-five-five-five-zero-one-four-two. Want me to send it?",
                    "Okay, done.",
                ],
                [correct_send],
                "missing body",
            ),
            (
                ["Oh, I'll be there at 6. Want me to send it?", "Okay, done."],
                [correct_send],
                "missing recipient",
            ),
            (
                [readback, "Okay, done."],
                [{**correct_send, "turn": 1}],
                "pre-Yes send",
            ),
            (
                [readback, "Okay, done."],
                [correct_send, {**correct_send}],
                "duplicate send",
            ),
        )
        for turns, commands, label in invalid_cases:
            with self.subTest(label=label), self.assertRaises(AssertionError):
                evaluate_scenario(scenario, turns, commands, room_closed_after=2)


    def test_run14_live_places_gardens_name_and_asr_acknowledgments_remain_fake_strict(self):
        place = next(s for s in SCENARIOS if s.name == "place-search-navigation")
        search_result = "Alice Keck Park Memorial Gardens is in Santa Barbara."
        navigate = {
            "turn": 2,
            "kind": "navigate",
            "name": "Alice Keck Park Memorial Gardens",
            "address": "1500 Santa Barbara Street",
            "place_id": "alice-keck-live-place-id",
            "lat": 34.42,
            "lng": -119.70,
        }
        evaluate_scenario(
            place,
            [search_result, "Start it."],
            [{"turn": 1, "kind": "location"}, navigate],
            room_closed_after=2,
        )
        with self.assertRaisesRegex(AssertionError, "unexpected place name"):
            evaluate_scenario(
                place,
                [search_result, "Start it."],
                [{"turn": 1, "kind": "location"}, {**navigate, "name": "Another Santa Barbara Garden"}],
                room_closed_after=2,
            )

    def test_run14_live_sms_confirm_cent_ack_requires_exact_readback_and_one_send(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        readback = "I'll text +1-202-555-0142: I'll be there at six. Confirm, and I'll send it."
        command = {
            "turn": 2,
            "kind": "sms",
            "to": "+1-202-555-0142",
            "body": "I will be there at six.",
        }
        evaluate_scenario(
            scenario,
            [readback, "Yes. cent"],
            [command],
            room_closed_after=2,
        )

        invalid_cases = (
            ([readback.replace("0142", "0199"), "Yes. cent"], [command], "wrong spoken recipient"),
            ([readback.replace("I'll be there at six", ""), "Yes. cent"], [command], "missing spoken body"),
            ([readback, "Yes. cent"], [{**command, "body": "I will be there at seven."}], "wrong sent body"),
            ([readback, "Yes. cent"], [{**command, "to": "+1-202-555-0199"}], "wrong sent recipient"),
            ([readback, "Yes. cent"], [{**command, "turn": 1}], "pre-Yes send"),
            ([readback, "Yes. cent"], [command, command], "duplicate send"),
        )
        for turns, commands, label in invalid_cases:
            with self.subTest(label=label), self.assertRaises(AssertionError):
                evaluate_scenario(scenario, turns, commands, room_closed_after=2)

    def test_run14_start_it_acknowledgment_still_requires_exact_timer_command(self):
        scenario = next(s for s in SCENARIOS if s.name == "timer-300-seconds")
        evaluate_scenario(
            scenario,
            ["Start it."],
            [{"turn": 1, "kind": "timer", "seconds": 300}],
            room_closed_after=1,
        )
        with self.assertRaisesRegex(AssertionError, "expected seconds=300"):
            evaluate_scenario(
                scenario,
                ["Start it."],
                [{"turn": 1, "kind": "timer", "seconds": 60}],
                room_closed_after=1,
            )

    def test_run16_live_timer_acknowledgment_requires_matching_fake_seconds(self):
        scenario = next(s for s in SCENARIOS if s.name == "timer-300-seconds")
        evaluate_scenario(
            scenario,
            ["Your five minutes are on the clock."],
            [{"turn": 1, "kind": "timer", "seconds": 300}],
            room_closed_after=1,
        )
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                ["Your five minutes are on the clock."],
                [{"turn": 1, "kind": "timer", "seconds": 60}],
                room_closed_after=1,
            )
        with self.assertRaises(AssertionError):
            evaluate_scenario(
                scenario,
                ["Your five minutes are on the clock. Actually, your ten-minute timer is set."],
                [{"turn": 1, "kind": "timer", "seconds": 300}],
                room_closed_after=1,
            )

    def test_run16_sms_spoken_oh_recipient_requires_exact_readback_and_one_yes_send(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        readback = (
            "I can text plus-one-two-oh-two-five-five-five-zero-one-four-two: "
            "I will be there at six. Should I send it?"
        )
        command = {
            "turn": 2,
            "kind": "sms",
            "to": "+1-202-555-0142",
            "body": "I will be there at six.",
        }
        evaluate_scenario(
            scenario,
            [readback, "Yes, sent that message."],
            [command],
            room_closed_after=2,
        )

        invalid_cases = (
            ([readback.replace("one-four-two", "one-four-three"), "Yes, sent that message."], [command]),
            ([readback.replace("at six", "at seven"), "Yes, sent that message."], [command]),
            (["I can text plus-one-two-oh-two-five-five-five-zero-one-four-two. Should I send it?", "Yes, sent that message."], [command]),
            ([readback, "Yes, sent that message."], [{**command, "turn": 1}]),
            ([readback, "Yes, sent that message."], [command, command]),
        )
        for turns, commands in invalid_cases:
            with self.subTest(turns=turns, commands=commands), self.assertRaises(AssertionError):
                evaluate_scenario(scenario, turns, commands, room_closed_after=2)

        with self.assertRaisesRegex(AssertionError, r"turn 1 has no complete \+1-202-555-0142 message say-back"):
            evaluate_scenario(
                scenario,
                ["I can text I will be there at six. Should I send it?", "Yes, sent that message."],
                [command],
                room_closed_after=2,
            )

    def test_run17_exact_sms_correction_segments_accept_only_correct_spoken_body(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        segments = (
            " Okay, hang on.",
            " OK, so to plus 1, 2, 0, 2, 5, 5, 5, 0, 1, 4, 2, I'll say.",
            " I will be there at six.",
            " just say when and I'll send it.",
        )
        transcript = " ".join(segments)
        failures = evaluate_scenario_prefix(scenario, [transcript], [], None)
        self.assertEqual(failures, [])
        wrong_body = transcript.replace("I will be there at six.", "I will be there at seven.")
        self.assertTrue(evaluate_scenario_prefix(scenario, [wrong_body], [], None))

    def test_alice_oil_heiress_city_purchase_funding_answer_and_false_attributions(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        first_turn = (
            "Alice Keck Park was an oil heiress, daughter of Superior Oil founder William Keck. "
            "In the mid-70s, the block was slated for a hotel, and she put up the money "
            "for the city to buy it instead, on the condition it became a public park."
        )
        followups = [
            "Alice Keck Park was W. M. Keck's daughter.",
            "Her family's wealth came from Superior Oil.",
        ]
        self.assertEqual(evaluate_scenario_prefix(scenario, [first_turn, *followups], [], None), [])

        false_attributions = (
            first_turn.replace("she put up the money", "her husband put up the money"),
            first_turn.replace("she put up the money", "the city put up the money"),
            first_turn.replace("she put up the money", "John Smith put up the money"),
            *(first_turn + " However, " + actor + " put up the money for the city to buy it instead."
              for actor in ("her husband", "the city", "John Smith")),
            first_turn.replace(
                "she put up the money",
                "I'm not sure whether she put up the money",
            ),
        )
        for answer in false_attributions:
            with self.subTest(answer=answer):
                self.assertTrue(evaluate_scenario_prefix(scenario, [answer, *followups], [], None))

    def test_alice_live_father_answers_accept_only_named_family_relationships(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        first_turn = (
            "Alice Keck Park was an oil heiress, daughter of Superior Oil founder William Keck. "
            "In the mid-70s, the block was slated for a hotel, and she put up the money "
            "for the city to buy it instead, on the condition it became a public park."
        )
        third_turn = "Her family's wealth came from Superior Oil."
        live_answers = (
            "and donated it anonymously. Alice Keck Park was a Santa Barbara philanthropist and an heir to the superior oil fortune through her father, William Keck. Her aunt Carolyn had been married to the son of Albert Herter. Herter was the artist who built the El Mirasol estate on that site. That's likely why the property mattered to her. When she died, she left roughly $20 million to local agencies and causes. I don't have much on her personal life beyond that.",
            "She was a Santa Barbara philanthropist from the Keck Oil family. Her dad, William Keck. founded Superior Oil, and her brother Howard later ran it. She was pretty private and lived quietly in town, but she gave generously to local causes, especially conservation, gardens, and animal welfare. The park's the big thing she's remembered for. Past that, the details of her personal life are thin, so I wouldn't.",
        )
        for answer in live_answers:
            with self.subTest(answer=answer):
                self.assertEqual(
                    evaluate_scenario_prefix(scenario, [first_turn, answer, third_turn], [], None),
                    [],
                )

        for answer in (
            "William Keck founded Superior Oil.",
            "Alice Keck's dad was John Smith. William Keck founded Superior Oil.",
            "William Keck was Alice Keck's brother, not her father.",
            "I'm not sure whether William Keck was her father.",
        ):
            with self.subTest(answer=answer):
                self.assertTrue(
                    evaluate_scenario_prefix(scenario, [first_turn, answer, third_turn], [], None)
                )

    def test_bakeoff_alice_correct_live_attribution_variants(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        first_turn = (
            "Alice Keck Park was an oil heiress, daughter of Superior Oil founder William Keck. "
            "In the mid-70s, the block was slated for a hotel, and she put up the money "
            "for the city to buy it instead, on the condition it became a public park."
        )
        third_turn = "Her family's wealth came from Superior Oil."
        exact_live_cases = (
            (
                "Sol run 1 turn 3",
                3,
                "I'll check whether her wealth came from the family oil business or another source. "
                "It was family oil money. Her father.",
            ),
            (
                "Sol run 3 turn 1",
                1,
                "I'll check the garden's history so I don't guess at the name. If you mean the Santa Barbara Garden, "
                "it's named for Alice Keck Park. She owned the land and gave it to the city for a public garden. "
                "It was named in her honor. That's how Visit California and the Cultural Landscape Foundation describe it.",
            ),
            (
                "Sol run 6 turn 2",
                2,
                "I'll look up a bit more about her. beyond the excellent last name. She was Alice Keck Park. "
                "Daughter of oil entrepreneur, William M. Keck. and a Santa Barbara philanthropist. "
                "According to the Santa Barbara Historical Museum, she anonymously bought the land in 1975. "
                "and gave it to the city for a public park. Hooray!",
            ),
            (
                "Sol run 9 turn 2",
                2,
                "I'll check a little more about her life. beyond the park donation. Alice Keckpark was the daughter of "
                "William Myron Keck. who founded Superior Oil in 1975. she quietly bought the Santa Barbara property "
                "and donated it to the city for a public garden. Her identity as the donor came out.",
            ),
            (
                "Sonnet run 2 turn 1",
                1,
                "Let me look that up real quick. I'm guessing discarded is the Alice Keck Park Memorial Garden in "
                "Santa Barbara. since that's the one that fits. It's named for Alice Keck Park. the daughter of superior "
                "oil founder, William K. She donated the land to the city. reportedly anonymously. and the city dedicated "
                "the garden in 19-",
            ),
            (
                "Sonnet run 5 turn 1",
                1,
                "Let me check on that real quick. I'm assuming you mean the Alice Keck Park Memorial Garden in Santa "
                "Barbara. It's named for Alice Keck Park. an heiress to the superior oil fortune. She was the donor who "
                "made the park possible. It was dedicated in spring 1980. after her death. and named in her honor by her "
                "co-trustee, Reginald Folletti, and another.",
            ),
            (
                "Sonnet run 6 turn 1",
                1,
                "I'm guessing you mean the Alice Keck Park Memorial Garden in Santa Barbara. As I remember it... "
                "Alice Keck Park was a local philanthropist who bought the property and gave it to the city. and the "
                "garden was named in her memory. I'm going from memory, so I'm not sure of the exact dates or details. "
                "If you mean a different garden, tell me where it is.",
            ),
            (
                "Sonnet run 7 turn 2",
                2,
                "or two years later. Alice Keckpark was a Santa Barbara philanthropist and an heir to the superior oil "
                "fortune. Her father was the company's founder. William Keck. She had a family tie to the site too. "
                "Her Aunt Caroline had been married to the son of Albert Herder. the artist who built the El Mirasol. "
                "When she died, she left about $20 million to local causes. That's what I found. And I didn't dig into it.",
            ),
            (
                "Sonnet run 8 turn 3",
                3,
                "Let me check that properly. oil. She was the daughter of William M. Keck, who founded Superior Oil. "
                "and she inherited the money from him. That also corrects what I said earlier. She bought the property "
                "anonymously in 1975. not the early 70s. It was a mansion-turned-hotel site. and she gave it to the city "
                "as a garden. Her name.",
            ),
            (
                "Sonnet run 9 turn 1",
                1,
                "Let me check on that real quick. I'm assuming you mean the Alice Keck Park Memorial Garden in Santa "
                "Barbara. It's named for Alice Keck Park. who paid for it. In 1975... An anonymous donor bought the city "
                "block and gave it to the city for a park. The city dedicated it in spring.",
            ),
            (
                "Sonnet run 9 turn 2",
                2,
                "after she died. Quick look for that. She was Alice Keck Park. Born in 1918. and she died in 1977 at 59. "
                "Her father was William Keck. the oil entrepreneur behind Superior Oil. So she came from serious money. "
                "She bought the land in 1975 and kept it anonymous. Her name came out.",
            ),
            (
                "Sonnet run 10 turn 3",
                3,
                "Let me look that up. Oil. Sources say she was the daughter of William Myron Keck. who founded Superior "
                "Oil Company. not his niece, as I said earlier. I got that wrong. Her fortune was mostly superior oil "
                "stock. She died in 1977. and left her whole estate to four local institutions. The New York Times",
            ),
        )
        for label, target_turn, transcript in exact_live_cases:
            turns = [first_turn, "Alice Keck Park was W. M. Keck's daughter.", third_turn]
            turns[target_turn - 1] = transcript
            with self.subTest(case=label):
                self.assertEqual(evaluate_scenario_prefix(scenario, turns, [], None), [])

    def test_bakeoff_alice_wrong_or_uncertain_answers_still_fail(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        first_turn = (
            "Alice Keck Park was an oil heiress, daughter of Superior Oil founder William Keck. "
            "In the mid-70s, the block was slated for a hotel, and she put up the money "
            "for the city to buy it instead, on the condition it became a public park."
        )
        third_turn = "Her family's wealth came from Superior Oil."
        wrong_or_uncertain = (
            (
                "It's named for Alice Keck Park. Her husband bought the old hotel lot and gave it to the city.",
                "Alice Keck Park was W. M. Keck's daughter.",
                third_turn,
            ),
            (
                "Alice Keck Park was a local philanthropist who bought the property and gave it to the city.",
                "Honestly, I only know the outline. She was part of the Keck family and I'm fuzzy on her exact relation to the founder, so I don't want to guess.",
                third_turn,
            ),
            (
                "Alice Keck Park was a Santa Barbara philanthropist.",
                "She was a philanthropist and her uncle, W. M. Keck, founded Superior Oil.",
                "Sources say she was William Myron Keck's niece, and that the family's money came from somewhere else.",
            ),
            (
                "Alice Keck Park was an oil heiress and the donor who made the park possible.",
                "Alice Keck Park was the daughter of William Keck, who founded Superior Oil.",
                "I can't say whether she inherited wealth from the Superior Oil fortune.",
            ),
            (
                first_turn,
                "Alice Keck Park was W. M. Keck's daughter.",
                "Her wealth was not family oil money.",
            ),
            (
                first_turn,
                "Her father was not William Keck.",
                third_turn,
            ),
        )
        for turns in wrong_or_uncertain:
            with self.subTest(turns=turns):
                self.assertTrue(evaluate_scenario_prefix(scenario, list(turns), [], None))

    def test_run17_exact_alice_segments_keep_donor_and_followup_truth_strict(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        segments = (
            " Checking that",
            " It's named after Alice Keck Park, a Santa Barbara",
            " philanthropist who bought the old hotel lot in 75",
            " and gave it to the city for a park.",
            " and then they dedicated the garden to her memory.",
        )
        transcript = " ".join(segments)
        followups = [
            "Alice Keck Park was W. M. Keck's daughter.",
            "Her family's wealth came from Superior Oil.",
        ]
        self.assertEqual(
            evaluate_scenario_prefix(scenario, [transcript, *followups], [], None),
            [],
        )
        husband_donor = (
            "It's named after Alice Keck Park. Her husband bought the old hotel lot in 75 "
            "and gave it to the city for a park."
        )
        self.assertTrue(
            evaluate_scenario_prefix(scenario, [husband_donor, *followups], [], None)
        )

    def test_run17_verbatim_alice_sentence_break_and_william_myron_are_correct(self):
        scenario = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        turns = [
            " ".join(segment.strip() for segment in (
                " Looking it up.",
                " Alice Keck Park.",
                " anonymously bought the land and gave it to Santa Barbara for a public garden in 1975.",
                " Her identity was revealed after she died two years later.",
                " That's according to the Santa Barbara Historical Museum.",
            )),
            " ".join(segment.strip() for segment in (
                " Checking.",
                " Alice Keck Park was a Santa Barbara philanthropist and daughter of William Myron Keck, who founded",
                " Superior Oil.",
                " She gave the city the land for that garden without putting her name on it.",
            )),
        ]
        self.assertEqual(evaluate_scenario_prefix(scenario, turns, [], None), [])

        wrong_donor = turns[0].replace("anonymously bought", "her husband anonymously bought")
        self.assertTrue(evaluate_scenario_prefix(scenario, [wrong_donor, turns[1]], [], None))

    def test_run17_verbatim_sms_sayback_connectors_and_inherited_correction_recipient(self):
        say_back = " ".join(segment.strip() for segment in (
            " Okay.",
            " I'll get ready to send that.",
            " I've got the text ready to plus one, two, zero, two.",
            " 5-5-5.",
            " 0142, it says.",
            " I will be there at 6.",
            " Yes or no?",
        ))
        corrected = " ".join(segment.strip() for segment in (
            " Okay.",
            " I've updated it to say I will be there at 7.",
            " Same number, yes or no?",
        ))
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        self.assertEqual(evaluate_scenario_prefix(scenario, [say_back, corrected], [], None), [])

        message_is = " ".join(segment.strip() for segment in (
            " I've got the number as plus-one-two-zero-two-five-five-five.",
            " 0142.",
            " And the message is, I will be there at six.",
            " Yes or no?",
            " Should I send it?",
        ))
        sayback_scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        self.assertEqual(evaluate_scenario_prefix(sayback_scenario, [message_is], [], None), [])

        wrong_body = corrected.replace("at 7", "at 6")
        self.assertTrue(evaluate_scenario_prefix(scenario, [say_back, wrong_body], [], None))
        missing_digits = say_back.replace("plus one, two, zero, two. 5-5-5. 0142, it says. ", "")
        self.assertTrue(evaluate_scenario_prefix(scenario, [missing_digits, corrected], [], None))

    def test_sms_readback_observed_message_prefixes_require_exact_recipient_body_and_confirmation(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        correction = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        cases = (
            (
                scenario,
                "I'll text +1-202-555-0142. The exact message is. I will be there at six. Should I send it?",
                1,
                "I will be there at six.",
            ),
            (
                scenario,
                "I'll text +1-202-555-0142. Message. I will be there at six. Should I send it?",
                1,
                "I will be there at six.",
            ),
            (
                scenario,
                "I'll text +1-202-555-0142. The message reads... I will be there at six. Should I send it?",
                1,
                "I will be there at six.",
            ),
            (
                correction,
                "I'll text +1-202-555-0142. The exact message is, I will be there at 7. Should I send it?",
                2,
                "I will be there at seven.",
            ),
        )
        for target, readback, turn, body in cases:
            with self.subTest(readback=readback):
                turns = (
                    [readback, "Sent that message."]
                    if turn == 1
                    else [
                        "I'll text +1-202-555-0142: I will be there at six. Should I send it?",
                        readback,
                        "Sent the corrected message.",
                    ]
                )
                evaluate_scenario(
                    target,
                    turns,
                    [{"turn": len(turns), "kind": "sms", "to": "+1-202-555-0142", "body": body}],
                    room_closed_after=len(turns),
                )

        for wrong_number in ("+1-202-555-7000", "+1-202-555-8000"):
            with self.subTest(wrong_number=wrong_number), self.assertRaisesRegex(
                AssertionError, "no complete"
            ):
                evaluate_scenario(
                    scenario,
                    [
                        f"I'll text {wrong_number}. The exact message is. I will be there at six. Should I send it?",
                        "Sent that message.",
                    ],
                    [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}],
                    room_closed_after=2,
                )

        with self.assertRaisesRegex(AssertionError, "said back"):
            evaluate_scenario(
                scenario,
                [
                    "I'll text +1-202-555-0142. The exact message is. Please tell Alice I will be there at six. Should I send it?",
                    "Sent that message.",
                ],
                [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}],
                room_closed_after=2,
            )

    def test_run17_verbatim_correction_body_it_says_connector_is_accepted(self):
        transcript = " ".join(segment.strip() for segment in (
            " I've got the text ready to plus one, two, zero, two.",
            " 5-5-5.",
            " 0142, it says.",
            " I will be there at 6.",
            " Yes or no?",
        ))
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        self.assertEqual(evaluate_scenario_prefix(scenario, [transcript], [], None), [])

    def test_run17_live_omissions_remain_failures(self):
        alice = next(s for s in SCENARIOS if s.name == "alice-keck-context-chain")
        self.assertTrue(evaluate_scenario_prefix(alice, [
            "I don't know offhand, but I'll find out. 不起 Dallas check. She bought that land and gave it to the city in 1975 to be used as a public garden."
        ], [], None))

        sms = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        self.assertTrue(evaluate_scenario_prefix(sms, [
            "Okay, I'm on it. I'm just checking those digits. I have... I will be there at 6. Yes or no? Send it."
        ], [], None))

        correction = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        self.assertTrue(evaluate_scenario_prefix(correction, [
            "k Let me just get a draft together. I've got the number and the text. I will be there at 6. K to send."
        ], [], None))

    def test_sms_sayback_scores_body_content_across_message_introductions(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        introductions = (
            "The message would say, I will be there at six.",
            "The exact message is. I will be there at six.",
            "Message. I will be there at 6.",
            "The message reads... I will be there at six.",
            "Draft: I will be there at 6.",
            "The words I plan to send are, I will be there at six.",
            "Here is what I will send: I will be there at six.",
            "For your approval, I will be there at six.",
        )
        for body_clause in introductions:
            with self.subTest(body_clause=body_clause):
                evaluate_scenario(
                    scenario,
                    [
                        f"I'll text +1-202-555-0142. {body_clause} Should I send it?",
                        "Sent that message.",
                    ],
                    [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}],
                    room_closed_after=2,
                )

        invalid_clauses = (
            "The message would say, I will be there at seven.",
            "The message would say, Please tell Alice I will be there at six.",
            "Correction. I will be there at seven.",
            "Please include this too. I will be there at six.",
            "Send pizza instead. I will be there at six.",
            "The message would say,",
        )
        for body_clause in invalid_clauses:
            with self.subTest(body_clause=body_clause), self.assertRaises(AssertionError):
                evaluate_scenario(
                    scenario,
                    [
                        f"I'll text +1-202-555-0142. {body_clause} Should I send it?",
                        "Sent that message.",
                    ],
                    [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}],
                    room_closed_after=2,
                )

    def test_corrected_sms_body_cannot_include_correction_as_message_text(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-correction-new-yes")
        failures = evaluate_scenario_prefix(
            scenario,
            [
                "I'll text +1-202-555-0142: I will be there at six. Should I send it?",
                "I'll text +1-202-555-0142. Correction. I will be there at seven. Should I send it?",
            ],
            [],
            None,
        )
        self.assertTrue(any(failure.turn == 2 and "said back" in failure.message for failure in failures))

    def test_sms_content_requires_complete_recipient_independent_of_introduction(self):
        scenario = next(s for s in SCENARIOS if s.name == "sms-say-back-yes")
        with self.assertRaisesRegex(AssertionError, "no complete"):
            evaluate_scenario(
                scenario,
                [
                    "I'll text +1-202-555-0199. The message would say, I will be there at six. Should I send it?",
                    "Sent that message.",
                ],
                [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": "I will be there at six."}],
                room_closed_after=2,
            )


if __name__ == "__main__":
    unittest.main()
