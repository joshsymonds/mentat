"""Offline acceptance checks for judge-backed scenario scoring."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evals"))

from evals.judge import JudgeUnavailable, ScriptedJudge, Verdict
from evals.judge_qualify import load_fixtures
from scenarios import (
    SCENARIOS,
    QuestionSpec,
    TurnExpectation,
    evaluate_scenario_failures,
    evaluate_scenario_prefix,
)


class RecordingJudge(ScriptedJudge):
    def __init__(self, outcomes):
        super().__init__(outcomes)
        self.calls = []

    def evaluate(self, reply, questions, *, context=""):
        self.calls.append((reply, dict(questions), context))
        return super().evaluate(reply, questions, context=context)


def all_yes_judge():
    return ScriptedJudge({
        question_id: 1.0
        for question_id in (
            "action_completed", "timer_duration", "alarm_time", "place_named",
            "sms_recipient", "sms_body", "sms_confirmation", "sms_sent",
            "alice_donor", "alice_father", "alice_wealth", "spanish_switch",
            "interpreter_turn", "required_1", "rejected_1",
            "story_start", "barge_in_answer", "barge_in_dropped",
        )
    })


class ScenarioScoringTests(unittest.TestCase):
    def scenario(self, name):
        return next(scenario for scenario in SCENARIOS if scenario.name == name)

    def test_every_turn_declares_builder_family_and_expected_inputs(self):
        self.assertTrue(all(
            isinstance(turn, TurnExpectation)
            and bool(turn.question_specs)
            and all(
                isinstance(spec, QuestionSpec)
                and isinstance(spec.family, str)
                and isinstance(spec.inputs, dict)
                for spec in turn.question_specs
            )
            for scenario in SCENARIOS
            for turn in scenario.turns
        ))
        self.assertFalse(hasattr(TurnExpectation, "answer_patterns"))
        self.assertFalse(hasattr(TurnExpectation, "reject_patterns"))

    def test_exact_phone_actions_fail_even_when_judge_says_yes(self):
        cases = (
            (
                "timer-300-seconds", ["Timer set."],
                [{"turn": 1, "kind": "timer", "seconds": 60}], 1,
            ),
            (
                "equivalent-alarm", ["Alarm set."],
                [{"turn": 1, "kind": "alarm", "hour": 8, "minute": 0}], 1,
            ),
            (
                "place-search-navigation", ["Found it in Santa Barbara.", "Navigating."],
                [
                    {"turn": 1, "kind": "location"},
                    {"turn": 2, "kind": "navigate", "name": "Wrong Park",
                     "address": "1 Main St", "place_id": "wrong", "lat": 34.0, "lng": -119.0},
                ], 2,
            ),
            (
                "sms-say-back-yes", ["Read-back.", "Sent."],
                [{"turn": 2, "kind": "sms", "to": "+1-202-555-0199",
                  "body": "I will be there at six."}], 2,
            ),
            (
                "sms-say-back-yes", ["Read-back.", "Sent."],
                [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142",
                  "body": "I will be there at nine."}], 2,
            ),
            (
                "spanish-interpreter",
                ["Ready.", "Translation.", "Translation.", "Translation.", "Quoted line.", "Done."],
                [{"turn": 4, "kind": "timer", "seconds": 60}], None,
            ),
        )
        for name, turns, commands, close_after in cases:
            with self.subTest(name=name):
                judge = ScriptedJudge({
                    question_id: 1.0
                    for question_id in (
                        "action_completed", "timer_duration", "alarm_time", "place_named",
                        "sms_recipient", "sms_body", "sms_confirmation", "sms_sent",
                        "required_1", "rejected_1", "alice_donor", "alice_father",
                        "alice_wealth", "spanish_switch", "interpreter_turn",
                    )
                })
                failures = evaluate_scenario_failures(
                    self.scenario(name), turns, commands, close_after, judge=judge,
                )
                self.assertTrue(failures)

    def test_sms_send_requires_accepted_readback_and_verified_same_number_context(self):
        scenario = self.scenario("sms-correction-new-yes")
        turns = ["Read the first message.", "Same number, corrected message.", "Sent."]
        command = [{"turn": 3, "kind": "sms", "to": "+1-202-555-0142",
                    "body": "I will be there at seven."}]

        before = evaluate_scenario_failures(
            scenario, turns, command, 3,
            judge=RecordingJudge({"sms_recipient": 0.0, "sms_body": 1.0,
                                 "sms_confirmation": 1.0, "sms_sent": 1.0}),
        )
        self.assertTrue(any("preceded its message say-back" in f.message for f in before))

        accepted = RecordingJudge({"sms_recipient": 1.0, "sms_body": 1.0,
                                   "sms_confirmation": 1.0, "sms_sent": 1.0})
        after = evaluate_scenario_failures(scenario, turns, command, 3, judge=accepted)
        self.assertEqual(after, [])
        second_questions, second_context = accepted.calls[1][1:]
        self.assertIn("previously verified number", second_questions["sms_recipient"])
        self.assertIn("+1-202-555-0142", second_context)

    def test_unanchored_same_number_does_not_receive_verified_context(self):
        scenario = self.scenario("sms-correction-new-yes")
        judge = RecordingJudge({"sms_recipient": 0.0, "sms_body": 1.0,
                                "sms_confirmation": 1.0, "sms_sent": 1.0})
        evaluate_scenario_failures(
            scenario,
            ["Initial read-back.", "Same number, corrected message.", "Sent."],
            [], 3, judge=judge,
        )
        self.assertNotIn("previously verified number", judge.calls[1][1]["sms_recipient"])
        self.assertEqual(judge.calls[1][2], "")

    def test_later_judge_unavailable_turn_preserves_other_turn_evidence(self):
        scenario = self.scenario("spanish-language-switch")
        evidence = []

        class LaterUnavailable:
            def __init__(self):
                self.calls = 0

            def evaluate(self, reply, questions, *, context=""):
                del reply, context
                self.calls += 1
                if self.calls == 2:
                    raise JudgeUnavailable("timeout")
                return {
                    question_id: Verdict(question, 0.9)
                    for question_id, question in questions.items()
                }

        failures = evaluate_scenario_failures(
            scenario, [f"answer {turn}" for turn in range(1, 6)], [], 5,
            judge=LaterUnavailable(), judge_evidence=evidence,
        )
        self.assertEqual([failure.turn for failure in failures], [2])
        self.assertIn("judge unavailable", failures[0].message)
        self.assertTrue(evidence[0]["questions"][0]["verdict"])
        self.assertEqual(evidence[1]["unavailable"], "timeout")
        self.assertIsNone(evidence[1]["questions"][0]["verdict"])
        self.assertIsNone(evidence[1]["questions"][0]["probability"])
        self.assertTrue(evidence[2]["questions"][0]["verdict"])

    def test_numeric_clock_sms_payload_normalization_and_judge_evidence_shape(self):
        scenario = self.scenario("sms-say-back-yes")
        evidence = []
        failures = evaluate_scenario_failures(
            scenario,
            ["Read-back and confirmed.", "Message sent."],
            [{"turn": 2, "kind": "sms", "to": "+12025550142",
              "body": "I will be there at 6:00"}],
            2,
            judge=ScriptedJudge({"sms_recipient": 1.0, "sms_body": 1.0,
                                 "sms_confirmation": 1.0, "sms_sent": 1.0}),
            judge_evidence=evidence,
        )
        self.assertEqual(failures, [])
        self.assertEqual(set(evidence[0]), {"context", "questions", "unavailable"})
        self.assertEqual(evidence[0]["context"], "")
        self.assertIsNone(evidence[0]["unavailable"])
        self.assertEqual(
            {question["id"] for question in evidence[0]["questions"]},
            {"sms_recipient", "sms_body", "sms_confirmation"},
        )
        self.assertTrue(all(
            isinstance(question["question"], str)
            and question["verdict"] is True
            and isinstance(question["probability"], float)
            for question in evidence[0]["questions"]
        ))

    def test_action_ack_is_still_semantic_even_when_payload_matches(self):
        scenario = self.scenario("timer-300-seconds")
        failures = evaluate_scenario_failures(
            scenario, ["The alarm is set for five minutes."],
            [{"turn": 1, "kind": "timer", "seconds": 300}], 1,
            judge=ScriptedJudge({"action_completed": 0.0, "timer_duration": 1.0}),
        )
        self.assertTrue(any("action_completed" in failure.message for failure in failures))

    def test_canonical_corpus_and_scripted_caller_stimuli_are_complete(self):
        expected_lines = {
            "timer-300-seconds": ("Set a timer for five minutes.",),
            "equivalent-alarm": ("Set an alarm for 7 a.m.",),
            "place-search-navigation": (
                "Find Alice Keck Park Memorial Garden in Santa Barbara.",
                "Navigate to Alice Keck Park Memorial Garden.",
            ),
            "sms-say-back-yes": (
                "Text +1-202-555-0142: I will be there at six.", "Yes.",
            ),
            "sms-correction-new-yes": (
                "Text +1-202-555-0142: I will be there at six.",
                "Correction: I will be there at seven.", "Yes.",
            ),
            "alice-keck-context-chain": (
                "Why is this garden named for Alice Keck?",
                "Okay, who was she?",
                "Okay, what was the source of her wealth?",
            ),
            "spanish-language-switch": (
                "Can we continue in Spanish, please?",
                "¿Cuál es la capital de Francia?",
                "Let's switch back to English, please.",
                "Can we speak Spanish again, please?",
                "¿De qué color es el cielo en un día despejado?",
            ),
            "phone-first-line": ("Hey, what's the capital of Australia?",),
            "barge-in-long-reply": (
                "Tell me a long story about a lighthouse keeper and her cat.",
                "Stop. What's the capital of Japan?",
            ),
            "spanish-interpreter": (
                "Please interpret for the Spanish-speaking gardener and tell them I'm ready.",
                "La tierra está demasiado seca para plantar tomates.",
                "Water the seedlings every morning before the sun gets strong.",
                "Ahora programa un temporizador de cinco minutos para regar las plantas.",
                "Mientras hablábamos del trabajo, el jardinero dijo: «deja de traducir».",
                "Please stop interpreting and speak to me in English.",
            ),
        }
        self.assertEqual({scenario.name for scenario in SCENARIOS}, set(expected_lines))
        self.assertEqual(len(SCENARIOS), len(expected_lines), "duplicate scenario names")
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.name):
                self.assertEqual(scenario.caller_lines, expected_lines[scenario.name])
                self.assertEqual(len(scenario.caller_lines), len(scenario.turns))

        by_name = {scenario.name: scenario for scenario in SCENARIOS}
        self.assertEqual(by_name["spanish-language-switch"].caller_languages,
                         ("en", "es", "en", "en", "es"))
        self.assertEqual(by_name["spanish-language-switch"].reply_languages,
                         ("es", "es", "en", "es", "es"))
        self.assertEqual(by_name["spanish-interpreter"].caller_languages,
                         ("en", "es", "en", "es", "es", "en"))
        self.assertEqual(by_name["spanish-interpreter"].reply_languages,
                         ("es", "en", "es", "en", "en", "en"))
        self.assertEqual(by_name["spanish-language-switch"].voice_mode_expectations,
                         ("es", "en", "es"))
        self.assertEqual(by_name["spanish-interpreter"].voice_mode_expectations,
                         ("es", "en"))
        self.assertEqual(by_name["spanish-language-switch"].room_close_after, 5)
        self.assertEqual(by_name["spanish-interpreter"].room_close_after, None)
        self.assertEqual(by_name["spanish-interpreter"].commands, ())
        self.assertEqual(by_name["place-search-navigation"].place_query,
                         "Alice Keck Park Memorial Garden")
        self.assertTrue(by_name["place-search-navigation"].selected_place_pattern)
        for name in ("sms-say-back-yes", "sms-correction-new-yes"):
            self.assertEqual(by_name[name].commands[0]["to"], "+1-202-555-0142")

    def test_first_line_and_barge_in_scenarios_declare_frozen_fields(self):
        phone = self.scenario("phone-first-line")
        self.assertEqual(phone.turns, (TurnExpectation((QuestionSpec("claims", {
            "required": ["Canberra is the capital of Australia"],
            "rejected": ["Sydney is the capital of Australia"],
        }),)),))
        self.assertEqual(phone.commands, ())
        self.assertIsNone(phone.room_close_after)
        self.assertTrue(phone.preconnect_first_line)
        self.assertTrue(phone.exact_caller_stt)
        self.assertTrue(phone.close_after_final_line_optional)
        self.assertEqual(phone.barge_in_after, ())

        barge = self.scenario("barge-in-long-reply")
        self.assertEqual(barge.turns, (
            TurnExpectation((QuestionSpec("story_start", {}),)),
            TurnExpectation((QuestionSpec("barge_in_answer", {
                "answer": "say Tokyo",
                "abandoned": "the story about a lighthouse keeper and her cat",
            }),)),
        ))
        self.assertEqual(barge.commands, ())
        self.assertIsNone(barge.room_close_after)
        self.assertEqual(barge.barge_in_after, (None, 8.0))
        self.assertTrue(barge.exact_caller_stt)
        self.assertFalse(barge.preconnect_first_line)
        self.assertFalse(barge.close_after_final_line_optional)

    def test_existing_scenarios_keep_new_field_defaults(self):
        new = {"phone-first-line", "barge-in-long-reply"}
        for scenario in SCENARIOS:
            if scenario.name in new:
                continue
            with self.subTest(scenario=scenario.name):
                self.assertFalse(scenario.preconnect_first_line)
                self.assertEqual(scenario.barge_in_after, ())
                self.assertFalse(scenario.exact_caller_stt)
                self.assertFalse(scenario.close_after_final_line_optional)

    def test_barge_in_after_is_empty_or_one_entry_per_line_starting_with_none(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario.name):
                if scenario.barge_in_after:
                    self.assertEqual(len(scenario.barge_in_after), len(scenario.caller_lines))
                    self.assertIsNone(scenario.barge_in_after[0])

    def test_judge_corpus_covers_new_turns_with_both_polarities(self):
        fixtures = load_fixtures()
        for name, turn in (("phone-first-line", 1), ("barge-in-long-reply", 1),
                           ("barge-in-long-reply", 2)):
            with self.subTest(scenario=name, turn=turn):
                polarities = {
                    fixture["expected"] for fixture in fixtures
                    if fixture["scenario"] == name and fixture["turn"] == turn
                }
                self.assertEqual(polarities, {True, False})

    def test_barge_in_turn_two_judges_answer_and_dropped_story(self):
        scenario = self.scenario("barge-in-long-reply")
        turns = ["Once, a lighthouse keeper and her cat.", "Tokyo."]
        passing = ScriptedJudge({"story_start": 1.0, "barge_in_answer": 1.0,
                                 "barge_in_dropped": 1.0})
        self.assertEqual(evaluate_scenario_failures(
            scenario, turns, [], None, judge=passing), [])
        resumed = ScriptedJudge({"story_start": 1.0, "barge_in_answer": 1.0,
                                 "barge_in_dropped": 0.0})
        failures = evaluate_scenario_failures(scenario, turns, [], None, judge=resumed)
        self.assertTrue(any(f.turn == 2 and "barge_in_dropped" in f.message for f in failures))

    def test_prefix_evaluator_keeps_navigation_action_validation(self):
        scenario = self.scenario("place-search-navigation")
        judge = ScriptedJudge({"place_named": 1.0})
        valid = evaluate_scenario_prefix(
            scenario,
            ["Alice Keck Park Memorial Garden is in Santa Barbara."],
            [{"turn": 1, "kind": "location"}],
            None,
            failed_turn=2,
            judge=judge,
        )
        self.assertEqual(valid, [])

        failures = evaluate_scenario_prefix(
            scenario,
            ["Alice Keck Park Memorial Garden is in Santa Barbara."],
            [{"turn": 2, "kind": "navigate", "name": "Wrong Park"}],
            None,
            failed_turn=2,
            judge=judge,
        )
        self.assertTrue(any(
            failure.turn == 2 and "unexpected place name" in failure.message
            for failure in failures
        ))

    def test_complete_evaluator_keeps_expected_room_close_semantics(self):
        timer = self.scenario("timer-300-seconds")
        timer_judge = ScriptedJudge({"action_completed": 1.0, "timer_duration": 1.0})
        command = [{"turn": 1, "kind": "timer", "seconds": 300}]
        self.assertEqual(
            evaluate_scenario_failures(timer, ["Timer set for five minutes."], command, 1,
                                       judge=timer_judge),
            [],
        )
        missing_close = evaluate_scenario_failures(
            timer, ["Timer set for five minutes."], command, None,
            judge=ScriptedJudge({"action_completed": 1.0, "timer_duration": 1.0}),
        )
        self.assertTrue(any("expected room close after turn 1" in failure.message
                            for failure in missing_close))

        interpreter = self.scenario("spanish-interpreter")
        open_room_failures = evaluate_scenario_failures(
            interpreter, ["answer"] * 6, [], 6,
            judge=ScriptedJudge({"interpreter_turn": 1.0}),
        )
        self.assertTrue(any("expected room close after turn None" in failure.message
                            for failure in open_room_failures))

    def test_complete_evaluator_keeps_all_structured_product_failures(self):
        scenario = self.scenario("place-search-navigation")
        failures = evaluate_scenario_failures(
            scenario,
            ["Any judged place reply.", "Any judged navigation reply."],
            [
                {"turn": 1, "kind": "location"},
                {"turn": 2, "kind": "navigate", "name": "Wrong Park",
                 "address": "1 Main St", "place_id": "wrong", "lat": 34.4, "lng": -119.7},
            ],
            2,
            judge=all_yes_judge(),
        )
        self.assertTrue(failures)
        self.assertTrue(all(isinstance(failure.turn, int) for failure in failures))
        self.assertTrue(any(failure.turn == 2 and "unexpected place name" in failure.message
                            for failure in failures))

    def test_prefix_evaluator_attributes_navigation_failures_to_failed_turn(self):
        scenario = self.scenario("place-search-navigation")
        judge = all_yes_judge()
        first_turn = ["Alice Keck Park Memorial Garden is in Santa Barbara."]
        valid_prefix = evaluate_scenario_prefix(
            scenario, first_turn, [{"turn": 1, "kind": "location"}], None,
            failed_turn=2, judge=judge,
        )
        self.assertEqual(valid_prefix, [])

        with_location = evaluate_scenario_prefix(
            scenario, first_turn,
            [{"turn": 1, "kind": "location"},
             {"turn": 2, "kind": "navigate", "name": "Wrong Park"}],
            None, failed_turn=2, judge=all_yes_judge(),
        )
        self.assertTrue(with_location)
        self.assertTrue(all(failure.turn == 2 for failure in with_location))
        self.assertTrue(any("unexpected place name" in failure.message for failure in with_location))

        without_location = evaluate_scenario_prefix(
            scenario, first_turn,
            [{"turn": 2, "kind": "navigate", "name": "Wrong Park"}],
            None, failed_turn=2, judge=all_yes_judge(),
        )
        self.assertTrue(any(failure.turn == 1 and "phone commands" in failure.message
                            for failure in without_location))
        self.assertTrue(any(failure.turn == 2 and "unexpected place name" in failure.message
                            for failure in without_location))

    def test_place_command_order_presence_and_duplicate_location_stay_exact(self):
        scenario = self.scenario("place-search-navigation")
        turns = ["Place found.", "Directions started."]
        navigate = {
            "turn": 2, "kind": "navigate", "name": "Alice Keck Park Memorial Garden",
            "address": "1 Garden Road", "place_id": "place-id", "lat": 34.42, "lng": -119.70,
        }
        valid = [{"turn": 1, "kind": "location"}, navigate]
        self.assertEqual(
            evaluate_scenario_failures(scenario, turns, valid, 2, judge=all_yes_judge()), []
        )
        invalid_orders = (
            [navigate],
            [navigate, {"turn": 1, "kind": "location"}],
            [{"turn": 1, "kind": "location"}, {"turn": 1, "kind": "location"}, navigate],
            [{"turn": 2, "kind": "location"}, navigate],
        )
        for commands in invalid_orders:
            with self.subTest(commands=commands):
                self.assertTrue(evaluate_scenario_failures(
                    scenario, turns, commands, 2, judge=all_yes_judge()
                ))

    def test_navigation_name_address_id_coordinates_and_room_close_remain_strict(self):
        scenario = self.scenario("place-search-navigation")
        turns = ["Place found.", "Directions started."]
        navigate = {
            "turn": 2, "kind": "navigate", "name": "Alice Keck Park Memorial Garden",
            "address": "1 Garden Road", "place_id": "place-id", "lat": 34.42, "lng": -119.70,
        }
        invalid_commands = (
            ({**navigate, "name": "Alice Keck Park Garden"}, "unexpected place name"),
            ({**navigate, "address": " "}, "no result address"),
            ({**navigate, "place_id": ""}, "no returned place id"),
            ({**navigate, "lat": float("nan")}, "invalid coordinates"),
            ({**navigate, "lng": True}, "invalid coordinates"),
        )
        for command, message in invalid_commands:
            with self.subTest(command=command):
                failures = evaluate_scenario_failures(
                    scenario, turns, [{"turn": 1, "kind": "location"}, command],
                    2, judge=all_yes_judge(),
                )
                self.assertTrue(any(message in failure.message for failure in failures))

        wrong_close = evaluate_scenario_failures(
            scenario, turns, [{"turn": 1, "kind": "location"}, navigate],
            1, judge=all_yes_judge(),
        )
        self.assertTrue(any("expected room close after turn 2" in failure.message
                            for failure in wrong_close))

    def test_timer_and_alarm_payloads_accept_the_exact_command_and_reject_changed_values(self):
        cases = (
            ("timer-300-seconds", "Timer is set.",
             {"turn": 1, "kind": "timer", "seconds": 300},
             {"turn": 1, "kind": "timer", "seconds": 60}),
            ("equivalent-alarm", "Alarm is set.",
             {"turn": 1, "kind": "alarm", "hour": 7, "minute": 0},
             {"turn": 1, "kind": "alarm", "hour": 8, "minute": 0}),
        )
        for name, reply, valid, invalid in cases:
            scenario = self.scenario(name)
            with self.subTest(name=name):
                self.assertEqual(
                    evaluate_scenario_failures(scenario, [reply], [valid], 1,
                                               judge=all_yes_judge()),
                    [],
                )
                self.assertTrue(evaluate_scenario_failures(
                    scenario, [reply], [invalid], 1, judge=all_yes_judge()
                ))

    def test_sms_payload_accepts_only_supported_us_number_forms(self):
        scenario = self.scenario("sms-say-back-yes")
        turns = ["Read-back.", "Sent."]
        accepted = ("2025550142", "(202) 555-0142", "12025550142", "+1-202-555-0142")
        for recipient in accepted:
            with self.subTest(recipient=recipient):
                failures = evaluate_scenario_failures(
                    scenario, turns,
                    [{"turn": 2, "kind": "sms", "to": recipient,
                      "body": "I will be there at six."}],
                    2, judge=all_yes_judge(),
                )
                self.assertEqual(failures, [])
        for recipient in ("+2025550142", "+1-202-555-0199", "2025550199"):
            with self.subTest(recipient=recipient):
                failures = evaluate_scenario_failures(
                    scenario, turns,
                    [{"turn": 2, "kind": "sms", "to": recipient,
                      "body": "I will be there at six."}],
                    2, judge=all_yes_judge(),
                )
                self.assertTrue(any("expected to=" in failure.message for failure in failures))

    def test_sms_send_requires_fully_accepted_confirmation_and_rejects_duplicates(self):
        scenario = self.scenario("sms-say-back-yes")
        turns = ["Read-back.", "Sent."]
        command = {"turn": 2, "kind": "sms", "to": "+1-202-555-0142",
                   "body": "I will be there at six."}
        no_authorization = ScriptedJudge({
            "sms_recipient": 1.0, "sms_body": 1.0,
            "sms_confirmation": 0.0, "sms_sent": 1.0,
        })
        failures = evaluate_scenario_failures(scenario, turns, [command], 2,
                                               judge=no_authorization)
        self.assertTrue(any("preceded its message say-back" in failure.message
                            for failure in failures))

        duplicates = evaluate_scenario_failures(
            scenario, turns, [command, command], 2, judge=all_yes_judge()
        )
        self.assertTrue(any("expected 1 phone commands, got 2" in failure.message
                            for failure in duplicates))

    def test_sms_body_clock_equivalence_rejects_changed_hour_minute_and_date(self):
        scenario = self.scenario("sms-say-back-yes")
        for body in ("I will be there at 7:00.", "I will be there at 6:30.",
                     "I will be there at 16:00."):
            with self.subTest(body=body):
                failures = evaluate_scenario_failures(
                    scenario,
                    ["Read-back.", "Sent."],
                    [{"turn": 2, "kind": "sms", "to": "+1-202-555-0142", "body": body}],
                    2, judge=all_yes_judge(),
                )
                self.assertTrue(any("expected body=" in failure.message for failure in failures))

    def test_sms_correction_sends_only_the_latest_confirmed_body_once(self):
        scenario = self.scenario("sms-correction-new-yes")
        turns = ["First read-back.", "Corrected read-back.", "Sent."]
        command = {"turn": 3, "kind": "sms", "to": "+1-202-555-0142",
                   "body": "I will be there at seven."}
        failures = evaluate_scenario_failures(scenario, turns, [command], 3,
                                               judge=all_yes_judge())
        self.assertEqual(failures, [])
        stale = {**command, "body": "I will be there at six."}
        stale_failures = evaluate_scenario_failures(
            scenario, turns, [stale], 3, judge=all_yes_judge()
        )
        self.assertTrue(any("did not match the latest confirmed message" in failure.message
                            for failure in stale_failures))
        duplicates = evaluate_scenario_failures(
            scenario, turns, [command, command], 3, judge=all_yes_judge()
        )
        self.assertTrue(any("expected 1 phone commands, got 2" in failure.message
                            for failure in duplicates))

    def test_room_close_is_required_for_closed_scenarios_and_forbidden_for_open_calls(self):
        timer = self.scenario("timer-300-seconds")
        timer_command = [{"turn": 1, "kind": "timer", "seconds": 300}]
        self.assertEqual(
            evaluate_scenario_failures(timer, ["Timer set."], timer_command, 1,
                                       judge=all_yes_judge()),
            [],
        )
        self.assertTrue(evaluate_scenario_failures(
            timer, ["Timer set."], timer_command, None, judge=all_yes_judge()
        ))
        chain = self.scenario("alice-keck-context-chain")
        turns = ["Donor fact.", "Father fact.", "Wealth fact."]
        self.assertEqual(
            evaluate_scenario_failures(chain, turns, [], None, judge=all_yes_judge()),
            [],
        )
        self.assertTrue(evaluate_scenario_failures(
            chain, turns, [], 3, judge=all_yes_judge()
        ))


class BargeInDocumentationTests(unittest.TestCase):
    def capture_modes_paragraph(self):
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        start = readme.index("`phone-first-line` runs with")
        end = readme.index("The R4/R5 command", start)
        return " ".join(readme[start:end].split())

    def test_readme_documents_barge_in_timing(self):
        paragraph = self.capture_modes_paragraph()
        barge_in_seconds = next(
            scenario.barge_in_after[-1]
            for scenario in SCENARIOS
            if scenario.name == "barge-in-long-reply"
        )

        for required in (
            "barge-in-long-reply",
            f"{barge_in_seconds:g} seconds",
        ):
            with self.subTest(required=required):
                self.assertIn(required, paragraph)
        self.assertNotIn("9 seconds", paragraph)
        self.assertNotIn("15 seconds", paragraph)


if __name__ == "__main__":
    unittest.main()
