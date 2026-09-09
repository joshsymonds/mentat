"""Tests for the eval harness's pure half: the scenario schema and the scoring.

voice/evals/run.py is the online part — it imports livekit and spends money, so
it is not imported here. Everything it decides *about* a response once the
response exists lives in voice/evals/scoring.py, which is stdlib-only, and that
is what this file pins: the schema of the real scenario file, the
tool-call-vs-text classification, the hit rule, and the report the runner
prints.

The real scenarios.jsonl is validated here rather than a fixture copy. The file
is hand-edited — the user grows the set — so a typo in it is exactly the
failure this suite exists to catch.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

VOICE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VOICE))
sys.path.insert(0, str(VOICE / "evals"))

from request import CONSULT_WINDOW_TURNS
from scoring import (
    CONSULT,
    DIRECT,
    HISTORY_TURNS,
    INVALID,
    PHONE_TOOL_NAMES,
    SKIP_CATEGORY,
    TOOL_NAME,
    Response,
    Scenario,
    ScenarioError,
    ToolCall,
    accuracy,
    classify,
    exit_code,
    judge,
    load_scenarios,
    parse_scenario,
    per_category,
    report,
)

SCENARIOS_PATH = VOICE / "evals" / "scenarios.jsonl"


def scenario(**overrides):
    """A valid scenario dict, with the field under test overridden."""
    base = {
        "id": "cal-today",
        "category": "calendar",
        "utterance": "what's on my calendar today?",
        "expect": "consult",
        "note": "his own calendar; guessing it is worse than not answering",
    }
    base.update(overrides)
    return base


def consulted(**args):
    """A response that called ask_mentat with the given arguments."""
    return Response(tool_calls=(ToolCall(TOOL_NAME, {"question": "q?", **args}),))


class ScenarioSchemaTest(unittest.TestCase):
    """What a scenario line must look like before it costs a token to run."""

    def test_a_full_scenario_round_trips(self):
        parsed = parse_scenario(
            scenario(
                history=[["user", "morning"], ["assistant", "morning yourself"]],
                expect_args={"effort": "high", "model": "fable"},
            ),
            "line 1",
        )
        self.assertEqual(parsed.id, "cal-today")
        self.assertEqual(parsed.category, "calendar")
        self.assertEqual(parsed.expect, CONSULT)
        self.assertEqual(
            parsed.history, (("user", "morning"), ("assistant", "morning yourself"))
        )
        self.assertEqual(parsed.expect_args, {"effort": "high", "model": "fable"})

    def test_history_and_expect_args_are_optional(self):
        parsed = parse_scenario(scenario(), "line 1")
        self.assertEqual(parsed.history, ())
        self.assertEqual(parsed.expect_args, {})

    def test_a_missing_required_field_is_an_error(self):
        for field in ("id", "category", "utterance", "expect", "note"):
            with self.subTest(field=field):
                incomplete = scenario()
                del incomplete[field]
                with self.assertRaises(ScenarioError) as caught:
                    parse_scenario(incomplete, "line 4")
                self.assertIn(field, str(caught.exception))
                # The location is in the message: the file is hand-edited, and
                # "some scenario is wrong" is not a fixable report.
                self.assertIn("line 4", str(caught.exception))

    def test_an_unknown_field_is_an_error(self):
        # A misspelled key would otherwise silently drop its meaning — an
        # "expected" that never scores, an "expect_arg" that never checks.
        with self.assertRaises(ScenarioError) as caught:
            parse_scenario(scenario(expected="consult"), "line 2")
        self.assertIn("expected", str(caught.exception))

    def test_expect_accepts_consult_direct_and_known_phone_tools(self):
        for expected in (CONSULT, DIRECT, *PHONE_TOOL_NAMES):
            with self.subTest(expected=expected):
                parsed = parse_scenario(scenario(expect=expected), "line 3")
                self.assertEqual(parsed.expect, expected)
        with self.assertRaises(ScenarioError):
            parse_scenario(scenario(expect="maybe"), "line 3")

    def test_expect_rejects_unknown_tool_names(self):
        with self.assertRaises(ScenarioError):
            parse_scenario(scenario(expect="search_web"), "line 3")

    def test_empty_strings_are_errors(self):
        for field in ("id", "category", "utterance", "note"):
            with self.subTest(field=field):
                with self.assertRaises(ScenarioError):
                    parse_scenario(scenario(**{field: "  "}), "line 5")

    def test_history_turns_are_role_text_pairs(self):
        for bad in (
            [["user"]],
            [["user", "hi", "extra"]],
            [["narrator", "hi"]],
            [["user", ""]],
            ["user: hi"],
            {"user": "hi"},
        ):
            with self.subTest(history=bad):
                with self.assertRaises(ScenarioError):
                    parse_scenario(scenario(history=bad), "line 6")

    def test_history_is_capped_at_the_window_the_front_sees(self):
        ok = [["user", "one"], ["assistant", "two"]]
        self.assertEqual(len(parse_scenario(scenario(history=ok), "l").history), 2)
        with self.assertRaises(ScenarioError):
            parse_scenario(scenario(history=[*ok, ["user", "three"]]), "line 7")

    def test_expect_args_keys_are_the_expected_tools_own_knobs(self):
        with self.assertRaises(ScenarioError) as caught:
            parse_scenario(scenario(expect_args={"temperature": "hot"}), "line 8")
        self.assertIn("temperature", str(caught.exception))
        parsed = parse_scenario(
            scenario(expect="find_places", expect_args={"locality": "Beaverton"}),
            "line 8",
        )
        self.assertEqual(parsed.expect_args, {"locality": "Beaverton"})
        with self.assertRaises(ScenarioError):
            parse_scenario(
                scenario(expect="find_places", expect_args={"choice": "2"}),
                "line 8",
            )

    def test_expect_args_on_a_direct_scenario_is_an_error(self):
        # Nothing to check them against: a direct answer calls no tool, so the
        # line would assert something that can never be observed.
        with self.assertRaises(ScenarioError):
            parse_scenario(
                scenario(expect="direct", expect_args={"effort": "high"}), "line 9"
            )

    def test_expect_args_values_must_be_non_empty_strings(self):
        for bad in ({"effort": ""}, {"effort": 3}, {"effort": None}):
            with self.subTest(expect_args=bad):
                with self.assertRaises(ScenarioError):
                    parse_scenario(scenario(expect_args=bad), "line 10")


class LoadScenariosTest(unittest.TestCase):
    def _write(self, text):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        target = directory / "scenarios.jsonl"
        target.write_text(text)
        return target

    def test_reads_one_scenario_per_line(self):
        path = self._write(
            json.dumps(scenario())
            + "\n\n"
            + json.dumps(scenario(id="banter-1", category="banter", expect="direct"))
            + "\n"
        )
        loaded = load_scenarios(path)
        self.assertEqual([s.id for s in loaded], ["cal-today", "banter-1"])

    def test_a_malformed_line_names_its_line_number(self):
        path = self._write('{"id": "a"\n')
        with self.assertRaises(ScenarioError) as caught:
            load_scenarios(path)
        self.assertIn("line 1", str(caught.exception))

    def test_duplicate_ids_are_an_error(self):
        path = self._write(json.dumps(scenario()) + "\n" + json.dumps(scenario()) + "\n")
        with self.assertRaises(ScenarioError) as caught:
            load_scenarios(path)
        self.assertIn("cal-today", str(caught.exception))

    def test_an_empty_file_is_an_error(self):
        with self.assertRaises(ScenarioError):
            load_scenarios(self._write("\n\n"))


class RealScenarioFileTest(unittest.TestCase):
    """The shipped scenario set, checked as data.

    Coverage is asserted by category because the set is meant to grow: the
    point of the file is that it exercises both sides of the consult decision
    across the kinds of question the front actually gets.
    """

    @classmethod
    def setUpClass(cls):
        cls.scenarios = load_scenarios(SCENARIOS_PATH)

    def test_the_shipped_file_parses(self):
        self.assertGreaterEqual(len(self.scenarios), 20)

    def test_both_decisions_are_represented(self):
        expectations = {s.expect for s in self.scenarios if s.scored}
        self.assertEqual(expectations, {CONSULT, DIRECT, *PHONE_TOOL_NAMES})

    def test_phone_category_has_the_required_scenarios(self):
        phone = [s for s in self.scenarios if s.category == "phone"]
        self.assertGreaterEqual(len(phone), 12)
        self.assertTrue(any(s.expect == "find_places" for s in phone))
        self.assertTrue(
            any(
                s.expect == "navigate_to" and s.expect_args == {"choice": "2"}
                for s in phone
            )
        )
        self.assertTrue(
            any(s.expect == "find_places" and s.expect_args.get("locality") for s in phone)
        )

    def test_the_named_categories_are_covered(self):
        # The categories the epic asks for by name; a file that quietly lost
        # one would still score 100% and prove less than it looks like.
        categories = {s.category for s in self.scenarios}
        for required in (
            "calendar",
            "home",
            "memory",
            "messages",
            "current-events",
            "banter",
            "opinion",
            "knowledge",
            "borderline",
            "escalation",
        ):
            self.assertIn(required, categories)

    def test_escalation_scenarios_pin_the_tools_knobs(self):
        escalation = [s for s in self.scenarios if s.category == "escalation"]
        self.assertTrue(escalation)
        for s in escalation:
            with self.subTest(id=s.id):
                self.assertTrue(s.expect_args, s.id)

    def test_the_unscorable_case_is_marked_skip(self):
        # The duplicate-consult behaviour needs session state to judge, so it
        # is run for the record and excluded from the accuracy.
        self.assertTrue(any(s.category == SKIP_CATEGORY for s in self.scenarios))


class ClassifyTest(unittest.TestCase):
    """Tool call versus text — the whole decision the eval measures."""

    def test_an_ask_mentat_call_is_a_consult(self):
        self.assertEqual(classify(consulted()), CONSULT)

    def test_plain_text_is_a_direct_answer(self):
        self.assertEqual(classify(Response(text="Rome, and it's not close.")), DIRECT)

    def test_each_known_phone_tool_classifies_as_its_own_name(self):
        for name in PHONE_TOOL_NAMES:
            with self.subTest(name=name):
                self.assertEqual(classify(Response(tool_calls=(ToolCall(name, {}),))), name)

    def test_a_holding_line_before_the_call_is_still_a_consult(self):
        # The front often says something while reaching for the tool; the text
        # riding along with the call must not read as an answer.
        self.assertEqual(
            classify(Response(text="Let me check.", tool_calls=consulted().tool_calls)),
            CONSULT,
        )

    def test_an_empty_response_is_neither(self):
        # Luna can finish a turn with no content at all (the silent-mute case
        # agent.py warns about). Scoring that as a direct answer would count a
        # mute as a pass.
        self.assertEqual(classify(Response()), INVALID)
        self.assertEqual(classify(Response(text="   ")), INVALID)

    def test_a_response_with_expected_tool_and_another_tool_is_invalid(self):
        self.assertEqual(
            classify(
                Response(
                    tool_calls=(
                        ToolCall("find_places", {"query": "Trader Joe's"}),
                        ToolCall(TOOL_NAME, {"question": "q?"}),
                    )
                )
            ),
            INVALID,
    PHONE_TOOL_NAMES,
        )

    def test_a_call_to_some_other_tool_is_neither(self):
        # Only ask_mentat is declared, so this is a hallucinated name; it is
        # not an answer and it is not a consult.
        self.assertEqual(classify(Response(tool_calls=(ToolCall("search", {}),))), INVALID)

    def test_a_failed_turn_is_neither(self):
        self.assertEqual(classify(Response(text="anything", error="gateway 503")), INVALID)


class JudgeTest(unittest.TestCase):
    CONSULT_CASE = Scenario(
        id="cal-today",
        category="calendar",
        utterance="what's on today?",
        expect=CONSULT,
        note="his calendar",
    )
    DIRECT_CASE = Scenario(
        id="banter-1",
        category="banter",
        utterance="tell me a joke",
        expect=DIRECT,
        note="pure banter",
    )
    ESCALATION_CASE = Scenario(
        id="esc-hard",
        category="escalation",
        utterance="should I take the job?",
        expect=CONSULT,
        note="deserves the best thinking",
        expect_args={"effort": "high", "model": "fable"},
    )
    SKIPPED_CASE = Scenario(
        id="dup-consult",
        category=SKIP_CATEGORY,
        utterance="and what about the house?",
        expect=DIRECT,
        note="needs session state to judge",
    )

    def test_matching_the_expectation_is_a_hit(self):
        self.assertTrue(judge(self.CONSULT_CASE, consulted()).hit)
        self.assertTrue(judge(self.DIRECT_CASE, Response(text="a joke")).hit)

    def test_missing_the_expectation_is_a_miss(self):
        self.assertFalse(judge(self.CONSULT_CASE, Response(text="you're free")).hit)
        self.assertFalse(judge(self.DIRECT_CASE, consulted()).hit)

    def test_an_invalid_response_is_a_miss_either_way(self):
        self.assertFalse(judge(self.CONSULT_CASE, Response()).hit)
        self.assertFalse(judge(self.DIRECT_CASE, Response()).hit)

    def test_expected_phone_arguments_must_match(self):
        phone_case = Scenario(
            id="phone-pick",
            category="phone",
            utterance="the second one",
            expect="navigate_to",
            note="pick the listed place",
            expect_args={"choice": "2"},
        )
        self.assertTrue(
            judge(phone_case, Response(tool_calls=(ToolCall("navigate_to", {"choice": "2"}),))).hit
        )
        self.assertFalse(
            judge(phone_case, Response(tool_calls=(ToolCall("navigate_to", {"choice": "1"}),))).hit
        )

    def test_expected_arguments_must_match(self):
        self.assertTrue(
            judge(
                self.ESCALATION_CASE, consulted(effort="high", model="fable")
            ).hit
        )
        self.assertFalse(
            judge(self.ESCALATION_CASE, consulted(effort="low", model="fable")).hit
        )
        self.assertFalse(
            judge(self.ESCALATION_CASE, consulted(effort="high", model="sonnet")).hit
        )

    def test_a_missing_argument_is_a_miss(self):
        # The tool defaults effort and model, so an omitted argument means the
        # front never chose one — which is exactly what these cases measure.
        self.assertFalse(judge(self.ESCALATION_CASE, consulted(effort="high")).hit)

    def test_only_the_named_arguments_are_checked(self):
        low_effort_only = Scenario(
            id="esc-easy",
            category="escalation",
            utterance="what time is my flight?",
            expect=CONSULT,
            note="a lookup, not a think",
            expect_args={"effort": "low"},
        )
        self.assertTrue(judge(low_effort_only, consulted(effort="low", model="fable")).hit)

    def test_a_skip_category_scenario_is_run_but_not_scored(self):
        result = judge(self.SKIPPED_CASE, consulted())
        self.assertFalse(result.scored)
        self.assertIsNone(result.hit)
        self.assertEqual(result.got, CONSULT)

    def test_the_result_carries_the_classification_and_the_response(self):
        response = Response(text="you're free all day")
        result = judge(self.CONSULT_CASE, response)
        self.assertEqual(result.got, DIRECT)
        self.assertIs(result.response, response)
        self.assertIs(result.scenario, self.CONSULT_CASE)


class ScoringTest(unittest.TestCase):
    def setUp(self):
        self.results = [
            judge(JudgeTest.CONSULT_CASE, consulted()),
            judge(JudgeTest.DIRECT_CASE, Response(text="a joke")),
            judge(JudgeTest.ESCALATION_CASE, consulted(effort="low", model="sonnet")),
            judge(JudgeTest.SKIPPED_CASE, consulted()),
        ]

    def test_accuracy_counts_only_scored_results(self):
        self.assertEqual(accuracy(self.results), (2, 3, 2 / 3))

    def test_accuracy_of_nothing_is_zero_not_a_crash(self):
        self.assertEqual(accuracy([judge(JudgeTest.SKIPPED_CASE, consulted())]), (0, 0, 0.0))

    def test_per_category_splits_the_score(self):
        self.assertEqual(
            per_category(self.results),
            {"calendar": (1, 1), "banter": (1, 1), "escalation": (0, 1)},
        )

    def test_per_category_excludes_the_unscored(self):
        self.assertNotIn(SKIP_CATEGORY, per_category(self.results))

    def test_exit_code_holds_the_bar_the_caller_set(self):
        self.assertEqual(exit_code(self.results, 0.0), 0)
        self.assertEqual(exit_code(self.results, 0.66), 0)
        self.assertEqual(exit_code(self.results, 0.9), 1)

    def test_a_run_that_scored_nothing_fails(self):
        # A run proving nothing must not report success, whatever the bar.
        self.assertEqual(exit_code([judge(JudgeTest.SKIPPED_CASE, consulted())], 0.0), 1)


class ReportTest(unittest.TestCase):
    def setUp(self):
        self.results = [
            judge(JudgeTest.CONSULT_CASE, consulted()),
            judge(JudgeTest.DIRECT_CASE, consulted()),
            judge(
                JudgeTest.ESCALATION_CASE,
                consulted(effort="low", model="sonnet"),
            ),
            judge(JudgeTest.SKIPPED_CASE, consulted()),
        ]
        self.text = report(self.results)

    def test_leads_with_the_overall_score(self):
        self.assertIn("1/3", self.text)
        self.assertIn("33.3%", self.text)

    def test_lists_every_category(self):
        for category in ("calendar", "banter", "escalation"):
            self.assertIn(category, self.text)

    def test_a_miss_names_the_scenario_and_both_sides(self):
        self.assertIn("banter-1", self.text)
        self.assertIn("expected direct", self.text)
        self.assertIn("got consult", self.text)

    def test_a_miss_shows_what_the_model_actually_did(self):
        # The point of the report: the argument that was wrong is in the line,
        # so a miss can be read without re-running the scenario.
        self.assertIn("effort=low", self.text)
        self.assertIn("model=sonnet", self.text)

    def test_a_missed_direct_answer_quotes_the_text(self):
        text = report([judge(JudgeTest.CONSULT_CASE, Response(text="you're free all day"))])
        self.assertIn("you're free all day", text)

    def test_hits_are_not_listed(self):
        self.assertNotIn("cal-today", self.text)

    def test_the_unscored_are_reported_separately(self):
        self.assertIn("dup-consult", self.text)
        self.assertIn("unscored", self.text)

    def test_an_errored_scenario_shows_its_error(self):
        text = report([judge(JudgeTest.CONSULT_CASE, Response(error="gateway 503"))])
        self.assertIn("gateway 503", text)

    def test_the_history_window_matches_the_front(self):
        # A scenario may carry no more history than a real consult sends, or it
        # would measure the front on context the room never gives it. Pinned
        # against the front's own constant rather than a literal, so widening
        # one side alone fails here instead of silently drifting.
        self.assertEqual(HISTORY_TURNS, CONSULT_WINDOW_TURNS)


if __name__ == "__main__":
    unittest.main()
