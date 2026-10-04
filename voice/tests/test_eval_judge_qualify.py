"""Offline qualification tests for the semantic reply judge."""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.judge import JudgeUnavailable, ScriptedJudge
from evals.judge_qualify import (
    FIXTURE_PATH,
    load_fixtures,
    qualify_fixtures,
    qualification_exit_code,
)
from evals.questions import build_questions


class QualificationTests(unittest.TestCase):
    def test_public_corpus_covers_registered_families_and_required_turns(self):
        fixtures = load_fixtures(FIXTURE_PATH)
        self.assertEqual(
            {fixture["family"] for fixture in fixtures},
            {
                "action_ack", "timer_duration", "alarm_time", "place_lookup",
                "sms_say_back", "sms_send_ack", "sms_confirmation", "claims",
                "alice_donor", "alice_father", "alice_wealth", "spanish_switch",
                "interpreter_turn",
            },
        )
        for family, first, last in (("spanish_switch", 1, 5), ("interpreter_turn", 1, 6)):
            self.assertEqual(
                {
                    fixture["inputs"]["turn"]
                    for fixture in fixtures
                    if fixture["family"] == family and fixture["expected"]
                },
                set(range(first, last + 1)),
            )
        by_id = {fixture["id"]: fixture for fixture in fixtures}
        for fixture in fixtures:
            with self.subTest(fixture=fixture["id"]):
                self.assertIn("reply", fixture)
                self.assertTrue(fixture["reply"].strip())
                self.assertIsInstance(fixture["expected"], bool)
                self.assertIsInstance(fixture["source"], dict)
                self.assertEqual(build_questions(fixture["family"], fixture["inputs"]), fixture["questions"])
                if fixture["expected"]:
                    self.assertIn(fixture["source"]["kind"], {"transcript", "trace_segment_join", "record_segment"})
                else:
                    self.assertEqual(fixture["source"]["kind"], "scenario_corruption")
                    self.assertIn(fixture["source"]["based_on"], by_id)
                    self.assertTrue(fixture["source"]["change"])
        required_tags = {
            "initial-sms", "corrected-sms", "inherited-recipient", "unanchored-sms",
            "six-oclock-equivalence", "missing-confirmation", "wrong-digits", "wrong-body",
            "wrong-time", "wrong-amount", "wrong-attribution", "retained-2026-10-03",
        }
        self.assertTrue(required_tags.issubset({tag for fixture in fixtures for tag in fixture["tags"]}))
        for family in {fixture["family"] for fixture in fixtures}:
            family_fixtures = [fixture for fixture in fixtures if fixture["family"] == family]
            self.assertTrue(any(fixture["expected"] for fixture in family_fixtures))
            self.assertTrue(any(not fixture["expected"] for fixture in family_fixtures))

    def test_interpreter_turn_one_corpus_reply_excludes_private_context(self):
        fixture = next(
            item for item in load_fixtures(FIXTURE_PATH)
            if item["id"] == "interpreter-turn-1"
        )
        self.assertNotIn("Josh", fixture["reply"])
        self.assertTrue(fixture["expected"])
        self.assertEqual(fixture["reply"], "[[es]] Estoy listo.")
        self.assertEqual(fixture["source"]["kind"], "record_segment")
        self.assertEqual(
            fixture["source"]["artifact"],
            "/tmp/mentat-voice-eval-869o6jy0/records/voice-android-986b96b0-8407-429c-9cd6-00887716702a.jsonl",
        )
        self.assertEqual(fixture["source"]["line"], 31)
        self.assertEqual(fixture["source"]["block"], 0)

    def test_sms_corpus_retains_saying_exactly_source_wording(self):
        fixture = next(
            item for item in load_fixtures(FIXTURE_PATH)
            if item["id"] == "sms-initial-six-oclock"
        )
        self.assertEqual(fixture["source"]["kind"], "transcript")
        self.assertEqual(fixture["source"]["line"], 1)
        self.assertIn("saying exactly", fixture["reply"])

    def test_sms_inheritance_has_prior_context_and_unanchored_same_number_fails(self):
        fixtures = load_fixtures(FIXTURE_PATH)
        inherited = next(item for item in fixtures if item["id"] == "sms-corrected-inherited-recipient")
        unanchored = next(item for item in fixtures if item["id"] == "sms-unanchored-recipient")
        self.assertEqual(
            inherited["inputs"]["verified_recipient"],
            inherited["inputs"]["recipient"],
        )
        self.assertIn("previously confirmed", inherited["context"].lower())
        self.assertIn("same number", unanchored["reply"].lower())
        self.assertFalse(unanchored.get("context", ""))
        self.assertNotIn("verified_recipient", unanchored["inputs"])

    def test_live_recipe_is_opt_in_and_not_part_of_the_default(self):
        root = Path(__file__).resolve().parents[2]
        justfile = (root / "Justfile").read_text(encoding="utf-8")
        default_recipe = next(line for line in justfile.splitlines() if line.startswith("default:"))
        self.assertNotIn("eval-judge", default_recipe)
        self.assertIn("\neval-judge:\n", justfile)
        recipe = justfile.split("\neval-judge:\n", 1)[1].split("\n\n", 1)[0]
        self.assertIn("python3 -m voice.evals.judge_qualify", recipe)

    def test_scripted_judge_qualifies_the_real_corpus_and_reports_per_family_counts(self):
        fixtures = load_fixtures(FIXTURE_PATH)
        result = qualify_fixtures(
            fixtures,
            judge_factory=lambda fixture, _run: ScriptedJudge({
                question_id: (0.91 if fixture["expected"] else 0.09)
                for question_id in fixture["questions"]
            }),
            runs=3,
        )
        self.assertEqual(qualification_exit_code(result), 0)
        self.assertEqual(result["run_count"], 3)
        for run in result["runs"]:
            self.assertEqual(run["status"], "pass")
            self.assertEqual(run["counts"]["correct"], run["counts"]["correct_total"])
            self.assertEqual(run["counts"]["wrong"], run["counts"]["wrong_total"])
            self.assertEqual(run["counts"]["unavailable"], 0)
            self.assertEqual(set(run["family_counts"]), {
                "action_ack", "timer_duration", "alarm_time", "place_lookup", "sms_say_back",
                "sms_send_ack", "sms_confirmation", "claims", "alice_donor", "alice_father",
                "alice_wealth", "spanish_switch", "interpreter_turn",
            })
            for counts in run["family_counts"].values():
                self.assertEqual(counts["correct"], counts["correct_total"])
                self.assertEqual(counts["wrong"], counts["wrong_total"])

    def test_three_runs_keep_per_fixture_questions_verdicts_probabilities_and_no_cache(self):
        fixtures = [
            {
                "id": "good",
                "family": "action_ack",
                "inputs": {"action": "timer", "expected": "five-minute timer"},
                "reply": "Your five minute timer is set.",
                "expected": True,
                "source": {"kind": "test"},
            },
            {
                "id": "bad",
                "family": "action_ack",
                "inputs": {"action": "timer", "expected": "five-minute timer"},
                "reply": "Your alarm is set.",
                "expected": False,
                "source": {"kind": "test"},
            },
        ]
        for fixture in fixtures:
            fixture["questions"] = build_questions(fixture["family"], fixture["inputs"])
        factory_calls = []
        active = 0
        peak = 0
        lock = threading.Lock()
        judge_ids = set()
        judges = []
        overlap_barrier = threading.Barrier(2, timeout=2)

        class RecordingScriptedJudge(ScriptedJudge):
            def __init__(self, outcomes):
                super().__init__(outcomes)
                judges.append(self)
                judge_ids.add(id(self))

            def evaluate(self, reply, questions, *, context=""):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    overlap_barrier.wait()
                    return super().evaluate(reply, questions, context=context)
                finally:
                    with lock:
                        active -= 1

        def factory(fixture, run_number):
            factory_calls.append((fixture["id"], run_number))
            answer = 0.91 if fixture["expected"] else 0.09
            outcomes = {question_id: answer for question_id in fixture["questions"]}
            return RecordingScriptedJudge(outcomes)

        result = qualify_fixtures(fixtures, judge_factory=factory, runs=3)
        self.assertEqual(result["run_count"], 3)
        self.assertEqual([run["counts"] for run in result["runs"]], [
            {"correct": 1, "correct_total": 1, "wrong": 1, "wrong_total": 1, "passed": 2, "failed": 0, "unavailable": 0},
        ] * 3)
        self.assertEqual(len(factory_calls), len(fixtures) * 3)
        self.assertEqual(len(judges), len(factory_calls))
        self.assertEqual(len(judge_ids), len(judges))
        self.assertGreaterEqual(peak, 2)
        for run in result["runs"]:
            self.assertEqual(len(run["fixtures"]), len(fixtures))
            for evidence in run["fixtures"]:
                self.assertTrue(evidence["questions"])
                self.assertEqual(set(evidence["verdicts"]), set(evidence["questions"]))
                self.assertEqual(set(evidence["probabilities"]), set(evidence["questions"]))

    def test_correct_fixture_threshold_is_at_least_ninety_five_percent_per_run(self):
        fixtures = [
            {
                "id": f"correct-{index}", "family": "action_ack",
                "inputs": {"action": "timer", "expected": "five-minute timer"},
                "reply": "Your five minute timer is set.", "expected": True,
                "source": {"kind": "test"},
            }
            for index in range(20)
        ]
        fixtures.append({
            "id": "wrong", "family": "action_ack",
            "inputs": {"action": "timer", "expected": "five-minute timer"},
            "reply": "Your alarm is set.", "expected": False,
            "source": {"kind": "test"},
        })
        for fixture in fixtures:
            fixture["questions"] = build_questions(fixture["family"], fixture["inputs"])

        def factory(fixture, run_number):
            if fixture["expected"] is False:
                probability = 0.1
            elif run_number == 1 and fixture["id"] == "correct-0":
                probability = 0.1
            elif run_number == 2 and fixture["id"] in {"correct-0", "correct-1"}:
                probability = 0.1
            else:
                probability = 0.9
            return ScriptedJudge({
                question_id: probability for question_id in fixture["questions"]
            })

        result = qualify_fixtures(fixtures, judge_factory=factory, runs=3)
        self.assertEqual([run["correct_rate"] for run in result["runs"]], [0.95, 0.9, 1.0])
        self.assertEqual([run["status"] for run in result["runs"]], ["pass", "fail", "pass"])
        self.assertEqual(qualification_exit_code(result), 1)

    def test_always_yes_fails_wrong_fixture_bar_and_unavailability_is_not_a_no(self):
        fixtures = [
            {
                "id": "correct", "family": "action_ack",
                "inputs": {"action": "timer", "expected": "five-minute timer"},
                "reply": "Your five minute timer is set.", "expected": True,
                "source": {"kind": "test"},
            },
            {
                "id": "wrong", "family": "action_ack",
                "inputs": {"action": "timer", "expected": "five-minute timer"},
                "reply": "Your alarm is set.", "expected": False,
                "source": {"kind": "test"},
            },
        ]
        for fixture in fixtures:
            fixture["questions"] = build_questions(fixture["family"], fixture["inputs"])
        always_yes = qualify_fixtures(
            fixtures,
            judge_factory=lambda fixture, run: ScriptedJudge(
                {question_id: 1.0 for question_id in fixture["questions"]}
            ),
            runs=3,
        )
        self.assertEqual(qualification_exit_code(always_yes), 1)
        self.assertEqual([run["counts"]["wrong"] for run in always_yes["runs"]], [0, 0, 0])
        self.assertTrue(all(run["counts"]["failed"] == 1 for run in always_yes["runs"]))

        unavailable = qualify_fixtures(
            fixtures,
            judge_factory=lambda fixture, run: ScriptedJudge(
                {question_id: JudgeUnavailable("offline") for question_id in fixture["questions"]}
            ),
            runs=3,
        )
        self.assertEqual(qualification_exit_code(unavailable), 1)
        for run in unavailable["runs"]:
            self.assertEqual(run["counts"]["unavailable"], 2)
            self.assertEqual(run["counts"]["wrong"], 0)
            self.assertEqual(len(run["fixtures"]), 2)
            self.assertTrue(all(item["error"] for item in run["fixtures"]))

    def test_sms_confirmation_requires_both_questions(self):
        questions = build_questions(
            "sms_confirmation",
            {"recipient": "+1-202-555-0142", "body": "I will be there at 6:00."},
        )
        self.assertEqual(set(questions), {"sms_readback", "sms_confirmation"})
        fixtures = [{
            "id": "confirmation", "family": "sms_confirmation",
            "inputs": {"recipient": "+1-202-555-0142", "body": "I will be there at 6:00."},
            "reply": "I'll text the message. Should I send it?",
            "expected": True, "source": {"kind": "test"}, "questions": questions,
        }]
        result = qualify_fixtures(
            fixtures,
            judge_factory=lambda _fixture, _run: ScriptedJudge({
                "sms_readback": 0.9,
                "sms_confirmation": 0.1,
            }),
            runs=3,
        )
        self.assertEqual([run["counts"]["failed"] for run in result["runs"]], [1, 1, 1])

    def test_all_runs_are_concurrent_and_unavailable_does_not_stop_other_fixture_evidence(self):
        fixtures = [
            {
                "id": f"fixture-{index}", "family": "action_ack",
                "inputs": {"action": "timer", "expected": "five-minute timer"},
                "reply": "Your five minute timer is set.", "expected": True,
                "source": {"kind": "test"},
            }
            for index in range(6)
        ]
        for fixture in fixtures:
            fixture["questions"] = build_questions(fixture["family"], fixture["inputs"])
        barrier = threading.Barrier(18, timeout=2)
        factory_calls = []
        active_runs = set()
        peak_active_runs = 0
        run_lock = threading.Lock()

        class BarrierJudge(ScriptedJudge):
            def __init__(self, outcomes, run_number):
                super().__init__(outcomes)
                self.run_number = run_number

            def evaluate(self, reply, questions, *, context=""):
                nonlocal peak_active_runs
                with run_lock:
                    active_runs.add(self.run_number)
                    peak_active_runs = max(peak_active_runs, len(active_runs))
                try:
                    barrier.wait()
                    return super().evaluate(reply, questions, context=context)
                finally:
                    with run_lock:
                        active_runs.discard(self.run_number)

        def factory(fixture, run):
            factory_calls.append((fixture["id"], run))
            outcomes = {
                question_id: (JudgeUnavailable("offline") if fixture["id"] == "fixture-0" else 0.9)
                for question_id in fixture["questions"]
            }
            return BarrierJudge(outcomes, run)

        result = qualify_fixtures(fixtures, judge_factory=factory, runs=3)
        self.assertEqual(len(factory_calls), len(fixtures) * 3)
        self.assertEqual(peak_active_runs, 3)
        self.assertEqual([run["counts"]["unavailable"] for run in result["runs"]], [1, 1, 1])
        self.assertTrue(all(len(run["fixtures"]) == len(fixtures) for run in result["runs"]))


if __name__ == "__main__":
    unittest.main()
