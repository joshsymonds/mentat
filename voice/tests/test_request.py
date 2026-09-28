"""Offline tests for voice request construction and ending policy."""

import asyncio
import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from request import (
    ATTR_DRIVING,
    ATTR_LOCATION,
    ATTR_TIME_ZONE,
    CLOSE_TAIL_S,
    CLOSE_UNSPOKEN_S,
    CONSULT_FRAMING,
    CONSULT_HISTORY_CHARS,
    CONSULT_TURN_CHARS,
    IDLE_S,
    TURN_EFFORT,
    TURN_META,
    VOICE_CARD_MARKER,
    DelegationRunner,
    EndingPolicy,
    Place,
    PrivateContext,
    call_context,
    consult_envelope,
    load_private_context,
    parse_private_context,
    recent_turns,
    run_close_sequence,
    split_persona,
    turn_request,
    with_private_context,
)


class Message:
    type = "message"

    def __init__(self, role, text):
        self.role = role
        self.text_content = text


class RequestTest(unittest.TestCase):
    def test_turn_request_json_is_pinned(self):
        self.assertEqual(
            turn_request("kitchen", "what's on today?"),
            {
                "session_id": "voice-kitchen",
                "text": "what's on today?",
                "meta": {"surface": "voice", "user": "josh"},
                "effort": "low",
            },
        )

    def test_turn_request_preserves_effort_without_exposing_model_selection(self):
        request = turn_request("kitchen", "what's on today?", effort="high")
        self.assertEqual(request["effort"], "high")
        self.assertEqual(request["session_id"], "voice-kitchen")
        self.assertEqual(request["meta"], TURN_META)
        self.assertNotIn("model", request)
        with self.assertRaises(TypeError):
            turn_request("kitchen", "what's on today?", model="another-model")

    def test_recent_turns_keeps_all_recent_messages_in_order(self):
        self.assertEqual(
            recent_turns(
                [Message("user", "old"), Message("assistant", "reply"), Message("user", "new")]
            ),
            [("user", "old"), ("assistant", "reply"), ("user", "new")],
        )

    def test_recent_turns_omits_non_messages_and_empty_messages(self):
        class NonMessage:
            type = "function_call"
            role = "assistant"
            text_content = "tool payload"

        self.assertEqual(
            recent_turns(
                [NonMessage(), Message("user", ""), Message("assistant", "kept")]
            ),
            [("assistant", "kept")],
        )

    def test_consult_envelope_keeps_request_context_for_short_delegated_question(self):
        request = "Who is the person I met at the conference named Alice?"
        for question in ("PI", ""):
            with self.subTest(question=question):
                envelope = consult_envelope(
                    "card",
                    "",
                    recent_turns(
                        [Message("user", request), Message("assistant", "I will look that up."), Message("user", question)]
                    ),
                    question,
                )
                self.assertIn(f"user: {request}", envelope)
                self.assertIn("assistant: I will look that up.", envelope)

    def test_consult_envelope_caps_each_turn_and_history_budget(self):
        turns = [("user", "a" * 2000), ("assistant", "b" * 2000), ("user", "c" * 2000), ("assistant", "newest")]
        envelope = consult_envelope("card", "", turns, "question")
        history = envelope.split("Recent conversation:\n", 1)[1].split("\n\nQuestion:\n", 1)[0]
        rendered_turns = history.splitlines()
        self.assertEqual(CONSULT_HISTORY_CHARS, 6000)
        self.assertEqual(CONSULT_TURN_CHARS, 1500)
        self.assertLessEqual(len(history), CONSULT_HISTORY_CHARS)
        self.assertIn("assistant: newest", history)
        self.assertTrue(any(line.endswith("…") for line in rendered_turns))
        self.assertTrue(all(len(line.split(": ", 1)[1]) <= CONSULT_TURN_CHARS for line in rendered_turns))

    def test_consult_envelope_truncates_oldest_turn_to_history_budget(self):
        turns = [("user", "a" * 2000), ("assistant", "b" * 2000), ("user", "c" * 2000), ("assistant", "d" * 2000)]
        envelope = consult_envelope("card", "", turns, "question")
        history = envelope.split("Recent conversation:\n", 1)[1].split("\n\nQuestion:\n", 1)[0]
        rendered_turns = history.splitlines()
        self.assertLessEqual(len(history), CONSULT_HISTORY_CHARS)
        self.assertTrue(rendered_turns[0].startswith("user: a"))
        self.assertTrue(rendered_turns[0].endswith("…"))
        self.assertLess(len(rendered_turns[0].split(": ", 1)[1]), CONSULT_TURN_CHARS)
        self.assertTrue(rendered_turns[1].startswith("assistant: b"))
        self.assertTrue(rendered_turns[-1].startswith("assistant: d"))

    def test_recent_turns_does_not_filter_former_opening_cue_text(self):
        former_cue = "(Josh just opened the call.)"
        self.assertEqual(
            recent_turns([Message("user", former_cue), Message("assistant", "Reply")]),
            [("user", former_cue), ("assistant", "Reply")],
        )

    def test_consult_envelope_acknowledges_truthfully_before_backend_action(self):
        envelope = " ".join(consult_envelope("card", "", [], "Set a timer").lower().split())
        self.assertIn(
            "immediately stream a brief, truthful spoken acknowledgement before searching for tools or taking backend action",
            envelope,
        )
        self.assertIn("acknowledge the request, not its outcome", envelope)
        self.assertIn(
            "do not say or imply anything is set, sent, or done until the relevant tool succeeds",
            envelope,
        )

    def test_consult_envelope_ends_after_a_fulfilled_action_and_confirmation(self):
        envelope = " ".join(consult_envelope("card", "", [], "Set a timer").lower().split())
        self.assertIn("has the intent been fulfilled of the conversation? if so, hang up. if plausibly not, stay.", envelope)
        self.assertIn("after a successful action that fulfills josh's intent, such as setting a timer or alarm, confirm it briefly and call end_conversation with reason done in the same response", envelope)
        self.assertIn("do not offer anything else or more help", envelope)
        self.assertIn("end_conversation with reason signoff", envelope)

    def test_consult_envelope_keeps_plausible_followups_open(self):
        envelope = " ".join(consult_envelope("card", "", [], "Who was Alice Keck?").lower().split())
        self.assertIn("keep the call open when the conversation plausibly continues", envelope)
        self.assertIn("who was she", envelope)
        self.assertIn("do not offer anything else or more help", envelope)

    def test_consult_envelope_separates_place_finding_from_navigation(self):
        for question in ("Find 1500 Santa Barbara Street", "Where is 1500 Santa Barbara Street?"):
            with self.subTest(question=question):
                envelope = " ".join(consult_envelope("card", "", [], question).lower().split())
                self.assertIn("use find_places", envelope)
                self.assertIn("speak the returned name and address", envelope)
                self.assertIn("wait for an explicit navigation request", envelope)
                self.assertIn("before calling navigate_to", envelope)
                self.assertIn("keep the call open for josh's explicit navigation request", envelope)
                self.assertIn("a yes authorizes exactly that message once", envelope)
                self.assertIn("has the intent been fulfilled of the conversation? if so, hang up. if plausibly not, stay.", envelope)

    def test_consult_envelope_preserves_sms_confirmation_wording(self):
        envelope = " ".join(consult_envelope("card", "", [], "Text Alice").lower().split())
        self.assertIn("a yes authorizes exactly that message once", envelope)
        self.assertIn("then call send_sms with send=true", envelope)

    def test_consult_envelope_has_backend_rules_and_question(self):
        envelope = consult_envelope(
            "Warm, concise voice.",
            "",
            [("user", "earlier"), ("assistant", "answer")],
            "What is on my calendar?",
        )
        self.assertNotIn("spoken in its own words", envelope.lower())
        self.assertIn("keep facts, outcomes and uncertainty intact", envelope.lower())
        self.assertIn("verbatim", envelope.lower())
        self.assertIn("a yes authorizes exactly that message once", envelope.lower())
        self.assertIn("send=true", envelope)
        self.assertIn("say the closing words", envelope.lower())
        self.assertIn("end_conversation", envelope)
        self.assertIn("say nothing after", envelope.lower())
        self.assertIn("user: earlier", envelope)
        self.assertIn("Question:\nWhat is on my calendar?", envelope)

    def test_consult_envelope_requires_complete_sms_say_back_before_yes(self):
        envelope = " ".join(consult_envelope("card", "", [], "Text this number").lower().split())
        self.assertIn("say the recipient's full phone number with every digit", envelope)
        self.assertIn("say the exact message verbatim without paraphrasing", envelope)
        self.assertIn("ask an explicit yes-or-no question", envelope)
        self.assertIn("wait for a yes in a later turn", envelope)

    def test_consult_envelope_requires_corrected_sms_say_back_and_new_yes(self):
        envelope = " ".join(consult_envelope("card", "", [], "Change the text").lower().split())
        self.assertIn("if josh corrects the recipient or message, repeat the full phone number and corrected message verbatim", envelope)
        self.assertIn("ask an explicit yes-or-no question again", envelope)
        self.assertIn("the correction voids the previous yes", envelope)
        self.assertIn("call send_sms with send=true exactly once", envelope)

    def test_consult_envelope_never_claims_sms_delivery_before_success(self):
        envelope = " ".join(consult_envelope("card", "", [], "Text this number").lower().split())
        self.assertIn("while waiting for confirmation, do not say or imply that the text is sending, sent, or done", envelope)
        self.assertIn("only say the text was sent after send_sms returns successfully", envelope)
        self.assertIn("if send_sms fails, say it was not sent", envelope)

    def test_consult_turn_cap_and_persona_split(self):
        self.assertEqual(CONSULT_TURN_CHARS, 1500)
        text = f"front\n{VOICE_CARD_MARKER}\ncard"
        self.assertEqual(split_persona(text), ("front", "card"))
        with self.assertRaises(ValueError):
            split_persona("front only")

    def test_private_context_about_and_pronunciations_are_rendered(self):
        private = PrivateContext(about="Josh is here.", pronunciations={"Mentat": "men-tat"})
        self.assertEqual(
            with_private_context("Base instructions.", private),
            "Base instructions.\n\nJosh is here.\nSay Mentat as men-tat.",
        )

    def test_private_context_toml_stays_strict(self):
        parsed = parse_private_context(
            'about = "A person."\n[pronunciations]\nMentat = "men-tat"\n'
        )
        self.assertEqual(parsed.about, "A person.")
        self.assertEqual(parsed.pronunciations, {"Mentat": "men-tat"})
        self.assertEqual(load_private_context(None), PrivateContext())
        with self.assertRaises(ValueError):
            parse_private_context('keyterms = ["Mentat"]')
        with self.assertRaises(ValueError):
            parse_private_context('unknown = "x"')

    def test_private_context_places_parse_strictly(self):
        parsed = parse_private_context(
            "[places.home]\nlat = 47.6\nlng = -122.3\nradius_m = 150\n"
        )
        self.assertEqual(parsed.places, {"home": Place(47.6, -122.3, 150.0)})
        for bad in (
            "places = 3",
            "[places.home]\nlat = 47.6\nlng = -122.3\n",
            '[places.home]\nlat = "47.6"\nlng = -122.3\nradius_m = 150\n',
            "[places.home]\nlat = 91\nlng = -122.3\nradius_m = 150\n",
            "[places.home]\nlat = 47.6\nlng = -122.3\nradius_m = 0\n",
            "[places.home]\nlat = 47.6\nlng = -122.3\nradius_m = 150\nname = 'x'\n",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_private_context(bad)

    def test_turn_constants_are_pinned(self):
        self.assertEqual(TURN_META, {"surface": "voice", "user": "josh"})
        self.assertEqual(TURN_EFFORT, "low")


class Delegation:
    def __init__(self, delegation_id):
        self.id = delegation_id


class DelegationRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_new_delegation_cancels_previous_and_close_cancels_current(self):
        started = []
        cancelled = []
        release = asyncio.Event()

        async def run(delegation):
            started.append(delegation)
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(delegation)
                raise

        runner = DelegationRunner(run, lambda _id, _error: None)
        first = Delegation("first")
        second = Delegation("second")
        runner.start(first)
        await asyncio.sleep(0)
        self.assertEqual(started, [first])
        runner.start(second)
        await asyncio.sleep(0)
        self.assertEqual(cancelled, [first])
        self.assertEqual(started, [first, second])
        await runner.close()
        self.assertEqual(cancelled, [first, second])

    async def test_cancelled_before_first_step_does_not_leave_bookkeeping(self):
        received = []
        release = asyncio.Event()

        async def run(delegation):
            received.append(delegation)
            await release.wait()

        runner = DelegationRunner(run, lambda _id, _error: None)
        first = Delegation("first")
        second = Delegation("second")
        runner.start(first)
        runner.start(second)
        await asyncio.sleep(0)
        self.assertEqual(received, [second])
        self.assertFalse(hasattr(runner, "_pending"))
        await runner.close()

    async def test_finished_and_closed_task_errors_are_reported_and_retrieved(self):
        errors = []
        uncaught = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: uncaught.append(context))
        try:
            async def raises(_delegation):
                raise RuntimeError("finished")

            runner = DelegationRunner(raises, lambda delegation_id, error: errors.append((delegation_id, error)))
            runner.start(Delegation("finished"))
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            self.assertEqual(errors[0][0], "finished")
            self.assertIsInstance(errors[0][1], RuntimeError)
            self.assertEqual(str(errors[0][1]), "finished")

            async def raises_on_cancel(_delegation):
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    raise RuntimeError("closed")

            runner = DelegationRunner(
                raises_on_cancel,
                lambda delegation_id, error: errors.append((delegation_id, error)),
            )
            runner.start(Delegation("closed"))
            await asyncio.sleep(0)
            await runner.close()
            await asyncio.sleep(0)
            self.assertEqual(errors[1][0], "closed")
            self.assertIsInstance(errors[1][1], RuntimeError)
            self.assertEqual(str(errors[1][1]), "closed")
            self.assertFalse(
                any("Task exception was never retrieved" in context.get("message", "") for context in uncaught)
            )
        finally:
            loop.set_exception_handler(previous_handler)


class EndingPolicyTest(unittest.TestCase):
    def test_end_tool_closes_a_second_after_the_goodbye_finishes(self):
        policy = EndingPolicy()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=False)
        policy.agent_speaking()
        policy.turn_done(103.0)
        self.assertEqual(policy.deadline, CLOSE_TAIL_S)
        self.assertIsNone(policy.elapsed(120.0))
        policy.agent_quiet(121.0)
        self.assertEqual(policy.deadline, CLOSE_TAIL_S)
        self.assertIsNone(policy.elapsed(121.9))
        self.assertEqual(policy.elapsed(122.0), "close")

    def test_end_tool_waits_briefly_for_a_goodbye_that_has_not_started(self):
        policy = EndingPolicy()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=False)
        policy.turn_done(100.0)
        self.assertEqual(policy.deadline, CLOSE_UNSPOKEN_S)
        policy.agent_speaking()
        self.assertIsNone(policy.elapsed(110.0))
        policy.agent_quiet(110.0)
        self.assertIsNone(policy.elapsed(110.9))
        self.assertEqual(policy.elapsed(111.0), "close")

    def test_end_tool_with_no_goodbye_still_closes(self):
        policy = EndingPolicy()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=False)
        policy.turn_done(100.0)
        self.assertIsNone(policy.elapsed(100.0 + CLOSE_UNSPOKEN_S - 0.1))
        self.assertEqual(policy.elapsed(100.0 + CLOSE_UNSPOKEN_S), "close")

    def test_remaining_counts_down_from_when_the_window_was_armed(self):
        policy = EndingPolicy()
        self.assertIsNone(policy.remaining(100.0))
        policy.agent_listening(100.0)
        self.assertEqual(policy.remaining(110.0), IDLE_S - 10.0)
        self.assertEqual(policy.remaining(200.0), 0.0)

    def test_failed_end_tool_does_not_arm(self):
        policy = EndingPolicy()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=True)
        policy.turn_done(100.0)
        self.assertIsNone(policy.deadline)

    def test_other_tools_do_not_arm(self):
        policy = EndingPolicy()
        policy.tool_result_seen("mcp__mentat__get_accounts", is_error=False)
        policy.turn_done(100.0)
        self.assertIsNone(policy.deadline)

    def test_new_delegation_cancels_ending_policy(self):
        policy = EndingPolicy()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=False)
        policy.turn_done(100.0)
        self.assertEqual(policy.deadline, CLOSE_UNSPOKEN_S)
        policy.delegation_started()
        self.assertIsNone(policy.deadline)
        self.assertIsNone(policy.elapsed(200.0))
        policy.turn_done(300.0)
        self.assertIsNone(policy.deadline)

    def test_caller_saying_bye_does_not_revoke_the_ending(self):
        policy = EndingPolicy()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=False)
        policy.agent_speaking()
        policy.turn_done(100.0)
        policy.user_spoke()
        policy.agent_quiet(102.0)
        policy.user_quiet()
        self.assertEqual(policy.elapsed(102.0 + CLOSE_TAIL_S), "close")

    def test_caller_speaking_as_the_turn_finishes_still_arms(self):
        policy = EndingPolicy()
        policy.user_spoke()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=False)
        policy.turn_done(200.0)
        self.assertEqual(policy.deadline, CLOSE_UNSPOKEN_S)

    def test_idle_listening_still_closes_after_thirty_seconds(self):
        policy = EndingPolicy()
        policy.agent_listening(100.0)
        self.assertEqual(policy.deadline, IDLE_S)
        self.assertIsNone(policy.elapsed(129.9))
        self.assertEqual(policy.elapsed(130.0), "close")

    def test_user_speech_cancels_idle(self):
        policy = EndingPolicy()
        policy.agent_listening(100.0)
        policy.user_spoke()
        self.assertIsNone(policy.deadline)
        self.assertIsNone(policy.elapsed(200.0))

    def test_busy_cancels_idle_and_user_speech_does_not_cancel_runner(self):
        policy = EndingPolicy()
        policy.agent_listening(100.0)
        policy.agent_busy()
        self.assertIsNone(policy.deadline)
        policy.user_spoke()
        self.assertIsNone(policy.deadline)


class CloseSequenceTest(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_runs_after_cleanup_even_when_room_delete_fails(self):
        order = []
        logs = []

        async def close_player():
            order.append("close_player")

        async def delete_room():
            order.append("delete_room")
            raise RuntimeError("room already gone")

        async def shutdown_job():
            order.append("shutdown_job")

        await run_close_sequence(close_player, delete_room, shutdown_job, logs.append)
        self.assertEqual(order, ["close_player", "delete_room", "shutdown_job"])
        self.assertIn("room already gone", logs[0])


PACIFIC = timezone(timedelta(hours=-7))
# Thursday 2026-09-24 19:40 Pacific
THURSDAY_EVENING = datetime(2026, 9, 24, 19, 40, tzinfo=PACIFIC)
PLACES = {"home": Place(47.6062, -122.3321, 150.0), "the gym": Place(47.62, -122.35, 80.0)}


def located(lat, lng, accuracy=10.0):
    return {ATTR_LOCATION: json.dumps({"lat": lat, "lng": lng, "accuracy_m": accuracy, "age_s": 4})}


class CallContextTest(unittest.TestCase):
    def test_time_of_day_without_attributes_uses_worker_time(self):
        self.assertEqual(
            call_context({}, PLACES, THURSDAY_EVENING),
            "Call context at the start of this call: It's Thursday evening, 7:40 pm.",
        )

    def test_parts_of_day(self):
        for hour, part in ((6, "morning"), (13, "afternoon"), (22, "night"), (2, "the middle of the night")):
            with self.subTest(hour=hour):
                self.assertIn(part, call_context({}, {}, THURSDAY_EVENING.replace(hour=hour)))

    def test_place_driving_and_home_zone(self):
        attributes = {
            ATTR_TIME_ZONE: "America/Los_Angeles",
            ATTR_DRIVING: "true",
            **located(47.6065, -122.3325),
        }
        self.assertEqual(
            call_context(attributes, PLACES, THURSDAY_EVENING),
            "Call context at the start of this call: It's Thursday evening, 7:40 pm. "
            "Josh is at home. Josh is driving.",
        )

    def test_away_zone_is_local_time_and_flags_travel(self):
        context = call_context({ATTR_TIME_ZONE: "America/New_York"}, PLACES, THURSDAY_EVENING)
        self.assertIn("It's Thursday night, 10:40 pm.", context)
        self.assertIn("Josh's phone is on New York time, not home time, so he's probably traveling.", context)

    def test_unknown_zone_falls_back_to_worker_time(self):
        context = call_context({ATTR_TIME_ZONE: "Mars/Olympus"}, PLACES, THURSDAY_EVENING)
        self.assertIn("7:40 pm", context)
        self.assertNotIn("traveling", context)

    def test_far_fuzzy_or_malformed_locations_name_no_place(self):
        for attributes in (
            located(40.0, -100.0),
            located(47.6100, -122.3321, accuracy=5000.0),
            {ATTR_LOCATION: "not json"},
            {ATTR_LOCATION: json.dumps({"lat": "x"})},
        ):
            with self.subTest(attributes=attributes):
                self.assertNotIn("Josh is at", call_context(attributes, PLACES, THURSDAY_EVENING))

    def test_fuzzy_fix_counts_within_its_accuracy(self):
        # ~250 m north of home, 150 m radius, 120 m accuracy
        context = call_context(located(47.6085, -122.3321, 120.0), PLACES, THURSDAY_EVENING)
        self.assertIn("Josh is at home.", context)

    def test_nearest_matching_place_wins_and_needs_an_aware_now(self):
        places = {"the office": Place(47.6062, -122.3321, 500.0), "home": Place(47.6063, -122.3321, 50.0)}
        self.assertIn("Josh is at home.", call_context(located(47.6063, -122.3321), places, THURSDAY_EVENING))
        with self.assertRaises(ValueError):
            call_context({}, {}, THURSDAY_EVENING.replace(tzinfo=None))


if __name__ == "__main__":
    unittest.main()
