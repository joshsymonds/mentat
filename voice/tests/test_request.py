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
    CALL_OPENED,
    CLOSE_TAIL_S,
    CLOSE_UNSPOKEN_S,
    CONSULT_FRAMING,
    CONSULT_TURN_CHARS,
    IDLE_S,
    TURN_EFFORT,
    TURN_META,
    TURN_MODEL,
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
                "model": "sonnet",
            },
        )

    def test_recent_turns_keeps_latest_messages_in_order(self):
        self.assertEqual(
            recent_turns(
                [Message("user", "old"), Message("assistant", "reply"), Message("user", "new")]
            ),
            [("assistant", "reply"), ("user", "new")],
        )

    def test_recent_turns_drops_the_opening_cue(self):
        self.assertEqual(
            recent_turns(
                [Message("user", CALL_OPENED), Message("assistant", "Hey, what's up?")]
            ),
            [("assistant", "Hey, what's up?")],
        )

    def test_consult_envelope_ends_the_call_once_intent_is_complete(self):
        envelope = " ".join(consult_envelope("card", "", [], "Set a timer").lower().split())
        self.assertIn("end the call as soon as josh's intent is complete", envelope)
        self.assertIn("no offer of more help", envelope)
        self.assertIn("end_conversation with reason done in the same turn", envelope)
        self.assertIn("end_conversation with reason signoff", envelope)
        self.assertIn("keep the call open only when you asked josh something", envelope)

    def test_consult_envelope_has_backend_rules_and_question(self):
        envelope = consult_envelope(
            "Warm, concise voice.",
            "",
            [("user", "earlier"), ("assistant", "answer")],
            "What is on my calendar?",
        )
        self.assertIn("spoken in its own words", envelope)
        self.assertIn("keep facts, outcomes and uncertainty intact", envelope.lower())
        self.assertNotIn("verbatim", envelope.lower())
        self.assertIn("a yes authorizes exactly that message once", envelope.lower())
        self.assertIn("send=true", envelope)
        self.assertIn("say the closing words", envelope.lower())
        self.assertIn("end_conversation", envelope)
        self.assertIn("say nothing after", envelope.lower())
        self.assertIn("user: earlier", envelope)
        self.assertIn("Question:\nWhat is on my calendar?", envelope)

    def test_consult_turn_cap_and_persona_split(self):
        self.assertEqual(CONSULT_TURN_CHARS, 500)
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
        self.assertEqual(TURN_MODEL, "sonnet")


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
