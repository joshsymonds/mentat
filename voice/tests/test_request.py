"""Offline tests for voice request construction and ending policy."""

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from request import (
    CLOSE_QUIET_S,
    CONSULT_FRAMING,
    CONSULT_TURN_CHARS,
    IDLE_S,
    TURN_EFFORT,
    TURN_META,
    TURN_MODEL,
    VOICE_CARD_MARKER,
    DelegationRunner,
    EndingPolicy,
    PrivateContext,
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
    def test_successful_end_tool_waits_for_speech_to_finish_before_closing(self):
        policy = EndingPolicy()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=False)
        policy.agent_speaking()
        policy.turn_done(103.0)
        self.assertEqual(policy.deadline, CLOSE_QUIET_S)
        self.assertIsNone(policy.elapsed(120.0))
        policy.agent_quiet(121.0)
        self.assertIsNone(policy.elapsed(126.9))
        self.assertEqual(policy.elapsed(127.0), "close")

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
        self.assertEqual(policy.deadline, CLOSE_QUIET_S)
        policy.delegation_started()
        self.assertIsNone(policy.deadline)
        self.assertIsNone(policy.elapsed(200.0))
        policy.turn_done(300.0)
        self.assertIsNone(policy.deadline)

    def test_user_speech_revokes_armed_close(self):
        policy = EndingPolicy()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=False)
        policy.turn_done(100.0)
        policy.user_spoke()
        self.assertIsNone(policy.deadline)
        self.assertIsNone(policy.elapsed(200.0))

    def test_idle_listening_still_closes_after_thirty_seconds(self):
        policy = EndingPolicy()
        policy.agent_listening(100.0)
        self.assertEqual(policy.deadline, IDLE_S)
        self.assertIsNone(policy.elapsed(129.9))
        self.assertEqual(policy.elapsed(130.0), "close")

    def test_busy_cancels_idle_and_user_speech_does_not_cancel_runner(self):
        policy = EndingPolicy()
        policy.agent_listening(100.0)
        policy.agent_busy()
        self.assertIsNone(policy.deadline)
        policy.user_spoke()
        self.assertIsNone(policy.deadline)

    def test_user_quiet_then_new_successful_turn_rearms(self):
        policy = EndingPolicy()
        policy.user_spoke()
        policy.user_quiet()
        policy.tool_result_seen("mcp__mentat__end_conversation", is_error=False)
        policy.turn_done(200.0)
        self.assertEqual(policy.deadline, CLOSE_QUIET_S)
        self.assertIsNone(policy.elapsed(205.9))
        self.assertEqual(policy.elapsed(206.0), "close")


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


if __name__ == "__main__":
    unittest.main()
