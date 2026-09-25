"""Source contract for the runtime-only LiveKit glue."""

import unittest
from pathlib import Path


class AgentSourceContractTest(unittest.TestCase):
    def test_agent_uses_gpt_live_client_delegation(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        for required in (
            "GPTLiveModel(",
            'delegation="client"',
            "vad=",
            "delegation_created",
            "append_commentary",
        ):
            self.assertIn(required, source)

    def test_agent_opens_with_the_call_opened_cue_as_startup_history(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        self.assertIn('opening.add_message(role="user", content=CALL_OPENED)', source)
        self.assertIn("chat_ctx=opening", source)

    def test_agent_reads_call_context_before_the_session_starts(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        entry = source.split("async def entrypoint(", 1)[1]
        wait = entry.index("await ctx.wait_for_participant()")
        context = entry.index("call_context(caller.attributes, private.places")
        folded = entry.index('instructions += "\\n\\n" + context')
        start = entry.index("await session.start(")
        self.assertLess(wait, context)
        self.assertLess(context, folded)
        self.assertLess(folded, start)

    def test_agent_handles_turn_done_and_turn_failure(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        self.assertIn("TurnDone", source)
        self.assertIn("TurnFailure", source)

    def test_request_has_no_keyterms_context(self):
        source = (Path(__file__).resolve().parents[1] / "request.py").read_text()
        self.assertNotIn("keyterms", source)

    def test_policy_cancellation_helpers_return_no_tokens(self):
        source = (Path(__file__).resolve().parents[1] / "request.py").read_text()
        self.assertNotIn('return "cancel"', source)

    def test_old_front_and_tools_are_absent(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        self.assertNotIn("user_input_transcribed", source)
        for forbidden in (
            "inference.STT",
            "inference.LLM",
            "inference.TTS",
            "function_tool",
            "cartesia",
            "deepgram",
            "Respeller",
            "PhoneActions",
            "session.say",
            "_provider_format",
            "keyterms",
            "_pending",
        ):
            self.assertNotIn(forbidden, source)

    def test_each_delegation_logs_exactly_at_first_commentary_append(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        run_delegation = source.split("    async def _run_delegation(", 1)[1].split(
            "    async def _stream_backend(", 1
        )[0]
        stream_backend = source.split("    async def _stream_backend(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]

        log_line = 'logger.info("delegation %s first commentary", delegation.id)'
        guarded_log = run_delegation.index(log_line)
        first_check = run_delegation.index("if not commentary_logged")
        mark_logged = run_delegation.index("commentary_logged = True")
        append = run_delegation.index("self.duplex_session.append_commentary(")
        self.assertEqual(run_delegation.count(log_line), 1)
        self.assertLess(first_check, guarded_log)
        self.assertLess(guarded_log, mark_logged)
        self.assertLess(mark_logged, append)
        self.assertIn("append_commentary=append_commentary", run_delegation)
        self.assertEqual(stream_backend.count("append_commentary(chunk)"), 2)
        self.assertNotIn("duplex_session.append_commentary(", stream_backend)


if __name__ == "__main__":
    unittest.main()
