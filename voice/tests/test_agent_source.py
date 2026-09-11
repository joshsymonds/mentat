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


if __name__ == "__main__":
    unittest.main()
