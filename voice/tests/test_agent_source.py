"""Source contract for the runtime-only LiveKit glue."""

import ast
import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


class AgentSourceContractTest(unittest.TestCase):
    def test_agent_uses_flux_turn_detection_and_sonic_without_gpt_live(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        self.assertIn('inference.STT("deepgram/flux-general")', source)
        self.assertIn('"turn_detection": "stt"', source)
        self.assertIn('inference.TTS("cartesia/sonic-3.6", voice=TTS_VOICE)', source)
        self.assertIn("TTS_VOICE =", source)
        self.assertIn("tts_text_transforms=None", source)
        self.assertIn("async def on_user_turn_completed(", source)
        self.assertNotIn("GPTLive", source)
        self.assertNotIn("openai.realtime", source)

    def test_each_user_turn_streams_backend_text_directly_to_speech(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        completed = source.split("    async def on_user_turn_completed(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        self.assertIn("self._backend_text(", completed)
        self.assertIn("self.session.say(", completed)
        self.assertIn("allow_interruptions=True", completed)
        self.assertIn("await speech_handle.wait_for_playout()", completed)
        self.assertIn("speech_handle.interrupted", completed)
        self.assertIn("speech_handle.exception()", completed)
        self.assertIn("await backend_text.aclose()", completed)
        backend = source.split("    async def _backend_text(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        self.assertIn('f"{self._mentat_url}/v1/conversation"', backend)
        self.assertIn("turn_request(self._room_name, envelope)", backend)
        self.assertIn("yield item", backend)
        self.assertNotIn("chunker", backend)

    def test_backend_turn_marker_and_first_commentary_log_are_private_and_exactly_timed(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        backend = source.split("    async def _backend_text(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        marker = source.split("def write_turn_marker(", 1)[1].split("\n\n", 1)[0]
        self.assertIn('os.environ.get("MENTAT_EVAL_DELEGATION_LOG")', marker)
        self.assertIn('"room": room_name', marker)
        self.assertIn('"id": turn_id', marker)
        self.assertIn('"created_at": time.time()', marker)
        self.assertIn('separators=(",", ":")', marker)
        self.assertIn("write_turn_marker(self._room_name, turn_id)", backend)
        self.assertLess(backend.index("write_turn_marker("), backend.index("http.post("))
        self.assertNotIn("pending_transcript", marker + backend)
        self.assertNotIn("credential", (marker + backend).lower())

    def test_backend_turn_writes_only_private_timed_eval_jsonl_when_opted_in(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        marker = source.split("def write_turn_marker(", 1)[1].split("\n\n", 1)[0]
        self.assertIn('os.environ.get("MENTAT_EVAL_DELEGATION_LOG")', marker)
        self.assertIn('"room": room_name', marker)
        self.assertIn('"id": turn_id', marker)
        self.assertIn('"created_at": time.time()', marker)
        self.assertIn('separators=(",", ":")', marker)
        self.assertNotIn("question", marker)
        self.assertNotIn("credential", marker.lower())

    def test_turn_marker_is_opt_in_compact_jsonl_without_text_or_credentials(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "write_turn_marker"
        )
        namespace = {
            "os": os,
            "json": json,
            "time": time,
            "Path": Path,
            "logger": Mock(),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(agent_path), "exec"), namespace)

        with tempfile.TemporaryDirectory() as temporary:
            marker_path = Path(temporary) / "turns.jsonl"
            with patch.dict(os.environ, {}, clear=True):
                namespace["write_turn_marker"]("eval-room", "same-id")
            self.assertFalse(marker_path.exists())

            with patch.dict(os.environ, {"MENTAT_EVAL_DELEGATION_LOG": str(marker_path)}, clear=True):
                namespace["write_turn_marker"]("eval-room", "same-id")
                namespace["write_turn_marker"]("eval-room", "second-id")

            lines = marker_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            markers = [json.loads(line) for line in lines]
            self.assertTrue(all(set(marker) == {"room", "id", "created_at"} for marker in markers))
            self.assertEqual([marker["room"] for marker in markers], ["eval-room", "eval-room"])
            self.assertEqual([marker["id"] for marker in markers], ["same-id", "second-id"])
            self.assertTrue(all(isinstance(marker["created_at"], (int, float)) for marker in markers))
            self.assertNotIn("transcript", "\n".join(lines))
            self.assertNotIn("credential", "\n".join(lines).lower())

    def test_agent_has_no_startup_greeting_history(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        self.assertNotIn("CALL_OPENED", source)
        self.assertNotIn("chat_ctx=", source)
        self.assertNotIn("opening.add_message", source)

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
            "GPTLive",
            "openai.realtime",
            "inference.LLM",
            "function_tool",
            "Respeller",
            "PhoneActions",
            "_provider_format",
            "keyterms",
            "_pending",
        ):
            self.assertNotIn(forbidden, source)

    def test_room_io_starts_before_connect_with_default_subscription(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        entry = source.split("async def entrypoint(", 1)[1]
        io_start = entry.index("await voice_room_io.start()")
        connect = entry.index("await ctx.connect()")
        wait = entry.index("await ctx.wait_for_participant()")
        context = entry.index("call_context(caller.attributes, private.places")
        folded = entry.index('instructions += "\\n\\n" + context')
        session_start = entry.index("await session.start(")

        self.assertLess(io_start, connect)
        self.assertLess(connect, wait)
        self.assertLess(wait, context)
        self.assertLess(context, folded)
        self.assertLess(folded, session_start)
        self.assertNotIn("room=ctx.room", entry[session_start:])
        self.assertIn("room_io.RoomIO(", entry)
        self.assertNotIn("auto_subscribe=", entry)
        self.assertNotIn("AutoSubscribe.SUBSCRIBE_NONE", entry)

    def test_room_io_is_closed_with_the_session(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        entry = source.split("async def entrypoint(", 1)[1]
        close_sequence = entry.split("run_close_sequence(", 1)[1].split("\n                )", 1)[0]
        self.assertIn("_close_audio", close_sequence)
        close_audio = entry.split("async def _close_audio()", 1)[1].split(
            '@session.on("close")', 1
        )[0]
        self.assertIn("await _close_voice_room_io()", close_audio)

    def test_close_sequence_call_matches_helper_signature(self):
        voice_dir = Path(__file__).resolve().parents[1]
        agent_tree = ast.parse((voice_dir / "agent.py").read_text())
        request_tree = ast.parse((voice_dir / "request.py").read_text())
        close_call = next(
            node
            for node in ast.walk(agent_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_close_sequence"
        )
        close_definition = next(
            node
            for node in ast.walk(request_tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "run_close_sequence"
        )
        positional_parameters = (
            close_definition.args.posonlyargs + close_definition.args.args
        )
        required_parameters = len(positional_parameters) - len(close_definition.args.defaults)

        self.assertGreaterEqual(len(close_call.args), required_parameters)
        self.assertLessEqual(len(close_call.args), len(positional_parameters))
        self.assertFalse(close_call.keywords)

    def test_room_io_shutdown_cleanup_is_registered_before_connect(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        entry = source.split("async def entrypoint(", 1)[1]
        register = entry.index("ctx.add_shutdown_callback(_close_voice_room_io)")
        connect = entry.index("await ctx.connect()")
        close_helper = entry.split("async def _close_voice_room_io()", 1)[1].split(
            "ctx.add_shutdown_callback(_close_voice_room_io)", 1
        )[0]
        self.assertLess(register, connect)
        self.assertIn("await voice_room_io.aclose()", close_helper)

    def test_explicit_microphone_subscription_helpers_are_absent(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        for superseded in (
            "caller_identity",
            "_subscribe_microphone",
            "_on_track_published",
            "set_subscribed(True)",
            'ctx.room.on("track_published"',
            "caller.track_publications.values()",
        ):
            self.assertNotIn(superseded, source)

    def test_sms_intent_and_correction_detection_covers_initial_and_corrected_sayback(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        helpers = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"is_sms_request", "is_sms_followup", "is_sms_decline"}
        ]
        self.assertEqual({node.name for node in helpers}, {
            "is_sms_request", "is_sms_followup", "is_sms_decline"
        })
        namespace = {"re": __import__("re")}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), str(agent_path), "exec"), namespace)
        self.assertTrue(namespace["is_sms_request"]("Text Alex that I'll be there at six"))
        self.assertTrue(namespace["is_sms_followup"]("Actually, change that to seven"))
        self.assertTrue(namespace["is_sms_followup"]("Yes, send it"))
        self.assertTrue(namespace["is_sms_decline"]("No, don't send it"))
        self.assertFalse(namespace["is_sms_decline"]("No, I meant change that to seven"))
        self.assertFalse(namespace["is_sms_request"]("What time is it?"))
        self.assertFalse(namespace["is_sms_request"]("What did you hear in the text from Alex?"))
        self.assertFalse(namespace["is_sms_request"]("What does the text say?"))
        self.assertFalse(namespace["is_sms_followup"]("What time is it?"))

    def test_sms_commentary_buffer_withholds_until_send_result_and_appends_atomically(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        buffer_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SmsCommentaryBuffer"
        )
        namespace = {"SEND_SMS_TOOL": "mcp__mentat__send_sms"}
        exec(compile(ast.Module(body=[buffer_class], type_ignores=[]), str(agent_path), "exec"), namespace)
        appended = []
        buffer = namespace["SmsCommentaryBuffer"](appended.append)
        buffer.add("Text to +1 202 555 0142: I'll be there at 6. Should I send it?")
        self.assertEqual(appended, [])
        buffer.tool_result("mcp__mentat__send_sms", is_error=False)
        self.assertEqual(appended, [])
        buffer.add(" Sent.")
        buffer.finish()
        self.assertEqual(
            appended,
            ["Text to +1 202 555 0142: I'll be there at 6. Should I send it? Sent."],
        )

    def test_pending_sms_sayback_is_appended_once_without_a_send_call(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        buffer_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SmsCommentaryBuffer"
        )
        namespace = {"SEND_SMS_TOOL": "mcp__mentat__send_sms"}
        exec(compile(ast.Module(body=[buffer_class], type_ignores=[]), str(agent_path), "exec"), namespace)
        appended = []
        buffer = namespace["SmsCommentaryBuffer"](appended.append)
        buffer.add("Text to +1 202 555 0142: I'll be there at 6. Should I send it?")
        buffer.add(" Please confirm.")
        self.assertEqual(appended, [])
        buffer.finish()
        buffer.finish()
        self.assertEqual(
            appended,
            ["Text to +1 202 555 0142: I'll be there at 6. Should I send it? Please confirm."],
        )

    def test_pending_sayback_preserves_common_words_inside_the_message_body(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        buffer_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SmsCommentaryBuffer"
        )
        namespace = {"SEND_SMS_TOOL": "mcp__mentat__send_sms"}
        exec(compile(ast.Module(body=[buffer_class], type_ignores=[]), str(agent_path), "exec"), namespace)
        appended = []
        buffer = namespace["SmsCommentaryBuffer"](appended.append)
        sayback = (
            "Text to +1 202 555 0142: I sent the report after sending the update. "
            "Should I send it?"
        )
        buffer.add(sayback)
        buffer.finish()
        self.assertEqual(appended, [sayback])

    def test_sms_commentary_buffer_discards_success_claim_when_send_fails(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        buffer_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SmsCommentaryBuffer"
        )
        namespace = {"SEND_SMS_TOOL": "mcp__mentat__send_sms"}
        exec(compile(ast.Module(body=[buffer_class], type_ignores=[]), str(agent_path), "exec"), namespace)
        appended = []
        buffer = namespace["SmsCommentaryBuffer"](appended.append)
        buffer.add("Sending it now. Okay, sent.")
        buffer.tool_result("mcp__mentat__send_sms", is_error=True)
        buffer.finish()
        self.assertEqual(appended, [])

    def test_sms_buffering_keeps_send_sayback_atomic_and_other_text_streaming(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        stream_backend = source.split("    async def _backend_text(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        self.assertIn("SmsCommentaryBuffer", stream_backend)
        self.assertIn("sms_buffer.add(item)", stream_backend)
        self.assertIn("yield item", stream_backend)
        self.assertIn("sms_buffer.finish()", stream_backend)
        self.assertIn("self._sms_consent and (is_sms_followup(question) or sms_declined)", stream_backend)
        self.assertIn('item.name == SEND_SMS_TOOL', stream_backend)

    def test_backend_failures_produce_truthful_speech_without_marking_turn_done(self):
        from collections.abc import AsyncGenerator

        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        method = next(
            node for node in front_agent.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_backend_text"
        )

        class DaemonFailure(Exception):
            pass

        class FakeTurnDone:
            pass

        class FakeTurnFailure:
            def __init__(self, message):
                self.message = message

        class FakeTurnStream:
            def __init__(self):
                self.done = False

            def feed(self, data):
                event = json.loads(data)
                if event["kind"] == "text_delta":
                    return [event["text"]]
                if event["kind"] == "error":
                    return [FakeTurnFailure(event["message"])]
                if event["kind"] == "done":
                    self.done = True
                    return [FakeTurnDone()]
                return []

        class FakeContent:
            def __init__(self, chunks):
                self._chunks = chunks

            async def iter_any(self):
                for chunk in self._chunks:
                    yield chunk

        class FakeResponse:
            def __init__(self, status, chunks):
                self.status = status
                self.content = FakeContent(chunks)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

        class FakeHttp:
            def __init__(self, response):
                self._response = response

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def post(self, *_args, **_kwargs):
                return self._response

        policy = SimpleNamespace(
            tool_result_seen=Mock(),
            turn_done=Mock(),
            describe=lambda: "test policy",
        )
        response = None
        namespace = {
            "AsyncGenerator": AsyncGenerator,
            "asyncio": asyncio,
            "json": json,
            "aiohttp": SimpleNamespace(
                ClientSession=lambda **_kwargs: FakeHttp(response),
                ClientError=Exception,
            ),
            "TIMEOUT": None,
            "CONSULT_FAILED": "I couldn't complete that request with Mentat. Please try again.",
            "TurnError": DaemonFailure,
            "TurnStream": FakeTurnStream,
            "is_sms_request": lambda _text: False,
            "is_sms_decline": lambda _text: False,
            "is_sms_followup": lambda _text: False,
            "consult_envelope": lambda *_args, **_kwargs: "envelope",
            "recent_turns": lambda _items: [],
            "turn_request": lambda *_args: {},
            "write_turn_marker": Mock(),
            "logger": Mock(),
            "SEND_SMS_TOOL": "mcp__mentat__send_sms",
            "END_CONVERSATION_TOOL": "mcp__mentat__end_conversation",
            "ToolResult": type("ToolResult", (), {}),
            "TurnDone": FakeTurnDone,
            "TurnFailure": FakeTurnFailure,
            "time": time,
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _sms_consent=False,
            _voice_card="voice card",
            _room_name="eval-room",
            _mentat_url="http://127.0.0.1:8484",
            _ending_policy=policy,
            _ending_changed=Mock(),
        )

        async def collect():
            return [
                text
                async for text in namespace["_backend_text"](
                    agent, "What time is it?", "turn-id", SimpleNamespace(items=[])
                )
            ]

        error_line = b'{"kind":"error","message":"daemon failure"}\n'
        text_line = b'{"kind":"text_delta","text":"Partial answer."}\n'
        cases = (
            (503, [], [namespace["CONSULT_FAILED"]]),
            (200, [error_line], [namespace["CONSULT_FAILED"]]),
            (200, [text_line], ["Partial answer.", namespace["CONSULT_FAILED"]]),
        )
        for status, chunks, expected in cases:
            with self.subTest(status=status, chunks=chunks):
                response = FakeResponse(status, chunks)
                agent._sms_consent = False
                self.assertEqual(asyncio.run(collect()), expected)
                policy.turn_done.assert_not_called()

    def test_first_commentary_is_logged_at_the_tts_handoff(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        backend = source.split("    async def _backend_text(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        log_line = 'logger.info("delegation %s first commentary", turn_id)'
        self.assertEqual(backend.count(log_line), 1)
        self.assertLess(backend.index("if not commentary_logged"), backend.index(log_line))
        self.assertLess(backend.index(log_line), backend.index("yield item"))
        self.assertIn("self.session.say(backend_text, allow_interruptions=True)", source)
        self.assertLess(backend.index("write_turn_marker("), backend.index("http.post("))
        self.assertNotIn("append_commentary", source)


if __name__ == "__main__":
    unittest.main()
