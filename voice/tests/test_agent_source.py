"""Source contract for the runtime-only LiveKit glue."""

import ast
import asyncio
import json
import os
import re
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
        self.assertNotIn("tts_text_transforms=", source)
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
        self.assertIn("handle.wait_for_playout()", completed)
        self.assertIn("handle.interrupted", completed)
        self.assertIn("handle.exception()", completed)
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

    def test_sms_text_blocks_reach_speech_as_separate_sentences_after_done(self):
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

        class FakeToolStart:
            pass

        class FakeToolResult:
            def __init__(self, name, is_error=False):
                self.name = name
                self.is_error = is_error

        class FakeTurnDone:
            pass

        class FakeTurnStream:
            done = False

            def feed(self, data):
                event = json.loads(data)
                if event["kind"] == "text_delta":
                    return [event["text"]]
                if event["kind"] == "tool_start":
                    return [FakeToolStart()]
                if event["kind"] == "tool_result":
                    return [FakeToolResult(event["tool"], event["is_error"])]
                if event["kind"] == "done":
                    self.done = True
                    return [FakeTurnDone()]
                return []

        class FakeContent:
            def __init__(self, chunks, speech):
                self._chunks = chunks
                self._speech = speech
                self.observed_before_done = []

            async def iter_any(self):
                for chunk in self._chunks:
                    yield chunk
                    if b'"kind":"done"' not in chunk:
                        self.observed_before_done.append(list(self._speech))

        class FakeResponse:
            status = 200

            def __init__(self, chunks, speech):
                self.content = FakeContent(chunks, speech)

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

        speech = []
        response = None
        namespace = {
            "AsyncGenerator": AsyncGenerator,
            "asyncio": asyncio,
            "json": json,
            "re": re,
            "aiohttp": SimpleNamespace(
                ClientSession=lambda **_kwargs: FakeHttp(response),
                ClientError=Exception,
            ),
            "TIMEOUT": None,
            "CONSULT_FAILED": "request failed",
            "TurnError": RuntimeError,
            "TurnStream": FakeTurnStream,
            "is_sms_request": lambda _text: True,
            "is_sms_decline": lambda _text: False,
            "is_sms_followup": lambda _text: False,
            "consult_envelope": lambda *_args, **_kwargs: "envelope",
            "recent_turns": lambda _items: [],
            "turn_request": lambda *_args: {},
            "write_turn_marker": Mock(),
            "logger": Mock(),
            "SEND_SMS_TOOL": "mcp__mentat__send_sms",
            "END_CONVERSATION_TOOL": "mcp__mentat__end_conversation",
            "ToolResult": FakeToolResult,
            "ToolStart": FakeToolStart,
            "TurnDone": FakeTurnDone,
            "TurnFailure": type("FakeTurnFailure", (), {}),
            "time": time,
        }
        buffer_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "SmsCommentaryBuffer"
        )
        exec(compile(ast.Module(body=[buffer_class, method], type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _sms_consent=False,
            _voice_card="voice card",
            _room_name="eval-room",
            _mentat_url="http://127.0.0.1:8484",
            _ending_policy=SimpleNamespace(
                tool_result_seen=Mock(), turn_done=Mock(), describe=lambda: "test policy"
            ),
            _ending_changed=Mock(),
        )
        wire = [
            b'{"kind":"text_delta","text":"Text Alex: I will be there at six"}',
            b'{"kind":"tool_start","tool":"lookup"}',
            b'{"kind":"tool_result","tool":"lookup","is_error":false}',
            b'{"kind":"text_delta","text":"The message is sent."}',
            b'{"kind":"done"}',
        ]
        response = FakeResponse(wire, speech)

        async def consume_as_speech():
            async for text in namespace["_backend_text"](
                agent, "Text Alex that I'll be there at six", "turn-id", SimpleNamespace(items=[])
            ):
                speech.append(text)

        asyncio.run(consume_as_speech())

        self.assertTrue(response.content.observed_before_done)
        self.assertTrue(all(not observed for observed in response.content.observed_before_done))
        self.assertEqual(
            speech,
            ["Text Alex: I will be there at six.", None, "The message is sent."],
        )

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

    def test_sms_commentary_buffer_reports_failed_send_truthfully_and_idempotently(self):
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
        buffer.finish()
        self.assertEqual(appended, ["I couldn't send that text. Please try again."])
        self.assertNotIn("sent", appended[0].lower())

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

    def test_interrupt_closes_backend_stream_while_next_read_is_pending(self):
        from collections.abc import AsyncGenerator

        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        method = next(
            node for node in front_agent.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "on_user_turn_completed"
        )

        backend_open = asyncio.Event()
        backend_closed = asyncio.Event()
        speech_started = asyncio.Event()
        hold_backend = asyncio.Event()

        async def backend_text(_question, _turn_id, _chat_ctx):
            try:
                yield "On it."
                backend_open.set()
                await hold_backend.wait()
            finally:
                backend_closed.set()

        class FakeSpeechHandle:
            def __init__(self, task):
                self._task = task
                self.interrupted = False

            async def wait_for_playout(self):
                await self._task

            def interrupt(self):
                self.interrupted = True
                self._task.cancel()

            def exception(self):
                return None

        class FakeSession:
            def __init__(self):
                self.handle = None

            def say(self, source, *, allow_interruptions):
                self.assert_interruptions = allow_interruptions

                async def capture():
                    async for _text in source:
                        speech_started.set()

                self.handle = FakeSpeechHandle(asyncio.create_task(capture()))
                return self.handle

        namespace = {
            "AsyncGenerator": AsyncGenerator,
            "asyncio": asyncio,
            "uuid4": lambda: SimpleNamespace(hex="turn-id"),
            "AudioConfig": lambda *_args, **_kwargs: object(),
            "EARCON_PATH": Path("earcon.wav"),
            "logger": Mock(),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(agent_path), "exec"), namespace)
        session = FakeSession()
        agent = SimpleNamespace(
            _ending_policy=SimpleNamespace(delegation_started=Mock()),
            _ending_changed=Mock(),
            _background=SimpleNamespace(play=Mock()),
            _backend_text=backend_text,
            session=session,
        )

        async def interrupt_while_backend_open():
            turn = asyncio.create_task(
                namespace["on_user_turn_completed"](
                    agent,
                    SimpleNamespace(items=[]),
                    SimpleNamespace(text_content="Tell me now"),
                )
            )
            await asyncio.wait_for(backend_open.wait(), timeout=1)
            await asyncio.wait_for(speech_started.wait(), timeout=1)
            self.assertFalse(backend_closed.is_set())
            session.handle.interrupt()
            await asyncio.wait_for(turn, timeout=1)

        asyncio.run(interrupt_while_backend_open())
        self.assertTrue(session.handle.interrupted)
        self.assertTrue(backend_closed.is_set())

    def test_speech_interrupt_during_segment_flush_returns_without_cancelling_turn(self):
        from collections.abc import AsyncGenerator

        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        method = next(
            node for node in front_agent.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name == "on_user_turn_completed"
        )

        playout_waiting = asyncio.Event()
        backend_closed = asyncio.Event()
        hold_playout = asyncio.Event()

        async def backend_text(_question, _turn_id, _chat_ctx):
            try:
                yield "On it."
                yield None
            finally:
                backend_closed.set()

        class FakeSpeechHandle:
            def __init__(self, task):
                self._task = task
                self.interrupted = False

            async def wait_for_playout(self):
                await self._task

            def interrupt(self):
                self.interrupted = True
                self._task.cancel()

            def exception(self):
                return None

        class FakeSession:
            def __init__(self):
                self.handle = None

            def say(self, source, *, allow_interruptions):
                async def play():
                    async for _text in source:
                        pass
                    playout_waiting.set()
                    await hold_playout.wait()

                self.handle = FakeSpeechHandle(asyncio.create_task(play()))
                return self.handle

        namespace = {
            "AsyncGenerator": AsyncGenerator,
            "asyncio": asyncio,
            "uuid4": lambda: SimpleNamespace(hex="turn-id"),
            "AudioConfig": lambda *_args, **_kwargs: object(),
            "EARCON_PATH": Path("earcon.wav"),
            "logger": Mock(),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(agent_path), "exec"), namespace)
        session = FakeSession()
        agent = SimpleNamespace(
            _ending_policy=SimpleNamespace(delegation_started=Mock()),
            _ending_changed=Mock(),
            _background=SimpleNamespace(play=Mock()),
            _backend_text=backend_text,
            session=session,
        )

        async def interrupt_during_flush():
            turn = asyncio.create_task(
                namespace["on_user_turn_completed"](
                    agent,
                    SimpleNamespace(items=[]),
                    SimpleNamespace(text_content="Tell me now"),
                )
            )
            await asyncio.wait_for(playout_waiting.wait(), timeout=1)
            session.handle.interrupt()
            await asyncio.wait_for(turn, timeout=1)

        asyncio.run(interrupt_during_flush())
        self.assertTrue(session.handle.interrupted)
        self.assertTrue(backend_closed.is_set())

    def test_default_tts_path_flushes_punctuated_text_before_tool_result(self):
        from collections.abc import AsyncGenerator
        from types import MethodType

        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        methods = {
            node.name: node
            for node in front_agent.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name in {"_backend_text", "on_user_turn_completed"}
        }

        class FakeTurnStart:
            pass

        class FakeToolResult:
            def __init__(self, name, is_error=False):
                self.name = name
                self.is_error = is_error

        class FakeTurnDone:
            pass

        class FakeTurnStream:
            def __init__(self):
                self.done = False

            def feed(self, data):
                event = json.loads(data)
                if event["kind"] == "text_delta":
                    return [event["text"]]
                if event["kind"] == "tool_start":
                    return [FakeTurnStart()]
                if event["kind"] == "tool_result":
                    return [FakeToolResult(event["tool"])]
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
            status = 200

            def __init__(self, chunks):
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

        async def livekit_markdown_filter(source):
            # LiveKit 1.8.1 filter_markdown holds the last inline split token.
            buffer = ""
            async for chunk in source:
                buffer += chunk
                last_split = max(buffer.rfind(char) for char in " ,.?!;，。？！；")
                if last_split >= 1:
                    yield buffer[:last_split]
                    buffer = buffer[last_split:]
            if buffer:
                yield buffer

        def cartesia_sentence_flush(text):
            # Plain-text subset of LiveKit's 1.8.1 defaults: min context 10,
            # min sentence 20; end_input flushes its remaining short sentence.
            ends = [match.end() for match in re.finditer(r"[.!?](?=\s|$)", text)]
            sentences = []
            start = 0
            for end in ends:
                if len(text[start:end]) >= 20:
                    sentences.append(text[start:end].strip())
                    start = end
            if start < len(text) and text[start:].strip():
                sentences.append(text[start:].strip())
            return sentences

        timeline = []
        response = None

        class FakeSpeechHandle:
            interrupted = False

            def __init__(self, task):
                self._task = task

            async def wait_for_playout(self):
                await self._task

            def interrupt(self):
                self._task.cancel()

            def exception(self):
                return None

        class FakeSession:
            def say(self, source, *, allow_interruptions):
                self.assert_interruptions = allow_interruptions

                async def synthesize():
                    filtered = []
                    async for text in livekit_markdown_filter(source):
                        filtered.append(text)
                    for sentence in cartesia_sentence_flush("".join(filtered)):
                        timeline.append(("tts", sentence))

                return FakeSpeechHandle(asyncio.create_task(synthesize()))

        policy = SimpleNamespace(
            delegation_started=Mock(),
            tool_result_seen=lambda name, _error: timeline.append(("tool", name)),
            turn_done=Mock(),
            describe=lambda: "test policy",
        )
        namespace = {
            "AsyncGenerator": AsyncGenerator,
            "asyncio": asyncio,
            "json": json,
            "re": re,
            "aiohttp": SimpleNamespace(
                ClientSession=lambda **_kwargs: FakeHttp(response),
                ClientError=Exception,
            ),
            "TIMEOUT": None,
            "CONSULT_FAILED": "request failed",
            "TurnError": RuntimeError,
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
            "ToolResult": FakeToolResult,
            "ToolStart": FakeTurnStart,
            "TurnDone": FakeTurnDone,
            "TurnFailure": type("FakeTurnFailure", (), {}),
            "time": time,
            "uuid4": lambda: SimpleNamespace(hex="turn-id"),
            "AudioConfig": lambda *_args, **_kwargs: object(),
            "EARCON_PATH": Path("earcon.wav"),
        }
        exec(compile(ast.Module(body=list(methods.values()), type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _sms_consent=False,
            _voice_card="voice card",
            _room_name="eval-room",
            _mentat_url="http://127.0.0.1:8484",
            _ending_policy=policy,
            _ending_changed=Mock(),
            _background=SimpleNamespace(play=Mock()),
            session=FakeSession(),
        )
        agent._backend_text = MethodType(namespace["_backend_text"], agent)
        wire = [
            b'{"kind":"text_delta","text":"On it."}',
            b'{"kind":"tool_start","tool":"lookup"}',
            b'{"kind":"tool_result","tool":"lookup","is_error":false}',
            b'{"kind":"text_delta","text":"The time is 3:45."}',
            b'{"kind":"done"}',
        ]
        response = FakeResponse(wire)

        asyncio.run(
            namespace["on_user_turn_completed"](
                agent,
                SimpleNamespace(items=[]),
                SimpleNamespace(text_content="Check the time"),
            )
        )

        self.assertEqual(
            [text for kind, text in timeline if kind == "tts"],
            ["On it.", "The time is 3:45."],
        )
        tool_result = timeline.index(("tool", "lookup"))
        self.assertLess(timeline.index(("tts", "On it.")), tool_result)

    def test_text_before_tool_start_reaches_tts_as_a_sentence_before_tool_result(self):
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

        class FakeTurnStart:
            pass

        class FakeToolResult:
            def __init__(self, name, is_error=False):
                self.name = name
                self.is_error = is_error

        class FakeTurnDone:
            pass

        class FakeTurnStream:
            def __init__(self):
                self.done = False

            def feed(self, data):
                items = []
                for line in data.splitlines():
                    event = json.loads(line)
                    if event["kind"] == "text_delta":
                        items.append(event["text"])
                    elif event["kind"] == "tool_start":
                        items.append(FakeTurnStart())
                    elif event["kind"] == "tool_result":
                        items.append(FakeToolResult(event["tool"]))
                    elif event["kind"] == "done":
                        self.done = True
                        items.append(FakeTurnDone())
                return items

        class FakeContent:
            def __init__(self, chunks):
                self._chunks = chunks

            async def iter_any(self):
                for chunk in self._chunks:
                    yield chunk

        class FakeResponse:
            status = 200

            def __init__(self, chunks):
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

        timeline = []
        response = None
        policy = SimpleNamespace(
            tool_result_seen=lambda name, is_error: timeline.append(("tool", name)),
            turn_done=Mock(),
            describe=lambda: "test policy",
        )
        namespace = {
            "AsyncGenerator": AsyncGenerator,
            "asyncio": asyncio,
            "json": json,
            "re": re,
            "aiohttp": SimpleNamespace(
                ClientSession=lambda **_kwargs: FakeHttp(response),
                ClientError=Exception,
            ),
            "TIMEOUT": None,
            "CONSULT_FAILED": "request failed",
            "TurnError": RuntimeError,
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
            "ToolResult": FakeToolResult,
            "ToolStart": FakeTurnStart,
            "TurnDone": FakeTurnDone,
            "TurnFailure": type("FakeTurnFailure", (), {}),
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
        wire = (
            b'{"kind":"text_delta","text":"The time is 3:45"}\n'
            b'{"kind":"tool_start","tool":"lookup"}\n'
            b'{"kind":"tool_result","tool":"lookup","is_error":false}\n'
            b'{"kind":"text_delta","text":"That is correct."}\n'
            b'{"kind":"done"}\n'
        )

        async def consume_as_tts():
            return [
                (timeline.append(("tts", text)) or text)
                async for text in namespace["_backend_text"](
                    agent, "What time is it?", "turn-id", SimpleNamespace(items=[])
                )
            ]

        response = FakeResponse([wire])
        spoken = asyncio.run(consume_as_tts())
        self.assertEqual(spoken, ["The time is 3:45", ". ", None, "That is correct."])
        tool_result = timeline.index(("tool", "lookup"))
        self.assertLess(timeline.index(("tts", "The time is 3:45")), tool_result)
        self.assertLess(timeline.index(("tts", ". ")), tool_result)
        self.assertEqual("".join(text for text in spoken if text is not None), "The time is 3:45. That is correct.")

    def test_first_commentary_is_logged_at_the_tts_handoff(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        backend = source.split("    async def _backend_text(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        log_line = 'logger.info("delegation %s first commentary", turn_id)'
        self.assertEqual(backend.count(log_line), 1)
        self.assertLess(backend.index("if not commentary_logged"), backend.index(log_line))
        self.assertLess(backend.index(log_line), backend.index("yield item"))
        self.assertIn("self.session.say(speech_source(queue), allow_interruptions=True)", source)
        self.assertLess(backend.index("write_turn_marker("), backend.index("http.post("))
        self.assertNotIn("append_commentary", source)

    def test_agent_configures_nonzero_flux_endpointing_grace(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        tree = ast.parse(source)
        entrypoint = next(
            node for node in tree.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "entrypoint"
        )
        session_call = next(
            node for node in ast.walk(entrypoint)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AgentSession"
        )
        turn_handling = next(
            keyword.value for keyword in session_call.keywords
            if keyword.arg == "turn_handling"
        )
        endpointing = next(
            value for key, value in zip(turn_handling.keys, turn_handling.values)
            if isinstance(key, ast.Constant) and key.value == "endpointing"
        )
        min_delay = ast.literal_eval(next(
            value for key, value in zip(endpointing.keys, endpointing.values)
            if isinstance(key, ast.Constant) and key.value == "min_delay"
        ))
        self.assertEqual(min_delay, 0.5)


if __name__ == "__main__":
    unittest.main()
