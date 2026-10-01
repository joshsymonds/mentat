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
    def test_agent_uses_openai_transcription_and_elevenlabs_http_stream_adapter(self):
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
        stt = next(
            keyword.value for keyword in session_call.keywords if keyword.arg == "stt"
        )
        expected_stt = ast.parse("stt", mode="eval").body
        self.assertEqual(
            ast.dump(stt, include_attributes=False),
            ast.dump(expected_stt, include_attributes=False),
        )
        stt_provider = next(
            node.value for node in entrypoint.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "stt" for target in node.targets)
        )
        expected_stt_provider = ast.parse(
            'openai.STT(model="gpt-live-transcribe", api_key=os.environ["OPENAI_API_KEY"], '
            'vad=ctx.proc.userdata["vad"], language="en")',
            mode="eval",
        ).body
        self.assertEqual(
            ast.dump(stt_provider, include_attributes=False),
            ast.dump(expected_stt_provider, include_attributes=False),
        )
        default_voice = next(
            node.value for node in entrypoint.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "default_voice"
                for target in node.targets
            )
        )
        expected_default_voice = ast.parse(
            'os.environ.get("MENTAT_VOICE_TTS_VOICE", TTS_VOICE)', mode="eval"
        ).body
        self.assertEqual(
            ast.dump(default_voice, include_attributes=False),
            ast.dump(expected_default_voice, include_attributes=False),
        )
        tts_provider = next(
            node.value for node in entrypoint.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "tts_provider"
                for target in node.targets
            )
        )
        expected_tts_provider = ast.parse(
            'elevenlabs.TTS(model="eleven_v4_turbo", '
            'api_key=os.environ["ELEVENLABS_API_KEY"], '
            'voice_id=default_voice, language="en")',
            mode="eval",
        ).body
        self.assertEqual(
            ast.dump(tts_provider, include_attributes=False),
            ast.dump(expected_tts_provider, include_attributes=False),
        )
        tts_call = next(
            keyword.value for keyword in session_call.keywords if keyword.arg == "tts"
        )
        expected_tts = ast.parse("tts.StreamAdapter(tts=tts_provider)", mode="eval").body
        self.assertEqual(
            ast.dump(tts_call, include_attributes=False),
            ast.dump(expected_tts, include_attributes=False),
        )
        self.assertIn('TTS_VOICE = "21m00Tcm4TlvDq8ikWAM"', source)
        self.assertIn("Rachel", (Path(__file__).resolve().parents[1] / "README.md").read_text())
        self.assertIn("MENTAT_VOICE_TTS_VOICE", (Path(__file__).resolve().parents[1] / "README.md").read_text())
        self.assertNotIn("inference.", source)
        self.assertNotIn("websocket", source.lower())
        self.assertNotIn("tts_text_transforms=", source)
        self.assertIn("async def on_user_turn_completed(", source)
        self.assertNotIn("GPTLive", source)
        self.assertNotIn("openai.realtime", source)

    def test_each_user_turn_streams_backend_text_directly_to_speech(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        completed = source.split("    async def _speak_turn(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        self.assertIn("self._backend_text(", completed)
        self.assertIn("self.session.say(", completed)
        self.assertIn("allow_interruptions=True", completed)
        self.assertIn("handle.wait_for_playout()", completed)
        self.assertIn("handle.interrupted", completed)
        self.assertIn("handle.exception()", completed)
        self.assertIn("await backend_text.aclose()", completed)
        dispatch = source.split("    async def on_user_turn_completed(", 1)[1].split(
            "    async def aclose(", 1
        )[0]
        self.assertIn("asyncio.create_task(self._run_turn(question, chat_ctx))", dispatch)
        backend = source.split("    async def _backend_text(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        self.assertIn('f"{self._mentat_url}/v1/conversation"', backend)
        self.assertIn("turn_request(", backend)
        self.assertIn("self._room_name", backend)
        self.assertIn("envelope", backend)
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
            "turn_request": lambda *_args, **_kwargs: {},
            "write_turn_marker": Mock(),
            "logger": Mock(),
            "SEND_SMS_TOOL": "mcp__mentat__send_sms",
            "SET_VOICE_MODE_TOOL": "mcp__mentat__set_voice_mode",
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
            _voice_mode="normal",
            _voice_language="en",
            _voice_mode_note=None,
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
            "turn_request": lambda *_args, **_kwargs: {},
            "write_turn_marker": Mock(),
            "logger": Mock(),
            "SEND_SMS_TOOL": "mcp__mentat__send_sms",
            "SET_VOICE_MODE_TOOL": "mcp__mentat__set_voice_mode",
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
            _voice_mode="normal",
            _voice_language="en",
            _voice_mode_note=None,
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
        done_line = b'{"kind":"done"}\n'
        cases = (
            (503, [], [namespace["CONSULT_FAILED"]]),
            (200, [error_line], [namespace["CONSULT_FAILED"]]),
            (200, [text_line], ["Partial answer.", namespace["CONSULT_FAILED"]]),
            (200, [done_line], [namespace["CONSULT_FAILED"]]),
        )
        for status, chunks, expected in cases:
            with self.subTest(status=status, chunks=chunks):
                response = FakeResponse(status, chunks)
                agent._sms_consent = False
                self.assertEqual(asyncio.run(collect()), expected)
                policy.turn_done.assert_not_called()

    def test_lone_turn_starts_backend_without_waiting_for_a_coalesce_window(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FrontAgent")
        methods = [node for node in front_agent.body if isinstance(node, ast.AsyncFunctionDef) and node.name in {"on_user_turn_completed", "aclose"}]
        method = methods[0]
        self.assertIn("self._run_turn(", ast.unparse(method))
        started = asyncio.Event()

        async def run_turn(question, _chat_ctx):
            self.assertEqual(question, "Tell me now")
            started.set()
            await asyncio.Event().wait()

        namespace = {"asyncio": asyncio, "time": time, "uuid4": lambda: SimpleNamespace(hex="turn-id")}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _closed=False,
            _turn_task=None,
            _turn_text="",
            _turn_started_at=None,
            _run_turn=run_turn,
        )

        async def submit_lone_turn():
            await namespace["on_user_turn_completed"](
                agent, SimpleNamespace(items=[]), SimpleNamespace(text_content="Tell me now")
            )
            await asyncio.wait_for(started.wait(), timeout=0.1)
            self.assertFalse(agent._turn_task.done())
            await namespace["aclose"](agent)

        asyncio.run(submit_lone_turn())

    def test_continuation_cancels_active_turn_and_reposts_combined_transcript(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FrontAgent")
        methods = [node for node in front_agent.body if isinstance(node, ast.AsyncFunctionDef) and node.name in {"on_user_turn_completed", "aclose"}]
        method = methods[0]
        self.assertIn("self._run_turn(", ast.unparse(method))
        started = [asyncio.Event() for _ in range(3)]
        posted = []
        now = [100.0]

        async def run_turn(question, _chat_ctx):
            posted.append(question)
            started[len(posted) - 1].set()
            await asyncio.Event().wait()

        namespace = {"asyncio": asyncio, "time": SimpleNamespace(monotonic=lambda: now[0]), "uuid4": lambda: SimpleNamespace(hex="turn-id"), "TURN_CONTINUATION_WINDOW": 3.0}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _closed=False,
            _turn_task=None,
            _turn_text="",
            _turn_started_at=None,
            _run_turn=run_turn,
        )

        async def submit_continuation():
            chat_ctx = SimpleNamespace(items=[])
            await namespace["on_user_turn_completed"](agent, chat_ctx, SimpleNamespace(text_content="Text this number"))
            first_started_at = agent._turn_started_at
            await asyncio.wait_for(started[0].wait(), timeout=1)
            now[0] = 102.9
            await namespace["on_user_turn_completed"](agent, chat_ctx, SimpleNamespace(text_content="I will be there at six"))
            self.assertEqual(agent._turn_started_at, first_started_at)
            await asyncio.wait_for(started[1].wait(), timeout=1)
            now[0] = 103.1
            await namespace["on_user_turn_completed"](agent, chat_ctx, SimpleNamespace(text_content="A separate new request"))
            await asyncio.wait_for(started[2].wait(), timeout=1)
            await namespace["aclose"](agent)

        asyncio.run(submit_continuation())
        self.assertEqual(
            posted,
            [
                "Text this number",
                "Text this number I will be there at six",
                "A separate new request",
            ],
        )

    def test_continuation_closes_backend_and_interrupts_provisional_speech(self):
        from collections.abc import AsyncGenerator
        from types import MethodType

        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        methods = [
            node for node in front_agent.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name in {"on_user_turn_completed", "aclose", "_run_turn", "_speak_turn"}
        ]
        now = [100.0]
        namespace = {
            "AsyncGenerator": AsyncGenerator,
            "asyncio": asyncio,
            "time": SimpleNamespace(monotonic=lambda: now[0]),
            "TURN_CONTINUATION_WINDOW": 3.0,
            "uuid4": lambda: SimpleNamespace(hex="turn-id"),
            "AudioConfig": lambda *_args, **_kwargs: object(),
            "EARCON_PATH": Path("earcon.wav"),
            "logger": Mock(),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(agent_path), "exec"), namespace)

        async def exercise():
            posted = []
            spoken = []
            handles = []
            first_post = asyncio.Event()
            first_speech = asyncio.Event()
            first_closed = asyncio.Event()
            hold_backend = asyncio.Event()
            hold_playout = asyncio.Event()

            async def backend_text(question, _turn_id, _chat_ctx):
                posted.append(question)
                if len(posted) == 1:
                    first_post.set()
                    try:
                        yield "Provisional answer."
                        await hold_backend.wait()
                    finally:
                        first_closed.set()
                else:
                    yield "Combined answer."

            class SpeechHandle:
                def __init__(self, task):
                    self.task = task
                    self.interrupted = False

                async def wait_for_playout(self):
                    await self.task

                def interrupt(self):
                    self.interrupted = True
                    self.task.cancel()

                def exception(self):
                    return None

            class Session:
                def say(self, source, *, allow_interruptions):
                    assert allow_interruptions
                    provisional = not handles

                    async def capture():
                        async for text in source:
                            if provisional:
                                first_speech.set()
                                await hold_playout.wait()
                            spoken.append(text)

                    handle = SpeechHandle(asyncio.create_task(capture()))
                    handles.append(handle)
                    return handle

            agent = SimpleNamespace(
                _closed=False,
                _turn_task=None,
                _turn_text="",
                _turn_started_at=None,
                _ending_policy=SimpleNamespace(delegation_started=Mock()),
                _ending_changed=Mock(),
                _background=SimpleNamespace(play=Mock()),
                _backend_text=backend_text,
                session=Session(),
            )
            agent._speak_turn = MethodType(namespace["_speak_turn"], agent)
            agent._run_turn = MethodType(namespace["_run_turn"], agent)
            callback = MethodType(namespace["on_user_turn_completed"], agent)
            chat_ctx = SimpleNamespace(items=[])
            await callback(chat_ctx, SimpleNamespace(text_content="Text this number"))
            await asyncio.wait_for(first_post.wait(), timeout=1)
            await asyncio.wait_for(first_speech.wait(), timeout=1)
            now[0] = 102.0
            await callback(chat_ctx, SimpleNamespace(text_content="I will be there at six"))
            await asyncio.wait_for(agent._turn_task, timeout=1)
            await namespace["aclose"](agent)
            return posted, spoken, handles, first_closed.is_set()

        posted, spoken, handles, first_closed = asyncio.run(exercise())
        self.assertEqual(posted, ["Text this number", "Text this number I will be there at six"])
        self.assertTrue(first_closed)
        self.assertEqual(len(handles), 2)
        self.assertTrue(handles[0].interrupted)
        self.assertNotIn("Provisional answer.", spoken)
        self.assertIn("Combined answer.", spoken)
        self.assertFalse(handles[1].interrupted)

    def test_late_barge_in_cancels_without_reusing_the_old_transcript(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FrontAgent")
        method = next(node for node in front_agent.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "on_user_turn_completed")
        self.assertIn("self._run_turn(", ast.unparse(method))
        started = asyncio.Event()
        posted = []

        async def run_turn(question, _chat_ctx):
            posted.append(question)
            if len(posted) == 1:
                started.set()
                await asyncio.Event().wait()

        namespace = {"asyncio": asyncio, "time": time, "uuid4": lambda: SimpleNamespace(hex="turn-id"), "TURN_CONTINUATION_WINDOW": 3.0}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _closed=False,
            _turn_task=None,
            _turn_text="",
            _turn_started_at=time.monotonic() - 4,
            _run_turn=run_turn,
        )

        async def submit_late_barge_in():
            chat_ctx = SimpleNamespace(items=[])
            await namespace["on_user_turn_completed"](agent, chat_ctx, SimpleNamespace(text_content="Old question"))
            await asyncio.wait_for(started.wait(), timeout=1)
            agent._turn_started_at = time.monotonic() - 4
            await namespace["on_user_turn_completed"](agent, chat_ctx, SimpleNamespace(text_content="New words only"))
            await asyncio.sleep(0)

        asyncio.run(submit_late_barge_in())
        self.assertEqual(posted, ["Old question", "New words only"])

    def test_close_cancels_and_drains_detached_turn_and_rejects_later_posts(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        source = agent_path.read_text()
        close_audio = source.split("    async def _close_audio()", 1)[1].split(
            "    @session.on(\"close\")", 1
        )[0]
        self.assertIn("await agent.aclose()", close_audio)
        tree = ast.parse(source)
        front_agent = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FrontAgent")
        methods = [node for node in front_agent.body if isinstance(node, ast.AsyncFunctionDef) and node.name in {"on_user_turn_completed", "aclose"}]
        self.assertIn("self._run_turn(", ast.unparse(methods[0]))
        started = asyncio.Event()
        drained = asyncio.Event()
        posted = []

        async def run_turn(question, _chat_ctx):
            posted.append(question)
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                drained.set()

        namespace = {"asyncio": asyncio, "time": time, "uuid4": lambda: SimpleNamespace(hex="turn-id")}
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _closed=False,
            _turn_task=None,
            _turn_text="",
            _turn_started_at=None,
            _run_turn=run_turn,
        )

        async def close_during_turn():
            chat_ctx = SimpleNamespace(items=[])
            await namespace["on_user_turn_completed"](agent, chat_ctx, SimpleNamespace(text_content="Do not post after close"))
            await asyncio.wait_for(started.wait(), timeout=1)
            await namespace["aclose"](agent)
            await namespace["on_user_turn_completed"](agent, chat_ctx, SimpleNamespace(text_content="Closed already"))

        asyncio.run(close_during_turn())
        self.assertTrue(drained.is_set())
        self.assertEqual(posted, ["Do not post after close"])

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
            and node.name == "_speak_turn"
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
                namespace["_speak_turn"](agent, "Tell me now", SimpleNamespace(items=[]))
            )
            await asyncio.wait_for(backend_open.wait(), timeout=1)
            await asyncio.wait_for(speech_started.wait(), timeout=1)
            self.assertFalse(backend_closed.is_set())
            session.handle.interrupt()
            await asyncio.wait_for(turn, timeout=1)

        asyncio.run(interrupt_while_backend_open())
        self.assertTrue(session.handle.interrupted)
        self.assertTrue(backend_closed.is_set())

    def test_completed_empty_backend_stream_ends_without_stop_async_iteration(self):
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
            and node.name == "_speak_turn"
        )
        backend_closed = asyncio.Event()

        async def backend_text(_question, _turn_id, _chat_ctx):
            try:
                if False:
                    yield "unreachable"
            finally:
                backend_closed.set()

        namespace = {
            "AsyncGenerator": AsyncGenerator,
            "asyncio": asyncio,
            "uuid4": lambda: SimpleNamespace(hex="turn-id"),
            "AudioConfig": lambda *_args, **_kwargs: object(),
            "EARCON_PATH": Path("earcon.wav"),
            "logger": Mock(),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _ending_policy=SimpleNamespace(delegation_started=Mock()),
            _ending_changed=Mock(),
            _background=SimpleNamespace(play=Mock()),
            _backend_text=backend_text,
            session=SimpleNamespace(say=Mock(side_effect=AssertionError("no speech expected"))),
        )

        asyncio.run(
            namespace["_speak_turn"](agent, "Tell me now", SimpleNamespace(items=[]))
        )
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
            and node.name == "_speak_turn"
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
                namespace["_speak_turn"](agent, "Tell me now", SimpleNamespace(items=[]))
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
            and node.name in {"_backend_text", "_speak_turn"}
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
            "turn_request": lambda *_args, **_kwargs: {},
            "write_turn_marker": Mock(),
            "logger": Mock(),
            "SEND_SMS_TOOL": "mcp__mentat__send_sms",
            "SET_VOICE_MODE_TOOL": "mcp__mentat__set_voice_mode",
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
            _voice_mode="normal",
            _voice_language="en",
            _voice_mode_note=None,
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
            namespace["_speak_turn"](agent, "Check the time", SimpleNamespace(items=[]))
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
            "turn_request": lambda *_args, **_kwargs: {},
            "write_turn_marker": Mock(),
            "logger": Mock(),
            "SEND_SMS_TOOL": "mcp__mentat__send_sms",
            "SET_VOICE_MODE_TOOL": "mcp__mentat__set_voice_mode",
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
            _voice_mode="normal",
            _voice_language="en",
            _voice_mode_note=None,
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

    def test_agent_uses_local_semantic_turn_detection_and_keeps_vad(self):
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
        vad = next(
            keyword.value for keyword in session_call.keywords if keyword.arg == "vad"
        )
        self.assertEqual(ast.unparse(vad), "ctx.proc.userdata['vad']")

        stt = next(
            node.value
            for node in entrypoint.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "stt" for target in node.targets)
        )
        stt_vad = next(
            keyword.value for keyword in stt.keywords if keyword.arg == "vad"
        )
        self.assertEqual(ast.unparse(stt_vad), "ctx.proc.userdata['vad']")

        turn_handling = next(
            keyword.value for keyword in session_call.keywords
            if keyword.arg == "turn_handling"
        )
        self.assertEqual(
            ast.unparse(turn_handling),
            "{'turn_detection': MultilingualModel(), "
            "'endpointing': {'min_delay': 0.5, 'max_delay': 3.0}}",
        )
        self.assertIn(
            "from livekit.plugins.turn_detector.multilingual import MultilingualModel",
            source,
        )
        self.assertNotIn("turn_detector.MultilingualModel", source)

    def test_input_audio_recording_is_opt_in_committed_and_keeps_real_stt_frames(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        recorder = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "InputAudioRecorder"
        )
        namespace = {
            "os": os,
            "Path": Path,
            "wave": __import__("wave"),
            "re": re,
            "Any": object,
            "logger": Mock(),
        }
        exec(compile(ast.Module(body=[recorder], type_ignores=[]), str(agent_path), "exec"), namespace)
        frame = SimpleNamespace(
            data=bytearray(b"\x01\x00\xfe\xff"),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=2,
        )

        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {}, clear=True):
                disabled = namespace["InputAudioRecorder"]("eval-room")
                disabled.capture(frame)
                disabled.commit("ignored")
            self.assertEqual(list(Path(temporary).iterdir()), [])

            with patch.dict(os.environ, {"MENTAT_VOICE_INPUT_RECORD_DIR": temporary}, clear=True):
                active = namespace["InputAudioRecorder"]("eval-room", temporary)
                active.begin()
                active.capture(frame)
                active.capture(frame)
                self.assertEqual(list(Path(temporary).iterdir()), [])
                active.commit("first final transcript")
                # A later STT input segment discards uncommitted prior frames.
                active.capture(frame)
                active.begin()
                active.capture(frame)
                active.commit("second final transcript")
                # An unfinished capture is never published as a turn.
                active.begin()
                active.capture(frame)
                restarted = namespace["InputAudioRecorder"]("eval-room", temporary)
                restarted.begin()
                restarted.capture(frame)
                restarted.commit("after worker restart")

            wavs = sorted(Path(temporary).glob("*.wav"))
            self.assertEqual([path.name for path in wavs], [
                "eval-room-turn-001.wav",
                "eval-room-turn-002.wav",
                "eval-room-turn-003.wav",
            ])
            expected_audio = {
                "eval-room-turn-001.wav": bytes(frame.data) * 2,
                "eval-room-turn-002.wav": bytes(frame.data),
                "eval-room-turn-003.wav": bytes(frame.data),
            }
            for path in wavs:
                with namespace["wave"].open(str(path), "rb") as audio:
                    self.assertEqual(audio.getnchannels(), 1)
                    self.assertEqual(audio.getsampwidth(), 2)
                    self.assertEqual(audio.getframerate(), 16000)
                    self.assertEqual(audio.readframes(audio.getnframes()), expected_audio[path.name])
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            transcripts = sorted(Path(temporary).glob("*.txt"))
            self.assertEqual(
                [path.read_text() for path in transcripts],
                [
                    "first final transcript",
                    "second final transcript",
                    "after worker restart",
                ],
            )
            self.assertTrue(all(path.stat().st_mode & 0o777 == 0o600 for path in transcripts))

    def test_front_agent_records_frames_before_stt_and_commits_final_text(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        tree = ast.parse(source)
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        stt_node = next(
            node for node in front_agent.body
            if isinstance(node, ast.FunctionDef) and node.name == "stt_node"
        )
        callback = next(
            node for node in front_agent.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "on_user_turn_completed"
        )
        stt_source = ast.unparse(stt_node)
        self.assertIn("self._input_audio.capture", stt_source)
        self.assertLess(stt_source.index("capture(frame)"), stt_source.index("yield frame"))
        self.assertIn("return super().stt_node(audio, model_settings)", stt_source)
        self.assertIn("return super().stt_node(recorded_audio(), model_settings)", stt_source)
        self.assertIn("input_audio.commit(question)", ast.unparse(callback))
        self.assertIn("text_content", ast.unparse(callback))
        init = next(
            node for node in front_agent.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self.assertIn("if input_audio_dir else None", ast.unparse(init))
        self.assertNotIn("user_input_transcribed", source)

    def test_voice_mode_result_updates_call_local_stt_and_tts_or_rolls_back(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        apply_mode = next(
            node for node in front_agent.body
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_apply_voice_mode"
        )

        class Provider:
            def __init__(self, *, fail_on=None):
                self.calls = []
                self.fail_on = fail_on

            def update_options(self, **options):
                self.calls.append(options)
                if options == self.fail_on:
                    raise RuntimeError("provider update failed")

        class Resolver:
            def __init__(self):
                self.calls = []

            def resolve(self, language, default_voice):
                self.calls.append((language, default_voice))
                return {"es": "library-spanish", "en": "saved-english"}.get(
                    language, default_voice
                )

        namespace = {"json": json, "re": re, "logger": Mock()}
        exec(compile(ast.Module(body=[apply_mode], type_ignores=[]), str(agent_path), "exec"), namespace)

        def make_agent(tts=None):
            resolver = Resolver()
            return SimpleNamespace(
                _voice_mode="normal",
                _voice_language="en",
                _voice_id="default-voice",
                _default_voice="default-voice",
                _stt=Provider(),
                _tts=tts or Provider(),
                _voice_resolver=resolver,
                _voice_mode_note=None,
            )

        spanish = '{"voice_mode":{"mode":"conversation","language":"es"}}'
        agent = make_agent()
        self.assertTrue(asyncio.run(namespace["_apply_voice_mode"](agent, spanish)))
        self.assertEqual(agent._voice_mode, "conversation")
        self.assertEqual(agent._voice_language, "es")
        self.assertEqual(agent._voice_id, "library-spanish")
        self.assertEqual(agent._voice_resolver.calls, [("es", "default-voice")])
        self.assertEqual(agent._stt.calls, [{"language": "es"}])
        self.assertEqual(
            agent._tts.calls,
            [{"voice_id": "library-spanish", "language": "es"}],
        )

        separate_call = make_agent()
        self.assertEqual(
            (separate_call._voice_mode, separate_call._voice_language, separate_call._voice_id),
            ("normal", "en", "default-voice"),
        )
        restore = '{"voice_mode":{"mode":"normal","language":"en"}}'
        self.assertTrue(asyncio.run(namespace["_apply_voice_mode"](agent, restore)))
        self.assertEqual(agent._voice_mode, "normal")
        self.assertEqual(agent._voice_language, "en")
        self.assertEqual(agent._voice_id, "default-voice")
        self.assertEqual(agent._voice_resolver.calls, [("es", "default-voice")])
        self.assertEqual(agent._stt.calls[-1], {"language": "en"})
        self.assertEqual(
            agent._tts.calls[-1], {"voice_id": "default-voice", "language": "en"}
        )

        calls_before_invalid = (len(agent._stt.calls), len(agent._tts.calls))
        self.assertFalse(
            asyncio.run(namespace["_apply_voice_mode"](
                agent, '{"voice_mode":{"mode":"unknown","language":"es"}}'
            ))
        )
        self.assertFalse(asyncio.run(namespace["_apply_voice_mode"](agent, spanish, is_error=True)))
        self.assertEqual((len(agent._stt.calls), len(agent._tts.calls)), calls_before_invalid)
        self.assertEqual((agent._voice_mode, agent._voice_language), ("normal", "en"))

        failing_tts = Provider(fail_on={"voice_id": "library-spanish", "language": "es"})
        failed = make_agent(failing_tts)
        self.assertFalse(asyncio.run(namespace["_apply_voice_mode"](failed, spanish)))
        self.assertEqual((failed._voice_mode, failed._voice_language, failed._voice_id),
                         ("normal", "en", "default-voice"))
        self.assertEqual(failed._stt.calls, [{"language": "es"}, {"language": "en"}])
        self.assertEqual(
            failing_tts.calls,
            [
                {"voice_id": "library-spanish", "language": "es"},
                {"voice_id": "default-voice", "language": "en"},
            ],
        )
        self.assertTrue(failed._voice_mode_note)
        self.assertLessEqual(len(failed._voice_mode_note), 160)

    def test_voice_mode_defaults_and_request_use_active_call_mode(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        tree = ast.parse(source)
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        init = next(
            node for node in front_agent.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        init_source = ast.unparse(init)
        backend = source.split("    async def _backend_text(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        entry = source.split("async def entrypoint(", 1)[1]
        self.assertIn("self._voice_mode = 'normal'", init_source)
        self.assertIn("self._voice_language = 'en'", init_source)
        self.assertIn("voice_mode=self._voice_mode", backend)
        self.assertIn("voice_language=self._voice_language", backend)
        self.assertIn("_apply_voice_mode(", backend)
        self.assertIn("item.content", backend)
        self.assertIn("is_error=item.is_error", backend)
        self.assertIn("stt=stt", entry)
        self.assertIn("tts=tts.StreamAdapter(tts=tts_provider)", entry)
        self.assertIn("voice_resolver=voice_resolver", entry)
        self.assertIn("_voice_mode_note", backend)

    def test_tool_result_changes_meta_before_the_next_backend_turn_and_speech(self):
        from collections.abc import AsyncGenerator
        from types import MethodType

        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        methods = [
            node for node in front_agent.body
            if isinstance(node, ast.AsyncFunctionDef)
            and node.name in {"_apply_voice_mode", "_backend_text"}
        ]

        class FakeToolResult:
            def __init__(self, name, is_error, content):
                self.name = name
                self.is_error = is_error
                self.content = content

        class FakeTurnDone:
            pass

        class FakeTurnStream:
            done = False

            def feed(self, data):
                event = json.loads(data)
                if event["kind"] == "text_delta":
                    return [event["text"]]
                if event["kind"] == "tool_result":
                    return [FakeToolResult(event["tool"], event["is_error"], event["content"])]
                if event["kind"] == "done":
                    self.done = True
                    return [FakeTurnDone()]
                return []

        class FakeContent:
            def __init__(self, chunks):
                self.chunks = chunks

            async def iter_any(self):
                for chunk in self.chunks:
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
                self.response = response

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def post(self, *_args, **_kwargs):
                return self.response

        timeline = []
        requests = []
        response = None

        class Provider:
            def update_options(self, **options):
                timeline.append(("update", options))

        class Resolver:
            def resolve(self, language, default_voice):
                return "library-spanish" if language == "es" else default_voice

        def make_turn_request(*args, **kwargs):
            requests.append((args, kwargs))
            return {}

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
            "turn_request": make_turn_request,
            "write_turn_marker": Mock(),
            "logger": Mock(),
            "SEND_SMS_TOOL": "mcp__mentat__send_sms",
            "SET_VOICE_MODE_TOOL": "mcp__mentat__set_voice_mode",
            "END_CONVERSATION_TOOL": "mcp__mentat__end_conversation",
            "ToolResult": FakeToolResult,
            "ToolStart": type("FakeToolStart", (), {}),
            "TurnDone": FakeTurnDone,
            "TurnFailure": type("FakeTurnFailure", (), {}),
            "time": time,
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _sms_consent=False,
            _voice_card="voice card",
            _voice_mode="normal",
            _voice_language="en",
            _voice_id="default-voice",
            _default_voice="default-voice",
            _voice_mode_note=None,
            _voice_resolver=Resolver(),
            _stt=Provider(),
            _tts=Provider(),
            _room_name="call-one",
            _mentat_url="http://127.0.0.1:8484",
            _ending_policy=SimpleNamespace(
                tool_result_seen=Mock(), turn_done=Mock(), describe=lambda: "test policy"
            ),
            _ending_changed=Mock(),
        )
        agent._apply_voice_mode = MethodType(namespace["_apply_voice_mode"], agent)
        backend_text = MethodType(namespace["_backend_text"], agent)
        mode_result = json.dumps({"voice_mode": {"mode": "conversation", "language": "es"}})

        async def run_turn(chunks, question):
            nonlocal response
            response = FakeResponse(chunks)
            speech = []
            async for text in backend_text(question, "turn-id", SimpleNamespace(items=[])):
                timeline.append(("speech", text))
                speech.append(text)
            return speech

        first_wire = [
            json.dumps({"kind": "text_delta", "text": "Switching now."}).encode(),
            json.dumps({
                "kind": "tool_result",
                "tool": "mcp__mentat__set_voice_mode",
                "is_error": False,
                "content": mode_result,
            }).encode(),
            json.dumps({"kind": "text_delta", "text": "Continuando en español."}).encode(),
            b'{"kind":"done"}',
        ]
        second_wire = [
            b'{"kind":"text_delta","text":"Siguiente turno."}',
            b'{"kind":"done"}',
        ]
        self.assertEqual(
            asyncio.run(run_turn(first_wire, "Switch to Spanish")),
            ["Switching now.", "Continuando en español."],
        )
        self.assertLess(
            timeline.index(("update", {"language": "es"})),
            timeline.index(("speech", "Continuando en español.")),
        )
        self.assertEqual(
            requests[0][1], {"voice_mode": "normal", "voice_language": "en"}
        )
        self.assertEqual(
            asyncio.run(run_turn(second_wire, "Continue")), ["Siguiente turno."]
        )
        self.assertEqual(
            requests[1][1], {"voice_mode": "conversation", "voice_language": "es"}
        )

        class FailingTTS(Provider):
            def update_options(self, **options):
                super().update_options(**options)
                if options.get("language") == "fr":
                    raise RuntimeError("synthetic provider failure")

        agent._tts = FailingTTS()
        failed_mode = json.dumps({"voice_mode": {"mode": "conversation", "language": "fr"}})
        failed_wire = [
            b'{"kind":"text_delta","text":"Trying another language."}',
            json.dumps({
                "kind": "tool_result",
                "tool": "mcp__mentat__set_voice_mode",
                "is_error": False,
                "content": failed_mode,
            }).encode(),
            b'{"kind":"done"}',
        ]
        self.assertEqual(
            asyncio.run(run_turn(failed_wire, "Switch again")),
            ["Trying another language."],
        )
        note = agent._voice_mode_note
        self.assertIsNotNone(note)
        self.assertEqual((agent._voice_mode, agent._voice_language), ("conversation", "es"))
        self.assertEqual(
            asyncio.run(run_turn(second_wire, "Continue after failure")),
            ["Siguiente turno."],
        )
        self.assertEqual(requests[3][0][1].count(note), 1)
        self.assertEqual(
            requests[3][1], {"voice_mode": "conversation", "voice_language": "es"}
        )
        self.assertIsNone(agent._voice_mode_note)
        self.assertEqual(asyncio.run(run_turn(second_wire, "Another turn")), ["Siguiente turno."])
        self.assertNotIn(note, requests[4][0][1])


if __name__ == "__main__":
    unittest.main()
