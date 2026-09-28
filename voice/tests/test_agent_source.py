"""Source contract for the runtime-only LiveKit glue."""

import ast
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


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

    def test_each_delegation_writes_only_private_timed_eval_jsonl_when_opted_in(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        callback = source.split("    def _on_delegation_created(", 1)[1].split(
            "    def _on_delegation_error(", 1
        )[0]
        self.assertIn('os.environ.get("MENTAT_EVAL_DELEGATION_LOG")', callback)
        self.assertIn('"room": self._room_name', callback)
        self.assertIn('"id": delegation.id', callback)
        self.assertIn('"created_at": time.time()', callback)
        self.assertIn('separators=(",", ":")', callback)
        marker = callback.split("marker = json.dumps(", 1)[1].split("try:", 1)[0]
        self.assertNotIn("pending_transcript", marker)
        self.assertNotIn("credential", marker)
        self.assertNotIn('"eval-delegation %s"', callback)

    def test_delegation_marker_is_opt_in_compact_jsonl_without_transcript_or_credentials(self):
        agent_path = Path(__file__).resolve().parents[1] / "agent.py"
        tree = ast.parse(agent_path.read_text())
        front_agent = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FrontAgent"
        )
        method = next(
            node for node in front_agent.body
            if isinstance(node, ast.FunctionDef) and node.name == "_on_delegation_created"
        )
        namespace = {
            "os": os,
            "json": json,
            "time": time,
            "Path": Path,
            "logger": Mock(),
            "AudioConfig": lambda *_args: object(),
            "EARCON_PATH": agent_path.parent / "assets" / "earcon.wav",
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(agent_path), "exec"), namespace)
        agent = SimpleNamespace(
            _room_name="eval-room",
            _ending_policy=SimpleNamespace(delegation_started=Mock()),
            _ending_changed=Mock(),
            _background=SimpleNamespace(play=Mock()),
            _delegations=SimpleNamespace(start=Mock()),
        )
        delegation = SimpleNamespace(id="same-id", pending_transcript="private transcript")

        with tempfile.TemporaryDirectory() as temporary:
            marker_path = Path(temporary) / "delegations.jsonl"
            with patch.dict(os.environ, {}, clear=True):
                namespace["_on_delegation_created"](agent, delegation)
            self.assertFalse(marker_path.exists())

            with patch.dict(os.environ, {"MENTAT_EVAL_DELEGATION_LOG": str(marker_path)}, clear=True):
                namespace["_on_delegation_created"](agent, delegation)
                namespace["_on_delegation_created"](
                    agent, SimpleNamespace(id="second-id", pending_transcript="another private transcript")
                )

            lines = marker_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            markers = [json.loads(line) for line in lines]
            self.assertTrue(all(set(marker) == {"room", "id", "created_at"} for marker in markers))
            self.assertEqual([marker["room"] for marker in markers], ["eval-room", "eval-room"])
            self.assertEqual([marker["id"] for marker in markers], ["same-id", "second-id"])
            self.assertTrue(all(isinstance(marker["created_at"], (int, float)) for marker in markers))
            self.assertNotIn("private transcript", "\n".join(lines))
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

    def test_sms_buffering_is_selective_and_keeps_non_sms_chunk_streaming(self):
        source = (Path(__file__).resolve().parents[1] / "agent.py").read_text()
        stream_backend = source.split("    async def _stream_backend(", 1)[1].split(
            "\n\ndef log_turn_metrics", 1
        )[0]
        self.assertIn("SmsCommentaryBuffer", stream_backend)
        self.assertIn("sms_buffer.add(item)", stream_backend)
        self.assertIn("append_commentary(chunk)", stream_backend)
        self.assertIn("sms_buffer.finish()", stream_backend)
        self.assertIn("self._sms_consent and (is_sms_followup(question) or sms_declined)", stream_backend)
        self.assertIn('item.name == SEND_SMS_TOOL', stream_backend)

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
