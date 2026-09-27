"""Offline lifecycle tests for the isolated voice-eval development stack."""

import io
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
from unittest.mock import patch

from voice.evals.dev_stack import (
    DevStack,
    _MCP_REWRITE_SOURCE,
    _PRIVATE_CREDENTIAL_SOURCE,
    _RUN_VOICE_SCRIPT,
    _SETUP_SCRIPT,
    _START_WORKER_SCRIPT,
)


CHECKOUT = Path(__file__).resolve().parents[2]


class DevStackTest(unittest.TestCase):
    def setUp(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.status = 200
        response.__enter__.return_value.read.return_value = b'{"status":"ok"}\n'
        self._health_patch = patch("urllib.request.urlopen", return_value=response)
        self._health_patch.start()
        self.addCleanup(self._health_patch.stop)

    def test_requires_explicit_opt_in_before_any_remote_action(self):
        run = unittest.mock.Mock()
        stack = DevStack(checkout=CHECKOUT, run=run)

        with self.assertRaisesRegex(RuntimeError, "explicit opt-in"):
            stack.__enter__()

        run.assert_not_called()

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_default_live_stack_selects_and_propagates_free_remote_ports(self, popen):
        popen.return_value = unittest.mock.Mock(poll=lambda: None)
        calls = []
        selected_ports = {"dev_port": 49151, "health_port": 49152}

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.ports\n", "")
            if "MENTAT_EVAL_PORT_PROBE" in kwargs.get("input", ""):
                probe = kwargs["input"]
                self.assertIn('("0.0.0.0", 0)', probe)
                self.assertIn("for _ in range(2)", probe)
                self.assertIn("for sock in sockets:", probe)
                return subprocess.CompletedProcess(args, 0, json.dumps(selected_ports), "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with DevStack(checkout=CHECKOUT, opt_in=True, run=run) as stack:
            self.assertEqual(stack.dev_port, selected_ports["dev_port"])
            self.assertEqual(stack.health_port, selected_ports["health_port"])
            local_port = int(urlsplit(stack.url).port)
            stack.start_worker("ports-test-room")
            stack.run_voice(["python3", "caller.py"], token="issued-token", livekit_url="wss://lk.invalid")

        self.assertNotIn(stack.dev_port, (8485, 8486))
        self.assertNotIn(stack.health_port, (8485, 8486))
        self.assertNotEqual(stack.dev_port, stack.health_port)
        remote_calls = [
            (args, kwargs) for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
        ]
        setup_args, daemon_setup = next(
            (args, kwargs["input"])
            for args, kwargs in remote_calls
            if '"MENTAT_LISTEN"' in kwargs.get("input", "")
        )
        self.assertEqual(
            setup_args[-2:],
            [str(selected_ports["dev_port"]), str(selected_ports["health_port"])],
        )
        self.assertIn('"MENTAT_LISTEN": f"127.0.0.1:{dev_port}"', daemon_setup)
        worker_args, _ = next(
            (args, kwargs) for args, kwargs in remote_calls if "--room" in kwargs.get("input", "")
        )
        self.assertEqual(worker_args[-3:-1], [str(selected_ports["dev_port"]), str(selected_ports["health_port"])])
        caller_script = next(
            kwargs["input"] for _, kwargs in remote_calls if "MENTAT_VOICE_GRANT" in kwargs.get("input", "")
        )
        self.assertIn(str(selected_ports["dev_port"]), caller_script)
        self.assertIn(str(selected_ports["health_port"]), caller_script)
        self.assertIn(
            f"127.0.0.1:{local_port}:127.0.0.1:{selected_ports['dev_port']}",
            popen.call_args.args[0],
        )
        self.assertNotIn("systemctl stop mentatd", "\n".join(kwargs.get("input", "") for _, kwargs in remote_calls))
        self.assertIn("systemctl start mentat-voice", "\n".join(kwargs.get("input", "") for _, kwargs in remote_calls))

    def test_remote_port_selection_failure_cleans_up_staging_and_restarts_voice(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.probefailure\n", "")
            if "MENTAT_EVAL_PORT_PROBE" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(args, 0, "invalid probe response", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        stack = DevStack(checkout=CHECKOUT, opt_in=True, run=run)
        with self.assertRaisesRegex(RuntimeError, "remote port probe"):
            stack.__enter__()

        cleanup = [
            kwargs.get("input", "") for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and "cleanup.sh" in kwargs.get("input", "")
        ]
        self.assertEqual(len(cleanup), 1)
        self.assertIn("systemctl start mentat-voice", cleanup[0])
        self.assertIsNone(stack._remote_dir)
        self.assertIsNone(stack._local_port)

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_stages_candidate_and_fake_phone_on_isolated_loopback_daemon(self, popen):
        popen.return_value = unittest.mock.Mock(poll=lambda: None)
        calls = []
        dev_port = 8485

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.test\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with DevStack(checkout=CHECKOUT, opt_in=True, run=run, local_port=0, dev_port=dev_port, health_port=8486) as stack:
            self.assertRegex(stack.url, r"^http://127\.0\.0\.1:\d+$")
            stack.run_remote(["test", "-f", "voice/evals/phone.py"])
            stack.start_worker("android-test-room")

        build = next(call for call in calls if call[0][:2] == ["nix", "build"])
        self.assertIn(".#mentatd", build[0])
        transfers = [call[0] for call in calls if call[0][0] == "scp"]
        self.assertTrue(any("voice/evals/phone.py" in " ".join(args) for args in transfers))
        self.assertTrue(any(
            any(arg.startswith("ultraviolet:") and arg.endswith("/voice/evals/") for arg in args)
            and any(arg.endswith("/voice/evals/runner.py") for arg in args)
            for args in transfers
        ))
        self.assertTrue(any("src" in " ".join(args) and "node_modules" in " ".join(args) for args in transfers))

        remote_scripts = [kwargs.get("input", "") for args, kwargs in calls if args[:2] == ["ssh", "ultraviolet"]]
        setup = "\n".join(script for script in remote_scripts if script)
        self.assertIn("127.0.0.1", setup)
        self.assertIn('"MENTAT_LISTEN": f"127.0.0.1:{dev_port}"', setup)
        self.assertIn('"MENTAT_STATE_PATH": str(dev_dir / "state.json")', setup)
        self.assertIn('"MENTAT_RECORD_DIR": str(dev_dir / "records")', setup)
        self.assertIn('"HOME": str(dev_dir / "home/mentat")', setup)
        self.assertIn('"MENTAT_MCP_CONFIG"', setup)
        self.assertIn('"MENTAT_URL": f"http://127.0.0.1:{dev_port}"', setup)
        self.assertIn('"--reuid=nobody"', setup)
        self.assertIn('voice/evals/phone.py', " ".join(transfers[1]))
        self.assertNotIn("systemctl stop mentatd", setup)
        self.assertIn("systemctl stop mentat-voice", setup)
        self.assertIn("systemctl start mentat-voice", setup)
        worker_script = next(script for script in remote_scripts if "--room" in script)
        self.assertLess(
            worker_script.index("trap 'systemctl start mentat-voice' EXIT"),
            worker_script.index("systemctl stop mentat-voice"),
        )
        self.assertIn("agent.pid", setup)

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_controlmaster_handoff_does_not_abort_a_live_forward(self, popen):
        calls = []
        tunnel_processes = []
        forward = {"available": False}
        refusals = [
            urllib.error.URLError("connection refused"),
            urllib.error.URLError("connection refused"),
        ]
        response = unittest.mock.MagicMock()
        response.__enter__.return_value.status = 200
        response.__enter__.return_value.read.return_value = b'{"status":"ok"}\n'
        health_calls = []

        def start_ssh(args, **kwargs):
            calls.append(args)
            dedicated = ["-o", "ControlMaster=no"] in [args[i:i + 2] for i in range(len(args) - 1)]
            dedicated = dedicated and ["-o", "ControlPath=none"] in [
                args[i:i + 2] for i in range(len(args) - 1)
            ]
            dedicated = dedicated and ["-o", "ExitOnForwardFailure=yes"] in [
                args[i:i + 2] for i in range(len(args) - 1)
            ]
            forward["available"] = True
            tunnel = unittest.mock.Mock(poll=lambda: None if dedicated else 0)
            tunnel_processes.append(tunnel)
            return tunnel

        def urlopen(url, *, timeout):
            health_calls.append((url, timeout))
            self.assertTrue(forward["available"], "the persistent ControlMaster keeps the forward open")
            if refusals:
                raise refusals.pop(0)
            return response

        def run(args, **kwargs):
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.controlmaster\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        popen.side_effect = start_ssh
        with patch("urllib.request.urlopen", side_effect=urlopen), patch("time.sleep") as sleep:
            with DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
                self.assertEqual(len(health_calls), 3)
                self.assertEqual(health_calls[-1][0], f"{stack.url}/healthz")
                sleep.assert_called()

        self.assertEqual(len(calls), 1)
        self.assertIn(["-o", "ControlMaster=no"], [
            calls[0][i:i + 2] for i in range(len(calls[0]) - 1)
        ])
        self.assertIn(["-o", "ControlPath=none"], [
            calls[0][i:i + 2] for i in range(len(calls[0]) - 1)
        ])
        self.assertIn(["-o", "ExitOnForwardFailure=yes"], [
            calls[0][i:i + 2] for i in range(len(calls[0]) - 1)
        ])
        tunnel_processes[0].terminate.assert_called_once()

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_enter_waits_for_tunneled_health_endpoint_before_returning(self, popen):
        tunnel = unittest.mock.Mock(poll=lambda: None)
        popen.return_value = tunnel
        responses = [
            urllib.error.URLError("connection refused"),
            urllib.error.URLError("connection refused"),
            unittest.mock.MagicMock(),
        ]
        responses[-1].__enter__.return_value.status = 200
        responses[-1].__enter__.return_value.read.return_value = b'{"status":"ok"}\n'
        calls = []

        def urlopen(url, *, timeout):
            calls.append((url, timeout))
            response = responses.pop(0)
            if isinstance(response, BaseException):
                raise response
            return response

        def run(args, **kwargs):
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.ready\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("urllib.request.urlopen", side_effect=urlopen), patch("time.sleep") as sleep:
            with DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
                self.assertEqual(len(calls), 3)
                self.assertEqual(calls[-1][0], f"{stack.url}/healthz")
                self.assertTrue(all(timeout > 0 for _, timeout in calls))
                sleep.assert_called()

        tunnel.terminate.assert_called_once()

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_never_ready_endpoint_times_out_and_cleans_up(self, popen):
        tunnel = unittest.mock.Mock(poll=lambda: None)
        popen.return_value = tunnel
        calls = []
        now = [0.0]

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.timeout\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        def sleep(seconds):
            now[0] += seconds

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")) as urlopen:
            with patch("time.monotonic", side_effect=lambda: now[0]), patch("time.sleep", side_effect=sleep):
                with patch("voice.evals.dev_stack._READINESS_TIMEOUT_SECONDS", 0.5, create=True):
                    stack = DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run)
                    with self.assertRaisesRegex(TimeoutError, "health endpoint"):
                        stack.__enter__()

        self.assertGreater(urlopen.call_count, 1)
        self.assertLessEqual(now[0], 0.7)
        tunnel.terminate.assert_called_once()
        self.assertTrue(any(
            "systemctl start mentat-voice" in kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
        ))
        self.assertFalse(stack._entered)
        self.assertIsNone(stack._local_port)

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_interruption_during_health_wait_cleans_up(self, popen):
        tunnel = unittest.mock.Mock(poll=lambda: None)
        popen.return_value = tunnel
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.interrupt\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("connection refused")):
            with patch("time.sleep", side_effect=KeyboardInterrupt):
                stack = DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run)
                with self.assertRaises(KeyboardInterrupt):
                    stack.__enter__()

        tunnel.terminate.assert_called_once()
        self.assertTrue(any(
            "systemctl start mentat-voice" in kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
        ))

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_worker_joins_the_room_returned_by_the_voice_token(self, popen):
        popen.return_value = unittest.mock.Mock(poll=lambda: None)
        calls = []
        room = "android-token-issued-room"

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.room\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
            stack.start_worker(room)

        scripts = [kwargs.get("input", "") for args, kwargs in calls if args[:2] == ["ssh", "ultraviolet"]]
        daemon_script = next(script for script in scripts if '"MENTAT_STATE_PATH"' in script)
        worker_args, worker_kwargs = next(
            (args, kwargs)
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and "--room" in kwargs.get("input", "")
        )
        worker_script = worker_kwargs["input"]
        self.assertNotIn("systemctl stop mentat-voice", daemon_script)
        self.assertIn(room, worker_args)
        self.assertIn('"$ROOM"', worker_script)
        self.assertIn("systemctl stop mentat-voice", worker_script)

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_successive_token_rooms_replace_worker_and_retain_private_environment(self, popen):
        popen.return_value = unittest.mock.Mock(poll=lambda: None)
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.repeat\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
            stack.start_worker("first-token-room")
            stack.start_worker("second-token-room")

        worker_calls = [
            (args, kwargs)
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and "--room" in kwargs.get("input", "")
        ]
        self.assertEqual(len(worker_calls), 2)
        self.assertIn("first-token-room", worker_calls[0][0])
        self.assertIn("second-token-room", worker_calls[1][0])
        self.assertIn('"$ROOM"', worker_calls[0][1]["input"])
        self.assertIn('"$ROOM"', worker_calls[1][1]["input"])
        replacement = worker_calls[1][1]["input"]
        self.assertIn('if [ -f "$DEV_DIR/voice.pid" ]', replacement)
        self.assertLess(replacement.index('kill -TERM -- "-$previous_pid"'), replacement.index('python3 - "$DEV_DIR"'))
        self.assertIn('if kill -0 "$previous_pid" 2>/dev/null; then', replacement)
        self.assertNotIn('(dev_dir / "voice.env.json").unlink()', _START_WORKER_SCRIPT)

    def test_run_voice_transfers_grant_over_stdin_and_redacts_captured_output(self):
        calls = []
        token = "issued-token.secret.signature"
        livekit_url = "wss://issued-livekit.invalid"

        def run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(
                args, 0, f"caller output {token}", f"caller warning {token}"
            )

        stack = DevStack(checkout=CHECKOUT, run=run)
        stack._entered = True
        stack._remote_dir = "/tmp/mentat-eval.test"
        result = stack.run_voice(
            ["evals/runner.py", "--room", "token-room"],
            token=token,
            livekit_url=livekit_url,
        )

        self.assertEqual(result.stdout, "caller output [REDACTED]")
        self.assertEqual(result.stderr, "caller warning [REDACTED]")
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(args[:2], ["ssh", "ultraviolet"])
        self.assertNotIn(token, " ".join(args))
        self.assertTrue(kwargs["capture_output"])
        self.assertTrue(kwargs["text"])
        self.assertTrue(kwargs["check"])
        self.assertIn('"command": ["evals/runner.py", "--room", "token-room"]', kwargs["input"])
        self.assertIn(f'"token": "{token}"', kwargs["input"])
        self.assertIn(f'"livekit_url": "{livekit_url}"', kwargs["input"])
        self.assertIn('"MENTAT_VOICE_TOKEN": token', kwargs["input"])
        self.assertIn('"LIVEKIT_URL": livekit_url', kwargs["input"])
        self.assertIn('"voice.env.json"', kwargs["input"])
        self.assertNotIn("ws://127.0.0.1:7880", kwargs["input"])

    def test_remote_and_run_voice_scrub_credential_forms_and_retain_cause(self):
        token = "issued-token.header.signature"
        secret_values = (
            "bare-assignment-secret",
            "double-quoted-secret",
            "single-quoted-secret",
            "json-secret",
            "single-json-secret",
            "colon-secret",
            "export-secret",
            "patchbay-header-secret",
            "authorization-header-secret",
            "api-header-secret",
        )
        diagnostics = "DISTINCTIVE-CAUSE\n" + "\n".join((
            f"API_KEY={secret_values[0]}",
            f'LIVEKIT_API_SECRET="{secret_values[1]}"',
            f"LIVEKIT_AUTH_TOKEN='{secret_values[2]}'",
            f'{{"LIVEKIT_API_SECRET": "{secret_values[3]}"}}',
            f"{{'LIVEKIT_API_KEY': '{secret_values[4]}'}}",
            f"VOICE_TOKEN: {secret_values[5]}",
            f"export DEVICE_PASSWORD={secret_values[6]}",
            f"X-Patchbay-Key: {secret_values[7]}",
            f"Authorization: Bearer {secret_values[8]}",
            f"x-api-key: {secret_values[9]}",
        )) + f"\n{token}"

        for path in ("_remote", "run_voice"):
            with self.subTest(path=path):
                failure = subprocess.CalledProcessError(
                    7,
                    ["ssh", "ultraviolet"],
                    output=diagnostics,
                    stderr=diagnostics,
                )

                def run(_args, **kwargs):
                    self.assertTrue(kwargs["check"])
                    raise failure

                stack = DevStack(checkout=CHECKOUT, run=run)
                stack._entered = True
                stack._remote_dir = "/tmp/mentat-eval.test"
                with self.assertRaises(subprocess.CalledProcessError) as raised:
                    if path == "_remote":
                        stack._remote("remote command", redact=(token,))
                    else:
                        stack.run_voice(
                            ["evals/runner.py"],
                            token=token,
                            livekit_url="wss://issued.invalid",
                        )

                error = raised.exception
                self.assertIn("DISTINCTIVE-CAUSE", str(error))
                self.assertIs(error.__cause__, failure)
                for secret in (*secret_values, token):
                    self.assertNotIn(secret, str(error))
                    self.assertNotIn(secret, error.stderr)
                    self.assertNotIn(secret, error.output)

    def test_voice_python_keeps_package_wrapper_and_checks_caller_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            underlying = root / "python3"
            invocation = root / "invocation"
            underlying.write_text(f"#!/bin/sh\nprintf '%s' \"$*\" > {shlex.quote(str(invocation))}\n")
            underlying.chmod(0o755)
            wrapper = root / "python-env"
            wrapper.symlink_to(underlying)
            proc_exe = root / "proc-exe"
            proc_exe.symlink_to(underlying)
            cmdline = root / "cmdline"
            cmdline.write_bytes(os.fsencode(wrapper) + b"\0agent.py\0start\0")

            capture = next(
                line for line in _SETUP_SCRIPT.splitlines()
                if "VOICE_PY" in line and "/proc/$VOICE_PID/" in line
            )
            capture = capture.replace('"/proc/$VOICE_PID/exe"', shlex.quote(str(proc_exe)))
            capture = capture.replace('"/proc/$VOICE_PID/cmdline"', shlex.quote(str(cmdline)))
            result = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", capture + '\nprintf "%s" "$VOICE_PY"'],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, str(wrapper))

            import_check = next(
                line for line in _SETUP_SCRIPT.splitlines()
                if line.startswith('"$VOICE_PY" -c ') and "aiohttp" in line
            )
            check = subprocess.run(
                ["bash", "-euo", "pipefail", "-c", f"VOICE_PY={shlex.quote(str(wrapper))}\n{import_check}"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(check.returncode, 0, check.stderr)
            self.assertIn("import aiohttp", invocation.read_text())

    def test_setpriv_is_resolved_outside_service_path_for_all_launches(self):
        def python_block(script, header):
            return script.split(header + chr(10), 1)[1].split(chr(10) + "PY" + chr(10), 1)[0]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev_dir = root / "stage"
            (dev_dir / "mentat").mkdir(parents=True)
            (dev_dir / "voice").mkdir()
            (dev_dir / "voice/evals").mkdir()
            shutil.copy2(
                Path(__file__).resolve().parents[1] / "evals/dev_stack.py",
                dev_dir / "voice/evals/dev_stack.py",
            )
            service_path = str(root / "service-bin")
            (dev_dir / "mentat.env.json").write_text(json.dumps({
                "PATH": service_path,
                "MENTAT_LISTEN": "127.0.0.1:8484",
            }))
            (dev_dir / "voice.env.json").write_text(json.dumps({
                "PATH": service_path,
                "LIVEKIT_API_SECRET": "private-env-secret",
            }))
            (dev_dir / "voice-python.path").write_text("/nix/store/python/bin/python3")

            setup_bin = root / "setup-bin"
            setup_bin.mkdir()
            setpriv = setup_bin / "setpriv"
            setpriv.write_text("fake setpriv executable")
            setpriv.chmod(0o755)
            resolved_setpriv = str(setpriv.resolve())

            setup = _SETUP_SCRIPT.replace(
                "__MCP_REWRITE_SOURCE__", _MCP_REWRITE_SOURCE
            ).replace("__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE)
            setup_header = 'python3 - "$DEV_DIR" "$DEV_PORT" "$NODE_BIN" "$VOICE_PY" <<\'PY\''
            setup_body = python_block(setup, setup_header)
            with patch.dict(os.environ, {"PATH": str(setup_bin)}):
                with patch("sys.argv", ["setup.py", str(dev_dir), "8485", "/nix/bin/node", "/nix/bin/python"]):
                    with patch("subprocess.Popen", return_value=SimpleNamespace(pid=1234)) as popen:
                        exec(setup_body, {})
            daemon_argv = popen.call_args.args[0]
            self.assertEqual(daemon_argv[0], resolved_setpriv)
            self.assertEqual(popen.call_args.kwargs["env"]["PATH"], service_path)
            self.assertEqual((dev_dir / "setpriv.path").read_text(), resolved_setpriv)
            self.assertEqual(stat.S_IMODE((dev_dir / "setpriv.path").stat().st_mode), 0o644)

            worker_body = python_block(
                _START_WORKER_SCRIPT,
                'python3 - "$DEV_DIR" "$DEV_PORT" "$HEALTH_PORT" "$ROOM" <<\'PY\'',
            )
            with patch("sys.argv", ["worker.py", str(dev_dir), "8485", "8486", "worker-room"]):
                with patch("subprocess.Popen") as popen:
                    exec(worker_body, {})
            worker_argv = popen.call_args.args[0]
            self.assertEqual(worker_argv[0], resolved_setpriv)
            self.assertEqual(popen.call_args.kwargs["env"]["PATH"], service_path)

            token = "issued-token.header.signature"
            caller_payload = json.dumps({
                "command": ["evals/runner.py", "--room", "caller-room"],
                "token": token,
                "livekit_url": "wss://issued-livekit.invalid",
            })
            result = SimpleNamespace(
                stdout="DISTINCTIVE-CAUSE private-env-secret " + token,
                stderr="private-env-secret " + token,
                returncode=0,
            )
            caller_stdout = io.StringIO()
            caller_stderr = io.StringIO()
            with patch("sys.argv", ["-c", "--", str(dev_dir), "8485", "8486"]):
                with patch("sys.stdin", io.StringIO(caller_payload)):
                    with patch("sys.stdout", caller_stdout), patch("sys.stderr", caller_stderr):
                        with patch("subprocess.run", return_value=result) as run:
                            with patch("sys.exit") as exit_process:
                                exec(_RUN_VOICE_SCRIPT, {})
            exit_process.assert_called_once_with(0)
            self.assertIn("DISTINCTIVE-CAUSE", caller_stdout.getvalue())
            self.assertNotIn("private-env-secret", caller_stdout.getvalue())
            self.assertNotIn(token, caller_stdout.getvalue())
            self.assertNotIn("private-env-secret", caller_stderr.getvalue())
            self.assertNotIn(token, caller_stderr.getvalue())
            caller_argv = run.call_args.args[0]
            self.assertEqual(caller_argv[0], resolved_setpriv)
            self.assertEqual(run.call_args.kwargs["env"]["PATH"], service_path)
            self.assertEqual(run.call_args.kwargs["env"]["MENTAT_VOICE_TOKEN"], token)
            self.assertEqual(
                run.call_args.kwargs["env"]["LIVEKIT_URL"],
                "wss://issued-livekit.invalid",
            )

    def test_setup_fails_if_setpriv_cannot_be_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dev_dir = root / "stage"
            (dev_dir / "mentat").mkdir(parents=True)
            setup = _SETUP_SCRIPT.replace(
                "__MCP_REWRITE_SOURCE__", _MCP_REWRITE_SOURCE
            ).replace("__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE)
            setup_header = 'python3 - "$DEV_DIR" "$DEV_PORT" "$NODE_BIN" "$VOICE_PY" <<\'PY\''
            setup_body = setup.split(setup_header + chr(10), 1)[1].split(chr(10) + "PY" + chr(10), 1)[0]
            with patch.dict(os.environ, {"PATH": str(root / "empty-bin")}):
                with patch("sys.argv", ["setup.py", str(dev_dir), "8485", "/nix/bin/node", "/nix/bin/python"]):
                    with patch("subprocess.Popen") as popen:
                        with self.assertRaisesRegex(RuntimeError, "setpriv executable is unavailable"):
                            exec(setup_body, {})
            popen.assert_not_called()

    def test_run_voice_requires_entered_stack_and_propagates_remote_failure(self):
        run = unittest.mock.Mock()
        stack = DevStack(checkout=CHECKOUT, run=run)

        grant = {"token": "issued-token.header.signature", "livekit_url": "wss://issued.invalid"}
        with self.assertRaisesRegex(RuntimeError, "dev stack is not running"):
            stack.run_voice(["voice/evals/caller.py"], **grant)
        run.assert_not_called()

        failure = subprocess.CalledProcessError(
            7, ["ssh"], output="caller output", stderr="caller failed"
        )
        run.side_effect = failure
        stack._entered = True
        stack._remote_dir = "/tmp/mentat-eval.test"
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            stack.run_voice(["voice/evals/caller.py"], **grant)
        self.assertIsInstance(raised.exception, subprocess.CalledProcessError)
        self.assertIs(raised.exception.__cause__, failure)
        self.assertIn("caller failed", str(raised.exception))

    def test_staging_directories_remain_writable_for_unprivileged_scp(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.permissions\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        stack = DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run)
        with patch("voice.evals.dev_stack.subprocess.Popen", return_value=unittest.mock.Mock(poll=lambda: None)):
            with stack:
                pass

        first_scp = next(i for i, (args, _) in enumerate(calls) if args[0] == "scp")
        before_scp = [args for args, _ in calls[:first_scp]]
        self.assertTrue(any(args[:3] == ["ssh", "ultraviolet", "mkdir"] for args in before_scp))
        self.assertFalse(any(args[:3] == ["ssh", "ultraviolet", "sudo"] for args in before_scp))

    def test_private_voice_credential_is_copied_before_stopping_production_worker(self):
        setup = _SETUP_SCRIPT.replace(
            "__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE
        )
        self.assertIn("MENTAT_VOICE_PRIVATE", setup)
        self.assertIn("stage_voice_private", setup)
        self.assertIn("os.chown", setup)
        self.assertIn("os.chmod", setup)
        self.assertNotIn("systemctl stop mentat-voice", setup)
        self.assertIn("systemctl stop mentat-voice", _START_WORKER_SCRIPT)

    def test_private_voice_credential_is_copied_and_owned_by_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "stage"
            staging.mkdir()
            source = Path(temporary) / "credential"
            source.write_text("private test context")
            namespace = {
                "Path": Path,
                "os": os,
                "pwd": SimpleNamespace(getpwnam=lambda _: SimpleNamespace(pw_uid=os.getuid())),
                "grp": SimpleNamespace(getgrnam=lambda _: SimpleNamespace(gr_gid=os.getgid())),
            }
            exec(_PRIVATE_CREDENTIAL_SOURCE, namespace)
            values = {"MENTAT_VOICE_PRIVATE": str(source)}

            namespace["stage_voice_private"](values, staging)

            staged = Path(values["MENTAT_VOICE_PRIVATE"])
            self.assertEqual(staged.read_text(), "private test context")
            self.assertNotEqual(staged, source)
            self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o400)
            self.assertEqual(staged.stat().st_uid, os.getuid())
            self.assertEqual(staged.stat().st_gid, os.getgid())

    def test_mcp_rewrite_changes_only_loopback_production_port(self):
        namespace = {"json": json, "urlsplit": urlsplit, "urlunsplit": urlunsplit}
        exec(_MCP_REWRITE_SOURCE, namespace)
        config = {
            "mcpServers": {
                "local": {"url": "http://127.0.0.1:8484/mcp?x=1"},
                "other-loopback-port": {"url": "http://localhost:9000/mcp"},
                "remote": {"url": "https://mcp.example.test/mcp"},
                "label": "http://127.0.0.1:8484 is text, not an endpoint",
            }
        }

        rewritten = json.loads(namespace["rewrite_mcp_config"](
            json.dumps(config), production_port=8484, dev_port=8485
        ))

        self.assertEqual(rewritten["mcpServers"]["local"]["url"], "http://127.0.0.1:8485/mcp?x=1")
        self.assertEqual(rewritten["mcpServers"]["other-loopback-port"]["url"], "http://localhost:9000/mcp")
        self.assertEqual(rewritten["mcpServers"]["remote"]["url"], "https://mcp.example.test/mcp")
        self.assertEqual(rewritten["mcpServers"]["label"], config["mcpServers"]["label"])

    def test_production_voice_restarts_when_setup_or_body_fails(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.failure\n", "")
            if args[:2] == ["nix", "build"]:
                result = subprocess.CompletedProcess(args, 1, "", "build failed")
                if kwargs.get("check"):
                    raise subprocess.CalledProcessError(result.returncode, args, result.stdout, result.stderr)
                return result
            return subprocess.CompletedProcess(args, 0, "", "")

        stack = DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run)
        with self.assertRaises(subprocess.CalledProcessError):
            with stack:
                self.fail("setup failure should prevent entering the context")

        scripts = [kwargs.get("input", "") for args, kwargs in calls if args[:2] == ["ssh", "ultraviolet"]]
        cleanup = "\n".join(script for script in scripts if script)
        self.assertIn("systemctl start mentat-voice", cleanup)

    def test_remote_setup_error_includes_captured_stderr_without_credentials(self):
        diagnostic = "DISTINCTIVE_SETUP_FAILURE"
        secrets = ("credential-value-double-quoted", "credential-value-single-quoted")

        def run(args, **kwargs):
            error = subprocess.CalledProcessError(
                1,
                args,
                output="",
                stderr=(
                    f"setup failed: {diagnostic}\\n"
                    f'LIVEKIT_API_SECRET="{secrets[0]}"\\n'
                    f"LIVEKIT_API_KEY='{secrets[1]}'"
                ),
            )
            raise error

        stack = DevStack(checkout=CHECKOUT, run=run)
        with self.assertRaisesRegex(subprocess.CalledProcessError, diagnostic) as caught:
            stack._remote("set -euo pipefail", "/tmp/mentat-eval.test")

        for secret in secrets:
            self.assertNotIn(secret, str(caught.exception))

    def test_interruption_during_setup_attempts_production_voice_restart(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, "", "")

        stack = DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run)
        with patch.object(DevStack, "_start", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                stack.__enter__()

        cleanup = "\n".join(
            kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
        )
        self.assertIn("systemctl start mentat-voice", cleanup)

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_production_voice_restarts_after_body_exception(self, popen):
        popen.return_value = unittest.mock.Mock(poll=lambda: None)
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.body\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with self.assertRaisesRegex(ValueError, "scenario failed"):
            with DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run):
                raise ValueError("scenario failed")

        cleanup = "\n".join(
            kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
        )
        self.assertIn("systemctl start mentat-voice", cleanup)
        self.assertIn("trap 'systemctl start mentat-voice' EXIT", cleanup)

    def test_staged_remote_runner_imports_from_the_uploaded_voice_files(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.imports\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("voice.evals.dev_stack.subprocess.Popen") as popen:
            popen.return_value = unittest.mock.Mock(poll=lambda: None)
            with DevStack(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            remote = Path(temporary)
            voice_root = remote / "voice"
            for args, _ in calls:
                if not args or args[0] != "scp":
                    continue
                remote_target = args[-1].split(":", 1)[1]
                target_is_directory = remote_target.endswith("/")
                target = remote / Path(remote_target).relative_to("/tmp/mentat-eval.imports")
                for source in args[1:-1]:
                    candidate = Path(source)
                    if not candidate.is_relative_to(CHECKOUT / "voice"):
                        continue
                    destination = target / candidate.name if target_is_directory else target
                    if candidate.is_dir():
                        shutil.copytree(candidate, destination)
                    elif candidate.is_file():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(candidate.read_bytes())

            runner = voice_root / "evals" / "runner.py"
            self.assertTrue(runner.is_file(), "runner.py must be in the staged voice tree")
            result = subprocess.run(
                [
                    os.fspath(sys.executable),
                    "-I",
                    "-c",
                    "import sys; sys.path.insert(0, sys.argv[1]); import evals.runner",
                    os.fspath(voice_root),
                ],
                cwd=remote,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"staged runner import failed\nstdout: {result.stdout}\nstderr: {result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
