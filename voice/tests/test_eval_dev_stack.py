"""Offline lifecycle tests for the isolated voice-eval development stack."""

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
from unittest.mock import patch

from voice.evals.dev_stack import (
    DevStack,
    _MCP_REWRITE_SOURCE,
    _PRIVATE_CREDENTIAL_SOURCE,
    _SETUP_SCRIPT,
    _START_WORKER_SCRIPT,
)


CHECKOUT = Path(__file__).resolve().parents[2]


class DevStackTest(unittest.TestCase):
    def test_requires_explicit_opt_in_before_any_remote_action(self):
        run = unittest.mock.Mock()
        stack = DevStack(checkout=CHECKOUT, run=run)

        with self.assertRaisesRegex(RuntimeError, "explicit opt-in"):
            stack.__enter__()

        run.assert_not_called()

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

        with DevStack(checkout=CHECKOUT, opt_in=True, run=run, local_port=0, dev_port=dev_port) as stack:
            self.assertRegex(stack.url, r"^http://127\.0\.0\.1:\d+$")
            stack.run_remote(["test", "-f", "voice/evals/phone.py"])
            stack.start_worker("android-test-room")

        build = next(call for call in calls if call[0][:2] == ["nix", "build"])
        self.assertIn(".#mentatd", build[0])
        transfers = [call[0] for call in calls if call[0][0] == "scp"]
        self.assertTrue(any("voice/evals/phone.py" in " ".join(args) for args in transfers))
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

        with DevStack(checkout=CHECKOUT, opt_in=True, run=run) as stack:
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

    def test_staging_directories_remain_writable_for_unprivileged_scp(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.permissions\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        stack = DevStack(checkout=CHECKOUT, opt_in=True, run=run)
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

        stack = DevStack(checkout=CHECKOUT, opt_in=True, run=run)
        with self.assertRaises(subprocess.CalledProcessError):
            with stack:
                self.fail("setup failure should prevent entering the context")

        scripts = [kwargs.get("input", "") for args, kwargs in calls if args[:2] == ["ssh", "ultraviolet"]]
        cleanup = "\n".join(script for script in scripts if script)
        self.assertIn("systemctl start mentat-voice", cleanup)

    def test_interruption_during_setup_attempts_production_voice_restart(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, "", "")

        stack = DevStack(checkout=CHECKOUT, opt_in=True, run=run)
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
            with DevStack(checkout=CHECKOUT, opt_in=True, run=run):
                raise ValueError("scenario failed")

        cleanup = "\n".join(
            kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
        )
        self.assertIn("systemctl start mentat-voice", cleanup)
        self.assertIn("trap 'systemctl start mentat-voice' EXIT", cleanup)


if __name__ == "__main__":
    unittest.main()
