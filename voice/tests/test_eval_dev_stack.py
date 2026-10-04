"""Offline lifecycle tests for the isolated voice-eval development stack."""

import io
import json
import os
import pwd
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import tarfile
import time
import unittest
from contextlib import contextmanager
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit, urlunsplit
from unittest.mock import patch

from voice.evals import runner
from voice.evals.scenarios import SCENARIOS
from voice.evals.dev_stack import (
    DevStack,
    _RunStack,
    _MCP_REWRITE_SOURCE,
    _BATCH_SETUP_SCRIPT,
    _BATCH_CLEANUP_SCRIPT,
    _REAP_DEAD_BATCH_SCRIPT,
    _PRIVATE_CREDENTIAL_SOURCE,
    _RETAIN_EVIDENCE_SCRIPT,
    _RUN_VOICE_SCRIPT,
    _RUN_SETUP_SCRIPT,
    _START_WORKER_SCRIPT,
    _STOP_RUN_SCRIPT,
    _REFRESH_RESTORE_GUARD_SCRIPT,
)


CHECKOUT = Path(__file__).resolve().parents[2]


def run_setup_environment(run_dir, model):
    setup = _RUN_SETUP_SCRIPT.replace("__MCP_REWRITE_SOURCE__", _MCP_REWRITE_SOURCE)
    header = 'python3 - "$RUN_DIR" "$DEV_PORT" "$HEALTH_PORT" "$VOICE_MODEL" <<\'PY\''
    body = setup.split(header + "\n", 1)[1].split("\nPY\n", 1)[0]
    with patch("sys.argv", ["setup.py", os.fspath(run_dir), "49151", "49152", model]):
        with patch("subprocess.Popen", return_value=SimpleNamespace(pid=12345)) as popen:
            exec(body, {})
    return popen.call_args.kwargs["env"]


@contextmanager
def isolated_run(**kwargs):
    command_runner = kwargs.pop("run", subprocess.run)
    run_settings = {
        key: kwargs.pop(key)
        for key in ("dev_port", "health_port", "local_port")
        if key in kwargs
    }
    probes = iter((
        {"dev_port": 49151, "health_port": 49152},
        {"dev_port": 49153, "health_port": 49154},
        {"dev_port": 49155, "health_port": 49156},
        {"dev_port": 49157, "health_port": 49158},
    ))

    def run(args, **options):
        result = command_runner(args, **options)
        if args[:2] == ["ssh", "ultraviolet"] and len(args) > 2 and args[2] == "mktemp":
            if re.fullmatch(r"/tmp/mentat-eval-batch\.[A-Za-z0-9]+", str(getattr(result, "stdout", "")).strip()):
                return result
            return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval-batch.test\n", "")
        if "MENTAT_EVAL_PORT_PROBE" in options.get("input", ""):
            if isinstance(getattr(result, "stdout", None), str) and result.stdout:
                return result
            return subprocess.CompletedProcess(args, 0, json.dumps(next(probes)), "")
        if "MENTAT_VOICE_GRANT" in options.get("input", "") and not result.stdout:
            return subprocess.CompletedProcess(args, 0, '{"turns":[]}\n', result.stderr)
        return result

    with DevStack(**{**kwargs, "run": run}) as batch:
        stack = batch.run("test-run")
        stack.dev_port = run_settings.get("dev_port")
        stack.health_port = run_settings.get("health_port")
        stack._requested_local_port = run_settings.get("local_port")
        with stack:
            yield stack


class DevStackTest(unittest.TestCase):
    def test_batch_shutdown_closes_launch_gate_before_restoring(self):
        batch = DevStack(checkout=CHECKOUT, opt_in=True, run=lambda *a, **k: subprocess.CompletedProcess(a, 0, "", ""))
        batch._entered = True
        batch._remote_dir = "/tmp/mentat-eval-batch.synthetic"
        batch._cleanup()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            batch.run("late-run")
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            batch._begin_launch()

    def test_cleanup_waits_for_in_flight_launch_before_restoration(self):
        from threading import Thread

        batch = DevStack(checkout=CHECKOUT, opt_in=True)
        batch._entered = True
        batch._remote_dir = "/tmp/mentat-eval-batch.synthetic"
        batch._begin_launch()
        restored = []
        with patch.object(batch, "_remote", side_effect=lambda *args: restored.append(args)):
            cleanup = Thread(target=batch._cleanup)
            cleanup.start()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                with batch._condition:
                    if batch._shutting_down:
                        break
                time.sleep(0.001)
            self.assertTrue(batch._shutting_down)
            self.assertTrue(cleanup.is_alive())
            self.assertEqual(restored, [])
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                batch.run("late-run")
            batch._end_launch()
            cleanup.join(timeout=2)
        self.assertFalse(cleanup.is_alive())
        self.assertEqual(len(restored), 1)

    def test_signal_during_shutdown_does_not_abort_cleanup(self):
        import signal

        batch = DevStack(checkout=CHECKOUT, opt_in=True)
        batch._shutting_down = True
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                interrupted = False
                try:
                    batch._interrupted(signum, None)
                except KeyboardInterrupt:
                    interrupted = True
                self.assertFalse(interrupted)

    def test_cleanup_suppresses_and_restores_active_scheduler_signal_handler(self):
        import signal

        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum):
                original = signal.getsignal(signum)
                received = []

                def scheduler_handler(received_signal, _frame):
                    received.append(received_signal)
                    raise KeyboardInterrupt("scheduler abort")

                signal.signal(signum, scheduler_handler)
                batch = DevStack(checkout=CHECKOUT, opt_in=True)
                batch._entered = True
                batch._remote_dir = "/tmp/mentat-eval-batch.synthetic"
                restored = []

                def remote(_script, path):
                    os.kill(os.getpid(), signum)
                    restored.append(path)
                    return subprocess.CompletedProcess(["ssh"], 0, "", "")

                try:
                    with patch.object(batch, "_remote", side_effect=remote):
                        aborted = False
                        try:
                            batch._cleanup()
                        except KeyboardInterrupt:
                            aborted = True
                    active_after_cleanup = signal.getsignal(signum)
                finally:
                    signal.signal(signum, original)
                self.assertFalse(aborted)
                self.assertEqual(received, [])
                self.assertIs(active_after_cleanup, scheduler_handler)
                self.assertEqual(restored, ["/tmp/mentat-eval-batch.synthetic"])

    def test_exit_preserves_scheduler_handler_installed_after_batch_enter(self):
        import signal

        original = signal.getsignal(signal.SIGTERM)
        batch = DevStack(checkout=CHECKOUT, opt_in=True)

        def scheduler_handler(_signum, _frame):
            return

        try:
            batch._install_signal_handlers()
            signal.signal(signal.SIGTERM, scheduler_handler)
            batch._restore_signal_handlers()
            self.assertIs(signal.getsignal(signal.SIGTERM), scheduler_handler)
        finally:
            signal.signal(signal.SIGTERM, original)

    def test_real_cleanup_shell_acquires_lock_and_restores_voice(self):
        cleanup = _BATCH_CLEANUP_SCRIPT
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch_dir = root / "batch"
            run_dir = batch_dir / "runs" / "one"
            (run_dir).mkdir(parents=True)
            (batch_dir / "shared").mkdir()
            (batch_dir / "restore-action").write_text("start\n")
            (batch_dir / "restore-unit").write_text("mentat-eval-restore-test\n")
            (run_dir / "launch.lock").touch()
            events = root / "systemctl.log"
            bindir = root / "bin"
            bindir.mkdir()
            systemctl = bindir / "systemctl"
            systemctl.write_text("#!/bin/sh\nprintf '%s\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n")
            systemctl.chmod(0o700)
            result = subprocess.run(
                ["bash", "-c", cleanup, "cleanup", os.fspath(batch_dir)],
                env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "SYSTEMCTL_LOG": os.fspath(events)},
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("start mentat-voice", events.read_text())

    def test_recovery_preserves_untrusted_cleanup_and_process_records(self):
        import signal
        import socket

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch_dir = root / "untrusted-batch"
            runs_dir = batch_dir / "runs"
            run_dir = runs_dir / "one"
            run_dir.mkdir(parents=True)
            batch_dir.chmod(0o711)
            runs_dir.chmod(0o711)
            run_dir.chmod(0o711)
            marker = root / "executed"
            unrelated = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
            )

            def stop_unrelated():
                try:
                    os.killpg(unrelated.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                unrelated.wait(timeout=2)

            self.addCleanup(stop_unrelated)
            dead_owner = subprocess.Popen([sys.executable, "-c", "pass"])
            dead_owner.wait()
            owner = {"host": socket.gethostname(), "pid": dead_owner.pid}
            (batch_dir / "shared").mkdir()
            (batch_dir / "owner.json").write_text(json.dumps(owner) + "\n")
            (batch_dir / "owner.json").chmod(0o600)
            (batch_dir / "restore-action").write_text("start\n")
            (batch_dir / "restore-action").chmod(0o600)
            (batch_dir / "restore-unit").write_text("mentat-eval-restore-fake\n")
            (batch_dir / "restore-unit").chmod(0o600)
            (run_dir / "launch.lock").touch(mode=0o600)
            (batch_dir / "cleanup.sh").write_text(
                f"kill -TERM {unrelated.pid}\ntouch {shlex.quote(os.fspath(marker))}\n"
            )
            (batch_dir / "cleanup.sh").chmod(0o700)
            (run_dir / "agent.pid").write_text(f"{unrelated.pid}\n")
            (run_dir / "agent.pid").chmod(0o600)
            response = [{"path": "/tmp/mentat-eval-batch.untrusted", "owner": json.dumps(owner)}]
            batch = DevStack(
                checkout=CHECKOUT,
                opt_in=True,
                run=lambda args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(response), ""),
            )

            def execute_remote(script, path, owner_host, owner_pid):
                return subprocess.run(
                    ["bash", "-c", script, "reaper", os.fspath(batch_dir), owner_host, owner_pid],
                    capture_output=True, text=True, check=True,
                )

            with patch.object(batch, "_remote", side_effect=execute_remote):
                batch._reap_dead_batches()
            self.assertFalse(marker.exists())
            self.assertIsNone(unrelated.poll())

    def test_owned_process_groups_stop_before_restore_after_launch_capture_abort(self):
        import signal
        from threading import Event, Thread

        cleanup = _BATCH_CLEANUP_SCRIPT
        for phase in ("launch", "capture"):
            for outcome in ("error", signal.SIGINT, signal.SIGTERM):
                with self.subTest(phase=phase, outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    batch_dir = root / "batch"
                    run_dir = batch_dir / "runs" / "one"
                    run_dir.mkdir(parents=True)
                    (batch_dir / "shared").mkdir()
                    (batch_dir / "restore-action").write_text("start\n")
                    (batch_dir / "restore-unit").write_text("mentat-eval-restore-test\n")
                    (batch_dir / "cleanup.sh").write_text(cleanup)
                    (run_dir / "launch.lock").touch()
                    bindir = root / "bin"
                    bindir.mkdir()
                    events = root / "events"
                    systemctl = bindir / "systemctl"
                    systemctl.write_text(
                        "#!/usr/bin/env bash\n"
                        "if [ \"$*\" = 'start mentat-voice' ]; then\n"
                        "  for record in \"$CHECK_RUN_DIR\"/*.pid; do\n"
                        "    [ -f \"$record\" ] || continue\n"
                        "    if kill -0 -- \"-$(cat \"$record\")\" 2>/dev/null; then exit 1; fi\n"
                        "  done\n"
                        "fi\n"
                        "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
                    )
                    systemctl.chmod(0o700)
                    leaders = []
                    phones = []
                    phase_entries = []
                    batch = DevStack(checkout=CHECKOUT, opt_in=True)
                    batch._entered = True
                    batch._remote_dir = os.fspath(batch_dir)
                    stack = batch.run("one")

                    def stop_test_group(process):
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=2)

                    def interrupt(active_phase):
                        phase_entries.append(active_phase)
                        self.assertEqual(active_phase, phase)
                        if active_phase == "launch":
                            self.assertEqual(batch._launches_in_progress, 1)
                            self.assertFalse(stack._entered)
                        else:
                            self.assertEqual(batch._launches_in_progress, 0)
                            self.assertTrue(stack._entered)
                        if outcome == "error":
                            raise RuntimeError(f"{active_phase} failed")
                        os.kill(os.getpid(), outcome)
                        self.fail("signal did not interrupt the active run")

                    def start_run():
                        stack._remote_dir = os.fspath(run_dir)
                        for name, with_phone in (("agent", False), ("voice", True), ("caller", True)):
                            child_pid_path = root / f"{name}.child"
                            code = (
                                "import pathlib,signal,subprocess,sys,time\n"
                                "child=None\n"
                                "if sys.argv[1] != '-':\n"
                                " child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
                                " pathlib.Path(sys.argv[1]).write_text(str(child.pid))\n"
                                "def stop(*args):\n"
                                " if child is not None: child.terminate(); child.wait()\n"
                                " raise SystemExit(0)\n"
                                "signal.signal(signal.SIGTERM,stop)\n"
                                "time.sleep(60)\n"
                            )
                            child_path = os.fspath(child_pid_path) if with_phone else "-"
                            leader = subprocess.Popen(
                                [sys.executable, "-c", code, child_path],
                                start_new_session=True,
                            )
                            leaders.append(leader)
                            self.addCleanup(stop_test_group, leader)
                            if with_phone:
                                deadline = time.monotonic() + 3
                                child_pid_text = ""
                                while time.monotonic() < deadline:
                                    if child_pid_path.exists():
                                        child_pid_text = child_pid_path.read_text().strip()
                                        if child_pid_text:
                                            break
                                    time.sleep(0.001)
                                self.assertTrue(child_pid_path.exists())
                                self.assertTrue(child_pid_text)
                                phones.append(int(child_pid_text))
                            (run_dir / f"{name}.pid").write_text(f"{leader.pid}\n")
                        if phase == "launch":
                            interrupt("launch")

                    def remote(script, *args, **_kwargs):
                        if "MENTAT_VOICE_GRANT" in script:
                            interrupt("capture")
                        return subprocess.run(
                            ["bash", "-c", script, "cleanup", *args],
                            env={
                                **os.environ,
                                "PATH": f"{bindir}:{os.environ['PATH']}",
                                "SYSTEMCTL_LOG": os.fspath(events),
                                "CHECK_RUN_DIR": os.fspath(run_dir),
                            },
                            capture_output=True, text=True, check=True,
                        )

                    poll_stop = Event()

                    def reap_leaders():
                        while not poll_stop.is_set():
                            for leader in tuple(leaders):
                                leader.poll()
                            time.sleep(0.001)

                    reaper = Thread(target=reap_leaders)
                    reaper.start()
                    batch._install_signal_handlers()
                    try:
                        with (
                            patch.object(stack, "_start", side_effect=start_run),
                            patch.object(stack, "_remote", side_effect=remote),
                            patch.object(batch, "_remote", side_effect=remote),
                        ):
                            expected_error = RuntimeError if outcome == "error" else KeyboardInterrupt
                            with self.assertRaises(expected_error):
                                try:
                                    with stack:
                                        stack.run_voice(
                                            ["-m", "evals.runner"], token="offline-token",
                                            livekit_url="ws://127.0.0.1:7880",
                                        )
                                finally:
                                    batch.__exit__(*sys.exc_info())
                    finally:
                        batch._restore_signal_handlers()
                        poll_stop.set()
                        reaper.join(timeout=2)
                    self.assertEqual(phase_entries, [phase])
                    self.assertEqual(batch._launches_in_progress, 0)
                    self.assertFalse(batch._entered)
                    for leader in leaders:
                        leader.wait(timeout=2)
                        self.assertIsNotNone(leader.returncode)
                    for pid in phones:
                        with self.assertRaises(ProcessLookupError):
                            os.kill(pid, 0)
                    self.assertEqual(events.read_text().splitlines()[-1], "start mentat-voice")
                    with self.assertRaisesRegex(RuntimeError, "shutting down"):
                        batch.run("late")

    def test_owner_death_requires_local_host_and_positive_absence(self):
        owner = {"host": "runner-host", "pid": 321}
        with patch("voice.evals.dev_stack.socket.gethostname", return_value="runner-host"), patch(
            "voice.evals.dev_stack.os.kill", side_effect=ProcessLookupError
        ):
            self.assertTrue(DevStack._owner_process_is_dead(owner))
        with patch("voice.evals.dev_stack.socket.gethostname", return_value="other-host"):
            self.assertFalse(DevStack._owner_process_is_dead(owner))
        with patch("voice.evals.dev_stack.socket.gethostname", return_value="runner-host"), patch(
            "voice.evals.dev_stack.os.kill", side_effect=PermissionError
        ):
            self.assertFalse(DevStack._owner_process_is_dead(owner))
        self.assertFalse(DevStack._owner_process_is_dead({"host": "runner-host", "pid": "bad"}))

    def test_dead_owner_reaping_kills_only_dead_owned_candidate_processes(self):
        import signal
        import socket
        from threading import Event, Thread

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindir = root / "bin"
            bindir.mkdir()
            events = root / "events"
            systemctl = bindir / "systemctl"
            systemctl.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$SYSTEMCTL_LOG\"\n"
            )
            systemctl.chmod(0o700)
            dead_batch = root / "dead-batch"
            dead_run = dead_batch / "runs" / "one"
            (dead_run).mkdir(parents=True)
            (dead_batch / "shared").mkdir()
            (dead_batch / "restore-action").write_text("start\n")
            (dead_batch / "restore-unit").write_text("mentat-eval-restore-test\n")
            (dead_run / "launch.lock").touch()

            def sleeper():
                return subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    start_new_session=True,
                )

            dead_owner = subprocess.Popen([sys.executable, "-c", "pass"])
            dead_owner.wait()
            live_owner = sleeper()
            dead_candidate = sleeper()
            live_candidate = sleeper()
            unrelated = sleeper()
            (dead_run / "agent.pid").write_text(f"{dead_candidate.pid}\n")
            response = [
                {"path": "/tmp/mentat-eval-batch.dead", "owner": json.dumps({"host": socket.gethostname(), "pid": dead_owner.pid})},
                {"path": "/tmp/mentat-eval-batch.live", "owner": json.dumps({"host": socket.gethostname(), "pid": live_owner.pid})},
                {"path": "/tmp/mentat-eval-batch.foreign", "owner": json.dumps({"host": "foreign-host", "pid": dead_owner.pid})},
                {"path": "/tmp/mentat-eval-batch.malformed", "owner": "not-json"},
            ]
            batch = DevStack(
                checkout=CHECKOUT,
                opt_in=True,
                run=lambda args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(response), ""),
            )

            def cleanup_dead(script, _remote_path, host, pid):
                self.assertEqual(host, socket.gethostname())
                self.assertEqual(int(pid), dead_owner.pid)
                self.assertTrue(script.startswith(_REAP_DEAD_BATCH_SCRIPT))
                return subprocess.run(
                    ["bash", "-c", _BATCH_CLEANUP_SCRIPT, "cleanup", os.fspath(dead_batch)],
                    env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "SYSTEMCTL_LOG": os.fspath(events)},
                    capture_output=True, text=True, check=True,
                )

            def stop_test_group(child):
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait(timeout=2)

            for process in (live_owner, dead_candidate, live_candidate, unrelated):
                self.addCleanup(stop_test_group, process)
            poll_complete = Event()

            def reap_dead_candidate():
                while not poll_complete.is_set():
                    dead_candidate.poll()
                    if dead_candidate.returncode is not None:
                        return
                    time.sleep(0.001)

            process_reaper = Thread(target=reap_dead_candidate)
            process_reaper.start()
            try:
                with patch.object(batch, "_remote", side_effect=cleanup_dead):
                    batch._reap_dead_batches()
            finally:
                poll_complete.set()
                process_reaper.join(timeout=2)
            dead_candidate.wait(timeout=2)
            self.assertIsNone(live_candidate.poll())
            self.assertIsNone(unrelated.poll())
            self.assertEqual(events.read_text().splitlines()[-1], "start mentat-voice")
            for process in (live_owner, live_candidate, unrelated):
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=2)

    def test_candidate_shutdown_records_and_stops_caller_process_group(self):
        self.assertIn('"$RUN_DIR/caller.pid"', _STOP_RUN_SCRIPT)
        self.assertIn("caller_pid_file = DEV_DIR / \"caller.pid\"", _RUN_VOICE_SCRIPT)
        self.assertIn("flock(launch_lock, fcntl.LOCK_SH)", _RUN_VOICE_SCRIPT)
        cleanup = _BATCH_CLEANUP_SCRIPT
        self.assertIn('"$BATCH_DIR"/runs/*/caller.pid', cleanup)
        self.assertIn('touch "$run_dir/shutdown"', cleanup)
        self.assertLess(cleanup.index('caller.pid'), cleanup.index('systemctl start mentat-voice'))

    def test_recovery_reaps_only_a_positively_dead_owned_batch(self):
        owners = [
            {"path": "/tmp/mentat-eval-batch.dead1", "owner": '{"host":"local","pid":101}'},
            {"path": "/tmp/mentat-eval-batch.live1", "owner": '{"host":"local","pid":102}'},
            {"path": "/tmp/mentat-eval-batch.foreign", "owner": '{"host":"remote","pid":103}'},
            {"path": "/tmp/mentat-eval-batch.bad", "owner": "not-json"},
            {"path": "/tmp/not-an-eval-batch", "owner": '{"host":"local","pid":104}'},
        ]
        batch = DevStack(
            checkout=CHECKOUT,
            opt_in=True,
            run=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps(owners), ""),
        )
        with patch.object(batch, "_owner_process_is_dead", side_effect=[True, False, False]), patch.object(
            batch, "_remote"
        ) as remote:
            batch._reap_dead_batches()
        remote.assert_called_once()
        self.assertEqual(remote.call_args.args[1], "/tmp/mentat-eval-batch.dead1")

    def test_batch_runs_share_staging_but_keep_independent_run_lifecycles(self):
        calls = []
        events = []
        probe_count = 0

        def run(args, **kwargs):
            nonlocal probe_count
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval-batch.test\n", "")
            if "MENTAT_EVAL_PORT_PROBE" in kwargs.get("input", ""):
                probe_count += 1
                return subprocess.CompletedProcess(
                    args, 0,
                    json.dumps({"dev_port": 49151 + probe_count * 2, "health_port": 49152 + probe_count * 2}),
                    "",
                )
            if '"MENTAT_STATE_PATH": str(run_dir' in kwargs.get("input", ""):
                events.append(("setup", args[-4], args[-3:-1]))
            if args[:2] == ["ssh", "ultraviolet"] and "--room" in kwargs.get("input", ""):
                events.append(("worker", args[-4], args[-3:-1]))
            if "MENTAT_VOICE_GRANT" in kwargs.get("input", ""):
                run_id = "same-scenario-2" if '"token": "second-token"' in kwargs["input"] else "same-scenario-1"
                events.append(("capture", run_id))
                return subprocess.CompletedProcess(
                    args, 0, json.dumps({"turns": [{"receipt": f"{run_id}-receipt"}]}), ""
                )
            if "RUN_DIR=$1" in kwargs.get("input", "") and "kill -TERM" in kwargs.get("input", ""):
                run_id = args[-1].rsplit("/", 1)[-1]
                events.append(("stop", run_id))
            if "exec cat voice/evals/phone.jsonl" in kwargs.get("input", ""):
                run_id = "same-scenario-2" if "same-scenario-2" in kwargs["input"] else "same-scenario-1"
                events.append(("evidence", run_id))
                return subprocess.CompletedProcess(
                    args, 0,
                    json.dumps({"event": "result", "run_id": run_id, "command_id": "same-command-id"}),
                    "",
                )
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("voice.evals.dev_stack.subprocess.Popen") as popen:
            popen.return_value = unittest.mock.Mock(poll=lambda: None)
            with DevStack(checkout=CHECKOUT, opt_in=True, run=run) as batch:
                with batch.run("same-scenario-1") as first:
                    with batch.run("same-scenario-2") as second:
                        self.assertNotEqual(first.base_url, second.base_url)
                        self.assertNotEqual(first.dev_port, second.dev_port)
                        self.assertNotEqual(first.health_port, second.health_port)
                        second.start_worker("second-room")
                        second_receipt = second.run_voice(
                            ["caller.py"], token="second-token", livekit_url="wss://second.invalid"
                        )
                        second_phone = second.run_remote(["cat", "voice/evals/phone.jsonl"])
                    self.assertTrue(first._entered)
                    self.assertEqual(json.loads(second_receipt.stdout)["turns"][0]["receipt"], "same-scenario-2-receipt")
                    self.assertEqual(json.loads(second_phone.stdout)["run_id"], "same-scenario-2")
                    self.assertFalse(any(
                        'bash "$BATCH_DIR/cleanup.sh"' in kwargs.get("input", "")
                        for args, kwargs in calls
                        if args[:2] == ["ssh", "ultraviolet"]
                    ))
                    first.start_worker("first-room")
                    first_receipt = first.run_voice(
                        ["caller.py"], token="first-token", livekit_url="wss://first.invalid"
                    )
                    first_phone = first.run_remote(["cat", "voice/evals/phone.jsonl"])
                    self.assertEqual(json.loads(first_receipt.stdout)["turns"][0]["receipt"], "same-scenario-1-receipt")
                    self.assertEqual(json.loads(first_phone.stdout)["run_id"], "same-scenario-1")

        builds = [args for args, _ in calls if args[:2] == ["nix", "build"]]
        transfers = [args for args, _ in calls if args and args[0] == "scp" and args[1:2] != ["-p"]]
        self.assertEqual(len(builds), 1)
        self.assertEqual(len(transfers), 4)  # package, voice modules, evals, and assets once each
        remote_scripts = [
            kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and kwargs.get("input")
        ]
        stop_run_indices = [
            index for index, script in enumerate(remote_scripts)
            if "RUN_DIR=$1" in script and 'kill -TERM -- "-$pid"' in script
        ]
        batch_cleanup_indices = [
            index for index, script in enumerate(remote_scripts)
            if 'bash "$BATCH_DIR/cleanup.sh"' in script
        ]
        self.assertEqual(len(stop_run_indices), 2)
        self.assertEqual(len(batch_cleanup_indices), 1)
        self.assertLess(max(stop_run_indices), batch_cleanup_indices[0])
        batch_cleanup = _BATCH_CLEANUP_SCRIPT
        self.assertLess(
            batch_cleanup.index('for pid_file in "$BATCH_DIR"/runs/*/agent.pid'),
            batch_cleanup.index("systemctl start mentat-voice"),
        )
        setup_invocations = [event for event in events if event[0] == "setup"]
        setup_counts = {}
        setup_ports = {}
        for _kind, run_dir, ports in setup_invocations:
            setup_counts[run_dir] = setup_counts.get(run_dir, 0) + 1
            setup_ports[run_dir] = tuple(ports)
        self.assertEqual(set(setup_counts.values()), {1})
        worker_events = [event for event in events if event[0] == "worker"]
        worker_ports = {
            run_dir: tuple(ports)
            for _kind, run_dir, ports in worker_events
        }
        self.assertEqual(worker_ports, setup_ports)
        for run_id in ("same-scenario-1", "same-scenario-2"):
            stop_index = events.index(("stop", run_id))
            evidence_index = events.index(("evidence", run_id))
            self.assertLess(stop_index, evidence_index)
        tunnel_ports = {
            int(call.args[0][-2].rsplit(":", 1)[1])
            for call in popen.call_args_list
        }
        self.assertEqual(tunnel_ports, {int(ports[0]) for ports in setup_ports.values()})
        self.assertEqual(popen.call_count, 2)

    def test_concurrent_run_stacks_keep_ports_and_remote_roots_independent(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock

        calls = []
        guard = Lock()
        probe_count = 0

        def run(args, **kwargs):
            nonlocal probe_count
            with guard:
                calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval-batch.parallel\n", "")
            if "MENTAT_EVAL_PORT_PROBE" in kwargs.get("input", ""):
                with guard:
                    probe_count += 1
                    number = probe_count
                return subprocess.CompletedProcess(
                    args, 0,
                    json.dumps({"dev_port": 49150 + number * 2, "health_port": 49151 + number * 2}),
                    "",
                )
            if "MENTAT_VOICE_GRANT" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(args, 0, '{"turns":[]}\n', "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("voice.evals.dev_stack.subprocess.Popen") as popen:
            popen.side_effect = lambda *_args, **_kwargs: unittest.mock.Mock(poll=lambda: None)
            with DevStack(checkout=CHECKOUT, opt_in=True, run=run) as batch:
                def launch(run_id):
                    with batch.run(run_id) as stack:
                        stack.start_worker(f"{run_id}-room")
                        stack.run_voice(["caller.py"], token=f"{run_id}-token", livekit_url="wss://lk.invalid")
                        return stack.dev_port, stack.health_port, stack._remote_dir, stack.base_url

                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(launch, ("parallel-one", "parallel-two")))

        self.assertEqual(len({(dev_port, health_port) for dev_port, health_port, _root, _url in results}), 2)
        self.assertEqual(len({root for _dev_port, _health_port, root, _url in results}), 2)
        self.assertEqual(len({url for _dev_port, _health_port, _root, url in results}), 2)
        self.assertEqual(popen.call_count, 2)

    def test_run_voice_stops_its_daemon_and_worker_before_evidence_reads(self):
        events = []
        stack = _RunStack(
            checkout=CHECKOUT, run=subprocess.run, batch=SimpleNamespace(), run_id="producer-order"
        )
        stack._entered = True
        stack._remote_dir = "/tmp/mentat-eval-batch.synthetic/runs/producer-order"
        stack.dev_port = 49151
        stack.health_port = 49152

        def remote(script, *_args, **_kwargs):
            if "MENTAT_VOICE_GRANT" in script:
                events.append("capture")
                return subprocess.CompletedProcess(["ssh"], 0, '{"turns":[]}\n', "")
            if "RUN_DIR=$1" in script and "agent.pid" in script and "voice.pid" in script:
                events.append(("stop", script))
                return subprocess.CompletedProcess(["ssh"], 0, "", "")
            raise AssertionError(f"unexpected remote command: {script[:80]}")

        def run(args, **kwargs):
            events.append("evidence-read")
            return subprocess.CompletedProcess(args, 0, "{}\n", "")

        with patch.object(stack, "_remote", side_effect=remote), patch.object(stack, "_run", side_effect=run):
            stack.run_voice(["caller.py"], token="private-token", livekit_url="wss://lk.invalid")
            self.assertEqual([event if isinstance(event, str) else event[0] for event in events], ["capture", "stop"])
            stop_script = events[-1][1]
            self.assertIn('kill -TERM -- "-$pid"', stop_script)
            self.assertIn('"$RUN_DIR/agent.pid" "$RUN_DIR/voice.pid"', stop_script)
            stack.run_remote(["cat", "voice/evals/phone.jsonl"])
        stop_index = next(index for index, event in enumerate(events) if isinstance(event, tuple) and event[0] == "stop")
        self.assertLess(stop_index, events.index("evidence-read"))

    def test_concurrent_duplicate_run_ids_are_rejected_atomically(self):
        from concurrent.futures import ThreadPoolExecutor

        def run(args, **kwargs):
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval-batch.ids\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        def slow_stack(**kwargs):
            time.sleep(0.05)
            return SimpleNamespace(run_id=kwargs["run_id"], _remote_dir=None, _tunnel=None)

        with DevStack(checkout=CHECKOUT, opt_in=True, run=run) as batch:
            with patch("voice.evals.dev_stack._RunStack", side_effect=slow_stack):
                def register():
                    try:
                        return batch.run("same-id")
                    except ValueError:
                        return None

                with ThreadPoolExecutor(max_workers=2) as executor:
                    outcomes = list(executor.map(lambda _index: register(), range(2)))
        self.assertEqual(sum(outcome is not None for outcome in outcomes), 1)
        self.assertEqual(sum(outcome is None for outcome in outcomes), 1)

    def test_run_ancestors_are_searchable_but_logs_and_state_remain_private(self):
        batch_permissions = (
            'install -d -m 700 "$BATCH_DIR/runs"',
            'chmod 711 "$BATCH_DIR"',
            'chmod 711 "$BATCH_DIR/runs"',
        )
        run_permissions = (
            'umask 077',
            'mkdir -m 700 "$RUN_DIR"',
            'mkdir -m 700 -p "$RUN_DIR/home/mentat" "$RUN_DIR/home/voice/cache" "$RUN_DIR/records" "$RUN_DIR/memory" "$RUN_DIR/voice/evals"',
            'chmod 711 "$RUN_DIR"',
            'chmod 711 "$RUN_DIR/home"',
        )
        for statement in batch_permissions:
            self.assertIn(statement, _BATCH_SETUP_SCRIPT)
        for statement in run_permissions:
            self.assertIn(statement, _RUN_SETUP_SCRIPT)
        self.assertIn('chmod 400 "$RUN_DIR/voice-private"', _RUN_SETUP_SCRIPT)
        self.assertIn('chmod 400 "$RUN_DIR/voice-gateway-key"', _RUN_SETUP_SCRIPT)
        self.assertIn('chmod 640 "$RUN_DIR/voice.env.json"', _RUN_SETUP_SCRIPT)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch"
            batch.mkdir(mode=0o700)
            run_dir = batch / "runs/run-one"
            commands = [*batch_permissions, *run_permissions]
            setup = "set -euo pipefail\n" + "\n".join(commands) + "\n"
            setup += f'python3 -c "from pathlib import Path; Path({str(run_dir / "agent.log")!r}).touch()"\n'
            result = subprocess.run(
                ["bash", "-c", setup],
                env={**os.environ, "BATCH_DIR": os.fspath(batch), "RUN_DIR": os.fspath(run_dir)},
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(stat.S_IMODE(batch.stat().st_mode), 0o711)
            self.assertEqual(stat.S_IMODE((batch / "runs").stat().st_mode), 0o711)
            self.assertEqual(stat.S_IMODE(run_dir.stat().st_mode), 0o711)
            self.assertEqual(stat.S_IMODE((run_dir / "home").stat().st_mode), 0o711)
            for relative in ("home/mentat", "home/voice/cache", "records", "memory", "voice/evals"):
                self.assertEqual(stat.S_IMODE((run_dir / relative).stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((run_dir / "agent.log").stat().st_mode), 0o600)

    def test_stop_run_script_waits_for_group_after_kill_and_rejects_live_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "agent.pid").write_text("12345\n")
            bash_env = root / "bash-env"
            bash_env.write_text(r'''PHASE=term
PROBES=0
kill() {
  printf '%s\n' "$*" >> "$KILL_LOG"
  case "$1" in
    -TERM) PHASE=term; PROBES=0; return 0 ;;
    -KILL) PHASE=killed; PROBES=0; return 0 ;;
    -0)
      if [ "$2" != -- ] || [ "$3" != -12345 ]; then return 1; fi
      PROBES=$((PROBES + 1))
      if [ "$PHASE" = killed ] && [ "$MODE" = drain ] && [ "$PROBES" -ge 3 ]; then return 1; fi
      return 0 ;;
    *) return 1 ;;
  esac
}
sleep() { printf 'sleep\n' >> "$KILL_LOG"; }
''')
            for mode in ("drain", "live"):
                with self.subTest(mode=mode):
                    log = root / (mode + ".log")
                    result = subprocess.run(
                        ["bash", "-c", _STOP_RUN_SCRIPT, "stop", os.fspath(run_dir)],
                        env={**os.environ, "BASH_ENV": os.fspath(bash_env), "KILL_LOG": os.fspath(log), "MODE": mode},
                        capture_output=True, text=True, check=False,
                    )
                    calls = [line for line in log.read_text().splitlines() if line.startswith("-")]
                    self.assertEqual(calls[0], "-TERM -- -12345")
                    kill_index = calls.index("-KILL -- -12345")
                    self.assertEqual(calls[1:kill_index], ["-0 -- -12345"] * 6)
                    self.assertTrue(calls[kill_index + 1:])
                    self.assertTrue(all(call == "-0 -- -12345" for call in calls[kill_index + 1:]))
                    if mode == "drain":
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(calls[kill_index + 1:], ["-0 -- -12345"] * 4)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("remains alive after KILL", result.stderr)

    def test_stop_producers_remains_unset_after_failed_group_shutdown(self):
        from voice.evals.dev_stack import RemoteCommandError

        stack = _RunStack(checkout=CHECKOUT, batch=SimpleNamespace())
        stack._remote_dir = "/tmp/mentat-eval-batch.synthetic/runs/failure"
        failure = RemoteCommandError(1, ["ssh"], stderr="producer group remains alive after KILL")
        with patch.object(stack, "_remote", side_effect=failure) as remote:
            with self.assertRaises(RemoteCommandError):
                stack._stop_producers()
        self.assertEqual(remote.call_args.args[0], _STOP_RUN_SCRIPT)
        self.assertFalse(stack._producers_stopped)

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
            if args[:2] == ["ssh", "ultraviolet"] and "python3 -c" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(args, 0, '{"turns":[]}\n', "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with isolated_run(checkout=CHECKOUT, opt_in=True, run=run) as stack:
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
            setup_args[-3:-1],
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

    def test_remote_port_selection_failure_cleans_up_the_run_without_restoring_siblings(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval-batch.probefailure\n", "")
            if "MENTAT_EVAL_PORT_PROBE" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(args, 0, "invalid probe response", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with DevStack(checkout=CHECKOUT, opt_in=True, run=run) as batch:
            with self.assertRaisesRegex(RuntimeError, "remote port probe"):
                with batch.run("bad-port-run"):
                    pass

        probe = [
            kwargs.get("input", "") for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and "MENTAT_EVAL_PORT_PROBE" in kwargs.get("input", "")
        ]
        cleanup = [
            kwargs.get("input", "") for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and 'bash "$BATCH_DIR/cleanup.sh"' in kwargs.get("input", "")
        ]
        self.assertEqual(len(probe), 1)
        self.assertEqual(len(cleanup), 1)
        self.assertIn('rm -rf -- "$BATCH_DIR"', cleanup[0])
        self.assertNotIn("systemctl start mentat-voice", probe[0])

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

        with isolated_run(checkout=CHECKOUT, opt_in=True, run=run, local_port=0, dev_port=dev_port, health_port=8486) as stack:
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
        self.assertIn('"MENTAT_STATE_PATH": str(run_dir / "home/mentat/state.json")', setup)
        self.assertIn('cp -a "$SHARED_DIR/mentat" "$RUN_DIR/mentat"', setup)
        self.assertIn('mkdir -m 700 -p "$RUN_DIR/home/mentat"', setup)
        self.assertIn('chown -R mentat:mentat "$RUN_DIR/mentat" "$RUN_DIR/home/mentat"', setup)
        with tempfile.TemporaryDirectory() as temporary_directory:
            state_home = Path(temporary_directory) / "home/mentat"
            state_home.mkdir(parents=True, mode=0o700)
            os.chmod(state_home, 0o700)
            state_path = state_home / "state.json"
            temporary_state = state_path.with_name("state.json.tmp")
            temporary_state.write_text('{"session":"persisted"}')
            os.replace(temporary_state, state_path)
            self.assertEqual(state_path.read_text(), '{"session":"persisted"}')
            self.assertEqual(stat.S_IMODE(state_home.stat().st_mode), 0o700)
        self.assertIn('"MENTAT_RECORD_DIR": str(run_dir / "records")', setup)
        self.assertIn('"HOME": str(run_dir / "home/mentat")', setup)
        self.assertIn('"MENTAT_MCP_CONFIG"', setup)
        self.assertIn('"MENTAT_LISTEN": f"127.0.0.1:{dev_port}"', setup)
        self.assertIn('"--reuid=nobody"', setup)
        self.assertIn('voice/evals/phone.py', " ".join(transfers[1]))
        self.assertNotIn("systemctl stop mentatd", setup)
        self.assertIn("systemctl stop mentat-voice", setup)
        self.assertIn("systemctl start mentat-voice", setup)
        worker_script = next(script for script in remote_scripts if "--room" in script)
        setup_script = next(script for script in remote_scripts if '"MENTAT_STATE_PATH"' in script)
        self.assertIn("systemd-run", _BATCH_SETUP_SCRIPT)
        self.assertIn("systemctl stop mentat-voice", worker_script)
        self.assertIn("agent.pid", setup)

    def test_candidate_daemon_uses_requested_model_over_production_environment(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval-batch.modeltest\n", "")
            if "MENTAT_EVAL_PORT_PROBE" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(
                    args, 0, json.dumps({"dev_port": 49151, "health_port": 49152}), ""
                )
            if "MENTAT_VOICE_GRANT" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(args, 0, '{"turns":[]}\n', "")
            return subprocess.CompletedProcess(args, 0, "", "")

        requested_model = "claude-opus-5-5"
        with patch("voice.evals.dev_stack.subprocess.Popen", return_value=unittest.mock.Mock(poll=lambda: None)):
            with patch.dict(os.environ, {"MENTAT_VOICE_MODEL": requested_model}, clear=False):
                with isolated_run(checkout=CHECKOUT, opt_in=True, run=run):
                    pass

        launch_args, launch_kwargs = next(
            (args, kwargs)
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
            and '"MENTAT_STATE_PATH"' in kwargs.get("input", "")
        )
        self.assertEqual(launch_args[-1], requested_model)
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            (run_dir / "mentat").mkdir(parents=True)
            (run_dir / "memory").mkdir(mode=0o700)
            (run_dir / "mentat/prompt.md").write_text("Synthetic candidate prompt")
            (run_dir / "mentat.env.json").write_text(json.dumps({
                "PATH": "/usr/bin",
                "MENTAT_LISTEN": "127.0.0.1:8484",
                "MENTAT_SYSTEM_PROMPT": "Production prompt",
                "MENTAT_MEMORY_DIR": "/production/memory",
            }))
            for name, content in (("node.path", "/nix/bin/node"), ("setpriv.path", "/usr/bin/setpriv")):
                (run_dir / name).write_text(content)
            candidate_env = run_setup_environment(run_dir, requested_model)

        self.assertEqual(candidate_env["MENTAT_VOICE_MODEL"], requested_model)
        self.assertEqual(candidate_env["MENTAT_SESSION_TTL"], "90s")
        self.assertEqual(candidate_env["MENTAT_MEMORY_DIR"], os.fspath(run_dir / "memory"))
        self.assertEqual(candidate_env["MENTAT_SYSTEM_PROMPT"], "Synthetic candidate prompt")
        self.assertNotEqual(candidate_env["MENTAT_MEMORY_DIR"], "/production/memory")
        self.assertEqual(launch_kwargs["input"].count("nix build"), 0)

        default_model = "chatgpt/sol-fast"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MENTAT_VOICE_MODEL", None)
            with patch("voice.evals.dev_stack.subprocess.Popen", return_value=unittest.mock.Mock(poll=lambda: None)):
                with isolated_run(checkout=CHECKOUT, opt_in=True, run=run):
                    pass
        default_launch_args, _ = next(
            (args, kwargs)
            for args, kwargs in reversed(calls)
            if args[:2] == ["ssh", "ultraviolet"]
            and '"MENTAT_STATE_PATH"' in kwargs.get("input", "")
        )
        self.assertEqual(default_launch_args[-1], default_model)

    def test_candidate_memory_store_is_private_and_overrides_production_path(self):
        setup = _RUN_SETUP_SCRIPT.replace("__MCP_REWRITE_SOURCE__", _MCP_REWRITE_SOURCE)
        self.assertIn('"$RUN_DIR/memory"', setup)
        self.assertIn('"MENTAT_MEMORY_DIR": str(run_dir / "memory")', setup)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            (run_dir / "mentat").mkdir(parents=True)
            memory_dir = run_dir / "memory"
            memory_dir.mkdir(mode=0o700)
            os.chmod(memory_dir, 0o700)
            (run_dir / "mentat/prompt.md").write_text("Candidate prompt")
            production_path = root / "production-memory"
            production_values = {
                "PATH": "/usr/bin",
                "MENTAT_LISTEN": "127.0.0.1:8484",
                "MENTAT_SYSTEM_PROMPT": "Production prompt",
                "MENTAT_MEMORY_DIR": os.fspath(production_path),
            }
            production_bytes = json.dumps(production_values).encode()
            (run_dir / "mentat.env.json").write_bytes(production_bytes)
            (run_dir / "node.path").write_text("/nix/bin/node")
            (run_dir / "setpriv.path").write_text("/usr/bin/setpriv")

            candidate_env = run_setup_environment(run_dir, "synthetic-model")

            self.assertEqual(candidate_env["MENTAT_MEMORY_DIR"], os.fspath(memory_dir))
            self.assertNotEqual(candidate_env["MENTAT_MEMORY_DIR"], os.fspath(production_path))
            self.assertEqual(list(memory_dir.iterdir()), [])
            self.assertEqual(stat.S_IMODE(memory_dir.stat().st_mode), 0o700)
            self.assertEqual((run_dir / "mentat.env.json").read_bytes(), production_bytes)

    def test_candidate_prompt_overrides_production_prompt_and_missing_fails_closed(self):
        calls = []

        def stage_run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval-batch.prompttest\n", "")
            if "MENTAT_EVAL_PORT_PROBE" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(
                    args, 0, json.dumps({"dev_port": 49151, "health_port": 49152}), ""
                )
            if "MENTAT_VOICE_GRANT" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(args, 0, '{"turns":[]}\n', "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("voice.evals.dev_stack.subprocess.Popen", return_value=unittest.mock.Mock(poll=lambda: None)):
            with isolated_run(checkout=CHECKOUT, opt_in=True, run=stage_run):
                pass
        transfers = [args for args, _ in calls if args and args[0] == "scp"]
        self.assertTrue(any(os.fspath(CHECKOUT / "prompt.md") in args for args in transfers))

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            mentat = run_dir / "mentat"
            mentat.mkdir(parents=True)
            (run_dir / "memory").mkdir(mode=0o700)
            candidate_prompt = "Candidate interpreter prompt: [[en]]\n"
            (mentat / "prompt.md").write_text(candidate_prompt)
            (run_dir / "mentat.env.json").write_text(json.dumps({
                "PATH": "/usr/bin",
                "MENTAT_LISTEN": "127.0.0.1:8484",
                "MENTAT_SYSTEM_PROMPT": "Production prompt without tags",
            }))
            (run_dir / "node.path").write_text("/nix/bin/node")
            (run_dir / "setpriv.path").write_text("/usr/bin/setpriv")
            candidate_env = run_setup_environment(run_dir, "synthetic-model")
            self.assertEqual(candidate_env["MENTAT_SYSTEM_PROMPT"], candidate_prompt)

            (mentat / "prompt.md").unlink()
            with self.assertRaisesRegex(RuntimeError, "candidate system prompt"):
                run_setup_environment(run_dir, "synthetic-model")

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
            with isolated_run(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
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
            with isolated_run(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
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
                    with self.assertRaisesRegex(TimeoutError, "health endpoint"):
                        with isolated_run(
                            checkout=CHECKOUT, opt_in=True, dev_port=8485,
                            health_port=8486, run=run,
                        ):
                            pass

        self.assertGreater(urlopen.call_count, 1)
        self.assertLessEqual(now[0], 0.7)
        tunnel.terminate.assert_called_once()
        self.assertTrue(any(
            "systemctl start mentat-voice" in kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
        ))

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
                with self.assertRaises(KeyboardInterrupt):
                    with isolated_run(
                        checkout=CHECKOUT, opt_in=True, dev_port=8485,
                        health_port=8486, run=run,
                    ):
                        pass

        tunnel.terminate.assert_called_once()
        self.assertTrue(any(
            "systemctl start mentat-voice" in kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
        ))

    def test_successive_worker_starts_preserve_room_scoped_voice_mode_entries(self):
        scenario = next(item for item in SCENARIOS if item.name == "spanish-interpreter")
        sidecars = list(scenario.caller_lines)

        def interpreter_trace(room, offset):
            return [
                {
                    "room": room,
                    "event": "mode",
                    "mode": "interpreter",
                    "language": "es",
                    "voice_id": "spanish-library",
                    "created_at": offset + 101.0,
                    "lookup_ms": 24.5,
                    "selection": "resolved",
                },
                {
                    "room": room,
                    "event": "mode",
                    "mode": "normal",
                    "language": "en",
                    "voice_id": "english-default",
                    "created_at": offset + 106.0,
                    "lookup_ms": 0.0,
                    "selection": "default",
                },
                *[
                    {
                        "room": room,
                        "event": "speech",
                        "reply": index,
                        "turn_id": f"{room}-turn-{index}",
                        "language": language,
                        "voice_id": voice,
                        "created_at": offset + timestamp,
                    }
                    for index, (language, voice, timestamp) in enumerate(
                        zip(
                            scenario.reply_languages,
                            (
                                "spanish-library",
                                "english-default",
                                "spanish-library",
                                "english-default",
                                "english-default",
                                "english-default",
                            ),
                            (102.0, 103.0, 104.0, 105.0, 105.5, 107.0),
                            strict=True,
                        ),
                        1,
                    )
                ],
            ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evals = root / "voice/evals"
            evals.mkdir(parents=True)
            trace = evals / "voice-modes.jsonl"
            self.assertFalse(trace.exists())

            bin_dir = root / "bin"
            bin_dir.mkdir()
            for command in ("systemctl", "chown", "chmod", "install"):
                executable = bin_dir / command
                executable.write_text("#!/bin/sh\nexit 0\n")
                executable.chmod(0o700)

            setup_script = _START_WORKER_SCRIPT.split(
                'python3 - "$DEV_DIR" "$DEV_PORT" "$HEALTH_PORT" "$ROOM" <<\'PY\'\n',
                1,
            )[0]
            environment = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}

            def start_worker(room):
                result = subprocess.run(
                    ["bash", "-s", "--", str(root), "8485", "8486", room],
                    input=setup_script,
                    text=True,
                    check=False,
                    capture_output=True,
                    env=environment,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            start_worker("interpreter-room-one")
            self.assertEqual(trace.read_text(), "")
            first_trace = interpreter_trace("interpreter-room-one", 0.0)
            trace.write_text("".join(json.dumps(entry) + "\n" for entry in first_trace))
            start_worker("interpreter-room-two")
            second_trace = interpreter_trace("interpreter-room-two", 10.0)
            with trace.open("a") as retained:
                retained.write("".join(json.dumps(entry) + "\n" for entry in second_trace))

            entries = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual(len(entries), len(first_trace) + len(second_trace))
            for room in ("interpreter-room-one", "interpreter-room-two"):
                with self.subTest(room=room):
                    failures, _ = runner._spanish_interpreter_evidence_failures(
                        scenario, room, entries, sidecars, []
                    )
                    self.assertEqual(failures, [])

            subprocess.run(
                ["bash", "-s", "--", str(root)],
                input=_RETAIN_EVIDENCE_SCRIPT,
                text=True,
                check=True,
                capture_output=True,
                env={**os.environ, "SUDO_USER": pwd.getpwuid(os.getuid()).pw_name},
            )
            with tarfile.open(root / "retained-evidence.tar.gz", "r:gz") as archive:
                member = archive.extractfile("voice/evals/voice-modes.jsonl")
                self.assertIsNotNone(member)
                archived_entries = [
                    json.loads(line) for line in member.read().decode().splitlines()
                ]
            self.assertEqual(
                {entry["room"] for entry in archived_entries},
                {"interpreter-room-one", "interpreter-room-two"},
            )
            self.assertEqual(archived_entries, entries)

            corrupted_entries = [dict(entry) for entry in entries]
            for entry in corrupted_entries:
                if entry["room"] == "interpreter-room-two":
                    if entry["event"] == "mode" and entry["mode"] == "interpreter":
                        entry["language"] = "fr"
                    elif entry["event"] == "speech" and entry["reply"] == 2:
                        entry["language"] = "es"
            first_failures, _ = runner._spanish_interpreter_evidence_failures(
                scenario, "interpreter-room-one", corrupted_entries, sidecars, []
            )
            second_failures, _ = runner._spanish_interpreter_evidence_failures(
                scenario, "interpreter-room-two", corrupted_entries, sidecars, []
            )
            self.assertEqual(first_failures, [])
            self.assertTrue(
                any("mode transition 1 did not select interpreter/es" in failure for failure in second_failures),
                second_failures,
            )
            self.assertTrue(
                any("speech reply 2 used the wrong language" in failure for failure in second_failures),
                second_failures,
            )

    @patch("voice.evals.dev_stack.subprocess.Popen")
    def test_worker_creates_private_mode_trace_and_sets_only_worker_opt_in(self, popen):
        popen.return_value = unittest.mock.Mock(poll=lambda: None)
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.trace\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with isolated_run(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
            stack.start_worker("trace-room")

        worker_script = next(
            kwargs["input"] for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and "--room" in kwargs.get("input", "")
        )
        trace_creation = worker_script.index(
            'if [ ! -e "$DEV_DIR/voice/evals/voice-modes.jsonl" ]; then'
        )
        worker_launch = worker_script.index("voice = subprocess.Popen(")
        self.assertLess(trace_creation, worker_launch)
        self.assertIn('chown nobody:nogroup "$DEV_DIR/voice/evals/voice-modes.jsonl"', worker_script)
        self.assertIn('chmod 600 "$DEV_DIR/voice/evals/voice-modes.jsonl"', worker_script)
        self.assertIn(
            '"MENTAT_EVAL_VOICE_LOG": str(dev_dir / "voice/evals/voice-modes.jsonl")',
            worker_script,
        )
        self.assertNotIn("MENTAT_EVAL_VOICE_LOG", _RUN_SETUP_SCRIPT)

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

        with isolated_run(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
            stack.start_worker(room)

        scripts = [kwargs.get("input", "") for args, kwargs in calls if args[:2] == ["ssh", "ultraviolet"]]
        daemon_script = next(script for script in scripts if '"MENTAT_STATE_PATH"' in script)
        worker_args, worker_kwargs = next(
            (args, kwargs)
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and "--room" in kwargs.get("input", "")
        )
        worker_script = worker_kwargs["input"]
        self.assertIn(room, worker_args)
        self.assertIn('"$ROOM"', worker_script)
        self.assertIn("systemd-run", _BATCH_SETUP_SCRIPT)
        self.assertIn("systemctl stop mentat-voice", worker_script)
        self.assertIn('"MENTAT_EVAL_DELEGATION_LOG": str(dev_dir / "voice/evals/delegations.jsonl")', worker_script)
        self.assertIn('voice/evals/delegations.jsonl', worker_script)
        marker_stage = worker_script.index(': > "$DEV_DIR/voice/evals/delegations.jsonl"')
        worker_launch = worker_script.index('voice = subprocess.Popen(')
        self.assertLess(marker_stage, worker_launch)
        self.assertIn('chown nobody:nogroup "$DEV_DIR/voice/evals/delegations.jsonl"', worker_script)
        self.assertIn('chmod 600 "$DEV_DIR/voice/evals/delegations.jsonl"', worker_script)
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

        with isolated_run(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
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
                args,
                0,
                json.dumps({
                    "turns": [{
                        "transcript": f"caller output {token}",
                        "credential": "API_KEY=assignment-secret",
                        "authorization": "Authorization: Bearer header-secret",
                    }],
                }),
                f"caller warning {token}",
            )

        stack = _RunStack(checkout=CHECKOUT, remote="ultraviolet", run=run, batch=SimpleNamespace())
        stack._entered = True
        stack._remote_dir = "/tmp/mentat-eval.test"
        result = stack.run_voice(
            ["evals/runner.py", "--room", "token-room"],
            token=token,
            livekit_url=livekit_url,
        )

        self.assertEqual(
            json.loads(result.stdout),
            {
                "turns": [{
                    "transcript": "caller output [REDACTED]",
                    "credential": "API_KEY=[REDACTED]",
                    "authorization": "Authorization: Bearer [REDACTED]",
                }],
            },
        )
        self.assertEqual(result.stderr, "caller warning [REDACTED]")
        self.assertEqual(len(calls), 2)
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
        self.assertIn('RUN_DIR=$1', calls[1][1]["input"])
        self.assertIn('"$RUN_DIR/agent.pid" "$RUN_DIR/voice.pid"', calls[1][1]["input"])

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

                stack = _RunStack(checkout=CHECKOUT, remote="ultraviolet", run=run, batch=SimpleNamespace())
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

    def test_candidate_voice_environment_is_gc_rooted_until_cleanup(self):
        setup_script = _BATCH_SETUP_SCRIPT
        cleanup_script = _BATCH_CLEANUP_SCRIPT

        self.assertIn('--out-link "$SHARED_DIR/voice-env-root"', setup_script)
        self.assertIn("--print-out-paths", setup_script)
        self.assertIn('/nix/store/*)', setup_script)
        self.assertIn('test -x "$VOICE_PY"', setup_script)
        self.assertIn('VOICE_PY="$VOICE_ENV_PATH/bin/python"', setup_script)
        self.assertIn('rm -f -- "$BATCH_DIR/shared/voice-env-root"', cleanup_script)
        self.assertLess(
            cleanup_script.index('rm -f -- "$BATCH_DIR/shared/voice-env-root"'),
            cleanup_script.index('rm -rf -- "$BATCH_DIR"'),
        )
        self.assertIn("systemctl start mentat-voice", cleanup_script)
        self.assertIn("systemctl stop mentat-voice", cleanup_script)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            systemctl_log = root / "systemctl.log"
            systemctl = bin_dir / "systemctl"
            systemctl.write_text(
                "#!/bin/sh\n"
                'if [ "$2" = mentat-voice ]; then\n'
                '  if [ -L "$BATCH_DIR/shared/voice-env-root" ]; then state=root-present; else state=root-missing; fi\n'
                '  printf "%s %s %s\\n" "$1" "$2" "$state" >> "$SYSTEMCTL_LOG"\n'
                'else\n'
                '  printf "%s\\n" "$*" >> "$SYSTEMCTL_LOG"\n'
                'fi\n'
            )
            systemctl.chmod(0o700)
            bash_env = root / "bash-env"
            bash_env.write_text(
                "kill() {\n"
                '  printf "%s\\n" "$*" >> "$KILL_LOG"\n'
                '  if [ "$1" = -0 ]; then return 1; fi\n'
                "  return 0\n"
                "}\n"
            )

            for action in ("start", "stop"):
                dev_dir = root / f"dev-stack-{action}"
                shared = dev_dir / "shared"
                run_dir = dev_dir / "runs" / "one"
                shared.mkdir(parents=True)
                run_dir.mkdir(parents=True)
                root_link = shared / "voice-env-root"
                root_link.symlink_to("/nix/store/candidate")
                (dev_dir / "restore-action").write_text(f"{action}\n")
                (dev_dir / "restore-unit").write_text("mentat-eval-restore-test\n")
                (run_dir / "agent.pid").write_text("12345\n")
                (run_dir / "voice.pid").write_text("12346\n")
                cleanup_path = dev_dir / "cleanup.sh"
                cleanup_path.write_text(cleanup_script)
                result = subprocess.run(
                    ["bash", os.fspath(cleanup_path), os.fspath(dev_dir)],
                    capture_output=True,
                    text=True,
                    check=False,
                    env={
                        **os.environ,
                        "BATCH_DIR": os.fspath(dev_dir),
                        "SYSTEMCTL_LOG": os.fspath(systemctl_log),
                        "KILL_LOG": os.fspath(root / "kill.log"),
                        "BASH_ENV": os.fspath(bash_env),
                        "PATH": f"{bin_dir}:{os.environ['PATH']}",
                    },
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                calls = systemctl_log.read_text() if systemctl_log.exists() else ""
                self.assertIn(f"{action} mentat-voice root-present", calls)
                self.assertIn("-TERM -- -12345", (root / "kill.log").read_text())
                self.assertFalse(dev_dir.exists())
                self.assertFalse(root_link.is_symlink())

    def test_candidate_voice_environment_uses_host_nixpkgs_and_both_candidate_launches(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval-batch.voiceenv\n", "")
            if "MENTAT_EVAL_PORT_PROBE" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(
                    args, 0, json.dumps({"dev_port": 49151, "health_port": 49152}), ""
                )
            if "MENTAT_VOICE_GRANT" in kwargs.get("input", ""):
                return subprocess.CompletedProcess(args, 0, '{"turns":[]}\n', "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("voice.evals.dev_stack.subprocess.Popen", return_value=unittest.mock.Mock(poll=lambda: None)):
            with isolated_run(checkout=CHECKOUT, opt_in=True, run=run) as stack:
                stack.start_worker("candidate-env-room")
                stack.run_voice(["caller.py"], token="test-token", livekit_url="wss://lk.invalid")

        scp_calls = [args for args, _ in calls if args and args[0] == "scp"]
        self.assertTrue(any(str(CHECKOUT / "nix/voice-env.nix") in args for args in scp_calls))
        batch_script = _BATCH_SETUP_SCRIPT.replace(
            "__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE
        )
        run_script = _RUN_SETUP_SCRIPT.replace("__MCP_REWRITE_SOURCE__", _MCP_REWRITE_SOURCE)
        self.assertIn("builtins.getFlake", batch_script)
        self.assertEqual(batch_script.count("nix build"), 1)
        self.assertIn("$SHARED_DIR/voice/voice-env.nix", batch_script)
        self.assertIn('"$SHARED_DIR/voice-env-root"', batch_script)
        self.assertIn('"voice-python.path"', batch_script)
        self.assertNotIn("nix build", run_script)
        for module in (
            "livekit.plugins.dtln", "livekit.plugins.elevenlabs",
            "livekit.plugins.turn_detector", "livekit.plugins.silero",
        ):
            self.assertIn(module, batch_script)
        self.assertNotIn("livekit.plugins.openai", batch_script)
        self.assertIn('voice_python = (dev_dir / "voice-python.path").read_text().strip()', _START_WORKER_SCRIPT)
        self.assertIn('voice_python = (DEV_DIR / "voice-python.path").read_text().strip()', _RUN_VOICE_SCRIPT)
        self.assertEqual(stack.run_id, "test-run")

    def test_candidate_voice_python_preflight_reports_the_missing_plugin_by_name(self):
        preflight = _BATCH_SETUP_SCRIPT.split('"$VOICE_PY" - <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
        observed = []

        def import_module(module):
            observed.append(module)
            if module == "livekit.plugins.turn_detector":
                raise ImportError("No module named livekit.plugins.turn_detector")
            return object()

        with patch("importlib.import_module", side_effect=import_module):
            with self.assertRaises(SystemExit) as failure:
                exec(preflight, {})

        self.assertEqual(
            str(failure.exception),
            "candidate voice environment missing required module livekit.plugins.turn_detector: "
            "No module named livekit.plugins.turn_detector",
        )
        self.assertEqual(
            observed,
            ["aiohttp", "livekit.api", "livekit.rtc", "livekit.plugins.dtln",
             "livekit.plugins.elevenlabs", "livekit.plugins.silero",
             "livekit.plugins.turn_detector"],
        )

    def test_setpriv_is_resolved_outside_service_path_for_all_launches(self):
        def python_block(script, header):
            return script.split(header + chr(10), 1)[1].split(chr(10) + "PY" + chr(10), 1)[0]

        self.assertIn('setpriv_path = shutil.which("setpriv")', _BATCH_SETUP_SCRIPT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            (run_dir / "mentat").mkdir(parents=True)
            (run_dir / "voice/evals").mkdir(parents=True)
            (run_dir / "memory").mkdir(mode=0o700)
            (run_dir / "mentat/prompt.md").write_text("Synthetic candidate prompt")
            service_path = str(root / "service-bin")
            production_env = {
                "PATH": service_path,
                "MENTAT_LISTEN": "127.0.0.1:8484",
            }
            (run_dir / "mentat.env.json").write_text(json.dumps(production_env))
            (run_dir / "voice.env.json").write_text(json.dumps({
                "PATH": service_path,
                "LIVEKIT_API_SECRET": "private-env-secret",
            }))
            setup_bin = root / "setup-bin"
            setup_bin.mkdir()
            setpriv = setup_bin / "setpriv"
            setpriv.write_text("fake setpriv executable")
            setpriv.chmod(0o755)
            resolved_setpriv = str(setpriv.resolve())
            (run_dir / "launch.lock").touch()
            (run_dir / "setpriv.path").write_text(resolved_setpriv)
            (run_dir / "node.path").write_text("/usr/bin/node")
            (run_dir / "voice-python.path").write_text("/nix/store/python/bin/python3")

            setup = _RUN_SETUP_SCRIPT.replace("__MCP_REWRITE_SOURCE__", _MCP_REWRITE_SOURCE)
            setup_body = python_block(
                setup,
                'python3 - "$RUN_DIR" "$DEV_PORT" "$HEALTH_PORT" "$VOICE_MODEL" <<\'PY\'',
            )
            with patch.dict(os.environ, {"PATH": str(setup_bin)}):
                with patch("sys.argv", ["setup.py", str(run_dir), "8485", "8486", "synthetic-model"]):
                    with patch("subprocess.Popen", return_value=SimpleNamespace(pid=1234)) as popen:
                        exec(setup_body, {})
            daemon_argv = popen.call_args.args[0]
            self.assertEqual(daemon_argv[0], resolved_setpriv)
            self.assertEqual(popen.call_args.kwargs["env"]["PATH"], service_path)
            self.assertEqual((run_dir / "setpriv.path").read_text(), resolved_setpriv)

            worker_body = python_block(
                _START_WORKER_SCRIPT,
                'python3 - "$DEV_DIR" "$DEV_PORT" "$HEALTH_PORT" "$ROOM" <<\'PY\'',
            )
            with patch("sys.argv", ["worker.py", str(run_dir), "8485", "8486", "worker-room"]):
                with patch("subprocess.Popen", return_value=SimpleNamespace(pid=1235)) as popen:
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
                stdout=json.dumps({"turns": [{"transcript": "DISTINCTIVE-CAUSE " + token}]}),
                stderr="private-env-secret " + token,
                returncode=0,
            )
            caller_stdout = io.StringIO()
            caller_stderr = io.StringIO()
            with patch("sys.argv", ["-c", "--", str(run_dir), "8485", "8486"]):
                with patch("sys.stdin", io.StringIO(caller_payload)):
                    with patch("sys.stdout", caller_stdout), patch("sys.stderr", caller_stderr):
                        with patch("subprocess.run", return_value=result) as remote_run:
                            with patch("sys.exit") as exit_process:
                                exec(_RUN_VOICE_SCRIPT, {})
            exit_process.assert_called_once_with(0)
            self.assertNotIn("private-env-secret", caller_stdout.getvalue())
            self.assertNotIn(token, caller_stdout.getvalue())
            caller_argv = remote_run.call_args.args[0]
            self.assertEqual(caller_argv[0], resolved_setpriv)
            self.assertEqual(remote_run.call_args.kwargs["env"]["PATH"], service_path)
            self.assertEqual(remote_run.call_args.kwargs["env"]["MENTAT_VOICE_TOKEN"], token)

    def test_voice_caller_redacts_json_values_without_corrupting_machine_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            dev_dir = Path(temporary)
            (dev_dir / "launch.lock").touch()
            (dev_dir / "voice-python.path").write_text("/nix/store/python/bin/python3")
            (dev_dir / "setpriv.path").write_text("/usr/bin/setpriv")
            (dev_dir / "voice.env.json").write_text(json.dumps({
                "PATH": "/usr/bin",
                "LIVEKIT_API_SECRET": "private-env-secret",
            }))
            token = "issued-token.header.signature"
            machine = {
                "turns": [{
                    "transcript": f'Assistant got "{token}"',
                    "diagnostic": "API_KEY=assignment-secret",
                    "authorization": "Authorization: Bearer header-secret",
                    "payload": {"LIVEKIT_API_SECRET": "private-env-secret"},
                }],
            }
            caller_result = SimpleNamespace(
                stdout=json.dumps(machine),
                stderr=f"caller warning {token} private-env-secret",
                returncode=0,
            )
            caller_payload = json.dumps({
                "command": ["evals/runner.py", "--room", "caller-room"],
                "token": token,
                "livekit_url": "wss://issued-livekit.invalid",
            })
            caller_stdout = io.StringIO()
            caller_stderr = io.StringIO()
            with patch("sys.argv", ["-c", "--", str(dev_dir), "8485", "8486"]):
                with patch("sys.stdin", io.StringIO(caller_payload)):
                    with patch("sys.stdout", caller_stdout), patch("sys.stderr", caller_stderr):
                        with patch("subprocess.run", return_value=caller_result):
                            with patch("sys.exit") as exit_process:
                                exec(_RUN_VOICE_SCRIPT, {})
            exit_process.assert_called_once_with(0)
            output = caller_stdout.getvalue()
            parsed = json.loads(output)
            self.assertEqual(parsed["turns"][0]["transcript"], "Assistant got \"[REDACTED]\"")
            self.assertEqual(parsed["turns"][0]["diagnostic"], "API_KEY=[REDACTED]")
            self.assertEqual(parsed["turns"][0]["authorization"], "Authorization: Bearer [REDACTED]")
            self.assertEqual(parsed["turns"][0]["payload"]["LIVEKIT_API_SECRET"], "[REDACTED]")
            for secret in (token, "private-env-secret", "assignment-secret", "header-secret"):
                self.assertNotIn(secret, output)
                self.assertNotIn(secret, caller_stderr.getvalue())

    def test_voice_caller_rejects_malformed_machine_json_and_scrubs_diagnostic(self):
        with tempfile.TemporaryDirectory() as temporary:
            dev_dir = Path(temporary)
            (dev_dir / "launch.lock").touch()
            (dev_dir / "voice-python.path").write_text("/nix/store/python/bin/python3")
            (dev_dir / "setpriv.path").write_text("/usr/bin/setpriv")
            (dev_dir / "voice.env.json").write_text(json.dumps({
                "PATH": "/usr/bin",
                "LIVEKIT_API_SECRET": "private-env-secret",
            }))
            token = "issued-token.header.signature"
            caller_payload = json.dumps({
                "command": ["evals/runner.py", "--room", "caller-room"],
                "token": token,
                "livekit_url": "wss://issued-livekit.invalid",
            })
            caller_stdout = io.StringIO()
            caller_stderr = io.StringIO()
            with patch("sys.argv", ["-c", "--", str(dev_dir), "8485", "8486"]):
                with patch("sys.stdin", io.StringIO(caller_payload)):
                    with patch("sys.stdout", caller_stdout), patch("sys.stderr", caller_stderr):
                        with patch("subprocess.run", return_value=SimpleNamespace(
                            stdout='{"turns":[{"transcript":"unterminated ' + token,
                            stderr="",
                            returncode=0,
                        )):
                            with self.assertRaises(SystemExit) as raised:
                                exec(_RUN_VOICE_SCRIPT, {})
            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(caller_stdout.getvalue(), "")
            self.assertIn("voice caller stdout was not valid JSON", caller_stderr.getvalue())
            self.assertNotIn(token, caller_stderr.getvalue())

    def test_setup_fails_if_setpriv_cannot_be_resolved(self):
        setup = _BATCH_SETUP_SCRIPT.replace(
            "__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE
        )
        header = 'python3 - "$BATCH_DIR" "$MENTAT_PID" "$VOICE_PID" "$NODE_BIN" "$VOICE_PY" <<\'PY\''
        body = setup.split(header + "\n", 1)[1].split("\nPY\n", 1)[0]
        with tempfile.TemporaryDirectory() as temporary:
            with patch.dict(os.environ, {"PATH": str(Path(temporary) / "empty-bin")}):
                with patch("sys.argv", ["setup.py", temporary, "123", "456", "/nix/bin/node", "/nix/bin/python"]):
                    with patch("subprocess.Popen") as popen:
                        with self.assertRaisesRegex(RuntimeError, "setpriv executable is unavailable"):
                            exec(body, {})
            popen.assert_not_called()

    def test_run_voice_requires_entered_stack_and_propagates_remote_failure(self):
        run = unittest.mock.Mock()
        batch = DevStack(checkout=CHECKOUT, run=run)
        with self.assertRaisesRegex(RuntimeError, "batch is not running"):
            batch.run("before-enter")
        run.assert_not_called()

        stack = _RunStack(
            checkout=CHECKOUT, remote="ultraviolet", run=run, batch=SimpleNamespace()
        )
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

        with patch("voice.evals.dev_stack.subprocess.Popen", return_value=unittest.mock.Mock(poll=lambda: None)):
            with isolated_run(checkout=CHECKOUT, opt_in=True, run=run):
                pass

        first_scp = next(i for i, (args, _) in enumerate(calls) if args[0] == "scp")
        before_scp = [args for args, _ in calls[:first_scp]]
        self.assertTrue(any(args[:3] == ["ssh", "ultraviolet", "mkdir"] for args in before_scp))
        self.assertFalse(any(args[:3] == ["ssh", "ultraviolet", "sudo"] for args in before_scp))

    def test_private_voice_credential_is_copied_before_stopping_production_worker(self):
        setup = _BATCH_SETUP_SCRIPT.replace(
            "__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE
        )
        self.assertIn("MENTAT_VOICE_PRIVATE", setup)
        self.assertIn("stage_voice_private", setup)
        self.assertIn("os.chown", setup)
        self.assertIn("os.chmod", setup)
        self.assertIn("systemd-run", setup)
        self.assertIn("systemctl stop mentat-voice", _START_WORKER_SCRIPT)

    def test_private_voice_credential_is_copied_and_owned_by_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary) / "stage"
            staging.mkdir()
            source = Path(temporary) / "credential"
            source.write_text("private test context")
            identity_lookups = []
            namespace = {
                "Path": Path,
                "os": os,
                "pwd": SimpleNamespace(
                    getpwnam=lambda name: identity_lookups.append(("user", name))
                    or SimpleNamespace(pw_uid=os.getuid())
                ),
                "grp": SimpleNamespace(
                    getgrnam=lambda name: identity_lookups.append(("group", name))
                    or SimpleNamespace(gr_gid=os.getgid())
                ),
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
            self.assertEqual(identity_lookups, [("user", "nobody"), ("group", "nogroup")])

    def test_setup_captures_gateway_url_and_staged_key_for_candidate(self):
        setup_script = _BATCH_SETUP_SCRIPT.replace(
            "__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE
        )
        header = 'python3 - "$BATCH_DIR" "$MENTAT_PID" "$VOICE_PID" "$NODE_BIN" "$VOICE_PY" <<\'PY\''
        setup_body = setup_script.split(header + "\n", 1)[1].split("\nPY\n", 1)[0]
        setup_body = setup_body.replace('Path(f"/proc/{pid}/environ")', 'Path(environ_dir / pid)')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "batch"
            shared = batch / "shared"
            shared.mkdir(parents=True)
            environ_dir = root / "environ"
            environ_dir.mkdir()
            source = root / "production-key"
            source.write_bytes(b"synthetic-setup-gateway-key")
            gateway_url = "https://gateway.example.test/v1"
            (environ_dir / "123").write_bytes(
                f"MENTAT_LISTEN=127.0.0.1:8484\0MENTAT_VOICE_GATEWAY_URL={gateway_url}\0"
                f"MENTAT_VOICE_GATEWAY_KEY_FILE={source}\0".encode()
            )
            (environ_dir / "456").write_bytes(b"")
            namespace = {"environ_dir": environ_dir}
            with patch("sys.argv", ["setup.py", str(batch), "123", "456", "/nix/bin/node", "/nix/bin/python"]):
                with patch("pwd.getpwnam", return_value=SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid())):
                    with patch("grp.getgrnam", return_value=SimpleNamespace(gr_gid=os.getgid())):
                        with patch("shutil.which", return_value="/usr/bin/setpriv"):
                            exec(setup_body, namespace)

            candidate_env = json.loads((shared / "mentat.env.json").read_text())
            self.assertEqual(candidate_env["MENTAT_VOICE_GATEWAY_URL"], gateway_url)
            staged = Path(candidate_env["MENTAT_VOICE_GATEWAY_KEY_FILE"])
            self.assertEqual(staged, shared / "voice-gateway-key")
            self.assertEqual(staged.read_bytes(), b"synthetic-setup-gateway-key")
            self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o400)
            self.assertEqual(staged.stat().st_uid, os.getuid())
            self.assertEqual(staged.stat().st_gid, os.getgid())

    def test_gateway_credential_is_staged_for_dev_mentatd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "stage"
            staging.mkdir()
            source = root / "production-gateway-key"
            key = b"synthetic-gateway-key-only"
            source.write_bytes(key)
            values = {
                "MENTAT_VOICE_GATEWAY_URL": "https://gateway.example.test/v1",
                "MENTAT_VOICE_GATEWAY_KEY_FILE": str(source),
            }
            calls = []
            def get_user(name):
                calls.append(("user", name))
                return SimpleNamespace(pw_uid=os.getuid())

            def get_group(name):
                calls.append(("group", name))
                return SimpleNamespace(gr_gid=os.getgid())

            namespace = {
                "Path": Path,
                "os": os,
                "pwd": SimpleNamespace(getpwnam=get_user),
                "grp": SimpleNamespace(getgrnam=get_group),
            }
            exec(_PRIVATE_CREDENTIAL_SOURCE, namespace)

            namespace["stage_gateway_key"](values, staging)

            staged = Path(values["MENTAT_VOICE_GATEWAY_KEY_FILE"])
            self.assertEqual(staged.read_bytes(), key)
            self.assertNotEqual(staged, source)
            self.assertEqual(stat.S_IMODE(staged.stat().st_mode), 0o400)
            self.assertEqual(staged.stat().st_uid, os.getuid())
            self.assertEqual(staged.stat().st_gid, os.getgid())
            self.assertEqual(values["MENTAT_VOICE_GATEWAY_URL"], "https://gateway.example.test/v1")
            self.assertEqual(calls, [("user", "mentat"), ("group", "mentat")])

    def test_gateway_credential_is_optional_but_missing_named_file_fails_safely(self):
        namespace = {
            "Path": Path,
            "os": os,
            "pwd": SimpleNamespace(getpwnam=lambda _: SimpleNamespace(pw_uid=os.getuid())),
            "grp": SimpleNamespace(getgrnam=lambda _: SimpleNamespace(gr_gid=os.getgid())),
        }
        exec(_PRIVATE_CREDENTIAL_SOURCE, namespace)
        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            unconfigured = {"MENTAT_VOICE_PRIVATE": "/synthetic/voice-private"}
            namespace["stage_gateway_key"](unconfigured, staging)
            self.assertNotIn("MENTAT_VOICE_GATEWAY_URL", unconfigured)
            self.assertNotIn("MENTAT_VOICE_GATEWAY_KEY_FILE", unconfigured)

            missing = staging / "private" / "missing-key"
            configured = {
                "MENTAT_VOICE_GATEWAY_URL": "https://gateway.example.test/v1",
                "MENTAT_VOICE_GATEWAY_KEY_FILE": str(missing),
            }
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                with self.assertRaisesRegex(RuntimeError, "gateway credential") as raised:
                    namespace["stage_gateway_key"](configured, staging)
            diagnostic = str(raised.exception)
            self.assertNotIn(str(missing), diagnostic)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), "")

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

    def test_independent_restore_guard_precedes_worker_stop_and_is_disarmed_on_cleanup(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.guard\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("voice.evals.dev_stack.subprocess.Popen") as popen:
            popen.return_value = unittest.mock.Mock(poll=lambda: None)
            with isolated_run(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
                stack.start_worker("guard-test-room")

        remote_scripts = [
            kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and kwargs.get("input")
        ]
        setup = next(script for script in remote_scripts if "systemd-run" in script)
        worker = next(script for script in remote_scripts if "--room" in script)
        self.assertIn("systemd-run", setup)
        self.assertIn("systemctl stop mentat-voice", worker)
        self.assertIn("if systemctl is-active --quiet mentat-voice; then", setup)
        self.assertIn("RESTORE_ACTION=start", setup)
        self.assertIn("RESTORE_ACTION=stop", setup)
        self.assertIn('"$RESTORE_ACTION" mentat-voice', setup)
        self.assertIn("--on-active=30m", setup)
        self.assertIn('systemctl stop "$RESTORE_UNIT.timer" "$RESTORE_UNIT.service"', setup)
        self.assertIn('elif [ "$RESTORE_ACTION" = stop ]; then', setup)
        self.assertIn("systemctl stop mentat-voice", setup)
        self.assertLess(remote_scripts.index(setup), remote_scripts.index(worker))

    def test_concurrent_run_activity_does_not_storm_shared_restore_guard(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Lock

        calls = []
        calls_lock = Lock()

        def run(args, **kwargs):
            with calls_lock:
                calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, "", "")

        batch = DevStack(checkout=CHECKOUT, remote="ultraviolet", run=run)
        stacks = []
        for index in range(16):
            stack = _RunStack(
                checkout=CHECKOUT, remote="ultraviolet", run=run,
                batch=batch, run_id=f"run-{index}",
            )
            stack._remote_dir = f"/tmp/mentat-eval-batch.shared/runs/run-{index}"
            stack._restore_guard_armed = True
            stacks.append(stack)

        def remote_work(stack):
            stack._remote("capture attempt")

        with ThreadPoolExecutor(max_workers=16) as executor:
            list(executor.map(remote_work, stacks))

        refreshes = [
            kwargs["input"]
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"]
            and "systemctl restart" in kwargs.get("input", "")
        ]
        self.assertLessEqual(len(refreshes), 1)
        for script in refreshes:
            self.assertIn('systemctl restart "$RESTORE_UNIT.timer"', script)
            self.assertNotIn("systemctl start mentat-voice", script)

    def test_restore_guard_refreshes_periodically_without_run_work_and_stops_on_teardown(self):
        from threading import Event, Lock

        batch = DevStack(checkout=CHECKOUT, remote="ultraviolet")
        remote_dir = "/tmp/mentat-eval-batch.heartbeat"
        batch._remote_dir = remote_dir
        batch._entered = True
        calls = []
        calls_lock = Lock()
        first_refresh = Event()

        def remote(script, remote_dir, **_kwargs):
            with calls_lock:
                calls.append((script, remote_dir, time.monotonic()))
            first_refresh.set()
            return subprocess.CompletedProcess(["ssh"], 0, "", "")

        with patch.object(batch, "_remote", side_effect=remote), patch(
            "voice.evals.dev_stack._RESTORE_GUARD_REFRESH_INTERVAL_SECONDS", 0.05
        ):
            started = time.monotonic()
            batch._start_restore_guard_refresh()
            self.assertTrue(first_refresh.wait(timeout=2))
            batch._cleanup()
            time.sleep(0.12)

        refreshes = [call for call in calls if call[0] == _REFRESH_RESTORE_GUARD_SCRIPT]
        self.assertGreaterEqual(refreshes[0][2] - started, 0.04)
        self.assertEqual(len(refreshes), 1)
        self.assertEqual(refreshes[0][1], remote_dir)

    def test_batch_cleanup_stops_candidates_before_waiting_for_guard_refresh(self):
        from threading import Event, Thread

        refresh_started = Event()
        release_refresh = Event()
        run_stopped = Event()
        cleanup_finished = Event()
        events = []
        refresh_timeouts = []
        batch = DevStack(checkout=CHECKOUT, remote="ultraviolet")
        batch._remote_dir = "/tmp/mentat-eval-batch.blocked-refresh"
        batch._entered = True

        def run(args, **kwargs):
            script = kwargs.get("input", "")
            if script == _REFRESH_RESTORE_GUARD_SCRIPT:
                refresh_timeouts.append(kwargs.get("timeout"))
                events.append("refresh-started")
                refresh_started.set()
                release_refresh.wait(timeout=kwargs.get("timeout", 2))
                events.append("refresh-finished")
            else:
                events.append("batch-restore")
            return subprocess.CompletedProcess(args, 0, "", "")

        def stop_run():
            events.append("candidate-stopped")
            run_stopped.set()

        batch._run_stacks["run-1"] = SimpleNamespace(
            _remote_dir="/tmp/mentat-eval-batch.blocked-refresh/runs/run-1",
            _tunnel=None,
            _cleanup=stop_run,
        )

        with patch.object(batch, "_run", side_effect=run), patch(
            "voice.evals.dev_stack._RESTORE_GUARD_REFRESH_INTERVAL_SECONDS", 0.01
        ):
            batch._start_restore_guard_refresh()
            self.assertTrue(refresh_started.wait(timeout=1))
            cleanup = Thread(
                target=lambda: (batch._cleanup_batch(), cleanup_finished.set()), daemon=True
            )
            cleanup.start()
            stopped_while_refresh_blocked = run_stopped.wait(timeout=0.2)
            release_refresh.set()
            cleanup.join(timeout=2)

        self.assertTrue(cleanup_finished.is_set())
        self.assertTrue(stopped_while_refresh_blocked)
        self.assertEqual(len(refresh_timeouts), 1)
        self.assertIsNotNone(refresh_timeouts[0])
        self.assertLess(refresh_timeouts[0], 60)
        self.assertLess(events.index("candidate-stopped"), events.index("refresh-finished"))
        self.assertLess(events.index("refresh-finished"), events.index("batch-restore"))

    def test_candidate_worker_alone_receives_private_input_audio_directory(self):
        self.assertIn(
            '"MENTAT_VOICE_INPUT_RECORD_DIR": str(\n'
            '        dev_dir / "voice/evals/retained-evidence/input-audio"\n'
            '    )',
            _START_WORKER_SCRIPT,
        )
        self.assertIn(
            'install -d -o nobody -g nogroup -m 700 '
            '"$DEV_DIR/voice/evals/retained-evidence"',
            _START_WORKER_SCRIPT,
        )
        self.assertIn(
            'install -d -o nobody -g nogroup -m 700 '
            '"$DEV_DIR/voice/evals/retained-evidence/input-audio"',
            _START_WORKER_SCRIPT,
        )
        self.assertLess(
            _START_WORKER_SCRIPT.index('"$DEV_DIR/voice/evals/retained-evidence"'),
            _START_WORKER_SCRIPT.index('"$DEV_DIR/voice/evals/retained-evidence/input-audio"'),
        )
        self.assertNotIn("MENTAT_VOICE_INPUT_RECORD_DIR", _RUN_SETUP_SCRIPT)
        self.assertNotIn("MENTAT_VOICE_INPUT_RECORD_DIR", _RUN_VOICE_SCRIPT)

    def test_input_audio_retention_allowlists_wav_and_matching_transcript_sidecars(self):
        with tempfile.TemporaryDirectory() as staging:
            root = Path(staging)
            audio = root / "voice/evals/retained-evidence/input-audio"
            audio.mkdir(parents=True)
            (audio / "room-turn-001.wav").write_bytes(b"RIFF synthetic wav")
            (audio / "room-turn-001.txt").write_text("synthetic transcript\\n")
            (audio / "room-turn-002.wav").write_bytes(b"RIFF second wav")
            (audio / "room-turn-003.txt").write_text("orphan transcript\\n")
            (audio / "room-turn-001.json").write_text("private metadata\\n")
            (audio / "voice.env.json").write_text("VOICE_TOKEN=synthetic-secret\\n")
            outside = root / "outside.wav"
            outside.write_bytes(b"not in input-audio")
            (audio / "room-turn-004.wav").symlink_to(outside)

            subprocess.run(
                ["bash", "-s", "--", str(root)],
                input=_RETAIN_EVIDENCE_SCRIPT,
                text=True,
                check=True,
                capture_output=True,
                env={**os.environ, "SUDO_USER": pwd.getpwuid(os.getuid()).pw_name},
            )

            with tarfile.open(root / "retained-evidence.tar.gz", "r:gz") as retained:
                self.assertEqual(
                    retained.getnames(),
                    [
                        "input-audio/room-turn-001.wav",
                        "input-audio/room-turn-001.txt",
                        "input-audio/room-turn-002.wav",
                    ],
                )

    def test_caller_audio_archive_allowlists_complete_line_groups_only(self):
        with tempfile.TemporaryDirectory() as staging:
            root = Path(staging)
            audio = root / "voice/evals/retained-evidence/caller-audio"
            audio.mkdir(parents=True)
            stem = "room-turn-001"
            (audio / f"{stem}-rendered.pcm").write_bytes(b"exact render")
            (audio / f"{stem}-pushed.pcm").write_bytes(b"actual push")
            (audio / f"{stem}.json").write_text('{"line":"private caller words"}')
            (audio / "orphan-turn-002.json").write_text("orphan")
            (audio / "voice.env.json").write_text("VOICE_TOKEN=synthetic-secret")
            outside = root / "outside.pcm"
            outside.write_bytes(b"outside")
            (audio / "linked-turn-003-rendered.pcm").symlink_to(outside)

            subprocess.run(
                ["bash", "-s", "--", str(root)],
                input=_RETAIN_EVIDENCE_SCRIPT,
                text=True,
                check=True,
                capture_output=True,
                env={**os.environ, "SUDO_USER": pwd.getpwuid(os.getuid()).pw_name},
            )

            with tarfile.open(root / "retained-evidence.tar.gz", "r:gz") as retained:
                self.assertEqual(retained.getnames(), [
                    "caller-audio/room-turn-001-pushed.pcm",
                    "caller-audio/room-turn-001-rendered.pcm",
                    "caller-audio/room-turn-001.json",
                ])

    def test_caller_audio_archive_preserves_complete_attempt_groups_only(self):
        with tempfile.TemporaryDirectory() as staging:
            root = Path(staging)
            audio = root / "voice/evals/retained-evidence/caller-audio"
            audio.mkdir(parents=True)
            complete_stems = (
                "room-turn-001-attempt-01",
                "room-turn-002-attempt-01",
                "room-turn-002-attempt-02",
            )
            for stem in complete_stems:
                (audio / f"{stem}-rendered.pcm").write_bytes(b"render")
                (audio / f"{stem}-pushed.pcm").write_bytes(b"push")
                (audio / f"{stem}.json").write_text("{}")
            (audio / "malformed-turn-003-attempt-1-rendered.pcm").write_bytes(b"bad")
            incomplete = "incomplete-turn-004-attempt-01"
            (audio / f"{incomplete}-rendered.pcm").write_bytes(b"render")
            (audio / f"{incomplete}-pushed.pcm").write_bytes(b"push")

            subprocess.run(
                ["bash", "-s", "--", str(root)],
                input=_RETAIN_EVIDENCE_SCRIPT,
                text=True,
                check=True,
                capture_output=True,
                env={**os.environ, "SUDO_USER": pwd.getpwuid(os.getuid()).pw_name},
            )

            expected = sorted(
                f"caller-audio/{stem}{suffix}"
                for stem in complete_stems
                for suffix in ("-pushed.pcm", "-rendered.pcm", ".json")
            )
            with tarfile.open(root / "retained-evidence.tar.gz", "r:gz") as retained:
                self.assertEqual(retained.getnames(), expected)

    def test_retained_archive_is_private_and_owned_by_the_scp_user(self):
        self.assertIn('os.environ["SUDO_USER"]', _RETAIN_EVIDENCE_SCRIPT)
        self.assertIn(
            'install -d -o nobody -g nogroup -m 700 '
            '"$DEV_DIR/voice/evals/retained-evidence/input-audio"',
            _START_WORKER_SCRIPT,
        )
        self.assertIn('"MENTAT_EVAL_RETAINED_EVIDENCE_DIR"', _RUN_VOICE_SCRIPT)
        with tempfile.TemporaryDirectory() as staging:
            root = Path(staging)
            (root / "agent.log").write_text("synthetic diagnostics\n")
            mode_trace = root / "voice/evals/voice-modes.jsonl"
            mode_trace.parent.mkdir(parents=True)
            mode_trace.write_text('{"room":"room-1","event":"mode"}\\n')
            sms_audio = root / "voice/evals/retained-evidence/sms-audio"
            sms_audio.mkdir(parents=True)
            (sms_audio / "room-turn-001.wav").write_bytes(b"RIFF synthetic wav")
            (sms_audio / "transcripts.jsonl").write_text('{"transcript":"private"}\\n')
            (sms_audio / "voice.env.json").write_text("VOICE_TOKEN=synthetic-secret\\n")
            subprocess.run(
                ["bash", "-s", "--", str(root)],
                input=_RETAIN_EVIDENCE_SCRIPT,
                text=True,
                check=True,
                capture_output=True,
                env={**os.environ, "SUDO_USER": pwd.getpwuid(os.getuid()).pw_name},
            )
            archive = root / "retained-evidence.tar.gz"
            self.assertEqual(archive.stat().st_uid, os.getuid())
            self.assertEqual(stat.S_IMODE(archive.stat().st_mode), 0o600)
            with tarfile.open(archive, "r:gz") as retained:
                self.assertEqual(
                    retained.getnames(),
                    [
                        "agent.log",
                        "voice/evals/voice-modes.jsonl",
                        "sms-audio/room-turn-001.wav",
                        "sms-audio/transcripts.jsonl",
                    ],
                )

    def test_cleanup_retains_private_eval_evidence_without_environment_files(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval.evidence\n", "")
            if args[:2] == ["scp", "-p"]:
                destination = Path(args[-1])
                with tempfile.TemporaryDirectory() as source_dir:
                    source = Path(source_dir)
                    for name, content in (
                        ("agent.log", "agent diagnostics\\n"),
                        ("voice.log", "voice diagnostics\\n"),
                        ("records/session.jsonl", "{}\\n"),
                        ("voice/evals/delegations.jsonl", "{}\\n"),
                        ("voice/evals/voice-modes.jsonl", '{"room":"room-1","event":"mode"}\\n'),
                        ("sms-audio/room-turn-001.wav", "RIFF synthetic wav"),
                        ("sms-audio/transcripts.jsonl", '{"turn":1,"transcript":"private"}\\n'),
                        ("input-audio/room-turn-001.wav", "RIFF input wav"),
                        ("input-audio/room-turn-001.txt", "private caller transcript\\n"),
                        ("caller-audio/room-turn-001-rendered.pcm", "exact render"),
                        ("caller-audio/room-turn-001-pushed.pcm", "actual push"),
                        ("caller-audio/room-turn-001.json", '{"line":"private caller words"}'),
                        ("caller-audio/room-turn-005-attempt-01-rendered.pcm", "render 5-1"),
                        ("caller-audio/room-turn-005-attempt-01-pushed.pcm", "push 5-1"),
                        ("caller-audio/room-turn-005-attempt-01.json", "metadata 5-1"),
                        ("caller-audio/room-turn-006-attempt-01-rendered.pcm", "render 6-1"),
                        ("caller-audio/room-turn-006-attempt-01-pushed.pcm", "push 6-1"),
                        ("caller-audio/room-turn-006-attempt-01.json", "metadata 6-1"),
                        ("caller-audio/room-turn-006-attempt-02-rendered.pcm", "render 6-2"),
                        ("caller-audio/room-turn-006-attempt-02-pushed.pcm", "push 6-2"),
                        ("caller-audio/room-turn-006-attempt-02.json", "metadata 6-2"),
                        ("caller-audio/room-turn-007-attempt-1-rendered.pcm", "malformed"),
                        ("caller-audio/room-turn-008-attempt-01-rendered.pcm", "incomplete render"),
                        ("caller-audio/room-turn-008-attempt-01-pushed.pcm", "incomplete push"),
                        ("caller-audio/orphan-turn-002.json", "orphan metadata"),
                        ("caller-audio/room-turn-004-rendered.pcm", "unpaired audio"),
                        ("input-audio/orphan-turn-002.txt", "orphan transcript\\n"),
                        ("input-audio/room-turn-001.json", "private metadata\\n"),
                        ("voice.env.json", "VOICE_TOKEN=synthetic-secret\\n"),
                    ):
                        path = source / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(content)
                    symlink = source / "input-audio/linked-turn-003.wav"
                    symlink.symlink_to(source / "input-audio/room-turn-001.wav")
                    with tarfile.open(destination, "w:gz") as archive:
                        for name in (
                            "agent.log", "voice.log", "records/session.jsonl",
                            "voice/evals/delegations.jsonl", "voice/evals/voice-modes.jsonl",
                            "sms-audio/room-turn-001.wav",
                            "sms-audio/transcripts.jsonl", "input-audio/room-turn-001.wav",
                            "input-audio/room-turn-001.txt", "input-audio/orphan-turn-002.txt",
                            "input-audio/room-turn-001.json", "input-audio/linked-turn-003.wav",
                            "caller-audio/room-turn-001-rendered.pcm",
                            "caller-audio/room-turn-001-pushed.pcm",
                            "caller-audio/room-turn-001.json",
                            "caller-audio/room-turn-005-attempt-01-rendered.pcm",
                            "caller-audio/room-turn-005-attempt-01-pushed.pcm",
                            "caller-audio/room-turn-005-attempt-01.json",
                            "caller-audio/room-turn-006-attempt-01-rendered.pcm",
                            "caller-audio/room-turn-006-attempt-01-pushed.pcm",
                            "caller-audio/room-turn-006-attempt-01.json",
                            "caller-audio/room-turn-006-attempt-02-rendered.pcm",
                            "caller-audio/room-turn-006-attempt-02-pushed.pcm",
                            "caller-audio/room-turn-006-attempt-02.json",
                            "caller-audio/room-turn-007-attempt-1-rendered.pcm",
                            "caller-audio/room-turn-008-attempt-01-rendered.pcm",
                            "caller-audio/room-turn-008-attempt-01-pushed.pcm",
                            "caller-audio/orphan-turn-002.json",
                            "caller-audio/room-turn-004-rendered.pcm",
                            "voice.env.json",
                        ):
                            archive.add(source / name, arcname=name)
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("voice.evals.dev_stack.subprocess.Popen") as popen:
            popen.return_value = unittest.mock.Mock(poll=lambda: None)
            with isolated_run(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run) as stack:
                pass

        retained = getattr(stack, "retained_evidence_dir", None)
        self.assertIsInstance(retained, Path)
        self.assertTrue(retained.is_dir())
        self.assertEqual(stat.S_IMODE(retained.stat().st_mode), 0o700)
        self.assertEqual(
            {path.relative_to(retained).as_posix() for path in retained.rglob("*") if path.is_file()},
            {
                "agent.log", "voice.log", "records/session.jsonl",
                "voice/evals/delegations.jsonl", "voice/evals/voice-modes.jsonl",
                "sms-audio/room-turn-001.wav",
                "sms-audio/transcripts.jsonl", "input-audio/room-turn-001.wav",
                "input-audio/room-turn-001.txt", "caller-audio/room-turn-001-rendered.pcm",
                "caller-audio/room-turn-001-pushed.pcm", "caller-audio/room-turn-001.json",
                "caller-audio/room-turn-005-attempt-01-rendered.pcm",
                "caller-audio/room-turn-005-attempt-01-pushed.pcm",
                "caller-audio/room-turn-005-attempt-01.json",
                "caller-audio/room-turn-006-attempt-01-rendered.pcm",
                "caller-audio/room-turn-006-attempt-01-pushed.pcm",
                "caller-audio/room-turn-006-attempt-01.json",
                "caller-audio/room-turn-006-attempt-02-rendered.pcm",
                "caller-audio/room-turn-006-attempt-02-pushed.pcm",
                "caller-audio/room-turn-006-attempt-02.json",
            },
        )
        for path in retained.rglob("*"):
            if path.is_dir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
            elif path.is_file():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertFalse(any(".env" in path.name for path in retained.rglob("*")))
        retained_text = "".join(path.read_text() for path in retained.rglob("*") if path.is_file())
        self.assertNotIn("synthetic-secret", retained_text)

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
                self.fail("build failure should prevent entering the batch")

        self.assertFalse(any(args[:2] == ["ssh", "ultraviolet"] for args, _ in calls))

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

        stack = _RunStack(checkout=CHECKOUT, remote="ultraviolet", run=run, batch=SimpleNamespace())
        with self.assertRaisesRegex(subprocess.CalledProcessError, diagnostic) as caught:
            stack._remote("set -euo pipefail", "/tmp/mentat-eval.test")

        for secret in secrets:
            self.assertNotIn(secret, str(caught.exception))

    def test_interruption_before_batch_staging_launches_no_candidate_processes(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            return subprocess.CompletedProcess(args, 0, "", "")

        stack = DevStack(checkout=CHECKOUT, opt_in=True, run=run)
        with patch.object(DevStack, "_start", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                stack.__enter__()

        self.assertEqual(calls, [])

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
            with isolated_run(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run):
                raise ValueError("scenario failed")

        cleanup = next(
            kwargs.get("input", "")
            for args, kwargs in calls
            if args[:2] == ["ssh", "ultraviolet"] and 'cat > "$BATCH_DIR/cleanup.sh"' in kwargs.get("input", "")
        )
        self.assertIn('systemctl start mentat-voice', cleanup)
        self.assertIn('systemctl stop mentat-voice', cleanup)
        self.assertLess(
            cleanup.index('for pid_file in "$BATCH_DIR"/runs/*/agent.pid'),
            cleanup.index('systemctl start mentat-voice'),
        )

    def test_staged_remote_runner_imports_from_the_uploaded_voice_files(self):
        calls = []

        def run(args, **kwargs):
            calls.append((args, kwargs))
            if args[:2] == ["nix", "build"]:
                return subprocess.CompletedProcess(args, 0, "/nix/store/candidate\n", "")
            if args[:2] == ["ssh", "ultraviolet"] and args[2] == "mktemp":
                return subprocess.CompletedProcess(args, 0, "/tmp/mentat-eval-batch.imports\n", "")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch("voice.evals.dev_stack.subprocess.Popen") as popen:
            popen.return_value = unittest.mock.Mock(poll=lambda: None)
            with isolated_run(checkout=CHECKOUT, opt_in=True, dev_port=8485, health_port=8486, run=run):
                pass

        with tempfile.TemporaryDirectory() as temporary:
            remote = Path(temporary)
            voice_root = remote / "shared/voice"
            for args, _ in calls:
                if not args or args[0] != "scp" or args[1:2] == ["-p"]:
                    continue
                remote_target = args[-1].split(":", 1)[1]
                target_is_directory = remote_target.endswith("/")
                target = remote / Path(remote_target).relative_to("/tmp/mentat-eval-batch.imports")
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
