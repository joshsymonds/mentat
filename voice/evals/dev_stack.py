"""Opt-in staging and lifecycle for an isolated remote voice-eval stack."""

from __future__ import annotations

import re
import shlex
import signal
import socket
import subprocess
import threading
from pathlib import Path
from types import FrameType
from typing import Callable, Sequence

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


_MCP_REWRITE_SOURCE = '''def rewrite_mcp_config(raw, production_port, dev_port):
    config = json.loads(raw)

    def rewrite(value):
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            parsed = urlsplit(value)
            try:
                port = parsed.port
            except ValueError:
                return value
            if parsed.hostname in ("127.0.0.1", "localhost", "::1") and port == production_port:
                userinfo = parsed.netloc.rpartition("@")[0] + "@" if "@" in parsed.netloc else ""
                hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
                netloc = f"{userinfo}{hostname}:{dev_port}"
                return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))
        return value

    return json.dumps(rewrite(config), separators=(",", ":"))
'''


_PRIVATE_CREDENTIAL_SOURCE = '''def stage_voice_private(values, dev_dir):
    private_path = values.get("MENTAT_VOICE_PRIVATE")
    if not private_path:
        return
    source = Path(private_path)
    if not source.is_file():
        raise RuntimeError("production voice private credential is unavailable")
    destination = dev_dir / "voice-private"
    destination.write_bytes(source.read_bytes())
    os.chown(destination, pwd.getpwnam("nobody").pw_uid, grp.getgrnam("nogroup").gr_gid)
    os.chmod(destination, 0o400)
    values["MENTAT_VOICE_PRIVATE"] = str(destination)
'''


_SETUP_SCRIPT = r'''set -euo pipefail
DEV_DIR=$1
DEV_PORT=$2
HEALTH_PORT=$3

# Find the running production executables before stopping only the voice worker.
MENTAT_PID=$(systemctl show mentatd --property=MainPID --value)
VOICE_PID=$(systemctl show mentat-voice --property=MainPID --value)
test "$MENTAT_PID" -gt 0
test "$VOICE_PID" -gt 0
NODE_BIN=$(readlink -f "/proc/$MENTAT_PID/exe")
VOICE_PY=$(readlink -f "/proc/$VOICE_PID/exe")

mkdir -p "$DEV_DIR/mentat" "$DEV_DIR/voice/assets" "$DEV_DIR/home/mentat" "$DEV_DIR/home/voice/cache" "$DEV_DIR/records"
umask 077
chown root:root "$DEV_DIR"
chmod 711 "$DEV_DIR" "$DEV_DIR/home"
chmod 700 "$DEV_DIR/home/mentat" "$DEV_DIR/home/voice" "$DEV_DIR/home/voice/cache" "$DEV_DIR/records"
chown -R mentat:mentat "$DEV_DIR/mentat" "$DEV_DIR/home/mentat" "$DEV_DIR/records"
chown -R nobody:nogroup "$DEV_DIR/voice" "$DEV_DIR/home/voice"

# Install recovery before the only production service transition.
cat > "$DEV_DIR/cleanup.sh" <<'CLEANUP'
set +e
trap 'systemctl start mentat-voice' EXIT
for pid_file in "$DEV_DIR/agent.pid" "$DEV_DIR/voice.pid"; do
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
done
systemctl start mentat-voice
rm -rf -- "$DEV_DIR"
CLEANUP
chmod 700 "$DEV_DIR/cleanup.sh"

# Preserve service environments in private files before stopping the worker.
python3 - "$DEV_DIR" "$MENTAT_PID" "$VOICE_PID" <<'PY'
import grp
import json
import os
import pwd
import sys
from pathlib import Path

__PRIVATE_CREDENTIAL_SOURCE__

dev_dir = Path(sys.argv[1])
for name, pid in (("mentat", sys.argv[2]), ("voice", sys.argv[3])):
    values = {}
    for field in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" in field:
            key, value = field.split(b"=", 1)
            values[key.decode()] = value.decode()
    if name == "voice":
        stage_voice_private(values, dev_dir)
    (dev_dir / f"{name}.env.json").write_text(json.dumps(values))
    os.chmod(dev_dir / f"{name}.env.json", 0o600)
PY

# Start only the candidate daemon; the worker waits for the room from the
# token returned by this daemon's voice-token endpoint.
python3 - "$DEV_DIR" "$DEV_PORT" "$NODE_BIN" "$VOICE_PY" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

__MCP_REWRITE_SOURCE__

dev_dir = Path(sys.argv[1])
dev_port = int(sys.argv[2])
node_bin = sys.argv[3]
voice_python = sys.argv[4]
source_env = json.loads((dev_dir / "mentat.env.json").read_text())
production_listen = source_env.get("MENTAT_LISTEN", "127.0.0.1:8484")
production_port = int(production_listen.rsplit(":", 1)[1])

env = {key: value for key, value in source_env.items() if key != "OPENAI_API_KEY"}
if "MENTAT_MCP_CONFIG" in env:
    env["MENTAT_MCP_CONFIG"] = rewrite_mcp_config(env["MENTAT_MCP_CONFIG"], production_port, dev_port)
env.update({
    "MENTAT_LISTEN": f"127.0.0.1:{dev_port}",
    "MENTAT_STATE_PATH": str(dev_dir / "state.json"),
    "MENTAT_RECORD_DIR": str(dev_dir / "records"),
    "HOME": str(dev_dir / "home/mentat"),
})
log = (dev_dir / "agent.log").open("ab", buffering=0)
process = subprocess.Popen(
    ["setpriv", "--reuid=mentat", "--regid=mentat", "--init-groups", node_bin, str(dev_dir / "mentat/src/main.ts")],
    cwd=dev_dir / "mentat", env=env, stdin=subprocess.DEVNULL,
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
)
(dev_dir / "agent.pid").write_text(f"{process.pid}\n")
log.close()
(dev_dir / "voice-python.path").write_text(f"{voice_python}\n")
(dev_dir / "voice-python.path").chmod(0o600)
(dev_dir / "mentat.env.json").unlink()
PY
'''


_START_WORKER_SCRIPT = r'''set -euo pipefail
DEV_DIR=$1
DEV_PORT=$2
HEALTH_PORT=$3
ROOM=$4
trap 'systemctl start mentat-voice' EXIT
systemctl stop mentat-voice
python3 - "$DEV_DIR" "$DEV_PORT" "$HEALTH_PORT" "$ROOM" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

dev_dir = Path(sys.argv[1])
dev_port = int(sys.argv[2])
health_port = int(sys.argv[3])
room = sys.argv[4]
voice_python = (dev_dir / "voice-python.path").read_text().strip()
voice_env = json.loads((dev_dir / "voice.env.json").read_text())
voice_env.update({
    "LIVEKIT_URL": "ws://127.0.0.1:7880",
    "MENTAT_URL": f"http://127.0.0.1:{dev_port}",
    "HOME": str(dev_dir / "home/voice"),
    "XDG_CACHE_HOME": str(dev_dir / "home/voice/cache"),
    "MENTAT_VOICE_HTTP_PORT": str(health_port),
})
voice_log = (dev_dir / "voice.log").open("ab", buffering=0)
voice = subprocess.Popen(
    ["setpriv", "--reuid=nobody", "--regid=nogroup", "--clear-groups", voice_python, str(dev_dir / "voice/agent.py"), "connect", "--room", room],
    cwd=dev_dir / "voice", env=voice_env, stdin=subprocess.DEVNULL,
    stdout=voice_log, stderr=subprocess.STDOUT, start_new_session=True,
)
(dev_dir / "voice.pid").write_text(f"{voice.pid}\n")
voice_log.close()
(dev_dir / "voice.env.json").unlink()
PY
trap - EXIT
'''


class DevStack:
    """Start candidate mentatd and join a token-selected room on ultraviolet.

    Construct with ``opt_in=True`` only from a deliberately invoked live run.
    Offline callers should inject ``run`` and exercise this lifecycle without
    making network calls. Call ``start_worker(room)`` after the dev daemon mints
    a voice token so the worker joins that token's room.
    """

    def __init__(
        self,
        *,
        checkout: Path,
        opt_in: bool = False,
        remote: str = "ultraviolet",
        dev_port: int = 8485,
        health_port: int = 8486,
        local_port: int | None = None,
        run: CommandRunner = subprocess.run,
    ) -> None:
        self.checkout = Path(checkout).resolve()
        self.opt_in = opt_in
        self.remote = remote
        if not 1 <= dev_port <= 65535 or not 1 <= health_port <= 65535 or dev_port == health_port:
            raise ValueError("dev and health ports must be distinct valid TCP ports")
        if local_port is not None and not 0 <= local_port <= 65535:
            raise ValueError("local port must be a valid TCP port")
        self.dev_port = dev_port
        self.health_port = health_port
        self._requested_local_port = local_port
        self._run = run
        self._local_port: int | None = None
        self._remote_dir: str | None = None
        self._tunnel: subprocess.Popen[bytes] | None = None
        self._signal_handlers: dict[int, signal.Handlers] = {}
        self._entered = False
        self._worker_started = False

    @property
    def url(self) -> str:
        if self._local_port is None:
            raise RuntimeError("dev stack is not running")
        return f"http://127.0.0.1:{self._local_port}"

    @property
    def base_url(self) -> str:
        return self.url

    def __enter__(self) -> DevStack:
        if not self.opt_in:
            raise RuntimeError("DevStack requires explicit opt-in for a live run")
        self._worker_started = False
        self._install_signal_handlers()
        try:
            self._start()
            self._entered = True
            return self
        except BaseException:
            try:
                self._cleanup()
            finally:
                self._restore_signal_handlers()
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self._cleanup()
        finally:
            self._restore_signal_handlers()
            self._entered = False
        return False

    def run_remote(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run a remote command and capture output instead of logging it."""
        if not self._entered:
            raise RuntimeError("dev stack is not running")
        if not command or any(not isinstance(arg, str) or "\x00" in arg for arg in command):
            raise ValueError("remote command must contain non-empty safe arguments")
        script = f"cd {shlex.quote(self._remote_dir or '')}\nexec {shlex.join(command)}\n"
        return self._run(
            ["ssh", self.remote, "bash", "-s"],
            input=script,
            check=True,
            capture_output=True,
            text=True,
        )

    def start_worker(self, room: str) -> None:
        """Join the room minted by the dev daemon's voice-token endpoint."""
        if not self._entered:
            raise RuntimeError("dev stack is not running")
        if self._worker_started:
            raise RuntimeError("voice worker has already been started")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", room) is None:
            raise ValueError("room must be a safe LiveKit room name")
        self._remote(
            _START_WORKER_SCRIPT,
            self._remote_dir or "",
            str(self.dev_port),
            str(self.health_port),
            room,
        )
        self._worker_started = True

    def _start(self) -> None:
        built = self._run(
            ["nix", "build", ".#mentatd", "--no-link", "--print-out-paths"],
            cwd=self.checkout,
            check=True,
            capture_output=True,
            text=True,
        )
        store_paths = [line.strip() for line in built.stdout.splitlines() if line.strip()]
        if len(store_paths) != 1:
            raise RuntimeError("candidate package build did not return exactly one store path")
        package = Path(store_paths[0]) / "lib/mentat"
        staged = self._run(
            ["ssh", self.remote, "mktemp", "-d", "/tmp/mentat-eval.XXXXXX"],
            check=True,
            capture_output=True,
            text=True,
        )
        self._remote_dir = staged.stdout.strip()
        if re.fullmatch(r"/tmp/mentat-eval\.[A-Za-z0-9]+", self._remote_dir) is None:
            self._remote_dir = None
            raise RuntimeError("remote staging returned an unsafe directory")

        self._run(
            [
                "ssh", self.remote, "mkdir", "-m", "700", "-p", "--",
                f"{self._remote_dir}/mentat",
                f"{self._remote_dir}/voice/assets",
                f"{self._remote_dir}/voice/evals",
                f"{self._remote_dir}/home/mentat",
                f"{self._remote_dir}/home/voice/cache",
                f"{self._remote_dir}/records",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self._run(
            [
                "scp", "-r",
                str(package / "src"),
                str(package / "node_modules"),
                str(package / "package.json"),
                f"{self.remote}:{self._remote_dir}/mentat/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        voice_files = [
            "agent.py", "persona.md", "request.py", "stream.py", "caller.py",
        ]
        self._run(
            [
                "scp", "-r",
                *(str(self.checkout / "voice" / name) for name in voice_files),
                str(self.checkout / "voice" / "evals" / "phone.py"),
                f"{self.remote}:{self._remote_dir}/voice/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self._run(
            [
                "scp", "-r", str(self.checkout / "voice" / "assets"),
                f"{self.remote}:{self._remote_dir}/voice/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        local_port = self._requested_local_port
        if local_port in (None, 0):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                local_port = sock.getsockname()[1]
        self._local_port = local_port

        setup_script = _SETUP_SCRIPT.replace("__MCP_REWRITE_SOURCE__", _MCP_REWRITE_SOURCE).replace(
            "__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE
        )
        self._remote(
            setup_script,
            self._remote_dir,
            str(self.dev_port),
            str(self.health_port),
        )
        self._tunnel = subprocess.Popen(
            [
                "ssh", "-N", "-L",
                f"127.0.0.1:{self._local_port}:127.0.0.1:{self.dev_port}",
                self.remote,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if self._tunnel.poll() is not None:
            raise RuntimeError("SSH port-forward exited before the dev stack became available")

    def _remote(self, script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["ssh", self.remote, "sudo", "bash", "-s", "--", *args],
            input=script,
            check=True,
            capture_output=True,
            text=True,
        )

    def _cleanup(self) -> None:
        try:
            if self._tunnel is not None:
                self._tunnel.terminate()
                try:
                    self._tunnel.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._tunnel.kill()
                    self._tunnel.wait()
                self._tunnel = None
        finally:
            script = r'''set +e
trap 'systemctl start mentat-voice' EXIT
DEV_DIR=$1
if [ -n "$DEV_DIR" ] && [ -f "$DEV_DIR/cleanup.sh" ]; then
  DEV_DIR="$DEV_DIR" bash "$DEV_DIR/cleanup.sh"
elif [ -n "$DEV_DIR" ]; then
  rm -rf -- "$DEV_DIR"
fi
systemctl start mentat-voice
'''
            try:
                self._remote(script, self._remote_dir or "")
            finally:
                self._remote_dir = None
                self._local_port = None
                self._worker_started = False

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._signal_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._interrupted)

    @staticmethod
    def _interrupted(signum: int, frame: FrameType | None) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is threading.main_thread():
            for signum, handler in self._signal_handlers.items():
                signal.signal(signum, handler)
        self._signal_handlers.clear()
