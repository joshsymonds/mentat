"""Opt-in staging and lifecycle for an isolated remote voice-eval stack."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
import time
import urllib.request
from pathlib import Path
from types import FrameType
from typing import Callable, Sequence

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_READINESS_TIMEOUT_SECONDS = 30.0
_READINESS_POLL_SECONDS = 0.2
_READINESS_REQUEST_TIMEOUT_SECONDS = 2.0
_RESTORE_GUARD_REFRESH_INTERVAL_SECONDS = 15 * 60.0
_RESTORE_GUARD_REFRESH_TIMEOUT_SECONDS = 30.0
_CLEANUP_TIMEOUT_SECONDS = 30.0
_REMOTE_CALL_TIMEOUT_SECONDS = 300.0
_SSH_HANDSHAKE_LIMIT = 9
_SSH_HANDSHAKE_SLOTS = threading.BoundedSemaphore(_SSH_HANDSHAKE_LIMIT)
_SSH_MUX_CLIENT_OPTIONS = ("-o", "ControlMaster=no", "-o", "ProxyCommand=false")
_REMOTE_PORT_PROBE_SCRIPT = r'''# MENTAT_EVAL_PORT_PROBE
set -euo pipefail
python3 - <<'PY'
import json
import socket

sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(2)]
try:
    for sock in sockets:
        sock.bind(("0.0.0.0", 0))
    print(json.dumps({
        "dev_port": sockets[0].getsockname()[1],
        "health_port": sockets[1].getsockname()[1],
    }))
finally:
    for sock in sockets:
        sock.close()
PY
'''


class _SSHTransport:
    """Own one authenticated SSH connection, isolated from user control sockets."""

    def __init__(self, remote: str, *, run: CommandRunner) -> None:
        self.remote = remote
        self._run = run
        self._directory: str | None = None
        self._control_path: str | None = None
        self._connected = False
        self._forwardings: set[str] = set()
        self.events: list[dict[str, object]] = []

    def _record_event(
        self,
        operation: str,
        *,
        exit_code: int | None,
        stderr: str | bytes | None = None,
        error: BaseException | None = None,
    ) -> None:
        event: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "exit_code": exit_code,
        }
        diagnostic = _redact_diagnostics(stderr)
        if diagnostic:
            event["stderr"] = diagnostic
        if error is not None and exit_code is None:
            event["error_type"] = type(error).__name__
        self.events.append(event)

    def connect(self) -> None:
        if self._connected:
            return
        directory = tempfile.mkdtemp(prefix="mentat-ssh-")
        control_path = os.fspath(Path(directory) / "c")
        try:
            with _SSH_HANDSHAKE_SLOTS:
                self._run(
                    [
                        "ssh", "-M", "-N", "-f", "-o", "ControlPersist=no",
                        "-S", control_path, self.remote,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=_CLEANUP_TIMEOUT_SECONDS,
                )
        except BaseException as error:
            self._record_event(
                "ssh_master_start",
                exit_code=error.returncode if isinstance(error, subprocess.CalledProcessError) else None,
                stderr=error.stderr if isinstance(error, subprocess.CalledProcessError) else None,
                error=error,
            )
            shutil.rmtree(directory, ignore_errors=True)
            raise
        self._record_event("ssh_master_start", exit_code=0)
        self._directory = directory
        self._control_path = control_path
        self._connected = True

    def ssh_command(self, *command: str) -> list[str]:
        self.connect()
        return ["ssh", "-S", self._control_path or "", *_SSH_MUX_CLIENT_OPTIONS, self.remote, *command]

    def scp_command(self, *arguments: str) -> list[str]:
        self.connect()
        return [
            "scp", "-o", f"ControlPath={self._control_path}",
            *_SSH_MUX_CLIENT_OPTIONS, *arguments,
        ]

    def _control(
        self, *options: str, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        if not self._connected or self._control_path is None:
            raise RuntimeError("SSH control master is not connected")
        run_options: dict[str, object] = {
            "check": False,
            "capture_output": True,
            "text": True,
        }
        run_options["timeout"] = (
            _CLEANUP_TIMEOUT_SECONDS if timeout is None else timeout
        )
        operation = (
            "ssh_master_check"
            if options[:2] == ("-O", "check")
            else "ssh_master_control"
        )
        try:
            result = self._run(
                ["ssh", "-S", self._control_path, *options, self.remote],
                **run_options,
            )
            if result.returncode != 0:
                error = subprocess.CalledProcessError(
                    result.returncode,
                    result.args,
                    output=result.stdout,
                    stderr=result.stderr,
                )
                if operation == "ssh_master_check":
                    self._record_event(operation, exit_code=result.returncode, stderr=result.stderr)
                raise error
        except BaseException as error:
            if operation == "ssh_master_check" and not (
                isinstance(error, subprocess.CalledProcessError)
                and error.returncode != 0
                and self.events
                and self.events[-1].get("operation") == operation
            ):
                self._record_event(
                    operation,
                    exit_code=error.returncode if isinstance(error, subprocess.CalledProcessError) else None,
                    stderr=error.stderr if isinstance(error, subprocess.CalledProcessError) else None,
                    error=error,
                )
            raise
        if operation == "ssh_master_check":
            self._record_event(operation, exit_code=result.returncode, stderr=result.stderr)
        return result

    def check(self) -> None:
        self._control("-O", "check")

    def forward(self, forwarding: str) -> None:
        self._control(
            "-o", "ExitOnForwardFailure=yes", "-O", "forward", "-L", forwarding
        )
        self._forwardings.add(forwarding)

    def cancel_forward(self, forwarding: str, *, timeout: float | None = None) -> None:
        if forwarding not in self._forwardings:
            return
        self._control("-O", "cancel", "-L", forwarding, timeout=timeout)
        self._forwardings.remove(forwarding)

    def close(self, *, timeout: float | None = None) -> None:
        if self._directory is None:
            return
        directory = self._directory
        control_path = self._control_path
        if control_path is not None:
            run_options: dict[str, object] = {
                "check": False,
                "capture_output": True,
                "text": True,
            }
            if timeout is not None:
                run_options["timeout"] = timeout
            try:
                result = self._run(
                    ["ssh", "-S", control_path, "-O", "exit", self.remote],
                    **run_options,
                )
                if result.returncode != 0:
                    error = subprocess.CalledProcessError(
                        result.returncode,
                        result.args,
                        output=result.stdout,
                        stderr=result.stderr,
                    )
                    self._record_event("ssh_master_close", exit_code=result.returncode, stderr=result.stderr)
                    raise error
            except BaseException as error:
                if not (
                    isinstance(error, subprocess.CalledProcessError)
                    and error.returncode != 0
                    and self.events
                    and self.events[-1].get("operation") == "ssh_master_close"
                ):
                    self._record_event(
                        "ssh_master_close",
                        exit_code=error.returncode if isinstance(error, subprocess.CalledProcessError) else None,
                        stderr=error.stderr if isinstance(error, subprocess.CalledProcessError) else None,
                        error=error,
                    )
                raise
            self._record_event("ssh_master_close", exit_code=result.returncode, stderr=result.stderr)
        self._directory = None
        self._control_path = None
        self._connected = False
        self._forwardings.clear()
        shutil.rmtree(directory, ignore_errors=True)


class RemoteCommandError(subprocess.CalledProcessError):
    """A remote command failure whose diagnostic includes its operation and stderr."""

    def __init__(
        self,
        returncode: int,
        cmd: object,
        *,
        output: str | bytes | None = None,
        stderr: str | bytes | None = None,
        operation: str = "remote script",
    ) -> None:
        super().__init__(returncode, cmd, output=output, stderr=stderr)
        self.operation = operation

    def __str__(self) -> str:
        message = f"Remote operation {self.operation!r}: {super().__str__()}"
        stderr = self.stderr.strip() if isinstance(self.stderr, str) else ""
        return f"{message}: {stderr}" if stderr else message


_SECRET_NAME = re.compile(r"(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"""(?ix)
    (?P<prefix>
        (?:\bexport\s+)?
        [\"']?
        [A-Z0-9_.-]*(?:KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)[A-Z0-9_.-]*
        [\"']?
        \s*(?:=|:)\s*(?:Bearer\s+)?
    )
    (?:\"(?:\\\\.|[^\"\\\\])*\"|'(?:\\\\.|[^'\\\\])*'|[^\s,;}\]]+)
    """
)
_AUTHORIZATION_BEARER = re.compile(
    r"(?i)(\bAuthorization\s*:\s*Bearer\s+)[^\s,;}\]]+"
)
_BEARER_VALUE = re.compile(r"(?i)(\bBearer\s+)[A-Za-z0-9._~+/=-]+")


def _secret_environment_values(values: dict[str, object]) -> list[str]:
    return [
        value
        for name, value in values.items()
        if _SECRET_NAME.search(name) and isinstance(value, str) and value
    ]


def _redact_diagnostics(value: str | bytes | None, secrets: Sequence[str] = ()) -> str:
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = value if isinstance(value, str) else ""
    for secret in sorted((secret for secret in secrets if secret), key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = _SECRET_ASSIGNMENT.sub(lambda match: match.group("prefix") + "[REDACTED]", text)
    text = _AUTHORIZATION_BEARER.sub(r"\1[REDACTED]", text)
    return _BEARER_VALUE.sub(r"\1[REDACTED]", text)


def _redact_machine_json(value: str, secrets: Sequence[str] = ()) -> str:
    """Redact JSON string values and serialize a valid machine-channel result."""
    def redact(item: object) -> object:
        if isinstance(item, dict):
            return {key: redact(child) for key, child in item.items()}
        if isinstance(item, list):
            return [redact(child) for child in item]
        if isinstance(item, str):
            return _redact_diagnostics(item, secrets)
        return item

    parsed = json.loads(value)
    return json.dumps(redact(parsed), separators=(",", ":"), allow_nan=False) + "\n"


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


def stage_gateway_key(values, dev_dir):
    key_path = values.get("MENTAT_VOICE_GATEWAY_KEY_FILE")
    if not key_path:
        return
    source = Path(key_path)
    if not source.is_file():
        raise RuntimeError("production voice gateway credential is unavailable")
    destination = dev_dir / "voice-gateway-key"
    destination.write_bytes(source.read_bytes())
    os.chown(destination, pwd.getpwnam("mentat").pw_uid, grp.getgrnam("mentat").gr_gid)
    os.chmod(destination, 0o400)
    values["MENTAT_VOICE_GATEWAY_KEY_FILE"] = str(destination)
'''


_BATCH_CLEANUP_SCRIPT = r'''set -euo pipefail
umask 077
BATCH_DIR=$1
RESTORE_ACTION=$(cat "$BATCH_DIR/restore-action")
RESTORE_UNIT=$(cat "$BATCH_DIR/restore-unit")
for run_dir in "$BATCH_DIR"/runs/*; do
  [ -d "$run_dir" ] || continue
  touch "$run_dir/shutdown"
  if [ -f "$run_dir/launch.lock" ]; then
    flock -x "$run_dir/launch.lock" -c ':'
  fi
done
FAILED=0
for pid_file in "$BATCH_DIR"/runs/*/agent.pid "$BATCH_DIR"/runs/*/voice.pid "$BATCH_DIR"/runs/*/caller.pid; do
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    case "$pid" in ''|*[!0-9]*|0) echo "invalid candidate process id in $pid_file" >&2; FAILED=1; continue ;; esac
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do kill -0 -- "-$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 -- "-$pid" 2>/dev/null; then
      echo "candidate process group -$pid remains alive after KILL" >&2
      FAILED=1
    fi
  fi
done
if [ "$FAILED" -ne 0 ]; then exit 1; fi
if [ -n "$RESTORE_UNIT" ]; then systemctl stop "$RESTORE_UNIT.timer" "$RESTORE_UNIT.service" >/dev/null 2>&1 || true; fi
if [ "$RESTORE_ACTION" = start ]; then systemctl start mentat-voice; elif [ "$RESTORE_ACTION" = stop ]; then systemctl stop mentat-voice; else echo "invalid production restore action" >&2; exit 1; fi
rm -f -- "$BATCH_DIR/shared/voice-env-root"
rm -rf -- "$BATCH_DIR"
'''

_REAP_BATCHES_SCRIPT = r'''set -euo pipefail
python3 - <<'PY'
import json
import os
import stat
from pathlib import Path

def trusted(path, kind, mode):
    try:
        info = path.lstat()
    except OSError:
        return False
    file_type = stat.S_ISDIR(info.st_mode) if kind == "dir" else stat.S_ISREG(info.st_mode)
    return file_type and info.st_uid == 0 and stat.S_IMODE(info.st_mode) == mode

def trusted_run_tree(runs):
    for run_dir in runs.iterdir():
        if not trusted(run_dir, "dir", 0o711):
            return False
        for name in ("launch.lock", "shutdown", "agent.pid", "voice.pid", "caller.pid"):
            path = run_dir / name
            if path.exists() or path.is_symlink():
                if not trusted(path, "file", 0o600):
                    return False
                if name.endswith(".pid"):
                    try:
                        if int(path.read_text()) <= 0:
                            return False
                    except (OSError, ValueError):
                        return False
    return True

def read_owner(path):
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or stat.S_IMODE(info.st_mode) != 0o600:
            return None
        with os.fdopen(descriptor, "r", closefd=False) as source:
            return source.read()
    finally:
        os.close(descriptor)

batches = []
for path in sorted(Path("/tmp").glob("mentat-eval-batch.*")):
    if not trusted(path, "dir", 0o711):
        continue
    if not trusted(path / "runs", "dir", 0o711):
        continue
    if not trusted_run_tree(path / "runs"):
        continue
    if not trusted(path / "cleanup.sh", "file", 0o700):
        continue
    if not trusted(path / "restore-action", "file", 0o600):
        continue
    if not trusted(path / "restore-unit", "file", 0o600):
        continue
    try:
        owner = read_owner(path / "owner.json")
    except OSError:
        owner = None
    batches.append({"path": str(path), "owner": owner})
print(json.dumps(batches))
PY
'''

_REAP_DEAD_BATCH_SCRIPT = r'''set -euo pipefail
BATCH_DIR=$1
EXPECTED_HOST=$2
EXPECTED_PID=$3
if ! python3 - "$BATCH_DIR" "$EXPECTED_HOST" "$EXPECTED_PID" <<'PY'
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
def trusted(path, kind, mode):
    try:
        info = path.lstat()
    except OSError:
        return False
    file_type = stat.S_ISDIR(info.st_mode) if kind == "dir" else stat.S_ISREG(info.st_mode)
    return file_type and info.st_uid == 0 and stat.S_IMODE(info.st_mode) == mode

def trusted_run_tree(runs):
    for run_dir in runs.iterdir():
        if not trusted(run_dir, "dir", 0o711): return False
        for name in ("launch.lock", "shutdown", "agent.pid", "voice.pid", "caller.pid"):
            path = run_dir / name
            if path.exists() or path.is_symlink():
                if not trusted(path, "file", 0o600): return False
                if name.endswith(".pid"):
                    try:
                        if int(path.read_text()) <= 0: return False
                    except (OSError, ValueError):
                        return False
    return True

if not trusted(root, "dir", 0o711): raise SystemExit(1)
for name, kind, mode in (
    ("runs", "dir", 0o711),
    ("owner.json", "file", 0o600),
    ("cleanup.sh", "file", 0o700),
    ("restore-action", "file", 0o600),
    ("restore-unit", "file", 0o600),
):
    if not trusted(root / name, kind, mode): raise SystemExit(1)
if not trusted_run_tree(root / "runs"): raise SystemExit(1)
try:
    owner = json.loads((root / "owner.json").read_text())
except (OSError, ValueError, TypeError):
    raise SystemExit(1)
if owner.get("host") != sys.argv[2] or owner.get("pid") != int(sys.argv[3]):
    raise SystemExit(1)
PY
then
  echo "preserving batch with uncertain ownership: $BATCH_DIR" >&2
  exit 0
fi
'''
_BATCH_SETUP_SCRIPT = r'''set -euo pipefail
BATCH_DIR=$1
OWNER_HOST=$2
OWNER_PID=$3
umask 077
test -d "$BATCH_DIR"
test ! -L "$BATCH_DIR"
chown root:root "$BATCH_DIR"
chmod 711 "$BATCH_DIR"
for name in owner.json cleanup.sh restore-action restore-unit; do
  rm -f -- "$BATCH_DIR/$name"
done
if [ -L "$BATCH_DIR/runs" ] || { [ -e "$BATCH_DIR/runs" ] && [ ! -d "$BATCH_DIR/runs" ]; }; then
  rm -f -- "$BATCH_DIR/runs"
fi
if [ -d "$BATCH_DIR/runs" ]; then rmdir "$BATCH_DIR/runs"; fi
install -d -m 700 "$BATCH_DIR/runs"
chown root:root "$BATCH_DIR/runs"
chmod 711 "$BATCH_DIR/runs"
SHARED_DIR="$BATCH_DIR/shared"
chown -R root:root "$SHARED_DIR"
python3 - "$BATCH_DIR" "$OWNER_HOST" "$OWNER_PID" <<'PY'
import json
import os
import sys
from pathlib import Path
owner = {"host": sys.argv[2], "pid": int(sys.argv[3])}
path = Path(sys.argv[1]) / "owner.json"
path.write_text(json.dumps(owner, separators=(",", ":")) + "\n")
os.chmod(path, 0o600)
PY
MENTAT_PID=$(systemctl show mentatd --property=MainPID --value)
VOICE_PID=$(systemctl show mentat-voice --property=MainPID --value)
test "$MENTAT_PID" -gt 0
test "$VOICE_PID" -gt 0
NODE_BIN=$(readlink -f "/proc/$MENTAT_PID/exe")
VOICE_ENV_PATH=$(nix build --impure --expr "let pkgs = import (builtins.getFlake \"nixpkgs\").outPath {}; in import $SHARED_DIR/voice/voice-env.nix { inherit pkgs; }" --out-link "$SHARED_DIR/voice-env-root" --print-out-paths)
case "$VOICE_ENV_PATH" in /nix/store/*) ;; *) echo "candidate voice environment build did not return one Nix store path" >&2; exit 1 ;; esac
VOICE_PY="$VOICE_ENV_PATH/bin/python"
test -x "$VOICE_PY"
"$VOICE_PY" - <<'PY'
import importlib

for module in (
    "aiohttp", "livekit.api", "livekit.rtc", "livekit.plugins.dtln",
    "livekit.plugins.elevenlabs", "livekit.plugins.silero",
    "livekit.plugins.turn_detector",
):
    try:
        importlib.import_module(module)
    except ImportError as error:
        raise SystemExit(f"candidate voice environment missing required module {module}: {error}") from error
PY
if systemctl is-active --quiet mentat-voice; then RESTORE_ACTION=start; else RESTORE_ACTION=stop; fi
RESTORE_UNIT="mentat-eval-restore-${BATCH_DIR##*.}"
printf '%s\n' "$RESTORE_ACTION" > "$BATCH_DIR/restore-action"
printf '%s\n' "$RESTORE_UNIT" > "$BATCH_DIR/restore-unit"
cat > "$BATCH_DIR/cleanup.sh" <<'CLEANUP'
__BATCH_CLEANUP_SCRIPT__
CLEANUP
chmod 700 "$BATCH_DIR/cleanup.sh"
systemd-run --quiet --unit="$RESTORE_UNIT" --on-active=30m "$(command -v systemctl)" "$RESTORE_ACTION" mentat-voice
python3 - "$BATCH_DIR" "$MENTAT_PID" "$VOICE_PID" "$NODE_BIN" "$VOICE_PY" <<'PY'
import grp
import json
import os
import pwd
import shutil
import sys
from pathlib import Path

__PRIVATE_CREDENTIAL_SOURCE__
batch_dir = Path(sys.argv[1])
shared = batch_dir / "shared"
setpriv_path = shutil.which("setpriv")
if setpriv_path is None: raise RuntimeError("setpriv executable is unavailable in the setup environment")
for name, pid in (("mentat", sys.argv[2]), ("voice", sys.argv[3])):
    values = {}
    for field in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        if b"=" in field:
            key, value = field.split(b"=", 1)
            values[key.decode()] = value.decode()
    if name == "voice": stage_voice_private(values, shared)
    else: stage_gateway_key(values, shared)
    (shared / f"{name}.env.json").write_text(json.dumps(values))
    os.chmod(shared / f"{name}.env.json", 0o600)
for name, value in (("setpriv.path", str(Path(setpriv_path).resolve())), ("node.path", sys.argv[4]), ("voice-python.path", sys.argv[5])):
    destination = shared / name
    destination.write_text(value)
    destination.chmod(0o600)
PY
'''

_RUN_SETUP_SCRIPT = r'''set -euo pipefail
BATCH_DIR=$1
RUN_DIR=$2
DEV_PORT=$3
HEALTH_PORT=$4
VOICE_MODEL=$5
umask 077
SHARED_DIR="$BATCH_DIR/shared"
mkdir -m 700 "$RUN_DIR"
cp -a "$SHARED_DIR/mentat" "$RUN_DIR/mentat"
cp -a "$SHARED_DIR/voice" "$RUN_DIR/voice"
for name in mentat.env.json voice.env.json setpriv.path node.path voice-python.path; do cp "$SHARED_DIR/$name" "$RUN_DIR/$name"; done
cp "$BATCH_DIR/restore-unit" "$RUN_DIR/restore-unit"
if [ -f "$SHARED_DIR/voice-private" ]; then
  cp "$SHARED_DIR/voice-private" "$RUN_DIR/voice-private"
  chown nobody:nogroup "$RUN_DIR/voice-private"
  chmod 400 "$RUN_DIR/voice-private"
fi
if [ -f "$SHARED_DIR/voice-gateway-key" ]; then
  cp "$SHARED_DIR/voice-gateway-key" "$RUN_DIR/voice-gateway-key"
  chown mentat:mentat "$RUN_DIR/voice-gateway-key"
  chmod 400 "$RUN_DIR/voice-gateway-key"
fi
mkdir -m 700 -p "$RUN_DIR/home/mentat" "$RUN_DIR/home/voice/cache" "$RUN_DIR/records" "$RUN_DIR/memory" "$RUN_DIR/voice/evals"
: > "$RUN_DIR/launch.lock"
chmod 600 "$RUN_DIR/launch.lock"
chmod 711 "$RUN_DIR/home"
chown root:nogroup "$RUN_DIR/voice.env.json"
chmod 640 "$RUN_DIR/voice.env.json"
chmod 711 "$RUN_DIR"
chown -R mentat:mentat "$RUN_DIR/mentat" "$RUN_DIR/home/mentat" "$RUN_DIR/records" "$RUN_DIR/memory"
chown -R nobody:nogroup "$RUN_DIR/voice" "$RUN_DIR/home/voice"
python3 - "$RUN_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
for env_name, field in (("mentat.env.json", "MENTAT_VOICE_GATEWAY_KEY_FILE"), ("voice.env.json", "MENTAT_VOICE_PRIVATE")):
    path = run_dir / env_name
    values = json.loads(path.read_text())
    source = values.get(field)
    if source:
        destination = run_dir / Path(source).name
        if not destination.is_file():
            raise RuntimeError("candidate voice credential is unavailable")
        values[field] = str(destination)
        path.write_text(json.dumps(values))
        os.chmod(path, 0o600 if env_name == "mentat.env.json" else 0o640)
PY
python3 - "$RUN_DIR" "$DEV_PORT" "$HEALTH_PORT" "$VOICE_MODEL" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

__MCP_REWRITE_SOURCE__
run_dir = Path(sys.argv[1])
dev_port = int(sys.argv[2])
voice_model = sys.argv[4]
node_bin = (run_dir / "node.path").read_text().strip()
setpriv_path = (run_dir / "setpriv.path").read_text().strip()
source_env = json.loads((run_dir / "mentat.env.json").read_text())
production_port = int(source_env.get("MENTAT_LISTEN", "127.0.0.1:8484").rsplit(":", 1)[1])
env = {key: value for key, value in source_env.items() if key != "OPENAI_API_KEY"}
prompt = run_dir / "mentat/prompt.md"
if not prompt.is_file(): raise RuntimeError("candidate system prompt is unavailable")
env["MENTAT_SYSTEM_PROMPT"] = prompt.read_text(encoding="utf-8")
env["MENTAT_VOICE_MODEL"] = voice_model
env["MENTAT_SESSION_TTL"] = "90s"
if "MENTAT_MCP_CONFIG" in env: env["MENTAT_MCP_CONFIG"] = rewrite_mcp_config(env["MENTAT_MCP_CONFIG"], production_port, dev_port)
env.update({
    "MENTAT_LISTEN": f"127.0.0.1:{dev_port}",
    "MENTAT_STATE_PATH": str(run_dir / "home/mentat/state.json"),
    "MENTAT_RECORD_DIR": str(run_dir / "records"),
    "MENTAT_MEMORY_DIR": str(run_dir / "memory"),
    "HOME": str(run_dir / "home/mentat"),
})
log = (run_dir / "agent.log").open("ab", buffering=0)
process = subprocess.Popen(
    [setpriv_path, "--reuid=mentat", "--regid=mentat", "--init-groups", node_bin, str(run_dir / "mentat/src/main.ts")],
    cwd=run_dir / "mentat", env=env, stdin=subprocess.DEVNULL,
    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
)
(run_dir / "agent.pid").write_text(f"{process.pid}\n")
log.close()
PY
'''

_STOP_RUN_SCRIPT = r'''set -euo pipefail
RUN_DIR=$1
umask 077
touch "$RUN_DIR/shutdown"
if [ -f "$RUN_DIR/launch.lock" ]; then flock -x "$RUN_DIR/launch.lock" -c ':'; fi
for pid_file in "$RUN_DIR/agent.pid" "$RUN_DIR/voice.pid" "$RUN_DIR/caller.pid"; do
  if [ -f "$pid_file" ]; then
    pid=$(cat "$pid_file")
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do kill -0 -- "-$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 -- "-$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || true
      for _ in 1 2 3 4 5; do kill -0 -- "-$pid" 2>/dev/null || break; sleep 1; done
      if kill -0 -- "-$pid" 2>/dev/null; then
        printf 'candidate producer group -%s remains alive after KILL\n' "$pid" >&2
        exit 1
      fi
    fi
  fi
done
'''


_RETAIN_EVIDENCE_SCRIPT = r'''set -euo pipefail
DEV_DIR=$1
python3 - "$DEV_DIR" <<'PY'
import os
import pwd
import re
import sys
import tarfile
from pathlib import Path

root = Path(sys.argv[1])
archive = root / "retained-evidence.tar.gz"
with tarfile.open(archive, "w:gz") as output:
    for relative in (
        "agent.log",
        "voice.log",
        "voice/evals/delegations.jsonl",
        "voice/evals/voice-modes.jsonl",
    ):
        source = root / relative
        if source.is_file() and not source.is_symlink():
            output.add(source, arcname=relative, recursive=False)
    records = root / "records"
    if records.is_dir() and not records.is_symlink():
        for source in sorted(records.glob("*.jsonl")):
            if source.is_file() and not source.is_symlink():
                output.add(source, arcname=f"records/{source.name}", recursive=False)
    sms_audio = root / "voice/evals/retained-evidence/sms-audio"
    if sms_audio.is_dir() and not sms_audio.is_symlink():
        for source in sorted(sms_audio.iterdir()):
            allowed = source.name == "transcripts.jsonl" or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}-turn-[0-9]{3}\.wav",
                source.name,
            ) is not None
            if allowed and source.is_file() and not source.is_symlink():
                output.add(source, arcname=f"sms-audio/{source.name}", recursive=False)
    input_audio = root / "voice/evals/retained-evidence/input-audio"
    if input_audio.is_dir() and not input_audio.is_symlink():
        sources = {
            source.name: source
            for source in input_audio.iterdir()
            if source.is_file()
            and not source.is_symlink()
            and re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}-turn-[0-9]{3}\.wav",
                source.name,
            ) is not None
        }
        for wav_name in sorted(sources):
            source = sources[wav_name]
            output.add(source, arcname=f"input-audio/{wav_name}", recursive=False)
            transcript_name = f"{wav_name[:-4]}.txt"
            transcript = input_audio / transcript_name
            if transcript.is_file() and not transcript.is_symlink():
                output.add(
                    transcript,
                    arcname=f"input-audio/{transcript_name}",
                    recursive=False,
                )
    caller_audio = root / "voice/evals/retained-evidence/caller-audio"
    if caller_audio.is_dir() and not caller_audio.is_symlink():
        groups = {}
        for source in caller_audio.iterdir():
            match = re.fullmatch(
                r"([A-Za-z0-9][A-Za-z0-9_.-]{0,127}-turn-[0-9]{3}(?:-attempt-[0-9]{2})?)"
                r"(?:-(?:rendered|pushed)\.pcm|\.json)",
                source.name,
            )
            if match is not None and source.is_file() and not source.is_symlink():
                groups.setdefault(match.group(1), set()).add(source.name)
        for stem, names in sorted(groups.items()):
            expected = {
                f"{stem}-rendered.pcm",
                f"{stem}-pushed.pcm",
                f"{stem}.json",
            }
            if names == expected:
                for name in sorted(expected):
                    output.add(
                        caller_audio / name,
                        arcname=f"caller-audio/{name}",
                        recursive=False,
                    )
os.chmod(archive, 0o600)
owner = pwd.getpwnam(os.environ["SUDO_USER"])
os.chown(archive, owner.pw_uid, owner.pw_gid)
PY
'''


_START_WORKER_SCRIPT = r'''set -euo pipefail
umask 077
DEV_DIR=$1
DEV_PORT=$2
HEALTH_PORT=$3
ROOM=$4
exec 9>"$DEV_DIR/launch.lock"
flock -s 9
if [ -e "$DEV_DIR/shutdown" ]; then exit 1; fi
systemctl stop mentat-voice
if [ -f "$DEV_DIR/voice.pid" ]; then
  previous_pid=$(cat "$DEV_DIR/voice.pid")
  kill -TERM -- "-$previous_pid" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    kill -0 "$previous_pid" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$previous_pid" 2>/dev/null; then
    kill -KILL -- "-$previous_pid" 2>/dev/null || true
  fi
  rm -f -- "$DEV_DIR/voice.pid"
fi
if [ ! -e "$DEV_DIR/voice/evals/voice-modes.jsonl" ]; then
  : > "$DEV_DIR/voice/evals/voice-modes.jsonl"
  chown nobody:nogroup "$DEV_DIR/voice/evals/voice-modes.jsonl"
  chmod 600 "$DEV_DIR/voice/evals/voice-modes.jsonl"
fi
: > "$DEV_DIR/voice/evals/delegations.jsonl"
chown nobody:nogroup "$DEV_DIR/voice/evals/delegations.jsonl"
chmod 600 "$DEV_DIR/voice/evals/delegations.jsonl"
install -d -o nobody -g nogroup -m 700 "$DEV_DIR/voice/evals/retained-evidence"
install -d -o nobody -g nogroup -m 700 "$DEV_DIR/voice/evals/retained-evidence/input-audio"
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
setpriv_path = (dev_dir / "setpriv.path").read_text().strip()
voice_env.update({
    "LIVEKIT_URL": "ws://127.0.0.1:7880",
    "MENTAT_URL": f"http://127.0.0.1:{dev_port}",
    "HOME": str(dev_dir / "home/voice"),
    "XDG_CACHE_HOME": str(dev_dir / "home/voice/cache"),
    "MENTAT_VOICE_HTTP_PORT": str(health_port),
    "MENTAT_EVAL_DELEGATION_LOG": str(dev_dir / "voice/evals/delegations.jsonl"),
    "MENTAT_EVAL_VOICE_LOG": str(dev_dir / "voice/evals/voice-modes.jsonl"),
    "MENTAT_VOICE_INPUT_RECORD_DIR": str(
        dev_dir / "voice/evals/retained-evidence/input-audio"
    ),
})
voice_log = (dev_dir / "voice.log").open("ab", buffering=0)
voice = subprocess.Popen(
    [setpriv_path, "--reuid=nobody", "--regid=nogroup", "--clear-groups", voice_python, str(dev_dir / "voice/agent.py"), "connect", "--room", room],
    cwd=dev_dir / "voice", env=voice_env, stdin=subprocess.DEVNULL,
    stdout=voice_log, stderr=subprocess.STDOUT, start_new_session=True,
)
(dev_dir / "voice.pid").write_text(f"{voice.pid}\n")
voice_log.close()
PY
trap - EXIT
'''


_REFRESH_RESTORE_GUARD_SCRIPT = r'''set -euo pipefail
DEV_DIR=$1
RESTORE_UNIT=$(cat "$DEV_DIR/restore-unit")
test -n "$RESTORE_UNIT"
systemctl restart "$RESTORE_UNIT.timer"
'''


_RUN_VOICE_SCRIPT = r'''import atexit
import json
import os
import subprocess
import sys
from pathlib import Path

os.umask(0o077)
if len(sys.argv) != 5 or sys.argv[1] != "--":
    raise RuntimeError("remote caller received invalid staging arguments")
DEV_DIR = Path(sys.argv[2])
DEV_PORT = sys.argv[3]
HEALTH_PORT = sys.argv[4]
import fcntl
caller_pid_file = DEV_DIR / "caller.pid"
atexit.register(lambda: caller_pid_file.unlink(missing_ok=True))
launch_lock = (DEV_DIR / "launch.lock").open("rb")
fcntl.flock(launch_lock, fcntl.LOCK_SH)
if (DEV_DIR / "shutdown").exists():
    raise RuntimeError("candidate batch is shutting down")
(DEV_DIR / "caller.pid").write_text(f"{os.getpid()}\n")
sys.path.insert(0, str(DEV_DIR / "voice"))
from evals.dev_stack import (
    _redact_diagnostics,
    _redact_machine_json,
    _secret_environment_values,
)

payload = json.load(sys.stdin)
command = payload.get("command")
token = payload.get("token")
livekit_url = payload.get("livekit_url")
if (
    not isinstance(command, list)
    or not command
    or any(not isinstance(arg, str) or "\x00" in arg for arg in command)
    or not isinstance(token, str)
    or not token
    or not isinstance(livekit_url, str)
    or not livekit_url
):
    raise RuntimeError("remote caller received an incomplete voice grant")
voice_python = (DEV_DIR / "voice-python.path").read_text().strip()
setpriv_path = (DEV_DIR / "setpriv.path").read_text().strip()
voice_env = json.loads((DEV_DIR / "voice.env.json").read_text())
voice_env.update({
    "LIVEKIT_URL": livekit_url,
    "MENTAT_VOICE_TOKEN": token,
    "MENTAT_URL": f"http://127.0.0.1:{DEV_PORT}",
    "HOME": str(DEV_DIR / "home/voice"),
    "XDG_CACHE_HOME": str(DEV_DIR / "home/voice/cache"),
    "MENTAT_VOICE_HTTP_PORT": HEALTH_PORT,
    "MENTAT_EVAL_RETAINED_EVIDENCE_DIR": str(
        DEV_DIR / "voice/evals/retained-evidence"
    ),
})
secret_values = _secret_environment_values(voice_env) + [token]
fcntl.flock(launch_lock, fcntl.LOCK_UN)
launch_lock.close()
result = subprocess.run(
    [
        setpriv_path, "--reuid=nobody", "--regid=nogroup", "--clear-groups",
        voice_python, *command,
    ],
    cwd=DEV_DIR / "voice",
    env=voice_env,
    stdin=subprocess.DEVNULL,
    capture_output=True, text=True, check=False,
)
sys.stderr.write(_redact_diagnostics(result.stderr, secret_values))
try:
    machine_output = _redact_machine_json(result.stdout, secret_values)
except (ValueError, TypeError):
    diagnostic = _redact_diagnostics(result.stdout, secret_values)
    if diagnostic:
        sys.stderr.write(diagnostic)
        if not diagnostic.endswith("\n"):
            sys.stderr.write("\n")
    sys.stderr.write("voice caller stdout was not valid JSON\n")
    sys.exit(result.returncode or 1)
sys.stdout.write(machine_output)
sys.exit(result.returncode)
'''


class _RunStack:
    """A single candidate run inside a staged DevStack batch."""

    def __init__(
        self,
        *,
        checkout: Path,
        remote: str = "ultraviolet",
        dev_port: int | None = None,
        health_port: int | None = None,
        local_port: int | None = None,
        run: CommandRunner = subprocess.run,
        batch: DevStack | None = None,
        run_id: str | None = None,
    ) -> None:
        self.checkout = Path(checkout).resolve()
        self.remote = remote
        self._batch = batch
        self.run_id = run_id
        if (
            (dev_port is not None and not 1 <= dev_port <= 65535)
            or (health_port is not None and not 1 <= health_port <= 65535)
            or (dev_port is not None and dev_port == health_port)
        ):
            raise ValueError("dev and health ports must be distinct valid TCP ports")
        if local_port is not None and not 0 <= local_port <= 65535:
            raise ValueError("local port must be a valid TCP port")
        self.dev_port = dev_port
        self.health_port = health_port
        self._requested_local_port = local_port
        self._run = run
        self._local_port: int | None = None
        self._remote_dir: str | None = None
        self._transport = _SSHTransport(remote, run=run)
        self._forward_spec: str | None = None
        self._entered = False
        self._restore_guard_armed = False
        self._producers_stopped = False
        self._cleanup_lock = threading.Lock()
        self._cleanup_complete = False
        self._cleanup_error: BaseException | None = None
        self._call_events: list[dict[str, object]] = []
        self._diagnostics_lock = threading.Lock()
        self._diagnostics_finalized = False
        self.retained_evidence_dir: Path | None = None

    def _record_call(
        self,
        operation: str,
        *,
        exit_code: int | None,
        stderr: str | bytes | None = None,
        error: BaseException | None = None,
        secrets: Sequence[str] = (),
    ) -> None:
        event: dict[str, object] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "exit_code": exit_code,
        }
        diagnostic = _redact_diagnostics(stderr, secrets)
        if diagnostic:
            event["stderr"] = diagnostic
        if error is not None and exit_code is None:
            event["error_type"] = type(error).__name__
        with self._diagnostics_lock:
            self._call_events.append(event)
            finalized = self._diagnostics_finalized
        if finalized:
            try:
                self._write_transport_diagnostics()
            except OSError as write_error:
                if error is None:
                    raise
                error.add_note(f"Transport diagnostics write also failed: {write_error}")

    def _write_transport_diagnostics(self) -> None:
        with self._diagnostics_lock:
            evidence_dir = self.retained_evidence_dir
            if evidence_dir is None or not evidence_dir.is_dir():
                return
            self._diagnostics_finalized = True
            events = sorted(
                (*self._transport.events, *self._call_events),
                key=lambda event: str(event["timestamp"]),
            )
            destination = evidence_dir / "transport-diagnostics.json"
            destination.write_text(json.dumps({"schema_version": 1, "events": events}, indent=2) + "\n")
            os.chmod(destination, 0o600)

    @property
    def url(self) -> str:
        if self._local_port is None:
            raise RuntimeError("dev stack is not running")
        return f"http://127.0.0.1:{self._local_port}"

    @property
    def base_url(self) -> str:
        return self.url

    def __enter__(self) -> _RunStack:
        if self._batch is None:
            raise RuntimeError("run stacks must be created by DevStack.run")
        try:
            self._batch._begin_launch()
            try:
                self._start()
                self._entered = True
            finally:
                self._batch._end_launch()
            return self
        except BaseException as error:
            try:
                self._cleanup()
            except BaseException as cleanup_error:
                error.add_note(f"candidate run cleanup also failed: {cleanup_error}")
            raise

    def __exit__(self, exc_type, exc, traceback) -> bool:
        try:
            self._cleanup()
        finally:
            self._entered = False
        return False

    def run_remote(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        """Run a remote command and capture output instead of logging it."""
        if not self._entered:
            raise RuntimeError("dev stack is not running")
        if not command or any(not isinstance(arg, str) or "\x00" in arg for arg in command):
            raise ValueError("remote command must contain non-empty safe arguments")
        script = f"cd {shlex.quote(self._remote_dir or '')}\nexec {shlex.join(command)}\n"
        try:
            result = self._run(
                self._transport.ssh_command("bash", "-s"),
                input=script,
                check=True,
                capture_output=True,
                text=True,
                timeout=_REMOTE_CALL_TIMEOUT_SECONDS,
            )
        except subprocess.CalledProcessError as error:
            stderr = _redact_diagnostics(error.stderr)
            output = _redact_diagnostics(error.output)
            self._record_call(
                "run_remote",
                exit_code=error.returncode,
                stderr=stderr,
                error=error,
            )
            raise RemoteCommandError(
                error.returncode,
                error.cmd,
                output=output,
                stderr=stderr,
                operation="run_remote",
            ) from error
        except BaseException as error:
            self._record_call("run_remote", exit_code=None, error=error)
            raise
        self._record_call("run_remote", exit_code=result.returncode, stderr=result.stderr)
        return result

    def run_voice(
        self, command: Sequence[str], *, token: str, livekit_url: str
    ) -> subprocess.CompletedProcess[str]:
        """Run a staged caller with the dev endpoint's private voice grant."""
        if not self._entered:
            raise RuntimeError("dev stack is not running")
        if not command or any(not isinstance(arg, str) or "\x00" in arg for arg in command):
            raise ValueError("voice command must contain non-empty safe arguments")
        if not isinstance(token, str) or not token:
            raise ValueError("voice token must be non-empty")
        if not isinstance(livekit_url, str) or not livekit_url:
            raise ValueError("LiveKit URL must be non-empty")
        payload = json.dumps({
            "command": list(command),
            "token": token,
            "livekit_url": livekit_url,
        })
        python_command = shlex.join([
            "setsid", "python3", "-c", _RUN_VOICE_SCRIPT, "--",
            self._remote_dir or "", str(self.dev_port), str(self.health_port),
        ])
        script = (
            "set -euo pipefail\n"
            f"{python_command} <<'MENTAT_VOICE_GRANT'\n"
            f"{payload}\n"
            "MENTAT_VOICE_GRANT\n"
        )
        try:
            result = self._remote(
                script,
                redact=(token,),
                timeout=_REMOTE_CALL_TIMEOUT_SECONDS,
                operation="run_voice",
            )
        except BaseException as error:
            try:
                self._stop_producers()
            except BaseException as cleanup_error:
                error.add_note(f"candidate producer shutdown also failed: {cleanup_error}")
            raise
        self._stop_producers()
        return subprocess.CompletedProcess(
            result.args,
            result.returncode,
            _redact_machine_json(result.stdout, (token,)),
            _redact_diagnostics(result.stderr, (token,)),
        )

    def _stop_producers(self) -> None:
        if self._producers_stopped or self._remote_dir is None:
            return
        self._remote(
            _STOP_RUN_SCRIPT,
            self._remote_dir,
            timeout=_CLEANUP_TIMEOUT_SECONDS,
            operation="stop_candidate_producers",
        )
        self._producers_stopped = True

    def start_worker(self, room: str) -> None:
        """Join the room minted by the dev daemon's voice-token endpoint."""
        if not self._entered:
            raise RuntimeError("dev stack is not running")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", room) is None:
            raise ValueError("room must be a safe LiveKit room name")
        self._remote(
            _START_WORKER_SCRIPT,
            self._remote_dir or "",
            str(self.dev_port),
            str(self.health_port),
            room,
            operation="start_worker",
        )

    def _start(self) -> None:
        if self._batch is None or self.run_id is None:
            raise RuntimeError("run stack is not attached to a staging batch")
        self._remote_dir, self.dev_port, self.health_port = self._batch._prepare_run(self.run_id)
        local_port = self._requested_local_port
        if local_port in (None, 0):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                local_port = sock.getsockname()[1]
        self._local_port = local_port
        self._remote(
            _RUN_SETUP_SCRIPT.replace("__MCP_REWRITE_SOURCE__", _MCP_REWRITE_SOURCE),
            self._batch._remote_dir or "",
            self._remote_dir,
            str(self.dev_port),
            str(self.health_port),
            os.environ.get("MENTAT_VOICE_MODEL", "chatgpt/sol-fast"),
            operation="candidate_setup",
        )
        self._restore_guard_armed = True
        self._forward_spec = f"127.0.0.1:{self._local_port}:127.0.0.1:{self.dev_port}"
        self._transport.forward(self._forward_spec)
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + _READINESS_TIMEOUT_SECONDS
        health_url = f"{self.url}/healthz"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for dev stack health endpoint")
            if self._forward_spec is None:
                raise RuntimeError("SSH port-forward was not requested")
            self._transport.check()
            try:
                with urllib.request.urlopen(
                    health_url,
                    timeout=min(remaining, _READINESS_REQUEST_TIMEOUT_SECONDS),
                ) as response:
                    payload = json.loads(response.read())
                if response.status == 200 and payload == {"status": "ok"}:
                    return
            except (OSError, ValueError):
                pass
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for dev stack health endpoint")
            time.sleep(min(_READINESS_POLL_SECONDS, remaining))

    def _remote(
        self,
        script: str,
        *args: str,
        redact: Sequence[str] = (),
        timeout: float | None = None,
        operation: str = "remote script",
    ) -> subprocess.CompletedProcess[str]:
        try:
            options: dict[str, object] = {
                "input": script,
                "check": True,
                "capture_output": True,
                "text": True,
            }
            options["timeout"] = (
                _CLEANUP_TIMEOUT_SECONDS if timeout is None else timeout
            )
            result = self._run(
                self._transport.ssh_command("sudo", "bash", "-s", "--", *args),
                **options,
            )
        except subprocess.CalledProcessError as error:
            output = _redact_diagnostics(error.output, redact)
            stderr = _redact_diagnostics(error.stderr, redact)
            self._record_call(
                operation,
                exit_code=error.returncode,
                stderr=stderr,
                error=error,
                secrets=redact,
            )
            raise RemoteCommandError(
                error.returncode,
                error.cmd,
                output=output,
                stderr=stderr,
                operation=operation,
            ) from error
        except BaseException as error:
            self._record_call(operation, exit_code=None, error=error, secrets=redact)
            raise
        self._record_call(
            operation,
            exit_code=result.returncode,
            stderr=result.stderr,
            secrets=redact,
        )
        return result

    def _retain_evidence(self) -> None:
        if self._remote_dir is None:
            return
        evidence_dir = Path(tempfile.mkdtemp(prefix="mentat-voice-eval-"))
        os.chmod(evidence_dir, 0o700)
        self.retained_evidence_dir = evidence_dir
        if not self._restore_guard_armed:
            return
        self._remote(
            _RETAIN_EVIDENCE_SCRIPT,
            self._remote_dir,
            timeout=_CLEANUP_TIMEOUT_SECONDS,
            operation="retain_remote_evidence",
        )
        archive = evidence_dir / "retained-evidence.tar.gz"
        try:
            result = self._run(
                self._transport.scp_command(
                    "-p",
                    f"{self.remote}:{self._remote_dir}/retained-evidence.tar.gz",
                    str(archive),
                ),
                check=True,
                capture_output=True,
                text=True,
                timeout=_CLEANUP_TIMEOUT_SECONDS,
            )
        except BaseException as error:
            self._record_call(
                "download_retained_evidence",
                exit_code=error.returncode if isinstance(error, subprocess.CalledProcessError) else None,
                stderr=error.stderr if isinstance(error, subprocess.CalledProcessError) else None,
                error=error,
            )
            raise
        self._record_call("download_retained_evidence", exit_code=result.returncode, stderr=result.stderr)
        if not archive.is_file():
            return
        try:
            import tarfile

            with tarfile.open(archive, "r:gz") as retained:
                members = retained.getmembers()
                input_audio_wavs = {
                    member.name.removeprefix("input-audio/")
                    for member in members
                    if member.name.startswith("input-audio/")
                    and re.fullmatch(
                        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}-turn-[0-9]{3}\.wav",
                        member.name.removeprefix("input-audio/"),
                    ) is not None
                    and member.isfile()
                }
                caller_audio_groups = {}
                for member in members:
                    if not member.name.startswith("caller-audio/") or not member.isfile():
                        continue
                    filename = member.name.removeprefix("caller-audio/")
                    match = re.fullmatch(
                        r"([A-Za-z0-9][A-Za-z0-9_.-]{0,127}-turn-[0-9]{3}(?:-attempt-[0-9]{2})?)"
                        r"(?:-(?:rendered|pushed)\.pcm|\.json)",
                        filename,
                    )
                    if match is not None:
                        caller_audio_groups.setdefault(match.group(1), set()).add(filename)
                caller_audio_names = set()
                for stem, names in caller_audio_groups.items():
                    expected = {
                        f"{stem}-rendered.pcm",
                        f"{stem}-pushed.pcm",
                        f"{stem}.json",
                    }
                    if names == expected:
                        caller_audio_names.update(f"caller-audio/{name}" for name in expected)
                for member in members:
                    name = member.name
                    allowed = name in {
                        "agent.log",
                        "voice.log",
                        "voice/evals/delegations.jsonl",
                        "voice/evals/voice-modes.jsonl",
                    }
                    if name == "sms-audio/transcripts.jsonl":
                        allowed = True
                    elif name.startswith("sms-audio/"):
                        filename = name.removeprefix("sms-audio/")
                        allowed = re.fullmatch(
                            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}-turn-[0-9]{3}\.wav",
                            filename,
                        ) is not None
                    if name.startswith("input-audio/"):
                        filename = name.removeprefix("input-audio/")
                        wav_name = filename if filename.endswith(".wav") else f"{filename[:-4]}.wav"
                        allowed = (
                            re.fullmatch(
                                r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}-turn-[0-9]{3}\.(?:wav|txt)",
                                filename,
                            ) is not None
                            and (filename.endswith(".wav") or wav_name in input_audio_wavs)
                        )
                    if name.startswith("caller-audio/"):
                        allowed = name in caller_audio_names
                    if name.startswith("records/"):
                        filename = name.removeprefix("records/")
                        allowed = bool(filename) and Path(filename).name == filename and filename.endswith(".jsonl")
                    if not allowed or not member.isfile():
                        continue
                    source = retained.extractfile(member)
                    if source is None:
                        continue
                    destination = evidence_dir / name
                    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                    destination.write_bytes(source.read())
                    os.chmod(destination, 0o600)
        finally:
            archive.unlink(missing_ok=True)
        for directory in evidence_dir.rglob("*"):
            if directory.is_dir() and not directory.is_symlink():
                os.chmod(directory, 0o700)

    def _cleanup(self) -> None:
        with self._cleanup_lock:
            if self._cleanup_complete:
                if self._cleanup_error is not None:
                    raise self._cleanup_error
                return
            self._entered = False
            try:
                self._cleanup_once()
            except BaseException as error:
                self._cleanup_error = error
                raise
            finally:
                self._cleanup_complete = True

    def _cleanup_once(self) -> None:
        cleanup_error: BaseException | None = None

        def remember(error: BaseException, context: str) -> None:
            nonlocal cleanup_error
            if cleanup_error is None:
                cleanup_error = error
            else:
                cleanup_error.add_note(context + str(error))

        if self._forward_spec is not None:
            try:
                self._transport.cancel_forward(
                    self._forward_spec,
                    timeout=_CLEANUP_TIMEOUT_SECONDS,
                )
            except BaseException as error:
                remember(error, "SSH forward cancellation also failed: ")
            else:
                self._forward_spec = None
        try:
            self._stop_producers()
        except BaseException as error:
            remember(error, "Candidate producer shutdown also failed: ")
        try:
            self._retain_evidence()
        except BaseException as error:
            remember(error, "Evidence retention also failed: ")
        self._remote_dir = None
        self._local_port = None
        self._restore_guard_armed = False
        try:
            self._transport.close(timeout=_CLEANUP_TIMEOUT_SECONDS)
        except BaseException as error:
            remember(error, "SSH transport shutdown also failed: ")
        else:
            self._forward_spec = None
        try:
            self._write_transport_diagnostics()
        except BaseException as error:
            remember(error, "Transport diagnostics write also failed: ")
        if cleanup_error is not None:
            raise cleanup_error

class DevStack:
    """Build and stage shared voice-eval dependencies for isolated runs."""

    def __init__(
        self,
        *,
        checkout: Path,
        opt_in: bool = False,
        remote: str = "ultraviolet",
        run: CommandRunner = subprocess.run,
    ) -> None:
        self.checkout = Path(checkout).resolve()
        self.opt_in = opt_in
        self.remote = remote
        self._run = run
        self._transport = _SSHTransport(remote, run=run)
        self._remote_dir: str | None = None
        self._entered = False
        self._signal_handlers: dict[int, signal.Handlers] = {}
        self._installed_signal_handlers: dict[int, signal.Handlers] = {}
        self._run_stacks: dict[str, _RunStack] = {}
        self._ports: set[int] = set()
        self._condition = threading.Condition()
        self._lock = threading.Lock()
        self._shutting_down = False
        self._launches_in_progress = 0
        self._restore_guard_refresh_stop: threading.Event | None = None
        self._restore_guard_refresh_thread: threading.Thread | None = None

    def __enter__(self) -> DevStack:
        if not self.opt_in:
            raise RuntimeError("DevStack requires explicit opt-in for a live run")
        self._install_signal_handlers()
        try:
            self._start()
            with self._condition:
                self._entered = True
                self._shutting_down = False
            self._start_restore_guard_refresh()
            return self
        except BaseException as error:
            try:
                self._cleanup()
            except BaseException as cleanup_error:
                error.add_note(f"DevStack batch cleanup also failed: {cleanup_error}")
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

    def _start_restore_guard_refresh(self) -> None:
        if self._remote_dir is None:
            raise RuntimeError("restore guard refresh requires a staged batch")
        stop = threading.Event()
        self._restore_guard_refresh_stop = stop
        self._restore_guard_refresh_thread = threading.Thread(
            target=self._refresh_restore_guard_until_stopped,
            args=(stop, self._remote_dir),
            name="mentat-eval-restore-guard",
            daemon=True,
        )
        self._restore_guard_refresh_thread.start()

    def _refresh_restore_guard_until_stopped(
        self, stop: threading.Event, remote_dir: str
    ) -> None:
        while not stop.wait(_RESTORE_GUARD_REFRESH_INTERVAL_SECONDS):
            try:
                self._remote(
                    _REFRESH_RESTORE_GUARD_SCRIPT,
                    remote_dir,
                    timeout=_RESTORE_GUARD_REFRESH_TIMEOUT_SECONDS,
                    operation="refresh_restore_guard",
                )
            except (RemoteCommandError, OSError, subprocess.TimeoutExpired):
                continue

    def _stop_restore_guard_refresh(self) -> None:
        stop = self._restore_guard_refresh_stop
        thread = self._restore_guard_refresh_thread
        self._restore_guard_refresh_stop = None
        self._restore_guard_refresh_thread = None
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=_CLEANUP_TIMEOUT_SECONDS)

    def run(self, run_id: str) -> _RunStack:
        if not isinstance(run_id, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id) is None:
            raise ValueError("run id must be a safe path component")
        with self._condition:
            if not self._entered or self._shutting_down:
                raise RuntimeError("DevStack batch is not running or is shutting down")
            if run_id in self._run_stacks:
                raise ValueError(f"run id is already in use: {run_id}")
            stack = _RunStack(
                checkout=self.checkout,
                remote=self.remote,
                run=self._run,
                batch=self,
                run_id=run_id,
            )
            self._run_stacks[run_id] = stack
            return stack

    @staticmethod
    def _owner_process_is_dead(owner: object) -> bool:
        if not isinstance(owner, dict):
            return False
        host, pid = owner.get("host"), owner.get("pid")
        if host != socket.gethostname() or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return False
        return False

    def _reap_dead_batches(self) -> None:
        listed = self._run(
            self._transport.ssh_command("sudo", "bash", "-s"), input=_REAP_BATCHES_SCRIPT,
            check=True, capture_output=True, text=True,
        )
        try:
            batches = json.loads(listed.stdout)
        except (TypeError, json.JSONDecodeError):
            return
        if not isinstance(batches, list):
            return
        for item in batches:
            if not isinstance(item, dict):
                continue
            path, raw_owner = item.get("path"), item.get("owner")
            if not isinstance(path, str) or re.fullmatch(r"/tmp/mentat-eval-batch\.[A-Za-z0-9]+", path) is None:
                continue
            try:
                owner = json.loads(raw_owner) if isinstance(raw_owner, str) else None
            except (TypeError, json.JSONDecodeError):
                continue
            if not self._owner_process_is_dead(owner):
                continue
            self._remote(
                _REAP_DEAD_BATCH_SCRIPT + _BATCH_CLEANUP_SCRIPT,
                path,
                str(owner["host"]),
                str(owner["pid"]),
                operation="reap_dead_batch",
            )

    def _begin_launch(self) -> None:
        with self._condition:
            if not self._entered or self._shutting_down:
                raise RuntimeError("DevStack batch is shutting down")
            self._launches_in_progress += 1

    def _end_launch(self) -> None:
        with self._condition:
            self._launches_in_progress -= 1
            self._condition.notify_all()

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
            self._transport.ssh_command("mktemp", "-d", "/tmp/mentat-eval-batch.XXXXXX"),
            check=True,
            capture_output=True,
            text=True,
        )
        self._remote_dir = staged.stdout.strip()
        if re.fullmatch(r"/tmp/mentat-eval-batch\.[A-Za-z0-9]+", self._remote_dir) is None:
            self._remote_dir = None
            raise RuntimeError("remote staging returned an unsafe directory")
        shared = f"{self._remote_dir}/shared"
        self._run(
            self._transport.ssh_command(
                "mkdir", "-m", "700", "-p", "--",
                f"{shared}/mentat", f"{shared}/voice/assets", f"{shared}/voice/evals",
            ),
            check=True,
            capture_output=True,
            text=True,
        )
        self._run(
            self._transport.scp_command(
                "-r", str(package / "src"), str(package / "node_modules"),
                str(package / "package.json"), str(self.checkout / "prompt.md"),
                f"{self.remote}:{shared}/mentat/",
            ),
            check=True, capture_output=True, text=True,
        )
        voice_files = [
            "agent.py", "persona.md", "request.py", "stream.py", "voices.py", "caller.py",
        ]
        self._run(
            self._transport.scp_command(
                "-r", *(str(self.checkout / "voice" / name) for name in voice_files),
                str(self.checkout / "voice" / "evals" / "phone.py"),
                str(self.checkout / "nix" / "voice-env.nix"),
                f"{self.remote}:{shared}/voice/",
            ),
            check=True, capture_output=True, text=True,
        )
        self._run(
            self._transport.scp_command(
                "-r",
                *(str(self.checkout / "voice" / "evals" / name) for name in (
                    "runner.py", "dev_stack.py", "report.py", "scenarios.py",
                )),
                f"{self.remote}:{shared}/voice/evals/",
            ),
            check=True, capture_output=True, text=True,
        )
        self._run(
            self._transport.scp_command(
                "-r", str(self.checkout / "voice" / "assets"), f"{self.remote}:{shared}/voice/"
            ),
            check=True, capture_output=True, text=True,
        )
        self._reap_dead_batches()
        self._remote(
            _BATCH_SETUP_SCRIPT.replace(
                "__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE
            ).replace("__BATCH_CLEANUP_SCRIPT__", _BATCH_CLEANUP_SCRIPT),
            self._remote_dir,
            socket.gethostname(),
            str(os.getpid()),
            operation="batch_setup",
        )

    def _prepare_run(self, run_id: str) -> tuple[str, int, int]:
        if not self._entered or self._remote_dir is None:
            raise RuntimeError("DevStack batch is not running")
        run_dir = f"{self._remote_dir}/runs/{run_id}"
        with self._lock:
            for _ in range(10):
                ports = self._remote(
                    _REMOTE_PORT_PROBE_SCRIPT,
                    self._remote_dir,
                    operation="allocate_run_ports",
                )
                try:
                    available = json.loads(ports.stdout)
                except (TypeError, json.JSONDecodeError) as error:
                    raise RuntimeError("remote port probe returned invalid port data") from error
                if not isinstance(available, dict):
                    raise RuntimeError("remote port probe returned invalid port data")
                dev_port, health_port = available.get("dev_port"), available.get("health_port")
                if (
                    isinstance(dev_port, bool) or not isinstance(dev_port, int) or not 1 <= dev_port <= 65535
                    or isinstance(health_port, bool) or not isinstance(health_port, int)
                    or not 1 <= health_port <= 65535 or dev_port == health_port
                ):
                    raise RuntimeError("remote port probe returned invalid port data")
                if dev_port not in self._ports and health_port not in self._ports:
                    self._ports.update((dev_port, health_port))
                    return run_dir, dev_port, health_port
        raise RuntimeError("remote port probe could not allocate distinct run ports")

    def _remote(
        self,
        script: str,
        *args: str,
        timeout: float | None = None,
        operation: str = "remote script",
    ) -> subprocess.CompletedProcess[str]:
        try:
            options: dict[str, object] = {
                "input": script,
                "check": True,
                "capture_output": True,
                "text": True,
            }
            options["timeout"] = (
                _CLEANUP_TIMEOUT_SECONDS if timeout is None else timeout
            )
            return self._run(
                self._transport.ssh_command("sudo", "bash", "-s", "--", *args),
                **options,
            )
        except subprocess.CalledProcessError as error:
            raise RemoteCommandError(
                error.returncode,
                error.cmd,
                output=_redact_diagnostics(error.output),
                stderr=_redact_diagnostics(error.stderr),
                operation=operation,
            ) from error

    def _cleanup(self) -> None:
        active_handlers = self._suppress_cleanup_signals()
        try:
            self._cleanup_batch()
        finally:
            self._restore_cleanup_signals(active_handlers)

    def _cleanup_batch(self) -> None:
        first_error: BaseException | None = None
        with self._condition:
            self._shutting_down = True
            self._entered = False
            while self._launches_in_progress:
                self._condition.wait()
        with self._condition:
            run_stacks = tuple(self._run_stacks.values())
        for stack in run_stacks:
            if stack._remote_dir is not None or stack._forward_spec is not None:
                try:
                    stack._cleanup()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        self._stop_restore_guard_refresh()
        remote_dir = self._remote_dir
        if remote_dir is not None:
            try:
                script = '''set -euo pipefail
BATCH_DIR=$1
if [ -f "$BATCH_DIR/cleanup.sh" ]; then
  bash "$BATCH_DIR/cleanup.sh" "$BATCH_DIR"
elif [ -n "$BATCH_DIR" ]; then
  rm -rf -- "$BATCH_DIR"
fi
'''
                self._remote(
                    script,
                    remote_dir,
                    timeout=_CLEANUP_TIMEOUT_SECONDS,
                    operation="batch_cleanup",
                )
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                self._remote_dir = None
        try:
            self._transport.close(timeout=_CLEANUP_TIMEOUT_SECONDS)
        except BaseException as error:
            if first_error is None:
                first_error = error
        if first_error is not None:
            raise first_error

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            handler = self._interrupted
            self._signal_handlers[signum] = signal.getsignal(signum)
            self._installed_signal_handlers[signum] = handler
            signal.signal(signum, handler)

    def _interrupted(self, signum: int, frame: FrameType | None) -> None:
        if self._shutting_down:
            return
        raise KeyboardInterrupt(f"received signal {signum}")

    def _ignore_cleanup_signal(self, _signum: int, _frame: FrameType | None) -> None:
        return

    def _suppress_cleanup_signals(
        self,
    ) -> dict[int, tuple[signal.Handlers, signal.Handlers]]:
        if threading.current_thread() is not threading.main_thread():
            return {}
        handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM):
            active = signal.getsignal(signum)
            cleanup_handler = self._ignore_cleanup_signal
            handlers[signum] = (active, cleanup_handler)
            signal.signal(signum, cleanup_handler)
        return handlers

    def _restore_cleanup_signals(
        self, handlers: dict[int, tuple[signal.Handlers, signal.Handlers]]
    ) -> None:
        if threading.current_thread() is threading.main_thread():
            for signum, (active, cleanup_handler) in handlers.items():
                if signal.getsignal(signum) == cleanup_handler:
                    signal.signal(signum, active)

    def _restore_signal_handlers(self) -> None:
        if threading.current_thread() is threading.main_thread():
            for signum, handler in self._signal_handlers.items():
                if signal.getsignal(signum) == self._installed_signal_handlers[signum]:
                    signal.signal(signum, handler)
        self._signal_handlers.clear()
        self._installed_signal_handlers.clear()
