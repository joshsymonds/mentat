"""Opt-in staging and lifecycle for an isolated remote voice-eval stack."""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from types import FrameType
from typing import Callable, Sequence

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_READINESS_TIMEOUT_SECONDS = 30.0
_READINESS_POLL_SECONDS = 0.2
_READINESS_REQUEST_TIMEOUT_SECONDS = 2.0
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


class RemoteCommandError(subprocess.CalledProcessError):
    """A remote command failure whose diagnostic includes scrubbed stderr."""

    def __str__(self) -> str:
        message = super().__str__()
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


_SETUP_SCRIPT = r'''set -euo pipefail
DEV_DIR=$1
DEV_PORT=$2
HEALTH_PORT=$3
VOICE_MODEL=$4

# Find the running production executables before stopping only the voice worker.
MENTAT_PID=$(systemctl show mentatd --property=MainPID --value)
VOICE_PID=$(systemctl show mentat-voice --property=MainPID --value)
test "$MENTAT_PID" -gt 0
test "$VOICE_PID" -gt 0
NODE_BIN=$(readlink -f "/proc/$MENTAT_PID/exe")
VOICE_ENV_PATH=$(
  nix build \
    --impure \
    --expr "let pkgs = import (builtins.getFlake \"nixpkgs\").outPath {}; in import $DEV_DIR/voice/voice-env.nix { inherit pkgs; }" \
    --no-link \
    --print-out-paths
)
case "$VOICE_ENV_PATH" in
  /nix/store/*) ;;
  *) echo "candidate voice environment build did not return one Nix store path" >&2; exit 1 ;;
esac
case "$VOICE_ENV_PATH" in
  *$'\n'*) echo "candidate voice environment build returned multiple store paths" >&2; exit 1 ;;
esac
VOICE_PY="$VOICE_ENV_PATH/bin/python"
test -x "$VOICE_PY"
case "$VOICE_PY" in
  /*) test -x "$VOICE_PY" ;;
  *) echo "candidate voice Python executable must be an absolute path" >&2; exit 1 ;;
esac
"$VOICE_PY" - <<'PY'
import importlib

for module in (
    "aiohttp",
    "livekit.api",
    "livekit.rtc",
    "livekit.plugins.dtln",
    "livekit.plugins.elevenlabs",
    "livekit.plugins.openai",
    "livekit.plugins.silero",
    "livekit.plugins.turn_detector",
):
    try:
        importlib.import_module(module)
    except ImportError as error:
        raise SystemExit(
            f"candidate voice environment missing required module {module}: {error}"
        ) from error
PY

mkdir -p "$DEV_DIR/mentat" "$DEV_DIR/voice/assets" "$DEV_DIR/home/mentat" "$DEV_DIR/home/voice/cache" "$DEV_DIR/records"
umask 077
chown root:root "$DEV_DIR"
chmod 711 "$DEV_DIR" "$DEV_DIR/home"
chmod 700 "$DEV_DIR/home/mentat" "$DEV_DIR/home/voice" "$DEV_DIR/home/voice/cache" "$DEV_DIR/records"
chown -R mentat:mentat "$DEV_DIR/mentat" "$DEV_DIR/home/mentat" "$DEV_DIR/records"
chown -R nobody:nogroup "$DEV_DIR/voice" "$DEV_DIR/home/voice"

if systemctl is-active --quiet mentat-voice; then
  RESTORE_ACTION=start
else
  RESTORE_ACTION=stop
fi
RESTORE_UNIT="mentat-eval-restore-${DEV_DIR##*.}"
printf '%s\n' "$RESTORE_ACTION" > "$DEV_DIR/restore-action"
printf '%s\n' "$RESTORE_UNIT" > "$DEV_DIR/restore-unit"

# The transient timer runs outside this shell and survives runner death.
cat > "$DEV_DIR/cleanup.sh" <<'CLEANUP'
set +e
RESTORE_ACTION=$(cat "$DEV_DIR/restore-action" 2>/dev/null)
RESTORE_UNIT=$(cat "$DEV_DIR/restore-unit" 2>/dev/null)
restore_voice() {
  if [ -n "$RESTORE_UNIT" ]; then
    systemctl stop "$RESTORE_UNIT.timer" "$RESTORE_UNIT.service" >/dev/null 2>&1
  fi
  if [ "$RESTORE_ACTION" = start ]; then
    systemctl start mentat-voice
  elif [ "$RESTORE_ACTION" = stop ]; then
    systemctl stop mentat-voice
  fi
}
trap restore_voice EXIT
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
restore_voice
trap - EXIT
rm -rf -- "$DEV_DIR"
CLEANUP
chmod 700 "$DEV_DIR/cleanup.sh"
systemd-run --quiet --unit="$RESTORE_UNIT" --on-active=30m "$(command -v systemctl)" "$RESTORE_ACTION" mentat-voice

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
    elif name == "mentat":
        stage_gateway_key(values, dev_dir)
    (dev_dir / f"{name}.env.json").write_text(json.dumps(values))
    os.chmod(dev_dir / f"{name}.env.json", 0o600)
PY

# Start only the candidate daemon; the worker waits for the room from the
# token returned by this daemon's voice-token endpoint.
MENTAT_VOICE_MODEL="$VOICE_MODEL" python3 - "$DEV_DIR" "$DEV_PORT" "$NODE_BIN" "$VOICE_PY" <<'PY'
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

__MCP_REWRITE_SOURCE__

dev_dir = Path(sys.argv[1])
dev_port = int(sys.argv[2])
node_bin = sys.argv[3]
voice_python = sys.argv[4]
voice_model = os.environ.get("MENTAT_VOICE_MODEL", "chatgpt/sol-fast")
setpriv_path = shutil.which("setpriv")
if setpriv_path is None:
    raise RuntimeError("setpriv executable is unavailable in the setup environment")
setpriv_path = str(Path(setpriv_path).resolve())
setpriv_file = dev_dir / "setpriv.path"
setpriv_file.write_text(setpriv_path)
setpriv_file.chmod(0o644)
source_env = json.loads((dev_dir / "mentat.env.json").read_text())
production_listen = source_env.get("MENTAT_LISTEN", "127.0.0.1:8484")
production_port = int(production_listen.rsplit(":", 1)[1])

env = {key: value for key, value in source_env.items() if key != "OPENAI_API_KEY"}
env["MENTAT_VOICE_MODEL"] = voice_model
env["MENTAT_SESSION_TTL"] = "90s"
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
    [setpriv_path, "--reuid=mentat", "--regid=mentat", "--init-groups", node_bin, str(dev_dir / "mentat/src/main.ts")],
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
    for relative in ("agent.log", "voice.log", "voice/evals/delegations.jsonl"):
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
os.chmod(archive, 0o600)
owner = pwd.getpwnam(os.environ["SUDO_USER"])
os.chown(archive, owner.pw_uid, owner.pw_gid)
PY
'''


_START_WORKER_SCRIPT = r'''set -euo pipefail
DEV_DIR=$1
DEV_PORT=$2
HEALTH_PORT=$3
ROOM=$4
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


_RUN_VOICE_SCRIPT = r'''import json
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 5 or sys.argv[1] != "--":
    raise RuntimeError("remote caller received invalid staging arguments")
DEV_DIR = Path(sys.argv[2])
DEV_PORT = sys.argv[3]
HEALTH_PORT = sys.argv[4]
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
        dev_port: int | None = None,
        health_port: int | None = None,
        local_port: int | None = None,
        run: CommandRunner = subprocess.run,
    ) -> None:
        self.checkout = Path(checkout).resolve()
        self.opt_in = opt_in
        self.remote = remote
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
        self._tunnel: subprocess.Popen[bytes] | None = None
        self._signal_handlers: dict[int, signal.Handlers] = {}
        self._entered = False
        self._restore_guard_armed = False
        self.retained_evidence_dir: Path | None = None

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
            "python3", "-c", _RUN_VOICE_SCRIPT, "--",
            self._remote_dir or "", str(self.dev_port), str(self.health_port),
        ])
        script = (
            "set -euo pipefail\n"
            f"{python_command} <<'MENTAT_VOICE_GRANT'\n"
            f"{payload}\n"
            "MENTAT_VOICE_GRANT\n"
        )
        result = self._remote(script, redact=(token,))
        return subprocess.CompletedProcess(
            result.args,
            result.returncode,
            _redact_machine_json(result.stdout, (token,)),
            _redact_diagnostics(result.stderr, (token,)),
        )

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
        )

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
                str(self.checkout / "nix" / "voice-env.nix"),
                f"{self.remote}:{self._remote_dir}/voice/",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self._run(
            [
                "scp", "-r",
                *(str(self.checkout / "voice" / "evals" / name) for name in (
                    "runner.py", "dev_stack.py", "report.py", "scenarios.py",
                )),
                f"{self.remote}:{self._remote_dir}/voice/evals/",
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

        self._select_remote_ports()

        local_port = self._requested_local_port
        if local_port in (None, 0):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                local_port = sock.getsockname()[1]
        self._local_port = local_port

        setup_script = _SETUP_SCRIPT.replace("__MCP_REWRITE_SOURCE__", _MCP_REWRITE_SOURCE).replace(
            "__PRIVATE_CREDENTIAL_SOURCE__", _PRIVATE_CREDENTIAL_SOURCE
        )
        requested_model = os.environ.get("MENTAT_VOICE_MODEL", "chatgpt/sol-fast")
        self._remote(
            setup_script,
            self._remote_dir,
            str(self.dev_port),
            str(self.health_port),
            requested_model,
        )
        self._restore_guard_armed = True
        self._tunnel = subprocess.Popen(
            [
                "ssh",
                "-o", "ControlMaster=no",
                "-o", "ControlPath=none",
                "-o", "ExitOnForwardFailure=yes",
                "-N", "-L",
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
        self._wait_until_ready()

    def _select_remote_ports(self) -> None:
        if self.dev_port is not None and self.health_port is not None:
            return
        result = self._remote(_REMOTE_PORT_PROBE_SCRIPT, self._remote_dir or "")
        try:
            ports = json.loads(result.stdout)
        except (TypeError, json.JSONDecodeError) as error:
            raise RuntimeError("remote port probe returned invalid port data") from error
        if not isinstance(ports, dict):
            raise RuntimeError("remote port probe returned invalid port data")
        dev_port = ports.get("dev_port")
        health_port = ports.get("health_port")
        if (
            isinstance(dev_port, bool)
            or not isinstance(dev_port, int)
            or not 1 <= dev_port <= 65535
            or isinstance(health_port, bool)
            or not isinstance(health_port, int)
            or not 1 <= health_port <= 65535
        ):
            raise RuntimeError("remote port probe returned invalid port data")
        selected_dev_port = self.dev_port if self.dev_port is not None else dev_port
        selected_health_port = self.health_port if self.health_port is not None else health_port
        if selected_dev_port == selected_health_port:
            raise RuntimeError("remote port probe returned invalid port data")
        self.dev_port = selected_dev_port
        self.health_port = selected_health_port

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + _READINESS_TIMEOUT_SECONDS
        health_url = f"{self.url}/healthz"
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("timed out waiting for dev stack health endpoint")
            if self._tunnel is None or self._tunnel.poll() is not None:
                raise RuntimeError("SSH port-forward exited before the dev stack became ready")
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
        refresh_guard: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        if self._restore_guard_armed and refresh_guard:
            self._remote(
                _REFRESH_RESTORE_GUARD_SCRIPT,
                self._remote_dir or "",
                refresh_guard=False,
            )
        try:
            return self._run(
                ["ssh", self.remote, "sudo", "bash", "-s", "--", *args],
                input=script,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            raise RemoteCommandError(
                error.returncode,
                error.cmd,
                output=_redact_diagnostics(error.output, redact),
                stderr=_redact_diagnostics(error.stderr, redact),
            ) from error

    def _retain_evidence(self) -> None:
        if self._remote_dir is None or not self._restore_guard_armed:
            return
        evidence_dir = Path(tempfile.mkdtemp(prefix="mentat-voice-eval-"))
        os.chmod(evidence_dir, 0o700)
        self.retained_evidence_dir = evidence_dir
        self._remote(_RETAIN_EVIDENCE_SCRIPT, self._remote_dir)
        archive = evidence_dir / "retained-evidence.tar.gz"
        self._run(
            [
                "scp", "-p",
                f"{self.remote}:{self._remote_dir}/retained-evidence.tar.gz",
                str(archive),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if not archive.is_file():
            evidence_dir.rmdir()
            self.retained_evidence_dir = None
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
                for member in members:
                    name = member.name
                    allowed = name in {"agent.log", "voice.log", "voice/evals/delegations.jsonl"}
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
DEV_DIR=$1
if [ -n "$DEV_DIR" ] && [ -f "$DEV_DIR/cleanup.sh" ]; then
  DEV_DIR="$DEV_DIR" bash "$DEV_DIR/cleanup.sh"
elif [ -n "$DEV_DIR" ]; then
  rm -rf -- "$DEV_DIR"
fi
'''
            retention_error: BaseException | None = None
            try:
                self._retain_evidence()
            except BaseException as error:
                retention_error = error
            try:
                self._remote(
                    script,
                    self._remote_dir or "",
                    refresh_guard=False,
                )
            finally:
                self._remote_dir = None
                self._local_port = None
                self._restore_guard_armed = False
            if retention_error is not None:
                raise retention_error

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
