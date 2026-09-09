# Voice surface

Front, the LiveKit voice agent: `agent.py` joins a LiveKit room, runs local
turns through LiveKit Cloud STT/TTS, and consults mentatd (`ask_mentat`) for
anything with memory or tools. Deployed on ultraviolet as the `mentat-voice`
systemd unit (see `nix/module.nix`); the SFU, mentatd, and the agent all share
that host, so the agent talks to both over loopback.

## Private context

The repository is public, so `persona.md` describes the voice and nothing
about the person. Who Josh is lives in a TOML file the deploy hands the unit
as a systemd credential (`services.mentat.voice.privateContextFile`, an
agenix secret in nix-config), named to the agent by `MENTAT_VOICE_PRIVATE`:

```toml
about = """
Who he is: ...one paragraph, folded into the instructions on every turn.
"""
keyterms = ["Symonds", "Olive"]          # names speech recognition should expect

[pronunciations]                         # word = what the synthesizer is handed
Symonds = "Sigh-monds"                   # captions keep the real spelling
```

Unset means the voice knows no one (dev rooms, CI); a set but unreadable
path fails the worker at start. To run a dev room with it, add
`MENTAT_VOICE_PRIVATE=/run/agenix/mentat-voice-private` to the launch
environment in step 2.

## Testing branch code in a live room

Audio changes can't be accepted from unit tests — someone has to listen. The
workflow below runs *branch* agent code against the *production* SFU and
mentatd on ultraviolet, in a throwaway room, without redeploying anything.

The mechanism is the LiveKit Agents CLI `connect` mode: `agent.py connect
--room <name>` pins one process to one named room instead of registering for
dispatch. Use a unique room name per test (e.g. `dev-<feature>-<rev>`).

**Stop the production worker first.** `mentat-voice` registers with no
`agent_name`, so the SFU dispatches it into *every* new room — including your
dev room, where its (old) audio will play on top of your branch agent's.
Verified the hard way 2026-08-21: two agents in the room, both audible.

```sh
ssh ultraviolet sudo systemctl stop mentat-voice   # restart when done!
```

### 1. Stage the branch files on ultraviolet

The agent is a flat directory: `agent.py persona.md request.py stream.py
assets/*.wav`. Copy those to a private scratch dir with a writable HOME for
the livekit plugin caches. `mktemp -d`, not a fixed name: a predictable
`/tmp` path with `mkdir -p` silently reuses a directory another local user
could have pre-created, and step 2 executes code out of this directory.

```sh
dev=$(ssh ultraviolet 'mktemp -d /tmp/mentat-voice-dev.XXXXXX')
ssh ultraviolet "mkdir -p $dev/assets $dev/home/cache"
scp voice/agent.py voice/persona.md voice/request.py voice/stream.py ultraviolet:$dev/
scp voice/assets/*.wav ultraviolet:$dev/assets/
```

### 2. Launch the agent pinned to a dev room

Run it with the production secrets and the same python env as the unit
(`systemctl cat mentat-voice` shows the store path in `ExecStart`). Loopback
URLs because everything is co-located; a distinct `MENTAT_VOICE_HTTP_PORT`
because the unit owns 8482; `timeout` so a forgotten process can't outlive
the session by more than an hour. Root is needed only to read the agenix
secrets — the agent itself runs as `nobody` via `setpriv`, mirroring the
production unit's `DynamicUser` isolation (a network-facing agent should not
hold a privileged identity, dev run or not):

```sh
ssh ultraviolet sudo env DEV_DIR=$dev DEV_ROOM=dev-myfeature-$(git rev-parse --short HEAD) \
  DEV_PY=<python-from-unit> \
  bash -c 'set -euo pipefail
    set -a; . /run/agenix/mentat-voice-env; set +a
    export LIVEKIT_URL=ws://127.0.0.1:7880 MENTAT_URL=http://127.0.0.1:8484 \
           HOME="$DEV_DIR/home" XDG_CACHE_HOME="$DEV_DIR/home/cache" \
           MENTAT_VOICE_HTTP_PORT=8483
    umask 077
    chown -R nobody:nogroup "$DEV_DIR"
    setsid nohup timeout 3600 \
      setpriv --reuid=nobody --regid=nogroup --clear-groups \
      "$DEV_PY" "$DEV_DIR/agent.py" connect --room "$DEV_ROOM" \
      >"$DEV_DIR/agent.log" 2>&1 </dev/null &
    printf "%s\n" "$!" >"$DEV_DIR/agent.pid"'
```

Confirm startup: `sudo grep -E 'starting worker|job-' $dev/agent.log`.

### 3. Mint a join token and join from a browser

Credentials go in the environment, never on `lk`'s argv —
`/proc/<pid>/cmdline` is world-readable (same rule as the `voice-token`
script in nix-config, which does exactly this for the pinned `office` room):

```sh
ssh ultraviolet sudo bash -c 'set -euo pipefail
  keyfile=/run/agenix/livekit-keys   # one-line YAML: <api-key>: <api-secret>
  export LIVEKIT_API_KEY=$(sed -n "s/^\([^:[:space:]]\+\)[[:space:]]*:.*$/\1/p" "$keyfile" | head -n1)
  export LIVEKIT_API_SECRET=$(sed -n "s/^[^:]\+:[[:space:]]*\(.\+\)$/\1/p" "$keyfile" | head -n1)
  token=$(lk token create --join --room dev-myfeature-<rev> \
          --identity acceptance --valid-for 45m --token-only)
  printf "https://meet.livekit.io/custom?liveKitUrl=wss%%3A%%2F%%2Fultraviolet.tail82223.ts.net%%3A7443&token=%s\n" "$token"'
```

Open the printed URL in a browser on the tailnet, allow the microphone, talk.
The token is a real (if short-lived) room credential — treat the URL
accordingly.

### Gotchas

- **The connect job self-terminates** about five minutes after the room has
  no human participant, and again when the last human leaves. If the room
  "stopped answering", check the log for `room disconnected` and relaunch —
  the same token still works while it's valid.
- **meet.livekit.io hides agents.** An "empty" room can still contain the
  agent; participant tiles only show humans. The SFU is the authority:
  `lk room participants list <room>` (same env vars as token minting, plus
  `LIVEKIT_URL=ws://127.0.0.1:7880`).
- **Watch the log during the test.** `sudo tail -F $dev/agent.log` filtered
  for `turn latency|consult|ERROR|Traceback|disconnected` shows every turn
  land in real time. rtc_session errors during teardown are normal.

### 4. Clean up

```sh
ssh ultraviolet "sudo sh -c 'kill \$(cat $dev/agent.pid) 2>/dev/null; rm -rf $dev'"
ssh ultraviolet sudo systemctl start mentat-voice
```

## Tests

`just test-voice` (part of `just`) runs the offline unittest suite in
`tests/`, including the asset-wiring contract that pins which sounds exist
and how `agent.py` uses them. Sound files are generated, not authored — see
`assets/generate.py`; `tests/test_assets.py` re-derives them and fails on
drift.
