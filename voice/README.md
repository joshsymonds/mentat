# Voice surface

The voice worker joins a LiveKit room and runs GPT-Live-1 in client delegation
mode. GPT-Live handles the conversation and hands backend requests to the
worker. The worker streams each request to mentatd, then feeds the response
back as commentary while it arrives. The daemon owns memory, systems, lookups,
and actions. The worker runs on ultraviolet beside mentatd and reaches it over
loopback.

## Private context

The repository is public, so `persona.md` contains the voice and no private
facts. Deployment supplies a TOML file as `MENTAT_VOICE_PRIVATE`:

```toml
about = """
Who he is: ...one paragraph, folded into the worker instructions.
"""
[pronunciations]
Symonds = "Sigh-monds"
```

Unset means a development room has no private context. A set but unreadable
path fails the worker at startup. To run a dev room with it, add
`MENTAT_VOICE_PRIVATE=/run/agenix/mentat-voice-private` to the launch
environment in step 2.

## Testing branch code in a live room

Audio changes need a real room. Run branch code against the production SFU and
mentatd in a throwaway room without redeploying the worker. For this web-search
change, run calls A-D below and compare each printed latency with the eight
second target; the worker log records whether A, B and D used `web_search`.

Use one shell session for the procedure. Install the cleanup trap before
stopping production voice; it removes the dev process/files and starts
`mentat-voice` on normal exit, failure, Ctrl-C or termination:

```sh
DEV_DIR=
cleanup() {
  if [[ -n "$DEV_DIR" ]]; then
    ssh ultraviolet sudo env DEV_DIR="$DEV_DIR" sh -s <<'REMOTE_CLEANUP' || true
set -eu
if [ -f "$DEV_DIR/agent.pid" ]; then
  kill "$(cat "$DEV_DIR/agent.pid")" 2>/dev/null || true
fi
rm -rf -- "$DEV_DIR"
REMOTE_CLEANUP
  fi
  ssh ultraviolet sudo systemctl start mentat-voice
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
ssh ultraviolet sudo systemctl stop mentat-voice
```

### 1. Stage the branch files on ultraviolet

The deployed worker needs `agent.py`, `persona.md`, `request.py`, `stream.py`,
and `assets/earcon.wav`. Copy them to a private scratch directory with a
writable home for plugin caches. Use `mktemp -d` rather than a predictable
path.

```sh
DEV_DIR=$(ssh ultraviolet 'mktemp -d /tmp/mentat-voice-dev.XXXXXX')
ssh ultraviolet "mkdir -p $DEV_DIR/assets $DEV_DIR/home/cache"
scp voice/agent.py voice/persona.md voice/request.py voice/stream.py voice/caller.py ultraviolet:$DEV_DIR/
scp voice/assets/earcon.wav ultraviolet:$DEV_DIR/assets/
```

### 2. Launch the agent pinned to a dev room

Load the production secrets, including `OPENAI_API_KEY`, from the environment
file. Use loopback URLs, a distinct health port, a writable home, and a
bounded timeout around the process. Set `DEV_PY` to the Python executable used
by the production unit. Keep credentials out of command arguments.

```sh
DEV_ROOM=dev-myfeature-REV
DEV_PY=/path/to/unit-python
ssh ultraviolet sudo env DEV_DIR="$DEV_DIR" DEV_ROOM="$DEV_ROOM" \
  DEV_PY="$DEV_PY" \
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

Confirm startup with `ssh ultraviolet sudo grep -E 'starting worker|job-' $DEV_DIR/agent.log`.
The LiveKit Agents 1.8.1 `connect --room <name>` mode still exists, behind a
deprecation warning. Use a unique room name for every test.

### 3. Mint a join token and join from a browser

Credentials go in environment variables rather than the `lk` command line.
Mint a short-lived token, open the printed browser URL on the tailnet, allow
the microphone, and talk.

```sh
ssh ultraviolet sudo env DEV_ROOM="$DEV_ROOM" bash -c 'set -euo pipefail
  keyfile=/run/agenix/livekit-keys
  export LIVEKIT_API_KEY=$(sed -n "s/^\([^:[:space:]]\+\)[[:space:]]*:.*$/\1/p" "$keyfile" | head -n1)
  export LIVEKIT_API_SECRET=$(sed -n "s/^[^:]\+:[[:space:]]*\(.\+\)$/\1/p" "$keyfile" | head -n1)
  token=$(lk token create --join --room "$DEV_ROOM" \
          --identity acceptance --valid-for 45m --token-only)
  printf "https://meet.livekit.io/custom?liveKitUrl=wss%%3A%%2F%%2Fultraviolet.tail82223.ts.net%%3A7443&token=%s\n" "$token"'
```

The connect job stops after the room has no human participant for several
minutes or when the last human leaves. The SFU remains authoritative about
participants, and the worker log shows delegation, stream, and disconnect
events.

### 4. Run calls A-D

Use `LINE@DELAY_SECONDS::ANSWER_REGEX` for each scripted prompt. Regexes are
matched against timestamped Whisper segments; output reports the first match's
start relative to the caller's speech end. These four calls exercise current
lookup, price/source freshness, stable knowledge, and stale-summary caution:

```sh
ssh ultraviolet sudo env DEV_DIR="$DEV_DIR" DEV_ROOM="$DEV_ROOM" DEV_PY="$DEV_PY" bash -s <<'REMOTE'
set -euo pipefail
set -a; . /run/agenix/mentat-voice-env; set +a
export LIVEKIT_URL=ws://127.0.0.1:7880
exec timeout 240 "$DEV_PY" "$DEV_DIR/caller.py" "$DEV_ROOM" \
  'What is the latest released version of livekit-agents on PyPI?@1::[0-9]+\.[0-9]+\.[0-9]+' \
  'What is the current Bitcoin price in USD according to CoinGecko?@1::(?i)(?:\$|USD\s*)[0-9,]+(?:\.[0-9]+)?|[0-9,]+(?:\.[0-9]+)?\s*USD' \
  'Who wrote Pride and Prejudice?@1::(?i)Jane\s+Austen' \
  'What was the latest Formula 1 Grand Prix, and who won it?@1::(?i)\b(?:won|winner|unsure|uncertain)\b'
REMOTE
```

Compare A with PyPI, B with CoinGecko (record its currency and fetch time), C
with Jane Austen and no delegation, and D with a live F1 results page or an
explicit statement of uncertainty. Check `agent.log` for `web_search` on A, B,
and D and for no delegation on C.

### Gotchas

- **The connect job self-terminates** about five minutes after the room has
  no human participant, and again when the last human leaves. If the room
  "stopped answering", check the log for `room disconnected` and relaunch —
  the same token still works while it's valid.
- **meet.livekit.io hides agents.** An "empty" room can still contain the
  agent; participant tiles only show humans. The SFU is the authority:
  `lk room participants list <room>` (same env vars as token minting, plus
  `LIVEKIT_URL=ws://127.0.0.1:7880`).
- **Watch the log during the test.** `sudo tail -F $DEV_DIR/agent.log` filtered
  for `session duration|delegation failed|voice session closing|ERROR|Traceback|disconnected` shows every turn
  land in real time. rtc_session errors during teardown are normal.


## Phone control

Phone actions are Mentat MCP tools executed by the Android bridge service.
The voice worker delegates navigation, dialing, texting, alarms, timers, links,
and place searches to mentatd. mentatd applies the send confirmation rule and
returns the action result before the voice reports completion.

The bridge is the location boundary. It supplies a fresh location only for a
place search and never persists that location. If no fresh or acceptably recent
fix exists, the backend reports that location is unavailable and asks for a
rough locality.

The Places search key is optional for local development and lives in the voice
unit's environment file, which mentatd also loads. Create a billed Google Cloud
project and a Places-only key from a shell authenticated as Josh:

```sh
gcloud projects create mentat-voice-places
gcloud billing accounts list
gcloud billing projects link mentat-voice-places --billing-account=<the listed open account>
gcloud services enable places.googleapis.com --project=mentat-voice-places
gcloud services api-keys create --project=mentat-voice-places --display-name=mentat-voice --api-target=service=places.googleapis.com
gcloud services api-keys get-key-string <resource name from the create output>
```

Put the resulting value in the voice environment file as
`MENTAT_PLACES_API_KEY=<key>`.

## Tests

`just test-voice` runs the offline standard-library unittest suite in
`voice/tests`. It covers request construction, ending policy, NDJSON streaming,
commentary byte limits, source wiring, and the earcon asset. The earcon is
generated by `assets/generate.py` and checked for deterministic regeneration.
