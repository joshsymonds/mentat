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
mentatd in a throwaway room without redeploying the worker.

Stop the production worker first so it does not join the development room.

```sh
ssh ultraviolet sudo systemctl stop mentat-voice
```

### 1. Stage the branch files on ultraviolet

The deployed worker needs `agent.py`, `persona.md`, `request.py`, `stream.py`,
and `assets/earcon.wav`. Copy them to a private scratch directory with a
writable home for plugin caches. Use `mktemp -d` rather than a predictable
path.

```sh
dev=$(ssh ultraviolet 'mktemp -d /tmp/mentat-voice-dev.XXXXXX')
ssh ultraviolet "mkdir -p $dev/assets $dev/home/cache"
scp voice/agent.py voice/persona.md voice/request.py voice/stream.py ultraviolet:$dev/
scp voice/assets/earcon.wav ultraviolet:$dev/assets/
```

### 2. Launch the agent pinned to a dev room

Load the production secrets, including `OPENAI_API_KEY`, from the environment
file. Use loopback URLs, a distinct health port, a writable home, and a
bounded timeout around the process. Keep credentials out of command arguments.

```sh
ssh ultraviolet sudo env DEV_DIR=$dev DEV_ROOM=dev-myfeature-<rev> \
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

Confirm startup with `sudo grep -E 'starting worker|job-' $dev/agent.log`.
The LiveKit Agents 1.8.1 `connect --room <name>` mode still exists, behind a
deprecation warning. Use a unique room name for every test.

### 3. Mint a join token and join from a browser

Credentials go in environment variables rather than the `lk` command line.
Mint a short-lived token, open the printed browser URL on the tailnet, allow
the microphone, and talk.

```sh
ssh ultraviolet sudo bash -c 'set -euo pipefail
  keyfile=/run/agenix/livekit-keys
  export LIVEKIT_API_KEY=$(sed -n "s/^\([^:[:space:]]\+\)[[:space:]]*:.*$/\1/p" "$keyfile" | head -n1)
  export LIVEKIT_API_SECRET=$(sed -n "s/^[^:]\+:[[:space:]]*\(.\+\)$/\1/p" "$keyfile" | head -n1)
  token=$(lk token create --join --room dev-myfeature-<rev> \
          --identity acceptance --valid-for 45m --token-only)
  printf "https://meet.livekit.io/custom?liveKitUrl=wss%%3A%%2F%%2Fultraviolet.tail82223.ts.net%%3A7443&token=%s\n" "$token"'
```

The connect job stops after the room has no human participant for several
minutes or when the last human leaves. The SFU remains authoritative about
participants, and the worker log shows delegation, stream, and disconnect
events.

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
  for `session duration|delegation failed|voice session closing|ERROR|Traceback|disconnected` shows every turn
  land in real time. rtc_session errors during teardown are normal.


### 4. Clean up

```sh
ssh ultraviolet "sudo sh -c 'kill \$(cat $dev/agent.pid) 2>/dev/null; rm -rf $dev'"
ssh ultraviolet sudo systemctl start mentat-voice
```

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
