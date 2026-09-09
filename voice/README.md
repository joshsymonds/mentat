# Voice surface

Front, the LiveKit voice agent: `agent.py` joins a LiveKit room, runs local
turns through LiveKit Cloud STT/TTS, and consults mentatd (`ask_mentat`) for
anything with memory or tools. Deployed on ultraviolet as the `mentat-voice`
systemd unit (see `nix/module.nix`); the SFU, mentatd, and the agent all share
that host, so the agent talks to both over loopback.

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

## Phone control

The voice agent reaches the Android phone through two LiveKit remote procedure
call (RPC) methods. The phone registers `mentat.command` and `mentat.location`
when it joins the room. The phone is the trust boundary: the front sends only
the closed `kind` values below, and never supplies an Android action,
component, extras, or an arbitrary scheme.

`mentat.command` accepts one JSON object with one of these payloads:

```json
{"kind":"navigate","name":"Trader Joe's","address":"123 Main St","place_id":"ChIJ...","lat":45.5,"lng":-122.6}
{"kind":"dial","number":"503-555-0199"}
{"kind":"sms","number":"503-555-0199","body":"I'll be ten minutes late"}
{"kind":"alarm","hour":7,"minute":30,"label":"wake up"}
{"kind":"timer","seconds":300,"label":"tea"}
{"kind":"open","url":"https://example.com"}
```

The optional `label` may be omitted for `alarm` and `timer`. `navigate` opens
Google Maps with an `ACTION_VIEW` directions URL whose destination is
`name + ", " + address`, plus `destination_place_id`,
`travelmode=driving`, and `dir_action=navigate`, targeted to the Google Maps
package. The phone maps the other kinds to their corresponding typed Android
intents. It returns `{"ok":true}` after launching. Dial and SMS open their
prefilled apps for a tap-to-confirm action; they do not place a call or send a
message directly. `open` accepts absolute `http` and `https` URLs only.

`mentat.location` accepts `{}` and returns a fresh location when possible:

```json
{"lat":45.5,"lng":-122.6,"accuracy_m":12.0,"age_s":1.2}
```

The location policy accepts a current fused fix no older than five seconds. If
that is unavailable, it accepts last-known location no older than ten minutes.
It returns error 1602 when neither is available or no location permission is
granted. Location is used for the current search only and is not persisted.

RPC errors are handled as spoken outcomes by the front:

| Code | Meaning | Front behavior |
| --- | --- | --- |
| 1600 | Invalid or refused command, including an unknown kind, missing or invalid field, or a non-HTTP(S) link | Return `PHONE_REFUSED` and say it could not carry out that phone action. |
| 1601 | No phone activity can handle the requested intent | Return `PHONE_REFUSED` and say the phone could not open that action. |
| 1602 | Location is unavailable or permission is denied | Return `LOCATION_UNAVAILABLE`; ask roughly where Josh is, then search again with that locality. |
| 1603 | The assist screen is not in front | Return `PHONE_NOT_IN_FRONT`; tell Josh to tap the side button first. |

The phone launches an action only while the Mentat talk screen is visible. It
never queues or retries a command after another app covers that screen. Tap the
side button first, then repeat the request. A browser-only room reports the
phone-unreachable result instead of raising an error.

The Places search key is optional for local development and lives in the voice
unit's agenix environment file. Create a billed Google Cloud project and a
Places-only key from a shell authenticated as Josh:

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
