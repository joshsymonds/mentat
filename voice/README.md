# Voice surface

The voice worker joins a LiveKit room and runs a single cascade: ElevenLabs
Scribe v2 realtime transcription with local Silero and semantic turn detection
captures each user turn, the
worker posts it to mentatd, and ElevenLabs v4 Turbo speaks the daemon's text
stream verbatim as it arrives, using per-sentence HTTP synthesis. There is no
local LLM reply or second voice. Interruptions stop speech and close the
in-flight backend response; an end-conversation result closes after the
goodbye, while follow-ups keep the room open. RoomIO starts before the LiveKit
connection so speech around the opening chime is captured, and the worker does
not greet on connect.

The worker reads `ELEVENLABS_API_KEY` from its environment for both
transcription and speech. Scribe commits a transcript after 1 s of silence;
the local turn detector decides when the turn ends but waits for that commit,
so the window must outlast ordinary mid-sentence pauses.
Scribe keeps English as its primary language, and a Spanish or interpreter
voice mode adds that language alongside it. The live eval caller still uses
`OPENAI_API_KEY` from the same secrets file to synthesize and check speech.
`MENTAT_VOICE_TTS_VOICE` selects the ElevenLabs voice ID; when unset, it uses
Rachel (`21m00Tcm4TlvDq8ikWAM`), an ElevenLabs premade voice.

The daemon owns memory, systems, lookups, and actions. The worker runs beside
mentatd and reaches it over loopback.

## Private context

The repository is public, so `persona.md` contains the voice and no private
facts. Deployment supplies a TOML file as `MENTAT_VOICE_PRIVATE`:

```toml
about = """
Who he is: ...one paragraph, folded into the worker instructions.
"""
# names transcription should expect; at most 50, each at most 20 characters
keyterms = ["Symonds"]
[pronunciations]
Symonds = "Sigh-monds"

# each key finishes the sentence "Josh is at ..."; radius_m is how close counts
[places.home]
lat = 47.6
lng = -122.3
radius_m = 150
```

Places and phone state shape the per-call context without adding a greeting.
The phone sends its time zone, last-known location, and whether Android Auto
has it in car mode with the token request;
mentatd stamps them on the token as participant attributes, and the worker
turns them into a one-paragraph call context in that call's instructions.

Unset means a development room has no private context. A set but unreadable
path fails the worker at startup. To run a dev room with it, add
`MENTAT_VOICE_PRIVATE=/run/agenix/mentat-voice-private` to the launch
environment in step 2.

## Testing branch code in a live room

Audio changes need a real room. For this cascade, the four scripted calls
below exercise the complete Scribe transcription → mentatd → ElevenLabs path.
Calls A, B, and D
should use `web_search`; call C still reaches mentatd and should answer without
searching. Compare each printed latency with the eight-second target.

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

Load the LiveKit and mentatd credentials from the environment file. Use
loopback URLs, a distinct health port, a writable home, and a bounded timeout
around the process. Set `DEV_PY` to the Python executable used by the service.
Keep credentials out of command arguments.

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
matched against the words of one ElevenLabs Scribe batch transcription of the
whole reply clip, with language auto-detected; output reports the start of the
word where the first match begins, relative to the caller's speech end. These four calls exercise current
lookup, price/source freshness, a stable fact from mentatd, and stale-summary
caution:

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
with Jane Austen, and D with a live F1 results page or an explicit statement of
uncertainty. All four calls go to mentatd; check `agent.log` for `web_search`
on A, B, and D and no `web_search` on C.

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

### Manual live evaluation

The opt-in evaluation uses isolated development services and a fake phone; it
never sends real SMS messages or acts on the real phone. During the run, the
production voice worker is paused. The remote cleanup handler restarts
`mentat-voice` on normal, error, and signal exits. The candidate worker records
the synthetic caller audio sent to STT as per-turn WAV files with matching
transcript sidecars. The eval transfers only those files into a private local
retained-evidence directory (mode 0700, files mode 0600); they are not written
to the repository. Production workers do not receive the recording setting.

Before starting, account for remote service access and API usage costs. The recipe
runs the default ten observations per scenario; preserve the positional `RUNS`
and `MODEL` arguments, then optionally pass a positive concurrency cap as the
third positional argument. Without it, the runner's named default cap of 8 applies,
fitting within ElevenLabs' limit of 9 concurrent requests.

```sh
just eval-voice
just eval-voice 2
just eval-voice 2 chatgpt/sol-fast 4
```

One candidate build and staging are shared by the batch. Each run gets its own
candidate daemon, fake phone, worker, ports, state, and logs; runs execute in
parallel up to the cap, with remaining runs queued. The JSON report records each
run's start and end times and peak concurrent-run count, plus total batch wall
time, so concurrency is visible alongside latency results. It also emits a
batch-level `capacity_failure_count`, each case's `capacity_failure_count`, and
`run_capacity_failures` entries with the run number and named source/cause.
Capacity counts include explicitly named candidate process-start or room-dispatch
failures (including voice-token dispatch HTTP 429/503 or request failures) and a
provider concurrency refusal found in that run's retained `voice.log`; repeated
copies of one provider refusal count once per run. Generic setup, cleanup, or
scenario-check failures do not count as capacity evidence.

Each case also carries `phantom_finals`, the per-run count of final transcripts
the worker logged with no user speech since the previous final (a phantom
line that can cut the voice off), and the report totals them in the
top-level `phantom_final_count`. A run whose retained voice log cannot be read
reports `null` for that run and adds nothing to the total.

The eval transcribes each reply clip with one Scribe batch request, so a 0.3 s
acknowledgment such as "Got it." keeps its words. Each returned word goes to
the pause window holding its start/end midpoint, or to the nearest window; the
windows alone set segment timing. A clip whose transcription returns text
without word timestamps fails as "Transcription rejected reply clip
(missing_word_timestamps)" with the text kept in the failure evidence. Scribe
HTTP 4xx rejections and concurrency 429 refusals are likewise named for the
reply clip.

On normal exit, errors, or SIGINT/SIGTERM, the runner stops launching work,
finishes stopping all candidate processes, then restores production
`mentat-voice`. A later setup reaps leftovers only when their recorded batch
owner is positively dead; uncertain ownership is left untouched.

To inspect the contracted scenarios without starting services, run:

```sh
python3 -m voice.evals.runner eval --list
```

To run only some scenarios, repeat `--scenario NAME`. Names are checked against
the scenario list below (an unknown name is an error), repeated names select a
scenario once, and selected scenarios run in registry order. `--runs` still sets
the repetitions of each selected scenario. Without the flag every scenario
runs; with `--list`, known names still print the full listing. For example, the
Spanish subset, eight runs each:

```sh
python3 -m voice.evals.runner eval --live --runs 8 \
  --scenario spanish-language-switch --scenario spanish-interpreter
```

The current scenarios are `timer-300-seconds`, `equivalent-alarm`,
`place-search-navigation`, `sms-say-back-yes`, `sms-correction-new-yes`,
`alice-keck-context-chain`, `spanish-language-switch`,
`spanish-interpreter`, `phone-first-line`, and `barge-in-long-reply`. The
Spanish scenario switches from English to Spanish,
back to English, then to Spanish again; its pass verdict requires the recorded
input transcripts, mode transitions, and synthesis voices to agree with each
scripted turn. The report includes the first Spanish voice lookup duration. The
interpreter scenario translates six English and Spanish turns for a generic
gardener, including a timer imperative and a quoted stop-translation
phrase; neither is acted on as a phone command or mode change.

Two capture modes serve the first-line and barge-in scenarios.
`phone-first-line` runs with `--preconnect-first-line`: the opening second of
line 1 goes to the worker as the phone's pre-connect buffer while line 1 also
streams live from 0.3 s, so the two overlap by 0.7 s the way the Pixel's do, and
the room must still hear the line once. In `barge-in-long-reply`,
`--barge-in TURN:SECONDS` (one-based TURN, repeatable) starts that line SECONDS
after the agent's reply begins; the scenario speaks its second line 15 seconds
into a long story to check that the interruption commits during the agent's
speech. Both scenarios set `exact_caller_stt`:
each turn's input STT sidecar must match its scripted line, so a misheard or
duplicated first line, or a missing sidecar, fails that turn, and the
observation records the sidecars as `caller_stt`. The R4/R5 command runs both
scenarios eight times on Opus 5.5:

```sh
MENTAT_VOICE_MODEL=claude-opus-5-5 python3 -m voice.evals.runner eval --live --runs 8 \
  --scenario phone-first-line --scenario barge-in-long-reply
```

The eval prints a JSON report to stdout and exits nonzero if any scenario,
observation-count, or latency gate fails. Keep the report when diagnosing a
failure; a successful process exit means every strict gate passed.

### Semantic reply-judge qualification

`just eval-judge` is a separate, opt-in live call to the semantic judge. It
requires `TYPESAFE_API_KEY` and sends retained-reply, scripted, and deliberate
scenario-corruption fixtures to the judge in three concurrent sweeps. Each
fixture/run gets a new judge instance so cached verdicts cannot cross fixtures
or runs. Every fixture names a live scenario and one-based turn; its complete
question set and verified-recipient context are derived through the same
runtime scenario path used by the voice eval. The corpus has a correct and a
known-wrong fixture for every runtime turn, including each Spanish-switch and
interpreter turn, corrected and inherited SMS recipients, navigation, and
read-back/confirmation distinctions. A correct shape no retained reply
exhibits (an already-confirmed recipient called "the same number") is a
`scripted` fixture with its reason; scripted fixtures are scored apart from
retained replies and must clear the same bar on their own.

The command prints one JSON report with per-run and per-family correct/wrong
counts, each fixture's built question and probability/verdict evidence, and
unavailable outcomes. The voice eval report preserves each captured turn's
judge questions, verdicts, probabilities, and unavailable reason alongside its
transcript evidence; an unavailable judge result fails that turn as a product
failure while earlier turns and phone-command evidence remain in the report.
Every sweep must pass at least 95% of retained correct
fixtures and, separately, of scripted correct fixtures,
judge every known-wrong fixture no, and have zero unavailable results; any
failed bar returns a nonzero exit. Offline coverage is `just test-voice`; it
uses `ScriptedJudge` and does not call the live service.
