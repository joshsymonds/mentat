"""LiveKit voice agent: the front that talks, with mentat as the brain behind.

The voice in the room is its own fast model (Luna, over LiveKit Inference)
wearing the persona in voice/persona.md. It owns the conversation: banter,
opinions and ordinary knowledge it answers itself, at conversation speed. For
anything that touches Josh's life — his memory, calendar, house, files, the
tools that act on them — or anything that needs real thinking, it calls the
ask_mentat tool, which runs one mentatd turn and speaks the daemon's answer
verbatim. mentatd is no longer the voice; it is the deep brain the voice
consults.

The thinking parts live next door and are unit-tested offline: voice/stream.py
turns mentat's NDJSON into speakable text, voice/request.py builds the turn
request and the consult envelope. What is left here is the part that cannot be
tested without a livekit runtime — the pipeline and the session's earcon
wiring — so it stays deliberately thin, and it is verified against a real
room rather than in CI.

Run as `python agent.py start`; livekit-agents ships no console script. The
SDK reads LIVEKIT_URL and the LIVEKIT_API_*/LIVEKIT_INFERENCE_API_* credentials
straight from the environment.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterable, Callable, Mapping
from pathlib import Path
from typing import Any, Literal

import aiohttp
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    AudioConfig,
    BackgroundAudioPlayer,
    ConversationItemAddedEvent,
    JobContext,
    RunContext,
    get_job_context,
    ModelSettings,
    StopResponse,
    WorkerOptions,
    function_tool,
    inference,
    llm,
)
from livekit.agents.voice import room_io
from livekit.plugins import dtln, silero
from livekit.agents.tts import _provider_format

from phone import PhoneActions, RpcFailure
from request import (
    PRIVATE_CONTEXT_ENV,
    EndingPolicy,
    PrivateContext,
    consult_envelope,
    conversation_advanced,
    count_user_messages,
    recent_turns,
    load_private_context,
    run_close_sequence,
    split_persona,
    turn_latency,
    turn_request,
    with_private_context,
    without_last_user_message,
)
from stream import Respeller, TurnError, TurnStream

logger = logging.getLogger("mentat.voice")

DEFAULT_MENTAT_URL = "http://127.0.0.1:8484"

# The front's own voice, named here rather than left as a literal at the
# constructor because voice/evals scores this exact model against this exact
# persona. An eval measuring a model the room does not hear would be worse
# than no eval, so the two read the name from one place.
FRONT_MODEL = "openai/gpt-5.6-luna"

# Words Flux mishears on its own — "Mentat" came back as "man, uh". Keyterm
# prompting boosts recall of exactly these; Deepgram caps the list at 100
# terms totalling 1200 characters, and the terms are plain words, no weights.
# Only the public vocabulary lives here; the people and places come from the
# private context (request.PrivateContext), which is never in the repository.
# Cartesia's Daniel, from the Sonic 3.6 recommended voices.
TTS_VOICE = "47c38ca4-5f35-497b-b1a3-415245fb35e1"

# Appended to the expressive-mode tag instructions. The default template
# already lists the tags; this is the persona's rule for when to reach for one.
DELIVERY_RULE = (
    "Choose each label from what the words already carry. A plain answer is "
    "neutral or content, not forced brighter; go to excited, sympathetic, "
    "joking, or apologetic only when the line itself is that. A pause goes "
    "before the part that matters and nowhere else. The delivery should shift "
    "the way a real voice does across a conversation, not perform. One thing "
    "the tag list above leaves out: this voice does laugh. Write the literal "
    "text [laughter] where the laugh goes, as its own word, and it is voiced. "
    "It is rare — only when something is actually funny, never at your own "
    "line, and never as a substitute for saying the thing."
)

# Under expressive mode the SDK batches sentences up to the provider's chunk
# size before the first TTS request, to keep prosody continuous across a turn.
# Cartesia's entry is 400 characters — longer than most of Luna's replies, so
# first audio would wait for the whole turn. Cartesia's emotion tags are per
# sentence anyway, so one sentence at a time loses nothing and keeps the
# time-to-first-audio the journal already measures. The table is private to
# agents 1.6.10 (pinned in nix/voice-env.nix); the assert makes a bump that
# moves it fail at worker start rather than silently batch again.
EXPRESSIVE_BATCH_CHARS = 120
assert "cartesia" in _provider_format._MAX_INPUT_LEN, "agents SDK moved the TTS chunking table"
_provider_format._MAX_INPUT_LEN["cartesia"] = EXPRESSIVE_BATCH_CHARS

STT_KEYTERMS = [
    "Mentat",
    "Luna",
    "Josh",
    "ultraviolet",
    "gnomon",
    "vermissian",
    "ninuan",
    "bluedesert",
    "echelon",
    "Home Assistant",
    "Tailscale",
    "LiveKit",
]

# A consult legitimately runs for minutes while the daemon uses tools; only the
# connect and the gap between chunks are bounded.
TIMEOUT = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)

# The persona and the sound ship in the same directory as this file — the nix
# fileset in nix/module.nix puts both there — so they are found relative to it
# rather than through configuration nobody would ever set differently.
HERE = Path(__file__).parent
PERSONA_PATH = HERE / "persona.md"
EARCON_PATH = HERE / "assets" / "earcon.wav"

#: The shortest gap between two bings. A turn needs acknowledging once; the
#: session re-enters "thinking" for the same turn's plumbing (segmented
#: replies, fast flaps), and overlapping bells beat into a buzz.
EARCON_MIN_INTERVAL_S = 2.0

# Never spoken. ctx.update hands control back to the front with this as the
# tool's synthetic return, and the front says its own holding line from it —
# so this is written as an instruction to a model, not as speech. Speaking a
# fixed line here instead would make every consult in a call sound identical.
# The holding line is instructed here and nowhere else: persona.md asking for
# one too would earn two near-identical lines per consult, and this is the
# path the framework guarantees runs.
CONSULT_CUE = (
    "Sent to Mentat. Say one short line in your own voice about going to "
    "check — make it natural and different every time; it is you thinking out "
    "loud for a second, not an announcement. Then stop and wait. The answer "
    "will be spoken aloud for you the moment it lands, so do not answer the "
    "question yourself."
)

# Returned in place of an answer when the consult fails. Returning a string
# (rather than saying one) lets the front break the news in its own words and
# in context — it may already have said it was checking.
CONSULT_FAILED = (
    "Mentat could not be reached, so there is no answer to that question. "
    "Tell Josh briefly, and offer to try again."
)

#: Prepended when the caller kept talking during the wait. A deep answer can
#: land a minute after its question, and dropping it into a conversation that
#: has moved on, unmarked, is a non sequitur.
REORIENTATION_PREFIX = "About your earlier question — "


def load_persona(path: Path = PERSONA_PATH) -> tuple[str, str]:
    """The front's instructions and its voice card, read off disk."""
    return split_persona(path.read_text())


class FrontAgent(Agent):
    """The voice in the room: answers what it can, consults mentat for the rest."""

    def __init__(
        self,
        *,
        instructions: str,
        voice_card: str,
        room_name: str,
        mentat_url: str,
        pronunciations: Mapping[str, str],
        room: rtc.Room | None = None,
        ending_policy: EndingPolicy | None = None,
        ending_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(instructions=instructions)
        self._voice_card = voice_card
        self._room_name = room_name
        self._mentat_url = mentat_url
        self._pronunciations = pronunciations
        self._ending_policy = ending_policy or EndingPolicy()
        self._ending_changed = ending_changed or (lambda: None)

        async def perform_rpc(
            identity: str, method: str, payload: str, timeout: float
        ) -> str:
            if room is None:
                raise RpcFailure(1401, "no room")
            try:
                return await room.local_participant.perform_rpc(
                    destination_identity=identity,
                    method=method,
                    payload=payload,
                    response_timeout=timeout,
                )
            except rtc.RpcError as error:
                raise RpcFailure(int(error.code), str(error.message)) from error

        async def post_json(
            url: str, headers: Mapping[str, str], body: Mapping[str, Any]
        ) -> Any:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as http:
                async with http.post(url, headers=headers, json=body) as response:
                    if response.status != 200:
                        raise aiohttp.ClientResponseError(
                            response.request_info,
                            response.history,
                            status=response.status,
                        )
                    return await response.json()

        self._phone_actions = PhoneActions(
            perform_rpc,
            post_json,
            lambda: room.remote_participants if room is not None else (),
            os.environ.get("MENTAT_PLACES_API_KEY", ""),
        )

    @function_tool()
    async def not_for_me(self, ctx: RunContext) -> None:
        """What just came through was not said to you: someone else in the
        room, a pet, a TV, a fragment of noise, or a stray line that belongs to
        no conversation you are having. Call this instead of replying. The
        line is erased and you say nothing — never ask what it was."""
        # The line is already in the context, appended before this turn's
        # generation started; the reply it would get is stopped below before
        # anything is generated, so this one removal leaves no trace of it.
        pruned = self.chat_ctx.copy()
        pruned.items = without_last_user_message(pruned.items)
        await self.update_chat_ctx(pruned)
        logger.info("not_for_me: a turn was erased as not addressed to the front")
        raise StopResponse()

    @function_tool()
    async def end_conversation(
        self,
        ctx: RunContext,
        reason: Literal["signoff", "done"],
        farewell: str = "",
    ) -> None:
        """End the conversation after a sign-off or a completed request."""
        if self._ending_policy.consult_in_flight:
            return None

        self._ending_policy.end_requested(reason)
        logger.info("end_conversation: reason=%s", reason)
        await ctx.wait_for_playout()
        if self._ending_policy.pending_reason != reason:
            self._ending_changed()
            return None

        if reason == "signoff":
            handle = ctx.session.say(farewell)
            await handle
            delivered = not handle.interrupted and handle.exception() is None
        else:
            handle = ctx.speech_handle
            delivered = not handle.interrupted and handle.exception() is None

        if self._ending_policy.playout_finished(delivered) == "close":
            ctx.session.shutdown()
        self._ending_changed()
        return None

    async def tts_node(
        self, text: AsyncIterable[str], model_settings: ModelSettings
    ):  # type: ignore[override]  # the SDK's own signature; return type is its AudioFrame stream
        """The default TTS path, fed respelled text so names are said right."""

        async def respelled() -> AsyncIterable[str]:
            respeller = Respeller(self._pronunciations)
            async for chunk in text:
                if out := respeller.feed(chunk):
                    yield out
            if out := respeller.flush():
                yield out

        async for frame in Agent.default.tts_node(self, respelled(), model_settings):
            yield frame

    @function_tool(
        # CANCELLABLE puts the framework's cancel tool in front of the model,
        # so "never mind" during a minute-long consult actually drops it.
        # reject keeps a second question from starting a second daemon turn on
        # top of the first; the front is told to offer the swap instead.
        flags=llm.ToolFlag.CANCELLABLE,
        on_duplicate="reject",
    )
    # The knobs are Literals, not strs: the framework renders each as a
    # JSON-schema enum and validates the call against it, so a hallucinated
    # "medium" or "opus" is rejected at parse time instead of travelling into
    # the daemon's turn request, which takes these values on trust.
    async def ask_mentat(
        self,
        ctx: RunContext,
        question: str,
        effort: Literal["low", "high"] = "low",
        model: Literal["sonnet", "fable"] = "sonnet",
    ) -> str | None:
        """Ask Mentat, the deep brain behind you, and let it answer Josh.

        Mentat holds Josh's memory, calendar, home, files and messages, and
        the tools that act on them, and it can think for as long as a question
        deserves. Send it anything about his life or his systems, anything
        that needs a lookup or an action, anything about the current state of
        the world, and any question where a quick answer would really be a
        guess. Use find_places and navigate_to for navigation, dial for
        dialing, send_text for texting, set_alarm for alarms, set_timer for
        timers and open_link for links instead of this tool. Place questions
        such as hours, reviews or distance, and
        contact lookups, still belong here. Do not answer these from your own
        head.

        Ask in full sentences, and carry over whatever context the question
        needs to stand on its own: Mentat cannot hear Josh, it only reads what
        you send.

        The answer is spoken aloud to Josh automatically, in your voice, as
        soon as it arrives. Do not repeat it, summarize it or introduce it —
        just carry on from there.

        Args:
            question: The question to ask, self-contained.
            effort: How hard Mentat should think. "low" for a lookup or a
                simple fact; "high" for a genuinely hard, important or
                open-ended question. High effort is slower, so spend it where
                it earns the wait.
            model: "sonnet" for everyday questions, "fable" for the ones that
                deserve the best thinking available. Pair "fable" with "high".
        """
        # Counted before control goes back to the front, so the holding line
        # it is about to speak cannot be mistaken for the caller talking on.
        baseline = count_user_messages(self.chat_ctx.items)
        self._ending_policy.consult_started()
        self._ending_changed()

        try:
            await ctx.update(CONSULT_CUE)
            try:
                envelope = consult_envelope(
                    self._voice_card,
                    # No rolling summary is kept yet — deferred by design; the
                    # envelope drops the section entirely when it is blank.
                    summary="",
                    last_turns=recent_turns(self.chat_ctx.items),
                    question=question,
                )
                answer = await self._consult(envelope, effort, model)
                if not answer:
                    # A turn that spent itself on tools can finish cleanly with
                    # no text at all. Saying an empty string would leave the
                    # caller listening to nothing, which sounds exactly like a
                    # hang — so an answer with nothing in it is a failed consult,
                    # and gets the same apology as one that never arrived.
                    raise TurnError("turn produced no speakable text")
            except (TurnError, aiohttp.ClientError, TimeoutError) as err:
                logger.warning("consult failed: %s", err)
                return CONSULT_FAILED

            prefix = (
                REORIENTATION_PREFIX
                if conversation_advanced(self.chat_ctx.items, baseline)
                else ""
            )
            spoken = prefix + answer
            handle = ctx.session.say(spoken)
            await handle
            delivered = not handle.interrupted and handle.exception() is None
            self._ending_policy.consult_answered(answer, delivered)
            self._ending_changed()
            # Nothing left to say. Returning None after an update is what stops
            # the framework generating a reply on top of the answer just spoken.
            return None
        finally:
            self._ending_policy.consult_finished()
            self._ending_changed()

    @function_tool
    async def find_places(
        self, ctx: RunContext, query: str, locality: str = ""
    ) -> str:
        """Search places near the phone for the place he named.

        Pass locality only after LOCATION_UNAVAILABLE, using the place he named.
        A spoken description gives up to three matches; LOCATION_UNAVAILABLE,
        PHONE_UNREACHABLE, PLACES_UNCONFIGURED, PLACES_FAILED and NO_RESULTS are
        token-prefixed instructions to act on, never text to read aloud verbatim.
        """
        return await self._phone_actions.find_places(query, locality)

    @function_tool
    async def navigate_to(self, ctx: RunContext, choice: int) -> str:
        """Navigate to the 1-based choice in the most recent find_places result.

        A confirmation means navigation started; token-prefixed NO_CANDIDATES,
        PHONE_NOT_IN_FRONT, PHONE_UNREACHABLE or PHONE_REFUSED means it did not.
        """
        return await self._phone_actions.navigate_to(choice)

    @function_tool
    async def dial(self, ctx: RunContext, number: str) -> str:
        """Open the dialer with the number prefilled without placing a call.

        A token-prefixed return means the dialer did not open; do not read it aloud.
        """
        return await self._phone_actions.dial(number)

    @function_tool
    async def send_text(self, ctx: RunContext, number: str, body: str) -> str:
        """Open messaging with a drafted message, which is not sent automatically.

        A token-prefixed return means the message was not opened; do not read it aloud.
        """
        return await self._phone_actions.send_text(number, body)

    @function_tool
    async def set_alarm(
        self, ctx: RunContext, hour: int, minute: int, label: str = ""
    ) -> str:
        """Set an alarm silently on the phone without opening a confirmation screen.

        A token-prefixed return means the alarm was not set; do not read it aloud.
        """
        return await self._phone_actions.set_alarm(hour, minute, label)

    @function_tool
    async def set_timer(
        self, ctx: RunContext, minutes: int, seconds: int = 0, label: str = ""
    ) -> str:
        """Start a timer on the phone without opening a confirmation screen.

        A token-prefixed return means the timer did not start; do not read it aloud.
        """
        return await self._phone_actions.set_timer(minutes, seconds, label)

    @function_tool
    async def open_link(self, ctx: RunContext, url: str) -> str:
        """Open an absolute link in the phone's browser or another suitable app.

        A token-prefixed return means the link did not open; do not read it aloud.
        """
        return await self._phone_actions.open_link(url)

    async def _consult(self, envelope: str, effort: str, model: str) -> str:
        """One mentatd turn, collected whole rather than spoken as it streams.

        The answer is delivered as a single piece of speech after the fact, so
        the chunks are joined instead of yielded: speaking them as they arrive
        would talk over the front's own holding line and start mid-thought.
        """
        stream = TurnStream()
        chunks: list[str] = []
        async with aiohttp.ClientSession(timeout=TIMEOUT) as http:
            async with http.post(
                f"{self._mentat_url}/v1/conversation",
                json=turn_request(self._room_name, envelope, effort, model),
            ) as response:
                if response.status != 200:
                    raise TurnError(f"daemon answered HTTP {response.status}")
                async for chunk in response.content.iter_any():
                    chunks.extend(stream.feed(chunk))
        if not stream.done:
            # The body ended without a done event — mentatd restarted mid-turn,
            # so the socket closed cleanly and aiohttp raises nothing. Speaking
            # what arrived would hand the caller half an answer with nothing to
            # signal that it was half.
            raise TurnError("stream ended without done")
        return "".join(chunks).strip()


def log_turn_metrics(session: AgentSession) -> None:
    """Put each turn's latency in the journal, where nothing else records it.

    Without this, a complaint that "it felt slow" has no evidence behind it;
    with it, one line per turn says which stage spent the time. The judgment —
    what counts as a turn, and how its two halves of numbers are joined — is
    `turn_latency` next door, where it is pinned offline; this is the
    subscription around it.

    The numbers arrive on the chat items themselves rather than through the
    metrics_collected event, which agents 1.6.10 deprecates and warns about on
    every subscription.
    """
    pending: Mapping[str, Any] = {}

    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        nonlocal pending
        item = ev.item
        # The event also carries agent handoffs, which have no metrics and no
        # role to switch on.
        if not isinstance(item, llm.ChatMessage):
            return
        line, pending = turn_latency(pending, item.role, item.metrics)
        if line is not None:
            logger.info("turn latency %s", line)


def prewarm(proc: agents.JobProcess) -> None:
    """Load the VAD once per worker process, before a job is ever assigned.

    Loading it inside the entrypoint would put model load on the critical path
    of the first utterance in every room, after the caller is already
    listening; the worker warms up idle instead.
    """
    proc.userdata["vad"] = silero.VAD.load()
    # Loaded here rather than per job so a broken file fails the worker at
    # start, once, instead of every room at its first utterance.
    private = load_private_context(os.environ.get(PRIVATE_CONTEXT_ENV))
    proc.userdata["private"] = private
    logger.info(
        "private context: about=%d words, keyterms=%d, pronunciations=%d",
        len(private.about.split()),
        len(private.keyterms),
        len(private.pronunciations),
    )


async def entrypoint(ctx: JobContext) -> None:
    """Serve one room for as long as it lives."""
    await ctx.connect()

    instructions, voice_card = load_persona()
    private: PrivateContext = ctx.proc.userdata["private"]
    instructions = with_private_context(instructions, private)
    ending_policy = EndingPolicy()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # The docs' two-argument form is the one agents 1.6.10 takes:
        # "deepgram/flux-general" is a model literal it knows and `language`
        # is a separate keyword, not a ":en" suffix on the model string.
        stt=inference.STT(
            "deepgram/flux-general",
            language="en",
            extra_kwargs={"keyterm": [*STT_KEYTERMS, *private.keyterms]},
        ),
        # Luna is the front's own voice, fast enough to hold a conversation
        # with the depth delegated to ask_mentat.
        #
        # A bare constructor, for two separate reasons. Luna does not reason,
        # and the gateway takes reasoning_effort only to ignore it, so passing
        # one would be a knob that does nothing and a comment that lies. And
        # max_completion_tokens is an outright trap here: a small cap is spent
        # on the model's internal tokens before any text is generated, and the
        # turn returns empty content with finish_reason "length" and no error
        # at all — a silent mute. Brevity is asked for in persona.md instead,
        # where it costs nothing.
        #
        # Built once and held for the session's life: the instance owns an
        # httpx connection pool, and keeping it warm is most of the difference
        # between a ~0.85s first token and a wait the caller can hear.
        llm=inference.LLM(FRONT_MODEL),
        # Sonic 3.6: Cartesia's current GA model, routed by the gateway from the
        # string alone — agents 1.6.10 predates it, so it is not a known literal.
        tts=inference.TTS("cartesia/sonic-3.6", voice=TTS_VOICE),
        # Luna marks up her own delivery: the session tells her which tags the
        # TTS renders (emotion, speed, volume, pauses) and the appended rule
        # keeps them rare — a tag is a shift in the voice, not decoration.
        expressive={"tts_instructions_append": DELIVERY_RULE},
        # Flux emits end-of-turn itself, so waiting out a silence window after
        # it would only add latency to every reply. (Spelled as turn_handling
        # rather than the turn_detection/min_endpointing_delay arguments: those
        # are deprecated in agents 1.6.10 and migrate to exactly this.)
        turn_handling={
            "turn_detection": "stt",
            "endpointing": {"min_delay": 0.0},
        },
    )

    log_turn_metrics(session)

    # The earcon is not wired as thinking_sound: the SDK replays that on every
    # entry into the thinking state and only guards overlap while a play is in
    # flight. The debounced handler below owns the bing instead.
    background = BackgroundAudioPlayer()
    # Before session.start: the player publishes its own track and watches the
    # session for state changes, and a consult can start on the first utterance.
    await background.start(room=ctx.room, agent_session=session)

    timer_task: asyncio.Task[None] | None = None

    async def _wait_for_deadline(seconds: float) -> None:
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        if ending_policy.elapsed(seconds) == "close":
            session.shutdown()

    def _rearm_timer() -> None:
        nonlocal timer_task
        if timer_task is not None:
            timer_task.cancel()
            timer_task = None
        if ending_policy.deadline is not None:
            timer_task = asyncio.create_task(_wait_for_deadline(ending_policy.deadline))

    cleanup_task: asyncio.Task[None] | None = None

    async def _delete_room() -> None:
        await ctx.delete_room()

    async def _shutdown_job() -> None:
        get_job_context().shutdown()

    @session.on("close")
    def _on_close(_: Any) -> None:
        nonlocal cleanup_task, timer_task
        if timer_task is not None:
            timer_task.cancel()
            timer_task = None
        if cleanup_task is None:
            logger.info("voice session closing")
            cleanup_task = asyncio.create_task(
                run_close_sequence(
                    background.aclose,
                    _delete_room,
                    _shutdown_job,
                    logger.error,
                )
            )

    async def _await_cleanup() -> None:
        if cleanup_task is not None:
            await cleanup_task

    ctx.add_shutdown_callback(_await_cleanup)

    @session.on("user_state_changed")
    def _on_user_state(ev: Any) -> None:
        if ev.new_state == "speaking":
            ending_policy.user_spoke()
            _rearm_timer()

    # One bing per turn, debounced: the acknowledgment means "I heard you",
    # and a turn only needs hearing once — re-entries into thinking within the
    # window are the same turn's plumbing, not a new utterance.
    last_bing = 0.0

    @session.on("agent_state_changed")
    def _bing(ev: AgentStateChangedEvent) -> None:
        nonlocal last_bing
        if ev.new_state == "listening":
            ending_policy.agent_listening()
            _rearm_timer()
        elif ev.new_state in {"thinking", "speaking"}:
            ending_policy.agent_busy()
            _rearm_timer()
        now = time.monotonic()
        if ev.new_state == "thinking" and now - last_bing >= EARCON_MIN_INTERVAL_S:
            last_bing = now
            background.play(AudioConfig(str(EARCON_PATH)))

    await session.start(
        agent=FrontAgent(
            pronunciations=private.pronunciations,
            instructions=instructions,
            voice_card=voice_card,
            room_name=ctx.room.name,
            mentat_url=os.environ.get("MENTAT_URL", DEFAULT_MENTAT_URL),
            room=ctx.room,
            ending_policy=ending_policy,
            ending_changed=_rearm_timer,
        ),
        room=ctx.room,
        # Self-hosted noise suppression on the inbound track, one stateful
        # instance per session as the plugin requires. Cleaner audio into
        # Flux means fewer phantom words becoming turns in the first place.
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=dtln.noise_suppression(),
            ),
        ),
    )


if __name__ == "__main__":
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # The worker's own health/debug HTTP server. Loopback because
            # nothing off-host consumes it, and off the SDK default 8081,
            # which collides with atticd on ultraviolet.
            host="127.0.0.1",
            port=int(os.environ.get("MENTAT_VOICE_HTTP_PORT", "8482")),
        )
    )
