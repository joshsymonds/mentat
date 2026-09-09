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

import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import aiohttp
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    AudioConfig,
    BackgroundAudioPlayer,
    ConversationItemAddedEvent,
    JobContext,
    RunContext,
    WorkerOptions,
    function_tool,
    inference,
    llm,
)
from livekit.plugins import silero

from request import (
    consult_envelope,
    conversation_advanced,
    count_user_messages,
    recent_turns,
    split_persona,
    turn_latency,
    turn_request,
)
from stream import TurnError, TurnStream

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
    ) -> None:
        super().__init__(instructions=instructions)
        self._voice_card = voice_card
        self._room_name = room_name
        self._mentat_url = mentat_url

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
        guess. Do not answer those from your own head.

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
                # A turn that spent itself on tools can finish cleanly with no
                # text at all. Saying an empty string would leave the caller
                # listening to nothing, which sounds exactly like a hang — so
                # an answer with nothing in it is a failed consult, and gets
                # the same apology as one that never arrived.
                raise TurnError("turn produced no speakable text")
        except (TurnError, aiohttp.ClientError, TimeoutError) as err:
            logger.warning("consult failed: %s", err)
            return CONSULT_FAILED

        prefix = (
            REORIENTATION_PREFIX
            if conversation_advanced(self.chat_ctx.items, baseline)
            else ""
        )
        await ctx.session.say(prefix + answer)
        # Nothing left to say. Returning None after an update is what stops
        # the framework generating a reply on top of the answer just spoken.
        return None

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


async def entrypoint(ctx: JobContext) -> None:
    """Serve one room for as long as it lives."""
    await ctx.connect()

    instructions, voice_card = load_persona()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        # The docs' two-argument form is the one agents 1.6.10 takes:
        # "deepgram/flux-general" is a model literal it knows and `language`
        # is a separate keyword, not a ":en" suffix on the model string.
        stt=inference.STT(
            "deepgram/flux-general",
            language="en",
            extra_kwargs={"keyterm": STT_KEYTERMS},
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
        tts=inference.TTS("cartesia/sonic-3.6"),
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

    # One bing per turn, debounced: the acknowledgment means "I heard you",
    # and a turn only needs hearing once — re-entries into thinking within the
    # window are the same turn's plumbing, not a new utterance.
    last_bing = 0.0

    @session.on("agent_state_changed")
    def _bing(ev: AgentStateChangedEvent) -> None:
        nonlocal last_bing
        now = time.monotonic()
        if ev.new_state == "thinking" and now - last_bing >= EARCON_MIN_INTERVAL_S:
            last_bing = now
            background.play(AudioConfig(str(EARCON_PATH)))

    await session.start(
        agent=FrontAgent(
            instructions=instructions,
            voice_card=voice_card,
            room_name=ctx.room.name,
            mentat_url=os.environ.get("MENTAT_URL", DEFAULT_MENTAT_URL),
        ),
        room=ctx.room,
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
