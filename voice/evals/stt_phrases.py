"""Phrase-garble check: stream common voice phrases through the production STT.

Each phrase is rendered once by ElevenLabs TTS, then streamed at real time
through the worker's own ``build_stt`` configuration, followed by trailing
silence. A phrase passes only when its transcript, normalized for case,
punctuation and accents, exactly matches one of its accepted spellings.
Private keyterms from ``MENTAT_VOICE_PRIVATE`` add one phrase each, so the
names transcription is biased toward are checked without entering the repo.

Run with the voice environment and ``ELEVENLABS_API_KEY``: ``just eval-stt``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import statistics
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from request import PRIVATE_CONTEXT_ENV, load_private_context


SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE // 100
LEADING_SILENCE_S = 0.3
TRAILING_SILENCE_S = 2.0
FINAL_GRACE_S = 3.0
TTS_MODEL = "eleven_multilingual_v2"
ENGLISH_VOICES = (
    "CwhRBWXzGAHq8TQ4Fs17",  # Roger, American male
    "EXAVITQu4vr4xnSDxMaL",  # Sarah, American female
    "JBFqnCBsd6RMkjVDRZzb",  # George, British male
    "cgSgspJ2msm6clMCkdW9",  # Jessica, American female
)
SPANISH_VOICE = "Zb72NBThGN2Rv3y1RSTT"  # peninsular male


@dataclass(frozen=True)
class Phrase:
    language: str
    text: str
    accepted: tuple[str, ...]


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, and join digit groups."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    plain = re.sub(r"['’]", "", plain)
    plain = re.sub(r"[^a-z0-9ñ ]+", " ", plain)
    plain = " ".join(plain.split())
    return re.sub(r"(?<=\d) (?=\d)", "", plain)


PHRASES = (
    Phrase("en", "Set a timer for five minutes.", (
        "set a timer for five minutes", "set a timer for 5 minutes")),
    Phrase("en", "Set an alarm for seven thirty tomorrow morning.", (
        "set an alarm for seven thirty tomorrow morning",
        "set an alarm for 730 tomorrow morning")),
    Phrase("en", "Stop the timer.", ("stop the timer",)),
    Phrase("en", "What's the weather like today?", ("whats the weather like today",)),
    Phrase("en", "Find a coffee shop near me.", ("find a coffee shop near me",)),
    Phrase("en", "Navigate to the nearest gas station.", (
        "navigate to the nearest gas station",)),
    Phrase("en", "Read me my latest text messages.", ("read me my latest text messages",)),
    Phrase("en", "Text my husband that I'm running late.", (
        "text my husband that im running late",)),
    Phrase("en", "Send a text to five five five, one two three, four five six seven.", (
        "send a text to 5551234567",
        "send a text to five five five one two three four five six seven")),
    Phrase("en", "Who was Alice Keck?", ("who was alice keck",)),
    Phrase("en", "Hey Mentat, what time is it in Tokyo?", (
        "hey mentat what time is it in tokyo",)),
    Phrase("en", "This is Josh Symonds.", ("this is josh symonds",)),
    Phrase("en", "Josh wants to know what's on his calendar tomorrow.", (
        "josh wants to know whats on his calendar tomorrow",)),
    Phrase("en", "Never mind, cancel that.", ("never mind cancel that", "nevermind cancel that")),
    Phrase("en", "Yes, send it.", ("yes send it",)),
    Phrase("en", "Switch to Spanish.", ("switch to spanish",)),
    Phrase("en", "Thanks, that's all.", ("thanks thats all",)),
    Phrase("es", "Pon un temporizador de diez minutos.", (
        "pon un temporizador de diez minutos", "pon un temporizador de 10 minutos")),
    Phrase("es", "¿Qué tiempo hace hoy en Madrid?", ("que tiempo hace hoy en madrid",)),
    Phrase("es", "Gracias, eso es todo.", ("gracias eso es todo",)),
)


def keyterm_phrases(keyterms: tuple[str, ...]) -> tuple[Phrase, ...]:
    """One English phrase per private keyterm, accepted only with its exact spelling."""
    return tuple(
        Phrase("en", f"I was talking about {term} earlier.", (
            normalize(f"I was talking about {term} earlier"),))
        for term in keyterms
    )


def verdict(phrase: Phrase, transcript: str) -> bool:
    return normalize(transcript) in phrase.accepted


async def render(http: Any, api_key: str, voice_id: str, text: str) -> bytes:
    async with http.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        params={"output_format": f"pcm_{SAMPLE_RATE}"},
        headers={"xi-api-key": api_key},
        json={"text": text, "model_id": TTS_MODEL},
    ) as response:
        response.raise_for_status()
        return await response.read()


async def transcribe(stt: Any, pcm: bytes) -> tuple[str, float | None]:
    """Stream ``pcm`` at real time; return the joined finals and last-final latency."""
    from livekit import rtc
    from livekit.agents.stt import SpeechEventType

    stream = stt.stream()
    finals: list[str] = []
    last_final_at: float | None = None

    async def read() -> None:
        nonlocal last_final_at
        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT and event.alternatives:
                text = event.alternatives[0].text.strip()
                if text:
                    finals.append(text)
                    last_final_at = time.monotonic()

    silence = bytes(FRAME_SAMPLES * 2)
    frame_bytes = FRAME_SAMPLES * 2
    chunks = [silence] * round(LEADING_SILENCE_S * 100)
    chunks += [
        pcm[offset:offset + frame_bytes].ljust(frame_bytes, b"\0")
        for offset in range(0, len(pcm), frame_bytes)
    ]
    speech_frames = len(chunks)
    chunks += [silence] * round(TRAILING_SILENCE_S * 100)

    reader = asyncio.create_task(read())
    speech_end = 0.0
    started = time.monotonic()
    for index, chunk in enumerate(chunks):
        delay = started + index / 100 - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        stream.push_frame(rtc.AudioFrame(chunk, SAMPLE_RATE, 1, FRAME_SAMPLES))
        if index + 1 == speech_frames:
            speech_end = time.monotonic()
    await asyncio.sleep(FINAL_GRACE_S)
    stream.end_input()
    try:
        await asyncio.wait_for(reader, FINAL_GRACE_S)
    except TimeoutError:
        reader.cancel()
    await stream.aclose()
    latency = None if last_final_at is None else max(0.0, last_final_at - speech_end)
    return " ".join(finals), latency


async def run(runs: int) -> int:
    from livekit.agents.utils import http_context

    import agent

    api_key = os.environ["ELEVENLABS_API_KEY"]
    private = load_private_context(os.environ.get(PRIVATE_CONTEXT_ENV))
    phrases = PHRASES + keyterm_phrases(private.keyterms)
    failures = 0
    latencies: list[float] = []
    async with http_context.open() as http:
        stts = {}
        for language in ("en", "es"):
            stt = agent.build_stt(api_key, private.keyterms)
            stt.update_options(secondary_languages=agent.stt_secondary_languages(language))
            stts[language] = stt
        english_index = 0
        for phrase in phrases:
            if phrase.language == "en":
                voice = ENGLISH_VOICES[english_index % len(ENGLISH_VOICES)]
                english_index += 1
            else:
                voice = SPANISH_VOICE
            pcm = await render(http, api_key, voice, phrase.text)
            for run_number in range(1, runs + 1):
                transcript, latency = await transcribe(stts[phrase.language], pcm)
                passed = verdict(phrase, transcript)
                failures += not passed
                if latency is not None:
                    latencies.append(latency)
                shown = "-" if latency is None else f"{latency:.2f}s"
                print(
                    f"{'PASS' if passed else 'FAIL'} [{phrase.language} {run_number}/{runs} {shown}] "
                    f"{phrase.text!r} -> {transcript!r}",
                    flush=True,
                )
    total = len(phrases) * runs
    print(f"\n{total - failures}/{total} phrases transcribed exactly")
    if latencies:
        print(
            f"final transcript after speech end: p50 {statistics.median(latencies):.2f}s, "
            f"max {max(latencies):.2f}s"
        )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream common voice phrases through the production STT."
    )
    parser.add_argument("--runs", type=int, default=1, help="transcriptions per rendered phrase")
    arguments = parser.parse_args()
    if arguments.runs < 1:
        parser.error("--runs must be positive")
    raise SystemExit(asyncio.run(run(arguments.runs)))


if __name__ == "__main__":
    main()
