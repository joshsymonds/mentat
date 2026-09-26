"""Generate the voice surface's sound asset.

The voice earcon and Android listening chime are checked in as .wav files,
but authored here rather than in an editor: a reviewer cannot diff a binary,
and an opaque sound asset can never be audited. Generation is deterministic —
pure arithmetic, no randomness, no timestamps — so re-running this script
reproduces the committed bytes exactly, and the code below is the real source
of truth for what the assistant sounds like.

The clip feeds LiveKit's BackgroundAudioPlayer:

  earcon.wav   the "I heard you" blip, fired the moment the turn enters its
               thinking state after the user stops talking — roughly a second
               ahead of first speech, so the silence never reads as a failure.

Usage: generate.py [output-dir [listening-output]]
       With no arguments, regenerates the voice earcon and Android chime.
"""

from __future__ import annotations

import math
import sys
import wave
from array import array
from pathlib import Path

SAMPLE_RATE = 48000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 16-bit PCM
FULL_SCALE = 32767

# --- earcon ----------------------------------------------------------------
# A rising two-tone chime (A5 then E6, a perfect fifth up). Rising and brief
# is the acknowledgment register — Siri's and Alexa's blips both rise; a
# falling or sustained tone reads as an error or an alarm.
EARCON_TONES = ((880.0, 0.0), (1320.0, 0.10))  # (frequency hz, onset seconds)
EARCON_TONE_S = 0.15  # each tone's length; total clip = 0.10 + 0.15 = 0.25s
EARCON_ATTACK_S = 0.006  # long enough that the onset is a swell, not a click
EARCON_DECAY_S = 0.045  # exponential decay constant, giving a struck-bell tail
# Clearly audible over room noise while staying well under speech, which the
# background player mixes at full level.
EARCON_PEAK_DBFS = -12.0

EARCON_NAME = "earcon.wav"

# --- phone listening chime --------------------------------------------------
# Three notes below the delegation earcon's A5, with a clear upward contour.
# This is the phone's listening cue, not a change to the delegation earcon.
LISTENING_TONES = ((440.0, 0.0), (554.37, 0.10), (659.25, 0.20))
LISTENING_TONE_S = 0.13
LISTENING_NAME = "listening.wav"


def _amplitude(dbfs: float) -> float:
    """Linear amplitude (0..1) for a level in dBFS."""
    return 10.0 ** (dbfs / 20.0)


def _normalize(signal: list[float], peak_dbfs: float) -> list[float]:
    """Scale a signal so its loudest sample sits exactly at peak_dbfs."""
    peak = max(abs(value) for value in signal)
    scale = _amplitude(peak_dbfs) / peak
    return [value * scale for value in signal]


def _chime_envelope(index: int, length: int) -> float:
    """Soft attack into exponential decay, exactly 0 at both ends.

    Both ends matter: a hard onset or a truncated tail is a step against the
    silence around it, which is the click this envelope exists to prevent. The
    raised-cosine attack removes the first, and subtracting the decay curve's
    value at the final sample pins the tail to true zero rather than to the
    small-but-nonzero value an exponential would otherwise still hold.
    """
    tau = EARCON_DECAY_S * SAMPLE_RATE
    floor = math.exp(-(length - 1) / tau)
    value = (math.exp(-index / tau) - floor) / (1.0 - floor)
    attack = round(EARCON_ATTACK_S * SAMPLE_RATE)
    if index < attack:
        value *= 0.5 - 0.5 * math.cos(math.pi * index / attack)
    return value


def earcon_signal() -> list[float]:
    """The earcon as floats, peak-normalized to EARCON_PEAK_DBFS."""
    tone_len = round(EARCON_TONE_S * SAMPLE_RATE)
    total = max(round(onset * SAMPLE_RATE) for _, onset in EARCON_TONES) + tone_len
    signal = [0.0] * total
    for frequency, onset in EARCON_TONES:
        start = round(onset * SAMPLE_RATE)
        step = math.tau * frequency / SAMPLE_RATE
        for i in range(tone_len):
            signal[start + i] += math.sin(step * i) * _chime_envelope(i, tone_len)
    return _normalize(signal, EARCON_PEAK_DBFS)


def listening_signal() -> list[float]:
    """The phone's distinct, lower three-note listening rise."""
    tone_len = round(LISTENING_TONE_S * SAMPLE_RATE)
    total = max(round(onset * SAMPLE_RATE) for _, onset in LISTENING_TONES) + tone_len
    signal = [0.0] * total
    for frequency, onset in LISTENING_TONES:
        start = round(onset * SAMPLE_RATE)
        step = math.tau * frequency / SAMPLE_RATE
        for i in range(tone_len):
            signal[start + i] += math.sin(step * i) * _chime_envelope(i, tone_len)
    return _normalize(signal, EARCON_PEAK_DBFS)


def to_pcm16(signal: list[float]) -> array[int]:
    """Quantize floats in -1..1 to signed 16-bit samples."""
    return array("h", [round(value * FULL_SCALE) for value in signal])


def write_wav(path: Path, samples: array[int]) -> None:
    """Write mono 16-bit PCM at SAMPLE_RATE.

    array("h") is the right pairing for wave: both speak native byte order and
    the module handles the little-endian file format itself, so the bytes match
    on any host.
    """
    with wave.open(str(path), "wb") as out:
        out.setnchannels(CHANNELS)
        out.setsampwidth(SAMPLE_WIDTH)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(samples.tobytes())


def generate_listening(path: Path) -> None:
    """Write the phone listening chime to its Android raw-resource path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, to_pcm16(listening_signal()))


def generate(out_dir: Path) -> None:
    """Write the earcon into out_dir."""
    write_wav(out_dir / EARCON_NAME, to_pcm16(earcon_signal()))


def main(argv: list[str]) -> None:
    out_dir = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent
    generate(out_dir)
    if len(argv) > 2:
        listening_path = Path(argv[2])
    elif len(argv) == 1:
        listening_path = (
            Path(__file__).resolve().parents[2]
            / "android/app/src/main/res/raw"
            / LISTENING_NAME
        )
    else:
        listening_path = None
    if listening_path is not None:
        generate_listening(listening_path)


if __name__ == "__main__":
    main(sys.argv)
