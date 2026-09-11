"""Tests for the generated voice sound asset.

The asset is checked in as a .wav file but authored as code, so these tests
guard both halves: the committed file has the format the LiveKit
BackgroundAudioPlayer needs (48kHz mono 16-bit, right duration, right
loudness), and re-running the generator reproduces it byte for byte. The
second half is what makes the binary asset reviewable — a reviewer who cannot
diff a .wav can still read generate.py and confirm the bytes follow from it.
"""

import math
import subprocess
import sys
import tempfile
import unittest
import wave
from array import array
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "assets"))

import generate

ASSETS = Path(__file__).resolve().parent.parent / "assets"
EARCON = ASSETS / "earcon.wav"

FULL_SCALE = 32767


def read_wav(path):
    """(samples, framerate, channels, sampwidth) for a 16-bit PCM file.

    array("h") is the right pairing for wave: both sides speak native byte
    order and the module handles the little-endian file format itself.
    """
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        samples = array("h", w.readframes(params.nframes))
    return samples, params.framerate, params.nchannels, params.sampwidth


def peak_dbfs(samples):
    return 20.0 * math.log10(max(abs(s) for s in samples) / FULL_SCALE)


class FormatTest(unittest.TestCase):
    """The clip must be what the playback path expects: 48kHz mono 16-bit."""

    def test_earcon_exists(self):
        self.assertTrue(EARCON.is_file(), f"{EARCON} not generated")

    def test_earcon_format(self):
        _, framerate, channels, width = read_wav(EARCON)
        self.assertEqual((framerate, channels, width), (48000, 1, 2))


class EarconTest(unittest.TestCase):
    """The acknowledgment blip: short, polite, click-free."""

    def test_duration_is_a_blip(self):
        # It lands the instant the user stops talking, roughly a second ahead
        # of first speech; long enough to register, short enough not to delay.
        samples, framerate, _, _ = read_wav(EARCON)
        self.assertGreaterEqual(len(samples) / framerate, 0.1)
        self.assertLessEqual(len(samples) / framerate, 0.5)

    def test_peak_is_minus_12_dbfs(self):
        # Audible over room noise but well under speech, which the player
        # mixes at full level.
        samples, _, _, _ = read_wav(EARCON)
        self.assertAlmostEqual(peak_dbfs(samples), -12.0, delta=0.1)

    def test_starts_and_ends_in_silence(self):
        # A non-zero first or last sample is a step discontinuity against the
        # silence around it — the click the envelope exists to prevent.
        samples, _, _, _ = read_wav(EARCON)
        self.assertEqual(samples[0], 0)
        self.assertEqual(samples[-1], 0)

    def test_rises_in_pitch(self):
        # The two-tone rise is what makes it read as "listening" rather than
        # as an alarm: the second tone starts above the first.
        self.assertLess(generate.EARCON_TONES[0][0], generate.EARCON_TONES[1][0])
        self.assertLess(generate.EARCON_TONES[0][1], generate.EARCON_TONES[1][1])


class SoundWiringContractTest(unittest.TestCase):
    """Generation, deployment, and runtime wiring retain only the earcon."""

    def test_only_earcon_is_generated_deployed_and_wired(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate.generate(Path(tmp))
            self.assertEqual(
                {path.name for path in Path(tmp).iterdir()},
                {"earcon.wav"},
            )

        self.assertEqual(
            {path.name for path in ASSETS.glob("*.wav")},
            {"earcon.wav"},
        )

        repository = ASSETS.parent.parent
        agent_source = (repository / "voice" / "agent.py").read_text()
        module_source = (repository / "nix" / "module.nix").read_text()
        for stem in ("waiting", "ambient"):
            obsolete = f"{stem}.wav"
            self.assertNotIn(obsolete, agent_source)
            self.assertNotIn(obsolete, module_source)
        self.assertNotIn("ambient_sound=", agent_source)
        self.assertNotIn("background_audio", agent_source)
        self.assertNotIn("loop=True", agent_source)
        self.assertIn("background = BackgroundAudioPlayer()", agent_source)
        self.assertIn("earcon.wav", agent_source)
        self.assertIn("earcon.wav", module_source)


class DeterminismTest(unittest.TestCase):
    """Regeneration must be byte-identical or the asset stops being auditable."""

    def test_regeneration_reproduces_the_committed_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, str(ASSETS / "generate.py"), tmp],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            fresh = Path(tmp) / EARCON.name
            self.assertEqual(
                fresh.read_bytes(),
                EARCON.read_bytes(),
                f"{EARCON.name} does not match generate.py output",
            )


if __name__ == "__main__":
    unittest.main()
