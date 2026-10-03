"""Offline checks for the phrase-garble STT evaluation."""

import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.stt_phrases import PHRASES, Phrase, keyterm_phrases, normalize, verdict


ROOT = Path(__file__).resolve().parents[2]


class SttPhrasesTests(unittest.TestCase):
    def test_normalize_ignores_case_punctuation_accents_and_digit_grouping(self):
        self.assertEqual(normalize("What's the weather, Josh?"), "whats the weather josh")
        self.assertEqual(normalize("¿Qué tiempo hace?"), "que tiempo hace")
        self.assertEqual(normalize("Call 555-123-4567."), "call 5551234567")
        self.assertEqual(normalize("Wake me at 7:30."), "wake me at 730")
        self.assertEqual(normalize("  Stop   the timer "), "stop the timer")

    def test_every_accepted_spelling_is_already_normalized(self):
        for phrase in PHRASES:
            with self.subTest(phrase=phrase.text):
                self.assertTrue(phrase.accepted)
                self.assertIn(phrase.language, {"en", "es"})
                for accepted in phrase.accepted:
                    self.assertEqual(normalize(accepted), accepted)

    def test_verdict_requires_an_exact_accepted_transcript(self):
        phrase = Phrase("en", "Stop the timer.", ("stop the timer",))
        self.assertTrue(verdict(phrase, "Stop the timer."))
        self.assertFalse(verdict(phrase, "Stop the time."))
        self.assertFalse(verdict(phrase, "Stop the timer now."))
        self.assertFalse(verdict(phrase, ""))

    def test_keyterm_phrases_require_the_exact_term(self):
        (phrase,) = keyterm_phrases(("Symonds",))
        self.assertEqual(phrase.text, "I was talking about Symonds earlier.")
        self.assertTrue(verdict(phrase, "I was talking about Symonds earlier."))
        self.assertFalse(verdict(phrase, "I was talking about Simmons earlier."))
        self.assertEqual(keyterm_phrases(()), ())

    def test_recipe_is_opt_in_and_runs_the_voice_environment(self):
        default = subprocess.run(
            ["just", "--dry-run"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertNotIn("stt_phrases", default.stdout + default.stderr)
        recipe = subprocess.run(
            ["just", "--dry-run", "eval-stt", "3"],
            cwd=ROOT, capture_output=True, text=True, check=False,
        )
        self.assertEqual(recipe.returncode, 0, recipe.stderr)
        output = recipe.stdout + recipe.stderr
        self.assertIn("nix build .#voice-env", output)
        self.assertIn("-m voice.evals.stt_phrases --runs '3'", output)


if __name__ == "__main__":
    unittest.main()
