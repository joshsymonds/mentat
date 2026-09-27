"""Offline checks for the opt-in voice evaluation recipe."""

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class EvalRecipeTests(unittest.TestCase):
    def dry_run(self, *arguments):
        return subprocess.run(
            ["just", "--dry-run", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    @staticmethod
    def output(result):
        return result.stdout + result.stderr

    def test_default_recipe_is_opt_in_and_never_runs_voice_eval(self):
        result = self.dry_run()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("voice.evals.runner eval", self.output(result))

    def test_eval_recipe_defaults_to_ten_and_accepts_run_count(self):
        default = self.dry_run("eval-voice")
        configured = self.dry_run("eval-voice", "2")

        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(configured.returncode, 0, configured.stderr)
        self.assertIn(
            "python3 -m voice.evals.runner eval --live --runs '10'",
            self.output(default),
        )
        self.assertIn(
            "python3 -m voice.evals.runner eval --live --runs '2'",
            self.output(configured),
        )

    def test_readme_documents_safety_scenarios_and_strict_report(self):
        readme = (ROOT / "voice" / "README.md").read_text()

        for required in (
            "just eval-voice",
            "just eval-voice 2",
            "python3 -m voice.evals.runner eval --list",
            "fake phone",
            "normal, error, and signal exits",
            "remote service access and API usage costs",
            "timer-300-seconds",
            "equivalent-alarm",
            "place-search-navigation",
            "sms-say-back-yes",
            "sms-correction-new-yes",
            "alice-keck-context-chain",
            "JSON report",
            "exits nonzero",
        ):
            with self.subTest(required=required):
                self.assertIn(required, readme)


if __name__ == "__main__":
    unittest.main()
