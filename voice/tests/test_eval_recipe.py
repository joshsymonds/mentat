"""Offline checks for the opt-in voice evaluation recipe."""

import contextlib
import io
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from voice.evals import runner


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

    def test_eval_recipe_forwards_optional_concurrency_without_changing_old_arguments(self):
        default = self.dry_run("eval-voice")
        configured = self.dry_run("eval-voice", "2")
        capped = self.dry_run("eval-voice", "2", "claude-opus-5-5", "4")
        quoted = self.dry_run(
            "eval-voice", "2", "chatgpt/sol-fast", "$(touch /tmp/not-run)"
        )

        for result in (default, configured, capped, quoted):
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "python3 -m voice.evals.runner eval --live --runs '10'",
            self.output(default),
        )
        self.assertIn(
            "python3 -m voice.evals.runner eval --live --runs '2'",
            self.output(configured),
        )
        self.assertNotIn("--concurrency", self.output(default))
        self.assertNotIn("--concurrency", self.output(configured))
        self.assertIn(
            "python3 -m voice.evals.runner eval --live --runs '2' --concurrency '4'",
            self.output(capped),
        )
        self.assertIn("'$(touch /tmp/not-run)'", self.output(quoted))
        self.assertIn("--concurrency", self.output(quoted))

    def test_eval_recipe_exports_supported_model_for_each_command_arm(self):
        default = self.dry_run("eval-voice")
        sol = self.dry_run("eval-voice", "2", "chatgpt/sol-fast")
        opus = self.dry_run("eval-voice", "10", "claude-opus-5-5")

        for result in (default, sol, opus):
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(
            "MENTAT_VOICE_MODEL='chatgpt/sol-fast' python3 -m voice.evals.runner eval --live --runs '10'",
            self.output(default),
        )
        self.assertIn(
            "MENTAT_VOICE_MODEL='chatgpt/sol-fast' python3 -m voice.evals.runner eval --live --runs '2'",
            self.output(sol),
        )
        self.assertIn(
            "MENTAT_VOICE_MODEL='claude-opus-5-5' python3 -m voice.evals.runner eval --live --runs '10'",
            self.output(opus),
        )
        self.assertNotIn("fast-mode", self.output(opus))
        self.assertNotIn("--speed", self.output(opus))

    def test_runner_rejects_unknown_model_before_constructing_dev_stack(self):
        stderr = io.StringIO()
        with (
            patch.dict(os.environ, {"MENTAT_VOICE_MODEL": "unapproved-model"}),
            patch.object(runner, "DevStack") as dev_stack,
            contextlib.redirect_stderr(stderr),
        ):
            result = runner._run_local_eval(["--live", "--runs", "1"])

        self.assertEqual(result, 2)
        self.assertIn("unsupported requested voice model", stderr.getvalue())
        dev_stack.assert_not_called()

    def test_readme_documents_safety_scenarios_and_strict_report(self):
        readme = (ROOT / "voice" / "README.md").read_text()

        for required in (
            "just eval-voice",
            "just eval-voice 2",
            "just eval-voice 2 chatgpt/sol-fast 4",
            "third positional argument",
            "runner's named default cap",
            "One candidate build and staging",
            "Each run gets its own",
            "candidate daemon, fake phone, worker, ports, state, and logs",
            "peak concurrent-run count",
            "total batch wall",
            "SIGINT/SIGTERM",
            "recorded batch",
            "positively dead",
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
