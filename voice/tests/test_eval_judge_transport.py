"""Offline transport regression tests for truncated Jev responses."""

import io
import sys
import unittest
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.judge import JudgeUnavailable, JevJudge, Verdict
from evals.judge_qualify import qualify_fixtures
from evals.scenarios import SCENARIOS, evaluate_scenario_failures


_BODY = b'{"answers":{"q":{"type":"noul","noul":0.9}}}'


class _Socket:
    def makefile(self, *_args, **_kwargs):
        return io.BytesIO(
            b"HTTP/1.1 200 OK\r\nContent-Length: "
            + str(len(_BODY) + 40).encode("ascii")
            + b"\r\n\r\n"
            + _BODY
        )


def _truncated_response():
    response = HTTPResponse(_Socket())
    response.begin()
    return response


class JudgeTransportTests(unittest.TestCase):
    def test_truncated_http_response_retries_then_raises_typed_unavailability(self):
        calls = []
        sleeps = []

        def open_request(*_args, **_kwargs):
            calls.append(None)
            return _truncated_response()

        with patch("evals.judge.urllib.request.urlopen", open_request), \
                patch("evals.judge.time.sleep", sleeps.append):
            with self.assertRaisesRegex(JudgeUnavailable, "judge transport failed after retries") as raised:
                JevJudge(api_key="offline-test").evaluate("reply", {"q": "question"})

        self.assertEqual(type(raised.exception).__name__, "JudgeUnavailable")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [0.1, 0.2])
        self.assertNotIn("offline-test", str(raised.exception))

    def test_http_exception_disconnect_retries_offline(self):
        calls = []
        sleeps = []

        def open_request(*_args, **_kwargs):
            calls.append(None)
            raise HTTPException("connection closed during response")

        with patch("evals.judge.urllib.request.urlopen", open_request), \
                patch("evals.judge.time.sleep", sleeps.append):
            with self.assertRaisesRegex(JudgeUnavailable, "judge transport failed after retries") as raised:
                JevJudge(api_key="offline-test").evaluate("reply", {"q": "question"})

        self.assertEqual(type(raised.exception).__name__, "JudgeUnavailable")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [0.1, 0.2])

    def test_truncated_later_scenario_turn_preserves_earlier_and_later_evidence(self):
        scenario = next(item for item in SCENARIOS if item.name == "spanish-language-switch")

        class LaterTruncatedJudge:
            def __init__(self):
                self.calls = 0

            def evaluate(self, reply, questions, *, context=""):
                del reply, context
                self.calls += 1
                if self.calls == 2:
                    with patch("evals.judge.urllib.request.urlopen", return_value=_truncated_response()), \
                            patch("evals.judge.time.sleep"):
                        return JevJudge(api_key="offline-test", max_attempts=1).evaluate(
                            "reply", {"q": "question"}
                        )
                return {
                    key: Verdict(question, 0.9)
                    for key, question in questions.items()
                }

        evidence = []
        failures = evaluate_scenario_failures(
            scenario,
            [f"answer {turn}" for turn in range(1, 6)],
            [],
            5,
            judge=LaterTruncatedJudge(),
            judge_evidence=evidence,
        )

        self.assertEqual([failure.turn for failure in failures], [2])
        self.assertIn("judge unavailable", failures[0].message)
        self.assertEqual(len(evidence), 5)
        self.assertTrue(evidence[0]["questions"][0]["verdict"])
        self.assertIsNotNone(evidence[1]["unavailable"])
        self.assertIsNone(evidence[1]["questions"][0]["verdict"])
        self.assertTrue(evidence[2]["questions"][0]["verdict"])
        self.assertTrue(evidence[4]["questions"][0]["verdict"])

    def test_truncated_qualification_response_is_reported_unavailable(self):
        fixture = {
            "id": "offline-truncated",
            "family": "test-family",
            "reply": "reply",
            "expected": True,
            "questions": {"q": "question"},
        }

        def judge_factory(_fixture, _run_number):
            return JevJudge(api_key="offline-test", max_attempts=1)

        with patch("evals.judge.urllib.request.urlopen", return_value=_truncated_response()):
            result = qualify_fixtures([fixture], judge_factory=judge_factory, runs=1)

        self.assertEqual(result["status"], "fail")
        run = result["runs"][0]
        self.assertEqual(run["status"], "fail")
        self.assertEqual(run["counts"]["unavailable"], 1)
        self.assertEqual(run["fixtures"][0]["status"], "unavailable")
        self.assertEqual(run["fixtures"][0]["observation"], "unavailable")


if __name__ == "__main__":
    unittest.main()
