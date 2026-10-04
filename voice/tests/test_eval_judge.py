"""Offline tests for the semantic reply judge."""

import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import judge
from evals.judge import JudgeUnavailable, JevJudge, ScriptedJudge, Verdict


class FakeResponse:
    def __init__(self, body):
        self.body = json.dumps(body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


class EvalJudgeTests(unittest.TestCase):
    def test_flat_system_one_request_and_probability_verdict(self):
        requests = []

        def open_request(request, timeout):
            requests.append((request, timeout))
            return FakeResponse({"answers": {"q1": {"type": "noul", "noul": 0.73}}})

        with patch.object(judge.urllib.request, "urlopen", open_request):
            result = JevJudge(api_key="test-secret").evaluate(
                "The timer is set for 8:30.", {"q1": "Did it confirm 8:30?"},
                context="The caller requested a timer for 8:30.",
            )

        self.assertEqual(len(requests), 1)
        request, timeout = requests[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        payload = json.loads(request.data)
        self.assertEqual(payload["model"], "jev-1.13.0")
        self.assertEqual(payload["state"],
            "The caller requested a timer for 8:30.\n\nThe timer is set for 8:30.")
        self.assertEqual(payload["questions"], {
            "q1": {"type": "noul", "instructions": "Did it confirm 8:30?"},
        })
        self.assertGreater(timeout, 0)
        self.assertEqual(result["q1"], Verdict("Did it confirm 8:30?", 0.73))
        self.assertTrue(result["q1"].verdict)

    def test_yes_threshold_and_probability_preservation(self):
        response_bodies = iter([
            {"answers": {"q": {"type": "noul", "noul": 0.5}}},
            {"answers": {"q": {"type": "noul", "noul": 0.499}}},
        ])
        with patch.object(judge.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse(next(response_bodies))):
            at_threshold = JevJudge(api_key="x").evaluate("reply", {"q": "question"})["q"]
            below_threshold = JevJudge(api_key="x").evaluate("reply", {"q": "question"})["q"]
        self.assertTrue(at_threshold.verdict)
        self.assertFalse(below_threshold.verdict)
        self.assertEqual(at_threshold.probability, 0.5)
        self.assertEqual(below_threshold.probability, 0.499)

    def test_cache_is_per_judge_and_includes_verified_context(self):
        calls = []

        def open_request(request, timeout):
            calls.append(request)
            return FakeResponse({"answers": {"q": {"type": "noul", "noul": 0.8}}})

        with patch.object(judge.urllib.request, "urlopen", open_request):
            judge_run = JevJudge(api_key="x")
            first = judge_run.evaluate("yes", {"q": "Was this correct?"}, context="asked yes")
            cached = judge_run.evaluate("yes", {"renamed": "Was this correct?"}, context="asked yes")
            other_context = judge_run.evaluate("yes", {"q": "Was this correct?"}, context="asked no")
            other_run = JevJudge(api_key="x").evaluate("yes", {"q": "Was this correct?"}, context="asked yes")

        self.assertEqual(len(calls), 3)
        self.assertEqual(first["q"], cached["renamed"])
        self.assertEqual(other_context["q"].probability, 0.8)
        self.assertEqual(other_run["q"].probability, 0.8)

    def test_questions_are_requested_concurrently(self):
        barrier = threading.Barrier(2, timeout=2)
        completed = []

        def open_request(request, timeout):
            question_id = next(iter(json.loads(request.data)["questions"]))
            barrier.wait()
            completed.append(question_id)
            return FakeResponse({"answers": {question_id: {"type": "noul", "noul": 0.6}}})

        with patch.object(judge.urllib.request, "urlopen", open_request):
            answers = JevJudge(api_key="x").evaluate(
                "reply", {"q1": "First?", "q2": "Second?"},
            )
        self.assertEqual(set(answers), {"q1", "q2"})
        self.assertCountEqual(completed, ["q1", "q2"])

    def test_environment_key_precedes_repo_dotenv_and_secret_is_not_disclosed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_path = root / "voice" / "evals" / "judge.py"
            module_path.parent.mkdir(parents=True)
            module_path.touch()
            (root / ".env.local").write_text("TYPESAFE_API_KEY=file-secret\n")
            seen = []

            def open_request(request, timeout):
                seen.append(request.get_header("Authorization"))
                return FakeResponse({"answers": {"q": {"type": "noul", "noul": 0.7}}})

            with patch.object(judge, "__file__", str(module_path)), \
                    patch.dict(os.environ, {"TYPESAFE_API_KEY": "environment-secret"}), \
                    patch.object(judge.urllib.request, "urlopen", open_request):
                JevJudge().evaluate("reply", {"q": "question"})

        self.assertEqual(seen, ["Bearer environment-secret"])

    def test_dotenv_key_is_used_when_environment_is_unset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module_path = root / "voice" / "evals" / "judge.py"
            module_path.parent.mkdir(parents=True)
            module_path.touch()
            (root / ".env.local").write_text("OTHER=value\nTYPESAFE_API_KEY='file-secret'\n")
            seen = []

            def open_request(request, timeout):
                seen.append(request.get_header("Authorization"))
                return FakeResponse({"answers": {"q": {"type": "noul", "noul": 0.7}}})

            with patch.object(judge, "__file__", str(module_path)), \
                    patch.dict(os.environ, {}, clear=True), \
                    patch.object(judge.urllib.request, "urlopen", open_request):
                JevJudge().evaluate("reply", {"q": "question"})

        self.assertEqual(seen, ["Bearer file-secret"])

    def test_retryable_responses_back_off_then_succeed(self):
        calls = []
        sleeps = []

        def open_request(*_args, **_kwargs):
            calls.append(None)
            if len(calls) < 3:
                raise HTTPError("https://api.typesafe.ai/v1/systemone", 503, "busy", {}, None)
            return FakeResponse({"answers": {"q": {"type": "noul", "noul": 0.9}}})

        with patch.object(judge.urllib.request, "urlopen", open_request), \
                patch.object(judge.time, "sleep", sleeps.append):
            result = JevJudge(api_key="x").evaluate("reply", {"q": "question"})

        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [0.1, 0.2])
        self.assertEqual(result["q"].probability, 0.9)

    def test_retry_exhaustion_returns_typed_unavailability_without_key(self):
        calls = []

        def open_request(*_args, **_kwargs):
            calls.append(None)
            raise HTTPError("https://api.typesafe.ai/v1/systemone", 429, "limited", {}, None)

        with patch.object(judge.urllib.request, "urlopen", open_request), \
                patch.object(judge.time, "sleep", lambda _seconds: None):
            with self.assertRaises(JudgeUnavailable) as raised:
                JevJudge(api_key="secret-value").evaluate("reply", {"q": "question"})
        self.assertGreater(len(calls), 1)
        self.assertNotIn("secret-value", str(raised.exception))

    def test_timeout_and_transport_failures_retry_then_raise_unavailable(self):
        for failure in (TimeoutError("slow"), URLError("offline")):
            calls = []
            sleeps = []

            def open_request(*_args, **_kwargs):
                calls.append(None)
                raise failure

            with self.subTest(failure=type(failure).__name__), \
                    patch.object(judge.urllib.request, "urlopen", open_request), \
                    patch.object(judge.time, "sleep", sleeps.append):
                with self.assertRaises(JudgeUnavailable):
                    JevJudge(api_key="x").evaluate("reply", {"q": "question"})
            self.assertEqual(len(calls), 3)
            self.assertEqual(sleeps, [0.1, 0.2])

    def test_malformed_response_is_typed_unavailability(self):
        malformed = [
            b"not json",
            json.dumps({"answers": {}}).encode(),
            json.dumps({"answers": {"q": {"type": "noul", "noul": "0.8"}}}).encode(),
            json.dumps({"answers": {"q": {"type": "noul", "noul": 1.2}}}).encode(),
        ]
        for body in malformed:
            with self.subTest(body=body), \
                    patch.object(judge.urllib.request, "urlopen", side_effect=lambda *_a, **_k: FakeResponseBytes(body)):
                with self.assertRaises(JudgeUnavailable):
                    JevJudge(api_key="x").evaluate("reply", {"q": "question"})

    def test_scripted_judge_implements_protocol_and_unavailable_outcome(self):
        scripted = ScriptedJudge({"q1": 0.5, "q2": JudgeUnavailable("scripted outage")})
        self.assertIsInstance(scripted, judge.Judge)
        answers = scripted.evaluate("reply", {"q1": "Pass?"})
        self.assertTrue(answers["q1"].verdict)
        self.assertEqual(answers["q1"].probability, 0.5)
        with self.assertRaises(JudgeUnavailable):
            scripted.evaluate("reply", {"q2": "Unavailable?"})


class FakeResponseBytes(FakeResponse):
    def __init__(self, body):
        self.body = body


if __name__ == "__main__":
    unittest.main()
