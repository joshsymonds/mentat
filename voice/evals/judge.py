"""Semantic reply judging through TypeSafe System One."""

from __future__ import annotations

import json
import os
from http.client import HTTPException
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, runtime_checkable


_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
_MODEL = "jev-1.13.0"
_MAX_ATTEMPTS = 3
_REQUEST_TIMEOUT = 10.0


class JudgeUnavailable(RuntimeError):
    """Raised when a judge result cannot be obtained safely."""


@dataclass(frozen=True)
class Verdict:
    """A question's semantic probability and its thresholded yes/no value."""

    question: str
    probability: float

    @property
    def verdict(self) -> bool:
        return self.probability >= 0.5


@runtime_checkable
class Judge(Protocol):
    """Evaluate yes/no questions against one assistant reply."""

    def evaluate(
        self,
        reply: str,
        questions: Mapping[str, str],
        *,
        context: str = "",
    ) -> dict[str, Verdict]: ...


class JevJudge:
    """TypeSafe Jev judge with a per-instance verdict cache."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout: float = _REQUEST_TIMEOUT,
        max_attempts: int = _MAX_ATTEMPTS,
    ) -> None:
        self._api_key = api_key if api_key is not None else _load_api_key()
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._cache: dict[tuple[str, str, str], Verdict] = {}
        self._cache_lock = threading.Lock()

    def evaluate(
        self,
        reply: str,
        questions: Mapping[str, str],
        *,
        context: str = "",
    ) -> dict[str, Verdict]:
        pending: dict[str, str] = {}
        result: dict[str, Verdict] = {}
        for question_id, question in questions.items():
            cache_key = (question, context, reply)
            with self._cache_lock:
                cached = self._cache.get(cache_key)
            if cached is None:
                pending[question_id] = question
            else:
                result[question_id] = cached

        if pending:
            with ThreadPoolExecutor(max_workers=len(pending)) as pool:
                futures = {
                    question_id: pool.submit(self._evaluate_one, question_id, reply, question, context)
                    for question_id, question in pending.items()
                }
                for question_id, future in futures.items():
                    verdict = future.result()
                    result[question_id] = verdict
                    cache_key = (pending[question_id], context, reply)
                    with self._cache_lock:
                        self._cache[cache_key] = verdict

        return {question_id: result[question_id] for question_id in questions}

    def _evaluate_one(self, question_id: str, reply: str, question: str, context: str) -> Verdict:
        if not self._api_key:
            raise JudgeUnavailable("judge credential is not configured")

        payload = {
            "state": context + "\n\n" + reply,
            "model": _MODEL,
            "questions": {question_id: {"type": "noul", "instructions": question}},
        }
        request = urllib.request.Request(
            _ENDPOINT,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        response_body = self._request(request)
        try:
            response = json.loads(response_body)
            answer = response["answers"][question_id]
            if answer["type"] != "noul":
                raise ValueError("unexpected answer type")
            probability = answer["noul"]
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise ValueError("invalid probability type")
            if not 0.0 <= probability <= 1.0:
                raise ValueError("probability outside range")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            raise JudgeUnavailable("judge returned a malformed response") from None
        return Verdict(question, float(probability))

    def _request(self, request: urllib.request.Request) -> bytes:
        for attempt in range(self._max_attempts):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as error:
                error.close()
                if error.code != 429 and not 500 <= error.code <= 599:
                    raise JudgeUnavailable("judge rejected the request") from None
                if attempt + 1 == self._max_attempts:
                    raise JudgeUnavailable("judge request failed after retries") from None
            except (TimeoutError, OSError, urllib.error.URLError, HTTPException):
                if attempt + 1 == self._max_attempts:
                    raise JudgeUnavailable("judge transport failed after retries") from None
            time.sleep(0.1 * (2 ** attempt))
        raise JudgeUnavailable("judge request failed")


class ScriptedJudge:
    """Deterministic judge for offline scripted evaluation."""

    def __init__(self, outcomes: Mapping[str, float | JudgeUnavailable]) -> None:
        self._outcomes = dict(outcomes)

    def evaluate(
        self,
        reply: str,
        questions: Mapping[str, str],
        *,
        context: str = "",
    ) -> dict[str, Verdict]:
        del reply, context
        result: dict[str, Verdict] = {}
        for question_id, question in questions.items():
            if question_id not in self._outcomes:
                raise JudgeUnavailable("scripted judge has no outcome for a question")
            outcome = self._outcomes[question_id]
            if isinstance(outcome, JudgeUnavailable):
                raise JudgeUnavailable(str(outcome)) from None
            if isinstance(outcome, bool) or not isinstance(outcome, (int, float)) or not 0 <= outcome <= 1:
                raise ValueError("scripted probability must be between 0 and 1")
            result[question_id] = Verdict(question, float(outcome))
        return result


def _load_api_key() -> str | None:
    """Read the key from the environment first, then the repository .env.local."""
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key

    root = Path(__file__).resolve().parents[2]
    env_file = root / ".env.local"
    try:
        lines = env_file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, separator, value = line.partition("=")
        if separator and name.strip() == "TYPESAFE_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            return value or None
    return None
