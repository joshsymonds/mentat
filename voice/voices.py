"""Resolve non-English voices from the user's ElevenLabs library."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

API_BASE = "https://api.elevenlabs.io/v1"
VOICE_STATE_ENV = "MENTAT_VOICE_STATE_DIR"
VOICE_STATE_FILE = "voices.json"
VOICE_MODEL = "eleven_v4_turbo"
HTTP_TIMEOUT_S = 10

Fetch = Callable[..., Any]


class VoiceResolver:
    """Pick, add, and persist the newest eligible library voice per language.

    ``state_dir`` contains ``voices.json``, a language-code-to-voice-ID map.
    When omitted, use ``MENTAT_VOICE_STATE_DIR``, then ``HOME`` (the voice
    service's state directory), then ``/var/lib/mentat-voice``. ``fetch`` is
    called as ``fetch(request, timeout=10)`` and must return a context manager
    whose response provides ``read()`` bytes, matching ``urllib.request.urlopen``.
    """

    def __init__(
        self,
        api_key: str | None,
        state_dir: str | Path | None = None,
        *,
        fetch: Fetch = urlopen,
    ) -> None:
        self._api_key = api_key
        self._state_dir = Path(
            state_dir
            if state_dir is not None
            else os.environ.get(VOICE_STATE_ENV, os.environ.get("HOME", "/var/lib/mentat-voice"))
        )
        self._fetch = fetch

    def resolve(self, language: str, default_voice: str) -> str:
        """Return a saved voice or add and save the newest verified library voice."""
        try:
            saved = self._read_saved()
        except (OSError, UnicodeError, ValueError) as error:
            logger.warning("voice lookup for %r failed reading saved voices: %s", language, error)
            return default_voice
        if language in saved:
            return saved[language]
        if not self._api_key:
            logger.warning("voice lookup for %r failed: ElevenLabs API key is unavailable", language)
            return default_voice
        try:
            voice = self._find_voice(language)
            if voice is None:
                raise LookupError("no verified conversational v4 library voice")
            added_voice_id = self._add_voice(voice, language)
            saved[language] = added_voice_id
            self._write_saved(saved)
            return added_voice_id
        except (OSError, UnicodeError, ValueError, KeyError, TypeError, LookupError) as error:
            logger.warning("voice lookup for %r failed: %s", language, error)
            return default_voice

    def _read_saved(self) -> dict[str, str]:
        path = self._state_dir / VOICE_STATE_FILE
        try:
            contents = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        raw = json.loads(contents)
        if not isinstance(raw, dict) or not all(
            isinstance(language, str)
            and language
            and isinstance(voice_id, str)
            and voice_id
            for language, voice_id in raw.items()
        ):
            raise ValueError("saved voices must map language codes to voice IDs")
        return raw

    def _find_voice(self, language: str) -> dict[str, str] | None:
        query = urlencode({"language": language, "use_cases": "conversational"})
        payload = self._request_json("GET", f"{API_BASE}/shared-voices?{query}")
        voices = payload.get("voices")
        if not isinstance(voices, list):
            raise ValueError("shared voices response has no voices list")
        for candidate in voices:
            if not isinstance(candidate, Mapping):
                continue
            voice_id = candidate.get("voice_id")
            owner_id = candidate.get("public_owner_id")
            verified = candidate.get("verified_languages")
            if not isinstance(voice_id, str) or not voice_id or not isinstance(owner_id, str) or not owner_id:
                continue
            if not isinstance(verified, list):
                continue
            if any(
                isinstance(item, Mapping)
                and item.get("language") == language
                and item.get("model_id") == VOICE_MODEL
                for item in verified
            ):
                return {"voice_id": voice_id, "public_owner_id": owner_id}
        return None

    def _add_voice(self, voice: Mapping[str, str], language: str) -> str:
        owner_id = quote(voice["public_owner_id"], safe="")
        voice_id = quote(voice["voice_id"], safe="")
        url = f"{API_BASE}/voices/add/{owner_id}/{voice_id}"
        payload = self._request_json("POST", url, body={"new_name": f"Mentat {language}"})
        added_voice_id = payload.get("voice_id")
        if not isinstance(added_voice_id, str) or not added_voice_id:
            raise ValueError("add voice response has no voice_id")
        return added_voice_id

    def _request_json(self, method: str, url: str, *, body: Any = None) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"xi-api-key": self._api_key or ""}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        with self._fetch(request, timeout=HTTP_TIMEOUT_S) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, dict):
            raise ValueError("ElevenLabs response must be a JSON object")
        return payload

    def _write_saved(self, saved: Mapping[str, str]) -> None:
        self._state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        path = self._state_dir / VOICE_STATE_FILE
        content = json.dumps(saved, ensure_ascii=False, indent=2) + "\n"
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._state_dir,
                prefix=f".{VOICE_STATE_FILE}.",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                os.chmod(temporary_path, 0o600)
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
