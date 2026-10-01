"""Offline tests for selecting and persisting ElevenLabs library voices."""

import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from voice.voices import VoiceResolver


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class FakeHTTP:
    def __init__(self, voices):
        self.voices = voices
        self.requests = []
        self.fail = False
        self.added_voice_id = "newest-eligible"

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self.fail:
            raise OSError("synthetic provider failure")
        if request.method == "GET":
            return FakeResponse({"voices": self.voices})
        assert isinstance(json.loads(request.data).get("new_name"), str)
        assert json.loads(request.data)["new_name"]
        return FakeResponse({"voice_id": self.added_voice_id})


class VoiceResolverTest(unittest.TestCase):
    def voice(self, voice_id, verified_languages, owner="owner"):
        return {
            "voice_id": voice_id,
            "public_owner_id": owner,
            "verified_languages": verified_languages,
        }

    def verified(self, language, model_id="eleven_v4_turbo"):
        return {"language": language, "model_id": model_id}

    def test_picks_newest_eligible_voice_adds_and_persists(self):
        voices = [
            self.voice("newest-ineligible", [self.verified("es", "eleven_multilingual_v2")]),
            self.voice("newest-eligible", [self.verified("es")]),
            self.voice("older-eligible", [self.verified("es")]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            fetch = FakeHTTP(voices)
            fetch.added_voice_id = "added-library-voice"
            resolver = VoiceResolver("synthetic-key", Path(temporary), fetch=fetch)

            self.assertEqual(resolver.resolve("es", "mentat-default"), "added-library-voice")

            self.assertEqual(len(fetch.requests), 2)
            lookup, add = fetch.requests
            self.assertEqual(lookup.method, "GET")
            self.assertEqual(
                parse_qs(urlparse(lookup.full_url).query),
                {"language": ["es"], "use_cases": ["conversational"]},
            )
            self.assertEqual(add.method, "POST")
            self.assertTrue(add.full_url.endswith("/v1/voices/add/owner/newest-eligible"))
            self.assertEqual(
                json.loads((Path(temporary) / "voices.json").read_text()),
                {"es": "added-library-voice"},
            )

    def test_saved_voice_is_reused_by_a_new_resolver(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "voices.json"
            state.write_text(json.dumps({"es": "saved-voice"}))
            fetch = FakeHTTP([])

            actual = VoiceResolver("synthetic-key", Path(temporary), fetch=fetch).resolve(
                "es", "mentat-default"
            )

            self.assertEqual(actual, "saved-voice")
            self.assertEqual(fetch.requests, [])

    def test_hand_edited_saved_entry_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary) / "voices.json"
            state.write_text(json.dumps({"es": "hand-picked"}))
            fetch = FakeHTTP([self.voice("new-provider-choice", [self.verified("es")])])

            actual = VoiceResolver("synthetic-key", Path(temporary), fetch=fetch).resolve(
                "es", "mentat-default"
            )

            self.assertEqual(actual, "hand-picked")
            self.assertEqual(fetch.requests, [])

    def test_failed_lookup_logs_and_returns_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            fetch = FakeHTTP([])
            fetch.fail = True
            resolver = VoiceResolver("synthetic-key", Path(temporary), fetch=fetch)

            with self.assertLogs("voice.voices", level="WARNING") as logs:
                actual = resolver.resolve("es", "mentat-default")

            self.assertEqual(actual, "mentat-default")
            self.assertIn("voice lookup for 'es' failed", " ".join(logs.output))
            self.assertFalse((Path(temporary) / "voices.json").exists())


if __name__ == "__main__":
    unittest.main()
