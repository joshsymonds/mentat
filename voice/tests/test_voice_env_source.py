"""Source contract for the packaged LiveKit voice environment."""

import re
import unittest
from pathlib import Path


class VoiceEnvironmentSourceContractTest(unittest.TestCase):
    def test_elevenlabs_plugin_is_pinned_in_the_production_interpreter(self):
        source = (Path(__file__).resolve().parents[2] / "nix" / "voice-env.nix").read_text()

        package = re.search(
            r"livekit-plugins-elevenlabs\s*=\s*wheelPackage\s*\{(?P<body>.*?)\n  \};",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(package, "ElevenLabs plugin must be packaged as an upstream wheel")
        body = package.group("body")
        self.assertRegex(body, r'pname\s*=\s*"livekit-plugins-elevenlabs";')
        self.assertRegex(body, r'wheelName\s*=\s*"livekit_plugins_elevenlabs";')
        self.assertRegex(body, r'version\s*=\s*"1\.8\.1";')
        self.assertRegex(
            body,
            r'hash\s*=\s*"sha256-2HW1pViPU3ZZcgeDJOEtxsS49zHunhjF/PeTBV86w/o=";',
        )
        self.assertRegex(body, r"dependencies\s*=\s*\[\s*livekit-agents\s*\];")
        self.assertIn("pythonImportsCheck = [ \"livekit.plugins.elevenlabs\" ];", body)

        runtime = source.split("(pkgs.python3.withPackages (_: [", 1)[1].split(
            "])).overrideAttrs", 1
        )[0]
        self.assertIn("livekit-plugins-elevenlabs", runtime)
        self.assertIn("from livekit.plugins import dtln, elevenlabs, silero", source)
        self.assertIn('elevenlabs.TTS(api_key="synthetic-test-key")', source)


if __name__ == "__main__":
    unittest.main()
