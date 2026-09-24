import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import unittest


README = Path(__file__).resolve().parents[1] / "README.md"


class CallerDocumentationTests(unittest.TestCase):
    def test_ssh_steps_reach_remote_caller_as_four_exact_arguments(self):
        readme = README.read_text()
        section = readme.split("### 4. Run calls A-D", 1)[1]
        command = re.search(r"```sh\n(.*?)\n```", section, re.DOTALL)
        self.assertIsNotNone(command, "SSH launch command is missing")

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bin_dir = root / "bin"
            bin_dir.mkdir()
            captured = root / "arguments.json"

            real_bash = shutil.which("bash")
            self.assertIsNotNone(real_bash, "bash is required by the documented command")
            scripts = {
                "ssh": "#!/bin/sh\nshift\nexec /bin/sh -c \"$*\"\n",
                "sudo": "#!/bin/sh\nexec \"$@\"\n",
                "bash": textwrap.dedent("""\
                    #!/usr/bin/env python3
                    import os
                    import subprocess
                    import sys
                    script = sys.stdin.read().replace(
                        '. /run/agenix/mentat-voice-env', ':'
                    )
                    raise SystemExit(subprocess.run(
                        [REAL_BASH, *sys.argv[1:]], input=script, text=True,
                        env=os.environ,
                    ).returncode)
                    """),
                "timeout": "#!/bin/sh\n[ \"$1\" = 240 ] && shift\nexec \"$@\"\n",
            }
            scripts["bash"] = scripts["bash"].replace("REAL_BASH", repr(real_bash))
            for name, content in scripts.items():
                path = bin_dir / name
                path.write_text(content)
                path.chmod(0o755)

            fake_python = root / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                f"with open({str(captured)!r}, 'w') as output:\n"
                "    json.dump(sys.argv[2:], output)\n"
            )
            fake_python.chmod(0o755)

            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{bin_dir}:{environment['PATH']}",
                    "DEV_DIR": str(root),
                    "DEV_ROOM": "test-room",
                    "DEV_PY": str(fake_python),
                }
            )
            completed = subprocess.run(
                ["sh", "-c", command.group(1)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                f"documented command failed: {completed.stderr}",
            )
            self.assertTrue(captured.exists(), "remote caller was not invoked")
            self.assertEqual(
                json.loads(captured.read_text()),
                [
                    "test-room",
                    "What is the latest released version of livekit-agents on PyPI?@1::[0-9]+\\.[0-9]+\\.[0-9]+",
                    "What is the current Bitcoin price in USD according to CoinGecko?@1::(?i)(?:\\$|USD\\s*)[0-9,]+(?:\\.[0-9]+)?|[0-9,]+(?:\\.[0-9]+)?\\s*USD",
                    "Who wrote Pride and Prejudice?@1::(?i)Jane\\s+Austen",
                    "What was the latest Formula 1 Grand Prix, and who won it?@1::(?i)\\b(?:won|winner|unsure|uncertain)\\b",
                ],
            )


if __name__ == "__main__":
    unittest.main()
