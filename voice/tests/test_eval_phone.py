import asyncio
import http.server
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

from phone import run_fake_phone


class _PhoneServer(http.server.ThreadingHTTPServer):
    daemon_threads = True


class FakePhoneTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.commands = [
            {"id": "command-1", "kind": "sms", "to": "+12065550123", "body": "Hello"},
            {"id": "command-2", "kind": "location"},
        ]
        self.results = []
        commands = self.commands
        results = self.results
        command_delay = {"seconds": 0.0}
        self.command_delay = command_delay
        result_status = {"code": 204}
        self.result_status = result_status

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_GET(self):
                if self.path != "/v1/phone/commands" or self.headers.get("X-Mentat-Phone") != "fake-phone":
                    self.send_error(403)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for index, command in enumerate(commands):
                    if index == 1:
                        time.sleep(command_delay["seconds"])
                    payload = (json.dumps(command) + "\n").encode()
                    self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                self.close_connection = True

            def do_POST(self):
                self.assert_result_path()
                body = self.rfile.read(int(self.headers["Content-Length"]))
                results.append(json.loads(body))
                self.send_response(result_status["code"])
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True

            def assert_result_path(self):
                if self.path != "/v1/phone/results":
                    self.send_error(404)
                    raise AssertionError("unexpected path")

        self.server = _PhoneServer(("127.0.0.1", 0), Handler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output = Path(self.temp_dir.name) / "phone.jsonl"

    async def asyncTearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)
        self.temp_dir.cleanup()

    async def test_success_records_commands_and_posts_matching_deterministic_results(self):
        await run_fake_phone(self.base_url, self.output, "success")

        recorded = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([entry["command"] for entry in recorded if entry["event"] == "command"], self.commands)
        self.assertEqual([result["id"] for result in self.results], ["command-1", "command-2"])
        self.assertEqual(self.results[0], {
            "id": "command-1", "status": "ok", "detail": "Fake phone completed sms"
        })
        self.assertEqual(self.results[1], {
            "id": "command-2", "status": "ok", "detail": "Fake phone location",
            "payload": {"lat": 47.6205, "lng": -122.3493, "accuracy_m": 8, "age_s": 3},
        })
        self.assertEqual([entry["result"] for entry in recorded if entry["event"] == "result"], self.results)

    async def test_command_after_five_second_idle_window_is_received_and_recorded(self):
        self.command_delay["seconds"] = 5.2

        await run_fake_phone(self.base_url, self.output, "success")

        recorded_commands = [
            entry["command"]
            for entry in map(json.loads, self.output.read_text().splitlines())
            if entry["event"] == "command"
        ]
        self.assertEqual(recorded_commands, self.commands)
        self.assertEqual([result["id"] for result in self.results], ["command-1", "command-2"])

    async def test_missing_fake_result_remains_a_failure(self):
        self.result_status["code"] = 503

        with self.assertRaisesRegex(RuntimeError, "phone result returned HTTP 503"):
            await run_fake_phone(self.base_url, self.output, "success")

    async def test_rejects_removed_and_unknown_modes(self):
        for mode in ("offline", "unknown", "other"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "unsupported fake phone mode"):
                await run_fake_phone(self.base_url, self.output, mode)

    async def test_rejects_non_loopback_urls_without_contacting_them(self):
        with self.assertRaises(ValueError):
            await run_fake_phone("http://example.com", self.output, "success")


if __name__ == "__main__":
    unittest.main()
