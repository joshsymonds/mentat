import asyncio
import http.server
import json
import sys
import tempfile
import threading
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
                for command in commands:
                    payload = (json.dumps(command) + "\n").encode()
                    self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

            def do_POST(self):
                self.assert_result_path()
                body = self.rfile.read(int(self.headers["Content-Length"]))
                results.append(json.loads(body))
                self.send_response(204)
                self.send_header("Content-Length", "0")
                self.end_headers()

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

    async def test_offline_mode_does_not_connect_or_post_success(self):
        await run_fake_phone(self.base_url, self.output, "offline")

        self.assertEqual(self.results, [])
        self.assertEqual(self.output.read_text(), "")

    async def test_unknown_mode_posts_only_error_results(self):
        await run_fake_phone(self.base_url, self.output, "unknown")

        self.assertEqual([result["id"] for result in self.results], ["command-1", "command-2"])
        self.assertTrue(all(result["status"] == "error" for result in self.results))
        self.assertTrue(all("unknown" in result["detail"] for result in self.results))
        recorded = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual([entry["command"] for entry in recorded if entry["event"] == "command"], self.commands)
        self.assertEqual([entry["result"] for entry in recorded if entry["event"] == "result"], self.results)

    async def test_rejects_non_loopback_urls_without_contacting_them(self):
        with self.assertRaises(ValueError):
            await run_fake_phone("http://example.com", self.output, "success")


if __name__ == "__main__":
    unittest.main()
