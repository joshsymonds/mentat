"""Deterministic loopback-only fake phone for offline voice evaluations."""

import asyncio
import http.client
import ipaddress
import json
import math
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Literal
from urllib.parse import urlsplit

Mode = Literal["success"]
_LOCATION = {"lat": 47.6205, "lng": -122.3493, "accuracy_m": 8, "age_s": 3}


def _loopback_endpoint(base_url: str) -> tuple[str, int]:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.hostname is None:
        raise ValueError("fake phone URL must be an HTTP loopback IP address")
    try:
        address = ipaddress.ip_address(parsed.hostname)
        port = parsed.port or 80
    except ValueError as error:
        raise ValueError("fake phone URL must be an HTTP loopback IP address") from error
    if not address.is_loopback or parsed.username is not None or parsed.password is not None:
        raise ValueError("fake phone URL must be an HTTP loopback IP address")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("fake phone URL must contain only the loopback origin")
    return parsed.hostname, port


def _result(command: dict[str, object]) -> dict[str, object]:
    command_id = command.get("id")
    kind = command.get("kind")
    if not isinstance(command_id, str) or not command_id or not isinstance(kind, str):
        raise ValueError("phone command must contain a non-empty id and kind")
    if kind == "location":
        return {
            "id": command_id,
            "status": "ok",
            "detail": "Fake phone location",
            "payload": _LOCATION,
        }
    return {"id": command_id, "status": "ok", "detail": f"Fake phone completed {kind}"}


def _record(output: Path, entry: dict[str, object]) -> None:
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, sort_keys=True) + "\n")


def _run_fake_phone(
    base_url: str,
    output_path: Path,
    clock: Callable[[], float],
    shutdown_requested: threading.Event,
    result_in_progress: threading.Event,
    command_connection_ref: list[http.client.HTTPConnection | None],
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")

    host, port = _loopback_endpoint(base_url)
    connection = http.client.HTTPConnection(host, port, timeout=25)
    command_connection_ref[0] = connection
    try:
        connection.request("GET", "/v1/phone/commands", headers={"X-Mentat-Phone": "fake-phone"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"phone command stream returned HTTP {response.status}")
        while not shutdown_requested.is_set():
            try:
                line = response.readline()
            except OSError:
                if shutdown_requested.is_set():
                    break
                raise
            if not line:
                break
            received_at = clock()
            if not math.isfinite(received_at):
                raise ValueError("phone command receipt timestamp must be finite")
            if not line.strip():
                continue
            command = json.loads(line)
            if not isinstance(command, dict):
                raise ValueError("phone command must be a JSON object")
            if command.get("kind") == "ping":
                continue
            result_in_progress.set()
            try:
                _record(output, {"event": "command", "command": command, "received_at": received_at})
                result = _result(command)
                result_connection = http.client.HTTPConnection(host, port, timeout=5)
                try:
                    result_connection.request(
                        "POST",
                        "/v1/phone/results",
                        body=json.dumps(result),
                        headers={"Content-Type": "application/json"},
                    )
                    result_response = result_connection.getresponse()
                    result_response.read()
                    if result_response.status != 204:
                        raise RuntimeError(f"phone result returned HTTP {result_response.status}")
                finally:
                    result_connection.close()
                _record(output, {"event": "result", "result": result})
            finally:
                result_in_progress.clear()
    finally:
        command_connection_ref[0] = None
        connection.close()


async def run_fake_phone(
    base_url: str,
    output_path: str | Path,
    mode: Mode,
    *,
    clock: Callable[[], float] = time.time,
) -> None:
    """Listen for commands and record loopback-only deterministic results."""
    if mode != "success":
        raise ValueError(f"unsupported fake phone mode: {mode}")
    _loopback_endpoint(base_url)
    shutdown_requested = threading.Event()
    result_in_progress = threading.Event()
    command_connection_ref: list[http.client.HTTPConnection | None] = [None]

    def handle_shutdown(_signum: int, _frame: object) -> None:
        shutdown_requested.set()
        if not result_in_progress.is_set() and command_connection_ref[0] is not None:
            connection = command_connection_ref[0]
            if connection.sock is not None:
                try:
                    connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    previous_handlers = {
        signum: signal.signal(signum, handle_shutdown)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        await asyncio.to_thread(
            _run_fake_phone,
            base_url,
            Path(output_path),
            clock,
            shutdown_requested,
            result_in_progress,
            command_connection_ref,
        )
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
