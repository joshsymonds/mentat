"""Deterministic loopback-only fake phone for offline voice evaluations."""

import asyncio
import http.client
import ipaddress
import json
from pathlib import Path
from typing import Literal
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


def _run_fake_phone(base_url: str, output_path: Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")

    host, port = _loopback_endpoint(base_url)
    connection = http.client.HTTPConnection(host, port, timeout=5)
    try:
        connection.request("GET", "/v1/phone/commands", headers={"X-Mentat-Phone": "fake-phone"})
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"phone command stream returned HTTP {response.status}")
        while line := response.readline():
            if not line.strip():
                continue
            command = json.loads(line)
            if not isinstance(command, dict):
                raise ValueError("phone command must be a JSON object")
            if command.get("kind") == "ping":
                continue
            _record(output, {"event": "command", "command": command})
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
        connection.close()


async def run_fake_phone(base_url: str, output_path: str | Path, mode: Mode) -> None:
    """Listen for commands and record loopback-only deterministic results."""
    if mode != "success":
        raise ValueError(f"unsupported fake phone mode: {mode}")
    _loopback_endpoint(base_url)
    await asyncio.to_thread(_run_fake_phone, base_url, Path(output_path))
