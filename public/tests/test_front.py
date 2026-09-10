"""Tests for the public FastMCP proxy and its OAuth HTTP surface."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

import fastmcp
import httpx
from fastmcp import FastMCP, Client
from fastmcp.exceptions import ToolError
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from starlette.requests import Request

from public.front import build_front, load_config
from public.tests.fake_upstream import FakeOIDCProvider


class FrontTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = FakeOIDCProvider()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.upstream.close()

    def config(self, server_url: str = "https://front.test", max_clients: int | None = None):
        env = {
                "ACCESS_CLIENT_ID": "client-id",
                "ACCESS_CLIENT_SECRET": "client-secret",
                "ACCESS_CONFIG_URL": self.upstream.config_url,
                "MCP_JWT_SECRET": "a sufficiently long test secret",
                "MCP_SERVER_URL": server_url,
                "MENTAT_PUBLIC_LISTEN": "127.0.0.1:8486",
            }
        if max_clients is not None:
            env["MENTAT_PUBLIC_MAX_CLIENTS"] = str(max_clients)
        return load_config(env)

    async def test_proxy_lists_and_forwards_backend_tools(self) -> None:
        backend = FastMCP("backend")

        @backend.tool
        def echo(text: str) -> str:
            return text

        @backend.tool
        def fail(text: str) -> str:
            raise ToolError(f"backend rejected {text}")

        front = build_front(backend, self.config())

        # The in-memory client intentionally bypasses HTTP authentication. This
        # isolates R1's proxy behavior from R2's authenticated HTTP surface.
        async with Client(backend) as backend_client, Client(front) as front_client:
            backend_tools = await backend_client.list_tools()
            front_tools = await front_client.list_tools()
            self.assertEqual(
                [tool.name for tool in front_tools], [tool.name for tool in backend_tools]
            )
            self.assertEqual(
                {tool.name: tool.inputSchema for tool in front_tools},
                {tool.name: tool.inputSchema for tool in backend_tools},
            )
            result = await front_client.call_tool("echo", {"text": "hello"})
            self.assertEqual(result.data, "hello")
            with self.assertRaisesRegex(ToolError, "backend rejected nope"):
                await front_client.call_tool("fail", {"text": "nope"})

    async def test_oauth_metadata_and_unauthenticated_challenge(self) -> None:
        backend = FastMCP("backend")
        config = self.config()
        front = build_front(backend, config)
        app = front.http_app()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://front.test",
        ) as client:
            protected = await client.get(
                "/.well-known/oauth-protected-resource/mcp"
            )
            self.assertEqual(protected.status_code, 200)
            self.assertEqual(protected.json()["resource"], config.server_url + "mcp")
            self.assertEqual(
                protected.json()["authorization_servers"], [config.server_url]
            )

            authorization = await client.get("/.well-known/oauth-authorization-server")
            self.assertEqual(authorization.status_code, 200)
            metadata = authorization.json()
            self.assertEqual(metadata["authorization_endpoint"], "https://front.test/authorize")
            self.assertEqual(metadata["token_endpoint"], "https://front.test/token")
            self.assertEqual(metadata["registration_endpoint"], "https://front.test/register")
            self.assertEqual(metadata["issuer"], config.server_url)
            self.assertIn("S256", metadata["code_challenge_methods_supported"])

            response = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            self.assertEqual(response.status_code, 401)
            self.assertIn(
                'resource_metadata="https://front.test/.well-known/oauth-protected-resource/mcp"',
                response.headers["www-authenticate"],
            )

    async def test_dynamic_registration_is_capped_across_reconstruction(self) -> None:
        registration = {
            "redirect_uris": ["http://localhost/callback"],
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "scope": "openid",
        }
        with tempfile.TemporaryDirectory() as home:
            backend = FastMCP("backend")
            config = self.config(max_clients=3)
            with patch.object(fastmcp.settings, "home", Path(home)):
                first = build_front(backend, config)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=first.http_app()),
                    base_url="https://front.test",
                ) as client:
                    first_response = await client.post("/register", json=registration)
                    self.assertEqual(first_response.status_code, 201)
                    client_info = OAuthClientInformationFull.model_validate(
                        first_response.json()
                    )
                    first.auth.get_routes("/mcp")
                    params = AuthorizationParams(
                        state="client-state",
                        scopes=["openid"],
                        code_challenge="client-challenge",
                        redirect_uri="http://localhost/callback",
                        redirect_uri_provided_explicitly=True,
                    )
                    consent_url = await first.auth.authorize(client_info, params)
                    txn_id = parse_qs(urlparse(consent_url).query)["txn_id"][0]
                    consent_page = await client.get(consent_url)
                    self.assertEqual(consent_page.status_code, 200)
                    csrf_match = re.search(
                        r'name="csrf_token" value="([^"]+)"', consent_page.text
                    )
                    self.assertIsNotNone(csrf_match)
                    assert csrf_match is not None
                    consent_response = await client.post(
                        "/consent",
                        data={
                            "txn_id": txn_id,
                            "csrf_token": csrf_match.group(1),
                            "action": "approve",
                        },
                        follow_redirects=False,
                    )
                    self.assertEqual(consent_response.status_code, 302)
                    upstream_url = consent_response.headers["location"]
                    transaction_id = parse_qs(urlparse(upstream_url).query)["state"][0]
                    cookie_header = "; ".join(
                        f"{key}={value}" for key, value in client.cookies.items()
                    )
                    request = Request(
                        {
                            "type": "http",
                            "method": "GET",
                            "path": "/auth/callback",
                            "query_string": (
                                f"code=fake-code&state={transaction_id}"
                            ).encode(),
                            "headers": [(b"cookie", cookie_header.encode())],
                            "scheme": "https",
                            "server": ("front.test", 443),
                            "client": ("127.0.0.1", 1),
                        }
                    )
                    callback = await first.auth._handle_idp_callback(request)
                    client_code = parse_qs(urlparse(callback.headers["location"]).query)[
                        "code"
                    ][0]
                    authorization_code = await first.auth.load_authorization_code(
                        client_info, client_code
                    )
                    self.assertIsNotNone(authorization_code)
                    assert authorization_code is not None
                    token = await first.auth.exchange_authorization_code(
                        client_info, authorization_code
                    )
                    self.assertIsNotNone(token.access_token)

                    remaining_responses = [
                        await client.post("/register", json=registration)
                        for _ in range(2)
                    ]
                self.assertEqual(
                    [response.status_code for response in remaining_responses],
                    [201, 201],
                )
                files_after_three = {
                    path.relative_to(home)
                    for path in Path(home).rglob("*")
                    if path.is_file()
                }

                second = build_front(backend, config)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=second.http_app()),
                    base_url="https://front.test",
                ) as client:
                    fourth = await client.post("/register", json=registration)
                self.assertEqual(fourth.status_code, 400)
                self.assertEqual(fourth.json()["error"], "invalid_client_metadata")
                files_after_four = {
                    path.relative_to(home)
                    for path in Path(home).rglob("*")
                    if path.is_file()
                }
                self.assertEqual(files_after_four, files_after_three)
                loaded = await second.auth.load_access_token(token.access_token)
                self.assertIsNotNone(loaded)
