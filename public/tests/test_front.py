"""Tests for the public FastMCP proxy and its OAuth HTTP surface."""

from __future__ import annotations

import unittest

import httpx
from fastmcp import FastMCP, Client
from fastmcp.exceptions import ToolError

from public.front import build_front, load_config
from public.tests.fake_upstream import FakeOIDCProvider


class FrontTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.upstream = FakeOIDCProvider()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.upstream.close()

    def config(self, server_url: str = "https://front.test"):
        return load_config(
            {
                "ACCESS_CLIENT_ID": "client-id",
                "ACCESS_CLIENT_SECRET": "client-secret",
                "ACCESS_CONFIG_URL": self.upstream.config_url,
                "MCP_JWT_SECRET": "a sufficiently long test secret",
                "MCP_SERVER_URL": server_url,
                "MENTAT_PUBLIC_LISTEN": "127.0.0.1:8486",
            }
        )

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
        front = build_front(backend, self.config())
        app = front.http_app()

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="https://front.test",
        ) as client:
            protected = await client.get(
                "/.well-known/oauth-protected-resource/mcp"
            )
            self.assertEqual(protected.status_code, 200)
            self.assertEqual(protected.json()["resource"], "https://front.test/mcp")
            self.assertEqual(
                protected.json()["authorization_servers"], ["https://front.test/"]
            )

            authorization = await client.get("/.well-known/oauth-authorization-server")
            self.assertEqual(authorization.status_code, 200)
            metadata = authorization.json()
            self.assertEqual(metadata["authorization_endpoint"], "https://front.test/authorize")
            self.assertEqual(metadata["token_endpoint"], "https://front.test/token")
            self.assertEqual(metadata["registration_endpoint"], "https://front.test/register")
            self.assertIn("S256", metadata["code_challenge_methods_supported"])

            response = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
            self.assertEqual(response.status_code, 401)
            self.assertIn(
                'resource_metadata="https://front.test/.well-known/oauth-protected-resource/mcp"',
                response.headers["www-authenticate"],
            )
