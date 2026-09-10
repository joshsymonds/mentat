"""Tests for the public front's long-lived OAuth token handling."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from authlib.jose import JsonWebKey, JsonWebToken
import fastmcp
from fastmcp.server.auth.providers.jwt import JWTVerifier
from mcp.server.auth.provider import AuthorizationParams
from mcp.shared.auth import OAuthClientInformationFull
from starlette.requests import Request

from public.front import LongLivedOIDCProxy, NoExpiryJWTVerifier
from public.tests.fake_upstream import FakeOIDCProvider


_RSA_KEY = JsonWebKey.generate_key("RSA", 2048, is_private=True)
_PRIVATE_PEM = _RSA_KEY.as_pem(is_private=True).decode()
_PUBLIC_PEM = _RSA_KEY.as_pem(is_private=False).decode()
_ISSUER = "https://test.cloudflareaccess.com"


def _make_token(*, expired: bool = False, issuer: str = _ISSUER, include_scope: bool = True) -> str:
    now = int(time.time())
    payload = {
        "sub": "test-user",
        "iss": issuer,
        "exp": now - 3600 if expired else now + 3600,
        "iat": now - 7200,
    }
    if include_scope:
        payload["scope"] = "openid"
    return JsonWebToken(["RS256"]).encode(
        {"alg": "RS256"}, payload, _PRIVATE_PEM
    ).decode()


def _make_verifier(cls, required_scopes: list[str] | None = None):
    return cls(public_key=_PUBLIC_PEM, issuer=_ISSUER, required_scopes=required_scopes)


class NoExpiryVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_valid_token_has_no_expiry_and_scopes(self) -> None:
        result = await _make_verifier(NoExpiryJWTVerifier).load_access_token(
            _make_token()
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIsNone(result.expires_at)
        self.assertEqual(result.scopes, ["openid"])

    async def test_token_without_scope_claim_is_accepted(self) -> None:
        result = await _make_verifier(NoExpiryJWTVerifier, ["openid"]).load_access_token(
            _make_token(include_scope=False)
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.scopes, [])
        self.assertIsNone(result.expires_at)

    async def test_bad_signature_is_rejected(self) -> None:
        other_key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        token = JsonWebToken(["RS256"]).encode(
            {"alg": "RS256"},
            {"sub": "attacker", "iss": _ISSUER, "exp": int(time.time()) + 3600},
            other_key.as_pem(is_private=True).decode(),
        ).decode()
        self.assertIsNone(
            await _make_verifier(NoExpiryJWTVerifier).load_access_token(token)
        )

    async def test_wrong_issuer_is_rejected(self) -> None:
        self.assertIsNone(
            await _make_verifier(NoExpiryJWTVerifier).load_access_token(
                _make_token(expired=True, issuer="https://evil.example.com")
            )
        )

    async def test_standard_verifier_still_rejects_expired_token(self) -> None:
        self.assertIsNone(
            await _make_verifier(JWTVerifier).load_access_token(_make_token(expired=True))
        )
        self.assertIsNotNone(
            await _make_verifier(NoExpiryJWTVerifier).load_access_token(
                _make_token(expired=True)
            )
        )


class LongLivedProxyTests(unittest.IsolatedAsyncioTestCase):
    async def test_exchange_strips_upstream_expiry_and_survives_restart(self) -> None:
        with FakeOIDCProvider() as upstream, tempfile.TemporaryDirectory() as home:
            client = OAuthClientInformationFull(
                client_id="mcp-client",
                client_secret=None,
                redirect_uris=["http://localhost/callback"],
                grant_types=["authorization_code"],
                response_types=["code"],
                scope="openid",
                token_endpoint_auth_method="none",
            )
            key = "a sufficiently long test secret"
            env = {"HOME": home}
            with patch.dict(os.environ, env, clear=False), patch.object(
                fastmcp.settings, "home", __import__("pathlib").Path(home)
            ):
                proxy = LongLivedOIDCProxy(
                    config_url=upstream.config_url,
                    client_id="access-client",
                    client_secret="access-secret",
                    base_url="https://front.test",
                    required_scopes=["openid"],
                    jwt_signing_key=key,
                    fallback_access_token_expiry_seconds=315360000,
                    require_authorization_consent="external",
                )
                proxy.get_routes("/mcp")
                await proxy.register_client(client)
                params = AuthorizationParams(
                    state="client-state",
                    scopes=["openid"],
                    code_challenge="client-challenge",
                    redirect_uri="http://localhost/callback",
                    redirect_uri_provided_explicitly=True,
                )
                upstream_url = await proxy.authorize(client, params)
                transaction_id = parse_qs(urlparse(upstream_url).query)["state"][0]
                request = Request(
                    {
                        "type": "http",
                        "method": "GET",
                        "path": "/auth/callback",
                        "query_string": f"code=fake-code&state={transaction_id}".encode(),
                        "headers": [],
                        "scheme": "https",
                        "server": ("front.test", 443),
                        "client": ("127.0.0.1", 1),
                    }
                )
                callback = await proxy._handle_idp_callback(request)
                client_code = parse_qs(urlparse(callback.headers["location"]).query)[
                    "code"
                ][0]
                authorization_code = await proxy.load_authorization_code(client, client_code)
                self.assertIsNotNone(authorization_code)
                assert authorization_code is not None
                token = await proxy.exchange_authorization_code(client, authorization_code)
                self.assertEqual(token.expires_in, 315360000)
                self.assertIsNotNone(token.access_token)

                restarted = LongLivedOIDCProxy(
                    config_url=upstream.config_url,
                    client_id="access-client",
                    client_secret="access-secret",
                    base_url="https://front.test",
                    required_scopes=["openid"],
                    jwt_signing_key=key,
                    fallback_access_token_expiry_seconds=315360000,
                    require_authorization_consent="external",
                )
                restarted.get_routes("/mcp")
                loaded = await restarted.load_access_token(token.access_token)
                self.assertIsNotNone(loaded)
                assert loaded is not None
                self.assertIsNone(loaded.expires_at)

                wrong_key = LongLivedOIDCProxy(
                    config_url=upstream.config_url,
                    client_id="access-client",
                    client_secret="access-secret",
                    base_url="https://front.test",
                    required_scopes=["openid"],
                    jwt_signing_key="different sufficiently long key",
                    fallback_access_token_expiry_seconds=315360000,
                    require_authorization_consent="external",
                )
                wrong_key.get_routes("/mcp")
                self.assertIsNone(await wrong_key.load_access_token(token.access_token))
