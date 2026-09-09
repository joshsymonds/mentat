"""Loopback OIDC provider used by the public-front tests."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

from authlib.jose import JsonWebKey, JsonWebToken


class _Handler(BaseHTTPRequestHandler):
    provider: "FakeOIDCProvider"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _json(self, value: object, status: int = 200) -> None:
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        route = urlparse(self.path)
        if route.path == "/.well-known/openid-configuration":
            self._json(self.provider.configuration)
        elif route.path == "/jwks":
            self._json(self.provider.jwks)
        elif route.path == "/authorize":
            params = parse_qs(route.query)
            redirect_uri = params["redirect_uri"][0]
            callback = f"{redirect_uri}?{urlencode({'code': 'fake-code', 'state': params['state'][0]})}"
            self.send_response(302)
            self.send_header("Location", callback)
            self.end_headers()
        else:
            self._json({"error": "not_found"}, 404)

    def do_POST(self) -> None:
        route = urlparse(self.path)
        if route.path != "/token":
            self._json({"error": "not_found"}, 404)
            return

        size = int(self.headers.get("Content-Length", "0"))
        params = parse_qs(self.rfile.read(size).decode())
        if params.get("code", [""])[0] != "fake-code":
            self._json({"error": "invalid_grant"}, 400)
            return
        self._json(self.provider.token_response())


class FakeOIDCProvider:
    """A local, deterministic OIDC server with a signed test token."""

    def __init__(self) -> None:
        self._key = JsonWebKey.generate_key("RSA", 2048, is_private=True)
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.RequestHandlerClass.provider = self
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}"

    @property
    def config_url(self) -> str:
        return f"{self.base_url}/.well-known/openid-configuration"

    @property
    def configuration(self) -> dict[str, object]:
        return {
            "issuer": self.base_url,
            "authorization_endpoint": f"{self.base_url}/authorize",
            "token_endpoint": f"{self.base_url}/token",
            "jwks_uri": f"{self.base_url}/jwks",
            "scopes_supported": ["openid"],
            "response_types_supported": ["code"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "code_challenge_methods_supported": ["S256"],
        }

    @property
    def jwks(self) -> dict[str, object]:
        public = self._key.as_dict(is_private=False)
        public["kid"] = "fake-key"
        public["alg"] = "RS256"
        public["use"] = "sig"
        return {"keys": [public]}

    def token_response(self) -> dict[str, object]:
        now = int(time.time())
        payload = {
            "sub": "fake-user",
            "iss": self.base_url,
            "iat": now,
            "exp": now - 1,
            "scope": "openid",
        }
        token = JsonWebToken(["RS256"]).encode(
            {"alg": "RS256", "kid": "fake-key"},
            payload,
            self._key.as_pem(is_private=True).decode(),
        ).decode()
        return {
            "access_token": token,
            "id_token": token,
            "token_type": "Bearer",
            "expires_in": 86400,
            "scope": "openid",
        }

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def __enter__(self) -> "FakeOIDCProvider":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
