"""OAuth-authenticated public front for mentatd's loopback MCP endpoint."""

from __future__ import annotations

import ipaddress
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping

from fastmcp import FastMCP
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.server import create_proxy
from mcp.server.auth.provider import AuthorizationCode, OAuthToken
from mcp.shared.auth import OAuthClientInformationFull


class ConfigError(ValueError):
    """Raised when the public front has unsafe or incomplete configuration."""


@dataclass(frozen=True)
class Config:
    """Validated public-front configuration."""

    access_client_id: str
    access_client_secret: str
    access_config_url: str
    jwt_secret: str
    server_url: str
    backend_url: str
    listen_host: str
    listen_port: int
    token_expiry: int


REQUIRED = (
    "ACCESS_CLIENT_ID",
    "ACCESS_CLIENT_SECRET",
    "ACCESS_CONFIG_URL",
    "MCP_JWT_SECRET",
    "MCP_SERVER_URL",
)
DEFAULT_LISTEN = "127.0.0.1:8486"
DEFAULT_TOKEN_EXPIRY = 315360000


# Adapted from shimmer/shared/auth.py. This component intentionally carries its
# own copy so the public front does not depend on the private shimmer package.
class NoExpiryJWTVerifier(JWTVerifier):
    """Validate upstream JWT signatures without enforcing their expiration."""

    async def load_access_token(self, token: str) -> AccessToken | None:
        try:
            verification_key = await self._get_verification_key(token)
            claims = self.jwt.decode(token, verification_key)
            client_id = (
                claims.get("client_id")
                or claims.get("azp")
                or claims.get("sub")
                or "unknown"
            )

            if self.issuer:
                issuer = claims.get("iss")
                if isinstance(self.issuer, list):
                    issuer_valid = issuer in self.issuer
                else:
                    issuer_valid = issuer == self.issuer
                if not issuer_valid:
                    self.logger.debug("Upstream token issuer mismatch for client %s", client_id)
                    return None

            scopes = self._extract_scopes(claims)
            if self.required_scopes and not set(self.required_scopes).issubset(scopes):
                return None
            return AccessToken(
                token=token,
                client_id=str(client_id),
                scopes=scopes,
                expires_at=None,
                claims=claims,
            )
        except Exception as error:
            self.logger.debug("Upstream token validation failed: %s", error)
            return None


class LongLivedOIDCProxy(OIDCProxy):
    """OIDC proxy that keeps FastMCP tokens independent of Access expiry."""

    def get_token_verifier(
        self,
        *,
        algorithm: str | None = None,
        audience: str | None = None,
        required_scopes: list[str] | None = None,
        timeout_seconds: int | None = None,
    ) -> NoExpiryJWTVerifier:
        return NoExpiryJWTVerifier(
            jwks_uri=str(self.oidc_config.jwks_uri),
            issuer=str(self.oidc_config.issuer),
            algorithm=algorithm,
            audience=audience,
            required_scopes=required_scopes,
        )

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        code_model = await self._code_store.get(key=authorization_code.code)
        if code_model and "expires_in" in code_model.idp_tokens:
            code_model.idp_tokens.pop("expires_in")
            await self._code_store.put(key=authorization_code.code, value=code_model)
        return await super().exchange_authorization_code(client, authorization_code)


def parse_listen(value: str) -> tuple[str, int]:
    """Parse a host and port, accepting loopback hosts only."""
    if value.startswith("["):
        closing = value.find("]:")
        if closing < 0:
            raise ConfigError("MENTAT_PUBLIC_LISTEN must be host:port")
        host = value[1:closing]
        raw_port = value[closing + 2 :]
    else:
        try:
            host, raw_port = value.rsplit(":", 1)
        except ValueError as error:
            raise ConfigError("MENTAT_PUBLIC_LISTEN must be host:port") from error

    try:
        port = int(raw_port)
    except ValueError as error:
        raise ConfigError("MENTAT_PUBLIC_LISTEN port must be an integer") from error
    if not 1 <= port <= 65535:
        raise ConfigError("MENTAT_PUBLIC_LISTEN port must be between 1 and 65535")

    normalized_host = host.lower()
    is_loopback = normalized_host == "localhost"
    if not is_loopback:
        try:
            address = ipaddress.ip_address(normalized_host)
            is_loopback = address.is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ConfigError("MENTAT_PUBLIC_LISTEN must be a loopback address")
    return host, port


def load_config(env: Mapping[str, str] | None = None) -> Config:
    """Load and validate the required environment configuration."""
    values = os.environ if env is None else env
    missing = [name for name in REQUIRED if not values.get(name)]
    if missing:
        raise ConfigError("Missing required environment variables: " + ", ".join(missing))

    listen_host, listen_port = parse_listen(values.get("MENTAT_PUBLIC_LISTEN", DEFAULT_LISTEN))
    raw_expiry = values.get("MCP_TOKEN_EXPIRY", str(DEFAULT_TOKEN_EXPIRY))
    try:
        token_expiry = int(raw_expiry)
    except ValueError as error:
        raise ConfigError("MCP_TOKEN_EXPIRY must be an integer") from error
    if token_expiry <= 0:
        raise ConfigError("MCP_TOKEN_EXPIRY must be positive")

    return Config(
        access_client_id=values["ACCESS_CLIENT_ID"],
        access_client_secret=values["ACCESS_CLIENT_SECRET"],
        access_config_url=values["ACCESS_CONFIG_URL"],
        jwt_secret=values["MCP_JWT_SECRET"],
        server_url=values["MCP_SERVER_URL"].rstrip("/"),
        backend_url=values.get("MENTAT_PUBLIC_BACKEND", "http://127.0.0.1:8484/mcp"),
        listen_host=listen_host,
        listen_port=listen_port,
        token_expiry=token_expiry,
    )


def build_front(backend: Any, config: Config) -> FastMCP:
    """Build a named FastMCP proxy around the loopback backend."""
    auth = LongLivedOIDCProxy(
        config_url=config.access_config_url,
        client_id=config.access_client_id,
        client_secret=config.access_client_secret,
        base_url=config.server_url,
        required_scopes=["openid"],
        jwt_signing_key=config.jwt_secret,
        fallback_access_token_expiry_seconds=config.token_expiry,
    )
    return create_proxy(backend, name="mentat", auth=auth)


def main() -> None:
    """Run the OAuth front over HTTP."""
    try:
        config = load_config()
    except ConfigError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error

    # The public front only proxies the configured loopback daemon endpoint.
    front = build_front(config.backend_url, config)
    front.run(transport="http", host=config.listen_host, port=config.listen_port)


if __name__ == "__main__":
    main()
