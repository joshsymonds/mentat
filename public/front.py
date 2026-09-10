"""OAuth-authenticated public front for mentatd's loopback MCP endpoint."""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Mapping, SupportsFloat

from cryptography.fernet import Fernet
from fastmcp import FastMCP, settings
from fastmcp.server.auth.auth import AccessToken
from fastmcp.server.auth.oidc_proxy import OIDCProxy
from fastmcp.server.auth.jwt_issuer import derive_jwt_key
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.server import create_proxy
from key_value.aio.protocols import AsyncKeyValue
from key_value.aio.stores.filetree import (
    FileTreeStore,
    FileTreeV1CollectionSanitizationStrategy,
    FileTreeV1KeySanitizationStrategy,
)
from key_value.aio.wrappers.encryption import FernetEncryptionWrapper
from mcp.server.auth.provider import AuthorizationCode, OAuthToken, RegistrationError
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
    max_clients: int


class CappedClientStore:
    """Encrypted FastMCP storage with a bounded client-registration collection."""

    def __init__(
        self,
        store: AsyncKeyValue,
        file_store: FileTreeStore,
        max_clients: int,
    ) -> None:
        self._store = store
        self._file_store = file_store
        self._max_clients = max_clients
        self._registration_lock = asyncio.Lock()

    async def _client_count(self) -> int:
        await self._file_store.setup_collection(collection=CLIENT_COLLECTION)
        info = self._file_store._collection_infos[CLIENT_COLLECTION]
        count = 0
        async for _ in info._list_file_paths():
            count += 1
        return count

    async def get(
        self, key: str, *, collection: str | None = None
    ) -> dict[str, Any] | None:
        return await self._store.get(key=key, collection=collection)

    async def put(
        self,
        key: str,
        value: Mapping[str, Any],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        if collection != CLIENT_COLLECTION:
            await self._store.put(key=key, value=value, collection=collection, ttl=ttl)
            return

        async with self._registration_lock:
            if await self._client_count() >= self._max_clients:
                raise RegistrationError(
                    error="invalid_client_metadata",
                    error_description="Maximum client registrations reached",
                )
            await self._store.put(key=key, value=value, collection=collection, ttl=ttl)

    async def delete(self, key: str, *, collection: str | None = None) -> bool:
        return await self._store.delete(key=key, collection=collection)

    async def ttl(
        self, key: str, *, collection: str | None = None
    ) -> tuple[dict[str, Any] | None, float | None]:
        return await self._store.ttl(key=key, collection=collection)

    async def get_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> list[dict[str, Any] | None]:
        return await self._store.get_many(keys=keys, collection=collection)

    async def ttl_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> list[tuple[dict[str, Any] | None, float | None]]:
        return await self._store.ttl_many(keys=keys, collection=collection)

    async def put_many(
        self,
        keys: Sequence[str],
        values: Sequence[Mapping[str, Any]],
        *,
        collection: str | None = None,
        ttl: SupportsFloat | None = None,
    ) -> None:
        if collection != CLIENT_COLLECTION:
            await self._store.put_many(
                keys=keys, values=values, collection=collection, ttl=ttl
            )
            return

        async with self._registration_lock:
            existing = await self._client_count()
            new_keys = sum(
                1
                for key in keys
                if await self._store.get(key=key, collection=collection) is None
            )
            if existing + new_keys > self._max_clients:
                raise RegistrationError(
                    error="invalid_client_metadata",
                    error_description="Maximum client registrations reached",
                )
            await self._store.put_many(
                keys=keys, values=values, collection=collection, ttl=ttl
            )

    async def delete_many(
        self, keys: Sequence[str], *, collection: str | None = None
    ) -> int:
        return await self._store.delete_many(keys=keys, collection=collection)


def _build_client_storage(jwt_signing_key: bytes, max_clients: int) -> CappedClientStore:
    storage_encryption_key = derive_jwt_key(
        high_entropy_material=jwt_signing_key.decode(),
        salt="fastmcp-storage-encryption-key",
    )
    key_fingerprint = hashlib.sha256(storage_encryption_key).hexdigest()[:12]
    storage_dir = settings.home / "oauth-proxy" / key_fingerprint
    storage_dir.mkdir(parents=True, exist_ok=True)

    file_store = FileTreeStore(
        data_directory=storage_dir,
        key_sanitization_strategy=FileTreeV1KeySanitizationStrategy(storage_dir),
        collection_sanitization_strategy=FileTreeV1CollectionSanitizationStrategy(
            storage_dir
        ),
    )
    encrypted_store = FernetEncryptionWrapper(
        key_value=file_store,
        fernet=Fernet(key=storage_encryption_key),
        raise_on_decryption_error=False,
    )
    return CappedClientStore(encrypted_store, file_store, max_clients)


REQUIRED = (
    "ACCESS_CLIENT_ID",
    "ACCESS_CLIENT_SECRET",
    "ACCESS_CONFIG_URL",
    "MCP_JWT_SECRET",
    "MCP_SERVER_URL",
)
DEFAULT_LISTEN = "127.0.0.1:8486"
DEFAULT_TOKEN_EXPIRY = 315360000
DEFAULT_MAX_CLIENTS = 32
CLIENT_COLLECTION = "mcp-oauth-proxy-clients"


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

    raw_max_clients = values.get("MENTAT_PUBLIC_MAX_CLIENTS", str(DEFAULT_MAX_CLIENTS))
    try:
        max_clients = int(raw_max_clients)
    except ValueError as error:
        raise ConfigError("MENTAT_PUBLIC_MAX_CLIENTS must be an integer") from error
    if max_clients <= 0:
        raise ConfigError("MENTAT_PUBLIC_MAX_CLIENTS must be positive")

    return Config(
        access_client_id=values["ACCESS_CLIENT_ID"],
        access_client_secret=values["ACCESS_CLIENT_SECRET"],
        access_config_url=values["ACCESS_CONFIG_URL"],
        jwt_secret=values["MCP_JWT_SECRET"],
        server_url=values["MCP_SERVER_URL"].rstrip("/") + "/",
        backend_url=values.get("MENTAT_PUBLIC_BACKEND", "http://127.0.0.1:8484/mcp"),
        listen_host=listen_host,
        listen_port=listen_port,
        token_expiry=token_expiry,
        max_clients=max_clients,
    )


def build_front(backend: Any, config: Config) -> FastMCP:
    """Build a named FastMCP proxy around the loopback backend."""
    jwt_signing_key = derive_jwt_key(
        low_entropy_material=config.jwt_secret,
        salt="fastmcp-jwt-signing-key",
    )
    auth = LongLivedOIDCProxy(
        config_url=config.access_config_url,
        client_id=config.access_client_id,
        client_secret=config.access_client_secret,
        base_url=config.server_url,
        required_scopes=["openid"],
        jwt_signing_key=jwt_signing_key,
        fallback_access_token_expiry_seconds=config.token_expiry,
        client_storage=_build_client_storage(jwt_signing_key, config.max_clients),
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
