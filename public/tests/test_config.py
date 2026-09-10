"""Tests for strict public-front configuration."""

from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from public.front import ConfigError, load_config, main, parse_listen


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = {
            "ACCESS_CLIENT_ID": "client-id",
            "ACCESS_CLIENT_SECRET": "client-secret",
            "ACCESS_CONFIG_URL": "https://issuer.example/.well-known/openid-configuration",
            "MCP_JWT_SECRET": "a sufficiently long test secret",
            "MCP_SERVER_URL": "https://front.example",
            "MENTAT_PUBLIC_LISTEN": "127.0.0.1:8486",
        }

    def test_missing_required_values_are_named(self) -> None:
        for variable in (
            "ACCESS_CLIENT_ID",
            "ACCESS_CLIENT_SECRET",
            "ACCESS_CONFIG_URL",
            "MCP_JWT_SECRET",
            "MCP_SERVER_URL",
        ):
            with self.subTest(variable=variable):
                env = self.env.copy()
                del env[variable]
                with self.assertRaisesRegex(ConfigError, variable):
                    load_config(env)

    def test_multiple_missing_values_are_named_together(self) -> None:
        env = self.env.copy()
        for variable in ("ACCESS_CLIENT_ID", "MCP_JWT_SECRET", "MCP_SERVER_URL"):
            del env[variable]
        with self.assertRaises(ConfigError) as raised:
            load_config(env)
        for variable in ("ACCESS_CLIENT_ID", "MCP_JWT_SECRET", "MCP_SERVER_URL"):
            self.assertIn(variable, str(raised.exception))

    def test_listen_must_be_loopback(self) -> None:
        with self.assertRaisesRegex(ConfigError, "loopback"):
            parse_listen("0.0.0.0:8486")
        self.assertEqual(parse_listen("127.0.0.1:8486"), ("127.0.0.1", 8486))
        self.assertEqual(parse_listen("[::1]:8486"), ("::1", 8486))
        self.assertEqual(parse_listen("localhost:8486"), ("localhost", 8486))

    def test_token_expiry_defaults_to_ten_years(self) -> None:
        self.assertEqual(load_config(self.env).token_expiry, 315360000)

    def test_server_url_is_canonical_with_one_trailing_slash(self) -> None:
        without_slash = load_config(self.env)
        with_slash = load_config({**self.env, "MCP_SERVER_URL": "https://front.example/"})
        self.assertEqual(without_slash.server_url, "https://front.example/")
        self.assertEqual(without_slash.server_url, with_slash.server_url)

    def test_max_clients_defaults_to_thirty_two_and_requires_positive_integer(self) -> None:
        self.assertEqual(load_config(self.env).max_clients, 32)
        with self.assertRaisesRegex(ConfigError, "MENTAT_PUBLIC_MAX_CLIENTS"):
            load_config({**self.env, "MENTAT_PUBLIC_MAX_CLIENTS": "0"})
        with self.assertRaisesRegex(ConfigError, "MENTAT_PUBLIC_MAX_CLIENTS"):
            load_config({**self.env, "MENTAT_PUBLIC_MAX_CLIENTS": "not-an-integer"})

    def test_main_reports_missing_configuration_and_exits(self) -> None:
        stderr = io.StringIO()
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)
        for variable in (
            "ACCESS_CLIENT_ID",
            "ACCESS_CLIENT_SECRET",
            "ACCESS_CONFIG_URL",
            "MCP_JWT_SECRET",
            "MCP_SERVER_URL",
        ):
            self.assertIn(variable, stderr.getvalue())
