"""Tests for deterministic LSP setup helpers."""

from __future__ import annotations

import os
import socket
import sys
from unittest.mock import patch

import pytest

from astrograph.lsp_setup import (
    auto_bind_missing_servers,
    collect_lsp_statuses,
    language_variant_policy,
    load_lsp_bindings,
    parse_attach_endpoint,
    probe_command,
    resolve_lsp_command,
    save_lsp_bindings,
)


@pytest.mark.parametrize(
    ("binding", "env_command", "expected_source", "expected_command"),
    [
        (["custom-pylsp"], "env-pylsp", "binding", ["custom-pylsp"]),
        (None, "python -m pylsp", "env", ["python", "-m", "pylsp"]),
    ],
)
def test_resolve_lsp_command_precedence(
    tmp_path,
    binding,
    env_command,
    expected_source,
    expected_command,
):
    if binding is not None:
        save_lsp_bindings({"python": binding}, workspace=tmp_path)

    with patch.dict(os.environ, {"ASTROGRAPH_PY_LSP_COMMAND": env_command}, clear=False):
        command, source = resolve_lsp_command(
            language_id="python",
            default_command=("pylsp",),
            command_env_var="ASTROGRAPH_PY_LSP_COMMAND",
            workspace=tmp_path,
        )

    assert source == expected_source
    assert command == expected_command


def test_auto_bind_missing_servers_uses_agent_observations(tmp_path):
    with patch.dict(
        os.environ,
        {
            "ASTROGRAPH_PY_LSP_COMMAND": "missing-python-lsp-xyz",
            "ASTROGRAPH_JS_LSP_COMMAND": "missing-js-lsp-xyz",
        },
        clear=False,
    ):
        result = auto_bind_missing_servers(
            workspace=tmp_path,
            observations=[
                {
                    "language": "python",
                    "command": [sys.executable, "-m", "pylsp"],
                }
            ],
        )

    python_change = next(change for change in result["changes"] if change["language"] == "python")
    assert python_change["source"] == "observation"
    assert python_change["command"][0] == sys.executable

    persisted = load_lsp_bindings(tmp_path)
    assert persisted["python"][0] == sys.executable


def test_parse_attach_endpoint_tcp():
    parsed = parse_attach_endpoint("tcp://127.0.0.1:2088")
    assert parsed == {
        "transport": "tcp",
        "host": "127.0.0.1",
        "port": 2088,
        "target": "127.0.0.1:2088",
    }


def test_probe_command_attach_tcp_endpoint_available():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        _host, port = server.getsockname()

        probe = probe_command(f"tcp://127.0.0.1:{port}")

    assert probe["available"] is True
    assert probe["transport"] == "tcp"
    assert probe["endpoint"] == f"127.0.0.1:{port}"


def test_probe_command_attach_tcp_endpoint_falls_back_across_address_families(monkeypatch):
    class _FakeSocket:
        attempts = 0

        def __init__(self, _family, _socktype, _proto):
            pass

        def settimeout(self, _timeout):
            pass

        def connect(self, _sockaddr):
            _FakeSocket.attempts += 1
            if _FakeSocket.attempts == 1:
                raise OSError("unreachable")

        def close(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _tb):
            return False

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (
                socket.AF_INET6,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("fdc4:f303:9324::254", 2088, 0, 0),
            ),
            (
                socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                ("192.168.65.254", 2088),
            ),
        ],
    )
    monkeypatch.setattr(socket, "socket", _FakeSocket)

    probe = probe_command("tcp://host.docker.internal:2088")
    assert probe["available"] is True
    assert probe["transport"] == "tcp"
    assert _FakeSocket.attempts == 2


def test_auto_bind_missing_servers_accepts_attach_observations(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        _host, port = server.getsockname()
        endpoint = f"tcp://127.0.0.1:{port}"

        result = auto_bind_missing_servers(
            workspace=tmp_path,
            observations=[
                {
                    "language": "c_lsp",
                    "command": endpoint,
                }
            ],
        )

    c_change = next(change for change in result["changes"] if change["language"] == "c_lsp")
    assert c_change["source"] == "observation"
    assert c_change["transport"] == "tcp"
    assert c_change["command"] == [endpoint]

    persisted = load_lsp_bindings(tmp_path)
    assert persisted["c_lsp"] == [endpoint]


def test_auto_bind_missing_servers_scopes_languages(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        _host, port = server.getsockname()
        endpoint = f"tcp://127.0.0.1:{port}"

        result = auto_bind_missing_servers(
            workspace=tmp_path,
            languages=["cpp_lsp"],
            observations=[
                {
                    "language": "cpp_lsp",
                    "command": endpoint,
                }
            ],
        )

    assert result["scope_languages"] == ["cpp_lsp"]
    assert all(status["language"] == "cpp_lsp" for status in result["statuses"])
    assert set(result["probes"]) == {"cpp_lsp"}


def test_collect_lsp_statuses_include_version_and_variant_metadata(tmp_path):
    statuses = collect_lsp_statuses(tmp_path)
    assert statuses
    for status in statuses:
        assert "version_status" in status
        assert "state" in status["version_status"]
        assert "language_variant_policy" in status


def test_language_variant_policy_scoped():
    scoped = language_variant_policy("cpp_lsp")
    assert set(scoped) == {"cpp_lsp"}
    assert "supported" in scoped["cpp_lsp"]
