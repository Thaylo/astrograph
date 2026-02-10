"""Tests for deterministic LSP setup helpers."""

from __future__ import annotations

import json
import os
import socket
import sys
from unittest.mock import patch

import pytest

import astrograph.lsp_setup as lsp_setup
from astrograph.lsp_setup import (
    auto_bind_missing_servers,
    bundled_lsp_specs,
    collect_lsp_statuses,
    language_variant_policy,
    load_lsp_bindings,
    parse_attach_endpoint,
    probe_command,
    resolve_lsp_command,
    save_lsp_bindings,
)


def _write_compile_commands(path, *, file_name: str = "src/main.cpp") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "directory": str(path.parent.parent),
            "command": f"clang++ -std=c++14 -c {file_name}",
            "file": file_name,
        }
    ]
    path.write_text(json.dumps(payload))


def _make_fake_probe(endpoint: str):
    """Return a probe function that reports *endpoint* as available and everything else as not."""

    def _probe(command):
        parsed = lsp_setup.parse_command(command)
        if parsed == [endpoint]:
            host_port = endpoint.split("://", 1)[1]
            return {
                "command": parsed,
                "available": True,
                "executable": host_port,
                "transport": "tcp",
                "endpoint": host_port,
            }
        return {
            "command": parsed,
            "available": False,
            "executable": None,
            "transport": "subprocess",
            "endpoint": None,
        }

    return _probe


@pytest.mark.parametrize(
    ("binding", "expected_source", "expected_command"),
    [
        (["custom-pylsp"], "binding", ["custom-pylsp"]),
        (None, "default", ["pylsp"]),
    ],
)
def test_resolve_lsp_command_precedence(
    tmp_path,
    binding,
    expected_source,
    expected_command,
):
    if binding is not None:
        save_lsp_bindings({"python": binding}, workspace=tmp_path)

    command, source = resolve_lsp_command(
        language_id="python",
        default_command=("pylsp",),
        workspace=tmp_path,
    )

    assert source == expected_source
    assert command == expected_command


def test_auto_bind_missing_servers_uses_agent_observations(tmp_path):
    # No bindings → python is unconfigured → auto_bind should use the observation
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


def test_auto_bind_missing_servers_accepts_attach_observations(tmp_path, monkeypatch):
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


@pytest.mark.parametrize(
    ("detected", "expected_state", "reason_substring"),
    [
        ("clangd version 17.0.6", "supported", "clangd"),
        ("ccls version 0.20250117", "best_effort", "ccls"),
        ("mycpp-lsp 2.4.1", "best_effort", "Non-clangd C/C++ language server"),
    ],
    ids=["clangd_supported", "ccls_best_effort", "non_clangd_best_effort"],
)
def test_evaluate_version_status_cpp(detected, expected_state, reason_substring):
    status = lsp_setup._evaluate_version_status(
        language_id="cpp_lsp",
        detected=detected,
        probe_kind="server",
        transport="subprocess",
        available=True,
    )
    assert status["state"] == expected_state
    assert reason_substring in status["reason"]


def test_collect_lsp_statuses_cpp_fail_closed_when_reachable_only_in_production(
    tmp_path, monkeypatch
):
    endpoint = "tcp://127.0.0.1:2088"
    save_lsp_bindings({"cpp_lsp": [endpoint]}, workspace=tmp_path)

    monkeypatch.setenv("ASTROGRAPH_LSP_VALIDATION_MODE", "production")

    monkeypatch.setattr(lsp_setup, "probe_command", _make_fake_probe(endpoint))
    monkeypatch.setattr(
        lsp_setup,
        "_probe_attach_lsp_semantics",
        lambda **_kwargs: {
            "executed": True,
            "handshake_ok": False,
            "semantic_ok": False,
            "symbol_count": 0,
            "reason": "LSP initialize handshake failed.",
        },
    )
    monkeypatch.setattr(
        lsp_setup,
        "_compile_commands_status",
        lambda **_kwargs: {
            "required": True,
            "present": True,
            "readable": True,
            "valid": True,
            "entry_count": 1,
            "selected_path": "build/compile_commands.json",
            "paths": ["build/compile_commands.json"],
            "reason": None,
        },
    )

    statuses = collect_lsp_statuses(tmp_path)
    cpp = next(status for status in statuses if status["language"] == "cpp_lsp")
    assert cpp["probe_available"] is True
    assert cpp["available"] is False
    assert cpp["verification_state"] == "reachable_only"
    assert cpp["validation_mode"] == "production"


def test_compile_commands_status_prefers_workspace_build_path(tmp_path):
    stale = tmp_path / "demo_old" / "build" / "compile_commands.json"
    preferred = tmp_path / "build" / "compile_commands.json"

    _write_compile_commands(stale, file_name="src/stale.cpp")
    _write_compile_commands(preferred, file_name="src/preferred.cpp")
    os.utime(stale, (1_000, 1_000))
    os.utime(preferred, (500, 500))

    status = lsp_setup._compile_commands_status(language_id="cpp_lsp", workspace=tmp_path)
    assert status is not None
    assert status["valid"] is True
    assert status["selected_path"] == str(preferred)


def test_compile_commands_status_prefers_newer_nested_candidate(tmp_path):
    older = tmp_path / "demo_old" / "build" / "compile_commands.json"
    newer = tmp_path / "demo_new" / "build" / "compile_commands.json"

    _write_compile_commands(older, file_name="src/older.cpp")
    _write_compile_commands(newer, file_name="src/newer.cpp")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    status = lsp_setup._compile_commands_status(language_id="cpp_lsp", workspace=tmp_path)
    assert status is not None
    assert status["valid"] is True
    assert status["selected_path"] == str(newer)


def test_compile_commands_status_respects_override_env(tmp_path, monkeypatch):
    default_path = tmp_path / "build" / "compile_commands.json"
    override_path = tmp_path / "manual" / "compile_commands.json"

    _write_compile_commands(default_path, file_name="src/default.cpp")
    _write_compile_commands(override_path, file_name="src/override.cpp")
    monkeypatch.setenv("ASTROGRAPH_COMPILE_COMMANDS_PATH", str(override_path))

    status = lsp_setup._compile_commands_status(language_id="cpp_lsp", workspace=tmp_path)
    assert status is not None
    assert status["valid"] is True
    assert status["selected_path"] == str(override_path)


def test_collect_lsp_statuses_cpp_reachable_only_allowed_in_bootstrap_mode(tmp_path, monkeypatch):
    endpoint = "tcp://127.0.0.1:2088"
    save_lsp_bindings({"cpp_lsp": [endpoint]}, workspace=tmp_path)

    monkeypatch.setenv("ASTROGRAPH_LSP_VALIDATION_MODE", "bootstrap")

    monkeypatch.setattr(lsp_setup, "probe_command", _make_fake_probe(endpoint))
    monkeypatch.setattr(
        lsp_setup,
        "_probe_attach_lsp_semantics",
        lambda **_kwargs: {
            "executed": True,
            "handshake_ok": False,
            "semantic_ok": False,
            "symbol_count": 0,
            "reason": "LSP initialize handshake failed.",
        },
    )
    monkeypatch.setattr(
        lsp_setup,
        "_compile_commands_status",
        lambda **_kwargs: {
            "required": True,
            "present": False,
            "readable": False,
            "valid": False,
            "entry_count": 0,
            "selected_path": None,
            "paths": [],
            "reason": "No compile_commands.json found.",
        },
    )

    statuses = collect_lsp_statuses(tmp_path)
    cpp = next(status for status in statuses if status["language"] == "cpp_lsp")
    assert cpp["probe_available"] is True
    assert cpp["available"] is True
    assert cpp["verification_state"] == "reachable_only"
    assert cpp["validation_mode"] == "bootstrap"


def test_auto_bind_cpp_reports_validation_reason_when_reachable_only(tmp_path, monkeypatch):
    endpoint = "tcp://127.0.0.1:2088"
    monkeypatch.setenv("ASTROGRAPH_LSP_VALIDATION_MODE", "production")

    monkeypatch.setattr(
        lsp_setup,
        "probe_command",
        lambda _command: {
            "command": [endpoint],
            "available": True,
            "executable": "127.0.0.1:2088",
            "transport": "tcp",
            "endpoint": "127.0.0.1:2088",
        },
    )
    monkeypatch.setattr(
        lsp_setup,
        "_probe_attach_lsp_semantics",
        lambda **_kwargs: {
            "executed": True,
            "handshake_ok": True,
            "semantic_ok": False,
            "symbol_count": 0,
            "reason": "semantic probe failed",
        },
    )
    monkeypatch.setattr(
        lsp_setup,
        "_compile_commands_status",
        lambda **_kwargs: {
            "required": True,
            "present": False,
            "readable": False,
            "valid": False,
            "entry_count": 0,
            "selected_path": None,
            "paths": [],
            "reason": "compile_commands.json missing or invalid",
        },
    )

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

    assert result["changes"] == []
    unresolved = next(item for item in result["unresolved"] if item["language"] == "cpp_lsp")
    assert "compile_commands.json" in unresolved["reason"]


def test_relaxed_alias_resolves_to_bootstrap(monkeypatch):
    monkeypatch.setenv("ASTROGRAPH_LSP_VALIDATION_MODE", "relaxed")
    mode = lsp_setup._normalized_validation_mode()
    assert mode == "bootstrap"


@pytest.mark.parametrize(
    ("env_mode", "call_mode"),
    [("production", "bootstrap"), ("bootstrap", "production")],
    ids=["override_to_bootstrap", "override_to_production"],
)
def test_per_call_validation_mode_overrides_env(tmp_path, monkeypatch, env_mode, call_mode):
    """Per-call validation_mode takes priority over ASTROGRAPH_LSP_VALIDATION_MODE."""
    monkeypatch.setenv("ASTROGRAPH_LSP_VALIDATION_MODE", env_mode)
    statuses = collect_lsp_statuses(tmp_path, validation_mode=call_mode)
    for status in statuses:
        assert status["validation_mode"] == call_mode


def test_compile_db_path_overrides_env_and_discovery(tmp_path, monkeypatch):
    """Per-call compile_db_path takes priority over env override and auto-discovery."""
    default_path = tmp_path / "build" / "compile_commands.json"
    env_path = tmp_path / "env_override" / "compile_commands.json"
    explicit_path = tmp_path / "explicit" / "compile_commands.json"

    _write_compile_commands(default_path, file_name="src/default.cpp")
    _write_compile_commands(env_path, file_name="src/env.cpp")
    _write_compile_commands(explicit_path, file_name="src/explicit.cpp")

    monkeypatch.setenv("ASTROGRAPH_COMPILE_COMMANDS_PATH", str(env_path))

    status = lsp_setup._compile_commands_status(
        language_id="cpp_lsp",
        workspace=tmp_path,
        compile_db_path=str(explicit_path),
    )
    assert status is not None
    assert status["valid"] is True
    assert status["selected_path"] == str(explicit_path)


def test_project_root_scopes_compile_commands_search(tmp_path):
    """project_root narrows compile_commands.json discovery to a subdirectory."""
    sub_a = tmp_path / "project_a" / "build" / "compile_commands.json"
    sub_b = tmp_path / "project_b" / "build" / "compile_commands.json"

    _write_compile_commands(sub_a, file_name="src/a.cpp")
    _write_compile_commands(sub_b, file_name="src/b.cpp")

    status = lsp_setup._compile_commands_status(
        language_id="cpp_lsp",
        workspace=tmp_path,
        project_root=str(tmp_path / "project_a"),
    )
    assert status is not None
    assert status["valid"] is True
    assert "project_a" in status["selected_path"]


@pytest.mark.parametrize(
    ("validation_mode", "expect_available", "expect_state"),
    [
        ("production", False, "reachable_only"),
        ("bootstrap", True, "verified"),
    ],
    ids=["production_downgrades", "bootstrap_still_available"],
)
def test_subprocess_version_unsupported(
    tmp_path, monkeypatch, validation_mode, expect_available, expect_state
):
    """Unsupported version behaviour depends on validation mode."""
    monkeypatch.setattr(
        lsp_setup,
        "_run_version_probe",
        lambda _lang, _cmd: {"detected": "pylsp 0.1.0", "probe_kind": "server"},
    )
    monkeypatch.setattr(
        lsp_setup,
        "_evaluate_version_status",
        lambda **_kwargs: {
            "state": "unsupported",
            "detected": "pylsp 0.1.0",
            "probe_kind": "server",
            "reason": "python-lsp-server version is below supported baseline (1.11).",
            "variant_policy": {},
        },
    )

    probe = {"available": True, "executable": "/usr/bin/pylsp", "transport": "subprocess"}
    result = lsp_setup._availability_validation(
        language_id="python",
        command=["pylsp"],
        probe=probe,
        workspace=tmp_path,
        validation_mode=validation_mode,
    )
    assert result["available"] is expect_available
    assert result["verification"]["state"] == expect_state


# ---------------------------------------------------------------------------
# Unbundled architecture: all languages use TCP attach
# ---------------------------------------------------------------------------


class TestUnbundledLSPSpecs:
    """Verify bundled_lsp_specs() reflects the fully unbundled architecture."""

    def test_all_specs_use_tcp_defaults(self):
        """Every language must default to a tcp:// endpoint."""
        for spec in bundled_lsp_specs():
            cmd = " ".join(spec.default_command)
            assert cmd.startswith("tcp://"), (
                f"{spec.language_id} default_command should be tcp://, got {cmd}"
            )

    def test_all_specs_are_optional(self):
        """No language should be required — all are attach-mode."""
        for spec in bundled_lsp_specs():
            assert spec.required is False, f"{spec.language_id} should have required=False"

    def test_no_duplicate_ports(self):
        """Each language must have a unique TCP port."""
        ports = []
        for spec in bundled_lsp_specs():
            endpoint = parse_attach_endpoint(spec.default_command[0])
            assert endpoint is not None, f"{spec.language_id} is not a valid TCP endpoint"
            ports.append(endpoint["port"])
        assert len(ports) == len(set(ports)), f"Duplicate ports: {ports}"

    def test_plugin_defaults_match_specs(self):
        """Plugin DEFAULT_COMMAND constants must agree with bundled_lsp_specs()."""
        from astrograph.languages.c_lsp_plugin import CLSPPlugin
        from astrograph.languages.cpp_lsp_plugin import CppLSPPlugin
        from astrograph.languages.go_lsp_plugin import GoLSPPlugin
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin
        from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin
        from astrograph.languages.python_lsp_plugin import PythonLSPPlugin
        from astrograph.languages.typescript_lsp_plugin import TypeScriptLSPPlugin

        plugin_map = {
            "python": PythonLSPPlugin,
            "javascript_lsp": JavaScriptLSPPlugin,
            "typescript_lsp": TypeScriptLSPPlugin,
            "go_lsp": GoLSPPlugin,
            "c_lsp": CLSPPlugin,
            "cpp_lsp": CppLSPPlugin,
            "java_lsp": JavaLSPPlugin,
        }

        for spec in bundled_lsp_specs():
            plugin_cls = plugin_map.get(spec.language_id)
            if plugin_cls is None:
                continue
            assert spec.default_command == plugin_cls.DEFAULT_COMMAND, (
                f"{spec.language_id}: spec {spec.default_command} != "
                f"plugin {plugin_cls.DEFAULT_COMMAND}"
            )

    def test_expected_port_assignments(self):
        """Verify the canonical port mapping."""
        expected = {
            "c_lsp": 2087,
            "cpp_lsp": 2088,
            "java_lsp": 2089,
            "python": 2090,
            "go_lsp": 2091,
            "javascript_lsp": 2092,
            "typescript_lsp": 2093,
        }
        for spec in bundled_lsp_specs():
            endpoint = parse_attach_endpoint(spec.default_command[0])
            assert endpoint is not None
            assert endpoint["port"] == expected[spec.language_id], (
                f"{spec.language_id}: expected port {expected[spec.language_id]}, "
                f"got {endpoint['port']}"
            )


class TestFailFastUnconfigured:
    """Unconfigured languages (no binding, no env) must fail fast — no probe attempted."""

    def test_unconfigured_language_is_immediately_unavailable(self, tmp_path):
        """collect_lsp_statuses marks default-sourced languages as not_configured."""
        with patch.dict(os.environ, {}, clear=True):
            statuses = collect_lsp_statuses(workspace=tmp_path)

        for status in statuses:
            assert status["effective_source"] == "default"
            assert status["available"] is False
            assert status["verification_state"] == "not_configured"

    def test_configured_language_is_probed(self, tmp_path):
        """A bound language should NOT be marked not_configured."""
        save_lsp_bindings({"python": ["nonexistent-pylsp-xyz"]}, workspace=tmp_path)
        statuses = collect_lsp_statuses(workspace=tmp_path)

        python_status = next(s for s in statuses if s["language"] == "python")
        assert python_status["effective_source"] == "binding"
        # It was probed (not fast-failed) — verification_state reflects the probe result
        assert python_status["verification_state"] != "not_configured"
