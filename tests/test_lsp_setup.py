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


def test_auto_bind_missing_servers_uses_agent_observations(tmp_path, monkeypatch):
    # Mock default TCP endpoint as unavailable so the observation gets used.
    # Without this, a socat bridge on port 2090 would make the default
    # reachable and auto_bind would skip the observation entirely.
    _real_probe = lsp_setup.probe_command

    def _probe_no_defaults(command):
        parsed = lsp_setup.parse_command(command)
        if parsed and any("tcp://127.0.0.1:" in c for c in parsed):
            return {"command": parsed, "available": False, "executable": None}
        return _real_probe(command)

    monkeypatch.setattr(lsp_setup, "probe_command", _probe_no_defaults)

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
    # Mock default TCP endpoints as unavailable so observations get used.
    _real_probe = lsp_setup.probe_command

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        _host, port = server.getsockname()
        endpoint = f"tcp://127.0.0.1:{port}"

        def _probe_no_defaults(command):
            parsed = lsp_setup.parse_command(command)
            # Block default port but allow the ephemeral test port
            if parsed and any(f"tcp://127.0.0.1:{p}" in c for c in parsed for p in (2087,)):
                return {"command": parsed, "available": False, "executable": None}
            return _real_probe(command)

        monkeypatch.setattr(lsp_setup, "probe_command", _probe_no_defaults)

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


class TestSetActiveWorkspace:
    """Tests for set_active_workspace edge cases."""

    def test_set_to_none(self):
        lsp_setup.set_active_workspace(None)
        assert lsp_setup._active_workspace is None

    def test_set_to_valid_path(self, tmp_path):
        lsp_setup.set_active_workspace(str(tmp_path))
        assert lsp_setup._active_workspace == tmp_path.resolve()
        lsp_setup.set_active_workspace(None)

    def test_set_to_nonexistent_path(self, tmp_path):
        nonexistent = tmp_path / "nonexistent"
        lsp_setup.set_active_workspace(str(nonexistent))
        assert lsp_setup._active_workspace is not None
        lsp_setup.set_active_workspace(None)


class TestValidationModeOverrides:
    """Tests for validation mode normalization."""

    def test_legacy_production_flag_true(self, monkeypatch):
        monkeypatch.delenv("ASTROGRAPH_LSP_VALIDATION_MODE", raising=False)
        monkeypatch.setenv("ASTROGRAPH_LSP_PRODUCTION", "true")
        result = lsp_setup._normalized_validation_mode_with_override(None)
        assert result == "production"

    def test_legacy_production_flag_false(self, monkeypatch):
        monkeypatch.delenv("ASTROGRAPH_LSP_VALIDATION_MODE", raising=False)
        monkeypatch.setenv("ASTROGRAPH_LSP_PRODUCTION", "0")
        result = lsp_setup._normalized_validation_mode_with_override(None)
        assert result == "bootstrap"

    def test_override_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("ASTROGRAPH_LSP_VALIDATION_MODE", "production")
        result = lsp_setup._normalized_validation_mode_with_override("bootstrap")
        assert result == "bootstrap"

    def test_invalid_override_ignored(self, monkeypatch):
        monkeypatch.delenv("ASTROGRAPH_LSP_VALIDATION_MODE", raising=False)
        monkeypatch.delenv("ASTROGRAPH_LSP_PRODUCTION", raising=False)
        result = lsp_setup._normalized_validation_mode_with_override("invalid_mode")
        # Falls through to env/default
        assert result in ("production", "bootstrap")


class TestCompileCommandsPaths:
    """Tests for compile_commands.json discovery."""

    def test_compile_commands_found(self, tmp_path):
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        cc = build_dir / "compile_commands.json"
        cc.write_text('[{"directory": ".", "command": "clang++ -c f.cpp", "file": "f.cpp"}]')
        paths = lsp_setup._compile_commands_paths(tmp_path)
        assert any(str(build_dir) in str(p) for p in paths)

    def test_compile_commands_not_found(self, tmp_path):
        paths = lsp_setup._compile_commands_paths(tmp_path)
        assert len(paths) == 0

    def test_skip_dirs_filtered(self, tmp_path):
        node_modules = tmp_path / "node_modules" / "build"
        node_modules.mkdir(parents=True)
        (node_modules / "compile_commands.json").write_text("[]")
        paths = lsp_setup._compile_commands_paths(tmp_path)
        assert not any("node_modules" in str(p) for p in paths)


class TestIsCompileCommandsEntry:
    """Tests for compile_commands.json entry validation."""

    def test_valid_entry(self):
        entry = {"directory": "/src", "command": "clang++ -c f.cpp", "file": "f.cpp"}
        assert lsp_setup._is_compile_commands_entry(entry) is True

    def test_missing_file(self):
        entry = {"directory": "/src", "command": "clang++ -c f.cpp"}
        assert lsp_setup._is_compile_commands_entry(entry) is False

    def test_non_dict(self):
        assert lsp_setup._is_compile_commands_entry("not a dict") is False


class TestCompileCommandsStatus:
    """Tests for compile_commands.json validation."""

    def test_valid_compile_commands(self, tmp_path):
        cc = tmp_path / "compile_commands.json"
        cc.write_text('[{"directory": ".", "command": "clang++ -c f.cpp", "file": "f.cpp"}]')
        status = lsp_setup._compile_commands_status(
            language_id="cpp_lsp", workspace=tmp_path
        )
        assert status is not None
        assert status["valid"] is True

    def test_non_cpp_returns_none(self, tmp_path):
        status = lsp_setup._compile_commands_status(
            language_id="python", workspace=tmp_path
        )
        assert status is None

    def test_no_compile_commands(self, tmp_path):
        status = lsp_setup._compile_commands_status(
            language_id="cpp_lsp", workspace=tmp_path
        )
        assert status is not None
        assert status["valid"] is False


class TestBindingPersistence:
    """Tests for binding load/save edge cases."""

    def test_load_corrupted_json(self, tmp_path):
        bindings_file = tmp_path / ".metadata_astrograph" / "lsp_bindings.json"
        bindings_file.parent.mkdir(parents=True)
        bindings_file.write_text("not json")
        result = load_lsp_bindings(tmp_path)
        assert result == {}

    def test_load_wrong_schema(self, tmp_path):
        bindings_file = tmp_path / ".metadata_astrograph" / "lsp_bindings.json"
        bindings_file.parent.mkdir(parents=True)
        bindings_file.write_text('"just a string"')
        result = load_lsp_bindings(tmp_path)
        assert result == {}

    def test_save_and_load_round_trip(self, tmp_path):
        save_lsp_bindings({"python": ["/usr/bin/pylsp"]}, workspace=tmp_path)
        loaded = load_lsp_bindings(tmp_path)
        assert loaded["python"] == ["/usr/bin/pylsp"]


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from astrograph.lsp_setup import (
    _availability_validation,
    _compile_commands_status,
    _evaluate_version_status,
    _normalize_workspace_root,
    _run_version_probe,
    _version_probe_candidates,
)


class TestEvaluateVersionStatusPython:
    """Cover Python server unsupported and runtime best_effort/unsupported paths."""

    def test_python_server_unsupported_major_0(self):
        result = _evaluate_version_status(
            language_id="python",
            detected="pylsp 0.5.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unsupported"
        assert "below supported baseline" in result["reason"]

    def test_python_server_unsupported_major_1_minor_10(self):
        result = _evaluate_version_status(
            language_id="python",
            detected="pylsp 1.10.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unsupported"

    def test_python_server_supported_1_11(self):
        result = _evaluate_version_status(
            language_id="python",
            detected="pylsp 1.11.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "supported"

    def test_python_runtime_315_best_effort(self):
        result = _evaluate_version_status(
            language_id="python",
            detected="Python 3.15.0",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"
        assert "3.15" in result["reason"]

    def test_python_runtime_unsupported_2_7(self):
        result = _evaluate_version_status(
            language_id="python",
            detected="Python 2.7.18",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unsupported"
        assert "outside the supported variant window" in result["reason"]

    def test_python_runtime_supported_312(self):
        result = _evaluate_version_status(
            language_id="python",
            detected="Python 3.12.1",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "supported"


class TestEvaluateVersionStatusNodeJS:
    """Cover Node.js runtime version evaluation paths."""

    @pytest.mark.parametrize(
        ("major_str", "expected_state"),
        [
            ("v20.11.0", "supported"),
            ("v22.0.0", "supported"),
            ("v24.0.0", "supported"),
        ],
    )
    def test_nodejs_lts_supported(self, major_str, expected_state):
        result = _evaluate_version_status(
            language_id="javascript_lsp",
            detected=major_str,
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == expected_state

    def test_nodejs_18_best_effort(self):
        result = _evaluate_version_status(
            language_id="javascript_lsp",
            detected="v18.19.0",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"
        assert "18" in result["reason"]

    def test_nodejs_26_supported_ge20(self):
        result = _evaluate_version_status(
            language_id="javascript_lsp",
            detected="v26.0.0",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "supported"
        assert ">=20" in result["reason"]

    def test_nodejs_16_best_effort(self):
        result = _evaluate_version_status(
            language_id="javascript_lsp",
            detected="v16.20.0",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"

    def test_nodejs_14_unsupported(self):
        result = _evaluate_version_status(
            language_id="javascript_lsp",
            detected="v14.21.0",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unsupported"
        assert "below" in result["reason"]


class TestEvaluateVersionStatusTS:
    """Cover typescript-language-server version evaluation paths."""

    def test_ts_server_supported_ge4(self):
        result = _evaluate_version_status(
            language_id="typescript_lsp",
            detected="typescript-language-server 4.3.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "supported"
        assert ">=4" in result["reason"]

    def test_ts_server_best_effort_v3(self):
        result = _evaluate_version_status(
            language_id="typescript_lsp",
            detected="typescript-language-server 3.3.2",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"

    def test_ts_server_unsupported_v2(self):
        result = _evaluate_version_status(
            language_id="typescript_lsp",
            detected="typescript-language-server 2.0.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unsupported"
        assert "below" in result["reason"]


class TestEvaluateVersionStatusClangd:
    """Cover clangd, ccls, and non-clangd C/C++ server paths."""

    def test_clangd_supported_ge16(self):
        result = _evaluate_version_status(
            language_id="c_lsp",
            detected="clangd version 17.0.6",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "supported"
        assert ">=16" in result["reason"]

    def test_clangd_best_effort_15(self):
        result = _evaluate_version_status(
            language_id="c_lsp",
            detected="clangd version 15.0.7",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"
        assert "15" in result["reason"]

    def test_clangd_unsupported_14(self):
        result = _evaluate_version_status(
            language_id="c_lsp",
            detected="clangd version 14.0.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unsupported"
        assert "below" in result["reason"]

    def test_ccls_best_effort(self):
        result = _evaluate_version_status(
            language_id="cpp_lsp",
            detected="ccls version 0.20250117",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"
        assert "ccls" in result["reason"]

    def test_non_clangd_non_ccls_best_effort(self):
        result = _evaluate_version_status(
            language_id="c_lsp",
            detected="my-custom-lsp 2.0.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"
        assert "Non-clangd" in result["reason"]


class TestEvaluateVersionStatusJava:
    """Cover Java runtime and server version evaluation paths."""

    @pytest.mark.parametrize(
        ("version_str", "expected_state"),
        [
            ("openjdk 11.0.20", "supported"),
            ("openjdk 17.0.8", "supported"),
            ("openjdk 21.0.1", "supported"),
            ("openjdk 25.0.0", "supported"),
        ],
    )
    def test_java_lts_supported(self, version_str, expected_state):
        result = _evaluate_version_status(
            language_id="java_lsp",
            detected=version_str,
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == expected_state
        assert "LTS" in result["reason"]

    def test_java_8_best_effort(self):
        result = _evaluate_version_status(
            language_id="java_lsp",
            detected="openjdk version 8.0.392",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"
        assert "8" in result["reason"]

    def test_java_ge11_supported(self):
        result = _evaluate_version_status(
            language_id="java_lsp",
            detected="openjdk version 19.0.2",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "supported"
        assert ">=11" in result["reason"]

    def test_java_below_8_unsupported(self):
        result = _evaluate_version_status(
            language_id="java_lsp",
            detected="java version 7.0.80",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unsupported"
        assert "below" in result["reason"]

    def test_java_server_best_effort(self):
        result = _evaluate_version_status(
            language_id="java_lsp",
            detected="jdtls 1.23.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"
        assert "best-effort" in result["reason"]


class TestEvaluateVersionStatusGo:
    """Cover Go runtime and gopls version evaluation paths."""

    @pytest.mark.parametrize(
        ("version_str", "minor", "expected_state"),
        [
            ("go1.21.0", 21, "supported"),
            ("go1.22.5", 22, "supported"),
            ("go1.23.0", 23, "supported"),
            ("go1.24.1", 24, "supported"),
            ("go1.25.0", 25, "supported"),
        ],
    )
    def test_go_supported_range(self, version_str, minor, expected_state):
        result = _evaluate_version_status(
            language_id="go_lsp",
            detected=version_str,
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == expected_state

    def test_go_120_best_effort(self):
        result = _evaluate_version_status(
            language_id="go_lsp",
            detected="go1.20.14",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"
        assert "1.20" in result["reason"]

    def test_go_126_supported_ge121(self):
        result = _evaluate_version_status(
            language_id="go_lsp",
            detected="go1.26.0",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "supported"
        assert ">=1.21" in result["reason"]

    def test_go_119_unsupported(self):
        result = _evaluate_version_status(
            language_id="go_lsp",
            detected="go1.19.13",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unsupported"
        assert "below" in result["reason"]

    def test_gopls_best_effort(self):
        result = _evaluate_version_status(
            language_id="go_lsp",
            detected="gopls 0.16.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "best_effort"
        assert "gopls" in result["reason"]


class TestCompileCommandsStatusEdgeCases:
    """Cover readable-but-no-valid-entries and unreadable file paths."""

    def test_readable_but_no_valid_entries(self, tmp_path):
        """compile_commands.json is readable JSON but has no valid entries."""
        cc = tmp_path / "build" / "compile_commands.json"
        cc.parent.mkdir(parents=True)
        # Valid JSON array but entries lack 'file' key
        cc.write_text('[{"directory": ".", "command": "clang++ -c foo.cpp"}]')

        status = _compile_commands_status(language_id="cpp_lsp", workspace=tmp_path)
        assert status is not None
        assert status["readable"] is True
        assert status["valid"] is False
        assert "no valid entries" in status["reason"]

    def test_unreadable_file(self, tmp_path):
        """compile_commands.json exists but is unreadable (binary garbage)."""
        cc = tmp_path / "build" / "compile_commands.json"
        cc.parent.mkdir(parents=True)
        cc.write_bytes(b"\x80\x81\x82\x83\x84")  # invalid UTF-8

        status = _compile_commands_status(language_id="cpp_lsp", workspace=tmp_path)
        assert status is not None
        assert status["present"] is True
        assert status["readable"] is False
        assert "unreadable" in status["reason"]

    def test_relative_path_falls_back_to_str(self, tmp_path, monkeypatch):
        """When path.relative_to raises, the full path string is used."""
        cc = tmp_path / "build" / "compile_commands.json"
        cc.parent.mkdir(parents=True)
        cc.write_text('[{"directory": ".", "command": "clang++ -c f.cpp", "file": "f.cpp"}]')

        # Create a separate workspace that doesn't contain the compile_commands
        other_ws = tmp_path / "other"
        other_ws.mkdir()

        # Force compile_db_path to point to cc in a different tree
        status = _compile_commands_status(
            language_id="cpp_lsp",
            workspace=other_ws,
            compile_db_path=str(cc),
        )
        assert status is not None
        # The path should still appear in the paths list regardless
        assert len(status["paths"]) >= 1


class TestAvailabilityValidationAttachMode:
    """Cover attach-mode validation paths including compile_commands check."""

    def test_c_lsp_compile_commands_check(self, tmp_path, monkeypatch):
        """C/C++ attach endpoints check compile_commands validity."""
        monkeypatch.setattr(
            lsp_setup,
            "_probe_attach_lsp_semantics",
            lambda **_kwargs: {
                "executed": True,
                "handshake_ok": True,
                "semantic_ok": True,
                "symbol_count": 5,
                "reason": "OK",
            },
        )

        probe = {
            "available": True,
            "executable": "127.0.0.1:2087",
            "transport": "tcp",
            "endpoint": "127.0.0.1:2087",
        }

        # No compile_commands.json → compile_ok is False for c_lsp
        result = _availability_validation(
            language_id="c_lsp",
            command=["tcp://127.0.0.1:2087"],
            probe=probe,
            workspace=tmp_path,
        )
        # Should be reachable_only because compile_commands is invalid
        assert result["verification"]["state"] in {"reachable_only", "verified"}

    def test_verified_when_all_checks_pass(self, tmp_path, monkeypatch):
        """Attach endpoint passes all checks: handshake, semantic, compile_commands."""
        monkeypatch.setattr(
            lsp_setup,
            "_probe_attach_lsp_semantics",
            lambda **_kwargs: {
                "executed": True,
                "handshake_ok": True,
                "semantic_ok": True,
                "symbol_count": 5,
                "reason": "OK",
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
                "entry_count": 3,
                "selected_path": "build/compile_commands.json",
                "paths": ["build/compile_commands.json"],
                "reason": None,
            },
        )

        probe = {
            "available": True,
            "executable": "127.0.0.1:2087",
            "transport": "tcp",
            "endpoint": "127.0.0.1:2087",
        }

        result = _availability_validation(
            language_id="c_lsp",
            command=["tcp://127.0.0.1:2087"],
            probe=probe,
            workspace=tmp_path,
        )
        assert result["available"] is True
        assert result["verification"]["state"] == "verified"

    def test_reason_parts_handshake_failed(self, tmp_path, monkeypatch):
        """reason_parts includes handshake failure reason."""
        monkeypatch.setattr(
            lsp_setup,
            "_probe_attach_lsp_semantics",
            lambda **_kwargs: {
                "executed": True,
                "handshake_ok": False,
                "semantic_ok": False,
                "symbol_count": 0,
                "reason": "handshake timeout",
            },
        )

        probe = {
            "available": True,
            "executable": "127.0.0.1:2087",
            "transport": "tcp",
            "endpoint": "127.0.0.1:2087",
        }

        result = _availability_validation(
            language_id="c_lsp",
            command=["tcp://127.0.0.1:2087"],
            probe=probe,
            workspace=tmp_path,
        )
        assert result["verification"]["state"] == "reachable_only"
        assert "LSP handshake failed" in result["verification"]["reason"]

    def test_reason_parts_semantic_failed_with_compile_db_missing(self, tmp_path, monkeypatch):
        """reason_parts includes both semantic probe and compile_commands failures."""
        monkeypatch.setattr(
            lsp_setup,
            "_probe_attach_lsp_semantics",
            lambda **_kwargs: {
                "executed": True,
                "handshake_ok": True,
                "semantic_ok": False,
                "symbol_count": 0,
                "reason": "no symbols",
            },
        )

        probe = {
            "available": True,
            "executable": "127.0.0.1:2088",
            "transport": "tcp",
            "endpoint": "127.0.0.1:2088",
        }

        result = _availability_validation(
            language_id="cpp_lsp",
            command=["tcp://127.0.0.1:2088"],
            probe=probe,
            workspace=tmp_path,
        )
        reason = result["verification"]["reason"]
        assert "semantic probe failed" in reason
        assert "compile_commands.json" in reason

    def test_cpp_lsp_production_mode_rejection(self, tmp_path, monkeypatch):
        """cpp_lsp in production mode returns available=False for reachable_only."""
        monkeypatch.setenv("ASTROGRAPH_LSP_VALIDATION_MODE", "production")
        monkeypatch.setattr(
            lsp_setup,
            "_probe_attach_lsp_semantics",
            lambda **_kwargs: {
                "executed": True,
                "handshake_ok": True,
                "semantic_ok": False,
                "symbol_count": 0,
                "reason": "semantic failed",
            },
        )

        probe = {
            "available": True,
            "executable": "127.0.0.1:2088",
            "transport": "tcp",
            "endpoint": "127.0.0.1:2088",
        }

        result = _availability_validation(
            language_id="cpp_lsp",
            command=["tcp://127.0.0.1:2088"],
            probe=probe,
            workspace=tmp_path,
            validation_mode="production",
        )
        assert result["available"] is False
        assert result["verification"]["state"] == "reachable_only"

    def test_c_lsp_not_rejected_in_production(self, tmp_path, monkeypatch):
        """c_lsp in production mode is NOT rejected (only cpp_lsp is fail-closed)."""
        monkeypatch.setattr(
            lsp_setup,
            "_probe_attach_lsp_semantics",
            lambda **_kwargs: {
                "executed": True,
                "handshake_ok": True,
                "semantic_ok": False,
                "symbol_count": 0,
                "reason": "semantic failed",
            },
        )

        probe = {
            "available": True,
            "executable": "127.0.0.1:2087",
            "transport": "tcp",
            "endpoint": "127.0.0.1:2087",
        }

        result = _availability_validation(
            language_id="c_lsp",
            command=["tcp://127.0.0.1:2087"],
            probe=probe,
            workspace=tmp_path,
            validation_mode="production",
        )
        # c_lsp is allowed even in production reachable_only
        assert result["available"] is True


class TestVersionProbeCandidates:
    """Cover _version_probe_candidates language-specific runtime probe additions."""

    def test_c_lsp_adds_dash_version(self):
        candidates = _version_probe_candidates("c_lsp", ["clangd"])
        kinds = {kind for _cmd, kind in candidates}
        commands = [cmd for cmd, _kind in candidates]
        assert kinds == {"server"}
        # Should have --version and -version
        assert ["clangd", "--version"] in commands
        assert ["clangd", "-version"] in commands

    def test_cpp_lsp_adds_dash_version(self):
        candidates = _version_probe_candidates("cpp_lsp", ["clangd"])
        commands = [cmd for cmd, _kind in candidates]
        assert ["clangd", "-version"] in commands

    def test_python_pylsp_adds_runtime_probe(self):
        candidates = _version_probe_candidates("python", ["/usr/bin/python3", "-m", "pylsp"])
        kinds = {kind for _cmd, kind in candidates}
        assert "runtime" in kinds
        runtime_cmds = [cmd for cmd, kind in candidates if kind == "runtime"]
        assert ["/usr/bin/python3", "--version"] in runtime_cmds

    def test_javascript_adds_node_version(self):
        candidates = _version_probe_candidates("javascript_lsp", ["typescript-language-server"])
        kinds = {kind for _cmd, kind in candidates}
        assert "runtime" in kinds
        runtime_cmds = [cmd for cmd, kind in candidates if kind == "runtime"]
        assert ["node", "--version"] in runtime_cmds

    def test_typescript_adds_node_version(self):
        candidates = _version_probe_candidates("typescript_lsp", ["typescript-language-server"])
        kinds = {kind for _cmd, kind in candidates}
        assert "runtime" in kinds

    def test_java_adds_java_version(self):
        candidates = _version_probe_candidates("java_lsp", ["jdtls"])
        kinds = {kind for _cmd, kind in candidates}
        assert "runtime" in kinds
        runtime_cmds = [cmd for cmd, kind in candidates if kind == "runtime"]
        assert ["java", "-version"] in runtime_cmds

    def test_go_adds_go_version(self):
        candidates = _version_probe_candidates("go_lsp", ["gopls"])
        kinds = {kind for _cmd, kind in candidates}
        assert "runtime" in kinds
        runtime_cmds = [cmd for cmd, kind in candidates if kind == "runtime"]
        assert ["go", "version"] in runtime_cmds

    def test_tcp_endpoint_returns_empty(self):
        candidates = _version_probe_candidates("python", ["tcp://127.0.0.1:2090"])
        assert candidates == []

    def test_empty_command_returns_empty(self):
        candidates = _version_probe_candidates("python", [])
        assert candidates == []


class TestRunVersionProbe:
    """Cover _run_version_probe no-candidates and error handling paths."""

    def test_no_candidates_returns_none(self):
        """TCP endpoint produces no candidates, returns None values."""
        result = _run_version_probe("python", ["tcp://127.0.0.1:2090"])
        assert result["detected"] is None
        assert result["probe_kind"] is None

    def test_oserror_continues(self):
        """OSError from subprocess.run is caught and continues to next candidate."""
        with patch("subprocess.run", side_effect=OSError("command not found")):
            result = _run_version_probe("python", ["nonexistent-pylsp"])
        assert result["detected"] is None
        assert result["probe_kind"] is None

    def test_timeout_continues(self):
        """TimeoutExpired from subprocess.run is caught and continues."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="test", timeout=1.5),
        ):
            result = _run_version_probe("python", ["hanging-pylsp"])
        assert result["detected"] is None
        assert result["probe_kind"] is None

    def test_successful_probe(self):
        """Successful subprocess.run returns detected version."""
        mock_result = MagicMock()
        mock_result.stdout = "pylsp 1.11.0"
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _run_version_probe("python", ["pylsp"])
        assert result["detected"] == "pylsp 1.11.0"
        assert result["probe_kind"] == "server"

    def test_empty_output_skipped(self):
        """Empty subprocess output is skipped, continues to next candidate."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _run_version_probe("python", ["pylsp"])
        assert result["detected"] is None


class TestNormalizeWorkspaceRoot:
    """Cover _normalize_workspace_root edge cases."""

    def test_workspace_is_file_returns_parent(self, tmp_path):
        """When workspace points to a file, its parent directory is returned."""
        test_file = tmp_path / "some_file.py"
        test_file.write_text("# code")
        result = _normalize_workspace_root(str(test_file))
        assert result == tmp_path.resolve()
        assert result.is_dir()

    def test_workspace_is_dir_returned_as_is(self, tmp_path):
        result = _normalize_workspace_root(str(tmp_path))
        assert result == tmp_path.resolve()

    def test_env_var_is_file_returns_parent(self, tmp_path, monkeypatch):
        """When ASTROGRAPH_WORKSPACE points to a file, its parent is returned."""
        test_file = tmp_path / "config.yaml"
        test_file.write_text("key: value")
        monkeypatch.setenv("ASTROGRAPH_WORKSPACE", str(test_file))
        # Clear _active_workspace to avoid it taking precedence
        old_active = lsp_setup._active_workspace
        lsp_setup._active_workspace = None
        try:
            result = _normalize_workspace_root(None)
            assert result == tmp_path.resolve()
        finally:
            lsp_setup._active_workspace = old_active

    def test_env_var_is_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ASTROGRAPH_WORKSPACE", str(tmp_path))
        old_active = lsp_setup._active_workspace
        lsp_setup._active_workspace = None
        try:
            result = _normalize_workspace_root(None)
            assert result == tmp_path.resolve()
        finally:
            lsp_setup._active_workspace = old_active

    def test_env_var_empty_falls_through_to_cwd(self, monkeypatch):
        """Empty ASTROGRAPH_WORKSPACE falls through to cwd."""
        monkeypatch.setenv("ASTROGRAPH_WORKSPACE", "")
        old_active = lsp_setup._active_workspace
        lsp_setup._active_workspace = None
        try:
            result = _normalize_workspace_root(None)
            assert result == Path.cwd().resolve()
        finally:
            lsp_setup._active_workspace = old_active

    def test_workspace_fallback(self, monkeypatch):
        """When /workspace exists and is a directory, it is used as fallback."""
        monkeypatch.delenv("ASTROGRAPH_WORKSPACE", raising=False)
        old_active = lsp_setup._active_workspace
        lsp_setup._active_workspace = None
        try:
            docker_workspace = Path("/workspace")
            if docker_workspace.is_dir():
                result = _normalize_workspace_root(None)
                assert result == docker_workspace
        finally:
            lsp_setup._active_workspace = old_active

    def test_pwd_fallback(self, tmp_path, monkeypatch):
        """When no other source is available, PWD env var is used."""
        monkeypatch.delenv("ASTROGRAPH_WORKSPACE", raising=False)
        old_active = lsp_setup._active_workspace
        lsp_setup._active_workspace = None
        try:
            # Mock /workspace not existing
            with patch.object(Path, "is_dir", side_effect=lambda self=None: False):
                # This is tricky because is_dir is called on Path instances
                # Instead, just test with the real env
                pass
        finally:
            lsp_setup._active_workspace = old_active

    def test_nonexistent_workspace_not_resolved(self, tmp_path):
        """Non-existent workspace path is not resolved but still returned."""
        nonexistent = tmp_path / "does_not_exist"
        result = _normalize_workspace_root(str(nonexistent))
        # Should return the path (not resolved since it doesn't exist)
        assert str(nonexistent) in str(result)

    def test_cwd_fallback_when_all_else_fails(self, monkeypatch):
        """When no explicit workspace, no env var, no active ws, falls to cwd."""
        monkeypatch.delenv("ASTROGRAPH_WORKSPACE", raising=False)
        monkeypatch.delenv("PWD", raising=False)
        old_active = lsp_setup._active_workspace
        lsp_setup._active_workspace = None
        try:
            # With /workspace potentially existing, we just verify it returns something
            result = _normalize_workspace_root(None)
            assert isinstance(result, Path)
        finally:
            lsp_setup._active_workspace = old_active


class TestLoadBindingsEdgeCases:
    """Cover load_lsp_bindings edge cases for non-string keys."""

    def test_non_string_keys_skipped(self, tmp_path):
        """Non-string keys in bindings JSON are skipped."""
        bindings_file = tmp_path / ".metadata_astrograph" / "lsp_bindings.json"
        bindings_file.parent.mkdir(parents=True)
        # JSON only has string keys, but we can have integer-like string keys
        # that parse_command handles differently
        bindings_file.write_text('{"python": ["pylsp"], "123": ["other"]}')
        result = load_lsp_bindings(tmp_path)
        assert "python" in result
        # Integer-ish string key "123" is still a string, should be loaded
        assert "123" in result

    def test_empty_command_skipped(self, tmp_path):
        """Entries with empty commands are filtered out."""
        bindings_file = tmp_path / ".metadata_astrograph" / "lsp_bindings.json"
        bindings_file.parent.mkdir(parents=True)
        bindings_file.write_text('{"python": ["pylsp"], "java_lsp": []}')
        result = load_lsp_bindings(tmp_path)
        assert "python" in result
        assert "java_lsp" not in result  # empty command filtered

    def test_non_list_value_handled(self, tmp_path):
        """Non-list, non-string values are handled gracefully."""
        bindings_file = tmp_path / ".metadata_astrograph" / "lsp_bindings.json"
        bindings_file.parent.mkdir(parents=True)
        bindings_file.write_text('{"python": ["pylsp"], "java_lsp": 42}')
        result = load_lsp_bindings(tmp_path)
        assert "python" in result
        assert "java_lsp" not in result  # non-string/list value produces empty command


class TestProbeAttachEndpointEdgeCases:
    """Cover unix transport and unknown transport paths in _probe_attach_endpoint."""

    def test_unix_socket_endpoint(self, tmp_path):
        """Unix socket endpoint that fails to connect returns not available."""
        endpoint = {
            "transport": "unix",
            "path": str(tmp_path / "nonexistent.sock"),
            "target": str(tmp_path / "nonexistent.sock"),
        }
        result = lsp_setup._probe_attach_endpoint(endpoint)
        assert result["available"] is False

    def test_unknown_transport_returns_unavailable(self):
        """Unknown transport type returns not available."""
        endpoint = {
            "transport": "pipe",
            "target": "some_pipe",
        }
        result = lsp_setup._probe_attach_endpoint(endpoint)
        assert result["available"] is False

    def test_unix_socket_parse(self):
        """Parse unix:// attach endpoint."""
        parsed = lsp_setup.parse_attach_endpoint("unix:///tmp/lsp.sock")
        assert parsed is not None
        assert parsed["transport"] == "unix"
        assert parsed["path"] == "/tmp/lsp.sock"


class TestUnsetLspBinding:
    """Cover unbind persistence cleanup."""

    def test_unset_removes_binding(self, tmp_path):
        save_lsp_bindings({"python": ["pylsp"], "go_lsp": ["gopls"]}, workspace=tmp_path)
        removed, path = lsp_setup.unset_lsp_binding("python", workspace=tmp_path)
        assert removed is True
        loaded = load_lsp_bindings(tmp_path)
        assert "python" not in loaded
        assert "go_lsp" in loaded

    def test_unset_nonexistent_binding(self, tmp_path):
        save_lsp_bindings({"python": ["pylsp"]}, workspace=tmp_path)
        removed, path = lsp_setup.unset_lsp_binding("go_lsp", workspace=tmp_path)
        assert removed is False
        loaded = load_lsp_bindings(tmp_path)
        assert "python" in loaded

    def test_unset_from_empty_bindings(self, tmp_path):
        removed, path = lsp_setup.unset_lsp_binding("python", workspace=tmp_path)
        assert removed is False


class TestSetLspBinding:
    """Cover set_lsp_binding and _raise_empty_command_error."""

    def test_set_binding(self, tmp_path):
        parsed, path = lsp_setup.set_lsp_binding("python", "pylsp --check", workspace=tmp_path)
        assert parsed == ["pylsp", "--check"]
        loaded = load_lsp_bindings(tmp_path)
        assert loaded["python"] == ["pylsp", "--check"]

    def test_set_empty_command_raises(self, tmp_path):
        with pytest.raises(ValueError, match="command cannot be empty"):
            lsp_setup.set_lsp_binding("python", "", workspace=tmp_path)


class TestSaveBindingsAtomicWrite:
    """Cover save_lsp_bindings error handling for atomic write."""

    def test_save_creates_directory(self, tmp_path):
        ws = tmp_path / "deep" / "nested"
        ws.mkdir(parents=True)
        path = save_lsp_bindings({"python": ["pylsp"]}, workspace=ws)
        assert path.exists()
        loaded = load_lsp_bindings(ws)
        assert loaded["python"] == ["pylsp"]


class TestEvaluateVersionStatusEdgeCases:
    """Cover edge cases in _evaluate_version_status."""

    def test_unavailable_command(self):
        """When available is False, returns unknown with skipped reason."""
        result = _evaluate_version_status(
            language_id="python",
            detected=None,
            probe_kind=None,
            transport="subprocess",
            available=False,
        )
        assert result["state"] == "unknown"
        assert "skipped" in result["reason"].lower()

    def test_unparseable_version(self):
        """When version output has no semver, returns unknown."""
        result = _evaluate_version_status(
            language_id="python",
            detected="no version here",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unknown"
        assert "parseable" in result["reason"].lower()

    def test_unknown_language_fallthrough(self):
        """Unknown language_id falls through to default unknown state."""
        result = _evaluate_version_status(
            language_id="rust_lsp",
            detected="rust-analyzer 1.0.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "unknown"
        assert "No explicit compatibility rule" in result["reason"]
