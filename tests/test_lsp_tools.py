"""Tests for the extracted lsp_tools module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from astrograph import lsp_tools


class TestResolveWorkspace:
    def test_with_indexed_dir(self, tmp_path):
        ws = lsp_tools.resolve_lsp_workspace(str(tmp_path), lambda: None)
        assert ws == tmp_path

    def test_with_indexed_file(self, tmp_path):
        f = tmp_path / "file.py"
        f.touch()
        ws = lsp_tools.resolve_lsp_workspace(str(f), lambda: None)
        assert ws == tmp_path

    def test_detect_startup_fallback(self, tmp_path):
        ws = lsp_tools.resolve_lsp_workspace(None, lambda: str(tmp_path))
        assert ws == tmp_path

    def test_cwd_fallback(self):
        ws = lsp_tools.resolve_lsp_workspace(None, lambda: None)
        assert ws == Path.cwd()


class TestStaticHelpers:
    def test_is_docker_runtime_false(self):
        with patch("astrograph.lsp_tools.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            result = lsp_tools.is_docker_runtime()
        assert not result

    def test_default_install_command_python(self):
        cmd = lsp_tools.default_install_command("python")
        assert "python-lsp-server" in cmd[-1]

    def test_default_install_command_js(self):
        cmd = lsp_tools.default_install_command("javascript_lsp")
        assert "typescript-language-server" in cmd[-1]

    def test_default_install_command_unknown(self):
        assert lsp_tools.default_install_command("go_lsp") is None

    def test_dedupe_preserve_order(self):
        assert lsp_tools.dedupe_preserve_order(["a", "b", "a", "c"]) == ["a", "b", "c"]

    def test_dedupe_empty(self):
        assert lsp_tools.dedupe_preserve_order([]) == []


class TestAttachCandidateCommands:
    def test_tcp_endpoint_no_docker(self):
        status = {"default_command": ["tcp://127.0.0.1:2088"]}
        candidates = lsp_tools.attach_candidate_commands(status, docker_runtime=False)
        assert "tcp://127.0.0.1:2088" in candidates
        assert not any("docker" in c for c in candidates)

    def test_tcp_endpoint_with_docker(self):
        status = {"default_command": ["tcp://127.0.0.1:2088"]}
        candidates = lsp_tools.attach_candidate_commands(status, docker_runtime=True)
        assert "tcp://host.docker.internal:2088" in candidates

    def test_unix_endpoint(self):
        status = {"default_command": ["unix:///tmp/lsp.sock"]}
        candidates = lsp_tools.attach_candidate_commands(status, docker_runtime=False)
        assert candidates == ["unix:///tmp/lsp.sock"]

    def test_no_endpoint(self):
        status = {"default_command": ["pylsp"]}
        candidates = lsp_tools.attach_candidate_commands(status, docker_runtime=False)
        assert candidates == []


class TestServerBridgeInfo:
    def test_cpp_bridge(self):
        info = lsp_tools.server_bridge_info("cpp_lsp", "tcp://127.0.0.1:2088")
        assert info is not None
        assert info["server_binary"] == "clangd"
        assert info["shared_with"] == "c_lsp"
        assert "socat" in info["requires"]

    def test_c_bridge(self):
        info = lsp_tools.server_bridge_info("c_lsp", "tcp://127.0.0.1:2087")
        assert info["shared_with"] == "cpp_lsp"

    def test_java_bridge(self):
        info = lsp_tools.server_bridge_info("java_lsp", "tcp://127.0.0.1:2089")
        assert info is not None
        assert info["server_binary"] == "jdtls"

    def test_unknown_language(self):
        assert lsp_tools.server_bridge_info("python", "tcp://127.0.0.1:2090") is None

    def test_non_tcp(self):
        assert lsp_tools.server_bridge_info("cpp_lsp", "unix:///tmp/lsp.sock") is None


class TestBuildRecommendedActions:
    def test_no_missing(self):
        statuses = [{"language": "python", "available": True, "required": True}]
        actions = lsp_tools.build_lsp_recommended_actions(statuses=statuses)
        assert len(actions) == 1
        assert actions[0]["id"] == "verify_lsp_setup"

    def test_missing_required(self):
        statuses = [
            {
                "language": "python",
                "available": False,
                "required": True,
                "transport": "subprocess",
                "effective_command": ["pylsp"],
                "default_command": ["pylsp"],
            }
        ]
        actions = lsp_tools.build_lsp_recommended_actions(statuses=statuses)
        assert any(a["id"] == "auto_bind_missing" for a in actions)
        assert any(a["priority"] == "high" for a in actions)

    def test_scoped_language(self):
        statuses = [{"language": "python", "available": True, "required": True}]
        actions = lsp_tools.build_lsp_recommended_actions(
            statuses=statuses, scope_language="python"
        )
        assert actions[0]["arguments"].get("language") == "python"


class TestInjectLspSetupGuidance:
    def test_inject_adds_fields(self, tmp_path):
        payload = {
            "servers": [{"language": "python", "available": True, "required": True}],
        }
        lsp_tools.inject_lsp_setup_guidance(payload, workspace=tmp_path, docker_runtime=False)
        assert "missing_languages" in payload
        assert "recommended_actions" in payload
        assert "resolution_loop" in payload
        assert payload["execution_context"] == "host"

    def test_inject_docker_context(self, tmp_path):
        payload = {
            "servers": [{"language": "python", "available": False, "required": True}],
        }
        lsp_tools.inject_lsp_setup_guidance(payload, workspace=tmp_path, docker_runtime=True)
        assert payload["execution_context"] == "docker"
        assert "observation_note" in payload


class TestLspSetupResultJson:
    def test_produces_valid_json(self):
        payload = {"ok": True, "mode": "inspect"}
        result = lsp_tools.lsp_setup_result_json(payload)
        parsed = json.loads(result)
        assert parsed["ok"] is True


class TestHandleLspSetup:
    def test_invalid_mode(self, tmp_path):
        result = lsp_tools.handle_lsp_setup(workspace=tmp_path, mode="bogus")
        parsed = json.loads(result)
        assert parsed["ok"] is False
        assert "Invalid mode" in parsed["error"]

    def test_bind_requires_language(self, tmp_path):
        result = lsp_tools.handle_lsp_setup(workspace=tmp_path, mode="bind")
        parsed = json.loads(result)
        assert parsed["ok"] is False
        assert "'language' is required" in parsed["error"]

    def test_unbind_requires_language(self, tmp_path):
        result = lsp_tools.handle_lsp_setup(workspace=tmp_path, mode="unbind")
        parsed = json.loads(result)
        assert parsed["ok"] is False

    def test_bind_empty_command(self, tmp_path):
        result = lsp_tools.handle_lsp_setup(
            workspace=tmp_path, mode="bind", language="python", command=""
        )
        parsed = json.loads(result)
        assert parsed["ok"] is False
        assert "non-empty" in parsed["error"]

    def test_unsupported_language(self, tmp_path):
        result = lsp_tools.handle_lsp_setup(
            workspace=tmp_path, mode="inspect", language="nonexistent_xyz"
        )
        parsed = json.loads(result)
        assert parsed["ok"] is False
        assert "Unsupported" in parsed["error"]
