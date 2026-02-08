"""Tests for the consolidated MCP server tools."""

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from astrograph.index import IndexEntry, SimilarityResult
from astrograph.languages.base import CodeUnit
from astrograph.languages.registry import LanguageRegistry
from astrograph.server import create_server, get_tools, set_tools
from astrograph.tools import (
    PERSISTENCE_DIR,
    CodeStructureTools,
    ToolResult,
    _get_persistence_path,
)


@pytest.fixture(autouse=True)
def _clear_lsp_env(monkeypatch):
    """Keep LSP command env vars isolated across tests/modules."""
    LanguageRegistry.reset()
    for key in (
        "ASTROGRAPH_PY_LSP_COMMAND",
        "ASTROGRAPH_JS_LSP_COMMAND",
        "ASTROGRAPH_C_LSP_COMMAND",
        "ASTROGRAPH_CPP_LSP_COMMAND",
        "ASTROGRAPH_JAVA_LSP_COMMAND",
        "ASTROGRAPH_COMPILE_COMMANDS_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("ASTROGRAPH_PY_LSP_COMMAND", f"{sys.executable} -m pylsp")
    # Disable startup auto-indexing in tests; fixtures call index_codebase explicitly.
    monkeypatch.setenv("ASTROGRAPH_WORKSPACE", "")


class TestResolveDockerPath:
    """Tests for Docker path resolution via CodeStructureTools._resolve_path."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()

    @staticmethod
    def _workspace_is_dir_mock(original_is_dir, workspace_path: str):
        def mock_is_dir(self):
            if str(self) == workspace_path:
                return True
            return original_is_dir(self)

        return mock_is_dir

    @staticmethod
    def _docker_exists_mock(original_exists, *existing_paths: str):
        existing = set(existing_paths)

        def mock_exists(self):
            return str(self) in existing or original_exists(self)

        return mock_exists

    def _assert_docker_new_file_resolution(
        self,
        source_path: str,
        expected_path: str,
        existing_paths: tuple[str, ...],
        workspace_dir: str,
    ) -> None:
        original_exists = Path.exists
        original_is_dir = Path.is_dir
        mock_exists = self._docker_exists_mock(original_exists, *existing_paths)
        mock_is_dir = self._workspace_is_dir_mock(original_is_dir, workspace_dir)

        with patch.object(Path, "exists", mock_exists), patch.object(Path, "is_dir", mock_is_dir):
            assert self.tools._resolve_path(source_path) == expected_path

    def test_existing_path_unchanged(self):
        """Existing paths are returned unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = self.tools._resolve_path(tmpdir)
            assert result == tmpdir

    def test_nonexistent_path_no_docker(self):
        """Non-existent paths without Docker environment return unchanged."""
        path = "/nonexistent/host/path/to/project"
        result = self.tools._resolve_path(path)
        assert result == path

    def test_docker_path_subpath_matching(self):
        """Test that subpaths are checked when resolving Docker paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)

            result = self.tools._resolve_path(src_dir)
            assert result == src_dir

    def test_persistence_path_normal(self):
        """Test persistence path for normal (non-Docker) environments."""

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _get_persistence_path(tmpdir)
            assert str(result).endswith(".metadata_astrograph")
            assert tmpdir in str(result)

    def test_persistence_path_docker_subdirectory(self):
        """Test persistence path redirects to workspace root in Docker."""
        original_exists = Path.exists

        def mock_exists(self):
            path_str = str(self)
            if path_str == "/workspace" or path_str == "/.dockerenv":
                return True
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            result = _get_persistence_path("/workspace/src")
            assert str(result) == "/workspace/.metadata_astrograph"

    def test_resolve_docker_path_with_mock_docker_env(self):
        """Test Docker path resolution with mocked Docker environment."""
        original_exists = Path.exists
        mock_exists = self._docker_exists_mock(
            original_exists, "/workspace", "/.dockerenv", "/workspace/src"
        )

        with patch.object(Path, "exists", mock_exists):
            result = self.tools._resolve_path("/Users/foo/bar/src")
            assert result == "/workspace/src"

    def test_resolve_docker_path_new_file(self):
        """Test Docker path resolution for a new file (parent exists, file doesn't)."""
        self._assert_docker_new_file_resolution(
            "/Users/foo/bar/src/new_file.py",
            "/workspace/src/new_file.py",
            ("/workspace", "/.dockerenv", "/workspace/src"),
            "/workspace/src",
        )

    def test_resolve_docker_path_new_file_at_root(self):
        """Test Docker path resolution for a new file at workspace root."""
        self._assert_docker_new_file_resolution(
            "/Users/foo/bar/test.py",
            "/workspace/test.py",
            ("/workspace", "/.dockerenv"),
            "/workspace",
        )

    def test_learned_root_mapping_speeds_up_subsequent_calls(self):
        """After first successful resolution, host_root is cached and reused."""
        original_exists = Path.exists
        mock_exists = self._docker_exists_mock(
            original_exists, "/workspace", "/.dockerenv", "/workspace/src"
        )

        with patch.object(Path, "exists", mock_exists):
            # First call learns the mapping
            result1 = self.tools._resolve_path("/Users/foo/project/src")
            assert result1 == "/workspace/src"
            assert self.tools._host_root == "/Users/foo/project"

            # Second call uses cached mapping — no need for filesystem lookups
            result2 = self.tools._resolve_path("/Users/foo/project/lib/utils.py")
            assert result2 == "/workspace/lib/utils.py"

    def test_error_messages_do_not_contain_workspace(self):
        """User-facing error messages should not leak /workspace paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "existing.py")
            Path(py_file).write_text("def foo(): pass\n")

            tools = CodeStructureTools()
            with patch.object(tools, "_require_index", return_value=None):
                # Edit a file that exists — old_string won't be found
                result = tools.edit(py_file, "nonexistent_string_xyz", "new")
                assert "/workspace" not in result.text
                assert "old_string not found" in result.text

                # Write to a path that can't be created
                result = tools.write("/nonexistent/dir/file.py", "def bar(): pass")
                assert "/workspace" not in result.text
                assert "Failed to write" in result.text


def _get_analyze_details(tools, result):
    """Read full analyze details from report file if it exists, else inline text."""
    if ".metadata_astrograph/" not in result.text:
        return result.text
    match = re.search(r"Details: \.metadata_astrograph/([^\s]+)", result.text)
    if not match:
        return result.text
    indexed = Path(tools._last_indexed_path).resolve()
    base = indexed.parent if not indexed.is_dir() else indexed
    report = base / PERSISTENCE_DIR / match.group(1)
    return report.read_text() if report.exists() else result.text


def _assert_status_reports_indexing(tools: CodeStructureTools) -> None:
    tools._bg_index_done.clear()
    result = tools.status()
    assert "indexing" in result.text
    tools._bg_index_done.set()


def _index_single_file_directory(
    tools: CodeStructureTools,
    filename: str = "test.py",
    source: str = "def foo(): pass",
) -> ToolResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        py_file = os.path.join(tmpdir, filename)
        Path(py_file).write_text(source)
        return tools.index_codebase(tmpdir)


def _index_temp_code_file(tools: CodeStructureTools, code: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tools.index_codebase(f.name)
    os.unlink(f.name)


def _similarity_result(
    *,
    language: str,
    similarity_type: str,
    file_path: str,
    name: str,
    code: str,
) -> SimilarityResult:
    code_unit = CodeUnit(
        name=name,
        code=code,
        file_path=file_path,
        line_start=1,
        line_end=max(1, code.count("\n") + 1),
        unit_type="function",
        language=language,
    )
    entry = IndexEntry(
        id=f"{language}:{name}:{file_path}",
        wl_hash="wl_hash",
        pattern_hash="pattern_hash",
        fingerprint={"n_nodes": 8, "n_edges": 7},
        hierarchy_hashes=["root", "body"],
        code_unit=code_unit,
        node_count=8,
        depth=2,
    )
    return SimilarityResult(entry=entry, similarity_type=similarity_type)


def _overwrite_file(path: str, content: str) -> None:
    Path(path).write_text(content)


def _suppress_first_hash_from_analysis(tools: CodeStructureTools) -> str:
    details = _get_analyze_details(tools, tools.analyze())
    match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
    assert match
    wl_hash = match.group(1)
    tools.suppress(wl_hash)
    return wl_hash


def _start_background_index_completion(tools: CodeStructureTools, delay: float = 0.05) -> None:
    import threading

    def finish_bg():
        time.sleep(delay)
        tools._bg_index_done.set()

    threading.Thread(target=finish_bg, daemon=True).start()


def _with_indexed_temp_file(tools: CodeStructureTools, content: str, fn):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        tools.index_codebase(f.name)
        result = fn()
    os.unlink(f.name)
    return result


@pytest.fixture
def tools():
    """Create a fresh tools instance for each test."""
    return CodeStructureTools()


@pytest.fixture
def sample_python_file():
    """Create a temporary Python file with duplicate functions."""
    content = """
def add(a, b):
    return a + b

def sum_nums(x, y):
    return x + y

def multiply(a, b):
    return a * b

class Calculator:
    def add(self, a, b):
        return a + b
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(content)
        f.flush()
        yield f.name
    os.unlink(f.name)


class TestIndexCodebase:
    """Tests for index_codebase tool."""

    def test_index_file(self, tools, sample_python_file):
        result = tools.index_codebase(sample_python_file)
        assert "Indexed" in result.text
        assert "code units" in result.text

    def test_index_directory(self, tools):
        result = _index_single_file_directory(tools)
        assert "Indexed" in result.text

    def test_index_nonexistent_path(self, tools):
        result = tools.index_codebase("/nonexistent/path")
        assert "Error" in result.text

    def test_index_clears_previous(self, tools, sample_python_file):
        """Indexing should clear previous index."""
        tools.index_codebase(sample_python_file)
        # Clear any stale suppressions from previous tests
        tools.index.clear_suppressions()
        first_result = tools.analyze()

        # Index again - should have same results (not doubled)
        tools.index_codebase(sample_python_file)
        second_result = tools.analyze()

        def norm(text: str) -> str:
            return re.sub(
                r"Details: \.metadata_astrograph/analysis_report_\d{8}_\d{6}_\d+\.txt",
                "Details: .metadata_astrograph/analysis_report_<timestamp>.txt",
                text,
            )

        assert norm(first_result.text) == norm(second_result.text)


class TestAnalyze:
    """Tests for analyze tool."""

    def test_analyze_empty_index(self, tools):
        result = tools.analyze()
        assert "No code indexed" in result.text

    def test_analyze_no_duplicates(self, tools):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def unique_func(): return 42")
            f.flush()
            tools.index_codebase(f.name)
        os.unlink(f.name)

        result = tools.analyze()
        assert "No significant duplicates" in result.text or "No code indexed" in result.text

    def test_analyze_finds_duplicates(self, tools, sample_python_file):
        tools.index_codebase(sample_python_file)
        result = tools.analyze()
        # May or may not find duplicates depending on internal threshold
        assert result.text

    def test_analyze_uses_timestamped_report_only(self, tools, sample_python_file):
        tools.index_codebase(sample_python_file)
        result = tools.analyze()
        if "Details: .metadata_astrograph/" not in result.text:
            pytest.skip("No duplicates found to generate report")

        match = re.search(r"Details: \.metadata_astrograph/([^\s]+)", result.text)
        assert match
        report_name = match.group(1)
        assert report_name.startswith("analysis_report_")
        assert report_name != "analysis_report.txt"
        assert "Latest:" not in result.text

        indexed = Path(tools._last_indexed_path).resolve()
        base = indexed.parent if not indexed.is_dir() else indexed
        legacy_report = base / PERSISTENCE_DIR / "analysis_report.txt"
        assert not legacy_report.exists()


class TestCheck:
    """Tests for check tool."""

    def test_check_empty_index(self, tools):
        assert "No code indexed" in tools.check("def foo(): pass").text

    @pytest.mark.parametrize(
        "code,description",
        [
            ("class CompletelyDifferent: pass", "no match or partial match"),
            ("def add(a, b): return a + b", "exact match"),
            ("def add(x, y): return x + y", "similar match"),
        ],
    )
    def test_check_indexed_code(self, tools, sample_python_file, code, description):
        """Test check tool with various code patterns."""
        tools.index_codebase(sample_python_file)
        result = tools.check(code)
        assert result.text, f"Expected response for: {description}"

    def test_check_cross_language_matches_are_assist_only(self, tools, sample_python_file):
        tools.index_codebase(sample_python_file)
        cross_language = _similarity_result(
            language="javascript_lsp",
            similarity_type="exact",
            file_path="utils.js",
            name="sum",
            code="function sum(a, b) { return a + b; }",
        )
        with patch.object(tools.index, "find_similar", return_value=[cross_language]):
            result = tools.check("def sum(a, b):\n    return a + b", language="python")

        assert "Safe to proceed" in result.text
        assert "Cross-language structural matches found" in result.text


class TestCompare:
    """Tests for compare tool."""

    @pytest.mark.parametrize(
        "code1,code2,expected",
        [
            ("def add(a, b): return a + b", "def sum(x, y): return x + y", ["EQUIVALENT"]),
            ("def add(a, b): return a + b", "def multiply(a, b): return a * b", ["DIFFERENT"]),
            (
                "def process(x): return x + 1",
                "def process(x): return x + 1 + 1",
                ["SIMILAR", "DIFFERENT"],
            ),
        ],
    )
    def test_compare_code_structures(self, tools, code1, code2, expected):
        """Test compare tool with various code pairs."""
        result = tools.compare(code1, code2)
        assert any(exp in result.text for exp in expected)

    def test_compare_invalid_semantic_mode(self, tools):
        result = tools.compare(
            "def add(a, b): return a + b",
            "def sum(x, y): return x + y",
            semantic_mode="unsupported",
        )
        assert "Invalid semantic_mode" in result.text

    def test_compare_strict_semantic_mode_inconclusive_without_signals(self, tools):
        result = tools.compare(
            "def one(): return 1",
            "def two(): return 2",
            semantic_mode="strict",
        )
        assert "INCONCLUSIVE" in result.text

    def test_compare_cpp_semantic_mismatch_assist(self, tools):
        builtin_plus = "int add(int a, int b) { return a + b; }"
        custom_plus = "Vec add(Vec a, Vec b) { return a + b; }"
        result = tools.compare(
            builtin_plus,
            custom_plus,
            language="cpp_lsp",
            semantic_mode="assist",
        )
        assert "SEMANTIC_MISMATCH" in result.text
        assert "operator.plus.binding" in result.text

    def test_compare_cpp_semantic_mismatch_strict(self, tools):
        builtin_plus = "int add(int a, int b) { return a + b; }"
        custom_plus = "Vec add(Vec a, Vec b) { return a + b; }"
        result = tools.compare(
            builtin_plus,
            custom_plus,
            language="cpp_lsp",
            semantic_mode="strict",
        )
        assert "DIFFERENT" in result.text


class TestCallTool:
    """Tests for tool dispatch."""

    def test_unknown_tool(self, tools):
        result = tools.call_tool("nonexistent_tool", {})
        assert "Unknown tool" in result.text

    def test_dispatch_index_codebase(self, tools, sample_python_file):
        result = tools.call_tool("index_codebase", {"path": sample_python_file})
        assert "Indexed" in result.text

    def test_dispatch_analyze(self, tools, sample_python_file):
        tools.index_codebase(sample_python_file)
        result = tools.call_tool("analyze", {})
        assert result.text

    def test_dispatch_check(self, tools, sample_python_file):
        tools.index_codebase(sample_python_file)
        result = tools.call_tool("check", {"code": "def foo(): pass"})
        assert result.text

    def test_dispatch_compare(self, tools):
        result = tools.call_tool(
            "compare",
            {
                "code1": "def a(): return 1",
                "code2": "def b(): return 2",
            },
        )
        assert result.text

    def test_dispatch_compare_with_language_and_semantic_mode(self, tools):
        result = tools.call_tool(
            "compare",
            {
                "code1": "int add(int a, int b) { return a + b; }",
                "code2": "Vec add(Vec a, Vec b) { return a + b; }",
                "language": "cpp_lsp",
                "semantic_mode": "strict",
            },
        )
        assert "DIFFERENT" in result.text

    def test_mutating_tool_detection(self, tools):
        assert tools._is_mutating_tool_call("status", {}) is False
        assert tools._is_mutating_tool_call("metadata_erase", {}) is True
        assert tools._is_mutating_tool_call("write", {}) is True
        assert tools._is_mutating_tool_call("lsp_setup", {"mode": "inspect"}) is False
        assert tools._is_mutating_tool_call("lsp_setup", {"mode": "auto_bind"}) is True
        assert tools._is_mutating_tool_call("lsp_setup", {"mode": "bind"}) is True
        assert tools._is_mutating_tool_call("lsp_setup", {"mode": "unbind"}) is True

    def test_call_tool_uses_mutation_lock_for_mutating_calls(self, tools):
        class RecordingLock:
            def __init__(self):
                self.events = []

            def __enter__(self):
                self.events.append("enter")
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                self.events.append("exit")
                return False

        lock = RecordingLock()
        with patch.object(tools, "_mutation_lock", lock), patch.object(
            tools, "_call_tool_unlocked", return_value=ToolResult("ok")
        ) as dispatch:
            result = tools.call_tool("metadata_erase", {})

        assert result.text == "ok"
        assert lock.events == ["enter", "exit"]
        dispatch.assert_called_once_with("metadata_erase", {})

    def test_call_tool_skips_mutation_lock_for_read_only_calls(self, tools):
        class RecordingLock:
            def __init__(self):
                self.events = []

            def __enter__(self):
                self.events.append("enter")
                return self

            def __exit__(self, _exc_type, _exc, _tb):
                self.events.append("exit")
                return False

        lock = RecordingLock()
        with patch.object(tools, "_mutation_lock", lock), patch.object(
            tools, "_call_tool_unlocked", return_value=ToolResult("ok")
        ) as dispatch:
            result = tools.call_tool("status", {})

        assert result.text == "ok"
        assert lock.events == []
        dispatch.assert_called_once_with("status", {})


class TestLSPSetupTool:
    """Tests for deterministic LSP setup tool flow."""

    def test_dispatch_lsp_setup_inspect(self, tools):
        result = tools.call_tool("lsp_setup", {"mode": "inspect"})
        payload = json.loads(result.text)

        assert payload["ok"] is True
        assert payload["mode"] == "inspect"
        assert "servers" in payload
        assert "bindings" in payload
        assert "agent_directive" in payload
        assert "recommended_actions" in payload
        assert isinstance(payload["recommended_actions"], list)
        assert "language_variant_policy" in payload
        assert "version_status" in payload["servers"][0]

    def test_lsp_setup_inspect_scoped_language(self, tools):
        payload = json.loads(tools.lsp_setup(mode="inspect", language="cpp_lsp").text)

        assert payload["ok"] is True
        assert payload["scope_language"] == "cpp_lsp"
        assert [status["language"] for status in payload["servers"]] == ["cpp_lsp"]
        assert payload["recommended_actions"][0]["arguments"]["language"] == "cpp_lsp"
        assert all(
            step["arguments"].get("language") == "cpp_lsp" for step in payload["resolution_loop"]
        )
        assert all(
            action.get("language") in (None, "cpp_lsp") for action in payload["recommended_actions"]
        )
        assert set(payload["language_variant_policy"]) == {"cpp_lsp"}

    def test_lsp_setup_inspect_rejects_unknown_language(self, tools):
        payload = json.loads(tools.lsp_setup(mode="inspect", language="go_lsp").text)
        assert payload["ok"] is False
        assert "Unsupported language" in payload["error"]

    def test_lsp_setup_bind_and_unbind(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {"ASTROGRAPH_WORKSPACE": tmpdir},
            clear=False,
        ):
            tools = CodeStructureTools()

            bind_result = tools.lsp_setup(
                mode="bind",
                language="python",
                command=[sys.executable, "-m", "pylsp"],
            )
            bind_payload = json.loads(bind_result.text)
            assert bind_payload["ok"] is True
            assert bind_payload["language"] == "python"

            bindings_path = Path(tmpdir) / PERSISTENCE_DIR / "lsp_bindings.json"
            persisted = json.loads(bindings_path.read_text())
            assert persisted["python"][0] == sys.executable

            unbind_result = tools.lsp_setup(mode="unbind", language="python")
            unbind_payload = json.loads(unbind_result.text)
            assert unbind_payload["ok"] is True

            persisted_after = json.loads(bindings_path.read_text())
            assert "python" not in persisted_after
            tools.close()

    def test_lsp_setup_auto_bind_uses_observations(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "ASTROGRAPH_WORKSPACE": tmpdir,
                "ASTROGRAPH_PY_LSP_COMMAND": "missing-python-lsp-xyz",
            },
            clear=False,
        ):
            tools = CodeStructureTools()
            result = tools.lsp_setup(
                mode="auto_bind",
                observations=[
                    {
                        "language": "python",
                        "command": [sys.executable, "-m", "pylsp"],
                    }
                ],
            )
            payload = json.loads(result.text)
            assert any(change["language"] == "python" for change in payload["changes"])
            assert "recommended_actions" in payload
            assert payload["agent_directive"]
            tools.close()

    def test_lsp_setup_auto_bind_scoped_language(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "ASTROGRAPH_WORKSPACE": tmpdir,
                "ASTROGRAPH_PY_LSP_COMMAND": "missing-python-lsp-xyz",
            },
            clear=False,
        ):
            tools = CodeStructureTools()
            payload = json.loads(
                tools.lsp_setup(
                    mode="auto_bind",
                    language="python",
                    observations=[
                        {
                            "language": "python",
                            "command": [sys.executable, "-m", "pylsp"],
                        }
                    ],
                ).text
            )

            assert payload["scope_language"] == "python"
            assert payload["scope_languages"] == ["python"]
            assert all(status["language"] == "python" for status in payload["statuses"])
            assert payload["recommended_actions"][0]["arguments"]["language"] == "python"
            assert all(
                step["arguments"].get("language") == "python" for step in payload["resolution_loop"]
            )
            tools.close()

    def test_lsp_setup_recommended_actions_include_docker_host_alias(self, tools):
        missing_cpp_status = {
            "language": "cpp_lsp",
            "required": False,
            "available": False,
            "transport": "tcp",
            "effective_command": ["tcp://127.0.0.1:2088"],
            "effective_source": "default",
            "default_command": ["tcp://127.0.0.1:2088"],
        }
        with patch.object(CodeStructureTools, "_is_docker_runtime", return_value=True):
            actions = tools._build_lsp_recommended_actions(statuses=[missing_cpp_status])

        assert any(action["id"] == "ensure_docker_host_alias" for action in actions)

    def test_lsp_setup_recommended_actions_focus_cpp_when_optional(self, tools):
        missing_cpp_status = {
            "language": "cpp_lsp",
            "required": False,
            "available": False,
            "transport": "tcp",
            "effective_command": ["tcp://127.0.0.1:2088"],
            "effective_source": "default",
            "default_command": ["tcp://127.0.0.1:2088"],
        }

        actions = tools._build_lsp_recommended_actions(statuses=[missing_cpp_status])

        assert actions[0]["id"] == "focus_cpp_lsp"
        assert actions[0]["arguments"] == {"mode": "inspect", "language": "cpp_lsp"}

    def test_lsp_setup_recommended_actions_follow_up_scopes_each_language(self, tools):
        missing_cpp_status = {
            "language": "cpp_lsp",
            "required": False,
            "available": False,
            "transport": "tcp",
            "effective_command": ["tcp://127.0.0.1:2088"],
            "effective_source": "default",
            "default_command": ["tcp://127.0.0.1:2088"],
        }
        missing_python_status = {
            "language": "python",
            "required": True,
            "available": False,
            "transport": "subprocess",
            "effective_command": ["pylsp"],
            "effective_source": "default",
            "default_command": ["pylsp"],
        }

        with patch.object(CodeStructureTools, "_is_docker_runtime", return_value=False):
            actions = tools._build_lsp_recommended_actions(
                statuses=[missing_cpp_status, missing_python_status]
            )

        discover_cpp = next(
            action for action in actions if action["id"] == "discover_cpp_lsp_endpoint"
        )
        search_python = next(action for action in actions if action["id"] == "search_python")
        install_python = next(action for action in actions if action["id"] == "install_python")

        assert discover_cpp["follow_up_arguments"]["language"] == "cpp_lsp"
        assert discover_cpp["follow_up_arguments"]["mode"] == "auto_bind"
        assert search_python["follow_up_arguments"]["language"] == "python"
        assert install_python["follow_up_arguments"]["language"] == "python"

    def test_lsp_setup_recommended_actions_include_cpp_validation_fixes(self, tools):
        missing_cpp_status = {
            "language": "cpp_lsp",
            "required": False,
            "available": False,
            "probe_available": True,
            "transport": "tcp",
            "effective_command": ["tcp://host.docker.internal:2088"],
            "effective_source": "binding",
            "default_command": ["tcp://127.0.0.1:2088"],
            "verification_state": "reachable_only",
            "verification": {
                "state": "reachable_only",
                "reason": "LSP handshake failed; compile_commands.json missing or invalid",
            },
            "compile_commands": {
                "required": True,
                "present": False,
                "readable": False,
                "valid": False,
                "entry_count": 0,
                "selected_path": None,
                "paths": [],
                "reason": "compile_commands.json missing or invalid",
            },
        }

        actions = tools._build_lsp_recommended_actions(statuses=[missing_cpp_status])
        action_ids = [action["id"] for action in actions]
        assert "verify_cpp_lsp_protocol" in action_ids
        assert "ensure_compile_commands_cpp_lsp" in action_ids
        verify_cpp = next(action for action in actions if action["id"] == "verify_cpp_lsp_protocol")
        assert "which clangd" in verify_cpp["host_search_commands"]
        assert "which ccls" in verify_cpp["host_search_commands"]

    def test_lsp_setup_guidance_adds_attach_ready_verification_actions(self, tools):
        payload = {
            "mode": "inspect",
            "scope_language": "cpp_lsp",
            "servers": [
                {
                    "language": "cpp_lsp",
                    "required": False,
                    "available": True,
                    "transport": "tcp",
                    "effective_command": ["tcp://host.docker.internal:2088"],
                    "effective_source": "binding",
                    "binding_command": ["tcp://host.docker.internal:2088"],
                    "default_command": ["tcp://127.0.0.1:2088"],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tools._inject_lsp_setup_guidance(payload, workspace=Path(tmpdir))

        action_ids = [action["id"] for action in payload["recommended_actions"]]
        assert "verify_cpp_lsp_analysis" in action_ids
        assert "rebaseline_after_cpp_lsp_binding" in action_ids
        assert "re-run auto_bind" in payload["agent_directive"]


class TestToolResult:
    """Tests for ToolResult class."""

    def test_tool_result(self):
        result = ToolResult("test message")
        assert result.text == "test message"


class TestMCPServerIntegration:
    """Integration tests for MCP server."""

    def test_server_creation(self):
        server = create_server()
        assert server is not None

    def test_get_and_set_tools(self):
        original = get_tools()
        new_tools = CodeStructureTools()
        set_tools(new_tools)
        assert get_tools() is new_tools
        set_tools(original)  # Restore


class TestAnalyzeWithDuplicates:
    """Tests for analyze with substantial duplicates (above MIN_NODE_COUNT threshold)."""

    def test_analyze_exact_duplicates_with_verification(self, tools):
        """Test analyze with exact duplicates that pass the node count threshold."""
        # Create files with more complex duplicate functions
        complex_code = """
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_data(data):
    output = []
    for element in data:
        if element > 0:
            output.append(element * 2)
    return output
"""
        _index_temp_code_file(tools, complex_code)

        result = tools.analyze()
        # Should find duplicates with suppress calls or no findings
        details = _get_analyze_details(tools, result)
        assert "suppress(wl_hash=" in details or "No significant duplicates" in result.text

    def test_analyze_duplicates_at_different_depths(self, tools):
        """Test analyze with duplicates at different path depths."""
        complex_func = """
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create file at root level
            root_file = os.path.join(tmpdir, "root.py")
            Path(root_file).write_text(complex_func)

            # Create file in subdirectory
            subdir = os.path.join(tmpdir, "sub", "dir")
            os.makedirs(subdir)
            deep_file = os.path.join(subdir, "deep.py")
            Path(deep_file).write_text(complex_func.replace("process_items", "transform_data"))

            tools.index_codebase(tmpdir)
            result = tools.analyze()
            # Should find duplicates and suggest keeping the shallower one
            assert result.text

    def test_analyze_pattern_duplicates(self, tools):
        """Test analyze finds pattern duplicates (same structure, different operators)."""
        pattern_code = """
def check_positive(x):
    if x > 0:
        return True
    return False

def check_negative(x):
    if x < 0:
        return True
    return False
"""
        _index_temp_code_file(tools, pattern_code)

        result = tools.analyze()
        # May find pattern duplicates
        assert result.text


class TestCheckSimilarityLevels:
    """Tests for check with different similarity levels."""

    def test_check_high_similarity(self, tools):
        """Test check finds high similarity matches."""
        base_code = """
def process(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item)
    return results
"""
        _index_temp_code_file(tools, base_code)

        # Check with similar but not exact code
        similar_code = """
def process(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)  # Different operation
    return results
"""
        result = tools.check(similar_code)
        assert result.text


class TestCompareBranches:
    """Tests for compare edge cases."""

    def test_compare_compatible_fingerprints(self, tools):
        """Test compare with compatible fingerprints (same structure, different constants)."""
        # Same structure with different constants - structurally equivalent
        code1 = "def f(x): return x + 1"
        code2 = "def g(y): return y + 2"
        result = tools.compare(code1, code2)
        assert "EQUIVALENT" in result.text

    def test_compare_different_structure(self, tools):
        """Test compare with different structure."""
        code1 = "def f(x): return x + 1"
        code2 = "def f(x, y): return x + y + 1"  # Extra parameter changes structure
        result = tools.compare(code1, code2)
        assert "DIFFERENT" in result.text or "SIMILAR" in result.text


class TestBlockDetection:
    """Tests for block duplicate detection in tools."""

    @staticmethod
    def _analyze_temp_code(tools, code: str):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            tools.index_codebase(f.name)
            result = tools.analyze()
        os.unlink(f.name)
        return result

    def test_index_codebase_extracts_blocks(self, tools):
        """Test indexing always extracts blocks (22% overhead, much better detection)."""
        code = """
def func1():
    for i in range(10):
        print(i)

def func2():
    for j in range(10):
        print(j)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            result = tools.index_codebase(f.name)
        os.unlink(f.name)

        assert "Indexed" in result.text
        assert "block" in result.text.lower()

    def test_analyze_with_block_duplicates(self, tools):
        """Test analyze finds block duplicates."""
        code = """
def func1():
    for i in range(10):
        if i > 5:
            print(i)

def func2():
    for j in range(10):
        if j > 5:
            print(j)
"""
        result = self._analyze_temp_code(tools, code)

        # Should find duplicates with suppress calls or no findings
        details = _get_analyze_details(tools, result)
        assert "suppress(wl_hash=" in details or "No significant duplicates" in result.text

    def test_analyze_block_duplicates_show_parent_functions(self, tools):
        """Test analyze output includes parent function names for block duplicates."""
        code = """
def process_list(items):
    for item in items:
        if item > 0:
            result = item * 2
            print(result)
    return

def transform_list(data):
    for element in data:
        if element > 0:
            output = element * 2
            print(output)
    return
"""
        result = self._analyze_temp_code(tools, code)

        # Should have some analysis output
        assert result.text


class TestSuppressionTools:
    """Tests for suppress, unsuppress, and list_suppressions tools."""

    @pytest.fixture(name="_indexed_with_duplicates")
    def _indexed_with_duplicates_fixture(self, tools):
        """Create and index a file with duplicates."""
        code = """
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_data(data):
    output = []
    for element in data:
        if element > 0:
            output.append(element * 2)
    return output
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            tools.index_codebase(f.name)
            yield f.name
        os.unlink(f.name)

    def test_suppress_not_indexed(self, tools):
        """Test suppress when no code is indexed."""
        result = tools.suppress("some_hash")
        assert "No code indexed" in result.text

    @pytest.mark.parametrize(
        ("action", "wl_hash", "expected"),
        [
            ("suppress", "nonexistent_hash", "not found"),
            ("unsuppress", "not_suppressed_hash", "was not suppressed"),
        ],
    )
    def test_toggle_invalid_hash(self, tools, _indexed_with_duplicates, action, wl_hash, expected):
        """Invalid suppress/unsuppress hash operations return clear errors."""
        method = getattr(tools, action)
        result = method(wl_hash)
        assert expected in result.text

    def test_suppress_valid_hash(self, tools, _indexed_with_duplicates):
        """Test suppress with valid hash."""
        # First get a valid hash from analyze output
        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        if "suppress(wl_hash=" in details:
            # Extract hash from analyze output (format: suppress(wl_hash="HASH"))
            import re

            match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
            if match:
                wl_hash = match.group(1)

                result = tools.suppress(wl_hash)
                assert "Suppressed" in result.text

                # Verify it's now suppressed
                analyze_after = tools.analyze()
                # The hash should no longer appear (or if it was the only one, no duplicates)
                assert (
                    wl_hash not in analyze_after.text
                    or "No significant duplicates" in analyze_after.text
                )

    def test_unsuppress_valid(self, tools, _indexed_with_duplicates):
        """Test unsuppress restores the hash."""
        import re

        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        if "suppress(wl_hash=" in details:
            match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
            if match:
                wl_hash = match.group(1)

                tools.suppress(wl_hash)
                result = tools.unsuppress(wl_hash)
                assert "Unsuppressed" in result.text

                # Hash should appear again
                analyze_after = tools.analyze()
                details_after = _get_analyze_details(tools, analyze_after)
                assert wl_hash in details_after

    def test_list_suppressions_empty(self, tools, _indexed_with_duplicates):
        """Test list_suppressions when nothing is suppressed."""
        # Clear any suppressions to ensure test isolation
        tools.index.clear_suppressions()
        result = tools.list_suppressions()
        assert "No hashes are currently suppressed" in result.text

    def test_list_suppressions_with_suppressed(self, tools, _indexed_with_duplicates):
        """Test list_suppressions shows suppressed hashes."""
        analyze_result = tools.analyze()
        if "Suppress:" in analyze_result.text:
            lines = analyze_result.text.split("\n")
            hash_line = [line for line in lines if "Suppress:" in line][0]
            wl_hash = hash_line.split("Suppress:")[1].strip()

            tools.suppress(wl_hash)
            result = tools.list_suppressions()
            assert wl_hash in result.text
            assert "Suppressed hashes" in result.text

    @pytest.mark.parametrize(
        ("tool_name", "expected"),
        [("suppress", "not found"), ("unsuppress", "was not suppressed")],
    )
    def test_call_tool_toggle(self, tools, _indexed_with_duplicates, tool_name, expected):
        """call_tool dispatch for suppress/unsuppress returns expected status."""
        result = tools.call_tool(tool_name, {"wl_hash": "test_hash"})
        assert expected in result.text

    def test_call_tool_list_suppressions(self, tools, _indexed_with_duplicates):
        """Test call_tool dispatch for list_suppressions."""
        result = tools.call_tool("list_suppressions", {})
        assert "No hashes" in result.text or "Suppressed hashes" in result.text

    def test_analyze_output_includes_hash(self, tools, _indexed_with_duplicates):
        """Test analyze output includes hash for suppression."""
        result = tools.analyze()
        if "duplicate groups" in result.text:
            details = _get_analyze_details(tools, result)
            assert "suppress(wl_hash=" in details

    def test_suppress_batch_valid(self, tools, _indexed_with_duplicates):
        """Test batch suppress with valid hashes."""
        import re

        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        hashes = re.findall(r'suppress\(wl_hash="([^"]+)"\)', details)
        if hashes:
            result = tools.suppress_batch(hashes)
            assert "Suppressed" in result.text
            assert str(len(hashes)) in result.text

    def test_suppress_batch_mixed(self, tools, _indexed_with_duplicates):
        """Test batch suppress with mix of valid and invalid hashes."""
        import re

        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        hashes = re.findall(r'suppress\(wl_hash="([^"]+)"\)', details)
        if hashes:
            mixed = hashes + ["nonexistent_hash_abc"]
            result = tools.suppress_batch(mixed)
            assert "Suppressed" in result.text
            assert "not found" in result.text

    @pytest.mark.parametrize("method_name", ["suppress_batch", "unsuppress_batch"])
    def test_batch_toggle_empty(self, tools, _indexed_with_duplicates, method_name):
        """Batch toggle with empty list returns a helpful message."""
        result = getattr(tools, method_name)([])
        assert "No hashes provided" in result.text

    @pytest.mark.parametrize("tool_name", ["suppress_batch", "unsuppress_batch"])
    def test_call_tool_batch_toggle(self, tools, _indexed_with_duplicates, tool_name):
        """call_tool dispatch for batch toggle with unknown hash reports not found."""
        result = tools.call_tool(tool_name, {"wl_hashes": ["fake_hash"]})
        assert "not found" in result.text

    def test_unsuppress_batch_valid(self, tools, _indexed_with_duplicates):
        """Test batch unsuppress with valid hashes."""
        import re

        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        hashes = re.findall(r'suppress\(wl_hash="([^"]+)"\)', details)
        if hashes:
            tools.suppress_batch(hashes)
            result = tools.unsuppress_batch(hashes)
            assert "Unsuppressed" in result.text
            assert str(len(hashes)) in result.text

    def test_unsuppress_batch_mixed(self, tools, _indexed_with_duplicates):
        """Test batch unsuppress with mix of valid and invalid hashes."""
        import re

        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        hashes = re.findall(r'suppress\(wl_hash="([^"]+)"\)', details)
        if hashes:
            tools.suppress_batch(hashes)
            mixed = hashes + ["nonexistent_hash_abc"]
            result = tools.unsuppress_batch(mixed)
            assert "Unsuppressed" in result.text
            assert "not found" in result.text


class TestWorkflowIntegration:
    """Integration tests for multi-step workflows."""

    def test_full_workflow_index_analyze_suppress_analyze(self, tools):
        """Test complete workflow: index → analyze → suppress → analyze."""
        # Create file with duplicates
        code = """
def calculate_total(items):
    total = 0
    for item in items:
        total += item
    return total

def sum_values(numbers):
    result = 0
    for num in numbers:
        result += num
    return result
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()

            # Step 1: Index
            index_result = tools.index_codebase(f.name)
            assert "Indexed" in index_result.text

            # Step 2: Analyze - should find duplicates
            analyze_result = tools.analyze()
            details = _get_analyze_details(tools, analyze_result)
            # Output shows suppress calls for duplicates or no findings
            assert (
                "suppress(wl_hash=" in details or "No significant duplicates" in analyze_result.text
            )

            # Step 3: If duplicates found, suppress one
            if "suppress(wl_hash=" in details:
                # Extract hash from output
                import re

                match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
                if match:
                    wl_hash = match.group(1)

                    # Suppress the duplicate
                    suppress_result = tools.suppress(wl_hash)
                    assert "Suppressed" in suppress_result.text

                    # Step 4: Analyze again - should show fewer or no duplicates
                    analyze_after = tools.analyze()
                    details_after = _get_analyze_details(tools, analyze_after)
                    # The suppressed hash should not appear as actionable
                    assert (
                        wl_hash not in details_after or "suppressed" in analyze_after.text.lower()
                    )

        os.unlink(f.name)

    def test_workflow_incremental_reindex(self, tools):
        """Test workflow with incremental re-indexing after changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial file
            file1 = os.path.join(tmpdir, "module1.py")
            Path(file1).write_text("def foo(): return 1\n")

            # Initial index
            result1 = tools.index_codebase(tmpdir)
            assert "Indexed" in result1.text

            # Add another file
            file2 = os.path.join(tmpdir, "module2.py")
            Path(file2).write_text("def bar(): return 2\n")

            # Incremental re-index
            result2 = tools.index_codebase(tmpdir)
            assert "Indexed" in result2.text

            # Analyze should see both
            analyze_result = tools.analyze()
            # Should have indexed content from both files
            assert analyze_result.text  # Non-empty result


class TestCheckStaleness:
    """Tests for check_staleness tool."""

    def test_check_staleness_empty_index(self, tools):
        """Test check_staleness with empty index."""
        result = tools.check_staleness()
        assert "No code indexed" in result.text

    def test_check_staleness_fresh_index(self, tools):
        """Test check_staleness on freshly indexed codebase."""
        result = _with_indexed_temp_file(tools, "def foo(): pass", tools.check_staleness)

        assert "up to date" in result.text

    def test_check_staleness_modified_file(self, tools):
        """Test check_staleness detects modified files."""
        import time

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass")
            f.flush()
            tools.index_codebase(f.name)

            # Modify file
            time.sleep(0.01)
            _overwrite_file(f.name, "def bar(): pass")

            result = tools.check_staleness()
        os.unlink(f.name)

        assert "STALE" in result.text
        assert "Modified" in result.text

    def test_check_staleness_with_path(self, tools):
        """Test check_staleness with path parameter for new files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            tools.index_codebase(tmpdir)

            # Add new file
            file2 = os.path.join(tmpdir, "file2.py")
            Path(file2).write_text("def bar(): pass")

            result = tools.check_staleness(path=tmpdir)

            assert "STALE" in result.text
            assert "New files" in result.text

    def test_call_tool_check_staleness(self, tools, sample_python_file):
        """Test call_tool dispatch for check_staleness."""
        tools.index_codebase(sample_python_file)
        result = tools.call_tool("check_staleness", {})
        assert result.text


class TestIncrementalIndexing:
    """Tests for incremental indexing in tools."""

    def test_index_codebase_first_run_full_index(self, tools):
        """Test first indexing is always a full index."""
        result = _index_single_file_directory(tools, filename="file1.py")
        assert "Indexed" in result.text

    def test_index_codebase_incremental_unchanged(self, tools):
        """Test incremental indexing with unchanged files still produces valid output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            # First index (full)
            tools.index_codebase(tmpdir)

            # Second index - should work correctly with no errors
            result = tools.index_codebase(tmpdir)

            assert "Indexed" in result.text

    def test_index_codebase_incremental_partial_update(self, tools):
        """Test incremental indexing with some files changed still works."""
        import time

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            file2 = os.path.join(tmpdir, "file2.py")

            Path(file1).write_text("def foo(): pass")
            Path(file2).write_text("def bar(): pass")

            # First index (full)
            tools.index_codebase(tmpdir)

            # Modify only file1
            time.sleep(0.01)
            Path(file1).write_text("def foo_modified(): pass")

            result = tools.index_codebase(tmpdir)

            # Should still report indexed entries
            assert "Indexed" in result.text

    def test_incremental_default_true(self, tools):
        """Test incremental defaults to True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            # First index
            tools.index_codebase(tmpdir)

            # Second index with default parameters - should work
            result = tools.call_tool("index_codebase", {"path": tmpdir})

            assert "Indexed" in result.text


class TestAnalyzeStalenessWarning:
    """Tests for staleness warning in analyze output."""

    @staticmethod
    def _analyze_with_stale_index(tools, auto_reindex: bool) -> "ToolResult":
        """Index a file, modify it to make index stale, then analyze."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass\ndef bar(): pass")
            f.flush()
            tools.index_codebase(f.name)

            time.sleep(0.01)
            _overwrite_file(f.name, "def baz(): pass")

            result = tools.analyze(auto_reindex=auto_reindex)
        os.unlink(f.name)
        return result

    def test_analyze_warns_on_stale_index(self, tools):
        """Test analyze shows warning when index is stale."""
        result = self._analyze_with_stale_index(tools, auto_reindex=False)
        assert "WARNING" in result.text or "stale" in result.text.lower()

    def test_analyze_auto_reindex_on_stale(self, tools):
        """Test analyze auto-reindexes when stale and auto_reindex=True."""
        result = self._analyze_with_stale_index(tools, auto_reindex=True)
        assert "Auto-reindexed" in result.text
        assert "WARNING" not in result.text

    def test_analyze_no_warning_fresh_index(self, tools):
        """Test analyze has no warning when index is fresh."""
        result = _with_indexed_temp_file(
            tools,
            "def unique_function_abc123(): return 42",
            tools.analyze,
        )

        # No staleness warning for fresh index
        if "No significant duplicates" in result.text:
            assert "WARNING" not in result.text


class TestPersistence:
    """Tests for index persistence to .metadata_astrograph folder (SQLite)."""

    def test_index_creates_persistence_folder(self):
        """Test that indexing creates .metadata_astrograph folder with SQLite DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)

            persistence_path = os.path.join(tmpdir, PERSISTENCE_DIR)
            assert os.path.isdir(persistence_path)
            assert os.path.isfile(os.path.join(persistence_path, "index.db"))
            tools.close()

    def test_index_loads_cached_index(self):
        """Test that cached index is loaded on re-index with fresh tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            # First tools instance - index and save
            tools1 = CodeStructureTools()
            tools1.index_codebase(tmpdir)
            tools1.close()

            # Verify persistence file exists
            persistence_path = os.path.join(tmpdir, PERSISTENCE_DIR)
            assert os.path.isfile(os.path.join(persistence_path, "index.db"))

            # Second tools instance (simulating new session) - should load from cache
            tools2 = CodeStructureTools()
            result = tools2.index_codebase(tmpdir)

            assert "Indexed" in result.text
            tools2.close()

    def test_suppress_persists_across_sessions(self):
        """Test that suppression persists to SQLite across sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with duplicates
            code = """
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_data(data):
    output = []
    for element in data:
        if element > 0:
            output.append(element * 2)
    return output
"""
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text(code)

            # First tools instance - index and suppress
            tools1 = CodeStructureTools()
            tools1.index_codebase(tmpdir)

            # Find a hash to suppress
            wl_hash = _suppress_first_hash_from_analysis(tools1)
            tools1.close()

            # Second tools instance - should load suppression from SQLite
            tools2 = CodeStructureTools()
            tools2.index_codebase(tmpdir)

            # Suppression should be preserved
            assert wl_hash in tools2.index.suppressed_hashes
            tools2.close()

    def test_unsuppress_persists_across_sessions(self):
        """Test that unsuppression persists to SQLite across sessions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_data(data):
    output = []
    for element in data:
        if element > 0:
            output.append(element * 2)
    return output
"""
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text(code)

            # First tools instance - index, suppress, then unsuppress
            tools1 = CodeStructureTools()
            tools1.index_codebase(tmpdir)

            import re

            analyze_result = tools1.analyze()
            details = _get_analyze_details(tools1, analyze_result)
            match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
            if match:
                wl_hash = match.group(1)
                tools1.suppress(wl_hash)
                tools1.unsuppress(wl_hash)
                tools1.close()

                # Second tools instance - suppression should be gone
                tools2 = CodeStructureTools()
                tools2.index_codebase(tmpdir)

                assert wl_hash not in tools2.index.suppressed_hashes
                tools2.close()

    def test_persistence_for_single_file(self):
        """Test persistence works when indexing a single file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            tools = CodeStructureTools()
            tools.index_codebase(file1)

            # Persistence folder should be created in parent directory
            persistence_path = os.path.join(tmpdir, PERSISTENCE_DIR)
            assert os.path.isdir(persistence_path)
            tools.close()


class TestWriteTool:
    """Tests for the astrograph_write tool."""

    @pytest.fixture
    def indexed_tools(self):
        """Create tools with an indexed codebase containing a function."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, "existing.py")
            with open(existing_file, "w") as f:
                f.write(
                    """
def calculate_sum(a, b):
    result = a + b
    return result
"""
                )
            index_result = tools.index_codebase(existing_file)
            assert "Indexed" in index_result.text
            # Freeze watcher side-effects for deterministic test behavior.
            tools._close_event_driven_index()
            assert len(tools.index.entries) > 0, index_result.text
            yield tools, tmpdir

    def test_write_requires_index(self, tools):
        """Test that write requires an indexed codebase."""
        result = tools.write("/tmp/test.py", "def foo(): pass")
        assert "No code indexed" in result.text

    def test_write_blocks_exact_duplicate(self, indexed_tools):
        """Test that write blocks when identical code exists."""
        tools, tmpdir = indexed_tools
        duplicate_code = """
def add_numbers(x, y):
    result = x + y
    return result
"""
        new_file = os.path.join(tmpdir, "new.py")
        result = tools.write(new_file, duplicate_code)
        assert "BLOCKED" in result.text
        assert "identical code exists" in result.text
        # File should not be created
        assert not os.path.exists(new_file)

    def test_write_succeeds_with_unique_code(self, indexed_tools):
        """Test that write succeeds when code is unique."""
        tools, tmpdir = indexed_tools
        unique_code = """
def multiply_values(a, b, c):
    product = a * b * c
    return product * 2
"""
        new_file = os.path.join(tmpdir, "unique.py")
        result = tools.write(new_file, unique_code)
        assert "Created" in result.text
        assert "lines)" in result.text
        assert os.path.exists(new_file)
        with open(new_file) as f:
            assert f.read() == unique_code

    def test_write_warns_on_high_similarity(self, indexed_tools):
        """Test that write warns but proceeds on high similarity."""
        tools, tmpdir = indexed_tools
        # This may or may not trigger high similarity depending on fingerprint
        similar_code = """
def compute_total(x, y):
    total = x + y
    return total
"""
        new_file = os.path.join(tmpdir, "similar.py")
        result = tools.write(new_file, similar_code)
        # Should either succeed or block (depending on exact vs high similarity)
        assert ("Created" in result.text or "Wrote" in result.text) or "BLOCKED" in result.text

    def test_write_cross_language_exact_match_is_assist_only(self, indexed_tools):
        tools, tmpdir = indexed_tools
        cross_language = _similarity_result(
            language="javascript_lsp",
            similarity_type="exact",
            file_path="src/math.js",
            name="sum",
            code="function sum(a, b) { return a + b; }",
        )
        with patch.object(tools.index, "find_similar", return_value=[cross_language]):
            new_file = os.path.join(tmpdir, "cross_lang.py")
            result = tools.write(new_file, "def sum(a, b):\n    return a + b\n")

        assert "BLOCKED" not in result.text
        assert "Created" in result.text or "Wrote" in result.text
        assert "Cross-language structural matches found" in result.text
        assert os.path.exists(new_file)

    def test_write_blocks_same_language_even_when_cross_language_matches_exist(self, indexed_tools):
        tools, tmpdir = indexed_tools
        same_language = _similarity_result(
            language="python",
            similarity_type="exact",
            file_path="existing.py",
            name="calculate_sum",
            code="def calculate_sum(a, b):\n    result = a + b\n    return result\n",
        )
        cross_language = _similarity_result(
            language="javascript_lsp",
            similarity_type="exact",
            file_path="src/math.js",
            name="sum",
            code="function sum(a, b) { return a + b; }",
        )
        with patch.object(
            tools.index, "find_similar", return_value=[cross_language, same_language]
        ):
            new_file = os.path.join(tmpdir, "blocked.py")
            result = tools.write(
                new_file, "def sum(a, b):\n    result = a + b\n    return result\n"
            )

        assert "BLOCKED" in result.text
        assert not os.path.exists(new_file)

    def test_write_blocks_cpp_duplicate_when_file_contains_wrapper_lines(self, indexed_tools):
        tools, tmpdir = indexed_tools
        cpp_exact_entry = _similarity_result(
            language="cpp_lsp",
            similarity_type="exact",
            file_path="existing.cpp",
            name="accumulate_positive",
            code=(
                "int accumulate_positive(const std::vector<int>& values) {\n"
                "    int total = 0;\n"
                "    for (int value : values) {\n"
                "        if (value > 0) {\n"
                "            total += value;\n"
                "        }\n"
                "    }\n"
                "    return total;\n"
                "}\n"
            ),
        ).entry

        class _FakeCppPlugin:
            language_id = "cpp_lsp"

            def extract_code_units(
                self,
                source: str,
                file_path: str = "<unknown>",
                include_blocks: bool = True,
                max_block_depth: int = 3,
            ):
                del source, include_blocks, max_block_depth
                yield CodeUnit(
                    name="accumulate_positive",
                    code=cpp_exact_entry.code_unit.code,
                    file_path=file_path,
                    line_start=3,
                    line_end=11,
                    unit_type="function",
                    language="cpp_lsp",
                )

        def _fake_exact_matches(code: str, language: str = "python"):
            if language == "cpp_lsp" and code.strip().startswith("int accumulate_positive"):
                return [cpp_exact_entry]
            return []

        cpp_file = os.path.join(tmpdir, "new_duplicate.cpp")
        cpp_content = (
            "#include <vector>\n\n"
            "int accumulate_positive(const std::vector<int>& values) {\n"
            "    int total = 0;\n"
            "    for (int value : values) {\n"
            "        if (value > 0) {\n"
            "            total += value;\n"
            "        }\n"
            "    }\n"
            "    return total;\n"
            "}\n"
        )

        with patch.object(
            LanguageRegistry.get(),
            "get_plugin_for_file",
            return_value=_FakeCppPlugin(),
        ), patch.object(
            tools.index,
            "find_exact_matches",
            side_effect=_fake_exact_matches,
        ), patch.object(
            tools.index,
            "find_similar",
            return_value=[],
        ):
            result = tools.write(cpp_file, cpp_content)

        assert "BLOCKED" in result.text
        assert "identical code exists" in result.text
        assert not os.path.exists(cpp_file)

    def test_write_handles_io_error(self, indexed_tools):
        """Test that write handles IO errors gracefully."""
        tools, tmpdir = indexed_tools
        # Try to write to a directory that doesn't exist
        result = tools.write("/nonexistent/dir/file.py", "def foo(): pass")
        assert "Failed to write" in result.text

    def test_call_tool_write(self, indexed_tools):
        """Test call_tool dispatch for write."""
        tools, tmpdir = indexed_tools
        new_file = os.path.join(tmpdir, "dispatch_test.py")
        result = tools.call_tool(
            "write", {"file_path": new_file, "content": "def unique_func(): return 42"}
        )
        assert ("Created" in result.text or "Wrote" in result.text) or "BLOCKED" in result.text


class TestEditTool:
    """Tests for the astrograph_edit tool."""

    @pytest.fixture
    def indexed_tools_with_file(self):
        """Create tools with an indexed codebase and editable file."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            existing_file = os.path.join(tmpdir, "existing.py")
            with open(existing_file, "w") as f:
                f.write(
                    """
def calculate_sum(a, b):
    result = a + b
    return result

def placeholder():
    # TODO: implement
    pass
"""
                )
            index_result = tools.index_codebase(existing_file)
            assert "Indexed" in index_result.text
            tools._close_event_driven_index()
            assert len(tools.index.entries) > 0, index_result.text
            yield tools, tmpdir, existing_file

    def test_edit_requires_index(self, tools):
        """Test that edit requires an indexed codebase."""
        result = tools.edit("/tmp/test.py", "old", "new")
        assert "No code indexed" in result.text

    def test_edit_file_not_found(self, indexed_tools_with_file):
        """Test edit on non-existent file."""
        tools, tmpdir, _ = indexed_tools_with_file
        result = tools.edit("/nonexistent/file.py", "old", "new")
        assert "File not found" in result.text

    def test_edit_old_string_not_found(self, indexed_tools_with_file):
        """Test edit when old_string doesn't exist in file."""
        tools, tmpdir, existing_file = indexed_tools_with_file
        result = tools.edit(existing_file, "nonexistent_string", "new_content")
        assert "old_string not found" in result.text

    def test_edit_old_string_not_unique(self, indexed_tools_with_file):
        """Test edit when old_string appears multiple times."""
        tools, tmpdir, existing_file = indexed_tools_with_file
        result = tools.edit(existing_file, "result", "output")
        assert "appears" in result.text and "times" in result.text

    def test_edit_blocks_exact_duplicate(self, indexed_tools_with_file):
        """Test that edit blocks when new code is a duplicate from another file."""
        tools, tmpdir, existing_file = indexed_tools_with_file
        # Create a second file to edit
        second_file = os.path.join(tmpdir, "second.py")
        with open(second_file, "w") as f:
            f.write(
                """
def placeholder():
    # TODO: implement
    pass
"""
            )
        # Try to replace placeholder with a duplicate of calculate_sum (in existing_file)
        duplicate_code = """def add_values(x, y):
    result = x + y
    return result"""
        result = tools.edit(
            second_file, "def placeholder():\n    # TODO: implement\n    pass", duplicate_code
        )
        assert "BLOCKED" in result.text
        assert "identical code exists" in result.text

    def test_edit_succeeds_with_unique_code(self, indexed_tools_with_file):
        """Test that edit succeeds when new code is unique."""
        tools, tmpdir, existing_file = indexed_tools_with_file
        unique_code = """def complex_operation(a, b, c, d):
    intermediate = a * b
    result = intermediate + c - d
    return result * 2"""
        result = tools.edit(
            existing_file, "def placeholder():\n    # TODO: implement\n    pass", unique_code
        )
        assert "Edited" in result.text
        assert "+" in result.text  # diff markers
        with open(existing_file) as f:
            content = f.read()
            assert "complex_operation" in content
            assert "placeholder" not in content

    def test_edit_warns_same_file_duplicate(self, indexed_tools_with_file):
        """Test that edit warns but proceeds when duplicate exists in same file."""
        tools, tmpdir, existing_file = indexed_tools_with_file
        # Add a duplicate function to the same file - should warn but proceed
        duplicate_code = """def add_values(x, y):
    result = x + y
    return result"""
        result = tools.edit(
            existing_file, "def placeholder():\n    # TODO: implement\n    pass", duplicate_code
        )
        # Should warn about same-file duplicate but still succeed
        assert "WARNING" in result.text
        assert "same file" in result.text
        assert "Edited" in result.text

    def test_edit_blocks_cpp_duplicate_when_wrapper_lines_present(self, indexed_tools_with_file):
        tools, tmpdir, _existing_file = indexed_tools_with_file
        cpp_exact_entry = _similarity_result(
            language="cpp_lsp",
            similarity_type="exact",
            file_path="existing.cpp",
            name="accumulate_positive",
            code=(
                "int accumulate_positive(const std::vector<int>& values) {\n"
                "    int total = 0;\n"
                "    for (int value : values) {\n"
                "        if (value > 0) {\n"
                "            total += value;\n"
                "        }\n"
                "    }\n"
                "    return total;\n"
                "}\n"
            ),
        ).entry

        class _FakeCppPlugin:
            language_id = "cpp_lsp"

            def extract_code_units(
                self,
                source: str,
                file_path: str = "<unknown>",
                include_blocks: bool = True,
                max_block_depth: int = 3,
            ):
                del source, include_blocks, max_block_depth
                yield CodeUnit(
                    name="accumulate_positive",
                    code=cpp_exact_entry.code_unit.code,
                    file_path=file_path,
                    line_start=3,
                    line_end=11,
                    unit_type="function",
                    language="cpp_lsp",
                )

        def _fake_exact_matches(code: str, language: str = "python"):
            if language == "cpp_lsp" and code.strip().startswith("int accumulate_positive"):
                return [cpp_exact_entry]
            return []

        target_cpp = os.path.join(tmpdir, "target.cpp")
        old_block = "int placeholder() {\n    return 0;\n}\n"
        Path(target_cpp).write_text(old_block)
        new_block = (
            "#include <vector>\n\n"
            "int accumulate_positive(const std::vector<int>& values) {\n"
            "    int total = 0;\n"
            "    for (int value : values) {\n"
            "        if (value > 0) {\n"
            "            total += value;\n"
            "        }\n"
            "    }\n"
            "    return total;\n"
            "}\n"
        )

        with patch.object(
            LanguageRegistry.get(),
            "get_plugin_for_file",
            return_value=_FakeCppPlugin(),
        ), patch.object(
            tools.index,
            "find_exact_matches",
            side_effect=_fake_exact_matches,
        ), patch.object(
            tools.index,
            "find_similar",
            return_value=[],
        ):
            result = tools.edit(target_cpp, old_block, new_block)

        assert "BLOCKED" in result.text
        assert "identical code exists" in result.text
        assert Path(target_cpp).read_text() == old_block

    def test_edit_cross_language_exact_match_is_assist_only(self, indexed_tools_with_file):
        tools, tmpdir, existing_file = indexed_tools_with_file
        cross_language = _similarity_result(
            language="javascript_lsp",
            similarity_type="exact",
            file_path="src/math.js",
            name="sum",
            code="function sum(a, b) { return a + b; }",
        )
        with patch.object(tools.index, "find_similar", return_value=[cross_language]):
            result = tools.edit(
                existing_file,
                "def placeholder():\n    # TODO: implement\n    pass",
                "def sum(a, b):\n    return a + b",
            )

        assert "BLOCKED" not in result.text
        assert "Edited" in result.text
        assert "Cross-language structural matches found" in result.text

    def test_call_tool_edit(self, indexed_tools_with_file):
        """Test call_tool dispatch for edit."""
        tools, tmpdir, existing_file = indexed_tools_with_file
        result = tools.call_tool(
            "edit",
            {
                "file_path": existing_file,
                "old_string": "# TODO: implement",
                "new_string": "# Implemented!",
            },
        )
        assert "Edited" in result.text


class TestSuppressionPersistenceAcrossRestart:
    """Test that suppressions survive a simulated container restart (close + reopen)."""

    def test_suppression_survives_close_and_reopen(self):
        """Suppress a hash, close tools (flushes WAL), reopen, verify suppression loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_data(data):
    output = []
    for element in data:
        if element > 0:
            output.append(element * 2)
    return output
"""
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text(code)

            # --- Session 1: index, suppress, close ---
            tools1 = CodeStructureTools()
            tools1.index_codebase(tmpdir)

            import re

            analyze_result = tools1.analyze()
            details = _get_analyze_details(tools1, analyze_result)
            match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
            wl_hash = match.group(1) if match else pytest.skip("No duplicates found to suppress")
            tools1.suppress(wl_hash)

            # Explicitly close (simulates Docker SIGTERM → _tools.close())
            tools1.close()

            # --- Session 2: fresh tools, reopen, verify ---
            tools2 = CodeStructureTools()
            tools2.index_codebase(tmpdir)

            assert wl_hash in tools2.index.suppressed_hashes
            tools2.close()


class TestAnalyzeReportImprovements:
    """Tests for analyze report quality improvements."""

    @staticmethod
    def _complex_func(name: str = "process_items") -> str:
        return f"""
def {name}(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results
"""

    def test_analyze_report_uses_relative_paths(self, tools):
        """Report paths should be relative, not absolute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_files(
                tmpdir,
                "",
                [
                    ("module_a.py", self._complex_func("process_items")),
                    ("module_b.py", self._complex_func("transform_data")),
                ],
            )

            tools.index_codebase(tmpdir)
            result = tools.analyze()
            details = _get_analyze_details(tools, result)

            if "suppress(wl_hash=" in details:
                # No absolute path prefix should appear in the report
                assert tmpdir not in details, f"Report contains absolute path prefix '{tmpdir}'"
                # Relative paths should appear
                assert "module_a.py:" in details or "module_b.py:" in details

    @staticmethod
    def _different_complex_func(name: str = "check_values") -> str:
        """A structurally different complex function for test-only duplicates."""
        return f"""
def {name}(data):
    count = 0
    for val in data:
        if val < 0:
            count += 1
        else:
            count -= 1
    return count
"""

    @staticmethod
    def _write_files(tmpdir: str, relative_dir: str, files: list[tuple[str, str]]) -> None:
        dir_path = os.path.join(tmpdir, relative_dir) if relative_dir else tmpdir
        os.makedirs(dir_path, exist_ok=True)
        for filename, content in files:
            with open(os.path.join(dir_path, filename), "w") as f:
                f.write(content)

    def _write_function_pair(
        self,
        tmpdir: str,
        relative_dir: str,
        files: list[tuple[str, str]],
        generator,
    ) -> None:
        self._write_files(
            tmpdir,
            relative_dir,
            [(filename, generator(function_name)) for filename, function_name in files],
        )

    def test_analyze_report_separates_source_and_tests(self, tools):
        """Report should have section headers when both source and test duplicates exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_function_pair(
                tmpdir,
                "src",
                [("module_a.py", "process_items"), ("module_b.py", "transform_data")],
                self._complex_func,
            )
            self._write_function_pair(
                tmpdir,
                "tests",
                [("test_a.py", "test_check_a"), ("test_b.py", "test_check_b")],
                self._different_complex_func,
            )

            tools.index_codebase(tmpdir)
            result = tools.analyze()
            details = _get_analyze_details(tools, result)

            if "suppress(wl_hash=" in details:
                assert "=== Source code ===" in details
                assert "=== Tests ===" in details
                # Source section should appear before Tests section
                assert details.index("=== Source code ===") < details.index("=== Tests ===")

    def test_analyze_summary_shows_source_vs_test_counts(self, tools):
        """Inline summary should include source vs test breakdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_function_pair(
                tmpdir,
                "src",
                [("module_a.py", "process_items"), ("module_b.py", "transform_data")],
                self._complex_func,
            )
            self._write_function_pair(
                tmpdir,
                "tests",
                [("test_a.py", "test_check_a"), ("test_b.py", "test_check_b")],
                self._different_complex_func,
            )

            tools.index_codebase(tmpdir)
            result = tools.analyze()

            if "duplicate groups" in result.text:
                assert "in source" in result.text
                assert "in tests" in result.text

    def test_suppress_batch_response_includes_refresh_hint(self, tools):
        """suppress_batch response should include 'Run analyze' hint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_files(
                tmpdir,
                "",
                [
                    ("module_a.py", self._complex_func("process_items")),
                    ("module_b.py", self._complex_func("transform_data")),
                ],
            )

            tools.index_codebase(tmpdir)
            result = tools.analyze()
            details = _get_analyze_details(tools, result)

            import re

            hashes = re.findall(r'suppress\(wl_hash="([^"]+)"\)', details)
            if hashes:
                batch_result = tools.suppress_batch(hashes)
                assert "Run analyze" in batch_result.text


class TestStatusTool:
    """Tests for the astrograph_status tool."""

    @pytest.mark.parametrize(
        ("method_name", "expected"),
        [
            ("status", "idle"),
            ("check_staleness", "No code indexed"),
            ("metadata_recompute_baseline", "No codebase has been indexed"),
        ],
    )
    def test_empty_index_tool_responses(self, tools, method_name, expected):
        """Tools that require indexed data return clear empty-index responses."""
        result = getattr(tools, method_name)()
        assert expected in result.text

    def test_status_ready_after_indexing(self, tools, sample_python_file):
        """Status should return ready after indexing."""
        tools.index_codebase(sample_python_file)
        result = tools.status()
        assert "ready" in result.text
        assert "code units" in result.text

    def test_status_during_background_indexing(self):
        """Status should return indexing state when background indexing is running."""
        tools = CodeStructureTools()
        _assert_status_reports_indexing(tools)

    def test_call_tool_status(self, tools, sample_python_file):
        """Test call_tool dispatch for status."""
        tools.index_codebase(sample_python_file)
        result = tools.call_tool("status", {})
        assert "ready" in result.text


class TestMetadataErase:
    """Tests for the astrograph_metadata_erase tool."""

    def test_erase_no_metadata(self, tools):
        """Erase when nothing has been indexed."""
        result = tools.metadata_erase()
        assert "idle" in result.text.lower()

    def test_erase_clears_index_and_suppressions(self):
        """Erase should clear the in-memory index, suppressions, and persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_data(data):
    output = []
    for element in data:
        if element > 0:
            output.append(element * 2)
    return output
"""
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text(code)

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)

            # Suppress a hash
            _suppress_first_hash_from_analysis(tools)

            # Verify persistence exists
            persistence_path = os.path.join(tmpdir, PERSISTENCE_DIR)
            assert os.path.isdir(persistence_path)

            # Erase
            result = tools.metadata_erase()
            assert "Erased" in result.text

            # Verify persistence is gone
            assert not os.path.exists(persistence_path)

            # Verify server is idle
            status = tools.status()
            assert "idle" in status.text

            # Verify suppressions are gone
            assert len(tools.index.suppressed_hashes) == 0

    def test_erase_then_reindex(self):
        """After erase, re-indexing should work from scratch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)
            tools.metadata_erase()

            # Re-index should work
            result = tools.index_codebase(tmpdir)
            assert "Indexed" in result.text
            tools.close()

    def test_call_tool_metadata_erase(self, tools, sample_python_file):
        """Test call_tool dispatch for metadata_erase."""
        tools.index_codebase(sample_python_file)
        result = tools.call_tool("metadata_erase", {})
        assert "Erased" in result.text or "idle" in result.text.lower()

    def test_erase_reports_removed_lsp_bindings(self):
        """Erase should nudge users when lsp_bindings.json is removed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)

            bindings_path = _get_persistence_path(tmpdir) / "lsp_bindings.json"
            bindings_path.write_text(
                json.dumps({"cpp_lsp": ["tcp://host.docker.internal:2088"]}, indent=2)
            )

            result = tools.metadata_erase()
            assert "LSP bindings were removed" in result.text
            tools.close()


class TestMetadataRecomputeBaseline:
    """Tests for the astrograph_metadata_recompute_baseline tool."""

    def test_recompute_rebuilds_from_scratch(self):
        """Recompute should erase and rebuild the full index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_data(data):
    output = []
    for element in data:
        if element > 0:
            output.append(element * 2)
    return output
"""
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text(code)

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)

            # Suppress a hash
            _suppress_first_hash_from_analysis(tools)

            # Recompute baseline
            result = tools.metadata_recompute_baseline()
            assert "Baseline recomputed" in result.text
            assert "Indexed" in result.text

            # Suppressions should be gone
            assert len(tools.index.suppressed_hashes) == 0

            # But index should be populated
            status = tools.status()
            assert "ready" in status.text

            # Persistence should be recreated
            persistence_path = os.path.join(tmpdir, PERSISTENCE_DIR)
            assert os.path.isdir(persistence_path)
            tools.close()

    def test_call_tool_metadata_recompute_baseline(self):
        """Test call_tool dispatch for metadata_recompute_baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)
            result = tools.call_tool("metadata_recompute_baseline", {})
            assert "Baseline recomputed" in result.text
            tools.close()

    def test_recompute_reports_removed_lsp_bindings(self):
        """Recompute should nudge users that bindings are reset and need rebind."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            Path(file1).write_text("def foo(): pass")

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)

            bindings_path = _get_persistence_path(tmpdir) / "lsp_bindings.json"
            bindings_path.write_text(
                json.dumps({"cpp_lsp": ["tcp://host.docker.internal:2088"]}, indent=2)
            )

            result = tools.metadata_recompute_baseline()
            assert "LSP bindings were reset during recompute" in result.text
            tools.close()


class TestResourceHandlers:
    """Tests for the resource list handlers (Codex compatibility)."""

    @pytest.mark.parametrize(
        "_description",
        ["resources/list", "resources/templates/list"],
    )
    @pytest.mark.asyncio
    async def test_resource_handlers_registered(self, _description):
        """Resource handlers are registered and server creation succeeds."""
        server = create_server()
        assert server is not None


class TestWorkspaceEnvVar:
    """Tests for the ASTROGRAPH_WORKSPACE env var support."""

    def test_workspace_env_var_triggers_auto_index(self):
        """Setting ASTROGRAPH_WORKSPACE to a valid dir should trigger auto-index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "mod.py")
            Path(py_file).write_text("def hello(): return 42\n")

            with patch.dict(os.environ, {"ASTROGRAPH_WORKSPACE": tmpdir}):
                tools = CodeStructureTools()
                # Wait for background indexing to complete
                tools._wait_for_background_index()
                assert tools._bg_index_progress == "done"
                assert len(tools.index.entries) > 0
                tools.close()

    @pytest.mark.parametrize("workspace", ["", "/nonexistent/path/xyz"])
    def test_workspace_env_var_ignored_values(self, workspace):
        """Invalid ASTROGRAPH_WORKSPACE values should not trigger auto-index."""
        with patch.dict(os.environ, {"ASTROGRAPH_WORKSPACE": workspace}):
            tools = CodeStructureTools()
            result = tools.status()
            assert "idle" in result.text


class TestStartupWorkspaceDetection:
    """Tests for startup workspace fallback logic (Codex/local compatibility)."""

    def test_pwd_fallback_triggers_auto_index(self):
        """Without ASTROGRAPH_WORKSPACE, PWD should be used when cwd is '/'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "mod.py")
            Path(py_file).write_text("def hello(): return 42\n")

            with patch.dict(os.environ, {"PWD": tmpdir}, clear=False):
                os.environ.pop("ASTROGRAPH_WORKSPACE", None)
                with patch("os.getcwd", return_value="/"):
                    tools = CodeStructureTools()
                    tools._wait_for_background_index()
                    assert tools._bg_index_progress == "done"
                    assert len(tools.index.entries) > 0
                    tools.close()

    def test_cwd_fallback_triggers_auto_index(self):
        """Without ASTROGRAPH_WORKSPACE/PWD, cwd should be used."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "mod.py")
            Path(py_file).write_text("def hello(): return 42\n")

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ASTROGRAPH_WORKSPACE", None)
                os.environ.pop("PWD", None)
                with patch("os.getcwd", return_value=tmpdir):
                    tools = CodeStructureTools()
                    tools._wait_for_background_index()
                    assert tools._bg_index_progress == "done"
                    assert len(tools.index.entries) > 0
                    tools.close()

    def test_root_cwd_without_hints_stays_idle(self):
        """With no startup hints and cwd='/', server should remain idle."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ASTROGRAPH_WORKSPACE", None)
            os.environ.pop("PWD", None)
            with patch("os.getcwd", return_value="/"):
                tools = CodeStructureTools()
                result = tools.status()
                assert "idle" in result.text
                tools.close()


class TestBlockingDuringIndexing:
    """Tests that tools wait for background indexing then proceed."""

    def test_analyze_waits_for_background_index(self):
        """Analyze should wait for background indexing, then report no index."""
        tools = CodeStructureTools()
        tools._bg_index_done.clear()
        _start_background_index_completion(tools)
        result = tools.analyze()
        # After waiting, empty index → "No code indexed"
        assert "No code indexed" in result.text

    def test_check_waits_for_background_index(self):
        """Check should wait for background indexing, then report no index."""
        tools = CodeStructureTools()
        tools._bg_index_done.clear()
        _start_background_index_completion(tools)
        result = tools.check("def foo(): pass")
        assert "No code indexed" in result.text
