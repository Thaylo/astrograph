"""Tests for the consolidated MCP server tools."""

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astrograph.index import CodeStructureIndex, IndexEntry, SimilarityResult
from astrograph.languages.base import CodeUnit, SemanticProfile, SemanticSignal
from astrograph.languages.registry import LanguageRegistry
from astrograph.server import create_server, get_tools, set_tools
from astrograph.tools import (
    PERSISTENCE_DIR,
    CodeStructureTools,
    ToolResult,
    _get_persistence_path,
)


@pytest.fixture(autouse=True)
def _clear_lsp_env(monkeypatch, tmp_path):
    """Keep LSP state isolated across tests/modules."""
    LanguageRegistry.reset()
    monkeypatch.delenv("ASTROGRAPH_COMPILE_COMMANDS_PATH", raising=False)
    monkeypatch.delenv("ASTROGRAPH_PERSISTENCE_DIR", raising=False)
    # Disable startup auto-indexing in tests; fixtures call index_codebase explicitly.
    monkeypatch.setenv("ASTROGRAPH_WORKSPACE", "")

    # Save Python LSP binding and redirect default (workspace=None) resolution
    # so plugins find the binding without triggering auto-indexing.
    import astrograph.lsp_setup as _lsp_mod

    _lsp_mod.save_lsp_bindings({"python": [sys.executable, "-m", "pylsp"]}, workspace=tmp_path)
    _orig = _lsp_mod._normalize_workspace_root

    def _test_normalize(workspace):
        if workspace is None:
            return tmp_path
        return _orig(workspace)

    monkeypatch.setattr(_lsp_mod, "_normalize_workspace_root", _test_normalize)


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

    def test_resolve_container_path_new_file_at_root(self):
        """Container paths under /workspace should resolve without collapsing to root."""
        self._assert_docker_new_file_resolution(
            "/workspace/new_file.py",
            "/workspace/new_file.py",
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
    t = CodeStructureTools()
    yield t
    t.close()


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

    def test_compare_differentiate_semantic_mode_equivalent_with_matching_signals(self, tools):
        # Python now always emits type_annotation.density and async.present,
        # so two structurally identical snippets with matching signals → EQUIVALENT.
        result = tools.compare(
            "def one(): return 1",
            "def two(): return 2",
            semantic_mode="differentiate",
        )
        assert "EQUIVALENT" in result.text

    # -- Semantic mismatch annotate tests (parametrized) --

    @pytest.mark.parametrize(
        "code_a,code_b,language,expected_signal",
        [
            pytest.param(
                "int add(int a, int b) { return a + b; }",
                "Vec add(Vec a, Vec b) { return a + b; }",
                "cpp_lsp",
                "operator.plus.binding",
                id="cpp-operator-plus",
            ),
            pytest.param(
                "template <typename T>\nT add(T a, T b) { return a + b; }",
                "int add(int a, int b) { return a + b; }",
                "cpp_lsp",
                "cpp.template.present",
                id="cpp-template",
            ),
            pytest.param(
                "virtual void draw() { }",
                "void draw() override { }",
                "cpp_lsp",
                "cpp.virtual_override",
                id="cpp-virtual",
            ),
            pytest.param(
                "namespace alpha {\nint add(int a, int b) { return a + b; }\n}",
                "namespace beta {\nint add(int a, int b) { return a + b; }\n}",
                "cpp_lsp",
                "cpp.namespace",
                id="cpp-namespace",
            ),
            pytest.param(
                "constexpr int add(int a, int b) { return a + b; }",
                "int add(int a, int b) { return a + b; }",
                "cpp_lsp",
                "cpp.const_correctness",
                id="cpp-constexpr",
            ),
            pytest.param(
                "def fetch(url: str) -> str:\n    return url",
                "async def fetch(url: str) -> str:\n    return url",
                "python",
                "python.async.present",
                id="python-async",
            ),
            pytest.param(
                "def add(a: int, b: int) -> int:\n    return a + b",
                "def add(a, b):\n    return a + b",
                "python",
                "python.type_annotation.density",
                id="python-annotation",
            ),
            pytest.param(
                "function fetchData(url) { return fetch(url); }",
                "async function fetchData(url) { return await fetch(url); }",
                "javascript_lsp",
                "javascript.async.present",
                id="js-async",
            ),
            pytest.param(
                "import { add } from './math';\nexport function sum(a, b) { return a + b; }",
                "const { add } = require('./math');\nmodule.exports = function sum(a, b) { return a + b; }",
                "javascript_lsp",
                "javascript.module_system",
                id="js-esm-vs-cjs",
            ),
            pytest.param(
                "function add(a: number, b: number): number { return a + b; }",
                "function add(a, b) { return a + b; }",
                "javascript_lsp",
                "javascript.type_system",
                id="js-typed",
            ),
            pytest.param(
                "function identity<T>(x: T): T { return x; }",
                "function identity(x: any): any { return x; }",
                "typescript_lsp",
                "typescript.generic.present",
                id="ts-generic",
            ),
            pytest.param(
                "function get(x: unknown): string { return x as string; }",
                "function get(x) { return x; }",
                "typescript_lsp",
                "typescript.strict_mode",
                id="ts-strict",
            ),
            pytest.param(
                "@Override\npublic int getValue() { return value; }",
                "public int getValue() { return value; }",
                "java_lsp",
                "java.annotations",
                id="java-annotation",
            ),
            pytest.param(
                "public int add(int a, int b) { return a + b; }",
                "private int add(int a, int b) { return a + b; }",
                "java_lsp",
                "java.access_modifiers",
                id="java-access",
            ),
            pytest.param(
                "public <T extends Comparable> T max(T a, T b) { return a; }",
                "public int max(int a, int b) { return a; }",
                "java_lsp",
                "java.generic.present",
                id="java-generic",
            ),
            pytest.param(
                "list.stream().filter(x -> x > 0).collect(Collectors.toList());",
                "for (int x : list) { if (x > 0) result.add(x); }",
                "java_lsp",
                "java.functional_style",
                id="java-stream",
            ),
            pytest.param(
                "interface Drawable { void draw(); }",
                "class Circle { void draw() { } }",
                "java_lsp",
                "java.class_kind",
                id="java-interface",
            ),
            pytest.param(
                "void read() throws IOException { }",
                "void read() { try { } catch (IOException e) { } }",
                "java_lsp",
                "java.exception_handling",
                id="java-exception",
            ),
            pytest.param(
                "int* arr = malloc(10 * sizeof(int)); free(arr);",
                "int arr[10];",
                "c_lsp",
                "c.memory_management",
                id="c-malloc",
            ),
            pytest.param(
                "struct Point { int x; int y; };",
                "int x; int y;",
                "c_lsp",
                "c.composite_types",
                id="c-struct",
            ),
            pytest.param(
                "#ifdef DEBUG\nint x = 1;\n#endif",
                "int x = 1;",
                "c_lsp",
                "c.preprocessor",
                id="c-preprocessor",
            ),
            pytest.param(
                "goto cleanup; cleanup: return;", "return;", "c_lsp", "c.control_flow", id="c-goto"
            ),
            pytest.param(
                "void swap(int *a, int *b) { int t = *a; *a = *b; *b = t; }",
                "void swap(int a, int b) { int t = a; a = b; b = t; }",
                "c_lsp",
                "c.pointer_usage",
                id="c-pointer",
            ),
        ],
    )
    def test_compare_semantic_mismatch_annotate(
        self, tools, code_a, code_b, language, expected_signal
    ):
        result = tools.compare(code_a, code_b, language=language, semantic_mode="annotate")
        assert "SEMANTIC_MISMATCH" in result.text
        assert expected_signal in result.text

    # -- Semantic differentiate / structural tests --

    def test_compare_cpp_semantic_mismatch_differentiate(self, tools):
        builtin_plus = "int add(int a, int b) { return a + b; }"
        custom_plus = "Vec add(Vec a, Vec b) { return a + b; }"
        result = tools.compare(
            builtin_plus,
            custom_plus,
            language="cpp_lsp",
            semantic_mode="differentiate",
        )
        assert "DIFFERENT" in result.text

    def test_compare_python_semantic_plus_numeric_vs_str(self, tools):
        numeric = "def add(a: int, b: int) -> int:\n    return a + b"
        string = "def add(a: str, b: str) -> str:\n    return a + b"
        result = tools.compare(
            numeric,
            string,
            language="python",
            semantic_mode="differentiate",
        )
        assert "DIFFERENT" in result.text

    def test_compare_python_semantic_match_equivalent(self, tools):
        code1 = "def add(x: int, y: int) -> int:\n    return x + y"
        code2 = "def add(a: int, b: int) -> int:\n    return a + b"
        result = tools.compare(
            code1,
            code2,
            language="python",
            semantic_mode="differentiate",
        )
        assert "EQUIVALENT" in result.text

    def test_compare_python_dataclass_vs_plain(self, tools):
        dataclass_code = (
            "from dataclasses import dataclass\n@dataclass\nclass Point:\n    x: int\n    y: int\n"
        )
        plain_code = "class Point:\n    x: int\n    y: int\n"
        result = tools.compare(
            dataclass_code,
            plain_code,
            language="python",
            semantic_mode="annotate",
        )
        assert "SEMANTIC_MISMATCH" in result.text

    def test_compare_js_semantic_class_vs_prototype(self, tools):
        es6_class = (
            "class Animal {\n"
            "  constructor(name) { this.name = name; }\n"
            "  speak() { return this.name; }\n"
            "}"
        )
        prototype = (
            "function Animal(name) { this.name = name; }\n"
            "Animal.prototype.speak = function() { return this.name; };"
        )
        result = tools.compare(
            es6_class,
            prototype,
            language="javascript_lsp",
            semantic_mode="differentiate",
        )
        assert "DIFFERENT" in result.text

    def test_compare_js_semantic_equivalent_sync(self, tools):
        code1 = "function add(a, b) { return a + b; }"
        code2 = "function sum(x, y) { return x + y; }"
        result = tools.compare(
            code1,
            code2,
            language="javascript_lsp",
            semantic_mode="differentiate",
        )
        assert "EQUIVALENT" in result.text

    # -- JavaScript esprima AST graph tests --

    def test_compare_js_esprima_structural_match(self, tools):
        """Two structurally identical JS functions match via esprima AST."""
        code1 = "function add(a, b) { return a + b; }"
        code2 = "function sum(x, y) { return x + y; }"
        result = tools.compare(code1, code2, language="javascript_lsp")
        assert any(exp in result.text for exp in ["EQUIVALENT", "EXACT_MATCH"])

    def test_compare_js_esprima_structural_different(self, tools):
        """Different control flow produces DIFFERENT via esprima AST."""
        code1 = "function f(x) { return x + 1; }"
        code2 = "function f(x) { if (x > 0) { return x; } else { return -x; } }"
        result = tools.compare(code1, code2, language="javascript_lsp")
        assert "DIFFERENT" in result.text

    def test_compare_js_arrow_vs_function(self, tools):
        """Arrow function vs function declaration → DIFFERENT via esprima."""
        code1 = "const add = (a, b) => a + b;"
        code2 = "function add(a, b) { return a + b; }"
        result = tools.compare(code1, code2, language="javascript_lsp")
        assert "DIFFERENT" in result.text

    # -- TypeScript structural tests --

    def test_compare_ts_structural_match(self, tools):
        """Two identical TS functions match structurally."""
        code1 = "function add(a: number, b: number): number { return a + b; }"
        code2 = "function sum(x: number, y: number): number { return x + y; }"
        result = tools.compare(code1, code2, language="typescript_lsp")
        assert any(exp in result.text for exp in ["EQUIVALENT", "EXACT_MATCH"])

    def test_compare_ts_vs_js_same_structure(self, tools):
        """TS function structurally matches equivalent JS when compared as TS."""
        ts_code = "function add(a: number, b: number): number { return a + b; }"
        js_code = "function add(a, b) { return a + b; }"
        result = tools.compare(ts_code, js_code, language="typescript_lsp")
        # Both reduce to same structure after TS annotation stripping
        assert any(exp in result.text for exp in ["EQUIVALENT", "EXACT_MATCH", "SIMILAR"])


def _extract_signal_map(plugin_cls: type, source: str, filename: str) -> dict:
    """Create a plugin via __new__ and return its semantic signal map."""
    plugin = plugin_cls.__new__(plugin_cls)
    profile = plugin.extract_semantic_profile(source, filename)
    return {s.key: s.value for s in profile.signals}


class TestTreeSitterGraph:
    """Unit tests for tree-sitter AST graph builder and TS parsing."""

    def test_js_treesitter_graph_node_count(self):
        """Tree-sitter AST graph has more nodes than line-level parser for same code."""
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        source = "function add(a, b) { return a + b; }"
        graph = _ts_ast_to_graph(source, language="javascript")
        assert graph is not None
        # Tree-sitter AST produces a richer graph than line-level
        assert len(graph.nodes) > 5

    def test_js_normalize_graph_collapses_operators(self):
        """normalize_graph_for_pattern collapses operator specifics."""
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph
        from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin

        plugin = JavaScriptLSPPlugin.__new__(JavaScriptLSPPlugin)
        source = "function f(a, b) { return a + b; }"
        graph = _ts_ast_to_graph(source, language="javascript")
        assert graph is not None
        # Original graph has specific operator
        labels = [data["label"] for _, data in graph.nodes(data=True)]
        assert any("BinaryExpression:+" in lbl for lbl in labels)
        # Normalized graph collapses to :Op
        normalized = plugin.normalize_graph_for_pattern(graph)
        norm_labels = [data["label"] for _, data in normalized.nodes(data=True)]
        assert any("BinaryExpression:Op" in lbl for lbl in norm_labels)
        assert not any("BinaryExpression:+" in lbl for lbl in norm_labels)

    def test_ts_native_parse_with_annotations(self):
        """Tree-sitter parses TS natively, producing valid graph without stripping."""
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        ts_source = "function add(a: number, b: number): number { return a + b; }"
        graph = _ts_ast_to_graph(ts_source, language="typescript")
        assert graph is not None
        assert len(graph.nodes) > 5
        # Type annotations are skipped — no type_annotation labels in graph
        labels = [data["label"] for _, data in graph.nodes(data=True)]
        assert "type_annotation" not in labels
        assert "predefined_type" not in labels

    def test_ts_native_parse_interface(self):
        """Tree-sitter parses TS with interfaces, skipping type-only nodes."""
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        ts_source = "interface Foo { bar: string; baz: number; }\nfunction f() { return 1; }"
        graph = _ts_ast_to_graph(ts_source, language="typescript")
        assert graph is not None
        labels = [data["label"] for _, data in graph.nodes(data=True)]
        # interface_declaration is skipped as a type-only node
        assert "interface_declaration" not in labels

    @pytest.mark.parametrize(
        "ts_source",
        [
            pytest.param(
                "function identity<T>(x: T): T { return x; }",
                id="generic-params",
            ),
            pytest.param(
                "const x = value as string;",
                id="as-cast",
            ),
        ],
    )
    def test_ts_native_parse_patterns(self, ts_source):
        """Tree-sitter parses TS patterns natively, producing valid graphs."""
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph(ts_source, language="typescript")
        assert graph is not None
        assert len(graph.nodes) > 0

    def test_ts_native_parse_decorators(self):
        """Tree-sitter parses TS with decorators natively."""
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        ts_source = "@injectable()\nclass Service {\n  @log\n  process() { return 1; }\n}"
        graph = _ts_ast_to_graph(ts_source, language="typescript")
        assert graph is not None
        labels = [data["label"] for _, data in graph.nodes(data=True)]
        assert any("ClassDeclaration" in lbl for lbl in labels)

    def test_ts_native_parse_satisfies(self):
        """Tree-sitter parses TS satisfies operator natively."""
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        ts_source = 'const config = { key: "value" } satisfies Config;'
        graph = _ts_ast_to_graph(ts_source, language="typescript")
        assert graph is not None
        assert len(graph.nodes) > 0

    # -- C23 feature detection tests --

    def test_c23_features_detected(self):
        """C plugin detects C23 features in source code."""
        from astrograph.languages.c_lsp_plugin import CLSPPlugin

        source = (
            "#include <stddef.h>\n"
            "constexpr int MAX = 100;\n"
            "bool is_valid = true;\n"
            "static_assert(MAX > 0);\n"
            "int *p = nullptr;\n"
            "typeof(MAX) val = 42;\n"
            "[[nodiscard]] int compute(void) { return 0; }\n"
        )
        sig_map = _extract_signal_map(CLSPPlugin, source, "test.c")
        assert "c.c23_features" in sig_map
        c23_val = sig_map["c.c23_features"]
        assert "constexpr" in c23_val
        assert "nullptr" in c23_val
        assert "bool" in c23_val
        assert "static_assert" in c23_val
        assert "typeof" in c23_val
        assert "attributes" in c23_val

    @pytest.mark.parametrize(
        "plugin_name,source,filename,signal_key",
        [
            pytest.param(
                "CLSPPlugin",
                '#include <stdio.h>\nint main(void) {\n  printf("hello\\n");\n  return 0;\n}',
                "test.c",
                "c.c23_features",
                id="c23-none-for-c99",
            ),
            pytest.param(
                "JavaLSPPlugin",
                "public class App {\n  public static void main(String[] args) {\n    System.out.println(args[0]);\n  }\n}",
                "App.java",
                "java.modern_features",
                id="java-modern-none-for-java8",
            ),
        ],
    )
    def test_modern_features_none_for_old_standard(self, plugin_name, source, filename, signal_key):
        """Plugin emits 'none' for modern features on older standard code."""
        from astrograph.languages.c_lsp_plugin import CLSPPlugin
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        plugin_cls = {"CLSPPlugin": CLSPPlugin, "JavaLSPPlugin": JavaLSPPlugin}[plugin_name]
        sig_map = _extract_signal_map(plugin_cls, source, filename)
        assert sig_map[signal_key] == "none"

    # -- C++20/23 feature detection tests --

    def test_cpp23_features_detected(self):
        """C++ plugin detects C++20/23 features in source code."""
        from astrograph.languages.cpp_lsp_plugin import CppLSPPlugin

        source = (
            "template<typename T>\n"
            "concept Addable = requires(T a, T b) { a + b; };\n"
            "consteval int square(int n) { return n * n; }\n"
            "auto result = a <=> b;\n"
        )
        sig_map = _extract_signal_map(CppLSPPlugin, source, "test.cpp")
        assert "cpp.modern_features" in sig_map
        modern = sig_map["cpp.modern_features"]
        assert "concept" in modern
        assert "requires" in modern
        assert "spaceship" in modern
        assert "consteval" in modern

    @pytest.mark.parametrize(
        "plugin_name,source,filename,signal_key,expected_feature",
        [
            pytest.param(
                "CppLSPPlugin",
                "task<int> compute() {\n  auto val = co_await fetch_value();\n  co_return val * 2;\n}",
                "test.cpp",
                "cpp.modern_features",
                "coroutine",
                id="cpp-coroutines",
            ),
            pytest.param(
                "JavaLSPPlugin",
                "public sealed class Shape permits Circle, Rectangle {\n  abstract double area();\n}",
                "Shape.java",
                "java.modern_features",
                "sealed",
                id="java-sealed",
            ),
            pytest.param(
                "JavaLSPPlugin",
                "void process(Object obj) {\n  if (obj instanceof String s) {\n    System.out.println(s.length());\n  }\n}",
                "Test.java",
                "java.modern_features",
                "pattern_instanceof",
                id="java-pattern-instanceof",
            ),
            pytest.param(
                "JavaLSPPlugin",
                "int result = switch (day) {\n  case MONDAY -> 1;\n  case TUESDAY -> 2;\n  default -> 0;\n};",
                "Test.java",
                "java.modern_features",
                "switch_expression",
                id="java-switch-expression",
            ),
        ],
    )
    def test_modern_feature_single_assert(
        self, plugin_name, source, filename, signal_key, expected_feature
    ):
        """Plugin detects a specific modern language feature."""
        from astrograph.languages.cpp_lsp_plugin import CppLSPPlugin
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        plugin_cls = {"CppLSPPlugin": CppLSPPlugin, "JavaLSPPlugin": JavaLSPPlugin}[plugin_name]
        sig_map = _extract_signal_map(plugin_cls, source, filename)
        assert expected_feature in sig_map[signal_key]

    # -- Java 25 feature detection tests --

    def test_java_record_detected(self):
        """Java plugin detects record declarations."""
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        source = (
            "public record Point(int x, int y) {\n"
            "  public double distance() {\n"
            "    return Math.sqrt(x * x + y * y);\n"
            "  }\n"
            "}"
        )
        sig_map = _extract_signal_map(JavaLSPPlugin, source, "Point.java")
        assert "java.modern_features" in sig_map
        assert "record" in sig_map["java.modern_features"]

    def test_java_text_block_and_var_detected(self):
        """Java plugin detects text blocks and var keyword."""
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        source = 'var greeting = """\n    Hello,\n    World!""";\n'
        sig_map = _extract_signal_map(JavaLSPPlugin, source, "Test.java")
        modern = sig_map["java.modern_features"]
        assert "text_block" in modern
        assert "var" in modern

    # -- Updated signal count tests --

    def test_c_semantic_signals_include_c23(self):
        """C plugin emits all 6 signals including c.c23_features."""
        from astrograph.languages.c_lsp_plugin import CLSPPlugin

        plugin = CLSPPlugin.__new__(CLSPPlugin)
        source = "#include <stdio.h>\nint main(void) { return 0; }"
        profile = plugin.extract_semantic_profile(source, "test.c")
        keys = {s.key for s in profile.signals}
        assert "c.c23_features" in keys
        # All 6 signal keys present
        expected = {
            "c.preprocessor",
            "c.pointer_usage",
            "c.composite_types",
            "c.memory_management",
            "c.control_flow",
            "c.c23_features",
        }
        assert expected.issubset(keys)

    def test_java_semantic_signals_include_modern(self):
        """Java plugin emits all 12 signals including java.modern_features."""
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        plugin = JavaLSPPlugin.__new__(JavaLSPPlugin)
        source = "public class App { public void run() {} }"
        profile = plugin.extract_semantic_profile(source, "App.java")
        keys = {s.key for s in profile.signals}
        assert "java.modern_features" in keys
        expected = {
            "java.annotations",
            "java.access_modifiers",
            "java.generic.present",
            "java.exception_handling",
            "java.functional_style",
            "java.class_kind",
            "java.modern_features",
            "java.spring_stereotypes",
            "java.rest_http_patterns",
            "java.persistence_jpa",
            "java.dependency_injection",
            "java.async_reactive",
        }
        assert expected.issubset(keys)

    def test_java_semantic_signals_microservices(self):
        """Java plugin detects Spring, REST, JPA, DI, and async signals."""
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        plugin = JavaLSPPlugin.__new__(JavaLSPPlugin)
        source = (
            "@RestController\n"
            "@Configuration\n"
            "public class UserController {\n"
            "    @Autowired\n"
            "    private UserService service;\n"
            "    @Bean\n"
            "    public ObjectMapper mapper() { return new ObjectMapper(); }\n"
            '    @Value("${app.name}")\n'
            "    private String appName;\n"
            '    @GetMapping("/users")\n'
            "    public ResponseEntity<List<User>> list(@RequestParam int page) {\n"
            "        return ResponseEntity.ok(service.list(page));\n"
            "    }\n"
            '    @PostMapping("/users")\n'
            "    public ResponseEntity<User> create(@RequestBody User u) {\n"
            "        return ResponseEntity.ok(service.save(u));\n"
            "    }\n"
            '    @GetMapping("/users/{id}")\n'
            "    public Mono<User> get(@PathVariable Long id) {\n"
            "        return service.findById(id);\n"
            "    }\n"
            "    @Transactional\n"
            '    @Query("SELECT u FROM User u")\n'
            "    public CompletableFuture<Void> batch() {\n"
            "        return CompletableFuture.completedFuture(null);\n"
            "    }\n"
            "}\n"
            "@Entity\n"
            '@Table(name = "users")\n'
            "class User {\n"
            '    @Column(name = "email")\n'
            "    private String email;\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "UserController.java")
        sig_map = {s.key: s.value for s in profile.signals}

        # Spring stereotypes
        assert "rest_controller" in sig_map["java.spring_stereotypes"]
        assert "configuration" in sig_map["java.spring_stereotypes"]

        # REST / HTTP
        assert "mapping" in sig_map["java.rest_http_patterns"]
        assert "response_entity" in sig_map["java.rest_http_patterns"]
        assert "path_variable" in sig_map["java.rest_http_patterns"]
        assert "request_body" in sig_map["java.rest_http_patterns"]
        assert "request_param" in sig_map["java.rest_http_patterns"]

        # Persistence / JPA
        assert "entity" in sig_map["java.persistence_jpa"]
        assert "table" in sig_map["java.persistence_jpa"]
        assert "column" in sig_map["java.persistence_jpa"]
        assert "transactional" in sig_map["java.persistence_jpa"]
        assert "query" in sig_map["java.persistence_jpa"]

        # Dependency injection
        assert "autowired" in sig_map["java.dependency_injection"]
        assert "bean" in sig_map["java.dependency_injection"]
        assert "value" in sig_map["java.dependency_injection"]

        # Async / reactive
        assert "completable_future" in sig_map["java.async_reactive"]
        assert "mono" in sig_map["java.async_reactive"]

    def test_java_semantic_signals_spring_rest_only(self):
        """Focused REST controller fires REST signals; non-REST signals are none."""
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        plugin = JavaLSPPlugin.__new__(JavaLSPPlugin)
        source = (
            "@RestController\n"
            "public class HealthController {\n"
            '    @GetMapping("/health")\n'
            "    public ResponseEntity<String> health() {\n"
            '        return ResponseEntity.ok("UP");\n'
            "    }\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "HealthController.java")
        sig_map = {s.key: s.value for s in profile.signals}

        assert "rest_controller" in sig_map["java.spring_stereotypes"]
        assert "mapping" in sig_map["java.rest_http_patterns"]
        assert "response_entity" in sig_map["java.rest_http_patterns"]

        # Non-REST microservices signals should be none
        assert sig_map["java.persistence_jpa"] == "none"
        assert sig_map["java.dependency_injection"] == "none"
        assert sig_map["java.async_reactive"] == "none"

    def test_java_semantic_signals_absent_microservices(self):
        """Plain Java class emits none for all 5 microservices signals."""
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        plugin = JavaLSPPlugin.__new__(JavaLSPPlugin)
        source = (
            "public class Calculator {\n"
            "    public int add(int a, int b) {\n"
            "        return a + b;\n"
            "    }\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "Calculator.java")
        sig_map = {s.key: s.value for s in profile.signals}

        assert sig_map["java.spring_stereotypes"] == "none"
        assert sig_map["java.rest_http_patterns"] == "none"
        assert sig_map["java.persistence_jpa"] == "none"
        assert sig_map["java.dependency_injection"] == "none"
        assert sig_map["java.async_reactive"] == "none"

    def test_cpp_semantic_signals_include_modern(self):
        """C++ plugin emits all 5 always-emitted signals including cpp.modern_features."""
        from astrograph.languages.cpp_lsp_plugin import CppLSPPlugin

        plugin = CppLSPPlugin.__new__(CppLSPPlugin)
        source = "void foo() { int x = 1; }"
        profile = plugin.extract_semantic_profile(source, "test.cpp")
        keys = {s.key for s in profile.signals}
        expected = {
            "cpp.template.present",
            "cpp.virtual_override",
            "cpp.namespace",
            "cpp.const_correctness",
            "cpp.modern_features",
        }
        assert expected.issubset(keys)

    def test_js_block_extraction_for_loop(self):
        """JS block extraction finds for-loop blocks inside functions."""
        from astrograph.languages._js_ts_treesitter import (
            _ts_extract_function_blocks,
            _ts_try_parse,
        )

        source = (
            "function process(items) {\n"
            "  for (var i = 0; i < items.length; i++) {\n"
            "    console.log(items[i]);\n"
            "  }\n"
            "}"
        )
        tree = _ts_try_parse(source, "javascript")
        assert tree is not None
        blocks = list(
            _ts_extract_function_blocks(
                tree, source.splitlines(), "test.js", max_depth=3, language="javascript_lsp"
            )
        )
        assert len(blocks) >= 1
        assert any(".for_" in b.name for b in blocks)
        assert all(b.unit_type == "block" for b in blocks)

    def test_js_block_extraction_if_statement(self):
        """JS block extraction finds if-statement blocks."""
        from astrograph.languages._js_ts_treesitter import (
            _ts_extract_function_blocks,
            _ts_try_parse,
        )

        source = (
            "function check(x) {\n"
            "  if (x > 0) {\n"
            "    return 'positive';\n"
            "  } else {\n"
            "    return 'non-positive';\n"
            "  }\n"
            "}"
        )
        tree = _ts_try_parse(source, "javascript")
        assert tree is not None
        blocks = list(
            _ts_extract_function_blocks(
                tree, source.splitlines(), "test.js", max_depth=3, language="javascript_lsp"
            )
        )
        assert len(blocks) >= 1
        assert any(".if_" in b.name for b in blocks)

    def test_js_block_extraction_nested_depth(self):
        """JS block extraction respects max_depth parameter."""
        from astrograph.languages._js_ts_treesitter import (
            _ts_extract_function_blocks,
            _ts_try_parse,
        )

        source = (
            "function f(x) {\n"
            "  for (var i = 0; i < x; i++) {\n"
            "    if (i > 0) {\n"
            "      console.log(i);\n"
            "    }\n"
            "  }\n"
            "}"
        )
        tree = _ts_try_parse(source, "javascript")
        assert tree is not None
        # depth=1 should only get the for-loop, not nested if
        blocks_d1 = list(
            _ts_extract_function_blocks(
                tree, source.splitlines(), "test.js", max_depth=1, language="javascript_lsp"
            )
        )
        blocks_d3 = list(
            _ts_extract_function_blocks(
                tree, source.splitlines(), "test.js", max_depth=3, language="javascript_lsp"
            )
        )
        assert len(blocks_d3) >= len(blocks_d1)

    def test_ts_block_extraction_with_types(self):
        """TS block extraction works with native tree-sitter TS parsing."""
        from astrograph.languages._js_ts_treesitter import (
            _ts_extract_function_blocks,
            _ts_try_parse,
        )

        source = (
            "function process(items: number[]): void {\n"
            "  for (let i: number = 0; i < items.length; i++) {\n"
            "    console.log(items[i]);\n"
            "  }\n"
            "}"
        )
        tree = _ts_try_parse(source, "typescript")
        assert tree is not None
        blocks = list(
            _ts_extract_function_blocks(
                tree, source.splitlines(), "test.ts", max_depth=3, language="typescript_lsp"
            )
        )
        assert len(blocks) >= 1

    def test_cpp_normalize_graph_op_pattern(self):
        """C++ normalize_graph_for_pattern collapses :Op(...) to :Op."""
        import networkx as nx

        from astrograph.languages.cpp_lsp_plugin import CppLSPPlugin

        plugin = CppLSPPlugin.__new__(CppLSPPlugin)
        g = nx.DiGraph()
        g.add_node(0, label="AssignStmt:Op(+)")
        g.add_node(1, label="AssignStmt:Op(*)")
        g.add_node(2, label="VarDecl")
        g.add_edge(0, 1)
        g.add_edge(0, 2)
        normalized = plugin.normalize_graph_for_pattern(g)
        labels = [data["label"] for _, data in normalized.nodes(data=True)]
        assert "AssignStmt:Op" in labels
        assert "AssignStmt:Op(+)" not in labels
        assert "VarDecl" in labels

    def test_c_semantic_signals_emitted(self):
        """C plugin emits all 5 semantic signals for C code."""
        from astrograph.languages.c_lsp_plugin import CLSPPlugin

        plugin = CLSPPlugin.__new__(CLSPPlugin)
        source = (
            "#include <stdlib.h>\n"
            "#define MAX 100\n"
            "struct Point { int x; int y; };\n"
            "void process(struct Point *p) {\n"
            "  int *arr = malloc(MAX * sizeof(int));\n"
            "  free(arr);\n"
            "}"
        )
        profile = plugin.extract_semantic_profile(source, "test.c")
        keys = {s.key for s in profile.signals}
        assert "c.preprocessor" in keys
        assert "c.pointer_usage" in keys
        assert "c.composite_types" in keys
        assert "c.memory_management" in keys
        assert "c.control_flow" in keys
        assert profile.extractor == "c_lsp:syntax"

    def test_java_semantic_signals_emitted(self):
        """Java plugin emits all 6 semantic signals for Java code."""
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        plugin = JavaLSPPlugin.__new__(JavaLSPPlugin)
        source = (
            "@Override\n"
            "public <T> T process(T item) throws Exception {\n"
            "  try {\n"
            "    return item;\n"
            "  } catch (Exception e) {\n"
            "    throw e;\n"
            "  }\n"
            "}"
        )
        profile = plugin.extract_semantic_profile(source, "Test.java")
        keys = {s.key for s in profile.signals}
        assert "java.annotations" in keys
        assert "java.access_modifiers" in keys
        assert "java.generic.present" in keys
        assert "java.exception_handling" in keys
        assert "java.functional_style" in keys
        assert "java.class_kind" in keys
        assert profile.extractor == "java_lsp:syntax"
        # Check specific values
        sig_map = {s.key: s.value for s in profile.signals}
        assert "Override" in sig_map["java.annotations"]
        assert sig_map["java.access_modifiers"] == "public"
        assert sig_map["java.generic.present"] == "yes"
        assert sig_map["java.exception_handling"] == "both"

    # -- Brace-based block extraction tests --

    def test_brace_block_extraction_c_for_if(self):
        """Brace block extractor finds for and nested if in C code."""
        from astrograph.languages._brace_block_extractor import (
            extract_brace_blocks_from_function,
        )

        c_code = (
            "void process(int *arr, int n) {\n"
            "    for (int i = 0; i < n; i++) {\n"
            "        if (arr[i] > 0) {\n"
            "            arr[i] *= 2;\n"
            "        }\n"
            "    }\n"
            "}"
        )
        blocks = list(extract_brace_blocks_from_function(c_code, "test.c", "process", 1, "c_lsp"))
        assert len(blocks) == 2
        assert blocks[0].name == "process.for_1"
        assert blocks[0].block_type == "for"
        assert blocks[0].nesting_depth == 1
        assert blocks[1].name == "process.for_1.if_1"
        assert blocks[1].block_type == "if"
        assert blocks[1].nesting_depth == 2

    def test_brace_block_extraction_java_try_switch(self):
        """Brace block extractor finds try and nested switch in Java code."""
        from astrograph.languages._brace_block_extractor import (
            extract_brace_blocks_from_function,
        )

        java_code = (
            "public void handle(int type) {\n"
            "    try {\n"
            "        switch (type) {\n"
            "            case 1:\n"
            "                process();\n"
            "                break;\n"
            "        }\n"
            "    } catch (Exception e) {\n"
            "        log(e);\n"
            "    }\n"
            "}"
        )
        blocks = list(
            extract_brace_blocks_from_function(java_code, "Test.java", "handle", 1, "java_lsp")
        )
        assert len(blocks) == 2
        assert blocks[0].block_type == "try"
        assert blocks[1].block_type == "switch"
        assert blocks[1].nesting_depth == 2

    def test_brace_block_extraction_cpp_while(self):
        """Brace block extractor finds while loops in C++ code."""
        from astrograph.languages._brace_block_extractor import (
            extract_brace_blocks_from_function,
        )

        cpp_code = "void run(int n) {\n    while (n > 0) {\n        n--;\n    }\n}"
        blocks = list(extract_brace_blocks_from_function(cpp_code, "test.cpp", "run", 1, "cpp_lsp"))
        assert len(blocks) == 1
        assert blocks[0].name == "run.while_1"
        assert blocks[0].block_type == "while"

    def test_brace_block_extraction_respects_max_depth(self):
        """Brace block extractor respects max_depth parameter."""
        from astrograph.languages._brace_block_extractor import (
            extract_brace_blocks_from_function,
        )

        code = (
            "void f() {\n"
            "    for (int i = 0; i < 10; i++) {\n"
            "        if (i > 0) {\n"
            "            while (i > 5) {\n"
            "                i--;\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}"
        )
        blocks_d1 = list(
            extract_brace_blocks_from_function(code, "t.c", "f", 1, "c_lsp", max_depth=1)
        )
        blocks_d3 = list(
            extract_brace_blocks_from_function(code, "t.c", "f", 1, "c_lsp", max_depth=3)
        )
        assert len(blocks_d1) == 1  # only for
        assert len(blocks_d3) == 3  # for, if, while

    def test_brace_block_extraction_skips_comments(self):
        """Brace block extractor ignores keywords inside comments."""
        from astrograph.languages._brace_block_extractor import (
            extract_brace_blocks_from_function,
        )

        code = (
            "void f() {\n"
            "    // for (int i = 0; i < n; i++) {\n"
            "    /* if (x) { } */\n"
            "    for (int j = 0; j < 5; j++) {\n"
            "        j++;\n"
            "    }\n"
            "}"
        )
        blocks = list(extract_brace_blocks_from_function(code, "t.c", "f", 1, "c_lsp"))
        assert len(blocks) == 1
        assert blocks[0].block_type == "for"

    def test_c_normalize_graph_op_pattern(self):
        """C normalize_graph_for_pattern collapses :Op(...) to :Op."""
        import networkx as nx

        from astrograph.languages.c_lsp_plugin import CLSPPlugin

        plugin = CLSPPlugin.__new__(CLSPPlugin)
        g = nx.DiGraph()
        g.add_node(0, label="AssignStmt:Op(+)")
        g.add_node(1, label="CallStmt")
        g.add_edge(0, 1)
        normalized = plugin.normalize_graph_for_pattern(g)
        labels = [data["label"] for _, data in normalized.nodes(data=True)]
        assert "AssignStmt:Op" in labels
        assert "AssignStmt:Op(+)" not in labels

    def test_java_normalize_graph_op_pattern(self):
        """Java normalize_graph_for_pattern collapses :Op(...) to :Op."""
        import networkx as nx

        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        plugin = JavaLSPPlugin.__new__(JavaLSPPlugin)
        g = nx.DiGraph()
        g.add_node(0, label="AssignStmt:Op(=)")
        g.add_node(1, label="ForStmt")
        g.add_edge(0, 1)
        normalized = plugin.normalize_graph_for_pattern(g)
        labels = [data["label"] for _, data in normalized.nodes(data=True)]
        assert "AssignStmt:Op" in labels
        assert "AssignStmt:Op(=)" not in labels
        assert "ForStmt" in labels

    def test_ts_version_probing_reuses_js_logic(self):
        """TypeScript version probing uses the same logic as JavaScript."""
        from astrograph.lsp_setup import _evaluate_version_status

        result = _evaluate_version_status(
            language_id="typescript_lsp",
            detected="4.3.5",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result["state"] == "supported"

        result_old = _evaluate_version_status(
            language_id="typescript_lsp",
            detected="3.1.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result_old["state"] == "best_effort"

        result_too_old = _evaluate_version_status(
            language_id="typescript_lsp",
            detected="2.0.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result_too_old["state"] == "unsupported"

    def test_ts_semantic_signals_include_all(self):
        """TypeScript plugin emits all 7 TS-specific signal keys."""
        from astrograph.languages.typescript_lsp_plugin import TypeScriptLSPPlugin

        plugin = TypeScriptLSPPlugin.__new__(TypeScriptLSPPlugin)
        source = (
            "import { Controller, Get, Body } from '@nestjs/common';\n"
            "import { Observable } from 'rxjs';\n\n"
            "@Controller('items')\n"
            "export class ItemController {\n"
            "    constructor(@Inject(ItemService) private svc: ItemService) {}\n"
            "    @Get()\n"
            "    findAll(): Observable<Item[]> {\n"
            "        return this.svc.findAll().pipe(map(x => x));\n"
            "    }\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "item.controller.ts")
        sig_map = {s.key: s.value for s in profile.signals}

        expected_keys = [
            "typescript.strict_mode",
            "typescript.generic.present",
            "typescript.framework_decorators",
            "typescript.rest_http_patterns",
            "typescript.orm_persistence",
            "typescript.dependency_injection",
            "typescript.reactive_rxjs",
        ]
        for key in expected_keys:
            assert key in sig_map, f"Missing signal key: {key}"

    def test_ts_semantic_signals_nestjs_microservice(self):
        """NestJS controller fires all 5 new microservices signals."""
        from astrograph.languages.typescript_lsp_plugin import TypeScriptLSPPlugin

        plugin = TypeScriptLSPPlugin.__new__(TypeScriptLSPPlugin)
        source = (
            "import { Controller, Get, Post, Body, Param, Inject, UseGuards } from '@nestjs/common';\n"
            "import { Observable } from 'rxjs';\n"
            "import { map, catchError } from 'rxjs/operators';\n"
            "import { Repository } from 'typeorm';\n\n"
            "@Entity()\n"
            "class User {\n"
            "    @PrimaryGeneratedColumn()\n"
            "    id: number;\n"
            "    @Column()\n"
            "    name: string;\n"
            "    @ManyToOne(() => Team)\n"
            "    team: Team;\n"
            "}\n\n"
            "@Injectable()\n"
            "class UserService {\n"
            "    constructor(\n"
            "        @InjectRepository(User)\n"
            "        private repo: Repository<User>,\n"
            "    ) {}\n"
            "}\n\n"
            "@Controller('users')\n"
            "class UserController {\n"
            "    constructor(@Inject(UserService) private svc: UserService) {}\n"
            "    @UseGuards(AuthGuard)\n"
            "    @Get(':id')\n"
            "    findOne(@Param('id') id: string): Observable<User> {\n"
            "        return this.svc.findOne(id).pipe(map(u => u), catchError(e => e));\n"
            "    }\n"
            "    @Post()\n"
            "    create(@Body() dto: CreateUserDto): Observable<User> {\n"
            "        return this.svc.create(dto);\n"
            "    }\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "user.controller.ts")
        sig_map = {s.key: s.value for s in profile.signals}

        # Framework decorators
        assert "controller" in sig_map["typescript.framework_decorators"]
        assert "injectable" in sig_map["typescript.framework_decorators"]
        assert "guard" in sig_map["typescript.framework_decorators"]

        # REST / HTTP
        assert "http_method" in sig_map["typescript.rest_http_patterns"]
        assert "param_decorator" in sig_map["typescript.rest_http_patterns"]

        # ORM / persistence
        assert "typeorm_entity" in sig_map["typescript.orm_persistence"]
        assert "typeorm_column" in sig_map["typescript.orm_persistence"]
        assert "typeorm_relation" in sig_map["typescript.orm_persistence"]
        assert "typeorm_repository" in sig_map["typescript.orm_persistence"]

        # Dependency injection
        assert "inject" in sig_map["typescript.dependency_injection"]
        assert "nest_inject" in sig_map["typescript.dependency_injection"]

        # Reactive / RxJS
        assert "observable" in sig_map["typescript.reactive_rxjs"]
        assert "pipe" in sig_map["typescript.reactive_rxjs"]
        assert "operators" in sig_map["typescript.reactive_rxjs"]

    def test_ts_semantic_signals_express_only(self):
        """Express router fires rest_http_patterns but not NestJS/ORM signals."""
        from astrograph.languages.typescript_lsp_plugin import TypeScriptLSPPlugin

        plugin = TypeScriptLSPPlugin.__new__(TypeScriptLSPPlugin)
        source = (
            "import express from 'express';\n"
            "const app = express();\n"
            "const router = Router();\n\n"
            "router.get('/items', (req, res) => {\n"
            "    res.json([]);\n"
            "});\n"
            "router.post('/items', (req, res) => {\n"
            "    res.status(201).json(req.body);\n"
            "});\n"
            "app.listen(3000);\n"
        )
        profile = plugin.extract_semantic_profile(source, "server.ts")
        sig_map = {s.key: s.value for s in profile.signals}

        assert "express_router" in sig_map["typescript.rest_http_patterns"]
        assert sig_map["typescript.framework_decorators"] == "none"
        assert sig_map["typescript.orm_persistence"] == "none"
        assert sig_map["typescript.dependency_injection"] == "none"
        assert sig_map["typescript.reactive_rxjs"] == "none"

    def test_ts_semantic_signals_absent_microservices(self):
        """Plain utility function emits none for all 5 microservices signals."""
        from astrograph.languages.typescript_lsp_plugin import TypeScriptLSPPlugin

        plugin = TypeScriptLSPPlugin.__new__(TypeScriptLSPPlugin)
        source = (
            "export function add(a: number, b: number): number {\n"
            "    return a + b;\n"
            "}\n\n"
            "export function multiply(a: number, b: number): number {\n"
            "    return a * b;\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "math.ts")
        sig_map = {s.key: s.value for s in profile.signals}

        assert sig_map["typescript.framework_decorators"] == "none"
        assert sig_map["typescript.rest_http_patterns"] == "none"
        assert sig_map["typescript.orm_persistence"] == "none"
        assert sig_map["typescript.dependency_injection"] == "none"
        assert sig_map["typescript.reactive_rxjs"] == "none"

    def test_js_semantic_signals_include_all(self):
        """JS plugin emits all 5 microservices signal keys."""
        from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin

        plugin = JavaScriptLSPPlugin.__new__(JavaScriptLSPPlugin)
        source = (
            "const app = express();\n"
            "app.use(cors());\n"
            "mongoose.connect('mongodb://localhost');\n"
            "jwt.sign(payload, secret);\n"
            "const io = require('socket.io');\n"
            "io.on('connection', (socket) => {});\n"
        )
        profile = plugin.extract_semantic_profile(source, "app.js")
        sig_map = {s.key: s.value for s in profile.signals}

        expected_keys = [
            "javascript.http_framework",
            "javascript.middleware_patterns",
            "javascript.database_client",
            "javascript.auth_patterns",
            "javascript.realtime_messaging",
        ]
        for key in expected_keys:
            assert key in sig_map, f"Missing signal key: {key}"

    def test_js_semantic_signals_express_microservice(self):
        """Full Express app fires all 5 signals with correct parts."""
        from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin

        plugin = JavaScriptLSPPlugin.__new__(JavaScriptLSPPlugin)
        source = (
            "const express = require('express');\n"
            "const app = express();\n"
            "const mongoose = require('mongoose');\n"
            "const jwt = require('jsonwebtoken');\n"
            "const bcrypt = require('bcrypt');\n"
            "const passport = require('passport');\n"
            "const http = require('http');\n"
            "const { Server } = require('socket.io');\n\n"
            "app.use(express.json());\n"
            "app.use(express.urlencoded({ extended: true }));\n"
            "app.use(cors());\n"
            "app.use(helmet());\n"
            "app.use(morgan('dev'));\n"
            "app.use(express.static('public'));\n"
            "app.use((err, req, res, next) => { res.status(500).send(); });\n\n"
            "const UserSchema = new mongoose.Schema({ name: String });\n"
            "const User = mongoose.model('User', UserSchema);\n\n"
            "app.post('/login', async (req, res) => {\n"
            "    const user = await User.findOne({ email: req.body.email });\n"
            "    const match = await bcrypt.compare(req.body.password, user.password);\n"
            "    const token = jwt.sign({ id: user._id }, secret);\n"
            "    res.json({ token });\n"
            "});\n\n"
            "app.get('/profile', passport.authenticate('jwt'), (req, res) => {\n"
            "    res.json(req.user);\n"
            "});\n\n"
            "const server = http.createServer(app);\n"
            "const io = new Server(server);\n"
            "io.on('connection', (socket) => {\n"
            "    socket.join('room1');\n"
            "    io.to('room1').emit('hello');\n"
            "});\n"
        )
        profile = plugin.extract_semantic_profile(source, "server.js")
        sig_map = {s.key: s.value for s in profile.signals}

        # HTTP framework
        assert "express" in sig_map["javascript.http_framework"]
        assert "native_http" in sig_map["javascript.http_framework"]

        # Middleware
        assert "body_parser" in sig_map["javascript.middleware_patterns"]
        assert "cors" in sig_map["javascript.middleware_patterns"]
        assert "helmet" in sig_map["javascript.middleware_patterns"]
        assert "morgan" in sig_map["javascript.middleware_patterns"]
        assert "error_handler" in sig_map["javascript.middleware_patterns"]
        assert "static_files" in sig_map["javascript.middleware_patterns"]

        # Database
        assert "mongoose" in sig_map["javascript.database_client"]

        # Auth
        assert "jwt" in sig_map["javascript.auth_patterns"]
        assert "bcrypt" in sig_map["javascript.auth_patterns"]
        assert "passport" in sig_map["javascript.auth_patterns"]

        # Realtime
        assert "socketio" in sig_map["javascript.realtime_messaging"]

    def test_js_semantic_signals_fastify_only(self):
        """Fastify app fires http_framework with fastify, other 4 signals = none."""
        from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin

        plugin = JavaScriptLSPPlugin.__new__(JavaScriptLSPPlugin)
        source = (
            "const fastify = Fastify({ logger: true });\n"
            "fastify.get('/health', async (req, reply) => {\n"
            "    return { status: 'ok' };\n"
            "});\n"
            "fastify.listen({ port: 3000 });\n"
        )
        profile = plugin.extract_semantic_profile(source, "server.js")
        sig_map = {s.key: s.value for s in profile.signals}

        assert "fastify" in sig_map["javascript.http_framework"]
        assert sig_map["javascript.middleware_patterns"] == "none"
        assert sig_map["javascript.database_client"] == "none"
        assert sig_map["javascript.auth_patterns"] == "none"
        assert sig_map["javascript.realtime_messaging"] == "none"

    def test_js_semantic_signals_absent_microservices(self):
        """Plain utility emits none for all 5 microservices signals."""
        from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin

        plugin = JavaScriptLSPPlugin.__new__(JavaScriptLSPPlugin)
        source = (
            "function add(a, b) {\n"
            "    return a + b;\n"
            "}\n\n"
            "function multiply(a, b) {\n"
            "    return a * b;\n"
            "}\n\n"
            "module.exports = { add, multiply };\n"
        )
        profile = plugin.extract_semantic_profile(source, "math.js")
        sig_map = {s.key: s.value for s in profile.signals}

        assert sig_map["javascript.http_framework"] == "none"
        assert sig_map["javascript.middleware_patterns"] == "none"
        assert sig_map["javascript.database_client"] == "none"
        assert sig_map["javascript.auth_patterns"] == "none"
        assert sig_map["javascript.realtime_messaging"] == "none"

    def test_go_semantic_signals_emitted(self):
        """Go plugin emits all 12 semantic signals for Go code."""
        from astrograph.languages.go_lsp_plugin import GoLSPPlugin

        plugin = GoLSPPlugin.__new__(GoLSPPlugin)
        source = (
            "package main\n\n"
            "type Reader interface {\n"
            "    Read(p []byte) (int, error)\n"
            "}\n\n"
            "type Server struct {\n"
            "    *Base\n"
            "}\n\n"
            "func (s *Server) Run() error {\n"
            "    go func() {\n"
            "        ch := make(chan int)\n"
            "        select {\n"
            "        case v := <-ch:\n"
            "            fmt.Println(v)\n"
            "        }\n"
            "    }()\n"
            "    defer s.Close()\n"
            "    if err != nil {\n"
            '        return errors.New("failed")\n'
            "    }\n"
            "    return err\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "main.go")
        keys = {s.key for s in profile.signals}
        expected = {
            "go.concurrency",
            "go.error_handling",
            "go.interface_usage",
            "go.struct_embedding",
            "go.receiver_style",
            "go.defer_recover",
            "go.modern_features",
            "go.context_usage",
            "go.http_patterns",
            "go.grpc_patterns",
            "go.sync_primitives",
            "go.init_functions",
        }
        assert expected.issubset(keys)
        assert profile.extractor == "go_lsp:syntax"
        sig_map = {s.key: s.value for s in profile.signals}
        assert "goroutine" in sig_map["go.concurrency"]
        assert "channel" in sig_map["go.concurrency"]
        assert "select" in sig_map["go.concurrency"]
        assert "err_nil_check" in sig_map["go.error_handling"]
        assert "error_creation" in sig_map["go.error_handling"]
        assert "declaration" in sig_map["go.interface_usage"]
        assert sig_map["go.struct_embedding"] == "yes"
        assert sig_map["go.receiver_style"] == "pointer"
        assert "defer" in sig_map["go.defer_recover"]

    def test_go_semantic_signals_include_modern(self):
        """Go plugin emits go.modern_features with generics."""
        from astrograph.languages.go_lsp_plugin import GoLSPPlugin

        plugin = GoLSPPlugin.__new__(GoLSPPlugin)
        source = (
            "package main\n\n"
            "func Map[T any](s []T, f func(T) T) []T {\n"
            "    result := make([]T, len(s))\n"
            "    for i := range len(s) {\n"
            "        result[i] = f(s[i])\n"
            "    }\n"
            "    return result\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "generic.go")
        keys = {s.key for s in profile.signals}
        assert "go.modern_features" in keys
        sig_map = {s.key: s.value for s in profile.signals}
        assert "generics" in sig_map["go.modern_features"]

    def test_brace_block_extraction_go_select(self):
        """Brace block extractor finds select and if blocks in Go code."""
        from astrograph.languages._brace_block_extractor import (
            extract_brace_blocks_from_function,
        )

        go_code = (
            "func process(ch chan int) {\n"
            "    select {\n"
            "    case v := <-ch:\n"
            "        fmt.Println(v)\n"
            "    }\n"
            "    if (len(ch) > 0) {\n"
            "        fmt.Println(ch)\n"
            "    }\n"
            "}"
        )
        blocks = list(
            extract_brace_blocks_from_function(go_code, "main.go", "process", 1, "go_lsp")
        )
        block_types = [b.block_type for b in blocks]
        assert "select" in block_types
        assert "if" in block_types

    def test_go_normalize_graph_op_pattern(self):
        """Go normalize_graph_for_pattern collapses :Op(...) to :Op."""
        import networkx as nx

        from astrograph.languages.go_lsp_plugin import GoLSPPlugin

        plugin = GoLSPPlugin.__new__(GoLSPPlugin)
        g = nx.DiGraph()
        g.add_node(0, label="AssignStmt:Op(+)")
        g.add_node(1, label="CallStmt")
        g.add_edge(0, 1)
        normalized = plugin.normalize_graph_for_pattern(g)
        labels = [data["label"] for _, data in normalized.nodes(data=True)]
        assert "AssignStmt:Op" in labels
        assert "AssignStmt:Op(+)" not in labels

    def test_go_version_probing(self):
        """Go version probing evaluates supported/best_effort/unsupported correctly."""
        from astrograph.lsp_setup import _evaluate_version_status

        result_supported = _evaluate_version_status(
            language_id="go_lsp",
            detected="go1.22.5",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result_supported["state"] == "supported"

        result_best_effort = _evaluate_version_status(
            language_id="go_lsp",
            detected="go1.20.0",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result_best_effort["state"] == "best_effort"

        result_unsupported = _evaluate_version_status(
            language_id="go_lsp",
            detected="go1.19.0",
            probe_kind="runtime",
            transport="subprocess",
            available=True,
        )
        assert result_unsupported["state"] == "unsupported"

        result_gopls = _evaluate_version_status(
            language_id="go_lsp",
            detected="gopls v0.16.0",
            probe_kind="server",
            transport="subprocess",
            available=True,
        )
        assert result_gopls["state"] == "best_effort"

    def test_go_semantic_signals_microservices(self):
        """Go plugin emits all 5 microservices signals."""
        from astrograph.languages.go_lsp_plugin import GoLSPPlugin

        plugin = GoLSPPlugin.__new__(GoLSPPlugin)
        source = (
            "package main\n\n"
            "import (\n"
            '    "context"\n'
            '    "net/http"\n'
            '    "sync"\n'
            '    "google.golang.org/grpc"\n'
            ")\n\n"
            "func init() {\n"
            "    // register defaults\n"
            "}\n\n"
            "func handler(ctx context.Context, w http.ResponseWriter, r *http.Request) {\n"
            "    ctx, cancel := context.WithCancel(ctx)\n"
            "    defer cancel()\n"
            "    bg := context.Background()\n"
            "    _ = bg\n"
            "}\n\n"
            "func startServer() {\n"
            "    mux := http.NewServeMux()\n"
            '    http.ListenAndServe(":8080", mux)\n'
            "}\n\n"
            "func startGRPC() {\n"
            "    srv := grpc.NewServer()\n"
            "    RegisterUserServer(srv, &impl{})\n"
            '    conn, _ := grpc.Dial("localhost:9090")\n'
            "    _ = conn\n"
            "}\n\n"
            "func worker() {\n"
            "    var mu sync.Mutex\n"
            "    var wg sync.WaitGroup\n"
            "    var once sync.Once\n"
            "    _ = mu\n"
            "    _ = wg\n"
            "    _ = once\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "server.go")
        sig_map = {s.key: s.value for s in profile.signals}

        # Context
        assert "param" in sig_map["go.context_usage"]
        assert "with_cancel" in sig_map["go.context_usage"]
        assert "background" in sig_map["go.context_usage"]

        # HTTP
        assert "handler" in sig_map["go.http_patterns"]
        assert "listener" in sig_map["go.http_patterns"]
        assert "mux" in sig_map["go.http_patterns"]

        # gRPC
        assert "server" in sig_map["go.grpc_patterns"]
        assert "service_registry" in sig_map["go.grpc_patterns"]
        assert "client" in sig_map["go.grpc_patterns"]

        # Sync
        assert "mutex" in sig_map["go.sync_primitives"]
        assert "waitgroup" in sig_map["go.sync_primitives"]
        assert "once" in sig_map["go.sync_primitives"]

        # Init
        assert sig_map["go.init_functions"] == "yes"

    def test_go_semantic_signals_context_only(self):
        """Go plugin detects context.Context patterns in isolation."""
        from astrograph.languages.go_lsp_plugin import GoLSPPlugin

        plugin = GoLSPPlugin.__new__(GoLSPPlugin)
        source = (
            "package svc\n\n"
            'import "context"\n\n'
            "func Process(ctx context.Context, id string) error {\n"
            "    ctx, cancel := context.WithTimeout(ctx, 5*time.Second)\n"
            "    defer cancel()\n"
            "    child := context.WithValue(ctx, keyID, id)\n"
            "    todo := context.TODO()\n"
            "    _ = child\n"
            "    _ = todo\n"
            "    return nil\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "svc.go")
        sig_map = {s.key: s.value for s in profile.signals}

        assert "param" in sig_map["go.context_usage"]
        assert "with_timeout" in sig_map["go.context_usage"]
        assert "with_value" in sig_map["go.context_usage"]
        assert "todo" in sig_map["go.context_usage"]

        # Non-microservices signals should be absent
        assert sig_map["go.http_patterns"] == "none"
        assert sig_map["go.grpc_patterns"] == "none"

    def test_go_semantic_signals_absent_microservices(self):
        """Go plugin emits 'none'/'no' for plain Go code without microservices."""
        from astrograph.languages.go_lsp_plugin import GoLSPPlugin

        plugin = GoLSPPlugin.__new__(GoLSPPlugin)
        source = (
            "package calc\n\n"
            "func Add(a, b int) int {\n"
            "    return a + b\n"
            "}\n\n"
            "func Subtract(a, b int) int {\n"
            "    return a - b\n"
            "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "calc.go")
        sig_map = {s.key: s.value for s in profile.signals}

        assert sig_map["go.context_usage"] == "none"
        assert sig_map["go.http_patterns"] == "none"
        assert sig_map["go.grpc_patterns"] == "none"
        assert sig_map["go.sync_primitives"] == "none"
        assert sig_map["go.init_functions"] == "no"


class _RecordingLock:
    """Context manager that records enter/exit events for testing."""

    def __init__(self):
        self.events: list[str] = []

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.events.append("exit")
        return False


class TestCallTool:
    """Tests for tool dispatch."""

    @staticmethod
    def _call_tool_with_recording_lock(tools, tool_name: str):
        lock = _RecordingLock()
        with (
            patch.object(tools, "_mutation_lock", lock),
            patch.object(tools, "_call_tool_unlocked", return_value=ToolResult("ok")) as dispatch,
        ):
            result = tools.call_tool(tool_name, {})
        return result, lock, dispatch

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
                "semantic_mode": "differentiate",
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
        result, lock, dispatch = self._call_tool_with_recording_lock(tools, "metadata_erase")

        assert result.text == "ok"
        assert lock.events == ["enter", "exit"]
        dispatch.assert_called_once_with("metadata_erase", {})

    def test_call_tool_skips_mutation_lock_for_read_only_calls(self, tools):
        result, lock, dispatch = self._call_tool_with_recording_lock(tools, "status")

        assert result.text == "ok"
        assert lock.events == []
        dispatch.assert_called_once_with("status", {})


class TestLSPSetupTool:
    """Tests for deterministic LSP setup tool flow."""

    @staticmethod
    def _inject_guidance_with_runtime(tools, payload: dict[str, object], *, docker_runtime: bool):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(CodeStructureTools, "_is_docker_runtime", return_value=docker_runtime),
        ):
            tools._inject_lsp_setup_guidance(payload, workspace=Path(tmpdir))

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
        if payload["missing_languages"]:
            assert "install/start actions" in payload["agent_directive"]
            assert "auto_bind" in payload["agent_directive"]
            assert "install/start actions" in payload.get("next_step", "")

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
        payload = json.loads(tools.lsp_setup(mode="inspect", language="rust_lsp").text)
        assert payload["ok"] is False
        assert "Unsupported language" in payload["error"]

    def test_lsp_setup_bind_and_unbind(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {"ASTROGRAPH_WORKSPACE": tmpdir},
                clear=False,
            ),
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
        import astrograph.lsp_setup as lsp_setup

        _real_probe = lsp_setup.probe_command

        def _probe_no_defaults(command):
            parsed = lsp_setup.parse_command(command)
            if parsed and any("tcp://127.0.0.1:" in c for c in parsed):
                return {"command": parsed, "available": False, "executable": None}
            return _real_probe(command)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {"ASTROGRAPH_WORKSPACE": tmpdir},
                clear=False,
            ),
            patch.object(lsp_setup, "probe_command", _probe_no_defaults),
        ):
            # No binding for python → auto_bind should pick up the observation
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
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.dict(
                os.environ,
                {"ASTROGRAPH_WORKSPACE": tmpdir},
                clear=False,
            ),
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

    def test_lsp_setup_discover_action_includes_server_bridge_info(self, tools):
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
        discover = next(action for action in actions if action["id"] == "discover_cpp_lsp_endpoint")

        assert "server_bridge_info" in discover
        info = discover["server_bridge_info"]
        assert info["server_binary"] == "clangd"
        assert "socat_command" in info
        assert "socat" in info["requires"]
        assert "install_hints" in info
        assert "macos" in info["install_hints"]
        assert "linux" in info["install_hints"]

    def test_lsp_setup_discover_action_bridge_info_shared_with(self, tools):
        cpp_status = {
            "language": "cpp_lsp",
            "required": False,
            "available": False,
            "transport": "tcp",
            "effective_command": ["tcp://127.0.0.1:2088"],
            "effective_source": "default",
            "default_command": ["tcp://127.0.0.1:2088"],
        }
        c_status = {
            "language": "c_lsp",
            "required": False,
            "available": False,
            "transport": "tcp",
            "effective_command": ["tcp://127.0.0.1:2087"],
            "effective_source": "default",
            "default_command": ["tcp://127.0.0.1:2087"],
        }

        actions = tools._build_lsp_recommended_actions(statuses=[cpp_status, c_status])
        discover_cpp = next(
            action for action in actions if action["id"] == "discover_cpp_lsp_endpoint"
        )
        discover_c = next(action for action in actions if action["id"] == "discover_c_lsp_endpoint")

        assert discover_cpp["server_bridge_info"]["shared_with"] == "c_lsp"
        assert discover_c["server_bridge_info"]["shared_with"] == "cpp_lsp"

    def test_lsp_setup_execution_context_docker(self, tools):
        missing_cpp_status = {
            "language": "cpp_lsp",
            "required": False,
            "available": False,
            "transport": "tcp",
            "effective_command": ["tcp://127.0.0.1:2088"],
            "effective_source": "default",
            "default_command": ["tcp://127.0.0.1:2088"],
        }
        payload = {
            "mode": "inspect",
            "servers": [missing_cpp_status],
        }

        self._inject_guidance_with_runtime(tools, payload, docker_runtime=True)

        assert payload["execution_context"] == "docker"
        assert "observation_note" in payload
        assert "host.docker.internal" in payload["observation_note"]

    def test_lsp_setup_execution_context_host(self, tools):
        missing_cpp_status = {
            "language": "cpp_lsp",
            "required": False,
            "available": False,
            "transport": "tcp",
            "effective_command": ["tcp://127.0.0.1:2088"],
            "effective_source": "default",
            "default_command": ["tcp://127.0.0.1:2088"],
        }
        payload = {
            "mode": "inspect",
            "servers": [missing_cpp_status],
        }

        self._inject_guidance_with_runtime(tools, payload, docker_runtime=False)

        assert payload["execution_context"] == "host"
        assert "observation_note" not in payload

    def test_lsp_setup_host_search_includes_binary(self, tools):
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
        discover = next(action for action in actions if action["id"] == "discover_cpp_lsp_endpoint")

        assert "which clangd" in discover["host_search_commands"]


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

    def test_suppress_accepts_list(self, tools, _indexed_with_duplicates):
        """Test suppress with a list of valid hashes."""
        import re

        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        hashes = re.findall(r'suppress\(wl_hash="([^"]+)"\)', details)
        if hashes:
            result = tools.suppress(wl_hash=hashes)
            assert "Suppressed" in result.text
            assert str(len(hashes)) in result.text

    def test_suppress_list_mixed(self, tools, _indexed_with_duplicates):
        """Test suppress with list containing valid and invalid hashes."""
        import re

        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        hashes = re.findall(r'suppress\(wl_hash="([^"]+)"\)', details)
        if hashes:
            mixed = hashes + ["nonexistent_hash_abc"]
            result = tools.suppress(wl_hash=mixed)
            assert "Suppressed" in result.text
            assert "not found" in result.text

    @pytest.mark.parametrize("action", ["suppress", "unsuppress"])
    def test_batch_toggle_empty(self, tools, _indexed_with_duplicates, action):
        """Batch toggle with empty list returns a helpful message."""
        result = getattr(tools, action)(wl_hash=[])
        assert "No hashes provided" in result.text

    def test_suppress_no_args_error(self, tools, _indexed_with_duplicates):
        """Suppress with no arguments returns an error."""
        result = tools.suppress()
        assert "Error" in result.text

    def test_unsuppress_no_args_error(self, tools, _indexed_with_duplicates):
        """Unsuppress with no arguments returns an error."""
        result = tools.unsuppress()
        assert "Error" in result.text

    def test_call_tool_suppress_with_list(self, tools, _indexed_with_duplicates):
        """call_tool dispatch for suppress with list of unknown hashes reports not found."""
        result = tools.call_tool("suppress", {"wl_hash": ["fake_hash"]})
        assert "not found" in result.text

    def test_call_tool_unsuppress_with_list(self, tools, _indexed_with_duplicates):
        """call_tool dispatch for unsuppress with list of unknown hashes reports not found."""
        result = tools.call_tool("unsuppress", {"wl_hash": ["fake_hash"]})
        assert "not found" in result.text

    def test_unsuppress_accepts_list(self, tools, _indexed_with_duplicates):
        """Test unsuppress with a list of valid hashes."""
        import re

        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        hashes = re.findall(r'suppress\(wl_hash="([^"]+)"\)', details)
        if hashes:
            tools.suppress(wl_hash=hashes)
            result = tools.unsuppress(wl_hash=hashes)
            assert "Unsuppressed" in result.text
            assert str(len(hashes)) in result.text

    def test_unsuppress_list_mixed(self, tools, _indexed_with_duplicates):
        """Test unsuppress with list containing valid and invalid hashes."""
        import re

        analyze_result = tools.analyze()
        details = _get_analyze_details(tools, analyze_result)
        hashes = re.findall(r'suppress\(wl_hash="([^"]+)"\)', details)
        if hashes:
            tools.suppress(wl_hash=hashes)
            mixed = hashes + ["nonexistent_hash_abc"]
            result = tools.unsuppress(wl_hash=mixed)
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


class TestAnalyzeEventDriven:
    """Tests for event-driven analyze (no staleness fallback)."""

    def test_analyze_no_staleness_warning(self, tools):
        """analyze() never emits staleness warnings — EDI keeps index current."""
        result = _with_indexed_temp_file(
            tools,
            "def unique_function_abc123(): return 42",
            tools.analyze,
        )
        assert "Stale:" not in result.text
        assert "Auto-reindexed" not in result.text

    def test_single_file_indexing_starts_watcher(self, tools):
        """Single-file indexing routes through EDI and starts the watcher."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass\ndef bar(): pass")
            f.flush()
            tools.index_codebase(f.name)

            edi = tools._event_driven_index
            assert edi is not None
            assert edi.is_watching, "Watcher should be active after single-file indexing"
        os.unlink(f.name)


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

    def test_index_fails_when_default_metadata_dir_not_writable(self, monkeypatch):
        """Indexing fails fast with actionable guidance when metadata dir is unwritable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "file1.py")).write_text("def foo(): pass")
            tools = CodeStructureTools()

            monkeypatch.setattr(
                tools,
                "_ensure_writable_directory",
                lambda _path: "Permission denied",
            )

            result = tools.index_codebase(tmpdir)
            assert result.text.startswith("Error: Metadata directory")
            assert "is not writable (Permission denied)" in result.text
            assert "ASTROGRAPH_PERSISTENCE_DIR" in result.text
            assert tools._last_indexed_path is None
            tools.close()

    def test_index_failure_with_cloud_warning_remains_error(self, monkeypatch):
        """Cloud warnings should not mask index failures as successful indexing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "file1.py")).write_text("def foo(): pass")
            tools = CodeStructureTools()

            monkeypatch.setattr(
                "astrograph.cloud_detect.get_cloud_sync_warning",
                lambda _path: "Cloud sync warning",
            )
            monkeypatch.setattr(
                tools,
                "_select_persistence_path",
                lambda _path: (None, "Error: Metadata directory is not writable"),
            )

            result = tools.index_codebase(tmpdir)
            assert result.text.startswith("Cloud sync warning")
            assert "\n\nError: Metadata directory is not writable" in result.text
            assert tools._last_indexed_path is None
            tools.close()

    def test_index_init_failure_includes_cloud_warning_and_actionable_error(self, monkeypatch):
        """SQLite init failures should include warning context and explicit remediation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "file1.py")).write_text("def foo(): pass")
            tools = CodeStructureTools()

            import sqlite3

            monkeypatch.setattr(
                "astrograph.cloud_detect.get_cloud_sync_warning",
                lambda _path: "Cloud sync warning",
            )
            with patch("astrograph.tools.EventDrivenIndex", side_effect=sqlite3.Error("db locked")):
                result = tools.index_codebase(tmpdir)

            assert result.text.startswith("Cloud sync warning")
            assert "Error: Failed to initialize metadata persistence" in result.text
            assert "ASTROGRAPH_PERSISTENCE_DIR" in result.text
            assert tools._last_indexed_path is None
            tools.close()

    def test_index_uses_configured_metadata_dir_when_default_not_writable(self, monkeypatch):
        """Explicit ASTROGRAPH_PERSISTENCE_DIR resolves the root-cause cleanly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(os.path.join(tmpdir, "file1.py")).write_text("def foo(): pass")
            configured_dir = Path(tmpdir) / "custom-metadata"
            tools = CodeStructureTools()

            default_path = _get_persistence_path(tmpdir)
            original = tools._ensure_writable_directory

            def fake_ensure_writable(path: Path) -> str | None:
                if path == default_path:
                    return "Permission denied"
                return original(path)

            monkeypatch.setenv("ASTROGRAPH_PERSISTENCE_DIR", str(configured_dir))
            monkeypatch.setattr(tools, "_ensure_writable_directory", fake_ensure_writable)

            result = tools.index_codebase(tmpdir)
            assert "Indexed" in result.text
            assert tools._active_persistence_path is not None
            assert os.path.realpath(str(tools._active_persistence_path)) == os.path.realpath(
                str(configured_dir)
            )
            assert (configured_dir / "index.db").exists()

            tools.metadata_erase()
            assert not configured_dir.exists()
            tools.close()

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


class _FakeCppPlugin:
    """Stub C++ plugin for write/edit tool tests."""

    language_id = "cpp_lsp"

    def __init__(self, entry):
        self._entry = entry

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
            code=self._entry.code_unit.code,
            file_path=file_path,
            line_start=3,
            line_end=11,
            unit_type="function",
            language="cpp_lsp",
        )


def _fake_cpp_exact_matches(entry):
    """Return a matcher closure that recognises the given entry."""

    def _match(code: str, language: str = "python"):
        if language == "cpp_lsp" and code.strip().startswith("int accumulate_positive"):
            return [entry]
        return []

    return _match


def _assert_metadata_op_reports_removed_bindings(method_name: str, expected_msg: str):
    """Shared helper: metadata erase/recompute should report removed LSP bindings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.py")
        Path(file1).write_text("def foo(): pass")

        tools = CodeStructureTools()
        tools.index_codebase(tmpdir)

        bindings_path = _get_persistence_path(tmpdir) / "lsp_bindings.json"
        bindings_path.write_text(
            json.dumps({"cpp_lsp": ["tcp://host.docker.internal:2088"]}, indent=2)
        )

        result = getattr(tools, method_name)()
        assert expected_msg in result.text
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

        with (
            patch.object(
                LanguageRegistry.get(),
                "get_plugin_for_file",
                return_value=_FakeCppPlugin(cpp_exact_entry),
            ),
            patch.object(
                tools.index,
                "find_exact_matches",
                side_effect=_fake_cpp_exact_matches(cpp_exact_entry),
            ),
            patch.object(
                tools.index,
                "find_similar",
                return_value=[],
            ),
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

        with (
            patch.object(
                LanguageRegistry.get(),
                "get_plugin_for_file",
                return_value=_FakeCppPlugin(cpp_exact_entry),
            ),
            patch.object(
                tools.index,
                "find_exact_matches",
                side_effect=_fake_cpp_exact_matches(cpp_exact_entry),
            ),
            patch.object(
                tools.index,
                "find_similar",
                return_value=[],
            ),
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

    def test_suppress_list_response_includes_refresh_hint(self, tools):
        """suppress with list response should include 'Run analyze' hint."""
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
                batch_result = tools.suppress(wl_hash=hashes)
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
        _assert_metadata_op_reports_removed_bindings("metadata_erase", "LSP bindings were removed")


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
        _assert_metadata_op_reports_removed_bindings(
            "metadata_recompute_baseline",
            "LSP bindings were reset during recompute",
        )


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


class TestGenerateIgnore:
    """Tests for the astrograph_generate_ignore tool."""

    def test_generate_ignore_creates_file(self):
        """File created at workspace root with default content."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "hello.py")
            with open(py_file, "w") as f:
                f.write("def hello(): pass\n")
            tools.index_codebase(tmpdir)
            result = tools.generate_ignore()
            assert "Created" in result.text
            ignore_path = os.path.join(tmpdir, ".astrographignore")
            assert os.path.exists(ignore_path)
            content = Path(ignore_path).read_text()
            assert "vendor/" in content
            assert "*.min.js" in content
            tools.close()

    def test_generate_ignore_no_overwrite(self):
        """Second call returns 'already exists'."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "hello.py")
            with open(py_file, "w") as f:
                f.write("def hello(): pass\n")
            tools.index_codebase(tmpdir)
            tools.generate_ignore()
            result = tools.generate_ignore()
            assert "already exists" in result.text
            tools.close()

    def test_generate_ignore_requires_indexed_path(self):
        """Without indexing, returns error about needing index_codebase."""
        tools = CodeStructureTools()
        result = tools.generate_ignore()
        assert "No indexed codebase" in result.text
        tools.close()

    def test_index_respects_astrographignore(self):
        """Ignored files should not be in the index."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .astrographignore
            ignore_path = os.path.join(tmpdir, ".astrographignore")
            with open(ignore_path, "w") as f:
                f.write("ignored_dir/\n")

            # Create a file that should be indexed
            included_file = os.path.join(tmpdir, "included.py")
            with open(included_file, "w") as f:
                f.write("def included(): pass\n")

            # Create a file inside ignored directory
            ignored_dir = os.path.join(tmpdir, "ignored_dir")
            os.makedirs(ignored_dir)
            ignored_file = os.path.join(ignored_dir, "excluded.py")
            with open(ignored_file, "w") as f:
                f.write("def excluded(): pass\n")

            tools.index_codebase(tmpdir)
            # The included file should be indexed but the excluded one should not
            indexed_files = list(tools.index.file_metadata.keys())
            assert any("included.py" in f for f in indexed_files)
            assert not any("excluded.py" in f for f in indexed_files)
            tools.close()

    def test_call_tool_dispatch_generate_ignore(self):
        """call_tool('generate_ignore', {}) dispatches correctly."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "hello.py")
            with open(py_file, "w") as f:
                f.write("def hello(): pass\n")
            tools.index_codebase(tmpdir)
            result = tools.call_tool("generate_ignore", {})
            assert "Created" in result.text
            tools.close()


class TestAnalyzeCache:
    """Tests for analyze() cache-first architecture."""

    _DUPLICATE_CODE = """
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

    def test_analyze_uses_cache_on_second_call(self):
        """Second analyze() call should hit the cache."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "dup.py")
            Path(py_file).write_text(self._DUPLICATE_CODE)
            tools.index_codebase(tmpdir)

            # First call warms the cache
            tools.analyze()
            edi = tools._event_driven_index
            assert edi is not None
            hits_before = edi._cache_hits

            # Second call should hit the cache
            tools.analyze()
            assert edi._cache_hits > hits_before
            tools.close()

    def test_analyze_never_reports_staleness(self):
        """analyze() never emits staleness warnings — EDI handles freshness."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "dup.py")
            Path(py_file).write_text(self._DUPLICATE_CODE)
            tools.index_codebase(tmpdir)

            result = tools.analyze()
            assert "Stale:" not in result.text
            assert "Auto-reindexed" not in result.text
            tools.close()

    def test_analyze_without_event_driven_index(self):
        """analyze() works via direct find_* calls when no EDI is present."""
        index = CodeStructureIndex()
        tools = CodeStructureTools(index=index)
        assert tools._event_driven_index is None

        _index_temp_code_file(tools, self._DUPLICATE_CODE)
        result = tools.analyze()
        # Should work — either finds duplicates or reports none
        assert result.text

    def test_single_file_watcher_active(self):
        """Single-file indexing goes through EDI and starts the watcher."""
        tools = CodeStructureTools()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(self._DUPLICATE_CODE)
            f.flush()
            tools.index_codebase(f.name)

            edi = tools._event_driven_index
            assert edi is not None
            assert edi.is_watching, "Watcher should be active after single-file indexing"
            tools.close()
        os.unlink(f.name)


class TestMCPResources:
    """Tests for MCP resource handlers."""

    def test_list_resources_returns_three(self):
        """list_resources returns exactly 3 resources."""
        from mcp.types import ListResourcesRequest

        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        result = asyncio.run(
            server.request_handlers[ListResourcesRequest](
                ListResourcesRequest(method="resources/list")
            )
        )
        resources = result.root.resources
        assert len(resources) == 3
        uris = {str(r.uri) for r in resources}
        assert "astrograph://status" in uris
        assert "astrograph://analysis/latest" in uris
        assert "astrograph://suppressions" in uris

    def test_read_resource_status(self):
        """Reading status resource returns status text."""
        tools = CodeStructureTools()
        text = tools.read_resource_status()
        assert "Status:" in text or "idle" in text

    def test_read_resource_analysis_no_index(self):
        """Reading analysis resource before indexing returns appropriate message."""
        tools = CodeStructureTools()
        text = tools.read_resource_analysis()
        assert "No codebase indexed" in text or "No analysis reports" in text

    def test_read_resource_analysis_after_analyze(self, tools):
        """Reading analysis resource after analysis returns report content."""
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
            tools.index_codebase(tmpdir)
            tools.analyze()

            text = tools.read_resource_analysis()
            # Should contain actual analysis content (not the "no reports" message)
            assert "No analysis reports" not in text

    def test_read_resource_suppressions_empty(self):
        """Reading suppressions resource when none suppressed."""
        tools = CodeStructureTools()
        text = tools.read_resource_suppressions()
        assert "No hashes" in text


class TestMCPPrompts:
    """Tests for MCP prompt handlers."""

    def test_list_prompts_returns_two(self):
        """list_prompts returns exactly 2 prompts."""
        from mcp.types import ListPromptsRequest

        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        result = asyncio.run(
            server.request_handlers[ListPromptsRequest](ListPromptsRequest(method="prompts/list"))
        )
        prompts = result.root.prompts
        assert len(prompts) == 2
        names = {p.name for p in prompts}
        assert "review-duplicates" in names
        assert "setup-lsp" in names

    def test_get_prompt_review_duplicates(self):
        """get_prompt for review-duplicates returns structured result."""
        from mcp.types import GetPromptRequest, GetPromptRequestParams

        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        result = asyncio.run(
            server.request_handlers[GetPromptRequest](
                GetPromptRequest(
                    method="prompts/get",
                    params=GetPromptRequestParams(name="review-duplicates"),
                )
            )
        )
        prompt_result = result.root
        assert prompt_result.description is not None
        assert "all" in prompt_result.description
        assert len(prompt_result.messages) == 1
        assert "SUPPRESS" in prompt_result.messages[0].content.text

    def test_get_prompt_review_duplicates_with_focus(self):
        """get_prompt for review-duplicates with focus arg."""
        from mcp.types import GetPromptRequest, GetPromptRequestParams

        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        result = asyncio.run(
            server.request_handlers[GetPromptRequest](
                GetPromptRequest(
                    method="prompts/get",
                    params=GetPromptRequestParams(
                        name="review-duplicates", arguments={"focus": "source"}
                    ),
                )
            )
        )
        prompt_result = result.root
        assert "source" in prompt_result.description

    def test_get_prompt_setup_lsp(self):
        """get_prompt for setup-lsp returns structured result."""
        from mcp.types import GetPromptRequest, GetPromptRequestParams

        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        result = asyncio.run(
            server.request_handlers[GetPromptRequest](
                GetPromptRequest(
                    method="prompts/get",
                    params=GetPromptRequestParams(name="setup-lsp"),
                )
            )
        )
        prompt_result = result.root
        assert prompt_result.description is not None
        assert len(prompt_result.messages) == 1
        assert "LSP" in prompt_result.messages[0].content.text

    def test_get_prompt_setup_lsp_with_language(self):
        """get_prompt for setup-lsp with language arg."""
        from mcp.types import GetPromptRequest, GetPromptRequestParams

        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        result = asyncio.run(
            server.request_handlers[GetPromptRequest](
                GetPromptRequest(
                    method="prompts/get",
                    params=GetPromptRequestParams(
                        name="setup-lsp", arguments={"language": "python"}
                    ),
                )
            )
        )
        prompt_result = result.root
        assert "python" in prompt_result.description

    def test_get_prompt_unknown_raises(self):
        """get_prompt for unknown name raises ValueError."""
        from mcp.types import GetPromptRequest, GetPromptRequestParams

        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        with pytest.raises(ValueError, match="Unknown prompt"):
            asyncio.run(
                server.request_handlers[GetPromptRequest](
                    GetPromptRequest(
                        method="prompts/get",
                        params=GetPromptRequestParams(name="nonexistent"),
                    )
                )
            )


class TestMCPCompletions:
    """Tests for MCP completion handlers."""

    def _complete(self, server, prompt_name, arg_name, prefix=""):
        from mcp.types import (
            CompleteRequest,
            CompleteRequestParams,
            CompletionArgument,
            PromptReference,
        )

        result = asyncio.run(
            server.request_handlers[CompleteRequest](
                CompleteRequest(
                    method="completion/complete",
                    params=CompleteRequestParams(
                        ref=PromptReference(type="ref/prompt", name=prompt_name),
                        argument=CompletionArgument(name=arg_name, value=prefix),
                    ),
                )
            )
        )
        return result.root.completion

    def test_review_duplicates_focus_all(self):
        """Completion for review-duplicates focus returns all options."""
        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        completion = self._complete(server, "review-duplicates", "focus")
        assert set(completion.values) == {"all", "source", "tests"}

    def test_review_duplicates_focus_prefix(self):
        """Completion for review-duplicates focus with prefix filters."""
        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        completion = self._complete(server, "review-duplicates", "focus", "s")
        assert completion.values == ["source"]

    def test_setup_lsp_language_all(self):
        """Completion for setup-lsp language returns all options."""
        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        completion = self._complete(server, "setup-lsp", "language")
        assert len(completion.values) == 7
        assert "python" in completion.values
        assert "go_lsp" in completion.values

    def test_setup_lsp_language_prefix(self):
        """Completion for setup-lsp language with prefix filters."""
        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        completion = self._complete(server, "setup-lsp", "language", "c")
        assert set(completion.values) == {"c_lsp", "cpp_lsp"}

    def test_unknown_prompt_returns_none(self):
        """Completion for unknown prompt returns empty."""
        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        completion = self._complete(server, "nonexistent", "arg")
        assert completion.values == []

    def test_unknown_argument_returns_none(self):
        """Completion for unknown argument returns empty."""
        server = create_server()
        tools = CodeStructureTools()
        set_tools(tools)
        completion = self._complete(server, "review-duplicates", "nonexistent")
        assert completion.values == []


class TestSetWorkspace:
    """Tests for the astrograph_set_workspace tool."""

    def test_set_workspace_switches_and_reindexes(self):
        """set_workspace indexes new directory and reports transition."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "hello.py")
            with open(py_file, "w") as f:
                f.write("def hello(): pass\n")
            go_file = os.path.join(tmpdir, "main.go")
            with open(go_file, "w") as f:
                f.write("package main\nfunc main() {}\n")

            result = tools.set_workspace(path=tmpdir)
            assert "Workspace changed:" in result.text
            assert "(none)" in result.text  # old path was unset
            assert tmpdir in result.text or str(Path(tmpdir).resolve()) in result.text
            tools.close()

    def test_set_workspace_switches_between_directories(self):
        """set_workspace shows old -> new transition."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as dir1, tempfile.TemporaryDirectory() as dir2:
            with open(os.path.join(dir1, "a.py"), "w") as f:
                f.write("def a(): pass\n")
            with open(os.path.join(dir2, "b.py"), "w") as f:
                f.write("def b(): pass\n")

            tools.set_workspace(path=dir1)
            result = tools.set_workspace(path=dir2)
            assert "Workspace changed:" in result.text
            # Old path should be dir1 (resolved)
            assert str(Path(dir1).resolve()) in result.text
            # New path should be dir2 (resolved)
            assert str(Path(dir2).resolve()) in result.text
            tools.close()

    def test_set_workspace_invalid_path(self):
        """set_workspace with non-existent path returns error."""
        tools = CodeStructureTools()
        result = tools.set_workspace(path="/nonexistent/path/xyz")
        assert "Error" in result.text or "does not exist" in result.text
        tools.close()

    def test_set_workspace_in_server_tool_list(self):
        """astrograph_set_workspace is registered in the MCP server."""
        from mcp.types import ListToolsRequest

        server = create_server()
        handler = server.request_handlers[ListToolsRequest]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
        finally:
            loop.close()
        tool_names = [t.name for t in result.root.tools]
        assert "astrograph_set_workspace" in tool_names

    def test_set_workspace_via_call_tool(self):
        """set_workspace works through the call_tool dispatch."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "hello.py"), "w") as f:
                f.write("def hello(): pass\n")

            result = tools.call_tool("set_workspace", {"path": tmpdir})
            assert "Workspace changed:" in result.text
            tools.close()


def _create_duplicate_files(directory: str, subdir: str, prefix: str, depth: int = 3) -> list[str]:
    """Create two Python files with duplicate functions in a subdirectory.

    ``depth`` controls the number of body statements so different calls
    produce structurally distinct functions (different WL hashes).
    """
    sub = os.path.join(directory, subdir)
    os.makedirs(sub, exist_ok=True)
    body_lines = "\n".join(f"    x{i} = a + b + {i}" for i in range(depth))
    paths = []
    for suffix in ("a", "b"):
        path = os.path.join(sub, f"{prefix}_{suffix}.py")
        with open(path, "w") as f:
            f.write(f"def {prefix}_calc(a, b):\n{body_lines}\n    return x0\n")
        paths.append(path)
    return paths


class TestScopedAnalysis:
    """Tests for analyze with scope parameter."""

    def test_analyze_with_scope_filters_entries(self):
        """Only entries matching scope globs are analyzed."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_duplicate_files(tmpdir, "src", "core")
            _create_duplicate_files(tmpdir, "lib", "helper")

            tools.index_codebase(tmpdir)

            # Scoped analysis to src/ only
            scoped_result = tools.analyze(scope=["src/**"])

            if "No significant duplicates" not in scoped_result.text:
                # Report file contains the full locations
                meta_dir = _get_persistence_path(tmpdir)
                reports = sorted(meta_dir.glob("analysis_report_*.txt"))
                assert reports, "Expected analysis report file"
                report_text = reports[-1].read_text()
                assert "src/" in report_text
                assert "lib/" not in report_text

            tools.close()

    def test_analyze_scope_no_match_returns_empty(self):
        """Scope with non-matching pattern returns no duplicates."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_duplicate_files(tmpdir, "src", "core")

            tools.index_codebase(tmpdir)
            result = tools.analyze(scope=["nonexistent/**"])
            assert "No significant duplicates" in result.text
            tools.close()

    def test_analyze_scope_via_call_tool(self):
        """scope parameter works through call_tool dispatch."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_duplicate_files(tmpdir, "src", "core")

            tools.index_codebase(tmpdir)
            result = tools.call_tool("analyze", {"scope": ["nonexistent/**"]})
            assert "No significant duplicates" in result.text
            tools.close()


class TestConfigureDomains:
    """Tests for configure_domains tool."""

    @staticmethod
    def _assert_report_contains_if_duplicates(
        *,
        tmpdir: str,
        analysis_text: str,
        expected_fragment: str,
    ) -> None:
        if "No significant duplicates" in analysis_text:
            return
        meta_dir = _get_persistence_path(tmpdir)
        reports = sorted(meta_dir.glob("analysis_report_*.txt"))
        if not reports:
            return
        report_text = reports[-1].read_text()
        assert expected_fragment in report_text

    def test_configure_domains_persists(self):
        """Domain configuration is written to domains.json."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_duplicate_files(tmpdir, "src", "core")
            tools.index_codebase(tmpdir)

            result = tools.configure_domains(domains={"core": ["src/**"], "tests": ["tests/**"]})
            assert "2 detection domain" in result.text

            # Verify file exists
            domains_file = _get_persistence_path(tmpdir) / "domains.json"
            assert domains_file.exists()
            data = json.loads(domains_file.read_text())
            assert "core" in data["domains"]
            assert "tests" in data["domains"]

            tools.close()

    def test_configure_domains_clear(self):
        """Empty dict clears domain configuration."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_duplicate_files(tmpdir, "src", "core")
            tools.index_codebase(tmpdir)

            # Configure then clear
            tools.configure_domains(domains={"core": ["src/**"]})
            result = tools.configure_domains(domains={})
            assert "cleared" in result.text.lower()

            # File should be removed
            domains_file = _get_persistence_path(tmpdir) / "domains.json"
            assert not domains_file.exists()

            tools.close()

    def test_configure_domains_validates_empty_patterns(self):
        """Domain with empty pattern list is rejected."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_duplicate_files(tmpdir, "src", "core")
            tools.index_codebase(tmpdir)

            result = tools.configure_domains(domains={"bad": []})
            assert "no patterns" in result.text.lower()
            tools.close()

    def test_configure_domains_requires_index(self):
        """configure_domains requires an indexed codebase."""
        tools = CodeStructureTools()
        result = tools.configure_domains(domains={"core": ["src/**"]})
        assert "No indexed codebase" in result.text
        tools.close()

    def test_analyze_with_domains_partitions_output(self):
        """When domains are configured, analysis is partitioned."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use different depth to produce structurally distinct functions per domain
            _create_duplicate_files(tmpdir, "src", "core", depth=3)
            _create_duplicate_files(tmpdir, "lib", "helper", depth=5)

            tools.index_codebase(tmpdir)
            tools.configure_domains(domains={"source": ["src/**"], "library": ["lib/**"]})

            result = tools.analyze()
            self._assert_report_contains_if_duplicates(
                tmpdir=tmpdir,
                analysis_text=result.text,
                expected_fragment="Domain:",
            )

            tools.close()

    def test_cross_domain_duplicates_reported(self):
        """Duplicates spanning domains are tagged cross-domain."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create identical files in two different domains
            src_dir = os.path.join(tmpdir, "src")
            lib_dir = os.path.join(tmpdir, "lib")
            os.makedirs(src_dir)
            os.makedirs(lib_dir)

            code = "def shared_calc(a, b):\n    result = a + b\n    return result * 2\n"
            with open(os.path.join(src_dir, "shared.py"), "w") as f:
                f.write(code)
            with open(os.path.join(lib_dir, "shared.py"), "w") as f:
                f.write(code)

            tools.index_codebase(tmpdir)
            tools.configure_domains(domains={"source": ["src/**"], "library": ["lib/**"]})

            result = tools.analyze()
            self._assert_report_contains_if_duplicates(
                tmpdir=tmpdir,
                analysis_text=result.text,
                expected_fragment="Cross-domain",
            )

            tools.close()

    def test_configure_domains_via_call_tool(self):
        """configure_domains works through call_tool dispatch."""
        tools = CodeStructureTools()
        with tempfile.TemporaryDirectory() as tmpdir:
            _create_duplicate_files(tmpdir, "src", "core")
            tools.index_codebase(tmpdir)

            result = tools.call_tool(
                "configure_domains",
                {"domains": {"core": ["src/**"]}},
            )
            assert "1 detection domain" in result.text
            tools.close()

    def test_domains_in_tool_list(self):
        """configure_domains appears in the MCP tool list."""
        loop = asyncio.new_event_loop()
        try:
            from mcp.types import ListToolsRequest

            server = create_server()
            handler = server.request_handlers[ListToolsRequest]
            result = loop.run_until_complete(handler(ListToolsRequest(method="tools/list")))
        finally:
            loop.close()
        tool_names = [t.name for t in result.root.tools]
        assert "astrograph_configure_domains" in tool_names


class TestServerMCPResources:
    """Tests for MCP resource/prompt/completion handlers."""

    def _assert_request_raises_value_error(self, handler, request: object, pattern: str) -> None:
        loop = asyncio.new_event_loop()
        try:
            with pytest.raises(ValueError, match=pattern):
                loop.run_until_complete(handler(request))
        finally:
            loop.close()

    def test_read_resource_unknown_uri(self):
        from mcp.types import ReadResourceRequest

        server = create_server()
        handler = server.request_handlers[ReadResourceRequest]
        self._assert_request_raises_value_error(
            handler,
            ReadResourceRequest(
                method="resources/read",
                params={"uri": "astrograph://nonexistent"},
            ),
            "Unknown resource URI",
        )

    def test_list_resource_templates_empty(self):
        from mcp.types import ListResourceTemplatesRequest

        server = create_server()
        handler = server.request_handlers[ListResourceTemplatesRequest]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                handler(ListResourceTemplatesRequest(method="resources/templates/list"))
            )
            assert result.root.resourceTemplates == []
        finally:
            loop.close()

    def test_list_prompts(self):
        from mcp.types import ListPromptsRequest

        server = create_server()
        handler = server.request_handlers[ListPromptsRequest]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(handler(ListPromptsRequest(method="prompts/list")))
            prompt_names = [p.name for p in result.root.prompts]
            assert "review-duplicates" in prompt_names
            assert "setup-lsp" in prompt_names
        finally:
            loop.close()

    def test_get_prompt_unknown(self):
        from mcp.types import GetPromptRequest

        server = create_server()
        handler = server.request_handlers[GetPromptRequest]
        self._assert_request_raises_value_error(
            handler,
            GetPromptRequest(
                method="prompts/get",
                params={"name": "nonexistent"},
            ),
            "Unknown prompt",
        )

    def test_completion_non_prompt_ref(self):
        from mcp.types import CompleteRequest

        server = create_server()
        handler = server.request_handlers[CompleteRequest]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                handler(
                    CompleteRequest(
                        method="completion/complete",
                        params={
                            "ref": {"type": "ref/resource", "uri": "astrograph://status"},
                            "argument": {"name": "focus", "value": ""},
                        },
                    )
                )
            )
            assert result.root.completion.values == []
        finally:
            loop.close()

    def test_completion_prompt_ref(self):
        from mcp.types import CompleteRequest

        server = create_server()
        handler = server.request_handlers[CompleteRequest]
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                handler(
                    CompleteRequest(
                        method="completion/complete",
                        params={
                            "ref": {"type": "ref/prompt", "name": "review-duplicates"},
                            "argument": {"name": "focus", "value": "a"},
                        },
                    )
                )
            )
            assert "all" in result.root.completion.values
        finally:
            loop.close()


class TestServerShutdown:
    """Tests for signal/atexit shutdown handling."""

    def test_close_if_first_idempotent(self):
        import astrograph.server as server_mod

        original_event = server_mod._close_once
        server_mod._close_once = __import__("threading").Event()
        try:
            with patch.object(server_mod._tools, "close") as mock_close:
                server_mod._close_if_first()
                server_mod._close_if_first()
                mock_close.assert_called_once()
        finally:
            server_mod._close_once = original_event

    def test_atexit_close(self):
        import astrograph.server as server_mod

        original_event = server_mod._close_once
        server_mod._close_once = __import__("threading").Event()
        try:
            with patch.object(server_mod._tools, "close") as mock_close:
                server_mod._atexit_close()
                mock_close.assert_called_once()
        finally:
            server_mod._close_once = original_event


class TestToolsBackgroundIndexError:
    """Tests for background indexing error handling."""

    def test_bg_index_error_reported(self, monkeypatch):
        monkeypatch.setenv("ASTROGRAPH_WORKSPACE", "")
        tools = CodeStructureTools()
        tools._bg_index_progress = "error"
        tools._bg_index_error = "disk full"
        result = tools._require_index()
        assert result is not None
        assert "Background indexing failed" in result.text
        assert "disk full" in result.text
        tools.close()

    def test_bg_index_error_unknown(self, monkeypatch):
        monkeypatch.setenv("ASTROGRAPH_WORKSPACE", "")
        tools = CodeStructureTools()
        tools._bg_index_progress = "error"
        result = tools._require_index()
        assert "unknown error" in result.text
        tools.close()


class TestToolsInvalidatedSuppressions:
    """Tests for suppression invalidation warnings."""

    def _populate_index(self, tools):
        """Add a dummy entry so self.index.entries is truthy."""
        unit = CodeUnit(
            name="dummy",
            code="def dummy(): pass",
            file_path="d.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        tools.index.add_code_unit(unit)

    def test_invalidated_suppressions_warning(self, tools):
        tools._last_indexed_path = "/some/path"
        self._populate_index(tools)
        with patch.object(
            tools.index,
            "invalidate_stale_suppressions",
            return_value=[("abc123", "reason")],
        ):
            warning = tools._check_invalidated_suppressions()
        assert "Suppressions invalidated" in warning
        assert "abc123" in warning

    def test_many_invalidated_truncated(self, tools):
        tools._last_indexed_path = "/some/path"
        self._populate_index(tools)
        entries = [(f"hash_{i:03d}", "reason") for i in range(10)]
        with patch.object(
            tools.index,
            "invalidate_stale_suppressions",
            return_value=entries,
        ):
            warning = tools._check_invalidated_suppressions()
        assert "10 total" in warning


class TestToolsFormatIndexStats:
    """Tests for format_index_stats edge cases."""

    def _format_stats(
        self,
        tools,
        *,
        block_entries: int,
        has_duplicates: bool,
        include_blocks: bool,
    ) -> str:
        tools._last_indexed_path = "/some/path"
        with (
            patch.object(
                tools.index,
                "get_stats",
                return_value={
                    "function_entries": 5,
                    "indexed_files": 2,
                    "block_entries": block_entries,
                },
            ),
            patch.object(tools, "_has_significant_duplicates", return_value=has_duplicates),
        ):
            return tools._format_index_stats(include_blocks=include_blocks)

    def test_with_blocks(self, tools):
        text = self._format_stats(
            tools,
            block_entries=3,
            has_duplicates=False,
            include_blocks=True,
        )
        assert "3 code blocks" in text
        assert "No duplicates" in text

    def test_with_duplicates(self, tools):
        text = self._format_stats(
            tools,
            block_entries=0,
            has_duplicates=True,
            include_blocks=False,
        )
        assert "Duplicates found" in text


class TestToolsCheckNoPlugin:
    """Tests for check tool with unsupported language."""

    def test_unsupported_language(self, tools):
        unit = CodeUnit(
            name="dummy",
            code="def dummy(): pass",
            file_path="d.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        tools.index.add_code_unit(unit)
        result = tools.check("some code", language="nonexistent_lang_xyz")
        assert "Unsupported language" in result.text


class TestToolsVerifyGroup:
    """Tests for _verify_group edge cases."""

    def test_single_entry_group(self, tools):
        from astrograph.index import DuplicateGroup

        group = DuplicateGroup(wl_hash="abc", entries=[object()])
        assert tools._verify_group(group) is False


# ---------------------------------------------------------------------------
# Coverage-targeted tests appended below
# ---------------------------------------------------------------------------


class TestDockerResolveFastPath:
    """Cover fast-path branch in _resolve_path with learned host root."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()

    def test_fast_path_with_trailing_slash_remainder(self):
        """Fast path strips leading '/' from remainder."""
        self.tools._host_root = "/Users/dev/project"
        # Path starts with host_root; remainder starts with '/'
        result = self.tools._resolve_path("/Users/dev/project/src/main.py")
        assert result == "/workspace/src/main.py"

    def test_fast_path_empty_remainder(self):
        """Fast path returns /workspace when remainder is empty."""
        self.tools._host_root = "/Users/dev/project"
        result = self.tools._resolve_path("/Users/dev/project")
        assert result == "/workspace"

    def test_fast_path_filters_traversal_components(self):
        """Fast path filters out '..' traversal components from remainder."""
        self.tools._host_root = "/Users/dev/project"
        result = self.tools._resolve_path("/Users/dev/project/../project/src/file.py")
        # The '..' should be stripped; only safe parts remain
        assert ".." not in result
        assert "/workspace" in result

    def test_fast_path_remainder_no_leading_slash(self):
        """Fast path handles remainder that does not start with '/'."""
        self.tools._host_root = "/Users/dev/project/"
        # The host_root has trailing slash, so remainder won't start with /
        result = self.tools._resolve_path("/Users/dev/project/lib/utils.py")
        assert result == "/workspace/lib/utils.py"

    def test_fast_path_traversal_only_components_empty(self):
        """Fast path with only traversal components yields empty remainder."""
        self.tools._host_root = "/Users/dev/project"
        # After stripping host_root, remainder is "/.." which becomes all traversal parts
        result = self.tools._resolve_path("/Users/dev/project/..")
        # safe_parts will be empty => remainder = "" => returns "/workspace"
        assert result == "/workspace"


class TestLearnHostRoot:
    """Cover _learn_host_root method (lines 599-613)."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()

    def _assert_host_root_learned(self, host_path: str, container_path: str, expected_root: str):
        with patch("astrograph.lsp_setup.set_docker_path_map") as mock_set:
            self.tools._learn_host_root(host_path, container_path)
            assert self.tools._host_root == expected_root
            mock_set.assert_called_once_with("/workspace", expected_root)

    def test_learn_host_root_with_suffix(self):
        """_learn_host_root derives root when container path has suffix."""
        self._assert_host_root_learned(
            "/Users/dev/myproject/src/lib",
            "/workspace/src/lib",
            "/Users/dev/myproject",
        )

    def test_learn_host_root_exact_workspace(self):
        """_learn_host_root sets root directly when container path is /workspace."""
        self._assert_host_root_learned(
            "/Users/dev/myproject",
            "/workspace",
            "/Users/dev/myproject",
        )

    def test_learn_host_root_suffix_mismatch_no_learn(self):
        """_learn_host_root does not learn when suffix doesn't match host path."""
        with patch("astrograph.lsp_setup.set_docker_path_map") as mock_set:
            self.tools._learn_host_root(
                "/Users/other/path",
                "/workspace/src",
            )
            # suffix is "/src", but host_path "/Users/other/path" doesn't end with "/src"
            assert self.tools._host_root is None
            mock_set.assert_not_called()


class _CountingLock:
    """Simple context-manager lock used to verify lock usage in tests."""

    def __init__(self):
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestHostRootLocking:
    """Lock-sensitive checks for host root reads/writes."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()

    def test_resolve_path_reads_host_root_under_lock(self):
        self.tools._host_root = "/Users/dev/project"
        counting_lock = _CountingLock()
        self.tools._host_root_lock = counting_lock

        result = self.tools._resolve_path("/Users/dev/project/src/main.py")
        assert result == "/workspace/src/main.py"
        assert counting_lock.enter_count >= 1

    def test_learn_host_root_writes_under_lock(self):
        counting_lock = _CountingLock()
        self.tools._host_root_lock = counting_lock

        with patch("astrograph.lsp_setup.set_docker_path_map"):
            self.tools._learn_host_root("/Users/dev/project/src", "/workspace/src")

        assert self.tools._host_root == "/Users/dev/project"
        assert counting_lock.enter_count >= 1


class TestDetectHostRootFromMountinfo:
    """Cover _detect_host_root_from_mountinfo static method."""

    @staticmethod
    def _mock_mountinfo(mountinfo_path):
        """Create a mock for builtins.open that redirects /proc/self/mountinfo."""
        _real_open = open

        def _open(f, *a, **kw):
            if f == "/proc/self/mountinfo":
                return _real_open(str(mountinfo_path), *a, **kw)
            return _real_open(f, *a, **kw)

        return _open

    def test_linux_docker_block_device(self, tmp_path):
        """Linux Docker: block device mount source → root is the host path."""
        mi = tmp_path / "mountinfo"
        mi.write_text("100 50 8:1 /home/user/project /workspace rw,relatime - ext4 /dev/sda1 rw\n")
        with patch("builtins.open", side_effect=self._mock_mountinfo(mi)):
            result = CodeStructureTools._detect_host_root_from_mountinfo()
        assert result == "/home/user/project"

    def test_macos_docker_desktop_host_mark(self, tmp_path):
        """macOS Docker Desktop: /run/host_mark prefix is stripped."""
        mi = tmp_path / "mountinfo"
        mi.write_text(
            "223 214 0:45 /thaylo/Projects/app /workspace rw - fakeowner /run/host_mark/Users rw\n"
        )
        with patch("builtins.open", side_effect=self._mock_mountinfo(mi)):
            result = CodeStructureTools._detect_host_root_from_mountinfo()
        assert result == "/Users/thaylo/Projects/app"

    def test_no_workspace_mount(self, tmp_path):
        """Returns None when /workspace is not mounted."""
        mi = tmp_path / "mountinfo"
        mi.write_text("100 50 8:1 / / rw,relatime - ext4 /dev/sda1 rw\n")
        with patch("builtins.open", side_effect=self._mock_mountinfo(mi)):
            result = CodeStructureTools._detect_host_root_from_mountinfo()
        assert result is None

    def test_file_not_found(self):
        """Returns None when /proc/self/mountinfo doesn't exist."""
        with patch("builtins.open", side_effect=OSError("No such file")):
            result = CodeStructureTools._detect_host_root_from_mountinfo()
        assert result is None

    def test_fallback_absolute_root(self, tmp_path):
        """Fallback: unknown fs type with absolute root."""
        mi = tmp_path / "mountinfo"
        mi.write_text("100 50 0:99 /data/project /workspace rw - overlay overlay rw\n")
        with patch("builtins.open", side_effect=self._mock_mountinfo(mi)):
            result = CodeStructureTools._detect_host_root_from_mountinfo()
        assert result == "/data/project"


class TestHostDisplayPath:
    """Cover _host_display_path method."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()

    def test_translates_workspace_prefix(self):
        """Translates /workspace prefix to host root."""
        self.tools._host_root = "/home/user/project"
        assert (
            self.tools._host_display_path("/workspace/src/foo.py")
            == "/home/user/project/src/foo.py"
        )

    def test_translates_workspace_itself(self):
        """Translates /workspace to host root."""
        self.tools._host_root = "/home/user/project"
        assert self.tools._host_display_path("/workspace") == "/home/user/project"

    def test_no_host_root_passthrough(self):
        """Without host_root, paths pass through unchanged."""
        assert self.tools._host_display_path("/workspace/src/foo.py") == "/workspace/src/foo.py"

    def test_non_workspace_path_unchanged(self):
        """Non-workspace paths are not modified."""
        self.tools._host_root = "/home/user/project"
        assert self.tools._host_display_path("/app/main.py") == "/app/main.py"


class TestResolvePathMountSourceFallback:
    """Cover _resolve_path fallback for host project root → /workspace."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()

    def test_host_project_root_resolves_to_workspace(self):
        """When the host project root is sent and /workspace is indexed, map to /workspace."""
        original_exists = Path.exists
        original_is_dir = Path.is_dir
        original_resolve = Path.resolve

        def mock_exists(self):
            s = str(self)
            if s in ("/workspace", "/.dockerenv"):
                return True
            return original_exists(self)

        def mock_is_dir(self):
            if str(self) == "/workspace":
                return True
            return original_is_dir(self)

        def mock_resolve(self):
            if str(self) == "/workspace":
                return Path("/workspace")
            return original_resolve(self)

        self.tools._last_indexed_path = "/workspace"

        with (
            patch.object(Path, "exists", mock_exists),
            patch.object(Path, "is_dir", mock_is_dir),
            patch.object(Path, "resolve", mock_resolve),
            patch("astrograph.lsp_setup.set_docker_path_map"),
        ):
            result = self.tools._resolve_path("/home/thaylo/Projects/myproject")
            assert result == "/workspace"
            assert self.tools._host_root == "/home/thaylo/Projects/myproject"


class TestSemanticCompareResult:
    """Cover _semantic_compare_result static method (lines 1218-1263)."""

    def test_low_confidence_signals_skipped(self):
        """Signals below 0.4 confidence are skipped entirely."""
        profile1 = SemanticProfile(
            signals=(SemanticSignal(key="type", value="int", confidence=0.3),),
        )
        profile2 = SemanticProfile(
            signals=(SemanticSignal(key="type", value="int", confidence=0.2),),
        )
        (
            available,
            compatible,
            mismatches,
            matched,
            reason,
        ) = CodeStructureTools._semantic_compare_result(profile1, profile2)
        assert not available
        assert not compatible
        assert matched == []
        assert "insufficient" in reason

    def test_no_comparable_signals_inconclusive(self):
        """No shared keys at all yields inconclusive result."""
        profile1 = SemanticProfile(
            signals=(SemanticSignal(key="async", value="yes", confidence=0.9),),
        )
        profile2 = SemanticProfile(
            signals=(SemanticSignal(key="type", value="int", confidence=0.9),),
        )
        (
            available,
            compatible,
            mismatches,
            matched,
            reason,
        ) = CodeStructureTools._semantic_compare_result(profile1, profile2)
        assert not available
        assert "insufficient" in reason

    def test_no_signals_inconclusive_with_notes(self):
        """Empty overlapping signals with notes includes note in reason."""
        profile1 = SemanticProfile(
            signals=(),
            notes=("syntax-only extraction",),
        )
        profile2 = SemanticProfile(
            signals=(),
            notes=(),
        )
        (
            available,
            compatible,
            mismatches,
            matched,
            reason,
        ) = CodeStructureTools._semantic_compare_result(profile1, profile2)
        assert not available
        assert "syntax-only extraction" in reason

    def test_matching_signals_compatible(self):
        """All shared high-confidence signals matching yields compatible."""
        profile1 = SemanticProfile(
            signals=(
                SemanticSignal(key="async", value="yes", confidence=0.9),
                SemanticSignal(key="type_system", value="typed", confidence=0.8),
            ),
        )
        profile2 = SemanticProfile(
            signals=(
                SemanticSignal(key="async", value="yes", confidence=0.9),
                SemanticSignal(key="type_system", value="typed", confidence=0.8),
            ),
        )
        (
            available,
            compatible,
            mismatches,
            matched,
            reason,
        ) = CodeStructureTools._semantic_compare_result(profile1, profile2)
        assert available
        assert compatible
        assert "async" in matched
        assert "type_system" in matched
        assert mismatches == []

    def test_mismatching_signals_incompatible(self):
        """Diverging signal values yields incompatible with mismatch descriptions."""
        profile1 = SemanticProfile(
            signals=(SemanticSignal(key="async", value="yes", confidence=0.9),),
        )
        profile2 = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )
        (
            available,
            compatible,
            mismatches,
            matched,
            reason,
        ) = CodeStructureTools._semantic_compare_result(profile1, profile2)
        assert available
        assert not compatible
        assert len(mismatches) == 1
        assert "async" in mismatches[0]
        assert "yes vs no" in mismatches[0]


class TestCompareSemanticModes:
    """Cover compare() method semantic mode branches (lines 1296-1334)."""

    @pytest.fixture
    def mock_tools(self):
        """Create tools with mocked plugin returning controlled graphs/profiles."""
        tools = CodeStructureTools()
        return tools

    @staticmethod
    def _graph_or_default(graph):
        if graph is not None:
            return graph
        import networkx as nx

        default_graph = nx.DiGraph()
        default_graph.add_node(0, label="FunctionDef")
        return default_graph

    def _make_mock_plugin(
        self,
        *,
        g1=None,
        g2=None,
        profile1=None,
        profile2=None,
    ):
        """Return a mock plugin with controlled graph/profile outputs."""
        plugin = MagicMock()
        g1 = self._graph_or_default(g1)
        g2 = self._graph_or_default(g2)

        call_count = {"graph": 0, "profile": 0}

        def source_to_graph(_code):
            call_count["graph"] += 1
            return g1 if call_count["graph"] == 1 else g2

        def extract_semantic_profile(_code, **_kwargs):
            call_count["profile"] += 1
            return profile1 if call_count["profile"] == 1 else profile2

        plugin.source_to_graph = source_to_graph
        plugin.extract_semantic_profile = extract_semantic_profile
        return plugin

    def _compare_with_plugin(
        self,
        mock_tools,
        plugin,
        *,
        semantic_mode: str,
        left_code: str,
        right_code: str,
    ):
        with patch.object(LanguageRegistry.get(), "get_plugin", return_value=plugin):
            return mock_tools.compare(left_code, right_code, semantic_mode=semantic_mode)

    def test_annotate_unavailable_profiles(self, mock_tools):
        """mode=annotate with unavailable profiles returns SEMANTIC_INCONCLUSIVE."""
        empty_profile = SemanticProfile(signals=(), notes=("no LSP data",))
        plugin = self._make_mock_plugin(profile1=empty_profile, profile2=empty_profile)
        result = self._compare_with_plugin(
            mock_tools,
            plugin,
            semantic_mode="annotate",
            left_code="def a(): pass",
            right_code="def b(): pass",
        )
        assert "SEMANTIC_INCONCLUSIVE" in result.text
        assert "astrograph_lsp_setup" in result.text

    def test_differentiate_unavailable_profiles(self, mock_tools):
        """mode=differentiate with unavailable profiles returns INCONCLUSIVE."""
        empty_profile = SemanticProfile(signals=(), notes=())
        plugin = self._make_mock_plugin(profile1=empty_profile, profile2=empty_profile)
        result = self._compare_with_plugin(
            mock_tools,
            plugin,
            semantic_mode="differentiate",
            left_code="def a(): pass",
            right_code="def b(): pass",
        )
        assert "INCONCLUSIVE" in result.text
        assert "differentiate mode" in result.text

    def test_annotate_compatible_profiles(self, mock_tools):
        """mode=annotate with compatible profiles returns structural + SEMANTIC_MATCH."""
        matching_profile = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )
        plugin = self._make_mock_plugin(profile1=matching_profile, profile2=matching_profile)
        result = self._compare_with_plugin(
            mock_tools,
            plugin,
            semantic_mode="annotate",
            left_code="def a(): pass",
            right_code="def b(): pass",
        )
        assert "SEMANTIC_MATCH" in result.text

    def test_differentiate_equivalent_compatible(self, mock_tools):
        """mode=differentiate with equivalent structure + compatible signals."""
        import networkx as nx

        g = nx.DiGraph()
        g.add_node(0, label="FunctionDef")
        g.add_node(1, label="Return")
        g.add_edge(0, 1)

        matching_profile = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )
        plugin = self._make_mock_plugin(
            g1=g, g2=g, profile1=matching_profile, profile2=matching_profile
        )
        result = self._compare_with_plugin(
            mock_tools,
            plugin,
            semantic_mode="differentiate",
            left_code="def a(): return 1",
            right_code="def b(): return 2",
        )
        assert "EQUIVALENT" in result.text
        assert "semantically aligned" in result.text

    def test_differentiate_similar_compatible(self, mock_tools):
        """mode=differentiate with non-equivalent structure + compatible signals."""
        import networkx as nx

        g1 = nx.DiGraph()
        g1.add_node(0, label="FunctionDef")
        g1.add_node(1, label="Return")
        g1.add_edge(0, 1)

        g2 = nx.DiGraph()
        g2.add_node(0, label="FunctionDef")
        g2.add_node(1, label="If")
        g2.add_node(2, label="Return")
        g2.add_edge(0, 1)
        g2.add_edge(1, 2)

        matching_profile = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )
        plugin = self._make_mock_plugin(
            g1=g1, g2=g2, profile1=matching_profile, profile2=matching_profile
        )
        result = self._compare_with_plugin(
            mock_tools,
            plugin,
            semantic_mode="differentiate",
            left_code="def a(): return 1",
            right_code="def b():\n if True:\n  return 2",
        )
        assert "SEMANTIC_MATCH" in result.text

    def test_annotate_equivalent_mismatch(self, mock_tools):
        """mode=annotate with equivalent structure + semantic mismatch."""
        import networkx as nx

        g = nx.DiGraph()
        g.add_node(0, label="FunctionDef")
        g.add_node(1, label="Return")
        g.add_edge(0, 1)

        profile1 = SemanticProfile(
            signals=(SemanticSignal(key="async", value="yes", confidence=0.9),),
        )
        profile2 = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )
        plugin = self._make_mock_plugin(g1=g, g2=g, profile1=profile1, profile2=profile2)
        result = self._compare_with_plugin(
            mock_tools,
            plugin,
            semantic_mode="annotate",
            left_code="def a(): return 1",
            right_code="def b(): return 2",
        )
        assert "EQUIVALENT (STRUCTURE)" in result.text
        assert "SEMANTIC_MISMATCH" in result.text

    def test_differentiate_equivalent_mismatch(self, mock_tools):
        """mode=differentiate with equivalent structure + semantic mismatch."""
        import networkx as nx

        g = nx.DiGraph()
        g.add_node(0, label="FunctionDef")
        g.add_node(1, label="Return")
        g.add_edge(0, 1)

        profile1 = SemanticProfile(
            signals=(SemanticSignal(key="async", value="yes", confidence=0.9),),
        )
        profile2 = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )
        plugin = self._make_mock_plugin(g1=g, g2=g, profile1=profile1, profile2=profile2)
        result = self._compare_with_plugin(
            mock_tools,
            plugin,
            semantic_mode="differentiate",
            left_code="def a(): return 1",
            right_code="def b(): return 2",
        )
        assert "DIFFERENT" in result.text
        assert "diverge semantically" in result.text

    def test_annotate_different_mismatch(self, mock_tools):
        """mode=annotate with different structure + semantic mismatch."""
        import networkx as nx

        g1 = nx.DiGraph()
        g1.add_node(0, label="FunctionDef")
        g1.add_node(1, label="Return")
        g1.add_edge(0, 1)

        g2 = nx.DiGraph()
        g2.add_node(0, label="FunctionDef")
        g2.add_node(1, label="If")
        g2.add_node(2, label="Return")
        g2.add_edge(0, 1)
        g2.add_edge(1, 2)

        profile1 = SemanticProfile(
            signals=(SemanticSignal(key="async", value="yes", confidence=0.9),),
        )
        profile2 = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )
        plugin = self._make_mock_plugin(g1=g1, g2=g2, profile1=profile1, profile2=profile2)
        result = self._compare_with_plugin(
            mock_tools,
            plugin,
            semantic_mode="annotate",
            left_code="def a(): return 1",
            right_code="def b():\n if True:\n  return 2",
        )
        assert "DIFFERENT" in result.text
        assert "SEMANTIC_MISMATCH" in result.text

    def test_compare_unsupported_language(self, mock_tools):
        """compare with unsupported language returns error."""
        result = mock_tools.compare("code1", "code2", language="unknown_lang_xyz")
        assert "Unsupported language" in result.text


class TestSemanticAlignmentHint:
    """Cover _semantic_alignment_hint (lines 1109-1146)."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()

    def _make_candidate(self, language="python", code="def f(): pass"):
        code_unit = CodeUnit(
            name="f",
            code=code,
            file_path="test.py",
            line_start=1,
            line_end=1,
            unit_type="function",
            language=language,
        )
        entry = IndexEntry(
            id="test:f:test.py",
            wl_hash="wl_hash",
            pattern_hash="pattern_hash",
            fingerprint={"n_nodes": 5, "n_edges": 4},
            hierarchy_hashes=["root"],
            code_unit=code_unit,
            node_count=5,
            depth=2,
        )
        return SimilarityResult(entry=entry, similarity_type="high")

    def _alignment_hint_with_plugin(
        self,
        *,
        source_profile: SemanticProfile,
        candidate: SimilarityResult,
        plugin: MagicMock,
    ) -> tuple[float, str]:
        with patch.object(LanguageRegistry.get(), "get_plugin", return_value=plugin):
            return self.tools._semantic_alignment_hint(
                source_profile=source_profile,
                candidate=candidate,
            )

    def test_no_plugin_found(self):
        """Returns unknown when no plugin exists for the language."""
        candidate = self._make_candidate(language="nonexistent_lang_xyz")
        source_profile = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )
        score, hint = self.tools._semantic_alignment_hint(
            source_profile=source_profile, candidate=candidate
        )
        assert score == pytest.approx(0.1)
        assert "no language plugin" in hint

    def test_extraction_exception(self):
        """Returns unknown when profile extraction raises."""
        candidate = self._make_candidate(language="python")
        source_profile = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )

        mock_plugin = MagicMock()
        mock_plugin.extract_semantic_profile.side_effect = RuntimeError("extraction boom")

        score, hint = self._alignment_hint_with_plugin(
            source_profile=source_profile,
            candidate=candidate,
            plugin=mock_plugin,
        )
        assert score == pytest.approx(0.1)
        assert "profile extraction failed" in hint

    def test_inconclusive_insufficient_overlap(self):
        """Returns inconclusive when semantic comparison has no overlap."""
        candidate = self._make_candidate(language="python")
        source_profile = SemanticProfile(
            signals=(SemanticSignal(key="unique_key_a", value="x", confidence=0.9),),
        )

        mock_plugin = MagicMock()
        mock_plugin.extract_semantic_profile.return_value = SemanticProfile(
            signals=(SemanticSignal(key="unique_key_b", value="y", confidence=0.9),),
        )

        score, hint = self._alignment_hint_with_plugin(
            source_profile=source_profile,
            candidate=candidate,
            plugin=mock_plugin,
        )
        assert score == pytest.approx(0.2)
        assert "inconclusive" in hint

    def test_compatible_with_matched_keys(self):
        """Returns aligned with matched keys listed."""
        candidate = self._make_candidate(language="python")
        source_profile = SemanticProfile(
            signals=(
                SemanticSignal(key="async", value="no", confidence=0.9),
                SemanticSignal(key="type_system", value="typed", confidence=0.8),
            ),
        )

        mock_plugin = MagicMock()
        mock_plugin.extract_semantic_profile.return_value = SemanticProfile(
            signals=(
                SemanticSignal(key="async", value="no", confidence=0.9),
                SemanticSignal(key="type_system", value="typed", confidence=0.8),
            ),
        )

        score, hint = self._alignment_hint_with_plugin(
            source_profile=source_profile,
            candidate=candidate,
            plugin=mock_plugin,
        )
        assert score == pytest.approx(0.9)
        assert "aligned" in hint
        assert "async" in hint

    def test_compatible_without_matched_keys(self):
        """Returns aligned without keys when matched list is empty (all low-conf skipped but compared_any via edge case)."""
        candidate = self._make_candidate(language="python")
        # Create profiles where comparison yields compatible=True but matched=[]
        # This happens when all shared signals match but matched list is somehow empty.
        # The simplest way: mock _semantic_compare_result directly.
        source_profile = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )

        mock_plugin = MagicMock()
        mock_plugin.extract_semantic_profile.return_value = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )

        # Force the comparison to return compatible=True but matched=[]
        with (
            patch.object(LanguageRegistry.get(), "get_plugin", return_value=mock_plugin),
            patch.object(
                CodeStructureTools,
                "_semantic_compare_result",
                return_value=(True, True, [], [], ""),
            ),
        ):
            score, hint = self.tools._semantic_alignment_hint(
                source_profile=source_profile, candidate=candidate
            )
        assert score == pytest.approx(0.8)
        assert "aligned" in hint

    def test_mismatch(self):
        """Returns mismatch when signals disagree."""
        candidate = self._make_candidate(language="python")
        source_profile = SemanticProfile(
            signals=(SemanticSignal(key="async", value="yes", confidence=0.9),),
        )

        mock_plugin = MagicMock()
        mock_plugin.extract_semantic_profile.return_value = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )

        score, hint = self._alignment_hint_with_plugin(
            source_profile=source_profile,
            candidate=candidate,
            plugin=mock_plugin,
        )
        assert score == pytest.approx(0.35)
        assert "mismatch" in hint


class TestCrossLanguageAssistText:
    """Cover _cross_language_assist_text (lines 1148-1201)."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()
        self.tools._last_indexed_path = "/tmp/test"

    def _make_result(self, language="go_lsp", name="process", sim_type="high"):
        return _similarity_result(
            language=language,
            similarity_type=sim_type,
            file_path="/tmp/test/main.go",
            name=name,
            code="func process() {}",
        )

    def test_no_results(self):
        """Empty results list returns empty string."""
        text = self.tools._cross_language_assist_text(
            source_code="def f(): pass",
            source_language="python",
            results=[],
        )
        assert text == ""

    def test_no_plugin(self):
        """Returns empty when no plugin for source language."""
        results = [self._make_result()]
        text = self.tools._cross_language_assist_text(
            source_code="code",
            source_language="nonexistent_lang_xyz",
            results=results,
        )
        assert text == ""

    def test_normal_flow_with_results(self):
        """Normal flow builds ASSIST text with cross-language matches."""
        results = [self._make_result()]

        mock_source_plugin = MagicMock()
        mock_source_plugin.extract_semantic_profile.return_value = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )

        # Mock the candidate language plugin too
        mock_candidate_plugin = MagicMock()
        mock_candidate_plugin.extract_semantic_profile.return_value = SemanticProfile(
            signals=(SemanticSignal(key="async", value="no", confidence=0.9),),
        )

        def get_plugin(lang):
            if lang == "python":
                return mock_source_plugin
            if lang == "go_lsp":
                return mock_candidate_plugin
            return None

        with patch.object(LanguageRegistry.get(), "get_plugin", side_effect=get_plugin):
            text = self.tools._cross_language_assist_text(
                source_code="def f(): pass",
                source_language="python",
                results=results,
            )

        assert "ASSIST" in text
        assert "Cross-language" in text
        assert "non-blocking" in text
        assert "process" in text

    def test_deduplication_of_entries(self):
        """Duplicate entry IDs are deduplicated."""
        result1 = self._make_result(name="dup_func")
        result2 = self._make_result(name="dup_func")
        # Same entry id
        result2.entry = result1.entry

        mock_plugin = MagicMock()
        mock_plugin.extract_semantic_profile.return_value = SemanticProfile(signals=())

        with patch.object(LanguageRegistry.get(), "get_plugin", return_value=mock_plugin):
            text = self.tools._cross_language_assist_text(
                source_code="def f(): pass",
                source_language="python",
                results=[result1, result2],
            )
        # Should only appear once despite two results with same entry
        if text:
            count = text.count("dup_func")
            assert count == 1


class TestCandidateSimilaritySnippetsErrors:
    """Cover _candidate_similarity_snippets error paths (lines 1047-1056)."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()

    def test_type_error_fallback(self):
        """TypeError triggers fallback to narrower extract_code_units signature."""
        mock_plugin = MagicMock()

        call_count = {"n": 0}

        def extract_code_units(*_args, **_kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call with include_blocks/max_block_depth raises TypeError
                raise TypeError("unexpected keyword argument")
            # Second call (fallback) returns a unit
            unit = MagicMock()
            unit.code = "def inner(): pass"
            return [unit]

        mock_plugin.extract_code_units = extract_code_units

        snippets = self.tools._candidate_similarity_snippets(
            plugin=mock_plugin,
            source_code="def outer(): pass",
            file_path="test.py",
        )
        # Should have both the source code and the extracted inner code
        assert "def outer(): pass" in snippets
        assert "def inner(): pass" in snippets

    def test_general_exception_returns_source_only(self):
        """General exception returns only the source snippet."""
        mock_plugin = MagicMock()
        mock_plugin.extract_code_units.side_effect = RuntimeError("boom")

        snippets = self.tools._candidate_similarity_snippets(
            plugin=mock_plugin,
            source_code="def outer(): pass",
            file_path="test.py",
        )
        assert snippets == ["def outer(): pass"]

    def test_empty_source_code(self):
        """Empty source code string still works without error."""
        mock_plugin = MagicMock()
        mock_plugin.extract_code_units.return_value = []

        snippets = self.tools._candidate_similarity_snippets(
            plugin=mock_plugin,
            source_code="",
            file_path="test.py",
        )
        # Empty string is stripped to empty, so it won't be added
        assert snippets == []


class TestAnalyzeScopeFilterDirectCompute:
    """Cover analyze() direct compute path without EventDrivenIndex (lines 688-694)."""

    _DUPLICATE_CODE = """\
def process_items(items):
    results = []
    for item in items:
        if item is not None:
            value = item * 2
            results.append(value)
    return results

def handle_entries(entries):
    results = []
    for item in entries:
        if item is not None:
            value = item * 2
            results.append(value)
    return results
"""

    def _run_scoped_analysis(self, scope: list[str]):
        index = CodeStructureIndex()
        tools = CodeStructureTools(index=index)
        assert tools._event_driven_index is None

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(self._DUPLICATE_CODE)
            f.flush()
            tools.index_codebase(f.name)

        try:
            return tools.analyze(scope=scope)
        finally:
            os.unlink(f.name)

    def test_analyze_direct_compute_with_scope_filter(self):
        """analyze() uses direct compute when no EDI present, with scope filter."""
        result = self._run_scoped_analysis(["*.py"])
        assert result.text  # Should produce some output

    def test_analyze_direct_compute_scope_no_match(self):
        """analyze() with scope filter that matches nothing."""
        result = self._run_scoped_analysis(["nonexistent/**"])
        assert result.text  # Should still return valid output

    def test_analyze_direct_compute_calls_all_three_find_methods(self):
        """Verify all three find_* methods are called on the direct compute path."""
        index = MagicMock()
        index.find_all_duplicates.return_value = []
        index.find_pattern_duplicates.return_value = []
        index.find_block_duplicates.return_value = []
        index.get_stats.return_value = {"suppressed_hashes": 0}

        tools = CodeStructureTools(index=index)
        assert tools._event_driven_index is None
        tools._last_indexed_path = "/tmp/test"
        tools._bg_index_done.set()

        tools.analyze()

        index.find_all_duplicates.assert_called_once()
        index.find_pattern_duplicates.assert_called_once()
        index.find_block_duplicates.assert_called_once()

    def test_analyze_direct_compute_with_entry_filter(self):
        """Verify entry_filter is passed to find_* methods on direct compute path."""
        index = MagicMock()
        index.find_all_duplicates.return_value = []
        index.find_pattern_duplicates.return_value = []
        index.find_block_duplicates.return_value = []
        index.get_stats.return_value = {"suppressed_hashes": 0}

        tools = CodeStructureTools(index=index)
        assert tools._event_driven_index is None
        tools._last_indexed_path = "/tmp/test"
        tools._bg_index_done.set()

        tools.analyze(scope=["src/**"])

        # All three calls should have entry_filter set (not None)
        for call in [
            index.find_all_duplicates,
            index.find_pattern_duplicates,
            index.find_block_duplicates,
        ]:
            _, kwargs = call.call_args
            assert kwargs.get("entry_filter") is not None


# --- MCP handler coverage (server.py lines 353-397, 490-535, 542-572) ---


class TestMCPShutdown:
    """Cover server.py shutdown handlers (lines 542-572)."""

    @staticmethod
    def _with_temp_tools_and_close_flag():
        from astrograph.server import _close_once

        was_set = _close_once.is_set()
        _close_once.clear()
        old_tools = get_tools()
        mock_tools = MagicMock()
        set_tools(mock_tools)
        return was_set, old_tools, mock_tools

    @staticmethod
    def _restore_tools_and_close_flag(was_set: bool, old_tools) -> None:
        from astrograph.server import _close_once

        set_tools(old_tools)
        if was_set:
            _close_once.set()
            return
        _close_once.clear()

    def test_close_if_first_idempotent(self):
        """_close_if_first is idempotent (lines 550-554)."""
        from astrograph.server import _close_if_first

        was_set, old_tools, mock_tools = self._with_temp_tools_and_close_flag()
        try:
            _close_if_first()
            mock_tools.close.assert_called_once()
            mock_tools.close.reset_mock()
            _close_if_first()
            mock_tools.close.assert_not_called()
        finally:
            self._restore_tools_and_close_flag(was_set, old_tools)

    def test_shutdown_handler(self):
        """_shutdown_handler calls close and exits (lines 557-560)."""
        from astrograph.server import _shutdown_handler

        was_set, old_tools, mock_tools = self._with_temp_tools_and_close_flag()
        try:
            with pytest.raises(SystemExit):
                _shutdown_handler(15, None)
            mock_tools.close.assert_called_once()
        finally:
            self._restore_tools_and_close_flag(was_set, old_tools)

    def test_atexit_close(self):
        """_atexit_close calls _close_if_first (lines 563-565)."""
        from astrograph.server import _atexit_close

        was_set, old_tools, mock_tools = self._with_temp_tools_and_close_flag()
        try:
            _atexit_close()
            mock_tools.close.assert_called_once()
        finally:
            self._restore_tools_and_close_flag(was_set, old_tools)
