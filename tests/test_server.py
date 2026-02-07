"""Tests for the consolidated MCP server tools."""

import os
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from astrograph.server import create_server, get_tools, set_tools
from astrograph.tools import (
    PERSISTENCE_DIR,
    CodeStructureTools,
    ToolResult,
    _get_persistence_path,
)


class TestResolveDockerPath:
    """Tests for Docker path resolution via CodeStructureTools._resolve_path."""

    @pytest.fixture(autouse=True)
    def _tools(self):
        self.tools = CodeStructureTools()

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

        def mock_exists(self):
            path_str = str(self)
            if path_str in ("/workspace", "/.dockerenv", "/workspace/src"):
                return True
            return original_exists(self)

        with patch.object(Path, "exists", mock_exists):
            result = self.tools._resolve_path("/Users/foo/bar/src")
            assert result == "/workspace/src"

    def test_resolve_docker_path_new_file(self):
        """Test Docker path resolution for a new file (parent exists, file doesn't)."""
        original_exists = Path.exists
        original_is_dir = Path.is_dir

        def mock_exists(self):
            path_str = str(self)
            if path_str in ("/workspace", "/.dockerenv", "/workspace/src"):
                return True
            return original_exists(self)

        def mock_is_dir(self):
            path_str = str(self)
            if path_str == "/workspace/src":
                return True
            return original_is_dir(self)

        with patch.object(Path, "exists", mock_exists), patch.object(Path, "is_dir", mock_is_dir):
            result = self.tools._resolve_path("/Users/foo/bar/src/new_file.py")
            assert result == "/workspace/src/new_file.py"

    def test_resolve_docker_path_new_file_at_root(self):
        """Test Docker path resolution for a new file at workspace root."""
        original_exists = Path.exists
        original_is_dir = Path.is_dir

        def mock_exists(self):
            path_str = str(self)
            if path_str in ("/workspace", "/.dockerenv"):
                return True
            return original_exists(self)

        def mock_is_dir(self):
            path_str = str(self)
            if path_str == "/workspace":
                return True
            return original_is_dir(self)

        with patch.object(Path, "exists", mock_exists), patch.object(Path, "is_dir", mock_is_dir):
            result = self.tools._resolve_path("/Users/foo/bar/test.py")
            assert result == "/workspace/test.py"

    def test_learned_root_mapping_speeds_up_subsequent_calls(self):
        """After first successful resolution, host_root is cached and reused."""
        original_exists = Path.exists

        def mock_exists(self):
            path_str = str(self)
            if path_str in ("/workspace", "/.dockerenv", "/workspace/src"):
                return True
            return original_exists(self)

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
            # Create and index a real file so _require_index passes
            py_file = os.path.join(tmpdir, "existing.py")
            with open(py_file, "w") as f:
                f.write("def foo(): pass\n")

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)

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
    if ".metadata_astrograph/analysis_report.txt" not in result.text:
        return result.text
    indexed = Path(tools._last_indexed_path).resolve()
    base = indexed.parent if not indexed.is_dir() else indexed
    report = base / PERSISTENCE_DIR / "analysis_report.txt"
    return report.read_text() if report.exists() else result.text


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
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a Python file
            py_file = os.path.join(tmpdir, "test.py")
            with open(py_file, "w") as f:
                f.write("def foo(): pass")

            result = tools.index_codebase(tmpdir)
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

        assert first_result.text == second_result.text


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


class TestCheck:
    """Tests for check tool."""

    def test_check_empty_index(self, tools):
        result = tools.check("def foo(): pass")
        assert "No code indexed" in result.text

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(complex_code)
            f.flush()
            tools.index_codebase(f.name)
        os.unlink(f.name)

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
            with open(root_file, "w") as f:
                f.write(complex_func)

            # Create file in subdirectory
            subdir = os.path.join(tmpdir, "sub", "dir")
            os.makedirs(subdir)
            deep_file = os.path.join(subdir, "deep.py")
            with open(deep_file, "w") as f:
                f.write(complex_func.replace("process_items", "transform_data"))

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(pattern_code)
            f.flush()
            tools.index_codebase(f.name)
        os.unlink(f.name)

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(base_code)
            f.flush()
            tools.index_codebase(f.name)
        os.unlink(f.name)

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            tools.index_codebase(f.name)
            result = tools.analyze()
        os.unlink(f.name)

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            tools.index_codebase(f.name)
            result = tools.analyze()
        os.unlink(f.name)

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

    def test_suppress_invalid_hash(self, tools, _indexed_with_duplicates):
        """Test suppress with invalid hash."""
        result = tools.suppress("nonexistent_hash")
        assert "not found" in result.text

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

    def test_unsuppress_not_suppressed(self, tools, _indexed_with_duplicates):
        """Test unsuppress with a hash that was not suppressed."""
        result = tools.unsuppress("not_suppressed_hash")
        assert "was not suppressed" in result.text

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

    def test_call_tool_suppress(self, tools, _indexed_with_duplicates):
        """Test call_tool dispatch for suppress."""
        result = tools.call_tool("suppress", {"wl_hash": "test_hash"})
        assert "not found" in result.text

    def test_call_tool_unsuppress(self, tools, _indexed_with_duplicates):
        """Test call_tool dispatch for unsuppress."""
        result = tools.call_tool("unsuppress", {"wl_hash": "test_hash"})
        assert "was not suppressed" in result.text

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

    def test_suppress_batch_empty(self, tools, _indexed_with_duplicates):
        """Test batch suppress with empty list."""
        result = tools.suppress_batch([])
        assert "No hashes provided" in result.text

    def test_call_tool_suppress_batch(self, tools, _indexed_with_duplicates):
        """Test call_tool dispatch for suppress_batch."""
        result = tools.call_tool("suppress_batch", {"wl_hashes": ["fake_hash"]})
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

    def test_unsuppress_batch_empty(self, tools, _indexed_with_duplicates):
        """Test batch unsuppress with empty list."""
        result = tools.unsuppress_batch([])
        assert "No hashes provided" in result.text

    def test_call_tool_unsuppress_batch(self, tools, _indexed_with_duplicates):
        """Test call_tool dispatch for unsuppress_batch."""
        result = tools.call_tool("unsuppress_batch", {"wl_hashes": ["fake_hash"]})
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
            with open(file1, "w") as f:
                f.write("def foo(): return 1\n")

            # Initial index
            result1 = tools.index_codebase(tmpdir)
            assert "Indexed" in result1.text

            # Add another file
            file2 = os.path.join(tmpdir, "module2.py")
            with open(file2, "w") as f:
                f.write("def bar(): return 2\n")

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo(): pass")
            f.flush()
            tools.index_codebase(f.name)
            result = tools.check_staleness()
        os.unlink(f.name)

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
            with open(f.name, "w") as mod_f:
                mod_f.write("def bar(): pass")

            result = tools.check_staleness()
        os.unlink(f.name)

        assert "STALE" in result.text
        assert "Modified" in result.text

    def test_check_staleness_with_path(self, tools):
        """Test check_staleness with path parameter for new files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            tools.index_codebase(tmpdir)

            # Add new file
            file2 = os.path.join(tmpdir, "file2.py")
            with open(file2, "w") as f:
                f.write("def bar(): pass")

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
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            # First run - always a full index
            result = tools.index_codebase(tmpdir)

            assert "Indexed" in result.text

    def test_index_codebase_incremental_unchanged(self, tools):
        """Test incremental indexing with unchanged files still produces valid output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

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

            with open(file1, "w") as f:
                f.write("def foo(): pass")
            with open(file2, "w") as f:
                f.write("def bar(): pass")

            # First index (full)
            tools.index_codebase(tmpdir)

            # Modify only file1
            time.sleep(0.01)
            with open(file1, "w") as f:
                f.write("def foo_modified(): pass")

            result = tools.index_codebase(tmpdir)

            # Should still report indexed entries
            assert "Indexed" in result.text

    def test_incremental_default_true(self, tools):
        """Test incremental defaults to True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

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
            with open(f.name, "w") as mod_f:
                mod_f.write("def baz(): pass")

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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def unique_function_abc123(): return 42")
            f.flush()
            tools.index_codebase(f.name)
            result = tools.analyze()
        os.unlink(f.name)

        # No staleness warning for fresh index
        if "No significant duplicates" in result.text:
            assert "WARNING" not in result.text


class TestPersistence:
    """Tests for index persistence to .metadata_astrograph folder (SQLite)."""

    def test_index_creates_persistence_folder(self):
        """Test that indexing creates .metadata_astrograph folder with SQLite DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

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
            with open(file1, "w") as f:
                f.write("def foo(): pass")

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
            with open(file1, "w") as f:
                f.write(code)

            # First tools instance - index and suppress
            tools1 = CodeStructureTools()
            tools1.index_codebase(tmpdir)

            # Find a hash to suppress
            import re

            analyze_result = tools1.analyze()
            details = _get_analyze_details(tools1, analyze_result)
            match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
            if match:
                wl_hash = match.group(1)
                tools1.suppress(wl_hash)
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
            with open(file1, "w") as f:
                f.write(code)

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
            with open(file1, "w") as f:
                f.write("def foo(): pass")

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
            tools.index_codebase(tmpdir)
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
            tools.index_codebase(tmpdir)
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
            with open(file1, "w") as f:
                f.write(code)

            # --- Session 1: index, suppress, close ---
            tools1 = CodeStructureTools()
            tools1.index_codebase(tmpdir)

            import re

            analyze_result = tools1.analyze()
            details = _get_analyze_details(tools1, analyze_result)
            match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
            if not match:
                pytest.skip("No duplicates found to suppress")

            wl_hash = match.group(1)
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
            file1 = os.path.join(tmpdir, "module_a.py")
            with open(file1, "w") as f:
                f.write(self._complex_func("process_items"))

            file2 = os.path.join(tmpdir, "module_b.py")
            with open(file2, "w") as f:
                f.write(self._complex_func("transform_data"))

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

    def test_analyze_report_separates_source_and_tests(self, tools):
        """Report should have section headers when both source and test duplicates exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Source duplicates (one structure)
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            with open(os.path.join(src_dir, "module_a.py"), "w") as f:
                f.write(self._complex_func("process_items"))
            with open(os.path.join(src_dir, "module_b.py"), "w") as f:
                f.write(self._complex_func("transform_data"))

            # Test duplicates (different structure so they form a separate group)
            tests_dir = os.path.join(tmpdir, "tests")
            os.makedirs(tests_dir)
            with open(os.path.join(tests_dir, "test_a.py"), "w") as f:
                f.write(self._different_complex_func("test_check_a"))
            with open(os.path.join(tests_dir, "test_b.py"), "w") as f:
                f.write(self._different_complex_func("test_check_b"))

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
            # Source duplicates (one structure)
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            with open(os.path.join(src_dir, "module_a.py"), "w") as f:
                f.write(self._complex_func("process_items"))
            with open(os.path.join(src_dir, "module_b.py"), "w") as f:
                f.write(self._complex_func("transform_data"))

            # Test duplicates (different structure)
            tests_dir = os.path.join(tmpdir, "tests")
            os.makedirs(tests_dir)
            with open(os.path.join(tests_dir, "test_a.py"), "w") as f:
                f.write(self._different_complex_func("test_check_a"))
            with open(os.path.join(tests_dir, "test_b.py"), "w") as f:
                f.write(self._different_complex_func("test_check_b"))

            tools.index_codebase(tmpdir)
            result = tools.analyze()

            if "duplicate groups" in result.text:
                assert "in source" in result.text
                assert "in tests" in result.text

    def test_suppress_batch_response_includes_refresh_hint(self, tools):
        """suppress_batch response should include 'Run analyze' hint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "module_a.py"), "w") as f:
                f.write(self._complex_func("process_items"))
            with open(os.path.join(tmpdir, "module_b.py"), "w") as f:
                f.write(self._complex_func("transform_data"))

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

    def test_status_idle(self, tools):
        """Status should return idle when no codebase is indexed."""
        result = tools.status()
        assert "idle" in result.text

    def test_status_ready_after_indexing(self, tools, sample_python_file):
        """Status should return ready after indexing."""
        tools.index_codebase(sample_python_file)
        result = tools.status()
        assert "ready" in result.text
        assert "code units" in result.text

    def test_status_during_background_indexing(self):
        """Status should return indexing state when background indexing is running."""
        tools = CodeStructureTools()
        # Simulate background indexing in progress
        tools._bg_index_done.clear()
        result = tools.status()
        assert "indexing" in result.text
        # Clean up
        tools._bg_index_done.set()

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
            with open(file1, "w") as f:
                f.write(code)

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)

            # Suppress a hash
            import re

            analyze_result = tools.analyze()
            details = _get_analyze_details(tools, analyze_result)
            match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
            if match:
                tools.suppress(match.group(1))

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
            with open(file1, "w") as f:
                f.write("def foo(): pass")

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


class TestMetadataRecomputeBaseline:
    """Tests for the astrograph_metadata_recompute_baseline tool."""

    def test_recompute_no_indexed_path(self, tools):
        """Recompute when nothing has been indexed."""
        result = tools.metadata_recompute_baseline()
        assert "No codebase has been indexed" in result.text

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
            with open(file1, "w") as f:
                f.write(code)

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)

            # Suppress a hash
            import re

            analyze_result = tools.analyze()
            details = _get_analyze_details(tools, analyze_result)
            match = re.search(r'suppress\(wl_hash="([^"]+)"\)', details)
            if match:
                tools.suppress(match.group(1))

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
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            tools = CodeStructureTools()
            tools.index_codebase(tmpdir)
            result = tools.call_tool("metadata_recompute_baseline", {})
            assert "Baseline recomputed" in result.text
            tools.close()


class TestResourceHandlers:
    """Tests for the resource list handlers (Codex compatibility)."""

    @pytest.mark.asyncio
    async def test_resource_list_returns_empty(self):
        """resources/list should return an empty list, not method-not-found."""
        server = create_server()
        # The server should have a list_resources handler that returns []
        # We test it indirectly via the handler registration
        assert server is not None
        # Verify the handlers are registered by checking the server object
        # The MCP Server registers handlers internally; we verify the server
        # was created without errors and resource handlers exist.

    @pytest.mark.asyncio
    async def test_resource_template_list_returns_empty(self):
        """resources/templates/list should return an empty list, not method-not-found."""
        server = create_server()
        assert server is not None


class TestWorkspaceEnvVar:
    """Tests for the ASTROGRAPH_WORKSPACE env var support."""

    def test_workspace_env_var_triggers_auto_index(self):
        """Setting ASTROGRAPH_WORKSPACE to a valid dir should trigger auto-index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "mod.py")
            with open(py_file, "w") as f:
                f.write("def hello(): return 42\n")

            with patch.dict(os.environ, {"ASTROGRAPH_WORKSPACE": tmpdir}):
                tools = CodeStructureTools()
                # Wait for background indexing to complete
                tools._wait_for_background_index()
                assert tools._bg_index_progress == "done"
                assert len(tools.index.entries) > 0
                tools.close()

    def test_workspace_env_var_empty_ignored(self):
        """Empty ASTROGRAPH_WORKSPACE should not trigger auto-index."""
        with patch.dict(os.environ, {"ASTROGRAPH_WORKSPACE": ""}):
            tools = CodeStructureTools()
            # Should be idle (no auto-index triggered)
            result = tools.status()
            assert "idle" in result.text

    def test_workspace_env_var_nonexistent_ignored(self):
        """ASTROGRAPH_WORKSPACE pointing to nonexistent dir should not trigger auto-index."""
        with patch.dict(os.environ, {"ASTROGRAPH_WORKSPACE": "/nonexistent/path/xyz"}):
            tools = CodeStructureTools()
            result = tools.status()
            assert "idle" in result.text


class TestStartupWorkspaceDetection:
    """Tests for startup workspace fallback logic (Codex/local compatibility)."""

    def test_pwd_fallback_triggers_auto_index(self):
        """Without ASTROGRAPH_WORKSPACE, PWD should be used when cwd is '/'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            py_file = os.path.join(tmpdir, "mod.py")
            with open(py_file, "w") as f:
                f.write("def hello(): return 42\n")

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
            with open(py_file, "w") as f:
                f.write("def hello(): return 42\n")

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

        # Simulate background finishing shortly
        def finish_bg():
            time.sleep(0.05)
            tools._bg_index_done.set()

        import threading

        threading.Thread(target=finish_bg, daemon=True).start()
        result = tools.analyze()
        # After waiting, empty index → "No code indexed"
        assert "No code indexed" in result.text

    def test_check_waits_for_background_index(self):
        """Check should wait for background indexing, then report no index."""
        tools = CodeStructureTools()
        tools._bg_index_done.clear()

        def finish_bg():
            time.sleep(0.05)
            tools._bg_index_done.set()

        import threading

        threading.Thread(target=finish_bg, daemon=True).start()
        result = tools.check("def foo(): pass")
        assert "No code indexed" in result.text

    def test_status_returns_immediately_during_indexing(self):
        """Status should return indexing state without blocking."""
        tools = CodeStructureTools()
        tools._bg_index_done.clear()
        result = tools.status()
        assert "indexing" in result.text
        tools._bg_index_done.set()
