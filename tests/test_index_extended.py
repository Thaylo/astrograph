"""Extended tests for the index module."""

import pytest

from astrograph.ast_to_graph import CodeUnit, code_unit_to_ast_graph
from astrograph.index import (
    CodeStructureIndex,
    IndexEntry,
)


class TestIndexEntry:
    """Tests for IndexEntry serialization."""

    def test_to_dict_and_back(self):
        unit = CodeUnit(
            name="test_func",
            code="def test_func(x): return x",
            file_path="/test/file.py",
            line_start=1,
            line_end=2,
            unit_type="function",
            parent_name="TestClass",
        )

        entry = IndexEntry(
            id="entry_1",
            wl_hash="abc123",
            pattern_hash="pat123",
            fingerprint={"n_nodes": 5, "n_edges": 4},
            hierarchy_hashes=["h1", "h2", "h3"],
            code_unit=unit,
            node_count=5,
            depth=3,
        )

        # Convert to dict and back
        data = entry.to_dict()
        restored = IndexEntry.from_dict(data)

        assert restored.id == entry.id
        assert restored.wl_hash == entry.wl_hash
        assert restored.code_unit.name == entry.code_unit.name
        assert restored.code_unit.parent_name == entry.code_unit.parent_name

    def test_to_location_dict_basic(self):
        unit = CodeUnit(
            name="my_func",
            code="def my_func(): pass",
            file_path="/src/module.py",
            line_start=10,
            line_end=15,
            unit_type="function",
        )
        entry = IndexEntry(
            id="entry_1",
            wl_hash="abc",
            pattern_hash="pat",
            fingerprint={},
            hierarchy_hashes=[],
            code_unit=unit,
            node_count=3,
            depth=2,
        )

        loc = entry.to_location_dict()

        assert loc["file"] == "/src/module.py"
        assert loc["name"] == "my_func"
        assert loc["type"] == "function"
        assert loc["lines"] == "10-15"
        assert "parent" not in loc
        assert "code" not in loc

    def test_to_location_dict_with_parent(self):
        unit = CodeUnit(
            name="method",
            code="def method(self): return 1",
            file_path="/src/class.py",
            line_start=5,
            line_end=6,
            unit_type="method",
            parent_name="MyClass",
        )
        entry = IndexEntry(
            id="entry_2",
            wl_hash="def",
            pattern_hash="pat2",
            fingerprint={},
            hierarchy_hashes=[],
            code_unit=unit,
            node_count=4,
            depth=2,
        )

        loc = entry.to_location_dict()

        assert loc["parent"] == "MyClass"

    def test_to_location_dict_with_code(self):
        long_code = "def func(): " + "x = 1; " * 100  # Long code
        unit = CodeUnit(
            name="func",
            code=long_code,
            file_path="/src/long.py",
            line_start=1,
            line_end=10,
            unit_type="function",
        )
        entry = IndexEntry(
            id="entry_3",
            wl_hash="ghi",
            pattern_hash="pat3",
            fingerprint={},
            hierarchy_hashes=[],
            code_unit=unit,
            node_count=100,
            depth=2,
        )

        loc = entry.to_location_dict(include_code=True)

        assert "code" in loc
        assert len(loc["code"]) <= 503  # 500 + "..."
        assert loc["code"].endswith("...")


class TestCodeStructureIndexExtended:
    """Extended tests for CodeStructureIndex."""

    @staticmethod
    def _add_simple_function_units(
        index: CodeStructureIndex,
        count: int,
        name_prefix: str,
        code: str,
        file_prefix: str,
    ) -> None:
        for i in range(count):
            index.add_code_unit(
                CodeUnit(
                    name=f"{name_prefix}_{i}",
                    code=code,
                    file_path=f"{file_prefix}_{i}.py",
                    line_start=1,
                    line_end=1,
                    unit_type="function",
                )
            )

    @pytest.mark.parametrize(
        "method,path",
        [
            ("index_file", "/nonexistent/file.py"),
            ("index_directory", "/nonexistent/dir"),
        ],
    )
    def test_nonexistent_path_returns_empty(self, method, path):
        """Indexing nonexistent paths should return empty list."""
        index = CodeStructureIndex()
        entries = getattr(index, method)(path)
        assert entries == []

    def test_index_file_non_python(self, tmp_path):
        index = CodeStructureIndex()
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("hello world")
        entries = index.index_file(str(txt_file))
        assert entries == []

    def test_index_file_invalid_encoding(self, tmp_path):
        index = CodeStructureIndex()
        bin_file = tmp_path / "file.py"
        bin_file.write_bytes(b"\xff\xfe invalid utf-8")
        entries = index.index_file(str(bin_file))
        assert entries == []

    def test_index_directory_skips_pycache(self, tmp_path):
        index = CodeStructureIndex()

        # Create __pycache__ directory with Python file
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.py").write_text("def f(): pass")

        # Create regular file
        (tmp_path / "regular.py").write_text("def g(): pass")

        entries = index.index_directory(str(tmp_path))

        # Should only index regular.py, not the one in __pycache__
        file_paths = [e.code_unit.file_path for e in entries]
        assert not any("__pycache__" in fp for fp in file_paths)

    def test_index_directory_skips_venv_variants(self, tmp_path):
        """Test that numbered/prefixed venv directories are skipped."""
        index = CodeStructureIndex()

        # Create venv variant directories with Python files
        for venv_dir in [".venv311", "venv3.11", ".env", "env", "virtualenv"]:
            d = tmp_path / venv_dir / "lib"
            d.mkdir(parents=True)
            (d / "site.py").write_text("def f(): pass")

        # Create a normal project file
        (tmp_path / "app.py").write_text("def main(): pass")

        entries = index.index_directory(str(tmp_path))

        # Should only index app.py
        file_paths = [e.code_unit.file_path for e in entries]
        assert len(file_paths) == 1
        assert file_paths[0].endswith("app.py")

    def test_index_directory_does_not_skip_similar_names(self, tmp_path):
        """Directories like 'environment', 'envoy', 'vengeance' should NOT be skipped.

        Note: 'vendor' IS a legitimate skip dir (Go plugin), so we use
        other names that merely look like venv prefixes but aren't.
        """
        index = CodeStructureIndex()

        for normal_dir in ["environment", "envoy", "vengeance"]:
            d = tmp_path / normal_dir
            d.mkdir()
            (d / "module.py").write_text("def f(): pass")

        entries = index.index_directory(str(tmp_path))

        # All three should be indexed
        file_paths = [e.code_unit.file_path for e in entries]
        assert len(file_paths) == 3

    def test_index_directory_non_recursive(self, tmp_path):
        index = CodeStructureIndex()

        # Create subdirectory with file
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "sub.py").write_text("def f(): pass")

        # Create file in root
        (tmp_path / "root.py").write_text("def g(): pass")

        entries = index.index_directory(str(tmp_path), recursive=False)

        # Should only index root.py
        file_paths = [e.code_unit.file_path for e in entries]
        assert any("root.py" in fp for fp in file_paths)
        assert not any("sub.py" in fp for fp in file_paths)

    def test_remove_file(self, tmp_path):
        index = CodeStructureIndex()

        file1 = tmp_path / "file1.py"
        file1.write_text("def f(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def g(): pass")

        index.index_file(str(file1))
        index.index_file(str(file2))

        assert index.get_stats()["total_entries"] == 2

        index.remove_file(str(file1))

        assert index.get_stats()["total_entries"] == 1

    def test_remove_nonexistent_file(self):
        index = CodeStructureIndex()
        # Should not raise
        index.remove_file("/nonexistent/file.py")

    def test_find_exact_matches_empty_index(self):
        index = CodeStructureIndex()
        matches = index.find_exact_matches("def f(): pass")
        assert matches == []

    def test_find_similar_small_code(self):
        index = CodeStructureIndex()
        # Code too small to match
        results = index.find_similar("x = 1", min_node_count=10)
        assert results == []

    def test_add_and_query(self):
        index = CodeStructureIndex()

        unit = CodeUnit(
            name="test",
            code="def test(): pass",
            file_path="test.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        index.add_code_unit(unit)

        assert index.get_stats()["total_entries"] == 1

    @pytest.mark.parametrize(
        "code1,code2,expected",
        [
            ("def f(x): return x + 1", "def g(y): return y + 1", True),
            ("def f(x): return x + 1", "def g(x, y): return x + y", False),
        ],
    )
    def test_verify_isomorphism(self, code1, code2, expected):
        """Test isomorphism verification with expected result."""
        index = CodeStructureIndex()

        unit1 = CodeUnit(
            name="f", code=code1, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="g", code=code2, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        e1 = index.add_code_unit(unit1)
        e2 = index.add_code_unit(unit2)

        assert index.verify_isomorphism(e1, e2) is expected

    def test_find_duplicates_returns_sorted(self):
        index = CodeStructureIndex()

        # Add functions with different duplication counts
        self._add_simple_function_units(index, 3, "func_a", "def f(x): return x + 1", "file")
        self._add_simple_function_units(index, 2, "func_b", "def g(y): return y * 2", "other")

        groups = index.find_all_duplicates(min_node_count=3)

        # Groups should be sorted by size (largest first)
        if len(groups) >= 2:
            assert len(groups[0].entries) >= len(groups[1].entries)

    def test_find_similar_returns_sorted(self):
        index = CodeStructureIndex()

        # Add an exact match
        exact_code = "def f(x): return x + 1"
        unit = CodeUnit(
            name="f",
            code=exact_code,
            file_path="exact.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        index.add_code_unit(unit)

        # Search with the same code
        results = index.find_similar(exact_code, min_node_count=3)

        # First result should be exact match
        if results:
            assert results[0].similarity_type == "exact"


class TestCodeUnitToAstGraph:
    """Tests for code unit to AST graph conversion."""

    def test_empty_graph_depth(self):
        unit = CodeUnit(
            name="empty",
            code="# just a comment",
            file_path="empty.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        ast_graph = code_unit_to_ast_graph(unit)
        # Empty/invalid code should result in depth 0 or minimal depth
        assert ast_graph.depth >= 0

    def test_complex_code_depth(self):
        code = """
def nested(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                print(i)
    return x
"""
        unit = CodeUnit(
            name="nested",
            code=code,
            file_path="nested.py",
            line_start=1,
            line_end=7,
            unit_type="function",
        )
        ast_graph = code_unit_to_ast_graph(unit)
        # Should have significant depth due to nesting
        assert ast_graph.depth >= 3


class TestPatternDuplicates:
    """Tests for pattern-based duplicate detection."""

    def test_finds_pattern_duplicates_with_different_operators(self):
        """Pattern duplicates should find code with same structure but different operators."""
        index = CodeStructureIndex()

        # Same structure, different comparison operators
        code1 = "def check(x): return x == 0"
        code2 = "def check(x): return x != 0"

        unit1 = CodeUnit(
            name="check_equal",
            code=code1,
            file_path="file1.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        unit2 = CodeUnit(
            name="check_not_equal",
            code=code2,
            file_path="file2.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        # They should have different WL hashes (exact match)
        groups = index.find_all_duplicates(min_node_count=3)
        assert len(groups) == 0, "Different operators should not be exact matches"

        # But they should have same pattern hash
        pattern_groups = index.find_pattern_duplicates(min_node_count=3)
        assert len(pattern_groups) >= 1, "Same pattern with different operators should match"

    def test_finds_pattern_duplicates_with_different_binary_ops(self):
        """Pattern duplicates should find code with same structure but different binary operators."""
        index = CodeStructureIndex()

        code1 = "def calc(a, b): return a + b"
        code2 = "def calc(a, b): return a * b"

        unit1 = CodeUnit(
            name="add",
            code=code1,
            file_path="f1.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        unit2 = CodeUnit(
            name="mult",
            code=code2,
            file_path="f2.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        pattern_groups = index.find_pattern_duplicates(min_node_count=3)
        assert len(pattern_groups) >= 1

    def test_excludes_exact_duplicates_from_pattern_results(self):
        """Pattern duplicates should not include groups that are already exact duplicates."""
        index = CodeStructureIndex()

        # These are exact duplicates (same structure AND same operators)
        code = "def f(x): return x + 1"

        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        exact_groups = index.find_all_duplicates(min_node_count=3)
        pattern_groups = index.find_pattern_duplicates(min_node_count=3)

        # Should have exact duplicates
        assert len(exact_groups) >= 1

        # Pattern duplicates should exclude the exact duplicate group
        # (they have the same wl_hash)
        {g.wl_hash for g in pattern_groups}
        exact_wl_hashes = {g.wl_hash for g in exact_groups}

        # Pattern groups should use pattern_hash, not overlap with exact wl_hash
        # Actually, since we're using pattern_hash as wl_hash in DuplicateGroup,
        # the check is that pattern groups exclude groups where all entries share same wl_hash
        for pg in pattern_groups:
            entry_wl_hashes = {e.wl_hash for e in pg.entries}
            # Should not all be the same wl_hash (which would mean exact duplicates)
            if len(pg.entries) >= 2:
                assert len(entry_wl_hashes) > 1 or entry_wl_hashes.pop() not in exact_wl_hashes

    def test_pattern_hash_stored_in_entry(self):
        """Index entries should have pattern_hash field."""
        index = CodeStructureIndex()

        unit = CodeUnit(
            name="f",
            code="def f(x): return x + 1",
            file_path="f.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        entry = index.add_code_unit(unit)

        assert hasattr(entry, "pattern_hash")
        assert entry.pattern_hash is not None
        assert isinstance(entry.pattern_hash, str)

    def test_stats_include_pattern_groups(self):
        """Index stats should include pattern group count."""
        index = CodeStructureIndex()

        code1 = "def check(x): return x == 0"
        code2 = "def check(x): return x != 0"

        unit1 = CodeUnit(
            name="f1", code=code1, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code2, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        stats = index.get_stats()
        assert "unique_patterns" in stats


class TestPatternHashRelabeling:
    """Test that graph relabeling produces identical hashes to re-parsing."""

    @pytest.mark.parametrize(
        "code",
        [
            # Binary operators
            "def f(a, b): return a + b",
            "def f(a, b): return a - b",
            "def f(a, b): return a * b",
            "def f(a, b): return a / b",
            "def f(a, b): return a // b",
            "def f(a, b): return a % b",
            "def f(a, b): return a ** b",
            "def f(a, b): return a << b",
            "def f(a, b): return a >> b",
            "def f(a, b): return a | b",
            "def f(a, b): return a ^ b",
            "def f(a, b): return a & b",
            "def f(a, b): return a @ b",
            # Comparison operators
            "def f(x): return x == 0",
            "def f(x): return x != 0",
            "def f(x): return x < 0",
            "def f(x): return x <= 0",
            "def f(x): return x > 0",
            "def f(x): return x >= 0",
            "def f(x): return x is None",
            "def f(x): return x is not None",
            "def f(x): return x in [1, 2]",
            "def f(x): return x not in [1, 2]",
            # Multi-comparator
            "def f(x): return 0 < x < 10",
            "def f(x): return 0 <= x != 10",
            # Unary operators
            "def f(x): return +x",
            "def f(x): return -x",
            "def f(x): return not x",
            "def f(x): return ~x",
            # Boolean operators
            "def f(x, y): return x and y",
            "def f(x, y): return x or y",
            # Augmented assignment
            "def f(x):\n    x += 1\n    return x",
            "def f(x):\n    x *= 2\n    return x",
            # Constants (should be unchanged)
            "def f(): return 42",
            "def f(): return 'hello'",
            # Mixed operators
            "def f(a, b): return (a + b) * (a - b)",
            "def f(x, y): return x > 0 and y < 0",
            # No operators
            "def f(x): return x",
            "def f(): pass",
        ],
        ids=lambda code: code.split(":")[0]
        .strip()
        .replace("def f", "")
        .replace("(", "_")
        .replace(")", "")[:40],
    )
    def test_relabel_hash_equals_reparse_hash(self, code):
        """Verify relabeled hash matches reparsed hash for all operator types."""
        from astrograph.canonical_hash import weisfeiler_leman_hash
        from astrograph.languages.python_plugin import PythonPlugin

        plugin = PythonPlugin()

        # Method 1: Re-parse with normalize_ops=True (old approach)
        reparsed_graph = plugin.source_to_graph(code, normalize_ops=True)
        reparsed_hash = weisfeiler_leman_hash(reparsed_graph)

        # Method 2: Parse once, then relabel (new approach)
        normal_graph = plugin.source_to_graph(code, normalize_ops=False)
        relabeled_graph = plugin.normalize_graph_for_pattern(normal_graph)
        relabeled_hash = weisfeiler_leman_hash(relabeled_graph)

        assert relabeled_hash == reparsed_hash, (
            f"Hash mismatch for: {code}\n"
            f"  reparsed:  {reparsed_hash}\n"
            f"  relabeled: {relabeled_hash}"
        )

    def test_normalize_returns_copy(self):
        """normalize_graph_for_pattern should return a copy, not modify the original."""
        from astrograph.languages.python_plugin import PythonPlugin

        plugin = PythonPlugin()
        graph = plugin.source_to_graph("def f(a, b): return a + b")

        original_labels = {n: d["label"] for n, d in graph.nodes(data=True)}
        normalized = plugin.normalize_graph_for_pattern(graph)

        # Original should be unchanged
        for n, d in graph.nodes(data=True):
            assert d["label"] == original_labels[n]

        # Normalized should have different labels for operator nodes
        has_change = any(
            graph.nodes[n]["label"] != normalized.nodes[n]["label"] for n in graph.nodes()
        )
        assert has_change, "Expected at least one label to change for code with operators"

    def test_non_operator_labels_preserved(self):
        """Labels like 'Call', 'Name', 'Constant:int' should be unchanged."""
        from astrograph.languages.python_plugin import PythonPlugin

        plugin = PythonPlugin()
        graph = plugin.source_to_graph("def f(): return len([1, 2, 3])")
        normalized = plugin.normalize_graph_for_pattern(graph)

        for n, d in normalized.nodes(data=True):
            label = d["label"]
            # These common non-operator labels should appear unchanged
            if label in ("Call", "Name", "Constant:int", "List", "Return", "FunctionDef"):
                assert graph.nodes[n]["label"] == label


class TestBlockDuplicates:
    """Tests for block duplicate detection."""

    def test_block_stored_in_block_buckets(self, tmp_path):
        """Block entries should be stored in block_buckets, not hash_buckets."""
        index = CodeStructureIndex()

        source = """
def func():
    for i in range(10):
        print(i)
"""
        file = tmp_path / "test.py"
        file.write_text(source)

        entries = index.index_file(str(file), include_blocks=True)
        blocks = [e for e in entries if e.code_unit.unit_type == "block"]
        funcs = [e for e in entries if e.code_unit.unit_type == "function"]

        assert len(blocks) == 1
        assert len(funcs) == 1

        # Block should be in block_buckets
        assert len(index.block_buckets) > 0

        # Function should be in hash_buckets
        assert len(index.hash_buckets) > 0

    def test_find_block_duplicates(self, tmp_path):
        """Find duplicate blocks across functions."""
        index = CodeStructureIndex()

        source = """
def func1():
    for i in range(10):
        print(i)

def func2():
    for j in range(10):
        print(j)
"""
        file = tmp_path / "test.py"
        file.write_text(source)

        index.index_file(str(file), include_blocks=True)

        groups = index.find_block_duplicates(min_node_count=5)

        # Should find one duplicate group of blocks
        assert len(groups) >= 1
        assert len(groups[0].entries) >= 2

    def test_find_block_duplicates_filter_by_type(self, tmp_path):
        """Filter block duplicates by type."""
        index = CodeStructureIndex()

        source = """
def func1():
    for i in range(10):
        print(i)
    if True:
        pass

def func2():
    for j in range(10):
        print(j)
    if True:
        pass
"""
        file = tmp_path / "test.py"
        file.write_text(source)

        index.index_file(str(file), include_blocks=True)

        # Filter for only 'for' blocks
        for_groups = index.find_block_duplicates(min_node_count=3, block_types=["for"])
        for group in for_groups:
            for entry in group.entries:
                assert entry.code_unit.block_type == "for"

    def test_block_type_index_populated(self, tmp_path):
        """Block type index should be populated with block entries."""
        index = CodeStructureIndex()

        source = """
def func():
    for i in range(10):
        pass
    if True:
        pass
"""
        file = tmp_path / "test.py"
        file.write_text(source)

        index.index_file(str(file), include_blocks=True)

        assert "for" in index.block_type_index
        assert "if" in index.block_type_index

    def test_remove_file_clears_block_entries(self, tmp_path):
        """Removing a file should clear its block entries from block_buckets."""
        index = CodeStructureIndex()

        source = """
def func():
    for i in range(10):
        print(i)
"""
        file = tmp_path / "test.py"
        file.write_text(source)

        index.index_file(str(file), include_blocks=True)

        assert index.get_stats()["block_entries"] == 1
        assert len(index.block_buckets) > 0

        index.remove_file(str(file))

        assert index.get_stats()["block_entries"] == 0
        assert len(index.block_buckets) == 0
        assert len(index.block_type_index) == 0

    def test_index_with_blocks(self, tmp_path):
        """Index should correctly track block entries."""
        index = CodeStructureIndex()

        source = """
def func():
    for i in range(10):
        print(i)
"""
        file = tmp_path / "source.py"
        file.write_text(source)

        index.index_file(str(file), include_blocks=True)

        assert index.get_stats()["block_entries"] == 1
        assert len(index.block_buckets) >= 1

    def test_stats_include_block_info(self, tmp_path):
        """Stats should include block-related information."""
        index = CodeStructureIndex()

        source = """
def func1():
    for i in range(10):
        print(i)

def func2():
    for j in range(10):
        print(j)
"""
        file = tmp_path / "test.py"
        file.write_text(source)

        index.index_file(str(file), include_blocks=True)

        stats = index.get_stats()

        assert "block_entries" in stats
        assert "unique_block_hashes" in stats
        assert stats["block_entries"] == 2

    def test_clear_clears_block_data(self, tmp_path):
        """Clear should also clear block-related data."""
        index = CodeStructureIndex()

        source = """
def func():
    for i in range(10):
        print(i)
"""
        file = tmp_path / "test.py"
        file.write_text(source)

        index.index_file(str(file), include_blocks=True)

        assert len(index.block_buckets) > 0
        assert len(index.block_type_index) > 0

        index.clear()

        assert len(index.block_buckets) == 0
        assert len(index.block_type_index) == 0

    def test_index_entry_block_fields_serialization(self):
        """Block-specific fields should be serialized and deserialized correctly."""
        unit = CodeUnit(
            name="func.for_1",
            code="for i in range(10): print(i)",
            file_path="/test/file.py",
            line_start=5,
            line_end=6,
            unit_type="block",
            parent_name="func",
            block_type="for",
            nesting_depth=1,
            parent_block_name=None,
        )

        entry = IndexEntry(
            id="entry_1",
            wl_hash="abc123",
            pattern_hash="pat123",
            fingerprint={"n_nodes": 10},
            hierarchy_hashes=["h1"],
            code_unit=unit,
            node_count=10,
            depth=3,
        )

        data = entry.to_dict()
        assert data["code_unit"]["block_type"] == "for"
        assert data["code_unit"]["nesting_depth"] == 1
        assert "parent_block_name" not in data["code_unit"]  # None is not serialized

        restored = IndexEntry.from_dict(data)
        assert restored.code_unit.block_type == "for"
        assert restored.code_unit.nesting_depth == 1
        assert restored.code_unit.parent_block_name is None


class TestSuppression:
    """Tests for duplicate suppression functionality."""

    def test_suppress_valid_hash(self):
        """Suppressing a valid hash should return True."""
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        assert len(groups) >= 1

        wl_hash = groups[0].wl_hash
        success = index.suppress(wl_hash)
        assert success is True

    def test_suppress_invalid_hash(self):
        """Suppressing an invalid hash should return False."""
        index = CodeStructureIndex()
        success = index.suppress("nonexistent_hash")
        assert success is False

    def test_suppressed_groups_not_in_duplicates(self):
        """Suppressed groups should not appear in find_all_duplicates."""
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        assert len(groups) >= 1

        wl_hash = groups[0].wl_hash
        index.suppress(wl_hash)

        # After suppression, the group should not appear
        groups_after = index.find_all_duplicates(min_node_count=3)
        assert len(groups_after) == 0

    def test_suppressed_blocks_not_in_block_duplicates(self, tmp_path):
        """Suppressed blocks should not appear in find_block_duplicates."""
        index = CodeStructureIndex()

        source = """
def func1():
    for i in range(10):
        print(i)

def func2():
    for j in range(10):
        print(j)
"""
        file = tmp_path / "test.py"
        file.write_text(source)

        index.index_file(str(file), include_blocks=True)

        groups = index.find_block_duplicates(min_node_count=5)
        assert len(groups) >= 1

        wl_hash = groups[0].wl_hash
        index.suppress(wl_hash)

        groups_after = index.find_block_duplicates(min_node_count=5)
        assert len(groups_after) == 0

    def test_unsuppress(self):
        """Unsuppressing should make the group appear again."""
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        wl_hash = groups[0].wl_hash

        index.suppress(wl_hash)
        assert len(index.find_all_duplicates(min_node_count=3)) == 0

        index.unsuppress(wl_hash)
        assert len(index.find_all_duplicates(min_node_count=3)) >= 1

    def test_unsuppress_not_suppressed(self):
        """Unsuppressing a non-suppressed hash should return False."""
        index = CodeStructureIndex()
        assert index.unsuppress("not_suppressed") is False

    def test_get_suppressed(self):
        """get_suppressed should return list of suppressed hashes."""
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        wl_hash = groups[0].wl_hash

        assert len(index.get_suppressed()) == 0
        index.suppress(wl_hash)
        assert wl_hash in index.get_suppressed()

    def test_clear_suppressions(self):
        """clear_suppressions should remove all suppressions."""
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        index.suppress(groups[0].wl_hash)

        assert len(index.get_suppressed()) == 1
        index.clear_suppressions()
        assert len(index.get_suppressed()) == 0

    def test_suppression_methods_use_lock(self, monkeypatch):
        """Suppression accessors/mutators should acquire the index lock."""

        class SpyLock:
            def __init__(self) -> None:
                self.enter_count = 0

            def __enter__(self) -> "SpyLock":
                self.enter_count += 1
                return self

            def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
                return None

        index = CodeStructureIndex()
        index.suppressed_hashes.add("abc")

        spy_lock = SpyLock()
        monkeypatch.setattr(index, "_lock", spy_lock)

        assert index.get_suppressed() == ["abc"]
        assert index.get_suppression_info("missing") is None

        index.clear_suppressions()

        assert index.get_suppressed() == []
        assert index.get_suppression_info("abc") is None
        assert spy_lock.enter_count == 5

    def test_clear_preserves_suppressions(self):
        """index.clear() should preserve suppressions."""
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        wl_hash = groups[0].wl_hash
        index.suppress(wl_hash)

        index.clear()

        # Suppressions should still be there
        assert wl_hash in index.get_suppressed()

    def test_suppress_filters_duplicates(self):
        """Suppressed hashes should be excluded from duplicate results."""
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        wl_hash = groups[0].wl_hash
        index.suppress(wl_hash)

        assert wl_hash in index.get_suppressed()
        assert len(index.find_all_duplicates(min_node_count=3)) == 0

    def test_stats_include_suppressed_count(self):
        """Stats should include suppressed_hashes count."""
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        index.suppress(groups[0].wl_hash)

        stats = index.get_stats()
        assert "suppressed_hashes" in stats
        assert stats["suppressed_hashes"] == 1


class TestFileMetadata:
    """Tests for FileMetadata dataclass."""

    def test_to_dict_and_from_dict(self):
        from astrograph.index import FileMetadata

        metadata = FileMetadata(
            file_path="/test/file.py",
            mtime=1234567890.0,
            content_hash="abc123",
            indexed_at=1234567891.0,
            entry_count=5,
        )

        data = metadata.to_dict()
        restored = FileMetadata.from_dict(data)

        assert restored.file_path == metadata.file_path
        assert restored.mtime == metadata.mtime
        assert restored.content_hash == metadata.content_hash
        assert restored.indexed_at == metadata.indexed_at
        assert restored.entry_count == metadata.entry_count


class TestSuppressionInfo:
    """Tests for SuppressionInfo dataclass."""

    def test_to_dict_and_from_dict(self):
        from astrograph.index import SuppressionInfo

        info = SuppressionInfo(
            wl_hash="hash123",
            reason="Idiomatic pattern",
            created_at=1234567890.0,
            source_name="my_func",
            code_preview="def my_func(): pass",
            entry_count=2,
            source_files=["/src/file.py", "/src/file2.py"],
            file_hashes={"/src/file.py": "abc123", "/src/file2.py": "def456"},
        )

        data = info.to_dict()
        restored = SuppressionInfo.from_dict(data)

        assert restored.wl_hash == info.wl_hash
        assert restored.reason == info.reason
        assert restored.created_at == info.created_at
        assert restored.source_files == info.source_files
        assert restored.primary_file == "/src/file.py"
        assert restored.source_name == info.source_name
        assert restored.code_preview == info.code_preview
        assert restored.entry_count == info.entry_count
        assert restored.file_hashes == info.file_hashes

    def test_from_dict_with_missing_optional_fields(self):
        from astrograph.index import SuppressionInfo

        data = {
            "wl_hash": "hash123",
            "created_at": 1234567890.0,
        }
        restored = SuppressionInfo.from_dict(data)

        assert restored.wl_hash == "hash123"
        assert restored.reason is None
        assert restored.primary_file is None
        assert restored.source_files == []
        assert restored.entry_count == 0


class TestFileChangeDetection:
    """Tests for file change detection."""

    def test_compute_file_hash(self, tmp_path):
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        hash1 = index._compute_file_hash(str(file))
        assert hash1 is not None
        assert len(hash1) == 64  # SHA256 hex

        # Same content = same hash
        hash2 = index._compute_file_hash(str(file))
        assert hash1 == hash2

        # Different content = different hash
        file.write_text("def g(): pass")
        hash3 = index._compute_file_hash(str(file))
        assert hash3 != hash1

    def test_compute_file_hash_nonexistent(self):
        index = CodeStructureIndex()
        hash_result = index._compute_file_hash("/nonexistent/file.py")
        assert hash_result is None

    def test_check_file_changed_not_tracked(self):
        index = CodeStructureIndex()
        assert index.check_file_changed("/any/file.py") is True

    def test_check_file_changed_unchanged(self, tmp_path):
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        index.index_file(str(file))

        # File hasn't changed
        assert index.check_file_changed(str(file)) is False

    def test_check_file_changed_modified(self, tmp_path):
        import time

        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        index.index_file(str(file))

        # Modify file
        time.sleep(0.01)  # Ensure mtime changes
        file.write_text("def g(): pass")

        assert index.check_file_changed(str(file)) is True

    def test_check_file_changed_deleted(self, tmp_path):
        import os

        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        index.index_file(str(file))
        os.unlink(str(file))

        assert index.check_file_changed(str(file)) is True

    def test_file_metadata_populated_on_index(self, tmp_path):
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        index.index_file(str(file))

        assert str(file) in index.file_metadata
        metadata = index.file_metadata[str(file)]
        assert metadata.file_path == str(file)
        assert metadata.mtime > 0
        assert len(metadata.content_hash) == 64
        assert metadata.indexed_at > 0
        assert metadata.entry_count == 1

    def test_file_metadata_cleared_on_remove(self, tmp_path):
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        index.index_file(str(file))
        assert str(file) in index.file_metadata

        index.remove_file(str(file))
        assert str(file) not in index.file_metadata


class TestIncrementalIndexing:
    """Tests for incremental indexing."""

    def test_index_file_if_changed_new_file(self, tmp_path):
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        entries, was_changed = index.index_file_if_changed(str(file))

        assert was_changed is True
        assert len(entries) == 1

    def test_index_file_if_changed_unchanged(self, tmp_path):
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        # First index
        index.index_file(str(file))

        # Second index - should be unchanged
        entries, was_changed = index.index_file_if_changed(str(file))

        assert was_changed is False
        assert len(entries) == 0

    def test_index_file_if_changed_modified(self, tmp_path):
        import time

        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        index.index_file(str(file))

        time.sleep(0.01)
        file.write_text("def g(): pass")

        entries, was_changed = index.index_file_if_changed(str(file))

        assert was_changed is True
        assert len(entries) == 1

    def test_index_directory_incremental(self, tmp_path):
        index = CodeStructureIndex()

        file1 = tmp_path / "file1.py"
        file1.write_text("def f(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def g(): pass")

        # Initial index
        (
            entries,
            added,
            updated,
            unchanged,
            changed_files,
            removed_files,
        ) = index.index_directory_incremental(str(tmp_path))

        assert added == 2
        assert updated == 0
        assert unchanged == 0
        assert len(changed_files) == 2
        assert not removed_files

    def test_index_directory_incremental_partial_change(self, tmp_path):
        import time

        index = CodeStructureIndex()

        file1 = tmp_path / "file1.py"
        file1.write_text("def f(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def g(): pass")

        # Initial index
        index.index_directory_incremental(str(tmp_path))

        # Modify only file1
        time.sleep(0.01)
        file1.write_text("def f_modified(): pass")

        (
            entries,
            added,
            updated,
            unchanged,
            changed_files,
            removed_files,
        ) = index.index_directory_incremental(str(tmp_path))

        assert added == 0
        assert updated == 1
        assert unchanged == 1
        assert changed_files == {str(file1)}
        assert not removed_files

    def test_index_directory_incremental_removes_deleted_files(self, tmp_path):
        import os

        index = CodeStructureIndex()

        file1 = tmp_path / "file1.py"
        file1.write_text("def f(): pass")

        file2 = tmp_path / "file2.py"
        file2.write_text("def g(): pass")

        index.index_directory_incremental(str(tmp_path))
        assert len(index.file_metadata) == 2

        # Delete file2
        os.unlink(str(file2))

        (
            entries,
            added,
            updated,
            unchanged,
            changed_files,
            removed_files,
        ) = index.index_directory_incremental(str(tmp_path))

        assert added == 0
        assert updated == 0
        assert unchanged == 1
        assert not changed_files
        assert removed_files == {str(file2)}
        assert len(index.file_metadata) == 1
        assert str(file2) not in index.file_metadata


class TestStalenessReport:
    """Tests for staleness reporting."""

    def test_get_staleness_report_fresh_index(self, tmp_path):
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        index.index_file(str(file))

        report = index.get_staleness_report()

        assert report.is_stale is False
        assert len(report.modified_files) == 0
        assert len(report.deleted_files) == 0

    def test_get_staleness_report_modified_file(self, tmp_path):
        import time

        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        index.index_file(str(file))

        time.sleep(0.01)
        file.write_text("def g(): pass")

        report = index.get_staleness_report()

        assert report.is_stale is True
        assert str(file) in report.modified_files

    def test_get_staleness_report_deleted_file(self, tmp_path):
        import os

        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(): pass")

        index.index_file(str(file))
        os.unlink(str(file))

        report = index.get_staleness_report()

        assert report.is_stale is True
        assert str(file) in report.deleted_files

    def test_get_staleness_report_new_files(self, tmp_path):
        index = CodeStructureIndex()

        file1 = tmp_path / "file1.py"
        file1.write_text("def f(): pass")

        index.index_file(str(file1))

        # Add new file
        file2 = tmp_path / "file2.py"
        file2.write_text("def g(): pass")

        report = index.get_staleness_report(root_path=str(tmp_path))

        assert report.is_stale is True
        assert str(file2) in report.new_files


class TestEnhancedSuppressions:
    """Tests for enhanced suppression functionality."""

    def test_suppress_captures_details(self):
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        wl_hash = groups[0].wl_hash

        index.suppress(wl_hash, reason="Test suppression")

        info = index.get_suppression_info(wl_hash)
        assert info is not None
        assert info.wl_hash == wl_hash
        assert info.reason == "Test suppression"
        assert info.primary_file in ["f1.py", "f2.py"]
        assert info.source_name in ["f1", "f2"]
        assert info.code_preview is not None
        assert info.entry_count == 2

    def test_unsuppress_clears_details(self):
        index = CodeStructureIndex()

        code = "def f(x): return x + 1"
        unit1 = CodeUnit(
            name="f1", code=code, file_path="f1.py", line_start=1, line_end=1, unit_type="function"
        )
        unit2 = CodeUnit(
            name="f2", code=code, file_path="f2.py", line_start=1, line_end=1, unit_type="function"
        )

        index.add_code_unit(unit1)
        index.add_code_unit(unit2)

        groups = index.find_all_duplicates(min_node_count=3)
        wl_hash = groups[0].wl_hash

        index.suppress(wl_hash)
        assert index.get_suppression_info(wl_hash) is not None

        index.unsuppress(wl_hash)
        assert index.get_suppression_info(wl_hash) is None

    def test_cleanup_orphaned_suppressions(self, tmp_path):
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(x): return x + 1\ndef g(y): return y + 1")

        index.index_file(str(file))

        groups = index.find_all_duplicates(min_node_count=3)
        if groups:
            wl_hash = groups[0].wl_hash
            index.suppress(wl_hash)
            assert wl_hash in index.suppressed_hashes

            # Clear index (simulating file removal/change)
            index.clear()

            # invalidate_stale_suppressions consolidates orphan cleanup
            invalidated = index.invalidate_stale_suppressions()
            removed_hashes = [h for h, _ in invalidated]
            assert wl_hash in removed_hashes
            assert wl_hash not in index.suppressed_hashes

    def test_check_suppression_staleness(self, tmp_path):
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(x): return x + 1\ndef g(y): return y + 1")

        index.index_file(str(file))

        groups = index.find_all_duplicates(min_node_count=3)
        if groups:
            wl_hash = groups[0].wl_hash
            index.suppress(wl_hash)

            # Clear index to make suppression stale
            index.clear()

            stale = index.check_suppression_staleness()
            assert any(wl_hash in s for s in stale)

    def test_invalidate_modified_suppressions_structure_unchanged(self, tmp_path):
        """Test that suppressions are NOT invalidated when file changes but structure is same."""
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(x): return x + 1\ndef g(y): return y + 1")

        index.index_file(str(file))

        groups = index.find_all_duplicates(min_node_count=3)
        assert groups, "Should have duplicates"

        wl_hash = groups[0].wl_hash
        index.suppress(wl_hash)

        # Verify suppression was created with file tracking
        info = index.get_suppression_info(wl_hash)
        assert info is not None
        assert str(file) in info.source_files
        assert str(file) in info.file_hashes
        original_file_hash = info.file_hashes[str(file)]

        # No invalidation when file unchanged
        invalidated = index.invalidate_stale_suppressions()
        assert len(invalidated) == 0
        assert wl_hash in index.suppressed_hashes

        # Add new code to the file (but don't change the suppressed structures)
        file.write_text("def f(x): return x + 1\ndef g(y): return y + 1\ndef h(z): return z * 2")

        # Suppression should NOT be invalidated - structure still exists
        invalidated = index.invalidate_stale_suppressions()
        assert len(invalidated) == 0
        assert wl_hash in index.suppressed_hashes

        # File hash should be updated since file changed
        info = index.get_suppression_info(wl_hash)
        assert info.file_hashes[str(file)] != original_file_hash

    def test_invalidate_modified_suppressions_structure_changed(self, tmp_path):
        """Test that suppressions ARE invalidated when suppressed code structure changes."""
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(x): return x + 1\ndef g(y): return y + 1")

        index.index_file(str(file))

        groups = index.find_all_duplicates(min_node_count=3)
        assert groups, "Should have duplicates"

        wl_hash = groups[0].wl_hash
        index.suppress(wl_hash)

        # Modify the suppressed code structure (different operations)
        file.write_text("def f(x): return x * 2\ndef g(y): return y - 3")

        # NOW invalidation should occur - structure changed
        invalidated = index.invalidate_stale_suppressions()
        assert len(invalidated) == 1
        assert invalidated[0][0] == wl_hash
        assert "no longer exists" in invalidated[0][1]
        assert wl_hash not in index.suppressed_hashes

    def test_invalidate_deleted_file_suppressions(self, tmp_path):
        """Test that suppressions are invalidated when ALL source files are deleted."""
        index = CodeStructureIndex()

        file = tmp_path / "test.py"
        file.write_text("def f(x): return x + 1\ndef g(y): return y + 1")

        index.index_file(str(file))

        groups = index.find_all_duplicates(min_node_count=3)
        assert groups, "Should have duplicates"

        wl_hash = groups[0].wl_hash
        index.suppress(wl_hash)

        # Delete the file - structure no longer exists anywhere
        file.unlink()

        # Invalidation should occur since structure can't be found
        invalidated = index.invalidate_stale_suppressions()
        assert len(invalidated) == 1
        assert invalidated[0][0] == wl_hash
        assert "no longer exists" in invalidated[0][1]
        assert wl_hash not in index.suppressed_hashes

    def test_suppression_survives_partial_file_deletion(self, tmp_path):
        """Test that suppression remains if structure exists in other files."""
        index = CodeStructureIndex()

        file1 = tmp_path / "test1.py"
        file2 = tmp_path / "test2.py"
        file1.write_text("def f(x): return x + 1")
        file2.write_text("def g(y): return y + 1")  # Same structure

        index.index_file(str(file1))
        index.index_file(str(file2))

        groups = index.find_all_duplicates(min_node_count=3)
        assert groups, "Should have duplicates"

        wl_hash = groups[0].wl_hash
        index.suppress(wl_hash)

        # Delete one file - structure still exists in the other
        file1.unlink()

        # Suppression should NOT be invalidated
        invalidated = index.invalidate_stale_suppressions()
        assert len(invalidated) == 0
        assert wl_hash in index.suppressed_hashes

        # file1 should be removed from tracking
        info = index.get_suppression_info(wl_hash)
        assert str(file1) not in info.source_files
        assert str(file2) in info.source_files


class TestEntryFilter:
    """Tests for the entry_filter parameter on find_*_duplicates methods."""

    def test_find_all_duplicates_with_entry_filter(self, tmp_path):
        """entry_filter restricts which entries appear in duplicate groups."""
        index = CodeStructureIndex()
        # Create two pairs of duplicates in different directories
        src_dir = tmp_path / "src"
        lib_dir = tmp_path / "lib"
        src_dir.mkdir()
        lib_dir.mkdir()

        code = "def calc(a, b):\n    result = a + b\n    return result * 2\n"
        (src_dir / "a.py").write_text(code)
        (src_dir / "b.py").write_text(code)
        (lib_dir / "a.py").write_text(code)

        index.index_file(str(src_dir / "a.py"))
        index.index_file(str(src_dir / "b.py"))
        index.index_file(str(lib_dir / "a.py"))

        # Without filter: all 3 entries in one group
        all_groups = index.find_all_duplicates()
        if all_groups:
            total_entries = sum(len(g.entries) for g in all_groups)
            assert total_entries >= 3

        # With filter: only src/ entries
        def src_only(entry: IndexEntry) -> bool:
            return "/src/" in entry.code_unit.file_path

        filtered = index.find_all_duplicates(entry_filter=src_only)
        if filtered:
            for g in filtered:
                for e in g.entries:
                    assert "/src/" in e.code_unit.file_path

    def test_find_all_duplicates_filter_too_strict_returns_empty(self, tmp_path):
        """If filter removes too many entries, no groups are returned."""
        index = CodeStructureIndex()
        code = "def calc(a, b):\n    result = a + b\n    return result * 2\n"
        (tmp_path / "a.py").write_text(code)
        (tmp_path / "b.py").write_text(code)
        index.index_file(str(tmp_path / "a.py"))
        index.index_file(str(tmp_path / "b.py"))

        # Filter that matches nothing
        groups = index.find_all_duplicates(entry_filter=lambda e: False)
        assert groups == []

    def test_find_pattern_duplicates_with_entry_filter(self, tmp_path):
        """entry_filter works on pattern duplicates too."""
        index = CodeStructureIndex()
        (tmp_path / "add.py").write_text("def f(a, b):\n    return a + b\n")
        (tmp_path / "mul.py").write_text("def f(a, b):\n    return a * b\n")
        index.index_file(str(tmp_path / "add.py"))
        index.index_file(str(tmp_path / "mul.py"))

        # Filter that matches nothing — no groups
        groups = index.find_pattern_duplicates(entry_filter=lambda e: False)
        assert groups == []

    def test_find_block_duplicates_composes_filters(self, tmp_path):
        """block_types filter composes with entry_filter."""
        index = CodeStructureIndex()
        code = (
            "def func():\n"
            "    for i in range(10):\n"
            "        x = i * 2\n"
            "        y = x + 1\n"
            "        print(y)\n"
        )
        (tmp_path / "a.py").write_text(code)
        (tmp_path / "b.py").write_text(code)
        index.index_file(str(tmp_path / "a.py"), include_blocks=True)
        index.index_file(str(tmp_path / "b.py"), include_blocks=True)

        # Filter that rejects everything — no block groups even with valid block_types
        groups = index.find_block_duplicates(
            block_types=["for"], entry_filter=lambda e: False
        )
        assert groups == []
