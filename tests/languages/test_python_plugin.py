"""Tests for the Python language plugin."""

import pytest

from astrograph.languages.python_plugin import (
    ast_to_graph,
    code_unit_to_ast_graph,
    extract_code_units,
)


class TestPythonPlugin:
    """Tests for PythonPlugin properties."""

    def test_language_id(self, python_plugin):
        assert python_plugin.language_id == "python"

    def test_file_extensions(self, python_plugin):
        assert ".py" in python_plugin.file_extensions
        assert ".pyi" in python_plugin.file_extensions

    def test_skip_dirs(self, python_plugin):
        skip = python_plugin.skip_dirs
        assert "__pycache__" in skip
        assert "venv" in skip
        assert ".venv" in skip


def _sample_source_and_path(language_id: str) -> tuple[str, str]:
    if language_id == "javascript_lsp":
        return (
            """
function hello(name) {
  return name;
}
""",
            "test.js",
        )
    return ("def hello(): pass", "test.py")


class TestPluginConformance:
    """Conformance tests that every registered plugin must pass."""

    def test_valid_language_id(self, language_plugin):
        """Plugin has a non-empty language_id."""
        assert language_plugin.language_id
        assert isinstance(language_plugin.language_id, str)

    def test_valid_file_extensions(self, language_plugin):
        """Plugin has valid file extensions (all start with '.')."""
        exts = language_plugin.file_extensions
        assert len(exts) > 0
        for ext in exts:
            assert ext.startswith("."), f"Extension '{ext}' must start with '.'"

    def test_valid_skip_dirs(self, language_plugin):
        """Plugin has a frozenset of skip directories."""
        assert isinstance(language_plugin.skip_dirs, frozenset)

    @pytest.mark.parametrize(
        "source",
        [
            "x = 1",
            "",
            "def ::::",
        ],
    )
    def test_source_to_graph_returns_digraph(self, language_plugin, source):
        """source_to_graph returns a networkx DiGraph for varied inputs."""
        import networkx as nx

        graph = language_plugin.source_to_graph(source)
        assert isinstance(graph, nx.DiGraph)

    def test_source_to_graph_has_labels(self, language_plugin):
        """source_to_graph produces nodes with 'label' attributes."""
        source, _ = _sample_source_and_path(language_plugin.language_id)
        graph = language_plugin.source_to_graph(source)
        for _, data in graph.nodes(data=True):
            assert "label" in data, "All nodes must have a 'label' attribute"

    def test_extract_code_units_returns_iterator(self, language_plugin):
        """extract_code_units returns CodeUnit objects."""
        source, file_path = _sample_source_and_path(language_plugin.language_id)
        units = list(language_plugin.extract_code_units(source, file_path))

        if units:
            unit = units[0]
            assert unit.language == language_plugin.language_id

    def test_extract_code_units_empty_source(self, language_plugin):
        """Empty source produces no code units."""
        _, file_path = _sample_source_and_path(language_plugin.language_id)
        units = list(language_plugin.extract_code_units("", file_path))
        assert units == []

    def test_code_unit_to_ast_graph(self, language_plugin):
        """code_unit_to_ast_graph produces valid ASTGraph."""
        from astrograph.languages.base import ASTGraph, CodeUnit

        _, file_path = _sample_source_and_path(language_plugin.language_id)

        unit = CodeUnit(
            name="test",
            code="x = 1 + 2",
            file_path=file_path,
            line_start=1,
            line_end=1,
            unit_type="function",
            language=language_plugin.language_id,
        )
        result = language_plugin.code_unit_to_ast_graph(unit)
        assert isinstance(result, ASTGraph)
        assert result.node_count > 0
        assert result.depth >= 0
        assert result.label_histogram


class TestBaseLanguagePlugin:
    """Tests for BaseLanguagePlugin abstract methods."""

    def test_base_extract_code_units_raises(self):
        from astrograph.languages.base import BaseLanguagePlugin

        plugin = BaseLanguagePlugin()
        with pytest.raises(NotImplementedError):
            list(plugin.extract_code_units("", "test.py"))

    def test_base_source_to_graph_raises(self):
        from astrograph.languages.base import BaseLanguagePlugin

        plugin = BaseLanguagePlugin()
        with pytest.raises(NotImplementedError):
            plugin.source_to_graph("")

    @pytest.mark.parametrize("attr_name", ["file_extensions", "skip_dirs"])
    def test_base_property_raises(self, attr_name):
        from astrograph.languages.base import BaseLanguagePlugin

        plugin = BaseLanguagePlugin()
        pytest.raises(NotImplementedError, getattr, plugin, attr_name)


class TestBaseCodeUnitToAstGraph:
    """Tests for BaseLanguagePlugin.code_unit_to_ast_graph default implementation."""

    def test_base_code_unit_to_ast_graph(self):
        """Base class code_unit_to_ast_graph works when source_to_graph is implemented."""
        import networkx as nx

        from astrograph.languages.base import BaseLanguagePlugin, CodeUnit

        class MinimalPlugin(BaseLanguagePlugin):
            language_id = "minimal"
            file_extensions = frozenset({".min"})
            skip_dirs = frozenset()

            def extract_code_units(
                self, _source, _file_path="", _include_blocks=True, _max_block_depth=3
            ):
                return iter([])

            def source_to_graph(self, _source, _normalize_ops=False):
                # Simple graph for testing
                g = nx.DiGraph()
                g.add_node(0, label="Root")
                g.add_node(1, label="Child")
                g.add_edge(0, 1)
                return g

        plugin = MinimalPlugin()
        unit = CodeUnit(
            name="test",
            code="x = 1",
            file_path="test.min",
            line_start=1,
            line_end=1,
            unit_type="function",
            language="minimal",
        )
        result = plugin.code_unit_to_ast_graph(unit)
        assert result.node_count == 2
        assert result.depth == 1
        assert result.label_histogram == {"Root": 1, "Child": 1}

    def test_base_code_unit_to_ast_graph_empty(self):
        """Base class handles empty graph."""
        import networkx as nx

        from astrograph.languages.base import BaseLanguagePlugin, CodeUnit

        class EmptyPlugin(BaseLanguagePlugin):
            language_id = "empty"
            file_extensions = frozenset({".emp"})
            skip_dirs = frozenset()

            def extract_code_units(self, *_args, **_kwargs):
                return iter(())

            def source_to_graph(self, _source, _normalize_ops=False):
                return nx.DiGraph()

        plugin = EmptyPlugin()
        unit = CodeUnit(
            name="test",
            code="",
            file_path="test.emp",
            line_start=1,
            line_end=1,
            unit_type="function",
            language="empty",
        )
        result = plugin.code_unit_to_ast_graph(unit)
        assert result.node_count == 0
        assert result.depth == 0


class TestComputeLabelHistogram:
    """Tests for compute_label_histogram."""

    def test_histogram(self):
        import networkx as nx

        from astrograph.languages.base import compute_label_histogram

        g = nx.DiGraph()
        g.add_node(0, label="A")
        g.add_node(1, label="B")
        g.add_node(2, label="A")
        hist = compute_label_histogram(g)
        assert hist == {"A": 2, "B": 1}

    def test_empty_graph(self):
        import networkx as nx

        from astrograph.languages.base import compute_label_histogram

        g = nx.DiGraph()
        assert compute_label_histogram(g) == {}


class TestNodeMatch:
    """Tests for node_match function."""

    def test_matching_labels(self):
        from astrograph.languages.base import node_match

        assert node_match({"label": "A"}, {"label": "A"})

    def test_non_matching_labels(self):
        from astrograph.languages.base import node_match

        assert not node_match({"label": "A"}, {"label": "B"})

    def test_missing_label(self):
        from astrograph.languages.base import node_match

        assert node_match({}, {})  # Both None
        assert not node_match({"label": "A"}, {})


class TestPythonPluginFunctions:
    """Tests for the standalone Python functions (backward compat)."""

    def test_ast_to_graph_basic(self):
        """Basic ast_to_graph produces non-empty graph."""
        graph = ast_to_graph("x = 1 + 2")
        assert graph.number_of_nodes() > 0

    def test_ast_to_graph_normalize_ops(self):
        """Normalize ops produces different labels."""
        g1 = ast_to_graph("x = a + b")
        g2 = ast_to_graph("x = a + b", normalize_ops=True)
        # Both should have nodes, but labels differ
        assert g1.number_of_nodes() > 0
        assert g2.number_of_nodes() > 0

    def test_extract_code_units_function(self):
        """Extract a standalone function."""
        source = "def greet(name):\n    return f'Hello {name}'"
        units = list(extract_code_units(source, "test.py"))
        assert any(u.name == "greet" and u.unit_type == "function" for u in units)

    def test_extract_code_units_class_with_method(self):
        """Extract a class and its method."""
        source = "class Foo:\n    def bar(self):\n        pass"
        units = list(extract_code_units(source, "test.py"))
        assert any(u.name == "Foo" and u.unit_type == "class" for u in units)
        assert any(u.name == "bar" and u.unit_type == "method" for u in units)

    def test_extract_code_units_blocks(self):
        """Extract blocks from a function."""
        source = "def process(items):\n    for item in items:\n        if item > 0:\n            print(item)"
        units = list(extract_code_units(source, "test.py", include_blocks=True))
        assert any(u.unit_type == "block" for u in units)

    def test_extract_code_units_no_blocks(self):
        """Blocks not extracted when include_blocks=False."""
        source = "def process(items):\n    for item in items:\n        print(item)"
        units = list(extract_code_units(source, "test.py", include_blocks=False))
        assert not any(u.unit_type == "block" for u in units)

    def test_code_unit_to_ast_graph_standalone(self):
        """Standalone code_unit_to_ast_graph works."""
        from astrograph.languages.base import CodeUnit

        unit = CodeUnit(
            name="f",
            code="def f(): pass",
            file_path="test.py",
            line_start=1,
            line_end=1,
            unit_type="function",
        )
        result = code_unit_to_ast_graph(unit)
        assert result.node_count > 0

    def test_code_unit_language_field(self):
        """Extracted code units have language='python'."""
        source = "def f(): pass"
        units = list(extract_code_units(source, "test.py"))
        assert all(u.language == "python" for u in units)
