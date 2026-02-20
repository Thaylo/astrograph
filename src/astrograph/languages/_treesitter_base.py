"""
Tree-sitter based language plugin base class.

Provides a generic implementation for languages using tree-sitter parsers.
Contributors implementing a new language override ~5 small methods.

Requires optional dependency: pip install astrograph[treesitter]
"""

import textwrap
from abc import abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

import networkx as nx

from .base import BaseLanguagePlugin, CodeUnit

try:
    import tree_sitter

    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False
    tree_sitter = None


class TreeSitterLanguagePlugin(BaseLanguagePlugin):
    """
    Base class for tree-sitter based language plugins.

    Subclasses must implement:
    - language_id (property)
    - file_extensions (property)
    - skip_dirs (property)
    - _tree_sitter_language() -> the tree-sitter Language object
    - _node_label(node, normalize_ops) -> structural label for a CST node
    - _is_function_node(node) -> True if node represents a function/method
    - _is_class_node(node) -> True if node represents a class
    - _get_name(node) -> extract the name from a function/class node

    Optionally override:
    - _is_block_node(node) -> True if node represents a block (for/while/if/etc)
    - _should_skip_node(node) -> True to skip noise nodes (punctuation, etc)
    - _get_block_type(node) -> string block type name

    Example:
        class JavaScriptPlugin(TreeSitterLanguagePlugin):
            language_id = "javascript"
            file_extensions = frozenset({".js", ".mjs", ".cjs"})
            skip_dirs = frozenset({"node_modules", "dist", "build"})

            def _tree_sitter_language(self):
                import tree_sitter_javascript
                return tree_sitter_javascript.language()

            def _node_label(self, node, normalize_ops):
                return node.type

            def _is_function_node(self, node):
                return node.type in ("function_declaration", "arrow_function")

            def _is_class_node(self, node):
                return node.type == "class_declaration"

            def _get_name(self, node):
                for child in node.children:
                    if child.type == "identifier":
                        return child.text.decode()
                return "<anonymous>"
    """

    _parser: Any = None

    def _get_parser(self) -> Any:
        """Get or create the tree-sitter parser for this language."""
        if not HAS_TREE_SITTER:
            raise ImportError(
                "tree-sitter is required for this language plugin. "
                "Install with: pip install astrograph[treesitter]"
            )

        if self._parser is None:
            self._parser = tree_sitter.Parser(self._tree_sitter_language())
        return self._parser

    @abstractmethod
    def _tree_sitter_language(self) -> Any:
        """Return the tree-sitter Language object for this language."""
        raise NotImplementedError("Subclasses must provide a tree-sitter Language object")

    @abstractmethod
    def _node_label(self, node: Any, normalize_ops: bool = False) -> str:
        """Get structural label for a CST node."""
        ...

    @abstractmethod
    def _is_function_node(self, node: Any) -> bool:
        """Check if a node represents a function/method definition."""
        raise NotImplementedError("Subclasses must identify function nodes")

    @abstractmethod
    def _is_class_node(self, node: Any) -> bool:
        """Check if a node represents a class definition."""
        node_type = getattr(node, "type", "<unknown>")
        raise NotImplementedError(f"Subclasses must identify class nodes (node.type={node_type!r})")

    @abstractmethod
    def _get_name(self, node: Any) -> str:
        """Extract the name from a function/class/block node."""
        node_text = getattr(node, "text", b"")
        raise NotImplementedError(
            f"Subclasses must extract declaration names (text_len={len(node_text)})"
        )

    def _is_block_node(self, node: Any) -> bool:
        """Check if a node represents a block (for/while/if/try/with).

        Override this to enable block-level duplicate detection.
        Default: no block extraction.
        """
        return False

    def _get_block_type(self, node: Any) -> str:
        """Get the block type name (e.g., 'for', 'while', 'if').

        Default: uses the node type directly.
        """
        return str(node.type)

    def _should_skip_node(self, node: Any) -> bool:
        """Check if a node should be skipped in graph construction.

        Override to filter noise nodes like punctuation, whitespace, etc.
        Default: skip nodes with no children whose type is a single punctuation char.
        """
        return len(node.children) == 0 and len(node.type) == 1 and not node.type.isalnum()

    def source_to_graph(self, source: str, normalize_ops: bool = False) -> nx.DiGraph:
        """Convert source code to a labeled directed graph using tree-sitter."""
        parser = self._get_parser()
        tree = parser.parse(source.encode("utf-8"))

        graph = nx.DiGraph()
        node_counter = 0

        def add_node_recursive(ts_node: Any, parent_id: int | None = None) -> int:
            nonlocal node_counter

            def add_children(next_parent: int | None) -> None:
                for child in ts_node.children:
                    add_node_recursive(child, next_parent)

            if self._should_skip_node(ts_node):
                # Still process children even if skipping this node
                add_children(parent_id)
                return -1

            current_id = node_counter
            node_counter += 1

            label = self._node_label(ts_node, normalize_ops)
            graph.add_node(current_id, label=label)

            if parent_id is None:
                pass
            else:
                graph.add_edge(parent_id, current_id)

            add_children(current_id)

            return current_id

        add_node_recursive(tree.root_node)
        return graph

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract functions, methods, classes, and blocks from source code."""
        parser = self._get_parser()
        tree = parser.parse(source.encode("utf-8"))
        source_lines = source.splitlines()

        method_locations: set[tuple[int, int]] = set()

        def _walk_children(
            node: Any, walker: Callable[[Any], Iterator[CodeUnit]]
        ) -> Iterator[CodeUnit]:
            for child in node.children:
                yield from walker(child)

        # Walk the tree to find classes and their methods
        def walk(node: Any) -> Iterator[CodeUnit]:
            if self._is_class_node(node):
                name = self._get_name(node)
                start = node.start_point[0]
                end = node.end_point[0] + 1
                code = textwrap.dedent("\n".join(source_lines[start:end]))

                yield CodeUnit(
                    name=name,
                    code=code,
                    file_path=file_path,
                    line_start=start + 1,
                    line_end=end,
                    unit_type="class",
                    parent_name=None,
                    language=self.language_id,
                )

                # Extract methods from the class
                for child in node.children:
                    if self._is_function_node(child):
                        m_name = self._get_name(child)
                        m_start = child.start_point[0]
                        m_end = child.end_point[0] + 1
                        m_code = textwrap.dedent("\n".join(source_lines[m_start:m_end]))
                        method_locations.add((m_start + 1, m_end))

                        yield CodeUnit(
                            name=m_name,
                            code=m_code,
                            file_path=file_path,
                            line_start=m_start + 1,
                            line_end=m_end,
                            unit_type="method",
                            parent_name=name,
                            language=self.language_id,
                        )

            yield from _walk_children(node, walk)

        yield from walk(tree.root_node)

        # Second pass: standalone functions (not methods)
        def walk_functions(node: Any) -> Iterator[CodeUnit]:
            if self._is_function_node(node):
                start = node.start_point[0]
                end = node.end_point[0] + 1
                if (start + 1, end) not in method_locations:
                    name = self._get_name(node)
                    code = textwrap.dedent("\n".join(source_lines[start:end]))

                    yield CodeUnit(
                        name=name,
                        code=code,
                        file_path=file_path,
                        line_start=start + 1,
                        line_end=end,
                        unit_type="function",
                        parent_name=None,
                        language=self.language_id,
                    )

            yield from _walk_children(node, walk_functions)

        yield from walk_functions(tree.root_node)
