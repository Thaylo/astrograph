"""
Language plugin interface and shared data structures.

Defines the LanguagePlugin protocol that all language implementations must satisfy,
plus language-agnostic data structures (CodeUnit, ASTGraph) used throughout the system.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import networkx as nx


@dataclass
class CodeUnit:
    """A parseable unit of code (function, class, method, or block)."""

    name: str
    code: str
    file_path: str
    line_start: int
    line_end: int
    unit_type: str  # 'function', 'class', 'method', 'module', 'block'
    parent_name: str | None = None
    # Block-specific fields
    block_type: str | None = None  # 'for', 'while', 'if', 'try', 'with'
    nesting_depth: int = 0  # 0 for functions, 1+ for nested blocks
    parent_block_name: str | None = None  # Full hierarchical name of parent block
    language: str = "python"  # Language identifier


@dataclass
class ASTGraph:
    """A graph representation of an AST with metadata."""

    graph: nx.DiGraph
    code_unit: CodeUnit
    node_count: int
    depth: int
    label_histogram: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticSignal:
    """Single semantic fact extracted from code."""

    key: str
    value: str
    confidence: float = 1.0
    origin: str = "syntax"


@dataclass(frozen=True)
class SemanticProfile:
    """Best-effort semantic profile for a source snippet."""

    signals: tuple[SemanticSignal, ...] = ()
    coverage: float = 0.0
    notes: tuple[str, ...] = ()
    extractor: str = "none"

    def signal_map(self) -> dict[str, SemanticSignal]:
        """Return signals keyed by name (last writer wins)."""
        mapped: dict[str, SemanticSignal] = {}
        for signal in self.signals:
            mapped[signal.key] = signal
        return mapped


def compute_label_histogram(graph: nx.DiGraph) -> dict[str, int]:
    """Compute histogram of node labels in a graph."""
    histogram: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        label = data.get("label", "Unknown")
        histogram[label] = histogram.get(label, 0) + 1
    return histogram


def node_match(n1_attrs: dict, n2_attrs: dict) -> bool:
    """Check if two graph nodes match by their label attribute.

    Used for graph isomorphism checking with NetworkX.
    """
    return n1_attrs.get("label") == n2_attrs.get("label")


def build_ast_graph(graph: nx.DiGraph, unit: CodeUnit) -> ASTGraph:
    """Build an ASTGraph from a pre-computed graph and code unit.

    Shared logic for computing depth, label histogram, and assembling metadata.
    Used by BaseLanguagePlugin and standalone code_unit_to_ast_graph functions.
    """
    label_histogram = compute_label_histogram(graph)

    if graph.number_of_nodes() == 0:
        depth = 0
    else:
        roots = [n for n in graph.nodes() if graph.in_degree(n) == 0]
        if roots:
            depth = 0
            for root in roots:
                lengths = nx.single_source_shortest_path_length(graph, root)
                if lengths:
                    depth = max(depth, max(lengths.values()))
        else:
            depth = 0

    return ASTGraph(
        graph=graph,
        code_unit=unit,
        node_count=graph.number_of_nodes(),
        depth=depth,
        label_histogram=label_histogram,
    )


@runtime_checkable
class LanguagePlugin(Protocol):
    """Protocol defining what a language plugin must implement."""

    @property
    def language_id(self) -> str:
        """Unique identifier for this language (e.g., 'python', 'javascript')."""
        ...

    @property
    def file_extensions(self) -> frozenset[str]:
        """File extensions handled by this plugin (e.g., frozenset({'.py', '.pyi'}))."""
        ...

    @property
    def skip_dirs(self) -> frozenset[str]:
        """Language-specific directories to skip during indexing."""
        raise NotImplementedError

    def extract_code_units(
        self,
        source: str,
        file_path: str,
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract functions, methods, classes, and blocks from source code."""
        ...

    def source_to_graph(
        self,
        source: str,
        normalize_ops: bool = False,
    ) -> nx.DiGraph:
        """Convert source code to a labeled directed graph."""
        ...

    def code_unit_to_ast_graph(self, unit: CodeUnit) -> ASTGraph:
        """Convert a CodeUnit to an ASTGraph with metadata."""
        ...

    def normalize_graph_for_pattern(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Normalize node labels for pattern matching (operators → base types).

        Returns a copy of the graph with operator-specific labels replaced by
        base type labels. Default: returns the graph unchanged.
        """
        ...

    def extract_semantic_profile(
        self,
        source: str,
        file_path: str = "<unknown>",
    ) -> SemanticProfile:
        """Extract a best-effort semantic profile (types/operators/etc)."""
        ...


class BaseLanguagePlugin:
    """
    Base class for language plugins with shared logic.

    Subclasses must implement:
    - language_id (property)
    - file_extensions (property)
    - skip_dirs (property)
    - extract_code_units()
    - source_to_graph()

    Provides a default code_unit_to_ast_graph() implementation.
    """

    @property
    def language_id(self) -> str:
        raise NotImplementedError

    @property
    def file_extensions(self) -> frozenset[str]:
        raise NotImplementedError

    @property
    def skip_dirs(self) -> frozenset[str]:
        raise NotImplementedError("Language plugins must define skip_dirs")

    def extract_code_units(
        self,
        source: str,
        file_path: str,
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        raise NotImplementedError

    def source_to_graph(
        self,
        source: str,
        normalize_ops: bool = False,
    ) -> nx.DiGraph:
        raise NotImplementedError

    def code_unit_to_ast_graph(self, unit: CodeUnit) -> ASTGraph:
        """Convert a CodeUnit to an ASTGraph with metadata."""
        return build_ast_graph(self.source_to_graph(unit.code), unit)

    def normalize_graph_for_pattern(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Normalize node labels for pattern matching. Default: return unchanged."""
        return graph

    def extract_semantic_profile(
        self,
        source: str,
        file_path: str = "<unknown>",
    ) -> SemanticProfile:
        """Extract semantic facts from source. Default: no semantic signal."""
        del source, file_path
        return SemanticProfile()
