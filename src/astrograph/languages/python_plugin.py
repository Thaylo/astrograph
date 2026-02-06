"""
Python language plugin using stdlib ast module.

Extracts functions, methods, classes, and code blocks from Python source,
and converts them to labeled directed graphs for structural comparison.
"""

import ast
import textwrap
from collections.abc import Iterator

import networkx as nx

from .base import ASTGraph, BaseLanguagePlugin, CodeUnit, compute_label_histogram

# Block types to extract from functions
BLOCK_TYPES = (ast.For, ast.While, ast.If, ast.Try, ast.With, ast.AsyncFor, ast.AsyncWith)
BLOCK_TYPE_NAMES = {
    ast.For: "for",
    ast.While: "while",
    ast.If: "if",
    ast.Try: "try",
    ast.With: "with",
    ast.AsyncFor: "async_for",
    ast.AsyncWith: "async_with",
}


def _extract_blocks_recursive(
    node: ast.AST,
    source_lines: list[str],
    file_path: str,
    func_name: str,
    parent_block_name: str,
    current_depth: int,
    max_depth: int,
    block_counters: dict[str, int],
) -> Iterator[CodeUnit]:
    """Recursively extract blocks from an AST node up to max_depth."""
    if current_depth > max_depth:
        return

    for child in ast.iter_child_nodes(node):
        if isinstance(child, BLOCK_TYPES):
            block_type = BLOCK_TYPE_NAMES[type(child)]

            # Generate unique name for this block type at this level
            counter_key = f"{parent_block_name}.{block_type}"
            if counter_key not in block_counters:
                block_counters[counter_key] = 0
            block_counters[counter_key] += 1
            block_num = block_counters[counter_key]

            # Hierarchical name: func.for_1 or func.for_1.if_1
            if parent_block_name == func_name:
                block_name = f"{func_name}.{block_type}_{block_num}"
            else:
                block_name = f"{parent_block_name}.{block_type}_{block_num}"

            start = child.lineno - 1
            end = child.end_lineno if child.end_lineno else start + 1
            code = textwrap.dedent("\n".join(source_lines[start:end]))

            yield CodeUnit(
                name=block_name,
                code=code,
                file_path=file_path,
                line_start=child.lineno,
                line_end=end,
                unit_type="block",
                parent_name=func_name,
                block_type=block_type,
                nesting_depth=current_depth,
                parent_block_name=parent_block_name if parent_block_name != func_name else None,
                language="python",
            )

            # Recursively extract nested blocks
            yield from _extract_blocks_recursive(
                child,
                source_lines,
                file_path,
                func_name,
                block_name,
                current_depth + 1,
                max_depth,
                block_counters,
            )


def extract_blocks_from_function(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    file_path: str,
    max_depth: int = 3,
) -> Iterator[CodeUnit]:
    """
    Extract code blocks (for, while, if, try, with) from a function.

    Args:
        func_node: The function AST node
        source_lines: All source lines from the file
        file_path: Path to the source file
        max_depth: Maximum nesting depth to extract (default 3)

    Yields:
        CodeUnit objects for each block found
    """
    func_name = func_node.name
    block_counters: dict[str, int] = {}

    yield from _extract_blocks_recursive(
        func_node,
        source_lines,
        file_path,
        func_name,
        func_name,
        1,  # Start at depth 1 (function body is depth 0)
        max_depth,
        block_counters,
    )


def _get_node_label(node: ast.AST, normalize_ops: bool = False) -> str:
    """
    Get structural label for AST node, ignoring specific names/values.

    Args:
        node: The AST node to label
        normalize_ops: If True, normalize all operators to base types
                      (e.g., BinOp:Add -> BinOp, Compare:Eq -> Compare)
                      This enables "pattern matching" that ignores operator differences.
    """
    label = node.__class__.__name__

    if normalize_ops:
        # Pattern mode: ignore specific operators, keep only node types
        # This makes "a + b" and "a * b" structurally equivalent

        # Normalize binary operators (Add, Sub, Mult, etc.) to "BinaryOp"
        if isinstance(
            node,
            ast.Add
            | ast.Sub
            | ast.Mult
            | ast.Div
            | ast.FloorDiv
            | ast.Mod
            | ast.Pow
            | ast.LShift
            | ast.RShift
            | ast.BitOr
            | ast.BitXor
            | ast.BitAnd
            | ast.MatMult,
        ):
            return "BinaryOp"

        # Normalize comparison operators (Eq, NotEq, Lt, Gt, etc.) to "CmpOp"
        if isinstance(
            node,
            ast.Eq
            | ast.NotEq
            | ast.Lt
            | ast.LtE
            | ast.Gt
            | ast.GtE
            | ast.Is
            | ast.IsNot
            | ast.In
            | ast.NotIn,
        ):
            return "CmpOp"

        # Normalize unary operators (UAdd, USub, Not, Invert) to "UnaryOp"
        if isinstance(node, ast.UAdd | ast.USub | ast.Not | ast.Invert):
            return "UnaryOp"

        # Normalize boolean operators (And, Or) to "BoolOp"
        if isinstance(node, ast.And | ast.Or):
            return "BoolOperator"

        if isinstance(node, ast.Constant):
            label += f":{type(node.value).__name__}"
        # All other nodes just use the class name
        return label

    # Precise mode: include operator details for exact matching
    if isinstance(node, ast.BinOp | ast.UnaryOp | ast.BoolOp):
        label += f":{node.op.__class__.__name__}"
    elif isinstance(node, ast.Compare):
        ops = "_".join(op.__class__.__name__ for op in node.ops)
        label += f":{ops}"
    elif isinstance(node, ast.AugAssign):
        label += f":{node.op.__class__.__name__}"
    elif isinstance(node, ast.Constant):
        # Include type but not value
        label += f":{type(node.value).__name__}"

    return label


def ast_to_graph(source: str, normalize_ops: bool = False) -> nx.DiGraph:
    """
    Convert Python source code to a directed graph.

    Nodes are labeled by AST node type (structure only, ignoring names/values).
    Edges represent parent-child relationships in the AST.

    Args:
        source: Python source code
        normalize_ops: If True, normalize operators for pattern matching
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return nx.DiGraph()

    graph = nx.DiGraph()
    node_counter = 0

    def add_node_recursive(node: ast.AST, parent_id: int | None = None) -> int:
        nonlocal node_counter

        current_id = node_counter
        node_counter += 1

        label = _get_node_label(node, normalize_ops=normalize_ops)
        graph.add_node(current_id, label=label)

        if parent_id is not None:
            graph.add_edge(parent_id, current_id)

        # Process children in order
        for child in ast.iter_child_nodes(node):
            add_node_recursive(child, current_id)

        return current_id

    add_node_recursive(tree)
    return graph


def extract_code_units(
    source: str,
    file_path: str = "<unknown>",
    include_blocks: bool = True,
    max_block_depth: int = 3,
) -> Iterator[CodeUnit]:
    """
    Extract functions, methods, and classes from Python source code.

    Args:
        source: Python source code
        file_path: Path to the source file
        include_blocks: If True, also extract code blocks (for, while, if, try, with)
        max_block_depth: Maximum nesting depth for block extraction (default 3)
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    source_lines = source.splitlines()

    # Track method locations to avoid duplicates (methods are extracted from classes)
    method_locations: set[tuple[int, int]] = set()

    # First pass: extract classes and their methods
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            start = node.lineno - 1
            end = node.end_lineno if node.end_lineno else start + 1
            code = textwrap.dedent("\n".join(source_lines[start:end]))

            yield CodeUnit(
                name=node.name,
                code=code,
                file_path=file_path,
                line_start=node.lineno,
                line_end=end,
                unit_type="class",
                parent_name=None,
                language="python",
            )

            # Extract methods from the class
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    m_start = item.lineno - 1
                    m_end = item.end_lineno if item.end_lineno else m_start + 1
                    method_code = textwrap.dedent("\n".join(source_lines[m_start:m_end]))
                    method_locations.add((item.lineno, m_end))

                    yield CodeUnit(
                        name=item.name,
                        code=method_code,
                        file_path=file_path,
                        line_start=item.lineno,
                        line_end=m_end,
                        unit_type="method",
                        parent_name=node.name,
                        language="python",
                    )

                    # Extract blocks from method if requested
                    if include_blocks:
                        yield from extract_blocks_from_function(
                            item, source_lines, file_path, max_block_depth
                        )

    # Second pass: extract standalone functions (not methods)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            start = node.lineno - 1
            end = node.end_lineno if node.end_lineno else start + 1

            # Skip if this was already extracted as a method
            if (node.lineno, end) in method_locations:
                continue

            code = textwrap.dedent("\n".join(source_lines[start:end]))

            yield CodeUnit(
                name=node.name,
                code=code,
                file_path=file_path,
                line_start=node.lineno,
                line_end=end,
                unit_type="function",
                parent_name=None,
                language="python",
            )

            # Extract blocks from function if requested
            if include_blocks:
                yield from extract_blocks_from_function(
                    node, source_lines, file_path, max_block_depth
                )


def code_unit_to_ast_graph(unit: CodeUnit) -> ASTGraph:
    """Convert a CodeUnit to an ASTGraph with metadata."""
    graph = ast_to_graph(unit.code)
    label_histogram = compute_label_histogram(graph)

    # Compute depth
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


class PythonPlugin(BaseLanguagePlugin):
    """Python language plugin using stdlib ast module."""

    @property
    def language_id(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".py", ".pyi"})

    @property
    def skip_dirs(self) -> frozenset[str]:
        return frozenset({"__pycache__", "venv", ".venv", ".tox", ".mypy_cache"})

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        return extract_code_units(source, file_path, include_blocks, max_block_depth)

    def source_to_graph(
        self,
        source: str,
        normalize_ops: bool = False,
    ) -> nx.DiGraph:
        return ast_to_graph(source, normalize_ops)

    def code_unit_to_ast_graph(self, unit: CodeUnit) -> ASTGraph:
        return code_unit_to_ast_graph(unit)
