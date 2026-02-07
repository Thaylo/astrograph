"""Python language adapter using the LSP plugin base."""

from __future__ import annotations

import ast
import os
import shlex
from collections.abc import Iterator

import networkx as nx

from ._lsp_base import LSPClient, LSPLanguagePluginBase
from .base import CodeUnit
from .lsp_client import SubprocessLSPClient
from .python_plugin import PythonPlugin, ast_to_graph, extract_blocks_from_function

_DEFAULT_PY_LSP_COMMAND = ("pylsp",)
_PY_LSP_COMMAND_ENV = "ASTROGRAPH_PY_LSP_COMMAND"
_PY_LSP_TIMEOUT_ENV = "ASTROGRAPH_PY_LSP_TIMEOUT"


def _default_python_lsp_client() -> LSPClient:
    """Create a default LSP client for Python language servers."""
    command_text = os.getenv(_PY_LSP_COMMAND_ENV, "")
    command = shlex.split(command_text) if command_text.strip() else list(_DEFAULT_PY_LSP_COMMAND)

    timeout_text = os.getenv(_PY_LSP_TIMEOUT_ENV, "5")
    try:
        timeout = float(timeout_text)
    except ValueError:
        timeout = 5.0

    return SubprocessLSPClient(command, request_timeout=max(timeout, 0.1))


class PythonLSPPlugin(LSPLanguagePluginBase):
    """Python support via LSP symbols + AST block extraction."""

    def __init__(self, lsp_client: LSPClient | None = None) -> None:
        super().__init__(lsp_client=lsp_client or _default_python_lsp_client())
        self._graph_plugin = PythonPlugin()

    @property
    def language_id(self) -> str:
        return "python"

    @property
    def lsp_language_id(self) -> str:
        return "python"

    @property
    def file_extensions(self) -> frozenset[str]:
        return frozenset({".py", ".pyi"})

    @property
    def skip_dirs(self) -> frozenset[str]:
        return frozenset({"__pycache__", "venv", ".venv", ".tox", ".mypy_cache"})

    def source_to_graph(self, source: str, normalize_ops: bool = False) -> nx.DiGraph:
        """Use the Python AST graph builder for structural fidelity."""
        return ast_to_graph(source, normalize_ops=normalize_ops)

    def normalize_graph_for_pattern(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Reuse Python operator normalization used by the legacy plugin."""
        return self._graph_plugin.normalize_graph_for_pattern(graph)

    def _method_parent_map(self, tree: ast.AST) -> dict[tuple[str, int, int], str]:
        """Build mapping from method (name,start,end) to its class name."""
        mapping: dict[tuple[str, int, int], str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    end = item.end_lineno if item.end_lineno is not None else item.lineno
                    mapping[(item.name, item.lineno, end)] = node.name
        return mapping

    def _apply_method_parents(
        self,
        units: list[CodeUnit],
        method_map: dict[tuple[str, int, int], str],
    ) -> list[CodeUnit]:
        """Patch parent_name/unit_type for methods when LSP returns flat symbols."""
        adjusted: list[CodeUnit] = []
        for unit in units:
            key = (unit.name, unit.line_start, unit.line_end)
            parent_name = method_map.get(key)
            if parent_name is None:
                adjusted.append(unit)
                continue

            unit_type = "method" if unit.unit_type in {"method", "function"} else unit.unit_type
            adjusted.append(
                CodeUnit(
                    name=unit.name,
                    code=unit.code,
                    file_path=unit.file_path,
                    line_start=unit.line_start,
                    line_end=unit.line_end,
                    unit_type=unit_type,
                    parent_name=parent_name,
                    block_type=unit.block_type,
                    nesting_depth=unit.nesting_depth,
                    parent_block_name=unit.parent_block_name,
                    language=unit.language,
                )
            )
        return adjusted

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract units via LSP and optionally derive Python blocks via AST."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None

        method_map = self._method_parent_map(tree) if tree is not None else {}

        units = list(
            super().extract_code_units(
                source,
                file_path,
                include_blocks=False,
                max_block_depth=max_block_depth,
            )
        )
        adjusted_units = self._apply_method_parents(units, method_map)
        yield from adjusted_units

        if not include_blocks or tree is None:
            return

        source_lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield from extract_blocks_from_function(
                    node,
                    source_lines,
                    file_path,
                    max_depth=max_block_depth,
                )
