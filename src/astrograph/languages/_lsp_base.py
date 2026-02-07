"""
LSP-backed language plugin base class.

This base allows language plugins to rely on LSP document symbols for code-unit
extraction while using a lightweight structural graph builder for hashing and
duplicate detection.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Protocol

import networkx as nx

from .base import BaseLanguagePlugin, CodeUnit

_KEYWORD_LABELS = {
    "class": "ClassDecl",
    "function": "FunctionDecl",
    "def": "FunctionDecl",
    "if": "IfStmt",
    "elif": "IfStmt",
    "else": "ElseStmt",
    "for": "ForStmt",
    "while": "WhileStmt",
    "switch": "SwitchStmt",
    "case": "CaseStmt",
    "try": "TryStmt",
    "catch": "CatchStmt",
    "finally": "FinallyStmt",
    "with": "WithStmt",
    "return": "ReturnStmt",
}
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_STRING_RE = re.compile(r"'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\"")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_EXACT_OPS = (
    "===",
    "!==",
    "==",
    "!=",
    "<=",
    ">=",
    "+=",
    "-=",
    "*=",
    "/=",
    "&&",
    "||",
    "+",
    "-",
    "*",
    "/",
    "%",
    "<",
    ">",
)
_BLOCK_OPENERS = ("{", ":")
_BLOCK_CLOSERS = ("}",)

# LSP SymbolKind constants (https://microsoft.github.io/language-server-protocol/)
_SYMBOL_KIND_CLASS = 5
_SYMBOL_KIND_METHOD = 6
_SYMBOL_KIND_CONSTRUCTOR = 9
_SYMBOL_KIND_FUNCTION = 12
_SYMBOL_KIND_INTERFACE = 11
_SYMBOL_KIND_STRUCT = 23


@dataclass(frozen=True)
class LSPPosition:
    """LSP position (0-based)."""

    line: int
    character: int = 0


@dataclass(frozen=True)
class LSPRange:
    """LSP range (0-based, inclusive lines in this local model)."""

    start: LSPPosition
    end: LSPPosition


@dataclass(frozen=True)
class LSPSymbol:
    """Minimal symbol tree used for extraction."""

    name: str
    kind: int
    symbol_range: LSPRange
    children: tuple[LSPSymbol, ...] = field(default_factory=tuple)


class LSPClient(Protocol):
    """Protocol for LSP clients used by LSPLanguagePluginBase."""

    def document_symbols(
        self,
        *,
        source: str,
        file_path: str,
        language_id: str,
    ) -> list[LSPSymbol]:
        """Return document symbols for a source document."""
        ...


class NullLSPClient:
    """Fallback LSP client when no backend is configured."""

    def document_symbols(
        self,
        *,
        source: str,
        file_path: str,
        language_id: str,
    ) -> list[LSPSymbol]:
        del source, file_path, language_id
        return []


class LSPLanguagePluginBase(BaseLanguagePlugin):
    """
    Base class for language plugins that extract units from LSP symbols.

    Subclasses can either override properties directly or provide class constants:
    - LANGUAGE_ID
    - FILE_EXTENSIONS
    - SKIP_DIRS
    - LSP_LANGUAGE_ID (optional; defaults to LANGUAGE_ID)
    """

    LANGUAGE_ID: str = ""
    LSP_LANGUAGE_ID: str = ""
    FILE_EXTENSIONS: frozenset[str] = frozenset()
    SKIP_DIRS: frozenset[str] = frozenset()

    def __init__(self, lsp_client: LSPClient | None = None) -> None:
        self._lsp_client: LSPClient = lsp_client if lsp_client is not None else NullLSPClient()

    @property
    def language_id(self) -> str:
        """Unique language identifier used by the registry."""
        if self.LANGUAGE_ID:
            return self.LANGUAGE_ID
        raise NotImplementedError("LSP plugin must define LANGUAGE_ID or override language_id")

    @property
    def file_extensions(self) -> frozenset[str]:
        """Extensions handled by this plugin."""
        extensions = self.FILE_EXTENSIONS
        return extensions

    @property
    def skip_dirs(self) -> frozenset[str]:
        """Language-specific skip directories."""
        return self.SKIP_DIRS or frozenset()

    @property
    def lsp_language_id(self) -> str:
        """Language ID sent to the LSP backend."""
        return self.LSP_LANGUAGE_ID or self.language_id

    def _get_document_symbols(self, source: str, file_path: str) -> list[LSPSymbol]:
        return self._lsp_client.document_symbols(
            source=source,
            file_path=file_path,
            language_id=self.lsp_language_id,
        )

    def _unit_type_for_symbol(
        self,
        symbol: LSPSymbol,
        parent_unit_type: str | None,
    ) -> str | None:
        """Map LSP symbol kinds to ASTrograph unit types."""
        if symbol.kind in (_SYMBOL_KIND_CLASS, _SYMBOL_KIND_INTERFACE, _SYMBOL_KIND_STRUCT):
            return "class"
        if symbol.kind in (_SYMBOL_KIND_METHOD, _SYMBOL_KIND_CONSTRUCTOR):
            return "method"
        if symbol.kind == _SYMBOL_KIND_FUNCTION:
            return "method" if parent_unit_type == "class" else "function"
        return None

    def _extract_symbol_code(
        self,
        *,
        source_lines: list[str],
        symbol: LSPSymbol,
    ) -> tuple[str, int, int]:
        """Extract source snippet and 1-based line range for a symbol."""
        max_line_index = len(source_lines) - 1
        if max_line_index < 0:
            return "", 1, 1

        start_line = max(0, min(symbol.symbol_range.start.line, max_line_index))

        end_line_exclusive = symbol.symbol_range.end.line
        if (
            symbol.symbol_range.end.character == 0
            and end_line_exclusive > symbol.symbol_range.start.line
        ):
            inclusive_end = end_line_exclusive - 1
        else:
            inclusive_end = end_line_exclusive

        raw_end_line = max(0, min(inclusive_end, max_line_index))
        end_line = max(start_line, raw_end_line)
        # Slicing is end-exclusive and line_end is displayed as inclusive.
        code = textwrap.dedent("\n".join(source_lines[start_line : end_line + 1]))
        return code, start_line + 1, end_line + 1

    def _iter_symbols(
        self,
        symbols: tuple[LSPSymbol, ...],
        *,
        parent_name: str | None = None,
        parent_unit_type: str | None = None,
    ) -> Iterator[tuple[LSPSymbol, str | None, str | None]]:
        for symbol in symbols:
            unit_type = self._unit_type_for_symbol(symbol, parent_unit_type)
            yield symbol, parent_name, unit_type

            next_parent_name = (
                symbol.name if unit_type in {"class", "function", "method"} else parent_name
            )
            next_parent_type = unit_type if unit_type is not None else parent_unit_type

            if symbol.children:
                yield from self._iter_symbols(
                    symbol.children,
                    parent_name=next_parent_name,
                    parent_unit_type=next_parent_type,
                )

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract classes/functions/methods from LSP symbols."""
        del (
            include_blocks,
            max_block_depth,
        )  # Block extraction depends on language-specific symbols.

        source_lines = source.splitlines()
        symbols = tuple(self._get_document_symbols(source, file_path))

        for symbol, parent_name, unit_type in self._iter_symbols(symbols):
            if unit_type is None:
                continue

            code, line_start, line_end = self._extract_symbol_code(
                source_lines=source_lines,
                symbol=symbol,
            )

            yield CodeUnit(
                name=symbol.name,
                code=code,
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                unit_type=unit_type,
                parent_name=parent_name,
                language=self.language_id,
            )

    def _strip_literals(self, line: str) -> str:
        no_strings = _STRING_RE.sub("STR", line)
        return _NUMBER_RE.sub("NUM", no_strings)

    def _line_operator(self, line: str) -> str | None:
        for op in _EXACT_OPS:
            if op in line:
                return op
        return None

    def _line_label(self, line: str, normalize_ops: bool = False) -> str:
        """Produce a structural line label independent of identifiers/literals."""
        normalized = self._strip_literals(line.strip())
        if not normalized:
            return "Stmt"

        word_match = _WORD_RE.match(normalized)
        keyword = word_match.group(0).lower() if word_match else ""
        base_label = _KEYWORD_LABELS.get(keyword)

        if base_label is None:
            if "=" in normalized and "==" not in normalized and "===" not in normalized:
                base_label = "AssignStmt"
            elif "(" in normalized and ")" in normalized:
                base_label = "CallStmt"
            else:
                base_label = "Stmt"

        op = self._line_operator(normalized)
        if op is None:
            return base_label
        if normalize_ops:
            return f"{base_label}:Op"
        return f"{base_label}:Op({op})"

    def _opens_block(self, stripped_line: str) -> bool:
        return stripped_line.endswith(_BLOCK_OPENERS)

    def _closes_block_prefix(self, stripped_line: str) -> bool:
        return stripped_line.startswith(_BLOCK_CLOSERS)

    def source_to_graph(
        self,
        source: str,
        normalize_ops: bool = False,
    ) -> nx.DiGraph:
        """
        Convert source text to a structural graph.

        This parser is intentionally lightweight and language-agnostic, and
        designed to provide deterministic structure for duplicate detection.
        """
        graph = nx.DiGraph()
        graph.add_node(0, label="Module")

        parent_stack: list[int] = [0]
        node_counter = 1

        for raw_line in source.splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue

            if self._closes_block_prefix(stripped) and len(parent_stack) > 1:
                parent_stack.pop()

            node_id = node_counter
            node_counter += 1
            graph.add_node(node_id, label=self._line_label(stripped, normalize_ops))
            graph.add_edge(parent_stack[-1], node_id)

            if self._opens_block(stripped):
                parent_stack.append(node_id)

        return graph
