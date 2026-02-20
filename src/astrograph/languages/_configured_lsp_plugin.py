"""Shared configuration-driven base for LSP-backed language plugins."""

from __future__ import annotations

from collections.abc import Iterator

import networkx as nx

from ._brace_block_extractor import extract_brace_blocks_from_function
from ._lsp_base import LSPClient, LSPLanguagePluginBase
from .base import CodeUnit
from .lsp_client import create_subprocess_client_from_env


class ConfiguredLSPLanguagePluginBase(LSPLanguagePluginBase):
    """LSPLanguagePluginBase with env/default command wiring."""

    DEFAULT_COMMAND: tuple[str, ...] = ()
    COMMAND_ENV_VAR: str = ""
    TIMEOUT_ENV_VAR: str = ""

    def __init__(self, lsp_client: LSPClient | None = None) -> None:
        client = lsp_client
        if client is None:
            client = create_subprocess_client_from_env(
                default_command=self.DEFAULT_COMMAND,
                command_env_var=self.COMMAND_ENV_VAR,
                timeout_env_var=self.TIMEOUT_ENV_VAR,
                language_id=self.LANGUAGE_ID,
            )
        super().__init__(lsp_client=client)


class BraceLanguageLSPPlugin(ConfiguredLSPLanguagePluginBase):
    """Shared logic for brace-delimited languages (C, C++, Java).

    Provides operator-label normalization and brace-block extraction
    so subclasses don't need to duplicate these methods.
    """

    def normalize_graph_for_pattern(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Normalize operator labels from the line-level parser for pattern matching."""
        normalized: nx.DiGraph = graph.copy()
        for _node_id, data in normalized.nodes(data=True):
            label = data.get("label", "")
            if ":Op(" in label:
                data["label"] = label[: label.index(":Op(") + 3]
        return normalized

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract units via LSP, then optionally extract inner blocks via brace matching."""
        func_units: list[CodeUnit] = []
        for unit in super().extract_code_units(
            source, file_path, include_blocks=False, max_block_depth=max_block_depth
        ):
            yield unit
            if unit.unit_type in {"function", "method"}:
                func_units.append(unit)

        if not include_blocks:
            return

        for unit in func_units:
            yield from extract_brace_blocks_from_function(
                func_code=unit.code,
                file_path=file_path,
                func_name=unit.name,
                func_line_start=unit.line_start,
                language=self.LANGUAGE_ID,
                max_depth=max_block_depth,
            )
