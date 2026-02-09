"""TypeScript language adapter extending the JavaScript LSP plugin."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import esprima
import networkx as nx

from .base import CodeUnit, SemanticSignal
from .javascript_lsp_plugin import (
    JavaScriptLSPPlugin,
    _esprima_ast_to_graph,
    _esprima_extract_function_blocks,
)

logger = logging.getLogger(__name__)

# -- TS annotation stripping patterns --
# Type annotations after identifiers: `param: Type` → `param`
_TS_TYPE_ANNOTATION_RE = re.compile(
    r"([\w$]+)\s*:\s*" r"(?:[A-Za-z_$][\w$.<>,\s|&\[\]]*)" r"(?=\s*[,);=\]}])"
)
# Return type annotations: `): Type` → `)`
_TS_RETURN_TYPE_RE = re.compile(r"\)\s*:\s*[A-Za-z_$][\w$.<>,\s|&\[\]]*(?=\s*[{=>;])")
# Interface / type alias declarations (entire blocks)
_TS_INTERFACE_BLOCK_RE = re.compile(
    r"\b(?:interface|type)\s+[A-Z][\w$]*(?:<[^>]*>)?\s*(?:extends\s+[^{]*)?\{[^}]*\}",
    re.DOTALL,
)
# Generic type parameters: `<T>`, `<T extends Foo>`
_TS_GENERIC_PARAMS_RE = re.compile(
    r"<\s*[A-Z][\w$]*(?:\s+extends\s+[^>]*)?\s*(?:,\s*[A-Z][\w$]*(?:\s+extends\s+[^>]*)?\s*)*>"
)
# `as Type` casts
_TS_AS_CAST_RE = re.compile(r"\bas\s+[A-Za-z_$][\w$.<>,\s|&\[\]]*")
# Non-null assertion: `x!.` or `x!)`
_TS_NON_NULL_RE = re.compile(r"(\w)!(?=[.\[)\],;])")
# `readonly` modifier
_TS_READONLY_RE = re.compile(r"\breadonly\s+")
# `declare` keyword
_TS_DECLARE_RE = re.compile(r"\bdeclare\s+")
# Enum declarations
_TS_ENUM_RE = re.compile(r"\b(?:const\s+)?enum\s+\w+\s*\{[^}]*\}", re.DOTALL)
# Namespace declarations
_TS_NAMESPACE_BLOCK_RE = re.compile(
    r"\bnamespace\s+\w+\s*\{[^}]*\}",
    re.DOTALL,
)

# -- TS 5.x+ syntax (decorators, using, satisfies) --
# Stage 3 decorators (esprima doesn't support them)
_TS_DECORATOR_RE = re.compile(r"^\s*@\w[\w$.]*(?:\([^)]*\))?\s*$", re.MULTILINE)
# `using` / `await using` (TS 5.2+, explicit resource management)
_TS_USING_RE = re.compile(r"\b(?:await\s+)?using\s+[A-Za-z_$][\w$]*\s*=")
# `satisfies` operator (TS 4.9+)
_TS_SATISFIES_RE = re.compile(r"\bsatisfies\s+[A-Za-z_$][\w$.<>,\s|&\[\]]*")
# `accessor` keyword (TS 4.9+ auto-accessor)
_TS_ACCESSOR_RE = re.compile(r"\baccessor\s+\w+")

# -- TS-specific semantic patterns --
_TS_GENERIC_DETECT_RE = re.compile(
    r"<\s*[A-Z][\w$]*(?:\s+extends\b)?" r"|(?:function|class|interface|type)\s+\w+\s*<"
)
_TS_STRICT_MODE_RE = re.compile(
    r"\bas\s+\w" r"|\w\s*!\s*[.\[]" r"|\w+\s*:\s*(?:string|number|boolean|void|any|never|unknown)"
)


def _strip_ts_annotations(source: str) -> str:
    """Remove TypeScript-specific syntax so esprima can parse the result."""
    result = source
    # Remove decorators (Stage 3, not supported by esprima)
    result = _TS_DECORATOR_RE.sub("", result)
    # Remove enum declarations
    result = _TS_ENUM_RE.sub("", result)
    # Remove namespace blocks
    result = _TS_NAMESPACE_BLOCK_RE.sub("", result)
    # Remove interface / type alias blocks
    result = _TS_INTERFACE_BLOCK_RE.sub("", result)
    # Remove declare statements
    result = _TS_DECLARE_RE.sub("", result)
    # Replace `using`/`await using` with `const` (preserves structure)
    result = _TS_USING_RE.sub(
        lambda m: "const " + m.group(0).split("=")[0].split()[-1] + " =", result
    )
    # Remove `satisfies Type` (TS 4.9+)
    result = _TS_SATISFIES_RE.sub("", result)
    # Remove `accessor` keyword (TS 4.9+)
    result = _TS_ACCESSOR_RE.sub(lambda m: m.group(0).replace("accessor ", ""), result)
    # Remove generic type params
    result = _TS_GENERIC_PARAMS_RE.sub("", result)
    # Remove as casts
    result = _TS_AS_CAST_RE.sub("", result)
    # Remove non-null assertions
    result = _TS_NON_NULL_RE.sub(r"\1", result)
    # Remove readonly
    result = _TS_READONLY_RE.sub("", result)
    # Remove return type annotations (before param annotations to avoid overlap)
    result = _TS_RETURN_TYPE_RE.sub(")", result)
    # Remove param type annotations
    result = _TS_TYPE_ANNOTATION_RE.sub(r"\1", result)
    return result


class TypeScriptLSPPlugin(JavaScriptLSPPlugin):
    """TypeScript support via esprima (with annotation stripping) + JS semantic signals."""

    LANGUAGE_ID = "typescript_lsp"
    LSP_LANGUAGE_ID = "typescript"
    FILE_EXTENSIONS = frozenset({".ts", ".tsx"})
    SKIP_DIRS = frozenset({"node_modules", ".next", "dist", "build"})
    DEFAULT_COMMAND = ("typescript-language-server", "--stdio")
    COMMAND_ENV_VAR = "ASTROGRAPH_TS_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_TS_LSP_TIMEOUT"

    # ------------------------------------------------------------------
    # AST graph builder (strip TS → parse as JS)
    # ------------------------------------------------------------------

    def source_to_graph(
        self,
        source: str,
        normalize_ops: bool = False,
    ) -> nx.DiGraph:
        """Strip TS annotations then build an esprima AST graph."""
        stripped = _strip_ts_annotations(source)
        graph = _esprima_ast_to_graph(stripped, normalize_ops=normalize_ops)
        if graph is not None:
            return graph
        logger.debug("esprima parse failed for TS source, falling back to line-level parser")
        return super(JavaScriptLSPPlugin, self).source_to_graph(source, normalize_ops=normalize_ops)

    # ------------------------------------------------------------------
    # Block extraction (strip TS → parse as JS → extract blocks)
    # ------------------------------------------------------------------

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract units via LSP, then extract inner blocks from stripped TS."""
        yield from super(JavaScriptLSPPlugin, self).extract_code_units(
            source,
            file_path,
            include_blocks=False,
            max_block_depth=max_block_depth,
        )

        if not include_blocks:
            return

        stripped = _strip_ts_annotations(source)
        tree = None
        for parser in (esprima.parseScript, esprima.parseModule):
            try:
                tree = parser(stripped, loc=True)
                break
            except esprima.Error:
                continue

        if tree is None:
            return

        source_lines = source.splitlines()
        yield from _esprima_extract_function_blocks(
            tree,
            source_lines,
            file_path,
            max_depth=max_block_depth,
            language=self.LANGUAGE_ID,
        )

    # ------------------------------------------------------------------
    # Semantic profiling (inherited JS signals + TS-specific)
    # ------------------------------------------------------------------

    def _language_signals(self, source: str) -> list[SemanticSignal]:
        """TypeScript language signals (extends JS base signals).

        Adds TS-specific strict-mode and generic-usage indicators on top of
        the JavaScript signals provided by the parent class.
        """
        signals = super()._language_signals(source)

        # 1. Strict mode indicators (as/non-null/typed params)
        has_strict = bool(_TS_STRICT_MODE_RE.search(source))
        signals.append(
            SemanticSignal(
                key="typescript.strict_mode",
                value="yes" if has_strict else "no",
                confidence=0.85,
                origin="syntax",
            )
        )

        # 2. Generic usage (<T>, <T extends ...>, generic declarations)
        has_generic = bool(_TS_GENERIC_DETECT_RE.search(source))
        signals.append(
            SemanticSignal(
                key="typescript.generic.present",
                value="yes" if has_generic else "no",
                confidence=0.90,
                origin="syntax",
            )
        )

        return signals
