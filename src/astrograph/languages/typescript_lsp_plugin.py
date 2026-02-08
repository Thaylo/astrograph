"""TypeScript language adapter extending the JavaScript LSP plugin."""

from __future__ import annotations

import logging
import re

import networkx as nx

from .base import SemanticProfile, SemanticSignal
from .javascript_lsp_plugin import JavaScriptLSPPlugin, _esprima_ast_to_graph

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
    # Remove enum declarations
    result = _TS_ENUM_RE.sub("", result)
    # Remove namespace blocks
    result = _TS_NAMESPACE_BLOCK_RE.sub("", result)
    # Remove interface / type alias blocks
    result = _TS_INTERFACE_BLOCK_RE.sub("", result)
    # Remove declare statements
    result = _TS_DECLARE_RE.sub("", result)
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
    # Semantic profiling (inherited JS signals + TS-specific)
    # ------------------------------------------------------------------

    def extract_semantic_profile(
        self,
        source: str,
        file_path: str = "<unknown>",
    ) -> SemanticProfile:
        """Extend JS semantic profile with TypeScript-specific signals."""
        base_profile = super().extract_semantic_profile(source=source, file_path=file_path)
        signals = list(base_profile.signals)
        notes = list(base_profile.notes)
        extra_coverage = 0.0

        # 1. Strict mode indicators (always emitted)
        has_strict = bool(_TS_STRICT_MODE_RE.search(source))
        signals.append(
            SemanticSignal(
                key="typescript.strict_mode",
                value="yes" if has_strict else "no",
                confidence=0.85,
                origin="syntax",
            )
        )
        extra_coverage += 0.05

        # 2. Generic usage (always emitted)
        has_generic = bool(_TS_GENERIC_DETECT_RE.search(source))
        signals.append(
            SemanticSignal(
                key="typescript.generic.present",
                value="yes" if has_generic else "no",
                confidence=0.90,
                origin="syntax",
            )
        )
        extra_coverage += 0.05

        return SemanticProfile(
            signals=tuple(signals),
            coverage=min(1.0, base_profile.coverage + extra_coverage),
            notes=tuple(notes),
            extractor="typescript_lsp:syntax",
        )
