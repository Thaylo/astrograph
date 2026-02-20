"""C language adapter using the LSP plugin base."""

from __future__ import annotations

import re

from ._configured_lsp_plugin import BraceLanguageLSPPlugin
from .base import SemanticSignal

# -- Preprocessor patterns --
_C_INCLUDE_RE = re.compile(r"^\s*#\s*include\b", re.MULTILINE)
_C_DEFINE_RE = re.compile(r"^\s*#\s*define\b", re.MULTILINE)
_C_IFDEF_RE = re.compile(r"^\s*#\s*(?:ifdef|ifndef|if)\b", re.MULTILINE)

# -- Pointer patterns --
_C_POINTER_DEREF_RE = re.compile(r"\*\s*[A-Za-z_]\w*(?:\s*[+\-\[\]])?")
_C_ADDRESS_OF_RE = re.compile(r"&\s*[A-Za-z_]\w*")
_C_POINTER_ARITH_RE = re.compile(r"[A-Za-z_]\w*\s*(?:\+\+|--|\+=|-=|\+\s+\d|-\s+\d)")

# -- Struct / union / enum --
_C_STRUCT_RE = re.compile(r"\bstruct\s+[A-Za-z_]\w*")
_C_UNION_RE = re.compile(r"\bunion\s+[A-Za-z_]\w*")
_C_ENUM_RE = re.compile(r"\benum\s+[A-Za-z_]\w*")
_C_TYPEDEF_RE = re.compile(r"\btypedef\b")

# -- Memory management --
_C_MALLOC_RE = re.compile(r"\b(?:malloc|calloc|realloc)\s*\(")
_C_FREE_RE = re.compile(r"\bfree\s*\(")

# -- Control flow --
_C_GOTO_RE = re.compile(r"\bgoto\s+\w+")
_C_SWITCH_RE = re.compile(r"\bswitch\s*\(")

# -- Function pointer --
_C_FUNC_PTR_RE = re.compile(r"\(\s*\*\s*[A-Za-z_]\w*\s*\)\s*\(")

# -- C23 features --
_C_CONSTEXPR_RE = re.compile(r"\bconstexpr\b")
_C_NULLPTR_RE = re.compile(r"\bnullptr\b")
_C_TYPEOF_RE = re.compile(r"\btypeof\s*\(")
_C_STATIC_ASSERT_RE = re.compile(r"\b(?:static_assert|_Static_assert)\s*\(")
_C_EMBED_RE = re.compile(r"^\s*#\s*embed\b", re.MULTILINE)
_C_ATTRIBUTE_RE = re.compile(r"\[\[\s*\w+")
_C_BOOL_KEYWORD_RE = re.compile(r"\bbool\b")


class CLSPPlugin(BraceLanguageLSPPlugin):
    """C support via an attached or spawned LSP backend."""

    LANGUAGE_ID = "c_lsp"
    LSP_LANGUAGE_ID = "c"
    FILE_EXTENSIONS = frozenset({".c", ".h"})
    SKIP_DIRS = frozenset({"build", "cmake-build-debug", "cmake-build-release"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2087",)
    COMMAND_ENV_VAR = "ASTROGRAPH_C_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_C_LSP_TIMEOUT"

    def _language_signals(self, source: str) -> list[SemanticSignal]:
        """C-specific regex signals for semantic profiling."""
        signals = super()._language_signals(source)

        # 1. Preprocessor usage
        parts = []
        if _C_INCLUDE_RE.search(source):
            parts.append("include")
        if _C_DEFINE_RE.search(source):
            parts.append("define")
        if _C_IFDEF_RE.search(source):
            parts.append("conditional")
        signals.append(
            SemanticSignal(
                key="c.preprocessor",
                value=",".join(parts) if parts else "none",
                confidence=0.95,
                origin="syntax",
            )
        )

        # 2. Pointer usage
        ptr_parts = []
        if _C_POINTER_DEREF_RE.search(source):
            ptr_parts.append("dereference")
        if _C_ADDRESS_OF_RE.search(source):
            ptr_parts.append("address_of")
        if _C_POINTER_ARITH_RE.search(source):
            ptr_parts.append("arithmetic")
        signals.append(
            SemanticSignal(
                key="c.pointer_usage",
                value=",".join(ptr_parts) if ptr_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 3. Composite types
        type_parts = []
        if _C_STRUCT_RE.search(source):
            type_parts.append("struct")
        if _C_UNION_RE.search(source):
            type_parts.append("union")
        if _C_ENUM_RE.search(source):
            type_parts.append("enum")
        if _C_TYPEDEF_RE.search(source):
            type_parts.append("typedef")
        signals.append(
            SemanticSignal(
                key="c.composite_types",
                value=",".join(type_parts) if type_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 4. Memory management
        has_alloc = bool(_C_MALLOC_RE.search(source))
        has_free = bool(_C_FREE_RE.search(source))
        if has_alloc and has_free:
            mem_val = "alloc_free"
        elif has_alloc:
            mem_val = "alloc_only"
        elif has_free:
            mem_val = "free_only"
        else:
            mem_val = "none"
        signals.append(
            SemanticSignal(
                key="c.memory_management",
                value=mem_val,
                confidence=0.90,
                origin="syntax",
            )
        )

        # 5. Control flow style
        flow_parts = []
        if _C_GOTO_RE.search(source):
            flow_parts.append("goto")
        if _C_FUNC_PTR_RE.search(source):
            flow_parts.append("function_pointer")
        if _C_SWITCH_RE.search(source):
            flow_parts.append("switch")
        signals.append(
            SemanticSignal(
                key="c.control_flow",
                value=",".join(flow_parts) if flow_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 6. C23 features
        c23_parts = []
        if _C_CONSTEXPR_RE.search(source):
            c23_parts.append("constexpr")
        if _C_NULLPTR_RE.search(source):
            c23_parts.append("nullptr")
        if _C_TYPEOF_RE.search(source):
            c23_parts.append("typeof")
        if _C_STATIC_ASSERT_RE.search(source):
            c23_parts.append("static_assert")
        if _C_EMBED_RE.search(source):
            c23_parts.append("embed")
        if _C_ATTRIBUTE_RE.search(source):
            c23_parts.append("attributes")
        if _C_BOOL_KEYWORD_RE.search(source):
            c23_parts.append("bool")
        signals.append(
            SemanticSignal(
                key="c.c23_features",
                value=",".join(c23_parts) if c23_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        return signals
