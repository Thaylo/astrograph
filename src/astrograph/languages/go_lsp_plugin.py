"""Go language adapter using the LSP plugin base."""

from __future__ import annotations

import re

from ._configured_lsp_plugin import BraceLanguageLSPPlugin
from .base import SemanticSignal, binary_signal_value

# -- Concurrency patterns --
_GO_GOROUTINE_RE = re.compile(r"\bgo\s+(?:func\b|\w+\s*\()")
_GO_CHAN_RE = re.compile(r"\bmake\s*\(\s*chan\b|<-")
_GO_SELECT_RE = re.compile(r"\bselect\s*\{")

# -- Error handling patterns --
_GO_ERR_NIL_RE = re.compile(r"\bif\s+err\s*!=\s*nil\b")
_GO_ERRORS_NEW_RE = re.compile(r"\berrors\.New\s*\(|\bfmt\.Errorf\s*\(")
_GO_ERR_RETURN_RE = re.compile(r"\breturn\b.*\berr\b")

# -- Interface patterns --
_GO_INTERFACE_DECL_RE = re.compile(r"\binterface\s*\{")
_GO_TYPE_ASSERTION_RE = re.compile(r"\.\(\s*[A-Z*]")
_GO_TYPE_SWITCH_RE = re.compile(r"\bswitch\b.*\.\(\s*type\s*\)")

# -- Struct embedding --
_GO_STRUCT_EMBED_RE = re.compile(
    r"\bstruct\s*\{[^}]*\n\s*(?:\*\s*)?[A-Z][A-Za-z0-9_]*\s*\n", re.DOTALL
)

# -- Receiver style --
_GO_POINTER_RECEIVER_RE = re.compile(r"\bfunc\s*\(\s*\w+\s+\*\w+\s*\)")
_GO_VALUE_RECEIVER_RE = re.compile(r"\bfunc\s*\(\s*\w+\s+[A-Z]\w*\s*\)")

# -- Defer / recover --
_GO_DEFER_RE = re.compile(r"\bdefer\b")
_GO_PANIC_RE = re.compile(r"\bpanic\s*\(")
_GO_RECOVER_RE = re.compile(r"\brecover\s*\(")

# -- Modern features (Go 1.18+) --
_GO_GENERIC_RE = re.compile(r"\[(?:T|[A-Z]\w*)\s+(?:any|comparable|constraints\.\w+|~)")
_GO_EMBED_DIRECTIVE_RE = re.compile(r"//go:embed\b")
_GO_RANGE_INT_RE = re.compile(r"\brange\s+\d+\b")
_GO_RANGE_FUNC_RE = re.compile(r"\brange\s+\w+\s*\(")


class GoLSPPlugin(BraceLanguageLSPPlugin):
    """Go support via an attached or spawned LSP backend."""

    LANGUAGE_ID = "go_lsp"
    LSP_LANGUAGE_ID = "go"
    FILE_EXTENSIONS = frozenset({".go"})
    SKIP_DIRS = frozenset({"vendor"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2091",)
    COMMAND_ENV_VAR = "ASTROGRAPH_GO_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_GO_LSP_TIMEOUT"

    def _language_signals(self, source: str) -> list[SemanticSignal]:
        """Go-specific regex signals for semantic profiling."""
        signals = super()._language_signals(source)

        # 1. Concurrency (goroutines, channels, select)
        has_goroutine = bool(_GO_GOROUTINE_RE.search(source))
        has_chan = bool(_GO_CHAN_RE.search(source))
        has_select = bool(_GO_SELECT_RE.search(source))
        concurrency_parts = []
        if has_goroutine:
            concurrency_parts.append("goroutine")
        if has_chan:
            concurrency_parts.append("channel")
        if has_select:
            concurrency_parts.append("select")
        signals.append(
            SemanticSignal(
                key="go.concurrency",
                value=",".join(concurrency_parts) if concurrency_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 2. Error handling
        has_err_nil = bool(_GO_ERR_NIL_RE.search(source))
        has_errors_new = bool(_GO_ERRORS_NEW_RE.search(source))
        has_err_return = bool(_GO_ERR_RETURN_RE.search(source))
        error_parts = []
        if has_err_nil:
            error_parts.append("err_nil_check")
        if has_errors_new:
            error_parts.append("error_creation")
        if has_err_return:
            error_parts.append("error_return")
        signals.append(
            SemanticSignal(
                key="go.error_handling",
                value=",".join(error_parts) if error_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 3. Interface usage
        has_interface_decl = bool(_GO_INTERFACE_DECL_RE.search(source))
        has_type_assertion = bool(_GO_TYPE_ASSERTION_RE.search(source))
        has_type_switch = bool(_GO_TYPE_SWITCH_RE.search(source))
        interface_parts = []
        if has_interface_decl:
            interface_parts.append("declaration")
        if has_type_assertion:
            interface_parts.append("type_assertion")
        if has_type_switch:
            interface_parts.append("type_switch")
        signals.append(
            SemanticSignal(
                key="go.interface_usage",
                value=",".join(interface_parts) if interface_parts else "none",
                confidence=0.85,
                origin="syntax",
            )
        )

        # 4. Struct embedding
        has_embedding = bool(_GO_STRUCT_EMBED_RE.search(source))
        signals.append(
            SemanticSignal(
                key="go.struct_embedding",
                value="yes" if has_embedding else "no",
                confidence=0.85,
                origin="syntax",
            )
        )

        # 5. Receiver style
        has_pointer = bool(_GO_POINTER_RECEIVER_RE.search(source))
        has_value = bool(_GO_VALUE_RECEIVER_RE.search(source))
        signals.append(
            SemanticSignal(
                key="go.receiver_style",
                value=binary_signal_value(
                    has_pointer, "pointer", has_value, "value", both_label="mixed"
                ),
                confidence=0.90,
                origin="syntax",
            )
        )

        # 6. Defer / recover
        has_defer = bool(_GO_DEFER_RE.search(source))
        has_panic = bool(_GO_PANIC_RE.search(source))
        has_recover = bool(_GO_RECOVER_RE.search(source))
        defer_parts = []
        if has_defer:
            defer_parts.append("defer")
        if has_panic:
            defer_parts.append("panic")
        if has_recover:
            defer_parts.append("recover")
        signals.append(
            SemanticSignal(
                key="go.defer_recover",
                value=",".join(defer_parts) if defer_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 7. Modern features (generics, go:embed, range-over-int, iterators)
        modern_parts = []
        if _GO_GENERIC_RE.search(source):
            modern_parts.append("generics")
        if _GO_EMBED_DIRECTIVE_RE.search(source):
            modern_parts.append("go_embed")
        if _GO_RANGE_INT_RE.search(source):
            modern_parts.append("range_over_int")
        if _GO_RANGE_FUNC_RE.search(source):
            modern_parts.append("range_over_func")
        signals.append(
            SemanticSignal(
                key="go.modern_features",
                value=",".join(modern_parts) if modern_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        return signals
