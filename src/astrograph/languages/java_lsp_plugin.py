"""Java language adapter using the LSP plugin base."""

from __future__ import annotations

import re
from collections.abc import Iterator

import networkx as nx

from ._brace_block_extractor import extract_brace_blocks_from_function
from ._configured_lsp_plugin import ConfiguredLSPLanguagePluginBase
from .base import CodeUnit, SemanticSignal

# -- Annotation patterns --
_JAVA_ANNOTATION_RE = re.compile(r"^\s*@([A-Za-z_]\w*)", re.MULTILINE)

# -- Access modifier patterns --
_JAVA_PUBLIC_RE = re.compile(r"\bpublic\b")
_JAVA_PRIVATE_RE = re.compile(r"\bprivate\b")
_JAVA_PROTECTED_RE = re.compile(r"\bprotected\b")

# -- Generic type patterns --
_JAVA_GENERIC_RE = re.compile(r"<\s*[A-Z]\w*(?:\s+extends\b)?")

# -- Exception handling patterns --
_JAVA_THROWS_RE = re.compile(r"\bthrows\s+\w")
_JAVA_TRY_CATCH_RE = re.compile(r"\btry\s*\{|\bcatch\s*\(")

# -- Stream / lambda patterns --
_JAVA_STREAM_RE = re.compile(r"\.stream\s*\(|\.map\s*\(|\.filter\s*\(|\.collect\s*\(")
_JAVA_LAMBDA_RE = re.compile(r"\)\s*->|[A-Za-z_]\w*\s*->")

# -- Static / final patterns --
_JAVA_STATIC_RE = re.compile(r"\bstatic\b")
_JAVA_FINAL_RE = re.compile(r"\bfinal\b")

# -- Interface vs class --
_JAVA_INTERFACE_RE = re.compile(r"\binterface\s+[A-Z]")
_JAVA_ABSTRACT_RE = re.compile(r"\babstract\s+class\b")
_JAVA_CLASS_RE = re.compile(r"\bclass\s+[A-Z]")

# -- Java 17-25 modern features --
_JAVA_RECORD_RE = re.compile(r"\brecord\s+[A-Z]\w*\s*\(")
_JAVA_SEALED_RE = re.compile(r"\bsealed\s+(?:class|interface)\b")
_JAVA_PERMITS_RE = re.compile(r"\bpermits\s+[A-Z]")
_JAVA_PATTERN_INSTANCEOF_RE = re.compile(r"\binstanceof\s+[A-Z]\w+\s+\w+")
_JAVA_SWITCH_EXPR_ARROW_RE = re.compile(r"\bcase\s+[^:]+\s*->")
_JAVA_TEXT_BLOCK_RE = re.compile(r'"""')
_JAVA_VAR_RE = re.compile(r"\bvar\s+\w+")


class JavaLSPPlugin(ConfiguredLSPLanguagePluginBase):
    """Java support via an attached or spawned LSP backend."""

    LANGUAGE_ID = "java_lsp"
    LSP_LANGUAGE_ID = "java"
    FILE_EXTENSIONS = frozenset({".java"})
    SKIP_DIRS = frozenset({"build", "out", "target", ".gradle"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2089",)
    COMMAND_ENV_VAR = "ASTROGRAPH_JAVA_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_JAVA_LSP_TIMEOUT"

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

    def _language_signals(self, source: str) -> list[SemanticSignal]:
        """Java-specific regex signals for semantic profiling."""
        signals = super()._language_signals(source)

        # 1. Annotations
        annotations = sorted({m.group(1) for m in _JAVA_ANNOTATION_RE.finditer(source)})
        signals.append(
            SemanticSignal(
                key="java.annotations",
                value=",".join(annotations) if annotations else "none",
                confidence=0.95,
                origin="syntax",
            )
        )

        # 2. Access modifiers
        modifiers = []
        if _JAVA_PUBLIC_RE.search(source):
            modifiers.append("public")
        if _JAVA_PRIVATE_RE.search(source):
            modifiers.append("private")
        if _JAVA_PROTECTED_RE.search(source):
            modifiers.append("protected")
        signals.append(
            SemanticSignal(
                key="java.access_modifiers",
                value=",".join(modifiers) if modifiers else "package",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 3. Generic usage
        has_generic = bool(_JAVA_GENERIC_RE.search(source))
        signals.append(
            SemanticSignal(
                key="java.generic.present",
                value="yes" if has_generic else "no",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 4. Exception handling
        has_throws = bool(_JAVA_THROWS_RE.search(source))
        has_try_catch = bool(_JAVA_TRY_CATCH_RE.search(source))
        if has_throws and has_try_catch:
            exception_val = "both"
        elif has_throws:
            exception_val = "throws"
        elif has_try_catch:
            exception_val = "try_catch"
        else:
            exception_val = "none"
        signals.append(
            SemanticSignal(
                key="java.exception_handling",
                value=exception_val,
                confidence=0.90,
                origin="syntax",
            )
        )

        # 5. Stream / lambda style
        has_stream = bool(_JAVA_STREAM_RE.search(source))
        has_lambda = bool(_JAVA_LAMBDA_RE.search(source))
        if has_stream and has_lambda:
            functional_val = "stream_lambda"
        elif has_stream:
            functional_val = "stream"
        elif has_lambda:
            functional_val = "lambda"
        else:
            functional_val = "none"
        signals.append(
            SemanticSignal(
                key="java.functional_style",
                value=functional_val,
                confidence=0.85,
                origin="syntax",
            )
        )

        # 6. Class kind
        has_interface = bool(_JAVA_INTERFACE_RE.search(source))
        has_abstract = bool(_JAVA_ABSTRACT_RE.search(source))
        has_class = bool(_JAVA_CLASS_RE.search(source))
        if has_interface:
            class_kind = "interface"
        elif has_abstract:
            class_kind = "abstract"
        elif has_class:
            class_kind = "class"
        else:
            class_kind = "none"
        signals.append(
            SemanticSignal(
                key="java.class_kind",
                value=class_kind,
                confidence=0.90,
                origin="syntax",
            )
        )

        # 7. Modern Java features (Java 14-25)
        modern_parts = []
        if _JAVA_RECORD_RE.search(source):
            modern_parts.append("record")
        if _JAVA_SEALED_RE.search(source):
            modern_parts.append("sealed")
        if _JAVA_PATTERN_INSTANCEOF_RE.search(source):
            modern_parts.append("pattern_instanceof")
        if _JAVA_SWITCH_EXPR_ARROW_RE.search(source):
            modern_parts.append("switch_expression")
        if _JAVA_TEXT_BLOCK_RE.search(source):
            modern_parts.append("text_block")
        if _JAVA_VAR_RE.search(source):
            modern_parts.append("var")
        signals.append(
            SemanticSignal(
                key="java.modern_features",
                value=",".join(modern_parts) if modern_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        return signals
