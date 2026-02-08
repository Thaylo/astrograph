"""C++ language adapter using the LSP plugin base."""

from __future__ import annotations

import re

from ._configured_lsp_plugin import ConfiguredLSPLanguagePluginBase
from .base import SemanticProfile, SemanticSignal

_CPP_USER_TYPE_RE = re.compile(r"\b(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)")
_CPP_OPERATOR_PLUS_DECL_RE = re.compile(r"\boperator\s*\+\s*\(")
_CPP_PLUS_EXPR_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)\b")
_CPP_FUNCTION_SIG_RE = re.compile(
    r"\b[A-Za-z_][A-Za-z0-9_:<>]*\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^()]*)\)"
)
_CPP_LOCAL_DECL_RE = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_:<>]*(?:\s+[A-Za-z_][A-Za-z0-9_:<>]*)?)\s+[*&\s]*"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|;)"
)
_CPP_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CPP_QUALIFIER_RE = re.compile(
    r"\b(?:const|constexpr|volatile|static|inline|friend|typename|mutable|register)\b"
)
_CPP_PRIMITIVE_TYPES = frozenset(
    {
        "auto",
        "bool",
        "char",
        "char8_t",
        "char16_t",
        "char32_t",
        "double",
        "float",
        "int",
        "long",
        "long double",
        "long long",
        "short",
        "signed",
        "signed char",
        "signed int",
        "signed long",
        "signed long long",
        "signed short",
        "size_t",
        "std::size_t",
        "std::string",
        "unsigned",
        "unsigned char",
        "unsigned int",
        "unsigned long",
        "unsigned long long",
        "unsigned short",
        "void",
        "wchar_t",
    }
)


class CppLSPPlugin(ConfiguredLSPLanguagePluginBase):
    """C++ support via an attached or spawned LSP backend."""

    LANGUAGE_ID = "cpp_lsp"
    LSP_LANGUAGE_ID = "cpp"
    FILE_EXTENSIONS = frozenset({".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".ipp"})
    SKIP_DIRS = frozenset({"build", "cmake-build-debug", "cmake-build-release"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2088",)
    COMMAND_ENV_VAR = "ASTROGRAPH_CPP_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_CPP_LSP_TIMEOUT"

    def _normalize_cpp_type(self, raw_type: str) -> str:
        cleaned = raw_type.replace("&", " ").replace("*", " ")
        cleaned = _CPP_QUALIFIER_RE.sub(" ", cleaned)
        return " ".join(cleaned.split())

    def _is_builtin_cpp_type(self, type_name: str) -> bool:
        return type_name in _CPP_PRIMITIVE_TYPES

    def _split_parameters(self, raw_params: str) -> list[str]:
        if not raw_params.strip():
            return []
        parts: list[str] = []
        start = 0
        depth = 0
        for idx, char in enumerate(raw_params):
            if char in "<([{":
                depth += 1
            elif char in ">)]}":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                parts.append(raw_params[start:idx])
                start = idx + 1
        parts.append(raw_params[start:])
        return parts

    def _typed_name_from_fragment(self, fragment: str) -> tuple[str, str] | None:
        body = fragment.split("=", 1)[0].strip()
        if not body or body == "void":
            return None

        tokens = body.replace("&", " ").replace("*", " ").split()
        if len(tokens) < 2:
            return None

        name = tokens[-1]
        if _CPP_IDENTIFIER_RE.fullmatch(name) is None:
            return None

        type_name = self._normalize_cpp_type(" ".join(tokens[:-1]))
        if not type_name:
            return None
        return name, type_name

    def _collect_variable_types(self, source: str) -> dict[str, str]:
        var_types: dict[str, str] = {}

        for signature_match in _CPP_FUNCTION_SIG_RE.finditer(source):
            for fragment in self._split_parameters(signature_match.group(1)):
                parsed = self._typed_name_from_fragment(fragment)
                if parsed is None:
                    continue
                var_name, type_name = parsed
                var_types[var_name] = type_name

        for local_match in _CPP_LOCAL_DECL_RE.finditer(source):
            type_name = self._normalize_cpp_type(local_match.group(1))
            var_name = local_match.group(2)
            if type_name:
                var_types.setdefault(var_name, type_name)

        return var_types

    def _infer_plus_binding(
        self,
        *,
        source: str,
        var_types: dict[str, str],
        operator_plus_declared: bool,
    ) -> tuple[str, float]:
        plus_pairs = _CPP_PLUS_EXPR_RE.findall(source)
        if not plus_pairs:
            return "absent", 0.0

        saw_unknown = False
        saw_builtin = False
        saw_user_defined = False
        for left, right in plus_pairs:
            left_type = var_types.get(left)
            right_type = var_types.get(right)
            if left_type is None or right_type is None:
                saw_unknown = True
                continue

            if left_type == right_type and self._is_builtin_cpp_type(left_type):
                saw_builtin = True
                continue

            if left_type == right_type and not self._is_builtin_cpp_type(left_type):
                saw_user_defined = True
                continue

            saw_unknown = True

        if saw_user_defined:
            if operator_plus_declared:
                return "user_defined_overload", 0.95
            return "user_defined_overload_required", 0.7
        if saw_builtin:
            return "builtin", 0.9
        if saw_unknown:
            return "unknown", 0.25
        return "absent", 0.0

    def extract_semantic_profile(
        self,
        source: str,
        file_path: str = "<unknown>",
    ) -> SemanticProfile:
        base_profile = super().extract_semantic_profile(source=source, file_path=file_path)
        signals = list(base_profile.signals)
        notes = list(base_profile.notes)

        user_types = {
            self._normalize_cpp_type(match) for match in _CPP_USER_TYPE_RE.findall(source)
        }
        if user_types:
            signals.append(
                SemanticSignal(
                    key="typing.user_types.present",
                    value="yes",
                    confidence=0.9,
                    origin="syntax",
                )
            )

        var_types = self._collect_variable_types(source)
        operator_plus_declared = bool(_CPP_OPERATOR_PLUS_DECL_RE.search(source))
        plus_binding, plus_confidence = self._infer_plus_binding(
            source=source,
            var_types=var_types,
            operator_plus_declared=operator_plus_declared,
        )
        if plus_binding != "absent":
            signals.append(
                SemanticSignal(
                    key="operator.plus.binding",
                    value=plus_binding,
                    confidence=plus_confidence,
                    origin="syntax",
                )
            )

        if plus_binding == "unknown":
            notes.append(
                "C++ plus-expression types are partially unresolved; C++ LSP hover/type info would improve confidence."
            )

        extra_coverage = 0.0
        if user_types:
            extra_coverage += 0.2
        if plus_binding != "absent":
            extra_coverage += 0.4

        return SemanticProfile(
            signals=tuple(signals),
            coverage=min(1.0, base_profile.coverage + extra_coverage),
            notes=tuple(notes),
            extractor="cpp_lsp:syntax",
        )
