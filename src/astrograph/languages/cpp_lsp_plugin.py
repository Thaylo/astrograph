"""C++ language adapter using the LSP plugin base."""

from __future__ import annotations

import re

from ._configured_lsp_plugin import BraceLanguageLSPPlugin
from .base import SemanticSignal, binary_signal_value

_CPP_USER_TYPE_RE = re.compile(r"\b(?:class|struct)\s+([A-Za-z_][A-Za-z0-9_]*)")
_CPP_OPERATOR_PLUS_DECL_RE = re.compile(r"\boperator\s*\+\s*\(")

# -- New semantic signal patterns --
_CPP_TEMPLATE_RE = re.compile(r"\btemplate\s*<")
_CPP_VIRTUAL_RE = re.compile(r"\bvirtual\b")
_CPP_OVERRIDE_RE = re.compile(r"\boverride\b")
_CPP_NAMESPACE_RE = re.compile(r"\bnamespace\s+([A-Za-z_]\w*)")
_CPP_CONST_METHOD_RE = re.compile(r"\)\s*const\b")
_CPP_CONSTEXPR_RE = re.compile(r"\bconstexpr\b")
# -- C++20/23 features --
_CPP_CONCEPT_RE = re.compile(r"\bconcept\s+[A-Za-z_]\w*")
_CPP_REQUIRES_RE = re.compile(r"\brequires\b")
_CPP_CO_AWAIT_RE = re.compile(r"\bco_(?:await|return|yield)\b")
_CPP_MODULE_RE = re.compile(r"^\s*(?:export\s+)?(?:module|import)\s+\w", re.MULTILINE)
_CPP_SPACESHIP_RE = re.compile(r"<=>")
_CPP_CONSTEVAL_RE = re.compile(r"\bconsteval\b")
_CPP_CONSTINIT_RE = re.compile(r"\bconstinit\b")

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


class CppLSPPlugin(BraceLanguageLSPPlugin):
    """C++ support via an attached or spawned LSP backend."""

    LANGUAGE_ID = "cpp_lsp"
    LSP_LANGUAGE_ID = "cpp"
    FILE_EXTENSIONS = frozenset({".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".ipp"})
    SKIP_DIRS = frozenset({"build", "cmake-build-debug", "cmake-build-release"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2088",)

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
            if any(candidate is None for candidate in (left_type, right_type)):
                saw_unknown = True
                continue
            assert left_type is not None and right_type is not None

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

    def _language_signals(self, source: str) -> list[SemanticSignal]:
        """All C++-specific regex/syntax signals."""
        signals = super()._language_signals(source)

        # 1. User types
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

        # 2. Operator plus binding (type-resolved)
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

        # 3. Template presence (always emitted)
        has_template = bool(_CPP_TEMPLATE_RE.search(source))
        signals.append(
            SemanticSignal(
                key="cpp.template.present",
                value="yes" if has_template else "no",
                confidence=0.95,
                origin="syntax",
            )
        )

        # 4. Virtual / override (always emitted)
        has_virtual = bool(_CPP_VIRTUAL_RE.search(source))
        has_override = bool(_CPP_OVERRIDE_RE.search(source))
        signals.append(
            SemanticSignal(
                key="cpp.virtual_override",
                value=binary_signal_value(has_virtual, "virtual", has_override, "override"),
                confidence=0.90,
                origin="syntax",
            )
        )

        # 5. Namespace names (always emitted)
        namespaces = sorted({m.group(1) for m in _CPP_NAMESPACE_RE.finditer(source)})
        signals.append(
            SemanticSignal(
                key="cpp.namespace",
                value=",".join(namespaces) if namespaces else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 6. Const correctness (always emitted)
        has_const_method = bool(_CPP_CONST_METHOD_RE.search(source))
        has_constexpr = bool(_CPP_CONSTEXPR_RE.search(source))
        signals.append(
            SemanticSignal(
                key="cpp.const_correctness",
                value=binary_signal_value(
                    has_const_method, "const_method", has_constexpr, "constexpr"
                ),
                confidence=0.85,
                origin="syntax",
            )
        )

        # 7. C++20/23 features (always emitted)
        cpp_modern_parts = []
        if _CPP_CONCEPT_RE.search(source):
            cpp_modern_parts.append("concept")
        if _CPP_REQUIRES_RE.search(source):
            cpp_modern_parts.append("requires")
        if _CPP_CO_AWAIT_RE.search(source):
            cpp_modern_parts.append("coroutine")
        if _CPP_MODULE_RE.search(source):
            cpp_modern_parts.append("module")
        if _CPP_SPACESHIP_RE.search(source):
            cpp_modern_parts.append("spaceship")
        if _CPP_CONSTEVAL_RE.search(source):
            cpp_modern_parts.append("consteval")
        if _CPP_CONSTINIT_RE.search(source):
            cpp_modern_parts.append("constinit")
        signals.append(
            SemanticSignal(
                key="cpp.modern_features",
                value=",".join(cpp_modern_parts) if cpp_modern_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        return signals
