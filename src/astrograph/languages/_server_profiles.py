"""LSP server profiles for semantic token extraction.

Each profile encodes server-specific knowledge about how a particular LSP server
maps language concepts to semantic token types and modifiers.  The assembler in
:class:`LSPLanguagePluginBase` calls ``extract_signals`` to get what the server
can reliably provide, then overlays language-specific regex/AST signals for
everything the server does not cover.

Profiles are resolved automatically from the ``serverInfo.name`` field in the
LSP ``initialize`` response.
"""

from __future__ import annotations

from typing import Protocol

from ._semantic_tokens import TokenIndex
from .base import SemanticSignal, binary_signal_value


class ServerProfile(Protocol):
    """What a specific LSP server can provide via semantic tokens."""

    name: str

    def extract_signals(self, token_index: TokenIndex, source: str) -> list[SemanticSignal]:
        """Extract all signals this server can reliably produce."""
        ...


def _lsp_type_signal(
    signals: list[SemanticSignal],
    token_index: TokenIndex,
    token_type: str,
    key: str,
    value: str = "yes",
    confidence: float = 0.95,
) -> None:
    """Append an ``origin='lsp'`` signal if *token_type* is present."""
    if token_index.has_type(token_type):
        signals.append(SemanticSignal(key=key, value=value, confidence=confidence, origin="lsp"))


def _lsp_texts_signal(
    signals: list[SemanticSignal],
    token_index: TokenIndex,
    token_type: str,
    key: str,
    confidence: float = 0.95,
) -> None:
    """Append a comma-joined signal from sorted texts of *token_type*, if any."""
    texts = sorted(token_index.texts_of_type(token_type))
    if texts:
        signals.append(
            SemanticSignal(key=key, value=",".join(texts), confidence=confidence, origin="lsp")
        )


# ---------------------------------------------------------------------------
# clangd  (C / C++)
# ---------------------------------------------------------------------------
# clangd emits rich semantic tokens for identifiers and operators but does NOT
# emit tokens for keywords, strings, numbers, or comments.  Token types:
# class, struct, enum, enumMember, namespace, typeParameter, concept, macro,
# function, method, variable, parameter, property, operator, type, label.
# Modifiers: declaration, definition, readonly, static, virtual, abstract,
# deprecated, deduced, defaultLibrary, constructorOrDestructor, userDefined,
# functionScope, classScope, fileScope, globalScope.
# ---------------------------------------------------------------------------


class ClangdProfile:
    """Server profile for clangd (LLVM)."""

    name = "clangd"

    def extract_signals(self, token_index: TokenIndex, source: str) -> list[SemanticSignal]:
        del source
        signals: list[SemanticSignal] = []

        # User-defined types (struct/class declarations emit struct/class tokens)
        if token_index.has_type("class") or token_index.has_type("struct"):
            signals.append(
                SemanticSignal(
                    key="typing.user_types.present",
                    value="yes",
                    confidence=0.95,
                    origin="lsp",
                )
            )

        # Namespace names
        _lsp_texts_signal(signals, token_index, "namespace", "cpp.namespace")

        # Template parameters → template presence
        _lsp_type_signal(signals, token_index, "typeParameter", "cpp.template.present")

        # Spaceship operator (clangd emits operator tokens)
        if token_index.has_text("<=>", token_type="operator"):
            signals.append(
                SemanticSignal(
                    key="cpp.has_spaceship",
                    value="yes",
                    confidence=0.95,
                    origin="lsp",
                )
            )

        # Virtual / abstract modifiers on methods
        _methods = token_index._by_type.get("method", [])
        has_virtual = any("virtual" in t.modifiers for t in _methods)
        has_abstract = any("abstract" in t.modifiers for t in _methods)
        if has_virtual or has_abstract:
            signals.append(
                SemanticSignal(
                    key="cpp.virtual_override",
                    value=binary_signal_value(has_virtual, "virtual", has_abstract, "override"),
                    confidence=0.95,
                    origin="lsp",
                )
            )

        # Const correctness via readonly modifier
        has_readonly = token_index.has_modifier("readonly")
        if has_readonly:
            signals.append(
                SemanticSignal(
                    key="cpp.const_correctness",
                    value="const_method",
                    confidence=0.85,
                    origin="lsp",
                )
            )

        # Composite types for C
        type_parts = []
        if token_index.has_type("struct"):
            type_parts.append("struct")
        if token_index.has_type("enum"):
            type_parts.append("enum")
        if type_parts:
            signals.append(
                SemanticSignal(
                    key="c.composite_types",
                    value=",".join(type_parts),
                    confidence=0.90,
                    origin="lsp",
                )
            )

        # Macro presence
        _lsp_type_signal(signals, token_index, "macro", "c.has_macros")

        return signals


# ---------------------------------------------------------------------------
# Eclipse JDT Language Server  (Java)
# ---------------------------------------------------------------------------
# jdtls emits rich tokens including custom Java-specific types:
# Token types: namespace, class, interface, enum, enumMember, type,
#   typeParameter, method, property, variable, parameter, modifier, keyword,
#   annotation (custom), annotationMember (custom), record (custom),
#   recordComponent (custom).
# Modifiers: abstract, static, final, deprecated, declaration, documentation,
#   public (custom), private (custom), protected (custom), native (custom),
#   generic (custom), typeArgument (custom), importDeclaration (custom),
#   constructor (custom).
# ---------------------------------------------------------------------------


class JdtlsProfile:
    """Server profile for Eclipse JDT Language Server."""

    name = "eclipse.jdt.ls"

    def extract_signals(self, token_index: TokenIndex, source: str) -> list[SemanticSignal]:
        del source
        signals: list[SemanticSignal] = []

        # Annotations (jdtls uses custom "annotation" token type)
        annotations = sorted(token_index.texts_of_type("annotation"))
        signals.append(
            SemanticSignal(
                key="java.annotations",
                value=",".join(annotations) if annotations else "none",
                confidence=0.95,
                origin="lsp",
            )
        )

        # Access modifiers (jdtls uses "modifier" token type AND custom modifiers)
        # Check both: tokens of type "modifier" with text public/private/protected,
        # and tokens with public/private/protected as modifiers
        modifiers = []
        modifier_texts = token_index.texts_of_type("modifier")
        for kw in ("public", "private", "protected"):
            if kw in modifier_texts or token_index.has_modifier(kw):
                modifiers.append(kw)
        signals.append(
            SemanticSignal(
                key="java.access_modifiers",
                value=",".join(modifiers) if modifiers else "package",
                confidence=0.95,
                origin="lsp",
            )
        )

        # Generic usage
        has_generic = token_index.has_type("typeParameter")
        signals.append(
            SemanticSignal(
                key="java.generic.present",
                value="yes" if has_generic else "no",
                confidence=0.95,
                origin="lsp",
            )
        )

        # Exception handling (jdtls emits keyword tokens)
        keyword_texts = token_index.texts_of_type("keyword")
        has_throws = "throws" in keyword_texts
        has_try = "try" in keyword_texts
        has_catch = "catch" in keyword_texts
        has_try_catch = has_try or has_catch
        signals.append(
            SemanticSignal(
                key="java.exception_handling",
                value=binary_signal_value(has_throws, "throws", has_try_catch, "try_catch"),
                confidence=0.95,
                origin="lsp",
            )
        )

        # Class kind (jdtls emits keyword tokens for class/interface/abstract)
        has_interface_kw = "interface" in keyword_texts
        has_abstract_kw = "abstract" in keyword_texts or token_index.has_modifier("abstract")
        has_class_type = token_index.has_type("class")
        has_interface_type = token_index.has_type("interface")
        if has_interface_kw or has_interface_type:
            class_kind = "interface"
        elif has_abstract_kw:
            class_kind = "abstract"
        elif has_class_type:
            class_kind = "class"
        else:
            class_kind = "none"
        signals.append(
            SemanticSignal(
                key="java.class_kind",
                value=class_kind,
                confidence=0.95,
                origin="lsp",
            )
        )

        # Modern Java features
        modern_parts = []
        if token_index.has_type("record"):
            modern_parts.append("record")
        if "sealed" in keyword_texts or "sealed" in token_index.texts_of_type("modifier"):
            modern_parts.append("sealed")
        if "instanceof" in keyword_texts:
            modern_parts.append("pattern_instanceof")
        if "var" in keyword_texts:
            modern_parts.append("var")
        signals.append(
            SemanticSignal(
                key="java.modern_features",
                value=",".join(modern_parts) if modern_parts else "none",
                confidence=0.95,
                origin="lsp",
            )
        )

        return signals


# ---------------------------------------------------------------------------
# typescript-language-server / vtsls  (JavaScript / TypeScript)
# ---------------------------------------------------------------------------
# Both wrap TypeScript's getEncodedSemanticClassifications and return the same
# token set.  Token types: class, enum, interface, namespace, typeParameter,
# type, parameter, variable, enumMember, property, function, member.
# Modifiers: declaration, static, async, readonly, defaultLibrary, local.
# Does NOT emit: keyword, operator, string, number, comment, decorator.
# Decorators are classified as "function".
# ---------------------------------------------------------------------------


class TypeScriptLanguageServerProfile:
    """Server profile for typescript-language-server and vtsls."""

    name = "typescript-language-server"

    def extract_signals(self, token_index: TokenIndex, source: str) -> list[SemanticSignal]:
        del source
        signals: list[SemanticSignal] = []

        # Async detection (via modifier on function/member tokens)
        has_async = token_index.has_modifier("async")
        signals.append(
            SemanticSignal(
                key="javascript.async.present",
                value="yes" if has_async else "no",
                confidence=0.95,
                origin="lsp",
            )
        )

        # Class pattern (ES6 class via class token type)
        _lsp_type_signal(signals, token_index, "class", "javascript.class_pattern", value="class")

        # Generic usage (TypeScript)
        _lsp_type_signal(signals, token_index, "typeParameter", "typescript.generic.present")

        # Namespace detection
        _lsp_texts_signal(signals, token_index, "namespace", "javascript.namespaces")

        # Enum detection
        _lsp_type_signal(signals, token_index, "enum", "typescript.has_enum")

        return signals


# ---------------------------------------------------------------------------
# basedpyright  (Python)
# ---------------------------------------------------------------------------
# basedpyright re-implements Pylance's semantic highlighting as open-source.
# Rich token set: class, function, method, variable, parameter, property,
# namespace, type, typeParameter, decorator, keyword, string, number,
# operator, comment.
# Modifiers: declaration, definition, readonly, static, deprecated, abstract,
# async, defaultLibrary.
# ---------------------------------------------------------------------------


class BasedPyrightProfile:
    """Server profile for basedpyright."""

    name = "basedpyright"

    def extract_signals(self, token_index: TokenIndex, source: str) -> list[SemanticSignal]:
        del source
        signals: list[SemanticSignal] = []

        # Decorators (basedpyright emits "decorator" token type)
        _lsp_texts_signal(signals, token_index, "decorator", "python.decorators.present")

        # Async detection (via modifier or keyword)
        has_async = token_index.has_modifier("async") or token_index.has_text(
            "async", token_type="keyword"
        )
        signals.append(
            SemanticSignal(
                key="python.async.present",
                value="yes" if has_async else "no",
                confidence=0.95,
                origin="lsp",
            )
        )

        # Type parameter usage (Python 3.12+ generics)
        _lsp_type_signal(signals, token_index, "typeParameter", "python.has_type_parameters")

        return signals


# ---------------------------------------------------------------------------
# Profile registry — maps serverInfo.name → profile instance
# ---------------------------------------------------------------------------

_PROFILE_REGISTRY: dict[str, ServerProfile] = {}


def _register_defaults() -> None:
    """Populate the registry with built-in profiles."""
    defaults: tuple[ServerProfile, ...] = (
        ClangdProfile(),
        JdtlsProfile(),
        TypeScriptLanguageServerProfile(),
        BasedPyrightProfile(),
    )
    for profile in defaults:
        _PROFILE_REGISTRY[profile.name] = profile
    # Aliases
    _PROFILE_REGISTRY["vtsls"] = _PROFILE_REGISTRY["typescript-language-server"]


_register_defaults()


def resolve_server_profile(server_name: str | None) -> ServerProfile | None:
    """Look up a server profile by ``serverInfo.name``.

    Uses substring matching so that e.g. ``"clangd version 17.0.3"`` still
    resolves to the clangd profile.
    """
    if not server_name:
        return None
    # Exact match first
    if server_name in _PROFILE_REGISTRY:
        return _PROFILE_REGISTRY[server_name]
    # Substring match (server names often include version info)
    lower = server_name.lower()
    for key, profile in _PROFILE_REGISTRY.items():
        if key.lower() in lower:
            return profile
    return None
