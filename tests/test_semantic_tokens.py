"""Tests for LSP semantic token decoding, TokenIndex, and server profile signals."""

from __future__ import annotations

import pytest

from astrograph.languages._semantic_tokens import (
    SemanticToken,
    SemanticTokenLegend,
    TokenIndex,
    decode_semantic_tokens,
)
from astrograph.languages._server_profiles import (
    BasedPyrightProfile,
    ClangdProfile,
    JdtlsProfile,
    TypeScriptLanguageServerProfile,
    resolve_server_profile,
)
from astrograph.languages.base import SemanticSignal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def basic_legend() -> SemanticTokenLegend:
    return SemanticTokenLegend(
        token_types=(
            "namespace",
            "type",
            "class",
            "enum",
            "interface",
            "struct",
            "typeParameter",
            "parameter",
            "variable",
            "property",
            "function",
            "method",
            "macro",
            "keyword",
            "modifier",
            "comment",
            "string",
            "number",
            "regexp",
            "operator",
            "decorator",
        ),
        token_modifiers=(
            "declaration",
            "definition",
            "readonly",
            "static",
            "deprecated",
            "abstract",
            "async",
        ),
    )


def _make_token(
    line: int = 0,
    start_char: int = 0,
    length: int = 1,
    token_type: str = "keyword",
    modifiers: frozenset[str] | None = None,
    text: str = "",
) -> SemanticToken:
    return SemanticToken(
        line=line,
        start_char=start_char,
        length=length,
        token_type=token_type,
        modifiers=modifiers or frozenset(),
        text=text,
    )


# ---------------------------------------------------------------------------
# Decoder tests
# ---------------------------------------------------------------------------


class TestDecodeSemanticTokens:
    def test_empty_data(self, basic_legend):
        result = decode_semantic_tokens([], basic_legend, [])
        assert result == ()

    def test_invalid_length(self, basic_legend):
        """Data length not divisible by 5 returns empty."""
        result = decode_semantic_tokens([0, 0, 3, 13, 0, 1], basic_legend, ["class Foo {}"])
        assert result == ()

    def test_single_token(self, basic_legend):
        # keyword "class" at line 0, char 0, length 5, type index 13 (keyword), no modifiers
        data = [0, 0, 5, 13, 0]
        lines = ["class Foo {}"]
        tokens = decode_semantic_tokens(data, basic_legend, lines)
        assert len(tokens) == 1
        assert tokens[0].line == 0
        assert tokens[0].start_char == 0
        assert tokens[0].length == 5
        assert tokens[0].token_type == "keyword"
        assert tokens[0].modifiers == frozenset()
        assert tokens[0].text == "class"

    def test_multi_line_deltas(self, basic_legend):
        # Token 1: line 0, char 0, len 5, keyword (13)
        # Token 2: delta_line=1, delta_start=4, len 3, variable (8)
        data = [0, 0, 5, 13, 0, 1, 4, 3, 8, 0]
        lines = ["class Foo {", "    bar = 1"]
        tokens = decode_semantic_tokens(data, basic_legend, lines)
        assert len(tokens) == 2
        assert tokens[0].line == 0
        assert tokens[0].text == "class"
        assert tokens[1].line == 1
        assert tokens[1].start_char == 4
        assert tokens[1].text == "bar"
        assert tokens[1].token_type == "variable"

    def test_same_line_deltas(self, basic_legend):
        # Two tokens on same line: "public" (keyword) then "class" (keyword)
        # Token 1: line 0, char 0, len 6, keyword
        # Token 2: delta_line=0, delta_start=7, len 5, keyword
        data = [0, 0, 6, 13, 0, 0, 7, 5, 13, 0]
        lines = ["public class Foo {}"]
        tokens = decode_semantic_tokens(data, basic_legend, lines)
        assert len(tokens) == 2
        assert tokens[0].text == "public"
        assert tokens[1].start_char == 7
        assert tokens[1].text == "class"

    def test_out_of_bounds_type_index(self, basic_legend):
        data = [0, 0, 3, 999, 0]
        lines = ["foo"]
        tokens = decode_semantic_tokens(data, basic_legend, lines)
        assert len(tokens) == 1
        assert tokens[0].token_type == "unknown_999"

    def test_modifier_bitmask(self, basic_legend):
        # Modifier bits: bit 0 = declaration, bit 2 = readonly → 0b101 = 5
        data = [0, 0, 3, 8, 5]
        lines = ["foo"]
        tokens = decode_semantic_tokens(data, basic_legend, lines)
        assert len(tokens) == 1
        assert tokens[0].modifiers == frozenset({"declaration", "readonly"})

    def test_out_of_bounds_line(self, basic_legend):
        """Line beyond source_lines returns empty text."""
        data = [5, 0, 3, 13, 0]
        lines = ["only one line"]
        tokens = decode_semantic_tokens(data, basic_legend, lines)
        assert len(tokens) == 1
        assert tokens[0].text == ""


# ---------------------------------------------------------------------------
# TokenIndex tests
# ---------------------------------------------------------------------------


class TestTokenIndex:
    def test_has_text_basic(self):
        tokens = (
            _make_token(text="class", token_type="keyword"),
            _make_token(text="Foo", token_type="class"),
        )
        index = TokenIndex(tokens)
        assert index.has_text("class")
        assert index.has_text("Foo")
        assert not index.has_text("bar")

    def test_has_text_with_type(self):
        tokens = (
            _make_token(text="class", token_type="keyword"),
            _make_token(text="Foo", token_type="class"),
        )
        index = TokenIndex(tokens)
        assert index.has_text("class", token_type="keyword")
        assert not index.has_text("class", token_type="class")
        assert index.has_text("Foo", token_type="class")

    def test_has_type(self):
        tokens = (_make_token(text="T", token_type="typeParameter"),)
        index = TokenIndex(tokens)
        assert index.has_type("typeParameter")
        assert not index.has_type("keyword")

    def test_has_modifier(self):
        tokens = (_make_token(text="x", modifiers=frozenset({"static", "readonly"})),)
        index = TokenIndex(tokens)
        assert index.has_modifier("static")
        assert index.has_modifier("readonly")
        assert not index.has_modifier("async")

    def test_texts_of_type(self):
        tokens = (
            _make_token(text="MyClass", token_type="class"),
            _make_token(text="Widget", token_type="class"),
            _make_token(text="public", token_type="keyword"),
        )
        index = TokenIndex(tokens)
        assert index.texts_of_type("class") == {"MyClass", "Widget"}
        assert index.texts_of_type("keyword") == {"public"}
        assert index.texts_of_type("function") == set()

    def test_count_type(self):
        tokens = (
            _make_token(text="a", token_type="variable"),
            _make_token(text="b", token_type="variable"),
            _make_token(text="c", token_type="keyword"),
        )
        index = TokenIndex(tokens)
        assert index.count_type("variable") == 2
        assert index.count_type("keyword") == 1
        assert index.count_type("class") == 0

    def test_empty_index(self):
        index = TokenIndex(())
        assert not index.has_text("anything")
        assert not index.has_type("keyword")
        assert not index.has_modifier("static")
        assert index.texts_of_type("keyword") == set()
        assert index.count_type("keyword") == 0


# ---------------------------------------------------------------------------
# Server profile tests (replaced per-plugin _token_based_signals tests)
# ---------------------------------------------------------------------------


def _signal_map(signals: list[SemanticSignal]) -> dict[str, str]:
    return {s.key: s.value for s in signals}


class TestProfileRegistry:
    """Test server profile registry and resolution."""

    def test_resolve_exact_match(self):
        profile = resolve_server_profile("clangd")
        assert profile is not None
        assert profile.name == "clangd"

    def test_resolve_substring_match(self):
        profile = resolve_server_profile("clangd version 17.0.3")
        assert profile is not None
        assert profile.name == "clangd"

    def test_resolve_vtsls_alias(self):
        profile = resolve_server_profile("vtsls")
        assert profile is not None
        assert profile.name == "typescript-language-server"

    def test_resolve_none(self):
        assert resolve_server_profile(None) is None
        assert resolve_server_profile("unknown-server") is None


class TestJdtlsProfile:
    """Test JdtlsProfile.extract_signals with hand-crafted token indices."""

    def test_annotations_detected(self):
        tokens = (
            _make_token(text="Override", token_type="annotation"),
            _make_token(text="Deprecated", token_type="annotation"),
        )
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["java.annotations"] == "Deprecated,Override"

    def test_annotations_none(self):
        tokens = (_make_token(text="class", token_type="keyword"),)
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["java.annotations"] == "none"

    def test_access_modifiers_via_modifier_tokens(self):
        tokens = (
            _make_token(text="public", token_type="modifier"),
            _make_token(text="private", token_type="modifier"),
        )
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert "public" in sig["java.access_modifiers"]
        assert "private" in sig["java.access_modifiers"]

    def test_access_modifiers_package_default(self):
        tokens = (_make_token(text="class", token_type="keyword"),)
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["java.access_modifiers"] == "package"

    def test_generic_present(self):
        tokens = (_make_token(text="T", token_type="typeParameter"),)
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["java.generic.present"] == "yes"

    def test_generic_absent(self):
        tokens = (_make_token(text="class", token_type="keyword"),)
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["java.generic.present"] == "no"

    def test_exception_handling_both(self):
        tokens = (
            _make_token(text="throws", token_type="keyword"),
            _make_token(text="try", token_type="keyword"),
            _make_token(text="catch", token_type="keyword"),
        )
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["java.exception_handling"] == "both"

    def test_class_kind_interface(self):
        tokens = (_make_token(text="interface", token_type="keyword"),)
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["java.class_kind"] == "interface"

    def test_modern_features_record(self):
        tokens = (_make_token(text="var", token_type="keyword"),)
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        modern = sig["java.modern_features"]
        assert "var" in modern

    def test_modern_features_record_type(self):
        tokens = (_make_token(text="Point", token_type="record"),)
        profile = JdtlsProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        modern = sig["java.modern_features"]
        assert "record" in modern


class TestClangdProfile:
    """Test ClangdProfile.extract_signals with hand-crafted token indices."""

    def test_user_types_from_class(self):
        tokens = (_make_token(text="MyClass", token_type="class"),)
        profile = ClangdProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["typing.user_types.present"] == "yes"

    def test_user_types_from_struct(self):
        tokens = (_make_token(text="Point", token_type="struct"),)
        profile = ClangdProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["typing.user_types.present"] == "yes"

    def test_template_detected(self):
        tokens = (_make_token(text="T", token_type="typeParameter"),)
        profile = ClangdProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["cpp.template.present"] == "yes"

    def test_virtual_override_both(self):
        tokens = (
            _make_token(text="foo", token_type="method", modifiers=frozenset({"virtual"})),
            _make_token(text="bar", token_type="method", modifiers=frozenset({"abstract"})),
        )
        profile = ClangdProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["cpp.virtual_override"] == "both"

    def test_namespace_names(self):
        tokens = (
            _make_token(text="std", token_type="namespace"),
            _make_token(text="mylib", token_type="namespace"),
        )
        profile = ClangdProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["cpp.namespace"] == "mylib,std"

    def test_const_correctness(self):
        tokens = (_make_token(text="x", token_type="variable", modifiers=frozenset({"readonly"})),)
        profile = ClangdProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["cpp.const_correctness"] == "const_method"

    def test_spaceship_operator(self):
        tokens = (_make_token(text="<=>", token_type="operator"),)
        profile = ClangdProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["cpp.has_spaceship"] == "yes"

    def test_composite_types_for_c(self):
        tokens = (
            _make_token(text="Point", token_type="struct"),
            _make_token(text="Color", token_type="enum"),
        )
        profile = ClangdProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert "struct" in sig["c.composite_types"]
        assert "enum" in sig["c.composite_types"]

    def test_macro_presence(self):
        tokens = (_make_token(text="MAX_SIZE", token_type="macro"),)
        profile = ClangdProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["c.has_macros"] == "yes"


class TestTypeScriptLanguageServerProfile:
    """Test TypeScriptLanguageServerProfile.extract_signals."""

    def test_async_via_modifier(self):
        tokens = (
            _make_token(text="fetchData", token_type="function", modifiers=frozenset({"async"})),
        )
        profile = TypeScriptLanguageServerProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["javascript.async.present"] == "yes"

    def test_async_absent(self):
        tokens = (_make_token(text="foo", token_type="function"),)
        profile = TypeScriptLanguageServerProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["javascript.async.present"] == "no"

    def test_class_pattern_via_tokens(self):
        tokens = (_make_token(text="MyWidget", token_type="class"),)
        profile = TypeScriptLanguageServerProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["javascript.class_pattern"] == "class"

    def test_generic_detected(self):
        tokens = (_make_token(text="T", token_type="typeParameter"),)
        profile = TypeScriptLanguageServerProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["typescript.generic.present"] == "yes"

    def test_namespace_detected(self):
        tokens = (
            _make_token(text="React", token_type="namespace"),
            _make_token(text="Utils", token_type="namespace"),
        )
        profile = TypeScriptLanguageServerProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["javascript.namespaces"] == "React,Utils"

    def test_enum_detected(self):
        tokens = (_make_token(text="Direction", token_type="enum"),)
        profile = TypeScriptLanguageServerProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["typescript.has_enum"] == "yes"


class TestBasedPyrightProfile:
    """Test BasedPyrightProfile.extract_signals."""

    def test_decorators_detected(self):
        tokens = (
            _make_token(text="dataclass", token_type="decorator"),
            _make_token(text="staticmethod", token_type="decorator"),
        )
        profile = BasedPyrightProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["python.decorators.present"] == "dataclass,staticmethod"

    def test_async_via_modifier(self):
        tokens = (_make_token(text="fetch", token_type="function", modifiers=frozenset({"async"})),)
        profile = BasedPyrightProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["python.async.present"] == "yes"

    def test_async_via_keyword(self):
        tokens = (_make_token(text="async", token_type="keyword"),)
        profile = BasedPyrightProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["python.async.present"] == "yes"

    def test_type_parameters(self):
        tokens = (_make_token(text="T", token_type="typeParameter"),)
        profile = BasedPyrightProfile()
        signals = profile.extract_signals(TokenIndex(tokens), "")
        sig = _signal_map(signals)
        assert sig["python.has_type_parameters"] == "yes"


# ---------------------------------------------------------------------------
# Regex fallback tests
# ---------------------------------------------------------------------------


class TestRegexFallback:
    """Verify that when semantic_tokens() returns None, the regex path is used."""

    def test_java_regex_fallback(self):
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        plugin = JavaLSPPlugin.__new__(JavaLSPPlugin)
        source = "@Override\n" "public class Foo {\n" "    private int x;\n" "}"
        profile = plugin.extract_semantic_profile(source, "Test.java")
        sig = {s.key: s.value for s in profile.signals}
        assert sig["java.annotations"] == "Override"
        assert "public" in sig["java.access_modifiers"]
        assert profile.extractor == "java_lsp:syntax"

    def test_cpp_regex_fallback(self):
        from astrograph.languages.cpp_lsp_plugin import CppLSPPlugin

        plugin = CppLSPPlugin.__new__(CppLSPPlugin)
        source = (
            "template<typename T>\n" "class Foo {\n" "    virtual void bar() override;\n" "};\n"
        )
        profile = plugin.extract_semantic_profile(source, "test.cpp")
        sig = {s.key: s.value for s in profile.signals}
        assert sig["cpp.template.present"] == "yes"
        assert sig["cpp.virtual_override"] == "both"
        assert profile.extractor == "cpp_lsp:syntax"

    def test_c_regex_fallback(self):
        from astrograph.languages.c_lsp_plugin import CLSPPlugin

        plugin = CLSPPlugin.__new__(CLSPPlugin)
        source = (
            "#include <stdio.h>\n" "struct Point { int x; int y; };\n" "int main() { return 0; }\n"
        )
        profile = plugin.extract_semantic_profile(source, "test.c")
        sig = {s.key: s.value for s in profile.signals}
        assert "include" in sig["c.preprocessor"]
        assert "struct" in sig["c.composite_types"]
        assert profile.extractor == "c_lsp:syntax"

    def test_javascript_regex_fallback(self):
        from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin

        plugin = JavaScriptLSPPlugin.__new__(JavaScriptLSPPlugin)
        source = (
            "import { foo } from './bar';\n" "async function main() {\n" "    await foo();\n" "}\n"
        )
        profile = plugin.extract_semantic_profile(source, "test.js")
        sig = {s.key: s.value for s in profile.signals}
        assert sig["javascript.async.present"] == "yes"
        assert sig["javascript.module_system"] == "esm"
        assert profile.extractor == "javascript_lsp:syntax"

    def test_typescript_regex_fallback(self):
        from astrograph.languages.typescript_lsp_plugin import TypeScriptLSPPlugin

        plugin = TypeScriptLSPPlugin.__new__(TypeScriptLSPPlugin)
        source = "function identity<T>(arg: T): T {\n" "    return arg;\n" "}\n"
        profile = plugin.extract_semantic_profile(source, "test.ts")
        sig = {s.key: s.value for s in profile.signals}
        assert sig["typescript.generic.present"] == "yes"
        assert profile.extractor == "typescript_lsp:syntax"


# ---------------------------------------------------------------------------
# Assembler integration: server profile + language signals
# ---------------------------------------------------------------------------


class TestAssemblerIntegration:
    """Verify the assembler correctly combines server profile + language signals."""

    def test_java_with_jdtls_server(self):
        from unittest.mock import MagicMock, PropertyMock

        from astrograph.languages._lsp_base import NullLSPClient
        from astrograph.languages._semantic_tokens import SemanticTokenResult
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        legend = SemanticTokenLegend(
            token_types=("keyword", "annotation", "typeParameter", "modifier"),
            token_modifiers=(),
        )
        tokens = (
            _make_token(text="public", token_type="modifier"),
            _make_token(text="class", token_type="keyword"),
            _make_token(text="Override", token_type="annotation"),
        )
        mock_result = SemanticTokenResult(tokens=tokens, legend=legend)

        mock_client = MagicMock(spec=NullLSPClient)
        mock_client.document_symbols.return_value = []
        mock_client.semantic_tokens.return_value = mock_result
        type(mock_client).server_name = PropertyMock(return_value="eclipse.jdt.ls")

        plugin = JavaLSPPlugin.__new__(JavaLSPPlugin)
        plugin._lsp_client = mock_client

        source = "@Override\npublic class Foo {}"
        profile = plugin.extract_semantic_profile(source, "Test.java")
        assert profile.extractor == "java_lsp:lsp+syntax"
        sig = {s.key: s.value for s in profile.signals}
        # Server profile provides these
        assert sig["java.annotations"] == "Override"
        assert "public" in sig["java.access_modifiers"]
        # Language signals fill gaps
        assert "java.functional_style" in sig

    def test_cpp_with_clangd_server(self):
        from unittest.mock import MagicMock, PropertyMock

        from astrograph.languages._lsp_base import NullLSPClient
        from astrograph.languages._semantic_tokens import SemanticTokenResult
        from astrograph.languages.cpp_lsp_plugin import CppLSPPlugin

        legend = SemanticTokenLegend(
            token_types=("class", "namespace", "typeParameter"),
            token_modifiers=(),
        )
        tokens = (
            _make_token(text="MyClass", token_type="class"),
            _make_token(text="T", token_type="typeParameter"),
        )
        mock_result = SemanticTokenResult(tokens=tokens, legend=legend)

        mock_client = MagicMock(spec=NullLSPClient)
        mock_client.document_symbols.return_value = []
        mock_client.semantic_tokens.return_value = mock_result
        type(mock_client).server_name = PropertyMock(return_value="clangd")

        plugin = CppLSPPlugin.__new__(CppLSPPlugin)
        plugin._lsp_client = mock_client

        source = "template<typename T> class MyClass {};"
        profile = plugin.extract_semantic_profile(source, "test.cpp")
        assert profile.extractor == "cpp_lsp:lsp+syntax"
        sig = {s.key: s.value for s in profile.signals}
        # Server profile provides
        assert sig["typing.user_types.present"] == "yes"
        assert sig["cpp.template.present"] == "yes"
        # Language signals fill remaining keys
        assert "cpp.virtual_override" in sig

    def test_no_server_name_falls_back_to_language_only(self):
        from unittest.mock import MagicMock, PropertyMock

        from astrograph.languages._lsp_base import NullLSPClient
        from astrograph.languages._semantic_tokens import SemanticTokenResult
        from astrograph.languages.java_lsp_plugin import JavaLSPPlugin

        legend = SemanticTokenLegend(
            token_types=("keyword",),
            token_modifiers=(),
        )
        tokens = (_make_token(text="class", token_type="keyword"),)
        mock_result = SemanticTokenResult(tokens=tokens, legend=legend)

        mock_client = MagicMock(spec=NullLSPClient)
        mock_client.document_symbols.return_value = []
        mock_client.semantic_tokens.return_value = mock_result
        # Unknown server → no profile resolved
        type(mock_client).server_name = PropertyMock(return_value="unknown-server")

        plugin = JavaLSPPlugin.__new__(JavaLSPPlugin)
        plugin._lsp_client = mock_client

        source = "@Override\npublic class Foo {}"
        profile = plugin.extract_semantic_profile(source, "Test.java")
        # Falls back to language-only (no server profile matched)
        assert profile.extractor == "java_lsp:syntax"

    def test_server_signals_take_priority_over_language(self):
        from unittest.mock import MagicMock, PropertyMock

        from astrograph.languages._lsp_base import NullLSPClient
        from astrograph.languages._semantic_tokens import SemanticTokenResult
        from astrograph.languages.cpp_lsp_plugin import CppLSPPlugin

        legend = SemanticTokenLegend(
            token_types=("class", "struct", "namespace"),
            token_modifiers=("readonly",),
        )
        tokens = (
            _make_token(text="MyClass", token_type="class"),
            _make_token(text="x", token_type="variable", modifiers=frozenset({"readonly"})),
        )
        mock_result = SemanticTokenResult(tokens=tokens, legend=legend)

        mock_client = MagicMock(spec=NullLSPClient)
        mock_client.document_symbols.return_value = []
        mock_client.semantic_tokens.return_value = mock_result
        type(mock_client).server_name = PropertyMock(return_value="clangd")

        plugin = CppLSPPlugin.__new__(CppLSPPlugin)
        plugin._lsp_client = mock_client

        # Source has "const" in it — language regex would detect it too
        source = "class MyClass { const int x = 5; };"
        profile = plugin.extract_semantic_profile(source, "test.cpp")
        origins = {s.key: s.origin for s in profile.signals}
        # Server profile provides typing.user_types.present with "lsp" origin
        assert origins["typing.user_types.present"] == "lsp"
        # Server covers cpp.const_correctness, so language signal is de-duped
        assert origins["cpp.const_correctness"] == "lsp"
