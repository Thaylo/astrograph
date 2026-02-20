"""Tests for LSP-backed language plugins."""

from __future__ import annotations

import re

import networkx as nx

from astrograph.canonical_hash import weisfeiler_leman_hash
from astrograph.index import CodeStructureIndex
from astrograph.languages._lsp_base import LSPPosition, LSPRange, LSPSymbol
from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin
from astrograph.languages.registry import LanguageRegistry


def _make_symbol(
    name: str,
    kind: int,
    start_line: int,
    end_line: int,
    children: tuple[LSPSymbol, ...] = (),
) -> LSPSymbol:
    return LSPSymbol(
        name=name,
        kind=kind,
        symbol_range=LSPRange(
            start=LSPPosition(line=start_line, character=0),
            end=LSPPosition(line=end_line, character=0),
        ),
        children=children,
    )


class FakeJavaScriptLSPClient:
    """Minimal symbol provider used for tests."""

    _class_re = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)")
    _function_re = re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
    _method_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\([^)]*\)\s*\{")
    _skip_method_names = frozenset({"if", "for", "while", "switch", "catch", "function"})

    def _find_block_end(self, lines: list[str], start_line: int) -> int:
        depth = 0
        saw_block = False
        for idx in range(start_line, len(lines)):
            line = lines[idx]
            opens = line.count("{")
            closes = line.count("}")
            if opens > 0:
                saw_block = True
            depth += opens - closes
            if saw_block and depth <= 0:
                return idx
        return start_line

    def _parse_class_methods(
        self,
        lines: list[str],
        class_start: int,
        class_end: int,
    ) -> tuple[LSPSymbol, ...]:
        methods: list[LSPSymbol] = []
        line_index = class_start + 1

        while line_index <= class_end and line_index < len(lines):
            match = self._method_re.match(lines[line_index])
            if match and match.group(1) not in self._skip_method_names:
                method_end = self._find_block_end(lines, line_index)
                methods.append(
                    _make_symbol(
                        name=match.group(1),
                        kind=6,  # SymbolKind.Method
                        start_line=line_index,
                        end_line=method_end,
                    )
                )
                line_index = max(line_index + 1, method_end + 1)
                continue
            line_index += 1

        return tuple(methods)

    def document_symbols(
        self,
        *,
        source: str,
        file_path: str,
        language_id: str,
    ) -> list[LSPSymbol]:
        del file_path, language_id

        lines = source.splitlines()
        symbols: list[LSPSymbol] = []
        line_index = 0

        while line_index < len(lines):
            line = lines[line_index]
            class_match = self._class_re.match(line)
            if class_match:
                class_end = self._find_block_end(lines, line_index)
                children = self._parse_class_methods(lines, line_index, class_end)
                symbols.append(
                    _make_symbol(
                        name=class_match.group(1),
                        kind=5,  # SymbolKind.Class
                        start_line=line_index,
                        end_line=class_end,
                        children=children,
                    )
                )
                line_index = max(line_index + 1, class_end + 1)
                continue

            function_match = self._function_re.match(line)
            if function_match:
                function_end = self._find_block_end(lines, line_index)
                symbols.append(
                    _make_symbol(
                        name=function_match.group(1),
                        kind=12,  # SymbolKind.Function
                        start_line=line_index,
                        end_line=function_end,
                    )
                )
                line_index = max(line_index + 1, function_end + 1)
                continue

            line_index += 1

        return symbols


class FakeImportNoiseClient:
    """Client that returns an import symbol plus a real class symbol."""

    def document_symbols(
        self,
        *,
        source: str,
        file_path: str,
        language_id: str,
    ) -> list[LSPSymbol]:
        del source, file_path, language_id
        return [
            _make_symbol(
                name="Foo",
                kind=5,  # SymbolKind.Class
                start_line=0,
                end_line=0,
            ),
            _make_symbol(
                name="Greeter",
                kind=5,  # SymbolKind.Class
                start_line=2,
                end_line=4,
            ),
        ]


class TestJavaScriptLSPPlugin:
    """Tests for JavaScriptLSPPlugin behavior."""

    def test_extract_code_units_from_lsp_symbols(self):
        source = """
class Greeter {
  greet(name) {
    return "hello " + name;
  }
}

function helper(value) {
  return value;
}
"""
        plugin = JavaScriptLSPPlugin(lsp_client=FakeJavaScriptLSPClient())
        units = list(plugin.extract_code_units(source, "sample.js"))

        assert ("Greeter", "class", None) in {(u.name, u.unit_type, u.parent_name) for u in units}
        assert ("greet", "method", "Greeter") in {
            (u.name, u.unit_type, u.parent_name) for u in units
        }
        assert ("helper", "function", None) in {(u.name, u.unit_type, u.parent_name) for u in units}
        assert all(u.language == "javascript_lsp" for u in units)

    def test_source_to_graph_ignores_identifier_names(self):
        plugin = JavaScriptLSPPlugin(lsp_client=FakeJavaScriptLSPClient())

        code_one = """
function add(a, b) {
  if (a > 0) {
    return a + b;
  }
  return b;
}
"""
        code_two = """
function sum(x, y) {
  if (x > 0) {
    return x + y;
  }
  return y;
}
"""
        graph_one = plugin.source_to_graph(code_one)
        graph_two = plugin.source_to_graph(code_two)

        assert isinstance(graph_one, nx.DiGraph)
        assert isinstance(graph_two, nx.DiGraph)
        assert weisfeiler_leman_hash(graph_one) == weisfeiler_leman_hash(graph_two)

    def test_registry_and_index_flow_for_lsp_plugin(self, tmp_path):
        class TestJavaScriptPlugin(JavaScriptLSPPlugin):
            LANGUAGE_ID = "javascript_lsp_test"
            FILE_EXTENSIONS = frozenset({".fjs"})

        plugin = TestJavaScriptPlugin(lsp_client=FakeJavaScriptLSPClient())
        registry = LanguageRegistry.get()
        registry.register(plugin)

        file_one = tmp_path / "first.fjs"
        file_two = tmp_path / "second.fjs"
        file_one.write_text(
            """
function add(a, b) {
  if (a > 0) {
    return a + b;
  }
  return b;
}
"""
        )
        file_two.write_text(
            """
function sum(x, y) {
  if (x > 0) {
    return x + y;
  }
  return y;
}
"""
        )

        index = CodeStructureIndex()
        entries = index.index_directory(str(tmp_path))

        assert registry.get_plugin_for_file(file_one) is plugin
        assert registry.get_plugin_for_file(file_two) is plugin
        assert len(entries) >= 2
        assert index.has_duplicates(min_node_count=3)

    def test_extract_code_units_skips_import_symbol_noise(self):
        source = 'import { Foo } from "./foo";\n\n' "class Greeter {\n" "  greet() {}\n" "}\n"
        plugin = JavaScriptLSPPlugin(lsp_client=FakeImportNoiseClient())
        units = list(plugin.extract_code_units(source, "sample.js"))
        assert all(unit.name != "Foo" for unit in units)
        assert any(unit.name == "Greeter" for unit in units)
