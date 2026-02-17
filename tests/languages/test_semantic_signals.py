"""Tests for JS and Python plugin semantic signal extraction.

Targets coverage gaps in:
- javascript_lsp_plugin.py lines 533-660, 748-842
- python_lsp_plugin.py lines 145-304
"""

from __future__ import annotations

import pytest

from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin
from astrograph.languages.python_lsp_plugin import PythonLSPPlugin
from astrograph.languages.typescript_lsp_plugin import TypeScriptLSPPlugin


class TestTreeSitterAnnotationName:
    """Cover _ts_annotation_name with real tree-sitter parsed nodes."""

    def test_none_annotation(self):
        from astrograph.languages._js_ts_treesitter import _ts_annotation_name

        assert _ts_annotation_name(None) is None

    def test_predefined_type_number(self):
        """Parse TS snippet and extract 'number' from annotation."""
        from astrograph.languages._js_ts_treesitter import (
            _ts_build_annotation_map,
            _ts_try_parse,
        )

        tree = _ts_try_parse("function f(x: number) { return x; }", "typescript")
        assert tree is not None
        ann_map = _ts_build_annotation_map(tree)
        assert ann_map.get("x") == "number"

    def test_predefined_type_string(self):
        from astrograph.languages._js_ts_treesitter import (
            _ts_build_annotation_map,
            _ts_try_parse,
        )

        tree = _ts_try_parse("function f(s: string) { return s; }", "typescript")
        assert tree is not None
        ann_map = _ts_build_annotation_map(tree)
        assert ann_map.get("s") == "string"

    def test_predefined_type_boolean(self):
        from astrograph.languages._js_ts_treesitter import (
            _ts_build_annotation_map,
            _ts_try_parse,
        )

        tree = _ts_try_parse("function f(b: boolean) { return b; }", "typescript")
        assert tree is not None
        ann_map = _ts_build_annotation_map(tree)
        assert ann_map.get("b") == "boolean"

    def test_type_reference(self):
        from astrograph.languages._js_ts_treesitter import (
            _ts_build_annotation_map,
            _ts_try_parse,
        )

        tree = _ts_try_parse("function f(x: MyType) { return x; }", "typescript")
        assert tree is not None
        ann_map = _ts_build_annotation_map(tree)
        assert ann_map.get("x") == "MyType"

    def test_multiple_params(self):
        from astrograph.languages._js_ts_treesitter import (
            _ts_build_annotation_map,
            _ts_try_parse,
        )

        tree = _ts_try_parse("function f(a: number, b: string) { return a; }", "typescript")
        assert tree is not None
        ann_map = _ts_build_annotation_map(tree)
        assert ann_map.get("a") == "number"
        assert ann_map.get("b") == "string"

    def test_no_annotations(self):
        from astrograph.languages._js_ts_treesitter import (
            _ts_build_annotation_map,
            _ts_try_parse,
        )

        tree = _ts_try_parse("function f(x) { return x; }", "javascript")
        assert tree is not None
        ann_map = _ts_build_annotation_map(tree)
        assert len(ann_map) == 0


class TestTreeSitterGraphLabels:
    """Cover tree-sitter node label mappings and edge cases."""

    def test_generator_function(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("function* gen() { yield 1; }", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert "FunctionDeclaration" in labels

    def test_logical_expression(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("const x = a && b;", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert any("LogicalExpression:" in lbl for lbl in labels)

    def test_update_expression(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("let x = 0; x++;", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert any("UpdateExpression:" in lbl for lbl in labels)

    def test_var_declaration(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("var x = 1;", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert "VariableDeclaration:var" in labels

    def test_let_declaration(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("let x = 1;", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert "VariableDeclaration:let" in labels

    def test_bool_and_null_literals(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("const a = true; const b = null;", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert "Literal:bool" in labels
        assert "Literal:NoneType" in labels

    def test_template_literal(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("const x = `hello ${name}`;", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert "TemplateLiteral" in labels

    def test_getter_setter_method(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph(
            "class Foo { get bar() { return 1; } set bar(v) { this._bar = v; } }",
            language="javascript",
        )
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert "MethodDefinition:get" in labels
        assert "MethodDefinition:set" in labels

    def test_as_expression_skips_cast(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("const x = value as string;", language="typescript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        # as_expression wrapper should be skipped; inner expression kept
        assert "as_expression" not in labels

    def test_parse_failure_returns_none(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        _result = _ts_ast_to_graph("{{{{invalid", language="javascript")
        # tree-sitter is resilient; may return partial graph or None
        # (just verifying no exception)

    def test_method_block_extraction(self):
        from astrograph.languages._js_ts_treesitter import (
            _ts_extract_function_blocks,
            _ts_try_parse,
        )

        source = (
            "class Foo {\n"
            "  process(items) {\n"
            "    for (const item of items) {\n"
            "      console.log(item);\n"
            "    }\n"
            "  }\n"
            "}"
        )
        tree = _ts_try_parse(source, "javascript")
        assert tree is not None
        blocks = list(
            _ts_extract_function_blocks(
                tree, source.splitlines(), "test.js", max_depth=3, language="javascript_lsp"
            )
        )
        assert len(blocks) >= 1
        assert any(".for_in_" in b.name or ".for_" in b.name for b in blocks)

    def test_generic_type_annotation(self):
        from astrograph.languages._js_ts_treesitter import (
            _ts_build_annotation_map,
            _ts_try_parse,
        )

        tree = _ts_try_parse("function f(items: Array<number>) { return items; }", "typescript")
        assert tree is not None
        ann_map = _ts_build_annotation_map(tree)
        assert ann_map.get("items") == "Array"

    def test_anonymous_arrow_function(self):
        from astrograph.languages._js_ts_treesitter import (
            _ts_extract_function_blocks,
            _ts_try_parse,
        )

        source = "const f = (x) => { if (x > 0) { return x; } return 0; };"
        tree = _ts_try_parse(source, "javascript")
        assert tree is not None
        blocks = list(
            _ts_extract_function_blocks(
                tree, source.splitlines(), "test.js", max_depth=3, language="javascript_lsp"
            )
        )
        assert len(blocks) >= 1

    def test_function_expression(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("const f = function() { return 1; };", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert "FunctionExpression" in labels

    def test_default_param_annotation(self):
        from astrograph.languages._js_ts_treesitter import (
            _ts_build_annotation_map,
            _ts_try_parse,
        )

        tree = _ts_try_parse("function f(x: number = 0) { return x; }", "typescript")
        assert tree is not None
        ann_map = _ts_build_annotation_map(tree)
        assert ann_map.get("x") == "number"

    def test_ts_method_blocks(self):
        """Method definitions inside classes should extract inner blocks."""
        from astrograph.languages._js_ts_treesitter import (
            _ts_extract_function_blocks,
            _ts_try_parse,
        )

        source = (
            "class Svc {\n"
            "  handle(req: Request): void {\n"
            "    if (req.ok) {\n"
            "      console.log('ok');\n"
            "    }\n"
            "  }\n"
            "}"
        )
        tree = _ts_try_parse(source, "typescript")
        assert tree is not None
        blocks = list(
            _ts_extract_function_blocks(
                tree, source.splitlines(), "test.ts", max_depth=3, language="typescript_lsp"
            )
        )
        assert len(blocks) >= 1
        assert any(".if_" in b.name for b in blocks)

    def test_unary_expression(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("const x = !flag;", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert any("UnaryExpression:" in lbl for lbl in labels)

    def test_assignment_expression(self):
        from astrograph.languages._js_ts_treesitter import _ts_ast_to_graph

        graph = _ts_ast_to_graph("let x; x = 5;", language="javascript")
        assert graph is not None
        labels = [d["label"] for _, d in graph.nodes(data=True)]
        assert any("AssignmentExpression:" in lbl for lbl in labels)


class TestJavaScriptPlusBinding:
    """Cover _infer_plus_binding (lines 603-641)."""

    def test_no_plus_returns_none(self):
        plugin = JavaScriptLSPPlugin()
        result = plugin._infer_plus_binding("const x = 1;")
        assert result is None

    def test_numeric_plus(self):
        plugin = JavaScriptLSPPlugin()
        result = plugin._infer_plus_binding("const x = 1 + 2;")
        assert result is not None
        assert result[0] == "numeric"

    def test_string_concat(self):
        plugin = JavaScriptLSPPlugin()
        result = plugin._infer_plus_binding('const x = "hello" + "world";')
        assert result is not None
        assert result[0] == "str_concat"

    def test_unknown_plus(self):
        plugin = JavaScriptLSPPlugin()
        result = plugin._infer_plus_binding("const x = a + b;")
        assert result is not None
        assert result[0] == "unknown"

    def test_parse_failure_returns_none(self):
        plugin = JavaScriptLSPPlugin()
        result = plugin._infer_plus_binding("this is not valid {{{ javascript")
        assert result is None


class TestJavaScriptModuleSystem:
    """Cover _detect_module_system (lines 643-653)."""

    def test_esm(self):
        plugin = JavaScriptLSPPlugin()
        assert plugin._detect_module_system('import { foo } from "bar";') == "esm"

    def test_commonjs(self):
        plugin = JavaScriptLSPPlugin()
        assert plugin._detect_module_system('const foo = require("bar");') == "commonjs"

    def test_mixed(self):
        plugin = JavaScriptLSPPlugin()
        source = 'import { foo } from "bar";\nconst baz = require("qux");'
        assert plugin._detect_module_system(source) == "mixed"

    def test_none(self):
        plugin = JavaScriptLSPPlugin()
        assert plugin._detect_module_system("const x = 1;") == "none"


class TestJavaScriptClassPattern:
    """Cover _detect_class_pattern (lines 655-665)."""

    def test_es6_class(self):
        plugin = JavaScriptLSPPlugin()
        assert plugin._detect_class_pattern("class Foo {}") == "class"

    def test_prototype(self):
        plugin = JavaScriptLSPPlugin()
        assert plugin._detect_class_pattern("Foo.prototype.bar = function() {};") == "prototype"

    def test_both(self):
        plugin = JavaScriptLSPPlugin()
        source = "class Foo {}\nBar.prototype.baz = function() {};"
        assert plugin._detect_class_pattern(source) == "class"

    def test_neither(self):
        plugin = JavaScriptLSPPlugin()
        assert plugin._detect_class_pattern("const x = 1;") is None


class TestJavaScriptDecorators:
    """Cover _collect_decorators (line 667-669)."""

    def test_decorators_found(self):
        plugin = JavaScriptLSPPlugin()
        source = "@Injectable\nclass Foo {}\n@Component\nclass Bar {}"
        decorators = plugin._collect_decorators(source)
        assert "Injectable" in decorators
        assert "Component" in decorators

    def test_no_decorators(self):
        plugin = JavaScriptLSPPlugin()
        assert plugin._collect_decorators("const x = 1;") == set()


class TestJavaScriptLanguageSignals:
    """Cover _language_signals detection patterns (lines 748-842)."""

    def test_express_framework(self):
        plugin = JavaScriptLSPPlugin()
        source = (
            'const express = require("express");\nconst app = express();\napp.get("/", handler);'
        )
        signals = plugin._language_signals(source)
        http = next(s for s in signals if s.key == "javascript.http_framework")
        assert "express" in http.value

    def test_mongoose_database(self):
        plugin = JavaScriptLSPPlugin()
        source = 'const mongoose = require("mongoose");\nnew mongoose.Schema({ name: String });'
        signals = plugin._language_signals(source)
        db = next(s for s in signals if s.key == "javascript.database_client")
        assert "mongoose" in db.value

    def test_jwt_auth(self):
        plugin = JavaScriptLSPPlugin()
        source = 'const jwt = require("jsonwebtoken");\njwt.sign(payload, secret);'
        signals = plugin._language_signals(source)
        auth = next(s for s in signals if s.key == "javascript.auth_patterns")
        assert "jwt" in auth.value

    def test_socketio_realtime(self):
        plugin = JavaScriptLSPPlugin()
        source = 'const io = require("socket.io")(server);\nio.on("connection", fn);'
        signals = plugin._language_signals(source)
        rt = next(s for s in signals if s.key == "javascript.realtime_messaging")
        assert "socketio" in rt.value

    def test_websocket_realtime(self):
        plugin = JavaScriptLSPPlugin()
        source = "const ws = new WebSocket('ws://localhost:8080');"
        signals = plugin._language_signals(source)
        rt = next(s for s in signals if s.key == "javascript.realtime_messaging")
        assert "websocket" in rt.value

    def test_body_parser_middleware(self):
        plugin = JavaScriptLSPPlugin()
        source = "app.use(bodyParser.json());"
        signals = plugin._language_signals(source)
        mw = next(s for s in signals if s.key == "javascript.middleware_patterns")
        assert "body_parser" in mw.value

    def test_no_frameworks(self):
        plugin = JavaScriptLSPPlugin()
        source = "function add(a, b) { return a + b; }"
        signals = plugin._language_signals(source)
        http = next(s for s in signals if s.key == "javascript.http_framework")
        assert http.value == "none"


class TestTreeSitterResolveOperandType:
    """Cover _resolve_ts_operand_type with real tree-sitter parsed nodes."""

    @staticmethod
    def _resolve_node_operand_type(source: str, target_node_type: str):
        from astrograph.languages._js_ts_treesitter import (
            _resolve_ts_operand_type,
            _ts_try_parse,
            _ts_walk,
        )

        tree = _ts_try_parse(source, "javascript")
        assert tree is not None
        target = None
        for node in _ts_walk(tree.root_node):
            if node.type == target_node_type:
                target = node
                break
        assert target is not None
        return _resolve_ts_operand_type(target, {})

    def test_identifier_with_annotation(self):
        from astrograph.languages._js_ts_treesitter import (
            _resolve_ts_operand_type,
            _ts_try_parse,
            _ts_walk,
        )

        tree = _ts_try_parse("const x = a + b;", "javascript")
        assert tree is not None
        # Find an identifier node named 'a'
        ident = None
        for node in _ts_walk(tree.root_node):
            if node.type == "identifier" and node.text == b"a":
                ident = node
                break
        assert ident is not None
        result = _resolve_ts_operand_type(ident, {"a": "number"})
        assert result == "number"

    @pytest.mark.parametrize(
        ("source", "target_node_type", "expected"),
        [
            ('const x = "hello";', "string", "string"),
            ("const x = 42;", "number", "number"),
            ("const x = `hello`;", "template_string", "string"),
            ("const x = foo();", "call_expression", None),
        ],
        ids=["literal_string", "literal_number", "template_literal", "unknown_type"],
    )
    def test_literal_and_unknown_nodes(
        self, source: str, target_node_type: str, expected: str | None
    ):
        result = self._resolve_node_operand_type(source, target_node_type)
        assert result == expected


# ---- Python LSP plugin semantic signal tests ----


class TestPythonAnnotationName:
    """Cover _annotation_to_name edge cases (lines 145-149)."""

    def test_constant_string_annotation(self):
        import ast

        plugin = PythonLSPPlugin()
        # Forward reference annotation: "MyType"
        node = ast.Constant(value="MyType")
        result = plugin._annotation_to_name(node)
        assert result == "MyType"

    def test_attribute_annotation(self):
        import ast

        plugin = PythonLSPPlugin()
        node = ast.Attribute(attr="Optional", value=ast.Name(id="typing"))
        result = plugin._annotation_to_name(node)
        assert result == "Optional"

    def test_none_annotation(self):
        plugin = PythonLSPPlugin()
        assert plugin._annotation_to_name(None) is None

    def test_complex_annotation_returns_none(self):
        import ast

        plugin = PythonLSPPlugin()
        node = ast.Subscript(value=ast.Name(id="List"), slice=ast.Name(id="int"))
        assert plugin._annotation_to_name(node) is None


class TestPythonPlusBinding:
    """Cover _infer_plus_binding (lines 233-272)."""

    def _parse(self, source):
        import ast

        return ast.parse(source)

    def test_numeric_plus(self):
        plugin = PythonLSPPlugin()
        tree = self._parse("def f(x: int, y: int):\n    return x + y")
        result = plugin._infer_plus_binding(tree)
        assert result is not None
        assert result[0] == "numeric"

    def test_string_concat(self):
        plugin = PythonLSPPlugin()
        tree = self._parse('def f():\n    return "a" + "b"')
        result = plugin._infer_plus_binding(tree)
        assert result is not None
        assert result[0] == "str_concat"

    def test_user_defined_types(self):
        plugin = PythonLSPPlugin()
        tree = self._parse("def f(a: MyType, b: MyType):\n    return a + b")
        result = plugin._infer_plus_binding(tree)
        assert result is not None
        assert result[0] == "user_defined"

    def test_mixed_types(self):
        plugin = PythonLSPPlugin()
        tree = self._parse('def f(x: int, s: str):\n    a = x + x\n    b = "a" + s')
        result = plugin._infer_plus_binding(tree)
        assert result is not None
        assert result[0] == "mixed"

    def test_no_plus(self):
        plugin = PythonLSPPlugin()
        tree = self._parse("def f():\n    return 1 - 2")
        result = plugin._infer_plus_binding(tree)
        assert result is None

    def test_unknown_operands(self):
        plugin = PythonLSPPlugin()
        tree = self._parse("def f():\n    return a + b")
        result = plugin._infer_plus_binding(tree)
        assert result is not None
        assert result[0] == "unknown"


class TestPythonClassStyle:
    """Cover _detect_class_style (lines 274-306)."""

    def test_dataclass(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "@dataclass\nclass Foo:\n    x: int = 0"
        tree = ast.parse(source)
        result = plugin._detect_class_style(tree)
        assert result == "dataclass"

    def test_protocol(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "class Foo(Protocol):\n    def bar(self): ..."
        tree = ast.parse(source)
        result = plugin._detect_class_style(tree)
        assert result == "protocol"

    def test_abstract(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "class Foo(ABC):\n    pass"
        tree = ast.parse(source)
        result = plugin._detect_class_style(tree)
        assert result == "abstract"

    def test_plain(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "class Foo:\n    pass"
        tree = ast.parse(source)
        result = plugin._detect_class_style(tree)
        assert result == "plain"

    def test_no_class(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "def foo():\n    pass"
        tree = ast.parse(source)
        result = plugin._detect_class_style(tree)
        assert result is None

    def test_dataclass_via_call(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "@dataclass(frozen=True)\nclass Foo:\n    x: int = 0"
        tree = ast.parse(source)
        result = plugin._detect_class_style(tree)
        assert result == "dataclass"

    def test_abstract_via_attribute(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "@abc.abstractmethod\nclass Foo(abc.ABC):\n    pass"
        tree = ast.parse(source)
        result = plugin._detect_class_style(tree)
        assert result == "abstract"


class TestPythonAnnotationDensity:
    """Cover _compute_annotation_density (lines 151-176)."""

    def test_full_annotations(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "def f(x: int, y: str) -> bool:\n    pass"
        tree = ast.parse(source)
        assert plugin._compute_annotation_density(tree) == "full"

    def test_partial_annotations(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "def f(x: int, y):\n    pass"
        tree = ast.parse(source)
        assert plugin._compute_annotation_density(tree) == "partial"

    def test_no_annotations(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "def f(x, y):\n    pass"
        tree = ast.parse(source)
        assert plugin._compute_annotation_density(tree) == "none"

    def test_no_functions(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "x = 1"
        tree = ast.parse(source)
        assert plugin._compute_annotation_density(tree) == "none"

    def test_self_excluded(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "class Foo:\n    def bar(self, x: int) -> int:\n        pass"
        tree = ast.parse(source)
        # self is excluded, x is annotated, return annotated → full
        assert plugin._compute_annotation_density(tree) == "full"


class TestPythonDunderMethods:
    """Cover _collect_dunder_methods (lines 124-137)."""

    def test_finds_dunders(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "class Foo:\n    def __init__(self): pass\n    def __str__(self): pass"
        tree = ast.parse(source)
        dunders = plugin._collect_dunder_methods(tree)
        assert "__init__" in dunders
        assert "__str__" in dunders

    def test_no_dunders(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "class Foo:\n    def bar(self): pass"
        tree = ast.parse(source)
        dunders = plugin._collect_dunder_methods(tree)
        assert dunders == set()


class TestPythonDecorators:
    """Cover _collect_decorators (lines 178-195)."""

    def test_simple_decorator(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "@staticmethod\ndef f(): pass"
        tree = ast.parse(source)
        decorators = plugin._collect_decorators(tree)
        assert "staticmethod" in decorators

    def test_attribute_decorator(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "@app.route\ndef f(): pass"
        tree = ast.parse(source)
        decorators = plugin._collect_decorators(tree)
        assert "route" in decorators

    def test_call_decorator(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "@pytest.mark.parametrize('x', [1])\ndef f(): pass"
        tree = ast.parse(source)
        decorators = plugin._collect_decorators(tree)
        assert "parametrize" in decorators


class TestPythonAsyncDetection:
    """Cover _detect_async_constructs (lines 197-202)."""

    def test_async_function(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "async def f():\n    await something()"
        tree = ast.parse(source)
        assert plugin._detect_async_constructs(tree) is True

    def test_no_async(self):
        plugin = PythonLSPPlugin()
        import ast

        source = "def f():\n    pass"
        tree = ast.parse(source)
        assert plugin._detect_async_constructs(tree) is False


class TestPythonSemanticProfile:
    """Cover extract_semantic_profile integration (lines 325-346)."""

    def test_profile_with_dunders(self):
        plugin = PythonLSPPlugin()
        source = "class Foo:\n    def __init__(self): pass\n    def __eq__(self, o): pass"
        profile = plugin.extract_semantic_profile(source)
        keys = {s.key for s in profile.signals}
        assert "python.dunder_methods.defined" in keys

    def test_profile_with_decorators(self):
        plugin = PythonLSPPlugin()
        source = "@staticmethod\ndef f():\n    pass"
        profile = plugin.extract_semantic_profile(source)
        keys = {s.key for s in profile.signals}
        assert "python.decorators.present" in keys

    def test_profile_syntax_error(self):
        plugin = PythonLSPPlugin()
        profile = plugin.extract_semantic_profile("def f(:\n  pass")
        # Should return base profile without crashing
        assert profile is not None

    def test_profile_async_code(self):
        plugin = PythonLSPPlugin()
        source = "async def handler():\n    await db.query()"
        profile = plugin.extract_semantic_profile(source)
        async_signal = next((s for s in profile.signals if s.key == "python.async.present"), None)
        assert async_signal is not None
        assert async_signal.value == "yes"

    def test_profile_with_class_style(self):
        plugin = PythonLSPPlugin()
        source = "@dataclass\nclass Config:\n    host: str = 'localhost'\n    port: int = 8080"
        profile = plugin.extract_semantic_profile(source)
        style_signal = next((s for s in profile.signals if s.key == "python.class_style"), None)
        assert style_signal is not None
        assert style_signal.value == "dataclass"

    def test_profile_with_plus_binding(self):
        plugin = PythonLSPPlugin()
        source = "def add(x: int, y: int):\n    return x + y"
        profile = plugin.extract_semantic_profile(source)
        plus_signal = next((s for s in profile.signals if s.key == "python.plus_binding"), None)
        assert plus_signal is not None
        assert plus_signal.value == "numeric"

    def test_profile_annotation_density(self):
        plugin = PythonLSPPlugin()
        source = "def f(x: int, y: str) -> bool:\n    return True"
        profile = plugin.extract_semantic_profile(source)
        density_signal = next(
            (s for s in profile.signals if s.key == "python.type_annotation.density"), None
        )
        assert density_signal is not None
        assert density_signal.value == "full"


# ---- Additional JS _language_signals coverage ----


class TestJavaScriptLanguageSignalsExtended:
    """Cover remaining uncovered branches in _language_signals."""

    def test_koa_framework(self):
        plugin = JavaScriptLSPPlugin()
        source = "const app = new Koa();\napp.use(ctx => ctx.body = 'ok');"
        signals = plugin._language_signals(source)
        http = next(s for s in signals if s.key == "javascript.http_framework")
        assert "koa" in http.value

    def test_hapi_framework(self):
        plugin = JavaScriptLSPPlugin()
        source = "const server = Hapi.server({ port: 3000 });"
        signals = plugin._language_signals(source)
        http = next(s for s in signals if s.key == "javascript.http_framework")
        assert "hapi" in http.value

    def test_sequelize_database(self):
        plugin = JavaScriptLSPPlugin()
        source = "const seq = new Sequelize('sqlite::memory:');"
        signals = plugin._language_signals(source)
        db = next(s for s in signals if s.key == "javascript.database_client")
        assert "sequelize" in db.value

    def test_knex_database(self):
        plugin = JavaScriptLSPPlugin()
        source = "const db = knex({ client: 'pg' });"
        signals = plugin._language_signals(source)
        db = next(s for s in signals if s.key == "javascript.database_client")
        assert "knex" in db.value

    def test_mongodb_native_database(self):
        plugin = JavaScriptLSPPlugin()
        source = "const client = new MongoClient(uri);"
        signals = plugin._language_signals(source)
        db = next(s for s in signals if s.key == "javascript.database_client")
        assert "mongodb_native" in db.value

    def test_redis_database(self):
        plugin = JavaScriptLSPPlugin()
        source = "const client = redis.createClient();"
        signals = plugin._language_signals(source)
        db = next(s for s in signals if s.key == "javascript.database_client")
        assert "redis" in db.value

    def test_prisma_database(self):
        plugin = JavaScriptLSPPlugin()
        source = "const prisma = new PrismaClient();"
        signals = plugin._language_signals(source)
        db = next(s for s in signals if s.key == "javascript.database_client")
        assert "prisma" in db.value

    def test_oauth_auth(self):
        plugin = JavaScriptLSPPlugin()
        source = "passport.use(new OAuth2Strategy(options, verify));"
        signals = plugin._language_signals(source)
        auth = next(s for s in signals if s.key == "javascript.auth_patterns")
        assert "oauth" in auth.value

    def test_session_auth(self):
        plugin = JavaScriptLSPPlugin()
        source = "if (req.isAuthenticated()) { next(); }"
        signals = plugin._language_signals(source)
        auth = next(s for s in signals if s.key == "javascript.auth_patterns")
        assert "session_auth" in auth.value

    def test_event_emitter_realtime(self):
        plugin = JavaScriptLSPPlugin()
        source = "class Bus extends EventEmitter {}"
        signals = plugin._language_signals(source)
        rt = next(s for s in signals if s.key == "javascript.realtime_messaging")
        assert "event_emitter" in rt.value

    def test_message_queue_realtime(self):
        plugin = JavaScriptLSPPlugin()
        source = "await channel.sendToQueue('task', Buffer.from(msg));"
        signals = plugin._language_signals(source)
        rt = next(s for s in signals if s.key == "javascript.realtime_messaging")
        assert "message_queue" in rt.value


class TestJavaScriptTypeSystem:
    """Cover _detect_type_system branches."""

    def test_flow_type(self):
        plugin = JavaScriptLSPPlugin()
        assert plugin._detect_type_system("// @flow\nconst x: number = 1;") == "flow"

    def test_jsdoc_type(self):
        plugin = JavaScriptLSPPlugin()
        assert plugin._detect_type_system("/** @param {number} x */\nfunction f(x) {}") == "jsdoc"

    def test_typescript_and_jsdoc(self):
        plugin = JavaScriptLSPPlugin()
        source = "/** @param {number} x */\ninterface Foo { x: string; }"
        assert plugin._detect_type_system(source) == "typescript"


class TestJavaScriptPlusBindingExtended:
    """Cover remaining _infer_plus_binding branches."""

    def test_mixed_plus_binding(self):
        plugin = JavaScriptLSPPlugin()
        source = 'const a = 1 + 2;\nconst b = "x" + "y";'
        result = plugin._infer_plus_binding(source)
        assert result is not None
        assert result[0] == "mixed"


# ---- Additional TS _language_signals coverage ----


class TestTypeScriptLanguageSignalsExtended:
    """Cover uncovered branches in TypeScript _language_signals."""

    def test_nest_guard_pipe_interceptor(self):
        plugin = TypeScriptLSPPlugin()
        source = (
            "@UseGuards(AuthGuard)\n"
            "@UsePipes(ValidationPipe)\n"
            "@UseInterceptors(LoggingInterceptor)\n"
            "export class AppController {}"
        )
        signals = plugin._language_signals(source)
        fd = next(s for s in signals if s.key == "typescript.framework_decorators")
        assert "guard" in fd.value
        assert "pipe" in fd.value
        assert "interceptor" in fd.value

    def test_nest_module(self):
        plugin = TypeScriptLSPPlugin()
        source = "@Module({ imports: [] })\nexport class AppModule {}"
        signals = plugin._language_signals(source)
        fd = next(s for s in signals if s.key == "typescript.framework_decorators")
        assert "module" in fd.value

    def test_http_method_and_param_decorators(self):
        plugin = TypeScriptLSPPlugin()
        source = "@Get('/users')\nasync getUsers(@Query() query: any) {}"
        signals = plugin._language_signals(source)
        rh = next(s for s in signals if s.key == "typescript.rest_http_patterns")
        assert "http_method" in rh.value
        assert "param_decorator" in rh.value

    def test_http_code_and_fastify(self):
        plugin = TypeScriptLSPPlugin()
        source = "@HttpCode(201)\nconst app = Fastify();"
        signals = plugin._language_signals(source)
        rh = next(s for s in signals if s.key == "typescript.rest_http_patterns")
        assert "status_header" in rh.value
        assert "fastify" in rh.value

    def test_typeorm_entity_column_relation(self):
        plugin = TypeScriptLSPPlugin()
        source = (
            "@Entity()\nexport class User {\n"
            "  @PrimaryGeneratedColumn() id: number;\n"
            "  @Column() name: string;\n"
            "  @OneToMany(() => Post, p => p.author) posts: Post[];\n"
            "}"
        )
        signals = plugin._language_signals(source)
        orm = next(s for s in signals if s.key == "typescript.orm_persistence")
        assert "typeorm_entity" in orm.value
        assert "typeorm_column" in orm.value
        assert "typeorm_relation" in orm.value

    def test_typeorm_repository(self):
        plugin = TypeScriptLSPPlugin()
        source = "@InjectRepository(User) private repo: Repository<User>"
        signals = plugin._language_signals(source)
        orm = next(s for s in signals if s.key == "typescript.orm_persistence")
        assert "typeorm_repository" in orm.value

    def test_prisma_client(self):
        plugin = TypeScriptLSPPlugin()
        source = "const users = await prisma.user.findMany();"
        signals = plugin._language_signals(source)
        orm = next(s for s in signals if s.key == "typescript.orm_persistence")
        assert "prisma" in orm.value

    def test_mongoose(self):
        plugin = TypeScriptLSPPlugin()
        source = "const schema = new Schema({ name: String });"
        signals = plugin._language_signals(source)
        orm = next(s for s in signals if s.key == "typescript.orm_persistence")
        assert "mongoose" in orm.value

    def test_inject_and_nest_inject(self):
        plugin = TypeScriptLSPPlugin()
        source = "constructor(@Inject(TOKEN) private svc, @InjectRepository(User) private repo) {}"
        signals = plugin._language_signals(source)
        di = next(s for s in signals if s.key == "typescript.dependency_injection")
        assert "inject" in di.value
        assert "nest_inject" in di.value

    def test_inversify_and_provide(self):
        plugin = TypeScriptLSPPlugin()
        source = "@injectable()\nclass Foo {}\nconst provider = { useFactory: () => new Foo() };"
        signals = plugin._language_signals(source)
        di = next(s for s in signals if s.key == "typescript.dependency_injection")
        assert "inversify" in di.value
        assert "provide" in di.value

    def test_observable_subject_pipe_operators(self):
        plugin = TypeScriptLSPPlugin()
        source = (
            "const data$: Observable<string> = subject$;\n"
            "const s = new BehaviorSubject<number>(0);\n"
            "data$.pipe(switchMap(x => of(x)));"
        )
        signals = plugin._language_signals(source)
        rx = next(s for s in signals if s.key == "typescript.reactive_rxjs")
        assert "observable" in rx.value
        assert "subject" in rx.value
        assert "pipe" in rx.value
        assert "operators" in rx.value

    def test_express_router_pattern(self):
        plugin = TypeScriptLSPPlugin()
        source = "const router = Router();"
        signals = plugin._language_signals(source)
        rh = next(s for s in signals if s.key == "typescript.rest_http_patterns")
        assert "express_router" in rh.value
