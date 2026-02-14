"""Tests for JS and Python plugin semantic signal extraction.

Targets coverage gaps in:
- javascript_lsp_plugin.py lines 533-660, 748-842
- python_lsp_plugin.py lines 145-304
"""

from __future__ import annotations

import pytest

from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin
from astrograph.languages.python_lsp_plugin import PythonLSPPlugin


class TestJavaScriptAnnotationName:
    """Cover _esprima_annotation_name edge cases (lines 533-551)."""

    def test_none_annotation(self):
        assert JavaScriptLSPPlugin._esprima_annotation_name(None) is None

    def test_ts_type_reference(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="TSTypeReference", typeName=SimpleNamespace(name="MyType"))
        result = JavaScriptLSPPlugin._esprima_annotation_name(node)
        assert result == "MyType"

    def test_ts_string_keyword(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="TSStringKeyword")
        result = JavaScriptLSPPlugin._esprima_annotation_name(node)
        assert result == "string"

    def test_ts_number_keyword(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="TSNumberKeyword")
        result = JavaScriptLSPPlugin._esprima_annotation_name(node)
        assert result == "number"

    def test_ts_boolean_keyword(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="TSBooleanKeyword")
        result = JavaScriptLSPPlugin._esprima_annotation_name(node)
        assert result == "boolean"

    def test_ts_type_annotation_wrapper(self):
        from types import SimpleNamespace

        inner = SimpleNamespace(type="TSNumberKeyword")
        wrapper = SimpleNamespace(type="TSTypeAnnotation", typeAnnotation=inner)
        result = JavaScriptLSPPlugin._esprima_annotation_name(wrapper)
        assert result == "number"

    def test_unknown_type(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="SomeOtherType")
        result = JavaScriptLSPPlugin._esprima_annotation_name(node)
        assert result is None


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
        source = 'const express = require("express");\nconst app = express();\napp.get("/", handler);'
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
        source = 'app.use(bodyParser.json());'
        signals = plugin._language_signals(source)
        mw = next(s for s in signals if s.key == "javascript.middleware_patterns")
        assert "body_parser" in mw.value

    def test_no_frameworks(self):
        plugin = JavaScriptLSPPlugin()
        source = "function add(a, b) { return a + b; }"
        signals = plugin._language_signals(source)
        http = next(s for s in signals if s.key == "javascript.http_framework")
        assert http.value == "none"


class TestJavaScriptResolveOperandType:
    """Cover _resolve_js_operand_type (lines 584-601)."""

    def test_identifier_with_annotation(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="Identifier", name="x")
        result = JavaScriptLSPPlugin._resolve_js_operand_type(node, {"x": "number"})
        assert result == "number"

    def test_literal_string(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="Literal", value="hello")
        result = JavaScriptLSPPlugin._resolve_js_operand_type(node, {})
        assert result == "string"

    def test_literal_number(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="Literal", value=42)
        result = JavaScriptLSPPlugin._resolve_js_operand_type(node, {})
        assert result == "number"

    def test_template_literal(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="TemplateLiteral")
        result = JavaScriptLSPPlugin._resolve_js_operand_type(node, {})
        assert result == "string"

    def test_unknown_type(self):
        from types import SimpleNamespace

        node = SimpleNamespace(type="CallExpression")
        result = JavaScriptLSPPlugin._resolve_js_operand_type(node, {})
        assert result is None


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
        async_signal = next(
            (s for s in profile.signals if s.key == "python.async.present"), None
        )
        assert async_signal is not None
        assert async_signal.value == "yes"

    def test_profile_with_class_style(self):
        plugin = PythonLSPPlugin()
        source = "@dataclass\nclass Config:\n    host: str = 'localhost'\n    port: int = 8080"
        profile = plugin.extract_semantic_profile(source)
        style_signal = next(
            (s for s in profile.signals if s.key == "python.class_style"), None
        )
        assert style_signal is not None
        assert style_signal.value == "dataclass"

    def test_profile_with_plus_binding(self):
        plugin = PythonLSPPlugin()
        source = "def add(x: int, y: int):\n    return x + y"
        profile = plugin.extract_semantic_profile(source)
        plus_signal = next(
            (s for s in profile.signals if s.key == "python.plus_binding"), None
        )
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
