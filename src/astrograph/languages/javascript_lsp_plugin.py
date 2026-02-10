"""JavaScript language adapter using the LSP plugin base."""

from __future__ import annotations

import logging
import re
import textwrap
from collections.abc import Iterator

import esprima
import networkx as nx

from ._configured_lsp_plugin import ConfiguredLSPLanguagePluginBase
from .base import CodeUnit, SemanticSignal, next_block_name

logger = logging.getLogger(__name__)

# -- Async patterns --
_JS_ASYNC_RE = re.compile(r"\basync\s+(?:function\b|[\w$]+\s*\(|\()" r"|\bawait\s" r"|\basync\s*\(")

# -- Type system patterns --
# TypeScript-style annotations: `param: Type`, `): ReturnType`
_TS_ANNOTATION_RE = re.compile(
    r"(?:[\w$]+)\s*:\s*(?:string|number|boolean|void|any|never|unknown|object"
    r"|Array|Promise|Record|Map|Set|null|undefined|[\w$]+(?:<[^>]+>)?)"
    r"(?:\s*[,);=\]\|}])"
)
_TS_INTERFACE_RE = re.compile(r"\b(?:interface|type)\s+[A-Z][\w$]*")
_JSDOC_RE = re.compile(r"@(?:type|param|returns?|typedef)\b")
_FLOW_PRAGMA_RE = re.compile(r"(?://|/\*)\s*@flow\b")

# -- Module system --
_ESM_RE = re.compile(r"\b(?:import|export)\s")
_CJS_RE = re.compile(r"\brequire\s*\(|module\.exports\b|exports\.\w")

# -- Class / prototype patterns --
_ES6_CLASS_RE = re.compile(r"\bclass\s+[A-Z][\w$]*")
_PROTOTYPE_RE = re.compile(r"\.prototype\s*[.=]")

# -- Decorators --
_DECORATOR_RE = re.compile(r"^\s*@([A-Za-z_$][A-Za-z0-9_$]*)", re.MULTILINE)

_JS_NUMERIC_TYPES = frozenset({"number", "bigint", "Number", "BigInt"})

# -- HTTP framework patterns --
_JS_EXPRESS_RE = re.compile(
    r"\bexpress\s*\(\)" r"|\bRouter\s*\(\)" r"|\b(?:app|router)\.(?:get|post|put|delete|patch)\s*\("
)
_JS_FASTIFY_RE = re.compile(
    r"\b[Ff]astify\s*\(" r"|\bfastify\.(?:get|post|register|decorate|addHook)\s*\("
)
_JS_KOA_RE = re.compile(r"\bnew\s+Koa\s*\(")
_JS_HAPI_RE = re.compile(r"\bHapi\.(?:server|Server)\s*\(")
_JS_NATIVE_HTTP_RE = re.compile(r"\bhttps?\.createServer\s*\(")

# -- Middleware patterns --
_JS_BODY_PARSER_RE = re.compile(
    r"\bbodyParser\.(?:json|urlencoded)\s*\(" r"|\bexpress\.(?:json|urlencoded)\s*\("
)
_JS_CORS_RE = re.compile(r"\bcors\s*\(")
_JS_HELMET_RE = re.compile(r"\bhelmet\s*\(")
_JS_MORGAN_RE = re.compile(r"\bmorgan\s*\(")
_JS_ERROR_HANDLER_RE = re.compile(r"\(\s*err\s*,\s*req\s*,\s*res\s*,\s*next\s*\)")
_JS_STATIC_FILES_RE = re.compile(r"\bexpress\.static\s*\(")

# -- Database client patterns --
_JS_MONGOOSE_SCHEMA_RE = re.compile(
    r"\bmongoose\.(?:Schema|model|connect)\b" r"|\bnew\s+Schema\s*\("
)
_JS_SEQUELIZE_RE = re.compile(
    r"\bnew\s+Sequelize\s*\(" r"|\bsequelize\.define\s*\(" r"|\bDataTypes\."
)
_JS_KNEX_RE = re.compile(r"\bknex\s*\(" r"|\bknex\.schema\." r"|\bknex\.migrate\.")
_JS_MONGODB_NATIVE_RE = re.compile(r"\bMongoClient\b")
_JS_REDIS_RE = re.compile(r"\bnew\s+Redis\s*\(" r"|\bredis\.createClient\s*\(")
_JS_PRISMA_RE = re.compile(
    r"\bPrismaClient\b" r"|prisma\.\w+\.(?:findMany|create|update|delete)\s*\("
)

# -- Auth patterns --
_JS_JWT_RE = re.compile(r"\bjwt\.(?:sign|verify|decode)\s*\(")
_JS_PASSPORT_RE = re.compile(r"\bpassport\.(?:authenticate|use|initialize|session)\s*\(")
_JS_BCRYPT_RE = re.compile(r"\bbcrypt\.(?:hash|compare|genSalt|hashSync|compareSync)\s*\(")
_JS_OAUTH_RE = re.compile(r"\bOAuth2(?:Client|Strategy)\b")
_JS_SESSION_AUTH_RE = re.compile(r"\breq\.session\b" r"|\breq\.isAuthenticated\s*\(")

# -- Realtime / messaging patterns --
_JS_SOCKETIO_RE = re.compile(
    r"\bsocket\.(?:join|leave)\s*\("
    r"|\bio\.(?:of|to|emit)\s*\("
    r"|\bio\.on\s*\(\s*['\"]connection['\"]"
)
_JS_WEBSOCKET_RE = re.compile(r"\bnew\s+WebSocket\s*\(" r"|\bWebSocketServer\b")
_JS_EVENT_EMITTER_RE = re.compile(r"\bEventEmitter\b")
_JS_MESSAGE_QUEUE_RE = re.compile(
    r"\bchannel\.(?:sendToQueue|consume|assertQueue)\s*\(" r"|\bamqp\.connect\s*\("
)

# -- Esprima AST child fields (order matters for graph structure) --
_ESPRIMA_CHILD_FIELDS = (
    "body",
    "declarations",
    "init",
    "test",
    "update",
    "consequent",
    "alternate",
    "left",
    "right",
    "argument",
    "arguments",
    "callee",
    "object",
    "property",
    "params",
    "expression",
    "elements",
    "properties",
    "value",
    "key",
    "discriminant",
    "cases",
    "handler",
    "block",
    "finalizer",
    "guardedHandlers",
    "param",
    "id",
    "superClass",
    "tag",
    "quasi",
    "quasis",
    "expressions",
)


_OPERATOR_EXPR_TYPES = frozenset(
    {
        "BinaryExpression",
        "UnaryExpression",
        "UpdateExpression",
        "AssignmentExpression",
        "LogicalExpression",
    }
)


def _esprima_node_label(node: object, normalize_ops: bool = False) -> str:
    """Compute a structural label for an esprima AST node."""
    node_type = getattr(node, "type", None)
    if node_type is None:
        return "Unknown"

    if node_type in _OPERATOR_EXPR_TYPES:
        op = getattr(node, "operator", "")
        return f"{node_type}:Op" if normalize_ops else f"{node_type}:{op}"
    _kind_defaults = {"VariableDeclaration": "var", "MethodDefinition": "method"}
    if node_type in _kind_defaults:
        kind = getattr(node, "kind", _kind_defaults[node_type])
        return f"{node_type}:{kind}"
    if node_type == "Literal":
        val = getattr(node, "value", None)
        return f"Literal:{type(val).__name__}"
    if node_type == "ArrowFunctionExpression":
        return "ArrowFunction"
    if node_type == "TemplateLiteral":
        return "TemplateLiteral"

    return str(node_type)


def _esprima_ast_to_graph(source: str, normalize_ops: bool = False) -> nx.DiGraph | None:
    """Parse JS source with esprima and build a structural directed graph.

    Returns None if esprima cannot parse the source.
    """
    tree = _esprima_try_parse(source)
    if tree is None:
        return None

    graph = nx.DiGraph()
    counter = 0

    def walk(node: object, parent_id: int | None = None) -> None:
        nonlocal counter
        if node is None:
            return
        # Only process objects that look like AST nodes (have a 'type' attribute)
        if not hasattr(node, "type"):
            return

        node_id = counter
        counter += 1
        label = _esprima_node_label(node, normalize_ops=normalize_ops)
        graph.add_node(node_id, label=label)
        if parent_id is not None:
            graph.add_edge(parent_id, node_id)

        for field in _ESPRIMA_CHILD_FIELDS:
            child = getattr(node, field, None)
            if child is None:
                continue
            if isinstance(child, list):
                for item in child:
                    walk(item, node_id)
            else:
                walk(child, node_id)

    walk(tree)
    return graph


# -- Esprima block types for sub-function extraction --
_ESPRIMA_BLOCK_TYPES = frozenset(
    {
        "ForStatement",
        "ForInStatement",
        "ForOfStatement",
        "WhileStatement",
        "DoWhileStatement",
        "IfStatement",
        "TryStatement",
        "SwitchStatement",
    }
)
_ESPRIMA_BLOCK_TYPE_NAMES: dict[str, str] = {
    "ForStatement": "for",
    "ForInStatement": "for_in",
    "ForOfStatement": "for_of",
    "WhileStatement": "while",
    "DoWhileStatement": "do_while",
    "IfStatement": "if",
    "TryStatement": "try",
    "SwitchStatement": "switch",
}
_ESPRIMA_FUNCTION_TYPES = frozenset(
    {
        "FunctionDeclaration",
        "FunctionExpression",
        "ArrowFunctionExpression",
    }
)


def _esprima_extract_blocks(
    node: object,
    source_lines: list[str],
    file_path: str,
    func_name: str,
    parent_block_name: str,
    current_depth: int,
    max_depth: int,
    block_counters: dict[str, int],
    language: str,
) -> Iterator[CodeUnit]:
    """Recursively extract blocks from an esprima AST node."""
    if current_depth > max_depth:
        return

    for field in _ESPRIMA_CHILD_FIELDS:
        child = getattr(node, field, None)
        if child is None:
            continue
        children = child if isinstance(child, list) else [child]
        for item in children:
            node_type = getattr(item, "type", None)
            if node_type is None:
                continue

            if node_type in _ESPRIMA_BLOCK_TYPES:
                block_type = _ESPRIMA_BLOCK_TYPE_NAMES[node_type]
                block_name = next_block_name(
                    block_type, func_name, parent_block_name, block_counters
                )

                loc = getattr(item, "loc", None)
                if loc is None:
                    continue
                start_line = getattr(loc.start, "line", 1)
                end_line = getattr(loc.end, "line", start_line)
                code = textwrap.dedent("\n".join(source_lines[start_line - 1 : end_line]))

                yield CodeUnit(
                    name=block_name,
                    code=code,
                    file_path=file_path,
                    line_start=start_line,
                    line_end=end_line,
                    unit_type="block",
                    parent_name=func_name,
                    block_type=block_type,
                    nesting_depth=current_depth,
                    parent_block_name=(
                        parent_block_name if parent_block_name != func_name else None
                    ),
                    language=language,
                )

                yield from _esprima_extract_blocks(
                    item,
                    source_lines,
                    file_path,
                    func_name,
                    block_name,
                    current_depth + 1,
                    max_depth,
                    block_counters,
                    language,
                )
            else:
                # Recurse into non-block nodes to find nested blocks
                yield from _esprima_extract_blocks(
                    item,
                    source_lines,
                    file_path,
                    func_name,
                    parent_block_name,
                    current_depth,
                    max_depth,
                    block_counters,
                    language,
                )


def _esprima_try_parse(source: str, *, loc: bool = False) -> object | None:
    """Try parseScript then parseModule, return AST or None."""
    for parser in (esprima.parseScript, esprima.parseModule):
        try:
            tree: object = parser(source, loc=loc) if loc else parser(source)
            return tree
        except esprima.Error:
            continue
    return None


def _esprima_extract_function_blocks(
    tree: object,
    source_lines: list[str],
    file_path: str,
    max_depth: int,
    language: str,
) -> Iterator[CodeUnit]:
    """Walk esprima AST to find functions and extract their inner blocks."""

    def walk(node: object) -> Iterator[CodeUnit]:
        if node is None or not hasattr(node, "type"):
            return
        node_type = getattr(node, "type", None)

        if node_type in _ESPRIMA_FUNCTION_TYPES:
            # Determine function name
            node_id = getattr(node, "id", None)
            func_name = getattr(node_id, "name", None) if node_id else None
            if func_name is None:
                func_name = "<anonymous>"

            body = getattr(node, "body", None)
            if body is not None:
                block_counters: dict[str, int] = {}
                yield from _esprima_extract_blocks(
                    body,
                    source_lines,
                    file_path,
                    func_name,
                    func_name,
                    1,
                    max_depth,
                    block_counters,
                    language,
                )

        if node_type == "MethodDefinition":
            key = getattr(node, "key", None)
            func_name = getattr(key, "name", None) if key else None
            if func_name is None:
                func_name = "<anonymous>"
            value = getattr(node, "value", None)
            if value is not None:
                body = getattr(value, "body", None)
                if body is not None:
                    block_counters = {}
                    yield from _esprima_extract_blocks(
                        body,
                        source_lines,
                        file_path,
                        func_name,
                        func_name,
                        1,
                        max_depth,
                        block_counters,
                        language,
                    )

        # Continue walking into children (but don't re-enter function bodies)
        for field_name in _ESPRIMA_CHILD_FIELDS:
            child = getattr(node, field_name, None)
            if child is None:
                continue
            if isinstance(child, list):
                for item in child:
                    yield from walk(item)
            else:
                yield from walk(child)

    yield from walk(tree)


class JavaScriptLSPPlugin(ConfiguredLSPLanguagePluginBase):
    """JavaScript support via LSP symbols + structural graphing."""

    LANGUAGE_ID = "javascript_lsp"
    LSP_LANGUAGE_ID = "javascript"
    FILE_EXTENSIONS = frozenset({".js", ".jsx", ".mjs", ".cjs"})
    SKIP_DIRS = frozenset({"node_modules", ".next", ".nuxt", "coverage"})
    DEFAULT_COMMAND = ("typescript-language-server", "--stdio")
    COMMAND_ENV_VAR = "ASTROGRAPH_JS_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_JS_LSP_TIMEOUT"

    # ------------------------------------------------------------------
    # AST graph builder (esprima)
    # ------------------------------------------------------------------

    def source_to_graph(
        self,
        source: str,
        normalize_ops: bool = False,
    ) -> nx.DiGraph:
        """Build a structural graph from JS source using esprima.

        Falls back to the base line-level parser if esprima cannot parse.
        """
        graph = _esprima_ast_to_graph(source, normalize_ops=normalize_ops)
        if graph is not None:
            return graph
        logger.debug("esprima parse failed, falling back to line-level parser")
        return super().source_to_graph(source, normalize_ops=normalize_ops)

    def normalize_graph_for_pattern(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Normalize operators for pattern matching (collapse op variants)."""
        normalized: nx.DiGraph = graph.copy()
        for _node_id, data in normalized.nodes(data=True):
            label = data.get("label", "")
            for prefix in (
                "BinaryExpression:",
                "UnaryExpression:",
                "UpdateExpression:",
                "AssignmentExpression:",
                "LogicalExpression:",
            ):
                if label.startswith(prefix):
                    data["label"] = f"{prefix.rstrip(':')}:Op"
                    break
        return normalized

    # ------------------------------------------------------------------
    # Block extraction (esprima)
    # ------------------------------------------------------------------

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract units via LSP, then optionally extract inner blocks via esprima."""
        yield from super().extract_code_units(
            source,
            file_path,
            include_blocks=False,
            max_block_depth=max_block_depth,
        )

        if not include_blocks:
            return

        tree = _esprima_try_parse(source, loc=True)
        if tree is None:
            return

        source_lines = source.splitlines()
        yield from _esprima_extract_function_blocks(
            tree,
            source_lines,
            file_path,
            max_depth=max_block_depth,
            language=self.LANGUAGE_ID,
        )

    # ------------------------------------------------------------------
    # Semantic profiling helpers (regex-based)
    # ------------------------------------------------------------------

    def _detect_async(self, source: str) -> bool:
        """Check for async/await constructs."""
        return bool(_JS_ASYNC_RE.search(source))

    def _detect_type_system(self, source: str) -> str:
        """Detect type system: typescript, jsdoc, flow, or none."""
        if _FLOW_PRAGMA_RE.search(source):
            return "flow"
        has_ts = bool(_TS_ANNOTATION_RE.search(source) or _TS_INTERFACE_RE.search(source))
        has_jsdoc = bool(_JSDOC_RE.search(source))
        if has_ts and has_jsdoc:
            return "typescript"
        if has_ts:
            return "typescript"
        if has_jsdoc:
            return "jsdoc"
        return "none"

    # -- AST-based plus binding helpers --

    @staticmethod
    def _esprima_walk(node: object) -> Iterator[object]:
        """Yield all AST nodes reachable from *node* (depth-first)."""
        if node is None or not hasattr(node, "type"):
            return
        yield node
        for field in _ESPRIMA_CHILD_FIELDS:
            child = getattr(node, field, None)
            if child is None:
                continue
            if isinstance(child, list):
                for item in child:
                    yield from JavaScriptLSPPlugin._esprima_walk(item)
            else:
                yield from JavaScriptLSPPlugin._esprima_walk(child)

    @staticmethod
    def _esprima_annotation_name(annotation: object | None) -> str | None:
        """Extract a simple type name from an esprima type annotation."""
        if annotation is None:
            return None
        ann_type = getattr(annotation, "type", None)
        # TSTypeAnnotation wraps the actual type node
        if ann_type == "TSTypeAnnotation":
            return JavaScriptLSPPlugin._esprima_annotation_name(
                getattr(annotation, "typeAnnotation", None)
            )
        if ann_type == "TSTypeReference":
            type_name_node = getattr(annotation, "typeName", None)
            return getattr(type_name_node, "name", None)
        if ann_type in (
            "TSStringKeyword",
            "TSNumberKeyword",
            "TSBooleanKeyword",
            "TSBigIntKeyword",
            "TSAnyKeyword",
        ):
            # e.g. "TSNumberKeyword" → "number"
            return str(ann_type[2:-7]).lower()  # strip TS...Keyword
        return None

    @staticmethod
    def _esprima_build_annotation_map(tree: object) -> dict[str, str]:
        """Build variable→type mapping from function parameter annotations."""
        ann_map: dict[str, str] = {}
        for node in JavaScriptLSPPlugin._esprima_walk(tree):
            node_type = getattr(node, "type", None)
            if node_type not in _ESPRIMA_FUNCTION_TYPES:
                continue
            params = getattr(node, "params", None)
            if not params:
                continue
            for param in params:
                param_type = getattr(param, "type", None)
                # Plain identifier with optional annotation: `x: number`
                if param_type == "Identifier":
                    name = getattr(param, "name", None)
                    ann = getattr(param, "typeAnnotation", None)
                    type_name = JavaScriptLSPPlugin._esprima_annotation_name(ann)
                    if name and type_name:
                        ann_map[name] = type_name
                # Assignment pattern (default value): `x: number = 0`
                elif param_type == "AssignmentPattern":
                    left = getattr(param, "left", None)
                    if left and getattr(left, "type", None) == "Identifier":
                        name = getattr(left, "name", None)
                        ann = getattr(left, "typeAnnotation", None)
                        type_name = JavaScriptLSPPlugin._esprima_annotation_name(ann)
                        if name and type_name:
                            ann_map[name] = type_name
        return ann_map

    @staticmethod
    def _resolve_js_operand_type(
        node: object,
        annotation_map: dict[str, str],
    ) -> str | None:
        """Resolve an operand to a type string via annotation lookup or literal."""
        node_type = getattr(node, "type", None)
        if node_type == "Identifier":
            return annotation_map.get(getattr(node, "name", ""))
        if node_type == "Literal":
            val = getattr(node, "value", None)
            if isinstance(val, str):
                return "string"
            if isinstance(val, int | float):
                return "number"
        if node_type == "TemplateLiteral":
            return "string"
        return None

    def _infer_plus_binding(self, source: str) -> tuple[str, float] | None:
        """Find BinaryExpression(+) via esprima AST, resolve operand types."""
        tree = _esprima_try_parse(source)
        if tree is None:
            return None

        annotation_map = self._esprima_build_annotation_map(tree)
        found_plus = False
        saw_numeric = False
        saw_str = False
        saw_unknown = False

        for node in self._esprima_walk(tree):
            if getattr(node, "type", None) != "BinaryExpression":
                continue
            if getattr(node, "operator", None) != "+":
                continue
            found_plus = True
            left_type = self._resolve_js_operand_type(getattr(node, "left", None), annotation_map)
            right_type = self._resolve_js_operand_type(getattr(node, "right", None), annotation_map)
            if left_type is None or right_type is None:
                saw_unknown = True
                continue
            if left_type in _JS_NUMERIC_TYPES and right_type in _JS_NUMERIC_TYPES:
                saw_numeric = True
            elif left_type == "string" and right_type == "string":
                saw_str = True

        if not found_plus:
            return None
        if saw_str and saw_numeric:
            return "mixed", 0.6
        if saw_str:
            return "str_concat", 0.9
        if saw_numeric:
            return "numeric", 0.9
        if saw_unknown:
            return "unknown", 0.5
        return "unknown", 0.5

    def _detect_module_system(self, source: str) -> str:
        """Detect module system: esm, commonjs, mixed, or none."""
        has_esm = bool(_ESM_RE.search(source))
        has_cjs = bool(_CJS_RE.search(source))
        if has_esm and has_cjs:
            return "mixed"
        if has_esm:
            return "esm"
        if has_cjs:
            return "commonjs"
        return "none"

    def _detect_class_pattern(self, source: str) -> str | None:
        """Detect class style: class (ES6) or prototype."""
        has_class = bool(_ES6_CLASS_RE.search(source))
        has_proto = bool(_PROTOTYPE_RE.search(source))
        if has_class and has_proto:
            return "class"
        if has_class:
            return "class"
        if has_proto:
            return "prototype"
        return None

    def _collect_decorators(self, source: str) -> set[str]:
        """Extract decorator names from @decorator patterns."""
        return {m.group(1) for m in _DECORATOR_RE.finditer(source)}

    def _language_signals(self, source: str) -> list[SemanticSignal]:
        """JavaScript-specific signals from regex analysis."""
        signals = super()._language_signals(source)

        # 1. Async (always emitted)
        has_async = self._detect_async(source)
        signals.append(
            SemanticSignal(
                key="javascript.async.present",
                value="yes" if has_async else "no",
                confidence=0.95,
                origin="syntax",
            )
        )

        # 2. Type system (always emitted)
        type_system = self._detect_type_system(source)
        signals.append(
            SemanticSignal(
                key="javascript.type_system",
                value=type_system,
                confidence=0.85,
                origin="syntax",
            )
        )

        # 3. Plus binding
        plus_result = self._infer_plus_binding(source)
        if plus_result is not None:
            binding, confidence = plus_result
            signals.append(
                SemanticSignal(
                    key="javascript.plus_binding",
                    value=binding,
                    confidence=confidence,
                    origin="syntax",
                )
            )

        # 4. Module system (always emitted)
        module_system = self._detect_module_system(source)
        signals.append(
            SemanticSignal(
                key="javascript.module_system",
                value=module_system,
                confidence=0.95,
                origin="syntax",
            )
        )

        # 5. Class pattern
        class_pattern = self._detect_class_pattern(source)
        if class_pattern is not None:
            signals.append(
                SemanticSignal(
                    key="javascript.class_pattern",
                    value=class_pattern,
                    confidence=0.9,
                    origin="syntax",
                )
            )

        # 6. Decorators
        decorators = self._collect_decorators(source)
        if decorators:
            signals.append(
                SemanticSignal(
                    key="javascript.decorators.present",
                    value=",".join(sorted(decorators)),
                    confidence=0.95,
                    origin="syntax",
                )
            )

        # 7. HTTP framework
        http_parts: list[str] = []
        if _JS_EXPRESS_RE.search(source):
            http_parts.append("express")
        if _JS_FASTIFY_RE.search(source):
            http_parts.append("fastify")
        if _JS_KOA_RE.search(source):
            http_parts.append("koa")
        if _JS_HAPI_RE.search(source):
            http_parts.append("hapi")
        if _JS_NATIVE_HTTP_RE.search(source):
            http_parts.append("native_http")
        signals.append(
            SemanticSignal(
                key="javascript.http_framework",
                value=",".join(http_parts) if http_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 8. Middleware patterns
        mw_parts: list[str] = []
        if _JS_BODY_PARSER_RE.search(source):
            mw_parts.append("body_parser")
        if _JS_CORS_RE.search(source):
            mw_parts.append("cors")
        if _JS_HELMET_RE.search(source):
            mw_parts.append("helmet")
        if _JS_MORGAN_RE.search(source):
            mw_parts.append("morgan")
        if _JS_ERROR_HANDLER_RE.search(source):
            mw_parts.append("error_handler")
        if _JS_STATIC_FILES_RE.search(source):
            mw_parts.append("static_files")
        signals.append(
            SemanticSignal(
                key="javascript.middleware_patterns",
                value=",".join(mw_parts) if mw_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 9. Database client
        db_parts: list[str] = []
        if _JS_MONGOOSE_SCHEMA_RE.search(source):
            db_parts.append("mongoose")
        if _JS_SEQUELIZE_RE.search(source):
            db_parts.append("sequelize")
        if _JS_KNEX_RE.search(source):
            db_parts.append("knex")
        if _JS_MONGODB_NATIVE_RE.search(source):
            db_parts.append("mongodb_native")
        if _JS_REDIS_RE.search(source):
            db_parts.append("redis")
        if _JS_PRISMA_RE.search(source):
            db_parts.append("prisma")
        signals.append(
            SemanticSignal(
                key="javascript.database_client",
                value=",".join(db_parts) if db_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 10. Auth patterns
        auth_parts: list[str] = []
        if _JS_JWT_RE.search(source):
            auth_parts.append("jwt")
        if _JS_PASSPORT_RE.search(source):
            auth_parts.append("passport")
        if _JS_BCRYPT_RE.search(source):
            auth_parts.append("bcrypt")
        if _JS_OAUTH_RE.search(source):
            auth_parts.append("oauth")
        if _JS_SESSION_AUTH_RE.search(source):
            auth_parts.append("session_auth")
        signals.append(
            SemanticSignal(
                key="javascript.auth_patterns",
                value=",".join(auth_parts) if auth_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 11. Realtime / messaging
        rt_parts: list[str] = []
        if _JS_SOCKETIO_RE.search(source):
            rt_parts.append("socketio")
        if _JS_WEBSOCKET_RE.search(source):
            rt_parts.append("websocket")
        if _JS_EVENT_EMITTER_RE.search(source):
            rt_parts.append("event_emitter")
        if _JS_MESSAGE_QUEUE_RE.search(source):
            rt_parts.append("message_queue")
        signals.append(
            SemanticSignal(
                key="javascript.realtime_messaging",
                value=",".join(rt_parts) if rt_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        return signals
