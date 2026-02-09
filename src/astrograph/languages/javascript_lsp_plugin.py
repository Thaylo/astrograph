"""JavaScript language adapter using the LSP plugin base."""

from __future__ import annotations

import logging
import re
import textwrap
from collections.abc import Iterator

import esprima
import networkx as nx

from ._configured_lsp_plugin import ConfiguredLSPLanguagePluginBase
from .base import CodeUnit, SemanticSignal

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

# -- Plus expression binding --
_JS_PLUS_EXPR_RE = re.compile(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*\+\s*([A-Za-z_$][A-Za-z0-9_$]*)\b")
_JS_PARAM_ANNOTATION_RE = re.compile(
    r"\b([A-Za-z_$][A-Za-z0-9_$]*)\s*:\s*" r"(string|number|boolean|bigint|any|[\w$]+)\b"
)
_JS_STRING_LITERAL_PLUS_RE = re.compile(
    r"""(?:["'`][^"'`]*["'`])\s*\+|""" r"""\+\s*(?:["'`][^"'`]*["'`])"""
)
_JS_NUMBER_LITERAL_PLUS_RE = re.compile(r"\b\d+(?:\.\d+)?\s*\+|\+\s*\d+(?:\.\d+)?\b")

# -- Module system --
_ESM_RE = re.compile(r"\b(?:import|export)\s")
_CJS_RE = re.compile(r"\brequire\s*\(|module\.exports\b|exports\.\w")

# -- Class / prototype patterns --
_ES6_CLASS_RE = re.compile(r"\bclass\s+[A-Z][\w$]*")
_PROTOTYPE_RE = re.compile(r"\.prototype\s*[.=]")

# -- Decorators --
_DECORATOR_RE = re.compile(r"^\s*@([A-Za-z_$][A-Za-z0-9_$]*)", re.MULTILINE)

_JS_NUMERIC_TYPES = frozenset({"number", "bigint", "Number", "BigInt"})

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


def _esprima_node_label(node: object, normalize_ops: bool = False) -> str:
    """Compute a structural label for an esprima AST node."""
    node_type = getattr(node, "type", None)
    if node_type is None:
        return "Unknown"

    if node_type == "BinaryExpression":
        op = getattr(node, "operator", "")
        if normalize_ops:
            return "BinaryExpression:Op"
        return f"BinaryExpression:{op}"
    if node_type == "UnaryExpression":
        op = getattr(node, "operator", "")
        if normalize_ops:
            return "UnaryExpression:Op"
        return f"UnaryExpression:{op}"
    if node_type == "UpdateExpression":
        op = getattr(node, "operator", "")
        if normalize_ops:
            return "UpdateExpression:Op"
        return f"UpdateExpression:{op}"
    if node_type == "AssignmentExpression":
        op = getattr(node, "operator", "")
        if normalize_ops:
            return "AssignmentExpression:Op"
        return f"AssignmentExpression:{op}"
    if node_type == "LogicalExpression":
        op = getattr(node, "operator", "")
        if normalize_ops:
            return "LogicalExpression:Op"
        return f"LogicalExpression:{op}"
    if node_type == "VariableDeclaration":
        kind = getattr(node, "kind", "var")
        return f"VariableDeclaration:{kind}"
    if node_type == "Literal":
        val = getattr(node, "value", None)
        return f"Literal:{type(val).__name__}"
    if node_type == "MethodDefinition":
        kind = getattr(node, "kind", "method")
        return f"MethodDefinition:{kind}"
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
                counter_key = f"{parent_block_name}.{block_type}"
                if counter_key not in block_counters:
                    block_counters[counter_key] = 0
                block_counters[counter_key] += 1
                block_num = block_counters[counter_key]

                if parent_block_name == func_name:
                    block_name = f"{func_name}.{block_type}_{block_num}"
                else:
                    block_name = f"{parent_block_name}.{block_type}_{block_num}"

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

    def _build_annotation_map(self, source: str) -> dict[str, str]:
        """Build variable→type mapping from TS-style annotations."""
        ann_map: dict[str, str] = {}
        for match in _JS_PARAM_ANNOTATION_RE.finditer(source):
            ann_map[match.group(1)] = match.group(2)
        return ann_map

    def _infer_plus_binding(self, source: str) -> tuple[str, float] | None:
        """Resolve + operand types from annotations and literals."""
        has_var_plus = bool(_JS_PLUS_EXPR_RE.search(source))
        has_str_literal_plus = bool(_JS_STRING_LITERAL_PLUS_RE.search(source))
        has_num_literal_plus = bool(_JS_NUMBER_LITERAL_PLUS_RE.search(source))

        if not has_var_plus and not has_str_literal_plus and not has_num_literal_plus:
            return None

        if has_str_literal_plus and not has_num_literal_plus and not has_var_plus:
            return "str_concat", 0.9
        if has_num_literal_plus and not has_str_literal_plus and not has_var_plus:
            return "numeric", 0.9

        if not has_var_plus:
            return "mixed", 0.6

        ann_map = self._build_annotation_map(source)
        saw_numeric = has_num_literal_plus
        saw_str = has_str_literal_plus

        for match in _JS_PLUS_EXPR_RE.finditer(source):
            left_type = ann_map.get(match.group(1))
            right_type = ann_map.get(match.group(2))
            if left_type is None or right_type is None:
                continue
            if left_type in _JS_NUMERIC_TYPES and right_type in _JS_NUMERIC_TYPES:
                saw_numeric = True
            elif left_type == "string" and right_type == "string":
                saw_str = True

        if saw_str and saw_numeric:
            return "mixed", 0.6
        if saw_str:
            return "str_concat", 0.85
        if saw_numeric:
            return "numeric", 0.85
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

        return signals
