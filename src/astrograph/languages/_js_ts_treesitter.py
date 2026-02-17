"""Shared tree-sitter parsing for JavaScript and TypeScript plugins.

Replaces all esprima-based AST functions with tree-sitter equivalents.
Used by JavaScriptLSPPlugin and TypeScriptLSPPlugin — they keep their
LSP class hierarchy and call these module-level functions instead.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

import networkx as nx
import tree_sitter

from .base import CodeUnit, next_block_name

# -- Language / parser caching -------------------------------------------------


@lru_cache(maxsize=1)
def _get_js_language() -> tree_sitter.Language:
    import tree_sitter_javascript

    return tree_sitter.Language(tree_sitter_javascript.language())


@lru_cache(maxsize=1)
def _get_ts_language() -> tree_sitter.Language:
    import tree_sitter_typescript

    return tree_sitter.Language(tree_sitter_typescript.language_typescript())


@lru_cache(maxsize=2)
def _get_parser(language: str) -> tree_sitter.Parser:
    """Return a cached parser for 'javascript' or 'typescript'."""
    lang_obj = _get_ts_language() if language == "typescript" else _get_js_language()
    return tree_sitter.Parser(lang_obj)


# -- Node label mapping --------------------------------------------------------

# Arithmetic / comparison operators → BinaryExpression
_BINARY_OPS = frozenset(
    {
        "+",
        "-",
        "*",
        "/",
        "%",
        "**",
        "&",
        "|",
        "^",
        "<<",
        ">>",
        ">>>",
        "==",
        "!=",
        "===",
        "!==",
        "<",
        ">",
        "<=",
        ">=",
    }
)
# Logical / nullish operators → LogicalExpression
_LOGICAL_OPS = frozenset({"&&", "||", "??"})

# TS type-only node types to skip in graph building
_TS_TYPE_NODE_TYPES = frozenset(
    {
        "type_annotation",
        "type_identifier",
        "predefined_type",
        "generic_type",
        "type_parameters",
        "type_parameter",
        "type_arguments",
        "union_type",
        "intersection_type",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "required_parameter",  # skip wrapper, children handled separately
    }
)

# Block types for sub-function extraction
_BLOCK_NODE_TYPES = frozenset(
    {
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "if_statement",
        "try_statement",
        "switch_statement",
    }
)

_BLOCK_TYPE_NAMES: dict[str, str] = {
    "for_statement": "for",
    "for_in_statement": "for_in",
    "while_statement": "while",
    "do_statement": "do_while",
    "if_statement": "if",
    "try_statement": "try",
    "switch_statement": "switch",
}

_FUNCTION_NODE_TYPES = frozenset(
    {
        "function_declaration",
        "function",
        "function_expression",
        "arrow_function",
        "generator_function_declaration",
        "generator_function",
    }
)

_JS_NUMERIC_TYPES = frozenset({"number", "bigint", "Number", "BigInt"})


def _label_with_operator(node: Any, label: str, normalize_ops: bool) -> str:
    """Build a structural label that includes operator text (or normalized Op)."""
    op = _get_operator(node)
    return f"{label}:Op" if normalize_ops else f"{label}:{op}"


def _ts_node_label(node: Any, normalize_ops: bool = False) -> str:
    """Map a tree-sitter node type to an esprima-compatible structural label."""
    ntype = node.type

    # Function nodes
    if ntype == "function_declaration" or ntype == "generator_function_declaration":
        return "FunctionDeclaration"
    if ntype in ("function", "function_expression", "generator_function"):
        return "FunctionExpression"
    if ntype == "arrow_function":
        return "ArrowFunction"

    # Binary / logical expressions
    if ntype in ("binary_expression", "augmented_assignment_expression"):
        op = _get_operator(node)
        if op in _LOGICAL_OPS:
            return "LogicalExpression:Op" if normalize_ops else f"LogicalExpression:{op}"
        return "BinaryExpression:Op" if normalize_ops else f"BinaryExpression:{op}"

    # Unary / update
    if ntype == "unary_expression":
        return _label_with_operator(node, "UnaryExpression", normalize_ops)
    if ntype == "update_expression":
        return _label_with_operator(node, "UpdateExpression", normalize_ops)

    # Assignment
    if ntype == "assignment_expression":
        return _label_with_operator(node, "AssignmentExpression", normalize_ops)

    # Variable declarations
    if ntype == "variable_declaration":
        return "VariableDeclaration:var"
    if ntype == "lexical_declaration":
        # Determine const vs let from first child
        for child in node.children:
            if child.type in ("const", "let"):
                return f"VariableDeclaration:{child.type}"
        return "VariableDeclaration:const"

    # Literals
    if ntype == "string" or ntype == "string_fragment":
        return "Literal:str"
    if ntype == "number":
        return "Literal:int"
    if ntype == "true" or ntype == "false":
        return "Literal:bool"
    if ntype == "null":
        return "Literal:NoneType"
    if ntype == "template_string":
        return "TemplateLiteral"

    # Method definition
    if ntype == "method_definition":
        kind = "method"
        for child in node.children:
            if child.type in ("get", "set"):
                kind = child.type
                break
        return f"MethodDefinition:{kind}"

    # Statement mappings
    _STATEMENT_MAP = {
        "statement_block": "BlockStatement",
        "for_statement": "ForStatement",
        "for_in_statement": "ForInStatement",
        "while_statement": "WhileStatement",
        "do_statement": "DoWhileStatement",
        "if_statement": "IfStatement",
        "try_statement": "TryStatement",
        "switch_statement": "SwitchStatement",
        "return_statement": "ReturnStatement",
        "throw_statement": "ThrowStatement",
        "break_statement": "BreakStatement",
        "continue_statement": "ContinueStatement",
        "expression_statement": "ExpressionStatement",
        "empty_statement": "EmptyStatement",
        "labeled_statement": "LabeledStatement",
        "with_statement": "WithStatement",
        "debugger_statement": "DebuggerStatement",
    }
    if ntype in _STATEMENT_MAP:
        return _STATEMENT_MAP[ntype]

    # Other common nodes
    _COMMON_MAP = {
        "program": "Program",
        "identifier": "Identifier",
        "property_identifier": "Identifier",
        "shorthand_property_identifier": "Identifier",
        "shorthand_property_identifier_pattern": "Identifier",
        "call_expression": "CallExpression",
        "member_expression": "MemberExpression",
        "subscript_expression": "MemberExpression",
        "new_expression": "NewExpression",
        "this": "ThisExpression",
        "array": "ArrayExpression",
        "object": "ObjectExpression",
        "pair": "Property",
        "spread_element": "SpreadElement",
        "rest_pattern": "RestElement",
        "conditional_expression": "ConditionalExpression",
        "sequence_expression": "SequenceExpression",
        "yield_expression": "YieldExpression",
        "await_expression": "AwaitExpression",
        "class_declaration": "ClassDeclaration",
        "class": "ClassExpression",
        "class_body": "ClassBody",
        "import_statement": "ImportDeclaration",
        "export_statement": "ExportDeclaration",
        "variable_declarator": "VariableDeclarator",
        "parenthesized_expression": "ParenthesizedExpression",
        "regex": "Literal:regex",
        "catch_clause": "CatchClause",
        "switch_case": "SwitchCase",
        "switch_default": "SwitchCase",
        "formal_parameters": "FormalParameters",
    }
    if ntype in _COMMON_MAP:
        return _COMMON_MAP[ntype]

    return str(ntype)


def _get_operator(node: Any) -> str:
    """Extract operator string from a binary/unary/update/assignment node."""
    for child in node.children:
        if child.is_named:
            continue
        text = child.text
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        # Operators are typically short punctuation
        if len(text) <= 4 and not text.isalnum():
            return str(text)
    return ""


def _ts_should_skip_node(node: Any) -> bool:
    """Return True for nodes that should be skipped in graph building."""
    # Skip unnamed nodes (punctuation, keywords as syntax)
    if not node.is_named:
        return True
    # Skip TS type-only nodes
    return node.type in _TS_TYPE_NODE_TYPES


# -- Graph building ------------------------------------------------------------


def _ts_ast_to_graph(
    source: str,
    normalize_ops: bool = False,
    language: str = "javascript",
) -> nx.DiGraph | None:
    """Parse source with tree-sitter and build a structural directed graph.

    Returns None if the parser produces an ERROR root node.
    """
    tree = _ts_try_parse(source, language)
    if tree is None:
        return None

    graph: nx.DiGraph = nx.DiGraph()
    counter = 0

    def walk(ts_node: Any, parent_id: int | None = None) -> None:
        nonlocal counter

        # For as_expression (TS cast), skip the cast wrapper but keep the inner expression
        if ts_node.type == "as_expression":
            # First named child is the expression, rest is the type
            for child in ts_node.children:
                if child.is_named and child.type not in _TS_TYPE_NODE_TYPES:
                    walk(child, parent_id)
                    return
            return

        if _ts_should_skip_node(ts_node):
            # Still recurse into children (e.g. required_parameter wraps useful nodes)
            for child in ts_node.children:
                walk(child, parent_id)
            return

        node_id = counter
        counter += 1
        label = _ts_node_label(ts_node, normalize_ops=normalize_ops)
        graph.add_node(node_id, label=label)
        if parent_id is not None:
            graph.add_edge(parent_id, node_id)

        for child in ts_node.children:
            walk(child, node_id)

    walk(tree.root_node)
    return graph


# -- Parsing -------------------------------------------------------------------


def _ts_try_parse(source: str, language: str = "javascript") -> Any | None:
    """Parse source and return the tree, or None on total failure."""
    parser = _get_parser(language)
    tree = parser.parse(source.encode("utf-8"))
    if tree is None:
        return None
    # If the entire root is an ERROR node, treat as parse failure
    if tree.root_node.type == "ERROR":
        return None
    return tree


# -- Block extraction ----------------------------------------------------------


def _ts_extract_function_blocks(
    tree: Any,
    source_lines: list[str],
    file_path: str,
    max_depth: int,
    language: str,
) -> Iterator[CodeUnit]:
    """Walk tree-sitter tree to find functions and extract their inner blocks."""

    def walk(node: Any) -> Iterator[CodeUnit]:
        ntype = node.type

        if ntype in _FUNCTION_NODE_TYPES:
            func_name = _get_function_name(node)
            body = _get_child_by_field(node, "body")
            if body is not None:
                block_counters: dict[str, int] = {}
                yield from _ts_extract_blocks(
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

        if ntype == "method_definition":
            func_name = _get_method_name(node)
            body = _get_child_by_field(node, "body")
            if body is not None:
                block_counters = {}
                yield from _ts_extract_blocks(
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

        for child in node.children:
            yield from walk(child)

    yield from walk(tree.root_node)


def _ts_extract_blocks(
    node: Any,
    source_lines: list[str],
    file_path: str,
    func_name: str,
    parent_block_name: str,
    current_depth: int,
    max_depth: int,
    block_counters: dict[str, int],
    language: str,
) -> Iterator[CodeUnit]:
    """Recursively extract blocks from a tree-sitter node."""
    if current_depth > max_depth:
        return

    for child in node.children:
        ntype = child.type

        if ntype in _BLOCK_NODE_TYPES:
            block_type = _BLOCK_TYPE_NAMES[ntype]
            block_name = next_block_name(block_type, func_name, parent_block_name, block_counters)

            start_line = child.start_point[0] + 1  # 1-based
            end_line = child.end_point[0] + 1
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
                parent_block_name=(parent_block_name if parent_block_name != func_name else None),
                language=language,
            )

            yield from _ts_extract_blocks(
                child,
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
            yield from _ts_extract_blocks(
                child,
                source_lines,
                file_path,
                func_name,
                parent_block_name,
                current_depth,
                max_depth,
                block_counters,
                language,
            )


# -- Tree walking --------------------------------------------------------------


def _ts_walk(node: Any) -> Iterator[Any]:
    """Yield all tree-sitter nodes reachable from *node* (depth-first)."""
    yield node
    for child in node.children:
        yield from _ts_walk(child)


# -- Annotation / type helpers -------------------------------------------------


def _ts_build_annotation_map(tree: Any) -> dict[str, str]:
    """Build variable→type mapping from function parameter type annotations."""
    ann_map: dict[str, str] = {}
    for node in _ts_walk(tree.root_node):
        if node.type not in _FUNCTION_NODE_TYPES:
            continue
        params_node = _get_child_by_field(node, "parameters")
        if params_node is None:
            continue
        for param in params_node.children:
            _extract_param_annotation(param, ann_map)
    return ann_map


def _extract_param_annotation(param: Any, ann_map: dict[str, str]) -> None:
    """Extract type annotation from a single parameter node."""
    ptype = param.type

    if ptype == "identifier":
        # Plain identifier without annotation — check if sibling is type_annotation
        # In tree-sitter, typed params are wrapped in required_parameter
        return

    if ptype in ("required_parameter", "optional_parameter"):
        name_node = None
        ann_node = None
        for child in param.children:
            if child.type == "identifier":
                name_node = child
            elif child.type == "type_annotation":
                ann_node = child
        if name_node is not None and ann_node is not None:
            name = name_node.text.decode("utf-8", errors="replace")
            type_name = _ts_annotation_name(ann_node)
            if type_name is not None:
                ann_map[name] = type_name
        return

    # Assignment pattern with annotation (default value)
    if ptype == "assignment_pattern":
        left = None
        for child in param.children:
            if child.type in ("required_parameter", "identifier"):
                left = child
                break
        if left is not None and left.type == "required_parameter":
            _extract_param_annotation(left, ann_map)


def _ts_annotation_name(node: Any) -> str | None:
    """Resolve a tree-sitter type annotation node to a simple type name."""
    if node is None:
        return None

    ntype = node.type

    # type_annotation wraps the actual type node
    if ntype == "type_annotation":
        for child in node.children:
            if child.is_named:
                return _ts_annotation_name(child)
        return None

    if ntype == "type_identifier":
        return str(node.text.decode("utf-8", errors="replace"))

    if ntype == "predefined_type":
        return str(node.text.decode("utf-8", errors="replace"))

    if ntype == "generic_type":
        # e.g. Array<number> → "Array"
        for child in node.children:
            if child.type == "type_identifier":
                return str(child.text.decode("utf-8", errors="replace"))
        return None

    return None


def _resolve_ts_operand_type(
    node: Any,
    annotation_map: dict[str, str],
) -> str | None:
    """Resolve an operand to a type string via annotation lookup or literal."""
    ntype = node.type

    if ntype == "identifier":
        name = node.text.decode("utf-8", errors="replace")
        return annotation_map.get(name)

    if ntype == "string":
        return "string"

    if ntype == "number":
        return "number"

    if ntype == "template_string":
        return "string"

    return None


# -- Internal helpers ----------------------------------------------------------


def _get_function_name(node: Any) -> str:
    """Extract the function name from a function declaration/expression node."""
    name_node = _get_child_by_field(node, "name")
    if name_node is not None:
        return str(name_node.text.decode("utf-8", errors="replace"))
    return "<anonymous>"


def _get_method_name(node: Any) -> str:
    """Extract the method name from a method_definition node."""
    name_node = _get_child_by_field(node, "name")
    if name_node is not None:
        return str(name_node.text.decode("utf-8", errors="replace"))
    return "<anonymous>"


def _get_child_by_field(node: Any, field_name: str) -> Any | None:
    """Get child by field name, returning None if not found."""
    result = node.child_by_field_name(field_name)
    return result
