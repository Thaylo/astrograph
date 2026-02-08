"""Python language adapter using the LSP plugin base."""

from __future__ import annotations

import ast
from collections.abc import Iterator

import networkx as nx

from ._configured_lsp_plugin import ConfiguredLSPLanguagePluginBase
from ._lsp_base import LSPClient
from .base import CodeUnit, SemanticProfile, SemanticSignal
from .python_plugin import PythonPlugin, ast_to_graph, extract_blocks_from_function


class PythonLSPPlugin(ConfiguredLSPLanguagePluginBase):
    """Python support via LSP symbols + AST block extraction."""

    LANGUAGE_ID = "python"
    LSP_LANGUAGE_ID = "python"
    FILE_EXTENSIONS = frozenset({".py", ".pyi"})
    SKIP_DIRS = frozenset({"__pycache__", "venv", ".venv", ".tox", ".mypy_cache"})
    DEFAULT_COMMAND = ("pylsp",)
    COMMAND_ENV_VAR = "ASTROGRAPH_PY_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_PY_LSP_TIMEOUT"

    def __init__(self, lsp_client: LSPClient | None = None) -> None:
        super().__init__(lsp_client=lsp_client)
        self._graph_plugin = PythonPlugin()

    def source_to_graph(self, source: str, normalize_ops: bool = False) -> nx.DiGraph:
        """Use the Python AST graph builder for structural fidelity."""
        return ast_to_graph(source, normalize_ops=normalize_ops)

    def normalize_graph_for_pattern(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Reuse Python operator normalization used by the legacy plugin."""
        return self._graph_plugin.normalize_graph_for_pattern(graph)

    def _method_parent_map(self, tree: ast.AST) -> dict[tuple[str, int, int], str]:
        """Build mapping from method (name,start,end) to its class name."""
        mapping: dict[tuple[str, int, int], str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    end = item.end_lineno if item.end_lineno is not None else item.lineno
                    mapping[(item.name, item.lineno, end)] = node.name
        return mapping

    def _apply_method_parents(
        self,
        units: list[CodeUnit],
        method_map: dict[tuple[str, int, int], str],
    ) -> list[CodeUnit]:
        """Patch parent_name/unit_type for methods when LSP returns flat symbols."""
        adjusted: list[CodeUnit] = []
        for unit in units:
            key = (unit.name, unit.line_start, unit.line_end)
            parent_name = method_map.get(key)
            if parent_name is None:
                adjusted.append(unit)
                continue

            unit_type = "method" if unit.unit_type in {"method", "function"} else unit.unit_type
            adjusted.append(
                CodeUnit(
                    name=unit.name,
                    code=unit.code,
                    file_path=unit.file_path,
                    line_start=unit.line_start,
                    line_end=unit.line_end,
                    unit_type=unit_type,
                    parent_name=parent_name,
                    block_type=unit.block_type,
                    nesting_depth=unit.nesting_depth,
                    parent_block_name=unit.parent_block_name,
                    language=unit.language,
                )
            )
        return adjusted

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract units via LSP and optionally derive Python blocks via AST."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            tree = None

        method_map = self._method_parent_map(tree) if tree is not None else {}

        units = list(
            super().extract_code_units(
                source,
                file_path,
                include_blocks=False,
                max_block_depth=max_block_depth,
            )
        )
        adjusted_units = self._apply_method_parents(units, method_map)
        yield from adjusted_units

        if not include_blocks or tree is None:
            return

        source_lines = source.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                yield from extract_blocks_from_function(
                    node,
                    source_lines,
                    file_path,
                    max_depth=max_block_depth,
                )

    # ------------------------------------------------------------------
    # Semantic profiling helpers (AST-based, no regex)
    # ------------------------------------------------------------------

    def _collect_dunder_methods(self, tree: ast.AST) -> set[str]:
        """Find ``__x__`` methods defined in class bodies."""
        dunders: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    and item.name.startswith("__")
                    and item.name.endswith("__")
                ):
                    dunders.add(item.name)
        return dunders

    def _annotation_to_name(self, annotation: ast.expr | None) -> str | None:
        """Extract a simple type name from an annotation AST node."""
        if annotation is None:
            return None
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            return annotation.value
        if isinstance(annotation, ast.Attribute):
            return annotation.attr
        return None

    def _compute_annotation_density(self, tree: ast.AST) -> str:
        """Ratio of annotated params + returns to total (excludes self/cls)."""
        total = 0
        annotated = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for arg in node.args.args:
                if arg.arg in ("self", "cls"):
                    continue
                total += 1
                if arg.annotation is not None:
                    annotated += 1
            if node.returns is not None:
                annotated += 1
                total += 1
            elif total > 0:
                total += 1  # count missing return annotation
        if total == 0:
            return "none"
        ratio = annotated / total
        if ratio >= 0.9:
            return "full"
        if ratio > 0.0:
            return "partial"
        return "none"

    def _collect_decorators(self, tree: ast.AST) -> set[str]:
        """Extract terminal decorator names from ClassDef/FunctionDef nodes."""
        decorators: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    decorators.add(dec.id)
                elif isinstance(dec, ast.Attribute):
                    decorators.add(dec.attr)
                elif isinstance(dec, ast.Call):
                    func = dec.func
                    if isinstance(func, ast.Name):
                        decorators.add(func.id)
                    elif isinstance(func, ast.Attribute):
                        decorators.add(func.attr)
        return decorators

    def _detect_async_constructs(self, tree: ast.AST) -> bool:
        """Return True if any async construct is present."""
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef | ast.Await | ast.AsyncFor | ast.AsyncWith):
                return True
        return False

    def _build_annotation_map(self, tree: ast.AST) -> dict[str, str]:
        """Build a mapping from variable name to its annotated type name."""
        ann_map: dict[str, str] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for arg in node.args.args:
                type_name = self._annotation_to_name(arg.annotation)
                if type_name is not None:
                    ann_map[arg.arg] = type_name
        return ann_map

    def _resolve_operand_type(
        self,
        node: ast.expr,
        annotation_map: dict[str, str],
    ) -> str | None:
        """Resolve operand type from Name lookup or Constant inference."""
        if isinstance(node, ast.Name):
            return annotation_map.get(node.id)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float):
                return "numeric"
            if isinstance(node.value, str):
                return "str"
        return None

    _PYTHON_NUMERIC_TYPES = frozenset({"int", "float", "complex", "Decimal", "numeric"})

    def _infer_plus_binding(self, tree: ast.AST) -> tuple[str, float] | None:
        """Find BinOp(Add), resolve operand types via annotation map."""
        annotation_map = self._build_annotation_map(tree)
        found_plus = False
        saw_numeric = False
        saw_str = False
        saw_user = False
        saw_unknown = False

        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
                continue
            found_plus = True
            left_type = self._resolve_operand_type(node.left, annotation_map)
            right_type = self._resolve_operand_type(node.right, annotation_map)

            if left_type is None or right_type is None:
                saw_unknown = True
                continue
            if left_type in self._PYTHON_NUMERIC_TYPES and right_type in self._PYTHON_NUMERIC_TYPES:
                saw_numeric = True
            elif left_type == "str" and right_type == "str":
                saw_str = True
            else:
                saw_user = True

        if not found_plus:
            return None

        if saw_user:
            return "user_defined", 0.7
        if saw_str and saw_numeric:
            return "mixed", 0.6
        if saw_str:
            return "str_concat", 0.9
        if saw_numeric:
            return "numeric", 0.9
        if saw_unknown:
            return "unknown", 0.5
        return "unknown", 0.5

    def _detect_class_style(self, tree: ast.AST) -> str | None:
        """Detect class style: dataclass, protocol, abstract, or plain."""
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # Check decorators for @dataclass
            for dec in node.decorator_list:
                name = None
                if isinstance(dec, ast.Name):
                    name = dec.id
                elif isinstance(dec, ast.Attribute):
                    name = dec.attr
                elif isinstance(dec, ast.Call):
                    func = dec.func
                    if isinstance(func, ast.Name):
                        name = func.id
                    elif isinstance(func, ast.Attribute):
                        name = func.attr
                if name == "dataclass":
                    return "dataclass"
            # Check bases for Protocol / ABC
            for base in node.bases:
                base_name = None
                if isinstance(base, ast.Name):
                    base_name = base.id
                elif isinstance(base, ast.Attribute):
                    base_name = base.attr
                if base_name == "Protocol":
                    return "protocol"
                if base_name in ("ABC", "ABCMeta"):
                    return "abstract"
            return "plain"
        return None

    @staticmethod
    def _append_set_signal(
        signals: list[SemanticSignal],
        items: set[str],
        key: str,
        confidence: float,
    ) -> None:
        """Append a sorted, comma-joined set signal."""
        signals.append(
            SemanticSignal(
                key=key,
                value=",".join(sorted(items)),
                confidence=confidence,
                origin="ast",
            )
        )

    def extract_semantic_profile(
        self,
        source: str,
        file_path: str = "<unknown>",
    ) -> SemanticProfile:
        """Extract Python-specific semantic signals using the ast module."""
        base_profile = super().extract_semantic_profile(source=source, file_path=file_path)
        signals = list(base_profile.signals)
        notes = list(base_profile.notes)

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return base_profile

        extra_coverage = 0.0

        # 1. Dunder methods
        dunders = self._collect_dunder_methods(tree)
        if dunders:
            self._append_set_signal(signals, dunders, "python.dunder_methods.defined", 0.95)
            extra_coverage += 0.15

        # 2. Annotation density (always emitted)
        density = self._compute_annotation_density(tree)
        signals.append(
            SemanticSignal(
                key="python.type_annotation.density",
                value=density,
                confidence=0.85,
                origin="ast",
            )
        )
        extra_coverage += 0.10

        # 3. Decorators
        decorators = self._collect_decorators(tree)
        if decorators:
            self._append_set_signal(signals, decorators, "python.decorators.present", 0.95)
            extra_coverage += 0.10

        # 4. Async constructs (always emitted)
        has_async = self._detect_async_constructs(tree)
        signals.append(
            SemanticSignal(
                key="python.async.present",
                value="yes" if has_async else "no",
                confidence=0.95,
                origin="ast",
            )
        )
        extra_coverage += 0.10

        # 5. Plus binding
        plus_result = self._infer_plus_binding(tree)
        if plus_result is not None:
            binding, confidence = plus_result
            signals.append(
                SemanticSignal(
                    key="python.plus_binding",
                    value=binding,
                    confidence=confidence,
                    origin="ast",
                )
            )
            extra_coverage += 0.15

        # 6. Class style
        class_style = self._detect_class_style(tree)
        if class_style is not None:
            signals.append(
                SemanticSignal(
                    key="python.class_style",
                    value=class_style,
                    confidence=0.9,
                    origin="ast",
                )
            )
            extra_coverage += 0.10

        return SemanticProfile(
            signals=tuple(signals),
            coverage=min(1.0, base_profile.coverage + extra_coverage),
            notes=tuple(notes),
            extractor="python:ast",
        )
