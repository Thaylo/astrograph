"""TypeScript language adapter extending the JavaScript LSP plugin."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import networkx as nx

from ._js_ts_treesitter import (
    _ts_ast_to_graph,
    _ts_extract_function_blocks,
    _ts_try_parse,
)
from .base import CodeUnit, SemanticSignal
from .javascript_lsp_plugin import JavaScriptLSPPlugin

logger = logging.getLogger(__name__)

# -- NestJS decorator patterns --
_TS_NEST_CONTROLLER_RE = re.compile(r"@Controller\s*\(")
_TS_NEST_INJECTABLE_RE = re.compile(r"@Injectable\s*\(")
_TS_NEST_MODULE_RE = re.compile(r"@Module\s*\(")
_TS_NEST_GUARD_RE = re.compile(r"@UseGuards\s*\(")
_TS_NEST_PIPE_RE = re.compile(r"@UsePipes\s*\(")
_TS_NEST_INTERCEPTOR_RE = re.compile(r"@UseInterceptors\s*\(")

# -- REST / HTTP decorators + framework patterns --
_TS_HTTP_METHOD_RE = re.compile(r"@(?:Get|Post|Put|Delete|Patch|Head|Options|All)\s*\(")
_TS_PARAM_DECORATOR_RE = re.compile(r"@(?:Param|Query|Body|Headers)\s*\(")
_TS_RES_STATUS_RE = re.compile(r"@(?:HttpCode|Header)\s*\(")
_TS_EXPRESS_ROUTER_RE = re.compile(
    r"\bRouter\s*\(\)" r"|\b(?:app|router)\.(?:get|post|put|delete|patch)\s*\(" r"|\bexpress\s*\(\)"
)
_TS_FASTIFY_RE = re.compile(r"\b[Ff]astify\s*\(" r"|\.register\s*\(")

# -- ORM / persistence patterns --
_TS_TYPEORM_ENTITY_RE = re.compile(r"@Entity\s*\(")
_TS_TYPEORM_COLUMN_RE = re.compile(r"@(?:Column|PrimaryGeneratedColumn|PrimaryColumn)\s*\(")
_TS_TYPEORM_RELATION_RE = re.compile(r"@(?:OneToMany|ManyToOne|OneToOne|ManyToMany)\s*\(")
_TS_TYPEORM_REPOSITORY_RE = re.compile(
    r"\bRepository\s*<" r"|getRepository\s*\(" r"|@InjectRepository\s*\("
)
_TS_PRISMA_CLIENT_RE = re.compile(
    r"\bPrismaClient\b" r"|prisma\.\w+\.(?:findMany|findUnique|create|update|delete)\s*\("
)
_TS_MONGOOSE_RE = re.compile(
    r"\bmongoose\.Schema\b" r"|\bmongoose\.model\s*\(" r"|\bnew\s+Schema\s*\("
)

# -- Dependency injection patterns --
_TS_INJECT_RE = re.compile(r"@Inject\s*\(")
_TS_NEST_INJECT_RE = re.compile(r"@(?:InjectRepository|InjectModel|InjectConnection)\s*\(")
_TS_INVERSIFY_RE = re.compile(r"@(?:injectable|inject)\s*\(\s*\)")
_TS_PROVIDE_RE = re.compile(r"\b(?:useFactory|useClass|useValue)\s*:")

# -- Reactive / RxJS patterns --
_TS_OBSERVABLE_RE = re.compile(r"\bObservable\s*<")
_TS_SUBJECT_RE = re.compile(r"\b(?:Subject|BehaviorSubject|ReplaySubject)\s*<")
_TS_RXJS_PIPE_RE = re.compile(r"\.pipe\s*\(")
_TS_RXJS_OPERATORS_RE = re.compile(r"\b(?:switchMap|mergeMap|concatMap|exhaustMap|catchError)\s*\(")

# -- TS-specific semantic patterns --
_TS_GENERIC_DETECT_RE = re.compile(
    r"<\s*[A-Z][\w$]*(?:\s+extends\b)?" r"|(?:function|class|interface|type)\s+\w+\s*<"
)
_TS_STRICT_MODE_RE = re.compile(
    r"\bas\s+\w" r"|\w\s*!\s*[.\[]" r"|\w+\s*:\s*(?:string|number|boolean|void|any|never|unknown)"
)


class TypeScriptLSPPlugin(JavaScriptLSPPlugin):
    """TypeScript support via native tree-sitter TS parsing + JS semantic signals."""

    LANGUAGE_ID = "typescript_lsp"
    LSP_LANGUAGE_ID = "typescript"
    FILE_EXTENSIONS = frozenset({".ts", ".tsx"})
    SKIP_DIRS = frozenset({"node_modules", ".next", "dist", "build"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2093",)

    # ------------------------------------------------------------------
    # AST graph builder (native tree-sitter TypeScript)
    # ------------------------------------------------------------------

    def source_to_graph(
        self,
        source: str,
        normalize_ops: bool = False,
    ) -> nx.DiGraph:
        """Build a structural graph from TS source using tree-sitter."""
        graph = _ts_ast_to_graph(source, normalize_ops=normalize_ops, language="typescript")
        if graph is not None:
            return graph
        logger.debug("tree-sitter parse failed for TS source, falling back to line-level parser")
        return super(JavaScriptLSPPlugin, self).source_to_graph(source, normalize_ops=normalize_ops)

    # ------------------------------------------------------------------
    # Block extraction (native tree-sitter TypeScript)
    # ------------------------------------------------------------------

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract units via LSP, then extract inner blocks via tree-sitter TS."""
        yield from super(JavaScriptLSPPlugin, self).extract_code_units(
            source,
            file_path,
            include_blocks=False,
            max_block_depth=max_block_depth,
        )

        if not include_blocks:
            return

        tree = _ts_try_parse(source, "typescript")
        if tree is None:
            return

        source_lines = source.splitlines()
        yield from _ts_extract_function_blocks(
            tree,
            source_lines,
            file_path,
            max_depth=max_block_depth,
            language=self.LANGUAGE_ID,
        )

    # ------------------------------------------------------------------
    # Semantic profiling (inherited JS signals + TS-specific)
    # ------------------------------------------------------------------

    def _language_signals(self, source: str) -> list[SemanticSignal]:
        """TypeScript language signals (extends JS base signals).

        Adds TS-specific strict-mode and generic-usage indicators on top of
        the JavaScript signals provided by the parent class.
        """
        signals = super()._language_signals(source)

        # 1. Strict mode indicators (as/non-null/typed params)
        has_strict = bool(_TS_STRICT_MODE_RE.search(source))
        signals.append(
            SemanticSignal(
                key="typescript.strict_mode",
                value="yes" if has_strict else "no",
                confidence=0.85,
                origin="syntax",
            )
        )

        # 2. Generic usage (<T>, <T extends ...>, generic declarations)
        has_generic = bool(_TS_GENERIC_DETECT_RE.search(source))
        signals.append(
            SemanticSignal(
                key="typescript.generic.present",
                value="yes" if has_generic else "no",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 3. Framework decorators (NestJS)
        fd_parts: list[str] = []
        if _TS_NEST_CONTROLLER_RE.search(source):
            fd_parts.append("controller")
        if _TS_NEST_INJECTABLE_RE.search(source):
            fd_parts.append("injectable")
        if _TS_NEST_MODULE_RE.search(source):
            fd_parts.append("module")
        if _TS_NEST_GUARD_RE.search(source):
            fd_parts.append("guard")
        if _TS_NEST_PIPE_RE.search(source):
            fd_parts.append("pipe")
        if _TS_NEST_INTERCEPTOR_RE.search(source):
            fd_parts.append("interceptor")
        signals.append(
            SemanticSignal(
                key="typescript.framework_decorators",
                value=",".join(fd_parts) if fd_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 4. REST / HTTP patterns
        rh_parts: list[str] = []
        if _TS_HTTP_METHOD_RE.search(source):
            rh_parts.append("http_method")
        if _TS_PARAM_DECORATOR_RE.search(source):
            rh_parts.append("param_decorator")
        if _TS_RES_STATUS_RE.search(source):
            rh_parts.append("status_header")
        if _TS_EXPRESS_ROUTER_RE.search(source):
            rh_parts.append("express_router")
        if _TS_FASTIFY_RE.search(source):
            rh_parts.append("fastify")
        signals.append(
            SemanticSignal(
                key="typescript.rest_http_patterns",
                value=",".join(rh_parts) if rh_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 5. ORM / persistence
        orm_parts: list[str] = []
        if _TS_TYPEORM_ENTITY_RE.search(source):
            orm_parts.append("typeorm_entity")
        if _TS_TYPEORM_COLUMN_RE.search(source):
            orm_parts.append("typeorm_column")
        if _TS_TYPEORM_RELATION_RE.search(source):
            orm_parts.append("typeorm_relation")
        if _TS_TYPEORM_REPOSITORY_RE.search(source):
            orm_parts.append("typeorm_repository")
        if _TS_PRISMA_CLIENT_RE.search(source):
            orm_parts.append("prisma")
        if _TS_MONGOOSE_RE.search(source):
            orm_parts.append("mongoose")
        signals.append(
            SemanticSignal(
                key="typescript.orm_persistence",
                value=",".join(orm_parts) if orm_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 6. Dependency injection
        di_parts: list[str] = []
        if _TS_INJECT_RE.search(source):
            di_parts.append("inject")
        if _TS_NEST_INJECT_RE.search(source):
            di_parts.append("nest_inject")
        if _TS_INVERSIFY_RE.search(source):
            di_parts.append("inversify")
        if _TS_PROVIDE_RE.search(source):
            di_parts.append("provide")
        signals.append(
            SemanticSignal(
                key="typescript.dependency_injection",
                value=",".join(di_parts) if di_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 7. Reactive / RxJS
        rx_parts: list[str] = []
        if _TS_OBSERVABLE_RE.search(source):
            rx_parts.append("observable")
        if _TS_SUBJECT_RE.search(source):
            rx_parts.append("subject")
        if _TS_RXJS_PIPE_RE.search(source):
            rx_parts.append("pipe")
        if _TS_RXJS_OPERATORS_RE.search(source):
            rx_parts.append("operators")
        signals.append(
            SemanticSignal(
                key="typescript.reactive_rxjs",
                value=",".join(rx_parts) if rx_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        return signals
