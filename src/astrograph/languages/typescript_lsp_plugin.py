"""TypeScript language adapter extending the JavaScript LSP plugin."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import networkx as nx

from .base import CodeUnit, SemanticSignal
from .javascript_lsp_plugin import (
    JavaScriptLSPPlugin,
    _esprima_ast_to_graph,
    _esprima_extract_function_blocks,
    _esprima_try_parse,
)

logger = logging.getLogger(__name__)

# -- TS annotation stripping patterns --
# Type annotations after identifiers: `param: Type` → `param`
_TS_TYPE_ANNOTATION_RE = re.compile(
    r"([\w$]+)\s*:\s*" r"(?:[A-Za-z_$][\w$.<>,\s|&\[\]]*)" r"(?=\s*[,);=\]}])"
)
# Return type annotations: `): Type` → `)`
_TS_RETURN_TYPE_RE = re.compile(r"\)\s*:\s*[A-Za-z_$][\w$.<>,\s|&\[\]]*(?=\s*[{=>;])")
# Interface / type alias declarations (entire blocks)
_TS_INTERFACE_BLOCK_RE = re.compile(
    r"\b(?:interface|type)\s+[A-Z][\w$]*(?:<[^>]*>)?\s*(?:extends\s+[^{]*)?\{[^}]*\}",
    re.DOTALL,
)
# Generic type parameters: `<T>`, `<T extends Foo>`
_TS_GENERIC_PARAMS_RE = re.compile(
    r"<\s*[A-Z][\w$]*(?:\s+extends\s+[^>]*)?\s*(?:,\s*[A-Z][\w$]*(?:\s+extends\s+[^>]*)?\s*)*>"
)
# `as Type` casts
_TS_AS_CAST_RE = re.compile(r"\bas\s+[A-Za-z_$][\w$.<>,\s|&\[\]]*")
# Non-null assertion: `x!.` or `x!)`
_TS_NON_NULL_RE = re.compile(r"(\w)!(?=[.\[)\],;])")
# `readonly` modifier
_TS_READONLY_RE = re.compile(r"\breadonly\s+")
# `declare` keyword
_TS_DECLARE_RE = re.compile(r"\bdeclare\s+")
# Enum declarations
_TS_ENUM_RE = re.compile(r"\b(?:const\s+)?enum\s+\w+\s*\{[^}]*\}", re.DOTALL)
# Namespace declarations
_TS_NAMESPACE_BLOCK_RE = re.compile(
    r"\bnamespace\s+\w+\s*\{[^}]*\}",
    re.DOTALL,
)

# -- TS 5.x+ syntax (decorators, using, satisfies) --
# Stage 3 decorators (esprima doesn't support them)
_TS_DECORATOR_RE = re.compile(r"^\s*@\w[\w$.]*(?:\([^)]*\))?\s*$", re.MULTILINE)
# `using` / `await using` (TS 5.2+, explicit resource management)
_TS_USING_RE = re.compile(r"\b(?:await\s+)?using\s+[A-Za-z_$][\w$]*\s*=")
# `satisfies` operator (TS 4.9+)
_TS_SATISFIES_RE = re.compile(r"\bsatisfies\s+[A-Za-z_$][\w$.<>,\s|&\[\]]*")
# `accessor` keyword (TS 4.9+ auto-accessor)
_TS_ACCESSOR_RE = re.compile(r"\baccessor\s+\w+")

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


def _strip_ts_annotations(source: str) -> str:
    """Remove TypeScript-specific syntax so esprima can parse the result."""
    result = source
    # Remove decorators (Stage 3, not supported by esprima)
    result = _TS_DECORATOR_RE.sub("", result)
    # Remove enum declarations
    result = _TS_ENUM_RE.sub("", result)
    # Remove namespace blocks
    result = _TS_NAMESPACE_BLOCK_RE.sub("", result)
    # Remove interface / type alias blocks
    result = _TS_INTERFACE_BLOCK_RE.sub("", result)
    # Remove declare statements
    result = _TS_DECLARE_RE.sub("", result)
    # Replace `using`/`await using` with `const` (preserves structure)
    result = _TS_USING_RE.sub(
        lambda m: "const " + m.group(0).split("=")[0].split()[-1] + " =", result
    )
    # Remove `satisfies Type` (TS 4.9+)
    result = _TS_SATISFIES_RE.sub("", result)
    # Remove `accessor` keyword (TS 4.9+)
    result = _TS_ACCESSOR_RE.sub(lambda m: m.group(0).replace("accessor ", ""), result)
    # Remove generic type params
    result = _TS_GENERIC_PARAMS_RE.sub("", result)
    # Remove as casts
    result = _TS_AS_CAST_RE.sub("", result)
    # Remove non-null assertions
    result = _TS_NON_NULL_RE.sub(r"\1", result)
    # Remove readonly
    result = _TS_READONLY_RE.sub("", result)
    # Remove return type annotations (before param annotations to avoid overlap)
    result = _TS_RETURN_TYPE_RE.sub(")", result)
    # Remove param type annotations
    result = _TS_TYPE_ANNOTATION_RE.sub(r"\1", result)
    return result


class TypeScriptLSPPlugin(JavaScriptLSPPlugin):
    """TypeScript support via esprima (with annotation stripping) + JS semantic signals."""

    LANGUAGE_ID = "typescript_lsp"
    LSP_LANGUAGE_ID = "typescript"
    FILE_EXTENSIONS = frozenset({".ts", ".tsx"})
    SKIP_DIRS = frozenset({"node_modules", ".next", "dist", "build"})
    DEFAULT_COMMAND = ("typescript-language-server", "--stdio")
    COMMAND_ENV_VAR = "ASTROGRAPH_TS_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_TS_LSP_TIMEOUT"

    # ------------------------------------------------------------------
    # AST graph builder (strip TS → parse as JS)
    # ------------------------------------------------------------------

    def source_to_graph(
        self,
        source: str,
        normalize_ops: bool = False,
    ) -> nx.DiGraph:
        """Strip TS annotations then build an esprima AST graph."""
        stripped = _strip_ts_annotations(source)
        graph = _esprima_ast_to_graph(stripped, normalize_ops=normalize_ops)
        if graph is not None:
            return graph
        logger.debug("esprima parse failed for TS source, falling back to line-level parser")
        return super(JavaScriptLSPPlugin, self).source_to_graph(source, normalize_ops=normalize_ops)

    # ------------------------------------------------------------------
    # Block extraction (strip TS → parse as JS → extract blocks)
    # ------------------------------------------------------------------

    def extract_code_units(
        self,
        source: str,
        file_path: str = "<unknown>",
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> Iterator[CodeUnit]:
        """Extract units via LSP, then extract inner blocks from stripped TS."""
        yield from super(JavaScriptLSPPlugin, self).extract_code_units(
            source,
            file_path,
            include_blocks=False,
            max_block_depth=max_block_depth,
        )

        if not include_blocks:
            return

        stripped = _strip_ts_annotations(source)
        tree = _esprima_try_parse(stripped, loc=True)
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
