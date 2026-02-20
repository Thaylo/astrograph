"""Java language adapter using the LSP plugin base."""

from __future__ import annotations

import re

from ._configured_lsp_plugin import BraceLanguageLSPPlugin
from .base import SemanticSignal, binary_signal_value

# -- Annotation patterns --
_JAVA_ANNOTATION_RE = re.compile(r"^\s*@([A-Za-z_]\w*)", re.MULTILINE)

# -- Access modifier patterns --
_JAVA_PUBLIC_RE = re.compile(r"\bpublic\b")
_JAVA_PRIVATE_RE = re.compile(r"\bprivate\b")
_JAVA_PROTECTED_RE = re.compile(r"\bprotected\b")

# -- Generic type patterns --
_JAVA_GENERIC_RE = re.compile(r"<\s*[A-Z]\w*(?:\s+extends\b)?")

# -- Exception handling patterns --
_JAVA_THROWS_RE = re.compile(r"\bthrows\s+\w")
_JAVA_TRY_CATCH_RE = re.compile(r"\btry\s*\{|\bcatch\s*\(")

# -- Stream / lambda patterns --
_JAVA_STREAM_RE = re.compile(r"\.stream\s*\(|\.map\s*\(|\.filter\s*\(|\.collect\s*\(")
_JAVA_LAMBDA_RE = re.compile(r"\)\s*->|[A-Za-z_]\w*\s*->")

# -- Static / final patterns --
_JAVA_STATIC_RE = re.compile(r"\bstatic\b")
_JAVA_FINAL_RE = re.compile(r"\bfinal\b")

# -- Interface vs class --
_JAVA_INTERFACE_RE = re.compile(r"\binterface\s+[A-Z]")
_JAVA_ABSTRACT_RE = re.compile(r"\babstract\s+class\b")
_JAVA_CLASS_RE = re.compile(r"\bclass\s+[A-Z]")

# -- Java 17-25 modern features --
_JAVA_RECORD_RE = re.compile(r"\brecord\s+[A-Z]\w*\s*\(")
_JAVA_SEALED_RE = re.compile(r"\bsealed\s+(?:class|interface)\b")
_JAVA_PERMITS_RE = re.compile(r"\bpermits\s+[A-Z]")
_JAVA_PATTERN_INSTANCEOF_RE = re.compile(r"\binstanceof\s+[A-Z]\w+\s+\w+")
_JAVA_SWITCH_EXPR_ARROW_RE = re.compile(r"\bcase\s+[^:]+\s*->")
_JAVA_TEXT_BLOCK_RE = re.compile(r'"""')
_JAVA_VAR_RE = re.compile(r"\bvar\s+\w+")

# -- Spring stereotype annotations --
_JAVA_SPRING_REST_CONTROLLER_RE = re.compile(r"@RestController\b")
_JAVA_SPRING_CONTROLLER_RE = re.compile(r"@Controller\b")
_JAVA_SPRING_SERVICE_RE = re.compile(r"@Service\b")
_JAVA_SPRING_REPOSITORY_RE = re.compile(r"@Repository\b")
_JAVA_SPRING_COMPONENT_RE = re.compile(r"@Component\b")
_JAVA_SPRING_CONFIGURATION_RE = re.compile(r"@Configuration\b")

# -- REST / HTTP patterns --
_JAVA_REQUEST_MAPPING_RE = re.compile(r"@(?:Request|Get|Post|Put|Delete|Patch)Mapping\b")
_JAVA_RESPONSE_ENTITY_RE = re.compile(r"\bResponseEntity\b")
_JAVA_PATH_VARIABLE_RE = re.compile(r"@PathVariable\b")
_JAVA_REQUEST_BODY_RE = re.compile(r"@RequestBody\b")
_JAVA_REQUEST_PARAM_RE = re.compile(r"@RequestParam\b")

# -- Persistence / JPA patterns --
_JAVA_JPA_ENTITY_RE = re.compile(r"@Entity\b")
_JAVA_JPA_TABLE_RE = re.compile(r"@Table\b")
_JAVA_JPA_COLUMN_RE = re.compile(r"@Column\b")
_JAVA_JPA_REPOSITORY_RE = re.compile(
    r"\bextends\s+(?:JpaRepository|CrudRepository|PagingAndSortingRepository)\b"
)
_JAVA_ENTITY_MANAGER_RE = re.compile(r"\bEntityManager\b")
_JAVA_TRANSACTIONAL_RE = re.compile(r"@Transactional\b")
_JAVA_QUERY_RE = re.compile(r"@Query\b")

# -- Dependency injection patterns --
_JAVA_AUTOWIRED_RE = re.compile(r"@Autowired\b")
_JAVA_INJECT_RE = re.compile(r"@Inject\b")
_JAVA_BEAN_RE = re.compile(r"@Bean\b")
_JAVA_QUALIFIER_RE = re.compile(r"@Qualifier\b")
_JAVA_VALUE_ANNOTATION_RE = re.compile(r"@Value\s*\(")

# -- Async / reactive patterns --
_JAVA_COMPLETABLE_FUTURE_RE = re.compile(r"\bCompletableFuture\b")
_JAVA_EXECUTOR_SERVICE_RE = re.compile(r"\bExecutorService\b")
_JAVA_MONO_RE = re.compile(r"\bMono<")
_JAVA_FLUX_RE = re.compile(r"\bFlux<")
_JAVA_ASYNC_RE = re.compile(r"@Async\b")
_JAVA_SCHEDULED_RE = re.compile(r"@Scheduled\b")


class JavaLSPPlugin(BraceLanguageLSPPlugin):
    """Java support via an attached or spawned LSP backend."""

    LANGUAGE_ID = "java_lsp"
    LSP_LANGUAGE_ID = "java"
    FILE_EXTENSIONS = frozenset({".java"})
    SKIP_DIRS = frozenset({"build", "out", "target", ".gradle"})
    DEFAULT_COMMAND = ("tcp://127.0.0.1:2089",)
    COMMAND_ENV_VAR = "ASTROGRAPH_JAVA_LSP_COMMAND"
    TIMEOUT_ENV_VAR = "ASTROGRAPH_JAVA_LSP_TIMEOUT"

    def _language_signals(self, source: str) -> list[SemanticSignal]:
        """Java-specific regex signals for semantic profiling."""
        signals = super()._language_signals(source)

        # 1. Annotations
        annotations = sorted({m.group(1) for m in _JAVA_ANNOTATION_RE.finditer(source)})
        signals.append(
            SemanticSignal(
                key="java.annotations",
                value=",".join(annotations) if annotations else "none",
                confidence=0.95,
                origin="syntax",
            )
        )

        # 2. Access modifiers
        modifiers = []
        if _JAVA_PUBLIC_RE.search(source):
            modifiers.append("public")
        if _JAVA_PRIVATE_RE.search(source):
            modifiers.append("private")
        if _JAVA_PROTECTED_RE.search(source):
            modifiers.append("protected")
        signals.append(
            SemanticSignal(
                key="java.access_modifiers",
                value=",".join(modifiers) if modifiers else "package",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 3. Generic usage
        has_generic = bool(_JAVA_GENERIC_RE.search(source))
        signals.append(
            SemanticSignal(
                key="java.generic.present",
                value="yes" if has_generic else "no",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 4. Exception handling
        has_throws = bool(_JAVA_THROWS_RE.search(source))
        has_try_catch = bool(_JAVA_TRY_CATCH_RE.search(source))
        signals.append(
            SemanticSignal(
                key="java.exception_handling",
                value=binary_signal_value(has_throws, "throws", has_try_catch, "try_catch"),
                confidence=0.90,
                origin="syntax",
            )
        )

        # 5. Stream / lambda style
        has_stream = bool(_JAVA_STREAM_RE.search(source))
        has_lambda = bool(_JAVA_LAMBDA_RE.search(source))
        signals.append(
            SemanticSignal(
                key="java.functional_style",
                value=binary_signal_value(
                    has_stream, "stream", has_lambda, "lambda", both_label="stream_lambda"
                ),
                confidence=0.85,
                origin="syntax",
            )
        )

        # 6. Class kind
        has_interface = bool(_JAVA_INTERFACE_RE.search(source))
        has_abstract = bool(_JAVA_ABSTRACT_RE.search(source))
        has_class = bool(_JAVA_CLASS_RE.search(source))
        if has_interface:
            class_kind = "interface"
        elif has_abstract:
            class_kind = "abstract"
        elif has_class:
            class_kind = "class"
        else:
            class_kind = "none"
        signals.append(
            SemanticSignal(
                key="java.class_kind",
                value=class_kind,
                confidence=0.90,
                origin="syntax",
            )
        )

        # 7. Modern Java features (Java 14-25)
        modern_parts = []
        if _JAVA_RECORD_RE.search(source):
            modern_parts.append("record")
        if _JAVA_SEALED_RE.search(source):
            modern_parts.append("sealed")
        if _JAVA_PATTERN_INSTANCEOF_RE.search(source):
            modern_parts.append("pattern_instanceof")
        if _JAVA_SWITCH_EXPR_ARROW_RE.search(source):
            modern_parts.append("switch_expression")
        if _JAVA_TEXT_BLOCK_RE.search(source):
            modern_parts.append("text_block")
        if _JAVA_VAR_RE.search(source):
            modern_parts.append("var")
        signals.append(
            SemanticSignal(
                key="java.modern_features",
                value=",".join(modern_parts) if modern_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 8. Spring stereotypes
        stereo_parts = []
        if _JAVA_SPRING_REST_CONTROLLER_RE.search(source):
            stereo_parts.append("rest_controller")
        if _JAVA_SPRING_CONTROLLER_RE.search(source):
            stereo_parts.append("controller")
        if _JAVA_SPRING_SERVICE_RE.search(source):
            stereo_parts.append("service")
        if _JAVA_SPRING_REPOSITORY_RE.search(source):
            stereo_parts.append("repository")
        if _JAVA_SPRING_COMPONENT_RE.search(source):
            stereo_parts.append("component")
        if _JAVA_SPRING_CONFIGURATION_RE.search(source):
            stereo_parts.append("configuration")
        signals.append(
            SemanticSignal(
                key="java.spring_stereotypes",
                value=",".join(stereo_parts) if stereo_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 9. REST / HTTP patterns
        rest_parts = []
        if _JAVA_REQUEST_MAPPING_RE.search(source):
            rest_parts.append("mapping")
        if _JAVA_RESPONSE_ENTITY_RE.search(source):
            rest_parts.append("response_entity")
        if _JAVA_PATH_VARIABLE_RE.search(source):
            rest_parts.append("path_variable")
        if _JAVA_REQUEST_BODY_RE.search(source):
            rest_parts.append("request_body")
        if _JAVA_REQUEST_PARAM_RE.search(source):
            rest_parts.append("request_param")
        signals.append(
            SemanticSignal(
                key="java.rest_http_patterns",
                value=",".join(rest_parts) if rest_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 10. Persistence / JPA
        jpa_parts = []
        if _JAVA_JPA_ENTITY_RE.search(source):
            jpa_parts.append("entity")
        if _JAVA_JPA_TABLE_RE.search(source):
            jpa_parts.append("table")
        if _JAVA_JPA_COLUMN_RE.search(source):
            jpa_parts.append("column")
        if _JAVA_JPA_REPOSITORY_RE.search(source):
            jpa_parts.append("repository_interface")
        if _JAVA_ENTITY_MANAGER_RE.search(source):
            jpa_parts.append("entity_manager")
        if _JAVA_TRANSACTIONAL_RE.search(source):
            jpa_parts.append("transactional")
        if _JAVA_QUERY_RE.search(source):
            jpa_parts.append("query")
        signals.append(
            SemanticSignal(
                key="java.persistence_jpa",
                value=",".join(jpa_parts) if jpa_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 11. Dependency injection
        di_parts = []
        if _JAVA_AUTOWIRED_RE.search(source):
            di_parts.append("autowired")
        if _JAVA_INJECT_RE.search(source):
            di_parts.append("inject")
        if _JAVA_BEAN_RE.search(source):
            di_parts.append("bean")
        if _JAVA_QUALIFIER_RE.search(source):
            di_parts.append("qualifier")
        if _JAVA_VALUE_ANNOTATION_RE.search(source):
            di_parts.append("value")
        signals.append(
            SemanticSignal(
                key="java.dependency_injection",
                value=",".join(di_parts) if di_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        # 12. Async / reactive
        async_parts = []
        if _JAVA_COMPLETABLE_FUTURE_RE.search(source):
            async_parts.append("completable_future")
        if _JAVA_EXECUTOR_SERVICE_RE.search(source):
            async_parts.append("executor_service")
        if _JAVA_MONO_RE.search(source):
            async_parts.append("mono")
        if _JAVA_FLUX_RE.search(source):
            async_parts.append("flux")
        if _JAVA_ASYNC_RE.search(source):
            async_parts.append("async")
        if _JAVA_SCHEDULED_RE.search(source):
            async_parts.append("scheduled")
        signals.append(
            SemanticSignal(
                key="java.async_reactive",
                value=",".join(async_parts) if async_parts else "none",
                confidence=0.90,
                origin="syntax",
            )
        )

        return signals
