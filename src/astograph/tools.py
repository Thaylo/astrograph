"""
Tool implementations for code structure analysis.

Simplified API with 4 core tools:
- index_codebase: Index Python files
- analyze: Find duplicates and similar patterns
- check: Check if code exists before creating
- compare: Compare two code snippets

Supports two modes:
- Standard mode: JSON persistence, manual re-indexing
- Event-driven mode: SQLite persistence, file watching, pre-computed analysis
"""

import itertools
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from .event_driven import EventDrivenIndex

from .ast_to_graph import ast_to_graph, node_match
from .canonical_hash import (
    fingerprints_compatible,
    structural_fingerprint,
    weisfeiler_leman_hash,
)
from .index import CodeStructureIndex, DuplicateGroup, IndexEntry

if TYPE_CHECKING:
    from .event_driven import EventDrivenIndex


# Persistence directory name for cached index
PERSISTENCE_DIR = ".metadata_astograph"


def _resolve_docker_path(path: str) -> str:
    """
    Resolve path, handling Docker volume mounts.

    When running in Docker with a volume mount like `-v ".:/workspace"`,
    the host paths don't exist inside the container. This function detects
    that situation and translates paths like:
      /Users/.../project/src → /workspace/src
    """
    # If path exists, use it directly
    if os.path.exists(path):
        return path

    # Check if we're in a Docker container with /workspace mount
    workspace = Path("/workspace")
    dockerenv = Path("/.dockerenv")
    if not (workspace.exists() and dockerenv.exists()):
        return path

    # Try to find a matching subpath under /workspace
    # For /Users/foo/project/src, try: /workspace/foo/project/src, /workspace/project/src, /workspace/src
    parts = Path(path).parts
    for i in range(len(parts)):
        candidate = workspace.joinpath(*parts[i:])
        if candidate.exists():
            return str(candidate)

    # Fallback to /workspace if nothing else matches
    return str(workspace)


def _get_persistence_path(indexed_path: str) -> Path:
    """
    Get metadata directory for an indexed codebase.

    In Docker with read-only mounts, the metadata is stored at the workspace
    root (which has a tmpfs mount) rather than the indexed subdirectory.
    """
    base = Path(indexed_path).resolve()
    if base.is_file():
        base = base.parent

    persistence_path = base / PERSISTENCE_DIR

    # In Docker, /workspace might be read-only but /workspace/.metadata_astograph
    # has a tmpfs mount. If we're indexing a subdirectory, use the root.
    workspace = Path("/workspace")
    if workspace.exists() and Path("/.dockerenv").exists() and str(base).startswith("/workspace/"):
        # Use workspace root for persistence (has tmpfs mount)
        persistence_path = workspace / PERSISTENCE_DIR

    return persistence_path


def _get_sqlite_path(indexed_path: str) -> Path:
    """Get SQLite database path for an indexed codebase."""
    return _get_persistence_path(indexed_path) / "index.db"


class DuplicateCategory(Enum):
    """Categories of duplicate code with different recommended actions."""

    IDIOMATIC_GUARD = "idiomatic_guard"  # Walrus guards, early returns
    IDIOMATIC_DICT_BUILD = "idiomatic_dict_build"  # Conditional dict building
    TEST_SETUP = "test_setup"  # Test fixture/setup patterns
    DELEGATE_METHOD = "delegate_method"  # Thin wrappers
    REFACTORABLE = "refactorable"  # True duplicates worth extracting


@dataclass
class DuplicateClassification:
    """Classification result for a duplicate group."""

    category: DuplicateCategory
    confidence: float  # 0.0 - 1.0
    reason: str
    recommendation: str
    suppress_suggestion: bool  # Whether to suggest suppression


@dataclass
class PatternSpec:
    """Specification for an idiomatic code pattern."""

    pattern: re.Pattern[str]
    category: DuplicateCategory
    confidence: float
    reason: str
    recommendation: str
    requires_block: bool = False  # If True, only match when is_block=True


class PatternClassifier:
    """
    Classifies duplicate code patterns to provide context-aware recommendations.

    Recognizes idiomatic Python patterns that often appear as "duplicates" but
    are intentional and shouldn't be refactored.
    """

    # Patterns for test file detection
    TEST_PATTERNS = ("test_", "_test.py", "tests/", "test/", "conftest.py")

    # Idiomatic patterns with their classifications
    IDIOMATIC_PATTERNS = [
        PatternSpec(
            pattern=re.compile(r"if\s+\w+\s*:=\s*\w+.*:\s*(return|raise)", re.MULTILINE),
            category=DuplicateCategory.IDIOMATIC_GUARD,
            confidence=0.95,
            reason="Walrus operator guard clause",
            recommendation=(
                "Guard clauses with walrus operator are idiomatic Python for early validation. "
                "The pattern 'if error := check(): return error' is intentionally repeated "
                "for clarity at each call site."
            ),
        ),
        PatternSpec(
            pattern=re.compile(r"if\s+(not\s+)?\w+.*:\s*return\b", re.MULTILINE),
            category=DuplicateCategory.IDIOMATIC_GUARD,
            confidence=0.9,
            reason="Early return guard clause",
            recommendation=(
                "Early returns for precondition checks are idiomatic. "
                "Inlining these guards improves readability vs. extracting to a helper."
            ),
            requires_block=True,
        ),
        PatternSpec(
            pattern=re.compile(r"if\s+[\w.]+:\s*\w+\[.+\]\s*=", re.MULTILINE),
            category=DuplicateCategory.IDIOMATIC_DICT_BUILD,
            confidence=0.85,
            reason="Conditional dictionary building",
            recommendation=(
                "Conditionally adding dict keys is idiomatic Python. "
                "Extracting this pattern would reduce clarity without meaningful benefit."
            ),
        ),
        PatternSpec(
            pattern=re.compile(r"if\s+.*:\s*continue\b", re.MULTILINE),
            category=DuplicateCategory.IDIOMATIC_GUARD,
            confidence=0.85,
            reason="Loop skip pattern",
            recommendation=(
                "Skip patterns in loops are idiomatic and context-specific. "
                "The condition often depends on loop-local variables."
            ),
            requires_block=True,
        ),
    ]

    def classify_group(self, group: DuplicateGroup) -> DuplicateClassification:
        """Classify a duplicate group and provide context-aware recommendation."""
        entries = group.entries
        if not entries:
            return self._make_classification(
                DuplicateCategory.REFACTORABLE,
                0.5,
                "Empty group",
                "Review manually",
                False,
            )

        # Get representative code sample
        sample_code = entries[0].code_unit.code
        line_count = entries[0].code_unit.line_end - entries[0].code_unit.line_start + 1
        is_block = entries[0].code_unit.block_type is not None

        # Check if all in test files
        all_test = all(self._is_test_file(e.code_unit.file_path) for e in entries)

        # Small blocks (2-4 lines) are likely idiomatic
        if line_count <= 4 or entries[0].node_count <= 12:
            classification = self._classify_small_pattern(sample_code, all_test, is_block)
            if classification:
                return classification

        # Test file duplicates
        if all_test:
            return self._make_classification(
                DuplicateCategory.TEST_SETUP,
                0.8,
                "All instances in test files",
                "Test duplication is often intentional for isolation and readability. "
                "Consider suppressing unless the setup is complex enough to warrant a shared fixture.",
                True,
            )

        # Check for delegate/wrapper methods
        if self._is_delegate_method(sample_code, entries):
            return self._make_classification(
                DuplicateCategory.DELEGATE_METHOD,
                0.75,
                "Thin wrapper delegating to shared implementation",
                "Delegate methods provide clean APIs while sharing implementation. "
                "This is good design, not problematic duplication.",
                True,
            )

        # Default: refactorable duplicate
        occurrence_count = len(entries)
        total_lines = sum(e.code_unit.line_end - e.code_unit.line_start + 1 for e in entries)
        return self._make_classification(
            DuplicateCategory.REFACTORABLE,
            0.9,
            f"{occurrence_count} identical implementations ({total_lines} total lines)",
            f"Extract to shared function to reduce {total_lines - line_count} duplicate lines. "
            "This will improve maintainability and ensure consistent behavior across all usages.",
            False,
        )

    def _classify_small_pattern(
        self, code: str, all_test: bool, is_block: bool
    ) -> DuplicateClassification | None:
        """Classify small code patterns that are often idiomatic."""
        # Check each idiomatic pattern
        for spec in self.IDIOMATIC_PATTERNS:
            if spec.requires_block and not is_block:
                continue
            if spec.pattern.search(code):
                return self._make_classification(
                    spec.category,
                    spec.confidence,
                    spec.reason,
                    spec.recommendation,
                    True,
                )

        # Small test setup
        if all_test:
            return self._make_classification(
                DuplicateCategory.TEST_SETUP,
                0.85,
                "Small test setup pattern",
                "Small test setup blocks are often duplicated intentionally. "
                "Self-contained tests are easier to understand and debug.",
                True,
            )

        return None

    def _is_test_file(self, file_path: str) -> bool:
        """Check if a file path indicates a test file."""
        path_lower = file_path.lower()
        return any(pattern in path_lower for pattern in self.TEST_PATTERNS)

    def _is_delegate_method(self, code: str, entries: list[IndexEntry]) -> bool:
        """Check if code is a thin wrapper delegating to another method."""
        # Short methods that call another method and return its result
        if len(entries) < 2:
            return False

        # Check if all are methods calling a common implementation
        lines = [line.strip() for line in code.split("\n") if line.strip()]
        non_def_lines = [
            ln for ln in lines if not ln.startswith("def ") and not ln.startswith('"""')
        ]

        # Delegate: 1-2 lines, contains "return self." or "return _"
        if len(non_def_lines) <= 2:
            code_body = " ".join(non_def_lines)
            if "return self._" in code_body or "return self." in code_body:
                return True

        return False

    def _make_classification(
        self,
        category: DuplicateCategory,
        confidence: float,
        reason: str,
        recommendation: str,
        suppress_suggestion: bool,
    ) -> DuplicateClassification:
        """Create a classification result."""
        return DuplicateClassification(
            category=category,
            confidence=confidence,
            reason=reason,
            recommendation=recommendation,
            suppress_suggestion=suppress_suggestion,
        )


class ToolResult:
    """Result from a tool invocation."""

    def __init__(self, text: str) -> None:
        self.text = text


class CodeStructureTools:
    """
    Simplified code structure analysis tools.

    4 core tools with sensible defaults - no configuration needed.

    Supports two modes:
    - Standard mode (default): JSON persistence, manual re-indexing
    - Event-driven mode: SQLite persistence, file watching, cached analysis
    """

    # Internal defaults - not exposed to MCP users
    _MIN_STATEMENTS = 25  # Filter trivial code (~6+ lines of Python)
    _CHECK_MIN_STATEMENTS = 10  # Lower threshold for pre-creation checks

    def __init__(
        self,
        index: CodeStructureIndex | None = None,
        event_driven: bool = False,
    ) -> None:
        """
        Initialize the tools.

        Args:
            index: Optional pre-existing index (ignored if event_driven=True)
            event_driven: Enable event-driven mode with file watching and
                         SQLite persistence. Provides instant analyze() responses.
        """
        self._event_driven_mode = event_driven
        self._event_driven_index: "EventDrivenIndex | None" = None

        if event_driven:
            # Lazy import to avoid circular dependency
            from .event_driven import EventDrivenIndex

            self._event_driven_index = EventDrivenIndex(
                persistence_path=None,  # Set during index_codebase
                watch_enabled=True,
            )
            self.index = self._event_driven_index.index
        else:
            self.index = index or CodeStructureIndex()

        self._classifier = PatternClassifier()
        self._last_indexed_path: str | None = None

    def _require_index(self) -> ToolResult | None:
        """Return error result if index is empty, None if populated."""
        if not self.index.entries:
            return ToolResult("No code indexed. Call index_codebase first.")
        return None

    def _check_invalidated_suppressions(self) -> str:
        """
        Check for and report any suppressions invalidated by code changes.

        This is called at the start of every tool interaction to proactively
        notify the agent when previously-suppressed duplicates need re-evaluation.

        Returns:
            Warning string if suppressions were invalidated, empty string otherwise.
        """
        if not self.index.entries:
            return ""

        invalidated = self.index.invalidate_stale_suppressions()
        if not invalidated:
            return ""

        lines = [
            "[!] SUPPRESSION(S) INVALIDATED - Code has changed since suppression:",
        ]
        for wl_hash, reason in invalidated:
            lines.append(f"    - {wl_hash}: {reason}")
        lines.append("    These duplicates may need re-evaluation. Run analyze() to review.\n")

        return "\n".join(lines)

    def _format_duplicate_warnings(
        self, result_parts: list[str], dup_count: int, block_dup_count: int, verbose: bool = True
    ) -> int:
        """Format duplicate warnings and append to result_parts. Returns total duplicates."""
        total_dups = dup_count + block_dup_count
        if total_dups > 0:
            result_parts.append(f"\n\n[!] ATTENTION: {total_dups} duplicate groups detected!")
            for count, label in [(dup_count, "function-level"), (block_dup_count, "block-level")]:
                if count > 0:
                    result_parts.append(f"    - {count} {label} duplicates")
            result_parts.append("\n    Run analyze() NOW to see details and take action.")
            if verbose:
                result_parts.append(
                    "    Every duplicate is a maintenance burden waiting to cause bugs."
                )
        else:
            result_parts.append("\nNo duplicates detected - codebase is clean!")
        return total_dups

    def _format_index_stats(self, include_blocks: bool, incremental_info: str = "") -> str:
        """Format index statistics for output."""
        stats = self.index.get_stats()
        result_parts = [
            f"Indexed {stats['function_entries']} code units from {stats['indexed_files']} files{incremental_info}.",
        ]
        if include_blocks:
            result_parts.append(f"Extracted {stats['block_entries']} code blocks.")

        dup_count = stats["duplicate_groups"]
        block_dup_count = stats["block_duplicate_groups"] if include_blocks else 0
        total_dups = self._format_duplicate_warnings(result_parts, dup_count, block_dup_count)

        return "\n".join(result_parts) if total_dups > 0 else " ".join(result_parts)

    def index_codebase(
        self,
        path: str,
        recursive: bool = True,
        incremental: bool = True,
    ) -> ToolResult:
        """
        Index a Python codebase for structural analysis.

        Automatically extracts functions, classes, methods, and code blocks
        (for, while, if, try, with) for comprehensive duplicate detection.

        Index and suppressions are persisted to `.metadata_astograph/` in the
        indexed directory. Add to `.gitignore` if desired.

        In event-driven mode, also starts file watching for automatic updates.

        Args:
            path: Path to file or directory to index
            recursive: Search directories recursively (default True)
            incremental: Only re-index changed files (default True, 98% faster)
        """
        # Resolve path (handles Docker volume mounts)
        path = _resolve_docker_path(path)

        if not os.path.exists(path):
            return ToolResult(f"Error: Path does not exist: {path}")

        # Store path for auto-reindex in analyze()
        self._last_indexed_path = path

        # Always include blocks - only 22% overhead for much better detection
        include_blocks = True

        # Event-driven mode: use SQLite persistence and file watching
        if self._event_driven_mode and self._event_driven_index is not None:
            return self._index_codebase_event_driven(path, recursive)

        # Standard mode: JSON persistence
        return self._index_codebase_standard(path, recursive, incremental, include_blocks)

    def _index_codebase_event_driven(self, path: str, recursive: bool) -> ToolResult:
        """Index codebase using event-driven mode with SQLite and file watching."""
        from .cloud_detect import get_cloud_sync_warning
        from .event_driven import EventDrivenIndex

        # Check for cloud-synced folders and prepare warning
        cloud_warning = get_cloud_sync_warning(path)

        persistence_path = _get_persistence_path(path)
        persistence_path.mkdir(exist_ok=True)
        sqlite_path = persistence_path / "index.db"

        # Create new event-driven index with persistence
        self._event_driven_index = EventDrivenIndex(
            persistence_path=sqlite_path,
            watch_enabled=True,
        )
        self.index = self._event_driven_index.index

        # Index the directory (loads from cache if available)
        self._event_driven_index.index_directory(path, recursive=recursive)

        stats = self.index.get_stats()
        ed_stats = self._event_driven_index.get_stats()

        result_parts = [
            f"Indexed {stats['function_entries']} code units from {stats['indexed_files']} files.",
            f"Extracted {stats['block_entries']} code blocks.",
        ]

        # Event-driven mode info
        if ed_stats.get("watching"):
            result_parts.append("\n[EVENT-DRIVEN MODE ACTIVE]")
            result_parts.append("  - File watching: enabled")
            result_parts.append(f"  - SQLite persistence: {sqlite_path}")
            result_parts.append(
                f"  - Analysis cache: {'ready' if ed_stats.get('cache_valid') else 'computing...'}"
            )

        # Duplicate warnings (reuse helper, verbose=False for event-driven mode)
        dup_count = stats["duplicate_groups"]
        block_dup_count = stats["block_duplicate_groups"]
        self._format_duplicate_warnings(result_parts, dup_count, block_dup_count, verbose=False)

        # Prepend cloud warning if detected
        output = "\n".join(result_parts)
        if cloud_warning:
            output = cloud_warning + "\n\n" + output

        return ToolResult(output)

    def _index_codebase_standard(
        self, path: str, recursive: bool, incremental: bool, include_blocks: bool
    ) -> ToolResult:
        """Index codebase using standard JSON persistence mode."""
        # Set up persistence path
        persistence_path = _get_persistence_path(path)
        index_file = persistence_path / "index.json"

        # Try to load cached index if it exists and we don't have one yet
        loaded_from_cache = False
        if index_file.exists() and not self.index.entries:
            try:
                self.index.load(str(index_file))
                self.index.set_persistence_path(persistence_path)
                loaded_from_cache = True
            except (OSError, ValueError, KeyError):
                # Corrupted cache, ignore and re-index
                pass

        # Create persistence directory if needed
        persistence_path.mkdir(exist_ok=True)
        self.index.set_persistence_path(persistence_path)

        if incremental and os.path.isdir(path) and self.index.file_metadata:
            # Incremental indexing for directories (when index already exists)
            entries, added, updated, unchanged = self.index.index_directory_incremental(
                path, recursive=recursive, include_blocks=include_blocks
            )
            incremental_info = f" ({added} added, {updated} updated, {unchanged} unchanged)"
            if loaded_from_cache:
                incremental_info += " [loaded from cache]"
            self.index._auto_save()
            return ToolResult(self._format_index_stats(include_blocks, incremental_info))

        # Full re-index (clear and rebuild)
        self.index.clear()

        if os.path.isfile(path):
            self.index.index_file(path, include_blocks=include_blocks)
        else:
            self.index.index_directory(path, recursive=recursive, include_blocks=include_blocks)

        self.index._auto_save()
        return ToolResult(self._format_index_stats(include_blocks))

    def _verify_group(self, group: DuplicateGroup) -> bool:
        """Verify a duplicate group via graph isomorphism."""
        if len(group.entries) >= 2:
            return self.index.verify_isomorphism(group.entries[0], group.entries[1])
        return False

    def _format_locations(self, entries: list[IndexEntry]) -> list[str]:
        """Format entry locations for output."""
        return [
            f"{e.code_unit.file_path}:{e.code_unit.name}:L{e.code_unit.line_start}-{e.code_unit.line_end}"
            for e in entries
        ]

    def analyze(self, thorough: bool = True, auto_reindex: bool = True) -> ToolResult:
        """
        Analyze the indexed codebase for duplicates and similar patterns.

        Args:
            thorough: If True, show ALL duplicates including small ones (~2+ lines).
                     If False, show only significant duplicates (~6+ lines).
                     Default True - overhead is only 18%.
            auto_reindex: If True and index is stale, automatically re-index before
                         analyzing. Default True for convenience.

        Returns findings sorted by relevance:
        - Exact duplicates (verified via graph isomorphism)
        - Pattern duplicates (same structure, different operators)
        - Block duplicates (duplicate for/while/if/try/with blocks within functions)
        """
        if error := self._require_index():
            return error

        # Check for invalidated suppressions (proactive notification)
        invalidation_warning = self._check_invalidated_suppressions()

        # Check for staleness and auto-reindex if enabled
        staleness_warning = ""
        report = self.index.get_staleness_report()
        if report.is_stale:
            if auto_reindex and self._last_indexed_path:
                # Auto-reindex for accurate results
                self.index_codebase(self._last_indexed_path, incremental=True)
                staleness_warning = "[Auto-reindexed for accurate results]\n\n"
            else:
                counts = [
                    f"{len(items)} {label}"
                    for items, label in [
                        (report.modified_files, "modified"),
                        (report.deleted_files, "deleted"),
                        (report.stale_suppressions, "stale suppressions"),
                    ]
                    if items
                ]
                staleness_warning = (
                    f"WARNING: Index may be stale. {', '.join(counts)} detected.\n"
                    "Consider re-indexing for accurate results.\n\n"
                )

        findings: list[dict[str, Any]] = []

        # Use lower threshold in thorough mode to catch small duplicates
        min_nodes = 5 if thorough else self._MIN_STATEMENTS

        # Find exact duplicates
        groups = self.index.find_all_duplicates(min_node_count=min_nodes)
        for group in groups:
            locations = self._format_locations(group.entries)
            classification = self._classifier.classify_group(group)

            # Determine keep suggestion based on path depth
            keep = None
            keep_reason = None
            depths = [(e.code_unit.file_path.count("/"), e) for e in group.entries]
            depths.sort(key=lambda x: x[0])
            if len(depths) >= 2 and depths[0][0] < depths[1][0]:
                e = depths[0][1]
                keep = f"{e.code_unit.file_path}:{e.code_unit.name}:L{e.code_unit.line_start}-{e.code_unit.line_end}"
                keep_reason = "shallowest path"

            findings.append(
                {
                    "type": "exact",
                    "hash": group.wl_hash,
                    "verified": self._verify_group(group),
                    "locations": locations,
                    "keep": keep,
                    "keep_reason": keep_reason,
                    "classification": classification,
                }
            )

        # Find pattern duplicates (same structure, different operators)
        pattern_groups = self.index.find_pattern_duplicates(min_node_count=min_nodes)
        for group in pattern_groups:
            classification = self._classifier.classify_group(group)
            findings.append(
                {
                    "type": "pattern",
                    "hash": group.wl_hash,
                    "locations": self._format_locations(group.entries),
                    "classification": classification,
                }
            )

        # Find block duplicates (duplicate code blocks within functions)
        block_groups = self.index.find_block_duplicates(min_node_count=min_nodes)
        for group in block_groups:
            block_type = group.entries[0].code_unit.block_type or "block"
            parent_funcs = list(
                {e.code_unit.parent_name for e in group.entries if e.code_unit.parent_name}
            )
            classification = self._classifier.classify_group(group)

            findings.append(
                {
                    "type": "block",
                    "hash": group.wl_hash,
                    "block_type": block_type,
                    "verified": self._verify_group(group),
                    "locations": self._format_locations(group.entries),
                    "parent_funcs": parent_funcs,
                    "classification": classification,
                }
            )

        # Count hidden duplicates (filtered by min_node_count)
        stats = self.index.get_stats()
        total_dup_groups = stats["duplicate_groups"]
        total_block_groups = stats["block_duplicate_groups"]
        shown_exact = sum(1 for f in findings if f["type"] == "exact")
        shown_block = sum(1 for f in findings if f["type"] == "block")
        hidden_exact = total_dup_groups - shown_exact
        hidden_block = total_block_groups - shown_block
        suppressed_count = stats["suppressed_hashes"]

        if not findings:
            # Still warn about hidden/suppressed duplicates
            hidden_msg = ""
            if hidden_exact > 0 or hidden_block > 0:
                if thorough:
                    hidden_msg = f"\n\nAll {hidden_exact + hidden_block} duplicate groups are below minimum threshold (~2 lines) - these are trivial and can be ignored."
                else:
                    hidden_msg = f"\n\n[!] {hidden_exact + hidden_block} small duplicates hidden (< ~6 lines each). Run analyze(thorough=True) to see ALL duplicates and take action."
            if suppressed_count > 0:
                hidden_msg += f"\n{suppressed_count} duplicate groups are suppressed. Run list_suppressions to review."

            # Show clean status when no pending work
            if (hidden_exact + hidden_block == 0 or thorough) and suppressed_count >= 0:
                hidden_msg += "\n\n[CLEAN] All duplicates have been addressed!"

            return ToolResult(
                invalidation_warning
                + staleness_warning
                + "No significant duplicates found."
                + hidden_msg
            )

        # Separate findings by whether they need action or are likely idiomatic
        action_needed = [f for f in findings if not f["classification"].suppress_suggestion]
        likely_idiomatic = [f for f in findings if f["classification"].suppress_suggestion]

        lines = [f"{'='*60}"]

        if action_needed:
            lines.extend(
                [
                    "  REFACTORING OPPORTUNITIES FOUND",
                    f"{'='*60}",
                    "",
                    f"Found {len(action_needed)} duplicate(s) worth addressing:",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "  DUPLICATE ANALYSIS COMPLETE",
                    f"{'='*60}",
                    "",
                ]
            )

        # Show action-needed findings first (these are true duplicates)
        for i, f in enumerate(action_needed, 1):
            locs = ", ".join(f["locations"])
            wl_hash = f.get("hash", "")
            classification = f["classification"]
            verified = " (verified)" if f.get("verified") else ""

            if f["type"] == "exact":
                lines.append(f"{i}. [REFACTOR]{verified} {locs}")
            elif f["type"] == "block":
                block_type = f.get("block_type", "block")
                lines.append(f"{i}. [REFACTOR: {block_type}]{verified} {locs}")
                if f.get("parent_funcs"):
                    lines.append(f"   Found in: {', '.join(f['parent_funcs'])}")
            else:
                lines.append(f"{i}. [SIMILAR STRUCTURE] {locs}")

            # Context-aware recommendation from classifier
            lines.append(f"   {classification.recommendation}")
            if f.get("keep"):
                lines.append(f"   Suggestion: Keep {f['keep']} ({f['keep_reason']})")
            lines.append(f'   suppress(wl_hash="{wl_hash}")')
            lines.append("")

        # Show idiomatic/test patterns with explanations (collapsed by default in non-thorough)
        if likely_idiomatic:
            if action_needed:
                lines.append(f"{'-'*60}")
            lines.append(f"  IDIOMATIC PATTERNS ({len(likely_idiomatic)} detected)")
            lines.append("  These are common Python patterns that may not need refactoring:")
            lines.append("")

            for j, f in enumerate(likely_idiomatic, len(action_needed) + 1):
                locs = ", ".join(f["locations"])
                wl_hash = f.get("hash", "")
                classification = f["classification"]

                # Category-based label
                cat = classification.category
                if cat == DuplicateCategory.IDIOMATIC_GUARD:
                    label = "GUARD CLAUSE"
                elif cat == DuplicateCategory.IDIOMATIC_DICT_BUILD:
                    label = "DICT BUILD"
                elif cat == DuplicateCategory.TEST_SETUP:
                    label = "TEST SETUP"
                elif cat == DuplicateCategory.DELEGATE_METHOD:
                    label = "DELEGATE"
                else:
                    label = "PATTERN"

                if f["type"] == "block":
                    block_type = f.get("block_type", "block")
                    lines.append(f"{j}. [{label}: {block_type}] {locs}")
                else:
                    lines.append(f"{j}. [{label}] {locs}")

                # Show classification reason
                lines.append(f"   Why: {classification.reason}")
                lines.append(f"   {classification.recommendation}")
                lines.append(f'   suppress(wl_hash="{wl_hash}")')
                lines.append("")

        # Footer
        lines.append(f"{'='*60}")
        if hidden_exact > 0 or hidden_block > 0:
            if thorough:
                lines.append(
                    f"  + {hidden_exact + hidden_block} trivial duplicates not shown (< ~2 lines)"
                )
            else:
                lines.append(
                    f"  + {hidden_exact + hidden_block} small duplicates hidden (< ~6 lines)"
                )
                lines.append("    Run analyze(thorough=True) to see all patterns.")
        if suppressed_count > 0:
            lines.append(
                f"  + {suppressed_count} groups suppressed (run list_suppressions to review)"
            )

        # Summary guidance with strong nudge to complete all items
        total_pending = len(action_needed) + len(likely_idiomatic)
        if action_needed:
            lines.append("")
            lines.append(f"  {len(action_needed)} finding(s) warrant refactoring.")
            lines.append("  Address these to reduce maintenance burden.")
        if likely_idiomatic:
            lines.append("")
            if not action_needed:
                lines.append("  All detected duplicates appear to be idiomatic patterns.")
            else:
                lines.append(f"  {len(likely_idiomatic)} idiomatic pattern(s) can be suppressed.")
            lines.append("  Suppress them to clean up future analysis runs.")

        # Strong nudge to handle ALL pending duplicates
        if total_pending > 0:
            lines.append("")
            lines.append(f"  [ACTION REQUIRED] {total_pending} duplicate group(s) pending.")
            lines.append("  Continue until all are refactored or suppressed for a clean codebase.")
        else:
            lines.append("")
            lines.append("  [CLEAN] All duplicates have been addressed!")

        lines.append(f"{'='*60}")

        return ToolResult(invalidation_warning + staleness_warning + "\n".join(lines))

    def check(self, code: str) -> ToolResult:
        """
        Check if code similar to the provided snippet exists.

        Use this BEFORE creating new code to avoid duplication.
        """
        if error := self._require_index():
            return error

        # Proactive notification of invalidated suppressions
        prefix = self._check_invalidated_suppressions()

        results = self.index.find_similar(code, min_node_count=self._CHECK_MIN_STATEMENTS)

        if not results:
            return ToolResult(prefix + "No similar code found. Safe to proceed.")

        exact = [r for r in results if r.similarity_type == "exact"]
        high = [r for r in results if r.similarity_type == "high"]
        partial = [r for r in results if r.similarity_type == "partial"]

        if exact:
            entry = exact[0].entry
            return ToolResult(
                prefix
                + f"STOP: Identical code exists at {entry.code_unit.file_path}:{entry.code_unit.name} "
                f"(lines {entry.code_unit.line_start}-{entry.code_unit.line_end}). Reuse it."
            )
        elif high:
            entry = high[0].entry
            return ToolResult(
                prefix
                + f"CAUTION: Very similar code at {entry.code_unit.file_path}:{entry.code_unit.name}. "
                f"Consider reusing or extending."
            )
        elif partial:
            entry = partial[0].entry
            return ToolResult(
                prefix
                + f"NOTE: Partially similar code at {entry.code_unit.file_path}:{entry.code_unit.name}. "
                f"Review for potential reuse."
            )

        return ToolResult(prefix + "No similar code found. Safe to proceed.")

    def compare(self, code1: str, code2: str) -> ToolResult:
        """Compare two code snippets for structural equivalence."""
        g1 = ast_to_graph(code1)
        g2 = ast_to_graph(code2)

        h1 = weisfeiler_leman_hash(g1)
        h2 = weisfeiler_leman_hash(g2)

        is_isomorphic = nx.is_isomorphic(g1, g2, node_match=node_match)

        if is_isomorphic:
            return ToolResult("EQUIVALENT: The code snippets are structurally identical.")
        elif h1 == h2:
            return ToolResult("SIMILAR: Same hash but not fully isomorphic (rare edge case).")
        elif fingerprints_compatible(structural_fingerprint(g1), structural_fingerprint(g2)):
            return ToolResult("SIMILAR: Compatible structure but not identical.")
        else:
            return ToolResult("DIFFERENT: The code snippets are structurally different.")

    def _toggle_suppression(self, wl_hash: str, suppress: bool) -> ToolResult:
        """Toggle hash suppression status with index check."""
        if error := self._require_index():
            return error

        # Proactive notification of invalidated suppressions
        prefix = self._check_invalidated_suppressions()

        # Toggle suppression state using appropriate index
        active_index: CodeStructureIndex | EventDrivenIndex = (
            self._event_driven_index
            if self._event_driven_mode and self._event_driven_index
            else self.index
        )
        if suppress:
            success, linked_hashes = active_index.suppress(wl_hash)
        else:
            success, linked_hashes = active_index.unsuppress(wl_hash), []

        if suppress:
            success_msg = f"Suppressed hash {wl_hash}. It will no longer appear in analyze results."
            if linked_hashes:
                success_msg += f"\n\n  Also suppressed {len(linked_hashes)} linked duplicate(s):"
                for linked in linked_hashes:
                    success_msg += f"\n    - {linked}"
            failure_msg = f"Hash {wl_hash} not found in index."
        else:
            success_msg = f"Unsuppressed hash {wl_hash}. It will appear in analyze results again."
            failure_msg = f"Hash {wl_hash} was not suppressed."

        return ToolResult(prefix + (success_msg if success else failure_msg))

    def suppress(self, wl_hash: str) -> ToolResult:
        """
        Suppress a duplicate group by its WL hash.

        Use this to mute idiomatic patterns or acceptable duplications
        that don't need to be refactored.
        """
        return self._toggle_suppression(wl_hash, suppress=True)

    def unsuppress(self, wl_hash: str) -> ToolResult:
        """Remove a hash from the suppressed set."""
        return self._toggle_suppression(wl_hash, suppress=False)

    def list_suppressions(self) -> ToolResult:
        """List all suppressed hashes."""
        prefix = self._check_invalidated_suppressions()
        suppressed = self.index.get_suppressed()
        if not suppressed:
            return ToolResult(prefix + "No hashes are currently suppressed.")
        return ToolResult(
            prefix + f"Suppressed hashes ({len(suppressed)}):\n" + "\n".join(suppressed)
        )

    def suppress_idiomatic(self) -> ToolResult:
        """
        Suppress all idiomatic patterns at once.

        Convenience method to quickly suppress all patterns classified as idiomatic
        (guard clauses, test setup, delegate methods, dict building, etc.).
        """
        if not self.index.entries:
            return ToolResult("No code indexed. Use index_codebase first.")

        # Find all idiomatic patterns from both function and block duplicates
        suppressed = self.index.get_suppressed()
        idiomatic_hashes: list[str] = []

        # Check both function and block duplicates in a single pass
        all_groups = itertools.chain(
            self.index.find_all_duplicates(),
            self.index.find_block_duplicates(),
        )
        for group in all_groups:
            classification = self._classifier.classify_group(group)
            if classification.suppress_suggestion:
                wl_hash = group.wl_hash
                if wl_hash and wl_hash not in suppressed:
                    idiomatic_hashes.append(wl_hash)

        if not idiomatic_hashes:
            return ToolResult("No idiomatic patterns found to suppress.")

        # Suppress all idiomatic hashes
        suppressed_count = 0
        for wl_hash in idiomatic_hashes:
            if self._event_driven_mode and self._event_driven_index is not None:
                success, _ = self._event_driven_index.suppress(wl_hash)
            else:
                success, _ = self.index.suppress(wl_hash)
            if success:
                suppressed_count += 1

        return ToolResult(
            f"Suppressed {suppressed_count} idiomatic pattern(s).\n"
            f"These will no longer appear in analyze results."
        )

    def _format_file_list(self, files: list[str], label: str, max_items: int = 10) -> list[str]:
        """Format a list of files for output, with truncation."""
        if not files:
            return []
        lines = [f"\n{label} ({len(files)}):"]
        for f in files[:max_items]:
            lines.append(f"  - {f}")
        if len(files) > max_items:
            lines.append(f"  ... and {len(files) - max_items} more")
        return lines

    def check_staleness(self, path: str | None = None) -> ToolResult:
        """
        Check if the code index is stale (files changed since indexing).

        Args:
            path: Optional root path to also check for new files.
        """
        if error := self._require_index():
            return error

        # Proactive notification of invalidated suppressions
        prefix = self._check_invalidated_suppressions()

        # Resolve path (handles Docker volume mounts)
        if path is not None:
            path = _resolve_docker_path(path)

        report = self.index.get_staleness_report(path)

        if not report.is_stale:
            return ToolResult(prefix + "Index is up to date. No changes detected.")

        lines = ["Index is STALE. Changes detected:"]
        lines.extend(self._format_file_list(report.modified_files, "Modified"))
        lines.extend(self._format_file_list(report.deleted_files, "Deleted"))
        lines.extend(self._format_file_list(report.new_files, "New files"))

        if report.stale_suppressions:
            lines.append(f"\nStale suppressions ({len(report.stale_suppressions)}):")
            for s in report.stale_suppressions:
                lines.append(f"  - {s}")

        lines.append("\nConsider re-indexing with index_codebase for accurate results.")

        return ToolResult(prefix + "\n".join(lines))

    def write(self, file_path: str, content: str) -> ToolResult:
        """
        Write Python code to a file with automatic duplicate detection.

        Checks the content for structural duplicates before writing.
        Blocks if identical code exists elsewhere; warns on high similarity.
        """
        if error := self._require_index():
            return error

        # Check for duplicates in the content
        results = self.index.find_similar(content, min_node_count=self._CHECK_MIN_STATEMENTS)

        exact = [r for r in results if r.similarity_type == "exact"]
        high = [r for r in results if r.similarity_type == "high"]

        if exact:
            entry = exact[0].entry
            return ToolResult(
                f"BLOCKED: Cannot write - identical code exists at "
                f"{entry.code_unit.file_path}:{entry.code_unit.name} "
                f"(lines {entry.code_unit.line_start}-{entry.code_unit.line_end}). "
                f"Reuse the existing implementation instead."
            )

        warning = ""
        if high:
            entry = high[0].entry
            warning = (
                f"WARNING: Similar code exists at {entry.code_unit.file_path}:{entry.code_unit.name}. "
                f"Consider reusing. Proceeding with write.\n\n"
            )

        return self._write_file(file_path, content, "wrote", warning)

    def _write_file(
        self, file_path: str, content: str, action: str, prefix: str = ""
    ) -> ToolResult:
        """Write content to a file with error handling."""
        try:
            with open(file_path, "w") as f:
                f.write(content)
            return ToolResult(prefix + f"Successfully {action} {file_path}")
        except OSError as e:
            return ToolResult(f"Failed to write {file_path}: {e}")

    def edit(self, file_path: str, old_string: str, new_string: str) -> ToolResult:
        """
        Edit a Python file with automatic duplicate detection.

        Checks the new_string for structural duplicates before applying.
        Blocks if identical code exists elsewhere; warns on high similarity.
        """
        if error := self._require_index():
            return error

        # Check for duplicates in the new code being introduced
        results = self.index.find_similar(new_string, min_node_count=self._CHECK_MIN_STATEMENTS)

        exact = [r for r in results if r.similarity_type == "exact"]
        high = [r for r in results if r.similarity_type == "high"]

        warning = ""
        if exact:
            entry = exact[0].entry
            if entry.code_unit.file_path != file_path:
                # Cross-file duplicate: block
                return ToolResult(
                    f"BLOCKED: Cannot edit - identical code exists at "
                    f"{entry.code_unit.file_path}:{entry.code_unit.name} "
                    f"(lines {entry.code_unit.line_start}-{entry.code_unit.line_end}). "
                    f"Reuse the existing implementation instead."
                )
            else:
                # Same-file duplicate: warn
                warning = (
                    f"WARNING: Identical code exists in same file at {entry.code_unit.name} "
                    f"(lines {entry.code_unit.line_start}-{entry.code_unit.line_end}). "
                    f"Consider reusing. Proceeding with edit.\n\n"
                )
        elif high:
            entry = high[0].entry
            warning = (
                f"WARNING: Similar code exists at {entry.code_unit.file_path}:{entry.code_unit.name}. "
                f"Consider reusing. Proceeding with edit.\n\n"
            )

        # Read the file
        try:
            with open(file_path) as f:
                content = f.read()
        except FileNotFoundError:
            return ToolResult(f"File not found: {file_path}")
        except OSError as e:
            return ToolResult(f"Failed to read {file_path}: {e}")

        # Check that old_string exists
        if old_string not in content:
            return ToolResult(
                f"Edit failed: old_string not found in {file_path}. " f"The file may have changed."
            )

        # Check for uniqueness
        count = content.count(old_string)
        if count > 1:
            return ToolResult(
                f"Edit failed: old_string appears {count} times in {file_path}. "
                f"Provide more context to make it unique."
            )

        # Apply the edit
        new_content = content.replace(old_string, new_string, 1)

        return self._write_file(file_path, new_content, "edited", warning)

    def call_tool(self, name: str, arguments: dict) -> ToolResult:
        """Dispatch a tool call by name."""
        if name == "index_codebase":
            return self.index_codebase(
                path=arguments["path"],
                recursive=arguments.get("recursive", True),
                incremental=arguments.get("incremental", True),
            )
        elif name == "analyze":
            return self.analyze(
                thorough=arguments.get("thorough", True),
                auto_reindex=arguments.get("auto_reindex", True),
            )
        elif name == "check":
            return self.check(code=arguments["code"])
        elif name == "compare":
            return self.compare(
                code1=arguments["code1"],
                code2=arguments["code2"],
            )
        elif name == "suppress":
            return self.suppress(wl_hash=arguments["wl_hash"])
        elif name == "unsuppress":
            return self.unsuppress(wl_hash=arguments["wl_hash"])
        elif name == "list_suppressions":
            return self.list_suppressions()
        elif name == "suppress_idiomatic":
            return self.suppress_idiomatic()
        elif name == "check_staleness":
            return self.check_staleness(path=arguments.get("path"))
        elif name == "write":
            return self.write(
                file_path=arguments["file_path"],
                content=arguments["content"],
            )
        elif name == "edit":
            return self.edit(
                file_path=arguments["file_path"],
                old_string=arguments["old_string"],
                new_string=arguments["new_string"],
            )
        else:
            return ToolResult(f"Unknown tool: {name}")

    def close(self) -> None:
        """Clean up resources (file watchers, database connections)."""
        if self._event_driven_index is not None:
            self._event_driven_index.close()
            self._event_driven_index = None

    def __enter__(self) -> "CodeStructureTools":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get_event_driven_stats(self) -> dict | None:
        """Get event-driven mode statistics (returns None if not in event-driven mode)."""
        if self._event_driven_index is not None:
            return self._event_driven_index.get_stats()
        return None
