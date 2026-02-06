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

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx

from .ast_to_graph import ast_to_graph, node_match
from .canonical_hash import (
    fingerprints_compatible,
    structural_fingerprint,
    weisfeiler_leman_hash,
)
from .index import CodeStructureIndex, DuplicateGroup, IndexEntry
from .languages.registry import LanguageRegistry

if TYPE_CHECKING:
    from .event_driven import EventDrivenIndex

logger = logging.getLogger(__name__)


# Persistence directory name for cached index
PERSISTENCE_DIR = ".metadata_astrograph"


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

    # In Docker, /workspace might be read-only but /workspace/.metadata_astrograph
    # has a tmpfs mount. If we're indexing a subdirectory, use the root.
    workspace = Path("/workspace")
    if workspace.exists() and Path("/.dockerenv").exists() and str(base).startswith("/workspace/"):
        # Use workspace root for persistence (has tmpfs mount)
        persistence_path = workspace / PERSISTENCE_DIR

    return persistence_path


def _get_sqlite_path(indexed_path: str) -> Path:
    """Get SQLite database path for an indexed codebase."""
    return _get_persistence_path(indexed_path) / "index.db"


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
        self._event_driven_index: EventDrivenIndex | None = None

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

        self._last_indexed_path: str | None = None

        # Background indexing state
        self._bg_index_thread: threading.Thread | None = None
        self._bg_index_done = threading.Event()
        self._bg_index_done.set()  # Initially "done" (no background work)

        # Auto-index /workspace at startup in event-driven mode (Docker)
        # Run in background so the MCP handshake completes immediately.
        if event_driven and os.path.isdir("/workspace"):
            self._bg_index_done.clear()
            self._bg_index_thread = threading.Thread(
                target=self._background_index,
                args=("/workspace",),
                daemon=True,
            )
            self._bg_index_thread.start()

    def _background_index(self, path: str) -> None:
        """Index a codebase in the background."""
        try:
            self.index_codebase(path)
        except Exception:
            logger.exception(f"Background indexing failed for {path}")
        finally:
            self._bg_index_done.set()

    def _wait_for_background_index(self) -> None:
        """Block until background indexing completes (if running)."""
        timeout = int(os.environ.get("ASTROGRAPH_INDEX_TIMEOUT", "300"))
        if not self._bg_index_done.wait(timeout=timeout):
            logger.warning("Background indexing timed out after %ds", timeout)

    def _require_index(self) -> ToolResult | None:
        """Return error result if index is empty, None if populated."""
        self._wait_for_background_index()
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

        return f"Suppressions invalidated: {', '.join(h for h, _ in invalidated)}. Run analyze().\n"

    def _has_significant_duplicates(self) -> bool:
        """Check if there are duplicates above the trivial threshold."""
        min_nodes = 5
        return bool(
            self.index.find_all_duplicates(min_node_count=min_nodes)
            or self.index.find_block_duplicates(min_node_count=min_nodes)
        )

    def _format_index_stats(self, include_blocks: bool, incremental_info: str = "") -> str:
        """Format index statistics for output."""
        stats = self.index.get_stats()
        result_parts = [
            f"Indexed {stats['function_entries']} code units from {stats['indexed_files']} files{incremental_info}.",
        ]
        if include_blocks:
            result_parts.append(f"Extracted {stats['block_entries']} code blocks.")

        if self._has_significant_duplicates():
            result_parts.append("\nDuplicates found. Run analyze().")
            return "\n".join(result_parts)

        result_parts.append("\nNo duplicates.")
        return " ".join(result_parts)

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

        Index and suppressions are persisted to `.metadata_astrograph/` in the
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

        # Close old event-driven index (stops watcher, closes SQLite connection)
        if self._event_driven_index is not None:
            self._event_driven_index.close()

        # Create new event-driven index with persistence
        self._event_driven_index = EventDrivenIndex(
            persistence_path=sqlite_path,
            watch_enabled=True,
        )
        self.index = self._event_driven_index.index

        # Index the directory (loads from cache if available)
        self._event_driven_index.index_directory(path, recursive=recursive)

        stats = self.index.get_stats()

        result_parts = [
            f"Indexed {stats['function_entries']} code units from {stats['indexed_files']} files.",
            f"Extracted {stats['block_entries']} code blocks.",
        ]

        if self._has_significant_duplicates():
            result_parts.append("\nDuplicates found. Run analyze().")
        else:
            result_parts.append("\nNo duplicates.")

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

    def _write_analysis_report(self, content: str) -> Path | None:
        """Write full analysis report to persistence directory.

        Returns the file path on success, None on failure.
        """
        if self._last_indexed_path is None:
            return None
        try:
            persistence_path = _get_persistence_path(self._last_indexed_path)
            persistence_path.mkdir(exist_ok=True)
            report_file = persistence_path / "analysis_report.txt"
            report_file.write_text(content)
            return report_file
        except OSError:
            return None

    def _clear_analysis_report(self) -> None:
        """Remove stale analysis report file when no findings."""
        if self._last_indexed_path is None:
            return
        try:
            report_file = _get_persistence_path(self._last_indexed_path) / "analysis_report.txt"
            report_file.unlink(missing_ok=True)
        except OSError:
            pass

    def analyze(self, auto_reindex: bool = True) -> ToolResult:
        """
        Analyze the indexed codebase for duplicates and similar patterns.

        Args:
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
                staleness_warning = "[Auto-reindexed]\n"
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
                staleness_warning = f"Stale: {', '.join(counts)}. Re-index recommended.\n"

        findings: list[dict[str, Any]] = []

        min_nodes = 5

        # Find exact duplicates
        groups = self.index.find_all_duplicates(min_node_count=min_nodes)
        for group in groups:
            locations = self._format_locations(group.entries)
            first = group.entries[0]
            line_count = first.code_unit.line_end - first.code_unit.line_start + 1

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
                    "line_count": line_count,
                }
            )

        # Find block duplicates (duplicate code blocks within functions)
        block_groups = self.index.find_block_duplicates(min_node_count=min_nodes)
        for group in block_groups:
            block_type = group.entries[0].code_unit.block_type or "block"
            first = group.entries[0]
            line_count = first.code_unit.line_end - first.code_unit.line_start + 1
            parent_funcs = list(
                {e.code_unit.parent_name for e in group.entries if e.code_unit.parent_name}
            )

            findings.append(
                {
                    "type": "block",
                    "hash": group.wl_hash,
                    "block_type": block_type,
                    "verified": self._verify_group(group),
                    "locations": self._format_locations(group.entries),
                    "parent_funcs": parent_funcs,
                    "line_count": line_count,
                }
            )

        # Find pattern duplicates (same structure, different operators)
        pattern_groups = self.index.find_pattern_duplicates(min_node_count=min_nodes)
        for group in pattern_groups:
            first = group.entries[0]
            line_count = first.code_unit.line_end - first.code_unit.line_start + 1
            findings.append(
                {
                    "type": "pattern",
                    "hash": group.wl_hash,
                    "locations": self._format_locations(group.entries),
                    "line_count": line_count,
                }
            )

        suppressed_count = self.index.get_stats()["suppressed_hashes"]

        if not findings:
            self._clear_analysis_report()
            msg = "No significant duplicates."
            if suppressed_count:
                msg += f" {suppressed_count} suppressed."
            return ToolResult(invalidation_warning + staleness_warning + msg)

        lines: list[str] = []

        for i, f in enumerate(findings, 1):
            locs = ", ".join(f["locations"])
            wl_hash = f.get("hash", "")
            line_count = f.get("line_count", 0)
            verified = " (verified)" if f.get("verified") else ""

            if f["type"] == "exact":
                lines.append(f"{i}.{verified} {locs} ({line_count} lines)")
            elif f["type"] == "block":
                block_type = f.get("block_type", "block")
                parents = ", ".join(f.get("parent_funcs", []))
                parent_info = f", in {parents}" if parents else ""
                lines.append(
                    f"{i}. [{block_type}]{verified} {locs} ({line_count} lines{parent_info})"
                )
            else:
                lines.append(f"{i}. [pattern] {locs} ({line_count} lines)")

            if f.get("keep"):
                lines.append(f"   Keep: {f['keep']} ({f['keep_reason']})")
            lines.append(f'   suppress(wl_hash="{wl_hash}")')
            lines.append("")

        # Compact footer
        if suppressed_count > 0:
            lines.append(f"+ {suppressed_count} suppressed")
        lines.append(f"{len(findings)} duplicate groups.")

        full_output = "\n".join(lines)

        # Write full report to file, return compact summary inline
        report_path = self._write_analysis_report(full_output)
        if report_path is not None:
            # Count type breakdown
            exact_count = sum(1 for f in findings if f["type"] == "exact")
            block_count = sum(1 for f in findings if f["type"] == "block")
            pattern_count = sum(1 for f in findings if f["type"] == "pattern")
            type_parts = []
            if exact_count:
                type_parts.append(f"{exact_count} exact")
            if block_count:
                type_parts.append(f"{block_count} block")
            if pattern_count:
                type_parts.append(f"{pattern_count} pattern")
            summary_parts = [f"Found {len(findings)} duplicate groups: {', '.join(type_parts)}."]
            if suppressed_count > 0:
                summary_parts.append(f"+ {suppressed_count} suppressed.")
            line_count_report = full_output.count("\n") + 1
            summary_parts.append(
                f"Details: {PERSISTENCE_DIR}/analysis_report.txt ({line_count_report} lines)"
            )
            summary_parts.append("Read the file to see locations and suppress commands.")
            return ToolResult(invalidation_warning + staleness_warning + "\n".join(summary_parts))

        # Fallback: file write failed or no indexed path — return full output inline
        return ToolResult(invalidation_warning + staleness_warning + full_output)

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

    def compare(self, code1: str, code2: str, language: str = "python") -> ToolResult:
        """Compare two code snippets for structural equivalence."""
        plugin = LanguageRegistry.get().get_plugin(language)
        g1 = plugin.source_to_graph(code1) if plugin else ast_to_graph(code1)
        g2 = plugin.source_to_graph(code2) if plugin else ast_to_graph(code2)

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

        # Get duplicate locations BEFORE suppressing (for display)
        duplicate_locations: list[str] = []
        if suppress:
            entries = self.index._get_entries_for_hash(wl_hash)
            duplicate_locations = [
                f"{e.code_unit.file_path}:{e.code_unit.name}:L{e.code_unit.line_start}-{e.code_unit.line_end}"
                for e in entries
            ]

        # Toggle suppression state using appropriate index
        active_index: CodeStructureIndex | EventDrivenIndex = (
            self._event_driven_index
            if self._event_driven_mode and self._event_driven_index
            else self.index
        )
        success = active_index.suppress(wl_hash) if suppress else active_index.unsuppress(wl_hash)

        if suppress:
            success_msg = f"Suppressed {wl_hash}."
            if duplicate_locations:
                success_msg += f" {len(duplicate_locations)} locations."
            failure_msg = f"Hash {wl_hash} not found in index."
        else:
            success_msg = f"Unsuppressed {wl_hash}."
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

    def _batch_toggle_suppression(self, wl_hashes: list[str], suppress: bool) -> ToolResult:
        """Batch suppress or unsuppress hashes."""
        if error := self._require_index():
            return error
        prefix = self._check_invalidated_suppressions()
        active_index: CodeStructureIndex | EventDrivenIndex = (
            self._event_driven_index
            if self._event_driven_mode and self._event_driven_index
            else self.index
        )
        action = "suppress" if suppress else "unsuppress"
        method = getattr(active_index, f"{action}_batch")
        changed, not_found = method(wl_hashes)
        parts = []
        if changed:
            label = "Suppressed" if suppress else "Unsuppressed"
            parts.append(f"{label} {len(changed)} hashes.")
        if not_found:
            parts.append(f"{len(not_found)} not found: {', '.join(not_found)}")
        if not parts:
            parts.append("No hashes provided.")
        return ToolResult(prefix + " ".join(parts))

    def suppress_batch(self, wl_hashes: list[str]) -> ToolResult:
        """Suppress multiple duplicate groups by WL hash list."""
        return self._batch_toggle_suppression(wl_hashes, suppress=True)

    def unsuppress_batch(self, wl_hashes: list[str]) -> ToolResult:
        """Unsuppress multiple hashes."""
        return self._batch_toggle_suppression(wl_hashes, suppress=False)

    def list_suppressions(self) -> ToolResult:
        """List all suppressed hashes."""
        prefix = self._check_invalidated_suppressions()
        suppressed = self.index.get_suppressed()
        if not suppressed:
            return ToolResult(prefix + "No hashes are currently suppressed.")
        return ToolResult(
            prefix + f"Suppressed hashes ({len(suppressed)}):\n" + "\n".join(suppressed)
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
            return ToolResult(prefix + "Index up to date.")

        lines = ["Index is STALE. Changes detected:"]
        lines.extend(self._format_file_list(report.modified_files, "Modified"))
        lines.extend(self._format_file_list(report.deleted_files, "Deleted"))
        lines.extend(self._format_file_list(report.new_files, "New files"))

        if report.stale_suppressions:
            lines.append(f"\nStale suppressions ({len(report.stale_suppressions)}):")
            for s in report.stale_suppressions:
                lines.append(f"  - {s}")

        return ToolResult(prefix + "\n".join(lines))

    def _format_edit_diff(
        self, file_content: str, old_string: str, new_string: str, file_path: str
    ) -> str:
        """Generate a diff-style string showing what changed in an edit."""
        context_lines = 1

        file_lines = file_content.split("\n")
        old_lines = old_string.split("\n")
        new_lines = new_string.split("\n")

        # Find where old_string starts in the file
        old_start_offset = file_content.index(old_string)
        start_line = file_content[:old_start_offset].count("\n")

        # Build summary
        removed = len(old_lines)
        added = len(new_lines)
        if removed == 1 and added == 1:
            summary = "  Changed 1 line"
        else:
            parts = []
            if added > removed:
                parts.append(f"Added {added - removed} line{'s' if added - removed != 1 else ''}")
            if removed > added:
                parts.append(f"Removed {removed - added} line{'s' if removed - added != 1 else ''}")
            if not parts:
                summary = f"  Changed {removed} line{'s' if removed != 1 else ''}"
            else:
                summary = "  " + ", removed ".join(parts) if len(parts) > 1 else "  " + parts[0]

        # Build diff lines
        diff = []
        # Context before
        ctx_start = max(0, start_line - context_lines)
        for i in range(ctx_start, start_line):
            diff.append(f"  {i + 1:>4}  {file_lines[i]}")

        # Removed lines (original numbering)
        for i, line in enumerate(old_lines):
            diff.append(f"  {start_line + i + 1:>4} -{line}")

        # Added lines (new numbering)
        for i, line in enumerate(new_lines):
            diff.append(f"  {start_line + i + 1:>4} +{line}")

        # Context after
        end_line = start_line + len(old_lines)
        ctx_end = min(len(file_lines), end_line + context_lines)
        # Adjust context line numbers for the new file
        line_offset = added - removed
        for i in range(end_line, ctx_end):
            diff.append(f"  {i + 1 + line_offset:>4}  {file_lines[i]}")

        return f"Edited {file_path}\n{summary}\n" + "\n".join(diff)

    def write(self, file_path: str, content: str) -> ToolResult:
        """
        Write code to a file with automatic duplicate detection.

        Checks the content for structural duplicates before writing.
        Blocks if identical code exists elsewhere; warns on high similarity.
        """
        if error := self._require_index():
            return error

        # Infer language from file extension
        plugin = LanguageRegistry.get().get_plugin_for_file(file_path)
        language = plugin.language_id if plugin else "python"

        # Check for duplicates in the content
        results = self.index.find_similar(
            content, min_node_count=self._CHECK_MIN_STATEMENTS, language=language
        )

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

        import os.path

        is_new = not os.path.exists(file_path)
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        action = "Created" if is_new else "Wrote"
        summary = f"{action} {file_path} ({line_count} lines)"
        return self._write_file(file_path, content, summary=warning + summary)

    def _write_file(self, file_path: str, content: str, summary: str = "") -> ToolResult:
        """Write content to a file with error handling."""
        try:
            with open(file_path, "w") as f:
                f.write(content)
            return ToolResult(summary or f"Wrote {file_path}")
        except OSError as e:
            return ToolResult(f"Failed to write {file_path}: {e}")

    def edit(self, file_path: str, old_string: str, new_string: str) -> ToolResult:
        """
        Edit a file with automatic duplicate detection.

        Checks the new_string for structural duplicates before applying.
        Blocks if identical code exists elsewhere; warns on high similarity.
        """
        if error := self._require_index():
            return error

        # Infer language from file extension
        plugin = LanguageRegistry.get().get_plugin_for_file(file_path)
        language = plugin.language_id if plugin else "python"

        # Check for duplicates in the new code being introduced
        results = self.index.find_similar(
            new_string, min_node_count=self._CHECK_MIN_STATEMENTS, language=language
        )

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
                f"Edit failed: old_string not found in {file_path}. The file may have changed."
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
        diff = self._format_edit_diff(content, old_string, new_string, file_path)

        return self._write_file(file_path, new_content, summary=warning + diff)

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
        elif name == "suppress_batch":
            return self.suppress_batch(wl_hashes=arguments["wl_hashes"])
        elif name == "unsuppress":
            return self.unsuppress(wl_hash=arguments["wl_hash"])
        elif name == "unsuppress_batch":
            return self.unsuppress_batch(wl_hashes=arguments["wl_hashes"])
        elif name == "list_suppressions":
            return self.list_suppressions()
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
        self._wait_for_background_index()
        if self._event_driven_index is not None:
            self._event_driven_index.close()
            self._event_driven_index = None

    def __enter__(self) -> CodeStructureTools:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def get_event_driven_stats(self) -> dict | None:
        """Get event-driven mode statistics (returns None if not in event-driven mode)."""
        if self._event_driven_index is not None:
            stats = self._event_driven_index.get_stats()
            # Add process memory usage (stdlib, no new deps)
            try:
                import resource

                rusage = resource.getrusage(resource.RUSAGE_SELF)
                stats["process_rss_bytes"] = rusage.ru_maxrss * 1024  # macOS uses KB
            except (ImportError, AttributeError):
                pass
            return stats
        return None
