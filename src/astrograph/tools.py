"""
Tool implementations for code structure analysis.

Simplified API with 4 core tools:
- index_codebase: Index source files
- analyze: Find duplicates and similar patterns
- check: Check if code exists before creating
- compare: Compare two code snippets

Uses SQLite persistence with file watching for automatic index updates.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from collections.abc import Callable, Sequence
from datetime import datetime
from functools import partialmethod, wraps
from pathlib import Path
from typing import Any, cast

import networkx as nx

from .canonical_hash import (
    fingerprints_compatible,
    structural_fingerprint,
    weisfeiler_leman_hash,
)
from .context import CloseOnExitMixin
from .event_driven import EventDrivenIndex
from .index import CodeStructureIndex, DuplicateGroup, IndexEntry
from .languages.base import node_match
from .languages.registry import LanguageRegistry
from .lsp_setup import (
    auto_bind_missing_servers,
    bundled_lsp_specs,
    collect_lsp_statuses,
    format_command,
    get_lsp_spec,
    lsp_bindings_path,
    parse_command,
    probe_command,
    set_lsp_binding,
    unset_lsp_binding,
)

logger = logging.getLogger(__name__)


# Persistence directory name for cached index
PERSISTENCE_DIR = ".metadata_astrograph"
LEGACY_ANALYSIS_REPORT = "analysis_report.txt"


def _requires_index(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    """Guard a tool method so it returns the index-not-ready error when needed."""

    @wraps(func)
    def wrapper(self: CodeStructureTools, *args: Any, **kwargs: Any) -> ToolResult:
        if error := self._require_index():
            return error
        return func(self, *args, **kwargs)

    return wrapper


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


class CodeStructureTools(CloseOnExitMixin):
    """
    Simplified code structure analysis tools.

    4 core tools with sensible defaults - no configuration needed.

    Uses SQLite persistence with file watching for automatic index updates.
    """

    # Internal defaults - not exposed to MCP users
    _MIN_STATEMENTS = 25  # Filter trivial code (~6+ lines of Python)
    _CHECK_MIN_STATEMENTS = 10  # Lower threshold for pre-creation checks
    _MIN_BLOCK_DUPLICATE_NODES = 10  # Reduce noisy tiny if/for block duplicates
    _MIN_BLOCK_DUPLICATE_LINES = 3  # Treat 1-2 line guards as acceptable repetition

    def __init__(
        self,
        index: CodeStructureIndex | None = None,
    ) -> None:
        """
        Initialize the tools.

        Args:
            index: Optional pre-existing index (for testing only).
        """
        self._event_driven_index: EventDrivenIndex | None = None

        if index is not None:
            # Testing path: use provided index directly without event-driven wrapper
            self.index = index
        else:
            self._event_driven_index = EventDrivenIndex(
                persistence_path=None,  # Set during index_codebase
                watch_enabled=True,
            )
            self.index = self._event_driven_index.index

        self._last_indexed_path: str | None = None
        self._host_root: str | None = None

        # Background indexing state
        self._bg_index_thread: threading.Thread | None = None
        self._bg_index_done = threading.Event()
        self._bg_index_done.set()  # Initially "done" (no background work)
        self._bg_index_progress: str = ""

        # Auto-index workspace at startup (Docker / Codex / local MCP clients).
        # Run in background so the MCP handshake completes immediately.
        workspace = self._detect_startup_workspace()
        if workspace and os.path.isdir(workspace):
            self._bg_index_done.clear()
            self._bg_index_thread = threading.Thread(
                target=self._background_index,
                args=(workspace,),
                daemon=True,
            )
            self._bg_index_thread.start()

    def _detect_startup_workspace(self) -> str | None:
        """Pick a safe workspace path to auto-index at startup.

        Detection order:
        1) ``ASTROGRAPH_WORKSPACE`` (explicit override; empty disables auto-index)
        2) ``/workspace`` (Docker convention used by Claude/Cursor setups)
        3) ``PWD`` env var (Codex may preserve this even when cwd is ``/``)
        4) ``os.getcwd()`` (local MCP clients launched in a project directory)
        """
        env_workspace = os.environ.get("ASTROGRAPH_WORKSPACE")
        if env_workspace is not None:
            if env_workspace and os.path.isdir(env_workspace):
                return env_workspace
            return None

        if os.path.isdir("/workspace"):
            return "/workspace"

        for candidate in (os.environ.get("PWD"), os.getcwd()):
            if candidate and candidate != "/" and os.path.isdir(candidate):
                return candidate

        return None

    def _background_index(self, path: str) -> None:
        """Index a codebase in the background."""
        try:
            self._bg_index_progress = "starting"
            self.index_codebase(path)
            self._bg_index_progress = "done"
        except Exception:
            logger.exception(f"Background indexing failed for {path}")
            self._bg_index_progress = "error"
        finally:
            self._bg_index_done.set()

    def _wait_for_background_index(self) -> None:
        """Block until background indexing completes (if running)."""
        timeout = int(os.environ.get("ASTROGRAPH_INDEX_TIMEOUT", "300"))
        if not self._bg_index_done.wait(timeout=timeout):
            logger.warning("Background indexing timed out after %ds", timeout)

    def _require_index(self) -> ToolResult | None:
        """Wait for background indexing to finish, then check readiness.

        Blocks until background indexing completes (with timeout) so that
        tools like analyze/write/edit get a usable index.  The ``status``
        tool has its own non-blocking path for quick readiness checks.
        """
        self._wait_for_background_index()
        return (
            None
            if self.index.entries
            else ToolResult("No code indexed. Call index_codebase first.")
        )

    def _active_index(self) -> CodeStructureIndex | EventDrivenIndex:
        """Return event-driven index when enabled, otherwise in-memory index."""
        return self._event_driven_index if self._event_driven_index else self.index

    def _close_event_driven_index(self) -> None:
        """Close and detach the current event-driven index, if present."""
        if self._event_driven_index:
            self._event_driven_index.close()
            self._event_driven_index = None

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
        if invalidated:
            max_shown = 5
            hashes = [h for h, _ in invalidated]
            shown = self._format_hash_preview(hashes, max_shown=max_shown)
            return f"Suppressions invalidated: {shown}. Run analyze().\n"
        return ""

    @staticmethod
    def _format_hash_preview(hashes: list[str], max_shown: int = 5) -> str:
        """Format a compact preview string for hash lists."""
        if len(hashes) > max_shown:
            return ", ".join(hashes[:max_shown]) + f" ... ({len(hashes)} total)"
        return ", ".join(hashes)

    def _has_significant_duplicates(self) -> bool:
        """Check if there are duplicates above the trivial threshold."""
        return self.index.has_duplicates(min_node_count=5)

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
    ) -> ToolResult:
        """
        Index a Python codebase for structural analysis.

        Automatically extracts functions, classes, methods, and code blocks
        (for, while, if, try, with) for comprehensive duplicate detection.

        Index and suppressions are persisted to `.metadata_astrograph/` in the
        indexed directory. Add to `.gitignore` if desired.

        Also starts file watching for automatic updates on changes.

        Args:
            path: Path to file or directory to index
            recursive: Search directories recursively (default True)
        """
        # Wait for background indexing to avoid racing on shared state
        # (skip if we ARE the background thread to avoid deadlock)
        if threading.current_thread() is not self._bg_index_thread:
            self._wait_for_background_index()

        # Resolve path (handles Docker volume mounts)
        original_path = path
        path = self._resolve_path(path)

        if not os.path.exists(path):
            return ToolResult(f"Error: Path does not exist: {original_path}")

        # Store resolved path for auto-reindex and relative path computation
        self._last_indexed_path = str(Path(path).resolve())

        # Always include blocks - only 22% overhead for much better detection
        return self._index_codebase_event_driven(path, recursive)

    def _index_codebase_event_driven(self, path: str, recursive: bool) -> ToolResult:
        """Index codebase using event-driven mode with SQLite and file watching."""
        from .cloud_detect import get_cloud_sync_warning

        # Check for cloud-synced folders and prepare warning
        cloud_warning = get_cloud_sync_warning(path)

        persistence_path = _get_persistence_path(path)
        persistence_path.mkdir(exist_ok=True)
        sqlite_path = persistence_path / "index.db"

        # Close old event-driven index (stops watcher, closes SQLite connection)
        self._close_event_driven_index()

        # Create new event-driven index with persistence
        self._event_driven_index = EventDrivenIndex(
            persistence_path=sqlite_path,
            watch_enabled=True,
        )
        self.index = self._event_driven_index.index

        # Index the path (loads from cache if available)
        if os.path.isfile(path):
            # Single file: index directly, watch its parent directory
            self.index.index_file(path)
        else:
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

    def _verify_group(self, group: DuplicateGroup) -> bool:
        """Verify a duplicate group via graph isomorphism."""
        if len(group.entries) >= 2:
            return self.index.verify_isomorphism(group.entries[0], group.entries[1])
        return False

    def _relative_path(self, file_path: str) -> str:
        """Strip the indexed root to produce a relative path."""
        root = self._last_indexed_path or ""
        try:
            return str(Path(file_path).relative_to(root))
        except ValueError:
            return file_path

    def _resolve_path(self, path: str) -> str:
        """
        Resolve path, handling Docker volume mounts.

        When running in Docker with a volume mount like ``-v ".:/workspace"``,
        host paths don't exist inside the container.  This translates paths like
        ``/Users/.../project/src`` → ``/workspace/src``.

        Learns the host↔container root mapping on first successful match
        so future resolutions use fast prefix replacement.
        """
        if os.path.exists(path):
            return path

        # Fast path: use learned mapping
        if self._host_root is not None and path.startswith(self._host_root):
            remainder = path[len(self._host_root) :]
            if remainder.startswith("/"):
                remainder = remainder[1:]
            return str(Path("/workspace") / remainder) if remainder else "/workspace"

        workspace = Path("/workspace")
        dockerenv = Path("/.dockerenv")
        if not (workspace.exists() and dockerenv.exists()):
            return path

        p = Path(path)
        # Skip leading '/' to avoid joinpath resetting to root
        parts = tuple(pt for pt in p.parts if pt != "/")

        for i in range(len(parts)):
            candidate = workspace.joinpath(*parts[i:])
            if candidate.exists():
                self._learn_host_root(path, str(candidate))
                return str(candidate)

        # For new files: resolve the parent directory and append the filename
        parent_parts = tuple(pt for pt in p.parent.parts if pt != "/")
        for i in range(len(parent_parts)):
            candidate = workspace.joinpath(*parent_parts[i:])
            if candidate.is_dir():
                resolved = str(candidate / p.name)
                self._learn_host_root(str(p.parent), str(candidate))
                return resolved

        # Fallback: assume file is directly under /workspace
        return str(workspace / p.name)

    def _learn_host_root(self, host_path: str, container_path: str) -> None:
        """Derive and store the host↔container root mapping."""
        suffix = container_path[len("/workspace") :]
        if suffix and host_path.endswith(suffix):
            self._host_root = host_path[: -len(suffix)]

    def _format_locations(self, entries: list[IndexEntry]) -> list[str]:
        """Format entry locations for output."""
        return [
            f"{self._relative_path(e.code_unit.file_path)}:{e.code_unit.name}:L{e.code_unit.line_start}-{e.code_unit.line_end}"
            for e in entries
        ]

    def _write_analysis_report(self, content: str) -> Path | None:
        """Write full analysis report to persistence directory.

        Returns the timestamped file path on success, None on failure.
        """
        if self._last_indexed_path is None:
            return None
        try:
            persistence_path = _get_persistence_path(self._last_indexed_path)
            persistence_path.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            report_file = persistence_path / f"analysis_report_{timestamp}.txt"
            report_file.write_text(content)
            return report_file
        except OSError:
            return None

    def _clear_analysis_report(self) -> None:
        """Remove stale legacy report alias when no findings."""
        if self._last_indexed_path is not None:
            try:
                legacy_report = (
                    _get_persistence_path(self._last_indexed_path) / LEGACY_ANALYSIS_REPORT
                )
                legacy_report.unlink(missing_ok=True)
            except OSError:
                pass

    @_requires_index
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
        # Check for invalidated suppressions (proactive notification)
        invalidation_warning = self._check_invalidated_suppressions()

        # Check for staleness and auto-reindex if enabled
        staleness_warning = ""
        report = self.index.get_staleness_report()
        if report.is_stale:
            if auto_reindex and self._last_indexed_path:
                # Auto-reindex for accurate results
                self.index_codebase(self._last_indexed_path)
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
                keep = f"{self._relative_path(e.code_unit.file_path)}:{e.code_unit.name}:L{e.code_unit.line_start}-{e.code_unit.line_end}"
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
        block_groups = self.index.find_block_duplicates(
            min_node_count=self._MIN_BLOCK_DUPLICATE_NODES
        )
        for group in block_groups:
            block_type = group.entries[0].code_unit.block_type or "block"
            first = group.entries[0]
            line_count = first.code_unit.line_end - first.code_unit.line_start + 1
            if line_count < self._MIN_BLOCK_DUPLICATE_LINES:
                continue
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

        def _suppressed_line(with_period: bool = False) -> str | None:
            if suppressed_count <= 0:
                return None
            return f"+ {suppressed_count} suppressed{'.' if with_period else ''}"

        if not findings:
            self._clear_analysis_report()
            msg = "No significant duplicates."
            if suppressed_count:
                msg += f" {suppressed_count} suppressed."
            return ToolResult(invalidation_warning + staleness_warning + msg)

        # Classify findings as source or test
        def _is_test_location(loc: str) -> bool:
            return "/tests/" in loc or loc.startswith("tests/") or "/test_" in loc

        for f in findings:
            f["is_test"] = all(_is_test_location(loc) for loc in f["locations"])

        # Sort: source findings first, then test findings
        source_findings = [f for f in findings if not f["is_test"]]
        test_findings = [f for f in findings if f["is_test"]]

        lines: list[str] = []

        def _format_finding(num: int, f: dict[str, Any]) -> list[str]:
            result: list[str] = []
            all_locs = f["locations"]
            max_locs = 5
            if len(all_locs) > max_locs:
                locs = ", ".join(all_locs[:max_locs]) + f" ... +{len(all_locs) - max_locs} more"
            else:
                locs = ", ".join(all_locs)
            wl_hash = f.get("hash", "")
            line_count = f.get("line_count", 0)
            verified = " (verified)" if f.get("verified") else ""

            if f["type"] == "exact":
                result.append(f"{num}.{verified} {locs} ({line_count} lines)")
            elif f["type"] == "block":
                block_type = f.get("block_type", "block")
                result.append(f"{num}. [{block_type}]{verified} {locs} ({line_count} lines)")
            else:
                result.append(f"{num}. [pattern] {locs} ({line_count} lines)")

            if f.get("keep"):
                result.append(f"   Keep: {f['keep']} ({f['keep_reason']})")
            result.append(f'   suppress(wl_hash="{wl_hash}")')
            result.append("")
            return result

        # Emit section headers when both source and test findings exist
        has_both = bool(source_findings) and bool(test_findings)
        num = 1
        for header, section in [
            ("=== Source code ===", source_findings),
            ("=== Tests ===", test_findings),
        ]:
            if has_both:
                lines.append(header)
                lines.append("")
            for f in section:
                lines.extend(_format_finding(num, f))
                num += 1

        # Compact footer
        if suppressed_line := _suppressed_line():
            lines.append(suppressed_line)
        lines.append(f"{len(findings)} duplicate groups.")

        full_output = "\n".join(lines)

        # Write full report to file, return compact summary inline
        report_path = self._write_analysis_report(full_output)
        if report_path is not None:
            # Count type breakdown
            type_parts = [
                f"{count} {name}"
                for name in ("exact", "block", "pattern")
                if (count := sum(1 for f in findings if f["type"] == name))
            ]
            summary_parts = [f"Found {len(findings)} duplicate groups: {', '.join(type_parts)}."]

            # Source vs test breakdown with duplicated line estimate
            source_count = len(source_findings)
            test_count = len(test_findings)
            if source_count or test_count:
                source_lines = sum(f["line_count"] * len(f["locations"]) for f in source_findings)
                breakdown_parts = []
                if source_count:
                    breakdown_parts.append(
                        f"{source_count} in source (~{source_lines} duplicated lines)"
                    )
                if test_count:
                    breakdown_parts.append(f"{test_count} in tests")
                summary_parts.append(f"  {', '.join(breakdown_parts)}.")

            if suppressed_line := _suppressed_line(with_period=True):
                summary_parts.append(suppressed_line)
            line_count_report = full_output.count("\n") + 1
            summary_parts.append(
                f"Details: {PERSISTENCE_DIR}/{report_path.name} ({line_count_report} lines)"
            )
            summary_parts.append("Read the file to see locations and suppress commands.")
            return ToolResult(invalidation_warning + staleness_warning + "\n".join(summary_parts))

        # Fallback: file write failed or no indexed path — return full output inline
        return ToolResult(invalidation_warning + staleness_warning + full_output)

    @_requires_index
    def check(self, code: str) -> ToolResult:
        """
        Check if code similar to the provided snippet exists.

        Use this BEFORE creating new code to avoid duplication.
        """
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
            rel = self._relative_path(entry.code_unit.file_path)
            return ToolResult(
                prefix + f"STOP: Identical code exists at {rel}:{entry.code_unit.name} "
                f"(lines {entry.code_unit.line_start}-{entry.code_unit.line_end}). Reuse it."
            )
        elif high:
            entry = high[0].entry
            rel = self._relative_path(entry.code_unit.file_path)
            return ToolResult(
                prefix + f"CAUTION: Very similar code at {rel}:{entry.code_unit.name}. "
                f"Consider reusing or extending."
            )
        elif partial:
            entry = partial[0].entry
            rel = self._relative_path(entry.code_unit.file_path)
            return ToolResult(
                prefix + f"NOTE: Partially similar code at {rel}:{entry.code_unit.name}. "
                f"Review for potential reuse."
            )

        return ToolResult(prefix + "No similar code found. Safe to proceed.")

    def compare(self, code1: str, code2: str, language: str = "python") -> ToolResult:
        """Compare two code snippets for structural equivalence."""
        plugin = LanguageRegistry.get().get_plugin(language)
        if plugin is None:
            return ToolResult(f"Unsupported language '{language}': no registered language plugin.")

        g1 = plugin.source_to_graph(code1)
        g2 = plugin.source_to_graph(code2)

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

    @_requires_index
    def _toggle_suppression(self, wl_hash: str, suppress: bool) -> ToolResult:
        """Toggle hash suppression status with index check."""
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

        # Toggle suppression state using event-driven index if available (persists to SQLite)
        active_index = self._active_index()
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

    suppress = partialmethod(_toggle_suppression, suppress=True)
    unsuppress = partialmethod(_toggle_suppression, suppress=False)

    @_requires_index
    def _batch_toggle_suppression(self, wl_hashes: list[str], suppress: bool) -> ToolResult:
        """Batch suppress or unsuppress hashes."""
        prefix = self._check_invalidated_suppressions()
        active_index = self._active_index()
        action = "suppress" if suppress else "unsuppress"
        method = getattr(active_index, f"{action}_batch")
        changed, not_found = method(wl_hashes)
        parts = []
        if changed:
            label = "Suppressed" if suppress else "Unsuppressed"
            parts.append(f"{label} {len(changed)} hashes.")
        if not_found:
            max_shown = 5
            shown = self._format_hash_preview(not_found, max_shown=max_shown)
            parts.append(f"{len(not_found)} not found: {shown}")
        parts = parts or ["No hashes provided."]
        if changed and suppress:
            parts.append("Run analyze to refresh.")
        return ToolResult(prefix + " ".join(parts))

    suppress_batch = partialmethod(_batch_toggle_suppression, suppress=True)
    unsuppress_batch = partialmethod(_batch_toggle_suppression, suppress=False)

    def list_suppressions(self) -> ToolResult:
        """List all suppressed hashes."""
        prefix = self._check_invalidated_suppressions()
        suppressed = self.index.get_suppressed()
        if suppressed:
            max_shown = 20
            shown = suppressed[:max_shown]
            text = prefix + f"Suppressed hashes ({len(suppressed)}):\n" + "\n".join(shown)
            if len(suppressed) > max_shown:
                text += f"\n... +{len(suppressed) - max_shown} more"
            return ToolResult(text)
        return ToolResult(prefix + "No hashes are currently suppressed.")

    def status(self) -> ToolResult:
        """Return current server status without blocking."""
        if not self._bg_index_done.is_set():
            entry_count = len(self.index.entries)
            return ToolResult(f"Status: indexing ({entry_count} entries so far)")
        if not self.index.entries:
            return ToolResult("Status: idle (no codebase indexed)")
        stats = self.index.get_stats()
        return ToolResult(
            f"Status: ready ({stats['function_entries']} code units, "
            f"{stats['indexed_files']} files)"
        )

    def _lsp_setup_workspace(self) -> Path:
        """Resolve workspace root used for LSP binding persistence."""
        if self._last_indexed_path:
            indexed = Path(self._last_indexed_path)
            return indexed.parent if indexed.is_file() else indexed

        detected = self._detect_startup_workspace()
        if detected:
            return Path(detected)

        return Path.cwd()

    def _lsp_setup_result(self, payload: dict[str, Any]) -> ToolResult:
        """Serialize structured LSP setup responses as JSON."""
        return ToolResult(json.dumps(payload, indent=2, sort_keys=True))

    def lsp_setup(
        self,
        mode: str = "inspect",
        language: str | None = None,
        command: Sequence[str] | str | None = None,
        observations: list[dict[str, Any]] | None = None,
    ) -> ToolResult:
        """Inspect and configure deterministic command bindings for bundled LSP plugins."""
        workspace = self._lsp_setup_workspace()
        normalized_mode = (mode or "inspect").strip().lower()
        known_languages = [
            spec.language_id for spec in sorted(bundled_lsp_specs(), key=lambda s: s.language_id)
        ]

        def _validate_language(mode_name: str) -> ToolResult | None:
            if not language:
                return self._lsp_setup_result(
                    {
                        "ok": False,
                        "mode": mode_name,
                        "error": f"'language' is required when mode='{mode_name}'",
                        "supported_languages": known_languages,
                    }
                )

            if get_lsp_spec(language) is None:
                return self._lsp_setup_result(
                    {
                        "ok": False,
                        "mode": mode_name,
                        "error": f"Unsupported language '{language}'",
                        "supported_languages": known_languages,
                    }
                )

            return None

        def _status_for_language(language_id: str) -> dict[str, Any] | None:
            return next(
                (
                    current
                    for current in collect_lsp_statuses(workspace)
                    if current["language"] == language_id
                ),
                None,
            )

        if normalized_mode == "inspect":
            statuses = collect_lsp_statuses(workspace)
            missing = [status["language"] for status in statuses if not status["available"]]
            missing_required = [
                status["language"]
                for status in statuses
                if status.get("required", True) and not status["available"]
            ]
            payload: dict[str, Any] = {
                "ok": True,
                "mode": normalized_mode,
                "workspace": str(workspace),
                "bindings_path": str(lsp_bindings_path(workspace)),
                "servers": statuses,
                "missing_languages": missing,
                "missing_required_languages": missing_required,
                "supported_languages": known_languages,
            }
            if missing_required:
                payload["next_step"] = (
                    "Call astrograph_lsp_setup with mode='auto_bind'. "
                    "If still missing, provide observations with language + command."
                )
            elif missing:
                payload["next_step"] = (
                    "Optional attach endpoints are currently unavailable. "
                    "Call astrograph_lsp_setup with mode='auto_bind' to configure them."
                )

            if missing:
                payload["observation_format"] = {
                    "language": "cpp_lsp",
                    "command": "tcp://127.0.0.1:2088",
                }
                payload["observation_examples"] = [
                    {
                        "language": "python",
                        "command": ["/absolute/path/to/pylsp"],
                    },
                    {
                        "language": "java_lsp",
                        "command": "tcp://127.0.0.1:2089",
                    },
                ]
            return self._lsp_setup_result(payload)

        if normalized_mode == "auto_bind":
            outcome = auto_bind_missing_servers(workspace=workspace, observations=observations)
            if outcome["changes"]:
                LanguageRegistry.reset()
            outcome.update(
                {
                    "ok": True,
                    "mode": normalized_mode,
                    "supported_languages": known_languages,
                }
            )
            return self._lsp_setup_result(outcome)

        if normalized_mode == "bind":
            if validation_error := _validate_language(normalized_mode):
                return validation_error

            target_language = cast(str, language)
            parsed_command = parse_command(command)
            if not parsed_command:
                return self._lsp_setup_result(
                    {
                        "ok": False,
                        "mode": normalized_mode,
                        "error": "'command' must be a non-empty string or array",
                    }
                )

            saved_command, path = set_lsp_binding(target_language, parsed_command, workspace)
            LanguageRegistry.reset()

            status = _status_for_language(target_language)
            probe = probe_command(saved_command)
            return self._lsp_setup_result(
                {
                    "ok": True,
                    "mode": normalized_mode,
                    "language": target_language,
                    "workspace": str(workspace),
                    "bindings_path": str(path),
                    "saved_command": saved_command,
                    "saved_command_text": format_command(saved_command),
                    "available": probe["available"],
                    "executable": probe["executable"],
                    "status": status,
                }
            )

        if normalized_mode == "unbind":
            if validation_error := _validate_language(normalized_mode):
                return validation_error

            target_language = cast(str, language)
            removed, path = unset_lsp_binding(target_language, workspace)
            LanguageRegistry.reset()
            status = _status_for_language(target_language)
            return self._lsp_setup_result(
                {
                    "ok": True,
                    "mode": normalized_mode,
                    "language": target_language,
                    "workspace": str(workspace),
                    "bindings_path": str(path),
                    "removed": removed,
                    "status": status,
                }
            )

        return self._lsp_setup_result(
            {
                "ok": False,
                "mode": normalized_mode,
                "error": "Invalid mode",
                "supported_modes": ["inspect", "auto_bind", "bind", "unbind"],
                "supported_languages": known_languages,
            }
        )

    def metadata_erase(self) -> ToolResult:
        """
        Erase all persisted metadata (.metadata_astrograph/).

        Deletes the SQLite database, suppressions, analysis reports,
        and resets the in-memory index. Server returns to idle state.
        """
        self._wait_for_background_index()

        # Close event-driven index (stops watcher, closes SQLite)
        self._close_event_driven_index()

        # Clear in-memory index + suppressions
        self.index.clear()
        self.index.clear_suppressions()

        # Delete persistence directory
        erased = False
        if self._last_indexed_path:
            persistence_path = _get_persistence_path(self._last_indexed_path)
            if persistence_path.exists():
                shutil.rmtree(persistence_path, ignore_errors=True)
                erased = True

        # Create fresh event-driven index (no persistence until next index_codebase)
        self._event_driven_index = EventDrivenIndex(
            persistence_path=None,
            watch_enabled=True,
        )
        self.index = self._event_driven_index.index

        if erased:
            return ToolResult("Erased all metadata. Server reset to idle state.")
        return ToolResult("No metadata to erase. Server is idle.")

    def metadata_recompute_baseline(self) -> ToolResult:
        """
        Erase metadata and re-index the codebase from scratch.

        Equivalent to erasing all persisted data and running a fresh
        full index. Suppressions are cleared.
        """
        if not self._last_indexed_path:
            return ToolResult("No codebase has been indexed. Nothing to recompute.")

        path = self._last_indexed_path

        # Erase everything
        self.metadata_erase()

        # Re-index from scratch
        result = self.index_codebase(path)
        return ToolResult(f"Baseline recomputed from scratch.\n{result.text}")

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

    @_requires_index
    def check_staleness(self, path: str | None = None) -> ToolResult:
        """
        Check if the code index is stale (files changed since indexing).

        Args:
            path: Optional root path to also check for new files.
        """
        # Proactive notification of invalidated suppressions
        prefix = self._check_invalidated_suppressions()

        # Resolve path (handles Docker volume mounts)
        if path is not None:
            path = self._resolve_path(path)

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
            delta = added - removed
            if delta == 0:
                summary = f"  Changed {removed} line{'s' if removed != 1 else ''}"
            elif delta > 0:
                summary = f"  Added {delta} line{'s' if delta != 1 else ''}"
            else:
                removed_delta = -delta
                summary = f"  Removed {removed_delta} line{'s' if removed_delta != 1 else ''}"

        # Build diff lines
        diff = []
        # Context before
        ctx_start = max(0, start_line - context_lines)
        for i in range(ctx_start, start_line):
            diff.append(f"  {i + 1:>4}  {file_lines[i]}")

        def _append_marked(lines: list[str], marker: str) -> None:
            for i, line in enumerate(lines):
                diff.append(f"  {start_line + i + 1:>4} {marker}{line}")

        # Removed lines (original numbering)
        _append_marked(old_lines, "-")

        # Added lines (new numbering)
        _append_marked(new_lines, "+")

        # Context after
        end_line = start_line + len(old_lines)
        ctx_end = min(len(file_lines), end_line + context_lines)
        # Adjust context line numbers for the new file
        line_offset = added - removed
        for i in range(end_line, ctx_end):
            diff.append(f"  {i + 1 + line_offset:>4}  {file_lines[i]}")

        return f"Edited {file_path}\n{summary}\n" + "\n".join(diff)

    @_requires_index
    def write(self, file_path: str, content: str) -> ToolResult:
        """
        Write code to a file with automatic duplicate detection.

        Checks the content for structural duplicates before writing.
        Blocks if identical code exists elsewhere; warns on high similarity.
        """
        file_path = self._resolve_path(file_path)

        warning = ""
        # Infer language from file extension
        plugin = LanguageRegistry.get().get_plugin_for_file(file_path)
        if plugin is None:
            warning = (
                "NOTE: No language plugin is registered for this file extension. "
                "Skipping structural duplicate checks.\n\n"
            )
        else:
            # Check for duplicates in the content
            results = self.index.find_similar(
                content,
                min_node_count=self._CHECK_MIN_STATEMENTS,
                language=plugin.language_id,
            )

            exact = [r for r in results if r.similarity_type == "exact"]
            high = [r for r in results if r.similarity_type == "high"]

            if exact:
                entry = exact[0].entry
                rel = self._relative_path(entry.code_unit.file_path)
                return ToolResult(
                    f"BLOCKED: Cannot write - identical code exists at "
                    f"{rel}:{entry.code_unit.name} "
                    f"(lines {entry.code_unit.line_start}-{entry.code_unit.line_end}). "
                    f"Reuse the existing implementation instead."
                )

            if high:
                entry = high[0].entry
                rel = self._relative_path(entry.code_unit.file_path)
                warning = (
                    f"WARNING: Similar code exists at {rel}:{entry.code_unit.name}. "
                    f"Consider reusing. Proceeding with write.\n\n"
                )

        import os.path

        is_new = not os.path.exists(file_path)
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        action = "Created" if is_new else "Wrote"
        rel_file = self._relative_path(file_path)
        summary = f"{action} {rel_file} ({line_count} lines)"
        return self._write_file(
            file_path, content, summary=warning + summary, display_path=rel_file
        )

    def _write_file(
        self, file_path: str, content: str, summary: str = "", display_path: str = ""
    ) -> ToolResult:
        """Write content to a file with error handling."""
        try:
            Path(file_path).write_text(content)
            return ToolResult(summary or f"Wrote {display_path or file_path}")
        except OSError as e:
            return ToolResult(f"Failed to write {display_path or file_path}: {e}")

    @_requires_index
    def edit(self, file_path: str, old_string: str, new_string: str) -> ToolResult:
        """
        Edit a file with automatic duplicate detection.

        Checks the new_string for structural duplicates before applying.
        Blocks if identical code exists elsewhere; warns on high similarity.
        """
        file_path = self._resolve_path(file_path)
        display_path = self._relative_path(file_path)

        warning = ""
        # Infer language from file extension
        plugin = LanguageRegistry.get().get_plugin_for_file(file_path)
        if plugin is None:
            warning = (
                "NOTE: No language plugin is registered for this file extension. "
                "Skipping structural duplicate checks.\n\n"
            )
        else:
            # Check for duplicates in the new code being introduced
            results = self.index.find_similar(
                new_string,
                min_node_count=self._CHECK_MIN_STATEMENTS,
                language=plugin.language_id,
            )

            exact = [r for r in results if r.similarity_type == "exact"]
            high = [r for r in results if r.similarity_type == "high"]

            if exact:
                entry = exact[0].entry
                # Compare resolved paths to handle symlinks (e.g., /var -> /private/var on macOS)
                if str(Path(entry.code_unit.file_path).resolve()) != str(Path(file_path).resolve()):
                    # Cross-file duplicate: block
                    rel = self._relative_path(entry.code_unit.file_path)
                    return ToolResult(
                        f"BLOCKED: Cannot edit - identical code exists at "
                        f"{rel}:{entry.code_unit.name} "
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
                rel = self._relative_path(entry.code_unit.file_path)
                warning = (
                    f"WARNING: Similar code exists at {rel}:{entry.code_unit.name}. "
                    f"Consider reusing. Proceeding with edit.\n\n"
                )

        # Read the file
        try:
            with open(file_path) as f:
                content = f.read()
        except FileNotFoundError:
            return ToolResult(f"File not found: {display_path}")
        except OSError as e:
            return ToolResult(f"Failed to read {display_path}: {e}")

        # Check that old_string exists
        if old_string not in content:
            return ToolResult(
                f"Edit failed: old_string not found in {display_path}. The file may have changed."
            )

        # Check for uniqueness
        count = content.count(old_string)
        if count > 1:
            return ToolResult(
                f"Edit failed: old_string appears {count} times in {display_path}. "
                f"Provide more context to make it unique."
            )

        # Apply the edit
        new_content = content.replace(old_string, new_string, 1)
        diff = self._format_edit_diff(content, old_string, new_string, display_path)

        return self._write_file(
            file_path, new_content, summary=warning + diff, display_path=display_path
        )

    def call_tool(self, name: str, arguments: dict) -> ToolResult:
        """Dispatch a tool call by name."""
        if name == "index_codebase":
            return self.index_codebase(
                path=arguments["path"],
                recursive=arguments.get("recursive", True),
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
            return cast(ToolResult, self.suppress(wl_hash=arguments["wl_hash"]))
        elif name == "suppress_batch":
            return cast(ToolResult, self.suppress_batch(wl_hashes=arguments["wl_hashes"]))
        elif name == "unsuppress":
            return cast(ToolResult, self.unsuppress(wl_hash=arguments["wl_hash"]))
        elif name == "unsuppress_batch":
            return cast(ToolResult, self.unsuppress_batch(wl_hashes=arguments["wl_hashes"]))
        elif name == "list_suppressions":
            return self.list_suppressions()
        elif name == "status":
            return self.status()
        elif name == "metadata_erase":
            return self.metadata_erase()
        elif name == "metadata_recompute_baseline":
            return self.metadata_recompute_baseline()
        elif name == "check_staleness":
            return self.check_staleness(path=arguments.get("path"))
        elif name == "lsp_setup":
            return self.lsp_setup(
                mode=arguments.get("mode", "inspect"),
                language=arguments.get("language"),
                command=arguments.get("command"),
                observations=arguments.get("observations"),
            )
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
        self._close_event_driven_index()

    def get_event_driven_stats(self) -> dict | None:
        """Get event-driven mode statistics (returns None if no event-driven index)."""
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
