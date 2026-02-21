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
from functools import wraps
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
from .ignorefile import ASTROGRAPHIGNORE_FILENAME, IgnoreSpec
from .index import CodeStructureIndex, DuplicateGroup, IndexEntry, SimilarityResult
from .languages.base import SemanticProfile, node_match
from .languages.registry import LanguageRegistry
from .lsp_setup import (
    auto_bind_missing_servers,
    bundled_lsp_specs,
    collect_lsp_statuses,
    format_command,
    get_lsp_spec,
    language_variant_policy,
    load_lsp_bindings,
    lsp_bindings_path,
    parse_attach_endpoint,
    parse_command,
    probe_command,
    set_lsp_binding,
    unset_lsp_binding,
)


def _env_flag(name: str) -> bool:
    """Return True if an env flag is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


logger = logging.getLogger(__name__)


# Persistence directory name for cached index
PERSISTENCE_DIR = ".metadata_astrograph"
LEGACY_ANALYSIS_REPORT = "analysis_report.txt"
_SEMANTIC_MODES = frozenset({"off", "annotate", "differentiate"})
_MUTATING_TOOL_NAMES = frozenset(
    {
        "edit",
        "generate_ignore",
        "index_codebase",
        "metadata_erase",
        "metadata_recompute_baseline",
        "set_workspace",
        "suppress",
        "unsuppress",
        "write",
    }
)
_MUTATING_LSP_SETUP_MODES = frozenset({"auto_bind", "bind", "unbind"})


def _generate_default_ignore_content() -> str:
    """Return default .astrographignore content."""
    return """\
# .astrographignore — files/directories excluded from ASTrograph indexing
# Syntax: gitignore-like (globs, directory/, /anchored, **, !negation)

# Vendored / third-party
vendor/
third_party/
third-party/
external/

# Minified / generated code
*.min.js
*.min.css
*.bundle.js
*.chunk.js
*.generated.*

# Build output
dist/
build/
out/
target/
_build/

# Package managers
node_modules/
bower_components/
jspm_packages/

# Language caches
__pycache__/
*.pyc
.tox/
.mypy_cache/
.pytest_cache/
.ruff_cache/
.eggs/

# IDE / editor
.idea/
.vscode/
*.swp
*.swo

# Version control
.git/

# Virtual environments
venv/
.venv/
env/
.env/

# Test snapshots
__snapshots__/
*.snap

# Large data files
*.csv
*.json
*.xml
*.yaml
*.yml
*.lock
"""


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
        self._ignore_spec: IgnoreSpec | None = None

        # Background indexing state
        self._bg_index_thread: threading.Thread | None = None
        self._bg_index_done = threading.Event()
        self._bg_index_done.set()  # Initially "done" (no background work)
        self._bg_index_progress: str = ""
        # Serialize mutating tool calls for shared-agent/container deployments.
        self._mutation_lock = threading.RLock()

        # Auto-index workspace at startup (Docker / Codex / local MCP clients).
        # Run in background so the MCP handshake completes immediately.
        workspace = self._detect_startup_workspace()
        if (
            workspace
            and os.path.isdir(workspace)
            and not _env_flag("ASTROGRAPH_DISABLE_AUTO_INDEX")
        ):
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
        except Exception as exc:
            logger.exception(f"Background indexing failed for {path}")
            self._bg_index_progress = "error"
            self._bg_index_error = str(exc)
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
        if self.index.entries:
            return None
        if self._bg_index_progress == "error":
            error_detail = getattr(self, "_bg_index_error", "unknown error")
            return ToolResult(f"Background indexing failed: {error_detail}")
        return ToolResult("No code indexed. Call index_codebase first.")

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
        if _env_flag("ASTROGRAPH_DISABLE_EVENT_DRIVEN"):
            return self._index_codebase_simple(path, recursive)
        return self._index_codebase_event_driven(path, recursive)

    def set_workspace(self, path: str) -> ToolResult:
        """Switch to a new workspace directory and re-index."""
        old_path = self._last_indexed_path or "(none)"
        result = self.index_codebase(path, recursive=True)
        new_path = self._last_indexed_path or path

        # Prepend workspace transition info
        header = f"Workspace changed: {old_path} -> {new_path}\n"
        return ToolResult(header + result.text)

    def _index_codebase_simple(self, path: str, recursive: bool) -> ToolResult:
        """Index codebase using a plain in-memory index (no EDI, no SQLite, no watcher)."""
        self._close_event_driven_index()
        self.index = CodeStructureIndex()

        if os.path.isfile(path):
            self.index.index_file(path)
        else:
            self.index.index_directory(path, recursive=recursive, ignore_spec=self._ignore_spec)

        stats = self.index.get_stats()
        result_parts = [
            f"Indexed {stats['function_entries']} code units from {stats['indexed_files']} files.",
            f"Extracted {stats['block_entries']} code blocks.",
        ]
        if self._has_significant_duplicates():
            result_parts.append("\nDuplicates found. Run analyze().")
        else:
            result_parts.append("\nNo duplicates.")
        return ToolResult("\n".join(result_parts))

    def _index_codebase_event_driven(self, path: str, recursive: bool) -> ToolResult:
        """Index codebase using event-driven mode with SQLite and file watching."""
        from .cloud_detect import get_cloud_sync_warning

        # Check for cloud-synced folders and prepare warning
        cloud_warning = get_cloud_sync_warning(path)

        disable_watch = _env_flag("ASTROGRAPH_DISABLE_WATCH")
        disable_persistence = _env_flag("ASTROGRAPH_DISABLE_PERSISTENCE")

        sqlite_path = None
        if not disable_persistence:
            persistence_path = _get_persistence_path(path)
            persistence_path.mkdir(exist_ok=True)
            sqlite_path = persistence_path / "index.db"

        # Close old event-driven index (stops watcher, closes SQLite connection)
        self._close_event_driven_index()

        # Load .astrographignore if present
        ignore_dir = Path(path) if os.path.isdir(path) else Path(path).parent
        self._ignore_spec = IgnoreSpec.from_file(str(ignore_dir / ASTROGRAPHIGNORE_FILENAME))

        # Create new event-driven index with persistence
        self._event_driven_index = EventDrivenIndex(
            persistence_path=sqlite_path,
            watch_enabled=not disable_watch,
            ignore_spec=self._ignore_spec,
        )
        self.index = self._event_driven_index.index

        # Index the path (loads from cache if available)
        if os.path.isfile(path):
            # Single file: route through EDI for watching + cache
            self._event_driven_index.index_file(path)
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
        """Verify a duplicate group via graph isomorphism (checks all pairs against first)."""
        if len(group.entries) < 2:
            return False
        first = group.entries[0]
        return all(self.index.verify_isomorphism(first, entry) for entry in group.entries[1:])

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
            # Filter traversal components from remainder
            if remainder:
                safe_parts = [pt for pt in Path(remainder).parts if pt not in ("/", ".", "..")]
                remainder = str(Path(*safe_parts)) if safe_parts else ""
            return str(Path("/workspace") / remainder) if remainder else "/workspace"

        workspace = Path("/workspace")
        dockerenv = Path("/.dockerenv")
        if not (workspace.exists() and dockerenv.exists()):
            return path

        p = Path(path)
        # Skip leading '/' and filter traversal components (..)
        parts = tuple(pt for pt in p.parts if pt not in ("/", ".", ".."))

        for i in range(len(parts)):
            candidate = workspace.joinpath(*parts[i:])
            if candidate.exists():
                self._learn_host_root(path, str(candidate))
                return str(candidate)

        # For new files: resolve the parent directory and append the filename
        parent_parts = tuple(pt for pt in p.parent.parts if pt not in ("/", ".", ".."))
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
        elif not suffix:
            # container_path is exactly "/workspace" — host_path IS the root
            self._host_root = host_path

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
            auto_reindex: Kept for API compatibility. The event-driven index
                         keeps the index current via file watching automatically.

        Returns findings sorted by relevance:
        - Exact duplicates (verified via graph isomorphism)
        - Pattern duplicates (same structure, different operators)
        - Block duplicates (duplicate for/while/if/try/with blocks within functions)
        """
        del auto_reindex  # Kept for API compatibility; event-driven index auto-reindexes
        # Check for invalidated suppressions (proactive notification)
        invalidation_warning = self._check_invalidated_suppressions()

        # Event-driven index keeps things current via file watching + cache.
        edi = self._event_driven_index
        if edi is not None and not _env_flag("ASTROGRAPH_DISABLE_ANALYSIS_CACHE"):
            exact_groups, pattern_groups, block_groups = edi.get_cached_analysis()
        else:
            # No EDI, cache disabled, or testing path: direct compute
            exact_groups = self.index.find_all_duplicates(min_node_count=5)
            pattern_groups = self.index.find_pattern_duplicates(min_node_count=5)
            block_groups = self.index.find_block_duplicates(
                min_node_count=self._MIN_BLOCK_DUPLICATE_NODES
            )

        # Filter block groups by analyze threshold (cache uses min_node_count=5,
        # analyze wants _MIN_BLOCK_DUPLICATE_NODES=10)
        block_groups = [
            g for g in block_groups if g.entries[0].node_count >= self._MIN_BLOCK_DUPLICATE_NODES
        ]

        findings: list[dict[str, Any]] = []

        for group in exact_groups:
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

        # Block duplicates (duplicate code blocks within functions)
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

        # Pattern duplicates (same structure, different operators)
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
            return ToolResult(invalidation_warning + msg)

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
            result.append(
                f'   Action: refactor to eliminate duplication, or suppress(wl_hash="{wl_hash}") if intentional.'
            )
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
            summary_parts.append(
                "Read the file to see locations and refactoring opportunities.\nRefactor duplicates first. Only suppress intentional patterns (API symmetry, test isolation, framework boilerplate)."
            )
            return ToolResult(invalidation_warning + "\n".join(summary_parts))

        # Fallback: file write failed or no indexed path — return full output inline
        return ToolResult(invalidation_warning + full_output)

    @_requires_index
    def check(self, code: str, language: str = "python") -> ToolResult:
        """
        Check if code similar to the provided snippet exists.

        Use this BEFORE creating new code to avoid duplication.
        """
        # Proactive notification of invalidated suppressions
        prefix = self._check_invalidated_suppressions()
        plugin = LanguageRegistry.get().get_plugin(language)
        if plugin is None:
            return ToolResult(
                prefix + f"Unsupported language '{language}': no registered language plugin."
            )

        results = self.index.find_similar(
            code,
            min_node_count=self._CHECK_MIN_STATEMENTS,
            language=language,
        )
        same_language, cross_language = self._split_similarity_results_by_language(
            results, language
        )
        assist = self._cross_language_assist_text(
            source_code=code,
            source_language=language,
            results=cross_language,
        )

        if not same_language:
            base = "No same-language similar code found. Safe to proceed."
            if assist:
                base += "\n\n" + assist
            return ToolResult(prefix + base)

        tiers = [
            ("exact", "STOP: Identical {lang} code exists at {loc} (lines {lines}). Reuse it."),
            ("high", "CAUTION: Very similar {lang} code at {loc}. Consider reusing or extending."),
            (
                "partial",
                "NOTE: Partially similar {lang} code at {loc}. Review for potential reuse.",
            ),
        ]
        for tier_type, template in tiers:
            matches = [r for r in same_language if r.similarity_type == tier_type]
            if matches:
                entry = matches[0].entry
                rel = self._relative_path(entry.code_unit.file_path)
                loc = f"{rel}:{entry.code_unit.name}"
                lines = f"{entry.code_unit.line_start}-{entry.code_unit.line_end}"
                message = template.format(lang=language, loc=loc, lines=lines)
                if assist:
                    message += "\n\n" + assist
                return ToolResult(prefix + message)

        message = "No same-language similar code found. Safe to proceed."
        if assist:
            message += "\n\n" + assist
        return ToolResult(prefix + message)

    @staticmethod
    def _split_similarity_results_by_language(
        results: list[SimilarityResult],
        language_id: str,
    ) -> tuple[list[SimilarityResult], list[SimilarityResult]]:
        """Split similarity results into same-language vs cross-language buckets."""
        same_language = [
            result for result in results if result.entry.code_unit.language == language_id
        ]
        cross_language = [
            result for result in results if result.entry.code_unit.language != language_id
        ]
        return same_language, cross_language

    @staticmethod
    def _similarity_sort_key(result: SimilarityResult) -> tuple[int, int]:
        """Sort key for similarity results (exact > high > partial)."""
        type_order = {"exact": 0, "high": 1, "partial": 2}
        return (
            type_order.get(result.similarity_type, 99),
            -(result.matching_depth or 0),
        )

    @classmethod
    def _prefer_similarity_result(
        cls,
        current: SimilarityResult | None,
        candidate: SimilarityResult,
    ) -> SimilarityResult:
        """Pick the stronger of two similarity results for the same entry."""
        if current is None:
            return candidate
        if cls._similarity_sort_key(candidate) < cls._similarity_sort_key(current):
            return candidate
        return current

    def _candidate_similarity_snippets(
        self,
        *,
        plugin: Any,
        source_code: str,
        file_path: str,
    ) -> list[str]:
        """
        Build deduplicated snippets for duplicate checks.

        Includes the full source and, when available, extracted code units so
        write/edit checks work even when wrapper lines (imports/includes) are present.
        """
        snippets: list[str] = []
        seen: set[str] = set()

        def _add(candidate: str) -> None:
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                return
            seen.add(normalized)
            snippets.append(normalized)

        _add(source_code)

        try:
            extracted_units = plugin.extract_code_units(
                source_code,
                file_path,
                include_blocks=False,
                max_block_depth=0,
            )
        except TypeError:
            # Backward-compatible fallback for plugins with narrower signatures.
            extracted_units = plugin.extract_code_units(source_code, file_path)
        except Exception:
            logger.debug(
                "Failed to extract similarity snippets for %s",
                file_path,
                exc_info=True,
            )
            return snippets

        for unit in extracted_units:
            code = getattr(unit, "code", None)
            if isinstance(code, str):
                _add(code)

        return snippets

    def _find_similar_for_content(
        self,
        *,
        plugin: Any,
        source_code: str,
        file_path: str,
        min_node_count: int,
    ) -> list[SimilarityResult]:
        """
        Find similarity results for source content plus extracted code units.

        This improves write/edit checks for languages where indexed units are
        function-level symbols while authored content may include wrappers.
        """
        by_entry_id: dict[str, SimilarityResult] = {}
        snippets = self._candidate_similarity_snippets(
            plugin=plugin,
            source_code=source_code,
            file_path=file_path,
        )

        for snippet in snippets:
            for entry in self.index.find_exact_matches(snippet, language=plugin.language_id):
                candidate = SimilarityResult(entry=entry, similarity_type="exact")
                by_entry_id[entry.id] = self._prefer_similarity_result(
                    by_entry_id.get(entry.id),
                    candidate,
                )

            for result in self.index.find_similar(
                snippet,
                min_node_count=min_node_count,
                language=plugin.language_id,
            ):
                entry_id = result.entry.id
                by_entry_id[entry_id] = self._prefer_similarity_result(
                    by_entry_id.get(entry_id),
                    result,
                )

        ordered = list(by_entry_id.values())
        ordered.sort(key=self._similarity_sort_key)
        return ordered

    def _semantic_alignment_hint(
        self,
        *,
        source_profile: SemanticProfile,
        candidate: SimilarityResult,
    ) -> tuple[float, str]:
        """Score and describe semantic alignment for a cross-language candidate."""
        language = candidate.entry.code_unit.language
        plugin = LanguageRegistry.get().get_plugin(language)
        if plugin is None:
            return 0.1, "semantic=unknown (no language plugin)"

        try:
            candidate_profile = plugin.extract_semantic_profile(
                candidate.entry.code_unit.code,
                file_path=candidate.entry.code_unit.file_path,
            )
            available, compatible, mismatches, matched, reason = self._semantic_compare_result(
                source_profile,
                candidate_profile,
            )
        except Exception:
            logger.debug(
                "Failed semantic assist profile extraction for %s",
                language,
                exc_info=True,
            )
            return 0.1, "semantic=unknown (profile extraction failed)"

        if not available:
            detail = reason if reason else "insufficient overlap"
            return 0.2, f"semantic=inconclusive ({detail})"
        if compatible:
            if matched:
                return 0.9, f"semantic=aligned ({', '.join(matched[:2])})"
            return 0.8, "semantic=aligned"
        mismatch = "; ".join(mismatches[:1]) if mismatches else "signal mismatch"
        return 0.35, f"semantic=mismatch ({mismatch})"

    def _cross_language_assist_text(
        self,
        *,
        source_code: str,
        source_language: str,
        results: list[SimilarityResult],
        max_items: int = 3,
    ) -> str:
        """Build non-blocking cross-language assist guidance."""
        if not results:
            return ""

        source_plugin = LanguageRegistry.get().get_plugin(source_language)
        if source_plugin is None:
            return ""

        source_profile = source_plugin.extract_semantic_profile(source_code)
        similarity_rank = {"exact": 0, "high": 1, "partial": 2}
        ranked: list[tuple[int, float, SimilarityResult, str]] = []

        seen_entries: set[str] = set()
        for result in results:
            entry = result.entry
            if entry.id in seen_entries:
                continue
            seen_entries.add(entry.id)
            semantic_score, semantic_hint = self._semantic_alignment_hint(
                source_profile=source_profile,
                candidate=result,
            )
            ranked.append(
                (
                    similarity_rank.get(result.similarity_type, 99),
                    -semantic_score,
                    result,
                    semantic_hint,
                )
            )

        ranked.sort(key=lambda item: (item[0], item[1]))
        selected = ranked[:max_items]
        if not selected:
            return ""

        lines = ["ASSIST: Cross-language structural matches found (non-blocking):"]
        for _priority, _score, result, semantic_hint in selected:
            entry = result.entry
            rel = self._relative_path(entry.code_unit.file_path)
            lines.append(
                f"  - [{entry.code_unit.language}] {rel}:{entry.code_unit.name} "
                f"({result.similarity_type}; {semantic_hint})"
            )
        lines.append("These hints never block writes/edits; use them only for reuse guidance.")
        return "\n".join(lines)

    @staticmethod
    def _structural_compare_result(g1: nx.DiGraph, g2: nx.DiGraph) -> tuple[str, str]:
        """Return structural comparison kind and user-facing message."""
        h1 = weisfeiler_leman_hash(g1)
        h2 = weisfeiler_leman_hash(g2)

        is_isomorphic = nx.is_isomorphic(g1, g2, node_match=node_match)
        if is_isomorphic:
            return "equivalent", "EQUIVALENT: The code snippets are structurally identical."
        if h1 == h2:
            return "similar", "SIMILAR: Same hash but not fully isomorphic (rare edge case)."
        if fingerprints_compatible(structural_fingerprint(g1), structural_fingerprint(g2)):
            return "similar", "SIMILAR: Compatible structure but not identical."
        return "different", "DIFFERENT: The code snippets are structurally different."

    @staticmethod
    def _semantic_compare_result(
        profile1: SemanticProfile,
        profile2: SemanticProfile,
    ) -> tuple[bool, bool, list[str], list[str], str]:
        """Compare semantic profiles.

        Returns:
            available: whether enough overlapping signals were found
            compatible: whether overlapping signals agree
            mismatches: mismatch descriptions
            matched: matched signal keys
            reason: context for inconclusive results
        """
        map1 = profile1.signal_map()
        map2 = profile2.signal_map()
        shared_keys = sorted(set(map1) & set(map2))

        mismatches: list[str] = []
        matched: list[str] = []
        compared_any = False

        for key in shared_keys:
            signal1 = map1[key]
            signal2 = map2[key]
            confidence = min(signal1.confidence, signal2.confidence)
            if confidence < 0.4:
                continue
            compared_any = True
            if signal1.value == signal2.value:
                matched.append(key)
            else:
                mismatches.append(f"{key} ({signal1.value} vs {signal2.value})")

        if compared_any:
            return True, not mismatches, mismatches, matched, ""

        combined_notes = [*profile1.notes, *profile2.notes]
        note_suffix = f" ({combined_notes[0]})" if combined_notes else ""
        return (
            False,
            False,
            [],
            [],
            "insufficient overlapping semantic signals" f"{note_suffix}",
        )

    def compare(
        self,
        code1: str,
        code2: str,
        language: str = "python",
        semantic_mode: str = "off",
    ) -> ToolResult:
        """Compare two code snippets with structural and optional semantic checks."""
        mode = semantic_mode.strip().lower()
        if mode not in _SEMANTIC_MODES:
            return ToolResult(
                "Invalid semantic_mode. Supported values: off, annotate, differentiate."
            )

        plugin = LanguageRegistry.get().get_plugin(language)
        if plugin is None:
            return ToolResult(f"Unsupported language '{language}': no registered language plugin.")

        g1 = plugin.source_to_graph(code1)
        g2 = plugin.source_to_graph(code2)
        structural_kind, structural_message = self._structural_compare_result(g1, g2)

        if mode == "off":
            return ToolResult(structural_message)

        profile1 = plugin.extract_semantic_profile(code1)
        profile2 = plugin.extract_semantic_profile(code2)
        available, compatible, mismatches, matched, reason = self._semantic_compare_result(
            profile1, profile2
        )

        if not available:
            guidance = (
                "For stronger compile-time differentiation, run "
                "astrograph_lsp_setup(mode='inspect', language='"
                f"{language}')."
            )
            if mode == "annotate":
                return ToolResult(
                    f"{structural_message} SEMANTIC_INCONCLUSIVE: {reason}. {guidance}"
                )
            return ToolResult(
                "INCONCLUSIVE: differentiate mode could not reach high-confidence overlap. "
                f"{reason}. {guidance}"
            )

        if compatible:
            semantic_match_text = (
                f"SEMANTIC_MATCH: compared {', '.join(matched)}." if matched else "SEMANTIC_MATCH."
            )
            if mode == "annotate":
                return ToolResult(f"{structural_message} {semantic_match_text}")
            if structural_kind == "equivalent":
                return ToolResult(
                    "EQUIVALENT: The code snippets are structurally identical and semantically aligned."
                )
            return ToolResult(f"{structural_message} {semantic_match_text}")

        mismatch_text = "; ".join(mismatches[:3])
        if structural_kind == "equivalent":
            if mode == "annotate":
                return ToolResult(
                    "EQUIVALENT (STRUCTURE) but SEMANTIC_MISMATCH: " f"{mismatch_text}."
                )
            return ToolResult(
                "DIFFERENT: Structurally equivalent snippets diverge semantically "
                f"({mismatch_text})."
            )

        return ToolResult(f"{structural_message} SEMANTIC_MISMATCH: {mismatch_text}.")

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

    @staticmethod
    def _normalize_hash_input(wl_hash: str | list[str] | None) -> str | list[str] | ToolResult:
        """Normalize wl_hash input: str stays str, list stays list, None returns error."""
        if wl_hash is None:
            return ToolResult("Error: wl_hash is required.")
        if isinstance(wl_hash, list):
            return wl_hash
        return wl_hash

    def _suppress_dispatch(self, wl_hash: str | list[str] | None, suppress: bool) -> ToolResult:
        """Shared dispatch for suppress/unsuppress."""
        normalized = self._normalize_hash_input(wl_hash)
        if isinstance(normalized, ToolResult):
            return normalized
        if isinstance(normalized, list):
            return self._batch_toggle_suppression(normalized, suppress=suppress)
        return self._toggle_suppression(normalized, suppress=suppress)

    @_requires_index
    def suppress(self, wl_hash: str | list[str] | None = None) -> ToolResult:
        """Suppress one or more hashes. Accepts a string or list of strings."""
        return self._suppress_dispatch(wl_hash, suppress=True)

    @_requires_index
    def unsuppress(self, wl_hash: str | list[str] | None = None) -> ToolResult:
        """Unsuppress one or more hashes. Accepts a string or list of strings."""
        return self._suppress_dispatch(wl_hash, suppress=False)

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
            parts.append("Reminder: suppression hides duplicates — refactoring eliminates them.")
            parts.append("Run analyze to refresh.")
        return ToolResult(prefix + " ".join(parts))

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

    # --- MCP Resource readers ---

    def read_resource_status(self) -> str:
        """Return server status text for the MCP resource."""
        return self.status().text

    def read_resource_analysis(self) -> str:
        """Return the latest analysis report text for the MCP resource."""
        if self._last_indexed_path is None:
            return "No codebase indexed yet."
        persistence_path = _get_persistence_path(self._last_indexed_path)
        if not persistence_path.exists():
            return "No analysis reports available."
        reports = sorted(persistence_path.glob("analysis_report_*.txt"), reverse=True)
        if not reports:
            return "No analysis reports available."
        return reports[0].read_text()

    def read_resource_suppressions(self) -> str:
        """Return suppression list text for the MCP resource."""
        return self.list_suppressions().text

    def status(self) -> ToolResult:
        """Return current server status without blocking."""
        if not self._bg_index_done.is_set():
            entry_count = len(self.index.entries)
            return ToolResult(f"Status: indexing ({entry_count} entries so far)")
        if not self.index.entries:
            return ToolResult(
                "Status: idle (no codebase indexed). "
                "Next: call index_codebase(path=...) and astrograph_lsp_setup(mode='inspect')."
            )
        stats = self.index.get_stats()
        return ToolResult(
            f"Status: ready ({stats['function_entries']} code units, "
            f"{stats['indexed_files']} files). "
            "Next: call astrograph_lsp_setup(mode='inspect') for guided LSP setup actions."
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

    @staticmethod
    def _is_docker_runtime() -> bool:
        """Return whether the server appears to run inside Docker."""
        return Path("/.dockerenv").exists()

    @staticmethod
    def _default_install_command(language_id: str) -> list[str] | None:
        """Return a known install command for bundled subprocess servers."""
        if language_id == "python":
            return ["python3", "-m", "pip", "install", "python-lsp-server>=1.11"]
        if language_id in ("javascript_lsp", "typescript_lsp"):
            return ["npm", "install", "-g", "typescript", "typescript-language-server"]
        return None

    @staticmethod
    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        """Return values with duplicates removed while preserving first occurrence."""
        seen: set[str] = set()
        result: list[str] = []
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            result.append(value)
        return result

    def _attach_candidate_commands(self, status: dict[str, Any]) -> list[str]:
        """Build endpoint candidates for attach-based language plugins."""
        endpoint = parse_attach_endpoint(status.get("default_command"))
        if endpoint is None:
            endpoint = parse_attach_endpoint(status.get("effective_command"))
        if endpoint is None:
            return []

        if endpoint["transport"] == "unix":
            return [f"unix://{endpoint['path']}"]

        port = int(endpoint["port"])
        candidates = [
            f"tcp://127.0.0.1:{port}",
            f"tcp://localhost:{port}",
        ]
        if self._is_docker_runtime():
            candidates = [
                f"tcp://host.docker.internal:{port}",
                f"tcp://gateway.docker.internal:{port}",
                *candidates,
            ]
        return self._dedupe_preserve_order(candidates)

    @staticmethod
    def _server_bridge_info(language_id: str, candidate: str) -> dict[str, Any] | None:
        """Return structured bridge info for attach-based adapters."""
        endpoint = parse_attach_endpoint(candidate)
        if endpoint is None or endpoint["transport"] != "tcp":
            return None

        port = int(endpoint["port"])
        info: dict[str, Any] = {"requires": ["socat"], "port": port}
        if language_id in {"c_lsp", "cpp_lsp"}:
            info["server_binary"] = "clangd"
            info["server_binary_alternatives"] = ["ccls"]
            info["shared_with"] = "c_lsp" if language_id == "cpp_lsp" else "cpp_lsp"
            info["socat_command"] = (
                f'socat "TCP-LISTEN:{port},bind=0.0.0.0,reuseaddr,fork" ' '"EXEC:clangd",stderr'
            )
            info["install_hints"] = {
                "macos": "brew install llvm",
                "linux": "apt-get install -y clangd",
            }
        elif language_id == "java_lsp":
            info["server_binary"] = "jdtls"
            info["socat_command"] = (
                f'socat "TCP-LISTEN:{port},bind=0.0.0.0,reuseaddr,fork" '
                '"EXEC:jdtls -data /tmp/jdtls-workspace",stderr'
            )
            info["install_hints"] = {
                "macos": "brew install jdtls",
                "linux": "See https://github.com/eclipse-jdtls/eclipse.jdt.ls#installation",
            }
        else:
            return None
        return info

    def _build_lsp_recommended_actions(
        self,
        *,
        statuses: list[dict[str, Any]],
        scope_language: str | None = None,
    ) -> list[dict[str, Any]]:
        """Build actionable setup steps that AI agents can execute directly."""
        missing = [status for status in statuses if not status.get("available")]
        missing_required = [status for status in missing if status.get("required", True)]
        scoped = bool(scope_language)
        all_missing_optional = bool(missing) and not missing_required
        has_missing_cpp = any(str(status.get("language")) == "cpp_lsp" for status in missing)

        def _with_scope(arguments: dict[str, Any]) -> dict[str, Any]:
            if not scoped:
                return arguments
            return {**arguments, "language": scope_language}

        def _follow_up_auto_bind_arguments(language_id: str) -> dict[str, Any]:
            if scoped:
                return _with_scope({"mode": "auto_bind"})
            return {"mode": "auto_bind", "language": language_id}

        if not missing:
            return [
                {
                    "id": "verify_lsp_setup",
                    "priority": "low",
                    "title": "Re-check language server health after environment changes",
                    "tool": "astrograph_lsp_setup",
                    "arguments": _with_scope({"mode": "inspect"}),
                }
            ]

        actions: list[dict[str, Any]] = []
        if not scoped and all_missing_optional and has_missing_cpp:
            actions.append(
                {
                    "id": "focus_cpp_lsp",
                    "priority": "high",
                    "title": "Resolve cpp_lsp first",
                    "why": (
                        "Multiple attach languages are unavailable. Resolve them one at a time, "
                        "starting with C++. Run the host_search_commands, then auto_bind."
                    ),
                    "tool": "astrograph_lsp_setup",
                    "arguments": {"mode": "inspect", "language": "cpp_lsp"},
                }
            )
        actions.append(
            {
                "id": "auto_bind_missing",
                "priority": "high" if missing_required else "medium",
                "title": "Auto-bind any reachable language servers",
                "why": (
                    f"{len(missing)} language server(s) are unreachable. Execute this action to probe "
                    "all candidate endpoints and bind any that respond."
                ),
                "tool": "astrograph_lsp_setup",
                "arguments": _with_scope({"mode": "auto_bind"}),
            }
        )
        host_alias_action_added = False

        spec_by_language = {spec.language_id: spec for spec in bundled_lsp_specs()}
        for status in missing:
            language = str(status["language"])
            required = bool(status.get("required", True))
            priority = "high" if required else "medium"
            transport = str(status.get("transport", "subprocess"))
            verification = status.get("verification", {})
            if not isinstance(verification, dict):
                verification = {}
            verification_state = str(
                status.get("verification_state") or verification.get("state") or "unknown"
            )
            compile_commands = status.get("compile_commands")
            if not isinstance(compile_commands, dict):
                compile_commands = verification.get("compile_commands")
            if not isinstance(compile_commands, dict):
                compile_commands = None

            if transport == "subprocess":
                effective_command = parse_command(status.get("effective_command"))
                binary = effective_command[0] if effective_command else "<binary>"
                search_commands = [f"which {binary}"]
                if language == "python":
                    search_commands.append("python3 -m pylsp --help")
                if language == "javascript_lsp":
                    search_commands.extend(
                        [
                            "which typescript-language-server",
                            "npm list -g typescript-language-server",
                        ]
                    )

                actions.append(
                    {
                        "id": f"search_{language}",
                        "priority": priority,
                        "title": f"Search host for a working {language} command",
                        "language": language,
                        "host_search_commands": search_commands,
                        "follow_up_tool": "astrograph_lsp_setup",
                        "follow_up_arguments": _follow_up_auto_bind_arguments(language),
                    }
                )

                if install_command := self._default_install_command(language):
                    actions.append(
                        {
                            "id": f"install_{language}",
                            "priority": priority,
                            "title": f"Install missing {language} language server",
                            "language": language,
                            "shell_command": format_command(install_command),
                            "follow_up_tool": "astrograph_lsp_setup",
                            "follow_up_arguments": _follow_up_auto_bind_arguments(language),
                        }
                    )

                if status.get("effective_source") == "env":
                    spec = spec_by_language.get(language)
                    if spec is not None:
                        actions.append(
                            {
                                "id": f"fix_env_override_{language}",
                                "priority": priority,
                                "title": f"Fix unavailable env override for {language}",
                                "language": language,
                                "env_var": spec.command_env_var,
                                "note": (
                                    f"{spec.command_env_var} currently resolves to an "
                                    "unavailable command."
                                ),
                            }
                        )
                continue

            if verification_state == "reachable_only":
                actions.append(
                    {
                        "id": f"verify_{language}_protocol",
                        "priority": "high" if language == "cpp_lsp" else priority,
                        "title": f"Replace non-verified endpoint for {language}",
                        "language": language,
                        "note": (
                            verification.get("reason")
                            or "Endpoint accepts TCP connections but did not pass LSP verification."
                        ),
                        "host_search_commands": [
                            "which clangd",
                            "which ccls",
                            "clangd --version",
                            "ccls --version",
                        ]
                        if language in {"c_lsp", "cpp_lsp"}
                        else [],
                        "follow_up_tool": "astrograph_lsp_setup",
                        "follow_up_arguments": _follow_up_auto_bind_arguments(language),
                    }
                )

            if (
                language in {"c_lsp", "cpp_lsp"}
                and compile_commands is not None
                and not bool(compile_commands.get("valid"))
            ):
                paths = compile_commands.get("paths")
                known_paths = [str(path) for path in paths] if isinstance(paths, list) else []
                actions.append(
                    {
                        "id": f"ensure_compile_commands_{language}",
                        "priority": "high" if language == "cpp_lsp" else priority,
                        "title": f"Generate a valid compile_commands.json for {language}",
                        "language": language,
                        "why": (
                            compile_commands.get("reason")
                            or "C/C++ language servers require compile_commands.json for "
                            "production-grade type and operator resolution."
                        ),
                        "host_search_commands": [
                            "find . -maxdepth 4 -name compile_commands.json",
                        ],
                        "shell_commands": [
                            "cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                            "ln -sf build/compile_commands.json compile_commands.json",
                        ],
                        "known_paths": known_paths,
                        "follow_up_tool": "astrograph_lsp_setup",
                        "follow_up_arguments": _follow_up_auto_bind_arguments(language),
                    }
                )

            if self._is_docker_runtime() and not host_alias_action_added:
                actions.append(
                    {
                        "id": "ensure_docker_host_alias",
                        "priority": "high",
                        "title": "Ensure Docker can reach host.docker.internal",
                        "why": (
                            "Attach endpoints for C/C++/Java usually run on the host. "
                            "Add a host alias so tcp://host.docker.internal:<port> is routable."
                        ),
                        "docker_run_args": [
                            "--add-host",
                            "host.docker.internal:host-gateway",
                        ],
                        "follow_up_tool": "astrograph_lsp_setup",
                        "follow_up_arguments": _with_scope({"mode": "inspect"}),
                    }
                )
                host_alias_action_added = True

            candidates = self._attach_candidate_commands(status)
            endpoint = parse_attach_endpoint(candidates[0]) if candidates else None
            port_hint = (
                int(endpoint["port"]) if endpoint and endpoint["transport"] == "tcp" else None
            )
            bridge_info = self._server_bridge_info(language, candidates[0]) if candidates else None
            endpoint_search_commands: list[str] = []
            if port_hint is not None:
                endpoint_search_commands = [
                    f"lsof -nP -iTCP:{port_hint} -sTCP:LISTEN",
                    f"ss -ltnp | rg ':{port_hint}'",
                ]
            if bridge_info:
                endpoint_search_commands.append(f"which {bridge_info['server_binary']}")

            action: dict[str, Any] = {
                "id": f"discover_{language}_endpoint",
                "priority": priority,
                "title": f"Discover a reachable attach endpoint for {language}",
                "why": (
                    f"No server is listening on the expected {language} port. "
                    "Run the host_search_commands on the host machine to check, "
                    "then provide the working endpoint via auto_bind with observations."
                ),
                "language": language,
                "candidate_endpoints": candidates,
                "host_search_commands": endpoint_search_commands,
            }
            if candidates:
                follow_up_arguments = {
                    **_follow_up_auto_bind_arguments(language),
                    "observations": [{"language": language, "command": candidates[0]}],
                }

                action["follow_up_tool"] = "astrograph_lsp_setup"
                action["follow_up_arguments"] = follow_up_arguments
                if bridge_info:
                    action["server_bridge_info"] = bridge_info
            actions.append(action)

        actions.append(
            {
                "id": "verify_lsp_setup",
                "priority": "high" if missing_required else "medium",
                "title": "Verify missing_required_languages is empty",
                "tool": "astrograph_lsp_setup",
                "arguments": _with_scope({"mode": "inspect"}),
            }
        )
        return actions

    def _inject_lsp_setup_guidance(self, payload: dict[str, Any], *, workspace: Path) -> None:
        """Attach agent-friendly setup guidance to lsp_setup responses."""
        statuses_value = payload.get("servers")
        if not isinstance(statuses_value, list):
            statuses_value = payload.get("statuses")
        statuses = (
            [status for status in statuses_value if isinstance(status, dict)]
            if statuses_value
            else []
        )
        if not statuses:
            statuses = collect_lsp_statuses(workspace)

        missing = [status["language"] for status in statuses if not status.get("available")]
        missing_required = [
            status["language"]
            for status in statuses
            if status.get("required", True) and not status.get("available")
        ]

        payload["missing_languages"] = missing
        payload["missing_required_languages"] = missing_required
        payload["bindings"] = load_lsp_bindings(workspace)
        payload["execution_context"] = "docker" if self._is_docker_runtime() else "host"
        scope_language = payload.get("scope_language")
        if not isinstance(scope_language, str):
            scope_language = None

        recommended_actions = self._build_lsp_recommended_actions(
            statuses=statuses,
            scope_language=scope_language,
        )
        cpp_reachable_only = any(
            str(status.get("language")) == "cpp_lsp"
            and str(status.get("verification_state") or "") == "reachable_only"
            for status in statuses
        )
        scope_status = (
            next((status for status in statuses if status.get("language") == scope_language), None)
            if scope_language
            else None
        )
        attach_scope_ready = bool(
            scope_status
            and not missing
            and str(scope_status.get("transport", "subprocess")) in {"tcp", "unix"}
        )
        if attach_scope_ready and scope_language:
            recommended_actions.extend(
                [
                    {
                        "id": f"verify_{scope_language}_analysis",
                        "priority": "medium",
                        "title": f"Run duplicate analysis after enabling {scope_language}",
                        "why": (
                            "Confirms the active index is healthy after LSP setup changes "
                            "and generates a current duplicate report."
                        ),
                        "tool": "astrograph_analyze",
                        "arguments": {"auto_reindex": True},
                    },
                    {
                        "id": f"rebaseline_after_{scope_language}_binding",
                        "priority": "medium",
                        "title": "If this language was bound after startup, rebuild baseline",
                        "why": (
                            "Binding changes do not retroactively update already-indexed files. "
                            "Recompute to apply extraction across the full workspace."
                        ),
                        "tool": "astrograph_metadata_recompute_baseline",
                        "arguments": {},
                        "note": (
                            "metadata_recompute_baseline clears lsp_bindings.json. "
                            "Run follow_up_steps immediately after recompute."
                        ),
                        "follow_up_steps": [
                            {
                                "tool": "astrograph_lsp_setup",
                                "arguments": {"mode": "inspect", "language": scope_language},
                            },
                            {
                                "tool": "astrograph_lsp_setup",
                                "arguments": {"mode": "auto_bind", "language": scope_language},
                            },
                            {
                                "tool": "astrograph_lsp_setup",
                                "arguments": {"mode": "inspect", "language": scope_language},
                            },
                        ],
                    },
                ]
            )
        payload["recommended_actions"] = recommended_actions

        inspect_arguments: dict[str, Any] = {"mode": "inspect"}
        auto_bind_arguments: dict[str, Any] = {"mode": "auto_bind"}
        if scope_language:
            inspect_arguments["language"] = scope_language
            auto_bind_arguments["language"] = scope_language

        payload["resolution_loop"] = [
            {"tool": "astrograph_lsp_setup", "arguments": inspect_arguments},
            {"tool": "astrograph_lsp_setup", "arguments": auto_bind_arguments},
            {"tool": "astrograph_lsp_setup", "arguments": inspect_arguments},
        ]
        if missing_required:
            payload["agent_directive"] = (
                "Do not stop at diagnostics. Execute recommended_actions until "
                "missing_required_languages is empty."
            )
        elif cpp_reachable_only:
            payload["agent_directive"] = (
                "cpp_lsp endpoint is reachable but not production-verified. "
                "Treat it as unavailable and execute recommended_actions "
                "(real C++ LSP endpoint, such as clangd/ccls, + valid compile_commands.json)."
            )
        elif missing:
            missing_names = ", ".join(missing)
            payload["agent_directive"] = (
                f"{len(missing)} attach language(s) ({missing_names}) are unavailable. "
                "Execute recommended_actions now: run the host_search_commands, "
                "then call auto_bind with any discovered endpoints."
            )
        elif attach_scope_ready and scope_language:
            payload["agent_directive"] = (
                f"{scope_language} endpoint is reachable. "
                "Run recommended_actions to verify analysis coverage. "
                "If you recompute baseline, re-run auto_bind for this language afterward."
            )
        else:
            payload["agent_directive"] = (
                "All bundled language servers are reachable. "
                "Re-run inspect after environment changes."
            )

        if missing and self._is_docker_runtime():
            payload["observation_note"] = (
                "This MCP server runs inside Docker. When providing observations, "
                "use host.docker.internal (not 127.0.0.1) to reach host-side servers."
            )

    def _lsp_setup_result(self, payload: dict[str, Any]) -> ToolResult:
        """Serialize structured LSP setup responses as JSON."""
        return ToolResult(json.dumps(payload, indent=2, sort_keys=True))

    def lsp_setup(
        self,
        mode: str = "inspect",
        language: str | None = None,
        command: Sequence[str] | str | None = None,
        observations: list[dict[str, Any]] | None = None,
        validation_mode: str | None = None,
        compile_db_path: str | None = None,
        project_root: str | None = None,
    ) -> ToolResult:
        """Inspect and configure deterministic command bindings for bundled LSP plugins."""
        workspace = self._lsp_setup_workspace()
        normalized_mode = (mode or "inspect").strip().lower()
        known_languages = [
            spec.language_id for spec in sorted(bundled_lsp_specs(), key=lambda s: s.language_id)
        ]

        def _validate_language(mode_name: str, *, required: bool = False) -> ToolResult | None:
            if not language:
                if required:
                    return self._lsp_setup_result(
                        {
                            "ok": False,
                            "mode": mode_name,
                            "error": f"'language' is required when mode='{mode_name}'",
                            "supported_languages": known_languages,
                        }
                    )
                return None

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
            if validation_error := _validate_language(normalized_mode):
                return validation_error

            statuses = collect_lsp_statuses(
                workspace,
                validation_mode=validation_mode,
                compile_db_path=compile_db_path,
                project_root=project_root,
            )
            if language:
                statuses = [status for status in statuses if status["language"] == language]
            missing = [status["language"] for status in statuses if not status["available"]]
            missing_required = [
                status["language"]
                for status in statuses
                if status.get("required", True) and not status["available"]
            ]
            cpp_reachable_only = any(
                str(status.get("language")) == "cpp_lsp"
                and str(status.get("verification_state") or "") == "reachable_only"
                for status in statuses
            )
            payload: dict[str, Any] = {
                "ok": True,
                "mode": normalized_mode,
                "workspace": str(workspace),
                "bindings_path": str(lsp_bindings_path(workspace)),
                "bindings": load_lsp_bindings(workspace),
                "servers": statuses,
                "missing_languages": missing,
                "missing_required_languages": missing_required,
                "supported_languages": known_languages,
                "language_variant_policy": (
                    language_variant_policy(language) if language else language_variant_policy()
                ),
            }
            if language:
                payload["scope_language"] = language

            if missing_required:
                if language:
                    payload["next_step"] = (
                        "Call astrograph_lsp_setup with mode='auto_bind' and this language. "
                        "If still missing, provide observations with language + command."
                    )
                else:
                    payload["next_step"] = (
                        "Call astrograph_lsp_setup with mode='auto_bind'. "
                        "If still missing, provide observations with language + command."
                    )
            elif cpp_reachable_only:
                payload["next_step"] = (
                    "cpp_lsp is reachable but not production-verified. "
                    "Use a real C++ LSP endpoint (for example clangd/ccls) and a valid "
                    "compile_commands.json, "
                    "then re-run auto_bind and inspect."
                )
            elif missing:
                if not language and "cpp_lsp" in missing:
                    payload["next_step"] = (
                        f"{len(missing)} attach endpoint(s) unreachable. "
                        "Start by running the host_search_commands from recommended_actions "
                        "to check if language servers are already running on the host. "
                        "Then call astrograph_lsp_setup(mode='auto_bind') with any discovered endpoints as observations."
                    )
                else:
                    payload["next_step"] = (
                        f"{len(missing)} attach endpoint(s) unreachable. "
                        "Run the host_search_commands from recommended_actions to discover running servers, "
                        "then call astrograph_lsp_setup(mode='auto_bind') with results as observations."
                    )

            if missing:
                payload["observation_format"] = {
                    "language": language or "cpp_lsp",
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
                if not language and "cpp_lsp" in missing:
                    payload["focus_hint"] = (
                        "Start by resolving one language at a time. "
                        "Run astrograph_lsp_setup(mode='inspect', language='cpp_lsp') "
                        "and execute its recommended_actions before moving to the next."
                    )
            if cpp_reachable_only:
                payload["production_gate"] = {
                    "language": "cpp_lsp",
                    "state": "reachable_only",
                    "requirement": (
                        "In production mode, cpp_lsp is treated as unavailable until endpoint "
                        "handshake, semantic probe, and compile_commands.json checks pass."
                    ),
                }
            self._inject_lsp_setup_guidance(payload, workspace=workspace)
            return self._lsp_setup_result(payload)

        if normalized_mode == "auto_bind":
            if validation_error := _validate_language(normalized_mode):
                return validation_error

            scope_languages = [language] if language else None
            outcome = auto_bind_missing_servers(
                workspace=workspace,
                observations=observations,
                languages=scope_languages,
                validation_mode=validation_mode,
                compile_db_path=compile_db_path,
                project_root=project_root,
            )
            if outcome["changes"]:
                LanguageRegistry.reset()
            outcome["bindings"] = load_lsp_bindings(workspace)
            outcome.update(
                {
                    "ok": True,
                    "mode": normalized_mode,
                    "supported_languages": known_languages,
                }
            )
            if language:
                outcome["scope_language"] = language
            self._inject_lsp_setup_guidance(outcome, workspace=workspace)
            return self._lsp_setup_result(outcome)

        if normalized_mode == "bind":
            if validation_error := _validate_language(normalized_mode, required=True):
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
            payload = {
                "ok": True,
                "mode": normalized_mode,
                "language": target_language,
                "workspace": str(workspace),
                "bindings_path": str(path),
                "bindings": load_lsp_bindings(workspace),
                "saved_command": saved_command,
                "saved_command_text": format_command(saved_command),
                "available": probe["available"],
                "executable": probe["executable"],
                "status": status,
            }
            self._inject_lsp_setup_guidance(payload, workspace=workspace)
            return self._lsp_setup_result(payload)

        if normalized_mode == "unbind":
            if validation_error := _validate_language(normalized_mode, required=True):
                return validation_error

            target_language = cast(str, language)
            removed, path = unset_lsp_binding(target_language, workspace)
            LanguageRegistry.reset()
            status = _status_for_language(target_language)
            payload = {
                "ok": True,
                "mode": normalized_mode,
                "language": target_language,
                "workspace": str(workspace),
                "bindings_path": str(path),
                "bindings": load_lsp_bindings(workspace),
                "removed": removed,
                "status": status,
            }
            self._inject_lsp_setup_guidance(payload, workspace=workspace)
            return self._lsp_setup_result(payload)

        return self._lsp_setup_result(
            {
                "ok": False,
                "mode": normalized_mode,
                "error": "Invalid mode",
                "supported_modes": ["inspect", "auto_bind", "bind", "unbind"],
                "supported_languages": known_languages,
            }
        )

    def generate_ignore(self) -> ToolResult:
        """Generate a .astrographignore file with sensible defaults."""
        if not self._last_indexed_path:
            return ToolResult(
                "No indexed codebase. Run index_codebase first to establish a workspace root."
            )

        root = Path(self._last_indexed_path)
        ignore_path = root / ASTROGRAPHIGNORE_FILENAME

        if ignore_path.exists():
            return ToolResult(
                f".astrographignore already exists at {ignore_path}. "
                "Edit it manually to change ignore patterns."
            )

        content = _generate_default_ignore_content()
        result = self._write_file(
            str(ignore_path),
            content,
            summary=f"Created {ASTROGRAPHIGNORE_FILENAME} with default ignore patterns. "
            "Re-index to apply.",
            display_path=ASTROGRAPHIGNORE_FILENAME,
        )
        return result

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
        bindings_removed = False
        if self._last_indexed_path:
            persistence_path = _get_persistence_path(self._last_indexed_path)
            bindings_removed = (persistence_path / "lsp_bindings.json").exists()
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
            message = "Erased all metadata. Server reset to idle state."
            if bindings_removed:
                message += (
                    " LSP bindings were removed. Next: call astrograph_lsp_setup(mode='inspect') "
                    "and re-run auto_bind for required languages."
                )
            return ToolResult(message)
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
        bindings_removed = (_get_persistence_path(path) / "lsp_bindings.json").exists()

        # Erase everything
        self.metadata_erase()

        # Re-index from scratch
        result = self.index_codebase(path)
        message = f"Baseline recomputed from scratch.\n{result.text}"
        if bindings_removed:
            message += (
                "\nLSP bindings were reset during recompute. Next: call "
                "astrograph_lsp_setup(mode='inspect') and re-run auto_bind for required languages."
            )
        return ToolResult(message)

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

        report = self.index.get_staleness_report(path, ignore_spec=self._ignore_spec)

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
            results = self._find_similar_for_content(
                plugin=plugin,
                source_code=content,
                file_path=file_path,
                min_node_count=self._CHECK_MIN_STATEMENTS,
            )
            same_language, cross_language = self._split_similarity_results_by_language(
                results,
                plugin.language_id,
            )
            assist = self._cross_language_assist_text(
                source_code=content,
                source_language=plugin.language_id,
                results=cross_language,
            )

            exact = [r for r in same_language if r.similarity_type == "exact"]
            high = [r for r in same_language if r.similarity_type == "high"]

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
            if assist:
                warning += assist + "\n\n"

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
            results = self._find_similar_for_content(
                plugin=plugin,
                source_code=new_string,
                file_path=file_path,
                min_node_count=self._CHECK_MIN_STATEMENTS,
            )
            same_language, cross_language = self._split_similarity_results_by_language(
                results,
                plugin.language_id,
            )
            assist = self._cross_language_assist_text(
                source_code=new_string,
                source_language=plugin.language_id,
                results=cross_language,
            )

            exact = [r for r in same_language if r.similarity_type == "exact"]
            high = [r for r in same_language if r.similarity_type == "high"]

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
            if assist:
                warning += assist + "\n\n"

        # Read the file
        try:
            with open(file_path, encoding="utf-8") as f:
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

    @staticmethod
    def _is_mutating_tool_call(name: str, arguments: dict[str, Any]) -> bool:
        """Return whether a tool call changes shared server state."""
        if name in _MUTATING_TOOL_NAMES:
            return True
        if name != "lsp_setup":
            return False

        mode = str(arguments.get("mode", "inspect")).strip().lower()
        return mode in _MUTATING_LSP_SETUP_MODES

    def _call_tool_unlocked(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call by name without synchronization."""
        if name == "index_codebase":
            return self.index_codebase(
                path=arguments["path"],
                recursive=arguments.get("recursive", True),
            )
        elif name == "set_workspace":
            return self.set_workspace(path=arguments["path"])
        elif name == "analyze":
            return self.analyze(
                auto_reindex=arguments.get("auto_reindex", True),
            )
        elif name == "check":
            return self.check(
                code=arguments["code"],
                language=arguments.get("language", "python"),
            )
        elif name == "compare":
            return self.compare(
                code1=arguments["code1"],
                code2=arguments["code2"],
                language=arguments.get("language", "python"),
                semantic_mode=arguments.get("semantic_mode", "off"),
            )
        elif name == "suppress":
            return self.suppress(wl_hash=arguments.get("wl_hash"))
        elif name == "unsuppress":
            return self.unsuppress(wl_hash=arguments.get("wl_hash"))
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
                validation_mode=arguments.get("validation_mode"),
                compile_db_path=arguments.get("compile_db_path"),
                project_root=arguments.get("project_root"),
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
        elif name == "generate_ignore":
            return self.generate_ignore()
        else:
            return ToolResult(f"Unknown tool: {name}")

    def call_tool(self, name: str, arguments: dict) -> ToolResult:
        """Dispatch a tool call by name."""
        safe_arguments: dict[str, Any] = arguments if arguments is not None else {}
        if self._is_mutating_tool_call(name, safe_arguments):
            with self._mutation_lock:
                return self._call_tool_unlocked(name, safe_arguments)
        return self._call_tool_unlocked(name, safe_arguments)

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
                import sys as _sys

                rusage = resource.getrusage(resource.RUSAGE_SELF)
                # macOS reports ru_maxrss in bytes, Linux reports in KB
                if _sys.platform == "darwin":
                    stats["process_rss_bytes"] = rusage.ru_maxrss
                else:
                    stats["process_rss_bytes"] = rusage.ru_maxrss * 1024
            except (ImportError, AttributeError):
                pass
            return stats
        return None
