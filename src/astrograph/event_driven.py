"""
Event-driven index with file watching and pre-computed analysis.

Keeps the index hot in memory, automatically updates on file changes,
and pre-computes analysis results so LLM tool calls return instantly.
"""

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from .cloud_detect import check_and_warn_cloud_sync, is_cloud_synced_path
from .context import CloseOnExitMixin
from .ignorefile import IgnoreSpec
from .index import CodeStructureIndex, DuplicateGroup, batch_hash_operation
from .persistence import SQLitePersistence

logger = logging.getLogger(__name__)

# Try to import watchdog components
try:
    from .watcher import HAS_WATCHDOG, FileWatcher
except ImportError:
    HAS_WATCHDOG = False
    FileWatcher = None  # type: ignore


def _make_file_event_handler(
    event_type: str, target_attr: str
) -> Callable[["EventDrivenIndex", str], None]:
    """Build a thin file-event handler bound to a target method name."""

    def _handler(self: "EventDrivenIndex", path: str) -> None:
        self._handle_file_event(path, event_type, getattr(self, target_attr))

    return _handler


class AnalysisCache:
    """
    Cache for pre-computed analysis results.

    Invalidated when the index changes, recomputed in background.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._valid = False
        self._exact_duplicates: list[DuplicateGroup] = []
        self._pattern_duplicates: list[DuplicateGroup] = []
        self._block_duplicates: list[DuplicateGroup] = []
        self._computed_at: float = 0

    def invalidate(self) -> None:
        """Mark cache as invalid."""
        with self._lock:
            self._valid = False

    def _read_attr(self, attr: str) -> Any:
        """Read an attribute while holding the cache lock."""
        with self._lock:
            return getattr(self, attr)

    def is_valid(self) -> bool:
        """Check if cache is valid."""
        with self._lock:
            return self._valid

    def get(self) -> tuple[list[DuplicateGroup], list[DuplicateGroup], list[DuplicateGroup]] | None:
        """Get cached results if valid."""
        with self._lock:
            if not self._valid:
                return None
            return (
                self._exact_duplicates.copy(),
                self._pattern_duplicates.copy(),
                self._block_duplicates.copy(),
            )

    def set(
        self,
        exact: list[DuplicateGroup],
        pattern: list[DuplicateGroup],
        blocks: list[DuplicateGroup],
    ) -> None:
        """Set cached results."""
        with self._lock:
            self._exact_duplicates = exact
            self._pattern_duplicates = pattern
            self._block_duplicates = blocks
            self._computed_at = time.time()
            self._valid = True

    @property
    def computed_at(self) -> float:
        """When the cache was last computed."""
        return cast(float, self._read_attr("_computed_at"))


class EventDrivenIndex(CloseOnExitMixin):
    """
    Event-driven code structure index.

    Features:
    - In-memory index always hot and ready
    - SQLite persistence for durability and incremental updates
    - Optional file watching for automatic updates
    - Pre-computed analysis cache for instant LLM responses
    """

    def __init__(
        self,
        persistence_path: str | Path | None = None,
        watch_enabled: bool = True,
        debounce_delay: float = 0.1,
        ignore_spec: IgnoreSpec | None = None,
    ) -> None:
        """
        Initialize the event-driven index.

        Args:
            persistence_path: Path to SQLite database (None = no persistence)
            watch_enabled: Enable file system watching (requires watchdog)
            debounce_delay: Seconds to wait before processing rapid file changes
            ignore_spec: Optional ignore patterns for file/directory exclusion
        """
        self._ignore_spec = ignore_spec
        # In-memory index (always hot)
        self.index = CodeStructureIndex()

        # SQLite persistence
        self._persistence: SQLitePersistence | None = None
        if persistence_path:
            self._persistence = SQLitePersistence(persistence_path)
            # Wire persistence to EntryStore for LRU eviction support
            self.index.entries.set_persistence(self._persistence)

        # File watching
        self._watch_enabled = watch_enabled and HAS_WATCHDOG
        self._watcher: FileWatcher | None = None
        self._debounce_delay = debounce_delay
        self._watched_path: Path | None = None

        # Analysis cache
        self._cache = AnalysisCache()

        # Background processing
        self._bg_lock = threading.Lock()
        self._bg_thread: threading.Thread | None = None
        self._shutdown = threading.Event()

        # Stats
        self._file_events_processed = 0
        self._cache_hits = 0
        self._cache_misses = 0

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def load_from_persistence(self) -> bool:
        """Load index from SQLite persistence. Returns True if loaded."""
        if self._persistence is None:
            return False

        loaded = self._persistence.load_full_index(self.index)
        if loaded:
            logger.info(f"Loaded {len(self.index.entries)} entries from persistence")
            self._cache.invalidate()
        return loaded

    def start_watching(self, root_path: str | Path) -> None:
        """Start watching a directory for file changes."""
        if not self._watch_enabled:
            logger.warning("File watching not available (watchdog not installed)")
            return

        if self._watcher is not None:
            self.stop_watching()

        root = Path(root_path).resolve()
        if not root.is_dir():
            logger.warning(f"Cannot watch non-directory: {root}")
            return

        self._watched_path = root
        self._watcher = FileWatcher(
            root_path=root,
            on_file_changed=self._on_file_changed,
            on_file_created=self._on_file_created,
            on_file_deleted=self._on_file_deleted,
            debounce_delay=self._debounce_delay,
            ignore_spec=self._ignore_spec,
        )
        self._watcher.start()
        logger.info(f"Started file watching: {root}")

    def stop_watching(self) -> None:
        """Stop watching for file changes."""
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher = None
            self._watched_path = None
            logger.info("Stopped file watching")

    def close(self) -> None:
        """Clean up resources."""
        self._shutdown.set()
        self.stop_watching()

        if self._bg_thread is not None:
            self._bg_thread.join(timeout=2.0)
            self._bg_thread = None

        if self._persistence is not None:
            self._persistence.close()
            self._persistence = None

    # =========================================================================
    # File Event Handlers
    # =========================================================================

    def _handle_file_event(
        self, path: str, event_type: str, handler: Callable[[str], None]
    ) -> None:
        """Generic file event handler to reduce duplication."""
        if self._shutdown.is_set():
            logger.debug(f"Ignoring file {event_type} during shutdown: {path}")
            return
        logger.debug(f"Processing file {event_type}: {path}")
        handler(path)
        self._file_events_processed += 1

    _on_file_changed = _make_file_event_handler("change", "_reindex_file")
    _on_file_created = _make_file_event_handler("creation", "_reindex_file")
    _on_file_deleted = _make_file_event_handler("deletion", "_remove_file")

    def _reindex_file(self, path: str) -> None:
        """Re-index a single file and persist incrementally."""
        if self._shutdown.is_set():
            return
        if not os.path.exists(path):
            return

        # Update in-memory index
        entries = self.index.index_file(path, include_blocks=True)

        # Persist incrementally
        if self._persistence is not None and path in self.index.file_metadata:
            metadata = self.index.file_metadata[path]
            self._persistence.save_file_entries(path, entries, metadata)

        self._invalidate_cache_and_recompute()

    def _remove_file(self, path: str) -> None:
        """Remove a file from the index and persistence."""
        if self._shutdown.is_set():
            return
        # Update in-memory index
        self.index.remove_file(path)

        # Remove from persistence
        if self._persistence is not None:
            self._persistence.delete_file_entries(path)

        self._invalidate_cache_and_recompute()

    # =========================================================================
    # Analysis Cache
    # =========================================================================

    def _schedule_analysis_recompute(self) -> None:
        """Schedule background analysis recomputation."""
        with self._bg_lock:
            if self._bg_thread is not None and self._bg_thread.is_alive():
                return  # Already running

            self._bg_thread = threading.Thread(
                target=self._recompute_analysis,
                daemon=True,
            )
            self._bg_thread.start()

    def _invalidate_cache_and_recompute(self) -> None:
        """Invalidate analysis cache and schedule recomputation."""
        self._cache.invalidate()
        self._schedule_analysis_recompute()

    def _maybe_start_watching(self, path: Path) -> None:
        """Start file watching when watcher support is enabled."""
        if self._watch_enabled:
            self.start_watching(path)

    def _recompute_analysis(self) -> None:
        """Recompute analysis in background thread."""
        if self._shutdown.is_set():
            logger.debug("Skipping analysis recompute during shutdown")
            return

        try:
            # Small delay to batch rapid changes
            time.sleep(0.05)

            if self._cache.is_valid():
                return  # Already valid (another thread computed it)

            # Compute analysis
            exact = self.index.find_all_duplicates(min_node_count=5)
            pattern = self.index.find_pattern_duplicates(min_node_count=5)
            blocks = self.index.find_block_duplicates(min_node_count=5)

            self._cache.set(exact, pattern, blocks)
            logger.debug("Analysis cache recomputed")

        except Exception:
            logger.exception("Error recomputing analysis cache")

    def get_cached_analysis(
        self,
        min_node_count: int = 5,
    ) -> tuple[list[DuplicateGroup], list[DuplicateGroup], list[DuplicateGroup]]:
        """
        Get analysis results, using cache if available.

        Returns (exact_duplicates, pattern_duplicates, block_duplicates).
        """
        # Try cache first
        cached = self._cache.get()
        if cached is not None:
            self._cache_hits += 1
            return cached

        # Cache miss - compute synchronously
        self._cache_misses += 1
        exact = self.index.find_all_duplicates(min_node_count=min_node_count)
        pattern = self.index.find_pattern_duplicates(min_node_count=min_node_count)
        blocks = self.index.find_block_duplicates(min_node_count=min_node_count)

        self._cache.set(exact, pattern, blocks)
        return exact, pattern, blocks

    # =========================================================================
    # Index Operations
    # =========================================================================

    def index_file(self, file_path: str | Path) -> int:
        """
        Index a single file with persistence, watching, and cache warming.

        Watches the file's parent directory so the index stays current.
        Returns the number of entries indexed.
        """
        file_path = Path(file_path).resolve()
        parent = file_path.parent

        self.index.clear()
        self.index.index_file(str(file_path))

        # Persist (full save — clears stale DB from prior sessions)
        if self._persistence is not None:
            self._persistence.save_full_index(self.index)

        self._maybe_start_watching(parent)
        self._invalidate_cache_and_recompute()

        logger.info(f"Indexed {len(self.index.entries)} entries from {file_path}")
        return len(self.index.entries)

    def index_directory(
        self,
        path: str | Path,
        recursive: bool = True,
        include_blocks: bool = True,
    ) -> int:
        """
        Index a directory and optionally start watching it.

        Returns the number of entries indexed.
        """
        path = Path(path).resolve()

        # Check for cloud-synced folders and warn
        is_cloud, _ = is_cloud_synced_path(path)
        if is_cloud and self._watch_enabled:
            check_and_warn_cloud_sync(path, logger)

        # Try to load from persistence first
        if (
            self._persistence is not None
            and not self.index.entries
            and self.load_from_persistence()
        ):
            # Check for staleness and do incremental update
            (
                entries,
                added,
                updated,
                unchanged,
                changed_files,
                removed_files,
            ) = self.index.index_directory_incremental(
                str(path),
                recursive=recursive,
                include_blocks=include_blocks,
                ignore_spec=self._ignore_spec,
            )

            # Persist only changed/removed files (delta), not the full index
            if changed_files or removed_files:
                for fp in changed_files:
                    if fp in self.index.file_metadata:
                        file_entries = [
                            self.index.entries[eid]
                            for eid in self.index.file_entries.get(fp, [])
                            if eid in self.index.entries
                        ]
                        self._persistence.save_file_entries(
                            fp, file_entries, self.index.file_metadata[fp]
                        )
                for fp in removed_files:
                    self._persistence.delete_file_entries(fp)
                self._persistence.save_index_metadata(self.index)

            logger.info(
                f"Incremental update: {added} added, {updated} updated, {unchanged} unchanged"
            )

            self._maybe_start_watching(path)
            self._invalidate_cache_and_recompute()

            return len(self.index.entries)

        # Full index
        self.index.clear()
        self.index.index_directory(
            str(path),
            recursive=recursive,
            include_blocks=include_blocks,
            ignore_spec=self._ignore_spec,
        )

        # Persist
        if self._persistence is not None:
            self._persistence.save_full_index(self.index)

        self._maybe_start_watching(path)
        self._invalidate_cache_and_recompute()

        logger.info(f"Indexed {len(self.index.entries)} entries from {path}")
        return len(self.index.entries)

    def suppress(self, wl_hash: str, reason: str | None = None) -> bool:
        """Suppress a hash and persist."""
        return self._toggle_suppression(wl_hash, suppress=True, reason=reason)

    def unsuppress(self, wl_hash: str) -> bool:
        """Unsuppress a hash and persist."""
        return self._toggle_suppression(wl_hash, suppress=False)

    def _persist_suppression(self, wl_hash: str) -> None:
        """Persist a single suppression if persistence is enabled."""
        if self._persistence:
            info = self.index.get_suppression_info(wl_hash)
            if info:
                self._persistence.save_suppression(info)

    def _persist_unsuppression(self, wl_hash: str) -> None:
        """Persist a single unsuppression if persistence is enabled."""
        if self._persistence:
            self._persistence.delete_suppression(wl_hash)

    def _toggle_suppression(self, wl_hash: str, suppress: bool, reason: str | None = None) -> bool:
        """Toggle suppression for one hash and persist/cache-update on change."""
        success = (
            self.index.suppress(wl_hash, reason) if suppress else self.index.unsuppress(wl_hash)
        )
        if success:
            if suppress:
                self._persist_suppression(wl_hash)
            else:
                self._persist_unsuppression(wl_hash)

            self._invalidate_cache_and_recompute()
            return True
        return False

    def _run_batch_hash_operation(
        self,
        wl_hashes: list[str],
        operation: Callable[[str], bool],
        persist_change: Callable[[str], None],
    ) -> tuple[list[str], list[str]]:
        """Run a batch hash operation, persist changed hashes, and refresh cache once."""
        changed, not_found = batch_hash_operation(wl_hashes, operation)
        for wl_hash in changed:
            persist_change(wl_hash)
        if changed:
            self._invalidate_cache_and_recompute()
        return changed, not_found

    def unsuppress_batch(self, wl_hashes: list[str]) -> tuple[list[str], list[str]]:
        """Unsuppress multiple hashes and persist. Returns (unsuppressed, not_found).

        Optimized: does all in-memory work first, then batch-persists, then
        invalidates cache and recomputes once.
        """
        return self._run_batch_hash_operation(
            wl_hashes=wl_hashes,
            operation=self.index.unsuppress,
            persist_change=self._persist_unsuppression,
        )

    def suppress_batch(
        self, wl_hashes: list[str], reason: str | None = None
    ) -> tuple[list[str], list[str]]:
        """Suppress multiple hashes and persist. Returns (suppressed, not_found).

        Optimized: does all in-memory work first, then batch-persists, then
        invalidates cache and recomputes once.
        """
        return self._run_batch_hash_operation(
            wl_hashes=wl_hashes,
            operation=lambda h: self.index.suppress(h, reason),
            persist_change=self._persist_suppression,
        )

    # =========================================================================
    # Stats
    # =========================================================================

    @property
    def is_watching(self) -> bool:
        """Whether file watching is actively keeping the index current."""
        return self._watcher is not None and self._watcher.is_watching

    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        index_stats = self.index.get_stats()

        persistence_stats = {}
        if self._persistence is not None:
            persistence_stats = self._persistence.get_stats()

        return {
            **index_stats,
            "persistence": persistence_stats,
            "watching": self.is_watching,
            "watched_path": str(self._watched_path) if self._watched_path else None,
            "cache_valid": self._cache.is_valid(),
            "cache_computed_at": self._cache.computed_at,
            "file_events_processed": self._file_events_processed,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }
