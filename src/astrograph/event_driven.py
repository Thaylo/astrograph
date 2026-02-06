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
from typing import Any

from .cloud_detect import check_and_warn_cloud_sync, is_cloud_synced_path
from .index import CodeStructureIndex, DuplicateGroup, batch_hash_operation
from .persistence import SQLitePersistence

logger = logging.getLogger(__name__)

# Try to import watchdog components
try:
    from .watcher import HAS_WATCHDOG, FileWatcher
except ImportError:
    HAS_WATCHDOG = False
    FileWatcher = None  # type: ignore


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
        with self._lock:
            return self._computed_at


class EventDrivenIndex:
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
    ) -> None:
        """
        Initialize the event-driven index.

        Args:
            persistence_path: Path to SQLite database (None = no persistence)
            watch_enabled: Enable file system watching (requires watchdog)
            debounce_delay: Seconds to wait before processing rapid file changes
        """
        # In-memory index (always hot)
        self.index = CodeStructureIndex()

        # SQLite persistence
        self._persistence: SQLitePersistence | None = None
        if persistence_path:
            self._persistence = SQLitePersistence(persistence_path)

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

    def __enter__(self) -> "EventDrivenIndex":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # =========================================================================
    # File Event Handlers
    # =========================================================================

    def _handle_file_event(self, path: str, event_type: str, handler: Callable) -> None:
        """Generic file event handler to reduce duplication."""
        logger.debug(f"Processing file {event_type}: {path}")
        handler(path)
        self._file_events_processed += 1

    def _on_file_changed(self, path: str) -> None:
        """Handle file modification event."""
        self._handle_file_event(path, "change", self._reindex_file)

    def _on_file_created(self, path: str) -> None:
        """Handle file creation event."""
        self._handle_file_event(path, "creation", self._reindex_file)

    def _on_file_deleted(self, path: str) -> None:
        """Handle file deletion event."""
        self._handle_file_event(path, "deletion", self._remove_file)

    def _reindex_file(self, path: str) -> None:
        """Re-index a single file and persist incrementally."""
        if not os.path.exists(path):
            return

        # Update in-memory index
        entries = self.index.index_file(path, include_blocks=True)

        # Persist incrementally
        if self._persistence is not None and path in self.index.file_metadata:
            metadata = self.index.file_metadata[path]
            self._persistence.save_file_entries(path, entries, metadata)

        # Invalidate cache
        self._cache.invalidate()

        # Trigger background analysis recompute
        self._schedule_analysis_recompute()

    def _remove_file(self, path: str) -> None:
        """Remove a file from the index and persistence."""
        # Update in-memory index
        self.index.remove_file(path)

        # Remove from persistence
        if self._persistence is not None:
            self._persistence.delete_file_entries(path)

        # Invalidate cache
        self._cache.invalidate()

        # Trigger background analysis recompute
        self._schedule_analysis_recompute()

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

    def _recompute_analysis(self) -> None:
        """Recompute analysis in background thread."""
        if self._shutdown.is_set():
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
        is_cloud, service = is_cloud_synced_path(path)
        if is_cloud and self._watch_enabled:
            check_and_warn_cloud_sync(path, logger)

        # Try to load from persistence first
        if (
            self._persistence is not None
            and not self.index.entries
            and self.load_from_persistence()
        ):
            # Check for staleness and do incremental update
            entries, added, updated, unchanged = self.index.index_directory_incremental(
                str(path), recursive=recursive, include_blocks=include_blocks
            )

            # Persist any changes
            if added > 0 or updated > 0:
                self._persistence.save_full_index(self.index)

            logger.info(
                f"Incremental update: {added} added, {updated} updated, {unchanged} unchanged"
            )

            # Start watching if enabled
            if self._watch_enabled:
                self.start_watching(path)

            # Trigger cache recompute
            self._cache.invalidate()
            self._schedule_analysis_recompute()

            return len(self.index.entries)

        # Full index
        self.index.clear()
        self.index.index_directory(str(path), recursive=recursive, include_blocks=include_blocks)

        # Persist
        if self._persistence is not None:
            self._persistence.save_full_index(self.index)

        # Start watching if enabled
        if self._watch_enabled:
            self.start_watching(path)

        # Trigger cache recompute
        self._cache.invalidate()
        self._schedule_analysis_recompute()

        logger.info(f"Indexed {len(self.index.entries)} entries from {path}")
        return len(self.index.entries)

    def suppress(self, wl_hash: str, reason: str | None = None) -> bool:
        """Suppress a hash and persist."""
        success = self.index.suppress(wl_hash, reason)

        if success and self._persistence is not None:
            info = self.index.get_suppression_info(wl_hash)
            if info:
                self._persistence.save_suppression(info)

        # Invalidate cache (suppressions affect analysis)
        self._cache.invalidate()
        self._schedule_analysis_recompute()

        return success

    def unsuppress(self, wl_hash: str) -> bool:
        """Unsuppress a hash and persist."""
        success = self.index.unsuppress(wl_hash)

        if success and self._persistence is not None:
            self._persistence.delete_suppression(wl_hash)

        # Invalidate cache
        self._cache.invalidate()
        self._schedule_analysis_recompute()

        return success

    def unsuppress_batch(self, wl_hashes: list[str]) -> tuple[list[str], list[str]]:
        """Unsuppress multiple hashes and persist. Returns (unsuppressed, not_found)."""
        return batch_hash_operation(wl_hashes, self.unsuppress)

    def suppress_batch(
        self, wl_hashes: list[str], reason: str | None = None
    ) -> tuple[list[str], list[str]]:
        """Suppress multiple hashes and persist. Returns (suppressed, not_found)."""
        return batch_hash_operation(wl_hashes, lambda h: self.suppress(h, reason))

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> dict:
        """Get comprehensive statistics."""
        index_stats = self.index.get_stats()

        persistence_stats = {}
        if self._persistence is not None:
            persistence_stats = self._persistence.get_stats()

        return {
            **index_stats,
            "persistence": persistence_stats,
            "watching": self._watcher is not None and self._watcher.is_watching,
            "watched_path": str(self._watched_path) if self._watched_path else None,
            "cache_valid": self._cache.is_valid(),
            "cache_computed_at": self._cache.computed_at,
            "file_events_processed": self._file_events_processed,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
        }
