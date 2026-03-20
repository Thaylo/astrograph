"""
File system watcher for event-driven indexing.

Uses watchdog library to monitor filesystem changes and trigger
index updates automatically.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .context import CloseOnExitMixin, StartCloseOnExitMixin
from .ignorefile import IgnoreSpec
from .index import _is_skip_dir
from .languages.registry import LanguageRegistry

HAS_WATCHDOG = True

logger = logging.getLogger(__name__)

_WATCHER_MAX_PENDING = max(1, int(os.environ.get("ASTROGRAPH_WATCH_MAX_PENDING", "256")))
_WATCHER_MAX_EVENTS_PER_WINDOW = max(
    1, int(os.environ.get("ASTROGRAPH_WATCH_MAX_EVENTS_PER_WINDOW", "512"))
)
_WATCHER_STORM_WINDOW_SECONDS = max(
    0.1, float(os.environ.get("ASTROGRAPH_WATCH_STORM_WINDOW_SECONDS", "1.0"))
)
_WATCHER_COOLDOWN_SECONDS = max(
    0.1, float(os.environ.get("ASTROGRAPH_WATCH_COOLDOWN_SECONDS", "5.0"))
)
_WATCHER_SHUTDOWN_JOIN_TIMEOUT = max(
    0.1, float(os.environ.get("ASTROGRAPH_WATCH_SHUTDOWN_JOIN_TIMEOUT", "1.0"))
)
_WATCHER_OBSERVER_JOIN_TIMEOUT = max(
    0.1, float(os.environ.get("ASTROGRAPH_WATCH_OBSERVER_JOIN_TIMEOUT", "2.0"))
)
_WATCHER_DISABLED = os.environ.get("ASTROGRAPH_DISABLE_WATCH", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_WATCHER_MAX_ACTIVE = max(1, int(os.environ.get("ASTROGRAPH_WATCH_MAX_ACTIVE", "4")))


class WatcherGovernor:
    """Process-wide safety governor for live file watching."""

    _lock = threading.Lock()
    _active_roots: set[str] = set()

    @classmethod
    def try_acquire(cls, root: Path) -> bool:
        """Reserve one watcher slot for *root* if capacity allows."""
        key = str(root)
        with cls._lock:
            if key in cls._active_roots:
                return True
            if len(cls._active_roots) >= _WATCHER_MAX_ACTIVE:
                return False
            cls._active_roots.add(key)
            return True

    @classmethod
    def release(cls, root: Path) -> None:
        """Release the watcher slot for *root*."""
        with cls._lock:
            cls._active_roots.discard(str(root))

    @classmethod
    def active_count(cls) -> int:
        """Expose the active watcher count for tests and diagnostics."""
        with cls._lock:
            return len(cls._active_roots)

    @classmethod
    def reset(cls) -> None:
        """Clear all watcher slots (for test isolation)."""
        with cls._lock:
            cls._active_roots.clear()


def _should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped based on directory names."""
    return any(_is_skip_dir(part) for part in path.parts)


def _is_tooling_artifact(path: Path) -> bool:
    """Ignore tool-owned metadata and SQLite churn sources."""
    name = path.name
    if name in {"index.db", "index.db-wal", "index.db-shm"}:
        return True
    if ".metadata_astrograph" in path.parts:
        return True
    return False


class DebouncedCallback:
    """
    Debounces rapid file system events.

    Many editors save files multiple times in rapid succession.
    This class ensures we only process the final state.
    """

    def __init__(self, callback: Callable[[str], None], delay: float = 0.1) -> None:
        self.callback = callback
        self.delay = delay
        self._pending: dict[str, float] = {}
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._running = True
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def __call__(self, path: str) -> None:
        """Schedule a debounced callback for the given path."""
        with self._cv:
            if not self._running:
                return
            if len(self._pending) >= _WATCHER_MAX_PENDING and path not in self._pending:
                logger.warning(
                    "Dropping debounced path %s because pending queue reached %d items",
                    path,
                    _WATCHER_MAX_PENDING,
                )
                return
            self._pending[path] = time.monotonic() + self.delay
            self._cv.notify()

    def _run(self) -> None:
        """Worker loop to execute debounced callbacks without spawning threads."""
        while True:
            with self._cv:
                if not self._running and not self._pending:
                    return
                if not self._pending:
                    self._cv.wait()
                    continue
                next_due = min(self._pending.values())
                now = time.monotonic()
                if next_due > now:
                    self._cv.wait(timeout=next_due - now)
                    continue
                due_paths = [path for path, due in self._pending.items() if due <= now]
                for path in due_paths:
                    self._pending.pop(path, None)

            for path in due_paths:
                try:
                    self.callback(path)
                except Exception:
                    logger.exception(f"Error in debounced callback for {path}")

    def cancel_all(self) -> None:
        """Cancel all pending callbacks."""
        with self._cv:
            self._pending.clear()
            self._cv.notify()

    def shutdown(self) -> None:
        """Stop the worker thread after clearing pending callbacks."""
        with self._cv:
            if not self._running:
                return
            self._running = False
            self._pending.clear()
            self._cv.notify()
        self._worker.join(timeout=_WATCHER_SHUTDOWN_JOIN_TIMEOUT)


class SourceFileHandler(FileSystemEventHandler):
    """
    Handles file system events for supported source files.

    Filters by registered plugin extensions and dispatches to callbacks.
    """

    def __init__(
        self,
        on_modified: Callable[[str], None],
        on_created: Callable[[str], None],
        on_deleted: Callable[[str], None],
        debounce_delay: float = 0.1,
        ignore_spec: IgnoreSpec | None = None,
        root_path: Path | None = None,
        on_storm_quarantine: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_modified = DebouncedCallback(on_modified, debounce_delay)
        self._on_created = DebouncedCallback(on_created, debounce_delay)
        self._on_deleted = on_deleted  # No debounce for deletes
        self._ignore_spec = ignore_spec
        self._root_path = root_path
        self._event_times: list[float] = []
        self._storm_suppressed_until = 0.0
        self._quarantined = False
        self._on_storm_quarantine = on_storm_quarantine

    def _record_event_rate(self) -> bool:
        """Return True when the watcher should suppress events due to a storm."""
        now = time.monotonic()
        if self._quarantined or now < self._storm_suppressed_until:
            return True

        window_start = now - _WATCHER_STORM_WINDOW_SECONDS
        self._event_times = [t for t in self._event_times if t >= window_start]
        self._event_times.append(now)

        if len(self._event_times) > _WATCHER_MAX_EVENTS_PER_WINDOW:
            self._storm_suppressed_until = now + _WATCHER_COOLDOWN_SECONDS
            self._event_times.clear()
            self._quarantined = True
            logger.warning(
                "Suppressing file-watch events for %.2fs after storm detection on %s",
                _WATCHER_COOLDOWN_SECONDS,
                self._root_path,
            )
            if self._on_storm_quarantine is not None:
                try:
                    self._on_storm_quarantine()
                except Exception:  # pragma: no cover
                    logger.exception("Error in storm quarantine callback")
            return True
        return False

    @property
    def quarantined(self) -> bool:
        """Whether the handler is currently in storm quarantine."""
        if self._quarantined and time.monotonic() >= self._storm_suppressed_until:
            self._quarantined = False
        return self._quarantined

    def _is_supported_source_file(self, path: str) -> bool:
        """Check if path is a supported source file we should track."""
        p = Path(path)
        if _is_tooling_artifact(p):
            return False
        registry = LanguageRegistry.get()
        if registry is None:
            return False
        if p.suffix not in registry.supported_extensions:
            return False
        if _should_skip_path(p):
            return False
        if self._ignore_spec is not None and self._root_path is not None:
            try:
                rel = str(p.relative_to(self._root_path))
                if self._ignore_spec.is_file_ignored(rel):
                    return False
            except ValueError:
                pass  # path not under root_path
        return True

    def _handle_event(self, event: FileSystemEvent, event_type: str, handler: Callable) -> None:
        """Generic event handler to reduce duplication."""
        if not event.is_directory:
            src = str(event.src_path)
            if self._is_supported_source_file(src):
                if self._record_event_rate():
                    logger.debug("Suppressed %s event during watcher cooldown: %s", event_type, src)
                    return
                logger.debug(f"File {event_type}: {src}")
                handler(src)

    def on_any_event(self, event: FileSystemEvent) -> None:
        """Route core watchdog events to the right callback."""
        event_type = getattr(event, "event_type", "")
        if event_type == "modified":
            self._handle_event(event, "modified", self._on_modified)
        elif event_type == "created":
            self._handle_event(event, "created", self._on_created)
        elif event_type == "deleted":
            self._handle_event(event, "deleted", self._on_deleted)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            # Handle as delete + create
            src = str(event.src_path)
            if self._is_supported_source_file(src):
                logger.debug(f"File moved from: {src}")
                self._on_deleted(src)

            dest = str(getattr(event, "dest_path", ""))
            if dest and self._is_supported_source_file(dest):
                logger.debug(f"File moved to: {dest}")
                self._on_created(dest)

    def cancel_pending(self) -> None:
        """Cancel pending debounced callbacks."""
        self._on_modified.shutdown()
        self._on_created.shutdown()


class FileWatcher(StartCloseOnExitMixin):
    """
    Watches a directory for source file changes.

    Provides a simple interface to start/stop watching and
    register callbacks for file events.
    """

    def __init__(
        self,
        root_path: str | Path,
        on_file_changed: Callable[[str], None],
        on_file_created: Callable[[str], None],
        on_file_deleted: Callable[[str], None],
        debounce_delay: float = 0.1,
        ignore_spec: IgnoreSpec | None = None,
        on_storm_quarantine: Callable[[], None] | None = None,
    ) -> None:
        """
        Initialize the file watcher.

        Args:
            root_path: Directory to watch
            on_file_changed: Callback when a source file is modified
            on_file_created: Callback when a source file is created
            on_file_deleted: Callback when a source file is deleted
            debounce_delay: Seconds to wait before processing rapid events
            ignore_spec: Optional ignore patterns for file exclusion
            on_storm_quarantine: Callback when storm quarantine activates
        """
        self.root_path = Path(root_path).resolve()
        self._handler = SourceFileHandler(
            on_modified=on_file_changed,
            on_created=on_file_created,
            on_deleted=on_file_deleted,
            debounce_delay=debounce_delay,
            ignore_spec=ignore_spec,
            root_path=self.root_path,
            on_storm_quarantine=on_storm_quarantine,
        )
        self._observer: BaseObserver | None = None
        self._started = False
        self._governor_acquired = False

    def start(self) -> None:
        """Start watching for file changes."""
        if self._started:
            logger.debug(f"Already watching: {self.root_path}")
            return

        if _WATCHER_DISABLED:
            logger.info("File watching disabled by ASTROGRAPH_DISABLE_WATCH for %s", self.root_path)
            return

        if not self.root_path.is_dir():
            raise ValueError(f"Watch path is not a directory: {self.root_path}")

        if not WatcherGovernor.try_acquire(self.root_path):
            logger.warning(
                "Refusing to watch %s because the global watcher budget (%d) is exhausted",
                self.root_path,
                _WATCHER_MAX_ACTIVE,
            )
            return

        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.root_path), recursive=True)
        self._observer.start()
        self._governor_acquired = True
        self._started = True
        logger.info(f"Started watching: {self.root_path}")

    def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._started:
            return

        self._handler.cancel_pending()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=_WATCHER_OBSERVER_JOIN_TIMEOUT)
            self._observer = None

        if self._governor_acquired:
            WatcherGovernor.release(self.root_path)
            self._governor_acquired = False

        self._started = False
        logger.info(f"Stopped watching: {self.root_path}")

    close = stop

    @property
    def is_watching(self) -> bool:
        """Check if the watcher is active."""
        return self._started


class FileWatcherPool(CloseOnExitMixin):
    """
    Manages multiple file watchers for different directories.

    Useful when indexing multiple separate codebases.
    """

    def __init__(self) -> None:
        self._watchers: dict[str, FileWatcher] = {}
        self._lock = threading.Lock()

    def watch(
        self,
        root_path: str | Path,
        on_file_changed: Callable[[str], None],
        on_file_created: Callable[[str], None],
        on_file_deleted: Callable[[str], None],
        debounce_delay: float = 0.1,
    ) -> FileWatcher:
        """Add a directory to watch."""
        root = str(Path(root_path).resolve())

        with self._lock:
            if root in self._watchers:
                return self._watchers[root]

            watcher = FileWatcher(
                root_path=root,
                on_file_changed=on_file_changed,
                on_file_created=on_file_created,
                on_file_deleted=on_file_deleted,
                debounce_delay=debounce_delay,
            )
            watcher.start()
            self._watchers[root] = watcher
            return watcher

    def unwatch(self, root_path: str | Path) -> None:
        """Stop watching a directory."""
        root = str(Path(root_path).resolve())

        with self._lock:
            if root in self._watchers:
                self._watchers[root].stop()
                del self._watchers[root]

    def stop_all(self) -> None:
        """Stop all watchers."""
        with self._lock:
            for watcher in self._watchers.values():
                watcher.stop()
            self._watchers.clear()

    close = stop_all


# Backward compatibility alias (historically watcher was Python-only).
PythonFileHandler = SourceFileHandler
