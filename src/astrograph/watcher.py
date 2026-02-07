"""
File system watcher for event-driven indexing.

Uses watchdog library to monitor filesystem changes and trigger
index updates automatically.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from watchdog.events import (
    DirCreatedEvent,
    DirDeletedEvent,
    DirModifiedEvent,
    FileCreatedEvent,
    FileDeletedEvent,
    FileModifiedEvent,
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .context import CloseOnExitMixin, StartCloseOnExitMixin
from .index import _is_skip_dir
from .languages.registry import LanguageRegistry

HAS_WATCHDOG = True

logger = logging.getLogger(__name__)


def _should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped based on directory names."""
    return any(_is_skip_dir(part) for part in path.parts)


def _apply_and_clear(items: dict[str, Any], action: Callable[[Any], None]) -> None:
    """Apply an action to all values and clear the mapping."""
    for item in items.values():
        action(item)
    items.clear()


def _apply_and_clear_locked(
    lock: Any, items: dict[str, Any], action: Callable[[Any], None]
) -> None:
    """Apply action to all values and clear mapping while holding lock."""
    with lock:
        _apply_and_clear(items, action)


class DebouncedCallback:
    """
    Debounces rapid file system events.

    Many editors save files multiple times in rapid succession.
    This class ensures we only process the final state.
    """

    def __init__(self, callback: Callable[[str], None], delay: float = 0.1) -> None:
        self.callback = callback
        self.delay = delay
        self._pending: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def __call__(self, path: str) -> None:
        """Schedule a debounced callback for the given path."""
        with self._lock:
            # Cancel any pending callback for this path
            if path in self._pending:
                self._pending[path].cancel()

            # Schedule new callback
            timer = threading.Timer(self.delay, self._execute, args=[path])
            timer.daemon = True
            self._pending[path] = timer
            timer.start()

    def _execute(self, path: str) -> None:
        """Execute the callback after debounce delay."""
        with self._lock:
            self._pending.pop(path, None)

        try:
            self.callback(path)
        except Exception:
            logger.exception(f"Error in debounced callback for {path}")

    def cancel_all(self) -> None:
        """Cancel all pending callbacks."""
        _apply_and_clear_locked(self._lock, self._pending, lambda timer: timer.cancel())


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
    ) -> None:
        super().__init__()
        self._on_modified = DebouncedCallback(on_modified, debounce_delay)
        self._on_created = DebouncedCallback(on_created, debounce_delay)
        self._on_deleted = on_deleted  # No debounce for deletes

    def _is_supported_source_file(self, path: str) -> bool:
        """Check if path is a supported source file we should track."""
        p = Path(path)
        return p.suffix in LanguageRegistry.get().supported_extensions and not _should_skip_path(p)

    def _handle_event(self, event: FileSystemEvent, event_type: str, handler: Callable) -> None:
        """Generic event handler to reduce duplication."""
        if not event.is_directory:
            src = str(event.src_path)
            if self._is_supported_source_file(src):
                logger.debug(f"File {event_type}: {src}")
                handler(src)

    def on_modified(self, event: DirModifiedEvent | FileModifiedEvent) -> None:
        self._handle_event(event, "modified", self._on_modified)

    def on_created(self, event: DirCreatedEvent | FileCreatedEvent) -> None:
        self._handle_event(event, "created", self._on_created)

    def on_deleted(self, event: DirDeletedEvent | FileDeletedEvent) -> None:
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
        self._on_modified.cancel_all()
        self._on_created.cancel_all()


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
    ) -> None:
        """
        Initialize the file watcher.

        Args:
            root_path: Directory to watch
            on_file_changed: Callback when a source file is modified
            on_file_created: Callback when a source file is created
            on_file_deleted: Callback when a source file is deleted
            debounce_delay: Seconds to wait before processing rapid events
        """
        self.root_path = Path(root_path).resolve()
        self._handler = SourceFileHandler(
            on_modified=on_file_changed,
            on_created=on_file_created,
            on_deleted=on_file_deleted,
            debounce_delay=debounce_delay,
        )
        self._observer: BaseObserver | None = None
        self._started = False

    def start(self) -> None:
        """Start watching for file changes."""
        if self._started:
            logger.debug(f"Already watching: {self.root_path}")
            return

        if not self.root_path.is_dir():
            raise ValueError(f"Watch path is not a directory: {self.root_path}")

        self._observer = Observer()
        self._observer.schedule(self._handler, str(self.root_path), recursive=True)
        self._observer.start()
        self._started = True
        logger.info(f"Started watching: {self.root_path}")

    def stop(self) -> None:
        """Stop watching for file changes."""
        if not self._started:
            return

        self._handler.cancel_pending()

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None

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
