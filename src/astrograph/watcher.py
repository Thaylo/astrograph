"""
File system watcher for event-driven indexing.

Uses watchdog library to monitor filesystem changes and trigger
index updates automatically.
"""

from __future__ import annotations

import logging
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


def _should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped based on directory names."""
    return any(_is_skip_dir(part) for part in path.parts)


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
        self._worker.join(timeout=1.0)


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
    ) -> None:
        super().__init__()
        self._on_modified = DebouncedCallback(on_modified, debounce_delay)
        self._on_created = DebouncedCallback(on_created, debounce_delay)
        self._on_deleted = on_deleted  # No debounce for deletes
        self._ignore_spec = ignore_spec
        self._root_path = root_path

    def _is_supported_source_file(self, path: str) -> bool:
        """Check if path is a supported source file we should track."""
        p = Path(path)
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
        """
        self.root_path = Path(root_path).resolve()
        self._handler = SourceFileHandler(
            on_modified=on_file_changed,
            on_created=on_file_created,
            on_deleted=on_file_deleted,
            debounce_delay=debounce_delay,
            ignore_spec=ignore_spec,
            root_path=self.root_path,
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
