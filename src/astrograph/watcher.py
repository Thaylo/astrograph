"""
File system watcher for event-driven indexing.

Uses watchdog library to monitor filesystem changes and trigger
index updates automatically.
"""

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .index import _is_skip_dir

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer

    HAS_WATCHDOG = True
except ImportError:
    HAS_WATCHDOG = False
    Observer = None
    FileSystemEventHandler = object
    FileSystemEvent = Any

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
        with self._lock:
            for timer in self._pending.values():
                timer.cancel()
            self._pending.clear()


class PythonFileHandler(FileSystemEventHandler):
    """
    Handles file system events for Python files.

    Filters for .py files and dispatches to appropriate callbacks.
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

    def _is_python_file(self, path: str) -> bool:
        """Check if path is a Python file we should track."""
        p = Path(path)
        return p.suffix == ".py" and not _should_skip_path(p)

    def _handle_event(self, event: "FileSystemEvent", event_type: str, handler: Callable) -> None:
        """Generic event handler to reduce duplication."""
        if event.is_directory:
            return
        if self._is_python_file(event.src_path):
            logger.debug(f"File {event_type}: {event.src_path}")
            handler(event.src_path)

    def on_modified(self, event: "FileSystemEvent") -> None:
        self._handle_event(event, "modified", self._on_modified)

    def on_created(self, event: "FileSystemEvent") -> None:
        self._handle_event(event, "created", self._on_created)

    def on_deleted(self, event: "FileSystemEvent") -> None:
        self._handle_event(event, "deleted", self._on_deleted)

    def on_moved(self, event: "FileSystemEvent") -> None:
        if event.is_directory:
            return

        # Handle as delete + create
        if hasattr(event, "src_path") and self._is_python_file(event.src_path):
            logger.debug(f"File moved from: {event.src_path}")
            self._on_deleted(event.src_path)

        if hasattr(event, "dest_path") and self._is_python_file(event.dest_path):
            logger.debug(f"File moved to: {event.dest_path}")
            self._on_created(event.dest_path)

    def cancel_pending(self) -> None:
        """Cancel pending debounced callbacks."""
        self._on_modified.cancel_all()
        self._on_created.cancel_all()


class FileWatcher:
    """
    Watches a directory for Python file changes.

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
            on_file_changed: Callback when a Python file is modified
            on_file_created: Callback when a Python file is created
            on_file_deleted: Callback when a Python file is deleted
            debounce_delay: Seconds to wait before processing rapid events
        """
        if not HAS_WATCHDOG:
            raise ImportError(
                "watchdog is required for file watching. Install with: pip install watchdog"
            )

        self.root_path = Path(root_path).resolve()
        self._handler = PythonFileHandler(
            on_modified=on_file_changed,
            on_created=on_file_created,
            on_deleted=on_file_deleted,
            debounce_delay=debounce_delay,
        )
        self._observer: Observer | None = None
        self._started = False

    def start(self) -> None:
        """Start watching for file changes."""
        if self._started:
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

    @property
    def is_watching(self) -> bool:
        """Check if the watcher is active."""
        return self._started

    def __enter__(self) -> "FileWatcher":
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()


class FileWatcherPool:
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

    def __enter__(self) -> "FileWatcherPool":
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop_all()
