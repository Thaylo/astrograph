"""Tests for watcher module coverage gaps."""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from astrograph.watcher import (
    DebouncedCallback,
    FileWatcher,
    FileWatcherPool,
    SourceFileHandler,
    _should_skip_path,
)


class TestShouldSkipPath:
    def test_skip_node_modules(self):
        assert _should_skip_path(Path("project/node_modules/foo.js")) is True

    def test_skip_pycache(self):
        assert _should_skip_path(Path("src/__pycache__/mod.pyc")) is True

    def test_allow_normal_path(self):
        assert _should_skip_path(Path("src/main.py")) is False


class TestDebouncedCallback:
    def test_callback_fires_after_delay(self):
        results = []
        cb = DebouncedCallback(lambda p: results.append(p), delay=0.01)
        cb("test.py")
        import time
        time.sleep(0.1)
        assert "test.py" in results

    def test_cancel_all(self):
        results = []
        cb = DebouncedCallback(lambda p: results.append(p), delay=1.0)
        cb("test.py")
        cb.cancel_all()
        import time
        time.sleep(0.05)
        assert results == []

    def test_callback_exception_logged(self):
        """Exception in callback is caught and logged (lines 84-85)."""
        def boom(path):
            raise RuntimeError("boom")

        cb = DebouncedCallback(boom, delay=0.01)
        cb("test.py")
        import time
        time.sleep(0.1)
        # Should not raise — exception is logged


class TestSourceFileHandler:
    def test_registry_none_returns_false(self):
        """When LanguageRegistry.get() is None, _is_supported_source_file → False (line 120)."""
        handler = SourceFileHandler(
            on_modified=lambda p: None,
            on_created=lambda p: None,
            on_deleted=lambda p: None,
        )
        with patch("astrograph.watcher.LanguageRegistry") as mock_reg:
            mock_reg.get.return_value = None
            assert handler._is_supported_source_file("/some/file.py") is False

    def test_skip_dir_returns_false(self):
        """Files in skip dirs (node_modules etc.) are rejected (line 124)."""
        handler = SourceFileHandler(
            on_modified=lambda p: None,
            on_created=lambda p: None,
            on_deleted=lambda p: None,
        )
        mock_registry = MagicMock()
        mock_registry.supported_extensions = frozenset({".py"})
        with patch("astrograph.watcher.LanguageRegistry") as mock_reg:
            mock_reg.get.return_value = mock_registry
            assert handler._is_supported_source_file("/project/node_modules/test.py") is False

    def test_ignore_spec_filters_file(self):
        """Files matching ignore_spec are rejected (lines 126-131)."""
        from astrograph.ignorefile import IgnoreSpec

        spec = IgnoreSpec.from_lines(["*.min.js"])
        root = Path("/project")
        handler = SourceFileHandler(
            on_modified=lambda p: None,
            on_created=lambda p: None,
            on_deleted=lambda p: None,
            ignore_spec=spec,
            root_path=root,
        )
        mock_registry = MagicMock()
        mock_registry.supported_extensions = frozenset({".js"})
        with patch("astrograph.watcher.LanguageRegistry") as mock_reg:
            mock_reg.get.return_value = mock_registry
            assert handler._is_supported_source_file("/project/dist/app.min.js") is False

    def test_ignore_spec_value_error_passes(self):
        """ValueError from relative_to is silently passed (line 130-131)."""
        from astrograph.ignorefile import IgnoreSpec

        spec = IgnoreSpec.from_lines(["*.min.js"])
        root = Path("/project")
        handler = SourceFileHandler(
            on_modified=lambda p: None,
            on_created=lambda p: None,
            on_deleted=lambda p: None,
            ignore_spec=spec,
            root_path=root,
        )
        mock_registry = MagicMock()
        mock_registry.supported_extensions = frozenset({".py"})
        with patch("astrograph.watcher.LanguageRegistry") as mock_reg:
            mock_reg.get.return_value = mock_registry
            # Path not under root_path → ValueError from relative_to → passes through
            result = handler._is_supported_source_file("/other/location/test.py")
            assert result is True


class TestFileWatcher:
    def test_start_already_watching(self, tmp_path):
        """Starting a watcher that's already started is a no-op (lines 214-215)."""
        watcher = FileWatcher(
            root_path=tmp_path,
            on_file_changed=lambda p: None,
            on_file_created=lambda p: None,
            on_file_deleted=lambda p: None,
        )
        watcher.start()
        try:
            # Second start should be a no-op
            watcher.start()
            assert watcher.is_watching is True
        finally:
            watcher.stop()

    def test_stop_not_started(self):
        """Stopping a watcher that hasn't started is a no-op (line 229)."""
        watcher = FileWatcher(
            root_path="/tmp",
            on_file_changed=lambda p: None,
            on_file_created=lambda p: None,
            on_file_deleted=lambda p: None,
        )
        watcher.stop()  # Should not raise
        assert watcher.is_watching is False


class TestFileWatcherPool:
    def test_watch_and_unwatch(self, tmp_path):
        """Pool can add and remove watchers (lines 269-284, 288-293)."""
        pool = FileWatcherPool()
        watcher = pool.watch(
            root_path=tmp_path,
            on_file_changed=lambda p: None,
            on_file_created=lambda p: None,
            on_file_deleted=lambda p: None,
        )
        assert watcher.is_watching is True

        # Watch same path returns same watcher
        watcher2 = pool.watch(
            root_path=tmp_path,
            on_file_changed=lambda p: None,
            on_file_created=lambda p: None,
            on_file_deleted=lambda p: None,
        )
        assert watcher2 is watcher

        pool.unwatch(tmp_path)
        assert watcher.is_watching is False

    def test_unwatch_nonexistent(self, tmp_path):
        """Unwatching a path not being watched is a no-op."""
        pool = FileWatcherPool()
        pool.unwatch(tmp_path)  # Should not raise

    def test_stop_all(self, tmp_path):
        """stop_all stops all watchers (lines 297-300)."""
        pool = FileWatcherPool()
        d1 = tmp_path / "a"
        d2 = tmp_path / "b"
        d1.mkdir()
        d2.mkdir()
        w1 = pool.watch(d1, lambda p: None, lambda p: None, lambda p: None)
        w2 = pool.watch(d2, lambda p: None, lambda p: None, lambda p: None)
        assert w1.is_watching and w2.is_watching
        pool.stop_all()
        assert not w1.is_watching and not w2.is_watching
