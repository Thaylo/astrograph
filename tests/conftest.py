"""Global pytest fixtures for deterministic test behavior."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _safe_test_runtime(monkeypatch):
    """Force the suite into a non-destructive, non-watching runtime posture."""
    monkeypatch.setenv("ASTROGRAPH_WORKSPACE", "")
    monkeypatch.setenv("ASTROGRAPH_DISABLE_WATCH", "1")
    monkeypatch.delenv("ASTROGRAPH_ENABLE_WATCH", raising=False)
    monkeypatch.setenv("ASTROGRAPH_INDEX_TIMEOUT", "15")
    monkeypatch.setenv("ASTROGRAPH_WATCH_MAX_PENDING", "64")
    monkeypatch.setenv("ASTROGRAPH_WATCH_MAX_EVENTS_PER_WINDOW", "64")
    monkeypatch.setenv("ASTROGRAPH_WATCH_STORM_WINDOW_SECONDS", "0.5")
    monkeypatch.setenv("ASTROGRAPH_WATCH_COOLDOWN_SECONDS", "10")

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        home = Path(tmpdir)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(home / ".cache"))
        monkeypatch.setenv("XDG_DATA_HOME", str(home / ".local" / "share"))
        yield
        # Reset singletons to prevent cross-test resource leaks
        import contextlib

        from astrograph.tools import CodeStructureTools
        from astrograph.watcher import WatcherGovernor

        with CodeStructureTools._live_lock:
            live = CodeStructureTools._live_instance
            CodeStructureTools._live_instance = None
        if live is not None:
            with contextlib.suppress(Exception):
                live.close()
        WatcherGovernor.reset()
        shutil.rmtree(home, ignore_errors=True)


def pytest_sessionstart(session):
    """Apply an emergency timeout guard for the entire pytest process."""
    del session
    if hasattr(os, "alarm"):
        os.alarm(600)


def pytest_sessionfinish(session, exitstatus):
    """Clear the emergency timeout guard when the session ends."""
    del session, exitstatus
    if hasattr(os, "alarm"):
        os.alarm(0)
