"""Shared test utilities."""

import pytest


def skip_if_watchdog_missing() -> None:
    """Skip the calling test if watchdog is not installed."""
    from astrograph.watcher import HAS_WATCHDOG

    if HAS_WATCHDOG:
        return
    pytest.skip("watchdog not installed")
