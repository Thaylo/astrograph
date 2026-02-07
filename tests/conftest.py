"""Global pytest fixtures for deterministic test behavior."""

import pytest


@pytest.fixture(autouse=True)
def _disable_startup_autoindex(monkeypatch):
    """Disable implicit startup indexing unless a test opts in explicitly."""
    monkeypatch.setenv("ASTROGRAPH_WORKSPACE", "")
