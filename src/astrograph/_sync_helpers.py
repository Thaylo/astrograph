"""Small synchronization helpers shared across modules."""

from __future__ import annotations

from typing import Any


def pop_attr_with_lock(lock: Any, owner: Any, attr: str) -> Any:
    """Return owner.attr while holding lock, then clear owner.attr."""
    with lock:
        value = getattr(owner, attr)
        setattr(owner, attr, None)
    return value
