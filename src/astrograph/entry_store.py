"""
LRU-bounded entry store for the code structure index.

Replaces a plain dict[str, IndexEntry] with a container that:
- Keeps lightweight metadata in memory for every entry (hot path)
- Evicts full IndexEntry objects from memory when over the LRU limit
- Reloads evicted entries from SQLite on demand (event-driven mode only)
- In standard mode (no SQLite), all entries stay in memory
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .index import IndexEntry
    from .persistence import SQLitePersistence

logger = logging.getLogger(__name__)

# Default max resident entries (0 = unlimited)
_DEFAULT_MAX_RESIDENT = 50_000


def _get_max_resident() -> int:
    """Read max_resident from environment variable."""
    raw = os.environ.get("ASTROGRAPH_MAX_ENTRIES", "")
    if raw.strip():
        try:
            return int(raw)
        except ValueError:
            pass
    return _DEFAULT_MAX_RESIDENT


@dataclass(slots=True)
class EntryMeta:
    """Lightweight metadata kept in memory for every entry (~400 bytes)."""

    node_count: int
    wl_hash: str
    pattern_hash: str
    unit_type: str
    block_type: str | None
    fingerprint_key: tuple[int, int] | None
    hierarchy_hashes: list[str]


class EntryStore:
    """
    Dict-compatible LRU-bounded container for IndexEntry objects.

    Keeps hot metadata in memory for every entry while evicting full
    IndexEntry objects when the cache exceeds ``max_resident``.
    Evicted entries are reloaded from SQLite on demand.

    When no persistence is configured (standard mode), eviction is
    disabled regardless of ``max_resident``.
    """

    def __init__(self, max_resident: int | None = None) -> None:
        if max_resident is None:
            max_resident = _get_max_resident()
        self._max_resident: int = max_resident
        self._cache: OrderedDict[str, IndexEntry] = OrderedDict()
        self._persistence: SQLitePersistence | None = None
        self._all_ids: set[str] = set()
        self._meta: dict[str, EntryMeta] = {}
        self._bulk_loading: bool = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_persistence(self, persistence: SQLitePersistence | None) -> None:
        """Wire the SQLite backing store for eviction support."""
        self._persistence = persistence

    # ------------------------------------------------------------------
    # Dict-compatible interface
    # ------------------------------------------------------------------

    def __getitem__(self, eid: str) -> IndexEntry:
        if eid in self._cache:
            self._cache.move_to_end(eid)
            return self._cache[eid]

        # Not in cache – try SQLite
        if self._persistence is not None and eid in self._all_ids:
            entry = self._persistence.get_entry(eid)
            if entry is not None:
                # Re-admit to cache
                self._cache[eid] = entry
                self._cache.move_to_end(eid)
                self._maybe_evict()
                return entry

        raise KeyError(eid)

    def __setitem__(self, eid: str, entry: IndexEntry) -> None:
        self._cache[eid] = entry
        self._cache.move_to_end(eid)
        self._all_ids.add(eid)
        self._meta[eid] = self._build_meta(entry)
        self._maybe_evict()

    def __delitem__(self, eid: str) -> None:
        self._cache.pop(eid, None)
        self._all_ids.discard(eid)
        self._meta.pop(eid, None)

    def __contains__(self, eid: object) -> bool:
        return eid in self._all_ids

    def __len__(self) -> int:
        return len(self._all_ids)

    def __bool__(self) -> bool:
        return bool(self._all_ids)

    def __iter__(self) -> Iterator[str]:
        return iter(self._all_ids)

    def get(self, eid: str, default: IndexEntry | None = None) -> IndexEntry | None:
        try:
            return self[eid]
        except KeyError:
            return default

    def items(self) -> Iterator[tuple[str, IndexEntry]]:
        """Yield all (id, entry) pairs: cached first, then evicted from SQLite."""
        yielded: set[str] = set()

        for eid, entry in self._cache.items():
            yielded.add(eid)
            yield eid, entry

        evicted_ids = self._all_ids - yielded
        if evicted_ids and self._persistence is not None:
            yield from self._persistence.get_entries_batch(evicted_ids)

    def values(self) -> Iterator[IndexEntry]:
        """Yield all entries: cached first, then evicted from SQLite."""
        for _, entry in self.items():
            yield entry

    def keys(self) -> set[str]:
        """Return all entry IDs."""
        return set(self._all_ids)

    def pop(self, eid: str, *args: IndexEntry | None) -> IndexEntry | None:
        """Remove and return entry, or default if not found."""
        try:
            entry = self[eid]
            del self[eid]
            return entry
        except KeyError:
            if args:
                return args[0]
            raise

    # ------------------------------------------------------------------
    # Hot metadata access (no full entry load)
    # ------------------------------------------------------------------

    def get_node_count(self, eid: str) -> int | None:
        """Get node_count from hot metadata without loading the full entry."""
        meta = self._meta.get(eid)
        return meta.node_count if meta else None

    def get_hierarchy_hashes(self, eid: str) -> list[str] | None:
        """Get hierarchy_hashes from hot metadata without loading the full entry."""
        meta = self._meta.get(eid)
        return meta.hierarchy_hashes if meta else None

    def get_meta(self, eid: str) -> EntryMeta | None:
        """Get full hot metadata for bucket cleanup in remove_file()."""
        return self._meta.get(eid)

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------

    @contextmanager
    def bulk_load(self) -> Iterator[None]:
        """Context manager: suppress eviction during bulk load, trim after."""
        self._bulk_loading = True
        try:
            yield
        finally:
            self._bulk_loading = False
            self._trim_cache()

    # ------------------------------------------------------------------
    # Cache stats
    # ------------------------------------------------------------------

    @property
    def resident_count(self) -> int:
        """Number of entries currently in the in-memory cache."""
        return len(self._cache)

    @property
    def total_count(self) -> int:
        """Total number of entries (cached + evicted)."""
        return len(self._all_ids)

    # ------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Remove all entries from the store."""
        self._cache.clear()
        self._all_ids.clear()
        self._meta.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_meta(entry: IndexEntry) -> EntryMeta:
        fp = entry.fingerprint
        fp_key = (fp["n_nodes"], fp["n_edges"]) if "n_nodes" in fp else None
        return EntryMeta(
            node_count=entry.node_count,
            wl_hash=entry.wl_hash,
            pattern_hash=entry.pattern_hash,
            unit_type=entry.code_unit.unit_type,
            block_type=entry.code_unit.block_type,
            fingerprint_key=fp_key,
            hierarchy_hashes=entry.hierarchy_hashes,
        )

    def _maybe_evict(self) -> None:
        """Evict oldest entries if over the limit."""
        if self._bulk_loading:
            return
        self._trim_cache()

    def _trim_cache(self) -> None:
        """Trim the cache to max_resident size."""
        if self._max_resident <= 0:
            return  # Unlimited
        if self._persistence is None:
            return  # No eviction without persistence

        while len(self._cache) > self._max_resident:
            # Pop oldest (FIFO end of OrderedDict)
            self._cache.popitem(last=False)
