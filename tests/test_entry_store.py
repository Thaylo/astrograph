"""Tests for the EntryStore LRU-bounded container."""

import os
import tempfile

import pytest

from astrograph.entry_store import EntryMeta, EntryStore
from astrograph.index import CodeStructureIndex
from astrograph.persistence import SQLitePersistence


def _make_index_with_entries(count: int) -> tuple[CodeStructureIndex, list[str]]:
    """Create a CodeStructureIndex with `count` unique functions indexed."""
    index = CodeStructureIndex()
    funcs = []
    for i in range(count):
        # Each function is structurally unique to avoid deduplication
        func = f"def func_{i}(x):\n    return x + {i}\n"
        funcs.append(func)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("\n".join(funcs))
        f.flush()
        tmp_path = f.name

    try:
        index.index_file(tmp_path, include_blocks=False)
    finally:
        os.unlink(tmp_path)

    entry_ids = list(index.entries.keys())
    return index, entry_ids


def _populate_store(store: EntryStore, index: CodeStructureIndex, entry_ids: list[str]) -> None:
    for eid in entry_ids:
        store[eid] = index.entries[eid]


class TestEntryStoreLRU:
    """Tests for LRU eviction behavior."""

    def test_lru_eviction_with_persistence(self):
        """Entries are evicted from cache when over max_resident with persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLitePersistence(db_path)

            index, entry_ids = _make_index_with_entries(10)
            persistence.save_full_index(index)

            # Create a store with low max_resident
            store = EntryStore(max_resident=3)
            store.set_persistence(persistence)

            # Populate store
            _populate_store(store, index, entry_ids)

            # Cache should be trimmed to max_resident
            assert store.resident_count <= 3
            assert store.total_count == len(entry_ids)
            assert len(store) == len(entry_ids)

            persistence.close()

    def test_no_eviction_without_persistence(self):
        """Without persistence, all entries stay in cache regardless of max_resident."""
        store = EntryStore(max_resident=3)

        index, entry_ids = _make_index_with_entries(10)

        _populate_store(store, index, entry_ids)

        # No eviction without persistence
        assert store.resident_count == len(entry_ids)
        assert store.total_count == len(entry_ids)

    def test_sqlite_fallback_for_evicted_entries(self):
        """Evicted entries can be reloaded from SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLitePersistence(db_path)

            index, entry_ids = _make_index_with_entries(10)
            persistence.save_full_index(index)

            store = EntryStore(max_resident=3)
            store.set_persistence(persistence)

            # Populate store (will evict oldest)
            _populate_store(store, index, entry_ids)

            # Access an evicted entry - should reload from SQLite
            first_eid = entry_ids[0]
            entry = store[first_eid]
            assert entry is not None
            assert entry.id == first_eid

            persistence.close()

    def test_getitem_raises_keyerror_for_missing(self):
        """__getitem__ raises KeyError for entries that don't exist."""
        store = EntryStore(max_resident=10)
        with pytest.raises(KeyError):
            store["nonexistent_id"]


class TestEntryStoreHotMetadata:
    """Tests for hot metadata access without full entry load."""

    def test_get_node_count(self):
        """get_node_count works for all entries (cached and evicted)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLitePersistence(db_path)

            index, entry_ids = _make_index_with_entries(10)
            persistence.save_full_index(index)

            store = EntryStore(max_resident=3)
            store.set_persistence(persistence)

            _populate_store(store, index, entry_ids)

            # All entries should have node_count available via hot metadata
            for eid in entry_ids:
                nc = store.get_node_count(eid)
                assert nc is not None
                assert nc > 0

            persistence.close()

    def test_get_hierarchy_hashes(self):
        """get_hierarchy_hashes works for all entries."""
        index, entry_ids = _make_index_with_entries(5)
        store = EntryStore(max_resident=100)

        _populate_store(store, index, entry_ids)

        for eid in entry_ids:
            hh = store.get_hierarchy_hashes(eid)
            assert hh is not None
            assert isinstance(hh, list)

    def test_get_meta(self):
        """get_meta returns full EntryMeta for bucket cleanup."""
        index, entry_ids = _make_index_with_entries(3)
        store = EntryStore(max_resident=100)

        _populate_store(store, index, entry_ids)

        for eid in entry_ids:
            meta = store.get_meta(eid)
            assert meta is not None
            assert isinstance(meta, EntryMeta)
            assert meta.wl_hash
            assert meta.unit_type in ("function", "class", "block")

    @pytest.mark.parametrize(
        "method_name",
        ["get_node_count", "get_hierarchy_hashes", "get_meta"],
    )
    def test_missing_lookup_methods_return_none(self, method_name):
        """Lookup methods return None for missing entries."""
        store = EntryStore(max_resident=10)
        assert getattr(store, method_name)("nonexistent") is None


class TestEntryStoreBulkLoad:
    """Tests for bulk_load context manager."""

    def test_no_eviction_during_bulk_load(self):
        """bulk_load suppresses eviction during loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLitePersistence(db_path)

            index, entry_ids = _make_index_with_entries(10)
            persistence.save_full_index(index)

            store = EntryStore(max_resident=3)
            store.set_persistence(persistence)

            with store.bulk_load():
                _populate_store(store, index, entry_ids)

                # During bulk load, no eviction
                assert store.resident_count == len(entry_ids)

            # After bulk load, cache trimmed
            assert store.resident_count <= 3
            assert store.total_count == len(entry_ids)

            persistence.close()


class TestEntryStoreDictInterface:
    """Tests for dict-compatible interface."""

    @staticmethod
    def _assert_contains_all(store: EntryStore, entry_ids: list[str]) -> None:
        assert all(eid in store for eid in entry_ids)

    def test_clear(self):
        """clear() removes all entries from cache, metadata, and all_ids."""
        index, entry_ids = _make_index_with_entries(5)
        store = EntryStore(max_resident=100)

        _populate_store(store, index, entry_ids)

        assert len(store) > 0
        store.clear()
        assert len(store) == 0
        assert store.resident_count == 0
        assert store.total_count == 0

    def test_bool_empty(self):
        """Empty store is falsy."""
        store = EntryStore(max_resident=10)
        assert not store

    def test_bool_populated(self):
        """Populated store is truthy."""
        index, entry_ids = _make_index_with_entries(1)
        store = EntryStore(max_resident=10)
        store[entry_ids[0]] = index.entries[entry_ids[0]]
        assert store

    def test_contains_cached(self):
        """__contains__ works for cached entries."""
        index, entry_ids = _make_index_with_entries(3)
        store = EntryStore(max_resident=100)

        _populate_store(store, index, entry_ids)

        self._assert_contains_all(store, entry_ids)

    def test_contains_evicted(self):
        """__contains__ returns True for evicted entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLitePersistence(db_path)

            index, entry_ids = _make_index_with_entries(10)
            persistence.save_full_index(index)

            store = EntryStore(max_resident=3)
            store.set_persistence(persistence)

            _populate_store(store, index, entry_ids)

            # All entries should be "in" the store even if evicted from cache
            self._assert_contains_all(store, entry_ids)

            persistence.close()

    def test_contains_missing(self):
        """__contains__ returns False for non-existent entries."""
        store = EntryStore(max_resident=10)
        assert "nonexistent" not in store

    def test_delitem(self):
        """__delitem__ removes from cache, metadata, and all_ids."""
        index, entry_ids = _make_index_with_entries(3)
        store = EntryStore(max_resident=100)

        _populate_store(store, index, entry_ids)

        eid_to_remove = entry_ids[0]
        del store[eid_to_remove]

        assert eid_to_remove not in store
        assert store.get_meta(eid_to_remove) is None
        assert len(store) == len(entry_ids) - 1

    def test_get_default(self):
        """get() returns default for missing entries."""
        store = EntryStore(max_resident=10)
        assert store.get("nonexistent") is None
        assert store.get("nonexistent", None) is None

    def test_items_includes_all(self):
        """items() yields both cached and evicted entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLitePersistence(db_path)

            index, entry_ids = _make_index_with_entries(10)
            persistence.save_full_index(index)

            store = EntryStore(max_resident=3)
            store.set_persistence(persistence)

            _populate_store(store, index, entry_ids)

            all_items = list(store.items())
            assert len(all_items) == len(entry_ids)

            yielded_ids = {eid for eid, _ in all_items}
            assert yielded_ids == set(entry_ids)

            persistence.close()

    def test_values_includes_all(self):
        """values() yields both cached and evicted entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLitePersistence(db_path)

            index, entry_ids = _make_index_with_entries(10)
            persistence.save_full_index(index)

            store = EntryStore(max_resident=3)
            store.set_persistence(persistence)

            _populate_store(store, index, entry_ids)

            all_values = list(store.values())
            assert len(all_values) == len(entry_ids)

            persistence.close()

    def test_iter(self):
        """__iter__ yields all entry IDs."""
        index, entry_ids = _make_index_with_entries(5)
        store = EntryStore(max_resident=100)

        _populate_store(store, index, entry_ids)

        iterated = set(store)
        assert iterated == set(entry_ids)

    def test_pop_existing(self):
        """pop() removes and returns an existing entry."""
        index, entry_ids = _make_index_with_entries(3)
        store = EntryStore(max_resident=100)

        _populate_store(store, index, entry_ids)

        eid = entry_ids[0]
        entry = store.pop(eid)
        assert entry is not None
        assert eid not in store

    def test_pop_missing_with_default(self):
        """pop() returns default for missing entries."""
        store = EntryStore(max_resident=10)
        result = store.pop("nonexistent", None)
        assert result is None

    def test_pop_missing_raises(self):
        """pop() raises KeyError for missing entries without default."""
        store = EntryStore(max_resident=10)
        pytest.raises(KeyError, store.pop, "nonexistent")


class TestEntryStoreEnvironment:
    """Tests for environment variable configuration."""

    def test_default_max_resident(self):
        """Default max_resident is used when env var not set."""
        # Remove env var if set
        old_val = os.environ.pop("ASTROGRAPH_MAX_ENTRIES", None)
        try:
            store = EntryStore()
            assert store._max_resident == 50_000
        finally:
            if old_val is not None:
                os.environ["ASTROGRAPH_MAX_ENTRIES"] = old_val

    def test_env_var_max_resident(self):
        """max_resident reads from ASTROGRAPH_MAX_ENTRIES env var."""
        old_val = os.environ.get("ASTROGRAPH_MAX_ENTRIES")
        os.environ["ASTROGRAPH_MAX_ENTRIES"] = "1000"
        try:
            store = EntryStore()
            assert store._max_resident == 1000
        finally:
            if old_val is not None:
                os.environ["ASTROGRAPH_MAX_ENTRIES"] = old_val
            else:
                del os.environ["ASTROGRAPH_MAX_ENTRIES"]

    def test_explicit_max_resident(self):
        """Explicit max_resident overrides env var."""
        store = EntryStore(max_resident=42)
        assert store._max_resident == 42

    def test_unlimited_max_resident(self):
        """max_resident=0 means unlimited."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            persistence = SQLitePersistence(db_path)

            index, entry_ids = _make_index_with_entries(10)
            persistence.save_full_index(index)

            store = EntryStore(max_resident=0)
            store.set_persistence(persistence)

            _populate_store(store, index, entry_ids)

            # With max_resident=0, no eviction even with persistence
            assert store.resident_count == len(entry_ids)

            persistence.close()
