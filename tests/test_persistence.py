"""Tests for SQLite persistence resilience and recovery."""

import threading
import time
from pathlib import Path

from astrograph.index import CodeStructureIndex
from astrograph.persistence import SQLitePersistence


def test_recovers_from_corrupt_db_file(tmp_path: Path) -> None:
    """A malformed DB file is quarantined and replaced with a fresh DB."""
    db_path = tmp_path / "index.db"
    db_path.write_bytes(b"not-a-valid-sqlite-db")

    persistence = SQLitePersistence(db_path)
    try:
        cursor = persistence.conn.execute("SELECT 1")
        assert cursor.fetchone()[0] == 1
    finally:
        persistence.close()

    backups = list(tmp_path.glob("index.db.corrupt.*"))
    assert backups


def test_recovers_with_wal_and_shm_artifacts(tmp_path: Path) -> None:
    """Recovery also quarantines stale WAL/SHM siblings."""
    db_path = tmp_path / "index.db"
    db_path.write_bytes(b"bad-db")

    wal_path = Path(str(db_path) + "-wal")
    shm_path = Path(str(db_path) + "-shm")
    wal_path.write_bytes(b"bad-wal")
    shm_path.write_bytes(b"bad-shm")

    persistence = SQLitePersistence(db_path)
    try:
        cursor = persistence.conn.execute("SELECT name FROM sqlite_master LIMIT 1")
        assert cursor is not None
    finally:
        persistence.close()

    db_backups = list(tmp_path.glob("index.db.corrupt.*"))
    assert db_backups


def test_closed_persistence_operations_are_safe(tmp_path: Path) -> None:
    """Closed persistence should no-op for writes and return empty reads."""
    db_path = tmp_path / "index.db"
    persistence = SQLitePersistence(db_path)
    persistence.close()

    # Read operations should return empty defaults.
    assert persistence.get_entries_for_file("x.py") == []
    assert persistence.get_suppressions() == []
    assert persistence.get_file_metadata("x.py") is None
    assert persistence.get_all_indexed_files() == []
    assert persistence.get_entry("missing") is None
    assert list(persistence.get_entries_batch({"a", "b"})) == []
    assert list(persistence.iter_entries()) == []
    assert persistence.load_full_index(CodeStructureIndex()) is False
    assert persistence.get_stats() == {
        "entry_count": 0,
        "file_count": 0,
        "suppression_count": 0,
        "db_size_bytes": 0,
    }

    # Write/maintenance operations should be safe no-ops.
    persistence.delete_file_entries("x.py")
    persistence.delete_suppression("deadbeef")
    persistence.save_full_index(CodeStructureIndex())
    persistence.save_index_metadata(CodeStructureIndex())
    persistence.vacuum()


def test_close_races_with_background_access_without_exceptions(tmp_path: Path) -> None:
    """Concurrent DB access while closing should not throw from worker threads."""
    db_path = tmp_path / "index.db"
    persistence = SQLitePersistence(db_path)

    source_file = tmp_path / "sample.py"
    source_file.write_text("def add(a, b):\n    return a + b\n")
    index = CodeStructureIndex()
    index.index_file(str(source_file))
    metadata = index.file_metadata[str(source_file)]
    entries = [index.entries[eid] for eid in index.file_entries[str(source_file)]]

    stop = threading.Event()
    errors: list[Exception] = []

    def _worker() -> None:
        while not stop.is_set():
            try:
                persistence.save_file_entries(str(source_file), entries, metadata)
                persistence.get_entries_for_file(str(source_file))
                persistence.get_stats()
            except Exception as exc:  # pragma: no cover - asserts capture this
                errors.append(exc)
                stop.set()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    time.sleep(0.05)

    persistence.close()
    stop.set()
    thread.join(timeout=2.0)

    assert not errors
