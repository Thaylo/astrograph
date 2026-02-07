"""Tests for SQLite persistence resilience and recovery."""

from pathlib import Path

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
