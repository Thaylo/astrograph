"""
SQLite-based persistence layer for event-driven indexing.

Provides incremental updates, WAL mode for concurrent access,
and crash safety for the code structure index.
"""

import json
import logging
import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .index import CodeStructureIndex, FileMetadata, IndexEntry, SuppressionInfo

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 1


class SQLitePersistence:
    """
    SQLite persistence layer with incremental update support.

    Features:
    - WAL mode for concurrent read/write
    - Incremental updates (only changed entries)
    - ACID guarantees for crash safety
    - Efficient batch operations
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazy connection with WAL mode."""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,  # Autocommit for WAL
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self._conn.execute("PRAGMA temp_store=MEMORY")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );

            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                wl_hash TEXT NOT NULL,
                pattern_hash TEXT NOT NULL,
                data TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS file_metadata (
                file_path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                content_hash TEXT NOT NULL,
                indexed_at REAL NOT NULL,
                entry_count INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS suppressed_hashes (
                wl_hash TEXT PRIMARY KEY,
                reason TEXT,
                created_at REAL NOT NULL,
                data TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_entries_file ON entries(file_path);
            CREATE INDEX IF NOT EXISTS idx_entries_wl_hash ON entries(wl_hash);
            CREATE INDEX IF NOT EXISTS idx_entries_pattern_hash ON entries(pattern_hash);
        """
        )

        # Check/set schema version
        cursor = self.conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            self.conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.vacuum()
            self._conn.close()
            self._conn = None

    # =========================================================================
    # Entry Operations (incremental)
    # =========================================================================

    def save_file_entries(
        self,
        file_path: str,
        entries: list["IndexEntry"],
        metadata: "FileMetadata",
    ) -> None:
        """
        Save entries for a single file (incremental update).

        Deletes old entries for this file and inserts new ones.
        This is the key operation for event-driven updates.
        """
        conn = self.conn

        # Use explicit transaction for atomicity
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Delete old entries for this file
            conn.execute("DELETE FROM entries WHERE file_path = ?", (file_path,))

            # Insert new entries
            if entries:
                conn.executemany(
                    "INSERT INTO entries (id, file_path, wl_hash, pattern_hash, data) VALUES (?, ?, ?, ?, ?)",
                    [
                        (e.id, file_path, e.wl_hash, e.pattern_hash, json.dumps(e.to_dict()))
                        for e in entries
                    ],
                )

            # Update file metadata
            conn.execute(
                """INSERT OR REPLACE INTO file_metadata
                   (file_path, mtime, content_hash, indexed_at, entry_count)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    file_path,
                    metadata.mtime,
                    metadata.content_hash,
                    metadata.indexed_at,
                    metadata.entry_count,
                ),
            )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def delete_file_entries(self, file_path: str) -> None:
        """Delete all entries for a file (when file is deleted)."""
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute("DELETE FROM entries WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM file_metadata WHERE file_path = ?", (file_path,))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def get_entries_for_file(self, file_path: str) -> list[dict]:
        """Get all entries for a specific file."""
        cursor = self.conn.execute("SELECT data FROM entries WHERE file_path = ?", (file_path,))
        return [json.loads(row["data"]) for row in cursor.fetchall()]

    # =========================================================================
    # Suppression Operations (incremental)
    # =========================================================================

    def save_suppression(self, info: "SuppressionInfo") -> None:
        """Save a single suppression (incremental)."""
        self.conn.execute(
            """INSERT OR REPLACE INTO suppressed_hashes
               (wl_hash, reason, created_at, data)
               VALUES (?, ?, ?, ?)""",
            (info.wl_hash, info.reason, info.created_at, json.dumps(info.to_dict())),
        )

    def delete_suppression(self, wl_hash: str) -> None:
        """Delete a suppression."""
        self.conn.execute("DELETE FROM suppressed_hashes WHERE wl_hash = ?", (wl_hash,))

    def get_suppressions(self) -> list[dict]:
        """Get all suppressions."""
        cursor = self.conn.execute("SELECT data FROM suppressed_hashes")
        return [json.loads(row["data"]) for row in cursor.fetchall()]

    # =========================================================================
    # Bulk Operations (for full save/load)
    # =========================================================================

    def save_full_index(self, index: "CodeStructureIndex") -> None:
        """Save the entire index (used for initial persistence or migration)."""
        conn = self.conn

        conn.execute("BEGIN IMMEDIATE")
        try:
            # Clear existing data
            conn.execute("DELETE FROM entries")
            conn.execute("DELETE FROM file_metadata")
            conn.execute("DELETE FROM suppressed_hashes")
            conn.execute("DELETE FROM index_metadata")

            # Save entries
            if index.entries:
                conn.executemany(
                    "INSERT INTO entries (id, file_path, wl_hash, pattern_hash, data) VALUES (?, ?, ?, ?, ?)",
                    [
                        (
                            eid,
                            e.code_unit.file_path,
                            e.wl_hash,
                            e.pattern_hash,
                            json.dumps(e.to_dict()),
                        )
                        for eid, e in index.entries.items()
                    ],
                )

            # Save file metadata
            if index.file_metadata:
                conn.executemany(
                    """INSERT INTO file_metadata
                       (file_path, mtime, content_hash, indexed_at, entry_count)
                       VALUES (?, ?, ?, ?, ?)""",
                    [
                        (fm.file_path, fm.mtime, fm.content_hash, fm.indexed_at, fm.entry_count)
                        for fm in index.file_metadata.values()
                    ],
                )

            # Save suppressions
            for wl_hash, info in index.suppression_details.items():
                conn.execute(
                    """INSERT INTO suppressed_hashes (wl_hash, reason, created_at, data)
                       VALUES (?, ?, ?, ?)""",
                    (wl_hash, info.reason, info.created_at, json.dumps(info.to_dict())),
                )

            # Save metadata
            conn.executemany(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                [
                    ("entry_counter", str(index._entry_counter)),
                    ("block_entry_count", str(index._block_entry_count)),
                    ("function_entry_count", str(index._function_entry_count)),
                    ("saved_at", str(time.time())),
                ],
            )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def load_full_index(self, index: "CodeStructureIndex") -> bool:
        """
        Load the entire index from database.

        Returns True if data was loaded, False if database is empty.
        """
        from .index import FileMetadata, IndexEntry, SuppressionInfo

        # Check if we have data
        cursor = self.conn.execute("SELECT COUNT(*) FROM entries")
        if cursor.fetchone()[0] == 0:
            return False

        # Load entries
        cursor = self.conn.execute("SELECT id, data FROM entries")
        index.entries.clear()
        index.hash_buckets.clear()
        index.pattern_buckets.clear()
        index.block_buckets.clear()
        index.block_type_index.clear()
        index.fingerprint_index.clear()
        index.file_entries.clear()
        index._block_entry_count = 0
        index._function_entry_count = 0

        with index.entries.bulk_load():
            for row in cursor.fetchall():
                entry = IndexEntry.from_dict(json.loads(row["data"]))
                index.entries[row["id"]] = entry

        for eid, entry in index.entries.items():
            is_block = entry.code_unit.unit_type == "block"

            if is_block:
                index.block_buckets.setdefault(entry.wl_hash, set()).add(eid)
                if entry.code_unit.block_type:
                    index.block_type_index.setdefault(entry.code_unit.block_type, set()).add(eid)
                index._block_entry_count += 1
            else:
                index.hash_buckets.setdefault(entry.wl_hash, set()).add(eid)
                index.pattern_buckets.setdefault(entry.pattern_hash, set()).add(eid)
                index._function_entry_count += 1

            # Fingerprint index
            if "n_nodes" in entry.fingerprint:
                fp_key = (entry.fingerprint["n_nodes"], entry.fingerprint["n_edges"])
                index.fingerprint_index.setdefault(fp_key, set()).add(eid)

            # File entries
            index.file_entries.setdefault(entry.code_unit.file_path, []).append(eid)

        # Load file metadata
        cursor = self.conn.execute(
            "SELECT file_path, mtime, content_hash, indexed_at, entry_count FROM file_metadata"
        )
        index.file_metadata = {
            row["file_path"]: FileMetadata(
                file_path=row["file_path"],
                mtime=row["mtime"],
                content_hash=row["content_hash"],
                indexed_at=row["indexed_at"],
                entry_count=row["entry_count"],
            )
            for row in cursor.fetchall()
        }

        # Load suppressions
        cursor = self.conn.execute("SELECT wl_hash, data FROM suppressed_hashes")
        index.suppressed_hashes = set()
        index.suppression_details = {}
        for row in cursor.fetchall():
            wl_hash = row["wl_hash"]
            index.suppressed_hashes.add(wl_hash)
            index.suppression_details[wl_hash] = SuppressionInfo.from_dict(json.loads(row["data"]))

        # Load metadata
        cursor = self.conn.execute("SELECT key, value FROM index_metadata")
        metadata = {row["key"]: row["value"] for row in cursor.fetchall()}
        index._entry_counter = int(metadata.get("entry_counter", 0))

        return True

    def get_file_metadata(self, file_path: str) -> dict | None:
        """Get metadata for a specific file."""
        cursor = self.conn.execute(
            "SELECT mtime, content_hash, indexed_at, entry_count FROM file_metadata WHERE file_path = ?",
            (file_path,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "file_path": file_path,
            "mtime": row["mtime"],
            "content_hash": row["content_hash"],
            "indexed_at": row["indexed_at"],
            "entry_count": row["entry_count"],
        }

    def get_all_indexed_files(self) -> list[str]:
        """Get list of all indexed file paths."""
        cursor = self.conn.execute("SELECT file_path FROM file_metadata")
        return [row["file_path"] for row in cursor.fetchall()]

    def get_stats(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM entries")
        entry_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM file_metadata")
        file_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM suppressed_hashes")
        suppression_count = cursor.fetchone()[0]

        # Get database file size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        wal_path = Path(str(self.db_path) + "-wal")
        if wal_path.exists():
            db_size += wal_path.stat().st_size

        return {
            "entry_count": entry_count,
            "file_count": file_count,
            "suppression_count": suppression_count,
            "db_size_bytes": db_size,
        }

    # =========================================================================
    # Single/Batch Entry Lookup (for LRU cache misses)
    # =========================================================================

    def get_entry(self, entry_id: str) -> "IndexEntry | None":
        """Load a single entry by primary key (for LRU cache miss)."""
        from .index import IndexEntry

        cursor = self.conn.execute("SELECT data FROM entries WHERE id = ?", (entry_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return IndexEntry.from_dict(json.loads(row["data"]))

    def get_entries_batch(self, entry_ids: set[str]) -> Iterator[tuple[str, "IndexEntry"]]:
        """Load multiple entries by ID (batch lookup for iteration)."""
        from .index import IndexEntry

        if not entry_ids:
            return

        # SQLite has a limit on the number of variables; chunk if needed
        ids_list = list(entry_ids)
        chunk_size = 500
        for i in range(0, len(ids_list), chunk_size):
            chunk = ids_list[i : i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cursor = self.conn.execute(
                f"SELECT id, data FROM entries WHERE id IN ({placeholders})",  # noqa: S608
                chunk,
            )
            for row in cursor.fetchall():
                yield row["id"], IndexEntry.from_dict(json.loads(row["data"]))

    def iter_entries(self) -> Iterator[tuple[str, "IndexEntry"]]:
        """Stream all entries from the database."""
        from .index import IndexEntry

        cursor = self.conn.execute("SELECT id, data FROM entries")
        for row in cursor.fetchall():
            yield row["id"], IndexEntry.from_dict(json.loads(row["data"]))

    # =========================================================================
    # Maintenance
    # =========================================================================

    def vacuum(self) -> None:
        """Run VACUUM to reclaim unused space in the database."""
        if self._conn is not None:
            try:
                self._conn.execute("VACUUM")
            except sqlite3.OperationalError:
                # VACUUM can fail if a transaction is active
                logger.debug("VACUUM skipped (transaction active)")
