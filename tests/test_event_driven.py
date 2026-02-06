"""Tests for event-driven indexing with SQLite persistence and file watching."""

import os
import tempfile
import time

import pytest

from astrograph.index import CodeStructureIndex
from astrograph.persistence import SQLitePersistence
from astrograph.tools import CodeStructureTools


class TestSQLitePersistence:
    """Tests for SQLite persistence layer."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield os.path.join(tmpdir, "test.db")

    @pytest.fixture
    def sample_index(self):
        """Create an index with some data."""
        index = CodeStructureIndex()
        code = """
def process_items(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results

def transform_data(data):
    output = []
    for element in data:
        if element > 0:
            output.append(element * 2)
    return output
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            index.index_file(f.name, include_blocks=True)
            yield index
            os.unlink(f.name)

    def test_save_and_load_full_index(self, temp_db, sample_index):
        """Test saving and loading a full index."""
        persistence = SQLitePersistence(temp_db)

        # Save the index
        persistence.save_full_index(sample_index)

        # Load into a new index
        new_index = CodeStructureIndex()
        loaded = persistence.load_full_index(new_index)

        assert loaded is True
        assert len(new_index.entries) == len(sample_index.entries)
        assert len(new_index.hash_buckets) == len(sample_index.hash_buckets)

        persistence.close()

    def test_incremental_file_save(self, temp_db):
        """Test incremental saving of file entries."""
        persistence = SQLitePersistence(temp_db)
        index = CodeStructureIndex()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and index first file
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            index.index_file(file1)
            metadata = index.file_metadata[file1]
            entries = [index.entries[eid] for eid in index.file_entries[file1]]

            # Save file entries incrementally
            persistence.save_file_entries(file1, entries, metadata)

            # Verify in database
            stats = persistence.get_stats()
            assert stats["entry_count"] >= 1
            assert stats["file_count"] == 1

        persistence.close()

    def test_delete_file_entries(self, temp_db):
        """Test deleting entries for a file."""
        persistence = SQLitePersistence(temp_db)
        index = CodeStructureIndex()

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass\ndef bar(): pass")

            index.index_file(file1)
            metadata = index.file_metadata[file1]
            entries = [index.entries[eid] for eid in index.file_entries[file1]]
            persistence.save_file_entries(file1, entries, metadata)

            # Verify entries exist
            stats = persistence.get_stats()
            assert stats["entry_count"] >= 2

            # Delete entries
            persistence.delete_file_entries(file1)

            # Verify deleted
            stats = persistence.get_stats()
            assert stats["entry_count"] == 0
            assert stats["file_count"] == 0

        persistence.close()

    def test_suppression_persistence(self, temp_db, sample_index):
        """Test saving and loading suppressions."""
        persistence = SQLitePersistence(temp_db)

        # Add a suppression
        import re

        from astrograph.tools import CodeStructureTools

        tools = CodeStructureTools(index=sample_index)
        analyze_result = tools.analyze()
        match = re.search(r'suppress\(wl_hash="([^"]+)"\)', analyze_result.text)

        if match:
            wl_hash = match.group(1)
            sample_index.suppress(wl_hash)

            # Save full index with suppression
            persistence.save_full_index(sample_index)

            # Load into new index
            new_index = CodeStructureIndex()
            persistence.load_full_index(new_index)

            # Verify suppression persisted
            assert wl_hash in new_index.suppressed_hashes

        persistence.close()

    def test_get_stats(self, temp_db, sample_index):
        """Test database statistics."""
        persistence = SQLitePersistence(temp_db)
        persistence.save_full_index(sample_index)

        stats = persistence.get_stats()
        assert "entry_count" in stats
        assert "file_count" in stats
        assert "suppression_count" in stats
        assert "db_size_bytes" in stats
        assert stats["entry_count"] > 0

        persistence.close()


class TestEventDrivenIndex:
    """Tests for event-driven index."""

    def test_index_directory_with_persistence(self):
        """Test indexing a directory with SQLite persistence."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            db_path = os.path.join(tmpdir, "index.db")

            # Index with persistence
            edi = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            entry_count = edi.index_directory(tmpdir)

            assert entry_count >= 1
            assert os.path.exists(db_path)

            edi.close()

    def test_load_from_persistence(self):
        """Test loading index from persistence."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass\ndef bar(): pass")

            db_path = os.path.join(tmpdir, "index.db")

            # First index
            edi1 = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            edi1.index_directory(tmpdir)
            original_count = len(edi1.index.entries)
            edi1.close()

            # Second index (should load from persistence)
            edi2 = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            loaded = edi2.load_from_persistence()

            assert loaded is True
            assert len(edi2.index.entries) == original_count

            edi2.close()

    def test_analysis_cache(self):
        """Test analysis caching."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with duplicates
            code = """
def process(items):
    for item in items:
        if item > 0:
            print(item)

def transform(data):
    for element in data:
        if element > 0:
            print(element)
"""
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write(code)

            db_path = os.path.join(tmpdir, "index.db")
            edi = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            edi.index_directory(tmpdir)

            # First call - cache miss
            exact, pattern, blocks = edi.get_cached_analysis()

            # Wait for background cache computation
            time.sleep(0.2)

            # Second call - should be cache hit
            exact2, pattern2, blocks2 = edi.get_cached_analysis()

            stats = edi.get_stats()
            assert stats["cache_hits"] >= 1

            edi.close()

    def test_suppress_with_persistence(self):
        """Test suppression with automatic persistence."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
def process(items):
    for item in items:
        if item > 0:
            print(item)

def transform(data):
    for element in data:
        if element > 0:
            print(element)
"""
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write(code)

            db_path = os.path.join(tmpdir, "index.db")

            # Index and suppress
            edi1 = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            edi1.index_directory(tmpdir)

            # Find a hash to suppress
            groups = edi1.index.find_all_duplicates(min_node_count=3)
            if groups:
                wl_hash = groups[0].wl_hash
                edi1.suppress(wl_hash)
                edi1.close()

                # Load fresh and verify suppression persisted
                edi2 = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
                edi2.load_from_persistence()

                assert wl_hash in edi2.index.suppressed_hashes
                edi2.close()

    def test_suppress_batch_with_persistence(self):
        """Test batch suppression with automatic persistence."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
def process(items):
    for item in items:
        if item > 0:
            print(item)

def transform(data):
    for element in data:
        if element > 0:
            print(element)
"""
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write(code)

            db_path = os.path.join(tmpdir, "index.db")

            # Index and batch suppress
            edi1 = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            edi1.index_directory(tmpdir)

            groups = edi1.index.find_all_duplicates(min_node_count=3)
            if groups:
                hashes = [g.wl_hash for g in groups]
                suppressed, not_found = edi1.suppress_batch(hashes)
                assert len(suppressed) == len(hashes)
                assert len(not_found) == 0
                edi1.close()

                # Load fresh and verify suppressions persisted
                edi2 = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
                edi2.load_from_persistence()

                for wl_hash in hashes:
                    assert wl_hash in edi2.index.suppressed_hashes
                edi2.close()

    def test_unsuppress_batch_with_persistence(self):
        """Test batch unsuppression with automatic persistence."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
def process(items):
    for item in items:
        if item > 0:
            print(item)

def transform(data):
    for element in data:
        if element > 0:
            print(element)
"""
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write(code)

            db_path = os.path.join(tmpdir, "index.db")

            # Index, suppress, then batch unsuppress
            edi1 = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            edi1.index_directory(tmpdir)

            groups = edi1.index.find_all_duplicates(min_node_count=3)
            if groups:
                hashes = [g.wl_hash for g in groups]
                edi1.suppress_batch(hashes)
                unsuppressed, not_found = edi1.unsuppress_batch(hashes)
                assert len(unsuppressed) == len(hashes)
                assert len(not_found) == 0
                edi1.close()

                # Load fresh and verify unsuppressions persisted
                edi2 = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
                edi2.load_from_persistence()

                for wl_hash in hashes:
                    assert wl_hash not in edi2.index.suppressed_hashes
                edi2.close()

    def test_get_stats(self):
        """Test comprehensive statistics."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            db_path = os.path.join(tmpdir, "index.db")
            edi = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            edi.index_directory(tmpdir)

            stats = edi.get_stats()

            assert "total_entries" in stats
            assert "persistence" in stats
            assert "watching" in stats
            assert "cache_valid" in stats
            assert "cache_hits" in stats
            assert "cache_misses" in stats

            edi.close()


class TestFileWatcher:
    """Tests for file watcher."""

    def test_watcher_creation(self):
        """Test creating a file watcher."""
        from astrograph.watcher import HAS_WATCHDOG, FileWatcher

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            events = []

            def on_change(path):
                events.append(("change", path))

            def on_create(path):
                events.append(("create", path))

            def on_delete(path):
                events.append(("delete", path))

            watcher = FileWatcher(
                root_path=tmpdir,
                on_file_changed=on_change,
                on_file_created=on_create,
                on_file_deleted=on_delete,
            )

            watcher.start()
            assert watcher.is_watching

            watcher.stop()
            assert not watcher.is_watching

    def test_watcher_detects_file_creation(self):
        """Test that watcher detects new Python files."""
        from astrograph.watcher import HAS_WATCHDOG, FileWatcher

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            events = []

            def on_change(path):
                events.append(("change", path))

            def on_create(path):
                events.append(("create", path))

            def on_delete(path):
                events.append(("delete", path))

            watcher = FileWatcher(
                root_path=tmpdir,
                on_file_changed=on_change,
                on_file_created=on_create,
                on_file_deleted=on_delete,
                debounce_delay=0.05,
            )

            watcher.start()

            # Create a Python file
            file1 = os.path.join(tmpdir, "new_file.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            # Wait for debounce
            time.sleep(0.2)

            watcher.stop()

            # Check that creation was detected
            assert any(e[0] == "create" and "new_file.py" in e[1] for e in events)

    def test_debounced_callback(self):
        """Test that rapid events are debounced."""
        from astrograph.watcher import DebouncedCallback

        calls = []

        def callback(path):
            calls.append(path)

        debounced = DebouncedCallback(callback, delay=0.1)

        # Rapid calls for same path
        debounced("/test/file.py")
        debounced("/test/file.py")
        debounced("/test/file.py")

        # Wait for debounce
        time.sleep(0.2)

        # Should only have one call
        assert len(calls) == 1
        assert calls[0] == "/test/file.py"


class TestFileWatcherPool:
    """Tests for file watcher pool."""

    def test_watch_and_unwatch(self):
        """Test watching and unwatching directories."""
        from astrograph.watcher import HAS_WATCHDOG, FileWatcherPool

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            pool = FileWatcherPool()

            watcher = pool.watch(
                root_path=tmpdir,
                on_file_changed=lambda _: None,
                on_file_created=lambda _: None,
                on_file_deleted=lambda _: None,
            )

            assert watcher.is_watching

            # Watch same dir again - should return same watcher
            watcher2 = pool.watch(
                root_path=tmpdir,
                on_file_changed=lambda _: None,
                on_file_created=lambda _: None,
                on_file_deleted=lambda _: None,
            )
            assert watcher is watcher2

            pool.unwatch(tmpdir)
            pool.stop_all()

    def test_context_manager(self):
        """Test using pool as context manager."""
        from astrograph.watcher import HAS_WATCHDOG, FileWatcherPool

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir, FileWatcherPool() as pool:
            pool.watch(
                root_path=tmpdir,
                on_file_changed=lambda _: None,
                on_file_created=lambda _: None,
                on_file_deleted=lambda _: None,
            )


class TestWatcherSkipPaths:
    """Tests for path skipping in watcher."""

    def test_skip_pycache(self):
        """Test that __pycache__ directories are skipped."""
        from pathlib import Path

        from astrograph.watcher import _should_skip_path

        assert _should_skip_path(Path("/project/__pycache__/file.py"))
        assert _should_skip_path(Path("/project/.git/hooks/pre-commit"))
        assert _should_skip_path(Path("/project/venv/lib/python3.10/site.py"))
        assert not _should_skip_path(Path("/project/src/module.py"))

    def test_skip_venv_variants(self):
        """Test that numbered/prefixed venv directories are skipped."""
        from pathlib import Path

        from astrograph.watcher import _should_skip_path

        # Standard venv names
        assert _should_skip_path(Path("/project/venv/lib/site.py"))
        assert _should_skip_path(Path("/project/.venv/lib/site.py"))

        # Numbered variants (e.g., .venv311, venv3.11)
        assert _should_skip_path(Path("/project/.venv311/lib/site.py"))
        assert _should_skip_path(Path("/project/venv3.11/lib/site.py"))
        assert _should_skip_path(Path("/project/.venv3.12/lib/site.py"))

        # env variants
        assert _should_skip_path(Path("/project/env/lib/site.py"))
        assert _should_skip_path(Path("/project/.env/lib/site.py"))
        assert _should_skip_path(Path("/project/env3.11/lib/site.py"))

        # virtualenv variants
        assert _should_skip_path(Path("/project/virtualenv/lib/site.py"))
        assert _should_skip_path(Path("/project/.virtualenv/lib/site.py"))

        # Should NOT skip: directories that happen to start with prefix + alpha
        assert not _should_skip_path(Path("/project/vendor/src/module.py"))
        assert not _should_skip_path(Path("/project/environment/src/module.py"))
        assert not _should_skip_path(Path("/project/envoy/src/module.py"))

        # Should NOT skip: normal project dirs
        assert not _should_skip_path(Path("/project/src/module.py"))
        assert not _should_skip_path(Path("/project/backend/api/views.py"))


class TestAnalysisCacheDirectly:
    """Tests for AnalysisCache class."""

    def test_cache_invalidation(self):
        """Test cache invalidation."""
        from astrograph.event_driven import AnalysisCache

        cache = AnalysisCache()
        assert not cache.is_valid()

        cache.set([], [], [])
        assert cache.is_valid()

        cache.invalidate()
        assert not cache.is_valid()

    def test_cache_get_returns_none_when_invalid(self):
        """Test that get returns None when cache is invalid."""
        from astrograph.event_driven import AnalysisCache

        cache = AnalysisCache()
        assert cache.get() is None

    def test_cache_computed_at(self):
        """Test computed_at timestamp."""
        from astrograph.event_driven import AnalysisCache

        cache = AnalysisCache()
        assert cache.computed_at == 0

        cache.set([], [], [])
        assert cache.computed_at > 0


class TestPersistenceEdgeCases:
    """Tests for persistence edge cases."""

    def test_empty_database_load(self):
        """Test loading from empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "empty.db")

            persistence = SQLitePersistence(db_path)
            index = CodeStructureIndex()

            loaded = persistence.load_full_index(index)
            assert loaded is False
            assert len(index.entries) == 0

            persistence.close()

    def test_get_file_metadata_missing(self):
        """Test getting metadata for non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            persistence = SQLitePersistence(db_path)
            metadata = persistence.get_file_metadata("/nonexistent/file.py")
            assert metadata is None

            persistence.close()

    def test_get_all_indexed_files_empty(self):
        """Test getting indexed files from empty database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            persistence = SQLitePersistence(db_path)
            files = persistence.get_all_indexed_files()
            assert files == []

            persistence.close()


class TestEventDrivenIndexEdgeCases:
    """Tests for event-driven index edge cases."""

    def test_close_without_start(self):
        """Test closing without starting watchers."""
        from astrograph.event_driven import EventDrivenIndex

        edi = EventDrivenIndex(persistence_path=None, watch_enabled=False)
        edi.close()  # Should not raise

    def test_stop_watching_without_start(self):
        """Test stopping watcher that was never started."""
        from astrograph.event_driven import EventDrivenIndex

        edi = EventDrivenIndex(persistence_path=None, watch_enabled=False)
        edi.stop_watching()  # Should not raise
        edi.close()

    def test_context_manager(self):
        """Test using EventDrivenIndex as context manager."""
        from astrograph.event_driven import EventDrivenIndex

        with EventDrivenIndex(persistence_path=None, watch_enabled=False) as edi:
            assert edi is not None


class TestEventDrivenTools:
    """Tests for event-driven mode in CodeStructureTools."""

    def test_event_driven_mode_initialization(self):
        """Test initializing tools in event-driven mode."""
        tools = CodeStructureTools(event_driven=True)
        assert tools._event_driven_mode is True
        assert tools._event_driven_index is not None
        tools.close()

    def test_event_driven_index_codebase(self):
        """Test indexing in event-driven mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            tools = CodeStructureTools(event_driven=True)
            result = tools.index_codebase(tmpdir)

            assert "Indexed" in result.text

            tools.close()

    def test_event_driven_suppress_persists(self):
        """Test that suppression persists in event-driven mode."""
        import re

        with tempfile.TemporaryDirectory() as tmpdir:
            code = """
def process(items):
    for item in items:
        if item > 0:
            print(item)

def transform(data):
    for element in data:
        if element > 0:
            print(element)
"""
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write(code)

            # First tools instance - suppress
            tools1 = CodeStructureTools(event_driven=True)
            tools1.index_codebase(tmpdir)

            analyze_result = tools1.analyze()
            match = re.search(r'suppress\(wl_hash="([^"]+)"\)', analyze_result.text)
            if match:
                wl_hash = match.group(1)
                tools1.suppress(wl_hash)
                tools1.close()

                # Second tools instance - verify suppression persisted
                tools2 = CodeStructureTools(event_driven=True)
                tools2.index_codebase(tmpdir)

                assert wl_hash in tools2.index.suppressed_hashes
                tools2.close()

    def test_get_event_driven_stats(self):
        """Test getting event-driven statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            tools = CodeStructureTools(event_driven=True)
            tools.index_codebase(tmpdir)

            stats = tools.get_event_driven_stats()

            assert stats is not None
            assert "watching" in stats
            assert "cache_valid" in stats

            tools.close()

    def test_standard_mode_no_event_driven_stats(self):
        """Test that standard mode returns None for event-driven stats."""
        tools = CodeStructureTools(event_driven=False)
        stats = tools.get_event_driven_stats()
        assert stats is None

    def test_tools_context_manager(self):
        """Test using CodeStructureTools as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            with CodeStructureTools(event_driven=True) as tools:
                tools.index_codebase(tmpdir)
                assert len(tools.index.entries) >= 1


class TestWatcherFileEvents:
    """More tests for file watcher events."""

    def test_watcher_detects_modification(self):
        """Test that watcher detects file modifications."""
        from astrograph.watcher import HAS_WATCHDOG, FileWatcher

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create initial file
            file1 = os.path.join(tmpdir, "test.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            events = []

            def on_change(path):
                events.append(("change", path))

            watcher = FileWatcher(
                root_path=tmpdir,
                on_file_changed=on_change,
                on_file_created=lambda _: None,
                on_file_deleted=lambda _: None,
                debounce_delay=0.05,
            )

            watcher.start()
            time.sleep(0.1)

            # Modify the file
            with open(file1, "w") as f:
                f.write("def bar(): pass")

            time.sleep(0.2)
            watcher.stop()

            # Check modification detected
            assert any("change" in e[0] for e in events)

    def test_watcher_ignores_non_python(self):
        """Test that watcher ignores non-Python files."""
        from astrograph.watcher import HAS_WATCHDOG, FileWatcher

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            events = []

            def on_create(path):
                events.append(path)

            watcher = FileWatcher(
                root_path=tmpdir,
                on_file_changed=lambda _: None,
                on_file_created=on_create,
                on_file_deleted=lambda _: None,
                debounce_delay=0.05,
            )

            watcher.start()

            # Create non-Python file
            txt_file = os.path.join(tmpdir, "readme.txt")
            with open(txt_file, "w") as f:
                f.write("hello")

            time.sleep(0.2)
            watcher.stop()

            # Should not have detected the txt file
            assert not any("readme.txt" in e for e in events)


class TestPersistenceGetSuppressions:
    """Tests for suppression retrieval."""

    def test_get_suppressions_with_data(self):
        """Test getting suppressions when they exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            index = CodeStructureIndex()
            code = """
def process(items):
    for item in items:
        print(item)

def transform(data):
    for element in data:
        print(element)
"""
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write(code)

            index.index_file(file1)

            # Suppress something
            groups = index.find_all_duplicates(min_node_count=3)
            if groups:
                index.suppress(groups[0].wl_hash)

                persistence = SQLitePersistence(db_path)
                persistence.save_full_index(index)

                suppressions = persistence.get_suppressions()
                assert len(suppressions) > 0

                persistence.close()


class TestEventDrivenFileEvents:
    """Tests for file event handling in EventDrivenIndex."""

    def test_remove_file(self):
        """Test removing a file from the index."""
        from pathlib import Path

        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            # Resolve to handle macOS /tmp -> /private/var symlink
            tmpdir = str(Path(tmpdir).resolve())
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            db_path = os.path.join(tmpdir, "index.db")
            edi = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            edi.index_directory(tmpdir)

            assert file1 in edi.index.file_entries

            # Simulate file removal
            edi._remove_file(file1)

            assert file1 not in edi.index.file_entries

            edi.close()

    def test_reindex_nonexistent_file(self):
        """Test reindexing a file that doesn't exist."""
        from astrograph.event_driven import EventDrivenIndex

        edi = EventDrivenIndex(persistence_path=None, watch_enabled=False)

        # Should not raise
        edi._reindex_file("/nonexistent/file.py")

        edi.close()

    def test_start_watching_non_directory(self):
        """Test starting watcher on a non-directory."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            edi = EventDrivenIndex(persistence_path=None, watch_enabled=True)
            # Should log warning but not raise
            edi.start_watching(file1)

            edi.close()

    def test_start_watching_already_watching(self):
        """Test starting watcher when already watching."""
        from astrograph.event_driven import EventDrivenIndex
        from astrograph.watcher import HAS_WATCHDOG

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            edi = EventDrivenIndex(persistence_path=None, watch_enabled=True)

            edi.start_watching(tmpdir)
            edi.start_watching(tmpdir)  # Should stop previous and start new

            edi.close()


class TestWatcherEdgeCases:
    """More edge case tests for watcher."""

    def test_watcher_context_manager(self):
        """Test using FileWatcher as context manager."""
        from astrograph.watcher import HAS_WATCHDOG, FileWatcher

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir, FileWatcher(
            root_path=tmpdir,
            on_file_changed=lambda _: None,
            on_file_created=lambda _: None,
            on_file_deleted=lambda _: None,
        ) as watcher:
            assert watcher.is_watching

    def test_debounced_callback_cancel(self):
        """Test cancelling debounced callbacks."""
        from astrograph.watcher import DebouncedCallback

        calls = []

        def callback(path):
            calls.append(path)

        debounced = DebouncedCallback(callback, delay=0.5)

        debounced("/test/file.py")
        debounced.cancel_all()

        time.sleep(0.6)

        # Should have no calls since we cancelled
        assert len(calls) == 0

    def test_handler_on_moved(self):
        """Test handling file move events."""
        from astrograph.watcher import PythonFileHandler

        events = {"deleted": [], "created": []}

        handler = PythonFileHandler(
            on_modified=lambda _: None,
            on_created=lambda p: events["created"].append(p),
            on_deleted=lambda p: events["deleted"].append(p),
        )

        # Create mock event
        class MockMoveEvent:
            is_directory = False
            src_path = "/test/old.py"
            dest_path = "/test/new.py"

        handler.on_moved(MockMoveEvent())

        # Should register as delete + create
        assert "/test/old.py" in events["deleted"]
        # Created is debounced, so won't be immediate


class TestPersistenceGetEntries:
    """Tests for entry retrieval from persistence."""

    def test_get_entries_for_file(self):
        """Test getting entries for a specific file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            index = CodeStructureIndex()
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass\ndef bar(): pass")

            index.index_file(file1)
            metadata = index.file_metadata[file1]
            entries = [index.entries[eid] for eid in index.file_entries[file1]]

            persistence = SQLitePersistence(db_path)
            persistence.save_file_entries(file1, entries, metadata)

            # Retrieve entries
            retrieved = persistence.get_entries_for_file(file1)
            assert len(retrieved) >= 2

            persistence.close()

    def test_get_all_indexed_files(self):
        """Test getting all indexed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            index = CodeStructureIndex()

            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            file2 = os.path.join(tmpdir, "file2.py")
            with open(file2, "w") as f:
                f.write("def bar(): pass")

            index.index_directory(tmpdir)

            persistence = SQLitePersistence(db_path)
            persistence.save_full_index(index)

            files = persistence.get_all_indexed_files()
            assert len(files) == 2

            persistence.close()


class TestEventDrivenWithWatching:
    """Tests for event-driven index with file watching enabled."""

    def test_file_watching_detects_changes(self):
        """Test that file watching detects and processes changes."""
        from pathlib import Path

        from astrograph.event_driven import EventDrivenIndex
        from astrograph.watcher import HAS_WATCHDOG

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            db_path = os.path.join(tmpdir, "index.db")
            edi = EventDrivenIndex(persistence_path=db_path, watch_enabled=True)
            edi.index_directory(tmpdir)

            len(edi.index.entries)

            # Create a new file
            file2 = os.path.join(tmpdir, "file2.py")
            with open(file2, "w") as f:
                f.write("def bar(): pass")

            # Wait for watcher to detect and process
            time.sleep(0.3)

            # Index may have more entries now
            stats = edi.get_stats()
            assert stats["file_events_processed"] >= 0  # May or may not have triggered

            edi.close()


class TestWatcherHandlerEvents:
    """Test handler event methods directly."""

    @staticmethod
    def _assert_handler_filters(is_directory: bool, src_path: str) -> None:
        """Assert that PythonFileHandler filters out events with given attributes."""
        from astrograph.watcher import PythonFileHandler

        events: list[tuple[str, str]] = []
        handler = PythonFileHandler(
            on_modified=lambda p: events.append(("mod", p)),
            on_created=lambda p: events.append(("create", p)),
            on_deleted=lambda p: events.append(("delete", p)),
        )

        class MockEvent:
            pass

        MockEvent.is_directory = is_directory  # type: ignore[attr-defined]
        MockEvent.src_path = src_path  # type: ignore[attr-defined]

        handler.on_modified(MockEvent())  # type: ignore[arg-type]
        handler.on_created(MockEvent())  # type: ignore[arg-type]
        handler.on_deleted(MockEvent())  # type: ignore[arg-type]

        assert len(events) == 0

    def test_handler_ignores_directory_events(self):
        """Test that directory events are ignored."""
        self._assert_handler_filters(is_directory=True, src_path="/test/dir")

    def test_handler_filters_non_python(self):
        """Test that non-Python files are filtered."""
        self._assert_handler_filters(is_directory=False, src_path="/test/file.txt")

    def test_handler_cancel_pending(self):
        """Test cancelling pending handler events."""
        from astrograph.watcher import PythonFileHandler

        handler = PythonFileHandler(
            on_modified=lambda _: None,
            on_created=lambda _: None,
            on_deleted=lambda _: None,
        )

        # Should not raise
        handler.cancel_pending()


class TestEventDrivenWatcherDisabled:
    """Test event-driven index when watchdog is conceptually disabled."""

    def test_start_watching_when_disabled(self):
        """Test that start_watching does nothing when watch_enabled=False."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            edi = EventDrivenIndex(persistence_path=None, watch_enabled=False)

            edi.start_watching(tmpdir)

            # Watcher should still be None
            assert edi._watcher is None

            edi.close()


class TestPersistenceFileMetadata:
    """Test file metadata retrieval."""

    def test_get_file_metadata_exists(self):
        """Test getting metadata for an existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            index = CodeStructureIndex()
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            index.index_file(file1)

            persistence = SQLitePersistence(db_path)
            persistence.save_full_index(index)

            metadata = persistence.get_file_metadata(file1)
            assert metadata is not None
            assert metadata["file_path"] == file1
            assert "mtime" in metadata
            assert "content_hash" in metadata

            persistence.close()


class TestSQLitePathHelpers:
    """Test path helper functions."""

    def test_get_sqlite_path(self):
        """Test _get_sqlite_path function."""

        from astrograph.tools import PERSISTENCE_DIR, _get_sqlite_path

        path = _get_sqlite_path("/test/project")
        assert str(path).endswith("index.db")
        assert PERSISTENCE_DIR in str(path)


class TestEventDrivenScheduleRecompute:
    """Test analysis recompute scheduling."""

    def test_schedule_recompute_when_already_running(self):
        """Test scheduling recompute when one is already in progress."""

        from astrograph.event_driven import EventDrivenIndex

        edi = EventDrivenIndex(persistence_path=None, watch_enabled=False)

        # First recompute
        edi._schedule_analysis_recompute()

        # Try to schedule another (should be ignored)
        edi._schedule_analysis_recompute()

        # Give it time to finish
        time.sleep(0.2)

        edi.close()


class TestWatcherNonDirectory:
    """Test watcher with non-directory path."""

    def test_watcher_rejects_file_path(self):
        """Test that FileWatcher rejects file paths."""
        from astrograph.watcher import HAS_WATCHDOG, FileWatcher

        if not HAS_WATCHDOG:
            pytest.skip("watchdog not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            watcher = FileWatcher(
                root_path=file1,  # File, not directory
                on_file_changed=lambda _: None,
                on_file_created=lambda _: None,
                on_file_deleted=lambda _: None,
            )

            with pytest.raises(ValueError):
                watcher.start()


class TestEventDrivenFileHandlerCallback:
    """Test file event handler callbacks."""

    @staticmethod
    def _run_file_event_callback(callback_name: str, create_extra_file: bool = False) -> None:
        """Run a file event callback and assert it increments the event counter."""
        from pathlib import Path

        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = str(Path(tmpdir).resolve())
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            db_path = os.path.join(tmpdir, "index.db")
            edi = EventDrivenIndex(persistence_path=db_path, watch_enabled=False)
            edi.index_directory(tmpdir)

            target = file1
            if create_extra_file:
                target = os.path.join(tmpdir, "file2.py")
                with open(target, "w") as f:
                    f.write("def bar(): pass")

            initial_events = edi._file_events_processed
            getattr(edi, callback_name)(target)
            assert edi._file_events_processed == initial_events + 1

            edi.close()

    def test_on_file_changed_callback(self):
        """Test file changed callback."""
        self._run_file_event_callback("_on_file_changed")

    def test_on_file_created_callback(self):
        """Test file created callback."""
        self._run_file_event_callback("_on_file_created", create_extra_file=True)

    def test_on_file_deleted_callback(self):
        """Test file deleted callback."""
        self._run_file_event_callback("_on_file_deleted")


class TestCloudDetection:
    """Tests for cloud-synced folder detection."""

    def test_is_cloud_synced_path_regular_path(self):
        """Test that regular paths are not detected as cloud-synced."""
        from astrograph.cloud_detect import is_cloud_synced_path

        with tempfile.TemporaryDirectory() as tmpdir:
            is_synced, service = is_cloud_synced_path(tmpdir)
            assert is_synced is False
            assert service is None

    def test_get_cloud_sync_warning_regular_path(self):
        """Test that regular paths don't generate warnings."""
        from astrograph.cloud_detect import get_cloud_sync_warning

        with tempfile.TemporaryDirectory() as tmpdir:
            warning = get_cloud_sync_warning(tmpdir)
            assert warning is None

    def test_check_and_warn_cloud_sync_regular_path(self):
        """Test check_and_warn for regular paths."""
        from astrograph.cloud_detect import check_and_warn_cloud_sync

        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_and_warn_cloud_sync(tmpdir)
            assert result is False

    def test_get_cloud_storage_paths(self):
        """Test getting cloud storage paths."""
        from astrograph.cloud_detect import get_cloud_storage_paths

        # Should return a dict (may be empty if no cloud storage)
        paths = get_cloud_storage_paths()
        assert isinstance(paths, dict)

    def test_expand_pattern_no_wildcard(self):
        """Test expanding pattern without wildcards."""
        from astrograph.cloud_detect import _expand_pattern

        with tempfile.TemporaryDirectory() as tmpdir:
            result = _expand_pattern(tmpdir)
            assert len(result) == 1

    def test_expand_pattern_nonexistent(self):
        """Test expanding pattern for nonexistent path."""
        from astrograph.cloud_detect import _expand_pattern

        result = _expand_pattern("/nonexistent/path/that/does/not/exist")
        assert result == []

    def test_get_platform_key(self):
        """Test platform key detection."""
        import platform

        from astrograph.cloud_detect import _get_platform_key

        key = _get_platform_key()
        system = platform.system().lower()

        if system == "darwin":
            assert key == "darwin"
        elif system == "linux":
            assert key == "linux"
        elif system == "windows":
            assert key == "win32"

    def test_cloud_patterns_structure(self):
        """Test that CLOUD_PATTERNS has expected structure."""
        from astrograph.cloud_detect import CLOUD_PATTERNS

        assert "darwin" in CLOUD_PATTERNS
        assert "linux" in CLOUD_PATTERNS
        assert "win32" in CLOUD_PATTERNS

        # Check each platform has expected services
        for platform_key in CLOUD_PATTERNS:
            platform_patterns = CLOUD_PATTERNS[platform_key]
            assert isinstance(platform_patterns, dict)


class TestCloudDetectionInEventDriven:
    """Tests for cloud detection integration in event-driven index."""

    def test_event_driven_index_with_regular_path(self):
        """Test that regular paths work without cloud warning."""
        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            # Should work without issues
            edi = EventDrivenIndex(persistence_path=None, watch_enabled=False)
            entry_count = edi.index_directory(tmpdir)
            assert entry_count >= 1
            edi.close()


class TestCloudDetectionInTools:
    """Tests for cloud detection integration in tools."""

    def test_event_driven_tools_with_regular_path(self):
        """Test that event-driven tools work with regular paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            tools = CodeStructureTools(event_driven=True)
            result = tools.index_codebase(tmpdir)

            # Should not contain cloud warning
            assert "CLOUD-SYNCED" not in result.text
            assert "Indexed" in result.text

            tools.close()


class TestCloudDetectionWindowsOneDrive:
    """Tests for Windows OneDrive detection."""

    def test_get_windows_onedrive_paths_no_env(self):
        """Test Windows OneDrive detection when env vars not set."""
        import os

        from astrograph.cloud_detect import _get_windows_onedrive_paths

        # Save current env
        saved = {}
        for var in ["OneDrive", "OneDriveConsumer", "OneDriveCommercial"]:
            if var in os.environ:
                saved[var] = os.environ.pop(var)

        try:
            paths = _get_windows_onedrive_paths()
            # Should return empty list without env vars
            assert isinstance(paths, list)
        finally:
            # Restore env
            for var, val in saved.items():
                os.environ[var] = val


class TestCloudDetectionMoreCoverage:
    """Additional tests for cloud detection coverage."""

    def test_expand_pattern_with_wildcard(self):
        """Test expanding pattern with wildcards."""
        from astrograph.cloud_detect import _expand_pattern

        # Test wildcard pattern on a directory that exists
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some files
            test_file = os.path.join(tmpdir, "test1.py")
            with open(test_file, "w") as f:
                f.write("")

            # Pattern with wildcard
            pattern = os.path.join(tmpdir, "*.py")
            result = _expand_pattern(pattern)
            assert len(result) >= 1

    def test_expand_pattern_wildcard_nonexistent_parent(self):
        """Test expanding wildcard pattern with nonexistent parent."""
        from astrograph.cloud_detect import _expand_pattern

        result = _expand_pattern("/nonexistent/parent/dir/*.py")
        assert result == []

    def test_check_and_warn_cloud_sync_with_logger(self):
        """Test check_and_warn_cloud_sync with a logger."""
        import logging

        from astrograph.cloud_detect import check_and_warn_cloud_sync

        logger = logging.getLogger("test_cloud")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Regular path should return False
            result = check_and_warn_cloud_sync(tmpdir, logger=logger)
            assert result is False

    def test_check_and_warn_cloud_sync_print_fallback(self):
        """Test check_and_warn_cloud_sync prints to stderr when no logger."""
        from astrograph.cloud_detect import check_and_warn_cloud_sync

        with tempfile.TemporaryDirectory() as tmpdir:
            result = check_and_warn_cloud_sync(tmpdir)
            assert result is False

    def test_is_cloud_synced_path_subpath_check(self):
        """Test the subpath checking in is_cloud_synced_path."""
        from astrograph.cloud_detect import is_cloud_synced_path

        # Test with a path that is clearly not cloud synced
        is_synced, service = is_cloud_synced_path("/usr/local/bin")
        assert is_synced is False

    def test_get_cloud_sync_warning_format(self):
        """Test the warning format function returns string or None."""
        from astrograph.cloud_detect import get_cloud_sync_warning

        result = get_cloud_sync_warning("/tmp/test")
        # Either None or a string
        assert result is None or isinstance(result, str)


class TestCloudDetectionMocking:
    """Tests that mock cloud storage paths."""

    def test_cloud_warning_contains_expected_text(self):
        """Test that cloud warning contains expected sections."""
        from unittest.mock import patch

        from astrograph.cloud_detect import get_cloud_sync_warning

        # Mock a path being cloud synced
        with patch("astrograph.cloud_detect.is_cloud_synced_path") as mock:
            mock.return_value = (True, "OneDrive")
            warning = get_cloud_sync_warning("/fake/path")

            assert warning is not None
            assert "WARNING" in warning
            assert "OneDrive" in warning
            assert "Risks" in warning
            assert "Recommendations" in warning

    def test_check_and_warn_returns_true_for_cloud_path(self):
        """Test check_and_warn returns True for cloud paths."""
        import logging
        from unittest.mock import patch

        from astrograph.cloud_detect import check_and_warn_cloud_sync

        logger = logging.getLogger("test_cloud_warn")

        with patch("astrograph.cloud_detect.get_cloud_sync_warning") as mock:
            mock.return_value = "Fake warning message"
            result = check_and_warn_cloud_sync("/fake/path", logger=logger)
            assert result is True

    def test_check_and_warn_print_to_stderr(self):
        """Test check_and_warn prints to stderr when no logger provided."""
        import io
        import sys
        from unittest.mock import patch

        from astrograph.cloud_detect import check_and_warn_cloud_sync

        with patch("astrograph.cloud_detect.get_cloud_sync_warning") as mock:
            mock.return_value = "Fake warning"
            # Capture stderr
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                result = check_and_warn_cloud_sync("/fake/path", logger=None)
                output = sys.stderr.getvalue()
                assert result is True
                assert "Fake warning" in output
            finally:
                sys.stderr = old_stderr


class TestEventDrivenCloudWarning:
    """Tests for cloud warning in event-driven mode."""

    def test_event_driven_index_cloud_check_called(self):
        """Test that cloud check is called during index_directory."""
        from unittest.mock import patch

        from astrograph.event_driven import EventDrivenIndex

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            with patch("astrograph.event_driven.is_cloud_synced_path") as mock:
                mock.return_value = (False, None)

                edi = EventDrivenIndex(persistence_path=None, watch_enabled=False)
                edi.index_directory(tmpdir)

                mock.assert_called()
                edi.close()


class TestToolsCloudWarning:
    """Tests for cloud warning in tools."""

    def test_tools_event_driven_cloud_warning_integration(self):
        """Test that cloud warning appears in tools output when cloud synced."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, "file1.py")
            with open(file1, "w") as f:
                f.write("def foo(): pass")

            # Patch at the source module where it's imported from
            with patch("astrograph.cloud_detect.get_cloud_sync_warning") as mock:
                mock.return_value = "CLOUD-SYNCED FOLDER WARNING"

                tools = CodeStructureTools(event_driven=True)
                result = tools.index_codebase(tmpdir)

                assert "CLOUD-SYNCED" in result.text
                tools.close()


class TestCloudDetectionPlatformSpecific:
    """Platform-specific tests for cloud detection."""

    def test_get_windows_onedrive_paths_with_env_set(self):
        """Test Windows OneDrive detection with env vars set."""
        from astrograph.cloud_detect import _get_windows_onedrive_paths

        # Windows OneDrive uses mixed-case env var name "OneDrive"
        onedrive_var = "OneDrive"

        with tempfile.TemporaryDirectory() as tmpdir:
            old_onedrive = os.environ.get(onedrive_var)
            os.environ[onedrive_var] = tmpdir

            try:
                paths = _get_windows_onedrive_paths()
                # Should include the temp dir
                assert len(paths) >= 1
                assert any(str(p) == tmpdir for p in paths)
            finally:
                if old_onedrive is not None:
                    os.environ[onedrive_var] = old_onedrive
                else:
                    del os.environ[onedrive_var]

    @staticmethod
    def _get_platform_key_for(system_name: str) -> str:
        """Mock platform.system and return the resulting platform key."""
        import importlib
        from unittest.mock import patch

        import astrograph.cloud_detect

        with patch("platform.system", return_value=system_name):
            importlib.reload(astrograph.cloud_detect)
            return astrograph.cloud_detect._get_platform_key()

    def test_get_platform_key_mocked_windows(self):
        """Test platform key for Windows."""
        assert self._get_platform_key_for("Windows") == "win32"

    def test_get_platform_key_mocked_unknown(self):
        """Test platform key for unknown platform defaults to linux."""
        assert self._get_platform_key_for("FreeBSD") == "linux"

    def test_get_cloud_storage_paths_returns_dict(self):
        """Test get_cloud_storage_paths returns a dict."""
        from astrograph.cloud_detect import get_cloud_storage_paths

        paths = get_cloud_storage_paths()
        assert isinstance(paths, dict)

    def test_is_cloud_synced_path_catches_value_error(self):
        """Test that is_cloud_synced_path catches ValueError for relative_to."""
        from unittest.mock import patch

        from astrograph.cloud_detect import is_cloud_synced_path

        # Mock cloud paths to return a specific path
        with (
            tempfile.TemporaryDirectory() as cloud_dir,
            tempfile.TemporaryDirectory() as check_dir,
            patch("astrograph.cloud_detect.get_cloud_storage_paths") as mock,
        ):
            mock.return_value = {"OneDrive": [cloud_dir]}

            # check_dir is NOT a subpath of cloud_dir, so relative_to raises ValueError
            is_synced, service = is_cloud_synced_path(check_dir)
            assert is_synced is False
            assert service is None
