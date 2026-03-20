"""Tests for _paths module — lightweight, no resource-heavy objects."""

import tempfile
from pathlib import Path
from unittest.mock import patch

from astrograph._paths import (
    LEGACY_DIR_NAME,
    _get_data_dir,
    _maybe_migrate,
    _project_key,
    get_persistence_path,
    get_sqlite_path,
)


class TestGetDataDir:
    """Tests for _get_data_dir resolution order."""

    def test_custom_env_takes_priority(self, monkeypatch):
        monkeypatch.setenv("ASTROGRAPH_DATA_DIR", "/custom/data")
        assert _get_data_dir() == Path("/custom/data")

    def test_xdg_data_home(self, monkeypatch):
        monkeypatch.delenv("ASTROGRAPH_DATA_DIR", raising=False)
        monkeypatch.setenv("XDG_DATA_HOME", "/xdg/share")
        assert _get_data_dir() == Path("/xdg/share/astrograph")

    def test_darwin_default(self, monkeypatch):
        monkeypatch.delenv("ASTROGRAPH_DATA_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        with patch("astrograph._paths.sys") as mock_sys:
            mock_sys.platform = "darwin"
            result = _get_data_dir()
            assert "astrograph" in str(result)

    def test_linux_default(self, monkeypatch):
        monkeypatch.delenv("ASTROGRAPH_DATA_DIR", raising=False)
        monkeypatch.delenv("XDG_DATA_HOME", raising=False)
        with patch("astrograph._paths.sys") as mock_sys:
            mock_sys.platform = "linux"
            result = _get_data_dir()
            assert ".local/share/astrograph" in str(result)


class TestProjectKey:
    """Tests for _project_key generation."""

    def test_stable_across_calls(self):
        p = Path("/some/project")
        assert _project_key(p) == _project_key(p)

    def test_different_paths_differ(self):
        assert _project_key(Path("/a")) != _project_key(Path("/b"))

    def test_format(self):
        key = _project_key(Path("/home/user/myproject"))
        assert key.startswith("myproject-")
        assert len(key.split("-")[-1]) == 8  # 8-char hash

    def test_root_path(self):
        key = _project_key(Path("/"))
        assert key.startswith("root-")

    def test_special_characters_sanitized(self):
        key = _project_key(Path("/home/user/my project (2)"))
        # Spaces and parens should be replaced with underscores
        assert " " not in key
        assert "(" not in key


class TestMaybeMigrate:
    """Tests for legacy metadata migration."""

    def test_migrates_legacy_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            project.mkdir()
            legacy = project / LEGACY_DIR_NAME
            legacy.mkdir()
            (legacy / "index.db").write_text("data")

            new_path = Path(tmpdir) / "new_metadata"
            _maybe_migrate(project, new_path)

            assert not legacy.exists()
            assert (new_path / "index.db").read_text() == "data"

    def test_skips_if_no_legacy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            project.mkdir()
            new_path = Path(tmpdir) / "new_metadata"

            _maybe_migrate(project, new_path)

            assert not new_path.exists()

    def test_skips_if_target_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            project.mkdir()
            legacy = project / LEGACY_DIR_NAME
            legacy.mkdir()
            (legacy / "old.db").write_text("old")

            new_path = Path(tmpdir) / "new_metadata"
            new_path.mkdir()
            (new_path / "new.db").write_text("new")

            _maybe_migrate(project, new_path)

            # Legacy should NOT be moved (target already exists)
            assert legacy.exists()
            assert (new_path / "new.db").read_text() == "new"

    def test_handles_os_error_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            project.mkdir()
            legacy = project / LEGACY_DIR_NAME
            legacy.mkdir()

            # Target parent is a file, not a dir — mkdir will fail
            blocker = Path(tmpdir) / "blocker"
            blocker.write_text("not a dir")
            new_path = blocker / "sub" / "metadata"

            # Should not raise
            _maybe_migrate(project, new_path)


class TestGetPersistencePath:
    """Tests for get_persistence_path."""

    def test_returns_path_outside_project(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = get_persistence_path(tmpdir)
            assert tmpdir not in str(result)
            assert "astrograph" in str(result)

    def test_file_path_uses_parent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "file.py"
            f.write_text("")
            result_file = get_persistence_path(str(f))
            result_dir = get_persistence_path(tmpdir)
            assert result_file == result_dir

    def test_docker_path(self):
        with patch("astrograph._paths._is_docker", return_value=True):
            original_exists = Path.exists

            def mock_exists(self):
                if str(self) == "/workspace":
                    return True
                return original_exists(self)

            with patch.object(Path, "exists", mock_exists):
                result = get_persistence_path("/workspace/project")
                assert str(result) == "/workspace/.metadata_astrograph"

    def test_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            assert get_persistence_path(tmpdir) == get_persistence_path(tmpdir)


class TestGetSqlitePath:
    """Tests for get_sqlite_path."""

    def test_ends_with_index_db(self):
        result = get_sqlite_path("/some/project")
        assert str(result).endswith("index.db")

    def test_under_persistence_path(self):
        pp = get_persistence_path("/some/project")
        sp = get_sqlite_path("/some/project")
        assert sp.parent == pp
