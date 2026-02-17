"""Tests for cloud_detect module coverage gaps."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from astrograph.cloud_detect import (
    _expand_pattern,
    get_cloud_storage_paths,
    is_cloud_synced_path,
)


class TestExpandPattern:
    def test_wildcard_pattern_no_match(self, tmp_path):
        result = _expand_pattern(str(tmp_path / "nonexistent-*"))
        assert result == []

    def test_wildcard_pattern_parent_not_exists(self):
        result = _expand_pattern("/nonexistent_dir_abc123/some-*")
        assert result == []

    def test_wildcard_pattern_with_match(self, tmp_path):
        (tmp_path / "onedrive-personal").mkdir()
        (tmp_path / "onedrive-work").mkdir()
        result = _expand_pattern(str(tmp_path / "onedrive-*"))
        assert len(result) == 2

    def test_simple_path_exists(self, tmp_path):
        result = _expand_pattern(str(tmp_path))
        assert result == [tmp_path]

    def test_simple_path_not_exists(self):
        result = _expand_pattern("/nonexistent_path_xyz_123")
        assert result == []


class TestGetCloudStoragePaths:
    def test_returns_dict(self):
        result = get_cloud_storage_paths()
        assert isinstance(result, dict)

    def test_with_fake_dropbox(self, tmp_path):
        """Simulate Dropbox folder existing."""
        dropbox_dir = tmp_path / "Dropbox"
        dropbox_dir.mkdir()
        with (
            patch(
                "astrograph.cloud_detect.CLOUD_PATTERNS",
                {"linux": {"Dropbox": [str(dropbox_dir)]}},
            ),
            patch("astrograph.cloud_detect._get_platform_key", return_value="linux"),
        ):
            result = get_cloud_storage_paths()
        assert "Dropbox" in result
        assert dropbox_dir.resolve() in result["Dropbox"]

    def test_oserror_in_resolve_skipped(self, tmp_path):
        """OSError during path.resolve() is silently skipped."""
        with (
            patch(
                "astrograph.cloud_detect.CLOUD_PATTERNS",
                {"linux": {"Test": [str(tmp_path)]}},
            ),
            patch("astrograph.cloud_detect._get_platform_key", return_value="linux"),
            patch.object(Path, "resolve", side_effect=OSError("boom")),
        ):
            result = get_cloud_storage_paths()
        assert "Test" not in result


class TestIsCloudSyncedPath:
    def test_not_synced(self, tmp_path):
        synced, service = is_cloud_synced_path(tmp_path)
        assert synced is False
        assert service is None

    def test_synced_path(self, tmp_path):
        dropbox = tmp_path / "Dropbox"
        dropbox.mkdir()
        inner = dropbox / "project"
        inner.mkdir()
        with patch(
            "astrograph.cloud_detect.get_cloud_storage_paths",
            return_value={"Dropbox": [dropbox.resolve()]},
        ):
            synced, service = is_cloud_synced_path(inner)
        assert synced is True
        assert service == "Dropbox"
