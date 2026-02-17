"""Tests for language plugin discovery."""

from unittest.mock import MagicMock, patch

import pytest

from astrograph.languages.base import BaseLanguagePlugin
from astrograph.languages.plugin_loader import (
    _close_plugin_quietly,
    _entry_points_for_group,
    _iter_entry_point_plugins,
    _load_plugin_from_class_path,
    discover_language_plugins,
)


class EnvPlugin(BaseLanguagePlugin):
    """Test plugin loaded through ASTROGRAPH_LANGUAGE_PLUGINS."""

    language_id = "envtest"
    file_extensions = frozenset({".envtest"})
    skip_dirs = frozenset()

    def extract_code_units(
        self,
        _source: str,
        _file_path: str = "<unknown>",
        _include_blocks: bool = True,
        _max_block_depth: int = 3,
    ):
        return iter(())

    def source_to_graph(self, _source: str, _normalize_ops: bool = False):
        import networkx as nx

        return nx.DiGraph()


@pytest.mark.parametrize("language_id", ["python", "javascript_lsp", "typescript_lsp"])
def test_discover_includes_default_languages(language_id):
    """Default discovery includes bundled Python, JavaScript, and TypeScript plugins."""
    plugins = discover_language_plugins()
    assert any(plugin.language_id == language_id for plugin in plugins)


def test_discover_includes_attach_plugins():
    """Default discovery includes attach-model plugins for C/C++/Java."""
    plugins = discover_language_plugins()
    language_ids = {plugin.language_id for plugin in plugins}
    assert {"c_lsp", "cpp_lsp", "java_lsp"}.issubset(language_ids)


def test_discover_deduplicates_python_language():
    """Python should appear once even if discovered from multiple sources."""
    plugins = discover_language_plugins()
    language_ids = [plugin.language_id for plugin in plugins]
    assert language_ids.count("python") == 1


@pytest.mark.parametrize(
    ("env_value", "expected_language"),
    [
        ("tests.languages.test_plugin_loader:EnvPlugin", "envtest"),
        ("not_a_valid_path", "python"),
    ],
)
def test_discover_handles_env_plugin_paths(monkeypatch, env_value, expected_language):
    """Env plugin paths can add plugins or fail safely without breaking defaults."""
    monkeypatch.setenv("ASTROGRAPH_LANGUAGE_PLUGINS", env_value)
    plugins = discover_language_plugins()
    assert any(plugin.language_id == expected_language for plugin in plugins)


# ----- Coverage tests for plugin_loader edge cases (lines 41-81, 127-130) -----


class TestLoadPluginFromClassPath:
    def test_missing_colon_separator(self):
        result = _load_plugin_from_class_path("no_colon_path")
        assert result is None

    def test_import_error(self):
        result = _load_plugin_from_class_path("nonexistent_module_xyz:Foo")
        assert result is None

    def test_not_language_plugin(self):
        """Class exists but doesn't satisfy LanguagePlugin protocol."""
        result = _load_plugin_from_class_path("builtins:int")
        assert result is None


class TestEntryPointsForGroup:
    def test_exception_returns_empty(self):
        with patch("astrograph.languages.plugin_loader.entry_points", side_effect=RuntimeError):
            result = _entry_points_for_group("test_group")
            assert len(list(result)) == 0

    def test_legacy_dict_style(self):
        """Python < 3.12 returns dict-like entry_points."""
        mock_eps = {"test_group": ["ep1"]}
        with patch("astrograph.languages.plugin_loader.entry_points", return_value=mock_eps):
            result = _entry_points_for_group("test_group")
            assert list(result) == ["ep1"]


class TestIterEntryPointPlugins:
    @staticmethod
    def _iter_from_single_entry_point(mock_ep: MagicMock):
        with patch(
            "astrograph.languages.plugin_loader._entry_points_for_group",
            return_value=[mock_ep],
        ):
            return _iter_entry_point_plugins("test_group")

    def test_entry_point_load_exception(self):
        mock_ep = MagicMock()
        mock_ep.name = "broken"
        mock_ep.load.side_effect = RuntimeError("boom")
        result = self._iter_from_single_entry_point(mock_ep)
        assert result == []

    def test_entry_point_not_language_plugin(self):
        mock_ep = MagicMock()
        mock_ep.name = "not_plugin"
        mock_ep.load.return_value = lambda: 42  # returns int, not LanguagePlugin
        result = self._iter_from_single_entry_point(mock_ep)
        assert result == []


class TestClosePluginQuietly:
    def test_close_exception_swallowed(self):
        mock_plugin = MagicMock()
        mock_plugin.close.side_effect = RuntimeError("boom")
        mock_plugin.language_id = "test"
        _close_plugin_quietly(mock_plugin)  # should not raise

    def test_no_close_method(self):
        mock_plugin = MagicMock(spec=[])  # no close method
        mock_plugin.language_id = "test"
        _close_plugin_quietly(mock_plugin)  # should not raise
