"""Tests for language plugin discovery."""

from astrograph.languages.base import BaseLanguagePlugin
from astrograph.languages.plugin_loader import discover_language_plugins


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


def test_discover_includes_python_plugin():
    """Default discovery always includes Python support."""
    plugins = discover_language_plugins()
    assert any(plugin.language_id == "python" for plugin in plugins)


def test_discover_includes_javascript_lsp_plugin():
    """Entry-point discovery includes the JavaScript LSP plugin."""
    plugins = discover_language_plugins()
    assert any(plugin.language_id == "javascript_lsp" for plugin in plugins)


def test_discover_deduplicates_python_language():
    """Python should appear once even if discovered from multiple sources."""
    plugins = discover_language_plugins()
    language_ids = [plugin.language_id for plugin in plugins]
    assert language_ids.count("python") == 1


def test_discover_loads_plugins_from_env(monkeypatch):
    """Class paths in ASTROGRAPH_LANGUAGE_PLUGINS are loaded."""
    monkeypatch.setenv(
        "ASTROGRAPH_LANGUAGE_PLUGINS",
        "tests.languages.test_plugin_loader:EnvPlugin",
    )
    plugins = discover_language_plugins()
    assert any(plugin.language_id == "envtest" for plugin in plugins)


def test_discover_ignores_invalid_env_plugin_path(monkeypatch):
    """Invalid env paths are skipped without breaking built-in discovery."""
    monkeypatch.setenv("ASTROGRAPH_LANGUAGE_PLUGINS", "not_a_valid_path")
    plugins = discover_language_plugins()
    assert any(plugin.language_id == "python" for plugin in plugins)
