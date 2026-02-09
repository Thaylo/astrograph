"""Tests for language plugin discovery."""

import pytest

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
