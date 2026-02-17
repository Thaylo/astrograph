"""Tests for the language registry."""

import pytest

from astrograph.languages.base import BaseLanguagePlugin, LanguagePlugin
from astrograph.languages.python_lsp_plugin import PythonLSPPlugin
from astrograph.languages.registry import LanguageRegistry


class TestLanguageRegistry:
    """Tests for LanguageRegistry singleton."""

    def test_singleton(self):
        """Registry returns same instance."""
        r1 = LanguageRegistry.get()
        r2 = LanguageRegistry.get()
        assert r1 is r2

    def test_reset(self):
        """Reset creates new instance."""
        r1 = LanguageRegistry.get()
        LanguageRegistry.reset()
        r2 = LanguageRegistry.get()
        assert r1 is not r2

    def test_builtin_python_registered(self):
        """Python plugin is registered by default."""
        registry = LanguageRegistry.get()
        assert "python" in registry.registered_languages

    @pytest.mark.parametrize(
        "resolver",
        [
            lambda registry: registry.get_plugin_for_file("src/main.py"),
            lambda registry: registry.get_plugin_for_file("src/types.pyi"),
            lambda registry: registry.get_plugin_for_file("src/MAIN.PY"),
            lambda registry: registry.get_plugin("python"),
        ],
    )
    def test_get_python_plugin(self, resolver):
        """Python plugin is retrievable by file extension and language ID."""
        registry = LanguageRegistry.get()
        plugin = resolver(registry)
        assert plugin is not None
        assert plugin.language_id == "python"

    def test_get_plugin_for_unsupported_file(self):
        """Returns None for unsupported file types."""
        registry = LanguageRegistry.get()
        assert registry.get_plugin_for_file("style.css") is None
        assert registry.get_plugin_for_file("data.json") is None

    @pytest.mark.parametrize(
        ("collection", "expected"),
        [
            ("supported_extensions", (".py", ".pyi")),
            ("skip_dirs", ("__pycache__", ".mypy_cache")),
        ],
    )
    def test_python_registry_collections(self, collection, expected):
        """Python-related registry collections include expected values."""
        registry = LanguageRegistry.get()
        values = getattr(registry, collection)
        assert set(expected).issubset(set(values))

    def test_skip_dirs_includes_common(self):
        """Skip dirs include common directories."""
        registry = LanguageRegistry.get()
        skip = registry.skip_dirs
        assert ".git" in skip
        assert "node_modules" in skip
        assert ".metadata_astrograph" in skip

    def test_extension_conflict(self):
        """Registering conflicting extension raises ValueError."""
        registry = LanguageRegistry.get()

        class ConflictPlugin(BaseLanguagePlugin):
            language_id = "conflict"
            file_extensions = frozenset({".py"})  # Conflicts with Python
            skip_dirs = frozenset()

        with pytest.raises(ValueError, match="already registered"):
            registry.register(ConflictPlugin())

    def test_extension_conflict_is_case_insensitive(self):
        """Registering .PY should conflict with existing .py registration."""
        registry = LanguageRegistry.get()

        class ConflictPluginUpper(BaseLanguagePlugin):
            language_id = "conflict_upper"
            file_extensions = frozenset({".PY"})
            skip_dirs = frozenset()

        with pytest.raises(ValueError, match="already registered"):
            registry.register(ConflictPluginUpper())

    def test_register_new_language(self):
        """Can register a new language plugin."""
        registry = LanguageRegistry.get()

        class MockPlugin(BaseLanguagePlugin):
            language_id = "mock"
            file_extensions = frozenset({".mock"})
            skip_dirs = frozenset({"mock_cache"})

        registry.register(MockPlugin())
        assert "mock" in registry.registered_languages
        assert ".mock" in registry.supported_extensions
        assert "mock_cache" in registry.skip_dirs
        assert registry.get_plugin_for_file("test.mock") is not None

    def test_register_uppercase_extension_lookup_is_case_insensitive(self):
        """A plugin registered with uppercase extensions should match lowercase files."""
        registry = LanguageRegistry.get()

        class CapsMockPlugin(BaseLanguagePlugin):
            language_id = "capsmock"
            file_extensions = frozenset({".CAPSMOCK"})
            skip_dirs = frozenset()

        registry.register(CapsMockPlugin())
        assert registry.get_plugin_for_file("example.capsmock") is not None
        assert registry.get_plugin_for_file("example.CAPSMOCK") is not None

    def test_registered_languages_sorted(self):
        """Registered language list is stable and sorted."""
        registry = LanguageRegistry.get()

        languages = registry.registered_languages
        assert languages == sorted(languages)

    def test_get_nonexistent_plugin(self):
        """Returns None for nonexistent language."""
        registry = LanguageRegistry.get()
        assert registry.get_plugin("cobol") is None


class TestLanguagePluginProtocol:
    """Tests for the LanguagePlugin protocol."""

    def test_python_plugin_satisfies_protocol(self):
        """PythonLSPPlugin satisfies the LanguagePlugin protocol."""
        plugin = PythonLSPPlugin()
        assert isinstance(plugin, LanguagePlugin)

    def test_base_plugin_not_protocol(self):
        """Raw BaseLanguagePlugin doesn't satisfy protocol (abstract methods)."""
        plugin = BaseLanguagePlugin()
        pytest.raises(NotImplementedError, getattr, plugin, "language_id")
