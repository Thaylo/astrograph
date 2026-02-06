"""Tests for the language registry."""

import pytest

from astrograph.languages.base import BaseLanguagePlugin, LanguagePlugin
from astrograph.languages.python_plugin import PythonPlugin
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

    def test_get_plugin_for_python_file(self):
        """Routes .py files to Python plugin."""
        registry = LanguageRegistry.get()
        plugin = registry.get_plugin_for_file("src/main.py")
        assert plugin is not None
        assert plugin.language_id == "python"

    def test_get_plugin_for_pyi_file(self):
        """Routes .pyi files to Python plugin."""
        registry = LanguageRegistry.get()
        plugin = registry.get_plugin_for_file("src/types.pyi")
        assert plugin is not None
        assert plugin.language_id == "python"

    def test_get_plugin_for_unsupported_file(self):
        """Returns None for unsupported file types."""
        registry = LanguageRegistry.get()
        assert registry.get_plugin_for_file("style.css") is None
        assert registry.get_plugin_for_file("data.json") is None

    def test_supported_extensions(self):
        """Python extensions are in supported set."""
        registry = LanguageRegistry.get()
        exts = registry.supported_extensions
        assert ".py" in exts
        assert ".pyi" in exts

    def test_skip_dirs_includes_common(self):
        """Skip dirs include common directories."""
        registry = LanguageRegistry.get()
        skip = registry.skip_dirs
        assert ".git" in skip
        assert "node_modules" in skip
        assert ".metadata_astrograph" in skip

    def test_skip_dirs_includes_python(self):
        """Skip dirs include Python-specific directories."""
        registry = LanguageRegistry.get()
        skip = registry.skip_dirs
        assert "__pycache__" in skip
        assert ".mypy_cache" in skip

    def test_extension_conflict(self):
        """Registering conflicting extension raises ValueError."""
        registry = LanguageRegistry.get()

        class ConflictPlugin(BaseLanguagePlugin):
            @property
            def language_id(self):
                return "conflict"

            @property
            def file_extensions(self):
                return frozenset({".py"})  # Conflicts with Python

            @property
            def skip_dirs(self):
                return frozenset()

        with pytest.raises(ValueError, match="already registered"):
            registry.register(ConflictPlugin())

    def test_register_new_language(self):
        """Can register a new language plugin."""
        registry = LanguageRegistry.get()

        class MockPlugin(BaseLanguagePlugin):
            @property
            def language_id(self):
                return "mock"

            @property
            def file_extensions(self):
                return frozenset({".mock"})

            @property
            def skip_dirs(self):
                return frozenset({"mock_cache"})

        registry.register(MockPlugin())
        assert "mock" in registry.registered_languages
        assert ".mock" in registry.supported_extensions
        assert "mock_cache" in registry.skip_dirs
        assert registry.get_plugin_for_file("test.mock") is not None

    def test_get_plugin_by_id(self):
        """Can retrieve plugin by language ID."""
        registry = LanguageRegistry.get()
        plugin = registry.get_plugin("python")
        assert plugin is not None
        assert plugin.language_id == "python"

    def test_get_nonexistent_plugin(self):
        """Returns None for nonexistent language."""
        registry = LanguageRegistry.get()
        assert registry.get_plugin("cobol") is None


class TestLanguagePluginProtocol:
    """Tests for the LanguagePlugin protocol."""

    def test_python_plugin_satisfies_protocol(self):
        """PythonPlugin satisfies the LanguagePlugin protocol."""
        plugin = PythonPlugin()
        assert isinstance(plugin, LanguagePlugin)

    def test_base_plugin_not_protocol(self):
        """Raw BaseLanguagePlugin doesn't satisfy protocol (abstract methods)."""
        # BaseLanguagePlugin itself raises NotImplementedError on abstract methods
        plugin = BaseLanguagePlugin()
        with pytest.raises(NotImplementedError):
            _ = plugin.language_id
