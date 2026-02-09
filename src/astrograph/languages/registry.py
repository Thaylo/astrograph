"""
Language plugin registry.

Singleton that routes files to their language plugin based on file extension.
Plugins are discovered via the shared plugin loader.
"""

import logging
import threading
from pathlib import Path

from .base import LanguagePlugin
from .plugin_loader import discover_language_plugins

# Common directories to skip regardless of language
_COMMON_SKIP_DIRS = frozenset(
    [
        ".git",
        "node_modules",
        ".metadata_astrograph",
        "dist",
        "build",
        ".eggs",
        ".pytest_cache",
        ".ruff_cache",
    ]
)


class LanguageRegistry:
    """
    Singleton registry mapping file extensions to language plugins.

    Usage:
        registry = LanguageRegistry.get()
        plugin = registry.get_plugin_for_file("src/main.py")
    """

    _instance: "LanguageRegistry | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._plugins: dict[str, LanguagePlugin] = {}  # language_id -> plugin
        self._extension_map: dict[str, str] = {}  # extension -> language_id
        self._initialized = False
        self._logger = logging.getLogger(__name__)

    @classmethod
    def get(cls) -> "LanguageRegistry":
        """Get the singleton registry instance."""
        inst = cls._instance
        if inst is None:
            with cls._lock:
                inst = cls._instance
                if inst is None:
                    inst = cls()
                    cls._instance = inst
        inst._ensure_plugins()
        return inst

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _ensure_plugins(self) -> None:
        """Discover and register plugins on first access."""
        if self._initialized:
            return

        discovered = discover_language_plugins()
        for plugin in discovered:
            try:
                self.register(plugin)
            except ValueError as exc:
                self._logger.warning("Skipping language plugin '%s': %s", plugin.language_id, exc)

        if self._plugins:
            self._initialized = True
            return

        self._logger.warning("No language plugins were discovered; will retry on next access")

    def register(self, plugin: LanguagePlugin) -> None:
        """
        Register a language plugin.

        Raises ValueError if any file extension conflicts with an existing plugin.
        """
        # Check for extension conflicts
        for ext in plugin.file_extensions:
            if ext in self._extension_map:
                existing_id = self._extension_map[ext]
                raise ValueError(
                    f"Extension '{ext}' already registered by '{existing_id}', "
                    f"cannot register '{plugin.language_id}'"
                )

        self._plugins[plugin.language_id] = plugin
        for ext in plugin.file_extensions:
            self._extension_map[ext] = plugin.language_id

    def get_plugin(self, language_id: str) -> LanguagePlugin | None:
        """Get a plugin by language ID."""
        return self._plugins.get(language_id)

    def get_plugin_for_file(self, path: str | Path) -> LanguagePlugin | None:
        """Get the appropriate plugin for a file based on its extension."""
        ext = Path(path).suffix
        language_id = self._extension_map.get(ext)
        return self._plugins.get(language_id) if language_id else None

    @property
    def supported_extensions(self) -> frozenset[str]:
        """Union of all registered file extensions."""
        extensions = set(self._extension_map)
        return frozenset(extensions)

    @property
    def skip_dirs(self) -> frozenset[str]:
        """Union of all language-specific skip dirs plus common ones."""
        all_skip: set[str] = set(_COMMON_SKIP_DIRS)
        for plugin in self._plugins.values():
            all_skip.update(plugin.skip_dirs)
        return frozenset(all_skip)

    @property
    def registered_languages(self) -> list[str]:
        """List of registered language IDs in stable sorted order."""
        languages = list(self._plugins)
        languages.sort()
        return languages
