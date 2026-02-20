"""Language plugin discovery and loading utilities."""

from __future__ import annotations

import logging
import os
from importlib import import_module
from importlib.metadata import EntryPoints, entry_points

from .base import LanguagePlugin

PLUGIN_ENTRYPOINT_GROUP = "astrograph.languages"
PLUGIN_PATHS_ENV_VAR = "ASTROGRAPH_LANGUAGE_PLUGINS"
DEFAULT_PLUGIN_CLASS_PATHS = (
    "astrograph.languages.python_lsp_plugin:PythonLSPPlugin",
    "astrograph.languages.javascript_lsp_plugin:JavaScriptLSPPlugin",
    "astrograph.languages.typescript_lsp_plugin:TypeScriptLSPPlugin",
    "astrograph.languages.c_lsp_plugin:CLSPPlugin",
    "astrograph.languages.cpp_lsp_plugin:CppLSPPlugin",
    "astrograph.languages.java_lsp_plugin:JavaLSPPlugin",
    "astrograph.languages.go_lsp_plugin:GoLSPPlugin",
)

logger = logging.getLogger(__name__)


def _load_plugin_from_class_path(class_path: str) -> LanguagePlugin | None:
    """Load and instantiate a plugin from 'module.path:ClassName'."""
    module_name, sep, class_name = class_path.partition(":")
    if not sep:
        logger.warning(
            "Invalid plugin path '%s' (expected 'module.path:ClassName'); skipping",
            class_path,
        )
        return None

    try:
        module = import_module(module_name)
        plugin_factory = getattr(module, class_name)
        plugin = plugin_factory()
    except Exception as exc:
        logger.warning("Failed to load plugin '%s': %s", class_path, exc)
        return None

    if not isinstance(plugin, LanguagePlugin):
        logger.warning(
            "Plugin '%s' does not satisfy LanguagePlugin protocol; skipping",
            class_path,
        )
        return None
    return plugin


def _entry_points_for_group(group: str) -> EntryPoints:
    """Return entry points for a group across Python versions."""
    try:
        eps = entry_points()
        if hasattr(eps, "select"):
            return eps.select(group=group)
        return EntryPoints(eps.get(group, []))
    except Exception as exc:
        logger.warning("Failed reading entry points for '%s': %s", group, exc)
        return EntryPoints(())


def _iter_entry_point_plugins(group: str) -> list[LanguagePlugin]:
    """Load plugins exposed through Python package entry points."""
    plugins: list[LanguagePlugin] = []
    for ep in _entry_points_for_group(group):
        try:
            loaded = ep.load()
            plugin = loaded() if callable(loaded) else loaded
        except Exception as exc:
            logger.warning("Failed loading plugin entry point '%s': %s", ep.name, exc)
            continue
        if not isinstance(plugin, LanguagePlugin):
            logger.warning(
                "Entry point '%s' does not satisfy LanguagePlugin protocol; skipping",
                ep.name,
            )
            continue
        plugins.append(plugin)
    return plugins


def _env_plugin_paths() -> list[str]:
    """Return extra plugin class paths from ASTROGRAPH_LANGUAGE_PLUGINS."""
    value = os.getenv(PLUGIN_PATHS_ENV_VAR, "")
    if not value.strip():
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def discover_language_plugins() -> list[LanguagePlugin]:
    """Discover language plugins from defaults, env class paths, and entry points."""
    discovered: list[LanguagePlugin] = []
    for class_path in [*DEFAULT_PLUGIN_CLASS_PATHS, *_env_plugin_paths()]:
        plugin = _load_plugin_from_class_path(class_path)
        if plugin is not None:
            discovered.append(plugin)

    discovered.extend(_iter_entry_point_plugins(PLUGIN_ENTRYPOINT_GROUP))

    # Deduplicate by language_id while preserving first-registered precedence.
    # Close discarded plugins to avoid leaking LSP subprocess handles.
    deduped: list[LanguagePlugin] = []
    seen_language_ids: set[str] = set()
    for plugin in discovered:
        language_id = plugin.language_id
        if language_id in seen_language_ids:
            logger.debug(
                "Duplicate language plugin '%s' discovered; keeping first instance",
                language_id,
            )
            _close_plugin_quietly(plugin)
            continue
        deduped.append(plugin)
        seen_language_ids.add(language_id)

    return deduped


def _close_plugin_quietly(plugin: LanguagePlugin) -> None:
    """Close a plugin's resources if it has a close method."""
    close = getattr(plugin, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.debug("Error closing discarded plugin '%s'", plugin.language_id)
