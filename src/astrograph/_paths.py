"""Centralized path resolution for astrograph metadata storage.

Metadata (SQLite index, analysis reports, LSP bindings) is stored outside
the project directory by default, preventing the file watcher from ever
observing tool-owned artifacts.

Location resolution order:
1. Docker — /workspace/.metadata_astrograph (tmpfs mount)
2. ASTROGRAPH_DATA_DIR env — explicit user override
3. XDG_DATA_HOME/astrograph — XDG-compliant
4. Platform default — ~/Library/Application Support/astrograph (macOS)
                      ~/.local/share/astrograph (Linux)
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Legacy in-project directory name (still used for Docker, skip-dir filtering,
# and detecting old metadata that needs migration).
LEGACY_DIR_NAME = ".metadata_astrograph"


def _get_data_dir() -> Path:
    """Get platform-appropriate data directory for astrograph metadata."""
    custom = os.environ.get("ASTROGRAPH_DATA_DIR")
    if custom:
        return Path(custom)

    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / "astrograph"

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "astrograph"

    return Path.home() / ".local" / "share" / "astrograph"


def _project_key(project_path: Path) -> str:
    """Generate a stable, human-readable key for a project path.

    Format: ``<dirname>-<8-char hash>``  (e.g. ``myproject-a1b2c3d4``).
    """
    path_hash = hashlib.sha256(str(project_path).encode()).hexdigest()[:8]
    dirname = project_path.name or "root"
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in dirname)
    return f"{safe_name}-{path_hash}"


def _is_docker() -> bool:
    return Path("/.dockerenv").exists()


def get_persistence_path(indexed_path: str | Path) -> Path:
    """Return the metadata directory for an indexed codebase.

    * Docker: ``/workspace/.metadata_astrograph`` (tmpfs mount).
    * Otherwise: ``<data_dir>/projects/<project-key>/``.

    Automatically migrates from the legacy in-project location on first access.
    """
    base = Path(indexed_path).resolve()
    if base.is_file():
        base = base.parent

    # Docker: use tmpfs mount at workspace root
    workspace = Path("/workspace")
    if _is_docker() and workspace.exists() and str(base).startswith("/workspace"):
        return workspace / LEGACY_DIR_NAME

    # Default: user data directory
    new_path = _get_data_dir() / "projects" / _project_key(base)

    # Auto-migrate from legacy in-project location
    _maybe_migrate(base, new_path)

    return new_path


def _maybe_migrate(project_path: Path, new_path: Path) -> None:
    """Move metadata from old in-project .metadata_astrograph/ to new location."""
    old_path = project_path / LEGACY_DIR_NAME
    if old_path.is_dir() and not new_path.exists():
        try:
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_path), str(new_path))
            logger.info("Migrated metadata: %s -> %s", old_path, new_path)
        except OSError:
            logger.warning("Could not migrate metadata from %s", old_path, exc_info=True)


def get_sqlite_path(indexed_path: str | Path) -> Path:
    """Return the SQLite database path for an indexed codebase."""
    return get_persistence_path(indexed_path) / "index.db"
