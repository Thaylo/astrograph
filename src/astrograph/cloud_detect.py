"""
Cloud-synced folder detection for data corruption prevention.

Detects OneDrive, Dropbox, iCloud, and Google Drive folders across
macOS, Windows, and Linux to warn users about potential conflicts
when using file watching in synced directories.
"""

import os
import platform
from pathlib import Path
from typing import Any

# Known cloud storage path patterns by platform
CLOUD_PATTERNS = {
    "darwin": {  # macOS
        "OneDrive": [
            "~/Library/CloudStorage/OneDrive-*",
            "~/OneDrive",
            "~/OneDrive - *",
        ],
        "Dropbox": [
            "~/Library/CloudStorage/Dropbox",
            "~/Dropbox",
        ],
        "iCloud": [
            "~/Library/Mobile Documents/com~apple~CloudDocs",
            "~/Library/CloudStorage/iCloud*",
        ],
        "Google Drive": [
            "~/Library/CloudStorage/GoogleDrive-*",
            "~/Google Drive",
        ],
    },
    "linux": {
        "OneDrive": [
            # Third-party clients (rclone, onedrive-linux)
            "~/OneDrive",
            "~/.local/share/onedrive",
        ],
        "Dropbox": [
            "~/Dropbox",
            "~/.dropbox",
        ],
        "Google Drive": [
            # Third-party clients
            "~/google-drive",
            "~/.local/share/google-drive",
        ],
    },
    "win32": {  # Windows
        "OneDrive": [
            "~/OneDrive",
            "~/OneDrive - *",
            # Also check environment variable
        ],
        "Dropbox": [
            "~/Dropbox",
        ],
        "Google Drive": [
            "~/Google Drive",
        ],
    },
}


def _expand_pattern(pattern: str) -> list[Path]:
    """Expand a path pattern with ~ and * wildcards."""
    expanded = os.path.expanduser(pattern)

    if "*" in expanded:
        # Use glob for wildcard patterns
        parent = Path(expanded).parent
        name_pattern = Path(expanded).name
        try:
            if parent.exists():
                return list(parent.glob(name_pattern))
        except OSError:
            pass
        return []
    else:
        path = Path(expanded)
        return [path] if path.exists() else []


def _get_windows_onedrive_paths() -> list[Path]:
    """Get OneDrive paths from Windows environment variables."""
    paths = []

    # Check common OneDrive environment variables
    for var in ["OneDrive", "OneDriveConsumer", "OneDriveCommercial"]:
        value = os.environ.get(var)
        if value:
            path = Path(value)
            if path.exists():
                paths.append(path)

    return paths


def _get_platform_key() -> str:
    """Get the platform key for CLOUD_PATTERNS."""
    system = platform.system().lower()
    platform_map = {"darwin": "darwin", "linux": "linux", "windows": "win32"}
    return platform_map.get(system, "linux")  # Default to Linux patterns


def get_cloud_storage_paths() -> dict[str, list[Path]]:
    """
    Detect all cloud storage paths on the current system.

    Returns:
        Dictionary mapping service name to list of detected paths.
    """
    platform_key = _get_platform_key()
    patterns = CLOUD_PATTERNS.get(platform_key, {})

    detected: dict[str, list[Path]] = {}

    for service, service_patterns in patterns.items():
        paths = []
        for pattern in service_patterns:
            paths.extend(_expand_pattern(pattern))

        # Windows-specific: check environment variables for OneDrive
        if platform_key == "win32" and service == "OneDrive":
            paths.extend(_get_windows_onedrive_paths())

        # Deduplicate and resolve
        unique_paths = []
        seen = set()
        for p in paths:
            try:
                resolved = p.resolve()
            except OSError:
                continue
            if resolved not in seen:
                seen.add(resolved)
                unique_paths.append(resolved)

        if unique_paths:
            detected[service] = unique_paths

    return detected


def is_cloud_synced_path(path: str | Path) -> tuple[bool, str | None]:
    """
    Check if a path is within a cloud-synced folder.

    Args:
        path: Path to check

    Returns:
        Tuple of (is_synced, service_name).
        If not synced, returns (False, None).
    """
    check_path = Path(path).resolve()
    cloud_paths = get_cloud_storage_paths()

    for service, service_paths in cloud_paths.items():
        for cloud_path in service_paths:
            try:
                # Check if check_path is inside cloud_path
                check_path.relative_to(cloud_path)
                return (True, service)
            except ValueError:
                # Not a subpath
                continue

    return (False, None)


def get_cloud_sync_warning(path: str | Path) -> str | None:
    """
    Get a warning message if path is cloud-synced.

    Args:
        path: Path to check

    Returns:
        Warning message string, or None if path is not cloud-synced.
    """
    is_synced, service = is_cloud_synced_path(path)

    if not is_synced:
        return None

    return (
        f"WARNING: Cloud-synced folder ({service}). "
        f"Risks: index corruption, sync conflicts, excessive re-indexing. "
        f"Recommendations: (1) move to local folder, "
        f"(2) add .metadata_astrograph/ to {service} ignore list, "
        f"(3) or disable event-driven mode."
    )


def check_and_warn_cloud_sync(path: str | Path, logger: Any = None) -> bool:
    """
    Check if path is cloud-synced and log a warning if so.

    Args:
        path: Path to check
        logger: Optional logger to use. If None, prints to stderr.

    Returns:
        True if path is cloud-synced, False otherwise.
    """
    warning = get_cloud_sync_warning(path)

    if warning:
        if logger:
            logger.warning(warning)
        else:
            import sys

            print(warning, file=sys.stderr)
        return True

    return False
