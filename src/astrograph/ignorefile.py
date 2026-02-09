"""
Gitignore-like pattern matching for .astrographignore files.

Supports: comments (#), blank lines, globs (*), directory patterns (dir/),
anchored patterns (/pattern), recursive match (**), and negation (!pattern).
"""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Iterable
from pathlib import Path

ASTROGRAPHIGNORE_FILENAME = ".astrographignore"


class IgnoreSpec:
    """Compiled gitignore-like patterns for file/directory exclusion."""

    __slots__ = ("_rules",)

    def __init__(self, rules: list[tuple[bool, bool, bool, str]]) -> None:
        # Each rule: (negated, dir_only, anchored, glob_pattern)
        self._rules = rules

    @classmethod
    def from_file(cls, path: str | Path) -> IgnoreSpec | None:
        """Load from file. Returns None if file doesn't exist."""
        p = Path(path)
        if not p.is_file():
            return None
        return cls.from_lines(p.read_text().splitlines())

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> IgnoreSpec:
        """Parse gitignore-like lines into an IgnoreSpec."""
        rules: list[tuple[bool, bool, bool, str]] = []
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            negated = False
            if line.startswith("!"):
                negated = True
                line = line[1:]

            dir_only = line.endswith("/")
            if dir_only:
                line = line.rstrip("/")

            anchored = "/" in line  # contains slash → anchored
            if line.startswith("/"):
                line = line[1:]  # strip leading slash, already anchored

            rules.append((negated, dir_only, anchored, line))
        return cls(rules)

    def _match(self, rel_path: str, is_dir: bool) -> bool:
        """Test if a relative path is ignored. Last matching rule wins."""
        # Normalize to forward slashes for consistent matching
        rel_path = rel_path.replace(os.sep, "/")
        result = False
        for negated, dir_only, anchored, pattern in self._rules:
            if dir_only:
                if is_dir:
                    if self._pattern_matches(pattern, rel_path, anchored):
                        result = not negated
                else:
                    # For files, check if any parent directory matches
                    parts = rel_path.split("/")
                    for i in range(1, len(parts)):
                        parent = "/".join(parts[:i])
                        if self._pattern_matches(pattern, parent, anchored):
                            result = not negated
                            break
            elif self._pattern_matches(pattern, rel_path, anchored):
                result = not negated
        return result

    @staticmethod
    def _pattern_matches(pattern: str, rel_path: str, anchored: bool) -> bool:
        """Check if a single pattern matches a relative path."""
        if "**" in pattern:
            # Expand ** to match any number of path components
            # "**/" at start: match at any depth
            # "a/**/b": match a/b, a/x/b, a/x/y/b, etc.
            regex_pat = pattern.replace("**", "__DOUBLESTAR__")
            # Convert to fnmatch parts, then restore ** semantics
            parts = regex_pat.split("__DOUBLESTAR__")
            # Build a list of path segments to try matching against
            if len(parts) == 2 and parts[0] == "" and parts[1].startswith("/"):
                # Pattern like "**/foo" — match basename or full path
                tail = parts[1][1:]  # strip leading /
                # Try matching against every suffix of the path
                segments = rel_path.split("/")
                for i in range(len(segments)):
                    candidate = "/".join(segments[i:])
                    if fnmatch.fnmatchcase(candidate, tail):
                        return True
                return False
            # General case: expand ** to match any path segments
            # Use a recursive approach
            return _doublestar_match(pattern, rel_path)

        if anchored:
            return fnmatch.fnmatchcase(rel_path, pattern)

        # Unanchored: match against basename or any suffix path
        basename = rel_path.rsplit("/", 1)[-1]
        if fnmatch.fnmatchcase(basename, pattern):
            return True
        # Also try matching full path (for patterns like "dir/file")
        return fnmatch.fnmatchcase(rel_path, pattern)

    def is_ignored(self, rel_path: str, is_dir: bool = False) -> bool:
        """Test if a relative path matches any ignore rule."""
        return self._match(rel_path, is_dir)

    def is_dir_ignored(self, _dirname: str, rel_dir_path: str) -> bool:
        """Fast check for directory pruning during os.walk."""
        return self._match(rel_dir_path, is_dir=True)

    def is_file_ignored(self, rel_file_path: str) -> bool:
        """Check if a file should be ignored."""
        return self._match(rel_file_path, is_dir=False)


def _doublestar_match(pattern: str, path: str) -> bool:
    """Match a pattern containing ** against a path.

    ** matches zero or more path segments.
    """
    # Split pattern on **
    parts = pattern.split("**")
    if len(parts) == 1:
        return fnmatch.fnmatchcase(path, pattern)

    # Two-part case (most common): prefix/**/suffix
    if len(parts) == 2:
        prefix, suffix = parts
        # Clean up slashes around **
        if prefix.endswith("/"):
            prefix = prefix[:-1]
        if suffix.startswith("/"):
            suffix = suffix[1:]

        segments = path.split("/")
        n = len(segments)
        # Try every split point: prefix matches segments[:i], suffix matches segments[j:]
        for i in range(n + 1):
            prefix_path = "/".join(segments[:i]) if i > 0 else ""
            prefix_ok = not prefix or fnmatch.fnmatchcase(prefix_path, prefix)
            if not prefix_ok:
                continue
            # Try matching suffix against segments[j:] for j >= i
            for j in range(i, n + 1):
                suffix_path = "/".join(segments[j:]) if j < n else ""
                suffix_ok = not suffix or fnmatch.fnmatchcase(suffix_path, suffix)
                if suffix_ok:
                    return True
        return False

    # Multi-** patterns: chain matching
    # This is rare and not worth optimizing heavily
    return fnmatch.fnmatchcase(path, pattern.replace("**", "*"))
