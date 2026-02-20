"""Tests for the .astrographignore pattern matching module."""

import tempfile
from pathlib import Path

from astrograph.ignorefile import ASTROGRAPHIGNORE_FILENAME, IgnoreSpec


class TestIgnoreSpec:
    """Unit tests for IgnoreSpec pattern matching."""

    def test_empty_spec_matches_nothing(self):
        spec = IgnoreSpec.from_lines([])
        assert not spec.is_file_ignored("foo.py")
        assert not spec.is_file_ignored("src/bar.js")
        assert not spec.is_dir_ignored("vendor", "vendor")

    def test_comment_lines_skipped(self):
        spec = IgnoreSpec.from_lines(["# this is a comment", "# *.py"])
        assert not spec.is_file_ignored("foo.py")

    def test_blank_lines_skipped(self):
        spec = IgnoreSpec.from_lines(["", "  ", "\t"])
        assert not spec.is_file_ignored("foo.py")

    def test_simple_file_glob(self):
        spec = IgnoreSpec.from_lines(["*.min.js"])
        assert spec.is_file_ignored("app.min.js")
        assert spec.is_file_ignored("src/lib/app.min.js")
        assert not spec.is_file_ignored("app.js")

    def test_directory_pattern(self):
        spec = IgnoreSpec.from_lines(["vendor/"])
        assert spec.is_dir_ignored("vendor", "vendor")
        assert spec.is_ignored("vendor/foo.js", is_dir=False)
        # A file literally named "vendor" (no trailing slash) should not be ignored
        # because the rule is dir-only
        assert not spec.is_file_ignored("vendor")

    def test_anchored_pattern(self):
        """Patterns with / are anchored to root."""
        spec = IgnoreSpec.from_lines(["/build"])
        assert spec.is_file_ignored("build")
        assert not spec.is_file_ignored("src/build")

    def test_negation_pattern(self):
        spec = IgnoreSpec.from_lines(["*.js", "!important.js"])
        assert spec.is_file_ignored("app.js")
        assert not spec.is_file_ignored("important.js")
        assert spec.is_file_ignored("other.js")

    def test_doublestar_pattern(self):
        spec = IgnoreSpec.from_lines(["**/test_*.py"])
        assert spec.is_file_ignored("test_foo.py")
        assert spec.is_file_ignored("tests/test_bar.py")
        assert spec.is_file_ignored("src/tests/test_baz.py")
        assert not spec.is_file_ignored("src/foo.py")

    def test_from_file_nonexistent(self):
        result = IgnoreSpec.from_file("/nonexistent/path/.astrographignore")
        assert result is None

    def test_from_file_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ignore_file = Path(tmpdir) / ASTROGRAPHIGNORE_FILENAME
            ignore_file.write_text("*.min.js\nvendor/\n")
            spec = IgnoreSpec.from_file(str(ignore_file))
            assert spec is not None
            assert spec.is_file_ignored("app.min.js")
            assert spec.is_dir_ignored("vendor", "vendor")

    def test_is_dir_ignored(self):
        spec = IgnoreSpec.from_lines(["vendor/"])
        assert spec.is_dir_ignored("vendor", "vendor")
        assert not spec.is_dir_ignored("src", "src")

    def test_is_dir_ignored_nested(self):
        """src/vendor/ only matches under src, not at root."""
        spec = IgnoreSpec.from_lines(["src/vendor/"])
        assert spec.is_dir_ignored("vendor", "src/vendor")
        assert not spec.is_dir_ignored("vendor", "vendor")

    def test_multiple_patterns_last_wins(self):
        spec = IgnoreSpec.from_lines(["*.js", "!app.js", "app.js"])
        # Last rule matches app.js → ignored
        assert spec.is_file_ignored("app.js")

    def test_doublestar_middle(self):
        """Pattern like a/**/b matches a/b, a/x/b, a/x/y/b."""
        spec = IgnoreSpec.from_lines(["src/**/test.py"])
        assert spec.is_file_ignored("src/test.py")
        assert spec.is_file_ignored("src/foo/test.py")
        assert spec.is_file_ignored("src/foo/bar/test.py")
        assert not spec.is_file_ignored("test.py")

    def test_doublestar_glob_min_js(self):
        """**/*.min.js matches .min.js files at any depth."""
        spec = IgnoreSpec.from_lines(["**/*.min.js"])
        assert spec.is_file_ignored("app.min.js")
        assert spec.is_file_ignored("src/app.min.js")
        assert spec.is_file_ignored("src/dist/app.min.js")
        assert spec.is_file_ignored("src/dist/vendor/jquery.min.js")
        assert not spec.is_file_ignored("app.js")
        assert not spec.is_file_ignored("src/app.js")
        assert not spec.is_file_ignored("app.min.css")

    def test_wildcard_in_path(self):
        spec = IgnoreSpec.from_lines(["dist/*.js"])
        assert spec.is_file_ignored("dist/bundle.js")
        assert not spec.is_file_ignored("src/bundle.js")
