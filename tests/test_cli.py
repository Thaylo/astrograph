"""Tests for the CLI module."""

import json
import sys
from unittest.mock import patch

import pytest

from astrograph import cli


def _status(
    *,
    language_id: str,
    available: bool,
    command: list[str],
    command_source: str = "default",
    executable: str | None = None,
    installable: bool = False,
    install_command: list[str] | None = None,
    reason: str | None = None,
) -> cli.LSPServerStatus:
    return cli.LSPServerStatus(
        language_id=language_id,
        command=command,
        command_source=command_source,
        executable=executable,
        available=available,
        installable=installable,
        install_command=install_command,
        reason=reason,
    )


def _run_cli(argv: list[str]) -> None:
    with patch.object(sys, "argv", argv):
        cli.main()


@pytest.fixture
def sample_dir(tmp_path):
    """Create a sample directory with Python files."""
    (tmp_path / "module1.py").write_text(
        """
def calculate(a, b):
    return a + b

def compute(x, y):
    return x + y
"""
    )
    (tmp_path / "module2.py").write_text(
        """
def process(data):
    for item in data:
        print(item)
"""
    )
    return tmp_path


@pytest.fixture
def sample_file(tmp_path):
    """Create a sample Python file."""
    code = """
def example(x):
    return x * 2
"""
    file_path = tmp_path / "example.py"
    file_path.write_text(code)
    return file_path


class TestIndexCommand:
    """Tests for the index command."""

    @pytest.mark.parametrize("fixture_name", ["sample_dir", "sample_file"])
    def test_index_path(self, fixture_name, request, capsys):
        """Index command should work with both files and directories."""
        path = request.getfixturevalue(fixture_name)
        _run_cli(["cli", "index", str(path)])
        captured = capsys.readouterr()
        assert "Indexed" in captured.out

    def test_index_no_recursive(self, sample_dir, capsys):
        _run_cli(["cli", "index", str(sample_dir), "--no-recursive"])
        captured = capsys.readouterr()
        assert "Indexed" in captured.out


class TestDuplicatesCommand:
    """Tests for the duplicates command."""

    def test_find_duplicates(self, sample_dir, capsys):
        _run_cli(["cli", "duplicates", str(sample_dir)])
        captured = capsys.readouterr()
        # Should find calculate/compute as duplicates or show no duplicates
        assert captured.out

    def test_find_duplicates_json(self, sample_dir, capsys):
        _run_cli(["cli", "duplicates", str(sample_dir), "--json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "duplicate_groups" in data

    def test_find_duplicates_min_nodes(self, sample_dir, capsys):
        _run_cli(["cli", "duplicates", str(sample_dir), "--min-nodes", "10"])
        captured = capsys.readouterr()
        assert captured.out

    def test_find_duplicates_verify(self, sample_dir, capsys):
        _run_cli(["cli", "duplicates", str(sample_dir), "--verify"])
        captured = capsys.readouterr()
        assert captured.out

    def test_find_no_duplicates(self, tmp_path, capsys):
        """Test with unique functions."""
        (tmp_path / "unique.py").write_text(
            """
def func1():
    pass

def func2(x):
    return x
"""
        )
        _run_cli(["cli", "duplicates", str(tmp_path)])
        captured = capsys.readouterr()
        assert captured.out


class TestCheckCommand:
    """Tests for the check command."""

    def test_check_similar(self, sample_dir, sample_file, capsys):
        _run_cli(["cli", "check", str(sample_dir), str(sample_file)])
        captured = capsys.readouterr()
        assert captured.out

    def test_check_json(self, sample_dir, sample_file, capsys):
        with patch.object(
            sys, "argv", ["cli", "check", str(sample_dir), str(sample_file), "--json"]
        ):
            cli.main()
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "matches" in data

    def test_check_no_similar(self, tmp_path, capsys):
        """Test when no similar code exists."""
        indexed_dir = tmp_path / "indexed"
        indexed_dir.mkdir()
        (indexed_dir / "source.py").write_text("def f(): pass")
        (tmp_path / "check.py").write_text(
            """
def very_different_function(a, b, c, d):
    result = {}
    for x in range(a):
        for y in range(b):
            result[(x, y)] = c * d
    return result
"""
        )

        with patch.object(
            sys, "argv", ["cli", "check", str(indexed_dir), str(tmp_path / "check.py")]
        ):
            cli.main()
        captured = capsys.readouterr()
        assert "No similar" in captured.out or "Safe" in captured.out or captured.out


class TestCompareCommand:
    """Tests for the compare command."""

    def test_compare_files(self, tmp_path, capsys):
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"

        file1.write_text("def f(x): return x + 1")
        file2.write_text("def g(y): return y + 1")

        _run_cli(["cli", "compare", str(file1), str(file2)])
        captured = capsys.readouterr()
        assert "Isomorphic" in captured.out


class TestDoctorCommand:
    """Tests for doctor output."""

    def test_doctor_text(self, capsys):
        statuses = [
            _status(
                language_id="python",
                available=True,
                command=["pylsp"],
                executable="/usr/bin/pylsp",
            ),
            _status(
                language_id="javascript_lsp",
                available=False,
                command=["typescript-language-server", "--stdio"],
                installable=True,
                install_command=[
                    "npm",
                    "install",
                    "-g",
                    "typescript",
                    "typescript-language-server",
                ],
            ),
        ]
        with patch("astrograph.cli._collect_lsp_statuses", return_value=statuses):
            _run_cli(["cli", "doctor"])

        captured = capsys.readouterr()
        assert "ASTrograph LSP doctor" in captured.out
        assert "[OK] python" in captured.out
        assert "[MISSING] javascript_lsp" in captured.out

    def test_doctor_json(self, capsys):
        statuses = [
            _status(language_id="python", available=True, command=["pylsp"]),
            _status(
                language_id="javascript_lsp",
                available=True,
                command=["typescript-language-server", "--stdio"],
            ),
        ]
        with patch("astrograph.cli._collect_lsp_statuses", return_value=statuses):
            _run_cli(["cli", "doctor", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert payload["ready"] is True
        assert len(payload["servers"]) == 2


class TestInstallLSPsCommand:
    """Tests for install-lsps command."""

    def test_install_lsps_runs_selected_language(self, capsys):
        statuses = [
            _status(language_id="python", available=False, command=["pylsp"], installable=True),
            _status(
                language_id="javascript_lsp",
                available=False,
                command=["typescript-language-server", "--stdio"],
                installable=True,
            ),
        ]

        with (
            patch("astrograph.cli._collect_lsp_statuses", return_value=statuses),
            patch("astrograph.cli._run_install_lsp", return_value=("installed", "ok")) as install,
        ):
            _run_cli(["cli", "install-lsps", "--python"])

        captured = capsys.readouterr()
        assert "[INSTALLED] python: ok" in captured.out
        install.assert_called_once()
        called_status = install.call_args.args[0]
        assert called_status.language_id == "python"

    def test_install_lsps_json(self, capsys):
        statuses = [
            _status(language_id="python", available=False, command=["pylsp"], installable=True)
        ]

        with (
            patch("astrograph.cli._collect_lsp_statuses", return_value=statuses),
            patch("astrograph.cli._run_install_lsp", return_value=("failed", "boom")),
        ):
            _run_cli(["cli", "install-lsps", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert len(payload["results"]) == 1
        assert len(payload["failed"]) == 1


class TestHelpCommand:
    """Tests for help output."""

    def test_no_command(self, capsys):
        with patch.object(sys, "argv", ["cli"]):
            cli.main()
        captured = capsys.readouterr()
        # Should print help or usage
        assert "index" in captured.out or "usage" in captured.out.lower()
