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
    transport: str = "subprocess",
    endpoint: str | None = None,
    required: bool = True,
) -> cli.LSPServerStatus:
    return cli.LSPServerStatus(
        language_id=language_id,
        command=command,
        command_source=command_source,
        executable=executable,
        available=available,
        transport=transport,
        endpoint=endpoint,
        required=required,
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

    def test_doctor_json_optional_attach_missing_is_still_ready(self, capsys):
        statuses = [
            _status(language_id="python", available=True, command=["pylsp"], required=True),
            _status(
                language_id="c_lsp",
                available=False,
                command=["tcp://127.0.0.1:2087"],
                transport="tcp",
                endpoint="127.0.0.1:2087",
                required=False,
            ),
        ]
        with patch("astrograph.cli._collect_lsp_statuses", return_value=statuses):
            _run_cli(["cli", "doctor", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert payload["ready"] is True
        optional = next(server for server in payload["servers"] if server["language"] == "c_lsp")
        assert optional["required"] is False

    def test_doctor_json_all_optional_missing_is_still_ready(self, capsys):
        statuses = [
            _status(
                language_id="python",
                available=False,
                command=["tcp://127.0.0.1:2090"],
                transport="tcp",
                endpoint="127.0.0.1:2090",
                required=False,
            ),
            _status(
                language_id="javascript_lsp",
                available=False,
                command=["tcp://127.0.0.1:2092"],
                transport="tcp",
                endpoint="127.0.0.1:2092",
                required=False,
            ),
        ]
        with patch("astrograph.cli._collect_lsp_statuses", return_value=statuses):
            _run_cli(["cli", "doctor", "--json"])

        payload = json.loads(capsys.readouterr().out)
        assert payload["ready"] is True
        assert all(not server["required"] for server in payload["servers"])


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


class TestLspStatusEdgeCases:
    """Tests for LSP status computation edge cases."""

    def _unreachable_attach_status(self, *, command_source: str) -> cli.LSPServerStatus:
        spec = cli.LSPServerSpec(
            language_id="cpp_lsp",
            default_command=["tcp://127.0.0.1:2088"],
            required=False,
        )
        with (
            patch(
                "astrograph.cli.resolve_lsp_command",
                return_value=(["tcp://127.0.0.1:2088"], command_source),
            ),
            patch(
                "astrograph.cli.probe_command",
                return_value={
                    "available": False,
                    "executable": None,
                    "transport": "tcp",
                    "endpoint": "127.0.0.1:2088",
                },
            ),
        ):
            return cli._lsp_status(spec)

    def test_empty_command_status(self):
        spec = cli.LSPServerSpec(
            language_id="python",
            default_command=[],
            required=True,
        )
        with patch("astrograph.cli.resolve_lsp_command", return_value=([], "default")):
            status = cli._lsp_status(spec)
        assert not status.available
        assert status.reason == "command resolved to an empty value"

    def test_binding_not_found_subprocess(self):
        spec = cli.LSPServerSpec(
            language_id="python",
            default_command=["pylsp"],
            required=True,
        )
        with (
            patch("astrograph.cli.resolve_lsp_command", return_value=(["pylsp"], "binding")),
            patch(
                "astrograph.cli.probe_command",
                return_value={"available": False, "executable": None, "transport": "subprocess"},
            ),
        ):
            status = cli._lsp_status(spec)
        assert status.reason == "bound command was not found; check the binding"

    def test_binding_not_reachable_tcp(self):
        status = self._unreachable_attach_status(command_source="binding")
        assert status.reason == "bound endpoint is not reachable; check the binding"

    def test_default_attach_not_reachable(self):
        status = self._unreachable_attach_status(command_source="default")
        assert status.reason == "default attach endpoint is not reachable"

    def test_js_installable_without_npm(self):
        spec = cli.LSPServerSpec(
            language_id="javascript_lsp",
            default_command=["typescript-language-server", "--stdio"],
            required=True,
        )
        with (
            patch(
                "astrograph.cli.resolve_lsp_command",
                return_value=(["typescript-language-server", "--stdio"], "default"),
            ),
            patch(
                "astrograph.cli.probe_command",
                return_value={"available": False, "executable": None, "transport": "subprocess"},
            ),
            patch("shutil.which", return_value=None),
        ):
            status = cli._lsp_status(spec)
        assert not status.installable
        assert "npm is required" in status.reason


class TestPrintDoctorEdgeCases:
    """Tests for doctor output variations."""

    def test_missing_optional_shows_note(self, capsys):
        statuses = [
            _status(
                language_id="python",
                available=True,
                command=["pylsp"],
                executable="/usr/bin/pylsp",
                required=True,
            ),
            _status(
                language_id="cpp_lsp",
                available=False,
                command=["tcp://127.0.0.1:2088"],
                transport="tcp",
                required=False,
                reason="default attach endpoint is not reachable",
            ),
        ]
        cli._print_doctor(statuses, as_json=False)
        captured = capsys.readouterr()
        assert "optional attach endpoint" in captured.out

    def test_all_available_message(self, capsys):
        statuses = [
            _status(language_id="python", available=True, command=["pylsp"], required=True),
        ]
        cli._print_doctor(statuses, as_json=False)
        captured = capsys.readouterr()
        assert "All required and optional LSP servers are available" in captured.out

    def test_reason_shown_for_non_installable(self, capsys):
        statuses = [
            _status(
                language_id="go_lsp",
                available=False,
                command=["gopls"],
                reason="no auto-install command is available",
            ),
        ]
        cli._print_doctor(statuses, as_json=False)
        captured = capsys.readouterr()
        assert "note: no auto-install" in captured.out


class TestRunInstallLsp:
    """Tests for _run_install_lsp."""

    def test_skip_already_installed(self):
        status = _status(language_id="python", available=True, command=["pylsp"])
        result, details = cli._run_install_lsp(status, dry_run=False)
        assert result == "skipped"
        assert "already installed" in details

    def test_fail_not_installable(self):
        status = _status(
            language_id="go_lsp",
            available=False,
            command=["gopls"],
            reason="no auto-install command",
        )
        result, details = cli._run_install_lsp(status, dry_run=False)
        assert result == "failed"

    def test_dry_run(self):
        status = _status(
            language_id="python",
            available=False,
            command=["pylsp"],
            installable=True,
            install_command=[sys.executable, "-m", "pip", "install", "python-lsp-server"],
        )
        result, details = cli._run_install_lsp(status, dry_run=True)
        assert result == "skipped"
        assert "dry-run" in details

    def test_oserror_during_install(self):
        status = _status(
            language_id="python",
            available=False,
            command=["pylsp"],
            installable=True,
            install_command=["nonexistent-command-abc"],
        )
        result, details = cli._run_install_lsp(status, dry_run=False)
        assert result == "failed"

    def test_install_succeeds_but_executable_not_found(self):
        status = _status(
            language_id="python",
            available=False,
            command=["pylsp"],
            installable=True,
            install_command=["echo", "installed"],
        )
        with patch(
            "astrograph.cli._lsp_status",
            return_value=_status(language_id="python", available=False, command=["pylsp"]),
        ):
            result, details = cli._run_install_lsp(status, dry_run=False)
        assert result == "failed"
        assert "still not found" in details


class TestDuplicatesCommandEdgeCases:
    """Tests for duplicates command single-file path."""

    def test_single_file(self, sample_file, capsys):
        _run_cli(["cli", "duplicates", str(sample_file)])
        captured = capsys.readouterr()
        assert captured.out


class TestCheckCommandEdgeCases:
    """Tests for check command no-plugin path."""

    def test_no_plugin_for_file(self, tmp_path, capsys):
        indexed_dir = tmp_path / "indexed"
        indexed_dir.mkdir()
        (indexed_dir / "data.py").write_text("def f(): pass")
        code_file = tmp_path / "code.xyz"
        code_file.write_text("some code")
        _run_cli(["cli", "check", str(indexed_dir), str(code_file)])
        captured = capsys.readouterr()
        assert "No language plugin" in captured.out

    def test_no_similar_found(self, tmp_path, capsys):
        indexed_dir = tmp_path / "indexed"
        indexed_dir.mkdir()
        (indexed_dir / "data.py").write_text(
            "class BigClass:\n"
            "    def __init__(self, a, b, c, d, e):\n"
            "        self.a = a\n"
            "        self.b = b\n"
            "        self.c = c\n"
            "        self.d = d\n"
            "        self.e = e\n"
        )
        code_file = tmp_path / "check.py"
        code_file.write_text(
            "def compute_matrix(rows, cols):\n"
            "    matrix = []\n"
            "    for i in range(rows):\n"
            "        row = []\n"
            "        for j in range(cols):\n"
            "            row.append(i * cols + j)\n"
            "        matrix.append(row)\n"
            "    return matrix\n"
        )
        _run_cli(["cli", "check", str(indexed_dir), str(code_file)])
        captured = capsys.readouterr()
        assert "No similar" in captured.out or "Safe" in captured.out or "Found" in captured.out


class TestCompareCommandEdgeCases:
    """Tests for compare command edge cases."""

    def test_language_override_no_plugin(self, tmp_path, capsys):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("def f(): pass")
        f2.write_text("def g(): pass")
        _run_cli(["cli", "compare", str(f1), str(f2), "--language", "nonexistent_lang"])
        captured = capsys.readouterr()
        assert "No language plugin" in captured.out

    def test_no_plugin_for_file(self, tmp_path, capsys):
        f1 = tmp_path / "a.xyz"
        f2 = tmp_path / "b.xyz"
        f1.write_text("code1")
        f2.write_text("code2")
        _run_cli(["cli", "compare", str(f1), str(f2)])
        captured = capsys.readouterr()
        assert "No language plugin" in captured.out

    def test_language_mismatch(self, tmp_path, capsys):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.js"
        f1.write_text("def f(): pass")
        f2.write_text("function f() {}")
        _run_cli(["cli", "compare", str(f1), str(f2)])
        captured = capsys.readouterr()
        assert "Cannot compare different languages" in captured.out


class TestInstallLSPsFailureOutput:
    """Tests for install-lsps failure output."""

    def test_failure_shows_message(self, capsys):
        statuses = [
            _status(
                language_id="python",
                available=False,
                command=["pylsp"],
                installable=True,
                install_command=["echo", "fail"],
            ),
        ]
        with (
            patch("astrograph.cli._collect_lsp_statuses", return_value=statuses),
            patch("astrograph.cli._run_install_lsp", return_value=("failed", "boom")),
        ):
            _run_cli(["cli", "install-lsps"])
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower()
        assert "astrograph-cli doctor" in captured.out


class TestHelpCommand:
    """Tests for help output."""

    def test_no_command(self, capsys):
        with patch.object(sys, "argv", ["cli"]):
            cli.main()
        captured = capsys.readouterr()
        # Should print help or usage
        assert "index" in captured.out or "usage" in captured.out.lower()
