#!/usr/bin/env python3
"""CLI tool for code structure analysis."""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from .canonical_hash import weisfeiler_leman_hash
from .index import CodeStructureIndex
from .languages.base import node_match
from .languages.registry import LanguageRegistry


@dataclass(frozen=True)
class LSPServerSpec:
    """Configuration for a bundled LSP-backed language adapter."""

    language_id: str
    command_env_var: str
    default_command: tuple[str, ...]


@dataclass
class LSPServerStatus:
    """Resolved availability and installation status for an LSP server."""

    language_id: str
    command: list[str]
    command_source: str  # "default" or "env"
    executable: str | None
    available: bool
    installable: bool
    install_command: list[str] | None
    reason: str | None = None


def _bundled_lsp_specs() -> tuple[LSPServerSpec, ...]:
    """Return built-in plugin LSP server specs."""
    return (
        LSPServerSpec(
            language_id="python",
            command_env_var="ASTROGRAPH_PY_LSP_COMMAND",
            default_command=("pylsp",),
        ),
        LSPServerSpec(
            language_id="javascript_lsp",
            command_env_var="ASTROGRAPH_JS_LSP_COMMAND",
            default_command=("typescript-language-server", "--stdio"),
        ),
    )


def _resolve_lsp_command(spec: LSPServerSpec) -> tuple[list[str], str]:
    """Resolve command from env override or default."""
    command_text = os.getenv(spec.command_env_var, "").strip()
    if command_text:
        return shlex.split(command_text), "env"
    return list(spec.default_command), "default"


def _default_install_command(spec: LSPServerSpec) -> list[str] | None:
    """Return auto-install command for known default servers."""
    if spec.language_id == "python":
        return [sys.executable, "-m", "pip", "install", "python-lsp-server>=1.11"]
    if spec.language_id == "javascript_lsp":
        return ["npm", "install", "-g", "typescript", "typescript-language-server"]
    return None


def _lsp_status(spec: LSPServerSpec) -> LSPServerStatus:
    """Compute runtime availability and auto-install options for one server."""
    command, source = _resolve_lsp_command(spec)
    executable_name = command[0] if command else ""

    if not executable_name:
        return LSPServerStatus(
            language_id=spec.language_id,
            command=command,
            command_source=source,
            executable=None,
            available=False,
            installable=False,
            install_command=None,
            reason=f"{spec.command_env_var} resolved to an empty command",
        )

    executable = shutil.which(executable_name)
    available = executable is not None

    install_command = _default_install_command(spec)
    installable = source == "default" and install_command is not None
    reason: str | None = None

    if source == "env" and not available:
        reason = (
            f"custom command from {spec.command_env_var} was not found; "
            "auto-install is disabled for custom commands"
        )
    elif not installable and not available:
        reason = "no auto-install command is available"

    if spec.language_id == "javascript_lsp" and installable and shutil.which("npm") is None:
        installable = False
        reason = "npm is required to install typescript-language-server"

    return LSPServerStatus(
        language_id=spec.language_id,
        command=command,
        command_source=source,
        executable=executable,
        available=available,
        installable=installable,
        install_command=install_command if installable else None,
        reason=reason,
    )


def _collect_lsp_statuses() -> list[LSPServerStatus]:
    """Collect status for all bundled LSP servers."""
    return [_lsp_status(spec) for spec in _bundled_lsp_specs()]


def _print_doctor(statuses: list[LSPServerStatus], as_json: bool) -> None:
    """Render `doctor` command output."""
    missing = [s for s in statuses if not s.available]

    if as_json:
        payload = {
            "ready": not missing,
            "servers": [
                {
                    "language": s.language_id,
                    "available": s.available,
                    "command": s.command,
                    "command_source": s.command_source,
                    "executable": s.executable,
                    "installable": s.installable,
                    "install_command": s.install_command,
                    "reason": s.reason,
                }
                for s in statuses
            ],
        }
        print(json.dumps(payload, indent=2))
        return

    print("ASTrograph LSP doctor")
    for status in statuses:
        state = "OK" if status.available else "MISSING"
        command = " ".join(status.command) if status.command else "<empty>"
        source = f" ({status.command_source})" if status.command_source == "env" else ""
        print(f"[{state}] {status.language_id}: {command}{source}")
        if status.executable:
            print(f"      executable: {status.executable}")
        elif status.installable and status.install_command:
            print(f"      fix: {' '.join(status.install_command)}")
        elif status.reason:
            print(f"      note: {status.reason}")

    if missing:
        print(f"\nMissing {len(missing)} LSP server(s).")
    else:
        print("\nAll bundled LSP servers are available.")


def _run_install_lsp(
    status: LSPServerStatus,
    *,
    dry_run: bool,
) -> tuple[str, str]:
    """Attempt installing one missing LSP server.

    Returns:
        (result, details) where result is one of: installed, skipped, failed.
    """
    if status.available:
        return "skipped", "already installed"

    if not status.installable or not status.install_command:
        return "failed", status.reason or "not installable"

    command_text = " ".join(status.install_command)
    if dry_run:
        return "skipped", f"dry-run: {command_text}"

    try:
        completed = subprocess.run(
            status.install_command,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return "failed", str(exc)
    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        details = stderr.splitlines()[-1] if stderr else f"exit code {completed.returncode}"
        return "failed", details

    refreshed = _lsp_status(
        LSPServerSpec(
            language_id=status.language_id,
            command_env_var=(
                "ASTROGRAPH_PY_LSP_COMMAND"
                if status.language_id == "python"
                else "ASTROGRAPH_JS_LSP_COMMAND"
            ),
            default_command=("pylsp",)
            if status.language_id == "python"
            else ("typescript-language-server", "--stdio"),
        )
    )
    if refreshed.available:
        return "installed", command_text
    return "failed", "installer ran but executable still not found"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze code structure and find duplicates using graph isomorphism"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Index command
    index_parser = subparsers.add_parser("index", help="Index a codebase")
    index_parser.add_argument("path", help="Path to directory or file to index")
    index_parser.add_argument(
        "--no-recursive", action="store_true", help="Don't recurse into subdirectories"
    )

    # Find duplicates command
    dups_parser = subparsers.add_parser("duplicates", help="Find structural duplicates")
    dups_parser.add_argument("path", help="Path to directory or file to analyze")
    dups_parser.add_argument(
        "--min-nodes", type=int, default=5, help="Minimum AST nodes to consider"
    )
    dups_parser.add_argument("--verify", action="store_true", help="Verify with full isomorphism")
    dups_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Check similar command
    check_parser = subparsers.add_parser("check", help="Check if similar code exists")
    check_parser.add_argument("path", help="Path to directory to index")
    check_parser.add_argument("code_file", help="Path to code file to check")
    check_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare two code files")
    compare_parser.add_argument("file1", help="First file")
    compare_parser.add_argument("file2", help="Second file")
    compare_parser.add_argument(
        "--language",
        help="Language ID override (otherwise inferred from file extensions)",
    )

    doctor_parser = subparsers.add_parser("doctor", help="Check bundled LSP server readiness")
    doctor_parser.add_argument("--json", action="store_true", help="Output as JSON")

    install_parser = subparsers.add_parser(
        "install-lsps",
        help="Install missing bundled LSP servers (python/javascript)",
    )
    install_parser.add_argument("--python", action="store_true", help="Install Python LSP only")
    install_parser.add_argument(
        "--javascript",
        action="store_true",
        help="Install JavaScript LSP only",
    )
    install_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show install commands without executing them",
    )
    install_parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.command == "index":
        index = CodeStructureIndex()
        path = Path(args.path)

        if path.is_file():
            entries = index.index_file(str(path))
        else:
            entries = index.index_directory(str(path), recursive=not args.no_recursive)

        print(f"Indexed {len(entries)} code units")
        print(json.dumps(index.get_stats(), indent=2))

    elif args.command == "duplicates":
        index = CodeStructureIndex()
        path = Path(args.path)

        if path.is_file():
            index.index_file(str(path))
        else:
            index.index_directory(str(path))

        groups = index.find_all_duplicates(min_node_count=args.min_nodes)

        if args.json:
            output = {
                "duplicate_groups": len(groups),
                "groups": [
                    {
                        "hash": g.wl_hash[:16],
                        "count": len(g.entries),
                        "locations": [
                            {
                                "file": e.code_unit.file_path,
                                "name": e.code_unit.name,
                                "lines": f"{e.code_unit.line_start}-{e.code_unit.line_end}",
                            }
                            for e in g.entries
                        ],
                    }
                    for g in groups
                ],
            }
            print(json.dumps(output, indent=2))
        else:
            if not groups:
                print("No structural duplicates found.")
            else:
                print(f"Found {len(groups)} duplicate group(s):\n")
                for i, group in enumerate(groups, 1):
                    print(f"Group {i} ({len(group.entries)} occurrences):")
                    for entry in group.entries:
                        loc = f"{entry.code_unit.file_path}:{entry.code_unit.line_start}"
                        print(f"  - {loc} ({entry.code_unit.name})")

                        if args.verify and len(group.entries) >= 2:
                            verified = index.verify_isomorphism(group.entries[0], group.entries[1])
                            print(f"    [Verified isomorphic: {verified}]")
                    print()

    elif args.command == "check":
        index = CodeStructureIndex()
        check_path = Path(args.path)
        if check_path.is_file():
            index.index_file(str(check_path))
        else:
            index.index_directory(str(check_path))

        code = Path(args.code_file).read_text()
        plugin = LanguageRegistry.get().get_plugin_for_file(args.code_file)
        if plugin is None:
            print(
                "No language plugin registered for "
                f"{Path(args.code_file).suffix or '<no extension>'} files."
            )
            return

        results = index.find_similar(code, min_node_count=3, language=plugin.language_id)

        if args.json:
            output = {
                "matches": [
                    {
                        "similarity": r.similarity_type,
                        "file": r.entry.code_unit.file_path,
                        "name": r.entry.code_unit.name,
                        "lines": f"{r.entry.code_unit.line_start}-{r.entry.code_unit.line_end}",
                    }
                    for r in results
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            if not results:
                print("No similar code found. Safe to proceed.")
            else:
                print(f"Found {len(results)} similar code unit(s):")
                for r in results:
                    loc = f"{r.entry.code_unit.file_path}:{r.entry.code_unit.line_start}"
                    print(f"  [{r.similarity_type}] {loc} ({r.entry.code_unit.name})")

    elif args.command == "compare":
        code1 = Path(args.file1).read_text()
        code2 = Path(args.file2).read_text()

        registry = LanguageRegistry.get()
        if args.language:
            plugin = registry.get_plugin(args.language)
            if plugin is None:
                print(f"No language plugin registered for '{args.language}'.")
                return
        else:
            plugin1 = registry.get_plugin_for_file(args.file1)
            plugin2 = registry.get_plugin_for_file(args.file2)
            if plugin1 is None or plugin2 is None:
                missing = args.file1 if plugin1 is None else args.file2
                print(
                    "No language plugin registered for "
                    f"{Path(missing).suffix or '<no extension>'} files."
                )
                return
            if plugin1.language_id != plugin2.language_id:
                print(
                    f"Cannot compare different languages: "
                    f"{plugin1.language_id} vs {plugin2.language_id}"
                )
                return
            plugin = plugin1

        g1 = plugin.source_to_graph(code1)
        g2 = plugin.source_to_graph(code2)

        h1 = weisfeiler_leman_hash(g1)
        h2 = weisfeiler_leman_hash(g2)

        is_iso = nx.is_isomorphic(g1, g2, node_match=node_match)

        print(f"File 1: {g1.number_of_nodes()} nodes, hash: {h1[:16]}...")
        print(f"File 2: {g2.number_of_nodes()} nodes, hash: {h2[:16]}...")
        print(f"Hash match: {h1 == h2}")
        print(f"Isomorphic: {is_iso}")

    elif args.command == "doctor":
        _print_doctor(_collect_lsp_statuses(), as_json=args.json)

    elif args.command == "install-lsps":
        statuses = _collect_lsp_statuses()
        selected_languages = {
            status.language_id
            for status in statuses
            if (
                (args.python and status.language_id == "python")
                or (args.javascript and status.language_id == "javascript_lsp")
            )
        }
        if not selected_languages:
            selected_languages = {status.language_id for status in statuses}

        selected_statuses = [s for s in statuses if s.language_id in selected_languages]
        install_results: list[dict[str, str]] = []

        for status in selected_statuses:
            result, details = _run_install_lsp(status, dry_run=args.dry_run)
            install_results.append(
                {
                    "language": status.language_id,
                    "result": result,
                    "details": details,
                }
            )

        if args.json:
            payload = {
                "results": install_results,
                "failed": [r for r in install_results if r["result"] == "failed"],
            }
            print(json.dumps(payload, indent=2))
        else:
            for item in install_results:
                print(f"[{item['result'].upper()}] {item['language']}: {item['details']}")
            failures = [r for r in install_results if r["result"] == "failed"]
            if failures:
                print(f"\n{len(failures)} install step(s) failed. Run `astrograph-cli doctor`.")
            else:
                print("\nLSP install step completed. Run `astrograph-cli doctor` to verify.")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
