#!/usr/bin/env python3
"""CLI tool for code structure analysis."""

import argparse
import json
from pathlib import Path

import networkx as nx

from .canonical_hash import weisfeiler_leman_hash
from .index import CodeStructureIndex
from .languages.base import node_match
from .languages.registry import LanguageRegistry


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

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
