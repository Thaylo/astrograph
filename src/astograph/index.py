"""
Code structure index for fast similarity queries.

Stores AST graphs with their hashes for efficient duplicate detection.
"""

import hashlib
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from .ast_to_graph import (
    BLOCK_TYPE_NAMES,
    ASTGraph,
    CodeUnit,
    ast_to_graph,
    code_unit_to_ast_graph,
    extract_code_units,
    node_match,
)
from .canonical_hash import (
    compute_hierarchy_hash,
    fingerprints_compatible,
    structural_fingerprint,
    weisfeiler_leman_hash,
)

# Directories to skip when indexing Python files
_SKIP_DIRS = frozenset(["__pycache__", ".git", "venv", ".venv", "node_modules", ".tox"])


def _should_skip_path(path_parts: tuple[str, ...]) -> bool:
    """Check if a path should be skipped based on directory names."""
    return any(p in _SKIP_DIRS for p in path_parts)


@dataclass
class IndexEntry:
    """An entry in the code structure index."""

    id: str
    wl_hash: str
    pattern_hash: str  # Hash with normalized operators for pattern matching
    fingerprint: dict
    hierarchy_hashes: list[str]
    code_unit: CodeUnit
    node_count: int
    depth: int

    def to_dict(self) -> dict:
        code_unit_dict = {
            "name": self.code_unit.name,
            "code": self.code_unit.code,
            "file_path": self.code_unit.file_path,
            "line_start": self.code_unit.line_start,
            "line_end": self.code_unit.line_end,
            "unit_type": self.code_unit.unit_type,
            "parent_name": self.code_unit.parent_name,
        }
        # Include block-specific fields if present
        if self.code_unit.block_type:
            code_unit_dict["block_type"] = self.code_unit.block_type
        if self.code_unit.nesting_depth > 0:
            code_unit_dict["nesting_depth"] = self.code_unit.nesting_depth
        if self.code_unit.parent_block_name:
            code_unit_dict["parent_block_name"] = self.code_unit.parent_block_name

        return {
            "id": self.id,
            "wl_hash": self.wl_hash,
            "pattern_hash": self.pattern_hash,
            "fingerprint": self.fingerprint,
            "hierarchy_hashes": self.hierarchy_hashes,
            "code_unit": code_unit_dict,
            "node_count": self.node_count,
            "depth": self.depth,
        }

    def to_location_dict(self, include_code: bool = False) -> dict:
        """Return a compact location dictionary for tool output."""
        result = {
            "file": self.code_unit.file_path,
            "name": self.code_unit.name,
            "type": self.code_unit.unit_type,
            "lines": f"{self.code_unit.line_start}-{self.code_unit.line_end}",
        }
        if self.code_unit.parent_name:
            result["parent"] = self.code_unit.parent_name
        if include_code:
            code = self.code_unit.code
            result["code"] = code[:500] + ("..." if len(code) > 500 else "")
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "IndexEntry":
        cu_data = data["code_unit"]
        code_unit = CodeUnit(
            name=cu_data["name"],
            code=cu_data["code"],
            file_path=cu_data["file_path"],
            line_start=cu_data["line_start"],
            line_end=cu_data["line_end"],
            unit_type=cu_data["unit_type"],
            parent_name=cu_data.get("parent_name"),
            block_type=cu_data.get("block_type"),
            nesting_depth=cu_data.get("nesting_depth", 0),
            parent_block_name=cu_data.get("parent_block_name"),
        )
        return cls(
            id=data["id"],
            wl_hash=data["wl_hash"],
            pattern_hash=data.get("pattern_hash", data["wl_hash"]),  # Fallback for old data
            fingerprint=data["fingerprint"],
            hierarchy_hashes=data["hierarchy_hashes"],
            code_unit=code_unit,
            node_count=data["node_count"],
            depth=data["depth"],
        )


@dataclass
class DuplicateGroup:
    """A group of structurally equivalent code units."""

    wl_hash: str
    entries: list[IndexEntry]
    is_verified: bool = False  # True if full isomorphism check passed


@dataclass
class SimilarityResult:
    """Result of a similarity query."""

    entry: IndexEntry
    similarity_type: str  # 'exact', 'high', 'partial'
    matching_depth: int | None = None  # For partial matches, how deep the match goes


@dataclass
class FileMetadata:
    """Metadata about an indexed file for staleness detection."""

    file_path: str
    mtime: float  # os.path.getmtime() at index time
    content_hash: str  # SHA256 of file content
    indexed_at: float  # time.time() when indexed
    entry_count: int  # number of entries from this file

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "mtime": self.mtime,
            "content_hash": self.content_hash,
            "indexed_at": self.indexed_at,
            "entry_count": self.entry_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FileMetadata":
        return cls(
            file_path=data["file_path"],
            mtime=data["mtime"],
            content_hash=data["content_hash"],
            indexed_at=data["indexed_at"],
            entry_count=data["entry_count"],
        )


@dataclass
class SuppressionInfo:
    """Detailed information about a suppressed hash."""

    wl_hash: str
    reason: str | None
    created_at: float
    source_name: str | None  # Code unit name (for display)
    code_preview: str | None  # First 200 chars (for display)
    entry_count: int  # Number of entries when suppressed
    source_files: list[str] = field(default_factory=list)  # All files containing duplicates
    file_hashes: dict[str, str] = field(default_factory=dict)  # file -> content hash

    @property
    def primary_file(self) -> str | None:
        """Primary file for display purposes."""
        return self.source_files[0] if self.source_files else None

    def to_dict(self) -> dict:
        return {
            "wl_hash": self.wl_hash,
            "reason": self.reason,
            "created_at": self.created_at,
            "source_name": self.source_name,
            "code_preview": self.code_preview,
            "entry_count": self.entry_count,
            "source_files": self.source_files,
            "file_hashes": self.file_hashes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SuppressionInfo":
        return cls(
            wl_hash=data["wl_hash"],
            reason=data.get("reason"),
            created_at=data["created_at"],
            source_name=data.get("source_name"),
            code_preview=data.get("code_preview"),
            entry_count=data.get("entry_count", 0),
            source_files=data.get("source_files", []),
            file_hashes=data.get("file_hashes", {}),
        )


@dataclass
class StalenessReport:
    """Report on index staleness relative to filesystem."""

    is_stale: bool
    modified_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    new_files: list[str] = field(default_factory=list)
    stale_suppressions: list[str] = field(default_factory=list)


class CodeStructureIndex:
    """
    Index for code structures enabling fast duplicate detection.

    Uses a multi-level approach:
    1. WL hash for exact structural matches (O(1) lookup)
    2. Fingerprint filtering for candidate reduction
    3. Hierarchy hashes for partial matching
    4. Full isomorphism check for verification
    """

    def __init__(self) -> None:
        self.entries: dict[str, IndexEntry] = {}  # id -> entry
        self.hash_buckets: dict[str, set[str]] = {}  # wl_hash -> {entry_ids}
        self.pattern_buckets: dict[str, set[str]] = {}  # pattern_hash -> {entry_ids}
        self.file_entries: dict[str, list[str]] = {}  # file_path -> [entry_ids]
        # Block-specific storage (keeps function lookups O(1))
        self.block_buckets: dict[str, set[str]] = {}  # wl_hash -> {block entry_ids}
        self.block_type_index: dict[str, set[str]] = {}  # block_type -> {entry_ids}
        # Fingerprint index for O(1) similarity lookups: (n_nodes, n_edges) -> {entry_ids}
        self.fingerprint_index: dict[tuple[int, int], set[str]] = {}
        # Suppressed entries (reviewed and deemed acceptable)
        self.suppressed_hashes: set[str] = set()  # wl_hashes to ignore in analysis
        # File metadata for staleness detection
        self.file_metadata: dict[str, FileMetadata] = {}  # file_path -> metadata
        # Detailed suppression info for lifecycle tracking
        self.suppression_details: dict[str, SuppressionInfo] = {}  # wl_hash -> info
        self._entry_counter = 0
        # Incremental counters for O(1) stats
        self._block_entry_count = 0
        self._function_entry_count = 0
        # Persistence path for auto-save (None = no auto-save)
        self._persistence_path: Path | None = None

    def _generate_id(self) -> str:
        self._entry_counter += 1
        return f"entry_{self._entry_counter}"

    def set_persistence_path(self, path: Path | None) -> None:
        """Set the persistence path for auto-save. None disables auto-save."""
        self._persistence_path = path

    def get_persistence_path(self) -> Path | None:
        """Get the current persistence path."""
        return self._persistence_path

    def _auto_save(self) -> None:
        """Save to persistence path if configured."""
        if self._persistence_path:
            index_file = self._persistence_path / "index.json"
            self.save(str(index_file))

    def _compute_file_hash(self, file_path: str) -> str | None:
        """Compute SHA256 hash of file content."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return None

    def check_file_changed(self, file_path: str) -> bool:
        """
        Check if a file has changed since it was indexed.

        Returns True if the file has changed or doesn't exist in metadata.
        Uses mtime for quick check, falls back to content hash for accuracy.
        """
        if file_path not in self.file_metadata:
            return True  # Not tracked, treat as changed

        metadata = self.file_metadata[file_path]

        try:
            current_mtime = os.path.getmtime(file_path)
        except OSError:
            return True  # File doesn't exist or can't be accessed

        # Quick mtime check
        if current_mtime != metadata.mtime:
            # Mtime differs, verify with content hash
            current_hash = self._compute_file_hash(file_path)
            return current_hash != metadata.content_hash

        return False  # mtime matches, assume unchanged

    def add_code_unit(self, unit: CodeUnit) -> IndexEntry:
        """Add a code unit to the index."""
        ast_graph = code_unit_to_ast_graph(unit)
        return self.add_ast_graph(ast_graph)

    def add_ast_graph(self, ast_graph: ASTGraph) -> IndexEntry:
        """Add an AST graph to the index."""
        entry_id = self._generate_id()

        wl_hash = weisfeiler_leman_hash(ast_graph.graph)
        fp = structural_fingerprint(ast_graph.graph)
        hierarchy = list(compute_hierarchy_hash(ast_graph.graph))

        # Compute pattern hash with normalized operators
        pattern_graph = ast_to_graph(ast_graph.code_unit.code, normalize_ops=True)
        pattern_hash = weisfeiler_leman_hash(pattern_graph)

        entry = IndexEntry(
            id=entry_id,
            wl_hash=wl_hash,
            pattern_hash=pattern_hash,
            fingerprint=fp,
            hierarchy_hashes=hierarchy,
            code_unit=ast_graph.code_unit,
            node_count=ast_graph.node_count,
            depth=ast_graph.depth,
        )

        self.entries[entry_id] = entry

        is_block = ast_graph.code_unit.unit_type == "block"

        if is_block:
            # Add to block-specific buckets (keeps function lookups O(1))
            self.block_buckets.setdefault(wl_hash, set()).add(entry_id)

            # Add to block type index
            block_type = ast_graph.code_unit.block_type
            if block_type:
                self.block_type_index.setdefault(block_type, set()).add(entry_id)
            self._block_entry_count += 1
        else:
            # Add to function/class hash bucket (exact matches)
            self.hash_buckets.setdefault(wl_hash, set()).add(entry_id)

            # Add to pattern bucket (pattern matches - same structure, different operators)
            self.pattern_buckets.setdefault(pattern_hash, set()).add(entry_id)
            self._function_entry_count += 1

        # Add to fingerprint index for O(1) similarity lookups (skip empty graphs)
        if "n_nodes" in fp:
            fp_key = (fp["n_nodes"], fp["n_edges"])
            self.fingerprint_index.setdefault(fp_key, set()).add(entry_id)

        # Add to file index
        self.file_entries.setdefault(ast_graph.code_unit.file_path, []).append(entry_id)

        return entry

    def index_file(
        self,
        file_path: str,
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> list[IndexEntry]:
        """
        Index all code units from a Python file.

        Args:
            file_path: Path to the Python file
            include_blocks: If True, also extract code blocks (for, while, if, try, with)
            max_block_depth: Maximum nesting depth for block extraction (default 3)
        """
        path = Path(file_path)
        if not path.exists() or path.suffix != ".py":
            return []

        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []

        # Remove existing entries for this file
        self.remove_file(file_path)

        entries = []
        for unit in extract_code_units(source, file_path, include_blocks, max_block_depth):
            entry = self.add_code_unit(unit)
            entries.append(entry)

        # Record file metadata for staleness detection
        content_hash = self._compute_file_hash(file_path)
        if content_hash:
            self.file_metadata[file_path] = FileMetadata(
                file_path=file_path,
                mtime=os.path.getmtime(file_path),
                content_hash=content_hash,
                indexed_at=time.time(),
                entry_count=len(entries),
            )

        return entries

    def index_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> list[IndexEntry]:
        """
        Index all Python files in a directory.

        Args:
            dir_path: Path to the directory
            recursive: If True, search recursively (default True)
            include_blocks: If True, also extract code blocks (for, while, if, try, with)
            max_block_depth: Maximum nesting depth for block extraction (default 3)
        """
        path = Path(dir_path)
        if not path.exists():
            return []

        all_entries = []
        pattern = "**/*.py" if recursive else "*.py"

        for py_file in path.glob(pattern):
            if _should_skip_path(py_file.parts):
                continue
            entries = self.index_file(str(py_file), include_blocks, max_block_depth)
            all_entries.extend(entries)

        return all_entries

    def index_file_if_changed(
        self,
        file_path: str,
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> tuple[list[IndexEntry], bool]:
        """
        Index a file only if it has changed since last indexing.

        Args:
            file_path: Path to the Python file
            include_blocks: If True, also extract code blocks
            max_block_depth: Maximum nesting depth for block extraction

        Returns:
            Tuple of (entries, was_changed). If unchanged, returns ([], False).
        """
        if not self.check_file_changed(file_path):
            return [], False

        entries = self.index_file(file_path, include_blocks, max_block_depth)
        return entries, True

    def index_directory_incremental(
        self,
        dir_path: str,
        recursive: bool = True,
        include_blocks: bool = True,
        max_block_depth: int = 3,
    ) -> tuple[list[IndexEntry], int, int, int]:
        """
        Incrementally index a directory, only processing changed files.

        Args:
            dir_path: Path to the directory
            recursive: If True, search recursively (default True)
            include_blocks: If True, also extract code blocks
            max_block_depth: Maximum nesting depth for block extraction

        Returns:
            Tuple of (all_entries, added_count, updated_count, unchanged_count)
        """
        path = Path(dir_path)
        if not path.exists():
            return [], 0, 0, 0

        all_entries: list[IndexEntry] = []
        added_count = 0
        updated_count = 0
        unchanged_count = 0
        seen_files: set[str] = set()

        pattern = "**/*.py" if recursive else "*.py"

        for py_file in path.glob(pattern):
            if _should_skip_path(py_file.parts):
                continue

            file_str = str(py_file)
            seen_files.add(file_str)

            was_new = file_str not in self.file_metadata
            entries, was_changed = self.index_file_if_changed(
                file_str, include_blocks, max_block_depth
            )

            if was_changed:
                if was_new:
                    added_count += 1
                else:
                    updated_count += 1
                all_entries.extend(entries)
            else:
                unchanged_count += 1
                # Include existing entries in result
                if file_str in self.file_entries:
                    for entry_id in self.file_entries[file_str]:
                        if entry_id in self.entries:
                            all_entries.append(self.entries[entry_id])

        # Remove files that no longer exist
        files_to_remove = [f for f in self.file_metadata if f not in seen_files]
        for file_path in files_to_remove:
            self.remove_file(file_path)

        return all_entries, added_count, updated_count, unchanged_count

    def remove_file(self, file_path: str) -> None:
        """Remove all entries for a file from the index. O(k) where k = entries in file."""
        if file_path not in self.file_entries:
            return

        for entry_id in self.file_entries[file_path]:
            entry = self.entries.get(entry_id)
            if entry:
                is_block = entry.code_unit.unit_type == "block"

                if is_block:
                    # O(1) removal from block bucket
                    if entry.wl_hash in self.block_buckets:
                        self.block_buckets[entry.wl_hash].discard(entry_id)
                        if not self.block_buckets[entry.wl_hash]:
                            del self.block_buckets[entry.wl_hash]
                    # O(1) removal from block type index
                    block_type = entry.code_unit.block_type
                    if block_type and block_type in self.block_type_index:
                        self.block_type_index[block_type].discard(entry_id)
                        if not self.block_type_index[block_type]:
                            del self.block_type_index[block_type]
                    self._block_entry_count -= 1
                else:
                    # O(1) removal from hash bucket
                    if entry.wl_hash in self.hash_buckets:
                        self.hash_buckets[entry.wl_hash].discard(entry_id)
                        if not self.hash_buckets[entry.wl_hash]:
                            del self.hash_buckets[entry.wl_hash]
                    # O(1) removal from pattern bucket
                    if entry.pattern_hash in self.pattern_buckets:
                        self.pattern_buckets[entry.pattern_hash].discard(entry_id)
                        if not self.pattern_buckets[entry.pattern_hash]:
                            del self.pattern_buckets[entry.pattern_hash]
                    self._function_entry_count -= 1

                # O(1) removal from fingerprint index (skip empty graphs)
                if "n_nodes" in entry.fingerprint:
                    fp_key = (entry.fingerprint["n_nodes"], entry.fingerprint["n_edges"])
                    if fp_key in self.fingerprint_index:
                        self.fingerprint_index[fp_key].discard(entry_id)
                        if not self.fingerprint_index[fp_key]:
                            del self.fingerprint_index[fp_key]

                # Remove entry
                del self.entries[entry_id]

        del self.file_entries[file_path]

        # Remove file metadata
        self.file_metadata.pop(file_path, None)

    def find_exact_matches(self, code: str) -> list[IndexEntry]:
        """Find entries with the same WL hash as the given code."""
        graph = ast_to_graph(code)
        wl_hash = weisfeiler_leman_hash(graph)

        if wl_hash not in self.hash_buckets:
            return []

        return [self.entries[eid] for eid in self.hash_buckets[wl_hash] if eid in self.entries]

    def find_similar(self, code: str, min_node_count: int = 5) -> list[SimilarityResult]:
        """
        Find entries similar to the given code.

        Returns results sorted by similarity (exact > high > partial).
        Uses fingerprint index for O(1) candidate lookup instead of O(n) linear scan.
        """
        graph = ast_to_graph(code)
        if graph.number_of_nodes() < min_node_count:
            return []

        wl_hash = weisfeiler_leman_hash(graph)
        fp = structural_fingerprint(graph)
        hierarchy = list(compute_hierarchy_hash(graph))

        results: list[SimilarityResult] = []
        seen_ids: set[str] = set()

        # Check for exact matches first - O(1) bucket lookup
        if wl_hash in self.hash_buckets:
            for eid in self.hash_buckets[wl_hash]:
                entry = self.entries.get(eid)
                if entry:
                    results.append(SimilarityResult(entry=entry, similarity_type="exact"))
                    seen_ids.add(eid)

        # Check for high similarity using fingerprint index - O(1) lookup
        # Fingerprints must have same n_nodes and n_edges to be compatible
        fp_key = (fp["n_nodes"], fp["n_edges"])
        if fp_key in self.fingerprint_index:
            for eid in self.fingerprint_index[fp_key]:
                if eid in seen_ids:
                    continue
                entry = self.entries.get(eid)
                if entry and fingerprints_compatible(fp, entry.fingerprint):
                    results.append(SimilarityResult(entry=entry, similarity_type="high"))
                    seen_ids.add(eid)

        # Check for partial matches via hierarchy hashes - still O(n) but only for
        # entries not already matched. Consider hierarchy index for further optimization.
        for entry in self.entries.values():
            if entry.id in seen_ids:
                continue

            # Check for partial matches via hierarchy hashes
            matching_depth = 0
            for i, (h1, h2) in enumerate(zip(hierarchy, entry.hierarchy_hashes, strict=False)):
                if h1 == h2:
                    matching_depth = i + 1
                else:
                    break

            if matching_depth >= 2:  # At least 2 levels of structural match
                results.append(
                    SimilarityResult(
                        entry=entry, similarity_type="partial", matching_depth=matching_depth
                    )
                )

        # Sort: exact first, then high, then partial (by depth)
        def sort_key(r: SimilarityResult) -> tuple:
            type_order = {"exact": 0, "high": 1, "partial": 2}
            return (type_order[r.similarity_type], -(r.matching_depth or 0))

        results.sort(key=sort_key)
        return results

    def _find_duplicates_in_buckets(
        self,
        buckets: dict[str, set[str]],
        min_node_count: int = 5,
        entry_filter: Callable | None = None,
    ) -> list[DuplicateGroup]:
        """
        Find duplicate groups from hash buckets.

        Args:
            buckets: Hash buckets to search (hash -> {entry_ids})
            min_node_count: Minimum AST node count to include
            entry_filter: Optional filter function for entries

        Returns:
            List of DuplicateGroup objects, sorted by size (largest first).

        Complexity: O(b * k) where b = unique hashes, k = avg bucket size
        """
        groups: list[DuplicateGroup] = []

        for wl_hash, entry_ids in buckets.items():
            # O(1) suppression check
            if wl_hash in self.suppressed_hashes:
                continue

            if len(entry_ids) < 2:
                continue

            # Single-pass filtering: combine all filters into one iteration
            entries: list[IndexEntry] = []
            for eid in entry_ids:
                entry = self.entries.get(eid)
                if entry is None:
                    continue
                if entry.node_count < min_node_count:
                    continue
                if entry_filter and not entry_filter(entry):
                    continue
                entries.append(entry)

            if len(entries) >= 2:
                groups.append(DuplicateGroup(wl_hash=wl_hash, entries=entries))

        groups.sort(key=lambda g: len(g.entries), reverse=True)
        return groups

    def _get_entries_for_hash(self, wl_hash: str) -> list[IndexEntry]:
        """Get all entries matching a hash from any bucket."""
        entry_ids: set[str] = (
            self.hash_buckets.get(wl_hash, set())
            or self.pattern_buckets.get(wl_hash, set())
            or self.block_buckets.get(wl_hash, set())
        )
        return [e for eid in entry_ids if (e := self.entries.get(eid)) is not None]

    def _find_linked_hashes(self, entries: list[IndexEntry]) -> set[str]:
        """
        Find duplicate hashes for entries contained within the given entries.

        When a function is suppressed, any blocks (if/for/while/with/try) nested
        inside it should also be suppressed since they're part of the same code.

        Args:
            entries: The entries being suppressed

        Returns:
            Set of WL hashes for linked (contained) duplicates
        """
        linked_hashes: set[str] = set()

        # Build lookup of suppressed regions by file
        regions_by_file: dict[str, list[tuple[int, int]]] = {}
        for entry in entries:
            fp = entry.code_unit.file_path
            if fp not in regions_by_file:
                regions_by_file[fp] = []
            regions_by_file[fp].append((entry.code_unit.line_start, entry.code_unit.line_end))

        # Find entries contained within the suppressed regions
        for other_entry in self.entries.values():
            fp = other_entry.code_unit.file_path
            if fp not in regions_by_file:
                continue

            # Check if this entry is contained within any suppressed region
            for region_start, region_end in regions_by_file[fp]:
                if (
                    other_entry.code_unit.line_start >= region_start
                    and other_entry.code_unit.line_end <= region_end
                ):
                    linked_hashes.add(other_entry.wl_hash)
                    break

        return linked_hashes

    def suppress(self, wl_hash: str, reason: str | None = None) -> tuple[bool, list[str]]:
        """
        Suppress a duplicate group by its hash, including linked duplicates.

        Suppressed groups are excluded from analysis results.
        Use this for idiomatic patterns or acceptable duplications.

        When suppressing a function/method, any nested blocks (if/for/while/with/try)
        that are also detected as duplicates will be automatically suppressed.

        The suppression tracks source files and their content hashes. If the
        suppressed code structure no longer exists (not just file modifications),
        the suppression is automatically invalidated on the next tool interaction.

        Args:
            wl_hash: The WL hash of the duplicate group to suppress
            reason: Optional reason for suppression (for documentation)

        Returns:
            Tuple of (success, linked_hashes) where linked_hashes are additionally
            suppressed hashes that were contained within the suppressed entries.
        """
        entry_ids: set[str] = (
            self.hash_buckets.get(wl_hash, set())
            or self.pattern_buckets.get(wl_hash, set())
            or self.block_buckets.get(wl_hash, set())
        )

        if not entry_ids:
            return False, []

        self.suppressed_hashes.add(wl_hash)

        entries: list[IndexEntry] = [
            e for eid in entry_ids if (e := self.entries.get(eid)) is not None
        ]

        # Find and suppress linked duplicates (nested blocks)
        linked_hashes = self._find_linked_hashes(entries)
        linked_hashes.discard(wl_hash)  # Don't include the original hash
        linked_hashes -= self.suppressed_hashes  # Only newly suppressed ones

        newly_suppressed: list[str] = []
        created_at = time.time()
        for linked_hash in linked_hashes:
            if linked_hash not in self.suppressed_hashes:
                self.suppressed_hashes.add(linked_hash)
                newly_suppressed.append(linked_hash)
                # Create SuppressionInfo for linked hash so it can be persisted
                linked_entries = self._get_entries_for_hash(linked_hash)
                linked_first = linked_entries[0] if linked_entries else None
                self.suppression_details[linked_hash] = SuppressionInfo(
                    wl_hash=linked_hash,
                    reason=f"Linked to {wl_hash}",
                    created_at=created_at,
                    source_name=linked_first.code_unit.name if linked_first else None,
                    code_preview=linked_first.code_unit.code[:200] if linked_first else None,
                    entry_count=len(linked_entries),
                    source_files=[],
                    file_hashes={},
                )

        # Capture source files and their content hashes for structural change detection
        source_files: list[str] = []
        file_hashes: dict[str, str] = {}
        for entry in entries:
            fp = entry.code_unit.file_path
            if fp not in file_hashes:
                source_files.append(fp)
                file_hashes[fp] = (
                    self.file_metadata[fp].content_hash
                    if fp in self.file_metadata
                    else self._compute_file_hash(fp) or ""
                )

        # Capture display info from first entry
        first_entry = entries[0] if entries else None
        self.suppression_details[wl_hash] = SuppressionInfo(
            wl_hash=wl_hash,
            reason=reason,
            created_at=time.time(),
            source_name=first_entry.code_unit.name if first_entry else None,
            code_preview=first_entry.code_unit.code[:200] if first_entry else None,
            entry_count=len(entry_ids),
            source_files=source_files,
            file_hashes=file_hashes,
        )

        self._auto_save()
        return True, newly_suppressed

    def unsuppress(self, wl_hash: str) -> bool:
        """Remove a hash from the suppressed set."""
        if wl_hash in self.suppressed_hashes:
            self.suppressed_hashes.remove(wl_hash)
            self.suppression_details.pop(wl_hash, None)
            self._auto_save()
            return True
        return False

    def get_suppressed(self) -> list[str]:
        """Get list of suppressed hashes."""
        return list(self.suppressed_hashes)

    def get_suppression_info(self, wl_hash: str) -> SuppressionInfo | None:
        """Get detailed info about a suppressed hash."""
        return self.suppression_details.get(wl_hash)

    def invalidate_stale_suppressions(self) -> list[tuple[str, str]]:
        """
        Invalidate suppressions that are no longer valid.

        Two invalidation scenarios:
        1. Orphaned: Hash no longer exists in any bucket (after re-index)
        2. Structural change: Code structure modified in source files

        Only invalidates if the suppressed code pattern no longer exists.
        File modifications that don't affect structure are ignored.

        Returns:
            List of (wl_hash, reason) tuples for invalidated suppressions.
        """
        invalidated: list[tuple[str, str]] = []

        for wl_hash in list(self.suppressed_hashes):
            # Check 1: Orphaned suppression (hash not in any bucket)
            hash_exists_in_buckets = (
                wl_hash in self.hash_buckets
                or wl_hash in self.pattern_buckets
                or wl_hash in self.block_buckets
            )
            if not hash_exists_in_buckets:
                self._remove_suppression(wl_hash)
                invalidated.append((wl_hash, "hash no longer exists in index"))
                continue

            # Check 2: Structural change in source files
            info = self.suppression_details.get(wl_hash)
            if not info or not info.file_hashes:
                continue

            deleted_files, files_changed = self._detect_file_changes(info)
            if not files_changed:
                continue

            if self._structure_still_exists(wl_hash, info, deleted_files):
                self._update_suppression_tracking(info, deleted_files)
            else:
                self._remove_suppression(wl_hash)
                invalidated.append((wl_hash, "suppressed code structure no longer exists"))

        return invalidated

    def _remove_suppression(self, wl_hash: str) -> None:
        """Remove a suppression and its details."""
        self.suppressed_hashes.discard(wl_hash)
        self.suppression_details.pop(wl_hash, None)

    def _detect_file_changes(self, info: SuppressionInfo) -> tuple[list[str], bool]:
        """Detect which files have changed or been deleted."""
        deleted_files: list[str] = []
        files_changed = False

        for file_path, stored_hash in info.file_hashes.items():
            if not os.path.exists(file_path):
                deleted_files.append(file_path)
                files_changed = True
            else:
                current_hash = self._compute_file_hash(file_path)
                if current_hash and current_hash != stored_hash:
                    files_changed = True

        return deleted_files, files_changed

    def _structure_still_exists(
        self, wl_hash: str, info: SuppressionInfo, deleted_files: list[str]
    ) -> bool:
        """Check if the suppressed code structure still exists in source files."""
        from .ast_to_graph import code_unit_to_ast_graph, extract_code_units
        from .canonical_hash import weisfeiler_leman_hash

        for file_path in info.source_files:
            if file_path in deleted_files or not os.path.exists(file_path):
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    source = f.read()

                for unit in extract_code_units(source, file_path, include_blocks=True):
                    ast_graph = code_unit_to_ast_graph(unit)
                    if weisfeiler_leman_hash(ast_graph.graph) == wl_hash:
                        return True
            except (OSError, SyntaxError, UnicodeDecodeError):
                continue  # File unreadable or unparseable

        return False

    def _update_suppression_tracking(self, info: SuppressionInfo, deleted_files: list[str]) -> None:
        """Update suppression tracking after files changed but structure remains."""
        # Update hashes for existing files
        for file_path in info.source_files:
            if os.path.exists(file_path):
                new_hash = self._compute_file_hash(file_path)
                if new_hash:
                    info.file_hashes[file_path] = new_hash

        # Remove deleted files from tracking
        for deleted in deleted_files:
            if deleted in info.source_files:
                info.source_files.remove(deleted)
            info.file_hashes.pop(deleted, None)

    def find_all_duplicates(self, min_node_count: int = 5) -> list[DuplicateGroup]:
        """Find all groups of structurally equivalent code units."""
        return self._find_duplicates_in_buckets(self.hash_buckets, min_node_count)

    def find_pattern_duplicates(self, min_node_count: int = 5) -> list[DuplicateGroup]:
        """
        Find groups of code with same pattern but different operators.

        Pattern duplicates have the same control flow structure but may differ in:
        - Binary operators (+, -, *, /, etc.)
        - Comparison operators (==, !=, <, >, etc.)
        - Boolean operators (and, or, not)

        This catches "same pattern, different operations" that exact matching misses.
        Excludes groups that are already exact duplicates.
        Respects suppressions (suppressed pattern hashes are excluded).
        """
        # Get candidate groups using shared logic
        groups = self._find_duplicates_in_buckets(self.pattern_buckets, min_node_count)

        # Exclude groups where ALL entries are exact duplicates of each other
        exact_hashes = {h for h, ids in self.hash_buckets.items() if len(ids) >= 2}

        return [g for g in groups if not self._group_is_exact_duplicate(g, exact_hashes)]

    def _group_is_exact_duplicate(self, group: DuplicateGroup, exact_hashes: set[str]) -> bool:
        """Check if all entries in a group share the same exact-duplicate wl_hash."""
        unique_wl_hashes = {e.wl_hash for e in group.entries}
        return len(unique_wl_hashes) == 1 and unique_wl_hashes.pop() in exact_hashes

    def find_block_duplicates(
        self,
        min_node_count: int = 5,
        block_types: list[str] | None = None,
    ) -> list[DuplicateGroup]:
        """
        Find groups of structurally equivalent code blocks.

        Args:
            min_node_count: Minimum AST node count to include (default 5)
            block_types: Optional list of block types to include (e.g., ['for', 'if']).
                        If None, includes all block types.

        Returns:
            List of DuplicateGroup objects for duplicate blocks.
        """
        entry_filter = None
        if block_types:
            valid_block_types = set(BLOCK_TYPE_NAMES.values())
            block_types_set = set(block_types) & valid_block_types

            def block_type_filter(e: IndexEntry) -> bool:
                return e.code_unit.block_type in block_types_set

            entry_filter = block_type_filter

        return self._find_duplicates_in_buckets(self.block_buckets, min_node_count, entry_filter)

    def verify_isomorphism(self, entry1: IndexEntry, entry2: IndexEntry) -> bool:
        """Verify that two entries are truly isomorphic using full graph isomorphism."""
        g1 = ast_to_graph(entry1.code_unit.code)
        g2 = ast_to_graph(entry2.code_unit.code)

        return bool(nx.is_isomorphic(g1, g2, node_match=node_match))

    def get_stats(self) -> dict:
        """Get statistics about the index. O(1) using incremental counters."""
        return {
            "total_entries": len(self.entries),
            "function_entries": self._function_entry_count,
            "block_entries": self._block_entry_count,
            "unique_hashes": len(self.hash_buckets),
            "unique_patterns": len(self.pattern_buckets),
            "unique_block_hashes": len(self.block_buckets),
            "indexed_files": len(self.file_entries),
            "duplicate_groups": sum(1 for ids in self.hash_buckets.values() if len(ids) > 1),
            "pattern_groups": sum(1 for ids in self.pattern_buckets.values() if len(ids) > 1),
            "block_duplicate_groups": sum(1 for ids in self.block_buckets.values() if len(ids) > 1),
            "suppressed_hashes": len(self.suppressed_hashes),
        }

    def save(self, path: str) -> None:
        """Save the index to a JSON file."""
        data = {
            "version": 3,  # Index format version for migration support
            "entries": {eid: e.to_dict() for eid, e in self.entries.items()},
            # Convert sets to lists for JSON serialization
            "hash_buckets": {k: list(v) for k, v in self.hash_buckets.items()},
            "pattern_buckets": {k: list(v) for k, v in self.pattern_buckets.items()},
            "block_buckets": {k: list(v) for k, v in self.block_buckets.items()},
            "block_type_index": {k: list(v) for k, v in self.block_type_index.items()},
            "file_entries": self.file_entries,
            "suppressed_hashes": list(self.suppressed_hashes),
            "file_metadata": {fp: fm.to_dict() for fp, fm in self.file_metadata.items()},
            "suppression_details": {h: si.to_dict() for h, si in self.suppression_details.items()},
            "entry_counter": self._entry_counter,
            "block_entry_count": self._block_entry_count,
            "function_entry_count": self._function_entry_count,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load the index from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # version = data.get("version", 1)  # Reserved for future format migrations

        self.entries = {eid: IndexEntry.from_dict(e) for eid, e in data["entries"].items()}
        # Convert lists back to sets for O(1) operations
        self.hash_buckets = {k: set(v) for k, v in data["hash_buckets"].items()}
        self.pattern_buckets = {k: set(v) for k, v in data.get("pattern_buckets", {}).items()}
        self.block_buckets = {k: set(v) for k, v in data.get("block_buckets", {}).items()}
        self.block_type_index = {k: set(v) for k, v in data.get("block_type_index", {}).items()}
        self.file_entries = data["file_entries"]
        self.suppressed_hashes = set(data.get("suppressed_hashes", []))
        self._entry_counter = data["entry_counter"]

        # Load counters or rebuild them for backwards compatibility
        if "block_entry_count" in data:
            self._block_entry_count = data["block_entry_count"]
            self._function_entry_count = data["function_entry_count"]
        else:
            # Rebuild counters from entries (backwards compatibility)
            self._block_entry_count = sum(
                1 for e in self.entries.values() if e.code_unit.unit_type == "block"
            )
            self._function_entry_count = len(self.entries) - self._block_entry_count

        # Rebuild fingerprint index from entries (skip empty graphs)
        self.fingerprint_index = {}
        for eid, entry in self.entries.items():
            if "n_nodes" in entry.fingerprint:
                fp_key = (entry.fingerprint["n_nodes"], entry.fingerprint["n_edges"])
                self.fingerprint_index.setdefault(fp_key, set()).add(eid)

        self.file_metadata = {
            fp: FileMetadata.from_dict(fm) for fp, fm in data.get("file_metadata", {}).items()
        }
        self.suppression_details = {
            h: SuppressionInfo.from_dict(si)
            for h, si in data.get("suppression_details", {}).items()
        }

    def clear(self) -> None:
        """Clear all entries from the index (preserves suppressions)."""
        self.entries.clear()
        self.hash_buckets.clear()
        self.pattern_buckets.clear()
        self.block_buckets.clear()
        self.block_type_index.clear()
        self.fingerprint_index.clear()
        self.file_entries.clear()
        self.file_metadata.clear()
        self._entry_counter = 0
        self._block_entry_count = 0
        self._function_entry_count = 0
        # Note: suppressed_hashes and suppression_details are preserved across clears

    def clear_suppressions(self) -> None:
        """Clear all suppressed hashes and their details."""
        self.suppressed_hashes.clear()
        self.suppression_details.clear()

    def get_staleness_report(self, root_path: str | None = None) -> StalenessReport:
        """
        Check if the index is stale relative to the filesystem.

        Args:
            root_path: Optional root path to also check for new files.
                      If provided, scans for Python files not in the index.

        Returns:
            StalenessReport with lists of modified, deleted, and new files.
        """
        modified_files: list[str] = []
        deleted_files: list[str] = []
        new_files: list[str] = []
        stale_suppressions: list[str] = []

        # Check tracked files for modifications and deletions
        for file_path in self.file_metadata:
            if not os.path.exists(file_path):
                deleted_files.append(file_path)
            elif self.check_file_changed(file_path):
                modified_files.append(file_path)

        # Check for new files if root_path is provided
        if root_path and os.path.isdir(root_path):
            for py_file in Path(root_path).glob("**/*.py"):
                if _should_skip_path(py_file.parts):
                    continue
                file_str = str(py_file)
                if file_str not in self.file_metadata:
                    new_files.append(file_str)

        # Check suppression staleness
        stale_suppressions = self.check_suppression_staleness()

        is_stale = bool(modified_files or deleted_files or new_files or stale_suppressions)

        return StalenessReport(
            is_stale=is_stale,
            modified_files=modified_files,
            deleted_files=deleted_files,
            new_files=new_files,
            stale_suppressions=stale_suppressions,
        )

    def check_suppression_staleness(self) -> list[str]:
        """
        Check if any suppressed hashes reference code that no longer exists.

        Returns list of stale suppression hashes with reasons.
        """
        stale: list[str] = []

        for wl_hash, info in self.suppression_details.items():
            # Check if the hash still exists in any bucket
            exists = (
                wl_hash in self.hash_buckets
                or wl_hash in self.pattern_buckets
                or wl_hash in self.block_buckets
            )

            if not exists:
                stale.append(f"{wl_hash}: code no longer exists in index")
            elif info.primary_file and info.primary_file not in self.file_entries:
                stale.append(f"{wl_hash}: source file {info.primary_file} removed from index")

        return stale
