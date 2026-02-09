"""Shared brace-based block extraction for C-family languages (C, C++, Java).

These languages share brace-delimited control flow blocks that can be extracted
via regex keyword detection + brace matching, without a full AST parser.
"""

from __future__ import annotations

import re
import textwrap
from collections.abc import Iterator

from .base import CodeUnit

# Block-introducing keywords and their block type names.
_BLOCK_KEYWORDS: dict[str, str] = {
    "for": "for",
    "while": "while",
    "do": "do_while",
    "if": "if",
    "switch": "switch",
    "try": "try",
    "select": "select",
    "defer": "defer",
}

# Regex to match a block-introducing keyword at the start of a statement.
_BLOCK_KEYWORD_RE = re.compile(r"\b(" + "|".join(_BLOCK_KEYWORDS) + r")\s*(?:[\({]|$)")

# Comments / strings to skip during brace matching.
_LINE_COMMENT_RE = re.compile(r"//.*$", re.MULTILINE)
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_RE = re.compile(r"'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\"")


def _strip_comments_and_strings(source: str) -> str:
    """Remove comments and string literals to avoid false brace matches."""
    result = _STRING_RE.sub("STR", source)
    result = _BLOCK_COMMENT_RE.sub(" ", result)
    result = _LINE_COMMENT_RE.sub("", result)
    return result


def _find_matching_brace(text: str, open_pos: int) -> int | None:
    """Find the matching closing brace for an opening brace at open_pos."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _line_of_offset(text: str, offset: int) -> int:
    """Return 1-based line number for a character offset."""
    return text[:offset].count("\n") + 1


def extract_brace_blocks_from_function(
    func_code: str,
    file_path: str,
    func_name: str,
    func_line_start: int,
    language: str,
    max_depth: int = 3,
) -> Iterator[CodeUnit]:
    """Extract nested brace-delimited blocks from a function body.

    Parameters
    ----------
    func_code:
        The source code of the function (including signature and braces).
    file_path:
        Path to the source file.
    func_name:
        Name of the enclosing function.
    func_line_start:
        1-based line number where the function starts in the file.
    language:
        Language identifier (e.g. "c_lsp", "cpp_lsp", "java_lsp").
    max_depth:
        Maximum nesting depth for block extraction.
    """
    cleaned = _strip_comments_and_strings(func_code)
    source_lines = func_code.splitlines()

    def _extract(
        cleaned_text: str,
        base_offset: int,
        parent_block_name: str,
        current_depth: int,
        block_counters: dict[str, int],
    ) -> Iterator[CodeUnit]:
        if current_depth > max_depth:
            return

        # Track consumed ranges to avoid double-counting nested blocks
        skip_until = 0
        for match in _BLOCK_KEYWORD_RE.finditer(cleaned_text):
            keyword_pos = match.start()
            if keyword_pos < skip_until:
                continue

            keyword = match.group(1)
            block_type = _BLOCK_KEYWORDS[keyword]

            # Find the opening brace after the keyword
            brace_pos = cleaned_text.find("{", keyword_pos)
            if brace_pos == -1:
                continue

            # Find matching closing brace
            close_pos = _find_matching_brace(cleaned_text, brace_pos)
            if close_pos is None:
                continue

            # Skip over this entire block for the outer iteration
            skip_until = close_pos + 1

            # Calculate line numbers relative to the function
            abs_start = base_offset + keyword_pos
            abs_end = base_offset + close_pos
            start_line_rel = _line_of_offset(func_code, abs_start)
            end_line_rel = _line_of_offset(func_code, abs_end)

            # Convert to file-level line numbers
            start_line = func_line_start + start_line_rel - 1
            end_line = func_line_start + end_line_rel - 1

            # Skip trivially small blocks (single-line)
            if start_line_rel == end_line_rel:
                continue

            # Build block name
            counter_key = f"{parent_block_name}.{block_type}"
            if counter_key not in block_counters:
                block_counters[counter_key] = 0
            block_counters[counter_key] += 1
            block_num = block_counters[counter_key]

            if parent_block_name == func_name:
                block_name = f"{func_name}.{block_type}_{block_num}"
            else:
                block_name = f"{parent_block_name}.{block_type}_{block_num}"

            # Extract code from function source lines
            code = textwrap.dedent("\n".join(source_lines[start_line_rel - 1 : end_line_rel]))

            yield CodeUnit(
                name=block_name,
                code=code,
                file_path=file_path,
                line_start=start_line,
                line_end=end_line,
                unit_type="block",
                parent_name=func_name,
                block_type=block_type,
                nesting_depth=current_depth,
                parent_block_name=(parent_block_name if parent_block_name != func_name else None),
                language=language,
            )

            # Recurse into the block body for nested blocks
            inner_start = brace_pos + 1
            inner_end = close_pos
            if inner_end > inner_start:
                yield from _extract(
                    cleaned_text[inner_start:inner_end],
                    base_offset + inner_start,
                    block_name,
                    current_depth + 1,
                    block_counters,
                )

    block_counters: dict[str, int] = {}
    yield from _extract(cleaned, 0, func_name, 1, block_counters)
