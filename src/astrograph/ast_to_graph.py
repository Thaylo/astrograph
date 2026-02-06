"""
Backward-compatibility facade.

All logic has been moved to the languages/ package.
This module re-exports everything for existing imports.
"""

# Re-export data structures from languages.base
from .languages.base import ASTGraph, CodeUnit, compute_label_histogram, node_match

# Re-export Python-specific functions from languages.python_plugin
from .languages.python_plugin import (
    BLOCK_TYPE_NAMES,
    BLOCK_TYPES,
    _extract_blocks_recursive,
    _get_node_label,
    ast_to_graph,
    code_unit_to_ast_graph,
    extract_blocks_from_function,
    extract_code_units,
)

__all__ = [
    "ASTGraph",
    "BLOCK_TYPE_NAMES",
    "BLOCK_TYPES",
    "CodeUnit",
    "_extract_blocks_recursive",
    "_get_node_label",
    "ast_to_graph",
    "code_unit_to_ast_graph",
    "compute_label_histogram",
    "extract_blocks_from_function",
    "extract_code_units",
    "node_match",
]
