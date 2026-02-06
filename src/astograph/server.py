"""
MCP server for code structure analysis.

Provides 11 tools (all prefixed with astograph_):
- astograph_index: Index Python files
- astograph_analyze: Find duplicates and similar patterns
- astograph_check: Check if code exists before creating
- astograph_compare: Compare two code snippets
- astograph_suppress: Suppress a duplicate group by hash
- astograph_unsuppress: Remove suppression from a hash
- astograph_list_suppressions: List all suppressed hashes
- astograph_suppress_idiomatic: Suppress all idiomatic patterns at once
- astograph_check_staleness: Check if index is stale
- astograph_write: Write Python file with duplicate detection (blocks if duplicate exists)
- astograph_edit: Edit Python file with duplicate detection (blocks if duplicate exists)
"""

import asyncio
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .tools import CodeStructureTools

# Check for event-driven mode via environment variable
_event_driven = os.environ.get("ASTOGRAPH_EVENT_DRIVEN", "").lower() in ("1", "true", "yes")

# Global tools instance
_tools = CodeStructureTools(event_driven=_event_driven)


def get_tools() -> CodeStructureTools:
    """Get the global tools instance."""
    return _tools


def set_tools(tools: CodeStructureTools) -> None:
    """Set the global tools instance (for testing)."""
    global _tools
    _tools = tools


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("code-structure-mcp")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="astograph_index",
                description=(
                    "Index a Python codebase for structural analysis. "
                    "Parses all .py files and builds a searchable index. "
                    "Currently supports Python only. "
                    "Note: The codebase is auto-indexed at startup."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the directory or file to index",
                        },
                        "incremental": {
                            "type": "boolean",
                            "description": (
                                "If true, only re-index files that have changed since last indexing. "
                                "Faster for large codebases with few changes. Default: true"
                            ),
                            "default": True,
                        },
                    },
                    "required": ["path"],
                },
            ),
            Tool(
                name="astograph_analyze",
                description=(
                    "Analyze the indexed Python codebase for duplicate functions, methods, and "
                    "code blocks (for/while/if/try/with). Returns exact duplicates verified via "
                    "graph isomorphism."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "thorough": {
                            "type": "boolean",
                            "description": (
                                "If true, show ALL duplicates including small ones (~2+ lines). "
                                "If false, show only significant duplicates (~6+ lines). "
                                "Default: true"
                            ),
                            "default": True,
                        },
                        "auto_reindex": {
                            "type": "boolean",
                            "description": (
                                "If true and index is stale, automatically re-index before analyzing. "
                                "Default: true"
                            ),
                            "default": True,
                        },
                    },
                },
            ),
            Tool(
                name="astograph_check",
                description=(
                    "Check if Python code similar to a snippet already exists in the indexed codebase. "
                    "Use this BEFORE creating new Python code to avoid duplication. "
                    "Only works with Python syntax."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code snippet to check for existing similar code",
                        },
                    },
                    "required": ["code"],
                },
            ),
            Tool(
                name="astograph_compare",
                description=(
                    "Compare two Python code snippets for structural equivalence. "
                    "Returns whether they are identical, similar, or different. "
                    "Only works with Python syntax."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code1": {
                            "type": "string",
                            "description": "First Python code snippet",
                        },
                        "code2": {
                            "type": "string",
                            "description": "Second Python code snippet",
                        },
                    },
                    "required": ["code1", "code2"],
                },
            ),
            Tool(
                name="astograph_suppress",
                description=(
                    "Suppress a duplicate group by its WL hash. "
                    "Use this to mute idiomatic patterns or acceptable duplications "
                    "that don't need to be refactored. The hash is shown in astograph_analyze output."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wl_hash": {
                            "type": "string",
                            "description": "The WL hash of the duplicate group to suppress",
                        },
                    },
                    "required": ["wl_hash"],
                },
            ),
            Tool(
                name="astograph_unsuppress",
                description=(
                    "Remove suppression from a hash, making it appear in astograph_analyze results again."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wl_hash": {
                            "type": "string",
                            "description": "The WL hash to unsuppress",
                        },
                    },
                    "required": ["wl_hash"],
                },
            ),
            Tool(
                name="astograph_list_suppressions",
                description="List all currently suppressed hashes.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="astograph_suppress_idiomatic",
                description=(
                    "Suppress ALL idiomatic patterns in one call. "
                    "Convenience method to quickly suppress all patterns classified as idiomatic "
                    "(guard clauses, test setup, delegate methods, dict building, etc.). "
                    "Use instead of calling suppress() for each idiomatic pattern."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="astograph_check_staleness",
                description=(
                    "Check if the code index is stale (files changed since indexing). "
                    "Returns details about modified, deleted, and new files."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Optional root path to also check for new files",
                        },
                    },
                },
            ),
            Tool(
                name="astograph_write",
                description=(
                    "Write Python code to a file with automatic duplicate detection. "
                    "Checks the content for structural duplicates before writing. "
                    "BLOCKS if identical code exists elsewhere (returns existing location). "
                    "WARNS on high similarity but proceeds with write."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "The Python code content to write",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            ),
            Tool(
                name="astograph_edit",
                description=(
                    "Edit a Python file with automatic duplicate detection. "
                    "Checks the new_string for structural duplicates before applying. "
                    "BLOCKS if identical code exists elsewhere (returns existing location). "
                    "WARNS on high similarity but proceeds with edit."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to the file to edit",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "The exact text to replace (must be unique in file)",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "The replacement Python code",
                        },
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            ),
        ]

    # Map external tool names to internal names
    TOOL_NAME_MAP = {
        "astograph_index": "index_codebase",
        "astograph_analyze": "analyze",
        "astograph_check": "check",
        "astograph_compare": "compare",
        "astograph_suppress": "suppress",
        "astograph_unsuppress": "unsuppress",
        "astograph_list_suppressions": "list_suppressions",
        "astograph_suppress_idiomatic": "suppress_idiomatic",
        "astograph_check_staleness": "check_staleness",
        "astograph_write": "write",
        "astograph_edit": "edit",
    }

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        internal_name = TOOL_NAME_MAP.get(name, name)
        result = _tools.call_tool(internal_name, arguments)
        return [TextContent(type="text", text=result.text)]

    return server


async def run_server() -> None:
    """Run the MCP server."""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Entry point for the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
