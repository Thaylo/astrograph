"""
MCP server for code structure analysis.

Auto-indexes the codebase at startup and maintains the index via file watching.

Provides 11 tools (all prefixed with astrograph_):
- astrograph_analyze: Find duplicates and similar patterns
- astrograph_write: Write file with duplicate detection (blocks if duplicate exists)
- astrograph_edit: Edit file with duplicate detection (blocks if duplicate exists)
- astrograph_suppress: Suppress a duplicate group by hash
- astrograph_suppress_batch: Suppress multiple duplicates by hash list
- astrograph_unsuppress: Remove suppression from a hash
- astrograph_unsuppress_batch: Remove suppression from multiple hashes
- astrograph_list_suppressions: List all suppressed hashes
- astrograph_status: Check server readiness (returns instantly even during indexing)
- astrograph_metadata_erase: Erase all persisted metadata
- astrograph_metadata_recompute_baseline: Erase metadata and re-index from scratch
"""

import asyncio
import atexit
import signal
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .tools import CodeStructureTools

# Global tools instance
_tools = CodeStructureTools()


def get_tools() -> CodeStructureTools:
    """Get the global tools instance."""
    return _tools


def set_tools(tools: CodeStructureTools) -> None:
    """Set the global tools instance (for testing)."""
    global _tools
    _tools = tools


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server(
        "code-structure-mcp",
        instructions=(
            "ASTrograph indexes the codebase in the background at startup. "
            "If the first tool call is slow, indexing is still in progress. "
            "Use astrograph_status to check readiness."
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="astrograph_analyze",
                description="Find duplicate code (verified via graph isomorphism).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "auto_reindex": {
                            "type": "boolean",
                            "description": "Auto re-index if stale (default: true)",
                            "default": True,
                        },
                    },
                },
            ),
            Tool(
                name="astrograph_suppress",
                description="Suppress a duplicate by WL hash (from analyze output).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wl_hash": {
                            "type": "string",
                            "description": "WL hash from analyze output",
                        },
                    },
                    "required": ["wl_hash"],
                },
            ),
            Tool(
                name="astrograph_suppress_batch",
                description="Suppress multiple duplicates by WL hash list.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wl_hashes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "WL hashes from analyze output",
                        },
                    },
                    "required": ["wl_hashes"],
                },
            ),
            Tool(
                name="astrograph_unsuppress",
                description="Unsuppress a hash.",
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
                name="astrograph_unsuppress_batch",
                description="Unsuppress multiple hashes.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wl_hashes": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "WL hashes to unsuppress",
                        },
                    },
                    "required": ["wl_hashes"],
                },
            ),
            Tool(
                name="astrograph_list_suppressions",
                description="List suppressed hashes.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="astrograph_status",
                description="Check server readiness. Returns instantly even during indexing.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="astrograph_metadata_erase",
                description="Erase all persisted metadata (.metadata_astrograph/). Resets server to idle.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="astrograph_metadata_recompute_baseline",
                description="Erase metadata and re-index the codebase from scratch.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="astrograph_write",
                description="Write file. Blocks if duplicate exists, warns on similarity.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute file path",
                        },
                        "content": {
                            "type": "string",
                            "description": "Code to write",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            ),
            Tool(
                name="astrograph_edit",
                description="Edit file. Blocks if new code duplicates existing, warns on similarity.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute file path",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Exact text to replace (must be unique)",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "Replacement code",
                        },
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            ),
        ]

    # Map external tool names to internal names
    TOOL_NAME_MAP = {
        "astrograph_analyze": "analyze",
        "astrograph_write": "write",
        "astrograph_edit": "edit",
        "astrograph_suppress": "suppress",
        "astrograph_suppress_batch": "suppress_batch",
        "astrograph_unsuppress": "unsuppress",
        "astrograph_unsuppress_batch": "unsuppress_batch",
        "astrograph_list_suppressions": "list_suppressions",
        "astrograph_status": "status",
        "astrograph_metadata_erase": "metadata_erase",
        "astrograph_metadata_recompute_baseline": "metadata_recompute_baseline",
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


def _shutdown_handler(_signum: int, _frame: object) -> None:
    """Handle SIGTERM from Docker by flushing and closing resources."""
    _tools.close()
    sys.exit(0)


def main() -> None:
    """Entry point for the MCP server."""
    signal.signal(signal.SIGTERM, _shutdown_handler)
    atexit.register(_tools.close)
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
