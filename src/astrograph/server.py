"""
MCP server for code structure analysis.

Auto-indexes the codebase at startup and maintains the index via file watching.

Provides 12 tools:
- analyze: Find duplicates and similar patterns
- write: Write file with duplicate detection (blocks if duplicate exists)
- edit: Edit file with duplicate detection (blocks if duplicate exists)
- suppress: Suppress one or more duplicates by WL hash (string or array)
- unsuppress: Unsuppress one or more hashes (string or array)
- list_suppressions: List all suppressed hashes
- status: Check server readiness (returns instantly even during indexing)
- lsp_setup: Inspect/bind LSP commands or attach endpoints for bundled language plugins
- metadata_erase: Erase all persisted metadata
- metadata_recompute_baseline: Erase metadata and re-index from scratch

Also exposes 3 MCP resources, 2 prompts, and prompt argument completions.
"""

import asyncio
import atexit
import os
import signal
import threading

from mcp.server import Server
from mcp.server.lowlevel.server import ReadResourceContents
from mcp.types import (
    AnyUrl,
    Completion,
    CompletionArgument,
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    PromptReference,
    Resource,
    ResourceTemplateReference,
    TextContent,
    Tool,
    ToolAnnotations,
)

from . import __version__
from .stdio_transport import dual_stdio_server
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
        version=__version__,
        instructions=(
            "ASTrograph indexes the codebase in the background at startup. "
            "If the first tool call is slow, indexing is still in progress. "
            "Use status to check readiness. "
            "When running inside Docker, use the same absolute host paths for set_workspace "
            "and file operations — they are accessible inside the container if the host "
            "filesystem is mounted (e.g. -v /Users:/Users). "
            "Before large refactors, call lsp_setup(mode='inspect') and execute "
            "recommended_actions (run host_search_commands, search/install/bind/auto_bind) "
            "until missing_languages is empty. Always attempt to resolve all languages, not just required ones."
        ),
    )

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="analyze",
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
                annotations=ToolAnnotations(
                    title="Find Duplicate Code",
                    readOnlyHint=True,
                    idempotentHint=True,
                ),
            ),
            Tool(
                name="suppress",
                description="Suppress a duplicate by WL hash (from analyze output). Only suppress AFTER confirming the duplication is intentional (e.g. API symmetry, test isolation). Refactoring should be the default action for duplicates.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wl_hash": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                            "description": "WL hash from analyze output",
                        },
                    },
                    "required": ["wl_hash"],
                },
                annotations=ToolAnnotations(
                    title="Suppress Duplicate",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                ),
            ),
            Tool(
                name="unsuppress",
                description="Unsuppress a hash.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "wl_hash": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                            "description": "The WL hash to unsuppress",
                        },
                    },
                    "required": ["wl_hash"],
                },
                annotations=ToolAnnotations(
                    title="Unsuppress Duplicate",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                ),
            ),
            Tool(
                name="list_suppressions",
                description="List suppressed hashes.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
                annotations=ToolAnnotations(
                    title="List Suppressions",
                    readOnlyHint=True,
                    idempotentHint=True,
                ),
            ),
            Tool(
                name="status",
                description="Check server readiness. Returns instantly even during indexing.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
                annotations=ToolAnnotations(
                    title="Server Status",
                    readOnlyHint=True,
                    idempotentHint=True,
                ),
            ),
            Tool(
                name="lsp_setup",
                description=(
                    "Inspect and configure deterministic LSP command bindings "
                    "for bundled language plugins. Returns a guided recommended_actions "
                    "plan for search/install/config workflows."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["inspect", "auto_bind", "bind", "unbind"],
                            "description": "Setup mode (default: inspect)",
                            "default": "inspect",
                        },
                        "language": {
                            "type": "string",
                            "description": (
                                "Language ID filter for inspect/auto_bind and required target "
                                "for bind/unbind (python, javascript_lsp, typescript_lsp, c_lsp, cpp_lsp, java_lsp, go_lsp)"
                            ),
                        },
                        "command": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                            "description": "LSP command for bind mode",
                        },
                        "observations": {
                            "type": "array",
                            "description": (
                                "Optional host-discovery hints used by auto_bind "
                                "(agent-provided search results such as commands/endpoints)."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "language": {"type": "string"},
                                    "command": {
                                        "oneOf": [
                                            {"type": "string"},
                                            {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        ]
                                    },
                                },
                                "required": ["language", "command"],
                            },
                        },
                        "validation_mode": {
                            "type": "string",
                            "enum": ["production", "bootstrap"],
                            "description": (
                                "Per-call validation strictness override "
                                "(default: production from env or config)"
                            ),
                        },
                        "compile_db_path": {
                            "type": "string",
                            "description": (
                                "Explicit compile_commands.json path for C/C++ "
                                "(overrides env and auto-discovery)"
                            ),
                        },
                        "project_root": {
                            "type": "string",
                            "description": (
                                "Project root for compile_commands.json search in monorepos "
                                "(narrows discovery scope)"
                            ),
                        },
                    },
                },
                annotations=ToolAnnotations(
                    title="Configure LSP",
                    readOnlyHint=False,
                    destructiveHint=False,
                    openWorldHint=True,
                ),
            ),
            Tool(
                name="metadata_erase",
                description="Erase all persisted metadata (.metadata_astrograph/). Resets server to idle.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
                annotations=ToolAnnotations(
                    title="Erase All Metadata",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=True,
                ),
            ),
            Tool(
                name="metadata_recompute_baseline",
                description="Erase metadata and re-index the codebase from scratch.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
                annotations=ToolAnnotations(
                    title="Recompute Index",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                ),
            ),
            Tool(
                name="generate_ignore",
                description="Auto-generate .astrographignore with reasonable defaults for excluding files from indexing.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
                annotations=ToolAnnotations(
                    title="Generate .astrographignore",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=True,
                ),
            ),
            Tool(
                name="set_workspace",
                description="Set or change the workspace directory. Re-indexes the codebase at the new path.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path to the new workspace directory",
                        },
                    },
                    "required": ["path"],
                },
                annotations=ToolAnnotations(
                    title="Set Workspace",
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                ),
            ),
            Tool(
                name="write",
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
                annotations=ToolAnnotations(
                    title="Write File",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                ),
            ),
            Tool(
                name="edit",
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
                annotations=ToolAnnotations(
                    title="Edit File",
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                ),
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        result = await asyncio.to_thread(_tools.call_tool, name, arguments)
        return [TextContent(type="text", text=result.text)]

    # --- MCP Resources ---

    _RESOURCE_URIS = {
        "astrograph://status": ("Server Status", "Current server readiness and statistics"),
        "astrograph://analysis/latest": (
            "Latest Analysis",
            "Most recent duplicate analysis report",
        ),
        "astrograph://suppressions": (
            "Suppressions",
            "Currently suppressed duplicate hashes",
        ),
    }

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        return [
            Resource(name=name, uri=uri, description=desc, mimeType="text/plain")
            for uri, (name, desc) in _RESOURCE_URIS.items()
        ]

    @server.list_resource_templates()
    async def list_resource_templates() -> list:
        return []

    @server.read_resource()
    async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        uri_str = str(uri)
        if uri_str == "astrograph://status":
            content = await asyncio.to_thread(_tools.read_resource_status)
            return [ReadResourceContents(content=content)]
        elif uri_str == "astrograph://analysis/latest":
            content = await asyncio.to_thread(_tools.read_resource_analysis)
            return [ReadResourceContents(content=content)]
        elif uri_str == "astrograph://suppressions":
            content = await asyncio.to_thread(_tools.read_resource_suppressions)
            return [ReadResourceContents(content=content)]
        else:
            raise ValueError(f"Unknown resource URI: {uri_str}")

    # --- MCP Prompts ---

    _AVAILABLE_PROMPTS = [
        Prompt(
            name="review-duplicates",
            description=(
                "Review duplicate code findings and decide whether to suppress, "
                "refactor, or skip each group."
            ),
            arguments=[
                PromptArgument(
                    name="focus",
                    description="Filter findings: all, source, or tests",
                    required=False,
                ),
            ],
        ),
        Prompt(
            name="setup-lsp",
            description="Guided workflow to install and configure LSP servers for language support.",
            arguments=[
                PromptArgument(
                    name="language",
                    description=(
                        "Target language (python, javascript_lsp, typescript_lsp, "
                        "c_lsp, cpp_lsp, java_lsp, go_lsp)"
                    ),
                    required=False,
                ),
            ],
        ),
    ]

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        return _AVAILABLE_PROMPTS

    def _build_review_duplicates_prompt(focus: str | None) -> GetPromptResult:
        analysis_text = _tools.read_resource_analysis()
        focus_label = focus or "all"
        return GetPromptResult(
            description=f"Review duplicates (focus: {focus_label})",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Here is the latest ASTrograph duplicate analysis report "
                            f"(focus: {focus_label}):\n\n{analysis_text}\n\n"
                            "For each duplicate group, decide one of:\n"
                            "- REFACTOR (preferred): extract shared logic to eliminate the duplication\n"
                            "- SUPPRESS (requires justification): only for intentional duplication "
                            "(API symmetry, test isolation, framework boilerplate). "
                            "Explain WHY the duplication is acceptable.\n"
                            "- SKIP: needs more context before deciding\n\n"
                            "Provide your decision and reasoning for each group."
                        ),
                    ),
                ),
            ],
        )

    def _build_setup_lsp_prompt(language: str | None) -> GetPromptResult:
        lsp_args: dict = {"mode": "inspect"}
        if language:
            lsp_args["language"] = language
        inspect_result = _tools.call_tool("lsp_setup", lsp_args)
        lang_label = language or "all languages"
        return GetPromptResult(
            description=f"LSP setup guide ({lang_label})",
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=(
                            f"Here is the current LSP setup status for {lang_label}:\n\n"
                            f"{inspect_result.text}\n\n"
                            "Follow the recommended_actions to complete LSP setup:\n"
                            "1. Run any host_search_commands to find installed servers\n"
                            "2. Install missing servers using the suggested commands\n"
                            "3. Bind discovered servers with lsp_setup(mode='bind', ...)\n"
                            "4. Verify with lsp_setup(mode='inspect')\n\n"
                            "Work through each missing language until all are resolved."
                        ),
                    ),
                ),
            ],
        )

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
        args = arguments or {}
        if name == "review-duplicates":
            return await asyncio.to_thread(_build_review_duplicates_prompt, args.get("focus"))
        elif name == "setup-lsp":
            return await asyncio.to_thread(_build_setup_lsp_prompt, args.get("language"))
        else:
            raise ValueError(f"Unknown prompt: {name}")

    # --- MCP Completions ---

    _COMPLETION_VALUES = {
        "review-duplicates": {
            "focus": ["all", "source", "tests"],
        },
        "setup-lsp": {
            "language": [
                "python",
                "javascript_lsp",
                "typescript_lsp",
                "c_lsp",
                "cpp_lsp",
                "java_lsp",
                "go_lsp",
            ],
        },
    }

    @server.completion()
    async def completion(
        ref: PromptReference | ResourceTemplateReference,
        argument: CompletionArgument,
        _context: object | None = None,
    ) -> Completion | None:
        if not isinstance(ref, PromptReference):
            return None
        prompt_completions = _COMPLETION_VALUES.get(ref.name)
        if prompt_completions is None:
            return None
        values = prompt_completions.get(argument.name)
        if values is None:
            return None
        prefix = argument.value or ""
        filtered = [v for v in values if v.startswith(prefix)]
        return Completion(values=filtered, total=len(filtered), hasMore=False)

    return server


async def run_server() -> None:
    """Run the MCP server."""
    server = create_server()
    async with dual_stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


_close_once = threading.Event()


def _close_if_first() -> None:
    """Idempotent close: only the first caller flushes and closes resources."""
    if not _close_once.is_set():
        _close_once.set()
        _tools.close()


def _shutdown_handler(_signum: int, _frame: object) -> None:
    """Handle SIGTERM from Docker by flushing and closing resources."""
    # Unblock any thread waiting in _wait_for_background_index so that the
    # atexit handler can complete without being interrupted by os._exit().
    _tools._bg_index_done.set()
    _close_if_first()
    # os._exit avoids raising SystemExit, which would otherwise be caught by
    # Python's atexit machinery and printed as "Exception ignored in atexit
    # callback" if the signal fires while the atexit handler is still running.
    os._exit(0)


def _atexit_close() -> None:
    """Idempotent close for atexit — skips if signal handler already ran."""
    _close_if_first()


def main() -> None:
    """Entry point for the MCP server."""
    signal.signal(signal.SIGTERM, _shutdown_handler)
    atexit.register(_atexit_close)
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
