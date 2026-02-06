# ASTograph

[![Docker](https://img.shields.io/badge/Docker-Hub-blue?logo=docker)](https://hub.docker.com/r/thaylo/astograph)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Server-purple)](https://modelcontextprotocol.io)

Detect structural code duplication in **Python** codebases using AST graph isomorphism. An MCP server that identifies duplicate code structures before they proliferate.

> **Note:** Currently supports Python only. Additional language support planned for future releases.

## Quick Start (Docker)

**30 seconds to get started:**

```bash
# Pull from Docker Hub
docker pull thaylo/astograph

# Or build locally
docker build -t astograph .
```

## Problem

Large codebases accumulate structural duplication over time:

1. Developers can't remember every existing implementation
2. Similar patterns get reimplemented independently
3. The codebase inflates with redundant code
4. Maintenance burden increases
5. Bugs fixed in one place remain in duplicates

## Solution

ASTograph provides 9 tools to break this cycle:

| Tool | Purpose |
|------|---------|
| `astograph_index` | Index a Python codebase's structural patterns |
| `astograph_analyze` | Find existing structural duplicates |
| `astograph_check` | Check before creating new code |
| `astograph_compare` | Compare two code snippets |
| `astograph_suppress` | Mute acceptable duplicates |
| `astograph_unsuppress` | Restore suppressed duplicates |
| `astograph_suppress_idiomatic` | Suppress all idiomatic patterns at once |
| `astograph_list_suppressions` | View suppressed hashes |
| `astograph_check_staleness` | Check if index needs refresh |

## How It Works

1. **AST to Graph**: Python code is parsed into AST, then converted to a labeled directed graph
2. **Weisfeiler-Leman Hashing**: Graphs are hashed using WL algorithm for O(1) lookup of potential matches
3. **Structural Fingerprinting**: Quick filtering based on node counts, label histograms, degree sequences
4. **Full Isomorphism Verification**: NetworkX VF2 algorithm for definitive structural equivalence check

## Event-Driven Mode

The Docker image runs in **event-driven mode** by default, providing:

- **In-memory index**: Always hot, no cold starts
- **File watching**: Automatic re-indexing when files change
- **Analysis cache**: Pre-computed results for instant `analyze()` responses

This is enabled via `ASTOGRAPH_EVENT_DRIVEN=1` in the Dockerfile. The `--tmpfs` mount in the configuration examples provides a writable space for the index while keeping your source code read-only.

> **Note:** The index is ephemeral with the default Docker configuration. For persistent indexing across sessions, replace `"--tmpfs", "/workspace/.metadata_astograph"` with `"-v", "astograph-data:/workspace/.metadata_astograph"` to use a named volume.

## Installation

### Option 1: Docker (Recommended)

```bash
docker pull thaylo/astograph
```

### Option 2: Local Python

```bash
pip install -e .
```

## MCP Configuration Examples

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "astograph": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "/path/to/your/project:/workspace:ro",
        "--tmpfs", "/workspace/.metadata_astograph",
        "thaylo/astograph"
      ]
    }
  }
}
```

### Generic MCP Client (Project-level)

Add to `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "astograph": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", ".:/workspace:ro",
        "--tmpfs", "/workspace/.metadata_astograph",
        "thaylo/astograph"
      ]
    }
  }
}
```

### Cursor

Add to Cursor's MCP settings:

```json
{
  "mcpServers": {
    "astograph": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "${workspaceFolder}:/workspace:ro",
        "--tmpfs", "/workspace/.metadata_astograph",
        "thaylo/astograph"
      ]
    }
  }
}
```

### Local Python (without Docker)

```json
{
  "mcpServers": {
    "astograph": {
      "command": "python",
      "args": ["-m", "astograph.server"],
      "cwd": "/path/to/astograph"
    }
  }
}
```

**Note:** When using Docker, paths are relative to `/workspace` (the mounted directory). Use `astograph_index` with path `/workspace` to index the current project.

## Tool Reference

### astograph_index

Index a Python codebase for structural analysis. **Call this first.**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | Yes | - | Path to directory or file to index |
| `incremental` | boolean | No | `true` | Only re-index changed files (98% faster) |

```
astograph_index(path="/workspace")
```

### astograph_analyze

Find duplicates in the indexed codebase.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `thorough` | boolean | No | `true` | Show all duplicates including small ones (~2+ lines) |

Returns:
- **Exact duplicates**: Structurally identical code (verified via graph isomorphism)
- **Block duplicates**: Duplicate for/while/if/try/with blocks
- **Pattern duplicates**: Same structure, different operators

### astograph_check

Check if similar code exists **before** creating new code.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code` | string | Yes | - | Python code to check |

Returns:
- **STOP**: Identical code exists - reuse it
- **CAUTION**: Very similar code exists - consider reusing
- **NOTE**: Partially similar code - review for potential reuse

### astograph_compare

Compare two code snippets for structural equivalence.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code1` | string | Yes | - | First Python code snippet |
| `code2` | string | Yes | - | Second Python code snippet |

Returns:
- **EQUIVALENT**: Structurally identical
- **SIMILAR**: Compatible structure but not identical
- **DIFFERENT**: Structurally different

### astograph_suppress

Suppress a duplicate group by its WL hash.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `wl_hash` | string | Yes | - | WL hash from analyze output |

### astograph_unsuppress

Remove suppression from a hash.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `wl_hash` | string | Yes | - | WL hash to unsuppress |

### astograph_suppress_idiomatic

Suppress all idiomatic patterns (guard clauses, test setup, etc.) in one call. No parameters.

### astograph_list_suppressions

List all currently suppressed hashes. No parameters.

### astograph_check_staleness

Check if the index is stale (files changed since indexing).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | No | - | Root path to check for new files |

## Workflow Example

```
1. astograph_index(path="/workspace")     # Index the codebase
2. astograph_analyze()                     # Find duplicates
3. For each duplicate:
   - Refactor to eliminate duplication, OR
   - astograph_suppress(wl_hash="...")     # If intentional
4. astograph_index(incremental=True)       # Re-index after changes
5. astograph_analyze()                     # Verify: 0 pending duplicates
```

## Code Example

These two functions are **structurally identical** (isomorphic ASTs):

```python
def calculate(a, b):
    return a + b

def sum_values(x, y):
    return x + y
```

ASTograph detects these as duplicates because they have the same AST structure, even though variable names differ.

## Architecture

```
+------------------+     +-------------------+
| Python Source    |---->| AST Parser        |
+------------------+     +---------+---------+
                                   |
                         +---------v---------+
                         | Graph Conversion  |
                         | (labeled DiGraph) |
                         +---------+---------+
                                   |
              +--------------------+--------------------+
              |                    |                    |
    +---------v-----+    +---------v-----+    +--------v--------+
    | WL Hash       |    | Fingerprint   |    | Hierarchy       |
    | (fast lookup) |    | (filtering)   |    | Hashes          |
    +---------------+    +---------------+    +-----------------+
              |                    |                    |
              +--------------------+--------------------+
                                   |
                         +---------v---------+
                         | Index Storage     |
                         | (hash buckets)    |
                         +-------------------+
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -q

# Build Docker image
docker build -t astograph .
```

## License

MIT
