# ASTograph

```
    █ █ █
  █ █ █ █ █
  █       █
  █ █ █ █ █
    █ █ █
  █ █   █ █
  █       █
```

[![Docker](https://img.shields.io/badge/Docker-Hub-blue?logo=docker)](https://hub.docker.com/r/thaylo/astograph)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Server-purple)](https://modelcontextprotocol.io)
[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ff69b4?logo=github)](https://github.com/sponsors/Thaylo)

**Stop writing code that already exists in your codebase.**

ASTograph is an MCP server that detects when you're about to create duplicate code - and blocks it before it happens.

## See It In Action

You write this:
```python
def add_numbers(x, y):
    return x + y
```

ASTograph says:
```
BLOCKED: Identical code exists at src/math.py:calculate_sum (lines 5-8).
Reuse the existing implementation instead.
```

It catches duplicates even when variable names differ - because it compares **code structure**, not text.

## Quick Start

```bash
docker pull thaylo/astograph
```

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

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

Then in Claude:
1. `astograph_index(path="/workspace")` - Index your codebase
2. `astograph_analyze()` - Find existing duplicates
3. Use `astograph_write` / `astograph_edit` - They'll block duplicates automatically

## The Problem

Large codebases accumulate duplicate code because:
- **AI agents** have limited context windows → they can't see all existing code → they rewrite what exists → duplicates inflate the codebase → smaller % of codebase fits in context → more duplicates (vicious cycle)
- Developers can't remember every function they wrote 6 months ago
- Similar patterns get reimplemented independently
- Copy-paste spreads bugs across multiple locations

## Key Tools

| Tool | What It Does |
|------|--------------|
| `astograph_index` | Scans your codebase (run this first) |
| `astograph_analyze` | Lists all duplicates found |
| `astograph_write` | Writes files, blocks if duplicate exists |
| `astograph_edit` | Edits files, blocks if new code is a duplicate |
| `astograph_check` | Check if code exists before writing |

[See all 11 tools →](#tool-reference)

## Works With

- **Claude Desktop** - Full MCP integration
- **Cursor** - Via MCP settings
- **Any MCP Client** - Standard protocol

> **Note:** Python only for now. More languages coming.

---

## Tool Reference

<details>
<summary><strong>Click to expand full tool documentation</strong></summary>

### astograph_index

Index a Python codebase for structural analysis. **Call this first.**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `path` | string | Yes | - | Path to directory or file to index |
| `incremental` | boolean | No | `true` | Only re-index changed files |

### astograph_analyze

Find duplicates in the indexed codebase.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `thorough` | boolean | No | `true` | Show all duplicates including small ones |

### astograph_check

Check if similar code exists **before** creating new code.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code` | string | Yes | - | Python code to check |

Returns: `STOP` (duplicate), `CAUTION` (similar), or `NOTE` (partial match)

### astograph_compare

Compare two code snippets for structural equivalence.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `code1` | string | Yes | - | First Python code snippet |
| `code2` | string | Yes | - | Second Python code snippet |

### astograph_write

Write Python code to a file with automatic duplicate detection.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Absolute path to the file to write |
| `content` | string | Yes | - | The Python code content to write |

Returns: `BLOCKED` if duplicate exists, `WARNING + Success` if similar, or `Success`

### astograph_edit

Edit a Python file with automatic duplicate detection.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Absolute path to the file to edit |
| `old_string` | string | Yes | - | The exact text to replace |
| `new_string` | string | Yes | - | The replacement Python code |

### astograph_suppress

Suppress a duplicate group by its WL hash (shown in analyze output).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `wl_hash` | string | Yes | - | WL hash from analyze output |

### astograph_unsuppress

Remove suppression from a hash.

### astograph_suppress_idiomatic

Suppress all idiomatic patterns (guard clauses, test setup, etc.) in one call.

### astograph_list_suppressions

List all currently suppressed hashes.

### astograph_check_staleness

Check if the index needs to be refreshed.

</details>

---

## Configuration Examples

<details>
<summary><strong>Claude Desktop</strong></summary>

macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
Windows: `%APPDATA%\Claude\claude_desktop_config.json`

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

</details>

<details>
<summary><strong>Cursor</strong></summary>

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

</details>

<details>
<summary><strong>Project-level (.mcp.json)</strong></summary>

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

</details>

<details>
<summary><strong>Local Python (without Docker)</strong></summary>

```bash
pip install -e .
```

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

</details>

---

## How It Works

<details>
<summary><strong>Technical details</strong></summary>

1. **AST to Graph**: Python code is parsed into AST, then converted to a labeled directed graph
2. **Weisfeiler-Leman Hashing**: Graphs are hashed using WL algorithm for O(1) lookup
3. **Structural Fingerprinting**: Quick filtering based on node counts and degree sequences
4. **Full Isomorphism Verification**: NetworkX VF2 algorithm confirms structural equivalence

```
Python Source → AST Parser → Graph → WL Hash → Index Lookup → Match
```

The Docker image runs in **event-driven mode** by default:
- In-memory index (no cold starts)
- File watching (auto re-index on changes)
- Analysis cache (instant responses)

</details>

---

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -q
docker build -t astograph .
```

## License

MIT
