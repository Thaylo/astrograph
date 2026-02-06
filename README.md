# ASTrograph

```
    █ █ █
  █ █ █ █ █
  █       █
  █ █ █ █ █
    █ █ █
  █ █   █ █
  █       █
```

[![Docker](https://img.shields.io/badge/Docker-Hub-blue?logo=docker)](https://hub.docker.com/r/thaylo/astrograph)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Server-purple)](https://modelcontextprotocol.io)
[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ff69b4?logo=github)](https://github.com/sponsors/Thaylo)

**Stop writing code that already exists in your codebase.**

ASTrograph is an MCP server that detects when you're about to create duplicate code - and blocks it before it happens.

## See It In Action

Your codebase already has this:
```python
# src/math.py
def calculate_sum(a, b):
    return a + b
```

Your AI agent tries to write this:
```python
# src/utils.py
def add_numbers(x, y):
    return x + y
```

ASTrograph intercepts and blocks:
```
BLOCKED: Cannot write - identical code exists at src/math.py:calculate_sum (lines 1-2).
Reuse the existing implementation instead.
```

Different variable names, same structure. ASTrograph compares **code graphs**, not text.

## Quick Start

```bash
docker pull thaylo/astrograph
```

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "astrograph": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--pull", "always",
        "-v", "/path/to/your/project:/workspace:ro",
        "-v", "/path/to/your/project/.metadata_astrograph:/workspace/.metadata_astrograph",
        "thaylo/astrograph"
      ]
    }
  }
}
```

That's it. The codebase is auto-indexed at startup. In Claude:
1. `astrograph_analyze()` - Find existing duplicates
2. Use `astrograph_write` / `astrograph_edit` - They'll block duplicates automatically

## The Problem

Large codebases accumulate duplicate code because:
- **AI agents** have limited context windows → they can't see all existing code → they rewrite what exists → duplicates inflate the codebase → smaller % of codebase fits in context → more duplicates (vicious cycle)
- Developers can't remember every function they wrote 6 months ago
- Similar patterns get reimplemented independently
- Copy-paste spreads bugs across multiple locations

## Key Tools

| Tool | What It Does |
|------|--------------|
| `astrograph_analyze` | Lists all duplicates found |
| `astrograph_write` | Writes files, blocks if duplicate exists |
| `astrograph_edit` | Edits files, blocks if new code is a duplicate |

[See all 6 tools →](#tool-reference)

## Works With

- **Claude Desktop** - Full MCP integration
- **Cursor** - Via MCP settings
- **Any MCP Client** - Standard protocol

> **Note:** Python only for now. More languages coming.

---

## Tool Reference

<details>
<summary><strong>Click to expand full tool documentation</strong></summary>

### astrograph_analyze

Find duplicates in the indexed codebase.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_reindex` | boolean | No | `true` | If stale, automatically re-index before analyzing |

### astrograph_write

Write Python code to a file with automatic duplicate detection.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Absolute path to the file to write |
| `content` | string | Yes | - | The Python code content to write |

Returns: `BLOCKED` if duplicate exists, `WARNING + Success` if similar, or `Success`

### astrograph_edit

Edit a Python file with automatic duplicate detection.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Absolute path to the file to edit |
| `old_string` | string | Yes | - | The exact text to replace |
| `new_string` | string | Yes | - | The replacement Python code |

### astrograph_suppress

Suppress a duplicate group by its WL hash (shown in analyze output).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `wl_hash` | string | Yes | - | WL hash from analyze output |

### astrograph_unsuppress

Remove suppression from a hash.

### astrograph_list_suppressions

List all currently suppressed hashes.

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
    "astrograph": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--pull", "always",
        "-v", "/path/to/your/project:/workspace:ro",
        "-v", "/path/to/your/project/.metadata_astrograph:/workspace/.metadata_astrograph",
        "thaylo/astrograph"
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
    "astrograph": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--pull", "always",
        "-v", "${workspaceFolder}:/workspace:ro",
        "-v", "${workspaceFolder}/.metadata_astrograph:/workspace/.metadata_astrograph",
        "thaylo/astrograph"
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
    "astrograph": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--pull", "always",
        "-v", ".:/workspace:ro",
        "-v", "./.metadata_astrograph:/workspace/.metadata_astrograph",
        "thaylo/astrograph"
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
    "astrograph": {
      "command": "python",
      "args": ["-m", "astrograph.server"],
      "cwd": "/path/to/astrograph"
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
docker build -t astrograph .
```

## License

MIT
