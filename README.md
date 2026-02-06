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

Add `.mcp.json` to your project root:

```json
{
  "mcpServers": {
    "astrograph": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--pull", "always",
        "-v", ".:/workspace",
        "thaylo/astrograph"
      ]
    }
  }
}
```

The codebase is auto-indexed at startup and re-indexed on file changes. Then:
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
| `astrograph_analyze` | Find duplicate Python code (verified via graph isomorphism) |
| `astrograph_write` | Write Python file. Blocks if duplicate exists, warns on similarity |
| `astrograph_edit` | Edit Python file. Blocks if new code duplicates existing, warns on similarity |

[See all 8 tools →](#tool-reference)

## Works With

- **Claude Code** - Tested. Project-level `.mcp.json`
- **Any MCP client** - Should work (standard stdio protocol), but untested

> **Note:** Python only for now.

---

## Tool Reference

<details>
<summary><strong>Click to expand full tool documentation</strong></summary>

### astrograph_analyze

Find duplicate Python code (verified via graph isomorphism).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_reindex` | boolean | No | `true` | Auto re-index if stale (default: true) |

### astrograph_write

Write Python file. Blocks if duplicate exists, warns on similarity.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Absolute file path |
| `content` | string | Yes | - | Python code to write |

### astrograph_edit

Edit Python file. Blocks if new code duplicates existing, warns on similarity.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Absolute file path |
| `old_string` | string | Yes | - | Exact text to replace (must be unique) |
| `new_string` | string | Yes | - | Replacement code |

### astrograph_suppress

Suppress a duplicate by WL hash (from analyze output).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `wl_hash` | string | Yes | - | WL hash from analyze output |

### astrograph_suppress_batch

Suppress multiple duplicates by WL hash list.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `wl_hashes` | string[] | Yes | - | WL hashes from analyze output |

### astrograph_unsuppress

Unsuppress a hash.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `wl_hash` | string | Yes | - | The WL hash to unsuppress |

### astrograph_unsuppress_batch

Unsuppress multiple hashes.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `wl_hashes` | string[] | Yes | - | WL hashes to unsuppress |

### astrograph_list_suppressions

List suppressed hashes.

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
        "-v", "/path/to/your/project:/workspace",
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
        "-v", "${workspaceFolder}:/workspace",
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
        "-v", ".:/workspace",
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
- SQLite persistence with WAL mode (survives restarts)
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
