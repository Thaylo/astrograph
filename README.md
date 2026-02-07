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
[![Arch](https://img.shields.io/badge/arch-amd64%20%7C%20arm64-blue?logo=linux)](https://hub.docker.com/r/thaylo/astrograph)
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
        "thaylo/astrograph:latest"
      ]
    }
  }
}
```

Multi-arch Docker image: works on both **amd64** (x86_64) and **arm64** (Apple Silicon, AWS Graviton).

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
| `astrograph_analyze` | Find duplicate code (verified via graph isomorphism) |
| `astrograph_write` | Write file. Blocks if duplicate exists, warns on similarity |
| `astrograph_edit` | Edit file. Blocks if new code duplicates existing, warns on similarity |

[See all 11 tools →](#tool-reference)

## Works With

- **Claude Code** - Tested. Project-level `.mcp.json`
- **Codex** - Tested. `stdio` framing + `resources/list` compatibility
- **Any MCP client** - Should work (standard stdio protocol)

> **Note:** Python runs through the `python` LSP plugin (`pylsp` by default), not a built-in parser path. More languages can be added via plugins — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Tool Reference

<details>
<summary><strong>Click to expand full tool documentation</strong></summary>

### astrograph_analyze

Find duplicate code (verified via graph isomorphism).

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `auto_reindex` | boolean | No | `true` | Auto re-index if stale (default: true) |

### astrograph_write

Write file. Blocks if duplicate exists, warns on similarity.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `file_path` | string | Yes | - | Absolute file path |
| `content` | string | Yes | - | Code to write |

### astrograph_edit

Edit file. Blocks if new code duplicates existing, warns on similarity.

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

### astrograph_status

Check server readiness. Returns instantly even during indexing.

### astrograph_metadata_erase

Erase all persisted metadata (.metadata_astrograph/). Resets server to idle.

### astrograph_metadata_recompute_baseline

Erase metadata and re-index the codebase from scratch.

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
        "thaylo/astrograph:latest"
      ]
    }
  }
}
```

</details>

<details>
<summary><strong>Codex (~/.codex/config.toml)</strong></summary>

```toml
[mcp_servers.astrograph]
command = "docker"
args = [
  "run", "--rm", "-i", "--pull", "always",
  "-v", "/path/to/your/project:/workspace",
  "-v", "/path/to/your/project/.metadata_astrograph:/workspace/.metadata_astrograph",
  "thaylo/astrograph:latest"
]
```

ASTrograph auto-indexes at startup using:
1. `ASTROGRAPH_WORKSPACE` (if set)
2. `/workspace` (Docker)
3. `PWD` / current working directory (local launches)

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
        "thaylo/astrograph:latest"
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
        "thaylo/astrograph:latest"
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

1. **AST to Graph**: Source code is parsed into AST, then converted to a labeled directed graph
2. **Weisfeiler-Leman Hashing**: Graphs are hashed using WL algorithm for O(1) lookup
3. **Structural Fingerprinting**: Quick filtering based on node counts and degree sequences
4. **Full Isomorphism Verification**: NetworkX VF2 algorithm confirms structural equivalence

```
Source → Parser → Graph → WL Hash → Index Lookup → Match
```

The server runs in **event-driven mode**:
- SQLite persistence with WAL mode (survives restarts)
- File watching (auto re-index on changes)
- Pre-computed analysis cache (instant responses)
- Incremental delta persistence (only changed files are persisted)

</details>

---

## Language Support

Python is provided by the `python` LSP plugin (default command: `pylsp`). The same plugin architecture supports adding new languages via tree-sitter or LSP-backed adapters — see [Adding a New Language](CONTRIBUTING.md#adding-a-new-language-plugin) for a step-by-step guide.

Python LSP env overrides:
- `ASTROGRAPH_PY_LSP_COMMAND`
- `ASTROGRAPH_PY_LSP_TIMEOUT`

A built-in `javascript_lsp` adapter is also available. By default it uses `typescript-language-server --stdio`; override with:
- `ASTROGRAPH_JS_LSP_COMMAND`
- `ASTROGRAPH_JS_LSP_TIMEOUT`

## CLI

ASTrograph also ships a standalone CLI for quick analysis outside of MCP:

```bash
pip install .
astrograph-cli duplicates /path/to/project     # Find duplicates
astrograph-cli check /path/to/project code.py   # Check for similar code
astrograph-cli compare file1.py file2.py         # Compare two files
```

## Development

```bash
pip install -e ".[dev]"
# If working on tree-sitter language plugins:
pip install -e ".[dev,treesitter]"
pytest tests/ -q
docker build -t astrograph .
```

## License

MIT
