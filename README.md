# ASTrograph

<p align="center">
  <img src="astrograph_poster.jpg" alt="ASTrograph" width="400">
</p>

[![Docker](https://img.shields.io/badge/Docker-Hub-blue?logo=docker)](https://hub.docker.com/r/thaylo/astrograph)
[![Arch](https://img.shields.io/badge/arch-amd64%20%7C%20arm64-blue?logo=linux)](https://hub.docker.com/r/thaylo/astrograph)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Server-purple)](https://modelcontextprotocol.io)

**Structural duplicate detection for codebases under AI-assisted development.**

ASTrograph is an [MCP server](https://modelcontextprotocol.io) that intercepts code writes and edits, compares them against every function and block already in your codebase using AST graph isomorphism, and blocks the operation when a structural duplicate is found. Variable names, formatting, and comments are irrelevant -- if two pieces of code compile to the same abstract structure, ASTrograph treats them as duplicates.

The problem it solves is specific and measurable: AI coding agents operate under limited context windows. They cannot see the entire codebase. They rewrite what already exists. The duplicates inflate the codebase, which means an even smaller percentage fits in context, which produces more duplicates. ASTrograph breaks that cycle.

## Installation

Add a `.mcp.json` file to your project root:

```json
{
  "mcpServers": {
    "astrograph": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i", "--pull", "always",
        "--add-host", "host.docker.internal:host-gateway",
        "-v", ".:/workspace",
        "-v", "./.metadata_astrograph:/workspace/.metadata_astrograph",
        "thaylo/astrograph:latest"
      ]
    }
  }
}
```

That is the entire setup. The image is multi-arch (amd64, arm64). The codebase is indexed automatically at startup and re-indexed on file changes. Metadata is persisted to `.metadata_astrograph/` so restarts are fast.

### Alternative configurations

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
        "--add-host", "host.docker.internal:host-gateway",
        "-v", "/absolute/path/to/project:/workspace",
        "-v", "/absolute/path/to/project/.metadata_astrograph:/workspace/.metadata_astrograph",
        "thaylo/astrograph:latest"
      ]
    }
  }
}
```

</details>

<details>
<summary><strong>Codex</strong></summary>

`~/.codex/config.toml`:

```toml
[mcp_servers.astrograph]
command = "docker"
args = [
  "run", "--rm", "-i", "--pull", "always",
  "--add-host", "host.docker.internal:host-gateway",
  "-v", "/absolute/path/to/project:/workspace",
  "-v", "/absolute/path/to/project/.metadata_astrograph:/workspace/.metadata_astrograph",
  "thaylo/astrograph:latest"
]
```

</details>

<details>
<summary><strong>Without Docker</strong></summary>

```bash
pip install .
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

## How it works

Your codebase already contains:

```python
# src/math.py
def calculate_sum(a, b):
    return a + b
```

An AI agent tries to write:

```python
# src/utils.py
def add_numbers(x, y):
    return x + y
```

ASTrograph blocks the write:

```
BLOCKED: Cannot write - identical code exists at src/math.py:calculate_sum (lines 1-2).
Reuse the existing implementation instead.
```

Different variable names, identical structure. ASTrograph converts source code into labeled directed graphs and compares them using Weisfeiler-Leman hashing with full VF2 isomorphism verification. Text similarity is not involved.

## Tools

Core operations:

| Tool | Purpose |
|------|---------|
| `astrograph_write` | Write a file. Blocks if the new code duplicates existing code. |
| `astrograph_edit` | Edit a file. Blocks if the replacement code duplicates existing code. |
| `astrograph_analyze` | Scan the indexed codebase and report all structural duplicates. |

Management:

| Tool | Purpose |
|------|---------|
| `astrograph_status` | Check server readiness. Returns instantly, even during indexing. |
| `astrograph_set_workspace` | Switch workspace directory at runtime. Triggers full re-index. |
| `astrograph_suppress` | Suppress a known duplicate by its WL hash. |
| `astrograph_unsuppress` | Re-enable a previously suppressed duplicate. |
| `astrograph_list_suppressions` | List all suppressed hashes. |
| `astrograph_lsp_setup` | Inspect and configure LSP bindings for language plugins. |
| `astrograph_generate_ignore` | Generate `.astrographignore` with sensible defaults. |
| `astrograph_metadata_erase` | Delete all persisted metadata. Resets to idle. |
| `astrograph_metadata_recompute_baseline` | Erase metadata and re-index from scratch. |

## Language support

ASTrograph uses LSP-backed language plugins. Python, JavaScript, TypeScript, and Go are supported out of the box. C, C++, and Java work by attaching to an already-running language server over TCP.

| Language | Versions | Mode | Default command |
|----------|----------|------|-----------------|
| Python | 3.11 -- 3.14 | bundled | `pylsp` |
| JavaScript | ES2021+, Node 20/22/24 LTS | bundled | `typescript-language-server --stdio` |
| TypeScript | TypeScript 5.x, Node 20/22/24 LTS | bundled | `typescript-language-server --stdio` |
| Go | 1.21 -- 1.25 | bundled | `gopls serve` |
| C | C11, C17, C23 | attach | `tcp://127.0.0.1:2087` |
| C++ | C++17, C++20, C++23 | attach | `tcp://127.0.0.1:2088` |
| Java | 11, 17, 21, 25 | attach | `tcp://127.0.0.1:2089` |

The Docker image bundles Python and JavaScript/TypeScript LSP runtimes. For attach-based languages, expose the language server on a TCP port and use `astrograph_lsp_setup` to bind.

New languages can be added via plugins. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Tested with

- **Claude Code** -- project-level `.mcp.json`
- **Codex** -- stdio framing, `resources/list` compatible
- **Any MCP client** -- standard stdio protocol

## License

MIT
