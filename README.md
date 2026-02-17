# ASTrograph

<p align="center">
  <img src="astrograph_poster.jpg" alt="ASTrograph" width="400">
</p>

[![Docker](https://img.shields.io/badge/Docker-Hub-blue?logo=docker)](https://hub.docker.com/r/thaylo/astrograph)
[![Arch](https://img.shields.io/badge/arch-amd64-blue?logo=linux)](https://hub.docker.com/r/thaylo/astrograph)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Server-purple)](https://modelcontextprotocol.io)

An [MCP server](https://modelcontextprotocol.io) that stops AI agents from writing duplicate code. It offers improved write and edit tools that compare new code against every function already in your codebase using AST graph isomorphism, blocking the operation when a structural duplicate is found. Variable names, formatting, and comments are irrelevant -- if two pieces of code have the same abstract structure, ASTrograph treats them as duplicates.

## Installation

Add `.mcp.json` to your project root:

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

That's it. The codebase is indexed at startup and re-indexed on file changes.

<details>
<summary><strong>Claude Desktop</strong></summary>

macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
Linux: `~/.config/Claude/claude_desktop_config.json`
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
<summary><strong>Codex CLI</strong></summary>

`~/.codex/config.toml` (global — works for every project):

```toml
[mcp_servers.astrograph]
command = "docker"
args = [
  "run", "--rm", "-i", "--pull", "always",
  "--add-host", "host.docker.internal:host-gateway",
  "-v", ".:/workspace",
  "-v", "./.metadata_astrograph:/workspace/.metadata_astrograph",
  "thaylo/astrograph:latest"
]
```

Codex resolves `.` from the current project directory, so the same config works
everywhere. You can also use a project-scoped `.codex/config.toml` if preferred.

> **Linux:** Docker must be installed and your user must be in the `docker`
> group (`sudo usermod -aG docker $USER`, then log out and back in).

</details>

<details>
<summary><strong>Global setup (all projects)</strong></summary>

Add astrograph to `~/.claude/mcp_servers.json` so it is available in every project
without a per-project `.mcp.json`:

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

</details>

<details>
<summary><strong>Without Docker</strong></summary>

```bash
pip install .          # macOS / venv
pipx install .         # Linux (avoids PEP 668 externally-managed errors)
```

```json
{
  "mcpServers": {
    "astrograph": {
      "command": "astrograph",
      "args": []
    }
  }
}
```

> **Linux note:** Modern distributions (Ubuntu 24.04+, Fedora 40+) mark the system
> Python as externally managed. Use `pipx`, install inside a virtualenv, or use the
> Docker method above.

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

Different variable names, identical structure. Source code is converted into labeled directed graphs and compared using Weisfeiler-Leman hashing with VF2 isomorphism verification. Text similarity is not involved.

## Language support

All languages require an explicit LSP binding via `astrograph_lsp_setup(mode='bind')`. No language server is bundled -- ASTrograph attaches to already-running servers over TCP or stdio. Unconfigured languages fail fast with no silent fallbacks.

| Language | Versions | Default endpoint |
|----------|----------|-----------------|
| Python | 3.11 -- 3.14 | `tcp://127.0.0.1:2090` |
| JavaScript | ES2021+, Node 20/22/24 LTS | `tcp://127.0.0.1:2092` |
| TypeScript | TypeScript 5.x, Node 20/22/24 LTS | `tcp://127.0.0.1:2093` |
| Go | 1.21 -- 1.25 | `tcp://127.0.0.1:2091` |
| C | C11, C17, C23 | `tcp://127.0.0.1:2087` |
| C++ | C++17, C++20, C++23 | `tcp://127.0.0.1:2088` |
| Java | 11, 17, 21, 25 | `tcp://127.0.0.1:2089` |

## Automated releases

Docker Hub publishing is automated from Git tags via GitHub Actions (`.github/workflows/release.yml`).

When you push a tag in the form `vMAJOR.MINOR.PATCH`, the workflow:

1. Verifies tag/version synchronization (`pyproject.toml` and `src/astrograph/__init__.py` must match the tag).
2. Runs `ruff`, `mypy`, and the full test suite.
3. Builds and pushes a multi-arch image (`linux/amd64`, `linux/arm64`) to Docker Hub.
4. Publishes synchronized tags:
   - `thaylo/astrograph:<major>.<minor>.<patch>`
   - `thaylo/astrograph:v<major>.<minor>.<patch>`
   - `thaylo/astrograph:<major>.<minor>`
   - `thaylo/astrograph:<major>`
   - `thaylo/astrograph:latest`

### One-time repository setup

Add these GitHub repository secrets:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (Docker Hub access token with push permissions)

### Release command

Use the helper script to bump version, run checks, and push the release tag:

```bash
./scripts/release.sh 0.5.74
```

That tag push triggers the automated Docker publish flow.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Thaylo/astrograph&type=date&legend=top-left)](https://www.star-history.com/#Thaylo/astrograph&type=date&legend=top-left)

## License

MIT
