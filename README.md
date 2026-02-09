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
[![JavaScript](https://img.shields.io/badge/JavaScript-LSP%20plugin-f7df1e?logo=javascript&logoColor=black)](https://hub.docker.com/r/thaylo/astrograph)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![MCP](https://img.shields.io/badge/MCP-Server-purple)](https://modelcontextprotocol.io)
[![Sponsor](https://img.shields.io/badge/Sponsor-❤-ff69b4?logo=github)](https://github.com/sponsors/Thaylo)

**Stop writing code that already exists in your codebase.**

ASTrograph is an MCP server that detects when you're about to create duplicate code - and blocks it before it happens.
Out of the box language support is plugin-based and LSP-backed: Python (`pylsp`) and JavaScript (`typescript-language-server`) are bundled, while C/C++/Java plugins can attach to already-running LSP servers.

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
        "--add-host", "host.docker.internal:host-gateway",
        "-v", ".:/workspace",
        "-v", "./.metadata_astrograph:/workspace/.metadata_astrograph",
        "thaylo/astrograph:latest"
      ]
    }
  }
}
```

`--add-host host.docker.internal:host-gateway` keeps attach-based C/C++/Java flows portable across Docker setups.

Multi-arch Docker image: works on both **amd64** (x86_64) and **arm64** (Apple Silicon, AWS Graviton).

The codebase is auto-indexed at startup and re-indexed on file changes. Then:
1. `astrograph_analyze()` - Find existing duplicates
2. Use `astrograph_write` / `astrograph_edit` - They'll block duplicates automatically

## First C++ Project (Docker + Host C++ LSP)

Use this exact flow for a first-time C++ setup where ASTrograph runs in Docker and a real C++ language server runs on the host (MacOS/Unix).

### 1) Prepare a tiny C++ workspace with `compile_commands.json`

```bash
mkdir -p sandbox_cpp_journey/build
cd sandbox_cpp_journey

cat > main.cpp <<'CPP'
int add(int a, int b) { return a + b; }
CPP

cat > build/compile_commands.json <<'JSON'
[
  {
    "directory": "__PROJECT_ROOT__",
    "command": "clang++ -std=c++20 -I. -c main.cpp",
    "file": "main.cpp"
  }
]
JSON

# Replace placeholder with absolute path
PROJECT="$(pwd)"
sed -i.bak "s|__PROJECT_ROOT__|${PROJECT}|g" build/compile_commands.json && rm -f build/compile_commands.json.bak
```

### 2) Start a host TCP bridge for your C++ LSP server on port `2088`

```bash
PORT=2088
PROJECT="$(pwd)"

# Example A: clangd
socat "TCP-LISTEN:${PORT},bind=0.0.0.0,reuseaddr,fork" \
  "EXEC:clangd --background-index --log=error --compile-commands-dir=${PROJECT}/build --path-mappings=/workspace=${PROJECT},stderr"

# Example B: ccls
socat "TCP-LISTEN:${PORT},bind=0.0.0.0,reuseaddr,fork" \
  "EXEC:ccls,stderr"
```

Keep this terminal open while testing.
For ccls, make sure `compile_commands.json` is discoverable from the project root.

### 3) Configure MCP to run ASTrograph in Docker

Project `.mcp.json`:

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

`--add-host` is required on Linux hosts and safe on Docker Desktop (MacOS).

### 4) Run the C++ setup loop inside your MCP client

1. `astrograph_metadata_erase()`
2. `astrograph_lsp_setup(mode="inspect", language="cpp_lsp")`
3. `astrograph_lsp_setup(mode="auto_bind", language="cpp_lsp", observations=[{"language":"cpp_lsp","command":"tcp://host.docker.internal:2088"}])`
4. `astrograph_lsp_setup(mode="inspect", language="cpp_lsp")`

You want `servers[].available=true` for `cpp_lsp`.

If `auto_bind` happened after startup indexing, follow the recommended action and run:
1. `astrograph_metadata_recompute_baseline()`
2. `astrograph_lsp_setup(mode="auto_bind", language="cpp_lsp", observations=[{"language":"cpp_lsp","command":"tcp://host.docker.internal:2088"}])`
3. `astrograph_lsp_setup(mode="inspect", language="cpp_lsp")`

### Production Guardrails for `cpp_lsp`

ASTrograph now validates C++ attach endpoints beyond raw TCP reachability.

- `verification_state="verified"`: endpoint passed LSP initialize + semantic probe, and a valid `compile_commands.json` is visible.
- `verification_state="reachable_only"`: endpoint accepted TCP but failed protocol/semantic/compile-db checks.
- In production mode (default), `reachable_only` is treated as unavailable for `cpp_lsp` (fail-closed nudges to real C++ LSP endpoints).

Validation mode:
- `ASTROGRAPH_LSP_VALIDATION_MODE=production` (default): strict for `cpp_lsp`
- `ASTROGRAPH_LSP_VALIDATION_MODE=relaxed`: allows reachable-only endpoints (not recommended for production)

### 5) Start normal duplicate-prevention workflow

1. `astrograph_analyze()`
2. Use `astrograph_write(...)` and `astrograph_edit(...)`
3. Follow `recommended_actions` in tool output when present

Attach ports are full-duplex LSP channels (query + update). Keep one stable port per language: `2087` (C), `2088` (C++), `2089` (Java).

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
| `astrograph_lsp_setup` | Inspect/bind LSP commands or attach endpoints for bundled language plugins |

[See all 12 tools →](#tool-reference)

## Works With

- **Claude Code** - Tested. Project-level `.mcp.json`
- **Codex** - Tested. `stdio` framing + `resources/list` compatibility
- **Any MCP client** - Should work (standard stdio protocol)

> **Language support (official):**
> - Bundled defaults: `python` (`pylsp`), `javascript_lsp` / `typescript_lsp` (`typescript-language-server --stdio`)
> - Attach defaults: `c_lsp` (`tcp://127.0.0.1:2087`), `cpp_lsp` (`tcp://127.0.0.1:2088`), `java_lsp` (`tcp://127.0.0.1:2089`)
>
> More languages can be added via plugins — see [CONTRIBUTING.md](CONTRIBUTING.md).

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

### astrograph_lsp_setup

Inspect and configure deterministic LSP command bindings for bundled language plugins.
Responses include `bindings`, `agent_directive`, and `recommended_actions` so AI agents can
continue with search/install/config loops instead of stopping at diagnostics.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `mode` | string | No | `inspect` | One of `inspect`, `auto_bind`, `bind`, `unbind` |
| `language` | string | Conditionally | - | Optional filter for `inspect`/`auto_bind`; required for `bind`/`unbind` (`python`, `javascript_lsp`, `typescript_lsp`, `c_lsp`, `cpp_lsp`, `java_lsp`) |
| `command` | string or string[] | Conditionally | - | Required for `bind`; executable command or attach endpoint (`tcp://host:port`, `unix:///path`) |
| `observations` | object[] | No | - | Optional host-discovery hints used by `auto_bind` (`language` + executable command or endpoint) |

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
        "--add-host", "host.docker.internal:host-gateway",
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
  "--add-host", "host.docker.internal:host-gateway",
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
        "--add-host", "host.docker.internal:host-gateway",
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

First-party language plugins are enabled by default:

| Language ID | Extensions | Default command | Mode | Env overrides |
|-------------|------------|-----------------|------|---------------|
| `python` | `.py`, `.pyi` | `pylsp` | bundled subprocess | `ASTROGRAPH_PY_LSP_COMMAND`, `ASTROGRAPH_PY_LSP_TIMEOUT` |
| `javascript_lsp` | `.js`, `.jsx`, `.mjs`, `.cjs` | `typescript-language-server --stdio` | bundled subprocess | `ASTROGRAPH_JS_LSP_COMMAND`, `ASTROGRAPH_JS_LSP_TIMEOUT` |
| `typescript_lsp` | `.ts`, `.tsx` | `typescript-language-server --stdio` | bundled subprocess | `ASTROGRAPH_TS_LSP_COMMAND`, `ASTROGRAPH_TS_LSP_TIMEOUT` |
| `c_lsp` | `.c`, `.h` | `tcp://127.0.0.1:2087` | attach to running server | `ASTROGRAPH_C_LSP_COMMAND`, `ASTROGRAPH_C_LSP_TIMEOUT` |
| `cpp_lsp` | `.cc`, `.cpp`, `.cxx`, `.hh`, `.hpp`, `.hxx`, `.ipp` | `tcp://127.0.0.1:2088` | attach to running server | `ASTROGRAPH_CPP_LSP_COMMAND`, `ASTROGRAPH_CPP_LSP_TIMEOUT` |
| `java_lsp` | `.java` | `tcp://127.0.0.1:2089` | attach to running server | `ASTROGRAPH_JAVA_LSP_COMMAND`, `ASTROGRAPH_JAVA_LSP_TIMEOUT` |

Official Docker images bundle the Python + JavaScript/TypeScript LSP runtime (`pylsp`, `node`, `npm`, `typescript`, `typescript-language-server`). C/C++/Java support uses attach endpoints and expects those language servers to be already running.
`astrograph-cli doctor` reports `ready=true` when required bundled languages are available, even if optional attach endpoints are currently offline.

For local (non-Docker) installs, verify and bootstrap bundled prerequisites with:

```bash
astrograph-cli doctor
astrograph-cli install-lsps
```

The same plugin architecture supports adding new languages via tree-sitter or LSP-backed adapters — see [Adding a New Language](CONTRIBUTING.md#adding-a-new-language-plugin).

## Detection Policy

Formal product and engineering decisions (including duplicate significance thresholds,
reporting format, and language-plugin trade-offs) are documented in
[`DECISIONS_AND_TRADE_OFFS.md`](DECISIONS_AND_TRADE_OFFS.md).

## CLI

ASTrograph also ships a standalone CLI for quick analysis outside of MCP:

```bash
pip install .

# Verify bundled LSP prerequisites and attach endpoint reachability
astrograph-cli doctor

# Auto-install missing bundled LSP servers (python + javascript/typescript)
astrograph-cli install-lsps

# Core analysis commands
astrograph-cli duplicates /path/to/project      # Find duplicates
astrograph-cli check /path/to/project code.py   # Check for similar code
astrograph-cli compare file1.py file2.py        # Compare two files
```

For scripting/CI:

```bash
astrograph-cli doctor --json
astrograph-cli install-lsps --dry-run --json
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
