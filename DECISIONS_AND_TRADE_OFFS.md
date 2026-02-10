# Decisions and Trade-Offs

This document records product and engineering decisions that affect ASTrograph behavior,
with explicit trade-offs, constraints, and consequences.

## Scope

- Audience: ASTrograph users, contributors, and integrators.
- Purpose: provide transparent, technical rationale for opinionated defaults.
- Status model:
  - `Accepted`: active and expected behavior.
  - `Superseded`: replaced by a newer decision.
  - `Proposed`: under evaluation, not yet default.

## Decision Log

## D-001: Plugin-First Language Architecture

- Status: `Accepted`
- Effective date: `2026-02-07`

### Context

ASTrograph historically had Python-centric internals. This increased coupling and made new
language onboarding costly.

### Decision

Language support is plugin-based. Python and JavaScript are both delivered through plugins:

- `python` -> `PythonLSPPlugin` (LSP symbols + AST graphing)
- `javascript_lsp` -> `JavaScriptLSPPlugin` (LSP symbols + structural graphing)

### Trade-offs

- Pros:
  - Uniform extension model for future languages.
  - Lower incremental engineering cost per language.
  - Clear runtime boundaries between core and language adapters.
- Cons:
  - LSP server availability becomes an operational dependency.
  - Symbol quality varies by language server implementation.

## D-002: Bundle JavaScript LSP Runtime in Official Docker Image

- Status: `Superseded` by D-009
- Effective date: `2026-02-07`
- Superseded date: `2026-02-10`

### Context

Unbundled JS LSP dependencies create setup friction and inconsistent local environments.

### Decision

Official Docker images bundle `node`, `npm`, `typescript`, and
`typescript-language-server`.

### Supersession

Replaced by D-009 (fully unbundled architecture). All language servers are external
and configured exclusively through `astrograph_lsp_setup bind`.

## D-003: Duplicate Significance Policy (Noise Reduction)

- Status: `Accepted`
- Effective date: `2026-02-07`

### Context

Suppression history showed a high ratio of false-positive or acceptable repetition findings
(mostly tiny guard blocks and boilerplate).

### Decision

Default significance thresholds are:

- Exact/pattern duplicate discovery: `min_node_count = 5`.
- Block duplicate discovery: `min_node_count = 10`.
- Block report filter: ignore block duplicates with `< 3` lines.
- Pre-create checks (`write`/`edit`): `min_node_count = 10`.

### Trade-offs

- Pros:
  - Large reduction in low-value duplicate alerts.
  - Lower suppression maintenance burden.
  - Better signal-to-noise ratio for actionable findings.
- Cons:
  - Some small-but-real duplicated blocks may not be reported.
  - Teams favoring strict micro-duplication control may require custom tuning.

## D-004: Ignore Import-Only LSP Symbol Units

- Status: `Accepted`
- Effective date: `2026-02-07`

### Context

Some LSP servers return symbols whose extracted ranges map to import/export boilerplate
(for example in package/module index files). These are poor duplication signals.

### Decision

LSP-based extraction filters out units whose extracted code is import-only:

- starts with `from `
- starts with `import `

### Trade-offs

- Pros:
  - Avoids non-actionable duplicates from module wiring code.
  - Improves relevance of exact duplicate reports.
- Cons:
  - Rare edge case: import-heavy generated code patterns are intentionally ignored.

## D-005: Timestamped Analysis Reports Only

- Status: `Accepted`
- Effective date: `2026-02-07`

### Context

A stable alias (`analysis_report.txt`) creates ambiguity in multi-run/multi-agent workflows.

### Decision

Analyze output is persisted only as timestamped files:

- `.metadata_astrograph/analysis_report_<YYYYMMDD>_<HHMMSS>_<microseconds>.txt`

Legacy alias `analysis_report.txt` is no longer produced.

### Trade-offs

- Pros:
  - Strong provenance and deterministic run attribution.
  - Better compatibility with concurrent workflows and audits.
- Cons:
  - Consumers must parse the returned `Details:` path instead of relying on a fixed filename.

## D-006: Suppression Persistence and Intent

- Status: `Accepted`
- Effective date: `2026-02-07`

### Context

Suppressions are necessary for tolerated duplication patterns and should survive process restarts.

### Decision

- Suppressions are persisted in index metadata storage.
- Suppressions are treated as explicit user intent, not automatic filtering.
- Stale suppressions are invalidated when tracked source evidence changes.

### Trade-offs

- Pros:
  - Stable operator control over tolerated findings.
  - Lower repetitive triage cost across sessions.
- Cons:
  - Suppression state must be managed intentionally in CI/reproducible environments.

## D-007: Deterministic Agent-Assisted LSP Wiring

- Status: `Superseded` by D-009
- Effective date: `2026-02-08`
- Superseded date: `2026-02-10`

### Context

Bundled language plugins depend on external LSP executables (`pylsp`,
`typescript-language-server`). In heterogeneous host environments, executable locations
or command wrappers may differ, and AI agents may not know internal wiring details.

### Decision

Expose one setup MCP tool (`astrograph_lsp_setup`) with deterministic modes:

- `inspect`: report effective command, source, and availability per bundled language.
- `auto_bind`: try deterministic probes first; if still missing, consume agent-provided
  observations (`language` + `command`) as fallback inputs.
- `bind`: persist an explicit command override for one language.
- `unbind`: remove a persisted override.

Command resolution precedence is:

1. persisted binding (`.metadata_astrograph/lsp_bindings.json`)
2. environment override (`ASTROGRAPH_*_LSP_COMMAND`)
3. built-in default command

### Supersession

Replaced by D-009 (binding-only resolution). Environment variable overrides and
the binding-env-default resolution chain have been removed entirely.

## D-008: Hybrid LSP Runtime Strategy (Bundle Python/JS, Attach C/C++/Java)

- Status: `Superseded` by D-009
- Effective date: `2026-02-08`
- Superseded date: `2026-02-10`

### Context

Bundling every language server in official images increases maintenance cost and image complexity.
At the same time, requiring end users to manually install Python/JavaScript prerequisites creates
high adoption friction for the most common workflows.

### Decision

ASTrograph ships with a hybrid runtime model:

- Bundled subprocess defaults:
  - `python` -> `pylsp`
  - `javascript_lsp` -> `typescript-language-server --stdio`
- Attach defaults (already-running server endpoints):
  - `c_lsp` -> `tcp://127.0.0.1:2087`
  - `cpp_lsp` -> `tcp://127.0.0.1:2088`
  - `java_lsp` -> `tcp://127.0.0.1:2089`

### Supersession

Replaced by D-009. All languages are now fully unbundled and use TCP attach mode.
No language runtimes are shipped in the Docker image.

## D-009: Fully Unbundled, Binding-Only LSP Architecture

- Status: `Accepted`
- Effective date: `2026-02-10`

### Context

The hybrid bundled/attach model (D-008) and environment variable overrides (D-007)
created multiple implicit configuration paths. Environment variables were invisible state,
hard to debug, and a potential security concern. Bundled runtimes (Python LSP, Node.js/npm)
bloated the Docker image and required maintenance.

### Decision

All 7 language adapters are fully unbundled:

- No language server runtimes are shipped in the Docker image.
- No environment variable overrides (`ASTROGRAPH_*_LSP_COMMAND`, `ASTROGRAPH_*_LSP_TIMEOUT`).
- The only valid configuration method is explicit binding via `astrograph_lsp_setup(mode='bind')`.
- Command resolution is: **binding -> fail fast**. No binding = not configured.
- Unconfigured languages return a `NullLSPClient` immediately -- no connection attempt,
  no silent degradation, no `_disabled` flag chain.
- All languages use TCP attach endpoints as defaults (only used for display/discovery,
  never for silent connection attempts).
- `auto_bind` remains as an explicit discovery tool for agents, not a runtime fallback.

Default TCP endpoints (for `auto_bind` discovery and documentation):

| Language | Endpoint |
|----------|----------|
| C | `tcp://127.0.0.1:2087` |
| C++ | `tcp://127.0.0.1:2088` |
| Java | `tcp://127.0.0.1:2089` |
| Python | `tcp://127.0.0.1:2090` |
| Go | `tcp://127.0.0.1:2091` |
| JavaScript | `tcp://127.0.0.1:2092` |
| TypeScript | `tcp://127.0.0.1:2093` |

### Trade-offs

- Pros:
  - Single, explicit configuration path -- no invisible env var state.
  - Minimal Docker image (Python-only, no Node.js/npm).
  - Fail-fast behavior -- misconfiguration is immediately visible.
  - Consistent architecture across all 7 languages.
  - Reduced attack surface (no bundled runtimes, no env var injection).
- Cons:
  - Users must start and bind language servers explicitly before use.
  - No zero-configuration experience for Python or JavaScript.
  - Timeout is hardcoded (5.0s) -- not configurable per-language without code change.
