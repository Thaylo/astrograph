"""Deterministic LSP setup primitives shared by MCP tools, plugins, and CLI."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

PERSISTENCE_DIR = ".metadata_astrograph"
LSP_BINDINGS_FILENAME = "lsp_bindings.json"
_SEMVER_RE = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
_VALIDATION_MODES = frozenset({"production", "bootstrap"})
_VALIDATION_MODE_ALIASES = {"relaxed": "bootstrap"}
_DEFAULT_VALIDATION_MODE = "production"
_COMPILE_COMMANDS_FILENAME = "compile_commands.json"
_COMPILE_COMMANDS_PATH_ENV = "ASTROGRAPH_COMPILE_COMMANDS_PATH"
_COMPILE_COMMANDS_SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".metadata_astrograph",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "node_modules",
        "dist",
        "build-cache",
    }
)

logger = logging.getLogger(__name__)

_LANGUAGE_VARIANT_POLICY: dict[str, dict[str, Any]] = {
    "python": {
        "supported": ["3.11", "3.12", "3.13", "3.14"],
        "best_effort": ["3.15"],
        "notes": (
            "Prefer project-specific interpreter bindings. "
            "For deterministic analysis, bind python to the interpreter that matches the repo."
        ),
    },
    "javascript_lsp": {
        "supported": [
            "TypeScript 5.x",
            "ES2021+",
            "Node.js 20 LTS",
            "Node.js 22 LTS",
            "Node.js 24 LTS",
        ],
        "best_effort": ["TypeScript 4.x", "Node.js 18 LTS"],
        "notes": (
            "Prefer Node LTS and a recent TypeScript toolchain for stable symbol extraction."
        ),
    },
    "typescript_lsp": {
        "supported": ["TypeScript 5.x", "Node.js 20 LTS", "Node.js 22 LTS", "Node.js 24 LTS"],
        "best_effort": ["TypeScript 4.x", "Node.js 18 LTS"],
        "notes": (
            "Shares typescript-language-server with javascript_lsp. "
            "Prefer Node LTS and a recent TypeScript toolchain."
        ),
    },
    "c_lsp": {
        "supported": ["C11", "C17", "C23"],
        "best_effort": ["C99"],
        "notes": "Requires compile flags consistent with the project build.",
    },
    "cpp_lsp": {
        "supported": ["C++17", "C++20", "C++23"],
        "best_effort": ["C++14"],
        "notes": (
            "Use a production C++ language server (for example clangd or ccls) "
            "with compile_commands.json for template/operator resolution and stronger "
            "semantic confidence."
        ),
    },
    "java_lsp": {
        "supported": ["Java 11", "Java 17", "Java 21", "Java 25"],
        "best_effort": ["Java 8"],
        "notes": "Prefer LTS JDKs for predictable language-server behavior.",
    },
    "go_lsp": {
        "supported": ["Go 1.21", "Go 1.22", "Go 1.23", "Go 1.24", "Go 1.25"],
        "best_effort": ["Go 1.20"],
        "notes": "gopls is the canonical Go LSP server; shipped with the Go toolchain.",
    },
}


@dataclass(frozen=True)
class LSPServerSpec:
    """Configuration of one LSP-backed language adapter."""

    language_id: str
    default_command: tuple[str, ...]
    probe_commands: tuple[tuple[str, ...], ...] = ()
    required: bool = True


def bundled_lsp_specs() -> tuple[LSPServerSpec, ...]:
    """Return built-in language plugin server specs."""
    return (
        LSPServerSpec(
            language_id="python",
            default_command=("tcp://127.0.0.1:2090",),
            probe_commands=(
                ("python", "-m", "pylsp"),
                ("python3", "-m", "pylsp"),
            ),
            required=False,
        ),
        LSPServerSpec(
            language_id="javascript_lsp",
            default_command=("tcp://127.0.0.1:2092",),
            probe_commands=(("typescript-language-server", "--version"),),
            required=False,
        ),
        LSPServerSpec(
            language_id="typescript_lsp",
            default_command=("tcp://127.0.0.1:2093",),
            probe_commands=(("typescript-language-server", "--version"),),
            required=False,
        ),
        LSPServerSpec(
            language_id="c_lsp",
            default_command=("tcp://127.0.0.1:2087",),
            probe_commands=(),
            required=False,
        ),
        LSPServerSpec(
            language_id="cpp_lsp",
            default_command=("tcp://127.0.0.1:2088",),
            probe_commands=(),
            required=False,
        ),
        LSPServerSpec(
            language_id="java_lsp",
            default_command=("tcp://127.0.0.1:2089",),
            probe_commands=(),
            required=False,
        ),
        LSPServerSpec(
            language_id="go_lsp",
            default_command=("tcp://127.0.0.1:2091",),
            probe_commands=(("gopls", "version"),),
            required=False,
        ),
    )


def get_lsp_spec(language_id: str) -> LSPServerSpec | None:
    """Return server spec for a language, when known."""
    for spec in bundled_lsp_specs():
        if spec.language_id == language_id:
            return spec
    return None


def language_variant_policy(language_id: str | None = None) -> dict[str, dict[str, Any]]:
    """Return supported source-language variant policy for one or all adapters."""
    if language_id:
        policy = _LANGUAGE_VARIANT_POLICY.get(language_id)
        return {language_id: dict(policy)} if isinstance(policy, dict) else {}
    return {key: dict(value) for key, value in _LANGUAGE_VARIANT_POLICY.items()}


def _normalized_validation_mode() -> str:
    """Resolve runtime validation mode for attach-based endpoints."""
    return _normalized_validation_mode_with_override(None)


def _normalize_validation_mode(value: str | None) -> str | None:
    """Normalize validation mode value and legacy aliases."""
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in _VALIDATION_MODES:
        return normalized
    alias = _VALIDATION_MODE_ALIASES.get(normalized)
    if alias in _VALIDATION_MODES:
        return alias
    return None


def _normalized_validation_mode_with_override(mode_override: str | None) -> str:
    """Resolve validation mode from explicit override, env, and defaults."""
    override = _normalize_validation_mode(mode_override)
    if override in _VALIDATION_MODES:
        return override

    mode = _normalize_validation_mode(os.getenv("ASTROGRAPH_LSP_VALIDATION_MODE"))
    if mode in _VALIDATION_MODES:
        return mode

    legacy_flag = os.getenv("ASTROGRAPH_LSP_PRODUCTION")
    if legacy_flag is not None:
        normalized = legacy_flag.strip().lower()
        if normalized in {"0", "false", "no", "off"}:
            return "bootstrap"
        if normalized in {"1", "true", "yes", "on"}:
            return "production"

    return _DEFAULT_VALIDATION_MODE


def _is_production_validation_mode(mode_override: str | None = None) -> bool:
    """Whether attach validation should fail closed for strict languages."""
    return _normalized_validation_mode_with_override(mode_override) == "production"


_PROBE_DOCUMENTS: dict[str, dict[str, str]] = {
    "cpp_lsp": {
        "lsp_language_id": "cpp",
        "suffix": "_probe.cpp",
        "source": (
            "class Greeter {\n"
            "public:\n"
            "  int greet(int value) { return value + 1; }\n"
            "};\n"
            "int helper(int value) { return value + 1; }\n"
        ),
    },
    "c_lsp": {
        "lsp_language_id": "c",
        "suffix": "_probe.c",
        "source": "int helper(int value) { return value + 1; }\n",
    },
    "java_lsp": {
        "lsp_language_id": "java",
        "suffix": "_probe.java",
        "source": "class Greeter {\n  int greet(int value) { return value + 1; }\n}\n",
    },
    "go_lsp": {
        "lsp_language_id": "go",
        "suffix": "_probe.go",
        "source": "package main\n\nfunc helper(value int) int { return value + 1 }\n",
    },
}


def _probe_document(language_id: str) -> dict[str, str]:
    """Return a tiny probe document per language for semantic verification."""
    if language_id in _PROBE_DOCUMENTS:
        return _PROBE_DOCUMENTS[language_id]
    return {
        "lsp_language_id": language_id,
        "suffix": "_probe.txt",
        "source": "int helper(int value) { return value + 1; }\n",
    }


def _probe_attach_lsp_semantics(
    *,
    language_id: str,
    command: Sequence[str],
    workspace: str | Path | None,
    timeout: float = 1.5,
) -> dict[str, Any]:
    """Run lightweight LSP initialize + documentSymbol probe over attach endpoints."""
    parsed = parse_command(command)
    endpoint = parse_attach_endpoint(parsed)
    if endpoint is None:
        return {
            "executed": False,
            "handshake_ok": False,
            "semantic_ok": False,
            "symbol_count": 0,
            "reason": "Command is not an attach endpoint.",
        }

    if endpoint["transport"] not in {"tcp", "unix"}:
        return {
            "executed": False,
            "handshake_ok": False,
            "semantic_ok": False,
            "symbol_count": 0,
            "reason": f"Unsupported attach transport '{endpoint['transport']}'.",
        }

    # Local import avoids module-load cycles (lsp_client imports lsp_setup helpers).
    from .languages.lsp_client import SocketLSPClient

    probe = _probe_document(language_id)
    workspace_root = _normalize_workspace_root(workspace)
    probe_path = workspace_root / PERSISTENCE_DIR / probe["suffix"]
    probe_path.parent.mkdir(parents=True, exist_ok=True)

    client = SocketLSPClient(parsed[0], request_timeout=timeout)
    symbols: list[Any] = []
    handshake_ok = False
    try:
        symbols = client.document_symbols(
            source=probe["source"],
            file_path=str(probe_path),
            language_id=probe["lsp_language_id"],
        )
        handshake_ok = bool(getattr(client, "_initialized", False))
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.debug("Attach LSP probe failed for %s: %s", parsed[0], exc)
    finally:
        # Avoid sending shutdown/exit to long-lived shared host servers.
        with contextlib.suppress(Exception):
            client.close(force=True)

    symbol_count = len(symbols)
    semantic_ok = handshake_ok and symbol_count > 0
    if semantic_ok:
        reason = "LSP initialize + documentSymbol probe succeeded."
    elif handshake_ok:
        reason = "LSP initialize succeeded but semantic probe returned no symbols."
    else:
        reason = "LSP initialize handshake failed."

    return {
        "executed": True,
        "handshake_ok": handshake_ok,
        "semantic_ok": semantic_ok,
        "symbol_count": symbol_count,
        "reason": reason,
    }


def _safe_resolve(path: Path) -> Path:
    """Resolve a path, returning it unchanged on OSError."""
    try:
        return path.resolve()
    except OSError:
        return path


def _compile_commands_paths(workspace: Path, *, max_depth: int = 4) -> list[Path]:
    """Discover candidate compile_commands.json files under the workspace."""
    roots: list[Path] = []
    priority = [
        workspace / _COMPILE_COMMANDS_FILENAME,
        workspace / "build" / _COMPILE_COMMANDS_FILENAME,
    ]
    for candidate in priority:
        if candidate.exists():
            roots.append(candidate)

    workspace_resolved = _safe_resolve(workspace)

    for current_root, dirnames, filenames in os.walk(workspace_resolved):
        current = Path(current_root)
        try:
            depth = len(current.relative_to(workspace_resolved).parts)
        except ValueError:
            continue

        if depth > max_depth:
            dirnames[:] = []
            continue

        dirnames[:] = [name for name in dirnames if name not in _COMPILE_COMMANDS_SKIP_DIRS]
        if _COMPILE_COMMANDS_FILENAME in filenames:
            roots.append(current / _COMPILE_COMMANDS_FILENAME)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in roots:
        resolved = _safe_resolve(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    return deduped


def _resolve_compile_commands_override(workspace: Path) -> Path | None:
    """Return an optional compile_commands override path from environment."""
    raw = os.getenv(_COMPILE_COMMANDS_PATH_ENV, "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return candidate


def _ordered_compile_commands_candidates(
    candidates: Sequence[Path], *, workspace: Path, override: Path | None
) -> list[Path]:
    """Order compile db candidates to prefer explicit, root, and recent paths."""
    workspace_resolved = _safe_resolve(workspace)
    override_resolved = _safe_resolve(override) if override is not None else None
    root_compile = _safe_resolve(workspace / _COMPILE_COMMANDS_FILENAME)
    build_compile = _safe_resolve(workspace / "build" / _COMPILE_COMMANDS_FILENAME)

    def _sort_key(path: Path) -> tuple[int, int, float, str]:
        resolved = _safe_resolve(path)
        if override_resolved is not None and resolved == override_resolved:
            priority = -1
        elif resolved == root_compile:
            priority = 0
        elif resolved == build_compile:
            priority = 1
        else:
            priority = 2

        try:
            depth = len(resolved.relative_to(workspace_resolved).parts)
        except ValueError:
            depth = 10_000

        try:
            modified = resolved.stat().st_mtime
        except OSError:
            modified = 0.0

        return (priority, depth, -modified, str(resolved))

    return sorted(candidates, key=_sort_key)


def _is_compile_commands_entry(entry: Any) -> bool:
    """Validate one compilation database entry shape."""
    if not isinstance(entry, dict):
        return False
    if "file" not in entry or "directory" not in entry:
        return False
    return not ("command" not in entry and "arguments" not in entry)


def _compile_commands_status(
    *,
    language_id: str,
    workspace: str | Path | None,
    compile_db_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any] | None:
    """Collect compile_commands.json visibility/validity for C/C++ adapters."""
    if language_id not in {"c_lsp", "cpp_lsp"}:
        return None

    workspace_root = _normalize_workspace_root(workspace)
    search_root = _normalize_workspace_root(project_root) if project_root else workspace_root
    required = language_id == "cpp_lsp"
    candidates = _compile_commands_paths(search_root)

    # Per-call compile_db_path takes priority over env override.
    explicit_db: Path | None = None
    if compile_db_path is not None:
        raw = Path(compile_db_path)
        explicit_db = raw if raw.is_absolute() else workspace_root / raw
    override = explicit_db or _resolve_compile_commands_override(workspace_root)
    if override is not None:
        candidates = [override, *candidates]
    existing = [path for path in candidates if path.exists()]
    ordered_existing = _ordered_compile_commands_candidates(
        existing,
        workspace=search_root,
        override=override,
    )

    result: dict[str, Any] = {
        "required": required,
        "present": bool(ordered_existing),
        "readable": False,
        "valid": False,
        "entry_count": 0,
        "selected_path": None,
        "paths": [],
        "reason": None,
    }

    for path in ordered_existing:
        try:
            rel = path.resolve().relative_to(search_root.resolve())
            rendered = str(rel)
        except (OSError, ValueError):
            rendered = str(path)
        result["paths"].append(rendered)

    if not ordered_existing:
        result["reason"] = (
            "No compile_commands.json found. Generate one via your build system "
            "(e.g. CMake with -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)."
        )
        return result

    for path in ordered_existing:
        try:
            payload = json.loads(path.read_text())
            result["readable"] = True
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

        if isinstance(payload, list):
            valid_entries = [entry for entry in payload if _is_compile_commands_entry(entry)]
            if valid_entries:
                result["valid"] = True
                result["entry_count"] = len(valid_entries)
                result["selected_path"] = str(path)
                result["reason"] = None
                return result

    if not result["readable"]:
        result["reason"] = "compile_commands.json exists but is unreadable."
    else:
        result["reason"] = (
            "compile_commands.json was found but has no valid entries with "
            "'directory', 'file', and 'command'/'arguments'."
        )

    return result


def _availability_validation(
    *,
    language_id: str,
    command: Sequence[str],
    probe: dict[str, Any],
    workspace: str | Path | None,
    validation_mode: str | None = None,
    compile_db_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate whether a discovered command should count as usable."""
    effective_mode = _normalized_validation_mode_with_override(validation_mode)
    transport = str(probe.get("transport", "subprocess"))
    endpoint_reachable = bool(probe.get("available", False))
    compile_status = _compile_commands_status(
        language_id=language_id,
        workspace=workspace,
        compile_db_path=compile_db_path,
        project_root=project_root,
    )

    verification: dict[str, Any] = {
        "mode": effective_mode,
        "state": "unavailable",
        "reason": "Command/endpoint is not reachable.",
        "endpoint_reachable": endpoint_reachable,
        "lsp_handshake": {
            "executed": False,
            "ok": False,
        },
        "semantic_probe": {
            "executed": False,
            "ok": False,
            "symbol_count": 0,
        },
        "compile_commands": compile_status,
    }

    if not endpoint_reachable:
        verification["version_status"] = {
            "state": "unknown",
            "reason": "Endpoint not reachable; version check skipped.",
        }
        return {
            "available": False,
            "verification": verification,
        }

    # Subprocess: executable-based reachability plus version-probe semantics.
    if transport == "subprocess":
        version_probe = _run_version_probe(language_id, command)
        version_status = _evaluate_version_status(
            language_id=language_id,
            detected=version_probe["detected"],
            probe_kind=version_probe["probe_kind"],
            transport=transport,
            available=True,
        )
        verification["endpoint_reachable"] = True
        verification["version_status"] = version_status
        version_state = str(version_status.get("state", "unknown"))

        if version_state == "unsupported" and _is_production_validation_mode(validation_mode):
            verification["state"] = "reachable_only"
            verification["reason"] = (
                f"Executable is available but version is unsupported: "
                f"{version_status.get('reason', 'version check failed')}."
            )
            return {"available": False, "verification": verification}

        verification["state"] = "verified"
        verification["reason"] = "Executable is available."
        if version_state not in {"unknown", "supported"}:
            verification["reason"] += f" Version status: {version_state}."
        return {
            "available": True,
            "verification": verification,
        }

    # Attach-mode endpoints don't have a subprocess to version-probe.
    verification["version_status"] = {
        "state": "unknown",
        "reason": "Attach transport; no version probe available.",
    }

    attach_probe = _probe_attach_lsp_semantics(
        language_id=language_id,
        command=command,
        workspace=workspace,
    )
    handshake_ok = bool(attach_probe.get("handshake_ok"))
    semantic_ok = bool(attach_probe.get("semantic_ok"))
    verification["lsp_handshake"] = {
        "executed": bool(attach_probe.get("executed")),
        "ok": handshake_ok,
        "reason": attach_probe.get("reason"),
    }
    verification["semantic_probe"] = {
        "executed": bool(attach_probe.get("executed")),
        "ok": semantic_ok,
        "symbol_count": int(attach_probe.get("symbol_count", 0)),
    }

    compile_ok = True
    if language_id in {"c_lsp", "cpp_lsp"} and isinstance(compile_status, dict):
        compile_ok = bool(compile_status.get("valid", False))

    verified = handshake_ok and semantic_ok and compile_ok
    if verified:
        verification["state"] = "verified"
        verification["reason"] = "Endpoint passed handshake, semantic probe, and compile-db checks."
        return {"available": True, "verification": verification}

    reason_parts: list[str] = []
    if not handshake_ok:
        reason_parts.append("LSP handshake failed")
    elif not semantic_ok:
        reason_parts.append("semantic probe failed")
    if language_id in {"c_lsp", "cpp_lsp"} and not compile_ok:
        reason_parts.append("compile_commands.json missing or invalid")

    verification["state"] = "reachable_only"
    verification["reason"] = (
        "; ".join(reason_parts) if reason_parts else "Endpoint is reachable only."
    )

    if language_id == "cpp_lsp" and _is_production_validation_mode(validation_mode):
        return {"available": False, "verification": verification}

    return {"available": True, "verification": verification}


def _parse_semver(text: str | None) -> tuple[int, int, int | None] | None:
    """Parse the first semantic version found in text."""
    if not text:
        return None
    match = _SEMVER_RE.search(text)
    if match is None:
        return None
    major = int(match.group(1))
    minor = int(match.group(2))
    patch = int(match.group(3)) if match.group(3) is not None else None
    return major, minor, patch


def _version_probe_candidates(
    language_id: str, command: Sequence[str]
) -> list[tuple[list[str], str]]:
    """Build safe version-probe command candidates."""
    parsed = parse_command(command)
    if not parsed or parse_attach_endpoint(parsed) is not None:
        return []

    candidates: list[tuple[list[str], str]] = [([*parsed, "--version"], "server")]

    if language_id in {"c_lsp", "cpp_lsp"}:
        candidates.append(([*parsed, "-version"], "server"))

    if language_id == "python" and len(parsed) >= 3 and parsed[1] == "-m" and parsed[2] == "pylsp":
        candidates.append(([parsed[0], "--version"], "runtime"))

    if language_id in {"javascript_lsp", "typescript_lsp"}:
        candidates.append((["node", "--version"], "runtime"))

    if language_id == "java_lsp":
        candidates.append((["java", "-version"], "runtime"))

    if language_id == "go_lsp":
        candidates.append((["go", "version"], "runtime"))

    deduped: list[tuple[list[str], str]] = []
    seen: set[tuple[str, ...]] = set()
    for candidate, kind in candidates:
        key = tuple(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((candidate, kind))
    return deduped


def _run_version_probe(language_id: str, command: Sequence[str]) -> dict[str, Any]:
    """Probe backend/runtime version text from a command when possible."""
    candidates = _version_probe_candidates(language_id, command)
    if not candidates:
        return {"detected": None, "probe_kind": None}

    for candidate, probe_kind in candidates:
        try:
            completed = subprocess.run(
                candidate,
                capture_output=True,
                text=True,
                check=False,
                timeout=1.5,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue

        combined = "\n".join(
            part.strip()
            for part in ((completed.stdout or ""), (completed.stderr or ""))
            if part.strip()
        ).strip()
        if not combined:
            continue

        first_line = combined.splitlines()[0].strip()
        if not first_line:
            continue
        return {"detected": first_line[:200], "probe_kind": probe_kind}

    return {"detected": None, "probe_kind": None}


def _evaluate_version_status(
    *,
    language_id: str,
    detected: str | None,
    probe_kind: str | None,
    transport: str,
    available: bool,
) -> dict[str, Any]:
    """Evaluate compatibility status for detected backend/runtime versions."""
    policy = _LANGUAGE_VARIANT_POLICY.get(language_id, {})
    if transport != "subprocess":
        return {
            "state": "unknown",
            "detected": None,
            "probe_kind": None,
            "reason": "Attach endpoint version cannot be inferred from inside the MCP container.",
            "variant_policy": dict(policy),
        }

    if not available:
        return {
            "state": "unknown",
            "detected": None,
            "probe_kind": None,
            "reason": "Command is unavailable, so version probing was skipped.",
            "variant_policy": dict(policy),
        }

    parsed = _parse_semver(detected)
    if parsed is None:
        return {
            "state": "unknown",
            "detected": detected,
            "probe_kind": probe_kind,
            "reason": "Version output did not expose a parseable semantic version.",
            "variant_policy": dict(policy),
        }

    major, minor, _patch = parsed
    state = "unknown"
    reason = "No explicit compatibility rule is defined for this adapter."

    if language_id == "python":
        if probe_kind == "server":
            if major > 1 or (major == 1 and minor >= 11):
                state = "supported"
                reason = "python-lsp-server version is within supported range (>=1.11)."
            else:
                state = "unsupported"
                reason = "python-lsp-server version is below supported baseline (1.11)."
        elif probe_kind == "runtime":
            if major == 3 and minor in {11, 12, 13, 14}:
                state = "supported"
                reason = "Python runtime matches the primary supported variants."
            elif major == 3 and minor == 15:
                state = "best_effort"
                reason = "Python 3.15 is currently treated as best-effort."
            else:
                state = "unsupported"
                reason = "Python runtime is outside the supported variant window."
    elif language_id in {"javascript_lsp", "typescript_lsp"}:
        if probe_kind == "runtime":
            # Node.js runtime version probing
            if major in {20, 22, 24}:
                state = "supported"
                reason = f"Node.js {major} LTS is within the supported range."
            elif major == 18:
                state = "best_effort"
                reason = "Node.js 18 LTS is accepted as best-effort."
            elif major >= 20:
                state = "supported"
                reason = f"Node.js {major} is within the supported range (>=20)."
            elif major >= 16:
                state = "best_effort"
                reason = f"Node.js {major} is accepted as best-effort."
            else:
                state = "unsupported"
                reason = "Node.js version is below the supported range."
        else:
            # typescript-language-server version probing
            if major >= 4:
                state = "supported"
                reason = "typescript-language-server major version is in supported range (>=4)."
            elif major == 3:
                state = "best_effort"
                reason = "typescript-language-server major version 3 is accepted as best-effort."
            else:
                state = "unsupported"
                reason = "typescript-language-server major version is below supported range."
    elif language_id in {"c_lsp", "cpp_lsp"}:
        detected_lower = (detected or "").lower()
        if "clangd" in detected_lower:
            if major >= 16:
                state = "supported"
                reason = "clangd major version is in supported range (>=16)."
            elif major == 15:
                state = "best_effort"
                reason = "clangd 15 is accepted as best-effort."
            else:
                state = "unsupported"
                reason = "clangd major version is below supported range."
        elif "ccls" in detected_lower:
            state = "best_effort"
            reason = (
                "ccls compatibility is treated as best-effort; rely on protocol and "
                "compile database verification."
            )
        else:
            state = "best_effort"
            reason = (
                "Non-clangd C/C++ language server detected; compatibility is best-effort "
                "and verified through protocol and compile database checks."
            )
    elif language_id == "java_lsp":
        if probe_kind == "runtime":
            # JDK version probing
            if major in {11, 17, 21, 25}:
                state = "supported"
                reason = f"Java {major} is a supported LTS version."
            elif major == 8:
                state = "best_effort"
                reason = "Java 8 is accepted as best-effort."
            elif major >= 11:
                state = "supported"
                reason = f"Java {major} is within the supported range (>=11)."
            else:
                state = "unsupported"
                reason = "Java version is below the supported range."
        elif major >= 1:
            state = "best_effort"
            reason = "Java adapter version probing is best-effort; rely on workspace JDK checks."
    elif language_id == "go_lsp":
        if probe_kind == "runtime":
            # Go toolchain version probing
            if major == 1 and minor in {21, 22, 23, 24, 25}:
                state = "supported"
                reason = f"Go 1.{minor} is within the supported range."
            elif major == 1 and minor == 20:
                state = "best_effort"
                reason = "Go 1.20 is accepted as best-effort."
            elif major == 1 and minor >= 21:
                state = "supported"
                reason = f"Go 1.{minor} is within the supported range (>=1.21)."
            else:
                state = "unsupported"
                reason = "Go version is below the supported range."
        elif major >= 0:
            # gopls version probing (gopls v0.16+)
            state = "best_effort"
            reason = "gopls version probing is best-effort; rely on protocol checks."

    return {
        "state": state,
        "detected": detected,
        "probe_kind": probe_kind,
        "reason": reason,
        "variant_policy": dict(policy),
    }


def _normalize_workspace_root(workspace: str | Path | None) -> Path:
    """Normalize workspace root from explicit value or runtime environment."""
    if workspace is not None:
        root = Path(workspace).expanduser()
        if root.exists():
            root = root.resolve()
        return root.parent if root.is_file() else root

    env_workspace = os.getenv("ASTROGRAPH_WORKSPACE")
    if env_workspace is not None:
        if env_workspace.strip():
            root = Path(env_workspace).expanduser()
            if root.exists():
                root = root.resolve()
            return root.parent if root.is_file() else root
        return Path.cwd().resolve()

    docker_workspace = Path("/workspace")
    if docker_workspace.is_dir():
        return docker_workspace

    for candidate in (os.getenv("PWD"), os.getcwd()):
        if candidate and candidate != "/":
            root = Path(candidate).expanduser()
            if root.exists():
                root = root.resolve()
            return root.parent if root.is_file() else root

    return Path.cwd().resolve()


def detect_workspace_root() -> Path:
    """Detect workspace root used for binding persistence."""
    return _normalize_workspace_root(None)


def lsp_bindings_path(workspace: str | Path | None = None) -> Path:
    """Path to persisted LSP binding overrides."""
    return _normalize_workspace_root(workspace) / PERSISTENCE_DIR / LSP_BINDINGS_FILENAME


def parse_command(command: Sequence[str] | str | None) -> list[str]:
    """Normalize command input into a tokenized argv list."""
    if command is None:
        return []

    if isinstance(command, str):
        return [part for part in shlex.split(command) if part]

    parsed: list[str] = []
    for part in command:
        text = str(part).strip()
        if text:
            parsed.append(text)
    return parsed


def format_command(command: Sequence[str]) -> str:
    """Return shell-safe command rendering for user-facing output."""
    return " ".join(shlex.quote(part) for part in parse_command(command))


def parse_attach_endpoint(command: Sequence[str] | str | None) -> dict[str, Any] | None:
    """Parse single-token attach commands like tcp://host:port or unix:///path."""
    parsed = parse_command(command)
    if len(parsed) != 1:
        return None

    token = parsed[0]
    url = urlparse(token)

    if url.scheme == "tcp":
        if not url.hostname or url.port is None:
            return None
        return {
            "transport": "tcp",
            "host": url.hostname,
            "port": int(url.port),
            "target": f"{url.hostname}:{int(url.port)}",
        }

    if url.scheme == "unix":
        raw_path = url.path or url.netloc
        if not raw_path:
            return None
        path = unquote(raw_path)
        return {
            "transport": "unix",
            "path": path,
            "target": path,
        }

    return None


def _probe_attach_endpoint(endpoint: dict[str, Any], timeout: float = 0.3) -> dict[str, Any]:
    """Probe attach endpoint reachability using short socket connect attempts."""
    transport = endpoint["transport"]

    try:
        if transport == "tcp":
            host = str(endpoint["host"])
            port = int(endpoint["port"])

            # Try all resolved addresses (IPv4/IPv6) so one failing family
            # does not mask a reachable endpoint.
            addrinfos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            connected = False
            for family, socktype, proto, _canonname, sockaddr in addrinfos:
                try:
                    with socket.socket(family, socktype, proto) as sock:
                        sock.settimeout(timeout)
                        sock.connect(sockaddr)
                    connected = True
                    break
                except OSError:
                    continue

            if not connected:
                return {
                    "available": False,
                    "executable": None,
                }
        elif transport == "unix":
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(timeout)
                sock.connect(str(endpoint["path"]))
        else:
            return {
                "available": False,
                "executable": None,
            }
    except OSError:
        return {
            "available": False,
            "executable": None,
        }

    return {
        "available": True,
        "executable": endpoint["target"],
    }


def load_lsp_bindings(workspace: str | Path | None = None) -> dict[str, list[str]]:
    """Load persisted LSP command bindings from disk."""
    path = lsp_bindings_path(workspace)
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    bindings: dict[str, list[str]] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        command = parse_command(value if isinstance(value, list | tuple | str) else None)
        if command:
            bindings[key] = command

    return bindings


def save_lsp_bindings(
    bindings: Mapping[str, Sequence[str] | str],
    workspace: str | Path | None = None,
) -> Path:
    """Persist LSP command bindings to disk, normalizing all commands."""
    normalized: dict[str, list[str]] = {}
    for language_id, command in bindings.items():
        parsed = parse_command(command)
        if parsed:
            normalized[str(language_id)] = parsed

    path = lsp_bindings_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    content = json.dumps(normalized, indent=2, sort_keys=True) + "\n"
    # Atomic write: write to temp file then rename to avoid corruption on interruption
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    fd_closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd_closed = True
        os.replace(tmp_path, str(path))
    except BaseException:
        if not fd_closed:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise
    return path


def _raise_empty_command_error() -> list[str]:
    """Raise the canonical validation error for empty commands."""
    raise ValueError("command cannot be empty")


def set_lsp_binding(
    language_id: str,
    command: Sequence[str] | str,
    workspace: str | Path | None = None,
) -> tuple[list[str], Path]:
    """Set and persist one LSP command binding."""
    parsed = parse_command(command) or _raise_empty_command_error()

    bindings = load_lsp_bindings(workspace)
    bindings[language_id] = parsed
    path = save_lsp_bindings(bindings, workspace)
    return parsed, path


def unset_lsp_binding(
    language_id: str,
    workspace: str | Path | None = None,
) -> tuple[bool, Path]:
    """Remove one persisted LSP command binding."""
    bindings = load_lsp_bindings(workspace)
    removed = language_id in bindings
    if removed:
        bindings.pop(language_id, None)
    path = save_lsp_bindings(bindings, workspace)
    return removed, path


def resolve_lsp_command(
    *,
    language_id: str,
    default_command: Sequence[str],
    workspace: str | Path | None = None,
) -> tuple[list[str], str]:
    """Resolve effective command with precedence: binding -> default."""
    bindings = load_lsp_bindings(workspace)
    bound = bindings.get(language_id)
    if bound:
        return list(bound), "binding"

    return parse_command(default_command), "default"


def probe_command(command: Sequence[str] | str | None) -> dict[str, Any]:
    """Check whether a command or attach endpoint is reachable."""
    parsed = parse_command(command)
    if not parsed:
        return {
            "command": [],
            "available": False,
            "executable": None,
        }

    endpoint = parse_attach_endpoint(parsed)
    if endpoint is not None:
        endpoint_probe = _probe_attach_endpoint(endpoint)
        return {
            "command": parsed,
            "available": endpoint_probe["available"],
            "executable": endpoint_probe["executable"],
            "transport": endpoint["transport"],
            "endpoint": endpoint["target"],
        }

    executable = shutil.which(parsed[0])
    return {
        "command": parsed,
        "available": executable is not None,
        "executable": executable,
        "transport": "subprocess",
    }


def collect_lsp_statuses(
    workspace: str | Path | None = None,
    *,
    validation_mode: str | None = None,
    compile_db_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Collect effective command status for each language adapter."""
    bindings = load_lsp_bindings(workspace)
    statuses: list[dict[str, Any]] = []
    for spec in bundled_lsp_specs():
        effective_command, effective_source = resolve_lsp_command(
            language_id=spec.language_id,
            default_command=spec.default_command,
            workspace=workspace,
        )

        # Fail fast: unconfigured languages (no binding)
        # are immediately marked unavailable — no probe attempted.
        if effective_source == "default":
            endpoint = parse_attach_endpoint(effective_command)
            statuses.append(
                {
                    "language": spec.language_id,
                    "required": spec.required,
                    "available": False,
                    "probe_available": False,
                    "executable": None,
                    "transport": endpoint["transport"] if endpoint else "subprocess",
                    "endpoint": endpoint["target"] if endpoint else None,
                    "effective_command": effective_command,
                    "effective_source": effective_source,
                    "binding_command": [],
                    "default_command": list(spec.default_command),
                    "version_status": {"state": "not_configured"},
                    "language_variant_policy": dict(
                        _LANGUAGE_VARIANT_POLICY.get(spec.language_id, {})
                    ),
                    "verification": {"state": "not_configured"},
                    "verification_state": "not_configured",
                    "validation_mode": _normalized_validation_mode_with_override(
                        validation_mode
                    ),
                    "compile_commands": None,
                }
            )
            continue

        probe = probe_command(effective_command)
        validation = _availability_validation(
            language_id=spec.language_id,
            command=effective_command,
            probe=probe,
            workspace=workspace,
            validation_mode=validation_mode,
            compile_db_path=compile_db_path,
            project_root=project_root,
        )
        verification = validation["verification"]
        # Reuse version_status already computed inside _availability_validation
        # to avoid spawning redundant subprocess probes.
        version_status = verification.get("version_status", {})
        statuses.append(
            {
                "language": spec.language_id,
                "required": spec.required,
                "available": bool(validation["available"]),
                "probe_available": bool(probe["available"]),
                "executable": probe["executable"],
                "transport": probe.get("transport", "subprocess"),
                "endpoint": probe.get("endpoint"),
                "effective_command": effective_command,
                "effective_source": effective_source,
                "binding_command": bindings.get(spec.language_id, []),
                "default_command": list(spec.default_command),
                "version_status": version_status,
                "language_variant_policy": dict(_LANGUAGE_VARIANT_POLICY.get(spec.language_id, {})),
                "verification": verification,
                "verification_state": verification.get("state", "unknown"),
                "validation_mode": verification.get("mode", _DEFAULT_VALIDATION_MODE),
                "compile_commands": verification.get("compile_commands"),
            }
        )

    return statuses


def _observed_candidates(
    *,
    language_id: str,
    observations: Iterable[dict[str, Any]] | None,
) -> list[list[str]]:
    """Extract normalized candidate commands from agent observations."""
    if observations is None:
        return []

    candidates: list[list[str]] = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        observed_language = observation.get("language") or observation.get("language_id")
        if observed_language != language_id:
            continue
        command = parse_command(observation.get("command"))
        if command:
            candidates.append(command)
    return candidates


def probe_candidates(
    spec: LSPServerSpec,
    *,
    workspace: str | Path | None = None,
    observations: Iterable[dict[str, Any]] | None = None,
    validation_mode: str | None = None,
    compile_db_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Probe candidate commands from observations, bindings, env, defaults, and heuristics."""
    seen: set[tuple[str, ...]] = set()
    candidates: list[dict[str, Any]] = []
    bindings = load_lsp_bindings(workspace)

    def add(source: str, command: Sequence[str] | str | None) -> None:
        parsed = parse_command(command)
        if not parsed:
            return
        key = tuple(parsed)
        if key in seen:
            return
        seen.add(key)
        probe = probe_command(parsed)
        validation = _availability_validation(
            language_id=spec.language_id,
            command=parsed,
            probe=probe,
            workspace=workspace,
            validation_mode=validation_mode,
            compile_db_path=compile_db_path,
            project_root=project_root,
        )
        verification = validation["verification"]
        candidates.append(
            {
                "source": source,
                "command": parsed,
                "available": bool(validation["available"]),
                "probe_available": bool(probe["available"]),
                "executable": probe["executable"],
                "transport": probe.get("transport", "subprocess"),
                "endpoint": probe.get("endpoint"),
                "verification": verification,
                "verification_state": verification.get("state", "unknown"),
                "validation_mode": verification.get("mode", _DEFAULT_VALIDATION_MODE),
                "compile_commands": verification.get("compile_commands"),
            }
        )

    for observed in _observed_candidates(language_id=spec.language_id, observations=observations):
        add("observation", observed)

    add("binding", bindings.get(spec.language_id, []))
    add("default", spec.default_command)
    for probe_command_tokens in spec.probe_commands:
        add("heuristic", probe_command_tokens)

    return candidates


def auto_bind_missing_servers(
    *,
    workspace: str | Path | None = None,
    observations: Iterable[dict[str, Any]] | None = None,
    languages: Iterable[str] | None = None,
    validation_mode: str | None = None,
    compile_db_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Auto-bind commands only for currently missing language adapters."""
    root = _normalize_workspace_root(workspace)
    bindings = load_lsp_bindings(root)
    language_scope = {
        str(language_id).strip() for language_id in (languages or []) if str(language_id).strip()
    }
    scoped = bool(language_scope)

    changes: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    probes: dict[str, list[dict[str, Any]]] = {}

    specs = [
        spec for spec in bundled_lsp_specs() if not scoped or spec.language_id in language_scope
    ]

    for spec in specs:
        effective_command, _source = resolve_lsp_command(
            language_id=spec.language_id,
            default_command=spec.default_command,
            workspace=root,
        )
        effective_probe = probe_command(effective_command)
        effective_validation = _availability_validation(
            language_id=spec.language_id,
            command=effective_command,
            probe=effective_probe,
            workspace=root,
            validation_mode=validation_mode,
            compile_db_path=compile_db_path,
            project_root=project_root,
        )
        if effective_validation["available"]:
            verification = effective_validation["verification"]
            probes[spec.language_id] = [
                {
                    "source": "effective",
                    "required": spec.required,
                    "command": effective_command,
                    "available": True,
                    "probe_available": bool(effective_probe["available"]),
                    "executable": effective_probe["executable"],
                    "transport": effective_probe.get("transport", "subprocess"),
                    "endpoint": effective_probe.get("endpoint"),
                    "verification": verification,
                    "verification_state": verification.get("state", "unknown"),
                    "validation_mode": verification.get("mode", _DEFAULT_VALIDATION_MODE),
                    "compile_commands": verification.get("compile_commands"),
                }
            ]
            continue

        language_candidates = probe_candidates(
            spec,
            workspace=root,
            observations=observations,
            validation_mode=validation_mode,
            compile_db_path=compile_db_path,
            project_root=project_root,
        )
        probes[spec.language_id] = language_candidates
        selected = next(
            (candidate for candidate in language_candidates if candidate["available"]), None
        )

        if selected is None:
            reachable_only = next(
                (
                    candidate
                    for candidate in language_candidates
                    if candidate.get("probe_available") and not candidate.get("available")
                ),
                None,
            )
            unresolved_reason = (
                str(reachable_only.get("verification", {}).get("reason") or "").strip()
                if isinstance(reachable_only, dict)
                else ""
            )
            if not unresolved_reason:
                unresolved_reason = (
                    "No reachable command/endpoint discovered. "
                    "Provide observations from host search."
                )
            unresolved.append(
                {
                    "language": spec.language_id,
                    "required": spec.required,
                    "reason": unresolved_reason,
                }
            )
            continue

        bindings[spec.language_id] = selected["command"]
        changes.append(
            {
                "language": spec.language_id,
                "required": spec.required,
                "command": selected["command"],
                "source": selected["source"],
                "executable": selected["executable"],
                "transport": selected.get("transport", "subprocess"),
                "endpoint": selected.get("endpoint"),
            }
        )

    bindings_path: str | None = None
    if changes:
        bindings_path = str(save_lsp_bindings(bindings, root))

    statuses = collect_lsp_statuses(
        root,
        validation_mode=validation_mode,
        compile_db_path=compile_db_path,
        project_root=project_root,
    )
    if scoped:
        statuses = [status for status in statuses if status["language"] in language_scope]

    return {
        "workspace": str(root),
        "bindings_path": bindings_path or str(lsp_bindings_path(root)),
        "scope_languages": sorted(language_scope) if scoped else [],
        "changes": changes,
        "unresolved": unresolved,
        "probes": probes,
        "statuses": statuses,
    }
