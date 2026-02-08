"""Deterministic LSP setup primitives shared by MCP tools, plugins, and CLI."""

from __future__ import annotations

import json
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

_LANGUAGE_VARIANT_POLICY: dict[str, dict[str, Any]] = {
    "python": {
        "supported": ["3.11", "3.12", "3.13"],
        "best_effort": ["3.14"],
        "notes": (
            "Prefer project-specific interpreter bindings. "
            "For deterministic analysis, bind python to the interpreter that matches the repo."
        ),
    },
    "javascript_lsp": {
        "supported": ["TypeScript 5.x", "ES2021+"],
        "best_effort": ["TypeScript 4.x"],
        "notes": (
            "Prefer Node LTS and a recent TypeScript toolchain for stable symbol extraction."
        ),
    },
    "c_lsp": {
        "supported": ["C11", "C17"],
        "best_effort": ["C99"],
        "notes": "Requires compile flags consistent with the project build.",
    },
    "cpp_lsp": {
        "supported": ["C++17", "C++20", "C++23"],
        "best_effort": ["C++14"],
        "notes": (
            "Use clangd with compile_commands.json for template/operator resolution and "
            "stronger semantic confidence."
        ),
    },
    "java_lsp": {
        "supported": ["Java 11", "Java 17", "Java 21"],
        "best_effort": ["Java 8"],
        "notes": "Prefer LTS JDKs for predictable language-server behavior.",
    },
}


@dataclass(frozen=True)
class LSPServerSpec:
    """Configuration of one bundled LSP-backed language adapter."""

    language_id: str
    command_env_var: str
    default_command: tuple[str, ...]
    probe_commands: tuple[tuple[str, ...], ...] = ()
    required: bool = True


def bundled_lsp_specs() -> tuple[LSPServerSpec, ...]:
    """Return built-in language plugin server specs."""
    return (
        LSPServerSpec(
            language_id="python",
            command_env_var="ASTROGRAPH_PY_LSP_COMMAND",
            default_command=("pylsp",),
            probe_commands=(
                ("python", "-m", "pylsp"),
                ("python3", "-m", "pylsp"),
            ),
        ),
        LSPServerSpec(
            language_id="javascript_lsp",
            command_env_var="ASTROGRAPH_JS_LSP_COMMAND",
            default_command=("typescript-language-server", "--stdio"),
            probe_commands=(),
        ),
        LSPServerSpec(
            language_id="c_lsp",
            command_env_var="ASTROGRAPH_C_LSP_COMMAND",
            default_command=("tcp://127.0.0.1:2087",),
            probe_commands=(),
            required=False,
        ),
        LSPServerSpec(
            language_id="cpp_lsp",
            command_env_var="ASTROGRAPH_CPP_LSP_COMMAND",
            default_command=("tcp://127.0.0.1:2088",),
            probe_commands=(),
            required=False,
        ),
        LSPServerSpec(
            language_id="java_lsp",
            command_env_var="ASTROGRAPH_JAVA_LSP_COMMAND",
            default_command=("tcp://127.0.0.1:2089",),
            probe_commands=(),
            required=False,
        ),
    )


def get_lsp_spec(language_id: str) -> LSPServerSpec | None:
    """Return bundled server spec for a language, when known."""
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
            if major == 3 and minor in {11, 12, 13}:
                state = "supported"
                reason = "Python runtime matches the primary supported variants."
            elif major == 3 and minor == 14:
                state = "best_effort"
                reason = "Python 3.14 is currently treated as best-effort."
            else:
                state = "unsupported"
                reason = "Python runtime is outside the supported variant window."
    elif language_id == "javascript_lsp":
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
        if major >= 16:
            state = "supported"
            reason = "clangd major version is in supported range (>=16)."
        elif major == 15:
            state = "best_effort"
            reason = "clangd 15 is accepted as best-effort."
        else:
            state = "unsupported"
            reason = "clangd major version is below supported range."
    elif language_id == "java_lsp" and major >= 1:
        state = "best_effort"
        reason = "Java adapter version probing is best-effort; rely on workspace JDK checks."

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
            if not addrinfos:
                return {
                    "available": False,
                    "executable": None,
                }

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
    path.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
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
    command_env_var: str,
    workspace: str | Path | None = None,
) -> tuple[list[str], str]:
    """Resolve effective command with precedence: binding -> env -> default."""
    bindings = load_lsp_bindings(workspace)
    bound = bindings.get(language_id)
    if bound:
        return list(bound), "binding"

    env_command = parse_command(os.getenv(command_env_var, ""))
    if env_command:
        return env_command, "env"

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


def collect_lsp_statuses(workspace: str | Path | None = None) -> list[dict[str, Any]]:
    """Collect effective command status for each bundled language adapter."""
    bindings = load_lsp_bindings(workspace)
    statuses: list[dict[str, Any]] = []
    for spec in bundled_lsp_specs():
        effective_command, effective_source = resolve_lsp_command(
            language_id=spec.language_id,
            default_command=spec.default_command,
            command_env_var=spec.command_env_var,
            workspace=workspace,
        )
        probe = probe_command(effective_command)
        version_probe = _run_version_probe(spec.language_id, effective_command)
        version_status = _evaluate_version_status(
            language_id=spec.language_id,
            detected=version_probe["detected"],
            probe_kind=version_probe["probe_kind"],
            transport=str(probe.get("transport", "subprocess")),
            available=bool(probe["available"]),
        )
        statuses.append(
            {
                "language": spec.language_id,
                "required": spec.required,
                "available": probe["available"],
                "executable": probe["executable"],
                "transport": probe.get("transport", "subprocess"),
                "endpoint": probe.get("endpoint"),
                "effective_command": effective_command,
                "effective_source": effective_source,
                "binding_command": bindings.get(spec.language_id, []),
                "env_command": parse_command(os.getenv(spec.command_env_var, "")),
                "default_command": list(spec.default_command),
                "version_status": version_status,
                "language_variant_policy": dict(_LANGUAGE_VARIANT_POLICY.get(spec.language_id, {})),
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
        candidates.append(
            {
                "source": source,
                "command": parsed,
                "available": probe["available"],
                "executable": probe["executable"],
                "transport": probe.get("transport", "subprocess"),
                "endpoint": probe.get("endpoint"),
            }
        )

    for observed in _observed_candidates(language_id=spec.language_id, observations=observations):
        add("observation", observed)

    add("binding", bindings.get(spec.language_id, []))
    add("env", os.getenv(spec.command_env_var, ""))
    add("default", spec.default_command)
    for probe_command_tokens in spec.probe_commands:
        add("heuristic", probe_command_tokens)

    return candidates


def auto_bind_missing_servers(
    *,
    workspace: str | Path | None = None,
    observations: Iterable[dict[str, Any]] | None = None,
    languages: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Auto-bind commands only for currently missing bundled language adapters."""
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
            command_env_var=spec.command_env_var,
            workspace=root,
        )
        effective_probe = probe_command(effective_command)
        if effective_probe["available"]:
            probes[spec.language_id] = [
                {
                    "source": "effective",
                    "required": spec.required,
                    "command": effective_command,
                    "available": True,
                    "executable": effective_probe["executable"],
                    "transport": effective_probe.get("transport", "subprocess"),
                    "endpoint": effective_probe.get("endpoint"),
                }
            ]
            continue

        language_candidates = probe_candidates(spec, workspace=root, observations=observations)
        probes[spec.language_id] = language_candidates
        selected = next(
            (candidate for candidate in language_candidates if candidate["available"]), None
        )

        if selected is None:
            unresolved.append(
                {
                    "language": spec.language_id,
                    "required": spec.required,
                    "reason": "No reachable command/endpoint discovered. Provide observations from host search.",
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

    statuses = collect_lsp_statuses(root)
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
