"""LSP setup tool logic extracted from tools.py.

Module-level functions that implement the LSP setup subsystem for
``CodeStructureTools``.  All functions are pure (no class state) — caller
threads in the workspace path and docker-runtime flag.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from .languages.registry import LanguageRegistry
from .lsp_setup import (
    auto_bind_missing_servers,
    bundled_lsp_specs,
    collect_lsp_statuses,
    format_command,
    get_lsp_spec,
    language_variant_policy,
    load_lsp_bindings,
    lsp_bindings_path,
    parse_attach_endpoint,
    parse_command,
    probe_command,
    set_lsp_binding,
    unset_lsp_binding,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Workspace resolution
# ---------------------------------------------------------------------------


def resolve_lsp_workspace(
    last_indexed_path: str | None,
    detect_startup_workspace: Any,
) -> Path:
    """Resolve workspace root used for LSP binding persistence.

    Parameters
    ----------
    last_indexed_path:
        Most recently indexed path (file or directory).
    detect_startup_workspace:
        Callable returning ``str | None`` — the fallback startup workspace.
    """
    if last_indexed_path:
        indexed = Path(last_indexed_path)
        return indexed.parent if indexed.is_file() else indexed

    detected = detect_startup_workspace()
    if detected:
        return Path(detected)

    return Path.cwd()


# ---------------------------------------------------------------------------
# Static helpers
# ---------------------------------------------------------------------------


def is_docker_runtime() -> bool:
    """Return whether the server appears to run inside Docker."""
    return Path("/.dockerenv").exists()


def default_install_command(language_id: str) -> list[str] | None:
    """Return a known install command for common language servers."""
    if language_id == "python":
        return ["python3", "-m", "pip", "install", "python-lsp-server>=1.11"]
    if language_id in ("javascript_lsp", "typescript_lsp"):
        return ["npm", "install", "-g", "typescript", "typescript-language-server"]
    return None


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """Return *values* with duplicates removed, preserving first occurrence."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def attach_candidate_commands(
    status: dict[str, Any],
    docker_runtime: bool,
) -> list[str]:
    """Build endpoint candidates for attach-based language plugins."""
    endpoint = parse_attach_endpoint(status.get("default_command"))
    if endpoint is None:
        endpoint = parse_attach_endpoint(status.get("effective_command"))
    if endpoint is None:
        return []

    if endpoint["transport"] == "unix":
        return [f"unix://{endpoint['path']}"]

    port = int(endpoint["port"])
    candidates = [
        f"tcp://127.0.0.1:{port}",
        f"tcp://localhost:{port}",
    ]
    if docker_runtime:
        candidates = [
            f"tcp://host.docker.internal:{port}",
            f"tcp://gateway.docker.internal:{port}",
            *candidates,
        ]
    return dedupe_preserve_order(candidates)


def server_bridge_info(language_id: str, candidate: str) -> dict[str, Any] | None:
    """Return structured bridge info for attach-based adapters."""
    endpoint = parse_attach_endpoint(candidate)
    if endpoint is None or endpoint["transport"] != "tcp":
        return None

    port = int(endpoint["port"])
    info: dict[str, Any] = {"requires": ["socat"], "port": port}
    if language_id in {"c_lsp", "cpp_lsp"}:
        info["server_binary"] = "clangd"
        info["server_binary_alternatives"] = ["ccls"]
        info["shared_with"] = "c_lsp" if language_id == "cpp_lsp" else "cpp_lsp"
        info[
            "socat_command"
        ] = f'socat "TCP-LISTEN:{port},bind=0.0.0.0,reuseaddr,fork" "EXEC:clangd",stderr'
        info["install_hints"] = {
            "macos": "brew install llvm",
            "linux": "apt-get install -y clangd",
        }
    elif language_id == "java_lsp":
        info["server_binary"] = "jdtls"
        info["socat_command"] = (
            f'socat "TCP-LISTEN:{port},bind=0.0.0.0,reuseaddr,fork" '
            '"EXEC:jdtls -data /tmp/jdtls-workspace",stderr'
        )
        info["install_hints"] = {
            "macos": "brew install jdtls",
            "linux": "See https://github.com/eclipse-jdtls/eclipse.jdt.ls#installation",
        }
    else:
        return None
    return info


# ---------------------------------------------------------------------------
# Recommended-actions builder
# ---------------------------------------------------------------------------


def build_lsp_recommended_actions(
    *,
    statuses: list[dict[str, Any]],
    scope_language: str | None = None,
    docker_runtime: bool = False,
) -> list[dict[str, Any]]:
    """Build actionable setup steps that AI agents can execute directly."""
    missing = [status for status in statuses if not status.get("available")]
    missing_required = [status for status in missing if status.get("required", True)]
    scoped = bool(scope_language)
    all_missing_optional = bool(missing) and not missing_required
    has_missing_cpp = any(str(status.get("language")) == "cpp_lsp" for status in missing)

    def _with_scope(arguments: dict[str, Any]) -> dict[str, Any]:
        if not scoped:
            return arguments
        return {**arguments, "language": scope_language}

    def _follow_up_auto_bind_arguments(language_id: str) -> dict[str, Any]:
        if scoped:
            return _with_scope({"mode": "auto_bind"})
        return {"mode": "auto_bind", "language": language_id}

    if not missing:
        return [
            {
                "id": "verify_lsp_setup",
                "priority": "low",
                "title": "Re-check language server health after environment changes",
                "tool": "astrograph_lsp_setup",
                "arguments": _with_scope({"mode": "inspect"}),
            }
        ]

    actions: list[dict[str, Any]] = []
    if not scoped and all_missing_optional and has_missing_cpp:
        actions.append(
            {
                "id": "focus_cpp_lsp",
                "priority": "high",
                "title": "Resolve cpp_lsp first",
                "why": (
                    "Multiple attach languages are unavailable. Resolve them one at a time, "
                    "starting with C++. Run the host_search_commands, then auto_bind."
                ),
                "tool": "astrograph_lsp_setup",
                "arguments": {"mode": "inspect", "language": "cpp_lsp"},
            }
        )
    actions.append(
        {
            "id": "auto_bind_missing",
            "priority": "high" if missing_required else "medium",
            "title": "Auto-bind any reachable language servers",
            "why": (
                f"{len(missing)} language server(s) are unreachable. Execute this action to probe "
                "all candidate endpoints and bind any that respond."
            ),
            "tool": "astrograph_lsp_setup",
            "arguments": _with_scope({"mode": "auto_bind"}),
        }
    )
    host_alias_action_added = False

    spec_by_language = {spec.language_id: spec for spec in bundled_lsp_specs()}  # noqa: F841
    for status in missing:
        language = str(status["language"])
        required = bool(status.get("required", True))
        priority = "high" if required else "medium"
        transport = str(status.get("transport", "subprocess"))
        verification = status.get("verification", {})
        if not isinstance(verification, dict):
            verification = {}
        verification_state = str(
            status.get("verification_state") or verification.get("state") or "unknown"
        )
        compile_commands = status.get("compile_commands")
        if not isinstance(compile_commands, dict):
            compile_commands = verification.get("compile_commands")
        if not isinstance(compile_commands, dict):
            compile_commands = None

        if transport == "subprocess":
            effective_command = parse_command(status.get("effective_command"))
            binary = effective_command[0] if effective_command else "<binary>"
            search_commands = [f"which {binary}"]
            if language == "python":
                search_commands.append("python3 -m pylsp --help")
            if language == "javascript_lsp":
                search_commands.extend(
                    [
                        "which typescript-language-server",
                        "npm list -g typescript-language-server",
                    ]
                )

            actions.append(
                {
                    "id": f"search_{language}",
                    "priority": priority,
                    "title": f"Search host for a working {language} command",
                    "language": language,
                    "host_search_commands": search_commands,
                    "follow_up_tool": "astrograph_lsp_setup",
                    "follow_up_arguments": _follow_up_auto_bind_arguments(language),
                }
            )

            if install_cmd := default_install_command(language):
                actions.append(
                    {
                        "id": f"install_{language}",
                        "priority": priority,
                        "title": f"Install missing {language} language server",
                        "language": language,
                        "shell_command": format_command(install_cmd),
                        "follow_up_tool": "astrograph_lsp_setup",
                        "follow_up_arguments": _follow_up_auto_bind_arguments(language),
                    }
                )

            continue

        if verification_state == "reachable_only":
            actions.append(
                {
                    "id": f"verify_{language}_protocol",
                    "priority": "high" if language == "cpp_lsp" else priority,
                    "title": f"Replace non-verified endpoint for {language}",
                    "language": language,
                    "note": (
                        verification.get("reason")
                        or "Endpoint accepts TCP connections but did not pass LSP verification."
                    ),
                    "host_search_commands": [
                        "which clangd",
                        "which ccls",
                        "clangd --version",
                        "ccls --version",
                    ]
                    if language in {"c_lsp", "cpp_lsp"}
                    else [],
                    "follow_up_tool": "astrograph_lsp_setup",
                    "follow_up_arguments": _follow_up_auto_bind_arguments(language),
                }
            )

        if (
            language in {"c_lsp", "cpp_lsp"}
            and compile_commands is not None
            and not bool(compile_commands.get("valid"))
        ):
            paths = compile_commands.get("paths")
            known_paths = [str(path) for path in paths] if isinstance(paths, list) else []
            actions.append(
                {
                    "id": f"ensure_compile_commands_{language}",
                    "priority": "high" if language == "cpp_lsp" else priority,
                    "title": f"Generate a valid compile_commands.json for {language}",
                    "language": language,
                    "why": (
                        compile_commands.get("reason")
                        or "C/C++ language servers require compile_commands.json for "
                        "production-grade type and operator resolution."
                    ),
                    "host_search_commands": [
                        "find . -maxdepth 4 -name compile_commands.json",
                    ],
                    "shell_commands": [
                        "cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                        "ln -sf build/compile_commands.json compile_commands.json",
                    ],
                    "known_paths": known_paths,
                    "follow_up_tool": "astrograph_lsp_setup",
                    "follow_up_arguments": _follow_up_auto_bind_arguments(language),
                }
            )

        if docker_runtime and not host_alias_action_added:
            actions.append(
                {
                    "id": "ensure_docker_host_alias",
                    "priority": "high",
                    "title": "Ensure Docker can reach host.docker.internal",
                    "why": (
                        "Attach endpoints for C/C++/Java usually run on the host. "
                        "Add a host alias so tcp://host.docker.internal:<port> is routable."
                    ),
                    "docker_run_args": [
                        "--add-host",
                        "host.docker.internal:host-gateway",
                    ],
                    "follow_up_tool": "astrograph_lsp_setup",
                    "follow_up_arguments": _with_scope({"mode": "inspect"}),
                }
            )
            host_alias_action_added = True

        candidates = attach_candidate_commands(status, docker_runtime)
        ep = parse_attach_endpoint(candidates[0]) if candidates else None
        port_hint = int(ep["port"]) if ep and ep["transport"] == "tcp" else None
        bridge = server_bridge_info(language, candidates[0]) if candidates else None
        endpoint_search_commands: list[str] = []
        if port_hint is not None:
            endpoint_search_commands = [
                f"lsof -nP -iTCP:{port_hint} -sTCP:LISTEN",
                f"ss -ltnp | rg ':{port_hint}'",
            ]
        if bridge:
            endpoint_search_commands.append(f"which {bridge['server_binary']}")
            follow_up_arguments: dict[str, Any] = _follow_up_auto_bind_arguments(language)
            if candidates:
                follow_up_arguments = {
                    **follow_up_arguments,
                    "observations": [{"language": language, "command": candidates[0]}],
                }
            actions.append(
                {
                    "id": f"install_start_{language}_bridge",
                    "priority": "high" if language == "cpp_lsp" else priority,
                    "title": f"Install and start bridge for {language}",
                    "why": (
                        f"Bring up a real {bridge['server_binary']} endpoint so auto_bind can "
                        "bind it immediately."
                    ),
                    "language": language,
                    "install_hints": bridge.get("install_hints", {}),
                    "host_search_commands": [f"which {bridge['server_binary']}"],
                    "shell_command": bridge.get("socat_command"),
                    "follow_up_tool": "astrograph_lsp_setup",
                    "follow_up_arguments": follow_up_arguments,
                }
            )

        action: dict[str, Any] = {
            "id": f"discover_{language}_endpoint",
            "priority": priority,
            "title": f"Discover a reachable attach endpoint for {language}",
            "why": (
                f"No server is listening on the expected {language} port. "
                "Run the host_search_commands on the host machine to check, "
                "then provide the working endpoint via auto_bind with observations."
            ),
            "language": language,
            "candidate_endpoints": candidates,
            "host_search_commands": endpoint_search_commands,
        }
        if candidates:
            follow_up_arguments = {
                **_follow_up_auto_bind_arguments(language),
                "observations": [{"language": language, "command": candidates[0]}],
            }

            action["follow_up_tool"] = "astrograph_lsp_setup"
            action["follow_up_arguments"] = follow_up_arguments
            if bridge:
                action["server_bridge_info"] = bridge
        actions.append(action)

    actions.append(
        {
            "id": "verify_lsp_setup",
            "priority": "high" if missing_required else "medium",
            "title": "Verify missing_required_languages is empty",
            "tool": "astrograph_lsp_setup",
            "arguments": _with_scope({"mode": "inspect"}),
        }
    )
    return actions


# ---------------------------------------------------------------------------
# Guidance injection
# ---------------------------------------------------------------------------


def inject_lsp_setup_guidance(
    payload: dict[str, Any],
    *,
    workspace: Path,
    docker_runtime: bool = False,
) -> None:
    """Attach agent-friendly setup guidance to lsp_setup responses."""
    statuses_value = payload.get("servers")
    if not isinstance(statuses_value, list):
        statuses_value = payload.get("statuses")
    statuses = (
        [status for status in statuses_value if isinstance(status, dict)] if statuses_value else []
    )
    if not statuses:
        statuses = collect_lsp_statuses(workspace)

    missing = [status["language"] for status in statuses if not status.get("available")]
    missing_required = [
        status["language"]
        for status in statuses
        if status.get("required", True) and not status.get("available")
    ]

    payload["missing_languages"] = missing
    payload["missing_required_languages"] = missing_required
    payload["bindings"] = load_lsp_bindings(workspace)
    payload["execution_context"] = "docker" if docker_runtime else "host"
    scope_language = payload.get("scope_language")
    if not isinstance(scope_language, str):
        scope_language = None

    recommended_actions = build_lsp_recommended_actions(
        statuses=statuses,
        scope_language=scope_language,
        docker_runtime=docker_runtime,
    )
    cpp_reachable_only = any(
        str(status.get("language")) == "cpp_lsp"
        and str(status.get("verification_state") or "") == "reachable_only"
        for status in statuses
    )
    scope_status = (
        next((status for status in statuses if status.get("language") == scope_language), None)
        if scope_language
        else None
    )
    attach_scope_ready = bool(
        scope_status
        and not missing
        and str(scope_status.get("transport", "subprocess")) in {"tcp", "unix"}
    )
    if attach_scope_ready and scope_language:
        recommended_actions.extend(
            [
                {
                    "id": f"verify_{scope_language}_analysis",
                    "priority": "medium",
                    "title": f"Run duplicate analysis after enabling {scope_language}",
                    "why": (
                        "Confirms the active index is healthy after LSP setup changes "
                        "and generates a current duplicate report."
                    ),
                    "tool": "astrograph_analyze",
                    "arguments": {"auto_reindex": True},
                },
                {
                    "id": f"rebaseline_after_{scope_language}_binding",
                    "priority": "medium",
                    "title": "If this language was bound after startup, rebuild baseline",
                    "why": (
                        "Binding changes do not retroactively update already-indexed files. "
                        "Recompute to apply extraction across the full workspace."
                    ),
                    "tool": "astrograph_metadata_recompute_baseline",
                    "arguments": {},
                    "note": (
                        "metadata_recompute_baseline clears lsp_bindings.json. "
                        "Run follow_up_steps immediately after recompute."
                    ),
                    "follow_up_steps": [
                        {
                            "tool": "astrograph_lsp_setup",
                            "arguments": {"mode": "inspect", "language": scope_language},
                        },
                        {
                            "tool": "astrograph_lsp_setup",
                            "arguments": {"mode": "auto_bind", "language": scope_language},
                        },
                        {
                            "tool": "astrograph_lsp_setup",
                            "arguments": {"mode": "inspect", "language": scope_language},
                        },
                    ],
                },
            ]
        )
    payload["recommended_actions"] = recommended_actions

    inspect_arguments: dict[str, Any] = {"mode": "inspect"}
    auto_bind_arguments: dict[str, Any] = {"mode": "auto_bind"}
    if scope_language:
        inspect_arguments["language"] = scope_language
        auto_bind_arguments["language"] = scope_language

    payload["resolution_loop"] = [
        {"tool": "astrograph_lsp_setup", "arguments": inspect_arguments},
        {"tool": "astrograph_lsp_setup", "arguments": auto_bind_arguments},
        {"tool": "astrograph_lsp_setup", "arguments": inspect_arguments},
    ]
    if missing_required:
        payload["agent_directive"] = (
            "Do not stop at diagnostics. Execute recommended_actions until "
            "missing_required_languages is empty."
        )
    elif cpp_reachable_only:
        payload["agent_directive"] = (
            "cpp_lsp endpoint is reachable but not production-verified. "
            "Treat it as unavailable and execute recommended_actions "
            "(real C++ LSP endpoint, such as clangd/ccls, + valid compile_commands.json)."
        )
    elif missing:
        missing_names = ", ".join(missing)
        payload["agent_directive"] = (
            f"{len(missing)} attach language(s) ({missing_names}) are unavailable. "
            "Execute recommended_actions now: run host_search_commands; when missing, run "
            "install/start actions; then call auto_bind with observations and finish with inspect."
        )
    elif attach_scope_ready and scope_language:
        payload["agent_directive"] = (
            f"{scope_language} endpoint is reachable. "
            "Run recommended_actions to verify analysis coverage. "
            "If you recompute baseline, re-run auto_bind for this language afterward."
        )
    else:
        payload["agent_directive"] = (
            "All configured language servers are reachable. "
            "Re-run inspect after environment changes."
        )

    if missing and docker_runtime:
        payload["observation_note"] = (
            "This MCP server runs inside Docker. When providing observations, "
            "use host.docker.internal (not 127.0.0.1) to reach host-side servers."
        )


# ---------------------------------------------------------------------------
# Result serializer
# ---------------------------------------------------------------------------


def lsp_setup_result_json(payload: dict[str, Any]) -> str:
    """Serialize structured LSP setup responses as JSON."""
    return json.dumps(payload, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


def handle_lsp_setup(
    *,
    workspace: Path,
    mode: str = "inspect",
    language: str | None = None,
    command: Sequence[str] | str | None = None,
    observations: list[dict[str, Any]] | None = None,
    validation_mode: str | None = None,
    compile_db_path: str | None = None,
    project_root: str | None = None,
) -> str:
    """Core lsp_setup logic.  Returns JSON string for the caller to wrap."""
    docker_rt = is_docker_runtime()
    normalized_mode = (mode or "inspect").strip().lower()
    known_languages = [
        spec.language_id for spec in sorted(bundled_lsp_specs(), key=lambda s: s.language_id)
    ]

    def _result(payload: dict[str, Any]) -> str:
        return lsp_setup_result_json(payload)

    def _validate_language(mode_name: str, *, required: bool = False) -> str | None:
        if not language:
            if required:
                return _result(
                    {
                        "ok": False,
                        "mode": mode_name,
                        "error": f"'language' is required when mode='{mode_name}'",
                        "supported_languages": known_languages,
                    }
                )
            return None

        if get_lsp_spec(language) is None:
            return _result(
                {
                    "ok": False,
                    "mode": mode_name,
                    "error": f"Unsupported language '{language}'",
                    "supported_languages": known_languages,
                }
            )

        return None

    def _status_for_language(language_id: str) -> dict[str, Any] | None:
        return next(
            (
                current
                for current in collect_lsp_statuses(workspace)
                if current["language"] == language_id
            ),
            None,
        )

    if normalized_mode == "inspect":
        if validation_error := _validate_language(normalized_mode):
            return validation_error

        statuses = collect_lsp_statuses(
            workspace,
            validation_mode=validation_mode,
            compile_db_path=compile_db_path,
            project_root=project_root,
        )
        if language:
            statuses = [status for status in statuses if status["language"] == language]
        missing = [status["language"] for status in statuses if not status["available"]]
        missing_required = [
            status["language"]
            for status in statuses
            if status.get("required", True) and not status["available"]
        ]
        cpp_reachable_only = any(
            str(status.get("language")) == "cpp_lsp"
            and str(status.get("verification_state") or "") == "reachable_only"
            for status in statuses
        )
        payload: dict[str, Any] = {
            "ok": True,
            "mode": normalized_mode,
            "workspace": str(workspace),
            "bindings_path": str(lsp_bindings_path(workspace)),
            "bindings": load_lsp_bindings(workspace),
            "servers": statuses,
            "missing_languages": missing,
            "missing_required_languages": missing_required,
            "supported_languages": known_languages,
            "language_variant_policy": (
                language_variant_policy(language) if language else language_variant_policy()
            ),
        }
        if language:
            payload["scope_language"] = language

        if missing_required:
            if language:
                payload["next_step"] = (
                    "Call astrograph_lsp_setup with mode='auto_bind' and this language. "
                    "If still missing, provide observations with language + command."
                )
            else:
                payload["next_step"] = (
                    "Call astrograph_lsp_setup with mode='auto_bind'. "
                    "If still missing, provide observations with language + command."
                )
        elif cpp_reachable_only:
            payload["next_step"] = (
                "cpp_lsp is reachable but not production-verified. "
                "Use a real C++ LSP endpoint (for example clangd/ccls) and a valid "
                "compile_commands.json, "
                "then re-run auto_bind and inspect."
            )
        elif missing:
            if not language and "cpp_lsp" in missing:
                payload["next_step"] = (
                    f"{len(missing)} attach endpoint(s) unreachable. "
                    "Start by running the host_search_commands from recommended_actions "
                    "to check if language servers are already running on the host. "
                    "If missing, run install/start actions from recommended_actions. "
                    "Then call astrograph_lsp_setup(mode='auto_bind') with discovered endpoints as observations."
                )
            else:
                payload["next_step"] = (
                    f"{len(missing)} attach endpoint(s) unreachable. "
                    "Run host_search_commands, run install/start actions when needed, then call "
                    "astrograph_lsp_setup(mode='auto_bind') with results as observations."
                )

        if missing:
            payload["observation_format"] = {
                "language": language or "cpp_lsp",
                "command": "tcp://127.0.0.1:2088",
            }
            payload["observation_examples"] = [
                {
                    "language": "python",
                    "command": ["/absolute/path/to/pylsp"],
                },
                {
                    "language": "java_lsp",
                    "command": "tcp://127.0.0.1:2089",
                },
            ]
            if not language and "cpp_lsp" in missing:
                payload["focus_hint"] = (
                    "Start by resolving one language at a time. "
                    "Run astrograph_lsp_setup(mode='inspect', language='cpp_lsp') "
                    "and execute its recommended_actions before moving to the next."
                )
        if cpp_reachable_only:
            payload["production_gate"] = {
                "language": "cpp_lsp",
                "state": "reachable_only",
                "requirement": (
                    "In production mode, cpp_lsp is treated as unavailable until endpoint "
                    "handshake, semantic probe, and compile_commands.json checks pass."
                ),
            }
        inject_lsp_setup_guidance(payload, workspace=workspace, docker_runtime=docker_rt)
        return _result(payload)

    if normalized_mode == "auto_bind":
        if validation_error := _validate_language(normalized_mode):
            return validation_error

        scope_languages = [language] if language else None
        outcome = auto_bind_missing_servers(
            workspace=workspace,
            observations=observations,
            languages=scope_languages,
            validation_mode=validation_mode,
            compile_db_path=compile_db_path,
            project_root=project_root,
        )
        if outcome["changes"]:
            LanguageRegistry.reset()
        outcome["bindings"] = load_lsp_bindings(workspace)
        outcome.update(
            {
                "ok": True,
                "mode": normalized_mode,
                "supported_languages": known_languages,
            }
        )
        if language:
            outcome["scope_language"] = language
        inject_lsp_setup_guidance(outcome, workspace=workspace, docker_runtime=docker_rt)
        return _result(outcome)

    if normalized_mode == "bind":
        if validation_error := _validate_language(normalized_mode, required=True):
            return validation_error

        target_language = cast(str, language)
        parsed_command = parse_command(command)
        if not parsed_command:
            return _result(
                {
                    "ok": False,
                    "mode": normalized_mode,
                    "error": "'command' must be a non-empty string or array",
                }
            )

        saved_command, path = set_lsp_binding(target_language, parsed_command, workspace)
        LanguageRegistry.reset()

        status = _status_for_language(target_language)
        probe = probe_command(saved_command)
        payload = {
            "ok": True,
            "mode": normalized_mode,
            "language": target_language,
            "workspace": str(workspace),
            "bindings_path": str(path),
            "bindings": load_lsp_bindings(workspace),
            "saved_command": saved_command,
            "saved_command_text": format_command(saved_command),
            "available": probe["available"],
            "executable": probe["executable"],
            "status": status,
        }
        inject_lsp_setup_guidance(payload, workspace=workspace, docker_runtime=docker_rt)
        return _result(payload)

    if normalized_mode == "unbind":
        if validation_error := _validate_language(normalized_mode, required=True):
            return validation_error

        target_language = cast(str, language)
        removed, path = unset_lsp_binding(target_language, workspace)
        LanguageRegistry.reset()
        status = _status_for_language(target_language)
        payload = {
            "ok": True,
            "mode": normalized_mode,
            "language": target_language,
            "workspace": str(workspace),
            "bindings_path": str(path),
            "bindings": load_lsp_bindings(workspace),
            "removed": removed,
            "status": status,
        }
        inject_lsp_setup_guidance(payload, workspace=workspace, docker_runtime=docker_rt)
        return _result(payload)

    return _result(
        {
            "ok": False,
            "mode": normalized_mode,
            "error": "Invalid mode",
            "supported_modes": ["inspect", "auto_bind", "bind", "unbind"],
            "supported_languages": known_languages,
        }
    )
