"""Subprocess JSON-RPC client for Language Server Protocol backends."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ._lsp_base import LSPClient, LSPPosition, LSPRange, LSPSymbol

logger = logging.getLogger(__name__)


class SubprocessLSPClient(LSPClient):
    """Minimal synchronous LSP client over stdio JSON-RPC framing."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        request_timeout: float = 5.0,
        initialization_options: dict[str, Any] | None = None,
    ) -> None:
        self._command = tuple(command)
        self._request_timeout = request_timeout
        self._initialization_options = initialization_options or {}

        self._proc: subprocess.Popen[bytes] | None = None
        self._initialized = False
        self._disabled = False

        self._next_id = 1
        self._lock = threading.RLock()

    @classmethod
    def from_command_string(
        cls,
        command: str,
        *,
        request_timeout: float = 5.0,
        initialization_options: dict[str, Any] | None = None,
    ) -> SubprocessLSPClient:
        """Create a client from a shell-like command string."""
        return cls(
            shlex.split(command),
            request_timeout=request_timeout,
            initialization_options=initialization_options,
        )

    def _path_to_uri(self, file_path: str) -> str:
        if file_path.startswith("<") and file_path.endswith(">"):
            return "untitled:astrograph"

        path = Path(file_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

        try:
            return path.as_uri()
        except ValueError:
            return f"file://{path.as_posix()}"

    def _start_process(self) -> bool:
        if self._disabled:
            return False

        if self._proc and self._proc.poll() is None:
            return True

        if not self._command:
            self._disabled = True
            logger.warning("Cannot start LSP process: empty command")
            return False

        try:
            self._proc = subprocess.Popen(
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._disabled = True
            logger.warning("Failed to start LSP process %s: %s", self._command, exc)
            return False

        self._initialized = False
        self._next_id = 1
        return True

    def _send_message(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("LSP process stdin is not available")

        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")

        self._proc.stdin.write(header)
        self._proc.stdin.write(body)
        self._proc.stdin.flush()

    def _read_message(self, timeout: float) -> dict[str, Any] | None:
        if self._proc is None or self._proc.stdout is None:
            return None

        stdout = self._proc.stdout
        deadline = time.monotonic() + timeout

        headers: dict[str, str] = {}
        while True:
            if time.monotonic() > deadline:
                return None

            raw = stdout.readline()
            if not raw:
                return None

            line = raw.decode("ascii", errors="replace").strip()
            if not line:
                break

            if ":" not in line:
                continue

            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()

        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            return None

        payload = stdout.read(content_length)
        if len(payload) != content_length:
            return None

        decoded = json.loads(payload.decode("utf-8"))
        return decoded if isinstance(decoded, dict) else None

    def _request(
        self,
        method: str,
        params: dict[str, Any] | None,
        *,
        timeout: float | None = None,
        allow_uninitialized: bool = False,
    ) -> Any:
        with self._lock:
            if not allow_uninitialized and not self._initialized:
                raise RuntimeError("LSP client is not initialized")

            request_id = self._next_id
            self._next_id += 1

            payload: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }
            if params is not None:
                payload["params"] = params

            self._send_message(payload)

            effective_timeout = timeout if timeout is not None else self._request_timeout
            deadline = time.monotonic() + effective_timeout

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for LSP response to {method}")

                message = self._read_message(remaining)
                if message is None:
                    raise TimeoutError(f"No response from LSP server for {method}")

                if "id" not in message:
                    # Notification from server; ignore.
                    continue

                if message.get("id") != request_id:
                    # Unrelated response; this client is strictly sequential.
                    continue

                if "error" in message:
                    raise RuntimeError(f"LSP error in {method}: {message['error']}")
                return message.get("result")

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            payload["params"] = params

        with self._lock:
            self._send_message(payload)

    def _initialize(self, file_path: str) -> bool:
        root_uri = self._path_to_uri(str(Path(file_path).parent))

        init_params = {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "documentSymbol": {
                        "hierarchicalDocumentSymbolSupport": True,
                    }
                }
            },
            "initializationOptions": self._initialization_options,
            "clientInfo": {
                "name": "astrograph",
            },
        }

        try:
            self._request(
                "initialize",
                init_params,
                timeout=self._request_timeout,
                allow_uninitialized=True,
            )
            self._notify("initialized", {})
        except Exception as exc:
            logger.debug("Failed to initialize LSP process %s: %s", self._command, exc)
            self.close(force=True)
            return False

        self._initialized = True
        return True

    def _parse_position(self, data: Any) -> LSPPosition:
        if not isinstance(data, dict):
            return LSPPosition(line=0, character=0)

        line = data.get("line", 0)
        character = data.get("character", 0)
        return LSPPosition(line=int(line), character=int(character))

    def _parse_range(self, data: Any) -> LSPRange:
        if not isinstance(data, dict):
            return LSPRange(start=LSPPosition(0, 0), end=LSPPosition(0, 0))

        return LSPRange(
            start=self._parse_position(data.get("start")),
            end=self._parse_position(data.get("end")),
        )

    def _parse_symbol(self, data: Any) -> LSPSymbol | None:
        if not isinstance(data, dict):
            return None

        name = str(data.get("name", "<anonymous>"))
        kind = int(data.get("kind", 0))

        # DocumentSymbol shape.
        if "range" in data:
            symbol_range = self._parse_range(data.get("range"))
            children_raw = data.get("children", [])
            children: list[LSPSymbol] = []
            if isinstance(children_raw, list):
                for child in children_raw:
                    parsed = self._parse_symbol(child)
                    if parsed is not None:
                        children.append(parsed)
            return LSPSymbol(
                name=name,
                kind=kind,
                symbol_range=symbol_range,
                children=tuple(children),
            )

        # SymbolInformation shape.
        location = data.get("location", {})
        symbol_range = self._parse_range(location.get("range"))
        return LSPSymbol(name=name, kind=kind, symbol_range=symbol_range)

    def _parse_symbols_result(self, result: Any) -> list[LSPSymbol]:
        if not isinstance(result, list):
            return []

        symbols: list[LSPSymbol] = []
        for item in result:
            parsed = self._parse_symbol(item)
            if parsed is not None:
                symbols.append(parsed)
        return symbols

    def document_symbols(
        self,
        *,
        source: str,
        file_path: str,
        language_id: str,
    ) -> list[LSPSymbol]:
        if self._disabled:
            return []

        if not self._start_process():
            return []

        if not self._initialized and not self._initialize(file_path):
            return []

        uri = self._path_to_uri(file_path)

        did_open_params = {
            "textDocument": {
                "uri": uri,
                "languageId": language_id,
                "version": 1,
                "text": source,
            }
        }
        request_params = {"textDocument": {"uri": uri}}

        try:
            self._notify("textDocument/didOpen", did_open_params)
            result = self._request("textDocument/documentSymbol", request_params)
            return self._parse_symbols_result(result)
        except Exception as exc:
            logger.debug("LSP document symbols failed for %s: %s", file_path, exc)
            return []
        finally:
            with contextlib.suppress(Exception):
                self._notify("textDocument/didClose", {"textDocument": {"uri": uri}})

    def close(self, *, force: bool = False) -> None:
        with self._lock:
            if self._proc is None:
                return

            if not force and self._initialized:
                try:
                    self._request(
                        "shutdown",
                        None,
                        timeout=min(self._request_timeout, 1.0),
                        allow_uninitialized=True,
                    )
                    self._notify("exit")
                except Exception:
                    pass

            proc = self._proc
            self._proc = None
            self._initialized = False

            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)

    def __enter__(self) -> SubprocessLSPClient:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()


def create_subprocess_client_from_env(
    *,
    default_command: Sequence[str],
    command_env_var: str,
    timeout_env_var: str,
    default_timeout: float = 5.0,
) -> SubprocessLSPClient:
    """Create a subprocess LSP client with env-based command/timeout overrides."""
    command_text = os.getenv(command_env_var, "")
    command = shlex.split(command_text) if command_text.strip() else list(default_command)

    timeout_text = os.getenv(timeout_env_var, str(default_timeout))
    try:
        timeout = float(timeout_text)
    except ValueError:
        timeout = default_timeout

    return SubprocessLSPClient(command, request_timeout=max(timeout, 0.1))
