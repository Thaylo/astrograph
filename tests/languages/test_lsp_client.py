"""Tests for the subprocess LSP client implementation."""

from __future__ import annotations

import json
import socket
import sys
import threading
from pathlib import Path

from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin
from astrograph.languages.lsp_client import SocketLSPClient, SubprocessLSPClient

_FAKE_LSP_SERVER = r"""
import json
import sys

_documents = {}


def _read_message():
    headers = {}
    while True:
        raw = sys.stdin.buffer.readline()
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

    payload = sys.stdin.buffer.read(content_length)
    if len(payload) != content_length:
        return None

    return json.loads(payload.decode("utf-8"))


def _write_message(payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    sys.stdout.buffer.write(header)
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def _document_symbols_for_text(text):
    symbols = []

    if "class Greeter" in text:
        symbols.append(
            {
                "name": "Greeter",
                "kind": 5,
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 4, "character": 1},
                },
                "children": [
                    {
                        "name": "greet",
                        "kind": 6,
                        "range": {
                            "start": {"line": 1, "character": 2},
                            "end": {"line": 3, "character": 3},
                        },
                    }
                ],
            }
        )

    if "function helper" in text:
        symbols.append(
            {
                "name": "helper",
                "kind": 12,
                "range": {
                    "start": {"line": 6, "character": 0},
                    "end": {"line": 8, "character": 1},
                },
            }
        )

    return symbols


while True:
    message = _read_message()
    if message is None:
        break

    method = message.get("method")

    if method == "initialize":
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": {
                    "capabilities": {
                        "documentSymbolProvider": True,
                    }
                },
            }
        )
    elif method == "initialized":
        continue
    elif method == "textDocument/didOpen":
        text_document = message.get("params", {}).get("textDocument", {})
        _documents[text_document.get("uri", "")] = text_document.get("text", "")
    elif method == "textDocument/didClose":
        text_document = message.get("params", {}).get("textDocument", {})
        _documents.pop(text_document.get("uri", ""), None)
    elif method == "textDocument/documentSymbol":
        uri = message.get("params", {}).get("textDocument", {}).get("uri", "")
        text = _documents.get(uri, "")
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": _document_symbols_for_text(text),
            }
        )
    elif method == "shutdown":
        _write_message(
            {
                "jsonrpc": "2.0",
                "id": message.get("id"),
                "result": None,
            }
        )
    elif method == "exit":
        break
"""


def _write_fake_lsp_server(tmp_path: Path) -> Path:
    server_script = tmp_path / "fake_lsp_server.py"
    server_script.write_text(_FAKE_LSP_SERVER)
    return server_script


def _read_wire_message(stream: socket.SocketIO) -> dict | None:
    headers: dict[str, str] = {}
    while True:
        raw = stream.readline()
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

    payload = stream.read(content_length)
    if len(payload) != content_length:
        return None

    decoded = json.loads(payload.decode("utf-8"))
    return decoded if isinstance(decoded, dict) else None


def _write_wire_message(stream: socket.SocketIO, payload: dict) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
    stream.write(header)
    stream.write(body)
    stream.flush()


def _socket_document_symbols(text: str) -> list[dict]:
    symbols: list[dict] = []

    if "class Greeter" in text:
        symbols.append(
            {
                "name": "Greeter",
                "kind": 5,
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 4, "character": 1},
                },
                "children": [
                    {
                        "name": "greet",
                        "kind": 6,
                        "range": {
                            "start": {"line": 1, "character": 2},
                            "end": {"line": 3, "character": 3},
                        },
                    }
                ],
            }
        )

    if "function helper" in text:
        symbols.append(
            {
                "name": "helper",
                "kind": 12,
                "range": {
                    "start": {"line": 6, "character": 0},
                    "end": {"line": 8, "character": 1},
                },
            }
        )

    return symbols


def _run_fake_socket_lsp(server: socket.socket) -> None:
    documents: dict[str, str] = {}
    connection, _address = server.accept()
    with connection:
        reader = connection.makefile("rb")
        writer = connection.makefile("wb")

        while True:
            message = _read_wire_message(reader)
            if message is None:
                break

            method = message.get("method")
            if method == "initialize":
                _write_wire_message(
                    writer,
                    {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "result": {"capabilities": {"documentSymbolProvider": True}},
                    },
                )
            elif method == "initialized":
                continue
            elif method == "textDocument/didOpen":
                text_document = message.get("params", {}).get("textDocument", {})
                if isinstance(text_document, dict):
                    documents[str(text_document.get("uri", ""))] = str(
                        text_document.get("text", "")
                    )
            elif method == "textDocument/didClose":
                text_document = message.get("params", {}).get("textDocument", {})
                if isinstance(text_document, dict):
                    documents.pop(str(text_document.get("uri", "")), None)
            elif method == "textDocument/documentSymbol":
                uri = message.get("params", {}).get("textDocument", {}).get("uri", "")
                symbols = _socket_document_symbols(documents.get(str(uri), ""))
                _write_wire_message(
                    writer,
                    {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "result": symbols,
                    },
                )
            elif method == "shutdown":
                _write_wire_message(
                    writer,
                    {
                        "jsonrpc": "2.0",
                        "id": message.get("id"),
                        "result": None,
                    },
                )
            elif method == "exit":
                break


class TestSubprocessLSPClient:
    def test_document_symbols_with_fake_server(self, tmp_path):
        server_script = _write_fake_lsp_server(tmp_path)
        client = SubprocessLSPClient([sys.executable, str(server_script)], request_timeout=2.0)

        source = """
class Greeter {
  greet(name) {
    return name;
  }
}

function helper(value) {
  return value;
}
"""
        symbols = client.document_symbols(
            source=source,
            file_path=str(tmp_path / "sample.js"),
            language_id="javascript",
        )

        assert [symbol.name for symbol in symbols] == ["Greeter", "helper"]
        assert symbols[0].children
        assert symbols[0].children[0].name == "greet"

        client.close()

    def test_javascript_plugin_uses_subprocess_client(self, tmp_path):
        server_script = _write_fake_lsp_server(tmp_path)
        lsp_client = SubprocessLSPClient([sys.executable, str(server_script)], request_timeout=2.0)
        plugin = JavaScriptLSPPlugin(lsp_client=lsp_client)

        source = """
class Greeter {
  greet(name) {
    return name;
  }
}

function helper(value) {
  return value;
}
"""
        units = list(plugin.extract_code_units(source, str(tmp_path / "sample.js")))

        assert ("Greeter", "class") in {(u.name, u.unit_type) for u in units}
        assert ("greet", "method") in {(u.name, u.unit_type) for u in units}
        assert ("helper", "function") in {(u.name, u.unit_type) for u in units}

        lsp_client.close()


class TestSocketLSPClient:
    def test_document_symbols_with_fake_socket_server(self, tmp_path):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            _host, port = server.getsockname()

            thread = threading.Thread(target=_run_fake_socket_lsp, args=(server,), daemon=True)
            thread.start()

            client = SocketLSPClient(f"tcp://127.0.0.1:{port}", request_timeout=2.0)

            source = """
class Greeter {
  greet(name) {
    return name;
  }
}

function helper(value) {
  return value;
}
"""
            symbols = client.document_symbols(
                source=source,
                file_path=str(tmp_path / "sample.js"),
                language_id="javascript",
            )

            assert [symbol.name for symbol in symbols] == ["Greeter", "helper"]
            assert symbols[0].children
            assert symbols[0].children[0].name == "greet"

            client.close()
            thread.join(timeout=1.0)
            assert not thread.is_alive()
