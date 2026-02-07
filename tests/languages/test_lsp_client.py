"""Tests for the subprocess LSP client implementation."""

from __future__ import annotations

import sys
from pathlib import Path

from astrograph.languages.javascript_lsp_plugin import JavaScriptLSPPlugin
from astrograph.languages.lsp_client import SubprocessLSPClient

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
