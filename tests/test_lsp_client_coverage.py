"""Tests for lsp_client module coverage gaps.

Targets the pure/edge-case code paths that don't require a running LSP server.
"""

from __future__ import annotations

import socket
import threading
from unittest.mock import MagicMock, patch

import pytest

from astrograph.languages.lsp_client import (
    SocketLSPClient,
    SubprocessLSPClient,
    _decode_lsp_payload,
    _parse_content_length,
)


class TestParseContentLength:
    def test_valid(self):
        assert _parse_content_length({"content-length": "42"}) == 42

    def test_missing(self):
        assert _parse_content_length({}) == 0

    def test_invalid(self):
        assert _parse_content_length({"content-length": "abc"}) == 0

    def test_negative(self):
        assert _parse_content_length({"content-length": "-5"}) == 0


class TestDecodeLspPayload:
    def test_valid(self):
        body = b'{"jsonrpc":"2.0","id":1}'
        result = _decode_lsp_payload(body, len(body))
        assert result is not None
        assert result["id"] == 1

    def test_length_mismatch(self):
        assert _decode_lsp_payload(b'{"id":1}', 999) is None

    def test_invalid_json(self):
        body = b"not json"
        assert _decode_lsp_payload(body, len(body)) is None

    def test_non_dict(self):
        body = b"[1, 2, 3]"
        assert _decode_lsp_payload(body, len(body)) is None


class TestSubprocessLSPClientEdgeCases:
    def test_from_command_string(self):
        client = SubprocessLSPClient.from_command_string("echo hello world")
        assert client._command == ("echo", "hello", "world")

    def test_path_to_uri_untitled(self):
        client = SubprocessLSPClient(["echo"])
        assert client._path_to_uri("<inline>") == "untitled:astrograph"

    def test_path_to_uri_relative(self):
        client = SubprocessLSPClient(["echo"])
        uri = client._path_to_uri("relative/path.py")
        assert uri.startswith("file://")

    def test_path_to_uri_absolute(self, tmp_path):
        client = SubprocessLSPClient(["echo"])
        p = tmp_path / "test.py"
        p.touch()
        uri = client._path_to_uri(str(p))
        assert uri.startswith("file://")

    def test_start_process_empty_command(self):
        client = SubprocessLSPClient([])
        assert client._start_process() is False
        assert client._disabled is True

    def test_start_process_nonexistent_command(self):
        client = SubprocessLSPClient(["nonexistent_command_xyz_abc"])
        assert client._start_process() is False
        assert client._disabled is True

    def test_start_process_disabled(self):
        client = SubprocessLSPClient(["echo"])
        client._disabled = True
        assert client._start_process() is False

    def test_close_no_process(self):
        client = SubprocessLSPClient(["echo"])
        client.close()  # should not raise

    def test_context_manager(self):
        with SubprocessLSPClient(["echo"]) as client:
            assert isinstance(client, SubprocessLSPClient)

    def test_request_uninitialized(self):
        client = SubprocessLSPClient(["echo"])
        with pytest.raises(RuntimeError, match="not initialized"):
            client._request("test/method", None)

    def test_send_message_no_proc(self):
        client = SubprocessLSPClient(["echo"])
        with pytest.raises(RuntimeError, match="stdin"):
            client._send_message({"method": "test"})

    def test_read_message_no_proc(self):
        client = SubprocessLSPClient(["echo"])
        assert client._read_message(0.1) is None

    def test_document_symbols_disabled(self):
        client = SubprocessLSPClient(["echo"])
        client._disabled = True
        result = client.document_symbols(
            source="def f(): pass", file_path="test.py", language_id="python"
        )
        assert result == []

    def test_semantic_tokens_disabled(self):
        client = SubprocessLSPClient(["echo"])
        client._disabled = True
        result = client.semantic_tokens(
            source="def f(): pass", file_path="test.py", language_id="python"
        )
        assert result is None

    def test_semantic_tokens_no_legend(self):
        client = SubprocessLSPClient(["echo"])
        client._semantic_token_legend = None
        result = client.semantic_tokens(
            source="def f(): pass", file_path="test.py", language_id="python"
        )
        assert result is None

    def test_parse_position_non_dict(self):
        client = SubprocessLSPClient(["echo"])
        pos = client._parse_position("not a dict")
        assert pos.line == 0 and pos.character == 0

    def test_parse_range_non_dict(self):
        client = SubprocessLSPClient(["echo"])
        r = client._parse_range("not a dict")
        assert r.start.line == 0

    def test_parse_symbol_list_non_list(self):
        client = SubprocessLSPClient(["echo"])
        assert client._parse_symbol_list("not a list") == []

    def test_parse_symbol_non_dict(self):
        client = SubprocessLSPClient(["echo"])
        assert client._parse_symbol(42) is None

    def test_close_quietly_none(self):
        client = SubprocessLSPClient(["echo"])
        client._close_quietly(None)  # should not raise

    def test_close_quietly_raises(self):
        client = SubprocessLSPClient(["echo"])
        mock = MagicMock()
        mock.close.side_effect = OSError("boom")
        client._close_quietly(mock)  # should not raise

    def test_server_name_property(self):
        client = SubprocessLSPClient(["echo"])
        assert client.server_name is None

    def test_stale_process_cleanup(self):
        client = SubprocessLSPClient(["echo"])
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already exited
        mock_proc.stdin = MagicMock()
        mock_proc.stdout = MagicMock()
        client._proc = mock_proc
        # When a stale process exists, _start_process should clean it up
        # The OSError from starting "echo" with Popen is fine
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.side_effect = OSError("test")
            result = client._start_process()
        assert result is False


class TestSocketLSPClient:
    def test_invalid_endpoint(self):
        with pytest.raises(ValueError, match="Invalid"):
            SocketLSPClient("not-a-valid-endpoint")

    def test_path_to_uri_with_prefix_map(self):
        client = SocketLSPClient(
            "tcp://127.0.0.1:9999",
            path_prefix_map=("/workspace", "/home/user/project"),
        )
        uri = client._path_to_uri("/workspace/src/main.py")
        assert "/home/user/project/src/main.py" in uri
        client.close()

    def test_path_to_uri_no_prefix_match(self):
        client = SocketLSPClient(
            "tcp://127.0.0.1:9999",
            path_prefix_map=("/workspace", "/home/user/project"),
        )
        uri = client._path_to_uri("/other/path/main.py")
        assert "/other/path/main.py" in uri
        client.close()

    def test_start_process_connection_refused(self):
        client = SocketLSPClient(
            "tcp://127.0.0.1:1",
            request_timeout=0.1,
        )
        assert client._start_process() is False
        assert client._disabled is True

    def test_start_process_disabled(self):
        client = SocketLSPClient(
            "tcp://127.0.0.1:9999",
            request_timeout=0.1,
        )
        client._disabled = True
        assert client._start_process() is False

    def test_start_process_already_connected(self):
        client = SocketLSPClient(
            "tcp://127.0.0.1:9999",
            request_timeout=0.1,
        )
        client._sock = MagicMock()
        assert client._start_process() is True
        client._sock = None

    def test_send_message_no_stdin(self):
        client = SocketLSPClient(
            "tcp://127.0.0.1:9999",
            request_timeout=0.1,
        )
        with pytest.raises(RuntimeError, match="writer"):
            client._send_message({"method": "test"})

    def test_read_message_no_stdout(self):
        client = SocketLSPClient(
            "tcp://127.0.0.1:9999",
            request_timeout=0.1,
        )
        assert client._read_message(0.1) is None

    def test_close_no_socket(self):
        client = SocketLSPClient(
            "tcp://127.0.0.1:9999",
            request_timeout=0.1,
        )
        client.close()  # should not raise

    def test_connect_to_ephemeral_server(self):
        """Connect to a real ephemeral server to cover the happy path."""
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        server_sock.listen(1)
        port = server_sock.getsockname()[1]

        def _accept_and_close():
            try:
                conn, _ = server_sock.accept()
                conn.close()
            except OSError:
                pass
            server_sock.close()

        t = threading.Thread(target=_accept_and_close, daemon=True)
        t.start()

        client = SocketLSPClient(
            f"tcp://127.0.0.1:{port}",
            request_timeout=1.0,
        )
        assert client._start_process() is True
        client.close()
        t.join(timeout=2.0)
