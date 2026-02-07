"""Tests for the dual-mode stdio transport."""

from __future__ import annotations

import io
import json
import sys
from unittest.mock import patch

import anyio
import pytest
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage

from astrograph.stdio_transport import _StdioReader, dual_stdio_server


def _make_initialize_request(id: int = 1) -> dict:
    """Build a minimal JSON-RPC initialize request."""
    return {
        "jsonrpc": "2.0",
        "id": id,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0.1.0"},
        },
    }


class _FakeStream:
    """Fake async file-like stream for testing."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0

    async def read(self, max_bytes: int = 65536) -> bytes:
        if self._pos >= len(self._data):
            return b""
        chunk = self._data[self._pos : self._pos + max_bytes]
        self._pos += len(chunk)
        return chunk


class TestStdioReader:
    """Tests for the _StdioReader class."""

    @pytest.mark.asyncio
    async def test_newline_mode_detection(self):
        """First byte '{' should trigger newline mode."""
        msg = json.dumps(_make_initialize_request()).encode("utf-8") + b"\n"
        reader = _StdioReader(_FakeStream(msg))
        data = await reader.read_message()
        assert reader.mode == "newline"
        parsed = json.loads(data)
        assert parsed["method"] == "initialize"

    @pytest.mark.asyncio
    async def test_framed_mode_detection(self):
        """First byte 'C' (Content-Length) should trigger framed mode."""
        body = json.dumps(_make_initialize_request()).encode("utf-8")
        msg = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        reader = _StdioReader(_FakeStream(msg))
        data = await reader.read_message()
        assert reader.mode == "framed"
        parsed = json.loads(data)
        assert parsed["method"] == "initialize"

    @pytest.mark.asyncio
    async def test_framed_mode_detection_lowercase_header(self):
        """Lowercase content-length header should still trigger framed mode."""
        body = json.dumps(_make_initialize_request()).encode("utf-8")
        msg = f"content-length: {len(body)}\r\n\r\n".encode("ascii") + body
        reader = _StdioReader(_FakeStream(msg))
        data = await reader.read_message()
        assert reader.mode == "framed"
        parsed = json.loads(data)
        assert parsed["method"] == "initialize"

    @pytest.mark.asyncio
    async def test_auto_detect_with_leading_whitespace(self):
        """Whitespace before '{' should still detect newline mode."""
        msg = b"  \n  " + json.dumps(_make_initialize_request()).encode("utf-8") + b"\n"
        reader = _StdioReader(_FakeStream(msg))
        data = await reader.read_message()
        assert reader.mode == "newline"
        parsed = json.loads(data)
        assert parsed["method"] == "initialize"

    @pytest.mark.asyncio
    async def test_framed_mode_multiple_messages(self):
        """Multiple framed messages should be read correctly."""
        msg1 = json.dumps(_make_initialize_request(1)).encode("utf-8")
        msg2 = json.dumps(_make_initialize_request(2)).encode("utf-8")
        data = (
            f"Content-Length: {len(msg1)}\r\n\r\n".encode("ascii")
            + msg1
            + f"Content-Length: {len(msg2)}\r\n\r\n".encode("ascii")
            + msg2
        )
        reader = _StdioReader(_FakeStream(data))
        d1 = await reader.read_message()
        d2 = await reader.read_message()
        assert json.loads(d1)["id"] == 1
        assert json.loads(d2)["id"] == 2

    @pytest.mark.asyncio
    async def test_framed_mode_with_lf_header_delimiter(self):
        """Framed mode should also accept LF-only header delimiter."""
        body = json.dumps(_make_initialize_request()).encode("utf-8")
        msg = f"Content-Length: {len(body)}\n\n".encode("ascii") + body
        reader = _StdioReader(_FakeStream(msg))
        data = await reader.read_message()
        assert reader.mode == "framed"
        parsed = json.loads(data)
        assert parsed["method"] == "initialize"

    @pytest.mark.asyncio
    async def test_newline_mode_multiple_messages(self):
        """Multiple newline messages should be read correctly."""
        msg1 = json.dumps(_make_initialize_request(1)).encode("utf-8")
        msg2 = json.dumps(_make_initialize_request(2)).encode("utf-8")
        data = msg1 + b"\n" + msg2 + b"\n"
        reader = _StdioReader(_FakeStream(data))
        d1 = await reader.read_message()
        d2 = await reader.read_message()
        assert json.loads(d1)["id"] == 1
        assert json.loads(d2)["id"] == 2

    @pytest.mark.asyncio
    async def test_mixed_mode_newline_then_framed(self):
        """Reader should handle mode switching between consecutive messages."""
        msg1 = json.dumps(_make_initialize_request(1)).encode("utf-8")
        msg2 = json.dumps(_make_initialize_request(2)).encode("utf-8")
        data = msg1 + b"\n" + f"Content-Length: {len(msg2)}\r\n\r\n".encode("ascii") + msg2
        reader = _StdioReader(_FakeStream(data))
        d1 = await reader.read_message()
        assert reader.mode == "newline"
        d2 = await reader.read_message()
        assert reader.mode == "framed"
        assert json.loads(d1)["id"] == 1
        assert json.loads(d2)["id"] == 2

    @pytest.mark.asyncio
    async def test_mixed_mode_framed_then_newline(self):
        """Reader should handle framed request followed by newline request."""
        msg1 = json.dumps(_make_initialize_request(1)).encode("utf-8")
        msg2 = json.dumps(_make_initialize_request(2)).encode("utf-8")
        data = f"Content-Length: {len(msg1)}\r\n\r\n".encode("ascii") + msg1 + b"\n" + msg2 + b"\n"
        reader = _StdioReader(_FakeStream(data))
        d1 = await reader.read_message()
        assert reader.mode == "framed"
        d2 = await reader.read_message()
        assert reader.mode == "newline"
        assert json.loads(d1)["id"] == 1
        assert json.loads(d2)["id"] == 2

    @pytest.mark.asyncio
    async def test_framed_missing_content_length(self):
        """Framed message without Content-Length should raise ValueError."""
        # Force framed mode and provide headers without Content-Length.
        data = b"Custom-Header: foo\r\n\r\n{}"
        reader = _StdioReader(_FakeStream(data))
        with pytest.raises(ValueError, match="Missing Content-Length"):
            await reader._read_framed()

    @pytest.mark.asyncio
    async def test_newline_mode_no_trailing_newline(self):
        """Message without trailing newline should still be returned on EOF."""
        msg = json.dumps(_make_initialize_request()).encode("utf-8")  # no \n
        reader = _StdioReader(_FakeStream(msg))
        data = await reader.read_message()
        assert reader.mode == "newline"
        parsed = json.loads(data)
        assert parsed["method"] == "initialize"


class TestDualStdioServer:
    """Integration tests for dual_stdio_server context manager."""

    @pytest.mark.asyncio
    async def test_newline_mode_roundtrip(self):
        """Pipe newline-delimited JSON-RPC → verify response is newline-delimited."""
        request = json.dumps(_make_initialize_request()).encode("utf-8") + b"\n"
        fake_stdin = io.BytesIO(request)
        fake_stdout = io.BytesIO()

        with patch.object(sys, "stdin", type("", (), {"buffer": fake_stdin})()), patch.object(
            sys, "stdout", type("", (), {"buffer": fake_stdout})()
        ):
            async with dual_stdio_server() as (read_stream, write_stream):
                # Read the incoming message
                msg = await read_stream.receive()
                assert isinstance(msg, SessionMessage)

                # Send a response back
                response = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"protocolVersion": "2024-11-05", "capabilities": {}},
                }
                response_msg = SessionMessage(message=JSONRPCMessage.model_validate(response))
                await write_stream.send(response_msg)

                # Give writer time to flush
                await anyio.sleep(0.05)

        # Verify output is newline-delimited (no Content-Length header)
        output = fake_stdout.getvalue()
        assert output.endswith(b"\n")
        assert b"Content-Length" not in output
        parsed = json.loads(output.strip())
        assert parsed["id"] == 1

    @pytest.mark.asyncio
    async def test_framed_mode_roundtrip(self):
        """Pipe Content-Length framed JSON-RPC → verify response has Content-Length header."""
        body = json.dumps(_make_initialize_request()).encode("utf-8")
        request = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body
        fake_stdin = io.BytesIO(request)
        fake_stdout = io.BytesIO()

        with patch.object(sys, "stdin", type("", (), {"buffer": fake_stdin})()), patch.object(
            sys, "stdout", type("", (), {"buffer": fake_stdout})()
        ):
            async with dual_stdio_server() as (read_stream, write_stream):
                msg = await read_stream.receive()
                assert isinstance(msg, SessionMessage)

                response = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"protocolVersion": "2024-11-05", "capabilities": {}},
                }
                response_msg = SessionMessage(message=JSONRPCMessage.model_validate(response))
                await write_stream.send(response_msg)

                await anyio.sleep(0.05)

        # Verify output uses Content-Length framing
        output = fake_stdout.getvalue()
        assert b"Content-Length:" in output
        # Parse: Content-Length: N\r\n\r\n<body>
        header_end = output.index(b"\r\n\r\n")
        header = output[:header_end].decode("ascii")
        body_out = output[header_end + 4 :]
        content_length = int(header.split(":")[1].strip())
        assert len(body_out) == content_length
        parsed = json.loads(body_out)
        assert parsed["id"] == 1

    @pytest.mark.asyncio
    async def test_eof_closes_read_stream(self):
        """When stdin is empty, read_stream should close after EOFError."""
        fake_stdin = io.BytesIO(b"")
        fake_stdout = io.BytesIO()

        with patch.object(sys, "stdin", type("", (), {"buffer": fake_stdin})()), patch.object(
            sys, "stdout", type("", (), {"buffer": fake_stdout})()
        ):
            async with dual_stdio_server() as (read_stream, write_stream):
                # stdin is empty, so reader hits EOF immediately.
                # The read_send channel closes, which means receive() raises EndOfStream.
                from anyio import EndOfStream

                with pytest.raises(EndOfStream):
                    await read_stream.receive()
