"""
Dual-mode stdio transport for MCP.

Auto-detects client framing from the first non-whitespace byte on stdin:
  '{' → newline mode  (read line-by-line, write <json>\n)
  'C' → framed mode   (Content-Length: N\r\n\r\n<N bytes>)

Drop-in replacement for ``mcp.server.stdio.stdio_server``.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast

import anyio
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from mcp.shared.message import SessionMessage
from mcp.types import JSONRPCMessage


class _UnclosableSendStream:
    """Write-stream proxy that ignores close.

    The MCP session's ``_receive_loop`` closes ``_write_stream`` when
    the read stream ends (EOF).  This proxy prevents that teardown so
    in-flight request handlers can still send their responses.  The
    real stream is closed by the outer transport scope after
    ``server.run()`` returns.
    """

    def __init__(self, inner: MemoryObjectSendStream[SessionMessage]) -> None:
        self._inner = inner

    async def send(self, item: SessionMessage) -> None:
        await self._inner.send(item)

    async def aclose(self) -> None:
        pass  # Intentional no-op — outer scope manages lifetime.

    async def __aenter__(self) -> _UnclosableSendStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass  # Intentional no-op.


class _StdioReader:
    """Buffered binary reader that auto-detects newline vs Content-Length framing."""

    def __init__(self, stream: anyio.AsyncFile[bytes]) -> None:
        self._stream = stream
        self._buf = b""
        self.mode: str | None = None  # "newline" or "framed"

    async def _fill(self, min_bytes: int = 1) -> None:
        """Read more data from the stream into the buffer."""
        while len(self._buf) < min_bytes:
            need = max(1, min_bytes - len(self._buf))
            chunk = await self._stream.read(need)
            if chunk:
                self._buf += chunk
                continue
            raise EOFError("stdin closed")

    async def _detect_mode(self) -> str:
        """Detect framing mode for the next message in the buffer."""
        while True:
            await self._fill(1)
            # Skip leading whitespace
            stripped = self._buf.lstrip()
            if not stripped:
                self._buf = b""
                continue
            # Update buffer to stripped version
            self._buf = stripped
            first = stripped[:1]
            # JSON-RPC newline mode always starts with a JSON value (object/array).
            mode = {
                b"{": "newline",
                b"[": "newline",
                b"C": "framed",
                b"c": "framed",
            }.get(first)
            if mode is not None:
                return mode

            # Fallback: if it looks like an HTTP-style header line, treat as framed.
            first_line = stripped.split(b"\n", 1)[0]
            if b":" in first_line and first.isalpha():
                return "framed"

            return "newline"

    async def read_message(self) -> bytes:
        """Read and return the next complete JSON-RPC message as bytes."""
        self.mode = await self._detect_mode()
        mode = self.mode

        if mode == "newline":
            return await self._read_newline()
        else:
            return await self._read_framed()

    async def _read_newline(self) -> bytes:
        """Read a newline-delimited message."""
        while b"\n" not in self._buf:
            try:
                # Force reading more data even if buffer is non-empty
                await self._fill(len(self._buf) + 1)
            except EOFError:
                # Return remaining buffer content on EOF (last message
                # may lack a trailing newline, e.g. subprocess.run input).
                remaining = self._buf.strip()
                self._buf = b""
                if remaining:
                    return remaining
                raise
        line, self._buf = self._buf.split(b"\n", 1)
        return line.strip()

    async def _read_framed(self) -> bytes:
        """Read a Content-Length framed message."""
        # Read headers until blank line (CRLF or LF style).
        header_end = -1
        delim_len = 0
        while True:
            for delimiter, delimiter_len in ((b"\r\n\r\n", 4), (b"\n\n", 2)):
                if delimiter in self._buf:
                    header_end = self._buf.index(delimiter)
                    delim_len = delimiter_len
                    break
            if header_end >= 0:
                break
            # Force progress even when buffer already has partial header bytes.
            await self._fill(len(self._buf) + 1)

        headers = self._buf[:header_end].decode("ascii")
        self._buf = self._buf[header_end + delim_len :]

        # Parse Content-Length
        content_length = None
        for header_line in headers.splitlines():
            if ":" not in header_line:
                continue
            name, value = header_line.split(":", 1)
            if name.strip().lower() == "content-length":
                content_length = int(value.strip())
                break

        if content_length is None:
            raise ValueError("Missing Content-Length header in framed message")

        # Guard against oversized Content-Length (max 256 MB)
        _MAX_CONTENT_LENGTH = 256 * 1024 * 1024
        if content_length > _MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Content-Length {content_length} exceeds maximum {_MAX_CONTENT_LENGTH}"
            )

        # Read body
        await self._fill(content_length)
        body = self._buf[:content_length]
        self._buf = self._buf[content_length:]
        return body


@asynccontextmanager
async def dual_stdio_server() -> (
    AsyncIterator[
        tuple[
            MemoryObjectReceiveStream[SessionMessage | Exception],
            MemoryObjectSendStream[SessionMessage],
        ]
    ]
):
    """
    Async context manager matching the interface of ``mcp.server.stdio.stdio_server``.

    Yields ``(read_stream, write_stream)`` where messages are automatically
    framed in whichever mode the client uses.

    The write stream is wrapped in ``_UnclosableSendStream`` so that the
    MCP session's ``_receive_loop`` cannot tear it down when stdin reaches
    EOF.  This prevents ``ClosedResourceError`` for handlers that are still
    writing their responses.
    """
    read_send, read_recv = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    write_send, write_recv = anyio.create_memory_object_stream[SessionMessage](0)

    stdin = anyio.wrap_file(sys.stdin.buffer)
    stdout = anyio.wrap_file(sys.stdout.buffer)

    reader = _StdioReader(stdin)
    shutdown_event = anyio.Event()

    # Wrap write_send so the MCP session's _receive_loop can't close it
    # when the read stream ends (EOF).
    write_proxy = _UnclosableSendStream(write_send)

    async def stdin_task() -> None:
        async with read_send:
            while True:
                try:
                    data = await reader.read_message()
                    if not data:
                        continue
                    msg = JSONRPCMessage.model_validate_json(data)
                    await read_send.send(SessionMessage(message=msg))
                except EOFError:
                    break
                except Exception as exc:
                    await read_send.send(exc)
        # Block until the server has finished processing all requests
        # and the outer scope signals shutdown.
        await shutdown_event.wait()

    async def stdout_task() -> None:
        async with write_recv:
            async for session_message in write_recv:
                json_bytes = session_message.message.model_dump_json(
                    by_alias=True, exclude_none=True
                ).encode("utf-8")

                if reader.mode == "framed":
                    header = f"Content-Length: {len(json_bytes)}\r\n\r\n".encode("ascii")
                    await stdout.write(header + json_bytes)
                else:
                    await stdout.write(json_bytes + b"\n")
                await stdout.flush()

    async with anyio.create_task_group() as tg:
        tg.start_soon(stdin_task)
        tg.start_soon(stdout_task)
        yield read_recv, cast(MemoryObjectSendStream[SessionMessage], write_proxy)
        # server.run() returned — all request handlers have completed.
        # Close the real write_send so stdout_task drains and exits.
        await write_send.aclose()
        # Unblock stdin_task so it can exit too.
        shutdown_event.set()
