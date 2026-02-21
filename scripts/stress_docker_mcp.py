#!/usr/bin/env python3
"""Stress the MCP server inside Docker using stdio JSON-RPC."""

from __future__ import annotations

import argparse
import json
import os
import select
import subprocess
import time
from collections import deque


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="astrograph:local-arm64")
    parser.add_argument("--platform", default="linux/arm64")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pause", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--root", default=os.getcwd())
    parser.add_argument("--container-name", default="")
    parser.add_argument("--disable-event-driven", action="store_true")
    parser.add_argument("--disable-watch", action="store_true")
    parser.add_argument("--disable-persistence", action="store_true")
    parser.add_argument("--disable-auto-index", action="store_true")
    parser.add_argument("--disable-analysis-cache", action="store_true")
    return parser.parse_args()


def _env_flags(args: argparse.Namespace) -> list[str]:
    env = []
    if args.disable_event_driven:
        env += ["-e", "ASTROGRAPH_DISABLE_EVENT_DRIVEN=1"]
    if args.disable_watch:
        env += ["-e", "ASTROGRAPH_DISABLE_WATCH=1"]
    if args.disable_persistence:
        env += ["-e", "ASTROGRAPH_DISABLE_PERSISTENCE=1"]
    if args.disable_auto_index:
        env += ["-e", "ASTROGRAPH_DISABLE_AUTO_INDEX=1"]
    if args.disable_analysis_cache:
        env += ["-e", "ASTROGRAPH_DISABLE_ANALYSIS_CACHE=1"]
    return env


def _rpc(req_id: int, method: str, params: dict | None = None) -> dict:
    req = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        req["params"] = params
    return req


def main() -> int:
    args = parse_args()

    cmd = (
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "--platform",
            args.platform,
        ]
        + (["--name", args.container_name] if args.container_name else [])
        + [
            "-v",
            f"{os.path.abspath(args.root)}:/repo",
            "-w",
            "/repo",
        ]
        + _env_flags(args)
        + [args.image]
    )

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    assert proc.stdin and proc.stdout
    # Per-call timeout; set_workspace on large repos can take 20-30s inside Docker ARM64.
    call_timeout = args.timeout
    stderr_tail: deque[str] = deque(maxlen=50)

    def send(obj: dict) -> None:
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        while True:
            ready, _, _ = select.select([proc.stderr], [], [], 0)
            if not ready:
                return
            line = proc.stderr.readline()
            if not line:
                return
            stderr_tail.append(line.rstrip())

    def recv() -> dict:
        if proc.poll() is not None:
            raise RuntimeError("server exited early")
        ready, _, _ = select.select([proc.stdout], [], [], call_timeout)
        _drain_stderr()
        if not ready:
            tail = "\n".join(stderr_tail)
            raise TimeoutError(f"timeout waiting for MCP response. stderr_tail:\n{tail}")
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("server closed stdout")
        return json.loads(line)

    send(
        _rpc(
            1,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "stress", "version": "0"},
            },
        )
    )
    _ = recv()
    # MCP initialized notification (no id)
    send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})

    # Index once at start to enable file watching and establish the index.
    send(_rpc(10, "tools/call", {"name": "astrograph_set_workspace", "arguments": {"path": "."}}))
    _ = recv()

    # Then loop on cheap status calls while provoke generates file churn.
    # File churn drives watcher callbacks → background recomputes — this is
    # the primary memory-pressure path we want to observe.
    for i in range(args.iterations):
        send(_rpc(20 + i, "tools/call", {"name": "astrograph_status", "arguments": {}}))
        _ = recv()
        time.sleep(args.pause)

    # Close stdin first so the server receives EOF and shuts down gracefully.
    proc.stdin.close()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
