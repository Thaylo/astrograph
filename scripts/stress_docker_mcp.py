#!/usr/bin/env python3
"""Stress the MCP server inside Docker using stdio JSON-RPC."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="astrograph:local-arm64")
    parser.add_argument("--platform", default="linux/arm64")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pause", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--root", default=os.getcwd())
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
            "-v",
            f"{args.root}:/repo",
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

    def send(obj: dict) -> None:
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    def recv() -> dict:
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
    send(_rpc(2, "initialized"))

    for i in range(args.iterations):
        send(_rpc(10 + i * 3, "tools/call", {"name": "status", "arguments": {}}))
        _ = recv()
        send(_rpc(11 + i * 3, "tools/call", {"name": "index_codebase", "arguments": {"path": "."}}))
        _ = recv()
        send(_rpc(12 + i * 3, "tools/call", {"name": "analyze", "arguments": {}}))
        _ = recv()
        time.sleep(args.pause)

    proc.terminate()
    try:
        proc.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
