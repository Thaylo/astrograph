#!/usr/bin/env python3
"""Kill a child process when resource usage exceeds thresholds."""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import time

import psutil

DEFAULT_INTERVAL = 1.0
DEFAULT_BREACH_COUNT = 3
DEFAULT_MAX_RAM_PERCENT = 85.0
DEFAULT_MAX_RSS_MB = 4096.0
DEFAULT_MAX_IO_MBPS = 200.0
DEFAULT_MAX_THREADS = 800


def _read_io_bytes(proc: psutil.Process) -> int:
    try:
        io = proc.io_counters()
        return int(io.read_bytes + io.write_bytes)
    except Exception:
        return 0


def _kill_tree(proc: psutil.Process) -> None:
    try:
        for child in proc.children(recursive=True):
            with contextlib.suppress(Exception):
                child.terminate()
        with contextlib.suppress(Exception):
            proc.terminate()
    except Exception:
        pass


def _kill_tree_hard(proc: psutil.Process) -> None:
    try:
        for child in proc.children(recursive=True):
            with contextlib.suppress(Exception):
                child.kill()
        with contextlib.suppress(Exception):
            proc.kill()
    except Exception:
        pass


def _try_gpu_util() -> float | None:
    # Optional: if `nvidia-smi` is available, use it.
    try:
        import shutil

        if shutil.which("nvidia-smi") is None:
            return None
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
        )
        vals = [float(x.strip()) for x in out.strip().splitlines() if x.strip()]
        if not vals:
            return None
        return max(vals)
    except Exception:
        return None


def _count_threads(proc: psutil.Process) -> int:
    count = 0
    with contextlib.suppress(Exception):
        count += proc.num_threads()
    with contextlib.suppress(Exception):
        for child in proc.children(recursive=True):
            with contextlib.suppress(Exception):
                count += child.num_threads()
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--breach-count", type=int, default=DEFAULT_BREACH_COUNT)
    parser.add_argument("--max-ram-percent", type=float, default=DEFAULT_MAX_RAM_PERCENT)
    parser.add_argument("--max-rss-mb", type=float, default=DEFAULT_MAX_RSS_MB)
    parser.add_argument("--max-io-mbps", type=float, default=DEFAULT_MAX_IO_MBPS)
    parser.add_argument("--max-threads", type=int, default=DEFAULT_MAX_THREADS)
    parser.add_argument("--max-gpu-util", type=float, default=95.0)
    parser.add_argument("--grace-seconds", type=float, default=2.0)
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run after --")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.cmd or args.cmd[0] != "--":
        print("tripwire: command required (use -- before the command)")
        return 2
    cmd = args.cmd[1:]
    if not cmd:
        print("tripwire: command required (use -- before the command)")
        return 2

    proc = subprocess.Popen(cmd)
    ps = psutil.Process(proc.pid)

    last_io = _read_io_bytes(ps)
    last_ts = time.time()
    breaches = 0
    start = time.time()

    try:
        while True:
            if proc.poll() is not None:
                return proc.returncode or 0

            now = time.time()
            if now - start < args.grace_seconds:
                time.sleep(args.interval)
                continue

            ram_percent = psutil.virtual_memory().percent
            rss_mb = ps.memory_info().rss / (1024 * 1024)

            io_bytes = _read_io_bytes(ps)
            dt = max(now - last_ts, 0.001)
            io_mbps = (io_bytes - last_io) / (1024 * 1024) / dt
            last_io = io_bytes
            last_ts = now
            thread_count = _count_threads(ps)

            gpu_util = _try_gpu_util()

            breach = False
            reasons = []
            if ram_percent > args.max_ram_percent:
                breach = True
                reasons.append(f"ram={ram_percent:.1f}%")
            if rss_mb > args.max_rss_mb:
                breach = True
                reasons.append(f"rss={rss_mb:.1f}MB")
            if io_mbps > args.max_io_mbps:
                breach = True
                reasons.append(f"io={io_mbps:.1f}MB/s")
            if thread_count > args.max_threads:
                breach = True
                reasons.append(f"threads={thread_count}")
            if gpu_util is not None and gpu_util > args.max_gpu_util:
                breach = True
                reasons.append(f"gpu={gpu_util:.1f}%")

            if breach:
                breaches += 1
                print(f"tripwire breach {breaches}/{args.breach_count}: {', '.join(reasons)}")
            else:
                breaches = 0

            if breaches >= args.breach_count:
                print("tripwire: killing process tree")
                _kill_tree(ps)
                time.sleep(1)
                if proc.poll() is None:
                    _kill_tree_hard(ps)
                return 3

            time.sleep(args.interval)
    except KeyboardInterrupt:
        _kill_tree(ps)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
