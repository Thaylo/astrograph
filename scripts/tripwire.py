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
DEFAULT_MAX_CONTAINER_MEM_MB = 0.0
DEFAULT_MAX_CONTAINER_MEM_PERCENT = 0.0


def _read_io_bytes(proc: psutil.Process) -> int:
    try:
        io = proc.io_counters()
        return int(io.read_bytes + io.write_bytes)
    except Exception:
        return 0


def _kill_tree(proc: psutil.Process, *, hard: bool = False) -> None:
    signal = "kill" if hard else "terminate"
    try:
        for child in proc.children(recursive=True):
            with contextlib.suppress(Exception):
                getattr(child, signal)()
        with contextlib.suppress(Exception):
            getattr(proc, signal)()
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
    parser.add_argument("--docker-name", type=str, default="")
    parser.add_argument("--max-container-mem-mb", type=float, default=DEFAULT_MAX_CONTAINER_MEM_MB)
    parser.add_argument(
        "--max-container-mem-percent", type=float, default=DEFAULT_MAX_CONTAINER_MEM_PERCENT
    )
    parser.add_argument("--max-gpu-util", type=float, default=95.0)
    parser.add_argument("--grace-seconds", type=float, default=2.0)
    parser.add_argument("--log-path", type=str, default="")
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
            container_mem_mb = None
            container_mem_percent = None
            if args.docker_name:
                try:
                    out = subprocess.check_output(
                        [
                            "docker",
                            "stats",
                            "--no-stream",
                            "--format",
                            "{{.MemUsage}} {{.MemPerc}}",
                            args.docker_name,
                        ],
                        text=True,
                    ).strip()
                    if out:
                        mem_usage, mem_perc = out.split(" ", 1)
                        mem_val, mem_unit = mem_usage.split("/")
                        mem_val = mem_val.strip()
                        mem_unit = mem_unit.strip()
                        container_mem_percent = float(mem_perc.strip().strip("%"))
                        if mem_val.endswith("MiB"):
                            container_mem_mb = float(mem_val[:-3])
                        elif mem_val.endswith("GiB"):
                            container_mem_mb = float(mem_val[:-3]) * 1024.0
                        elif mem_val.endswith("KiB"):
                            container_mem_mb = float(mem_val[:-3]) / 1024.0
                except Exception:
                    pass

            if args.log_path:
                payload = {
                    "ts": now,
                    "ram_percent": ram_percent,
                    "rss_mb": rss_mb,
                    "io_mbps": io_mbps,
                    "threads": thread_count,
                    "gpu_util": gpu_util,
                    "container_mem_mb": container_mem_mb,
                    "container_mem_percent": container_mem_percent,
                }
                with (
                    contextlib.suppress(Exception),
                    open(args.log_path, "a", encoding="utf-8") as handle,
                ):
                    handle.write(repr(payload) + "\n")

            breach = False
            reasons = []
            for value, threshold, label in (
                (ram_percent, args.max_ram_percent, f"ram={ram_percent:.1f}%"),
                (rss_mb, args.max_rss_mb, f"rss={rss_mb:.1f}MB"),
                (io_mbps, args.max_io_mbps, f"io={io_mbps:.1f}MB/s"),
                (thread_count, args.max_threads, f"threads={thread_count}"),
            ):
                if value > threshold:
                    breach = True
                    reasons.append(label)
            for limit, value, unit in (
                (args.max_container_mem_mb, container_mem_mb, "MB"),
                (args.max_container_mem_percent, container_mem_percent, "%"),
            ):
                if limit > 0.0 and value is not None and value > limit:
                    breach = True
                    reasons.append(f"container_mem={value:.1f}{unit}")
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
                    _kill_tree(ps, hard=True)
                return 3

            time.sleep(args.interval)
    except KeyboardInterrupt:
        _kill_tree(ps)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
