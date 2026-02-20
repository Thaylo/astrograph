#!/usr/bin/env python3
"""Generate filesystem churn to stress event-driven indexing."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--files", type=int, default=500)
    parser.add_argument("--rate", type=float, default=40.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--aggressive", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    files = []
    for i in range(args.files):
        p = root / f"file_{i}.py"
        p.write_text(f"def f{i}(): return {i}\n")
        files.append(p)

    start = time.time()
    while time.time() - start < args.duration:
        for _ in range(int(args.rate)):
            p = random.choice(files)
            p.write_text(f"def f{random.randint(0, 99999)}(): return {random.random()}\n")
        if args.aggressive:
            for _ in range(int(args.rate / 4)):
                p = random.choice(files)
                p.write_text(f"def f{random.randint(0, 99999)}(): return {random.random()}\n")
        time.sleep(0.02)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
