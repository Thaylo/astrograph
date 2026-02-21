#!/usr/bin/env python3
"""Analyze tripwire metrics JSONL files and print a summary.

Usage:
    python3 scripts/analyze_metrics.py /tmp/tripwire_metrics_*.jsonl
    python3 scripts/analyze_metrics.py --compare baseline.jsonl isolation.jsonl
"""

from __future__ import annotations

import argparse
import ast
import statistics
import sys
from pathlib import Path


def _parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    try:
        return ast.literal_eval(line)
    except Exception:
        return None


def load_metrics(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = _parse_line(line)
            if row:
                rows.append(row)
    return rows


def summarize(rows: list[dict], label: str) -> dict:
    if not rows:
        print(f"  {label}: no data")
        return {}

    def col(key: str) -> list[float]:
        return [r[key] for r in rows if r.get(key) is not None]

    def fmt(vals: list[float], unit: str = "") -> str:
        if not vals:
            return "n/a"
        return (
            f"min={min(vals):.1f}{unit}  "
            f"mean={statistics.mean(vals):.1f}{unit}  "
            f"max={max(vals):.1f}{unit}  "
            f"p95={sorted(vals)[int(len(vals) * 0.95)]:.1f}{unit}"
        )

    ram = col("ram_percent")
    rss = col("rss_mb")
    io = col("io_mbps")
    threads = col("threads")
    cmem = col("container_mem_mb")
    cperc = col("container_mem_percent")

    duration = rows[-1]["ts"] - rows[0]["ts"] if len(rows) > 1 else 0.0

    print(f"\n{'=' * 60}")
    print(f"  {label}  ({len(rows)} samples, {duration:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  host RAM %      {fmt(ram, '%')}")
    print(f"  process RSS     {fmt(rss, 'MB')}")
    print(f"  IO              {fmt(io, 'MB/s')}")
    print(f"  threads         {fmt(threads)}")
    if cmem:
        print(f"  container mem   {fmt(cmem, 'MB')}")
    if cperc:
        print(f"  container mem % {fmt(cperc, '%')}")

    # Slope: does RSS grow over time?
    if len(rss) >= 4:
        mid = len(rss) // 2
        early_mean = statistics.mean(rss[:mid])
        late_mean = statistics.mean(rss[mid:])
        delta = late_mean - early_mean
        trend = "GROWING" if delta > 50 else ("stable" if abs(delta) < 10 else "shrinking")
        print(
            f"  RSS trend       early={early_mean:.1f}MB → late={late_mean:.1f}MB  Δ={delta:+.1f}MB  [{trend}]"
        )

    return {
        "label": label,
        "samples": len(rows),
        "duration": duration,
        "ram_max": max(ram) if ram else None,
        "rss_max": max(rss) if rss else None,
        "rss_mean": statistics.mean(rss) if rss else None,
        "threads_max": max(threads) if threads else None,
        "container_mem_max": max(cmem) if cmem else None,
    }


def compare(summaries: list[dict]) -> None:
    if len(summaries) < 2:
        return
    print(f"\n{'=' * 60}")
    print("  COMPARISON")
    print(f"{'=' * 60}")
    baseline = summaries[0]
    for other in summaries[1:]:
        print(f"\n  {baseline['label']} vs {other['label']}:")
        for key in ("rss_mean", "rss_max", "ram_max", "threads_max", "container_mem_max"):
            a = baseline.get(key)
            b = other.get(key)
            if a is not None and b is not None:
                delta = b - a
                pct = (delta / a * 100) if a else 0
                direction = "+" if delta > 0 else ""
                print(
                    f"    {key:<22} {a:.1f} → {b:.1f}  ({direction}{delta:.1f}, {direction}{pct:.1f}%)"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", nargs="+", help="Metrics JSONL file(s)")
    parser.add_argument("--compare", action="store_true", help="Show side-by-side comparison")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = []
    for path in args.files:
        rows = load_metrics(path)
        label = Path(path).name
        s = summarize(rows, label)
        if s:
            summaries.append(s)

    if args.compare or len(summaries) > 1:
        compare(summaries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
