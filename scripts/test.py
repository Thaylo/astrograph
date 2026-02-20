#!/usr/bin/env python3
"""AI-friendly test runner - minimal output, maximum signal."""

import re
import subprocess
import sys


def _extract_int(pattern: str, text: str, default: int = 0) -> int:
    """Extract an integer from text using a regex pattern."""
    if match := re.search(pattern, text):
        return int(match.group(1))
    return default


def main() -> int:
    """Run tests and output only actionable information."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-v",  # Verbose to get "X passed" summary
            "--tb=line",  # One-line tracebacks for failures
            "--no-header",  # Skip pytest version
            "--cov=src/code_structure_mcp",
            "--cov-report=",  # No coverage table
            "--cov-fail-under=92",
        ],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr

    # Extract metrics
    passed = _extract_int(r"(\d+) passed", output)
    failed = _extract_int(r"(\d+) failed", output)
    errors = _extract_int(r"(\d+) error", output)

    coverage = match.group(1) if (match := re.search(r"Total coverage: ([\d.]+%)", output)) else "?"

    # Single-line output
    status = "PASS" if result.returncode == 0 else "FAIL"
    print(f"tests:{status} passed:{passed} failed:{failed} errors:{errors} coverage:{coverage}")

    # Show failures only (one per line, must contain :: for test location)
    if result.returncode != 0:
        for line in output.split("\n"):
            line = line.strip()
            is_failure = line.startswith(("FAILED", "ERROR")) or "::test_" in line
            if is_failure and "::" in line:
                print(line)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
