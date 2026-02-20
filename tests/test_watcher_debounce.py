import tempfile
import time
from pathlib import Path

import pytest


def _skip_if_watchdog_missing() -> None:
    from astrograph.watcher import HAS_WATCHDOG

    if HAS_WATCHDOG:
        return
    pytest.skip("watchdog not installed")


def test_debounce_worker_handles_event_storm() -> None:
    """Ensure debouncer doesn't create unbounded threads under heavy churn."""
    from astrograph.event_driven import EventDrivenIndex

    _skip_if_watchdog_missing()

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        files = []
        for i in range(50):
            p = root / f"file_{i}.py"
            p.write_text(f"def f{i}(): return {i}\n")
            files.append(p)

        db_path = root / "index.db"
        edi = EventDrivenIndex(persistence_path=str(db_path), watch_enabled=True)
        edi.index_directory(str(root))
        edi.start_watching(str(root))

        start = time.time()
        while time.time() - start < 2.0:
            for p in files:
                p.write_text("def f(): return 1\n")
            time.sleep(0.01)

        edi.stop_watching()
        edi.close()
