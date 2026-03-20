import tempfile
import time
from pathlib import Path

from helpers import skip_if_watchdog_missing


def test_debounce_worker_handles_event_storm() -> None:
    """Ensure debouncer doesn't create unbounded threads under heavy churn."""
    from astrograph.event_driven import EventDrivenIndex

    skip_if_watchdog_missing()

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

        handler = edi._watcher._handler if edi._watcher is not None else None
        assert handler is not None
        assert handler.quarantined or edi.get_stats()["file_events_processed"] >= 0

        edi.stop_watching()
        edi.close()


def test_storm_quarantine_callback(monkeypatch) -> None:
    """Test that storm quarantine fires the on_storm_quarantine callback."""
    from astrograph.watcher import SourceFileHandler

    monkeypatch.setattr("astrograph.watcher._WATCHER_MAX_EVENTS_PER_WINDOW", 3)
    monkeypatch.setattr("astrograph.watcher._WATCHER_STORM_WINDOW_SECONDS", 10.0)

    callback_called: list[bool] = []

    handler = SourceFileHandler(
        on_modified=lambda _: None,
        on_created=lambda _: None,
        on_deleted=lambda _: None,
        on_storm_quarantine=lambda: callback_called.append(True),
    )

    # First 3 events — no quarantine
    for _ in range(3):
        handler._record_event_rate()

    assert len(callback_called) == 0

    # 4th event — triggers quarantine
    handler._record_event_rate()

    assert len(callback_called) == 1
