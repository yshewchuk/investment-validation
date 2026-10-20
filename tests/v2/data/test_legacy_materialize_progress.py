"""Bounded stderr progress for legacy materialization (#1018): never stdout, never per item."""
from __future__ import annotations

from engine.v2.data import legacy_adapter, materialize_progress
from engine.v2.data.reference_inputs import LEGACY_SNAPSHOT_PATH, REGISTRY_PATH
from engine.v2.data.repository import Repository
from tests.v2.data.test_legacy_materialization import (
    CALENDAR_PATH,
    _build_request,
    _build_snapshot,
    _snapshot_object_ref,
)

#: Exactly the fixture's materialized layout: four curated tables over
#: years 2020/2021 (daily_market's 2020 partition is two fragments, so a
#: whole-table copy writes two parts), the two single-file tables, and the
#: three pinned refs this fixture's ``_pinned_refs`` carries (no
#: ``extra_registry_refs`` and no ``price_history`` in the snapshot).
_EXPECTED_MANIFEST_KEYS = {
    "data/curated/earnings_events/year=2020/part-0000.parquet",
    "data/curated/earnings_events/year=2021/part-0000.parquet",
    "data/curated/daily_market/year=2020/part-0000.parquet",
    "data/curated/daily_market/year=2020/part-0001.parquet",
    "data/curated/daily_market/year=2021/part-0000.parquet",
    "data/curated/trades/year=2020/part-0000.parquet",
    "data/curated/trades/year=2021/part-0000.parquet",
    "data/curated/option_chains/year=2020/part-0000.parquet",
    "data/curated/option_chains/year=2021/part-0000.parquet",
    "data/features/panel.parquet",
    "data/features/tier4_forecasts.parquet",
    LEGACY_SNAPSHOT_PATH,
    REGISTRY_PATH,
    CALENDAR_PATH,
}


def test_materialize_progress_goes_to_stderr_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(materialize_progress, "_PROGRESS_INTERVAL_S", 0.0)
    conn, store, snap = _build_snapshot(tmp_path)
    try:
        repository = Repository(conn, store)
        request = _build_request(repository, snap, _snapshot_object_ref(store), store)
        dest_root = tmp_path / "materialized"

        manifest = legacy_adapter.materialize(repository, store, request, dest_root)

        assert set(manifest) == _EXPECTED_MANIFEST_KEYS
        captured = capsys.readouterr()
        assert "[materialize]" in captured.err
        assert "tables" in captured.err
        assert "[materialize]" not in captured.out
        # Start and done lines bracket the run with elapsed seconds.
        assert "start" in captured.err
        assert "done in" in captured.err
        assert "peak RSS" in captured.err
        # The final done==total tick for both write and validation phases.
        assert "6/6 tables" in captured.err
    finally:
        conn.close()


def test_table_progress_respects_interval_and_final_tick(capsys, monkeypatch):
    """One line at most every 30 s; ``done == total`` always prints; finish prints ``done in``."""
    fake = {"now": 0.0}
    monkeypatch.setattr(materialize_progress.time, "monotonic", lambda: fake["now"])
    progress = materialize_progress.TableProgress(3, "tables written")
    capsys.readouterr()  # drop the construction start line

    fake["now"] = 1.0
    progress.tick()
    fake["now"] = 2.0
    progress.tick()
    assert capsys.readouterr().err == ""  # ticks 1 and 2 are inside the 30 s interval

    fake["now"] = 3.0
    progress.tick()  # done == total always prints
    assert "3/3 tables written" in capsys.readouterr().err

    fake["now"] = 3.5
    progress.finish()
    assert "done in" in capsys.readouterr().err


def test_table_progress_interval_zero_prints_every_tick(capsys, monkeypatch):
    fake = {"now": 0.0}
    monkeypatch.setattr(materialize_progress.time, "monotonic", lambda: fake["now"])
    monkeypatch.setattr(materialize_progress, "_PROGRESS_INTERVAL_S", 0.0)
    progress = materialize_progress.TableProgress(3)
    capsys.readouterr()

    for done, now in ((1, 1.0), (2, 2.0), (3, 3.0)):
        fake["now"] = now
        progress.tick()
        err = capsys.readouterr().err
        assert f"{done}/3 tables" in err, err

    fake["now"] = 4.0
    progress.finish()
    assert "done in" in capsys.readouterr().err
