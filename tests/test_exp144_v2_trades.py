"""Tier-0 tests for the v2 routing helpers EXP-144's runner now exposes.

EXP-144's full ``main()`` needs the feature panel, SPY daily history and the
EXP-142 factor-simulation module, none of which are test fixtures, so these
tests import its ``run.py`` as a module -- the same
``importlib.util.spec_from_file_location`` technique
``experiments/EXP-185_str_runup_t14_corrected_calendar_gate_rebaseline_registered/run.py``
uses -- and call the standalone helpers directly. ``main()`` is never run.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.research import experiment_trades  # noqa: E402
from tests.data_scan_support import catalog_and_store  # noqa: E402
from tests.test_v2_research_build_trades import _commit_all  # noqa: E402

PROVENANCE = experiment_trades.PROVENANCE
LEGACY_PROVENANCE = "engine.replay"
RUNNER_DIR = "EXP-144_str_runup_t14_corrected_calendar_gate_rebaseline"

#: gate_midfill_str_runup's registered threshold, as EXP-144's spec.yaml
#: incumbent block pins it.
STORED_THRESHOLD = 0.0725137593996064

#: The columns EXP-144's ``load_trades`` projects to, in its own order.
PROJECTED_COLUMNS = [
    "trade_id", "kind", "strategy", "variant", "ticker", "event_id",
    "event_date", "legs", "entry_date", "exit_date", "strike", "expiry",
    "fill_alpha", "entry_cost", "exit_value", "ret", "provenance",
]
DATE_COLUMNS = ("event_date", "entry_date", "exit_date", "expiry")


def _exp144():
    """A fresh module object per call, so a test can point ``V2_CATALOG`` /
    ``V2_STORE_ROOT`` at its own ``tmp_path`` without touching another test's
    (or the real ``private/ops`` root's) view of them."""
    source = ROOT / "experiments" / RUNNER_DIR / "run.py"
    module_spec = importlib.util.spec_from_file_location(
        "exp144_helpers_under_test", source
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


def _runup_row(trade_id, variant, provenance, *, fill_alpha=0.5):
    """One STR-RUNUP trade row, shaped like the v2 ``trades`` contract.

    ``event_id`` is the ``tests/test_v2_research_replay._event_rows`` event, so
    ``experiment_trades.load_trades`` finds a session to join.
    """
    event_date = pd.Timestamp("2024-05-02")
    return {
        "trade_id": trade_id, "kind": "sim", "strategy": "STR-RUNUP",
        "variant": variant, "ticker": "TEST", "event_id": "TEST_2024-05-02",
        "event_date": event_date, "year": 2024, "legs": "{}",
        "entry_date": pd.Timestamp("2024-04-30"),
        "exit_date": event_date, "strike": 100.0,
        "expiry": pd.Timestamp("2024-06-21"), "fill_alpha": fill_alpha,
        "entry_cost": 3.8, "exit_value": 4.2, "ret": (4.2 - 3.8) / 3.8,
        "provenance": provenance,
    }


def test_require_v2_snapshot_id_refuses_a_missing_or_null_pin():
    module = _exp144()

    for spec in ({"id": "EXP-144"}, {"id": "EXP-144", "v2_snapshot_id": None}):
        with pytest.raises(SystemExit) as excinfo:
            module.require_v2_snapshot_id(spec)
        assert "v2_snapshot_id" in str(excinfo.value)

    assert module.require_v2_snapshot_id(
        {"id": "EXP-185", "v2_snapshot_id": "snap_abc"}
    ) == "snap_abc"


def test_load_trades_reads_the_pinned_v2_snapshot(tmp_path):
    module = _exp144()
    module.V2_CATALOG = tmp_path / "catalog.sqlite"
    module.V2_STORE_ROOT = tmp_path / "store"
    conn, clock, store = catalog_and_store(tmp_path)
    wanted = module.VARIANT
    snapshot = _commit_all(conn, clock, store, receipt_id="r1", trades_rows=[
        _runup_row("T-RUNUP-MID", wanted, PROVENANCE, fill_alpha=0.5),
        _runup_row("T-RUNUP-WORST", wanted, PROVENANCE, fill_alpha=0.0),
        # planted decoys: a different variant, and a legacy-provenance row for
        # the wanted variant. Neither may survive the v2 read plus the filter.
        _runup_row("T-OTHER-VARIANT", "e+0_x+1", PROVENANCE),
        _runup_row("T-LEGACY-PROVENANCE", wanted, LEGACY_PROVENANCE),
    ])

    frame = module.load_trades(snapshot.snapshot_id)
    conn.close()

    assert sorted(frame["trade_id"].astype(str)) == ["T-RUNUP-MID", "T-RUNUP-WORST"]
    assert set(frame["event_id"].astype(str)) == {"TEST_2024-05-02"}
    assert set(frame["variant"].astype(str)) == {wanted}
    assert list(frame.columns) == PROJECTED_COLUMNS
    for column in DATE_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(frame[column]), column


def test_add_champion_decisions_selects_on_the_stored_threshold():
    module = _exp144()
    scores = pd.DataFrame({
        "event_id": ["E-LOW", "E-AT", "E-HIGH", "E-ZERO"],
        "incumbent_complete_case": [
            0.05, STORED_THRESHOLD, STORED_THRESHOLD + 0.02, 0.0,
        ],
        "incumbent_complete_case_pwin": [0.10, 0.40, 0.90, 0.05],
    })

    out = module.add_champion_decisions(scores, STORED_THRESHOLD)

    assert list(out["selected_champion"]) == [False, True, True, False]
    assert list(out["champion_pwin"]) == list(scores["incumbent_complete_case_pwin"])
    # the candidate's own columns are untouched by the copy
    assert "selected_champion" not in scores.columns
