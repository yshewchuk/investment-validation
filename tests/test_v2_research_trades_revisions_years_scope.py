"""Tier-0 tests for the ``--years`` scope of ``revisions_for_rebuild`` (#108).

A ``--years``-scoped rebuild must tombstone a rebuilt strategy's stale rows
ONLY in the years it rebuilt; a full rebuild (``years=None``) must keep
tombstoning every stale year, exactly as before. Pure and in-memory: no
repository, snapshot, or tmp_path machinery.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.research._trades_revisions import (  # noqa: E402
    PROVENANCE,
    revisions_for_rebuild,
)

STRATEGY = "STR-THRU"


def _trade_row(trade_id: str, strategy: str, *, year: int) -> dict:
    event_date = pd.Timestamp(f"{year}-05-02")
    return {
        "trade_id": trade_id, "kind": "sim", "strategy": strategy,
        "variant": "e+0_x+1", "ticker": "TEST",
        "event_id": f"TEST_{year}-05-02", "event_date": event_date,
        "year": year, "legs": "{}", "entry_date": event_date,
        "exit_date": pd.Timestamp(f"{year}-05-03"), "strike": 100.0,
        "expiry": pd.Timestamp(f"{year}-05-03"), "fill_alpha": 0.0,
        "entry_cost": 3.8, "exit_value": 5.0,
        "ret": (5.0 - 3.8) / 3.8, "provenance": PROVENANCE,
    }


def _row_ids(revisions, *, deleted: bool) -> list[str]:
    return [json.loads(rev.candidate.logical_key)[0]
            for rev in revisions if rev.deleted is deleted]


def _existing_two_years() -> pd.DataFrame:
    return pd.DataFrame([
        _trade_row("STR-THRU:old:2024", STRATEGY, year=2024),
        _trade_row("STR-THRU:old:2025", STRATEGY, year=2025),
    ])


def test_scoped_rebuild_leaves_other_years_intact():
    existing = _existing_two_years()

    revisions = revisions_for_rebuild(
        existing, pd.DataFrame(), {STRATEGY}, years=[2024])

    assert _row_ids(revisions, deleted=True) == ["STR-THRU:old:2024"]
    assert "STR-THRU:old:2025" not in _row_ids(revisions, deleted=True)


def test_full_rebuild_unscoped_tombstones_every_year():
    existing = _existing_two_years()

    revisions = revisions_for_rebuild(existing, pd.DataFrame(), {STRATEGY})
    implicit = revisions_for_rebuild(
        existing, pd.DataFrame(), {STRATEGY}, years=None)

    for result in (revisions, implicit):
        assert sorted(_row_ids(result, deleted=True)) == [
            "STR-THRU:old:2024", "STR-THRU:old:2025"]


def test_scoped_rebuild_still_tombstones_its_own_years_dead_rows():
    existing = _existing_two_years()
    engine_rows = pd.DataFrame(
        [_trade_row("STR-THRU:new:2024", STRATEGY, year=2024)])

    revisions = revisions_for_rebuild(
        existing, engine_rows, {STRATEGY}, years=[2024])

    assert _row_ids(revisions, deleted=True) == ["STR-THRU:old:2024"]
    assert _row_ids(revisions, deleted=False) == ["STR-THRU:new:2024"]
