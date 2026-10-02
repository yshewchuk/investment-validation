"""Tier-0 tests for ``experiments.v2_candidate_grid.price_candidate_grid``.

The same fixture machinery ``tests/test_common_v2_chain_quotes.py`` uses is
reused here (``tests/data_scan_support.catalog_and_store`` plus
``tests/test_v2_research_replay``'s ``_commit``/``_chain_rows``/``_event_rows``).
The happy path prices on a five-strike ladder wide enough for TWIN-P's
``steps=1`` and ``steps=2`` and compares each step's row count to a direct
``replay_one`` reference; the chain-read edge is spied in the grid module's own
namespace, where it is called.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.research import _chains, _plan, _pricing  # noqa: E402
from engine.v2.research.replay import replay_one  # noqa: E402
from experiments import v2_candidate_grid  # noqa: E402
from tests.data_scan_support import catalog_and_store  # noqa: E402
from tests.test_v2_research_replay import (  # noqa: E402
    _chain_rows,
    _commit,
    _event_rows,
)


def _wide_chain(ticker, obs_date, *, level=1.0, spot=100.0):
    """Seventeen $2-spaced strikes (84..116) around ``spot`` -- wide and
    dense enough for TWIN-P's full seven-leg layout (offsets at 1x/2x/4x the
    grid-step distance) at BOTH ``steps=1`` and ``steps=2``. Put/call mid is
    convex in distance from spot and the half-spread widens with that same
    distance, so every (step, alpha) prices to a genuinely nonzero net debit
    rather than a flat/degenerate one. ``level`` scales every quote, so the
    entry (2024-05-02) and exit (2024-05-03) observations can differ.
    """
    rows = []
    obs = pd.Timestamp(obs_date)
    strikes = [84.0 + 2.0 * i for i in range(17)]  # 84..116, $2 steps, spot at index 8
    for expiry, dte in ((pd.Timestamp("2024-05-03"), 2), (pd.Timestamp("2024-05-24"), 23)):
        dte_scale = 1.0 if dte < 10 else 2.0
        for strike in strikes:
            steps_from_spot = abs(strike - spot) / 2.0
            put_mid = level * dte_scale * (2.0 + 0.20 * steps_from_spot ** 2)
            call_mid = level * dte_scale * (2.0 + 0.20 * steps_from_spot ** 2)
            half_spread = dte_scale * (0.10 + 0.02 * steps_from_spot)
            rows.append({
                "ticker": ticker, "obs_date": obs, "expiry": expiry, "dte": dte,
                "strike": strike, "right": "P",
                "bid": put_mid - half_spread, "ask": put_mid + half_spread,
                "spot": spot, "quote_repaired": False,
            })
            rows.append({
                "ticker": ticker, "obs_date": obs, "expiry": expiry, "dte": dte,
                "strike": strike, "right": "C",
                "bid": call_mid - half_spread, "ask": call_mid + half_spread,
                "spot": spot, "quote_repaired": False,
            })
    return pd.DataFrame(rows)


def _wide_chain_rows(exit_level=1.5) -> list[dict]:
    rows: list[dict] = []
    for obs_date, level in (("2024-05-02", 1.0), ("2024-05-03", exit_level)):
        frame = _wide_chain("TEST", obs_date, level=level)
        frame["year"] = 2024
        rows.extend(frame.to_dict("records"))
    rows.sort(key=lambda row: (row["ticker"], row["obs_date"], row["expiry"],
                               row["strike"], row["right"]))
    return rows


def _events() -> pd.DataFrame:
    return pd.DataFrame(_event_rows())


def _grid(tmp_path, snapshot_id, events, **kwargs) -> pd.DataFrame:
    return v2_candidate_grid.price_candidate_grid(
        "TWIN-P", events, catalog=tmp_path / "catalog.sqlite",
        store_root=tmp_path / "store", snapshot_id=snapshot_id, **kwargs,
    )


def test_price_candidate_grid_matches_direct_replay_one_per_step(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(conn, clock, store, chain_rows=_wide_chain_rows(),
                       event_rows=_event_rows(), receipt_id="r1")
    events = _events()

    repository = Repository(conn, store)
    snapshot_ref = repository.resolve(snapshot.snapshot_id)
    calendar = _pricing.trading_calendar_from_snapshot(repository, snapshot_ref)
    plan = _plan.plan_events(
        _pricing.STRUCTURES["TWIN-P"](steps=1), events, calendar=calendar)
    plan = _chains.filter_plan_by_availability(
        plan, _chains.read_chain_keys(repository, snapshot_ref))
    index = _chains.load_chain_index(repository, snapshot_ref, plan.chain_keys)
    expected = {}
    expected_variant = {}
    for step in (1, 2):
        structure = _pricing.STRUCTURES["TWIN-P"](steps=step)
        expected_variant[step] = _pricing.execution_variant_label(structure)
        expected[step] = [
            priced_row
            for row in plan.frame.to_dict("records")
            for priced_row in replay_one(structure, row, index)[0]
        ]
    assert expected_variant[1] != expected_variant[2]

    grid = _grid(tmp_path, snapshot.snapshot_id, events, steps=(1, 2))
    conn.close()

    assert all(len(rows) > 0 for rows in expected.values())
    counts = {int(step): int(size) for step, size in grid.groupby("steps").size().items()}
    assert counts == {step: len(rows) for step, rows in expected.items()}
    assert set(grid["strategy"]) == {"TWIN-P"}

    key = ["ticker", "event_id", "event_date", "fill_alpha"]
    for step in (1, 2):
        step_slice = grid[grid["steps"] == step]
        assert (step_slice["variant"] == expected_variant[step]).all()
        expected_df = pd.DataFrame(expected[step])
        shared = list(expected_df.columns)
        expected_sorted = expected_df.sort_values(key, kind="mergesort").reset_index(drop=True)
        grid_sorted = (
            step_slice[shared]
            .sort_values(key, kind="mergesort")
            .reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(expected_sorted, grid_sorted,
                                      rtol=1e-12, atol=1e-12)


def test_price_candidate_grid_comparison_catches_a_corrupted_value(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(conn, clock, store, chain_rows=_wide_chain_rows(),
                       event_rows=_event_rows(), receipt_id="r1")
    events = _events()

    repository = Repository(conn, store)
    snapshot_ref = repository.resolve(snapshot.snapshot_id)
    calendar = _pricing.trading_calendar_from_snapshot(repository, snapshot_ref)
    plan = _plan.plan_events(
        _pricing.STRUCTURES["TWIN-P"](steps=1), events, calendar=calendar)
    plan = _chains.filter_plan_by_availability(
        plan, _chains.read_chain_keys(repository, snapshot_ref))
    index = _chains.load_chain_index(repository, snapshot_ref, plan.chain_keys)
    expected = {}
    for step in (1, 2):
        structure = _pricing.STRUCTURES["TWIN-P"](steps=step)
        expected[step] = [
            priced_row
            for row in plan.frame.to_dict("records")
            for priced_row in replay_one(structure, row, index)[0]
        ]

    grid = _grid(tmp_path, snapshot.snapshot_id, events, steps=(1, 2))
    conn.close()

    key = ["ticker", "event_id", "event_date", "fill_alpha"]
    step_slice = grid[grid["steps"] == 1]
    expected_df = pd.DataFrame(expected[1])
    assert "entry_cost" in expected_df.columns
    shared = list(expected_df.columns)
    expected_sorted = expected_df.sort_values(key, kind="mergesort").reset_index(drop=True)
    grid_sorted = (
        step_slice[shared]
        .sort_values(key, kind="mergesort")
        .reset_index(drop=True)
    )
    corrupted = grid_sorted.copy()
    corrupted.loc[corrupted.index[0], "entry_cost"] = (
        corrupted.loc[corrupted.index[0], "entry_cost"] + 1.0
    )
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(expected_sorted, corrupted,
                                      rtol=1e-12, atol=1e-12)


def test_price_candidate_grid_reads_chains_once_for_the_whole_grid(tmp_path, monkeypatch):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(conn, clock, store, chain_rows=_wide_chain_rows(),
                       event_rows=_event_rows(), receipt_id="r1")
    calls: list = []
    real = v2_candidate_grid.load_v2_chain_quotes

    def spy(keys, *, catalog, store_root, snapshot_id):
        calls.append((set(keys), snapshot_id))
        return real(keys, catalog=catalog, store_root=store_root, snapshot_id=snapshot_id)

    monkeypatch.setattr(v2_candidate_grid, "load_v2_chain_quotes", spy)
    grid = _grid(tmp_path, snapshot.snapshot_id, _events(), steps=(1, 2))
    conn.close()

    assert set(grid["steps"]) == {1, 2}
    assert len(calls) == 1
    keys, pinned = calls[0]
    assert pinned == snapshot.snapshot_id
    assert keys == {("TEST", pd.Timestamp("2024-05-02")),
                    ("TEST", pd.Timestamp("2024-05-03"))}


def test_price_candidate_grid_returns_empty_frame_when_plan_or_availability_is_empty(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    snapshot = _commit(conn, clock, store, chain_rows=_chain_rows(),
                       event_rows=_event_rows(), receipt_id="r1")

    no_events = pd.DataFrame(
        {"event_id": [], "ticker": [], "event_date": [], "session": []}
    )
    empty_by_plan = _grid(tmp_path, snapshot.snapshot_id, no_events, steps=(1, 2))

    elsewhere = _events()
    elsewhere["ticker"] = "NOPE"
    empty_by_availability = _grid(
        tmp_path, snapshot.snapshot_id, elsewhere, steps=(1, 2))
    conn.close()

    for frame in (empty_by_plan, empty_by_availability):
        assert list(frame.columns) == ["strategy", "steps"]
        assert frame.empty
