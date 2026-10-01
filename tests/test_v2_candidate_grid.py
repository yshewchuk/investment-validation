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


def _wide_chain(ticker, obs_date, *, call=(2.0, 2.4), put=(1.0, 1.4), spot=100.0):
    """Same row contract as ``tests.test_v2_research_replay._chain``, but a
    five-strike ladder wide enough for TWIN-P ``steps=1`` and ``steps=2``."""
    rows = []
    obs = pd.Timestamp(obs_date)
    for expiry, dte in ((pd.Timestamp("2024-05-03"), 2), (pd.Timestamp("2024-05-24"), 23)):
        for strike in (80.0, 90.0, 100.0, 110.0, 120.0):
            for right, (bid, ask) in (("C", call), ("P", put)):
                scale = 1.0 if dte < 10 else 2.0
                rows.append(
                    {
                        "ticker": ticker, "obs_date": obs, "expiry": expiry, "dte": dte,
                        "strike": strike, "right": right,
                        "bid": bid * scale, "ask": ask * scale,
                        "spot": spot, "quote_repaired": False,
                    }
                )
    return pd.DataFrame(rows)


def _wide_chain_rows(exit_call=(3.0, 3.4), exit_put=(2.0, 2.4)) -> list[dict]:
    rows: list[dict] = []
    for obs_date in ("2024-05-02", "2024-05-03"):
        call = (2.0, 2.4) if obs_date == "2024-05-02" else exit_call
        put = (1.0, 1.4) if obs_date == "2024-05-02" else exit_put
        frame = _wide_chain("TEST", obs_date, call=call, put=put)
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
    for step in (1, 2):
        structure = _pricing.STRUCTURES["TWIN-P"](steps=step)
        expected[step] = sum(
            len(replay_one(structure, row, index)[0])
            for row in plan.frame.to_dict("records")
        )

    grid = _grid(tmp_path, snapshot.snapshot_id, events, steps=(1, 2))
    conn.close()

    assert all(count > 0 for count in expected.values())
    counts = {int(step): int(size) for step, size in grid.groupby("steps").size().items()}
    assert counts == expected
    assert set(grid["strategy"]) == {"TWIN-P"}


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
