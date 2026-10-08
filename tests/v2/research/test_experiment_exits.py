"""Declared-exit consumption over real synthetic Parquet snapshots, no mark stubs."""
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import experiment_spec_from_document, resolve_experiment_plan
from engine.v2.research.experiment_exits import EnteredPosition, PositionLeg, walk_fixed_day
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

DATES = ("2024-05-24", "2024-05-28", "2024-05-29", "2024-05-30")


def _spec(days=2, alpha=0.25):
    return experiment_spec_from_document({
        "experiment_id": "synthetic-exit", "hypothesis": "synthetic",
        "primary_arm_id": "fixed", "arms": ["fixed"], "folds": ["fold"],
        "economic_params": {"exit": {"kind": "fixed_day", "trading_days": days},
                            "fill": alpha},
        "price_source": "option_chains",
    })


def _position():
    return EnteredPosition("trade-a", "TEST", DATES[0], (
        PositionLeg("2024-06-21", 100.0, "C", "buy", 2.0),
        PositionLeg("2024-06-21", 105.0, "P", "sell", 1.0),
    ))


def _rows():
    rows = []
    for offset, day in enumerate(DATES):
        for strike, right, bid in ((100.0, "C", 2.0 + offset),
                                   (105.0, "P", 1.0 + offset / 2)):
            rows.append({"ticker": "TEST", "obs_date": pd.Timestamp(day),
                         "expiry": pd.Timestamp("2024-06-21"), "strike": strike,
                         "right": right, "bid": bid, "ask": bid + 0.4,
                         "dte": 20, "spot": 100.0, "quote_repaired": False, "year": 2024})
    return rows


def _snapshot(conn, clock, store, rows, *, scope="shadow", receipt_id="r1", dates=DATES):
    tables, contracts = {}, {}
    sources = {"option_chains": rows, "daily_market": [
        {"ticker": "MKT", "date": pd.Timestamp(day), "year": 2024} for day in dates]}
    for name, source in sources.items():
        contract = contracts[name] = contract_for(name)
        tables[name] = ([publish_and_inspect(store, contract, contract_ref_for(contract),
                                           source, "2024")] if source else [])
    return commit_tables(conn, clock, tables, contracts, scope=scope,
                         receipt_id=receipt_id, store=store)


@pytest.fixture
def source(tmp_path):
    conn, clock, store = catalog_and_store(tmp_path)
    yield conn, clock, store
    conn.close()


def _walk(source, *, rows=None, days=2, alpha=0.25, positions=None, dates=DATES):
    conn, clock, store = source
    snapshot = _snapshot(conn, clock, store, _rows() if rows is None else rows, dates=dates)
    plan = resolve_experiment_plan(_spec(days, alpha))
    return walk_fixed_day(Repository(conn, store), snapshot,
                          (_position(),) if positions is None else positions,
                          economic_params=plan.economic_params), snapshot


@pytest.mark.parametrize("alpha,expected", [(0.0, 1.8), (0.25, 2.4), (0.5, 3.0),
                                           (0.75, 3.6), (1.0, 4.2)])
def test_fill_and_provenance(source, alpha, expected):
    decisions, snapshot = _walk(source, alpha=alpha)
    result, = decisions
    # Long two calls, short one put; entry costs 3.8 - 1.2*a and
    # exit receives 5.6 + 1.2*a. P&L = 1.8 + 2.4*a.
    assert result.entry_cost == pytest.approx(3.8 - 1.2 * alpha)
    assert result.exit_value == pytest.approx(5.6 + 1.2 * alpha)
    assert result.pnl == pytest.approx(expected)
    assert result.reason == "fixed_day"
    assert result.pnl_basis == "mark_based"
    assert result.mark_source == "option_chains"
    assert result.fill_convention == "alpha_ladder"
    assert result.fill_alpha == alpha
    assert result.snapshot_id == snapshot.snapshot_id
    assert result.trade_id == "trade-a"
    with pytest.raises(FrozenInstanceError):
        result.pnl = 0


def test_trading_days(source):
    (result,), _ = _walk(source)
    assert result.exit_date == "2024-05-29"
    assert result.visited_dates == DATES[:3]


@pytest.mark.parametrize("missing_day", DATES[:3])
def test_mark_refusal(source, missing_day):
    # First position can succeed; the second lacks a required put mark.
    call_only = replace(_position(), trade_id="call-only", legs=_position().legs[:1])
    rows = [row for row in _rows()
            if not (row["obs_date"] == pd.Timestamp(missing_day) and row["right"] == "P")]
    with pytest.raises(DataError) as caught:
        _walk(source, rows=rows, positions=(call_only, _position()))
    assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"
    assert caught.value.problem.category == "internal"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.details == {"trade_id": "trade-a", "session": missing_day}


def test_recipe_identity(source):
    conn, clock, store = source
    snapshot = _snapshot(conn, clock, store, _rows())
    repository = Repository(conn, store)
    short, long = _spec(1), _spec(3)
    first, second = resolve_experiment_plan(short), resolve_experiment_plan(long)
    assert short.spec_hash != long.spec_hash
    assert first.json_bytes() != second.json_bytes()
    results = [walk_fixed_day(repository, snapshot, (_position(),),
                              economic_params=p.economic_params)[0] for p in (first, second)]
    assert [r.exit_date for r in results] == [DATES[1], DATES[3]]
    assert [r.pnl for r in results] == pytest.approx([0.9, 3.9])
    short.economic_params["exit"]["trading_days"] = 3
    assert first.economic_params["exit"]["trading_days"] == 1
    with pytest.raises(TypeError):
        first.economic_params["exit"]["trading_days"] = 2


@pytest.mark.parametrize("days", [0, -1, True, 2.0, "2", None])
def test_recipe_validation(days):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        resolve_experiment_plan(_spec(days))


@pytest.mark.parametrize("economics", [
    {"exit": {"kind": "stop", "trading_days": 2}, "fill": 0.25},
    {"exit": {"kind": "fixed_day", "trading_days": 2, "unused": 1}, "fill": 0.25},
    {"exit": None, "fill": 0.25},
    {"exit": {"kind": "fixed_day", "trading_days": 2}},
    {"exit": {"kind": "fixed_day", "trading_days": 2}, "fill": True},
    {"exit": {"kind": "fixed_day", "trading_days": 2}, "fill": "mid"},
    {"exit": {"kind": "fixed_day", "trading_days": 2}, "fill": float("nan")},
    {"exit": {"kind": "fixed_day", "trading_days": 2}, "fill": 1.1},
])
def test_recipe_fields(economics):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        resolve_experiment_plan(replace(_spec(), economic_params=economics))


def test_source_contract():
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        resolve_experiment_plan(replace(_spec(), price_source="other"))
    legacy = replace(_spec(), economic_params={"fill": "mid"}, price_source="legacy")
    assert resolve_experiment_plan(legacy).economic_params == {"fill": "mid"}


def test_calendar_bound(source):
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        _walk(source, days=4)


def test_entry_session(source):
    position = replace(_position(), entry_date="2024-05-27")
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        _walk(source, positions=(position,))


def test_calendar_refusal(source):
    with pytest.raises(DataError, match="CALENDAR_UNAVAILABLE"):
        _walk(source, dates=())


@pytest.mark.parametrize("bid,ask", [(float("nan"), 2.0), (-1.0, 2.0), (3.0, 2.0)])
def test_bad_mark(source, bid, ask):
    rows = _rows()
    rows[2].update(bid=bid, ask=ask)
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        _walk(source, rows=rows)


@pytest.mark.parametrize("index", [0, 2, 4])
def test_zero_market_refusal(source, index):
    rows = _rows()
    rows[index].update(bid=0.0, ask=0.0)
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED") as caught:
        _walk(source, rows=rows)
    assert caught.value.problem.details["session"] == DATES[index // 2]


@pytest.mark.parametrize("index", [0, 2, 4])
def test_zero_bid_mark(source, index):
    rows = _rows()
    rows[index].update(bid=0.0, ask=0.4)
    decisions, _ = _walk(source, rows=rows)
    assert len(decisions) == 1
    assert decisions[0].visited_dates == DATES[:3]


def test_empty_chains(source):
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        _walk(source, rows=[])


def test_no_replacement(source):
    rows = _rows()
    rows[2]["strike"] = 100.000001
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        _walk(source, rows=rows)


def test_expiry_bound(source):
    expiry = "2024-05-28"
    position = replace(_position(), legs=tuple(replace(leg, expiry=expiry)
                                               for leg in _position().legs))
    rows = [dict(row, expiry=pd.Timestamp(expiry)) for row in _rows()]
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        _walk(source, rows=rows, positions=(position,))


@pytest.mark.parametrize("changes", [{"legs": ()}, {"entry_date": "20240524"},
                                     {"ticker": ""}, {"trade_id": ""}])
def test_position_validation(source, changes):
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        _walk(source, positions=(replace(_position(), **changes),))


def test_direct_recipe_validation():
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        walk_fixed_day(None, None, (), economic_params={"exit": {"kind": "stop"}, "fill": 0.5})


def test_pinned_replay(source):
    conn, clock, store = source
    pinned = _snapshot(conn, clock, store, _rows())
    changed = _rows()
    changed[4]["bid"], changed[4]["ask"] = 9.0, 9.4
    newer = _snapshot(conn, clock, store, changed, scope="newer", receipt_id="r2")
    repository = Repository(conn, store)
    assert repository.resolve_pinned("newer").snapshot_id == newer.snapshot_id
    economics = resolve_experiment_plan(_spec()).economic_params
    writes = conn.total_changes
    objects = {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()}
    first = walk_fixed_day(repository, pinned, (_position(),), economic_params=economics)
    again = walk_fixed_day(repository, pinned, (_position(),), economic_params=economics)
    other = walk_fixed_day(repository, newer, (_position(),), economic_params=economics)
    assert first == again
    assert first[0].snapshot_id == pinned.snapshot_id
    assert first[0].pnl != other[0].pnl
    assert conn.total_changes == writes
    assert {p: p.read_bytes() for p in store.root.rglob("*") if p.is_file()} == objects
