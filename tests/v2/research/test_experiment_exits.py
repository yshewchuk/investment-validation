"""Declared-exit consumption over real synthetic Parquet snapshots, no mark stubs."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.ops.experiments import experiment_spec_from_document, resolve_experiment_plan
from engine.v2.research.experiment_exits import (
    EnteredPosition,
    PositionLeg,
    exit_report_frame,
    walk_exit,
    walk_fixed_day,
)
from tests.data_scan_support import (
    catalog_and_store,
    commit_tables,
    contract_for,
    contract_ref_for,
    publish_and_inspect,
)

DATES = ("2024-05-24", "2024-05-28", "2024-05-29", "2024-05-30")
REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents
                 if (parent / "checks/layer_map.py").is_file())


def test_recipe_contract_table_boundary():
    document = REPO_ROOT / "engine/v2/ops/ARCHITECTURE.md"
    before, table = document.read_text().split("| Experiment execution condition | Outcome |", 1)
    assert before.endswith("\n\n"), "the experiment table needs its own Markdown block"
    assert "invalid fixed-day or target/stop exit recipe/source/fill" in table


def test_failure_contract_table():
    document = REPO_ROOT / "engine/v2/research/ARCHITECTURE.md"
    rows = document.read_text().splitlines()
    for condition in ("Malformed position or leg record",
                      "Missing required leg mark or mark past held-leg expiry",
                      "Unusable quote reaching pricing",
                      "Insufficient calendar coverage"):
        row, = [line for line in rows if line.startswith(f"| {condition} |")]
        assert "Non-retryable `EXPERIMENT_VARIANT_FAILED`" in row
        assert "whole call fails, no excluded trade or partial tuple" in row


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


def _rows_falling():
    rows = _rows()
    for offset in range(len(DATES)):
        call, put = rows[2 * offset], rows[2 * offset + 1]
        call["bid"], call["ask"] = 6.0 - 2.0 * offset, 6.4 - 2.0 * offset
        put["bid"], put["ask"] = 0.5, 0.9
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


def _exit_spec(recipe, alpha=0.25):
    return experiment_spec_from_document({
        "experiment_id": "synthetic-exit", "hypothesis": "synthetic",
        "primary_arm_id": "fixed", "arms": ["fixed"], "folds": ["fold"],
        "economic_params": {"exit": recipe, "fill": alpha},
        "price_source": "option_chains",
    })


def _walk_target_stop(source, *, rows=None, target=2.0, stop=-1.0, days=3,
                      alpha=0.25, positions=None, dates=DATES):
    conn, clock, store = source
    snapshot = _snapshot(conn, clock, store, _rows() if rows is None else rows, dates=dates)
    recipe = {"kind": "target_stop", "trading_days": days,
              "target_pnl": target, "stop_pnl": stop}
    plan = resolve_experiment_plan(_exit_spec(recipe, alpha))
    return walk_exit(Repository(conn, store), snapshot,
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
    assert result.exit_fill_alpha == alpha
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


def test_target_stop_mark_refusal(source):
    # Both held contracts' rows are gone for an intermediate session the walk
    # must reach before any threshold crossing; the whole target/stop call
    # refuses with the same typed missing-mark failure as the fixed-day walk.
    rows = [row for row in _rows() if row["obs_date"] != pd.Timestamp(DATES[1])]
    with pytest.raises(DataError) as caught:
        _walk_target_stop(source, rows=rows, target=100.0, stop=-100.0)
    assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"
    assert caught.value.problem.category == "internal"
    assert caught.value.problem.retryable is False
    assert caught.value.problem.details == {"trade_id": "trade-a", "session": DATES[1]}


def test_target_stop_expiry_bound(source):
    # A held leg expiring before the target/stop horizon's final day refuses
    # once the walker reaches the first session past that expiry.
    expiry = "2024-05-29"
    position = replace(_position(), legs=tuple(replace(leg, expiry=expiry)
                                               for leg in _position().legs))
    rows = [dict(row, expiry=pd.Timestamp(expiry)) for row in _rows()
            if row["obs_date"] <= pd.Timestamp(expiry)]
    with pytest.raises(DataError) as caught:
        _walk_target_stop(source, rows=rows, target=100.0, stop=-100.0,
                          positions=(position,))
    assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"
    assert caught.value.problem.details == {"trade_id": "trade-a",
                                            "session": DATES[3]}


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
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        walk_fixed_day(None, None, (), economic_params=_spec(days).economic_params)


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
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        walk_fixed_day(None, None, (), economic_params=economics)


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


@pytest.mark.parametrize("record", [None, {}, "record", object()])
@pytest.mark.parametrize("kind", ["position", "leg"])
def test_malformed_records_refuse(source, record, kind):
    bad = record if kind == "position" else replace(_position(), legs=(record,))
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED") as caught:
        _walk(source, positions=(_position(), bad))
    assert caught.value.problem.retryable is False


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


REPORT_FIELDS = ("trade_id", "exit_reason", "exit_day", "mark_based_pnl", "pnl_basis",
                 "mark_source", "fill_convention", "fill_alpha", "exit_fill_alpha",
                 "snapshot_id", "ambiguous_exit")


def test_walk_exit_fixed_day_matches(source):
    conn, clock, store = source
    snapshot = _snapshot(conn, clock, store, _rows())
    repository = Repository(conn, store)
    economics = resolve_experiment_plan(_spec(2, 0.25)).economic_params
    via_exit = walk_exit(repository, snapshot, (_position(),), economic_params=economics)
    fixed = walk_fixed_day(repository, snapshot, (_position(),), economic_params=economics)
    assert via_exit == fixed
    assert via_exit[0].ambiguous_exit is False


def test_target_exit(source):
    (result,), _ = _walk_target_stop(source, target=2.0, stop=-1.0)
    assert result.reason == "target"
    assert result.exit_date == DATES[2]
    assert result.visited_dates == DATES[:3]
    assert result.pnl == pytest.approx(2.4)
    assert result.ambiguous_exit is False
    assert result.pnl_basis == "mark_based"
    assert result.mark_source == "option_chains"
    assert result.fill_convention == "alpha_ladder"
    assert result.fill_alpha == 0.25
    assert result.exit_fill_alpha == 0.25


def test_stop_exit(source):
    (result,), _ = _walk_target_stop(source, rows=_rows_falling(), target=5.0,
                                     stop=-3.0)
    assert result.reason == "stop"
    assert result.exit_date == DATES[1]
    assert result.visited_dates == DATES[:2]
    assert result.pnl == pytest.approx(-4.6)
    assert result.ambiguous_exit is False
    assert result.fill_alpha == 0.25
    assert result.exit_fill_alpha == 0.25


def test_spanning_quote_selects_ambiguous_stop(source):
    rows = _rows()
    rows[2]["bid"], rows[2]["ask"] = 0.1, 20.0
    (result,), _ = _walk_target_stop(source, rows=rows, target=1.0, stop=-1.0)
    assert result.reason == "stop"
    assert result.ambiguous_exit is True
    assert result.exit_date == DATES[1]
    assert result.fill_alpha == 0.25
    assert result.exit_fill_alpha == 0.0
    assert result.pnl < 0
    assert result.pnl <= -1.0


def test_no_hit_falls_back_to_fixed_day(source):
    (result,), _ = _walk_target_stop(source, target=100.0, stop=-100.0)
    assert result.reason == "fixed_day"
    assert result.exit_date == DATES[3]
    assert result.visited_dates == DATES
    assert result.pnl == pytest.approx(3.9)
    assert result.ambiguous_exit is False
    assert result.fill_alpha == 0.25
    assert result.exit_fill_alpha == 0.25


def test_changed_threshold_changes_plan_identity():
    first = resolve_experiment_plan(_exit_spec(
        {"kind": "target_stop", "trading_days": 3, "target_pnl": 2.0, "stop_pnl": -1.0}))
    second = resolve_experiment_plan(_exit_spec(
        {"kind": "target_stop", "trading_days": 3, "target_pnl": 3.0, "stop_pnl": -1.0}))
    assert first.json_bytes() != second.json_bytes()
    assert dict(first.economic_params["exit"])["target_pnl"] == 2.0


@pytest.mark.parametrize("recipe", [
    {"kind": "target_stop", "trading_days": 0, "target_pnl": 1.0, "stop_pnl": -1.0},
    {"kind": "target_stop", "trading_days": True, "target_pnl": 1.0, "stop_pnl": -1.0},
    {"kind": "target_stop", "trading_days": 2, "target_pnl": True, "stop_pnl": -1.0},
    {"kind": "target_stop", "trading_days": 2, "target_pnl": -1.0, "stop_pnl": 1.0},
    {"kind": "target_stop", "trading_days": 2, "target_pnl": 1.0, "stop_pnl": 0.0},
    {"kind": "target_stop", "trading_days": 2, "target_pnl": float("inf"), "stop_pnl": -1.0},
    {"kind": "target_stop", "trading_days": 2, "target_pnl": float("nan"), "stop_pnl": -1.0},
    {"kind": "target_stop", "trading_days": 2, "target_pnl": 10**400, "stop_pnl": -1.0},
    {"kind": "target_stop", "trading_days": 2, "target_pnl": 1.0, "stop_pnl": -(10**400)},
    {"kind": "target_stop", "trading_days": 2, "target_pnl": 1.0, "stop_pnl": -1.0,
     "unused": 1},
])
def test_target_stop_recipe_validation(recipe):
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        resolve_experiment_plan(_exit_spec(recipe))
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        walk_exit(None, None, (), economic_params={"exit": recipe, "fill": 0.25})


@pytest.mark.parametrize("economics", [
    {"exit": {"kind": "fixed_day", "trading_days": 2}, "fill": 10**400},
    {"exit": {"kind": "target_stop", "trading_days": 2, "target_pnl": 1.0,
              "stop_pnl": -1.0}, "fill": 10**400},
])
def test_oversized_fill_validation(economics):
    # An int past float range overflows ``math.isfinite``; the spec check
    # refuses it as an invalid experiment spec and the exit walk refuses it
    # as invalid exit economics.
    with pytest.raises(OpsError, match="INVALID_EXPERIMENT_SPEC"):
        resolve_experiment_plan(replace(_spec(), economic_params=economics))
    with pytest.raises(DataError, match="EXPERIMENT_VARIANT_FAILED"):
        walk_exit(None, None, (), economic_params=economics)


def test_report_frame_columns_and_provenance(source):
    decisions, snapshot = _walk_target_stop(source, target=2.0, stop=-1.0)
    frame = exit_report_frame(decisions)
    assert list(frame.columns) == list(REPORT_FIELDS)
    row = frame.iloc[0]
    assert row["trade_id"] == "trade-a"
    assert row["exit_reason"] == "target"
    assert row["exit_day"] == DATES[2]
    assert row["mark_based_pnl"] == pytest.approx(2.4)
    assert row["pnl_basis"] == "mark_based"
    assert row["mark_source"] == "option_chains"
    assert row["fill_convention"] == "alpha_ladder"
    assert row["fill_alpha"] == 0.25
    assert row["exit_fill_alpha"] == 0.25
    assert row["snapshot_id"] == snapshot.snapshot_id
    assert bool(row["ambiguous_exit"]) is False


def test_report_frame_empty_is_stable():
    frame = exit_report_frame(())
    assert list(frame.columns) == list(REPORT_FIELDS)
    assert len(frame) == 0


def _fraction_recipe(target=0.5, stop=-0.5):
    return {"kind": "target_stop", "trading_days": 3, "target_pnl": target,
            "stop_pnl": stop, "threshold_units": "entry_debit_fraction"}


def _fraction_walk(source, *, recipe=None, rows=None, position=None, alpha=0.25):
    conn, clock, store = source
    snapshot = _snapshot(conn, clock, store, _rows() if rows is None else rows)
    plan = resolve_experiment_plan(_exit_spec(_fraction_recipe() if recipe is None else recipe, alpha))
    result, = walk_exit(Repository(conn, store), snapshot,
                        (_position() if position is None else position,),
                        economic_params=plan.economic_params)
    return result


def _assert_fraction_refusal(error, *, code, category, message, details):
    assert error.code == error.problem.code == code
    assert error.problem.category == category
    assert error.problem.message == message
    assert error.problem.retryable is False
    assert error.problem.details == details


@pytest.mark.parametrize("price_scale,quantity_scale", [(1, 1), (3, 1), (1, 4), (3, 4)])
def test_fraction_thresholds_scale_by_each_net_entry_debit(source, price_scale, quantity_scale):
    rows = _rows()
    for row in rows:
        if row["right"] == "C" and row["obs_date"] == pd.Timestamp(DATES[2]):
            row.update(bid=3.8, ask=4.2)
    rows = [{**row, "bid": row["bid"] * price_scale, "ask": row["ask"] * price_scale}
            for row in rows]
    position = replace(_position(), legs=tuple(replace(leg, qty=leg.qty * quantity_scale)
                                               for leg in _position().legs))
    result = _fraction_walk(source, rows=rows, position=position)
    scale = price_scale * quantity_scale
    # Day 2 crosses half the 3.5 net debit but not half the 4.6 gross long premium.
    assert result.entry_cost == pytest.approx(3.5 * scale)
    assert result.exit_value == pytest.approx(5.5 * scale)
    assert result.pnl == pytest.approx(2.0 * scale)
    assert 0.5 * result.entry_cost <= result.pnl < 0.5 * 4.6 * scale
    assert result.exit_date == DATES[2]
    assert result.reason == "target"
    assert result.ambiguous_exit is False
    assert result.fill_alpha == result.exit_fill_alpha == 0.25


@pytest.mark.parametrize("mode,bid,ask,reason,pnl,ambiguous,exit_alpha", [
    ("target", 3.0, 3.0, "target", 2.0, False, 0.5),
    ("stop", 1.0, 1.0, "stop", -2.0, False, 0.5),
    ("ambiguous", 0.0, 5.0, "stop", -4.0, True, 0.0),
])
def test_fraction_equality_and_ambiguous_stop(source, mode, bid, ask, reason, pnl,
                                            ambiguous, exit_alpha):
    rows = _rows()
    for row in rows:
        if row["right"] == "C":
            row.update(bid=2.0, ask=2.0)
            if row["obs_date"] == pd.Timestamp(DATES[1]):
                row.update(bid=bid, ask=ask)
    position = replace(_position(), legs=(_position().legs[0],))
    result = _fraction_walk(source, rows=rows, position=position, alpha=0.5)
    assert result.entry_cost == 4.0
    assert result.exit_date == DATES[1]
    assert result.reason == reason
    assert result.pnl == pnl
    assert result.ambiguous_exit is ambiguous
    assert result.exit_fill_alpha == exit_alpha


def test_fraction_stop_waits_for_full_converted_loss(source):
    rows = _rows()
    for row in rows:
        if row["right"] == "C":
            mark = {DATES[0]: 2.0, DATES[1]: 1.25, DATES[2]: 1.0, DATES[3]: 0.5}[
                row["obs_date"].date().isoformat()]
            row.update(bid=mark, ask=mark)
    position = replace(_position(), legs=(_position().legs[0],))
    result = _fraction_walk(source, rows=rows, position=position, alpha=0.5)
    # Two calls cost 4: day one's -1.5 crosses a halved -1 stop, but
    # the declared -0.5 fraction converts to -2 and first crosses on day two.
    assert result.entry_cost == 4.0
    assert result.exit_date == DATES[2]
    assert result.visited_dates == DATES[:3]
    assert result.exit_value == 2.0
    assert result.pnl == -2.0
    assert result.reason == "stop"
    assert result.ambiguous_exit is False
    assert result.fill_alpha == result.exit_fill_alpha == 0.5


@pytest.mark.parametrize("basis", ["credit", "zero"])
def test_fraction_refuses_credit_or_zero_debit_with_entry_context(source, basis):
    position = replace(_position(), legs=(replace(_position().legs[0], side="sell"),))
    rows = _rows()
    if basis == "zero":
        position = replace(_position(), legs=(replace(_position().legs[0], qty=1), _position().legs[1]))
        for row in rows:
            row.update(bid=2.0, ask=2.0)
    with pytest.raises(DataError) as caught:
        _fraction_walk(source, position=position, rows=rows)
    _assert_fraction_refusal(
        caught.value, code="EXPERIMENT_VARIANT_FAILED", category="internal",
        message="entry-debit exit thresholds unavailable",
        details={"trade_id": "trade-a", "session": DATES[0]})


@pytest.mark.parametrize("target,stop,price_scale", [(1e308, -0.5, 1), (0.5, -1e308, 1),
                                                      (1e-300, -0.5, 1e-100),
                                                      (0.5, -1e-300, 1e-100)])
def test_fraction_refuses_conversion_overflow_or_underflow(source, target, stop, price_scale):
    rows = [{**row, "bid": row["bid"] * price_scale, "ask": row["ask"] * price_scale}
            for row in _rows()]
    with pytest.raises(DataError) as caught:
        _fraction_walk(source, recipe=_fraction_recipe(target, stop), rows=rows)
    _assert_fraction_refusal(
        caught.value, code="EXPERIMENT_VARIANT_FAILED", category="internal",
        message="entry-debit exit thresholds unavailable",
        details={"trade_id": "trade-a", "session": DATES[0]})


@pytest.mark.parametrize("units", [None, "", "percent", "entry_credit_fraction", [], {}, True])
def test_invalid_threshold_units_refuse_before_repository_access(units):
    recipe = {**_fraction_recipe(), "threshold_units": units}
    with pytest.raises(OpsError) as caught:
        resolve_experiment_plan(_exit_spec(recipe))
    _assert_fraction_refusal(
        caught.value, code="INVALID_EXPERIMENT_SPEC", category="validation",
        message="invalid fixed-day or target/stop exit economics or mark source", details={})
    with pytest.raises(DataError) as caught:
        walk_exit(None, None, (), economic_params={"exit": recipe, "fill": 0.25})
    _assert_fraction_refusal(
        caught.value, code="EXPERIMENT_VARIANT_FAILED", category="internal",
        message="invalid exit economics", details={})


def test_units_are_identity_bound_and_absolute_behavior_is_unchanged(source):
    conn, clock, store = source
    snapshot = _snapshot(conn, clock, store, _rows())
    repository = Repository(conn, store)
    recipe = {"kind": "target_stop", "trading_days": 3, "target_pnl": 2, "stop_pnl": -1}
    plans = [resolve_experiment_plan(_exit_spec(value)) for value in (
        recipe, {**recipe, "threshold_units": "quote_price_times_quantity"},
        {**recipe, "threshold_units": "entry_debit_fraction"})]
    results = [walk_exit(repository, snapshot, (_position(),), economic_params=p.economic_params)
               for p in plans]
    assert results[0] == results[1]
    assert results[0][0].reason == "target"
    assert results[2][0].reason == "fixed_day"  # 2 is a ratio, never normalized to 2%.
    assert len({plan.json_bytes() for plan in plans}) == 3
    assert "threshold_units" not in plans[0].economic_params["exit"]
    with pytest.raises(TypeError):
        plans[2].economic_params["exit"]["threshold_units"] = "percent"


def test_fraction_missing_mark_still_refuses(source):
    rows = [row for row in _rows() if row["obs_date"] != pd.Timestamp(DATES[1])]
    with pytest.raises(DataError) as caught:
        _fraction_walk(source, rows=rows)
    _assert_fraction_refusal(
        caught.value, code="EXPERIMENT_VARIANT_FAILED", category="internal",
        message="required daily position mark unavailable",
        details={"trade_id": "trade-a", "session": DATES[1]})


@pytest.mark.parametrize("alpha,cost,day,pnl", [(0.0, 3.8, 3, 3.3), (0.5, 3.2, 2, 3.0),
                                                (1.0, 2.6, 1, 2.7)])
def test_fraction_basis_uses_declared_entry_fill(source, alpha, cost, day, pnl):
    result = _fraction_walk(source, alpha=alpha)
    assert result.entry_cost == pytest.approx(cost)
    assert result.exit_date == DATES[day]
    assert result.reason == "target"
    assert result.pnl == pytest.approx(pnl)
    assert result.fill_alpha == result.exit_fill_alpha == alpha


@pytest.mark.parametrize("basis,cost,pnl,reason", [("credit", -4.2, -2.4, "stop"),
                                                  ("zero", 0.0, 0.0, "fixed_day")])
def test_absolute_units_still_accept_credit_and_zero_debit(source, basis, cost, pnl, reason):
    position = replace(_position(), legs=(replace(_position().legs[0], side="sell"),))
    rows = _rows()
    if basis == "zero":
        position = replace(_position(), legs=(replace(_position().legs[0], qty=1), _position().legs[1]))
        for row in rows:
            row.update(bid=2.0, ask=2.0)
    conn, clock, store = source
    snapshot = _snapshot(conn, clock, store, rows)
    recipe = {"kind": "target_stop", "trading_days": 3, "target_pnl": 2, "stop_pnl": -1}
    results = [walk_exit(Repository(conn, store), snapshot, (position,),
                         economic_params=resolve_experiment_plan(_exit_spec(value)).economic_params)
               for value in (recipe, {**recipe, "threshold_units": "quote_price_times_quantity"})]
    assert results[0] == results[1]
    result, = results[0]
    assert result.entry_cost == pytest.approx(cost)
    assert result.pnl == pytest.approx(pnl)
    assert result.reason == reason
