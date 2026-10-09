"""``score_outcomes`` shares ONE chain index across every selection group.

The legacy settlement scorer replays each recorded ``(strategy, selection)``
rule. Before #503 every group built its own ``ChainIndex`` from the Tier-2
chain table, so a pass over N groups read each needed year partition N times.
This drives the REAL replay path over two stored groups while wrapping only
the storage reader to count option-chain partition reads: the available-key
scan and the index load must each touch 2024 exactly once, and every group's
outcome must equal an independent per-group ``replay(..., index=None)``.

``replay``/``load_chain_index``/``_group_events`` are never patched here — the
point is that the shared load is the production one. Two tests force the
shared loads to fail — the chain index and the trading calendar — and prove
the fallback keeps each group's own error in full projected unresolvable
outcome rows.
"""
from __future__ import annotations

import sys
from dataclasses import asdict

import pandas as pd
import pytest

from engine import ledger
from engine.jsonio import json_safe
from engine.ledger_settlement import POLICY, comparison
from engine.structures import STRUCTURES

OBS_ENTRY = "2024-05-02"
OBS_EXIT = "2024-05-03"
EVENT_ID = "TEST_2024-05-02"
RESOLVED_AT = pd.Timestamp("2024-05-04T00:00:00+00:00")


@pytest.fixture
def root(tmp_root, monkeypatch):
    from engine import paths

    monkeypatch.setattr(paths, "LEDGER", tmp_root / "ledger")
    monkeypatch.setattr(ledger, "_settlement_calendar", lambda: pd.DataFrame(
        [{"event_id": EVENT_ID, "ticker": "TEST",
          "event_date": OBS_ENTRY, "session": "AMC"}],
        columns=["event_id", "ticker", "event_date", "session"]))
    return tmp_root


@pytest.fixture
def calendar(monkeypatch):
    from engine import replay as replay_mod
    from engine.calendar import TradingCalendar

    cal = TradingCalendar(pd.bdate_range("2024-03-01", "2024-06-01"))
    monkeypatch.setattr(replay_mod, "trading_calendar", lambda *a, **k: cal)
    return cal


@pytest.fixture(autouse=True)
def _reset_available_keys():
    from engine import replay as replay_mod

    replay_mod._AVAILABLE_KEYS = None
    yield
    replay_mod._AVAILABLE_KEYS = None


@pytest.fixture
def reads(root, monkeypatch):
    """Record every ``(table, years)`` an option-chain read is asked for."""
    from engine.data import store as store_mod

    calls = []
    real = store_mod.iter_table

    def counting(name, **kwargs):
        years = kwargs.get("years")
        calls.append((name, None if years is None else tuple(sorted(years))))
        return real(name, **kwargs)

    monkeypatch.setattr(store_mod, "iter_table", counting)
    return calls


def _write_chains():
    from engine.data import store
    from tests.test_replay import chain

    frames = []
    for obs in (OBS_ENTRY, OBS_EXIT):
        frame = chain("TEST", obs)
        frame["year"] = 2024
        frames.append(frame)
    store.write_table(pd.concat(frames, ignore_index=True), "option_chains")


def _prediction(alpha, *, as_of="2024-05-01", strategy="STR-THRU"):
    settlement = {"policy": POLICY, "spec_version": 1,
                  "structure_spec": asdict(STRUCTURES[strategy]())}
    row_id = ledger.row_id(as_of, "TEST", strategy, None, OBS_EXIT)
    row_id += "|" + ledger.selection_key(settlement, alpha)
    return {
        "schema_version": ledger.SCHEMA_VERSION,
        "row_id": row_id,
        "written_at": "2024-05-01T21:05:03+00:00",
        "as_of": as_of,
        "decision_ts": f"{as_of}T20:00:00+00:00",
        "ticker": "TEST",
        "event_id": EVENT_ID,
        "event_date": OBS_ENTRY,
        "session": "AMC",
        "strategy": strategy,
        "settlement": settlement,
        "structure": {"strike": None, "expiry": OBS_EXIT},
        "intended_prices": {"alpha": alpha, "entry_cost": 2.0,
                            "quote_date": as_of},
        "score": {"win_model": 0.55, "exp_pnl_model": 0.04, "gate_pass": True},
        "model_versions": {"gate": "gate_midfill_str_thru@1"},
        "snapshot_hash": "dce985",
        "audit_receipt": None,
        "supersedes": None,
        "supersede_reason": None,
    }


def _events():
    return pd.DataFrame([{
        "event_id": EVENT_ID, "ticker": "TEST",
        "event_date": pd.Timestamp(OBS_ENTRY), "session": "AMC"}])


def _baseline(alpha, variant=None):
    """The per-group path: ``replay`` plans and loads its own chain index."""
    from engine import replay as replay_mod

    return replay_mod.replay(
        "STR-THRU", _events(), structure=STRUCTURES["STR-THRU"](),
        alphas=[alpha], include_legs=True, variant=variant)


def _shared_index_replay(alpha, variant=None):
    """The scorer's shared-index path, assembled from production helpers."""
    from engine import replay as replay_mod

    structure = STRUCTURES["STR-THRU"]()
    calendar = replay_mod.trading_calendar()
    plan = replay_mod.plan_events(structure, _events(), calendar=calendar)
    available = replay_mod.available_chain_keys()
    plan = replay_mod.filter_plan_by_availability(plan, available)
    index = replay_mod.load_chain_index(plan.chain_keys)
    return replay_mod.replay(
        "STR-THRU", _events(), structure=structure, alphas=[alpha],
        include_legs=True, calendar=calendar, index=index, available=available,
        variant=variant)


def _trades_json(frame):
    return frame.to_json(orient="records", date_format="iso")


def _assert_same_result(got, expected):
    """Every deterministic ReplayResult field; ``elapsed_s`` is telemetry."""
    assert got.strategy == expected.strategy
    assert got.variant == expected.variant
    assert got.planned == expected.planned
    assert got.replayable == expected.replayable
    assert got.skipped == expected.skipped
    assert _trades_json(got.trades) == _trades_json(expected.trades)


def _project_outcome(row, trade, resolved_at):
    scored = row.get("score") or {}
    return json_safe({
        "schema_version": ledger.SCHEMA_VERSION,
        "row_id": row["row_id"],
        "resolved_at": resolved_at.isoformat(),
        "ticker": row["ticker"],
        "strategy": row["strategy"],
        "event_date": row["event_date"],
        "settlement": row.get("settlement"),
        "settlement_source": "orats_quote_simulation",
        "exit_finality": None,
        "predicted_win": scored.get("win_model"),
        "predicted_pnl": scored.get("exp_pnl_model"),
        "predicted_win_analog": scored.get("win_analog"),
        "gate_pass": scored.get("gate_pass"),
        "event_date_changed": False,
        "session_changed": False,
        "calendar_status": "matched",
        "calendar_checked_at": resolved_at.isoformat(),
        "canonical_event_date": OBS_ENTRY,
        "canonical_session": "AMC",
        **comparison(row, trade),
        "status": "resolved",
        "reason": None,
        "fill_alpha_used": float(trade["fill_alpha"]),
        "realized_pnl": float(trade["ret"]),
        "realized_win": bool(float(trade["ret"]) > 0),
        "realized_entry_cost": float(trade["entry_cost"]),
        "realized_exit_value": float(trade["exit_value"]),
        "exit_source": trade.get("exit_mode") or "chain",
    })


def _project_unresolvable(row, reason, resolved_at):
    """The complete outcome row the scorer writes when replay cannot price."""
    scored = row.get("score") or {}
    return json_safe({
        "schema_version": ledger.SCHEMA_VERSION,
        "row_id": row["row_id"],
        "resolved_at": resolved_at.isoformat(),
        "ticker": row["ticker"],
        "strategy": row["strategy"],
        "event_date": row["event_date"],
        "settlement": row.get("settlement"),
        "settlement_source": "orats_quote_simulation",
        "exit_finality": None,
        "predicted_win": scored.get("win_model"),
        "predicted_pnl": scored.get("exp_pnl_model"),
        "predicted_win_analog": scored.get("win_analog"),
        "gate_pass": scored.get("gate_pass"),
        "event_date_changed": False,
        "session_changed": False,
        "calendar_status": "matched",
        "calendar_checked_at": resolved_at.isoformat(),
        "canonical_event_date": OBS_ENTRY,
        "canonical_session": "AMC",
        "status": "unresolvable",
        "reason": reason,
        "realized_pnl": None,
        "realized_win": None,
    })


def test_shared_index_reads_each_partition_once(root, calendar, reads):
    _write_chains()
    predictions = [_prediction(0.5), _prediction(0.25)]
    ledger.write_predictions(predictions)

    # A second recorded row in the SAME selection group, with an earlier
    # as_of date, names the same event and the same recorded rule. The group must
    # collapse that event to one before replay (or the trade lookup is
    # ambiguous) and both rows must settle to the same values.
    shared_event = _prediction(0.5, as_of="2024-04-30")
    ledger.write_predictions([shared_event])
    predictions.append(shared_event)

    # Independent per-group baselines, and a shared-index replay assembled from
    # the production plan/load helpers, must agree on every deterministic field.
    baselines = {row["row_id"]: _baseline(
        row["intended_prices"]["alpha"], row["settlement"].get("variant"))
        for row in predictions}
    for row in predictions:
        _assert_same_result(
            _shared_index_replay(row["intended_prices"]["alpha"],
                                 row["settlement"].get("variant")),
            baselines[row["row_id"]])

    from engine import replay as replay_mod

    replay_mod._AVAILABLE_KEYS = None
    reads.clear()

    result = ledger.score_outcomes(through=OBS_EXIT, resolved_at=RESOLVED_AT)
    assert result["resolved"] == 3, result
    assert result["unresolvable"] == 0, result

    chain_reads = [years for name, years in reads if name == "option_chains"]
    assert chain_reads.count(None) == 1, chain_reads
    assert chain_reads.count((2024,)) == 1, chain_reads

    outcomes = {o["row_id"]: o for o in ledger.read_outcomes()}
    assert set(outcomes) == {row["row_id"] for row in predictions}

    file_rows = {row["row_id"]: row for row in ledger.read_predictions()}
    for row in predictions:
        rid = row["row_id"]
        baseline = baselines[rid]
        assert len(baseline.trades) == 1
        trade = baseline.trades.iloc[0].to_dict()
        assert outcomes[rid] == _project_outcome(file_rows[rid], trade, RESOLVED_AT)


def test_shared_load_failure_falls_back_per_group(root, calendar, monkeypatch):
    _write_chains()
    predictions = [_prediction(0.5), _prediction(0.25)]
    ledger.write_predictions(predictions)

    from engine.data import store as store_mod

    real = store_mod.iter_table

    def unavailable(name, **kwargs):
        if name == "option_chains":
            raise ValueError("option chain store unavailable")
        return real(name, **kwargs)

    monkeypatch.setattr(store_mod, "iter_table", unavailable)

    result = ledger.score_outcomes(through=OBS_EXIT, resolved_at=RESOLVED_AT)
    assert result["resolved"] == 0, result
    assert result["unresolvable"] == 2, result

    expected = {
        row["row_id"]: _project_unresolvable(
            row, "recorded replay unavailable: option chain store unavailable",
            RESOLVED_AT)
        for row in predictions
    }
    outcomes = {o["row_id"]: o for o in ledger.read_outcomes()}
    assert set(outcomes) == set(expected), outcomes
    for rid in sorted(expected):
        assert outcomes[rid] == expected[rid]


def test_shared_calendar_load_failure_marks_all_groups_failed(root, monkeypatch):
    _write_chains()
    predictions = [_prediction(0.5), _prediction(0.25)]
    ledger.write_predictions(predictions)

    from engine import replay as replay_mod

    def unavailable_calendar(*args, **kwargs):
        raise ValueError("calendar unavailable")

    monkeypatch.setattr(replay_mod, "trading_calendar", unavailable_calendar)

    result = ledger.score_outcomes(through=OBS_EXIT, resolved_at=RESOLVED_AT)
    assert result["resolved"] == 0, result
    assert result["unresolvable"] == 2, result

    expected = {
        row["row_id"]: _project_unresolvable(
            row, "recorded replay unavailable: calendar unavailable",
            RESOLVED_AT)
        for row in predictions
    }
    outcomes = {o["row_id"]: o for o in ledger.read_outcomes()}
    assert set(outcomes) == set(expected), outcomes
    for rid in sorted(expected):
        assert outcomes[rid] == expected[rid]


def test_planning_failure_two_siblings_resolve_on_shared_year_read(
        root, calendar, reads, monkeypatch):
    _write_chains()
    ok_half = _prediction(0.5)
    ok_quarter = _prediction(0.25)
    broken = _prediction(0.5, strategy="CAL-P")
    ledger.write_predictions([ok_half, ok_quarter, broken])

    from engine import replay as replay_mod

    real_plan_events = replay_mod.plan_events
    scorer_plans = []

    def failing_plan_events(structure, events, calendar=None):
        if structure.name == "CAL-P":
            raise ValueError("synthetic planning failure")
        # ``replay`` re-plans internally, so count only the scorer's own planning
        # loop: two entries prove both STR-THRU siblings reached the real planner.
        if sys._getframe(1).f_code.co_name == "score_outcomes":
            scorer_plans.append(structure.name)
        return real_plan_events(structure, events, calendar=calendar)

    monkeypatch.setattr(replay_mod, "plan_events", failing_plan_events)

    # Independent per-group baselines: each sibling group's full outcome must
    # equal its own planned-and-loaded replay, not the shared one's.
    baselines = {ok_half["row_id"]: _baseline(0.5),
                 ok_quarter["row_id"]: _baseline(0.25)}

    replay_mod._AVAILABLE_KEYS = None
    reads.clear()

    result = ledger.score_outcomes(through=OBS_EXIT, resolved_at=RESOLVED_AT)
    assert result["resolved"] == 2, result
    assert result["unresolvable"] == 1, result

    # Two sibling groups must each delegate to the real planner; a single group
    # would plan once and still satisfy the one-year-read assertion below.
    assert scorer_plans == ["STR-THRU", "STR-THRU"], scorer_plans

    # Two sibling groups must share ONE year-partition load. A per-group load
    # would read 2024 twice; the shared path reads it exactly once.
    chain_reads = [years for name, years in reads if name == "option_chains"]
    assert chain_reads.count((2024,)) == 1, chain_reads

    outcomes = {o["row_id"]: o for o in ledger.read_outcomes()}
    assert set(outcomes) == {ok_half["row_id"], ok_quarter["row_id"],
                             broken["row_id"]}
    assert outcomes[broken["row_id"]] == _project_unresolvable(
        broken, "recorded replay unavailable: synthetic planning failure",
        RESOLVED_AT)

    file_rows = {row["row_id"]: row for row in ledger.read_predictions()}
    for row in (ok_half, ok_quarter):
        baseline = baselines[row["row_id"]]
        assert len(baseline.trades) == 1
        trade = baseline.trades.iloc[0].to_dict()
        assert outcomes[row["row_id"]] == _project_outcome(
            file_rows[row["row_id"]], trade, RESOLVED_AT)


def test_empty_availability_scans_once_and_loads_no_index(
        root, calendar, reads, monkeypatch):
    """A plan that filters to zero available chains keeps the per-group path.

    The stored chain table holds only unrelated ticker keys, so the real
    availability scan drops the one planned TEST event. The scorer must not
    build a shared ChainIndex (there is nothing to load), must touch the chain
    store exactly once for the availability scan, and must settle the
    prediction unresolvable rather than crash.
    """
    from engine.data import store
    from tests.test_replay import chain

    frame = chain("OTHER", OBS_ENTRY)
    frame["year"] = 2024
    store.write_table(frame, "option_chains")

    prediction = _prediction(0.5)
    ledger.write_predictions([prediction])

    from engine import replay as replay_mod

    real_load = replay_mod.load_chain_index
    load_calls = []

    def counting_load(keys, **kwargs):
        load_calls.append(tuple(keys))
        return real_load(keys, **kwargs)

    monkeypatch.setattr(replay_mod, "load_chain_index", counting_load)

    replay_mod._AVAILABLE_KEYS = None
    reads.clear()

    result = ledger.score_outcomes(through=OBS_EXIT, resolved_at=RESOLVED_AT)
    assert result["resolved"] == 0, result
    assert result["unresolvable"] == 1, result
    assert load_calls == [], load_calls

    chain_reads = [years for name, years in reads if name == "option_chains"]
    assert chain_reads.count(None) == 1, chain_reads
    assert not [years for years in chain_reads if years is not None], chain_reads

    outcomes = {o["row_id"]: o for o in ledger.read_outcomes()}
    assert outcomes[prediction["row_id"]] == _project_unresolvable(
        prediction,
        "no priced replay for the recorded rule; entry/exit evidence unavailable",
        RESOLVED_AT)
