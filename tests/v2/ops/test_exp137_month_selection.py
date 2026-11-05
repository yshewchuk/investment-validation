from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from engine.v2.models.training.fold_store import TrainingRefused

ROOT = Path(__file__).resolve().parents[3]
BUILDER_PATH = ROOT / "experiments" / "EXP-137_one_book_per_family_which_enumerated_str" / "build.py"
SPEC = importlib.util.spec_from_file_location("test_exp137_month_builder", BUILDER_PATH)
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_month_window_selects_only_exit_dates_in_requested_month():
    frame = pd.DataFrame({
        "event_id": ["prior", "first", "last", "next"],
        "exit_date": ["2026-09-30", "2026-10-01", "2026-10-31", "2026-11-01"],
    })

    assert builder._month_window(frame, "2026-10")["event_id"].tolist() == ["first", "last"]


def test_out_of_range_month_is_typed_refusal():
    with pytest.raises(TrainingRefused) as error:
        builder._month_window(pd.DataFrame({"exit_date": []}), "10000-01")
    assert error.value.issues[0].code == "INVALID_EXIT_MONTH"


@pytest.mark.parametrize("month", ["2026-1", "2026-10-01"])
def test_noncanonical_month_is_typed_refusal(month):
    with pytest.raises(TrainingRefused) as error:
        builder._month_window(pd.DataFrame({"exit_date": []}), month)
    assert error.value.issues[0].code == "INVALID_EXIT_MONTH"


def test_effective_exit_repair_updates_value_pnl_and_return_by_no_arb_only():
    trades = pd.DataFrame([
        {"event_id": "held-oct", "exit_date": "2026-09-30", "expiry": "2026-10-31",
         "exit_arb_ok": False, "exit_value": -2.0, "exit_value_expiry": 5.0,
         "entry_cost": 2.0, "pnl": -4.0, "ret": -2.0},
        {"event_id": "quoted-negative", "exit_date": "2026-10-10", "expiry": "2026-11-30",
         "exit_arb_ok": True, "exit_value": -0.5, "exit_value_expiry": 8.0,
         "entry_cost": 1.0, "pnl": -1.5, "ret": -1.5},
        {"event_id": "held-nov", "exit_date": "2026-10-25", "expiry": "2026-11-15",
         "exit_arb_ok": False, "exit_value": -1.0, "exit_value_expiry": 9.0,
         "entry_cost": 1.0, "pnl": -2.0, "ret": -2.0},
        {"event_id": "low-cost", "exit_date": "2026-10-11", "expiry": "2026-11-30",
         "exit_arb_ok": True, "exit_value": 1.0, "exit_value_expiry": 2.0,
         "entry_cost": builder.e133.MIN_MEANINGFUL_COST / 2, "pnl": 0.0, "ret": np.nan},
    ])

    result = builder._effective_exit_trades(trades, "2026-10").set_index("event_id")
    assert set(result.index) == {"held-oct", "quoted-negative", "low-cost"}
    assert result.loc["held-oct", "exit_date"] == pd.Timestamp("2026-10-31")
    assert result.loc["held-oct", ["exit_value", "pnl", "ret"]].tolist() == [5.0, 3.0, 1.5]
    assert result.loc["quoted-negative", ["exit_value", "pnl", "ret"]].tolist() == [
        -0.5, -1.5, -1.5]
    assert pd.isna(result.loc["low-cost", "ret"])


def test_effective_exit_missing_expiry_value_is_typed_refusal():
    trades = pd.DataFrame([{
        "exit_date": "2026-09-30", "expiry": "2026-10-31", "exit_arb_ok": False,
        "exit_value": -1.0, "exit_value_expiry": np.nan, "entry_cost": 1.0,
        "pnl": -2.0, "ret": -2.0,
    }])

    with pytest.raises(TrainingRefused) as error:
        builder._effective_exit_trades(trades, "2026-10")
    assert error.value.issues[0].code == "MONTHLY_EFFECTIVE_EXIT_DATA_MISSING"


def _availability_plan():
    return pd.DataFrame([
        {"event_id": "current", "ticker": "XYZ", "event_date": "2026-10-01",
         "entry_date": "2026-09-30", "exit_date": "2026-10-15", "session": "AMC"},
        {"event_id": "prior-expiry-oct", "ticker": "XYZ", "event_date": "2026-09-01",
         "entry_date": "2026-08-31", "exit_date": "2026-09-15", "session": "OCT"},
        {"event_id": "prior-expiry-nov", "ticker": "XYZ", "event_date": "2026-09-02",
         "entry_date": "2026-09-01", "exit_date": "2026-09-16", "session": "NOV"},
    ]).assign(event_date=lambda f: pd.to_datetime(f.event_date),
              entry_date=lambda f: pd.to_datetime(f.entry_date),
              exit_date=lambda f: pd.to_datetime(f.exit_date))


def test_repaired_candidates_are_selected_by_expiry_availability_month():
    class Index:
        def get(self, _ticker, entry_date):
            expiry = ("2026-10-30" if entry_date == pd.Timestamp("2026-08-31")
                      else "2026-11-20")
            return pd.DataFrame({
                "expiry": ["2026-09-15", expiry], "bid": [np.nan, 1.0],
                "ask": [np.nan, 2.0],
            })

    class Selector:
        def __init__(self, kind):
            assert kind == "first_post_event"

        def select(self, entry, _event_date, _session):
            assert entry["expiry"].tolist() in (["2026-10-30"], ["2026-11-20"])
            return entry["expiry"].iloc[0]

    replay = SimpleNamespace(load_chain_index=lambda _keys, progress_every: Index())
    e133 = SimpleNamespace(ExpirySelector=Selector)
    result = builder._effective_month_plan(_availability_plan(), "2026-10", replay, e133)

    assert result["event_id"].tolist() == ["current", "prior-expiry-oct"]


def test_prior_month_entry_chain_failure_is_typed_refusal():
    frame = _availability_plan().iloc[[1]].copy()
    replay = SimpleNamespace(load_chain_index=lambda _keys, progress_every: SimpleNamespace(
        get=lambda _ticker, _entry_date: None))

    with pytest.raises(TrainingRefused, match="required_prior_month_entry_chain_missing") as error:
        builder._effective_month_plan(frame, "2026-10", replay, SimpleNamespace())
    assert error.value.issues[0].code == "PRIOR_MONTH_ENTRY_CHAIN_MISSING"


def test_prior_month_expiry_selection_failure_is_typed_refusal():
    frame = _availability_plan().iloc[[1]].copy()

    class Index:
        def get(self, _ticker, _entry_date):
            return pd.DataFrame({"bid": [1.0], "ask": [2.0]})

    class Selector:
        def __init__(self, kind):
            pass

        def select(self, *_args):
            raise ValueError("expiry chain unavailable")

    replay = SimpleNamespace(load_chain_index=lambda _keys, progress_every: Index())
    with pytest.raises(TrainingRefused, match="prior_month_expiry_selection_failed") as error:
        builder._effective_month_plan(frame, "2026-10", replay,
                                      SimpleNamespace(ExpirySelector=Selector))
    assert error.value.issues[0].code == "PRIOR_MONTH_EXPIRY_SELECTION_FAILED"


def test_unclassified_monthly_event_skip_is_typed_refusal():
    with pytest.raises(TrainingRefused, match="required_monthly_candidate_input_failed") as error:
        builder._refuse_monthly_event_skip(ValueError("required chain is unavailable"), "event-1")
    assert error.value.issues[0].code == "MONTHLY_CANDIDATE_INPUT_FAILED"


@pytest.mark.parametrize("reason", [
    "structure_unresolved", "exit_structure_unresolved", "expiry_gone_at_exit", "bad_quote",
    "zero_cost",
])
def test_monthly_full_build_skip_reasons_are_allowed(reason):
    builder._refuse_monthly_event_skip(ValueError(reason), "event-1")


def test_monthly_build_filters_coverage_and_slices_output_to_scored_month(monkeypatch, tmp_path):
    event_dates = [
        ("oct-event", "2026-10-01"), ("no-candidate", "2026-10-02"),
        ("missing-forecast", "2026-10-03"), ("unavailable", "2026-10-04"),
        ("prior-missing-entry", "2026-09-05"),
    ]
    event = pd.DataFrame([
        {"event_id": event_id, "ticker": "XYZ", "event_date": pd.Timestamp(event_date),
         "session": "AMC"}
        for event_id, event_date in event_dates
    ])
    plan = event.assign(entry_date=event["event_date"] - pd.Timedelta(days=1))
    plan["exit_date"] = pd.to_datetime([
        "2026-09-15" if event_id == "prior-missing-entry" else "2026-10-15"
        for event_id in event["event_id"]
    ])
    chain = pd.DataFrame({"available": [True]})
    filter_calls = []

    def filter_available(replay_plan):
        filter_calls.append(True)
        frame = replay_plan.frame.loc[
            ~replay_plan.frame["event_id"].isin({"unavailable", "prior-missing-entry"})].copy()
        return SimpleNamespace(frame=frame, skipped={"no_entry_chain": 1})

    original_month_selection = builder._effective_month_plan

    def select_after_coverage(frame, month, replay_module, experiment):
        assert "prior-missing-entry" not in set(frame["event_id"])
        return original_month_selection(frame, month, replay_module, experiment)

    monkeypatch.setattr(builder, "_effective_month_plan", select_after_coverage)
    replay = SimpleNamespace(
        plan_events=lambda _structure, _events: SimpleNamespace(frame=plan.copy(), skipped={}),
        filter_plan_by_availability=filter_available,
        load_chain_index=lambda _keys, progress_every: SimpleNamespace(
            get=lambda _ticker, _date: chain),
        ReplayPlan=lambda frame, skipped: SimpleNamespace(frame=frame, skipped=skipped),
    )
    monkeypatch.setattr(builder, "replay_mod", replay)
    monkeypatch.setattr(builder, "RESULTS", tmp_path)
    monkeypatch.setattr(builder, "FAMILIES", ("family",))
    monkeypatch.setattr(builder, "FAMILY_MASK", {"family": np.array([True, True])})

    class Grid(SimpleNamespace):
        def __len__(self):
            return 2

    monkeypatch.setattr(builder, "GRID", Grid(
        family_key=np.array(["family", "family"]), slot_used=np.array([0, 0]),
        qty=np.array([1, 1]), key=np.array(["a", "b"])))
    monkeypatch.setattr(builder, "pnl_sim", SimpleNamespace(ResidualPool=lambda _history: object()))

    e133 = builder.e133
    e134 = builder.e134
    monkeypatch.setattr(e133, "event_universe", lambda: event.copy())
    monkeypatch.setattr(e133, "forecasts", lambda: event.loc[
        event["event_id"].ne("missing-forecast"), ["ticker", "event_date"]].assign(forecast=1.0))
    monkeypatch.setattr(e133, "residual_history", lambda: pd.DataFrame())
    monkeypatch.setattr(e133, "price_event", lambda *_args, **_kwargs: {
        "bid_e": np.zeros((2, 1)), "ask_e": np.ones((2, 1)), "strikes": np.array([100.0]),
        "idx": np.array([0, 1]), "admissible": np.array([True, True]),
        "tradeable": np.array([True, True]), "sim_sel": np.array([0.90, 0.25]),
        "sim_gate": np.array([0.20, 0.80]), "listed": np.array([True, True]),
    })
    monkeypatch.setattr(e133, "EQUIV_EVENTS", 0)

    def emit(*args):
        if args[1]["event_id"] == "no-candidate":
            return []
        return [{
            "event_id": args[1]["event_id"], "exit_date": args[1]["exit_date"],
            "expiry": pd.Timestamp("2026-10-30"), "legs": [], "exit_value": 1.0,
            "entry_cost": 0.5, "exp_pnl_sim_select": args[6],
        }]

    monkeypatch.setattr(e133, "_emit", emit)
    monkeypatch.setattr(e133, "_pattern_frame", lambda: pd.DataFrame({"pattern": [0, 0]}))
    monkeypatch.setattr(e133, "_equiv_summary", lambda _items: {})
    monkeypatch.setattr(e134, "no_arb_ok", lambda *_args, **_kwargs: np.array([True, False]))
    monkeypatch.setattr(e134, "expiry_spot", lambda: {("XYZ", pd.Timestamp("2026-10-30")): 100.0})
    monkeypatch.setattr(e134, "legs_no_arb_ok", lambda _legs: False)
    monkeypatch.setattr(e134, "terminal_payoff", lambda _legs, _spot: 3.0)

    result = builder.build_all(limit_exit_month="2026-10")

    assert filter_calls == [True]
    assert set(result) == {"trades"}
    trades = result["trades"]
    assert trades["event_id"].tolist() == ["oct-event"]
    assert trades["exp_pnl_sim_select"].tolist() == [0.90]
    assert trades["exit_date"].tolist() == [pd.Timestamp("2026-10-30")]
    assert trades["exit_date"].dt.to_period("M").eq("2026-10").all()
    assert trades["exit_value"].tolist() == [3.0]
    assert trades["pnl"].tolist() == [2.5]
    assert trades["ret"].tolist() == [5.0]
    assert list(tmp_path.iterdir()) == []

    monkeypatch.setattr(e134, "legs_no_arb_ok", lambda _legs: True)
    monkeypatch.setattr(e134, "expiry_spot", lambda: {})
    quoted_result = builder.build_all(limit_exit_month="2026-10")
    assert quoted_result["trades"]["exit_date"].tolist() == [pd.Timestamp("2026-10-15")]
    assert quoted_result["trades"]["exit_value_expiry"].isna().all()
    assert list(tmp_path.iterdir()) == []

    def fail_residual_load(_history):
        raise OSError("residual source unavailable")

    monkeypatch.setattr(builder.pnl_sim, "ResidualPool", fail_residual_load)
    with pytest.raises(TrainingRefused) as error:
        builder.build_all(limit_exit_month="2026-10")
    assert error.value.issues[0].code == "MONTHLY_SOURCE_LOAD_FAILED"
    assert list(tmp_path.iterdir()) == []
