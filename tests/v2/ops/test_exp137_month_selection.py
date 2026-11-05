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
        def get(self, _ticker, _entry_date):
            return pd.DataFrame({"available": [True]})

    class Selector:
        def __init__(self, kind):
            assert kind == "first_post_event"

        def select(self, _entry, _event_date, session):
            return {"OCT": "2026-10-30", "NOV": "2026-11-20"}[session]

    replay = SimpleNamespace(load_chain_index=lambda _keys, progress_every: Index())
    e133 = SimpleNamespace(ExpirySelector=Selector)
    result = builder._effective_month_plan(_availability_plan(), "2026-10", replay, e133)

    assert result["event_id"].tolist() == ["current", "prior-expiry-oct"]


def test_prior_month_entry_chain_failure_is_typed_refusal():
    frame = _availability_plan().iloc[[1]].copy()
    replay = SimpleNamespace(load_chain_index=lambda _keys, progress_every: SimpleNamespace(
        get=lambda _ticker, _entry_date: None))

    with pytest.raises(TrainingRefused, match="required_prior_month_entry_chain_missing"):
        builder._effective_month_plan(frame, "2026-10", replay, SimpleNamespace())


def test_prior_month_expiry_selection_failure_is_typed_refusal():
    frame = _availability_plan().iloc[[1]].copy()

    class Index:
        def get(self, _ticker, _entry_date):
            return pd.DataFrame({"available": [True]})

    class Selector:
        def __init__(self, kind):
            pass

        def select(self, *_args):
            raise ValueError("expiry chain unavailable")

    replay = SimpleNamespace(load_chain_index=lambda _keys, progress_every: Index())
    with pytest.raises(TrainingRefused, match="prior_month_expiry_selection_failed"):
        builder._effective_month_plan(frame, "2026-10", replay,
                                      SimpleNamespace(ExpirySelector=Selector))


def test_unclassified_monthly_event_skip_is_typed_refusal():
    with pytest.raises(TrainingRefused, match="required_monthly_candidate_input_failed"):
        builder._refuse_monthly_event_skip(ValueError("required chain is unavailable"), "event-1")


def test_monthly_score_matches_independent_maximum_oracle(monkeypatch, tmp_path):
    event = pd.DataFrame([{
        "event_id": "oct-event", "ticker": "XYZ", "event_date": pd.Timestamp("2026-10-01"),
        "session": "AMC",
    }])
    plan = event.assign(entry_date=pd.Timestamp("2026-09-30"),
                        exit_date=pd.Timestamp("2026-10-15"))
    chain = pd.DataFrame({"available": [True]})
    replay = SimpleNamespace(
        plan_events=lambda _structure, _events: SimpleNamespace(frame=plan.copy(), skipped={}),
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
    monkeypatch.setattr(e133, "forecasts", lambda: event[["ticker", "event_date"]].assign(forecast=1.0))
    monkeypatch.setattr(e133, "residual_history", lambda: pd.DataFrame())
    monkeypatch.setattr(e133, "price_event", lambda *_args, **_kwargs: {
        "bid_e": np.zeros((2, 1)), "ask_e": np.ones((2, 1)), "strikes": np.array([100.0]),
        "idx": np.array([0, 1]), "admissible": np.array([True, True]),
        "tradeable": np.array([True, True]), "sim_sel": np.array([0.25, 0.90]),
        "sim_gate": np.array([0.2, 0.8]), "listed": np.array([True, True]),
    })
    monkeypatch.setattr(e133, "EQUIV_EVENTS", 0)
    monkeypatch.setattr(e133, "_emit", lambda *_args: [{
        "exit_date": pd.Timestamp("2026-10-15"),
        "expiry": pd.Timestamp("2026-10-30"), "legs": [], "exit_value": 1.0,
        "oracle_score": _args[6],
    }])
    monkeypatch.setattr(e133, "_pattern_frame", lambda: pd.DataFrame({"pattern": [0, 0]}))
    monkeypatch.setattr(e133, "_equiv_summary", lambda _items: {})
    monkeypatch.setattr(e134, "no_arb_ok", lambda *_args, **_kwargs: np.array([True, True]))
    monkeypatch.setattr(e134, "expiry_spot", lambda: {("XYZ", pd.Timestamp("2026-10-30")): 100.0})
    monkeypatch.setattr(e134, "legs_no_arb_ok", lambda _legs: True)
    monkeypatch.setattr(e134, "terminal_payoff", lambda _legs, _spot: 3.0)

    result = builder.build_all(limit_exit_month="2026-10")

    # Independent oracle: the two eligible simulated values are .25 and .90, so the best is .90.
    assert result["trades"]["oracle_score"].tolist() == [0.90]
