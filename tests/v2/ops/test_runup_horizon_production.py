"""Synthetic real calendar/quote scans through producer and frozen consumers."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime

import pandas as pd
import pytest

from engine.v2.features.panel_row_inputs import PanelRowInputs
from engine.v2.foundation.market_calendar import CalendarInputError
from engine.v2.ops import nightly_raw_row_producer as producer
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import _event_inputs_from_document
from engine.v2.scoring.nightly_source_bundle import assemble_nightly_source_bundle
from engine.v2.scoring.source_inputs import build_native_score_inputs
from tests.v2.ops.test_native_score_batch_driver_alias import _assemble, _stage
from tests.v2.ops.test_v2_ops_nightly_calendar_inputs import (
    _CALENDAR,
    _common_pair,
    _event_row,
    _price_row,
    _snapshot,
)


def _produce(tmp_path, monkeypatch, *, decision="2024-01-05", session="BMO", calendar=_CALENDAR):
    tmp_path.mkdir(parents=True, exist_ok=True)
    keys = tuple(BoardRequest("AAA", strategy, pd.Timestamp("2024-01-16"), session)
                 for strategy in ("STR-RUNUP", "STR-THRU"))
    repository, snapshot = _snapshot(
        tmp_path, price={"AAA": [_price_row("AAA", decision, close_raw=100.0)]},
        chains=_common_pair("AAA", datetime.fromisoformat(decision), datetime(2024, 2, 16), 100.0),
        events=[_event_row("AAA", "2024-01-16", session)])
    panel = PanelRowInputs(panel_row={"date": "2024-01-16", "spy_ret252": 0.0,
                                     "days_before_print": 99}, panel_anchor=decision)
    before = deepcopy(panel.panel_row)
    # Isolate unrelated universe/panel-history work, keeping real pinned source
    # reads, expiry/exit policy, timing, producer refusal handling and encoding.
    monkeypatch.setattr(producer, "scan_forward_board_requests", lambda *a, **k: keys)
    monkeypatch.setattr(producer, "scan_decision_calendar", lambda *a, **k: calendar)
    monkeypatch.setattr(producer, "_panel_history_start", lambda *a: decision)
    monkeypatch.setattr(producer, "_shared_panel_rows", lambda *a, **k: {
        ("AAA", "2024-01-16", session): panel})
    result = producer.build_native_score_batch_events(
        repository, snapshot, as_of=decision, horizon_days=14)
    assert panel.panel_row == before
    return result


@pytest.mark.parametrize("decision,session,expected", [
    ("2024-01-05", "BMO", 5), ("2024-01-05", "AMC", 6),
    ("2024-01-12", "BMO", 0), ("2024-01-16", "AMC", 0),
])
def test_producer_horizon_reaches_both_real_frozen_consumers(tmp_path, monkeypatch, decision, session, expected):
    events, refusals = _produce(tmp_path / "sources", monkeypatch, decision=decision, session=session)
    assert refusals["refusals"] == []
    assert len(events) == 2
    assert events[0]["calendar_row"]["days_before_print"] == expected
    assert "days_before_print" not in events[1]["calendar_row"]
    binding = _stage(tmp_path / "release", [
        ("implied_t1", "*", ("days_before_print",)),
        ("runup_move", "*", ("days_before_print",)),
        ("gate", "STR-RUNUP", ("days_before_print",)),
    ])
    event = _event_inputs_from_document(events[0])
    recipes = {"driver_prediction": {"binding_id": "b-implied_t1-*"},
               "runup_move_prediction": {"binding_id": "b-runup_move-*"}}
    bundle = assemble_nightly_source_bundle(
        source_ref="synthetic:runup-horizon", strategy="STR-RUNUP", as_of=decision,
        calendar_row=event.calendar_row,
        panel_row={**event.panel_row, "days_to_print": 99, "dte_entry": 99},
        panel_anchor=event.panel_anchor, tier4_row=event.tier4_row,
        quote_rows=event.quote_rows, quote_status=event.quote_status,
        feature_names=("days_before_print", "days_to_print", "dte_entry"), driver_name="im_t1",
        forecast_recipes=recipes,
        model_artifact_refs={output: "synthetic-ref" for output in recipes},
        gate_recipe={"binding_id": "b-gate-STR-RUNUP", "threshold": 0.0})
    assert bundle.feature_vector == {"days_before_print": expected}
    assert {quote["observed_date"] for quote in bundle.raw_quotes.values()} == {decision}
    inputs = build_native_score_inputs(replace(
        bundle, model_release=binding.model_release, frozen_inference=binding.frozen_inference))
    assert inputs.context["days_before_print"] == expected
    assert inputs.context["as_of"] == decision
    assert inputs.features["missing_mask"] == {"days_before_print": False}
    assert set(inputs.forecast["executors"]) == {"driver_prediction", "runup_move_prediction"}
    for output, executor in inputs.forecast["executors"].items():
        assert executor.predict(inputs.features["model_inputs"]) == {output: expected + 1.0}
    # Timing alone does not supply the separately required release gate policy.
    assert _assemble(binding, [event])[1][0].code == "GATE_POLICY_NOT_STAGED"


@pytest.mark.parametrize("decision,calendar,code,detail", [
    ("2024-01-16", _CALENDAR, "INVALID_RUNUP_HORIZON", "RUNUP entry is after the pre-print close"),
    ("2024-01-05", replace(_CALENDAR, days=_CALENDAR.days[3:], observed_through="2024-01-08"),
     "RUNUP_HORIZON_NOT_AVAILABLE", "the pinned calendar does not cover the RUNUP horizon"),
])
def test_typed_runup_refusal_does_not_drop_other_strategy(tmp_path, monkeypatch, decision, calendar, code, detail):
    events, refusals = _produce(tmp_path, monkeypatch, decision=decision, calendar=calendar)
    assert [event["key"]["strategy"] for event in events] == ["STR-THRU"]
    assert len(refusals["refusals"]) == 1
    refusal = refusals["refusals"][0]
    assert refusal["key"] == {"ticker": "AAA", "strategy": "STR-RUNUP", "event_date": "2024-01-16", "session": "BMO"}
    assert (refusal["code"], refusal["detail"]) == (code, detail)


def test_malformed_calendar_still_aborts_build(tmp_path, monkeypatch):
    with pytest.raises(CalendarInputError) as error:
        _produce(tmp_path, monkeypatch, calendar=replace(_CALENDAR, observed_through="bad"))
    assert type(error.value) is CalendarInputError
