from types import SimpleNamespace

import pandas as pd

from engine.v2.ops import native_score_batch
from tests.v2.ops.test_native_score_batch import (
    _assemble,
    _event_inputs,
    _stage_release,
)


def test_str_thru_selector_receives_binding_event_date_and_as_of(
        tmp_path, monkeypatch):
    binding = _stage_release(tmp_path)
    event = _event_inputs()
    as_of = "2026-01-10"
    source = SimpleNamespace(binding_id="selected-gate-fold", artifact_hash="fold-hash")
    calls = []
    captured = []

    def selector(selected_binding, event_date, received_as_of):
        calls.append((selected_binding, event_date, received_as_of))
        return source

    monkeypatch.setattr(native_score_batch, "select_gate_forecast_source", selector)
    monkeypatch.setattr(
        native_score_batch,
        "build_native_score_inputs",
        lambda bundle: captured.append(bundle) or "native-inputs",
    )

    assembled, refusals = _assemble(binding, [event], as_of=as_of)

    assert refusals == ()
    assert calls == [(binding, event.key.event_date, as_of)]
    assert assembled[event.key][1] == "native-inputs"
    assert "fold-hash" in assembled[event.key][0].model_artifact_refs
    assert captured[0].gate_forecast_source is source


def test_missing_causal_fold_refuses_row_model_not_ready(tmp_path):
    binding = _stage_release(tmp_path, include_size_fold=False)
    event = _event_inputs()

    assembled, refusals = _assemble(binding, [event], as_of="2026-01-10")

    assert assembled == {}
    assert [refusal.code for refusal in refusals] == ["MODEL_NOT_READY"]
    assert refusals[0].detail == "causal size fold unavailable for gate forecast"


def test_gate_source_cache_reuses_only_same_causal_month(tmp_path, monkeypatch):
    binding = _stage_release(tmp_path)
    as_of = "2026-01-10"
    source = SimpleNamespace(binding_id="january-fold", artifact_hash="hash-january")
    calls = []
    captured = []

    def event_inputs(ticker, event_id, day):
        event_date = pd.Timestamp(day)
        following = (event_date + pd.Timedelta(days=1)).date().isoformat()
        return _event_inputs(
            ticker=ticker, event_date=event_date,
            calendar_row={
                "event_id": event_id, "ticker": ticker, "event_date": day,
                "entry_date": day, "exit_date": following, "expiry": following,
                "spot": 100.0, "calendar_observed_through": "2026-01-09",
            },
            panel_anchor="2026-01-09",
        )

    events = [
        event_inputs("JAN1", "evt-jan1", "2026-01-20"),
        event_inputs("JAN2", "evt-jan2", "2026-01-31"),
        event_inputs("FEB1", "evt-feb1", "2026-02-01"),
    ]

    def selector(selected_binding, event_date, received_as_of):
        calls.append((selected_binding, event_date, received_as_of))
        return source

    monkeypatch.setattr(native_score_batch, "select_gate_forecast_source", selector)
    monkeypatch.setattr(
        native_score_batch, "build_native_score_inputs",
        lambda bundle: captured.append(bundle) or "native-inputs",
    )

    assembled, refusals = _assemble(binding, events, as_of=as_of)

    assert refusals == ()
    assert len(assembled) == 3
    assert calls == [(binding, pd.Timestamp("2026-01-20"), as_of)]
    assert [bundle.gate_forecast_source for bundle in captured] == [source, source, source]
    assert all(
        "hash-january" in request.model_artifact_refs
        for request, _ in assembled.values()
    )
