from types import SimpleNamespace

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
    source = SimpleNamespace(binding_id="selected-gate-fold")
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
    assert captured[0].gate_forecast_source is source


def test_missing_causal_fold_refuses_row_model_not_ready(tmp_path):
    binding = _stage_release(tmp_path, include_size_fold=False)
    event = _event_inputs()

    assembled, refusals = _assemble(binding, [event], as_of="2026-01-10")

    assert assembled == {}
    assert [refusal.code for refusal in refusals] == ["MODEL_NOT_READY"]
    assert refusals[0].detail == "causal size fold unavailable for gate forecast"


def test_gate_source_cache_reuses_only_same_causal_month(monkeypatch):
    binding = object()
    key = _event_inputs().key
    source = SimpleNamespace(binding_id="selected-gate-fold")
    calls = []
    cache = {}

    def selector(selected_binding, event_date, as_of):
        calls.append((selected_binding, event_date, as_of))
        return source

    monkeypatch.setattr(native_score_batch, "select_gate_forecast_source", selector)

    first = native_score_batch._gate_source_or_refusal(
        key, binding, "2025-08-02", "2025-12-10", cache)
    same_month = native_score_batch._gate_source_or_refusal(
        key, binding, "2025-08-31", "2025-12-10", cache)
    next_month = native_score_batch._gate_source_or_refusal(
        key, binding, "2025-09-01", "2025-12-10", cache)

    assert first is same_month is next_month is source
    assert calls == [
        (binding, "2025-08-02", "2025-12-10"),
        (binding, "2025-09-01", "2025-12-10"),
    ]
