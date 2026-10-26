import json
from types import SimpleNamespace

import pandas as pd

from engine.v2.ops import native_score_batch
from tests.v2.ops.test_native_score_batch import (
    _assemble,
    _event_doc,
    _event_inputs,
    _stage_release,
    _worker_parameters,
)


def test_str_thru_selector_receives_binding_event_date_and_as_of(
        tmp_path, monkeypatch):
    binding = _stage_release(tmp_path)
    event = _event_inputs()
    as_of = pd.Timestamp("2026-01-10").to_datetime64()
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
    assert calls == [(binding, event.key.event_date, pd.Timestamp(as_of))]
    assert assembled[event.key][1] == "native-inputs"
    assert "fold-hash" in assembled[event.key][0].model_artifact_refs
    assert captured[0].gate_forecast_source is source


def test_missing_event_id_refuses_before_missing_fold(tmp_path):
    binding = _stage_release(tmp_path, include_size_fold=False)
    calendar_row = dict(_event_inputs().calendar_row)
    calendar_row.pop("event_id")
    event = _event_inputs(calendar_row=calendar_row)

    assembled, refusals = _assemble(binding, [event])

    assert assembled == {}
    assert [refusal.code for refusal in refusals] == ["MISSING_STAGED_INPUT"]
    assert refusals[0].detail == "calendar_row missing event_id"


def test_real_synthetic_fold_populates_gate_output(tmp_path):
    binding = _stage_release(
        tmp_path, gate_output_name="gate_score", gate_feature_name="pred_abs_move",
        size_feature_names=("x",))
    selected_source = native_score_batch.select_gate_forecast_source(
        binding, pd.Timestamp("2026-01-15"), pd.Timestamp("2026-01-10"))
    expected_descriptor = binding.size_folds.descriptors["2026-01-01"]
    assert selected_source.artifact_hash == expected_descriptor.estimator.content_hash
    assert selected_source.artifact_hash != binding.size_folds.catalog_hash
    staging = tmp_path / "staging"
    staging.mkdir()
    low_doc = _event_doc(event_id="evt-low")
    low_doc["panel_row"]["x"] = 0.0
    high_doc = _event_doc(event_id="evt-high")
    high_doc["key"]["ticker"] = "TEST2"
    high_doc["calendar_row"]["ticker"] = "TEST2"
    high_doc["panel_row"]["x"] = 1.5
    (staging / "events.json").write_text(json.dumps([low_doc, high_doc]))

    parameters = _worker_parameters(tmp_path, expected_ids=["native_score_batch"])
    parameters["feature_names"] = ["x"]
    parameters["gate_policy"] = {"STR-THRU": {"threshold": 1.02}}
    native_score_batch.run_native_score_batch_worker(parameters, staging)
    records = json.loads((staging / "records.json").read_text())["records"]
    assert all(
        selected_source.artifact_hash in record["canonical_request"]["model_artifact_refs"]
        for record in records.values()
    )
    low_terms = records["TEST|STR-THRU|2026-01-15|am"]["gate_terms"]
    high_terms = records["TEST2|STR-THRU|2026-01-15|am"]["gate_terms"]

    assert low_terms["gate_score"] == 1.0
    assert low_terms["gate_threshold"] == 1.02
    assert low_terms["gate_pass"] is False
    assert high_terms["gate_score"] == 1.03
    assert high_terms["gate_threshold"] == 1.02
    assert high_terms["gate_pass"] is True


def test_missing_causal_fold_refuses_row_model_not_ready(tmp_path):
    binding = _stage_release(tmp_path, include_size_fold=False)
    event = _event_inputs()

    assembled, refusals = _assemble(binding, [event], as_of="2026-01-10")

    assert assembled == {}
    assert [refusal.code for refusal in refusals] == ["MODEL_NOT_READY"]
    assert refusals[0].detail == "causal size fold unavailable for gate forecast"


def test_gate_source_cache_reuses_only_same_causal_month(tmp_path, monkeypatch):
    binding = _stage_release(tmp_path)
    as_of = pd.Timestamp("2026-03-10")
    january = SimpleNamespace(binding_id="january-fold", artifact_hash="hash-january")
    february = SimpleNamespace(binding_id="february-fold", artifact_hash="hash-february")
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
                "spot": 100.0, "calendar_observed_through": "2026-03-09",
            },
            panel_anchor="2026-03-09",
        )

    events = [
        event_inputs("JAN1", "evt-jan1", "2026-01-20"),
        event_inputs("JAN2", "evt-jan2", "2026-01-31"),
        event_inputs("FEB1", "evt-feb1", "2026-02-01"),
    ]

    def selector(selected_binding, event_date, received_as_of):
        calls.append((selected_binding, event_date, received_as_of))
        return february if event_date.month == 2 else january

    monkeypatch.setattr(native_score_batch, "select_gate_forecast_source", selector)
    monkeypatch.setattr(
        native_score_batch, "build_native_score_inputs",
        lambda bundle: captured.append(bundle) or "native-inputs",
    )

    assembled, refusals = _assemble(binding, events, as_of=as_of)

    assert refusals == ()
    assert len(assembled) == 3
    assert calls == [
        (binding, pd.Timestamp("2026-01-20"), as_of),
        (binding, pd.Timestamp("2026-02-01"), as_of),
    ]
    assert [bundle.gate_forecast_source for bundle in captured] == [
        january, january, february,
    ]
    assert "hash-january" in assembled[events[0].key][0].model_artifact_refs
    assert "hash-january" in assembled[events[1].key][0].model_artifact_refs
    assert "hash-february" in assembled[events[2].key][0].model_artifact_refs
