import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd

from engine.v2.ops import native_score_batch
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.scoring.application import score_one
from engine.v2.scoring.stages import _execute_forecast_band
from tests.v2.ops.analog_fixtures import with_synthetic_exact_analogs
from tests.v2.ops.test_native_score_batch import (
    _assemble,
    _calendar_row,
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
    source = SimpleNamespace(
        binding_id="selected-gate-fold", artifact_hash="fold-hash",
        pool_pred=(0.1, 0.2), pool_res=(0.01, 0.02), interval_floor=0.07,
    )
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

    binding = with_synthetic_exact_analogs(
        binding, [event], as_of=pd.Timestamp(as_of).date().isoformat())
    assembled, refusals = _assemble(binding, [event], as_of=as_of)

    assert refusals == ()
    assert calls == [(binding, event.key.event_date, pd.Timestamp(as_of))]
    assert assembled[event.key][1] == "native-inputs"
    assert "fold-hash" in assembled[event.key][0].model_artifact_refs
    assert captured[0].gate_forecast_source is source
    assert captured[0].forecast_pool == {
        "predictions": source.pool_pred,
        "residuals": source.pool_res,
        "interval_floor": source.interval_floor,
    }


def test_production_sizing_band_uses_selected_verified_fold_pool(tmp_path):
    binding = _stage_release(tmp_path)
    event = _event_inputs()
    selected = native_score_batch.select_gate_forecast_source(
        binding, event.key.event_date, "2026-01-10")

    assembled, refusals = _assemble(binding, [event])

    assert refusals == ()
    assert assembled[event.key][1].forecast["forecast_pool"] == {
        "predictions": selected.pool_pred,
        "residuals": selected.pool_res,
        "interval_floor": selected.interval_floor,
    }


def test_production_str_thru_score_emits_selected_fold_band(tmp_path, monkeypatch):
    binding = _stage_release(tmp_path)
    original_selector = native_score_batch.select_gate_forecast_source

    def selector(*args, **kwargs):
        source = original_selector(*args, **kwargs)
        return replace(
            source,
            pool_pred=(source.pool_pred[0],) * 250,
            pool_res=(-0.4,) * 83 + (-0.2,) * 83 + (0.1,) * 83 + (0.3,),
            interval_floor=0.15,
        )

    monkeypatch.setattr(
        native_score_batch, "select_gate_forecast_source", selector)
    event = _event_inputs(panel_row={"date": "2026-01-15", "x": -0.3})
    assembled, refusals = _assemble(
        binding, [event], as_of="2026-01-10", feature_names=("x",))

    assert refusals == ()
    request, native_inputs = assembled[event.key]
    observations = []
    score_one(request, native_inputs, observer=observations.append)
    forecast = next(
        item.output_document for item in observations
        if item.receipt.stage == "forecast"
    )

    assert "driver_prediction" in forecast
    assert "forecast_abs_move" not in forecast
    assert forecast["forecast_p10"] == 0.15
    assert abs(forecast["forecast_p90"] - 0.5) < 1e-12
    assert abs(forecast["forecast_sd"] - 0.2075893381976376) < 1e-12


def test_non_str_thru_does_not_use_driver_prediction_for_sizing_band():
    output = {"driver_prediction": 0.5}
    block = {"forecast_pool": {
        "predictions": (0.5,) * 250,
        "residuals": (0.2,) * 250,
        "interval_floor": 0.0,
    }}

    _execute_forecast_band(block, output, "STR-RUNUP")

    assert "forecast_abs_move" not in output
    assert "forecast_p10" not in output
    assert "forecast_p90" not in output
    assert "forecast_sd" not in output


def test_forecast_abs_move_precedes_str_thru_driver_location():
    output = {"forecast_abs_move": 0.4, "driver_prediction": 0.1}
    block = {"forecast_pool": {
        "predictions": (0.5,) * 250,
        "residuals": (0.2,) * 250,
        "interval_floor": 0.0,
    }}

    _execute_forecast_band(block, output, "STR-THRU")

    assert abs(output["forecast_p10"] - 0.6) < 1e-12
    assert abs(output["forecast_p90"] - 0.6) < 1e-12
    assert output["forecast_sd"] == 0.0


def test_missing_or_nonfinite_str_thru_driver_location_leaves_band_unavailable():
    block = {"forecast_pool": {
        "predictions": (0.5,) * 250,
        "residuals": (0.2,) * 250,
        "interval_floor": 0.0,
    }}

    for location in (None, float("nan"), float("inf"), float("-inf")):
        output = {"driver_prediction": location}

        _execute_forecast_band(block, output, "STR-THRU")

        assert "forecast_p10" not in output
        assert "forecast_p90" not in output
        assert "forecast_sd" not in output


def test_thin_str_thru_pool_leaves_band_unavailable():
    output = {"driver_prediction": 0.4}
    block = {"forecast_pool": {
        "predictions": (0.5,) * 249,
        "residuals": (0.2,) * 249,
        "interval_floor": 0.0,
    }}

    _execute_forecast_band(block, output, "STR-THRU")

    assert "forecast_p10" not in output
    assert "forecast_p90" not in output
    assert "forecast_sd" not in output


def test_supported_non_str_thru_forecast_location_emits_band():
    output = {"forecast_abs_move": 0.4, "driver_prediction": 0.1}
    block = {"forecast_pool": {
        "predictions": (0.5,) * 250,
        "residuals": (0.2,) * 250,
        "interval_floor": 0.0,
    }}

    _execute_forecast_band(block, output, "STR-RUNUP")

    assert abs(output["forecast_p10"] - 0.6) < 1e-12
    assert abs(output["forecast_p90"] - 0.6) < 1e-12
    assert output["forecast_sd"] == 0.0


def test_missing_event_id_refuses_before_missing_fold(tmp_path):
    binding = _stage_release(tmp_path, include_size_fold=False)
    calendar_row = dict(_event_inputs().calendar_row)
    calendar_row.pop("event_id")
    event = _event_inputs(calendar_row=calendar_row)

    assembled, refusals = _assemble(binding, [event])

    assert assembled == {}
    assert [refusal.code for refusal in refusals] == ["MISSING_STAGED_INPUT"]
    assert refusals[0].detail == "calendar_row missing event_id"


def test_real_synthetic_fold_populates_gate_output(tmp_path, monkeypatch):
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
    original_assemble = native_score_batch.assemble_score_batch_inputs

    def assemble_with_analogs(*args, **kwargs):
        kwargs["binding"] = with_synthetic_exact_analogs(
            kwargs["binding"], kwargs["events"],
            as_of=pd.Timestamp(kwargs["as_of"]).date().isoformat())
        return original_assemble(*args, **kwargs)

    monkeypatch.setattr(
        native_score_batch, "assemble_score_batch_inputs", assemble_with_analogs)
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


def test_corrupt_staged_fold_bytes_refuse_row_model_not_ready(tmp_path):
    binding = _stage_release(tmp_path)
    descriptor = binding.size_folds.descriptors["2026-01-01"]
    fold_path = tmp_path / "deployment" / descriptor.estimator.path
    fold_path.write_bytes(b"corrupted size fold")
    event = _event_inputs()

    assembled, refusals = _assemble(binding, [event])

    assert assembled == {}
    assert len(refusals) == 1
    assert refusals[0].code == "MODEL_NOT_READY"
    assert refusals[0].detail == "causal size fold unavailable for gate forecast"


def test_only_the_causal_month_with_a_staged_fold_is_assembled(tmp_path):
    binding = _stage_release(tmp_path)
    january = _event_inputs()
    february = _event_inputs(
        key=BoardRequest(ticker="TEST2", strategy="STR-THRU",
                         event_date=pd.Timestamp("2026-02-15"), session="am"),
        calendar_row=_calendar_row(
            event_id="evt-feb", ticker="TEST2", event_date="2026-02-15",
            entry_date="2026-02-15", exit_date="2026-02-16",
            expiry="2026-02-16", calendar_observed_through="2026-02-09"),
        panel_row={"date": "2026-02-15", "signal": 1.5},
        panel_anchor="2026-02-09",
        tier4_row={"pred_abs_move": 0.05,
                   "pred_abs_move_fold_start": "2026-02-01"},
        quote_rows=[
            {"right": "C", "strike": 100.0, "expiry": "2026-02-16",
             "bid": 1.0, "ask": 1.2, "observed_at": "2026-02-09"},
            {"right": "P", "strike": 100.0, "expiry": "2026-02-16",
             "bid": 1.1, "ask": 1.3, "observed_at": "2026-02-09"},
        ],
    )

    assembled, refusals = _assemble(
        binding, [january, february], as_of="2026-02-10")

    assert list(assembled) == [january.key]
    assert len(refusals) == 1
    assert refusals[0].key == february.key
    assert refusals[0].code == "MODEL_NOT_READY"
    assert refusals[0].detail == "causal size fold unavailable for gate forecast"


def test_gate_source_cache_reuses_only_same_causal_month(tmp_path, monkeypatch):
    binding = _stage_release(tmp_path)
    as_of = pd.Timestamp("2026-03-10")
    january = SimpleNamespace(
        binding_id="january-fold", artifact_hash="hash-january",
        pool_pred=(0.1, 0.2), pool_res=(0.01, 0.02), interval_floor=0.11,
    )
    february = SimpleNamespace(
        binding_id="february-fold", artifact_hash="hash-february",
        pool_pred=(0.3, 0.4), pool_res=(0.03, 0.04), interval_floor=0.17,
    )
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

    binding = with_synthetic_exact_analogs(
        binding, events, as_of=pd.Timestamp(as_of).date().isoformat())
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
    assert [bundle.forecast_pool for bundle in captured] == [
        {"predictions": january.pool_pred, "residuals": january.pool_res,
         "interval_floor": january.interval_floor},
        {"predictions": january.pool_pred, "residuals": january.pool_res,
         "interval_floor": january.interval_floor},
        {"predictions": february.pool_pred, "residuals": february.pool_res,
         "interval_floor": february.interval_floor},
    ]
    assert "hash-january" in assembled[events[0].key][0].model_artifact_refs
    assert "hash-january" in assembled[events[1].key][0].model_artifact_refs
    assert "hash-february" in assembled[events[2].key][0].model_artifact_refs
