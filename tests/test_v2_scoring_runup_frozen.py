from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import generate, price
from engine.v2.scoring import application
from engine.v2.scoring.stages import NativeScoreInputs, STAGE_NAMES, StageReceipt


def _request():
    return ScoreRequest(
        event_id="evt-runup",
        calendar_revision="cal-1",
        strategy_version="STR-RUNUP",
        deployment_id="dep-1",
        decision_clock_id="entry-close",
        requested_decision_at="2026-09-16",
        snapshot_id="snap-1",
        mode="replay",
        fill_model={"alpha": 0.5},
    )


def _receipts():
    return tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES
        if stage != "diagnostics"
    )


def _inputs():
    context = {
        "ticker": "AAA",
        "strategy": "STR-RUNUP",
        "event_date": "2026-09-16",
        "entry_date": "2026-09-01",
        "exit_date": "2026-09-16",
        "expiry": "2026-09-18",
        "spot": 100.0,
        "strike": 100.0,
        "days_before_print": 7.0,
        "runup_move_prediction": 999.0,
        "runup_move_p10": 998.0,
    }
    geometry = generate("STR-RUNUP", context)
    quotes = {
        (leg.right, leg.strike, leg.expiry): {"bid": 1.0, "ask": 2.0}
        for leg in geometry.legs
    }
    return NativeScoreInputs(
        context=context,
        features={
            "days_before_print": 14.0,
            "model_inputs": {"days_before_print": 14.0},
            "runup_move_prediction": 997.0,
        },
        forecast={
            "driver_name": "im_t1",
            "runup_move_prediction": 996.0,
            "models": {
                "driver_prediction": {"intercept": 95.0, "coefficients": {}},
                "runup_move_prediction": {"intercept": 94.0, "coefficients": {}},
            },
        },
        geometry=geometry,
        pricing=price(geometry, quotes, 0.5),
        analogs={"runup_move_prediction": 993.0},
        simulation={
            "runup_move_prediction": 992.0,
            "terminal_spots": (90.0, 110.0),
            "capital_at_risk": 3.0,
        },
        gate={
            "runup_move_prediction": 991.0,
            "model": {"intercept": 1.0, "coefficients": {}},
            "threshold": 0.5,
        },
        chooser={"runup_move_prediction": 990.0},
        diagnostics={
            "runup_move_prediction": 989.0,
            "runup_move_p10": 988.0,
        },
        source_ref="runup-fixture",
        stage_receipts=_receipts(),
    )


def _member(name, content_hash):
    return SimpleNamespace(name=name, content_hash=content_hash)


def _release(runup_output="prediction"):
    return SimpleNamespace(
        release_id="release-1",
        bindings=(
            SimpleNamespace(
                binding_id="implied-binding",
                role="implied_t1",
                output_names=("prediction",),
                members=(),
            ),
            SimpleNamespace(
                binding_id="runup-binding",
                role="runup_move",
                output_names=(runup_output,),
                members=(
                    _member("residual_interval", "sha256:interval"),
                    _member("calibration", "sha256:calibration"),
                ),
            ),
        ),
    )


def _inference_requests():
    return (
        SimpleNamespace(binding_id="implied-binding"),
        SimpleNamespace(
            binding_id="runup-binding",
            fold_start="2026-09-01",
            calibration_ref="calibration:runup-v1",
        ),
    )


class _Frozen:
    runup_status = "READY"
    runup_output = "prediction"

    def infer(self, release, inference_request):
        if inference_request.binding_id == "implied-binding":
            return SimpleNamespace(
                status="READY",
                release_id=release.release_id,
                binding_id="implied-binding",
                model_id="implied-model",
                output_names=("prediction",),
                predictions=((6.0,),),
                artifact_hashes=("sha256:implied",),
                reason_codes=(),
            )
        reasons = () if self.runup_status == "READY" else ("ARTIFACT_INVALID",)
        return SimpleNamespace(
            status=self.runup_status,
            release_id=release.release_id,
            binding_id="runup-binding",
            model_id="runup-model",
            output_names=(self.runup_output,),
            predictions=((8.0,),),
            prediction_interval=((6.0, 10.0, 2.0),),
            artifact_hashes=("sha256:runup",),
            reason_codes=reasons,
        )


def test_frozen_runup_scales_raw_d14_and_retains_provenance():
    record = application.score_frozen(
        _request(),
        _Frozen(),
        _release(),
        _inference_requests(),
        {"_native_inputs": _inputs()},
    )

    assert record.validation_status == "scored"
    assert record.forecasts["driver_prediction"] == pytest.approx(6.0)
    assert record.forecasts["runup_move_raw_d14"] == pytest.approx(8.0)
    assert record.forecasts["runup_move_prediction"] == pytest.approx(4.0)
    assert record.forecasts["runup_move_days"] == pytest.approx(7.0)
    assert record.forecasts["runup_move_scale"] == pytest.approx(0.5)
    assert record.uncertainty["runup_move_raw_d14_p10"] == pytest.approx(6.0)
    assert record.uncertainty["runup_move_raw_d14_p90"] == pytest.approx(10.0)
    assert record.uncertainty["runup_move_p10"] == pytest.approx(3.0)
    assert record.uncertainty["runup_move_p90"] == pytest.approx(5.0)
    assert record.uncertainty["runup_move_sd"] == pytest.approx(1.0)
    provenance = record.forecasts["runup_move_provenance"]
    assert provenance["fold"] == "2026-09-01"
    assert provenance["calibration_ref"] == "calibration:runup-v1"
    assert provenance["interval_artifact_hashes"] == ("sha256:interval",)
    assert provenance["calibration_artifact_hashes"] == ("sha256:calibration",)


def test_frozen_runup_rejects_pretransformed_model_output():
    frozen = _Frozen()
    frozen.runup_output = "runup_move_prediction"
    record = application.score_frozen(
        _request(),
        frozen,
        _release("runup_move_prediction"),
        _inference_requests(),
        {"_native_inputs": _inputs()},
    )

    assert record.forecasts["runup_move_prediction"] is None
    assert "PRETRANSFORMED_FROZEN_OUTPUT:runup_move" in record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:runup_move" in record.reason_codes
    assert record.validation_status == "refused"


def test_invalid_runup_artifact_refuses_without_local_or_supplied_fallback():
    frozen = _Frozen()
    frozen.runup_status = "MODEL_NOT_READY"
    record = application.score_frozen(
        _request(),
        frozen,
        _release(),
        _inference_requests(),
        {"_native_inputs": _inputs()},
    )

    assert record.forecasts["runup_move_prediction"] is None
    assert "ARTIFACT_INVALID" in record.reason_codes
    assert "MISSING_FORECAST_OUTPUT:runup_move" in record.reason_codes
    assert record.validation_status == "refused"
    assert record.readiness == "refused"


def test_default_native_runup_numbers_and_record_shape_are_unchanged():
    inputs = _inputs()
    ordinary = replace(
        inputs,
        context={
            key: value for key, value in inputs.context.items()
            if key not in {"runup_move_prediction", "runup_move_p10"}
        },
        features={
            "days_before_print": 7.0,
            "model_inputs": {"days_before_print": 7.0},
        },
        forecast={
            "driver_name": "im_t1",
            "models": {
                "driver_prediction": {"intercept": 6.0, "coefficients": {}},
                "runup_move_prediction": {"intercept": 8.0, "coefficients": {}},
            },
        },
        analogs={},
        simulation={},
        gate={},
        chooser={},
        diagnostics={},
    )

    record = application.score_one(_request(), ordinary)

    assert record.forecasts["driver_prediction"] == pytest.approx(6.0)
    assert record.forecasts["runup_move_prediction"] == pytest.approx(8.0)
    assert "runup_move_raw_d14" not in record.forecasts
    assert "runup_move_provenance" not in record.forecasts
    assert set(record.uncertainty) == {
        "model_p10", "model_p90", "forecast_p10", "forecast_p90",
    }
