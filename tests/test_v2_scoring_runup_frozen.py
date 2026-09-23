from dataclasses import replace
from types import SimpleNamespace

import pytest

from engine.v2.contracts import ScoreRequest
from engine.v2.domain.generation import generate, price
from engine.v2.scoring import application
from engine.v2.scoring.native_analog import source_population_hash
from engine.v2.scoring.stages import NativeScoreInputs, STAGE_NAMES, StageReceipt


def _analog_block() -> dict:
    # A minimal real analog recipe so a row can carry an actual
    # exp_pnl_analog number (engine/score.py:847-848 legacy parity, applied
    # in application._has_score_number). This fixture otherwise only tests
    # forecast-stage wiring and never populates a real financial number.
    rows = [
        {"row_id": "a", "features": {"move": 1.0}, "realized_pnl": 3.0},
        {"row_id": "b", "features": {"move": -1.0}, "realized_pnl": -1.0},
    ]
    return {
        "recipe": {
            "feature_names": ("move",),
            "neighbors": 2,
            "population_hash": source_population_hash(rows),
        },
        "source_rows": rows,
        "query_features": {"move": 1.0},
    }


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
                model_id="implied-model",
                role="implied_t1",
                feature_order=("days_before_print",),
                output_names=("prediction",),
                members=(_member("model", "sha256:implied"),),
            ),
            SimpleNamespace(
                binding_id="runup-binding",
                model_id="runup-model",
                role="runup_move",
                feature_order=("days_before_print",),
                output_names=(runup_output,),
                members=(
                    _member("model", "sha256:runup"),
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
            artifact_hashes=(
                "sha256:runup", "sha256:interval", "sha256:calibration",
            ),
            reason_codes=reasons,
        )


def test_frozen_runup_scales_raw_d14_and_retains_provenance():
    # 2026-09-18 fix: a scored row needs a real exp_pnl_model/exp_pnl_analog
    # number (engine/score.py:847-848 legacy parity); this fixture otherwise
    # only exercises forecast-stage wiring, so give it a real analog block.
    inputs = replace(_inputs(), analogs=_analog_block())
    record = application.score_frozen(
        _request(),
        _Frozen(),
        _release(),
        _inference_requests(),
        {"_native_inputs": inputs},
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


def test_captured_runup_binding_names_are_raw_not_final():
    """Regression for the 2026-09-21 defect: both places that declare the
    ``runup_move`` frozen binding's output name -- the Phase 4 capture side
    (``engine.score._RUNUP_MODEL_OUTPUTS``) and the real Phase 5 release
    stager (``tools.phase5_prepare_release.OUTPUT_NAMES``) -- must name the
    RAW artifact output (one of ``application._RUNUP_RAW_NAMES``), never a
    FINAL one (``application._RUNUP_FINAL_NAMES``). The artifact
    (``LogTargetRegressor``) only ever returns the raw T-14 magnitude;
    ``application._runup_frozen_output`` is the one place the day scale
    (``days_before_print / 14``) is applied, and it applies it exactly once.
    A binding mislabeled with the final name makes native refuse the row
    (PRETRANSFORMED_FROZEN_OUTPUT:runup_move) instead of scoring it -- the
    bug this test catches -- while a binding that already carried the final
    name and was ALSO scaled again would silently double the move.
    """
    import engine.score as score
    import tools.phase5_prepare_release as phase5_prepare_release

    captured_name = score._RUNUP_MODEL_OUTPUTS["runup_move"][0]
    staged_name = phase5_prepare_release.OUTPUT_NAMES["runup_move"][0]

    for name, label in ((captured_name, "phase4 capture"),
                        (staged_name, "phase5 release")):
        assert name in application._RUNUP_RAW_NAMES, (
            f"{label}: runup_move output name {name!r} must be raw"
        )
        assert name not in application._RUNUP_FINAL_NAMES, (
            f"{label}: runup_move output name {name!r} must not be final"
        )

    # End-to-end: the name the capture side actually writes must let native
    # score the row (no refusal) and must scale the raw artifact prediction
    # by exactly one factor of days_before_print / 14 -- not zero times
    # (refused, staying None) and not twice (a further /14 or *0.5).
    inputs = replace(_inputs(), analogs=_analog_block())
    frozen = _Frozen()
    frozen.runup_output = captured_name
    record = application.score_frozen(
        _request(),
        frozen,
        _release(captured_name),
        _inference_requests(),
        {"_native_inputs": inputs},
    )

    assert record.validation_status == "scored"
    assert "PRETRANSFORMED_FROZEN_OUTPUT:runup_move" not in record.reason_codes
    # raw prediction is 8.0 (see _Frozen.infer), days=7.0 -> scale 0.5.
    assert record.forecasts["runup_move_raw_d14"] == pytest.approx(8.0)
    assert record.forecasts["runup_move_scale"] == pytest.approx(0.5)
    assert record.forecasts["runup_move_prediction"] == pytest.approx(4.0)


def test_score_frozen_str_runup_empty_forecast_publishes_im_t1_driver_name():
    # STR-RUNUP with forecast={} (no driver_name carrier at all): the
    # frozen path must source driver_name "im_t1" from PAYOFF_DRIVER, and
    # because model_vs_market is gated on driver_name == "abs_move" in
    # financial.py, it must stay None for this strategy no matter what
    # driver_prediction/implied_move are.
    inputs = replace(_inputs(), analogs=_analog_block(), forecast={})
    record = application.score_frozen(
        _request(), _Frozen(), _release(), _inference_requests(),
        {"_native_inputs": inputs},
    )

    assert record.resolved_request["driver_name"] == "im_t1"
    assert record.financial_diagnostics["model_vs_market"] is None
