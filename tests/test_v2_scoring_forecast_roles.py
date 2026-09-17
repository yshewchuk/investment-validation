import pytest

from engine.v2.scoring.stages import (
    NativeScoreInputs,
    STAGE_NAMES,
    StageReceipt,
    assemble_native_values,
)


def _inputs(*, forecast=None, simulation=None, strategy="STR-THRU"):
    receipts = tuple(
        StageReceipt(stage, "declared-input", "declared-output")
        for stage in STAGE_NAMES
        if stage != "diagnostics"
    )
    return NativeScoreInputs(
        context={
            "strategy": strategy,
            "spot": 100.0,
            "expiry": "2026-10-16",
        },
        features={},
        forecast=forecast or {},
        geometry=None,
        pricing=None,
        analogs={},
        simulation=simulation or {},
        gate={},
        chooser={},
        diagnostics={},
        source_ref="forecast-role-test",
        stage_receipts=receipts,
    )


def _flags(inputs, strategy=None):
    return set(assemble_native_values(inputs, strategy=strategy)["flags"])


def _frozen(outputs, *, required_roles=()):
    return {
        "frozen_outputs": outputs,
        "artifact_hashes": ("sha256:model",),
        "required_roles": required_roles,
    }


def test_driver_name_metadata_cannot_satisfy_str_thru_driver():
    flags = _flags(_inputs(forecast={"driver_name": "abs_move"}))

    assert "MISSING_FORECAST_OUTPUT:driver" in flags
    assert "MISSING_FORECAST_INPUT" in flags


@pytest.mark.parametrize(
    ("strategy", "outputs", "missing"),
    (
        ("STR-RUNUP", {"driver_prediction": 6.0}, "runup_move"),
        ("STR-RUNUP", {"runup_move_prediction": 2.0}, "implied_t1"),
        ("TWIN-P", {}, "size"),
        ("TWIN-P5", {}, "size"),
        ("CND-PS", {}, "size"),
        ("BFLY-P", {}, "size"),
        ("BFLY-P5", {}, "size"),
        ("RAMP7", {}, "size"),
        ("CTR5", {}, "size"),
    ),
)
def test_strategy_contract_requires_its_forecast_roles(
    strategy, outputs, missing,
):
    flags = _flags(_inputs(
        strategy=strategy,
        forecast=_frozen(outputs),
    ))

    assert f"MISSING_FORECAST_OUTPUT:{missing}" in flags
    assert not any(flag.startswith("UNKNOWN_FORECAST_ROLE:") for flag in flags)


def test_valid_frozen_role_names_map_to_distinct_outputs():
    outputs = {
        "driver_prediction": 6.0,
        "forecast_abs_move": 7.0,
        "runup_move_prediction": 2.0,
        "pred_iv_crush_30": -18.0,
    }
    inputs = _inputs(
        strategy="CAL-P",
        forecast=_frozen(
            outputs,
            required_roles=(
                "driver", "size", "implied_t1", "runup_move", "iv_crush",
            ),
        ),
    )

    values = assemble_native_values(inputs)
    flags = set(values["flags"])

    assert not any(flag.startswith("UNKNOWN_FORECAST_ROLE:") for flag in flags)
    assert not any(flag.startswith("MISSING_FORECAST_OUTPUT:") for flag in flags)
    for field, expected in outputs.items():
        assert values[field] == pytest.approx(expected)


def test_planned_exit_adds_size_and_iv_crush_requirements():
    flags = _flags(_inputs(
        forecast=_frozen({"driver_prediction": 6.0}),
        simulation={"mode": "planned_exit"},
    ))

    assert "MISSING_FORECAST_OUTPUT:size" in flags
    assert "MISSING_FORECAST_OUTPUT:iv_crush" in flags
    assert "MISSING_FORECAST_OUTPUT:driver" not in flags


def test_optional_forecast_outputs_remain_optional():
    flags = _flags(_inputs(
        forecast=_frozen({"driver_prediction": 6.0}),
    ))

    assert not any(flag.startswith("MISSING_FORECAST_OUTPUT:") for flag in flags)


def test_nonfinite_required_output_is_not_misreported_as_missing():
    flags = _flags(_inputs(
        forecast=_frozen({"driver_prediction": float("nan")}),
    ))

    assert "NONFINITE_FORECAST_OUTPUT:driver_prediction" in flags
    assert "MISSING_FORECAST_OUTPUT:driver" not in flags


def test_null_required_output_is_reported_as_missing_not_nonfinite():
    flags = _flags(_inputs(
        forecast=_frozen({"driver_prediction": None}),
    ))

    assert "MISSING_FORECAST_OUTPUT:driver" in flags
    assert "NONFINITE_FORECAST_OUTPUT:driver_prediction" not in flags
