from checks.phase4_real import (
    _factory_structure_controls,
    _fake_result,
    _native_record,
    _numerical_independence_control,
)


def test_native_acceptance_input_excludes_legacy_numerical_outputs():
    legacy = _fake_result().as_dict()
    legacy.update({"forecast_abs_move": 8.0, "exp_pnl_sim": 0.25})

    inputs = _native_record(legacy, "focused-independence-test")

    assert inputs.forecast == {}
    assert inputs.simulation == {}
    assert inputs.gate == {}
    assert inputs.chooser == {}
    for key in ("driver_prediction", "forecast_abs_move", "exp_pnl_sim",
                "gate_score", "gate_threshold", "gate_pass"):
        assert key not in inputs.diagnostics


def test_preservation_only_native_stages_cannot_pass_acceptance():
    control = _numerical_independence_control()

    assert control["copied_outputs_absent"] is True
    assert control["preservation_only_detected"] is False
    assert control["preservation_only_rejected"] is True
    assert control["independent_recomputation"] is True


def test_factory_controls_exercise_generated_cnd_ps_geometry():
    assert _factory_structure_controls() == {
        "irregular_ladder_rejected": True,
        "exact_mirrors_preserved": True,
        "zero_quantity_reference_legs_preserved": True,
    }
