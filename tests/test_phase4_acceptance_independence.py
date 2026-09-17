from checks.phase4_real import (
    _factory_structure_controls,
    _fake_result,
    _native_record,
    _numerical_independence_control,
    _simulation_acceptance_controls,
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


def test_independent_simulation_covers_horizon_fill_and_recipe_binding():
    control = _simulation_acceptance_controls()

    assert control["expiry_parity"] is True
    assert control["pre_expiry_parity"] is True
    assert control["material_time_value"] is True
    assert control["fill_propagation"] is True
    assert control["executable_recipe_binding"] is True
    assert control["strict_gate_semantics"] is True
    assert len(control["artifact_refs"]) == 2
    assert control["residual_hash"].startswith("sha256:")

    comparisons = control["comparisons"]
    assert comparisons["dte_0_alpha_0.0"]["entry_cost"] != comparisons["dte_0_alpha_1.0"]["entry_cost"]
    assert comparisons["dte_30_alpha_0.0"]["native"] != comparisons["dte_0_alpha_0.0"]["native"]
