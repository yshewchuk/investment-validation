from dataclasses import replace

import pytest

from checks.phase4_real import (
    _ANALOG_CONTROL_DIMENSIONS,
    _ANALOG_VIEW_FIELDS,
    _analog_control_source,
    _compare_dimension,
    _factory_structure_controls,
    _fake_result,
    _legacy_analog_expected,
    _native_record,
    _numerical_independence_control,
    _numerical_independence_source,
    _simulation_acceptance_controls,
)
from engine.v2.scoring.source_inputs import build_native_score_inputs
from engine.v2.scoring.stages import _execute_analogs


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
    assert control["analog_independent_recomputation"] is True
    assert control["analog_planted_defects_rejected"] is True


def _analog_inputs(**changes):
    return build_native_score_inputs(
        replace(_numerical_independence_source(), **changes))


def _native_analogs(inputs):
    values: dict = {}
    flags: list = []
    output = _execute_analogs(inputs, values, flags)
    return output, flags


def test_analog_clean_case_recomputes_legacy_matcher_exactly():
    legacy = _legacy_analog_expected()
    output, flags = _native_analogs(_analog_inputs())

    assert flags == []
    # Three exact bucket matches: no widening, a real bootstrap interval.
    assert legacy["n_analogs"] == 3
    assert legacy["ci_low"] is not None and legacy["ci_high"] is not None
    assert output == legacy
    assert _compare_dimension(legacy, output, "analogs")["agree"] is True


def test_analog_inputs_carry_no_legacy_answer():
    inputs = _analog_inputs()
    rows = inputs.analogs["source_rows"]

    assert {row["row_id"] for row in rows} == {
        "exact-a", "exact-b", "exact-c", "wide-a", "wide-b", "wide-missing",
    }
    for row in rows:
        assert set(row) == {"row_id", "realized_return", *_ANALOG_CONTROL_DIMENSIONS}
    assert not set(inputs.analogs["recipe"]) & set(_ANALOG_VIEW_FIELDS)


@pytest.mark.parametrize("defect", [
    "perturbed_source_row", "changed_min_analogs",
    "changed_bootstrap_seed", "tampered_bound_row",
])
def test_every_planted_analog_defect_fails_the_comparison(defect):
    control = _numerical_independence_control()

    assert control["analog_planted_defects"][defect] is True


def test_perturbed_source_row_moves_the_native_answer_not_the_hash_check():
    rows = tuple(
        {**row, "realized_return": 0.35} if row["row_id"] == "exact-b" else row
        for row in _analog_control_source()["analog_source_rows"]
    )
    legacy = _legacy_analog_expected()
    output, flags = _native_analogs(_analog_inputs(analog_source_rows=rows))

    assert flags == []
    assert output["n_analogs"] == legacy["n_analogs"]
    assert output["exp_pnl_analog"] != legacy["exp_pnl_analog"]
    comparison = _compare_dimension(legacy, output, "analogs")
    assert comparison["agree"] is False
    assert "exp_pnl_analog" in comparison["finding_fields"]


def test_changed_min_analogs_widens_and_fails_the_comparison():
    source = _analog_control_source()
    legacy = _legacy_analog_expected()
    output, flags = _native_analogs(_analog_inputs(
        analog_recipe={**source["analog_recipe"], "min_analogs": 4}))

    assert flags == []
    # One widening step adds wide-a and wide-b (wide-missing has no return).
    assert output["n_analogs"] == 5
    comparison = _compare_dimension(legacy, output, "analogs")
    assert comparison["agree"] is False
    assert "n_analogs" in comparison["finding_fields"]


def test_row_tampered_after_hash_binding_refuses_and_fails_the_comparison():
    inputs = _analog_inputs()
    rows = [dict(row) for row in inputs.analogs["source_rows"]]
    rows[0]["realized_return"] = 0.35
    tampered = replace(inputs, analogs={**inputs.analogs, "source_rows": rows})
    output, flags = _native_analogs(tampered)

    assert output == {}
    assert any("ANALOG_POPULATION_CORRUPT" in flag for flag in flags)
    empty = {name: None for name in _ANALOG_VIEW_FIELDS}
    assert _compare_dimension(_legacy_analog_expected(), empty, "analogs")["agree"] is False


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
