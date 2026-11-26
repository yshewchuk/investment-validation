from __future__ import annotations

import pytest

from engine.v2.ops.errors import OpsError
from experiments import size_quantile_binding as binding

FIXED = {
    "learning_rate": 0.03, "l2_regularization": 10.0, "max_iter": 3000, "early_stopping": True,
    "validation_fraction": 0.1, "n_iter_no_change": 10, "tol": 1.0e-07, "random_state": 0,
}
SPEC = {"grid": {"axes": {"max_leaf_nodes": [7, 15], "min_samples_leaf": [300, 1000]}, "fixed": FIXED}}


def _reason(callable_):
    with pytest.raises(OpsError) as info:
        callable_()
    assert info.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    return info.value.problem.details["reason"]


# --------------------------------------------------------------------------
# bind_grid
# --------------------------------------------------------------------------


def _hgb(cid, subset, leaves=7, leaf=300, **extra):
    params = {**FIXED, "max_leaf_nodes": leaves, "min_samples_leaf": leaf, "loss": "quantile", "quantile": 0.5}
    params.update(extra)
    return {"candidate_id": cid, "feature_recipe": {"subset": subset},
            "recipe": {"schema_version": "magnitude_ridge_hgb_recipe.v1", "model": "history_hgb",
                       "estimator": {"params": params}}}


def _ignored(cid, subset, schema, model, params):
    return {"candidate_id": cid, "feature_recipe": {"subset": subset},
            "recipe": {"schema_version": schema, "model": model, "estimator": {"params": params}}}


def _grid_plan(*extra):
    candidates = [_hgb(f"h{leaves}_{leaf}", "HMOSXLP", leaves, leaf) for leaves in (7, 15) for leaf in (300, 1000)]
    candidates += [
        _ignored("ridge", "HMOSXLP", "magnitude_ridge_ridge_recipe.v1", "history_ridge", {"alpha": 1.0}),
        _ignored("enet", "HMOSXLP", "magnitude_elasticnet_recipe.v1", "history_elasticnet",
                 {"alpha": 0.1, "l1_ratio": 0.5}),
        _ignored("comparator", "HMOSXLP", "size_v1_4.comparator.v1", "size_v1_4", {"leaves": 3}),
        _hgb("abs_loss", "MO", 7, 300, loss="absolute_error"),
    ]
    return {"plan_id": "P1", "candidates": candidates + list(extra)}


def test_grid_binds_with_ignored_and_extra_loss_candidates():
    result = binding.bind_grid(SPEC, _grid_plan())
    assert result == {"plan_id": "P1",
                      "configurations": [[7, 300], [7, 1000], [15, 300], [15, 1000]],
                      "hgb_candidates": 5}


def test_changed_fixed_value_refuses():
    plan = _grid_plan(_hgb("bad", "MO", 7, 300, l2_regularization=5.0))
    assert _reason(lambda: binding.bind_grid(SPEC, plan)) == "v14_grid_mismatch"


def test_early_stopping_as_int_refuses_because_bool_must_match_bool():
    plan = _grid_plan(_hgb("bad", "MO", 7, 300, early_stopping=1))
    assert _reason(lambda: binding.bind_grid(SPEC, plan)) == "v14_grid_mismatch"


def test_extra_axis_value_in_a_non_hmosxlp_subset_refuses():
    plan = _grid_plan(_hgb("bad", "MO", 31, 300))
    assert _reason(lambda: binding.bind_grid(SPEC, plan)) == "v14_grid_mismatch"


def test_missing_pair_among_hmosxlp_candidates_refuses():
    plan = _grid_plan()
    plan["candidates"] = [c for c in plan["candidates"] if c["candidate_id"] != "h15_1000"]
    assert _reason(lambda: binding.bind_grid(SPEC, plan)) == "v14_grid_mismatch"


def test_no_hmosxlp_hgb_candidate_refuses():
    plan = {"plan_id": "P1", "candidates": [_hgb("m", "MO", 7, 300)]}
    assert _reason(lambda: binding.bind_grid(SPEC, plan)) == "v14_grid_mismatch"


@pytest.mark.parametrize("plan", [
    {"plan_id": "P1", "candidates": "nope"},
    {"plan_id": "P1", "candidates": [{"candidate_id": "x", "recipe": None}]},
    {"candidates": []},
])
def test_malformed_plan_is_a_typed_refusal(plan):
    assert _reason(lambda: binding.bind_grid(SPEC, plan)) == "v14_grid_mismatch"


# --------------------------------------------------------------------------
# bind_b3
# --------------------------------------------------------------------------


def _cand(cid, subset):
    return {"candidate_id": cid, "feature_recipe": {"subset": subset}, "recipe": {}}


def _plan():
    return {"plan_id": "P1",
            "candidates": [_cand("HM1", "HMOSXLP"), _cand("HM2", "HMOSXLP"), _cand("HM3", "HMOSXLP"),
                           _cand("MO1", "MO")],
            "blends": [{"blend_id": "B1", "component_candidate_ids": ["HM1", "HM2"]},
                       {"blend_id": "BMIX", "component_candidate_ids": ["HM1", "MO1"]}],
            "selection": {"rule_metrics": {"mae": "normalized.mae"}}}


def _ok(cid, norm, mae=2.0):
    return {"candidate_id": cid, "metrics": {"mae": mae}, "normalized": {"mae": norm}}


def _failed(cid):
    return {"candidate_id": cid, "metrics": None, "normalized": {"mae": None}}


def _selection(winner, scores, plan_id="P1", extra_items=()):
    item = {"outer_year": 2025, "winners": {"mae": {"candidate_id": winner}}, "scores": scores}
    return {"plan_id": plan_id, "selections": [item, *extra_items]}


def test_mae_winner_is_kept_even_when_another_candidate_has_lower_mae():
    sel = _selection("HM2", [_ok("HM1", 0.90), _ok("HM2", 1.00)])
    result = binding.bind_b3(_plan(), sel)
    assert result["b3_candidate_id"] == "HM2"
    assert result["rule"] == "v14_mae_winner"
    assert result["kind"] == "candidate"
    assert result["m3_pinball_p50_from_mae"] == 1.0


def test_blend_m3_of_only_hmosxlp_components_is_kept():
    sel = _selection("B1", [_ok("B1", 0.8, mae=3.0), _ok("HM1", 0.5)])
    result = binding.bind_b3(_plan(), sel)
    assert (result["b3_candidate_id"], result["kind"], result["rule"]) == ("B1", "blend", "v14_mae_winner")
    assert result["m3_pinball_p50_from_mae"] == 1.5


def test_non_hmosxlp_m3_falls_back_to_lowest_normalized_mae_then_lowest_id():
    scores = [_ok("MO1", 0.5), _failed("HM1"), _ok("HM3", 0.7), _ok("HM2", 0.7), _ok("B1", 0.05),
              _ok("BMIX", 0.04)]
    result = binding.bind_b3(_plan(), _selection("MO1", scores))
    assert result["b3_candidate_id"] == "HM2"
    assert result["rule"] == "lowest_inner_mae_hmosxlp"
    assert result["kind"] == "candidate"


def test_blend_with_a_non_hmosxlp_component_falls_back_to_lowest_inner_mae_hmosxlp():
    scores = [_ok("BMIX", 0.04, mae=3.0), _ok("HM1", 0.5), _ok("HM2", 0.7), _ok("MO1", 0.1)]
    result = binding.bind_b3(_plan(), _selection("BMIX", scores))
    assert result["m3_candidate_id"] == "BMIX"
    assert result["b3_candidate_id"] == "HM1"
    assert result["rule"] == "lowest_inner_mae_hmosxlp"
    assert result["kind"] == "candidate"
    assert result["m3_pinball_p50_from_mae"] == 1.5


def test_selection_plan_id_mismatch_refuses():
    sel = _selection("HM2", [_ok("HM2", 1.0)], plan_id="OTHER")
    assert _reason(lambda: binding.bind_b3(_plan(), sel)) == "v14_selection_mismatch"


def test_plan_rule_metric_other_than_normalized_mae_refuses():
    plan = _plan()
    plan["selection"]["rule_metrics"]["mae"] = "raw.mae"
    assert _reason(lambda: binding.bind_b3(plan, _selection("HM2", [_ok("HM2", 1.0)]))) == "v14_selection_mismatch"


def test_missing_winners_refuses_as_m3_unbound():
    sel = _selection("HM2", [_ok("HM2", 1.0)])
    del sel["selections"][0]["winners"]
    assert _reason(lambda: binding.bind_b3(_plan(), sel)) == "m3_unbound"


def test_two_2025_items_refuse_as_m3_unbound():
    sel = _selection("HM2", [_ok("HM2", 1.0)], extra_items=[
        {"outer_year": "2025", "winners": {"mae": {"candidate_id": "HM2"}}, "scores": [_ok("HM2", 1.0)]}])
    assert _reason(lambda: binding.bind_b3(_plan(), sel)) == "m3_unbound"


def test_unknown_m3_refuses_as_m3_unbound():
    sel = _selection("ZZ", [_ok("ZZ", 0.5)])
    assert _reason(lambda: binding.bind_b3(_plan(), sel)) == "m3_unbound"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "0.5", None, True])
def test_non_finite_or_non_number_normalized_mae_on_a_considered_row_refuses(bad):
    scores = [_ok("MO1", 0.5), {"candidate_id": "HM1", "metrics": {"mae": 2.0}, "normalized": {"mae": bad}}]
    assert _reason(lambda: binding.bind_b3(_plan(), _selection("MO1", scores))) == "b3_unbound"


def test_no_eligible_fallback_row_refuses_as_b3_unbound():
    scores = [_ok("MO1", 0.5), _failed("HM1"), _failed("HM2")]
    assert _reason(lambda: binding.bind_b3(_plan(), _selection("MO1", scores))) == "b3_unbound"


def _report(plan_id="P1", status="ok", inner=None):
    return {"plan_id": plan_id, "distribution": {"status": status, "inner": inner or []}}


def _inner_rows():
    return [
        {"outer_year": 2025, "candidate_id": "HM2", "inner_year": 2024, "status": "ok",
         "pooled": {"pinball_p50": 0.5, "pinball_mean": 0.6, "coverage_p90": 0.88, "coverage_p10_p90": 0.79}},
        {"outer_year": 2025, "candidate_id": "HM2", "inner_year": 2023, "status": "no_pool", "pooled": None},
        {"outer_year": 2024, "candidate_id": "HM2", "inner_year": 2023, "status": "ok",
         "pooled": {"pinball_p50": 9.0, "pinball_mean": 9.0, "coverage_p90": 0.0, "coverage_p10_p90": 0.0}},
    ]


def test_report_values_are_printed_to_stderr_only(capsys):
    sel = _selection("HM2", [_ok("HM2", 1.0)])
    result = binding.bind_b3(_plan(), sel, _report(inner=_inner_rows()))
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("[quantile_bind]") == 1
    assert "m3=HM2" in captured.err and "b3=HM2" in captured.err
    assert "2023:None/None" in captured.err and "2024:0.6/0.79" in captured.err
    assert [entry["inner_year"] for entry in result["m3_band"]] == [2023, 2024]
    assert result["m3_band_status"] == "no_pool"


def test_all_ok_band_reports_ok():
    rows = [row for row in _inner_rows() if row["inner_year"] == 2024]
    result = binding.bind_b3(_plan(), _selection("HM2", [_ok("HM2", 1.0)]), _report(inner=rows))
    assert result["m3_band_status"] == "ok"


def test_no_report_reports_report_not_given(capsys):
    result = binding.bind_b3(_plan(), _selection("HM2", [_ok("HM2", 1.0)]))
    assert result["m3_band_status"] == "report_not_given"
    assert result["m3_band"] == []
    assert capsys.readouterr().out == ""


def test_report_plan_id_mismatch_refuses():
    sel = _selection("HM2", [_ok("HM2", 1.0)])
    assert _reason(lambda: binding.bind_b3(_plan(), sel, _report(plan_id="OTHER"))) == "v14_report_mismatch"


def test_malformed_report_missing_distribution_refuses_as_v14_report_mismatch():
    sel = _selection("HM2", [_ok("HM2", 1.0)])
    assert _reason(lambda: binding.bind_b3(_plan(), sel, {"plan_id": "P1"})) == "v14_report_mismatch"


def test_blend_not_distributed_status_passes_through():
    sel = _selection("B1", [_ok("B1", 0.8)])
    result = binding.bind_b3(_plan(), sel, _report(status="blend_not_distributed"))
    assert result["m3_band_status"] == "blend_not_distributed"
    assert result["m3_band"] == []
