from experiments.magnitude_search_refits import outer_refit_bound
from experiments.magnitude_search_request_versions import compile_request
from experiments.size_search_report import resource_report
from tests.v2.research.test_magnitude_search_request_v13 import _request
from tests.v2.research.test_magnitude_search_request_v14_winner_rules import _with_rules

_RULES = ["overall", "rmse", "mae"]


def test_outer_refit_bound_counts_winner_units():
    assert outer_refit_bound(None) == 1
    assert outer_refit_bound({}) == 1
    assert outer_refit_bound({"winner_rules": list(_RULES)}) == 3
    assert outer_refit_bound({"winner_rules": list(_RULES), "extra_refits": ["a", "b"]}) == 5


def test_winner_rules_add_one_outer_refit_slot_per_fold():
    baseline = compile_request(_request(14))
    request = _with_rules(14, list(_RULES))
    added = len(baseline["horizons"]) * baseline["counts"]["outer_folds"] * 2
    compiled = compile_request(request)
    assert compiled["counts"]["search_fits"] - baseline["counts"]["search_fits"] == added


def test_resource_report_counts_winner_outer_challengers():
    plan = {"counts": {"candidates": 2, "inner_folds": 3, "outer_folds": 1},
            "request": {"selection": {"winner_rules": list(_RULES)},
                        "limits": {"child": {"wall_seconds": 10}}}}
    assert resource_report(plan)["child_jobs"]["outer_challengers"] == 3


def test_resource_report_counts_blend_outer_challengers(monkeypatch):
    monkeypatch.setattr(
        "experiments.magnitude_search_request_v3.distinct_component_count",
        lambda names: 4)
    outer_folds = 1
    plan = {"counts": {"candidates": 2, "inner_folds": 3, "outer_folds": outer_folds},
            "request": {"selection": {"winner_rules": list(_RULES)},
                        "blends": ["size_b1_v1"],
                        "limits": {"child": {"wall_seconds": 10}}}}
    assert resource_report(plan)["child_jobs"]["outer_challengers"] == (
        outer_folds * (3 + 4 + 1))
