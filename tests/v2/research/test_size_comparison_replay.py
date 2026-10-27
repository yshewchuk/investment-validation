"""Typed replay-shape refusals do not weaken the existing comparison verifier."""
from copy import deepcopy

import pytest

from engine.v2.ops.errors import OpsError
from experiments import size_comparator, size_comparison_run
from tests.v2.research.test_size_comparator import case


@pytest.mark.parametrize("poison", ["none", "list", "missing", "forecasts_none", "forecasts_dict",
    "row_none", "row_list", "row_missing_prediction", "row_extra", "fit_details"])
def test_invalid_result_shapes_have_typed_conflicts_without_fitting(monkeypatch, poison):
    indexed, train, fold, plan, execution = case(1)
    monkeypatch.setattr(size_comparator, "fit_recipe_estimator", lambda *a: pytest.fail("unexpected fit"))
    result = size_comparator.forecasts(indexed, train, fold, plan, execution, progress=lambda *a: None)
    inputs = {"features": list(indexed.values()), "training": train.to_dict("records")}
    size_comparison_run._validate(result, inputs, fold, plan)
    result = deepcopy(result)
    if poison in ("none", "list"):
        result = None if poison == "none" else []
    elif poison == "missing":
        del result["forecasts"]
    elif poison.startswith("forecasts_"):
        result["forecasts"] = None if poison == "forecasts_none" else {}
    elif poison in ("row_none", "row_list"):
        result["forecasts"][0] = None if poison == "row_none" else []
    elif poison == "row_missing_prediction":
        del result["forecasts"][0]["predicted_absolute_move_pct"]
    elif poison == "row_extra":
        result["forecasts"][0]["extra"] = True
    else:
        result["fit_details"][0]["component_fits"] = 999
    with pytest.raises(OpsError) as caught:
        size_comparison_run._validate(result, inputs, fold, plan)
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
