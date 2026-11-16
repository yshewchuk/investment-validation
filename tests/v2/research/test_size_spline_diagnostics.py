"""Spline worker claims must follow the closed coefficient attribution layout."""
import numpy as np
import pytest

from engine.v2.models.magnitude_elasticnet_spline_recipe import (
    SPLINE_COLUMNS, elasticnet_spline_recipe, spline_coefficient_layout)
from engine.v2.ops.errors import OpsError
from experiments.magnitude_candidate_fit import fit_candidate
from experiments.size_fit_diagnostics import evidence

COLS = list(SPLINE_COLUMNS) + ["extra_a", "extra_b"]
H = "sha256:" + "0" * 64


def diagnostic():
    recipe = elasticnet_spline_recipe(columns=COLS, feature_dictionary_hash=H,
                                      alpha=0.1, l1_ratio=0.25, n_knots=4)
    rng = np.random.default_rng(0)
    train = rng.normal(size=(300, len(COLS)))
    prediction = rng.normal(size=(20, len(COLS)))
    labels = np.abs(rng.normal(size=300)) + 0.1
    result = fit_candidate(recipe, train, labels, prediction, columns=COLS)
    detail = {"horizon_sessions": 1,
              **{key: value for key, value in result.items() if key != "predictions"}}
    return recipe, {"event": "succeeded", "diagnostics": [detail]}


def execution(recipe):
    return {"schema_version": "size_candidate_execution.v1", "fit_recipe": recipe}


def refuses(recipe, document):
    with pytest.raises(OpsError) as caught:
        evidence(document, execution(recipe), [1])
    assert caught.value.code == "EXPERIMENT_IDENTITY_CONFLICT"


def test_spline_coefficient_layout_is_accepted():
    recipe, document = diagnostic()
    entries = document["diagnostics"][0]["coefficients"]
    assert ([(entry["kind"], entry["column"]) for entry in entries]
            == spline_coefficient_layout(recipe))
    assert evidence(document, execution(recipe), [1]) == {"diagnostics": document["diagnostics"]}


def test_old_value_and_missing_indicator_layout_is_refused():
    recipe, document = diagnostic()
    document["diagnostics"][0]["coefficients"] = [
        {"kind": kind, "column": column, "coefficient": 0.0}
        for kind in ("value", "missing_indicator") for column in recipe["features"]]
    refuses(recipe, document)


def test_swapped_spline_entries_are_refused():
    recipe, document = diagnostic()
    entries = document["diagnostics"][0]["coefficients"]
    start = len(recipe["features"])
    width = recipe["preprocessing"]["spline"]["n_knots"] + 1
    entries[start], entries[start + width] = entries[start + width], entries[start]
    refuses(recipe, document)
