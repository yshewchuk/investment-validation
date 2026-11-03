"""Independent hash/matrix oracles for shared source-free horizon preparation."""
import copy
import hashlib
import json

import numpy as np
import pytest

from engine.v2.models.training.fit_inputs import FitInputs
from engine.v2.research import horizon_fitted
from engine.v2.research.horizon_fit_inputs import fit_evidence, prepare_horizon_inputs
from engine.v2.research.horizon_fitted import HorizonInputs, fitted_forecasts

COLUMNS = ["signal", "missing"]


def indexed():
    return {(event, horizon): dict(event_id=event, context_id="synthetic-context",
        horizon_sessions=horizon, signal=value, missing=None, unused="must-not-enter")
        for horizon in (1, 3, 5) for event, value in (("train-b", 2.), ("train-a", 1.), ("test", 9.))}


def digest(document):
    # This fixture uses integral-valued numbers; canonical JSON encodes them as integers.
    return "sha256:" + hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare(source=None, *, target="crush_pct_iv30", labels=(-140., 25.), horizon=3):
    return prepare_horizon_inputs(source if source is not None else indexed(),
        ["train-b", "train-a"], ["test"], horizon, COLUMNS, labels, target_column=target)


@pytest.mark.parametrize("horizon", [1, 3, 5])
@pytest.mark.parametrize("generated", [(), ("training",), ("evaluation",), ("columns",),
                                      ("training", "evaluation", "columns")])
def test_signed_labels_order_and_null_evidence_match_independent_hashes(horizon, generated):
    inputs = {"training": ["train-b", "train-a"], "evaluation": ["test"], "columns": COLUMNS}
    for name in generated:
        inputs[name] = (value for value in inputs[name])
    matrix, prediction, evidence = prepare_horizon_inputs(indexed(), inputs["training"],
        inputs["evaluation"], horizon, inputs["columns"], [-140., 25.], target_column="crush_pct_iv30")
    training = dict(columns=COLUMNS, row_keys=[[event, "synthetic-context", horizon]
        for event in ("train-b", "train-a")], values=[[2, None], [1, None]], crush_pct_iv30=[-140, 25])
    test = dict(columns=COLUMNS, row_keys=[["test", "synthetic-context", horizon]], values=[[9, None]])
    assert evidence == dict(horizon_sessions=horizon, training_event_ids=["train-b", "train-a"],
        evaluation_event_ids=["test"], training_matrix_id=digest(training), prediction_matrix_id=digest(test))
    np.testing.assert_array_equal(matrix, [[2., np.nan], [1., np.nan]])
    np.testing.assert_array_equal(prediction, [[9., np.nan]])


def test_target_column_changes_only_training_identity_and_never_floors_labels():
    _, _, signed = prepare()
    _, _, other = prepare(target="different_signed_outcome")
    _, _, floored = prepare(labels=(-100., 25.))
    assert len({signed["training_matrix_id"], other["training_matrix_id"], floored["training_matrix_id"]}) == 3
    assert signed["prediction_matrix_id"] == other["prediction_matrix_id"] == floored["prediction_matrix_id"]


def test_estimator_mutation_cannot_change_source_or_frozen_evidence():
    source = indexed()
    original = copy.deepcopy(source)
    labels = np.array([-140., 25.])
    matrix, prediction, evidence = prepare(source, labels=labels)
    frozen = copy.deepcopy(evidence)
    matrix[:] = 999
    prediction[:] = -999
    labels[:] = 0
    assert source == original and evidence == frozen
    assert prepare(source)[2] == frozen


def test_evaluation_target_and_unselected_features_are_never_inspected():
    class Trap(dict):
        def __getitem__(self, key):
            if key in ("crush_pct_iv30", "absolute_move_pct", "unused"):
                pytest.fail("read an evaluation target or unselected field")
            return super().__getitem__(key)
    source = indexed()
    source["test", 3] = Trap(source["test", 3])
    assert prepare(source)[2] == prepare()[2]


@pytest.mark.parametrize("target", [None, True, [], "", " ", "columns", "row_keys", "values", "event_dates"])
def test_unsupported_target_column_refuses_before_index_access(target):
    class Trap(dict):
        def __getitem__(self, key):
            pytest.fail("indexed inputs before target-column check")
    with pytest.raises(ValueError, match="target column"):
        prepare(Trap(), target=target)


def test_missing_membership_is_not_silently_dropped():
    source = indexed()
    del source["train-a", 3]
    with pytest.raises(KeyError):
        prepare(source)


def test_invalid_target_refuses_before_consuming_iterables():
    def forbidden():
        pytest.fail("iterable consumed before target validation")
        yield
    with pytest.raises(ValueError, match="target column"):
        prepare_horizon_inputs({}, forbidden(), forbidden(), 3, forbidden(), [], target_column="values")


def test_prebuilt_inputs_refuse_opted_in_recipe_without_event_dates():
    inputs = FitInputs([[1.]], [0.], [[2.]], COLUMNS)
    evidence = {3: {"evaluation_event_ids": ["test"]}}
    prebuilt = HorizonInputs({3: inputs}, evidence, "crush_pct_iv30")
    with pytest.raises(ValueError, match="training event dates"):
        fitted_forecasts(prebuilt, ["test"], [3], target_domain="nonnegative",
            route="magnitude", recipe={"requires_event_dates": True})


def test_prebuilt_inputs_keep_snapshot_dates_when_caller_mutates_list(monkeypatch):
    dates = ["2024-01-02"]
    inputs = FitInputs([[1.]], [0.], [[2.]], COLUMNS, dates)
    dates[0] = "2024-01-03"
    assert inputs.train_event_dates == ("2024-01-02",)
    received = []
    def capture(route, recipe, fit_inputs):
        received.append(fit_inputs)
        return {"predictions": [1.0]}
    monkeypatch.setattr(horizon_fitted, "fit_execution", capture)
    evidence = {3: {"evaluation_event_ids": ["test"]}}
    prebuilt = HorizonInputs({3: inputs}, evidence, "crush_pct_iv30")
    fitted_forecasts(prebuilt, ["test"], [3], target_domain="nonnegative",
        route="magnitude", recipe={"requires_event_dates": True})
    assert received[0].train_event_dates == ("2024-01-02",)


def test_fit_inputs_accepts_ordered_training_event_dates():
    dates = ("2024-01-02", "2024-01-01")
    inputs = FitInputs([[1.], [2.]], [0., 1.], [[3.]], COLUMNS, dates)
    assert inputs.train_event_dates is dates


def test_fit_inputs_rejects_date_count_mismatch():
    with pytest.raises(ValueError, match="training event dates.*does not match"):
        FitInputs([[1.], [2.]], [0., 1.], [[3.]], COLUMNS, ("2024-01-02",))


@pytest.mark.parametrize("value", [20240102, "2024-02-30", "2024-1-2", "20240102"])
def test_fit_inputs_rejects_malformed_training_event_dates(value):
    with pytest.raises(ValueError, match="training event dates.*ISO"):
        FitInputs([[1.]], [0.], [[3.]], COLUMNS, (value,))


@pytest.mark.parametrize("dates, message", [
    (("2024-02-30", "2024-01-01"), "canonical ISO dates"),
    (("2024-1-2", "2024-01-01"), "canonical ISO dates"),
    ((20240102, "2024-01-01"), "ISO date strings"),
    (("2024-01-02",), "length does not match"),
])
def test_fit_evidence_rejects_invalid_training_event_dates(dates, message):
    training_keys = [["train-b", "synthetic-context", 3], ["train-a", "synthetic-context", 3]]
    prediction_keys = [["test", "synthetic-context", 3]]
    with pytest.raises(ValueError, match="training event dates.*" + message):
        fit_evidence(COLUMNS, [[2, None], [1, None]], [[9, None]], 3,
            ["train-b", "train-a"], ["test"], training_keys, prediction_keys,
            target_column="crush_pct_iv30", labels=[-140, 25], train_event_dates=dates)


def test_fit_evidence_adds_ordered_dates_only_when_provided():
    _, _, prepared = prepare()
    training_keys = [["train-b", "synthetic-context", 3], ["train-a", "synthetic-context", 3]]
    prediction_keys = [["test", "synthetic-context", 3]]
    training_values = [[2, None], [1, None]]
    prediction_values = [[9, None]]
    labels = [-140, 25]
    dates = ["2024-01-02", "2024-01-01"]
    dated = fit_evidence(COLUMNS, training_values, prediction_values, 3,
        ["train-b", "train-a"], ["test"], training_keys, prediction_keys,
        target_column="crush_pct_iv30", labels=labels, train_event_dates=dates)
    oracle = dict(columns=COLUMNS, row_keys=training_keys, values=training_values,
        crush_pct_iv30=labels, event_dates=dates)
    assert dated["training_matrix_id"] == digest(oracle)
    assert dated["prediction_matrix_id"] == prepared["prediction_matrix_id"]
    undated = fit_evidence(COLUMNS, training_values, prediction_values, 3,
        ["train-b", "train-a"], ["test"], training_keys, prediction_keys,
        target_column="crush_pct_iv30", labels=labels)
    assert undated == prepared
