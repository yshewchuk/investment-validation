"""Independent hash/matrix oracles for shared source-free horizon preparation."""
import copy
import hashlib
import json

import numpy as np
import pytest

from engine.v2.research.horizon_fit_inputs import prepare_horizon_inputs

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


@pytest.mark.parametrize("target", [None, True, [], "", " ", "columns", "row_keys", "values"])
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
