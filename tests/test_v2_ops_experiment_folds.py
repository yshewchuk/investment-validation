"""Leak-poison regression for the training-only walk-forward fold fit.

``poisoned_test`` carries future-only magnitudes belonging to the held-out
test fold; the clean and poisoned walk-forward matrices share the same training prefix. ``fit_walk_forward_fold`` consumes training rows alone, so the fitted
coefficients and the derived threshold must be identical with or without it, and
the caller's estimator must come back unmutated (the fit clones). Synthetic
in-memory arrays only -- no catalog, ledger, filesystem or runner.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from engine.v2.ops.experiment_folds import TrainFoldRule, fit_walk_forward_fold


def test_future_poison_in_test_fold_cannot_change_training_fit_or_threshold():
    from engine.v2.ops.errors import OpsError

    import pytest
    from sklearn.base import clone

    train_x = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    train_y = np.array([0, 0, 1, 1])
    clean_test = np.array([[0.0, 0.2], [0.5, 0.2]])
    poisoned_test = np.array([[1e12, -1e12], [-1e12, 1e12]])
    estimator = LogisticRegression(random_state=0)
    rule = TrainFoldRule(top_fraction=0.5)
    baseline = fit_walk_forward_fold(estimator, train_x, train_y, clean_test, rule)
    poisoned = fit_walk_forward_fold(estimator, train_x, train_y, poisoned_test, rule)
    expected_model = clone(estimator).fit(train_x, train_y)
    expected_scores = expected_model.predict_proba(train_x)[:, 1]
    expected_threshold = float(np.quantile(expected_scores, 0.5))
    np.testing.assert_allclose(baseline.threshold, expected_threshold)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(baseline.threshold, expected_threshold + 1.0)
    expected_test_scores = expected_model.predict_proba(clean_test)[:, 1]
    np.testing.assert_allclose(baseline.test_scores, expected_test_scores)
    corrupted_test_scores = expected_test_scores.copy()
    corrupted_test_scores[0] += 1.0
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(baseline.test_scores, corrupted_test_scores)
    np.testing.assert_array_equal(baseline.estimator.coef_, poisoned.estimator.coef_)
    np.testing.assert_array_equal(baseline.estimator.intercept_, poisoned.estimator.intercept_)
    assert baseline.threshold == poisoned.threshold
    assert not np.array_equal(baseline.test_scores, poisoned.test_scores)
    try:
        fit_walk_forward_fold(estimator, np.vstack((train_x, poisoned_test)), train_y, clean_test, rule)
    except OpsError as refusal:
        assert refusal.code == "INVALID_EXPERIMENT_SPEC"
    else:
        raise AssertionError("extra future rows were accepted as training features")
    assert not hasattr(estimator, "coef_")


def test_named_column_and_numeric_overflow_inputs_are_typed_refusals():
    import pandas as pd
    import pytest

    from engine.v2.ops.errors import OpsError

    def invalid(action):
        with pytest.raises(OpsError) as raised:
            action()
        assert raised.value.code == "INVALID_EXPERIMENT_SPEC"

    estimator = LogisticRegression(random_state=0)
    rule = TrainFoldRule()
    train_y = np.array([0, 0, 1, 1])
    train = pd.DataFrame([[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 1.0]], columns=["left", "right"])
    reversed_test = pd.DataFrame([[0.2, 0.1]], columns=["right", "left"])
    positional_test = np.array([[0.1, 0.2]])
    invalid(lambda: fit_walk_forward_fold(estimator, train, train_y, reversed_test, rule))
    invalid(lambda: fit_walk_forward_fold(estimator, train, train_y, positional_test, rule))

    huge = 10**400
    positional_train = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    invalid(lambda: fit_walk_forward_fold(estimator, [[huge], [0], [1], [2]], train_y, [[0]], rule))
    invalid(lambda: fit_walk_forward_fold(estimator, positional_train, train_y, [[huge]], rule))
    invalid(lambda: TrainFoldRule(top_fraction=huge).fit_threshold([0.1], [0]))
    invalid(lambda: rule.fit_threshold([huge], [0]))
    invalid(lambda: rule.fit_threshold([0.1], [huge]))


def test_invalid_fold_threshold_refuses_before_estimator_clone(monkeypatch):
    import pytest

    from engine.v2.ops.errors import OpsError

    def boom(*args, **kwargs):
        raise AssertionError("estimator must not be cloned when the fold spec is invalid")

    monkeypatch.setattr("sklearn.base.clone", boom)

    train_x = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    train_y = np.array([0, 0, 1, 1])
    test_x = np.array([[0.0, 0.2], [0.5, 0.2]])
    estimator = LogisticRegression(random_state=0)
    rule = TrainFoldRule(top_fraction=0.0)
    with pytest.raises(OpsError) as raised:
        fit_walk_forward_fold(estimator, train_x, train_y, test_x, rule)
    assert raised.value.code == "INVALID_EXPERIMENT_SPEC"
    assert not hasattr(estimator, "coef_")


def test_nonfinite_fold_quantile_is_a_typed_variant_failure():
    import pytest

    from engine.v2.ops.errors import OpsError

    with pytest.raises(OpsError) as raised:
        TrainFoldRule(top_fraction=0.5).fit_threshold([-1e308, 1e308], [0, 1])
    assert raised.value.code == "EXPERIMENT_VARIANT_FAILED"


def test_estimator_mutation_cannot_change_reused_fold_rows():
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train_x = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    train_y = np.array([0, 0, 1, 1])
    test_x = np.array([[0.0, 0.2], [0.5, 0.2]])
    train_snapshot = train_x.copy()
    test_snapshot = test_x.copy()
    estimator = Pipeline([("scale", StandardScaler(copy=False)),
                          ("model", LogisticRegression(random_state=0))])
    rule = TrainFoldRule(top_fraction=0.5)
    first_fold = fit_walk_forward_fold(estimator, train_x, train_y, test_x, rule)
    reference = clone(estimator).fit(train_snapshot.copy(), train_y)
    expected_scores = reference.predict_proba(train_snapshot.copy())[:, 1]
    assert first_fold.threshold == float(np.quantile(expected_scores, 0.5))
    assert np.asarray(first_fold.test_scores).shape == (test_x.shape[0],)
    np.testing.assert_array_equal(train_x, train_snapshot)
    np.testing.assert_array_equal(test_x, test_snapshot)
    second_fold = fit_walk_forward_fold(estimator, train_x, train_y, test_x, rule)
    assert second_fold.threshold == first_fold.threshold
    np.testing.assert_array_equal(train_x, train_snapshot)
    np.testing.assert_array_equal(test_x, test_snapshot)


def test_positive_scores_follow_classes_order_and_reject_malformed_classes():
    import math

    import pytest
    from sklearn.base import BaseEstimator

    from engine.v2.ops.errors import OpsError

    def class_one_probability(value):
        return 1.0 / (1.0 + math.exp(-2.0 * value))

    class _ClassesOrderEstimator(BaseEstimator):
        """Synthetic scorer whose ``predict_proba`` columns follow ``classes_``."""

        def __init__(self, class_order=(0, 1), has_classes=True):
            self.class_order = class_order
            self.has_classes = has_classes

        def fit(self, features, labels):
            if self.has_classes:
                self.classes_ = np.asarray(self.class_order)
            return self

        def predict_proba(self, features):
            positive = np.array([class_one_probability(float(row[0]))
                                 for row in np.asarray(features, dtype=float)])
            return np.column_stack([positive if int(label) == 1 else 1.0 - positive
                                    for label in self.classes_])

    train_values = (-2.0, -1.5, 1.0, 3.0)
    test_values = (-1.5, 0.5, 4.0)
    train_x = np.array(train_values).reshape(4, 1)
    test_x = np.array(test_values).reshape(3, 1)
    train_labels = [0, 0, 1, 1]
    rule = TrainFoldRule(top_fraction=0.5)
    expected_scores = np.array([class_one_probability(v) for v in test_values])
    train_positive = np.array([class_one_probability(v) for v in train_values])
    expected_threshold = float(np.quantile(train_positive, 0.5))

    reversed_caller = _ClassesOrderEstimator(class_order=np.array([1, 0], dtype=object))
    reversed_fold = fit_walk_forward_fold(reversed_caller, train_x, train_labels, test_x, rule)
    np.testing.assert_array_equal(reversed_fold.test_scores, expected_scores)
    assert reversed_fold.threshold == expected_threshold
    assert reversed_fold.estimator is not reversed_caller
    np.testing.assert_array_equal(reversed_fold.estimator.classes_, (1, 0))
    column_one = reversed_fold.estimator.predict_proba(test_x)[:, 1]
    np.testing.assert_array_equal(column_one, 1.0 - expected_scores)
    assert not np.array_equal(reversed_fold.test_scores, column_one)

    forward_caller = _ClassesOrderEstimator(class_order=(0, 1))
    forward_fold = fit_walk_forward_fold(forward_caller, train_x, train_labels, test_x, rule)
    np.testing.assert_array_equal(forward_fold.test_scores, expected_scores)
    assert forward_fold.threshold == expected_threshold
    np.testing.assert_array_equal(forward_fold.test_scores, reversed_fold.test_scores)
    assert not hasattr(reversed_caller, "classes_")
    assert not hasattr(forward_caller, "classes_")

    for malformed in ((0, 2), (0, 0)):
        with pytest.raises(OpsError) as raised:
            fit_walk_forward_fold(_ClassesOrderEstimator(class_order=malformed),
                                  train_x, train_labels, test_x, rule)
        assert raised.value.code == "EXPERIMENT_VARIANT_FAILED"

    with pytest.raises(OpsError) as raised:
        fit_walk_forward_fold(_ClassesOrderEstimator(has_classes=False),
                              train_x, train_labels, test_x, rule)
    assert raised.value.code == "EXPERIMENT_VARIANT_FAILED"
