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

    train_x = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    train_y = np.array([0, 0, 1, 1])
    clean_test = np.array([[0.0, 0.2], [0.5, 0.2]])
    poisoned_test = np.array([[1e12, -1e12], [-1e12, 1e12]])
    estimator = LogisticRegression(random_state=0)
    rule = TrainFoldRule(top_fraction=0.5)
    baseline = fit_walk_forward_fold(estimator, train_x, train_y, clean_test, rule)
    poisoned = fit_walk_forward_fold(estimator, train_x, train_y, poisoned_test, rule)
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
