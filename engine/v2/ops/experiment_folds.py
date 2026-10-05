"""Pure fold-local experiment fitting primitives."""
import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

from engine.v2.ops.errors import OpsError, fail


@dataclass(frozen=True)
class TrainFoldRule:
    top_fraction: float = 0.5

    def fit_threshold(self, scores: Sequence[float], labels: Sequence[int]) -> float:
        import numpy as np

        fraction = self.top_fraction
        if isinstance(fraction, bool) or not isinstance(fraction, numbers.Real):
            raise fail("INVALID_EXPERIMENT_SPEC", "threshold fraction must be finite in (0, 1]")
        try:
            fraction = float(fraction)
        except (OverflowError, TypeError, ValueError):
            raise fail("INVALID_EXPERIMENT_SPEC", "threshold fraction must be finite in (0, 1]") from None
        if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
            raise fail("INVALID_EXPERIMENT_SPEC", "threshold fraction must be finite in (0, 1]")
        try:
            values = np.asarray(scores, dtype=float)
            targets = np.asarray(labels, dtype=float)
        except (OverflowError, TypeError, ValueError):
            raise fail("INVALID_EXPERIMENT_SPEC", "fold scores and labels must be numeric") from None
        if values.ndim != 1 or targets.ndim != 1 or not values.size or values.size != targets.size \
                or not np.all(np.isfinite(values)) or not np.all(np.isfinite(targets)) \
                or not np.all(np.isin(targets, (0.0, 1.0))):
            raise fail("INVALID_EXPERIMENT_SPEC", "fold scores and binary labels must be finite and aligned")
        threshold = float(np.quantile(values, 1.0 - fraction))
        if not math.isfinite(threshold):
            raise fail("EXPERIMENT_VARIANT_FAILED", "fold threshold calculation failed")
        return threshold


@dataclass(frozen=True)
class WalkForwardFoldFit:
    estimator: object
    threshold: float
    test_scores: object


def _training_arrays(train_features, train_labels):
    import numpy as np

    try:
        matrix = np.array(train_features, dtype=float, copy=True)
        labels = np.asarray(train_labels, dtype=float)
    except (OverflowError, TypeError, ValueError):
        raise fail("INVALID_EXPERIMENT_SPEC", "training fold must be numeric") from None
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1] \
            or labels.ndim != 1 or labels.size != matrix.shape[0] \
            or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(labels)) \
            or not np.all(np.isin(labels, (0.0, 1.0))):
        raise fail("INVALID_EXPERIMENT_SPEC", "training fold rows and binary labels must be finite and aligned")
    return matrix, labels.astype(int)


def _validate_feature_columns(train_features, test_features) -> None:
    try:
        train_columns = getattr(train_features, "columns", None)
        test_columns = getattr(test_features, "columns", None)
        if (train_columns is None) != (test_columns is None):
            raise fail("INVALID_EXPERIMENT_SPEC", "training and test features must both be named or positional")
        if train_columns is not None and tuple(train_columns) != tuple(test_columns):
            raise fail("INVALID_EXPERIMENT_SPEC", "training and test feature columns must match in order")
    except OpsError:
        raise
    except Exception:
        raise fail("INVALID_EXPERIMENT_SPEC", "feature column names are invalid") from None


def _feature_matrix(features):
    import numpy as np

    try:
        matrix = np.array(features, dtype=float, copy=True)
    except (OverflowError, TypeError, ValueError):
        raise fail("INVALID_EXPERIMENT_SPEC", "feature rows must be numeric") from None
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1] \
            or not np.all(np.isfinite(matrix)):
        raise fail("INVALID_EXPERIMENT_SPEC", "feature rows must be finite and nonempty")
    return matrix


def _positive_scores(estimator, matrix):
    import numpy as np

    try:
        classes = np.asarray(estimator.classes_)
        if classes.ndim != 1 or classes.shape[0] != 2 \
                or not np.array_equal(np.sort(classes), (0, 1)):
            raise ValueError("estimator classes are not binary 0 and 1")
        positive_index = int(np.flatnonzero(classes == 1)[0])
        probabilities = np.asarray(estimator.predict_proba(matrix), dtype=float)
        if probabilities.shape != (matrix.shape[0], 2) or not np.all(np.isfinite(probabilities)):
            raise ValueError("estimator returned invalid binary scores")
        return probabilities[:, positive_index]
    except OpsError:
        raise
    except Exception:
        raise fail("EXPERIMENT_VARIANT_FAILED", "fold score calculation failed") from None


def fit_walk_forward_fold(estimator, train_features, train_labels, test_features,
                          threshold_rule: TrainFoldRule) -> WalkForwardFoldFit:
    if not isinstance(threshold_rule, TrainFoldRule):
        raise fail("INVALID_EXPERIMENT_SPEC", "fold threshold rule has an unsupported type")
    fraction = threshold_rule.top_fraction
    if isinstance(fraction, bool) or not isinstance(fraction, numbers.Real):
        raise fail("INVALID_EXPERIMENT_SPEC", "threshold fraction must be finite in (0, 1]")
    try:
        fraction = float(fraction)
    except (OverflowError, TypeError, ValueError):
        raise fail("INVALID_EXPERIMENT_SPEC", "threshold fraction must be finite in (0, 1]") from None
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise fail("INVALID_EXPERIMENT_SPEC", "threshold fraction must be finite in (0, 1]")
    _validate_feature_columns(train_features, test_features)
    matrix, labels = _training_arrays(train_features, train_labels)
    test_matrix = _feature_matrix(test_features)
    if test_matrix.shape[1] != matrix.shape[1]:
        raise fail("INVALID_EXPERIMENT_SPEC", "training and test feature widths must match")
    if not callable(getattr(estimator, "fit", None)) \
            or not callable(getattr(estimator, "predict_proba", None)):
        raise fail("INVALID_EXPERIMENT_SPEC", "estimator must expose fit and predict_proba")
    try:
        from sklearn.base import clone

        fitted = clone(estimator)
        fitted.fit(matrix.copy(), labels)
        train_scores = _positive_scores(fitted, matrix)
        threshold = threshold_rule.fit_threshold(train_scores, labels)
        test_scores = _positive_scores(fitted, test_matrix)
    except Exception:
        raise fail("EXPERIMENT_VARIANT_FAILED", "walk-forward fold fitting failed") from None
    return WalkForwardFoldFit(fitted, threshold, test_scores)
