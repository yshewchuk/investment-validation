"""Fixed source-free histogram/point fit on caller-admitted numeric matrices.

No source admission, folds, holdout policy, registration or sealed receipt is
created here. A numeric result has one shared training-only tail, not one copy
per prediction. Bounded parent execution owns resource limits and heartbeats.
"""
from __future__ import annotations

import re
from contextlib import redirect_stdout

import numpy as np
from threadpoolctl import threadpool_limits

from engine.models.no_fit import forbid_fitting as forbid_legacy_fitting
from engine.v2.data.errors import DataError
from engine.v2.models.no_fit import forbid_fitting
from engine.v2.ops.errors import fail
from engine.v2.research.magnitude_distribution import bin_index
from engine.v2.research.progress import Progress, stage
from experiments.distribution_candidate_recipe import validate_candidate_recipe
from experiments.distribution_class_support import (
    class_histogram, early_stopping_decision, training_support)

__all__ = ["fit_candidate"]

_ITERATION_LINE = re.compile(r"\s*\[(\d+)/(\d+)\]")


class _FitProgress:
    """Tick a Progress by the completed delta from sklearn verbose iteration lines."""

    def __init__(self, progress, max_iter):
        self.progress, self.max_iter = progress, max_iter
        self._buffer, self._seen = "", 0

    def write(self, text):
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            match = _ITERATION_LINE.match(line)
            if match is not None:
                done, total = (int(match.group(index)) for index in (1, 2))
                if total == self.max_iter and 1 <= done <= self.max_iter and done > self._seen:
                    self.progress.tick(done - self._seen)
                    self._seen = done
        return len(text)

    def flush(self):
        pass


def _invalid():
    raise fail("INVALID_EXPERIMENT_SPEC", "invalid distribution fit inputs") from None


def _failed():
    raise fail("EXPERIMENT_VARIANT_FAILED", "distribution numeric fit failed") from None


def _matrices(train_features, train_labels, prediction_features, width):
    try:
        if any(np.iscomplexobj(value) for value in (train_features, train_labels, prediction_features)):
            _invalid()
        train, labels, prediction = (np.array(value, dtype=float, copy=True)
            for value in (train_features, train_labels, prediction_features))
    except (TypeError, ValueError, OverflowError):
        _invalid()
    for matrix in (train, prediction):
        if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] != width
                or np.isinf(matrix).any()):
            _invalid()
    if labels.shape != (len(train),) or not np.isfinite(labels).all() or (labels < 0).any():
        _invalid()
    return train, labels, prediction


def _outputs(value, shape):
    try:
        if np.iscomplexobj(value):
            _failed()
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError):
        _failed()
    if array.shape != shape or not np.isfinite(array).all() or (array < 0).any():
        _failed()
    return array


def _classifier_diagnostics(model, support, max_iter, early_stopping_used):
    classes = np.asarray(model.classes_)
    if classes.ndim != 1 or classes.dtype.kind not in "iu":
        _failed()
    labels = classes.tolist()
    if sorted(labels) != support["observed_class_labels"]:
        _failed()
    iterations, trees = model.n_iter_, model.n_trees_per_iteration_
    for value in (iterations, trees):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
            _failed()
    if trees != support["trees_per_iteration"]:
        _failed()
    if early_stopping_used:
        if not 1 <= iterations <= max_iter:
            _failed()
    elif iterations != max_iter:
        _failed()
    return {"class_labels": labels, "iterations": int(iterations),
            "trees_per_iteration": int(trees), "total_trees": int(iterations * trees)}


def fit_candidate(recipe, train_features, train_labels, prediction_features, *, columns) -> dict:
    """Fit one raw probability/point pair, without retries or source authority.

    Both no-fit guards precede validation and lazy estimator imports. Features
    accept float-convertible matrices with native partial NaNs, never infinity;
    labels require one finite nonnegative magnitude per training row. All-null
    training columns become zero in BOTH matrices before either estimator.

    The fixed classifier uses the declared recipe parameters. The shared
    early-stopping decision records requested/used/fallback reason; a requested
    regularised split that cannot be stratified is recorded as unused and a
    taken fallback is logged before fitting. Its observed classes, tree
    multiplicity and iterations are checked; the public point fitter exposes
    predictions only, so its iteration count is labelled declared.
    Numeric ValueError/FloatingPointError failures become typed candidate
    failures; resources, native no-fit guards and programming exceptions escape.
    """
    forbid_legacy_fitting("experiments.distribution_candidate_fit.fit_candidate")
    forbid_fitting("experiments.distribution_candidate_fit.fit_candidate")
    recipe = validate_candidate_recipe(recipe)
    expected_columns = recipe["point_recipe"]["features"]
    if not isinstance(columns, (list, tuple)) or list(columns) != expected_columns:
        _invalid()
    train, labels, prediction = _matrices(
        train_features, train_labels, prediction_features, len(expected_columns))
    bins = np.array([bin_index(float(value)) for value in labels], dtype=int)
    support = training_support(np.bincount(bins, minlength=81).tolist())
    regularised = recipe.get("classifier_variant") == "regularised_v1"
    empty = np.isnan(train).all(axis=0)
    train[:, empty] = 0.0
    prediction[:, empty] = 0.0

    from sklearn.ensemble import HistGradientBoostingClassifier

    from engine.v2.models.training.magnitude_ridge_hgb import fit_ridge_hgb_predictions

    params = recipe["classifier"]["params"]
    decision = early_stopping_decision(recipe, support)
    requested, used, fallback = (decision[key] for key in ("requested", "used", "fallback_reason"))
    estimator_params = {**params, "max_features": float(params["max_features"]), "verbose": 2}
    if regularised:
        estimator_params["early_stopping"] = used
    fit_stage = "distribution classifier fit/predict"
    if fallback is not None:
        fit_stage += " fallback " + fallback
    fit_progress = Progress(fit_stage, params["max_iter"])
    try:
        model = HistGradientBoostingClassifier(**estimator_params)
        with stage(fit_stage), threadpool_limits(limits=1):
            with redirect_stdout(_FitProgress(fit_progress, params["max_iter"])):
                model.fit(train, bins)
            classifier = _classifier_diagnostics(model, support, params["max_iter"], used)
            if regularised:
                classifier.update(early_stopping_requested=bool(requested),
                    early_stopping_used=bool(used), fallback_reason=fallback)
            probabilities = _outputs(model.predict_proba(prediction),
                (len(prediction), len(classifier["class_labels"])))
        with stage("distribution matched point fit"):
            point = fit_ridge_hgb_predictions(recipe["point_recipe"], train, labels, prediction,
                                               columns=expected_columns)
    except (ValueError, FloatingPointError):
        _failed()
    if type(point) is not dict or "predictions" not in point:
        _failed()
    point_predictions = _outputs(point["predictions"], (len(prediction),)).tolist()
    tail_excesses = (labels[labels >= 20] - 20).tolist() or None
    rows, shared_tail = [], None
    progress = Progress("distribution probability mapping", len(prediction))
    with stage("distribution probability mapping"):
        for index, row in enumerate(probabilities):
            try:
                mapped = class_histogram(classifier["class_labels"], row.tolist(),
                    tail_excesses=tail_excesses if index == 0 else None)
            except DataError as error:
                if error.code != "CONTRACT_MISMATCH":
                    raise
                _failed()
            rows.append(mapped["probabilities"])
            if index == 0:
                shared_tail = mapped["tail"]
            progress.tick()
    return {"schema_version": "distribution_candidate_fit.v1", "recipe_id": recipe["recipe_id"],
        "probabilities": rows, "tail": shared_tail, "point_predictions": point_predictions,
        "diagnostics": {"training_support": support, "classifier": classifier,
            "point": {"iterations_declared": recipe["point_recipe"]["estimator"]["params"]["max_iter"],
                      "trees_per_iteration_declared": 1},
            "all_null_training_columns": [name for name, missing in zip(expected_columns, empty) if missing],
            "training_rows": len(train), "prediction_rows": len(prediction)}}
