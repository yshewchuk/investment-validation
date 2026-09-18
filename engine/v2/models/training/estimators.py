"""Native estimators built from a recipe's :class:`EstimatorSpec`.

Each builder reproduces one legacy ``fit(X, y, seed)`` call for call — the
same sklearn classes, hyperparameters, seeds, fit order and prediction
arithmetic — so a fold fitted here predicts bit-identically to the legacy
function on the same matrix (``tests/test_v2_models_training_recipes.py``
proves it per kind). Target transforms are explicit wrappers rather than a
hidden detail of a legacy class:

* ``log1p_clip0`` — ``runup_move.LogTargetRegressor``: fit ``log1p(max(y,0))``,
  predict ``max(expm1(pred), 0)``.
* ``quantile_normal`` — EXP-169 ``fit_head``: a ``QuantileTransformer``
  (normal output) fitted on the target; predictions stay in that space and
  only rank candidates within an event.

Calibration kinds (``payoff_line``, ``payoff_surface``, ``isotonic``) are
P5-4's and are refused here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing

from .legacy_adapter import forbid_fitting
from .recipes import OWNER_TRAINING_JOB, TrainingRecipe

__all__ = ["EqualWeightBlend", "LogTargetModel", "SeedMeanEnsemble", "UnsupportedEstimator",
           "fit_recipe_estimator"]


class UnsupportedEstimator(ValueError):
    """The recipe's estimator is not fitted by this job (unknown kind, or P5-4's)."""


@dataclass
class EqualWeightBlend:
    """``common.BlendModel``: mean of the member predictions, axis 0."""

    models: tuple

    def predict(self, X) -> np.ndarray:
        preds = [np.asarray(m.predict(X), dtype=float).ravel() for m in self.models]
        return np.mean(preds, axis=0)


@dataclass
class LogTargetModel:
    """``runup_move.LogTargetRegressor``: public predictions in target units."""

    estimator: Any

    def predict(self, X) -> np.ndarray:
        transformed = np.asarray(self.estimator.predict(X), dtype=float)
        return np.maximum(np.expm1(transformed), 0.0)


class SeedMeanEnsemble:
    """``engine.models.ensemble.MeanEnsemble`` arithmetic: running sum / n."""

    def __init__(self, models: Sequence[Any], target_transformer=None):
        self.models = list(models)
        self.target_transformer = target_transformer

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        out = preds[0].copy()
        for p in preds[1:]:
            out = out + p
        return out / len(preds)


def _mlp(params: dict, seed: int):
    kwargs = dict(params)
    kwargs["hidden_layer_sizes"] = tuple(kwargs["hidden_layer_sizes"])
    return sklearn.pipeline.make_pipeline(
        sklearn.preprocessing.StandardScaler(),
        sklearn.neural_network.MLPRegressor(**kwargs, random_state=seed))


def _hgb(params: dict, seed: int):
    return sklearn.ensemble.HistGradientBoostingRegressor(**dict(params), random_state=seed)


def fit_recipe_estimator(recipe: TrainingRecipe, X: np.ndarray, y: np.ndarray):
    """Fit the recipe's estimator on ``(X, y)``. Guarded by the no-fit switch."""
    forbid_fitting("engine.v2.models.training.estimators.fit_recipe_estimator")
    if recipe.fit_owner != OWNER_TRAINING_JOB:
        raise UnsupportedEstimator(f"{recipe.recipe_id} is fitted by {recipe.fit_owner}, not this job")
    spec = recipe.estimator
    transform = recipe.target.transform
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if spec.kind == "ols_mlp_blend" and transform is None:
        (seed,) = spec.seeds
        ols = sklearn.linear_model.LinearRegression().fit(X, y)
        nn = _mlp(spec.params["mlp"], seed).fit(X, y)
        return EqualWeightBlend(models=(ols, nn))
    if spec.kind == "hgb" and transform is None:
        (seed,) = spec.seeds
        return _hgb(spec.params, seed).fit(X, y)
    if spec.kind == "hgb" and transform == "log1p_clip0":
        (seed,) = spec.seeds
        target = np.log1p(np.maximum(y, 0.0))
        return LogTargetModel(_hgb(spec.params, seed).fit(X, target))
    if spec.kind == "quantile_mlp_ensemble" and transform == "quantile_normal":
        quant = spec.params["target_quantile"]
        qt = sklearn.preprocessing.QuantileTransformer(
            output_distribution=quant["output_distribution"],
            n_quantiles=min(int(quant["n_quantiles_cap"]), len(y)))
        qt.fit(y.reshape(-1, 1))
        z = qt.transform(y.reshape(-1, 1)).ravel()
        models = [_mlp(spec.params["mlp"], seed).fit(X, z) for seed in spec.seeds]
        return SeedMeanEnsemble(models, target_transformer=qt)
    raise UnsupportedEstimator(f"{recipe.recipe_id}: no native builder for {spec.kind!r} / {transform!r}")
