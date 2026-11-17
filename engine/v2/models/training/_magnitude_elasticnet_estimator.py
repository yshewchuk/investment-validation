"""Construct a fresh maintained pipeline; fitting and guards stay in the adapter."""
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from ._magnitude_spline_step import spline_step

__all__ = ["ConvergenceWarning", "build_estimator"]


def _imputer_branch(recipe):
    preprocessing = recipe["preprocessing"]
    if "spline" not in preprocessing:
        return SimpleImputer(**preprocessing["imputer"])
    spline = preprocessing["spline"]
    return make_pipeline(SimpleImputer(**preprocessing["imputer"]),
                         spline_step(recipe["features"], spline["columns"], spline["n_knots"]))


def build_estimator(recipe):
    preprocessing = recipe["preprocessing"]
    pipeline = make_pipeline(FeatureUnion([
        ("imputer", _imputer_branch(recipe)),
        ("missing_indicator", MissingIndicator(**preprocessing["missing_indicator"])),
    ], n_jobs=1), StandardScaler(**preprocessing["scaler"]),
        ElasticNet(**recipe["estimator"]["params"]))
    if recipe["target_transform"] != "none":
        return TransformedTargetRegressor(regressor=pipeline,
            transformer=FunctionTransformer(np.log1p, np.expm1, check_inverse=False))
    return pipeline
