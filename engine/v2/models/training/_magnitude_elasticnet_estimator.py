"""Construct a fresh maintained pipeline; fitting and guards stay in the adapter."""
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

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
    return make_pipeline(FeatureUnion([
        ("imputer", _imputer_branch(recipe)),
        ("missing_indicator", MissingIndicator(**preprocessing["missing_indicator"])),
    ], n_jobs=1), StandardScaler(**preprocessing["scaler"]),
        ElasticNet(**recipe["estimator"]["params"]))
