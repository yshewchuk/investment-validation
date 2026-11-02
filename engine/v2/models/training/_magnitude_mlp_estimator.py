"""Construct maintained neural preprocessing; fitting/guards stay in the adapter."""
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["ConvergenceWarning", "build_estimator"]


def build_estimator(recipe):
    preprocessing = recipe["preprocessing"]
    pipeline = make_pipeline(FeatureUnion([
        ("imputer", SimpleImputer(**preprocessing["imputer"])),
        ("missing_indicator", MissingIndicator(**preprocessing["missing_indicator"])),
    ], n_jobs=1), VarianceThreshold(**preprocessing["variance_filter"]),
        StandardScaler(**preprocessing["scaler"]),
        MLPRegressor(**recipe["estimator"]["params"]))
    return TransformedTargetRegressor(regressor=pipeline,
        transformer=StandardScaler(with_mean=True, with_std=True))
