"""Construct a fresh maintained pipeline; fitting and guards stay in the adapter."""
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["ConvergenceWarning", "build_estimator"]


def build_estimator(recipe):
    preprocessing = recipe["preprocessing"]
    return make_pipeline(FeatureUnion([
        ("imputer", SimpleImputer(**preprocessing["imputer"])),
        ("missing_indicator", MissingIndicator(**preprocessing["missing_indicator"])),
    ], n_jobs=1), StandardScaler(**preprocessing["scaler"]),
        ElasticNet(**recipe["estimator"]["params"]))
