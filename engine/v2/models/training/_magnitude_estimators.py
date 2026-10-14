"""Construct fresh maintained estimators from the closed magnitude recipe."""
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler


def build_estimator(recipe):
    """Construct only: fitting and both no-fit guards belong to magnitude.py."""
    if recipe["model"] == "history_hgb":
        return HistGradientBoostingRegressor(**recipe["estimator"]["params"])
    preprocessing = recipe["preprocessing"]
    return make_pipeline(FeatureUnion([
        ("imputer", SimpleImputer(**preprocessing["imputer"])),
        ("missing_indicator", MissingIndicator(**preprocessing["missing_indicator"])),
    ], n_jobs=1), StandardScaler(**preprocessing["scaler"]),
        Ridge(**recipe["estimator"]["params"]))
