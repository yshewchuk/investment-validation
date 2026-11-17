"""Signed/nonnegative MLP fit entry; delegates the guarded fit to magnitude_mlp."""
from engine.v2.models.regression_mlp_recipe import validate_regression_mlp_recipe

from .fit_inputs import FitInputs, check_fitting_allowed
from .magnitude_mlp import _fit

__all__ = ["fit_regression_mlp_predictions"]


def fit_regression_mlp_predictions(recipe, train_features, train_labels, prediction_features, *,
                                   columns, train_event_dates=None):
    """Fit one explicit signed/nonnegative MLP; caller owns folds and source admission."""
    check_fitting_allowed(
        "engine.v2.models.training.regression_mlp.fit_regression_mlp_predictions")
    recipe = validate_regression_mlp_recipe(recipe)
    inputs = FitInputs.from_recipe(
        recipe, train_features, train_labels, prediction_features, columns,
        train_event_dates=train_event_dates,
        column_error="MLP matrix columns must exactly match the ordered recipe subset",
        guard_path="engine.v2.models.training.regression_mlp.fit_regression_mlp_predictions")
    return _fit(recipe, inputs, nonnegative=recipe["target_domain"] == "nonnegative",
                result_schema="regression_mlp_fit.v1")
