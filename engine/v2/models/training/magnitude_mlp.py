"""One guarded standalone neural fit; no source admission, selection or I/O."""
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_mlp_recipe import validate_mlp_recipe

from ._magnitude_mlp_estimator import ConvergenceWarning, build_estimator
from .fit_inputs import FitInputs, check_fitting_allowed
from .magnitude_ridge_hgb import _matrices

__all__ = ["fit_mlp_predictions"]


def fit_mlp_predictions(recipe, train_features, train_labels, prediction_features, *, columns,
                        train_event_dates=None):
    """Caller owns causal folds/budgets; preprocessing and stopping use training only."""
    check_fitting_allowed("engine.v2.models.training.magnitude_mlp.fit_mlp_predictions")
    recipe = validate_mlp_recipe(recipe)
    inputs = FitInputs.from_recipe(
        recipe, train_features, train_labels, prediction_features, columns,
        train_event_dates=train_event_dates,
        column_error="MLP matrix columns must exactly match the ordered recipe subset",
        guard_path="engine.v2.models.training.magnitude_mlp.fit_mlp_predictions")
    return _fit(recipe, inputs)


def _fit(recipe, inputs):
    train, labels, prediction = _matrices(
        inputs, len(inputs.columns),
        "magnitude labels require one finite nonnegative value per training row")
    chrono = recipe["profile"].startswith("chrono")
    if chrono and inputs.train_event_dates is None:
        raise ValueError("chronological MLP profile requires training event dates")
    model = build_estimator(recipe)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.filterwarnings("error", message=r"^Training interrupted by user\.$",
            category=UserWarning, module=r"^sklearn\.neural_network\._multilayer_perceptron$")
        if chrono:
            predictions, converged, n_iter, loss, input_features = _fit_chrono(
                model, recipe, train, labels, prediction, inputs.train_event_dates)
        else:
            try:
                model.fit(train, labels)
            except UserWarning as error:
                if (str(error) == "Training interrupted by user."
                        and isinstance(error.__context__, KeyboardInterrupt)):
                    raise error.__context__ from None
                raise
            budget_complete = (
                recipe["nonconvergence"], model.regressor_[-1].n_iter_
            ) == ("epoch_budget_success_no_retry",
                  recipe["estimator"]["params"]["max_iter"])
            convergence_warning = any(
                issubclass(item.category, ConvergenceWarning) for item in caught)
            if (convergence_warning, budget_complete) == (True, False):
                raise ValueError("MLP convergence failed; no retry permitted")
            network = model.regressor_[-1]
            if (not np.isfinite(network.loss_)
                    or any(not np.isfinite(array).all()
                           for array in (*network.coefs_, *network.intercepts_))):
                raise ValueError("MLP produced nonfinite fit evidence")
            predictions, converged, n_iter, loss, input_features = (
                np.asarray(model.predict(prediction), dtype=float),
                not convergence_warning, int(network.n_iter_),
                float(network.loss_), int(network.n_features_in_))
    if predictions.shape != (len(prediction),) or not np.isfinite(predictions).all():
        raise ValueError("MLP predictions require one finite value per prediction row")
    return {"schema_version": "magnitude_mlp_fit.v1", "recipe_id": content_hash(recipe),
        "predictions": np.maximum(predictions, 0.0).tolist(), "component_fits": 1,
        "converged": converged, "n_iter": n_iter,
        "input_features": input_features,
        "loss": loss, "loss_scale": "standardized_target"}


def _fit_chrono(model, recipe, train, labels, prediction, event_dates):
    fraction = 0.15 if recipe["profile"].startswith("chrono15_") else 0.10
    fit_rows, validation_rows = _chrono_split(event_dates, fraction)
    preprocessing, transformer, network = (
        model.regressor[:-1], model.transformer, model.regressor[-1])
    fit_features = preprocessing.fit_transform(train[fit_rows])
    fit_labels = transformer.fit_transform(labels[fit_rows].reshape(-1, 1)).ravel()
    validation_features = preprocessing.transform(train[validation_rows])
    validation_labels = transformer.transform(
        labels[validation_rows].reshape(-1, 1)).ravel()
    best_loss, best_predictions, best_coefs, best_intercepts, stale = (
        np.inf, None, None, None, 0)
    max_iter = recipe["estimator"]["params"]["max_iter"]
    patience = recipe["estimator"]["params"]["n_iter_no_change"]
    for epoch in range(1, max_iter + 1):
        _partial_fit_preserving_interrupt(network, fit_features, fit_labels)
        predicted = network.predict(validation_features)
        loss = float(np.mean((predicted - validation_labels) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_predictions = predicted.copy()
            best_coefs = [coef.copy() for coef in network.coefs_]
            best_intercepts = [intercept.copy() for intercept in network.intercepts_]
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_coefs is None:
        raise ValueError("MLP produced nonfinite fit evidence")
    network.coefs_, network.intercepts_ = best_coefs, best_intercepts
    restored_loss = float(np.mean((best_predictions - validation_labels) ** 2))
    network.loss_ = restored_loss
    scaled = network.predict(preprocessing.transform(prediction))
    predictions = transformer.inverse_transform(scaled.reshape(-1, 1)).ravel()
    if (not np.isfinite(restored_loss)
            or any(not np.isfinite(array).all()
                   for array in (*network.coefs_, *network.intercepts_))):
        raise ValueError("MLP produced nonfinite fit evidence")
    return (predictions, epoch < max_iter, int(epoch), restored_loss,
            int(network.n_features_in_))


def _partial_fit_preserving_interrupt(network, features, labels):
    try:
        network.partial_fit(features, labels)
    except UserWarning as error:
        if (str(error) == "Training interrupted by user."
                and isinstance(error.__context__, KeyboardInterrupt)):
            raise error.__context__ from None
        raise


def _chrono_split(dates, fraction):
    ordered = sorted(range(len(dates)), key=lambda index: (dates[index], index))
    holdout = int(len(ordered) * fraction)
    if holdout < len(ordered) * fraction:
        holdout += 1
    holdout = max(1, holdout)
    split = len(ordered) - holdout
    if split < 1:
        raise ValueError("MLP chronological holdout requires fit and validation rows")
    return ordered[:split], ordered[split:]
