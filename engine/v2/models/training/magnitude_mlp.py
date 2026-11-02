"""One guarded standalone neural fit; no source admission, selection or I/O."""
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from engine.v2.foundation import content_hash
from engine.v2.models.magnitude_mlp_recipe import validate_mlp_recipe

from ._magnitude_mlp_estimator import ConvergenceWarning, build_estimator
from .fit_routes import check_fitting_allowed
from .magnitude_ridge_hgb import _matrices

__all__ = ["fit_mlp_predictions"]


def fit_mlp_predictions(recipe, train_features, train_labels, prediction_features, *, columns,
                        event_dates=None):
    """Caller owns causal folds/budgets; preprocessing and stopping use training only."""
    check_fitting_allowed("engine.v2.models.training.magnitude_mlp.fit_mlp_predictions")
    recipe = validate_mlp_recipe(recipe)
    if not isinstance(columns, (list, tuple)) or list(columns) != recipe["features"]:
        raise ValueError("MLP matrix columns must exactly match the ordered recipe subset")
    train, labels, prediction = _matrices(train_features, train_labels, prediction_features,
                                         len(columns))
    if recipe["profile"].startswith("chrono"):
        with threadpool_limits(limits=1):
            return _fit_chrono(recipe, train, labels, prediction, event_dates)
    return _fit_standard(recipe, train, labels, prediction)


def _fit_standard(recipe, train, labels, prediction):
    model = build_estimator(recipe)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.filterwarnings("error", message=r"^Training interrupted by user\.$",
            category=UserWarning, module=r"^sklearn\.neural_network\._multilayer_perceptron$")
        try:
            model.fit(train, labels)
        except UserWarning as error:
            if (str(error) == "Training interrupted by user."
                    and isinstance(error.__context__, KeyboardInterrupt)):
                raise error.__context__ from None
            raise
        budget_complete = (
            recipe["nonconvergence"], model.regressor_[-1].n_iter_
        ) == ("epoch_budget_success_no_retry", recipe["estimator"]["params"]["max_iter"])
        convergence_warning = any(
            issubclass(item.category, ConvergenceWarning) for item in caught)
        if (convergence_warning, budget_complete) == (True, False):
            raise ValueError("MLP convergence failed; no retry permitted")
        network = model.regressor_[-1]
        if (not np.isfinite(network.loss_)
                or any(not np.isfinite(array).all()
                       for array in (*network.coefs_, *network.intercepts_))):
            raise ValueError("MLP produced nonfinite fit evidence")
        predictions = np.asarray(model.predict(prediction), dtype=float)
    if predictions.shape != (len(prediction),) or not np.isfinite(predictions).all():
        raise ValueError("MLP predictions require one finite value per prediction row")
    return {"schema_version": "magnitude_mlp_fit.v1", "recipe_id": content_hash(recipe),
        "predictions": np.maximum(predictions, 0.0).tolist(), "component_fits": 1,
        "converged": not convergence_warning, "n_iter": int(network.n_iter_),
        "input_features": int(network.n_features_in_),
        "loss": float(network.loss_), "loss_scale": "standardized_target"}


def _chrono_days(event_dates, rows):
    if not isinstance(event_dates, (list, tuple)) or len(event_dates) != rows:
        raise ValueError("chrono MLP requires one event date per training row")
    days = []
    for value in event_dates:
        if isinstance(value, np.datetime64):
            if np.isnat(value) or np.datetime_data(value.dtype)[0] in ("Y", "M"):
                raise ValueError("chrono MLP requires valid event dates")
        elif type(value) is str:
            if len(value) < 10:
                raise ValueError("chrono MLP requires valid event dates")
        elif not hasattr(value, "toordinal"):
            raise ValueError("chrono MLP requires valid event dates")
        try:
            day = np.datetime64(value, "D")
        except (TypeError, ValueError):
            raise ValueError("chrono MLP requires valid event dates") from None
        if np.isnat(day):
            raise ValueError("chrono MLP requires valid event dates")
        days.append(int(day.astype("timedelta64[D]").astype(np.int64)))
    return days


def _chrono_split(days, fraction):
    holdout = int(np.ceil(len(days) * fraction))
    if holdout <= 0 or holdout >= len(days):
        raise ValueError("chrono MLP requires nonempty fit and holdout subsets")
    order = sorted(range(len(days)), key=lambda row: (days[row], row))
    return (sorted(order[:len(days) - holdout]),
            sorted(order[len(days) - holdout:]))


def _partial_fit_preserving_interrupt(net, x_fit, y_fit):
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r"^Training interrupted by user\.$",
            category=UserWarning, module=r"^sklearn\.neural_network\._multilayer_perceptron$")
        try:
            net.partial_fit(x_fit, y_fit)
        except UserWarning as error:
            if (str(error) == "Training interrupted by user."
                    and isinstance(error.__context__, KeyboardInterrupt)):
                raise error.__context__ from None
            raise


def _fit_chrono(recipe, train, labels, prediction, event_dates):
    """Epoch-wise partial_fit with a date split; restore best holdout weights."""
    params = recipe["estimator"]["params"]
    fit_idx, val_idx = _chrono_split(
        _chrono_days(event_dates, len(train)), float(params["validation_fraction"]))
    model = build_estimator(recipe)
    prep, net = model.regressor[:-1], model.regressor[-1]
    x_fit = prep.fit_transform(train[fit_idx])
    x_val = prep.transform(train[val_idx])
    labels = np.asarray(labels, dtype=float)
    target = model.transformer
    y_fit = target.fit_transform(labels[fit_idx].reshape(-1, 1)).ravel()
    y_val = target.transform(labels[val_idx].reshape(-1, 1)).ravel()
    patience, budget = int(params["n_iter_no_change"]), int(params["max_iter"])
    best_loss, best_weights, stale = np.inf, None, 0
    for epoch in range(1, budget + 1):
        _partial_fit_preserving_interrupt(net, x_fit, y_fit)
        loss = float(np.mean((net.predict(x_val) - y_val) ** 2))
        if not np.isfinite(loss):
            raise ValueError("MLP produced nonfinite fit evidence")
        if loss < best_loss:
            best_loss, stale = loss, 0
            best_weights = ([array.copy() for array in net.coefs_],
                            [array.copy() for array in net.intercepts_])
        else:
            stale += 1
            if stale >= patience:
                break
    net.coefs_, net.intercepts_ = best_weights
    restored = float(np.mean((net.predict(x_val) - y_val) ** 2))
    if not np.isfinite(restored) or not all(np.isfinite(array).all()
               for array in (*net.coefs_, *net.intercepts_)):
        raise ValueError("MLP produced nonfinite fit evidence")
    predictions = np.asarray(target.inverse_transform(
        net.predict(prep.transform(prediction)).reshape(-1, 1))).ravel()
    if predictions.shape != (len(prediction),) or not np.isfinite(predictions).all():
        raise ValueError("MLP predictions require one finite value per prediction row")
    return {"schema_version": "magnitude_mlp_fit.v1", "recipe_id": content_hash(recipe),
        "predictions": np.maximum(predictions, 0.0).tolist(), "component_fits": 1,
        "converged": epoch < budget, "n_iter": int(epoch),
        "input_features": int(net.n_features_in_),
        "loss": restored, "loss_scale": "standardized_target"}
