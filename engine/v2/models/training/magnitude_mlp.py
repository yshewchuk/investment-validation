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


def fit_mlp_predictions(recipe, train_features, train_labels, prediction_features, *, columns):
    """Caller owns causal folds/budgets; preprocessing and stopping use training only."""
    check_fitting_allowed("engine.v2.models.training.magnitude_mlp.fit_mlp_predictions")
    recipe = validate_mlp_recipe(recipe)
    if not isinstance(columns, (list, tuple)) or list(columns) != recipe["features"]:
        raise ValueError("MLP matrix columns must exactly match the ordered recipe subset")
    train, labels, prediction = _matrices(train_features, train_labels, prediction_features,
                                         len(columns))
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
        if any(issubclass(item.category, ConvergenceWarning) for item in caught):
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
        "converged": True, "n_iter": int(network.n_iter_),
        "input_features": int(network.n_features_in_),
        "loss": float(network.loss_), "loss_scale": "standardized_target"}
