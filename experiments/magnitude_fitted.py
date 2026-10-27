"""Assemble sealed per-horizon fit inputs from admitted stored history only."""
import math

import numpy as np

from engine.models.no_fit import RuntimeFitForbidden as LegacyRuntimeFitForbidden
from engine.v2.foundation import content_hash
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.training.magnitude import fit_magnitude_predictions
from engine.v2.ops.errors import OpsError, fail


def _matrix(indexed, event_ids, horizon, columns):
    """Keep nulls in canonical evidence while exposing float/NaN estimator arrays."""
    rows = [indexed[event, horizon] for event in event_ids]
    document = {"columns": columns, "row_keys": [
        [row[key] for key in ("event_id", "context_id", "horizon_sessions")] for row in rows],
        "values": [[row[column] for column in columns] for row in rows]}
    return document, np.asarray(document["values"], dtype=float)


def fitted_forecasts(indexed, train, fold, plan, execution, *, progress=None, strict_failures=False, diagnostics=None):
    """Freeze fresh fits and input digests before the caller opens evaluation labels."""
    predictions, evidence = {}, []
    train_ids, test_ids = train.event_id.tolist(), fold["evaluation_event_ids"]
    labels = train.absolute_move_pct.to_numpy(dtype=float)
    for horizon in plan["horizons"]:
        training, matrix = _matrix(indexed, train_ids, horizon, execution["features"])
        prediction, test_matrix = _matrix(indexed, test_ids, horizon, execution["features"])
        training["absolute_move_pct"] = labels.tolist()
        # Hash original ordered values before an estimator can mutate its input buffers.
        fit = {"horizon_sessions": horizon, "training_event_ids": list(train_ids),
            "evaluation_event_ids": list(test_ids), "training_matrix_id": content_hash(training),
            "prediction_matrix_id": content_hash(prediction)}
        if progress is not None:
            progress("started", fit)
        try:
            if execution.get("schema_version") == "size_candidate_execution.v1":
                from experiments.magnitude_candidate_fit import fit_candidate
                result = fit_candidate(execution["fit_recipe"], matrix, labels.copy(), test_matrix,
                                       columns=execution["features"])
                values = result["predictions"]
                if diagnostics is not None:
                    diagnostics.append({"horizon_sessions": horizon, **{k: v for k, v in result.items() if k != "predictions"}})
            else:
                values = fit_magnitude_predictions(execution["fit_recipe"], matrix, labels.copy(), test_matrix)
            if (len(values) != len(test_ids) or any(isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in values)):
                raise ValueError
        except (RuntimeFitForbidden, LegacyRuntimeFitForbidden):
            raise
        except (MemoryError, OSError, ImportError, OpsError):
            if strict_failures:
                raise
            raise fail("EXPERIMENT_VARIANT_FAILED", "fitted magnitude estimator failed") from None
        except Exception as error:
            if strict_failures and not isinstance(error, (ValueError, ArithmeticError)):
                raise
            raise fail("EXPERIMENT_VARIANT_FAILED", "fitted magnitude estimator failed") from None
        predictions.update({(event, horizon): float(value) for event, value in zip(test_ids, values)})
        evidence.append(fit)
        if progress is not None:
            progress("completed", fit)
    return predictions, evidence
