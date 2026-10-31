"""Assemble sealed per-horizon fit inputs from admitted stored history only."""
import math
import time

from engine.models.no_fit import RuntimeFitForbidden as LegacyRuntimeFitForbidden
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.training.magnitude import fit_magnitude_predictions
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.horizon_fit_inputs import prepare_horizon_inputs


def fitted_forecasts(indexed, train, fold, plan, execution, *, progress=None, strict_failures=False, diagnostics=None):
    """Freeze fresh fits and input digests before the caller opens evaluation labels."""
    predictions, evidence = {}, []
    train_ids, test_ids = train.event_id.tolist(), fold["evaluation_event_ids"]
    labels = train.absolute_move_pct.to_numpy(dtype=float)
    for horizon in plan["horizons"]:
        matrix, test_matrix, fit = prepare_horizon_inputs(indexed, train_ids, test_ids,
            horizon, execution["features"], labels, target_column="absolute_move_pct")
        if progress is not None:
            progress("started", fit)
        started = time.monotonic()
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
            elapsed = max(0.0, time.monotonic() - started)
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
            progress("completed", {**fit, "fit_seconds": elapsed})
    return predictions, evidence
