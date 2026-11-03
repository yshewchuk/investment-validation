"""Fixed-magnitude compatibility policy over shared admitted horizon fitting."""
from engine.models.no_fit import RuntimeFitForbidden as LegacyRuntimeFitForbidden
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.models.training.fit_routes import check_fitting_allowed
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.horizon_fitted import HorizonFitFailure, build_fit_inputs
from engine.v2.research.horizon_fitted import fitted_forecasts as fit_horizons


def fitted_forecasts(indexed, train, fold, plan, execution, *, progress=None, strict_failures=False, diagnostics=None,
                     fit_inputs=None):
    """Keep legacy magnitude evidence, error policy and fixed nonnegative semantics."""
    check_fitting_allowed("fitted_forecasts")
    route = "magnitude_candidate" if execution.get("schema_version") == "size_candidate_execution.v1" else "magnitude"
    try:
        if fit_inputs is None:
            fit_inputs = build_fit_inputs(indexed, train.event_id.tolist(), fold["evaluation_event_ids"],
                plan["horizons"], execution["features"], train.absolute_move_pct.to_numpy(dtype=float),
                target_column="absolute_move_pct")
        return fit_horizons(fit_inputs, fold["evaluation_event_ids"], plan["horizons"],
            target_domain="nonnegative", route=route, recipe=execution["fit_recipe"] if plan["horizons"] else None,
            progress=progress, diagnostics=diagnostics, measure_fit_seconds=True)
    except HorizonFitFailure as failure:
        error = failure.__cause__
        if isinstance(error, (RuntimeFitForbidden, LegacyRuntimeFitForbidden)):
            raise error
        if strict_failures and (isinstance(error, (MemoryError, OSError, ImportError, OpsError))
                                or not isinstance(error, (ValueError, ArithmeticError))):
            raise error
        raise fail("EXPERIMENT_VARIANT_FAILED", "fitted magnitude estimator failed") from None
