"""Fixed-magnitude compatibility policy over shared admitted horizon fitting."""
from engine.models.no_fit import RuntimeFitForbidden as LegacyRuntimeFitForbidden
from engine.v2.models.adapters import RuntimeFitForbidden
from engine.v2.ops.errors import OpsError, fail
from engine.v2.research.horizon_fitted import HorizonFitFailure
from engine.v2.research.horizon_fitted import fitted_forecasts as fit_horizons


def fitted_forecasts(indexed, train, fold, plan, execution, *, progress=None, strict_failures=False, diagnostics=None):
    """Keep legacy magnitude evidence, error policy and fixed nonnegative semantics."""
    route = "magnitude_candidate" if execution.get("schema_version") == "size_candidate_execution.v1" else "magnitude"
    try:
        return fit_horizons(indexed, train.event_id.tolist(), train.absolute_move_pct.to_numpy(dtype=float),
            fold["evaluation_event_ids"], plan["horizons"], execution["features"], target_column="absolute_move_pct",
            target_domain="nonnegative", route=route, recipe=execution["fit_recipe"],
            progress=progress, diagnostics=diagnostics)
    except HorizonFitFailure as failure:
        error = failure.__cause__
        if isinstance(error, (RuntimeFitForbidden, LegacyRuntimeFitForbidden)):
            raise error
        if strict_failures and (isinstance(error, (MemoryError, OSError, ImportError, OpsError))
                                or not isinstance(error, (ValueError, ArithmeticError))):
            raise error
        raise fail("EXPERIMENT_VARIANT_FAILED", "fitted magnitude estimator failed") from None
