"""Calibration-surface recipes: the P5-4 seam.

``engine.payoff.fit_payoff`` / ``fit_runup_payoff`` and
``engine.recalibrate.fit_recalibration`` are refit live inside every Scorer
(P5-1 ``NON_MODEL_STATE_ITEMS``). Their fit and frozen artifact belong to
P5-4; these recipes record only what those fits consume — membership
(strategy, per-request fill alpha, ``exit_date < before``), target, drivers,
minimum counts and residual seed — so the training job can write membership
and label receipts per cutoff (``plan_only=True``) and refuse to fit them.
"""
from __future__ import annotations

from .recipes import (
    CLOCK_ID,
    MISSING_DROP_INCOMPLETE,
    OWNER_P5_4,
    EstimatorSpec,
    FoldScheme,
    LabelRule,
    RecipeKey,
    ResidualRule,
    RowFilter,
    TargetSpec,
    TrainingRecipe,
)

__all__ = ["calibration_recipes"]

#: payoff.MIN_TRADES / MAX_RESIDUALS / RESIDUAL_SEED; recalibrate.MIN_PAIRS.
_PAYOFF_MIN_TRADES = 200
_PAYOFF_MAX_RESIDUALS = 5000
_PAYOFF_RESIDUAL_SEED = 20260829
_RECAL_MIN_PAIRS = 120


def calibration_recipes() -> list[TrainingRecipe]:
    recipes: list[TrainingRecipe] = []
    payoff_refs = ("engine.payoff.fit_payoff", "engine.score.Scorer.payoff")
    for strategy, driver in (("STR-THRU", "abs_move"), ("STR-RUNUP", "im_t1")):
        recipes.append(_calibration(
            RecipeKey("payoff_line", strategy, "calibration"), f"payoff_line.{strategy}.v1",
            dataset="trades table (Scorer.trades)",
            target=TargetSpec("exit_value", "exit_value / spot_entry, linear in the driver", signed=True),
            features=(driver,), filters=(RowFilter("strategy", "eq", strategy),),
            estimator=EstimatorSpec("payoff_line", {"degree": 1, "min_trades": _PAYOFF_MIN_TRADES,
                                                    "max_residuals": _PAYOFF_MAX_RESIDUALS,
                                                    "alpha": "per request"},
                                    (_PAYOFF_RESIDUAL_SEED,)),
            residuals=ResidualRule("payoff_fit_residuals"), refs=payoff_refs,
        ))
    recipes.append(_calibration(
        RecipeKey("payoff_surface", "STR-RUNUP", "calibration"), "payoff_surface.STR-RUNUP.v1",
        dataset="trades table (Scorer.trades)",
        target=TargetSpec("exit_value", "exit_value / spot_entry on runup_payoff_design(im_t1, "
                          "100*log(spot_exit/strike))", signed=True),
        features=("im_t1", "spot_entry", "spot_exit", "strike"),
        filters=(RowFilter("strategy", "eq", "STR-RUNUP"),),
        estimator=EstimatorSpec("payoff_surface", {"min_trades": _PAYOFF_MIN_TRADES, "alpha": "per request"},
                                (_PAYOFF_RESIDUAL_SEED,)),
        residuals=ResidualRule("payoff_fit_residuals"),
        refs=("engine.payoff.fit_runup_payoff", "engine.score.Scorer.runup_payoff"),
    ))
    for strategy in ("STR-THRU", "STR-RUNUP"):
        recipes.append(_calibration(
            RecipeKey("recalibration_map", strategy, "calibration"), f"recalibration_map.{strategy}.v1",
            dataset="data/features/recalibration_pairs.parquet (engine.recalibrate.load_pairs)",
            target=TargetSpec("outcome", "realized win indicator", signed=False),
            features=("raw_win",), filters=(RowFilter("strategy", "eq", strategy),),
            estimator=EstimatorSpec("isotonic", {"y_min": 0.0, "y_max": 1.0, "out_of_bounds": "clip",
                                                 "min_pairs": _RECAL_MIN_PAIRS, "alpha": "per request"}, ()),
            residuals=ResidualRule("none"),
            refs=("engine.recalibrate.fit_recalibration", "engine.score.Scorer.recalibration"),
        ))
    return recipes


def _calibration(key, recipe_id, *, dataset, target, features, filters, estimator, residuals, refs):
    return TrainingRecipe(
        key=key, recipe_id=recipe_id, output_id=None, produces=None, clock_id=CLOCK_ID,
        dataset_source=dataset, key_columns=("event_id",), membership_time_column="exit_date",
        year_column=None, filters=filters, value_masks=(), target=target, features=tuple(features),
        missing_mask=MISSING_DROP_INCOMPLETE,
        # Legacy admits a row only when it had CLOSED before the cutoff:
        # exit_date < before, so the label is observable strictly before it.
        label=LabelRule("exit_date", "trade exit close; exit_date < request cutoff", 0),
        folds=FoldScheme("request_cutoff", 0),
        estimator=estimator, residuals=residuals, fit_owner=OWNER_P5_4, legacy_refs=refs,
        notes="Seam for P5-4: the fit and its frozen artifact belong to the payoff/recalibration "
              "task; this recipe supplies membership and label receipts per cutoff only. "
              "fill_alpha is a per-request parameter (legacy np.isclose on it).",
    )
