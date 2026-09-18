"""Calibration-surface recipes and their fit through P5-4's artifact builders.

``engine.payoff.fit_payoff`` / ``fit_runup_payoff`` and
``engine.recalibrate.fit_recalibration`` are refit live inside every Scorer
(P5-1 ``NON_MODEL_STATE_ITEMS``). These recipes record what those fits
consume — membership (strategy, per-request fill alpha, ``exit_date <
before``, the finite/positive guards), target, drivers, minimum counts and
residual seed.

The two payoff recipes are fitted by the training job through P5-4's
``engine.v2.models.training.payoff`` builders, which call
``native_payoff``'s unchanged math, so a job fold's artifact is the one the
builder makes on the same rows and cutoff. The fold's members are exactly
the rows that fit keeps: the ``gt 0`` filters and the completeness mask
reproduce ``fit_payoff``/``fit_runup_payoff``'s ``ok`` mask. The
recalibration maps stay a receipt-only seam (``fit_owner`` P5-4): no frozen
recalibration builder exists yet.
"""
from __future__ import annotations

import pandas as pd

from engine.v2.models.payoff_artifact import serialize_payoff_artifact

from .payoff import build_payoff_line_artifact, build_payoff_surface_artifact
from .recipes import (
    CLOCK_ID,
    MISSING_DROP_INCOMPLETE,
    OWNER_P5_4,
    OWNER_TRAINING_JOB,
    EstimatorSpec,
    FoldScheme,
    LabelRule,
    RecipeKey,
    ResidualRule,
    RowFilter,
    TargetSpec,
    TrainingRecipe,
)

__all__ = ["PAYOFF_KINDS", "calibration_recipes", "fit_payoff_fold"]

#: P5-4's canonical bytes (``serialize_payoff_artifact``): the file's sha256
#: equals the artifact's own ``content_hash``, so a ``PayoffArtifactRef`` can
#: point straight at it.
PAYOFF_ARTIFACT_FILE = "payoff_artifact.json"

#: Estimator kinds the job fits through the P5-4 builders.
PAYOFF_KINDS = ("payoff_line", "payoff_surface")

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
            features=(driver, "spot_entry"),
            filters=(RowFilter("strategy", "eq", strategy), RowFilter("spot_entry", "gt", 0.0)),
            estimator=EstimatorSpec("payoff_line", {"degree": 1, "min_trades": _PAYOFF_MIN_TRADES,
                                                    "max_residuals": _PAYOFF_MAX_RESIDUALS,
                                                    "alpha": "per request"},
                                    (_PAYOFF_RESIDUAL_SEED,)),
            residuals=ResidualRule("payoff_fit_residuals"), refs=payoff_refs,
            fit_owner=OWNER_TRAINING_JOB,
        ))
    recipes.append(_calibration(
        RecipeKey("payoff_surface", "STR-RUNUP", "calibration"), "payoff_surface.STR-RUNUP.v1",
        dataset="trades table (Scorer.trades)",
        target=TargetSpec("exit_value", "exit_value / spot_entry on runup_payoff_design(im_t1, "
                          "100*log(spot_exit/strike))", signed=True),
        features=("im_t1", "spot_entry", "spot_exit", "strike"),
        filters=(RowFilter("strategy", "eq", "STR-RUNUP"), RowFilter("spot_entry", "gt", 0.0),
                 RowFilter("spot_exit", "gt", 0.0), RowFilter("strike", "gt", 0.0)),
        estimator=EstimatorSpec("payoff_surface", {"min_trades": _PAYOFF_MIN_TRADES,
                                                   "max_residuals": _PAYOFF_MAX_RESIDUALS,
                                                   "alpha": "per request"},
                                (_PAYOFF_RESIDUAL_SEED,)),
        residuals=ResidualRule("payoff_fit_residuals"),
        refs=("engine.payoff.fit_runup_payoff", "engine.score.Scorer.runup_payoff"),
        fit_owner=OWNER_TRAINING_JOB,
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


def _calibration(key, recipe_id, *, dataset, target, features, filters, estimator, residuals, refs,
                 fit_owner=OWNER_P5_4):
    return TrainingRecipe(
        key=key, recipe_id=recipe_id, output_id=None, produces=None, clock_id=CLOCK_ID,
        dataset_source=dataset, key_columns=("event_id",), membership_time_column="exit_date",
        year_column=None, filters=filters, value_masks=(), target=target, features=tuple(features),
        missing_mask=MISSING_DROP_INCOMPLETE,
        # Legacy admits a row only when it had CLOSED before the cutoff:
        # exit_date < before, so the label is observable strictly before it.
        label=LabelRule("exit_date", "trade exit close; exit_date < request cutoff", 0),
        folds=FoldScheme("request_cutoff", 0),
        estimator=estimator, residuals=residuals, fit_owner=fit_owner, legacy_refs=refs,
        notes=("Fitted by the job through engine.v2.models.training.payoff (P5-4's frozen "
               "artifact builder); " if fit_owner == OWNER_TRAINING_JOB else
               "Seam for P5-4: no frozen builder yet; this recipe supplies membership and label "
               "receipts per cutoff only; ")
              + "fill_alpha is a per-request parameter (legacy np.isclose on it).",
    )


def fit_payoff_fold(recipe: TrainingRecipe, rows: pd.DataFrame, *, alpha: float, before, out_dir):
    """Write the frozen payoff artifact for one cutoff into ``out_dir``.

    Returns False (and writes nothing) below ``min_trades``.

    ``rows`` are the fold's members in dataset order (order fixes which
    residuals the seeded subsample keeps, exactly as in the inline fit).
    """
    params = recipe.estimator.params
    driver = recipe.features[0]
    records = pd.DataFrame({
        "driver": rows[driver].to_numpy(), "spot_entry": rows["spot_entry"].to_numpy(),
        "exit_value": rows["exit_value"].to_numpy(),
        "exit_date": pd.to_datetime(rows["exit_date"]).dt.strftime("%Y-%m-%d").to_numpy(),
    })
    common = dict(alpha=float(alpha), before=before, min_trades=int(params["min_trades"]),
                  max_residuals=int(params["max_residuals"]),
                  residual_seed=int(recipe.estimator.seeds[0]))
    if recipe.estimator.kind == "payoff_line":
        artifact = build_payoff_line_artifact(records.to_dict("records"),
                                              strategy=recipe.key.strategy, driver=driver, **common)
    else:
        records["spot_exit"] = rows["spot_exit"].to_numpy()
        records["strike"] = rows["strike"].to_numpy()
        artifact = build_payoff_surface_artifact(records.to_dict("records"),
                                                 strategy=recipe.key.strategy, **common)
    if artifact is None:
        return False
    (out_dir / PAYOFF_ARTIFACT_FILE).write_bytes(serialize_payoff_artifact(artifact))
    return True
