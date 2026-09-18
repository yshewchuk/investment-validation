"""The explicit, resumable multi-fold training job for one recipe.

This is the only place a P5-3 recipe is fitted. It is never reachable from
scoring: ``engine.v2.scoring``/``features``/``models`` sit on lower layers
and cannot import this package (``checks/import_layers.py``), legacy never
imports v2, and the job's first statement is the shared no-fit switch
(:mod:`.legacy_adapter`), so a call from inside any ``no_fit_guard()`` block
raises before it reads a row or writes a byte.

Layout under ``out_dir``::

    job.json                      recipe + dataset identity; resume refuses a mismatch
    folds/<fold_id>/              one directory per fold, published by an atomic rename
        membership_receipt.json
        label_availability_receipt.json
        estimator.joblib          (fitted folds only)
        predictions.parquet       (walk-forward/monthly folds: test keys, target, pred)
        COMPLETE.json             file hashes; written inside the directory before the rename
    summary.json                  written once every fold is complete

**Resume.** A fold whose directory exists is complete by construction (the
rename is the commit). On a rerun its receipts are re-derived from the data
and must equal the stored ones and its files must match ``COMPLETE.json``;
then it is kept, never refit. A leftover ``*.partial-*`` directory is an
interrupted fold and is discarded. A different recipe or dataset refuses
(``RESUME_MISMATCH``) rather than mixing artifacts from two identities.

**Refusal.** Before a fold is fitted its receipts are checked
(:func:`.receipts.receipt_issues`). Any issue raises :class:`TrainingRefused`
with the codes; folds already published stay published.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import fold_store as store
from .calibration import PAYOFF_KINDS, fit_payoff_fold
from .estimators import UnsupportedEstimator, fit_recipe_estimator, forbid_fitting
from .folds import dataset_fingerprint, plan_folds, prepare_dataset
from .receipts import fold_receipts, receipt_issues
from .recipes import OWNER_TRAINING_JOB, RowFilter, TrainingRecipe, recipe_fingerprint

__all__ = ["TRAINING_JOB_V1", "run_training_job"]

TRAINING_JOB_V1 = "p5.training_job.v1"


def _matrix(recipe, frame, rows) -> np.ndarray:
    return frame.iloc[rows][list(recipe.features)].apply(
        pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _fit_fold(recipe, prepared, fold, fit, partial) -> None:
    frame = prepared.frame
    y = pd.to_numeric(frame.iloc[fold.train][recipe.target.column], errors="coerce").to_numpy(dtype=float)
    model = fit(recipe, _matrix(recipe, frame, fold.train), y)
    store.dump_estimator(partial, model)
    if len(fold.test):
        pred = np.asarray(model.predict(_matrix(recipe, frame, fold.test)), dtype=float).ravel()
        cols = list(dict.fromkeys([*recipe.key_columns, recipe.membership_time_column]))
        out = frame.iloc[fold.test][cols].reset_index(drop=True)
        out["target"] = pd.to_numeric(frame.iloc[fold.test][recipe.target.column],
                                      errors="coerce").to_numpy()
        out["pred"] = pred
        out.to_parquet(partial / store.PREDICTIONS_FILE, index=False)


def _fit_payoff(recipe, prepared, fold, alpha, partial) -> str:
    written = fit_payoff_fold(recipe, prepared.frame.iloc[fold.train], alpha=alpha,
                              before=fold.cutoff.date().isoformat(), out_dir=partial)
    # Under min_trades nothing is written: legacy PayoffError / NO_PAYOFF_MAP.
    return "fitted" if written else "skipped"


def _summary(recipe, out_dir, outcomes) -> dict:
    summary = {"recipe_id": recipe.recipe_id, "recipe_fingerprint": recipe_fingerprint(recipe),
               "folds": {o.fold_id: o.status for o in outcomes}}
    if recipe.threshold is not None and recipe.threshold.kind == "oos_top_fraction_quantile":
        # gate.choose_threshold over every walk-forward OOS prediction.
        paths = [out_dir / "folds" / o.fold_id / store.PREDICTIONS_FILE for o in outcomes]
        preds = [pd.read_parquet(p)["pred"].to_numpy(dtype=float) for p in paths if p.is_file()]
        values = np.concatenate(preds) if preds else np.array([])
        values = values[np.isfinite(values)]
        summary["threshold"] = (float(np.quantile(values, 1.0 - recipe.threshold.top_fraction))
                                if values.size else None)
        summary["threshold_n"] = int(values.size)
    return summary


def _prepare(recipe, dataset, *, plan_only, cutoffs, alpha, extra_filters):
    """Owner/alpha checks, membership, dataset identity and the fold plan."""
    if recipe.fit_owner != OWNER_TRAINING_JOB and not plan_only:
        raise UnsupportedEstimator(
            f"{recipe.recipe_id} is fitted by {recipe.fit_owner}; run it with plan_only=True "
            "for its receipts")
    if recipe.estimator.kind in PAYOFF_KINDS and alpha is None:
        raise UnsupportedEstimator(f"{recipe.recipe_id} needs the request's fill alpha")
    if alpha is not None:
        extra_filters = (*extra_filters, RowFilter("fill_alpha", "isclose", float(alpha)))
    prepared = prepare_dataset(recipe, dataset, extra_filters=extra_filters)
    if recipe.folds.kind == "request_cutoff":
        folds = tuple(f for cut in cutoffs for f in plan_folds(recipe, prepared, cutoff=cut))
    else:
        folds = plan_folds(recipe, prepared)
    return prepared, dataset_fingerprint(recipe, prepared), folds


def _build_fold(recipe, prepared, fold, *, fit, plan_only, alpha, partial) -> str:
    if plan_only:
        return "planned"
    if fold.skipped:
        return "skipped"
    if recipe.estimator.kind in PAYOFF_KINDS:
        return _fit_payoff(recipe, prepared, fold, alpha, partial)
    _fit_fold(recipe, prepared, fold, fit, partial)
    return "fitted"


def run_training_job(recipe: TrainingRecipe, dataset: pd.DataFrame, out_dir, *,
                     fit=fit_recipe_estimator, plan_only: bool = False, cutoffs=(),
                     alpha: float | None = None, extra_filters=()) -> store.TrainingJobResult:
    """Run (or resume) every fold of ``recipe`` over ``dataset`` into ``out_dir``.

    ``plan_only`` writes both receipts per fold and fits nothing — the cheap
    first pass over real data. ``cutoffs``/``extra_filters`` serve the
    ``request_cutoff`` calibration recipes; ``alpha`` is their per-request
    fill alpha (a ``fill_alpha`` isclose filter, as legacy). The payoff
    recipes are fitted through P5-4's frozen-artifact builders; the
    recalibration maps stay receipt-only. ``fit(recipe, X, y)`` is
    injectable for tests; the default is the native estimator.
    """
    forbid_fitting("engine.v2.models.training.job.run_training_job")
    prepared, dataset_fp, folds = _prepare(recipe, dataset, plan_only=plan_only, cutoffs=cutoffs,
                                           alpha=alpha, extra_filters=extra_filters)
    identity = {"schema_version": TRAINING_JOB_V1, "recipe_id": recipe.recipe_id,
                "recipe_key": recipe.key.label(), "recipe_fingerprint": recipe_fingerprint(recipe),
                "dataset_fingerprint": dataset_fp, "plan_only": bool(plan_only),
                "alpha": alpha, "folds": [f.fold_id for f in folds]}
    out_dir = store.open_job(out_dir, identity)
    folds_dir = out_dir / "folds"

    outcomes: list[store.FoldOutcome] = []
    for fold in folds:
        membership, label = fold_receipts(recipe, prepared, fold, dataset_fp=dataset_fp)
        if (folds_dir / fold.fold_id).exists():
            store.verify_complete(folds_dir / fold.fold_id, membership, label)
            outcomes.append(store.FoldOutcome(fold.fold_id, "resumed", len(fold.train), len(fold.test)))
            continue
        issues = receipt_issues(recipe, membership, label)
        if issues:
            raise store.TrainingRefused(issues)
        partial = store.start_fold(folds_dir, fold.fold_id, membership, label)
        status = _build_fold(recipe, prepared, fold, fit=fit, plan_only=plan_only, alpha=alpha,
                             partial=partial)
        store.publish_fold(partial, fold.fold_id, status)
        outcomes.append(store.FoldOutcome(fold.fold_id, status, len(fold.train), len(fold.test)))

    store.write_json(out_dir / "summary.json", _summary(recipe, out_dir, outcomes))
    return store.TrainingJobResult(recipe.recipe_id, out_dir, tuple(outcomes))
