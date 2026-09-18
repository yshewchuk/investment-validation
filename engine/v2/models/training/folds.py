"""Recipe membership and fold planning, reproducing legacy's rules exactly.

Legacy is the spec (``guides/rearchitecture_phase5_models.md`` "Parity rules"):

* ``expanding_year`` is ``engine.models.training.common.walk_forward``: rows
  with a non-finite feature or target are dropped up front; test years are
  the sorted years present in the complete rows, from ``first_test_year``;
  fold *Y* trains on ``year < Y`` and is skipped when that pool is under
  ``min_train_rows`` or the year has no rows. The full refit
  (``fit_final``) trains on every complete row.
* ``monthly_cutoff`` is ``engine.data.features.tier4.build_producer``:
  ``scorable`` rows have complete features, ``trainable`` rows also a finite
  target; one fold per month present in ``scorable`` from ``FIRST_FOLD``,
  training on ``trainable`` rows dated strictly before the month start and
  predicting the month's scorable rows; skipped under ``MIN_TRAIN_ROWS``.
* ``request_cutoff`` (calibration) is one fold per scoring cutoff:
  ``exit_date < before``.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .recipes import FILTER_OPS, TrainingRecipe

__all__ = ["FoldPlan", "PreparedDataset", "RecipeDataError", "dataset_fingerprint",
           "plan_folds", "prepare_dataset"]


class RecipeDataError(ValueError):
    """The dataset cannot be read under this recipe (missing column, bad op)."""


@dataclass(frozen=True)
class PreparedDataset:
    """The recipe's candidate rows, masks applied, plus the completeness masks."""

    frame: pd.DataFrame
    features_complete: np.ndarray
    target_finite: np.ndarray
    n_source_rows: int

    @property
    def complete(self) -> np.ndarray:
        return self.features_complete & self.target_finite


@dataclass(frozen=True)
class FoldPlan:
    fold_id: str
    kind: str  # "walk_forward" | "full_refit" | "monthly" | "request_cutoff"
    cutoff: pd.Timestamp | None
    train: np.ndarray  # positional indices into PreparedDataset.frame
    test: np.ndarray
    skipped: str | None = None


def _numeric(frame: pd.DataFrame, columns) -> np.ndarray:
    return frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _apply_filter(frame: pd.DataFrame, rule) -> pd.Series:
    if rule.op not in FILTER_OPS:
        raise RecipeDataError(f"unknown filter op {rule.op!r}")
    if rule.column not in frame.columns:
        if rule.op == "not_before_column":
            # train_all._engine_trades applies it only "if 'decision_date' in rows.columns".
            return pd.Series(True, index=frame.index)
        raise RecipeDataError(f"filter column {rule.column!r} is missing")
    col = frame[rule.column]
    if rule.op == "eq":
        return col.astype(str) == str(rule.value)
    if rule.op == "isclose":
        return pd.Series(np.isclose(col.astype(float), float(rule.value)), index=frame.index)
    if rule.op == "ge":
        return pd.to_numeric(col, errors="coerce") >= rule.value
    if rule.op == "notna":
        return col.notna()
    if rule.op == "year_between":
        lo, hi = rule.value
        year = pd.to_numeric(col, errors="coerce")
        return (year >= lo) & (year <= hi)
    if rule.op == "not_before_column":
        before = pd.to_datetime(col, errors="coerce") < pd.to_datetime(frame[rule.value], errors="coerce")
        return ~before
    # group_nunique_ge: keep groups offering at least N distinct values.
    other, minimum = rule.value
    counts = frame.groupby(rule.column)[other].transform(lambda s: len(set(s)))
    return counts >= minimum


def prepare_dataset(recipe: TrainingRecipe, dataset: pd.DataFrame, *, extra_filters=()) -> PreparedDataset:
    """Filters, value masks and completeness, as legacy's prepare/train apply them."""
    needed = {*recipe.features, recipe.target.column, recipe.membership_time_column,
              *recipe.key_columns}
    if recipe.year_column:
        needed.add(recipe.year_column)
    missing = sorted(c for c in needed if c not in dataset.columns)
    if missing:
        raise RecipeDataError(f"{recipe.recipe_id}: dataset is missing {missing}")
    frame = dataset
    keep = pd.Series(True, index=frame.index)
    for rule in (*recipe.filters, *extra_filters):
        keep &= _apply_filter(frame, rule).fillna(False).astype(bool)
    frame = frame[keep.to_numpy()].copy()
    for mask in recipe.value_masks:
        if mask.column in frame.columns:
            values = pd.to_numeric(frame[mask.column], errors="coerce")
            frame[mask.column] = values.where((values >= mask.lo) & (values <= mask.hi))
    frame = frame.reset_index(drop=True)
    features_complete = np.isfinite(_numeric(frame, recipe.features)).all(axis=1)
    target_finite = np.isfinite(_numeric(frame, [recipe.target.column]).ravel())
    return PreparedDataset(frame, features_complete, target_finite, int(len(dataset)))


def dataset_fingerprint(recipe: TrainingRecipe, prepared: PreparedDataset) -> str:
    """sha256 over exactly the values a fit or a receipt can read."""
    cols = list(dict.fromkeys([*recipe.key_columns, recipe.membership_time_column,
                               *([recipe.year_column] if recipe.year_column else []),
                               *recipe.features, recipe.target.column,
                               *([recipe.label.time_column]
                                 if recipe.label.time_column in prepared.frame.columns else []),
                               *[c for dep in recipe.upstream
                                 for c in (dep.fold_column, dep.model_id_column)
                                 if c in prepared.frame.columns]]))
    digest = hashlib.sha256()
    digest.update("\x1f".join(cols).encode())
    digest.update(pd.util.hash_pandas_object(prepared.frame[cols], index=False).to_numpy().tobytes())
    return "sha256:" + digest.hexdigest()


def _times(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce")


def _expanding_year(recipe, prepared) -> tuple[FoldPlan, ...]:
    scheme, frame = recipe.folds, prepared.frame
    complete = np.flatnonzero(prepared.complete)
    years = pd.to_numeric(frame[recipe.year_column], errors="coerce").to_numpy()
    present = sorted(int(y) for y in pd.unique(years[complete]) if np.isfinite(y))
    if scheme.first_test_year is not None:
        present = [y for y in present if y >= scheme.first_test_year]
    plans = []
    for year in present:
        train = complete[years[complete] < year]
        test = complete[years[complete] == year]
        skipped = None
        if len(train) < scheme.min_train_rows or len(test) == 0:
            skipped = f"train pool {len(train)} under {scheme.min_train_rows}"
        plans.append(FoldPlan(f"wf-{year}", "walk_forward", pd.Timestamp(f"{year}-01-01"),
                              train, test, skipped))
    if scheme.full_refit:
        plans.append(FoldPlan("full-refit", "full_refit", None, complete, np.array([], dtype=np.int64)))
    return tuple(plans)


def _monthly(recipe, prepared) -> tuple[FoldPlan, ...]:
    scheme = recipe.folds
    times = _times(prepared.frame, recipe.membership_time_column)
    month = times.dt.to_period("M").dt.start_time
    scorable = np.flatnonzero(prepared.features_complete)
    trainable = np.flatnonzero(prepared.complete)
    first = pd.Timestamp(scheme.first_fold)
    folds = sorted({m for m in month.iloc[scorable] if pd.notna(m) and m >= first})
    month_values, time_values = month.to_numpy(), times.to_numpy()
    plans = []
    for fold in folds:
        stamp = np.datetime64(fold)
        train = trainable[time_values[trainable] < stamp]
        test = scorable[month_values[scorable] == stamp]
        skipped = (f"train pool {len(train)} under {scheme.min_train_rows}"
                   if len(train) < scheme.min_train_rows else None)
        plans.append(FoldPlan(f"m-{fold:%Y-%m}", "monthly", pd.Timestamp(fold), train, test, skipped))
    return tuple(plans)


def _request_cutoff(recipe, prepared, cutoff) -> tuple[FoldPlan, ...]:
    if cutoff is None:
        raise RecipeDataError(f"{recipe.recipe_id}: a request_cutoff recipe needs a cutoff")
    stamp = pd.Timestamp(cutoff).normalize()
    times = _times(prepared.frame, recipe.membership_time_column).to_numpy()
    complete = np.flatnonzero(prepared.complete)
    train = complete[times[complete] < np.datetime64(stamp)]
    return (FoldPlan(f"cut-{stamp:%Y-%m-%d}", "request_cutoff", stamp, train,
                     np.array([], dtype=np.int64)),)


def plan_folds(recipe: TrainingRecipe, prepared: PreparedDataset, *, cutoff=None) -> tuple[FoldPlan, ...]:
    """The recipe's folds over ``prepared``, in legacy order."""
    kind = recipe.folds.kind
    if kind == "expanding_year":
        return _expanding_year(recipe, prepared)
    if kind == "monthly_cutoff":
        return _monthly(recipe, prepared)
    if kind == "request_cutoff":
        return _request_cutoff(recipe, prepared, cutoff)
    raise RecipeDataError(f"unknown fold scheme {kind!r}")
