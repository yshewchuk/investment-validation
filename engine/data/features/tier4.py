"""Tier 4 — a model's forecast as a feature, made causal by folds.

Tier 3 is a deterministic function of Tier 2 and ``data_snapshot`` pins it, so a
model cannot live inside it: promoting a champion would silently invalidate the
provenance of every experiment that never used a forecast. Tier 4 is the layer
where a model's output becomes data, and it is deliberately **narrow** — keys,
outputs, provenance, and nothing else. Tier 3 stays the single authority for its
own columns; the join happens at read time
(``engine.features.load_panel(with_forecasts=True)``) so the dependency is
visible rather than something a reader has to remember.

**The leak rule here is different from Tier 3's, and so is its failure mode.**
Tier 3 enforces causality with *dates*: nothing in row ``k`` comes from event
``k`` onward. Tier 4 enforces it with *folds*: the forecast on a row whose event
falls in period ``P`` comes from a model fit only on events strictly before
``P.start``.

That distinction matters because the existing audit cannot see this one.
``assert_causal`` compares feature *stamps* against ``as_of``, so a leak inside a
model's **training set** is structurally invisible to it — a row built from the
final refit model would pass every check and be worthless. Hence ``fold_start``
is a stored column rather than a build-time detail: a row that cannot say which
model and which fold produced it cannot be checked, and a partially rebuilt
table must be distinguishable from a complete one.

**Cadence is monthly** (:data:`CADENCE`). Strict causality needs *train on
< D*; it does not need *retrain at every D*. Monthly forgoes at most one month
of new events — a few hundred out of a ~90,000-event training pool — on a model
whose OOS MAE moves by hundredths between folds. Per-date retraining costs 6-10
hours, and its real risk is not compute: it is that a rebuild that long does not
get run, and a table that drifts from its own definition costs more causality
than the extra recency ever bought.

Monthly also buys a property worth naming. The *current* month's model — fit on
everything before the month began — is the same artifact the live scorer needs
for an upcoming event. The historical Tier-4 row and the live board's forecast
for that month therefore come from the identical fitted model, agreeing by
construction rather than by a test that hopes they agree.
:func:`fit_fold` is the one entry point both sides use.

**Backfills cascade forward.** Correcting Tier 2 at date D changes the training
set of every fold from D onward, so every forecast from D onward is stale — not
only the rows whose own inputs moved. That is what "fit on strictly-before"
means, and it is why the build takes ``--since`` and is idempotent: rows before
the cut are carried over untouched, rows from the cut forward are recomputed.
The carry-over is guarded (:func:`_carried_prefix`) because a silently short
prefix is the one failure here that would not announce itself.

Usage::

    python3 -m engine.data.features.tier4                  # full rebuild
    python3 -m engine.data.features.tier4 --since 2024-01-01
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from engine import paths
from engine.data import store
from engine.models import registry as registry_mod
from engine.models.training import size_model
from engine.models.training.common import SEED

__all__ = [
    "CADENCE",
    "FIRST_FOLD",
    "MIN_TRAIN_ROWS",
    "COLUMNS",
    "KEY_COLUMNS",
    "PRODUCES",
    "FEATURE_MODELS",
    "column_group",
    "Tier4Error",
    "FeatureModel",
    "size_feature_model",
    "im_t1_feature_model",
    "iv_crush_feature_model",
    "feature_model",
    "fold_start_of",
    "fit_fold",
    "build_forecasts",
    "build_producer",
    "write_forecasts",
    "load_forecasts",
    "forecasts_digest",
]


class Tier4Error(RuntimeError):
    """The forecast table, or a request to build it, failed a causality check."""


#: The retrain period. A registered parameter rather than a constant, so
#: tightening it later is a spec change and not a rewrite. Changing it
#: invalidates the whole table — the fold boundaries stored in ``fold_start``
#: would no longer match — and :func:`build_forecasts` refuses ``--since``
#: against a table built on a different cadence rather than stitching two
#: incompatible halves together.
CADENCE = "monthly"

#: No fold starts before this. Matches the champion size model's walk-forward
#: OOS window (``first_test_year=2013``), so a Tier-4 value and the model's
#: published metrics describe the same regime.
FIRST_FOLD = pd.Timestamp("2013-01-01")

#: A fold with a thinner training pool than this is skipped and its rows carry
#: a NULL forecast. Same threshold as ``walk_forward``'s ``min_train_rows``.
MIN_TRAIN_ROWS = 500

#: The grain. One row per ``(ticker, event_date)`` — the same as Tier 3, and
#: total over it: every Tier-3 event gets a row, NULL where no forecast was
#: possible. Totality is deliberate. If Tier 4 held only the rows it could
#: predict, a *missing* row would be ambiguous between "no forecast available"
#: and "this table is stale", and only one of those is acceptable to a consumer.
KEY_COLUMNS = ("ticker", "event_date")

#: Every producer contributes the same seven columns, named off the one its
#: forecast lands in: the point, the band, the pool size behind the band, and
#: two provenance stamps.
#:
#: ``model_id`` and ``fold_start`` are per producer because they genuinely
#: differ inside a single build: two models on the same monthly cadence still
#: skip different folds — one has enough training rows in 2013 and the other
#: does not — and one column cannot name two models.
#:
#: ``tier3_snapshot`` is NOT per producer, and briefly was. Every producer is
#: built in the same pass from the same panel (see :func:`build_forecasts`), so
#: one hash answers for all of them; splitting it only described a partial
#: build, which is a state this module already refuses elsewhere. It stays
#: per ROW, which is a different thing and predates producers: a ``--since``
#: build stamps the rows it recomputed and leaves the carried prefix with the
#: hash it was built under, so a stitched table says so.
GROUP_SUFFIXES = ("", "_p10", "_p90", "_sd", "_resid_n", "_model_id", "_fold_start")


def column_group(produces: str) -> tuple[str, ...]:
    """The seven columns ``produces`` owns, in canonical order."""
    return tuple(f"{produces}{suffix}" for suffix in GROUP_SUFFIXES)


#: What Tier 4 materialises, in column order. Adding a name here and a factory
#: to :data:`FEATURE_MODELS` is the whole of what a new feature model must do to
#: reach the table: the build, the ``--since`` carry-over, the band, the
#: provenance and the serving path all key off these two.
#:
#: The horizon is IN THE NAME for ``pred_im_t1_d14``, not in a constant beside
#: it. The implied-move model is fit per decision day, so "the predicted T−1
#: implied move" is not one number — it is one per horizon, and a rebuild at a
#: different horizon writing the same column name would be exactly the silent
#: substitution ``fold_start`` exists to prevent. Adding T−7 later is then a new
#: column rather than a reinterpretation of this one.
PRODUCES = (
    "pred_abs_move",
    "pred_im_t1_d14",
    "pred_iv_crush_30",
)

COLUMNS = KEY_COLUMNS + tuple(
    column for produces in PRODUCES for column in column_group(produces)
) + ("tier3_snapshot",)

#: Columns whose NULL means "no forecast", never zero. A consumer given one of
#: these as NULL must decline to size a structure rather than sizing it at zero;
#: the TWIN-P entry filters already drop those rows, but that is a property to
#: state rather than to inherit by luck.
FORECAST_COLUMNS = tuple(
    column for produces in PRODUCES for column in column_group(produces)[:4]
)

#: Flat-pool floor: fewer held-out residuals than this and the row carries no
#: interval at all. An 80% band from 40 errors is not a distribution, and a
#: number that looks like a confidence interval is read as one.
MIN_RESIDUALS = 250


# --------------------------------------------------------------------------
# feature models
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureModel:
    """A model whose output is materialised as a Tier-4 column.

    ``fit`` and ``prepare`` come from the training module; ``model_id``,
    ``features``, ``target`` and ``seed`` come from the registry entry, so a
    champion promotion flows into the provenance columns instead of leaving
    rows labelled with a model that no longer exists.
    """

    model_id: str
    produces: str
    features: tuple[str, ...]
    target: str
    fit: Callable
    prepare: Callable
    seed: int = SEED
    #: Lower clip for BOTH interval bounds, or ``None`` for a signed target.
    #: See :func:`interval_for`: a hard-coded zero collapses every band on a
    #: target that can be negative, and does it without inverting one, so the
    #: check that catches an inverted interval would not notice.
    interval_floor: float | None = 0.0


def size_feature_model(registry=None) -> FeatureModel:
    """The champion ``size`` model, as a Tier-4 producer.

    The registry entry supplies identity; the training module supplies the fit.
    The registry's own artifact is deliberately **not** loaded: it is the final
    refit model, fit on everything, and a Tier-4 row built from it would be the
    exact leak this module exists to prevent.

    The feature lists are cross-checked because they can drift apart — a
    champion registered from a different feature list would otherwise be fit
    here on this module's list and labelled with the registry's id.
    """
    reg = registry_mod.load_registry() if registry is None else registry
    entry = reg.champion("size")
    if tuple(entry.features) != tuple(size_model.FEATURES):
        raise Tier4Error(
            f"champion {entry.id} was registered on a different feature list than "
            f"engine.models.training.size_model.FEATURES — refusing to fit one and "
            f"label it the other.\n"
            f"  registry: {list(entry.features)}\n"
            f"  module:   {list(size_model.FEATURES)}"
        )
    if entry.target != size_model.TARGET:
        raise Tier4Error(
            f"champion {entry.id} targets {entry.target!r}, module targets "
            f"{size_model.TARGET!r}"
        )
    return FeatureModel(
        model_id=entry.id,
        produces="pred_abs_move",
        features=tuple(entry.features),
        target=entry.target,
        fit=size_model.fit,
        prepare=size_model.prepare,
        seed=entry.seed or SEED,
    )


#: Trading days before the last pre-print close that ``pred_im_t1_d14`` is made
#: at. Fourteen because that is ``straddle_runup``'s default ``entry_offset``:
#: the forecast a stored column can serve is the one the trade is actually
#: decided on. It is also in :data:`implied_t1.DECISION_DAYS`, so the stored
#: column and the champion's published metrics describe the same horizon.
IM_T1_HORIZON = 14

#: Event window the ``implied_t1`` champion was trained over
#: (``train_all._events_with_session``). Kept identical here rather than widened
#: to Tier 3's full history: the extra years are reachable — ``daily_market``
#: starts 2007 — but a Tier-4 column fit on a decade the champion was never
#: validated on would be labelled with that champion's id and mean something
#: else. Widening it is an experiment, not a build flag.
IM_T1_YEARS = range(2017, 2027)


def im_t1_feature_model(registry=None) -> FeatureModel:
    """The champion ``implied_t1`` model, as a Tier-4 producer.

    Same contract as :func:`size_feature_model` and one structural difference
    that shapes the whole module: this model does **not** read the Tier-3 panel
    row for its event. Its features are the event-history block plus the daily
    market state at the DECISION date — fourteen trading days before the print
    — where the panel's market-state block is anchored at the last pre-print
    close. Reading the panel row would hand it the quantity it is predicting.

    So ``prepare`` here builds its own frame from Tier 2 rather than deriving
    columns from the panel, and it is why :class:`FeatureModel` carries a
    ``prepare`` callable at all instead of a list of column names.
    """
    # Deferred: `implied_t1` imports `engine.features`, which reads this table.
    # Importing it at module scope would make Tier 4 unimportable from inside
    # `engine.features` itself.
    from engine.models.training import implied_t1

    reg = registry_mod.load_registry() if registry is None else registry
    entry = reg.champion("implied_t1")
    if tuple(entry.features) != tuple(implied_t1.FEATURES):
        raise Tier4Error(
            f"champion {entry.id} was registered on a different feature list than "
            f"engine.models.training.implied_t1.FEATURES — refusing to fit one and "
            f"label it the other.\n"
            f"  registry: {list(entry.features)}\n"
            f"  module:   {list(implied_t1.FEATURES)}"
        )
    if entry.target != implied_t1.TARGET:
        raise Tier4Error(
            f"champion {entry.id} targets {entry.target!r}, module targets "
            f"{implied_t1.TARGET!r}"
        )

    def prepare(panel: pd.DataFrame) -> pd.DataFrame:
        from engine.features import DAILY_STATE_FIELDS

        events = store.read_table(
            "earnings_events",
            years=IM_T1_YEARS,
            columns=["event_id", "ticker", "event_date", "session"],
        )
        events = events[events["session"].notna()].reset_index(drop=True)
        # `build_dataset` reads `daily_market` twice when handed None — once for
        # the decision-date features and once for the target at the last
        # pre-print close. That table is ~9M rows, and the second read returns
        # the identical frame. Reading it here makes the build a minute rather
        # than several, and is why this closure exists at all.
        daily = store.read_table(
            "daily_market",
            years=range(min(IM_T1_YEARS) - 1, max(IM_T1_YEARS) + 1),
            columns=["ticker", "date", "src_iv", *DAILY_STATE_FIELDS.keys()],
        )
        frame = implied_t1.build_dataset(
            events, panel=panel, daily=daily, decision_days=(IM_T1_HORIZON,)
        )
        if frame.empty:
            return frame
        # Tier 4 is keyed on the EVENT, and `build_dataset` keys on
        # (event, decision day). One decision day makes those the same grain,
        # and `date` is the name every Tier-4 function joins on.
        return frame.rename(columns={"event_date": "date"})

    return FeatureModel(
        model_id=entry.id,
        produces=f"pred_im_t1_d{IM_T1_HORIZON}",
        features=tuple(entry.features),
        target=entry.target,
        fit=implied_t1.fit,
        prepare=prepare,
        seed=entry.seed or SEED,
    )


def iv_crush_feature_model(registry=None) -> FeatureModel:
    """The champion ``iv_crush`` model, as a Tier-4 producer.

    The one producer so far whose target is SIGNED — 30-day implied vol falls at
    83.2% of prints — so it declares ``interval_floor=None``. With the
    magnitude models' zero floor, ``p10`` and ``p90`` would both clip to zero on
    the large majority of rows, and ``[0, 0]`` is not an inverted interval, so
    the band check would pass while every band said nothing.

    Like ``implied_t1`` and unlike ``size``, its frame is not the panel: the
    target is read at the close AFTER the print, which Tier 3 does not carry and
    must not.
    """
    from engine.models.training import iv_crush

    reg = registry_mod.load_registry() if registry is None else registry
    entry = reg.champion("iv_crush")
    if tuple(entry.features) != tuple(iv_crush.FEATURES):
        raise Tier4Error(
            f"champion {entry.id} was registered on a different feature list than "
            f"engine.models.training.iv_crush.FEATURES — refusing to fit one and "
            f"label it the other.\n"
            f"  registry: {list(entry.features)}\n"
            f"  module:   {list(iv_crush.FEATURES)}"
        )
    if entry.target != iv_crush.TARGET:
        raise Tier4Error(
            f"champion {entry.id} targets {entry.target!r}, module targets "
            f"{iv_crush.TARGET!r}"
        )
    return FeatureModel(
        model_id=entry.id,
        produces="pred_iv_crush_30",
        features=tuple(entry.features),
        target=entry.target,
        fit=iv_crush.fit,
        prepare=iv_crush.prepare,
        seed=entry.seed or SEED,
        interval_floor=None,
    )


#: Column -> the factory that produces it. Parallel to :data:`PRODUCES`, and
#: checked against it at import so a name added to one and forgotten in the
#: other fails loudly here rather than as an all-NULL column in the table.
FEATURE_MODELS: dict[str, Callable[..., FeatureModel]] = {
    "pred_abs_move": size_feature_model,
    "pred_im_t1_d14": im_t1_feature_model,
    "pred_iv_crush_30": iv_crush_feature_model,
}

if tuple(FEATURE_MODELS) != PRODUCES:  # pragma: no cover - import-time guard
    raise Tier4Error(
        f"PRODUCES {PRODUCES} and FEATURE_MODELS {tuple(FEATURE_MODELS)} disagree"
    )


def feature_model(produces: str, registry=None) -> FeatureModel:
    """The producer for one column, built from the current champion."""
    try:
        factory = FEATURE_MODELS[produces]
    except KeyError:
        raise Tier4Error(
            f"{produces!r} is not a Tier-4 producer; known: {list(FEATURE_MODELS)}"
        ) from None
    model = factory(registry)
    if model.produces != produces:
        raise Tier4Error(
            f"factory for {produces!r} produces {model.produces!r}"
        )
    return model


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------


def fold_start_of(dates) -> pd.Series:
    """The first day of the period each date falls in — a row's fold key."""
    if CADENCE != "monthly":  # pragma: no cover - guarded by test_tier4
        raise Tier4Error(f"unsupported cadence {CADENCE!r}")
    values = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    return values.dt.to_period("M").dt.start_time.astype("datetime64[us]")


def _is_fold_start(values: pd.Series) -> pd.Series:
    """Whether each value is a legal fold boundary under the current cadence."""
    stamps = pd.to_datetime(values)
    return stamps.isna() | (fold_start_of(stamps).to_numpy() == stamps.to_numpy())


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def _matrix(frame: pd.DataFrame, columns) -> np.ndarray:
    return frame[list(columns)].to_numpy(dtype=float)


def _complete(frame: pd.DataFrame, columns) -> np.ndarray:
    return np.isfinite(_matrix(frame, columns)).all(axis=1)


def fit_fold(trainable: pd.DataFrame, model: FeatureModel, fold_start) -> object:
    """Fit ``model`` on every trainable event strictly before ``fold_start``.

    The one place a Tier-4 model is ever fit. The historical build calls it per
    fold and the live scorer calls it for the current month; sharing the
    function is what makes those two agree by construction rather than by
    coincidence.
    """
    cut = pd.Timestamp(fold_start)
    train = trainable[trainable["date"] < cut]
    if len(train) < MIN_TRAIN_ROWS:
        raise Tier4Error(
            f"fold {cut.date()}: only {len(train):,} trainable rows before it, "
            f"need {MIN_TRAIN_ROWS:,}"
        )
    return model.fit(
        _matrix(train, model.features),
        train[model.target].to_numpy(dtype=float),
        model.seed,
    )


#: Last prepared frame per model, keyed by ``(model_id, id(panel))``.
#:
#: ``prepare`` is a cheap column derivation for the size model and a 100k-event
#: Python loop over the trading calendar for ``implied_t1``, which reads its
#: features at a decision date the panel is not anchored on. ``training_frames``
#: is called once per build and again for every serving model the live board
#: constructs, so without this the board would pay minutes per score.
#:
#: One entry per model, holding the panel it was prepared from so identity is
#: compared against a live object rather than a recycled ``id()``. A panel
#: rebuild hands in a different object and misses.
_PREPARED: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}


def training_frames(panel: pd.DataFrame, model: FeatureModel):
    """``(scorable, trainable)`` — the rows that can be predicted, and learned from.

    These are not the same set, and conflating them is what would make Tier 4
    useless live. A prediction needs complete *features*; only training also
    needs a realized target. An event that has not printed yet has no
    ``abs_move`` and must still receive a forecast — that is the entire point of
    materialising one.
    """
    hit = _PREPARED.get(model.model_id)
    if hit is not None and hit[0] is panel:
        return hit[1].copy(), hit[2].copy()
    prepared = model.prepare(panel)
    scorable = prepared[_complete(prepared, model.features)].copy()
    target = pd.to_numeric(scorable[model.target], errors="coerce").to_numpy(dtype=float)
    trainable = scorable[np.isfinite(target)].copy()
    _PREPARED[model.model_id] = (panel, scorable, trainable)
    return scorable.copy(), trainable.copy()


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------


def _pool_stats(residuals: np.ndarray) -> tuple[float, float, float, int]:
    """``(q10, q90, sd, n)`` of a residual pool."""
    clean = residuals[np.isfinite(residuals)]
    if clean.size < MIN_RESIDUALS:
        return float("nan"), float("nan"), float("nan"), int(clean.size)
    return (
        float(np.quantile(clean, 0.10)),
        float(np.quantile(clean, 0.90)),
        float(clean.std(ddof=1)),
        int(clean.size),
    )


def _floored(values: np.ndarray, floor: float | None) -> np.ndarray:
    return values if floor is None else np.maximum(values, floor)


def interval_for(
    predictions: np.ndarray,
    pool_pred: np.ndarray,
    pool_res: np.ndarray,
    *,
    floor: float | None = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """80% band and residual SD for ``predictions``, from a held-out pool.

    Residuals are signed ``y_true − y_pred``, so a draw ADDS to a prediction to
    make a plausible realization — the convention ``WalkForwardResult.residuals``
    established and the scorer already relies on.

    Conditioned on where the prediction falls, via
    :func:`engine.models.registry.bucket_residuals` — the same decile machinery
    EXP-115 promoted, reused rather than reimplemented. A big forecast and a
    small one have differently-shaped errors, and one pool for both understates
    the band at the top of the range and overstates it at the bottom. A bucket
    thinner than its floor falls back to the flat pool, so conditioning can only
    refine the estimate and never leave a sparse region with a pool too thin to
    be a distribution.

    ``floor`` clips BOTH bounds. It defaults to zero because both producers so
    far predict a MAGNITUDE — an absolute move, an implied move — where an
    interval reaching below zero reports an outcome that cannot occur, and on
    real data the lower bound would cross often: the size champion's MAE is
    ~3.9pp against a median forecast near 5pp.

    Flooring only the lower bound was the first version and it was wrong: a
    prediction far enough below zero puts ``p90`` under the floor while ``p10``
    sits on it, and the band comes out INVERTED. Clipping both keeps
    ``p10 <= p90`` unconditionally, and a degenerate ``[0, 0]`` is a visible
    signal that the point estimate itself left the support rather than a
    plausible-looking interval hiding it. (No real forecast has: 0 of 85,618,
    minimum +0.39. The fixture's linear target can, which is how this surfaced.)

    **It is a parameter and not a constant because the floor is a claim about
    the TARGET, not about intervals.** A producer whose target is signed — an
    IV crush is negative at 83% of prints — would have every band on every row
    clipped to ``[0, 0]`` by a hard-coded zero, and the check that catches an
    inverted interval would pass, because ``[0, 0]`` is not inverted. Pass
    ``floor=None`` for such a target.
    """
    predictions = np.asarray(predictions, dtype=float)
    p10 = np.full(predictions.shape, np.nan)
    p90 = np.full(predictions.shape, np.nan)
    sd = np.full(predictions.shape, np.nan)
    n = np.full(predictions.shape, np.nan)

    flat = np.asarray(pool_res, dtype=float)
    flat_stats = _pool_stats(flat)
    if flat_stats[3] < MIN_RESIDUALS:
        return p10, p90, sd, n

    buckets = registry_mod.bucket_residuals(pool_pred, pool_res)
    if buckets is None:
        q10, q90, spread, count = flat_stats
        known = np.isfinite(predictions)
        p10[known] = _floored(predictions[known] + q10, floor)
        p90[known] = _floored(predictions[known] + q90, floor)
        sd[known] = spread
        n[known] = count
        return p10, p90, sd, n

    edges, pools = buckets["edges"], buckets["pools"]
    # Named for what it is. It was `floor`, which collided with this function's
    # own `floor` parameter the moment that existed and silently clipped every
    # band to the minimum pool SIZE — a number in the target's units by
    # coincidence and in nobody's units by intent.
    min_pool = int(buckets.get("min_pool", 0))
    # Ten buckets, so the stats are computed once each rather than per row.
    stats = [
        _pool_stats(pool) if pool.size >= min_pool else flat_stats for pool in pools
    ]
    index = np.clip(
        np.searchsorted(edges, predictions, side="right") - 1, 0, len(pools) - 1
    )
    for i, (q10, q90, spread, count) in enumerate(stats):
        rows = np.isfinite(predictions) & (index == i)
        if not rows.any():
            continue
        p10[rows] = _floored(predictions[rows] + q10, floor)
        p90[rows] = _floored(predictions[rows] + q90, floor)
        sd[rows] = spread
        n[rows] = count
    return p10, p90, sd, n


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


def _log(message: str) -> None:
    print(f"  [tier4] {message}", flush=True)


def _keys(panel: pd.DataFrame) -> pd.DataFrame:
    keys = panel[["ticker", "date"]].rename(columns={"date": "event_date"}).copy()
    keys["event_date"] = pd.to_datetime(keys["event_date"]).astype("datetime64[us]")
    dupes = int(keys.duplicated(["ticker", "event_date"]).sum())
    if dupes:
        raise Tier4Error(
            f"Tier 3 has {dupes:,} duplicate (ticker, event_date) rows — the Tier-4 "
            "grain is not well defined against it"
        )
    return keys.sort_values(["ticker", "event_date"]).reset_index(drop=True)


def _empty() -> pd.DataFrame:
    return normalize(pd.DataFrame({c: [] for c in COLUMNS}))


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical dtypes and row order, applied on both write and read.

    ``--since`` equivalence is asserted by comparing frames, so the comparison
    must not be able to fail on a parquet round-trip's dtype choices.
    """
    out = frame.reindex(columns=list(COLUMNS)).copy()
    # Nullable `string` rather than `object`: a parquet round-trip turns an
    # object column's ``None`` into ``NaN``, which is enough on its own to make
    # the --since equivalence assertion fail on a table that is in fact correct.
    out["ticker"] = out["ticker"].astype("string")
    out["event_date"] = pd.to_datetime(out["event_date"]).astype("datetime64[us]")
    for produces in PRODUCES:
        out = _cast_group(out, produces)
    out["tier3_snapshot"] = out["tier3_snapshot"].astype("string")
    return out.sort_values(["ticker", "event_date"]).reset_index(drop=True)


def _cast_group(out: pd.DataFrame, produces: str) -> pd.DataFrame:
    point, p10, p90, sd, resid_n, model_id, fold_start = column_group(produces)
    for column in (point, p10, p90, sd):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype(float)
    out[resid_n] = pd.to_numeric(out[resid_n], errors="coerce").astype("Int64")
    out[model_id] = out[model_id].astype("string")
    out[fold_start] = pd.to_datetime(out[fold_start]).astype("datetime64[us]")
    return out


def _carried_prefix(existing: pd.DataFrame, keys: pd.DataFrame, cut: pd.Timestamp, model):
    """The rows before ``cut`` that a ``--since`` build may reuse, or an error.

    Three ways a carry-over is unsafe, all of them silent if unchecked:

    * the existing table was built on different fold boundaries, so its rows and
      the new ones would answer to different definitions of causality;
    * it was produced by a different model than the one about to run, which
      would leave the table labelled with two model ids and comparable to
      neither;
    * Tier 3 gained events inside the retained prefix, so carrying it over would
      leave permanent holes in a table whose totality consumers rely on.
    """
    _, _, _, _, _, model_id_col, fold_col = column_group(model.produces)
    prefix_keys = keys[keys["event_date"] < cut]
    have = existing[existing["event_date"] < cut]

    illegal = int((~_is_fold_start(have[fold_col])).sum())
    if illegal:
        raise Tier4Error(
            f"{illegal:,} carried rows have a {fold_col} that is not a {CADENCE} "
            "boundary — the existing table was built on a different cadence. "
            "Rebuild in full."
        )

    ids = set(have[model_id_col].dropna().unique())
    if ids - {model.model_id}:
        raise Tier4Error(
            f"carried rows were produced by {sorted(ids)}, this build runs "
            f"{model.model_id!r} — a champion promotion invalidates Tier 4 in full "
            "(see guides/tier4_feature_models.md §6). Rebuild in full."
        )

    merged = prefix_keys.merge(
        have, on=["ticker", "event_date"], how="left", indicator=True
    )
    missing = int((merged["_merge"] == "left_only").sum())
    if missing:
        raise Tier4Error(
            f"Tier 3 has {missing:,} events before {cut.date()} that the existing "
            "Tier-4 table does not cover — carrying the prefix over would leave "
            "permanent holes. Rebuild in full, or move --since earlier."
        )
    stale = len(have) - len(prefix_keys)
    if stale > 0:
        _log(f"dropping {stale:,} carried row(s) whose Tier-3 event no longer exists")
    return merged.drop(columns=["_merge"])


def _seed_residuals(carried: pd.DataFrame, realized: pd.Series, produces: str):
    """``(predictions, residuals)`` from an already-built prefix, in fold order.

    A full build starts these empty and grows them; a ``--since`` build starts
    from the rows it carried over. Both reach the same pool at the first
    recomputed fold, which is what the equivalence test asserts.
    """
    empty = (np.empty(0, dtype=float), np.empty(0, dtype=float))
    if carried.empty:
        return empty
    scored = carried.dropna(subset=[produces])
    if scored.empty:
        return empty
    keys = pd.MultiIndex.from_arrays([scored["ticker"], scored["event_date"]])
    truth = realized.reindex(keys).to_numpy(dtype=float)
    made = scored[produces].to_numpy(dtype=float)
    usable = np.isfinite(truth) & np.isfinite(made)
    return made[usable], (truth - made)[usable]


def _empty_group(produces: str) -> pd.DataFrame:
    columns = list(KEY_COLUMNS) + list(column_group(produces))
    return _normalize_group(pd.DataFrame({c: [] for c in columns}), produces)


def _normalize_group(frame: pd.DataFrame, produces: str) -> pd.DataFrame:
    """``normalize`` for one producer's slice, so a carried prefix round-trips."""
    out = frame.reindex(columns=list(KEY_COLUMNS) + list(column_group(produces))).copy()
    out["ticker"] = out["ticker"].astype("string")
    out["event_date"] = pd.to_datetime(out["event_date"]).astype("datetime64[us]")
    out = _cast_group(out, produces)
    return out.sort_values(["ticker", "event_date"]).reset_index(drop=True)


def build_producer(
    panel: pd.DataFrame,
    model: FeatureModel,
    *,
    keys: pd.DataFrame | None = None,
    since=None,
    existing: pd.DataFrame | None = None,
    log: Callable[[str], None] = _log,
) -> pd.DataFrame:
    """One producer's column group, for every Tier-3 event.

    ``since`` is rounded **down to its fold boundary**, because a fold is the
    unit of recomputation: half a month cannot be rebuilt without fitting the
    model that month's other half already used. Rounding down recomputes a
    superset of what was asked for, which is identical to what was already
    there, so the equivalence in §9 of the design note holds either way.
    """
    started = time.time()
    point, p10c, p90c, sdc, resid_c, model_c, fold_c = column_group(model.produces)
    keys = _keys(panel) if keys is None else keys

    scorable, trainable = training_frames(panel, model)
    log(
        f"{model.produces}: {len(keys):,} Tier-3 events · {len(scorable):,} scorable "
        f"on {len(model.features)} features · {len(trainable):,} trainable"
    )

    scorable = scorable.copy()
    scorable["fold_start"] = fold_start_of(scorable["date"]).to_numpy()

    cut = None
    carried = _empty_group(model.produces)
    if since is not None:
        cut = pd.Timestamp(fold_start_of([pd.Timestamp(since)]).iloc[0])
        prior = (
            _empty_group(model.produces)
            if existing is None
            else _normalize_group(existing, model.produces)
        )
        carried = _carried_prefix(prior, keys, cut, model)
        log(
            f"{model.produces}: --since {pd.Timestamp(since).date()} → fold "
            f"{cut.date()}; carrying {len(carried):,} row(s)"
        )

    folds = sorted({f for f in scorable["fold_start"].unique() if pd.Timestamp(f) >= FIRST_FOLD})
    if cut is not None:
        folds = [f for f in folds if pd.Timestamp(f) >= cut]

    predictions = pd.Series(np.nan, index=scorable.index, dtype=float)
    fitted_for = pd.Series(pd.NaT, index=scorable.index, dtype="datetime64[us]")
    band = {c: pd.Series(np.nan, index=scorable.index, dtype=float)
            for c in ("p10", "p90", "sd", "n")}

    # The held-out residual pool, grown fold by fold. At fold F it holds the
    # errors of every EARLIER fold and nothing else, so the interval carries the
    # same causality as the point forecast — an 80% band computed from the
    # errors a model had not yet made would be exactly the leak `fold_start`
    # exists to prevent, wearing a different hat.
    #
    # A --since build seeds it from the carried prefix, whose stored forecasts
    # ARE the earlier folds' predictions. That is what keeps the incremental
    # path equivalent to a full one here too.
    realized = trainable.set_index(["ticker", "date"])[model.target]
    pool_pred, pool_res = _seed_residuals(carried, realized, point)

    skipped = 0
    for fold in folds:
        stamp = pd.Timestamp(fold)
        test = scorable.index[scorable["fold_start"] == fold]
        n_train = int((trainable["date"] < stamp).sum())
        if n_train < MIN_TRAIN_ROWS:
            skipped += 1
            continue
        estimator = fit_fold(trainable, model, stamp)
        made = np.asarray(
            estimator.predict(_matrix(scorable.loc[test], model.features)), dtype=float
        ).ravel()
        predictions.loc[test] = made
        fitted_for.loc[test] = stamp

        p10, p90, sd, n = interval_for(
            made, pool_pred, pool_res, floor=model.interval_floor
        )
        for key, values in (("p10", p10), ("p90", p90), ("sd", sd), ("n", n)):
            band[key].loc[test] = values

        # Only now does this fold join the pool, and only where the event has
        # actually printed — an unrealized event has no error to contribute.
        priced_keys = pd.MultiIndex.from_arrays(
            [scorable.loc[test, "ticker"], scorable.loc[test, "date"]]
        )
        truth = realized.reindex(priced_keys).to_numpy(dtype=float)
        usable = np.isfinite(truth) & np.isfinite(made)
        if usable.any():
            pool_pred = np.concatenate([pool_pred, made[usable]])
            pool_res = np.concatenate([pool_res, truth[usable] - made[usable]])

        log(
            f"{model.produces} fold {stamp.date()}: train {n_train:,} → "
            f"{len(test):,} forecast(s), residual pool {pool_res.size:,} "
            f"[{time.time() - started:.0f}s]"
        )
    if skipped:
        log(
            f"{model.produces}: {skipped} fold(s) skipped for a training pool "
            f"under {MIN_TRAIN_ROWS:,}"
        )

    built = pd.DataFrame(
        {
            "ticker": scorable["ticker"].to_numpy(),
            "event_date": pd.to_datetime(scorable["date"]).astype("datetime64[us]").to_numpy(),
            point: predictions.to_numpy(),
            p10c: band["p10"].to_numpy(),
            p90c: band["p90"].to_numpy(),
            sdc: band["sd"].to_numpy(),
            resid_c: band["n"].to_numpy(),
            fold_c: fitted_for.to_numpy(),
        }
    )
    # A row that got no forecast carries no model id: the provenance columns
    # describe a prediction, and there is none to describe.
    built[model_c] = np.where(built[fold_c].notna(), model.model_id, None)
    built = built[built[fold_c].notna() | built[point].notna()]

    scope = keys if cut is None else keys[keys["event_date"] >= cut]
    fresh = scope.merge(built, on=["ticker", "event_date"], how="left")
    out = _normalize_group(pd.concat([carried, fresh], ignore_index=True), model.produces)
    if len(out) != len(keys):
        raise Tier4Error(
            f"{model.produces}: built {len(out):,} rows for {len(keys):,} Tier-3 "
            "events — Tier 4 must be total over Tier 3"
        )
    have = int(out[point].notna().sum())
    log(
        f"{model.produces}: {len(out):,} rows · {have:,} with a forecast "
        f"({have / max(len(out), 1):.1%}) · {time.time() - started:.0f}s"
    )
    return out


def build_forecasts(
    panel: pd.DataFrame,
    *,
    produces: Sequence[str] | None = None,
    models: Mapping[str, FeatureModel] | None = None,
    since=None,
    existing: pd.DataFrame | None = None,
    tier3_snapshot: str | None = None,
    log: Callable[[str], None] = _log,
) -> pd.DataFrame:
    """The whole Tier-4 table: every producer's group, joined on the keys.

    **Every producer is built, from one panel, in one pass.** There is no
    option to refresh one column group and carry the others, and the absence is
    deliberate: this module already refuses a ``--since`` carry-over whose
    stored ``model_id`` is not the model about to run, on the grounds that a
    champion promotion invalidates Tier 4 in full. A flag that rebuilt one
    producer against another producer's older numbers would manufacture exactly
    that state, and a single ``tier3_snapshot`` could no longer answer for the
    row. A full rebuild is ~168 fits — minutes, not hours — which is what makes
    the strict version affordable.

    ``produces`` therefore narrows only WHICH MODELS ARE FIT for a caller that
    already knows the rest is unchanged (the tests, and nothing else); it never
    changes what a written table means. ``models`` supplies stand-in producers
    for the same reason.
    """
    keys = _keys(panel)
    snapshot = store.file_sha256(paths.PANEL) if tier3_snapshot is None else tier3_snapshot
    wanted = tuple(PRODUCES if produces is None else produces)
    unknown = [p for p in wanted if p not in FEATURE_MODELS]
    if unknown:
        raise Tier4Error(f"unknown producer(s) {unknown}; known: {list(FEATURE_MODELS)}")

    table = keys.copy()
    for name in PRODUCES:
        if name in wanted:
            model = (models or {}).get(name) or feature_model(name)
            group = build_producer(
                panel, model, keys=keys, since=since, existing=existing, log=log
            )
        elif existing is not None:
            group = _normalize_group(
                keys.merge(existing, on=list(KEY_COLUMNS), how="left"), name
            )
        else:
            group = _normalize_group(keys.copy(), name)
        table = table.merge(group, on=list(KEY_COLUMNS), how="left")

    # The snapshot is per ROW, and that predates producers. A --since build
    # recomputed everything from `cut` forward against TODAY's panel and left
    # the prefix as it was, so stamping the whole table would erase the
    # evidence of the stitch — which is the one thing this column is for.
    table["tier3_snapshot"] = snapshot
    if since is not None and existing is not None:
        cut = pd.Timestamp(fold_start_of([pd.Timestamp(since)]).iloc[0])
        carried = existing.loc[
            existing["event_date"] < cut, ["ticker", "event_date", "tier3_snapshot"]
        ]
        table = table.merge(carried, on=list(KEY_COLUMNS), how="left", suffixes=("", "_kept"))
        keep = table["tier3_snapshot_kept"].notna()
        table.loc[keep, "tier3_snapshot"] = table.loc[keep, "tier3_snapshot_kept"]
        table = table.drop(columns=["tier3_snapshot_kept"])

    out = normalize(table)
    if len(out) != len(keys):
        raise Tier4Error(
            f"built {len(out):,} rows for {len(keys):,} Tier-3 events — Tier 4 "
            "must be total over Tier 3"
        )
    return out


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


def write_forecasts(frame: pd.DataFrame, path: Path | None = None) -> Path:
    out = paths.assert_writable(paths.TIER4 if path is None else path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if store.HAVE_PARQUET:
        normalize(frame).to_parquet(out, engine="pyarrow", index=False, compression="snappy")
    else:  # pragma: no cover - fallback path
        out = out.with_suffix(".csv.gz")
        normalize(frame).to_csv(out, index=False, compression="gzip")
    return out


def _assert_current_schema(frame: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Refuse a table that does not carry every column :data:`COLUMNS` names.

    ``normalize`` reindexes onto :data:`COLUMNS`, so a stale file does not fail
    — it comes back with the missing columns silently NULL, which reads
    downstream as "no forecast for any event" and is indistinguishable from a
    legitimately unbuilt table. Loud is the only safe behaviour.

    Stated as "everything present" rather than as a list of known-old layouts,
    because an enumeration only catches the layouts someone thought of. The
    first version listed the single-producer column names and would have waved
    through the intermediate layout that existed between two commits of this
    very refactor.
    """
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise Tier4Error(
            f"{path} is missing {len(missing)} of the {len(COLUMNS)} columns Tier 4 "
            f"declares ({missing[:6]}{'...' if len(missing) > 6 else ''}) — it was "
            "written against an older layout. Rebuild it: "
            "`python3 -m engine.data.features.tier4`. There is no in-place "
            "migration on purpose: a table half in one layout and half in another "
            "is the state this module refuses everywhere else."
        )
    return frame


def load_forecasts(path: Path | None = None) -> pd.DataFrame | None:
    """The forecast table, or ``None`` if it has never been built."""
    path = paths.TIER4 if path is None else Path(path)
    if not path.exists():
        alt = path.with_suffix(".csv.gz")
        if not alt.exists():
            return None
        return normalize(_assert_current_schema(pd.read_csv(alt), alt))
    return normalize(_assert_current_schema(pd.read_parquet(path), path))


def forecasts_digest(path: Path | None = None) -> str | None:
    """The Tier-4 provenance hash — the second one the programme now carries.

    Deliberately **not** folded into ``manifest.snapshot_hash``. An experiment
    that reads forecasts pins this; one that does not pins Tier 3 alone, and
    must not be invalidated by a Tier-4 rebuild it never depended on.
    """
    path = paths.TIER4 if path is None else Path(path)
    if path.exists():
        return store.file_sha256(path)
    alt = path.with_suffix(".csv.gz")
    return store.file_sha256(alt) if alt.exists() else None


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_table(since=None, out: Path | None = None) -> dict:
    """Build and write Tier 4. The entry point ``engine.data.rebuild`` calls."""
    from engine.features import load_panel  # deferred: engine.features reads Tier 4

    panel = load_panel()
    target = paths.TIER4 if out is None else Path(out)
    existing = load_forecasts(target) if since is not None else None
    frame = build_forecasts(panel, since=since, existing=existing)
    written = write_forecasts(frame, target)
    return {
        "rows": int(len(frame)),
        "producers": {
            name: {
                "with_forecast": int(frame[name].notna().sum()),
                "model_ids": sorted(
                    frame[column_group(name)[5]].dropna().unique().tolist()
                ),
                "folds": int(frame[column_group(name)[6]].nunique()),
            }
            for name in PRODUCES
        },
        "cadence": CADENCE,
        "tier3_snapshot": sorted(frame["tier3_snapshot"].dropna().unique().tolist()),
        "path": str(written),
        "tier4": forecasts_digest(target),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--since",
        default=None,
        help="recompute from this date's fold forward; earlier rows are carried over",
    )
    ap.add_argument("--out", default=None, help="write here instead of the canonical path")
    ap.add_argument("--json", default=None, help="write the run report to this path")
    args = ap.parse_args(argv)

    report = build_table(since=args.since, out=args.out)
    print(json.dumps(report, indent=1), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


# --------------------------------------------------------------------------
# serving
# --------------------------------------------------------------------------

#: Where the served fold's artifact is cached. Only the CURRENT fold's model
#: needs to persist: the historical ones are deterministic given the seed and
#: the Tier-3 snapshot, so they regenerate on rebuild rather than accumulating
#: ~168 joblib files nobody reads.
SERVING_DIR = paths.DATA / "models" / "tier4"


@dataclass(frozen=True)
class ServingModel:
    """The fitted fold model the live scorer uses, plus what it was fit from."""

    estimator: object
    model_id: str
    fold_start: pd.Timestamp
    tier3_snapshot: str
    features: tuple[str, ...]
    #: Mirrors the producer's :attr:`FeatureModel.interval_floor`, so a live
    #: band and a stored one for the same event are clipped identically.
    interval_floor: float | None = 0.0
    #: The held-out ``(prediction, residual)`` pool for folds strictly before
    #: this one — the same pool the build used for the same fold, read back
    #: from the stored table rather than recomputed. Empty when Tier 4 has not
    #: been built, in which case a live forecast comes with no band, which is
    #: the correct answer rather than a fabricated one.
    pool_pred: np.ndarray = field(default_factory=lambda: np.empty(0))
    pool_res: np.ndarray = field(default_factory=lambda: np.empty(0))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Forecasts for ``frame``, NaN wherever a feature is missing.

        NaN rather than an exception because the caller is a live board: one
        name with an incomplete feature row must not take the page down, and a
        NULL forecast already means "decline to size" everywhere downstream.
        """
        values = _matrix(frame, self.features)
        out = np.full(len(frame), np.nan)
        ok = np.isfinite(values).all(axis=1)
        if ok.any():
            out[ok] = np.asarray(self.estimator.predict(values[ok]), dtype=float).ravel()
        return out

    def interval(self, predictions):
        """``(p10, p90, sd, n)`` for ``predictions``, from this fold's own pool.

        The same function and the same pool the build used, so a live band and
        a stored one for the same event agree for the same reason their centres
        do — one implementation, reached from both sides.
        """
        return interval_for(
            np.asarray(predictions, dtype=float), self.pool_pred, self.pool_res,
            floor=self.interval_floor,
        )


def serving_fold(event_date, as_of) -> pd.Timestamp:
    """Which fold's model may size a trade decided at ``as_of`` for ``event_date``.

    The event's own fold, unless that fold has not begun yet — a model for next
    month would be fit on events between now and then, which have not happened.
    So the served fold is the earlier of the two, and the live board falls back
    to an OLDER model rather than an impossible one.

    Consequence, stated rather than hidden: a Tier-4 row and a live forecast for
    the same event agree exactly when the trade is decided inside the event's
    own month, which on a three-week board is nearly always. When it is not, the
    live row was sized by the previous fold's model, and the score says which.
    """
    event_fold = pd.Timestamp(fold_start_of([pd.Timestamp(event_date)]).iloc[0])
    decision_fold = pd.Timestamp(fold_start_of([pd.Timestamp(as_of)]).iloc[0])
    return min(event_fold, decision_fold)


def _serving_path(model_id: str, fold: pd.Timestamp, snapshot: str) -> Path:
    return SERVING_DIR / f"{model_id}_{fold:%Y%m}_{snapshot[:12]}.joblib"


def _pool_before(fold, model: FeatureModel, panel: pd.DataFrame):
    """The held-out ``(prediction, residual)`` pool for folds before ``fold``.

    Read back from the STORED table rather than recomputed. That is the cheap
    way and also the correct one: recomputing would mean re-predicting every
    historical row, and any drift between that computation and the build's
    would show up as a live band that disagrees with the recorded one for no
    visible reason.
    """
    empty = (np.empty(0, dtype=float), np.empty(0, dtype=float))
    stored = load_forecasts()
    if stored is None:
        return empty
    point = model.produces
    fold_col = column_group(point)[6]
    earlier = stored[stored[point].notna() & (stored[fold_col] < pd.Timestamp(fold))]
    if earlier.empty:
        return empty
    _, trainable = training_frames(panel, model)
    realized = trainable.set_index(["ticker", "date"])[model.target]
    keys = pd.MultiIndex.from_arrays([earlier["ticker"], earlier["event_date"]])
    truth = realized.reindex(keys).to_numpy(dtype=float)
    made = earlier[point].to_numpy(dtype=float)
    ok = np.isfinite(truth) & np.isfinite(made)
    return made[ok], (truth - made)[ok]


def serving_model(
    fold_start,
    *,
    panel: pd.DataFrame | None = None,
    model: FeatureModel | None = None,
    cache: bool = True,
) -> ServingModel:
    """The fold's fitted model, from cache or freshly fit.

    This is what makes §5's claim true rather than hopeful: the historical build
    and the live scorer both reach a fold's model through :func:`fit_fold`, so
    there is one fitted estimator per fold and no second code path that could
    drift from it.

    The cache key carries the Tier-3 snapshot, so a panel rebuild writes a
    different file instead of silently serving a model fit on data that no
    longer exists.
    """
    from engine.features import load_panel  # deferred: engine.features reads Tier 4

    model = size_feature_model() if model is None else model
    fold = pd.Timestamp(fold_start)
    snapshot = store.file_sha256(paths.PANEL)
    path = _serving_path(model.model_id, fold, snapshot)

    if cache and path.exists():
        import joblib

        stored = joblib.load(path)
        if (
            stored.get("model_id") == model.model_id
            and pd.Timestamp(stored.get("fold_start")) == fold
            and stored.get("tier3_snapshot") == snapshot
            and tuple(stored.get("features", ())) == tuple(model.features)
        ):
            pool_pred, pool_res = _pool_before(
                fold, model, load_panel() if panel is None else panel
            )
            return ServingModel(
                estimator=stored["estimator"],
                model_id=model.model_id,
                fold_start=fold,
                tier3_snapshot=snapshot,
                features=tuple(model.features),
                interval_floor=model.interval_floor,
                pool_pred=pool_pred,
                pool_res=pool_res,
            )

    panel = load_panel() if panel is None else panel
    _, trainable = training_frames(panel, model)
    pool_pred, pool_res = _pool_before(fold, model, panel)
    served = ServingModel(
        estimator=fit_fold(trainable, model, fold),
        model_id=model.model_id,
        fold_start=fold,
        tier3_snapshot=snapshot,
        features=tuple(model.features),
        interval_floor=model.interval_floor,
        pool_pred=pool_pred,
        pool_res=pool_res,
    )
    if cache:
        import joblib

        paths.assert_writable(SERVING_DIR).mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "estimator": served.estimator,
                "model_id": served.model_id,
                "fold_start": str(fold.date()),
                "tier3_snapshot": snapshot,
                "features": list(served.features),
            },
            path,
        )
    return served
