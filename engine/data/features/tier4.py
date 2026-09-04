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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
    "Tier4Error",
    "FeatureModel",
    "size_feature_model",
    "fold_start_of",
    "fit_fold",
    "build_forecasts",
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

#: The table. One row per ``(ticker, event_date)`` — the same grain as Tier 3,
#: and total over it: every Tier-3 event gets a row, NULL where no forecast was
#: possible. Totality is deliberate. If Tier 4 held only the rows it could
#: predict, a *missing* row would be ambiguous between "no forecast available"
#: and "this table is stale", and only one of those is acceptable to a consumer.
COLUMNS = (
    "ticker",
    "event_date",
    "pred_abs_move",
    "model_id",
    "fold_start",
    "tier3_snapshot",
)

#: Columns whose NULL means "no forecast", never zero. A consumer given one of
#: these as NULL must decline to size a structure rather than sizing it at zero;
#: the TWIN-P entry filters already drop such rows, but that is a property to
#: state rather than to inherit by luck.
FORECAST_COLUMNS = ("pred_abs_move",)


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


def training_frames(panel: pd.DataFrame, model: FeatureModel):
    """``(scorable, trainable)`` — the rows that can be predicted, and learned from.

    These are not the same set, and conflating them is what would make Tier 4
    useless live. A prediction needs complete *features*; only training also
    needs a realized target. An event that has not printed yet has no
    ``abs_move`` and must still receive a forecast — that is the entire point of
    materialising one.
    """
    prepared = model.prepare(panel)
    scorable = prepared[_complete(prepared, model.features)].copy()
    target = pd.to_numeric(scorable[model.target], errors="coerce").to_numpy(dtype=float)
    trainable = scorable[np.isfinite(target)].copy()
    return scorable, trainable


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
    out["pred_abs_move"] = pd.to_numeric(out["pred_abs_move"], errors="coerce").astype(float)
    out["model_id"] = out["model_id"].astype("string")
    out["fold_start"] = pd.to_datetime(out["fold_start"]).astype("datetime64[us]")
    out["tier3_snapshot"] = out["tier3_snapshot"].astype("string")
    return out.sort_values(["ticker", "event_date"]).reset_index(drop=True)


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
    prefix_keys = keys[keys["event_date"] < cut]
    have = existing[existing["event_date"] < cut]

    illegal = int((~_is_fold_start(have["fold_start"])).sum())
    if illegal:
        raise Tier4Error(
            f"{illegal:,} carried rows have a fold_start that is not a {CADENCE} "
            "boundary — the existing table was built on a different cadence. "
            "Rebuild in full."
        )

    ids = set(have["model_id"].dropna().unique())
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


def build_forecasts(
    panel: pd.DataFrame,
    *,
    model: FeatureModel | None = None,
    since=None,
    existing: pd.DataFrame | None = None,
    tier3_snapshot: str | None = None,
    log: Callable[[str], None] = _log,
) -> pd.DataFrame:
    """Build the forecast table for ``panel``.

    ``since`` is rounded **down to its fold boundary**, because a fold is the
    unit of recomputation: half a month cannot be rebuilt without fitting the
    model that month's other half already used. Rounding down recomputes a
    superset of what was asked for, which is identical to what was already
    there, so the equivalence in §9 of the design note holds either way.
    """
    started = time.time()
    model = size_feature_model() if model is None else model
    snapshot = store.file_sha256(paths.PANEL) if tier3_snapshot is None else tier3_snapshot

    keys = _keys(panel)
    scorable, trainable = training_frames(panel, model)
    log(
        f"{len(keys):,} Tier-3 events · {len(scorable):,} scorable on "
        f"{len(model.features)} features · {len(trainable):,} trainable"
    )

    scorable = scorable.copy()
    scorable["fold_start"] = fold_start_of(scorable["date"]).to_numpy()

    cut = None
    carried = _empty()
    if since is not None:
        cut = pd.Timestamp(fold_start_of([pd.Timestamp(since)]).iloc[0])
        prior = _empty() if existing is None else normalize(existing)
        carried = _carried_prefix(prior, keys, cut, model)
        log(f"--since {pd.Timestamp(since).date()} → fold {cut.date()}; carrying {len(carried):,} row(s)")

    folds = sorted({f for f in scorable["fold_start"].unique() if pd.Timestamp(f) >= FIRST_FOLD})
    if cut is not None:
        folds = [f for f in folds if pd.Timestamp(f) >= cut]

    predictions = pd.Series(np.nan, index=scorable.index, dtype=float)
    fitted_for = pd.Series(pd.NaT, index=scorable.index, dtype="datetime64[us]")
    skipped = 0
    for fold in folds:
        stamp = pd.Timestamp(fold)
        test = scorable.index[scorable["fold_start"] == fold]
        n_train = int((trainable["date"] < stamp).sum())
        if n_train < MIN_TRAIN_ROWS:
            skipped += 1
            continue
        estimator = fit_fold(trainable, model, stamp)
        predictions.loc[test] = np.asarray(
            estimator.predict(_matrix(scorable.loc[test], model.features)), dtype=float
        ).ravel()
        fitted_for.loc[test] = stamp
        log(
            f"fold {stamp.date()}: train {n_train:,} → {len(test):,} forecast(s) "
            f"[{time.time() - started:.0f}s]"
        )
    if skipped:
        log(f"{skipped} fold(s) skipped for a training pool under {MIN_TRAIN_ROWS:,}")

    built = pd.DataFrame(
        {
            "ticker": scorable["ticker"].to_numpy(),
            "event_date": pd.to_datetime(scorable["date"]).astype("datetime64[us]").to_numpy(),
            "pred_abs_move": predictions.to_numpy(),
            "fold_start": fitted_for.to_numpy(),
        }
    )
    # A row that got no forecast carries no model id: the provenance columns
    # describe a prediction, and there is none to describe.
    built["model_id"] = np.where(built["fold_start"].notna(), model.model_id, None)
    built = built[built["fold_start"].notna() | built["pred_abs_move"].notna()]

    scope = keys if cut is None else keys[keys["event_date"] >= cut]
    fresh = scope.merge(built, on=["ticker", "event_date"], how="left")
    fresh["tier3_snapshot"] = snapshot

    out = normalize(pd.concat([carried, fresh], ignore_index=True))
    if len(out) != len(keys):
        raise Tier4Error(
            f"built {len(out):,} rows for {len(keys):,} Tier-3 events — Tier 4 must be "
            "total over Tier 3"
        )
    have = int(out["pred_abs_move"].notna().sum())
    log(
        f"{len(out):,} rows · {have:,} with a forecast ({have / max(len(out), 1):.1%}) · "
        f"{time.time() - started:.0f}s"
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


def load_forecasts(path: Path | None = None) -> pd.DataFrame | None:
    """The forecast table, or ``None`` if it has never been built."""
    path = paths.TIER4 if path is None else Path(path)
    if not path.exists():
        alt = path.with_suffix(".csv.gz")
        if not alt.exists():
            return None
        return normalize(pd.read_csv(alt))
    return normalize(pd.read_parquet(path))


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
        "with_forecast": int(frame["pred_abs_move"].notna().sum()),
        "model_ids": sorted(frame["model_id"].dropna().unique().tolist()),
        "folds": int(frame["fold_start"].nunique()),
        "cadence": CADENCE,
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
            return ServingModel(
                estimator=stored["estimator"],
                model_id=model.model_id,
                fold_start=fold,
                tier3_snapshot=snapshot,
                features=tuple(model.features),
            )

    panel = load_panel() if panel is None else panel
    _, trainable = training_frames(panel, model)
    served = ServingModel(
        estimator=fit_fold(trainable, model, fold),
        model_id=model.model_id,
        fold_start=fold,
        tier3_snapshot=snapshot,
        features=tuple(model.features),
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
