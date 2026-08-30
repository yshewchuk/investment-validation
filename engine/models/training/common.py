"""The walk-forward harness every champion is evaluated by.

One evaluation protocol, used identically by all three models, so their metrics
are comparable and none of them can quietly grade itself on a friendlier split.

**Expanding window, parameters frozen before each test year.** Train on
everything strictly before year *Y*, predict *Y*, step forward. This is the
convention the existing research used (train ≤ Y−1, trade Y) and it is the only
one whose numbers are allowed to be headlines.

**The served model is refit on everything; the reported metrics are not.** A
model that will score next week's prints should use every event up to now — but
its published r and MAE come from the walk-forward, where each prediction was
made by a model that had not seen its target year. Reporting the refit model's
in-sample fit would be the single most flattering number available and the least
informative.

**Residuals travel with the artifact.** The walk-forward's out-of-sample errors
are stored on the model so the scoring engine can push a *distribution* through
a payoff instead of a point estimate. Earnings-move residuals are right-skewed
and fat-tailed; a normal assumption would misprice exactly the tails that decide
whether a long-vol structure pays.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "SEED",
    "WalkForwardResult",
    "walk_forward",
    "regression_metrics",
    "BlendModel",
    "log",
]

#: One seed for the whole program. Recorded in every registry entry, so a
#: retrain that changes it is visible rather than mysterious.
SEED = 20260829


def log(message: str) -> None:
    print(f"  [train] {message}", flush=True)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def regression_metrics(y_true, y_pred) -> dict:
    """Pearson r, MAE, RMSE, bias, n — the common table for every model."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 2:
        return {"n": int(len(y_true)), "r": None, "mae": None, "rmse": None, "bias": None}
    err = y_pred - y_true
    # A constant prediction has zero variance and no defined correlation; report
    # None rather than a NaN that later formats as a number.
    r = (
        float(np.corrcoef(y_true, y_pred)[0, 1])
        if y_true.std() > 0 and y_pred.std() > 0
        else None
    )
    return {
        "n": int(len(y_true)),
        "r": r,
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
    }


def decile_spread(y_true, y_pred, k: int = 10) -> float | None:
    """Realized mean in the top predicted decile minus the bottom.

    The money statistic for a ranking model: a gate does not need calibrated
    levels, it needs the trades it likes to beat the trades it does not.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < k * 5:
        return None
    order = np.argsort(y_pred)
    size = len(order) // k
    return float(y_true[order[-size:]].mean() - y_true[order[:size]].mean())


# --------------------------------------------------------------------------
# walk-forward
# --------------------------------------------------------------------------


@dataclass
class WalkForwardResult:
    """OOS predictions, per-year metrics, and the residuals the artifact keeps."""

    frame: pd.DataFrame  # the input rows that got an OOS prediction, plus `pred`
    target: str
    features: tuple[str, ...]
    by_year: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    years: tuple[int, ...] = ()
    elapsed_s: float = 0.0

    @property
    def residuals(self) -> np.ndarray:
        """``y_true − y_pred`` out of sample.

        Signed this way round so a draw can be *added* to a prediction to make a
        plausible realization: ``y ≈ pred + residual``.
        """
        return (self.frame[self.target] - self.frame["pred"]).to_numpy(dtype=float)

    def as_dict(self) -> dict:
        return {
            "target": self.target,
            "features": list(self.features),
            "oos_years": list(self.years),
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in self.metrics.items()},
            "by_year": self.by_year.to_dict("records"),
        }


def walk_forward(
    frame: pd.DataFrame,
    features: Sequence[str],
    target: str,
    fit: Callable,
    *,
    year_column: str = "year",
    first_test_year: int | None = None,
    min_train_rows: int = 500,
    seed: int = SEED,
) -> WalkForwardResult:
    """Expanding-window walk-forward. ``fit(X, y, seed) -> model``.

    Rows with a missing feature or target are dropped up front rather than
    imputed: an imputed feature is a fabricated observation, and the models here
    are small enough that dropping is affordable. The count that survives is
    reported so the cost is visible.
    """
    started = time.time()
    features = tuple(features)
    missing = [c for c in (*features, target, year_column) if c not in frame.columns]
    if missing:
        raise KeyError(f"training frame is missing {missing}")

    usable = frame[list(features) + [target, year_column]].copy()
    complete = np.isfinite(usable[list(features)].to_numpy(dtype=float)).all(axis=1)
    complete &= np.isfinite(usable[target].to_numpy(dtype=float))
    data = frame[complete].copy()
    log(
        f"{target}: {len(data):,} of {len(frame):,} rows complete on "
        f"{len(features)} features"
    )
    if data.empty:
        raise ValueError(f"no rows are complete on {list(features)} + {target}")

    years = sorted(int(y) for y in data[year_column].dropna().unique())
    if first_test_year is not None:
        years = [y for y in years if y >= first_test_year]

    predictions = np.full(len(data), np.nan)
    year_values = data[year_column].to_numpy()
    tested: list[int] = []
    rows: list[dict] = []

    for year in years:
        train_mask = year_values < year
        test_mask = year_values == year
        n_train = int(train_mask.sum())
        if n_train < min_train_rows or not test_mask.any():
            continue
        X_train = data.loc[train_mask, list(features)].to_numpy(dtype=float)
        y_train = data.loc[train_mask, target].to_numpy(dtype=float)
        X_test = data.loc[test_mask, list(features)].to_numpy(dtype=float)
        model = fit(X_train, y_train, seed)
        pred = np.asarray(model.predict(X_test), dtype=float).ravel()
        predictions[test_mask] = pred
        tested.append(year)
        stats = regression_metrics(data.loc[test_mask, target], pred)
        rows.append({"year": year, "n_train": n_train, **stats})
        log(
            f"  {year}: train {n_train:,} → test {stats['n']:,}  "
            f"r={stats['r']:.3f} mae={stats['mae']:.3f}"
            if stats["r"] is not None
            else f"  {year}: train {n_train:,} → test {stats['n']:,}"
        )

    data["pred"] = predictions
    scored = data[np.isfinite(data["pred"])].copy()
    metrics = regression_metrics(scored[target], scored["pred"])
    metrics["decile_spread"] = decile_spread(scored[target], scored["pred"])
    metrics["oos_years"] = len(tested)

    result = WalkForwardResult(
        frame=scored,
        target=target,
        features=features,
        by_year=pd.DataFrame(rows),
        metrics=metrics,
        years=tuple(tested),
        elapsed_s=time.time() - started,
    )
    log(
        f"{target}: OOS n={metrics['n']:,} r={metrics['r']:.4f} "
        f"mae={metrics['mae']:.4f} over {len(tested)} years "
        f"({result.elapsed_s:.0f}s)"
        if metrics["r"] is not None
        else f"{target}: OOS n={metrics['n']:,}"
    )
    return result


def fit_final(
    frame: pd.DataFrame, features: Sequence[str], target: str, fit: Callable, seed: int = SEED
):
    """Refit on every complete row — the model that actually scores live events."""
    features = tuple(features)
    values = frame[list(features)].to_numpy(dtype=float)
    y = frame[target].to_numpy(dtype=float)
    ok = np.isfinite(values).all(axis=1) & np.isfinite(y)
    log(f"final fit on {int(ok.sum()):,} rows")
    return fit(values[ok], y[ok], seed)


# --------------------------------------------------------------------------
# the blend
# --------------------------------------------------------------------------


@dataclass
class BlendModel:
    """Equal-weight average of several fitted regressors.

    The champion size model is an OLS+NN blend, which beat either component
    alone across the EXP-030/031/036 bake-offs. Equal weights are deliberate:
    fitting blend weights on the same data would be one more parameter chosen by
    looking at the answer, on a dataset that ~50 experiments have already
    touched.
    """

    models: tuple

    def predict(self, X) -> np.ndarray:
        preds = [np.asarray(m.predict(X), dtype=float).ravel() for m in self.models]
        return np.mean(preds, axis=0)
