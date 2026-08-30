"""Calibration: is a predicted 60% win rate actually a 60% win rate?

Accuracy and calibration are different virtues and the program needs the second
one. A gate that ranks trades perfectly but claims 80% win rates on a 55% book
would still make money and would still size every position wrong, because the
sizing rules in Phase 5 read the probability as a probability.

What is measured, on out-of-sample years only:

**Reliability.** Bucket events by predicted win rate; compare each bucket's
predicted rate against the share that actually won. A calibrated model tracks
the diagonal.

**Brier score**, against the base rate as the benchmark. Predicting the
unconditional win rate for everything is a perfectly calibrated and perfectly
useless model, and it is the bar a real one has to clear — a Brier score that
does not beat it means the predictions carry no information about *which* trades
win, however well the average lines up.

**Expected against realized P&L**, by decile of prediction. Rank quality and
level accuracy fail separately: a model can order trades correctly while being
uniformly too optimistic, and only the level error shows up here.

The ledger (Phase 4) re-runs this on frozen live predictions, where it becomes
the out-of-time test that catches whatever the backtests missed. The shape of
the output is the same so the two are directly comparable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "CalibrationReport",
    "reliability_table",
    "brier",
    "brier_skill",
    "decile_table",
    "calibrate",
]


def brier(predicted_prob, outcome) -> float:
    """Mean squared error of a probability forecast. Lower is better."""
    p = np.asarray(predicted_prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    if not ok.any():
        return float("nan")
    return float(((p[ok] - y[ok]) ** 2).mean())


def brier_skill(predicted_prob, outcome) -> float:
    """Improvement over always predicting the base rate. 0 = no better; 1 = perfect.

    Negative means the model is *worse* than a constant, which is a real and
    important outcome to be able to report.
    """
    p = np.asarray(predicted_prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    if not ok.any():
        return float("nan")
    base = float(y[ok].mean())
    reference = float(((base - y[ok]) ** 2).mean())
    if reference == 0:
        return float("nan")
    return float(1.0 - brier(p[ok], y[ok]) / reference)


def reliability_table(
    predicted_prob, outcome, *, bins: int = 10, min_per_bin: int = 20
) -> pd.DataFrame:
    """Predicted vs realized win rate, bucketed by prediction.

    Equal-count buckets, not equal-width: predicted win rates cluster, and
    equal-width bins would produce a table of empty cells and one crowded one.
    """
    p = np.asarray(predicted_prob, dtype=float)
    y = np.asarray(outcome, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < bins * min_per_bin:
        bins = max(1, len(p) // max(min_per_bin, 1))
    if bins < 1 or len(p) == 0:
        return pd.DataFrame(columns=["bin", "n", "predicted", "realized", "gap"])

    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, bins)
    rows = []
    for i, idx in enumerate(groups):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "bin": i,
                "n": int(len(idx)),
                "predicted": float(p[idx].mean()),
                "realized": float(y[idx].mean()),
                "gap": float(y[idx].mean() - p[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def decile_table(predicted_pnl, realized_pnl, *, bins: int = 10) -> pd.DataFrame:
    """Predicted vs realized mean P&L by decile of prediction."""
    p = np.asarray(predicted_pnl, dtype=float)
    y = np.asarray(realized_pnl, dtype=float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    if len(p) < bins:
        return pd.DataFrame(columns=["decile", "n", "predicted", "realized", "gap"])
    order = np.argsort(p, kind="stable")
    rows = []
    for i, idx in enumerate(np.array_split(order, bins)):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "decile": i,
                "n": int(len(idx)),
                "predicted": float(p[idx].mean()),
                "realized": float(y[idx].mean()),
                "gap": float(y[idx].mean() - p[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def monotonicity(table: pd.DataFrame, column: str = "realized") -> float:
    """Spearman correlation of bucket index against the realized column.

    The guide's sanity floor is "monotone-ish". Quantifying it beats eyeballing
    a curve: 1.0 is perfectly ordered, 0 is no ordering at all.
    """
    if len(table) < 3:
        return float("nan")
    x = np.arange(len(table), dtype=float)
    y = table[column].to_numpy(dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    xr = pd.Series(x[ok]).rank().to_numpy()
    yr = pd.Series(y[ok]).rank().to_numpy()
    if xr.std() == 0 or yr.std() == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


@dataclass
class CalibrationReport:
    label: str
    n: int
    base_rate: float
    brier: float
    brier_base: float
    brier_skill: float
    reliability: pd.DataFrame
    deciles: pd.DataFrame
    reliability_monotonicity: float
    pnl_monotonicity: float
    mean_predicted_pnl: float
    mean_realized_pnl: float
    years: tuple[int, ...] = ()
    notes: list[str] = field(default_factory=list)

    @property
    def beats_base_rate(self) -> bool:
        """The guide's floor: better than predicting the unconditional rate."""
        return np.isfinite(self.brier) and self.brier < self.brier_base

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "n": self.n,
            "years": list(self.years),
            "base_rate": round(self.base_rate, 4),
            "brier": round(self.brier, 6),
            "brier_base_rate": round(self.brier_base, 6),
            "brier_skill": round(self.brier_skill, 6),
            "beats_base_rate": self.beats_base_rate,
            "reliability_monotonicity": round(self.reliability_monotonicity, 4),
            "pnl_monotonicity": round(self.pnl_monotonicity, 4),
            "mean_predicted_pnl": round(self.mean_predicted_pnl, 6),
            "mean_realized_pnl": round(self.mean_realized_pnl, 6),
            "reliability": self.reliability.to_dict("records"),
            "deciles": self.deciles.to_dict("records"),
            "notes": self.notes,
        }


def calibrate(
    frame: pd.DataFrame,
    *,
    label: str,
    prob_column: str = "win_model",
    pnl_column: str = "exp_pnl_model",
    realized_column: str = "ret",
    year_column: str = "year",
) -> CalibrationReport:
    """Build the full calibration picture for a scored, then realized, set."""
    rows = frame.dropna(subset=[prob_column, realized_column]).copy()
    outcome = (rows[realized_column].to_numpy(dtype=float) > 0).astype(float)
    prob = rows[prob_column].to_numpy(dtype=float)

    base = float(outcome.mean()) if len(outcome) else float("nan")
    reliability = reliability_table(prob, outcome)

    predicted_pnl = (
        rows[pnl_column].to_numpy(dtype=float)
        if pnl_column in rows.columns
        else np.full(len(rows), np.nan)
    )
    realized_pnl = rows[realized_column].to_numpy(dtype=float)
    deciles = decile_table(predicted_pnl, realized_pnl)

    notes: list[str] = []
    if len(rows) < 200:
        notes.append(
            f"only {len(rows)} scored events — calibration on this sample is indicative, "
            "not a verdict"
        )

    years = ()
    if year_column in rows.columns:
        years = tuple(sorted(int(y) for y in rows[year_column].dropna().unique()))

    return CalibrationReport(
        label=label,
        n=int(len(rows)),
        base_rate=base,
        brier=brier(prob, outcome),
        brier_base=brier(np.full(len(outcome), base), outcome),
        brier_skill=brier_skill(prob, outcome),
        reliability=reliability,
        deciles=deciles,
        reliability_monotonicity=monotonicity(reliability, "realized"),
        pnl_monotonicity=monotonicity(deciles, "realized"),
        mean_predicted_pnl=float(np.nanmean(predicted_pnl)) if len(predicted_pnl) else float("nan"),
        mean_realized_pnl=float(np.nanmean(realized_pnl)) if len(realized_pnl) else float("nan"),
        years=years,
        notes=notes,
    )
