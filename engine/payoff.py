"""From a predicted quantity to an expected P&L, without inventing a pricer.

The model layer's job is to answer "what do we expect this structure to make?"
from a model that predicts something else — a move size, an implied move at
T−1. Something has to bridge the two, and there are two ways to build that
bridge:

*Price it theoretically.* Take the predicted move, feed it through
Black-Scholes at an assumed post-print volatility, subtract the entry cost.
This requires assuming the very thing the program has spent fifty experiments
establishing it cannot assume — how implied volatility behaves through a print,
per name, per regime. The result would be a number with no error bars and no
way to check it.

*Calibrate it empirically.* We already have thousands of these exact structures
priced on real chains at real fills by :mod:`engine.replay`. So fit the map from
the predicted quantity to the realized exit value, on those trades, and use the
entry cost from the actual chain the trade would be entered on.

This module does the second. The map is deliberately a straight line through one
variable — with a few thousand trades and an admittedly lumpy edge, a flexible
fit would model 2022 and 2024 rather than the mechanism.

**The map is fitted causally.** ``fit_payoff(..., before=as_of)`` uses only
trades that had already closed by the decision date, so a payoff map used to
score January 2021 knows nothing about how 2021 turned out. This is the same
discipline the walk-forward applies to the models themselves, and skipping it
here would reintroduce look-ahead through the back door after the models had
been so careful to avoid it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "PayoffMap",
    "PAYOFF_DRIVER",
    "fit_payoff",
    "driver_for",
    "simulate_returns",
]

#: What each structure's exit value is a function of.
#:
#: ``STR-THRU`` is held *through* the print, so its exit value is driven by the
#: realized move — that is the entire trade.
#:
#: ``STR-RUNUP`` is sold *before* the print, so the realized move is irrelevant
#: to it by construction. What it is worth at exit is set by the implied move
#: the market is quoting at that moment, which is what the ``implied_t1`` model
#: predicts.
PAYOFF_DRIVER = {
    "STR-THRU": "abs_move",
    "STR-RUNUP": "im_t1",
}


def driver_for(strategy: str) -> str:
    if strategy not in PAYOFF_DRIVER:
        raise KeyError(
            f"no payoff driver for {strategy!r}. CAL-P is deliberately absent: its "
            "exact spec has never been backtested (Phase 2 backlog 1-2), so there "
            "is nothing honest to calibrate a payoff against."
        )
    return PAYOFF_DRIVER[strategy]


@dataclass(frozen=True)
class PayoffMap:
    """``exit_value / spot ≈ intercept + slope · driver``, fitted on real trades.

    Quoted per unit of spot so a $400 name and a $12 name are the same
    observation. ``resid_sd`` carries the spread the line does not explain,
    which the scorer folds in alongside the model's own residuals — the
    prediction's uncertainty and the payoff's are different uncertainties, and
    reporting only the first would make every interval too narrow.
    """

    strategy: str
    driver: str
    alpha: float
    intercept: float
    slope: float
    resid_sd: float
    n: int
    r: float | None
    fitted_through: pd.Timestamp | None = None
    #: Empirical residuals of ``exit_value / spot`` around the fitted line.
    #:
    #: Stored, rather than summarized by ``resid_sd``, for the same reason
    #: :class:`~engine.models.registry.ModelArtifact` stores the model's: the
    #: residuals of a long-vol payoff are strongly right-skewed — many small
    #: losses against a few large gains — and a symmetric Gaussian of the same
    #: standard deviation puts as much mass above the line as below it. That
    #: overstates P(profit) on every event, which is precisely the way a win-rate
    #: forecast fails to beat its own base rate.
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    def residual_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """``n`` draws from the empirical residual distribution, per unit of spot.

        Falls back to a Gaussian only when the fit kept no residuals, which
        happens for a map restored from a serialized summary.
        """
        if self.residuals.size:
            return rng.choice(self.residuals, size=n, replace=True)
        if np.isfinite(self.resid_sd) and self.resid_sd > 0:  # pragma: no cover
            return rng.normal(0.0, self.resid_sd, n)
        return np.zeros(n)  # pragma: no cover - a degenerate fit

    def exit_value(self, driver_values, spot: float) -> np.ndarray:
        """Predicted exit value in dollars, floored at zero.

        A long option structure cannot be worth less than nothing, and a linear
        fit extrapolated to a small realized move will happily say it is.
        """
        values = np.asarray(driver_values, dtype=float)
        return np.maximum(0.0, (self.intercept + self.slope * values) * spot)

    def pnl(self, driver_values, spot: float, cost: float) -> np.ndarray:
        return self.exit_value(driver_values, spot) - cost

    def ret(self, driver_values, spot: float, cost: float) -> np.ndarray:
        if cost <= 0:
            return np.full(np.shape(driver_values), np.nan)
        return self.pnl(driver_values, spot, cost) / cost

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "driver": self.driver,
            "alpha": self.alpha,
            "intercept": round(self.intercept, 8),
            "slope": round(self.slope, 8),
            "resid_sd": round(self.resid_sd, 8),
            "n": self.n,
            "r": round(self.r, 4) if self.r is not None else None,
            "n_residuals": int(self.residuals.size),
            "fitted_through": (
                str(self.fitted_through.date()) if self.fitted_through is not None else None
            ),
        }


class PayoffError(RuntimeError):
    """Not enough closed trades to calibrate a payoff map."""


#: Below this many closed trades a fitted line is noise with a slope.
MIN_TRADES = 200

#: Residuals kept per map. Enough to describe a skewed distribution; small
#: enough that caching a map per (strategy, alpha, cutoff) stays cheap.
MAX_RESIDUALS = 5000

#: Fixed, so the kept subsample is a function of the data rather than of when
#: the map happened to be fitted.
RESIDUAL_SEED = 20260829


def fit_payoff(
    trades: pd.DataFrame,
    strategy: str,
    *,
    alpha: float,
    driver: str | None = None,
    before=None,
    min_trades: int = MIN_TRADES,
) -> PayoffMap:
    """Fit the exit-value line for one strategy at one fill alpha.

    ``trades`` needs ``strategy``, ``fill_alpha``, ``exit_value``,
    ``spot_entry``, ``exit_date`` and the driver column. ``before`` restricts to
    trades that had *closed* by then — closed, not entered, because a trade
    still open on the decision date has not yet told us what it was worth.
    """
    driver = driver or driver_for(strategy)
    rows = trades[
        (trades["strategy"] == strategy)
        & np.isclose(trades["fill_alpha"].astype(float), float(alpha))
    ]
    if before is not None:
        before = pd.Timestamp(before).normalize()
        rows = rows[pd.to_datetime(rows["exit_date"]) < before]

    needed = [driver, "exit_value", "spot_entry"]
    missing = [c for c in needed if c not in rows.columns]
    if missing:
        raise PayoffError(f"{strategy}: payoff fit needs columns {missing}")

    x = pd.to_numeric(rows[driver], errors="coerce").to_numpy(dtype=float)
    spot = pd.to_numeric(rows["spot_entry"], errors="coerce").to_numpy(dtype=float)
    exit_value = pd.to_numeric(rows["exit_value"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(spot) & np.isfinite(exit_value) & (spot > 0)
    x, y = x[ok], exit_value[ok] / spot[ok]

    if len(x) < min_trades:
        raise PayoffError(
            f"{strategy}: {len(x)} closed trades before "
            f"{before.date() if before is not None else 'the end'} — need {min_trades} "
            "to calibrate a payoff map"
        )

    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (intercept + slope * x)
    r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None

    # Cap the stored residuals: a map fitted on 17k trades does not need all of
    # them to describe its own error distribution, and the scorer caches one map
    # per (strategy, alpha, cutoff).
    kept = resid
    if kept.size > MAX_RESIDUALS:
        kept = np.random.default_rng(RESIDUAL_SEED).choice(
            kept, size=MAX_RESIDUALS, replace=False
        )

    return PayoffMap(
        strategy=strategy,
        driver=driver,
        alpha=float(alpha),
        intercept=float(intercept),
        slope=float(slope),
        resid_sd=float(resid.std(ddof=2)) if len(resid) > 2 else float("nan"),
        n=int(len(x)),
        r=r,
        fitted_through=before,
        residuals=np.sort(kept),
    )


def simulate_returns(
    driver_draws,
    payoff: PayoffMap,
    spot: float,
    cost: float,
    payoff_noise=None,
) -> np.ndarray:
    """Return distribution from driver draws pushed through the payoff map.

    ``driver_draws`` are the model's draws of the driver (point prediction plus
    the model's own residuals). ``payoff_noise`` — per unit of spot, drawn from
    the payoff map's empirical residuals — is the scatter the line does not
    explain, added after the exit-value floor. The win rate is the share of the
    result above zero; this function is the whole of that computation, in one
    place, so the scoring path and the mechanics tests run the same code.
    """
    pnl = payoff.pnl(driver_draws, spot, cost)
    if payoff_noise is not None:
        pnl = pnl + np.asarray(payoff_noise, dtype=float) * spot
    if cost <= 0:
        return np.full(np.shape(pnl), np.nan)
    return pnl / cost
