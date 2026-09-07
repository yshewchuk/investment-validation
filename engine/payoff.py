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

This module does the second. Most structures use a straight line through one
driver. STR-RUNUP is the deliberate exception introduced by EXP-149: its fixed
entry strike becomes off-centre when spot moves before exit, so its calibrated
surface uses both the predicted T-1 implied move and exit moneyness.

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
    "RunupPayoffSurface",
    "PAYOFF_DRIVER",
    "fit_payoff",
    "fit_runup_payoff",
    "driver_for",
    "runup_payoff_design",
    "scale_runup_move",
    "simulate_returns",
    "simulate_runup_returns",
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
#: ``TWIN-P`` and ``TWIN-P5`` are deliberately absent, and not because they lack
#: trades. Their exit value IS a function of the realized move — both are held
#: through the print — but the function is TWIN-PEAKED: for TWIN-P, zero beyond
#: +/-4w, maximum on the band between w and 2w in either direction, and dipping
#: again at a dead-flat print. A linear `intercept + slope * driver` map cannot
#: represent a shape that rises, falls and rises again; fitting one would
#: produce a confident number that is wrong in a specific, predictable
#: direction — overstating the flat prints the structure pays least on. So both
#: score with NO_PAYOFF_MAP and are decided by their arithmetic entry rule
#: alone (engine/entry_rules.py). That is a real limitation of the model layer
#: for these structures, recorded here rather than papered over with a map that
#: would fit.
#:
#: TWIN-P5 inherits the limitation exactly rather than escaping it. Five strikes
#: instead of seven change where the peaks sit, not that there are two of them:
#: the promoted wide wing still pays `2a` at +/-a, dips to `a` at the anchor and
#: reaches zero at +/-3a. A straight line through that is no more honest than it
#: was for the seven-strike tent.
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


RUNUP_TERMS = (
    "intercept",
    "implied_move",
    "abs_moneyness",
    "moneyness_sq_div10",
    "signed_moneyness",
    "implied_x_abs_moneyness_div10",
)
RUNUP_BASE_DAYS = 14.0


def scale_runup_move(values, days_before_print: float) -> np.ndarray:
    """Linearly interpolate or extrapolate a T-14 move distribution."""
    scale = float(days_before_print) / RUNUP_BASE_DAYS
    return np.asarray(values, dtype=float) * scale


def runup_payoff_design(implied_move, moneyness) -> np.ndarray:
    """EXP-149 surface terms for pre-print straddle exit value."""
    implied = np.asarray(implied_move, dtype=float)
    money = np.asarray(moneyness, dtype=float)
    absolute = np.abs(money)
    return np.column_stack(
        [
            np.ones(len(implied)),
            implied,
            absolute,
            np.square(money) / 10.0,
            money,
            implied * absolute / 10.0,
        ]
    )


@dataclass(frozen=True)
class RunupPayoffSurface:
    """STR-RUNUP exit value as a function of implied move and moneyness."""

    alpha: float
    coefficients: tuple[float, ...]
    resid_sd: float
    n: int
    r: float | None
    fitted_through: pd.Timestamp | None = None
    residuals: np.ndarray = field(default_factory=lambda: np.array([]))

    def residual_draws(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.residuals.size:
            return rng.choice(self.residuals, size=n, replace=True)
        if np.isfinite(self.resid_sd) and self.resid_sd > 0:
            return rng.normal(0.0, self.resid_sd, n)
        return np.zeros(n)

    def exit_value(
        self,
        implied_move,
        signed_move,
        *,
        spot: float,
        strike: float,
    ) -> np.ndarray:
        value_per_spot = self.value_per_spot(
            implied_move,
            signed_move,
            spot=spot,
            strike=strike,
        )
        return np.maximum(value_per_spot, 0.0) * float(spot)

    def value_per_spot(
        self,
        implied_move,
        signed_move,
        *,
        spot: float,
        strike: float,
    ) -> np.ndarray:
        implied = np.asarray(implied_move, dtype=float)
        move = np.asarray(signed_move, dtype=float)
        implied, move = np.broadcast_arrays(implied, move)
        exit_spot = float(spot) * np.exp(move / 100.0)
        moneyness = 100.0 * np.log(exit_spot / float(strike))
        design = runup_payoff_design(implied.ravel(), moneyness.ravel())
        value_per_spot = design @ np.asarray(self.coefficients, dtype=float)
        return value_per_spot.reshape(implied.shape)

    def as_dict(self) -> dict:
        return {
            "strategy": "STR-RUNUP",
            "driver": "im_t1+runup_move",
            "kind": "runup_payoff_surface",
            "alpha": round(self.alpha, 4),
            "coefficients": {
                name: round(value, 8)
                for name, value in zip(RUNUP_TERMS, self.coefficients)
            },
            "resid_sd": round(self.resid_sd, 8),
            "n": self.n,
            "r": round(self.r, 4) if self.r is not None else None,
            "n_residuals": int(self.residuals.size),
            "fitted_through": (
                str(self.fitted_through.date())
                if self.fitted_through is not None
                else None
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


def fit_runup_payoff(
    trades: pd.DataFrame,
    *,
    alpha: float,
    before=None,
    min_trades: int = MIN_TRADES,
) -> RunupPayoffSurface:
    """Fit the causal EXP-149 STR-RUNUP exit-value surface."""
    rows = trades[
        (trades["strategy"] == "STR-RUNUP")
        & np.isclose(trades["fill_alpha"].astype(float), float(alpha))
    ]
    if before is not None:
        before = pd.Timestamp(before).normalize()
        rows = rows[pd.to_datetime(rows["exit_date"]) < before]

    needed = ["im_t1", "spot_entry", "spot_exit", "strike", "exit_value"]
    missing = [column for column in needed if column not in rows.columns]
    if missing:
        raise PayoffError(f"STR-RUNUP: payoff surface needs columns {missing}")

    implied = pd.to_numeric(rows["im_t1"], errors="coerce").to_numpy(float)
    spot_entry = pd.to_numeric(rows["spot_entry"], errors="coerce").to_numpy(float)
    spot_exit = pd.to_numeric(rows["spot_exit"], errors="coerce").to_numpy(float)
    strike = pd.to_numeric(rows["strike"], errors="coerce").to_numpy(float)
    exit_value = pd.to_numeric(rows["exit_value"], errors="coerce").to_numpy(float)
    ok = (
        np.isfinite(implied)
        & np.isfinite(spot_entry)
        & np.isfinite(spot_exit)
        & np.isfinite(strike)
        & np.isfinite(exit_value)
        & (spot_entry > 0)
        & (spot_exit > 0)
        & (strike > 0)
    )
    implied = implied[ok]
    spot_entry = spot_entry[ok]
    spot_exit = spot_exit[ok]
    strike = strike[ok]
    exit_value = exit_value[ok]
    if len(implied) < min_trades:
        label = before.date() if before is not None else "the end"
        raise PayoffError(
            f"STR-RUNUP: {len(implied)} closed trades before {label} "
            f"for payoff surface, need {min_trades}"
        )

    moneyness = 100.0 * np.log(spot_exit / strike)
    design = runup_payoff_design(implied, moneyness)
    target = exit_value / spot_entry
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    fitted = design @ coefficients
    residuals = target - fitted
    r = (
        float(np.corrcoef(fitted, target)[0, 1])
        if fitted.std() > 0 and target.std() > 0
        else None
    )
    kept = residuals
    if kept.size > MAX_RESIDUALS:
        kept = np.random.default_rng(RESIDUAL_SEED).choice(
            kept, size=MAX_RESIDUALS, replace=False
        )
    return RunupPayoffSurface(
        alpha=float(alpha),
        coefficients=tuple(float(value) for value in coefficients),
        resid_sd=(
            float(residuals.std(ddof=2))
            if len(residuals) > 2
            else float("nan")
        ),
        n=int(len(implied)),
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


def simulate_runup_returns(
    implied_draws,
    signed_move_draws,
    payoff: RunupPayoffSurface,
    *,
    spot: float,
    strike: float,
    cost: float,
    payoff_noise=None,
) -> np.ndarray:
    """Push implied-move and pre-print spot-path draws through the surface."""
    value_per_spot = payoff.value_per_spot(
        implied_draws,
        signed_move_draws,
        spot=spot,
        strike=strike,
    )
    if payoff_noise is not None:
        value_per_spot = value_per_spot + np.asarray(payoff_noise, dtype=float)
    value = np.maximum(value_per_spot, 0.0) * float(spot)
    if cost <= 0:
        return np.full(np.shape(value), np.nan)
    return (value - float(cost)) / float(cost)
