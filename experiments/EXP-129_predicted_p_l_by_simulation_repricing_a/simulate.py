"""Repricing a twin peak under simulated (move, IV crush) outcomes.

The structure's exit value is a deterministic function of three things — where
spot lands, what implied vol survives the print, and how much time is left —
and two of the three now have out-of-sample forecasts with calibrated error
distributions in Tier 4. So the expected P&L is an integral over those two
random variables, and this module evaluates it by simulation.

**Why not a closed form.** ``PnL(m, c)`` is a sum of Black-Scholes prices, each
carrying ``N(d1)`` and ``N(d2)`` where ``d1`` depends on ``log(S(1+m)/K)`` and on
``sigma(1+c)*sqrt(T)``. Integrating the normal CDF against a distribution over
BOTH a shift in spot and a multiplier on vol has no elementary antiderivative.
It collapses only at ``T_exit = 0``, where the value is piecewise-linear
intrinsic and the crush drops out — and that escape is not available: measured
over 1,214 sampled events, the median DTE remaining at TWIN-P's exit is 9, and
only 15.0% of exits fall within a day of expiry.

**Why the draws are PAIRED.** ``r(|move|, crush) = +0.0284`` on 115,195 events,
which reads as independence and is not. Spearman is −0.0951, the median crush
walks −13.51% → −18.53% across |move| deciles, and the conditional SD nearly
doubles, 14.9 → 27.3. The dependence lives in the higher moments, exactly where
a Gaussian copula fitted on r = 0.028 finds nothing. For a payoff that is a
BAND rather than a monotone function of the move, the shape of the joint
distribution IS the expected value.

So no copula is fitted and no correlation estimated: a draw takes both
residuals from the SAME historical event, and whatever dependence exists —
including the heteroskedasticity — travels with the pair for free.

**What bounds the accuracy.** Not the draw count. Black-Scholes priced against
132,023 real quoted puts (DTE 3-60, mid > $0.05) lands 3.88% from the mid at
the median, but 14.41% for |delta| < 0.10 on a median $0.40 contract. TWIN-P5's
wings are those contracts. More draws shrink the estimate's variance and do
nothing to that bias, which is why ``bs_only`` is an arm rather than the only
option: where a real quoted exit exists it is used, and the share is reported.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "DRAWS",
    "MIN_VOL",
    "MIN_POOL",
    "ResidualPool",
    "parse_legs",
    "black_scholes_put",
    "simulate_event",
]

#: Registered in spec.yaml before this ran.
DRAWS = 4000

#: Vol floor after a crush draw, in decimals. A crush of −100% or worse is
#: inside the residual distribution's support but is not a market state; the
#: option would be worth intrinsic and the model would divide by zero getting
#: there. Clipped rather than dropped, so a heavy-crush draw still contributes
#: its (correct, near-intrinsic) value instead of silently thinning the sample.
MIN_VOL = 0.01

#: Below this many paired residuals a fold gets no simulation at all. Same
#: reasoning as Tier 4's MIN_RESIDUALS: an expectation over forty draws from
#: forty errors is not an expectation, and a number shaped like one is read as
#: one.
MIN_POOL = 250

#: Floor on ``S_exit / S_entry``. The move residual pool is heavy-tailed and a
#: down draw beyond -100% is not a market state; a stock at zero is the limit.
MIN_SPOT_FRACTION = 1e-4


def black_scholes_put(spot, strike, years, vol):
    """Put value. Vectorised over draws; zero rates and dividends.

    Rates are omitted deliberately rather than forgotten. The horizon here is
    the DTE remaining at exit — median 9 days — over which ``exp(-rT)`` differs
    from 1 by under 0.1% at any plausible rate, which is an order of magnitude
    below the repricing error this function already carries.
    """
    spot = np.asarray(spot, dtype=float)
    vol = np.maximum(np.asarray(vol, dtype=float), MIN_VOL)
    years = float(years)
    if years <= 0:
        return np.maximum(strike - spot, 0.0)
    sT = vol * np.sqrt(years)
    d1 = (np.log(spot / strike) + 0.5 * vol**2 * years) / sT
    d2 = d1 - sT
    return strike * norm.cdf(-d2) - spot * norm.cdf(-d1)


def parse_legs(raw) -> dict:
    """The ``legs`` blob EXP-126 stored, as a dict."""
    return json.loads(raw) if isinstance(raw, str) else raw


class ResidualPool:
    """Paired ``(move error, crush error)`` draws from strictly-earlier events.

    The pairing is the point and the causality is the constraint. A pool for an
    event dated D holds only errors from events that had already printed by D,
    so a simulation can never be informed by an outcome the model had not yet
    been wrong about. This is the same discipline Tier 4's stored bands carry,
    applied to a joint distribution rather than a marginal one.

    Conditioned on the predicted move by decile, reusing EXP-115's finding that
    error scales with the prediction: one pool for every row understates the
    spread at the top of the range and overstates it at the bottom. A decile
    thinner than :data:`MIN_POOL` falls back to the whole pool, so conditioning
    can only refine and never leaves a sparse region with too few pairs.
    """

    def __init__(self, history: pd.DataFrame, buckets: int = 10) -> None:
        need = ["event_date", "pred_abs_move", "err_move", "err_crush"]
        missing = [c for c in need if c not in history.columns]
        if missing:
            raise ValueError(f"residual history is missing {missing}")
        h = history.dropna(subset=need).sort_values("event_date").reset_index(drop=True)
        self._dates = pd.to_datetime(h["event_date"]).to_numpy()
        self._pred = h["pred_abs_move"].to_numpy(dtype=float)
        self._move = h["err_move"].to_numpy(dtype=float)
        self._crush = h["err_crush"].to_numpy(dtype=float)
        self._buckets = int(buckets)

    def __len__(self) -> int:
        return len(self._dates)

    def before(self, cutoff) -> int:
        return int(np.searchsorted(self._dates, np.datetime64(pd.Timestamp(cutoff)), "left"))

    def draw(self, cutoff, prediction: float, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
        """``(move errors, crush errors)`` — n paired draws, or empty if too thin."""
        end = self.before(cutoff)
        if end < MIN_POOL:
            return np.empty(0), np.empty(0)
        pred = self._pred[:end]
        # The decile the prediction falls in, computed on the pool that exists
        # at this cutoff — never on the whole history, which would be a leak
        # wearing the same hat as a full-sample refit.
        edges = np.quantile(pred, np.linspace(0, 1, self._buckets + 1)[1:-1])
        index = int(np.searchsorted(edges, prediction, side="right"))
        in_bucket = np.searchsorted(edges, pred, side="right") == index
        rows = np.flatnonzero(in_bucket)
        if rows.size < MIN_POOL:
            rows = np.arange(end)
        picks = rng.integers(0, rows.size, size=n)
        chosen = rows[picks]
        return self._move[chosen], self._crush[chosen]


def simulate_event(
    row: pd.Series,
    pool: ResidualPool,
    *,
    draws: int = DRAWS,
    paired: bool = True,
    use_crush: bool = True,
    use_move: bool = True,
    seed_key: str = "",
) -> dict | None:
    """Expected return, win probability and band for one candidate event.

    ``None`` when the event cannot be simulated — no pre-print vol, no forecast,
    an unusable exit horizon, or a residual pool too thin at its date. A None is
    reported as ungateable rather than defaulted, because a structure sized on a
    missing forecast is a trade nobody chose.
    """
    legs = parse_legs(row["legs"])
    entry = legs.get("entry") or []
    exits = legs.get("exit") or []
    if not entry or not exits:
        return None

    sigma = float(row.get("pre_iv30", np.nan))
    forecast = float(row.get("pred_abs_move", np.nan))
    crush_hat = float(row.get("pred_iv_crush_30", np.nan))
    spot = float(row.get("spot_entry", np.nan))
    cost = float(row.get("entry_cost", np.nan))
    if not np.isfinite([sigma, forecast, spot, cost]).all() or sigma <= 0 or cost <= 0:
        return None
    if use_crush and not np.isfinite(crush_hat):
        return None

    dte_exit = float(np.median([float(leg.get("dte", np.nan)) for leg in exits]))
    if not np.isfinite(dte_exit) or dte_exit < 0:
        return None
    years = dte_exit / 365.0

    # SHA-256, not Python's hash(): `hash()` on strings is salted per process
    # by PYTHONHASHSEED, so two runs of this file drew different samples and the
    # gated set moved between them — 1,489 events one run, 1,482 the next, with
    # the primary's mean shifting 2.79% -> 3.05%. The spec registers this as
    # seeded per (key, event) so a rerun is bit-identical; it was not.
    # engine.score seeds its own residual draws this way for the same reason.
    rng = np.random.default_rng(
        int.from_bytes(
            hashlib.sha256(f"{seed_key}|{row.get('event_id', '')}".encode()).digest()[:8],
            "big",
        )
    )
    err_move, err_crush = pool.draw(row["event_date"], forecast, draws, rng)
    if err_move.size == 0:
        return None
    if not paired:
        # The arm that measures the pairing: same marginals, dependence broken.
        err_crush = err_crush[rng.permutation(err_crush.size)]

    # `use_move` / `use_crush` switch each variable between DRAWN and held at
    # its point forecast. Holding one is not a leak — the forecast is known at
    # decision time — but it estimates E[PnL | x = x_hat] rather than E[PnL],
    # and the payoff is nonlinear in both, so the two are different quantities.
    #
    # Holding BOTH is the control that matters: TWIN-P5's peak is placed at
    # exactly +/-forecast, so with zero variance the structure sits ON its peak
    # by construction and the simulated return collapses toward
    # (peak - cost)/cost — the reward:risk ratio the incumbent already ranks on.
    # If that arm still produces a good book, the simulation is decoration.
    move = np.maximum(forecast + err_move, 0.0) if use_move else np.full(err_move.size, forecast)
    crush = (crush_hat + err_crush) if use_crush else np.full(err_move.size, crush_hat)
    sign = rng.choice((-1.0, 1.0), size=move.size)              # exact: the shape is symmetric
    # A move draw past -100% is inside the residual distribution's support and
    # outside the world's: a share price cannot go negative. Left untruncated
    # it produced NaN through log(S/K) and silently poisoned the mean of every
    # affected event. Floored at MIN_SPOT_FRACTION of the entry spot, which is
    # the correct limit — the puts are then worth their full strike.
    growth = np.maximum(1.0 + sign * move / 100.0, MIN_SPOT_FRACTION)
    spot_exit = spot * growth
    vol_exit = (sigma / 100.0) * (1.0 + crush / 100.0)

    value = np.zeros(move.size)
    for leg in exits:
        strike = float(leg.get("strike", np.nan))
        qty = float(leg.get("qty", 0.0))
        if not np.isfinite(strike) or qty == 0:
            continue
        side = 1.0 if str(leg.get("side", "")).lower() == "sell" else -1.0
        # `sell` at exit means the position is LONG that leg and receives its
        # value; `buy` at exit closes a short and pays it.
        value += side * qty * black_scholes_put(spot_exit, strike, years, vol_exit)

    ret = (value - cost) / cost
    return {
        "exp_pnl_sim": float(np.mean(ret)),
        "win_sim": float(np.mean(ret > 0)),
        "sim_p10": float(np.quantile(ret, 0.10)),
        "sim_p90": float(np.quantile(ret, 0.90)),
        "sim_sd": float(np.std(ret)),
        "pool_n": int(pool.before(row["event_date"])),
        "dte_exit": dte_exit,
        "draws": int(move.size),
    }
