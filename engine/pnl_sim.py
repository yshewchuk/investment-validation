"""Expected P&L for a structure, by simulation — the gate that replaced arithmetic.

TWIN-P5's reward term was ``cost < peak / 2``: max profit beats max loss, decided
without reference to where the print was likely to land. EXP-129 replaced it with
the quantity that term is a proxy for. Each leg's exit value is a deterministic
function of where spot lands, what implied vol survives the print, and how much
time is left, and two of those three have out-of-sample forecasts with calibrated
error distributions in Tier 4. So the expectation is an integral over two random
variables, evaluated here by simulation.

**Why not closed form.** ``PnL(m, c)`` is a sum of Black-Scholes prices carrying
``N(d1)`` and ``N(d2)``, where ``d1`` depends on ``log(S(1+m)/K)`` and on
``sigma(1+c)sqrt(T)``. Integrating the normal CDF against a distribution over
both a shift in spot and a multiplier on vol has no elementary antiderivative. It
collapses only at ``T_exit = 0`` — and the exit is not at expiry: median 9 DTE
remain when TWIN-P5 closes, and only 15% of exits fall within a day of it.

**What EXP-129 and EXP-131 established, and what they did not.** The gate beats
``cost < peak/2`` at every matched selectivity, and on 2023-2026 held out from
its own selection it compounded 1.0 to 2.53 against the incumbent's 1.68, at
Sharpe 1.54 against 1.15. It also did NOT clear its distinguishability bar: a
block bootstrap on the held-out CAGR difference spans [-28.99, +43.71]pp. Four
years cannot separate these rules, and this module is live on a decision the
statistics could not make. See ``guides/pnl_gate_promotion.md``.

**The mechanism is simpler than the machinery suggests.** A post-hoc 2x2 found
that the MOVE's variance is the whole effect: holding it at its point forecast
costs ~9pp of CAGR, while the crush's variance moves nothing in either
direction. The crush model earns its place as a LEVEL — it sets exit vol — and
its distribution does not. The paired draw is kept because it costs nothing and
because ``independent`` measured 3.3pp of CAGR worse, but nobody should describe
this as joint integration doing the work.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from scipy.stats import norm

__all__ = [
    "DRAWS", "MIN_VOL", "MIN_SPOT_FRACTION", "MIN_POOL", "WINDOW_MONTHS",
    "QUANTILE", "black_scholes_put", "ResidualPool", "expected_pnl",
    "trailing_cutoff",
    "load_history",
    "HISTORY_PATH",
]

#: Registered in EXP-129's spec before it ran. `draws_500` measured the estimate
#: as stable at a fifth of this; 4,000 is kept because it costs milliseconds.
DRAWS = 4000

#: Vol floor after a crush draw, in decimals. A crush at or past -100% is inside
#: the residual pool's support and is not a market state.
MIN_VOL = 0.01

#: Floor on ``S_exit / S_entry``. A move draw past -100% is likewise inside the
#: pool's support and outside the world's. Left untruncated it produced a
#: negative spot, NaN through ``log(S/K)``, and silently poisoned the mean of
#: every affected event — found in EXP-129 before any result was reported.
MIN_SPOT_FRACTION = 1e-4

#: Below this many paired residuals an event is not simulated at all. Same
#: reasoning as Tier 4's MIN_RESIDUALS: an expectation over forty errors is not
#: an expectation, and a number shaped like one gets read as one.
MIN_POOL = 250

#: The trailing window the gate's cutoff is computed over, and the share it
#: admits. Chosen in EXP-131 on 2018-2022 alone and confirmed on 2023-2026.
#:
#: Six months rather than twelve is NOT a performance choice — no window from 6
#: to 36 months is distinguishable from any other on returns. It is a VOLUME
#: choice: the realized share of candidates admitted has a yearly SD of 0.027 at
#: six months against 0.051 at twelve and 0.074 for a calendar-year rule. The
#: requirement it serves is a predictable, bounded trade count.
WINDOW_MONTHS = 6
QUANTILE = 0.20


def black_scholes_put(spot, strike, years, vol):
    """Put value, vectorised over draws. Zero rates and dividends.

    Omitted deliberately rather than forgotten: over the ~9 DTE that remain at
    exit, ``exp(-rT)`` differs from 1 by under 0.1% at any plausible rate, an
    order of magnitude below this function's own repricing error. Measured
    against 2,893 known outcomes: median error $0.216 on a $5.70 median exit,
    r = 0.998, and a +$0.043 bias that points the wrong way for a gate at zero.
    """
    spot = np.asarray(spot, dtype=float)
    vol = np.maximum(np.asarray(vol, dtype=float), MIN_VOL)
    years = float(years)
    if years <= 0:
        return np.maximum(strike - spot, 0.0)
    sT = vol * np.sqrt(years)
    d1 = (np.log(spot / strike) + 0.5 * vol**2 * years) / sT
    return strike * norm.cdf(-(d1 - sT)) - spot * norm.cdf(-d1)


class ResidualPool:
    """Paired ``(move error, crush error)`` draws from strictly-earlier events.

    Both residuals come from the SAME historical event, so the dependence
    between them travels with the pair and nothing has to be estimated. That
    matters because the dependence is invisible to correlation: Pearson r is
    +0.028 across 115,195 events, while Spearman is -0.095, the median crush
    walks -13.5% to -18.5% across move deciles, and the conditional SD nearly
    doubles. A copula fitted on r = 0.028 would find none of it.

    Causality is the constraint: a pool for an event dated D holds only errors
    from events that had already printed by D.
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

    def draw(self, cutoff, prediction: float, n: int, rng):
        end = self.before(cutoff)
        if end < MIN_POOL:
            return np.empty(0), np.empty(0)
        pred = self._pred[:end]
        # Deciles from the pool that exists AT THIS CUTOFF, never the whole
        # history — that would be a leak wearing a full-sample refit's hat.
        edges = np.quantile(pred, np.linspace(0, 1, self._buckets + 1)[1:-1])
        index = int(np.searchsorted(edges, prediction, side="right"))
        rows = np.flatnonzero(np.searchsorted(edges, pred, side="right") == index)
        if rows.size < MIN_POOL:
            rows = np.arange(end)
        chosen = rows[rng.integers(0, rows.size, size=n)]
        return self._move[chosen], self._crush[chosen]


def expected_pnl(
    *,
    exit_legs,
    spot: float,
    entry_cost: float,
    pre_iv30: float,
    pred_abs_move: float,
    pred_iv_crush: float,
    dte_exit: float,
    event_date,
    pool: ResidualPool,
    key: str = "",
    draws: int = DRAWS,
) -> dict | None:
    """Expected return, win probability and band, or ``None`` if unsimulable.

    ``None`` is the third outcome and it is load-bearing: an event with no
    forecast, no pre-print vol or too thin a pool is UNDETERMINED, never
    rejected. Collapsing "we could not tell" into "no" is how a data gap becomes
    a silent permanent decline that looks like a decision.
    """
    if not exit_legs:
        return None
    values = (spot, entry_cost, pre_iv30, pred_abs_move, pred_iv_crush, dte_exit)
    if not all(v is not None and np.isfinite(v) for v in values):
        return None
    if pre_iv30 <= 0 or entry_cost <= 0 or dte_exit < 0:
        return None

    # SHA-256, not hash(): Python salts string hashing per process, and the
    # first implementation drew different samples on every run — 7 events and
    # 0.26pp of mean apart, which is this estimator's noise floor.
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(f"{key}|{event_date}".encode()).digest()[:8], "big")
    )
    err_move, err_crush = pool.draw(event_date, pred_abs_move, draws, rng)
    if err_move.size == 0:
        return None

    move = np.maximum(pred_abs_move + err_move, 0.0)
    crush = pred_iv_crush + err_crush
    # The sign is drawn rather than modelled, and that is EXACT rather than an
    # approximation: TWIN-P5 is symmetric about its anchor, so its payoff
    # depends on |move| only.
    sign = rng.choice((-1.0, 1.0), size=move.size)
    spot_exit = spot * np.maximum(1.0 + sign * move / 100.0, MIN_SPOT_FRACTION)
    vol_exit = (pre_iv30 / 100.0) * (1.0 + crush / 100.0)

    value = np.zeros(move.size)
    for leg in exit_legs:
        strike, qty = leg.get("strike"), float(leg.get("qty", 0.0))
        if strike is None or not np.isfinite(float(strike)) or qty == 0:
            continue
        # `sell` at exit means the position is LONG that leg and receives its
        # value; `buy` closes a short and pays it.
        side = 1.0 if str(leg.get("side", "")).lower() == "sell" else -1.0
        value += side * qty * black_scholes_put(spot_exit, float(strike), dte_exit / 365.0, vol_exit)

    ret = (value - entry_cost) / entry_cost
    return {
        "exp_pnl_sim": float(np.mean(ret)),
        "win_sim": float(np.mean(ret > 0)),
        "sim_p10": float(np.quantile(ret, 0.10)),
        "sim_p90": float(np.quantile(ret, 0.90)),
        "pool_n": int(pool.before(event_date)),
    }


#: Where the gate's trailing history lives. Model OUTPUT, like Tier 4, so it
#: sits beside the panel rather than inside it — Tier 3 is a deterministic
#: function of Tier 2 and `data_snapshot` pins it, and a simulated expectation
#: in there would make a champion promotion invalidate experiments that never
#: read one.
HISTORY_PATH = "data/features/pnl_sim_history.parquet"


def load_history(path: str | None = None) -> pd.DataFrame | None:
    """The stored ``exp_pnl_sim`` series the cutoff is computed from.

    ``None`` when it has never been built, which makes every gate verdict
    UNDETERMINED rather than admitting or rejecting on a bar that does not
    exist. Seeded from EXP-129's simulated universe (2,802 events, 2018-2026)
    and extended by the replay as new events price.
    """
    from pathlib import Path
    from engine import paths

    target = Path(path) if path else paths.ROOT / HISTORY_PATH
    if not target.exists():
        return None
    frame = pd.read_parquet(target)
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    return frame


def trailing_cutoff(history: pd.DataFrame, as_of, *, window_months: int = WINDOW_MONTHS,
                    quantile: float = QUANTILE, min_window: int = 100) -> float | None:
    """The bar an event must clear: the top ``quantile`` of the trailing window.

    ``history`` needs ``event_date`` and ``exp_pnl_sim``. The window is
    ``[as_of - window_months, as_of)`` — strictly before, so an event is never
    ranked against itself or anything later. ``None`` when the window is too
    thin, which makes the event UNDETERMINED rather than admitted by default.
    """
    if history is None or history.empty:
        return None
    start = pd.Timestamp(as_of).to_period("M").to_timestamp()
    window_start = start - pd.DateOffset(months=window_months)
    dates = pd.to_datetime(history["event_date"])
    prior = history[(dates >= window_start) & (dates < start)]["exp_pnl_sim"].dropna()
    if len(prior) < min_window:
        return None
    return float(np.quantile(prior.to_numpy(dtype=float), 1.0 - quantile))
