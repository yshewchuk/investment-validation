#!/usr/bin/env python3
"""A residual pool that knows how big the company is.

``engine.pnl_sim.ResidualPool`` conditions its draw on the predicted-move
decile and nothing else. That was EXP-115's finding applied honestly — error
scales with the prediction — but it leaves a mismatch this program only noticed
in EXP-139: **the pool is 87,121 events of which 27.5% clear the $10B floor the
strategy actually trades.**

Market cap carries shape information the prediction level does not. Measured on
the pool, with the ratio realized/predicted:

    mcap        n        |move| med   ratio med   p99    skew    cv
    <$2B        29,078   4.83%        0.79        4.27   +2.95   0.93
    $2-10B      28,632   3.65%        0.81        3.90   +1.71   0.85
    $10-50B     20,028   3.08%        0.85        3.81   +1.54   0.82
    $50-200B     7,074   2.84%        0.89        3.68   +1.46   0.80
    >$200B       1,801   2.88%        0.92        3.70   +1.37   0.77

Two-sample KS against the smallest bucket rises monotonically — 0.015, 0.036,
0.051, 0.074 — every one significant, the largest at p = 1.9e-8. So it is not
only scale: the NORMALISED shape differs. Large companies move less, the model
is better calibrated on them (ratio median 0.79 -> 0.92), and their right tail
is far thinner.

Where that lands on a payoff: the >$200B bucket puts **33.9%** of its mass in
the 1-2x band against **26.9%** for <$2B — a 7pp swing into exactly the region
where the twin peaks pay their maximum — and halves the beyond-the-wings tail.
Drawing a mega-cap's simulated move from a pool three-quarters composed of
micro-caps therefore simulates a fatter, worse-calibrated distribution than the
name deserves, which is the leading candidate for why every family's expected
return is over-predicted by 5.9pp to 12.3pp.

**What is deliberately unchanged**: the causality cutoff, the pairing of the
move and crush residuals, the SHA-256 seeding, and the 250-event floor. Only
the set of rows eligible to be drawn narrows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.pnl_sim import MIN_POOL, ResidualPool

__all__ = ["CAP_EDGES", "CAP_LABELS", "bucket_of", "CapResidualPool"]

#: Registered in spec.yaml before the run so they cannot be tuned to the answer.
CAP_EDGES = (0.0, 2e9, 10e9, 50e9, 200e9, float("inf"))
CAP_LABELS = ("<2B", "2-10B", "10-50B", "50-200B", ">200B")


def bucket_of(mcap) -> int:
    """Index of the market-cap bucket, or -1 when the cap is unknown."""
    m = np.asarray(mcap, dtype=float)
    out = np.full(m.shape, -1, dtype=int)
    ok = np.isfinite(m) & (m > 0)
    out[ok] = np.clip(np.searchsorted(CAP_EDGES[1:-1], m[ok], side="right"), 0,
                      len(CAP_LABELS) - 1)
    return out


class CapResidualPool(ResidualPool):
    """``ResidualPool`` conditioned on market cap as well as predicted move.

    The bucket is set per event rather than passed to ``draw``, because
    ``engine.pnl_sim._sim_means`` calls ``draw`` through a fixed signature this
    experiment must not change — EXP-133/134/137/138 all price through it and
    would stop reproducing. The run is single-threaded and sets the bucket
    immediately before each call; ``fallbacks`` records how often the
    intersection was too thin to use, which is a required output because a
    treatment applied to a minority of events is not a treatment.
    """

    def __init__(self, history: pd.DataFrame, buckets: int = 10) -> None:
        super().__init__(history, buckets=buckets)
        if "mcap_usd" not in history.columns:
            raise ValueError("CapResidualPool needs an mcap_usd column")
        h = history.dropna(subset=["event_date", "pred_abs_move", "err_move",
                                   "err_crush"]).sort_values("event_date")
        self._cap = bucket_of(h["mcap_usd"].to_numpy())
        self.current_bucket = -1
        self.fallbacks = {"cap": 0, "decile": 0, "whole": 0}
        self.pool_sizes: list[int] = []

    def on_event(self, row) -> None:
        """Called by ``price_event`` before each event's simulation.

        The market cap of the event being simulated, not of the pool rows —
        that is the whole point. Unknown cap leaves the bucket at -1, which
        cannot match any row and therefore falls back to decile-only, i.e. to
        exactly the existing behaviour.
        """
        self.current_bucket = int(bucket_of(row.get("mcap_usd", float("nan"))))

    def draw(self, cutoff, prediction: float, n: int, rng):
        """Paired residuals from events that are earlier, similar in predicted
        move, AND similar in size. Falls back exactly as the parent does."""
        end = self.before(cutoff)
        if end < MIN_POOL:
            return np.empty(0), np.empty(0)
        pred = self._pred[:end]
        edges = np.quantile(pred, np.linspace(0, 1, self._buckets + 1)[1:-1])
        index = int(np.searchsorted(edges, prediction, side="right"))
        in_decile = np.searchsorted(edges, pred, side="right") == index

        rows = np.flatnonzero(in_decile & (self._cap[:end] == self.current_bucket))
        if rows.size >= MIN_POOL and self.current_bucket >= 0:
            self.fallbacks["cap"] += 1
        else:
            rows = np.flatnonzero(in_decile)
            if rows.size >= MIN_POOL:
                self.fallbacks["decile"] += 1
            else:
                rows = np.arange(end)
                self.fallbacks["whole"] += 1
        self.pool_sizes.append(int(rows.size))
        chosen = rows[rng.integers(0, rows.size, size=n)]
        return self._move[chosen], self._crush[chosen]


def history_with_caps(history: pd.DataFrame, store) -> pd.DataFrame:
    """The residual history, joined to the market cap of each event's year."""
    h = history.copy()
    h["year"] = pd.to_datetime(h["event_date"]).dt.year
    sec = store.read_table("securities", years=range(2013, 2027),
                           columns=["ticker", "year", "mcap_usd"])
    return h.merge(sec, on=["ticker", "year"], how="left")
