#!/usr/bin/env python3
"""How the simulated earnings move is drawn.

``engine.pnl_sim`` draws it additively and clips::

    move = max(pred_abs_move + err_move, 0)

Measured over the 87,121-event residual pool, that is wrong in shape rather
than in location. Mean 5.61% against a realized 5.62%, median 3.72% against
3.70% — but **2.74% of draws land at exactly zero against 0.56% in reality**,
and the skew is +3.43 against a realized +5.00. The clip manufactures a spike
at a dead-flat print and thins the right tail.

That bias has a direction. A spike at zero move is a spike at the CENTRE of the
payoff, where every centre-peaked family (condor, butterfly) pays its maximum
and both twin-peaked families dip. So the clip systematically flatters exactly
the structures that took 78% of EXP-134's book.

**The construction.** Divide realized |move| by predicted |move| and the shape
is a scale family: across a 5.5x range of predicted move the ratio's median
stays 0.80-0.84, its p90 2.15-2.29, its coefficient of variation 0.84-0.93. One
shape rescaled per event is therefore enough — and the scale is the predicted
mean ALONE. The predicted SD adds nothing: ``sd/mean`` spans only 0.80-0.99 and
the realized spread is flat across its quintiles (ratio sd 0.91, 0.90, 0.89,
0.91, 0.89), so rescaling by both would add a parameter that does nothing and
import that model's noise for free.

Fitted on the ratio, loc fixed at 0, judged on the mass in the bands these
structures actually pay over::

    band          <0.5x  0.5-1x   1-2x  2-3.7x  >3.7x   worst error
    empirical     32.6%   25.7%  28.7%   11.5%   1.5%             -
    weibull       32.9%   26.3%  27.4%   11.7%   1.6%         1.3pp
    gamma         33.5%   26.4%  26.6%   11.6%   2.0%         2.1pp
    lognormal     39.6%   24.5%  19.7%   10.1%   6.2%         9.0pp

Lognormal is fitted in log space, where the near-zero moves have huge leverage
(minimum ratio 0, p1 = 0.02). That drags sigma to 1.111 and gives a p99 of 8.88
against a realized 3.99 — it strips 9pp out of the 1-2x band where the twin
peaks pay most and quadruples the beyond-the-wings tail. Rejected on evidence.

**Keeping the pairing.** EXP-129 established that the (move, crush) draw must
stay paired, because the dependence lives in the higher moments where a Pearson
correlation of +0.028 finds nothing. So the parametric models do NOT draw
independently. A pool row is drawn exactly as before — same decile
conditioning, same causality cutoff, same seed — its ratio's rank within the
pool supplies the uniform, and the fitted quantile function is evaluated at that
rank. The crush comes from the SAME row. Only the move's marginal is reshaped;
the copula is untouched.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = ["MOVE_MODELS", "RatioPool", "WEIBULL_PARAMS", "GAMMA_PARAMS"]

#: Fitted on the ratio over the whole pool with loc fixed at 0, and registered
#: in spec.yaml before the run so the shape cannot be re-fitted to taste later.
WEIBULL_PARAMS = (1.1666, 0.0, 1.0978)
GAMMA_PARAMS = (1.2669, 0.0, 0.8252)


class RatioPool:
    """The residual pool, plus each row's realized/predicted ratio and its rank.

    Wraps rather than replaces ``engine.pnl_sim.ResidualPool``: the draw, the
    decile conditioning, the causality cutoff and the seeding all remain its, so
    WHICH rows are drawn is identical across every arm. This adds only the two
    lookups a reshaped marginal needs.
    """

    def __init__(self, pool):
        self.pool = pool
        pred = pool._pred
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(pred > 0, (pred + pool._move) / pred, np.nan)
        self.ratio = np.nan_to_num(ratio, nan=1.0, posinf=1.0)
        # Rank of every row's ratio in the pooled distribution, as a uniform.
        # Pooled rather than per-decile because the shape is scale-stable, which
        # is the measured premise of this whole construction — using it here is
        # the same assumption, not a second one.
        order = np.argsort(self.ratio)
        rank = np.empty(len(order), dtype=float)
        rank[order] = (np.arange(len(order)) + 0.5) / len(order)
        self.rank = rank
        self._order = np.argsort(pool._move)
        self._sorted = pool._move[self._order]

    def rows_for(self, err_move: np.ndarray) -> np.ndarray:
        """Recover which pool rows produced these residuals.

        ``ResidualPool.draw`` returns values rather than indices. Rather than
        change engine code to expose them, the rows are recovered by matching
        the residual value, which is exact: the draw is with replacement from a
        fixed array, so every returned value is one of its entries.
        """
        idx = np.searchsorted(self._sorted, err_move)
        idx = np.clip(idx, 0, len(self._sorted) - 1)
        return self._order[idx]


def additive(pred, err_move, pool, event_date):
    """``engine.pnl_sim``'s own rule, unchanged, for the reference arm."""
    return np.maximum(pred + err_move, 0.0)


def _make(ratio_pool, dist=None, params=None):
    def f(pred, err_move, pool, event_date):
        rows = ratio_pool.rows_for(err_move)
        if dist is None:                       # empirical resampling
            return pred * np.maximum(ratio_pool.ratio[rows], 0.0)
        u = np.clip(ratio_pool.rank[rows], 1e-6, 1 - 1e-6)
        return pred * np.maximum(dist.ppf(u, *params), 0.0)
    return f


def MOVE_MODELS(pool) -> dict:
    """Every registered move model, built against one residual pool."""
    rp = RatioPool(pool)
    return {
        "additive": additive,
        "ratio": _make(rp),
        "weibull": _make(rp, stats.weibull_min, WEIBULL_PARAMS),
        "gamma": _make(rp, stats.gamma, GAMMA_PARAMS),
    }
