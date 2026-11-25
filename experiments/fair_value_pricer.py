"""Fair-value pricer for fill model F (guides/fair_value_fill_design.md, sections 2, 4, 10).

American puts by a Cox-Ross-Rubinstein tree; European calls by Black-Scholes.
Dividends are zero: this is a documented approximation. A dividend before expiry
under-values puts and over-values calls. Output is the unclamped model value;
NaN means "no value" and is never replaced by another number.
"""

from __future__ import annotations

import math

import numpy as np

TREE_STEPS = 200
PUT_VOL_FLOOR = 0.01

_ERFC = np.frompyfunc(math.erfc, 1, 1)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * np.asarray(_ERFC(-x / math.sqrt(2.0)), dtype=float)


def _prepare(spot, strike, years, vol, rate):
    """Broadcast inputs to flat float arrays and classify each row."""
    arrays = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (spot, strike, years, vol, rate))
    )
    shape = arrays[0].shape
    s, k, t, v, r = (a.reshape(-1) for a in arrays)
    with np.errstate(invalid="ignore"):
        base = np.isfinite(s) & (s > 0) & np.isfinite(k) & (k > 0)
        expired = base & (t <= 0)
        live = (
            base & (t > 0) & np.isfinite(t)
            & np.isfinite(v) & (v > 0) & np.isfinite(r)
        )
    return shape, s, k, t, v, r, expired, live


def _put_tree(spot, strike, years, vol, rate):
    """Tree values for live rows (vol already floored); NaN where p is not in (0, 1)."""
    n = TREE_STEPS
    with np.errstate(all="ignore"):
        dt = years / n
        vsq = vol * np.sqrt(dt)
        u = np.exp(vsq)
        d = 1.0 / u
        p = (np.exp(rate * dt) - d) / (u - d)
        ok = (p > 0) & (p < 1)

        out = np.full(spot.shape, np.nan)
        s, k, pr, sig = spot[ok], strike[ok], p[ok], vsq[ok]
        q = 1.0 - pr
        disc = np.exp(-rate[ok] * dt[ok])

        def nodes(i):
            m = np.arange(i + 1)
            return s[:, None] * np.exp((2.0 * m - i)[None, :] * sig[:, None])

        values = np.maximum(k[:, None] - nodes(n), 0.0)
        for i in range(n - 1, -1, -1):
            cont = disc[:, None] * (pr[:, None] * values[:, 1:] + q[:, None] * values[:, :-1])
            values = np.maximum(cont, np.maximum(k[:, None] - nodes(i), 0.0))
        out[ok] = values[:, 0]
        return np.where(np.isfinite(out), out, np.nan)


def american_put(spot, strike, years, vol, rate) -> np.ndarray:
    """American put value per row: CRR tree, TREE_STEPS steps, early exercise at every step.

    Years <= 0 is expired (intrinsic). Vol is floored at PUT_VOL_FLOOR.
    """
    shape, s, k, t, v, r, expired, live = _prepare(spot, strike, years, vol, rate)
    out = np.full(s.shape, np.nan)
    out[expired] = np.maximum(k[expired] - s[expired], 0.0)
    out[live] = _put_tree(
        s[live], k[live], t[live], np.maximum(v[live], PUT_VOL_FLOOR), r[live]
    )
    return out.reshape(shape)


def european_call(spot, strike, years, vol, rate) -> np.ndarray:
    """European call value per row: Black-Scholes, zero dividends, any rate sign.

    Years <= 0 is expired (intrinsic). Vol is not floored.
    """
    shape, s, k, t, v, r, expired, live = _prepare(spot, strike, years, vol, rate)
    out = np.full(s.shape, np.nan)
    out[expired] = np.maximum(s[expired] - k[expired], 0.0)
    with np.errstate(all="ignore"):
        sl, kl, tl, vl, rl = s[live], k[live], t[live], v[live], r[live]
        vs = vl * np.sqrt(tl)
        d1 = (np.log(sl / kl) + (rl + 0.5 * vl * vl) * tl) / vs
        d2 = d1 - vs
        out[live] = sl * _norm_cdf(d1) - kl * np.exp(-rl * tl) * _norm_cdf(d2)
    return out.reshape(shape)
