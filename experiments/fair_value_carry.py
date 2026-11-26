"""Carry and dividend pricer (design section 4, slice F7). NaN means no value; unclamped.

A dividend-free call with carry >= rate >= 0 is European with carry b; every other live row is an
escrowed American CRR tree. Puts and tree calls floor vol at PUT_VOL_FLOOR, as fair_value_pricer does.
"""

from __future__ import annotations

import numpy as np

from experiments.fair_value_pricer import PUT_VOL_FLOOR, TREE_STEPS, _norm_cdf, _prepare


def carry_rate(days, borrow30, borrow2yr, rate):
    """(b, fallback): borrow30 to 30 days, linear to borrow2yr at 730, flat beyond; one missing uses the other."""
    days, b30, b2 = (np.asarray(a, dtype=float) for a in (days, borrow30, borrow2yr))
    with np.errstate(all="ignore"):
        w = np.clip((days - 30.0) / 700.0, 0.0, 1.0)
        single = np.where(np.isfinite(b30), b30, b2)
        interp = np.where(np.isfinite(b30) & np.isfinite(b2), b30 + w * (b2 - b30), single)
    fallback = ~np.isfinite(b30) & ~np.isfinite(b2)
    return np.where(fallback, rate, interp), fallback


def dividend(div_amt, div_days, expiry_days, spot, rate):
    """(D, fallback, applied): the first case that holds decides; D is div_amt where applied, else 0.0."""
    amt, dd, ex, s, r = (np.asarray(a, dtype=float) for a in (div_amt, div_days, expiry_days, spot, rate))
    with np.errstate(all="ignore"):
        bad = ~np.isfinite(amt) | (amt < 0)
        pre = ~bad & (amt != 0)
        no_day = np.isnan(dd) | (dd <= 0)
        late = dd > ex
        covered = amt * np.exp(-r * dd / 365.0) >= s
    applied = pre & ~no_day & ~late & ~covered
    fallback = bad | (pre & (no_day | (~late & covered)))
    return np.where(applied, amt, 0.0), fallback, applied


def _tree(spot, strike, years, vol, rate, carry, div, div_years, sign, early=True):
    """Escrowed American CRR values; NaN where p is not in (0, 1). early=False drops exercise in backward steps."""
    n = TREE_STEPS
    with np.errstate(all="ignore"):
        dt = years / n
        vsq = vol * np.sqrt(dt)
        u = np.exp(vsq)
        p = (np.exp(carry * dt) - 1.0 / u) / (u - 1.0 / u)
    ok = (p > 0) & (p < 1)
    s, k, t, r, dv, dy, sg, pr, sig, h = (a[ok] for a in (spot, strike, years, rate, div, div_years, sign, p, vsq, dt))
    with np.errstate(all="ignore"):
        s = s - dv * np.exp(-r * dy)
        q, disc = 1.0 - pr, np.exp(-r * h)
        ladder = s[:, None] * np.exp(np.arange(-n, n + 1, dtype=float)[None, :] * sig[:, None])

        def payoff(i):
            ti = t * (i / n)
            esc = np.where(ti < dy, dv * np.exp(-r * (dy - ti)), 0.0)
            return np.maximum(sg[:, None] * (ladder[:, n - i : n + i + 1 : 2] + esc[:, None] - k[:, None]), 0.0)

        values = payoff(n)
        for i in range(n - 1, -1, -1):
            cont = disc[:, None] * (pr[:, None] * values[:, 1:] + q[:, None] * values[:, :-1])
            values = np.maximum(cont, payoff(i)) if early else cont
    out = np.full(spot.shape, np.nan)
    out[ok] = values[:, 0]
    return out


def value(spot, strike, years, vol, rate, carry, div, div_years, is_put):
    """Value per row: expired rows are intrinsic, ignoring carry and dividend; live rows need finite carry."""
    cols = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (spot, strike, years, vol, rate, carry, div, div_years, is_put)))
    shape = cols[0].shape
    s, k, t, v, r, b, dv, dy, put = (a.reshape(-1) for a in cols)
    _, s, k, t, v, r, expired, live = _prepare(s, k, t, v, r)
    live = live & np.isfinite(b)
    sign = np.where(put != 0, -1.0, 1.0)
    euro = live & (put == 0) & (dv == 0) & (b >= r) & (r >= 0)
    tree = live & ~euro
    out = np.where(expired, np.maximum(sign * (s - k), 0.0), np.nan)
    with np.errstate(all="ignore"):
        vs = v[euro] * np.sqrt(t[euro])
        d1 = (np.log(s[euro] / k[euro]) + (b[euro] + 0.5 * v[euro] * v[euro]) * t[euro]) / vs
        out[euro] = s[euro] * np.exp((b[euro] - r[euro]) * t[euro]) * _norm_cdf(d1) - k[euro] * np.exp(-r[euro] * t[euro]) * _norm_cdf(d1 - vs)
    out[tree] = _tree(s[tree], k[tree], t[tree], np.maximum(v[tree], PUT_VOL_FLOOR), r[tree], b[tree], dv[tree], dy[tree], sign[tree])
    return out.reshape(shape)
