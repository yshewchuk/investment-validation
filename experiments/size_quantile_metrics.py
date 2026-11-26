"""EXP-190 Q5: per-event and per-cell metrics (design section 5).

This slice holds the per-event and per-cell metrics only: pinball, coverage, CRPS,
crossing rate, level-only width L and within-cell scores. A later slice adds tables,
bootstraps and the pass rule. The settlements applied here are "supervisor-recommended
2026-11-25" (design section 5).

Pure functions over numpy arrays and pandas DataFrames: no I/O, no logging, no
engine imports. Pinball is numpy, since scikit-learn's mean_pinball_loss returns
only an aggregate and we need per-event losses.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

# (lower level, upper level, nominal coverage) for each central band.
BANDS = {"p25_p75": (0.25, 0.75, 0.5), "p10_p90": (0.10, 0.90, 0.8)}


@dataclass(frozen=True)
class Rules:
    """The design's defaults (design section 5). Not tuned here.

    Stage A settings (resamples, seed, periods, pass thresholds) arrive with the
    Stage A module.
    """

    levels: tuple = (0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90)
    min_cell: int = 30
    zero_tol: float = 1e-9
    level_bins: int = 20
    events_per_bin: int = 100
    min_level_history: int = 500


def level_index(levels, value):
    """Column index of a level value; ValueError if absent."""
    for k, a in enumerate(levels):
        if abs(a - value) < 1e-12:
            return k
    raise ValueError(f"level {value} is missing from levels")


def pinball(y, q, levels):
    """Per-event pinball loss per level, shape (n, L): max(a*(y-q), (a-1)*(y-q))."""
    a = np.asarray(levels, float)
    d = np.asarray(y, float)[:, None] - q
    return np.maximum(a * d, (a - 1) * d)


def coverage(y, q, levels):
    """Per-event flags: inclusive p25-p75 and p10-p90 bands, and y <= q_k per level."""
    y = np.asarray(y, float)
    out = {
        name: (y >= q[:, level_index(levels, lo)]) & (y <= q[:, level_index(levels, hi)])
        for name, (lo, hi, _) in BANDS.items()
    }
    for k, a in enumerate(levels):
        out[f"le_p{round(a * 100)}"] = y <= q[:, k]
    return pd.DataFrame(out)


def crps_quantile(y, u, x):
    """Exact CRPS of a piecewise-linear quantile function through points (u, x).

    Each segment is split where Q crosses y, so the indicator is constant on each
    piece and the integrand is quadratic: Simpson's rule on a piece is exact.
    """
    u = np.asarray(u, float)
    x = np.asarray(x, float)
    if u.ndim != 1 or u.shape != x.shape or u.size < 2:
        raise ValueError("u and x must be 1-D, of equal length, with at least 2 points")
    if not (np.isfinite(u).all() and np.isfinite(x).all() and np.isfinite(y)):
        raise ValueError("u, x and y must be finite")
    if u[0] != 0 or u[-1] != 1:
        raise ValueError("u must run from 0 to 1")
    if np.any(np.diff(u) < 0) or np.any(np.diff(x) < 0):
        raise ValueError("u and x must be nondecreasing")
    total = 0.0
    for a, b, x0, x1 in zip(u[:-1], u[1:], x[:-1], x[1:]):
        if b <= a:
            continue
        cuts = [a, b]
        if x0 < y < x1:
            cuts = [a, a + (b - a) * (y - x0) / (x1 - x0), b]
        for p, r in zip(cuts[:-1], cuts[1:]):
            if r <= p:
                continue
            m = 0.5 * (p + r)
            ind = float(y < x0 + (x1 - x0) * (m - a) / (b - a))
            gp, gm, gr = (
                (ind - t) * (x0 + (x1 - x0) * (t - a) / (b - a) - y) for t in (p, m, r))
            total += (r - p) / 6 * (gp + 4 * gm + gr)
    return 2 * total


def crossing_rate(raw):
    """Share of rows where any neighbouring pair is strictly out of order; NaN if empty."""
    raw = np.asarray(raw, float)
    if raw.shape[0] == 0:
        return float("nan")
    return float(np.mean(np.any(raw[:, :-1] > raw[:, 1:], axis=1)))


def level_width(month_start, q50, history_q50, history_y, history_label_available,
                rules=Rules()):
    """Level-only width L per event (issue #2871), causal by month.

    For each month, history is the finite (q50, y) pairs labelled strictly before
    the month starts. Returns (L, reason) arrays aligned with the events.
    """
    ms = np.asarray(month_start, dtype="datetime64[ns]")
    if np.isnat(ms).any():
        raise ValueError("current month_start must not be NaT")
    months = ms.astype("datetime64[M]")
    q = np.asarray(q50, float)
    if not np.isfinite(q).all():
        raise ValueError("current q50 must be finite")
    hq = np.asarray(history_q50, float)
    hy = np.asarray(history_y, float)
    hl = np.asarray(history_label_available, dtype="datetime64[ns]")
    if (ms.ndim != 1 or q.ndim != 1 or ms.shape != q.shape
            or hq.ndim != 1 or hq.shape != hy.shape or hq.shape != hl.shape):
        raise ValueError("level_width inputs must be 1-D and of equal length")
    keep = np.isfinite(hq) & np.isfinite(hy)
    width = np.full(q.shape, np.nan)
    reason = np.full(q.shape, "", dtype=object)
    for mo in np.unique(months):
        ev = months == mo
        hist = keep & (hl < mo.astype("datetime64[ns]"))
        n = int(hist.sum())
        if n < rules.min_level_history:
            reason[ev] = "level_width_burn_in"
            continue
        m = max(1, min(rules.level_bins, n // rules.events_per_bin))
        hq_in, hy_in = hq[hist], hy[hist]
        edges = np.unique(np.quantile(hq_in, np.linspace(0, 1, m + 1)))
        h_bin = np.searchsorted(edges[1:-1], hq_in, side="left")
        e_bin = np.searchsorted(edges[1:-1], q[ev], side="left")
        rows_all = np.flatnonzero(ev)
        for b in np.unique(e_bin):
            rows = rows_all[e_bin == b]
            in_bin = h_bin == b
            if not in_bin.any():
                reason[rows] = "level_width_empty_bin"
                continue
            p25, p75 = np.quantile(hy_in[in_bin], [0.25, 0.75])
            width[rows] = p75 - p25
    return width, reason


def _resid(v, rq):
    """Residual of rank(v) after least squares on rank(q50) with an intercept."""
    r = rankdata(v)
    if np.ptp(rq) == 0:
        return r - r.mean()
    design = np.column_stack([np.ones_like(rq), rq])
    beta = np.linalg.lstsq(design, r, rcond=None)[0]
    return r - design @ beta


def cell_scores(event_id, cell, e, q50, score, rules=Rules()):
    """Within-cell partial-Spearman contributions and tercile differences for one score.

    Returns (per_event, per_cell) in input row order. The skip reasons never depend
    on score, so W and L always use the same cells.
    W and L (any two scores compared) must be scored on the same rows, in the same order.
    """
    event_id = np.asarray(event_id)
    e = np.asarray(e, float)
    q50 = np.asarray(q50, float)
    score = np.asarray(score, float)
    if len({len(event_id), len(cell), e.size, q50.size, score.size}) != 1:
        raise ValueError("event_id, cell, e, q50 and score must have equal length")
    if not (np.isfinite(e).all() and np.isfinite(q50).all() and np.isfinite(score).all()):
        raise ValueError("e, q50 and score must be finite")
    cells = {}
    for pos, c in enumerate(cell):
        cells.setdefault(c, []).append(pos)
    used = np.zeros(e.size, bool)
    contrib = np.full(e.size, np.nan)
    terc = np.full(e.size, np.nan)
    rows = []
    for c, pos in cells.items():
        idx = np.array(pos)
        n = idx.size
        ee = e[idx]
        rq = rankdata(q50[idx])
        skip = ""
        re = None
        if n < rules.min_cell:
            skip = "cell_small"
        elif np.ptp(ee) == 0:
            skip = "constant_e"
        else:
            re = _resid(ee, rq)
            if np.all(np.abs(re) <= rules.zero_tol):
                skip = "zero_e_residual"
        if skip:
            rows.append({"cell": c, "n": n, "skip": skip,
                         "correlation": np.nan, "tercile_diff": np.nan})
            continue
        rs = _resid(score[idx], rq)
        zero = bool(np.all(np.abs(rs) <= rules.zero_tol))
        if zero:
            rs = np.zeros(n)
            corr, diff = 0.0, 0.0
            contrib[idx] = 0.0
        else:
            denom = np.sqrt(np.sum(rs ** 2) * np.sum(re ** 2))
            corr = float(np.sum(rs * re) / denom)
            contrib[idx] = n * rs * re / denom
        order = np.lexsort((event_id[idx], rs))
        parts = np.array_split(order, 3)
        for k, part in enumerate(parts):
            terc[idx[part]] = k
        if not zero:
            diff = float(ee[parts[2]].mean() - ee[parts[0]].mean())
        used[idx] = True
        rows.append({"cell": c, "n": n, "skip": "", "correlation": corr, "tercile_diff": diff})
    per_event = pd.DataFrame({"used": used, "contribution": contrib, "tercile": terc})
    per_cell = pd.DataFrame(rows, columns=["cell", "n", "skip", "correlation", "tercile_diff"])
    return per_event, per_cell
