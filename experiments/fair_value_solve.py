"""Fill model F, slice F6b: mid-anchored (model A) price-space fit of the concession constants (design section 9).
The prediction is x_pred = c = min(1, base + spread * term + thin penalties); V enters only the current-F residuals."""
import logging
import time

import numpy as np

from experiments import fair_value_constants as fvk, fair_value_fills as fvf, fair_value_run as fvr

GRID, TIE, GROUPS = np.arange(101) / 100.0, 1e-9, ("unflagged", "size_only", "oi_only", "both", "unknown")
_conc = lambda g, k: fvk.concession(k["C_BASE"], k["C_SPREAD"], k["C_THIN_SIZE"], k["C_THIN_OI"], g["st"], g["ts"], g["to"])  # noqa: E731


def _resid(g, k):
    return g["x"] - _conc(g, k)


def _current_resid(g, k):
    v = g["v"]
    return g["x"] - np.minimum(1.0, v + _conc(g, k) * (1.0 - v))


def _stats(r):
    if len(r) == 0:
        return {"n": 0, **dict.fromkeys(("mean", "median", "mean_abs", "p10", "p90"))}
    return {"n": len(r), "mean": float(np.mean(r)), "median": float(np.median(r)), "mean_abs": float(np.mean(np.abs(r))),
            "p10": float(np.quantile(r, 0.1)), "p90": float(np.quantile(r, 0.9))}


def _block(g, k, ties):
    n = len(g["x"])
    return {"n": n, "fitted": ties is not None, "loss": float(np.sum(np.abs(_resid(g, k)))) / n if n else None, "tie_count": ties}


def _fit_base(u, started):
    losses = np.empty((101, 101))
    for i, base in enumerate(GRID):
        losses[i] = np.mean(np.abs(u["x"] - fvk.concession(base, GRID[:, None], 0.0, 0.0, u["st"], 0.0, 0.0)), axis=1)
        if (i + 1) % 25 == 0:
            rate = (i + 1) / max(time.monotonic() - started, 1e-9)
            logging.info("[fv_fit] base grid %d/101 rate=%.1f/s eta=%.0fs", i + 1, rate, (100 - i) / rate)
    ties = np.argwhere(losses <= losses.min() + TIE)
    pick = ties[np.lexsort((ties[:, 0], ties[:, 0] + ties[:, 1]))[-1]]
    return float(GRID[pick[0]]), float(GRID[pick[1]]), len(ties), float(losses.min())


def _penalty(g, base, spread, size, name, kept):
    if len(g["x"]) < 30:
        logging.info("[fv_fit] %s penalty kept n=%d", name, len(g["x"]))
        return kept, None
    w = GRID[:, None]
    c = fvk.concession(base, spread, w, 0.0, g["st"], 1.0, 0.0) if size else fvk.concession(base, spread, 0.0, w, g["st"], 0.0, 1.0)
    losses = np.mean(np.abs(g["x"] - c), axis=1)
    ties = np.flatnonzero(losses <= losses.min() + TIE)
    logging.info("[fv_fit] %s penalty fitted n=%d minimum=%.6f", name, len(g["x"]), losses.min())
    return float(GRID[ties[-1]]), len(ties)


def solve(d):
    started = time.monotonic()
    logging.info("[fv_fit] solve start used=%d", len(d))
    a = {"x": d["x"].to_numpy(float), "v": d["v"].to_numpy(float), "st": d["spread_term"].to_numpy(float),
         "ts": d["thin_size"].to_numpy(float), "to": d["thin_oi"].to_numpy(float)}
    g = {name: {k: v[d["group"].to_numpy() == name] for k, v in a.items()} for name in GROUPS}
    n_u, base, spread, ties_bs = len(g["unflagged"]["x"]), fvf.C_BASE, fvf.C_SPREAD, None
    if n_u >= 100:
        base, spread, ties_bs, best = _fit_base(g["unflagged"], started)
        logging.info("[fv_fit] base and spread fitted n=%d minimum=%.6f", n_u, best)
    else:
        logging.info("[fv_fit] base and spread kept n=%d", n_u)
    thin_size, ties_size = _penalty(g["size_only"], base, spread, True, "size_only", fvf.C_THIN_SIZE)
    thin_oi, ties_oi = _penalty(g["oi_only"], base, spread, False, "oi_only", fvf.C_THIN_OI)
    constants = fvk.primary_constants(base, spread, thin_size, thin_oi)
    fit = {name: _block(g[name], constants, t) for name, t in (("unflagged", ties_bs), ("size_only", ties_size), ("oi_only", ties_oi))}
    current = {"C_BASE": fvf.C_BASE, "C_SPREAD": fvf.C_SPREAD, "C_THIN_SIZE": fvf.C_THIN_SIZE, "C_THIN_OI": fvf.C_THIN_OI}
    residuals = {"fitted": {name: _stats(_resid(g[name], constants)) for name in GROUPS},
                 "current_unflagged": _stats(_current_resid(g["unflagged"], current))}
    logging.info("[fv_fit] solve end elapsed=%.1f s peak_rss=%.0f MB", time.monotonic() - started, fvr._peak_mb())
    return constants, fit, residuals
