"""EXP-188 SY2: X1-X7 evaluation and quarter-bootstrap slopes of the synthetic arm (design sections 4-5). SYNTHETIC only."""
import numpy as np
import pandas as pd

from engine.v2.foundation import content_hash
from experiments import size_pnl_contrast as contrast

TYPES = ("median_noise", "mean_noise", "regression_direction")
METRICS = ("rmse", "mae")
STOP_IDS = ("I1", "X1", "X2", "X6", "X7")
SHORT, STRADDLE = "short_vol", "straddle"
SHIFTS, RATIO_S = (0.1, -0.1), 0.6
TAIL = 100 * (1 - contrast.INTERVAL) / 2


def _wide(g):
    """Per-event P&L (NaN -> 0), index event_id, one column per model; a missing cell refuses."""
    frame = g.assign(event_id=g["event_id"].astype(str), pnl=g["pnl_dollars"].fillna(0.0).astype(float))
    wide = frame.pivot(index="event_id", columns="model_id", values="pnl")
    if wide.isna().to_numpy().any(): raise ValueError("a (model, event) cell of the synthetic decisions is missing")
    return wide


def _units(decisions, tickers):
    """Unit name -> book, wide P&L, quarters and tickers in wide's row order, and the unit's rows."""
    short = decisions[(decisions["book"] == SHORT) & (decisions["arm_or_rule"] == "as_is")]
    strad = decisions[decisions["book"] == STRADDLE]
    data = {}
    for book, frame, key in ((SHORT, short, "structure"), (STRADDLE, strad, "arm_or_rule")):
        for unit, g in frame.groupby(key):
            wide = _wide(g)
            dates = {str(e): str(pd.Timestamp(d).to_period("Q")) for e, d in zip(g["event_id"], g["event_date"])}
            data[str(unit)] = {"book": book, "wide": wide, "rows": g, "quarters": [dates[e] for e in wide.index],
                               "tickers": [tickers[e] for e in wide.index]}
    return data


def _find(meta, family, params):
    return next((mid for mid, m in meta.items() if m["family"] == family and m["params"] == params), None)


def _sums(wide, mids, codes):
    grouped = wide[mids].groupby(codes)
    return grouped.sum().to_numpy(), grouped.size().to_numpy().astype(float)


def _boot_means(sums, counts, resamples, seed):
    """Quarter-resampled rung means, (resamples, rungs); one draw per (unit, type)."""
    idx = np.random.default_rng(seed).integers(0, len(counts), (resamples, len(counts)))
    return sums[idx].sum(1) / counts[idx].sum(1)[:, None]


def _slope(x, point, boot):
    """OLS slope of the rung means on x (numpy.polyfit point) with the quarter-bootstrap interval."""
    xc = x - x.mean()
    lo, hi = np.percentile((boot * xc).sum(1) / (xc ** 2).sum(), [TAIL, 100 - TAIL])
    return float(np.polyfit(x, point, 1)[0]), float(lo), float(hi)


def _quarter_var(sub):
    """The largest within-quarter population variance of width (NaN ignored); 0 when there is none."""
    q = pd.to_datetime(sub["event_date"]).dt.to_period("Q").to_numpy()
    v = pd.to_numeric(sub["width"]).groupby(q).var(ddof=0)
    return float(v.max()) if v.notna().any() else 0.0


def _rung_rows(id_, units, meta, names, sign, arm, resamples, seed):
    """X3/X4: each rung s >= level_gap_min_s, median-minus-mean P&L per named unit, classified against sign."""
    rows = []
    for s in sorted({m["params"]["s"] for m in meta.values() if m["family"] == "median_noise"}):
        med, mean = _find(meta, "median_noise", {"s": s}), _find(meta, "mean_noise", {"s": s})
        if s < arm["level_gap_min_s"] or med is None or mean is None or not names: continue
        pairs = {}
        for name in names:
            w, u = units[name]["wide"], units[name]
            pairs[name] = {"inner": contrast.contrast(w[med].to_numpy() - w[mean].to_numpy(), u["quarters"], u["tickers"],
                                                      resamples=resamples, seed=seed)}
        rows.append([id_, f"s={s}", contrast.classify(pairs, sign, full_years=(arm["inner_years"][1],)), f"pairs={len(pairs)}"])
    return rows


def _safe(rows):
    """Plain JSON-safe rows: floats as float, NaN as None."""
    return [[(None if not np.isfinite(v) else float(v)) if isinstance(v, (float, np.floating)) else v for v in r] for r in rows]


def evaluate(decisions, meta, tickers, arm, *, replay_equal=None, resamples=contrast.RESAMPLES, seed=contrast.SEED):
    """SY2 evaluation of the synthetic decisions: quarter-bootstrap slopes, the X1-X7 table and the stop ids."""
    units, slopes, x = _units(decisions, tickers), [], []
    for unit, u in units.items():
        wide, quarters = u["wide"], u["quarters"]
        codes = pd.factorize(np.asarray(quarters))[0]
        spread = float(wide.mean().max() - wide.mean().min())
        x.append(["I1", unit, "fail" if spread < 1e-9 else "pass", f"spread={spread:.3g}"])
        oracle = _find(meta, "oracle", {})
        if oracle is not None:
            ok = []
            for mid in wide.columns:
                if mid == oracle: continue
                s = contrast.contrast(wide[oracle].to_numpy() - wide[mid].to_numpy(), quarters, u["tickers"], resamples=resamples, seed=seed)
                qc = s["quarter_counts"]
                ok.append(bool(s["quarter_interval"][1] >= 0 and qc["positive"] / sum(qc.values()) >= 0.6))
            x.append(["X1", unit, "pass" if all(ok) else "fail", f"failing={ok.count(False)} of {len(ok)}"])
        for typ in TYPES:
            mids = sorted((m for m in wide.columns if meta[m]["family"] == typ), key=lambda m: meta[m]["params"]["s"])
            if not mids: continue
            sums, counts = _sums(wide, mids, codes)
            point, boot = sums.sum(0) / counts.sum(), _boot_means(sums, counts, resamples, seed)
            xs = {metric: np.array([meta[m][metric] for m in mids], dtype=float) for metric in METRICS}
            steps = np.diff(point)
            if (steps <= 0).all(): verdict = "Clear"
            elif (steps > 0).sum() == 1 and _slope(xs["rmse"], point, boot)[0] < 0: verdict = "Directional"
            else: verdict = "Failed"
            x.append(["X2", f"{typ}/{unit}", verdict, f"steps_up={int((steps > 0).sum())}"])
            if len(mids) < 2: continue
            for metric in METRICS: slopes.append([typ, unit, metric, *_slope(xs[metric], point, boot)])
        up, dn = _find(meta, "level_shift", {"shift": SHIFTS[0]}), _find(meta, "level_shift", {"shift": SHIFTS[1]})
        med, mean = _find(meta, "median_noise", {"s": RATIO_S}), _find(meta, "mean_noise", {"s": RATIO_S})
        if None not in (up, dn, med, mean):
            a, b = (wide[up] - wide[dn]).mean(), (wide[med] - wide[mean]).mean()
            x.append(["X5", unit, "pass" if np.sign(a) == np.sign(b) else "fail", f"shift={a:.3g} noise={b:.3g}"])
        shr = _find(meta, "shrinkage", {"L": 1.0})
        if shr is not None:
            sub = u["rows"][u["rows"]["model_id"] == shr]
            taken = int((sub["status"] == "taken").sum())
            if u["book"] == SHORT:
                var = _quarter_var(sub)
                x.append(["X6", unit, "pass" if var <= 1e-12 else "fail", f"var={var:.3g} taken={taken}"])
            else:
                x.append(["X6", unit, "printed", f"taken={taken}"])
    short = [n for n, u in units.items() if u["book"] == SHORT]
    x += _rung_rows("X3", units, meta, short, 1, arm, resamples, seed)
    x += _rung_rows("X4", units, meta, [n for n in ("R_L", "R_L2") if n in units], -1, arm, resamples, seed)
    x.append(["X7", "replay", "not_run" if replay_equal is None else ("pass" if replay_equal else "fail"), f"replay_equal={replay_equal}"])
    stop = [f"{r[0]} {r[1]}" for r in x if r[0] in STOP_IDS and r[2] in ("fail", "Failed")]
    slopes, x = _safe(slopes), _safe(x)
    return {"slopes": slopes, "x": x, "stop": stop, "hash": content_hash({"slopes": slopes, "x": [r for r in x if r[0] != "X7"]})}
