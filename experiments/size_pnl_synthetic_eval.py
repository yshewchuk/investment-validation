"""EXP-188 SY2: X1-X7 evaluation and quarter-bootstrap slopes of the synthetic arm (design sections 4-5). SYNTHETIC only."""
import json

import numpy as np
import pandas as pd

from engine.v2.foundation import content_hash
from experiments import size_pnl_contrast as contrast

TYPES = ("median_noise", "mean_noise", "regression_direction")
METRICS = ("rmse", "mae")
STOP_IDS = ("X1", "X2", "X6", "X7")  # spec.yaml:129 and design section 5; I1 is reported but never stops
SHORT, STRADDLE = "short_vol", "straddle"
SHIFTS, RATIO_S = (0.1, -0.1), 0.6
TAIL = 100 * (1 - contrast.INTERVAL) / 2
# Model-dependent geometry/economics outcomes (design sections 4-5): width declined, ladder too coarse, negative floor, entry not a debit.
# On a short-vol row these are non-trades: P&L 0 and the event stays in E; on a straddle row the event leaves E.
# Missing quotes or strikes (entry_quote_unusable, structure_unresolved) are missing facts, not non-trades: they stay undetermined and drop the event.
NON_TRADE_REASONS = frozenset({"width_declined", "ladder_too_coarse", "negative_floor", "entry_not_debit"})


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


def _quarter_var(sub, column="width"):
    """The largest within-quarter population variance of ``column`` (NaN ignored); 0 when there is none."""
    q = pd.to_datetime(sub["event_date"]).dt.to_period("Q").to_numpy()
    v = pd.to_numeric(sub[column]).groupby(q).var(ddof=0)
    return float(v.max()) if v.notna().any() else 0.0


_MARKS = {1: "+", -1: "-", 0: "0"}


def _year_signs(summary):
    """A contrast summary's yearly signs as ``2023:+ 2024:-``, sorted by year."""
    return " ".join(f"{y}:{_MARKS[int(np.sign(v))]}" for y, v in sorted(summary["year_signs"].items()))


def _rung_rows(id_, units, meta, names, sign, arm, resamples, seed):
    """X3/X4: each rung s >= level_gap_min_s, median-minus-mean P&L per named unit, classified against sign.
    Also returns each named unit's contrast summary at that rung (the T11 contrasts table)."""
    rows, summaries = [], []
    for s in sorted({m["params"]["s"] for m in meta.values() if m["family"] == "median_noise"}):
        med, mean = _find(meta, "median_noise", {"s": s}), _find(meta, "mean_noise", {"s": s})
        if s < arm["level_gap_min_s"] or med is None or mean is None or not names: continue
        pairs = {}
        for name in names:
            w, u = units[name]["wide"], units[name]
            res = contrast.contrast(w[med].to_numpy() - w[mean].to_numpy(), u["quarters"], u["tickers"], resamples=resamples, seed=seed)
            pairs[name] = {"inner": res}
            qc = res["quarter_counts"]
            summaries.append([name, f"s={s}", res["mean"], f"{qc['positive']}/{qc['negative']}/{qc['zero']}", *res["quarter_interval"], *res["ticker_interval"], _year_signs(res)])
        rows.append([id_, f"s={s}", contrast.classify(pairs, sign, full_years=tuple(range(arm["inner_years"][0], arm["inner_years"][1] + 1))), f"pairs={len(pairs)}"])
    return rows, summaries


def _safe(rows):
    """Plain JSON-safe rows: floats as float, NaN as None."""
    return [[(None if not np.isfinite(v) else float(v)) if isinstance(v, (float, np.floating)) else v for v in r] for r in rows]


def population(decisions):
    """The evaluated population E (design sections 4-5): the events with no undetermined row under any (model, unit), except short-vol non-trades.
    Only a short-vol row (book == SHORT) whose reason is in NON_TRADE_REASONS is exempt: model-dependent geometry/economics
    outcomes (width declined, ladder too coarse, negative floor, entry not a debit) are non-trades and stay in E with P&L 0.
    Missing quotes or strikes (entry_quote_unusable, structure_unresolved) are undetermined and drop the event under any book.
    Every other undetermined row drops its event, including a straddle row with a non-trade reason (missing fact = undetermined,
    design section 4 (a missing fact is undetermined)), and burn-in or data-missing short-vol rows (unsimulable, thin_history, missing_spread, missing_mcap).
    A missing "book" or "reason" column exempts nothing."""
    undetermined = decisions["status"] == "undetermined"
    if "book" in decisions.columns and "reason" in decisions.columns:
        undetermined &= ~((decisions["book"] == SHORT) & decisions["reason"].isin(NON_TRADE_REASONS))
    bad = set(decisions.loc[undetermined, "event_id"].astype(str))
    return decisions[~decisions["event_id"].astype(str).isin(bad)]


def _means(units, meta, declined):
    """Per (predictor, unit) on E: events, taken and not_taken counts, mean P&L per event (untaken rows 0), the fit errors,
    and the declined share (declined over candidates) of a short-vol unit, None for a straddle unit or when absent."""
    rows = []
    for unit, u in units.items():
        for mid in u["wide"].columns:
            status = u["rows"].loc[u["rows"]["model_id"] == mid, "status"]
            got = declined.get(mid, {}).get(unit) if u["book"] == SHORT else None
            share = got[0] / got[1] if got and got[1] else None
            rows.append([mid, meta[mid]["family"], json.dumps(meta[mid]["params"], sort_keys=True), unit, len(u["wide"]),
                         int((status == "taken").sum()), int((status == "not_taken").sum()),
                         float(u["wide"][mid].mean()), meta[mid]["rmse"], meta[mid]["mae"], share])
    return rows


def evaluate(decisions, meta, tickers, arm, *, replay_equal=None, resamples=contrast.RESAMPLES, seed=contrast.SEED, declined=None):
    """SY2 evaluation of the synthetic decisions on E: quarter-bootstrap slopes, the X1-X7 table, the calibration means
    and contrasts, and the stop ids. ``declined`` is {model: {short-vol unit: [declined, candidates]}}, the width declines."""
    if replay_equal is not None and not isinstance(replay_equal, bool):
        raise ValueError("replay_equal must be a bool or None")
    everything = decisions  # every scored row, including events outside E (the X6 variance)
    decisions = population(decisions)
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
                d = wide[oracle].to_numpy() - wide[mid].to_numpy()
                hi = contrast.bootstrap_interval(d, quarters, resamples=resamples, seed=seed)[2]  # quarter bootstrap only
                signs = contrast.group_signs(d, quarters)
                ok.append(bool(hi >= 0 and sum(v > 0 for v in signs.values()) / len(signs) >= 0.6))
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
            # variance over every scored row of the predictor, including events outside E; NaN checks on E
            if u["book"] == SHORT:  # completeness first: any NaN width fails, before the variance test
                full_sub = everything[(everything["model_id"] == shr) & (everything["book"] == SHORT) & (everything["arm_or_rule"] == "as_is") & (everything["structure"] == unit)]
                declined_width = (sub["reason"] == "width_declined") if "reason" in sub.columns else pd.Series(False, index=sub.index)  # a declined width is never defined
                missing, var = int((sub["width"].isna() & ~declined_width).sum()), _quarter_var(full_sub)
                if missing: x.append(["X6", unit, "fail", f"missing_widths={missing} taken={taken}"])
                else: x.append(["X6", unit, "pass" if var <= 1e-12 else "fail", f"var={var:.3g} taken={taken}"])
            else:  # straddle edge = pred_move - implied_move: constant pred_move per quarter ranks by implied move alone
                full_sub = everything[(everything["model_id"] == shr) & (everything["book"] == STRADDLE) & (everything["arm_or_rule"] == unit)]
                missing, var = int(sub["pred_move"].isna().sum()), _quarter_var(full_sub, "pred_move")
                if missing: x.append(["X6", unit, "fail", f"missing_pred={missing} taken={taken}"])
                else: x.append(["X6", unit, "pass" if var <= 1e-12 else "fail", f"pred_var={var:.3g} taken={taken}"])
    short = [n for n, u in units.items() if u["book"] == SHORT]
    rung3, sums3 = _rung_rows("X3", units, meta, short, 1, arm, resamples, seed)
    rung4, sums4 = _rung_rows("X4", units, meta, [n for n in ("R_L", "R_L2") if n in units], -1, arm, resamples, seed)
    x += rung3 + rung4
    x.append(["X7", "replay", "not_run" if replay_equal is None else ("pass" if replay_equal else "fail"), f"replay_equal={replay_equal}"])
    stop = [f"{r[0]} {r[1]}" for r in x if r[0] in STOP_IDS and r[2] in ("fail", "Failed")]
    slopes, x, means, contrasts = _safe(slopes), _safe(x), _safe(_means(units, meta, declined or {})), _safe(sums3 + sums4)
    hashed = {"slopes": slopes, "x": [r for r in x if r[0] != "X7"], "means": means, "contrasts": contrasts}
    return {"slopes": slopes, "x": x, "means": means, "contrasts": contrasts, "stop": stop, "hash": content_hash(hashed)}
