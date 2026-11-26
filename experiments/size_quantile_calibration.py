"""EXP-190 Q3b: calibration and tails for Q and Q_raw (design sections 4, 5, 7). Pure over caller-loaded frames of
admitted events: no file I/O, no production caller yet (Q5b/Q6 first). The Q6 report must print
`floor_report`; the 0.5 floor is provisional."""
from collections import Counter

import numpy as np
import pandas as pd

from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, decision, stage
from experiments.label_availability import available_before
from experiments.size_quantile_metrics import Rules

LEVELS = Rules().levels
QCOLS = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
WINDOW_MONTHS, MIN_CALIBRATION, MIN_TAIL, FLOOR = 12, 500, 50, 0.5
ARMS = ("Q", "Q_raw")


def predictions_from_receipts(receipts):
    """Each year's own chosen configuration, sorted per row and unfloored, indexed by event id."""
    if not receipts:
        raise ValueError("receipts must not be empty")
    blocks, ids = [], []
    for year in sorted(receipts):
        r = receipts[year]
        if r["status"] != "ok" or list(r["levels"]) != list(LEVELS):
            raise ValueError(f"receipt {year} is not an ok receipt on LEVELS")
        raw = np.asarray(r["raw"]["{}x{}".format(*r["configuration"])], dtype=float)
        if raw.shape != (len(r["event_ids"]), len(QCOLS)):
            raise ValueError(f"receipt {year} raw does not match its event_ids")
        blocks.append(np.sort(raw, axis=1))
        ids.extend(r["event_ids"])
    if len(set(ids)) != len(ids):
        raise ValueError("an event id appears more than once across receipts")
    return pd.DataFrame(np.vstack(blocks), index=ids, columns=QCOLS)


def _inputs(predictions, events):
    if not set(QCOLS) <= set(predictions.columns) or not {"event_date", "y", "label_end"} <= set(events.columns):
        raise ValueError("predictions need the seven QCOLS; events need event_date, y and label_end")
    if predictions.index.has_duplicates or events.index.has_duplicates:
        raise ValueError("predictions and events must have unique indexes")
    if not set(predictions.index) <= set(events.index):
        raise ValueError("every prediction id must be in events")
    ev = events.loc[predictions.index]
    dates = pd.DatetimeIndex(pd.to_datetime(ev["event_date"]))
    y = np.asarray(ev["y"], dtype=float)
    if dates.isna().any() or np.isinf(y).any() or (y < 0).any():
        raise ValueError("event_date must be set; y must be finite or NaN and non-negative")
    if (np.isfinite(y) & ev["label_end"].isna().to_numpy()).any():
        raise ValueError("a finite y needs a label_end")
    return dates, y, ev["label_end"].to_numpy(dtype=object), np.sort(predictions[QCOLS].to_numpy(dtype=float), axis=1)


def _floor_rows(arm, out, dq, donors, years):
    """One floor_report row per scoring year: scored events and distinct donors, with their 0.5 bounds."""
    rows, scored_all = [], np.isfinite(out).all(axis=1)
    for year in sorted(set(years.tolist())):
        scored = scored_all & (years == year)
        keys = [k for k in donors if k[0] == arm and k[1].year == year]
        hi = np.unique(np.concatenate([np.empty(0, dtype=int)] + [donors[k][1] for k in keys]))
        lo = np.unique(np.concatenate([np.empty(0, dtype=int)] + [donors[k][0] for k in keys]))
        row = {"arm": arm, "year": int(year), "events": int(scored.sum()),
               "event_hi_bound": int((scored & (out[:, 6] - out[:, 3] < FLOOR)).sum()),
               "event_lo_bound": int((scored & (out[:, 3] - out[:, 0] < FLOOR)).sum()),
               "donors_hi": int(hi.size), "donors_hi_bound": int((dq[hi, 6] - dq[hi, 3] < FLOOR).sum()),
               "donors_lo": int(lo.size), "donors_lo_bound": int((dq[lo, 3] - dq[lo, 0] < FLOOR).sum())}
        pairs = (("event_hi", "event_hi_bound", "events"), ("event_lo", "event_lo_bound", "events"),
                 ("donor_hi", "donors_hi_bound", "donors_hi"), ("donor_lo", "donors_lo_bound", "donors_lo"))
        rows.append({**row, **{f"{p}_share": row[b] / row[w] if row[w] else float("nan") for p, b, w in pairs}})
    return rows


def calibrate(predictions, events):
    """Return (arms, pools, floor_report) for the Q and Q_raw arms."""
    try:
        with stage("quantile calibration"):
            dates, y, label_end, qhat = _inputs(predictions, events)
            n, width = len(predictions), len(QCOLS)
            finite = np.isfinite(qhat).all(axis=1)
            months, years = dates.to_period("M").to_timestamp(), dates.year.to_numpy()
            donor_q = {a: np.full((n, width), np.nan) for a in ARMS}
            out = {a: np.full((n, width), np.nan) for a in ARMS}
            reason = {a: np.where(finite, None, "non_finite_output") for a in ARMS}
            pools, donors, month_list = {}, {}, months.unique().sort_values()
            progress = Progress("quantile calibration", len(month_list))
            for m in month_list:
                mine = np.flatnonzero(months == m)
                ok = mine[finite[mine]]
                window = (dates >= m - pd.DateOffset(months=WINDOW_MONTHS)) & (dates < m) & np.isfinite(y)
                cut = m.date().isoformat()
                pool = np.array([i for i in np.flatnonzero(window) if available_before(label_end[i], cut)], dtype=int)
                donor_q["Q_raw"][ok] = np.maximum(qhat[ok], 0.0)
                calib = pool[finite[pool]]
                if calib.size < MIN_CALIBRATION:
                    reason["Q"][ok] = "calibration_burn_in"
                else:
                    err = y[calib][:, None] - qhat[calib]
                    delta = np.array([np.quantile(err[:, k], LEVELS[k]) for k in range(width)])
                    fitted = np.maximum(np.sort(qhat[ok] + delta, axis=1), 0.0)
                    good = np.isfinite(fitted).all(axis=1)
                    donor_q["Q"][ok[good]] = fitted[good]
                    reason["Q"][ok[~good]] = "non_finite_output"
                for arm in ARMS:
                    have = mine[np.isfinite(donor_q[arm][mine]).all(axis=1)]
                    if have.size:
                        pool_arm = pool[np.isfinite(donor_q[arm][pool]).all(axis=1)]
                        q, yy = donor_q[arm][pool_arm], y[pool_arm]
                        hi, lo = yy > q[:, 6], yy < q[:, 0]
                        d_hi = np.sort((yy[hi] - q[hi, 6]) / np.maximum(q[hi, 6] - q[hi, 3], FLOOR))
                        d_lo = np.sort((yy[lo] - q[lo, 0]) / np.maximum(q[lo, 3] - q[lo, 0], FLOOR))
                        if min(d_lo.size, d_hi.size) < MIN_TAIL:
                            reason[arm][have] = "tail_pool_thin"
                        else:
                            out[arm][have] = donor_q[arm][have]
                            pools[(arm, m)], donors[(arm, m)] = (d_lo, d_hi), (pool_arm[lo], pool_arm[hi])
                progress.tick()
            arms, frames = {}, []
            for arm in ARMS:
                counts = Counter(r or "scored" for r in reason[arm])
                decision("quantile calibration", f"arm={arm} events={n} scored={counts['scored']} "
                         f"by_reason={dict(sorted(counts.items()))}")
                frame = pd.DataFrame(out[arm], index=predictions.index, columns=QCOLS)
                arms[arm] = frame.assign(month=months.to_numpy(), reason=reason[arm])
                frames += _floor_rows(arm, out[arm], donor_q[arm], donors, years)
            for r in frames:
                decision("quantile calibration", "floor " + " ".join(f"{k}={v}" for k, v in r.items()))
            return arms, pools, pd.DataFrame(frames)
    except MemoryError:
        raise fail("RESOURCE_LIMIT_EXCEEDED", "quantile calibration ran out of memory") from None


def quantile_points(quantiles, d_lo, d_hi):
    """Points (u, x) of the event's floored inverse CDF: lower tail, the seven knots, upper tail."""
    q, d_lo, d_hi = (np.asarray(v, dtype=float) for v in (quantiles, d_lo, d_hi))
    if q.shape != (len(QCOLS),) or d_lo.size < 2 or d_hi.size < 2:
        raise ValueError("need seven quantiles and two or more donors on each side")
    s_lo, s_hi = max(q[3] - q[0], FLOOR), max(q[6] - q[3], FLOOR)
    u_lo = np.linspace(0.0, 0.1, d_lo.size)
    line = q[0] + s_lo * np.quantile(d_lo, np.linspace(0.0, 1.0, d_lo.size))
    cross = np.flatnonzero((line[:-1] < 0.0) & (line[1:] > 0.0))  # the zero floor bites inside these segments
    u_cross = u_lo[cross] + (u_lo[cross + 1] - u_lo[cross]) * -line[cross] / (line[cross + 1] - line[cross])
    upper = q[6] + s_hi * np.quantile(d_hi, np.linspace(0.0, 1.0, d_hi.size))
    u = np.concatenate([np.insert(u_lo, cross + 1, u_cross), np.array(LEVELS), np.linspace(0.9, 1.0, d_hi.size)])
    x = np.concatenate([np.insert(np.maximum(line, 0.0), cross + 1, 0.0), q, upper])
    return u, x
