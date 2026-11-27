"""EXP-190 Q3b: calibration and tails for Q and Q_raw (design sections 4, 5, 7), and CRPS for Q, Q_raw and B0 (Q6b).
Pure over caller-loaded frames of admitted events: no file I/O. `month_tables` is the one tail rule: calibrate uses it
for Q and Q_raw, score_crps for every other frame (B0, and S, B1-B3, also scored). A thin tail drops only CRPS. Print
`floor_report`; the 0.5 floor is provisional."""
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

from engine.v2.ops.errors import fail
from engine.v2.research.progress import Progress, decision, stage
from experiments.label_availability import available_before, printed_before
from experiments.size_quantile_metrics import Rules, crps_quantile

LEVELS = Rules().levels
QCOLS = ["q10", "q25", "q40", "q50", "q60", "q75", "q90"]
WINDOW_MONTHS, MIN_CALIBRATION, MIN_TAIL, FLOOR = 12, 500, 30, 0.5
ARMS = ("Q", "Q_raw")
PERIOD = (pd.Timestamp("2023-07-01"), pd.Timestamp("2025-12-31"))


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
    need = {"event_date", "y", "label_end", "available"}
    if not set(QCOLS) <= set(predictions.columns) or not need <= set(events.columns):
        raise ValueError("predictions need the seven QCOLS; events need event_date, y, label_end and available")
    if predictions.index.has_duplicates or events.index.has_duplicates:
        raise ValueError("predictions and events must have unique indexes")
    if not set(predictions.index) <= set(events.index):
        raise ValueError("every prediction id must be in events")
    ev = events.loc[predictions.index]
    dates = pd.DatetimeIndex(pd.to_datetime(ev["event_date"]))
    y = np.asarray(ev["y"], dtype=float)
    if dates.isna().any() or np.isinf(y).any() or (y < 0).any():
        raise ValueError("event_date must be set; y must be finite or NaN and non-negative")
    label_end, available = (ev[c].to_numpy(dtype=object) for c in ("label_end", "available"))
    try:
        for end, day in zip(label_end[np.isfinite(y)], available[np.isfinite(y)]):
            date.fromisoformat(end)
            date.fromisoformat(day)
    except (TypeError, ValueError):
        raise ValueError("a finite y needs a label_end and available that are ISO dates") from None
    return dates, y, label_end, available, np.sort(predictions[QCOLS].to_numpy(dtype=float), axis=1)


def donor_pool(dates, y, label_end, available, m):
    """Positions dated in [m - 12 months, m) with finite y, a label available before month m (label_end + 1 day) and
    a native availability printed before m. Both are read only for those rows (finite y, in the window)."""
    cutoff = f"{m:%Y-%m-%d}"
    window = (dates >= m - pd.DateOffset(months=WINDOW_MONTHS)) & (dates < m) & np.isfinite(y)
    return np.array([i for i in np.flatnonzero(window)
                     if available_before(label_end[i], cutoff) and printed_before(available[i], cutoff)], dtype=int)


def month_tables(quantiles, y, dates, label_end, available, months):
    """(pools, donors, thin) keyed (arm, month) for {arm: quantiles}: the one tail rule. Each month's pool rows are the
    donors with finite quantiles; a tail with under MIN_TAIL donors makes the month thin: no entry, counted in thin."""
    pools, donors, thin = {}, {}, Counter()
    for m in months.unique().sort_values():
        pool = donor_pool(dates, y, label_end, available, m)
        for arm, q in quantiles.items():
            if not np.isfinite(q[months == m]).all(axis=1).any():
                continue
            keep = pool[np.isfinite(q[pool]).all(axis=1)]
            qq, yy = q[keep], y[keep]
            hi, lo = yy > qq[:, 6], yy < qq[:, 0]
            d_hi = np.sort((yy[hi] - qq[hi, 6]) / np.maximum(qq[hi, 6] - qq[hi, 3], FLOOR))
            d_lo = np.sort((yy[lo] - qq[lo, 0]) / np.maximum(qq[lo, 3] - qq[lo, 0], FLOOR))
            if min(d_lo.size, d_hi.size) < MIN_TAIL:
                thin[arm] += 1
            else:
                pools[(arm, m)], donors[(arm, m)] = (d_lo, d_hi), (keep[lo], keep[hi])
    return pools, donors, thin


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
            dates, y, label_end, available, qhat = _inputs(predictions, events)
            n, width = len(predictions), len(QCOLS)
            finite = np.isfinite(qhat).all(axis=1)
            months, years = dates.to_period("M").to_timestamp(), dates.year.to_numpy()
            donor_q = {a: np.full((n, width), np.nan) for a in ARMS}
            reason = {a: np.where(finite, None, "non_finite_output") for a in ARMS}
            month_list = months.unique().sort_values()
            progress = Progress("quantile calibration", len(month_list))
            for m in month_list:
                mine = np.flatnonzero(months == m)
                ok = mine[finite[mine]]
                pool = donor_pool(dates, y, label_end, available, m)
                donor_q["Q_raw"][ok] = np.maximum(qhat[ok], 0.0)
                calib = pool[finite[pool]]
                if calib.size < MIN_CALIBRATION:
                    reason["Q"][ok] = "calibration_burn_in"
                else:
                    delta = np.array([np.quantile(y[calib] - qhat[calib, k], LEVELS[k]) for k in range(width)])
                    fitted = np.maximum(np.sort(qhat[ok] + delta, axis=1), 0.0)
                    good = np.isfinite(fitted).all(axis=1)
                    donor_q["Q"][ok[good]] = fitted[good]
                    reason["Q"][ok[~good]] = "non_finite_output"
                progress.tick()
            pools, donors, thin = month_tables(donor_q, y, dates, label_end, available, months)
            arms, frames = {}, []
            for arm in ARMS:
                counts = Counter(r or "scored" for r in reason[arm])
                decision("quantile calibration", f"arm={arm} events={n} scored={counts['scored']} "
                         f"by_reason={dict(sorted(counts.items()))} thin_months={thin[arm]}")
                arms[arm] = pd.DataFrame(donor_q[arm], predictions.index, QCOLS).assign(
                    month=months, reason=reason[arm])
                frames += _floor_rows(arm, donor_q[arm], donor_q[arm], donors, years)
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


def score_crps(frames, events, calibrated):
    """(frames with a crps column, Q3 CRPS exclusion rows, floor rows of the frames not in ARMS). An ARMS frame uses
    calibrate's pools (`calibrated`); any other frame builds its own pools from its quantiles with month_tables."""
    pools, out, exclusions, floors = dict(calibrated), {}, [], []
    for name, frame in frames.items():
        if not set(QCOLS + ["reason"]) <= set(frame.columns):
            raise ValueError(f"{name} frame needs the seven QCOLS and reason")
        dates, y, label_end, available, _ = _inputs(frame, events)
        months, q = dates.to_period("M").to_timestamp(), frame[QCOLS].to_numpy(float)
        if name not in ARMS:
            own, donors, _ = month_tables({name: q}, y, dates, label_end, available, months)
            pools.update(own)
            floors += _floor_rows(name, q, q, donors, dates.year.to_numpy())
        finite = np.isfinite(q).all(axis=1)
        ok = frame["reason"].isna().to_numpy() & finite & np.isfinite(y)
        pooled = np.array([(name, m) in pools for m in months], dtype=bool)
        crps = np.full(len(frame), np.nan)
        todo = np.flatnonzero(ok & pooled)
        progress = Progress(f"stage A CRPS {name}", int(todo.size))
        for k in todo:
            crps[k] = crps_quantile(y[k], *quantile_points(q[k], *pools[(name, months[k])]))
            progress.tick()
        thin = len({m for m in months[finite] if (name, m) not in pools})
        decision("stage A CRPS", f"arm={name} scored={int(ok.sum())} thin_months={thin}")
        out[name] = frame.assign(crps=crps)
        inper = ok & (dates >= PERIOD[0]) & (dates <= PERIOD[1])
        exclusions += [{"arm": name, "year": yr, "scored": int((inper & (dates.year == yr)).sum()),
                        "crps_excluded": int((inper & (dates.year == yr) & ~pooled).sum())}
                       for yr in (2023, 2024, 2025)]
    return out, exclusions, floors
