"""EXP-190 Q4a: baselines B0, B1 and B2 (design sections 5 and 7).
B0 is the production forecast with a native ResidualPool.

Each event gets the seven quantiles of max(f + err_move, 0) over the rows ``ResidualPool.draw`` samples,
each once (not random draws): the forecast's bucket, or the whole earlier pool when the bucket is under MIN_POOL.
B1 is M0 with the v14 size_distribution.v1 band over the full history, labels before the event's month.
B2 is option-implied: implied_move times trailing-12-month y / implied_move quantiles, from at least 500
donors, labels before the month.
Pure over caller-loaded frames of admitted events only; reads no files. No production caller yet (Q5b/Q6 first).
"""
from collections import Counter

import numpy as np
import pandas as pd

from engine.pnl_sim import MIN_POOL
from engine.v2.research.progress import Progress, decision, stage
from experiments import size_distribution_report as v14
from experiments.size_pnl_pool import build_pool
from experiments.size_quantile_metrics import Rules

BUCKETS = 10  # ResidualPool's default bucket count, as build_pool uses it
LEVELS = Rules().levels
_EXTRA = ["bucket", "fallback", "n_rows", "reason"]
_QCOLS = [f"q{round(a * 100)}" for a in LEVELS]
MIN_IMPLIED_DONORS = 500


class _EveryRow:
    """Stand-in rng: ``draw`` indexes its rows with integers(0, rows.size), so this selects every row once."""

    def integers(self, low, high, size=None):
        return np.arange(low, high)


def _score(pool, pred, cutoff, forecast, cache):
    """(quantiles, bucket, fallback, n_rows, reason) for one event; reason is None when scored.

    ``cache`` maps end -> (edges, counts), where counts is the per-bucket donor count for pred[:end].
    """
    blank = np.full(len(LEVELS), np.nan)
    if not np.isfinite(forecast):
        return blank, np.nan, False, 0, "forecast_missing"
    move, _ = pool.draw(cutoff, forecast, 0, _EveryRow())
    if move.size == 0:
        return blank, np.nan, False, 0, "refused_draw"
    end = pool.before(cutoff)
    if end not in cache:
        edges = np.quantile(pred[:end], np.linspace(0, 1, BUCKETS + 1)[1:-1])
        counts = np.bincount(np.searchsorted(edges, pred[:end], "right"), minlength=BUCKETS)
        cache[end] = (edges, counts)
    edges, counts = cache[end]
    bucket = int(np.searchsorted(edges, forecast, "right"))
    members = int(counts[bucket])
    fallback = members < MIN_POOL
    if move.size != (end if fallback else members):
        raise RuntimeError("B0 bucket labels disagree with ResidualPool.draw")
    quant = np.quantile(np.maximum(forecast + move, 0.0), LEVELS)
    reason = None
    if not np.isfinite(quant).all():
        quant, reason = blank, "non_finite_output"
    return quant, float(bucket), bool(fallback), int(move.size), reason


def _validate(table, labels, events):
    for label, frame in (("table", table), ("labels", labels), ("events", events)):
        if frame.index.has_duplicates:
            raise ValueError(f"{label} has a duplicate index")
    if pd.to_datetime(events["event_date"]).isna().any():
        raise ValueError("events event_date must not be null")


def pool_quantiles(table, labels, events, *, name="B0"):
    """Per-event quantile forecast of ``events`` from the earlier donors in ``table`` and ``labels``.

    Returns a frame indexed like ``events`` with one column per level (q10, q25, ...), then bucket, fallback,
    n_rows and reason. Unscored rows have NaN quantiles.
    """
    _validate(table, labels, events)
    dates = pd.to_datetime(events["event_date"])
    forecasts = [np.nan if v is None else float(v) for v in events["pred_move"]]
    cols = [f"q{round(a * 100)}" for a in LEVELS] + _EXTRA
    rows, reasons, counts, fallbacks = [], [], Counter(), 0
    with stage(f"quantile {name}"):
        pool = build_pool(table, labels, floor=None)
        pred = np.array([r["pred_abs_move"] for r in pool.evidence_rows()], dtype=float)
        cache = {}
        progress = Progress(f"quantile {name}", len(events))
        for cutoff, forecast in zip(dates, forecasts):
            quant, bucket, fallback, n_rows, reason = _score(pool, pred, cutoff, forecast, cache)
            rows.append([*quant, bucket, fallback, n_rows])
            reasons.append(reason)
            counts[reason or "scored"] += 1
            fallbacks += bool(fallback)
            progress.tick()
        decision(f"quantile {name}", f"events={len(events)} scored={counts['scored']} "
                 f"fallback={fallbacks} by_reason={dict(sorted(counts.items()))}")
    out = pd.DataFrame(rows, index=events.index, columns=cols[:-1])
    out = out.astype({"bucket": float, "fallback": bool, "n_rows": int})
    out["reason"] = pd.Series(reasons, index=events.index, dtype=object)
    return out


def _floats(series):
    return pd.to_numeric(series).to_numpy(dtype=float, na_value=np.nan)


def _donors(table, labels, model):
    """Inner join on event id: date, label availability ``avail``, x (the model column), y (abs_move)."""
    joined = table[["event_date", model]].join(labels[["abs_move", "target_available_on"]], how="inner")
    dates = pd.to_datetime(joined["event_date"])
    if dates.isna().any():
        raise ValueError("donor event_date must not be null")
    return pd.DataFrame({"date": dates, "avail": pd.to_datetime(joined["target_available_on"]),
                         "x": _floats(joined[model]), "y": _floats(joined["abs_move"])}, index=joined.index)


def _band_pool(donors, month):
    """B1 pool for ``month``: donors available before it with finite pred and abs_move; None when too small."""
    usable = (donors["avail"] < month).to_numpy()
    finite = np.isfinite(donors["x"].to_numpy()) & np.isfinite(donors["y"].to_numpy())
    found = {"invalid_donors": set(donors.index[usable & ~finite])}
    pool = donors[usable & finite]
    pred, y = pool["x"].to_numpy(), pool["y"].to_numpy()
    if pred.size == 0:
        return None, found
    edges = np.quantile(pred, [0.2, 0.4, 0.6, 0.8])
    bucket = np.digitize(pred, edges)
    counts = np.bincount(bucket, minlength=v14.BUCKETS)
    if counts.min() < v14.MIN_BUCKET_ROWS:
        return None, found
    ratio = y / np.maximum(pred, v14.RATIO_FLOOR)
    bucket_q = [np.quantile(ratio[bucket == b], LEVELS) for b in range(v14.BUCKETS)]
    return (edges, counts, bucket_q), found


def _band_score(state, forecast):
    if not np.isfinite(forecast):
        return None, {}, "forecast_missing"
    if state is None:
        return None, {}, "pool_too_small"
    edges, counts, bucket_q = state
    bucket = int(np.digitize(forecast, edges))
    quant = max(forecast, v14.RATIO_FLOOR) * bucket_q[bucket]
    return quant, {"bucket": float(bucket), "n_rows": int(counts[bucket])}, None


def _implied_pool(donors, month):
    """B2 pool for ``month``: donors in the 12 months before it, labelled before it, with a valid implied_move."""
    in_window = (donors["date"] >= month - pd.DateOffset(months=12)) & (donors["date"] < month)
    window = (in_window & (donors["avail"] < month)).to_numpy()
    implied, y = donors["x"].to_numpy(), donors["y"].to_numpy()
    good = np.isfinite(implied) & (implied > 0)
    label_ok = np.isfinite(y)
    found = {"invalid_implied_donors": set(donors.index[window & ~good]),
             "invalid_label_donors": set(donors.index[window & good & ~label_ok])}
    keep = window & good & label_ok
    return (implied[keep], y[keep], int((window & ~good).sum())), found


def _implied_score(state, forecast):
    pool_x, pool_y, n_dropped = state
    values = {"n_dropped": n_dropped}
    if not (np.isfinite(forecast) and forecast > 0):
        return None, values, "implied_move_invalid"
    if pool_x.size < MIN_IMPLIED_DONORS:
        return None, values, "pool_too_small"
    quant = forecast * np.quantile(pool_y / pool_x, LEVELS)
    return quant, {**values, "n_rows": int(pool_x.size)}, None


def _quantile_frame(name, table, labels, events, *, model, month_pool, score, extra, blank, invalid_keys):
    """Shared B1/B2 path: validate, one donor pool per event month, score each event, log, assemble the frame."""
    _validate(table, labels, events)
    donors = _donors(table, labels, model)
    dates, forecasts = pd.to_datetime(events["event_date"]), _floats(events[model])
    cache, invalid = {}, {key: set() for key in invalid_keys}
    rows, reasons, counts = [], [], Counter()
    with stage(f"quantile {name}"):
        progress = Progress(f"quantile {name}", len(events))
        for cutoff, forecast in zip(dates, forecasts):
            month = cutoff.to_period("M").to_timestamp()
            if month not in cache:
                cache[month], found = month_pool(donors, month)
                for key, ids in found.items():
                    invalid[key].update(ids)
            quant, values, reason = score(cache[month], forecast)
            if reason is None and not np.isfinite(quant).all():
                reason = "non_finite_output"
            if reason is not None:
                quant, values = np.full(len(LEVELS), np.nan), {**values, **blank}
            rows.append([*quant, *(values[col] for col in extra)])
            reasons.append(reason)
            counts[reason or "scored"] += 1
            progress.tick()
        tally = " ".join(f"{key}={len(ids)}" for key, ids in invalid.items())
        decision(f"quantile {name}", f"events={len(events)} scored={counts['scored']} "
                 f"by_reason={dict(sorted(counts.items()))} {tally}")
    out = pd.DataFrame(rows, index=events.index, columns=[*_QCOLS, *extra])
    out = out.astype({**{c: float for c in _QCOLS}, **extra})
    out["reason"] = pd.Series(reasons, index=events.index, dtype=object)
    return out


def band_quantiles(table, labels, events):
    """B1: M0 with the v14 size_distribution.v1 band over every donor available before the event's month."""
    return _quantile_frame("B1", table, labels, events, model="pred_move", month_pool=_band_pool,
                           score=_band_score, extra={"bucket": float, "n_rows": int},
                           blank={"bucket": np.nan, "n_rows": 0}, invalid_keys=("invalid_donors",))


def implied_quantiles(table, labels, events):
    """B2: implied_move times trailing-12-month y / implied_move quantiles, from at least 500 donors."""
    return _quantile_frame("B2", table, labels, events, model="implied_move", month_pool=_implied_pool,
                           score=_implied_score, extra={"n_rows": int, "n_dropped": int},
                           blank={"n_rows": 0}, invalid_keys=("invalid_implied_donors", "invalid_label_donors"))
