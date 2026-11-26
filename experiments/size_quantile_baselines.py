"""EXP-190 Q4a1: baseline B0, the production forecast with a native ResidualPool (design sections 5 and 7).

Each event gets the seven quantiles of max(f + err_move, 0) over the rows ``ResidualPool.draw`` samples,
each once (not random draws): the forecast's bucket, or the whole earlier pool when the bucket is under MIN_POOL.
Pure over caller-loaded frames of admitted events only; reads no files. No production caller yet (Q5b/Q6 first);
B1 and B2 are not here.
"""
from collections import Counter

import numpy as np
import pandas as pd

from engine.pnl_sim import MIN_POOL
from engine.v2.research.progress import Progress, decision, stage
from experiments.size_pnl_pool import build_pool
from experiments.size_quantile_metrics import Rules

BUCKETS = 10  # ResidualPool's default bucket count, as build_pool uses it
LEVELS = Rules().levels
_EXTRA = ["bucket", "fallback", "n_rows", "reason"]


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


def pool_quantiles(table, labels, events, *, name="B0"):
    """Per-event quantile forecast of ``events`` from the earlier donors in ``table`` and ``labels``.

    Returns a frame indexed like ``events`` with one column per level (q10, q25, ...), then bucket, fallback,
    n_rows and reason. Unscored rows have NaN quantiles.
    """
    for label, frame in (("table", table), ("labels", labels), ("events", events)):
        if frame.index.has_duplicates:
            raise ValueError(f"{label} has a duplicate index")
    dates = pd.to_datetime(events["event_date"])
    if dates.isna().any():
        raise ValueError("events event_date must not be null")
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
