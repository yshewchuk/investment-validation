"""EXP-190 Q4a1: baseline B0, the production forecast with a native ResidualPool (design sections 5 and 7).

Each event gets the seven quantiles of max(f + err, 0), where err is drawn from the rows ``ResidualPool.draw``
samples: the forecast bucket, or the whole earlier pool when the bucket is under MIN_POOL. Pure over caller-loaded
frames of admitted events only; reads no files. No production caller yet (Q5b/Q6 first); B1 and B2 are not here.
"""
from collections import Counter

import numpy as np
import pandas as pd

from engine.v2.research.progress import Progress, decision, stage
from experiments.size_pnl_pool import build_pool
from experiments.size_quantile_metrics import Rules

_EXTRA = ["bucket", "fallback", "n_rows", "reason"]


def _score(pool, err, cutoff, forecast, levels):
    """(quantiles, bucket, fallback, n_rows, reason) for one event; reason is None when scored."""
    blank = np.full(len(levels), np.nan)
    if not np.isfinite(forecast):
        return blank, np.nan, False, 0, "forecast_missing"
    ev = {}
    pool.draw(cutoff, forecast, 0, np.random.default_rng(0), evidence=ev)
    if ev.get("status") != "selected":
        return blank, np.nan, False, 0, "refused_draw"
    rows = np.asarray(ev["fallback_indices"] if ev["fallback_used"] else ev["eligible_indices"], dtype=int)
    quant = np.quantile(np.maximum(forecast + err[rows], 0.0), levels)
    reason = None
    if not np.isfinite(quant).all():
        quant, reason = blank, "non_finite_output"
    return quant, float(ev["bucket_index"]), bool(ev["fallback_used"]), int(rows.size), reason


def pool_quantiles(table, labels, events, *, levels=Rules().levels, name="B0"):
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
    cols = [f"q{round(a * 100)}" for a in levels] + _EXTRA
    rows, reasons, counts, fallbacks = [], [], Counter(), 0
    with stage(f"quantile {name}"):
        pool = build_pool(table, labels, floor=None)
        err = np.array([r["err_move"] for r in pool.evidence_rows()], dtype=float)
        progress = Progress(f"quantile {name}", len(events))
        for cutoff, forecast in zip(dates, forecasts):
            quant, bucket, fallback, n_rows, reason = _score(pool, err, cutoff, forecast, levels)
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
