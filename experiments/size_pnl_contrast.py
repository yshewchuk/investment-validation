"""Contrast classifier for EXP-188 slice S8a.

Implements design guides/size_decision_pnl_design.md section 7: paired P&L
contrasts (median-type minus mean-type, or any other matched-pair contrast)
are summarised by quarter, year and ticker, and each predicted-sign pair of
arms is classified as clear, directional, no_difference, reversed or
not_testable. classify works on hypothesis-level pair/arm summaries (one
contrast() result per pair and arm), not on raw per-event arrays.

Pure functions, no data access: callers pass per-event arrays and receive
summaries and a category label. No I/O, no logging, no engine imports.
"""

import numpy as np

CLEAR_SHARE = 0.75
DIRECTIONAL_SHARE = 0.60
RESAMPLES = 10000
INTERVAL = 0.90
SEED = 188
FULL_YEARS = (2024, 2025)
_CHUNK = 500


def _check(d, *labels):
    arr = np.asarray(d, dtype=float)
    n = len(arr)
    if any(len(lab) != n for lab in labels):
        raise ValueError("d and labels must have equal length")
    if n == 0:
        raise ValueError("empty input")
    if not np.all(np.isfinite(arr)):
        raise ValueError("d must be finite")
    return arr


def _totals(d, labels):
    keys, inv = np.unique(np.asarray(labels), return_inverse=True)
    inv = inv.reshape(-1)
    sums = np.bincount(inv, weights=d, minlength=len(keys))
    counts = np.bincount(inv, minlength=len(keys))
    return keys.tolist(), sums, counts


def group_signs(d, labels):
    """Map each sorted label to the sign (-1/0/1) of its sum of d."""
    d = _check(d, labels)
    keys, sums, _ = _totals(d, labels)
    return {k: int(np.sign(s)) for k, s in zip(keys, sums)}


def bootstrap_interval(d, labels, *, resamples=RESAMPLES, interval=INTERVAL, seed=SEED):
    """Paired cluster bootstrap of the event-weighted mean, resampling groups.

    Returns (mean, lo, hi). Each resample draws whole groups (unique labels)
    with replacement, so events sharing a label stay together.
    """
    if resamples < 1:
        raise ValueError("resamples must be at least 1")
    if not 0.0 <= interval < 1.0:
        raise ValueError("interval must satisfy 0 <= interval < 1")
    d = _check(d, labels)
    _, sums, counts = _totals(d, labels)
    g = len(sums)
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples)
    for start in range(0, resamples, _CHUNK):
        k = min(_CHUNK, resamples - start)
        idx = rng.integers(0, g, size=(k, g))
        stats[start:start + k] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    tail = 50.0 * (1.0 - interval)
    lo, hi = np.percentile(stats, [tail, 100.0 - tail])
    return float(d.mean()), float(lo), float(hi)


def contrast(d, quarters, tickers, *, resamples=RESAMPLES, interval=INTERVAL, seed=SEED):
    """Summarise one contrast series by quarter, year and ticker."""
    d = _check(d, quarters, tickers)
    quarter_signs = group_signs(d, quarters)
    year_signs = group_signs(d, [int(q[:4]) for q in quarters])
    vals = list(quarter_signs.values())
    _, q_lo, q_hi = bootstrap_interval(
        d, quarters, resamples=resamples, interval=interval, seed=seed)
    _, t_lo, t_hi = bootstrap_interval(
        d, tickers, resamples=resamples, interval=interval, seed=seed)
    return {
        "n_events": int(d.size),
        "mean": float(d.mean()),
        "quarter_signs": quarter_signs,
        "year_signs": year_signs,
        "quarter_interval": (q_lo, q_hi),
        "ticker_interval": (t_lo, t_hi),
        "quarter_counts": {
            "positive": sum(v > 0 for v in vals),
            "negative": sum(v < 0 for v in vals),
            "zero": sum(v == 0 for v in vals),
        },
    }


def _side(lo, hi, sign):
    if sign > 0:
        return lo > 0
    if sign < 0:
        return hi < 0
    return False


def pair_status(summary, predicted_sign, *, share, full_years=FULL_YEARS):
    """Status of one arm summary against the predicted sign at a given share."""
    qs = list(summary["quarter_signs"].values())
    year_signs = summary["year_signs"]
    quarter_share = sum(s == predicted_sign for s in qs) / len(qs)
    years_ok = all(y in year_signs and year_signs[y] == predicted_sign for y in full_years)
    intervals_ok = (_side(*summary["quarter_interval"], predicted_sign)
                    and _side(*summary["ticker_interval"], predicted_sign))
    pair_sign = int(np.sign(summary["mean"]))
    return {
        "pair_sign": pair_sign,
        "quarter_share": quarter_share,
        "years_ok": bool(years_ok),
        "intervals_exclude_zero": bool(intervals_ok),
        "meets_share": bool(quarter_share >= share),
        "predicted": bool(pair_sign == predicted_sign),
    }


def _clearly_opposite(summary, sign, full_years):
    st = pair_status(summary, -sign, share=CLEAR_SHARE, full_years=full_years)
    return st["predicted"] and st["years_ok"] and st["meets_share"] and st["intervals_exclude_zero"]


def _level(summaries, sign, share, require_intervals, full_years):
    """True if one arm's pair summaries show the sign at the given share.

    A pair is "used" (qualifies) when it has the predicted sign, years_ok,
    meets_share and, if require_intervals, intervals_exclude_zero. Needs at
    least 2 used pairs and no pair clearly opposite. Other pairs (predicted
    sign but weak consistency) do not block.
    """
    used = 0
    for s in summaries:
        st = pair_status(s, sign, share=share, full_years=full_years)
        if not (st["predicted"] and st["years_ok"] and st["meets_share"]):
            continue
        if require_intervals and not st["intervals_exclude_zero"]:
            continue
        used += 1
    if used < 2:
        return False
    return not any(_clearly_opposite(s, sign, full_years) for s in summaries)


def classify(pairs, predicted_sign, *, full_years=FULL_YEARS):
    """Classify {pair_id: {arm: summary-or-None}} against predicted_sign.

    Returns one of "clear", "directional", "no_difference", "reversed",
    "not_testable". Returns "not_testable" if there are fewer than 2 pairs or
    if any pair has a None arm (an excluded model) (design section 7).

    A level holds when at least 2 pairs are "used" (qualify: predicted sign,
    years_ok, meets the share, and intervals excluding zero where required)
    and no pair is clearly opposite. Pairs that do not qualify but carry the
    predicted-sign mean do not block the level.

    Raises ValueError if predicted_sign is not 1 or -1, if there are no pairs,
    if any pair has empty arms, or if pairs do not all share one arm-name set.

    The 75% / 60% quarter shares and the "clearly opposite" rule are the
    design's defaults and the author's judgement calls (design section 7,
    decision 2), not pre-registered in themselves.
    No category depends on a P&L magnitude.
    """
    if predicted_sign not in (1, -1):
        raise ValueError("predicted_sign must be 1 or -1")
    if not pairs:
        raise ValueError("no pairs to classify")
    for pid, arms in pairs.items():
        if not arms:
            raise ValueError(f"pair {pid!r} has empty arms")
    if len({frozenset(arms) for arms in pairs.values()}) > 1:
        raise ValueError("pairs must all share the same arm names")
    if len(pairs) < 2:
        return "not_testable"
    if any(v is None for arms in pairs.values() for v in arms.values()):
        return "not_testable"
    arms = list(next(iter(pairs.values())))
    by_arm = {arm: [pairs[pid][arm] for pid in pairs] for arm in arms}

    def clear(sign):
        return all(_level(by_arm[a], sign, CLEAR_SHARE, True, full_years) for a in arms)

    def directional(sign):
        if clear(sign):
            return False
        if not any(_level(by_arm[a], sign, DIRECTIONAL_SHARE, False, full_years) for a in arms):
            return False
        return not any(_clearly_opposite(s, sign, full_years)
                       for a in arms for s in by_arm[a])

    if clear(predicted_sign):
        return "clear"
    if clear(-predicted_sign):
        return "reversed"
    if directional(predicted_sign):
        return "directional"
    return "no_difference"
