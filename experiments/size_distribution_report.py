"""Report-only empirical forecast-distribution bands over stored trial predictions."""
import sys
import time

import numpy as np

from experiments.magnitude_search_plan import trial_key

SCHEMA = "size_distribution.v1"
TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
BUCKETS = 5
MIN_BUCKET_ROWS = 20
RATIO_FLOOR = 0.05
_METRICS = ("pinball_p50", "pinball_p90", "pinball_mean", "coverage_p90", "coverage_p10_p90")


class TrialArrays(dict):
    MAX_BYTES = 256 * 1024 * 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capped = False
        self.nbytes = 0

    def add(self, key, arrays):
        for value in arrays.values():
            if not all(np.isfinite(np.asarray(part, dtype=float)).all() for part in value[:2]):
                raise ValueError("non-finite distribution input")
        if self.capped:
            return
        size = sum(part.nbytes for value in arrays.values() for part in value)
        if self.nbytes + size > self.MAX_BYTES:
            self.clear()
            self.capped = True
            self.nbytes = 0
            print("[distribution] retained trial arrays exceed cap; distribution skipped",
                  file=sys.stderr)
            return
        self[key] = arrays
        self.nbytes += size


def _validate(arrays):
    for array in arrays:
        for value in array.values():
            if not isinstance(value, tuple) or len(value) != 3:
                raise ValueError("malformed distribution input")
            parts = [np.asarray(part) for part in value]
            if any(part.ndim != 1 for part in parts) or len({part.size for part in parts}) != 1:
                raise ValueError("malformed distribution input")
            if not all(np.isfinite(np.asarray(part, dtype=float)).all() for part in parts[:2]):
                raise ValueError("non-finite distribution input")


def _restrict(arrays, allowed):
    restricted = {}
    for horizon, value in arrays.items():
        pred, y, event_ids = value
        event_ids = np.asarray(event_ids, dtype=object)
        keep = np.array([event_id in allowed for event_id in event_ids], dtype=bool)
        restricted[horizon] = (np.asarray(pred)[keep], np.asarray(y)[keep], event_ids[keep])
    return restricted


def _has_rows(arrays):
    return any(len(value[0]) for value in arrays.values())


def _values(arrays, horizon):
    present = [array[horizon] for array in arrays if horizon in array]
    if not present:
        return (np.array([], dtype=float), np.array([], dtype=float),
                np.array([], dtype=object))
    return (np.concatenate([np.asarray(pred, dtype=float) for pred, _, _ in present]),
            np.concatenate([np.asarray(y, dtype=float) for _, y, _ in present]),
            np.concatenate([np.asarray(event_ids, dtype=object) for _, _, event_ids in present]))


def band_metrics(pool, scored, horizons, allowed=None):
    _validate(pool)
    _validate([scored])
    per_horizon, measured = [], []
    for horizon in horizons:
        pool_pred, pool_y, pool_ids = _values(pool, horizon)
        scored_pred, scored_y, _ = _values([scored], horizon)
        if allowed is not None:
            keep = np.array([event_id in allowed for event_id in pool_ids], dtype=bool)
            pool_pred, pool_y = pool_pred[keep], pool_y[keep]
        if not (np.isfinite(pool_pred).all() and np.isfinite(pool_y).all()
                and np.isfinite(scored_pred).all() and np.isfinite(scored_y).all()):
            raise ValueError("non-finite distribution input")
        metrics = {}
        if pool_pred.size == 0:
            status = "no_pool"
        else:
            p = np.maximum(pool_pred, RATIO_FLOOR)
            edges = np.quantile(pool_pred, [0.2, 0.4, 0.6, 0.8])
            pool_bucket = np.digitize(pool_pred, edges)
            if np.bincount(pool_bucket, minlength=BUCKETS).min() < MIN_BUCKET_ROWS:
                status = "pool_too_small"
            elif scored_pred.size == 0:
                status = "no_scored_rows"
            else:
                f = scored_pred
                quantiles = np.empty((BUCKETS, len(TAUS)))
                for bucket in range(BUCKETS):
                    member = pool_bucket == bucket
                    quantiles[bucket] = np.quantile(pool_y[member] / p[member], TAUS)
                band = np.maximum(f, RATIO_FLOOR)[:, None] * quantiles[np.digitize(f, edges)]
                d = scored_y[:, None] - band
                taus = np.asarray(TAUS)
                per_tau = np.maximum(taus * d, (taus - 1) * d).mean(axis=0)
                q10, q90 = band[:, TAUS.index(0.1)], band[:, TAUS.index(0.9)]
                metrics = {"n_scored": int(scored_pred.size), "n_pool": int(pool_pred.size),
                    "pinball_p50": float(per_tau[TAUS.index(0.5)]),
                    "pinball_p90": float(per_tau[TAUS.index(0.9)]),
                    "pinball_mean": float(per_tau.mean()),
                    "coverage_p90": float(np.mean(scored_y <= q90)),
                    "coverage_p10_p90": float(np.mean((q10 <= scored_y) & (scored_y <= q90)))}
                status = "ok"
        entry = {"horizon_sessions": horizon, "status": status, **metrics}
        per_horizon.append(entry)
        if status == "ok":
            measured.append(entry)
    status = "ok" if len(measured) == len(per_horizon) else next(
        entry["status"] for entry in per_horizon if entry["status"] != "ok")
    pooled = None
    if status == "ok":
        scored_total = sum(entry["n_scored"] for entry in measured)
        pooled = {"n_scored": scored_total,
                  "n_pool": sum(entry["n_pool"] for entry in measured)}
        pooled.update({key: float(sum(entry[key] * entry["n_scored"] for entry in measured) / scored_total)
                       for key in _METRICS})
    return {"status": status, "per_horizon": per_horizon, "pooled": pooled}


def _failed():
    return {"status": "trial_failed", "pooled": None, "per_horizon": []}


def collect_distribution(plan, trial_arrays, outer_rows, selected_ids):
    if "target" in plan:
        return {"schema_version": SCHEMA, "status": "unsupported_target"}
    if getattr(trial_arrays, "capped", False):
        return {"schema_version": SCHEMA, "status": "skipped_resource_cap"}
    _validate(list(trial_arrays.values()) + list(outer_rows.values()))
    horizons, groups = plan["horizons"], plan["folds"]
    total = sum(len(group["inner"]) for group in groups) * len(plan["candidates"])
    started = last_progress = time.monotonic()
    processed = 0
    print(f"[distribution] start inner_rows={total}", file=sys.stderr)
    inner = []
    for group in groups:
        outer_year = group["outer"]["year"]
        years = sorted(fold["year"] for fold in group["inner"])
        folds_by_year = {fold["year"]: fold for fold in group["inner"]}
        for candidate in plan["candidates"]:
            candidate_id = candidate["candidate_id"]
            keys = {year: trial_key(plan, outer_year, year, candidate_id) for year in years}
            for year in years:
                allowed = set(folds_by_year[year]["training_candidate_event_ids"])
                pool_years, pool = [], []
                for earlier in years:
                    if earlier >= year or keys[earlier] not in trial_arrays:
                        continue
                    restricted = _restrict(trial_arrays[keys[earlier]], allowed)
                    if _has_rows(restricted):
                        pool_years.append(earlier)
                        pool.append(restricted)
                metrics = (band_metrics(pool, trial_arrays[keys[year]], horizons)
                           if keys[year] in trial_arrays else _failed())
                inner.append({"outer_year": outer_year, "inner_year": year,
                    "candidate_id": candidate_id, "trial_id": keys[year],
                    "pool_years": pool_years, **metrics})
                processed += 1
                now = time.monotonic()
                if now - last_progress >= 60:
                    elapsed = now - started
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    eta = (total - processed) / rate if rate > 0 else 0.0
                    print(f"[distribution] {processed}/{total} rows, {rate:.3f} rows/s, "
                          f"ETA {eta:.0f}s", file=sys.stderr)
                    last_progress = now
    outer = []
    for group in groups:
        outer_year = group["outer"]["year"]
        candidate_id = selected_ids[outer_year]
        years = sorted(fold["year"] for fold in group["inner"])
        keys = {year: trial_key(plan, outer_year, year, candidate_id) for year in years}
        allowed = set(group["outer"]["training_candidate_event_ids"])
        pool_years, pool = [], []
        for year in years:
            if keys[year] not in trial_arrays:
                continue
            restricted = _restrict(trial_arrays[keys[year]], allowed)
            if _has_rows(restricted):
                pool_years.append(year)
                pool.append(restricted)
        outer.append({"outer_year": outer_year, "arm": "selected", "candidate_id": candidate_id,
            "pool_years": pool_years,
            **band_metrics(pool, outer_rows.get(outer_year, {}), horizons)})
        outer.extend({"outer_year": outer_year, "arm": arm, "status": "no_pool"}
                     for arm in ("incumbent", "history_mean", "implied_move"))
    print(f"[distribution] done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return {"schema_version": SCHEMA, "status": "ok", "quantiles": list(TAUS),
        "buckets": BUCKETS, "min_bucket_rows": MIN_BUCKET_ROWS, "ratio_floor": RATIO_FLOOR,
        "inner": inner, "outer": outer}
