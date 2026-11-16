"""Report-only IV empirical forecast-distribution bands over stored trial predictions."""
import math
import sys
import time

import numpy as np

from experiments.magnitude_search_plan import trial_key
from experiments.magnitude_search_records import read_record
from experiments.size_distribution_report import (
    BUCKETS,
    MIN_BUCKET_ROWS,
    TAUS,
    TrialArrays,
    _has_rows,
    _restrict,
    _validate,
    _values,
)

SCHEMA = "iv_distribution.v1"
_METRICS = ("pinball_p10", "pinball_p50", "pinball_p90", "pinball_mean",
            "coverage_p10", "coverage_p90", "coverage_p10_p90")


def read_trial_arrays(conn, store, run_id, plan):
    arrays = TrialArrays()
    visited = set()
    for group in plan["folds"]:
        outer_year = group["outer"]["year"]
        for fold in group["inner"]:
            inner_year = fold["year"]
            for candidate in plan["candidates"]:
                key = trial_key(plan, outer_year, inner_year, candidate["candidate_id"])
                if key in visited:
                    continue
                visited.add(key)
                terminal = read_record(conn, store, run_id, "terminal:" + key)
                if not isinstance(terminal, dict) or terminal.get("status") != "succeeded":
                    continue
                value = {}
                for horizon in plan["horizons"]:
                    rows = [row for row in terminal["predictions"]
                            if row["horizon_sessions"] == horizon]
                    value[horizon] = (
                        np.asarray([float(row["predicted_crush_pct_iv30"]) for row in rows],
                                   dtype=float),
                        np.asarray([float(row["crush_pct_iv30"]) for row in rows], dtype=float),
                        np.asarray([sys.intern(str(row["event_id"])) for row in rows],
                                   dtype=object))
                arrays.add(key, value)
    return arrays


def outer_arrays(plan, predictions):
    arrays = {}
    rows_by_year = {}
    for row in predictions:
        rows_by_year.setdefault(row["outer_year"], []).append(row)
    for outer_year, year_rows in rows_by_year.items():
        arrays[outer_year] = {}
        for horizon in plan["horizons"]:
            rows = [row for row in year_rows if row["horizon_sessions"] == horizon]
            arrays[outer_year][horizon] = (
                np.asarray([float(row["predicted_crush_pct_iv30"]) for row in rows],
                           dtype=float),
                np.asarray([float(row["crush_pct_iv30"]) for row in rows], dtype=float),
                np.asarray([sys.intern(str(row["event_id"])) for row in rows], dtype=object))
    return arrays


def band_metrics(pool, scored, horizons, allowed=None):
    _validate(pool + [scored])
    per_horizon, measured = [], []
    for horizon in horizons:
        pool_pred, pool_y, pool_ids = _values(pool, horizon)
        scored_pred, scored_y, _ = _values([scored], horizon)
        if allowed is not None:
            keep = np.array([event_id in allowed for event_id in pool_ids], dtype=bool)
            pool_pred, pool_y = pool_pred[keep], pool_y[keep]
        metrics = {}
        if pool_pred.size == 0:
            status = "no_pool"
        else:
            edges = np.quantile(pool_pred, [0.2, 0.4, 0.6, 0.8])
            pool_bucket = np.digitize(pool_pred, edges)
            if np.bincount(pool_bucket, minlength=BUCKETS).min() < MIN_BUCKET_ROWS:
                status = "pool_too_small"
            elif scored_pred.size == 0:
                status = "no_scored_rows"
            else:
                quantiles = np.empty((BUCKETS, len(TAUS)))
                for bucket in range(BUCKETS):
                    member = pool_bucket == bucket
                    quantiles[bucket] = np.quantile(pool_y[member] - pool_pred[member], TAUS)
                band = scored_pred[:, None] + quantiles[np.digitize(scored_pred, edges)]
                d = scored_y[:, None] - band
                taus = np.asarray(TAUS)
                per_tau = np.maximum(taus * d, (taus - 1) * d).mean(axis=0)
                q10 = band[:, TAUS.index(0.1)]
                q50 = band[:, TAUS.index(0.5)]
                q90 = band[:, TAUS.index(0.9)]
                metrics = {"n_scored": int(scored_pred.size), "n_pool": int(pool_pred.size),
                    "pinball_p10": float(per_tau[TAUS.index(0.1)]),
                    "pinball_p50": float(per_tau[TAUS.index(0.5)]),
                    "pinball_p90": float(per_tau[TAUS.index(0.9)]),
                    "pinball_mean": float(per_tau.mean()),
                    "coverage_p10": float(np.mean(scored_y <= q10)),
                    "coverage_p90": float(np.mean(scored_y <= q90)),
                    "coverage_p10_p90": float(np.mean((q10 <= scored_y) & (scored_y <= q90))),
                    "band_p10": float(np.mean(q10)),
                    "band_p50": float(np.mean(q50)),
                    "band_p90": float(np.mean(q90))}
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
        pooled.update({key: float(math.fsum(entry[key] * entry["n_scored"]
                                            for entry in measured) / scored_total)
                       for key in _METRICS})
    return {"status": status, "per_horizon": per_horizon, "pooled": pooled}


def _failed():
    return {"status": "trial_failed", "pooled": None, "per_horizon": []}


def collect_distribution(plan, trial_arrays, outer_rows, selected_ids):
    _validate(list(outer_rows.values()))
    if getattr(trial_arrays, "capped", False):
        return {"schema_version": SCHEMA, "status": "skipped_resource_cap"}
    horizons, groups = plan["horizons"], plan["folds"]
    total = sum(len(group["inner"]) for group in groups) * len(plan["candidates"])
    started = last_progress = time.monotonic()
    processed = 0
    print(f"[iv_distribution] start inner_rows={total}", file=sys.stderr)
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
                    print(f"[iv_distribution] {processed}/{total} rows, {rate:.3f} rows/s, "
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
                     for arm in ("incumbent", "training_mean"))
    print(f"[iv_distribution] done in {time.monotonic() - started:.1f}s", file=sys.stderr)
    return {"schema_version": SCHEMA, "status": "ok", "quantiles": list(TAUS),
        "buckets": BUCKETS, "min_bucket_rows": MIN_BUCKET_ROWS,
        "inner": inner, "outer": outer}
