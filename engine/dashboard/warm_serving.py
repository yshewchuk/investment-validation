"""Pre-fit the Tier-4 serving models a nightly score will touch (legacy tree).

`Scorer._serving` -> `tier4.serving_model` refits a (model, fold) in-process on
a cache miss, adding ~2-2.4 GB on top of score's 6.4-6.8 GB (two nightlies were
floor-killed, 2026-10-20). The cache key carries the panel sha, so every first
run after a panel rebuild misses. `warm` runs the missing fits in a SUBPROCESS
(its memory returns to the OS) BEFORE score, through `serving_model` itself, so
the cache files are exactly what score would have written. Any failure raises
`WarmError`; the caller must stop, never fall back to in-process refits.

Folds: score takes `as_of=None`, so each strategy's own decision date (session
and decision offset aware) picks the fold, and it can fall before the nightly
date. The warmer therefore takes a SUPERSET: every month from the one holding
`decision_start - DECISION_LOOKBACK_DAYS` to the latest event's month. A fold
no row ends up using costs one spare fit; a fold left out costs an in-process
refit inside score.
"""
from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Earliest a strategy's decision date can fall before the nightly decision start.
DECISION_LOOKBACK_DAYS = 10
#: Hard bound on the whole warm subprocess.
WARM_TIMEOUT_S = 3 * 3600


class WarmError(RuntimeError):
    """The warm subprocess failed; the nightly must not go on to score."""


def needed_folds(event_dates, decision_start, decision_end=None) -> list[pd.Timestamp]:
    """Every month fold from the earliest possible decision to the latest event.

    `decision_end` is accepted for call compatibility; a decision never falls
    after its event, so the latest event's month bounds the range.
    """
    from engine.data.features import tier4

    events = pd.to_datetime(pd.Series(list(event_dates))).dropna()
    if events.empty:
        return []
    first = pd.Timestamp(decision_start).normalize() - pd.Timedelta(days=DECISION_LOOKBACK_DAYS)
    first_fold = pd.Timestamp(tier4.fold_start_of([first]).iloc[0])
    last_fold = pd.Timestamp(tier4.fold_start_of([events.max()]).iloc[0])
    return list(pd.date_range(first_fold, max(first_fold, last_fold), freq="MS"))


def _panel_snapshot() -> str:
    from engine.data.features import tier4

    return tier4.store.file_sha256(tier4.paths.PANEL)


def _serving_file(produces: str, fold, snapshot: str):
    from engine.data.features import tier4

    model = tier4.feature_model(produces)
    return model, tier4._serving_path(model.model_id, pd.Timestamp(fold), snapshot)


def _ready(model, fold, snapshot: str, path) -> bool:
    """Whether `path` is a cache file `serving_model` would serve verbatim.

    Same checks as its cache-hit branch, plus the embedded residual pool: an
    old-format file without it would be served only by rebuilding a multi-GB
    pool inside score, so it counts as not ready.
    """
    if not path.exists():
        return False
    import joblib

    try:
        stored = joblib.load(path)
        return bool(
            stored.get("model_id") == model.model_id
            and pd.Timestamp(stored.get("fold_start")) == pd.Timestamp(fold)
            and stored.get("tier3_snapshot") == snapshot
            and tuple(stored.get("features", ())) == tuple(model.features)
            and "pool_pred" in stored
            and "pool_res" in stored
        )
    except Exception:  # noqa: BLE001 - a corrupt file is simply not ready
        return False


def missing_jobs(folds, snapshot: str | None = None) -> list[tuple[str, str]]:
    """`(produces, fold ISO date)` pairs with no ready cache file for the panel."""
    from engine.data.features import tier4

    snapshot = snapshot or _panel_snapshot()
    jobs = []
    for fold in folds:
        for produces in tier4.PRODUCES:
            model, path = _serving_file(produces, fold, snapshot)
            if not _ready(model, fold, snapshot, path):
                jobs.append((produces, str(pd.Timestamp(fold).date())))
    return jobs


def warm(event_dates, decision_start, decision_end=None, *, command=None, log=None,
         timeout_s: float = WARM_TIMEOUT_S) -> dict:
    """Fit every missing serving model in a subprocess; raise `WarmError` on failure."""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    started = time.time()
    folds = needed_folds(event_dates, decision_start, decision_end)
    snapshot = _panel_snapshot()
    jobs = missing_jobs(folds, snapshot)
    log(f"[warm] {len(folds)} fold(s) needed, {len(jobs)} model fit(s) missing")
    if not jobs:
        return {"folds": len(folds), "fitted": 0, "elapsed_s": round(time.time() - started, 1)}
    cmd = command or [sys.executable, "-m", "engine.dashboard.warm_serving",
                      "--jobs-json", json.dumps(jobs), "--snapshot", snapshot]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdin=subprocess.DEVNULL,
                              check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise WarmError(f"warm subprocess timed out after {timeout_s:.0f}s "
                        f"with {len(jobs)} fit(s) pending") from exc
    elapsed = round(time.time() - started, 1)
    if proc.returncode != 0:
        raise WarmError(f"warm subprocess exited {proc.returncode} after {elapsed}s "
                        f"with {len(jobs)} fit(s) pending")
    remaining = missing_jobs(folds, snapshot)
    if remaining:
        raise WarmError(f"warm subprocess exited 0 but {len(remaining)} fit(s) are still missing")
    if _panel_snapshot() != snapshot:
        raise WarmError("panel changed while the warm subprocess ran; refusing to score")
    log(f"[warm] done: {len(jobs)} fit(s) in {elapsed}s")
    return {"folds": len(folds), "fitted": len(jobs), "elapsed_s": elapsed}


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def _child(jobs, snapshot: str) -> int:
    from engine.data.features import tier4
    from engine.features import load_panel

    started = time.time()
    if _panel_snapshot() != snapshot:
        print("[warm] panel changed since planning; refusing", file=sys.stderr, flush=True)
        return 2
    panel = load_panel()
    print(f"[warm] panel loaded, peak RSS {_peak_rss_mb():.0f} MB", file=sys.stderr, flush=True)
    written = []
    for i, (produces, fold) in enumerate(jobs, 1):
        model, path = _serving_file(produces, fold, snapshot)
        if path.exists():  # present but not ready (old format / corrupt): refit it
            path.unlink()
        tier4.serving_model(pd.Timestamp(fold), panel=panel, model=model)
        written.append(path)
        elapsed = time.time() - started
        eta = elapsed / i * (len(jobs) - i)
        print(f"[warm] fold {i}/{len(jobs)} {produces} {fold} done, "
              f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s, peak RSS {_peak_rss_mb():.0f} MB",
              file=sys.stderr, flush=True)
    if _panel_snapshot() != snapshot:
        for path in written:
            path.unlink(missing_ok=True)
        print("[warm] panel changed during fitting; caches removed", file=sys.stderr, flush=True)
        return 2
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-json", required=True)
    parser.add_argument("--snapshot", required=True)
    args = parser.parse_args(argv)
    return _child([tuple(job) for job in json.loads(args.jobs_json)], args.snapshot)


if __name__ == "__main__":
    sys.exit(main())
