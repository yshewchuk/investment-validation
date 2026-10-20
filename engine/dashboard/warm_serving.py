"""Pre-fit the Tier-4 serving models a nightly score will touch (legacy tree).

`Scorer._serving` -> `tier4.serving_model` refits a (model, fold) in-process on
a cache miss, adding ~2-2.4 GB on top of score's 6.4-6.8 GB (two nightlies were
floor-killed, 2026-10-20). The cache key carries the panel sha, so every first
run after a panel rebuild misses. `warm` runs the missing fits in a SUBPROCESS
(its memory returns to the OS) BEFORE score, through `serving_model` itself, so
the cache files are exactly what score would have written. Any failure raises
`WarmError`; the caller must stop, never fall back to in-process refits.
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


class WarmError(RuntimeError):
    """The warm subprocess failed; the nightly must not go on to score."""


def needed_folds(event_dates, decision_start, decision_end) -> list[pd.Timestamp]:
    """Every fold `serving_fold(event, decision)` can return for the window.

    Decisions run from `decision_start` (the backfill start, else as_of) to
    `decision_end` (as_of); every month start between is included too.
    """
    from engine.data.features import tier4

    start = pd.Timestamp(decision_start).normalize()
    end = pd.Timestamp(decision_end).normalize()
    decisions = {start, end}
    decisions.update(pd.date_range(start, end, freq="MS"))
    folds = set()
    for event in pd.to_datetime(pd.Series(list(event_dates))).dropna().unique():
        for decision in decisions:
            folds.add(pd.Timestamp(tier4.serving_fold(event, decision)))
    return sorted(folds)


def missing_jobs(folds) -> list[tuple[str, str]]:
    """`(produces, fold ISO date)` pairs whose cache file is absent for the current panel."""
    from engine.data.features import tier4

    snapshot = tier4.store.file_sha256(tier4.paths.PANEL)
    jobs = []
    for fold in folds:
        for produces in tier4.PRODUCES:
            model = tier4.feature_model(produces)
            if not tier4._serving_path(model.model_id, pd.Timestamp(fold), snapshot).exists():
                jobs.append((produces, str(pd.Timestamp(fold).date())))
    return jobs


def warm(event_dates, decision_start, decision_end, *, command=None, log=None) -> dict:
    """Fit every missing serving model in a subprocess; raise `WarmError` on failure."""
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    started = time.time()
    folds = needed_folds(event_dates, decision_start, decision_end)
    jobs = missing_jobs(folds)
    log(f"[warm] {len(folds)} fold(s) needed, {len(jobs)} model fit(s) missing")
    if not jobs:
        return {"folds": len(folds), "fitted": 0, "elapsed_s": round(time.time() - started, 1)}
    cmd = command or [sys.executable, "-m", "engine.dashboard.warm_serving",
                      "--jobs-json", json.dumps(jobs)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, check=False)
    elapsed = round(time.time() - started, 1)
    if proc.returncode != 0:
        raise WarmError(f"warm subprocess exited {proc.returncode} after {elapsed}s "
                        f"with {len(jobs)} fit(s) pending")
    remaining = missing_jobs(folds)
    if remaining:
        raise WarmError(f"warm subprocess exited 0 but {len(remaining)} fit(s) are still missing")
    log(f"[warm] done: {len(jobs)} fit(s) in {elapsed}s")
    return {"folds": len(folds), "fitted": len(jobs), "elapsed_s": elapsed}


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KiB


def _child(jobs) -> int:
    from engine.data.features import tier4
    from engine.features import load_panel

    started = time.time()
    panel = load_panel()
    print(f"[warm] panel loaded, peak RSS {_peak_rss_mb():.0f} MB", file=sys.stderr, flush=True)
    for i, (produces, fold) in enumerate(jobs, 1):
        tier4.serving_model(pd.Timestamp(fold), panel=panel, model=tier4.feature_model(produces))
        print(f"[warm] fold {i}/{len(jobs)} {produces} {fold} done, "
              f"{time.time() - started:.0f}s elapsed, peak RSS {_peak_rss_mb():.0f} MB",
              file=sys.stderr, flush=True)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs-json", required=True)
    args = parser.parse_args(argv)
    return _child([tuple(job) for job in json.loads(args.jobs_json)])


if __name__ == "__main__":
    sys.exit(main())
