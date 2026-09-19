#!/usr/bin/env python3
"""Diagnostic harness for the boundary-pass RSS burst (2026-09-18, round 3).

Round 2's fix (f3ae53f: byte-budget LRU on the causal-family cache, plus
cross-candidate content dedup on `chosen` hydration) left the forward-pass
RSS plateau unchanged (4.7-4.8 GB both runs) and moved the breach point by
only ~4 min: the real capture still died right after "boundary events: 8"
printed, watchdog jumping +700 MB in ~12 seconds (4.83 GB -> 5.52 GB CAP
BREACH). That is a BURST at the start of `boundary_pass`, not a retention
climb the byte-budget cache fix could have touched -- this harness measures
that burst directly instead of guessing again.

READING `boundary_pass` AND WHAT IT CALLS (see the module docstring's
"candidates for a ~700 MB allocation" list below) turned up a real
structural difference from `forward_pass`, not yet ruled out:

  `forward_pass` builds ONE `ChainIndex` up front, scoped to exactly the
  (ticker, date) keys its own events need
  (`replay_mod.load_chain_index(keys, ...)`, `tools/capture_tier0_corpus.py`
  around line 1040), and passes it into every `_score(..., index=index)`
  call for the whole pass.

  `boundary_pass` (line ~1081, `_score(scorer, request)`) and `_rescore`
  (line ~1103, used by `pinned_and_strike_pass`/`coarse_ladder_pass`) call
  `_score` with NO index at all. `Scorer._price_entry` (engine/score.py
  ~1773-1778) then falls back to building its OWN one-request `ChainIndex`
  per call: `load_chain_index([(ticker, quote_date), (ticker, exit_date)],
  ...)`. `load_chain_index` (engine/replay.py) loads keys "one year
  partition at a time" via `store.iter_table("option_chains", years=...)`,
  which reads and concats EVERY partition FILE for that year, for EVERY
  ticker, into one DataFrame (`engine/data/store.py:369`) -- filtering down
  to the 1-2 requested keys only AFTER that full read. Boundary events span
  up to 2500 days back and are chosen specifically to cross month/year
  boundaries, so a run of 8 events x 11 strategies (up to 88 `_score`
  calls, `boundary_pass` iterates strategy-outer, event-inner) can trigger
  up to 88 independent full-year `option_chains` reads, each one a
  transient several-hundred-MB-scale DataFrame (the whole table is cited
  elsewhere as 15.3M rows) that could easily still be resident when the
  next call starts its own read, if a `malloc_trim`-eligible free does not
  happen in the same 12-second window. THIS instrumentation wraps
  `score_mod.load_chain_index` directly to confirm or refute it: a call
  that takes noticeably long despite requesting only 1-2 keys is the
  signature of a full-partition read, and its call count across the pass is
  the multiplier.

CANDIDATES FOR THE ~700 MB ALLOCATION, in the order this reading ranks them:
  1. `load_chain_index`'s per-call, no-shared-index fallback in
     `boundary_pass`/`_rescore` (above) -- re-reading a full year partition
     of `option_chains` per request, up to 88 times, with no cross-call
     cache. Instrumented directly below.
  2. The causal-family cache (`_causal_pools`/`_causal_row_caches`/
     `phase4_recipe_cache`) picking up NEW (strategy, alpha, as_of) keys for
     boundary dates it has never seen -- but this is now byte-budgeted at
     600 MB combined (f3ae53f) and evicted LRU, so a burst from here would
     show as this harness's own `matcher.*` container prints growing past
     ~620 MB, not as an untraceable jump; tracemalloc would also attribute
     it to `engine/analogs.py`, not `engine/replay.py`/`engine/data/store.py`.
  3. `Phase4TraceCollector`'s per-candidate `quote_domain` capture
     (engine/score.py, UNCACHED, not `_Predocumented`) -- one full listed
     option-chain snapshot per candidate, but this is bounded by the SIZE OF
     ONE CHAIN (thousands of strikes/expiries for one ticker/date, not a
     whole year of all tickers), so it is a per-candidate linear cost, not a
     700 MB-in-12-seconds step -- ruled UNLIKELY as the primary driver, kept
     here because it is a real per-candidate allocation this harness's
     tracemalloc snapshots would also surface if wrong.
  4. `ResidualPool`/`_price_entry`'s other per-request DataFrame slices
     (`_clean`, entry/exit spot lookups) -- all scoped to one ticker's rows,
     not ruled out but structurally too small to be the primary source.

Usage (run under bounded_run so the box is protected either way -- this
harness does not raise or change the cap itself; it also does not run
`main()` or write any output, so it never reaches `write()`/`select()`):

    cd /root/investing-plan && \\
    python3 -u tools/bounded_run.py --max-rss-gb 5.5 --min-free-gb 0.5 -- \\
    python3 -u .claude/worktrees/agent-a5e983cb5cb96e00c/scratch/diag_boundary_pass.py \\
        --as-of 2026-09-12 --boundary-events 4

``--repo-root`` defaults to the current working directory, same convention
as ``diag_capture_retention.py``.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any


def _current_rss_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


class _PeakTracker:
    """Tracks the max RSS this process has reported across every snapshot,
    so the run's own peak is stated once at the end without re-parsing logs.
    """

    def __init__(self) -> None:
        self.peak_mb = 0.0
        self.peak_label = "<none>"

    def note(self, label: str, rss_mb: float) -> None:
        if rss_mb > self.peak_mb:
            self.peak_mb = rss_mb
            self.peak_label = label


def _snapshot(label: str, tracker: _PeakTracker, *, top: int, nframe: int) -> None:
    """RSS (as-is and post-gc.collect()) plus a tracemalloc top-N-by-
    traceback dump -- exactly what the coordinator asked for at "right
    before boundary_pass and at each boundary event": cheap (no full-heap
    walk of any container, unlike diag_capture_retention.py's approach),
    since tracemalloc already tracks allocations as they happen and a
    snapshot only reads that existing table.
    """
    rss = _current_rss_mb()
    gc.collect()
    rss_after_gc = _current_rss_mb()
    tracker.note(label, rss)
    print(f"[snap] {label}: rss_mb={rss:.1f} rss_after_gc_mb={rss_after_gc:.1f}",
          flush=True)
    snap = tracemalloc.take_snapshot()
    stats = snap.statistics("traceback")
    print(f"[snap] {label}: top {top} allocations by traceback "
          f"({nframe} frames each, current total {sum(s.size for s in stats)/1e6:.1f} MB "
          f"traced across {len(stats)} tracebacks):", flush=True)
    for stat in stats[:top]:
        print(f"    {stat.size / 1e6:.2f} MB  ({stat.count} blocks)", flush=True)
        for line in stat.traceback.format():
            print(f"        {line}", flush=True)


def _instrument_load_chain_index(score_mod: Any) -> Any:
    """Wrap `score_mod.load_chain_index` (the name `_price_entry` actually
    calls -- `engine.score` imported it directly, so patching
    `engine.replay.load_chain_index` after that import would be invisible
    here) to report every call's key count and wall time. A call that takes
    noticeably long despite 1-2 requested keys is the signature of a full
    year-partition `option_chains` read -- see the module docstring's
    candidate #1.
    """
    original = score_mod.load_chain_index
    calls = {"n": 0, "total_s": 0.0}

    def wrapped(keys, **kwargs):
        calls["n"] += 1
        n = calls["n"]
        key_list = list(keys)
        years = sorted({getattr(d, "year", None) or __import__("pandas").Timestamp(d).year
                        for _, d in key_list})
        started = time.monotonic()
        result = original(key_list, **kwargs)
        elapsed = time.monotonic() - started
        calls["total_s"] += elapsed
        print(f"[chain] load_chain_index call {n}: {len(key_list)} key(s) requested "
              f"(years={years}), {len(result)} resolved, {elapsed:.2f}s "
              f"(cumulative {calls['total_s']:.2f}s over {n} calls)", flush=True)
        return result

    score_mod.load_chain_index = wrapped
    return original


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--as-of", default="2026-09-12")
    ap.add_argument("--boundary-events", type=int, default=4,
                     help="matches capture_tier0_corpus.py main()'s own default")
    ap.add_argument("--repo-root", default=None,
                     help="defaults to the current working directory")
    ap.add_argument("--nframe", type=int, default=10,
                     help="tracemalloc traceback depth")
    ap.add_argument("--top", type=int, default=25,
                     help="top-N allocations printed per snapshot")
    args = ap.parse_args()

    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    repo_root = repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    tracemalloc.start(args.nframe)
    tracker = _PeakTracker()

    import pandas as pd

    import tools.capture_tier0_corpus as capture
    from engine import score as score_mod

    started = time.time()
    print("[diag] building the scorer (panel + replayed trades)...", flush=True)
    scorer = score_mod.Scorer()
    print(f"[diag] scorer ready in {time.time() - started:.0f}s", flush=True)
    _snapshot("post-build", tracker, top=args.top, nframe=args.nframe)

    # Forward pass is SKIPPED, per the coordinator: "monkeypatch it to
    # return nothing; that's fine for diagnosis" -- this keeps the run to
    # the ~2 min the boundary pass alone needs instead of the ~6 min a full
    # forward+boundary run takes. The hook point for a real forward-pass
    # run stays here: if this harness is later run WITHOUT skipping (e.g. a
    # `--run-forward-pass` variant calling `capture.forward_pass(...)`
    # here instead), the "post-forward-pass" snapshot below is the one that
    # would explain the ~2.2 GB plateau (2.5 GB post-build -> 4.7 GB
    # plateau) the coordinator asked about -- this run's own reading of it
    # is necessarily just the post-build baseline again, since nothing ran.
    print("[diag] forward pass: SKIPPED for this run (see module docstring)",
          flush=True)
    _snapshot("post-forward-pass (skipped -- see note above)", tracker,
              top=args.top, nframe=args.nframe)

    as_of = pd.Timestamp(args.as_of).normalize()
    boundaries = capture._boundary_events(
        as_of, args.boundary_events, scorer.calendar)
    print(f"[diag] boundary events: {len(boundaries)}", flush=True)
    _snapshot("pre-boundary-pass", tracker, top=args.top, nframe=args.nframe)

    original_load_chain_index = _instrument_load_chain_index(score_mod)
    original_score = capture._score
    counter = {"n": 0}

    def wrapped_score(scorer_arg, request, *, index=None):
        result = original_score(scorer_arg, request, index=index)
        counter["n"] += 1
        print(f"[diag] boundary _score call {counter['n']}: "
              f"{request.ticker} {request.strategy} as_of={request.as_of}",
              flush=True)
        _snapshot(f"boundary-call-{counter['n']}", tracker,
                  top=args.top, nframe=args.nframe)
        return result

    capture._score = wrapped_score
    try:
        out = capture.boundary_pass(scorer, boundaries, None)
    finally:
        capture._score = original_score
        score_mod.load_chain_index = original_load_chain_index

    print(f"[diag] boundary_pass candidates: {len(out)}", flush=True)
    _snapshot("post-boundary-pass", tracker, top=args.top, nframe=args.nframe)

    print(f"[diag] PEAK rss_mb={tracker.peak_mb:.1f} at {tracker.peak_label!r}",
          flush=True)
    tracemalloc.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
