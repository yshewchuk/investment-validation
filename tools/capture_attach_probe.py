#!/usr/bin/env python3
"""Replay the strict-trace attach/write step against a CAPTURE_DUMP_SELECTED dump.

``tools/capture_tier0_corpus.py --strict-phase4-trace`` builds a real Scorer
(the panel + replayed trades), scores every pass, selects the covering
subset, and only THEN runs the strict-trace attach step that has been
OOMing on this box. Setting ``CAPTURE_DUMP_SELECTED=<path>`` on that run
pickles exactly the post-selection state (``chosen``, ``index``, ``as_of``,
``snapshot``) attach/write need. This tool loads that dump and re-runs
``attach_strict_probe``/``write`` against it directly, into a temp dir --
so the expensive, one-time Scorer build and candidate selection never have
to happen again while iterating on the attach step itself.

Per-trace/per-member RSS is already logged by
``attach_strict_probe``/``chooser_trace`` themselves (unconditional
``[corpus] strict trace ...``/``[corpus]   member N ...`` prints); this
tool does not duplicate that. ``--tracemalloc`` additionally logs a
tracemalloc top-20 (by line) snapshot after every ``strict_trace_one`` call
-- the one call site both a plain ``score_result`` row and each DYN-SV
chooser member go through.

The supervisor runs this against a real dump, under bounded_run, per
AGENTS.md ("Running jobs on this box"). Never run it against real data
except as the supervisor.

Usage:
    python3 tools/capture_attach_probe.py <dump.pkl> [--out DIR] [--tracemalloc]
"""
from __future__ import annotations

import argparse
import gc
import pickle
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.capture_tier0_corpus as capture  # noqa: E402
from tools.capture_heartbeat import CaptureHeartbeat  # noqa: E402

#: This path replays the STRICT-TRACE ATTACH and the corpus WRITE against an
#: already-gathered dump -- it never rebuilds the Scorer or re-scores, so
#: its provisional whole-operation ETA must use only the measured attach +
#: write half (~62 min) of the last full strict capture, NOT the 81-min
#: total that also carries the ~19 min of gathering it skips. The "load"
#: stage carries zero prior of its own: while the dump is still being read,
#: the whole operation's remaining estimate is the prior-run attach + write.
PROBE_STAGE_PRIORS_MINUTES = {"load": 0.0, "attach": 41.0, "write": 21.0}
PROBE_STAGE_ORDER = ("load", "attach", "write")


def _new_heartbeat(**overrides) -> CaptureHeartbeat:
    """The probe's own heartbeat (attach/write-only prior), injectable in
    tests for a fake clock / capturing emitter."""
    overrides.setdefault("prefix", "[probe]")
    overrides.setdefault("stage_priors_minutes", PROBE_STAGE_PRIORS_MINUTES)
    overrides.setdefault("stage_order", PROBE_STAGE_ORDER)
    return CaptureHeartbeat(**overrides)


def _wrap_strict_trace_one_for_tracemalloc() -> None:
    """Patch ``capture_tier0_corpus.strict_trace_one`` (module-level name,
    looked up at call time by both ``attach_strict_probe`` and
    ``chooser_trace`` -- patching it here reaches both) to log RSS and a
    tracemalloc top-20 snapshot after every call, in addition to what it
    already returns."""
    original = capture.strict_trace_one
    calls = {"n": 0}

    def wrapped(candidate, snapshot, release_root):
        result = original(candidate, snapshot, release_root)
        calls["n"] += 1
        snap = tracemalloc.take_snapshot()
        top = snap.statistics("lineno")[:20]
        print(f"[probe] strict_trace_one call {calls['n']}: "
              f"rss {capture._rss_gb():.2f}G", flush=True)
        for stat in top:
            print(f"[probe]   {stat}", flush=True)
        return result

    capture.strict_trace_one = wrapped


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump", help="path to a CAPTURE_DUMP_SELECTED pickle")
    ap.add_argument("--out", default=None,
                    help="write into this directory (default: a fresh temp dir)")
    ap.add_argument("--tracemalloc", action="store_true",
                    help="log a tracemalloc top-20 snapshot after every native member")
    args = ap.parse_args(argv)

    dump_path = Path(args.dump)
    # One lifecycle-managed heartbeat for the WHOLE offline replay, with a
    # provisional profile that uses ONLY the attach/write priors (this path
    # skips gathering entirely). It is installed as the capture module's
    # singleton so capture.write()/attach_strict_probe() below tick it
    # exactly as a real capture would; the previous singleton is restored on
    # the way out so a second replay in the same process starts clean.
    hb = _new_heartbeat()
    previous = capture.set_heartbeat(hb)
    try:
        with hb:
            hb.begin_stage("load", units_total=1)
            hb.phase("load-dump", total=1)
            print(f"[probe] loading {dump_path} ...", flush=True)
            with dump_path.open("rb") as fh:
                payload = pickle.load(fh)
            hb.advance()  # the (potentially large) dump is fully resident
            chosen = payload["chosen"]
            index = payload["index"]
            as_of = payload["as_of"]
            snapshot = payload["snapshot"]
            print(f"[probe] loaded {len(chosen)} chosen candidates, "
                  f"rss {capture._rss_gb():.2f}G", flush=True)

            original = None
            if args.tracemalloc:
                original = capture.strict_trace_one
                tracemalloc.start(1)
                _wrap_strict_trace_one_for_tracemalloc()

            out_dir = (Path(args.out) if args.out
                       else Path(tempfile.mkdtemp(prefix="capture-attach-probe-")))

            started = time.time()
            try:
                # capture.write() owns the "attach" and "write" stages
                # (begin_stage + per-unit advance), reporting into this same
                # installed heartbeat, so its per-minute line and whole-
                # operation ETA cover the strict-trace attach, every pair
                # file AND the finalize/publish tail exactly as a live run.
                doc = capture.write(
                    out_dir, chosen, index, as_of, snapshot,
                    replace_existing=True, strict_trace=True,
                )
            finally:
                if args.tracemalloc:
                    capture.strict_trace_one = original
                    tracemalloc.stop()
            gc.collect()
            print(f"[probe] wrote {len(chosen)} pairs to {out_dir} in "
                  f"{time.time()-started:.0f}s, final rss {capture._rss_gb():.2f}G",
                  flush=True)
            print(f"[probe] corpus hash {doc['corpus_hash']}", flush=True)
            return 0
    finally:
        hb.stop()
        capture.set_heartbeat(previous)


if __name__ == "__main__":
    raise SystemExit(main())