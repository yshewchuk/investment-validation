"""Prove the incremental rebuild is byte-identical to the full rebuild.

    python3 -u tools/verify_incremental_rebuild.py

Builds ``earnings_events``, ``daily_market`` and ``option_chains`` three times
from the SAME raw data, each into the same throwaway root with everything a run
writes cleared first (raw trees are symlinked read-only; curated outputs,
quarantine flags and the cache stay inside the temp root):

1. full           -- today's default path, no cache;
2. incremental    -- cold cache (a full parse that writes the cache);
3. incremental    -- warm cache (reuses every unchanged parse).

Each run's WHOLE ``data/curated`` tree is compared byte for byte against the
full run's -- every file, and a file present in one tree and not the other is a
failure -- along with the build/manifest stats (only wall-clock fields and the
snapshot id are excluded; see :data:`IGNORED_STAT_KEYS`) and the quarantine
flags (ignoring ``flagged_at``). The warm run FAILs unless every cached table
reused at least one input with nothing re-parsed. The worker calls
``engine.data.rebuild.rebuild(tables=("events", "daily", "chains"),
incremental=...)`` exactly as ``engine/dashboard/nightly.py`` does, so the
``_with_cache`` path and the manifest/snapshot writes are exercised.

A source root missing any raw input tree the rebuild reads is refused with exit
status 2, and an empty build report is a failure: a proof tool run from a root
with no data must never "PASS" by building nothing. Prints PASS, or FAIL with
the differing files; exit status 0 / 1 / 2.

Heavy on real data: run it under ``tools/bounded_run.py``.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

#: The build/manifest comparison ignores ONLY these keys: wall-clock fields and
#: the run's snapshot id. Nothing else may be excluded -- every other field is
#: data and must match exactly.
IGNORED_STAT_KEYS = frozenset({
    "elapsed_s",     # how long that run took, not a property of the data
    "flagged_at",    # quarantine flags are compared separately, without it
    "generated_at",  # SNAPSHOT's wall-clock stamp
    "snapshot",      # the run's snapshot id
})

#: The tables the nightly's rebuild call builds, with the report field that must
#: be non-zero beside ``rows`` for the build to count as real.
BUILT_TABLES = {"events": "tickers", "daily": "tickers", "chains": "sources"}

#: Raw input trees the three-table rebuild reads. A source root missing any of
#: them cannot produce a non-empty build, so it is refused rather than run.
RAW_SOURCES = (
    "data/raw/fetch",
    "earnings_predictions/data/raw/orats/summaries",
    "earnings_predictions/data/raw/orats/cores",
    "earnings_predictions/data/raw/orats/strikes",
    "earnings_predictions/data/raw/orats/earnings",
    "earnings_predictions/data/raw/oquants/moves",
    "earnings_predictions/data/raw/polygon",
)


def _require_raw(source_root: Path) -> None:
    """Refuse (exit 2) a source root that lacks any raw tree the rebuild reads."""
    missing = [rel for rel in RAW_SOURCES if not (source_root / rel).is_dir()]
    if missing:
        print(
            f"FAIL: source root {source_root} is missing raw source "
            f"{'directory' if len(missing) == 1 else 'directories'}: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        raise SystemExit(2)


def _link_raw(source_root: Path, root: Path) -> None:
    """A throwaway root whose raw inputs are the source root's, read-only."""
    for rel in ("data/raw/fetch", "earnings_predictions/data/raw"):
        link = root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(source_root / rel, target_is_directory=True)


def _reset(root: Path, *, keep_cache: bool) -> None:
    """Clear everything a run writes, so each run's output stands alone."""
    for rel in ("data/curated", "data/raw/quarantine", "data/features"):
        path = root / rel
        if path.exists():
            shutil.rmtree(path)
    if not keep_cache:
        cache = root / "data/cache"
        if cache.exists():
            shutil.rmtree(cache)


def _run_worker(root: Path, mode: str, out: Path, cache_out: Path) -> None:
    env = dict(os.environ, INVESTING_PLAN_ROOT=str(root), PYTHONPATH=str(REPO))
    env.pop("INVESTING_PLAN_REBUILD_FULL", None)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", mode,
           "--report", str(out), "--cache-report", str(cache_out)]
    print(f"--- {mode} build in {root}", flush=True)
    subprocess.run(cmd, env=env, check=True, cwd=REPO)


def _worker(mode: str, report: Path, cache_report: Path) -> int:
    """One subprocess build, called exactly as the nightly calls it."""
    from engine import paths
    from engine.data import rebuild

    paths.ensure_dirs()
    built: list = []

    class _RecordingCache(rebuild.InputCache):
        def __init__(self, table: str, **kw):
            super().__init__(table, **kw)
            built.append(self)

    original = rebuild.InputCache
    rebuild.InputCache = _RecordingCache
    try:
        result = rebuild.rebuild(
            tables=("events", "daily", "chains"),
            incremental=mode == "incremental",
        )
    finally:
        rebuild.InputCache = original
    report.write_text(json.dumps(result.as_dict(), sort_keys=True, default=str, indent=1))
    cache_report.write_text(json.dumps(
        {cache.table: dict(cache.stats, full=cache.full) for cache in built},
        sort_keys=True, default=str, indent=1,
    ))
    return 0


def _files(base: Path) -> dict[str, Path]:
    if not base.exists():
        return {}
    return {str(p.relative_to(base)): p for p in sorted(base.rglob("*")) if p.is_file()}


def _flags(base: Path) -> dict[str, list]:
    out = {}
    for rel, path in _files(base).items():
        entries = json.loads(path.read_text())
        entries = entries if isinstance(entries, list) else [entries]
        out[rel] = [{k: v for k, v in e.items() if k != "flagged_at"} for e in entries]
    return out


def compare_trees(expected: Path, actual: Path) -> list[str]:
    """Relative paths that differ (missing on either side, or different bytes)."""
    a, b = _files(expected), _files(actual)
    diffs = [f"only in full: {k}" for k in sorted(set(a) - set(b))]
    diffs += [f"only in incremental: {k}" for k in sorted(set(b) - set(a))]
    diffs += [f"bytes differ: {k}" for k in sorted(set(a) & set(b))
              if not filecmp.cmp(a[k], b[k], shallow=False)]
    return diffs


def compare_stats(expected, actual, path: str = "") -> list[str]:
    """Field-level differences, ignoring :data:`IGNORED_STAT_KEYS` only."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        diffs = []
        for key in sorted(set(expected) | set(actual)):
            if key in IGNORED_STAT_KEYS:
                continue
            leaf = f"{path}.{key}" if path else key
            diffs += compare_stats(expected.get(key), actual.get(key), leaf)
        return diffs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: {len(expected)} item(s) vs {len(actual)}"]
        diffs = []
        for i, (x, y) in enumerate(zip(expected, actual)):
            diffs += compare_stats(x, y, f"{path}[{i}]")
        return diffs
    if expected != actual:
        return [f"{path}: {expected!r} != {actual!r}"]
    return []


def _empty_diffs(label: str, doc: dict) -> list[str]:
    """FAIL a build report whose row count or source-unit count is zero."""
    diffs = []
    reports = doc.get("reports", {})
    for table, unit in BUILT_TABLES.items():
        report = reports.get(table) or {}
        empty = [key for key in ("rows", unit) if not report.get(key)]
        if empty:
            diffs.append(
                f"{label} {table}: empty build report (" +
                ", ".join(f"{key}=0" for key in empty) + ")"
            )
    return diffs


def _warm_diffs(caches: dict) -> list[str]:
    """The warm run must reuse every cached input of every cached table."""
    diffs = []
    for table in ("daily_market", "option_chains"):
        stats = (caches or {}).get(table, {})
        if not stats.get("reused") or stats.get("parsed") or stats.get("full"):
            diffs.append(
                f"warm cache miss for {table}: full={stats.get('full')}, "
                f"reused={stats.get('reused')}, parsed={stats.get('parsed')}"
            )
    return diffs


def _compare(full: Path, other: Path, label: str) -> list[str]:
    diffs = [f"{label} curated: {d}" for d in
             compare_trees(full / "curated", other / "curated")]
    expected = json.loads((full / "report.json").read_text())
    actual = json.loads((other / "report.json").read_text())
    diffs += [f"{label} stats: {d}" for d in compare_stats(expected, actual)]
    if _flags(full / "quarantine") != _flags(other / "quarantine"):
        diffs.append(f"{label}: quarantine flags differ")
    return diffs


def _snapshot(root: Path, dest: Path, report: Path, cache_report: Path) -> None:
    dest.mkdir(parents=True)
    shutil.copytree(root / "data/curated", dest / "curated")
    quarantine = root / "data/raw/quarantine"
    if quarantine.exists():
        shutil.copytree(quarantine, dest / "quarantine")
    shutil.copy(report, dest / "report.json")
    shutil.copy(cache_report, dest / "caches.json")


def verify(source_root: Path, work: Path) -> list[str]:
    source_root = source_root.resolve()
    _require_raw(source_root)
    root = work / "root"
    _link_raw(source_root, root)
    runs = (("full", "full"), ("cold", "incremental"), ("warm", "incremental"))
    for label, mode in runs:
        _reset(root, keep_cache=label == "warm")
        report, cache_report = work / f"{label}.json", work / f"{label}.caches.json"
        _run_worker(root, mode, report, cache_report)
        _snapshot(root, work / f"out_{label}", report, cache_report)
    diffs: list[str] = []
    for label, _ in runs:
        doc = json.loads((work / f"{label}.json").read_text())
        diffs += _empty_diffs(label, doc)
    diffs += _warm_diffs(json.loads((work / "warm.caches.json").read_text()))
    diffs += _compare(work / "out_full", work / "out_cold", "cold")
    diffs += _compare(work / "out_full", work / "out_warm", "warm")
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source-root", default=str(REPO))
    ap.add_argument("--work-dir", default=None, help="kept after the run when given")
    ap.add_argument("--worker", choices=("full", "incremental"), help=argparse.SUPPRESS)
    ap.add_argument("--report", help=argparse.SUPPRESS)
    ap.add_argument("--cache-report", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        return _worker(args.worker, Path(args.report), Path(args.cache_report))
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="incr_verify_"))
    work.mkdir(parents=True, exist_ok=True)
    diffs = verify(Path(args.source_root), work)
    if not args.work_dir:
        shutil.rmtree(work, ignore_errors=True)
    if diffs:
        print("FAIL: incremental rebuild differs from full rebuild")
        for line in diffs:
            print(f"  {line}")
        return 1
    print("PASS: incremental (cold and warm) byte-identical to full for "
          + ", ".join(BUILT_TABLES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
