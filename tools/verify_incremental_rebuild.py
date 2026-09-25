"""Prove the incremental rebuild is byte-identical to the full rebuild.

    python3 -u tools/verify_incremental_rebuild.py [--tables daily,chains] [--sample N]

Builds ``daily_market`` and ``option_chains`` three times from the SAME raw
data, each into its own throwaway root (raw trees are symlinked read-only;
curated outputs, quarantine flags and the cache stay inside the temp root):

1. full           -- today's default path, no cache;
2. incremental    -- cold cache (a full parse that writes the cache);
3. incremental    -- warm cache (reuses every unchanged parse).

Every curated file of 2 and 3 is compared byte for byte against 1, along with
the build reports and the quarantine flags (ignoring their timestamps). Prints
PASS, or FAIL with the differing files; exit status 0 / 1.

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
TABLES = {"daily": "daily_market", "chains": "option_chains"}


def _link_raw(source_root: Path, root: Path) -> None:
    """A throwaway root whose raw inputs are the source root's, read-only."""
    for rel in ("data/raw/fetch", "earnings_predictions/data/raw"):
        target = source_root / rel
        link = root / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            link.symlink_to(target, target_is_directory=True)


def _run_worker(root: Path, mode: str, tables: list[str], sample: int | None, out: Path) -> None:
    env = dict(os.environ, INVESTING_PLAN_ROOT=str(root), PYTHONPATH=str(REPO))
    env.pop("INVESTING_PLAN_REBUILD_FULL", None)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", mode,
           "--tables", ",".join(tables), "--report", str(out)]
    if sample:
        cmd += ["--sample", str(sample)]
    print(f"--- {mode} build in {root}", flush=True)
    subprocess.run(cmd, env=env, check=True, cwd=REPO)


def _worker(mode: str, tables: list[str], sample: int | None, report: Path) -> int:
    from engine import paths
    from engine.data import rebuild
    from engine.data.rebuild_cache import InputCache

    paths.ensure_dirs()
    reports = {}
    for name in tables:
        builder = rebuild.build_daily_table if name == "daily" else rebuild.build_chains_table
        cache = InputCache(TABLES[name]) if mode == "incremental" else None
        reports[name] = builder(sample, cache)
        if cache is not None:
            cache.commit()
            print(f"  {cache.summary()}", flush=True)
            reports[f"{name}_cache"] = dict(cache.stats, full=cache.full)
    report.write_text(json.dumps(reports, sort_keys=True, default=str, indent=1))
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


def _compare(full: Path, other: Path, label: str, tables: list[str]) -> list[str]:
    diffs = []
    for name in tables:
        table = TABLES[name]
        diffs += [f"{label} {table}: {d}" for d in
                  compare_trees(full / "curated" / table, other / "curated" / table)]
    reports = [json.loads((p / "report.json").read_text()) for p in (full, other)]
    for name in tables:
        if reports[0].get(name) != reports[1].get(name):
            diffs.append(f"{label}: build report differs for {name}")
    if _flags(full / "quarantine") != _flags(other / "quarantine"):
        diffs.append(f"{label}: quarantine flags differ")
    return diffs


def _snapshot(root: Path, dest: Path, report: Path) -> None:
    dest.mkdir(parents=True)
    shutil.copytree(root / "data/curated", dest / "curated")
    quarantine = root / "data/raw/quarantine"
    if quarantine.exists():
        shutil.copytree(quarantine, dest / "quarantine")
        shutil.rmtree(quarantine)
    shutil.copy(report, dest / "report.json")


def verify(source_root: Path, work: Path, tables: list[str], sample: int | None) -> list[str]:
    roots = {"full": work / "root_full", "incremental": work / "root_incr"}
    for root in roots.values():
        _link_raw(source_root, root)
    runs = (("full", "full"), ("cold", "incremental"), ("warm", "incremental"))
    for label, mode in runs:
        report = work / f"{label}.json"
        _run_worker(roots[mode], mode, tables, sample, report)
        _snapshot(roots[mode], work / f"out_{label}", report)
    warm = json.loads((work / "warm.json").read_text())
    diffs = _compare(work / "out_full", work / "out_cold", "cold", tables)
    diffs += _compare(work / "out_full", work / "out_warm", "warm", tables)
    for name in tables:
        stats = warm.get(f"{name}_cache", {})
        print(f"warm {name}: reused {stats.get('reused')}, parsed {stats.get('parsed')}")
    return diffs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tables", default="daily,chains")
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--source-root", default=str(REPO))
    ap.add_argument("--work-dir", default=None, help="kept after the run when given")
    ap.add_argument("--worker", choices=("full", "incremental"), help=argparse.SUPPRESS)
    ap.add_argument("--report", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    tables = [t for t in args.tables.split(",") if t]
    if args.worker:
        return _worker(args.worker, tables, args.sample, Path(args.report))
    work = Path(args.work_dir) if args.work_dir else Path(tempfile.mkdtemp(prefix="incr_verify_"))
    work.mkdir(parents=True, exist_ok=True)
    diffs = verify(Path(args.source_root), work, tables, args.sample)
    if not args.work_dir:
        shutil.rmtree(work, ignore_errors=True)
    if diffs:
        print("FAIL: incremental rebuild differs from full rebuild")
        for line in diffs:
            print(f"  {line}")
        return 1
    print(f"PASS: incremental (cold and warm) byte-identical to full for {', '.join(tables)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
