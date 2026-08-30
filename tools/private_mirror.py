#!/usr/bin/env python3
"""Sync the irreplaceable non-code artifacts to the private mirror.

    python3 tools/private_mirror.py --dry-run
    python3 tools/private_mirror.py --push

The public repo carries code. This carries everything that cannot be
regenerated and must not be published:

* ``ledger/`` — the append-only prediction ledger, the out-of-time validator;
* ``reports/`` — generated reports and their provenance;
* research findings — HANDOFF, VERDICT and strategy documents;
* ``experiments/`` records — the multiple-testing LEDGER.csv, each experiment's
  REPORT.md, its results/*.json evidence artifacts and figures. The code
  (run.py) and spec.yaml ship publicly; the findings do not;
* ``config/thesis/`` — the watchlist, i.e. position intent;
* the pre-engine research code, whose build scripts have already been lost once
  (the panel builders lived in ``/tmp``), and the dashboard, which hardcodes the
  thesis universe.

Deliberately **not** mirrored: raw and curated market data. It is re-pullable at
quota cost, and 57k files do not belong in git. That residual gap is accepted
and documented in ``RECOVERY.md``.

Secrets are excluded from this mirror too. Private is not the same as safe: a
credential in any remote is a credential to rotate.
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks.repo_hygiene import (  # noqa: E402
    MAX_BYTES,
    check_blob,
    load_secrets,
    Report,
)
from engine import paths  # noqa: E402

#: ``(directory, glob)`` pairs to mirror. Explicit allowlist, same discipline as
#: the public ``.gitignore``: nothing ships unless it is named here.
INCLUDE = (
    ("ledger", "**/*"),
    ("reports", "**/*.md"),
    ("config/thesis", "**/*"),
    ("experiments", "LEDGER.csv"),
    ("experiments", "**/REPORT.md"),
    ("experiments", "**/results/*.json"),
    # The transaction logs behind the plotted equity curves: real ORATS quotes,
    # so they can never reach the PUBLIC repo — and the reason a reported chart
    # can be spot-checked at all, so they must reach this one.
    ("experiments", "**/results/transactions_*.csv"),
    ("experiments", "**/figures/*.png"),
    ("earnings_predictions", "**/*.py"),
    ("earnings_predictions", "**/*.md"),
    ("bt", "**/*.py"),
    ("bt", "**/*.md"),
    ("dashboard", "**/*.py"),
    ("dashboard", "**/*.html"),
    ("dashboard", "**/*.css"),
    ("dashboard", "**/*.js"),
    ("dashboard", "**/*.md"),
)

#: Individual files at the repo root.
INCLUDE_FILES = (
    "AGENTS.md",
    "STRATEGY.md",
    "INVESTMENT_PLAN.md",
    "ADVISOR_BRIEF.md",
    "STRADDLE_AUDIT.md",
    "EARNINGS_VOL_PROGRAM_PLAN.md",
)

#: Root-level loose research scripts.
INCLUDE_ROOT_GLOBS = ("*.py",)

EXCLUDE_PARTS = ("__pycache__", ".git", ".venv", "node_modules", ".pytest_cache")

#: The 1 MB cap exists to keep a CODE repo from swallowing data. Two artifact
#: kinds are deliberately exempt here, because they are the evidence the mirror
#: exists for and they are useless truncated: the per-trade transaction logs
#: behind the reported equity curves (a few MB each — a report's chart is only
#: checkable if its rows travel with it) and the prediction ledger.
LARGE_ARTIFACT_CAP = 25_000_000


def _size_cap(path: Path) -> int:
    if path.name.startswith("transactions_") or paths.LEDGER in path.parents:
        return LARGE_ARTIFACT_CAP
    return MAX_BYTES


def collect() -> tuple[list[Path], list[str]]:
    """Files to mirror, and the reasons anything was skipped."""
    found: list[Path] = []
    skipped: list[str] = []

    def consider(path: Path) -> None:
        if not path.is_file():
            return
        if any(part in EXCLUDE_PARTS for part in path.parts):
            return
        cap = _size_cap(path)
        if path.stat().st_size > cap:
            skipped.append(f"{path.relative_to(ROOT)} (>{cap:,} bytes)")
            return
        found.append(path)

    for name in INCLUDE_FILES:
        consider(ROOT / name)
    for pattern in INCLUDE_ROOT_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            consider(path)
    for directory, pattern in INCLUDE:
        base = ROOT / directory
        if not base.exists():
            continue
        for path in sorted(base.glob(pattern)):
            consider(path)

    return sorted(set(found)), skipped


def scan_for_secrets(files: list[Path]) -> Report:
    """The mirror is private, not safe. A credential here still has to rotate."""
    needles = load_secrets(paths.ENV_FILE)
    report = Report(secrets_loaded=len(needles))
    for path in files:
        report.checked += 1
        try:
            check_blob(str(path.relative_to(ROOT)), path.read_bytes(), needles, report,
                       max_bytes=_size_cap(path))
        except OSError:
            continue
    # Size and secret findings block; the data-extension rule does not apply
    # here, because reports and ledgers are exactly what this mirror is for.
    report.violations = [v for v in report.violations if v.rule in ("secret", "credential-shape", "oversize")]
    return report


def sync(target: Path, files: list[Path]) -> int:
    copied = 0
    for path in files:
        rel = path.relative_to(ROOT)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = path.read_bytes()
        if dest.exists() and dest.read_bytes() == data:
            continue
        dest.write_bytes(data)
        copied += 1
    return copied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", default=str(Path.home() / ".private_mirror" / "investing-plan"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--push", action="store_true", help="commit and push the mirror clone")
    args = ap.parse_args(argv)

    files, skipped = collect()
    total = sum(p.stat().st_size for p in files)
    print(f"private mirror: {len(files)} file(s), {total:,} bytes")
    for reason in skipped[:10]:
        print(f"  skipped: {reason}")

    report = scan_for_secrets(files)
    if not report.ok:
        print("\nBLOCKED — the mirror would carry a credential:", file=sys.stderr)
        for violation in report.violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print(f"  secret scan clean ({report.secrets_loaded} pattern(s) loaded)")

    if args.dry_run:
        by_top: dict[str, int] = {}
        for path in files:
            top = path.relative_to(ROOT).parts[0]
            by_top[top] = by_top.get(top, 0) + 1
        for top, count in sorted(by_top.items(), key=lambda kv: -kv[1]):
            print(f"    {top:<28s} {count:>4d}")
        print("\n  dry run — nothing written. Re-run with --push to sync.")
        return 0

    target = Path(args.target)
    if not (target / ".git").exists():
        print(
            f"\n{target} is not a git clone.\n"
            "Create the private repo, then:\n"
            f"  git clone <private-repo-url> {target}",
            file=sys.stderr,
        )
        return 1

    copied = sync(target, files)
    print(f"  synced {copied} changed file(s) → {target}")

    if args.push:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        subprocess.run(["git", "add", "-A"], cwd=target, check=True)
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=target, capture_output=True, text=True
        )
        if not status.stdout.strip():
            print("  nothing to commit")
            return 0
        subprocess.run(
            ["git", "commit", "-q", "-m", f"mirror sync {stamp}"], cwd=target, check=True
        )
        subprocess.run(["git", "push", "-q"], cwd=target, check=True)
        print("  pushed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
