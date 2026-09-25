"""Replay strategies over one pinned snapshot and publish the v2 trades table.

Example::

    python3 tools/v2_build_trades.py --catalog private/ops/catalog.sqlite \
        --store-root private/ops/objects --scope shadow --strategy STR-THRU

The legacy ``engine/build_trades.py`` and ``tools/reconcile_trades.py`` are
untouched; this writes a NEW dataset version of the same ``trades`` table
under the snapshot scope alongside them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactStore, SystemClock
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.research import _snapshot, build_trades
from engine.v2.research._pricing import STRUCTURES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--scope", default=_snapshot.DEFAULT_SCOPE)
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--strategy", action="append", choices=sorted(STRUCTURES),
                        default=None, help="repeatable; default is every strategy")
    parser.add_argument("--years", nargs="*", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="plan and price, commit nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    strategies = args.strategy or sorted(STRUCTURES)
    conn = open_catalog(args.catalog, clock=SystemClock())
    try:
        repository = Repository(conn, ArtifactStore(args.store_root))
        scope = _snapshot.DEFAULT_SCOPE if args.scope is None else args.scope
        outcome = build_trades.run(
            repository, strategies=strategies, years=args.years, scope=scope,
            snapshot_id=args.snapshot_id, reports_dir=args.reports_dir,
            dry_run=args.dry_run,
        )
    except DataError as exc:
        print(
            json.dumps({"refused": exc.code, "message": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    finally:
        conn.close()

    summary = {key: value for key, value in outcome.items() if key != "trades"}
    print(json.dumps(summary, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
