"""Reconcile the v2 ``trades`` table to the canonical event universe.

Example::

    python3 tools/v2_reconcile_trades.py --catalog private/ops/catalog.sqlite \
        --store-root private/ops/objects --scope shadow

The v2 counterpart of ``tools/reconcile_trades.py``: it tombstones only the
non-canonical simulated rows as a new ``trades`` dataset version under the
snapshot scope. The legacy tool and the legacy mutable table are untouched;
``--snapshot-id`` reconciles the pinned snapshot instead of the scope head
(use ``--dry-run`` when the head has moved).
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
from engine.v2.research import _snapshot, reconcile_trades


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--scope", default=_snapshot.DEFAULT_SCOPE)
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--dry-run", action="store_true",
                        help="compute the removals, commit nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    conn = open_catalog(args.catalog, clock=SystemClock())
    try:
        repository = Repository(conn, ArtifactStore(args.store_root))
        scope = _snapshot.DEFAULT_SCOPE if args.scope is None else args.scope
        outcome = reconcile_trades.run(
            repository, scope=scope, snapshot_id=args.snapshot_id,
            reports_dir=args.reports_dir, dry_run=args.dry_run,
        )
    except DataError as exc:
        print(
            json.dumps({"refused": exc.code, "message": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    finally:
        conn.close()

    print(json.dumps(outcome, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
