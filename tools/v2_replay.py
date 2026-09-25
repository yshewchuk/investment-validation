"""Replay strategies over one pinned snapshot — the v2 research CLI.

Example::

    python3 tools/v2_replay.py --catalog private/ops/catalog.sqlite \
        --store-root private/ops/objects --scope shadow --strategy STR-THRU

Reads the pinned snapshot's ``earnings_events`` and ``option_chains`` through
``engine.v2.data.Repository``, replays every named strategy, writes a JSON
report under ``--reports-dir`` and prints the per-strategy summary.
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
from engine.v2.research import _snapshot
from engine.v2.research._pricing import STRUCTURES
from engine.v2.research import replay as replay_tool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--scope", default=_snapshot.DEFAULT_SCOPE)
    parser.add_argument("--snapshot-id", default=None)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--strategy", action="append", choices=sorted(STRUCTURES),
                        default=None, help="repeatable; default is every strategy")
    parser.add_argument("--years", nargs="*", type=int, default=None,
                        help="restrict events to these event_date years")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    strategies = args.strategy or sorted(STRUCTURES)
    conn = open_catalog(args.catalog, clock=SystemClock())
    try:
        repository = Repository(conn, ArtifactStore(args.store_root))
        scope = _snapshot.DEFAULT_SCOPE if args.scope is None else args.scope
        snapshot = _snapshot.resolve_snapshot(
            repository, scope=scope, snapshot_id=args.snapshot_id
        )
        events = replay_tool._events_frame(
            repository, snapshot, years=args.years
        )
        outcome = replay_tool.run(
            repository, strategies=strategies, events=events,
            reports_dir=args.reports_dir, scope=scope,
            snapshot_id=args.snapshot_id,
        )
    except DataError as exc:
        print(
            json.dumps({"refused": exc.code, "message": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    finally:
        conn.close()

    print(f"snapshot_id: {outcome['snapshot_id']}")
    for result in outcome["results"]:
        print(json.dumps(result, sort_keys=True))
    print(f"report: {outcome['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
