#!/usr/bin/env python3
"""v2 polygon fills: the real-trade pull's universe, read from a PINNED snapshot.

The `engine.v2.research` move of `engine/data/pulls/polygon_fills.py`'s read
path: the same contract inventory derived from the Tier-2 `trades` table, but
read as a bounded `Repository.scan` against one snapshot, resolved either from
`--scope`'s pinned head or from an explicit `--snapshot-id`. The snapshot id is
written into the plan. This tool fetches nothing: the legacy
`build_plan`/`execute` network half stays in `engine/data/pulls/`.

Usage::

    python3 tools/v2_polygon_fills.py --catalog <catalog.sqlite> \\
        --store-root <objects> [--scope shadow] [--snapshot-id snap_...] \\
        [--min-date 2024-08-19] [--out-dir reports]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2.data.errors import DataError  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.foundation import ArtifactStore, SystemClock  # noqa: E402
from engine.v2.ops.bootstrap import open_catalog  # noqa: E402
from engine.v2.research.polygon_fills import POLYGON_OPTIONS_START, run  # noqa: E402

__all__ = ["main"]


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--scope", default="shadow")
    parser.add_argument("--snapshot-id", default=None,
                        help="read this exact snapshot instead of the scope's pinned head")
    parser.add_argument("--min-date", default=POLYGON_OPTIONS_START)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    conn = open_catalog(args.catalog, clock=SystemClock())
    try:
        repository = Repository(conn, store=ArtifactStore(args.store_root))
        plan = run(repository, out_dir=args.out_dir, scope=args.scope,
                   snapshot_id=args.snapshot_id, min_date=args.min_date)
    except DataError as exc:
        print(f"v2-polygon-fills: refused: {exc.code}: {exc.problem.message}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    print(json.dumps(plan, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
