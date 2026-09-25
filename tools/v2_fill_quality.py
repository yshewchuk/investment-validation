#!/usr/bin/env python3
"""v2 fill quality: real Polygon trades vs ORATS quotes on a PINNED snapshot.

The `engine.v2.research` move of `tools/fill_quality.py`: the same join and
liquidity-bucket summary, but both table reads are bounded `Repository.scan`
calls against one snapshot, resolved either from `--scope`'s pinned head or
from an explicit `--snapshot-id`. The snapshot id is written into the output.

Usage::

    python3 tools/v2_fill_quality.py --catalog <catalog.sqlite> \\
        --store-root <objects> [--scope shadow] [--snapshot-id snap_...] \\
        [--since 2026-07-30] [--csv out.csv] [--reports-dir reports]
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
from engine.v2.research.fill_quality import run  # noqa: E402

__all__ = ["main"]


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--scope", default="shadow")
    parser.add_argument("--snapshot-id", default=None,
                        help="read this exact snapshot instead of the scope's pinned head")
    parser.add_argument("--since", default=None,
                        help="restrict to contract-days on/after this date (e.g. 2026-07-30)")
    parser.add_argument("--csv", default=None, help="also write the joined rows here")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    conn = open_catalog(args.catalog, clock=SystemClock())
    try:
        repository = Repository(conn, store=ArtifactStore(args.store_root))
        result = run(repository, reports_dir=args.reports_dir, scope=args.scope,
                     snapshot_id=args.snapshot_id, since=args.since, csv=args.csv)
    except DataError as exc:
        print(f"v2-fill-quality: refused: {exc.code}: {exc.problem.message}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    print(
        f"\nFill quality, measured: {result['rows']:,} contract-days with BOTH an "
        f"ORATS quote and a Polygon trade ({result['contracts']:,} contracts).\n",
        flush=True,
    )
    print(result["summary"].to_string(index=False), flush=True)
    print(json.dumps({"snapshot_id": result["snapshot_id"], "paths": result["paths"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
