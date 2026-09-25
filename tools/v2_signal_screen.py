#!/usr/bin/env python3
"""v2 signal screen: the long-put hypothesis screen over a PINNED snapshot.

The `engine.v2.research` move of `tools/signal_screen.py`: same features and
tables, but every read is a bounded `Repository.scan` against one snapshot,
resolved either from `--scope`'s pinned head or from an explicit
`--snapshot-id` (for reproducing a run after the head moved). The snapshot id
is written into both outputs.

Usage::

    python3 tools/v2_signal_screen.py --catalog <catalog.sqlite> \\
        --store-root <objects> [--scope shadow] [--snapshot-id snap_...] \\
        [--reports-dir reports]
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
from engine.v2.research.signal_screen import run  # noqa: E402

__all__ = ["main"]


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--scope", default="shadow")
    parser.add_argument("--snapshot-id", default=None,
                        help="read this exact snapshot instead of the scope's pinned head")
    parser.add_argument("--reports-dir", type=Path, default=ROOT / "reports")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    conn = open_catalog(args.catalog, clock=SystemClock())
    try:
        repository = Repository(conn, store=ArtifactStore(args.store_root))
        result = run(repository, reports_dir=args.reports_dir, scope=args.scope,
                     snapshot_id=args.snapshot_id)
    except DataError as exc:
        print(f"v2-signal-screen: refused: {exc.code}: {exc.problem.message}", file=sys.stderr)
        return 2
    finally:
        conn.close()
    print("\n== candidate signals vs base rate (fwd 20 trading days) ==\n", flush=True)
    print(result["table"].to_string(index=False), flush=True)
    print("\n== mcap slices (A / A2) ==\n", flush=True)
    print(result["slices"].to_string(index=False), flush=True)
    print(json.dumps({"snapshot_id": result["snapshot_id"], "paths": result["paths"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
