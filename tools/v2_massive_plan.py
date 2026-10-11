"""Print a network-free Basic-tier Massive cache plan: python -m tools.v2_massive_plan.

Normal engine startup may load .env; planning never contacts providers or emits credentials.
"""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from engine.v2.data.errors import DataError
from engine.v2.foundation import ArtifactStore
from engine.v2.foundation.artifacts import ArtifactError
from engine.v2.ops.errors import OpsError
from engine.v2.ops.massive_plan import plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--ticker", required=True, action="append", dest="tickers")
    for name in ("start", "end", "as-of"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--adjustment", required=True, choices=("raw", "split_adjusted"))
    parser.add_argument("--max-calls-per-request", required=True, type=int)
    parser.add_argument("--transport", default="rest", choices=("rest", "s3"))
    args = vars(parser.parse_args(argv))
    catalog, root = args.pop("catalog"), args.pop("store_root")
    try:
        plan(None, None, **args)
        with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("BEGIN")
            result = plan(conn, ArtifactStore(root), **args)
    except (OpsError, DataError, ArtifactError) as exc:
        reason = exc.problem.details.get("reason") if isinstance(exc, OpsError) else None
        print(json.dumps({"refused": exc.code, "reason": reason}))
        return 2
    except (OSError, sqlite3.Error):
        print(json.dumps({"refused": "RESOURCE_UNAVAILABLE"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
