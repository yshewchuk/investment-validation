"""Print a network-free Basic-tier Massive cache plan: python -m tools.v2_massive_plan.

Normal engine startup may load .env; planning never contacts providers or emits credentials.
"""
import argparse
import json
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

from engine.v2.data.errors import DataError
from engine.v2.foundation import ArtifactStore, SystemClock
from engine.v2.foundation.artifacts import ArtifactError
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.massive_plan import plan
from engine.v2.ops.stores.massive import coverage
from engine.v2.ops.stores.massive.pages import closed_day

PLAN_TIMEOUT_SECONDS = 30


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise fail("INVALID_REQUEST", "invalid planning arguments") from None


def _main(argv=None):
    parser = _Parser(description=__doc__, prog="tools.v2_massive_plan")
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--store-root", required=True, type=Path)
    parser.add_argument("--ticker", required=True, action="append", dest="tickers")
    for name in ("start", "end", "as-of"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--adjustment", required=True, choices=("raw", "split_adjusted"))
    parser.add_argument("--max-calls-per-request", required=True, type=int)
    parser.add_argument("--transport", default="rest")
    parser.add_argument("--generation")
    try:
        args = vars(parser.parse_args(argv))
        catalog, root = args.pop("catalog"), args.pop("store_root")
        generation = args.pop("generation")
        requested = plan(None, None, **args)
        if generation is not None:
            coverage.validate_generation(generation)
            observed_at = SystemClock().now()
            for unit in requested["units"]:
                closed_day(unit["request"], observed_at)
        if "\0" in str(catalog) or "\0" in str(root) or not catalog.is_file():
            raise OSError("planning paths require a regular catalog and no NUL characters")
        with closing(sqlite3.connect(catalog.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("BEGIN")
            if generation is None:
                result = plan(conn, ArtifactStore(root), **args)
            else:
                plans = [coverage.plan(conn, ArtifactStore(root), unit["request"], generation)
                         for unit in requested["units"]]
                result = {"schema_version": "massive_interval_plan.v1", "plan_only": True,
                          "policy_version": requested["policy_version"], "plans": plans,
                          "maximum_http_attempts": sum(bool(p["gaps"]) for p in plans)
                          * args["max_calls_per_request"]}
    except (OpsError, DataError, ArtifactError) as exc:
        reason = exc.problem.details.get("reason") if isinstance(exc, OpsError) else None
        print(json.dumps({"refused": exc.code, "reason": reason}))
        return 2
    except (OSError, sqlite3.Error):
        print(json.dumps({"refused": "RESOURCE_UNAVAILABLE"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


def main(argv=None):
    """Bound all worker filesystem/SQLite I/O, including raced special-file opens."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    worker = ("import json, sys; paths, arguments = json.load(sys.stdin); sys.path[:] = paths; "
              "from tools.v2_massive_plan import _main; "
              "raise SystemExit(_main(arguments))")
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", worker],
            input=json.dumps([sys.path, arguments]), capture_output=True, text=True,
            timeout=PLAN_TIMEOUT_SECONDS, check=False)
        if result.returncode not in (0, 2):
            raise OSError("planning worker failed")
    except (OSError, subprocess.TimeoutExpired):
        print(json.dumps({"refused": "RESOURCE_UNAVAILABLE"}))
        return 2
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
