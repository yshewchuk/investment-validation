"""Submit one bounded cache-first request; run it separately with ops serve."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import SystemClock, content_hash, to_document
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.massive_plan import SOURCE, plan
from engine.v2.ops.stages import registry
from engine.v2.ops.stores.massive.acquisition import KIND, SCHEMA, Parameters
from engine.v2.ops.submission import NamespacePolicy, submit


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise fail("INVALID_REQUEST", "invalid Massive submission arguments") from None


def main(argv=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--idempotency-key", required=True)
    for name in ("ticker", "dataset", "start", "end", "as-of", "adjustment"):
        parser.add_argument("--" + name, required=True)
    try:
        args = vars(parser.parse_args(argv))
        root, key = args.pop("root"), args.pop("idempotency_key")
        scoped = plan(None, None, tickers=[args["ticker"]], datasets=[args["dataset"]],
                      start=args["start"], end=args["end"], as_of=args["as_of"],
                      adjustment=args["adjustment"], max_calls_per_request=1)
        params = Parameters(**args, expected_ids=(scoped["units"][0]["request_hash"],))
        source = Path(__file__).resolve().parents[3]
        job = JobSpec(kind=KIND, implementation_ref=content_hash(worker_source_manifest(source)),
                      environment_ref=content_hash(environment_identity(1)), spec_hash=None,
                      parameters=to_document(params), output_namespace="shadow", resource_class="io_fetch",
                      provider_budget_ref=SOURCE, retry_policy_ref="bounded", checkpoint_contract_ref=SCHEMA)
        with closing(connect(root / "catalog.sqlite", must_exist=True)) as conn:
            result = submit(conn, registry(), NamespacePolicy({"operator": frozenset({"shadow"})}),
                            SubmitRequest(namespace="shadow", principal="operator", idempotency_key=key,
                                          job=job), clock=SystemClock())
        print(json.dumps(to_document(result), sort_keys=True))
        return 0
    except OpsError as exc:
        print(json.dumps({"refused": exc.code}))
    except (OSError, sqlite3.Error, ValueError):
        print(json.dumps({"refused": "RESOURCE_UNAVAILABLE"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
