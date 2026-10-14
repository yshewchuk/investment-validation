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
from engine.v2.ops.legacy_adapter import configured_root
from engine.v2.ops.massive_plan import SOURCE, plan
from engine.v2.ops.stages import registry
from engine.v2.ops.stores.massive.acquisition import (
    HISTORY_KIND,
    HISTORY_SCHEMA,
    KIND,
    SCHEMA,
    Parameters,
    scope_plan,
    validate_as_of,
)
from engine.v2.ops.stores.massive.pages import closed_day
from engine.v2.ops.submission import NamespacePolicy, submit


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise fail("INVALID_REQUEST", "invalid Massive submission arguments") from None


def main(argv=None, *, clock=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--generation")
    parser.add_argument("--max-http-calls", type=int, default=1)
    for name in ("ticker", "dataset", "start", "end", "as-of", "adjustment"):
        parser.add_argument("--" + name, required=True)
    try:
        args = vars(parser.parse_args(argv))
        root, key = args.pop("root"), args.pop("idempotency_key")
        scoped = plan(None, None, tickers=[args["ticker"]], datasets=[args["dataset"]],
                      start=args["start"], end=args["end"], as_of=args["as_of"],
                      adjustment=args["adjustment"], max_calls_per_request=1)
        clock = clock if clock is not None else SystemClock()
        validate_as_of(args["as_of"], clock)
        params = Parameters(**args, expected_ids=(scoped["units"][0]["request_hash"],))
        scope_plan(params)
        if params.generation is not None:
            closed_day(scoped["units"][0]["request"], clock.now())
        kind, schema = (HISTORY_KIND, HISTORY_SCHEMA) if params.generation is not None else (KIND, SCHEMA)
        source = configured_root()
        job = JobSpec(kind=kind, implementation_ref=content_hash(worker_source_manifest(source)),
                      environment_ref=content_hash(environment_identity(1)), spec_hash=None,
                      parameters=to_document(params), output_namespace="shadow", resource_class="io_fetch",
                      provider_budget_ref=SOURCE, retry_policy_ref="bounded", checkpoint_contract_ref=schema)
        with closing(connect(root / "catalog.sqlite", must_exist=True)) as conn:
            result = submit(conn, registry(), NamespacePolicy({"operator": frozenset({"shadow"})}),
                            SubmitRequest(namespace="shadow", principal="operator", idempotency_key=key,
                                          job=job), clock=clock)
        print(json.dumps(to_document(result), sort_keys=True))
        return 0
    except OpsError as exc:
        print(json.dumps({"refused": exc.code}))
    except (OSError, sqlite3.Error, ValueError):
        print(json.dumps({"refused": "RESOURCE_UNAVAILABLE"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
