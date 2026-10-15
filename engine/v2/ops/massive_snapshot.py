"""Submit saved acquisition pins for separate native snapshot publication."""
import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import SystemClock, content_hash, to_document
from engine.v2.ops import catalog, fingerprints, stages, submission
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.legacy_adapter import configured_root
from engine.v2.ops.stores.massive import snapshot


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise fail("INVALID_REQUEST", "invalid Massive snapshot arguments") from None


def main(argv=None, *, clock=None):
    parser = _Parser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--acquisition-job", required=True)
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--parent-snapshot")
    parser.add_argument("--scope")
    parser.add_argument("--head-generation", type=int)
    try:
        args = parser.parse_args(argv)
        params = snapshot.Parameters(args.acquisition_job,
            {"acquisition.json": args.acquisition_job + "#massive_acquisition"}, (args.acquisition_job,),
            args.parent_snapshot, args.scope, args.head_generation)
        job = JobSpec(kind=snapshot.KIND, implementation_ref="pending", environment_ref="pending",
            spec_hash=None, parameters=to_document(params), output_namespace="shadow", resource_class="io_fetch",
            retry_policy_ref="bounded", checkpoint_contract_ref=snapshot.SCHEMA,
            dependency_job_ids=(args.acquisition_job,))
        if snapshot.parameter_problems(job, params):
            raise fail("INVALID_REQUEST", "invalid Massive acquisition dependency")
        source = configured_root()
        job = replace(job, implementation_ref=content_hash(fingerprints.worker_source_manifest(source)),
                                 environment_ref=content_hash(fingerprints.environment_identity(1)))
        with closing(catalog.connect(args.root / "catalog.sqlite", must_exist=True)) as conn:
            result = submission.submit(conn, stages.registry(),
                submission.NamespacePolicy({"operator": frozenset({"shadow"})}),
                SubmitRequest(namespace="shadow", principal="operator", idempotency_key=args.idempotency_key,
                              job=job), clock=clock or SystemClock())
        print(json.dumps(to_document(result), sort_keys=True))
        return 0
    except OpsError as error:
        print(json.dumps({"refused": error.code}))
    except (OSError, sqlite3.Error, ValueError):
        print(json.dumps({"refused": "RESOURCE_UNAVAILABLE"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
