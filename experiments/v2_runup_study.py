"""Register/run one pinned RUNUP comparison with immutable no-ledger JSON evidence."""
import argparse
import json
import sqlite3
import sys
from contextlib import redirect_stdout

from engine.v2.data.documents import reject_duplicate_keys
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore, canonical_json
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments.native_gate_worker import read_native_staged_input
from experiments.native_registration import MAX_NATIVE_EVIDENCE_BYTES, valid_registration_read_budget
from experiments.registered_runup import comparison_request, register_runup, run_runup


def _spec(path):
    data = read_native_staged_input(path)
    try:
        return experiment_spec_from_document(json.loads(data, object_pairs_hook=reject_duplicate_keys))
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise fail("INVALID_EXPERIMENT_SPEC", "RUNUP requires bounded valid JSON specification") from None
    except OpsError as error:
        if error.code != "INVALID_REQUEST":
            raise
        raise fail("INVALID_EXPERIMENT_SPEC", "RUNUP specification is incomplete") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    for name in ("spec", "catalog", "store-root", "snapshot-id", "scope", "as-of-month"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--event-id", action="append", required=True)
    parser.add_argument("--max-evidence-bytes", type=int, default=MAX_NATIVE_EVIDENCE_BYTES)
    args = parser.parse_args(argv)
    if not valid_registration_read_budget(args.max_evidence_bytes):
        parser.error("--max-evidence-bytes must be a positive SQLite-safe byte budget")
    conn = None
    try:
        spec = _spec(args.spec)
        comparison_request(spec, args.event_id)
        conn = connect(args.catalog, must_exist=True)
        store = ArtifactStore(args.store_root)
        repository = Repository(conn, store)
        binding = dict(snapshot=repository.resolve(args.snapshot_id), scope=args.scope,
                       as_of_month=args.as_of_month, event_ids=args.event_id)
        with redirect_stdout(sys.stderr):
            if args.action == "register":
                registration = register_runup(conn, store, repository, spec, **binding)
                result = {"registered": registration.variant_id, "run_id": registration.run_id}
            else:
                result = run_runup(conn, store, repository, spec, **binding,
                                   max_evidence_bytes=args.max_evidence_bytes)
        status = 2 if result.get("outcome", {}).get("failure_code") else 0
    except (OpsError, DataError) as error:
        result, status = {"refused": error.code, "details": error.problem.details}, 2
    except (ArtifactError, OSError, sqlite3.Error):
        result, status = {"refused": "RESOURCE_UNAVAILABLE"}, 2
    finally:
        if conn is not None:
            conn.close()
    print(canonical_json(result))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
