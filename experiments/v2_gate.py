"""Register and evaluate one reconstructed-history gate arm on pinned native inputs."""
import argparse
import json
import sqlite3
from pathlib import Path

from engine.v2 import foundation
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments import gate_report as gate
from experiments import native_outcomes as outcomes
from experiments import native_registration as native


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    for name in ("spec", "catalog", "store-root", "as-of-month"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scope", default="shadow")
    parser.add_argument("--event-id", action="append")
    parser.add_argument("--purpose", default="sweep")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--max-registration-bytes", type=int, default=native.MAX_NATIVE_EVIDENCE_BYTES)
    args = parser.parse_args(argv)
    if not 0 < args.max_registration_bytes < 2**62:
        parser.error("--max-registration-bytes must be a positive SQLite-safe byte budget")
    conn, store, registration, attempted = None, None, None, False
    try:
        if args.purpose != "sweep":
            raise fail("HOLDOUT_ACCESS_DENIED", "gate evaluation admits sweep research only")
        spec = experiment_spec_from_document(json.loads(Path(args.spec).read_text()))
        gate.validate_gate_spec(spec)
        if args.action == "run" and not args.no_ledger and args.ledger is None:
            raise fail("INVALID_EXPERIMENT_SPEC", "recorded gate evaluation requires --ledger")
        store = foundation.ArtifactStore(args.store_root)
        conn = connect(args.catalog, must_exist=True)
        repository = Repository(conn, store)
        binding = dict(code_root=Path(__file__).resolve().parents[1], as_of_month=args.as_of_month,
                       scope=args.scope, event_ids=args.event_id)
        if args.action == "register":
            registration = native.register_native(conn, store, repository, spec, **binding)
            print(json.dumps({"registered": registration.variant_id}))
            return 0
        registration = native.read_native_registration(conn, store, spec,
                                                       max_evidence_bytes=args.max_registration_bytes)
        if "final_read" in registration.document:
            raise fail("HOLDOUT_ACCESS_DENIED", "gate evaluation has no final-read authority")
        registration = native.require_native_registration(conn, store, repository, spec,
                                                         expected=registration, **binding)
        recording = dict(no_ledger=args.no_ledger, ledger_path=args.ledger)
        saved = outcomes.replay_native_outcome(conn, store, registration, **recording)
        if saved is None:
            attempted = True
            result = gate.gate_result(repository, registration, spec)
            report = gate.render_gate_report(registration, result, spec=spec, no_ledger=args.no_ledger)
            saved = outcomes.publish_native_outcome(conn, store, registration, report=report,
                                                    attempted=True, **recording)
        if saved["outcome"]["failure_code"] is not None:
            print(json.dumps({"refused": saved["outcome"]["failure_code"], "replayed": True}))
            return 2
        path = outcomes.export_native_report(conn, store, registration, **recording)
        print(json.dumps({"report": str(path), "variant_id": registration.variant_id,
                          "recording_mode": saved["receipt"]["recording_mode"]}))
        return 0
    except (OpsError, DataError, foundation.ArtifactError, ValueError, TypeError, KeyError,
            OSError, sqlite3.Error, RuntimeError) as error:
        problem = error.problem if isinstance(error, (OpsError, DataError)) else fail(
            "RESOURCE_UNAVAILABLE" if isinstance(error, (foundation.ArtifactError, OSError, sqlite3.Error)) else
            "WORKER_FAILED" if attempted else "INVALID_EXPERIMENT_SPEC",
            "native gate request could not be completed").problem
        if problem.code == "INVALID_REQUEST":
            problem = fail("INVALID_EXPERIMENT_SPEC", "gate specification is incomplete").problem
        if (attempted and isinstance(error, DataError) and not problem.retryable
                and problem.code not in {*outcomes.STAGES, "SNAPSHOT_UNRESOLVED"}):
            problem = fail("EXPERIMENT_VARIANT_FAILED", "gate input failed", details={"data_code": problem.code}).problem
        ref = None
        try:
            if store is not None:
                ref = outcomes.publish_native_refusal(store, problem=problem,
                    request={"action": args.action, "holdout_as_of_month": args.as_of_month}, registration=registration)
            if registration is not None and problem.code in outcomes.STAGES and args.action == "run":
                outcomes.publish_native_outcome(conn, store, registration, problem=problem,
                    attempted=attempted, no_ledger=args.no_ledger, ledger_path=args.ledger)
        except OpsError as failure:
            if not (problem.code == "HOLDOUT_ACCESS_DENIED" and failure.code == "EXPERIMENT_IDENTITY_CONFLICT"):
                problem, ref = failure.problem, None
        print(json.dumps({"refused": problem.code, "refusal_ref":
                          foundation.to_document(ref) if ref else problem.details.get("refusal_ref")}))
        return 2
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
