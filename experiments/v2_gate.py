"""Register and evaluate one reconstructed-history gate arm on pinned native inputs."""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

from engine.v2 import foundation
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.ops.catalog import connect
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from experiments import gate_report as gate
from experiments import lib, record_export
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
    parser.add_argument("--exp-number", default=None)
    parser.add_argument("--experiments-root", type=Path, default=lib.EXPERIMENTS_DIR)
    parser.add_argument("--no-record-export", action="store_true")
    args = parser.parse_args(argv)
    if not native.valid_registration_read_budget(args.max_registration_bytes):
        parser.error("--max-registration-bytes must be a positive SQLite-safe byte budget")
    conn, store, registration, attempted = None, None, None, False
    try:
        if args.purpose != "sweep":
            raise fail("HOLDOUT_ACCESS_DENIED", "gate evaluation admits sweep research only")
        spec_document = record_export.read_spec_document(args.spec)
        spec = experiment_spec_from_document(spec_document)
        gate.validate_gate_spec(spec)
        if args.action == "run" and not args.no_ledger and args.ledger is None:
            raise fail("INVALID_EXPERIMENT_SPEC", "recorded gate evaluation requires --ledger")
        store = foundation.ArtifactStore(args.store_root)
        conn = connect(args.catalog, must_exist=True)
        repository = Repository(conn, store, pin_snapshots=True)
        binding = dict(code_root=Path(__file__).resolve().parents[1], as_of_month=args.as_of_month,
                       scope=args.scope, event_ids=args.event_id)
        if args.action == "register":
            registration = native.register_native(conn, store, repository, spec, **binding,
                                                  max_evidence_bytes=args.max_registration_bytes)
            print(json.dumps({"registered": registration.variant_id}))
            return 0
        registration = native.read_native_registration(conn, store, spec,
                                                       max_evidence_bytes=args.max_registration_bytes)
        if "final_read" in registration.document:
            raise fail("HOLDOUT_ACCESS_DENIED", "gate evaluation has no final-read authority")
        registration = native.require_native_registration(conn, store, repository, spec,
                                                          expected=registration, **binding)
        exp_number = None
        experiment_id = registration.document["execution_plan"]["experiment_id"]
        if args.no_record_export:
            print("WARNING: no EXP number; run record NOT exported", file=sys.stderr)
            if args.exp_number is not None:
                record_export.resolve_exp_number(
                    args.experiments_root, experiment_id, args.exp_number)
        else:
            exp_number = record_export.resolve_exp_number(
                args.experiments_root, experiment_id, args.exp_number)
        recording = dict(no_ledger=args.no_ledger, ledger_path=args.ledger)
        started = time.monotonic()
        saved = outcomes.replay_native_outcome(conn, store, registration, **recording)
        replayed = saved is not None
        if saved is None:
            attempted = True
            result = gate.gate_result(repository, registration, spec)
            report = gate.render_gate_report(registration, result, spec=spec, no_ledger=args.no_ledger)
            saved = outcomes.publish_native_outcome(conn, store, registration, report=report,
                                                    attempted=True, **recording)
        request_dict = {
            "action": args.action,
            "as_of_month": args.as_of_month,
            "scope": args.scope,
            "event_ids": args.event_id,
            "purpose": args.purpose,
            "no_ledger": args.no_ledger,
        }
        if saved["outcome"]["failure_code"] is not None:
            record, refusal = record_export.export_or_skip(
                args.experiments_root, registration, spec_document, request_dict,
                saved, exp_number, replayed, started, None)
            if refusal is not None:
                print(json.dumps(refusal))
                return 2
            print(json.dumps({"refused": saved["outcome"]["failure_code"], "replayed": replayed,
                              "record_export": record}))
            return 2
        path = outcomes.export_native_report(conn, store, registration, **recording)
        record, refusal = record_export.export_or_skip(
            args.experiments_root, registration, spec_document, request_dict,
            saved, exp_number, replayed, started, path or None)
        if refusal is not None:
            print(json.dumps(refusal))
            return 2
        print(json.dumps({"report": str(path), "variant_id": registration.variant_id,
                          "recording_mode": saved["receipt"]["recording_mode"],
                          "record_export": record}))
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
        print(json.dumps({"refused": problem.code,
                          **({"details": problem.details} if problem.code == "RESOURCE_LIMIT_EXCEEDED" else {}), "refusal_ref":
                          foundation.to_document(ref) if ref else problem.details.get("refusal_ref")}))
        return 2
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
