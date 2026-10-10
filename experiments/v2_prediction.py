"""One preregistered native prediction target, including a no-ledger smoke CLI."""
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
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from experiments import prediction_report as prediction


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    for name in ("spec", "catalog", "store-root", "as-of-month"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scope", default="shadow")
    parser.add_argument("--event-id", action="append")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args(argv)
    store, conn, registration, attempted = None, None, None, False
    try:
        store = foundation.ArtifactStore(args.store_root)
        spec = experiment_spec_from_document(json.loads(Path(args.spec).read_text()))
        prediction.validate_prediction_spec(spec)
        if args.action == "run" and not args.no_ledger and args.ledger is None:
            raise fail("INVALID_EXPERIMENT_SPEC", "recorded execution requires --ledger")
        conn = connect(args.catalog, must_exist=True)
        repository = Repository(conn, store)
        binding = dict(code_root=Path(__file__).resolve().parents[1], as_of_month=args.as_of_month,
                       scope=args.scope, event_ids=args.event_id)
        if args.action == "register":
            registration = native.register_native(conn, store, repository, spec, **binding)
            print(json.dumps({"registered": registration.variant_id}))
            return 0
        registration = native.read_native_registration(conn, store, spec)
        registration = native.require_native_registration(conn, store, repository, spec, expected=registration, **binding)
        recording = dict(no_ledger=args.no_ledger, ledger_path=args.ledger)
        saved = outcomes.replay_native_outcome(conn, store, registration, **recording)
        if saved is None:
            attempted = True
            result = prediction.prediction_result(repository, registration)
            report = prediction.render_prediction_report(registration, result, spec=spec, no_ledger=args.no_ledger)
            saved = outcomes.publish_native_outcome(conn, store, registration, report=report, attempted=True, **recording)
        if saved["outcome"]["failure_code"]:
            print(json.dumps({"refused": saved["outcome"]["failure_code"], "receipt_ref": saved["receipt_ref"]}))
            return 2
        report_path = outcomes.export_native_report(conn, store, registration, **recording)
        print(json.dumps({"report": str(report_path), "variant_id": registration.variant_id,
                          "recording_mode": saved["receipt"]["recording_mode"]}))
        return 0
    except (OpsError, DataError, foundation.ArtifactError, ValueError, TypeError, KeyError,
            OSError, sqlite3.Error, RuntimeError) as error:
        problem = error.problem if isinstance(error, (OpsError, DataError)) else fail(
            "RESOURCE_UNAVAILABLE" if isinstance(error, (foundation.ArtifactError, OSError, sqlite3.Error)) else
            "EXPERIMENT_VARIANT_FAILED" if attempted else "INVALID_EXPERIMENT_SPEC",
            "native prediction request could not be completed").problem
        if problem.code == "INVALID_REQUEST":
            problem = fail("INVALID_EXPERIMENT_SPEC", "prediction specification is incomplete").problem
        if (attempted and isinstance(error, DataError) and not problem.retryable
                and problem.code not in {*outcomes.STAGES, "SNAPSHOT_UNRESOLVED"}):
            problem = fail("EXPERIMENT_VARIANT_FAILED", "prediction input failed", details={"data_code": problem.code}).problem
        ref = None
        try:
            if store is not None:
                ref = outcomes.publish_native_refusal(store, problem=problem,
                    request={"action": args.action, "holdout_as_of_month": args.as_of_month}, registration=registration)
            if registration is not None and problem.code in outcomes.STAGES and args.action == "run":
                outcomes.publish_native_outcome(conn, store, registration, problem=problem, attempted=attempted,
                                               no_ledger=args.no_ledger, ledger_path=args.ledger)
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
