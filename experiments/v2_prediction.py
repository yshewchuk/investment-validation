"""One preregistered native prediction target, including a no-ledger smoke CLI."""
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
from experiments import lib
from experiments import magnitude_history_report, magnitude_report
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from experiments import prediction_report as prediction
from experiments import record_export
from experiments.magnitude_inputs import admit_magnitude_inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    for name in ("spec", "catalog", "store-root", "as-of-month"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scope", default="shadow")
    parser.add_argument("--event-id", action="append")
    parser.add_argument("--no-ledger", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--target", choices=("positive_move", "absolute_move"), default="positive_move")
    parser.add_argument("--magnitude-training-start")
    parser.add_argument("--magnitude-labels-before")
    parser.add_argument("--magnitude-model", choices=("pooled_mean", "history_mean", "history_ridge", "history_hgb"))
    parser.add_argument("--magnitude-feature-snapshot-id")
    parser.add_argument("--magnitude-history-start")
    parser.add_argument("--max-registration-bytes", type=int, default=native.MAX_NATIVE_EVIDENCE_BYTES)
    parser.add_argument("--exp-number", default=None)
    parser.add_argument("--experiments-root", type=Path, default=lib.EXPERIMENTS_DIR)
    parser.add_argument("--no-record-export", action="store_true")
    args = parser.parse_args(argv)
    if not native.valid_registration_read_budget(args.max_registration_bytes):
        parser.error("--max-registration-bytes must be a positive SQLite-safe byte budget")
    magnitude = args.target == "absolute_move"
    clocks = (args.magnitude_training_start, args.magnitude_labels_before)
    if (magnitude and not all(clocks)) or (not magnitude and any(clock is not None for clock in clocks)):
        parser.error("absolute_move requires both magnitude clocks; positive_move accepts neither")
    fitted = args.magnitude_model in ("history_ridge", "history_hgb")
    history = args.magnitude_model == "history_mean" or fitted
    history_options = (args.magnitude_feature_snapshot_id, args.magnitude_history_start)
    if (not magnitude and args.magnitude_model is not None) or (
            history and not all(history_options)) or (not history and any(v is not None for v in history_options)):
        parser.error("history_mean/history_ridge/history_hgb require absolute_move and both history options; other models accept neither")
    reporter = magnitude_history_report if history else magnitude_report if magnitude else prediction
    store, conn, registration, attempted = None, None, None, False
    try:
        store = foundation.ArtifactStore(args.store_root)
        spec_document = json.loads(Path(args.spec).read_text())
        spec = experiment_spec_from_document(spec_document)
        reporter.validate_prediction_spec(spec)
        if args.action == "run" and not args.no_ledger and args.ledger is None:
            raise fail("INVALID_EXPERIMENT_SPEC", "recorded execution requires --ledger")
        conn = connect(args.catalog, must_exist=True)
        repository = Repository(conn, store, pin_snapshots=True)
        binding = dict(code_root=Path(__file__).resolve().parents[1], as_of_month=args.as_of_month,
                       scope=args.scope, event_ids=args.event_id)
        if magnitude:
            binding.update(magnitude_training_start=clocks[0], magnitude_labels_before=clocks[1])
        if history:
            binding.update(magnitude_feature_snapshot_id=history_options[0], magnitude_history_start=history_options[1])
        if fitted:
            binding["magnitude_fit_model"] = args.magnitude_model
        if args.action == "register":
            registration = native.register_native(conn, store, repository, spec, **binding)
            print(json.dumps({"registered": registration.variant_id}))
            return 0
        registration = native.read_native_registration(conn, store, spec,
                                                       max_evidence_bytes=args.max_registration_bytes)
        if magnitude:
            registration, _ = admit_magnitude_inputs(conn, store, repository, spec, **binding)
        else:
            registration = native.require_native_registration(conn, store, repository, spec, expected=registration, **binding)
        exp_number = None
        if args.no_record_export:
            print("WARNING: no EXP number; run record NOT exported", file=sys.stderr)
        else:
            experiment_id = registration.document["execution_plan"]["experiment_id"]
            exp_number = record_export.resolve_exp_number(
                args.experiments_root, experiment_id, args.exp_number)
        recording = dict(no_ledger=args.no_ledger, ledger_path=args.ledger)
        started = time.monotonic()
        saved = outcomes.replay_native_outcome(conn, store, registration, **recording)
        replayed = saved is not None
        if saved is None:
            attempted = True
            result = reporter.prediction_result(repository, registration)
            report = reporter.render_prediction_report(registration, result, spec=spec, no_ledger=args.no_ledger)
            saved = outcomes.publish_native_outcome(conn, store, registration, report=report, attempted=True, **recording)
        request_dict = {
            "action": args.action,
            "as_of_month": args.as_of_month,
            "scope": args.scope,
            "event_ids": args.event_id,
            "no_ledger": args.no_ledger,
        }
        if saved["outcome"]["failure_code"]:
            record = "skipped"
            if exp_number is not None:
                try:
                    record = record_export.export_native_run(
                        args.experiments_root, registration, spec_document, request_dict,
                        saved, exp_number, replayed, started, report_path=None)
                except (OpsError, OSError, ValueError):
                    print(json.dumps({"refused": "record_export_failed"}))
                    return 2
            print(json.dumps({"refused": saved["outcome"]["failure_code"],
                              "receipt_ref": saved["receipt_ref"],
                              "record_export": record}))
            return 2
        report_path = outcomes.export_native_report(conn, store, registration, **recording)
        record = "skipped"
        if exp_number is not None:
            try:
                record = record_export.export_native_run(
                    args.experiments_root, registration, spec_document, request_dict,
                    saved, exp_number, replayed, started, report_path=report_path or None)
            except (OpsError, OSError, ValueError):
                print(json.dumps({"refused": "record_export_failed"}))
                return 2
        print(json.dumps({"report": str(report_path), "variant_id": registration.variant_id,
                          "recording_mode": saved["receipt"]["recording_mode"],
                          "record_export": record}))
        return 0
    except (OpsError, DataError, foundation.ArtifactError, ValueError, TypeError, KeyError,
            OSError, sqlite3.Error, RuntimeError) as error:
        problem = error.problem if isinstance(error, (OpsError, DataError)) else fail(
            "RESOURCE_UNAVAILABLE" if isinstance(error, (foundation.ArtifactError, OSError, sqlite3.Error)) else
            "WORKER_FAILED" if attempted else "INVALID_EXPERIMENT_SPEC",
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
