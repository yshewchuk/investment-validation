"""Immutable native outcomes and replay; filesystem effects stay outside SQL."""
import json
import os
import sqlite3
import tempfile
from datetime import date, datetime, timezone
from functools import wraps
from pathlib import Path

from engine.v2 import foundation
from engine.v2.contracts import ArtifactRef
from engine.v2.ops.catalog import transaction
from engine.v2.ops.errors import OpsError, fail
from experiments.lib import LedgerError, ledger_append
from experiments.native_registration import verify_native_registration

SCHEMA = "native_experiment_outcome.v1.0"
OUTCOME_FIELDS = {"schema_version", "run_id", "variant_id", "attempted_variants", "failure_code",
                  "failure_details", "report_ref", "ledger_row", "ledger_destination"}
STAGES = {"FEATURE_LOOKAHEAD": "refused", "HOLDOUT_ACCESS_DENIED": "refused",
          "EXPERIMENT_VARIANT_FAILED": "failed"}


def _publish(store, document, schema=SCHEMA):
    return store.publish_bytes(foundation.canonical_json(document).encode(), schema_ref=schema)


def publish_native_refusal(store, *, problem, request, registration=None):
    """Private receipt only; no raw exception text, report, catalog or CSV write."""
    document = {"schema_version": "native_experiment_refusal.v1.0",
                "failure_code": problem.code, "request": request,
                "variant_id": registration.variant_id if registration else None}
    try:
        return _publish(store, document, document["schema_version"])
    except (OSError, foundation.ArtifactError):
        raise fail("RESOURCE_UNAVAILABLE", "native refusal publication was interrupted") from None


def _conflict(store, registration):
    error = fail("EXPERIMENT_IDENTITY_CONFLICT", "native outcome differs from its immutable identity")
    ref = publish_native_refusal(store, problem=error.problem,
                                 request={"run_id": getattr(registration, "run_id", None)})
    return fail(error.code, error.problem.message, details={"refusal_ref": foundation.to_document(ref)})


def _slot(no_ledger, ledger_path):
    if type(no_ledger) is not bool or (not no_ledger and ledger_path is None):
        raise fail("INVALID_EXPERIMENT_SPEC", "recorded native outcomes require a ledger destination")
    if no_ledger:
        return "native_smoke", None, None
    raw = os.fspath(ledger_path)
    if not raw:
        raise fail("INVALID_EXPERIMENT_SPEC", "recorded ledger destination must not be empty")
    path = Path(raw)
    if path.is_symlink():
        raise fail("INVALID_EXPERIMENT_SPEC", "recorded ledger destination must not be a symbolic link")
    path = path.resolve()
    if path.is_dir():
        raise fail("INVALID_EXPERIMENT_SPEC", "recorded ledger destination must name a file")
    return "native_outcome", foundation.content_hash(str(path)), path


def _evidence(conn, registration):
    row = conn.execute("SELECT evidence_json FROM experiment_runs WHERE run_id=?",
                       (registration.run_id,)).fetchone()
    evidence = json.loads(row[0])
    if any(key + "_receipt" in evidence and key not in evidence for key in ("native_smoke", "native_outcome")):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native completion is missing its outcome intent")
    return evidence


def _report_path(store, registration, key, report=None, report_ref=None):
    mode = "smoke" if key == "native_smoke" else "recorded"
    path = store.root / "native_reports" / registration.run_id / mode / "REPORT.md"
    if report is not None:
        stored = store.root / report_ref["storage_key"]
        if (path.is_symlink() or path.resolve() != path or (path.exists()
                and (not path.is_file() or path.read_bytes() != report
                     or (stored.exists() and path.samefile(stored))))):
            raise _conflict(store, registration)
    return path


def _reserve(conn, registration, key, ref, expected=None):
    """Only catalog reads/writes inside this transaction; preserve other evidence."""
    with transaction(conn):
        evidence = _evidence(conn, registration)
        if (evidence.get("variant_id") != registration.variant_id
                or (expected is not None and evidence.get(expected[0]) != expected[1])):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registration changed during publication")
        if key not in evidence:
            evidence[key] = foundation.to_document(ref)
            conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                         (foundation.canonical_json(evidence), registration.run_id))
        return evidence[key]


def _verified(store, document, schema):
    """Verify bytes and their exact native publication reference without writing."""
    ref = foundation.from_document(ArtifactRef, document)
    if ref.schema_ref != schema:
        raise ValueError("unexpected native artifact schema")
    payload = store.read_verified(ref)
    expected = foundation.to_document(foundation.artifact_reference(payload, schema))
    if foundation.canonical_json(document) != foundation.canonical_json(expected):
        raise ValueError("native artifact reference differs from its content identity")
    return ref, payload


def _read(store, document, schema=SCHEMA):
    ref, payload = _verified(store, document, schema)
    value = json.loads(payload)
    if foundation.canonical_json(value).encode() != payload:
        raise ValueError("native evidence is not canonical JSON")
    return ref, value


def _reconcile(conn, store, registration, key, destination, ledger_path, ref_document, export_path=None):
    ref, outcome = _read(store, ref_document)
    if (not isinstance(outcome, dict) or set(outcome) != OUTCOME_FIELDS
            or outcome["schema_version"] != SCHEMA or outcome["variant_id"] != registration.variant_id
            or outcome["run_id"] != registration.run_id or outcome["ledger_destination"] != destination
            or not isinstance(outcome["failure_details"], dict)
            or (outcome["failure_code"] is None and outcome["failure_details"] != {})):
        raise _conflict(store, registration)
    row = outcome["ledger_row"]
    stage = "ran" if outcome["failure_code"] is None else STAGES[outcome["failure_code"]]
    expected_row = {"id": registration.document["execution_plan"]["experiment_id"],
                    "spec_hash": registration.variant_id, "date": row["date"], "stage": stage,
                    "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}
    if (row != expected_row or date.fromisoformat(row["date"]).isoformat() != row["date"]
            or type(outcome["attempted_variants"]) is not int or outcome["attempted_variants"] not in (0, 1)):
        raise _conflict(store, registration)
    report_ref = outcome["report_ref"]
    if (report_ref is not None) != (stage == "ran") or (stage == "ran" and outcome["attempted_variants"] != 1):
        raise _conflict(store, registration)
    if report_ref is not None:
        _, report = _verified(store, report_ref, "native_experiment_report.v1.0")
        if not report.decode("utf-8").strip():
            raise _conflict(store, registration)
    receipt = {"schema_version": "native_experiment_completion.v1.0", "outcome_ref": foundation.to_document(ref),
               "recording_mode": "smoke" if key == "native_smoke" else "recorded",
               "report_ref": report_ref, "ledger_destination": destination,
               "ledger_row_hash": None if key == "native_smoke" else foundation.content_hash(row)}
    evidence = _evidence(conn, registration)
    if key + "_receipt" in evidence:
        _, saved_receipt = _read(store, evidence[key + "_receipt"], receipt["schema_version"])
        if foundation.canonical_json(saved_receipt) != foundation.canonical_json(receipt):
            raise _conflict(store, registration)
    if export_path is not None and report_ref is None:
        raise fail("INVALID_EXPERIMENT_SPEC", "native outcome has no completed report")
    if report_ref is not None:
        _report_path(store, registration, key, report, report_ref)
    if key != "native_smoke":
        ledger_append([row], path=ledger_path, unique_by=("id", "spec_hash"))
    completion = _publish(store, receipt, receipt["schema_version"])
    existing = _reserve(conn, registration, key + "_receipt", completion, (key, foundation.to_document(ref)))
    if existing != foundation.to_document(completion):
        raise _conflict(store, registration)
    store.read_verified(completion)
    return {"outcome": outcome, "receipt": receipt, "receipt_ref": foundation.to_document(completion)}


def _typed(operation):
    """One public boundary for corruption, identity and interrupted storage."""
    @wraps(operation)
    def call(conn, store, registration, *args, **kwargs):
        try:
            if conn.in_transaction:
                raise fail("INVALID_EXPERIMENT_SPEC", "native outcomes require an idle catalog connection")
            return operation(conn, store, registration, *args, **kwargs)
        except (foundation.ArtifactError, LedgerError, ValueError, KeyError, TypeError):
            raise _conflict(store, registration) from None
        except (OSError, sqlite3.Error, RuntimeError):
            raise fail("RESOURCE_UNAVAILABLE", "native publication was interrupted; replay its intent") from None
        except OpsError as error:
            if error.code == "EXPERIMENT_IDENTITY_CONFLICT" and "refusal_ref" not in error.problem.details:
                raise _conflict(store, registration) from None
            raise
    return call


@_typed
def replay_native_outcome(conn, store, registration, *, no_ledger=False, ledger_path=None):
    """Verify and finish a saved intent, or return None without evaluating."""
    key, destination, ledger_path = _slot(no_ledger, ledger_path)
    verify_native_registration(conn, store, registration)
    evidence = _evidence(conn, registration)
    return None if key not in evidence else _reconcile(
        conn, store, registration, key, destination, ledger_path, evidence[key])


@_typed
def publish_native_outcome(conn, store, registration, *, report=None, problem=None,
                           attempted=False, no_ledger=False, ledger_path=None):
    """Globally bind one scientific result before replay-keyed recording."""
    key, destination, ledger_path = _slot(no_ledger, ledger_path)
    if (type(attempted) is not bool or (problem is None and (not attempted or not isinstance(report, str)
            or not report.strip())) or (problem is not None and (report is not None or problem.code not in STAGES))):
        raise fail("INVALID_EXPERIMENT_SPEC", "native outcome requires one success or supported typed refusal")
    verify_native_registration(conn, store, registration)
    stage = "ran" if problem is None else STAGES[problem.code]
    row = {"id": registration.document["execution_plan"]["experiment_id"],
           "spec_hash": registration.variant_id, "date": datetime.now(timezone.utc).date().isoformat(),
           "stage": stage, "oos_mean_mid": "", "sharpe_trade": "", "promoted": "False"}
    report_ref = foundation.artifact_reference(report.encode(), "native_experiment_report.v1.0") if report else None
    document = {"schema_version": SCHEMA, "run_id": registration.run_id,
                "variant_id": registration.variant_id,
                "attempted_variants": int(attempted), "failure_code": problem.code if problem else None,
                "failure_details": dict(problem.details) if problem else {},
                "report_ref": foundation.to_document(report_ref) if report_ref else None,
                "ledger_row": row, "ledger_destination": destination}
    evidence = _evidence(conn, registration)
    if key in evidence:
        existing = evidence[key]
    else:
        if report:
            _report_path(store, registration, key, report.encode(), document["report_ref"])
            store.publish_bytes(report.encode(), schema_ref=report_ref.schema_ref)
        existing = _reserve(conn, registration, key, _publish(store, document))
    _, saved = _read(store, existing)
    saved["ledger_row"]["date"] = row["date"]  # Compare result identity; reconciliation retains the first date.
    if foundation.canonical_json(saved) != foundation.canonical_json(document):
        raise _conflict(store, registration)
    return _reconcile(conn, store, registration, key, destination, ledger_path, existing)


@_typed
def export_native_report(conn, store, registration, *, no_ledger=False, ledger_path=None):
    """Materialize a verified completed report at its fixed store-local address."""
    key, destination, ledger_path = _slot(no_ledger, ledger_path)
    verify_native_registration(conn, store, registration)
    evidence = _evidence(conn, registration)
    if key not in evidence:
        raise fail("INVALID_EXPERIMENT_SPEC", "native outcome has no completed report")
    path = _report_path(store, registration, key)
    result = _reconcile(conn, store, registration, key, destination, ledger_path, evidence[key], export_path=path)
    ref = foundation.from_document(ArtifactRef, result["receipt"]["report_ref"])
    foundation.ensure_directory(path.parent)
    data = store.read_verified(ref)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".REPORT.", mode="wb") as staged:
        staged.write(data)
        staged.flush()
        os.fchmod(staged.fileno(), 0o444)
        os.fsync(staged.fileno())
        try:
            os.link(staged.name, path)
        except FileExistsError:
            if (path.is_symlink() or path.samefile(store.root / ref.storage_key)
                    or path.read_bytes() != data):
                raise _conflict(store, registration)
    foundation.fsync_directory(path.parent)
    return path
