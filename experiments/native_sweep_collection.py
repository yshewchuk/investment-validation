"""Coordinator-only collection of sealed gate jobs and immutable run indexes."""
import errno
import json
import os
import sqlite3
import stat
from functools import wraps
from pathlib import Path

from engine.v2.contracts import FAILURE_CODES, ArtifactRef, Problem
from engine.v2.foundation import (
    ArtifactError,
    artifact_reference,
    canonical_json,
    content_hash,
    from_document,
    to_document,
)
from engine.v2.ops.catalog import decode_bounded_text, sql_text_budget, transaction
from engine.v2.ops.checkpoints import registered_artifact
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from engine.v2.ops.fingerprints import source_closure
from engine.v2.ops.native_registration import (
    MAX_NATIVE_EVIDENCE_BYTES,
    native_evidence_json,
    read_native_evidence,
)
from engine.v2.ops.submission import MAX_JOB_FAILURE_BYTES, get_job
from experiments import native_outcomes as outcomes
from experiments import native_registration as native
from experiments.gate_report import render_gate_report
from experiments.v2_gate_sweep import submit_sweep

TERMINAL = frozenset({"succeeded", "failed", "cancelled", "blocked"})
MAX_GATE_RESULT_BYTES = 16 * 1024 * 1024
MAX_FAILURE_BYTES = MAX_JOB_FAILURE_BYTES
_COLLECTOR_ROOT = Path(__file__).resolve().parents[1]
_COLLECTOR_ENTRIES = ("experiments/native_sweep_collection.py", "experiments/v2_gate_collect.py")
try:
    _COLLECTOR_SOURCES = source_closure(_COLLECTOR_ROOT, _COLLECTOR_ENTRIES)
    _COLLECTOR_SOURCE_ERROR = None
except (OpsError, ArtifactError, OSError, SyntaxError, ValueError, RecursionError) as _error:
    _COLLECTOR_SOURCES, _COLLECTOR_SOURCE_ERROR = None, type(_error).__name__


def _implementation(code_root):
    """Verify an import-time on-disk snapshot, not a Python bytecode attestation."""
    try:
        root = Path(code_root)
        metadata = root.stat()
    except (TypeError, ValueError, FileNotFoundError, NotADirectoryError):
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep code root must name an existing directory") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep code root must name an existing directory")
    try:
        if _COLLECTOR_SOURCE_ERROR is not None:
            raise ValueError
        current = source_closure(_COLLECTOR_ROOT, _COLLECTOR_ENTRIES)
        if current != _COLLECTOR_SOURCES:
            raise ValueError
        caller = source_closure(root, _COLLECTOR_ENTRIES)
        if any(caller.get(entry) != current[entry] for entry in _COLLECTOR_ENTRIES):
            raise ValueError
        return current
    except OSError as error:
        if error.errno in {errno.EIO, errno.EMFILE, errno.ENFILE}:
            raise fail("RESOURCE_UNAVAILABLE", "collector source verification was interrupted") from None
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "installed collector source cannot be verified") from None
    except (OpsError, ArtifactError, SyntaxError, ValueError, TypeError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "installed collector source cannot be verified") from None


def _typed(action):
    """Unlike native outcomes, unexpected structural/programming errors propagate."""
    @wraps(action)
    def call(*args, **kwargs):
        try:
            return action(*args, **kwargs)
        except sqlite3.IntegrityError:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep collection catalog constraint failed") from None
        except (OSError, sqlite3.Error):
            raise fail("RESOURCE_UNAVAILABLE", "sweep collection publication was interrupted") from None
    return call


def _verified_bytes(store, ref):
    try:
        return store.read_verified(ref)
    except (ArtifactError, UnicodeEncodeError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep stored artifact cannot be verified") from None


def _stored_evidence(conn, run_id):
    saved = read_native_evidence(conn, run_id)
    if saved is None:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep registration evidence is missing")
    return saved[1]


def _mode(no_ledger, ledger_path):
    if type(no_ledger) is not bool or no_ledger == (ledger_path is not None):
        raise fail("INVALID_EXPERIMENT_SPEC", "choose smoke collection or one ledger destination")
    if no_ledger:
        return "smoke", None
    try:
        raw = os.fspath(ledger_path)
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise ValueError
        path = Path(raw)
    except (ValueError, TypeError):
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep ledger destination must be a valid path") from None
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        pass  # The ledger sink creates absent destinations and parent directories.
    except OSError as error:
        if error.errno in {errno.ENOTDIR, errno.ELOOP, errno.ENAMETOOLONG}:
            raise fail("INVALID_EXPERIMENT_SPEC", "sweep ledger destination must be a valid path") from None
        raise
    else:
        if not stat.S_ISREG(metadata.st_mode):
            raise fail("INVALID_EXPERIMENT_SPEC", "sweep ledger destination must name a regular file")
    return "recorded", path.resolve()


def _publish(store, document):
    return store.publish_bytes(native_evidence_json(document).encode(), schema_ref=document["schema_version"])


def _bind(conn, store, anchor, key, document):
    """Compare immutable intent/index bytes; preserve all other arm evidence."""
    payload = native_evidence_json(document).encode()
    expected = artifact_reference(payload, document["schema_version"])
    with transaction(conn):
        evidence = _stored_evidence(conn, anchor.run_id)
        existing = evidence.get(key)
    if existing is not None:
        try:
            ref = from_document(ArtifactRef, existing)
        except (ValueError, RecursionError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep stored artifact reference cannot be decoded") from None
        if ref != expected or _verified_bytes(store, ref) != payload:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep collection roster/source/mode/ledger intent or "
                       "terminal index differs; restore original context or seal a new experiment")
        return to_document(ref)
    ref = _publish(store, document)
    native.verify_native_registration(conn, store, anchor, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
    with transaction(conn):
        evidence = _stored_evidence(conn, anchor.run_id)
        if evidence.get("variant_id") != anchor.variant_id or (key in evidence and evidence[key] != to_document(ref)):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep collection identity changed during publication")
        evidence[key] = to_document(ref)
        conn.execute("UPDATE experiment_runs SET evidence_json=? WHERE run_id=?",
                     (native_evidence_json(evidence), anchor.run_id))
    return to_document(ref)


def _check_policy(problem):
    if (problem.code not in FAILURE_CODES
            or (problem.category, problem.retryable) != FAILURE_CODES[problem.code]):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep child refusal differs from its registered policy")


def _terminal(conn, store, job, registration):
    """Never use JobReceipt.output_refs: it unions outputs from old attempts."""
    encoding, limit = sql_text_budget(conn, MAX_FAILURE_BYTES)
    attempt = conn.execute(
        "SELECT attempt_id, state, ended_at, fence, failure_json IS NOT NULL AS has_failure, "
        "CASE WHEN typeof(failure_json)='text' AND length(CAST(failure_json AS BLOB))<=? "
        "THEN CAST(failure_json AS BLOB) END AS failure_json FROM attempts "
        "WHERE job_id=? AND attempt_number=?", (limit, job.job_id, job.attempt_count)).fetchone()
    if job.attempt_count and (attempt is None or attempt["state"] != job.state or attempt["ended_at"] is None
            or attempt["fence"] > job.fence or (job.state == "succeeded" and attempt["fence"] != job.fence)):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep terminal attempt differs from its job")
    outputs = [] if attempt is None else conn.execute(
        "SELECT name, artifact_id FROM attempt_outputs WHERE attempt_id=?", (attempt["attempt_id"],)).fetchall()
    result, problem = None, job.failure
    if attempt is not None and attempt["has_failure"]:
        payload = attempt["failure_json"]
        try:
            stored = from_document(Problem, json.loads(decode_bounded_text(payload, encoding, MAX_FAILURE_BYTES)))
        except (ValueError, RecursionError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep attempt failure cannot be decoded") from None
        problem = job.failure if job.failure is not None else stored
        _check_policy(stored)
    if job.failure is not None:
        _check_policy(job.failure)
    if problem is None and job.state == "cancelled":
        problem = fail("CANCELLED", "sweep child was cancelled").problem
    worker_problem = problem
    if job.state == "succeeded" and outputs:
        if len(outputs) != 1 or outputs[0]["name"] != "native_gate_result":
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep terminal attempt has unexpected outputs")
        try:
            ref = registered_artifact(conn, outputs[0]["artifact_id"], max_metadata_bytes=MAX_NATIVE_EVIDENCE_BYTES)
        except (ValueError, RecursionError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep result reference cannot be decoded") from None
        if ref.artifact_id != outputs[0]["artifact_id"]:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep result reference identity differs from its output")
        if ref.byte_size > MAX_GATE_RESULT_BYTES:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep result exceeds its byte bound")
        payload = _verified_bytes(store, ref)
        try:
            result = json.loads(payload)
            canonical = canonical_json(result).encode()
        except (ValueError, RecursionError):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep result cannot be decoded") from None
        if (not isinstance(result, dict) or ref.schema_ref != "native_gate_result.v1.0" or canonical != payload
                or ref != artifact_reference(payload, ref.schema_ref)
                or result.get("schema_version") != ref.schema_ref or type(result.get("attempted_variants")) is not int
                or result.get("attempted_variants") != 1 or result.get("variant_id") != registration.variant_id
                or result.get("run_id") != registration.run_id
                or canonical_json(result.get("provenance")) != canonical_json(registration.document)):
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep child result differs from its sealed arm")
    elif outputs:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "failed sweep attempt has successful output")
    if result is None:
        if problem is None or problem.code == "WORKER_FAILED":
            problem = fail("EXPERIMENT_VARIANT_FAILED", "sweep child produced no scientific result").problem
    elif problem is not None:
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "successful sweep child has a refusal")
    return attempt, result, problem, worker_problem


@_typed
def collect_sweep(conn, store, repository, document, *, code_root, as_of_month,
                  scope="shadow", event_ids=None, no_ledger=False, ledger_path=None):
    """Submit/revalidate exact jobs, collect terminal arms, and resume publication.

    Pending arms remain pending. Once all jobs terminate, the immutable index
    stays incomplete if any arm failed/refused. Smoke creates sweep artifacts
    only, never native outcome slots, CSV rows or native report exports.
    """
    mode, ledger_path = _mode(no_ledger, ledger_path)
    if conn.in_transaction:
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep collection requires an idle catalog connection")
    implementation = _implementation(code_root)
    jobs = submit_sweep(conn, store, repository, document, code_root=code_root,
                       as_of_month=as_of_month, scope=scope, event_ids=event_ids,
                       max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES,
                       receipt_decode_failure_code="EXPERIMENT_IDENTITY_CONFLICT",
                       max_failure_bytes=MAX_JOB_FAILURE_BYTES)
    sealed = native.decode_native_gate_sweep(native.read_native_gate_sweep(
        conn, document["variants"][0]["experiment_id"]))
    registrations = [native.NativeRegistration(item["run_id"], canonical_json(item["binding"]).encode())
                     for item in sealed["registrations"]]
    intent = {"schema_version": "native_gate_collection.v1.0", "roster_id": content_hash(sealed),
              "recording_mode": mode, "ledger_destination": None if no_ledger else content_hash(str(ledger_path)),
              "collector_source": implementation}
    key = "native_sweep_" + mode
    _bind(conn, store, registrations[0], key + "_intent", intent)
    arms, all_terminal = [], True
    for job, registration, spec_document in zip(jobs, registrations, sealed["request"]["variants"], strict=True):
        job = get_job(conn, job.job_id, receipt_decode_failure_code="EXPERIMENT_IDENTITY_CONFLICT",
                      max_failure_bytes=MAX_JOB_FAILURE_BYTES)
        arm = {"variant_id": registration.variant_id, "arm_id": spec_document["primary_arm_id"],
               "job_id": job.job_id, "attempted_variants": int(job.attempt_count > 0),
               "state": job.state, "attempt_id": None, "problem": None, "worker_problem": None, "report_ref": None,
               "outcome_ref": None, "completion_ref": None}
        if job.state not in TERMINAL:
            all_terminal = False
            arms.append(arm)
            continue
        attempt, result, problem, worker_problem = _terminal(conn, store, job, registration)
        arm["attempt_id"] = None if attempt is None else attempt["attempt_id"]
        arm["problem"] = None if problem is None else to_document(problem)
        arm["worker_problem"] = None if worker_problem is None else to_document(worker_problem)
        if problem is not None:
            arm["state"] = "failed" if problem.code == "EXPERIMENT_VARIANT_FAILED" else "refused"
        report = None if result is None else render_gate_report(
            registration, result, spec=experiment_spec_from_document(spec_document), no_ledger=no_ledger)
        if no_ledger:
            if report is not None:
                arm["report_ref"] = to_document(store.publish_bytes(
                    report.encode(), schema_ref="native_experiment_report.v1.0"))
        elif result is not None or problem.code in outcomes.STAGES:
            saved = outcomes.publish_native_outcome(conn, store, registration, report=report, problem=problem,
                attempted=bool(arm["attempted_variants"]), ledger_path=ledger_path, max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
            arm["report_ref"] = saved["outcome"]["report_ref"]
            arm["outcome_ref"] = saved["receipt"]["outcome_ref"]
            arm["completion_ref"] = saved["receipt_ref"]
            if report is not None:
                outcomes.export_native_report(conn, store, registration, ledger_path=ledger_path,
                                              max_evidence_bytes=MAX_NATIVE_EVIDENCE_BYTES)
        arms.append(arm)
    index = {"schema_version": "native_gate_sweep_index.v1.0", "run_id": content_hash(intent),
             "roster_id": intent["roster_id"], "recording_mode": mode,
             "primary_comparison": sealed["request"]["primary_comparison"], "planned_variants": len(arms),
             "attempted_variants": sum(arm["attempted_variants"] for arm in arms),
             "complete": all(arm["state"] == "succeeded" for arm in arms), "variants": arms}
    return {"index": index, "index_ref": _bind(conn, store, registrations[0], key + "_index", index)
            if all_terminal else None}
