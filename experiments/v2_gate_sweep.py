"""Seal and submit independent exploratory gate arms; collection is separate."""
import argparse
import json
import sqlite3
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path

from engine.v2.contracts import FAILURE_CODES, JobSpec, SubmitRequest
from engine.v2.contracts.data import SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import (
    ArtifactError,
    ArtifactStore,
    SystemClock,
    canonical_json,
    content_hash,
    from_document,
)
from engine.v2.ops.catalog import connect, transaction
from engine.v2.ops.checkpoints import register_artifact
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.experiments import experiment_spec_from_document
from engine.v2.ops.fingerprints import environment_identity, source_closure, worker_source_manifest
from engine.v2.ops.migrations import require_current
from engine.v2.ops.schema import MIGRATIONS, OWNER
from engine.v2.ops.stages import registry
from engine.v2.ops.submission import (
    NamespacePolicy,
    submit_graph,
    validate_receipt_read_policy,
)
from experiments import native_registration as native
from experiments.gate_report import validate_gate_spec
from experiments.native_gate_worker import read_native_staged_input


def _specs(document):
    try:
        if (set(document) != {"schema_version", "variants", "primary_comparison"}
                or document["schema_version"] != "native_gate_sweep_request.v1.0"
                or not isinstance(document["variants"], list) or len(document["variants"]) < 2):
            raise ValueError
        specs, common = [], []
        for item in document["variants"]:
            spec = experiment_spec_from_document(item)
            plan = validate_gate_spec(spec)
            spec = replace(spec, economic_params=plan.as_document()["economic_params"])
            specs.append(spec)
            common.append({**asdict(spec), "arms": (), "primary_arm_id": "",
                           "economic_params": {"strategy": spec.economic_params["strategy"]}})
        arms, pair = [spec.primary_arm_id for spec in specs], document["primary_comparison"]
        if (len(set(arms)) != len(arms) or not isinstance(pair, list) or len(pair) != 2
                or any(not isinstance(arm, str) or arm not in arms for arm in pair)
                or pair[0] == pair[1] or any(item != common[0] for item in common)):
            raise ValueError
        return sorted(specs, key=lambda spec: spec.primary_arm_id)
    except (ValueError, TypeError, KeyError, OpsError):
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep requires a closed, compatible multi-arm roster") from None


def _prepare(conn, repository, document, *, code_root, as_of_month, scope, event_ids, required=False):
    specs = _specs(document)
    if event_ids is not None and (not isinstance(event_ids, (list, tuple))
            or any(not isinstance(key, str) or not key for key in event_ids)):
        raise fail("INVALID_EXPERIMENT_SPEC", "sweep event IDs must be explicit strings")
    request = {"variants": [asdict(spec) for spec in specs], "primary_comparison": document["primary_comparison"],
               "scope": scope, "as_of_month": as_of_month,
               "event_ids": None if event_ids is None else sorted(set(event_ids))}
    row = None
    try:
        code_root = Path(code_root)
        require_current(conn, OWNER, MIGRATIONS)
        row = native.read_native_gate_sweep(conn, specs[0].experiment_id)
        if required and row is None:
            raise fail("INVALID_EXPERIMENT_SPEC", "sweep must be sealed before submission")
        stored = None if row is None else native.decode_native_gate_sweep(row)
        if stored is not None and canonical_json({**stored["request"], "event_ids": request["event_ids"]}) != canonical_json(request):
            raise ValueError
        if stored is None:
            try:
                snapshot = repository.resolve_pinned(scope)
            except DataError:
                raise fail("SNAPSHOT_UNRESOLVED", "sweep requires a committed snapshot") from None
        else:
            snapshot = from_document(SnapshotRef, stored["snapshot"])
        prepared = [native.prepare_native_registration(repository, spec, code_root=code_root, as_of_month=as_of_month,
                    scope=scope, event_ids=event_ids, snapshot=snapshot,
                    expected=None if stored is None else native.NativeRegistration(
                        saved["run_id"], canonical_json(saved["binding"]).encode()))
                    for spec, saved in zip(specs, [None] * len(specs) if stored is None else
                                           stored["registrations"], strict=True)]
        registrations = [item[0] for item in prepared]
        first = registrations[0].document
        if any(any(reg.document[name] != first[name] for name in
                   ("snapshot", "event_ids", "holdouts", "source_closure", "environment"))
               for reg in registrations):
            raise ValueError
        try:
            sources = source_closure(code_root, ["experiments/v2_gate_sweep.py"])
            implementation = content_hash(worker_source_manifest(code_root))
        except ArtifactError:
            raise fail("INVALID_EXPERIMENT_SPEC", "sweep source cannot be verified") from None
        sealed = {"schema_version": "native_gate_sweep.v1.0", "request": request, "snapshot": first["snapshot"],
                  "registrations": [{"run_id": reg.run_id, "binding": reg.document} for reg in registrations],
                  "coordinator_source": sources, "implementation_ref": implementation}
        if stored is not None and canonical_json(stored) != canonical_json(sealed):
            raise ValueError
        return sealed, specs, registrations
    except OpsError as error:
        if error.code == "INVALID_EXPERIMENT_SPEC" and row is not None:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sealed sweep source cannot be verified") from None
        raise
    except ArtifactError as error:
        raise fail(error.code if error.code in FAILURE_CODES else "INTEGRITY_FAILED",
                   "sweep artifact cannot be verified", details={"data_failure_code": error.code}) from None
    except (OSError, sqlite3.Error) as error:
        raise fail("INPUT_CHANGED", "sweep inputs are unavailable",
                   details={"data_failure_code": type(error).__name__}) from None
    except (ValueError, TypeError, KeyError, SyntaxError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT" if row is not None else "INVALID_EXPERIMENT_SPEC",
                   "sweep binding cannot be verified") from None


def _typed_sweep_io(action):
    @wraps(action)
    def call(*args, **kwargs):
        try:
            return action(*args, **kwargs)
        except (OSError, sqlite3.Error) as error:
            raise fail("INPUT_CHANGED", "sweep inputs are unavailable",
                       details={"data_failure_code": type(error).__name__}) from None
    return call


@_typed_sweep_io
def register_sweep(conn, store, repository, document, *, code_root, as_of_month, scope="shadow", event_ids=None):
    """Seal once before reserving arms; exact replay resumes partial reservation."""
    sealed, specs, registrations = _prepare(conn, repository, document, code_root=code_root,
                                           as_of_month=as_of_month, scope=scope, event_ids=event_ids)
    payload = canonical_json(sealed)
    native.decode_native_gate_sweep(payload, failure_code="INVALID_EXPERIMENT_SPEC")
    with transaction(conn):
        for spec, registration in zip(specs, registrations, strict=True):
            if (not native.native_registration_exists(conn, store, registration) and conn.execute(
                    "SELECT 1 FROM hypotheses WHERE spec_hash=?", (spec.spec_hash,)).fetchone()):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native specification is already reserved")
        row = native.read_native_gate_sweep(conn, specs[0].experiment_id)
        if row is not None and row != payload:
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep experiment already has another roster")
        if row is None:
            conn.execute("INSERT INTO native_gate_sweeps VALUES (?,?)", (specs[0].experiment_id, payload))
    for spec, registration in zip(specs, registrations, strict=True):
        native.reserve_native_registration(conn, store, registration, spec)
    return content_hash(sealed)


@_typed_sweep_io
def submit_sweep(conn, store, repository, document, *, code_root, as_of_month, scope="shadow", event_ids=None,
                 max_evidence_bytes=None, receipt_decode_failure_code=None, max_failure_bytes=None):
    """Verify all sealed arms before atomically submitting unrecorded child jobs."""
    validate_receipt_read_policy(receipt_decode_failure_code, max_failure_bytes)
    sealed, specs, registrations = _prepare(conn, repository, document, code_root=code_root,
                                           as_of_month=as_of_month, scope=scope, event_ids=event_ids, required=True)
    for registration in registrations:
        native.verify_native_registration(conn, store, registration, max_evidence_bytes=max_evidence_bytes)
    requests, clock = [], SystemClock()
    for spec, registration in zip(specs, registrations, strict=True):
        bindings = {}
        for name, payload in (("spec.json", canonical_json(asdict(spec)).encode()),
                              ("native_registration.json", registration.binding_json)):
            ref = store.publish_bytes(payload, schema_ref="native_staged_input.v1.0")
            with transaction(conn):
                register_artifact(conn, ref, None, clock)
            bindings[name] = ref.artifact_id
        parameters = {"run_id": registration.run_id, "variant_id": registration.variant_id,
                      "expected_ids": [registration.variant_id], "no_ledger": True, "purpose": "sweep",
                      "catalog_path": conn.execute("PRAGMA database_list").fetchone()["file"],
                      "objects_root": str(store.root), "input_bindings": bindings}
        job = JobSpec(kind="native_gate_evaluation", spec_hash=spec.spec_hash,
            implementation_ref=sealed["implementation_ref"], environment_ref=content_hash(environment_identity(1)),
            parameters=parameters, input_refs=tuple(bindings.values()), output_namespace="smoke",
            resource_class="native_experiment", retry_policy_ref="once", checkpoint_contract_ref="native_gate_result.v1.0")
        requests.append(SubmitRequest(namespace="smoke", idempotency_key="gate-sweep-" +
            content_hash({"roster": content_hash(sealed), "variant": registration.variant_id}).removeprefix("sha256:"),
            principal="operator", job=job))
    try:
        return submit_graph(conn, registry(), NamespacePolicy({"operator": frozenset({"smoke"})}), requests,
                            clock=clock, receipt_decode_failure_code=receipt_decode_failure_code,
                            max_failure_bytes=max_failure_bytes)
    except OpsError as error:
        if error.code == "IDEMPOTENCY_CONFLICT":
            raise fail("EXPERIMENT_IDENTITY_CONFLICT", "sweep submission differs from its first request") from None
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "submit"))
    for name in ("roster", "catalog", "store-root", "as-of-month", "code-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--scope", default="shadow")
    parser.add_argument("--event-id", action="append")
    parser.add_argument("--purpose", default="sweep")
    parser.add_argument("--no-ledger", action="store_true")
    args, conn = parser.parse_args(argv), None
    try:
        if args.purpose != "sweep":
            raise fail("HOLDOUT_ACCESS_DENIED", "gate sweeps have no final-read authority")
        if not args.no_ledger:
            raise fail("INVALID_EXPERIMENT_SPEC", "staged sweep submission requires --no-ledger")
        document = json.loads(read_native_staged_input(Path(args.roster)))
        _specs(document)
        conn = connect(args.catalog, must_exist=True)
        store = ArtifactStore(args.store_root)
        action = register_sweep if args.action == "register" else submit_sweep
        result = action(conn, store, Repository(conn, store), document,
            code_root=Path(args.code_root), as_of_month=args.as_of_month,
            scope=args.scope, event_ids=args.event_id)
        print(json.dumps({"roster": result} if args.action == "register" else
                         {"state": "submitted_uncollected", "job_ids": [job.job_id for job in result]}))
        return 0
    except (OpsError, DataError, ArtifactError, ValueError, TypeError, KeyError, RecursionError, OSError, sqlite3.Error) as error:
        print(json.dumps({"refused": error.code if isinstance(error, (OpsError, DataError, ArtifactError)) else
                          "INPUT_CHANGED" if isinstance(error, (OSError, sqlite3.Error)) else "INVALID_EXPERIMENT_SPEC"}))
        return 2
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
