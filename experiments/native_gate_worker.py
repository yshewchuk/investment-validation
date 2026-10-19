"""Admitted smoke pricing from staged registration; no scientific recording."""
import errno
import json
import os
import sqlite3
import stat
from collections.abc import Mapping
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from tempfile import mkdtemp

from engine.v2.contracts import DATA_FAILURE_CODES, FAILURE_CODES
from engine.v2.contracts.data import SnapshotRef
from engine.v2.data.errors import DataError
from engine.v2.data.repository import Repository
from engine.v2.foundation import ArtifactError, ArtifactStore, canonical_json, from_document
from engine.v2.ops.errors import OpsError, fail
from engine.v2.ops.executor import THREAD_VARIABLES
from engine.v2.ops.experiments import (
    NATIVE_GATE_ENTRYPOINT,
    NATIVE_GATE_WORKER,
    experiment_spec_from_document,
    resolve_experiment_plan,
)
from engine.v2.ops.fingerprints import environment_identity, source_closure
from engine.v2.ops.stages import NativeGateParameters
from experiments.native_registration import NativeRegistration, verify_native_registration
from experiments.v2_gate_variant import price_variant

RUNNER = NATIVE_GATE_WORKER
BINDINGS = {"spec.json", "native_registration.json"}
MAX_STAGED_BYTES = 16 * 1024 * 1024


def native_gate_data_failure(error):
    """Translate data-only refusals without changing the ops registry's policy."""
    code = error.code
    if code not in FAILURE_CODES:
        code = {("integrity", False): "INTEGRITY_FAILED",
                ("validation", False): "VALIDATION_FAILED",
                ("dependency", False): "INPUT_CHANGED",
                ("dependency", True): "SNAPSHOT_NOT_READY",
                ("resource", False): "RESOURCE_LIMIT_EXCEEDED",
                ("resource", True): "RESOURCE_UNAVAILABLE"}[DATA_FAILURE_CODES[code]]
    return fail(code, "native gate data refused", details={"data_failure_code": error.code})


def read_native_staged_input(path):
    """Bound allocation and validate the descriptor actually supplying bytes."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_STAGED_BYTES:
                raise fail("INVALID_EXPERIMENT_SPEC", "native staging requires bounded regular files")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                data = stream.read(MAX_STAGED_BYTES + 1)
            if len(data) > MAX_STAGED_BYTES:
                raise fail("INVALID_EXPERIMENT_SPEC", "native staged document exceeds its bound")
            return data
        finally:
            os.close(fd)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise fail("INVALID_EXPERIMENT_SPEC", "native staging must not follow symlinks") from None
        raise


def write_native_gate_receipt(path, receipt):
    """Install complete bytes create-only; clean only private temporary names."""
    directory = Path(mkdtemp(prefix=".native-gate-", dir=path.parent))
    private = directory / "receipt.json"
    try:
        fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", closefd=False) as stream:
                stream.write(canonical_json(receipt))
        finally:
            os.close(fd)
        os.link(private, path)
        private.unlink()
    finally:
        for cleanup in (private.unlink, directory.rmdir):
            try:
                cleanup()
            except OSError:
                pass


def admit_native_gate_inputs(parameters, root, conn, store, *, runner=RUNNER):
    """Validate the staged request against existing immutable native admission."""
    if (parameters.get("no_ledger") is not True
            or set(parameters.get("input_bindings") or {}) != BINDINGS):
        raise fail("INVALID_EXPERIMENT_SPEC", "native gate requires bound smoke inputs")
    try:
        spec = experiment_spec_from_document(json.loads(read_native_staged_input(root / "spec.json")))
    except OpsError as error:
        if error.code != "INVALID_REQUEST":
            raise
        raise fail("INVALID_EXPERIMENT_SPEC", "native staged specification is malformed") from None
    plan = resolve_experiment_plan(spec)
    if plan.runner != runner:
        raise fail("INVALID_EXPERIMENT_SPEC", "native gate runner is not allowlisted")
    if not isinstance(plan.economic_params.get("strategy"), str):
        raise fail("INVALID_EXPERIMENT_SPEC", "native worker requires a registered strategy")
    registration = NativeRegistration(parameters["run_id"],
                                      read_native_staged_input(root / "native_registration.json"))
    verify_native_registration(conn, store, registration)
    binding = registration.document
    try:
        entries = [runner] + ([NATIVE_GATE_ENTRYPOINT] if runner == RUNNER else [])
        current_source = source_closure(Path.cwd(), entries)
    except (OpsError, OSError, SyntaxError, ValueError, RecursionError):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native registered source cannot be verified") from None
    if (registration.variant_id != parameters["variant_id"]
            or list(parameters["expected_ids"]) != [registration.variant_id]
            or binding["spec_hash"] != spec.spec_hash
            or canonical_json(binding["execution_plan"]) != canonical_json(plan.as_document())
            or binding["source_closure"] != current_source
            or binding["environment"] != environment_identity(1)
            or any(os.environ.get(name) != "1" for name in THREAD_VARIABLES)):
        raise fail("EXPERIMENT_IDENTITY_CONFLICT", "native staged execution differs from registration")
    return spec, registration


def run(parameters, root):
    """Read only the registered snapshot and return a complete staged receipt."""
    if not isinstance(parameters, Mapping):
        raise fail("INVALID_EXPERIMENT_SPEC", "native worker arguments must be a mapping")
    purpose = parameters.get("purpose", "sweep")
    if not isinstance(purpose, str):
        raise fail("INVALID_EXPERIMENT_SPEC", "native pricing purpose must be a string")
    if purpose != "sweep":
        raise fail("HOLDOUT_ACCESS_DENIED", "native pricing permits sweep reads only")
    if parameters.get("no_ledger") is not True:
        raise fail("INVALID_EXPERIMENT_SPEC", "native pricing is smoke-only")
    pricing = False
    try:
        from_document(NativeGateParameters, dict(parameters))
        conn = sqlite3.connect(Path(parameters["catalog_path"]).resolve().as_uri() + "?mode=ro",
                               uri=True, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            store = ArtifactStore(parameters["objects_root"])
            spec, registration = admit_native_gate_inputs(parameters, root, conn, store)
            binding = registration.document
            pricing = True
            result = price_variant(
                Repository(conn, store, pin_snapshots=True), from_document(SnapshotRef, binding["snapshot"]), spec,
                strategy=spec.economic_params["strategy"], as_of_month=binding["holdouts"]["holdout_as_of_month"],
                event_ids=binding["event_ids"])
            inputs = asdict(result.stage_inputs)
            if (sorted(inputs["event_ids"]) != binding["event_ids"]
                    or any(inputs[name] != value for name, value in binding["holdouts"].items())):
                raise fail("EXPERIMENT_IDENTITY_CONFLICT", "consumed population differs from registration")
            cashflows = result.replay.trades[
                ["event_id", "fill_alpha", "entry_cost", "exit_value", "pnl"]].to_dict("records")
            if any(not isfinite(row[name]) for row in cashflows
                   for name in ("fill_alpha", "entry_cost", "exit_value", "pnl")):
                raise fail("EXPERIMENT_VARIANT_FAILED", "native pricing produced nonfinite cash flows")
            receipt = {"schema_version": "native_gate_consumption.v1.0",
                       "variant_id": registration.variant_id, "run_id": registration.run_id,
                       "recording_mode": "no-ledger", "execution_plan": result.execution_plan.as_document(),
                       "stage_inputs": inputs, "mark_based_cashflows": cashflows}
        finally:
            conn.close()
        pricing = False
        write_native_gate_receipt(root / "native_gate_consumption.json", receipt)
    except DataError as error:
        raise native_gate_data_failure(error) from None
    except (ArtifactError, ValueError, TypeError, KeyError, RecursionError, OSError, sqlite3.Error) as error:
        if pricing:
            raise fail("EXPERIMENT_VARIANT_FAILED", "native pinned pricing failed",
                       details={"data_failure_code": getattr(error, "code", type(error).__name__)}) from None
        if isinstance(error, (OSError, sqlite3.Error)):
            raise fail("INPUT_CHANGED", "native staged I/O is unavailable") from None
        raise fail("INVALID_EXPERIMENT_SPEC", "native staged input is malformed") from None
    return {"outputs": [{"name": "native_gate_consumption", "path": "native_gate_consumption.json",
                         "schema": "native_gate_consumption.v1.0"}],
            "completed_ids": [registration.variant_id], "no_work": False}
