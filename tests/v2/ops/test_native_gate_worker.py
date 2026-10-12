"""Real staged worker admission over committed synthetic option-chain snapshots."""
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from engine.v2.contracts import JobSpec, SubmitRequest
from engine.v2.foundation import ArtifactStore, SystemClock, content_hash
from engine.v2.ops import stages, worker
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.executor import THREAD_VARIABLES
from engine.v2.ops.experiments import experiment_spec_from_document, experiments_ledger_path
from engine.v2.ops.fingerprints import environment_identity, snapshot_code, worker_source_manifest
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from experiments.native_gate_worker import RUNNER
from experiments.native_registration import register_native
from tests.ops_support import TEST_POLICY, run_until
from tests.v2.research.test_gate_variant import _document
from tests.v2.research.test_gate_variant import source as source
from tests.v2.research.test_native_registration import _catalog

REPO = Path(__file__).resolve().parents[3]


def _registered(source, alpha=0.0, *, code_root=REPO, **changes):
    """Bind the worker and strategy without conflating strategy with arm identity."""
    arm = "candidate" if alpha == 0.0 else "control"
    document = {**_document(alpha), "experiment_id": "synthetic-worker",
                "runner": RUNNER, "primary_arm_id": arm, "arms": [arm],
                "economic_params": {"fill": alpha, "strategy": "STR-THRU"}, **changes}
    spec = experiment_spec_from_document(document)
    registration = register_native(source[0], source[2], source[3], spec,
                                   code_root=code_root, as_of_month="2025-01")
    parameters = {"run_id": registration.run_id, "variant_id": registration.variant_id,
                  "expected_ids": [registration.variant_id], "no_ledger": True,
                  "catalog_path": source[0].execute("PRAGMA database_list").fetchone()["file"],
                  "objects_root": str(source[2].root),
                  "input_bindings": {"spec.json": "unused", "native_registration.json": "unused"}}
    return document, registration, parameters


def _stage(path, document, registration):
    """Materialize the same two immutable files copied by the real executor."""
    path.mkdir()
    (path / "spec.json").write_text(json.dumps(document))
    (path / "native_registration.json").write_bytes(registration.binding_json)
    return path


@pytest.fixture
def single_thread(monkeypatch):
    """Match the declared executor environment for direct boundary tests."""
    monkeypatch.chdir(REPO)
    for name in THREAD_VARIABLES:
        monkeypatch.setenv(name, "1")


@pytest.fixture
def forbidden_pricing(monkeypatch):
    """Fail if an admission refusal reaches the actual pricing boundary."""
    def forbidden(*args, **kwargs):
        pytest.fail("pricing ran before admission refused")
    monkeypatch.setattr("experiments.native_gate_worker.price_variant", forbidden)


def _supervise(tmp_path, source, document, registration, parameters, *, key="smoke",
               staged_bytes=None, expected_details=None, shared_root=None, before_start=None,
               code_source=REPO):
    """Submit through the production registry, launcher, child and checkpoint path."""
    root = tmp_path / (shared_root or key)
    root.mkdir(exist_ok=shared_root is not None)
    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    store = ArtifactStore(root)
    try:
        bindings = {}
        for name, data in (("spec.json", json.dumps(document).encode()),
                           ("native_registration.json", registration.binding_json)):
            ref = store.publish_bytes((staged_bytes or {}).get(name, data),
                                      schema_ref="native_staged_input.v1.0")
            with transaction(conn):
                register_artifact(conn, ref, None, clock)
            bindings[name] = ref.artifact_id
        job = JobSpec(kind="native_gate_pricing", spec_hash=content_hash(document),
                      implementation_ref=content_hash(worker_source_manifest(code_source)),
                      environment_ref=content_hash(environment_identity(1)),
                      parameters={**parameters, "input_bindings": bindings},
                      input_refs=tuple(bindings.values()), output_namespace="smoke",
                      resource_class="native_experiment", retry_policy_ref="once",
                      checkpoint_contract_ref="native_gate_consumption.v1.0")
        receipt = submit(conn, stages.registry(), NamespacePolicy({"operator": frozenset({"smoke"})}),
                         SubmitRequest(namespace="smoke", idempotency_key=key, principal="operator", job=job),
                         clock=clock)
        service = Service(conn, root, stages.registry(), TEST_POLICY, clock=clock, code_source=code_source)
        if before_start is not None:
            before_start(conn, store, job, clock)
        checkpoint_count = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
        try:
            service.start()
            state = run_until(service, conn, receipt.job_id, timeout=90)
        finally:
            service.close()
        attempt = conn.execute("SELECT * FROM attempts WHERE job_id=?", (receipt.job_id,)).fetchone()
        outputs = conn.execute("SELECT * FROM attempt_outputs WHERE attempt_id=?",
                               (attempt["attempt_id"],)).fetchall()
        assert conn.execute("SELECT COUNT(*) FROM experiment_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] == checkpoint_count
        if state == "succeeded":
            assert len(outputs) == 1 and outputs[0]["name"] == "native_gate_consumption"
            ref = artifact(conn, store, outputs[0]["artifact_id"])
            assert ref.schema_ref == "native_gate_consumption.v1.0"
            return json.loads(store.read_verified(ref))
        assert outputs == []
        failure = json.loads(attempt["failure_json"])
        if expected_details is not None:
            assert "data_failure_code" not in json.dumps(failure)
            diagnostic = artifact(conn, store, failure["diagnostic_ref"])
            assert json.loads(store.read_verified(diagnostic)) == expected_details
        return failure
    finally:
        conn.close()


def test_supervised_arms_consume_independent_fill_and_never_record(source, tmp_path):
    """Two real child processes price independent registered economics, not shared trades."""
    from tests.test_v2_research_replay import _chain_rows, _commit, _event_rows

    other = "OTHER_2024-05-02"
    snapshot = _commit(
        source[0], source[1], source[2],
        chain_rows=[{**row, "ticker": "OTHER"} for row in _chain_rows(exit_call=(4.0, 4.4))]
                   + _chain_rows(),
        event_rows=[{**_event_rows()[0], "event_id": other, "ticker": "OTHER"}] + _event_rows(),
        receipt_id="two-event-provenance", expected_head=source[4].snapshot_id, generation=1)
    source = (*source[:4], snapshot)
    checkout = tmp_path / "checkout"
    snapshot_code(REPO, checkout, worker_source_manifest(REPO))
    requests = [_registered(source, alpha, code_root=checkout) for alpha in (0.0, 1.0)]
    before = _catalog(source[0])
    ledger = experiments_ledger_path(checkout)
    assert ledger == checkout / "experiments" / "LEDGER.csv"
    ledger.write_bytes(b"sentinel legacy ledger\n")
    results = [_supervise(tmp_path, source, *request, key=f"arm-{i}", code_source=checkout)
               for i, request in enumerate(requests)]
    for result, request, alpha, cost, exits, profits in zip(
            results, requests, (0.0, 1.0), (3.8, 3.0), ((6.0, 5.0), (6.8, 5.8)),
            ((2.2, 1.2), (3.8, 2.8))):
        _, registration, _ = request
        expected_cashflows = [
            {"event_id": event, "fill_alpha": alpha, "entry_cost": pytest.approx(cost),
             "exit_value": pytest.approx(exit_value), "pnl": pytest.approx(profit)}
            for event, exit_value, profit in zip((other, "TEST_2024-05-02"), exits, profits)]
        assert {**result, "mark_based_cashflows": sorted(
            result["mark_based_cashflows"], key=lambda row: row["event_id"])} == {
            "schema_version": "native_gate_consumption.v1.0",
            "variant_id": registration.variant_id, "run_id": registration.run_id,
            "recording_mode": "no-ledger", "execution_plan": registration.document["execution_plan"],
            "stage_inputs": {"snapshot_id": snapshot.snapshot_id, "strategy": "STR-THRU",
                             "fill_alpha": alpha, "event_ids": [other, "TEST_2024-05-02"],
                             "holdout_as_of_month": "2025-01",
                             "random_membership_version": "canonical-event-sha256.v1",
                             "rolling_membership_version": "calendar-months.v1"},
            "mark_based_cashflows": expected_cashflows}
    assert results[0]["variant_id"] != results[1]["variant_id"]
    assert {result["execution_plan"]["experiment_id"] for result in results} == {"synthetic-worker"}
    assert {result["execution_plan"]["arms"][0] for result in results} == {"candidate", "control"}
    assert _catalog(source[0]) == before
    assert ledger.read_bytes() == b"sentinel legacy ledger\n"
    private_checkouts = list(tmp_path.glob("arm-*/code/*"))
    assert len(private_checkouts) == 2
    for code in private_checkouts:
        assert (code / RUNNER).is_file()
        assert not experiments_ledger_path(code).exists()
    assert list(tmp_path.rglob("LEDGER.csv")) == [ledger]
    assert not list(tmp_path.rglob("REPORT.md"))
    assert not list(tmp_path.rglob("*.append.lock"))


def test_supervised_unconsumed_economics_refuse_without_receipt(source, tmp_path):
    """The real launcher must not admit an economic change with no consumer."""
    request = _registered(source, economic_params={"fill": 0.0, "strategy": "STR-THRU",
                                                  "exit": {"kind": "fixed_day", "trading_days": 1}})
    before = _catalog(source[0])
    result = _supervise(tmp_path, source, *request)
    assert result["code"] == "INVALID_EXPERIMENT_SPEC"
    assert _catalog(source[0]) == before
    assert not list(tmp_path.rglob("native_gate_consumption.json"))


@pytest.mark.parametrize("change", ["fill", "identity", "expected-ids", "run-id", "strategy",
                                    "arm", "environment", "registration"])
def test_changed_staged_binding_refuses_before_pricing(
        source, tmp_path, single_thread, forbidden_pricing, monkeypatch, change):
    """Canonical current-request differences cannot reuse another arm's registration."""
    document, registration, parameters = _registered(source)
    if change == "fill":
        document["economic_params"]["fill"] = 1.0
    if change == "identity":
        parameters["variant_id"] = "sha256:" + "0" * 64
    if change == "expected-ids":
        parameters["expected_ids"] = ["sha256:" + "0" * 64]
    if change == "run-id":
        parameters["run_id"] = "unregistered-run"
    if change == "strategy":
        document["economic_params"]["strategy"] = "SHORT-PUT"
    if change == "arm":
        document.update(primary_arm_id="SHORT-PUT", arms=["SHORT-PUT"])
    if change == "environment":
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    if change == "registration":
        registration = replace(registration, binding_json=b"{}")
    stage = _stage(tmp_path / "staging", document, registration)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert error.value.problem.retryable is False
    assert not (stage / "native_gate_consumption.json").exists()


@pytest.mark.parametrize("field", ["run_id", "variant_id"])
@pytest.mark.parametrize("value", [2**100, True, None, [], {}, b"identity", 1.5])
def test_ill_typed_worker_identities_refuse_before_database_binding(
        source, tmp_path, single_thread, forbidden_pricing, monkeypatch, field, value):
    """Valid registration cannot make malformed direct arguments SQL-safe."""
    from experiments import native_gate_worker as native

    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    parameters[field] = value
    before = _catalog(source[0])

    def forbidden(*args, **kwargs):
        pytest.fail("catalog opened before worker argument validation")

    monkeypatch.setattr(native.sqlite3, "connect", forbidden)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False
    assert _catalog(source[0]) == before
    assert not (stage / "native_gate_consumption.json").exists()


def test_recorded_execution_refuses_before_catalog_access(
        source, tmp_path, single_thread, forbidden_pricing, monkeypatch):
    """No-ledger admission cannot fall through to the legacy coordinator."""
    from experiments import native_gate_worker as native

    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    parameters["no_ledger"] = False
    before = _catalog(source[0])

    def forbidden(*args, **kwargs):
        pytest.fail("recorded request opened the catalog")

    monkeypatch.setattr(native.sqlite3, "connect", forbidden)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False
    kind = stages.registry().get("native_gate_pricing")
    assert kind.namespaces == {"smoke"}
    assert kind.retry.max_attempts == 1
    assert _catalog(source[0]) == before
    assert {path.name for path in stage.iterdir()} == {"spec.json", "native_registration.json"}


def test_catalog_connection_is_read_only_and_autocommit(source, tmp_path, single_thread, monkeypatch):
    """Probe the actual SQLite connection's write capability and transaction mode."""
    from experiments import native_gate_worker as native

    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    connect, checked = native.sqlite3.connect, []

    def observe(*args, **kwargs):
        conn = connect(*args, **kwargs)
        try:
            assert conn.isolation_level is None and not conn.in_transaction
            with pytest.raises(native.sqlite3.OperationalError, match="readonly"):
                conn.execute("CREATE TABLE forbidden_write(value)")
            checked.append(True)
            return conn
        except BaseException:
            conn.close()
            raise

    before = _catalog(source[0])
    monkeypatch.setattr(native.sqlite3, "connect", observe)
    result = worker.dispatch("native_gate_pricing", parameters, stage)
    assert checked == [True] and result["completed_ids"] == [registration.variant_id]
    assert _catalog(source[0]) == before


def test_library_environment_drift_refuses_before_pricing(
        source, tmp_path, single_thread, forbidden_pricing, monkeypatch):
    """Real package metadata discovery differs while code and threads stay fixed."""
    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    metadata = tmp_path / "library-metadata" / "numpy-0.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("Metadata-Version: 2.1\nName: numpy\nVersion: 0+synthetic\n")
    before = _catalog(source[0])
    monkeypatch.syspath_prepend(str(metadata.parent.parent))
    observed = environment_identity(1)
    expected = registration.document["environment"]
    assert observed["libraries"]["numpy"] == "0+synthetic" != expected["libraries"]["numpy"]
    assert observed == {**expected, "libraries": {**expected["libraries"], "numpy": "0+synthetic"}}
    assert all(os.environ.get(name) == "1" for name in THREAD_VARIABLES)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert error.value.problem.retryable is False
    assert _catalog(source[0]) == before
    assert not (stage / "native_gate_consumption.json").exists()


@pytest.mark.parametrize("name", ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
                                  "LOKY_MAX_CPU_COUNT"])
def test_each_launch_thread_variable_refuses_drift_before_pricing(
        source, tmp_path, single_thread, forbidden_pricing, monkeypatch, name):
    """Literal variable names independently cover every promised thread limit."""
    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    monkeypatch.setenv(name, "2")
    assert environment_identity(1) == registration.document["environment"]
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert error.value.problem.retryable is False
    assert not (stage / "native_gate_consumption.json").exists()


def test_registered_unsupported_runner_refuses_before_pricing(
        source, tmp_path, single_thread, forbidden_pricing):
    """A valid registered Python runner does not expand the worker allowlist."""
    document, registration, parameters = _registered(source, runner="experiments/v2_gate_variant.py")
    stage = _stage(tmp_path / "staging", document, registration)
    before = _catalog(source[0])
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False
    assert _catalog(source[0]) == before
    assert not (stage / "native_gate_consumption.json").exists()


@pytest.mark.parametrize("purpose", ["holdout", "final", "random_holdout"])
def test_holdout_request_refuses_before_catalog_or_staged_reads(tmp_path, purpose):
    """No explicit final-read purpose can touch prices or create a partial receipt."""
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", {"purpose": purpose, "no_ledger": True}, tmp_path)
    assert error.value.code == "HOLDOUT_ACCESS_DENIED"
    assert list(tmp_path.iterdir()) == []


def test_supervised_holdout_request_refuses_without_success_output(source, tmp_path):
    """The real supervisor preserves R5 and publishes no pricing result."""
    document, registration, parameters = _registered(source)
    parameters.update(purpose="final", catalog_path=str(tmp_path / "must-not-open.sqlite"))
    before = _catalog(source[0])
    result = _supervise(tmp_path, source, document, registration, parameters)
    assert result["code"] == "HOLDOUT_ACCESS_DENIED"
    assert _catalog(source[0]) == before
    assert not list(tmp_path.rglob("native_gate_consumption.json"))
    assert not (tmp_path / "must-not-open.sqlite").exists()


def test_worker_retains_registered_pin_when_head_moves(source, tmp_path, single_thread):
    """Admission cannot silently redirect to a newly published scope head."""
    from tests.test_v2_research_replay import _chain_rows, _commit, _event_rows

    document, registration, parameters = _registered(source)
    _commit(source[0], source[1], source[2], chain_rows=_chain_rows(exit_call=(8.0, 8.4)),
            event_rows=_event_rows(), receipt_id="changed-head",
            expected_head=source[4].snapshot_id, generation=1)
    stage = _stage(tmp_path / "staging", document, registration)
    worker.dispatch("native_gate_pricing", parameters, stage)
    result = json.loads((stage / "native_gate_consumption.json").read_text())
    assert result["stage_inputs"]["snapshot_id"] == source[4].snapshot_id
    assert result["mark_based_cashflows"][0]["pnl"] == pytest.approx(1.2)


@pytest.mark.parametrize("change,code", [("missing-spec", "INPUT_CHANGED"),
                                        ("symlink", "INVALID_EXPERIMENT_SPEC"),
                                        ("extra-binding", "INVALID_EXPERIMENT_SPEC"),
                                        ("bad-json", "INVALID_EXPERIMENT_SPEC")])
def test_malformed_staging_has_no_success_evidence(source, tmp_path, single_thread, change, code):
    """Staging errors are typed before replay; there is no current-path fallback."""
    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    if change == "missing-spec":
        (stage / "spec.json").unlink()
    elif change == "symlink":
        (stage / "spec.json").rename(stage / "outside.json")
        (stage / "spec.json").symlink_to(stage / "outside.json")
    elif change == "extra-binding":
        parameters["input_bindings"]["old-trades.parquet"] = "unused"
    else:
        (stage / "spec.json").write_text("not-json")
    before = _catalog(source[0])
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == code
    assert _catalog(source[0]) == before
    assert not (stage / "native_gate_consumption.json").exists()


@pytest.mark.parametrize("name", ["spec.json", "native_registration.json"])
def test_excessive_staged_json_depth_has_exact_refusal(source, tmp_path, single_thread, name):
    """Sub-cap nesting is a non-retryable refusal in dispatch and the real child."""
    document, registration, parameters = _registered(source)
    data = b"[" * 100_000 + b"0" + b"]" * 100_000
    stage = _stage(tmp_path / "staging", document, registration)
    (stage / name).write_bytes(data)
    before = _catalog(source[0])
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False
    assert error.value.problem.details == {}
    result = _supervise(tmp_path, source, document, registration, parameters,
                        staged_bytes={name: data}, expected_details={})
    assert result["code"] == "INVALID_EXPERIMENT_SPEC"
    assert result["retryable"] is False
    assert _catalog(source[0]) == before
    assert not list(tmp_path.rglob("native_gate_consumption.json"))


@pytest.mark.parametrize("case,data_code,ops_code", [
    ("corrupt-snapshot", "SNAPSHOT_UNRESOLVED", "SNAPSHOT_UNRESOLVED"),
    ("missing-calendar", "CONTRACT_MISMATCH", "EXPERIMENT_VARIANT_FAILED"),
    ("empty-calendar", "CALENDAR_UNAVAILABLE", "EXPERIMENT_VARIANT_FAILED"),
    ("malformed-contract", "JSONDecodeError", "EXPERIMENT_VARIANT_FAILED"),
    ("incomplete-contract", "MISSING_FIELD", "EXPERIMENT_VARIANT_FAILED"),
])
def test_real_data_failures_preserve_private_diagnostics(
        source, tmp_path, single_thread, case, data_code, ops_code):
    """Real repository failures cross the worker pipe without public detail leaks."""
    from tests.test_v2_research_replay import _chain_rows, _commit, _event_rows

    if case in {"missing-calendar", "empty-calendar"}:
        snapshot = _commit(source[0], source[1], source[2], chain_rows=_chain_rows(),
                           event_rows=_event_rows(), receipt_id=case,
                           expected_head=source[4].snapshot_id, generation=1,
                           include_daily_market=case != "missing-calendar",
                           daily_market_empty=case == "empty-calendar")
        source = (*source[:4], snapshot)
    document, registration, parameters = _registered(source)
    if case == "corrupt-snapshot":
        source[0].execute("DROP TRIGGER data_snapshots_no_update")
        source[0].execute("UPDATE data_snapshots SET manifest_hash=? WHERE snapshot_id=?",
                          ("sha256:" + "0" * 64, source[4].snapshot_id))
    elif case.endswith("-contract"):
        source[0].execute("DROP TRIGGER data_contracts_no_update")
        source[0].execute("PRAGMA ignore_check_constraints=ON")
        source[0].execute("UPDATE data_contracts SET definition_json=? WHERE table_name=?",
                          ("{" if case == "malformed-contract" else "{}", "earnings_events"))
        source[0].execute("PRAGMA ignore_check_constraints=OFF")
        assert source[3].resolve(source[4].snapshot_id) == source[4]
    stage = _stage(tmp_path / "staging", document, registration)
    before = _catalog(source[0])
    details = {"data_failure_code": data_code}
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == ops_code
    assert error.value.problem.details == details
    assert error.value.problem.retryable is False
    result = _supervise(tmp_path, source, document, registration, parameters,
                        expected_details=details)
    assert result["code"] == ops_code
    assert result["retryable"] is False
    assert _catalog(source[0]) == before
    assert not list(tmp_path.rglob("native_gate_consumption.json"))


def test_no_cross_job_cache_can_skip_native_admission(source, tmp_path):
    """A real compatible cache hit cannot bypass fresh pinned-data admission."""
    from engine.v2.contracts import CheckpointReceipt
    from engine.v2.foundation import format_timestamp
    from engine.v2.ops.catalog import dumps
    from engine.v2.ops.checkpoints import cache_identity, reuse
    from engine.v2.ops.input_bindings import resolve_bindings, resolved_inputs_hash

    document, registration, parameters = _registered(source)
    _supervise(tmp_path, source, document, registration, parameters,
               key="producer", shared_root="cache-boundary")

    def seed_compatible_checkpoint(conn, store, job, clock):
        producer = conn.execute("SELECT attempt_id, fence FROM attempts WHERE state='succeeded'").fetchone()
        output = conn.execute("SELECT artifact_id FROM attempt_outputs WHERE attempt_id=?",
                              (producer["attempt_id"],)).fetchone()
        ref = artifact(conn, store, output[0])
        inputs = resolved_inputs_hash(job, resolve_bindings(conn, store, job))
        key = cache_identity(kind=job.kind, inputs=inputs, implementation=job.implementation_ref,
                             parameters=content_hash(job.parameters), environment=job.environment_ref,
                             schema="legacy_action.v1.0", shard="default")
        receipt = CheckpointReceipt(
            stage_id=job.kind, shard_key="default", cache_key=key, input_hash=inputs,
            implementation_hash=job.implementation_ref, parameter_hash=content_hash(job.parameters),
            environment_hash=job.environment_ref, output_schema_ref="legacy_action.v1.0",
            artifact_refs=(ref,), validation_refs=(), producer_attempt_id=producer["attempt_id"],
            producer_fence=producer["fence"], committed_at=format_timestamp(clock.now()))
        with transaction(conn):
            conn.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)",
                         (key, job.kind, "default", dumps(receipt), producer["attempt_id"]))
        assert reuse(conn, store, key) == receipt
        source[0].execute("DROP TRIGGER data_snapshots_no_update")
        source[0].execute("UPDATE data_snapshots SET manifest_hash=? WHERE snapshot_id=?",
                          ("sha256:" + "0" * 64, source[4].snapshot_id))

    result = _supervise(tmp_path, source, document, registration, parameters,
                        key="consumer", shared_root="cache-boundary",
                        before_start=seed_compatible_checkpoint,
                        expected_details={"data_failure_code": "SNAPSHOT_UNRESOLVED"})
    assert result["code"] == "SNAPSHOT_UNRESOLVED"


def test_supervised_changed_fill_cannot_reuse_registered_identity(source, tmp_path):
    """R6 current-request admission occurs inside the actual launched process."""
    document, registration, parameters = _registered(source)
    document["economic_params"]["fill"] = 1.0
    before = _catalog(source[0])
    result = _supervise(tmp_path, source, document, registration, parameters)
    assert result["code"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _catalog(source[0]) == before
    assert not list(tmp_path.rglob("native_gate_consumption.json"))


def test_finite_quotes_that_overflow_cashflows_refuse_whole_worker(source, tmp_path, single_thread):
    """Real pricing overflow cannot become a canonical tagged-value success receipt."""
    from tests.test_v2_research_replay import _chain_rows, _commit, _event_rows

    rows = [{**row, "bid": 9e307, "ask": 9e307}
            if row["obs_date"].date().isoformat() == "2024-05-03" else row for row in _chain_rows()]
    _commit(source[0], source[1], source[2], chain_rows=rows,
            event_rows=_event_rows(), receipt_id="overflow-quotes",
            expected_head=source[4].snapshot_id, generation=1)
    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "EXPERIMENT_VARIANT_FAILED"
    assert error.value.problem.message == "native pricing produced nonfinite cash flows"
    assert not (stage / "native_gate_consumption.json").exists()


@pytest.mark.parametrize("opt_in", [False, None], ids=["false", "missing"])
def test_smoke_opt_in_is_required_by_submission_and_worker(tmp_path, opt_in):
    """Actual submission refuses missing/false opt-in before storing a job."""
    parameters = {"expected_ids": ["variant"], "catalog_path": "unopened.sqlite",
                  "objects_root": "unopened", "run_id": "run", "variant_id": "variant",
                  "input_bindings": {}}
    if opt_in is not None:
        parameters["no_ledger"] = opt_in
    clock = SystemClock()
    conn = open_catalog(tmp_path / "catalog.sqlite", clock=clock)
    job = JobSpec(kind="native_gate_pricing", spec_hash=content_hash({}),
                  implementation_ref=content_hash({}), environment_ref=content_hash({}),
                  parameters=parameters, output_namespace="smoke",
                  resource_class="native_experiment", retry_policy_ref="once",
                  checkpoint_contract_ref="native_gate_consumption.v1.0")
    try:
        before = list(conn.iterdump())
        with pytest.raises(OpsError) as error:
            submit(conn, stages.registry(), NamespacePolicy({"operator": frozenset({"smoke"})}),
                   SubmitRequest(namespace="smoke", idempotency_key="opt-in", principal="operator", job=job),
                   clock=clock)
        assert error.value.code == "INVALID_REQUEST"
        assert list(conn.iterdump()) == before
    finally:
        conn.close()
    stage = tmp_path / "unopened"
    stage.mkdir()
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert list(stage.iterdir()) == []


@pytest.mark.parametrize("change", ["bytes", "missing", "malformed", "encoding", "unreadable"])
def test_registered_code_must_match_the_running_worker(
        source, tmp_path, single_thread, forbidden_pricing, monkeypatch, change):
    """A real source-closure change cannot reuse a registration in another checkout."""
    from engine.v2.ops.fingerprints import source_closure

    document, registration, parameters = _registered(source)
    code = tmp_path / "changed-code"
    for name in source_closure(REPO, [RUNNER]):
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / name).read_bytes())
    runner = code / RUNNER
    if change == "bytes":
        runner.write_bytes(runner.read_bytes() + b"\n# Synthetic source identity change.\n")
    elif change == "missing":
        runner.unlink()
    elif change == "unreadable":
        original_open = Path.open

        def deny_source_open(path, *args, **kwargs):
            if path == runner:
                raise PermissionError("synthetic source read denial")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", deny_source_open)
    else:
        runner.write_bytes(b"def broken(:\n" if change == "malformed" else b"\xff")
    monkeypatch.chdir(code)
    before = _catalog(source[0])
    stage = _stage(tmp_path / "staging", document, registration)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert error.value.problem.retryable is False
    assert _catalog(source[0]) == before
    assert not (stage / "native_gate_consumption.json").exists()


def test_supervised_missing_strategy_refuses_before_population_reads(source, tmp_path):
    """A worker never substitutes the arm name for an undeclared strategy."""
    request = _registered(source, economic_params={"fill": 0.0})
    before = _catalog(source[0])
    result = _supervise(tmp_path, source, *request)
    assert result["code"] == "INVALID_EXPERIMENT_SPEC"
    assert _catalog(source[0]) == before
    assert not list(tmp_path.rglob("native_gate_consumption.json"))


def test_supervised_changed_strategy_cannot_reuse_registered_identity(source, tmp_path):
    """Changed strategy is an economic conflict, independent of unchanged arm identity."""
    document, registration, parameters = _registered(source)
    document["economic_params"]["strategy"] = "SHORT-PUT"
    before = _catalog(source[0])
    result = _supervise(tmp_path, source, document, registration, parameters)
    assert result["code"] == "EXPERIMENT_IDENTITY_CONFLICT"
    assert _catalog(source[0]) == before
    assert not list(tmp_path.rglob("native_gate_consumption.json"))


def test_manual_declared_strategy_must_match_consumed_strategy_before_reads():
    """Manual fill-only compatibility cannot turn a declared strategy into unused metadata."""
    from experiments.v2_gate_variant import price_variant

    spec = experiment_spec_from_document({**_document(), "economic_params": {
        "fill": 0.0, "strategy": "SHORT-PUT"}})
    with pytest.raises(OpsError) as error:
        price_variant(None, None, spec, strategy="STR-THRU", as_of_month="2025-01")
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"


def test_registered_strategy_changes_real_consumed_geometry(source, tmp_path, single_thread):
    """A separately bound strategy must change real legs/cashflows, not just receipt metadata."""
    document, registration, parameters = _registered(
        source, primary_arm_id="calendar", arms=["calendar"],
        economic_params={"fill": 0.0, "strategy": "CAL-P"})
    stage = _stage(tmp_path / "staging", document, registration)
    worker.dispatch("native_gate_pricing", parameters, stage)
    result = json.loads((stage / "native_gate_consumption.json").read_text())
    assert result["stage_inputs"]["strategy"] == "CAL-P"
    row, = result["mark_based_cashflows"]
    assert row["entry_cost"] == pytest.approx(1.8)
    assert row["exit_value"] == pytest.approx(1.6)
    assert row["pnl"] == pytest.approx(-0.2)


@pytest.mark.parametrize("field", ["hypothesis", "primary_arm_id", "economic_params", "price_source"])
@pytest.mark.parametrize("missing", [True, False], ids=["missing", "null"])
def test_supervised_required_spec_fields_use_experiment_refusal(source, tmp_path, field, missing):
    """The real staged parser's required-field errors keep the worker contract."""
    document, registration, parameters = _registered(source)
    if missing:
        del document[field]
    else:
        document[field] = None
    before = _catalog(source[0])
    result = _supervise(tmp_path, source, document, registration, parameters)
    assert result["code"] == "INVALID_EXPERIMENT_SPEC"
    assert _catalog(source[0]) == before
    assert not list(tmp_path.rglob("native_gate_consumption.json"))
    assert not list(tmp_path.rglob("REPORT.md"))


@pytest.mark.parametrize("name", ["spec.json", "native_registration.json"])
@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink", "oversized"])
def test_staged_documents_refuse_unsafe_files_without_blocking(source, tmp_path, single_thread, name, kind):
    """A real dispatch child must terminate on each unsafe staged document."""
    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    path = stage / name
    path.unlink()
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        path.symlink_to(stage / "absent-target")
    else:
        with path.open("wb") as stream:
            stream.truncate(16_777_217)
    before = _catalog(source[0])
    program = """
import json, sys
from pathlib import Path
from engine.v2.ops.worker import dispatch
from engine.v2.ops.errors import OpsError
try:
    dispatch('native_gate_pricing', json.loads(sys.argv[1]), Path(sys.argv[2]))
except OpsError as error:
    print(error.code)
else:
    raise AssertionError('unsafe staged input was accepted')
"""
    child = subprocess.run([sys.executable, "-c", program, json.dumps(parameters), str(stage)],
                           capture_output=True, text=True, timeout=10, check=True)
    assert child.stdout.strip() == "INVALID_EXPERIMENT_SPEC"
    assert _catalog(source[0]) == before
    assert not (stage / "native_gate_consumption.json").exists()


@pytest.mark.parametrize("name", ["spec.json", "native_registration.json"])
def test_staged_replacement_cannot_bypass_no_follow(source, tmp_path, single_thread, monkeypatch, name):
    """Swap immediately before the real open, retaining its descriptor checks."""
    from experiments import native_gate_worker as native

    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    path, outside = stage / name, tmp_path / "replacement.json"
    original_open = native.os.open
    swapped = []

    def replace_before_open(target, flags, *args, **kwargs):
        if Path(target) == path and not swapped:
            path.rename(outside)
            path.symlink_to(outside)
            swapped.append(True)
        return original_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(native.os, "open", replace_before_open)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert swapped == [True]
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert not (stage / "native_gate_consumption.json").exists()


@pytest.mark.parametrize("name", ["spec.json", "native_registration.json"])
def test_staged_replacement_after_open_consumes_original_bytes(
        source, tmp_path, single_thread, monkeypatch, name):
    """Replacing a validated pathname cannot redirect the live descriptor."""
    from experiments import native_gate_worker as native

    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    path, replacement = stage / name, tmp_path / "replacement.json"
    replacement.write_bytes(b"{}")
    original_stat, original_fstat = path.stat(), native.os.fstat
    swapped = []

    def replace_after_open(fd):
        info = original_fstat(fd)
        if (info.st_dev, info.st_ino) == (original_stat.st_dev, original_stat.st_ino) and not swapped:
            os.replace(replacement, path)
            swapped.append(True)
        return info

    before = _catalog(source[0])
    monkeypatch.setattr(native.os, "fstat", replace_after_open)
    result = worker.dispatch("native_gate_pricing", parameters, stage)
    receipt = json.loads((stage / "native_gate_consumption.json").read_bytes())
    assert swapped == [True] and path.read_bytes() == b"{}"
    assert result["completed_ids"] == [registration.variant_id]
    assert receipt["execution_plan"] == registration.document["execution_plan"]
    assert receipt["variant_id"] == registration.variant_id
    assert receipt["stage_inputs"]["event_ids"] == registration.document["event_ids"]
    assert _catalog(source[0]) == before


@pytest.mark.parametrize("field", ["event_ids", "holdout_as_of_month",
                                   "random_membership_version", "rolling_membership_version"])
def test_consumed_provenance_must_match_registration(
        source, tmp_path, single_thread, monkeypatch, field):
    """Exercise genuine pinned replay, then alter only its returned provenance."""
    from experiments import native_gate_worker as native

    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    original_price, priced = native.price_variant, []

    def mismatched(*args, **kwargs):
        result = original_price(*args, **kwargs)
        priced.append(result)
        value = ("OTHER_2024-05-02",) if field == "event_ids" else "changed-provenance"
        return replace(result, stage_inputs=replace(result.stage_inputs, **{field: value}))

    before = _catalog(source[0])
    monkeypatch.setattr(native, "price_variant", mismatched)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert len(priced) == 1 and len(priced[0].replay.trades) == 1
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
    assert error.value.problem.retryable is False
    assert not (stage / "native_gate_consumption.json").exists()
    assert _catalog(source[0]) == before


@pytest.mark.parametrize("name", ["spec.json", "native_registration.json"])
@pytest.mark.parametrize("size", [16_777_216, 16_777_217], ids=["at-cap", "above-cap"])
def test_staged_literal_size_boundary(tmp_path, name, size):
    """Independent byte counts pin the documented inclusive 16 MiB limit."""
    from experiments import native_gate_worker as native

    path = tmp_path / name
    expected = b"x" * size
    path.write_bytes(expected)
    if size == 16_777_216:
        assert native._read_staged(path) == expected
    else:
        with pytest.raises(OpsError) as error:
            native._read_staged(path)
        assert error.value.code == "INVALID_EXPERIMENT_SPEC"
        assert error.value.problem.retryable is False


@pytest.mark.parametrize("name", ["spec.json", "native_registration.json"])
@pytest.mark.parametrize("size", [16_777_216, 16_777_217], ids=["at-cap", "above-cap"])
def test_staged_growth_is_bounded_after_descriptor_validation(tmp_path, monkeypatch, name, size):
    """A real file growing after fstat cannot bypass the bounded read."""
    from experiments import native_gate_worker as native

    path = tmp_path / name
    path.write_bytes(b"{}")
    original_fstat = native.os.fstat

    def grow_after_stat(fd):
        info = original_fstat(fd)
        with path.open("ab") as stream:
            stream.truncate(size)
        return info

    monkeypatch.setattr(native.os, "fstat", grow_after_stat)
    if size == 16_777_216:
        assert native._read_staged(path) == b"{}" + b"\0" * 16_777_214
    else:
        with pytest.raises(OpsError) as error:
            native._read_staged(path)
        assert error.value.code == "INVALID_EXPERIMENT_SPEC"
        assert error.value.problem.retryable is False


@pytest.mark.parametrize("kind", ["file", "directory", "symlink", "hardlink", "fifo", "write-error",
                                 "replacement-write-error", "cleanup-error"])
def test_receipt_destination_refuses_without_truncation_or_blocking(source, tmp_path, single_thread, kind):
    """Real dispatch and OS writes preserve foreign targets and typed I/O failure."""
    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    path, outside = stage / "native_gate_consumption.json", tmp_path / "foreign.json"
    outside.write_bytes(b"foreign sentinel")
    replacement = tmp_path / "replacement.json"
    if kind == "file":
        path.write_bytes(b"existing sentinel")
    elif kind == "directory":
        path.mkdir()
    elif kind == "symlink":
        path.symlink_to(outside)
    elif kind == "hardlink":
        path.hardlink_to(outside)
        original_link = path.stat()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "replacement-write-error":
        replacement.write_bytes(b"replacement sentinel")
        replacement_stat = replacement.stat()
    before = _catalog(source[0])
    program = """
import errno, json, os, resource, signal, sys
from pathlib import Path
from engine.v2.ops.worker import dispatch
from engine.v2.ops.errors import OpsError
from experiments import native_gate_worker as native
replacements = []
if sys.argv[3] == 'cleanup-error':
    unlink = os.unlink
    def deny_private_cleanup(target, *args, **kwargs):
        if Path(target).name == 'receipt.json':
            raise PermissionError(errno.EACCES, 'injected private cleanup denial')
        return unlink(target, *args, **kwargs)
    native.os.unlink = deny_private_cleanup
if sys.argv[3] == 'replacement-write-error':
    encode = native.canonical_json
    def replace_destination(value):
        if value.get('schema_version') == 'native_gate_consumption.v1.0':
            os.replace(sys.argv[4], Path(sys.argv[2]) / 'native_gate_consumption.json')
            replacements.append(True)
        return encode(value)
    native.canonical_json = replace_destination
if sys.argv[3] in ('write-error', 'replacement-write-error'):
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    limit = 64 if sys.argv[3] == 'replacement-write-error' else 0
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
try:
    dispatch('native_gate_pricing', json.loads(sys.argv[1]), Path(sys.argv[2]))
except OpsError as error:
    assert len(replacements) == (sys.argv[3] == 'replacement-write-error')
    print(json.dumps({'code': error.code, 'retryable': error.problem.retryable}))
else:
    raise AssertionError('unsafe receipt destination was accepted')
"""
    child = subprocess.run([sys.executable, "-c", program, json.dumps(parameters), str(stage), kind,
                            str(replacement)],
                           capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(child.stdout.strip().splitlines()[-1]) == {"code": "INPUT_CHANGED", "retryable": False}
    assert _catalog(source[0]) == before
    assert outside.read_bytes() == b"foreign sentinel"
    if kind == "file":
        assert path.read_bytes() == b"existing sentinel"
    elif kind == "write-error":
        assert not path.exists()
    elif kind == "replacement-write-error":
        assert path.read_bytes() == b"replacement sentinel"
        current = path.stat()
        assert (current.st_dev, current.st_ino, current.st_nlink) == (
            replacement_stat.st_dev, replacement_stat.st_ino, replacement_stat.st_nlink)
        assert not replacement.exists()
        assert list(stage.glob(".native-gate-*")) == []
    elif kind == "hardlink":
        current = path.stat()
        assert path.read_bytes() == b"foreign sentinel" and path.samefile(outside)
        assert (current.st_dev, current.st_ino, current.st_nlink) == (
            original_link.st_dev, original_link.st_ino, original_link.st_nlink)
    elif kind == "cleanup-error":
        assert json.loads(path.read_bytes())["schema_version"] == "native_gate_consumption.v1.0"
        private = list(stage.glob(".native-gate-*/receipt.json"))
        assert len(private) == 1 and private[0].samefile(path)
    elif kind == "directory":
        assert path.is_dir() and list(path.iterdir()) == []
    elif kind == "symlink":
        assert path.is_symlink()
    elif kind == "fifo":
        import stat
        assert stat.S_ISFIFO(path.stat().st_mode)


def test_supervised_receipt_conflict_has_no_success_output(source, tmp_path, monkeypatch):
    """An occupied assigned destination stays private and cannot publish success."""
    request = _registered(source)
    outside = tmp_path / "foreign.json"
    outside.write_bytes(b"foreign sentinel")
    original_stage = Service._stage_legacy_inputs

    def occupy_destination(service, claim, launch):
        overlay = original_stage(service, claim, launch)
        path = service.store.staging_dir(claim.attempt_id) / "native_gate_consumption.json"
        path.symlink_to(outside)
        return overlay

    monkeypatch.setattr(Service, "_stage_legacy_inputs", occupy_destination)
    before = _catalog(source[0])
    result = _supervise(tmp_path, source, *request, expected_details={})
    assert result["code"] == "INPUT_CHANGED"
    assert result["retryable"] is False
    assert outside.read_bytes() == b"foreign sentinel"
    assert _catalog(source[0]) == before


@pytest.mark.parametrize("parameters", [None, [], 0, False, "", ()])
def test_non_mapping_worker_arguments_refuse_before_io(tmp_path, parameters):
    """The real dispatch boundary refuses malformed argument shapes without staging."""
    stage = tmp_path / "must-not-open"
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert error.value.problem.retryable is False
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_partial_receipt_cleanup_preserves_error_and_closes_fd(tmp_path, cleanup_fails):
    """Real partial OS writes retain their failure even if cleanup is denied."""
    path = tmp_path / "native_gate_consumption.json"
    program = """
import errno, json, os, resource, signal, sys
from pathlib import Path
from experiments import native_gate_worker as native
path = Path(sys.argv[1])
cleanup_fails = sys.argv[2] == 'True'
real_open, real_unlink = os.open, os.unlink
fds, failures, partials, private_paths = [], [], [], []
def observe_open(*args, **kwargs):
    fd = real_open(*args, **kwargs)
    fds.append(fd)
    return fd
def cleanup(target):
    target = Path(target)
    assert target != path and target.parent.parent == path.parent
    assert target.parent.stat().st_mode & 0o777 == 0o700
    private_paths.append(target)
    failures.append(sys.exception())
    partials.append(target.read_bytes())
    if cleanup_fails:
        raise PermissionError(errno.EACCES, 'injected cleanup denial')
    real_unlink(target)
native.os.open, native.os.unlink = observe_open, cleanup
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (64, resource.getrlimit(resource.RLIMIT_FSIZE)[1]))
try:
    native._write_receipt(path, {'padding': 'x' * 256})
except OSError as error:
    assert len(failures) == 1 and error is failures[0]
    assert error.errno == errno.EFBIG
else:
    raise AssertionError('partial write unexpectedly succeeded')
assert len(fds) == 1
try:
    os.fstat(fds[0])
except OSError as error:
    assert error.errno == errno.EBADF
else:
    raise AssertionError('receipt descriptor was not closed')
assert partials == [b'{"padding":"' + b'x' * 52]
assert not path.exists()
assert private_paths[0].exists() is cleanup_fails
if cleanup_fails:
    assert private_paths[0].read_bytes() == partials[0]
else:
    assert not private_paths[0].parent.exists()
print(json.dumps({'partial_bytes': len(partials[0]), 'cleanup_failed': cleanup_fails}))
"""
    child = subprocess.run([sys.executable, "-c", program, str(path), str(cleanup_fails)],
                           capture_output=True, text=True, timeout=10, check=True)
    assert json.loads(child.stdout) == {"partial_bytes": 64, "cleanup_failed": cleanup_fails}
