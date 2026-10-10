"""Real staged worker admission over committed synthetic option-chain snapshots."""
import json
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
from engine.v2.ops.experiments import experiment_spec_from_document
from engine.v2.ops.fingerprints import environment_identity, worker_source_manifest
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from experiments.native_gate_worker import RUNNER
from experiments.native_registration import register_native
from tests.ops_support import TEST_POLICY, run_until
from tests.v2.research.test_gate_variant import _document
from tests.v2.research.test_gate_variant import source as source
from tests.v2.research.test_native_registration import _catalog

REPO = Path(__file__).resolve().parents[3]


def _registered(source, alpha=0.0, **changes):
    """Bind the worker and strategy without conflating strategy with arm identity."""
    arm = "candidate" if alpha == 0.0 else "control"
    document = {**_document(alpha), "experiment_id": "synthetic-worker",
                "runner": RUNNER, "primary_arm_id": arm, "arms": [arm],
                "economic_params": {"fill": alpha, "strategy": "STR-THRU"}, **changes}
    spec = experiment_spec_from_document(document)
    registration = register_native(source[0], source[2], source[3], spec,
                                   code_root=REPO, as_of_month="2025-01")
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


def _supervise(tmp_path, source, document, registration, parameters, *, key="smoke"):
    """Submit through the production registry, launcher, child and checkpoint path."""
    root = tmp_path / key
    root.mkdir()
    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    store = ArtifactStore(root)
    try:
        bindings = {}
        for name, data in (("spec.json", json.dumps(document).encode()),
                           ("native_registration.json", registration.binding_json)):
            ref = store.publish_bytes(data, schema_ref="native_staged_input.v1.0")
            with transaction(conn):
                register_artifact(conn, ref, None, clock)
            bindings[name] = ref.artifact_id
        job = JobSpec(kind="native_gate_pricing", spec_hash=content_hash(document),
                      implementation_ref=content_hash(worker_source_manifest(REPO)),
                      environment_ref=content_hash(environment_identity(1)),
                      parameters={**parameters, "input_bindings": bindings},
                      input_refs=tuple(bindings.values()), output_namespace="smoke",
                      resource_class="native_experiment", retry_policy_ref="once",
                      checkpoint_contract_ref="native_gate_consumption.v1.0")
        receipt = submit(conn, stages.registry(), NamespacePolicy({"operator": frozenset({"smoke"})}),
                         SubmitRequest(namespace="smoke", idempotency_key=key, principal="operator", job=job),
                         clock=clock)
        service = Service(conn, root, stages.registry(), TEST_POLICY, clock=clock, code_source=REPO)
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
        if state == "succeeded":
            assert len(outputs) == 1 and outputs[0]["name"] == "native_gate_consumption"
            return json.loads(store.read_verified(artifact(conn, store, outputs[0]["artifact_id"])))
        assert outputs == []
        return json.loads(attempt["failure_json"])
    finally:
        conn.close()


def test_supervised_arms_consume_independent_fill_and_never_record(source, tmp_path):
    """Two real child processes price independent registered economics, not shared trades."""
    requests = [_registered(source, alpha) for alpha in (0.0, 1.0)]
    before = _catalog(source[0])
    ledger = source[2].root / "LEDGER.csv"
    ledger.write_bytes(b"sentinel legacy ledger\n")
    results = [_supervise(tmp_path, source, *request, key=f"arm-{i}")
               for i, request in enumerate(requests)]
    for result, alpha, cost, exit_value, pnl in zip(results, (0.0, 1.0), (3.8, 3.0), (5.0, 5.8), (1.2, 2.8)):
        assert result["recording_mode"] == "no-ledger"
        assert result["stage_inputs"]["fill_alpha"] == alpha
        assert result["execution_plan"]["economic_params"] == {"fill": alpha, "strategy": "STR-THRU"}
        assert result["stage_inputs"]["snapshot_id"] == source[4].snapshot_id
        assert result["stage_inputs"]["strategy"] == "STR-THRU"
        row, = result["mark_based_cashflows"]
        assert row["entry_cost"] == pytest.approx(cost)
        assert row["exit_value"] == pytest.approx(exit_value)
        assert row["pnl"] == pytest.approx(pnl)
    assert results[0]["variant_id"] != results[1]["variant_id"]
    assert {result["execution_plan"]["experiment_id"] for result in results} == {"synthetic-worker"}
    assert {result["execution_plan"]["arms"][0] for result in results} == {"candidate", "control"}
    assert _catalog(source[0]) == before
    assert ledger.read_bytes() == b"sentinel legacy ledger\n"
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


@pytest.mark.parametrize("change", ["fill", "identity", "strategy", "arm", "environment", "registration"])
def test_changed_staged_binding_refuses_before_pricing(source, tmp_path, single_thread, monkeypatch, change):
    """Canonical current-request differences cannot reuse another arm's registration."""
    document, registration, parameters = _registered(source)
    if change == "fill":
        document["economic_params"]["fill"] = 1.0
    if change == "identity":
        parameters["variant_id"] = "sha256:" + "0" * 64
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
    assert not (stage / "native_gate_consumption.json").exists()


def test_recorded_execution_refuses_before_catalog_access(tmp_path):
    """No-ledger admission cannot fall through to the legacy coordinator."""
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", {"no_ledger": False}, tmp_path)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    kind = stages.registry().get("native_gate_pricing")
    assert kind.namespaces == {"smoke"}
    assert kind.retry.max_attempts == 1
    assert list(tmp_path.iterdir()) == []


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


@pytest.mark.parametrize("change", ["missing-spec", "symlink", "extra-binding", "bad-json"])
def test_malformed_staging_has_no_success_evidence(source, tmp_path, single_thread, change):
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
    assert error.value.code in {"INPUT_CHANGED", "INVALID_EXPERIMENT_SPEC"}
    assert _catalog(source[0]) == before
    assert not (stage / "native_gate_consumption.json").exists()


def test_no_cross_job_cache_can_skip_native_admission(source):
    """Both supervisor checkpoint lookup and storage consult the same guard."""
    from types import SimpleNamespace

    claim = SimpleNamespace(spec=SimpleNamespace(kind="native_gate_pricing"))
    assert Service._cache_allowed(None, claim) is False


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

    rows = [{**row, "bid": 9e307, "ask": 9e307} for row in _chain_rows()]
    _commit(source[0], source[1], source[2], chain_rows=rows,
            event_rows=_event_rows(), receipt_id="overflow-quotes",
            expected_head=source[4].snapshot_id, generation=1)
    document, registration, parameters = _registered(source)
    stage = _stage(tmp_path / "staging", document, registration)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "EXPERIMENT_VARIANT_FAILED"
    assert not (stage / "native_gate_consumption.json").exists()


def test_smoke_opt_in_is_required_by_submission_and_worker(tmp_path):
    """The schema and dispatch both require explicit no-ledger intent."""
    from engine.v2.foundation import DocumentError, from_document

    parameters = {"expected_ids": ["variant"], "catalog_path": "unopened.sqlite",
                  "objects_root": "unopened", "run_id": "run", "variant_id": "variant",
                  "input_bindings": {}}
    with pytest.raises(DocumentError):
        from_document(stages.NativeGateParameters, parameters)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, tmp_path)
    assert error.value.code == "INVALID_EXPERIMENT_SPEC"
    assert list(tmp_path.iterdir()) == []


def test_registered_code_must_match_the_running_worker(source, tmp_path, single_thread):
    """A real source-closure change cannot reuse a registration in another checkout."""
    from engine.v2.ops.fingerprints import source_closure

    document, _, parameters = _registered(source)
    code = tmp_path / "changed-code"
    for name in source_closure(REPO, [RUNNER]):
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((REPO / name).read_bytes())
    runner = code / RUNNER
    runner.write_bytes(runner.read_bytes() + b"\n# Synthetic source identity change.\n")
    document["experiment_id"] = "synthetic-changed-code"
    registration = register_native(source[0], source[2], source[3],
                                   experiment_spec_from_document(document),
                                   code_root=code, as_of_month="2025-01")
    parameters.update(run_id=registration.run_id, variant_id=registration.variant_id,
                      expected_ids=[registration.variant_id])
    stage = _stage(tmp_path / "staging", document, registration)
    with pytest.raises(OpsError) as error:
        worker.dispatch("native_gate_pricing", parameters, stage)
    assert error.value.code == "EXPERIMENT_IDENTITY_CONFLICT"
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
