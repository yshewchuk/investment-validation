"""P6 slice 5: recipes, frozen states and artifacts, plus operator promotes, as jobs."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from engine.v2.contracts import LegacyFileRef, LegacyInputManifest
from engine.v2.foundation import ArtifactStore, SystemClock, to_document
from engine.v2.models import (
    ArtifactInventoryMember,
    ArtifactMember,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
    current_release,
    stage_release,
)
from engine.v2.models.training import ReceiptIssue, TrainingRefused, current_recipes
from engine.v2.ops import cli, executor, nightly, stages, training
from engine.v2.ops.bootstrap import open_catalog
from engine.v2.ops.catalog import transaction
from engine.v2.ops.checkpoints import artifact, register_artifact
from engine.v2.ops.errors import OpsError
from engine.v2.ops.plans import nightly_plan, request_from_plan
from engine.v2.ops.submission import NamespacePolicy, submit
from engine.v2.ops.supervisor import Service
from tests.ops_support import TEST_POLICY, catalog, run_until

REPO = Path(__file__).resolve().parents[1]
POLICY = NamespacePolicy({"operator": frozenset({"shadow"})})


def _params(**changes):
    fields = {"expected_ids": ("training",), "mode": "state",
              "state": "paired_residual_pool"}
    fields.update(changes)
    return training.TrainingParameters(**fields)


def _problems(**changes):
    return training.training_parameter_problems(None, _params(**changes))


def _worker_params(**changes):
    fields = {"expected_ids": ("training",), "mode": "state",
              "state": "driver_residual_pool:size", "recipe": "", "alpha": None,
              "cutoffs": (), "strategies": (), "pairs_path": "", "ticker_chunk": 1000}
    fields.update(changes)
    return fields


def _recipe_key(calibration):
    for key, recipe in current_recipes().items():
        if (recipe.folds.kind == "request_cutoff") == calibration:
            return key.label()
    raise AssertionError("no matching recipe in the registry")


def _release_fixture(release_id):
    """A one-binding synthetic ``ModelRelease`` plus its matching inventory,
    the same shape ``tests/test_v2_models_deployment.py`` stages."""
    payload = json.dumps({
        "schema_version": "linear_estimator.v1.0",
        "feature_order": ["x"],
        "outputs": [{"name": "prediction", "intercept": 1.0, "coefficients": [2.0]}],
    }, sort_keys=True).encode()
    member_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(
        binding_id="b1", model_id="m1", role="size", strategy_id="*",
        decision_clock_id="entry-close", adapter="json-linear.v1",
        feature_order=("x",), output_names=("prediction",), members=(member,))
    release = ModelRelease(release_id=release_id, deployment_id="d1", bindings=(binding,))

    inv_member = ArtifactInventoryMember(member_id="m1:estimator", kind="estimator",
                                         artifact_ref="artifact://m1", content_hash=member_hash)
    artifact = ModelArtifactInventory(
        artifact_id="m1", role="size", strategy_ids=("*",), compatible_clock_ids=("entry-close",),
        target_contract_ref="return.v1", ordered_features=("x",), members=(inv_member,))
    inv_binding = ReleaseBinding(
        role="size", strategy_id="*", clock_id="entry-close", artifact_id="m1",
        ordered_features=("x",), required_member_kinds=("estimator",))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=("entry-close",),
        artifacts=(artifact,), bindings=(inv_binding,),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",))
    return release, inventory, {member_hash: payload}


def test_training_kind_is_registered_with_experiment_heavy_profile():
    kind = stages.registry().get("training")
    assert kind.resource_classes == frozenset({"experiment_heavy"})
    assert kind.checkpoint_contract == "training_job_result.v1.0"
    assert kind.store_domains == (("legacy_store", "read"),)
    assert kind.retry.max_attempts == 1
    with pytest.raises(OpsError, match="unsupported job kind"):
        stages.registry().get("bogus_training_kind")


def test_training_parameters_reject_unknown_mode():
    problems = _problems(mode="forecast")
    assert any("mode" in problem for problem in problems)


def test_training_parameters_reject_expected_ids_mismatch():
    problems = _problems(expected_ids=("training", "extra"))
    assert any("expected_ids" in problem for problem in problems)


def test_recipe_mode_rejects_unknown_recipe_key():
    problems = _problems(mode="recipe", recipe="nope:*:champion", state="")
    assert any("recipe" in problem for problem in problems)


def test_recipe_mode_rejects_malformed_recipe_string():
    problems = _problems(mode="recipe", recipe="size", state="")
    assert any("recipe" in problem for problem in problems)


def test_calibration_recipe_requires_alpha_and_cutoff():
    key = _recipe_key(calibration=True)
    problems = _problems(mode="recipe", recipe=key, state="")
    assert any("alpha" in problem for problem in problems)
    assert any("cutoff" in problem for problem in problems)
    assert _problems(mode="recipe", recipe=key, state="", alpha=0.5,
                     cutoffs=("2026-09-18",)) == []


def test_non_calibration_recipe_rejects_alpha_or_cutoff():
    key = _recipe_key(calibration=False)
    assert _problems(mode="recipe", recipe=key, state="", alpha=0.5)
    assert _problems(mode="recipe", recipe=key, state="", cutoffs=("2026-09-18",))
    assert _problems(mode="recipe", recipe=key, state="") == []


def test_state_mode_rejects_board_analog_and_trailing_cutoff_state_names():
    for name in ("board_analog_matcher", "trailing_pnl_cutoff"):
        problems = _problems(mode="state", state=name)
        assert any("state" in problem for problem in problems), name


def test_state_mode_rejects_extra_cutoff_on_non_paired_pool_member():
    assert _problems(mode="state", state="driver_residual_pool:size",
                     cutoffs=("2026-09-18",))
    assert _problems(mode="state", state="paired_residual_pool",
                     cutoffs=("2026-09-18",)) == []


def test_training_plan_names_experiment_heavy_and_refuses_invalid_params():
    plan = training.training_plan(mode="state", state="paired_residual_pool")
    assert plan["kind"] == "training"
    assert plan["resource_class"] == "experiment_heavy"
    assert plan["effects"] == ["staged"]
    assert plan["parameters"]["mode"] == "state"
    with pytest.raises(OpsError, match="training plan is invalid"):
        training.training_plan(mode="nope")


def test_training_worker_routes_each_mode(tmp_path, monkeypatch):
    from tools import phase5_training_job as job

    calls = []

    def record(name):
        def run(*args, **kwargs):
            calls.append((name, kwargs.get("plan_only")))
            if name == "run_training_job":
                return SimpleNamespace(outcomes=(
                    SimpleNamespace(fold_id="fold-1", status="fitted"),))
            return {"mode": name}
        return run

    for name in ("run_training_job", "run_state_job", "run_board_analog_job",
                 "run_trailing_cutoff_job"):
        monkeypatch.setattr(job, name, record(name))
    monkeypatch.setattr(job, "build_dataset", lambda recipe, *, pairs_path=None: "dataset")
    monkeypatch.setattr(job, "_replay_strategies", lambda: (["STR-THRU"], 10))

    cases = {
        "recipe": ("run_training_job", {"recipe": "size:*:champion"}),
        "state": ("run_state_job", {"state": "driver_residual_pool:size"}),
        "board_analog": ("run_board_analog_job", {"alpha": 0.5, "cutoffs": ["2026-09-18"]}),
        "trailing_cutoff": ("run_trailing_cutoff_job", {"cutoffs": ["2026-01-05"]}),
    }
    base = {"expected_ids": ["training"], "recipe": "", "state": "", "alpha": None,
            "cutoffs": [], "strategies": [], "pairs_path": "", "ticker_chunk": 1000}
    for mode, (expected, extra) in cases.items():
        calls.clear()
        training.run_training_worker({**base, "mode": mode, **extra}, tmp_path)
        assert calls == [(expected, False)], mode
    assert (tmp_path / "training_result.json").is_file()


def test_board_analog_mode_refuses_over_budget(tmp_path, monkeypatch):
    from tools import phase5_training_job as job

    monkeypatch.setattr(job, "_replay_strategies", lambda: (["STR-THRU"], 10_000_000))

    def reached(*args, **kwargs):
        raise AssertionError("run_board_analog_job was reached")

    monkeypatch.setattr(job, "run_board_analog_job", reached)
    params = {"expected_ids": ["training"], "mode": "board_analog", "alpha": 0.5,
              "cutoffs": ["2026-09-18"], "strategies": [], "pairs_path": "",
              "ticker_chunk": 1000}
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(params, tmp_path)
    assert excinfo.value.code == "RESOURCE_LIMIT_EXCEEDED"


def _legacy_manifest_ref(tmp_path, conn, clock):
    """Publish and register a tiny real ``legacy_input_manifest.v1.0`` naming
    one real file under ``tmp_path`` (the Service's ``store_root``): the
    training kind's launch-time read-set pin resolves and stages it exactly
    like a nightly stage's own ``legacy_manifest.json`` binding."""
    source = tmp_path / "data" / "features" / "panel.parquet"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"panel")
    document = to_document(LegacyInputManifest(
        manifest_id="training-manifest",
        file_refs=(LegacyFileRef(path="data/features/panel.parquet",
                                 content_hash="sha256:" + hashlib.sha256(b"panel").hexdigest(),
                                 byte_size=len(b"panel")),),
        table_contract_refs=(), registry_and_model_refs=(), calendar_ref=None,
        selected_session="2026-09-12", finality_receipt_refs=(),
        knowledge_mode_by_table={}, availability_evidence_refs=(),
        read_set_complete=True, capture_implementation_ref="test.v1"))
    store = ArtifactStore(tmp_path)
    ref = store.publish_bytes(json.dumps(document, sort_keys=True).encode(),
                              schema_ref="legacy_input_manifest.v1.0")
    with transaction(conn):
        register_artifact(conn, ref, None, clock)
    return ref


#: A synthetic nightly capture that passes ``legacy_nightly_read_plan.
#: manifest_problems`` (one file for every required family) -- only the CLI's
#: plan-time guard reads it; no launch ever stages these bytes.
_NIGHTLY_CAPTURE = {
    "schema_version": "legacy_input_manifest.v1.0", "manifest_id": "m-training",
    "capture_implementation_ref": "legacy_nightly_capture.v1",
    "calendar_ref": "calendar://legacy", "registry_and_model_refs": ["registry://legacy"],
    "selected_session": "2026-02-01", "read_set_complete": True,
    "table_contract_refs": [], "finality_receipt_refs": [],
    "knowledge_mode_by_table": {}, "availability_evidence_refs": [],
    "file_refs": [
        {"path": "data/curated/daily_market/year=2024/part-0000.parquet",
         "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
        {"path": "data/curated/option_chains/year=2024/part-0000.parquet",
         "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
        {"path": "data/curated/earnings_events/year=2024/part-0000.parquet",
         "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
        {"path": "data/raw/fetch/orats/hist/summaries/2024-01-02.json",
         "content_hash": "sha256:" + "0" * 64, "byte_size": 1},
    ],
}


def _install_tiny_state_worker(monkeypatch):
    """Swap the worker subprocess's argv for a stub that binds a tiny
    ``run_state_job`` (the seam ``test_v2_ops_native_refresh_e2e.py`` uses)."""
    stub = (
        "import sys, types\n"
        "from engine.v2.ops import worker\n"
        "tools = types.ModuleType('tools')\n"
        "job = types.ModuleType('tools.phase5_training_job')\n"
        "def tiny_state(member, out, *, plan_only=False, **kwargs):\n"
        "    out.mkdir(parents=True, exist_ok=True)\n"
        "    (out / 'state.json').write_text('{\"state\": \"' + member + '\"}')\n"
        "    return {'state': member, 'plan_only': plan_only, 'status': 'written'}\n"
        "job.run_state_job = tiny_state\n"
        "tools.phase5_training_job = job\n"
        "sys.modules['tools'] = tools\n"
        "sys.modules['tools.phase5_training_job'] = job\n"
        "raise SystemExit(worker.main())\n"
    )
    real = subprocess.Popen

    def popen(args, **kwargs):
        if list(args[-2:]) == ["-m", "engine.v2.ops.worker"]:
            args = [sys.executable, "-u", "-c", stub]
        return real(args, **kwargs)

    monkeypatch.setattr(executor.subprocess, "Popen", popen)


def test_e2e_submit_training_through_supervisor(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    store = ArtifactStore(tmp_path)
    try:
        _install_tiny_state_worker(monkeypatch)
        # Real path end to end: the plan's pinned ``legacy_manifest.json``
        # binding drives the real launch-time ``_pin_read_set`` and the
        # staging copy, exactly like a nightly stage.
        manifest = _legacy_manifest_ref(tmp_path, conn, clock)
        plan = training.training_plan(mode="state", state="driver_residual_pool:size",
                                      manifest_ref=manifest.artifact_id)
        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "training-e2e"), clock=clock)
        service = Service(conn, tmp_path, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=tmp_path)
        try:
            service.start()
            state = run_until(service, conn, receipt.job_id, timeout=90)
            if state != "succeeded":
                failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id = ?",
                                       (receipt.job_id,)).fetchone()[0]
                attempt = conn.execute(
                    "SELECT attempt_id FROM attempts WHERE job_id = ? "
                    "ORDER BY attempt_number DESC LIMIT 1", (receipt.job_id,)).fetchone()
                diagnostic = service.store.staging_dir(attempt["attempt_id"]) / "diagnostics"
                stderr = (diagnostic / "worker.stderr")
                raise AssertionError(f"training attempt did not succeed: {failure}\n"
                                     f"{stderr.read_text() if stderr.is_file() else '(no stderr)'}")
        finally:
            service.close()
        rows = conn.execute(
            "SELECT ao.name, ao.artifact_id FROM attempt_outputs ao JOIN attempts a "
            "ON a.attempt_id = ao.attempt_id WHERE a.job_id = ?", (receipt.job_id,)).fetchall()
        names = sorted(item["name"] for item in rows)
        assert "training_result" in names
        assert "training/state.json" in names
        result_id = next(item["artifact_id"] for item in rows
                         if item["name"] == "training_result")
        document = json.loads(store.read_verified(artifact(conn, store, result_id)))
        assert document["state"] == "driver_residual_pool:size"
    finally:
        conn.close()


def test_training_plan_without_read_set_binding_is_refused_at_launch(tmp_path, monkeypatch):
    """Negative control: the plan's binding is what makes the launch pass. Drop
    it and the real ``_pin_read_set`` refuses, before a worker is ever
    launched."""
    conn, clock, _ = catalog(tmp_path)
    try:
        _install_tiny_state_worker(monkeypatch)
        manifest = _legacy_manifest_ref(tmp_path, conn, clock)
        plan = training.training_plan(mode="state", state="driver_residual_pool:size",
                                      manifest_ref=manifest.artifact_id)
        plan["parameters"]["input_bindings"] = None
        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "training-no-read-set"), clock=clock)
        service = Service(conn, tmp_path, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=tmp_path)
        try:
            service.start()
            assert run_until(service, conn, receipt.job_id, timeout=90) == "failed"
        finally:
            service.close()
        failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id = ?",
                               (receipt.job_id,)).fetchone()[0]
        assert "INPUT_CHANGED" in failure
    finally:
        conn.close()


def test_cli_plan_and_submit_training_never_runs_training_inline(tmp_path, capsys, monkeypatch):
    from tools import phase5_training_job as job

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not run during plan or submit")

    monkeypatch.setattr(job, "run_training_job", forbidden)
    root = tmp_path / "ops"
    manifest_path = tmp_path / "nightly-capture.json"
    manifest_path.write_text(json.dumps(_NIGHTLY_CAPTURE))
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "training",
                     "--training-mode", "state",
                     "--state", "driver_residual_pool:size",
                     "--input-manifest", str(manifest_path)]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "cli-training-1"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["kind"] == "training"
    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        row = conn.execute("SELECT state FROM jobs WHERE job_id=?",
                           (receipt["job_id"],)).fetchone()
        assert row is not None
        assert row["state"] == "queued"
    finally:
        conn.close()


def test_promote_kind_is_registered_with_delivery_profile():
    kind = stages.registry().get("models_promote")
    assert kind.resource_classes == frozenset({"delivery"})
    assert kind.checkpoint_contract == "promote_pointer_state.v1.0"
    assert kind.store_domains == (("deployment_pointer", "write"),)
    assert kind.retry.max_attempts == 1


def test_promote_worker_refuses_when_unstaged(tmp_path):
    with pytest.raises(OpsError) as excinfo:
        training.run_promote_worker(
            {"expected_ids": ["models_promote"], "release_root": str(tmp_path),
             "release_id": "ghost"}, tmp_path / "staging")
    assert excinfo.value.code == "VALIDATION_FAILED"
    assert excinfo.value.problem.message == "promote refused"
    assert "ghost" not in excinfo.value.problem.message
    assert excinfo.value.problem.details == {"exception_class": "ReleaseNotStaged"}


def test_promote_worker_promotes_a_staged_release(tmp_path):
    release, inventory, payloads = _release_fixture("r1")
    stage_release(tmp_path, release, inventory, payloads)
    result = training.run_promote_worker(
        {"expected_ids": ["models_promote"], "release_root": str(tmp_path),
         "release_id": "r1"}, tmp_path)
    assert result["completed_ids"] == ["models_promote"]
    assert result["outputs"] == [{"name": "pointer_state", "path": "pointer_state.json",
                                  "schema": "promote_pointer_state.v1.0"}]
    document = json.loads((tmp_path / "pointer_state.json").read_text())
    assert document["release_id"] == "r1"
    assert current_release(tmp_path).release_id == "r1"


def test_training_worker_never_reaches_promote(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("training must never call deployment.promote")

    monkeypatch.setattr("engine.v2.models.deployment.promote", forbidden)
    params = {"expected_ids": ["training"], "mode": "recipe", "recipe": "nope:*:champion",
              "state": "", "alpha": None, "cutoffs": [], "strategies": [],
              "pairs_path": "", "ticker_chunk": 1000}
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(params, tmp_path)
    assert excinfo.value.code == "VALIDATION_FAILED"


@pytest.mark.parametrize("error,code", [
    (SystemExit("RESUME_MISMATCH: state.json exists with different content"),
     "CHECKPOINT_INCOMPATIBLE"),
    (SystemExit("tier4_forecasts is missing; lineage cannot be attached"),
     "FEATURES_MISSING"),
    (SystemExit("size:*:champion: no dataset builder for this recipe"),
     "INVALID_REQUEST"),
])
def test_training_worker_types_tool_system_exits(tmp_path, monkeypatch, error, code):
    """One typed code per ``SystemExit`` cause the tool raises -- never an
    untyped WORKER_FAILED."""
    from tools import phase5_training_job as job

    def refuse(*args, **kwargs):
        raise error

    monkeypatch.setattr(job, "run_state_job", refuse)
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(), tmp_path)
    assert excinfo.value.code == code


def test_training_worker_types_training_refused(tmp_path, monkeypatch):
    from tools import phase5_training_job as job

    def refuse(*args, **kwargs):
        raise TrainingRefused((ReceiptIssue("$.job", "RESUME_MISMATCH", "detail"),))

    monkeypatch.setattr(job, "run_state_job", refuse)
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(), tmp_path)
    assert excinfo.value.code == "CHECKPOINT_INCOMPATIBLE"
    assert excinfo.value.problem.details == {"issues": ["RESUME_MISMATCH"]}


def test_training_worker_types_runtime_fit_forbidden(tmp_path, monkeypatch):
    from engine.v2.models.no_fit import RuntimeFitForbidden
    from tools import phase5_training_job as job

    def refuse(*args, **kwargs):
        raise RuntimeFitForbidden("no-fit guard active: test")

    monkeypatch.setattr(job, "run_state_job", refuse)
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(_worker_params(), tmp_path)
    assert excinfo.value.code == "VALIDATION_FAILED"


def test_run_recipe_guards_before_building_the_dataset(tmp_path, monkeypatch):
    """The recipe path calls the same ``_guard()`` ``main()`` calls, before any
    dataset builder runs; a tripped guard is typed, not WORKER_FAILED."""
    from engine.v2.models.no_fit import RuntimeFitForbidden
    from tools import phase5_training_job as job

    key = next(iter(current_recipes()))
    monkeypatch.setattr(job, "current_recipes", lambda: {key: current_recipes()[key]})
    monkeypatch.setattr(job, "_guard",
                        lambda where: (_ for _ in ()).throw(RuntimeFitForbidden(where)))
    monkeypatch.setattr(job, "build_dataset",
                        lambda *args, **kwargs: pytest.fail("build_dataset was reached"))
    with pytest.raises(OpsError) as excinfo:
        training.run_training_worker(
            _worker_params(mode="recipe", state="", recipe=key.label()), tmp_path)
    assert excinfo.value.code == "VALIDATION_FAILED"


def test_promote_plan_stores_release_root_absolute():
    """A relative ``--release-root`` promotes where the operator meant: the
    worker's cwd is its code snapshot, never the planner's cwd."""
    plan = training.promote_plan(release_root="relative/releases", release_id="r1")
    assert plan["parameters"]["release_root"] == str(Path("relative/releases").resolve())
    assert Path(plan["parameters"]["release_root"]).is_absolute()


def test_promote_never_submitted_by_nightly():
    """Structural, never a source grep: build the nightly plan with the real
    plan builder, build the actual job DAG it submits (every stage, with its
    ``_DAG_PARENTS`` prerequisites wired), and assert no job kind is
    ``models_promote``. The only way one is ever created is an operator's own
    ``plan promote``/``submit``."""
    plan = nightly_plan(REPO, "2026-09-12", tickers=("AAPL",), context_tickers=("AAPL",))
    requests = nightly.build_legacy_job_requests(
        plan, tickers=("AAPL",), context_tickers=("AAPL",), year_start=2024, year_end=2026)
    kinds = {request.job.kind for request in requests}
    assert "legacy_score" in kinds  # the graph is the real one, not empty
    assert len(kinds) > 1
    assert "models_promote" not in kinds


def test_e2e_submit_training_then_promote_through_supervisor(tmp_path, monkeypatch):
    conn, clock, _ = catalog(tmp_path)
    try:
        _install_tiny_state_worker(monkeypatch)
        # Same real read-set path as the training e2e above.
        manifest = _legacy_manifest_ref(tmp_path, conn, clock)
        plan = training.training_plan(mode="state", state="driver_residual_pool:size",
                                      manifest_ref=manifest.artifact_id)
        receipt = submit(conn, stages.registry(), POLICY,
                         request_from_plan(plan, "training-then-promote"), clock=clock)
        service = Service(conn, tmp_path, stages.registry(), TEST_POLICY, clock=clock,
                          code_source=REPO, store_root=tmp_path)
        try:
            service.start()
            assert run_until(service, conn, receipt.job_id, timeout=90) == "succeeded"

            # The registered candidate is staged by the TEST, not produced by
            # the training job (assembling one from real training output is
            # phase5_prepare_release.py's job, out of scope this slice).
            release, inventory, payloads = _release_fixture("r-e2e")
            stage_release(tmp_path, release, inventory, payloads)

            promote_plan = training.promote_plan(release_root=str(tmp_path),
                                                 release_id="r-e2e")
            promote_receipt = submit(conn, stages.registry(), POLICY,
                                     request_from_plan(promote_plan, "promote-e2e"), clock=clock)
            state = run_until(service, conn, promote_receipt.job_id, timeout=90)
            if state != "succeeded":
                failure = conn.execute("SELECT failure_json FROM jobs WHERE job_id = ?",
                                       (promote_receipt.job_id,)).fetchone()[0]
                attempt = conn.execute(
                    "SELECT attempt_id FROM attempts WHERE job_id = ? "
                    "ORDER BY attempt_number DESC LIMIT 1", (promote_receipt.job_id,)).fetchone()
                diagnostic = service.store.staging_dir(attempt["attempt_id"]) / "diagnostics"
                stderr = (diagnostic / "worker.stderr")
                raise AssertionError(f"promote attempt did not succeed: {failure}\n"
                                     f"{stderr.read_text() if stderr.is_file() else '(no stderr)'}")
        finally:
            service.close()
        assert current_release(tmp_path).release_id == "r-e2e"
    finally:
        conn.close()


def test_cli_plan_and_submit_promote_never_calls_deployment_inline(tmp_path, capsys, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("promote must not run during plan or submit")

    monkeypatch.setattr("engine.v2.models.deployment.promote", forbidden)
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    assert cli.main(["--root", str(root), "plan", "promote",
                     "--release-root", str(tmp_path), "--release-id", "r1"]) == 0
    plan_ref = json.loads(capsys.readouterr().out)["plan_ref"]
    assert cli.main(["--root", str(root), "submit", "--plan", plan_ref,
                     "--idempotency-key", "cli-promote-1"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["kind"] == "models_promote"
    clock = SystemClock()
    conn = open_catalog(root / "catalog.sqlite", clock=clock)
    try:
        row = conn.execute("SELECT state FROM jobs WHERE job_id=?",
                           (receipt["job_id"],)).fetchone()
        assert row is not None
        assert row["state"] == "queued"
    finally:
        conn.close()
