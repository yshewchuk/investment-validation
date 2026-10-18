"""Explicit D14 admission and CLI/worker wiring; no production data or fitting."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from engine.v2.data import errors
from engine.v2.foundation import to_document
from engine.v2.models import no_fit
from engine.v2.ops import cli, stages, training
from engine.v2.ops.errors import OpsError
from engine.v2.ops.store_barrier import domains_of
from engine.v2.ops.supervisor import Service
from engine.v2.ops.workflows import commands
from tests.v2.models.training.d14_support import forbidden
from tools import phase5_training_job as job

PINS = dict(input_mode="native_d14", catalog_path="/source/catalog.sqlite",
            objects_root="/source/store", snapshot_id="snapshot-1", receipt_id="receipt-1")


def params(**changes):
    return training.TrainingParameters(**{
        "expected_ids": ("training",), "mode": "recipe",
        "recipe": "implied_t1:*:tier4_monthly", **PINS, **changes})


INVALID = [
    *[{field: value} for field in ("catalog_path", "objects_root", "snapshot_id", "receipt_id")
      for value in ("", " ", "bad\x00pin")],
    {"catalog_path": "relative/catalog"}, {"objects_root": "relative/store"},
    {"input_mode": "legacy"}, {"input_mode": "snapshot"},
    {"mode": "state", "state": "paired_residual_pool"},
    {"recipe": "implied_t1:*:champion"}, {"recipe": "size:*:tier4_monthly"},
    {"recipe": "runup_move:STR-RUNUP:tier4_monthly"},
    {"alpha": 0.5}, {"cutoffs": ("2024-01-01",)}, {"state": "paired_residual_pool"},
    {"strategies": ("STR-RUNUP",)}, {"pairs_path": "pairs.parquet"}, {"input_bindings": {"legacy_manifest.json": "artifact"}},
]


@pytest.mark.parametrize("changes", INVALID)
def test_invalid_sources_refuse_before_reads(changes, monkeypatch, tmp_path):
    monkeypatch.setattr("engine.v2.models.training.build_d14_dataset", forbidden)
    monkeypatch.setattr(job, "build_dataset", forbidden)
    monkeypatch.setattr(job, "current_recipes", forbidden)
    parameters = params(**changes)
    assert training.training_parameter_problems(None, parameters)
    with pytest.raises(OpsError) as err:
        training.run_training_worker(to_document(parameters), tmp_path / "out")
    assert err.value.problem.code == "INVALID_REQUEST"
    assert not (tmp_path / "out").exists()


def test_native_plan_has_no_legacy_prerequisites_or_bindings(monkeypatch):
    monkeypatch.setattr("engine.v2.models.training.build_d14_dataset", forbidden)
    monkeypatch.setattr(job, "current_recipes", forbidden)
    plan = training.training_plan(mode="recipe", recipe="implied_t1:*:tier4_monthly", **PINS)
    assert plan["blocked_prerequisites"] == []
    assert plan["input_refs"] == []
    assert plan["parameters"]["input_bindings"] is None
    assert all(plan["parameters"][key] == value for key, value in PINS.items())
    with pytest.raises(OpsError) as err:
        training.training_plan(mode="recipe", recipe="implied_t1:*:tier4_monthly",
                               **PINS, manifest_ref="legacy-artifact")
    assert err.value.problem.code == "INVALID_REQUEST"


@pytest.mark.parametrize("field", tuple(PINS)[1:])
def test_every_pin_changes_plan_identity(field):
    first = training.training_plan(mode="recipe", recipe="implied_t1:*:tier4_monthly", **PINS)
    changes = {**PINS, field: PINS[field] + "-other"}
    second = training.training_plan(mode="recipe", recipe="implied_t1:*:tier4_monthly", **changes)
    assert first["spec_hash"] != second["spec_hash"]


@pytest.mark.parametrize("role", ("implied_t1", "runup_move"))
def test_worker_forwards_verified_identity_without_legacy_reads(role, monkeypatch, tmp_path):
    calls = []
    identity = {"schema_version": "d14.snapshot_dataset.v1", "receipt_id": "pinned"}
    dataset = SimpleNamespace(attrs={"training_inputs": identity})

    def build(actual_role, **kwargs):
        assert actual_role == role
        assert kwargs == dict(catalog=PINS["catalog_path"], artifact_root=PINS["objects_root"],
                              snapshot_id=PINS["snapshot_id"], receipt_id=PINS["receipt_id"])
        calls.append("build")
        return dataset

    def run(recipe, actual, out, **kwargs):
        assert recipe.key.role == role and recipe.key.output == "tier4_monthly"
        assert actual is dataset and out == tmp_path / "training"
        assert kwargs == {"plan_only": False, "input_identity": identity}
        calls.append("train")
        return SimpleNamespace(outcomes=())

    monkeypatch.setattr("engine.v2.models.training.build_d14_dataset", build)
    monkeypatch.setattr(job, "run_training_job", run)
    monkeypatch.setattr(job, "build_dataset", forbidden)
    result = training.run_training_worker(to_document(params(recipe=f"{role}:*:tier4_monthly")), tmp_path)
    assert calls == ["build", "train"]
    assert result["completed_ids"] == ["training"]
    assert json.loads((tmp_path / "training_result.json").read_text())["recipe_id"] == f"{role}.tier4_monthly.v1"


@pytest.mark.parametrize("native", (True, False))
def test_both_no_fit_guards_precede_native_reads(native, monkeypatch, tmp_path):
    from engine.models import no_fit as legacy_no_fit

    guard = no_fit if native else legacy_no_fit
    monkeypatch.setattr("engine.v2.models.training.build_d14_dataset", forbidden)
    monkeypatch.setattr(job, "current_recipes", forbidden)
    expected = OpsError if native else legacy_no_fit.RuntimeFitForbidden
    with guard.no_fit_guard(), pytest.raises(expected) as err:
        training.run_training_worker(to_document(params()), tmp_path)
    if native:
        assert err.value.problem.code == "VALIDATION_FAILED"
    assert not (tmp_path / "training").exists()


def test_source_refusal_retains_data_code(monkeypatch, tmp_path):
    def refuse(*args, **kwargs):
        raise errors.fail("IDENTITY_CONFLICT", "private source path")
    monkeypatch.setattr("engine.v2.models.training.build_d14_dataset", refuse)
    with pytest.raises(OpsError) as err:
        training.run_training_worker(to_document(params()), tmp_path)
    assert err.value.problem.code == "INPUT_CHANGED"
    assert err.value.problem.details == {"data_code": "IDENTITY_CONFLICT"}
    assert "private source path" not in str(err.value)
    assert not (tmp_path / "training_result.json").exists()


def test_lease_and_cache_exceptions_are_narrow():
    registry = stages.registry()
    native = to_document(params())
    assert domains_of(registry, "training", native) == ()
    assert domains_of(registry, "training", {}) == (("legacy_store", "read"),)
    assert domains_of(registry, "models_promote", native) == (("deployment_pointer", "write"),)
    service = object.__new__(Service)
    service.registry = registry
    claim = SimpleNamespace(spec=SimpleNamespace(kind="training", parameters=native))
    assert service._cache_allowed(claim) is False
    assert service._stage_legacy_inputs(claim, None) is False


@pytest.mark.parametrize("source_flags", (["--training-input-mode", "native_d14"],
                                          ["--training-snapshot-id", "snapshot-1"]))
def test_cli_rejects_mixed_manifest_before_read(monkeypatch, source_flags):
    monkeypatch.setattr(commands, "_read_input_manifest_ref", forbidden)
    args = cli.parser().parse_args(["plan", "training", *source_flags,
                                    "--input-manifest", "/missing"])
    with pytest.raises(OpsError) as err:
        commands._plan_command(args, None, None, None)
    assert err.value.problem.code == "INVALID_REQUEST"


def test_cli_preserves_all_pins_without_source_reads(monkeypatch, tmp_path):
    monkeypatch.setattr(commands, "_read_input_manifest_ref", forbidden)
    args = cli.parser().parse_args([
        "plan", "training", "--training-mode", "recipe", "--recipe", "runup_move:*:tier4_monthly",
        "--training-input-mode", "native_d14", "--training-catalog", "relative/catalog.sqlite",
        "--training-artifact-root", "relative/store", "--training-snapshot-id", "s1",
        "--training-receipt-id", "r1"])
    plan = commands._training_plan(args, tmp_path, None, None)
    actual = plan["parameters"]
    assert actual["catalog_path"].endswith("/relative/catalog.sqlite")
    assert actual["objects_root"].endswith("/relative/store")
    assert actual["snapshot_id"] == "s1" and actual["receipt_id"] == "r1"
    assert actual["recipe"] == "runup_move:*:tier4_monthly"


def test_cli_native_plan_and_submit_stay_queued_without_source_reads(monkeypatch, tmp_path, capsys):
    from engine.v2.foundation import ArtifactStore, SystemClock
    from engine.v2.ops.bootstrap import open_catalog
    from engine.v2.ops.checkpoints import artifact

    monkeypatch.setattr(job, "current_recipes", forbidden)
    monkeypatch.setattr("engine.v2.models.training.build_d14_dataset", forbidden)
    root = tmp_path / "ops"
    assert cli.main(["--root", str(root), "init"]) == 0
    capsys.readouterr()
    flags = ["--training-mode", "recipe", "--recipe", "implied_t1:*:tier4_monthly",
             "--training-input-mode", "native_d14", "--training-catalog", PINS["catalog_path"],
             "--training-artifact-root", PINS["objects_root"],
             "--training-snapshot-id", PINS["snapshot_id"], "--training-receipt-id", PINS["receipt_id"]]
    assert cli.main(["--root", str(root), "plan", "training", *flags]) == 0
    reference = json.loads(capsys.readouterr().out)["plan_ref"]
    assert cli.main(["--root", str(root), "submit", "--plan", reference,
                     "--idempotency-key", "native-cli"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    conn = open_catalog(root / "catalog.sqlite", clock=SystemClock())
    try:
        stored = json.loads(ArtifactStore(root).read_verified(artifact(conn, ArtifactStore(root), reference)))
        assert all(stored["parameters"][key] == value for key, value in PINS.items())
        assert conn.execute("SELECT state FROM jobs WHERE job_id=?", (receipt["job_id"],)).fetchone()[0] == "queued"
    finally:
        conn.close()
