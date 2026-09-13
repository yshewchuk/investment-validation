"""Phase 2 gate: registry shape, synthetic finding-code isolation, real-repo smoke.

Phase-2 guide §12.1. Every world below is synthetic (tmp_path artifacts, a
git-init'd tmp root, hand-built junit-shaped test outcomes) except the final
smoke test, which runs the REAL Phase 2 suite over THIS repo to prove the
gate reports RED today for the rows that have no tests or real receipts yet,
and not for the ones that do. The prerequisite runner is always injected:
this file never lets the gate shell out to the real Phase 0/1 gates.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from checks import rearchitecture_phase2_coverage as p2cov
from checks import rearchitecture_phase2_gate as p2gate
from checks.layer_map import PACKAGES
from checks.rearchitecture_phase2_coverage import REGISTRY as REAL_REGISTRY_PATH

ALL_D_IDS = [f"D{i:02d}" for i in range(1, 21)]
GREEN_RUNNER = lambda root: {  # noqa: E731
    "phase0": {"ok": True, "raw": {"ok": True}},
    "phase1": {"ok": True, "raw": {
        "structural": {"imports": {"ok": True}, "readmes": {"ok": True}, "hygiene": {"ok": True}},
        "engineering": {"budgets": {"ok": True}, "lint": {"ok": True}, "hook": {"ok": True},
                        "coverage": {"ok": False}},
    }},
}


def _git_root(tmp_path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def _ref(artifacts_dir: Path, rel: str, data: bytes) -> dict:
    path = artifacts_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": rel, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def _packages_doc(**overrides) -> dict:
    doc = {p.dotted: {"executed": 10, "executable": 10, "missing_files": [], "empty": False}
           for p in PACKAGES}
    for dotted, patch in overrides.items():
        doc[dotted].update(patch)
    return doc


def _write_baseline(root: Path, packages: dict, suite_version: str, test_files: list) -> None:
    baseline = root / "checks/rearchitecture_phase2_coverage_baseline.json"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(json.dumps({
        "schema_version": p2cov.SCHEMA_VERSION, "suite_version": suite_version,
        "test_files": test_files, "source_hash": "sha256:" + "0" * 64, "packages": packages,
    }))


def _outcomes(registry: dict) -> list:
    return [{"nodeid": f"{path}::test_it", "outcome": "passed"}
            for row in registry.values() for path in row.get("tests", [])]


def world(tmp_path, registry: dict, *, outcomes=None):
    """A fully self-consistent synthetic Phase 2 world: a git-init'd root with a
    committed-shaped baseline and a matching fresh measurement. Evidence, when
    a test needs it, is built afterwards by ``valid_evidence`` so its
    code_hash/environment_hash are computed against this root's FINAL tracked
    content -- nothing here is circular."""
    root = _git_root(tmp_path)
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"schema_version": "x", "rows": registry}))
    test_files = sorted({p for row in registry.values() for p in row.get("tests", [])})
    for rel in test_files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_it():\n    pass\n")
    suite_version = p2cov.suite_version(test_files)
    packages = _packages_doc()
    _write_baseline(root, packages, suite_version, test_files)
    code_hash = p2gate.source_hash(p2gate.source_files(root))
    measured = {"schema_version": p2cov.SCHEMA_VERSION, "suite_version": suite_version,
                "test_files": test_files, "source_hash": code_hash, "packages": packages,
                "test_outcomes": _outcomes(registry) if outcomes is None else outcomes}
    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(json.dumps(measured))
    return {"root": root, "registry_path": registry_path, "coverage_path": coverage_path,
            "code_hash": code_hash}


def valid_evidence(tmp_path, root, *, populations=(100, 80, 50), authority_mode="shadow",
                   verdict="agree", render_verdict="agree", include_render_receipt=True):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    env_hash, _ = p2gate.environment_hash(root)
    evidence = {
        "schema_version": "phase2_evidence.v1.0",
        "code_hash": p2gate.source_hash(p2gate.source_files(root)),
        "environment_hash": env_hash,
        "snapshot_ref": _ref(artifacts_dir, "snapshot.bin", b"snapshot-bytes"),
        "legacy_snapshot_object_ref": _ref(artifacts_dir, "legacy.bin", b"legacy-object-bytes"),
        "table_contract_mapping_hash": "sha256:" + hashlib.sha256(b"mapping").hexdigest(),
        "import_receipt_refs": [_ref(artifacts_dir, "import1.json", b"{}"),
                                _ref(artifacts_dir, "import2.json", b"{}")],
        "fault_matrix_ref": _ref(artifacts_dir, "fault_matrix.json", b"{}"),
        "dependency_plan_refs": [_ref(artifacts_dir, "dep1.json", b"{}")],
        "comparison_receipt_ref": _ref(artifacts_dir, "comparison.json",
                                       json.dumps({"verdict": verdict}).encode()),
        "rollback_receipt_ref": _ref(artifacts_dir, "rollback.json", b'{"ok": true}'),
        "expected_population": populations[0], "supported_population": populations[1],
        "compared_population": populations[2], "authority_mode": authority_mode,
    }
    if include_render_receipt:
        evidence["render_comparison_receipt_ref"] = _ref(
            artifacts_dir, "render_comparison.json",
            json.dumps({"verdict": render_verdict}).encode())
    return evidence, artifacts_dir


def codes(result) -> set:
    return {f["code"] for f in result["findings"]}


def d_ids_with(result, code) -> set:
    return {f["d_id"] for f in result["findings"] if f["code"] == code and "d_id" in f}


# -- registry shape -----------------------------------------------------------

def test_registry_covers_exactly_d01_through_d20():
    rows = p2cov.load_registry(REAL_REGISTRY_PATH)
    assert set(rows) == set(ALL_D_IDS)


# -- fully valid synthetic world: ok ------------------------------------------

def test_fully_valid_evidence_and_passing_junit_and_green_prerequisites_is_ok(tmp_path):
    registry = {
        "D01": {"tier": 0, "tests": ["tests/d01.py"]},
        "D17": {"tier": 0, "tests": ["tests/d17.py"], "reuse_phase1_structural_engineering": True},
        "D15": {"tier": 2, "tests": [], "evidence_fields": [
            "comparison_receipt_ref", "expected_population",
            "supported_population", "compared_population"]},
        "D19": {"tier": 2, "tests": [], "evidence_fields": ["render_comparison_receipt_ref"]},
    }
    w = world(tmp_path, registry)
    evidence, artifacts_dir = valid_evidence(tmp_path, w["root"])
    evidence_path = tmp_path / "evidence_final.json"
    evidence_path.write_text(json.dumps(evidence))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         evidence_manifest_path=evidence_path,
                         artifact_root=artifacts_dir, prerequisite_runner=GREEN_RUNNER,
                         registry_path=w["registry_path"])
    assert result["ok"] is True, result["findings"]
    assert result["findings"] == []


# -- D-ID scoped codes ---------------------------------------------------------

def test_missing_test_outcome_gives_missing_evidence_only(tmp_path):
    registry = {"D01": {"tier": 0, "tests": ["tests/d01.py"]}}
    w = world(tmp_path, registry, outcomes=[])
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert codes(result) == {"MISSING_EVIDENCE"}
    assert d_ids_with(result, "MISSING_EVIDENCE") == {"D01"}
    assert result["ok"] is False


def test_failing_test_outcome_gives_test_failed_only(tmp_path):
    registry = {"D01": {"tier": 0, "tests": ["tests/d01.py"]}}
    outcomes = [{"nodeid": "tests/d01.py::test_it", "outcome": "failed"}]
    w = world(tmp_path, registry, outcomes=outcomes)
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert codes(result) == {"TEST_FAILED"}
    assert d_ids_with(result, "TEST_FAILED") == {"D01"}


# -- prerequisites --------------------------------------------------------------

def test_phase0_prerequisite_failure_gives_prerequisite_failed_only(tmp_path):
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)
    runner = lambda root: {  # noqa: E731
        "phase0": {"ok": False, "raw": {"ok": False}}, "phase1": GREEN_RUNNER(root)["phase1"]}
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=runner, registry_path=w["registry_path"])
    assert codes(result) == {"PREREQUISITE_FAILED"}
    assert {f["prerequisite"] for f in result["findings"]} == {"phase0"}


def test_phase1_structural_failure_excluding_coverage_gives_prerequisite_failed(tmp_path):
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)

    def runner(root):
        raw = GREEN_RUNNER(root)
        raw["phase1"]["raw"]["structural"]["imports"] = {"ok": False}
        return raw
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=runner, registry_path=w["registry_path"])
    assert codes(result) == {"PREREQUISITE_FAILED"}
    assert {f["prerequisite"] for f in result["findings"]} == {"phase1"}


# -- coverage ratchet codes ------------------------------------------------------

def test_source_hash_drift_gives_stale_coverage_only(tmp_path):
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)
    measured = json.loads(w["coverage_path"].read_text())
    measured["source_hash"] = "sha256:" + "f" * 64
    w["coverage_path"].write_text(json.dumps(measured))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert codes(result) == {"STALE_COVERAGE"}


def test_package_regression_gives_coverage_regression_only(tmp_path):
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)
    measured = json.loads(w["coverage_path"].read_text())
    target = PACKAGES[0].dotted
    measured["packages"][target]["executed"] = 1
    w["coverage_path"].write_text(json.dumps(measured))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert codes(result) == {"COVERAGE_REGRESSION"}


def test_suite_version_drift_gives_suite_drift_only(tmp_path):
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)
    measured = json.loads(w["coverage_path"].read_text())
    measured["suite_version"] = "phase2_coverage_suite:different"
    w["coverage_path"].write_text(json.dumps(measured))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert codes(result) == {"SUITE_DRIFT"}


# -- evidence document codes ------------------------------------------------------

def _evidence_world(tmp_path, mutate=None):
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)
    evidence, artifacts_dir = valid_evidence(tmp_path, w["root"])
    if mutate:
        mutate(evidence, artifacts_dir)
    evidence_path = tmp_path / "evidence_final.json"
    evidence_path.write_text(json.dumps(evidence))
    return w, evidence_path, artifacts_dir


def _gate_with_evidence(w, evidence_path, artifacts_dir):
    return p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                       evidence_manifest_path=evidence_path, artifact_root=artifacts_dir,
                       prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])


def test_code_hash_mismatch_gives_code_hash_mismatch_only(tmp_path):
    def mutate(evidence, artifacts_dir):
        evidence["code_hash"] = "sha256:" + "a" * 64
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"CODE_HASH_MISMATCH"}


def test_missing_artifact_gives_artifact_missing_only(tmp_path):
    def mutate(evidence, artifacts_dir):
        (artifacts_dir / "fault_matrix.json").unlink()
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"ARTIFACT_MISSING"}


def test_corrupted_artifact_gives_artifact_hash_mismatch_only(tmp_path):
    def mutate(evidence, artifacts_dir):
        (artifacts_dir / "rollback.json").write_bytes(b'{"ok": false, "tamper": true}')
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"ARTIFACT_HASH_MISMATCH"}


def test_zero_population_gives_population_empty_only(tmp_path):
    def mutate(evidence, artifacts_dir):
        evidence["compared_population"] = 0
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"POPULATION_EMPTY"}


def test_out_of_order_population_gives_population_collapsed_only(tmp_path):
    def mutate(evidence, artifacts_dir):
        evidence["compared_population"], evidence["supported_population"] = 90, 50
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"POPULATION_COLLAPSED"}


def test_disagreeing_verdict_gives_verdict_not_agree_only(tmp_path):
    def mutate(evidence, artifacts_dir):
        data = json.dumps({"verdict": "differ"}).encode()
        (artifacts_dir / "comparison.json").write_bytes(data)
        evidence["comparison_receipt_ref"] = {
            "path": "comparison.json", "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"VERDICT_NOT_AGREE"}


def test_boolean_ref_gives_summary_boolean_refused_only(tmp_path):
    def mutate(evidence, artifacts_dir):
        evidence["rollback_receipt_ref"] = True
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"SUMMARY_BOOLEAN_REFUSED"}


def test_wrong_authority_mode_gives_authority_not_shadow_only(tmp_path):
    def mutate(evidence, artifacts_dir):
        evidence["authority_mode"] = "live"
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"AUTHORITY_NOT_SHADOW"}


# -- authority_mode: document-wide, so it invalidates EVERY tier-2 row --------

def test_wrong_authority_mode_invalidates_every_tier2_row(tmp_path):
    registry = {
        "D15": {"tier": 2, "tests": [], "evidence_fields": ["rollback_receipt_ref"]},
        "D19": {"tier": 2, "tests": [], "evidence_fields": ["render_comparison_receipt_ref"]},
    }
    w = world(tmp_path, registry)
    evidence, artifacts_dir = valid_evidence(tmp_path, w["root"], authority_mode="live")
    evidence_path = tmp_path / "evidence_final.json"
    evidence_path.write_text(json.dumps(evidence))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         evidence_manifest_path=evidence_path, artifact_root=artifacts_dir,
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert d_ids_with(result, "MISSING_EVIDENCE") == {"D15", "D19"}
    assert "AUTHORITY_NOT_SHADOW" in codes(result)
    assert "CODE_HASH_MISMATCH" not in codes(result)


# -- D15/D19 receipt separation: one row's evidence never proves the other ----

def test_d19_refused_when_only_the_score_receipt_is_present(tmp_path):
    registry = {
        "D15": {"tier": 2, "tests": [], "evidence_fields": [
            "comparison_receipt_ref", "expected_population",
            "supported_population", "compared_population"]},
        "D19": {"tier": 2, "tests": [], "evidence_fields": ["render_comparison_receipt_ref"]},
    }
    w = world(tmp_path, registry)
    evidence, artifacts_dir = valid_evidence(tmp_path, w["root"], include_render_receipt=False)
    evidence_path = tmp_path / "evidence_final.json"
    evidence_path.write_text(json.dumps(evidence))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         evidence_manifest_path=evidence_path, artifact_root=artifacts_dir,
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert d_ids_with(result, "MISSING_EVIDENCE") == {"D19"}


def test_d19_refused_when_render_receipt_verdict_is_not_agree(tmp_path):
    registry = {
        "D15": {"tier": 2, "tests": [], "evidence_fields": [
            "comparison_receipt_ref", "expected_population",
            "supported_population", "compared_population"]},
        "D19": {"tier": 2, "tests": [], "evidence_fields": ["render_comparison_receipt_ref"]},
    }
    w = world(tmp_path, registry)
    evidence, artifacts_dir = valid_evidence(tmp_path, w["root"], render_verdict="differ")
    evidence_path = tmp_path / "evidence_final.json"
    evidence_path.write_text(json.dumps(evidence))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         evidence_manifest_path=evidence_path, artifact_root=artifacts_dir,
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert d_ids_with(result, "MISSING_EVIDENCE") == {"D19"}
    assert "VERDICT_NOT_AGREE" in codes(result)


# -- real repo smoke ----------------------------------------------------------

def test_real_repo_smoke_is_red_for_rows_without_real_tests_or_receipts():
    measured = p2cov.measure()
    coverage_path = Path(measured["source_hash"].split(":")[1][:8] + "-phase2-smoke-coverage.json")
    coverage_path = Path("/tmp") / coverage_path.name
    coverage_path.write_text(json.dumps(measured))
    try:
        result = p2gate.gate(prerequisite_runner=GREEN_RUNNER, coverage_path=coverage_path)
    finally:
        coverage_path.unlink(missing_ok=True)
    assert result["ok"] is False
    missing = d_ids_with(result, "MISSING_EVIDENCE")
    for d_id in ("D05", "D06", "D07", "D13", "D14", "D15", "D16", "D20"):
        assert d_id in missing, (d_id, sorted(missing))
    for d_id in ("D01", "D02", "D03", "D08"):
        assert d_id not in missing, (d_id, sorted(missing))
