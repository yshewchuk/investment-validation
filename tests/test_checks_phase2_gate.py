"""Phase 2 gate: registry shape, synthetic finding-code isolation, real-repo smoke.

Phase-2 guide §12.1. Every world below is synthetic (tmp_path artifacts, a
git-init'd tmp root, hand-built junit-shaped test outcomes) except the final
smoke test, which runs the REAL Phase 2 suite over THIS repo to prove the
gate reports RED today for the rows that have no tests or real receipts yet,
and not for the ones that do. The prerequisite runner is always injected:
this file never lets the gate shell out to the real Phase 0/1 gates.

Task P2-C01 (Phase 2 review closeout): every evidence artifact below is a
REAL contract document (a real ``ComparisonReceipt`` payload, real
``SnapshotRef``/``SnapshotImportReceipt``/``DependencyPlan``/``RollbackReceipt``
documents), never a placeholder dict like ``{"verdict": "agree"}`` -- that
placeholder shape is exactly what the review reproduced as a false pass, and
is now its own negative test (``test_placeholder_agreement_json_gives_
artifact_shape_invalid``).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from checks import rearchitecture_phase2_coverage as p2cov
from checks import rearchitecture_phase2_evidence as p2evidence
from checks import rearchitecture_phase2_gate as p2gate
from checks.layer_map import PACKAGES
from checks.rearchitecture_phase2_coverage import REGISTRY as REAL_REGISTRY_PATH
from engine.v2.contracts import (
    DatasetVersionRef,
    DependencyEntry,
    DependencyPlan,
    FragmentRef,
    ObjectRef,
    RollbackReceipt,
    SnapshotImportReceipt,
    SnapshotRef,
    TableContractRef,
)
from engine.v2.diagnosis import AGREE, DIFFER, ComparisonReceipt, Envelope, Population
from engine.v2.foundation import to_document

ALL_D_IDS = [f"D{i:02d}" for i in range(1, 21)]
H = "sha256:" + "0" * 64


def _phase1_raw(*, imports_ok=True, coverage_ok=True):
    """The shape ``rearchitecture_phase1_gate.py::gate`` really returns,
    ``ok`` aggregated the same way it aggregates it: ``all()`` over every
    structural AND engineering row. Building it this way (rather than hand-
    setting a top-level ``"ok"``) makes it impossible for a test to claim a
    green Phase 1 while quietly leaving one row red.
    """
    structural = {"imports": {"ok": imports_ok}, "readmes": {"ok": True}, "hygiene": {"ok": True}}
    engineering = {"budgets": {"ok": True}, "lint": {"ok": True}, "hook": {"ok": True},
                   "coverage": {"ok": coverage_ok}}
    ok = all(r["ok"] for r in [*structural.values(), *engineering.values()])
    return {"ok": ok, "structural": structural, "engineering": engineering}


GREEN_RUNNER = lambda root: {  # noqa: E731
    "phase0": {"ok": True, "raw": {"ok": True}},
    "phase1": {"ok": True, "raw": _phase1_raw()},
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


def _dumps(obj) -> bytes:
    return json.dumps(to_document(obj)).encode()


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
        "mode": "serial",
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


# --------------------------------------------------------------------------
# real contract document builders -- task P2-C01
# --------------------------------------------------------------------------


def _snapshot_ref(snapshot_id="snap_current", manifest_hash=H) -> SnapshotRef:
    tcr = TableContractRef(contract_id="tc_securities", definition_hash=H)
    dvr = DatasetVersionRef(dataset_version_id="dv_1", table_contract_ref=tcr, manifest_hash=H)
    return SnapshotRef(
        snapshot_id=snapshot_id, manifest_hash=manifest_hash, table_versions={"securities": dvr},
        calendar_version="cal.v1", source_priority_version="prio.v1",
        finality_receipt_refs=("fin_1",), knowledge_mode_by_table={"securities": "reconstructed"})


def _dependency_plan(snapshot_ref: SnapshotRef) -> DependencyPlan:
    dvr = next(iter(snapshot_ref.table_versions.values()))
    frag_ref = FragmentRef(fragment_id="frag_1", manifest_hash=H)
    de = DependencyEntry(table_name="securities", dataset_version_ref=dvr, fragment_ref=frag_ref,
                         columns=("ticker",), predicates=(), estimated_rows=10, maximum_rows=100)
    return DependencyPlan(request_hash=H, snapshot_ref=snapshot_ref, dependencies=(de,))


def _import_receipt(snapshot_id, *, receipt_id, generation, status="committed") -> SnapshotImportReceipt:
    obj = ObjectRef(kind="parquet_fragment", object_id="obj_1", content_hash=H, byte_size=10)
    return SnapshotImportReceipt(
        receipt_id=receipt_id, request_hash=H, attempt_id="att_1", fence=1,
        snapshot_ref=_snapshot_ref(snapshot_id), legacy_snapshot_object_ref=obj,
        prior_head_snapshot_id=None, resulting_head_snapshot_id=snapshot_id,
        resulting_head_generation=generation, status=status, problem=None, envelope={})


def _explained_population(expected=100, supported=80, compared=50) -> Population:
    dropped_1 = [{"key": f"k{i}", "reason": "not_supported", "stage": "expected_to_supported"}
                for i in range(expected - supported)]
    dropped_2 = [{"key": f"j{i}", "reason": "tolerance_excluded", "stage": "supported_to_compared"}
                for i in range(supported - compared)]
    return Population(expected=expected, supported=supported, compared=compared,
                      excluded=tuple(dropped_1 + dropped_2))


def _comparison_receipt(*, kind, code_hash, environment_hash, snapshot_ref, verdict=AGREE,
                        population=None, receipt_id="recv_cmp", diagnostic_ref=None) -> ComparisonReceipt:
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash,
                        snapshot_id=snapshot_ref.snapshot_id,
                        snapshot_manifest_hash=snapshot_ref.manifest_hash,
                        diagnostic_ref=diagnostic_ref)
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=kind, tier=0, left_ref="legacy", right_ref="adapter",
        stage_plan_ref="plan.v1", tolerance_policy_ref="tol.v1", verdict=verdict,
        population=population or _explained_population(), envelope=envelope)


#: D14 review: the corpus receipt's ``diagnostic_ref`` must bind a real,
#: matching ``corpus_snapshot_binding.v1.0`` artifact. ``valid_evidence``
#: below builds a tiny synthetic corpus dir at ``tmp_path / "corpus"`` (the
#: bare/unversioned ``INDEX.json``-at-root layout -- ``corpus_version: ""``)
#: whenever it includes a corpus receipt; ``_gate_with_evidence`` derives the
#: SAME path from ``artifacts_dir.parent`` (== the same ``tmp_path``) so
#: every existing caller picks it up with no signature change.
_CORPUS_SNAPSHOT = "sha256:" + "c" * 64


def _write_synthetic_corpus_index(tmp_path) -> Path:
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir(exist_ok=True)
    (corpus_root / "INDEX.json").write_text(json.dumps({"snapshot": _CORPUS_SNAPSHOT}))
    return corpus_root


def _corpus_binding_ref(artifacts_dir: Path, *, control=False) -> dict:
    binding = {
        "schema_version": p2evidence.CORPUS_SNAPSHOT_BINDING_V1, "corpus_version": "",
        "corpus_snapshot_hash": _CORPUS_SNAPSHOT, "source_snapshot_hash": _CORPUS_SNAPSHOT,
        "control": control, "control_drop_ticker": None,
    }
    return _ref(artifacts_dir, "corpus_binding.json", json.dumps(binding, sort_keys=True).encode())


def _rollback_receipt(*, prior_id, resulting_id, prior_generation=2, resulting_generation=3
                      ) -> RollbackReceipt:
    return RollbackReceipt(receipt_id="recv_rollback", scope="legacy_shadow",
                           prior_snapshot_id=prior_id, resulting_snapshot_id=resulting_id,
                           prior_generation=prior_generation, resulting_generation=resulting_generation,
                           at="2026-01-01T00:00:00.000000Z")


def _fault_matrix(*, missing=()) -> list:
    return [{"point": p, "outcome": "old_head", "verified_objects": True}
            for p in p2evidence.FAULT_POINTS if p not in missing]


def valid_evidence(tmp_path, root, *, populations=(100, 80, 50), authority_mode="shadow",
                   verdict=AGREE, render_verdict=AGREE, include_render_receipt=True,
                   include_corpus_receipt=True):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    env_hash, _ = p2gate.environment_hash(root)
    code_hash = p2gate.source_hash(p2gate.source_files(root))
    snapshot_ref = _snapshot_ref("snap_current")
    prior = _import_receipt("snap_prior", receipt_id="recv_import_prior", generation=1)
    current = _import_receipt("snap_current", receipt_id="recv_import_current", generation=2)
    population = _explained_population(*populations)
    score = _comparison_receipt(kind=p2evidence.SCORE_PARITY_KIND, code_hash=code_hash,
                                environment_hash=env_hash, snapshot_ref=snapshot_ref,
                                verdict=verdict, population=population, receipt_id="recv_score")
    rollback = _rollback_receipt(prior_id="snap_current", resulting_id="snap_prior")

    evidence = {
        "schema_version": "phase2_evidence.v1.0",
        "code_hash": code_hash,
        "environment_hash": env_hash,
        "snapshot_ref": _ref(artifacts_dir, "snapshot.json", _dumps(snapshot_ref)),
        "legacy_snapshot_object_ref": _ref(artifacts_dir, "legacy.json", _dumps(
            ObjectRef(kind="legacy_snapshot", object_id="obj_legacy", content_hash=H, byte_size=99))),
        "table_contract_mapping_hash": "sha256:" + hashlib.sha256(b"mapping").hexdigest(),
        "import_receipt_refs": [_ref(artifacts_dir, "import_prior.json", _dumps(prior)),
                                _ref(artifacts_dir, "import_current.json", _dumps(current))],
        "fault_matrix_ref": _ref(artifacts_dir, "fault_matrix.json",
                                 json.dumps(_fault_matrix()).encode()),
        "dependency_plan_refs": [_ref(artifacts_dir, "dep1.json", _dumps(_dependency_plan(snapshot_ref)))],
        "comparison_receipt_ref": _ref(artifacts_dir, "comparison.json", _dumps(score)),
        "rollback_receipt_ref": _ref(artifacts_dir, "rollback.json", _dumps(rollback)),
        "expected_population": populations[0], "supported_population": populations[1],
        "compared_population": populations[2], "authority_mode": authority_mode,
    }
    if include_render_receipt:
        render = _comparison_receipt(kind=p2evidence.RENDER_PARITY_KIND, code_hash=code_hash,
                                     environment_hash=env_hash, snapshot_ref=snapshot_ref,
                                     verdict=render_verdict, receipt_id="recv_render")
        evidence["render_comparison_receipt_ref"] = _ref(
            artifacts_dir, "render_comparison.json", _dumps(render))
    if include_corpus_receipt:
        _write_synthetic_corpus_index(tmp_path)
        binding_ref = _corpus_binding_ref(artifacts_dir)
        corpus = _comparison_receipt(kind=p2evidence.CORPUS_PARITY_KIND, code_hash=code_hash,
                                     environment_hash=env_hash, snapshot_ref=snapshot_ref,
                                     receipt_id="recv_corpus",
                                     diagnostic_ref=json.dumps(binding_ref, sort_keys=True))
        evidence["corpus_comparison_receipt_ref"] = _ref(
            artifacts_dir, "corpus_comparison.json", _dumps(corpus))
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

_FULL_REGISTRY = {
    "D01": {"tier": 0, "tests": ["tests/d01.py"]},
    "D14": {"tier": 1, "tests": ["tests/d14.py"],
           "evidence_fields": ["corpus_comparison_receipt_ref"]},
    "D17": {"tier": 0, "tests": ["tests/d17.py"], "reuse_phase1_structural_engineering": True},
    "D15": {"tier": 2, "tests": [], "evidence_fields": [
        "comparison_receipt_ref", "expected_population",
        "supported_population", "compared_population"]},
    "D19": {"tier": 2, "tests": [], "evidence_fields": ["render_comparison_receipt_ref"]},
}


def test_fully_valid_evidence_and_passing_junit_and_green_prerequisites_is_ok(tmp_path):
    w = world(tmp_path, _FULL_REGISTRY)
    evidence, artifacts_dir = valid_evidence(tmp_path, w["root"])
    evidence_path = tmp_path / "evidence_final.json"
    evidence_path.write_text(json.dumps(evidence))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         evidence_manifest_path=evidence_path,
                         artifact_root=artifacts_dir, corpus_root=artifacts_dir.parent / "corpus",
                         prerequisite_runner=GREEN_RUNNER,
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


def test_phase1_structural_failure_gives_prerequisite_failed(tmp_path):
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)
    runner = lambda root: {  # noqa: E731
        "phase0": {"ok": True, "raw": {"ok": True}},
        "phase1": {"ok": False, "raw": _phase1_raw(imports_ok=False)}}
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=runner, registry_path=w["registry_path"])
    assert codes(result) == {"PREREQUISITE_FAILED"}
    assert {f["prerequisite"] for f in result["findings"]} == {"phase1"}


def test_phase1_passing_except_coverage_gives_prerequisite_failed(tmp_path):
    """Task P2-C01 decision 6: the outer gate enforces the FULL Phase 1
    prerequisite, coverage included -- a Phase 1 gate that is green on every
    structural/engineering row EXCEPT coverage must still fail this
    prerequisite, not pass it the way the removed exclusion helper let it."""
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)
    runner = lambda root: {  # noqa: E731
        "phase0": {"ok": True, "raw": {"ok": True}},
        "phase1": {"ok": False, "raw": _phase1_raw(coverage_ok=False)}}
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


def test_parallel_mode_baseline_gives_baseline_not_serial_only(tmp_path):
    """A baseline refreshed from a --parallel run must never become the
    truth a later serial measurement is judged against (phase-2 guide
    followup, 2026-09-13): it can bake in executor_watchdog.py's real
    /proc-race noise as if it belonged there."""
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)
    baseline_path = w["root"] / "checks/rearchitecture_phase2_coverage_baseline.json"
    baseline = json.loads(baseline_path.read_text())
    baseline["mode"] = "parallel"
    baseline_path.write_text(json.dumps(baseline))
    # The baseline file lives inside w["root"], so rewriting its bytes shifts
    # the whole-tree source_hash the fixed measurement was stamped with --
    # restamp it to the post-mutation tree so only BASELINE_NOT_SERIAL fires.
    measured = json.loads(w["coverage_path"].read_text())
    measured["source_hash"] = p2gate.source_hash(p2gate.source_files(w["root"]))
    w["coverage_path"].write_text(json.dumps(measured))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert codes(result) == {"BASELINE_NOT_SERIAL"}


def test_parallel_measurement_against_serial_baseline_has_no_mode_finding(tmp_path):
    """The refusal targets the BASELINE's mode, not the fresh measurement's:
    a --parallel measurement compared against an ordinary serial baseline
    stays allowed."""
    registry = {"D01": {"tier": 0, "tests": []}}
    w = world(tmp_path, registry)
    measured = json.loads(w["coverage_path"].read_text())
    measured["mode"] = "parallel"
    w["coverage_path"].write_text(json.dumps(measured))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert result["ok"] is True
    assert codes(result) == set()


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
                       corpus_root=artifacts_dir.parent / "corpus",
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
        snapshot_ref = _snapshot_ref("snap_current")
        differing = _comparison_receipt(
            kind=p2evidence.SCORE_PARITY_KIND, code_hash=evidence["code_hash"],
            environment_hash=evidence["environment_hash"], snapshot_ref=snapshot_ref,
            verdict=DIFFER, receipt_id="recv_score")
        evidence["comparison_receipt_ref"] = _ref(artifacts_dir, "comparison.json", _dumps(differing))
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
                         corpus_root=artifacts_dir.parent / "corpus",
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
                         corpus_root=artifacts_dir.parent / "corpus",
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
    evidence, artifacts_dir = valid_evidence(tmp_path, w["root"], render_verdict=DIFFER)
    evidence_path = tmp_path / "evidence_final.json"
    evidence_path.write_text(json.dumps(evidence))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         evidence_manifest_path=evidence_path, artifact_root=artifacts_dir,
                         corpus_root=artifacts_dir.parent / "corpus",
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert d_ids_with(result, "MISSING_EVIDENCE") == {"D19"}
    assert "VERDICT_NOT_AGREE" in codes(result)


def test_d14_refused_when_corpus_receipt_is_absent(tmp_path):
    """Task P2-C01 decision 7: D14 requires a real supervised corpus-scoring
    receipt, not just its two structural tests."""
    registry = {"D14": {"tier": 1, "tests": [],
                        "evidence_fields": ["corpus_comparison_receipt_ref"]}}
    w = world(tmp_path, registry)
    evidence, artifacts_dir = valid_evidence(tmp_path, w["root"], include_corpus_receipt=False)
    evidence_path = tmp_path / "evidence_final.json"
    evidence_path.write_text(json.dumps(evidence))
    result = p2gate.gate(root=w["root"], coverage_path=w["coverage_path"],
                         evidence_manifest_path=evidence_path, artifact_root=artifacts_dir,
                         corpus_root=artifacts_dir.parent / "corpus",
                         prerequisite_runner=GREEN_RUNNER, registry_path=w["registry_path"])
    assert d_ids_with(result, "MISSING_EVIDENCE") == {"D14"}


# -- task P2-C01: strict decode ------------------------------------------------

def test_placeholder_agreement_json_gives_artifact_shape_invalid(tmp_path):
    """The exact shape the review reproduced: a bare ``{"verdict": "agree"}``
    dict is not a ``ComparisonReceipt`` and must be refused, not tolerated."""
    def mutate(evidence, artifacts_dir):
        data = json.dumps({"verdict": "agree"}).encode()
        evidence["comparison_receipt_ref"] = _ref(artifacts_dir, "comparison.json", data)
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"ARTIFACT_SHAPE_INVALID"}


def test_missing_required_field_in_receipt_gives_artifact_shape_invalid(tmp_path):
    def mutate(evidence, artifacts_dir):
        doc = json.loads((artifacts_dir / "rollback.json").read_bytes())
        del doc["resulting_generation"]
        data = json.dumps(doc).encode()
        evidence["rollback_receipt_ref"] = _ref(artifacts_dir, "rollback.json", data)
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"ARTIFACT_SHAPE_INVALID"}


# -- task P2-C01: score vs render receipt separation ---------------------------

def test_same_artifact_for_score_and_render_gives_receipt_kind_mismatch(tmp_path):
    def mutate(evidence, artifacts_dir):
        evidence["render_comparison_receipt_ref"] = dict(evidence["comparison_receipt_ref"])
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"RECEIPT_KIND_MISMATCH"}


def test_swapped_comparison_kinds_gives_receipt_kind_mismatch(tmp_path):
    def mutate(evidence, artifacts_dir):
        snapshot_ref = _snapshot_ref("snap_current")
        score_as_render = _comparison_receipt(
            kind=p2evidence.SCORE_PARITY_KIND, code_hash=evidence["code_hash"],
            environment_hash=evidence["environment_hash"], snapshot_ref=snapshot_ref,
            receipt_id="recv_render_swapped")
        render_as_score = _comparison_receipt(
            kind=p2evidence.RENDER_PARITY_KIND, code_hash=evidence["code_hash"],
            environment_hash=evidence["environment_hash"], snapshot_ref=snapshot_ref,
            receipt_id="recv_score_swapped")
        evidence["render_comparison_receipt_ref"] = _ref(
            artifacts_dir, "render_swapped.json", _dumps(score_as_render))
        evidence["comparison_receipt_ref"] = _ref(
            artifacts_dir, "score_swapped.json", _dumps(render_as_score))
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"RECEIPT_KIND_MISMATCH"}


# -- task P2-C01: binding to code/environment/snapshot identity ---------------

def test_receipt_bound_to_wrong_code_hash_gives_code_hash_mismatch(tmp_path):
    def mutate(evidence, artifacts_dir):
        snapshot_ref = _snapshot_ref("snap_current")
        stale = _comparison_receipt(
            kind=p2evidence.SCORE_PARITY_KIND, code_hash="sha256:" + "9" * 64,
            environment_hash=evidence["environment_hash"], snapshot_ref=snapshot_ref)
        evidence["comparison_receipt_ref"] = _ref(artifacts_dir, "comparison.json", _dumps(stale))
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"CODE_HASH_MISMATCH"}


def test_receipt_bound_to_wrong_environment_hash_gives_environment_mismatch(tmp_path):
    def mutate(evidence, artifacts_dir):
        snapshot_ref = _snapshot_ref("snap_current")
        stale = _comparison_receipt(
            kind=p2evidence.SCORE_PARITY_KIND, code_hash=evidence["code_hash"],
            environment_hash="sha256:" + "8" * 64, snapshot_ref=snapshot_ref)
        evidence["comparison_receipt_ref"] = _ref(artifacts_dir, "comparison.json", _dumps(stale))
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"ENVIRONMENT_MISMATCH"}


def test_receipt_bound_to_wrong_snapshot_gives_snapshot_binding_mismatch(tmp_path):
    def mutate(evidence, artifacts_dir):
        wrong_snapshot = _snapshot_ref("snap_other")
        stale = _comparison_receipt(
            kind=p2evidence.SCORE_PARITY_KIND, code_hash=evidence["code_hash"],
            environment_hash=evidence["environment_hash"], snapshot_ref=wrong_snapshot)
        evidence["comparison_receipt_ref"] = _ref(artifacts_dir, "comparison.json", _dumps(stale))
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"SNAPSHOT_BINDING_MISMATCH"}


def test_import_receipt_not_committed_gives_snapshot_binding_mismatch(tmp_path):
    """Decision 2's last line is a conjunction -- committed AND names the
    snapshot. This isolates the "committed" half: the id is still
    "snap_current" (so rollback lineage, which only checks id membership,
    is untouched), but the receipt's own status is not "committed"."""
    def mutate(evidence, artifacts_dir):
        failed = _import_receipt("snap_current", receipt_id="recv_import_current",
                                 generation=2, status="failed")
        evidence["import_receipt_refs"][1] = _ref(artifacts_dir, "import_current.json", _dumps(failed))
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"SNAPSHOT_BINDING_MISMATCH"}


# -- task P2-C01: D15 population binding + explained drops --------------------

def test_receipt_population_mismatch_gives_population_collapsed(tmp_path):
    def mutate(evidence, artifacts_dir):
        evidence["compared_population"] = 49  # receipt's own population.compared is 50
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"POPULATION_COLLAPSED"}


def test_unexplained_population_drop_gives_population_unexplained(tmp_path):
    def mutate(evidence, artifacts_dir):
        snapshot_ref = _snapshot_ref("snap_current")
        unexplained = _comparison_receipt(
            kind=p2evidence.SCORE_PARITY_KIND, code_hash=evidence["code_hash"],
            environment_hash=evidence["environment_hash"], snapshot_ref=snapshot_ref,
            population=Population(expected=100, supported=80, compared=50))
        evidence["comparison_receipt_ref"] = _ref(
            artifacts_dir, "comparison.json", _dumps(unexplained))
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"POPULATION_UNEXPLAINED"}


def test_explained_population_drop_has_no_population_unexplained_finding(tmp_path):
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate=None)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert "POPULATION_UNEXPLAINED" not in codes(result)


# -- task P2-C01: rollback + fault matrix --------------------------------------

def test_rollback_receipt_with_non_increasing_generation_gives_rollback_evidence_invalid(tmp_path):
    def mutate(evidence, artifacts_dir):
        stale = _rollback_receipt(prior_id="snap_current", resulting_id="snap_prior",
                                  prior_generation=2, resulting_generation=2)
        evidence["rollback_receipt_ref"] = _ref(artifacts_dir, "rollback.json", _dumps(stale))
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"ROLLBACK_EVIDENCE_INVALID"}


def test_rollback_receipt_naming_an_unlisted_snapshot_gives_rollback_evidence_invalid(tmp_path):
    def mutate(evidence, artifacts_dir):
        stale = _rollback_receipt(prior_id="snap_current", resulting_id="snap_never_imported")
        evidence["rollback_receipt_ref"] = _ref(artifacts_dir, "rollback.json", _dumps(stale))
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"ROLLBACK_EVIDENCE_INVALID"}


def test_fault_matrix_missing_one_point_gives_fault_matrix_incomplete(tmp_path):
    def mutate(evidence, artifacts_dir):
        matrix = _fault_matrix(missing=("before_commit",))
        evidence["fault_matrix_ref"] = _ref(artifacts_dir, "fault_matrix.json",
                                            json.dumps(matrix).encode())
    w, evidence_path, artifacts_dir = _evidence_world(tmp_path, mutate)
    result = _gate_with_evidence(w, evidence_path, artifacts_dir)
    assert codes(result) == {"FAULT_MATRIX_INCOMPLETE"}


# -- real repo smoke ----------------------------------------------------------
#
# This used to run a full Phase 2 coverage measurement (the whole registered
# suite under `coverage run`, in a subprocess, from inside one unit test) --
# 48s, and pure overhead for what this test actually needs to prove: that the
# REAL acceptance registry matches the REAL tree (every D-row's test file
# exists and resolves to real collectible tests) and that rows lacking real
# evidence are correctly the ones still lacking a test file or a receipt.
#
# The full "fresh measurement -> gate is red for exactly the rows lacking
# evidence" check still exists and still runs against the real tree -- it now
# lives in the documented, runnable gate command (engine/v2/ops/README.md,
# Testing section), not inside the unit suite, since it needs a real
# `coverage run` over the fixed suite and that cost belongs to a deliberate
# invocation, not every test run. The gate-logic assertions above (synthetic
# worlds) are unchanged.
#
# `tests/test_checks_phase2_gate.py` itself is not named in any registry row's
# "tests" list (only tests/test_v2_ops_engineering.py is added implicitly by
# p2cov.suite()), so nothing here re-enters a suite measurement of this file.

def test_real_repo_smoke_matches_registry_against_the_real_tree():
    registry = p2cov.load_registry(REAL_REGISTRY_PATH)
    root = p2gate.ROOT

    all_test_paths = sorted({t for row in registry.values() for t in row.get("tests", [])})
    existing = [t for t in all_test_paths if (root / t).is_file()]
    missing_files = sorted(t for t in all_test_paths if not (root / t).is_file())
    # Task 6a (P2-6): D13 (synthetic) and the adapter half of D14 now have a
    # real test file (tests/test_v2_data_legacy_materialization.py), so no
    # registered test file is missing any more. Any file going missing is
    # registry/tree drift this test must catch, so the set is asserted
    # exactly, not just "non-empty is fine".
    assert missing_files == [], missing_files

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *existing],
        cwd=root, capture_output=True, text=True, timeout=60)
    assert collected.returncode == 0, collected.stdout + collected.stderr
    node_ids = [line for line in collected.stdout.splitlines() if "::" in line]

    def resolves(test_path):
        return any(n.startswith(test_path + "::") for n in node_ids)

    tier2_ids = {d_id for d_id, row in registry.items() if row.get("tier") == 2}
    assert tier2_ids == {"D15", "D16", "D19"}, tier2_ids

    for d_id, row in registry.items():
        for test_path in row.get("tests", []):
            assert resolves(test_path), (d_id, test_path)
        if d_id in tier2_ids:
            # D15 plus the other tier-2 rows (D16, D19) are evidence rows: a
            # passing test alone can never satisfy them, only a real receipt
            # can (checked end-to-end by the documented gate command above).
            assert row.get("evidence_fields"), d_id

    # Task P2-C01 decision 7: D14 (tier 1) also carries a required evidence
    # field now -- real supervised corpus-scoring evidence, not just its two
    # structural tests -- without being reclassified as tier 2.
    assert registry["D14"].get("evidence_fields") == ["corpus_comparison_receipt_ref"]
