"""Phase 3 gate: registry shape, synthetic finding-code isolation, real-repo smoke.

Phase-3 guide §10. Every world below is synthetic (tmp_path artifacts, a
git-init'd tmp root, hand-built real contract documents) except the final
smoke test, which runs the REAL gate over THIS repo with no evidence manifest
to prove it is RED today (guide: "the gate must be red today and green only
on complete, valid evidence").

Every evidence artifact is a REAL contract document (a real
``ComparisonReceipt``/``RollbackReceipt``/``PreviewInput``/``PreviewRelease``
payload, a real ``Phase2Evidence`` document with real nested contracts),
never a placeholder dict -- the same discipline
``tests/test_checks_phase2_gate.py`` established for Phase 2's own evidence,
reused here for the Phase 2 half (``valid_phase2_evidence`` below mirrors
that file's ``valid_evidence`` construction) and extended for Phase 3's own
fields.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from checks import rearchitecture_phase2_evidence as p2evidence
from checks import rearchitecture_phase2_gate as p2gate
from checks import rearchitecture_phase3_evidence as p3evidence
from checks import rearchitecture_phase3_gate as p3gate
from checks.rearchitecture_phase3_quality import FIXED_SUITE
from checks.rearchitecture_phase3_gate import REGISTRY as REAL_REGISTRY_PATH
from engine.v2.contracts import (
    DatasetVersionRef,
    DependencyEntry,
    DependencyPlan,
    FragmentRef,
    ObjectRef,
    PreviewInput,
    PreviewRelease,
    RollbackReceipt,
    SnapshotImportReceipt,
    SnapshotRef,
    TableContractRef,
)
from engine.v2.diagnosis.receipt import AGREE, DIFFER, ComparisonReceipt, Envelope, Population
from engine.v2.foundation import to_document

ALL_L_IDS = [f"L{i:02d}" for i in range(1, 15)]
H = "sha256:" + "0" * 64

AGREE_KINDS = ("preview_open_parity", "current_switch_parity", "bridge_identity_parity",
              "bridge_mapping_parity", "bridge_value_parity", "publish_idempotency_parity",
              "api_pagination_parity", "browser_initial_load_parity", "ui_state_parity",
              "full_population_parity", "ui_build_typecheck_parity")
NEGATIVE_KINDS = ("preview_auth_negative_control", "bridge_malformed_ref_negative_control",
                  "bridge_mapping_negative_control", "bridge_value_negative_control",
                  "publish_corruption_negative_control", "api_cursor_negative_control",
                  "api_auth_traversal_negative_control", "no_scoring_startup_negative_control",
                  "publish_failure_negative_control", "secret_scan_negative_control")


def _git_root(tmp_path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    return root


def _ref(artifact_root: Path, rel: str, data: bytes) -> dict:
    path = artifact_root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": rel, "content_hash": "sha256:" + hashlib.sha256(data).hexdigest()}


def _dumps(obj) -> bytes:
    return json.dumps(to_document(obj)).encode()


def codes(result) -> set:
    return {f["code"] for f in result["findings"]}


def l_ids_with(result, code) -> set:
    return {f["l_id"] for f in result["findings"] if f["code"] == code and "l_id" in f}


# --------------------------------------------------------------------------
# Phase 2 half: a real, valid Phase2Evidence document plus its artifacts,
# written under artifact_root/phase2/... -- the shared-root convention
# rearchitecture_phase3_evidence.py's module docstring documents.
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


#: D14 review: the corpus receipt's ``diagnostic_ref`` must bind a real,
#: matching ``corpus_snapshot_binding.v1.0`` artifact -- see
#: ``tests/test_checks_phase2_gate.py``'s identical fixture.
_CORPUS_SNAPSHOT = "sha256:" + "c" * 64


def _p2_receipt(*, kind, code_hash, environment_hash, snapshot_ref, verdict=AGREE,
                population=None, receipt_id, diagnostic_ref=None) -> ComparisonReceipt:
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash,
                        snapshot_id=snapshot_ref.snapshot_id,
                        snapshot_manifest_hash=snapshot_ref.manifest_hash,
                        diagnostic_ref=diagnostic_ref)
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=kind, tier=0, left_ref="legacy", right_ref="adapter",
        stage_plan_ref="plan.v1", tolerance_policy_ref="tol.v1", verdict=verdict,
        population=population or _explained_population(), envelope=envelope)


def _p2_rollback(*, prior_id, resulting_id, prior_generation=2, resulting_generation=3) -> RollbackReceipt:
    return RollbackReceipt(receipt_id="recv_rollback", scope="legacy_shadow",
                           prior_snapshot_id=prior_id, resulting_snapshot_id=resulting_id,
                           prior_generation=prior_generation, resulting_generation=resulting_generation,
                           at="2026-01-01T00:00:00.000000Z")


def _fault_matrix() -> list:
    return [{"point": p, "outcome": "old_head", "verified_objects": True} for p in p2evidence.FAULT_POINTS]


def valid_phase2_evidence(artifact_root: Path, root: Path) -> dict:
    """A complete, real, valid ``phase2_evidence.v1.0`` document, written
    under ``artifact_root/phase2/...`` so it resolves against the SAME
    artifact root the Phase 3 manifest uses."""
    env_hash, _ = p2gate.environment_hash(root)
    code_hash = p2gate.source_hash(p2gate.source_files(root))
    snapshot_ref = _snapshot_ref("snap_current")
    prior = _import_receipt("snap_prior", receipt_id="recv_import_prior", generation=1)
    current = _import_receipt("snap_current", receipt_id="recv_import_current", generation=2)
    score = _p2_receipt(kind=p2evidence.SCORE_PARITY_KIND, code_hash=code_hash, environment_hash=env_hash,
                        snapshot_ref=snapshot_ref, receipt_id="recv_score")
    render = _p2_receipt(kind=p2evidence.RENDER_PARITY_KIND, code_hash=code_hash, environment_hash=env_hash,
                         snapshot_ref=snapshot_ref, receipt_id="recv_render")
    p = lambda rel, data: _ref(artifact_root, f"phase2/{rel}", data)  # noqa: E731
    # D14 review: the corpus receipt's diagnostic_ref must bind a real,
    # matching corpus_snapshot_binding.v1.0 artifact. The synthetic corpus
    # dir lives at artifact_root.parent / "corpus" -- the SAME path
    # ``corpus_root=`` below points the validator at (mirrors
    # tests/test_checks_phase2_gate.py's _gate_with_evidence convention).
    corpus_root = artifact_root.parent / "corpus"
    corpus_root.mkdir(exist_ok=True)
    (corpus_root / "INDEX.json").write_text(json.dumps({"snapshot": _CORPUS_SNAPSHOT}))
    binding = {
        "schema_version": p2evidence.CORPUS_SNAPSHOT_BINDING_V1, "corpus_version": "",
        "corpus_snapshot_hash": _CORPUS_SNAPSHOT, "source_snapshot_hash": _CORPUS_SNAPSHOT,
        "control": False, "control_drop_ticker": None,
    }
    binding_ref = p("corpus_binding.json", json.dumps(binding, sort_keys=True).encode())
    corpus = _p2_receipt(kind=p2evidence.CORPUS_PARITY_KIND, code_hash=code_hash, environment_hash=env_hash,
                         snapshot_ref=snapshot_ref, receipt_id="recv_corpus",
                         diagnostic_ref=json.dumps(binding_ref, sort_keys=True))
    rollback = _p2_rollback(prior_id="snap_current", resulting_id="snap_prior")
    return {
        "schema_version": p2evidence.PHASE2_EVIDENCE_V1, "code_hash": code_hash,
        "environment_hash": env_hash, "authority_mode": "shadow",
        "snapshot_ref": p("snapshot.json", _dumps(snapshot_ref)),
        "legacy_snapshot_object_ref": p("legacy.json", _dumps(
            ObjectRef(kind="legacy_snapshot", object_id="obj_legacy", content_hash=H, byte_size=99))),
        "table_contract_mapping_hash": "sha256:" + hashlib.sha256(b"mapping").hexdigest(),
        "import_receipt_refs": [p("import_prior.json", _dumps(prior)), p("import_current.json", _dumps(current))],
        "fault_matrix_ref": p("fault_matrix.json", json.dumps(_fault_matrix()).encode()),
        "dependency_plan_refs": [p("dep1.json", _dumps(_dependency_plan(snapshot_ref)))],
        "comparison_receipt_ref": p("comparison.json", _dumps(score)),
        "render_comparison_receipt_ref": p("render_comparison.json", _dumps(render)),
        "corpus_comparison_receipt_ref": p("corpus_comparison.json", _dumps(corpus)),
        "rollback_receipt_ref": p("rollback.json", _dumps(rollback)),
        "expected_population": 100, "supported_population": 80, "compared_population": 50,
    }


# --------------------------------------------------------------------------
# Phase 3 half
# --------------------------------------------------------------------------


def _preview_input(source_release_id="SRC1") -> PreviewInput:
    obj = ObjectRef(kind="legacy_snapshot", object_id="obj_legacy", content_hash=H, byte_size=10)
    return PreviewInput(
        source_release_id=source_release_id, source_release_manifest_ref=H, snapshot_ref=H,
        legacy_snapshot_object_ref=obj, score_batch_ref=H, score_job_input_refs=(H,),
        bundle_manifest_ref=H, model_registry_artifact_refs=(H,), model_evidence_ref=None,
        finality_ref=H, expected_population_ref=H, score_comparison_receipt_ref=H,
        render_comparison_receipt_ref=H, source_code_hash=H, source_environment_hash=H)


def _accepted_release(release_id, *, source_release_id="SRC1") -> PreviewRelease:
    return PreviewRelease(
        release_id=release_id, source_release_id=source_release_id, projection_manifest_ref=H,
        snapshot_ref=H, score_batch_ref=H, bundle_manifest_ref=H, model_registry_artifact_refs=(H,),
        model_evidence_ref=None, comparison_receipt_refs=(H,), source_code_hash=H,
        projection_code_hash=H, requested_as_of="2026-09-01", resolved_as_of="2026-09-01",
        clock_ids=("clk1",))


def _projection_binding(release: PreviewRelease) -> dict:
    """A real-shaped ``projection_binding.v1.0`` document (P3-1c,
    ``engine.v2.serving.projections.projection_binding``) -- the actual
    fields the real function emits that this validator checks."""
    return {"schema_version": "projection_binding.v1.0", "projection_release_id": release.release_id,
           "source_release_id": release.source_release_id, "projection_manifest_ref": H,
           "projection_manifest_hash": H, "bundle_manifest_ref": H, "serving_index_identity": H,
           "comparison_receipt_refs": [], "requested_as_of": release.requested_as_of,
           "resolved_as_of": release.resolved_as_of}


def _accepted_release_ref(artifact_root, release_id, name, *, source_release_id="SRC1") -> dict:
    release = _accepted_release(release_id, source_release_id=source_release_id)
    ref = _ref(artifact_root, f"{name}.json", _dumps(release))
    ref["binding_ref"] = _ref(artifact_root, f"{name}_binding.json",
                              json.dumps(_projection_binding(release)).encode())
    return ref


def _p3_receipt(*, kind, code_hash, environment_hash, verdict=AGREE, right_ref="release:REL1",
                receipt_id, compared=10) -> ComparisonReceipt:
    envelope = Envelope(code_hash=code_hash, environment_hash=environment_hash)
    population = Population(expected=compared + 5, supported=compared + 2, compared=compared)
    return ComparisonReceipt(
        receipt_id=receipt_id, comparison_kind=kind, tier=1, left_ref="legacy", right_ref=right_ref,
        stage_plan_ref="plan.v1", tolerance_policy_ref="tol.v1", verdict=verdict,
        population=population, envelope=envelope)


def valid_evidence(tmp_path, *, populate_all=True) -> tuple[dict, Path, Path]:
    """A complete, real, valid ``phase3_evidence.v1.0`` document plus its
    artifact root and its synthetic git-init'd implementation root.
    ``populate_all=False`` skips writing any comparison/negative-control
    receipts, leaving the caller to add exactly the ones a test needs.
    """
    root = _git_root(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    code_hash = p2gate.source_hash(p2gate.source_files(root))
    env_hash, _ = p2gate.environment_hash(root)

    phase2_doc = valid_phase2_evidence(artifact_root, root)
    phase2_ref = _ref(artifact_root, "phase2/_evidence.json", json.dumps(phase2_doc).encode())

    evidence: dict = {
        "schema_version": p3evidence.PHASE3_EVIDENCE_V1, "authority_mode": "shadow",
        "implementation_code_hash": code_hash, "environment_hash": env_hash,
        "frontend_lock_hash": "sha256:" + "1" * 64,
        "phase2_acceptance_ref": phase2_ref,
        "source_code_hash": phase2_doc["code_hash"], "source_environment_hash": phase2_doc["environment_hash"],
        "mapping_version": "legacy_score_bridge.v1.0",
        "preview_input_refs": [_ref(artifact_root, "preview_input.json", _dumps(_preview_input()))],
        "accepted_release_refs": [
            _accepted_release_ref(artifact_root, "REL1", "release_1"),
            _accepted_release_ref(artifact_root, "REL2", "release_2"),
        ],
        "population_manifest_ref": _ref(artifact_root, "population_manifest.json",
                                        json.dumps({"expected": 100, "supported": 80, "compared": 50}).encode()),
        "browser_receipt_ref": _ref(artifact_root, "browser_receipt.json", json.dumps({
            "release_id": "REL1", "as_of": "2026-09-01", "url": "http://127.0.0.1:8765/",
            "screenshot_ref": _ref(artifact_root, "screenshot.png", b"\x89PNG\r\n"),
        }).encode()),
        "refresh_rollback_receipt_ref": _ref(artifact_root, "refresh_rollback.json", _dumps(
            RollbackReceipt(receipt_id="recv_refresh", scope="v2_serving_preview", prior_snapshot_id="REL1",
                            resulting_snapshot_id="REL2", prior_generation=1, resulting_generation=2,
                            at="2026-09-01T00:00:00.000000Z"))),
        "engineering_receipt_ref": _ref(artifact_root, "engineering_receipt.json", json.dumps({
            "nights": [{"date": "2026-09-01", "status": "observed"}, {"date": "2026-09-02", "status": "unknown"}],
        }).encode()),
        "coverage_receipt_ref": _ref(artifact_root, "coverage_receipt.json", json.dumps({
            "schema_version": "phase3_coverage.v1.0", "source_hash": code_hash,
            "suite": list(FIXED_SUITE),
            "suite_missing": [], "pytest_returncode": 0,
            "packages": json.loads((Path(__file__).resolve().parents[1] /
                                    "checks/rearchitecture_phase3_coverage_baseline.json").read_text())["packages"],
        }).encode()),
        "performance_receipt_ref": _ref(artifact_root, "performance_receipt.json", json.dumps({
            "api_latency_p50_ms": 120, "first_usable_page_seconds": 1.4, "bytes": 20000,
            "population": 50, "memory_mb": 180, "cache_state": "cold", "contention_note": "idle box",
        }).encode()),
        "view_field_inventory_ref": _ref(artifact_root, "view_field_inventory.json", json.dumps(
            [{"field": "exp_pnl_model", "shipped": True}]).encode()),
        "deferred_work_ref": _ref(artifact_root, "deferred_work.json", json.dumps(
            [{"item": "full health screen", "owner": "phase6"}]).encode()),
    }
    if populate_all:
        comparisons = [_p3_receipt(kind=k, code_hash=code_hash, environment_hash=env_hash,
                                   receipt_id=f"recv_{k}") for k in AGREE_KINDS]
        negatives = [_p3_receipt(kind=k, code_hash=code_hash, environment_hash=env_hash, verdict=DIFFER,
                                 receipt_id=f"recv_{k}") for k in NEGATIVE_KINDS]
        evidence["comparison_receipt_refs"] = [
            _ref(artifact_root, f"comparison_{r.comparison_kind}.json", _dumps(r)) for r in comparisons]
        evidence["negative_control_receipt_refs"] = [
            _ref(artifact_root, f"negative_{r.comparison_kind}.json", _dumps(r)) for r in negatives]
    return evidence, artifact_root, root


def _gate(evidence, artifact_root, root, registry_path=REAL_REGISTRY_PATH):
    evidence_path = artifact_root / "evidence_final.json"
    evidence_path.write_text(json.dumps(evidence))
    return p3gate.gate(root=root, evidence_manifest_path=evidence_path, artifact_root=artifact_root,
                       corpus_root=artifact_root.parent / "corpus", registry_path=registry_path)


def _retained_phase2_disposition(evidence, artifact_root):
    """Turn the synthetic Phase 2 prerequisite into the exact accepted red state."""
    phase2_ref = evidence["phase2_acceptance_ref"]
    phase2_doc = json.loads((artifact_root / phase2_ref["path"]).read_text())
    for field, stale_code in (
        ("corpus_comparison_receipt_ref", "sha256:" + "9" * 64),
        ("comparison_receipt_ref", None),
    ):
        receipt_ref = phase2_doc[field]
        receipt_path = artifact_root / receipt_ref["path"]
        receipt = json.loads(receipt_path.read_text())
        receipt["verdict"] = DIFFER
        if stale_code is not None:
            receipt["envelope"]["code_hash"] = stale_code
        phase2_doc[field] = _ref(artifact_root, receipt_ref["path"], json.dumps(receipt).encode())
    phase2_ref = _ref(artifact_root, "phase2/_evidence.json", json.dumps(phase2_doc).encode())
    evidence["phase2_acceptance_ref"] = phase2_ref
    accepted = []
    for d_id, field in (("D14", "corpus_comparison_receipt_ref"), ("D15", "comparison_receipt_ref")):
        receipt = json.loads((artifact_root / phase2_doc[field]["path"]).read_text())
        population = {key: receipt["population"][key] for key in ("expected", "supported", "compared")}
        accepted.append({"d_id": d_id, "receipt_field": field, "receipt_ref": phase2_doc[field],
                         "population": population, "cause": "stale_legacy_price_archive"})
    disposition = {
        "schema_version": p3evidence.PHASE2_HANDOFF_DISPOSITION_V1,
        "phase2_evidence_ref": phase2_ref,
        "candidate_code_hash": phase2_doc["code_hash"],
        "candidate_environment_hash": phase2_doc["environment_hash"],
        "accepted_findings": accepted,
    }
    evidence["phase2_handoff_disposition_ref"] = _ref(
        artifact_root, "phase2_handoff_disposition.json", json.dumps(disposition).encode())
    return disposition


# -- registry shape -----------------------------------------------------------

def test_registry_covers_exactly_l01_through_l14():
    rows = p3gate.load_registry(REAL_REGISTRY_PATH)
    assert set(rows) == set(ALL_L_IDS)


# -- fully valid synthetic world: ok ------------------------------------------

def test_fully_valid_evidence_is_ok(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    result = _gate(evidence, artifact_root, root)
    assert result["ok"] is True, result["findings"]
    assert result["findings"] == []
    assert set(result["l_rows"]) == set(ALL_L_IDS)
    assert all(row["ok"] for row in result["l_rows"].values())


def test_exact_phase2_disposition_retains_strict_findings_but_allows_readiness(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    _retained_phase2_disposition(evidence, artifact_root)
    result = _gate(evidence, artifact_root, root)
    assert result["ok"] is False
    assert result["accepted_readiness_ok"] is True
    assert codes(result) == {"PHASE2_STRICT_FINDINGS_RETAINED"}
    assert all(row["ok"] for row in result["l_rows"].values())


def test_phase2_disposition_refuses_new_or_different_prerequisite_findings(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    _retained_phase2_disposition(evidence, artifact_root)
    phase2_ref = evidence["phase2_acceptance_ref"]
    phase2_doc = json.loads((artifact_root / phase2_ref["path"]).read_text())
    phase2_doc["expected_population"] = 0
    evidence["phase2_acceptance_ref"] = _ref(
        artifact_root, "phase2/_evidence.json", json.dumps(phase2_doc).encode())
    result = _gate(evidence, artifact_root, root)
    assert result["accepted_readiness_ok"] is False
    assert "PREREQUISITE_FAILED" in codes(result)


def test_coverage_receipt_refuses_a_dropped_fixed_suite_file(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    receipt_path = artifact_root / evidence["coverage_receipt_ref"]["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["suite"] = receipt["suite"][:-1]
    evidence["coverage_receipt_ref"] = _ref(artifact_root, receipt_path.name, json.dumps(receipt).encode())
    result = _gate(evidence, artifact_root, root)
    assert "COVERAGE_SUITE_DRIFT" in codes(result)
    assert "L14" in l_ids_with(result, "MISSING_EVIDENCE")


def test_coverage_receipt_refuses_a_regression(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    receipt_path = artifact_root / evidence["coverage_receipt_ref"]["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["packages"]["engine.v2.serving"]["executed"] -= 1
    evidence["coverage_receipt_ref"] = _ref(artifact_root, receipt_path.name, json.dumps(receipt).encode())
    result = _gate(evidence, artifact_root, root)
    assert "COVERAGE_REGRESSION" in codes(result)


# -- required fields: each missing field gives its own code -------------------

@pytest.mark.parametrize("field", [
    "implementation_code_hash", "environment_hash", "frontend_lock_hash",
    "source_code_hash", "source_environment_hash", "mapping_version",
])
def test_missing_scalar_field_gives_missing_evidence(tmp_path, field):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    del evidence[field]
    result = _gate(evidence, artifact_root, root)
    assert {"code": "MISSING_EVIDENCE", "field": field} in result["findings"]


@pytest.mark.parametrize("field", [
    "phase2_acceptance_ref", "preview_input_refs", "accepted_release_refs",
    "population_manifest_ref", "browser_receipt_ref", "refresh_rollback_receipt_ref",
    "engineering_receipt_ref", "coverage_receipt_ref", "performance_receipt_ref",
    "view_field_inventory_ref", "deferred_work_ref", "comparison_receipt_refs",
    "negative_control_receipt_refs",
])
def test_missing_ref_field_gives_a_code_and_fails(tmp_path, field):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    del evidence[field]
    result = _gate(evidence, artifact_root, root)
    assert result["ok"] is False
    assert any(f.get("field") == field or f.get("field", "").startswith(field + "[") for f in result["findings"])


# -- unknown field / unsupported schema ---------------------------------------

def test_unknown_top_level_field_gives_unknown_field(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    evidence["surprise_field"] = "x"
    result = _gate(evidence, artifact_root, root)
    assert {"code": "UNKNOWN_FIELD", "field": "surprise_field"} in result["findings"]


def test_unsupported_schema_version_gives_schema_version_unsupported(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    evidence["schema_version"] = "phase3_evidence.v9.9"
    result = _gate(evidence, artifact_root, root)
    # An unsupported schema version short-circuits validate_evidence entirely
    # (field_ok/kind_ok come back empty), so every L-row's own requirement
    # ALSO reports MISSING_EVIDENCE -- SCHEMA_VERSION_UNSUPPORTED is the one
    # code that identifies the root cause, not the only finding.
    assert "SCHEMA_VERSION_UNSUPPORTED" in codes(result)
    assert result["ok"] is False


# -- L-row failures: bad verdict / wrong commit binding ------------------------

def test_l_row_failing_verdict_is_refused_and_scoped_to_that_row(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    code_hash = evidence["implementation_code_hash"]
    env_hash = evidence["environment_hash"]
    differing = _p3_receipt(kind="bridge_identity_parity", code_hash=code_hash, environment_hash=env_hash,
                            verdict=DIFFER, receipt_id="recv_bad")
    evidence["comparison_receipt_refs"] = [
        r for r in evidence["comparison_receipt_refs"] if json.loads(
            (artifact_root / r["path"]).read_text())["comparison_kind"] != "bridge_identity_parity"
    ] + [_ref(artifact_root, "comparison_bridge_identity_parity.json", _dumps(differing))]
    result = _gate(evidence, artifact_root, root)
    assert "VERDICT_NOT_AGREE" in codes(result)
    assert l_ids_with(result, "MISSING_EVIDENCE") == {"L03"}


def test_l_row_receipt_bound_to_different_commit_is_refused(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    stale = _p3_receipt(kind="api_pagination_parity", code_hash="sha256:" + "9" * 64,
                        environment_hash=evidence["environment_hash"], receipt_id="recv_stale")
    evidence["comparison_receipt_refs"] = [
        r for r in evidence["comparison_receipt_refs"] if json.loads(
            (artifact_root / r["path"]).read_text())["comparison_kind"] != "api_pagination_parity"
    ] + [_ref(artifact_root, "comparison_api_pagination_parity.json", _dumps(stale))]
    result = _gate(evidence, artifact_root, root)
    assert "CODE_HASH_MISMATCH" in codes(result)
    assert "L07" in l_ids_with(result, "MISSING_EVIDENCE")


def test_negative_control_that_agrees_did_not_fire(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    code_hash, env_hash = evidence["implementation_code_hash"], evidence["environment_hash"]
    never_fired = _p3_receipt(kind="api_auth_traversal_negative_control", code_hash=code_hash,
                              environment_hash=env_hash, verdict=AGREE, receipt_id="recv_never_fired")
    evidence["negative_control_receipt_refs"] = [
        r for r in evidence["negative_control_receipt_refs"] if json.loads(
            (artifact_root / r["path"]).read_text())["comparison_kind"] != "api_auth_traversal_negative_control"
    ] + [_ref(artifact_root, "negative_api_auth_traversal_negative_control.json", _dumps(never_fired))]
    result = _gate(evidence, artifact_root, root)
    assert "NEGATIVE_CONTROL_NOT_TRIGGERED" in codes(result)
    assert l_ids_with(result, "MISSING_EVIDENCE") == {"L08"}


# -- empty compared population -------------------------------------------------

def test_empty_compared_population_is_refused(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    evidence["population_manifest_ref"] = _ref(
        artifact_root, "population_manifest.json",
        json.dumps({"expected": 100, "supported": 80, "compared": 0}).encode())
    result = _gate(evidence, artifact_root, root)
    assert "POPULATION_EMPTY" in codes(result)
    assert result["ok"] is False


def test_full_population_parity_receipt_itself_zero_is_refused(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    code_hash, env_hash = evidence["implementation_code_hash"], evidence["environment_hash"]
    zero = _p3_receipt(kind="full_population_parity", code_hash=code_hash, environment_hash=env_hash,
                       compared=0, receipt_id="recv_zero")
    zero_doc = to_document(zero)
    zero_doc["population"]["compared"] = 0
    evidence["comparison_receipt_refs"] = [
        r for r in evidence["comparison_receipt_refs"] if json.loads(
            (artifact_root / r["path"]).read_text())["comparison_kind"] != "full_population_parity"
    ] + [_ref(artifact_root, "comparison_full_population_parity.json", json.dumps(zero_doc).encode())]
    result = _gate(evidence, artifact_root, root)
    assert "POPULATION_EMPTY" in codes(result)
    assert "L12" in l_ids_with(result, "MISSING_EVIDENCE")


# -- accepted release bound to a real projection_binding.v1.0 document --------

def test_accepted_release_missing_binding_ref_is_refused(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    stripped = dict(evidence["accepted_release_refs"][0])
    del stripped["binding_ref"]
    evidence["accepted_release_refs"][0] = stripped
    result = _gate(evidence, artifact_root, root)
    assert "ARTIFACT_MISSING" in codes(result)
    assert result["ok"] is False


def test_accepted_release_binding_naming_a_different_release_is_refused(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    wrong_release = _accepted_release("SOME_OTHER_RELEASE")
    evidence["accepted_release_refs"][0]["binding_ref"] = _ref(
        artifact_root, "release_1_binding.json", json.dumps(_projection_binding(wrong_release)).encode())
    result = _gate(evidence, artifact_root, root)
    assert "RELEASE_BINDING_MISMATCH" in codes(result)
    assert result["ok"] is False


# -- single generation ----------------------------------------------------------

def test_single_generation_rollback_is_refused(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    stale = RollbackReceipt(receipt_id="recv_refresh", scope="v2_serving_preview",
                            prior_snapshot_id="REL1", resulting_snapshot_id="REL1",
                            prior_generation=2, resulting_generation=2, at="2026-09-01T00:00:00.000000Z")
    evidence["refresh_rollback_receipt_ref"] = _ref(artifact_root, "refresh_rollback.json", _dumps(stale))
    result = _gate(evidence, artifact_root, root)
    assert codes(result) & {"SINGLE_GENERATION"}
    assert "L13" in l_ids_with(result, "MISSING_EVIDENCE")


# -- Phase 2 evidence must be valid --------------------------------------------

def test_invalid_phase2_evidence_gives_prerequisite_failed(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    phase2_doc = json.loads((artifact_root / "phase2/_evidence.json").read_text())
    phase2_doc["code_hash"] = "sha256:" + "7" * 64  # no longer matches its own receipts' bindings
    evidence["phase2_acceptance_ref"] = _ref(artifact_root, "phase2/_evidence.json",
                                             json.dumps(phase2_doc).encode())
    evidence["source_code_hash"] = phase2_doc["code_hash"]
    result = _gate(evidence, artifact_root, root)
    assert "PREREQUISITE_FAILED" in codes(result)
    assert result["ok"] is False


# -- report writer refuses a path inside the repo ------------------------------

def test_write_report_refuses_a_path_inside_the_repo(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    result = _gate(evidence, artifact_root, root)
    with pytest.raises(p3gate.ReportPathError):
        p3gate.write_report(root, root / "REPORT.md", result)


def test_write_report_accepts_a_private_path(tmp_path):
    evidence, artifact_root, root = valid_evidence(tmp_path)
    result = _gate(evidence, artifact_root, root)
    out = tmp_path / "private" / "REPORT.md"
    written = p3gate.write_report(root, out, result)
    assert written == out
    text = out.read_text()
    assert "L01-L14 matrix" in text
    assert "L03 | PASS" in text


# -- real repo smoke: bare gate is red today -----------------------------------

def test_real_repo_bare_gate_is_red_with_missing_evidence():
    result = p3gate.gate(root=p3gate.ROOT)
    assert result["ok"] is False
    assert codes(result) == {"MISSING_EVIDENCE"}
    assert set(result["l_rows"]) == set(ALL_L_IDS)
    assert not any(row["ok"] for row in result["l_rows"].values())
