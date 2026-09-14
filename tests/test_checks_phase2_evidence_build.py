"""``checks/rearchitecture_phase2_evidence_build.py``: on a synthetic catalog
with one committed import, the auto-resolved fields (snapshot_ref,
legacy_snapshot_object_ref, import_receipt_refs, table_contract_mapping_hash)
are present and valid without any CLI flag; only the CLI-supplied refs
(score/render/corpus/rollback/fault-matrix) start absent and become present
once supplied — never filled with a placeholder.
"""
from __future__ import annotations

import json

from checks import rearchitecture_phase2_evidence_build as eb
from checks.rearchitecture_phase2_evidence import validate_evidence
from engine.v2.data.catalog import commit_snapshot
from engine.v2.data.reference_catalog import ReferenceInput, insert_reference_inputs
from engine.v2.foundation import CONTENT_HASH_PREFIX
from tests.test_v2_data_commit import _catalog, _hash, _manifest_and_snapshot, _noop_fence
from tests.test_v2_data_manifests import _SEC_CONTRACT, _record_for

H = CONTENT_HASH_PREFIX + "0" * 64
_SUPPLIED_FIELDS = {"comparison_receipt_ref", "render_comparison_receipt_ref",
                    "corpus_comparison_receipt_ref", "rollback_receipt_ref", "fault_matrix_ref"}


def _synthetic_ops_root(root):
    """One committed ``shadow`` snapshot with pinned legacy-snapshot and
    calendar reference inputs -- everything the builder auto-resolves from
    the catalog, and nothing a CLI flag supplies."""
    root.mkdir(parents=True, exist_ok=True)
    conn, clock = _catalog(root)
    record = _record_for("2024")
    manifest, snap = _manifest_and_snapshot([record])
    refs = [ReferenceInput(kind="legacy_snapshot", legacy_path="legacy/snap.parquet",
                           object_id="obj_legacy", content_hash=H, byte_size=10),
           ReferenceInput(kind="calendar", legacy_path="legacy/calendar.csv",
                          object_id="obj_cal", content_hash=H, byte_size=5)]
    commit_snapshot(
        conn, scope="shadow", request_hash=_hash("req-a"), contracts=[_SEC_CONTRACT],
        objects=[record.object_ref], records=[record], manifests=[manifest], snapshot=snap,
        expected_head_snapshot_id=None, expected_head_generation=0, receipt_id="ra",
        attempt_id="att-a", fence=1, fence_check=_noop_fence, clock=clock,
        record_references=lambda c, rid: insert_reference_inputs(c, rid, refs))
    conn.close()
    return root


def _validate(evidence, artifact_root):
    return validate_evidence(evidence, artifact_root=artifact_root, code_hash=evidence["code_hash"],
                             environment_hash=evidence["environment_hash"])


def test_baseline_auto_resolves_catalog_fields_and_only_lacks_supplied_refs(tmp_path):
    root = _synthetic_ops_root(tmp_path / "ops")
    artifact_root = tmp_path / "artifacts"
    evidence = eb.build(root, scope="shadow", artifact_root=artifact_root, score_receipt=None,
                        render_receipt=None, corpus_receipt=None, rollback_receipt=None,
                        fault_matrix=None)
    assert evidence["authority_mode"] == "shadow"
    assert evidence["table_contract_mapping_hash"].startswith("sha256:")
    for field in ("snapshot_ref", "legacy_snapshot_object_ref", "import_receipt_refs"):
        assert field in evidence, field

    findings, field_ok, document_ok = _validate(evidence, artifact_root)
    assert document_ok is True
    # Populations are top-level evidence fields, not refs, so they are
    # checked unconditionally -- with no score receipt supplied they are
    # absent, which is the ONE expected finding here (POPULATION_EMPTY),
    # never a placeholder value filled in to silence it.
    assert findings == [{"code": "POPULATION_EMPTY", "field": "expected_population"}], findings
    assert field_ok.get("snapshot_ref") is True
    assert field_ok.get("legacy_snapshot_object_ref") is True
    assert field_ok.get("import_receipt_refs") is True
    # The unsupplied refs never entered `evidence` at all, so validate_evidence
    # never touched them either -- this IS "MISSING_EVIDENCE" as far as the
    # gate is concerned (checks/rearchitecture_phase2_gate.py's `_check_row`:
    # `if not field_ok.get(field)`), without needing a full gate run here.
    assert set(field_ok) & _SUPPLIED_FIELDS == set()


def test_supplying_render_corpus_rollback_and_fault_matrix_removes_them(tmp_path):
    root = _synthetic_ops_root(tmp_path / "ops")
    artifact_root = tmp_path / "artifacts"
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    receipt = {"schema_version": "comparison_receipt.v1.1", "receipt_id": "r1",
              "comparison_kind": "render_bundle_parity", "tier": 2, "left_ref": "l", "right_ref": "r",
              "stage_plan_ref": "p", "tolerance_policy_ref": "t", "verdict": "agree",
              "population": {"expected": 1, "supported": 1, "compared": 1}}
    render_path = inputs / "render.json"
    render_path.write_text(json.dumps(receipt))
    corpus_path = inputs / "corpus.json"
    corpus_path.write_text(json.dumps({**receipt, "comparison_kind": "corpus_score_parity"}))
    rollback_path = inputs / "rollback.json"
    rollback_path.write_text(json.dumps({
        "schema_version": "rollback_receipt.v1.0", "receipt_id": "rb1", "scope": "shadow",
        "prior_snapshot_id": "snap_prior", "resulting_snapshot_id": "snap_current",
        "prior_generation": 1, "resulting_generation": 2, "at": "2026-01-01T00:00:00.000000Z"}))
    fault_matrix_path = inputs / "fault_matrix.json"
    from checks import rearchitecture_phase2_evidence as p2evidence
    fault_matrix_path.write_text(json.dumps(
        [{"point": p, "outcome": "old_head", "verified_objects": True} for p in p2evidence.FAULT_POINTS]))

    evidence = eb.build(root, scope="shadow", artifact_root=artifact_root, score_receipt=None,
                        render_receipt=render_path, corpus_receipt=corpus_path,
                        rollback_receipt=rollback_path, fault_matrix=fault_matrix_path)
    findings, field_ok, document_ok = _validate(evidence, artifact_root)
    assert document_ok is True
    # These four are now present and structurally sound (the fault matrix
    # decodes clean; the render/corpus receipts, built with mismatched
    # code/environment/snapshot bindings on purpose, still resolve/decode --
    # only their BINDING checks fail, which is a separate concern from
    # "is this ref supplied at all").
    for field in ("render_comparison_receipt_ref", "corpus_comparison_receipt_ref",
                  "rollback_receipt_ref", "fault_matrix_ref"):
        assert field in field_ok, (field, findings)
    assert "comparison_receipt_ref" not in field_ok
