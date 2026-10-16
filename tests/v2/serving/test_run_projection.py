"""Part C: ``run_projection`` and the ``row_sources`` seam.

Covers ``tools.v2_dashboard_project.run_projection`` (the callable entry
point the CLI wraps), ``projections.build_candidate``'s ``row_sources``
validation/identity, and ``_plan_row_sources``'s None/legacy/native read.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from engine.v2.contracts import PreviewRelease  # noqa: E402
from engine.v2.data.repository import Repository  # noqa: E402
from engine.v2.diagnosis.receipt import ComparisonReceipt, Envelope, Population  # noqa: E402
from engine.v2.foundation import to_document  # noqa: E402
from engine.v2.ops.errors import OpsError  # noqa: E402
from engine.v2.serving import projections  # noqa: E402
from tests.test_v2_serving_projections import (  # noqa: E402
    _bundle,
    _compact,
    _event_row,
    _events_snapshot,
    _preview_input,
    _row,
    _score_doc,
    _serving,
)
from tests.v2.ops.test_native_shadow_render import _NATIVE_PLAN  # noqa: E402
from tools.v2_dashboard_project import _plan_row_sources, run_projection  # noqa: E402


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _flat_inputs(tmp_path: Path) -> dict:
    """Synthetic inputs mirroring ``tests/test_v2_serving_projections.py``'s
    ``main([...])`` CLI test (~lines 540-610) and ``release_support.py``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        tmp_path / "phase2", [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    conn.close()

    row = _row()
    score_path = tmp_path / "score.json"
    score_path.write_text(json.dumps(_score_doc(rows=[row])))
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "AAA.json").write_text(json.dumps([_compact(row)]))

    receipt = ComparisonReceipt(
        receipt_id="r", comparison_kind="score_record_parity", tier=1,
        left_ref="left", right_ref="right", stage_plan_ref="p",
        tolerance_policy_ref="t", verdict="agree",
        population=Population(expected=1, supported=1, compared=1),
        envelope=Envelope())
    score_receipt = json.dumps(to_document(receipt)).encode()
    render_receipt = json.dumps(to_document(
        dataclasses.replace(receipt, comparison_kind="render_bundle_parity"))).encode()
    preview = dataclasses.replace(
        _preview_input(), score_batch_ref=digest(score_path.read_bytes()),
        finality_ref=digest(b"finality"), model_evidence_ref=digest(b"models"),
        score_comparison_receipt_ref=digest(score_receipt),
        render_comparison_receipt_ref=digest(render_receipt))
    preview_input_path = tmp_path / "preview_input.json"
    preview_input_path.write_text(json.dumps(to_document(preview)))
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "score.json").write_bytes(score_receipt)
    (receipts / "render.json").write_bytes(render_receipt)
    proof_path = tmp_path / "source_provenance.json"
    proof_path.write_text(json.dumps({
        "schema_version": "phase3_source_provenance.v1.0",
        "release_id": preview.source_release_id,
        "release_manifest_hash": preview.source_release_manifest_ref,
        "bundle_manifest_ref": preview.bundle_manifest_ref,
        "score_artifact": {"content_hash": preview.score_batch_ref},
        "snapshot_artifact": {}, "materialization_request_artifact": {},
        "finality_artifact": {"content_hash": preview.finality_ref},
        "model_evidence_artifact": {"content_hash": preview.model_evidence_ref},
        "score_comparison_receipt": {"path": "receipts/score.json",
                                     "content_hash": preview.score_comparison_receipt_ref},
        "render_comparison_receipt": {"path": "receipts/render.json",
                                      "content_hash": preview.render_comparison_receipt_ref}}))

    return dict(
        preview_input=preview_input_path, source_provenance=proof_path,
        score_json=score_path, bundle_dir=bundle_dir, snapshot_id=snap.snapshot_id,
        catalog=tmp_path / "phase2" / "catalog.sqlite",
        store_root=tmp_path / "phase2" / "store",
        requested_as_of="2024-01-04", resolved_as_of="2024-01-04")


def _args(inputs: dict, serving_root: Path, **overrides) -> dict:
    kwargs = dict(inputs, serving_root=serving_root, bundle_format="flat")
    kwargs.update(overrides)
    return kwargs


# 1. run_projection(...) == main([...])'s printed JSON, fresh serving root each
def test_run_projection_matches_main_cli_print_for_same_inputs(tmp_path, capsys):
    from tools.v2_dashboard_project import main

    inputs = _flat_inputs(tmp_path / "inputs")

    code_direct, doc_direct = run_projection(
        **_args(inputs, tmp_path / "serving_direct"))
    printed = json.loads(capsys.readouterr().out) if False else None  # noqa: F841
    assert code_direct == 0
    assert doc_direct["ok"] is True

    code_cli = main([
        "--preview-input", str(inputs["preview_input"]),
        "--source-provenance", str(inputs["source_provenance"]),
        "--score-json", str(inputs["score_json"]),
        "--bundle-dir", str(inputs["bundle_dir"]),
        "--bundle-format", "flat",
        "--snapshot-id", inputs["snapshot_id"],
        "--catalog", str(inputs["catalog"]),
        "--store-root", str(inputs["store_root"]),
        "--serving-root", str(tmp_path / "serving_cli"),
        "--requested-as-of", inputs["requested_as_of"],
        "--resolved-as-of", inputs["resolved_as_of"],
    ])
    printed = json.loads(capsys.readouterr().out)
    assert code_cli == 0
    assert printed == doc_direct


# 2. row_sources differ -> different release_id; get_release carries each
def test_build_candidate_row_sources_change_release_id_and_get_release_keeps_each(tmp_path):
    conn, store, snap = _events_snapshot(
        tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row = _row()
    score_doc, bundle = _score_doc(rows=[row]), _bundle(_compact(row))
    preview = _preview_input()

    legacy = projections.build_candidate(
        preview, score_doc, bundle, repository=repo, snapshot_ref=snap,
        store=serving_store, conn=serving_conn, requested_as_of="2024-01-04",
        resolved_as_of="2024-01-04", row_sources=("legacy_bundle",))
    shadow = projections.build_candidate(
        preview, score_doc, bundle, repository=repo, snapshot_ref=snap,
        store=serving_store, conn=serving_conn, requested_as_of="2024-01-04",
        resolved_as_of="2024-01-04", row_sources=("legacy_bundle", "native_shadow"))

    assert isinstance(legacy, PreviewRelease)
    assert isinstance(shadow, PreviewRelease)
    assert legacy.release_id != shadow.release_id

    legacy_read = projections.get_release(serving_conn, legacy.release_id)
    shadow_read = projections.get_release(serving_conn, shadow.release_id)
    assert legacy_read == legacy and legacy_read.row_sources == ("legacy_bundle",)
    assert shadow_read == shadow and shadow_read.row_sources == (
        "legacy_bundle", "native_shadow")


# 3. invalid row_sources -> ValueError
def test_build_candidate_refuses_invalid_row_sources(tmp_path):
    conn, store, snap = _events_snapshot(
        tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row = _row()
    preview = _preview_input()

    for bad in ((), ("native_shadow",), ("legacy_bundle", "bogus")):
        with pytest.raises(ValueError, match="invalid row_sources"):
            projections.build_candidate(
                preview, _score_doc(rows=[row]), _bundle(_compact(row)),
                repository=repo, snapshot_ref=snap, store=serving_store,
                conn=serving_conn, requested_as_of="2024-01-04",
                resolved_as_of="2024-01-04", row_sources=bad)
    assert serving_conn.execute(
        "SELECT COUNT(*) FROM serving_release").fetchone()[0] == 0


# 4. native plan + native_inputs=None -> INVALID_REQUEST, no release row
def test_native_plan_without_native_inputs_raises_and_writes_no_release(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(dict(_NATIVE_PLAN)))
    serving_root = tmp_path / "serving"

    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root, nightly_plan=plan_path,
                               native_inputs=None))
    assert error.value.code == "INVALID_REQUEST"

    index_path = serving_root / "serving.sqlite"
    assert not index_path.exists() or index_path.stat().st_size == 0
    if index_path.exists():
        serving_conn = projections.connect(str(index_path))
        try:
            assert serving_conn.execute(
                "SELECT COUNT(*) FROM serving_release").fetchone()[0] == 0
        finally:
            serving_conn.close()


# 5. _plan_row_sources: None / legacy plan / native plan
def test_plan_row_sources_none_legacy_and_native(tmp_path):
    assert _plan_row_sources(None) == ("legacy_bundle",)

    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps({"shadow_serving_scorer": "legacy"}))
    assert _plan_row_sources(legacy_path) == ("legacy_bundle",)

    native_path = tmp_path / "native.json"
    native_path.write_text(json.dumps(dict(_NATIVE_PLAN)))
    assert _plan_row_sources(native_path) == ("legacy_bundle", "native_shadow")