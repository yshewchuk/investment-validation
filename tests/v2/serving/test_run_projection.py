"""Part C: ``run_projection`` and the ``row_sources`` seam.

Covers ``tools.v2_dashboard_project.run_projection`` (the callable entry
point the CLI wraps), ``projections.build_candidate``'s ``row_sources``
validation/identity, and the plan-driven row-source selection ``run_projection``
records on the stored release.
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
from engine.v2.serving.legacy_bundle import (  # noqa: E402
    LegacyBundleError,
    read_capped_bytes,
)
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
from tools.v2_dashboard_project import run_projection  # noqa: E402

_NATIVE_PLAN = {"shadow_serving_scorer": "native"}


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _flat_inputs(tmp_path: Path) -> dict:
    """Synthetic inputs mirroring ``tests/test_v2_serving_projections.py``'s
    ``main([...])`` CLI test (~lines 540-610) and ``release_support.py``."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "phase2").mkdir()
    conn, store, snap = _events_snapshot(
        tmp_path / "phase2",
        [_event_row("e1", "AAA", datetime(2024, 1, 5)),
         _event_row("e2", "BBB", datetime(2024, 1, 5))])
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


def _spy_row_source(calls: list):
    def spy(plan, score_doc, bundle_rows_by_ticker, pairs):
        calls.append(plan)
        return bundle_rows_by_ticker
    return spy


def _assert_no_release_row(serving_root: Path) -> None:
    index_path = serving_root / "serving.sqlite"
    assert not index_path.exists() or index_path.stat().st_size == 0
    if index_path.exists():
        serving_conn = projections.connect(str(index_path))
        try:
            assert serving_conn.execute(
                "SELECT COUNT(*) FROM serving_release").fetchone()[0] == 0
        finally:
            serving_conn.close()


# 1. run_projection(...) == main([...])'s printed JSON, fresh serving root each
def test_run_projection_matches_main_cli_print_for_same_inputs(tmp_path, capsys):
    from tools.v2_dashboard_project import main

    inputs = _flat_inputs(tmp_path / "inputs")

    code_direct, doc_direct = run_projection(
        **_args(inputs, tmp_path / "serving_direct"))
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
        resolved_as_of="2024-01-04", row_sources=("native_shadow",))

    assert isinstance(legacy, PreviewRelease)
    assert isinstance(shadow, PreviewRelease)
    assert legacy.release_id != shadow.release_id

    legacy_read = projections.get_release(serving_conn, legacy.release_id)
    shadow_read = projections.get_release(serving_conn, shadow.release_id)
    assert legacy_read == legacy and legacy_read.row_sources == ("legacy_bundle",)
    assert shadow_read == shadow and shadow_read.row_sources == ("native_shadow",)


# 3. invalid row_sources -> ValueError
def test_build_candidate_refuses_invalid_row_sources(tmp_path):
    conn, store, snap = _events_snapshot(
        tmp_path, [_event_row("e1", "AAA", datetime(2024, 1, 5))])
    repo = Repository(conn, store)
    serving_conn, serving_store = _serving(tmp_path)
    row = _row()
    preview = _preview_input()

    for bad in ((), ("legacy_bundle", "native_shadow"),
                ("native_shadow", "legacy_bundle"), ("bogus",),
                ("legacy_bundle", "legacy_bundle")):
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

    _assert_no_release_row(serving_root)


# 4b. a plan that is a JSON list, not JSON at all, or unreadable is INVALID_REQUEST
def test_run_projection_refuses_list_or_non_json_plan(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")

    cases = (
        ("list", "[]", "nightly plan is not a JSON object"),
        ("bad", "{not json", "nightly plan is not valid JSON"),
    )
    for name, payload, message in cases:
        plan_path = tmp_path / f"plan_{name}.json"
        plan_path.write_text(payload)
        serving_root = tmp_path / f"serving_{name}"

        with pytest.raises(OpsError) as error:
            run_projection(**_args(inputs, serving_root, nightly_plan=plan_path))
        assert error.value.code == "INVALID_REQUEST"
        assert error.value.problem.message == message
        _assert_no_release_row(serving_root)


# 11. read_capped_bytes refuses past the cap and returns the bytes up to it
def test_read_capped_refuses_over_cap_and_returns_at_cap(tmp_path):
    at_cap = tmp_path / "at_cap.bin"
    at_cap.write_bytes(b"0123456789")
    over_cap = tmp_path / "over_cap.bin"
    over_cap.write_bytes(b"0123456789A")

    assert read_capped_bytes(at_cap, cap=10) == b"0123456789"
    with pytest.raises(LegacyBundleError) as exc:
        read_capped_bytes(over_cap, cap=10)
    assert exc.value.code == "ARTIFACT_TOO_LARGE"


# 11e. a missing nightly plan -> INVALID_REQUEST ("nightly plan is unreadable"), no release row
def test_run_projection_refuses_missing_nightly_plan(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    missing_plan = tmp_path / "missing_plan.json"
    serving_root = tmp_path / "serving_missing_plan"
    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root, nightly_plan=missing_plan))
    assert error.value.code == "INVALID_REQUEST"
    assert error.value.problem.message == "nightly plan is unreadable"

    _assert_no_release_row(serving_root)


# 11a. a missing preview input -> INVALID_REQUEST ("<name> is unreadable"),
#      no release row
def test_run_projection_refuses_missing_preview_input(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    missing = tmp_path / "missing_preview_input.json"
    serving_root = tmp_path / "serving_missing_preview"

    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root, preview_input=missing))
    assert error.value.code == "INVALID_REQUEST"
    assert error.value.problem.message == f"{missing.name} is unreadable"
    _assert_no_release_row(serving_root)


# 11b. read_capped_bytes refuses a negative cap before opening the file
def test_read_capped_refuses_negative_cap(tmp_path):
    path = tmp_path / "any.bin"
    path.write_bytes(b"x")
    with pytest.raises(ValueError):
        read_capped_bytes(path, cap=-2)


# 11c. a symlinked score.json -> INVALID_REQUEST, no release row
def test_run_projection_refuses_symlinked_score_json(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    link = tmp_path / "score.json"
    link.symlink_to(inputs["score_json"])
    serving_root = tmp_path / "serving_symlink"

    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root, score_json=link))
    assert error.value.code == "INVALID_REQUEST"
    assert "SYMLINK_REFUSED" in error.value.problem.message
    assert link.name in error.value.problem.message
    _assert_no_release_row(serving_root)


# 11d. a missing score.json -> INVALID_REQUEST, no release row
def test_run_projection_refuses_missing_score_json(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    serving_root = tmp_path / "serving_missing_score"

    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root,
                               score_json=tmp_path / "missing_score.json"))
    assert error.value.code == "INVALID_REQUEST"
    assert "MISSING_FILE" in error.value.problem.message
    _assert_no_release_row(serving_root)


# 5. run_projection records the plan's row_sources on the stored release, and
#    the stored release's rows come from the native source
def test_run_projection_native_plan_records_native_shadow(tmp_path, monkeypatch):
    import tools.v2_dashboard_project as tool

    inputs = _flat_inputs(tmp_path / "inputs")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(dict(_NATIVE_PLAN)))
    native_inputs_path = tmp_path / "native_inputs.json"
    native_inputs_path.write_text("{}")

    monkeypatch.setattr(tool, "_load_native_inputs", lambda path: {})

    # The native source returns its own fresh rows mapping: only the native
    # ticker's display row, with no carry-over from the legacy bundle's rows.
    # It does not touch ``score_doc``.
    native_score_row = _row(ticker="BBB")
    native_display_row = _compact(native_score_row)

    calls = []

    def native_row_source(plan, score_doc, bundle_rows_by_ticker, pairs):
        calls.append(plan)
        return {"BBB": [native_display_row]}

    monkeypatch.setattr(tool, "shadow_serving_row_source", native_row_source)

    # Capture the rows the tool hands to ``build_candidate``: what the release
    # is built from, proving the native rows -- not the legacy bundle's -- are
    # the ones that reached it.  The wrapper also declares the native row's own
    # score record to the bridge (the shadow seam itself stays pure) so the
    # plan's extra population can be joined.
    captured: dict = {}
    real_build_candidate = tool.build_candidate

    def spy_build_candidate(preview_input, score_doc, bundle_rows_by_ticker, **kwargs):
        captured["rows"] = bundle_rows_by_ticker
        if "BBB" not in bundle_rows_by_ticker:
            return real_build_candidate(preview_input, score_doc, bundle_rows_by_ticker, **kwargs)
        joined = {
            **score_doc,
            "rows": [native_score_row],
            "expected_population": ["BBB|STR-THRU|2024-01-05"],
        }
        return real_build_candidate(preview_input, joined, bundle_rows_by_ticker, **kwargs)

    monkeypatch.setattr(tool, "build_candidate", spy_build_candidate)

    _, bundle_doc = run_projection(**_args(inputs, tmp_path / "serving_bundle"))
    assert calls == []
    _, native_doc = run_projection(
        **_args(inputs, tmp_path / "serving_native", nightly_plan=plan_path,
                native_inputs=native_inputs_path))
    assert native_doc["release_id"] != bundle_doc["release_id"]
    assert len(calls) == 1
    assert "BBB" in captured["rows"]
    assert "AAA" not in captured["rows"]

    conn = projections.connect(str(tmp_path / "serving_native" / "serving.sqlite"))
    try:
        release = projections.get_release(conn, native_doc["release_id"])
        page = projections.list_events(conn, native_doc["release_id"], limit=10)
    finally:
        conn.close()
    assert release.row_sources == ("native_shadow",)
    listed = {item.event_ref.event_id for item in page.items}
    assert listed == {"e2"}


# 6. legacy plan -> only the legacy_bundle row source, shadow source unused
def test_run_projection_legacy_plan_records_legacy_bundle_only(tmp_path, monkeypatch):
    import tools.v2_dashboard_project as tool

    inputs = _flat_inputs(tmp_path / "inputs")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"shadow_serving_scorer": "legacy"}))

    calls = []
    monkeypatch.setattr(
        tool, "shadow_serving_row_source", _spy_row_source(calls))

    _, doc = run_projection(
        **_args(inputs, tmp_path / "serving", nightly_plan=plan_path))
    assert len(calls) == 1
    assert calls[0]["shadow_serving_scorer"] == "legacy"
    conn = projections.connect(str(tmp_path / "serving" / "serving.sqlite"))
    try:
        release = projections.get_release(conn, doc["release_id"])
    finally:
        conn.close()
    assert release.row_sources == ("legacy_bundle",)


# 7. oversized nightly plan -> INVALID_REQUEST, no release row
def test_run_projection_oversized_nightly_plan_raises_and_writes_no_release(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    plan_path = tmp_path / "plan.json"
    # A valid JSON object, padded past the 1 MiB read cap with a long string:
    # the size refusal must be what fires, not the (valid) parse.
    plan_path.write_text(json.dumps(
        {"shadow_serving_scorer": "legacy", "pad": "x" * 1048576}))
    assert plan_path.stat().st_size > 1048576
    serving_root = tmp_path / "serving"

    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root, nightly_plan=plan_path))
    assert error.value.code == "INVALID_REQUEST"
    assert error.value.problem.message == "nightly plan exceeds 1 MiB"

    _assert_no_release_row(serving_root)


# 8. nightly plan size boundary: exactly 1 MiB is allowed, one byte over refused
def test_run_projection_plan_size_boundary(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    base = b'{"shadow_serving_scorer": "legacy"}'

    at_limit_path = tmp_path / "plan_at_limit.json"
    at_limit = base + b" " * (1048576 - len(base))
    at_limit_path.write_bytes(at_limit)
    assert len(at_limit) == 1048576

    _, doc = run_projection(
        **_args(inputs, tmp_path / "serving_at_limit", nightly_plan=at_limit_path))
    assert doc["ok"] is True

    over_path = tmp_path / "plan_over_limit.json"
    over = base + b" " * (1048577 - len(base))
    over_path.write_bytes(over)
    assert len(over) == 1048577

    serving_root = tmp_path / "serving_over_limit"
    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root, nightly_plan=over_path))
    assert error.value.code == "INVALID_REQUEST"

    _assert_no_release_row(serving_root)


# 9. a no-plan run records the legacy_bundle row source on the stored release
def test_run_projection_without_plan_records_legacy_bundle(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    serving_root = tmp_path / "serving_no_plan"

    _, doc = run_projection(**_args(inputs, serving_root))

    conn = projections.connect(str(serving_root / "serving.sqlite"))
    try:
        release = projections.get_release(conn, doc["release_id"])
    finally:
        conn.close()
    assert release.row_sources == ("legacy_bundle",)


# 10. a bogus bundle_format -> INVALID_REQUEST and no release row
def test_run_projection_refuses_bogus_bundle_format(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    serving_root = tmp_path / "serving_bogus"

    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root, bundle_format="bogus"))
    assert error.value.code == "INVALID_REQUEST"

    _assert_no_release_row(serving_root)


# 12. a directory as the nightly plan -> INVALID_REQUEST, no release row
def test_run_projection_plan_path_is_a_directory_refused(tmp_path):
    inputs = _flat_inputs(tmp_path / "inputs")
    plan_dir = tmp_path / "plan_dir"
    plan_dir.mkdir()
    serving_root = tmp_path / "serving"

    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root, nightly_plan=plan_dir))
    assert error.value.code == "INVALID_REQUEST"

    _assert_no_release_row(serving_root)


# 13. the preview input read is capped: over it -> INVALID_REQUEST, no release row
def test_run_projection_over_cap_preview_input_refused(tmp_path, monkeypatch):
    inputs = _flat_inputs(tmp_path / "inputs")
    monkeypatch.setattr("engine.v2.serving.legacy_bundle.MAX_ARTIFACT_BYTES", 64)
    serving_root = tmp_path / "serving"

    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root))
    assert error.value.code == "INVALID_REQUEST"
    assert inputs["preview_input"].name in error.value.problem.message
    assert "ARTIFACT_TOO_LARGE" in error.value.problem.message

    _assert_no_release_row(serving_root)


# 14. a cap larger than the preview input but smaller than score.json refuses
def test_run_projection_over_cap_score_json_refused(tmp_path, monkeypatch):
    inputs = _flat_inputs(tmp_path / "inputs")
    preview_size = inputs["preview_input"].stat().st_size
    score_path = inputs["score_json"]
    if score_path.stat().st_size <= preview_size:
        score_path.write_bytes(
            score_path.read_bytes()
            + b" " * (preview_size - score_path.stat().st_size + 1))
    score_size = score_path.stat().st_size
    assert score_size > preview_size

    cap = preview_size
    assert preview_size <= cap < score_size
    monkeypatch.setattr("engine.v2.serving.legacy_bundle.MAX_ARTIFACT_BYTES", cap)
    serving_root = tmp_path / "serving"

    with pytest.raises(OpsError) as error:
        run_projection(**_args(inputs, serving_root))
    assert error.value.code == "INVALID_REQUEST"
    assert score_path.name in error.value.problem.message
    assert "ARTIFACT_TOO_LARGE" in error.value.problem.message

    _assert_no_release_row(serving_root)
