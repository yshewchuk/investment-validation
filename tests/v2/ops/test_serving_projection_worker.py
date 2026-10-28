from io import BytesIO
import tarfile

import pytest

from engine.v2.ops.errors import OpsError
from pathlib import Path

from engine.v2.ops.nightly import OPTIONAL, _DAG_STAGES, build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.submission import job_id_for
from tools import v2_dashboard_projection_worker as worker


def test_projection_worker_uses_staged_artifacts_and_returns_binding(monkeypatch, tmp_path):
    root = tmp_path / "staging"
    root.mkdir()
    names = {
        "bundle.tar", "finality.json", "score.json", "preview_input.json",
        "source_provenance.json", "nightly_plan.json", "native_inputs.json",
    }
    for name in names - {"bundle.tar"}:
        (root / name).write_text("{}")
    with tarfile.open(root / "bundle.tar", "w") as archive:
        body = b"{}"
        member = tarfile.TarInfo("bundle.json")
        member.size = len(body)
        archive.addfile(member, BytesIO(body))

    seen = {}

    def run_projection(**kwargs):
        seen.update(kwargs)
        return 0, {"projection_binding": {"release_id": "synthetic"}}

    monkeypatch.setattr(worker, "run_projection", run_projection)
    refs = {name: "sha256:" + ("a" * 64) for name in names}
    parameters = {
        "expected_ids": ("serving_projection",),
        "input_bindings": refs,
        "snapshot_id": "snapshot-1",
        "requested_as_of": "2026-10-26",
        "resolved_as_of": "2026-10-26",
        "bundle_format": "legacy",
    }
    verified = {}

    class FakeConnection:
        def close(self):
            pass

    def fake_connect(path):
        verified["path"] = path
        return FakeConnection()

    def fake_verify_projection_binding(conn, binding):
        verified["binding"] = binding
        return True

    monkeypatch.setattr(worker.projections, "connect", fake_connect)
    monkeypatch.setattr(worker.projections, "verify_projection_binding",
                        fake_verify_projection_binding)

    objects = tmp_path / "objects"
    result = worker.dispatch(
        "serving_projection", parameters, root,
        envelope={"catalog_path": str(tmp_path / "catalog.sqlite"),
                  "objects_root": str(objects)})

    assert verified["path"] == str(objects / "serving" / "serving.sqlite")
    assert verified["binding"] == {"release_id": "synthetic"}
    assert seen["preview_input"] == root / "preview_input.json"
    assert seen["source_provenance"] == root / "source_provenance.json"
    assert seen["score_json"] == root / "score.json"
    assert seen["nightly_plan"] == root / "nightly_plan.json"
    assert seen["native_inputs"] == root / "native_inputs.json"
    assert seen["bundle_dir"].is_dir()
    assert seen["serving_root"] == objects / "serving"
    assert result["completed_ids"] == ["serving_projection"]
    assert [(item["name"], item["schema"]) for item in result["outputs"]] == [
        ("projection_result", "serving_projection_result.v1.0"),
        ("projection_binding", "projection_binding.v1.0"),
    ]


def test_projection_worker_refuses_missing_bindings(tmp_path):
    with pytest.raises(OpsError) as raised:
        worker.dispatch(
            "serving_projection", {"input_bindings": {}}, tmp_path,
            envelope={"catalog_path": "catalog.sqlite", "objects_root": "objects"})
    assert raised.value.problem.code == "INVALID_REQUEST"


def test_serving_projection_request_is_required_leaf_after_publication():
    source_root = Path(__file__).resolve().parents[3]
    plan = build_nightly_plan(str(source_root), "2026-10-26")
    plan["serving_projection_bindings"] = {
        "preview_input.json": "sha256:" + "a" * 64,
        "source_provenance.json": "sha256:" + "b" * 64,
        "nightly_plan.json": "sha256:" + "c" * 64,
        "native_inputs.json": "sha256:" + "d" * 64,
    }
    requests = build_legacy_job_requests(
        plan, tickers=("FAKE",), year_start=2025, year_end=2026)
    by_kind = {request.job.kind: request for request in requests}
    projection = by_kind["serving_projection"]
    projection_id = job_id_for("shadow", projection.idempotency_key)
    publication_id = job_id_for("shadow", by_kind["publication"].idempotency_key)

    assert "serving_projection" not in OPTIONAL
    assert projection.job.resource_class == "projection"
    assert projection.job.dependency_job_ids == (publication_id,)
    assert set(projection.job.parameters["input_bindings"]) == {
        "bundle.tar", "finality.json", "score.json", "preview_input.json",
        "source_provenance.json", "nightly_plan.json", "native_inputs.json",
    }
    assert _DAG_STAGES.index("publication") < _DAG_STAGES.index("serving_projection")
    assert projection_id not in by_kind["backup"].job.dependency_job_ids
