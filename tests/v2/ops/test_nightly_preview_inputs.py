"""Pinned nightly preview-input artifacts and the plan bindings that declare them.

``nightly_preview_inputs`` pins a snapshot once and hashes exactly four parent
job outputs into two immutable documents; the legacy plan pins the same four
bindings (and drops nothing else) only when the plan carries a snapshot id.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from engine.v2.ops.errors import OpsError
from engine.v2.ops.nightly import build_legacy_job_requests, build_nightly_plan
from engine.v2.ops.nightly_preview_inputs import run
from engine.v2.ops.submission import job_id_for

_CHECKOUT = Path(__file__).resolve().parents[3]
_SESSION = "2026-09-18"
_SNAPSHOT_ID = "snap_pinned"

_INPUT_PAYLOADS = {
    "score.json": b'{"score": 1}\n',
    "bundle.tar": b"bundle-tar-payload",
    "finality.json": b'{"finality": true}\n',
    "model_evidence.json": b'{"evidence": "opaque"}\n',
}

_BINDINGS = {
    "score.json": "job_score1#legacy_score",
    "bundle.tar": "job_render2#legacy_render",
    "finality.json": "job_finality3#legacy_finality",
    "model_evidence.json": "job_evidence4#legacy_model_evidence",
}


def _write_inputs(root: Path) -> None:
    for name, payload in _INPUT_PAYLOADS.items():
        (root / name).write_bytes(payload)


def _parameters(**overrides) -> dict:
    parameters = {"expected_ids": ("nightly_preview_inputs",),
                  "snapshot_id": _SNAPSHOT_ID,
                  "input_bindings": dict(_BINDINGS)}
    parameters.update(overrides)
    return parameters


def _stage_request(requests, stage):
    matches = [request for request in requests
               if request.idempotency_key.rsplit(":", 1)[-1] == stage]
    assert len(matches) == 1, stage
    return matches[0]


def _stage_job_id(requests, stage) -> str:
    request = _stage_request(requests, stage)
    return job_id_for(request.namespace, request.idempotency_key)


def test_nightly_preview_inputs_emit_deterministic_artifacts(tmp_path):
    _write_inputs(tmp_path)
    parameters = _parameters()
    result = run(parameters, tmp_path)
    assert [output["name"] for output in result["outputs"]] == ["preview_input",
                                                               "source_provenance"]

    preview = json.loads((tmp_path / "preview_input.json").read_bytes())
    provenance = json.loads((tmp_path / "source_provenance.json").read_bytes())
    assert preview["schema_version"] == "nightly_preview_input.v1.0"
    assert provenance["schema_version"] == "nightly_source_provenance.v1.0"
    assert preview["snapshot_id"] == _SNAPSHOT_ID
    assert provenance["snapshot_id"] == preview["snapshot_id"]
    assert set(preview["inputs"]) == set(_INPUT_PAYLOADS)
    assert set(provenance["inputs"]) == set(_INPUT_PAYLOADS)
    producing = sorted({binding.partition("#")[0] for binding in _BINDINGS.values()})
    assert provenance["producing_job_ids"] == producing
    for name, payload in _INPUT_PAYLOADS.items():
        parent, _, output = _BINDINGS[name].partition("#")
        expected_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
        for document in (preview["inputs"][name], provenance["inputs"][name]):
            assert document["producer_job_id"] == parent
            assert document["output_name"] == output
            assert document["content_hash"] == expected_hash
        assert provenance["inputs"][name]["hash_algorithm"] == "sha256"

    names = ("preview_input.json", "source_provenance.json")
    first = {name: (tmp_path / name).read_bytes() for name in names}
    assert run(parameters, tmp_path) == result
    for name, payload in first.items():
        assert (tmp_path / name).read_bytes() == payload


def test_nightly_preview_inputs_reject_missing_pin_or_binding(tmp_path):
    incomplete = dict(_BINDINGS)
    del incomplete["model_evidence.json"]
    with pytest.raises(OpsError) as missing:
        run(_parameters(input_bindings=incomplete), tmp_path)
    assert missing.value.code == "INVALID_REQUEST"
    with pytest.raises(OpsError) as blank:
        run(_parameters(snapshot_id=""), tmp_path)
    assert blank.value.code == "INVALID_REQUEST"


def test_pinned_nightly_plan_declares_preview_output_bindings():
    plan = build_nightly_plan(_CHECKOUT, _SESSION)
    plan["snapshot_id"] = _SNAPSHOT_ID
    requests = build_legacy_job_requests(plan, tickers=("FAKE",), year_start=2025,
                                         year_end=2026)
    preview = _stage_request(requests, "nightly_preview_inputs")
    assert preview.job.kind == "nightly_preview_inputs"
    assert preview.job.parameters["snapshot_id"] == _SNAPSHOT_ID

    score_id = _stage_job_id(requests, "score")
    projection_id = _stage_job_id(requests, "projection")
    finality_id = _stage_job_id(requests, "finality")
    evidence_id = _stage_job_id(requests, "model_evidence")
    assert preview.job.parameters["input_bindings"] == {
        "score.json": score_id + "#legacy_score",
        "bundle.tar": projection_id + "#legacy_render",
        "finality.json": finality_id + "#legacy_finality",
        "model_evidence.json": evidence_id + "#legacy_model_evidence",
    }
    assert preview.job.dependency_job_ids == (score_id, projection_id, finality_id,
                                              evidence_id)

    other_plan = dict(plan)
    other_plan["snapshot_id"] = "snap_other"
    other_requests = build_legacy_job_requests(other_plan, tickers=("FAKE",),
                                               year_start=2025, year_end=2026)
    other_preview = _stage_request(other_requests, "nightly_preview_inputs")
    assert other_preview.idempotency_key != preview.idempotency_key

    assert "serving_projection" not in {request.job.kind for request in requests}
