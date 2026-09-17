import copy
import hashlib
import json

import pytest

from checks import phase4_real
from checks.tier0_corpus import Corpus
from engine.v2.contracts import SCORE_REQUEST_V1
from engine.v2.foundation import content_hash
from engine.v2.scoring.identity import request_hash


def _request_document():
    return {
        "event_id": "event-1",
        "calendar_revision": "calendar-1",
        "strategy_version": "STR-THRU",
        "deployment_id": "deployment-1",
        "decision_clock_id": "entry-close-1",
        "requested_decision_at": "2026-09-16",
        "snapshot_id": "snapshot-1",
        "mode": "replay",
        "fill_model": {"alpha": 0.5},
        "event_revision": "event-revision-1",
        "contract_override": None,
        "geometry_override": None,
        "dependency_refs": ["residual:1"],
        "model_artifact_refs": [],
        "residual_state_ref": None,
        "analog_state_ref": None,
        "calibration_state_ref": None,
        "schema_version": SCORE_REQUEST_V1,
    }


def _stage_rows():
    rows = {}
    prior = "root"
    for stage in phase4_real._REQUIRED_TRACE_STAGES:
        inputs = {"prior": prior, "stage": stage}
        output = {"stage": stage, "value": 1}
        rows[stage] = {
            "input": inputs,
            "output": output,
            "input_hash": content_hash(inputs),
            "output_hash": content_hash(output),
            "owner": "engine.score",
        }
        prior = rows[stage]["output_hash"]
    return rows


def _pair(tmp_path):
    request_doc = _request_document()
    residuals = [{"event_date": "2025-01-01", "err_move": 0.1}]
    raw = json.dumps(residuals, sort_keys=True).encode()
    (tmp_path / "residuals.json").write_bytes(raw)
    shared_inputs = {"request": request_doc, "source_rows": ["row-1"]}
    shared_hash = content_hash(shared_inputs)
    native_inputs = {
        "context": {"strategy": "STR-THRU"},
        "features": {"model_inputs": {}},
        "forecast": {},
        "geometry": None,
        "pricing": None,
        "analogs": {},
        "simulation": {"residuals": residuals},
        "gate": {},
        "chooser": {},
        "diagnostics": {},
        "source_ref": shared_hash,
    }
    traced_inputs = copy.deepcopy(native_inputs)
    traced_inputs["simulation"]["residuals"] = {"$resource": "residuals"}
    trace = {
        "schema_version": phase4_real._TRACE_SCHEMA,
        "request": request_doc,
        "request_hash": request_hash(
            phase4_real.from_document(phase4_real.ScoreRequest, request_doc)
        ),
        "shared_inputs": shared_inputs,
        "shared_input_hash": shared_hash,
        "native_input_hash": shared_hash,
        "native_inputs": traced_inputs,
        "native_inputs_hash": content_hash(native_inputs),
        "stages": _stage_rows(),
        "resources": [{
            "resource_id": "residuals",
            "ref": "residual:1",
            "kind": "sidecar",
            "path": "residuals.json",
            "sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
            "content_hash": content_hash(residuals),
        }],
    }
    trace["trace_hash"] = content_hash(trace)
    return {
        "payload_hash": content_hash({"record": {}}),
        "payload": {
            "request": request_doc,
            "record": {},
            "legacy_input_hash": shared_hash,
            "input_trace": trace,
            "input_trace_hash": trace["trace_hash"],
        },
    }


def _resign(pair):
    trace = pair["payload"]["input_trace"]
    trace["trace_hash"] = content_hash({
        key: value for key, value in trace.items() if key != "trace_hash"
    })
    pair["payload"]["input_trace_hash"] = trace["trace_hash"]


def test_complete_trace_reconstructs_exact_request_and_inputs(tmp_path):
    pair = _pair(tmp_path)

    verified = phase4_real._verified_trace_bundle(pair, tmp_path)

    assert verified["request"].event_id == "event-1"
    assert verified["input_hash"] == pair["payload"]["legacy_input_hash"]
    assert verified["inputs"].source_ref == verified["input_hash"]
    assert verified["inputs"].simulation["residuals"][0]["err_move"] == 0.1
    assert tuple(receipt.stage for receipt in verified["inputs"].stage_receipts) == (
        phase4_real._REQUIRED_TRACE_STAGES
    )


def test_stage_hash_mismatch_is_incomparable(tmp_path):
    pair = _pair(tmp_path)
    pair["payload"]["input_trace"]["stages"]["forecast"]["output"]["value"] = 2
    _resign(pair)

    with pytest.raises(phase4_real._TraceError, match="content hash mismatch"):
        phase4_real._verified_trace_bundle(pair, tmp_path)


def test_sidecar_byte_hash_mismatch_is_incomparable(tmp_path):
    pair = _pair(tmp_path)
    (tmp_path / "residuals.json").write_text("[]")

    with pytest.raises(phase4_real._TraceError, match="sha256: mismatch"):
        phase4_real._verified_trace_bundle(pair, tmp_path)


def test_trace_free_release_never_executes_or_completes(tmp_path, monkeypatch):
    pair = _pair(tmp_path)
    del pair["payload"]["input_trace"]
    del pair["payload"]["input_trace_hash"]
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("score_one must not run")

    monkeypatch.setattr(phase4_real.application, "score_one", forbidden)
    corpus = Corpus(
        root=tmp_path,
        index={"corpus_hash": content_hash({"release": 1})},
        pairs={"pair-1": pair},
    )

    release, parity = phase4_real._native_parity(corpus)

    assert called is False
    assert release["complete"] is False
    assert release["population"]["compared"] == 0
    assert release["population"]["incomparable"] == 1
    assert parity["same_input_hashes"] is False
    assert parity["stages"] == ()
    assert parity["complete"] is False


def test_verified_bundle_is_passed_to_canonical_execution(tmp_path, monkeypatch):
    pair = _pair(tmp_path)
    seen = []

    def record_call(request, inputs):
        seen.append((request, inputs))
        raise RuntimeError("stop after boundary assertion")

    monkeypatch.setattr(phase4_real.application, "score_one", record_call)
    corpus = Corpus(
        root=tmp_path,
        index={"corpus_hash": content_hash({"release": 1})},
        pairs={"pair-1": pair},
    )

    release, parity = phase4_real._native_parity(corpus)

    assert len(seen) == 1
    assert seen[0][0].event_id == "event-1"
    assert seen[0][1].source_ref == pair["payload"]["legacy_input_hash"]
    assert release["population"]["incomparable"] == 1
    assert parity["complete"] is False
