"""P5-6 Phase 4 replay on a synthetic tmp_path corpus.

The traced pair is captured by Phase 4's own tooling: artifacts packaged by
``tools.phase4_frozen_resources``, the frozen plan built by
``checks.phase4_frozen_bridge``, scored and traced by
``tools.capture_tier0_corpus.package_strict_trace`` (``assemble_input_trace``)
and written with ``make_pair``. The replay then serves the same bindings from
the staged P5 release. No real data is read.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

from checks import phase4_real
from checks import phase5_phase4_replay as replay
from checks.phase4_frozen_bridge import prepare_frozen_replay
from engine.v2.contracts import ScoreRequest
from engine.v2.foundation import content_hash, to_document
from engine.v2.scoring.stages import NativeScoreInputs, receipt
from tests.test_checks_phase5_acceptance import CLOCK, _linear, _release, _run
from tools.capture_tier0_corpus import make_pair, package_strict_trace
from tools.phase4_frozen_resources import package_frozen_resources
from tools.phase4_request_translation import canonical_request_from_legacy

_STAGES = ("resolve_context", "features", "forecast", "geometry", "pricing",
           "analogs", "simulation", "gate", "chooser", "serialization")


def _inputs(request) -> tuple[NativeScoreInputs, dict]:
    blocks = {
        "context": {"strategy": "STR-THRU"},
        "features": {"model_inputs": {"x": 3.0}},
        "forecast": {}, "geometry": None, "pricing": None, "analogs": {},
        "simulation": {}, "gate": {}, "chooser": {}, "diagnostics": {},
    }
    shared = {"request": to_document(request), "native_inputs": blocks}
    source_ref = content_hash(shared)
    declared = tuple(receipt(stage, {"source_ref": source_ref}, {"execution": "native-runtime"})
                     for stage in _STAGES)
    return NativeScoreInputs(**blocks, source_ref=source_ref, stage_receipts=declared), shared


def _corpus(tmp_path: Path, *, artifact: bytes | None = None) -> Path:
    """One traced STR-THRU pair whose frozen ``size`` binding reads ``artifact``.

    By default the artifact is byte-identical to the staged release's
    ``size:*`` member (the fixture release's first json-linear model).
    """
    root, source = tmp_path / "corpus", tmp_path / "corpus-src"
    source.mkdir(parents=True)
    data = _linear(1.0) if artifact is None else artifact
    (source / "size.json").write_bytes(data)
    package = package_frozen_resources(
        model_bindings=[{
            "model_id": "m-size-all", "artifact": "size.json",
            "artifact_sha256": hashlib.sha256(data).hexdigest(), "role": "size",
            "feature_order": ["x"], "output_names": ["y"], "strategy": "STR-THRU",
            "decision_clock": CLOCK, "adapter": "json-linear.v1",
        }],
        deployment_id="dep", release_root=root, source_root=source)
    # payload.request is the legacy request; the traced V2 request is its
    # translation, with the capture environment's deployment, clock and refs.
    legacy_request = {"ticker": "ABC", "strategy": "STR-THRU", "as_of": "2026-09-16",
                      "event_date": "2026-09-17", "session": "AMC",
                      "fill": {"policy_id": "legacy.fill_alpha.v1", "alpha": 0.5}}
    request = replace(
        canonical_request_from_legacy(legacy_request, event_id="event-1",
                                      snapshot="snapshot-1"),
        deployment_id="dep", decision_clock_id=CLOCK,
        model_artifact_refs=package.request_refs)
    assert isinstance(request, ScoreRequest)
    inputs, shared = _inputs(request)
    resources = list(package.resource_rows)
    resources += [{"resource_id": b["binding_id"], "ref": b["request_ref"], "kind": "sidecar",
                   "document": b, "content_hash": content_hash(b)}
                  for b in package.sidecar_document["bindings"]]
    metadata = {"frozen_inference": package.trace_declaration}
    plan = prepare_frozen_replay(
        release_root=root, resource_rows=resources,
        verified_documents=phase4_real._verified_resources(root, resources),
        metadata=metadata, request=request, inputs=inputs)
    trace, native = package_strict_trace(
        request, inputs, shared, resources=resources, metadata=metadata,
        frozen_runtime=(plan.inference, plan.release, plan.requests))
    pair = make_pair("pair-1", ["frozen"], legacy_request, {"score_id": native.score_id},
                     record_kind="score_result", duration=0.0, input_trace=trace,
                     legacy_input_hash=trace["shared_input_hash"])
    (root / "pairs").mkdir()
    (root / "pairs" / "pair-1.json").write_text(json.dumps(pair, sort_keys=True))
    (root / "INDEX.json").write_text(json.dumps({"corpus_hash": content_hash({"p": 1})}))
    return root


def test_traced_pair_replays_from_the_staged_release(tmp_path):
    corpus = _corpus(tmp_path)
    evidence = _run(tmp_path, _release(tmp_path), phase4_corpus=corpus)

    assert evidence["findings"] == []
    assert evidence["status"] == "PASS"
    phase4 = evidence["phase4"]
    assert phase4["dispositions"]["replayed"] == 1
    assert phase4["members_exercised"] == ["model:size:*"]
    assert phase4["fit_attempts"] == 0
    assert "Phase 4 replay" in Path(evidence["report"]).read_text()


def test_replay_reads_the_staged_bytes_not_the_corpus_copy(tmp_path):
    from checks.phase5_release import deployment_root, object_relpath

    corpus = _corpus(tmp_path)
    root = _release(tmp_path)
    staged = deployment_root(root) / object_relpath(
        "sha256:" + hashlib.sha256(_linear(1.0)).hexdigest())
    staged.write_bytes(_linear(9.0))  # the corpus copy is untouched
    evidence = _run(tmp_path, root, phase4_corpus=corpus)

    assert "P5_MEMBER_HASH_MISMATCH" in evidence["finding_codes"]
    assert evidence["phase4"]["dispositions"]["replayed"] == 0
    assert {"P5_PHASE4_MISMATCH", "P5_PHASE4_ERROR"} & set(evidence["finding_codes"])


def test_rebound_plan_points_only_at_staged_objects(tmp_path):
    from checks.phase5_release import deployment_root, read_manifest
    from engine.v2.models.deployment import resolve_release

    corpus = _corpus(tmp_path)
    root = _release(tmp_path)
    loaded = phase4_real.load(corpus)
    verified = phase4_real._verified_trace_bundle(loaded.pairs["pair-1"], loaded.root)
    manifest = read_manifest(root)
    staged = replay.staged_model_objects(
        resolve_release(deployment_root(root), manifest["release_id"]), manifest)
    rebound, absent, used = replay.rebind_to_release(verified["frozen_replay"], root, staged)

    assert absent == [] and used == ["model:size:*"]
    member = rebound.release.bindings[0].members[0]
    assert (deployment_root(root) / member.path).read_bytes() == _linear(1.0)
    # the captured contract is kept exactly; only the member path moved
    captured = verified["frozen_replay"].release.bindings[0]
    assert rebound.release.release_id == verified["frozen_replay"].release.release_id
    assert (rebound.release.bindings[0].binding_id, rebound.release.bindings[0].output_names) == (
        captured.binding_id, captured.output_names)


def test_trace_model_not_in_the_staged_release_is_member_absent(tmp_path):
    corpus = _corpus(tmp_path, artifact=_linear(42.0))
    evidence = _run(tmp_path, _release(tmp_path), phase4_corpus=corpus)

    assert "P5_PHASE4_MEMBER_ABSENT" in evidence["finding_codes"]
    assert evidence["phase4"]["dispositions"]["member_absent"] == 1
    assert evidence["status"] == "FAIL"


def test_runtime_drift_from_the_capture_is_a_mismatch(tmp_path, monkeypatch):
    """Same staged bytes, drifted inference runtime: the receipts must differ."""
    from engine.v2.models.adapters import JsonLinearAdapter

    corpus = _corpus(tmp_path)
    real = JsonLinearAdapter.predict

    def drifted(self, artifact, rows, binding):
        return [tuple(value + 1.0 for value in row)
                for row in real(self, artifact, rows, binding)]

    monkeypatch.setattr(JsonLinearAdapter, "predict", drifted)
    evidence = _run(tmp_path, _release(tmp_path), phase4_corpus=corpus)

    mismatch = [f for f in evidence["findings"] if f["code"] == "P5_PHASE4_MISMATCH"]
    assert [f["subject"] for f in mismatch] == ["phase4:pair-1"]
    assert "execution." in mismatch[0]["detail"]


def test_fit_during_replay_fails_the_phase4_subject(tmp_path, monkeypatch):
    from engine.v2.models import no_fit
    from engine.v2.scoring import application

    corpus = _corpus(tmp_path)
    real = application.score_frozen

    def fitting(*args, **kwargs):
        try:
            no_fit.forbid_fitting("p5-6.replay_fit")
        except Exception:  # noqa: BLE001 -- swallowed on purpose
            pass
        return real(*args, **kwargs)

    monkeypatch.setattr(application, "score_frozen", fitting)
    evidence = _run(tmp_path, _release(tmp_path), phase4_corpus=corpus)

    assert {"code": "P5_RUNTIME_FIT", "subject": "phase4_replay",
            "detail": "p5-6.replay_fit"} in evidence["findings"]
    assert evidence["phase4"]["status"] == "FAIL"


def test_corpus_with_no_traced_pair_is_empty(tmp_path):
    root = tmp_path / "corpus"
    (root / "pairs").mkdir(parents=True)
    pair = make_pair("pair-0", ["x"], {"event_id": "e"}, {}, record_kind="score_result",
                     duration=0.0)
    (root / "pairs" / "pair-0.json").write_text(json.dumps(pair))
    (root / "INDEX.json").write_text(json.dumps({}))
    evidence = _run(tmp_path, _release(tmp_path), phase4_corpus=root)

    assert "P5_PHASE4_EMPTY" in evidence["finding_codes"]
    assert evidence["phase4"]["dispositions"]["untraced"] == 1
