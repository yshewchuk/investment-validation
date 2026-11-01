# packages: engine.v2.scoring, engine.v2.models, engine.v2.foundation, engine.v2.ops
"""Frozen chooser replay refuses folds without causal producer provenance."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from checks import phase4_real, phase5_phase4_replay
from engine.v2 import foundation
from engine.v2.models import deployment, loader, serving_fold_loader
from engine.v2.models.serving_folds import ChooserServingFoldDescriptor, ServingFoldDescriptor


@pytest.mark.parametrize(
    "provenance",
    [
        None,
        {"pred_im_t1_d14": {"training_cutoff": "2026-10-27", "fold_start": "2026-10-01",
                             "descriptor_hash": "descriptor-implied"},
         "pred_runup_abs_move_d14": {"training_cutoff": "2026-10-27", "fold_start": "2026-10-01",
                                      "descriptor_hash": "descriptor-runup"}},
        {"pred_im_t1_d14": {"training_cutoff": "2026-10-28", "fold_start": "2026-10-01",
                             "descriptor_hash": "descriptor-implied"},
         "pred_runup_abs_move_d14": {"training_cutoff": "2026-10-28", "fold_start": "2026-10-01",
                                      "descriptor_hash": "descriptor-runup"}},
        {"pred_im_t1_d14": {"training_cutoff": "2026-10-01", "fold_start": "2026-10-01",
                             "descriptor_hash": "descriptor-implied"},
         "pred_runup_abs_move_d14": {"training_cutoff": "2026-10-01", "fold_start": "2026-10-01",
                                      "descriptor_hash": "descriptor-runup"}},
        {"pred_im_t1_d14": {"training_cutoff": "2026-10-26", "fold_start": "2026-11-01",
                             "descriptor_hash": "descriptor-implied"},
         "pred_runup_abs_move_d14": {"training_cutoff": "2026-10-26", "fold_start": "2026-11-01",
                                      "descriptor_hash": "descriptor-runup"}},
    ],
)
def test_frozen_replay_refuses_missing_or_noncausal_fold_provenance(monkeypatch, provenance):
    recipe = {"binding_id": "chooser"}
    if provenance is not None:
        recipe["fold_provenance"] = provenance
        recipe["artifact_refs"] = ["descriptor-implied", "descriptor-runup"]
    chooser = SimpleNamespace(recipe=recipe)
    request = SimpleNamespace(
        requested_decision_at="2026-10-27", strategy_version="TWIN-P")
    verified = {
        "request": request,
        "legacy_event_date": "2026-10-27",
        "inputs": SimpleNamespace(gate={"mode": "chooser"}, chooser=None),
        "frozen_replay": None,
        "frozen_chooser": chooser,
    }
    monkeypatch.setattr(
        phase4_real, "_verified_trace_bundle", lambda pair, corpus_root: verified)

    staged = {}
    if provenance is not None:
        for output, item in provenance.items():
            staged[f"__chooser_descriptor__:{item['descriptor_hash']}"] = (
                f"{output}\x1f{item['training_cutoff']}\x1f{item['fold_start']}")
    result = phase5_phase4_replay._replay_pair(
        {"payload": {"input_trace": {"trace_hash": "sha256:trace"}}},
        corpus_root=None, release_root=None, staged=staged)

    assert result["disposition"] == "member_absent"
    assert result["code"] == phase5_phase4_replay.PHASE4_MEMBER_ABSENT
    assert "chooser:fold_provenance:unverifiable" in result["detail"]


def test_frozen_replay_requires_provenance_bound_to_staged_descriptors():
    provenance = {
        "pred_im_t1_d14": {
            "training_cutoff": "2026-08-31", "fold_start": "2026-09-01",
            "descriptor_hash": "descriptor-implied"},
        "pred_runup_abs_move_d14": {
            "training_cutoff": "2026-08-31", "fold_start": "2026-09-01",
            "descriptor_hash": "descriptor-runup"},
    }
    recipe = {
        "fold_provenance": provenance,
        "artifact_refs": ["descriptor-implied", "descriptor-runup"],
    }
    request = SimpleNamespace(requested_decision_at="2026-10-27")
    legacy_event_date = "2026-09-15"
    staged = {
        "__chooser_descriptor__:descriptor-implied":
            "pred_im_t1_d14\x1f2026-08-31\x1f2026-09-01",
        "__chooser_descriptor__:descriptor-runup":
            "pred_runup_abs_move_d14\x1f2026-08-31\x1f2026-09-01",
    }

    assert phase5_phase4_replay._has_causal_chooser_fold_provenance(
        recipe, request, staged, legacy_event_date)
    assert not phase5_phase4_replay._has_causal_chooser_fold_provenance(
        recipe, request, {}, legacy_event_date)
    mismatched = dict(staged)
    mismatched["__chooser_descriptor__:descriptor-implied"] = (
        "pred_im_t1_d14\x1f2026-08-01\x1f2026-09-01")
    assert not phase5_phase4_replay._has_causal_chooser_fold_provenance(
        recipe, request, mismatched, legacy_event_date)

    later_provenance = {
        output: {**item, "fold_start": "2026-10-01"}
        for output, item in provenance.items()
    }
    later_recipe = {**recipe, "fold_provenance": later_provenance}
    later_staged = {
        f"__chooser_descriptor__:{item['descriptor_hash']}":
            f"{output}\x1f{item['training_cutoff']}\x1f{item['fold_start']}"
        for output, item in later_provenance.items()
    }
    assert not phase5_phase4_replay._has_causal_chooser_fold_provenance(
        later_recipe, request, later_staged, legacy_event_date)


def test_replay_pool_values_must_match_staged_fold_estimators(monkeypatch, tmp_path):
    from engine.v2.scoring import release_bindings as rb

    outputs = ("pred_im_t1_d14", "pred_runup_abs_move_d14")
    descriptors, refs, provenance, rows = {}, {}, {}, []
    pools = {}
    for index, output in enumerate(outputs):
        digest = f"hash:{output}"
        descriptors[digest] = SimpleNamespace(
            descriptor_hash=digest, producer_output=output,
            training_cutoff="2026-08-31", fold_start="2026-09-01",
            parent_release_id="parent",
            policy=SimpleNamespace(interval_floor=0.25))
        refs[output] = SimpleNamespace(
            pool_pred=np.array([index + 1.0, index + 2.0]),
            pool_res=np.array([index + 3.0, index + 4.0]))
        provenance[output] = {
            "training_cutoff": "2026-08-31", "fold_start": "2026-09-01",
            "descriptor_hash": digest}
        rows.append({"member_id": f"tier4_folds:{'implied_t1' if index == 0 else 'runup_move'}",
                     "status": "STAGED",
                     "objects": [{"serving_fold": {"descriptor_hash": digest,
                                                    "fold_start": "2026-09-01"}}]})
        pools[output] = {
            "predictions": refs[output].pool_pred.copy(),
            "residuals": refs[output].pool_res.copy(),
            "interval_floor": 0.25}

    references = {
        (descriptor.producer_output, "2026-09-01"): SimpleNamespace(
            descriptor_hash=descriptor.descriptor_hash,
            release_id="parent",
            carrying_release_id="carrying",
            training_cutoff="2026-09-01",
        )
        for descriptor in descriptors.values()
    }
    contexts = {
        descriptor.descriptor_hash: (
            SimpleNamespace(release=SimpleNamespace(release_id="parent")), {})
        for descriptor in descriptors.values()
    }

    def make_binding(output):
        return SimpleNamespace(
            binding_id=f"binding:{output}", role="size", model_id=f"model:{output}",
            strategy_id="*", feature_order=("feature_a",), decision_clock_id="clock",
            output_names=(output,),
            members=(SimpleNamespace(content_hash=f"member:{output}"),))

    alias_bindings = []
    producers = {}
    for output in outputs:
        alias_bindings.append(make_binding(output))
        refs[output].release = SimpleNamespace(bindings=(make_binding(output),))
        producers[output] = {"binding_id": f"binding:{output}", "output": output}

    size_descriptor = SimpleNamespace(
        descriptor_hash="size", parent_release_id="parent",
        fold_start="2026-09-01",
        estimator=SimpleNamespace(path="size.joblib", content_hash="hash:size"),
        policy=SimpleNamespace(interval_floor=0.4))
    size_ref = SimpleNamespace(
        pool_pred=np.array([10.0, 11.0]), pool_res=np.array([12.0, 13.0]))
    descriptors["size"] = size_descriptor
    rows.append({"member_id": "tier4_folds:size", "status": "STAGED", "objects": [
        {"serving_fold": {"descriptor_hash": "size"},
         "path": "size.joblib", "content_hash": "hash:size", "bytes": 123}]})
    size_like = [size_descriptor]
    pools["pred_abs_move"] = {
        "predictions": size_ref.pool_pred.copy(),
        "residuals": size_ref.pool_res.copy(),
        "interval_floor": 0.4}

    def decode(kind, doc):
        descriptor = descriptors[doc["descriptor_hash"]]
        is_size = any(descriptor is candidate for candidate in size_like)
        if kind is ChooserServingFoldDescriptor and is_size:
            raise ValueError("size fold must not be parsed as chooser fold")
        if kind is ServingFoldDescriptor and not is_size:
            raise ValueError("chooser fold must not be parsed as size fold")
        return descriptor

    seen_expected_bytes = []

    def fake_load_serving_fold(*_args, **kwargs):
        seen_expected_bytes.append(kwargs.get("expected_bytes"))
        return size_ref

    monkeypatch.setattr(foundation, "from_document", decode)
    monkeypatch.setattr(phase5_phase4_replay, "deployment_root", lambda root: root)
    monkeypatch.setattr(deployment, "read_manifest", lambda _root, _release: object())
    monkeypatch.setattr(loader, "FrozenInference", lambda _root: object())
    monkeypatch.setattr(serving_fold_loader, "validate_serving_fold_descriptor",
                        lambda *_args: None)
    monkeypatch.setattr(serving_fold_loader, "validate_serving_fold_object", lambda *_args: None)
    monkeypatch.setattr(serving_fold_loader, "load_serving_fold", fake_load_serving_fold)
    monkeypatch.setattr(
        serving_fold_loader, "load_chooser_serving_fold",
        lambda _inference, descriptor, **_kwargs: refs[descriptor.producer_output])
    chooser = SimpleNamespace(
        recipe={"fold_provenance": provenance, "producers": producers,
                "artifact_refs": ["hash:size", *(f"hash:{output}" for output in outputs)]},
        fold_pools=pools,
        release=SimpleNamespace(bindings=tuple(alias_bindings)))
    manifest = {"manifest_hash": "sha256:catalog", "release_id": "carrying",
                "members": rows}

    monkeypatch.setattr(
        rb, "resolve_chooser_fold_authoring_contexts",
        lambda *_args: (references, contexts))

    request = SimpleNamespace(requested_decision_at="2026-10-27")
    legacy_event_date = "2026-09-15"
    size_descriptor.fold_start = "2026-10-01"
    assert phase5_phase4_replay._staged_chooser_pools(
        chooser, request, tmp_path, manifest,
        legacy_event_date=legacy_event_date) is None
    size_descriptor.fold_start = "2026-09-01"

    unselected_descriptor = SimpleNamespace(
        descriptor_hash="size-unselected", parent_release_id="parent",
        fold_start="2026-10-01",
        estimator=SimpleNamespace(path="unselected.joblib", content_hash="hash:unselected"),
        policy=SimpleNamespace(interval_floor=0.4))
    descriptors["size-unselected"] = unselected_descriptor
    size_like.append(unselected_descriptor)
    rows[-1]["objects"].append({
        "serving_fold": {"descriptor_hash": "size-unselected"},
        "path": "unselected.joblib", "content_hash": "hash:unselected",
        "bytes": "not-an-int"})
    assert phase5_phase4_replay._staged_chooser_pools(
        chooser, request, tmp_path, manifest,
        legacy_event_date=legacy_event_date) is None
    rows[-1]["objects"].pop()

    pools["pred_abs_move"]["predictions"] = np.array([99.0, 99.0])
    assert phase5_phase4_replay._staged_chooser_pools(
        chooser, request, tmp_path, manifest,
        legacy_event_date=legacy_event_date) is None

    pools["pred_abs_move"]["predictions"] = size_ref.pool_pred.copy()
    seen_expected_bytes.clear()
    verified = phase5_phase4_replay._staged_chooser_pools(
        chooser, request, tmp_path, manifest,
        legacy_event_date=legacy_event_date)
    assert verified is not None
    for output, ref in refs.items():
        assert np.array_equal(verified[output]["predictions"], ref.pool_pred)
    assert np.array_equal(verified["pred_abs_move"]["predictions"], size_ref.pool_pred)
    assert np.array_equal(verified["pred_abs_move"]["residuals"], size_ref.pool_res)
    assert seen_expected_bytes == [123]


def test_replay_checkpoint_scope_invalidation(tmp_path, monkeypatch):
    corpus_hash = "sha256:corpus"
    manifest = {"manifest_hash": "sha256:catalog", "members": []}
    other_manifest = {"manifest_hash": "sha256:other", "members": []}
    scope = phase5_phase4_replay._checkpoint_scope(corpus_hash, manifest)
    other_scope = phase5_phase4_replay._checkpoint_scope(corpus_hash, other_manifest)
    assert phase5_phase4_replay.REPLAY_VALIDATION_VERSION == 3
    assert scope != other_scope

    def scoped_row(row_scope):
        row = {"fixture_id": "fixture-1", "disposition": "replayed",
               "corpus_hash": corpus_hash, "checkpoint_scope": row_scope}
        row["checkpoint_row_hash"] = phase5_phase4_replay._checkpoint_row_hash(
            row, row_scope)
        return row

    checkpoint = tmp_path / "checkpoint.jsonl"
    valid = scoped_row(scope)
    checkpoint.write_text(json.dumps(valid) + "\n")
    loaded = phase5_phase4_replay._load_checkpoint(checkpoint, scope)
    assert set(loaded) == {"fixture-1"}
    assert loaded["fixture-1"]["corpus_hash"] == corpus_hash
    assert loaded["fixture-1"]["disposition"] == "replayed"

    tampered = {**valid, "disposition": "member_absent"}
    checkpoint.write_text(json.dumps(tampered) + "\n")
    assert phase5_phase4_replay._load_checkpoint(checkpoint, scope) == {}

    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(json.dumps({
        "fixture_id": "fixture-1", "disposition": "replayed",
        "corpus_hash": corpus_hash}) + "\n")
    assert phase5_phase4_replay._load_checkpoint(legacy, scope) == {}

    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps(scoped_row(other_scope)) + "\n")
    assert phase5_phase4_replay._load_checkpoint(other, scope) == {}

    corrupt = tmp_path / "corrupt.jsonl"
    corrupt.write_text("{not json}\n")
    assert phase5_phase4_replay._load_checkpoint(corrupt, scope) == {}

    oversized = tmp_path / "oversized.jsonl"
    oversized.write_bytes(b"x" * 33)
    monkeypatch.setattr(phase5_phase4_replay, "MAX_CHECKPOINT_BYTES", 32)
    assert phase5_phase4_replay._load_checkpoint(oversized, scope) == {}
