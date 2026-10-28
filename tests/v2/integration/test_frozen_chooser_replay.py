# packages: engine.v2.scoring, engine.v2.models, engine.v2.foundation, engine.v2.ops
"""Frozen chooser replay refuses folds without causal producer provenance."""
from types import SimpleNamespace

import numpy as np
import pytest

from checks import phase4_real, phase5_phase4_replay
from engine.v2 import foundation
from engine.v2.models import deployment, loader, serving_fold_loader


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
            "training_cutoff": "2026-09-30", "fold_start": "2026-10-01",
            "descriptor_hash": "descriptor-implied"},
        "pred_runup_abs_move_d14": {
            "training_cutoff": "2026-09-30", "fold_start": "2026-10-01",
            "descriptor_hash": "descriptor-runup"},
    }
    recipe = {
        "fold_provenance": provenance,
        "artifact_refs": ["descriptor-implied", "descriptor-runup"],
    }
    request = SimpleNamespace(requested_decision_at="2026-10-27")
    staged = {
        "__chooser_descriptor__:descriptor-implied":
            "pred_im_t1_d14\x1f2026-09-30\x1f2026-10-01",
        "__chooser_descriptor__:descriptor-runup":
            "pred_runup_abs_move_d14\x1f2026-09-30\x1f2026-10-01",
    }

    assert phase5_phase4_replay._has_causal_chooser_fold_provenance(
        recipe, request, staged)
    assert not phase5_phase4_replay._has_causal_chooser_fold_provenance(
        recipe, request, {})
    mismatched = dict(staged)
    mismatched["__chooser_descriptor__:descriptor-implied"] = (
        "pred_im_t1_d14\x1f2026-10-01\x1f2026-10-01")
    assert not phase5_phase4_replay._has_causal_chooser_fold_provenance(
        recipe, request, mismatched)


def test_replay_pool_values_must_match_staged_fold_estimators(monkeypatch, tmp_path):
    outputs = ("pred_im_t1_d14", "pred_runup_abs_move_d14")
    descriptors, refs, provenance, rows = {}, {}, {}, []
    pools = {}
    for index, output in enumerate(outputs):
        digest = f"hash:{output}"
        descriptors[digest] = SimpleNamespace(
            descriptor_hash=digest, producer_output=output,
            training_cutoff="2026-09-30", fold_start="2026-10-01",
            parent_release_id="parent",
            policy=SimpleNamespace(interval_floor=0.25))
        refs[output] = SimpleNamespace(
            pool_pred=np.array([index + 1.0, index + 2.0]),
            pool_res=np.array([index + 3.0, index + 4.0]))
        provenance[output] = {
            "training_cutoff": "2026-09-30", "fold_start": "2026-10-01",
            "descriptor_hash": digest}
        rows.append({"member_id": f"tier4_folds:{output}", "objects": [
            {"serving_fold": {"descriptor_hash": digest}}]})
        pools[output] = {
            "predictions": np.array([99.0, 99.0]),
            "residuals": refs[output].pool_res,
            "interval_floor": 0.25}
    monkeypatch.setattr(foundation, "from_document",
                        lambda _kind, doc: descriptors[doc["descriptor_hash"]])
    monkeypatch.setattr(phase5_phase4_replay, "deployment_root", lambda root: root)
    monkeypatch.setattr(deployment, "read_manifest", lambda _root, _release: object())
    monkeypatch.setattr(loader, "FrozenInference", lambda _root: object())
    monkeypatch.setattr(
        serving_fold_loader, "load_chooser_serving_fold",
        lambda _inference, descriptor, **_kwargs: refs[descriptor.producer_output])
    chooser = SimpleNamespace(
        recipe={"fold_provenance": provenance}, fold_pools=pools)
    manifest = {"members": rows}

    assert phase5_phase4_replay._staged_chooser_pools(
        chooser, tmp_path, manifest) is None
    for output, ref in refs.items():
        pools[output]["predictions"] = ref.pool_pred.copy()
    verified = phase5_phase4_replay._staged_chooser_pools(
        chooser, tmp_path, manifest)
    assert verified is not None
    for output, ref in refs.items():
        assert np.array_equal(verified[output]["predictions"], ref.pool_pred)
