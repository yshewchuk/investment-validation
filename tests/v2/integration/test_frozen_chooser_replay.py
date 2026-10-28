# packages: engine.v2.scoring, engine.v2.models, engine.v2.ops
"""Frozen chooser replay refuses folds without causal producer provenance."""
from types import SimpleNamespace

import pytest

from checks import phase4_real, phase5_phase4_replay


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

    result = phase5_phase4_replay._replay_pair(
        {"payload": {"input_trace": {"trace_hash": "sha256:trace"}}},
        corpus_root=None, release_root=None, staged={})

    assert result["disposition"] == "member_absent"
    assert result["code"] == phase5_phase4_replay.PHASE4_MEMBER_ABSENT
    assert "chooser:fold_provenance:unverifiable" in result["detail"]


def test_frozen_replay_accepts_canonical_causal_cutoffs():
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

    assert phase5_phase4_replay._has_causal_chooser_fold_provenance(recipe, request)
