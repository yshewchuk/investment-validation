"""Allowlisted worker names for the single legacy adapter."""
from __future__ import annotations

from engine.v2.ops.legacy_adapter import legacy_action

ACTION_NAMES = (
    "legacy_finality", "legacy_features", "legacy_score", "legacy_decisions",
    "legacy_settlement", "legacy_model_evidence", "legacy_render",
    "legacy_selfcheck", "legacy_score_requests", "legacy_decision_replay",
)


def run_action(action, parameters, staging, legacy_root=None, cross_check=None):
    if action not in ACTION_NAMES:
        raise ValueError("legacy action is not allowlisted")
    return legacy_action(action, parameters, staging, legacy_root=legacy_root,
                         cross_check=cross_check)
