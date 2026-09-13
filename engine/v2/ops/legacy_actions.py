"""Allowlisted worker names for the single legacy adapter."""
from __future__ import annotations

from engine.v2.ops.legacy_adapter import legacy_action

ACTION_NAMES = (
    "legacy_finality", "legacy_score", "legacy_decisions", "legacy_settlement",
    "legacy_model_evidence", "legacy_render", "legacy_selfcheck",
    "legacy_score_requests",
)


def run_action(action, parameters, staging):
    if action not in ACTION_NAMES:
        raise ValueError("legacy action is not allowlisted")
    return legacy_action(action, parameters, staging)
