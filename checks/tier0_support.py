"""Dependency-light Tier-0 defaults and synthetic negative-control helpers.

Corpus loading and process orchestration belong to ``checks.tier0_corpus``.
"""
from __future__ import annotations

import copy
from pathlib import Path

from engine.v2.diagnosis import ComparisonReceipt

# Preserve the runner's checkout-relative default, independently of cwd.
DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "tier0"


def round_params(record: dict) -> dict:
    """``structure_params`` rounded to six places — the `json_safe` defect."""
    out = copy.deepcopy(record)
    params = out.get("structure_params")
    if isinstance(params, dict):
        out["structure_params"] = {k: round(v, 6) if isinstance(v, float) else v
                                   for k, v in params.items()}
    return out


def finding_dicts(receipt: ComparisonReceipt) -> list[dict]:
    return [{"first_differing_stage": f.first_differing_stage,
             "field_path": f.field_path, "kind": f.kind}
            for f in receipt.findings]
