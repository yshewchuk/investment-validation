"""One narrow legacy scoring seam used while native stages are extracted."""
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from engine.v2.contracts import ScoreRequest

__all__ = ["score_legacy_request"]


def score_legacy_request(request: ScoreRequest, legacy_fields: Mapping[str, Any]):
    """Run the frozen scorer with explicit request fields.

    This adapter is intentionally small: no model fitting, experiment import,
    renderer arithmetic or mutable registry lookup is exposed to callers.
    Phase 5 replaces the backend while the canonical application stays stable.
    """
    from engine.fills import FillModel
    from engine.score import Scorer
    from engine.score import ScoreRequest as LegacyScoreRequest

    fill = request.fill_model
    alpha = float(fill.get("alpha", 0.5))
    legacy_request = LegacyScoreRequest(
        ticker=str(legacy_fields["ticker"]),
        strategy=request.strategy_version.split("@", 1)[0],
        as_of=pd.Timestamp(legacy_fields["as_of"]) if legacy_fields.get("as_of") else None,
        event_date=pd.Timestamp(legacy_fields["event_date"]),
        strike=legacy_fields.get("strike"),
        expiry=pd.Timestamp(legacy_fields["expiry"]) if legacy_fields.get("expiry") else None,
        fill=FillModel(alpha),
        session=legacy_fields.get("session"),
        decision_offset=legacy_fields.get("decision_offset"),
        quote_max_age_sessions=legacy_fields.get("quote_max_age_sessions"),
        chain_as_of=pd.Timestamp(legacy_fields["chain_as_of"])
        if legacy_fields.get("chain_as_of") else None,
        structure_params=legacy_fields.get("structure_params"),
    )
    scorer = legacy_fields.get("scorer") or Scorer()
    return scorer.score(legacy_request)
