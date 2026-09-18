"""Scoring application forecasts, shape, pricing, gate/chooser decisions, diagnostics

Layer 5 of `system_rearchitecture.md` §4.1. Replaces `score.py split by the stages in §6.3`, `entry_rules.py`, `replay.py`, `trailing_cutoff from pnl_sim.py`.

Phase 4 shared scoring application. Numerical execution remains behind the
declared compatibility seam until the Phase 5 inference artifacts are ready.
"""

from engine.v2.domain.generation import Geometry as _Geometry  # noqa: F401
from engine.v2.domain.generation import Pricing as _Pricing  # noqa: F401
from engine.v2.scoring.application import (
    replay,
    score_batch,
    score_event,
    score_frozen,
    score_many,
    score_one,
)
from engine.v2.scoring.financial import financial_diagnostics
from engine.v2.scoring.frozen_executor import (
    FrozenStageExecutor,
    FrozenStageRefusal,
    FrozenStageResult,
)
from engine.v2.scoring.identity import (
    canonical_request,
    dependency_hash,
    request_hash,
    score_id,
)
from engine.v2.scoring.source_inputs import SourceBundle, build_native_score_inputs
from engine.v2.scoring.stages import STAGE_NAMES, NativeScoreInputs, StageReceipt

__all__ = [
    "canonical_request", "dependency_hash", "financial_diagnostics",
    "NativeScoreInputs", "STAGE_NAMES", "StageReceipt",
    "SourceBundle", "build_native_score_inputs",
    "FrozenStageExecutor", "FrozenStageRefusal", "FrozenStageResult",
    "replay", "request_hash", "score_batch", "score_event", "score_frozen", "score_id", "score_many", "score_one",
]
