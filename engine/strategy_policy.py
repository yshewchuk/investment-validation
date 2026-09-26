"""The board's disabled-strategy policy — no dependency on legacy pricing.

Extracted from ``engine.score`` so a caller that only needs to know WHICH
strategies the board refuses to score (never HOW a strategy prices) does not
have to import ``engine.score``, whose own top-level import block pulls in
``engine.replay`` -> ``engine.fills`` (the legacy chain index and fill
model). ``engine.score`` re-exports this dict unchanged
(``from engine.strategy_policy import DISABLED_STRATEGIES``), so every
existing caller that reads ``engine.score.DISABLED_STRATEGIES`` keeps working
with no change.
"""
from __future__ import annotations

__all__ = ["DISABLED_STRATEGIES"]

#: Structures the scorer refuses to put a number on, and why.
DISABLED_STRATEGIES = {
    "CAL-P": (
        "The exact spec (put legs, entry shortly before the print with a ~1 DTE "
        "front, held THROUGH the print, both legs closed together after) has "
        "never been backtested. EXP-046b tested straddle legs at T-14 unwound "
        "pre-print — a different structure. Phase 2 backlog 1-2 must run first."
    ),
    "CND-P": (
        "EXP-121 registered and ran the risk-mechanics validation (defined-risk "
        "falsification, assignment exposure, the oracle ceiling) but nothing has "
        "reviewed its result or decided to promote the structure. There is no "
        "gate for it — the STR-THRU-shaped one the mechanics call for is not yet "
        "registered — and no fill/execution evidence beyond the replay. Added to "
        "STRUCTURES for engine.replay/build_trades so EXP-121 could price it; "
        "that registry is shared with the live board's default strategy list, "
        "which is what put it here before a single trade should ever be "
        "recommended off it."
    ),
}
