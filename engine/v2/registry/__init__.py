"""Strategy registry StrategySpec and DeploymentSpec

Layer 3 of `system_rearchitecture.md` §4.1. Replaces `structure_registry.py`, `the StrategySpec/DeploymentSpec store`.

Phase 4 frozen strategy inventory and deployment bindings.
"""

from engine.v2.registry.strategies import (
    DYNAMIC_MENU,
    STRATEGY_IDS,
    StrategyRegistry,
    default_registry,
)

__all__ = ["DYNAMIC_MENU", "STRATEGY_IDS", "StrategyRegistry", "default_registry"]
