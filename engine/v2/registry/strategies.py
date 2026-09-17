"""Frozen Phase 4 strategy inventory and deployment bindings."""
from __future__ import annotations

from dataclasses import dataclass

from engine.v2.contracts import DeploymentSpec, StrategySpec
from engine.v2.foundation import content_hash

__all__ = ["DYNAMIC_MENU", "STRATEGY_IDS", "StrategyRegistry", "default_registry"]

STRATEGY_IDS = (
    "CAL-P", "STR-THRU", "STR-RUNUP", "CND-P", "TWIN-P", "TWIN-P5",
    "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5",
)
DYNAMIC_MENU = ("TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5")

_DISABLED = {
    "CAL-P": "UNVALIDATED_STRUCTURE",
    "CND-P": "UNVALIDATED_STRUCTURE",
}
_DIVISORS = {
    "TWIN-P": 1.5,
    "TWIN-P5": 1.0,
    "CND-PS": 2.0,
    "BFLY-P": 1.0,
    "BFLY-P5": 3.0,
    "RAMP7": 3.0,
    "CTR5": 2.0,
}


def _spec(strategy: str) -> StrategySpec:
    disabled = strategy in _DISABLED
    payload = {
        "strategy_id": strategy,
        "structure_recipe": f"legacy.structure.{strategy.lower()}.v1",
        "structure_parameters": {"forecast_width_divisor": _DIVISORS[strategy]}
        if strategy in _DIVISORS else {},
        "decision_clock": "legacy.entry_close.v1",
        "entry_policy": f"legacy.{strategy.lower()}.entry.v1",
        "exit_policy": f"legacy.{strategy.lower()}.exit.v1",
        "quote_policy": "legacy.chain_quote.v1",
        "fill_policy": "legacy.fill_model.v1",
        "universe_policy": "legacy.scoring_universe.v1",
        "domain_policy": "legacy.domain.v1",
        "feature_recipe_ids": (
            "legacy.market_context.v1", "legacy.event_history.v1",
            "legacy.bucket_analogs.v1", "legacy.calibration.v1",
        ),
        "model_role_bindings": {
            "gate": f"legacy.gate.{strategy.lower()}.v1",
        },
        "forecast_sizing_recipe": "legacy.forecast_sizing.v1"
        if strategy not in {"CAL-P", "CND-P"} else None,
        "analog_recipe": "legacy.bucket_analogs.v1",
        "payoff_recipe": "legacy.payoff.v1",
        "gate_recipe": f"legacy.entry_rule.{strategy.lower()}.v1"
        if strategy not in _DISABLED else None,
        "chooser_recipe": None,
        "fallback_policy": "legacy.refusal.v1",
        "refusal_codes": (_DISABLED[strategy],) if disabled else (),
    }
    definition_hash = content_hash(payload)
    return StrategySpec(
        strategy_version="legacy-phase4.v1",
        definition_hash=definition_hash,
        validation_status="disabled" if disabled else "historical",
        component_graph_ref=f"legacy.components.{strategy.lower()}.v1",
        evidence_refs=("legacy-source-inventory",),
        **payload,
    )


def _inventory() -> tuple[StrategySpec, ...]:
    return tuple(_spec(strategy) for strategy in STRATEGY_IDS)


@dataclass(frozen=True)
class StrategyRegistry:
    """Read-only strategy/deployment view captured at construction."""

    strategies: tuple[StrategySpec, ...]
    deployments: tuple[DeploymentSpec, ...]

    def strategy(self, strategy_id: str) -> StrategySpec:
        for spec in self.strategies:
            if spec.strategy_id == strategy_id:
                return spec
        if strategy_id == "DYN-SV":
            return _dynamic_spec()
        raise KeyError(strategy_id)

    def deployment(self, deployment_id: str) -> DeploymentSpec:
        for deployment in self.deployments:
            if deployment.deployment_id == deployment_id:
                return deployment
        raise KeyError(deployment_id)


def _dynamic_spec() -> StrategySpec:
    payload = {
        "strategy_id": "DYN-SV",
        "structure_recipe": "legacy.dynamic_short_vol.v1",
        "structure_parameters": {"menu": DYNAMIC_MENU},
        "decision_clock": "legacy.entry_close.v1",
        "entry_policy": "legacy.dynamic.entry.v1",
        "exit_policy": "legacy.dynamic.exit.v1",
        "quote_policy": "legacy.chain_quote.v1",
        "fill_policy": "legacy.fill_model.v1",
        "universe_policy": "legacy.scoring_universe.v1",
        "domain_policy": "legacy.domain.v1",
        "feature_recipe_ids": ("legacy.chooser_knn_analogs.v1",),
        "model_role_bindings": {"chooser": "dyn_sv_chooser_v1_1"},
        "forecast_sizing_recipe": None,
        "analog_recipe": "legacy.bucket_analogs.v1",
        "payoff_recipe": "legacy.payoff.v1",
        "gate_recipe": "inherit-selected.v1",
        "chooser_recipe": "legacy.dynamic_menu.v1",
        "fallback_policy": "legacy.dynamic_fallback.v1",
        "refusal_codes": (),
    }
    return StrategySpec(
        strategy_version="legacy-phase4.v1", validation_status="promoted",
        component_graph_ref="legacy.components.dynamic_short_vol.v1",
        definition_hash=content_hash(payload), evidence_refs=("legacy-source-inventory",),
        **payload,
    )


def default_registry() -> StrategyRegistry:
    strategies = (*_inventory(), _dynamic_spec())
    deployment_payload = {
        "deployment_id": "legacy-phase4-deployment.v1",
        "strategy_spec_ref": "legacy-strategy-inventory.v1",
        "clock_contract_ref": "legacy.entry_close.v1",
        "model_role_bindings": {
            "size": "size_v1_4", "implied_t1": "opf_implied_t1_gbm",
            "runup_move": "runup_move_d14_v1_gbm", "iv_crush": "iv_crush_v1_gbm",
            "gate:STR-THRU": "gate_midfill_str_thru_forecast_analog",
            "gate:STR-RUNUP": "gate_midfill_str_runup",
            "chooser:DYN-SV": "dyn_sv_chooser_v1_1",
        },
        "feature_contract_bindings": {
            recipe: recipe for recipe in (
                "legacy.market_context.v1", "legacy.event_history.v1",
                "legacy.bucket_analogs.v1", "legacy.chooser_knn_analogs.v1",
                "legacy.calibration.v1",
            )
        },
        "evidence_state_refs": ("legacy-model-registry", "legacy-residual-state"),
        "validation_receipt_refs": ("phase1-scoring-baseline",),
        "effective_from": "2026-09-16",
        "mode": "historical",
    }
    deployment = DeploymentSpec(**deployment_payload)
    return StrategyRegistry(strategies=strategies, deployments=(deployment,))
