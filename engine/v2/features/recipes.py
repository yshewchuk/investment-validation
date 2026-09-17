"""Registered Phase 4 feature recipes and separate context scopes."""
from __future__ import annotations

from dataclasses import dataclass

from engine.v2.contracts import FeatureRecipe
from engine.v2.foundation import content_hash

__all__ = ["FeatureRegistry", "default_feature_registry"]


@dataclass(frozen=True)
class FeatureRegistry:
    recipes: tuple[FeatureRecipe, ...]

    def get(self, recipe_id: str) -> FeatureRecipe:
        for recipe in self.recipes:
            if recipe.recipe_id == recipe_id:
                return recipe
        raise KeyError(recipe_id)


def _recipe(recipe_id: str, source_scope: str, outputs: tuple[str, ...], history: str) -> FeatureRecipe:
    columns = tuple({"name": name, "type": "float", "unit": "native",
                     "nullable": True, "null_policy": "explicit"} for name in outputs)
    payload = {
        "recipe_id": recipe_id, "version": "v1", "input_contracts": ("snapshot.v1",),
        "dependency_recipe_ids": (), "output_columns": columns,
        "source_scope": source_scope, "history_scope": history,
        "lookback_rule": "legacy-source-defined", "observation_cutoff_rule": "at-or-before-decision",
        "label_availability_rule": "no-future-labels", "supported_clock_contracts": ("legacy.entry_close.v1",),
        "fallback_policy": "typed-missing", "determinism_policy": "ordered-source-rows",
    }
    return FeatureRecipe(implementation_hash=content_hash(payload), **payload)


def default_feature_registry() -> FeatureRegistry:
    return FeatureRegistry(recipes=(
        _recipe("legacy.market_context.v1", "event", ("im", "or_implied", "or_iv30", "mcap_usd"), "event-date market state"),
        _recipe("legacy.event_history.v1", "event", ("n_prior_events", "dte_band", "ret5", "ret20"), "prior events before cutoff"),
        _recipe("legacy.bucket_analogs.v1", "analog", ("analog_mean", "analog_win_rate", "analog_n"), "complete historical replay population"),
        _recipe("legacy.chooser_knn_analogs.v1", "analog", ("chooser_score", "n_admissible"), "complete chooser training population"),
        _recipe("legacy.calibration.v1", "calibration", ("calibration_probability",), "all eligible pairs before cutoff"),
    ))
