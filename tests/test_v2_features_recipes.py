from dataclasses import replace

from engine.v2.features import default_feature_registry
from engine.v2.foundation import content_hash, to_document


def test_feature_registry_separates_event_analog_and_calibration_scopes():
    registry = default_feature_registry()
    scopes = {recipe.source_scope for recipe in registry.recipes}
    assert scopes == {"event", "analog", "calibration"}
    assert {recipe.recipe_id for recipe in registry.recipes} >= {
        "legacy.bucket_analogs.v1", "legacy.chooser_knn_analogs.v1",
    }


def test_recipe_identity_changes_when_missing_policy_changes():
    recipe = default_feature_registry().get("legacy.market_context.v1")
    changed = replace(recipe, fallback_policy="refuse")
    assert content_hash(recipe.output_columns) == content_hash(changed.output_columns)
    assert content_hash(to_document(recipe)) != content_hash(to_document(changed))
