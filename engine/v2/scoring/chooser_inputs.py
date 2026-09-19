"""The chooser block of a ``SourceBundle`` (R4-20 gap 5 and remaining gap (a)).

``chooser_recipe`` names the DYN-SV champion binding (``{"binding_id"[,
"output"]}``) and, optionally, the declared inputs of its derived columns:

* ``producers``: ``{"pred_im_t1_d14": {"binding_id"}, "pred_runup_abs_move_d14":
  {"binding_id"}}`` -- the Tier-4 serving folds (roles ``implied_t1`` and
  ``runup_move``) legacy ``Scorer._chooser_frame`` serves those columns from;
* ``analog_pool``: ``{"pool_id", "cutoff"[, "content_hash"]}`` -- the key
  the frozen ``SourceBundle.chooser_analog_pool`` must carry;
* ``admissible_table``: ``{"table_id", "version"[, "content_hash"]}`` -- the
  key ``SourceBundle.chooser_admissible_table`` must carry.

``SourceBundle.chooser_fold_pools`` declares each served fold's pool. Nothing
here computes a feature: the block carries recipes, frozen state and source
rows, and ``native_chooser`` derives the columns when the stage runs.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from engine.v2.foundation import content_hash
from engine.v2.models.admissible_table import AdmissibleDepthTable
from engine.v2.models.chooser_analog_pool import ChooserAnalogPoolArtifact
from engine.v2.scoring import native_chooser as fields
from engine.v2.scoring.native_chooser_features import PRODUCER_COLUMNS
from engine.v2.scoring.source_inputs import (
    _CHOOSER_STRATEGIES,
    _FROZEN_RECIPE_FIELDS,
    SourceBundle,
    _bounded_recipe,
    _frozen_recipe_executor,
    _is_frozen_recipe,
    fold_pool,
)

__all__ = ["chooser_block", "frozen_chooser_block"]

_CHOOSER_RECIPE_FIELDS = _FROZEN_RECIPE_FIELDS | frozenset({
    "producers", "analog_pool", "admissible_table",
})
_ANALOG_KEY_FIELDS = frozenset({"pool_id", "cutoff", "content_hash"})
_TABLE_KEY_FIELDS = frozenset({"table_id", "version", "content_hash"})
_FOLD_POOL_OUTPUTS = frozenset({"pred_abs_move", *PRODUCER_COLUMNS})


def _producers(bundle: SourceBundle, declared: Any) -> dict[str, Any]:
    if not declared:
        return {}
    if not isinstance(declared, Mapping):
        raise ValueError("chooser_recipe.producers must be a mapping")
    unknown = sorted(set(declared) - set(PRODUCER_COLUMNS))
    if unknown:
        raise ValueError(f"chooser_recipe.producers has unsupported outputs: {unknown}")
    executors = {}
    for output, recipe in declared.items():
        if not _is_frozen_recipe(recipe):
            raise ValueError(f"chooser_recipe.producers.{output} must name a frozen binding")
        executors[str(output)] = _frozen_recipe_executor(bundle, str(output), recipe)
    return executors


def _fold_pools(bundle: SourceBundle) -> tuple[dict[str, Any], dict[str, str]]:
    unknown = sorted(set(bundle.chooser_fold_pools) - _FOLD_POOL_OUTPUTS)
    if unknown:
        raise ValueError(f"chooser_fold_pools has unsupported outputs: {unknown}")
    pools = {}
    for output, values in bundle.chooser_fold_pools.items():
        pool = fold_pool(f"chooser_fold_pools.{output}", values)
        if pool:
            pools[str(output)] = pool
    return pools, {output: content_hash(pool) for output, pool in pools.items()}


def _state(recipe: Mapping[str, Any], name: str, allowed: frozenset[str],
           state: Any, kind: type) -> tuple[dict[str, Any], Any]:
    key = _bounded_recipe(f"chooser_recipe.{name}", recipe.get(name) or {}, allowed)
    if state is not None and not isinstance(state, kind):
        raise ValueError(f"chooser {name} must be a {kind.__name__}")
    return key, state


def chooser_block(bundle: SourceBundle, strategy: str) -> dict[str, Any]:
    """The chooser champion as a frozen recipe plus its declared inputs.

    Empty unless ``chooser_recipe`` is declared; declared chooser state
    without a recipe, or a recipe for a strategy outside legacy's
    DYNAMIC_MENU, is a malformed bundle.
    """
    if not bundle.chooser_recipe:
        if (bundle.chooser_fold_pools or bundle.chooser_analog_pool is not None
                or bundle.chooser_admissible_table is not None):
            raise ValueError("chooser inputs declared without a chooser_recipe")
        return {}
    if strategy not in _CHOOSER_STRATEGIES:
        raise ValueError(f"chooser_recipe declared for {strategy}, which is not a "
                         "DYN-SV menu candidate")
    config = _bounded_recipe("chooser_recipe", bundle.chooser_recipe, _CHOOSER_RECIPE_FIELDS)
    if not _is_frozen_recipe(config):
        raise ValueError("chooser_recipe must name a frozen binding")
    champion = {name: config[name] for name in _FROZEN_RECIPE_FIELDS if name in config}
    pools, hashes = _fold_pools(bundle)
    analog_key, analog = _state(config, "analog_pool", _ANALOG_KEY_FIELDS,
                                bundle.chooser_analog_pool, ChooserAnalogPoolArtifact)
    table_key, table = _state(config, "admissible_table", _TABLE_KEY_FIELDS,
                              bundle.chooser_admissible_table, AdmissibleDepthTable)
    return {
        "binding_id": str(config["binding_id"]),
        "executors": {
            "chooser_score": _frozen_recipe_executor(bundle, "chooser_score", champion),
        },
        fields.PRODUCERS_FIELD: _producers(bundle, config.get("producers")),
        fields.FOLD_POOLS_FIELD: pools,
        fields.FOLD_POOL_HASHES_FIELD: hashes,
        fields.ANALOG_POOL_FIELD: analog,
        fields.ANALOG_KEY_FIELD: analog_key,
        fields.ADMISSIBLE_TABLE_FIELD: table,
        fields.ADMISSIBLE_KEY_FIELD: table_key,
    }


def frozen_chooser_block(*, strategy: str, recipe: Mapping[str, Any],
                         fold_pools: Mapping[str, Any],
                         analog_pool: ChooserAnalogPoolArtifact | None,
                         admissible_table: AdmissibleDepthTable | None,
                         inference: Any, release: Any) -> dict[str, Any]:
    """:func:`chooser_block` over declared parts instead of a whole bundle.

    A saved Phase 4 trace carries the chooser as a JSON declaration (recipe,
    fold pools) plus a verified release and frozen state; this runs the same
    resolution a ``SourceBundle`` gets, with no other bundle field involved.
    """
    parts = SimpleNamespace(
        chooser_recipe=dict(recipe), chooser_fold_pools=dict(fold_pools),
        chooser_analog_pool=analog_pool, chooser_admissible_table=admissible_table,
        frozen_inference=inference, model_release=release,
    )
    return chooser_block(parts, strategy)  # type: ignore[arg-type]
