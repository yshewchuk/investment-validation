"""Build bounded native scoring inputs from answer-free source material."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping, Sequence

from engine.v2.domain.generation import DISABLED, STRATEGIES
from engine.v2.scoring.native_analog import (
    BUCKET_RECIPE_SCHEMA,
    bucket_population_hash,
)
from engine.v2.scoring.stages import (
    NativeScoreInputs,
    StageReceipt,
    receipt,
)

__all__ = ["SourceBundle", "build_native_score_inputs"]

_ANSWER_FIELDS = frozenset({
    "legs", "selected_legs", "selected_contracts", "resolved_legs",
    "entry_cost", "structure_width", "model_artifact_ids", "forecasts",
    "driver_prediction", "forecast_abs_move", "forecast_p10",
    "forecast_p90", "forecast_sd", "runup_move_prediction",
    "runup_move_raw_d14", "runup_move_raw_d14_p10",
    "runup_move_raw_d14_p90", "runup_move_raw_d14_sd", "runup_move_p10",
    "runup_move_p90", "runup_move_sd", "pred_iv_crush",
    "pred_iv_crush_30", "model_fair_pct", "exp_pnl_sim", "win_sim",
    "sim_p10", "sim_p90", "pool_n", "exp_pnl_analog", "win_analog",
    "ci_low", "ci_high", "n_analogs", "gate_score", "gate_threshold",
    "gate_pass", "gate_decision", "chooser_score", "chooser_candidates",
    "chooser_selection", "chosen_strategy", "chosen_margin", "menu_size",
    "financial_diagnostics", "entry_cost_pct", "model_vs_market",
    "fair_premium_pct", "premium_vs_fair", "cost_over_width",
    "terminal_payoff", "exp_pnl_model", "win_model", "resolved_request",
    "validation_status", "readiness", "reason_codes", "warnings",
})
_FORECAST_OUTPUTS = frozenset({
    "driver_prediction", "forecast_abs_move", "runup_move_prediction",
    "pred_iv_crush", "pred_iv_crush_30", "model_fair_pct",
})
_LINEAR_RECIPE_FIELDS = frozenset({"intercept", "coefficients"})
_RESIDUAL_RECIPE_FIELDS = frozenset({
    "mode", "terminal_spots", "weights", "capital_at_risk", "pnl_cutoff",
    "population_ref", "recipe_id", "draw_count", "seed",
})
_ANALOG_RECIPE_FIELDS = frozenset({
    "recipe_id", "population_ref", "distance_metric", "neighbors",
})
# Fields describing the legacy bucket-analog recipe (the PRODUCTION analog
# construction; see native_analog.LegacyBucketRecipe). population_hash is
# deliberately excluded here: it is derived from analog_source_rows by the
# builder, never supplied by the caller, so a bundle cannot assert an
# unverified answer for its own population.
_BUCKET_ANALOG_RECIPE_FIELDS = frozenset({
    "bucket_dimensions", "widening_order", "min_analogs", "alpha",
    "bootstrap_draws", "bootstrap_seed", "ci_quantiles",
})
_GATE_RECIPE_FIELDS = frozenset({
    "model", "threshold", "recipe_id", "artifact_ref", "artifact_hashes",
})
_SIZE_STRATEGIES = frozenset({
    "TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7",
    "CTR5",
})
_STRATEGY_FORECAST_OUTPUTS = {
    "STR-THRU": frozenset({"driver_prediction"}),
    "STR-RUNUP": frozenset({
        "driver_prediction", "runup_move_prediction",
    }),
    **{
        strategy: frozenset({"forecast_abs_move"})
        for strategy in _SIZE_STRATEGIES
    },
}
_SUPPORTED_STRATEGIES = frozenset(_STRATEGY_FORECAST_OUTPUTS)


@dataclass(frozen=True, kw_only=True)
class SourceBundle:
    """Source-only inputs for one bounded native strategy execution.

    Recipes describe calculations. They do not carry calculated forecasts,
    selected contracts, prices, simulation summaries, or decisions.
    """

    source_ref: str
    context: Mapping[str, Any]
    raw_quotes: Mapping[Any, Mapping[str, Any]]
    feature_vector: Mapping[str, Any]
    feature_missing_mask: Mapping[str, bool]
    model_identity: Mapping[str, Any]
    forecast_recipes: Mapping[str, Mapping[str, Any]]
    model_artifact_refs: Mapping[str, str]
    residual_recipe: Mapping[str, Any]
    analog_recipe: Mapping[str, Any]
    gate_recipe: Mapping[str, Any]
    driver_name: str = "abs_move"
    strategy: str = "STR-THRU"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Causal bucket-analog population, source-only: each row is a PRIOR
    # event's bucket membership plus its realized outcome. Empty by default,
    # meaning no analog population was supplied — the analog stage is then
    # genuinely not applicable, not silently skipped despite a request.
    analog_source_rows: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    # This request's own bucket membership (no outcome, no legacy answer).
    analog_query: Mapping[str, Any] = field(default_factory=dict)


def _answer_paths(value: Any, path: str) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_name = str(key)
            child_path = f"{path}.{key_name}"
            if key_name in _ANSWER_FIELDS:
                found.append(child_path)
            found.extend(_answer_paths(child, child_path))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            found.extend(_answer_paths(child, f"{path}[{index}]"))
    return found


def _reject_answers(name: str, values: Any) -> None:
    forbidden = sorted(_answer_paths(values, name))
    if forbidden:
        raise ValueError(f"{name} contains calculated answer fields: {forbidden}")


def _finite_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _linear_recipe(name: str, value: Mapping[str, Any]) -> dict[str, Any]:
    extra = sorted(set(value) - _LINEAR_RECIPE_FIELDS)
    if extra:
        raise ValueError(f"{name} has unsupported recipe fields: {extra}")
    coefficients = value.get("coefficients", {})
    if not isinstance(coefficients, Mapping):
        raise ValueError(f"{name}.coefficients must be a mapping")
    return {
        "intercept": _finite_number(value.get("intercept", 0.0), f"{name}.intercept"),
        "coefficients": {
            str(feature): _finite_number(coefficient, f"{name}.{feature}")
            for feature, coefficient in coefficients.items()
        },
    }


def _forecast_block(bundle: SourceBundle, strategy: str) -> dict[str, Any]:
    unknown = sorted(set(bundle.forecast_recipes) - _FORECAST_OUTPUTS)
    if unknown:
        raise ValueError(f"unsupported forecast recipe outputs: {unknown}")
    missing = sorted(
        _STRATEGY_FORECAST_OUTPUTS[strategy] - set(bundle.forecast_recipes)
    )
    if missing:
        raise ValueError(
            f"{strategy} requires forecast recipes for: {missing}"
        )
    missing_refs = sorted(
        set(bundle.forecast_recipes) - set(bundle.model_artifact_refs)
    )
    if missing_refs:
        raise ValueError(f"forecast recipes lack artifact refs: {missing_refs}")
    refs = {
        str(role): str(reference)
        for role, reference in bundle.model_artifact_refs.items()
    }
    if any(not reference.strip() for reference in refs.values()):
        raise ValueError("model artifact refs must be non-empty")
    return {
        "driver_name": str(bundle.driver_name),
        "models": {
            str(output): _linear_recipe(str(output), recipe)
            for output, recipe in bundle.forecast_recipes.items()
        },
        "artifact_hashes": tuple(refs[output] for output in bundle.forecast_recipes),
        "model_artifact_refs": refs,
    }


def _quote_block(raw: Mapping[Any, Mapping[str, Any]]) -> dict[Any, dict[str, float]]:
    quotes: dict[Any, dict[str, float]] = {}
    for contract, quote in raw.items():
        if not isinstance(quote, Mapping):
            raise ValueError("each raw quote must be a mapping")
        extra = sorted(set(quote) - {"bid", "ask"})
        if extra:
            raise ValueError(f"raw quote contains unsupported fields: {extra}")
        if "bid" not in quote or "ask" not in quote:
            raise ValueError("raw quotes require bid and ask")
        quotes[contract] = {
            "bid": _finite_number(quote["bid"], "quote.bid"),
            "ask": _finite_number(quote["ask"], "quote.ask"),
        }
    if not quotes:
        raise ValueError("raw quote map must not be empty")
    return quotes


def _bounded_recipe(
    name: str,
    values: Mapping[str, Any],
    allowed: frozenset[str],
) -> dict[str, Any]:
    extra = sorted(set(values) - allowed)
    if extra:
        raise ValueError(f"{name} has unsupported recipe fields: {extra}")
    return dict(values)


def _gate_block(values: Mapping[str, Any]) -> dict[str, Any]:
    gate = _bounded_recipe("gate_recipe", values, _GATE_RECIPE_FIELDS)
    model = gate.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("gate_recipe requires a linear model")
    gate["model"] = _linear_recipe("gate_recipe.model", model)
    gate["threshold"] = _finite_number(
        gate.get("threshold"), "gate_recipe.threshold",
    )
    return gate


def _analog_block(bundle: SourceBundle) -> dict[str, Any]:
    """Express the legacy bucket-analog recipe, with inputs, when sourced.

    A recipe is only declared when the bundle carries a real, answer-free
    population (``analog_source_rows``). Without one there is nothing to
    execute, so the block reports genuinely not-applicable (``recipe: None``)
    rather than declaring a recipe the stage cannot satisfy.
    """
    config = _bounded_recipe(
        "analog_recipe", bundle.analog_recipe,
        _ANALOG_RECIPE_FIELDS | _BUCKET_ANALOG_RECIPE_FIELDS,
    )
    if not bundle.analog_source_rows:
        return {"recipe": None}
    missing = sorted(_BUCKET_ANALOG_RECIPE_FIELDS - set(config))
    if missing:
        raise ValueError(f"analog_recipe requires bucket fields: {missing}")
    _reject_answers("analog_source_rows", bundle.analog_source_rows)
    _reject_answers("analog_query", bundle.analog_query)
    dimensions = tuple(str(name) for name in config["bucket_dimensions"])
    population_hash = bucket_population_hash(bundle.analog_source_rows, dimensions)
    recipe = {
        "schema_version": BUCKET_RECIPE_SCHEMA,
        "bucket_dimensions": dimensions,
        "widening_order": tuple(str(name) for name in config["widening_order"]),
        "min_analogs": config["min_analogs"],
        "alpha": config["alpha"],
        "bootstrap_draws": config["bootstrap_draws"],
        "bootstrap_seed": config["bootstrap_seed"],
        "ci_quantiles": tuple(config["ci_quantiles"]),
        "population_hash": population_hash,
    }
    return {
        "recipe": recipe,
        "source_rows": [dict(row) for row in bundle.analog_source_rows],
        "query_features": dict(bundle.analog_query),
    }


def _declaration_receipts(
    source_ref: str,
    strategy: str,
    context: Mapping[str, Any],
    features: Mapping[str, Any],
    forecast: Mapping[str, Any],
    quotes: Mapping[Any, Any],
    analogs: Mapping[str, Any],
    simulation: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> tuple[StageReceipt, ...]:
    pending = {"execution": "native-runtime"}
    declarations = (
        ("resolve_context", {"source_ref": source_ref}, context),
        ("features", context, features),
        ("forecast", features, forecast),
        ("geometry", {"strategy": strategy, "context": context}, pending),
        ("pricing", {"raw_quotes": quotes}, pending),
        ("analogs", {"recipe": analogs}, pending),
        ("simulation", {"recipe": simulation}, pending),
        ("gate", {"recipe": gate}, pending),
        ("chooser", {}, pending),
        ("serialization", {"source_ref": source_ref}, pending),
    )
    return tuple(receipt(stage, inputs, output)
                 for stage, inputs, output in declarations)


def build_native_score_inputs(bundle: SourceBundle) -> NativeScoreInputs:
    """Translate a bounded source bundle into executable native stage inputs."""
    if not isinstance(bundle, SourceBundle):
        raise TypeError("source input builder requires SourceBundle")
    if not bundle.source_ref.strip():
        raise ValueError("source_ref must be non-empty")
    strategy = str(bundle.strategy)
    if strategy not in _SUPPORTED_STRATEGIES:
        if strategy in DISABLED:
            detail = DISABLED[strategy]
        elif strategy in STRATEGIES:
            detail = "UNSUPPORTED_SOURCE_CONTRACT"
        else:
            detail = "UNKNOWN_STRATEGY"
        raise ValueError(
            f"source input builder does not support {strategy}: {detail}"
        )
    for name, values in (
        ("context", bundle.context),
        ("feature_vector", bundle.feature_vector),
        ("feature_missing_mask", bundle.feature_missing_mask),
        ("model_identity", bundle.model_identity),
        ("metadata", bundle.metadata),
    ):
        _reject_answers(name, values)

    quotes = _quote_block(bundle.raw_quotes)
    context = {**dict(bundle.context), "strategy": strategy, "quotes": quotes}
    features = {
        "model_inputs": dict(bundle.feature_vector),
        "missing_mask": dict(bundle.feature_missing_mask),
        "model_identity": dict(bundle.model_identity),
        "source_metadata": dict(bundle.metadata),
    }
    forecast = _forecast_block(bundle, strategy)
    simulation = _bounded_recipe(
        "residual_recipe", bundle.residual_recipe, _RESIDUAL_RECIPE_FIELDS,
    )
    analogs = _analog_block(bundle)
    gate = _gate_block(bundle.gate_recipe)
    receipts = _declaration_receipts(
        bundle.source_ref, strategy, context, features, forecast, quotes,
        analogs, simulation, gate,
    )
    return NativeScoreInputs(
        context=context,
        features=features,
        forecast=forecast,
        geometry=None,
        pricing=None,
        analogs=analogs,
        simulation=simulation,
        gate=gate,
        chooser={},
        diagnostics={},
        source_ref=bundle.source_ref,
        stage_receipts=receipts,
    )
