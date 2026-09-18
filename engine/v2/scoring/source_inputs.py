"""Build bounded native scoring inputs from answer-free source material."""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping

from engine.v2.scoring.stages import (
    NativeScoreInputs,
    StageReceipt,
    receipt,
)

__all__ = ["SourceBundle", "build_native_score_inputs"]

_ANSWER_FIELDS = frozenset({
    "legs", "selected_legs", "selected_contracts", "resolved_legs",
    "entry_cost", "forecasts", "driver_prediction", "forecast_abs_move",
    "runup_move_prediction", "pred_iv_crush", "pred_iv_crush_30",
    "model_fair_pct", "exp_pnl_sim", "win_sim", "sim_p10", "sim_p90",
    "pool_n", "gate_score", "gate_pass", "gate_decision",
    "financial_diagnostics", "fair_premium_pct", "premium_vs_fair",
    "cost_over_width", "terminal_payoff", "chooser_selection",
    "validation_status", "readiness", "reason_codes",
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
_GATE_RECIPE_FIELDS = frozenset({
    "model", "threshold", "recipe_id", "artifact_ref", "artifact_hashes",
})


@dataclass(frozen=True, kw_only=True)
class SourceBundle:
    """Source-only inputs for one bounded STR-THRU execution.

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


def _reject_answers(name: str, values: Mapping[str, Any]) -> None:
    forbidden = sorted(str(key) for key in values if str(key) in _ANSWER_FIELDS)
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


def _forecast_block(bundle: SourceBundle) -> dict[str, Any]:
    unknown = sorted(set(bundle.forecast_recipes) - _FORECAST_OUTPUTS)
    if unknown:
        raise ValueError(f"unsupported forecast recipe outputs: {unknown}")
    if "driver_prediction" not in bundle.forecast_recipes:
        raise ValueError("STR-THRU requires a driver_prediction recipe")
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


def _declaration_receipts(
    source_ref: str,
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
        ("geometry", {"strategy": "STR-THRU", "context": context}, pending),
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
    if bundle.strategy != "STR-THRU":
        raise ValueError("bounded source input builder supports STR-THRU only")
    for name, values in (
        ("context", bundle.context),
        ("feature_vector", bundle.feature_vector),
        ("feature_missing_mask", bundle.feature_missing_mask),
        ("model_identity", bundle.model_identity),
        ("metadata", bundle.metadata),
    ):
        _reject_answers(name, values)

    quotes = _quote_block(bundle.raw_quotes)
    context = {**dict(bundle.context), "strategy": bundle.strategy, "quotes": quotes}
    features = {
        "model_inputs": dict(bundle.feature_vector),
        "missing_mask": dict(bundle.feature_missing_mask),
        "model_identity": dict(bundle.model_identity),
        "source_metadata": dict(bundle.metadata),
    }
    forecast = _forecast_block(bundle)
    simulation = _bounded_recipe(
        "residual_recipe", bundle.residual_recipe, _RESIDUAL_RECIPE_FIELDS,
    )
    analogs = {
        "recipe": _bounded_recipe(
            "analog_recipe", bundle.analog_recipe, _ANALOG_RECIPE_FIELDS,
        ),
    }
    gate = _gate_block(bundle.gate_recipe)
    receipts = _declaration_receipts(
        bundle.source_ref, context, features, forecast, quotes,
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
