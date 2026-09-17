"""Explicit inputs and receipts for the native scoring execution graph."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from engine.v2.domain.generation import Geometry, Pricing, generate, price
from engine.v2.domain.valuation import terminal_payoff
from engine.v2.foundation import content_hash

STAGE_NAMES = (
    "resolve_context", "features", "forecast", "geometry", "pricing",
    "analogs", "simulation", "gate", "chooser", "diagnostics", "serialization",
)


@dataclass(frozen=True)
class StageReceipt:
    stage: str
    input_hash: str
    output_hash: str
    owner: str = "engine.v2.scoring"


@dataclass(frozen=True)
class NativeScoreInputs:
    """All stage outputs required before a canonical record can be emitted."""

    context: Mapping[str, Any]
    features: Mapping[str, Any]
    forecast: Mapping[str, Any]
    geometry: Geometry | None
    pricing: Pricing | None
    analogs: Mapping[str, Any]
    simulation: Mapping[str, Any]
    gate: Mapping[str, Any]
    chooser: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    source_ref: str
    stage_receipts: tuple[StageReceipt, ...]

    def __post_init__(self) -> None:
        if not self.source_ref.strip():
            raise ValueError("native score inputs require source_ref")
        stages = tuple(receipt.stage for receipt in self.stage_receipts)
        seen = set(stages)
        required = set(STAGE_NAMES) - {"diagnostics"}
        missing = sorted(required - seen)
        if missing:
            raise ValueError(f"missing native stage receipts: {missing}")
        unknown = sorted(seen - set(STAGE_NAMES))
        if unknown:
            raise ValueError(f"unknown native stage receipts: {unknown}")
        duplicates = sorted(stage for stage in seen if stages.count(stage) > 1)
        if duplicates:
            raise ValueError(f"duplicate native stage receipts: {duplicates}")

    @classmethod
    def from_legacy_fields(cls, fields: Mapping[str, Any]) -> "NativeScoreInputs":
        """Create a migration input with explicit stage ownership.

        This compatibility constructor is only for old callers.  New scoring
        code should build the stage blocks directly and provide real receipts.
        """
        values = dict(fields)
        blocks = {name: values for name in ("context", "features", "forecast",
                                             "analogs", "simulation", "gate",
                                             "chooser", "diagnostics")}
        receipts = tuple(StageReceipt(
            stage,
            content_hash({"stage": stage, "values": values}),
            content_hash({"stage": stage, "values": values}),
        ) for stage in STAGE_NAMES if stage != "diagnostics")
        return cls(**blocks, geometry=None, pricing=None,
                   source_ref="compatibility-input",
                   stage_receipts=receipts)


def receipt(stage: str, inputs: Any, output: Any) -> StageReceipt:
    if stage not in STAGE_NAMES:
        raise ValueError(f"unknown scoring stage: {stage}")
    return StageReceipt(stage, content_hash(inputs), content_hash(output))


_PRICING_OUTPUTS = frozenset({
    "entry_cost", "fill", "legs", "model_artifact_ids", "selected_contracts",
    "structure_width",
})
_FORECAST_OUTPUTS = frozenset({
    "driver_prediction", "forecast_abs_move", "runup_move_prediction",
    "forecast_p10", "forecast_p90", "forecast_sd",
})
_SIMULATION_OUTPUTS = frozenset({"exp_pnl_sim", "win_sim"})
_GATE_OUTPUTS = frozenset({"gate_score", "gate_threshold", "gate_pass"})
_OWNED_OUTPUTS = (
    _PRICING_OUTPUTS | _FORECAST_OUTPUTS | _SIMULATION_OUTPUTS | _GATE_OUTPUTS
)


def _merge_stage(values: dict[str, Any], block: Mapping[str, Any]) -> None:
    values.update({key: value for key, value in block.items()
                   if key not in _OWNED_OUTPUTS and key != "flags"})


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _add_flag(flags: list[str], value: Any) -> None:
    code = str(value)
    if code and code not in flags:
        flags.append(code)


def _facts(inputs: NativeScoreInputs, values: Mapping[str, Any]) -> dict[str, Any]:
    facts = dict(inputs.context)
    facts.update(inputs.features)
    model_inputs = inputs.features.get("model_inputs")
    if isinstance(model_inputs, Mapping):
        facts.update(model_inputs)
    facts.update(values)
    return facts


def _linear(spec: Mapping[str, Any], facts: Mapping[str, Any],
            owner: str) -> float:
    value = _finite(spec.get("intercept", 0.0))
    coefficients = spec.get("coefficients", {})
    if value is None or not isinstance(coefficients, Mapping):
        raise ValueError(f"INVALID_{owner.upper()}_MODEL")
    for feature, raw_coefficient in coefficients.items():
        coefficient = _finite(raw_coefficient)
        feature_value = _finite(facts.get(feature))
        if coefficient is None:
            raise ValueError(f"INVALID_{owner.upper()}_MODEL")
        if feature_value is None:
            raise ValueError(f"MISSING_{owner.upper()}_INPUT:{feature}")
        value += coefficient * feature_value
    if not isfinite(value):
        raise ValueError(f"INVALID_{owner.upper()}_OUTPUT")
    return value


def _execute_forecast(inputs: NativeScoreInputs, values: dict[str, Any],
                      flags: list[str]) -> dict[str, Any]:
    block = inputs.forecast
    output: dict[str, Any] = {}
    if block.get("driver_name") is not None:
        output["driver_name"] = str(block["driver_name"])
    models = block.get("models", {})
    if not isinstance(models, Mapping):
        _add_flag(flags, "INVALID_FORECAST_MODELS")
        return output
    facts = _facts(inputs, values)
    declared = False
    for field in _FORECAST_OUTPUTS:
        spec = models.get(field, block.get(field))
        if spec is None:
            continue
        declared = True
        if not isinstance(spec, Mapping):
            _add_flag(flags, f"UNOWNED_FORECAST_OUTPUT:{field}")
            continue
        try:
            output[field] = _linear(spec, facts, "forecast")
        except ValueError as exc:
            _add_flag(flags, exc)
    if not declared and not output:
        _add_flag(flags, "MISSING_FORECAST_INPUT")
    values.update(output)
    return output


def _initial_values(
    inputs: NativeScoreInputs,
    flags: list[str],
    compatibility: bool,
) -> tuple[dict[str, Any], list[StageReceipt], Any]:
    values: dict[str, Any] = {}
    blocks = (inputs.context, inputs.features, inputs.forecast, inputs.analogs,
              inputs.simulation, inputs.gate, inputs.chooser, inputs.diagnostics)
    legacy_cost = next((block.get("entry_cost") for block in blocks
                        if block.get("entry_cost") is not None), None)
    executed: list[StageReceipt] = []
    prior: Any = {"source_ref": inputs.source_ref}
    for stage, block in (("resolve_context", inputs.context), ("features", inputs.features)):
        _merge_stage(values, block)
        output = {key: value for key, value in block.items()
                  if key not in _OWNED_OUTPUTS and key != "flags"}
        executed.append(receipt(stage, prior, output))
        prior = {"prior": executed[-1].output_hash, "values": values}
    if compatibility:
        values.update(inputs.forecast)
        forecast_output = dict(inputs.forecast)
    else:
        forecast_output = _execute_forecast(inputs, values, flags)
    executed.append(receipt("forecast", prior, forecast_output))
    return values, executed, legacy_cost


def _strategy_name(inputs: NativeScoreInputs, strategy: str | None,
                   values: Mapping[str, Any]) -> str:
    source = strategy or (inputs.geometry.strategy if inputs.geometry else values.get("strategy"))
    name = str(source or "").split("@", 1)[0]
    if not name:
        raise ValueError("native geometry stage requires a strategy")
    if inputs.geometry is not None and inputs.geometry.strategy != name:
        raise ValueError("request and geometry strategies disagree")
    return name


def _resolve_geometry(inputs: NativeScoreInputs, name: str,
                      values: dict[str, Any]):
    geometry_inputs = dict(values)
    if name not in {"CAL-P", "CND-P"}:
        spot = _finite(geometry_inputs.get("spot"))
        if spot is None or spot <= 0.0:
            return geometry_inputs, Geometry(name, 0.0, 0.0, (), "MISSING_SPOT")
        if (geometry_inputs.get("expiry") is None
                and geometry_inputs.get("post_event_expiry") is None):
            return geometry_inputs, Geometry(name, spot, 0.0, (), "MISSING_EXPIRY")
    if inputs.geometry is not None:
        geometry_inputs["resolved_legs"] = tuple(vars(leg) for leg in inputs.geometry.legs)
        geometry_inputs["width"] = inputs.geometry.width
    try:
        if name == "DYN-SV":
            geometry = Geometry(
                name, float(values["spot"]), 0.0, (), "DYNAMIC_CHOOSER",
            )
        else:
            geometry = generate(name, geometry_inputs)
    except Exception as exc:
        if inputs.geometry is not None or geometry_inputs.get("resolved_legs"):
            raise
        geometry = Geometry(name, 0.0, 0.0, (), str(exc))
    if geometry.refusal is None and not geometry.legs:
        geometry = Geometry(
            name, geometry.spot, geometry.width, (), "MISSING_CONTRACTS",
        )
    return geometry_inputs, geometry


def _quote_map(inputs: NativeScoreInputs, name: str) -> dict:
    if inputs.pricing is not None:
        quotes = {(leg.right, leg.strike, leg.expiry): {"bid": leg.bid, "ask": leg.ask}
                  for leg in inputs.pricing.legs}
    else:
        quotes = {}
    return quotes


def _resolve_pricing(inputs: NativeScoreInputs, name: str, geometry: Geometry,
                     quotes: Mapping, alpha: float, compatibility: Any):
    if inputs.pricing is not None and inputs.pricing.strategy != name:
        raise ValueError("request and pricing strategies disagree")
    if geometry.refusal:
        return Pricing(name, geometry.spot, 0.0, (), geometry.refusal)
    if not quotes and inputs.source_ref == "compatibility-input":
        return Pricing(name, geometry.spot, float(compatibility or 0.0), (), None)
    if not quotes:
        return Pricing(name, geometry.spot, 0.0, (), "MISSING_PRICING_INPUT")
    try:
        return price(geometry, quotes, alpha)
    except Exception as exc:
        return Pricing(name, geometry.spot, 0.0, (), str(exc))


def _simulation_spots(block: Mapping[str, Any],
                      flags: list[str]) -> tuple[float, ...] | None:
    raw_spots = block.get("terminal_spots")
    if raw_spots is None:
        declared = False
        for field in _SIMULATION_OUTPUTS:
            if block.get(field) is not None:
                declared = True
                _add_flag(flags, f"UNOWNED_SIMULATION_OUTPUT:{field}")
        if not declared:
            _add_flag(flags, "MISSING_SIMULATION_INPUT")
        return None
    try:
        spots = tuple(float(item) for item in raw_spots)
    except (TypeError, ValueError):
        _add_flag(flags, "INVALID_SIMULATION_SPOTS")
        return None
    if not spots or not all(isfinite(item) and item >= 0.0 for item in spots):
        _add_flag(flags, "INVALID_SIMULATION_SPOTS")
        return None
    return spots


def _simulation_weights(block: Mapping[str, Any], count: int,
                        flags: list[str]) -> tuple[float, ...] | None:
    raw_weights = block.get("weights")
    try:
        weights = (tuple(1.0 for _ in range(count)) if raw_weights is None
                   else tuple(float(item) for item in raw_weights))
    except (TypeError, ValueError):
        _add_flag(flags, "INVALID_SIMULATION_WEIGHTS")
        return None
    total_weight = sum(weights)
    if (len(weights) != count or total_weight <= 0.0
            or not all(isfinite(item) and item >= 0.0 for item in weights)):
        _add_flag(flags, "INVALID_SIMULATION_WEIGHTS")
        return None
    return weights


def _simulation_cutoff(block: Mapping[str, Any], output: dict[str, Any],
                       flags: list[str]) -> None:
    if block.get("pnl_cutoff") is None:
        return
    cutoff = _finite(block["pnl_cutoff"])
    if cutoff is None:
        _add_flag(flags, "INVALID_PNL_CUTOFF")
    else:
        output["pnl_cutoff"] = cutoff


def _execute_simulation(
    inputs: NativeScoreInputs,
    values: dict[str, Any],
    geometry: Geometry,
    pricing: Pricing,
    flags: list[str],
) -> dict[str, Any]:
    block = inputs.simulation
    output: dict[str, Any] = {}
    spots = _simulation_spots(block, flags)
    if spots is None:
        return output
    if geometry.refusal or pricing.refusal:
        _add_flag(flags, geometry.refusal or pricing.refusal)
        return output
    weights = _simulation_weights(block, len(spots), flags)
    if weights is None:
        return output
    capital = _finite(block.get("capital_at_risk", abs(pricing.entry_cost)))
    if capital is None or capital <= 0.0:
        _add_flag(flags, "INVALID_SIMULATION_CAPITAL")
        return output
    legs = tuple(vars(leg) for leg in pricing.legs)
    returns = tuple(
        (terminal_payoff(legs, spot) - pricing.entry_cost) / capital
        for spot in spots
    )
    total_weight = sum(weights)
    output["exp_pnl_sim"] = sum(
        weight * result for weight, result in zip(weights, returns)
    ) / total_weight
    output["win_sim"] = sum(
        weight for weight, result in zip(weights, returns) if result > 0.0
    ) / total_weight
    _simulation_cutoff(block, output, flags)
    values.update(output)
    return output


def _execute_gate(inputs: NativeScoreInputs, name: str,
                  values: dict[str, Any], flags: list[str]) -> dict[str, Any]:
    block = inputs.gate
    output: dict[str, Any] = {}
    recipe = block.get("recipe")
    if block.get("model") is not None:
        model = block["model"]
        if not isinstance(model, Mapping):
            _add_flag(flags, "INVALID_GATE_MODEL")
            return output
        try:
            score = _linear(model, _facts(inputs, values), "gate")
        except ValueError as exc:
            _add_flag(flags, exc)
            return output
        threshold = _finite(block.get("threshold"))
        if threshold is None:
            _add_flag(flags, "MISSING_GATE_THRESHOLD")
            return output
        output.update({"gate_score": score, "gate_threshold": threshold,
                       "gate_pass": score >= threshold})
    elif recipe is not None:
        _add_flag(flags, f"UNSUPPORTED_GATE_RECIPE:{recipe}")
    else:
        declared = False
        for field in _GATE_OUTPUTS:
            if block.get(field) is not None:
                declared = True
                _add_flag(flags, f"UNOWNED_GATE_OUTPUT:{field}")
        if not declared:
            _add_flag(flags, "MISSING_GATE_INPUT")
        return output
    values.update(output)
    return output


def _append_late_stages(
    inputs: NativeScoreInputs,
    values: dict[str, Any],
    executed: list[StageReceipt],
    geometry: Geometry,
    pricing: Pricing,
    compatibility: bool,
    flags: list[str],
) -> None:
    blocks = (inputs.context, inputs.features, inputs.forecast, inputs.analogs,
              inputs.simulation, inputs.gate, inputs.chooser, inputs.diagnostics)
    if compatibility:
        for stage, block in (
            ("analogs", inputs.analogs), ("simulation", inputs.simulation),
            ("gate", inputs.gate), ("chooser", inputs.chooser),
            ("diagnostics", inputs.diagnostics),
        ):
            values.update(block)
            executed.append(receipt(
                stage, {"prior": executed[-1].output_hash}, block,
            ))
    else:
        _merge_stage(values, inputs.analogs)
        analog_output = {key: value for key, value in inputs.analogs.items()
                         if key not in _OWNED_OUTPUTS and key != "flags"}
        executed.append(receipt(
            "analogs", {"prior": executed[-1].output_hash}, analog_output,
        ))
        simulation = _execute_simulation(
            inputs, values, geometry, pricing, flags,
        )
        executed.append(receipt(
            "simulation",
            {"prior": executed[-1].output_hash, "inputs": inputs.simulation,
             "entry_cost": pricing.entry_cost},
            simulation,
        ))
        gate = _execute_gate(inputs, geometry.strategy, values, flags)
        executed.append(receipt(
            "gate",
            {"prior": executed[-1].output_hash, "inputs": inputs.gate,
             "simulation": simulation},
            gate,
        ))
        for stage, block in (
            ("chooser", inputs.chooser), ("diagnostics", inputs.diagnostics),
        ):
            _merge_stage(values, block)
            output = {key: value for key, value in block.items()
                      if key not in _OWNED_OUTPUTS and key != "flags"}
            executed.append(receipt(
                stage, {"prior": executed[-1].output_hash}, output,
            ))
    for block in blocks:
        for item in block.get("flags") or ():
            _add_flag(flags, item)


def assemble_native_values(inputs: NativeScoreInputs, *, strategy: str | None = None,
                           fill_model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute declared stages and return execution-owned values."""
    if not isinstance(inputs, NativeScoreInputs):
        raise TypeError("native stage assembly requires NativeScoreInputs")
    is_compatibility = inputs.source_ref == "compatibility-input"
    flags: list[str] = []
    values, executed, legacy_cost = _initial_values(
        inputs, flags, is_compatibility,
    )
    name = _strategy_name(inputs, strategy, values)
    geometry_inputs, geometry = _resolve_geometry(inputs, name, values)
    executed.append(receipt("geometry", geometry_inputs, geometry))
    alpha = _finite((fill_model or {}).get("alpha", 0.5))
    if alpha is None or not 0.0 <= alpha <= 1.0:
        alpha = 0.5
        _add_flag(flags, "INVALID_FILL_ALPHA")
    quotes = _quote_map(inputs, name)
    pricing = _resolve_pricing(
        inputs, name, geometry, quotes, alpha, legacy_cost,
    )
    executed.append(receipt("pricing", {"geometry": geometry, "quotes": quotes,
                                         "fill_alpha": alpha}, pricing))
    _append_late_stages(
        inputs, values, executed, geometry, pricing, is_compatibility, flags,
    )
    for refusal in (geometry.refusal, pricing.refusal):
        if refusal:
            _add_flag(flags, refusal)
    legs = pricing.legs or geometry.legs
    values.update({"spot": geometry.spot, "structure_width": geometry.width,
                   "entry_cost": pricing.entry_cost if pricing.refusal is None else None,
                   "fill": alpha, "legs": tuple(vars(leg) for leg in legs),
                   "selected_contracts": tuple(vars(leg) for leg in geometry.legs),
                   "flags": tuple(flags), "native_source_ref": inputs.source_ref})
    executed.append(receipt("serialization", values, values))
    values["native_stage_receipts"] = tuple({"stage": item.stage,
        "input_hash": item.input_hash, "output_hash": item.output_hash,
        "owner": item.owner} for item in executed)
    return values


__all__ = [
    "NativeScoreInputs", "STAGE_NAMES", "StageReceipt",
    "assemble_native_values", "receipt",
]
