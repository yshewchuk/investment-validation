"""Explicit inputs and receipts for the native scoring execution graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from engine.v2.domain.generation import Geometry, NativeLeg, Pricing, generate, price
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
                   source_ref=str(values.get("native_source_ref", "compatibility-input")),
                   stage_receipts=receipts)


def receipt(stage: str, inputs: Any, output: Any) -> StageReceipt:
    if stage not in STAGE_NAMES:
        raise ValueError(f"unknown scoring stage: {stage}")
    return StageReceipt(stage, content_hash(inputs), content_hash(output))


_COMPUTED_FIELDS = frozenset({
    "entry_cost", "fill", "legs", "model_artifact_ids", "selected_contracts",
    "structure_width",
})


def _merge_stage(values: dict[str, Any], block: Mapping[str, Any]) -> None:
    values.update({key: value for key, value in block.items()
                   if key not in _COMPUTED_FIELDS and key != "flags"})


def _initial_values(inputs: NativeScoreInputs) -> tuple[dict[str, Any], list[StageReceipt], Any]:
    values: dict[str, Any] = {}
    blocks = (inputs.context, inputs.features, inputs.forecast, inputs.analogs,
              inputs.simulation, inputs.gate, inputs.chooser, inputs.diagnostics)
    compatibility = next((block.get("entry_cost") for block in blocks
                          if block.get("entry_cost") is not None), None)
    executed: list[StageReceipt] = []
    prior: Any = {"source_ref": inputs.source_ref}
    for stage, block in (("resolve_context", inputs.context), ("features", inputs.features),
                         ("forecast", inputs.forecast)):
        _merge_stage(values, block)
        executed.append(receipt(stage, prior, block))
        prior = {"prior": executed[-1].output_hash, "values": values}
    return values, executed, compatibility


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
    if geometry_inputs.get("forecast_abs_move") is None:
        geometry_inputs["forecast_abs_move"] = values.get("driver_prediction") or 0.0
    if inputs.geometry is not None:
        geometry_inputs["resolved_legs"] = tuple(vars(leg) for leg in inputs.geometry.legs)
        geometry_inputs["width"] = inputs.geometry.width
    try:
        if name == "DYN-SV":
            raw = tuple(inputs.diagnostics.get("legs") or ())
            legs = tuple(NativeLeg(
                str(leg.get("name", f"leg-{index}")),
                str(leg.get("right", "P")), str(leg.get("side", "buy")),
                float(leg.get("quantity", leg.get("qty", 0.0))),
                float(leg["strike"]), str(leg.get("expiry", values.get("expiry"))),
            ) for index, leg in enumerate(raw))
            geometry = Geometry(name, float(values.get("spot") or 0.0),
                                float(inputs.diagnostics.get("structure_width") or 0.0),
                                legs, None if raw else "DYNAMIC_CHOOSER")
        else:
            geometry = generate(name, geometry_inputs)
    except Exception as exc:
        if inputs.geometry is not None or geometry_inputs.get("resolved_legs"):
            raise
        geometry = Geometry(name, 0.0, 0.0, (), str(exc))
    return geometry_inputs, geometry


def _quote_map(inputs: NativeScoreInputs, name: str) -> dict:
    if inputs.pricing is not None:
        quotes = {(leg.right, leg.strike, leg.expiry): {"bid": leg.bid, "ask": leg.ask}
                  for leg in inputs.pricing.legs}
    else:
        quotes = {}
    if name == "DYN-SV" and not quotes:
        quotes = {(str(leg.get("right")), float(leg.get("strike")),
                   str(leg.get("expiry"))): {"bid": leg.get("bid"), "ask": leg.get("ask")}
                  for leg in inputs.diagnostics.get("legs") or ()
                  if leg.get("bid") is not None and leg.get("ask") is not None}
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
    return price(geometry, quotes, alpha)


def _append_late_stages(inputs: NativeScoreInputs, values: dict[str, Any],
                        executed: list[StageReceipt]) -> list[str]:
    blocks = (inputs.context, inputs.features, inputs.forecast, inputs.analogs,
              inputs.simulation, inputs.gate, inputs.chooser, inputs.diagnostics)
    flags: list[str] = []
    for stage, block in (("analogs", inputs.analogs), ("simulation", inputs.simulation),
                         ("gate", inputs.gate), ("chooser", inputs.chooser),
                         ("diagnostics", inputs.diagnostics)):
        _merge_stage(values, block)
        executed.append(receipt(stage, {"prior": executed[-1].output_hash}, block))
    for block in blocks:
        for item in block.get("flags") or ():
            if str(item) not in flags:
                flags.append(str(item))
    return flags


def assemble_native_values(inputs: NativeScoreInputs, *, strategy: str | None = None,
                           fill_model: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute declared stages and return execution-owned values."""
    if not isinstance(inputs, NativeScoreInputs):
        raise TypeError("native stage assembly requires NativeScoreInputs")
    values, executed, compatibility = _initial_values(inputs)
    name = _strategy_name(inputs, strategy, values)
    geometry_inputs, geometry = _resolve_geometry(inputs, name, values)
    executed.append(receipt("geometry", geometry_inputs, geometry))
    alpha = float((fill_model or {}).get("alpha", 0.5))
    quotes = _quote_map(inputs, name)
    pricing = _resolve_pricing(inputs, name, geometry, quotes, alpha, compatibility)
    executed.append(receipt("pricing", {"geometry": geometry, "quotes": quotes,
                                         "fill_alpha": alpha}, pricing))
    flags = _append_late_stages(inputs, values, executed)
    for refusal in (geometry.refusal, pricing.refusal):
        if refusal and refusal not in flags and not (
                values.get("spot") is None and refusal.endswith("must be finite")):
            flags.append(refusal)
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
