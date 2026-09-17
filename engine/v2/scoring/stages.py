"""Explicit inputs and receipts for the native scoring execution graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from engine.v2.domain.generation import Geometry, Pricing
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
        seen = {receipt.stage for receipt in self.stage_receipts}
        required = set(STAGE_NAMES) - {"diagnostics"}
        missing = sorted(required - seen)
        if missing:
            raise ValueError(f"missing native stage receipts: {missing}")

    @property
    def values(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for block in (self.context, self.features, self.forecast, self.analogs,
                      self.simulation, self.gate, self.chooser, self.diagnostics):
            merged.update(block)
        if self.geometry is not None:
            merged.setdefault("legs", tuple(vars(leg) for leg in self.geometry.legs))
            merged.setdefault("spot", self.geometry.spot)
            merged.setdefault("structure_width", self.geometry.width)
            if self.geometry.refusal:
                merged.setdefault("flags", (self.geometry.refusal,))
        if self.pricing is not None:
            merged.setdefault("entry_cost", self.pricing.entry_cost)
            merged.setdefault("legs", tuple(vars(leg) for leg in self.pricing.legs))
            if self.pricing.refusal:
                merged.setdefault("flags", (self.pricing.refusal,))
        merged["native_stage_receipts"] = tuple(
            {"stage": receipt.stage, "input_hash": receipt.input_hash,
             "output_hash": receipt.output_hash, "owner": receipt.owner}
            for receipt in self.stage_receipts
        )
        merged["native_source_ref"] = self.source_ref
        return merged

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


def assemble_native_values(inputs: NativeScoreInputs) -> dict[str, Any]:
    """Validate and flatten completed stage outputs at the final boundary."""
    if not isinstance(inputs, NativeScoreInputs):
        raise TypeError("native stage assembly requires NativeScoreInputs")
    if inputs.geometry is not None and inputs.geometry.refusal is None:
        if inputs.pricing is None:
            raise ValueError("priced geometry is required before serialization")
        if inputs.pricing.strategy != inputs.geometry.strategy:
            raise ValueError("geometry and pricing strategies disagree")
    return inputs.values


__all__ = [
    "NativeScoreInputs", "STAGE_NAMES", "StageReceipt",
    "assemble_native_values", "receipt",
]
