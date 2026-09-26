"""Native-vs-legacy parity: the record comparator core and the per-dimension policy.

Layer 6.5 of `system_rearchitecture.md` §4.1 (not a §4 owner-table row). It
imports only ``engine.v2.foundation``; only the layer-7 reporting packages
(``engine.v2.ops``, ``engine.v2.serving``) and the ``engine.v2.diagnosis``
sink sit above it, so no scored package can import the comparator that
compares it.

See ``README.md`` for the public interface and the layering.
"""
from __future__ import annotations

from engine.v2.parity.dimensions import (
    ANALOG_FIELDS,
    FINANCIAL_FIELDS,
    FORECAST_FIELDS,
    GATE_FIELDS,
    NEVER_RAN_DIMENSIONS,
    SIMULATION_FIELDS,
    compare_dimension,
)

__all__ = [
    "ANALOG_FIELDS",
    "FINANCIAL_FIELDS",
    "FORECAST_FIELDS",
    "GATE_FIELDS",
    "NEVER_RAN_DIMENSIONS",
    "SIMULATION_FIELDS",
    "compare_dimension",
]
