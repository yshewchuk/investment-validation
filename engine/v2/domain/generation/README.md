# `engine/v2/domain/generation`

## Ownership

Implements the **Structure generator — template resolution, finite placement search, completeness receipts** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**4a** of §4.1.

Replaces (§4.4): `structures.py`, `forecast_sizing.py`, `fills.py`.

## Responsibilities

- Resolve a structure template against a listed strike ladder and expiry set.
- Finite placement search with a validity and completeness receipt.
- Forecast-sized geometry, recording the forecast even when the shape is pinned.

## Non-responsibilities

- **Rank candidates by PnL** — `engine/v2/domain/simulation` does it instead.
- **Change strategy selection rules** — `engine/v2/scoring` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

The package exposes deterministic native geometry and quote pricing.

<!-- public-interface: DISABLED, STRATEGIES, Geometry, GeometryRefusal, NativeLeg, PricedLeg, Pricing, PricingRefusal, generate, price -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

engine.v2.scoring imports the generator and pricing functions for native
geometry and same-input comparisons. engine.v2.ops.cli imports `Geometry` and
`Pricing` to reconstruct a captured `NativeScoreInputs` document for the
read-only `ops rescore` command.

<!-- consumers: engine.v2.scoring, engine.v2.ops -->

## Usage

generate("STR-THRU", {"spot": 100, "forecast_abs_move": 6,
"expiry": "2026-10-01"}) creates immutable legs. price then applies the
declared worst-to-best fill alpha to explicit bid/ask quotes.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
