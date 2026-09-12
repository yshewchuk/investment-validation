# `engine/v2/scoring`

## Ownership

Implements the **Scoring application — forecasts, shape, pricing, gate/chooser decisions, diagnostics** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**5** of §4.1.

Replaces (§4.4): `score.py split by the stages in §6.3`, `entry_rules.py`, `replay.py`, `trailing_cutoff from pnl_sim.py`.

## Responsibilities

- The §6.3 execution order, one module per stage.
- Gate and chooser decisions, including DYN-SV menu resolution.
- Financial diagnostics and a validated immutable ScoreRecord.

## Non-responsibilities

- **Read future outcomes** — `engine/v2/evaluation` does it instead.
- **Mutate a strategy or model registry** — `engine/v2/registry` does it instead.
- **Fit a model during a score request** — `engine/v2/models/training` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

_Nothing yet — the package is an empty skeleton. The first name added here is added to this list in the same commit._

<!-- public-interface: none -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

_Nothing yet — no package imports this one. The first importer is added here in the same commit._

<!-- consumers: none -->

## Usage

No runnable example yet: phase 0 creates the package and writes no
production logic into it. The shortest real example lands with the first
public name, and is expected to run in under a second from frozen
fixtures.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
