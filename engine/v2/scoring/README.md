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

`application` provides the shared scoring kernel; `compatibility` is the
temporary legacy adapter; `financial` owns diagnostics; `identity` owns
content-addressed request identities.

<!-- public-interface: application, compatibility, financial, identity, stages, canonical_request, dependency_hash, financial_diagnostics, request_hash, replay, score_batch, score_event, score_frozen, score_id, score_many, score_one, NativeScoreInputs, STAGE_NAMES, StageReceipt -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

_Nothing yet — no package imports this one. The first importer is added here in the same commit._

<!-- consumers: none -->

## Usage

The application takes a ScoreRequest and NativeScoreInputs. Every score must
carry context, feature, forecast, geometry, pricing, analog, simulation, gate,
chooser and serialization receipts. Legacy scoring remains available only
through the explicit compatibility module for comparison.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
