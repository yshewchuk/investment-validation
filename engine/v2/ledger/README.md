# `engine/v2/ledger`

## Ownership

Implements the **Prediction and position ledger — append-only facts** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**6** of §4.1.

Replaces (§4.4): `ledger.py`, `ledger_settlement.py`, `portfolio.py`.

## Responsibilities

- Append-only prediction commits and position lifecycle events (contracts §12).
- Cash and position reconciliation.

## Non-responsibilities

- **Rewrite a committed record** — `a correcting append` does it instead.
- **Decide a trading verdict** — `engine/v2/scoring` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

- decisions: install, set_authority, insert, rows, import_lines and DecisionConflict.
- export: export_generation writes a verified compatibility generation.
- The caller owns the transaction and verifies execution authority. This package
  preserves payloads and enforces unique logical decision identities.

<!-- public-interface: decisions, export, install, set_authority, insert, rows, import_lines, DecisionConflict, export_generation -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

engine/v2/ops commits validated decisions and outbox intents in one transaction,
and rehearses writer changes and compatibility exports against private copies.

<!-- consumers: engine.v2.ops -->

## Usage

Run the isolated authority tests with:

    python3 -m pytest tests/test_v2_ops_effects.py -q

Production writer activation requires the separate cutover described in
guides/rearchitecture_phase1_operations.md. Exporting a compatibility generation
does not grant another writer ownership.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
