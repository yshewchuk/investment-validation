# `engine/v2/foundation`

## Ownership

Implements the **paths, env, canonical JSON, session/calendar arithmetic, causality primitives** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**0** of §4.1.

Replaces (§4.4): `paths.py`, `env.py`, `jsonio.py`, `audit.py`, `session arithmetic from calendar.py`.

## Responsibilities

- Canonical JSON (RFC 8785) and content hashing, per contracts §2.2.
- Path and environment resolution.
- Session arithmetic: BMO/AMC anchoring, trading-day offsets.
- Causality primitives — the cutoff comparison every feature respects.

## Non-responsibilities

- **Fetch a calendar** — `engine/v2/data` does it instead.
- **Decide whether a difference is acceptable** — `engine/v2/diagnosis` does it instead.

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
