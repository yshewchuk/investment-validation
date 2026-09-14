# `engine/v2/serving`

## Ownership

Implements the **API/projection layer — filter, paginate, authorize, serialize computed records** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**7** of §4.1.

Replaces (§4.4): `the data half of dashboard/render.py`, `dashboard/earnings_app.py`.

## Responsibilities

- Bounded, paginated reads over saved score records.
- The financial display values §6.4 moves out of rendering.
- Immutable release publication, one release per read.
- Resolve-once release identity: `GET /release/current.json` returns
  `{"release_id": ...}` for a client (the shell, the launcher) to pin and
  reuse, instead of re-following `/release/current`'s redirect on every
  navigation. `/release/current/...` keeps working for direct requests.

## Non-responsibilities

- **Fit a model** — `engine/v2/models/training` does it instead.
- **Simulate PnL** — `engine/v2/domain/simulation` does it instead.
- **Fetch vendor data in a GET request** — `engine/v2/data` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

operations.create_server starts the authenticated read-only health/release
surface. It reads operations_health.v1.0 documents and immutable legacy bundles,
and embeds the existing views beneath the current health banner.

<!-- public-interface: operations, create_server -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

`engine.v2.dashboard` imports `operations.create_server` — the compatibility
preview launcher (`engine/v2/dashboard/preview.py`) composes only this
surface, per its layer-8 "7 only" import rule.

<!-- consumers: engine.v2.dashboard -->

## Usage

Run the isolated HTTP contract tests:

    python3 -m pytest tests/test_v2_ops_serving.py -q

The server requires a nonempty authentication token supplied at construction.
Credentials never belong in a release manifest, URL or status artifact.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
