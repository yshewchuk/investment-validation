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

bridge (P3-1a): offline mapping of a verified `score.json` and a verified
rendered bundle to `LegacyScoreBridge` rows, over `engine.v2.contracts.serving`
shapes. `build_bridges` takes the caller's already-resolved Phase 2
event/calendar mapping (`event_refs`) and returns `(list[LegacyScoreBridge],
ProjectionFindings)`. `LEGACY_DISPLAY_MAPPING_V1` is the checked spec of
`dashboard/render.py` `compact_row`'s display fields. No legacy import, no
`engine.v2.ops` import (a peer), no financial arithmetic.

projections (P3-1b): the minimal serving index — a separate SQLite file,
own `schema_versions` sequence (owner `"serving"`; the checksum-and-refuse-
newer pattern of `engine.v2.ops.migrations` reimplemented locally, never
imported — ops is a layer-7 peer). `resolve_event_refs` is the `(ticker,
event_date) -> EventRef` resolver §5.3 point 3 asks for, built on
`Repository.scan` (never raw parquet); a pair with zero matches is left out
of the mapping (unmapped) and two or more distinct `event_id`s map to `None`
(ambiguous) — neither ever invents an id. `build_candidate` runs `bridge.
build_bridges` over the resolved refs, publishes every engine/detail payload
(the findings receipt, the projection manifest, each score's full
`LegacyScoreBridge`) as an immutable `ArtifactStore` object, and — only when
`ProjectionFindings.ok` — inserts one release plus its `serving_event_summary`/
`serving_score_summary` index rows in a single transaction, idempotent on
content-derived `release_id`. A findings failure still publishes the receipt
(`Problem.diagnostic_ref`) but writes no release row: no "current" pointer is
created here, that is a later task's single published pointer. `get_release`,
`list_events` (cursor-paginated, ordered `event_date, ticker, event_id`),
`event_scores` and `get_score_detail` are the bounded read helpers a future
read API (P3-2) wraps. `connect`/`ensure_schema` open and migrate the file.

**Summary-field gap (review fix, `EVENT_SCORE_SUMMARY_V1` v1.1).** The
rendered row carries no single headline "expected return" or closed
"verdict" field, so `_score_summary_fields` never invents one:
`expected_return` is always `None`; `expected_return_model`/`_analog`/`_sim`
are `exp_pnl_model`/`exp_pnl_analog`/`exp_pnl_sim` copied through unchanged
(a null model with a present analog stays two separate values, never
promoted into `expected_return`). The real board's headline is
`dashboard/static/assets/app.js` `pnlCell` — model, else sim, **never**
analog — computed client-side over the raw bundle, not a field
`dashboard/render.py` `compact_row` itself ever writes; reproducing it here
is future work, not this index's own computation. `verdict` is the row's own
`gate_pass`, kept as its raw JSON value (`"true"`/`"false"`/`None`), never
translated into an invented `TRADE`/`REFUSED` word. The real board's
decision pill (`app.js` `gatePill`) is a richer client-side tree over
`flags`/`gate_score`/`gate_threshold` with several distinct N/A reasons
(disabled, not sized, gate-declined, arithmetic-only) that this index does
not reproduce — an open gap, not a promise this summary makes.

<!-- public-interface: operations, create_server, bridge, LEGACY_DISPLAY_MAPPING_V1, build_bridges, projections, build_candidate, connect, ensure_schema, resolve_event_refs, get_release, list_events, event_scores, get_score_detail, ServingIndexError, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE -->

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

`tools/v2_dashboard_project.py` is the offline projection coordinator CLI
(P3-1b): given a saved `score.json`, a flat per-ticker render-bundle
directory, a Phase 2 catalog/store and a snapshot id, it builds one candidate
against a serving root (`serving.sqlite` plus its own `objects/`) and prints
`{"release_id": ..., "findings": {...}}` (or the refusal) as JSON. No
scoring, no provider calls, no legacy import — it composes `engine.v2.ops.
bootstrap.open_catalog` and this package in the one place (`tools/`) allowed
to import both.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.

`tests/test_v2_serving_bridge.py` (P3-1a) and `tests/test_v2_serving_
projections.py` (P3-1b) cover the offline bridge/index over synthetic
`score.json`/bundle pairs and a synthetic Phase 2 catalog (real
`earnings_events` fragments, never a real panel or renderer):

    python3 -m pytest -q tests/test_v2_serving_bridge.py tests/test_v2_serving_projections.py
