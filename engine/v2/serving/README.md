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
- The read-only v2 dashboard API (`api.py`, P3-2): authenticated, paginated,
  ETag'd JSON over `projections`' bounded read helpers — `/api/v1/releases/
  current`, `/api/v1/releases/{id}`, `/api/v1/events`, `/api/v1/events/{id}/
  scores`, `/api/v1/scores/{id}`, `/api/v1/operations`.

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

legacy_bundle (P3-4): the real legacy render-bundle adapter. `load_legacy_
bundle(bundle_root)` reads the actual `dashboard/render.py` `render_bundle`
output tree — `data/board.json`/`data/tickers/<ticker>.json`, accepting
either that JSON form or (only when the JSON is absent) its `.js` wrapper —
and returns `(bundle_rows_by_ticker, bundle_manifest)`: every ticker's rows
flattened out of its `events[].rows` (main-board and ladder rows together,
told apart only by `strike_offset`, exactly as `bridge.build_bridges`
already expects them), and a `{relative_path: "sha256:<hex>"}` manifest of
every file actually read, so a projection binds to exact bytes. When BOTH a
`.json` and its `.js` sibling exist, both are read, both hashed into the
manifest, and their parsed payloads compared: the legacy dashboard and the
compatibility preview load the `.js` wrapper in the browser, so what this
loader projects must equal what a user actually sees, and a disagreement
between the two forms is refused (`BUNDLE_FORM_MISMATCH`) rather than
silently trusting whichever form was read first. Strict and never a silent
skip otherwise: a symlink anywhere in a read path, a ticker string that
would escape the bundle root, an unrecognized top-level shape, two ticker
files claiming the same ticker, a malformed `.js` wrapper, or a ticker the
board references with no rendered file each raise a typed
`LegacyBundleError` (`.code`) rather than being coerced or dropped.
`load_score_document(path)` reads `score.json` with the same strictness:
`rows`/`ladder`/`expected_population` required as lists, no unrecognized
top-level key, no symlink. No legacy `engine.*` import, no `engine.v2.ops`
import — only bytes in, parsed JSON out.

`projections` also carries the two additions P3-2 needed to serve §6's
filtered `/events` route without a route ever touching a table directly:
`event_query_hash` is the stable identity of one `/events` query (release
plus every normalized filter, deliberately excluding `limit`/`cursor` — a
page-size change or a page turn is not a different query), and `get_event`
is the single-event lookup `/events/{id}/scores` wraps. `list_events`/
`event_scores` grew optional `event_date_from`/`event_date_to`/`ticker`/
`strategy`/`verdict` filters, backward compatible with every existing call
(all new parameters default to `None`, unfiltered): a strategy/verdict
filter selects EVENTS with at least one matching score row (an `EXISTS`
against `serving_score_summary`) and, on the SAME call, narrows that event's
own attached `scores` to the matching ones — §6's "selects matching events
and their matching visible summaries consistently" — by construction, not by
two independently-written filters that could drift apart.

api (P3-2): `create_app(*, serving_db, store_root, serving_root, token,
resolver=None) -> FastAPI` wires §6's six routes over `projections`' bounded
read helpers only — no route or app-startup path imports or initiates
scoring, a provider, `engine.v2.ops` or legacy `engine.*` (`tests/
test_v2_serving_api.py` proves this with a real subprocess and two
`sys.modules` snapshots). Auth is bearer-or-cookie, mirroring
`operations.py`'s own rule (reimplemented here, not imported — a peer
module this task does not touch); the token never appears in an error body.
Errors are one `Problem`-shaped envelope everywhere — see "Problem field
reference" below for the exact JSON — built as a plain dict rather than
importing the `engine.v2.contracts.Problem` dataclass — this module has no
other use for a contracts import, and the fan-out budget (8 distinct
modules, §4.3) is otherwise exactly spent on `fastapi`, `engine.v2.
foundation`, this package's own `projections`, and `hmac`/`os`/`json`/
`argparse`/`uvicorn`. Pagination cursors are opaque and HMAC-signed with a
key derived from the token (never the token itself): bound to `release_id`
plus `event_query_hash` (computed over `ticker`/`strategy`/`verdict`/
`date_from`/`date_to` — the wire names `ui/src/api/client.ts`'s
`EventQuery` uses, mapped to `projections`' own `event_date_from`/
`event_date_to` only at this boundary), so a cursor replayed against a
different release or filter set — or simply edited — fails closed as
`CURSOR_MISMATCH` (409).

### Problem field reference

Every non-2xx response is exactly this shape (`_problem`, `ApiError`) —
nothing added, nothing renamed for any one client:

```json
{
  "code": "UNKNOWN_RELEASE",
  "category": "validation",
  "retryable": false,
  "message": "unknown release id",
  "stage": null,
  "trace_id": null,
  "dependency_refs": [],
  "retry_after_seconds": null,
  "diagnostic_ref": null,
  "details": {},
  "schema_version": "problem.v1.0"
}
```

`code`/`category`/`retryable`/`message` are the fields a caller actually
needs; the rest are the operational envelope every other v2 `Problem`
carries (`engine.v2.contracts.operations.Problem`) — present for shape
consistency, always `null`/empty here since this module never has stage,
tracing or dependency information to report. `message` is the
human-readable string; there is no separate `title`. `status` is not a body
field — it is the HTTP status code the response was sent with. Codes this
module raises: `UNAUTHORIZED` (401), `RELEASE_ID_REQUIRED` (400, §5.4:
score/event-scope routes require one — never search across releases),
`INVALID_REQUEST` (422, bad `limit`/date format/`clock_id`),
`CURSOR_MISMATCH` (409), `NO_CURRENT_RELEASE` (503), `UNKNOWN_RELEASE`/
`UNKNOWN_EVENT`/`UNKNOWN_SCORE` (404), `OPERATIONS_UNAVAILABLE` (503).

**Current-release resolution (§5.4) is one injected seam.** `create_app`'s
`resolver` parameter is any zero-argument `Callable[[], str | None]`; the
default (`_default_resolver`) reads a plain `CURRENT` file directly under
`serving_root`, refusing a symlinked pointer or one whose content is not a
single clean path segment — the same check `operations.py`'s own
`_resolve_current_id` applies, reimplemented rather than imported. A pointer
naming a release with no committed projection is `NO_CURRENT_RELEASE` (503),
never a silent fallback to "latest". The publisher that will OWN this
pointer is a later task; this seam is what it plugs into. `/releases/current`
stays `Cache-Control: no-store` (its resolution can change between requests)
but now also carries a strong ETag and answers `If-None-Match` with 304 —
`no-store` and ETag/304 are not mutually exclusive: the first says a cache
must not reuse the response unasked, the second makes asking again cheap.
This keeps `ui/src/hooks.ts` `usePinnedRelease`'s ~4s background poll cheap
without ever letting a stale release get cached and reused.

**UI-alignment fixes (mid-task, after `ui/` P3-3a landed at `bc6da8d`),
reconciled against `tests/fixtures/v2_ui_mock_api.py` per the rule "if §6 and
the mock disagree, §6 wins":**

- `date_from`/`date_to` (not `event_date_from`/`event_date_to`) are the
  `/events` query param names — taken from `ui/src/api/client.ts`'s
  `EventQuery`, matched by both the mock and this API's own routes.
- `GET /events/{id}/scores` returns a **bare JSON array**, not
  `{release_id, event_id, scores}` — §6 names no envelope beyond "strategy
  score summaries", and both the mock and the TS `DataClient` (`Promise<
  EventScoreSummary[]>`) agree on an array, so there was no reason to keep
  the wrapper.
- **Judgement call: `CURSOR_MISMATCH` is HTTP 409, matching component_
  contracts.md §13.2 and the mock's `HTTPStatus.CONFLICT`.** An earlier
  version of this module used 400 per this task's own original brief text
  (which stated it twice); the coordinator's later UI-alignment instruction
  settled the question in the guide/mock's favor, and this implementation
  now follows that.
- **Reviewed and reversed: `/scores/{id}` and `/events/{id}/scores` require
  `release_id`; neither searches across releases.** An earlier version of
  this module let `/scores/{id}` omit `release_id` and search every release
  for the id (matching the mock's own all-releases scan and `ui/src/api/
  client.ts`'s optional `releaseId?`), reasoning that a shared score id
  necessarily shares content. Review (P3-2) corrected this: §5.4 states
  "event/score IDs are unique within a release" — scoped, not global — so a
  route keyed by one must always be given the release, never search for it.
  A missing `release_id` on either route is now `RELEASE_ID_REQUIRED` (400),
  distinct from the `UNKNOWN_RELEASE`/`UNKNOWN_EVENT`/`UNKNOWN_SCORE` (404)
  an unknown-but-present one or id gets.
  `test_score_detail_scopes_a_shared_score_id_to_the_requested_release`
  covers the case that motivated the original design (two releases sharing
  one content-addressed score id) without a cross-release search: fetching
  it under either release's own explicit id succeeds, and
  `test_score_detail_membership_validated_when_release_supplied` (a score
  minted under one release only, requested under the other) proves no
  fallback ever occurs.
  `ui/src/api/client.ts`'s optional `releaseId?` on `getScore`/its mock's
  all-releases scan is accordingly a UI-side gap flagged for the
  coordinator (who owns reconciling `ui/` for this review), not fixed here.
- Auth was already cookie-and-bearer (`_authorized` checks both
  independently); `tests/test_v2_serving_api.py`'s
  `test_cookie_only_auth_works_on_every_route` now pins that a
  cookie-only request (no `Authorization` header at all, exactly what `ui/
  src/api/client.ts` sends) succeeds on every route, not just proves the
  logic exists.

<!-- public-interface: operations, create_server, bridge, LEGACY_DISPLAY_MAPPING_V1, build_bridges, projections, build_candidate, connect, ensure_schema, resolve_event_refs, get_release, list_events, event_scores, get_score_detail, get_event, event_query_hash, ServingIndexError, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, api, create_app, ApiError, legacy_bundle, load_legacy_bundle, load_score_document, LegacyBundleError -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

`engine.v2.dashboard` imports `operations.create_server` — the compatibility
preview launcher (`engine/v2/dashboard/preview.py`) composes only this
surface, per its layer-8 "7 only" import rule.

`api.py` is launched directly (`python3 -m engine.v2.serving.api`) and read
by `tests/test_v2_serving_api.py`; neither is a v2 package import, so
neither appears in this directive (`checks/package_readmes.py` only tracks
`engine/v2/**` packages as consumers). No v2 package imports `api` yet.

<!-- consumers: engine.v2.dashboard -->

## Usage

Run the isolated HTTP contract tests:

    python3 -m pytest tests/test_v2_ops_serving.py -q

The server requires a nonempty authentication token supplied at construction.
Credentials never belong in a release manifest, URL or status artifact.

`tools/v2_dashboard_project.py` is the offline projection coordinator CLI
(P3-1b/P3-4): given a saved `score.json`, a rendered bundle, a Phase 2
catalog/store and a snapshot id, it builds one candidate against a serving
root (`serving.sqlite` plus its own `objects/`) and prints `{"release_id":
..., "findings": {...}}` (or the refusal) as JSON. `--bundle-format legacy`
(the default) reads a real `render_bundle` output tree through
`legacy_bundle.load_legacy_bundle` and folds `content_hash(bundle_manifest)`
into the release's `bundle_manifest_ref`, overriding whatever the
`--preview-input` document declared; `--bundle-format flat` keeps the
pre-P3-4 simplified one-array-per-ticker shape for tests that predate the
real adapter. No scoring, no provider calls, no legacy import — it composes
`engine.v2.ops.bootstrap.open_catalog` and this package in the one place
(`tools/`) allowed to import both.

`python3 -m engine.v2.serving.api --host 127.0.0.1 --port 8766 --serving-db
serving/serving.sqlite --store-root serving/objects --serving-root serving`
starts the read API under uvicorn. The token comes from the `V2_DASHBOARD_
TOKEN` environment variable — the launcher refuses to start without one, and
refuses a non-loopback `--host` without `--allow-non-loopback`, exactly the
two refusals `engine/v2/dashboard/preview.py`'s launcher already makes for
the compatibility surface.

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
`earnings_events` fragments, never a real panel or renderer).
`tests/test_v2_serving_legacy_bundle.py` (P3-4) covers the real adapter: a
small real-shaped bundle rendered once via the actual
`engine.dashboard.render.render_bundle` (never a real panel), round-tripped
through `load_legacy_bundle` into `build_bridges`; a byte changed in one
ticker file changing the manifest hash and the CLI's `release_id`; and every
refusal (symlink, traversal, a ticker the board references with no file,
duplicate ticker, malformed `.js` wrapper):

    python3 -m pytest -q tests/test_v2_serving_bridge.py tests/test_v2_serving_projections.py tests/test_v2_serving_legacy_bundle.py

`tests/test_v2_serving_api.py` (P3-2) builds a real `serving.sqlite` the
same way and serves it over REAL HTTP — a real `uvicorn.Server` bound to an
ephemeral loopback port in a background thread (`TestClient` alone is not
enough for the release-switch-mid-request and real-socket cases this needs).
Auth, pagination, cursor tampering/reuse, ETag/304, release-switch-mid-
pagination-and-mid-detail, typed errors, and the no-scoring/provider guard
(a real subprocess, two `sys.modules` snapshots) are all covered; the
launcher's own refusals run as real `python3 -m engine.v2.serving.api`
subprocesses:

    python3 -m pytest -q tests/test_v2_serving_api.py
