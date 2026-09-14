# `ui/` — v2 board + detail client (P3-3a, P3-3b)

React + TypeScript + Vite client for the rearchitecture Phase 3 board and
lazy event/score detail views, built against the §6 read API contract
(`guides/rearchitecture_phase3_parity_launch.md`). The real FastAPI server
(P3-2) is a separate, concurrent task — this slice was built and tested
against a stdlib mock (`tests/fixtures/v2_ui_mock_api.py`) shaped verbatim
from `engine/v2/contracts/serving.py`, so it needs no changes when P3-2
lands with the same field names. Contract decisions confirmed against
`engine/v2/serving/api.py` mid-P3-3b (coordinator's 2026-09-14 note, then
verified directly in that file): `release_id` is a *required* query param on
`GET /api/v1/scores/{id}` and `GET /api/v1/events/{id}/scores` (400
`RELEASE_ID_REQUIRED` if missing — `engine/v2/serving/api.py::
_require_release_id`), and every error body is the real `Problem` envelope
(`code`, `category`, `retryable`, `message`, plus operational fields) with
**no** `title`/`status` alias — `ApiError.status` in `client.ts` comes from
the HTTP response itself, never the body.

## Commands

```bash
npm install                 # needs the npm registry; stop and report if unreachable
npm --prefix ui run typecheck   # tsc --noEmit, strict
npm --prefix ui run build       # tsc --noEmit && vite build -> ui/dist (no source maps)
npm --prefix ui run dev         # vite dev server; proxies /api to 127.0.0.1:8765
```

Browser tests (Playwright, headless, serial xdist group — drives a real
browser and builds `ui/dist` via a real `npm run build`):

```bash
python3 -m pytest -q -p no:cacheprovider tests/test_v2_dashboard_browser.py
```

## What is shipped

**Board, event detail and score detail** (`src/App.tsx`, `src/components/
EventDetail.tsx`, `src/components/ScoreDetail.tsx`). Every row still also
links to the pinned compatibility surface (`/release/<release_id>/
index.html`) via `compatibilityLink()` (`src/format.ts`).

- **Release pinning** (`src/hooks.ts` `useResolvedRelease`, P3-3b's
  route-aware successor to P3-3a's `usePinnedRelease` — see Routing below):
  `GET
  /api/v1/releases/current` resolves exactly once per page load and is
  never re-resolved for that session. Every later request
  (`listEvents`/`getEventScores`/`getScore`) carries that pinned
  `release_id` explicitly, and the compatibility link is built from it too.
  A background poll (default 4s, `?pollMs=` overrides it for tests) keeps
  checking `current` without ever moving the pin; a mismatch surfaces as a
  banner ("the current release changed to X ... reload to see X") with a
  manual reload button — reloading is the only thing that re-resolves
  `current`, matching the compatibility preview's own "R1 readers retain
  R1; a new session resolves R2" rule (guide §9 L02).
- **Auth**: same-origin cookie (`operations_token`), mirroring
  `engine/v2/serving/operations.py`. `src/api/client.ts` is the only module
  that calls `fetch`; every request sets `credentials: "same-origin"` and
  none builds an `Authorization` header, puts a token in a URL, or touches
  `localStorage`. A 401 renders the "unauthenticated" state.
- **Release banner**: release id, producer, resolved as-of, coverage
  summary, stale/degraded reasons, and the release-changed notice above.
- **Filters**: ticker, strategy, verdict, date range (`src/components/
  EventFilters.tsx`). **No sort control is shipped** (judgement call): the
  API's documented default order (event date, ticker, event ID; guide §6)
  is used as-is, since §7 lists "sort" among the route's required
  behaviors, not as a UI widget requirement.
- **Event table** (`src/components/EventTable.tsx`): one row per
  event+strategy score. Every strategy/refusal row in the accepted
  population is shown, never filtered to passing/non-null rows. Columns:
  ticker, event date, session, strategy, verdict, driver forecast, market
  implied move, entry premium, expected return (headline choice, below),
  DYN-SV choice (`chosen_strategy`/`chosen_margin`/`menu_size`), and a
  compatibility link.
- **Headline expected return** (`src/format.ts`
  `headlineExpectedReturn`): `EventScoreSummary.expected_return` is always
  `null` (v1.1 — the rendered row carries no single merged field). This
  ports the legacy board's own choice — `engine/dashboard/static/assets/
  app.js::pnlCell` (~line 487): show `expected_return_model` if present,
  else `expected_return_sim`, **never** `expected_return_analog` — as one
  named, display-only function, not a financial computation. A `sim` badge
  marks the fallback case. Tested for the model-null+sim-present case and
  the both-null case (`test_headline_expected_return_model_null_sim_
  present_and_both_null`).
- **Verdict**: shown raw (`gate_pass: true/false/—`) plus the compatibility
  link, per the coordinator's 2026-09-14 instruction. This is a real gap,
  not a simplification that loses no information: a reader sees `false`
  where the legacy board would say "FAIL" or a rule-specific "N/A" with a
  reason. Still true in score detail (P3-3b) — see "`gatePill` decision
  tree" under Deferred below; it stayed out of scope for both slices.
- **Pagination**: forward-only (`next_cursor`), with a client-side cursor
  stack for "Previous" (judgement call — not required by §7, but cheap and
  the mock/real API both support fetching the same page again since the
  cursor for it was already seen). Count shows `total_matching`, the
  complete filtered population, never just the visible page.
- **States covered**: loading, unauthenticated, no current release (503),
  API error (events fetch), no matches (0 results), a row whose readiness
  is `unavailable` with no scores, null field vs. a real zero,
  a refusal row, and paging.

### Routing (`src/routes.ts`, P3-3b)

Hash routes, no router library (three view shapes don't need one — guide
§7): `#/release/<id>` (board), `#/release/<id>/events/<event_id>` (event
detail), `#/release/<id>/scores/<score_id>` or `#/release/<id>/events/
<event_id>/scores/<score_id>` (score detail, with or without an event
context). Every route names `release_id` explicitly.

- **Pin resolution is route-aware** (`src/hooks.ts` `useResolvedRelease`,
  replacing P3-3a's `usePinnedRelease`): a bare `#/` or no hash pins to
  whatever `current` resolves to (P3-3a behavior, unchanged) and then
  *rewrites* the address bar to name that id via `history.replaceState`
  (no extra back-button entry) so a copied link reopens the same release. A
  hash that already names a `release_id` pins to THAT id regardless of what
  `current` turns out to be, and is never repinned for the rest of the
  session (guide §9 L02: "R1 readers retain R1"; only a fresh page load —
  a real reload, not an in-app navigation — can change the pin). This is
  exactly what makes a deep link to a non-current release work.
- **No metadata route for a specific past release** (judgement call): §6
  only exposes `GET /api/v1/releases/current`, so when a deep-linked pin is
  NOT `current`, the banner cannot show that release's `resolved_as_of`/
  coverage/stale reasons — there is nothing to fetch them from. The board
  and detail views still work fully (`listEvents`/`getEventScores`/
  `getScore` all accept an explicit `release_id`); only the banner is
  reduced, and a `release-not-current-notice` says so explicitly rather
  than silently showing stale or wrong metadata.
- **The "reload to see the new release" button** (`ReleaseBanner`,
  background-poll case) strips the pinned hash before reloading
  (`location.href = pathname + search`), not a bare `location.reload()` —
  otherwise it would reload straight back into the URL P3-3b's own
  address-bar rewrite just pinned, defeating the button. Caught by
  `test_release_switch_mid_session_keeps_pinned_release_and_shows_notice`
  (P3-3a test, still required to pass).
- **Cache keys include `release_id`** everywhere a request is keyed
  (`useEventPage`, `useEventScores`, `useScoreDetail` — all take the pinned
  release id as an explicit hook argument, never `current`), so switching
  the release mid-session cannot bleed into an open detail view's fetch.
- **Board state survives a round trip to detail and back**: filters and the
  pagination cursor stack live in `App`'s own state, not the URL: the
  event/score board fetch (`useEventPage`) is only *issued* while
  `route.name === "board"` (an idle query, not a fetch, while viewing
  detail), and going back to the board re-issues the exact same
  `(filters, cursor)` query, reproducing the same page.

### Event detail (`src/components/EventDetail.tsx`)

Opened via a ticker link on any board row (`open-event-link`). Lazily calls
`GET /api/v1/events/{id}/scores` (never reuses the board's own embedded
`scores`, since that endpoint is the guide's dedicated lazy-fetch route) and
lists strategy, verdict/refusal, driver forecast, market implied move, entry
premium, headline expected return and DYN-SV choice per score — the same
columns the board shows, since `EventScoreSummary` is the same shape either
way. **Judgement call / real gap**: the deliverable text says event detail
should show "strategy, expiry, strike" — `EventScoreSummary` v1.1 (§6's
actual schema, confirmed in the Facts section) has neither field; only the
per-score columns above. Expiry/strike (and every other legacy field) are
shown one level down, in score detail's `display_record`. A ticker/date/
session header is shown when the event was opened from an already-loaded
board page (`App.tsx`'s `eventMetaCache`, populated from `EventPageItem`s
already on screen — never a new fetch); a bare deep link to an event route
shows just the event id, since `/events/{id}/scores` returns score
summaries only, no event metadata. Handles loading, error, empty (`readiness:
"unavailable"` or a genuinely empty population) and not-found (404
`UNKNOWN_EVENT`) states.

### Score detail (`src/components/ScoreDetail.tsx`)

Lazily calls `GET /api/v1/scores/{id}` with the pinned `release_id` (now
required — see the contract-decision note above). Shows `score_id` and
`release_id` explicitly, the raw verdict (`display_record.gate_pass`, if
present) plus a compatibility link (the legacy `gatePill` tree is still not
reproduced — unchanged gap from P3-3a), and:

- **`display_record` fields, grouped** (`src/displayFieldSpec.ts`
  `groupDisplayRecordFields`, `src/components/DisplayRecordFields.tsx`): a
  hand-transcribed mirror of `engine/v2/serving/bridge.py`'s
  `LEGACY_DISPLAY_MAPPING_V1` field→`unit` table, used as the "category" the
  deliverable asks for (the Python spec has no field literally named
  "category"; `unit` — identity/date/usd/percent/... — is the closest
  analog and the one this client uses). A `display_record` field this table
  doesn't know about falls into a synthetic "other" category rather than
  being dropped. Fields are sorted alphabetically within each category, and
  categories alphabetically ("other" last). Every value is printed via
  `fmtUnknown` (`src/format.ts`) — a generic passthrough formatter, never a
  computed/converted one. `payoff_curve` is pulled out of this generic list
  and rendered separately (below), not JSON-dumped twice.
- **Payoff curve** (`src/payoff.ts`, `src/components/PayoffChart.tsx`): a
  plain SVG polyline plus one `<circle>` per point, built ONLY from the
  curve's own `x`/`y` arrays — no interpolation, no added/dropped point,
  no computed value; the two numbers a point carries are scaled linearly
  into pixel space (guide §7: "map stored chart values to pixels" is
  allowed) and nothing else happens to them. Three distinct states: missing
  (field absent/null — no curve was saved), empty (a real `{x: [], y: []}`
  object with zero points), and drawn.
- **Engine evidence**: `engine_record` as raw JSON inside a collapsed
  `<details>` (`engine-evidence` test id) — never merged with or compared
  against `display_record`; this bridge already did that comparison
  offline (`bridge.py`'s own findings), the UI just shows the saved result.
- **Provenance**: `legacy_row_id`, `clock_id`, `event_ref`, `score_batch_ref`,
  `snapshot_ref`, `model_registry_artifact_refs`, and
  `unavailable_detail_reasons` (when nonempty) — all copied fields, no new
  ones. A "Back to event" link appears when the route carried an
  `event_id`, OR (bare score deep link) the fetched bridge's own
  `event_ref.event_id` — the bridge always knows its owning event even when
  the URL that opened it didn't.

Handles loading, error, and not-found (404 `UNKNOWN_SCORE`) states; a
detail-fetch failure never takes down the board (§7).

## Deferred (owners per the guide)

- **`gatePill` decision tree**: the legacy board's richer client-side
  decision tree over `flags`/`gate_score`/`gate_threshold` (several
  distinct N/A reasons: disabled, not sized, gate-declined, undetermined,
  arithmetic-only — `engine/dashboard/static/assets/app.js` ~line 235) is
  **not reproduced** anywhere in this client, board or score detail. Both
  views show the raw verdict (`verdict`/`gate_pass`) plus a compatibility
  link instead — a real, acknowledged gap, not a simplification that loses
  no information. Owner: P3-4 or later, per the coordinator's 2026-09-14
  instruction that this tree is out of scope for the bridge/UI slice.
- **Health/flags screens**: `GET /api/v1/operations` is implemented in the
  typed client (`DataClient.getOperations`) but still not surfaced in any
  view — the guide assigns a "dedicated full health/flags screen" to Phase
  6 (§5.5 point 3). Unchanged from P3-3a.
- **Sort control, portfolio/book aggregates, models/explorer/derivation/
  flags screens, job submission, live refresh** — Phase 6 (guide §3.2);
  unshipped screens link to the compatibility surface, not to a stub.
- **Event detail's expiry/strike**: see the Event detail section above —
  `EventScoreSummary` v1.1 has no such fields; only score detail's
  `display_record` carries them. Not a bug, a contract-shape mismatch
  between the deliverable's prose and the actual schema, recorded here
  rather than silently worked around by inventing fields.
- **No metadata for a pinned-but-non-current release**: see the Routing
  section above — §6 has no "get an arbitrary past release's metadata"
  route, only `current`. The board/detail data itself is unaffected.
- **P3-2 (real read API)** — separate, concurrent task, now confirmed
  compatible: `release_id` required (400 `RELEASE_ID_REQUIRED`) on
  `/scores/{id}` and `/events/{id}/scores`, and the real `Problem` envelope
  with no `title`/`status` alias, both verified directly against
  `engine/v2/serving/api.py` and matched in `client.ts`/the mock. Pointing
  `createHttpDataClient()` at the real server should need no further UI
  change; any drift found while integrating is P3-2's contract gap, not
  this client's.

## Allowlist

`ui/` files are opted back into the public repo's default-deny
`.gitignore` by exact path/extension (`package.json`, `package-lock.json`,
`tsconfig*.json`, `vite.config.ts`, `index.html`, `README.md`, `.gitignore`,
and `src/**/*.{ts,tsx,css}`) — TypeScript and JSON are not allowed globally
elsewhere in this repo. `ui/node_modules/` and `ui/dist/` are hard-blocked
(they are full of `.js`/`.css`, which the allowlist above would otherwise
let back in). The build output contains no source maps and no credentials —
verified: `grep sourceMappingURL ui/dist/assets/*` is empty after `npm run
build`.
