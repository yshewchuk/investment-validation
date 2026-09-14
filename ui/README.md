# `ui/` — v2 board client (P3-3a)

React + TypeScript + Vite client for the rearchitecture Phase 3 board view,
built against the §6 read API contract
(`guides/rearchitecture_phase3_parity_launch.md`). The real FastAPI server
(P3-2) is a separate, concurrent task — this slice was built and tested
against a stdlib mock (`tests/fixtures/v2_ui_mock_api.py`) shaped verbatim
from `engine/v2/contracts/serving.py`, so it needs no changes when P3-2
lands with the same field names.

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

**Board view only** (`src/App.tsx`). The score-detail view is P3-3b, out of
scope here; every row instead links to the pinned compatibility surface
(`/release/<release_id>/index.html`) via `compatibilityLink()`
(`src/format.ts`).

- **Release pinning** (`src/hooks.ts` `usePinnedRelease`): `GET
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
  link, per the coordinator's 2026-09-14 instruction. The legacy board's
  richer `gatePill` decision tree (`app.js` ~line 235 — several distinct
  N/A reasons: disabled, not sized, gate-declined, undetermined,
  arithmetic-only) is **not reproduced**. This is a real gap, not a
  simplification that loses no information: a P3-3a reader sees `false`
  where the legacy board would say "FAIL" or a rule-specific "N/A" with a
  reason. **Owner: P3-3b or P3-4.**
- **Pagination**: forward-only (`next_cursor`), with a client-side cursor
  stack for "Previous" (judgement call — not required by §7, but cheap and
  the mock/real API both support fetching the same page again since the
  cursor for it was already seen). Count shows `total_matching`, the
  complete filtered population, never just the visible page.
- **States covered**: loading, unauthenticated, no current release (503),
  API error (events fetch), no matches (0 results), a row whose readiness
  is `unavailable` with no scores, null field vs. a real zero,
  a refusal row, and paging.

## Deferred (owners per the guide)

- **Score detail view** — P3-3b. Every row already links to the
  compatibility surface as the interim path.
- **`gatePill` decision tree** (see above) — P3-3b/P3-4.
- **Sort control, portfolio/book aggregates, models/explorer/derivation/
  flags screens, job submission, live refresh** — Phase 6 (guide §3.2);
  unshipped screens link to the compatibility surface, not to a stub.
- **`GET /api/v1/operations`** is implemented in the typed client
  (`DataClient.getOperations`) but not yet surfaced in a view — the guide
  assigns a "dedicated full health/flags screen" to Phase 6 (§5.5 point 3).
- **P3-2 (real read API)** — separate, concurrent task. This client and the
  mock fixture both use the exact §6 route paths and contract field names,
  so pointing `createHttpDataClient()` at the real server should need no UI
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
