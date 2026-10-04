# `ui/` architecture

## Purpose
Layer 8 presentation in the [root architecture](../ARCHITECTURE.md): the existing
React/TypeScript app owns application components, layout, navigation and rendering.
Its presentation scope includes the legacy dashboard views still awaiting migration.
## Primary contracts and public interfaces
`src/App.tsx` composes board/event/score and native-parity views; `src/routes.ts` handles routes.
`src/api/client.ts` is the sole fetch boundary to the authenticated serving API.
Wire shapes and shipped views are detailed in the [README](README.md).

## Inputs
Saved JSON release metadata, paginated events, score details and operations data
from [serving](../engine/v2/serving/ARCHITECTURE.md), plus route/filter state.
Native parity renders the summary from `/api/v1/native_parity`. The typed
client also exposes `/mismatches` and `/unpaired` under that prefix; the
summary component does not request those pages.

## Outputs
React DOM and chart pixels, navigation/deep links, loading/error/refusal states
and links to the same pinned compatibility release. No engine records are written.

## Dependencies
React components use typed client/hooks; the browser calls serving over HTTP.
No direct scoring/model/ledger/evaluation/provider access; browser users enter
through `App`; integration/browser tests exercise the same API contract.

## External systems and libraries
React, TypeScript and Vite; browser fetch with the same-origin `operations_token`
cookie. Build output is static assets; serving those bytes is transport.

## Failure semantics
| Condition | Outcome |
|---|---|
| Loading, empty, 401 or unknown identity | Explicit loading/empty/auth/refusal state; detail failure preserves board. |
| Current release changes | Announce only; reload opts in. Cache keys retain explicit release pin. |
| Parity loading, 401 or `no_report` | Explicit loading/auth/no-report state, independent of release resolution; no-report rendering reads status only. |
| Parity `stale` | Banner with saved identity, counts and refusal reasons still shown. |
| Parity `unavailable` | Explicit error; failed data is withheld, never presented as an empty comparison. |
| Parity zero counts or no refusal reasons | Saved zero counts remain visible; an explicit message identifies absent refusal reasons. |
| Parity `partial` | Saved counts remain visible with an alert that refusal data is incomplete. |
| Retry or navigation | Reads keep pinned identity; no durable transaction or partial publication. |

## Invariants
All new application rendering belongs here. Serving returns data/API responses
and transports commands; application HTML/DOM scripts there violate the boundary.
Format saved financial values; do not fit, rescore, derive financial evidence or
fetch vendors. Preserve saved replay clocks and provenance without relabelling.
Existing operations HTML/legacy previews are compatibility exceptions with
migration outstanding, not evidence that a React application view is shipped.
Native parity summary presentation belongs here; mismatch/unpaired detail is
outside this component. The operations parity HTML/JSON preview is retired.
## Diagrams
```mermaid
flowchart LR
  App[React App and components] --> Hooks[release-pinned hooks]
  Hooks --> Client[typed client; same-origin cookie]
  Client --> API[authenticated serving JSON API]
  API --> Saved[saved release projections and records]
```
