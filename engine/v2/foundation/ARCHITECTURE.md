# `engine/v2/foundation` — architecture

## Purpose

Layer 0.5 in the root `/ARCHITECTURE.md`; owns pure session arithmetic ported from `engine/calendar.py`.

## Primary contracts and public interfaces

`iter_canonical_json(value, *, fragments=None)` yields canonical text chunks;
joining them gives `canonical_json` exactly, including its nonfinite normalization.
It owns no output I/O or byte ceiling. `experiments.v2_magnitude_search_run`
consumes it to stage and byte-bound a complete report before stdout emission.

`market_calendar.build_calendar_sessions` projects sessions from observed source dates; `planned_exit_date` applies native strategy anchors. `CalendarEventKey` is structural so ops requests satisfy it without a foundation dependency on ops.

`market_horizon.trading_days_before_print(calendar, *, entry_date, event_date,
session)` returns the signed session-index distance from entry (previous listed
session for an unlisted day) to the last pre-print close: AMC includes the
event close; BMO excludes it. Zero is valid. Both dates require calendar
coverage; the helper counts supplied sessions, preserving historical gaps.

`experiment_holdouts.ExperimentHoldouts` owns the shared, versioned membership
definitions from [#373](https://github.com/yshewchuk/investment-validation/pull/373).
Random membership hashes immutable canonical event identity, independently of
snapshot contents and arrival order; monthly release cannot change it.
The rolling set includes the explicit as-of month and its five preceding
calendar months (user decision 2026-10-08), inclusive of both boundary months.
Future-month dates, timezone-bearing timestamps and non-midnight timestamps
are ambiguous rather than released into selection.
Classification retains both labels for overlap and labels invalid identity/date
inputs `ambiguous`; no holdout read or authorization API is supplied.
Invalid as-of month raises `ValueError`; research converts it to its typed refusal.

`final_holdout` strictly decodes immutable declaration/request contracts and projects verified registration metadata into a request, without I/O or authority. It validates exact schema, content-addressed references, hashes, identities and shared month/membership versions; invalid metadata raises `ValueError`/`DocumentError` for caller translation.
`experiments.native_registration` uses these pure helpers; catalog registration verification stays with its existing owner. Request identity is the existing experiment ID; recipe/model/completeness references come from the sealed registration, not preparation-time overrides.

## Inputs

The factory accepts observed ISO session days and an event-through day. Exit planning accepts a structural event key and a `CalendarSessions` value.

## Outputs

`build_calendar_sessions` returns sorted unique observed and projected days with the unchanged maximum observed day. `planned_exit_date` returns an ISO session day.

## Dependencies

Pure standard-library arithmetic with no upward-layer imports. `ops.nightly_calendar_inputs` composes its calendar, exit and horizon helpers for pinned raw-row inputs, reached by the native worker through `nightly_raw_row_producer`.

## External systems and libraries

None. This module does not access repositories, providers, loaders, models, legs or prices.

## Failure semantics

Invalid/empty sessions, malformed/missing dates (`None`, `NaT`), unknown strategy/session, or insufficient pre-print or post-print anchor coverage raise `CalendarInputError(code="INVALID_REQUEST")`. The bounded search for the first future session stops at Python's maximum representable date: if a session remains in range it is returned, and if none remains the function raises the same typed refusal rather than leaking date-arithmetic overflow. R1–R6: source failures propagate at the adapter; no cache or retries; deterministic for the same inputs; read-only; no partial result or writes; stable output for the same request.

For `trading_days_before_print` only, missing date/anchor coverage raises the
`CalendarCoverageError` subclass (same code), distinguishable from malformed
calendar/date/session input. Negative distances are returned for caller policy;
no calendar days, missing sessions or horizon defaults are invented.

The shared typed-document decoder reports `UNSUPPORTED_VERSION` before `UNKNOWN_FIELD` when a document has both an unsupported schema version and undeclared fields. Supported versions still report `UNKNOWN_FIELD` for undeclared fields.

| Typed constructor condition | Outcome |
|---|---|
| Constructor `DocumentError` path beginning with `$` at nested/custom prefix | Prefix the path, including already-qualified `$` paths; preserve code/message and original cause |
| Root constructor error or constructor path not beginning with `$` | Retain path |

`score_population.population_key` preserves the `ticker|strategy|event_date` identity;
`population_difference` returns sorted missing and unplanned keys for ops scoring
and the serving bridge. An extra `DYN-SV` key is allowed only when its exact
ticker/date has a planned non-chooser key. Every explicit planned key, including
`DYN-SV`, remains required. Missing scalar keys retain their original values and
normal sort order; incomparable mixed types use type-name/string-value order.
Only string planned keys can authorize a derived chooser. These helpers perform no I/O, caching, retries,
transactions or writes; equal inputs yield equal differences without mutation.

## Invariants

Observed-through is the source maximum and never the projected endpoint. Projection uses weekdays excluding computed annual US market holidays and documented one-off NYSE full-closure dates (2012-10-29, 2012-10-30, 2018-12-05, and 2025-01-09); Martin Luther King Jr. Day is a scheduled market holiday beginning in 1998. Projection includes the first post-print session. STR-THRU/put-menu/DYN-SV exit first post-print; STR-RUNUP exits last pre-print.

## Diagrams

None: this component owns pure date calculations, not a job graph, data flow or state machine.
