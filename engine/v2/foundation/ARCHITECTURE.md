# `engine/v2/foundation` — architecture

## Purpose

Layer 0.5 in the root `/ARCHITECTURE.md`; owns pure session arithmetic ported from `engine/calendar.py`.

## Primary contracts and public interfaces

`market_calendar.build_calendar_sessions` projects sessions from observed source dates; `planned_exit_date` applies native strategy anchors. `CalendarEventKey` is structural so ops requests satisfy it without a foundation dependency on ops.

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

## Inputs

The factory accepts observed ISO session days and an event-through day. Exit planning accepts a structural event key and a `CalendarSessions` value.

## Outputs

`build_calendar_sessions` returns sorted unique observed and projected days with the unchanged maximum observed day. `planned_exit_date` returns an ISO session day.

## Dependencies

Pure standard-library arithmetic with no upward-layer imports. `ops.nightly_calendar_inputs` composes its calendar and exit helpers for pinned raw-row inputs; the later `nightly_raw_row_producer` remains a planned caller.

## External systems and libraries

None. This module does not access repositories, providers, loaders, models, legs or prices.

## Failure semantics

Invalid/empty sessions, malformed/missing dates (`None`, `NaT`), unknown strategy/session, or insufficient pre-print or post-print anchor coverage raise `CalendarInputError(code="INVALID_REQUEST")`. The bounded search for the first future session stops at Python's maximum representable date: if a session remains in range it is returned, and if none remains the function raises the same typed refusal rather than leaking date-arithmetic overflow. R1–R6: source failures propagate at the adapter; no cache or retries; deterministic for the same inputs; read-only; no partial result or writes; stable output for the same request.

The shared typed-document decoder reports `UNSUPPORTED_VERSION` before `UNKNOWN_FIELD` when a document has both an unsupported schema version and undeclared fields. Supported versions still report `UNKNOWN_FIELD` for undeclared fields.

`score_population.population_key` preserves the `ticker|strategy|event_date` identity;
`population_difference` returns sorted missing and unplanned keys for ops scoring
and the serving bridge. An extra `DYN-SV` key is allowed only when its exact
ticker/date has a planned non-chooser key. Every explicit planned key, including
`DYN-SV`, remains required. Missing scalar keys retain their original values and
normal sort order; incomparable mixed types use type-name/string-value order.
Only string planned keys can authorize a derived chooser. These helpers perform no I/O, caching, retries,
transactions or writes; equal inputs yield equal differences without mutation.

## Invariants

Observed-through is the source maximum and never the projected endpoint. Projection uses weekdays excluding computed annual US market holidays and documented one-off NYSE full-closure dates (2012-10-29, 2012-10-30, 2018-12-05, and 2025-01-09), and includes the first post-print session. STR-THRU/put-menu/DYN-SV exit first post-print; STR-RUNUP exits last pre-print.

## Diagrams

None: this component owns pure date calculations, not a job graph, data flow or state machine.
