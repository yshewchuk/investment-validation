# `engine/v2/foundation` — architecture

## Purpose

Layer 0.5 in the root `/ARCHITECTURE.md`; owns pure session arithmetic ported from `engine/calendar.py`.

## Primary contracts and public interfaces

`market_calendar.build_calendar_sessions` projects sessions from observed source dates; `planned_exit_date` applies native strategy anchors. `CalendarEventKey` is structural so ops requests satisfy it without a foundation dependency on ops.

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

## Invariants

Observed-through is the source maximum and never the projected endpoint. Projection uses weekdays excluding computed annual US market holidays and documented one-off NYSE full-closure dates (2018-12-05 and 2025-01-09), and includes the first post-print session. STR-THRU/put-menu/DYN-SV exit first post-print; STR-RUNUP exits last pre-print.

## Diagrams

None: this component owns pure date calculations, not a job graph, data flow or state machine.
