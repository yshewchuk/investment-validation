# `engine/v2/foundation` — architecture

## Purpose

Layer 0.5 in the root `/ARCHITECTURE.md`; owns pure session arithmetic ported from `engine/calendar.py`.

## Primary contracts and public interfaces

`iter_canonical_json(value, *, fragments=None)` yields canonical text chunks;
joining them gives `canonical_json` exactly, including its nonfinite normalization.
It owns no output I/O or byte ceiling. `experiments.v2_magnitude_search_run`
consumes it to stage and byte-bound a complete report before stdout emission.

`market_calendar.build_calendar_sessions` projects sessions from observed source dates; `planned_exit_date` applies native strategy anchors. `CalendarEventKey` is structural so ops requests satisfy it without a foundation dependency on ops.

`market_calendar.is_scheduled_session(day)` is the foundation package's one
public weekday/non-holiday predicate for a session day: projection, calendar
validation and `market_schedule.latest_completed_session` call it instead of
re-deriving holiday membership. It is not the repository's only session
predicate — `data.computed_moves` and `research._pricing` retain their own
pandas-rule projections bound to their listed-session data, outside this
package's contract. The scheduled America/New_York close hour is a private
`market_schedule._scheduled_close_hour` rule, not a `market_calendar` public
API: 13 on the three modelled early-close days (July 3 when it is a scheduled
session — an observed July 4 landing on July 3 makes it a full holiday, not a
session; the Friday after Thanksgiving; Christmas Eve when it is a scheduled
session) and 16 otherwise. Exceptional exchange-announced half-days beyond
these three modelled rules are not covered — on those the serving clock keeps
the regular 16:00 boundary, a bounded conservative lag, never authoritative
early-close coverage. `latest_completed_session(now=None, frozen_at=None)`
returns the latest completed NYSE session at `now`, or at the snapshot
`frozen_at` when `now` is omitted: today once its modelled close hour has
passed and it is a scheduled session, else the most recent earlier session
day. `frozen_at=None` and the exact string `"unknown"` mean live; every other
empty, malformed or non-string value raises `ValueError` rather than silently
switching to live time. That validation of a supplied `frozen_at` runs even
when `now` is given explicitly: `now` selects the clock, it never excuses a
malformed `frozen_at`.

| Serving-clock condition | Outcome |
|---|---|
| `now` supplied, or `frozen_at` a parseable ISO timestamp | Session at that moment (naive treated Eastern) |
| `frozen_at` absent or exactly `"unknown"`, `now` omitted | Live wall clock |
| `frozen_at` empty, malformed or non-string -- even when `now` is supplied | `ValueError`; never live time |
| Scheduled session day, clock at or after its modelled close-hour boundary | That day |
| Before the close hour, weekend or holiday | Most recent earlier session day |
| July 3 session / Black Friday / session Christmas Eve at 12:59 vs 13:00 | Prior session / that day |

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
`experiment_holdouts.calendar_date` coerces no noncanonical value and performs no I/O.

| Calendar-day input | Outcome |
|---|---|
| Canonical ISO date, date object or naive midnight datetime | Same calendar date |
| Noncanonical/invalid date, wrong type, aware or non-midnight datetime | `ValueError` |

`final_holdout` strictly decodes immutable declaration/request contracts and projects verified registration metadata into a request, without I/O or authority. It validates exact schema, content-addressed references, hashes, identities and shared month/membership versions; invalid metadata raises `ValueError`/`DocumentError` for caller translation.
`experiments.native_registration` uses these pure helpers; catalog registration verification stays with its existing owner. Request identity is the existing experiment ID; recipe/model/completeness references come from the sealed registration, not preparation-time overrides.

`experiment_exits.valid_exit_recipe` owns the closed fixed-day/target-stop
declaration predicate shared by ops planning and research walking;
`valid_threshold_units` also admits the RUNUP request's unit vocabulary.
Malformed shapes, horizons, units or signed finite-number thresholds return
`False`, without mutation, I/O, coercion or retries. Callers retain their typed
refusal envelopes; fill/source admission and position pricing stay with them.

## Inputs

The factory accepts observed ISO session days and an event-through day. Exit planning accepts a structural event key and a `CalendarSessions` value.

## Outputs

`build_calendar_sessions` returns sorted unique observed and projected days with the unchanged maximum observed day. `planned_exit_date` returns an ISO session day.

## Dependencies

Pure standard-library arithmetic with no upward-layer imports. `ops.nightly_calendar_inputs` composes its calendar, exit and horizon helpers for pinned raw-row inputs, reached by the native worker through `nightly_raw_row_producer`.

## External systems and libraries

None. This module does not access repositories, providers, loaders, models, legs or prices.

## Failure semantics

Invalid/empty sessions, malformed/missing dates (`None`, `NaT`), unknown strategy/session, or insufficient pre-print or post-print anchor coverage raise `CalendarInputError(code="INVALID_REQUEST")`. A malformed, empty or non-string `frozen_at` other than `None`/`"unknown"` raises `ValueError`; it never falls back to live time. The bounded search for the first future session stops at Python's maximum representable date: if a session remains in range it is returned, and if none remains the function raises the same typed refusal rather than leaking date-arithmetic overflow. R1–R6: source failures propagate at the adapter; no external-source or request-result cache or retries; deterministic for the same inputs; read-only; no partial result or writes; stable output for the same request.

Only pure annual holiday calculations use a process-local stdlib LRU (128 years),
returning immutable sets that include one-off closure dates. Observed sessions,
anchors and external data are not cached. `planned_exit_date` validates
`calendar.days`; it does not read `calendar.observed_through`.

| Holiday-cache condition | Outcome |
|---|---|
| Year is retained in the process-local LRU | Reuse its immutable scheduled/one-off holiday set. |
| Year is absent, including after eviction | Recompute the same pure year rules; retain at most 128 years, evicting the least recently used. |
| `planned_exit_date` receives invalid `calendar.days`, whether cache is warm or cold | Empty/non-tuple input, invalid dates, non-increasing/duplicate days, weekends or holidays raise `CalendarInputError`. |

For `trading_days_before_print` only, missing date/anchor coverage raises the
`CalendarCoverageError` subclass (same code), distinguishable from malformed
calendar/date/session input. Negative distances are returned for caller policy;
no calendar days, missing sessions or horizon defaults are invented.

The shared typed-document decoder reports `UNSUPPORTED_VERSION` before `UNKNOWN_FIELD` when a document has both an unsupported schema version and undeclared fields. Supported versions still report `UNKNOWN_FIELD` for undeclared fields.

| Typed constructor condition | Outcome |
|---|---|
| Constructor `DocumentError` path beginning with `$` at nested/custom prefix | Prefix the path, including already-qualified `$` paths; preserve code/message and original cause |
| Root constructor error or constructor path not beginning with `$` | Retain path |

| Exit-admission condition | Outcome |
|---|---|
| `valid_exit_recipe`: non-mapping, unknown/missing fields or kind, boolean/nonpositive/non-integer horizon, invalid units, or boolean/nonfinite/unrepresentable/wrong-sign/non-numeric thresholds | Return `False`; no coercion, mutation, I/O or retries. The caller supplies its typed refusal envelope. |
| `valid_threshold_units`: anything other than `quote_price_times_quantity` or `entry_debit_fraction` | Return `False`; no default or normalization of an explicit value. |

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
