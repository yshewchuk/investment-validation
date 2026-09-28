# `engine/v2/domain/generation` — architecture

## Purpose

Layer 4a of the root `ARCHITECTURE.md`'s layer table (§2): the structure
generator. Replaces legacy `engine/structures.py` (`ExpirySelector`,
`StrikeSelector`, `Structure`, `price_structure`) with a deterministic,
source-driven pair of pure functions — geometry resolution and quote
pricing — that need no calendar, store, catalog or other data dependency of
their own. Currently a single module, `structures.py` (`forecast_sizing.py`
and `fills.py`, named in the root doc's table as legacy modules this
package's scope also supersedes, do not exist as separate files yet; their
responsibilities are absorbed into `structures.py` for now).

## Primary contracts and public interfaces

Exported from `engine/v2/domain/generation/__init__.py`
(`engine.v2.domain.generation.__all__`; also enforced by this package's
`README.md` `<!-- public-interface: ... -->` marker):

- `generate(strategy: str, inputs: Mapping) -> Geometry` — resolves one
  strategy's legs (strikes, expiries, quantities) from `inputs`. Never reads
  a quote's bid/ask; that is `price`'s job.
- `price(geometry: Geometry, quotes: Mapping, fill_alpha: float = 0.5) ->
  Pricing` — prices a `Geometry`'s legs against a raw `quotes` mapping at a
  worst (`fill_alpha=0.0`) to best (`fill_alpha=1.0`) fill fraction.
- `has_resolvable_expiry(strategy, inputs, spot) -> bool` — a coarse
  EXISTENCE check (is there any contract domain to attempt at all), used by
  the scoring gate (`engine/v2/scoring/stages.py` `_resolve_geometry`) to
  tell a genuine `MISSING_EXPIRY` refusal apart from a captured-field gap
  `generate` can still resolve on its own. Deliberately not a full
  resolution: whether a date filter then leaves a survivor is `generate`'s
  job, and its refusal is more specific (see "Failure semantics").
- Dataclasses: `Geometry` (`strategy`, `spot`, `width`, `legs`, `refusal`,
  `detail`), `NativeLeg` (`name`, `right`, `side`, `quantity`, `strike`,
  `expiry`), `Pricing` (`strategy`, `spot`, `entry_cost`, `legs`, `refusal`),
  `PricedLeg` (`NativeLeg`'s fields plus `bid`, `ask`, `fill`, `cash_flow`).
- Exceptions: `GeometryRefusal` (raised by `generate`/`has_resolvable_expiry`
  helpers; carries the refusal code as both its message and `.code`, plus an
  optional `.detail`) and `PricingRefusal` (raised by `price`; the code is
  its message).
- `STRATEGIES` — every strategy name this package knows. `DISABLED` — the
  subset `generate` always refuses outright (`CAL-P`, `CND-P`, each mapped
  to `UNVALIDATED_STRUCTURE`).

## Inputs

`generate`/`has_resolvable_expiry` take one `inputs: Mapping[str, Any]` — no
store, catalog, calendar or network access. Every field is read as-is, never
computed by this package:

- `spot`, `forecast_abs_move` (or `forecast`), `width` — sizing.
- `strike`, `expiry`, `post_event_expiry` — a caller-already-resolved value,
  used exactly as given (bypasses native selection for that field entirely).
- `quotes` — a mapping keyed by `(right, strike, expiry)` (or an equivalent
  `"right:strike:expiry"` string), each value `{"bid", "ask"}`. The contract
  domain native selects from whenever `strike`/`expiry` is missing.
- `event_date`, `session` — the print date/session, read by the
  `first_post_event` expiry rule (STR-THRU and every put-ladder strategy).
- `entry_date` — the row's captured entry-session date (a raw fact, not a
  calculated answer — see `engine/v2/scoring/stages.py`'s
  `_check_stale_quote` docstring for the same characterization of this
  field). Read ONLY by STR-RUNUP's `first_dte_at_least` expiry rule (issue
  #95; see "Failure semantics").
- `resolved_legs` — an explicit leg list that bypasses geometry resolution
  entirely (the pinned/replay case).

`price` additionally takes the `quotes` mapping directly as its own
parameter, not through `inputs`.

## Outputs

`Geometry` and `Pricing` — plain, immutable dataclasses returned to the
caller, either populated with legs or carrying a `refusal`/`detail` pair
(and no legs). This package writes nothing itself: no store, no catalog row,
no file.

## Dependencies

Only the standard library (`dataclasses`, `datetime.date`, `math.isfinite`,
`typing`) — no calendar, store, catalog, or other `engine/v2` package. This
is deliberate: `generate`/`price` are pure functions over whatever `inputs`/
`quotes` the caller already resolved, so a date computation this package
needs (such as STR-RUNUP's DTE-from-entry rule) must be done from a field
the caller already captured, never from a calendar this package would
otherwise have to import.

Callers (checked by grep against the real tree, matching this package's
`README.md` "Consumers" section):

- `engine/v2/scoring/stages.py` — the only caller of `generate`, `price` and
  `has_resolvable_expiry`, from `_resolve_geometry`/`_resolve_pricing` in the
  native scoring pipeline.
- `engine/v2/scoring/source_inputs.py` — imports `DISABLED`, `STRATEGIES`.
- `engine/v2/scoring/__init__.py` — re-exports `Geometry`/`Pricing` under
  private aliases.
- `engine/v2/ops/cli.py` — imports `Geometry`, `Pricing` to reconstruct a
  captured `NativeScoreInputs` document for the read-only `ops rescore`
  command.

## External systems and libraries

None. No network, database, filesystem or subprocess access anywhere in this
package.

## Failure semantics

Every refusal is a raised exception carrying a specific code as its message
— never a bare exception with no code, and never a silent substitution of a
different answer than the one actually requested:

- `GeometryRefusal` codes raised by `generate`: `"<field> must be finite"`
  (a required numeric field, such as `spot`/`width`, is missing or not a
  finite number), `UNKNOWN_STRATEGY`,
  `ZERO_WIDTH`, `MISSING_EXPIRY` (no captured expiry and nothing listed to
  select from), `EXPIRY_NOT_LISTED:<date>` (a caller-supplied `expiry` not
  present in `quotes`), `NO_EXPIRY_ON_OR_AFTER:<date>` (STR-THRU/put-ladder
  `first_post_event`: no listed expiry survives the event-date/session
  filter), `NO_CHAIN`/`COARSE_LADDER` (a put-ladder leg has no listed
  strike, or two legs collide on one contract), and, for STR-RUNUP only
  (issue #95): `MISSING_ENTRY_DATE` (no `entry_date` captured),
  `INVALID_ENTRY_DATE:<value>` (`entry_date` present but not an ISO date),
  and `NO_EXPIRY_DTE_AT_LEAST:30` (no listed expiry reaches 30 DTE counted
  from `entry_date`).
- `PricingRefusal` codes raised by `price`: `INVALID_FILL_ALPHA`,
  `MISSING_QUOTE:<leg>`, `INVALID_QUOTE:<leg>`.
- This package never catches or downgrades its own refusals; the caller
  (`engine/v2/scoring/stages.py`) catches them and republishes the code as
  the resulting `Geometry`/`Pricing`'s `.refusal`.
- **STR-RUNUP's expiry rule (issue #95).** `_resolve_straddle_expiry`
  dispatches on `strategy`: a caller-supplied `expiry` bypasses every DTE
  rule unconditionally for every strategy (matching legacy's `fixed` kind).
  Otherwise, `strategy == "STR-RUNUP"` uses legacy's own `straddle_runup`
  rule (`engine/structures.py`, `ExpirySelector(kind="first_dte_at_least",
  target_dte=30)`): the earliest listed expiry whose DTE, counted from
  `entry_date` (not `event_date`), is at least 30. Every other strategy
  (STR-THRU, and every put-ladder strategy) is unaffected and keeps
  `first_post_event`: the earliest listed expiry on/after `event_date`, with
  the AMC/BMO distinction applied when `session` is known.

## Invariants

- `generate`/`price` never mutate `inputs`/`quotes`; every result is a fresh
  dataclass.
- A captured `expiry`/`strike`/`resolved_legs` field, when present, is used
  exactly as given — native selection only runs when the caller has nothing
  more specific to offer for that field.
- For STR-THRU/STR-RUNUP, expiry resolution happens strictly before strike
  selection (`_select_listed_straddle`): choosing by strike distance first
  could silently pick a strike from the wrong expiry, whenever a later
  expiry happened to list a strike nearer to spot.
- No date filter (event-date, session, or entry-date/DTE) is ever silently
  downgraded to a different answer than the one it was asked to find: a
  filter with no survivor refuses by name rather than substituting the
  nearest candidate it did find.

## Diagrams

Not added: this package's control flow is a single function (`generate`)
branching on `strategy` and on which fields `inputs` already carries, fully
described above; a diagram would not tell a reader anything the code and
this doc do not already state.
