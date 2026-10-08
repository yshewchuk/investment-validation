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
- `resolve_expiry(strategy: str, inputs: Mapping[str, Any], expiries: list[str]) -> str`
  — resolves only an expiry from sorted, distinct ISO candidate days by
  delegating to the existing native resolver. It does not select strikes or
  legs, price quotes, or mutate `inputs` or `expiries`.
- Dataclasses: `Geometry` (`strategy`, `spot`, `width`, `legs`, `refusal`,
  `detail`), `NativeLeg` (`name`, `right`, `side`, `quantity`, `strike`,
  `expiry`), `Pricing` (`strategy`, `spot`, `entry_cost`, `legs`, `refusal`),
  `PricedLeg` (`NativeLeg`'s fields plus `bid`, `ask`, `fill`, `cash_flow`).
- Exceptions: `GeometryRefusal` (raised by `generate`,
  `has_resolvable_expiry` helpers, and `resolve_expiry`; carries the refusal
  code as both its message and `.code`, plus an optional `.detail`) and
  `PricingRefusal` (raised by `price`; the code is its message).
- `STRATEGIES` — every strategy name this package knows. `DISABLED` — the
  subset `generate` always refuses outright (`CAL-P`, `CND-P`, each mapped
  to `UNVALIDATED_STRUCTURE`).

## Inputs

`generate`/`has_resolvable_expiry` take one `inputs: Mapping[str, Any]` — no
store, catalog, calendar or network access. Every field is read as-is, never
computed by this package:

- `spot`, `forecast_abs_move` (or `forecast`), `width` — sizing.
- `strike`, `expiry` — a caller-already-resolved value, used exactly as
  given wherever it is checked (bypasses native selection for that field).
- `post_event_expiry` — an alternate already-resolved expiry. One bypass rule
  covers every path (`_expiry()`, `has_resolvable_expiry`, and
  `_resolve_straddle_expiry` for STR-THRU/STR-RUNUP): the captured expiry is
  `expiry` if non-empty, else `post_event_expiry`, so `expiry` wins when both
  are present. `EXPIRY_NOT_LISTED` is raised only when a captured expiry is
  checked against a candidate list: STR-THRU/STR-RUNUP with `strike` or
  `expiry` missing AND at least one strike listed as both call and put
  (candidates = expiries of those common strikes), or a `resolve_expiry` call
  (candidates = the caller's list). Every other path uses the captured value
  as given, unvalidated against `quotes`: STR-THRU/STR-RUNUP with both
  `strike` and `expiry` supplied; STR-THRU/STR-RUNUP whose chain is empty or
  has no call/put common strike (selection returns nothing, so the captured
  value is the fallback); and every put-ladder strategy.
- `quotes` — a mapping keyed by `(right, strike, expiry)` (or an equivalent
  `"right:strike:expiry"` string), each value `{"bid", "ask"}`. The contract
  domain native selects from whenever `strike`/`expiry` is missing.
- `event_date`, `session` — the print date/session, read by the
  `first_post_event` expiry rule (STR-THRU and every put-ladder strategy).
- `entry_date`, `quote_date` — both raw facts, not calculated answers (see
  `engine/v2/scoring/stages.py`'s `_check_stale_quote` docstring for the
  same characterization). Read ONLY by STR-RUNUP's `first_dte_at_least`
  expiry rule (issue #95; see "Failure semantics"), which anchors its DTE
  count on `quote_date` when captured, falling back to `entry_date` — legacy
  anchors the same count on the chain's `obs_date`, which is `quote_date`
  (defaulting to `entry_date` when no stale-quote substitution happened).
- `resolved_legs` — an explicit leg list meant to bypass geometry resolution
  entirely (the pinned/replay case). **Known gap (issue #114, not fixed
  here):** `generate()` currently resolves expiry/strike BEFORE it checks
  `resolved_legs`, so a pinned input with legs but no top-level
  `expiry`/`strike` can still raise a refusal from ordinary resolution
  before the bypass is ever reached.

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

Most refusals are raised exceptions carrying a specific code as their
message — never a bare exception with no code, and never a silent
substitution of a different answer than the one actually requested. A few
are instead returned as data, never raised:

- `DISABLED` strategies: `generate` returns a `Geometry` whose `.refusal`
  is `UNVALIDATED_STRUCTURE` (empty legs); it never raises for this case.
- `price` given an already-refused `Geometry`: returns a `Pricing` carrying
  that SAME `.refusal` unchanged (empty legs, zero cost); it never
  re-raises or re-derives a different code.
- `has_resolvable_expiry` always returns a plain `bool`; it never raises,
  even where the resolution it is checking for would itself refuse.
- `resolve_expiry` raises `GeometryRefusal("MISSING_EXPIRY")` for an empty
  candidate list. For non-empty candidates, it preserves the native
  resolver's existing `GeometryRefusal` codes, including fixed-expiry and
  strategy-specific date-filter refusals.

Raised exceptions:

- `GeometryRefusal` codes raised by `generate`: `"<field> must be finite"`
  (a required numeric field, such as `spot`/`width`, is missing or not a
  finite number), `UNKNOWN_STRATEGY`,
  `ZERO_WIDTH`, `MISSING_EXPIRY` (no captured expiry and nothing listed to
  select from), `EXPIRY_NOT_LISTED:<date>` (a captured `expiry`/`post_event_expiry` not
  present in `quotes`), `NO_EXPIRY_ON_OR_AFTER:<date>` (STR-THRU/put-ladder
  `first_post_event`: no listed expiry survives the event-date/session
  filter), `NO_CHAIN`/`COARSE_LADDER` (a put-ladder leg has no listed
  strike, or two legs collide on one contract), and, for STR-RUNUP only
  (issue #95): `MISSING_ENTRY_DATE` (neither `quote_date` nor `entry_date`
  captured), `INVALID_ENTRY_DATE:<value>` (an anchor date is captured but
  not an ISO date), and `NO_EXPIRY_DTE_AT_LEAST:<threshold>` (no listed
  expiry reaches STR-RUNUP's own minimum-DTE threshold, counted from
  `quote_date` when captured, else `entry_date`; the threshold itself is a
  strategy parameter, not documented here — see `_resolve_first_dte_at_least`
  and legacy's `straddle_runup` factory).
  `GeometryRefusal` also surfaces from `price()` for a non-numeric/non-finite
  `fill_alpha` (via the shared `_finite_float` helper, same message shape as
  the sizing fields above) — a DIFFERENT exception type than the next bullet
  raises for a `fill_alpha` that parses fine but is out of range.
- `PricingRefusal` codes raised by `price`: `INVALID_FILL_ALPHA` (a finite
  `fill_alpha` outside `[0, 1]` — note a non-finite one is `GeometryRefusal`
  instead, per above), `MISSING_QUOTE:<leg>`, `INVALID_QUOTE:<leg>` (a
  negative bid, or an ask below its bid).
- This package never catches or downgrades one of its own RAISED refusals;
  the caller (`engine/v2/scoring/stages.py`) catches them and republishes
  the code as the resulting `Geometry`/`Pricing`'s `.refusal`.
- **STR-RUNUP's expiry rule (issue #95).** `_resolve_straddle_expiry`
  dispatches on `strategy`: a captured expiry (`expiry`, else `post_event_expiry`) bypasses every DTE
  rule unconditionally for every strategy (matching legacy's `fixed` kind).
  Otherwise, `strategy == "STR-RUNUP"` uses legacy's own `straddle_runup`
  rule (`engine/structures.py`, `ExpirySelector(kind="first_dte_at_least",
  ...)`): the earliest listed expiry whose DTE reaches STR-RUNUP's own
  minimum-DTE threshold, counted from `quote_date` when captured (legacy's
  chain `obs_date`, which a stale-quote fallback can set EARLIER than
  `entry_date` — `engine/score.py`'s `_fresh_quote_date`/`obs_date=
  result.quote_date`), falling back to `entry_date` only when `quote_date`
  is not captured (not `event_date` either way). Every other strategy
  (STR-THRU, and every put-ladder strategy) is unaffected and keeps
  `first_post_event`: the earliest listed expiry on/after `event_date`, with
  the AMC/BMO distinction applied when `session` is known — or, when
  `event_date` itself is not captured, the earliest listed expiry
  unconditionally (a deterministic default, not a refusal).
- **Known gaps, not fixed here** (see "Inputs" for detail): `generate()` resolves expiry/strike before it checks
  `resolved_legs` (issue #114).

## Invariants

- `generate`/`price` never mutate `inputs`/`quotes`; every result is a fresh
  dataclass.
- A captured `expiry`/`strike` field, when present, is used exactly as
  given — native selection only runs when the caller has nothing more
  specific to offer for that field. `resolved_legs` is MEANT to be the same
  kind of bypass but currently is not always one in practice — see the
  known gap in "Inputs"/"Failure semantics" (issue #114).
- For STR-THRU/STR-RUNUP, expiry resolution happens strictly before strike
  selection (`_select_listed_straddle`): choosing by strike distance first
  could silently pick a strike from the wrong expiry, whenever a later
  expiry happened to list a strike nearer to spot.
- No date filter (event-date, session, or entry-date/DTE) is ever silently
  downgraded to a different answer than the one it was asked to find: a
  filter with no survivor refuses by name rather than substituting the
  nearest candidate it did find.

## Diagrams

STR-THRU/STR-RUNUP expiry dispatch (`_resolve_straddle_expiry`), the
control flow issue #95 changed:

```mermaid
flowchart TD
    A["caller-supplied expiry present?"] -->|yes, listed| B["use it\n(legacy fixed)"]
    A -->|yes, not listed| C["raise EXPIRY_NOT_LISTED"]
    A -->|no| D{"strategy == STR-RUNUP?"}
    D -->|yes| E["earliest listed expiry with\nDTE from quote_date\n(or entry_date) >= threshold"]
    E -->|none qualifies| F["raise NO_EXPIRY_DTE_AT_LEAST"]
    E -->|found| G["use it"]
    D -->|no\n(STR-THRU / put-ladder)| J{"event_date known?"}
    J -->|no| K["earliest listed expiry\n(deterministic default)"]
    K --> G
    J -->|yes| H["earliest listed expiry\non/after event_date"]
    H -->|none survives| I["raise NO_EXPIRY_ON_OR_AFTER"]
    H -->|found| G
```

Not added for the rest of the package: strike selection, put-ladder
placement and pricing are each already single, short, linearly-described
functions with no branching this diagram's level would clarify further.
