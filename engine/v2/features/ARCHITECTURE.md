# `engine/v2/features` — architecture

## Purpose

Registered causal feature recipes and their population policy
(`recipes.py`, `context.py`), and — as of this change — a pure-math layer
(`panel_math.py`) that a later native feature-computation path will build
on. It replaces (per the system rearchitecture's §4.4) `features.py`,
`data/features/panel.py`, and `data/features/tier4.py`, but does so
incrementally: this change ports the first slice of that math without yet
wiring anything to call it.

## Primary contracts and interfaces

- `FeatureRegistry.get(recipe_id)` / `default_feature_registry()` —
  namespaced `FeatureRecipe` lookup (contract in
  `engine.v2.contracts.scoring.FeatureRecipe`: recipe id/version, its
  `input_contracts`, `output_columns`, `source_scope`/`history_scope`,
  `observation_cutoff_rule`, `fallback_policy`, `determinism_policy`).
- `FeatureContextPlanner.request(...)` — builds a `FeatureRequest` from
  event refs, a snapshot ref, and recipe refs. It resolves every
  `recipe_ref` against the registry (`FeatureRegistry.get`, raising
  `KeyError` for an unknown id) and records each event's `decision_at` and
  visibility as decision contexts on the request; it does **not** itself
  check any timestamp for timezone-awareness or causal ordering
  (`engine.v2.contracts.scoring.FeatureRequest`).
- `FeatureContextPlanner.frame(...)` — turns that `FeatureRequest` plus
  rows of already-computed values into an immutable `FeatureFrame`. The
  timezone and causal-cutoff checks live here, not in `request()`: every
  row's `observed_at` and its event's `decision_at` must each be an
  explicit, timezone-aware timestamp, and `observed_at` must be
  on-or-before the decision cutoff, or the call raises
  `FeatureContextError` (`engine.v2.contracts.scoring.FeatureFrame`).
- `panel_math` — six pure functions with no `FeatureRecipe`/`FeatureFrame`
  wrapping of their own (see below). Nothing in this package or any other
  yet constructs a `FeatureRecipe` or calls `FeatureContextPlanner` for
  this math; that binding is future work.

### `panel_math` functions

| Function | Shape | Status |
|---|---|---|
| `history_features(prior_moves, prior_abs) -> dict` | plain sequences in, plain mapping out | byte-identical copy of `engine.data.features.panel.history_features` |
| `advance_history(last_row) -> dict` | one prior panel row in, fresh mapping out | arithmetic port of `engine.features.advance_history`, resuming stored aggregates without rereading truncated panel history |
| `_causal_ema(history, span) -> float \| None` | plain list in, scalar out | byte-identical copy of `engine.data.features.panel._causal_ema`; `history_features`'s helper |
| `_anchor_index(series_dates, event_dates, as_of_dates) -> np.ndarray` | numpy arrays in and out | byte-identical copy of `engine.data.features.panel._anchor_index` |
| `add_implied_history(df) -> pd.DataFrame` | DataFrame in, DataFrame out | byte-identical copy of `engine.data.features.panel.add_implied_history` — the one function in this module that is pandas-shaped, because its ported body is itself pandas code (`groupby`/`shift`/`expanding`) |
| `daily_state_lookup(rows, decision_date) -> dict` | plain sequence of mappings + a date in, plain mapping out | new: factors the per-`(ticker, as_of)` extraction rule out of `engine.features.daily_state_frame`, which is batched (DataFrame-in/DataFrame-out) and stays in legacy unchanged |

### Runup feature math

`runup_math.add_runup_features(frame, prices_by_ticker, as_of_column)`
returns a new event DataFrame, sorted by ticker and event date, with
`signed_streak`, `ema12r_abs`, `dist_high`, `dist_ema`, `ret5`, `ret10`,
`ret20`, and `runup_asof`. It preserves the arithmetic and missing-value
rules of the legacy panel block, without its filesystem reads or fallback
loaders. These are shared model inputs, not RUNUP forecast/payoff outputs.

The caller supplies event rows with prior-history aggregates and a mapping
of ticker to date/adjusted-close DataFrames. The caller owns history-row
visibility and coherent price-source selection; the function neither selects
a capture nor stitches retrievals. Existing history arithmetic is reused by
the caller rather than recomputed here. The decision column must be named
explicitly; `"date"` is allowed for the historical event-date convention.
Inputs are not mutated.

| Condition | Outcome |
|---|---|
| Price anchor | Reuse `panel_math._anchor_index`: strictly before the event and on-or-before an explicit decision date; return the actual source date as `runup_asof` |
| Missing or insufficient price history | Preserve legacy missing market-feature values and `NaT` anchor; event-history features remain independently available |
| Anchor before the first price row | Leave market features missing; never use negative indexing |
| Malformed required event columns or incompatible values | Propagate pandas/NumPy input errors; no fabricated defaults |
| Cache, retry, transaction | No cache or side effects; identical inputs give identical outputs |

This module depends only on pandas, NumPy, and the sibling anchor helper.
It performs no I/O, training, inference, or legacy import. It has no production
caller yet; adding this arithmetic alone does not change a nightly or board.

## Inputs

### Pinned daily-state input boundary (implemented)

`scan_daily_state_inputs(repository, snapshot, *, ticker, history_start,
decision_session)` binds the existing `panel_math.daily_state_lookup` to a
bounded read of one ticker from `daily_market` in the supplied `SnapshotRef`.
This contract is implemented in `daily_state_inputs.py`. No production raw-row
assembler calls it yet, and it does not establish a complete forward panel or
qualified board.

The caller supplies explicit naive calendar dates, with `history_start <=
decision_session`; intraday and timezone-aware values refuse. The read uses
the supplied snapshot/table-contract identity, one ticker predicate, a
half-open interval (inclusive start, exclusive next-day end), primary-key
order, and fixed resource bounds. Those query limits keep the active
table-contract caps and the fixed caller limits; this caller may lower its
result limit using a positive recorded membership bound, while a zero bound
keeps the current active positive limit in this pre-E slice. The separate
`computed_moves` and `"SPY"` `daily_market` reads do not lower either query
limit from the selected membership bound; each keeps the minimum of its
active table-contract cap and fixed caller limit. No head lookup, provider
pull, training or legacy path occurs.
Rows outside that identity/date scope, duplicate dates, and invalid source
dates refuse before arithmetic. Repository integrity, scan-validation and
limit failures propagate. Missing tables refuse; no eligible IV-surface row
is an explicit empty result with no source session, rather than a fabricated
feature row.

The result carries immutable raw market values, the actual selected EOD
`source_session`, and snapshot/dataset identities. The existing lookup owns
the inclusive decision-session selection, `src_iv` eligibility, percent/log
units, absent-key null behavior, and positional 1/5/10-row differences.
Insufficient supplied history leaves the corresponding lag keys absent.
No arithmetic or missing-value policy is changed by this adapter.

`daily_market.date` is an EOD observation session. Its contract has no receipt
or publication timestamp or finality marker, so snapshot membership and a
session cutoff cannot prove intraday knowledge or original receipt causality.
`source_session` is neither an event date nor a whole-panel `panel_anchor`.
The raw-row boundary must cover every contributing history/regime/runup
observation before assigning that latter bound. Model forecasts, retained
scores, quote/expiry selection, complete panel assembly and nightly wiring
are outside this boundary; its consumer is the native raw-row producer,
before `NightlyEventInputs` assembly.

### Panel-row staging boundary (implemented)

`scan_panel_row(repository, snapshot, key, *, decision_session,
history_start)` is implemented in `panel_row_inputs.py`. No production
raw-row producer calls it yet (`engine/v2/ops/ARCHITECTURE.md` "Cutover
PR-6"); it is the raw-row producer's one call for one
`native_board_universe.BoardRequest` key's `panel_row`/`panel_anchor`
pair. `panel_row.date` is the scored event's ISO calendar date, required by
the scoring source-bundle consumer; it is never the decision or source-anchor
date. Before any repository read, the boundary validates `key.event_date`
as a non-missing, timezone-naive date or timestamp and normalizes it to
midnight. Intraday event timestamps represent the same calendar event day.
The normalized day supplies the computed-move bound, regime and runup event
anchors, and output date, so an event-day close remains excluded even for
an event-day decision. Numeric, invalid, missing, and timezone-aware event
dates refuse with `CONTRACT_MISMATCH`; decision and history-start inputs
continue to require explicit naive midnight days.
`panel_anchor` is the latest (freshest) of its contributing reads'
own source or outcome-availability dates, never a caller-asserted value — its consumer
(`../scoring/ARCHITECTURE.md`'s `nightly_source_bundle.py`) trusts it as
an observation-freshness upper bound, which only the latest, not the
earliest, contributing date can be: the earliest would let an
intervening freshness cutoff pass even though a later-dated input is
actually fresher than that cutoff. The bound includes the daily-state
`source_session`, `regime_asof`, `runup_asof`, and the latest
`computed_moves.available_as_of_date` among the eligible, non-skipped moves
actually used by history aggregates and history-derived runup fields.
Computed history contributes this bound even when price-history features
cannot resolve a `runup_asof`. Empty history contributes no date; skipped,
unavailable, and null-availability rows do not advance the anchor. Historical
event dates and provenance timestamps do not substitute for outcome availability.
It never assigns `tier4_row` or
`quote_rows` — those stay the raw-row producer's own job
(`engine/v2/ops/ARCHITECTURE.md` "Cutover PR-6"). `key.strategy` never
changes which reads it makes or which keys the result carries — every
call builds the full superset, `STR-RUNUP`'s fields included — so the one
`panel_row`/`panel_anchor` pair built per `(ticker, event_date, session)`
triple (`engine/v2/ops/ARCHITECTURE.md` "Cutover PR-6") is valid and
reused unchanged across every strategy sharing that triple; a strategy
that does not name a superset-only key in its `feature_names` simply
never selects it (`../scoring/ARCHITECTURE.md` "Inputs"). Its
pinned-snapshot dependencies, every one always made: `scan_daily_state_inputs`
(`key.ticker`); `computed_moves` (`key.ticker`, restricted to rows where
`event_date < min(key.event_date, decision_session)` and
`available_as_of_date <= decision_session`, feeding `panel_math`;
availability is the day after the actual outcome-source close, so a delayed
close cannot enter history at an earlier decision. A null availability date
is unavailable to every decision. Rows with `skipped=true`
carry no `realized_move_pct` and are excluded from that feed, never treated
as a zero move; the bounds are on data dates only, never on
`computed_at`, the row's own calculation timestamp: a backfilled or
corrected row remains eligible when its outcome was available by the decision,
even if it was written later);
a new bounded
`daily_market` read for the fixed ticker `"SPY"` (feeding `regime`, not
reused from `scan_daily_state_inputs` — a different, derived shape);
`price_history_query.get_price_series`, as of `decision_session` — its
selected source date sets `runup_asof` and is one input to the
`panel_anchor` composite bound (the latest of every contributing read's
own source date, never this read alone); when no `STR-RUNUP` history
resolves, `runup_asof` stays unset and `panel_anchor` is the latest of
the remaining reads'. Query construction and the
`PriceSeriesRow`-to-DataFrame conversion are implementation detail, not
contract — see the PR body.

The `computed_moves` and `"SPY"` `daily_market` reads do not lower either
query limit from the selected membership bound; each keeps the minimum of its
active table-contract cap and fixed caller limit. Normal repository scan
validation and failures propagate.

| Condition (R1-R6) | Outcome |
|---|---|
| `key.event_date` is numeric, invalid, missing, or timezone-aware | `CONTRACT_MISMATCH`, refused before any repository read |
| snapshot has no `daily_market`/`computed_moves` table | `CONTRACT_MISMATCH`, propagated from the underlying read unchanged |
| snapshot has no `price_history` table | `CONTRACT_MISMATCH`, propagated unchanged |
| `computed_moves` has no row for `key.ticker` | `panel_math.history_features`'s own empty-input behavior: every key is still present (`n_prior=0`, the mean/EMA keys `None`), never absent; `regime`'s fields are unaffected (independent read) |
| a non-null `computed_moves.available_as_of_date` is not a naive calendar day | `CONTRACT_MISMATCH`, refused before history arithmetic |
| an eligible non-skipped `computed_moves` row has a null `realized_move_pct` (a repository-integrity violation — the contract allows that only when `skipped=true`) | `CONTRACT_MISMATCH`, refused before `panel_math.history_features` runs, never silently coerced |
| `daily_market` has no row for `"SPY"` | `regime`'s own no-history behavior: its fields stay `NaN` (its own Inputs table); `panel_math`'s keys are unaffected (independent read) |
| `price_history` has no row for this ticker | `CONTRACT_MISMATCH`, propagated from `get_price_series` unchanged — this read has no empty-source fallback |
| `get_price_series`'s `session_date > observation_ceiling` | `QUERY_NOT_BOUNDED`, propagated unchanged — never a silent future read |
| retry with the same pinned snapshot/key/`decision_session`/`history_start` | identical result; no cache beyond the pinned reads themselves, no write, nothing to roll back |

`regime.add_regime_features(events, market, *, as_of_column="date")` is
pure regime arithmetic over explicit DataFrames. `market` supplies normalized,
chronologically ordered, timezone-naive `date` values and float-convertible
`close` values; `events` supplies normalized `date` and an optional decision
column. Source parsing, ordering, provenance and date validation belong to callers.
The anchor is strictly before the event AND on-or-before the decision when
provided. A fresh event frame preserves index/order and adds the nine legacy
`spy_*` return/drawdown/volatility fields plus the actual source `regime_asof`.
Returns/drawdown and annualized simple-return volatility retain percent units;
volatility uses sample standard deviation (`ddof=1`), and relative volatility
is a unitless ratio minus one. No market read, implicit clock or cache exists.

| Regime input condition | Outcome |
|---|---|
| No eligible source row, including an empty market | NaN features and NaT anchor |
| Insufficient history for one window | That feature stays NaN; eligible source date remains the anchor |
| Zero 252-day volatility | `spy_vol20_rel252` remains NaN |
| Empty events with required columns | Empty result with feature/anchor columns |
| Missing columns or invalid scalar conversion | Existing pandas/NumPy/Python error propagates |
| Retry with unchanged inputs | Safe recomputation produces unchanged outputs |
| Calculation fails | No transaction or partial write; inputs remain unchanged |

- `panel_math.history_features`/`_causal_ema`: a ticker's prior realized
  moves and their absolute values, as plain float sequences — no I/O, no
  source dependency.
- `panel_math.advance_history`: one caller-selected realized panel row with
  its count, move, absolute move, means and span EMAs; implied-move fields
  are optional. The caller owns ticker/event selection and causal cutoffs.
- `panel_math._anchor_index`: three numpy datetime arrays (a market
  series's own dates, event dates, and an optional as-of ceiling).
- `panel_math.add_implied_history`: a DataFrame that already carries
  `ticker`, `date`, and `or_implied` columns — it does not fetch or join
  anything itself.
- `panel_math.daily_state_lookup`: one ticker's own daily rows (each a
  plain mapping carrying `date`, `src_iv`, and the `DAILY_STATE_FIELDS`
  source keys) plus a single decision date. The caller is responsible for
  scoping rows to one ticker; this function does not filter by ticker.
- `recipes.py`/`context.py`: `FeatureRecipe`/`FeatureRequest` values built
  by the caller from event refs and a snapshot ref.

## Outputs

- `advance_history`: incremented count and stepped move/implied means and
  span EMAs, preserving legacy NaN behavior. Missing implied observations
  retain a known implied mean; unavailable means/EMAs stay unavailable.
- `history_features`: a fixed-key mapping (`n_prior`, `mean_prior_move`,
  `mean_prior_abs_move`, and `ema{2,4,8,12}_prior_{move,abs_move}`), every
  key always present, with `None` for a window that has not yet reached its
  required length — exactly legacy's own missing-value convention for this
  function, since this is a byte-identical copy of it, not a new function.
- `add_implied_history`: the input DataFrame with one added column,
  `mean_prior_or_implied`, `NaN` where legacy would also leave it `NaN`
  (same convention as `history_features`, for the same byte-identical-copy
  reason).
- `daily_state_lookup`: a mapping using `DAILY_STATE_FIELDS`' output keys
  (`im`, `iv10`, `iv30`, ..., `mcap_log`) and lagged-difference keys
  (`{key}_d1`, `{key}_d5`, `{key}_d10` for `im`/`iv10`/`iv30`/`exern_iv30`).
  Unlike the byte-identical functions above, a value legacy would represent
  as `NaN` in a DataFrame column is an **absent key** here, never a
  fabricated `NaN`/`0.0` — this is a new function, not a copy, and it
  follows this package's answer-free/no-fabrication convention instead of
  legacy's DataFrame-NaN one.
- `FeatureContextPlanner.request`: a `FeatureRequest` alone — event refs,
  per-event decision contexts, the snapshot ref, and the resolved recipe
  refs. No `FeatureFrame` and no content hash yet; those are `frame()`'s
  outputs, not `request()`'s.
- `FeatureContextPlanner.frame`: the `FeatureFrame` — immutable, with
  content-hashed `frame_ref`/`schema_ref`/`row_keys_ref`/`values_hash`/
  `null_mask_hash`, built only after every row has passed the causal
  timestamp check described above.

## Dependencies and callers

- `panel_math` depends only on `numpy` and `pandas` (both third-party). It
  imports nothing from `engine.v2.data`, `engine.v2.contracts`,
  `engine.v2.foundation`, or any legacy `engine.*` module, and no legacy
  module imports it — two independent copies of the same math, never a
  shared import, so the two paths can coexist before cutover without either
  one's edits silently reaching the other.
- `recipes.py`/`context.py` depend on `engine.v2.contracts` (the recipe/
  request/frame dataclasses) and `engine.v2.foundation` (`content_hash`).
- Consumer: `engine.v2.scoring` imports `default_feature_registry` to
  resolve feature scopes and recipe identities before scoring
  (`engine/v2/scoring/application.py`).
- `regime` uses `panel_math._anchor_index` plus NumPy/pandas; it has no
  filesystem/network access or legacy imports. `panel_row_inputs.scan_panel_row`
  calls `regime.add_regime_features` (its one production-adjacent caller so
  far; production forward-panel assembly itself is still absent); every
  other caller is a test. Neither input frame is mutated.
- `panel_math` supplies anchoring to `regime`; both have focused parity tests.
  Neither has a production forward-panel caller.

## External systems and libraries

`numpy` and `pandas`, both already ordinary dependencies elsewhere in
`engine.v2` (e.g. `engine/v2/models/training`, `engine/v2/research`) — this
package taking `pandas` for one function (`add_implied_history`) is not a
new pattern. No network, filesystem, or database access anywhere in
`panel_math`.

## Failure semantics

`advance_history` adds no validation or fallback: missing required keys and
invalid scalar arithmetic propagate Python/pandas errors. It does not mutate
the input, read dates, infer observation stamps, or enforce a cutoff.

`context.py`/`recipes.py` are not pure math and do raise:

- `FeatureRegistry.get(recipe_id)` raises `KeyError(recipe_id)` for an
  unregistered id. `FeatureContextPlanner.request` calls `get` for every
  `recipe_ref` it is given, so an unknown recipe id fails `request()`
  itself, before any frame is built.
- `FeatureContextPlanner.frame(...)` raises `FeatureContextError` (a
  `ValueError` subclass) for: a row whose `event_id` is not one of the
  request's own events; a missing, blank, or non-ISO timestamp on either a
  row's `observed_at` or its event's `decision_at`; a timestamp with no
  timezone (`tzinfo`/`utcoffset` absent); and a row whose `observed_at` is
  after its event's `decision_at` cutoff. None of these are caught or
  retried internally — every raise propagates straight to the caller, which
  is expected to have validated its inputs before calling `frame()`.

`panel_math` is pure math with no I/O, so most of the 4c template is "does
not apply" by construction rather than a policy choice:

- **Missing input:** an empty history, or `daily_state_lookup` given no
  eligible rows for its ticker, is not an error — the affected keys are
  simply absent (`daily_state_lookup`) or `None` (`history_features`,
  matching legacy) from the return value.
- **Cache:** none. Every call recomputes from its arguments.
- **Retry:** always safe — every function is a pure function of its
  arguments with no side effects.
- **Transaction / partial write:** not applicable; nothing here writes
  anything.
- **Idempotency:** guaranteed structurally — same arguments, same return
  value, always.

## Invariants

- **Native vs. legacy provenance:** `panel_math` and legacy's
  `data/features/panel.py`/`features.py` are two independent, byte-identical
  copies of the shared math, not one importing the other. Legacy stays
  frozen until cutover; nothing in this change edits
  `engine/data/features/panel.py` or `engine/features.py`.
- **No parity-only modes:** the parity check this math enables later calls
  real functions on both sides on the same real inputs — nothing in this
  package is a special code path that only runs under test.
- **Typed refusal / no fabricated values:** `daily_state_lookup`'s
  absent-key convention is this invariant applied to a feature mapping —
  "value present" and "value known-missing" are different, checkable
  states, and a consumer that forgets to check for a key gets a typed
  `KeyError`/`.get()` miss, never a fabricated number.
- **`_anchor_index`'s out-of-range result is never clamped.** It returns
  `-1` for an event before the first series date; every current legacy
  caller (`engine/data/features/panel.py`'s `add_regime_features`,
  `add_runup_features`, `add_orats_features`) guards that before indexing,
  and any future caller of this port's copy must carry the same guard
  (`tests/test_v2_features_panel_math.py::test_anchor_index_negative_one_is_guarded_in_legacy_callers`
  is a regression tripwire on legacy's own guards, not a test of this
  port).

## Diagram

```
                     ┌──────────────────────────┐
                     │   engine.v2.scoring       │
                     │   (application.py)        │
                     └────────────┬─────────────┘
                                  │ default_feature_registry()
                                  ▼
                     ┌──────────────────────────┐
                     │ engine.v2.features         │
                     │  recipes.py / context.py   │
                     └────────────┬─────────────┘
                                  │ (no edge yet)
                                  ▼
                     ┌──────────────────────────┐
                     │ engine.v2.features          │
                     │  panel_math.py (this change)│
                     │  history_features            │
                     │  _causal_ema / _anchor_index │
                     │  add_implied_history          │
                     │  daily_state_lookup            │
                     └──────────────────────────┘
                        ▲ byte-identical copy, no import edge
                        │
                     ┌──────────────────────────┐
                     │ engine.data.features.panel │
                     │ engine.features (legacy)    │
                     │  frozen until cutover        │
                     └──────────────────────────┘
```

`regime` calls `panel_math._anchor_index` within this package.
