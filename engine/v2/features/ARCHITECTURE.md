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
- `FeatureContextPlanner.request(...)` — builds a causal `FeatureRequest`
  from event refs, a snapshot ref, and recipe refs, enforcing that every
  timestamp involved is explicit and timezone-aware
  (`engine.v2.contracts.scoring.FeatureRequest`/`FeatureFrame`).
- `panel_math` — five pure functions with no `FeatureRecipe`/`FeatureFrame`
  wrapping of their own (see below). Nothing in this package or any other
  yet constructs a `FeatureRecipe` or calls `FeatureContextPlanner` for
  this math; that binding is future work.

### `panel_math` functions

| Function | Shape | Status |
|---|---|---|
| `history_features(prior_moves, prior_abs) -> dict` | plain sequences in, plain mapping out | byte-identical copy of `engine.data.features.panel.history_features` |
| `_causal_ema(history, span) -> float \| None` | plain list in, scalar out | byte-identical copy of `engine.data.features.panel._causal_ema`; `history_features`'s helper |
| `_anchor_index(series_dates, event_dates, as_of_dates) -> np.ndarray` | numpy arrays in and out | byte-identical copy of `engine.data.features.panel._anchor_index` |
| `add_implied_history(df) -> pd.DataFrame` | DataFrame in, DataFrame out | byte-identical copy of `engine.data.features.panel.add_implied_history` — the one function in this module that is pandas-shaped, because its ported body is itself pandas code (`groupby`/`shift`/`expanding`) |
| `daily_state_lookup(rows, decision_date) -> dict` | plain sequence of mappings + a date in, plain mapping out | new: factors the per-`(ticker, as_of)` extraction rule out of `engine.features.daily_state_frame`, which is batched (DataFrame-in/DataFrame-out) and stays in legacy unchanged |

## Inputs

- `panel_math.history_features`/`_causal_ema`: a ticker's prior realized
  moves and their absolute values, as plain float sequences — no I/O, no
  source dependency.
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
- `FeatureContextPlanner.request`: an immutable `FeatureFrame`/
  `FeatureRequest` pair with content-hashed provenance.

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
- `panel_math` has no consumer yet. The chain it is built for is
  `engine/v2/ops/native_feature_job.py` (a new job kind, not yet built) →
  a per-row feature orchestrator in this package (not yet built, planned
  as `board_features.py`) → `panel_math` plus a sibling market-state
  module that reuses `panel_math._anchor_index` (not yet built). Until
  those land and are wired into the nightly graph, `panel_math` is inert:
  it has unit tests of its own but cannot affect a board row.
- The correctness check planned to exercise `panel_math` beyond its unit
  tests is a parity proof tool (this package's `engine.v2.parity`
  counterpart), not yet built, that will call these functions directly on
  real inputs and compare against legacy's own output on the same inputs.

## External systems and libraries

`numpy` and `pandas`, both already ordinary dependencies elsewhere in
`engine.v2` (e.g. `engine/v2/models/training`, `engine/v2/research`) — this
package taking `pandas` for one function (`add_implied_history`) is not a
new pattern. No network, filesystem, or database access anywhere in
`panel_math`.

## Failure semantics

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

`panel_math` has no caller inside `engine.v2` yet (dashed relationship
above); the future orchestrator/job that will call it are separate,
later parts of the same build.
