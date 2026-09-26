# Design: Tier-4 port A — panel_math

Status: draft
Spec: this PR's description, `<details><summary>Spec</summary>` — Tier-4
feature-computation port build spec, §6 Part A and §9 (supervisor decisions).

## Context

The board-row scorer needs to stop reading legacy Tier-4 tables and instead
compute its feature inputs natively. That native computation depends on a
handful of pure numeric functions that legacy already defines correctly and
that must not drift from legacy's own math while both paths coexist before
cutover. This change ports the first, smallest slice of that dependency: the
event-history recursion, its EMA helper, the anchor-index rule the later
market-state functions reuse, the prior-implied-move running mean, and a new
pure per-decision-date extraction core for daily market state.

If this change does not happen, no later part of the Tier-4 port has a home
for this math: the orchestrator (Part C) and the market-state cores (Part A2)
both need these functions already ported and byte-identical to legacy, and
the parity tool (Part E) has nothing to call on the native side to compare
against legacy's own output.

This session covers design only — no code. Implementation is a later,
separate push to this PR once the design is approved, done through the
project's standard DeepSeek-authored-code workflow.

## Decisions

- **Decision:** Port `history_features`, `_causal_ema`, `_anchor_index`, and
  `add_implied_history` as byte-identical copies rather than reimplementing
  or vectorizing them for v2 idioms.
  - **Options considered:** (a) copy the math verbatim; (b) rewrite/simplify
    while porting.
  - **Choice:** (a).
  - **Why:** These are recursions (EMA seeding, expanding mean) where a
    reassociation changes the output at the float level. The project's own
    convention for this exact case (`history_features`'s docstring) is "one
    definition, not two, because two implementations of a recursion drift
    the moment either is touched." A byte-identical copy also lets the parity
    tool (Part E) assert exact equality instead of adopting a tolerance,
    which is the standing default in this codebase (start at exact match,
    only add tolerance if a genuine reassociation is found and never accept
    it silently).

- **Decision:** Legacy keeps its own copy of this math; v2 does not import
  it from legacy, and legacy does not import the new v2 copy.
  - **Options considered:** (a) two independent copies, same math, no import
    edge either direction; (b) v2 imports legacy's existing functions;
    (c) legacy is changed to import the new v2 copy.
  - **Choice:** (a).
  - **Why:** (b) would make a v2-layer package depend on legacy, which is
    backwards — v2 must not depend on legacy. (c) changes legacy's behavior
    (an import edge it does not have today) before cutover, which needs its
    own sign-off and is out of scope here. This mirrors a settled decision:
    legacy is frozen until cutover, and pure functions move to v2 as
    independent copies, not shared imports.

- **Decision:** Factor a new single-decision-date pure function
  (`daily_state_lookup`) out of legacy's `daily_state_frame`, rather than
  porting that function's DataFrame-vectorized shape verbatim.
  - **Options considered:** (a) port the whole batched, DataFrame-in/
    DataFrame-out function; (b) factor out the per-row extraction rule as a
    function over plain sequences/mappings, with no DataFrame dependency.
  - **Choice:** (b).
  - **Why:** The port's consumer (the future per-row orchestrator) resolves
    one `(ticker, event_date, as_of)` triple at a time; a batched function
    would need to be called with a one-row frame every time, carrying a
    pandas dependency into a file that otherwise needs none. This is the
    same shape decision the spec already makes for the sibling market-state
    functions ("the pure per-date extraction core... once its I/O half is
    separated," not a verbatim copy).

- **Decision:** A value legacy would represent as `NaN` in a DataFrame column
  is represented as an **absent key** in this port's output mapping, never as
  a fabricated `NaN`/`0.0`.
  - **Options considered:** (a) keep `NaN` as the missing marker, matching
    legacy's DataFrame convention; (b) omit the key entirely.
  - **Choice:** (b).
  - **Why:** The port's answer-free/no-fabrication rule (used everywhere
    else a feature vector can be partly unresolved) treats "value present"
    and "value known-missing" as different, checkable states. `NaN` collapses
    that distinction and risks a silent downstream mask failure (a consumer
    that forgets to check `isfinite` gets a fabricated number instead of a
    typed refusal).

- **Decision:** `_anchor_index` is ported in this part (Part A), not deferred
  to the market-state part that is its main consumer.
  - **Options considered:** (a) port it as part of this file; (b) port it
    alongside the market-state functions that use it.
  - **Choice:** (a).
  - **Why:** `history_features` and the market-state functions must share one
    definition of "which row is this feature read at," for the same
    drift reason as the EMA recursion. Landing it once, here, and having the
    market-state part import it from this module (not redefine it) keeps
    that single-definition property from the start rather than fixing it up
    later.

- **Decision:** `add_implied_history`'s pure running-mean function is ported
  in this part alongside `history_features`, not treated as a separate,
  later slice.
  - **Options considered:** (a) include it here; (b) leave it for a later
    part.
  - **Choice:** (a).
  - **Why:** Its body was read in full and confirmed to do no file/table
    access — it operates only on columns already resident on its input — so
    it is exactly as pure as `history_features`, and the spec places it
    beside `history_features` for that reason.

## Layers and modules touched

| Package | Layer | New/changed | Imports (new edges) |
|---|---|---|---|
| `engine.v2.features` (new file `panel_math.py`) | 2.0 | new | `numpy` and `pandas` (both third-party). No edge to `engine.v2.data`/`engine.v2.contracts`/`engine.v2.foundation`, and no edge to legacy `engine.*`. |

`engine/v2/features/` is the package's own declared home for this port (its
`layer_map` entry lists `data/features/panel.py` and `data/features/tier4.py`
among the modules it replaces). This file adds no import edge to another
`engine.v2` package or to legacy in either direction — `layer_map.py` places
no restriction on third-party dependencies, and `pandas` is already a normal
dependency elsewhere in `engine.v2` (e.g. `engine/v2/models/training/`,
`engine/v2/research/`), so taking it here does not introduce a new pattern.

`pandas` is required, not optional, because `add_implied_history`'s ported
body (`engine/data/features/panel.py:852-889`) is itself pandas code —
`sort_values`, `groupby`, `.shift(1)`, `.expanding().mean()` — and the byte-
identical-copy decision above means that body is copied as-is, DataFrame in,
DataFrame out. `history_features`, `_causal_ema`, `_anchor_index`, and the
new `daily_state_lookup` take plain arrays/sequences/mappings and return
plain mappings, exactly as before; `add_implied_history` is the one exception
in this file, and its signature stays `DataFrame -> DataFrame` to match
legacy's, not the plain-mapping shape the other four functions use. An
earlier draft of this table said "numpy only" and separately described every
function's I/O as "plain arrays/sequences/mappings," which is wrong for
`add_implied_history` — this revision corrects both places to agree.

`docs/ARCHITECTURE.md` does not exist in this repository yet, so this design
cannot be checked against it; the layer/ownership claims above are against
`checks/layer_map.py`'s existing `engine.v2.features` package entry instead.

## Production call path

**Nothing in production reaches this module yet.** The chain this math will
eventually sit on is:

`engine/v2/ops/native_feature_job.py` (new job kind, not yet built) →
`engine/v2/features/board_features.py`'s per-row orchestrator (not yet
built) → this part's `panel_math.py` (`history_features`, `daily_state_lookup`)
and the sibling market-state module that reuses `panel_math._anchor_index`
(not yet built).

Both intermediate layers are separate, later parts of the same build (the
orchestrator needs this part plus a market-state module; the job needs the
orchestrator plus the sources it reads from). Until those land and are wired
into the nightly `GRAPH`, this file has zero callers and cannot affect any
board row — it ships as inert math, with unit tests of its own (this part's
test plan below) but no consumer.

The correctness check **planned** to exercise it beyond those unit tests is
the parity proof tool, Part E of this same build, not yet built and not part
of this PR. Once it exists, it is intended to call this module's functions
directly on real inputs and compare the result to what legacy's existing
functions produce on the same inputs, so the port's correctness can be
checked independently of whether the orchestrator/job parts have landed.
Nothing about that tool's design, existence, or results is established by
this design doc — it is future work this part is a precondition for, not a
check this part has already passed.

## Changed interfaces and their callers

This part only adds a new file; it changes no existing interface. A grep for
every function this part ports (`history_features`, `_causal_ema`,
`_anchor_index`, `add_implied_history`, and legacy's `daily_state_frame`,
whose per-date rule this part's new `daily_state_lookup` factors out) shows
their only current callers are inside legacy itself (the panel batch build,
the live scorer's market-state assembly, and legacy's own model-training
modules) — none of that is touched by this part, and none of it will call
the new v2 copy; legacy keeps calling its own copies, unchanged.

The new functions in this part (`history_features`, `_causal_ema`,
`_anchor_index`, `add_implied_history`, `daily_state_lookup`) have no callers
yet within this PR. Their first callers will be later parts of this same
build (the market-state module and the per-row orchestrator), not part of
this change.

## Failure semantics

Using the 4c R1–R6 template. This file is pure math with no I/O, so most of
these are "does not apply" by construction rather than a policy choice:

- **Missing input:** An empty history (`prior_moves`/`prior_abs` with no
  events, or a `daily_state_lookup` given no rows for a ticker) is not an
  error — the affected keys are simply absent from the returned mapping
  (see the missing-value decision above), matching what legacy already does
  for the same case (a short history yields `None`/`NaN` for the
  window-length-gated fields).
- **Cache:** None. Every call recomputes from its arguments; there is
  nothing to invalidate.
- **Retry:** Always safe. Every function here is a pure function of its
  arguments with no side effects, so calling it twice with the same inputs
  is calling it once, observed twice.
- **Transaction:** Not applicable — this file performs no writes.
- **Partial write:** Not applicable, for the same reason.
- **Idempotency:** Guaranteed structurally: same arguments, same return
  value, always. There is no state for a re-run to collide with.

## Invariants touched

- **Native vs. legacy provenance:** this part keeps the two paths as two
  independent, byte-identical copies rather than one importing the other,
  per the "pure functions MOVE" decision — the v2 copy is a move of the
  math, not a rewrite, and legacy's copy is untouched.
- **No parity-only modes:** the parity check this part enables (Part E,
  later) calls this module's real functions and legacy's real functions on
  the same real inputs; nothing here is a special code path that only runs
  under test.
- **Legacy stays frozen until cutover:** no legacy file is edited by this
  part. `engine/data/features/panel.py` and `engine/features.py` are read
  from, never written to.
- **Typed refusal / no fabricated values:** the missing-value decision above
  (absent key, not `NaN`) is exactly this invariant applied to a feature
  mapping instead of a whole-row refusal.

## Test plan

| Acceptance criterion | Test | How this test could fail |
|---|---|---|
| `history_features`/`_causal_ema` match legacy exactly | `tests/test_v2_features_panel_math.py` calls both the new and legacy functions on the same fixed `(prior_moves, prior_abs)` fixtures (empty, shorter than every EMA span, exactly at a span boundary, longer than all spans) and asserts equal output, including which keys are present | A silent reassociation (e.g. summing in a different order) would produce a value close to but not equal to legacy's, which exact equality catches and a tolerance-based test would not |
| `_anchor_index` matches legacy exactly, including the out-of-range case | Same test file, fixed `(series_dates, event_dates, as_of_dates)` fixtures covering: `as_of` absent, `as_of` before the event's own anchor, `as_of` after it, a tie, and an event **before the first series date** (asserting the return is `-1`, matching `np.searchsorted(..., side="left") - 1` on an index-0 hit) | An off-by-one in the `searchsorted` side/offset would only show up on a boundary-date fixture, not a fixture where all dates are far apart; a fix that clamps the result to `0` instead of leaving it at `-1` would only show up on this specific before-the-first-date fixture |
| Every current legacy caller of `_anchor_index` guards the `-1`/out-of-range case before indexing with it | Not a new test on the port — a documentation check on the existing legacy call sites, cited here because a future v2 caller of the ported `_anchor_index` must copy the same guard. `engine/data/features/panel.py` has three call sites (`add_regime_features` ~line 408, `add_runup_features` ~line 566, `add_orats_features` ~line 696), and all three check the index before use: `if j < 0 or j >= len(closes): continue`, `if idx < 0: continue`, and `if j < 0: no_prior += 1` / `else: ...targets[...][i] = columns[...][j]` respectively — none indexes an array with a raw, unchecked `-1` | A future v2 caller (the market-state part, A2) that indexes on `_anchor_index`'s result without first checking `< 0` would silently read the last row of the array (Python's negative-index wraparound) instead of refusing — this is the failure mode the guard exists to prevent, and A2's design must carry the same guard forward |
| `add_implied_history`'s running mean matches legacy exactly | Same test file, a fixed multi-ticker frame with gaps, asserting the per-ticker expanding mean is shifted by exactly one row and never leaks the current event's own value | A missing `.shift(1)` equivalent would let the current row's own value leak into its own mean — must be tested with a fixture where that would change the result |
| `daily_state_lookup` matches legacy's `daily_state_frame` for a single decision date | Same test file, a fixed daily-market fixture (including rows with no IV surface, i.e. no `src_iv`) compared against calling legacy's `daily_state_frame` with a one-row request at the same date; normalize legacy `NaN` outputs to absent keys before comparison | Filtering the wrong subset of rows (e.g. not excluding non-surface rows) would only surface on a fixture that mixes surface and non-surface rows for the same ticker |
| Missing values are absent keys, not `NaN` | Same test file, a decision date before any coverage and a lag field with insufficient history, asserting the corresponding keys are not present in the returned mapping at all (not present-and-`NaN`) | A refactor that keeps computing the value and just leaves it as `NaN` instead of dropping the key would pass a "value is not a real number" check but fail an "is this key even in the dict" check — the test must check for absence, not just non-finiteness |

`oc_check`: `python3 tools/oc_check.py --files tests/test_v2_features_panel_math.py --verify`

## Out of scope

- The four "mixed" market-state functions (`add_regime_features`,
  `add_runup_features`, `add_orats_features`, `add_pre_print_vol`) and
  `days_before_print` — a separate part of this build (Part A2).
- The outward-reaching source functions that will feed this math from real
  snapshot data — a separate part (Part B).
- The per-row orchestrator that calls this math and assembles a full feature
  vector — a separate part (Part C).
- The job wiring that calls the orchestrator on a real board — a separate
  part (Part D), itself blocked on another spec's board-request builder that
  has not landed yet.
- The parity proof tool — a separate part (Part E).
- Any change to legacy's own `engine/data/features/panel.py` or
  `engine/features.py` — none is made or needed.
- Implementation of this design (code, tests) — this session is design only;
  code is a later push to this same PR, after design review.

## Changed during implementation

Not yet applicable — no implementation has happened against this doc yet.
