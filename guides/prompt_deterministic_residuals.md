# Deterministic residual pools for the PnL simulation

**Status:** proposed, not started. Needs an experiment before promotion.
**Written:** 2026-09-11.

---

## 1. The defect

`exp_pnl_sim`, `win_sim`, `ci_low` and `ci_high` are not functions of the
request. They are functions of the request *and of which tickers the scoring
process happened to load*.

The chain, in `engine/score.py`:

- `_crush_table()` (line ~2095) filters events to `context.daily`'s tickers, to
  stay inside the nightly's memory bound.
- `_residual_pool()` (line ~2122) inner-joins Tier-4 forecasts x panel x that
  crush frame, so the error pool inherits the same filter.
- `pnl_sim.ResidualPool.draw()` buckets the pool into deciles of
  `pred_abs_move` **computed from whatever rows survived**, then samples with
  `rng.integers(0, rows.size, n)`.

A deterministic seed does not rescue this. `rows.size` is ~8,500 against the
full universe and ~400 against a calendar cohort, so the same seed indexes a
different row.

### Measured, 2026-09-11

Global pool after the crush merge: **85,277 rows / 2,797 tickers, 2013-01-03 to
2026-09-10**. Cohort = tickers with events in the next 30 days, scored against
the full pool's own decile edges:

```
decile |   full n  mean err | cohort n  mean err |  shift
  1    |    9160   0.1194 |     301   0.6246 | +0.5053
  4    |    9161   0.2672 |     494   0.1022 | -0.1650
  8    |    9161   0.4133 |     361   0.8677 | +0.4544
  9    |    9161   0.1617 |     389  -0.1956 | -0.3573
ALL    |   91606   0.2153 |    4026   0.2844 | +0.0691
```

Shifts of +/-0.5pp of move against base errors of 0.06-0.41pp. This is a
different distribution, not Monte Carlo noise.

Two things this is **not**, both checked:

- **Not the `MIN_POOL=250` fallback.** Cohort deciles rebalance to ~400 rows
  each; the unbucketed fallback does not fire.
- **Not evidence of overstated profit.** The aggregate shift is +0.069 —
  cohort realized moves overshoot forecasts slightly *more* than the universe
  does, which for short-vol is conservative. Per-decile the sign flips both
  ways, so the direction for any individual row is unknowable until this is
  fixed. The claim here is that the number is unanchored, not that it is
  optimistic.

### Why it matters beyond reproducibility

`exp_pnl_sim` feeds the gate bar through `trailing_cutoff`, and (since EXP-180)
the funding order. A quantity that moves with the loaded ticker set is a
quantity the ledger cannot be held to.

---

## 2. What is already monthly — and already correct

This is the important context. **Tier 4 has solved this problem once already.**

| mechanism | where | grain |
|---|---|---|
| Tier-4 folds | `CADENCE = "monthly"`, `FIRST_FOLD = 2013-01-01`, `fold_start_of()` | monthly, per producer |
| Tier-4 residual pools | `_pool_stats`, `interval_for`, `registry.bucket_residuals` | monthly, per fold |
| per-row provenance | `GROUP_SUFFIXES` -> `_fold_start`, `_model_id`, `_resid_n` | monthly |
| the gate bar | `pnl_sim.trailing_cutoff`, trailing 6mo, `min_window=100` | monthly |

`interval_for` already conditions residuals on the decile the prediction falls
in, already floors thin buckets back to the flat pool, already tracks `_resid_n`
per row, and already does it per monthly fold under strict walk-forward. It is
the same decile machinery `ResidualPool` reimplements — both descend from
EXP-115.

**So this is not a request to invent an artifact. It is a request to stop
maintaining a second, worse copy of one.**

The one thing Tier 4's pools do not carry is the *pairing*. Tier 4 stores
marginal residuals per producer; the simulation needs `(err_move, err_crush)`
from the **same event**, because that pairing is what carries their dependence
without anyone estimating a copula. That is the gap to close, and it is the
only gap.

---

## 3. Design

Two artifacts at two grains. Keeping them distinct is the part to get right.

### 3.1 Per-event errors become Tier-4 columns

Grain is `ticker x event_date` — already `tier4.KEY_COLUMNS`.

- `err_move  = abs_move        - pred_abs_move`      (panel join)
- `err_crush = crush_pct_iv30  - pred_iv_crush_30`   (crush-frame join)

Both predictions are already Tier-4 columns. Both realizations are joins Tier 4
can do at build time. `err_move` joins the `pred_abs_move` column group,
`err_crush` the `pred_iv_crush_30` group.

Rows written before a print have no realization yet; they are filled when the
event resolves. **Do not build new machinery for this** — `_seed_residuals`
(`tier4.py:776`) already carries realized outcomes onto an existing prefix, and
that is the path to extend.

Consequence worth having on its own: the 9M-row `daily_market` read that
`_crush_table` performs per `Scorer` moves to Tier-4 build time, once a night.
Scoring measured 68 min with live forecasts; some of that comes back.

### 3.2 The monthly pool becomes a Tier-4 sibling table

Grain is `fold_start x decile` — *not* Tier-4's key, so it is its own table, not
more columns. Key it on `fold_start` so a row's pool is by construction the pool
of the fold that produced its prediction. That is the train/serve parity
property this program has been chasing all session, obtained for free.

Store the **paired rows**, not summary statistics. A moments-only version
discards exactly the dependence the pool exists to carry.

| variant | contents | size | stable across Tier-4 rebuilds |
|---|---|---|---|
| edges only | `fold_start, decile, edge_lo, edge_hi, n` | ~50KB | no |
| **full snapshot** | + the paired `(err_move, err_crush)` rows | **~110MB** | **yes** |
| capped snapshot | 2,000 pairs per decile per fold | ~40MB | yes, coarser tails |

Recommend the full snapshot. 165 months x ~43k average prefix at float32.
Edges-only is cheap but not stable: refitting a fold changes `pred_abs_move`,
which changes `err_move`, which silently changes every historical PnL estimate
ever recorded. Stability across rebuilds is the entire point.

### 3.3 `ResidualPool` reads the table instead of building one

`Scorer._residual_pool` and `Scorer._crush_table` both disappear from the
scoring path. `ResidualPool` is constructed from the stored fold pool for the
request's `fold_start`. It stops taking a context, so it stops having one.

Record the resolved `decile` and its `n` next to `exp_pnl_sim` on the board, so
a row can say which pool produced its number. Cheap, and it makes the estimate
auditable rather than merely reproducible.

---

## 4. The recency question

This is a real question with a real trade, and the experiment must answer it
rather than the plan asserting it.

Measured rows per window, as of 2026-09-10:

```
window     rows   per-decile   tail(1%)
   6mo     4859       485          48
  12mo     9278       927          92
  24mo    18249      1824         182
  36mo    26701      2670         267
  60mo    41397      4139         413
  all     85277      8527         852
```

Per-decile counts clear `MIN_POOL=250` at every window, so the choice is not
about whether bucketing survives. It is about **tails against regime**:

- `pnl_sim`'s own docstring states the case for depth: *"earnings residuals are
  skewed and fat-tailed, and the tails are the part that matters."* A 6-month
  window leaves **48 observations** in the 1% tail of a decile. That is not a
  tail estimate, it is an anecdote.
- Against that, the full window reaches back to 2013 and spans the 2020 vol
  regime. A pool that includes March 2020 may be describing a world the current
  book does not live in.

Note also that annual volume nearly doubled over the span (3,798 events in 2013
to 9,041 in 2025), so an unweighted full-history pool is *already* recency-
weighted, roughly 2:1 in favour of the recent half. Worth stating explicitly
before adding a second, deliberate weighting on top.

Precedent cuts toward a bounded window: `trailing_cutoff` uses trailing 6 months
with a `min_window=100` floor and returns `None` — UNDETERMINED — rather than
defaulting when thin. That pattern (window, floor, explicit undetermined) is the
one to copy whatever window wins.

---

## 5. The experiment

**Question: do deterministic residuals still carry information?** Removing the
cohort scoping is correct on principle, but the scoping may have been acting as
an accidental conditioning that happened to help. That must be measured, not
assumed.

### Arms

Cross **scope** x **recency**:

- scope: `cohort` (status quo, the control) / `universe`
- recency: `6mo` / `24mo` / `60mo` / `all`

8 arms. The cohort x all cell is the incumbent; it is the only arm whose
numbers are already on the ledger.

### Metrics

Report all four **per arm**, per the standing requirement that every arm gets
its own report:

1. **Interval coverage** — the share of realized PnL falling inside
   `[ci_low, ci_high]` against the nominal 80%. *This is the primary metric.*
   It tests the residual distribution directly, independent of the gate and of
   funding. An arm that is reproducible but miscalibrated fails here and
   nowhere else.
2. **Discrimination** — Spearman rank correlation of `exp_pnl_sim` against
   realized PnL. This is "carries information" in the narrow sense.
3. **Calibration of `win_sim`** — predicted win rate against realized, by decile
   of predicted.
4. **Book outcome** — total PnL and Sharpe under each arm's funding order,
   since EXP-180 made funding depend on this estimate.

Also report, per arm: pool size, per-decile counts, resolved-decile
distribution, and how often the thin-bucket fallback fires.

### Pre-registration

`spec.yaml` plus a planned LEDGER.csv row before the first real run. Per
`no-ledger-rows-from-smoke-runs`, give the runner `--no-ledger` for every subset
test. Per `prereg-guard-noops-without-planned-row`, confirm the planned row
actually exists — the guard does not enforce this.

### Decision rule

Promote `universe` at the winning window if coverage is no worse than the
incumbent's and discrimination is not materially degraded. Per
`promotion-evidence-standards`, judge on consistency across folds rather than a
magnitude cutoff, and label any judgement call as a judgement call.

If `cohort` wins on coverage, that is a genuine finding and not a reason to keep
the current code: it would mean cohort-conditioning carries real signal, which
should then be implemented *deliberately and deterministically* — as a stored
conditioning key — rather than as a side effect of a memory bound.

---

## 6. Rollout and risks

1. Tier-4 columns (3.1) with `_seed_residuals` extended; test the fill-on-resolve path.
2. Fold-pool table (3.2); assert it reproduces the current pool at `scope=cohort` so the migration is provably behaviour-preserving before anything changes.
3. Experiment (section 5).
4. Only then, switch `Scorer` to the table (3.3).

**Risks**

- **Every board number moves.** `exp_pnl_sim`, `win_sim` and both CI bounds
  change on every row. Measured before/after, not shipped silently.
- **The gate bar shifts with them.** `trailing_cutoff` ranks `exp_pnl_sim`
  against its own trailing history; changing the scale changes the bar. The
  history file (`pnl_sim.load_history`, seeded from EXP-129) is on the old
  scale and must be rebuilt, or the first nights after the switch gate against
  a bar from a different distribution.
- **Monthly freeze lags regime shifts** by up to a month. Minor against a
  6-month-plus window, and it errs causal — never forward-looking.
- **Tier-4 rebuild cost rises**, since the crush frame joins at build time.
  Offset against per-`Scorer` savings; measure both.

---

## 7. Why this keeps happening

Third instance of one pattern: a memory or performance bound quietly became
part of a statistical object's definition.

- the analog block (`analog-context-width-defect`) — 98.5% fallback rate
- the residual pool — this document
- `_crush_table`'s own scoping comment, which states the memory motive plainly
  and does not consider that it is also changing an answer

The generalizable rule: **a bound on what is loaded must never be a bound on
what is computed from it.** When a statistical object is scoped for memory,
that scope is an input to the result and belongs in the recorded artifact, or
it does not belong at all.
