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

Precisely: Tier 4 does not *persist* the pool arrays. It persists `p10`, `p90`,
`sd` and `n` per row, and `_pool_before(fold, model, panel)` (`tier4.py:1228`)
reconstructs the `(prediction, residual)` arrays by reading the STORED forecast
table back and filtering to earlier folds. Its docstring says why that is also
the correct choice and not merely the cheap one. The result is a function of
`(fold, model, panel)` and of nothing else.

### The scorer is already using it — for one of the two estimates

| | forecast band (`forecast_p10/p90/sd`) | PnL sim (`exp_pnl_sim`, `win_sim`, `ci_*`) |
|---|---|---|
| call | `served.interval()` -> `interval_for(pool_pred, pool_res)` | `pnl_sim.expected_pnl(pool=self._residual_pool())` |
| pool from | `_pool_before(fold, model, panel)`, stored table | `_crush_table()` -> `crush_frame()` over `context.daily` |
| context-dependent | **no** | **yes** |
| deterministic | **yes** | **no** |

So the defect is not that the machinery is missing. It is that the simulation
path does not use the machinery sitting next to it.

The reason it does not is legitimate and is the whole gap: `_pool_before`
returns **one producer's marginal residuals**, while `expected_pnl` needs
`err_move` and `err_crush` **paired on the same event**. That is a join across
two producers, and the join is where `_crush_table` — and its memory-motivated
filter — entered.

## 3. Design

**The join does not need `crush_frame()`.** This is what makes the fix small.

The realized crush is `iv_crush.TARGET` = `crush_pct_iv30`, and
`iv_crush.prepare(panel)` already joins it from Tier 2 — the build path,
context-free by construction, and the same values `_pool_before` already
differences to make that producer's residuals. Nothing here needs a 9M-row
`daily_market` read at score time.

So the joint pool is assembled by calling `_pool_before` twice — once for
`pred_abs_move`, once for `pred_iv_crush_30` — and inner-joining the two on
`(ticker, event_date)`.

### 3.0 What the pool must be — the contract

Before changing how it is built, be clear what it is. `ResidualPool` keeps four
ROW-ALIGNED arrays (`pnl_sim.py:114-126`):

| array | why it exists |
|---|---|
| `_dates` | the causal cut; `before(cutoff)` is a `searchsorted`, so the table must be date-sorted |
| `_pred` | the BUCKETING KEY — which decile of `pred_abs_move` this historical event sat in |
| `_move` | `err_move` |
| `_crush` | `err_crush` |

`draw()` returns `self._move[chosen], self._crush[chosen]` — **the same
`chosen`**. Downstream, `expected_pnl` does:

```python
move  = np.maximum(pred_abs_move + err_move, 0.0)
crush = pred_iv_crush + err_crush
spot_exit = spot * (1.0 + sign * move / 100.0)
vol_exit  = (pre_iv30 / 100.0) * (1.0 + crush / 100.0)
```

So draw *i* is **one coherent world**: this print moved this much AND vol
crushed this much, both repricing the same exit legs. A large move with a
shallow crush is not as likely as a large move with a deep crush, and the pool
carries that dependence empirically — by taking both errors off the SAME
historical event — rather than anyone fitting a copula.

Three rules follow, and breaking any of them breaks the estimator silently:

1. **Row identity is the mechanism.** The join in 3.1 is on
   `(ticker, event_date)` because that is the only way the size model's error
   and the crush model's error land on the same row. A join that loses row
   identity — or any reshaping that sorts the two error columns independently —
   keeps both marginals intact and destroys the thing being estimated.
2. **Summary statistics cannot substitute.** Per-decile moments give the
   marginals and throw away the dependence. This is the real reason not to
   store a precomputed pool artifact, stronger than the duplication argument in
   3.3.
3. **There is no ticker column, and there must not be one.** Ticker was never a
   matching key — `_pred` is. That is exactly why scoping the pool by ticker was
   always outside the design, and why the defect in section 1 was invisible:
   nothing in `ResidualPool` asks about tickers, so nothing objected when the
   set of them changed underneath it.

Unrelated to the pool but worth not mistaking for a defect: `sign` is drawn from
`rng` rather than taken from history. That is exact, not an approximation — the
symmetric structures depend on `|move|` only.

### 3.1 Rewrite `_residual_pool` as a join of two served pools

- `_pool_before(fold, size_model, panel)`     -> `(pred_abs_move, err_move)`
- `_pool_before(fold, crush_model, panel)`    -> `(pred_iv_crush_30, err_crush)`
- inner join on the event key; feed to `pnl_sim.ResidualPool`

`_pool_before` currently returns bare arrays, so it needs to also return the
event keys for the join. That is the only signature change.

### 3.2 Delete `_crush_table` from the scoring path

`Scorer._crush_table` exists for two callers: `_residual_pool` and
`_pre_print_iv`. Once 3.1 lands, check whether `_pre_print_iv` can read
`pre_iv30` from the panel — `iv_crush.prepare`'s docstring says the four
pre-print vol terms became Tier-3 panel columns on 2026-09-05 precisely so they
would be servable. If so, `_crush_table` goes away entirely and the scorer stops
touching `daily_market` for this at all.

### 3.3 What is NOT needed

Earlier drafts of this plan proposed new `err_move`/`err_crush` Tier-4 columns
and a monthly fold-pool sibling table. **Both are unnecessary.** The stored
forecast table plus `_pool_before` already provide a deterministic,
fold-keyed, walk-forward-safe pool; the only thing missing was the join, and
the join is three lines. Adding an artifact would mean maintaining a third copy
of a thing that already exists twice.

Do record the resolved decile and its `n` next to `exp_pnl_sim` on the board.
That is cheap and makes the estimate auditable rather than merely reproducible.

### 3.4 What to measure before writing the code

- **Join loss.** The two producers may not cover the same events; the old
  `crush_frame` merge cost 7% (91,616 -> 85,277). Measure the fold-joined loss
  and confirm per-decile counts still clear `MIN_POOL=250`.
- **Cost of `_pool_before` at score time.** It calls `load_forecasts()` and
  `training_frames(panel, model)`, and the latter runs `prepare`. Cache per
  `Scorer` the way `_crush_frame` is cached today, and measure against the
  68-minute scoring baseline.

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

1. Extend `_pool_before` to return event keys alongside its arrays.
2. Rewrite `_residual_pool` as the two-producer join (3.1). Assert it reproduces
   the current pool when restricted to the context tickers, so the migration is
   provably behaviour-preserving before anything changes.
3. Experiment (section 5) over the recency arms.
4. Retire `_crush_table` from the scoring path (3.2), once `_pre_print_iv` is
   confirmed servable from the panel.

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
- **`_pool_before` cost at score time** is unmeasured. It reads the stored
  forecast table and runs the model's `prepare`. Cache it per `Scorer` as
  `_crush_frame` is cached today, and measure against the 68-minute baseline.

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
