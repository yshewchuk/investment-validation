# Phase 2 Guide — Experiment Framework & Evaluation Suite

**Objective:** the EXP-000..050 discipline as reusable scaffolding: any
candidate (model, gate, structure, parameter change) runs one standardized
evaluation and either beats the champion or doesn't — evidence auto-generated,
every attempt logged.

---

## 1. Build order

1. `engine/evaluate.py` core: backtest → walk-forward → MC → metrics on an
   existing trades dataset (no new pricing code needed to start).
2. Structure-pricing path: candidate trades generated via
   `engine.structures` + chain cache (reuses Phase 0's single pricing path).
3. Stress battery.
4. Scaffolder + spec format + ledger.
5. `promote.py` + promotion report.
6. Harness regression vs EXP-050; then run the backlog.

## 2. Experiment layout & spec

New experiments live in `experiments/` at repo root, numbered **EXP-101+**
(the 0–50 range belongs to `earnings_predictions/`; never reuse it).

```
experiments/
  LEDGER.csv                 # append-only: every evaluated spec, ever
  EXP-101_calp_exact_spec/
    spec.yaml  run.py  results/  REPORT.md  figures/
```

`spec.yaml`:
```yaml
id: EXP-101
title: CAL-P exact-spec backtest
hypothesis: >
  Short ~1DTE put + long back put opened together pre-print, closed together
  post-print, is positive at mid fills.
primary_spec: {front_dte: 1, back_dte: 20, entry: T-1, strike: ATM-same}
grid: {back_dte: [7,14,20,30,45], entry: [T-1,T-2], diagonal: [0,1]}
data_snapshot: <hash from data/features/SNAPSHOT>
walk_forward: {unit: year, min_train_years: 2}
promotion_target: null | <champion id>
preregistered_at: <UTC timestamp — written by the scaffolder at creation>
```

`new_experiment.py` scaffolds the folder, stamps `preregistered_at`, and
appends a PLANNED row to LEDGER.csv. **Enforcement:** `evaluate` refuses to
run OOS years if `preregistered_at` is missing or later than the first run
timestamp (it writes `results/run_log.jsonl` per invocation). The primary
spec's OOS result is the headline; grid results are reported but labeled
secondary — this is the anti-post-hoc-selection mechanism, keep it strict.

## 3. Evaluation suite (`engine/evaluate.py`)

One entry point: `evaluate(spec, trades_or_generator) -> results dict + report`.
Stages, in order:

1. **Backtest** — price the candidate's trades from cached chains through
   `engine.structures` at alphas {0, 0.25, 0.5, 0.75, 1.0}; report
   worst/mid/best side by side plus **breakeven alpha** (linear interpolation
   of mean return across the sweep) — a headline stat on every report.
2. **Walk-forward** — expanding window by calendar year: fit/tune on years
   < Y, trade year Y, concatenate OOS years. Any tunable (gate threshold,
   model refit) refits inside the loop; nothing sees year Y before trading
   it. Headline numbers come from this stage only.
3. **Monte Carlo** — block bootstrap on the walk-forward OOS trade sequence:
   block = 20 trades (preserves earnings-week clustering), 1,000 paths, at
   sizing fractions {2%, 5%, 10%, 20%}: P(final loss), drawdown p50/p95,
   terminal-equity p5/p50/p95 → the sizing curve.
4. **Stress battery:**
   - Regime replays: date-filter to 2018Q4, 2020-02..04, 2022, and the 10
     worst realized earnings weeks; per-regime P&L table.
   - Tail injection (mandatory for any short leg): double the worst 1% of
     realized |moves|, re-price through the structure payoff; report the new
     worst trade and MC P(loss).
   - Slippage days: shift entry ±1 trading day where the adjacent chain is
     cached; report deltas and the coverage fraction (never fabricate a
     missing chain).
   - Stale-date simulation: mis-date 1% of events by one day; report P&L
     impact.
   - IV-regime split: high-vol vs low-vol years (SPY vol20 median split).
5. **Metrics dict** (canonical keys — identical across all strategies so the
   leaderboard is comparable): `n, mean, median, std, win_rate,
   profit_factor, sharpe_trade, sharpe_equity, sortino, max_dd, tail_ratio,
   by_year{}, breakeven_alpha, capacity{}, mc{}` where
    - `sharpe_trade` = mean(ret)/std(ret) × sqrt(avg trades/year),
    - `sharpe_equity` = mean(daily eq ret)/std × sqrt(252) on the 5%-sized
      walk-forward equity curve,
    - `sortino` uses downside deviation vs 0,
    - `tail_ratio` = |p95 win| / |p95 loss|,
    - `capacity` = spread width at the traded strikes (relative spread, mean
      and p95, wide-market fraction) with a note where the source carries no
      volume — sizing decisions must not be made without it.
    Equity construction: chronological by entry date, fixed-fraction sizing
    off **marked equity** (cash + open positions at cost — net liquidation
    value; a cash-only reading reports deployment as drawdown), overlapping
    positions allowed and counted (report max concurrency); premium debited at
    entry, value credited at exit, series marked at cost between events.
    Per-trade sizing times concurrency is implicit leverage, so every curve
    also reports **peak deployment and worst cash** next to the returns, and a
    spec may set ``max_deployed_fraction`` to cap total deployed notional
    (entries beyond the cap are skipped and counted, never levered).
6. **Report** via the Phase 4 generator — an experiment without REPORT.md
   does not exist.

## 4. Ledger & promotion

- `LEDGER.csv` append-only: id, spec_hash, date, stage (planned/ran),
  headline OOS mean @mid, sharpe_trade, promoted (bool). EVERY evaluated
  spec including grid cells and failures — this is the multiple-testing
  record cited in promotion decisions.
- `promote.py <exp-id> <champion-role>`: checks the plan's rules —
  challenger beats champion on OOS mean AND sharpe_trade; MC P(loss)@5% not
  worse; stress battery has no new red cell; preregistration valid. All
  green → updates `registry.json`, archives the old champion entry, emits a
  promotion report diffing the two (side-by-side metrics + LEDGER context:
  "N specs were tried before this one"). Any red → prints why and exits
  nonzero; no partial promotion.

## 5. Constraints

- Real chains only; `exit_mode=="chain"` only (intrinsic fallbacks are
  look-ahead — known trap); filter expiry ≥ event date (the 430 same-day
  expiry trap).
- Anti-selection guard: any sell-side or premium statistic computed on a
  model-selected subset must also be reported on the unselected universe
  (the S4 lesson).
- Survivorship note auto-included (current-listed universe).
- Runtime: cache intermediate priced-trades per (spec_hash, alpha) under
  `results/` so grid re-runs don't re-price.

## 6. Acceptance tests (`checks/phase2_checks.py`)

1. **Synthetic-known test:** feed trades drawn from a known distribution
   (e.g. mean +2%, std 10%, 6/yr) → metrics dict reproduces analytic
   Sharpe/win-rate within MC tolerance; MC P(loss) matches simulation.
2. **Harness regression — load-bearing:** re-run the EXP-050 configuration
   (GBM top-20% gate, walk-forward, 531 trades 2019–2026) through
   `evaluate`; assert trade count, per-year returns, 2.4×@5% terminal, MC
   P(loss)≈6% within tolerance. Root-cause and document ANY divergence
   before running new experiments — a harness that can't reproduce the
   evidence the plan rests on proves nothing.
3. **Preregistration enforcement:** deleting `preregistered_at` → evaluate
   refuses OOS; back-dating after a run → refuses.
4. **Walk-forward leak poison:** let a gate threshold be fitted on year Y
   data inside year Y → the WF stage must catch it (assert via an injected
   marker feature that only exists in-year).
5. **Ledger append-only:** attempting to rewrite a row fails.
6. **Promotion dry-run:** synthetic challenger better/worse than champion →
   promote/refuse correctly, report generated.

## 7. First runs (from the plan's backlog, in order)

EXP-101 CAL-P exact-spec (put legs, T−1 entry, post-print close, back-DTE
grid — vs the EXP-046b variant on the same events); EXP-102 CAL-P risk
mechanics (assignment/pin/max-loss); EXP-103 STR-RUNUP entry-day
optimization; EXP-104 moneyness edge-decay map (unblocks Phase 1's
extrapolated flag); EXP-105 full-universe mid-fill gate; EXP-106 1–10B slice
with Polygon real-fill spot checks.
