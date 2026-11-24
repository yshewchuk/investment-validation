# Short-vol structure width tuning for the board (design, EXP-189)

**Status: DESIGN ONLY (user decisions 2026-11-24).** The code slices (section 11) are NOT started until this design merges. This PR adds this file only: no code, no `spec.yaml`, no test, no edit to EXP-188's design or spec. Experiment id **EXP-189** (slice W1 re-checks that it is free).

**Purpose.** Choose **production's** short-vol structure widths for the board, using production's forecast model only (**M0**, `size_v1_4`). There is no roster, model contrast or per-model factor. If the board's width rules change (a separate user decision), EXP-188 (`guides/size_decision_pnl_design.md`, the forecast-model comparison, kept production-exact) is rerun on the new widths. EXP-189 has its own spec, pre-registration and PLANNED row, never changes EXP-188's spec or hash, and uses EXP-188's machinery (decisions, pool, fast path, marks, cash-secured accounts) as a library.

## 1. Question and evidence so far

Does a different strike width, set as a multiple `k` of production's width rule (`k = 1` is production), earn more for the seven put-side short-vol structures (TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5)? In EXP-188's 2025 run one `c` per model picked the grid edge `c = 0.5` (width 2x) for 11 of 12 models. The synthetic oracle (T11) shows P&L peaking at about 1.5-2x the actual move for BFLY-P, BFLY-P5, CND-PS, RAMP7 and CTR5; TWIN-P and TWIN-P5 are fine near 1x. EXP-137 found chosen widths at 1.3-1.7x the forecast (`engine/forecast_sizing.py` comment).

**Disclosure.** 2025 results of the EXP-188 runs (including `retuned` and T11) were seen while designing this. Every 2025 number is labelled **"2025 NOT blind for widths"**. The success rule and thresholds below are fixed before any EXP-189 run but are not blind to those results; all tuning reads 2023-24 only.

## 2. Constraints

- Production's rules stay the base. `k` multiplies the width `forecast_params` returns: `width = k * (forecast/100)/divisor`. `WIDTH_MIN`/`WIDTH_MAX` apply **after** `k`; a width outside them is declined (non-trade, P&L 0, counted). No edit to `engine/forecast_sizing.py`, `STRUCTURES`, any SIZE search module or file in `source_closure(root, ["experiments/size_panel.py"])`; `k` lives in new `size_width_*` modules.
- Library reuse, no EXP-188 edit: new modules import `size_pnl_*` unchanged. A missing hook is a separate preparatory change under EXP-188's owner, never part of an EXP-189 slice.
- Forecast and crush forecast are M0's, held fixed. No call-side short leg; marks only from `engine/fills.py`; holdouts only through the shared admission. Non-recorded runs use `--no-ledger`.
- **Standard-tool check.** numpy only (`default_rng`, `polyfit`); nothing hand-rolled beyond a 3-parameter fit.

## 3. The tunings (all for M0, each a factor on production's rule)

All tune on 2023-24 (scoring from 2023H2, EXP-188's burn-in), freeze before 2025, and run 2025 by month (R6). Same grid, objective and tie rule for all (section 5). Baseline: **production** (`k = 1`), recomputed in this run from the same pinned inputs, so a cited number is never mixed with a recomputed one.

| Id | Name | Factors | Tuned on |
| --- | --- | --- | --- |
| A | `global` | one `k` for all seven structures | M0's real 2023-24 P&L |
| B | `per_structure_real` | seven `k_s` | M0's real 2023-24 P&L (one path) |
| C | `per_structure_fake` | seven `k_s` | a fake forecast matching M0's error profile, over `N` seeded draws (section 4) |
| D | `two_factor` | `k_T` for TWIN-P and TWIN-P5, `k_C` for the five centre-peaked structures | M0's real 2023-24 P&L (optional; dropped first if cost binds) |
| anchor | `oracle_sweep` | descriptive only | forecast = actual move |

C versus B tests whether a smoother simulated objective picks better factors than B's single real path.

**Anchor (descriptive).** On 2023-24 only, per structure: width = `k` x the **actual** move (forecast replaced by the label `y`, production divisor, factor `k`), same gate, marks and exit. T4 prints the score by `k`. It never feeds a freeze or the success rule; it shows where a perfect forecast would peak and whether the grid is wide enough.

## 4. Tuning C: the fake forecast

**Error model fitted on M0's 2023-24 inner forecasts vs labels.** For inner paired events, M0 forecast `f`, label `y = abs_move` (non-positive dropped, counted), on the log scale (that of EXP-188's `regression_direction` generator):

`log f = L + b * (log y - mean(log y)) + s * z`, `z ~ N(0,1)`.

`L`, `b` are the OLS intercept (at the mean) and slope, `s` the residual sd; `b < 1` is the under-prediction of large moves. Judgement call (mine): log scale keeps a forecast positive; Gaussian noise is pre-stated; the report prints residual skew and excess kurtosis and flags "poor fit" above 1, changing nothing. T1 lists `(L, b, s, n)`. **Hard check:** the fake's RMSE and MAE against `y` on inner E (percent points, mean over draws) must be within 5% of M0's own (my judgement call); otherwise C halts `fake_not_calibrated` and A, B, D and the anchor still run.

**Why the fake forecast may use the label (causality).** The fake is a simulation of an error process, `f = g(y, z)`, so it depends on its own event's `y` by construction (as EXP-188's synthetic arm). It is used only to choose `k_s` from 2023-24 events whose labels printed before the 2025 stage; it never produces a 2025 decision, and no real decision uses its own event's label. Within a draw the pool and cutoff use strictly earlier events only (EXP-188 section 4). Stated limit: the optimum is conditional on the fitted error model, and the in-sample cells (section 7) are in-sample for it.

**Sweep.** For each structure and `k` in the grid, `N` seeded draws: draw `d`, event `e` uses `numpy.random.default_rng(int.from_bytes(sha256(f"{event_id}|exp189-v1|{d}".encode()).digest()[:8], "big"))`, so `z` is common across structures and `k`. Each draw is a fake model with its own `ResidualPool` (`err_move = y - f`), the same trailing-cutoff gate, `FillModel` marks, exit and burn-in as EXP-188, via the fast path (one simulation per (draw, event) at each ladder strike; each (structure, `k`) a dot product; cutoff recomputed per `k`). The seven `k_s` go to an immutable `tuning_C` receipt before 2025; real M0 then runs with them, sizing its own forecast (no level adjustment).

## 5. Grid, objective, tie rule

- **Grid (recommended; open decision 2):** `k in {0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}` (`k = 1/c` in EXP-188's terms); it extends past 2.0 so a 1.5-2x peak cannot sit at the edge. **Edge rule:** an argmax at 3.0 or 0.5 is not frozen: `k_at_edge`; that factor runs at its best interior value, flagged; widening needs a design change.
- **Objective (recommended; open decision 1):** per (unit, `k`), the **stand-alone cash-secured $100,000 account** return (EXP-188 Amendment 2 account (b); `margin.simulate` unchanged) under the primary marks and under `marks_mid`; the score is the **lower of the two**, averaged over draws for C. Units: for A, one account over the seven books; for D, one per factor group; for B and C, each structure's own account. Flat-dollar P&L is not used: it was dominated by penny-debit trades (EXP-188 Amendment 2: 6,666 lots on a $0.015 debit).
- **Tie rule:** a `k` within one standard error of the best (across draws for C; across quarters for A, B, D): the `k` nearest 1, then the smaller, so production stays unless the data clearly prefer otherwise.
- **Draws `N` (open decision 3):** 20, standard error reported.
- **Minimum trades.** A (unit, `k`) with fewer than 300 trades (summed over draws for C) is `not_scored`; a unit with no scored `k` is `k_defaulted` (section 8).

## 6. Run order, outputs

`inner` (fit, sweeps, tunings A-D, anchor, tolerance band of section 12) writes the `tuning_A`..`tuning_D` and `tuning_band` receipts; `outer` (2025) refuses without all of them. Each arm also runs on the inner window (2023H2, 2024) for the in-sample reading. One heavy job at a time, `bounded_run`, quarter chunks, stderr progress at a bounded cadence (stage start/end, done/total, rate, ETA, RSS). Cost (not measured; the slice smoke measures): C is `N` fast-path simulations of one model; A, B, D are dot products on the real path.

`REPORT.md` (2025 tables titled "2025 NOT blind for widths"): T1 error model and calibration; T2 chosen factors beside `k = 1`; T3 sweep curves per structure for B and C (score, standard error, trades, declined share, **flatness**: best minus the score at `k = 1` and its neighbours in standard errors, and the span of `k` within one standard error of the best); T4 anchor; T5 production versus A-D: per trade (return, count), `marks_mid`, seven-structure cash-secured account (both mark sets; final equity, return, drawdown, wanted/funded) for 2025 and the inner halves; T6 per-structure stand-alone accounts; T7 success table (section 7); T8-T9 section 12. No bootstrap interval is added; a flat curve is the main caution.

## 7. Success rule and the proposed board rule (written before any run)

Candidate X in {A, B, C, D} **passes** iff, in the cash-secured seven-structure account, its final return exceeds production's (`k = 1`) in **each of six cells**: {2025, 2023H2, 2024} x {primary marks, `marks_mid`} (each inner half starts a fresh $100,000 account). Guard (my judgement call): in 2025 X's stand-alone account (T6) must not lose to production in more than 2 of 7 structures under either mark set. 2023H2 and 2024 are in-sample for A, B, D (and for C's error model); 2025 is out-of-sample for the factors but **not blind for widths**.

**Recommendation among passers (fixed now).** Fewest free factors first: A (1), D (2), then B or C (7). If both B and C pass, recommend C (its objective is less noisy; fixed now, not chosen from 2025); T7 reports both in 2025 and in sample. If none passes, **recommend no change** (`k = 1`). The recommendation is a **proposed board width rule**: divisors `divisor / k_s` (`divisor / k` for A, per group for D), with T3's flatness (a flat curve argues for leaving the board alone) and the declined share per structure at each proposed `k`.

**Feeding back.** Evidence, not a change. The board change is a **separate user decision**, which also needs the DYN-SV chooser review of EXP-188 Decision 3 (the chooser reads `width_over_forecast` features). After a board change, EXP-188 is rerun on the new widths in its own amendment by its own owner. No pass leaves the board and EXP-188 unchanged.

## 8. Failure semantics (R1-R6)

| Case | Result |
| --- | --- |
| R1 M0 inputs missing or failed (no forecast series, no inner decisions, under 300 inner events) | refuse `EXPERIMENT_IDENTITY_CONFLICT`; no partial report; nothing imputed |
| R1 event with a missing label, forecast or fact | EXP-188's exclusion reasons (`move_label_missing`, `production_forecast_missing`, `facts_missing`), counted; the event leaves E for every arm |
| Whole-run refusals | `EXPERIMENT_IDENTITY_CONFLICT`, `details.reason` in `tuning_receipt_missing`, `tuning_receipt_changed`, `planned_row_missing`, `spec_hash_mismatch` |
| Fake outside 5% of M0's RMSE/MAE | `fake_not_calibrated`; C omitted, other arms run |
| (structure, `k`) with no feasible width or under 300 trades | `not_scored`, excluded from the argmax, declined share printed |
| All `k` of a unit excluded | `k_defaulted` = 1.0, flagged |
| Argmax at an edge | `k_at_edge` (section 5) |
| Candidate declined by width at `k` in 2025 | non-trade, P&L 0, stays in E (EXP-188 section 4) |
| R2 cost | one process, `bounded_run`, quarter chunks; over-limit refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report |
| R3 | seeds fixed by the sha256 recipe of section 4 |
| R4 | receipts atomic, immutable, bound to the inputs receipt; a changed input refuses at replay |
| R5 | fake models carry ids `synthetic-w-<draw>` and are never reported as models; M0's `model_id` unchanged |
| R6 | tuning reads 2023-24 only; 2025 refuses without the five receipts; 2025 decisions are written month by month before that month's labels are read (EXP-188 R6) |

## 9. Registration, spec hash

EXP-189 has its own `spec.yaml` (M0 reference by EXP-188's sealed plan id, arms, grid, `N`, thresholds, success rule), its own PLANNED ledger row with the spec hash, and a guard that cannot no-op (`planned_row_missing`). It reads the sealed v14 and production inputs through EXP-188's `Sources` adapter, unchanged. Nothing is registered in this PR.

## 10. Open decisions for the user (recommendation first)

1. **Objective.** Recommended: stand-alone cash-secured account return, lower of primary and `marks_mid`. Alternatives: flat-dollar P&L (outlier-prone), `marks_mid` alone, a rank combination. About 2 x 7 x 10 x `N` account simulations for C.
2. **Grid.** Recommended: ten points up to 3.0. Alternatives: add 4.0 (most candidates decline at `WIDTH_MAX`), or drop 0.5 and 0.67.
3. **`N` draws.** Recommended 20 (fewer than 10 not recommended; raise if curves are too noisy to rank).
4. **Arm D.** Recommended: include (cheap, real path only); drop if fewer candidates are preferred.
5. **Tolerance fraction `phi` and fallback step (section 12).** Recommended `phi = 0.90`, `rho` step 0.05. Alternatives: 0.95 (narrower band, fewer recoveries) or 0.80 (wider, more recoveries, more risk of a worse geometry).

## 11. Split list (code slices, NOT started until this design merges)

| Slice | Scope | Code / test lines / doc words | After |
| --- | --- | --- | --- |
| W1 | Scaffold EXP-189 (`spec.yaml`, PLANNED row, guard), `k` sizing module, id check | ~110 / ~120 / 60 | this PR |
| W2 | M0 error-model fit, calibration check, T1 | ~100 / ~110 / 40 | W1 |
| W3 | Seeded draws, sweep, objective, `tuning_C` receipt, anchor | ~170 / ~170 / 60 | W2 |
| W4 | Tunings A, B, D on the real path, 2025 and inner runs | ~150 / ~150 / 50 | W3 |
| W5 | Tolerance band and `tuning_band` receipt, realised-width and decline tables (T8), `fallback` geometry search and arm, T9, adoption rule | ~160 / ~170 / 80 | W4 |
| W6 | Report T2-T7, success table, proposed rule, driver and smoke, runbook | ~150 / ~140 / 100 | W5 |

Each slice stays near 150 code lines and adds only new `size_width_*` modules; EXP-188's `size_pnl_*` code is imported unchanged.

## 12. Width flexibility: tolerance band, nearest-feasible fallback, snap deviation (user 2026-11-24)

**Question.** `size_pnl_shortvol.resolve` snaps each leg to the nearest listed strike and declines the candidate (`ladder_too_coarse`) when two legs collapse. Supervisor measurement from the 2025 run (not re-verified; T8 reproduces it): for M0 at `k = 1` that decline hit about 27% of short-vol candidates at market cap >= $10B and about 40% at $1-10B, the largest single non-trade reason. How wide a range of widths is nearly as good as the best, and can a nearby feasible geometry recover those candidates? M0 only, tuned on 2023-24, 2025 NOT blind for widths.

**12.1 (a) Tolerance band.** Per structure, from the sweep curve (section 5 objective): the contiguous range of `k` around the argmax where the score stays at or above a fraction `phi` of the peak (recommended 0.90, open decision 5), ends located by linear interpolation between grid points, reported as a width ratio to production (`k`) and to the actual move (from the anchor). Computed on B's and C's curves (C primary, smoother) and the anchor for reference. A non-positive peak gives `no_positive_peak` (no band, no fallback); a band touching the grid edge is `band_open` (flagged). Frozen from 2023-24 in the `tuning_band` receipt.

**12.2 (c) Deviation of today's trades (descriptive, M0, `k = 1`).** Realised width: the width the listed strikes of a taken trade carry, measured as the structure's factory defines `width_moneyness` (`engine/structures.py`), against the target `forecast_params` returned. T8 gives quantiles (10, 25, 50, 75, 90) of realised/target per structure and cap tier (>= $10B, $1-10B, below) for 2023-24 and 2025; decline shares by reason and tier (a mismatch with 27% and 40% is a finding); return per taken trade by deviation bucket (below 0.9, 0.9 to 1.1, above 1.1). Cost of deviation is read from the curve (the C curve at the realised versus the target ratio) and from the buckets, which are **descriptive only**: snap deviation correlates with price level and ladder spacing, so no causal claim.

**12.3 (b) Nearest-feasible fallback.** Arm `fallback` (M0, at `k = 1` and at each section 7 candidate's tuned `k`) changes one thing. A candidate that would be declined `ladder_too_coarse` or `structure_unresolved` (not `width_declined`, `negative_floor` or `entry_not_debit`, kept as today) is retried at widths `target * rho`, `rho` stepping by 0.05 over the structure's tolerance band **expressed relative to the arm's target** (`k` band divided by the arm's `k`), nearest first (smallest `|log rho|`, smaller `rho` on ties). Each trial width must also lie inside `[WIDTH_MIN, WIDTH_MAX]`; `rho` never leaves the band. The first `rho` whose snapped geometry has distinct legs, passes the zero-floor re-check and gives a debit entry is used; if none exists the candidate is declined as today. Gate (`exp_pnl_sim` for the new geometry is a dot product on the existing ladder simulation), trailing cutoff (recovered scores enter the history like any determined candidate), marks, exit and accounts are unchanged. Choosing `rho` never looks at P&L or labels. T9 reports per structure and tier: declined, recovered, recovered share, `rho` distribution, and recovered trades' return and P&L under both mark sets **against trades already taken**, and the account with versus without the fallback.

**12.4 Outputs and the adoption rule (pre-stated).** (i) A **recommended tolerance band per structure**: the frozen band with `phi`, flatness and `band_open` flags. (ii) **Fallback adoption, yes or no:** **yes** iff the `fallback` arm's seven-structure account exceeds the same arm without it in each of the six cells of section 7, **and** the recovered trades' mean return per trade is not negative under either mark set in 2025 and in 2023-24 pooled, **and** at least 100 trades are recovered in 2025 (else `insufficient_recovery`, answer no). Otherwise no. The 100-trade floor is my judgement call. The board change is the user's separate decision, with the DYN-SV chooser review of section 7; EXP-188 is rerun if the widths or the fallback change.

**12.5 Failure semantics.**

| Case | Result |
| --- | --- |
| Peak not positive | `no_positive_peak`; no band or fallback for the structure |
| No feasible `rho` in the band | declined as today, counted `fallback_infeasible` |
| Under 100 recovered trades in 2025 | `insufficient_recovery`; answer no |
| Realised width not computable (a leg missing) | left out of T8 quantiles, counted `deviation_unavailable`; never imputed |
| R2, R3 | dot products on the existing ladder table, a few percent of the sweep cost (not measured); no randomness added |
| R4, R6 | `tuning_band` receipt atomic and immutable; `fallback` refuses without it (`tuning_receipt_missing`); band frozen from 2023-24; no label or P&L enters the choice of `rho` |
