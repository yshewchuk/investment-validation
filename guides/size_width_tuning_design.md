# Short-vol structure width tuning for the board (design, EXP-189)

**Status: DESIGN ONLY (user decisions 2026-11-24).** The code slices (section 11) are NOT started until this design merges. This PR adds this file only: no code, no `spec.yaml`, no test, and no edit to EXP-188's design or spec. Experiment id **EXP-189** (slice W1 re-checks that it is free).

**Purpose.** Choose **production's** short-vol structure widths for the board, using production's forecast model only (**M0**, `size_v1_4`). There is no roster, no model contrast and no per-model factor. If the board's width rules change (a separate user decision), EXP-188 (`guides/size_decision_pnl_design.md`, the forecast-model comparison, kept production-exact) is rerun later on the new production widths. EXP-189 has its own spec, pre-registration and PLANNED row, never changes EXP-188's spec or hash, and uses EXP-188's machinery (decisions, pool, fast path, marks, cash-secured accounts) as a library.

## 1. Question and evidence so far

Does a different strike width, set as a multiple `k` of production's width rule (`k = 1` is production exactly), earn more for the seven put-side short-vol structures (TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5)? Measured in EXP-188's 2025 run: one `c` per model picked the grid edge `c = 0.5` (width 2x) for 11 of 12 models. The synthetic oracle (T11) shows P&L peaking at about 1.5-2x the actual move for BFLY-P, BFLY-P5, CND-PS, RAMP7 and CTR5, while TWIN-P and TWIN-P5 are fine near 1x. EXP-137 found chosen widths at 1.3-1.7x the forecast (`engine/forecast_sizing.py` comment). Production under-predicts large moves.

**Disclosure.** 2025 results of the EXP-188 runs (including `retuned` and T11) were seen while designing this. Every 2025 number is labelled **"2025 NOT blind for widths"**. The success rule and every threshold below are fixed before any EXP-189 run, but they are not blind to those earlier 2025 results; all tuning reads 2023-24 only.

## 2. Constraints

- Production's rules stay the base. `k` multiplies the width `forecast_params` returns: `width = k * (forecast/100)/divisor`. `WIDTH_MIN`/`WIDTH_MAX` apply **after** `k`; a width outside them is declined (non-trade, P&L 0, counted, as today). No edit to `engine/forecast_sizing.py`, `STRUCTURES`, any SIZE search module or any file in `source_closure(root, ["experiments/size_panel.py"])`; `k` lives in a new `size_width_*` module that calls existing functions.
- Library reuse, no EXP-188 edit: new modules import `size_pnl_*` unchanged. A missing hook is a separate preparatory change under EXP-188's own owner, never part of an EXP-189 slice.
- Forecast and crush forecast are M0's, held fixed. No call-side short leg; marks only from `engine/fills.py`; holdouts only through the shared admission. Non-recorded runs use `--no-ledger`.
- **Standard-tool check.** numpy only (`default_rng`, `polyfit`); nothing hand-rolled beyond a 3-parameter fit.

## 3. The tunings (all for M0, each a factor on production's rule)

All tune on 2023-24 (scoring from 2023H2, EXP-188's burn-in), freeze before 2025, and run 2025 by month (R6). Same grid, objective and tie rule for all (section 5). Baseline: **production** (`k = 1`), recomputed in this run from the same pinned inputs so a cited number is never mixed with a recomputed one.

| Id | Name | Factors | Tuned on |
| --- | --- | --- | --- |
| A | `global` | one `k` for all seven structures | M0's real 2023-24 P&L |
| B | `per_structure_real` | seven `k_s` | M0's real 2023-24 P&L (one path) |
| C | `per_structure_fake` | seven `k_s` | a fake forecast matching M0's error profile, over `N` seeded draws (section 4) |
| D | `two_factor` | `k_T` for TWIN-P and TWIN-P5, `k_C` for the five centre-peaked structures | M0's real 2023-24 P&L (optional middle ground; dropped first if the cost budget binds) |
| anchor | `oracle_sweep` | descriptive only | forecast = actual move |

C versus B tests whether a smoother simulated objective picks better factors than B's single real path.

**Anchor (descriptive).** On 2023-24 only: per structure, width = `k` x the **actual** move (forecast replaced by the label `y`, production divisor, factor `k`), same gate, marks and exit. T4 prints the score by `k` per structure. It never feeds a freeze or the success rule; it shows where a perfect forecast would put the peak and whether the grid is wide enough.

## 4. Tuning C: the fake forecast

**Error model fitted on M0's 2023-24 inner forecasts vs labels.** For inner paired events `e`, M0 forecast `f`, label `y = abs_move` (non-positive dropped, counted), on the log scale (the scale of EXP-188's `regression_direction` generator):

`log f = L + b * (log y - mean_e(log y)) + s * z`, `z ~ N(0,1)`.

`L`, `b` are the OLS intercept (at the mean) and slope, `s` the residual sd. `b < 1` is the under-prediction of large moves. Judgement call (mine): log scale so a forecast stays positive; Gaussian noise is pre-stated; the report prints residual skew and excess kurtosis and flags "poor fit" above 1, changing nothing in this run. T1 lists `(L, b, s, n)`.

**Hard check.** The fake's RMSE and MAE against `y` on inner E (percent points, mean over draws) must lie within 5% of M0's own inner RMSE and MAE (my judgement call). Otherwise C halts `fake_not_calibrated`; A, B, D and the anchor still run.

**Why the fake forecast may use the label (causality).** The fake forecast is a simulation of an error process, `f = g(y, z)`, so it depends on its own event's `y` by construction: it stands in for a forecast whose joint distribution with `y` matches M0's (the same construction as EXP-188's synthetic arm). It is used only to choose `k_s` from 2023-24 events, whose labels have all printed before the 2025 stage; it never produces a 2025 decision, and no real decision uses a label of its own event. Within a draw the pool and cutoff still use strictly earlier events only (EXP-188 section 4), so the entry rule is causal given the fake forecast. The residual limit, stated rather than hidden: the sweep's optimum is conditional on the fitted error model, and the in-sample cells (section 7) are in-sample for it.

**Sweep.** For each structure and each `k` in the grid, score `N` seeded draws: draw `d`, event `e` uses `numpy.random.default_rng(sha256(f"{event_id}|exp189-v1|{d}"))`, so `z` is common across structures and `k`. Each draw is a fake model with its own per-model `ResidualPool` (strictly earlier events, `err_move = y - f`), the same trailing-cutoff gate, `FillModel` marks, exit and burn-in as EXP-188, via the fast path (one simulation per (draw, event) at each ladder strike; each (structure, `k`) a dot product; the cutoff recomputed per `k`). The seven `k_s` go to an immutable `tuning_C` receipt before 2025 is touched. The real M0 then runs with those `k_s`; no level adjustment is used (the arm sizes M0's own forecast).

## 5. Grid, objective, tie rule

- **Grid (recommended; open decision 2):** `k in {0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}` (`k = 1/c` in EXP-188's terms). It extends past 2.0 so a 1.5-2x peak cannot sit at the edge. **Edge rule:** an argmax at 3.0 or 0.5 is not frozen: `k_at_edge`; that factor runs at its best interior value, is flagged, and widening needs a design change.
- **Objective (recommended; open decision 1):** per (unit, `k`), the **stand-alone cash-secured $100,000 account** return (EXP-188 Amendment 2 account (b); `margin.simulate` unchanged) under the primary marks and under `marks_mid`; the score is the **lower of the two** (a `k` that wins only under optimistic marks is not picked), averaged over draws for C. For A and D the unit is the account over the seven books (one account for A, an account per factor group for D's two groups); for B and C the unit is the structure's own account. Why not flat-dollar P&L: it was dominated by penny-debit trades (EXP-188 Amendment 2: 6,666 lots on a $0.015 debit).
- **Tie rule:** a `k` within one standard error of the best (across draws for C; across quarters for A, B, D): the `k` nearest 1, then the smaller. This keeps production unless the data clearly prefer otherwise.
- **Draws `N` (open decision 3):** 20, with standard error reported.
- A unit with fewer than 300 selected inner trades summed over the grid's chosen path skips tuning (`k = 1`, labelled).

## 6. Run order, outputs

Stages: `inner` (fit, sweeps, tuning A-D, anchor) writes the `tuning_A`..`tuning_D` receipts; `outer` (2025) refuses without them. Each arm also runs on the inner window for the in-sample reading (2023H2, 2024). One heavy job at a time, `bounded_run`, quarter chunks, progress to stderr at a bounded cadence (stage start/end, done/total, rate, ETA, RSS). Cost (not measured; the slice smoke measures): C is `N` fast-path simulations of one model; A, B, D are dot products over the real path.

`REPORT.md`, 2025 tables titled "2025 NOT blind for widths": T1 error-model parameters and calibration check; T2 chosen factors (A, B, C, D) next to `k = 1`; T3 sweep curves per structure for B and C with score, standard error, trades, declined share and **flatness** (best minus the score at `k = 1` and its neighbours, in standard errors; the span of `k` within one standard error of the best); T4 the oracle anchor; T5 production versus A-D: per trade (return per taken trade, trade count), `marks_mid`, cash-secured accounts (the seven-structure account: both mark sets; final equity, return, drawdown, wanted/funded) for 2025 and the inner halves; T6 per-structure stand-alone accounts; T7 the success table (section 7). No bootstrap interval is added; a flat curve is the report's main caution.

## 7. Success rule and the proposed board rule (written before any run)

A candidate X in {A, B, C, D} **passes** iff, in the cash-secured seven-structure account, its final return exceeds production's (`k = 1`) in **each of six cells**: {2025, 2023H2, 2024} x {primary marks, `marks_mid`}; the two inner halves each start a fresh $100,000 account. In addition (a guard against one structure carrying the result; my judgement call), in 2025 X's stand-alone account (T6) must not lose to production in more than 2 of the 7 structures under either mark set. 2023H2 and 2024 are in-sample for A, B and D (and for C's error model); 2025 is out-of-sample for the factors but **not blind for widths**.

**Recommendation among passers (fixed now).** Fewest free factors first: A (1), then D (2), then B or C (7). If both B and C pass, recommend C (its objective is the less noisy; the choice is fixed now, not made from 2025); T7 reports B versus C in 2025 and in sample so a reader sees the gap. If none passes, **recommend no change** (`k = 1`). The recommendation is a **proposed board width rule**: new divisors `divisor / k_s` (or `divisor / k` for A, per group for D), a table in REPORT, with the flatness (T3: a flat curve argues for leaving the board alone) and the declined share per structure at each proposed `k`.

**Feeding back.** The recommendation is evidence, not a change. The board change is a **separate user decision**, which also needs the DYN-SV chooser review of EXP-188 Decision 3 (the chooser reads `width_over_forecast` features). After a board change, EXP-188 is rerun on the new production widths, in its own amendment, by its own owner. A failure to pass leaves the board unchanged and EXP-188 unaffected.

## 8. Failure semantics (R1-R6)

| Case | Result |
| --- | --- |
| R1 M0 inputs missing or failed (no forecast series, no inner decisions, fewer than 300 inner events) | the run refuses `EXPERIMENT_IDENTITY_CONFLICT` (unbound inputs); no partial report; nothing imputed |
| R1 an event with a missing label, forecast or fact | EXP-188's exclusion reasons (`move_label_missing`, `production_forecast_missing`, `facts_missing`), counted; the event leaves E for every arm |
| Whole-run refusals | `EXPERIMENT_IDENTITY_CONFLICT` with `details.reason` in `tuning_receipt_missing`, `tuning_receipt_changed`, `planned_row_missing`, `spec_hash_mismatch`; no partial run |
| Fake outside 5% of M0's RMSE/MAE | `fake_not_calibrated`; C omitted, other arms run |
| (structure, `k`) with no feasible width (every candidate declined) or under 300 trades summed over draws | `not_scored`, excluded from the argmax, declined share printed |
| All `k` of a unit excluded | `k_defaulted` = 1.0, flagged |
| Argmax at an edge | `k_at_edge` (section 5) |
| A candidate declined by width at `k` in 2025 | non-trade, P&L 0, stays in E (EXP-188 section 4) |
| R2 cost | one process, `bounded_run`, quarter chunks; over-limit refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report |
| R3 | seeds are the sha256 of event id, tag `exp189-v1` and draw `d` (section 4); fixed in the spec |
| R4 | receipts atomic, immutable, bound to the inputs receipt; a changed input refuses at replay |
| R5 | fake models carry ids `synthetic-w-<draw>` and are never reported as a model; M0's `model_id` is unchanged |
| R6 | tuning reads 2023-24 only; 2025 refuses without the four tuning receipts; each month's decisions are frozen before its labels are read |

## 9. Registration, spec hash

EXP-189 has its own `spec.yaml` (M0 reference by EXP-188's sealed plan id, arms, grid, `N`, thresholds, success rule), its own PLANNED ledger row with the spec hash, and a guard that cannot no-op (`planned_row_missing`). It reads the sealed v14 and production inputs through EXP-188's `Sources` adapter, unchanged. EXP-188's spec and hash are untouched by this design. Nothing is registered in this PR.

## 10. Open decisions for the user (recommendation first)

1. **Objective.** Recommended: stand-alone cash-secured account return, lower of primary and `marks_mid`. Alternatives: flat-dollar P&L (outlier-prone), `marks_mid` alone, a rank combination. The recommendation costs about 2 x 7 x 10 x `N` account simulations for C.
2. **Grid.** Recommended: ten points up to 3.0. Alternatives: add 4.0 (most candidates decline at `WIDTH_MAX`), or drop 0.5 and 0.67.
3. **`N` draws.** Recommended 20 (fewer than 10 not recommended; raise if curves are too noisy to rank).
4. **Arm D.** Recommended: include (cheap, real-path only). Drop it if the user prefers fewer candidates.

## 11. Split list (code slices, NOT started until this design merges)

| Slice | Scope | Code / test lines / doc words | After |
| --- | --- | --- | --- |
| W1 | Scaffold EXP-189 (`spec.yaml`, PLANNED row, guard), `k` sizing module, id check | ~110 / ~120 / 60 | this PR |
| W2 | M0 error-model fit, calibration check, T1 | ~100 / ~110 / 40 | W1 |
| W3 | Seeded draws, sweep, objective, `tuning_C` receipt, oracle anchor | ~170 / ~170 / 60 | W2 |
| W4 | Tunings A, B, D on the real path, 2025 and inner runs | ~150 / ~150 / 50 | W3 |
| W5 | Report T2-T7, success table, proposed rule, driver and smoke, runbook | ~150 / ~140 / 100 | W4 |

Each slice stays near 150 code lines and adds only new `size_width_*` modules; EXP-188's `size_pnl_*` code is imported unchanged.
