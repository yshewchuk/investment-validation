# Short-vol structure width tuning (design, EXP-189)

**Status: DESIGN ONLY (user decision 2026-11-24).** The code slices (section 11) are NOT started until this design merges. This PR adds this file only: no code, no `spec.yaml`, no test, and no edit to EXP-188's design or spec. Experiment id **EXP-189** (the slice W1 re-checks that it is free).

**Why a separate experiment.** EXP-188 (`guides/size_decision_pnl_design.md`) compares forecast models and is to stay production-exact. Width tuning is a different question, so it gets its own spec, pre-registration and PLANNED row, and never changes EXP-188's spec or hash. EXP-189 uses EXP-188's machinery as a library.

## 1. Question and evidence so far

Does a different strike width, set as a multiple `k` of production's width rule (`k = 1` is production exactly), earn more for the seven put-side short-vol structures? Measured on the EXP-188 2025 run: one `c` per model picked the grid edge `c = 0.5` (width 2x) for 11 of 12 models. The synthetic oracle (T11) shows P&L peaking at about 1.5-2x the actual move for BFLY-P, BFLY-P5, CND-PS, RAMP7 and CTR5, while TWIN-P and TWIN-P5 are fine near 1x. EXP-137 found chosen widths at 1.3-1.7x the forecast (`engine/forecast_sizing.py` comment). The roster's errors are close (inner RMSE 7.05-7.50, MAE 4.28-4.45) and all models under-predict large moves.

**Disclosure.** 2025 results of the EXP-188 runs (including `retuned` and T11) were seen while designing this. Every 2025 number here is therefore labelled **"2025 NOT blind for widths"**. What stays blind: the success rule and every threshold below are fixed before any EXP-189 run, and all tuning reads 2023-24 only.

## 2. Constraints

- Production's rules stay the base. `k` multiplies the width `forecast_params` returns: `width = k * (forecast/100)/divisor`. `WIDTH_MIN`/`WIDTH_MAX` apply **after** `k`; a width outside them is declined (non-trade, P&L 0, counted, as today). No edit to `engine/forecast_sizing.py`, `STRUCTURES`, any SIZE search module or any file in `source_closure(root, ["experiments/size_panel.py"])`; `k` lives in a new `size_width_*` module that calls the existing functions.
- Library reuse, no EXP-188 edit: the new modules import EXP-188's `size_pnl_*` code (decisions, pool, fast path, marks, `size_pnl_account`) unchanged. If a hook is missing, that is a separate preparatory change under EXP-188's own owner, never a change smuggled into an EXP-189 slice.
- No call-side short leg; marks only from `engine/fills.py`; holdouts only through the shared admission (EXP-188 sections 2 and constraints). Non-recorded runs use `--no-ledger`; the recorded run needs its PLANNED row (section 9).
- **Standard-tool check.** numpy only (`default_rng`, `polyfit`); nothing hand-rolled beyond a 3-parameter fit.

## 3. The five tunings

All tune on 2023-24 (burn-in as EXP-188: scoring from 2023H2), freeze before 2025, and run 2025 by month (R6). Same grid, objective and tie rule for every tuning (section 5).

| Id | Name | Factors | Tuned on | Role |
| --- | --- | --- | --- | --- |
| A | `global` | one `k` per model, pooled over the seven structures | the model's own real 2023-24 P&L | what EXP-188's `retuned` did, grid extended past 2.0 |
| B | `per_structure` | seven `k_s`, shared by every model, plus one level adjustment per model | reference fake model (section 4); no P&L of any real model | the headline candidate |
| C | `per_structure_model` | seven `k_s` per model | each model's own real 2023-24 P&L | overfitting check against B |
| D | `m0_width` | `k = 1` | none | every model's structures sized from M0's forecast (production rules); only the gate decision uses the model's own forecast; isolates selection from width |
| anchor | `oracle_sweep` | descriptive only | forecast = actual move | sets the grid's sense of scale |

Baseline for every comparison: **as-is** (`k = 1`, each model's own forecast), computed by EXP-188's code; EXP-189 recomputes it in its own run from the same pinned inputs so a cited number is never mixed with a recomputed one.

**Anchor (descriptive).** On 2023-24 only: for each structure, width = `k` x the **actual** move (the forecast is replaced by the label `y`, production divisor, factor `k`), through the same gate, marks and exit. Table T3 prints the score by `k` per structure. It never feeds a freeze or a success rule; it shows where a perfect forecast would put the peak and so whether the grid is wide enough.

## 4. Tuning B: error model, reference fake model, sweep

**B1. Error model fitted on 2023-24 inner real forecasts vs labels.** For model `m`, inner paired events `e`, forecast `f`, label `y = abs_move` (both positive; others dropped, counted). On the log scale (the scale of EXP-188's `regression_direction` generator and `sd(log(y/pred))`):

`log f = L_m + b_m * (log y - mean_e(log y)) + s_m * z`, `z ~ N(0,1)`.

`L_m`, `b_m` are the OLS intercept (at the mean) and slope; `s_m` the residual sd. `b_m < 1` is the under-prediction of large moves. Judgement call (mine): log scale so a forecast stays positive. The Gaussian noise is pre-stated; the report prints each model's residual skew and excess kurtosis, and a roster median excess kurtosis above 1 is flagged "poor fit" but changes nothing in this run.

**Reference fake model.** ONE model with `(L_ref, b_ref, s_ref)` = the roster median of each parameter over models with a fit. T1 lists every model's `(L, b, s, n)` and the reference. **Hard check:** the reference's RMSE and MAE on inner E (mean over draws, percent points) must lie inside the roster's inner `[min, max]` (7.05-7.50 and 4.28-4.45, re-read from the EXP-188 inner receipt), with 5% slack per bound (my judgement call). Otherwise the whole of B halts `reference_not_calibrated` (A, C, D and the anchor still run).

**B2. Sweep (2023-24 only).** For each structure `s` and each `k` in the grid, score the reference over `N` seeded draws: draw `d`, event `e` uses `numpy.random.default_rng(sha256(f"{event_id}|exp189-v1|{d}"))`, so `z` is common across structures and `k`. Each draw is a fake model with its own per-model `ResidualPool` (strictly earlier events, `err_move = y - f`), the same trailing-cutoff gate, `FillModel` marks, exit and burn-in as EXP-188, via the fast path (one simulation per (draw, event) at each ladder strike; each (structure, `k`) is a dot product; the cutoff is recomputed per `k`).

**B3. Freeze, level adjustment.** The seven `k_s` go to an immutable `tuning_B` receipt before any real model is scored under B. Per real model, ONE level adjustment `r_m = median_e(f_ref,e / f_m,e)` over 2023-24 inner E (`f_ref,e` = the mean over the `N` draws; no P&L used); the model's B forecast is `r_m * f_m` sized with `k_s`, and the gate uses it too. Median vs mean ratio is open decision 4; the report prints both.

## 5. Grid, objective, tie rule (shared by A, B, C and the anchor)

- **Grid (recommended; open decision 2):** `k in {0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}` (`k = 1/c` in EXP-188's terms). It extends past 2.0 so a 1.5-2x peak cannot sit at the edge. **Edge rule:** an argmax at 3.0 or at 0.5 is not frozen: `k_at_edge`; that factor runs at its best interior value, is flagged, and widening needs a new design change.
- **Objective (recommended; open decision 1):** per (unit, `k`), the **stand-alone cash-secured $100,000 account** return (EXP-188 Amendment 2 account (b); `margin.simulate` unchanged) under the primary marks and under `marks_mid`; the score is the **lower of the two** (a `k` that wins only under optimistic marks is not picked), averaged over draws for B. For A the unit is the model's single account over its seven books; for B and C the unit is the structure's account. Why not flat-dollar P&L: it was dominated by penny-debit trades (EXP-188 Amendment 2: 6,666 lots on a $0.015 debit).
- **Tie rule:** `k` within one standard error of the best (across draws for B; across quarters for A and C): the `k` nearest 1, then the smaller. This keeps production unless the data clearly prefer otherwise.
- **Draws `N` (open decision 3):** 20, standard error reported.
- A model with fewer than 300 selected inner trades for a unit skips tuning that unit (`k = 1`, labelled), as EXP-188 section 5.

## 6. Run order, outputs, report

Stages: `inner` (fit, sweeps, A and C tuning, anchor) writes the `tuning_A`, `tuning_B`, `tuning_C` receipts; `outer` (2025) refuses without them. Each arm is also run on the inner window for the in-sample reading (2023H2, 2024). One heavy job at a time, `bounded_run`, quarter chunks, progress to stderr at a bounded cadence (stage start/end, done/total, rate, ETA, RSS). Cost (not measured; the slice smoke measures): B is `N` fast-path simulations of one model; C is seven units x the roster.

`REPORT.md`, every 2025 table titled "2025 NOT blind for widths": T1 error-model parameters; T2 sweep curves per structure for B with score, standard error, trades, declined share and **flatness** (best minus the score at `k = 1` and its neighbours in standard errors; the span of `k` within one standard error of the best); T3 the oracle anchor; T4 chosen factors per model for A and C, and the seven `k_s` for B; T5 as-is, A, B, C, D side by side per model and per structure: per trade (return per taken trade, trade count), `marks_mid`, and cash-secured accounts (account (a): seven structures, both mark sets; final equity, return, drawdown, wanted/funded); T6 the success table (section 7); T7 the overfitting reading (C vs B). No bootstrap interval is added; a flat curve is the report's main caution.

## 7. Success rule (written before any run)

Per arm X in {A, B, C, D}, over testable roster models (at least 8 of 12), in the cash-secured seven-structure account (a): X **passes a cell** when its final return exceeds as-is for at least 9 of the testable models (my judgement call, as the 75% of EXP-188 section 7) and the median difference across models is positive. Cells: {2025, 2023H2, 2024} x {primary marks, `marks_mid`}. **B succeeds** iff all six cells pass (headline). A, C and D are reported with the same table and no multiplicity adjustment; none of them can make B succeed. 2023H2 and 2024 are in-sample for the error model, level adjustment and (for A, C) the factors; B's `k_s` come from fake forecasts, but 2025 is still not blind for widths.

- **Overfitting (C vs B).** C is flagged `overfit` when it passes the in-sample cells but fails a 2025 cell, or when B's median 2025 final return is above C's. If C beats B in 2025 too, the shared `k_s` are too coarse and the report says so.
- **D** is read beside B in prose only: if D captures most of B's gain, the gain is width; if B gains and D does not, the forecast, not the width, is limiting. No category.
- **Feeding the board (a separate decision).** A pass is evidence for a board width change, not a change. A candidate new divisor is `divisor / k_s` on the production forecast only (no level adjustment). The decision is the user's, needs M0's B row (`r_M0` near 1 shows the level is not the driver), the `marks_mid` cells, T2's flatness (a flat curve argues to leave the board alone) and, per EXP-188 Decision 3, a DYN-SV chooser review because the chooser reads `width_over_forecast` features. A failure leaves the board unchanged.

## 8. Failure semantics (R1-R6)

| Case | Result |
| --- | --- |
| R1 model with no fit (under 300 inner events, non-positive forecasts, no inner decisions) | no B row for it, `not_testable`; counted; nothing imputed |
| R1 missing label, or M0 forecast missing for an event under D | EXP-188's exclusion reasons (`move_label_missing`, ...); D drops the event for D only, counted |
| Whole-run refusals | EXP-188's identity ones; `EXPERIMENT_IDENTITY_CONFLICT` with `details.reason` in `tuning_receipt_missing`, `tuning_receipt_mismatch`, `planned_row_missing`, `spec_hash_mismatch` |
| Reference outside the roster range | `reference_not_calibrated`; B omitted, other arms run |
| (structure, `k`) with no feasible width (all declined) or under 300 trades summed over draws | `not_scored`, excluded from the argmax, declined share printed |
| All `k` of a unit excluded | `k_defaulted` = 1.0, flagged |
| Argmax at an edge | `k_at_edge` (section 5) |
| Candidate declined by width in 2025 | non-trade, P&L 0, stays in E (EXP-188 section 4) |
| R2 cost | one process, `bounded_run`, quarter chunks; over-limit refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report |
| R3 | seeds are the sha256 of event id, tag `exp189-v1` and draw `d` (section 4); roster order fixed in the spec |
| R4 | receipts atomic, immutable, bound to the inputs receipt; a changed input refuses at replay |
| R5 | fake models carry ids `synthetic-w-<draw>`, never in the roster; `model_id` of real models unchanged |
| R6 | tuning reads 2023-24 only; 2025 refuses without the three tuning receipts; each month's decisions are frozen before its labels are read |

## 9. Registration, spec hash

EXP-189 has its own `spec.yaml` (roster reference by EXP-188 plan id, arms, grid, `N`, thresholds), its own PLANNED ledger row with the spec hash, and the same guard that cannot no-op (`planned_row_missing`). It reads the sealed v14 inputs through EXP-188's `Sources` adapter, unchanged. EXP-188's spec and hash are untouched; the merge of this design changes neither. Nothing is registered in this PR.

## 10. Open decisions for the user (recommendation first)

1. **Objective.** Recommended: stand-alone cash-secured account return, lower of primary and `marks_mid`. Alternatives: flat-dollar P&L (outlier-prone), `marks_mid` alone, a rank combination. The recommendation costs about 2 x 7 x 10 x `N` account simulations for B.
2. **Grid.** Recommended: ten points up to 3.0. Alternatives: add 4.0 (most candidates decline at `WIDTH_MAX`), or drop 0.5/0.67.
3. **`N` draws.** Recommended 20 (fewer than 10 not recommended; raise if curves are too noisy to rank).
4. **Level adjustment.** Recommended median ratio (robust to tiny forecasts); mean ratio is the alternative.

## 11. Split list (code slices, NOT started until this design merges)

| Slice | Scope | Code / test lines / doc words | After |
| --- | --- | --- | --- |
| W1 | Scaffold EXP-189 (`spec.yaml`, PLANNED row, guard), `k` sizing module, id check | ~110 / ~120 / 60 | this PR |
| W2 | Error-model fit, reference model, calibration check, T1 | ~120 / ~120 / 40 | W1 |
| W3 | Seeded draws, sweep, objective, `tuning_B` receipt, T2, oracle anchor T3 | ~170 / ~170 / 60 | W2 |
| W4 | Tunings A and C, level adjustment, B and D arms on 2025 and inner | ~160 / ~160 / 50 | W3 |
| W5 | Report T4-T7, success table, driver and smoke, runbook | ~150 / ~140 / 100 | W4 |

Each slice stays near 150 code lines and adds only new `size_width_*` modules; EXP-188's `size_pnl_*` code is imported unchanged.
