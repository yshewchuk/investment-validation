# Per-event quantile model of the earnings move (design, EXP-190)

**Status: design only.** Adds this file; no code, `spec.yaml` or test. Experiment **EXP-190** (user, 2026-11-25; the ledger ends at EXP-189; slice Q1 re-checks). **[D1]**-style tags are choices with my default (section 10).

## 1. Question and evidence

Can one model give each event a usable distribution of |move|, sharper than what production has, and does that help the trading decisions?

**Motivation** (EXP-188 full7, flat sizing at mid, 2024 and 2025; **supervisor-measured 2026-11-24/25, not re-derived here**):

- SIZE point models sit at different percentiles of actual |move| (E-log about p43, M3/H-abs about p47.5, production M0 about p53). Forecasts correlate 0.95-0.975; v14 pinball and coverage are nearly identical across all 125 candidates.
- Centre-peaked short vol (BFLY-P, BFLY-P5, CND-PS, CTR5, RAMP7) earned more with a low forecast (E-log best both years). TWIN-P and TWIN-P5 earned more with an accurate median (M3 and M0 beat E-log both years).
- Straddles favoured high forecasts (M5, M1 top; E-log near bottom). **Weak:** 2025 only, and `G_PROD` is not production-faithful.
- The M0 / E-log gap predicted twin profit in both years for all three models examined.

Reading: each strategy wants a different point of the move distribution.

**Disclosure.** These 2024 and 2025 results shaped this design (the levels, families and filter), and 2023 was seen in EXP-188 inner diagnostics. **No period is blind for trading**; the quantile model itself has not been run. Independent confirmation is only the forward period after the data end, not read here.

## 2. What is different from earlier work

| Prior work | What it did and found | Difference here |
| --- | --- | --- |
| EXP-112, EXP-115 | Residual pools by forecast decile, adopted: today's `ResidualPool`, 10 buckets. 80% coverage 0.7276 to 0.7934; no established trade effect (STR-THRU only, under simulation noise) | The bar. Its width varies only with forecast level |
| EXP-114 | Quantile regression of the **residual** (p10, p90) had the worst conditioned coverage (0.7516). `sigma_model_oof` (post-hoc) had the best CRPS (2.7875) but lost on coverage | Quantiles of the **move**, 7 levels, tuned, causally calibrated; its scale idea returns as arm S |
| EXP-173 | Surface-conditioned residual scale lost 80% coverage (0.7932 to 0.7748) | No new features |
| `DISTRIBUTION_PREDICTION`, `STR_THRU_DISTRIBUTION` | Histogram classifier; never run on real data | Quantile loss, not bins; no reuse planned |
| `SUPER_SIZE/STAGE2_DESIGN.md` section 3 | HGB q = 0.1/0.5/0.9, no crossing repair; never run | 7 levels, fixed non-crossing rule, inner-trained calibration |

New: the output is **monotone**, it is judged on **conditional** sharpness (information beyond forecast level), and it connects to strikes, the entry simulation and two filters.

## 3. Constraints

- No edit to `source_closure(root, ["experiments/size_panel.py"])`, the v14 SIZE closure sealed by `experiments/size_search_binding.py` (`size_search.py`, `magnitude_search_worker.py` and imports, including `engine/structures.py`, `engine/fills.py`, `engine/pnl_sim.py`), or any EXP-188 or EXP-189 file or spec; neither closure imports a new module. Q1 tests both closure hashes unchanged.
- New code only in `experiments/size_quantile_*` modules and one `EXP-190_*` directory; `size_pnl_*` and `size_width_*` imported unchanged. Nothing enters `STRUCTURES`, the registry or the board.
- Pricing through `engine.chain_index` (`price_structure_indexed`); marks only from `engine/fills.py`; no call-side short leg.
- No holdout reads: shared population admission first; labels read by admitted event keys only. `--no-ledger` for non-recorded runs.
- **Heavy runs only after EXP-189's full run**, one at a time, under `bounded_run`.

## 4. Target, model, calibration

**Target and outputs.** `y = abs_move` (percent points), lead 1. Quantiles **p10, p25, p40, p50, p60, p75, p90** [D1].

**Data, folds, features.** The sealed v14 SIZE panel and its 2023, 2024 (inner) and 2025 (outer) evaluation folds. Sealed records are read by content hash, as in EXP-188 design section 13; never through `read_search_plan`, `prepare_plan` or `verify_installation`. Features are M0's 14 (`size_v1_4`), all in the panel through the columns of `experiments/size_comparator.py` (`FEATURES`, `COLUMNS`), with its masks and complete-case rule (`n_prior >= 4`, `or_implied` in [0, 60], `rvol30` in [0, 700], training target in [0, 200]) [D2]. A training row needs its label **available before the fold's cutoff** (data dates, never `computed_at`). Q2 refuses if panel or folds do not bind; nothing is rebuilt.

**Primary arm Q** [D3]: one `HistGradientBoostingRegressor(loss="quantile", quantile=q)` per level, fit once per fold [D4]. `early_stopping=False` (its random split ignores time order), fixed seed.

**Secondary arm S** (reported; it cannot pass in Q's place): median plus scale, using the chosen configuration. An `absolute_error` HGB gives the median; a second fits `|y - median|` on **out-of-fold** residuals from year-blocked expanding folds inside the training rows (the EXP-114 in-sample flaw); `q = median + scale * z_q`, `z_q` from earlier standardised out-of-sample residuals.

**Grid** [D5]: `learning_rate` 0.03, `max_leaf_nodes` {7, 15}, `min_samples_leaf` {300, 1000}, `max_iter` {500, 1500}: 8 configurations, **one shared by all levels**, lowest event-weighted mean 7-level pinball of raw sorted outputs. Selection is walk-forward: 2024 uses 2023 pinball, 2025 uses 2023-24, and 2023 uses the fixed default (leaves 7, leaf 300, 500 iterations). Ties: fewer leaves, then larger leaf, then fewer iterations. Main fits: 7 x 8 x 3 = 168.

**Non-crossing** [D6]: **sort** the seven outputs per event (rearrangement never raises summed pinball loss over the levels). Not chosen: a base quantile plus non-negative offsets, which needs a custom sequential fit.

**Causal conformal calibration** [D7]. For month m and level q, `delta_q` is the empirical q-quantile of `y - qhat_q` over the arm's own out-of-sample predictions whose labels were available before m began, trailing 12 months, at least 500 events. Apply `qhat_q + delta_q`, **sort again, floor at 0**. Fewer than 500: unscored (`calibration_burn_in`). No label of month m or later enters. Arm `Q_raw` (no calibration) is reported.

**Full distribution** (simulation, probabilities) [D8]: inverse CDF linear between the seven knots. Above p90, quantiles of `(y - q90)/max(q90 - q50, 0.5)` over strictly earlier calibrated predictions with `y > q90` (same window), scaled by the event's width; below p10 likewise with `q50 - q10`, floored at 0. Under 50 donors in a tail: unscored (`tail_pool_thin`). The 0.5-point width floor is my judgement call.

**Tool check.** scikit-learn (1.9.0 here) has the loss; numpy has quantiles, sort and interpolation. Per-level calibration is one `np.quantile`, so no conformal library is added. Reused: `size_pnl_contrast` bootstraps, `size_pnl_account`, `engine.evaluate.breakeven_alpha_from_sweep`.

## 5. Stage A: distribution skill (no trading)

**Baselines**, reduced to the same seven levels:

| Id | Baseline | Construction |
| --- | --- | --- |
| **B0 (the bar)** [D9] | production forecast + 10-bucket `ResidualPool` | stored tier-4 `pred_abs_move`; native full-history pool, strictly earlier; quantiles of `max(f + err_move, 0)` over the exact eligible bucket (from `draw` evidence) |
| B1 | M0 with the v14 band | the residual-ratio band of `size_distribution.v1` (5 forecast buckets, `max(f, 0.05) * Q_q(y / max(pool_f, 0.05))`), re-implemented at these levels with a parity test at shared ones; donors are earlier M0 forecasts |
| B2 | option-implied | panel `implied_move` times trailing earlier quantiles of `y / implied_move`; null and zero sentinel abstain; compared on its own subset. An implied centre with a historical shape, **not** a chain-implied density |

**Population.** Events where Q and B0 both have output; every drop counted by reason and year; B1 and B2 on common subsets. Periods: **2023H2, 2024, 2025** (2023H1 is burn-in); ten quarters.

**Metrics** per period, quarter and pooled, per arm: pinball per level and mean; coverage of p25-p75 and p10-p90 (inclusive) and each level; CRPS of the full distribution (B0 from its bucket draws); raw **crossing rate** (events and adjacent pairs); median width `q75 - q25`; MAE of p50 beside M0's.

**Conditional tests.** Cells are quarter x B0's own forecast decile, so forecast level, the only thing B0's width uses, is held nearly fixed. `W = q75 - q25`, `e = |y - q50|`:

- Spearman(`W`, `e`) within cells, event-weighted, with a ticker bootstrap; B0's value printed beside it.
- `W` terciles within cells, pooled by period: mean `e`, realised interquartile range of `y` and both coverages per tercile.
- Reported: the same with `W / q50` (relative width), and by plain predicted-width tercile (the unconditional view, expected to look good through level alone).

**Pre-registered pass rule** [D10]. Only Q can pass. All of:

1. **Pinball:** Q's mean is below B0's in each of the three periods and in at least 8 of 10 quarters; the paired 90% bootstrap intervals by quarter **and** by ticker (10,000 resamples, fixed seed) exclude 0.
2. **Conditional sharpness:** in each period the within-cell Spearman is positive and the wide tercile's mean `e` exceeds the narrow tercile's; the pooled ticker interval for the Spearman excludes 0.
3. **Calibration guard:** for each of the two bands, `|coverage - nominal|` is no worse than B0's in at least 5 of 10 quarters.

No magnitude cutoff. The quarter counts are my judgement calls; raw counts are printed. B1, B2, `Q_raw` and S are reported with the same tables; a pooled pinball loss to B1 or B2 is flagged in the headline, not gated. **Fail:** Stage B is not run. A pass for S or `Q_raw` alone is a lead for a new pre-registration.

Walk-forward selection keeps no year judged by a configuration chosen on it.

## 6. Stage B: trading (only if A passes)

EXP-188's production-exact machinery as a library (entry rule, per-structure trailing cutoffs, `marks_mid` and harsh marks, exits). **P0, production, is recomputed in the same run** as the control: board widths at the pinned commit [D11] and, unlike EXP-188's 2023+ pool, the native full-history `ResidualPool` (as B0). Q output before 2023-07 serves only as calibration, tail and cutoff history; comparisons use events determined in both arms.

**Distribution draws.** A per-event `QuantilePool` wraps production's pool and exposes the two methods the simulators call, `draw` and `before`. `draw` calls production's draw (same M0 forecast, count and random stream), takes each drawn `err_move`'s rank `u = (rank + 0.5) / n` among the draws, and returns `Qinv_event(u) - forecast`, so the simulated move is the event's own quantile at `u`. `err_crush` is returned unchanged, so the move / crush rank dependence of the historic pairs is kept [D12]. `pnl_sim.expected_pnl` is unmodified; the fast path must match it within 1e-9.

**Arms.**

| Arm | Strike width | Entry simulation or rule |
| --- | --- | --- |
| P0 | `forecast_params(name, M0)` | production |
| W | `forecast_params(name, Q_event(tau))` | production (M0, residual pool) on the new geometry |
| LW (control for W) | `forecast_params(name, kappa * M0)`; `kappa` is the trailing 12-month median of `Q(tau) / M0`, strictly earlier events | as W |
| D | production | `exp_pnl_sim` with `QuantilePool` |
| WD, LWD | as W, LW | `QuantilePool` |
| F (TWIN pair) | production | production, and `R = (q75 - q25)/q50` at or below its trailing median [D13] |
| ST, ST0 (straddle) | STR-THRU | buy when `P(move > implied_move)` is in the top 20% of that arm's own prior scores: ST from Q, ST0 from B0 |

**tau** [D14]: **fixed, not tuned**: 0.40 for the five centre-peaked structures, 0.50 for the TWIN pair, the families of EXP-189 arm D. The 0.40 comes from the post-hoc E-log observation. P&L at {0.25, 0.40, 0.50, 0.60} is printed descriptively and selects nothing. `WIDTH_MIN`/`WIDTH_MAX` declines apply as in production, counted.

**Why LW is needed.** A low quantile mostly narrows every strike, which is the level effect EXP-189 already tunes with a global multiple. LW has W's typical level without its per-event variation; **W counts only if it also beats LW.**

**Filter and straddle.** F's direction (narrow band kept: twins want an accurate median) is fixed now; the other half is printed. The F cutoff needs 100 prior scores, strictly before the month, 6-month window, like ST's. F is **not** a test of the M0 / E-log gap finding; band width is a different quantity. ST uses EXP-188 Amendment 5's universe, with ST0's rule differing only in the distribution; `G_PROD` is printed labelled "not production-faithful".

**Outputs.** By structure, family and year, at `marks_mid`: P&L per candidate event and per taken trade, trades, declines; **breakeven alpha**; harsh alpha-0 result (reported only); quarter and ticker bootstraps against each control. **Accounts: $200,000 cash-secured**, the EXP-188 / EXP-189 policy (`margin.simulate`: cap 0.66 of current equity, `target_share` 0.25, expensive first, short puts secured at strike x 100; straddle secured at its debit). Accounts: each structure, the centre-peaked five, the TWIN pair, all seven, straddle.

**Success rule** [D15]. W, D and WD are judged per family (centre-peaked, twin); F on the twin account; ST on the straddle account. A candidate passes iff:

- its account return at `marks_mid` beats its control's in **each** of 2023H2, 2024 and 2025 (P0; ST0 for ST; each period a fresh account), **and** W and WD also beat LW and LWD;
- its 2025 breakeven alpha is no more than P0's (ST0's) + 0.05;
- for the centre-peaked family, it loses to P0 in at most 1 of 5 stand-alone structure accounts in 2025; for the twin family (W, D, WD, F), in neither.

The margin and guards are judgement calls. That is eight candidates with no multiplicity correction, so one chance pass is plausible; all are printed. ST is evidence only and cannot recommend a change. A pass is a **proposal**: any board change is a separate user decision, with the DYN-SV chooser review (EXP-188 Decision 3).

## 7. Failure semantics (R1-R6)

| Case | Result |
| --- | --- |
| R1 panel, folds or M0 series do not bind | refuse `EXPERIMENT_IDENTITY_CONFLICT`; no partial report, nothing rebuilt or imputed |
| A quantile fit fails | that fold's arm is `fit_failed`, no retry; if it is Q, A is `not_testable` and B refuses |
| Burn-in, thin tail pool, incomplete features, non-finite output | unscored (an abstention, not "no"), counted; the event leaves the paired population |
| Width declined or ladder too coarse | non-trade at P&L 0, counted; the event stays |
| Missing implied move | leaves B2's subset and both ST and ST0 |
| F or ST cutoff under 100 prior scores | undetermined, counted |
| Breakeven alpha with no crossing, or no trades | 0.0 if always positive, 1.0 if always negative; no trades: the candidate fails |
| B without an A pass receipt, or a changed spec or missing PLANNED row | refuse (`stage_a_not_passed`, `spec_hash_mismatch`, `planned_row_missing`) |
| R2 cost | quarter chunks; over-limit refuses `RESOURCE_LIMIT_EXCEEDED` |
| R3, R4 | fixed seeds; the trading draw key is `event_id`, shared across arms; receipts atomic, immutable, bound to inputs; a changed input refuses |
| R6 causality | fits, selection, calibration, tail and cutoff pools use data available before the fold or month; a month's predictions and decisions are written before its labels are read |

Long loops log stage start and end, done/total, rate, ETA and RSS to stderr at a bounded cadence.

## 8. Limits

- **Refit cadence:** Q refits yearly, production monthly; monthly calibration only partly closes the gap, which handicaps Q.
- **Training data:** Q trains on the v14 panel; stored M0 forecasts come from production's own history. A fail does not separate model class from data; p50 MAE beside M0's is the diagnostic.
- Seven correlated structures and one 2025 are weak evidence; a Stage A pass need not move P&L (EXP-115).

## 9. Registration

Own `spec.yaml` (levels, grid, selection, calibration, tail rule, baselines, pass rules, arms, tau), own PLANNED row with the spec hash (supervisor registers), and a guard that cannot no-op, all before any fit on real data. Stages: `fit` (predictions and receipts), `stage_a` (verdict receipt), `stage_b`. Contract lines land in the `ARCHITECTURE.md` files with the slices that change them.

## 10. Open decisions for the user (defaults apply unless changed)

| # | Decision | Default | Main alternative |
| --- | --- | --- | --- |
| D1 | Levels | 7, as listed | add p05, p95 for tails |
| D2 | Features, missing values | M0's 14, complete case | HGB native missing; or the v14 HMOSXLP set (a later experiment) |
| D3 | Primary model | per-level quantile HGB | median + scale primary |
| D4 | Refit cadence | per fold (yearly) | monthly (about 12x fits) |
| D5 | Grid | 8 shared configs | choose per level |
| D6 | Non-crossing | sort | cumulative offsets |
| D7 | Calibration | per-level, 12 months, 500 | none; expanding window |
| D8 | Tails | scaled empirical exceedances | add p05/p95 knots |
| D9 | The bar | B0 | also require beating B1 |
| D10 | Stage A rule | section 5 | other quarter counts |
| D11 | Control if EXP-189 changes widths | pinned current production | add EXP-189's winner as a second control |
| D12 | Crush draws | rank-coupled to historic pairs | independent crush |
| D13 | Twin filter | keep at or below trailing median | omit; other cutoff |
| D14 | tau | fixed 0.40 / 0.50 | choose on 2023H2-2024 from the four values |
| D15 | Stage B rule | section 6 | drop LW requirement or change margin |

## 11. Split list (code slices; none started until this design merges)

Estimates: non-test code lines / test lines / doc words. Each is within the AGENTS.md limits (code target about 150).

| Slice | Scope | Size | After |
| --- | --- | --- | --- |
| Q1 | Scaffold: `spec.yaml`, PLANNED row, guard, id check, closure-hash tests | ~110 / ~120 / 60 | this PR |
| Q2 | Sealed panel, fold, feature and label reader; admission; drop counts | ~160 / ~180 / 80 | Q1 |
| Q3 | Grid fits, walk-forward selection, sort, calibration, tails, receipts; arm S split out as Q3b | ~170 / ~180 / 60; Q3b ~80 / ~90 / 20 | Q2 |
| Q4 | B0, B1, B2 | ~140 / ~160 / 60 | Q2 |
| Q5 | Metrics, conditional tests, bootstraps, pass rule | ~170 / ~180 / 60 | Q3, Q4 (pure; may start on fixtures) |
| Q6 | Stage A driver, report, 50-ticker smoke, runbook | ~130 / ~100 / 100 | Q5 |
| Q7 | `QuantilePool`, fast-path equivalence | ~110 / ~140 / 40 | A passes |
| Q8 | W, LW, D, WD, LWD arms and tau table | ~170 / ~170 / 60 | Q7 |
| Q9 | Twin filter; ST and ST0 | ~140 / ~150 / 40 | Q7 |
| Q10 | Accounts, breakeven, success table, Stage B driver and report | ~170 / ~170 / 80 | Q8, Q9 |

Q7-Q10 are written only after a Stage A pass.
