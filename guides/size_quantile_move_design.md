# Per-event quantile model of the earnings move (design, EXP-190, part 1: Stage A)

**Status: design only.** No code, `spec.yaml` or test. Experiment **EXP-190** (user, 2026-11-25; Q1 re-checks the ledger id). This revision supersedes PR #2869. **Part 1** (this file) is the model and Stage A, which does no trading. **Part 2** (`guides/size_quantile_trading_design.md`, a later design PR) is Stage B: fills, trading arms and success rule. **[D1]**-style tags point to section 10.

## 1. Question and evidence

Can one model give each event a usable distribution of |move|, sharper than production's, and does that help trading decisions?

**Motivation** (EXP-188 full7, mid marks, 2024-2025; **supervisor-measured, not re-derived here**):

- SIZE point models sit at different percentiles of actual |move| (E-log about p43, M3 about p47.5, production M0 about p53), though forecasts correlate 0.95-0.975 and v14 pinball and coverage are nearly identical across candidates.
- Centre-peaked short vol earned more with a low forecast (E-log best both years); the TWIN pair earned more with an accurate median.
- Straddles favoured high forecasts (weak: 2025 only, `G_PROD` not production-faithful).

Reading: each strategy wants a different point of the distribution.

**Disclosure.** 2024-2025 results shaped this design, and 2023 appeared in EXP-188 inner diagnostics. **No period is blind.** The quantile model has not been run; independent confirmation is only the forward period after the data end, not read here.

## 2. What is different from earlier work

| Prior work | Found | Difference here |
| --- | --- | --- |
| EXP-112, EXP-115 | Residual pools by forecast decile, now `ResidualPool` (10 buckets): 80% coverage 0.7276 to 0.7934; no established trade effect | The bar. Its width varies only with forecast level |
| EXP-114 | Quantile regression of the **residual** had the worst conditioned coverage (0.7516); `sigma_model_oof` had the best CRPS but lost on coverage | Quantiles of the **move**, causally calibrated; scale returns as arm S |
| EXP-173 | Surface-conditioned scale lowered 80% coverage (0.7932 to 0.7748) | Quantile fits, not a rescaled residual |
| `experiments/SUPER_SIZE/STAGE2_DESIGN.md` §3 | HGB q = 0.1/0.5/0.9, no crossing repair; never run | 7 monotone levels |

New: output judged on **conditional** sharpness (information beyond forecast level).

## 3. Constraints

Two sealed closures must hash unchanged (Q1 tests both against its merge base):

- **A, v14 panel:** `source_closure(root, ["experiments/size_panel.py"])`.
- **B, v14 search:** `source_closure(root, ["experiments/size_search.py", "experiments/magnitude_search_worker.py"])` plus the `engine/models/registry.json` hash, as `experiments/size_search_binding._source_files()` builds it.

Neither imports a new module; `engine/fills.py` and `engine/structures.py` do not change; no EXP-188 or EXP-189 file or spec changes. New code lives in `experiments/size_quantile_*` and one `EXP-190_*` directory (the shared fill module is `experiments/fair_value_*`). Nothing enters `STRUCTURES`, the registry or the board. No holdout reads: shared population admission first, labels by admitted event key only. `--no-ledger` for non-recorded runs. One heavy job at a time, under `bounded_run`.

## 4. Target, model, calibration

**Target.** `y = abs_move` (percent points), lead 1. Quantiles **p10, p25, p40, p50, p60, p75, p90** [D1]. No separate mean model. Production's point forecast M0 is **not** a quantile-model feature; it feeds part 2's selection models.

**Data and features** [D2]. The sealed v14 SIZE panel, with the **full v14 feature set**: the `HMOSXLP` dictionary (the largest: `HMOSX`, its quote indicator, and the derived `log1p` and product columns from `engine.v2.research.derived_features`, imported unchanged). The dictionary hash binds to the sealed v14 plan. Row masks are v14's; missing feature values stay NaN, handled natively as in v14's HGB cells. Sealed records are read by content hash (EXP-188 design §13), never through `read_search_plan`, `prepare_plan` or `verify_installation`. A training row needs its label **available before the fold's cutoff** (data dates, never `computed_at`). Nothing is rebuilt.

**Walk-forward range.** Annual folds, refit once per year (each level and configuration) [D4]. Part 2's selection models need earlier out-of-sample quantiles, so the walk-forward runs from **Y0 through 2025**. Y0 is the earliest year such that in every year Y0..2025 at least 50% of admitted panel events have stored entry and exit option chains, and the panel holds at least 3 earlier calendar years of training rows (both thresholds my judgement). **No range is claimed here:** the supervisor is checking chain coverage, and Q2 prints per-year counts and Y0 before any fit. Sealed v14 folds serve 2023-2025; this experiment builds earlier annual folds from sealed panel rows, and Q2 refuses unless its rule reproduces the sealed 2023-2025 training ids. The feature set and grid were chosen using 2023-2024, so earlier years have causal fits but non-blind design choices.

**Primary arms** [D3, D7]: one `HistGradientBoostingRegressor(loss="quantile", quantile=q)` per level and fold. **Q** (causally calibrated) and **Q_raw** (uncalibrated) are **both eligible** in Stage A and both carry into Stage B.

**Arm S** (reported; cannot pass): median plus scale at the chosen configuration. An `absolute_error` HGB gives the median; a second fits `|y - median|` on **out-of-fold** residuals from year-blocked expanding folds inside the training rows; `q = median + scale * z_q`, `z_q` from earlier standardised out-of-sample residuals.

**Grid** [D5]: v14's HGB grid, model parameters only (quantile loss replaces the loss axis): `max_leaf_nodes` {7, 15} x `min_samples_leaf` {300, 1000}, with `learning_rate` 0.03, `l2_regularization` 10, `max_iter` 3000 and v14's early stopping (`validation_fraction` 0.1, `n_iter_no_change` 10, `tol` 1e-7, seed 0). Its random split stays inside the training rows, so no later data enters. Q1 refuses if the sealed v14 plan's values differ. This is **4 configurations**, the whole grid, so nothing is trimmed: **28 fits per year** (7 levels x 4), 224 if Y0 were 2018; arm S adds its own. **One configuration is shared by all levels**: for year Y, the lowest event-weighted mean 7-level pinball of raw sorted outputs over out-of-sample years Y0..Y-1, counting only labels available before Y's cutoff. Y0 uses leaves 7, leaf 300. Ties: fewer leaves, then larger leaf. The Q6 smoke measures fit time before the full run.

**Non-crossing** [D6]: sort the seven outputs per event (this never raises an event's summed pinball loss). Q_raw is the sorted output floored at 0.

**Causal conformal calibration** (Q). For month m and level q, `delta_q` is the empirical q-quantile of `y - qhat_q` over out-of-sample predictions whose labels were available before m began, trailing 12 months, at least 500 events (numpy's default linear method, as everywhere here; no coverage guarantee). Apply `qhat_q + delta_q`, sort again, floor at 0. Under 500: unscored (`calibration_burn_in`).

**Full distribution** [D8], per arm from its own predictions: inverse CDF linear between knots. Above p90, quantiles of `(y - q90)/max(q90 - q50, 0.5)` over strictly earlier predictions with `y > q90` (same 12-month window), scaled by the event's width; below p10 likewise with `q50 - q10`, floored at 0. Under 50 donors in a tail: unscored (`tail_pool_thin`). The 0.5-point floor is my judgement.

**Zero median.** Quantiles flooring to 0 are valid output, but a ratio over `q50` (`W / q50`; part 2's band ratios) is **undetermined** when `q50 <= 1e-9`: counted `zero_median`, never 0/0.

**Tool check.** scikit-learn and numpy suffice; no conformal library. `size_pnl_contrast` bootstraps are reused.

## 5. Stage A: distribution skill (no trading)

**Baselines**, reduced to the same seven levels:

| Id | Baseline | Construction |
| --- | --- | --- |
| **B0 (the bar)** [D9] | production forecast + `ResidualPool` | stored tier-4 `pred_abs_move`; native full-history pool, strictly earlier; quantiles of `max(f + err_move, 0)` over the exact rows `draw` samples: the bucket, or the whole earlier pool when the bucket is under `MIN_POOL`; a refused draw is unscored |
| B1 | M0 with the v14 band | `size_distribution.v1` residual-ratio band (5 forecast buckets, `max(f, 0.05) * Q_q(y / max(pool_f, 0.05))`), re-implemented with a parity test at shared levels; earlier M0 donors |
| B2 | option-implied | panel `implied_move` times trailing earlier quantiles of `y / implied_move`; null or zero abstains. An implied centre with a historical shape, **not** a chain-implied density |
| **B3** [D2] | best v14 point model + its own pool | sealed lead-1 predictions of **M3, the v14 MAE winner** (EXP-188 design §3); B0's construction on M3's own strictly earlier residuals, which start in 2023 |

**Why M3.** MAE is twice the p50 pinball loss, so the MAE winner has the best inner median pinball. If M3 does not use `HMOSXLP`, B3 is the lowest inner-MAE `HMOSXLP` candidate with stored predictions; a missing year is `not_testable`. I did not read the v14 report; Q1 resolves the id from the sealed selection record and prints M3's inner pinball and coverage beside it. M3 was selected on 2023-2024, so B3 is optimistic there. **B3 minus B0** shows what the better point model gains; **Q minus B3** what the distribution model adds. Neither is pure: B3's loss differs and its pool is shorter than B0's.

**Population.** Events where the judged arm and B0 both have output, drops counted by reason and year; B1-B3 on common subsets. Periods: **2023H2, 2024, 2025**; ten quarters.

**Metrics** per period, quarter and pooled, per arm: pinball per level and mean; coverage of p25-p75 and p10-p90 (inclusive) and of each level; full-distribution CRPS; raw **crossing rate**; median width `q75 - q25`; MAE of p50 beside M0's and M3's.

**"More accurate" means lower mean pinball (quantile) loss** [D10], averaged over the seven levels and events. In plain words: each level is charged for how far the actual move lands from it, with the two sides weighted so that a p90 pays more for being too low than too high; a smaller average means the seven numbers sit closer to where moves really fall.

**Conditional tests.** Cells are quarter x B0's bucket index, holding B0's forecast level nearly fixed. `W = q75 - q25`, `e = |y - q50|`. The arm's `q50` still varies in a cell, so ranks of `W` and `e` are each residualised linearly on the rank of `q50`. Cells under 30 events or with constant `e` are skipped and counted. A `W` residual that is zero (within 1e-9) scores correlation 0 and tercile difference 0.

- Correlation of the residuals (partial Spearman) within cells, event-weighted, ticker bootstrap; B0's value beside it.
- `W`-residual terciles within cells, pooled by period: mean `e`, realised interquartile range of `y` and both coverages.
- Reported only: the same with `W / q50`, and unconditional width terciles.

**Level-only width L (issue #2871).** That residualisation removes only a monotone `q50` effect, so a non-monotone function of `q50` could pass on level alone. For each month, from the arm's own out-of-sample `(q50, y)` pairs with labels available before it (expanding, at least 500), cut earlier `q50` into `min(20, n // 100)` equal-count bins. `L` for an event is p75 minus p25 of earlier `y` in its `q50` bin (outside the edges: the end bin). `L` knows nothing but level. It goes through **the same tests**, on the same events and cells as `W`. Under 500: the event leaves both comparisons (`level_width_burn_in`). Limit: a smooth level function finer than the bins could still edge past `L`.

**Pre-registered pass rule** [D10], applied separately to Q and Q_raw. All of:

1. **Accuracy:** the arm's mean pinball is below B0's in each period and in at least 8 of 10 quarters; for the per-event difference, arm minus B0, the 90% percentile bootstrap intervals by quarter **and** by ticker (10,000 resamples, fixed seed) lie below 0.
2. **Conditional sharpness:** in each period `W`'s partial Spearman is positive **and exceeds L's**, and `W`'s wide-minus-narrow tercile difference in mean `e` is positive **and exceeds L's**. Pooled over periods, the ticker interval for `W`'s Spearman, and that for `W`'s minus `L`'s, both lie above 0.
3. **Calibration guard:** for each band, `|coverage - nominal|` is no worse than B0's in at least 5 of 10 quarters.

No magnitude cutoff; quarter counts are mine. B1-B3 and S get the same tables; a pooled pinball loss to B1, B2 or B3 is flagged in the headline, not gated.

**Verdict.** Stage A passes if **Q or Q_raw** passes. Both then enter Stage B, each labelled with its own verdict. **Multiplicity:** two eligible arms give two tries, uncorrected (they are highly correlated). Otherwise Stage B is not run.

**Periods used for selection** (the report repeats this) [D15]: the grid choice for year Y uses years before Y; B3's identity, the feature set and the grid values used 2023-2024; **the Stage A gate, which decides whether Stage B runs, uses 2023H2, 2024 and 2025.** **2025 is the clean test for those model choices only:** no feature set, grid value, configuration or baseline identity is chosen on it (its monthly calibration, tail and L pools use earlier 2025 labels, causally). It is **not a clean holdout**: Stage B scores 2025 only after a pass that read 2025 outcomes, which can flatter its 2025 result, and EXP-188's 2025 results shaped the design. Only the forward period after the data end is independent.

## 6. Stage B (part 2)

Fill model **F** is shared with EXP-189, so it is specified first and on its own: `guides/fair_value_fill_design.md`. Part 2 specifies how Stage B uses it, with controls (production P0, EXP-189 arm E), quantile-driven arms, selection models and their control, accounts and success rule. Section 10 records its settled decisions.

## 7. Failure semantics (R1-R6)

| Case | Result |
| --- | --- |
| R1 panel, folds, dictionary, M0 or M3 series do not bind | refuse `EXPERIMENT_IDENTITY_CONFLICT`; nothing rebuilt or imputed. A merely absent M3 year is B3 `not_testable` for that year |
| A fit fails | that fold's arm is `fit_failed`, no retry. Q_raw's fits failing makes Stage A `not_testable`; a calibration failure affects Q only |
| Burn-in, thin tail pool, level-width burn-in, non-finite output | unscored (an abstention, not "no"), counted; leaves the paired population |
| `zero_median` | ratio undetermined, counted; quantiles stay valid |
| R2 cost | quarter or year chunks; over-limit refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report |
| R3, R4 | fixed seeds; receipts atomic, immutable, bound to inputs; a changed input refuses |
| R5 | arm ids hash the source, level set and configuration |
| R6 causality | fits, selection, calibration, tail, level-width and baseline pools use data available before the fold or month; a month's predictions are written before its labels are read |

Progress logging follows AGENTS.md.

## 8. Limits

- Q refits yearly, production monthly; monthly calibration closes only part of that gap, and Q_raw closes none.
- Q trains on the v14 panel, M0 on production's history, so a result cannot separate model from data.
- B3 shows the effect of features and point model together, not of features alone.
- A Stage A pass need not move P&L (EXP-115).

## 9. Registration

One `spec.yaml` holds every fixed rule of **both parts**. Its PLANNED row, carrying the spec hash (supervisor registers), and a guard that cannot no-op come **before any fit on real data, and only after part 2 merges**, so no Stage B rule is chosen after seeing Stage A. Part 1 code slices may start earlier on fixtures. Stages: `fit` (receipts), `stage_a` (verdict receipt), `stage_b`. Contract lines land in the `ARCHITECTURE.md` files with the slices that change them.

## 10. Decisions (settled by the user, 2026-11-25)

"Default" is the rule proposed in #2869. Parameters marked "mine" in either part are judgement calls inside these decisions.

| # | Settled |
| --- | --- |
| D1 | 7 levels: p10, p25, p40, p50, p60, p75, p90. No separate mean model; M0's point forecast is a **feature in both selection models** (N1, N2) |
| D2 | Full v14 SIZE feature set on the sealed v14 panel. New baseline **B3** = best v14 point model on it (M3), own causal residual pool, separating "better features" from "better distribution" |
| D3, D4, D6, D8, D9, D12 | Defaults: per-level quantile HGB; refit per fold; sort; scaled empirical tails; the bar is B0; crush draws rank-coupled to historic pairs |
| D5 | v14's hyperparameter grid (model parameters only, not feature sets or targets), selected walk-forward on mean pinball; trim near v14's winners if too large (4 configurations: not trimmed) |
| D7 | Q and Q_raw both eligible in Stage A, both carried into Stage B; added multiplicity stated |
| D10 | Default rule; "more accurate" means lower mean pinball loss. The L comparison addresses #2871's sharpness gap, within the limit section 5 states |
| D11 | EXP-189 recommended its **arm E** (per event and structure, the width multiplier k in 0.5..3.0 maximising `exp_pnl_sim`). E is a **second** Stage B control beside production |
| D13 | Default twin filter; the band-width ratio is also a selection-model feature |
| D14 (changed) | Width quantile chosen **walk-forward** per family (centre-peaked, TWIN pair) from {p10, p25, p40, p50, p60, p75}: 2024 uses the best on 2023H2, 2025 the best on 2023H2-2024; every quantile printed per year |
| D15 | Default rule; the report states the periods used for any selection; 2025 is the clean test (section 5: for model and parameter choices; not a holdout, since the Stage A gate reads it) |
| N1 | Learned **selection model per family** (one centre-peaked, one twin): HGB predicting each candidate trade's return under F, from calibrated quantiles, band-width ratios, option-implied move, entry cost, breakevens relative to spot and M0's forecast; top 20% by predicted return, trailing window; production widths; walk-forward on out-of-sample quantile features only. |
| N2 | Its control: the same model **without** quantile features. The gap is the value of the distribution |
| N3 | Trade-everything pricing: exits priced for every candidate, to train N1/N2 and to report "trade every event" P&L per arm |
| N4 | Quantile walk-forward reaches back to the earliest year with stored option chains, made precise by section 4's Y0 rule (coverage and training-history floors) |
| F | **Fair-value marks and spread-aware fills**, default for Stage B scoring, selection and the N1/N2 target; mid and harsh still reported. Per leg: Black-Scholes (American adjustment, rates, dividends or a documented approximation) at ORATS `smvVol` (not stored in `option_chains`; re-ingested from cached raw payloads), clamped to [bid, ask]; fill = fair value plus a concession rising with relative spread (and thin size or open interest where available); structure guard on fill-over-fair and on summed spreads over net debit. Form and constants pre-registered, not tuned; calibration from the user's real fills, if provided, stated. Experiment-side only. Missing `smvVol`: a per-expiry smooth smile. Motivation: EXP-105 diagnostics; EXP-133's no-arbitrage caveat |

## 11. Split list (code slices for part 1; none starts until this design merges)

Estimates: non-test code lines / test lines / doc words. Part 2 lists Stage B's slices.

| Slice | Scope | Size | After |
| --- | --- | --- | --- |
| Q1 | `spec.yaml` scaffold, guard, id check, both closure-hash tests, v14 grid and B3 binding | ~150 / ~150 / 60 | this PR |
| Q2 | Sealed panel, dictionary, label reader; admission; Y0 report; earlier folds and parity | ~170 / ~180 / 80 | Q1 |
| Q3a | Quantile fits, walk-forward selection, sort, receipts | ~150 / ~160 / 50 | Q2 |
| Q3b | Calibration and tails for Q and Q_raw | ~110 / ~140 / 30 | Q3a |
| Q3c | Arm S | ~80 / ~90 / 20 | Q3a |
| Q4a | B0, B1, B2 | ~140 / ~160 / 60 | Q2 |
| Q4b | B3: bind M3 predictions, pool, quantiles | ~90 / ~110 / 40 | Q4a |
| Q5 | Metrics, conditional tests, L, bootstraps, pass rule | ~200 / ~220 / 60 | Q3b, Q4b (may start on fixtures) |
| Q6 | Stage A driver, report, 50-ticker smoke with fit timing, runbook | ~130 / ~100 / 100 | Q5 |
