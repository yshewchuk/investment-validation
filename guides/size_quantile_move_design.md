# Per-event quantile model of the earnings move (design, EXP-190)

**Status: design; code lands slice by slice** (section 11). Experiment **EXP-190** (user, 2026-11-25; Q1 re-checks the ledger id). This revision supersedes PR #2869. **Part 1** (sections 1-5, failure semantics in 7) is the model and Stage A, which does no trading. **Part 2** (section 6) is Stage B: fills, trading arms and success rule. **[D1]**-style tags point to section 10.

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

**Walk-forward range.** Annual folds, refit once per year (each level and configuration) [D4]. Part 2's selection models need earlier out-of-sample quantiles, so the walk-forward runs from **Y0 through 2025**. Y0 is the earliest year such that in every year Y0..2025 at least 50% of admitted panel events have stored entry and exit option chains, and the panel holds at least 3 earlier calendar years of training rows (both thresholds my judgement). **No range is claimed here:** the supervisor is checking chain coverage, and Q2 prints per-year counts and Y0 before any fit. Sealed v14 folds serve 2023-2025; this experiment builds earlier annual folds from sealed panel rows, and Q2 refuses unless its rule reproduces the sealed 2023-2025 training ids. The feature set and the grid values used 2023-2024 (section 5, periods used for selection), so folds through 2024 have causal fits but non-blind design choices; no period is blind (section 1).

**Primary arms** [D3, D7]: one `HistGradientBoostingRegressor(loss="quantile", quantile=q)` per level and fold. **Q** (causally calibrated) and **Q_raw** (uncalibrated) are **both eligible** in Stage A and both carry into Stage B.

**Arm S** (reported; cannot pass; details user, 2026-11-26): median plus scale. Both HGBs use the configuration the grid below chooses for year Y; S runs no selection of its own. An `absolute_error` HGB gives the median. A second, also `absolute_error` (a conditional **median** scale), fits `|y - median|` on **out-of-fold** residuals from year-blocked expanding folds inside the training rows: every training-row year that has an earlier training row gets out-of-fold predictions, from a fit trained only on labels available before that year. Scale is floored, `max(scale, 0.5)`, both when standardising and in `q`; the report MUST print, per year, the share of events where this floor binds (section 4's binding-share report). `z_q` comes from S's own earlier **out-of-sample** standardised residuals (`(y - median)` over the floored scale), labels available before the cutoff, the start of month m: trailing 12 months, refreshed monthly (same as Q), at least 500 events, else unscored (`scale_burn_in`); Y0 is unscored. Raw `q = median + scale * z_q`; it is scored after sort and floor at 0, and the crossing rate reads raw.

**Grid** [D5]: v14's HGB grid, model parameters only (quantile loss replaces the loss axis): `max_leaf_nodes` {7, 15} x `min_samples_leaf` {300, 1000}, with `learning_rate` 0.03, `l2_regularization` 10, `max_iter` 3000 and v14's early stopping (`validation_fraction` 0.1, `n_iter_no_change` 10, `tol` 1e-7, seed 0). Its random split stays inside the training rows, so no later data enters. Q1 refuses if the sealed v14 plan's values differ. This is **4 configurations**, the whole grid, so nothing is trimmed: **28 fits per year** (7 levels x 4), 224 if Y0 were 2018; arm S adds its own. **One configuration is shared by all levels**: for year Y, the lowest event-weighted mean 7-level pinball of raw sorted outputs over out-of-sample years Y0..Y-1, counting only labels available before Y's cutoff. Y0 uses leaves 7, leaf 300. Ties: fewer leaves, then larger leaf. The Q6 smoke measures fit time before the full run.

**Non-crossing** [D6]: sort the seven outputs per event (this never raises an event's summed pinball loss). Q_raw is the sorted output floored at 0.

**Causal conformal calibration** (Q; details user, 2026-11-26). For month m, the pool is out-of-sample events **dated** in the 12 months before m begins whose labels were available before m began; at least 500, else unscored (`calibration_burn_in`). Each pool event keeps the prediction of its **own year's chosen configuration** (what was actually forecast). Errors are **signed**, `y - qhat_q`, on **sorted, unfloored** predictions (never floored first). `delta_q` is their empirical q-quantile (numpy's default linear method, as everywhere here; no coverage guarantee). Add `delta_q` to the unfloored `qhat_q`, sort again, then floor at 0.

**Full distribution** [D8], per arm from its own predictions: inverse CDF linear between knots. Tails (user, 2026-11-26; kept simple and directional, not optimised) draw on a pool **fixed at the start of each month** (the calibration pool's events rule: dated in the prior 12 months, labels available before the month began). `quantile` is numpy linear; a jump at the knots is allowed.

- Above 0.9: over donors with `y > q90`, `d = (y - q90)/max(q90 - q50, 0.5)`; `Q(u) = q90 + s_hi * quantile(d, (u - 0.9)/0.1)`.
- Below 0.1: over donors with `y < q10`, `d = (y - q10)/max(q50 - q10, 0.5)`; `Q(u) = max(0, q10 + s_lo * quantile(d, u/0.1))`.

The event's own widths are `s_hi = max(q90 - q50, 0.5)` and `s_lo = max(q50 - q10, 0.5)`. Under 50 donors in a tail, the **whole event** is unscored for that arm (`tail_pool_thin`). The 0.5-point floor is a **user decision** (2026-11-26: "keep 0.5 floor and report"): the report **MUST** print, per arm and year, the share of events where the 0.5 floor binds, for the event width and for the donor normalisation; for arm S it also prints the scale-floor share above.

**Zero median.** Quantiles flooring to 0 are valid output, but a ratio over `q50` (`W / q50`; part 2's band ratios) is **undetermined** when `q50 <= 1e-9`: counted `zero_median`, never 0/0.

**Constants.** The 500-event minima (calibration, S's `z_q`, `L`, B2's donors), the 50-donor tail minimum and `L`'s `min(20, n // 100)` bins are **pre-registered judgement values**: no derivation or study supports them, and they were fixed before any quantile-model run.

**Tool check.** scikit-learn and numpy suffice; no conformal library. `size_pnl_contrast` bootstraps are reused.

## 5. Stage A: distribution skill (no trading)

**Baselines**, reduced to the same seven levels:

| Id | Baseline | Construction |
| --- | --- | --- |
| **B0 (the bar)** [D9] | production forecast + `ResidualPool` | stored tier-4 `pred_abs_move`; native full-history pool, strictly earlier; quantiles of `max(f + err_move, 0)` over the exact rows `draw` samples: the bucket, or the whole earlier pool when the bucket is under `MIN_POOL`; a refused draw is unscored |
| B1 | M0 with the v14 band | `size_distribution.v1` residual-ratio band, `max(f, 0.05) * Q_q(y / max(pool_f, 0.05))`, re-implemented with a parity test at shared levels. Monthly M0 donor pool, labels before the month, full history; **user decision (2026-11-26):** the v14 bucket rule: v14 quintile buckets of the forecast with the 20-row floor, below which the event is unscored (`pool_too_small`) |
| B2 | option-implied | panel `implied_move` times trailing earlier quantiles of `y / implied_move`: trailing 12 months, at least 500 donors, anchored at month start with labels before it. Donors with null, zero, negative or non-finite `implied_move` are dropped and counted; the event abstains if its own is invalid (user, 2026-11-26). An implied centre with a historical shape, **not** a chain-implied density |
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

**Periods used for selection** (the report repeats this) [D15]: the grid choice for year Y uses years before Y; B3's identity, the feature set and the grid values used 2023-2024; **the Stage A gate, which decides whether Stage B runs, uses 2023H2, 2024 and 2025.** **No feature set, grid value, configuration or baseline identity was selected by scoring 2025** (its monthly calibration, tail and L pools use earlier 2025 labels, causally). **It is not a clean test or holdout:** the Stage A gate reads it (Stage B scores 2025 only after a pass that read 2025 outcomes, which can flatter its 2025 result), and 2025 results shaped the design, EXP-188's and EXP-189's among them. Baselines here are B0-B3; Stage B's control arm E was recommended after its 2025 results were seen. Only the forward period after the data end is independent.

**Settlements 1-9** (supervisor-recommended on PR #2896, Q5a; **the user accepted them as proposed on 2026-11-25**; item 4 also records the user's two scored-share decisions). The user confirmed on 2026-11-26 that the acceptance covers the text as written here.

1. **Spearman bootstrap.** In a used cell with `n` events and rank residuals `rW`, `re`, each event gets `n * rW * re / sqrt(sum(rW^2) * sum(re^2))` (0 in a zero-`W` cell; `L` alike). Their mean is the event-weighted cell correlation. The `size_pnl_contrast` ticker bootstrap runs on them, and on their per-event difference for `W` minus `L`; ranks and residuals stay fixed across resamples. The seed is a parameter, default 188.
2. **Tercile difference** is taken inside each cell (top minus bottom tercile mean `e`) and weighted by cell event count; a zero-residual cell counts as 0. Events are ordered by residual, ties by event id, and split with `np.array_split`. The pooled-event tercile table is report only.
3. **Level-width burn-in** removes an event only from the conditional tests (`W`, `L` and B0's comparison value); it stays in accuracy and coverage.
4. **Minimum scored share: settled by the user 2026-11-25** (closes that point of #2879). `min_scored_share = 0.80` (the default), applied per period and per arm. **Denominator, settled by the user 2026-11-25** (option A of #2919): the share is measured against **all admitted events in the period** (shared population admission). The numerator counts admitted events scored by **both the arm and B0**, so gaps from either side count against it. Below the floor the arm is `not_testable`, reason `scored_share_below_floor`; the share is always reported. User's reason: known input discrepancies may leave some events unscored, and 80% is still solid. Q5b implements it.
5. **Missing data.** A period with no paired event or no usable cell, or a bootstrap with fewer than two groups, makes that arm `not_testable`. A missing quarter does not count towards a quarter requirement, which stays "of 10".
6. **Zero `e` residual.** A cell where `e` varies but its residual is all zero (the correlation would be 0/0) is skipped for both `W` and `L`, under its own count.
7. **`L` bins with tied `q50`.** Edges are numpy linear-method quantiles with duplicates removed; a value on an edge goes to the lower bin, for earlier and current events alike. **Added by the Q5a owner to the supervisor's recommendations, and part of the accepted item:** if ties leave the event's bin with no earlier `y`, the event has no `L` (`level_width_empty_bin`) and is treated as in item 3.
8. **CRPS** uses the section 4 tail rule for every arm. Tails are built upstream: the metric takes each event's piecewise-linear quantile function as `(u, x)` points and integrates it exactly.
9. **Crossing rate** is the share of events where any neighbouring pair of raw quantiles is strictly out of order.

## 6. Stage B: trading (part 2; runs only if Stage A passes)

Rules marked "mine" are judgement calls for the user's review.

### 6.1 Sources, machinery, fills

- **Sources.** `s` is Q (calibrated) or Q_raw (uncalibrated), each with its own tails; an arm suffixed `_s` uses only `s`. Both run whenever Stage A passes, each labelled with its own Stage A verdict. EQ alone runs only for a source that itself passed (section 10).
- **Machinery.** EXP-188's production-exact rules as a library (its design section 18: strikes through `price_structure`, no zero-floor drop, entry and exit sessions, spread and market-cap terms, per-arm trailing cutoffs, exit rungs). No `size_pnl_*` or `size_width_*` file changes. The draw key is `event_id`, shared by all arms.
- **Fill model F** (`guides/fair_value_fill_design.md`) is primary for **selection and scoring**: `debit_F` is the entry cost in `exp_pnl_sim`, in E's and EQ's `k` choice, in lots and in accounts; `debit_F` and `proceeds_F` give every deciding P&L and the selection-model target. The simulated exit inside `exp_pnl_sim` is unchanged (European, zero-rate, no concession). The spec pins the F design as merged at registration and its `fvf_id`. If carry and dividends (PR #2910, slice F7) are merged by then, F includes them; otherwise dividends are zero, as stated.
- **Mid and harsh** are reported on the same F-selected taken set, from existing code: mid is `FillModel(0.5)`, harsh is every leg at alpha 0, entry and exit. F's three sensitivities (unknown liquidity 0, unknown 1, mid-floored; F design section 5) are reported per trade and per family account. None decides or breaks a tie.
- **Guard and statuses** (mine, as EXP-189 Amendment 1 A1.4 proposes in PR #2925): after geometry resolves (a quote that fails resolution is already undetermined, 6.2), a leg that is not `ok` or an F guard failure is a counted **non-trade** for that arm (`f_<status or guard reason>`). It has no score and stays out of cutoff history; the event stays.
- **Conservative exit.** An `unpriceable` exit leg, or exit rung 3, marks the structure at its worst-case payoff (EXP-188 design section 6: zero for the straddle, the defined-risk floor for put structures), labelled `exit_unquoted_worst_case`. F's `exit_fair_mid` rule applies. Negative `proceeds_F` is kept and flagged, not floored (mine). A held trade never leaves.

### 6.2 Universe, gate, populations

A **candidate** is (event, structure), or an event for the straddle. Short vol (seven put-side structures; families: **centre-peaked** CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5; **twin** TWIN-P, TWIN-P5): market cap >= $10B and mean relative spread <= 25% on stored quotes, as production. Straddle: STR-THRU long legs, EXP-188 Amendment 5's universe (market cap >= $1B, `BAD_QUOTE` refused at mid cost, no spread term). No call-side short leg.

Unless stated, an arm **takes** a candidate when its score (`exp_pnl_sim` on `debit_F`; N1 and N2: predicted return; ST and ST0: `P`) is at or above `trailing_cutoff` (6 months strictly before the month, top 20%, at least 100 scores) over **that arm's own** earlier universe scores, per structure.

A row is **determined** for an arm when it has a decision or a counted non-trade (width declined, `ladder_too_coarse`, F status or guard, outside the universe, take rule not met). A fact the arm's rule needs but lacks makes it **undetermined, never "no"**: forecast, quantiles, pool, cutoff history, market cap, spread, implied move (ST, ST0), or, at geometry resolution (before F), an absent strike or unusable entry quote. Each candidate arm has **one paired population**: the rows determined by it and all its controls. Its account and theirs walk only those rows; drops are counted by arm, reason and year. The determined share of admitted candidates is reported, not gated.

### 6.3 Controls and arms

| Id | Role | Strike width | Entry simulation or rule |
| --- | --- | --- | --- |
| **P0** | control: production | `forecast_params(name, M0)` at the pinned commit | `exp_pnl_sim` with M0 and the native full-history `ResidualPool` (as B0) |
| **E** | second control [D11] | per event and structure, the feasible grid `k` maximising `exp_pnl_sim` | EXP-189 arm E with its own pool rule, M0 |
| W_s | candidate | `forecast_params(name, Q_s(tau))` | M0 and P0's pool, on the new geometry |
| LW_s | control for W_s | `forecast_params(name, kappa * M0)` | as W_s |
| D_s | candidate | production | `exp_pnl_sim` with `QuantilePool_s` |
| WD_s, LWD_s | candidate, its control | as W_s, LW_s | `QuantilePool_s` |
| **TF_s** (twin filter; the first draft called it "F") | candidate | production | P0 takes it **and** `R = (q75 - q25)/q50` from `s` is at or below its trailing median [D13] |
| **EQ_s** (arm EQ) | candidate | E's grid, chosen with `QuantilePool_s` | section 10 |
| **N1_s** | candidate | production | selection model with quantile features (6.5) |
| N2_s | control for N1_s | production | the same model without them |
| **ST_s** | candidate | straddle | take on the probability that the move beats the implied move (6.6) |
| ST0 | control for ST_s | straddle | the same rule from B0 |

**E** is recomputed in this run. The spec pins EXP-189's design, every merged amendment (Amendment 1, PR #2925, is open now) and modules. Whatever fill that version uses, Stage B scores E under F, with the F guard a feasibility condition. E and EQ keep EXP-189's grid, ties, no-feasible-`k`, own-score selection, funding and account rules, and the buildability fallback section 10 names, as that version defines it for arm E; SB0 writes the definition into the spec.

**`QuantilePool_s`** exposes the two methods the simulators call, `draw` and `before`. `draw` calls the wrapped pool's `draw` (same M0 forecast, count and random stream), takes each drawn `err_move`'s rank `u = (rank + 0.5)/n` among the draws (ties in draw order; an empty draw passes through) and returns `Qinv_event(u) - forecast`, so the simulated move is the event's own quantile at `u`. `err_crush` is unchanged, keeping the historic move/crush rank dependence [D12]. D, WD and LWD wrap P0's pool; EQ wraps E's. `pnl_sim.expected_pnl` is unmodified; the fast path must match it within 1e-9 or the run aborts.

**LW** isolates the level effect of a different sizing number: `kappa` is the median `Q_s(tau)/M0` over the prior 12 months, strictly earlier events with `M0 > 0`, at least 100. **TF**'s median uses the same 6-month, 100-value window over one `R` per universe event; the other half is printed, not tested. `q50 <= 1e-9` is `zero_median`.

### 6.4 Width quantile, walk-forward [D14]

One `tau` per family and source, from **{p10, p25, p40, p50, p60, p75}**. The objective (mine) is W_s's **family account** (6.8) return at F over the selection window, using only trades that also exit before the scored year: **2024** uses the best on 2023H2; **2025** the best on 2023H2-2024. All six are compared on rows determined at every `tau`. Ties: nearest p50, then smaller. A `tau` with fewer than 100 taken trades (mine) in the window is not eligible; with none eligible, p50. **2023H2** has no earlier window in this rule and uses **p50 for both families** (mine; no result chose it), labelled `tau_default`; the first draft's fixed 0.40 came from 2024-2025 observations and is dropped.

Each `tau` is a complete arm with its own cutoff history; the walk-forward arm takes each period's decisions from that period's `tau`. LW, WD and LWD reuse W's choice (mine). **Every quantile is printed for every year** (W, LW, WD, LWD; both sources and families): returns, trades, declines and the chosen `tau`.

### 6.5 Selection models N1, N2 and trade-everything pricing

**Trade-everything pricing [N3].** Before any selection fit, entry and exit are priced under F (mid and harsh beside it) for **every** universe candidate at production widths, whatever any gate says, from Y0 [N4]. Each other geometry (W and LW at each `tau`, E, EQ) and every straddle is priced alike. The report gives **"trade every event" results per geometry** and year: candidates, non-trades by reason, mean return per trade at F, mid and harsh, and flat-dollar P&L (`CAPITAL_PER_TRADE` whole lots at `debit_F`). P0, D, TF, N1 and N2 share one geometry, so one line. Exits sit in a separate table whose reader releases a row to a fit or choice only if its exit date is before that fold's or period's cutoff.

**Model [N1].** One HGB per **family** per source: centre-peaked and twin, the family's structures pooled with a categorical structure id. **Target:** the candidate's return on debit under F, `(proceeds_F - debit_F)/debit_F`, conservative exits included, unclipped; `squared_error`, since the gate ranks expected return (mine). Non-trades are not training rows.

| Block | Features |
| --- | --- |
| Both N1 and N2 | structure id; M0's forecast [D1]; `implied_move`; M0 / `implied_move`; `debit_F / spot` (entry cost); breakevens: the distance from spot, as a share of spot, of the nearest and farthest expiry prices where the resolved structure's payoff equals `debit_F` (none: NaN) |
| **N1 only** | the seven quantiles of `s`; each over `implied_move`; band-width ratios `(q75 - q25)/q50` and `(q90 - q10)/q50` [D13] |

A missing `implied_move` or breakeven, or a zero denominator (including `zero_median`), leaves only that feature NaN; the row stays. **Quantile features are out-of-sample only:** an event in year Y takes the walk-forward output of the Y fold (Q's calibration uses earlier labels), never an in-sample fit. A row without `s` output is in neither model, so **N2_s trains and scores on exactly N1_s's rows**.

**Windows.** One fit per year, on all rows from Y0 whose exit date is before 1 January of that year (expanding; mine), from 2023 through 2025; 2023 predictions give 2023H2 its cutoff history. **One fixed configuration, no tuning** (mine): `max_leaf_nodes` 7, `min_samples_leaf` 300, `learning_rate` 0.03, `l2_regularization` 10, `max_iter` 300, early stopping off (a random split would separate one event's structures), seed 0. Under 1,000 training rows for a family: that year is undetermined (`selection_burn_in`).

**Rule.** Take when the predicted return is in the top 20% of that model's own trailing predictions (6.2's window, per structure) and the universe terms hold. A cutoff window can mix two annual fits. **N1 minus N2** is the value of the distribution for selection, reported per family and year.

### 6.6 Straddle: does the move beat the implied move

This is the real straddle test, moved from EXP-188 (user, 2026-11-25). For each event, `P = Pr(|move| > implied_move)` is read from the event's **full distribution** (section 4: knots and tails) as one minus the largest `u` whose inverse CDF is at or below `implied_move` (none: 1). **ST_s** takes the straddle when `P` is in the top 20% of its own trailing universe values (6.2's window). **ST0** is the same rule on B0's distribution, built as for CRPS (section 5, settlement 8) from B0's own earlier predictions, so only the distribution differs. `implied_move` is the panel's, in percent points; missing or non-positive is undetermined.

Reported for ST_s and ST0 on shared events, by year (descriptive; the account rule decides): the frequency of `y > implied_move` by fifth of `P`; the Brier score of `P` beside that of the earlier base rate; mean F return of taken, untaken and all straddles. E has no straddle form. `G_PROD`, where its EXP-188 artifacts bind, is re-scored under F for 2025 and labelled "not production-faithful"; otherwise `not_available`.

### 6.7 Outputs

By arm, structure, family and year: candidates, determined share, trades, declines by reason, mean P&L per candidate and per trade at F, mid and harsh, F sensitivities, breakeven alpha and accounts. Quarter and ticker bootstraps (`size_pnl_contrast`, 10,000 resamples, 90% intervals, seed 188) of each candidate minus each control in per-candidate return are reported, not gated. Also F's diagnostics, the 6.4 table, N1 minus N2, **EQ minus E** and 6.6. F's entry-date coverage prints **before any outcome is read**.

### 6.8 Accounts

The EXP-188 / EXP-189 policy: **$200,000 cash-secured**; EXP-134's `margin.simulate` unchanged through the existing account module: cap 0.66 of current equity, `target_share` 0.25, expensive first, every short put secured at strike x 100; a straddle secured at its debit x 100. Entry cost `debit_F`, exit value `proceeds_F`. Accounts: each structure; centre-peaked five; twin pair; all seven (reported); straddle. Verdicts use the **family** account (centre-peaked, twin or straddle). Each period (2023H2, 2024, 2025) starts a fresh account. Mid and harsh accounts walk the same taken set at their own costs.

**Breakeven alpha** reprices the arm's F-selected taken set on the alpha ladder {0, 0.25, 0.5, 0.75, 1}, entry and exit, with `engine.evaluate.breakeven_alpha_from_sweep`. With no crossing it returns None, which the caller maps to 0.0 if always positive and 1.0 if always negative. It tests that set's fill robustness, not its F return. At a high alpha a repriced debit can be non-positive; that trade leaves that alpha's mean, counted.

### 6.9 Success rule and multiplicity [D15]

W, D, WD, N1 and EQ are judged per family; TF on twin; ST on straddle. LW, LWD, N2, ST0, P0 and E are never candidates.

| Candidate | Controls it must beat |
| --- | --- |
| W_s | P0, E, LW_s |
| D_s, EQ_s, TF_s | P0, E |
| WD_s | P0, E, LWD_s |
| N1_s | P0, E, N2_s |
| ST_s | ST0 |

A candidate **passes** iff all hold:

1. its family account return at F is **strictly** above every listed control's, in **each** of 2023H2, 2024 and 2025;
2. its 2025 breakeven alpha is at most P0's + 0.05 **and** at most E's + 0.05 (straddle: at most ST0's + 0.05);
3. short vol only, comparing each stand-alone structure account with P0's at F in 2025: at most 1 of the five centre-peaked accounts loses; neither twin account loses.

Requiring E as well as P0 (items 1 and 2) is my reading of "second control"; the margin and guard counts are mine. Results against P0 alone are printed. No trades in a period fails; an uncomputable comparison is `not_testable`.

**Multiplicity.** Per source: W, D, WD, N1 and EQ on two families, plus TF and ST, **12 verdicts**; both sources **24** (fewer if one gets no EQ). **No correction.** They are strongly correlated, not 24 independent tries, but chance passes are plausible, so the report lists all together. A pass is a **proposal**: any board change is a separate user decision, with the DYN-SV chooser review.

**Periods used for selection** (repeated in the report): `tau` as in 6.4; N1/N2 fit on earlier exits with a fixed configuration; cutoffs, `kappa` and TF's median trail causally; E and EQ choose `k` without realised outcomes. **2024 is not clean** (features, grid and `tau` used earlier results). No feature set, grid value, configuration, baseline identity, `tau` or selection-model setting was **selected by scoring 2025** (baselines are section 5's; control arm E was recommended after its 2025 results were seen); only causal trailing statistics read earlier 2025 data. **2025 is not a clean test or holdout:** the Stage A gate reads it, and 2025 results shaped the design (EXP-188 and EXP-189 results shaped the arms). Only the forward period is independent.

### 6.10 Stage B failure semantics (R1-R6)

Planned requirements; no Stage B code exists yet.

| Case | Result |
| --- | --- |
| R1: no bound Stage A pass receipt, changed spec or no PLANNED row | refuse (`stage_a_not_passed`, `spec_hash_mismatch`, `planned_row_missing`) |
| R1 whole run: panel, folds, M0, quantile receipts, snapshot, F tables or pinned EXP-189 version do not bind; F's own R1 refusals | `EXPERIMENT_IDENTITY_CONFLICT` with `details.reason`; no partial report, nothing rebuilt or imputed |
| R1 per row: quantiles in burn-in, thin pool or cutoff history, missing fact | undetermined (6.2), counted by reason; never "no" |
| Thin `kappa` or TF history; `zero_median` in TF's `R`; `selection_burn_in` | that arm's row undetermined, counted |
| Entry: width declined, ladder too coarse, F status or guard, no feasible `k` | non-trade at P&L 0, counted; the event stays |
| Exit unusable after entry | never a non-trade: 6.1's flagged conservative or `exit_fair_mid` exit |
| A selection fit fails | that year, family and source: N1 and N2 `fit_failed`, no retry; verdict `not_testable`; other arms run |
| `G_PROD` artifacts absent | `not_available`; no verdict depends on it |
| R2 cost | one heavy job, `bounded_run`, quarter chunks; each quote row priced once per chunk; over section 7's memory limit refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report |
| R3 | fixed seeds and orders; results do not depend on which arm asked first |
| R4 | `trade_all`, `selection`, `tau` and decision receipts: atomic, immutable, bound to inputs and `fvf_id`; replay is read-only; a changed input refuses |
| R5 | arm ids hash arm, source, family, configuration, spec and `fvf_id` |
| R6 causality | data dates only; fits and `tau` read exits dated before their cutoff; a month's decisions are written before its labels or exits are scored for it; trailing statistics use strictly earlier months |

Progress logging follows AGENTS.md.

### 6.11 Registration and Stage B limits

Under section 9's single `spec.yaml`, slice SB0 adds this section's fixed rules (arms, `tau` set and rule, features and configuration, fill pins, accounts, success rule) **before** the supervisor registers the **new PLANNED row** with the full spec hash. A later spec change needs an amendment and another row. No agent writes `experiments/LEDGER.csv`; non-recorded runs use `--no-ledger`. A 50-ticker smoke records time and peak RSS before the full run and tunes nothing.

- The simulated exit is more optimistic than F's (F design section 11); width searches (E, EQ, `tau`) may favour that gap.
- Years before 2023 have no stored size or open interest, and unverified vol and rate coverage, so N1/N2's training mix changes by era.
- Candidates and E can also differ in residual pool, so their gap is not one effect.
- Correlated structures, 2024's `tau` from half a year, and one 2025 are weak evidence.

## 7. Failure semantics (R1-R6)

| Case | Result |
| --- | --- |
| R1 panel, folds, dictionary, M0 or M3 series do not bind | refuse `EXPERIMENT_IDENTITY_CONFLICT`; nothing rebuilt or imputed. A merely absent M3 year is B3 `not_testable` for that year |
| A fit fails | that fold's arm is `fit_failed`, no retry (for Q's walk, the whole year, and the walk stops: below). Q_raw's fits failing makes Stage A `not_testable`; a calibration failure affects Q only |
| Burn-in (except level-width, below), thin tail pool, non-finite output | unscored (an abstention, not "no"), counted; leaves the paired population |
| B1 `pool_too_small`; B2 under 500 donors or invalid own `implied_move` | that baseline is unscored for the event, counted; its common-subset tables drop it (section 5) |
| B2 donor with null, zero, negative or non-finite `implied_move` | dropped from the pool and counted |
| Level-width burn-in or empty bin | no `L`, counted; leaves only the conditional tests (section 5, settlements 3 and 7) |
| `zero_median` | ratio undetermined, counted; quantiles stay valid |
| R2 cost | quarter or year chunks; over the memory limit below refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report |
| R3, R4 | fixed seeds; receipts atomic, immutable, bound to inputs; a changed input refuses |
| R5 | arm ids hash the source, level set and configuration |
| R6 causality | fits, selection, calibration, tail, level-width and baseline pools use data available before the fold or month; a month's predictions are written before its labels are read |

**Memory limit** (user decision, 2026-11-26; the run's peak is not yet measured). One heavy process on a 13 GiB box, which must be wrapped whole in `experiments.memory_guard.rss_watchdog(8 * 2**30)`: its own VmRSS, sampled every 50 ms, above **8 GiB** raises `MemoryError`, which becomes `RESOURCE_LIMIT_EXCEEDED`. No partial report is published; complete fit receipts stay (R4). The outer `bounded_run --heavy --max-rss-gb 9` is a backstop above it whose kill gives no typed refusal, and sampling can overshoot. The watchdog's interrupt also takes effect only when Python regains control, so a long native call can run past 8 GiB. Nothing installs the watchdog yet: the real-data entry point (#3047, not built) must; the library driver does not. There is no wall-clock limit. Receipt byte caps are separate. Raising the limit after registration needs an amendment.

**A failed quantile-fit year** (Q and Q_raw; arm S, whose failed year does not stop later ones, and Stage B's selection fits, 6.10, differ; user decision, 2026-11-26). One failed fit (any level or configuration) marks the whole year `fit_failed`. The Q walk runs years in order and **stops at the first such year**: later years are neither selected nor fitted, so no grid selection reads a failed year, and none is skipped or substituted. Stage A is then `not_testable`, even if that year precedes the judged periods. Each receipt is published whole or not at all; a rerun replays published ones, `fit_failed` included.

Progress logging follows AGENTS.md.

## 8. Limits

- Q refits yearly, production monthly; monthly calibration closes only part of that gap, and Q_raw closes none.
- Q trains on the v14 panel, M0 on production's history, so a result cannot separate model from data.
- B3 shows the effect of features and point model together, not of features alone.
- A Stage A pass need not move P&L (EXP-115).

## 9. Registration

One `spec.yaml` holds every fixed rule of **both parts**. Its PLANNED row, carrying the spec hash (supervisor registers), and a guard that cannot no-op come **before any fit on real data, and only after part 2 merges**, so no Stage B rule is chosen after seeing Stage A. Part 1 code slices may start earlier on fixtures. Stages: `fit` (receipts), `stage_a` (verdict receipt), `stage_b`. Contract lines land in the `ARCHITECTURE.md` files with the slices that change them.

**Order.** Part 2 has merged, but the spec still says `pending_part_2`, so the guard refuses `stage_b_pending` until SB0 writes section 6's rules into the spec and marks `stage_b` settled. Then: SB0, PLANNED row, 50-ticker smoke (it fits real data), full run. **Other real-data reads** (user decision, 2026-11-26)**:** before the row, a run may bind sealed records, count the panel and chains, print Y0 and check training-id parity (including the label reads these need), but may not fit a model on real data, or compute any arm's or baseline's score against realised moves (showing v14's stored metrics is allowed).

## 10. Decisions (settled by the user, 2026-11-25)

"Default" is the rule proposed in #2869. Parameters marked "mine" in either part are judgement calls inside these decisions.

Rows record the decisions as taken. Section 6 defines their Stage B terms (F: 6.1; D11-D13, arm E and the twin filter TF: 6.3; D14: 6.4; N1-N3: 6.5) and gives the operative rule: N1's quantiles are its source's (Q_raw's are uncalibrated), and N3's "per arm" P&L is per geometry.

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
| D15 | Default rule; the report states the periods used for any selection. **Revised by the user, 2026-11-26** (replaces "2025 is the clean test"): no feature set, grid value, configuration or baseline identity was selected by scoring 2025; it is not a clean test or holdout, since the Stage A gate reads it and 2025 results shaped the design |
| N1 | Learned **selection model per family** (one centre-peaked, one twin): HGB predicting each candidate trade's return under F, from calibrated quantiles, band-width ratios, option-implied move, entry cost, breakevens relative to spot and M0's forecast; top 20% by predicted return, trailing window; production widths; walk-forward on out-of-sample quantile features only. |
| N2 | Its control: the same model **without** quantile features. The gap is the value of the distribution |
| N3 | Trade-everything pricing: exits priced for every candidate, to train N1/N2 and to report "trade every event" P&L per arm |
| N4 | Quantile walk-forward reaches back to the earliest year with stored option chains, made precise by section 4's Y0 rule (coverage and training-history floors) |
| F | **Fair-value marks and spread-aware fills**, default for Stage B scoring, selection and the N1/N2 target; mid and harsh still reported. Per leg: Black-Scholes (American adjustment, rates, dividends or a documented approximation) at ORATS `smvVol` (not stored in `option_chains`; re-ingested from cached raw payloads), clamped to [bid, ask]; fill = fair value plus a concession rising with relative spread (and thin size or open interest where available); structure guard on fill-over-fair and on summed spreads over net debit. Form and constants pre-registered, not tuned; calibration from the user's real fills, if provided, stated. Experiment-side only. Missing `smvVol`: a per-expiry smooth smile. Motivation: EXP-105 diagnostics; EXP-133's no-arbitrage caveat |
| EQ | New Stage B arm: arm E simulated from the quantile distribution (below) |

**Arm EQ** (settled by the user, 2026-11-25). Per event and structure, take the `k` in arm E's grid {0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0} maximising expected P&L (`exp_pnl_sim`), with moves from the event's own calibrated quantile distribution (section 4's seven levels and tails; for Q_raw, its uncalibrated one), **not** M0 plus `ResidualPool`. Under D12 the pool's paired draws supply only move ranks and crush errors (my reading). F prices selection and scoring, for E and EQ alike. All else is arm E's, from EXP-189's design and amendments at a version Stage B's spec pins: buildability fallback, top-20% trailing selection (on EQ's own earlier scores), funding and accounts.

**EQ minus E**, on events both determine (drops counted), is the distribution's value for width choice, level and shape together. If both Q and Q_raw pass Stage A, EQ runs for each, both counted in the uncorrected multiplicity; if one passes, only for it (my reading). The Stage A gate is unchanged. Report per year and structure at F, mid and harsh, like other arms.

**2024 and 2025.** 2024 is **not** clean (features and grid used 2023-2024). None of EQ's width, selection or model parameters is selected by scoring 2025 (`k` uses no realised outcome). **2025 is not a clean test or holdout** (D15 as revised 2026-11-26): the Stage A gate reads it, 2025 results shaped the design, and E was recommended having seen it.

## 11. Split list (code slices; none starts until its part of this design merges)

Estimates: non-test code lines / test lines / doc words. Part 1:

| Slice | Scope | Size | After |
| --- | --- | --- | --- |
| Q1 | `spec.yaml` scaffold, guard, id check, both closure-hash tests, v14 grid and B3 binding | ~150 / ~150 / 60 | this PR |
| Q2 | Sealed panel, dictionary, label reader; admission; Y0 report; earlier folds and parity | ~170 / ~180 / 80 | Q1 |
| Q3a | Quantile fits, walk-forward selection, sort, receipts | ~150 / ~160 / 50 | Q2 |
| Q3b | Calibration and tails for Q and Q_raw | ~110 / ~140 / 30 | Q3a |
| Q3c | Arm S | ~80 / ~90 / 20 | Q3a |
| Q4a | B0, B1, B2 | ~140 / ~160 / 60 | Q2 |
| Q4b | B3: bind M3 predictions, pool, quantiles | ~90 / ~110 / 40 | Q4a |
| Q5a | Per-event and per-cell metrics: pinball, coverage, CRPS, crossing, `L`, cell scores | ~240 / ~220 / 60 | Q3b, Q4b (may start on fixtures) |
| Q5b | Arm tables, bootstraps, scored share, pass rule | ~280 / ~250 / 40 | Q5a |
| Q6 | Stage A driver, report, 50-ticker smoke with fit timing, runbook | ~130 / ~100 / 100 | Q5b |

Part 2 (`SB` ids; B0-B3 are section 5's baselines). **SB1-SB9b are written only after a Stage A pass.**

| Slice | Scope | Size | After |
| --- | --- | --- | --- |
| SB0 | Stage B rules in `spec.yaml`, with spec tests | ~40 / ~80 / 30 | part 2's design PR, Q1; before the PLANNED row |
| SB1 | F adapter: bound loads, leg tables, entry guard, exits, mid and harsh views (imports EXP-189's seam where it fits) | ~150 / ~170 / 40 | A passes, F4, F5 (F7 when 6.1 requires it) |
| SB2 | Trade-everything pricing, causal exit reader, receipts | ~150 / ~170 / 40 | SB1 |
| SB3 | `QuantilePool`, fast-path equivalence | ~110 / ~140 / 40 | A passes |
| SB4 | Universe, cutoffs, row statuses, P0, D, TF | ~160 / ~170 / 40 | SB2, SB3 |
| SB5 | W, LW, WD, LWD, `tau` sweep and walk-forward choice | ~160 / ~180 / 40 | SB4 |
| SB6 | E and EQ on pinned EXP-189 modules | ~140 / ~160 / 30 | SB4, those modules merged |
| SB7 | N1/N2 features, annual fits, selection | ~170 / ~190 / 40 | SB4 |
| SB8 | ST, ST0, probability diagnostics, `G_PROD` | ~140 / ~150 / 30 | SB1-SB3 |
| SB9a | Paired populations, accounts, breakeven, bootstraps, success table | ~170 / ~180 / 40 | SB5-SB8 |
| SB9b | Stage B driver, report, 50-ticker smoke, runbook | ~150 / ~100 / 100 | SB9a |
