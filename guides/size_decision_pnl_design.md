# SIZE decision-level P&L re-score (design)

Design; S1 (PR #2570) adds the scaffold and guard, later slices add the rest. Contract text lands in the ARCHITECTURE files with the slice that changes a contract (net-zero: both docs are at their word ceiling). Experiment id **EXP-188** (next free: the local ledger ends at EXP-187, no open PR claims it; S1 re-checks).

**Question (user, 2026-11-17).** Short-vol structures have capped loss and set strike width from the forecast, so they may prefer a *median* forecast (MAE-like). Long straddles pay for the tail, so they may prefer a *mean* forecast (RMSE-like). Does decision P&L agree? A SEPARATE experiment: it reads stored SIZE predictions and scores the trading decisions they produce, selects nothing for SIZE and feeds no P&L back into it.

**Constraints, all binding.**
- No edit to any SIZE search module or any file in `source_closure(root, ["experiments/size_panel.py"])`, and no new module imported by that closure. Only new `size_pnl_*` modules and one `EXP-188_*` directory. S1 recomputes the closure hash before and after and requires equality.
- Nothing enters `STRUCTURES` (`engine/structures.py:1414`), the registry or the board; the straddle rule lives in an experiment module and existing factories are only called.
- The account cannot sell calls: short-vol books are the seven put-side structures; the straddle is long call + long put. No call-side short leg anywhere, baselines included.
- Holdouts are never read: `ExperimentHoldouts.classify` (`engine/v2/foundation/experiment_holdouts.py:31`) only labels events. Admission therefore uses the shared population admission and outcome-isolation checks (the research boundary in `engine/v2/research/ARCHITECTURE.md`) first; member and conflicting-sibling events are dropped before any quote or label read, counted by reason. Neither the chain loader nor any raw reader substitutes for it.
- Marks come only from `engine/fills.py`'s alpha ladder (`FillModel.buy/sell`); ambiguous exits are conservative and labelled (section 6).
- Every non-recorded run uses `--no-ledger` (a required CLI flag, as on the distribution and IV CLIs); the recorded run needs a PLANNED ledger row (section 7).

**Standard-tool check.** numpy is the only resampling code. Reused unchanged: `engine.pnl_sim` (`ResidualPool`, `expected_pnl`, `trailing_cutoff`), `engine.forecast_sizing`, `engine.fills`, `engine.structures` and `price_structure`, `experiments.common_v2.load_v2_chain_quotes`, and the cluster bootstrap in `experiments/size_paired_report.py` if its API fits (S8a checks; else a local numpy resampler, justified in that Spec). No standard format is hand-rolled.

## 1. How the forecast is used today (verified at f9db9dab3)

| Use | Where | Mechanism |
| --- | --- | --- |
| Strike geometry of 7 put-side short-vol structures | `engine/forecast_sizing.py` | `width_moneyness = (forecast/100)/k`; `k` is `PLATEAU_CENTRE` (TWIN-P), `FIVE_STRIKE_PEAK` (TWIN-P5) or `SHORT_VOL_OUTER[strategy]` (CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5). A width outside `[WIDTH_MIN, WIDTH_MAX]` or a non-positive/NaN forecast returns `None`: the candidate is declined (a non-trade, P&L 0, the event stays in E), never clipped. |
| Entry gate of the same 7 | `engine/entry_rules.py` (`_short_vol_rule`), `engine/pnl_sim.py`, `engine/v2/scoring/native_entry_rule.py` | Terms: `exp_pnl_sim >= trailing cutoff`, `rel_spread <= MAX_REL_SPREAD`, `mcap_usd >= MCAP_FLOOR`. `exp_pnl_sim` is the mean simulated return over 4,000 draws: `move = max(forecast + err_move, 0)`, `crush = pred_crush + err_crush`, paired residuals from `ResidualPool` (strictly earlier events, deciles of the forecast at that cutoff, whole-pool fallback below `MIN_POOL`), exit legs repriced by Black-Scholes. The cutoff is the top 20% of the prior six months of stored `exp_pnl_sim` (`trailing_cutoff`, at least 100 scores). A missing fact is undetermined, not "no". |
| DYN-SV chooser | `engine/score.py` (`width_over_forecast`, forecast features) | A trained chooser reads forecast-derived features. **Out of scope**: scoring it means retraining it per forecast model (Decision 3). |
| Size | `engine/portfolio.py` (`CAPITAL_PER_TRADE`, whole lots) | Flat dollars per trade; the forecast does not size the position. |
| Long straddle | `engine/structures.py` `straddle_through` (STR-THRU, in `STRUCTURES`, not on the board) | Long ATM call + same-strike put, first post-event expiry; entry last pre-print close, exit first post-print close. EXP-105/-184 gate it with a trained 41-feature GBM plus forecast features (incl. `forecast_edge = pred_abs_move - implied move`, `engine/v2/scoring/native_gate_features.py`), top-20% selection; EXP-185 (T-14 STR-RUNUP) uses a trailing 80th-percentile score cutoff. |

Two facts shape the design: the forecast changes only the **geometry** and the **entry simulation**; and the gate is a *trailing rank* rule, so a pure level shift between a median and a mean forecast is absorbed by the cutoff. What survives is the level's effect on strike width and the cross-sectional ranking.

## 2. Inputs, bound by identity (R1-R6)

Consumes the sealed **SIZE v14** run (`guides/size_search_v14_design.md`, PR #2550) and production `size_v1_4`. Nothing is read until every binding holds.

| Input | Source | Binding |
| --- | --- | --- |
| Inner predictions (2023, 2024) per candidate | stored trial arrays `(pred, y, event_ids)`, lead 1 (the board's horizon; S2a verifies, else refuses). The SIZE panel already admitted these events, so no protected event is in them. This experiment reads only `pred` and `event_ids`; it never uses the stored `y` (labels come from the label tables below) | plan id, registration id, driver seal, candidate id; content hash of the whole stored array object (hashing bytes is not a read of label values); event ids equal the sealed inner population |
| Outer (2025) predictions | v14 refit units only: overall, RMSE and MAE winners, blend components, averaged blend forecasts | slot `[plan_id, 2025, candidate_id]`, `status: completed` |
| Production | `size_v1_4` walk-forward tier-4 `pred_abs_move` | content hash of the stored table; no refit |
| Labels | `abs_move`, `crush_pct_iv30` | read from the label tables by admitted event keys only (a protected event's label is never opened, so physical isolation is by key at the table reader, not by masking a decoded array); a month's labels are read only after that month's decisions are written |
| Crush forecast | production tier-4 `pred_iv_crush_30`, **held fixed for every model** | only the move forecast varies; stated in every table header |
| Chains, exit quotes | the SIZE panel's pinned v2 snapshot | snapshot id; changed bytes refuse `INPUT_CHANGED` |

- **R1.** A model with missing, partial or `failed` stored predictions is excluded with its reason; contrasts needing it are `not_testable`. Nothing is imputed, retried or substituted. Whole-run refusals: unbound or mismatched plan/seal/panel (`EXPERIMENT_IDENTITY_CONFLICT`) and a failed production series.
- **R2.** One process, one heavy job at a time (never beside the v14 run), under `bounded_run`. Work is chunked by quarter; tables are compact numpy arrays. Over-limit refuses `RESOURCE_LIMIT_EXCEEDED`; no partial report.
- **R3.** `pnl_sim.expected_pnl` seeds from `sha256(key|event_date)`; the experiment passes `key = event_id`, **identical for every model and every structure** (the draw depends on the key, date, pool and forecast, never on the structure, so one per-event simulation table reproduces `expected_pnl` for each structure exactly and the cost is not multiplied by seven). Every model draws the same random stream. The residuals drawn still differ because each model's pool differs; the shared stream reduces, not removes, simulation noise between models. Roster order and the bootstrap seed are fixed in the spec.
- **R4.** Stage receipts (`inputs`, `inner`, `tuning`, `outer`) carry content hashes, are atomic and immutable; exact replay is read-only. A changed input under an existing receipt refuses `EXPERIMENT_IDENTITY_CONFLICT`; no repair. Intermediate arrays are disposable scratch under `/tmp`.
- **R5.** `model_id = content_hash({source, candidate_id|blend_id|"size_v1_4", target_transform})`. The roster is sealed in the spec before any read; the spec hash is in the PLANNED row.
- **R6.** Causality by **data dates**, never `computed_at`. Pool rows for an event dated D are events strictly before D whose labels had printed; the cutoff window is `[month_start - 6 months, month_start)`; tuning reads 2023/2024 only; the 2025 stage refuses to start without the immutable `tuning` receipt and freezes decisions **per month**: a month's decisions are written before that month's labels are read, and a decision may use only labels of events that had printed before the month began (the rolling pool and cutoff).

## 3. Pre-registered roster (no P&L-based search)

Fixed before any run; none is chosen from P&L. Winner identities come from the v14 selection record (inner RMSE/MAE/blend). Duplicates collapse to one model listing every role.

| Id | Model | Type |
| --- | --- | --- |
| M0 | production `size_v1_4` | reference |
| M1 | v14 overall winner (blend score) | reference |
| M2 | v14 RMSE winner | mean-type |
| M3 | v14 MAE winner | median-type |
| M4, M5 | blends `size_b1_v1`, `size_b2_v1` | mixed |
| Pair H | HGB, HMOSXLP, lr 0.03, l2 10, max_iter 3000, leaves 15, min_samples_leaf 300: `squared_error` (H-sq, mean) vs `absolute_error` (H-abs, median) | only the loss changes |
| Pair E | spline ElasticNet, HMOSXLP, alpha 0.03, l1_ratio 0.75, knots 6: raw (E-raw, mean) vs `log1p` twin (E-log, median if log residuals are symmetric) | only the target changes |
| Pair N | MLP `chrono15_relu64x32_l2_1`, HMOSXLP: raw (N-raw) vs `log1p` twin (N-log) | only the target changes |

At most 12 distinct models. Pair cells are fixed grid cells in the v14 design (B1's EN, an HGB loss, B1's MLP profile), not score-chosen. **Gap, closed (Decision 1 = a)**: v14 refits only winners and blend components for 2025; the six pair members come from v14's pre-registered `selection.extra_refits` (v14 S2b, PR #2565), not from a refit here. The primary H1/H2 test is the three matched pairs; M2-vs-M3 (differs in more than loss) and the M0/M1/M4/M5 rows are secondary, unadjusted for multiplicity, and never re-select anything.

The v14 log1p caveat applies: a log-target forecast is the median only if log residuals are symmetric. Every table reports `actual_above_forecast_share` per model (about 0.5 supports the median reading) and the mean forecast ratio median-type/mean-type, since geometry reacts to level.

## 4. Per-model residual pool

Reusing production's pool would bias the test: its residuals pair with production's forecast, so another forecast lands in the wrong decile and draws the wrong error. For each model m the experiment builds its own `ResidualPool` (unmodified) from rows `(event_date, pred_m, err_move_m = abs_move - pred_m, err_crush = crush_pct_iv30 - pred_iv_crush_30)`, `dropna` as in `Scorer._residual_pool` (`engine/score.py:3739`), strictly earlier events, the same decile rule and `MIN_POOL`.

- **Common support.** Candidates have stored predictions only from 2023; production has 2018 onward. Primary arm: **every model, production included, uses only events dated 2023-01-01 or later**, so pool size and burn-in are equal. Sensitivity arm (labelled): production with its native full-history pool; the gap shows how much of production's edge is pool length.
- **Burn-in.** An event is undetermined until its pool has `MIN_POOL` rows and its cutoff has 100 prior-window scores, so scoring starts in 2023H2 for all models alike. Events undetermined by burn-in or missing data (thin pool or cutoff history, no simulation, missing spread or mcap) under any roster model leave the **paired population E** for all models (counts per quarter). A per-(model, structure, c) non-trade (width declined, ladder too coarse, negative floor, entry not a debit) is not a burn-in event: it scores P&L 0, is counted by reason (the inner receipt's `non_trades`, at c=1.0 only), and the event stays in E and in every other model's, structure's and c's items. This covers the short-vol book only: a missing fact (no usable entry quote, or a listed strike absent) stays undetermined on either book and drops the event from E (a missing fact is undetermined, not no); the inner receipt counts these separately as `missing_facts`. Inner tuning still keeps such events in its items (they add no trade and no history); only the reported E drops them. A model with fewer than 300 selected inner trades pooled over the seven structures skips tuning (`c` = 1), labelled (section 5).
- **Fast path.** As in EXP-133 (`build.py`), expected P&L is linear in the legs: simulate once per (model, event) the mean exit value of a single put at each listed ladder strike (about 45); every (structure, width) is then a dot product. The runner first checks equivalence against `pnl_sim.expected_pnl` and `price_structure` on a fixed sample and aborts above 1e-9.

## 5. Short-vol scoring and geometry divisors

Per (model, event, structure): forecast, width by `forecast_sizing` (declined if out of range), strikes snapped to the listed ladder in dollars, zero floor re-checked on listed strikes (EXP-133 rule; a negative-floor candidate is dropped, meaning not traded: a non-trade at P&L 0 that keeps its event in E), entry cost at the alpha mark, `exp_pnl_sim` through the model's pool, trailing cutoff on that (model, structure)'s own prior scores, spread and mcap terms exactly as `_short_vol_rule`, then realized exit value at marks. Entry/exit conventions are the board's for each structure; S5 pins them from the factories and refuses if a factory default disagrees.

Two geometry arms, both pre-registered:
- **As-is.** Registered divisors and bounds: what the board does.
- **Re-tuned.** A multiplier `c` on every structure's divisor, grid `{0.5, 0.6, 0.67, 0.75, 0.9, 1.0}` (widened by Amendment 4, section 15), chosen **once per model, pooled over the seven structures** (Decision 5, settled) on 2023/2024 by the primary metric (summed flat-dollar P&L over the seven books) through the full entry rule with the cutoff recomputed per `c`, ties to `c` nearest 1, frozen in the `tuning` receipt and applied unchanged to 2025. A median forecast is lower, hence narrower wings, so as-is partly tests the forecast's *scale*; re-tuning removes scale and asks about *shape*. Why one `c`: live DYN-SV uses one fixed width per structure (forecast divided by a divisor, `engine/forecast_sizing.py`); EXP-133's per-event width search was falsified (it overfit, widened the universe about 2.6x and did worse); EXP-137's chosen widths clustered at 1.3-1.7x forecast and the live divisors are the conservative end. So the re-tune originally asked where in that band each model's forecast should sit; Amendment 4 (section 15) widens the grid to c = 0.5 (width 2x the forecast) through 1.0 (1x), beyond that band; per (model, structure) `c` would add 7x the freedom for no live counterpart.

Primary metric: **flat-dollar P&L per candidate event** in E (non-trades count 0), lots by `CAPITAL_PER_TRADE` whole lots as in `portfolio.py`, alpha 0.5; the alpha grid {0, 0.25, 0.5, 0.75, 1} is reported. Secondary: return per taken trade, trade count, hit rate, share declined by width.

## 6. Long-straddle buy rule (proposed) and marks

STR-THRU's registered gate is a trained GBM (EXP-105, -184), STR-RUNUP's a trailing percentile of a trained score (EXP-185); neither is a bare function of the forecast, and trained gates would need refitting per forecast model. The proposal is the **forecast part of those gates as a bare rule**:

- **R_L (primary).** Buy `straddle_through` when `edge = pred_abs_move_m - implied_move` is at or above the top-20% cutoff of the model's own `edge` over the prior six months (`trailing_cutoff` called on a frame whose `exp_pnl_sim` column holds the `edge` series; at least 100 scores, strictly before the month), within the short-vol universe filters (mcap floor, spread over the two entry legs). `edge` is the existing `forecast_edge`. Flat dollars, same lots rule. Missing implied move or legs, or an entry quote that is crossed, locked or zero-bid: undetermined, not "no".
- **R_L2 (secondary).** Fixed `edge >= 0`, no cutoff. A mean forecast sits above a median one, so this changes *how many* trades it takes, which R_L cannot see. Trade counts are reported beside P&L.

Neither rule fits anything or looks at a P&L.

**Marks and ambiguous exits.** Every leg is priced with `FillModel(alpha).buy/sell`. Per-leg fallback, each step labelled in the row: (1) valid, non-wide, non-zero-bid quote: alpha mark; (2) zero-bid or wide (`FillModel.is_wide`): alpha 0 (worst) for that leg; (3) no usable exit quote, including crossed (`bid > ask`, which `FillModel` refuses; a locked quote, bid equal to ask, is valid and priced normally): the **structure** is marked at its worst-case payoff (zero for the straddle; the defined-risk floor for put structures), labelled `exit_unquoted_worst_case`. Tables report each model's share per rung, so geometry landing on thinly quoted strikes cannot hide; a sensitivity row drops rung 3.

## 7. Hypotheses, pre-registration and judgement

**Stated in `spec.yaml` before any 2025 look:**
- **H1 (short vol):** matched-pair `d = P&L(median-type) - P&L(mean-type) > 0`.
- **H2 (long vol, R_L):** `d < 0`.
- **H3 (secondary, interaction):** `d_short - d_long > 0` in return per candidate event, so the books are comparable. This is the direct reading of the user's question.

**Pre-registration that binds.** S1 scaffolds with `experiments/new_experiment.py` (PLANNED row). `engine/evaluate.py` sets `spec_hash_checked = False` and does not fail when no planned row exists, so the guard would silently no-op. The runner therefore refuses `EXPERIMENT_IDENTITY_CONFLICT` (reason `planned_row_missing`) unless the PLANNED rows number one plus the spec's declared `amendments` and the last has a `spec_hash` equal to the spec's, and the spec's own `preregistered_at` is before the run log's first entry (the check `engine/evaluate.py` makes). The spec carries the roster, contrasts, arms, `c` grid, bootstrap settings and category rules. v14's 2025 forecasts and SIZE's own 2025 RMSE/MAE exist before this runs; the claim is only that **no 2025 P&L** is seen before the spec is fixed, and this PR merges before the v14 run completes.

**Stages.** `inner` (2023/2024: pools, cutoffs, tuning) writes the immutable `tuning` receipt; `outer` (2025) refuses without it. Reports print inner and outer separately; the headline is outer.

**Judgement is by consistency, not a magnitude cutoff.** Units are calendar quarters (2023Q3 to 2025Q4) and years. For each contrast (a pair, per arm and book): sign by quarter (count and share), sign by year, mean `d` with a **paired bootstrap by quarter** (resample quarters) and a **paired bootstrap by ticker** (resample tickers, events kept together), 10,000 resamples, 90% intervals, fixed seed. Both are reported; neither alone decides. Categories, per hypothesis across the three pairs:
- **Clear:** predicted sign in at least 2 of 3 pairs, none clearly opposite; in each used pair, predicted sign in at least 75% of quarters and every full year (2024, 2025; 2023 partial); both intervals exclude 0; and the sign holds in **both** geometry arms (short vol) or both R_L and R_L2 (long vol).
- **Directional:** same pair and year signs, quarter share at least 60%, but an interval includes 0 or the sign fails in one arm.
- **No difference:** neither, and not reversed. **Reversed:** Clear in the opposite direction. **`not_testable`:** a pair with an excluded model.

The 75% and 60% shares are **my judgement calls**, not derived; the report shows raw counts so a reader can disagree. No category depends on a P&L magnitude. Results do not alter any SIZE winner.

## 8. Cost, memory, order, outputs

- **Order.** After the SIZE v14 run completes and its report is collected. Code PRs may land earlier and are smoke-tested on a 50-ticker `--event-ids` population with `--no-ledger`. One heavy job at a time, never beside the v14 run; container is 12 GiB.
- **Cost (estimates; the smoke measures, the report records RSS and wall time).** About 7,000 events, at most 12 models, 4,000 draws, about 45 strikes: roughly 1.5 x 10^10 Black-Scholes evaluations, tens of minutes single-core. Re-tuning adds dot products only. The model-independent ladder table is built once, tens of MB. Target RSS well under 4 GiB.
- **Progress logging (AGENTS.md).** Stage start/end with elapsed, quarter chunks done/total with rate and ETA, exclusions and skips with reasons, RSS at stage boundaries; stderr only.
- **Outputs** (`experiments/EXP-188_*/results/`): the four receipts; `decisions.parquet` (per model, structure, arm, event: forecast, width, `exp_pnl_sim`, cutoff, taken, marks with rung labels, P&L); `REPORT.md` with T1 roster and exclusions, T2 population E by quarter, T3 forecast level and `actual_above_forecast_share`, T4 short vol as-is, T5 short vol re-tuned with chosen `c`, T6 straddle R_L/R_L2 with trade counts, T7 pair contrasts (quarter and year signs, both bootstraps), T8 mark-rung shares, T9 pool-length sensitivity, T10 verdict per hypothesis. Scratch under `/tmp`, removed on success.

## 9. Split list (code PRs, about 150 non-test lines each)

| Slice | Scope | Code lines / test lines / doc words | After |
| --- | --- | --- | --- |
| S1 | Scaffold EXP-188 (`spec.yaml` from this design, PLANNED row), prereg guard that cannot no-op, closure-hash test | ~100 / ~120 / 60 | - |
| S2a | `size_pnl_inputs`: bind v14 plan/seal/panel, resolve roster, refusals; order: check event ids against the sealed population, apply admission, read `pred` and `event_ids` only (never stored `y`); the binding hash covers the array object and the admission result | ~160 / ~180 / 80 | S1, v14 S2c |
| S2b | Production and label readers, holdout drop, lead-1 check, per-model tables | ~140 / ~160 / 40 | S2a |
| S3 | Ladder-quote table over the pinned snapshot, quarter-chunked, progress | ~150 / ~150 / 40 | S1, S2b (admitted keys only) |
| S4 | Per-model pool, linear fast path, equivalence check | ~170 / ~180 / 40 | S2b, S3 |
| S5 | Short-vol scoring: geometry, entry rule, marks and rung labels | ~170 / ~200 / 60 | S4 |
| S6 | Divisor re-tune on inner, `tuning` receipt | ~110 / ~130 / 40 | S5 |
| S7 | Straddle rules R_L / R_L2 | ~110 / ~130 / 40 | S3, S2b |
| S8a | Contrast statistics: signs, two bootstraps, categories | ~140 / ~160 / 60 | - |
| S8b | Report tables T1-T10 | ~170 / ~150 / 40 | S6, S7, S8a |
| S9 | Driver (inner/outer), 50-ticker smoke, runbook, contract lines | ~120 / ~120 / 100 | S8b |

S1 and S8a can start in parallel (S3 waits for S2b's admitted keys); each slice changes behaviour only inside the new experiment.

## 10. Decisions (settled by the user, 2026-11-17)

1. **2025 forecasts for the matched pairs: (a).** The six pair members come from v14's pre-registered `selection.extra_refits` (v14 S2b, PR #2565). A pair member missing from the sealed v14 plan makes its pair `not_testable` (R1); this experiment never refits.
2. **Consistency thresholds: the design defaults.** 75% / 60% quarter shares, 90% intervals (section 7).
3. **Scope.** The DYN-SV chooser and the STR-THRU trained gate are OUT of EXP-188. Pre-registered **conditional follow-up**: if H1 or H2 is judged Clear or Directional, a per-model chooser retrain (and, for long vol, a STR-THRU gate refit) must run BEFORE any board change. No forecast swap reaches the board on per-structure evidence alone, because the chooser and gate were trained on the production forecast's features.
4. **Pool length: the design default.** Common 2023+ support is primary; production's native pool is the labelled sensitivity arm (section 4).
5. **Re-tune breadth: one `c` per model**, pooled over the seven structures (section 5 gives the rationale).

## 11. Addendum

A synthetic-predictor calibration arm (controls and power; never part of the roster or judgement) is designed in `guides/size_pnl_synthetic_calibration_design.md`.

## 12. Amendment: outer (2025) prediction binder (issue #2658)

**Problem.** `outer_predictions_unbound` refuses every non-production model because nothing binds v14's 2025 refit predictions. The `outer-terminal:<key>` record cannot be the source: `_result` writes the 2025 target (`absolute_move_pct`) into every prediction row, so reading it would open protected labels.

**Source (deviation from the issue's wording, safer).** v14 commits `outer-forecasts:<key>` (normalised forecasts, before any target opens: "all predictions and successful fit accounting are durable before any outer target opens") with `key = refit_key(plan, 2025, candidate_id)`. It holds only `event_id`, `horizon_sessions` and `predicted_absolute_move_pct` (plus fields the forecast normaliser already admits; none is a target). The binder reads that slot, never `outer-terminal`, `outer-validation` or `outer-inputs`. Labels keep coming from the S2b label reader with its holdout drop; no 2025 label reaches the binder.

**Module.** `experiments/size_pnl_outer_inputs.py`, one function `bind_outer(plan, read, bound, roster, admit, *, outer_year)`, called by `_prepare` for `--stage outer` only (the inner stage is unchanged). It adds `years[str(outer_year)]` to each `bound` candidate (and to each blend component) using the existing `bind_array` (lead-1 rows, duplicate ids, finiteness, admission check, array-object hash), so `_candidate_series` and `_blend_series` pick the year up unchanged. The model's `binding` hash is recomputed over all years, so the consumed-table hash and the receipts change with the outer binding.

**Identity checks per refit unit** (slots `outer-start`, `outer-execution`, `outer-forecasts`, all keyed by `refit_key(plan, 2025, candidate_id)`):

1. `outer-start.plan_id == plan_id`, `outer_year == 2025`, `candidate.candidate_id == candidate_id`.
2. `outer-execution.status == "succeeded"` (the worker event name; `failed` is a failed refit) and `forecasts_id == content_hash(forecasts)`: the forecasts are the ones the sealed run recorded.
3. Event ids equal the fold's sealed `evaluation_event_ids` (plan `folds[outer.year == 2025].outer`), through `bind_array`'s existing exact-set check. No label-omission logic is needed here: forecasts cover the whole sealed cohort, and the later labels join drops events without a printed label.

**Failure semantics (R1-R6 template).**

| Case | Result |
| --- | --- |
| R1 refit missing for a model (no `outer-forecasts` slot, or no `outer-execution`) | the model is excluded `outer_missing`; its contrasts are `not_testable`; nothing is imputed or retried |
| `outer-execution` is `failed` (a non-required refit) | excluded `outer_failed` |
| blend winner: any component missing or failed | the blend is excluded `component_outer_<reason>`; averaging is only over complete components (existing `_blend_series`) |
| pair member not in `extra_refits` | already `not_in_extra_refits` at the roster; the binder only sees roster models |
| R1 whole-run: `outer-start` identity mismatch, `forecasts_id` hash mismatch, duplicate or non-finite predictions, ids outside the population, a legacy (non per-candidate) plan | `EXPERIMENT_IDENTITY_CONFLICT` with `details.reason` in `outer_identity_mismatch`, `outer_forecasts_hash_mismatch`, `duplicate_event_id`, `nonfinite_prediction`, `event_ids_outside_population`, `outer_plan_unsupported`; no partial run |
| partial cohort (ids a strict subset of the sealed 2025 cohort) | excluded `partial` (same as inner) |
| R2 cost | reads one forecasts record per unit (at most 12 models, a few thousand rows each); no store scan, no chunking needed |
| R3 | no randomness |
| R4 | the outer receipts bind the per-model `binding`, so a changed forecasts record refuses at replay (`EXPERIMENT_IDENTITY_CONFLICT`) |
| R5 | `model_id` unchanged; the roster is already sealed |
| R6 | forecasts are labels-free; the 2025 decision month still precedes its label read in the driver |

**After the binder.** `_unbound_outer` stays as the safety net (a ready model with no 2025 rows still refuses); a model the binder excludes is simply not ready.

**Tests (synthetic records only).** Bound blend and candidate (the 2025 block present, ids/preds equal the record, no `absolute_move_pct` ever touched: the fake `read` raises for any `outer-terminal`/`outer-validation` slot); missing, failed, hash mismatch, wrong plan id, partial and duplicate cases; an outer-stage `_prepare` smoke with fake sources.

**Runbook step (supervisor, after v14 completes and is collected).** `guides/size_pnl_runbook.md` gains "Verify the outer binder": run the 50-ticker smoke `--stage outer --event-ids <ids.json> --no-ledger` against a fresh `/tmp` out-dir and read stderr `[pnl_outer]`: every roster model is `bound` or excluded with a reason, the production and v14 refit counts match `refit_units`, and no `outer_forecasts_hash_mismatch` appears. Only then the full outer run.

## 13. Amendment: real-store `--sources` adapter (issue #2659, first item)

**Problem.** Nothing builds `size_pnl_driver.Sources` for the real store, so EXP-188 cannot run, not even the smoke. `run.py` calls `--sources MODULE:FUNC` with only the parsed args.

**Interface.** `experiments/size_pnl_sources.py`, `build_sources(args) -> Sources`. `run.py` gains one optional argument, `--sources-config <json>` (a regular file of at most 1 MiB, read like `--expected`); the factory refuses `sources_config_invalid` without it. Config keys: `catalog`, `store_root`, `spec`, `plan_ref` (the v14 rerun's spec and sealed plan reference), `snapshot_id` (the pinned v2 snapshot), `crush_snapshot_id` (the second pinned snapshot that holds `iv_crush_targets`) and `production_hash`. Identities come from `args.expected_identities`; nothing else is configured. The catalog opens read-only (`file:...?mode=ro`), the store through `ArtifactStore`, records through `read_record`, the plan by loading the stored object, admission through `population_admit` and `admit_events`, chains through `default_load_index`. No record parsing is re-implemented.

**Sealed v14 records are consumed, not re-derived.** EXP-188 never calls `read_search_plan`, `prepare_plan` or `verify_installation`: they re-derive the plan with the installed code and require the installed source closure to equal v14's, which no later checkout can satisfy. The adapter reads the plan object named by `plan_ref` with `ArtifactStore.read_verified` (bounded by the ref's byte size, at most 64 MiB: `plan_too_large`; hashed against its content hash, a mismatch refuses `plan_hash_mismatch`; unparsable or schema-inconsistent bytes refuse `plan_malformed`), requires a known plan `schema_ref` (`plan_schema_unknown`), `plan["anchor"]["run_id"]` equal to the registration's run id (`plan_anchor_mismatch`), `variant_id` likewise, `plan["implementation"]` (source closure and environment) equal to the registration binding's own (`plan_registration_mismatch`) and `plan_id` equal to the content hash of the rest of the body (`plan_id_mismatch`; both are checks of stored bytes, not re-derivation), and leaves plan, seal and panel identities to `bind_run` against `--expected`. A read failure other than an integrity failure refuses `RESOURCE_UNAVAILABLE` (`plan_unreadable`). Integrity is tamper-evident through content hashes and the operator's expected ids; code identity is deliberately not required. The selection record is read from the store, never recomputed.

**Roster inputs from the sealed plan (no config).** `blend_ids = {M4: the plan blend named size_b1_v1, M5: named size_b2_v1}`. `pair_candidates[role]` is the unique `plan.selection.extra_refits` candidate matching `PAIR_RECIPES[role]` and `PAIR_TRANSFORM[role]` (`_recipe_matches`, `_transform`). Zero or several matches omit the role, and `resolve_roster` marks it `not_testable`.

**Table sources (read at the pinned snapshot; slice 2).**

| Input | Source |
| --- | --- |
| Production `pred_abs_move`, `pred_iv_crush_30` | `tier4_forecasts`, keyed (ticker, event_date), joined to `earnings_events` for `event_id`. `horizon_sessions` is not stored there; the adapter sets it to `LEAD` (1), the lead-horizon forecast. **Supervisor decision (real-store smoke 3):** an admitted event with no forecast row, or a NaN forecast, is not a whole-run refusal: it is left out of the fetch, excluded from the paired population E for every model (the existing common-support rule; `read_production` counts `production_missing` and each model table `no_crush_forecast`) and counted `production_forecast_missing` on stderr. `inf` or an FLT_MAX value in a present forecast still refuses `source_value_invalid`. Join key: (ticker, `event_date`) from `earnings_events`, exact (no session or horizon filter; `tier4_forecasts` has neither). |
| `production_hash`, `production_table_hash` | the `tier4_forecasts` dataset-version `manifest_hash` in the snapshot (`data_snapshot_tables` to `data_dataset_versions`). The expected value is run config. |
| Label `crush_pct_iv30` | `iv_crush_targets` **from the crush snapshot only** (config `crush_snapshot_id`), joined to the main snapshot's events on `event_id`; skipped or NaN rows drop in `read_labels`. The main snapshot has no `iv_crush_targets`, and there is no fallback to it. |
| Crush snapshot | a second pinned snapshot, admitted and bounded exactly like the main one (same caps, resolved read-only, no registration pin) and required to hold `iv_crush_targets`, else `crush_snapshot_invalid`. Measured on the real store: the IV-crush program's snapshots hold `earnings_events` and `iv_crush_targets` only and none shares v14's `earnings_events` version, so an event counts only when its `event_id` has the same ticker, event_date and session in the crush snapshot's `earnings_events`, that row is not a date conflict (flagged, or a repeated ticker and date, as in `experiment_population._canonical_events`), and an `iv_crush_targets` fragment exists for it. Any other event is excluded and counted as `crush_label_missing` (R1); nothing is imputed. The crush table's `manifest_hash` is receipted next to `production_hash` (`crush_table_hash`, bound between inner and outer). |
| Snapshot admission caps | the catalog evidence scans stop at cap+1 (`snapshot_metadata_too_large`): at most 1,048,576 fragments and 512 MiB of evidence rows. Measured on the v14 snapshot (11 tables): 256,680 fragments (`computed_moves` alone 249,177), about 120 MB of `data_fragments` rows and 59 MB of `data_objects` rows, so the first caps (65,536 and 64 MiB, copied from the label-bounds admission) refused the real input; the new caps leave about 4x headroom. |
| Label `abs_move` | **Supervisor default, the user may override:** absolute `computed_moves.realized_move_pct`, the labels v14 was selected on (price_source `computed_moves`). |
| Label availability | `available_as_of_date` of the same `computed_moves` and `iv_crush_targets` rows (a native data date). `read_labels` returns it per label and the driver admits a label to a month's pool only when it is strictly before the month start; this replaces `plan_frame.exit_date + 1`, which can precede it. A label with values but no valid `available_as_of_date` refuses `label_availability_missing`; there is no fallback to `exit_date`. The labels receipt hashes the availability date with each label, so a changed date is detected at replay. |
| Label reads | both label tables are read event-isolated (`event_history.plan_event_targets` plus one scan per event); only events with a single-day fragment (listed from fragment metadata) are planned; an event with no fragment, or a fragment with no row or a NaN value, is excluded from E and counted `move_label_missing` (R1, supervisor decision; nothing is imputed), while a fragment that is not event-isolated still refuses `HOLDOUT_ACCESS_DENIED` from the planner. The same per-event rule holds for `crush_label_missing` and `production_forecast_missing`; a missing table or column still refuses `source_table_missing`, and a run with no event left refuses `empty_population`. Excluded counts by reason are printed to stderr. |
| Facts `pre_iv30`, `mcap_usd` | `feature_panel` event row. Non-finite values refuse, and so does the FLT_MAX sentinel (`abs >= 3e38`); `mcap_usd` must be at least 1e6 (USD, not thousands or millions): a whole-run refusal, because it detects a unit error across the panel, not a bad row (each checked where present). **Supervisor decision (outer-stage refusal on 113 rows):** a non-positive `pre_iv30` or `implied_move` on a row is a missing-value sentinel (ORATS writes exactly 0.0 for micro-caps), not a refusal: `facts` returns NaN for that value and the event is excluded under `facts_missing` exactly like a NaN fact (no separate reason, so the driver, funnel, REPORT and receipts are unchanged; the count is not split out). Never imputed. **Supervisor decision (real-store smoke 4):** an event with no `feature_panel` row, or a NaN fact, is not a refusal: `facts` returns NaN for it and the driver drops it from the paired population E for every model, counted `facts_missing`; only a missing table or column refuses. The driver requests facts only for events still in E after the forecast exclusions. Nothing is imputed. |
| Fact `implied_move` | `feature_panel.or_implied`, the panel's own `implied_move` column (`engine/data/features/panel.py:77`), a pre-print quote; no outcome table is read for facts. |
| `plan_frame` | `event_id`, `ticker`, `event_date`, `session` from `earnings_events`. Entry is the last pre-print close, exit the first post-print close, on the snapshot calendar: AMC entry = event-day close, exit = next session close; BMO entry = prior session close, exit = event-day close. An existing calendar helper is reused. An event whose session is not AMC/BMO or that has no calendar window is dropped and counted on stderr; no surviving event refuses `plan_frame_empty`. `exit_date` dates the model rows only, never label availability. Dropped events are counted by reason on stderr: `session_missing` (null or empty), `session_unknown` (any other value, with the top values listed), `date_missing`, `no_entry_close` and `no_exit_close` (the calendar has no such session); any reason that is not a real calendar gap is a bug to fix. |

**Holdouts.** `admit_events` (`load_population`, `_exclude_holdouts`) drops holdout and conflicting-sibling events before any fetch; `fetch` does not filter. 2025 targets are reachable only through the label reader; from 2025 records the adapter reads predictions and ids only.

**Failure semantics (R1-R6).** Refusals use `EXPERIMENT_IDENTITY_CONFLICT` or `INVALID_EXPERIMENT_SPEC` with `details.reason`.

| Case | Result |
| --- | --- |
| R1 config missing, oversized, unknown key | `sources_config_invalid` |
| R1 snapshot missing or not committed | `snapshot_missing` |
| R1 registration, plan, driver seal or `--expected` mismatch | existing `bind_run` reasons; no receipt |
| R1 a record the roster needs is absent | existing per-model exclusions; a missing seal is `seal_unavailable` |
| hash mismatch (`tier4_forecasts` manifest hash vs `production_hash`) | `production_hash_mismatch` before any row is read |
| a table or column missing from the snapshot, or an admitted event absent from `earnings_events` | `source_table_missing`; no imputation |
| a `computed_moves` label fragment not event-isolated | `HOLDOUT_ACCESS_DENIED` from the planner, unchanged |
| an event with no forecast, no move label, no usable crush label or no facts | excluded from E, counted under exactly one reason: `production_forecast_missing`, `move_label_missing`, `crush_context_conflict` (the crush snapshot's `earnings_events` row is absent, differs in ticker, date or session, or is a date conflict), `crush_label_missing` (context fine, but no fragment, skipped or NaN target) or `facts_missing`; the per-reason counts sum to the dropped total and the inner stage prints the funnel (admitted, per-reason exclusions, final paired population); `empty_population` if none remains |
| the crush snapshot lacks `iv_crush_targets` | `crush_snapshot_invalid` |
| an event with no usable crush label | excluded, counted as `crush_label_missing` |
| R2 cost | each fetch is by sorted event ids, once per window or month; progress to stderr |
| R3, R5 | no randomness; model ids unchanged |
| R4 | receipts already bind the production rows, so a changed snapshot refuses at replay |
| R6 | labels are read per month through `read_labels`; the adapter adds no earlier read |

**Split list (estimates).** Slice 1: config, catalog and store open, `read`, `plan`, `selection`, `admit`, `load_index`, roster inputs, refusals (slice 2 replaced the `source_table_missing` stubs of `fetch`, `facts` and `plan_frame`): about 150 code lines (`size_pnl_sources.py` plus the one-line `run.py` argument), about 200 test lines, about 100 doc words (runbook). Slice 2: `fetch`, `facts`, `plan_frame`, hashes, and the runbook smoke command: about 200 code lines, about 250 test lines, about 200 doc words. Tests use synthetic fixtures only.

**Notes.** `--sources-config` is optional for `run.py` (other `--sources` factories ignore it) but required by `build_sources`. The `tier4_forecasts` row is the only `production_hash` source; `LEAD` makes the reader's `lead_unavailable` guard vacuous for this table, which is accepted because the walk-forward forecast is lead-1 by construction. The config file is read with the hostile-input pattern of `run.py` `_read_input`. Every runbook invocation adds `--sources-config <config.json>`.

## 14. Amendment 1 (2026-11-22): labelled exit-marking sensitivity arm `marks_mid`

**Reason.** User decision 2026-11-22, taken after a diagnostic on inner data only. **What was seen, verbatim:** a 500-ticker 2023-24 inner diagnostic (no 2025 data read) showed that design section 6's per-leg fallback rung 2 (a zero-bid or wide exit leg marked at alpha 0) fires on about 62% of taken short-vol trades and produces negative structure exit values on about 20% of them. Taken-trade mean return on debit was about -7% to -29% per model under the pre-registered marks vs about -14% to +27% with every leg at the alpha mark; EXP-133 marked every leg at the alpha mark.

**Unchanged.** No 2025 data was read for this amendment. The primary marks (section 6), primary metrics, hypotheses H1-H3, the verdict rules and the judgement thresholds, the roster, the tuning (the grid, as later widened by Amendment 4 in section 15, and `c` per model, frozen in the primary tuning receipts) and the T10 verdict are exactly as pre-registered. T10 is computed from the primary marks only.

**The arm.** `marks_mid` changes only the exit valuation of taken trades. Decisions are identical to the primary: same taken set, same `c` from the primary tuning receipt (no re-tuning), no change to entry, selection, cutoff or sizing. Rung-2 legs are marked at the run's alpha, like rung-1 legs; rung 3 (no usable exit quote: structure at its worst-case payoff) is unchanged. It applies to all seven short-vol structures and to the straddle (both rules), wherever the rung-2 rule applies.

**Storage.** Each taken row of `decisions.parquet` carries `exit_value` and `pnl_dollars` (primary) and `exit_value_mid` and `pnl_dollars_mid` (`marks_mid`); non-taken rows hold NaN in both mid columns. The rung label is the primary's.

**Report.** `REPORT.md` repeats the short-vol P&L tables (T4/T5, both arms), the straddle table (T6), the T7 contrasts and the T10 verdict computation on `pnl_dollars_mid`, each titled "Sensitivity (Amendment 1), not the pre-registered verdict". A `decisions.parquet` without the mid columns omits these sections with a not_testable note.

**Receipts.** Inner receipts keep their content apart from the spec hash binding: the spec change necessarily changes `spec_hash`, which the inputs receipts bind, so the amended spec needs a new PLANNED row and a new run. The outer receipt's `decisions_sha256` covers the added columns.

**Registration.** The spec declares the amendment in a top-level `amendments` list. The prereg guard then expects exactly one more PLANNED row than there are amendments, and the last must equal the spec hash (`--check` otherwise refuses `planned_row_ambiguous` or `spec_hash_mismatch`). `--register` appends the amendment's PLANNED row after the original one; the original row is kept.

## 15. Amendment 4 (2026-11-23): wider re-tune grid

**Reason (verbatim).** User decision 2026-11-23. The synthetic positive control X1 failed in every book, because a perfect-move oracle earned less than noisy and shrunk forecasts. A read-only analysis of `decisions_synthetic.parquet` found that under the next-day exit, short-vol P&L rises with width / actual move up to about 1.5-2x for BFLY-P, CTR5 and RAMP7 (TWIN-P is fine near 1x). The registered grid's widest point `c = 0.75` (1.33x the forecast) was chosen by 10 of 12 models, at the grid edge.

**Disclosure.** The 2025 outer results of the pre-registered run, and two earlier interim inspections, were seen before this amendment. The amendment is therefore not blind to 2025; the choice of direction (wider) comes from the synthetic arm, but the reader should weigh the retuned arm accordingly.

**Why the upper points go.** In the pre-registered inner tuning (the same 2023/2024 data and scoring the rerun will use), no model chose 1.1, 1.25 or 1.5. Every model's total at each of those points was below its chosen `c` (0.75 for 10 models, 0.9 for E-log, 1.0 for H-abs). On identical data, removing non-argmax points cannot change any argmax, and the tie rule (nearest 1, then smaller) is unaffected. The grid stays at six points, so tuning time is unchanged. 1.0 stays as the as-is anchor and tie reference.

**Change.** The re-tuned arm's grid becomes `{0.5, 0.6, 0.67, 0.75, 0.9, 1.0}` (`size_pnl_tuning.C_GRID`, `spec.yaml` `short_vol.arms.retuned.c_grid`). The tie rule (nearest 1, then smaller), `MIN_TRADES` (300) and the pooling over the seven structures are unchanged. `WIDTH_MIN` and `WIDTH_MAX` (`engine/forecast_sizing.py`) are unchanged: a width beyond them declines as today (a non-trade at P&L 0) and is counted. Each tuning receipt now also records `declined`, the share of candidate rows declined by width per `c`, so the declined share at the chosen `c` is reported per model.

**Unchanged.** Primary metrics, H1-H3, the verdict rules, the as-is arm and the synthetic arm's own registered settings (it reads no `C_GRID`; the synthetic models are never tuned).

**Registration and rerun.** The spec change changes the spec hash, so a new PLANNED row is required (the supervisor registers it). The tuning receipts change and an inner rerun is planned.

## 16. Amendment 2 (2026-11-22): cash-secured $100,000 account sensitivity

**Reason (user instruction 2026-11-22).** The registered flat $10,000-per-trade sizing lets penny-debit trades take thousands of lots. In the 2025 outer output, VFC_2025-01-29 BFLY-P5 at a $0.015 debit bought 6,666 lots for +$666,600, the top trade for several models, and 42% of the models' top-10 trades had debits under $0.25. The user's real constraint is that every short put must be cash-secured.

**Disclosure.** The supervisor inspected the 2025-01 and 2025-02 outer month outputs of the running pre-registered run before this amendment was written. The amendment adds a report-only sensitivity; it cannot change any registered number.

**Unchanged.** The primary metrics, hypotheses H1-H3, the verdict rules, the roster, the tuning (the grid and `c` per model) and the T10 verdict are exactly as pre-registered. T10 stays on the flat-dollar primary marks. Decisions, the taken set and `pnl_dollars` do not change.

**Driver.** Each TAKEN row of `decisions.parquet` also carries `entry_date` and `exit_date` (the actual plan sessions) and `secured_per_contract`: the sum over traded SELL put legs of `qty x strike x 100`, no spread offset, long puts securing nothing (EXP-134 `margin.secured_per_contract`, called, not copied). Straddle rows hold 0. Non-taken rows hold NaN. Only the outer stage rows and the report change: the inner stage computes nothing new (`size_pnl_shortvol` gains an import of the account module and a `secured` helper that only the outer path calls; the synthetic run never computes it and its decisions carry none of the three columns), and the inner receipt bodies are unchanged.

**Account simulation (report only).** `size_pnl_account` loads EXP-134's `margin.py` by path (as EXP-136 does) and calls `margin.simulate` unchanged. Policy: EXP-136's primary as inherited by EXP-161, except `start_equity` = $100,000 (user): `cap` 0.66 of CURRENT equity, `target_share` 0.25 of headroom, `expensive_first=True`, every short put secured at strike x 100. Inputs: every taken row of `decisions.parquet` (not restricted to the evaluated population E, so the accounts hold every trade the run took; wanted counts can therefore exceed the E-based tables), per row: `entry_cost` and `exit_value` (per share), run under the primary marks and under the Amendment 1 `marks_mid`. Accounts: (a) per model x arm, ONE account trading all seven short-vol structures (same-event structures compete for headroom, expensive first); (b) per model x arm x structure; (c) per model x straddle rule. Per account: wanted, funded, final equity, total return, max drawdown (of the realised equity curve at exit dates) and peak open positions. Pair differences (H, E, N: median-type minus mean-type) and each model minus M0 in final equity (positive = the model did better), per account key. Every table is titled "Sensitivity (Amendment 2), not the pre-registered verdict". No bootstrap interval is computed.

**Judgement call (mine).** `margin.simulate` sizes `target_share` as `target_share x headroom / secured`, which is undefined for secured 0 (it would return infinite contracts). The straddle account (c) therefore passes the straddle's debit per contract (`entry_cost x 100`, the cash a long position needs) as its secured amount to the simulator, so it is capped by the same 66% headroom. The driver column stays 0.

**Receipts.** `spec_hash` is bound by the inner inputs receipt, so declaring this amendment in `spec.yaml` necessarily changes it and an existing inner run cannot replay under the amended spec (as with Amendment 1); the inner receipt bodies are otherwise unchanged. A new PLANNED row is required (supervisor registers it after merge). The outer receipt's `decisions_sha256` covers the added columns. A `decisions.parquet` without them omits the account sections with a not_testable note.
