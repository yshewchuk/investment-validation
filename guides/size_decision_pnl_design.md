# SIZE decision-level P&L re-score (design)

Design only; no code. Contract text lands in the ARCHITECTURE files with the slice that changes a contract (net-zero: both docs are at their word ceiling). Experiment id **EXP-188** (next free: the local ledger ends at EXP-187, no open PR claims it; S1 re-checks).

**Question (user, 2026-11-17).** Short-vol structures have capped loss and set strike width from the forecast, so they may prefer a *median* forecast (MAE-like). Long straddles pay for the tail, so they may prefer a *mean* forecast (RMSE-like). Does decision P&L agree? A SEPARATE experiment: it reads stored SIZE predictions and scores the trading decisions they produce, selects nothing for SIZE and feeds no P&L back into it.

**Constraints, all binding.**
- No edit to any SIZE search module or any file in `source_closure(root, ["experiments/size_panel.py"])`, and no new module imported by that closure. Only new `size_pnl_*` modules and one `EXP-188_*` directory. S1 recomputes the closure hash before and after and requires equality.
- Nothing enters `STRUCTURES` (`engine/structures.py:1414`), the registry or the board; the straddle rule lives in an experiment module and existing factories are only called.
- The account cannot sell calls: short-vol books are the seven put-side structures; the straddle is long call + long put. No call-side short leg anywhere, baselines included.
- Holdouts are never read: `ExperimentHoldouts.classify` (`engine/v2/foundation/experiment_holdouts.py:31`) drops member events at the reader before any quote or label read, counted by reason.
- Marks come only from `engine/fills.py`'s alpha ladder (`FillModel.buy/sell`); ambiguous exits are conservative and labelled (section 6).
- Every non-recorded run uses `--no-ledger` (a required CLI flag, as on the distribution and IV CLIs); the recorded run needs a PLANNED ledger row (section 7).

**Standard-tool check.** numpy is the only resampling code. Reused unchanged: `engine.pnl_sim` (`ResidualPool`, `expected_pnl`, `trailing_cutoff`), `engine.forecast_sizing`, `engine.fills`, `engine.structures` and `price_structure`, `experiments.common_v2.load_v2_chain_quotes`, and the cluster bootstrap in `experiments/size_paired_report.py` if its API fits (S8a checks; else a local numpy resampler, justified in that Spec). No standard format is hand-rolled.

## 1. How the forecast is used today (verified at f9db9dab3)

| Use | Where | Mechanism |
| --- | --- | --- |
| Strike geometry of 7 put-side short-vol structures | `engine/forecast_sizing.py` | `width_moneyness = (forecast/100)/k`; `k` is `PLATEAU_CENTRE` (TWIN-P), `FIVE_STRIKE_PEAK` (TWIN-P5) or `SHORT_VOL_OUTER[strategy]` (CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5). A width outside `[WIDTH_MIN, WIDTH_MAX]` or a non-positive/NaN forecast returns `None`: the event is dropped, never clipped. |
| Entry gate of the same 7 | `engine/entry_rules.py` (`_short_vol_rule`), `engine/pnl_sim.py`, `engine/v2/scoring/native_entry_rule.py` | Terms: `exp_pnl_sim >= trailing cutoff`, `rel_spread <= MAX_REL_SPREAD`, `mcap_usd >= MCAP_FLOOR`. `exp_pnl_sim` is the mean simulated return over 4,000 draws: `move = max(forecast + err_move, 0)`, `crush = pred_crush + err_crush`, paired residuals from `ResidualPool` (strictly earlier events, deciles of the forecast at that cutoff, whole-pool fallback below `MIN_POOL`), exit legs repriced by Black-Scholes. The cutoff is the top 20% of the prior six months of stored `exp_pnl_sim` (`trailing_cutoff`, at least 100 scores). A missing fact is undetermined, not "no". |
| DYN-SV chooser | `engine/score.py` (`width_over_forecast`, forecast features) | A trained chooser reads forecast-derived features. **Out of scope**: scoring it means retraining it per forecast model (Decision 3). |
| Size | `engine/portfolio.py` (`CAPITAL_PER_TRADE`, whole lots) | Flat dollars per trade; the forecast does not size the position. |
| Long straddle | `engine/structures.py` `straddle_through` (STR-THRU, in `STRUCTURES`, not on the board) | Long ATM call + same-strike put, first post-event expiry; entry last pre-print close, exit first post-print close. EXP-105/-184 gate it with a trained 41-feature GBM plus forecast features (incl. `forecast_edge = pred_abs_move - implied move`, `engine/v2/scoring/native_gate_features.py`), top-20% selection; EXP-185 (T-14 STR-RUNUP) uses a trailing 80th-percentile score cutoff. |

Two facts shape the design: the forecast changes only the **geometry** and the **entry simulation**; and the gate is a *trailing rank* rule, so a pure level shift between a median and a mean forecast is absorbed by the cutoff. What survives is the level's effect on strike width and the cross-sectional ranking.

## 2. Inputs, bound by identity (R1-R6)

Consumes the sealed **SIZE v14** run (`guides/size_search_v14_design.md`, PR #2550) and production `size_v1_4`. Nothing is read until every binding holds.

| Input | Source | Binding |
| --- | --- | --- |
| Inner predictions (2023, 2024) per candidate | stored trial arrays `(pred, y, event_ids)`, lead 1 (the board's horizon; S2a verifies, else refuses) | plan id, registration id, driver seal, candidate id; content hash of each array set; event ids equal the sealed inner population |
| Outer (2025) predictions | v14 refit units only: overall, RMSE and MAE winners, blend components, averaged blend forecasts | slot `[plan_id, 2025, candidate_id]`, `status: completed` |
| Production | `size_v1_4` walk-forward tier-4 `pred_abs_move` | content hash of the stored table; no refit |
| Labels | `abs_move`, `crush_pct_iv30` | joined by event id; read only after a stage's decisions are frozen |
| Crush forecast | production tier-4 `pred_iv_crush_30`, **held fixed for every model** | only the move forecast varies; stated in every table header |
| Chains, exit quotes | the SIZE panel's pinned v2 snapshot | snapshot id; changed bytes refuse `INPUT_CHANGED` |

- **R1.** A model with missing, partial or `failed` stored predictions is excluded with its reason; contrasts needing it are `not_testable`. Nothing is imputed, retried or substituted. Whole-run refusals: unbound or mismatched plan/seal/panel (`EXPERIMENT_IDENTITY_CONFLICT`) and a failed production series.
- **R2.** One process, one heavy job at a time (never beside the v14 run), under `bounded_run`. Work is chunked by quarter; tables are compact numpy arrays. Over-limit refuses `RESOURCE_LIMIT_EXCEEDED`; no partial report.
- **R3.** `pnl_sim.expected_pnl` seeds from `sha256(key|event_date)`; the experiment passes `key = f"{structure}|{event_id}"`, **identical for every model**, so every model draws the same random stream. The residuals drawn still differ because each model's pool differs; the shared stream reduces, not removes, simulation noise between models. Roster order and the bootstrap seed are fixed in the spec.
- **R4.** Stage receipts (`inputs`, `inner`, `tuning`, `outer`) carry content hashes, are atomic and immutable; exact replay is read-only. A changed input under an existing receipt refuses `EXPERIMENT_IDENTITY_CONFLICT`; no repair. Intermediate arrays are disposable scratch under `/tmp`.
- **R5.** `model_id = content_hash({source, candidate_id|blend_id|"size_v1_4", target_transform})`. The roster is sealed in the spec before any read; the spec hash is in the PLANNED row.
- **R6.** Causality by **data dates**, never `computed_at`. Pool rows for an event dated D are events strictly before D whose labels had printed; the cutoff window is `[month_start - 6 months, month_start)`; tuning reads 2023/2024 only; the 2025 stage refuses to start without the immutable `tuning` receipt and reads no 2025 label until its decisions are written.

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

At most 12 distinct models. Pair cells are fixed grid cells in the v14 design (B1's EN, an HGB loss, B1's MLP profile), not score-chosen. **Gap**: v14 refits only winners and blend components for 2025, so the pairs have no stored 2025 forecast unless v14 adds them (Decision 1). The primary H1/H2 test is the three matched pairs; M2-vs-M3 (differs in more than loss) and the M0/M1/M4/M5 rows are secondary, unadjusted for multiplicity, and never re-select anything.

The v14 log1p caveat applies: a log-target forecast is the median only if log residuals are symmetric. Every table reports `actual_above_forecast_share` per model (about 0.5 supports the median reading) and the mean forecast ratio median-type/mean-type, since geometry reacts to level.

## 4. Per-model residual pool

Reusing production's pool would bias the test: its residuals pair with production's forecast, so another forecast lands in the wrong decile and draws the wrong error. For each model m the experiment builds its own `ResidualPool` (unmodified) from rows `(event_date, pred_m, err_move_m = abs_move - pred_m, err_crush = crush_pct_iv30 - pred_iv_crush_30)`, `dropna` as in `Scorer._residual_pool` (`engine/score.py:3739`), strictly earlier events, the same decile rule and `MIN_POOL`.

- **Common support.** Candidates have stored predictions only from 2023; production has 2018 onward. Primary arm: **every model, production included, uses only events dated 2023-01-01 or later**, so pool size and burn-in are equal. Sensitivity arm (labelled): production with its native full-history pool; the gap shows how much of production's edge is pool length.
- **Burn-in.** An event is undetermined until its pool has `MIN_POOL` rows and its cutoff has 100 prior-window scores, so scoring starts in 2023H2 for all models alike. Events undetermined under any roster model leave the **paired population E** for all models (counts per quarter). A (model, structure) with fewer than 300 selected inner trades skips tuning, labelled (section 5).
- **Fast path.** As in EXP-133 (`build.py`), expected P&L is linear in the legs: simulate once per (model, event) the mean exit value of a single put at each listed ladder strike (about 45); every (structure, width) is then a dot product. The runner first checks equivalence against `pnl_sim.expected_pnl` and `price_structure` on a fixed sample and aborts above 1e-9.

## 5. Short-vol scoring and geometry divisors

Per (model, event, structure): forecast, width by `forecast_sizing` (declined if out of range), strikes snapped to the listed ladder in dollars, zero floor re-checked on listed strikes (EXP-133 rule; a negative-floor candidate is dropped), entry cost at the alpha mark, `exp_pnl_sim` through the model's pool, trailing cutoff on that (model, structure)'s own prior scores, spread and mcap terms exactly as `_short_vol_rule`, then realized exit value at marks. Entry/exit conventions are the board's for each structure; S5 pins them from the factories and refuses if a factory default disagrees.

Two geometry arms, both pre-registered:
- **As-is.** Registered divisors and bounds: what the board does.
- **Re-tuned.** A multiplier `c` on each structure's divisor, grid `{0.75, 0.9, 1.0, 1.1, 1.25, 1.5}`, chosen **per (model, structure)** on 2023/2024 by the primary metric through the full entry rule with the cutoff recomputed per `c`, ties to `c` nearest 1, frozen in the `tuning` receipt and applied unchanged to 2025. A median forecast is lower, hence narrower wings, so as-is partly tests the forecast's *scale*; re-tuning removes scale and asks about *shape*.

Primary metric: **flat-dollar P&L per candidate event** in E (non-trades count 0), lots by `CAPITAL_PER_TRADE` whole lots as in `portfolio.py`, alpha 0.5; the alpha grid {0, 0.25, 0.5, 0.75, 1} is reported. Secondary: return per taken trade, trade count, hit rate, share declined by width.

## 6. Long-straddle buy rule (proposed) and marks

STR-THRU's registered gate is a trained GBM (EXP-105, -184), STR-RUNUP's a trailing percentile of a trained score (EXP-185); neither is a bare function of the forecast, and trained gates would need refitting per forecast model. The proposal is the **forecast part of those gates as a bare rule**:

- **R_L (primary).** Buy `straddle_through` when `edge = pred_abs_move_m - implied_move` is at or above the top-20% cutoff of the model's own `edge` over the prior six months (`trailing_cutoff` called on a frame whose `exp_pnl_sim` column holds the `edge` series; at least 100 scores, strictly before the month), within the short-vol universe filters (mcap floor, spread over the two entry legs). `edge` is the existing `forecast_edge`. Flat dollars, same lots rule. Missing implied move or legs, or an entry quote that is crossed, locked or zero-bid: undetermined, not "no".
- **R_L2 (secondary).** Fixed `edge >= 0`, no cutoff. A mean forecast sits above a median one, so this changes *how many* trades it takes, which R_L cannot see. Trade counts are reported beside P&L.

Neither rule fits anything or looks at a P&L.

**Marks and ambiguous exits.** Every leg is priced with `FillModel(alpha).buy/sell`. Per-leg fallback, each step labelled in the row: (1) valid bid and ask: alpha mark; (2) crossed, locked, zero-bid or wide (`FillModel.is_wide`): alpha 0 (worst) for that leg; (3) no exit quote: the **structure** is marked at its worst-case payoff (zero for the straddle; the defined-risk floor for put structures), labelled `exit_unquoted_worst_case`. Tables report each model's share per rung, so geometry landing on thinly quoted strikes cannot hide; a sensitivity row drops rung 3.

## 7. Hypotheses, pre-registration and judgement

**Stated in `spec.yaml` before any 2025 look:**
- **H1 (short vol):** matched-pair `d = P&L(median-type) - P&L(mean-type) > 0`.
- **H2 (long vol, R_L):** `d < 0`.
- **H3 (secondary, interaction):** `d_short - d_long > 0` in return per candidate event, so the books are comparable. This is the direct reading of the user's question.

**Pre-registration that binds.** S1 scaffolds with `experiments/new_experiment.py` (PLANNED row). `engine/evaluate.py` sets `spec_hash_checked = False` and does not fail when no planned row exists, so the guard would silently no-op. The runner therefore refuses `EXPERIMENT_IDENTITY_CONFLICT` (reason `planned_row_missing`) unless exactly one PLANNED row has a `spec_hash` equal to the spec's, and the spec's own `preregistered_at` is before the run log's first entry (the check `engine/evaluate.py` makes). The spec carries the roster, contrasts, arms, `c` grid, bootstrap settings and category rules. v14's 2025 forecasts and SIZE's own 2025 RMSE/MAE exist before this runs; the claim is only that **no 2025 P&L** is seen before the spec is fixed, and this PR merges before the v14 run completes.

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

| Slice | Scope | Code / test / doc words | After |
| --- | --- | --- | --- |
| S1 | Scaffold EXP-188 (`spec.yaml` from this design, PLANNED row), prereg guard that cannot no-op, closure-hash test | ~100 / ~120 / 60 | - |
| S2a | `size_pnl_inputs`: bind v14 plan/seal/panel, resolve roster, refusals | ~160 / ~180 / 80 | S1, v14 S2c |
| S2b | Production and label readers, holdout drop, lead-1 check, per-model tables | ~140 / ~160 / 40 | S2a |
| S3 | Ladder-quote table over the pinned snapshot, quarter-chunked, progress | ~150 / ~150 / 40 | S1 |
| S4 | Per-model pool, linear fast path, equivalence check | ~170 / ~180 / 40 | S2b, S3 |
| S5 | Short-vol scoring: geometry, entry rule, marks and rung labels | ~170 / ~200 / 60 | S4 |
| S6 | Divisor re-tune on inner, `tuning` receipt | ~110 / ~130 / 40 | S5 |
| S7 | Straddle rules R_L / R_L2 | ~110 / ~130 / 40 | S3, S2b |
| S8a | Contrast statistics: signs, two bootstraps, categories | ~140 / ~160 / 60 | - |
| S8b | Report tables T1-T10 | ~170 / ~150 / 40 | S6, S7, S8a |
| S9 | Driver (inner/outer), 50-ticker smoke, runbook, contract lines | ~120 / ~120 / 100 | S8b |

S1, S3 and S8a can start in parallel; each slice changes behaviour only inside the new experiment.

## 10. Decisions for the user

1. **2025 forecasts for the matched pairs (blocking).** v14 stores 2025 forecasts only for refit units. (a, default) amend v14 S2b/S3 with a small pre-registered `extra_refits` list, the six pair members, before v14's plan is sealed (six more outer refits); (b) score pairs on inner 2023/2024 only and test 2025 on winners/blends, weakening H1/H2; (c) this experiment refits the six itself via the public refit API: most cost, edges toward the SIZE search. A sealed plan cannot be extended later.
2. **Consistency thresholds.** 75% / 60% quarter shares and 90% intervals, or stricter?
3. **Chooser and trained straddle gate.** Out of scope here. Follow-up scoring DYN-SV and a refit STR-THRU gate per forecast model (default: no)?
4. **Pool length.** Primary uses common 2023+ support (default), or production's native pool as primary?
5. **Re-tune breadth.** Per (model, structure) `c` (default), or one `c` per model pooled over the seven structures (less overfit, less flexible)?
