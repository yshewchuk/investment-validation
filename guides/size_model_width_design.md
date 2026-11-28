# Model x width for the seven short-vol structures under model A (design, EXP-191)

**Status: design and spec only; no code, no ledger row.** This is a new experiment; EXP-189 is unchanged. It reuses EXP-189 (`size_width_tuning_design.md`, with Amendments 1 and 2), EXP-188 (`size_decision_pnl_design.md`) and fill model A. The machine-readable form is `experiments/EXP-191_model_x_width_short_vol/spec.yaml`. Items marked "mine" are my judgement calls; section 9 lists those still open.

## 1. Question, evidence, labels

Does another move forecast, with its own width factors, beat production's forecast at production widths on TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7 and CTR5?

Motivating evidence (supervisor analysis, 2026-11-28, aggregates only, not reproduced here):

- EXP-189 Amendment 2 at production width: `exp_pnl_sim` ranks taken trades backwards (Spearman about -0.17).
- Move-forecast error explains about 46% of realised-return variance; crush error under 1%.
- M0's move forecast runs about 21% low: about 0.55x implied, against about 0.67x actual.
- EXP-188 2025 at mid, $200,000: E-log +4.5%, M0 +3.8%.

**User decisions (2026-11-28, settled):** a new experiment; model arms M0, E-log, N-log, M3 and bias-corrected M0; the full 10-width grid per model, so A, B, D and E are rebuilt per model; a pass bar both relative and absolute (section 5).

**Labels.** The run is **post-hoc**: the 2025 results of EXP-188 and EXP-189, and the evidence above, were seen before this design, and model A was fitted on the user's own fills, which postdate every test period, so its constants use information unavailable at any decision date. The bias-corrected arm was motivated by a bias measured on a span that includes test periods. Every table and `proposed_rule.json` carries **"POST-HOC (EXP-191, model A)"**; 2025 titles add **"2025 NOT blind for widths or models"**. A recommendation is evidence for a separate board decision, which also needs EXP-188's conditional chooser and gate follow-up for any model change.

## 2. Model arms

| Arm | EXP-188 role | Forecast `f` (percent points) |
| --- | --- | --- |
| `M0` | M0 | production `size_v1_4` tier-4 `pred_abs_move`; reference |
| `BC` | derived from M0 | `r_month * f_M0` (2.1) |
| `M3` | M3 | v14 MAE winner, stored walk-forward predictions |
| `E-log` | E-log | spline ElasticNet, `log1p` twin |
| `N-log` | N-log | MLP, `log1p` twin |

M3, E-log and N-log come from EXP-188's `resolve_roster` and its inner and 2025 binders over the sealed v14 records; nothing is refitted. If two arms resolve to one `model_id`, it runs once under both names. M3 was chosen by 2023-24 inner MAE, so those years are a selection window for it; the pair members were not score-chosen. Production's `pred_iv_crush_30` is the crush forecast for every arm.

### 2.1 Bias-corrected production (all values mine, Q4)

For a decision month starting on date `S`:

- **Rows:** admitted events with `event_date` in `[S - 12 months, S)`, a finite M0 forecast `f > 0`, and a label `y = abs_move` whose `available_as_of_date` is strictly before `S` (data dates, never `computed_at`).
- **Statistic:** `r = sum(y) / sum(f)`, the aggregate realised-to-predicted ratio. Unlike a mean of per-event ratios, small forecasts cannot dominate it. No clipping.
- **Minimum n:** 500 rows. **Fallback:** `r = 1.0` (plain M0), flagged `bias_fallback` and listed by month; not a refusal.
- **Population:** M0's own coverage as the existing production and label readers return it (both need the crush column too), not the common set. Months early in 2023 need 2022 rows, read through the same admission and readers with a 2022-01-01 start.
- **Use:** every BC row of month `j`, in the pool or decided, is `r_j * f_M0` with that month's own `r_j`: the forecast it would have had then.

Months through 2025-01 use only pre-2025 labels and are fixed in `prepare`; later 2025 ratios are computed inside BC's outer loop, after the earlier month's decisions are written and its labels read.

A constant `r` would only rescale every width, which a global `k` already covers. BC can differ through `r`'s changes over time, the discrete grid, and the simulator, whose pool already carries M0's average error.

## 3. Common population and how a forecast enters

**Default (Q1): one common event set.** E is the intersection of the five arms' tables after EXP-188's rules (admission and holdout drops, crush forecast present, facts present, pre-2025 labels present). Every arm, pool, cutoff history and comparison uses E. M0's tier-4 forecast is missing for about 20% of 2023-26 events (supervisor figure); those leave for all arms, since the reference cannot be scored on them and production's crush forecast comes from the same row source. No own-coverage account is computed: arms trading different events would not be comparable.

`prepare` records per arm and year: admitted events, own-coverage events, the common count, and the events lost **only** because that arm lacks them. M0 here is not EXP-189's M0 population, so its numbers are not a reproduction; the report states both event counts.

For arm `m`, event `e`, grid factor `k`:

- **Simulator:** unchanged. `move = max(f_m + err_move, 0)`, `crush = pred_crush + err_crush`; paired residuals come from `m`'s **own** `ResidualPool`: `err_move = y - f_m` over strictly earlier E events dated 2023-01-01 or later whose labels had printed before the month (equal pool support for all arms). Draws are keyed by `event_id`, so arms share one random stream. `k` does **not** enter the simulator.
- **Width:** production's rule on the arm's forecast, times `k`: `width_moneyness = k * (f_m / 100) / divisor`; `WIDTH_MIN` and `WIDTH_MAX` apply after `k`; strikes resolve through the existing pricing seam. A declined geometry or model A guard skip is a counted non-trade, per arm, and the event stays in E.
- **Score and gate:** `exp_pnl_sim` uses model A's entry debit (no concession in the simulated exit, EXP-189 A1.3); the top-20% trailing cutoff uses the (arm, structure, width rule)'s own history; spread and market-cap terms are unchanged.

**Simulation eligibility must not differ by arm.** It depends on pool size, facts, the strike grid and finite inputs, all shared on E. Each arm's stage compares its monthly priced-event-id hash with M0's and refuses `population_mismatch` at the first difference.

**Error models.** The simulator's error model is the per-arm pool: rebuilt per arm by construction. EXP-189's T1 log-scale fit (`L`, `b`, `s`, `n`, residual shape) is also fitted per arm on inner E rows, as a **description only**. EXP-189 arm C, which feeds T1 into a fake forecast, is not run by default (Q2), nor are the oracle anchor, tolerance bands or fallback (Q3).

## 4. Width arms, tuning and fills

Per model, on `k in {0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}`: `P` (`k = 1`), `A` (one `k`), `B` (seven `k_s`), `D` (TWIN pair and the other five), `E` (per event, `k` maximising `exp_pnl_sim`).

EXP-189's rules are unchanged: each arm of a model is tuned on **that model's** 2023H2-2024 trades; the objective is the stand-alone cash-secured $200,000 account return at **model A** fills; same account policy, tie, edge, 300-trade, `k_defaulted` and `same_k` rules; tunings are frozen before 2025.

**Fills.** Model A primary, as EXP-189 Amendment 2: same constants file and `constants_id`, F leg inputs and pins, entry guard, exit rules and funding at the A debit. **Mid and harsh are reported and never decide**, as are the other beside-A views. `spread050` is a separate full run of all five arms, required only if some candidate passes; the verdict is read on `primary`, and any pass/fail or recommendation difference is reported as sensitive to wide-leg cost (Q7).

Open issue #3091 (the replay donor pool ignores label availability) is not addressed.

## 5. Pass bar and recommendation

A candidate is (model, width rule): 25, of which the reference R = (M0, P), leaving 24. **A candidate passes only if all hold, at model A fills, on seven-structure $200,000 accounts, each period starting fresh:**

1. **Relative (EXP-189):** its return beats R's in each of 2023H2, 2024 and 2025.
2. Its 2025 breakeven alpha (ladder repricing of its A-selected taken set, EXP-189 A1.6) is at most R's + 0.05; not computable fails.
3. Its 2025 per-structure accounts lose to R's in at most 2 of 7 structures.
4. For E, that model's E summary is not `same_k`.
5. **Absolute:** return strictly above 0 in at least 2 of the 3 periods. A missing or non-finite return is not positive.

`size_width_success.success_table` supplies items 1-4, called once per model against R (its `E` label keeps the `same_k` test); item 5 and the cross-model order are new code.

**Recommendation order among passers (mine, Q6):** by model first, `M0, BC, M3, E-log, N-log` (smallest board change first; swaps in EXP-188's roster order, not ordered by any P&L); within a model `P, A, D, B, E` (fewest fitted factors first). No passer: **no change**.

**Limits, printed in the report.** 24 candidates are tried with no multiplicity correction, so a lone narrow pass is weak evidence. 2023H2 and 2024 are in-sample for every A, B and D, and a candidate can meet item 5 on those two alone (Q8). 2025 selects the recommendation, so its return is biased upward; only data after the end of the test periods can confirm it.

## 6. Stages, cost, failure semantics, artifacts

`run.py --stage prepare | inner | outer | report`, with `--model` for inner and outer. Order: `prepare`; `inner` for every model, M0 first; `outer` for every model, M0 first; `report`. Models run **sequentially** as separate processes: one heavy job at a time under `bounded_run`. Each model writes under `models/<arm>/`. Resume unit: one model. A completed stage (summary present, bindings equal) logs `complete` and exits without recomputing. A model with a partial stage refuses `model_stage_partial`; the operator renames its directory aside (kept, never read) and reruns that model's inner and outer.
**Cost (estimate).** EXP-189 Amendment 2 took about 8 h for one model over 10 widths, peak about 5 GB (supervisor's figures). Five such runs give a **planning ceiling of about 40 h**, and as much again if `spread050` is required. EXP-191 skips arm C, the anchor, bands and fallback, so it should cost less; how much is unmeasured. Separate processes keep peak memory per run near the one-model figure, not five times it (expected, unmeasured). `prepare` runs no sweep. A 50-ticker `--no-ledger` smoke (EXP-189's rule, seed tag `exp191`) records wall time and RSS per stage and a full-run estimate before the full run.

**Failure semantics (R1-R6).** Checks run in the listed order; the first failure refuses. Codes are `EXPERIMENT_IDENTITY_CONFLICT` with `details.reason` unless the reused code defines another.

| | Rule |
| --- | --- |
| R1 `prepare` | registration guard; bounded run; `--expected`; sealed-run binding; each arm resolved (`model_unresolved`) and bound (`model_inputs_missing`, naming the arm: excluded, partial, failed or no 2025 forecasts); production and label reader refusals; nonempty E (`empty_population`). One missing arm refuses the whole run. |
| R1 `inner`, `outer` | guard; bounded run; arguments (`model_unknown`); fill-model and F input refusals before any quote read; `population_missing`; recomputed bindings differ (`population_changed`); `model_stage_partial`; M0 first for other models (`reference_missing`); `outer` needs **every** model's completed inner (`inner_incomplete`), then its own tuning receipts (`tuning_receipt_missing`, `tuning_receipt_changed`); during pricing, `population_mismatch` |
| R1 `report` | guard; every model's inner and outer summary (`model_stage_missing`); one population, spec hash, fill model and variant across them (`population_changed`) |
| R2 | quarter chunks; over limit `RESOURCE_LIMIT_EXCEEDED`; a late failure writes no summary and no `REPORT.md` or `proposed_rule.json`. Month files already frozen stay but are bound by no summary and never reported |
| R3 | no new randomness; draws keyed by `event_id`; fixed row and arm order |
| R4 | receipts atomic and immutable, hashing what was **loaded**: arm bindings, E ids, labels, facts, pools' inputs, BC rows and ratios, quote tables, constants, spec hash, and a source hash over the new `size_mw_*` files and model A's source list |
| R5 | `model_id` from `resolve_roster` (BC: a hash over M0's id and the 2.1 rule); a spec, constants id or variant mismatch refuses |
| R6 | tuning reads 2023-24; every inner completes before any outer; each 2025 month's decisions are written before its labels are read; BC ratios use only labels available before `S` |

Progress goes to stderr: stage start and end, elapsed, months done/total, rate, ETA, exclusions, RSS. New modules are named `size_mw_*`: a new `size_width_*` file would change EXP-189's model A source hash. No edit to `size_pnl_*`, `size_width_*`, SIZE search modules or the `experiments/size_panel.py` source closure.

**Stored artifacts and owners** (all under the run's out-dir; S numbers are section 7 slices):

| Artifact | Producer | Schema |
| --- | --- | --- |
| `receipts/population.json`, `population.parquet` | S1 (M0), widened S2 | bindings, coverage counts; E ids with each stored arm's forecast and crush forecast |
| `receipts/bias_ratio.json` (through 2025-01), BC `outer_summary` (rest) | S4 | per month: window, n, `sum_y`, `sum_f`, `r`, `bias_fallback`, rows hash |
| `models/<arm>/` inputs and summary receipts, `months/`, `labels/` | S1 | EXP-189 shapes plus population hash and priced-id hashes; `labels/`: event_id, `abs_move`, crush, available date |
| `models/<arm>/scored/<month>.parquet`, hash-bound | S1, all widths S3 | per event, structure and grid `k` (S3 adds E): `f`, pred_crush, implied_move, mcap, `exp_pnl_sim`, status, reason; for taken rows entry and exit dates, lots, A, mid and harsh entry, exit and P&L, ladder returns |
| `tuning_{A,B,D,E}` receipts, `e_choices/` | S3 | EXP-189 schemas |
| model A `legs_NNNN`, `taken_NNNN` | S1, through the existing store | EXP-189 A2 raw exit rows |
| `REPORT.md`, `proposed_rule.json` | S6 | EXP-189 schema plus model and pass/fail rows |

## 7. Report

Per model where applicable: coverage (section 3); forecast level (forecast and actual over implied, `actual_above_forecast_share`); BC ratio by month; T1; chosen factors and E's `k` shares; curves; non-trade counts by reason; seven-structure and per-structure accounts for P, A, B, D, E at A, mid and harsh, with breakeven alpha; the 24-candidate pass table with fail reasons and the recommendation.

It **must include** the standard diagnostics being built into the shared EXP-189 report code (owner `pn189dx`), per model: rank checks of predicted against realised return at mid and A, win rates by structure, move/crush error decomposition, and the top-fifth profile. EXP-191 calls that code and defines none of those statistics. It stores raw rows and computes them at report time; an undefined or uncomputable diagnostic prints `not_available` with a reason and never blocks a run.

## 8. Pre-registration and ledger

This PR's spec is `status: planned` with `preregistered_at` unset, and no ledger row is written. Order: settle section 9; merge the run-path code; a spec PR sets `preregistered_at`; the supervisor appends one EXP-191 PLANNED row with that hash; smoke; full run. The new guard cannot no-op: `planned_row_missing`, `planned_row_ambiguous`, `spec_hash_mismatch`, and an unset `preregistered_at` each refuse every stage. All stages require `--no-ledger`; no agent or code writes `experiments/LEDGER.csv`; recording the outcome is the supervisor's step. A later spec change adds an `amendments` entry and a new PLANNED row.

## 9. Open questions

A slice does not start until its questions are settled. The spec records the defaults.

| Q | Blocks | Question, **default**, trade-off |
| --- | --- | --- |
| Q1 | S1 | Common set or own coverage? **Common.** Own coverage keeps more events per model, but accounts then differ in events, and M0's gaps cannot be filled. |
| Q2 | S3 | Rebuild arm C per model? **No** (not in the user's arm list). Yes gives a less noisy per-structure tuning, at 20 fake draws per model (likely a large runtime share, unmeasured) and a sixth rule per model. |
| Q3 | S3 | Run anchor, tolerance bands and fallback? **No.** Fallback may recover coarse-ladder declines, at up to 15 extra sweeps per model and more candidates. |
| Q4 | S4 | BC rule: **12 months, ratio of sums, n >= 500, fallback 1.0, no clip.** A longer window is steadier but slower to adapt; a median ratio targets the typical event, not the level. |
| Q5 | S1 | Diagnostics inputs: **S1's rows plus the model A raw rows**; confirm with `pn189dx` before S1. A missing column otherwise forces a rerun. |
| Q6 | S5 | Recommendation order: **section 5.** Alternative: width rule first, which prefers a model swap at `k = 1` to M0 with tuned widths. |
| Q7 | S7 | `spread050`: **all five arms, only if something passes.** Running only passing arms and M0 is cheaper but needs a subset mode. |
| Q8 | S5 | Keep both bars as decided (**yes**), or also require 2025 above 0? That guards against an in-sample-only absolute pass, but changes a settled rule. |

## 10. Slice plan

Thin end-to-end slices; sizes are estimates of added non-test lines (hard ceiling 400; split before exceeding 300). Tests use synthetic fixtures and independent oracles (hand-computed ratios, accounts and pass rows). Each slice builds in the section 6 order, writes no partial summary, binds what it loaded and logs to stderr.

| Slice | Path added and its failure semantics | Lines | After |
| --- | --- | --- | --- |
| S1 | Walking skeleton: guard; `size_mw_*` driver and `run.py`; `prepare`, inner, outer and a minimal report for **M0 at P** under model A through the existing sweep. R1: guard, bounded, expected, inputs, population. R2: no report on late failure. R4: population, `scored/`, `labels/`, source hash. R6: month freeze. | ~290 | this PR, Q1, Q5 |
| S2 | Add M3, E-log, N-log: roster, common E, coverage, per-model stages and resume, per-model pool and T1. R1: `model_unresolved`, `model_inputs_missing`, `reference_missing`, `inner_incomplete`, `population_mismatch`, `model_stage_partial`. R4, R5: arm bindings. | ~150 | S1 |
| S3 | Widen to 10 widths: A, B, D tuning, E, `e_choices/`, tuning receipts. R1: tuning refusals. R4: all widths in `scored/`. R6: frozen before outer. | ~150 | S2, Q2, Q3 |
| S4 | BC: pure ratio module, `bias_ratio` receipt, monthly outer wiring. R1: ratio inputs refuse like labels; short n falls back, never refuses. R3, R4: deterministic, rows hashed. R6: availability before `S`. | ~149 | S2, Q4 |
| S5 | Pure pass bar and cross-model recommendation. R1: malformed input is `ValueError`; missing values fail the candidate. No I/O. | ~100 | Q6, Q8; parallel with S1-S4 |
| S6 | Full report, labels, per-model diagnostics, `proposed_rule.json`. R1: report row. R2: write nothing on refusal. R4: reads only bound receipts and rows. | ~250 | S3-S5, `pn189dx` merged |
| S7 | Runbook (smoke, full run, `spread050`, resume). The `preregistered_at` change is its own spec PR. No run-path code. | ~0 | S4, S5, Q7 |

The full run waits for S1-S5, S7 and registration. S6 blocks only the report, provided Q5 fixed the stored rows first.

**Standard-tool check.** New code is an array ratio, a pass table and wiring around existing pricing, simulation, account, tuning and receipt code. No standard format is involved.
