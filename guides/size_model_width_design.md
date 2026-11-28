# Model x width for the seven short-vol structures under model A (design, EXP-191)

**Design and spec only.** EXP-189 is unchanged. This reuses EXP-189 (`size_width_tuning_design.md`, Amendments 1 and 2), EXP-188 (`size_decision_pnl_design.md`) and fill model A. Spec: `experiments/EXP-191_model_x_width_short_vol/spec.yaml`. "Mine" marks my judgement calls.

## 1. Question, evidence, labels

Does another move forecast, with its own width factors, beat production's at production widths on TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7 and CTR5?

Supervisor analysis, 2026-11-28 (aggregates, not reproduced):

- EXP-189 A2 at production width: `exp_pnl_sim` ranks taken trades backwards (Spearman about -0.17).
- Move-forecast error explains about 46% of realised-return variance; crush error under 1%.
- M0 forecasts about 21% low: about 0.55x implied, against 0.67x actual.
- EXP-188 2025 at mid, $200,000: E-log +4.5%, M0 +3.8%.

**User decisions (2026-11-28, settled):** this new experiment, its five arms (section 2), the full 10-width grid per model (rebuilding A, B, D, E), and a relative plus absolute pass bar.

**Post-hoc.** EXP-188's and EXP-189's 2025 results and this evidence were seen first. Model A was fitted on the user's fills, which postdate every test period. The bias correction was motivated by a bias measured partly on test periods. Every table and `proposed_rule.json` carries **"POST-HOC (EXP-191, model A)"**; 2025 titles add **"2025 NOT blind for widths or models"**. A recommendation only informs a separate board decision; a model change also needs EXP-188's chooser and gate follow-up.

## 2. Model arms

| Arm | Forecast `f` (percent points) |
| --- | --- |
| `M0` (reference) | production `size_v1_4` tier-4 `pred_abs_move` |
| `BC` | `r_month * f_M0` (2.1) |
| `M3` | v14 MAE winner |
| `E-log`, `N-log` | spline ElasticNet and MLP `log1p` twins |

M3, E-log and N-log are stored walk-forward predictions bound by EXP-188's roster and binders; nothing is refitted. Arms sharing one `model_id` run once, both named. M3 was chosen on 2023-24 inner MAE; the pair members were not score-chosen. Every arm uses production's `pred_iv_crush_30` as crush forecast.

### 2.1 Bias-corrected production (values mine, Q4)

For the month starting on `S`:

- **Rows:** admitted events with `event_date` in `[S - 12 months, S)`, finite M0 forecast `f > 0`, and label `y = abs_move` with `available_as_of_date` strictly before `S` (data dates, never `computed_at`).
- **Statistic:** `r = sum(y) / sum(f)`; unlike a mean of per-event ratios, small forecasts cannot dominate it. No clipping.
- **Minimum** 500 rows; else **fallback** `r = 1.0`, flagged `bias_fallback` by month; not a refusal.
- **Population:** M0's own coverage under the existing production and label readers (both also need crush), not the common set; reading starts 2022-01-01.
- **Use:** `f_BC = r_j * f_M0`, with `j` the row's own month, for every decided event **and** every pool row, whose residual is `y - f_BC`.

Months through 2025-01 use pre-2025 labels and are fixed in `prepare`; later ratios are computed in BC's outer loop, after the earlier month's decisions are written and labels read.

A constant `r` rescales widths much as a global `k` does (bounds and grid aside). It also changes the simulation: a draw shifts by `(r - 1) * (f_event - f_donor)`. A changing `r` adds more.

## 3. Common population and forecast entry

**Common event set (default, Q1).** E is the intersection of the five arms' tables after EXP-188's rules (admission and holdout drops; crush forecast, facts and pre-2025 labels present). Every arm, pool, cutoff and comparison uses E. M0's forecast is missing for about 20% of 2023-26 events (supervisor figure); these leave for all arms, since the reference cannot score them. There are no own-coverage accounts.

`prepare` records per arm and year: admitted, own coverage, common, and events lost **only** by that arm. M0 here does not reproduce EXP-189 (different events).

For arm `m` and grid factor `k`:

- **Simulator:** `move = max(f_m + err_move, 0)`, `crush = pred_crush + err_crush`. Paired residuals come from `m`'s **own** `ResidualPool` (`err_move = y - f_m`) over E events dated from 2023-01-01 and before the month, labels available strictly before its start. Draws are keyed by `event_id`, a shared stream. `k` does not enter.
- **Width:** `width_moneyness = k * (f_m / 100) / divisor`; `WIDTH_MIN` and `WIDTH_MAX` apply after `k`; strikes resolve through the existing seam. A declined geometry or model A guard skip is a per-arm non-trade; the event stays in E.
- **Gate:** `exp_pnl_sim` uses model A's entry debit (EXP-189 A1.3); the top-20% trailing cutoff uses that (arm, structure, width rule)'s history; spread and market-cap terms are unchanged.

**Invariant.** Which events get a simulation depends on pool size, facts, strike grid and finite inputs, shared on E. Each arm compares its monthly **simulated** event ids (taken before any width decline or guard skip) with M0's, and refuses `population_mismatch` at the first difference.

**Error models.** The simulator's is the per-arm pool. EXP-189's T1 fit is repeated per arm on inner E rows, **description only**. By default: no arm C, which turns T1 into a fake forecast (Q2), and no oracle anchor, tolerance bands or fallback (Q3).

## 4. Width arms, tuning, fills

Per model on `k in {0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}`: `P` (`k = 1`), `A` (one `k`), `B` (one per structure), `D` (TWIN pair; other five), `E` (per event, `k` maximising `exp_pnl_sim`).

A, B and D are tuned on **that model's** 2023H2-2024 trades and frozen before 2025. EXP-189's rules are unchanged: stand-alone cash-secured $200,000 account return at **model A** fills; same account policy, tie, edge, 300-trade, `k_defaulted` and `same_k` rules.

**Fills.** Model A as in EXP-189 Amendment 2: same constants and `constants_id`, F inputs and pins, entry guard, exit rules and A-debit funding. **Mid, harsh and the other beside-A views never decide.** `spread050` is a separate full run of all five arms, required only if a candidate passes; `primary` decides, and any pass/fail or recommendation difference is reported as sensitive to wide-leg cost (Q7).

**Selecting model A.** `size_width_f_select.fill_model` reads only `amendments[]` and returns `None` for this spec's top-level `fill_model`, which would mean current marks. So the new guard reads the top-level block, passes it to that selector as `{"amendments": [{"fill_model": block}]}`, and puts the result in its receipt. A missing block or an id other than `A` refuses `fill_model_missing` at every stage. S1a tests this on the committed spec.

Open issue #3091 (replay donors ignore label availability) is not addressed.

## 5. Pass bar and recommendation

Candidates are (model, width rule): 25; excluding reference R = (M0, P), 24. A candidate passes only if all hold on seven-structure $200,000 accounts at A, fresh each period:

1. Its return beats R's in each of 2023H2, 2024 and 2025.
2. Its 2025 breakeven alpha (ladder repricing of its A-selected taken set) is at most R's + 0.05; not computable fails.
3. In 2025 it loses to R in at most 2 of 7 per-structure accounts.
4. For E, the model's E summary is not `same_k`.
5. Its return is above 0 in at least 2 of the 3 periods; missing or non-finite is not positive.

Items 1-4 are EXP-189's rules, from `size_width_success.success_table` per model against R.

**Order among passers (mine, Q6):** by model `M0, BC, M3, E-log, N-log` (smallest board change first; swaps in EXP-188's roster order, not by P&L), then width `P, A, D, B, E` (fewest fitted factors first). No passer: **no change**.

**Limits, printed in the report:** no multiplicity correction over 24 candidates; 2023H2 and 2024 are in-sample for A, B and D and alone can satisfy item 5 (Q8); 2025 selects the recommendation, biasing its return upward.

## 6. Stages, cost, failures, artifacts

`run.py --stage prepare | inner | outer | report`, with `--model` for inner and outer. Every stage takes `--out-dir`, `--expected`, `--sources`, `--sources-config` and `--no-ledger`; inner and outer also `--f-pins`, `--f-quotes` and optional `--fill-variant`. Order: `prepare`; all inners; all outers; `report`; M0 first. Models run **sequentially**, one heavy job at a time under `bounded_run`, writing under `models/<arm>/`. A completed stage with equal bindings logs `complete` and does not recompute. A partial model refuses `model_stage_partial`; the operator renames its directory aside (kept, never read) and reruns its inner and outer.

**Cost.** EXP-189 A2 took about 8 h for one model, peak about 5 GB (supervisor's figures): a **planning ceiling of about 40 h** for five, doubled if `spread050` runs. The skipped extras should lower it (unmeasured); separate processes should keep peak memory near one model's. A 50-ticker `--no-ledger` smoke (EXP-189's rule, seed tag `exp191`) first records wall time and RSS per stage.

**Failures.** Checks run in order. Codes are `EXPERIMENT_IDENTITY_CONFLICT` with `details.reason`; reused code keeps its own (unbounded run: `RESOURCE_UNAVAILABLE`). Nothing retries automatically.

| | Rule |
| --- | --- |
| R1 `prepare` | guard, incl. `fill_model_missing`; bounded run; `--expected`; sealed-run binding; arms resolved (`model_unresolved`) and bound (`model_inputs_missing`, naming an excluded, partial or failed arm, or one without 2025 forecasts); reader refusals; `empty_population`. One missing arm refuses the run. |
| R1 `inner`, `outer` | guard; bounded run; `model_unknown`; fill and F input refusals before any quote read; `population_missing`; `population_changed`; `model_stage_partial`; `reference_missing` (M0 first); for `outer`, every model's inner (`inner_incomplete`), then its tunings (`tuning_receipt_missing`, `tuning_receipt_changed`); then `population_mismatch` |
| R1 `report` | guard; `model_stage_missing`; one population, spec hash, fill model and variant (`population_changed`) |
| R2 | quarter chunks; over limit `RESOURCE_LIMIT_EXCEEDED`; a late failure writes no summary, `REPORT.md` or `proposed_rule.json`. Frozen month files stay, unbound and never reported |
| R3 | no new randomness; `event_id` draws; fixed row and arm order |
| R4 | atomic, immutable receipts hash what was **loaded**: arm bindings, E ids, labels, facts, pool inputs, BC rows and ratios, quote tables, constants, spec, and sources (`size_mw_*` plus model A's list) |
| R5 | `model_id` from `resolve_roster` (BC: hash of M0's id and rule 2.1); spec, constants id or variant mismatch refuses |
| R6 | tuning reads 2023-24; all inners precede any outer; 2025 decisions are written before that month's labels are read; BC uses only labels available before `S` |

Stderr shows stage start, end and elapsed, months done/total, rate, ETA, exclusions and RSS. New modules are `size_mw_*`, because a new `size_width_*` file would change EXP-189's model A source hash. No edit to `size_pnl_*`, `size_width_*`, SIZE search modules or the `experiments/size_panel.py` source closure.

**Artifacts and owning slices:**

| Artifact (under the out-dir) | Slice | Schema |
| --- | --- | --- |
| `receipts/population.json`, `population.parquet` | S1a, widened S2 | bindings, coverage counts; E ids, each arm's forecast, crush forecast |
| `receipts/bias_ratio.json`; BC `outer_summary` for later months | S4 | month, window, n, `sum_y`, `sum_f`, `r`, fallback flag, rows hash |
| `models/<arm>/` input and summary receipts, `months/`, `labels/` | S1a inner, S1b outer | EXP-189 shapes plus population and simulated-id hashes; labels: event_id, `abs_move`, crush, available date |
| `models/<arm>/scored/<month>.parquet`, hash-bound | S1a inner, S1b outer, all widths S3 | per event, structure, `k` (and E): forecasts, implied move, mcap, `exp_pnl_sim`, status, reason; taken rows: dates, lots, A, mid and harsh entry, exit, P&L, ladder returns |
| `tuning_{A,B,D,E}` receipts, `e_choices/` | S3 | EXP-189 |
| model A `legs_NNNN`, `taken_NNNN` | S1a, S1b, existing store | EXP-189 A2 raw exit rows |
| `REPORT.md`; `proposed_rule.json` | minimal S1a; S6a, S6b | EXP-189 plus model and pass rows |

## 7. Report

Per model: coverage; forecast level (including `actual_above_forecast_share`); BC ratios; T1; factors and E's `k` shares; curves; non-trades; accounts at A, mid and harsh with breakeven alpha. Then the pass table and recommendation.

It **must include**, per model, the standard diagnostics being built into shared EXP-189 report code (owner `pn189dx`): predicted-versus-realised rank checks at mid and A, win rates by structure, move/crush error decomposition, and top-fifth profile. EXP-191 calls that code and defines none of these statistics; it computes them at report time from raw rows. One that cannot be computed prints `not_available` with a reason and never blocks a run.

## 8. Registration and ledger

The spec is `status: planned`, `preregistered_at` unset. Order: settle section 9; merge run-path code; a spec PR sets `preregistered_at`; the supervisor appends one PLANNED row with that hash; smoke; full run. Every stage refuses `planned_row_missing`, `planned_row_ambiguous`, `spec_hash_mismatch` or unset `preregistered_at`. All stages require `--no-ledger`; no agent or code writes `experiments/LEDGER.csv`, and the supervisor records the outcome. A later spec change adds an `amendments` entry and a PLANNED row, never one with `fill_model`.

## 9. Open questions

A slice waits for its questions.

| Q | Blocks | Question, **default**, trade-off |
| --- | --- | --- |
| Q1 | S1a | Common set or own coverage? **Common.** Own coverage keeps more events, but accounts stop being comparable. |
| Q2 | S3 | Arm C per model? **No** (not in the user's list). It is less noisy but costs 20 fake draws per model and adds a rule. |
| Q3 | S3 | Anchor, bands, fallback? **No.** Fallback may recover coarse-ladder declines, with more sweeps and candidates. |
| Q4 | S4 | BC: **12 months, ratio of sums, n >= 500, fallback 1.0, no clip.** Longer is steadier but slower; a median targets the typical event, not the level. |
| Q5 | S1a | Diagnostics inputs: **the `scored/`, `labels/` and A raw rows;** confirm with `pn189dx`, or a missing column forces a rerun. |
| Q6 | S5 | **Section 5's order.** Width-first would prefer a model swap at `k = 1` to tuned M0. |
| Q7 | S7 | `spread050`: **all five, only after a pass.** A subset is cheaper but needs a new mode. |
| Q8 | S5 | **Keep the bars as decided,** or also require 2025 above 0, blocking an in-sample-only absolute pass but changing a settled rule. |

## 10. Slice plan

Lines are estimated added non-test code / test lines / doc words. Target about 150 code lines, limit about 300. S1a is the walking skeleton. Tests use synthetic fixtures and independent oracles (hand-computed ratios, accounts, pass rows). Every slice follows section 6's order, publishes no partial summary and receipts what it loaded. **Every slice's stderr logging** covers the section 6 items its path produces; S5, pure, logs nothing.

| Slice | Path added; failure semantics | Lines | After |
| --- | --- | --- | --- |
| S1a | Guard with model A selection, `run.py`, `prepare`, inner and one-table report for M0 at P. R1: guard, bounded, expected, inputs, `empty_population`. R2: no late-failure report. R4: population, rows, sources. | ~150 / ~180 / 40 | Q1, Q5 |
| S1b | Add outer. R1: `population_missing`, `population_changed`, `inner_incomplete`. R2: no summary. R6: month freeze. | ~110 / ~130 / 20 | S1a |
| S2 | Other forecast arms, common E, coverage, per-model stages and resume, pools, T1. R1: `model_unresolved`, `model_inputs_missing`, `model_unknown`, `reference_missing`, `population_mismatch`, `model_stage_partial`. R4, R5: arm bindings. | ~150 / ~180 / 30 | S1b |
| S3 | 10 widths, A, B, D, E, receipts. R1: tuning refusals. R2: no partial tuning summary. R4: all rows. R6: frozen before outer. | ~150 / ~160 / 30 | S2, Q2, Q3 |
| S4 | BC ratio module and monthly wiring. R1: label refusals; short n falls back. R3, R4: deterministic, hashed rows. R6: availability before `S`. | ~149 / ~170 / 30 | S2, Q4 |
| S5 | Pure pass bar and order. R1: malformed input `ValueError`; missing values fail the candidate. R2, R4: no I/O. | ~100 / ~150 / 20 | Q6, Q8; parallel |
| S6a | Pass table, recommendation, labels, `proposed_rule.json`. R1: report row. R2: write nothing on refusal. R4: bound receipts only. | ~130 / ~140 / 30 | S3-S5 |
| S6b | Per-model tables and diagnostics. R1, R2, R4: as S6a; diagnostics `not_available`, never refuse. | ~140 / ~140 / 30 | S6a, `pn189dx` merged |
| S7 | Runbook: smoke, run, `spread050`, resume, logs to check. No code; `preregistered_at` is a separate spec PR. | 0 / 0 / ~600 | S3, S4, Q7 |

The full run waits for S1a-S5, S7 and registration; S6a and S6b block only reporting.

**Standard-tool check.** No standard format is involved; new code wraps existing repository code. Nothing enters `STRUCTURES` or the board.
