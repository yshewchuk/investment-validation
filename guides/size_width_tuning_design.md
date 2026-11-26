# Short-vol structure width tuning for the board (design, EXP-189)

**Status: design merged (user decisions 2026-11-24); code slices (section 11) in progress.** The design PR added this file only, with no code, `spec.yaml` or test, and no edit to EXP-188's design or spec. Experiment id **EXP-189** (slice W1 re-checks that it is free).

**Purpose.** Choose **production's** short-vol structure widths for the board, using production's forecast model only (**M0**, `size_v1_4`); no roster or per-model factor. If the board's widths change (a separate user decision), EXP-188 (`guides/size_decision_pnl_design.md`, production-exact) is rerun on them. EXP-189 has its own spec, pre-registration and PLANNED row, never changes EXP-188's spec or hash, and uses EXP-188's machinery as a library.

## 1. Question and evidence so far

Does a different strike width, set as a multiple `k` of production's width rule (`k = 1` is production), earn more for the seven put-side short-vol structures (TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5)? In EXP-188's 2025 run one `c` per model picked the grid edge `c = 0.5` (width 2x) for 11 of 12 models. The synthetic oracle (T11) shows P&L peaking at about 1.5-2x the actual move for BFLY-P, BFLY-P5, CND-PS, RAMP7 and CTR5; TWIN-P and TWIN-P5 are fine near 1x. EXP-137 found chosen widths at 1.3-1.7x the forecast (`engine/forecast_sizing.py` comment).

**Disclosure.** 2025 results of the EXP-188 runs (including `retuned` and T11) were seen while designing this. Every 2025 number is labelled **"2025 NOT blind for widths"**. All tuning reads 2023-24 only.

## 2. Constraints

- Production's rules stay the base. `k` multiplies the width `forecast_params` returns: `width = k * (forecast/100)/divisor`. `WIDTH_MIN`/`WIDTH_MAX` apply **after** `k`; a width outside them is declined (non-trade, P&L 0, counted). No edit to `engine/forecast_sizing.py`, `STRUCTURES`, any SIZE search module or file in `source_closure(root, ["experiments/size_panel.py"])`; `k` lives in new `size_width_*` modules.
- No EXP-188 edit: new modules import `size_pnl_*` unchanged (a missing hook is a separate change under EXP-188's owner). Forecast and crush forecast are M0's. No call-side short leg; marks only from `engine/fills.py`; holdouts only through the shared admission; `--no-ledger` for non-recorded runs.
- **Account (user 2026-11-24).** Every cash-secured account in EXP-189 starts at **$200,000** (the user's real account); the EXP-188 Amendment 2 `margin.simulate` policy is otherwise unchanged (cap 0.66 of current equity, `target_share` 0.25, expensive first, every short put secured at strike x 100).
- **Standard-tool check.** numpy only (`default_rng`, `polyfit`); breakeven alpha reuses `engine.evaluate.breakeven_alpha_from_sweep`.

## 3. The tunings (all for M0, each a factor or rule on production's width)

All tune on 2023-24 (scoring from 2023H2, EXP-188's burn-in), freeze before 2025 and run 2025 by month (R6). Same grid, objective and tie rule (section 5). Baseline: **production** (`k = 1`), recomputed in this run from the same pinned inputs.

| Id | Name | Factors | Tuned on |
| --- | --- | --- | --- |
| A | `global` | one `k` for all seven structures | M0's real 2023-24 P&L |
| B | `per_structure_real` | seven `k_s` | M0's real 2023-24 P&L (one path) |
| C | `per_structure_fake` | seven `k_s` | a fake forecast matching M0's error profile, `N = 20` seeded draws (section 4) |
| D | `two_factor` | `k_T` for the **TWIN pair** (TWIN-P, TWIN-P5: the forecast sits at the plateau centre or peak), `k_C` for the **five centre-peaked structures** (CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5: the forecast sits at the outer zero edge) | M0's real 2023-24 P&L |
| E | `expected_pnl` | none: per event, `k` from the grid maximising `exp_pnl_sim` for that structure (section 3.1) | no tuning; no realised outcome |
| anchor | `oracle_sweep` | descriptive only | forecast = actual move |

C versus B tests whether a smoother simulated objective beats one real path. **Anchor (descriptive, 2023-24):** width = `k` x the **actual** move, same gate, marks and exit; T4 prints score by `k`; it feeds no freeze or rule.

**3.1 Arm E.** `exp_pnl_sim` is EXP-188's entry-gate quantity (`engine/pnl_sim.py`): the mean simulated return over 4,000 draws with `move = max(forecast + err_move, 0)`, `crush = pred_crush + err_crush`, paired residuals from M0's residual pool (strictly earlier events), exit legs repriced by Black-Scholes. For each candidate event and structure, the fast path (section 4) gives it at every grid `k` as a dot product; E takes the `k` with the highest value among feasible widths (ties: nearest 1, then smaller; none feasible: declined as today). Its score replaces the candidate's `exp_pnl_sim` in the gate, with the trailing cutoff taken over E's own prior scores. Chosen widths are traded and scored like every other arm; no realised outcome is used (no single-path tuning noise). **How it differs from earlier per-event searches:** EXP-133's per-event *structure* choice widened the universe about 2.6x and did worse, and EXP-137's per-event widths clustered at 1.3-1.7x the forecast. E varies width only, within one fixed structure, over a fixed grid, with an expectation (not a realised) objective. Cautions: a maximum over ten noisy expectations is biased upward (winner's curse), and a per-event choice has the most freedom. **E fails** if it does not pass the section 7 rule, or if its chosen `k` is the same for at least 80% of a structure's events (it then equals a global factor and adds nothing; reported, not a pass). T3 also prints, per structure, mean `exp_pnl_sim` by `k` on 2023-24 beside the realised curve (B) and the fake-model curve (C): the three **agree** when their argmax `k` lie within one grid step; agreement is evidence the shape is real.

## 4. Tuning C: the fake forecast

**Error model fitted on M0's 2023-24 inner forecasts vs labels.** For inner paired events, M0 forecast `f`, label `y = abs_move`, both required positive (others dropped and counted `nonpositive_forecast` or `nonpositive_label`), on the log scale (that of EXP-188's `regression_direction` generator): `log f = L + b * (log y - mean(log y)) + s * z`, `z ~ N(0,1)`. `L`, `b` are the OLS intercept (at the mean) and slope, `s` the residual sd; `b < 1` is the under-prediction of large moves. Judgement call (mine): log scale keeps a fake forecast positive; Gaussian noise is pre-stated; residual skew and excess kurtosis are printed, no change. T1 lists `(L, b, s, n)`. **Hard check:** the fake's RMSE and MAE against `y` on inner E (percent points, mean over draws) are within 5% of M0's own (my judgement call), else C halts `fake_not_calibrated` (A, B, D, E and the anchor still run; `tuning_band` then uses B's curve alone).

**Why the fake may use the label (causality).** The fake simulates an error process, `f = g(y, z)`, so it depends on its own event's `y` by construction (as EXP-188's synthetic arm). It only chooses `k_s` from 2023-24 events whose labels printed before the 2025 stage; no 2025 or real decision uses its own event's label. Within a draw the pool and cutoff use strictly earlier events only (EXP-188 section 4).
**Sweep.** For each structure and grid `k`, `N = 20` draws: draw `d`, event `e` uses `numpy.random.default_rng(int.from_bytes(sha256(f"{event_id}|exp189-v1|{d}".encode()).digest()[:8], "big"))`, so `z` is common across structures and `k`. Each draw is a fake model with its own `ResidualPool` (`err_move = y - f`), the same trailing-cutoff gate, `FillModel` marks, exit and burn-in as EXP-188, via the fast path (one simulation per (draw, event) at each ladder strike; each (structure, `k`) a dot product; cutoff recomputed per `k`). The seven `k_s` go to an immutable `tuning_C` receipt before 2025; real M0 then runs with them on its own forecast (no level adjustment).

## 5. Grid, objective, tie rule

- **Grid (settled):** `k in {0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0}` (`k = 1/c` in EXP-188 terms). **Edge rule:** an argmax at 3.0 or 0.5 is not frozen: `k_at_edge`; that factor runs at its best interior value, flagged.
- **Objective (settled):** per (unit, `k`), the **stand-alone cash-secured $200,000 account return at `marks_mid`** (alpha 0.5; EXP-188 Amendment 1 and Amendment 2 account (b), `margin.simulate` unchanged), averaged over draws for C. The harsh primary marks are not used for tuning (user: unrealistic); fill robustness is in the success rule. Units: A one seven-structure account; D one per factor group; B and C each structure's account.- **Tie rule:** a `k` within one standard error of the best (across draws for C; across calendar quarters, the sd of quarterly returns over the square root of the quarter count, for A, B, D): the `k` nearest 1, then the smaller.
- **Minimum trades.** A (unit, `k`) with fewer than 300 trades (summed over draws for C) is `not_scored`; a unit with no scored `k` is `k_defaulted` (1.0).

## 6. Run order, outputs

`inner` (fit, sweeps, tunings A-D with the `tuning_C` receipt, anchor, `exp_pnl_sim` curves, tolerance bands of section 12) writes the `tuning_A`..`tuning_D` and `tuning_band` receipts; `outer` (2025) refuses without the receipt of every arm it runs (an omitted arm needs none; `tuning_band` only for the band and `fallback`). Each arm also runs on the inner window (2023H2, 2024). One heavy job at a time, `bounded_run`, quarter chunks, bounded stderr progress (start/end, done/total, rate, ETA, RSS).
`REPORT.md` (2025 tables titled "2025 NOT blind for widths"): T1 error model; T2 chosen factors and E's `k` distribution; T3 curves per structure (B, C, `exp_pnl_sim`; score, standard error, trades, declined share, **flatness**: best minus the score at `k = 1` and neighbours, in standard errors); T4 anchor; T5 production versus A-E: per trade, $200,000 seven-structure account at `marks_mid`, **breakeven alpha** and the harsh alpha-0 result, 2025 and inner halves; T6 per-structure accounts; T7 success table; T8-T9 section 12.

## 7. Success rule and the proposed board rule (written before any run)

Candidate X in {A, B, C, D, E} **passes** iff **(a)** its seven-structure $200,000 account return at `marks_mid` beats production's (`k = 1`) in **each** of 2025, 2023H2 and 2024 (each inner half starts a fresh account), **and (b)** its fills are comparable: X's **breakeven alpha** (the alpha at which mean return per trade crosses zero over the alpha ladder {0, 0.25, 0.5, 0.75, 1}, `engine.evaluate.breakeven_alpha_from_sweep`, the measure EXP-187 reported; no crossing reads 0.0 if always positive, 1.0 if always negative) is no more than production's **+ 0.05 in 2025**. The 0.05 margin is my judgement call, adjustable by the user. Breakeven alpha is reported for all periods; the harsh alpha-0 result is reported, not a gate. Guard: in 2025 X's stand-alone account (T6) at mid must not lose to production in more than 2 of 7 structures. 2023H2 and 2024 are in-sample for A, B, D (and for C's error model). 2025 is **not an independent evaluation**: the factors were not fitted on it, but its results were seen before this design, and it decides which candidate passes and is recommended, so the 2025 return of the recommended candidate is selection-biased upward and the fallback test of 12.4 inherits that. The recommendation is therefore evidence for the user's board decision, not a validated result; independent confirmation is the forward period after the data end (not read here), monitored after any board change.

**Recommendation among passers (fixed now).** Fewest fixed factors first: A (1), D (2), then C, then B (7 each; C first because its objective is less noisy, fixed now, not chosen from 2025), then E (a per-event rule, not a fixed factor; adopting it would mean the rule "pick `k` maximising `exp_pnl_sim` per event"). None passing: **recommend no change**. The recommendation is a **proposed board width rule**: divisors `divisor / k_s` (`divisor / k` for A, per group for D), with T3's flatness (a flat curve argues for leaving the board alone) and the declined share per structure at each proposed `k`.

**Feeding back.** Evidence, not a change. The board change is a **separate user decision**, which also needs the DYN-SV chooser review of EXP-188 Decision 3 (the chooser reads `width_over_forecast` features). After a board change EXP-188 is rerun on the new widths, in its own amendment by its own owner.

## 8. Failure semantics (R1-R6)

| Case | Result |
| --- | --- |
| R1 M0 inputs missing or failed (no forecast series, no inner decisions, under 300 inner events) | refuse `EXPERIMENT_IDENTITY_CONFLICT`; no partial report; nothing imputed |
| Whole-run refusals | `EXPERIMENT_IDENTITY_CONFLICT`, `details.reason` in `tuning_receipt_missing`, `tuning_receipt_changed`, `planned_row_missing`, `spec_hash_mismatch` |
| Fake outside 5% of M0's RMSE/MAE | `fake_not_calibrated`; C omitted, other arms run |
| (structure, `k`) with no feasible width or under 300 trades | `not_scored`, excluded from the argmax, declined share printed |
| E: no feasible `k` for an event | declined as today, counted |
| Breakeven alpha not computable (no taken trades) | reported `not_available`; condition (b) fails (X does not pass) |
| All `k` of a unit excluded / argmax at an edge | `k_defaulted` = 1.0 / `k_at_edge`, flagged |
| R2 cost | one process, `bounded_run`, quarter chunks; over-limit refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report |
| R3, R4 | seeds fixed by the section 4 recipe; receipts atomic, immutable, bound to the inputs receipt; a changed input refuses at replay |
| R6 | tuning reads 2023-24 only; 2025 refuses without each arm's receipt; 2025 decisions are written month by month before that month's labels are read (EXP-188 R6) |

## 9. Registration, spec hash

EXP-189 has its own `spec.yaml` (arms, grid, `N`, thresholds, success rule), its own PLANNED row with the spec hash, and a guard that cannot no-op (`planned_row_missing`). It reads the sealed inputs through EXP-188's `Sources` adapter, unchanged. Nothing is registered in this PR.

## 11. Split list (code slices, NOT started until this design merges)

| Slice | Scope | Non-test code / test lines / doc words | After |
| --- | --- | --- | --- |
| W1 | Scaffold EXP-189 (`spec.yaml`, PLANNED row, guard), `k` sizing module, id check, $200,000 account wiring | ~110 / ~120 / 60 | this PR |
| W2 | M0 error-model fit, calibration check, T1 | ~100 / ~110 / 40 | W1 |
| W3 | Seeded draws, `tuning_C` receipt, anchor, on W4's sweep | ~170 / ~170 / 60 | W2, W4 |
| W4 | Shared sweep and objective, tunings A, B, D, arm E (per-event `exp_pnl_sim` argmax, curves), 2025 and inner runs | ~170 / ~170 / 60 | W1 (parallel with W2) |
| W5 | Tolerance bands for three `phi`, `tuning_band` receipt, realised-width and decline tables (T8), `fallback` arm, T9, adoption rule | ~170 / ~170 / 80 | W4 |
| W6a | Breakeven alpha (ladder sweep), T7 success table, recommendation, proposed rule | ~110 / ~170 / 20 | W4 |
| W6b | Driver and smoke, report T2-T6, runbook | ~100 / ~90 / 80 | W5, W6a |

Re-split (supervisor, user 2026-11-24): W2 and W4 run in parallel; W4 builds the shared sweep and objective, W3 adds only seeded draws, `tuning_C` and the anchor.

Re-split (supervisor, 2026-11-24): W6 splits into W6a (pure logic: breakeven alpha, the T7 success table, the recommendation order, the proposed board rule; `size_width_success.py`) and W6b (driver, smoke run, T2-T6, runbook; after W5).

New `size_width_*` modules only; `size_pnl_*` is imported unchanged.

## 12. Width flexibility: tolerance band, nearest-feasible fallback, snap deviation (user 2026-11-24)

**Question.** `size_pnl_shortvol.resolve` resolves each leg to a listed strike and declines the candidate (`ladder_too_coarse`) when two legs collapse. Supervisor measurement (2025, M0, `k = 1`; T8 reproduces it): that decline hit about 27% of short-vol candidates at market cap >= $10B and about 40% at $1-10B, the largest non-trade reason. How wide a range of widths is nearly as good as the best, and can a nearby feasible geometry recover those candidates? 2025 NOT blind for widths.

**12.1 (a) Tolerance band.** Per structure, from the sweep curve (section 5 objective): the contiguous range of `k` around the argmax where the score stays at or above a fraction `phi` of the peak, for **each `phi` in {0.80, 0.90, 0.95}** (no single `phi` is picked in advance). Ends are located by linear interpolation between grid points and reported as a width ratio to production (`k`) and to the actual move (from the anchor). Computed on B's and C's curves (C primary; B alone when C is omitted) and the anchor for reference. A non-positive peak gives `no_positive_peak` (no band, no fallback); a band touching the grid edge is `band_open`. Frozen from 2023-24 in the `tuning_band` receipt.

**12.2 (c) Deviation of today's trades (descriptive, M0, `k = 1`).** Realised width: the width the listed strikes of a taken trade carry, measured as the structure's factory defines `width_moneyness` (`engine/structures.py`), against the target `forecast_params` returned. T8 gives quantiles (10, 25, 50, 75, 90) of realised/target per structure and cap tier (>= $10B, $1-10B, below), 2023-24 and 2025; decline shares by reason and tier (a mismatch with 27% and 40% is a finding); return per taken trade by deviation bucket (below 0.9, 0.9 to 1.1, above 1.1). The buckets are **descriptive only** (snap deviation correlates with price level and ladder spacing; no causal claim).

**12.3 (b) Nearest-feasible fallback.** Arm `fallback`, run for each `phi`, on the recommended candidate's factors from section 7 (`k = 1` when no change is recommended) and also at `k = 1`. A candidate declined `ladder_too_coarse` (not `structure_unresolved`, a missing fact, nor `width_declined`, `negative_floor`, `entry_not_debit`) is retried at widths `target * rho`, `rho` stepping by 0.05 over the structure's tolerance band relative to the arm's target (`k` band divided by the arm's `k`), nearest first (smallest `|log rho|`, smaller `rho` on ties). Each trial width must lie inside `[WIDTH_MIN, WIDTH_MAX]`; `rho` never leaves the band. The first `rho` whose snapped geometry has distinct legs, passes the zero-floor re-check and gives a debit entry is used; none: declined as today. Gate, cutoff (recovered scores join the history), marks, exit and accounts are unchanged; `rho` never looks at P&L or labels. T9, per `phi`, structure and tier: the band, declined candidates, recovered, recovered share, `rho` distribution, and recovered trades' return and P&L at `marks_mid` **against trades already taken**, and the account with versus without the fallback.

**12.4 Outputs and the adoption rule (pre-stated).** (i) A **recommended tolerance band per structure** at the chosen `phi`. (ii) **The `phi` is chosen from 2023-24 only (2025 never selects it), by this rule fixed now:** the **widest** (smallest) `phi` in {0.80, 0.90, 0.95} whose recovered trades' mean return per trade at `marks_mid` is not below the taken trades', pooled over 2023-24 on the 12.3 factors, with at least 100 recovered trades in 2023-24; none qualifies: `insufficient_recovery`, answer no. (iii) **Fallback adoption, yes or no, names that `phi`:** **yes** iff, at that `phi` and on the factors of 12.3, the `fallback` arm's account at `marks_mid` exceeds the same arm without it in each of 2025, 2023H2 and 2024, **and** its 2025 breakeven alpha is no more than the no-fallback arm's + 0.05. Otherwise no. The 100-trade floor is my judgement call. The board change is the user's separate decision, with the DYN-SV chooser review; EXP-188 is rerun if the widths or the fallback change.

**12.5 Failure semantics.** A non-positive peak: `no_positive_peak` (no band or fallback for that structure). No feasible `rho` in the band: declined as today, counted `fallback_infeasible`. Under 100 recovered trades in 2023-24 for every `phi`: `insufficient_recovery`, answer no. A realised width not computable (a leg missing): left out of T8, counted `deviation_unavailable`, never imputed. The geometry search is dot products on the existing ladder table (R2) with no randomness (R3). The `tuning_band` receipt is atomic and immutable (R4); `fallback` refuses without it (`tuning_receipt_missing`); bands are frozen from 2023-24 and no label or P&L enters the choice of `rho` (R6).

## Amendment 1 (2026-11-25): fair-value fills (model F); post-hoc revision by the user

**Status: design only (user decision 2026-11-25).** No code or `spec.yaml` change here; sections 1-12 are unedited. Where this section conflicts with them, it governs the Amendment 1 run only. **Model F** is `guides/fair_value_fill_design.md` ("F design"); this section states how EXP-189 consumes it and does not restate its pricing. Rules marked "mine" are my judgement calls.

### A1.1 Labelling

- **Post-hoc.** The user chose this revision after the full run under mid fills had been seen (2026-11-25): its 2023-24 tuning and its 2025 results. It is not a pre-registered test, and nothing in it is blind. 2025 was already "2025 NOT blind for widths".
- **The original stays on record** as EXP-189's only pre-registered result: its original spec fields (A1.9 only adds an `amendments` entry), PLANNED row, receipts, report and ledger rows are not edited or deleted. If the two runs recommend differently, both are stated.
- Every Amendment 1 table and `proposed_rule.json` carries **"POST-HOC (Amendment 1, F fills)"**; 2025 tables also keep "2025 NOT blind for widths".

### A1.2 Scoring

Every arm's primary P&L (production, A-E, the anchor, C's draws, `fallback`) is under F: entry cost `debit_F`, exit value `proceeds_F`, each leg at F's fair value (ORATS `smvVol` or F's fallback smile; intrinsic at expiry; clamped to bid/ask) plus the spread/size concession, at entry and at exit. Every place where sections 3-7 and 12 let `marks_mid` decide something (objective, tie rule, curves, bands, success condition (a), the 2-of-7 guard, `phi` and adoption, T8 returns, T9) reads F. This supersedes, for F only, section 2's "marks only from `engine/fills.py`"; `engine/fills.py` is unchanged and still produces the reported views.

**Reported beside F, same taken set, same lots rule, never deciding or tie-breaking:** **mid** (the original `marks_mid`: alpha 0.5, entry and exit) and **harsh** (the original alpha-0 result, entry and exit), computed by the existing code and rung rules, per trade and in the T5-T6 accounts (each account is walked again, so its funded set may differ). F's three sensitivities (unknown liquidity 0, unknown 1, mid-floored F) are reported per trade and for the seven-structure account.

**F's exit.** F design section 7 applies: a held trade never leaves; a leg with a usable quote but `no_fair_value` or `rate_unknown` takes `V = mid` (`exit_fair_mid`) and pays the concession. An `unpriceable` leg, or a ladder rung 3, marks the **structure** at the defined-risk floor 0.0 (`exit_unquoted_worst_case`). For an F-`unpriceable` leg that the ladder did not rung 3, the widened seam returns 0.0 with a flag, and a `size_width_*` wrapper around the copy's `score` labels the row `exit_unquoted_worst_case` and applies `drop_unquoted` to it, so rung shares count it; mid and harsh keep the ladder's rungs. Negative `proceeds_F` is kept and flagged `exit_negative`, not floored (mine, as current marks are not).

### A1.3 Selection and where F enters `exp_pnl_sim`

The machinery is unchanged; F's entry cost goes in.

- `exp_pnl_sim = (simulated exit value - debit_F) / debit_F`. **F enters through the entry cost only.** The **simulated exit uses neither F's concession nor F's fair value**: it stays the current European, zero-rate Black-Scholes mean (`put_means`). `experiments/size_pnl_pool.py`'s `structure_pnl` already takes the entry cost as an argument, so `engine/pnl_sim.py`, `size_pnl_*` and every v14-closure file are unchanged. **Selection score (settled).** User decision 2026-11-25: "the same machinery with F's entry cost: arm E's per-event k choice, A/B/D tuning on the inner period, and the top-20% trailing cutoff all rank on exp_pnl_sim with F's entry cost"; the amendment was to state whether the simulated exit uses F's concession. It does not. So F design section 8's second rule (a selection model's target is the return on debit including the conservative exits) **does not apply to EXP-189's selection score**; that exception follows from the decision, which does not name section 8. A realised or conservative exit is known only after the decision, so it cannot be a per-event score (R6). F design section 11 records the simulated exit with no concession. A-D's account objective uses `proceeds_F` and the conservative exits.
- **Consequence:** the score is optimistic against F-scored exits, by amounts that differ by name and width (F design section 11); rank gating does not remove this. Arm E may therefore favour a `k` whose exit concession is large. The report gives, by arm and chosen `k`, the mean exit concession (exit value at `V` minus `proceeds_F`, over `debit_F`; rung-3 and `unpriceable` exits excluded and counted).
- **Arm E** picks each event's `k` by `exp_pnl_sim` on `debit_F`, among widths that pass F's guard.
- **The top-20% trailing cutoff** (`trailing_cutoff`) ranks each arm's own history of those scores. Only scored rows enter it, so F-skipped rows do not.
- **A, B, D (and C)** still choose `k` by the section 5 account objective, not by mean `exp_pnl_sim`; their trades are gated by the F-based score and cutoff, and the account is F-marked. Burn-in uses F too, and T3's `exp_pnl_sim` curves are F-based.
- The universe filter (market cap; mean relative spread of the stored quotes) is unchanged.

**The seam (#2884).** `size_width_pricing.load(fills=...)` with a `size_width_fills.Fills` object is the plug point, but as merged it is **not sufficient**: `entry(priced)` gets only a `StructurePrice` whose snapshot has a blank ticker and the expiry as its date, so F cannot find the event's rows, and neither method returns reasons or flags. The code slice widens the seam inside `size_width_*` only, so that entry receives the event and its resolved legs, and reasons and exit flags reach the rows; with `CurrentFills`, outputs stay identical. Rung 3 keeps bypassing the seam. **Mid and harsh must be priced through a separate `CurrentFills` instance** with `repriced=True`: in an F-loaded copy, `entry` returns `debit_F` at every alpha. This widens `repriced`, documented for the breakeven endpoints only, to these two views, because a decision carries `debit_F`; it skips the decision-cost and debit checks, and F's guard keeps the mid debit positive (harsh is no smaller). At ladder alphas above 0.5 a repriced debit can be zero or negative: as the sweep does now, that trade has no return at that alpha and is left out of its mean; the report counts these per arm and alpha.

**Quotes F prices (mine).** F gets every listed row of each requested expiry, on the entry and exit dates, from the snapshot the sealed inputs pin. Where the snapshot repeats a (right, strike), the adapter keeps the row the ladder keeps (same entry and exit ranking), counted, before F sees the quotes; a duplicate quote key still reaching F, or a duplicate vol or rate key, refuses under F's R1. A resolved leg whose stored bid/ask differ from the ladder chunk's (values the chunk blanks excepted) refuses `f_quote_mismatch`. A snapshot lacking a column F requires refuses; no other snapshot is substituted.

### A1.4 F's guard: a counted non-trade per arm

After the geometry resolves (the existing reasons keep their meaning and come first), F's entry statuses and then its guard apply, once, at entry:

| Reason (counted per structure and arm) | When |
| --- | --- |
| `f_unpriceable`, `f_rate_unknown`, `f_no_fair_value` | a leg is not `ok` (F's `undetermined`); with several, the first in this order (mine) |
| `f_not_a_debit`, `f_fill_over_fair`, `f_spread_over_debit` | F design section 6, first condition that holds; `f_not_a_debit` replaces `entry_not_debit` |

Such a row has no `exp_pnl_sim`, stays out of the cutoff history and is a non-trade (P&L 0) for that arm. **The event stays in the comparison** (mine, as with `ladder_too_coarse`). The result is a function of the resolved legs and F's entry-date inputs (the whole expiry's quotes, vols, rate and liquidity; under F7, carry and dividends), not of the arm or forecast, so arms, controls and draws resolving the same legs of the same event get the same answer, and mid, harsh and the sensitivities share the taken set. Different widths have different legs, so arms can differ, and an arm can score better by not trading. T2 and T5 therefore give counts and shares by reason, arm and year, and, against production, the events where exactly one of the two was F-skipped. For E, a failing `k` is infeasible; E can step around the guard where a fixed `k` cannot, so the report gives the share of E's events with at least one F-skipped `k`. When no `k` is feasible, the existing rule (the row of the `k` nearest 1) supplies the reason. `fallback` still retries only `ladder_too_coarse`; a trial `rho` must also pass F's statuses and guard, else the next is tried.

### A1.5 Account

Unchanged policy ($200,000, cap 0.66, `target_share` 0.25, expensive first, every short put secured at strike x 100). The deciding account **funds at F cost**: `debit_F` sets entry cost and lots, `proceeds_F` the exit. The mid and harsh accounts use their own costs.

### A1.6 Unchanged

Arms, `k` grid, `N` and seeds, windows and periods, M0 and the fake's calibration, account rules, edge rule, tie rule, 300-trade and 100-recovery floors, `phi` set, recommendation order, and the success rule and guards, now judged on F P&L. One definition needs care (mine, for the user to override): condition (b) and 12.4's margin are **breakeven alpha**, which is defined on the alpha ladder and has no F value. It keeps its definition and +0.05 margin, computed by ladder repricing of **the arm's F-selected taken set**, all five alphas priced by the separate `CurrentFills` instance with `repriced=True`, never the F-loaded one (which would flatten the ladder; the sweep now prices through one instance, so the code slice separates them); it tests that set's fill robustness, not its F return.

### A1.7 F failure states mapped to R1-R6 (adds to section 8)

| Case | EXP-189 result |
| --- | --- |
| R1 inputs: F's binding, hash, manifest, column, duplicate-key or unit refusals; `f_quote_mismatch` | refuse `EXPERIMENT_IDENTITY_CONFLICT` with `details.reason`; no partial report; nothing imputed |
| R1 `rate_unknown` | entry: `f_rate_unknown` non-trade; exit with a usable quote: `exit_fair_mid` |
| R1 no `smvVol` | not itself a failure when the fallback smile applies: the expiry uses it, source counted and shares printed by year. A row F leaves `no_fair_value` (smile cannot be fitted, a non-fit-set row without `smvVol` on an `smvVol` expiry, tree or spot rule): `f_no_fair_value` at entry, `exit_fair_mid` at exit |
| R1 unpriceable leg | entry: `f_unpriceable`, or the earlier `entry_quote_unusable`; exit: floor 0.0 |
| R1 missing size or open interest | not a failure: F's unknown value 0.5, counted per field and year; sensitivities 0 and 1 |
| R2 | F design chunking; each quote row priced once per quarter chunk and reused across `k`, arms and draws; F's load and the stage run under `bounded_run`; over a limit: `RESOURCE_LIMIT_EXCEEDED`, no partial report |
| R3 | F has no randomness; results do not depend on which structure or arm asked |
| R4 | the inputs receipts bind the fill-model id, `fvf_id`, both manifest and every table hash, the pinned quote snapshot id, units, F1's two rate rules and the carry/dividend mode; the same ids bind `months/`, `e_choices/` and tuning receipts through `inputs_hash`; a change refuses at replay |
| R5 | an `fvf_id` or fill-model id that differs from the spec's refuses; an F spec never runs on `CurrentFills` as primary |
| R6 | entry uses only entry-date quotes, vols and spot, and a rate row dated no later than the entry date (F design sections 4 and 10); exit-date rows are never read for a decision; sections 4 and 8 are otherwise unchanged. Limit (mine): we have not verified that vendor `smvVol` and rates are what was known on that date |

Status, vol-source and liquidity coverage on entry rows is printed **before any outcome is read**, and F design section 8's diagnostics are reported, with better-than-mid and no-arbitrage-violation shares also by arm (a search may favour both).

### A1.8 Borrow and dividends

PR #2910 (open) amends the F design to use ORATS borrow and dividend inputs, consumed by its slice F7, and orders F5 after F7. **If #2910 is merged before the run**, the spec binds an `fvf_id` that includes F7 and a re-ingest carrying those fields, and the run waits for F7's code (the merged document alone prices nothing). **If not**, F runs as built: **risk-free carry and zero dividends**, which spec and report state. The limit: every EXP-189 leg is a put, a missed dividend makes F under-value puts, and the clamp does not bound the error. The choice is fixed when the amended spec is registered; changing it needs a new amendment, hash and PLANNED row. The Amendment 1 spec entry takes the first branch (F7, #2957, is merged).

### A1.9 Registration and run

1. Slice W7e adds one `amendments` entry to `spec.yaml` (this section; post-hoc flag and label; fill model F as primary and what it supersedes in `constraints`, `objective` and `success_rule`; the A1.7 R4 bindings, units taken from the re-ingest using no outcome). `preregistered_at` and the original fields stay. The hash changes. The entry is not written until F design section 13 item 3 (the two rate rules) is settled (user, 2026-11-26: the defaults). **F is selected only by this entry**; without it the code prices as before.
2. The **supervisor registers a new PLANNED row** with the new hash (the guard then expects two EXP-189 PLANNED rows, the new one last). No agent writes `experiments/LEDGER.csv`. Until then every stage refuses.
3. **Smoke, 50 tickers, with F timing** (`--no-ledger`, fresh `/tmp` out-dir): all admitted events of the 50 plan tickers with the smallest `sha256("<ticker>|exp189-a1")` (mine; no outcome used), all three stages. Record per stage wall time, peak RSS, F load and pricing time, quote rows priced and their rate, entry coverage, and a full-run estimate, before the full run. No smoke result tunes F or a rule; a rule change needs a new amendment.
4. **Full rerun of all stages** (inner, outer, report) in a fresh out-dir. No earlier receipt, month file or tuning is reused or deleted.

### A1.10 Split list (code slices; no W7a1-W7d slice starts until this section and F4 and F5 are merged, plus F7 when A1.8 requires it)

W7a1-W7d are code slices sized to skip CodeRabbit: 1-149 **changed** production lines (additions plus deletions in `.py` outside `tests/`); every other file is a test under `tests/` or an `ARCHITECTURE.md`/`README.md` outside `guides/` (at most 40 changed lines in total); no path contains "design" or "spec"; no renames. All `spec.yaml` and `guides/` changes are in W7e, which needs CodeRabbit. Code edits are in `size_width_*` and EXP-189's tests. No EXP-188 file, F module (F's own slices own those) or v14-closure file changes. Sizes: changed production lines / test lines / doc words.

| Slice | Scope | Size | After |
| --- | --- | --- | --- |
| W7a1 | F quote adapter: bound loads, quote rows, `f_quote_mismatch`, per-chunk leg tables | ~120 / ~140 / 30 | F4, F5 |
| W7a2 | Widened seam (event, resolved legs, reasons and flags); `CurrentFills` identical | ~100 / ~100 / 20 | F4, F5 |
| W7a3 | F fills object: entry statuses and guard reasons, exit marks and flags | ~110 / ~140 / 30 | W7a1, W7a2 |
| W7b1 | F-primary rows with mid, harsh and sensitivity columns | ~100 / ~120 / 20 | W7a3 |
| W7b2 | Accounts, objective, fallback, T8 and T9 inputs read F; A1.4 counts; ladder breakeven and non-positive-debit counts | ~110 / ~140 / 30 | W7b1 |
| W7c1 | Report: labels, F-primary tables beside mid, harsh and sensitivities | ~110 / ~110 / 30 | W7b1 |
| W7c2 | Report: coverage, counts, concession and F diagnostics | ~110 / ~110 / 30 | W7b2 |
| W7d | Spec-driven F selection code, receipt bindings, driver F timing, smoke ticker ids | ~110 / ~130 / 40 | W7a3 |
| W7e | `spec.yaml` amendment and runbook (`guides/`); no production `.py` | 0 / ~50 / 150 | none (re-split) |

W7a1-W7d are tested with F selected by injection or a fixture spec on synthetic data; only W7e's `spec.yaml` entry lets a real run select it. A slice that would exceed 149 changed production lines or 40 doc lines is split again.

Re-split (supervisor, user 2026-11-26): W7e (spec and runbook) starts at once, in parallel with W7a1/W7a2, because it is the only CodeRabbit slice; W7d runs in parallel with W7b1 after W7a3; W7c1 in parallel with W7b2 after W7b1; W7c2 after W7b2. The run starts after W7b2, W7d and W7e; the report stage is re-run once W7c2 lands. A merged spec selects nothing before the supervisor registers the PLANNED row.

**Report inputs.** The report stage reads no sources. So before the run starts, the run stages must write, and bind by hash in a stage receipt, every input of every W7c1 and W7c2 table and diagnostic (this amendment and F design section 8). For A1.3's exit concession that includes, per taken trade, the exit value at `V`, `proceeds_F`, `debit_F` and the rung-3 and `unpriceable` exclusion flags. W7c1 and W7c2 read only stored, receipt-bound outputs. If any input is not so stored, the run waits until W7c2 has landed and the run stages write it.
