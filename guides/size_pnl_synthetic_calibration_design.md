# EXP-188 addendum: synthetic-predictor calibration arm (design)

Design only. Sibling of `guides/size_decision_pnl_design.md` (the main doc is dense and its slice list is long, so the addendum lives here and the main doc links to it). Section numbers below that read "main doc §N" refer to that file. User request, 2026-11-18: while the v14 run is going, validate the P&L machinery's assumptions with simulated predictors of known properties.

## 1. Purpose and boundary

The arm answers three questions about the EXP-188 machinery, not about SIZE:

- **Positive control.** A perfect forecast (oracle) must earn at least as much as any degraded one; a noisier forecast must earn less. If not, the machinery (or a hypothesis about it) is wrong before any real model is judged.
- **Negative control.** A forecast with no cross-sectional information (constant) must show no ranking edge.
- **Power.** How many dollars per event does one unit of forecast error cost? That scales what EXP-188's real contrasts (H1-H3) could ever detect.

It is **not** a P&L search. It never touches the pre-registered roster (main doc §3), its hypotheses, its categories or its multiplicity, and nothing it produces is an input to any real model, tuning receipt, SIZE winner or board decision. Every table is labelled **SYNTHETIC**.

## 2. Predictors

All are functions of the realized label `y` (`abs_move`, percent points, as in the main doc's inputs table), generated deterministically per event. For event `e`: `z_e ~ N(0,1)` drawn from a numpy generator seeded by `sha256(event_id | "exp188-synth-v1")`. The seed depends on the event only, so `z_e` is **common to every noisy predictor** (the ladder is then a pure change of scale and the mean/median pair differs only in level, which keeps contrasts low-noise). Every predictor must also pass the existing forecast rule: a non-positive or NaN forecast is declined by geometry, never clipped.

| Family | Definition | Count |
| --- | --- | --- |
| Oracle | `f = y` | 1 |
| Median-unbiased noise | `f = y * exp(s*z)`; `median(f given y) = y` | one per rung |
| Mean-unbiased noise | `f = y * exp(s*z - s^2/2)`; `E(f given y) = y` | one per rung |
| Regression-direction mean | `f = y * exp(s*z + s^2/2)`; the level a mean forecast has above a median one when `E(y given f) = f` (see §2.1) | one per rung |
| Shrinkage | `f = (1-L)*y + L*m`, `L` in {0.25, 0.5, 0.75, 1}; `m` = expanding mean of admitted inner labels dated strictly before the event's calendar quarter (undetermined until one quarter exists, absorbed by burn-in, main doc §4). `L = 1` is the constant (null) predictor | 4 |
| Level shift | `f = y * (1 +/- 0.10)` | 2 |

**Spread ladder** `s` in {0.2, 0.4, 0.6, 0.8} (log scale, residual standard deviation of `log(f/y)`). Realized RMSE and MAE of `f` against `y` are computed on population E per predictor and reported beside the rung. The report also prints production's own inner `sd(log(y/pred))`, so a reader can see which rung is realistic; the rungs are not scaled to it (Decision 1).

Total: 1 + 12 + 4 + 2 = **19 predictors**.

### 2.1 What the mean/median pair does and does not say

The construction conditions on `y`, whereas a real mean forecast satisfies `E(y given f) = f`. The two give opposite level orderings: mean-unbiased-given-`y` sits *below* the median-unbiased forecast by a factor `exp(-s^2/2)`, while a real mean-type forecast sits *above* a median-type one. The third family (regression direction) supplies the real-direction level so the pair is not misread. What the arm tests is exactly the **level effect on strike geometry and entry at fixed noise**; it cannot say whether SIZE's real mean- and median-type forecasts differ by that level (main doc §3 measures that). Noise is also independent of ticker, season and size, unlike real errors; accepted.

## 3. Scope and exclusion (inner years 2023-2024 only)

- The generator's only inputs are the sealed inner population's event ids as returned by S2a's binding (which has already applied shared population admission and outcome isolation, dropping member and conflicting-sibling events before any label read) and the labels the S2b reader returns for those admitted keys. It has no path to a raw label table, the 2025 slot, the random 3% holdout or the rolling 6-month holdouts.
- It asserts that every event date is in `[2023-01-01, 2024-12-31]` and that its output ids are a subset of its input ids, every input id kept or counted as dropped with its reason; each label date must also equal the admitted event date. A violation refuses `SYNTHETIC_OUT_OF_SCOPE` (no partial output, no repair).
- A test passes a fixture containing a holdout-classified event and a 2025 event: the first is absent from the admitted ids (so never generated), the second makes the generator refuse. Excluded counts by reason are logged to stderr and printed in the report header.
- The oracle and the noisy families use the event's own label by construction (look-ahead is the point). The main doc's R6 month-by-month label rule governs the real roster only; here it is deliberately void for the forecast, but still holds for the residual pools: the pool for an event contains strictly earlier events (pool row = earlier event, its synthetic forecast, its label), as in main doc §4.

## 4. Outputs

For each predictor and each of **9 books** (the 7 put-side short-vol structures, straddle R_L, straddle R_L2; main doc §§5-6), on population E, geometry **as-is** only (no `c` re-tune: nothing here is tuned), alpha 0.5, the same flat-dollar lots rule:

1. flat-dollar P&L per candidate event (non-trades count 0) and the quarterly series behind it;
2. trade count and declined-by-width share;
3. realized RMSE and MAE of the predictor on E, in percent points.

Derived, per type (median-unbiased, mean-unbiased, regression-direction) and per book:

- **Slope** of mean P&L per event on RMSE across the four rungs (ordinary least squares, 4 points), and the same on MAE: **dollars per event per error point**. Reported with the bootstrap-by-quarter 90% interval of the slope (resample quarters, rebuild the four means), fixed seed, 10,000 resamples, reusing S8a's resampler unchanged.
- **Contrasts**: `d_s = P&L(median-unbiased) - P&L(mean-unbiased)` per rung and book, with sign by quarter and by year, mean and quarter-bootstrap interval, all by S8a's code (§7).
- **Power read-across (descriptive only).** For each real matched pair, the gap in inner RMSE and MAE (read from the stored inner predictions already bound by S2a) times the slope gives an implied P&L gap per event; it is printed beside the width of EXP-188's own quarter-bootstrap interval for that contrast once that interval exists. "Implied gap smaller than half the interval width" is printed as `likely_underpowered`. It never gates, reorders or reinterprets any EXP-188 result.

## 5. Pre-registered expectations and what each failure means

Fixed in the spec's `synthetic_arm` block before any run. Directions for the mean/median pair follow the user's hypotheses (short vol prefers the median-unbiased level, straddles the mean-unbiased level), stated by predictor name; the report also prints them as level orderings. A rung counts only where the level gap is non-trivial: `s >= 0.4` (ratio of levels `exp(-s^2/2)` at most 0.93).

| # | Expectation | Judged how |
| --- | --- | --- |
| X1 | Oracle P&L per event is at least every other predictor's, in each book | oracle minus other: quarter-bootstrap interval not entirely below 0, and sign positive in at least 60% of quarters (my judgement call) |
| X2 | P&L is non-increasing down the ladder (each of the three noise types, each book) | all three adjacent steps non-positive = Clear; at most one inversion with a negative slope = Directional; else Failed |
| X3 | Short vol: `d_s > 0` (median-unbiased beats mean-unbiased) at every rung `s >= 0.4` | S8a categories on each rung's contrast, across the seven short-vol books |
| X4 | Straddle (R_L2 and R_L): `d_s < 0` at every rung `s >= 0.4` | as X3; R_L2 is the sharper test because its trade count follows the level, R_L's trailing cutoff absorbs most of it |
| X5 | Consistency of levels: sign of `P&L(+10%) - P&L(-10%)` (noise-free) equals the sign of `P&L(median) - P&L(mean)` at the rung whose level ratio `exp(s^2/2)` is nearest the shift pair's ratio 1.1/0.9 = 1.222 (`s = 0.6`, ratio 1.197) | sign agreement per book |
| X6 | Null predictor (`L = 1`): every short-vol book shows one width for every event within a calendar quarter (the constant is the expanding mean, so it changes only between quarters; zero within-quarter width variance) and the straddle edge ranks events by implied move alone | width variance and trade count printed; failing means information is entering the rule from somewhere else |
| X7 | Exact-replay: the same spec and inputs reproduce every table bit for bit | rerun on a 50-ticker smoke |

**Reading a failure** (machinery bug or hypothesis wrong, in order of check):

- **All predictors earn identical P&L to 1e-9 in a book:** the forecast never reaches geometry or the entry rule for that book. Machinery bug (wiring, wrong column, a constant overriding the forecast).
- **X1 fails in one structure only, by a small margin:** the structure's registered divisor is off-optimum for a perfect forecast; a hypothesis about the divisor, not a bug. Failing in all books, or by a large margin with the oracle's trade count far from the others': suspect the oracle's degenerate residual pool (its move residuals are zero, so the move draw is deterministic and only the crush residual varies) before suspecting the engine; a sanity check prints the oracle's pool size and residual spread.
- **X2 fails (P&L rises with noise) in several books:** a selection or ordering bug (labels reaching a decision, a cutoff computed on the wrong window) or an effect where noise widens wings that pay off. Check trade counts: if noisier forecasts trade far more, it is the trade filter, not the forecast information.
- **X2 holds but all slopes are near 0 with wide intervals:** forecast quality barely moves P&L at these spreads, or the sample is too short for this arm: EXP-188's real contrasts are then underpowered by the same measure (this is the intended read of §4's power line, not a failure of the machinery).
- **X3/X4 wrong sign, X5 right:** the level effect is real but the hypothesis direction (as set by the user) is wrong. **X3/X4 wrong sign and X5 wrong sign with the same direction:** consistent, hypothesis wrong. **X5 disagrees with X3/X4:** the noise-by-level interaction or a bug; compare trade counts and rerun the two level-only predictors.
- **X6 fails:** another source feeds the rule (leak from implied move, mcap or spread terms): inspect the book's entry terms before anything else.

A Failed X1, X2, X6 or X7 stops the real run until the cause is classified in the report; X3-X5 outcomes do not stop it (they are findings about the hypotheses). This stop is my judgement call.

## 6. How it plugs in

- **Source type.** S2a's binding yields, per model, arrays of `(event_id, pred)` for the sealed inner population. The generator returns the same shape under `source: "synthetic"`, with `model_id = "synthetic-" + content_hash({source: "synthetic", family, params, seed_tag})` (main doc R5; the prefix is what S6's guard keys on). Downstream stages (S3 ladder table, S4 pool, S5 short vol, S7 straddle) take any such array with no change: that generality is the contract the new slice checks, not a new reader.
- **Separate arm, receipt and report.** A `synthetic` stage receipt (immutable, atomic, content hashes, R4) and `decisions_synthetic.parquet`; the report gains **T11 (SYNTHETIC calibration)** after T10, with the X-table of §5. Synthetic models are absent from T1-T10, from the 12-model cap, from the pair and multiplicity accounting, and from the PLANNED row's roster; the `synthetic_arm` block (families, `s` ladder, `L` values, shifts, seed tag, expectations) is inside the spec and its hash.
- **Cannot feed tuning.** S6's `tuning` receipt and `c` search reject any synthetic `model_id` (`SYNTHETIC_IN_TUNING`), and a test asserts that deleting every synthetic file leaves every roster output byte-identical (SY1 pins the precondition, that no roster module imports the synthetic module; SY2 pins the byte-identity). The only read-across to real models is the descriptive power line of §4.
- **Stage and timing.** It needs inner data only, so it never reads the 2025 outer stage; but it still waits for the sealed v14 plan (event population) and runs after the v14 run completes.
- **Cost.** 19 models, inner years only (about two thirds of the events), as-is geometry and no re-tune: a rough estimate of about 55% of the main run (the smoke measures it), in the same quarter-chunked single process under `bounded_run`, one heavy job at a time (the main doc's R2 and §8). Smoke on the 50-ticker `--event-ids` population with `--no-ledger`. Progress logging as in main doc §8, plus the exclusion counts.

## 7. Standard-tool check and split

Reused unchanged: numpy (`default_rng`, OLS for the slope via `numpy.polyfit`), `ResidualPool`, `pnl_sim.expected_pnl`, S8a's quarter and ticker bootstraps and categories. No format is hand-rolled; a custom generator is needed because no library produces forecasts of a given conditional bias from a realized label.

| Slice | Scope | Code / test lines / doc words | After |
| --- | --- | --- | --- |
| SY1 (this PR) | Generator and guards (§2, §3); tables in the `model_tables` shape for `source: "synthetic"`; spec `synthetic_arm` block; `SYNTHETIC_IN_TUNING` | ~150 / ~320 / 150 | S2a, S2b, S6 |
| SY2 | The inner-year synthetic stage (as-is scoring of the 19 tables), `synthetic` receipt, slope, contrast and X1-X7 evaluation using S8a; T11 | split from SY, see below | SY1, S4, S5, S7, S8a, S8b |
| SY (original) | Generator and guards (§2, §3); `source: "synthetic"` wiring; `synthetic` receipt; slope, contrast and X1-X7 evaluation using S8a; T11; amend the spec with the `synthetic_arm` block; `SYNTHETIC_IN_TUNING` | ~140 / ~160 / 100 | S2a, S2b (admitted-label reader), S4, S5, S6 (hosts the `SYNTHETIC_IN_TUNING` guard), S7 and S8a (statistics); T11 needs S8b. If the ~140-line estimate proves low at Spec time, split into SY1 (generator, guards, wiring) and SY2 (evaluation, T11) |
| Run | Smoke, then the recorded synthetic stage and T11 | no code | v14 completed; S9 driver |

The run step is heavy: after v14 finishes, alone, bounded, as in the main doc §8. The new slice changes behaviour only inside the new experiment; it edits no SIZE search module and nothing in the `experiments/size_panel.py` closure, and S1's closure-hash equality check still applies to it.

## 8. Decisions (settled by the user, 2026-11-18)

1. **Spread rungs: fixed.** `s` in {0.2, 0.4, 0.6, 0.8}; production's measured inner `sd(log(y/pred))` is printed beside them, and the rungs are not rescaled to it.
2. **Regression-direction family: kept.** The third, real-direction mean-type family stays (19 predictors).
3. **Spec amendment: re-plan the single PLANNED row.** The `synthetic_arm` block changes the spec hash. No P&L has been seen, so the one EXP-188 PLANNED row is re-planned before any run, not a separate row. The ledger is data outside git, so the supervisor re-registers the row before the first run.
4. **Power read-across: allowed.** The arm may read real inner RMSE/MAE gaps descriptively (stored inner predictions) for the `likely_underpowered` line; it reads nothing from 2025 or any holdout.

Implementation notes (SY1, PR for `feat/pnsy1`): synthetic model ids are `"synthetic-" + content_hash(...)`, so S6's `SYNTHETIC_IN_TUNING` guard needs no registry. The out-of-scope refusal uses the registered code `HOLDOUT_ACCESS_DENIED` with `details.reason = SYNTHETIC_OUT_OF_SCOPE`. A shrinkage event with no prior quarter is dropped from that model's table and counted (`undetermined_mean`).
