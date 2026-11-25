# Per-event quantile model: trading (design, EXP-190, part 2: Stage B)

**Status: design only.** Part 1 (`guides/size_quantile_move_design.md`) holds the model, Stage A, shared constraints, the settled decisions (its section 10) and the registration rule: the PLANNED row waits for this part. **Stage B runs only if Q or Q_raw passes Stage A**; both then carry, labelled with their own verdicts. Below, `x` is Q or Q_raw. "Mine" marks my judgement calls.

## 1. Why a new fill model

- **EXP-105 diagnostics** (`reports/exp-105_log_diagnostics.md` §8.5, supervisor-local, not tracked): STR-THRU's edge lived in the widest-quoted names. The widest relative-spread quintile (median 0.597) earned +9.87% per trade and supplied 48.9% of summed returns. The whole book earned +4.04% at mid and -1.42% at alpha 0.40; its capital-weighted breakeven alpha is 0.423. Mid is a real price only where quotes are narrow.
- **EXP-133** (`guides/exp133_structure_search.md` §4, §6): a vendor mid surface violates no-arbitrage, and searching over strikes harvests it (44.6% of the free search's entries violated it, 1.0% of the incumbent's; 30.8% of TWIN-P5 exits). Its `zero_cost` bug also shows that a refusal applied at one fill price compares different books.

Per-event choices over widths and trades search the same surface.

## 2. Fill model F

Experiment-side (`experiments/size_quantile_fill*`); `engine/fills.py`, `engine/structures.py` and `engine/pnl_sim.py` do not change. Legs are resolved with `engine.chain_index.price_structure_indexed` at mid; F reprices each resolved leg from its bid and ask.

**Fair value** V per leg: Black-Scholes with American exercise at ORATS `smvVol`, then **clamped to [bid, ask]**.

- **Pricer** (mine): Cox-Ross-Rubinstein tree, 200 steps, discrete dividends (escrowed), for puts and calls. Tool check: QuantLib is not installed, and adding it would change the pinned environment sealed runs record; the need is one function on scipy/numpy, tested against European Black-Scholes and published American values.
- **Rate:** a table committed in the spec of 3-month T-bill monthly averages (FRED `DGS3MO`); a trade uses the previous month's.
- **Dividends:** `massive_dividends.v1` rows with `declaration_date` on or before the quote date and ex-date after it, up to expiry. A row without a declaration date is ignored; both that and a ticker with no rows (zero dividend, `dividend_unknown`) are counted. Coverage is unverified.
- ORATS's own rate and dividend inputs are not stored, so they may disagree with these. The clamp bounds the error; the report prints `|V - mid| / half-spread` by year.

**Where `smvVol` is.** The chain pull lists request it, but `option_chains` has no such column (`engine/data/normalize/n_chains.py` drops it). A read-only extractor takes it from cached raw ORATS strike payloads into an experiment table. Historical coverage is unverified and is reported by year **before any outcome is read**.

**Fallback smile.** Bid and ask IVs are **not stored** either (`iv` is the side's mid IV only). So the fallback inverts the stored bid and ask **prices** with the same pricer into `iv_bid` and `iv_ask`. Per ticker, date and expiry, fit `iv = a + b*k + c*k^2`, `k = ln(K/S)/sqrt(T)`, by `numpy.polyfit` on `(iv_bid + iv_ask)/2` with weight `1/max(iv_ask - iv_bid, 0.02)`, using out-of-the-money quotes (puts `K <= S`, calls `K > S`) whose two prices invert. It needs at least 5 strikes, one or more on each side; outside the fitted `k` range the edge value holds; a non-positive fitted IV voids the fit (all mine). One source serves a whole expiry on a date: `smvVol` only if every needed strike has it, else the smile; the two are never mixed. No fit: `no_fair_value`. Source share and the smile-minus-`smvVol` gap where both exist are printed by year.

**Concession** (constants mine, pre-registered, never tuned on results). With `s = (ask - bid)/mid` (a zero mid counts as capped):

`c = min(1, 0.25 + 0.50 * min(s/0.50, 1) + 0.15 * thin_size + 0.10 * thin_oi)`

Buy at `V + c*(ask - V)`; sell at `V - c*(V - bid)`. `thin_size`: size on the side hit (ask size to buy, bid size to sell) under 10 contracts. `thin_oi`: open interest under 100. A null field adds nothing and is counted `liquidity_unknown`; zero is not null. The schema says size and open interest exist **only on chains pulled from 2026-09**, so on 2023-2025 F is probably spread-only; the report states the share. For scale, with V at mid F equals ladder alpha `0.5 - c/2`: 0.375 for the tightest quotes, 0.125 at `s >= 0.50`, harsher than STR-THRU's 0.423 breakeven.

F applies at entry and, sides flipped, at exit. An unusable exit quote (missing or crossed) keeps EXP-188's worst-case rung. A zero-bid quote is clamped to `[0, ask]`.

**Entry guard** (structure level). Skip when `debit_F - debit_fair > 0.25 * debit_fair` (mine), or the quantity-weighted sum of leg spreads exceeds `debit_F`, or `debit_fair` or the mid debit is not positive. A skip is a non-trade at P&L 0, counted by reason, in every arm and control and in every fill view. It never applies to exits.

**Where F is used:** the `exp_pnl_sim` entry cost, the E width choice, every F-scored P&L and account, and the N1/N2 target. The simulated exit stays the engine's zero-concession model value: a stated limit.

**Mid and harsh** are reported beside F: the F-decided taken set re-marked at `marks_mid` and at EXP-188's harsh marks. One reference row, **P0 at mid** (mid selection and mid marks), keeps comparison with EXP-188 and EXP-189.

**Calibration from real fills (only if the user provides them).** Per fill: time, contract, side, quantity, price, and bid and ask at order time; V from the same pricer. Observed `c` is `(price - V)/(ask - V)` for a buy and `(V - price)/(V - bid)` for a sell (denominators under $0.005 skipped). Fit the base and spread constants by least absolute deviation with the 0.50 spread scale fixed; each thin penalty is the median residual gap. At least 100 fills, and 30 flagged for a penalty, else the constant stays. Combo orders without leg prices fit one structure-level multiplier on `c`. The guard fraction is not calibrated. New constants mean a new spec version and PLANNED row **before** a new run, never a re-scoring chosen by its result.

## 3. Controls and arms

Machinery is EXP-188's production-exact code used as a library (entry rule, per-structure trailing cutoffs, universe filters, exits). Trailing cutoffs use the `trailing_cutoff` window: 6 months strictly before the month, at least 100 of the arm's own prior out-of-sample scores.

**Controls**, recomputed in the same run, F-priced:

- **P0:** production: board widths at the pinned commit, M0, the native full-history `ResidualPool`, production gate.
- **E** [D11]: EXP-189's recommended **arm E**: per event and structure, the `k` in {0.5, 0.67, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0} times production's width that maximises `exp_pnl_sim` (feasible widths; ties nearest 1, then smaller), gated on E's own score history; the F guard applies to the chosen width. `size_width_*` is imported unchanged; if it cannot take F's cost, a new module repeats the rule with a parity test at mid. EXP-189's recommendation was made having seen 2025, under mid marks; E is recomputed here under F.

**Distribution draws** [D12]. A per-event `QuantilePool` wraps production's pool and exposes `draw` and `before`. `draw` calls production's draw (same M0 forecast, count, random stream), takes each drawn `err_move`'s rank `u = (rank + 0.5)/n` (ties in draw order; an empty draw passes through) and returns `Qinv_x(u) - forecast`. `err_crush` is unchanged, keeping the move/crush rank dependence. `pnl_sim.expected_pnl` is unmodified; the fast path must match within 1e-9.

| Arm (each for both `x`) | Strike width | Entry rule | Must beat |
| --- | --- | --- | --- |
| W | `forecast_params(name, Q_x(tau))` | production (M0, residual pool) on the new geometry | P0, E, LW |
| D | production | `exp_pnl_sim` with `QuantilePool` | P0, E |
| WD | as W | `QuantilePool` | P0, E, LWD |
| F-twin | production | P0's gate and `R` at or below its trailing median [D13] | P0, E (twin) |
| ST (straddle) | STR-THRU | buy when `P(move > implied_move)` under `x`'s full distribution is in its trailing top 20% | ST0 |
| N1 (selection) | production | top 20% by predicted return (section 4) | N2, P0, E |

Controls only: **LW** uses `forecast_params(name, kappa * M0)`, `kappa` the median `Q_x(tau)/M0` over the prior 12 months of strictly earlier events (at least 100), gated as W; **LWD** is LW with `QuantilePool`. They separate a level shift from information. **ST0** is ST with B0's distribution. Straddles use EXP-188 Amendment 5's universe; `G_PROD` is printed labelled "not production-faithful". `R = (q75 - q25)/q50` from `x`'s quantiles; its other half is printed. `WIDTH_MIN`/`WIDTH_MAX` declines apply as in production.

**tau** [D14]: chosen walk-forward, one per `x` and family (centre-peaked: CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5; TWIN: TWIN-P, TWIN-P5) from **{0.10, 0.25, 0.40, 0.50, 0.60, 0.75}**. Objective: arm W's family account return under F on the selection window, using only trades exited before the evaluation year. **2024 uses the best on 2023H2; 2025 the best on 2023H2-2024.** Ties: nearest 0.50, then lower. A level with under 50 taken trades in the window is ineligible; none eligible: 0.50 (both mine). LW, WD and LWD reuse W's `tau`. 2023H2 has no earlier window, so it runs at 2024's `tau`, labelled **in-sample**. **Every level's results are printed per year**, per family.

## 4. Learned selection models (N1-N4)

**Trade-everything pricing** [N3]. Entry and exit are priced for **every** candidate (event x structure) of each geometry (P0, E, W, LW; D, F-twin and N use P0's; WD uses W's, LWD LW's), and for every straddle, taken or not. This supplies the training rows, and each geometry's "trade every event" P&L is reported under F, mid and harsh, with and without the guard.

**Models** [N1]: two per feature set, **one pooling the five centre-peaked structures, one the TWIN pair**. `HistGradientBoostingRegressor(loss="squared_error")`, because ranking needs expected return. One fixed configuration, never selected: `max_leaf_nodes` 7, `min_samples_leaf` 300, other settings as part 1's grid (mine). Target: the candidate's return on debit under F, `(exit_F - debit_F)/debit_F`, worst-case exits included, no clipping.

| Feature | N1 | N2 |
| --- | --- | --- |
| `x`'s seven calibrated quantiles (Q_raw: the raw ones) | yes | no |
| Band-width ratios `R` and `(q90 - q10)/q50` (`zero_median`: NaN, counted) | yes | no |
| M0's out-of-sample point forecast [D1] | yes | yes |
| Option-implied move | yes | yes |
| Entry cost: `debit_F/spot` and debit over the widest strike gap | yes | yes |
| Breakevens: lowest and highest expiry-payoff zero crossing net of `debit_F`, signed percent from spot, and their count | yes | yes |
| Structure id (one-hot) | yes | yes |

**N2** [N2] is N1 without the quantile features: same rows, folds, settings and trade rule. **N1 minus N2 is the value of the distribution**, reported with bootstraps whether or not N1 passes.

**Rows and walk-forward** [N4]. Training rows pass production's universe filters and the F guard and have F entry and exit prices. Quantile features are **out-of-sample only**. N1(Q), N1(Q_raw) and N2 train on identical rows: those where both quantile arms, M0's out-of-sample forecast and F pricing all exist. The model scoring year Y trains on trades exited before Y. Part 1's walk-forward from Y0 is what gives the 2024 model more than six months of training; the realised first month, row counts and years are reported (no range is claimed before coverage is counted). Under 2,000 rows in a family: that year is `not_testable` (mine).

**Trade rule.** A candidate is taken when its prediction is at or above the trailing top-20% cutoff of that model's own prior predictions **for that structure**. It replaces the `exp_pnl_sim` gate; production widths, universe filters and the F guard stay. Cutoff history is out-of-sample: the year-Y model scores only year Y, so 2023H2 and early 2024 need the 2023 model, trained on pre-2023 rows.

## 5. Outputs, accounts, success rule

**Outputs**, by structure, family and year, under F with mid and harsh beside: P&L per candidate and per taken trade, trades, declines and guard skips, **breakeven alpha** (the taken set re-marked down the alpha ladder), quarter and ticker bootstraps against each control, the per-level `tau` tables, trade-every-event P&L, and F's diagnostics (IV source, `|V - mid|`, liquidity share, concession quantiles).

**Accounts:** $200,000 cash-secured, the EXP-188 / EXP-189 policy (`margin.simulate` through `size_pnl_account`: cap 0.66 of current equity, `target_share` 0.25, expensive first, short puts secured at strike x 100; straddles at their debit), with entry cost and marks under F. Accounts: each structure, the centre-peaked five, the TWIN pair, all seven, straddle.

**Success rule** [D15]. W, D, WD and N1 are judged per family; F-twin on the twin account; ST on the straddle account. A candidate passes iff:

- its account return under F beats **every** control in its "must beat" row in **each** of 2023H2, 2024 and 2025 (fresh account per period);
- its 2025 breakeven alpha is no more than each of P0's and E's (ST: ST0's) plus 0.05;
- against each of P0 and E, in 2025 stand-alone structure accounts it loses at most 1 of 5 centre-peaked, and neither twin.

The margin and guards are judgement calls. A pass is a **proposal**; a board change is a separate user decision, with the DYN-SV chooser review (EXP-188 Decision 3). ST is evidence only. A pass on an `x` that failed Stage A is reported but is not a proposal (mine). Mid and harsh never decide or break ties.

**Periods used for selection** (printed in the report):

| Quantity | 2023H2 evaluation | 2024 | 2025 |
| --- | --- | --- | --- |
| Quantile grid (part 1) | years before 2023 | before 2024 | before 2025 |
| `tau` | **2023H2 (in-sample)** | 2023H2 | 2023H2-2024 |
| N1/N2 training | before 2023 | before 2024 | before 2025 |
| Trailing cutoffs, kappa | strictly earlier | strictly earlier | strictly earlier |
| M3 identity, feature set, grid values | v14 used 2023-2024 | same | chosen on 2023-2024 |
| F's constants | fixed here | same | same |

For W and WD the 2023H2 condition is in-sample, so it adds little evidence (it can only remove a pass). **2025 is the clean test** for every selection. It is not blind for design: EXP-188 and EXP-189 results from 2025, including E's recommendation, were seen. Independent confirmation is only the forward period.

**Multiplicity.** Per `x`: W, D, WD and N1 for two families (8), F-twin and ST, so **10; 20 candidates** over Q and Q_raw, against 8 in #2869, on top of Stage A's two tries. No correction. Chance passes are likely (the candidates are correlated, so no rate is claimed); the report lists every candidate, passed or not, and names no best one.

## 6. Failure semantics (R1-R6)

| Case | Result |
| --- | --- |
| R1 no Stage A pass receipt, changed spec or no PLANNED row | refuse (`stage_a_not_passed`, `spec_hash_mismatch`, `planned_row_missing`) |
| R1 bound inputs, rate table or dividend snapshot do not bind | refuse `EXPERIMENT_IDENTITY_CONFLICT` |
| No fair value, crossed or missing entry quote | undetermined, counted; the event leaves that comparison for candidate and controls |
| Guard skip, width declined, ladder too coarse | non-trade at P&L 0, counted; the event stays |
| Unusable exit quote | worst-case rung, labelled; also in the training target |
| Quantiles unscored, `zero_median` | arms needing them are undetermined (in N1, a ratio is NaN, not a drop); counted |
| Missing implied move | leaves ST and ST0; NaN in N1/N2 |
| Cutoff or kappa history too short | undetermined, counted |
| N fit fails or has too few rows | that family-year `not_testable`; the candidate cannot pass; no retry |
| Breakeven alpha with no crossing, or no trades | 0.0 if always positive, 1.0 if always negative; no trades: fails |
| R2 cost | quarter chunks, cached leg fair values per run; over-limit refuses `RESOURCE_LIMIT_EXCEEDED` |
| R3, R4, R5 | fixed seeds; draw key `event_id` shared across arms; atomic immutable receipts bound to inputs; arm ids hash `x`, rule, `tau` source and F's constants |
| R6 | fits, `tau`, pools, cutoffs and dividends use only what was available by the decision date; a month's decisions are written before its labels and exits are read |

Pricing every candidate with a tree is the new cost; the smoke measures it before any full run. Progress logging follows AGENTS.md.

## 7. Limits

- F is a model of fills, not a measurement; until real fills calibrate it, its constants are assumptions.
- Coverage of `smvVol`, dividends and liquidity is unverified; a source mix that changes by year is a confound the by-year diagnostics expose but do not remove.
- N1 trains on strongly heavy-tailed returns, correlated within event and season.
- Seven correlated structures and one clean year are weak evidence.

## 8. Split list (code slices, only after a Stage A pass)

Non-test code lines / test lines / doc words. F1-F3 may start earlier on fixtures, since F1's coverage count reads no outcome.

| Slice | Scope | Size | After |
| --- | --- | --- | --- |
| F1 | `smvVol` extractor, coverage report | ~120 / ~130 / 60 | this PR |
| F2a | American pricer, rate and dividend inputs, IV inversion | ~170 / ~190 / 60 | F1 |
| F2b | Smile fallback and source choice | ~100 / ~130 / 40 | F2a |
| F3 | Concession, guard, three fill views, diagnostics | ~150 / ~180 / 60 | F2b |
| Q7 | `QuantilePool`, fast-path equivalence | ~110 / ~140 / 40 | A passes |
| Q8a | W, LW, D, WD, LWD | ~170 / ~170 / 60 | Q7, F3 |
| Q8b | `tau` selection and per-level tables | ~100 / ~130 / 40 | Q8a |
| Q8c | Control E under F | ~120 / ~150 / 40 | Q7, F3 |
| Q9 | F-twin, ST, ST0 | ~140 / ~150 / 40 | Q7, F3 |
| N-a | Trade-everything table and features | ~160 / ~180 / 60 | Q8a, Q8c |
| N-b | N1/N2 walk-forward, cutoffs, receipts | ~160 / ~180 / 60 | N-a |
| Q10a | Accounts, breakeven, success and periods tables | ~170 / ~170 / 60 | Q8b, Q9, N-b |
| Q10b | Stage B driver, report, smoke, runbook | ~130 / ~100 / 100 | Q10a |
