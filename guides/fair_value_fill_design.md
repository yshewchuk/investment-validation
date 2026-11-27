# Fair-value marks and spread-aware fills, model F (shared design)

**Status: design only.** A shared experiment-side module for EXP-190 Stage B (`guides/size_quantile_move_design.md`, section 6) and EXP-189 Amendment 1 (its owner's work); no EXP-188 or EXP-189 file or spec changes here. A consumer adopts F in its own spec and pre-registration. Content: user decisions of 2026-11-25; rules marked "mine" are my judgement calls inside them.

**Carry and dividend amendment (2026-11-25, user decision).** Fair value uses ORATS's options-implied borrow as the carry rate, and ORATS's dividend inputs, to reprice consistently with the inputs behind `smvVol` (an aim: we have not confirmed which inputs ORATS used). It replaces the zero-dividend approximation (sections 3, 4 and their consequences in sections 7-12). The re-ingest tool (version 4) carries the fields. **New slice F7** consumes them; F1, F2 (#2901) and F3 stay as they are, pricing with carry equal to the risk-free rate and no dividend. F5 now follows F7, so no `fvf_id`, and so no consumer registration, predates it.

## 1. Why

- **EXP-105 diagnostics** (`reports/exp-105_log_diagnostics.md` §8.5, untracked, supervisor-local). STR-THRU's widest relative-spread quintile (median 0.597) earned +9.87% per trade and 48.9% of summed returns. The whole book earned +4.04% at mid and -1.42% at alpha 0.40; capital-weighted breakeven alpha 0.423.
- **EXP-133** (`guides/exp133_structure_search.md` §4, §6). Per-event search over strikes harvests a vendor mid surface's no-arbitrage violations: 43.6% of its TWIN-P5 width-and-anchor arm's entries violated it (free search 44.6%, fixed incumbent 1.0%). Its `zero_cost` bug showed that a refusal applied at one fill price compares different books.

## 2. Module boundary

- New modules `experiments/fair_value_*`. Imports: numpy, pandas, and `engine.fills.MIN_MEANINGFUL_COST`, plus `pyarrow.parquet` for footer reads (Amendment 1, item 5). `engine/fills.py`, `engine/structures.py`, `engine/pnl_sim.py`, `engine/chain_index.py` and the production normalizer do not change. No sealed v14 closure imports F; the first slice tests both closure hashes (part 1 section 3) unchanged.
- F prices **resolved legs** only; the consumer resolves them and owns timing, gating and sizing.
- **Tool check.** Adding QuantLib (not installed) would change the pinned environment sealed runs record; one American put pricer on numpy suffices, with no IV solver. `engine.pnl_sim.black_scholes_put` and the `engine/v2/scoring/stages.py` helper are European with zero rates, and `pnl_sim` is in the v14 closure, so F is a new pricer.

## 3. Inputs

| Input | Content | Source |
| --- | --- | --- |
| Quotes | one row per (ticker, `obs_date`, expiry, strike, right): `bid`, `ask`, `iv`, `spot`, `quote_repaired`, `open_interest`, `bid_size`, `ask_size`, `chain_kind`. **Every listed row of each requested expiry**, not only the traded legs | pinned snapshot of curated `option_chains` |
| Vol table | `smvVol` per (ticker, `obs_date`, expiry, strike); no `right`, so both rights share it | experiment re-ingest of the raw ORATS cache, `quotes` table |
| Rate table | `iRate5wk`, `iRateLt` per (ticker, `obs_date`) | the same re-ingest, `rates` table (ORATS cores) |
| Carry and dividend inputs (F7) | `borrow30`, `borrow2yr`, `divAmt`, `divDate` price; `annIdiv`, `annActDiv`, `assetType` are diagnostics only | further columns of the same `rates` table |

**ORATS definitions** (orats.com/docs/definitions, Summaries and Cores tables). `borrow30`: "implied hard-to-borrow interest rate at 30 days to expiration given options prices put call parity"; `borrow2y` (cores name `borrow2yr`): the same at two years. `annIdiv`: "annual implied dividend given options prices put call parity". `annActDiv`: "annual dividend from the next year of expected dividends". `divAmt`: "dividend amount". `divDate`: the next dividend date. `assetType`: 0 ETB_NO_DIV, 1 HTB, 2 HTB_DIV_PAYING, 3 ETB_DIV_PAYING, 4 INDEX, 5 ETF. ORATS does not define borrow's relation to the risk-free rate. **Our reading, unconfirmed:** borrow is the net financing rate on the stock, after the loan fee. The supervisor checked that a real `hist/cores` body has these fields.

**Data facts (supervisor-checked 2026-11-25, not re-read here).** `engine/data/normalize/n_chains.py` drops `smvVol` from curated `option_chains`; the raw cache has it per strike. Open interest and both sizes exist only on `chain_kind=fetch` rows (2023-01-03 to 2025-12-31), about 75-80% of rows; `eod`, `t14` and `c2` rows have none, so earlier history has none.

**The re-ingest tool** (`tools/orats_smoothed_vol.py`) writes no production store. F binds the content hash of its `manifest.json` (written last) and of every table file read. A missing manifest, or one that reports unread payloads, refuses (Amendment 1). F reads only the columns above; extra columns and manifest keys are ignored, except that F7 also reads `units` (below).

**Units.** `spot`, strikes and quotes are dollars per share; vols and rates are annualised decimals. The raw units are confirmed from the re-ingest and pinned in each consumer's spec before registration, using no outcome. After conversion, a rate value outside [-0.05, 0.20] is by default missing (Amendment 1, item 3); a non-finite or non-positive vol is missing; a bound vol table whose median `smvVol` is outside (0.05, 5) refuses `vol_unit` (bounds mine).

**Units of the `rates` table as stored** (the decimals above are what F computes in, after conversion). In `hist/cores` the rates and borrow fields are **percent** (supervisor-checked: `borrow30` 5.3754 beside an `iRate5wk` of 5.53); the Summaries endpoint, which F does not read, uses decimals. The tool converts nothing and records each field's unit in the manifest's `units`; F divides percent by 100 explicitly. Dividend fields are taken as dollars per share (our reading, unconfirmed). F7 refuses `input_unit` if a bound manifest does not label borrow `percent` and `divAmt` `dollars_per_share`. After conversion, a borrow outside [-1.00, 0.20] is missing and counted (bounds mine; the floor is low because hard-to-borrow names can be strongly negative).

## 4. Fair value

Per quote row, with `S` the row's curated `spot` (the one strike resolution uses; the re-ingest's `stockPrice` is not used), `T = max(expiry - obs_date, 0) / 365` in calendar days, and `V` the fair value, **clamped to [bid, ask]**:

- `T = 0`: intrinsic value, then clamped.
- **Calls:** European Black-Scholes with carry `b` where that is the American value: no `dividend_applied` (the table below, expiry day included) and `b >= r >= 0`. Otherwise (F7) the American tree below, since early exercise can then pay. F2 and F3 as built take `b = r`, no dividend, and price every call as European, even with `r < 0`, where this rule would use the tree and the European value can be too low.
- **Puts:** American, by a Cox-Ross-Rubinstein tree with **200 steps** (mine), vectorised over rows, early exercise at every step. Vol is floored at 0.01; a tree probability outside (0, 1) is `no_fair_value`.
- **Rate `r`:** `iRate5wk` up to 35 calendar days to expiry, linear to `iRateLt` at 365, flat beyond; if one of the two is missing, the other is used (mine). The row is the ticker's for `obs_date`, else its latest earlier one within 7 calendar days; none: `rate_unknown`.
- **Carry rate `b` (F7):** ORATS borrow, from that same row. By the leg's calendar days to expiry: `borrow30` up to 30, linear to `borrow2yr` at 730, flat beyond; if one is missing or out of range, the other is used (interpolation mine). The underlying drifts at `b` (forward `S*e^(bT)`) and cash flows discount at `r`: in the tree `p = (e^(b*dt) - d)/(u - d)`. This is a yield of `r - b`, and rests on the unconfirmed reading in section 3 (using it this way is mine). **Both missing: `b = r`**, flagged `carry_fallback` and counted. No further search back.
- **Dividend (F7):** at most one discrete cash dividend, `D = divAmt` on `divDate`, read as the ex-date (unconfirmed), at `t_D = (divDate - obs_date) / 365` years (calendar days, as `T`; the tree's node times `t` are years from `obs_date` too), by the escrowed method: the tree runs on `S - D*e^(-r*t_D)`, and exercise values before `t_D` add back `D*e^(-r*(t_D - t))`. `k` and the fit set keep `S`. The first case that holds decides:

| Case | Dividend |
| --- | --- |
| `divAmt` missing or negative | none; `dividend_fallback`, counted |
| `divAmt = 0` | none (known) |
| `divDate` missing, or not after `obs_date` | none; `dividend_fallback`, counted |
| `divDate` after expiry | none (known) |
| `D*e^(-r*t_D) >= S` | none; `dividend_fallback`, counted |
| otherwise | `D` at `t_D`, flagged `dividend_applied` |

- **Not priced (mine):** `annIdiv` and borrow are both implied from put-call parity, and ORATS does not say what each solve holds fixed, so using both could count one parity gap twice. `annActDiv` gives no dates. Whether borrow was itself solved with `divAmt` and `divDate` is also unconfirmed. At `T = 0` neither `b` nor a dividend is used or flagged.

Tests (F7): with `b = r >= 0` and no dividend, values equal F2's; a call that takes the tree is never below the European value computed on the same lattice (same inputs and steps, no early exercise; Black-Scholes is not the comparator, since the tree's discretisation error can fall below it). F2's tests: published American put values; at zero rate the put tree matches the European put `engine.pnl_sim.black_scholes_put` within the tree's discretisation error, and the call matches that put by parity, `C = P + S - K`.

**Vol source, one per (ticker, `obs_date`, expiry)**, only when `T > 0` (at `T = 0` no vol is used, no smile is fitted and the source is `intrinsic`). It is decided from the whole listed expiry, so a row's `V` never depends on which structure asked. The **fit set** is the out-of-the-money rows (puts `K <= S`, calls `K > S`) with finite positive `iv`, `0 <= bid <= ask`, positive mid and `quote_repaired` false.

1. If the fit set is non-empty and every strike in it has `smvVol`, the expiry uses `smvVol`. A row outside the fit set with no `smvVol` is `no_fair_value`.
2. Otherwise the **fallback smile**: weighted least squares `iv = a + b*k + c*k^2`, `k = ln(K/S)/sqrt(T)`, on the fit set's stored `iv`, by `numpy.polyfit` with residual weight `1/max((ask - bid)/mid, 0.02)`. It needs at least 5 distinct strikes, with one or more below and one or more above `S`. Outside the fitted `k` range the edge value holds. All rows of the expiry use the smile.
3. A fit that cannot be made, or a non-positive fitted value inside the fitted range: `no_fair_value` for every row of the expiry (fit rules mine).

Sources never mix within an expiry. The smile only approximates `smvVol` (stored `iv` follows ORATS's conventions); their gap is printed.

## 5. Fill

With `s = (ask - bid)/mid`:

`c = min(1, 0.25 + 0.50 * min(s/0.50, 1) + 0.15 * thin_size + 0.10 * thin_oi)`

**Buy** at `V + c*(ask - V)`; **sell** at `V - c*(V - bid)`. The same rule applies at entry and, with sides flipped, at exit.

| Term | Rule (constants mine, pre-registered here, never tuned on results) |
| --- | --- |
| `thin_size` | 1 when the size on the side hit (`ask_size` to buy, `bid_size` to sell) is under 10 contracts, else 0 |
| `thin_oi` | 1 when `open_interest` is under 100, else 0 |
| **Unknown liquidity** | a null size sets that side's `thin_size` to **0.5**; a null open interest sets `thin_oi` to **0.5**. Zero is a known value and is thin. Counted per field |

**Repaired quotes.** A `quote_repaired` row is a crossed quote collapsed to `bid = ask = P` (`engine/data/validate.py`), so its zero spread is false, and no concession on it could change a price. Its **effective quote** is `bid = 0.75*P`, `ask = 1.25*P`: relative spread 0.50, the cap (width mine). Wherever this file clamps `V`, computes `s` or fills, or sums spreads, at entry and at exit, a repaired row uses its effective quote; its mid stays `P`. Statuses are read from the stored quote, so `P <= 0` stays `unpriceable`.

0.5 favours neither rows with sizes nor rows without. Missing sizes follow the pull type, hence the era, so two **sensitivities** re-mark the same taken set, entry and exit, with every unknown set to 0 and to 1. They never decide a verdict.

**Scale.** With `V` at mid, F equals ladder alpha `0.5 - c/2`: 0.375 for the tightest liquid quote, 0.125 at `s >= 0.50`, 0 when also thin on both counts. All are below EXP-105's breakeven 0.423. With `V` away from mid, one side can fill **better than mid** (at `V = ask`, `c = 0.25` sells at alpha 0.75). A search could favour such legs, so F reports their share by side, and a third sensitivity, **mid-floored F** (a buy never below mid, a sell never above), re-marks the same taken set.

**Replaced for the next constants (user's decision, 2026-11-27).** This section is F as built. Section 9's mid-anchored **model A** replaces its fill and is not built yet.

## 6. Structure level

Legs have side and positive quantity; every sum is per share and quantity-weighted. A debit is buys minus sells:

- `debit_mid` at each leg's mid, `debit_fair` at `V`, `debit_F` at F's fills; always `debit_F >= debit_fair` (`V` is inside the quote, so each buy fills at or above its `V` and each sell at or below).
- `spread_sum = sum(qty * (ask - bid))`.
- Exit `proceeds_F`: what closing at F's fills receives (opening longs sold, opening shorts bought).
- Return on debit: `(proceeds_F - debit_F)/debit_F`. No clipping. Negative proceeds are kept and flagged `exit_negative`; any floor is the consumer's pre-registered rule.

**Entry guard.** For debit structures. The first condition that holds gives the skip reason:

1. `not_a_debit`: `debit_mid` or `debit_fair` is at most `MIN_MEANINGFUL_COST`.
2. `fill_over_fair`: `debit_F - debit_fair > 0.25 * debit_fair` (fraction mine).
3. `spread_over_debit`: `spread_sum > debit_F`.

The guard is evaluated **once**, at entry, and never at exit. A skip is a non-trade, and the same skip applies to every fill view, sensitivity, arm and control of the consumer's comparison.

This is the guard as built. With model A (section 9), `fill_over_fair` becomes `fill_over_mid`.

## 7. Outputs

- **Leg table**, one row per quote row: `status` (`ok`, `unpriceable`, `no_fair_value`, `rate_unknown`), `V`, vol, source (`smvvol`, `smile`, `intrinsic`), rate, clamp side, `s`, `fill_buy`, `fill_sell`, `c_buy`, `c_sell`, thin, unknown and repaired flags, plus the three sensitivity fill pairs (F7 adds `b` and the three section 4 carry and dividend flags). A structure is a signed dot product over these rows.
- **Entry result:** the three debits, `spread_sum`, guard result, leg flags.
- **Exit result:** `proceeds_F`, flags, or `unpriceable`.

**Statuses**, first match wins: `unpriceable` is a missing or non-finite quote, `bid < 0`, `ask <= 0` or `bid > ask` (a zero bid with a positive ask is usable); then, only when `T > 0`, `rate_unknown`; then `no_fair_value` (section 4's vol and tree rules, or a missing or non-positive `spot`; at `T = 0` only the latter applies).

**Entry and exit differ.** At entry, any leg not `ok` makes the structure `undetermined` (a missing fact, not "no"), known at the decision. It has no entry fill and is not guard-tested, and it is `undetermined` in every fill view and sensitivity alike. Whether the event then leaves the whole comparison is the consumer's pre-registered rule, applied alike to candidates and controls. At exit, a held trade **never leaves the population**: a leg with a usable quote but `no_fair_value` or `rate_unknown` takes `V = mid`, flagged `exit_fair_mid`, and pays the concession. An `unpriceable` leg makes the exit `unpriceable`; the consumer then applies its own pre-registered, labelled conservative exit (EXP-188 design section 6, rung 3: the structure's worst-case payoff).

## 8. Use by a consumer

F is used in **selection and scoring alike**:

- `exp_pnl_sim` takes `debit_F` as its entry cost (`expected_pnl` already takes `entry_cost` as an argument, so `engine/pnl_sim.py` is unchanged). This also drives any width choice that maximises it.
- A selection model's target is the return on debit above, including the consumer's conservative exits.
- Every F-scored P&L, account and success test uses `debit_F` and `proceeds_F`.
- **Mid and harsh are still reported**, on the same taken set, computed by the consumer's existing code (`mid` is `FillModel(0.5)`; "harsh" keeps that experiment's own definition). They never decide or break a tie where F is the declared basis.

**Diagnostics**, by year, `chain_kind` and entry versus exit: vol source and status shares; smile minus `smvVol`; `|V - mid| / half-spread` (effective quote; rows with zero half-spread excluded and counted) by right (a wrong carry or dividend input would show as a put/call difference) and, from F7, by `assetType`; from F7, `carry_fallback`, `dividend_fallback` and `dividend_applied` shares, with `annIdiv` and `annActDiv` coverage; clamp, better-than-mid, unknown and repaired shares; `c` quantiles; guard skips by reason; `exit_fair_mid` and `unpriceable` counts; and the share of entries whose structure value violates a no-arbitrage payoff bound at mid and at `V`. Input coverage (F1, Amendment 1), then status, source and liquidity coverage (F5), on entry rows is printed **before any outcome is read**.

## 9. Recalibration from the user's real fills (only if supplied)

**Amended 2026-11-27** ([#3095](https://github.com/yshewchuk/investment-validation/issues/3095)). Model A, price space, the B1 window and a new constants id for new constants (B2) are the **user's decisions of 2026-11-27**. Judgement calls name their owner; unlabelled grid and minimum rules carry over from the earlier text (mine). The fills file is private: never commit it, fitted constants or fill-level values beyond the aggregates below.

**Not built.** The rest of this document, including everything that rests on `V`, describes F as built. The follow-up is **F6c: model A fit plus leg fills** ([#3104](https://github.com/yshewchuk/investment-validation/issues/3104), reworked). The merged F6a and F6b code is still the earlier fit, anchored on fair value with a one-tick floor, both now superseded: its constants and constants id are not model A's. The `fair_value_fit` and `fair_value_solve` rows of [`experiments/README.md`](../experiments/README.md) are the contract of that built code.

**Model A, mid-anchored.** `c = min(1, base + spread*min(s/0.50, 1) + thin_size_w*thin_size + thin_oi_w*thin_oi)`, with section 5's `s` and thin flags. On the effective quote a **buy fills at `mid + c*half_spread`** and a **sell at `mid - c*half_spread`**. The fill has **no fair value `V`, no one-tick floor and no tick rule**.

**Guard (supervisor's judgement).** `fill_over_fair` becomes `fill_over_mid`, the same 25% rule against the mid debit: `debit_F - debit_mid > 0.25 * debit_mid`.

**Why (user's near-close fills; aggregates from the supervisor's private run, 2026-11-27).** Fair value priced from the prior day's `smvVol` sits at the quote edge on 45% of fills. Every fair-value form fit worse than the mid anchor: mean absolute error 0.96-1.01 half-spreads against 0.90. The mid anchor's average bias is about 0; current F averages 0.13 too harsh. The spread weight fits to 0, but only 17 fills have a spread above 20%, hence the variant below.

**Fills used (B1 window).** Each fill is matched to a quote snapshot, and only fills within 60 minutes of it enter (two-sided, 60 included: supervisor's judgement), so the snapshot's quote can be later than the fill. A fill outside the window, with an unusable quote or with an effective half-spread of at most 1e-9, which leaves `x` undefined, is excluded and counted (that threshold: F6a owner's judgement).

**Price-space loss.** On the effective quote, `x = (price - mid)/half_spread` for a buy and `(mid - price)/half_spread` for a sell (+1 is the side hit). Observed `x` is not clipped, so price improvement stays. The fit minimises the mean `|x - min(1, c)|`, each fill weighted equally, whatever its quantity. The earlier `c`-space loss and its $0.005 denominator rule are removed.

**Fit.** A deterministic grid, 0.00 to 1.00 in 0.01 steps; losses within 1e-9 tie.

- **Base and spread**, jointly, on unflagged fills (both flags known and 0). Ties: the largest base plus spread, then the largest base (F6b owner's judgement). Under 100 such fills both keep F's current values, without refusing, and the penalties are still fitted (F6b owner's judgement).
- **Each thin penalty** (supervisor's judgement): with base and spread fixed, the grid value with the least same loss on fills with that flag set and the other known and 0; ties to the larger. It is fitted only with at least 30 such fills; otherwise the current value stays. Fills with both flags set or any unknown enter no fit.
- **Not calibrated:** the spread scale, guard fraction, thin thresholds and unknown value 0.5. A multiplier fit for spread orders without leg prices is not built.

**Constants id and variant.** Model A's constants get a **new constants id** (B2). The one named variant is **`spread050`** (spread = 0.50), a user-requested wide-leg sensitivity: it changes only the spread weight and leaves the primary id unchanged. A consumer needs a new spec hash and PLANNED row before a new run, pre-registers `spread050`, states the fills' date range and any overlap with its evaluation periods, and never re-scores an old run with constants chosen by that result.

**F6c's slices, failure semantics and open questions** are in section 12.

## 10. Failure semantics (R1-R6)

| Rule | Case | Result |
| --- | --- | --- |
| R1 inputs | snapshot, manifest or table hash does not bind; manifest absent or reporting unread payloads; required column absent; duplicate quote, vol or rate key; vol unit check fails | refuse `EXPERIMENT_IDENTITY_CONFLICT` with a `details.reason`; no partial output; nothing imputed except the rules written above |
| R1 per row | missing vol, rate or quote | section 7 statuses, counted; never silent or zero |
| R1 per row (F7) | missing carry or dividend input | section 4's fallbacks (`b = r`; no dividend), flagged and counted; not a status. A column absent from the bound table, or section 3's `input_unit` refusal, is `R1 inputs` |
| R2 resources | American tree (puts, and from F7 the calls routed to it) on unique rows | quarter chunks, each row priced once per run, under the consumer's `bounded_run` limit; over it refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report. Smoke first |
| R3 determinism | no randomness or clock | values never depend on row order, chunking or requested structures |
| R4 replay | stored leg tables | atomic writes with a receipt binding input hashes and F id; replay read-only; changed input refuses |
| R5 identity | `fvf_id` | content hash of F's source files and constants (pricer steps, rate, carry, dividend and smile rules, concession, guard); a consumer's spec binds it, and a mismatch refuses |
| R6 causality | all values | a row uses only its own `obs_date` quotes, vols and spot, and a rate row dated no later. Entry values never read exit-date rows. "Earlier" means data dates, never `computed_at`. Section 9 is the only fitted part |

Progress logging follows AGENTS.md.

## 11. Limits

- Uncalibrated constants are assumptions; no direction of error is claimed.
- F leaves cross-strike no-arbitrage violations (measured by the diagnostic).
- The concession ignores order size and spread-order fills.
- The `exp_pnl_sim` simulated exit stays European, zero-rate with **no concession**: optimistic against F-scored exits, and rank gating does not absorb differences between names.
- `smvVol`, rate, carry and dividend coverage is unverified; a changing source mix is a confound.
- The borrow reading, the dollar unit and the ex-date reading are unconfirmed.
- Only the next dividend is priced; an expiry spanning two omits the later one, uncounted.
- We have not verified that historical `divAmt` and `divDate` are what was known on `obs_date`; a later revision would be look-ahead that data dates cannot detect.

## 12. Split list (code slices; none starts until this design merges)

Estimates: non-test code lines / test lines / doc words, excluding the re-ingest tool.

| Slice | Scope | Size | After |
| --- | --- | --- | --- |
| F1 | Table and snapshot binding, unit and rate rules, closure-hash tests, entry **input** coverage report | ~130 / ~140 / 50 | re-ingest merged |
| F2 | American put tree, European call, pricer reference tests | ~100 / ~140 / 30 | design merge |
| F3 | Fit set, smile, source choice, leg table values | ~120 / ~150 / 40 | F2 |
| F4 | Concession, repaired and unknown rules, sensitivities, structure entry, guard, exit | ~150 / ~180 / 50 | F3 |
| F5 | `fvf_id`, receipts, diagnostics (with status and source coverage), chunked smoke | ~130 / ~130 / 60 | F1, F4, F7 |
| F6 | Calibration from real fills; only if supplied. Built as F6a and F6b, whose anchor and floor section 9 supersedes | ~110 / ~120 / 40 | F4, F7 |
| F7 | Carry and dividend amendment: a carry and dividend pricer in a new module (F2's unchanged), binding the new columns and units, section 4's fallbacks and flags in the leg values | ~150 / ~180 / 60 | F1, F3 |
| F6c-1 | **Model A fit** ([#3104](https://github.com/yshewchuk/investment-validation/issues/3104), reworked): the built CLI end to end, fills CSV to the two files | ~100 / ~160 / 150 | this amendment, Q1 |
| F6c-2 | **Model A leg fills** (#3104): entry, `fill_over_mid` guard, exit | ~140 / ~180 / 120 | F6c-1, Q2-Q4 |

### F6c: model A fit plus leg fills

Two PRs. Sizes are the amendment owner's estimates (added lines and words). The slices run in order, since F6c-2 consumes F6c-1's constants. Both log progress to stderr, use no randomness or clock in values (R3), and test against hand-computed values. A refusal is `EXPERIMENT_IDENTITY_CONFLICT` with a `details.reason`, except a size cap (`RESOURCE_LIMIT_EXCEEDED`). Where a rule below depends on an open question, the answer completes it.

| Rule | F6c-1 | F6c-2 |
| --- | --- | --- |
| R1 inputs, in order | existing output directory, size cap, unreadable file, missing column, empty file, invalid value, then any `V`-input check Q1 keeps; all before output | constants document (`constants_invalid`), then id (`constants_id_mismatch`), then variant (`variant_unknown`), all before pricing; then input columns and legs (`column_missing`, `leg_invalid`) |
| R1 per row | counted, first match: `outside_window`, `unpriceable`, any exclusion Q1 keeps, `zero_half_spread` | `unpriceable`; other statuses follow Q2 |
| R2 resources | the built 16 MiB cap | section 10's chunking and limits; no new store |
| R4 outputs | both files by one directory rename, receipt last, no partial output; an existing output directory refuses; no `fsync` | pure functions that store nothing; no partial result on a refusal |
| R5 identity | receipt binds the parsed fills bytes, the written constants bytes, the new constants id and the source hashes | the caller pins the constants id and variant (`spread050` or none); `fvf_id` follows Q3 |
| R6 causality | the snapshot quote may follow the fill by up to 60 minutes (B1); any `vol_date` or `rates_date` Q1 keeps is strictly before the order date | entry values never read exit-date rows |
| Stored artifacts | `constants.json` (section 9's four constants, F's unchanged ones, the id, `spread050`) and `fit_receipt.json` (those bindings, window, counts, used date range, per-group loss, tie counts, residual statistics) | none |

**Open questions, none settled here; each slice's are settled before it starts.**

| Question | Options and trade-offs |
| --- | --- |
| Q1 (F6c-1): do the `V` input columns, `not_causal`, `vol_unit`, `rate_unknown` and `no_fair_value` remain? | (a) Drop them: more fills used and a smaller reader; the used fills differ from the built fit's. (b) Keep them: the same fills as the built fit, but fills are lost for data model A does not use |
| Q2 (F6c-2): what happens to leg statuses, the `debit_fair` test in `not_a_debit`, `exit_fair_mid`, the mid-floored sensitivity and the `V` diagnostics? | (a) Take `V` out of the fill path: fewer `undetermined` structures; `V` diagnostics are lost and the population differs from F's. (b) Keep `V` for statuses and diagnostics: a population comparable with F runs, but trades are lost for unused inputs. Under either, mid-floored equals the base fill and could be dropped |
| Q3 (F6c-2): do F's source files change? | (a) Yes: the new `fvf_id` identifies model A; a run bound to the old id needs the old checkout. (b) New module outside `fvf_id`: built F keeps its id, but R5 must also pin the new module and constants id |
| Q4 (F6c-2): can entry and exit mix variants or constants? | (a) The leg table carries id and variant and exit refuses a mismatch: safer, more columns. (b) The caller's responsibility: smaller, but mixing is silent |

## 13. Amendment 1 (2026-11-25, before any F1 code)

Items 1 and 2 are **settled by the user (2026-11-25)**. Item 3 is **settled by the user (2026-11-26): the defaults**. Item 4 is the F1 owner's reading (mine), for the user's review.

1. **Manifest.** The merged tool writes no `input_fingerprint`. F refuses a missing manifest, and one where `tables.quotes` or `tables.rates` lacks an `unreadable` or `unrecognized` count, or has a nonzero one. The consumer supplies the expected SHA-256 of the manifest and of every table file it reads. **Limit:** the manifest lists no file hashes, so the pins do not prove that the files and the manifest came from the same run.
2. **F1 coverage** is input coverage only, by `obs_date` year and `chain_kind`: usable quotes (not section 7's `unpriceable`), a positive `smvVol` match, a rate found, and known open interest, bid size and ask size. Status and source shares move to F5. Match reporting on the tool's side is [#2891](https://github.com/yshewchuk/investment-validation/issues/2891).
3. **Rates** (settled, as above). Two rules are load parameters, applied to every lookup on the loaded table. `rate_out_of_range`: by default (`missing`) a value outside the bounds after conversion is missing, so the other is used; `row_unknown` removes both values of that row. A non-finite value is missing under either; with an infinite one, `row_unknown` treats it as outside. `rate_empty_row`: by default (`rate_unknown`), once section 4 has selected a row by its date F does not look further back, even if the row has neither value; `earlier_row` ignores rows with neither value, whatever the cause, within the same 7 days. In both, no usable value is `rate_unknown`. The vol and rate unit conversions are required caller parameters with no defaults.
4. **Binding details.** Section 2's imports exclude the snapshot repository, so the consumer reads the quote rows from its pinned snapshot and passes the id it read with the id its spec pins. F refuses a difference; it cannot verify that the rows came from that snapshot. The coverage report binds its rows the same way. A null or unconvertible key refuses wherever F1 takes keys: quote rows, both tables and the lookup helpers. The helpers also refuse a table value the loader could not have returned. The `vol_unit` median covers the positive finite vols of the files bound in one load; none at all also refuses. A duplicate rate key refuses like a duplicate vol key. A manifest or table file over F1's read cap refuses `RESOURCE_LIMIT_EXCEEDED`, as does one load whose parsed tables together pass its memory budget (default 2 GiB, caller-settable; concatenating peaks at about twice that).
5. **Admission before decoding** (the owner's change to section 2, for the user's review). pandas cannot read a Parquet footer, so F1 imports `pyarrow.parquet`, already pandas's engine here, for footer and schema reads. It adds nothing to the environment. A table is decoded only if the load's running total plus its footer estimate fits the budget: the larger of 128 bytes a row and the needed columns' uncompressed size. The measured size is checked again afterwards. This is an estimate, not a hard memory bound; the consumer's `bounded_run` limit stays the hard one.
