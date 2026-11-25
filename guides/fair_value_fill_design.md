# Fair-value marks and spread-aware fills, model F (shared design)

**Status: design only.** A shared experiment-side module for EXP-190 Stage B (`guides/size_quantile_move_design.md`, section 6) and for EXP-189 Amendment 1, which its own owner writes. This file changes no EXP-188 or EXP-189 file or spec. A consumer adopts F in its own spec and pre-registration. Content: user decisions of 2026-11-25; constants and rules marked "mine" are my judgement calls inside them.

## 1. Why

- **EXP-105 diagnostics** (`reports/exp-105_log_diagnostics.md` §8.5, untracked, supervisor-local). STR-THRU's edge was concentrated in the widest-quoted names, though not confined to them: the widest relative-spread quintile (median 0.597) earned +9.87% per trade and supplied 48.9% of summed returns. The whole book earned +4.04% at mid and -1.42% at alpha 0.40; its capital-weighted breakeven alpha is 0.423. Mid is a real price only where quotes are narrow.
- **EXP-133** (`guides/exp133_structure_search.md` §4, §6). A vendor mid surface violates no-arbitrage, and a per-event search over strikes harvests it: 43.6% of the entries of its TWIN-P5 width-and-anchor arm violated it (44.6% for the free search, 1.0% for the fixed incumbent). Its `zero_cost` bug also showed that a refusal applied at one fill price compares different books.

## 2. Module boundary

- New modules `experiments/fair_value_*` (not named for one experiment). Imports: numpy, pandas, and `engine.fills.MIN_MEANINGFUL_COST`. `engine/fills.py`, `engine/structures.py`, `engine/pnl_sim.py`, `engine/chain_index.py` and the production normalizer do not change. No sealed v14 closure imports F; the first slice tests both closure hashes (part 1 section 3) unchanged.
- F prices **resolved legs**. It does not choose expiries or strikes, decide timing, gate on scores, or size positions. The consumer resolves legs with its own pricer.
- **Tool check.** QuantLib is not installed, and adding it would change the pinned environment that sealed runs record. The need is one American put pricer on numpy. No IV solver is needed. The existing helpers (`engine.pnl_sim.black_scholes_put` and the one in `engine/v2/scoring/stages.py`) are European with zero rates; `pnl_sim` is in the v14 closure, so F is a new pricer and only compares with it in tests.

## 3. Inputs

| Input | Content | Source |
| --- | --- | --- |
| Quotes | one row per (ticker, `obs_date`, expiry, strike, right): `bid`, `ask`, `iv`, `spot`, `quote_repaired`, `open_interest`, `bid_size`, `ask_size`, `chain_kind`. **Every listed row of each requested expiry**, not only the traded legs | pinned snapshot of curated `option_chains` |
| Vol table | `smvVol` per (ticker, `obs_date`, expiry, strike); no `right`, so both rights share it | experiment re-ingest of the raw ORATS cache, `quotes` table |
| Rate table | `iRate5wk`, `iRateLt` per (ticker, `obs_date`) | the same re-ingest, `rates` table (ORATS cores) |

**Data facts (supervisor-checked 2026-11-25, not re-read here).** The curated `option_chains` has no `smvVol` (`engine/data/normalize/n_chains.py` drops it); the raw cache has `smvVol`, `callMidIv` and `putMidIv` per strike. Open interest and both sizes exist only on `chain_kind=fetch` rows (2023-01-03 to 2025-12-31), about 75-80% of rows; `eod`, `t14` and `c2` rows have none, so earlier history has none.

**The re-ingest tool** (`tools/orats_smoothed_vol.py`) is a separate owner's work, in progress and not yet merged; it writes no production store and needs no paid backfill. F reads its two tables and binds the content hash of its `manifest.json` (written last) and of every table file read. A missing manifest, or a manifest with a null `input_fingerprint` (some payloads unread), refuses. If the merged tool's columns differ from the table above, this file is amended before slice F1.

**Units.** `spot`, strikes and quotes are dollars per share; vols and rates are annualised decimals. The raw units are confirmed from the re-ingest and pinned in each consumer's spec before registration, using no outcome. After conversion, a rate outside [-0.05, 0.20] is `rate_unknown`; a non-finite or non-positive vol is missing; a bound vol table whose median `smvVol` is outside (0.05, 5) refuses `vol_unit` (bounds mine).

## 4. Fair value

Per quote row, with `S` the row's curated `spot` (the one strike resolution uses; the re-ingest's `stockPrice` is not used), `T = max(expiry - obs_date, 0) / 365` in calendar days, and `V` the fair value, **clamped to [bid, ask]**:

- `T = 0`: intrinsic value, then clamped.
- **Calls:** European Black-Scholes. Under this model's zero-dividend rule (below) and a non-negative rate, that equals the model's American value; it is not so for a real dividend payer, and with a negative rate it is an approximation.
- **Puts:** American, by a Cox-Ross-Rubinstein tree with **200 steps** (mine), vectorised over rows, early exercise at every step. Vol is floored at 0.01; a tree probability outside (0, 1) is `no_fair_value`.
- **Dividends: documented approximation, zero** (mine). The coverage of a causal dividend source is unconfirmed, and ORATS's own dividend inputs for `smvVol` are not carried, so a second source could disagree with them. Where a dividend falls before expiry, the omission under-values puts and over-values calls; the clamp bounds the price error by the quoted spread. It is not corrected.
- **Rate:** `iRate5wk` up to 35 calendar days to expiry, linear to `iRateLt` at 365, flat beyond; if one of the two is missing, the other is used (mine). The row is the ticker's for `obs_date`, else its latest earlier one within 7 calendar days; none: `rate_unknown`.

Tests: published American put values; zero rate equals European; the European branch at zero rate matches `engine.pnl_sim.black_scholes_put`.

**Vol source, one per (ticker, `obs_date`, expiry)**, decided from the whole listed expiry, so a row's `V` never depends on which structure asked. The **fit set** is the out-of-the-money rows (puts `K <= S`, calls `K > S`) with finite positive `iv`, `0 <= bid <= ask`, positive mid and `quote_repaired` false.

1. If the fit set is non-empty and every strike in it has `smvVol`, the expiry uses `smvVol`. A row outside the fit set with no `smvVol` is `no_fair_value`.
2. Otherwise the **fallback smile**: weighted least squares `iv = a + b*k + c*k^2`, `k = ln(K/S)/sqrt(T)`, on the fit set's stored `iv`, by `numpy.polyfit` with residual weight `1/max((ask - bid)/mid, 0.02)`. It needs at least 5 distinct strikes, with one or more below and one or more above `S`. Outside the fitted `k` range the edge value holds. All rows of the expiry use the smile.
3. A fit that cannot be made, or a non-positive fitted value inside the fitted range: `no_fair_value` for every row of the expiry (fit rules mine).

The two sources are never mixed within an expiry. Stored `iv` is ORATS's mid IV under ORATS's conventions, not this tree's, so a smile value is an approximation to `smvVol`; their gap is printed where both exist.

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

**Scale.** With `V` at mid, F equals ladder alpha `0.5 - c/2`: 0.375 for the tightest liquid quote, 0.125 at `s >= 0.50`, 0 when also thin on both counts. All are below EXP-105's breakeven 0.423. With `V` away from mid, one side can fill **better than mid** (at `V = ask`, `c = 0.25` sells at alpha 0.75). That follows from marking at the vendor's smoothed value, and a search could favour such legs. So F reports the share of better-than-mid fills by side, and a third sensitivity, **mid-floored F** (a buy never below mid, a sell never above), re-marks the same taken set.

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

## 7. Outputs

- **Leg table**, one row per quote row: `status` (`ok`, `unpriceable`, `no_fair_value`, `rate_unknown`), `V`, vol, source (`smvvol`, `smile`, `intrinsic`), rate, clamp side, `s`, `fill_buy`, `fill_sell`, `c_buy`, `c_sell`, thin, unknown and repaired flags, plus the three sensitivity fill pairs. A structure is a signed dot product over these rows, so a consumer's per-ladder fast path keeps its shape.
- **Entry result:** the three debits, `spread_sum`, guard result, leg flags.
- **Exit result:** `proceeds_F`, flags, or `unpriceable`.

**Statuses**, first match wins: `unpriceable` is a missing or non-finite quote, `bid < 0`, `ask <= 0` or `bid > ask` (a zero bid with a positive ask is usable); then, only when `T > 0`, `rate_unknown`; then `no_fair_value` (section 4's vol and tree rules, or a missing or non-positive `spot`; at `T = 0` only the latter applies).

**Entry and exit differ.** At entry, any leg not `ok` makes the structure `undetermined` (a missing fact, not "no"), known at the decision. At exit, a held trade **never leaves the population**: a leg with a usable quote but `no_fair_value` or `rate_unknown` takes `V = mid`, flagged `exit_fair_mid`, and pays the concession. An `unpriceable` leg makes the exit `unpriceable`; the consumer then applies its own pre-registered, labelled conservative exit (EXP-188 design section 6, rung 3: the structure's worst-case payoff).

## 8. Use by a consumer

F is used in **selection and scoring alike**:

- `exp_pnl_sim` takes `debit_F` as its entry cost (`expected_pnl` already takes `entry_cost` as an argument, so `engine/pnl_sim.py` is unchanged). This also drives any width choice that maximises it.
- A selection model's target is the return on debit above, including the consumer's conservative exits.
- Every F-scored P&L, account and success test uses `debit_F` and `proceeds_F`.
- **Mid and harsh are still reported**, on the same taken set, computed by the consumer's existing code (`mid` is `FillModel(0.5)`; "harsh" keeps that experiment's own definition). They never decide or break a tie where F is the declared basis.

**Diagnostics**, by year, `chain_kind` and entry versus exit: vol source and status shares; smile minus `smvVol`; `|V - mid| / half-spread` (effective quote; rows with zero half-spread excluded and counted) by right (the dividend omission would show as a put/call difference); clamp, better-than-mid, unknown and repaired shares; `c` quantiles; guard skips by reason; `exit_fair_mid` and `unpriceable` counts; and the share of entries whose structure value violates a no-arbitrage payoff bound at mid and at `V`. Status, source and liquidity coverage on entry rows is printed **before any outcome is read**.

## 9. Recalibration from the user's real fills (only if supplied)

Needed per fill: time, contract, side, quantity, price, and `bid`, `ask` and spot at order time. `V` comes from the same pricer at the order-time spot, with `smvVol` and rate from the latest end of day strictly before the order date, clamped to the order-time quote. Observed `c` is `(price - V)/(ask - V)` for a buy and `(V - price)/(V - bid)` for a sell; denominators under $0.005 are skipped and counted; price improvement is kept as negative.

The base and spread constants are fitted by least absolute deviation on fills with neither thin flag nor an unknown (a deterministic 0.01 grid over [0, 1] each; ties to the larger constants), with the 0.50 spread scale fixed. Each thin penalty is then the median residual of its flagged fills, floored at 0. The fit needs 100 unflagged fills, and 30 flagged ones for a penalty; otherwise the constant stays. Spread orders without leg prices instead fit one multiplier on `c`. The guard fraction, thin thresholds and the unknown value 0.5 are not calibrated (all mine).

New constants are a **new constants id**. A consumer needs a new spec hash and PLANNED row before a new run, states the fills' date range and any overlap with its evaluation periods, and never re-scores an old run with constants chosen by that result.

## 10. Failure semantics (R1-R6)

| Rule | Case | Result |
| --- | --- | --- |
| R1 inputs | snapshot, manifest or table hash does not bind; manifest absent or fingerprint null; required column absent; duplicate quote or vol key; vol unit check fails | refuse `EXPERIMENT_IDENTITY_CONFLICT` with a `details.reason`; no partial output; nothing imputed except the rules written above |
| R1 per row | missing vol, rate or quote | the row statuses of sections 4 and 7, counted by reason; never silent, never zero |
| R2 resources | cost is the put tree on unique rows | quarter chunks, each unique row priced once per run, under the consumer's `bounded_run` and declared limit; over it refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report. The smoke measures time and memory first |
| R3 determinism | no randomness, no clock | same bound inputs give the same values; results do not depend on row order, chunking or which structures are requested |
| R4 replay | a consumer that stores leg tables | writes them atomically with a receipt binding input hashes and the F id; exact replay is read-only; a changed input under a receipt refuses, no repair. Scratch follows AGENTS.md |
| R5 identity | `fvf_id` | content hash of F's source files and constants (pricer steps, rate, dividend and smile rules, concession, guard); a consumer's spec binds it, and a mismatch refuses |
| R6 causality | all values | a row uses only its own `obs_date` quotes, vols and spot, and a rate row dated no later. Entry values never read exit-date rows. "Earlier" means data dates, never `computed_at`. Section 9 is the only fitted part |

Progress logging follows AGENTS.md: stage start and end, rows done/total, rate, ETA, RSS, and counts by status; never per row.

## 11. Limits

- F is a model of fills, not a measurement. Until real fills calibrate it, its constants are assumptions, and no direction of error against real fills is claimed.
- `smvVol` and the end-of-day quote come from one vendor, and the clamp trusts the quote. F does not remove no-arbitrage violations across strikes; the diagnostic measures what is left.
- The concession ignores order size and does not credit a spread order that fills inside the sum of its legs.
- In `exp_pnl_sim` only the entry cost changes: the simulated exit stays the engine's European, zero-rate model value with **no concession**, so simulated returns are optimistic against F-scored exits. A trailing-rank gate absorbs a common level shift, not the difference between names.
- Coverage of `smvVol` and rates is not verified here; a source mix that changes by year or pull type remains a confound.

## 12. Split list (code slices; none starts until this design merges)

Estimates: non-test code lines / test lines / doc words. The re-ingest tool is the separate owner's PR and is not counted.

| Slice | Scope | Size | After |
| --- | --- | --- | --- |
| F1 | Table and snapshot binding, unit and rate rules, closure-hash tests, entry coverage report | ~130 / ~140 / 50 | re-ingest merged |
| F2 | American put tree, European call, pricer reference tests | ~100 / ~140 / 30 | this PR |
| F3 | Fit set, smile, source choice, leg table values | ~120 / ~150 / 40 | F2 |
| F4 | Concession, repaired and unknown rules, sensitivities, structure entry, guard, exit | ~150 / ~180 / 50 | F3 |
| F5 | `fvf_id`, receipts, diagnostics, chunked smoke | ~130 / ~130 / 60 | F1, F4 |
| F6 | Calibration from real fills; only if supplied | ~110 / ~120 / 40 | F4 |
