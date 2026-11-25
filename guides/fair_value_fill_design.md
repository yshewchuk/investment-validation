# Fair-value marks and spread-aware fills, model F (shared design)

**Status: design only.** A shared experiment-side module for EXP-190 Stage B (`guides/size_quantile_move_design.md`, section 6) and EXP-189 Amendment 1 (its owner's work); no EXP-188 or EXP-189 file or spec changes here. A consumer adopts F in its own spec and pre-registration. Content: user decisions of 2026-11-25; rules marked "mine" are my judgement calls inside them.

## 1. Why

- **EXP-105 diagnostics** (`reports/exp-105_log_diagnostics.md` §8.5, untracked, supervisor-local). STR-THRU's widest relative-spread quintile (median 0.597) earned +9.87% per trade and 48.9% of summed returns. The whole book earned +4.04% at mid and -1.42% at alpha 0.40; capital-weighted breakeven alpha 0.423.
- **EXP-133** (`guides/exp133_structure_search.md` §4, §6). Per-event search over strikes harvests a vendor mid surface's no-arbitrage violations: 43.6% of its TWIN-P5 width-and-anchor arm's entries violated it (free search 44.6%, fixed incumbent 1.0%). Its `zero_cost` bug showed that a refusal applied at one fill price compares different books.

## 2. Module boundary

- New modules `experiments/fair_value_*`. Imports: numpy, pandas, and `engine.fills.MIN_MEANINGFUL_COST`. `engine/fills.py`, `engine/structures.py`, `engine/pnl_sim.py`, `engine/chain_index.py` and the production normalizer do not change. No sealed v14 closure imports F; the first slice tests both closure hashes (part 1 section 3) unchanged.
- F prices **resolved legs** only; the consumer resolves them and owns timing, gating and sizing.
- **Tool check.** Adding QuantLib (not installed) would change the pinned environment sealed runs record; one American put pricer on numpy suffices, with no IV solver. `engine.pnl_sim.black_scholes_put` and the `engine/v2/scoring/stages.py` helper are European with zero rates, and `pnl_sim` is in the v14 closure, so F is a new pricer.

## 3. Inputs

| Input | Content | Source |
| --- | --- | --- |
| Quotes | one row per (ticker, `obs_date`, expiry, strike, right): `bid`, `ask`, `iv`, `spot`, `quote_repaired`, `open_interest`, `bid_size`, `ask_size`, `chain_kind`. **Every listed row of each requested expiry**, not only the traded legs | pinned snapshot of curated `option_chains` |
| Vol table | `smvVol` per (ticker, `obs_date`, expiry, strike); no `right`, so both rights share it | experiment re-ingest of the raw ORATS cache, `quotes` table |
| Rate table | `iRate5wk`, `iRateLt` per (ticker, `obs_date`) | the same re-ingest, `rates` table (ORATS cores) |

**Data facts (supervisor-checked 2026-11-25, not re-read here).** `engine/data/normalize/n_chains.py` drops `smvVol` from curated `option_chains`; the raw cache has it per strike. Open interest and both sizes exist only on `chain_kind=fetch` rows (2023-01-03 to 2025-12-31), about 75-80% of rows; `eod`, `t14` and `c2` rows have none, so earlier history has none.

**The re-ingest tool** (`tools/orats_smoothed_vol.py`) writes no production store. F binds the content hash of its `manifest.json` (written last) and of every table file read. A missing manifest, or one that reports unread payloads, refuses (Amendment 1). F reads only the columns above; extra columns and manifest keys are ignored.

**Units.** `spot`, strikes and quotes are dollars per share; vols and rates are annualised decimals. The raw units are confirmed from the re-ingest and pinned in each consumer's spec before registration, using no outcome. After conversion, a rate value outside [-0.05, 0.20] is by default missing (Amendment 1, item 3, pending); a non-finite or non-positive vol is missing; a bound vol table whose median `smvVol` is outside (0.05, 5) refuses `vol_unit` (bounds mine).

## 4. Fair value

Per quote row, with `S` the row's curated `spot` (the one strike resolution uses; the re-ingest's `stockPrice` is not used), `T = max(expiry - obs_date, 0) / 365` in calendar days, and `V` the fair value, **clamped to [bid, ask]**:

- `T = 0`: intrinsic value, then clamped.
- **Calls:** European Black-Scholes: the American value only under zero dividends and a non-negative rate.
- **Puts:** American, by a Cox-Ross-Rubinstein tree with **200 steps** (mine), vectorised over rows, early exercise at every step. Vol is floored at 0.01; a tree probability outside (0, 1) is `no_fair_value`.
- **Dividends: documented approximation, zero** (mine). Causal dividend coverage is unconfirmed, and ORATS's dividend inputs are not carried. A dividend before expiry makes F under-value puts and over-value calls; the clamp does not bound that error.
- **Rate:** `iRate5wk` up to 35 calendar days to expiry, linear to `iRateLt` at 365, flat beyond; if one of the two is missing, the other is used (mine). The row is the ticker's for `obs_date`, else its latest earlier one within 7 calendar days; none: `rate_unknown`.

Tests: published American put values; at zero rate the put tree matches the European put `engine.pnl_sim.black_scholes_put` within the tree's discretisation error, and the call matches that put by parity, `C = P + S - K`.

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

- **Leg table**, one row per quote row: `status` (`ok`, `unpriceable`, `no_fair_value`, `rate_unknown`), `V`, vol, source (`smvvol`, `smile`, `intrinsic`), rate, clamp side, `s`, `fill_buy`, `fill_sell`, `c_buy`, `c_sell`, thin, unknown and repaired flags, plus the three sensitivity fill pairs. A structure is a signed dot product over these rows.
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

**Diagnostics**, by year, `chain_kind` and entry versus exit: vol source and status shares; smile minus `smvVol`; `|V - mid| / half-spread` (effective quote; rows with zero half-spread excluded and counted) by right (the dividend omission would show as a put/call difference); clamp, better-than-mid, unknown and repaired shares; `c` quantiles; guard skips by reason; `exit_fair_mid` and `unpriceable` counts; and the share of entries whose structure value violates a no-arbitrage payoff bound at mid and at `V`. Input coverage (F1, Amendment 1), then status, source and liquidity coverage (F5), on entry rows is printed **before any outcome is read**.

## 9. Recalibration from the user's real fills (only if supplied)

Needed per fill: time, contract, side, quantity, price, and `bid`, `ask` and spot at order time, plus the size on the side hit and open interest as known at order time (either may be null, which is unknown; they set the flags by section 5's thresholds). `V` comes from the same pricer at the order-time spot, with `smvVol` and rate from the latest end of day strictly before the order date, clamped to the order-time quote. Observed `c` is `(price - V)/(ask - V)` for a buy and `(V - price)/(V - bid)` for a sell; denominators under $0.005 are skipped and counted; price improvement is kept as negative.

The base and spread constants are fitted by least absolute deviation on fills with neither thin flag nor an unknown (a deterministic 0.01 grid over [0, 1] each; ties to the larger constants), with the 0.50 spread scale fixed. Each thin penalty is then the median residual of the fills with that flag set and the other flag known and 0, floored at 0; fills with both flags set or any unknown enter neither penalty fit. The fit needs 100 unflagged fills, and 30 flagged ones for a penalty; otherwise the constant stays. Spread orders without leg prices instead fit one multiplier on `c`. The guard fraction, thin thresholds and the unknown value 0.5 are not calibrated (all mine).

New constants are a **new constants id**. A consumer needs a new spec hash and PLANNED row before a new run, states the fills' date range and any overlap with its evaluation periods, and never re-scores an old run with constants chosen by that result.

## 10. Failure semantics (R1-R6)

| Rule | Case | Result |
| --- | --- | --- |
| R1 inputs | snapshot, manifest or table hash does not bind; manifest absent or reporting unread payloads; required column absent; duplicate quote or vol key; vol unit check fails | refuse `EXPERIMENT_IDENTITY_CONFLICT` with a `details.reason`; no partial output; nothing imputed except the rules written above |
| R1 per row | missing vol, rate or quote | section 7 statuses, counted; never silent or zero |
| R2 resources | put tree on unique rows | quarter chunks, each row priced once per run, under the consumer's `bounded_run` limit; over it refuses `RESOURCE_LIMIT_EXCEEDED`, no partial report. Smoke first |
| R3 determinism | no randomness or clock | values never depend on row order, chunking or requested structures |
| R4 replay | stored leg tables | atomic writes with a receipt binding input hashes and F id; replay read-only; changed input refuses |
| R5 identity | `fvf_id` | content hash of F's source files and constants (pricer steps, rate, dividend and smile rules, concession, guard); a consumer's spec binds it, and a mismatch refuses |
| R6 causality | all values | a row uses only its own `obs_date` quotes, vols and spot, and a rate row dated no later. Entry values never read exit-date rows. "Earlier" means data dates, never `computed_at`. Section 9 is the only fitted part |

Progress logging follows AGENTS.md.

## 11. Limits

- Uncalibrated constants are assumptions; no direction of error is claimed.
- F leaves cross-strike no-arbitrage violations (measured by the diagnostic).
- The concession ignores order size and spread-order fills.
- The `exp_pnl_sim` simulated exit stays European, zero-rate with **no concession**: optimistic against F-scored exits, and rank gating does not absorb differences between names.
- `smvVol` and rate coverage is unverified; a changing source mix is a confound.

## 12. Split list (code slices; none starts until this design merges)

Estimates: non-test code lines / test lines / doc words, excluding the re-ingest tool.

| Slice | Scope | Size | After |
| --- | --- | --- | --- |
| F1 | Table and snapshot binding, unit and rate rules, closure-hash tests, entry **input** coverage report | ~130 / ~140 / 50 | re-ingest merged |
| F2 | American put tree, European call, pricer reference tests | ~100 / ~140 / 30 | this PR |
| F3 | Fit set, smile, source choice, leg table values | ~120 / ~150 / 40 | F2 |
| F4 | Concession, repaired and unknown rules, sensitivities, structure entry, guard, exit | ~150 / ~180 / 50 | F3 |
| F5 | `fvf_id`, receipts, diagnostics (with status and source coverage), chunked smoke | ~130 / ~130 / 60 | F1, F4 |
| F6 | Calibration from real fills; only if supplied | ~110 / ~120 / 40 | F4 |

## 13. Amendment 1 (2026-11-25, before any F1 code)

Items 1 and 2 are **settled by the user (2026-11-25)**. Item 3 is **pending the user's review of measured missing-rate frequency**. Item 4 is the F1 owner's reading (mine), for the user's review.

1. **Manifest.** The merged tool writes no `input_fingerprint`. F refuses a missing manifest, and one where `tables.quotes` or `tables.rates` lacks an `unreadable` or `unrecognized` count, or has a nonzero one. The consumer supplies the expected SHA-256 of the manifest and of every table file it reads. **Limit:** the manifest lists no file hashes, so the pins do not prove that the files and the manifest came from the same run.
2. **F1 coverage** is input coverage only, by `obs_date` year and `chain_kind`: usable quotes (not section 7's `unpriceable`), a positive `smvVol` match, a rate found, and known open interest, bid size and ask size. Status and source shares move to F5. Match reporting on the tool's side is [#2891](https://github.com/yshewchuk/investment-validation/issues/2891).
3. **Rates** (pending, as above). Two rules are load parameters, applied to every lookup on the loaded table. `rate_out_of_range`: by default (`missing`) a value outside the bounds after conversion is missing, so the other is used; `row_unknown` removes both values of that row. A non-finite value is missing under either; with an infinite one, `row_unknown` treats it as outside. `rate_empty_row`: by default (`rate_unknown`), once section 4 has selected a row by its date F does not look further back, even if the row has neither value; `earlier_row` ignores rows with neither value, whatever the cause, within the same 7 days. In both, no usable value is `rate_unknown`. The vol and rate unit conversions are required caller parameters with no defaults.
4. **Binding details.** Section 2's imports exclude the snapshot repository, so the consumer reads the quote rows from its pinned snapshot and passes the id it read with the id its spec pins. F refuses a difference; it cannot verify that the rows came from that snapshot. The `vol_unit` median covers the positive finite vols of the files bound in one load; none at all also refuses. A duplicate rate key refuses like a duplicate vol key. A manifest or table file over F1's read cap refuses `RESOURCE_LIMIT_EXCEEDED`, as does one load whose parsed tables together pass its memory budget (default 2 GiB, caller-settable; concatenating peaks at about twice that).
