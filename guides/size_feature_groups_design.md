# SIZE feature groups: tail history, IV changes, own volatility, calendar (design)

Design only; no code in this PR. The contract is promoted into `engine/v2/research/ARCHITECTURE.md` by the last slice (as season heat did in #2390). The pattern follows `guides/size_season_heat_population_design.md`: pure computation, then table/dictionary, then builder wiring, then request binding, then contract.

**Evidence (exploratory, not promotion evidence).** `feat_expand.py` (owner inputs, 2026-11-12): 11 yearly folds 2014-2024, 2025 never loaded. Adding the four groups to a production-like base gave RMSE -0.17%, MAE -0.64%, better in 9 of 11 years; no single group dominates. The script reads parquet directly and is a definition reference only. The official SIZE run is the confirmation; this design only makes the levers available to it.

## 1. What already exists (nothing is duplicated)

The HMO dictionary has 42 numeric columns, HMOS adds 5 season columns (47). Reuse, do not re-derive:

| Requested | Already in the panel | New column |
| --- | --- | --- |
| G1 tail history | `n_prior`, `mean_prior_abs_move`, `ema{2,4,8,12}_prior_abs_move` (mean and EMAs only; no spread or tail statistic) | all 8 |
| G2 IV changes | levels `implied_move`, `iv10`, `iv30`, `exern_iv30`, `rvol30`. The legacy `daily_state_lookup` makes `_d1/_d5/_d10` differences (not ratios) but the dictionary does not carry them | all 8 (5 lagged ratios; `iv10/iv30`, `iv30/rvol30`, `exern_iv30/iv30` are same-row ratios of existing columns, materialised so the model input has one source) |
| G3 own volatility | `ret5/10/20` (from `price_history` via `runup_math`), `rvol30` (ORATS, in `daily_market`) | 5: `own_rvol5/10/60`, `own_absret1`, `own_rvol5_60`. `ret*` and `rvol30` are not repeated; the `own_` prefix keeps them apart from vendor `rvol30` and `spy_vol*` |
| G4 calendar/crowding | none | 4: `cal_amc`, `cal_weekday`, `cal_month`, `cal_days_since_prior` (the same-date count is dropped, below) |

25 new columns; HMOSX has 72 numeric columns plus `has_implied_quote`. Names are final unless review changes them.

**Decision: drop the same-event-date count (`same_day_n`).** The exploratory script counted events on the target date from the full ex-post calendar. In the panel the confirmed-metadata policy `unconfirm_date_and_cluster_conflicts.v2` removes date-conflicted and ambiguous events after the fact, the context declares schedule availability `unverified`, and forward calendar sessions are only partly known (memory: Nasdaq ~52%). The count at a decision is therefore not shown knowable, and counting other tickers' dates also touches the protected-event membership. The other calendar columns are knowable: session and event date already define the forecast clock (AMC E-1 close, BMO E-2 close), so `cal_amc/weekday/month` add no information source beyond the target's own definition. Reversible only by a later design with a PIT calendar source.

## 2. Availability, causality and `on_missing`

Decision date = the forecast clock's `source_session_ceiling`. All rules are by data date; nothing filters on `computed_at`; no holdout data is read.

| Group | Rule | `on_missing` |
| --- | --- | --- |
| G1 tail | Computed from the same own-ticker contributor list as H11 (`history_json`, labels with `available_as_of_date <= decision`, strictly before the event, eligible, holdouts already excluded). Absolute values come from the existing `signed_moves` read, so no new outcome is opened. max, p75, p90 (linear interpolation), share above 10%, max of last 4 need `n_prior >= 1`; std (ddof 0) and std/mean need `n_prior >= 2`; std/mean and max/mean need mean > 0 | `null` |
| G2 IV | Current row = last `daily_market` surface row (`src_iv` present) dated `<= ceiling`, the rule `_daily` already uses. Lag reference = k-th previous surface row of that ticker (k = 5, 10 or 20 per column). If the reference row is more than `2k + 4` calendar days before the current row, the column is missing (stale reference). Zero or null denominator is missing, never infinity | `null` |
| G3 own vol | From `price_history` `close_adj` closes dated `<= ceiling` (the series `runup` already reads per ticker); log returns of consecutive rows; vol = sample stdev (ddof 1) of the last w returns times sqrt(252) times 100, needing w + 1 closes; `own_absret1` is the last return's absolute value times 100; `own_rvol5_60` is missing if either side is missing or `own_rvol60 == 0`. Last close more than 5 calendar days before the ceiling, or a non-positive close in the window, makes every G3 column missing | `null` |
| G4 calendar | `cal_amc`, `cal_weekday`, `cal_month` come from the target's stored session and event date. `cal_days_since_prior` = calendar days from the last H11 contributor's event date to the event date | amc/weekday/month: never missing, a session other than AMC/BMO refuses `CONTRACT_MISMATCH`; days_since_prior: `null` with no contributor |

Judgement calls (mine, labelled for review): own volatility uses `close_adj` rather than the script's `daily_market.spot` because spot is unadjusted across splits and `close_adj` is already a bound, version-digested source; values will differ slightly from the script, and the first authorized run reports the null counts per column. The `2k + 4` and 5-day staleness bounds are my proposal; they are asserted in the slice tests, and materiality (rows nulled) is measured at the first data run, not assumed.

Failure semantics for the pure functions: wrong shape, non-finite input, unsorted dates or an unknown lag refuse `CONTRACT_MISMATCH`; a non-finite result from finite input refuses `EXPERIMENT_VARIANT_FAILED`; every legitimate shortage returns `null`. Source rows are already range-checked by `_daily_ranges`; no clipping or winsorising.

## 3. Storage and identity

Features are materialised at panel build, once, in `_rows` of `magnitude_panel_qualified`, from data already in memory per active ticker (`history`/`moves`, `surface`, `prices`). No new source read, no per-run computation in the fit, no raw read by the search.

- Table `research_magnitude_panel` contract `.v4` becomes `.v5`: 25 appended columns (floats nullable; `cal_amc`, `cal_weekday`, `cal_month` int64 non-null, `cal_days_since_prior` int64 nullable). Row ids, contexts and snapshots move; sealed v4 panels rebuild, with no compatibility reader.
- Recipe `HMOSX` = `HMOS` plus four blocks (`tail`, `ivdyn`, `ownvol`, `calendar`), each with its own source dependency and hash. The `H/HM/HMO/MO-/M/HMOS` dictionary hashes and every v1-v10 request id stay byte-identical. A dropped group is simply absent from the blocks.
- Lineage: G1/G4 reuse the existing `history` block (the contributor ids already bind them). G2 records the reference row anchors it used. G3 records first and last window dates and the row count; the price version rows are already pinned by `price_row_digests` in the context, so a changed price version changes the context id. No window-wide anchor list is stored, which keeps the multi-MB `lineage` from growing.
- Context stays `magnitude_panel_context.v5` (no new population receipt); its `feature_recipe` and the registration validator widen to accept `HMOSX`.

R1 memory: per-ticker working sets only (<= 61 closes and 21 surface rows per target; nothing grows with ticker count). R2 nothing cached across builds. R3 refusals as above plus the existing panel refusals. R4 nothing persisted before panel publication. R5 no retry. R6 equal pinned inputs give equal values, row ids and context id. Progress: one stderr line per build with per-column null counts (no per-row logging); the existing event meter covers cadence.

## 4. Search request binding and candidate counts

Request `v11` requires the `HMOSX` dictionary hash (a superset of every subset, as `HMOS` was for v10) and allows `H/HM/HMO/MO-/M/HMOS/HMOSX` in that order; v10 behaviour is unchanged. `HMOSX` is one new set, not one per group.

Candidate count today is `len(feature_sets) * G + (1 if HMO in sets)`, where `G` is the product-sum of the non-size family axes (cap 1000). Adding `HMOSX` to a request with k sets adds one full grid: `(k+1)G` instead of `kG`, and search fits scale the same way. `size_v1_4` stays `HMO`-only. The intended official comparison is `[HMOS, HMOSX]` with the same grid: `2G` candidates, a paired test of the levers against the season-heat set. Per-group attribution would need one set (one grid) per group and is not proposed; no ablation is promised.

## 5. Split list (code lines are non-test added lines, estimates +/-30%)

| # | PR | Est. code | Est. test | Est. doc | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `tail_history` pure computation (G1) | 70 | 90 | 0 | `engine/v2/features`, source-free, like `season_heat` |
| 2 | `iv_dynamics` pure computation (G2) | 80 | 100 | 0 | lag and staleness rule |
| 3 | `own_vol` pure computation (G3) | 75 | 90 | 0 | |
| 4 | `calendar_features` pure computation (G4) | 45 | 50 | 0 | |
| 5 | `HMOSX` dictionary, contract v5, `admit_columns` | 155 | 100 | 0 | 30 base + 40/40/25/20 per group |
| 6a | Builder wiring for G1 + G4 (history side) | 85 | 90 | 0 | in `_rows`; null-count log; lineage |
| 6b | Builder wiring for G2 + G3 (market side) | 130 | 110 | 0 | passes `surface`/`prices`; lineage |
| 7 | Request v11, `HMOSX` binding, registration validator, `size_search_binding` | 80 | 120 | 0 | candidate-count tests |
| 8 | Contract promotion in `ARCHITECTURE.md` (doc) | 0 | 0 | ~55 | last, like #2390; run the doc-budget check first |

Totals: about 720 code, 750 test, 55 doc lines. Tests do not count against the size limit; every slice is under the ~300 cap and only slice 5 is above the ~150 target.

**Total about 720 non-test code lines in 8 code PRs, plus one doc PR.** Each slice has its own owner and one open PR; slices 1-5 may run in parallel (1-4 are independent pure modules, and slice 5 declares columns only), while 6a/6b need 1-5 merged, 7 needs 5, and 8 is last. No heavy job runs in any slice; data runs wait for separate authorization.

## 6. Cheaper subsets

Cost per group including its share of table and wiring, with the shared fixed part (base contract 30, wiring scaffolding 45, request 80) about 155:

| Rank by cost | Group | Marginal lines | Needs market data |
| --- | --- | --- | --- |
| 1 | G4 calendar | ~90 | no |
| 2 | G1 tail history | ~150 | no |
| 3 | G3 own volatility | ~155 | yes (price history) |
| 4 | G2 IV changes | ~170 | yes (daily market) |

- **Groups 1 + 4 only:** slices 1, 4, 5 (G1/G4 columns), 6a, 7, 8 = about 370 lines in 5 code PRs, with no market-data wiring.
- **Market groups 2 + 3 add** about 350 lines (slices 2, 3, 6b and their table columns).
- Dropping a group removes its pure slice, its columns in slice 5 and its share of slice 6; nothing else changes.

The exploratory result does not rank groups by value ("no single group dominates"), so this ranking is by cost only; a dropped group's effect is unmeasured, not shown to be zero.

## 7. Decisions (user, 2026-11-12)

All four groups are built. Accepted as proposed: (1) same-date count dropped (section 1); (2) own volatility from `close_adj` rather than `spot` (section 2); (3) one set `HMOSX` for all groups (section 4).
