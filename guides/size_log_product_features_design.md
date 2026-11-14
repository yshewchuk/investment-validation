# SIZE derived feature sets: log copies (L) and log copies + pairwise products (LP) (design)

Design only; nothing is built. Same pattern as `guides/size_feature_groups_design.md` (pure computation, then recipe, then request binding, then wiring, then contract promotion into `engine/v2/research/ARCHITECTURE.md`). User request 2026-11-14: two sets adjacent to `HMOSX`, so the next official run compares `HMOSX`, `HMOSX+L` and `HMOSX+LP` as separate arms and the existing RMSE selection chooses.

**Evidence (exploratory, not promotion evidence).** `/root/agent-ops/owner_inputs/size_logs_1114/` (`feat4.py`, `f4.out`): 11 yearly folds 2014-2024, no 2025, a 63-column HMOSX-like base, ElasticNet (alpha .03, l1 .25). L (42 `log1p` copies, raw kept): RMSE -0.31%, MAE -0.43%, better 11/11 on both. LP (L + 28 products): RMSE -0.36%, MAE -0.40%, 11/11 vs base; the increment over L alone is weak (RMSE 7/11, MAE 4/11), so L and LP are separate arms. HGB gains nothing (RMSE -0.00%; scale-invariant); the sets still apply to every family so arms stay comparable. The script reads parquet and is a definition reference only. The panel has 73 columns, not 63, and the mapped list below has 40 columns, not 42, so the effect size only approximately transfers; the official run is the confirmation.

## 1. Where the derived columns live

| | (a) New panel columns | (b) Derived at load/fit time from stored columns |
| --- | --- | --- |
| Panel | table `.v5` to `.v6`, 68 more float64 columns, full rebuild (~11 h, a heavy job needing its own authorization); sealed v5/v11 panels are discarded; row ids, contexts, snapshots all move | unchanged: table `.v5`, context `.v5`, `feature_recipe` `HMOSX`; the v11 panel is reused |
| Identity | in the panel dictionary hash | in the candidate's recipe dictionary hash (below) |
| Changing the list or a transform | another rebuild | a recipe change only |
| Cost | storage and rebuild; no per-run compute | one vectorised pass per run (rows x 68 x 8 B, once after the read, not per fold or candidate), reported at the stage boundary |
| Duplicated source of truth | stored value could in principle disagree with recipe | none: values are never stored |

**Decision (supervisor's judgement, adopted because it holds up on the three conditions below): (b).** Every transform is a pure row-wise function of stored columns, and the next run can reuse the v11 panel.

Conditions, stated exactly:

1. **Identity.** A derived set is a *model-feature recipe over the panel*, not a panel subset. `recipe("HMOSXL")` / `recipe("HMOSXLP")` in `magnitude_panel_qualified_table.py` return the `HMOSX` dictionary plus derived fields. Each derived field carries `source_dependencies: [{role: "panel_column", name}]` and `derivation_policy: {schema_version: "derived_feature.v1", function: "log1p"|"product", sources: [...]}`, float64, nullable, `on_missing: null`, unit `log1p(<unit>)` or `<u1>*<u2>`. These sit in `dictionary`, so they enter `dictionary_hash`, `model_columns` and `candidate_spec_id`, and the request id binds the hash (section 3). Derived recipes add `base_subset: "HMOSX"` and a derivation marker; the v1-v11 recipes gain no key, so their hashes and every v1-v11 request id stay byte-identical.
2. **Provenance.** Stored inputs are pinned by the snapshot id and `context_id` (unchanged); the function is in the registered `source_closure`; the dictionary pins which function and sources produce each column. Value = f(pinned stored value, pinned policy, pinned code), and nothing derived is persisted that could drift from that. Determinism is claimed per pinned environment (float64 `numpy.log1p` and multiply), as for the fits themselves.
3. **Stored-panel contract.** The panel build and the registration validator keep calling `recipe("HMOSX")`; the context `feature_recipe` stays `HMOSX`. `recipe()` gets a guard so a derived subset passed as a panel recipe refuses `CONTRACT_MISMATCH`. The panel read requests the stored columns only (`recipe("HMOSX")["model_columns"]`), identical for all arms; `_panel_columns` must not forward derived names to `read_panel`.

Standard tools: `numpy.log1p` and `*` are used directly. `sklearn` `PolynomialFeatures(interaction_only=True)` would form all C(73,2) products and `FunctionTransformer(np.log1p)` would hide the transform in estimator params; neither gives a fixed named list in the dictionary hash, and both would run per fit. Not used.

## 2. The fixed lists

The scratch selection was data-driven (min >= 0 and skew > 1 over the cached table). The design fixes the list as a constant in code; no rule is re-evaluated against data per run. Eligibility rule that is data-independent: a column is eligible only when its definition or the panel's range checks make it non-negative.

**L: 40 columns, `log1p_<name>`, appended in this order after the HMOSX fields (and before nothing else; `has_implied_quote` stays last in `model_columns`).** Scratch names map to panel names:

| Scratch | Panel |
| --- | --- |
| `mean_abs`, `ema2/4/8/12_abs` | `mean_prior_abs_move`, `ema{2,4,8,12}_prior_abs_move` |
| `implied_move`, `iv10`, `iv30`, `exern_iv10/30`, `fwd90_30`, `fexern90_30`, `rvol30`, `spy_vol5/20/60`, `spy_vol20_rel252` | same names |
| `heat5`, `heat20`, `heat_ratio` | `season_heat_mean_abs_5`, `season_heat_mean_abs_20`, `season_heat_ratio_20_252` (the ratio mapping is by name; slice 1 confirms it against the scratch definition) |
| `max_abs`, `std_abs`, `p75_abs`, `p90_abs`, `frac_gt10`, `max_last4`, `cv_abs`, `max_over_mean` | `tail_max_abs`, `tail_std_abs`, `tail_p75_abs`, `tail_p90_abs`, `tail_share_gt10`, `tail_max4_abs`, `tail_std_over_mean`, `tail_max_over_mean` |
| `iv10_30`, `iv_rv`, `ex_ratio`, `imp_chg5/10`, `iv30_chg5/20`, `iv10_chg5` | same names (all current/reference ratios of non-negative values, so >= 0) |
| `rvol5/10/60`, `absret1` | `own_rvol5`, `own_rvol10`, `own_rvol60`, `own_absret1` |
| `median_abs`, `last_abs` | **no panel equivalent (G1 stores mean, EMAs, max, std, p75, p90, share, max4 only); dropped** |

The 40 are 5 + 8 + 4 + 3 + 8 + 3 + 5 + 4. Of the scratch-set's 42, the two dropped are the only unmapped ones.

**LP adds 28 products `<a>_x_<b>`**, all pairs (i < j) of these 8 RAW columns in this order, not of the log copies (as in the scratch): `implied_move`, `mean_prior_abs_move`, `ema4_prior_abs_move`, `tail_p90_abs`, `tail_max_abs`, `rvol30`, `iv30`, `season_heat_mean_abs_20` (scratch `mean_abs`, `ema4_abs`, `p90_abs`, `max_abs`, `heat20`; all map). Products are float64; the scratch cast to float32, which the design does not copy.

## 3. Names, request binding, candidate counts

Set names: `HMOSXL` (HMOSX + 40 logs) and `HMOSXLP` (HMOSXL + 28 products). The letters extend the existing superset-name scheme; `HMOSXLP` is a strict superset of `HMOSXL`, which is a strict superset of `HMOSX`.

- **Request `v12`** requires the `HMOSXLP` dictionary hash (the widest superset, as `HMOSX` was for v11) and allows `H/HM/HMO/MO-/M/HMOS/HMOSX/HMOSXL/HMOSXLP` in that order. v11 and earlier behave exactly as today (`HMOSXL*` in a v11 request is refused as an unknown set).
- `size_search_binding._panel_subset` today maps version to one subset used both for the dictionary-hash check and for the panel read. It splits into two: dictionary subset (v12: `HMOSXLP`) and stored-panel subset (v12: `HMOSX`).
- **Candidate counts.** Unchanged formula `len(feature_sets) * G + (1 if HMO in sets)` (`G` capped at 1000). The intended official comparison `[HMOSX, HMOSXL, HMOSXLP]` is `3G`; with `HMOS` kept as a reference arm `4G`. Fits scale the same way. HGB-family candidates on L/LP are expected to tie the base within noise; that cost is accepted because the user asked for the sets on every family. No per-column ablation is promised.

## 4. Failure semantics

Pure function `derive_columns(columns, set_name)` takes a mapping of column name to a 1-D float array (NaN is a null) and returns the derived columns in recipe order. Rules:

| Case | Result |
| --- | --- |
| Source value is null | derived value null (a product is null if either input is null). Never zero-filled |
| Negative input to `log1p` (the panel's non-negative contract is violated) | `CONTRACT_MISMATCH`; no clip (the scratch clipped at 0). Not reachable from a conforming panel, since every listed column is >= 0 by definition or range check |
| Non-null `+-inf` input | `CONTRACT_MISMATCH` |
| Non-finite result from finite input (product overflow) | `EXPERIMENT_VARIANT_FAILED`, as the HMOSX groups |
| Source column absent from the frame, wrong length, or non-float dtype | `CONTRACT_MISMATCH` (a wiring error, not data shortage) |
| Source is an X column and the projection was not admitted by `admit_columns(..., x_groups=True)` | refused earlier by the read, `HOLDOUT_ACCESS_DENIED`; the derive step only sees admitted frames |
| Unknown set name, or a request/recipe whose derived fields differ from the installed lists | `INVALID_EXPERIMENT_SPEC` at request compile (dictionary-hash mismatch), as for other dictionary mismatches |
| Legitimate shortage | null, never a refusal |

**`on_missing` and imputation.** Derived columns declare `on_missing: null`. The fit recipes already impute by the recipe, not per column: Ridge/ElasticNet/MLP use median imputation fitted on training rows with `missing_indicator: all` and standard scaling; HGB uses native missing handling. A derived column's null set is the union of its sources' null sets, so its missing indicator duplicates a source indicator (or is a union of two for a product). Decision (mine, for review): keep the recipe's uniform `indicator: all` and add no special case. The effect is `output_features` growing from `2 x 73` to `2 x 113` (HMOSXL) and `2 x 141` (HMOSXLP) with perfectly collinear indicators, which the penalty and scaler tolerate; excluding them would add a per-column preprocessing rule to every family recipe validator. Materiality is measured: the first data run reports per-column derived null counts on stderr (one line, never per row).

## 5. Causality

Each derived value is a function of the same row's stored columns only: no other row, ticker, date window or fitted statistic enters (the transforms have no parameters estimated from data; the L list and the pair list are constants). Availability is therefore the intersection of the sources' availability, and every source is already causal at the row's decision date (the forecast clock's `source_session_ceiling`, by data date, never `computed_at`; the HMOSX contract in `engine/v2/research/ARCHITECTURE.md`). That is sufficient: a function of causal inputs that adds no inputs is causal, and because it uses no training statistics, applying it once to the whole read frame before the fold split cannot leak across folds. Holdout exclusion is inherited: the derive step runs on the frame the read already restricted to the admitted event ids; it opens no source and reads no holdout or outcome data. Only the median imputer and scaler remain fit-time, and they are already fitted on training rows per fold.

## 6. Resource and progress rules

R1 memory: rows x (up to 68) x 8 B once per run; no growth with candidate count; a measured RSS line at the derive boundary. R2 nothing cached across runs. R3 refusals above. R4 nothing persisted (no store write). R5 no retry. R6 equal pinned stored inputs, code and numpy give equal values and equal recipe hash. Progress: one stderr line per run (stage start/end with elapsed time, rows, derived column count, per-column null counts, RSS); no per-row logging.

## 7. Split list (code lines are non-test added lines, estimates +/-30%)

| # | PR | Est. code | Est. test | Est. doc | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | `derived_features` pure module: fixed lists, `derive_columns`, refusal codes | 95 | 130 | 0 | `engine/v2/research/`, source-free; confirms the `heat_ratio` mapping |
| 2 | `HMOSXL`/`HMOSXLP` recipes in `magnitude_panel_qualified_table.py`: derived field descriptors, `base_subset`, panel-recipe guard | 85 | 90 | 0 | v1-v11 hashes asserted byte-identical |
| 3 | Request `v12`: schema registry, `HMOSXLP` hash, set order, counts, `_panel_subset` split in `size_search_binding` | 95 | 120 | 0 | needs 2 |
| 4 | Fit-path wiring: stored-vs-derived column split in `_panel_columns`, one `derive_columns` call at the panel-to-matrix seam, null-count/RSS log | 120 | 110 | 0 | needs 1-3; the seam (consumers of `read_stored_panel` feeding `FitInputs`) is pinned read-only first; an overage over ~150 is split in half there, not grown |
| 5 | Contract promotion in `engine/v2/research/ARCHITECTURE.md` | 0 | 0 | ~45 | last; run `tools/oc_check.py tests/test_architecture_doc_budgets.py` first |

**Total about 395 non-test code lines in 4 code PRs (each under the ~300 cap; #4 is the only one at risk of passing the ~150 target), about 450 test lines, and one ~45-line doc PR.** PRs 1 and 2 are independent and may run in parallel; 3 needs 2; 4 needs 1-3; 5 is last. No heavy job runs in any slice; the official run waits for separate authorization.
