# IV crush v2: nested search at SIZE parity (design)

Design only; no code in this PR. The contract stays in this guide and lands with each code PR: `experiments/ARCHITECTURE.md` and the models/research ARCHITECTURE files are at their growth ceiling, so no code PR adds ARCHITECTURE growth (at most a one-line pointer replacing an existing line). Facts below were checked against origin/main 874ac5fd1 (2026-11-16); each code PR re-verifies the ones it relies on.

## 1. Scope and decisions (user, 2026-11-16, binding)

| # | Decision |
| --- | --- |
| 1 | Scope is SIZE parity first: feature sets HMO, HMOS, HMOSX, HMOSXL, HMOSXLP (SIZE's stored panel plus fit-time derived rows); wider grid (HGB loss/learning_rate/leaf/min-leaf/l2/max_iter, losses valid for a signed target only); signed spline ElasticNet; signed MLP profiles; leads selector; `workers`; blended selection; p10/p50/p90 distribution reporting; paired report v2. |
| 2 | Selection blend is RMSE and MAE only, no tail term: `0.67*RMSE + 0.33*MAE`, same normalization and tie-band rule as SIZE. |
| 3 | Target stays signed `crush_pct_iv30 = 100*(post_iv30/pre_iv30-1)`; no log-ratio or other transform. |
| 4 | Runs: a 50-ticker smoke, then the full panel; IV events reuse SIZE's stored panel; outer year 2025. |

**Phase 2 (future work only, not designed here):** prior-crush history (its own store, as-of logic mirroring `eligible_history`), and crush-specific derived features (252-day IV rank, implied move over `mean_prior_abs_move`, iv30 over rvol60). Log-ratio target is rejected by decision 3.

**Standard tool check.** sklearn (`HistGradientBoostingRegressor`, `ElasticNet`, `SplineTransformer`, `MLPRegressor`) and numpy remain the only estimator and bootstrap code; the recipes already wrap them. Nothing here hand-rolls a standard format.

## 2. Design constraint: the SIZE closure is frozen

The SIZE v12 full run executes from a pinned worktree, and its panel identity is the source closure of `experiments/size_panel.py`. **No IV PR may modify any file in `source_closure(Path('.'), ['experiments/size_panel.py'])` (150 files on 874ac5fd1) or `engine/v2/research/derived_features.py` (it is in that closure).** Owner check before every push:

```
python3 -c "from pathlib import Path; from engine.v2.ops.fingerprints import source_closure as s; print('\n'.join(s(Path('.'),['experiments/size_panel.py'])))" | sort > /tmp/closure.txt
git diff --name-only origin/main...HEAD | sort | comm -12 - /tmp/closure.txt   # must print nothing
```

Frozen files this design touches nearby: `engine/v2/models/regression_target.py`, `engine/models/training/iv_crush.py` (the legacy champion), `engine/v2/research/magnitude_panel_*`. The search stack is **outside** the closure and may change: `experiments/regression_search_binding.py`, `magnitude_search_request*.py`, `magnitude_search_selection.py`, `magnitude_search_{plan,trials,outer,worker}.py`, `size_search_binding.py`, `size_distribution_report.py`, `size_paired_report.py`, `engine/v2/models/*_recipe.py`, `engine/v2/models/regression_candidate.py`, `engine/v2/models/training/*`. Even so, IV-specific logic goes in new modules (`iv_*`, `regression_*`); a change to a shared SIZE module is allowed only as a behaviour-preserving generalization whose default argument reproduces today's bytes, and a test pins that (see section 8). Cross-PR landing is checked with `git log origin/main`, not PR state.

## 3. Request `regression_search_request.v2`

**Shape.** Every v1 key unchanged (snapshot ids, `panel_request`, `feature_dictionary_hash`, `eligibility_policy_ref`, `feature_sets`, `space`, `outer_years`, `inner_years`, `max_candidates`, `max_fits`, `comparison`, `limits`), plus `selection` (required) and `workers` (optional, integer 1..8, `validate_workers`). The lead selector is, as for SIZE, a compile argument (`horizons`, subset of 1/3/5 in ascending or legacy 5,3,1 order), not a request key; IV runs pass `(1,)`. `selection` is exactly `{metric_weights: {rmse: 0.67, mae: 0.33}, lead_weights: {"1": 1.0, "3": 0.0, "5": 0.0}, tie_band_floor: <finite number >= 0>}`; any other key, weight or a `large_move_*` field is refused (no tail term exists for a signed target).

**Version gating.** `compile_request` dispatches on `schema_version`. v1 is untouched: its compile path, the `_SETS == [H, HM, HMO]` rule, its HMO-only hash and every v1 request/candidate/plan id stay byte-identical (acceptance (c)). v2 is a new compiler module `regression_search_request_v2.py` reusing the v3 axis helpers; it never accepts a magnitude (`magnitude_search_request.*`) schema and v1/magnitude compilers never accept `.v2` regression. The plan stays `regression_search_plan.v1` with `selection.metric` changed to `"blend_rmse_mae"` for v2 plans (v1 plans keep `inner_pooled_rmse`); `regression_search_binding` dispatches on the request version.

**Admitted values.**
- `feature_sets`: ordered, unique, nonempty subset of `[HMO, HMOS, HMOSX, HMOSXL, HMOSXLP]` (H and HM stay v1-only; SIZE's panel always carries HMO). `feature_dictionary_hash` must be the HMOSXLP dictionary hash, as SIZE v12 does, so the request id binds the widest dictionary; each candidate's `feature_recipe` carries its own set's hash and columns (the v3 substitution already done in `magnitude_search_request_versions.py`).
- `space.ridge`: `alpha`. `space.elasticnet`: `alpha`, `l1_ratio`, optional `spline_knots` (0 or 4..6; >0 only on sets containing the spline source columns: HMOSX, HMOSXL, HMOSXLP). `space.hgb`: `max_leaf_nodes`, `learning_rate`, `min_samples_leaf`, `l2_regularization`, optional `max_iter` (default `[500]`) and `loss` (default `["squared_error"]`), plus `quantile` when and only when `loss == ["quantile"]` (SIZE's rule that quantile is the only declared loss stays). **Losses are `squared_error`, `absolute_error`, `quantile` only; `gamma` and `poisson` are refused at compile** because the target is signed (SIZE's `_HGB_LOSSES` filtered by `target_domain`). `space.mlp`: `profile` from the SIZE v13 profile table (18 names), all with signed variants (section 5). Not admitted: `size_v1_4`, `blend`, `n_seeds` (SIZE-specific or phase 2), `ridge` extras beyond v13's.
- Bounds are SIZE's v4+ table (`_BOUNDS` in `magnitude_search_request_v3.py`: for example `learning_rate` 1e-3..1, `max_leaf_nodes` 2..64, `min_samples_leaf` 1..1000, `l2_regularization` 0..1000, `max_iter` 100..5000, `l1_ratio` 1e-6..1, `alpha` per family); values outside are refused.

**Closed-grid rule.** Each axis is a nonempty, unique, ascending list of in-bounds numbers; absent optional axes take the default list, so declaring a default equals omitting it and gives the same recipe and candidate ids (the SIZE `n_seeds=[1]` precedent). The compiler computes candidates = sum over families of product of axis lengths, times the set count, and search fits = horizons x (candidates x inner folds + outer folds), the same accounting as `_counts`; `max_candidates` and `max_fits` must cover them, else `INVALID_EXPERIMENT_SPEC` before anything is sealed. The component total (`total_component_fits`) must fit `comparison.max_component_fits`; component weights: 1 per candidate, MLP 1, the comparator's refit as today.

**Compile-time refusals** (all `INVALID_EXPERIMENT_SPEC`, nothing partial): unknown or missing keys, unsupported loss, spline on a set without its source columns, duplicate or unsorted values, non-finite or bool/str numbers, `child.memory_bytes > parent.memory_bytes`, an `outer_years`/`inner_years` shape violating the v1 rule (inner strictly earlier, 30-day grouped ticker embargo policy ref).

**Binding (P1).** `regression_search_binding` currently hardcodes `panel_recipe("HMO")`, requires `panel["event_ids"] == binding["event_ids"]` and reads columns for HMO. v2 changes it to: columns = union of the declared sets' stored columns (H..HMOSX; L/P are derived at fit time by `derived_rows.with_derived`, unchanged), dictionary hash checked against the request's, and `set(binding["event_ids"]) <= set(panel["event_ids"])` (IV events are a subset of SIZE's panel). Events missing from the panel are a measurement, not a silent drop: the 50-ticker smoke reports `iv_events`, `in_size_panel`, `outside` and the labelled-and-would-have-traded count, and the default for `outside > 0` at plan time is refuse (`EXPERIMENT_IDENTITY_CONFLICT`) until the user rules; registration then lists the excluded ids.

## 4. Capability contracts and failure semantics (R1-R6)

R1 member/fit failure, R2 memory and CPU, R3 determinism, R4 persistence/evidence, R5 identity, R6 causality. Rules common to every capability, then deltas.

**Common.**
- **R1:** a failed fit fails its trial with the fit's own `ValueError`; no retry, no partial forecast stored, no reweighting. A failed trial is handled by the existing trial layer exactly as a failed SIZE trial is (P7 verifies how selection treats it and records the rule in its PR; it never invents a score). Compile and binding refusals are `INVALID_EXPERIMENT_SPEC` / `EXPERIMENT_IDENTITY_CONFLICT` and write nothing.
- **R2:** fits are single-threaded within a trial; `workers` (1..8) only parallelizes trials through the existing worker/shard machinery and is not part of request identity beyond today's rule (it is popped before sealing in the v3 compiler; v2 does the same). Peak memory is one panel copy (the shared `_PANEL_MEMO`) plus the largest in-flight fit per worker.
- **R3:** fixed seeds (HGB `random_state` 0, MLP per profile as today), fixed candidate order by `candidate_id`, `math.fsum` in selection: equal inputs give equal bytes.
- **R4:** stored artifacts are the per-trial forecast arrays, fit diagnostics per horizon (existing `size_fit_diagnostics.evidence` branches by schema) and the sealed plan; nothing carries paths or raw exception text.
- **R5:** `candidate_id` hashes the full recipe including `target_domain`; the plan seals all ids before any fit; nothing chooses candidates from scores.
- **R6:** training rows are fold-admitted by data date (`load_iv_crush_targets`, `labels_before`); evaluation labels are never read before the frozen choice; no `computed_at` filter; no new reader.

| Capability | Contract | Deltas to the common rules |
| --- | --- | --- |
| Request sets (P1) | v2 admits the five sets; binding by version | Missing column in the stored panel or derived rows fails the reader (`EXPERIMENT_IDENTITY_CONFLICT`), never imputes. Derived L/P rows come from `with_derived` on the same fit rows, so a set never sees a row another set did not. A cached panel is revalidated by fingerprint on every read (existing memo). Idempotent: same request, same plan id. |
| Grid (P2) | signed-valid HGB axes, EN bounds, leads, workers | `regression_candidate_recipe` (closed `_KEYS`) grows the optional HGB params it already forwards to `ridge_hgb_recipe` (`learning_rate`, `min_samples_leaf`, `l2_regularization`, `loss`, `max_iter`, `quantile`); an unsupported loss reaching a recipe raises `ValueError` there too (defence in depth). v1 recipes unchanged. |
| Signed spline EN (P3) | section 5 | R1: non-convergence raises (as the unsigned spline). R3: knots by training quantiles, deterministic; constant extrapolation. |
| Signed MLP (P4) | section 5 | R1: non-convergence and nonfinite weights raise, no retry (profile `variant_failure_no_retry`). R3: profile-fixed seed and early-stopping rules. |
| Blended selection (P7) | section 6 | R1: a zero comparator RMSE or MAE with a nonzero candidate error raises `ZeroComparatorError` (SIZE behaviour); empty or non-lead-1 rows raise. Pure function, no IO. |
| Distribution (P8) | section 6 | Statuses `no_pool`, `pool_too_small`, `no_scored_rows`, `trial_failed`, `skipped_resource_cap`, `ok` as SIZE; a resource cap skips the report, never the run. Report-only: it never influences selection. |
| Paired report v2 (P9) | section 7 | Coverage mismatch raises as v1 does; an abstaining arm is reported as abstention, not zero, and never imputed. Pure function of stored forecasts and labels. |

**Progress logging (conventions).** The existing trial loop's bounded stage progress carries done/total, rate, ETA and RSS; P2 weights nothing new. The distribution and report stages log stage start/end with elapsed seconds, per horizon, never per event.

## 5. Signed variants

**Pattern.** SIZE's nonnegative recipes carry no domain; the regression recipes carry `target_domain` and `prediction_projection` (`none` for signed, `clip_zero` for nonnegative), as `regression_ridge_hgb_recipe.v1` and `regression_elasticnet_recipe` already do for ridge/EN/HGB (their fit entries `fit_regression_*` pass `nonnegative=domain == "nonnegative"`). The two missing families follow the same shape; each adds a new schema and validator, leaves the magnitude schemas and ids untouched, and rebuilds-and-compares in its validator.

**Spline ElasticNet (P3).** `engine/v2/models/regression_elasticnet_spline_recipe.py`: `regression_elasticnet_spline_recipe(target_domain, columns, feature_dictionary_hash, alpha, l1_ratio, n_knots)` wraps `elasticnet_spline_recipe` (same `SPLINE_COLUMNS` of 34, cubic, quantile knots, `extrapolation: constant`, `include_bias: False`, `precompute: True`; same `n_knots` 4..6 admitted by the request, 4..8 by the recipe), changes the schema to `regression_elasticnet_spline_recipe.v1`, adds `target_domain` and `prediction_projection: none` (signed). The spline block is unchanged: it is a feature transform, independent of the label sign. In the fit, `select_validator` and `is_spline` gain the new schema, `fit_regression_elasticnet_predictions` calls `_fit(..., nonnegative=False)`, and `magnitude_candidate_family` maps the schema to `elasticnet`. The unsigned `magnitude_elasticnet_spline_recipe.v1` hard-codes `clip_zero` only through `_fit(nonnegative=True)`; nothing is edited there. Spline source columns are not IV-specific this phase.

**MLP (P4).** `regression_mlp_recipe.v1` = `mlp_recipe` plus `target_domain` and `prediction_projection: none`; the fit's final `np.maximum(predictions, 0.0)` (`magnitude_mlp.py:73`) becomes conditional on the recipe's projection, default `clip_zero` so the magnitude path is byte-identical. The estimator needs no change: `TransformedTargetRegressor(StandardScaler)` already centers and scales a signed target, and the chronological early-stopping holdouts are label-sign agnostic. All 18 SIZE profile names are admitted (identity differs through `target_domain`, so no id collision with SIZE).

## 6. Selection and distribution

**Blended selection (P7), module `experiments/iv_selection.py`.** Per candidate, the lead-1 inner event rows `{event_id, horizon_sessions, crush_pct_iv30, predicted_crush_pct_iv30}` are pooled over the inner folds, and the baseline is the fold's training mean (the mean of that fold's admitted training labels, one number per fold; the signed analogue of `history_mean`, which falls back to a training mean when no prior exists). With `e = y - yhat` and baseline errors `e0 = y - mean`:

```
rmse = sqrt(mean(e^2))   mae = mean(|e|)
nrmse = rmse / rmse0     nmae = mae / mae0          (rmse0, mae0 from the training-mean arm on the same events)
blend_score = 0.67*nrmse + 0.33*nmae               (lower is better; weights fsum'd)
```

A zero baseline with a nonzero candidate error raises `ZeroComparatorError` (reused); a zero baseline with zero error normalizes to 1.0, as SIZE does. Candidate and baseline rows must hold the same events and targets. `selection_tie_band` is reused unchanged (lowest `blend_score`, `candidate_id` ascending on exact equality; the `max(SE, tie_band_floor)` band is a diagnostic) and so is `ZeroComparatorError`; only the event-row function is new because `blended_event_score` is bound to `absolute_move_pct` and the large-move term. Yearly scores for the SE use the same function per inner year. Dispatch is on `"target" in plan` (as the distribution report already does). The v1 `inner_pooled_rmse` path remains for v1 plans.

**Signed distribution (P8), module `experiments/iv_distribution_report.py`, schema `iv_distribution.v1`.** SIZE's band is multiplicative (`y/pred` ratios with `max(pred, 0.05)`), which is wrong for a signed target near zero. The signed band is additive: pool the inner OOF `(pred, y)` pairs, cut `pred` into 5 quantile buckets (`MIN_BUCKET_ROWS` 20, else `pool_too_small`), take quantiles of `y - pred` per bucket at `TAUS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)`, and band each scored forecast `f` as `f + q[bucket(f)]`. Reported: p10/p50/p90 of the band, `pinball_p10/p50/p90/mean`, `coverage_p10` (target at or under 10%), `coverage_p90`, `coverage_p10_p90` (nominal 80%), `n_pool`, `n_scored`, per horizon and pooled. It consumes the same `{horizon: (pred, y, event_ids)}` arrays as `size_distribution_report`. **The `unsupported_target` gate in `size_distribution_report.collect_distribution` stays** (it is the correct answer for SIZE's ratio machinery); the IV search report calls the new collector instead, so SIZE's report is not edited.

## 7. Paired report v2 (P9)

`iv_crush_paired_report.paired_report` keeps its v1 output; a new `paired_report_v2` returns `iv_crush_paired_report.v2`. Arms and forecasts (all in `crush_pct_iv30` percent units):
- `challenger`: the frozen selected candidate on outer 2025.
- `incumbent`: the causal production-HGB refit of the comparator (abstains on `unavailable_input` or `insufficient_training`, as v1).
- `training_mean`: the fold's training mean (v1 field).
- `implied_ex_earnings`: `100*(exern_iv30/iv30 - 1)` from the stored panel features. It abstains when either input is missing or non-positive; it is a reported arm and never a fallback for another.

Metrics per arm on its matched set and, for paired comparisons, on the **common-matched** set (every arm forecasts): `rmse`, `mae`, `bias = mean(yhat - y)`, count. The v1 coverage table is kept and extended with the new arm. For the challenger only: the `iv_distribution.v1` band metrics of section 6 on outer 2025. Bootstrap: 95% percentile intervals for the challenger-minus-each-other-arm differences in RMSE and MAE (and incumbent-minus-training-mean), 2000 resamples, seed 0, blocks by `ticker` and, separately, by event month; the RMSE difference is `sqrt(mean e_a^2) - sqrt(mean e_b^2)`, errors scaled by the largest absolute error as in `size_paired_report._bootstrap`. That function and `_implied_bootstrap` hard-code `absolute_move_pct`; P9 adds a `truth` keyword defaulting to it (SIZE output byte-identical, pinned by a test) rather than copying 30 lines. Claim stays `frozen_dataset_development_only`, `final_read_authority: none`. Reported numbers are measured; no threshold or promotion rule is added here.

## 8. Reuse, generalize, port

| Module | Disposition | Why |
| --- | --- | --- |
| `size_panel` closure, `derived_features.py`, `derived_rows.with_derived`, `read_stored_panel` | **Reuse unchanged** | Frozen; IV events are a subset of the stored panel and L/P are derived at fit time. |
| `magnitude_search_plan/outer/trials/worker/journal`, `size_comparison_run`, `workers` | **Reuse** | Already target-aware (`"target" in plan`); candidates carry `target_domain`. |
| `regression_search_binding.py` | **Generalize** (P1) | HMO constants become per-version; v1 path untouched. |
| `magnitude_search_request_v3` axis helpers and `_BOUNDS` | **Generalize** (P2) | Add a `target_domain` argument to the loss validator, default today's behaviour; a new `regression_search_request_v2.py` composes them. |
| `regression_candidate.py` (models and training) | **Generalize** (P2-P4) | Closed `_KEYS` and dispatch gain families by schema; P3 then P4 serialize merges. |
| `magnitude_elasticnet_spline_recipe`, `magnitude_mlp_recipe`/`magnitude_mlp.py` | **Port to signed twins** (P3, P4) | Domain lives in the recipe, as for ridge/EN/HGB; the magnitude schemas must not change ids. |
| `selection_tie_band`, `ZeroComparatorError` | **Reuse** | Metric-agnostic. |
| `blended_event_score` | **Port** (`iv_selection.py`) | Bound to `absolute_move_pct` and the tail term. |
| `size_distribution_report` | **Port** (`iv_distribution_report.py`) | Multiplicative ratio band is wrong for signed. |
| `size_paired_report._bootstrap` | **Generalize** (P9, `truth` keyword) | Same cluster bootstrap; byte-identical default. |
| Prior-crush history, IV derived rows (`iv_derived_rows.py`) | **Phase 2, new modules** | Must not enter the SIZE closure. |

## 9. PR split

All code PRs target about 150 and at most ~300 added non-test code lines (limit as of 2026-10-28; the brief's ~400 bound is the ceiling). Each states its contract in the PR body and adds no ARCHITECTURE growth. Estimates are mine.

| PR | Scope | Code / test lines | Depends on | Parallel with |
| --- | --- | --- | --- | --- |
| P1 | `regression_search_request.v2` skeleton: shape, `selection` and `workers` keys, version dispatch, set list, dictionary hash, closed-grid accounting for the v1 families; binding by version, subset panel read, smoke coverage report | ~250 / ~200 | this design | none (all others branch from it) |
| P2 | Wider grid: signed-valid HGB axes, EN bounds, loss refusal, leads selector, `regression_candidate_recipe` params | ~200 / ~150 | P1 | P3, P4, P7, P8, P9 (design-wise) |
| P3 | Signed spline recipe, validator, fit dispatch, `spline_knots` axis admission | ~200 / ~150 | P1 (axis admission after P2 merges) | P2, P4, P7, P8, P9 |
| P4 | Signed MLP recipe, fit projection switch, profile axis admission | ~200 / ~150 | P1; merges after P3 (both edit `regression_candidate.py`) | P2, P7, P8, P9 |
| P7 | `iv_selection.py`, plan `selection.metric` dispatch, yearly SE rows | ~200 / ~180 | P1 | everything except plan wiring conflicts with P2 |
| P8 | `iv_distribution_report.py` and the IV search report call | ~200 / ~160 | none (reads stored arrays) | all |
| P9 | `paired_report_v2`, ex-earnings arm, bootstrap `truth` keyword, coverage extension | ~250 / ~200 | none (pure function) | all |

(P5 and P6 are phase 2: IV derived features and the prior-crush store; P10 is dropped by decision 3.) Merge order: P1 first; then P8 and P9 any time; P2, P3, P7 in any order with rebases on `regression_search_binding`/`regression_candidate.py`; P4 after P3. A smoke on 50 tickers (`--no-ledger`) follows P1+P2+P7 and reports the coverage counts above and candidate and fit counts; the full run follows the remaining PRs, outer 2025 only, with 2025 unread before the sealed plan.

**Acceptance for the series.** (a) a v2 request with every set, a signed grid, spline and MLP candidates compiles, seals and runs; (b) every v1 request, candidate and plan id is unchanged, and the unsigned spline/MLP and SIZE paired-report outputs are byte-identical (tests); (c) `gamma`/`poisson` and wrong-set spline are refused at compile; (d) a failing fit fails its trial with nothing stored; (e) the closure check of section 2 prints nothing for every PR; (f) preflight refuses counts above `max_candidates`, `max_fits` or `max_component_fits`.

## 10. Open questions for the user

1. `outside > 0` events at plan time: refuse (default here) or exclude and report? Decide after the smoke measurement.
2. `tie_band_floor` value for the signed blend (SIZE's carried over by default).
