# SIZE averaged candidates: seed bagging and the HGB+elastic-net blend (design)

Design only; no code in this PR. The contract is promoted into `experiments/ARCHITECTURE.md` and `engine/v2/models/ARCHITECTURE.md` by the last slice, as in `guides/size_feature_groups_design.md`.

**Evidence (exploratory, not promotion evidence).** `stack.py`/`stack1.out` (owner inputs, 2026-11-12): 7 yearly folds 2018-2024, 2025 never loaded, direct parquet reads (reference only; the pipeline reads none). Averaging 5 HGB fits that differ only in `random_state`: RMSE -0.1% vs one fit, better in 6/7 years. Averaging that HGB (gamma, 31 leaves, lr 0.03, min_samples_leaf 300) with an elastic net (alpha 0.01, l1_ratio 0.75) on the same features: a further -0.19% RMSE, better 7/7 years, but MAE +0.8%. So the blend is a candidate to compare, never a default. The official search scores both levers and decides.

## 1. Decisions (cheapest correct option first)

| # | Question | Decision | Why |
| --- | --- | --- | --- |
| 1 | Seed bagging | An HGB axis `n_seeds`, integer in [1, 10]; member i uses `random_state = i`. Default `[1]`. | `n_seeds=1` is today's fit exactly (`random_state` 0) and its recipe omits the field, so declaring `[1]` or nothing gives the same recipe and id. A recipe with `n_seeds > 1` uses schema `magnitude_ridge_hgb_recipe.v3` (v1/v2 and their validators unchanged; the v3 validator rebuilds with the field). Members differ only by seed, which also changes the seeded 10% early-stopping holdout the recipe already uses. |
| 2 | Blend | A fixed-recipe family `blend` with one axis `profile` naming code-declared recipes. One profile now: `hgb_en_mean_v1`. | Same pattern as the MLP `profile` axis: identity is a recipe hash fixed in the sealed plan before any fit. Combining stored member forecasts at scoring time is rejected (section 2). |
| 3 | Slower trees | Already inside the request bounds (v4+ grid bounds; the `loss` and `max_iter` axes themselves are v9+, so v9 to v11 requests can declare it today): learning_rate 0.03 in [1e-3, 1], max_leaf_nodes 31 in [2, 64], min_samples_leaf 300 in [1, 1000], l2 10 in [0, 1000], max_iter 3000 in [100, 5000], loss `gamma` (v9). **Config only**, no bounds change. | A request can declare it today. Only `n_seeds` and the `blend` family need a request bump (v12). |
| 4 | Distribution band (#2401) | Unchanged. A bagged or blended trial stores one averaged forecast array; `size_distribution_report` pools and scores that array like any other. | It reads stored per-trial forecasts only, never members. |

Known differences from the exploratory script, so the official run is read as its own measurement: the pipeline HGB recipe fixes `n_iter_no_change` 10 (script: 20), early stopping uses the recipe's seeded random 10% holdout, and fits run single-threaded. No attempt is made to reproduce the script's numbers.

## 2. Blend: why a fixed-recipe family

Rejected: combining stored member forecasts at scoring time. It needs every member stored for every trial (the `TrialArrays` retention cap already trips on large grids), makes "which pairs" a choice over stored scores (picking members after seeing scores), and adds a second scoring path with its own identity, receipts and failure rules. A blend candidate is instead one ordinary candidate:

- Recipe `magnitude_blend_recipe.v1`: `{schema_version, model: "blend", profile, features, feature_dictionary_hash, aggregation: "weighted_mean", prediction_projection: "clip_zero", fit_scope: "fold_and_horizon", members: [{weight, recipe}]}`. Each member `recipe` is an ordinary validated `magnitude_ridge_hgb_recipe` or `magnitude_elasticnet_recipe` on the same columns.
- `hgb_en_mean_v1`: member 1 HGB (loss gamma, target floor 0.01, lr 0.03, leaves 31, min_samples_leaf 300, l2 10, max_iter 3000, `n_seeds` 5), weight 0.5; member 2 ElasticNet (alpha 0.01, l1_ratio 0.75), weight 0.5.
- `validate_blend_recipe` rebuilds the expected recipe from `profile`, columns and dictionary hash and requires canonical equality (the `validate_ridge_hgb_recipe` pattern). The profile table is a code constant; a drifted member parameter is a different recipe hash and fails validation, so identity cannot change after the plan seals.
- The feature set is the candidate spec's (`H` to `HMOSX`), as for every family. A new profile is a later one-constant change plus a request bump.
- `magnitude_candidate_family` gains the `(magnitude_blend_recipe.v1, blend)` pair; `fit_candidate` dispatches to `engine/v2/models/training/magnitude_blend.py`, which calls the existing member fitters on the same `FitInputs` and averages. Selection, inner/outer folds, rmse/mae/large-move metrics and the #2401 metrics are untouched because a trial still yields one forecast array.

## 3. Failure semantics (R1-R6)

- **R1 member failure.** Any member error (HGB, EN non-convergence, non-finite, shape) fails the whole trial, propagating the member's own exception unchanged (members raise `ValueError`; the blend adds no refusal code and the trial layer records it as it records the same error from a single candidate). No partial average, no reweighting, no retry. A failed blend trial is handled as any single candidate's failed trial today (distribution reports `trial_failed`).
- **R2 memory/CPU.** Members, and the seeds inside an HGB member, fit one at a time, single-threaded; peak = the largest member plus one prediction vector per finished member, freed after the mean. Nothing is retained beyond the single stored forecast.
- **R3 determinism.** Seeds `0..n-1`, fixed member order, fixed weights and summation order: equal inputs give equal bytes. Members clip at zero already, so the mean of clipped members is nonnegative and the blend's own `clip_zero` is a no-op guard.
- **R4 persistence.** Only the averaged forecast array is stored, as for any trial. Member forecasts are never persisted or scored.
- **R5 identity.** `candidate_id` hashes the full recipe including members, weights and `n_seeds`. The plan seals all candidate ids before any fit; nothing selects members from scores. `n_seeds` appears in a recipe only when it is greater than 1 (a profile member such as `hgb_en_mean_v1`'s HGB carries it as part of the profile's fixed recipe), so every v4-v11 recipe id stays byte-identical.
- **R6 causality.** Members see exactly the rows a single candidate sees for the same fold and horizon (fold-admitted training rows by data date; prediction rows only for the scored events). No holdout rows, no `computed_at` filter, no new reader.

## 4. Request v12, bounds and fit accounting

Request `magnitude_search_request.v12` (v4-v11 unchanged):

- `space.hgb` may declare `n_seeds` (optional, strictly increasing integers in [1, 10], default `[1]`, joining the `loss`/`max_iter`/`quantile` optionals). Quantile loss with `n_seeds > 1` is refused: averaging pinball-loss quantile fits is not a point forecast.
- `space.blend` is an optional key, `{"profile": [...]}` of known, unique, sorted profile names. Absent means no blend candidates, so a v12 request without it means what v11 meant.
- The 1000-candidate ceiling and `max_candidates` count each blend candidate as one candidate.

Fit accounting. `search_fits` stays **logical** (horizons x (candidates x inner folds + outer folds), unchanged), so `max_fits`, selection contracts and reports keep their meaning. The real cost is component fits: a candidate costs `n_seeds` (HGB), 1 (other families) or the sum of its members' component counts (`hgb_en_mean_v1` = 6). `total_component_fits` exists today as logical fits plus extras plus baseline components; v12 **changes its meaning** to horizons x sum over candidates of component cost x (inner folds + outer fold), plus the baseline components, and the existing `comparison.max_component_fits` must cover that whole total instead of only the baseline (no new request key). Preflight reports both counts and refuses with `INVALID_EXPERIMENT_SPEC` when the component total exceeds the cap. Wall-clock is bounded by the sealed job `limits`. Cross-candidate reuse (a blend refitting the five HGB seeds a standalone `n_seeds=5` candidate also fits) is **not** done: it would add a cache with its own invalidation rules; the duplicate cost is visible in the component count and accepted.

Progress logging: the new fit paths add no per-event or per-member lines; the existing bounded stage progress in the trial loop carries done/total, rate and ETA. Because that loop counts trials, S4 weights each trial by its component cost so done/total and ETA are in component fits, and the stage-start line reports the component total.

## 5. Split list and total effort

Total: 4 code PRs, about 750 non-test code lines, about 650 test lines and about 60 doc lines (S4), after this design PR. Each merges alone and is unreachable from a request until S4, the slice that exposes the v12 axes, so no request can pass preflight before execution is wired. Every slice stays under the ~300 line limit; S2 and S3 are over the ~150 target because each is one indivisible contract (recipe plus validator plus fit; compiler plus accounting).

| Slice | Scope | Code / test / doc lines |
| --- | --- | --- |
| S1 | HGB `n_seeds` in the recipe (v3 schema, field only when greater than 1), validator, seed-averaging fit in `magnitude_ridge_hgb.py`; tests incl. `n_seeds=1` equals the single fit and the quantile refusal | ~130 / ~150 / 0 |
| S2 | `magnitude_blend_recipe.v1`, `validate_blend_recipe`, `hgb_en_mean_v1` profile table, family pair, `magnitude_blend.py` fit, `candidate_recipe` family, `execution_for_candidate` for `blend`; tests incl. member failure, weights, determinism | ~250 / ~200 / 0 |
| S3 | Search wiring: component-fit weighted progress/ETA, preflight and report component counts, stage-start line; tests over a tiny panel (a blend and a 2-seed HGB scored with the distribution metrics, built from recipes directly) | ~120 / ~150 / 0 |
| S4 | Request v12: `LATEST` bump, `n_seeds` and `blend` axes, bounds, closed-grid and candidate counts, component-fit accounting and `max_component_fits` check, example request, ARCHITECTURE promotion | ~250 / ~150 / ~60 |

Acceptance criteria for the series: (a) a v12 request with `n_seeds [1, 5]` and `blend [hgb_en_mean_v1]` compiles, seals and runs; (b) `n_seeds=1` forecasts equal the pre-change single fit byte for byte; (c) every v4-v11 request id and recipe id is unchanged; (d) a failing member fails the trial with no partial forecast stored; (e) the #2401 metrics are produced for bagged and blended candidates; (f) preflight refuses a component total above `comparison.max_component_fits`.

After S4, a v12 request with `n_seeds [1, 5]` on the slower-tree HGB and `blend [hgb_en_mean_v1]` is the official run; 2025 stays unread.
