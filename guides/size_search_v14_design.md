# SIZE search v14: median-trained EN/MLP, three winners, fixed blends (design)

Design only; no code in this PR. Contract text lands in the ARCHITECTURE files with the slice that changes it (both are near their growth ceiling).

**Constraint (user 2026-11-17).** The restarted v12 full run is held until this lands. It reuses the stored v12 panel snapshot unchanged. No slice edits `experiments/size_panel.py`, `derived_features.py` or any of the 150 files in `source_closure(root, ["experiments/size_panel.py"])` (closure hash prefix `b286c0f3` at the pin), and no new module may be imported by that closure. Every file named below was checked to be outside it. Acceptance test (S1): recompute that closure hash before and after and require equality. The search/registration closure does change, so the run needs a fresh experiment id and registration; the panel snapshot is reused.

## 1. Decisions made

| # | Question | Decision | Why |
| --- | --- | --- | --- |
| 1 | log vs log1p | `log1p` on the percentage-point target | Moves can be exactly 0. `log(max(y, 0.01))` (the gamma-HGB floor) puts a zero at -4.6 against a typical 1.4, giving a few rows large leverage under squared loss. `log1p(0) = 0`, needs no floor knob, and inverts exactly. |
| 2 | Back-transform | Plain `expm1`, then the existing `clip_zero`. No smearing or mean correction. | The target is the median; a smearing factor would move the forecast toward the mean. |
| 3 | Tool | sklearn `TransformedTargetRegressor` with `FunctionTransformer(np.log1p, np.expm1, check_inverse=False)`; for the MLP it is chained before the existing `StandardScaler`. | The MLP already uses `TransformedTargetRegressor` (`_magnitude_mlp_estimator.py`); the chrono path calls `transformer.fit_transform/transform/inverse_transform`, which a `Pipeline` transformer also provides, so no hand-rolled transform. |
| 4 | Twin cells | 24 twins, all existing grid values (section 2). Grid goes 105 to 129 candidates. | Bounded; no new axis value. |
| 5 | Ties | Exact score equality picks the lowest `candidate_id` (current `CURRENT_TIE_RULE`). The max(sample SE, 0.005) band stays a reported diagnostic per rule. | That is what current code does: the band never chooses (`selection_tie_band`). See Decision A. |
| 6 | Blends | Scored from components' stored inner lead-1 predictions; no inner fits. | The earlier `size_ensemble_design.md` section 2 rejected stored-forecast blends because "which pairs" would be picked after seeing scores and `TrialArrays` would need every member. Here B1/B2 are two fixed profiles in the sealed plan, and only four components' lead-1 rows are retained. That section is superseded for blends; its seed-bagging part is untouched and unbuilt. |

**Caveat on "median-trained".** Squared loss on `log1p(y)` estimates the conditional mean of the log target. Its `expm1` equals the median of y only if log-scale residuals are symmetric. That is an assumption, not a guarantee. The report therefore adds a check row (section 4): the share of outer events with actual above forecast, for every refit winner (about 0.5 supports the median reading).

## 2. P1: log-target recipes (engine)

- Recipes. `elasticnet_recipe`, `elasticnet_spline_recipe` and `mlp_recipe` take `target_transform="none"|"log1p"`. `none` emits today's bytes (schema `.v1`, unchanged ids). `log1p` emits a new schema (`magnitude_elasticnet_recipe.v2`, `magnitude_elasticnet_spline_recipe.v2`, `magnitude_mlp_recipe.v2`) with `target_transform: {"kind": "log1p", "inverse": "expm1", "statistic": "median", "loss": "squared_error_on_transformed"}`. Validators rebuild from fields and require canonical equality (existing pattern); a v2 recipe with `none` or a v1 recipe with `log1p` refuses.
- Candidate parameters. A twin carries `target: "log1p_median"`; base candidates omit it, so base recipe ids are unchanged. `candidate_recipe._keys_ok` admits the extra key for elasticnet (plain and spline) and mlp only; HGB with it refuses.
- Fit. EN: the pipeline is wrapped in `TransformedTargetRegressor`; coefficient evidence reads `model.regressor_[-1]` and `coefficient_scale` becomes `standardized_inputs_log1p_target`. MLP standard and chrono paths use the chained transformer (chrono early stopping then scores on the transformed, standardized scale, which is the trained scale). `size_fit_diagnostics.evidence` accepts the new schemas and fields; the fit result gains `target_transform: "log1p"`.
- Failure semantics. (a) Labels are already required finite and nonnegative, so `log1p` is defined. (b) A non-finite `expm1` (overflow) raises `ValueError`, the same path as any non-finite prediction: variant failure, no clipping to hide it, no retry. (c) An all-zero training target gives a constant transformed target; sklearn's scalers guard zero variance, EN returns zero coefficients and the forecast is 0. It is not special-cased and scores normally. (d) Non-convergence is unchanged (`variant_failure_no_retry` for EN, the profile's own rule for MLP).
- Twin cells (24). EN: alpha {0.003, 0.01, 0.03} x l1_ratio {0.75} x spline_knots {0, 6} x feature sets HMOSX/HMOSXL/HMOSXLP = 18. MLP: profiles `chrono10_relu64x32_l2_1` and `chrono15_relu64x32_l2_1` x 3 sets = 6. Alpha reaches down to 0.003 because `log1p` shrinks the target scale, so the same alpha penalises about 5x more relative to the loss. HGB gets no twin (absolute-error HGB already targets the median; the brief limits twins to EN and MLP).
- Cost: EN twins are cheap; the six MLP twins add about 6 of 21 current MLP trials' worth of time.

## 3. Request `magnitude_search_request.v14`

`LATEST` 13 to 14. v14 inherits v13 entirely (HMOSXLP dictionary hash, v13 MLP profiles, spline axis, optional `ridge`). Every new field is optional, and a v14 request without them means what v13 meant, so slices can land one at a time. A v14 request is not used in a real run until all slices have merged.

| Field | Slice | Rule |
| --- | --- | --- |
| `space.log_target` | P1 | Optional object with keys from {`elasticnet`, `mlp`}. Each is a sub-grid with the family's normal axes. Every axis value must be a member of the same axis in the base `space` (twins are of existing cells), ordered/unique as the base rules; `spline_knots > 0` needs a spline-capable set (existing rule). Base family must be present. Twins count toward `max_candidates`, the 1000-candidate ceiling and the fit counts. Violations: `INVALID_EXPERIMENT_SPEC` before expansion. |
| `selection.winner_rules` | P2 | Optional; if present exactly `["overall", "rmse", "mae"]` (closed registered value, like the weights). Absent means overall only. |
| `blends` | P3 | Optional sorted unique list of names from the code table `{"size_b1_v1", "size_b2_v1"}`. Each profile resolves its components against the compiled grid (section 5); a component with no matching compiled candidate, or whose feature set is not in `feature_sets`, refuses. Blends need `selection` (they are scored by it). |

Counts. `counts.blends` is reported. Blends cost no inner fits, so `max_candidates` is unchanged. Outer refits rise: `fits = leads x (candidates x inner_folds + outer_folds x R)` with `R = 3` plus the number of distinct blend components when blends are declared (`R = 1` for a request without `winner_rules`). `max_fits` must cover it. For the intended run: 129 candidates, 2 inner years, one outer year, `R = 3 + 4 = 7`, so `leads x 265` (the interrupted request's caps of 105/315 must be raised).

Plan. The plan schema stays `magnitude_search_plan.v3`. Keys appear only for v14 requests that use them: `plan.blends` (below), `plan.selection.winner_rules` and `plan.selection.rule_metrics = {overall: blend_score, rmse: normalized.rmse, mae: normalized.mae}`. `read_search_plan` reconstructs under the current contract, so a v13 plan and every v13 receipt stay byte-identical (acceptance test). Candidate ids change under v14 anyway (the request id is in the spec id), so nothing from the interrupted attempt is reusable.

## 4. P2: three winners

Selection (`magnitude_search_trials._freeze`). The scoring loop already holds, per candidate, the pooled `metrics`/`normalized` (lead-1 event rows, all inner years) and per-year scores. For each rule the score is `blend_score`, `normalized.rmse` or `normalized.mae`. Normalizing by the shared `history_mean` rows leaves the ordering of raw RMSE/MAE unchanged and keeps the 0.005 floor in the right units. `selection_tie_band` is called unchanged on rows whose `blend_score` key is set to the rule's score (a small adapter). The selection record gains, per outer year, `winners: {overall|rmse|mae: {candidate_id, tie_band}}` and `distinct_winner_ids` (sorted). `selected_candidate_id` and `prospective_candidate_id` remain the overall winner, which stays the run's primary result. Failed candidates are excluded from all three rules exactly as now.

Outer refits (`magnitude_search_outer`). Today one refit is keyed `[plan_id, outer_year]`. For plans with `winner_rules`, refit slots are keyed per candidate: `[plan_id, outer_year, candidate_id]`. Distinct winners are refit once each; if two rules pick the same candidate they share the refit. A blend winner is not a refit unit; its components are (section 5), so a component also winning alone is fitted once. Replay of old plans keeps the old key. Fits run sequentially, one heavy job at a time, with the existing child limits. New loops log at bounded cadence to stderr: stage start/end with elapsed, refit i/N, RSS at boundaries.

Failure. A refit failure is recorded as that unit's terminal failure; no retry or replacement. The overall winner's failure refuses the report (as today). A failure in another winner's refit does not block the others; the report shows that winner as `status: failed` with the error code. `collect` still refuses an unfinished refit.

Report (`size_search_report.collect`). `report["paired"]` stays the overall winner vs the incumbent (unchanged shape). New `report["winners"]`: one row per distinct winner with `candidate_id`, `rules`, `kind` (candidate or blend), `components`, `target_transform`, `inner` (pooled RMSE, MAE, blend, normalized, tie band per rule), `paired` (the existing `paired_report` output for that winner's outer forecasts vs the shared incumbent comparison, which is keyed by outer year only and fitted once), `actual_above_forecast_share` and `status`. Blend rows add `inner_scores_optimistic: true` and the provenance note. The report labels the overall winner primary; the other rows are secondary and carry no multiplicity adjustment, and no outer result re-selects anything. The distribution band keeps reading the overall winner; when that is a blend it reports an explicit `blend_not_distributed` status in P3 (no averaged inner arrays are stored).

## 5. P3: fixed blend candidates

Profiles (code constant in a new `experiments/size_blend_profiles.py`, outside the panel closure), all equal weight, all on HMOSXLP:

- `size_b1_v1` = {spline EN alpha 0.03, l1_ratio 0.75, spline_knots 6; HGB absolute_error, leaves 15, min_samples_leaf 300; HGB squared_error, leaves 15, min_samples_leaf 1000; MLP [64, 32] alpha 1}.
- `size_b2_v1` = B1 minus the spline EN.

Checked against the interrupted v13 request (`space`: EN alpha {0.003, 0.01, 0.03, 0.1}, l1 {0.25, 0.75}, knots {0, 6}; HGB lr 0.03, l2 10, max_iter 3000, losses {absolute_error, gamma, squared_error}, leaves {7, 15}, msl {300, 1000}; MLP chrono profiles): the EN and both HGB components are exact grid cells. The MLP is ambiguous: `chrono10_relu64x32_l2_1` and `chrono15_relu64x32_l2_1` both match (Decision B). The profile table names the exact profile.

Identity. `blend_id = content_hash({profile, aggregation: "equal_mean", component_candidate_ids (sorted)})`, sealed in `plan.blends` with `defined_from: "interrupted_v12_inner_2023_2024"` and `inner_scores_optimistic: true`. B1/B2 were defined after seeing the interrupted attempt's inner years 2023/2024, so their inner scores are optimistic and likely to win the inner rules; only the outer 2025 result is a fair test.

Inner scoring. After an outer fold's candidate loop, for each blend the freeze averages the four (three) components' stored lead-1 rows per event in sorted component-id order with `math.fsum`, then calls the same `score_rows` against the cached history rows, pooled and per year. Retained state is lead-1 rows of the component candidates only, dropped after the outer fold. Blends enter all three rules next to ordinary candidates. If any component trial failed in any inner year, the blend is excluded (failed-candidate rule). Rows that do not align on event ids and targets across components mean corrupted evidence: refuse `EXPERIMENT_IDENTITY_CONFLICT`, not exclusion.

Outer. A blend winner refits each component through the per-candidate refit slots above and averages the stored refit forecasts per (event, horizon), clip-zero as a no-op guard. A component refit failure fails the blend winner (section 4 failure rule). A blend record stores component refit references and the averaged forecast id.

## 6. Split list

Sizes are estimated non-test code lines. Targets are about 150, limit about 300.

| Slice | Scope | Code | Depends on |
| --- | --- | --- | --- |
| S1a | Log-target recipes (3), validators, `candidate_recipe` key, estimator wrapping, fit evidence and diagnostics; panel-closure-hash test | ~170 | none |
| S1b | Request v14 + `LATEST`, `space.log_target`, counts and `max_fits` formula | ~130 | S1a |
| S2a | `selection.winner_rules`, plan keys, three-rule freeze and selection record | ~150 | none |
| S2b | Per-candidate refit slots, multi-winner outer execution, collect, failure rules | ~220 | S2a |
| S2c | Report `winners` rows, check share, report tests | ~120 | S2b |
| S3a | `blends` request field, profile table, `plan.blends`, counts | ~140 | S1b, S2a |
| S3b | Blend inner scoring in the freeze | ~120 | S3a |
| S3c | Blend outer refit by components, blend report rows, `blend_not_distributed` | ~150 | S2c, S3b |

Order: S1a, S2a first (independent; S1b and S2a both touch the request/selection code, rebase small), then S1b, S2b, S2c, S3a, S3b, S3c. After S3c: fresh registration, panel snapshot reused, run. Each slice changes behaviour only for v14 requests that use its field, so v13 replay is unaffected. ARCHITECTURE text per slice: contract lines only, within the doc budget.

## 7. Acceptance criteria

1. A v14 request without new fields compiles, seals and verifies to the same schedule shape as v13; all v13 request, plan and receipt bytes are unchanged.
2. Base recipe ids are unchanged; a twin's recipe has the v2 schema and a different id.
3. `expm1` overflow fails the trial; an all-zero target forecasts 0; neither retries.
4. With `winner_rules`, two rules selecting one candidate give one refit and one report row listing both rules; exact ties pick the lowest id.
5. B1/B2 inner scores equal the scores of an explicitly averaged forecast array; a failed component excludes the blend.
6. The panel closure hash is unchanged.

## 8. Decisions for the user

- **A. Ties.** Current code chooses the lowest score and uses the 0.005 band only as a diagnostic. Keep that (default), or choose the simplest candidate inside the band? The second needs a complexity ordering across families, which does not exist yet.
- **B. Blend MLP.** `chrono15_relu64x32_l2_1` (default) or `chrono10_relu64x32_l2_1`?
- **C. Twin breadth.** 24 twins as above (default), or widen (add l1 0.25, alpha 0.1, all MLP profiles)?
- **D. Blends win everything.** Because B1/B2 are inner-optimistic they may win all three rules, leaving one refit. Add an optional extra refit for the best non-blend overall candidate (default: no, as specified)?
- **E. Run timing.** The run waits for all eight slices. Blends cannot be added to a sealed plan later. Accept the wait (default), or start the run without P3?
