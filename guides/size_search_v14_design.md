# SIZE search v14: median-trained EN/MLP, three winners, fixed blends (design)

Design document; the code lands in the slices of section 6. Contract text lands in the ARCHITECTURE files with the slice that changes it (both are near their growth ceiling).

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

- Recipes. `elasticnet_recipe`, `elasticnet_spline_recipe` and `mlp_recipe` take `target_transform="none"|"log1p"`. `none` emits today's bytes (schema `.v1`, unchanged ids). `log1p` emits a new schema (`magnitude_elasticnet_recipe.v2`, `magnitude_elasticnet_spline_recipe.v2`, `magnitude_mlp_recipe.v2`) with `target_transform: {"kind": "log1p", "inverse": "expm1", "statistic": "median_if_log_residuals_symmetric", "loss": "squared_error_on_transformed"}`. Validators rebuild from fields and require canonical equality (existing pattern); a v2 recipe with `none` or a v1 recipe with `log1p` refuses. For the MLP the existing `StandardScaler` target stage stays; `log1p` is applied before it (`target_transform.then: "StandardScaler"`). Refusals: invalid request fields are `INVALID_EXPERIMENT_SPEC`; invalid recipes raise `ValueError` as today.
- Candidate parameters. A twin carries `target: "log1p_median"`; base candidates omit it, so base recipe ids are unchanged. `candidate_recipe._keys_ok` admits the extra key for elasticnet (plain and spline) and mlp only; HGB with it refuses.
- Fit. EN: the pipeline is wrapped in `TransformedTargetRegressor`; coefficient evidence reads `model.regressor_[-1]` and `coefficient_scale` becomes `standardized_inputs_log1p_target`. MLP standard and chrono paths use the chained transformer (chrono early stopping then scores on the transformed, standardized scale, which is the trained scale). `size_fit_diagnostics.evidence` accepts the new schemas and fields; the fit result gains `target_transform: "log1p"`.
- Failure semantics. (a) Labels are already required finite and nonnegative, so `log1p` is defined. (b) A non-finite `expm1` (overflow) raises `ValueError`, the same path as any non-finite prediction: variant failure, no clipping to hide it, no retry. (c) An all-zero training target gives a constant transformed target; for EN, coefficients are zero and the forecast is 0 (the MLP's `StandardScaler` guards zero variance, so the MLP fit completes with a finite near-zero forecast; exact zero is guaranteed for EN only; S1a tests both). It is not special-cased and scores normally. (d) Non-convergence is unchanged (`variant_failure_no_retry` for EN, the profile's own rule for MLP).
- Twin cells (24). EN: alpha {0.003, 0.01, 0.03} x l1_ratio {0.75} x spline_knots {0, 6} x feature sets HMOSX/HMOSXL/HMOSXLP = 18. MLP: profiles `chrono10_relu64x32_l2_1` and `chrono15_relu64x32_l2_1` x 3 sets = 6. Alpha reaches down to 0.003 because `log1p` shrinks the target scale, so the same alpha penalises about 5x more relative to the loss. HGB gets no twin (absolute-error HGB already targets the median; the brief limits twins to EN and MLP).
- Cost: EN twins are cheap; the six MLP twins add about 6 of 21 current MLP trials' worth of time.

## 3. Request `magnitude_search_request.v14`

`LATEST` 13 to 14. v14 inherits v13 entirely (HMOSXLP dictionary hash, v13 MLP profiles, spline axis, optional `ridge`). Every new field is optional, and a v14 request without them means what v13 meant, so slices can land one at a time. A v14 request is not used in a real run until all slices have merged.

| Field | Slice | Rule |
| --- | --- | --- |
| `space.log_target` | P1 | Optional object with keys from {`elasticnet`, `mlp`}. Each is a sub-grid with the family's normal axes. Every axis value must be a member of the same axis in the base `space` (twins are of existing cells), ordered/unique as the base rules; `spline_knots > 0` needs a spline-capable set (existing rule). Base family must be present. Twins count toward `max_candidates`, the 1000-candidate ceiling and the fit counts. Violations: `INVALID_EXPERIMENT_SPEC` before expansion. |
| `selection.winner_rules` | P2 | Optional; if present exactly `["overall", "rmse", "mae"]` (closed registered value, like the weights). Absent means overall only. |
| `selection.extra_refits` | P2 | Optional (user 2026-11-17), pre-registered. A sorted unique list of candidate ids that get an outer-year refit in addition to the winners. Request compile checks only shape (sorted, unique `sha256:<64 hex>` strings (the `content_hash` format of candidate ids); needs `winner_rules`); plan build resolves each id against the compiled grid, and an unknown id is `INVALID_EXPERIMENT_SPEC`. Ids are plan candidate ids (a hash of recipe, feature recipe, source binding and candidate spec), known only after the plan compiles, so the list is run config written after a dry compile, not code. Sealed as `plan.selection.extra_refits`. Extra units are deduped against winners at run time, share the per-candidate refit slots and the failure semantics of section 4, are never selected from outer results, and enter no rule. |
| `blends` | P3 | Optional sorted unique list of names from the code table `{"size_b1_v1", "size_b2_v1"}`. Each profile resolves its components against the compiled grid (section 5); a component with no matching compiled candidate, or whose feature set is not in `feature_sets`, refuses. Blends require both `selection` and `selection.winner_rules` (refused otherwise with `INVALID_EXPERIMENT_SPEC`), since component refit slots exist only under `winner_rules`. |

Counts. `counts.blends` is reported. Blends cost no inner fits, so `max_candidates` is unchanged. Outer refits rise: `fits = leads x (candidates x inner_folds + outer_folds x R)` with `R = 3` plus the number of distinct blend components plus 1 (the best-single refit, section 5) when blends are declared (without `winner_rules` only the overall winner is refit, so `R = 1`; blends require `selection` and are reachable only with `winner_rules`, so a blend request always has `R >= 3 + components + 1`). `max_fits` must cover it. For the intended run: 129 candidates, 2 inner years, one outer year, `R = 3 + 4 + 1 = 8`, so `leads x 266` (the interrupted request's caps of 105/315 must be raised). `extra_refits` adds its list length to `R` (a worst-case bound; dedupe against winners happens at run time), so the EXP-188 config below gives `R = 8 + 6 = 14` at most.

Plan. The plan schema stays `magnitude_search_plan.v3`. Keys appear only for v14 requests that use them: `plan.blends` (below), `plan.selection.winner_rules`, `plan.selection.extra_refits` and `plan.selection.rule_metrics = {overall: blend_score, rmse: normalized.rmse, mae: normalized.mae}`. `read_search_plan` reconstructs under the current contract, so a v13 plan and every v13 receipt stay byte-identical (acceptance test). Candidate ids change under v14 anyway (the request id is in the spec id), so nothing from the interrupted attempt is reusable.

## 4. P2: three winners

Selection (`magnitude_search_trials._freeze`). The scoring loop already holds, per candidate, the pooled `metrics`/`normalized` (lead-1 event rows, all inner years) and per-year scores. For each rule the score is `blend_score`, `normalized.rmse` or `normalized.mae`. Normalizing by the shared `history_mean` rows leaves the ordering of raw RMSE/MAE unchanged and keeps the 0.005 floor in the right units. `selection_tie_band` is called unchanged on rows whose `blend_score` key (and each yearly row's `blend_score`) is set to the rule's score (a small adapter). The selection record gains, per outer year, `winners: {overall|rmse|mae: {candidate_id, tie_band}}` and `distinct_winner_ids` (sorted). `selected_candidate_id` and `prospective_candidate_id` remain the overall winner, which stays the run's primary result. Failed candidates are excluded from all three rules exactly as now.

Outer refits (`magnitude_search_outer`). Today one refit is keyed `[plan_id, outer_year]`. For plans with `winner_rules`, refit slots are keyed per candidate: `[plan_id, outer_year, candidate_id]`. Distinct winners are refit once each; if two rules pick the same candidate they share the refit. `R` is a worst-case bound on refit units per outer year (3 winners plus distinct blend components), not an expected count. A blend winner is not itself a refit unit; its components are (section 5), so a component also winning alone is fitted once. Units per outer year are the overall winner first (its failure refuses the run, so it must not follow heavy refits), then the other distinct winners sorted, then `extra_refits` not already listed, sorted. A refit slot is committed once: its start and terminal records are replayed on resume exactly like today's `outer-*` records, a started slot without a terminal refuses `EXPERIMENT_IDENTITY_CONFLICT`, and a failed non-overall unit writes a terminal `status: failed` record (error code `EXPERIMENT_VARIANT_FAILED`) so resume never retries it. `collect` returns the overall folds unchanged plus `refits`, one row per unit with its fold result or failure. Replay of old plans keeps the old key. Fits run sequentially, one heavy job at a time, with the existing child limits. New loops log at bounded cadence to stderr: stage start/end with elapsed, refit i/N, RSS at boundaries.

Failure. A refit failure is recorded as that unit's terminal failure; no retry or replacement. The overall winner's failure refuses the report (as today). A failure in another winner's refit does not block the others; the report shows that winner as `status: failed` with the error code. `collect` still refuses an unfinished refit.

Report (`size_search_report.collect`). `report["paired"]` stays the overall winner vs the incumbent (unchanged shape). New `report["winners"]`: one row per distinct winner with `candidate_id`, `rules`, `kind` (candidate or blend), `components`, `target_transform`, `inner` (pooled RMSE, MAE, blend, normalized, tie band per rule), `paired` (the existing `paired_report` output for that winner's outer forecasts vs the shared incumbent comparison, which is keyed by outer year only and fitted once), `actual_above_forecast_share` and `status`. Blend rows add `inner_scores_optimistic: true` and the provenance note. The report labels the overall winner primary; the other rows are secondary and carry no multiplicity adjustment, and no outer result re-selects anything. The distribution band keeps reading the overall winner; when that is a blend it reports an explicit `blend_not_distributed` status in P3 (no averaged inner arrays are stored).

## 5. P3: fixed blend candidates

Profiles (code constant in a new `experiments/size_blend_profiles.py`, outside the panel closure), all equal weight, all on HMOSXLP:

- `size_b1_v1` = {spline EN alpha 0.03, l1_ratio 0.75, spline_knots 6; HGB absolute_error, leaves 15, min_samples_leaf 300; HGB squared_error, leaves 15, min_samples_leaf 1000; MLP profile per Decision B (default `chrono15_relu64x32_l2_1`)}.
- `size_b2_v1` = B1 minus the spline EN.

Checked against the interrupted v13 request (`space`: EN alpha {0.003, 0.01, 0.03, 0.1}, l1 {0.25, 0.75}, knots {0, 6}; HGB lr 0.03, l2 10, max_iter 3000, losses {absolute_error, gamma, squared_error}, leaves {7, 15}, msl {300, 1000}; MLP chrono profiles): the EN and both HGB components are exact grid cells. The MLP is ambiguous: `chrono10_relu64x32_l2_1` and `chrono15_relu64x32_l2_1` both match (Decision B). The profile table names the exact profile.

Identity. `blend_id = content_hash({profile, aggregation: "equal_mean", component_candidate_ids (sorted)})`, sealed in `plan.blends` with `defined_from: "interrupted_v12_inner_2023_2024"` and `inner_scores_optimistic: true`. B1/B2 were defined after seeing the interrupted attempt's inner years 2023/2024, so their inner scores are optimistic and likely to win the inner rules; only the outer 2025 result is a fair test.

Inner scoring. After an outer fold's candidate loop, for each blend the freeze averages the four (three) components' stored lead-1 rows per event in sorted component-id order with `math.fsum`, then scores them with the freeze's local `score_rows` wrapper of `blended_event_score` against the cached history rows, pooled and per year. Retained state is lead-1 rows of the component candidates only, dropped after the outer fold. Blends enter all three rules next to ordinary candidates. If any component trial failed in any inner year, the blend is excluded (failed-candidate rule). Rows that do not align on event ids and targets across components mean corrupted evidence: refuse `EXPERIMENT_IDENTITY_CONFLICT`, not exclusion.

Outer. A blend winner refits each component through the per-candidate refit slots above and averages the stored refit forecasts per (event, horizon); no clipping, so signed forecasts keep their sign. A component refit failure fails the blend winner (section 4 failure rule). A blend record stores component refit references and the averaged forecast id.

Best single model (user 2026-11-17). When a plan has blends, the run always also refits the best non-blend candidate under the overall rule (lowest `blend_score` among non-blend candidates, lowest `candidate_id` on exact ties) on each outer year, so the report shows single model vs blend vs incumbent even when blends win all three rules. It is a derived refit unit, not a request field: it exists exactly when `plan.blends` does, is deduped against the winners, the blend components and `extra_refits` (fitted once if it is already one of them), shares the per-candidate refit slots and the failure semantics of section 4, enters no rule and never reselects. The selection record names it per outer year as `best_non_blend_id`. The report lists it as a secondary `winners` row with `kind: candidate` and `rules: []` unless it also wins a rule. S3a only counts it in `R`; S3c implements it.

## 6. Split list

Sizes are estimated code lines / test lines / doc words (ARCHITECTURE contract text). Code target about 150, limit about 300.

| Slice | Scope | Code / test / doc words | Depends on |
| --- | --- | --- | --- |
| S1a | Log-target recipes (3), validators, `candidate_recipe` key, estimator wrapping, fit evidence and diagnostics; panel-closure-hash test | ~170 / ~200 / 60 | none |
| S1b | Request v14 field block `space.log_target` (+ `LATEST` bump if first to merge), counts and `max_fits` formula | ~130 / ~160 / 120 | S1a |
| S2a | `selection.winner_rules` field block (+ `LATEST` bump if first to merge), plan keys, three-rule freeze and selection record | ~150 / ~180 / 80 | none |
| S2b | Per-candidate refit slots (including the `v2_magnitude_search_run` driver that enumerates outer keys), multi-winner outer execution, collect, failure and crash-resume rules (a refit slot is committed once, replayed like today's `outer-*` records) | ~220 / ~250 / 100 | S2a |
| S2b-x | `selection.extra_refits` request field, plan resolution against the compiled grid, `R` and `max_fits` counts. S2b's refit units already read `plan.selection.extra_refits`, so this adds no execution code. Its own PR immediately after S2b (size; user decision 2026-11-17) | ~60 / ~90 / 40 | S2b |
| S2c | Report `winners` rows, check share | ~120 / ~150 / 60 | S2b |
| S3a | `blends` request field, profile table, `plan.blends`, counts | ~140 / ~150 / 100 | S2a (started early, user 2026-11-17; rebases once after S2b/S2c) |
| S3b | Blend inner scoring in the freeze | ~120 / ~180 / 40 | S3a |
| S3c | Blend outer refit by components, best-single refit (section 5), blend report rows, `blend_not_distributed` | ~190 / ~220 / 100 | S3b |

Refusal codes: request and blend-resolution errors `INVALID_EXPERIMENT_SPEC`; misaligned component rows `EXPERIMENT_IDENTITY_CONFLICT`; a refit failure is `EXPERIMENT_VARIANT_FAILED` per unit.

Order (user 2026-11-17, faster delivery): track P1 = S1a, S1b (log-target) and track P2 = S2a, S2b, S2c (three winners) run in parallel, both branching from main right after this design merges. Their modules are disjoint (P1: engine recipes/fit, `magnitude_candidate_recipe.py`, `space.log_target`; P2: selection, trials, outer, report, `selection.winner_rules`). Each track owns its own v14 field block and validator section (`_log_target` and `_winner_rules` as separate functions in the request validator), so the schema bump does not race: whichever PR merges first carries the `LATEST` 13 to 14 bump, the second rebases once onto it. Track P3 = S3a, S3b, S3c (blends) depends on P2 only (components are base grid cells, not twins) and starts when P2 merges. No stacked PRs: every slice branches from and merges to main; within a track, the next slice starts after the previous merges. After the last slice: fresh registration, panel snapshot reused, run. Each slice changes behaviour only for v14 requests that use its field, so v13 replay is unaffected. ARCHITECTURE text per slice: contract lines only, within the doc budget.

## 7. Acceptance criteria

1. A v14 request without new fields compiles, seals and verifies to the same schedule shape as v13; all v13 request, plan and receipt bytes are unchanged.
2. Base recipe ids are unchanged; a twin's recipe has the v2 schema and a different id.
3. `expm1` overflow fails the trial; an all-zero target gives zero EN coefficients and a 0 EN forecast (the MLP gives a finite, near-zero forecast, not guaranteed exactly 0); neither retries.
4. With `winner_rules`, two rules selecting one candidate give one refit and one report row listing both rules; exact ties pick the lowest id.
5. B1/B2 inner scores equal the scores of an explicitly averaged forecast array; a failed component excludes the blend.
6. The panel closure hash is unchanged.

### EXP-188 run config for `extra_refits`

User decision 2026-11-17 (`guides/size_decision_pnl_design.md` decision 1, option a). The P&L re-score needs 2025 forecasts for its three matched pairs (that design's section 3), all on HMOSXLP: six units. The list is run config; the ids are resolved from the compiled grid when the plan is built, so the members below are the intended pairs, not code.

- Pair H: HGB lr 0.03, l2 10, max_iter 3000, leaves 15, min_samples_leaf 300, loss `squared_error` (H-sq) and `absolute_error` (H-abs).
- Pair E: spline ElasticNet alpha 0.03, l1_ratio 0.75, knots 6, raw (E-raw) and its `log1p` twin (E-log).
- Pair N: MLP `chrono15_relu64x32_l2_1`, raw (N-raw) and its `log1p` twin (N-log).

Several of the six will often also be winners or blend components, so the real extra count is smaller. E-log and N-log exist only once track P1 has merged.

## 8. Decisions for the user

- **A. Ties.** Current code chooses the lowest score and uses the 0.005 band only as a diagnostic. Keep that (default), or choose the simplest candidate inside the band? The second needs a complexity ordering across families, which does not exist yet.
- **B. Blend MLP.** `chrono15_relu64x32_l2_1` (default) or `chrono10_relu64x32_l2_1`?
- **C. Twin breadth.** 24 twins as above (default), or widen (add l1 0.25, alpha 0.1, all MLP profiles)?
- **D. Blends win everything.** Because B1/B2 are inner-optimistic they may win all three rules, leaving one refit. Add an optional extra refit for the best non-blend overall candidate (default: no, as specified)?
- **E. Run timing.** The run waits for all eight slices. Blends cannot be added to a sealed plan later. Accept the wait (default), or start the run without P3?
