# Rearchitecture Phase 5 — Frozen models and current training lifecycle

Status: implementation plan, 2026-09-16. Authority:
[delivery plan](rearchitecture_delivery_plan.md), system design §7 and model
contracts. Begin artifact inventory alongside Phase 4; final 4/5 integration
is one native scoring milestone, not a circular dependency.

## Outcome and minimum scope

Scoring loads exact immutable model, preprocessing, fold, calibration and
residual artifacts. It never trains or writes a model cache. Supervised jobs
reproduce the existing training and evidence workflows, and a deployment pointer
selects a compatible artifact set atomically. Preserve current model recipes;
do not turn this into a general ML platform.

The inventory includes the actual current champions, monthly feature-model
folds, full-refit diagnostics, gate and chooser models, feature order/transforms,
residual buckets, paired simulation state, analog state and trailing thresholds.
Saved point predictions alone are not a complete model release.

## Ownership and dependency with Phase 4

`engine/v2/models/` owns registry resolution, artifact verification and inference.
`models/training/` owns fitting/dataset preparation/evidence at layer 6.
`features/` and `scoring/` consume frozen state; no training import can point
upward into a score request. Ops owns admission and execution.

P5-1/P5-2 supply the inference seam needed by P4-3. Current preparation can run
as an explicitly supervised compatibility job while native training extraction
lands; its entire read set, output identity and barrier remain explicit. This
does not count as native Phase 5 completion until P5-3/P5-4 finish.

## Ordered implementation assignments

| Task | Deliverable | Acceptance / negative control |
|---|---|---|
| P5-1 Artifact inventory and release schema | Map current registry keys to strategy/role/clock without rewriting old manifests; list exact files/state needed for each producer/fold. | Missing transform/residual member, incompatible feature order and unknown clock refuse; every current model role is covered. |
| P5-2 Read-only inference | Verified frozen loaders, existing estimator adapters and cache-independent inference for Phase 4. | Rig all fitting/provider/cache-write paths to fail; cold and warm requests agree; missing artifact returns MODEL_NOT_READY, never trains. |
| P5-3 Current dataset/training recipes | Extract existing recipe membership, targets, masks, transforms, folds, seeds and upstream OOS dependencies into supervised jobs. | Training membership and label-availability receipts; planted future label/upstream in-sample leakage fails; interrupted fold resumes without losing completed artifacts. |
| P5-4 Residuals and correction propagation | Persist marginal and paired residual/calibration state and all uncertainty inputs; integrate 3B changesets. | Request context cannot change frozen state; historical correction invalidates every dependent later fold/state; equivalent rebuild agrees. |
| P5-5 Evidence and atomic deployment | Evidence built from exact dataset recipe and artifact fingerprints; stage complete deployments, promote/rollback via one pointer. | Partial/incompatible set cannot promote; prior scores remain replayable; Models evidence changes with the actual artifact/dataset, not feature-list guesses. |
| P5-6 Integrated acceptance | Current recipes, native inference and Phase 4 scoring on frozen real inputs; tested preparation and rollback runbook. | All roles and score consumers covered; exact pointer rollback; no runtime fitting or model-cache writes; private report. |

## Parity rules

Monthly Tier-4 and full-refit champion predictions are separate outputs.
Preserve expanding-year evaluation, monthly fold cutoffs, target transforms,
missing masks, calibrated thresholds and residual fallback rules. An available
new model family is not a reason to change any of them.

A model correction/promotion has an explicit blast radius across feature
producers, gates, chooser, simulation and evidence. Corrections can invalidate
all later folds. Full dependent-fold rebuilding is acceptable before Phase 8C
optimization, provided it is bounded and measured. Neither a stale cache hit
nor recomputing the residual pool from a smaller scoring context proves parity.

## Tests and handoff

Add a Phase 5 acceptance registry/gate as implementation work. Named subjects:
release completeness, inference cold/warm/no-fit, causal dataset membership,
residual identity, correction invalidation, promotion interruption and rollback.
Use pinned real artifacts for final equivalence, with synthetic corruption and
leakage controls. Keep heavy fits sequential and resource-admitted.

The report records recipe/model/runtime/seed identities, input membership,
OOS lineage, uncertainty/residual hashes, score parity and actual gate status.
Handoff includes the complete deployment, preparation commands, evidence,
correction dependencies and rollback ref. Do not promote a different economic
champion merely to exercise deployment plumbing.

Generalized backends, new calibration algorithms, model-selection studies and
training UX go to Phase 8C or separate experiments. Current retraining,
evidence, promotion and rollback capabilities remain pre-cutover requirements.

## P5-1 inventory results

Delivered 2026-09-18: `engine/v2/models/inventory.py`
(`current_release_inventory`, `registry_drift_issues`, `tier4_fold_coverage`,
`non_model_state_inventory`, `served_roles`) and `tools/phase5_inventory.py`,
which writes the whole thing to a JSON file and prints a role/strategy
coverage table. It reuses `ModelReleaseInventory`/`ReleaseBinding`/
`ReleaseRequirement`/`release_issues`/`require_complete_release` exactly as
Phase 0 defined them; nothing in `engine/v2/models/releases.py` changed.
Run against the real root on 2026-09-18 (`INVESTING_PLAN_ROOT=/root/investing-plan
python3 tools/phase5_inventory.py --out <path>`): **0 release_issues, 0
real-file issues, all 7 champion role/strategy bindings complete** —
`registry.json`'s recorded hashes and feature lists agree with the real
`data/models/*.joblib` files byte for byte.

### What's complete

All six `engine.models.registry.ROLES` have exactly one bound champion today
(`gate` twice, once per strategy — STR-THRU and STR-RUNUP): `size_v1_4`,
`opf_implied_t1_gbm`, `runup_move_d14_v1_gbm`, `iv_crush_v1_gbm`,
`gate_midfill_str_thru_forecast_analog`, `gate_midfill_str_runup`,
`dyn_sv_chooser_v1_1`. Each has a hash-verified `estimator` member and a
`residual` member (the flat held-out pool, embedded in the same pickle); the
two gates additionally carry a `threshold` member (hashed off `registry.json`
itself, since the threshold is manifest metadata, not a separate file); `size`
and `runup_move` additionally carry `residual_bucket` (decile-conditioned
pools) — the other two feature roles and both gates fall back to the flat pool
by design (`ModelArtifact.residual_pool`'s documented behavior), which is not
a gap.

### What's missing, per role — the P5-2/P5-4 gaps

- **No role ships an independent `transform` member.** A target transform
  (`LogTargetRegressor` for `runup_move`) or a blend/ensemble wrapper
  (`BlendModel` for `size`, `MeanEnsemble` for `chooser`) is embedded in the
  pickled Python class, not a separate, independently-hashable artifact.
  `current_release_inventory` deliberately does not require a `transform`
  member today (it would make every real release permanently, unfixably
  incomplete); a negative-control test proves the existing
  `MISSING_TRANSFORM_MEMBER` refusal still fires the moment a binding
  requires one. Extracting the transform as its own versioned member is
  P5-3 work (dataset/training recipes).
- **Tier-4 monthly fold coverage is uneven.** 103 fold files exist across the
  four feature roles (`size_v1_4` 41, `opf_implied_t1_gbm` 21,
  `runup_move_d14_v1_gbm` 21, `iv_crush_v1_gbm` 20). 47 of the 103 (46%) do
  not embed `pool_pred`/`pool_res` and would take `serving_model`'s
  documented backward-compatible recompute path if ever re-served — not a
  correctness bug, but real evidence the 2026-09-15 memory fix has not
  reached every historical fold. Separately, 10 of the 103 are **orphaned
  duplicate-snapshot files**: a month whose panel was rebuilt more than once
  (`tier3_snapshot` changed) accumulates one cache file per rebuild with no
  cleanup — `iv_crush_v1_gbm`'s 2026-09 fold alone has 7 distinct snapshot
  files on disk. Harmless (`serving_model`'s cache-hit check pins the exact
  current snapshot hash, so only one file per month is ever actually read)
  but unbounded disk growth with no retention policy — a P5-3/P5-4 cleanup
  item, not a correctness one.
- **Five of eight non-model serving states have no persisted artifact at
  all**, reported by `non_model_state_inventory`: the STR-THRU and STR-RUNUP
  payoff lines (`engine.payoff.fit_payoff`), the STR-RUNUP payoff surface
  (`fit_runup_payoff`), and the board's bucket-analog population
  (`engine.analogs.AnalogMatcher`) are refit **live, inside every Scorer
  construction**, from the trades ledger — no registry entry, no content
  hash, no completeness check, ever. Two more are persisted as *inputs* but
  not as the state itself: the paired move/crush residual pool
  (`engine.pnl_sim.ResidualPool`, built from `tier4_forecasts.parquet` joined
  against a live-recomputed panel/crush table — already flagged elsewhere as
  scorer-context-dependent) and the trailing `pnl_sim` gate cutoff (a
  quantile of `pnl_sim_history.parquet`, recomputed per `as_of` date, never
  written back). Only the DYN-SV chooser's k-NN analog pool
  (`chooser_analog_pool.parquet`) is a real, stable file — and even it is
  bound to no release or registry entry (`present_unversioned`). This whole
  category is P5-4's remit by name ("Residuals and correction propagation");
  recording it here rather than forcing it into a fake `ModelRole` is this
  module's answer to "what's missing," not a workaround.
- **No `decision_offset` variant exists yet.** `registry.json` has zero
  entries with a non-`None` `decision_offset`, so the early-decision clock
  plumbing the registry and `ReleaseBinding.clock_id` already support is
  entirely unexercised by real data — every current binding uses the single
  clock `legacy.entry_close.v1` (matching
  `engine.v2.registry.strategies.default_registry()`). Nothing to fix; just
  unproven until a D+1/D-1 gate is trained.

### What the guide's list did not anticipate

- **`iv_crush` is never served through `Scorer.model()`.** The other five
  roles reach `registry.load_champion` through `self.model(role, ...)`;
  `iv_crush` only resolves its champion inside
  `engine.data.features.tier4.iv_crush_feature_model()`, a different call
  path entirely. A hand-typed "roles the scorer serves" list would plausibly
  miss it — an early draft of this module's own role-coverage check did,
  until it was rewritten to scan `engine/score.py`'s literal text for every
  `engine.models.registry.ROLES` value instead of grepping for
  `self.model("...")` call sites. `served_roles()` and its test
  (`test_every_served_role_is_bound_in_the_real_registry`) are built the
  robust way for exactly this reason.
- **A second, code-embedded calibration surface exists outside payoff/
  recalibration.** `Scorer._N_ADMISSIBLE_BY_DEPTH` (an 18-point piecewise map
  from live chain depth to a training-time admissibility count, used to serve
  a feature the DYN-SV chooser was trained on but cannot see live) is a
  literal Python tuple in `engine/score.py`, not a versioned artifact of any
  kind. It is invisible to this inventory, to the registry, and to any
  release check — a third calibration category (fitted lookup table baked
  into source) the guide's "calibration (payoff line/surface)" language does
  not name.
- **Replay determinism for historical scores is not fully reconstructable
  from the registry alone.** Because the five live-refit states above carry
  no artifact identity, a past `ScoreResult`'s exact payoff line or
  recalibration map cannot be recovered from `registry.json` — only from the
  ledger state as of that score's `as_of`, which is not itself snapshotted.
  Worth flagging for P5-4/P5-5 (correction invalidation, replayable scores).

### Coverage table (2026-09-18, real root)

| role | strategy | members verified | status |
|---|---|---|---|
| size | * | estimator, residual, residual_bucket | complete |
| implied_t1 | * | estimator, residual | complete |
| runup_move | * | estimator, residual, residual_bucket | complete |
| iv_crush | * | estimator, residual | complete |
| gate | STR-THRU | estimator, residual, threshold | complete |
| gate | STR-RUNUP | estimator, residual, threshold | complete |
| chooser | DYN-SV | estimator, residual | complete |

Run it yourself:

    INVESTING_PLAN_ROOT=/root/investing-plan python3 tools/phase5_inventory.py \
        --out reports/phase5_model_inventory.json

Tests: `tests/test_v2_models_inventory.py` (14 cases: negative controls for a
missing residual/transform member, an incompatible feature order and an
unknown clock, all refusing through the existing `release_issues` codes;
real-file hash-drift and missing-file detection on synthetic fixtures; Tier-4
fold-coverage present/incompatible detection; non-model-state presence
detection; and served-role coverage against the real, git-tracked
`registry.json`).

## P5-3 recipe results

Delivered 2026-09-18 in `engine/v2/models/training/` (layer 6):
`recipes.py`/`calibration.py` (`current_recipes()`), `folds.py`
(`prepare_dataset`, `plan_folds`), `receipts.py` (`fold_receipts`,
`receipt_issues`), `estimators.py` (`fit_recipe_estimator`), `job.py` +
`fold_store.py` (`run_training_job`), and the real-data entry
`tools/phase5_training_job.py`. Tests: `tests/test_v2_models_training_recipes.py`.

- **Keys.** 16 recipes keyed `RecipeKey(role, strategy, output)`: the 7
  inventory champion bindings (`output="champion"`: expanding-year
  walk-forward plus full refit), the 4 Tier-4 producers
  (`output="tier4_monthly"`; separate identity from the champion, per the
  parity rules), and 5 calibration surfaces named after P5-1's
  `NON_MODEL_STATE_ITEMS` (`output="calibration"`). The two payoff lines and
  the STR-RUNUP surface are fitted per (cutoff, fill alpha) through P5-4's
  `engine.v2.models.training.payoff` builders: the fold writes
  `payoff_artifact.json` in `serialize_payoff_artifact` bytes, identical to
  the builder's own output on the same rows, and the fold's members are
  exactly the rows legacy `fit_payoff` keeps (`n` agrees). The two
  recalibration maps are fitted the same way since 2026-09-18 (see "P5-2
  acceptance audit and recalibration artifact" below).
- **Legacy is the spec.** Feature order comes from `registry.json`;
  constants are cross-checked against the legacy modules; `plan_folds`
  reproduces `walk_forward`/`fit_final` and `tier4.build_producer` row for
  row (tested by recording the legacy fits); each native estimator predicts
  bit-identically to the legacy `fit()`. Target transforms (`log1p_clip0`,
  `quantile_normal`) and blends are explicit recipe fields and named
  wrappers, not hidden in a pickled class.
- **Receipts.** Each fold directory carries a training-membership receipt
  (recipe/dataset fingerprints, member-key hash, counts, time range) and a
  label-availability receipt (label dates vs cutoff, upstream lineage).
  Refusal codes: `FUTURE_MEMBER`, `FUTURE_LABEL`, `LABEL_TIME_MISSING`,
  `UPSTREAM_IN_SAMPLE`, `UPSTREAM_LINEAGE_MISSING`,
  `UPSTREAM_MODEL_MISMATCH`, `RECEIPT_MISMATCH`, `RESUME_MISMATCH`,
  `ARTIFACT_CORRUPT`. Legacy admits a label up to one post-print session
  after a fold cutoff (a print on the fold's last day); receipts count those
  (`n_labels_after_cutoff`) and refuse only beyond the recipe's
  `max_days_after_cutoff`.
- **Resume.** A fold is published by an atomic directory rename after its
  hashes are written; a rerun re-derives and compares its receipts and keeps
  it, never refits it.
- **No-fit.** `run_training_job` and `fit_recipe_estimator` call the shared
  `engine.models.no_fit.forbid_fitting` first (one new declared adapter,
  ceiling 66 → 67: legacy cannot import a v2 guard, so this is the only way
  the P5-2 scoring guard covers v2 fits). No engine module outside the
  package imports it.

Not faithfully extractable, recorded rather than guessed:

- **Chooser full refit.** `dyn_sv_chooser_v1_1` was fit "on all
  exit-complete menu7-prime events, 2018-2026" by code that is not in the
  repo. The walk-forward folds are EXP-169 `generate()`; the recipe's
  `full-refit` fold (every complete row) is not proven to be the registered
  artifact's membership.
- **Chooser label bound.** Menu structures exit at varying dates and legacy
  sets no label-availability bound, so the chooser's `LabelRule` is
  unbounded: receipts report late labels, they cannot refuse them.
- **Unrecorded upstream lineage.** The STR-THRU gate's analog columns and
  the chooser's candidate-table forecasts (`pred_abs_move`,
  `pred_abs_move_sd`, `exp_pnl_sim*`) and causal analogs carry no fold
  lineage; receipts count them `n_unverified`, never verified.
- **Label dates for panel recipes.** The panel stores no post-print date:
  the tool uses next business day (size) and `date + MAX_GAP_DAYS`
  (iv_crush) as upper bounds, not observed dates.

## P5-2 acceptance audit and recalibration artifact

Delivered 2026-09-18. Tests: `tests/test_v2_models_p5_2_acceptance.py`,
`tests/test_v2_models_recalibration_artifact.py`.

| P5-2 acceptance point | test |
|---|---|
| legacy fit/cache-write paths rigged: Tier-4 `fit_fold`, `serving_model` miss, `ModelArtifact.save`, the six training `fit`s, `fit_payoff`, `fit_runup_payoff`, `fit_recalibration` | `tests/test_v2_models_no_fit.py` (on/off pair each) |
| legacy `Registry.save` (the `registry.json` write) and `recalibrate.build_pairs` (pairs-cache write) | added: `test_both_guards_really_rig_the_fit_and_write_paths` |
| v2 fits rigged: `native_payoff.fit_payoff_line`/`fit_runup_payoff_surface` (v2 guard); training job/estimators (legacy switch); recalibration builder (both) | `test_v2_scoring_native_payoff.py::test_under_v2_guard_inline_path_raises_and_artifact_path_succeeds`, `test_v2_models_training_recipes.py::test_training_job_trips_the_scoring_no_fit_guard_before_touching_disk`, added `test_each_guard_alone_rigs_the_recalibration_builder` |
| cold == warm | previously only same-process (`test_v2_models_inference.py`, `test_v2_models_no_fit.py::test_frozen_inference_joblib_cold_warm_and_missing_under_guard`); added `test_cold_process_and_warm_same_process_requests_agree` (fresh interpreter, cleared caches and warm caches give identical inference, payoff, recalibration and canonical score record; no file written) plus a changed-artifact negative control |
| missing artifact -> MODEL_NOT_READY, never trains, per v2 scoring consumer, both guards on | added: `FrozenInference`, `FrozenStageExecutor`, `score_frozen`, and the model stage's payoff line / payoff surface / recalibration map |

Audit result: no v2 scoring path fits or writes a cache once both guards are
on. The v2 guard does not rig legacy paths, and the legacy guard does not rig
`native_payoff`: only the pair covers everything. `score_frozen` now puts
`MODEL_NOT_READY` on the record ahead of the inference detail code (for
example `ARTIFACT_INVALID`). Before this change a record with a missing
frozen model carried only the detail code.

**Recalibration-map artifact.** `engine/v2/models/recalibration_artifact.py`
(layer 3) and `engine/v2/models/training/recalibration.py` (layer 6) copy the
payoff-artifact pattern. The builder re-derives legacy `fit_recalibration`
statement for statement. It is proven bit-identical: thresholds, `n`,
`base_rate` and `transform` match on pairs with mixed strategies and alphas,
post-cutoff rows, NaNs and ties. The P5-3 `recalibration_map` recipes are
now `fit_owner` training job and write `recalibration_artifact.json` per
(cutoff, alpha). Scoring reads it only when a bundle declares it
(`SourceBundle.recalibration_artifact` / `recalibration_declared`). It then
checks the full `(strategy, alpha, cutoff)` key against the request's fill
and the payoff recipe's `before`, and gives MODEL_NOT_READY on a missing or
mismatched map. Undeclared bundles, which include every Phase 4 capture, are
unchanged.

Not faithful, recorded:

- Below `min_pairs`, legacy returns `None` and ships the raw win. The
  artifact freezes that as `fitted=False`, and the job status is
  `passthrough`, not `skipped`. This is so a missing fold can refuse.
- Fold membership uses the job's `isfinite` completeness mask. Legacy uses
  `dropna`. They differ only on a ±inf `raw_win`/`outcome`.
- A declared map on STR-RUNUP refuses with `UNSUPPORTED_RECALIBRATION`.
  Legacy never recalibrates STR-RUNUP.
- Before this work the native STR-THRU path never applied recalibration.
  Legacy applies it whenever `recalibration_pairs.parquet` supports a map.
  Undeclared native records keep that pre-existing difference.
- `tools/phase5_training_job.py` still builds no calibration datasets and
  passes no cutoffs or alpha. Real recalibration folds need a pairs-dataset
  builder there.

## Real-data builders for the calibration and residual states

Delivered 2026-09-19: `tools/phase5_datasets.py` (legacy readers, read-only,
kept under `tools/` so no new v2 -> legacy adapter edge is needed),
`tools/phase5_training_job.py` (`--alpha`/`--cutoff`/`--pairs` for the
calibration recipes, `--state` for the frozen residual pools) and
`residuals.freeze_stored_driver_residual_pool`. Tests:
`tests/test_v2_models_phase5_datasets.py` (synthetic, `tmp_path` only).

| member | dataset | legacy selection mirrored |
|---|---|---|
| `payoff_line:*`, `payoff_surface:STR-RUNUP` | `payoff_trades()` | `Scorer.trades` as `Scorer.payoff`/`runup_payoff` feed `fit_payoff`/`fit_runup_payoff`: legs parsed a partition at a time, `provenance == "engine.replay"`, full-panel left merge of `abs_move`/`or_implied`, `im_t1 = or_implied`; legacy row order |
| `recalibration_map:*` | `recalibration_pairs()` | `recalibrate.load_pairs()`, the cached table `Scorer.recalibration` fits on |
| `paired_residual_pool` | `paired_pool_inputs()` | `Scorer._residual_pool`: Tier-4 forecasts x panel `abs_move` x `crush_frame()` over the full universe (computed per ticker chunk; exact) |
| `driver_residual_pool:{size,implied_t1,runup_move}` | `champion_driver_pool()` | the champion `ModelArtifact`'s stored `residuals`/`residual_buckets`, which `residual_draws` serves in the model stage (fold key `None`) |

Proved on a synthetic legacy `Scorer`: fold members equal legacy's kept rows
in order, `fit_payoff`/`fit_recalibration` agree with the artifacts, the paired
rows equal the full-universe `_residual_pool`, and the driver artifact serves
`ModelArtifact.residual_pool(prediction)` exactly. Job output is byte-identical
to calling the builder directly.

Differences from legacy, recorded:

- **Paired pool universe.** A bounded Scorer (the nightly's) scopes the crush
  table to its loaded tickers, so its pool depends on the board. The builder
  uses legacy's unbounded branch only; the test shows a scoped legacy pool is
  a strict subset. `--cutoff` (optional) bounds event dates; legacy has none.
  Mixed Tier-4 producer ids on pooled rows are keyed as their `+` join.
- **Driver pools** are the champions' embedded pools, not Tier-4 fold pools
  (`_pool_before`): the latter serve the forecast band and travel in the
  `tier4_folds:*` members. The stored buckets are wrapped unchanged; their
  training predictions are not saved, so they cannot be re-bucketed.
- **Recalibration pairs** are read, never rebuilt: `build_pairs` re-scores
  events with a full Scorer and writes into `data/`, which scoring never does.
- Payoff and recalibration artifacts are keyed per `(strategy, alpha,
  cutoff)`; legacy's cutoff is each request's `evidence_cutoff`, which for
  STR-RUNUP is the entry date, so a board needs one fold per distinct cutoff.

Preparer hook (P5-6): collect `recalibration_artifact.json` from
`--training-root` as it does `payoff_artifact.json`; the state JSONs
(`<out>/<member with : as __>.json`, not the `.summary.json` beside them) go
through `--frozen-state` unchanged.
