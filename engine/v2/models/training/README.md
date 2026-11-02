# `engine/v2/models/training`

Component contract: [ARCHITECTURE.md](ARCHITECTURE.md).

## Standalone neural magnitude primitive (opt-in foundation)

`magnitude_mlp_recipe.mlp_recipe` / `validate_mlp_recipe` bind the new closed
`magnitude_mlp_recipe.v1`: three profiles, ReLU `(32,)`/L2 0.01 and
`(64,32)`/L2 0.01 or 0.1. This small capacity/regularization comparison uses the
installed sklearn `MLPRegressor`, not a GPU framework or combinatorial sweep.
`training.magnitude_mlp.fit_mlp_predictions` receives caller-admitted matrices;
both no-fit guards precede inspection. Fresh sklearn median imputation (retaining
empty columns), all-column missing indicators, input scaling and target scaling
learn exclusively from supplied training rows. The union has fixed doubled width;
sklearn `VarianceThreshold(0)` then drops training-constant values/indicators before
scaling, preventing untrained neural responses when those inputs later change.
Prediction rows cannot alter the learned input width, which the fit evidence records.
An entirely constant union raises `ValueError` before neural fitting.
The affine target standardization improves
Adam's numerical conditioning without changing the squared-error estimand;
`TransformedTargetRegressor` restores original target units before finite checks
and zero projection. This differs intentionally from the untouched production blend.

All profiles fix seed 20260829, one numerical thread, Adam and training-loss
stopping (tolerance 0.0001, patience 15), with no internal random validation split.
The declared 400-epoch estimator ceiling matches the incumbent's finite iteration
policy, not a new wall-time allowance. No holdout or evaluation labels enter fit,
preprocessing or stopping; the caller still owns causal folds, source admission,
inner-only selection, resource budgets and progress reporting. One successful call
is one component fit and returns detached `magnitude_mlp_fit.v1` predictions,
recipe hash, iteration count and standardized-target optimizer loss (including L2,
not a selection score). A convergence warning,
nonfinite fitted weights/loss or nonfinite inverse-transformed prediction raises
`ValueError`, with no partial result, fallback or retry; unrelated warning classes
are not convergence failures. Sklearn's exact interrupted-training warning restores
its original `KeyboardInterrupt`, even under ignored warnings; no success result returns.
Invalid recipe/order/matrices/label domain refuse
before fitting. Repeated inputs in the same numerical environment fit afresh
deterministically; there is no cache, I/O, publication or journal replay here.

The generic magnitude dispatcher now accepts this closed recipe and preserves the
complete result; `size_fit_diagnostics` validates its per-horizon evidence for fresh
execution and replay. Existing recipe values and the comparator remain unchanged.
No existing search request admits MLP: later admission must be explicitly versioned.
New execution sources need fresh source seals; old reservations remain bound to
original installed sources, with no resealing or mutation of running plans.

## Ownership

Implements the **Model training — dataset and model recipes, folds, fitting, evidence, release candidates** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**6** of §4.1.

Replaces (§4.4): `models/training/, rewritten above scoring rather than moved`.

## Responsibilities

- Dataset and model recipes, folds, fitting and residual construction.
- Evidence and release candidates; atomic promotion.

## Non-responsibilities

- **Run inside a score request** — `engine/v2/models` does it instead.
- **Be imported by a feature or a scorer** — `engine/v2/models` does it instead.

## Public interface

`magnitude_mlp.fit_mlp_predictions` fits one guarded standalone MLP component on
caller-admitted matrices: training-only input/target scaling, deterministic fixed
profiles, original-unit nonnegative forecasts and convergence evidence. Invalid
inputs, convergence failure or nonfinite evidence raise `ValueError`; no retry,
source admission, persistence or production/search activation occurs.

`magnitude_candidate.fit_candidate` is the shared fixed-nonnegative Ridge/HGB/ElasticNet/MLP
numeric dispatcher; `experiments.magnitude_candidate_fit` retains a guarded lazy compatibility wrapper.
`fit_routes.fit_execution` accepts only `magnitude`, `magnitude_candidate` or
`regression_candidate` numeric routes. Both guards precede inspection; original
magnitude predictions gain a predictions-only dictionary, while candidate results
and all errors propagate unchanged. No retry, selection, source reads or writes.
Original magnitude requires its own schema and ordered `columns` matching recipe
features before fitting; caller retains its full recipe admission.
Route names are not persisted schemas; offline `research.horizon_fitted` and synthetic tests
consume this API, never production entrypoints. Legacy error translation stays at experiment boundaries.
`fit_routes.check_fitting_allowed` exposes both existing switches for pre-inspection orchestration guards.

`regression_candidate.fit_regression_candidate` dispatches the new domain-aware
Ridge/HGB or ElasticNet recipe to its existing guarded fitter. Both no-fit guards
precede recipe/array inspection; delegated validation, results and errors remain
unchanged. No source reads, retries, selection or persistence occur. Existing
experiment magnitude dispatch remains fixed to nonnegative recipe schemas (Ridge v1, HGB v1 and v2, ElasticNet v1 and MLP v1).

`magnitude.fit_magnitude_predictions` accepts an ops-admitted closed Ridge/HGB
recipe as data and eleven-column numeric matrices
(NaN features permitted), nonnegative finite labels and returns detached finite
nonnegative predictions. It fits with the shared no-fit guard and one numerical
thread; it never reads data, drops rows, writes models or publishes evidence.
The native experiment reporter validates the recipe before calling this API;
registration does not import training. Old training recipes are unchanged.
The fitter restores HGB's integral JSON `max_features` fraction to float for sklearn.

`magnitude_elasticnet.fit_elasticnet_predictions(recipe, train_features, train_labels,
prediction_features, *, columns)` is an add-only research adapter for explicit
selected-column matrices. It returns predictions, recipe ID, the full coefficient
vector, selected source columns and convergence evidence after one guarded training-only fit. `columns` must match
the recipe order. The caller owns causal membership, source identity, fit caps and
inner-only choice of subset/alpha/l1_ratio; this API cannot certify those from arrays.
The experiment dispatcher is a numeric caller, not a grid or production activation.
No validation-target selection, cache or persistence is added.
Nonconvergence raises `ValueError` with no retry or partial result.

`magnitude_elasticnet.fit_regression_elasticnet_predictions` accepts the distinct
domain-aware ElasticNet policy through the same guarded, training-only fitter.
Signed labels/forecasts are finite without a floor or transform; nonnegative
labels still refuse negatives and forecasts clip at zero. The result uses
`regression_elasticnet_fit.v1`, with bound recipe hash and unchanged coefficient
and convergence evidence. No source admission, search or release is inferred.

`magnitude_ridge_hgb.fit_ridge_hgb_predictions` separately accepts a closed
variable-width recipe, caller-admitted numeric matrices and exact ordered
`columns`. Both no-fit guards run before inspection; training-only preprocessing
and fresh one-thread estimators return a recipe hash and detached nonnegative
predictions. Invalid policy/order/values refuse; no source admission, selection,
I/O, cache or retry occurs. Existing eleven-column behavior is unchanged.

`magnitude_ridge_hgb.fit_regression_predictions` consumes the distinct
domain-aware regression recipe with the same guarded, training-only machinery.
Signed labels and forecasts remain finite and untransformed, with no lower bound;
nonnegative labels still reject negatives and predictions clip at zero. Results
use `regression_ridge_hgb_fit.v1` and bind the validated recipe hash. This is a
numeric primitive, not IV target preparation, search, admission or release.

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

P5-4 (`payoff.py`) adds the payoff-calibration artifact builders:
`build_payoff_line_artifact`, `build_payoff_surface_artifact`. Each fits via
`engine.v2.scoring.native_payoff`'s unchanged math (layer 5, strictly below
this package's layer 6) and wraps the result with `engine.v2.models`'s
(layer 3) `make_payoff_line_artifact`/`make_payoff_surface_artifact`, so the
returned artifact is bit-identical to the corresponding inline fit on the
same rows and cutoff.

P5-4 (`recalibration.py`) adds the recalibration-map builder:
`build_recalibration_map_artifact` and `fit_recalibration_map`. The fit is
legacy `engine.recalibrate.fit_recalibration` re-derived statement for
statement (v2 may not import legacy numerics) and tested bit-identical to
it; the result is wrapped with `engine.v2.models`'s
`make_recalibration_map_artifact`. It checks both no-fit switches.

P5-3 (current dataset/training recipes):
`current_recipes(registry_path=None)` bounds installed or alternate registry JSON to 16 MiB before decode via `read_bounded_state`; oversized/nonregular files raise its `ValueError` subclass, with no cache, retry or writes.

- `current_recipes()` — every current recipe as a `TrainingRecipe`, keyed by
  `RecipeKey(role, strategy, output)`: the P5-1 inventory's champion
  bindings (`output="champion"`: expanding-year walk-forward plus full
  refit), the four Tier-4 producers (`output="tier4_monthly"`) and the
  live-refit calibration surfaces (`output="calibration"`: the payoff line
  and surface are fitted per cutoff through `payoff.py`'s P5-4 builders into
  `PayoffLineArtifact`/`PayoffSurfaceArtifact`, and the recalibration maps
  through `recalibration.py` into `RecalibrationMapArtifact`). `recipe_fingerprint` is its identity.
- `prepare_dataset` / `plan_folds` / `dataset_fingerprint` — legacy's
  membership, masks and folds, reproduced.
- `fold_receipts` / `receipt_issues` — the per-fold training-membership and
  label-availability receipts and the check that refuses future members,
  future labels and upstream in-sample leakage.
- `run_training_job` — fits the current release recipes; resumable per fold,
  guarded by the shared no-fit switch.

`build_d14_dataset` builds the two monthly D14 datasets from explicit catalog,
artifact root, snapshot and receipt identifiers. It verifies receipt association
and calendar bytes before calculating any frame. Earnings and daily-state
calculations use native readers/math; the historical panel producer remains a
dependency through its snapshot-pinned history columns. It does not substitute
the live panel-row population. The CLI,
`python -m tools.phase5_d14_inputs --role implied_t1 --out OUT`, requires
`--catalog`, `--artifact-root`, `--snapshot-id`, and `--receipt-id`; optional
`--plan-only` writes receipts without fitting. Supervised `ops plan training`
can select the same builder with `--training-input-mode native_d14` and explicit
`--training-catalog`, `--training-artifact-root`, `--training-snapshot-id`,
`--training-receipt-id` pins, then `ops submit` uses the existing worker lifecycle.
Only the two `*:tier4_monthly` recipes are admitted; no legacy manifest or pairs
override may be mixed in. Both entry points record identical `training_inputs`.
The receipt must belong to that snapshot and pin its original
calendar. `run_training_job(input_identity=...)` records those source pins in
the resume identity, alongside the unchanged dataset and recipe fingerprints.

`d14_dataset.calculate_frame(repository, snapshot, *, calendar, role)` is the
calculation helper for caller-admitted inputs. It checks snapshot identity but
neither certifies calendar provenance nor attaches `training_inputs`; use
`build_d14_dataset` for verified training inputs. The same module exposes
`input_rows`/`input_frame` (bounded reads), `decision_rows` (session decisions),
`daily_features` (in-place daily-state attachment) and `validate_role` (early
role refusal). These are public module members, not package-root re-exports;
they do not imply receipt or source admission.

P5-4 (`residuals.py`) adds the frozen residual-pool builders:
`build_driver_residual_pool_artifact` (bucketing via `native_payoff`
unchanged) and `build_paired_residual_pool_artifact` (legacy
`Scorer._residual_pool`'s join and error arithmetic over an explicit universe,
with an explicit causal cutoff and no scorer context). Both open with the v2
no-fit guard and require a declared `Lineage`.
`freeze_stored_driver_residual_pool` wraps a full-refit champion's stored
flat pool and training-time buckets unchanged (the predictions behind them
are not saved, so they cannot be re-bucketed). The real-data dataset
builders for these and for the calibration recipes live in
`tools/phase5_datasets.py` (legacy readers stay outside `engine/v2`).

P5-4 (`chooser_pool.py`) adds `build_chooser_analog_pool_artifact`, freezing the caller's stored-order rows up to an exclusive causal cutoff with a declared `Lineage`.

`chooser_pool_source.py` holds the runtime admission of the frozen chooser
pool's parquet source — `CHOOSER_POOL_ID`, `CHOOSER_POOL_MAX_BYTES`,
`require_chooser_pool_columns` and `read_chooser_pool_frame` — imported by
both `tools/phase5_prepare_release.py` (which re-exports the names) and
`tools/phase5_training_job.py`; it imports no `checks` module so the training
job's worker path can read it.

<!-- public-interface: fit_mlp_predictions, check_fitting_allowed, fit_candidate, fit_execution, fit_regression_candidate, fit_regression_predictions, fit_ridge_hgb_predictions, fit_elasticnet_predictions, build_d14_dataset, calculate_frame, input_rows, input_frame, decision_rows, daily_features, validate_role, fit_magnitude_predictions, build_payoff_line_artifact, build_payoff_surface_artifact, build_recalibration_map_artifact, fit_recalibration_map, CLOCK_ID, EqualWeightBlend, EstimatorSpec, FoldOutcome, FoldPlan, FoldScheme, LABEL_RECEIPT_V1, LEGACY_SEED, LabelRule, LogTargetModel, MEMBERSHIP_RECEIPT_V1, OWNER_P5_4, OWNER_TRAINING_JOB, PreparedDataset, RECIPE_V1, ReceiptIssue, RecipeDataError, RecipeKey, ResidualRule, RowFilter, SeedMeanEnsemble, TRAINING_JOB_V1, TargetSpec, ThresholdRule, TrainingJobResult, TrainingRecipe, TrainingRefused, UnsupportedEstimator, UpstreamDependency, ValueMask, current_recipes, dataset_fingerprint, fit_recipe_estimator, fold_receipts, plan_folds, prepare_dataset, receipt_issues, recipe_fingerprint, run_training_job, build_driver_residual_pool_artifact, build_paired_residual_pool_artifact, freeze_stored_driver_residual_pool, CHOOSER_POOL_ID, CHOOSER_POOL_MAX_BYTES, require_chooser_pool_columns, read_chooser_pool_frame, fit_regression_elasticnet_predictions -->

## Consumers

`engine.v2.research` uses closed numeric routes and pre-inspection guards for offline horizon fits.

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

`engine.v2.ops` imports `build_d14_dataset` to construct explicitly pinned native
D14 datasets for supervised recipe jobs, and `TrainingRefused` to map refused
training receipts to typed `OpsError`s. Only these two v2 packages import this one,
and none below layer 6 may. Standalone real-data entry points are
`tools/phase5_training_job.py` and `tools/phase5_d14_inputs.py`, outside v2 packages.

`experiments/magnitude_fitted.py` reaches numeric fitters through shared research orchestration after native admission.
`experiments/magnitude_candidate_fit.py` lazily delegates to the shared candidate dispatcher;
its offline callers grant no source admission.

<!-- consumers: engine.v2.ops, engine.v2.research -->

## Usage

    from engine.v2.models.training.payoff import build_payoff_line_artifact

    artifact = build_payoff_line_artifact(
        rows, strategy="STR-THRU", driver="abs_move", alpha=0.5,
        before="2026-09-16",
    )
    # artifact is None when fewer than min_trades rows survive the causal
    # (exit_date < before) filter -- the same NO_PAYOFF_MAP condition the
    # inline fit refuses on today.

```python
from engine.v2.models.training import current_recipes, RecipeKey, run_training_job

recipe = current_recipes()[RecipeKey("size", "*", "champion")]
result = run_training_job(recipe, dataset, out_dir, plan_only=True)  # receipts only
result = run_training_job(recipe, dataset, other_out_dir)            # fit; rerun resumes
```

`dataset` must carry the recipe's keys, features, target, membership time,
year column, the label-availability column (`recipe.label.time_column`) and,
for `tier4_monthly_oos` upstream dependencies, the Tier-4
`<produces>_fold_start` / `<produces>_model_id` lineage columns.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
