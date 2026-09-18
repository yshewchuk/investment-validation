# `engine/v2/models/training`

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

P5-3 (current dataset/training recipes):

- `current_recipes()` — every current recipe as a `TrainingRecipe`, keyed by
  `RecipeKey(role, strategy, output)`: the P5-1 inventory's champion
  bindings (`output="champion"`: expanding-year walk-forward plus full
  refit), the four Tier-4 producers (`output="tier4_monthly"`) and the
  live-refit calibration surfaces (`output="calibration"`, fitted by P5-4,
  receipts only here). `recipe_fingerprint` is its identity.
- `prepare_dataset` / `plan_folds` / `dataset_fingerprint` — legacy's
  membership, masks and folds, reproduced.
- `fold_receipts` / `receipt_issues` — the per-fold training-membership and
  label-availability receipts and the check that refuses future members,
  future labels and upstream in-sample leakage.
- `run_training_job` — the only entry that fits a recipe; resumable per fold,
  guarded by the shared no-fit switch.

<!-- public-interface: build_payoff_line_artifact, build_payoff_surface_artifact, CLOCK_ID, EqualWeightBlend, EstimatorSpec, FoldOutcome, FoldPlan, FoldScheme, LABEL_RECEIPT_V1, LEGACY_SEED, LabelRule, LogTargetModel, MEMBERSHIP_RECEIPT_V1, OWNER_P5_4, OWNER_TRAINING_JOB, PreparedDataset, RECIPE_V1, ReceiptIssue, RecipeDataError, RecipeKey, ResidualRule, RowFilter, SeedMeanEnsemble, TRAINING_JOB_V1, TargetSpec, ThresholdRule, TrainingJobResult, TrainingRecipe, TrainingRefused, UnsupportedEstimator, UpstreamDependency, ValueMask, current_recipes, dataset_fingerprint, fit_recipe_estimator, fold_receipts, plan_folds, prepare_dataset, receipt_issues, recipe_fingerprint, run_training_job -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

No v2 package imports this one (and none below layer 6 may). The real-data
entry point is `tools/phase5_training_job.py`, which is not a v2 package.

<!-- consumers: none -->

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
