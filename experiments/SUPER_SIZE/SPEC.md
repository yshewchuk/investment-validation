# SUPER SIZE: fixed stored-history candidates

These native development candidates predict the conditional mean of absolute
earnings close-close return, in percentage points, using split-neutral,
non-dividend-adjusted prices. Forecasts use 5, 3 and 1 full trading sessions.

## History-mean policy (v1.4)

- One fixed model: stored ticker arithmetic mean of prior absolute moves.
- Empty history alone falls back to the strictly qualified training-fold mean.
- Expanding event-year folds and both explicit label/window clocks are sealed.
- Primary loss: pooled event/horizon RMSE; secondary: MAE and per-horizon errors.
- Complete three-horizon coverage gives each event equal total weight.
- No search or winner selection occurs in this candidate. Quantiles remain pending.

## Local execution contract

Use the native `ExperimentSpec` schema with a unique experiment/arm identity,
seed zero, empty economics, `computed_moves`, explicit evaluation years and
runner `experiments/v2_prediction.py`. Register before evaluation.
Build history separately with `python -m experiments.magnitude_history_store build`;
retain its exact feature snapshot ID. Source snapshot, selected events, holdout
month, years and training start must match the experiment. No build occurs on run.

For `python -m experiments.v2_prediction register|run`, supply the spec, existing
catalog/store, scope and holdout month, plus `--target absolute_move`,
`--magnitude-training-start`, `--magnitude-labels-before`, `--magnitude-model history_mean`,
`--magnitude-feature-snapshot-id` and `--magnitude-history-start`. Repeat the same
request for run; use `--no-ledger` for smoke before recorded execution with `--ledger`.

Random and rolling holdouts remain excluded. Native evidence binds consumed
stored rows and original source identities; schedules/publication remain unverified
reconstructions. Synthetic development tests do not establish real coverage,
accuracy, resource requirements or a winning model. The user runs real development locally.

## Fixed fitted candidates

Use a separate unique experiment/arm for `--magnitude-model history_ridge` or
`history_hgb`, retaining the same explicit clocks and stored-feature arguments.
Each recipe consumes all eleven history columns and fits separately per year/horizon
using only qualified earlier labels. Ridge uses train-only median imputation,
all-column missing flags, scaling and alpha 1; HGB uses squared-error loss,
native missingness except training-all-null columns frozen to zero, 100 iterations,
learning rate 0.1, seven leaves, minimum
20 samples per leaf, zero L2, seed zero and no early stopping or warm start.
Both project finite predictions to zero or above without transforming the target.
The preregistration binds complete fixed parameters and preprocessing; no runtime
overrides, automatic selection, tuning or model release occur. Feature-row and
training-label digests accompany complete forecasts and pooled RMSE/MAE.

The implementation uses sklearn Pipeline, FeatureUnion, SimpleImputer,
MissingIndicator, StandardScaler, Ridge and HistGradientBoostingRegressor.
Stored research admission owns membership; generic training folds are not reused
because their missing-row and fold-cutoff policies differ. Runtime never rebuilds
features. These candidates are local development benchmarks, not a selected winner.

## Plan a bounded nested search

The shared public annual-fold helper changes producer and runner source hashes.
Existing artifacts remain intact; new-code execution needs a newly built history
context and fresh experiment/arm registration. Existing source snapshots are reusable.
Search preparation requires the fitted registration, not a completed benchmark run.

After preregistering a fitted candidate, use `python -m experiments.v2_magnitude_search
prepare --no-ledger --spec <anchor-spec> --catalog <existing-catalog> --store-root <existing-store>
--request <request-json>`. The command returns an immutable plan reference, not a run.
Use `verify` with the same request plus `--plan-ref <reference-json>` to check it again.
Both actions accept `--no-ledger` as a no-op: planning never writes a ledger.
JSON syntax uses the standard library. `jsonschema` was considered for shape checks;
domain validation still needs ordered axes, fit caps, cross-field years and native
identity checks, so a second schema dependency would duplicate only part of admission.

The request has schema `magnitude_search_request.v1`, `inner_years` mapping every
outer year to explicit increasing earlier years, `space`, `max_candidates` and
`max_fits`. Space maps `history_ridge` to an `alpha` array and/or `history_hgb` to
arrays for all five knobs: `learning_rate`, `max_iter`, `max_leaf_nodes`,
`min_samples_leaf`, `l2_regularization`. Each axis is increasing and nonempty;
all other recipe settings stay fixed. No search size or compute budget is inferred.

Inner folds stay inside outer training candidates and group each event across all
horizons. Execution must later qualify training labels strictly before
each inner fit and inner evaluation labels as complete strictly before the outer fit.
The frozen policy selects by pooled inner RMSE, breaks exact ties by candidate ID,
and reserves outer evaluation for the selected candidate. Latest-outer selection
defines the prospective frozen winner; no winner exists until recorded execution.
Every planned candidate must be attempted and failures recorded before selection.
This command neither reads outcomes nor runs the search.

The library-only `experiments.magnitude_search_trials.SearchTrials` boundary
records every candidate/inner-fold attempt and freezes selections from submitted
forecasts. Explicitly seal one plan per anchor, start each attempt once, and record
complete three-horizon real forecasts or an explicit `EXPERIMENT_VARIANT_FAILED` fit failure. Numeric predictions and
integral horizons are normalized to JSON float/int; Booleans refuse. Start returns admitted stored
features and strictly qualified training labels; recording commits the forecasts
before opening complete inner validation labels. One immutable shared validation record binds native-derived targets and the full receipt to admitted preflight membership, cutoff and recipe before terminal publication; concurrent candidates must agree. Replay verifies terminal targets against that record without reopening labels. This consistency check does not authenticate wholesale fabrication of all journal evidence. Freeze refuses incomplete journals
and excludes a candidate's entire score for an admitted explicit fit failure or local scoring failure.
Every shared native-input/label admission failure leaves the attempt unfinished, regardless of its error code;
infrastructure/resource/shared-identity failures also block selection. Exact replay of a
terminal submission or freeze is idempotent; another start is a forbidden retry.
Journal writes above 16 MiB refuse nonretryable `RESOURCE_LIMIT_EXCEEDED` before publishing that record.
The submitted forecasts do not prove that an estimator ran. Frozen selections grant
no outer-evaluation, deployment or final-read authority. A resource-capped executor,
interrupted-attempt reconciliation, outer evaluation and reporting remain unimplemented.
