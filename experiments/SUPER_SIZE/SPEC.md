# SUPER SIZE: fixed stored-history candidate

This native development candidate predicts the conditional mean of absolute
earnings close-close return, in percentage points, using split-neutral,
non-dividend-adjusted prices. Forecasts use 5, 3 and 1 full trading sessions.

## Preregistered policy

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
native missingness, 100 iterations, learning rate 0.1, seven leaves, minimum
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
