# Forecast-only distribution evaluation

## Purpose and boundaries

This component evaluates admitted absolute-magnitude forecasts without economic decisions. It reuses the existing magnitude representation/proper-score mathematics, native distribution fits and earlier-OOF controls. No option quotes, payoff-bridge application, EV, trade selector or funded book belongs to this evaluation path. The existing strategy implementation remains a separate consumer of compatible forecast primitives.

Native registration, dataset/feature qualification, training-label availability, event grouping, random/rolling holdouts and immutable fit receipts retain their existing owners. Source-free numerical helpers cannot authenticate arbitrary caller data, open labels or grant execution authority. The empirical-baseline adapter consumes freshly verified native training evidence and independently matches the completed fit claim, ordered matrices and disjoint event axes; an arbitrary list claiming causality is not admitted.

| Condition | Contract |
| --- | --- |
| `magnitude_distribution_math.mean_matched_control(target_mean, residuals)` | Same nonnegative empirical draws anchored to a positive mean, zero atoms, achieved-mean metadata and refusal Problems as the existing control; no payoff bridge or trade inputs. |
| Existing `magnitude_distribution_payoff.mean_matched_control` call | Public re-export of that neutral implementation; existing strategy results and failures remain compatible. `distribution_oof` consumes the neutral import directly. |
| `prepare_empirical_crps(samples)` | Validate/copy/sort a nonempty empirical law once and return an observation scorer over immutable parts. Preserve duplicate/zero atoms; use the existing analytic CRPS core, no pairwise matrix or new formula. Preparation is O(n log n), each observation O(n), retained storage O(n). |
| Invalid sample or observation | Existing typed `CONTRACT_MISMATCH`, not silent clipping, smoothing or a manufactured distribution. Caller mutation after preparation cannot alter the retained law. |
| `distribution_prediction_scores.arm_forecasts` / `score_forecasts` | Source-free forecast controls and CRPS/RPS/log-loss/tail rows; verify supplied receipt/pool identity and outer horizon before reading targets. Preserve missing targets, unavailable controls and unresolved-tail CRPS. Caller owns admission, complete event axes and target-access order. |
| Legacy `distribution_scores` forecast exports | Re-export the identical neutral functions; economic/profit scoring stays separate. No numerical, refusal, row-order or progress-cadence change. |
| `training_baseline(fits, outer_year, horizon)` | Reverify admitted stored training and completed fit; bind plan/admission/trial/matrix/event identity and label cutoff, then copy the identical equal-weight training label rows. No evaluation outcome read, fitting or publication; caller must persist this evidence before target access. |
| `score_training_baseline(baseline, targets, receipt=...)` | Require unchanged baseline hash, matching completed fit claim and exact evaluation event/status roster; snapshot the validated event/claim/target values before yielding, so between-row caller mutation cannot change consumed identity or outcomes. Prepare CRPS/bin/tail reference once per cohort; retain missing targets and impossible-bin infinite log loss. The saved hash detects mutation, not forged admission; collection must reverify native provenance. |
| Malformed saved baseline/receipt, changed identity or incomplete/invalid target-status map | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT` before score preparation or row emission; omissions must have exactly one `omitted` key with a native reason. Invalid empirical numeric labels retain the numerical helper’s typed `CONTRACT_MISMATCH`; no source read or partial result. |
| Source-free numerical result | No I/O, fitting, native admission, publication or final-read authority. |

## Incremental delivery

Delivered helpers remove numerical strategy dependencies and bind the training baseline; they do not yet supply a standalone forecast-only lifecycle. Subsequent slices will publish complete all-horizon forecasts before admitted evaluation-target reads, aggregate proper-score/baseline comparisons and calibration/interval diagnostics, and expose a dedicated contained operator. They must retain every admitted event with explicit missingness, immutable source/request identity, native byte/resource bounds, consumed attempts and verified no-fit replay. They must not inherit the strategy request's quote/bridge/economics/book prerequisites.

The report will distinguish data availability from predictive interval coverage, and calibration diagnostics from fitted calibration. A frozen fit's paired uncertainty is not a refitted model-search experiment. Development results do not authorize protected holdout opening, empirical accuracy claims, production activation or a downstream strategy test.
