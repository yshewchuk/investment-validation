# Forecast-only distribution evaluation

## Purpose and boundaries

This component evaluates admitted absolute-magnitude forecasts without economic decisions. It reuses the existing magnitude representation/proper-score mathematics, native distribution fits and earlier-OOF controls. No option quotes, payoff-bridge application, EV, trade selector or funded book belongs to this evaluation path. The existing strategy implementation remains a separate consumer of compatible forecast primitives.

Native registration, dataset/feature qualification, training-label availability, event grouping, random/rolling holdouts and immutable fit receipts retain their existing owners. Source-free numerical helpers cannot authenticate arbitrary caller data, open labels or grant execution authority. A later empirical-baseline adapter must consume already-verified native training evidence, not an unauthenticated list that merely claims to be causal.

| Condition | Contract |
| --- | --- |
| `magnitude_distribution_math.mean_matched_control(target_mean, residuals)` | Same nonnegative empirical draws anchored to a positive mean, zero atoms, achieved-mean metadata and refusal Problems as the existing control; no payoff bridge or trade inputs. |
| Existing `magnitude_distribution_payoff.mean_matched_control` call | Public re-export of that neutral implementation; existing strategy results and failures remain compatible. `distribution_oof` consumes the neutral import directly. |
| `prepare_empirical_crps(samples)` | Validate/copy/sort a nonempty empirical law once and return an observation scorer over immutable parts. Preserve duplicate/zero atoms; use the existing analytic CRPS core, no pairwise matrix or new formula. Preparation is O(n log n), each observation O(n), retained storage O(n). |
| Invalid sample or observation | Existing typed `CONTRACT_MISMATCH`, not silent clipping, smoothing or a manufactured distribution. Caller mutation after preparation cannot alter the retained law. |
| `distribution_prediction_scores.arm_forecasts` / `score_forecasts` | Source-free forecast controls and CRPS/RPS/log-loss/tail rows; verify supplied receipt/pool identity and outer horizon before reading targets. Preserve missing targets, unavailable controls and unresolved-tail CRPS. Caller owns admission, complete event axes and target-access order. |
| Legacy `distribution_scores` forecast exports | Re-export the identical neutral functions; economic/profit scoring stays separate. No numerical, refusal, row-order or progress-cadence change. |
| Source-free numerical result | No I/O, fitting, native admission, publication or final-read authority. |

## Incremental delivery

This first slice removes numerical strategy dependencies; it does not yet bind a training baseline or supply a standalone forecast-only lifecycle. Subsequent slices will bind the baseline to the matching admitted training cohort and disjoint evaluation roster, publish complete all-horizon forecasts before admitted evaluation-target reads, aggregate proper-score/baseline comparisons and calibration/interval diagnostics, and expose a dedicated contained operator. They must retain every admitted event with explicit missingness, immutable source/request identity, native byte/resource bounds, consumed attempts and verified no-fit replay. They must not inherit the strategy request's quote/bridge/economics/book prerequisites.

The report will distinguish data availability from predictive interval coverage, and calibration diagnostics from fitted calibration. A frozen fit's paired uncertainty is not a refitted model-search experiment. Development results do not authorize protected holdout opening, empirical accuracy claims, production activation or a downstream strategy test.

## Prediction metadata lifecycle

`distribution_prediction_plan.PredictionPlan` validates and detaches a closed prediction-only request containing parent/whole-job limits, maximum admitted event count and explicit uncertainty settings. Preflight reuses native plan/source verification, requires all three horizons and unique outer event axes, and binds request, fit budget and installed source closure into a deterministic plan. Seal persists that identity; admission rejects any later policy, resource, source or native-plan drift. It adds no quote, bridge, book or economic policy.

Prediction journal slots are create-only in a separate namespace, including identical retries. Bounded CAS sections preserve ordered positions and reverify bytes on reading. These primitives do not execute, open targets, authenticate arbitrary section contents or provide completed-run collection; the subsequent runner must publish all forecasts before targets, retain complete native receipts, consume failures and reverify stored evidence during collection. Limits are declarations until a dedicated contained process enforces them; the metadata interface is not an operator.

## Reporting supplied forecast evidence

This source-free library API is currently exercised only by synthetic tests. The intended future consumer is the separate neutral prediction lifecycle; this slice has no wired production or existing strategy caller, runnable prediction-only CLI, or execution authority.

`distribution_prediction_report.prediction_report(scores, events)` consumes one pass of supplied score rows and the frozen event list (`event_id`, `ticker`, canonical `event_date`); it has no strategy, book, market or payoff dependency. Native callers still authenticate source/request/fit identity and bound input/output bytes. The helper checks unique events/row identities, consistent outer-year membership and observations across arms, and finite score/tail contracts. Missing rows remain explicit counts against the entire frozen population for every arm and horizon, including wholly absent horizons.

The shared row/event admission requires every retained payload, including nested provenance, to serialize with standard-library `json.dumps(..., allow_nan=False)` before copying; invalid keys/values, nonfinite numbers and recursive payloads refuse with `INVALID_EXPERIMENT_SPEC`. Present score/tail fields are validated for every status; only available rows require all fields. Missing-target tails retain boolean observations without comparing them to an absent target. Unknown provenance remains opaque and is preserved, not stripped or semantically authenticated; serialization adds a linear scan of retained payloads.

All four arms report scoreable-only CRPS/RPS/common-bin log-loss means and their sample counts; infinite log loss is never dropped from the overall mean. Fixed tail summaries pair predicted probability with realized frequency and count. Paired control-minus-histogram CRPS is positive for histogram improvement; only horizon-one same-mean is primary. Full-population deltas remain unavailable when any pair is missing; scoreable-only deltas retain exact paired and excluded event IDs. Detached, canonically ordered input rows and event metadata preserve supplied provenance and missingness for downstream diagnostics without granting provenance authority.

Malformed input/arithmetic refuses `INVALID_EXPERIMENT_SPEC` without mutation or partial results. The source-free helper has no cache, retry, writes, refitting or holdout reads. Index/sort plus fixed-arm/horizon summaries use O(rows log rows + events log events) time and O(rows + events) memory. Interval/PIT aggregation and sealed conditional uncertainty are separate reporting slices; this interface does not randomize PIT, fit calibration or choose promotion thresholds.
