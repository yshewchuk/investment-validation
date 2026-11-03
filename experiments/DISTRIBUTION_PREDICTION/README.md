# Distribution prediction evaluation

The first experiment asks whether conditional **absolute earnings-move magnitude** distributions predict held-out development outcomes well. It does not evaluate STR-THRU, signed-return direction, EV, trade selection or portfolio profit. The approved [design amendment](../STR_THRU_DISTRIBUTION/DESIGN.md#current-first-experiment-prediction-only) keeps those questions separate.

## Delivered scope and remaining wiring

The supplied neutral numerical interfaces are `magnitude_distribution_math.mean_matched_control`, `prepare_empirical_crps` and `distribution_prediction_scores.arm_forecasts` / `score_forecasts`. The scorer produces the existing three-arm proper-score/status rows from supplied admitted forecast/pool evidence and outcomes without importing economic policy. Earlier-OOF controls no longer import the payoff module; the existing payoff export remains compatible. The prepared CRPS scorer copies, validates and sorts one empirical distribution once, then evaluates observations with the existing analytic core; it neither constructs a causal baseline nor fits anything. These are library interfaces, **not yet a standalone prediction-only CLI**. The admitted empirical baseline is now supplied by `distribution_prediction_baseline.training_baseline(fits, outer_year, horizon)` and `score_training_baseline(baseline, targets, receipt=...)`: it uses the identical admitted training labels and matching completed fit, never evaluation labels or residuals. The scorer requires the complete evaluation event/status map and prepares the shared law once. Subsequent slices add prediction reports/diagnostics and a separate contained operator. The existing strategy operator still requires economic inputs and is not this experiment's entrypoint.

The predictive population is the admitted frozen event cohort, not all provider/world events. All admitted outer events must receive score/status rows, including explicit missing targets or unavailable controls. The native 5/3/1 horizon roster and chronological split/holdout rules remain unchanged. No events may disappear because a trade was not selected or quotes were unavailable.

Removing trading-evaluation inputs does not assert that predictor features are option-free: a qualified HMO panel can contain historically option-derived features. Feature availability and the chosen H/HM/HMO recipe still need their existing causal admission; no trading quote or payoff is needed to define a proper prediction score.

## Comparisons and interpretation

- Fixed raw histogram classifier, with its current training-only tail rule and explicit unresolved-tail status.
- Plain empirical distribution of the identical admitted outer-training magnitude labels: a no-feature baseline, never fitted to evaluation outcomes.
- Same-feature point prediction plus strictly earlier inner-OOF residuals.
- Same fitted histogram's mean plus the causal residual envelope: a shape-use control, not proof of equal learner capacity.

CRPS is the primary proper score at horizon1; horizons3/5 are robustness outputs. RPS and common-bin log loss expose bin-probability quality, including infinite loss for an impossible observed bin. Fixed-threshold tail probabilities and realized frequencies need sample counts and uncertainty. Predictive interval coverage must be paired with interval width; PIT/reliability diagnostics are planned reporting, not a claim that raw probabilities have been calibrated. Preserve the existing same-mean primary shape contrast and report the additional empirical/point comparisons explicitly, without tuning on outer results.

The current candidate has no parameter search, probability smoothing or fitted calibrator. Do not mistake an earlier-OOF residual control for calibrated classifier probabilities. No empirical performance, resource feasibility, promotion or real-run authority follows from synthetic tests or these interfaces.

See [architecture](ARCHITECTURE.md) for the boundary between supplied numeric evidence and native causal admission. The strategy integration remains in [STR_THRU_DISTRIBUTION](../STR_THRU_DISTRIBUTION/README.md), unchanged and separate.

## Frozen forecast evidence

The intentionally unwired `distribution_prediction_evaluation.evaluate_outer` / `read_evaluated_outer` library admits outer targets only after frozen evidence verification, and replays stored targets/scores without new outcome reads. Default metadata seals bind the evaluation implementation and its scorer/diagnostic dependencies as well as the freeze/baseline implementation. Source drift refuses before effects; a fresh final operator/source registration is still required for execution.

The intentionally unwired `distribution_prediction_evidence.freeze_outer` / `read_frozen_outer` library verifies all three admitted horizons before publishing and rechecks native provenance on replay. Shared controls and training references use versioned bounded canonical-text fragments rather than a single cohort-sized row; the shard caps are unchanged. Default metadata seals include this verifier and baseline implementation. Old single-row frozen evidence is refused, not silently migrated; an interrupted freeze remains consumed and cannot be retried. This is a synthetic-tested prerequisite, not a runnable prediction operator or empirical validation.

## Supplied-score reporting

This is a source-free library API currently called only by synthetic tests. Its intended future consumer is the separate neutral prediction lifecycle; no production or existing strategy caller is wired in this slice, and it provides no runnable prediction-only CLI or execution authority.

`distribution_prediction_report.prediction_report(scores, events)` summarizes supplied four-arm forecast rows against the complete frozen event metadata, with no trade decision date or economic inputs. Every horizon and arm retains available, unavailable, missing-target and missing-row counts. Means are scoreable-only with metric-specific counts; unresolved CRPS and infinite log loss remain explicit. Tail checks show predicted probability, observed frequency and sample count. Paired same-mean, matched-point and training-empirical CRPS improvements each use that contrast's paired scoreable subset and retain excluded event IDs; the full-population improvement is unavailable if any pair is missing. Positive improvement favors the histogram, and horizon-one same-mean stays the sole primary contrast. Raw supplied rows/metadata are retained, not authenticated by this helper; native admission and later operator wiring remain required.

All retained rows and event metadata must be finite-JSON serializable, including opaque nested provenance. Any present score/tail fields are validated even on unavailable or missing-target rows; those rows may omit the fields, and a missing target does not invent tail truth. Invalid retained payloads refuse rather than being silently stripped. Valid optional provenance and existing score/status semantics are preserved.
