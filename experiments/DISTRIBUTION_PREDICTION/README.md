# Distribution prediction evaluation

The first experiment asks whether conditional **absolute earnings-move magnitude** distributions predict held-out development outcomes well. It does not evaluate STR-THRU, signed-return direction, EV, trade selection or portfolio profit. The approved [design amendment](../STR_THRU_DISTRIBUTION/DESIGN.md#current-first-experiment-prediction-only) keeps those questions separate.

## Delivered scope and remaining wiring

The initial slice supplies neutral numerical prerequisites: `magnitude_distribution_math.mean_matched_control` and `prepare_empirical_crps`. Earlier-OOF controls no longer import the payoff module; the existing payoff export remains compatible. The prepared CRPS scorer copies, validates and sorts one empirical distribution once, then evaluates observations with the existing analytic core; it neither constructs a causal baseline nor fits anything. These are library interfaces, **not yet a standalone prediction-only CLI**. Subsequent slices bind the admitted training empirical baseline and forecast scores, add prediction reports/diagnostics and a separate contained operator. The existing strategy operator still requires economic inputs and is not this experiment's entrypoint.

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
