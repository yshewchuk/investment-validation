# Distribution prediction evaluation

The first experiment asks whether conditional **absolute earnings-move magnitude** distributions predict held-out development outcomes well. It does not evaluate STR-THRU, signed-return direction, EV, trade selection or portfolio profit. The approved [design amendment](../STR_THRU_DISTRIBUTION/DESIGN.md#current-first-experiment-prediction-only) keeps those questions separate.

## Prediction-only development lifecycle

`distribution_prediction_run.PredictionRun` composes native training admission, bounded fitting, all-horizon forecast freezing, admitted outer targets, four-arm scoring, diagnostics and reports. Its separate CLI is `python -m experiments.distribution_prediction_run`; the strategy operator is not this experiment's entrypoint. Neutral `arm_forecasts` / `score_forecasts` preserve the histogram, same-mean and matched-point controls. `training_baseline` reconstructs the identical admitted training labels and matching completed fit, never evaluation labels or residuals. Prepared empirical CRPS copies, validates and sorts one shared law per cohort. Existing payoff/strategy exports remain compatible and separate.

The predictive population is the admitted frozen event cohort, not all provider/world events. All admitted outer events must receive score/status rows, including explicit missing targets or unavailable controls. The native 5/3/1 horizon roster and chronological split/holdout rules remain unchanged. No events may disappear because a trade was not selected or quotes were unavailable.

Removing trading-evaluation inputs does not assert that predictor features are option-free: a qualified HMO panel can contain historically option-derived features. Feature availability and the chosen H/HM/HMO recipe still need their existing causal admission; no trading quote or payoff is needed to define a proper prediction score.

## Candidate classifier variants

`candidate_recipe(feature_set, classifier_variant="raw")` emits the unchanged raw recipe by default. Selecting `classifier_variant="regularised_v1"` in the recipe JSON selects that variant: it adds the variant marker to the document and changes only `l2_regularization` to `1.0` and `early_stopping` to `true`; the point recipe stays identical. The marker is part of the hashed recipe identity, so a plain recipe JSON selects raw and a marked JSON selects the regularised variant; unknown markers are refused.

The regularised fit enables early stopping, so sklearn may stop before `max_iter`. If training support contains any observed class with fewer than two rows (a singleton histogram bin), the fit keeps every row and class but sets the estimator's effective `early_stopping=False` for that fit only; L2 stays `1.0`. Its diagnostics record `early_stopping_requested=True`, `early_stopping_used=False` and `fallback_reason="class_count_below_2"`, and replay validates those fields, the singleton support and `iterations=max_iter`. Without a singleton, replay requires requested/used true, no fallback reason and an actual iteration count in `[1, max_iter]`. Raw diagnostics are unchanged.

## Comparisons and interpretation

- Fixed raw histogram classifier, with its current training-only tail rule and explicit unresolved-tail status.
- Plain empirical distribution of the identical admitted outer-training magnitude labels: a no-feature baseline, never fitted to evaluation outcomes.
- Same-feature point prediction plus strictly earlier inner-OOF residuals.
- Same fitted histogram's mean plus the causal residual envelope: a shape-use control, not proof of equal learner capacity.

CRPS is the primary proper score at horizon1; horizons3/5 are robustness outputs. RPS and common-bin log loss expose bin-probability quality, including infinite loss for an impossible observed bin. Fixed-threshold tail probabilities and realized frequencies include sample counts and conditional cluster uncertainty. Predictive interval coverage is paired with width; unrandomized PIT brackets remain in event evidence, not a claim that raw probabilities have been calibrated. Preserve the existing same-mean primary shape contrast and report the additional empirical/point comparisons explicitly, without tuning on outer results.

The current candidate has no parameter search, probability smoothing or fitted calibrator. Do not mistake an earlier-OOF residual control for calibrated classifier probabilities. No empirical performance, resource feasibility, promotion or real-run authority follows from synthetic tests or these interfaces.

See [architecture](ARCHITECTURE.md) for the boundary between supplied numeric evidence and native causal admission. The strategy integration remains in [STR_THRU_DISTRIBUTION](../STR_THRU_DISTRIBUTION/README.md), unchanged and separate.

## Frozen forecast evidence

`PredictionRun` calls `distribution_prediction_evidence.freeze_outer` before `distribution_prediction_evaluation.evaluate_outer`; evaluation and completed collection use the real frozen-evidence reader before scores or replay. Default metadata seals bind the evidence, training-baseline, evaluation and collection implementations with their transitive scoring, diagnostic and reporting dependencies. Explicit subclass roots must cover their complete implementation; the final runner's imports provide that closure. Source drift refuses before effects, and a fresh final operator/source registration remains required for execution.

Shared controls and training references use `distribution_prediction_frozen.v2` with `canonical_json_fragments.v1` and 65,536-character fragments under unchanged shard caps. Production replay compares stored fragments to rebuilt canonical text without joining or decoding them. Canonicalization still retains O(total reference) text; bounded persistence is not constant-memory execution. Old single-row evidence is refused, not migrated or reused. Non-string freeze/evaluation/replay years and malformed frozen horizon entries refuse before their respective downstream effects; an interrupted attempt remains consumed and cannot be retried.

## Supplied-score reporting

`distribution_prediction_collection.report_evidence` consumes the report helpers on regenerated native-admitted evidence: it calls `prediction_uncertainty`, which calls `prediction_report`, and calls `prediction_diagnostic_report` for interval/PIT summaries. The helpers have no standalone CLI or execution authority; collection and `PredictionRun` retain provenance, byte/resource limits and lifecycle ownership.

`distribution_prediction_report.prediction_report(scores, events)` summarizes supplied four-arm forecast rows against the complete frozen event metadata, with no trade decision date or economic inputs. Every horizon and arm retains available, unavailable, missing-target and missing-row counts. Means are scoreable-only with metric-specific counts; unresolved CRPS and infinite log loss remain explicit. Tail checks show predicted probability, observed frequency and sample count. Paired same-mean, matched-point and training-empirical CRPS improvements each use that contrast's paired scoreable subset and retain excluded event IDs; the full-population improvement is unavailable if any pair is missing. Positive improvement favors the histogram, and horizon-one same-mean stays the sole primary contrast. Raw supplied rows/metadata are retained, not authenticated by this helper; native admission and the runner’s bounded evidence verification remain required.

All retained rows and event metadata must be finite-JSON serializable, including opaque nested provenance. Any present score/tail fields are validated even on unavailable or missing-target rows; those rows may omit the fields, and a missing target does not invent tail truth. Invalid retained payloads refuse rather than being silently stripped. Valid optional provenance and existing score/status semantics are preserved.

`distribution_prediction_diagnostic_report.prediction_diagnostic_report(diagnostics, events)` separately summarizes supplied interval/PIT rows. Coverage and width always appear together with their available count, while all unavailable or missing rows remain in population counts and raw evidence. It preserves unrandomized PIT brackets rather than replacing atoms with arbitrary midpoints or claiming uniformity. Any present interval or PIT fields are validated on every status; only available rows require all fields. Missing-target coverage remains boolean without inventing an observation. Malformed retained diagnostics refuse with nonretryable `INVALID_EXPERIMENT_SPEC`, without stripping evidence. These are calibration diagnostics, not fitted calibration or evidence that raw probabilities are calibrated.

`distribution_prediction_uncertainty.prediction_uncertainty` adds conditional ticker/calendar cluster intervals for paired CRPS improvements and tail calibration gaps, using explicit seed, resample count, fixed block length/origin and confidence supplied from the admitted prediction seal by `report_evidence`. Every measurement shares each whole-cluster draw from the frozen population; missing scores stay missing. Full/scoreable cluster counts and empty replicate counts accompany the intervals. They condition on frozen fits and nonempty scoreable replicates, not a refitted experiment. They add no significance or promotion cutoff, PIT randomization or uncertainty for descriptive interval coverage/width.

## Operator lifecycle

The prediction operator uses `argparse`, SQLite read-only URIs, existing native registration/artifact storage and neutral request/containment. `PredictionRun` fits every admitted inner and outer horizon, freezes the complete 1/3/5 forecast roster before each outer target read, then publishes scores and compact report summaries. Completed `collect`, `report` and repeated `run` rebuild and compare native training/fit/evaluation evidence without fitting or new outcome reads. JSON output includes the forecast and diagnostic summaries; bounded per-event sections retain raw scores, statuses and PIT brackets. The operator does not call the strategy runner or require quotes, payoff bridges, economics or a book.

Register a **fresh qualified native SIZE v1.7 source against the final installed code**, immediately before the experiment. Use the existing SIZE preflight/register flow; registration returns the sealed source `plan_ref`. No SIZE execution seal or candidate-search run is required. Do not reuse an older source registration from before the prediction operator landed. Source metadata provides the frozen cohort, feature qualification and nested folds; this operator does not build/register a source automatically. Keep the same installed sources and environment for preparation, registration, execution and replay.

`--source-spec` is the registered SIZE spec, `--source-request` its exact search request, and `--source-plan-ref` the returned ArtifactRef alone, not the whole registration result or a distribution plan.

Use a separate native `ExperimentSpec` with runner `experiments/distribution_prediction_run.py`, seed 0, `price_source: computed_moves`, empty economics and no file inputs. Its outer fold years must match the source plan. The distinct prediction-owned v1.0 registration must preserve the source snapshot, scope, ordered event IDs and holdouts exactly. Both own and source specs must omit `input_files` or declare an empty array on every action. A final-read declaration is forbidden.

The closed `distribution_prediction_request.v1` document contains exactly `schema_version`, `limits`, `max_events` and `uncertainty`. For example, a policy shape is:

```json
{
  "schema_version": "distribution_prediction_request.v1",
  "limits": {
    "parent": {"memory_bytes": 4294967296, "cpu_seconds": 3600, "wall_seconds": 3600},
    "job_wall_seconds": 3600
  },
  "max_events": 10000,
  "uncertainty": {
    "seed": 0, "resamples": 1000, "block_days": 7,
    "block_origin": "2023-01-01", "confidence": 0.95
  }
}
```

These illustrative caps are not a resource-feasibility claim. Choose explicit, platform-representable limits for the admitted study. Parent CPU bounds its own process; each child has its separate CPU cap, while parent wall time bounds the complete job. `max_events` bounds outer evaluation events, not ticker count. The chosen qualified source defines the requested 50-ticker cohort; the recipe retains native horizons 5/3/1, with horizon1 primary. No ticker whitelist or separate roster limit is invented by the CLI. `--budget-caps` contains exactly `max_fits`, `max_trees` and `child_limits`; fit/tree counts come from the validated fixed candidate recipe and source folds. `--recipe` is the unchanged document returned by `distribution_candidate_recipe.candidate_recipe` for the chosen H/HM/HMO feature set. `child_limits` contains exactly positive, platform-representable integer `memory_bytes`, `cpu_seconds` and `wall_seconds`; child memory must not exceed the parent ceiling.

With the prediction runner installed, invoke each action using the same inputs:

```sh
python -m experiments.distribution_prediction_run ACTION \
  --spec prediction-spec.json --source-spec source-spec.json \
  --source-request source-request.json --source-plan-ref source-plan-ref.json \
  --recipe recipe.json --budget-caps budget-caps.json --request prediction-request.json \
  --catalog EXISTING_CATALOG --store-root EXISTING_STORE --no-ledger
```

Actions, in order:

1. `prepare` checks native source identity and computes the distinct owner registration without reserving it.
2. `register` explicitly reserves that owner registration. Neither action fits or reads evaluation outcomes.
3. `preflight` checks the prediction plan; `seal` explicitly binds it to the installed runner and request.
4. `run` first probes for completed evidence read-only. Only an unfinished run reopens the existing catalog writable, under the original whole-job deadline.
5. `collect` or `report` verifies and returns stored completion. A completed `run` takes the same read-only collection path, including committed live-WAL records, without refitting or new target reads.

Every action requires `--no-ledger` and existing catalog/store paths. Output is one JSON result (exit 0) or typed refusal (exit 2); progress goes to stderr. Input-path failures and malformed lifecycle specs refuse `INVALID_EXPERIMENT_SPEC`; actual artifact corruption retains `INTEGRITY_FAILED`, and unavailable runtime storage is `RESOURCE_UNAVAILABLE`. Source/holdout/identity refusals remain intact. Synthetic operator tests establish routing and these boundaries, not real-study accuracy or permission to open protected holdouts.

Use a dedicated Linux main-thread process and a responsive output sink under an external process deadline: the final bounded-memory stdout copy occurs after internal containment exits. Stop after any failed invocation and inspect the available evidence. A failure after the create-only `started` claim leaves the attempt consumed even if the failure record cannot be written; never retry that started run in place or reuse partial state.

`distribution_prediction_collection.report_evidence(run, fits)` connects these reports to freshly verified stored evaluation evidence for every sealed outer year. It checks the entire frozen event/arm/horizon roster, retains missing-target/unavailable diagnostics, and returns compact numeric summaries plus hashes of the original evaluation records. Per-event rows and exclusion identities remain in the original bounded evidence rather than being duplicated inside a giant summary row. This library adapter is used by the neutral prediction runner, which publishes bounded summaries and recomputes/compares them during later collection. The adapter itself adds no new outcome reads, fits, writes or execution authority.
