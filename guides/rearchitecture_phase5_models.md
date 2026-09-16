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
