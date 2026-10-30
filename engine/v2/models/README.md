# `engine/v2/models`

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full contract (primary
interfaces, inputs/outputs, dependencies, failure semantics, invariants).

## Ownership

Implements the **Model inference — registry, artifact loading, inference adapters** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**3** of §4.1.

Replaces (§4.4): `models/registry.py`, `artifact loading and inference adapters`.

## Responsibilities

- The model registry and its champion resolution.
- Artifact loading with fingerprint verification.
- Inference adapters; residual and calibration state as frozen data.

## Non-responsibilities

- **Fit anything** — `engine/v2/models/training` does it instead.
- **Reach into a training recipe** — `engine/v2/models/training` does it instead.

## Public interface

`regression_target.target_descriptor` / `validate_target` bind explicit column,
prediction column, signed/nonnegative domain and definition hash into a detached
closed identity. Invalid/tampered documents raise `ValueError`; these are metadata,
not evidence that labels exist or permission to read them. Used by temporal planning.



`magnitude_candidate.magnitude_candidate_family` checks only the fixed
nonnegative recipe schema/model pair, without importing training. Full closed
recipe validation remains with the family fitter; this grants no fit authority.

`regression_candidate.regression_candidate_recipe` selects the existing Ridge, HGB
or ElasticNet policy with exact family parameters and explicit signed/nonnegative
domain. It is source-free metadata, not selection, fitting or experiment admission.

`magnitude_ridge_hgb_recipe.ridge_hgb_recipe` and
`validate_ridge_hgb_recipe` construct/validate closed variable-width stage-2
Ridge/HGB metadata from explicit ordered columns, a dictionary hash and one
bounded family parameter. They return detached policy data, not source admission
or an executable search plan. No training, sklearn or dictionary imports occur.

`regression_ridge_hgb_recipe.regression_recipe` and `validate_regression_recipe`
add a distinct closed recipe identity with explicit `signed` or `nonnegative`
target domain. Signed recipes bind no projection; nonnegative recipes bind
zero clipping. The existing magnitude recipe remains unchanged.

`magnitude_elasticnet_recipe.regression_elasticnet_recipe` and its
`validate_regression_elasticnet_recipe` bind a distinct closed policy with explicit
`signed`/`nonnegative` target domain and corresponding none/zero-clipping projection.
The existing magnitude recipe remains unchanged; metadata grants no source admission.

The names other packages may import are declared in the `public-interface`
directive below, which `checks/package_readmes.py` enforces. Everything else
is internal regardless of underscore convention. The following describes that surface.

ArtifactMember, ModelBinding, ModelRelease, InferenceRequest, InferenceResult,
PredictionFrame,
SizeFoldPolicy, ServingFoldDescriptor (from `serving_folds`, offline metadata),
ChooserFoldPolicy, ChooserServingFoldDescriptor, chooser_fold_outputs
(from `serving_folds`, chooser offline metadata), semantic_release_hash (from `deployment`),
FOLD_PANEL_SHA256_PATTERN (from `serving_folds`, canonical panel digest pattern),
BoundedStateReadError, read_bounded_state (from `frozen_state_io`, bounded file admission),
ServingFoldMetadata (from `adapters`),
ServingFoldRef, ChooserServingFoldRef, ServingFoldError, load_serving_fold,
load_chooser_serving_fold, validate_serving_fold_object, validate_serving_fold_descriptor
(from `serving_fold_loader`; the validators are the shared size-fold catalog checks
scoring's `gate_fold_selection` reuses),
FrozenInference, the inference adapters, ModelArtifactInventory,
ArtifactInventoryMember, ReleaseBinding, ReleaseRequirement,
ModelReleaseInventory, ReleaseIssue, ModelReleaseRefusal, release_issues,
require_complete_release, and their constants.

P5-1 adds the generated current-release inventory: `current_release_inventory`,
`registry_drift_issues`, `non_model_state_inventory`, `tier4_fold_coverage`,
`served_roles`, `FoldCoverageEntry`, `NonModelStateEntry`, and the constants
`DEPLOYMENT_ID`, `RELEASE_ID`, `KNOWN_CLOCK_IDS`, `FEATURE_ROLES`,
`NON_MODEL_STATE_ITEMS`.

P5-4 adds the v2-native no-fit guard (`fitting_forbidden`, `forbid_fitting`,
`no_fit_guard`) and the payoff-calibration artifact type: `PayoffLineArtifact`,
`PayoffSurfaceArtifact`, `PayoffArtifactRef`, `PayoffArtifactLoader`,
`PayoffArtifactError`, `PayoffArtifactKey`, `payoff_artifact_key`,
`make_payoff_line_artifact`, `make_payoff_surface_artifact`,
`serialize_payoff_artifact`, and the constants `PAYOFF_LINE_ARTIFACT_V1`,
`PAYOFF_SURFACE_ARTIFACT_V1`. Fitting stays out of this package by design --
`engine/v2/models/training/payoff.py` is the only place that calls
`native_payoff.fit_payoff_line`/`fit_runup_payoff_surface` and wraps the
result with the `make_payoff_*_artifact` constructors above.

P5-4 also adds the win-rate recalibration-map artifact
(`recalibration_artifact.py`): `RecalibrationMapArtifact`,
`RecalibrationArtifactRef`, `RecalibrationArtifactLoader`,
`RecalibrationArtifactError`, `recalibration_artifact_key`,
`make_recalibration_map_artifact`, `serialize_recalibration_artifact` and
`RECALIBRATION_MAP_ARTIFACT_V1`. Same pattern as the payoff artifact: keyed
by `(strategy, alpha, cutoff)`, content-hashed canonical JSON, a verified
loader. Below legacy's `min_pairs` floor the artifact freezes legacy's "no
map, ship the raw probability" answer (`fitted=False`) rather than being
absent, so scoring can tell it from a missing fold. The fit lives in
`engine/v2/models/training/recalibration.py`.

P5-4 (residuals and correction propagation; import from the submodules, the
package `__init__` is at its fan-out budget) adds the frozen residual pools
(`residual_artifact.py`: `DriverResidualPoolArtifact` keyed `(role, model_id,
fold)`, `PairedResidualPoolArtifact` keyed `(move_model_id, crush_model_id,
cutoff)`), the versioned DYN-SV depth -> `n_admissible` table
(`admissible_table.py`, pinned as `N_ADMISSIBLE_BY_DEPTH_V1_HASH`), the
DYN-SV chooser's k-NN analog population (`chooser_analog_pool.py`:
`ChooserAnalogPoolArtifact` keyed `(pool_id, cutoff)`, rows in source order),
one hash-verified loader for all four (`frozen_state.py`), release-member
helpers (`frozen_release.py`), and dependency-driven invalidation (`lineage.py`: every
frozen state declares a `Lineage`, and `propagate_corrections` intersects it
with Phase 3B `ChangeSet`s; `rebuild_order` gives the upstream-first rebuild
plan). The builders live in `engine/v2/models/training/residuals.py` and
`engine/v2/models/training/chooser_pool.py`.

P5-5 adds atomic staging and deployment: `stage_release`,
`mark_staging_succeeded`, `invalidate_staging_success`, `promote`, `rollback`,
`resolve_release`, `current_release`, `current_pointer`, `pointer_history`,
`StagedManifest`, `PointerState`, `DeploymentError`, `ConcurrentPromote`, `StagingRefused`,
`StagingNotSuccessful`, `ReleaseNotStaged`, `NoPriorRelease`, and the
constants `STAGED_MANIFEST_V1`, `POINTER_STATE_V1`, `DEPLOYMENT_REFUSAL`.

`semantic_release_hash` (from `deployment`) exposes the same pure semantic
identity calculation staging uses, without validating or changing a release.

This PR adds the one production release-root config key and a re-stage
tool for the versioned release hash: `production_release_root` (reads
`MODEL_RELEASE_ROOT`, refusing `MissingReleaseRoot` when it is unset; the
STORE root, one level above `production_deployment_root`),
`production_deployment_root` (`production_release_root() / "deployment"` --
what this module's own `promote`/`rollback`/`stage_release`/etc. actually
take as their `root`), and `restage_semantic_hash` (rewrites an
already-staged legacy-hashed manifest to the current semantic hash
version, in place, with no retraining); `promote`/`rollback` now refuse
`StaleReleaseHash` for a release staged under a superseded hash version.
See `ARCHITECTURE.md` §7.2–§7.5.

This PR (closes #101) closes three deployment-gate gaps: `stage_release`
refuses `StagingRefused` (`DUPLICATE_BINDING`) for two inference bindings
declaring the same `(role, strategy_id)`;
`promote`/`rollback` refuse the new `CorruptManifest` when a staged
manifest's content hash no longer matches its declared `release_hash`,
recomputed at swap time rather than only checking the hash-version tag;
and `rollback`'s target is now the second-from-top id of the pointer
history replayed as a promote/rollback stack, so chained rollbacks undo
chained promotions instead of flipping back to the release just left. See
`ARCHITECTURE.md` §7.1–§7.2 and §8.

<!-- public-interface: regression_target.target_descriptor, regression_target.validate_target, magnitude_candidate_family, regression_candidate_recipe, regression_recipe, validate_regression_recipe, ridge_hgb_recipe, validate_ridge_hgb_recipe, elasticnet_recipe, validate_elasticnet_recipe, AdapterError, ArtifactInventoryMember, ArtifactMember, FrozenInference, InferenceAdapter, InferenceRequest, InferenceResult, JoblibEstimatorAdapter, JsonLinearAdapter, MODEL_ARTIFACT_INVENTORY_V1, MODEL_NOT_READY, MODEL_READY, MODEL_RELEASE_REFUSAL, MODEL_RELEASE_V1, ModelArtifactInventory, ModelBinding, ModelRelease, ModelReleaseInventory, ModelReleaseRefusal, PredictionFrame, RELEASE_BINDING_V1, RELEASE_REQUIREMENT_V1, ReleaseBinding, ReleaseIssue, ReleaseRequirement, RuntimeFitForbidden, default_adapters, release_issues, require_complete_release, ARTIFACT_INVENTORY_MEMBER_V1, DEPLOYMENT_ID, FEATURE_ROLES, FoldCoverageEntry, KNOWN_CLOCK_IDS, NON_MODEL_STATE_ITEMS, NonModelStateEntry, RELEASE_ID, current_release_inventory, non_model_state_inventory, registry_drift_issues, served_roles, tier4_fold_coverage, PAYOFF_LINE_ARTIFACT_V1, PAYOFF_SURFACE_ARTIFACT_V1, PayoffArtifactError, PayoffArtifactKey, PayoffArtifactLoader, PayoffArtifactRef, PayoffLineArtifact, PayoffSurfaceArtifact, fitting_forbidden, forbid_fitting, make_payoff_line_artifact, make_payoff_surface_artifact, no_fit_guard, payoff_artifact_key, serialize_payoff_artifact, RECALIBRATION_MAP_ARTIFACT_V1, RecalibrationArtifactError, RecalibrationArtifactLoader, RecalibrationArtifactRef, RecalibrationMapArtifact, make_recalibration_map_artifact, recalibration_artifact_key, serialize_recalibration_artifact, DEPLOYMENT_REFUSAL, MAX_MANIFEST_BYTES, POINTER_STATE_V1, STAGED_MANIFEST_V1, DeploymentError, ConcurrentPromote, NoPriorRelease, PointerState, ReleaseNotStaged, StagedManifest, StagingNotSuccessful, StagingRefused, current_pointer, current_release, invalidate_staging_success, mark_staging_succeeded, pointer_history, promote, resolve_release, rollback, rollback_target, semantic_release_hash, stage_release, manifest_path, atomic_write_bytes, manifest_hash_matches, read_manifest, ADMISSIBLE_DEPTH_TABLE_V1, N_ADMISSIBLE_BY_DEPTH_V1_HASH, AdmissibleDepthTable, AdmissibleTableError, admissible_table_from_document, legacy_n_admissible_table, CHOOSER_ANALOG_DIMS, CHOOSER_ANALOG_K, CHOOSER_ANALOG_POOL_ARTIFACT_V1, ChooserAnalogPoolArtifact, ChooserAnalogPoolError, chooser_analog_pool_from_document, chooser_analog_pool_key, make_chooser_analog_pool_artifact, make_admissible_depth_table, n_admissible_for, FrozenState, FrozenStateError, FrozenStateLoader, FrozenStateRef, serialize_frozen_state, inventory_member, member_kind, release_member, state_node, LINEAGE_V1, DataDependency, Invalidation, InvalidationReport, Lineage, LineageError, StateNode, lineage_from_document, propagate_corrections, rebuild_order, DRIVER_RESIDUAL_POOL_ARTIFACT_V1, PAIRED_COLUMNS, PAIRED_RESIDUAL_POOL_ARTIFACT_V1, DriverResidualPoolArtifact, DriverResidualPoolKey, PairedResidualPoolArtifact, PairedResidualPoolKey, ResidualArtifactError, driver_residual_pool_key, make_driver_residual_pool_artifact, make_paired_residual_pool_artifact, paired_residual_pool_key, residual_artifact_from_document, MissingReleaseRoot, StaleReleaseHash, production_deployment_root, production_release_root, restage_semantic_hash, CorruptManifest, SizeFoldPolicy, ServingFoldDescriptor, FOLD_PANEL_SHA256_PATTERN, BoundedStateReadError, read_bounded_state, ChooserFoldPolicy, ChooserServingFoldDescriptor, chooser_fold_outputs, ServingFoldMetadata, ServingFoldRef, ChooserServingFoldRef, ServingFoldError, load_serving_fold, load_chooser_serving_fold, validate_serving_fold_object, validate_serving_fold_descriptor, regression_elasticnet_recipe, validate_regression_elasticnet_recipe -->

`FrozenStateLoader` accepts optional `max_bytes`: a positive per-object byte
ceiling enforced before hashing/decoding, with a bounded regular-file read.
Omitting it preserves the existing loader behavior.
`read_bounded_state(path, limit, no_follow=False)` admits regular files within a nonnegative allowance; opting in refuses final-component symlinks.
`BoundedStateReadError.code` is `INVALID_LIMIT`, `NOT_REGULAR_FILE` or `SIZE_LIMIT`; I/O errors propagate. Frozen-loader defaults are unchanged.

`ChooserFoldPolicy`, `ChooserServingFoldDescriptor` and `chooser_fold_outputs`
declare the separate implied-t1/runup-move offline fold contract. Release
preparation binds exact role/output, parent, panel, bytes and pool metadata;
`load_chooser_serving_fold` verifies exact catalog membership and returns immutable
artifact metadata/pools with an ephemeral inference view. It preserves raw D14
outputs without runup scaling. It does not select a causal fold or construct a
source-ready chooser declaration; genuine source finality/publication and training
bounds are still missing upstream prerequisites. Nightly refusal remains.
`FrozenInference.prepare_serving_fold` returns detached `ServingFoldMetadata` with
a required positive byte limit and optional exact `expected_bytes`; bounded
regular-file/no-follow verification shares the inference cache without exposing it.
Serving folds must be trusted uncompressed output from `tier4.serving_model` or Phase-4 cache build/backfill; those writers use binary handles with explicit `compress=0`.
Imported `--tier4-dir` and raw-byte inputs have the same caller-owned requirement: hashes do not prove provenance, and preparation does not guarantee compressed-input refusal or bound decoded memory.
Champion registry artifacts keep their separate `compress=3` behavior.

`magnitude_elasticnet_recipe.elasticnet_recipe` constructs closed, JSON-only research
recipe metadata from explicit ordered columns, dictionary hash, alpha and l1_ratio.
`validate_elasticnet_recipe` returns a detached canonical recipe or raises
`ValueError`; neither imports sklearn/training or reads sources. The training
adapter and `experiments/magnitude_candidate_recipe.py` consume these helpers;
no production registration currently does.

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

The engine.v2.scoring package imports the verified frozen inference contract
and loader for stage bound model execution, and (P5-4) the payoff-calibration
artifact type read by the model stage's frozen path. engine.v2.models.training
imports the same artifact type plus the no-fit guard constructors to build one
from causal source rows, and closed ElasticNet and Ridge/HGB recipe validation for its numeric adapter. engine.v2.ops.cli imports the no-fit guard
(`no_fit_guard`) for the read-only `ops rescore` command. `tools/phase5_inventory.py`
(not a v2 package, so not part of this graph) is the P5-1 inventory's CLI.
`experiments/magnitude_search_trials.py` and `experiments/magnitude_search_execution.py`
consume `no_fit.forbid_fitting` before native trial claims and fixed-child fitting.
`experiments/magnitude_search_outer.py` consumes the legacy/v2 no-fit guards
before selected outer refit claims; completed evidence replay never fits.
`experiments/magnitude_candidate_recipe.py` consumes the closed family recipe
constructors; `magnitude_candidate_fit.py` checks both legacy/v2 no-fit guards
before recipe, column or array inspection, then delegates to the public fitter.
`experiments/distribution_candidate_recipe.py` consumes `magnitude_ridge_hgb_recipe.ridge_hgb_recipe` for its source-free matched-point declaration; it does not fit.
`serving_fold_preflight.py` consumes the chooser policy/descriptor types
and `semantic_release_hash`, plus `engine.v2.foundation` typed-document conversion,
for offline fold metadata; its helper `serving_fold_checks.py` imports
`FOLD_PANEL_SHA256_PATTERN`. `pinned_panel.py` checks the pinned panel copy against
`engine.v2.data`. `tools/phase5_prepare_release.py`
consumes `ChooserFoldPolicy` and `chooser_fold_outputs`, obtains registered
policies, and invokes `serving_fold_preflight.preflight_chooser_fold`
before publication.
`release_layout.py` reads catalogs via `release_objects.read_regular_file`, a bounded final-component no-follow open. `tools/phase5_prepare_release.py` bounds its serving-fold reads (`read_bounded_state`, `MAX_FOLD_BYTES`) and frozen-state reads (`MAX_STATE_BYTES` by default); its model-artifact reads (`MAX_MODEL_BYTES`) are bounded no-follow, and the bootstrap reads each payoff/recalibration artifact once the same way and validates those bytes with the loaders' `load_bytes` (no second read); its training-root payload readers are bounded no-follow reads too; `checks/phase5_acceptance.py`'s staged-member and report reads are uncapped (tracked in #819).
`checks/phase5_acceptance.py` publishes staging success after acceptance.
`tools/phase5_prepare_release.py` invalidates prior success before re-preparation.
`checks/phase5_release_assembly.py` re-stages a prior release under a new id
(`deployment.stage_release`, `resolve_release`, `StagingRefused`, `DeploymentError`,
`MAX_MANIFEST_BYTES`, and `releases.ModelReleaseRefusal`) when carrying pool objects forward;
`release_layout.py` (which re-exports the `release_objects.py` names `write_object`, `ReleaseLayoutError`, `ObjectVerificationError`, `deployment_root` and `object_relpath`, alongside `STATE_SPECS` and `read_catalog`) is imported by `checks/phase5_*.py`, `tools/phase5_*.py`, and `engine.v2.scoring.release_bindings` (for catalog cutoff invariants); `release_layout.py` reaches the public `manifest_path`, `read_manifest`, `manifest_hash_matches` and `atomic_write_bytes` for catalog layout checks.
`tools/phase5_state_bootstrap.py` (called by the stage job in `bootstrap` mode, §2.2.2) calls `write_release` with `candidate_only`, `classify_prior`, `deployment.resolve_release`, `read_bounded_state` and `current_release_inventory`, and publishes the candidate catalog through `release_layout.write_manifest(candidate_only=True)`, which raises `CatalogConflict`.
`tools/phase5_state_stage_job.py` (behind the `phase5_state_stage` ops job) calls `carry_forward_release`,
`classify_prior` (bootstrap mode) and `prior_carry_set` and the same bounded reader, and takes its inventory from `current_release_inventory`; it then calls `deployment.mark_staging_succeeded`.
`engine/v2/ops/pool_refresh.py` (the nightly pool sidecar, `Service._reconcile_pool_nightly_refresh`) reads the `DEPLOYED` pointer through the bounded state reader
each tick (`PointerState` decoded directly) and submits the stage and `models_promote` jobs; the supervisor and `engine/v2/ops/nightly_trigger.py` (scheduled-serve hold) resolve `production_release_root`.
engine.v2.serving.operations (P6-4) imports the deployment pointer
(`current_pointer`, `resolve_release`, plus the two refusal exception types)
to serve the deployed model release read-only at `/models/release.json` --
never fits or promotes anything, only resolves the one pointer already on
disk.

<!-- consumers: engine.v2.research, engine.v2.scoring, engine.v2.models.training, engine.v2.ops, engine.v2.serving -->

## Usage

Call require_complete_release before loading any artifact. It checks complete
role/strategy/clock bindings, exact feature order, and each binding required
estimator, transform, residual and calibration members.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
