# `engine/v2/models`

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

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

ArtifactMember, ModelBinding, ModelRelease, InferenceRequest, InferenceResult,
PredictionFrame,
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
(`admissible_table.py`, pinned as `N_ADMISSIBLE_BY_DEPTH_V1_HASH`), one
hash-verified loader for all three (`frozen_state.py`), release-member
helpers (`frozen_release.py`), and dependency-driven invalidation (`lineage.py`: every
frozen state declares a `Lineage`, and `propagate_corrections` intersects it
with Phase 3B `ChangeSet`s; `rebuild_order` gives the upstream-first rebuild
plan). The builders live in `engine/v2/models/training/residuals.py`.

P5-5 adds atomic staging and deployment: `stage_release`, `promote`,
`rollback`, `resolve_release`, `current_release`, `current_pointer`,
`pointer_history`, `StagedManifest`, `PointerState`, `DeploymentError`,
`StagingRefused`, `ReleaseNotStaged`, `NoPriorRelease`, and the constants
`STAGED_MANIFEST_V1`, `POINTER_STATE_V1`, `DEPLOYMENT_REFUSAL`.

<!-- public-interface: AdapterError, ArtifactInventoryMember, ArtifactMember, FrozenInference, InferenceAdapter, InferenceRequest, InferenceResult, JoblibEstimatorAdapter, JsonLinearAdapter, MODEL_ARTIFACT_INVENTORY_V1, MODEL_NOT_READY, MODEL_READY, MODEL_RELEASE_REFUSAL, MODEL_RELEASE_V1, ModelArtifactInventory, ModelBinding, ModelRelease, ModelReleaseInventory, ModelReleaseRefusal, PredictionFrame, RELEASE_BINDING_V1, RELEASE_REQUIREMENT_V1, ReleaseBinding, ReleaseIssue, ReleaseRequirement, RuntimeFitForbidden, default_adapters, release_issues, require_complete_release, ARTIFACT_INVENTORY_MEMBER_V1, DEPLOYMENT_ID, FEATURE_ROLES, FoldCoverageEntry, KNOWN_CLOCK_IDS, NON_MODEL_STATE_ITEMS, NonModelStateEntry, RELEASE_ID, current_release_inventory, non_model_state_inventory, registry_drift_issues, served_roles, tier4_fold_coverage, PAYOFF_LINE_ARTIFACT_V1, PAYOFF_SURFACE_ARTIFACT_V1, PayoffArtifactError, PayoffArtifactKey, PayoffArtifactLoader, PayoffArtifactRef, PayoffLineArtifact, PayoffSurfaceArtifact, fitting_forbidden, forbid_fitting, make_payoff_line_artifact, make_payoff_surface_artifact, no_fit_guard, payoff_artifact_key, serialize_payoff_artifact, RECALIBRATION_MAP_ARTIFACT_V1, RecalibrationArtifactError, RecalibrationArtifactLoader, RecalibrationArtifactRef, RecalibrationMapArtifact, make_recalibration_map_artifact, recalibration_artifact_key, serialize_recalibration_artifact, DEPLOYMENT_REFUSAL, POINTER_STATE_V1, STAGED_MANIFEST_V1, DeploymentError, NoPriorRelease, PointerState, ReleaseNotStaged, StagedManifest, StagingRefused, current_pointer, current_release, pointer_history, promote, resolve_release, rollback, stage_release, ADMISSIBLE_DEPTH_TABLE_V1, N_ADMISSIBLE_BY_DEPTH_V1_HASH, N_ADMISSIBLE_TABLE_ID, AdmissibleDepthTable, AdmissibleTableError, admissible_table_from_document, legacy_n_admissible_table, make_admissible_depth_table, n_admissible_for, FrozenState, FrozenStateError, FrozenStateLoader, FrozenStateRef, serialize_frozen_state, inventory_member, member_kind, release_member, state_node, LINEAGE_V1, DataDependency, Invalidation, InvalidationReport, Lineage, LineageError, StateNode, lineage_from_document, propagate_corrections, rebuild_order, DRIVER_RESIDUAL_POOL_ARTIFACT_V1, PAIRED_COLUMNS, PAIRED_RESIDUAL_POOL_ARTIFACT_V1, DriverResidualPoolArtifact, DriverResidualPoolKey, PairedResidualPoolArtifact, PairedResidualPoolKey, ResidualArtifactError, driver_residual_pool_key, make_driver_residual_pool_artifact, make_paired_residual_pool_artifact, paired_residual_pool_key, residual_artifact_from_document -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

The engine.v2.scoring package imports the verified frozen inference contract
and loader for stage bound model execution, and (P5-4) the payoff-calibration
artifact type read by the model stage's frozen path. engine.v2.models.training
imports the same artifact type plus the no-fit guard constructors to build one
from causal source rows. `tools/phase5_inventory.py` (not a v2 package, so not
part of this graph) is the P5-1 inventory's CLI.

<!-- consumers: engine.v2.scoring, engine.v2.models.training -->

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
