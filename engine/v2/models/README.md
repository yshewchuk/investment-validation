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

P5-5 adds atomic staging and deployment: `stage_release`, `promote`,
`rollback`, `resolve_release`, `current_release`, `current_pointer`,
`pointer_history`, `StagedManifest`, `PointerState`, `DeploymentError`,
`StagingRefused`, `ReleaseNotStaged`, `NoPriorRelease`, and the constants
`STAGED_MANIFEST_V1`, `POINTER_STATE_V1`, `DEPLOYMENT_REFUSAL`.

<!-- public-interface: AdapterError, ArtifactInventoryMember, ArtifactMember, FrozenInference, InferenceAdapter, InferenceRequest, InferenceResult, JoblibEstimatorAdapter, JsonLinearAdapter, MODEL_ARTIFACT_INVENTORY_V1, MODEL_NOT_READY, MODEL_READY, MODEL_RELEASE_REFUSAL, MODEL_RELEASE_V1, ModelArtifactInventory, ModelBinding, ModelRelease, ModelReleaseInventory, ModelReleaseRefusal, PredictionFrame, RELEASE_BINDING_V1, RELEASE_REQUIREMENT_V1, ReleaseBinding, ReleaseIssue, ReleaseRequirement, RuntimeFitForbidden, default_adapters, release_issues, require_complete_release, ARTIFACT_INVENTORY_MEMBER_V1, DEPLOYMENT_ID, FEATURE_ROLES, FoldCoverageEntry, KNOWN_CLOCK_IDS, NON_MODEL_STATE_ITEMS, NonModelStateEntry, RELEASE_ID, current_release_inventory, non_model_state_inventory, registry_drift_issues, served_roles, tier4_fold_coverage, DEPLOYMENT_REFUSAL, POINTER_STATE_V1, STAGED_MANIFEST_V1, DeploymentError, NoPriorRelease, PointerState, ReleaseNotStaged, StagedManifest, StagingRefused, current_pointer, current_release, pointer_history, promote, resolve_release, rollback, stage_release -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

The engine.v2.scoring package imports the verified frozen inference contract
and loader for stage bound model execution. `tools/phase5_inventory.py`
(not a v2 package, so not part of this graph) is the P5-1 inventory's CLI.

<!-- consumers: engine.v2.scoring -->

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
