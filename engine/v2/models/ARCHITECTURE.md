# `engine/v2/models` — architecture

See the root [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) for the layer
table and enforced import direction. This doc follows
`docs/COMPONENT_ARCHITECTURE_TEMPLATE.md`'s section order.

## 1. Purpose

Layer 3.0 (root doc §2). Replaces legacy `models/registry.py` and the
inline artifact/inference plumbing scattered through `engine/score.py`.
This package owns four things:

- **Contracts** (`contracts.py`) — the immutable shapes a frozen release is
  made of (`ModelBinding`, `ModelRelease`, `ArtifactMember`) and the
  inference call/response pair (`InferenceRequest`, `InferenceResult`,
  `PredictionFrame`). Schemas only: no logic, no I/O.
- **Verified loading and inference** (`adapters.py`, `loader.py`) — the
  adapters that turn a hash-verified member's bytes into a read-only
  predictor, and `FrozenInference`, the one class that runs a request
  through them. Never fits anything (`RuntimeFitForbidden`,
  `ReadOnlyArtifact` block every mutating estimator method by name).
- **Release completeness and today's real inventory** (`releases.py`,
  `inventory.py`) — the abstract "is this release complete" contract, and
  the concrete answer for what is on disk right now (`current_release_inventory`
  reads `data/models/*` and the champion registry; it never hand-writes
  `registry.json`).
- **Frozen non-model serving state** (`admissible_table.py`,
  `analog_artifact.py`, `chooser_analog_pool.py`, `payoff_artifact.py`,
  `recalibration_artifact.py`, `residual_artifact.py`,
  `trailing_cutoff_artifact.py`, dispatched through `frozen_state.py`,
  released through `frozen_release.py`, invalidated through `lineage.py`) —
  the frozen-artifact answer to every piece of calibration state legacy
  used to recompute live from an unversioned file or in-process cache. Each
  is an immutable, content-hashed record with a verified loader; none of
  them are models, so none of them go through `adapters.py`/`loader.py`.
- **Content-addressed staging and the deployment pointer**
  (`deployment.py`) — the durable store a `ModelRelease` is staged into
  once it passes completeness, and the one atomic pointer (`DEPLOYED`)
  that names which staged release is live.

`engine/v2/models/training/` (layer 6.0) is the *only* place any of this
gets fit — see its own package doc reference in the root index. Nothing in
`engine/v2/models` reads a live panel, calls `.fit()`, or writes
`registry.json`.

The research-only `training.magnitude` boundary fits two original-unit candidates
from ops-admitted recipe data for the native experiment CLI. Ridge's
median imputation, full missing indicators and scaling learn from training rows
only; HGB freezes training-all-null columns to zero and routes other nulls natively,
without a random early-stopping split. Both use
fresh per-fold/horizon estimators, one numerical thread and finite nonnegative
predictions. The caller owns protected admission, label qualification and evidence;
this module receives recipe data and numeric matrices, with no I/O or model release.
Invalid shapes/nonfinite labels or predictions raise `ValueError`; the experiment
caller translates fitting failures to `EXPERIMENT_VARIANT_FAILED`. The shared
no-fit guard runs before work. Existing legacy-parity recipes are unchanged.

`magnitude_elasticnet_recipe.elasticnet_recipe` builds closed JSON-only research
metadata without importing training or sklearn. Ordered selected columns and a
caller-supplied dictionary hash bind preprocessing, fixed estimator parameters,
seed and projection; a recipe hash is not a source/fold/candidate seal or proof
of dictionary membership. `training.magnitude_elasticnet.fit_elasticnet_predictions`
consumes this recipe and matching named numeric matrices. It learns imputation,
missing indicators and scaling from training rows only, returning finite nonnegative
forecasts, the full standardized coefficient vector (values and missing indicators),
selected source columns in recipe order, iterations and dual gap.
The experiment dispatcher calls this adapter; production entrypoints remain unwired.
Causal membership, source identities,
inner-only subset/parameter choice, resource/attempt caps and failure recording
are caller responsibilities, not guarantees derived from these arrays.

| ElasticNet condition | Outcome |
|---|---|
| Either no-fit guard is active | Its `RuntimeFitForbidden` precedes recipe or array inspection |
| Invalid closed recipe, column order, matrix dimensions, infinities or label domain | `ValueError`; no fit or partial result |
| Nonconvergence or nonfinite fit evidence/predictions | `ValueError`; no partial result, retry or iteration increase |
| Same valid inputs and numerical environment | Deterministic fresh fit, not cached or journal replay |
| Cache, transaction or partial-write concerns | No cache, I/O, persistence or publication; v1 recipes and results are unchanged |

The separate `magnitude_ridge_hgb_recipe` API constructs/validates source-free
`magnitude_ridge_hgb_recipe.v1` metadata for the bounded stage-2 Ridge/HGB grid.
It binds explicit ordered unique columns, a canonical dictionary hash, fixed
preprocessing/estimator policy and derived width; no dictionary or source is read.
The `history_ridge`/`history_hgb` family names reuse the training-layer private
estimator constructor (`training/_magnitude_estimators.build_estimator`,
imported by the training fitters) unchanged; each numeric adapter owns its
guards and fitting, not the constructor. This is not the fixed eleven-column
v1 execution envelope. Dictionary membership and truthful matrix order remain caller-owned; a hash alone grants no admission.
`training.magnitude_ridge_hgb.fit_ridge_hgb_predictions(..., columns=...)`
checks both no-fit guards before inspecting metadata or numeric inputs, validates
the closed recipe and exact declared column order, then copies numeric matrices.
A fresh one-thread fit learns only training rows: Ridge keeps every selected column
through median imputation, full missing flags and scaling; HGB freezes training-all-null columns to
zero at training/prediction while routing partial nulls natively. It returns
`magnitude_ridge_hgb_fit.v1` with the validated recipe hash and detached finite
nonnegative predictions. This is a production-unwired numeric boundary: no source/holdout/fold
admission, CLI, selection or registration; existing eleven-column recipes, fitters
and source closures remain unchanged.

| Ridge/HGB condition | Outcome |
|---|---|
| Either no-fit guard active | Its `RuntimeFitForbidden` precedes recipe, column or array inspection |
| Invalid closed policy/order, unconvertible inputs, matrix shape, infinite features or nonfinite/negative labels | `ValueError` before fit; no partial result |
| Wrong prediction shape or nonfinite predictions | `ValueError` after fit, before projection; no partial result |
| Same valid inputs and numerical environment | Deterministic fresh fit, not cached or journal replay |
| Cache, retry, transaction or partial-write concerns | No cache, retry, persistence or publication |

An automated nightly append cycle for the pool/residual/cutoff artifacts:
release assembly is §2.1, staging §2.2, and the three-phase tick sidecar is §2.2.1; see #192.

## 2. Primary contracts and public interfaces

The full list is `README.md`'s `<!-- public-interface: ... -->` directive
(machine-checked by `checks/package_readmes.py`); the deployment surface:

- `deployment.stage_release(root, release, inventory, payloads, *, clock) -> StagedManifest`
  — validate, then durably stage. Never touches `DEPLOYED`.
- `deployment.semantic_release_hash(release: ModelRelease) -> str` — pure
  `sha256:` identity over the typed release's inference meaning. Binding/member
  collection order and storage paths are ignored; feature/output order is not.
  No I/O, cache, completeness validation or artifact verification; those remain
  with staging/loaders. It does not rewrite stored manifest hashes or versions.
- `deployment.promote(root, release_id, *, clock, expected_previous_release_id=None, require_empty_store=False) -> PointerState` /
  `deployment.rollback(root, *, clock) -> PointerState` — the one atomic
  pointer swap, in either direction. Rollback selects the prior incumbent
  from deployment history; an empty or single-entry undo stack raises the
  typed `NoPriorRelease` refusal without changing the pointer. Operator plans
  pin incumbent sequence and target; stale plans refuse, and recovery recognizes
  only that plan's exact swap, even if history or receipt recording was lost.
- `deployment.rollback_target(root) -> str` — read-only resolution of the
  incumbent `rollback` would select; raises `NoPriorRelease` when none exists.
- `deployment.resolve_release(root, release_id) -> ModelRelease` /
  `deployment.current_release(root) -> ModelRelease | None` — read a staged
  release by id, or by the live pointer.
- `deployment.production_release_root() -> Path` — the one configured
  production STORE root; see §7.4.
- `deployment.production_deployment_root() -> Path` —
  `production_release_root() / "deployment"`, the directory this module's
  own root-taking functions actually want; see §7.4.
- `deployment.restage_semantic_hash(root, release_id) -> StagedManifest`
  — rewrite an already-staged manifest's hash to the current semantic
  version, in place, with no retraining; see §7.5.
- `deployment.mark_staging_succeeded(root, release_id) -> None`
  — after the Phase 5 acceptance gate passes, atomically record success for
  the exact staged `release_id` and `release_hash`; see §7.2.
- `deployment.invalidate_staging_success(root, release_id) -> None`
  — remove any prior candidate success record before Phase 5 re-preparation
  can replace that candidate's manifest or state catalog; see §7.2.
- `release_bindings.resolve_release_binding(release_root) -> ScoringReleaseBinding`
  and `release_bindings.resolve_production_release_binding() -> ScoringReleaseBinding`
  live in `engine/v2/scoring/` (layer 5.0, below `models` — this package
  does not import scoring); they are the production readers of what this
  package stages. Documented in full in
  `engine/v2/scoring/ARCHITECTURE.md`'s `release_bindings.py` section.

Every model-binding/frozen-state member and its adapter is reached only
through the loader for its family (`FrozenInference` for models,
`FrozenStateLoader`/`PayoffArtifactLoader`/`RecalibrationArtifactLoader` for
non-model state) — never by opening a staged object file directly outside
`deployment.py`'s own hash verification.

`FrozenStateLoader(root, max_bytes=None)` accepts an optional positive integer
byte ceiling below `sys.maxsize`, fixed for that loader. A bounded load admits
only a regular file, checks its opened size before reading, and reads at most the ceiling plus one
byte before hash or JSON decoding. The extra byte detects growth after the
size check. Existing callers omitting the ceiling retain their read contract.

| Bounded frozen-state condition | Outcome |
|---|---|
| Invalid limit, nonregular object, or object exceeds the ceiling | `FrozenStateError`; no decoded state or new cache entry |
| Object at/below the ceiling | Existing byte-hash, canonical-hash and typed validation apply |
| Cache, retry, transaction, partial write | Existing per-loader content-hash cache; no retries or writes; failure returns no partial state |

`PayoffArtifactLoader` and `RecalibrationArtifactLoader` read artifact files through `read_artifact_bytes` (regular file, at most `MAX_ARTIFACT_BYTES` = 512 MiB, else the loader's "missing artifact" error) and expose `load_bytes(raw, content_hash)`, which verifies `raw` against `content_hash` (hash mismatch refuses, also on a cache hit) and then applies the same parse and validation as `load`.

### Snapshot-pinned D14 monthly datasets

`training.build_d14_dataset` builds only the `implied_t1` and `runup_move`
monthly inputs from explicit catalog/store/snapshot/receipt identifiers.
It verifies provenance before calling `d14_dataset.calculate_frame`.
That public helper checks snapshot identity for direct callers but does not
certify calendar provenance or attach `training_inputs`. Its bounded-read,
decision and daily-state helpers are declared in the training README.
Bounded native Repository scans pin earnings,
daily state and the stored `feature_panel` history block to one SnapshotRef.
History is left-joined by ticker/event date; earnings defines decision membership.
Stored history values and missingness are retained from the historical producer,
without reconstruction through `scan_panel_row`.
This is root §5's named compatibility-history exception for these two recipes,
not a claim of native historical production or source availability/finality.
Only daily features at the D14 entry and labels at the session-aware exit are
calculated. The original calendar is read from one explicit committed import
receipt, verified against the snapshot and object hash, with native projection
math. The consumed calendar bytes are hash-verified and capped at 16 MiB.
No mutable calendar file, head resolution or legacy reader is used.

The opt-in `python -m tools.phase5_d14_inputs` CLI has no nightly or ops caller.
It requires explicit catalog, artifact root, snapshot and import-receipt
identifiers, delegating to that same public API. The fixed
2017–2026 event window is the current monthly recipe population, with daily
history from 2016. Snapshot coverage does not widen that recipe contract.
The existing preparation, fold planning and label/membership receipts remain
unchanged. `run_training_job(input_identity=...)` additionally binds the pinned
source identity in `job.json`: `schema_version`, `snapshot_id`, `receipt_id`,
`calendar_hash` and `table_versions`. Changed sources refuse resume even if values
match. Champion jobs, scheduling, source-finality admission, historical panel
reconstruction, held-out residual pools and serving publication are separate.
No eligible decisions refuse before history/daily reads, preventing a zero-fold
success. An absent panel history row stays missing; missing quotes retain legacy as-of
and positional-lag behavior. Nonpositive prices produce missing move labels.

| Condition | Outcome |
| --- | --- |
| Input exceeds table, ticker or calendar-byte bound | `RESULT_LIMIT_EXCEEDED` |
| Snapshot identity or committed receipt association differs | `IDENTITY_CONFLICT` |
| Committed receipt has no pinned reference inputs | `SNAPSHOT_NOT_READY` |
| Explicit snapshot identifier is absent from the catalog | `SNAPSHOT_NOT_FOUND` |
| Null session or fewer than 14 prior calendar sessions | Event excluded, matching legacy membership |
| Catalog, required table/calendar, non-null session outside AMC/BMO, role, integer history or decision population invalid | `CONTRACT_MISMATCH` |
| Calendar object bytes differ from their registered hash/size | `OBJECT_CORRUPT` |
| CLI output is inside configured data root or input store | argparse exit 2 before reads/writes |
| CLI receives a typed data/training refusal | stderr refusal detail and exit 2; no traceback |

### 2.1 Pool release assembly

`release_objects.write_object` returns hash/path in the flat deployment layout; objects are durable before return and publication is create-only. Whole-release concurrency and root ownership stay with callers.

| `write_object` condition | Outcome |
|---|---|
| New object | Durable before return; only invocation-owned staging is cleaned; no unbounded read or cross-filesystem copy fallback |
| Existing/racing identical regular object, or retry | Verify exact bytes (allowance = payload length, including zero) and sync directories; keep the incumbent |
| Wrong/oversized/unreadable bytes or symlink to a regular file | `ObjectVerificationError` (`ReleaseLayoutError`); incumbent retained |
| `read_regular_file(path, limit)`: `limit` not a non-negative int (bool excluded) | `ReleaseLayoutError`; nothing is opened |
| `read_regular_file`: absent path / regular file within `limit` (zero allowed) / directory, FIFO, or longer than `limit` | `None` / its bytes / `ReleaseLayoutError`; a symlink raises `OSError` (callers wrap it) |
| Dangling/nonregular-target symlink, other collision or publication I/O failure | `ReleaseLayoutError` with original `OSError`; never replace; no catalog/pointer (a complete unreferenced object may remain) |

`checks/phase5_release_assembly.py`, beside the catalog layout in `engine/v2/models/release_layout.py` (object hashing, `write_object` and the bounded catalog read live in `release_objects.py`, re-exported by `release_layout`; the `modules_available` probe stays in `checks/phase5_modules.py` because production code may not import dynamically):

- `derive_catalog(prior_catalog, *, release_id, expected_prior_release_id, outputs, chooser, cutoff, monthly=False, retrain_cutoff=None) -> dict`
  — pure. Returns the candidate catalog body: the six pool members
  (`driver_residual_pool:{size,implied_t1,runup_move}`, `paired_residual_pool`,
  `trailing_pnl_cutoff`, `board_analog_matcher`) and `chooser_analog_pool` each
  get the full set of supplied objects (`outputs` holds one or more
  `PoolOutput` per pool member; `chooser` is one or more; object names are
  unique within a member, and every non-carried object carries `cutoff`), replacing that
  row's objects wholesale; every other member row is copied unchanged;
  `sources.pool_cutoff` advances to `cutoff` and `sources.carried_from` names
  the prior release id. A `PoolOutput` may be `carried=True` (default
  `False`): an object that does not change between monthly refreshes (driver
  residual pools, which do not grow with new events and change only on a
  retrain; the chooser row until the planned monthly advance, §2.3, which does not exist yet) is re-supplied unchanged. It carries a
  caller-supplied `cutoff` (an extended ISO date not after `cutoff`; catalog rows
  record none, `prior_carry_set` supplies the prior `pool_cutoff`) and must equal an object of the
  same member in the prior row (name, `content_hash`, `payload_bytes`). Every
  non-carried object must carry `cutoff` exactly. `prior_carry_set(release_root,
  prior_release_id, monthly=False) -> (pool_cutoff, tuple[PoolOutput])` returns every prior
  pool/chooser object as `carried=True`, recorded at the prior `sources.pool_cutoff`
  (`PRIOR_MISMATCH` on a missing prior or structurally malformed rows, object keys or `pool_cutoff`;
  wrong-typed object fields are not checked here and refuse later as `POOL_SET_MISMATCH`, or `CHOOSER_MISMATCH` for the chooser). With `monthly=True`, `prior_carry_set` returns only board, paired and trailing objects. Monthly assembly supplies exactly the three driver residual pools and chooser; board, paired and trailing members carry from the stage-time base. It sets the candidate `pool_cutoff` to the later of the supplied base cutoff and `retrain_cutoff`, so a first retrain can stage when its exclusive boundary advances the base; the catalog invariant remains `retrain_cutoff <= pool_cutoff`.
- `classify_prior(release_root, prior_release_id) -> PriorClass` — read-only; handled refusal, validation and I/O failures return `DAMAGED` rather than raising. Kind `NONE`/`NO_POOL_CATALOG`/`USABLE`/`DAMAGED` with a fixed reason (§2.2.2). Every present catalog (with or without `pool_cutoff`) must verify against its staged release and `deployment_id`; a legacy catalog without `pool_cutoff` is not checked for pool rows or objects, which it need not have. `USABLE` needs a valid catalog with an ISO `pool_cutoff`, optional ISO `retrain_cutoff` not after it, well-formed pool rows, objects whose hash and size verify, and a `deployment_id` equal to the staged release's.

**Prior classification (#1102).** The `phase5_state_stage` job classifies the prior with one read-only helper, `classify_prior(release_root, prior_release_id)`; it never writes:

| Prior state | Class | Outcome (the `NONE`/`NO_POOL_CATALOG` bootstrap rows: the job and its sidecar submission are built, §2.2.2) |
|---|---|---|
| Pointer absent (nothing ever promoted) | `NONE` | bootstrap; `none` stands for the prior id in phase keys and `expected_previous_release_id` is null |
| Catalog file absent at `catalog_path` for an id that does not start with `pools-` (a release this cycle stages always has a catalog, so an absent one under a `pools-*` id is `DAMAGED`), or a valid catalog whose `sources` has no `pool_cutoff` (legacy-bootstrapped) | `NO_POOL_CATALOG` | bootstrap against that pointer id |
| Catalog valid with `pool_cutoff` | `USABLE` | nightly carry-forward (§2.2), unchanged |
| Catalog present but a symlink, not a regular file, oversize, unparseable, wrong `release_id`/`deployment_id` (also against the staged release), malformed rows or objects, an object whose hash or size does not verify, or a non-ISO `pool_cutoff` | `DAMAGED` | no bootstrap: refuses `VALIDATION_FAILED` (`details.code` `PRIOR_MISMATCH`); a damaged store is operator recovery (#154), never silently replaced |

In `bootstrap` mode the job classifies the prior (`classify_prior`, §2.1) before any object is written, with the refusals in the table below, and the light checks expect `sources.carried_from` null. The §2.2.2 sidecar submits `bootstrap` mode.

| Condition | Outcome |
|---|---|
| Prior `DAMAGED` | `VALIDATION_FAILED`, fixed message, `details.code` `PRIOR_MISMATCH` |
| Prior `NONE`/`NO_POOL_CATALOG` | the §2.2.2 assembly |
| Prior `USABLE` while `mode` is `bootstrap` | `VALIDATION_FAILED`, fixed message, `details.code` `PRIOR_USABLE` |

- `carry_forward_release(release_root, *, prior_release_id, release_id, inventory, outputs, chooser, cutoff, monthly=False, retrain_cutoff=None) -> Path`
  — verifies the prior staged release, untouched rows and supplied objects on disk,
  re-stages the prior bindings under the new `release_id` (`deployment.stage_release`),
  then writes the candidate catalog. It never touches `DEPLOYED`.

`derive_catalog` and `carry_forward_release` refuse with a stable `ReleaseAssemblyRefusal.code`; stray `KeyError`, `TypeError`, `AttributeError`,
`ValueError`, `RecursionError` map to `PRIOR_MISMATCH` (inventory or stage-time ones to `INVENTORY_MISMATCH`,
publish `OSError` to `PUBLICATION_FAILED`).

| Condition | Code |
|---|---|
| Prior release id empty, not a string or unsafe (`/`, `\`, NUL, leading `.`); prior catalog malformed, wrong id, or `deployment_id` differs from its staged release; a pool/chooser row missing, duplicated or not `STAGED`; no extended-ISO `sources.pool_cutoff`; prior manifest, model payload or untouched STAGED row object missing, corrupt, oversize or not a regular file | `PRIOR_MISMATCH` |
| Nightly outputs do not cover exactly the six pool members, or monthly outputs do not cover exactly the three driver members; a carried object absent from the prior row of its member; duplicate object name in a member; malformed field or hash | `POOL_SET_MISMATCH` |
| A carried chooser object absent from the prior chooser row; chooser empty, not `PoolOutput`, wrong member id, malformed name, hash or `payload_bytes`, or duplicate name | `CHOOSER_MISMATCH` |
| A non-carried nightly object's `cutoff` differs from `cutoff`; a monthly supplied object's `cutoff` differs from `retrain_cutoff`, or a monthly chooser's name-encoded cutoff differs from `retrain_cutoff` (driver names retain model/fold identity); a carried object's cutoff is not an extended date or is after `cutoff`; either cutoff is not an extended `YYYY-MM-DD` date | `CUTOFF_MISMATCH` |
| Neither `pool_cutoff` nor `retrain_cutoff` strictly advances its prior value (a retrain cutoff must strictly advance when the prior has one; the first retrain has no prior `retrain_cutoff`) | `CUTOFF_NOT_ADVANCED` |
| Supplied object missing, unreadable, not a regular file, `payload_bytes` above `MAX_OBJECT_BYTES`, longer than its claim, or bytes do not hash to the claim | `OBJECT_UNVERIFIED` |
| Candidate id empty, unsafe or equal to the prior's; inventory `release_id` differs from it; candidate catalog or staged manifest exists with different or corrupt content | `ID_CONFLICT` |
| Inventory missing, malformed, member hashes differ from the prior payloads, incomplete, incompatible with the prior bindings, or its `deployment_id` differs from the prior's | `INVENTORY_MISMATCH` |
| Filesystem error or lock timeout while publishing (inputs not at fault) | `PUBLICATION_FAILED` |

Supplied objects, prior payloads, both catalogs and staged manifests are
opened without following links, must be regular files, and are read under a
byte cap; the lock file is opened without following links and must be regular.
A missing, symlinked, non-regular or oversize prior refuses `PRIOR_MISMATCH`;
a corrupt, symlinked, non-regular or oversize existing candidate refuses `ID_CONFLICT`, an absent one is created. Staging and the catalog write
run under one exclusive `flock` on `deployment/.carry_forward.lock`; a loser
that waits out `LOCK_WAIT_S` refuses `PUBLICATION_FAILED`; otherwise identical
content succeeds and different content refuses `ID_CONFLICT`.

| Rule | Behaviour |
|---|---|
| R1 verified input | Prior release, each output, chooser row, identities, hashes, cutoff, inventory identity and prior model payload hashes validated before any candidate artifact is written |
| R2 cache | None; the prior catalog, staged manifest and objects are re-read each call |
| R3 retry | Repeatable only with the same validated inputs; no fallback to old pools or live registries |
| R4 transaction | Candidate artifacts are written only under the new candidate id (objects are content-addressed); the only other write is the `deployment/.carry_forward.lock` coordination file |
| R5 partial failure | Orphaned objects or a staged manifest without a catalog may remain and a retry completes it; no catalog, nothing promoted |
| R6 idempotency | Identical inputs give identical catalog bytes (no clock in the catalog); same id with different content refuses `ID_CONFLICT` |

### 2.2 `phase5_state_stage` job (nightly pool staging)

An ops job kind (`engine/v2/ops/state_stage.py`; logic in `tools/phase5_state_stage_job.py`
because the assembly helpers live in `checks/`; `projection` resource class (its pinned legacy read set exceeds `delivery`'s scratch), write lease on
`deployment_pointer`; `state_stage_plan` builds the plan; the §2.2.1 sidecar or an operator submits it). It
stages a candidate release; it never touches `DEPLOYED` (`models_promote` is separate).

- Parameters: `release_root` (the store's parent; empty resolves the production root at
  plan time), `prior_release_id` (null only with `mode` `bootstrap` and no prior), `mode` (`carry` default, `bootstrap` (§2.2.2), or `monthly`; `bootstrap` also takes the `bootstrap` mapping there), `release_id`, `cutoff` (extended ISO), `state_files` and
  `input_bindings` (a mapping from each state file name to its `training` output id; the job framework requires
  the producing job to have succeeded), plus a required `legacy_manifest.json` binding (the pinned legacy
  manifest, as `training` jobs bind it): the kind declares a `legacy_store` read, so the job's read set
  carries the legacy tree the inventory reads the champion registry from. A plan without the manifest
  is refused `INVALID_REQUEST`.
- In `carry` mode the bound files must cover exactly the advancing members `paired_residual_pool`,
  `board_analog_matcher`, `trailing_pnl_cutoff` (`bootstrap` mode binds the full set, §2.2.2). Paired and board objects carry `cutoff` as
  the last name segment; trailing objects are stamped with `cutoff`.
- In `carry` mode every other prior pool/chooser object, and every prior object of an advancing member not
  re-supplied, is carried (`prior_carry_set`), so older causal keys stay selectable and
  driver pools and the chooser keep their hash until a retrain (the planned monthly refresh, §2.3, is not built) re-supplies them.
- In `monthly` mode the worker reads the deployed pointer and catalog again at stage time, uses that release as the base, accepts exactly the three driver pools and chooser, and carries board, paired and trailing members. The public `classify` helper requires `monthly` to be a bool and refuses other values before reading inputs. The requested `cutoff` is `retrain_cutoff`. This stage path is plumbing only and has no production submitter until the monthly chain wiring split is implemented.
- When a deployed base is `NO_POOL_CATALOG`, the monthly retrain decision is `not_due` with reason `base_pool_catalog_unavailable`; this leaves the ordinary nightly bootstrap path available (§2.2.2). With no deployed release, the first eligible monthly decision remains due. A damaged base refuses planning, and monthly staging itself requires a `USABLE` base with carried members.
- Output: a re-verified candidate release (never `DEPLOYED`) and `stage_result.json`. The
  inventory comes from the legacy champion registry under the new id; drift refuses before any write.
- Refusals map to `VALIDATION_FAILED` with fixed messages and `details` (`code` of a
  `ReleaseAssemblyRefusal`, or `PRIOR_USABLE`, `BINDINGS_DIFFER`, `EMPTY_MEMBER` in `bootstrap` mode (§2.2.2), `members`, `member`, `issues`, `months`; plan-time `INVALID_REQUEST`
  carries `problems`). A trailing object's month may not start after `cutoff`. An unreadable prior
  refuses "prior release could not be read"; an inventory artifact or registry that is a symlink,
  not a regular file, or over its size cap refuses "release inventory could not be built". One attempt only.

| Rule | Behaviour |
|---|---|
| R1 verified input | Bound files are bounded regular files, re-classified by schema and member (self-consistency only: the loader hashes the bytes it read, no recorded hash is compared); assembly validates prior, cutoff, objects and inventory; in `carry` mode an incomplete or extra member set refuses; in `bootstrap` mode a bound member outside the catalog's states refuses (§2.2.2) |
| R2 cache | None; bound files, prior catalog and inventory are re-read each call |
| R3 retry | One attempt (`RetryPolicy("none", 1)`); a failed job is not retried |
| R4 transaction | New objects go to the shared content-addressed store; the staged manifest and catalog exist only under the new id; `deployment.mark_staging_succeeded` (the record `models_promote` requires) and then `stage_result.json` follow the light checks |
| R5 partial failure | Orphan objects, a staged manifest, or a candidate catalog written before a light-check refusal may remain; no success output, no promotion, the deployed release is unchanged. Inventory drift refuses before any object is written |
| R6 idempotency | Same inputs give the same catalog; the same `release_id` with different content refuses `ID_CONFLICT` |

### 2.2.1 Pool-nightly refresh sidecar (`Service._reconcile_pool_nightly_refresh`, #192 slice 4)

Called from `Service.tick()` beside the other sidecars (`engine/v2/ops/pool_refresh.py`); never a `GRAPH` node, and a
failure never blocks required job dispatch. It advances only the three members `phase5_state_stage` stages
(`paired_residual_pool`, `board_analog_matcher`, `trailing_pnl_cutoff`); driver pools and the chooser stay carried (§2.3). It trains at one cutoff per night: the session of the latest succeeded legacy `finality` or `features` job (the stages that keep the plan's `legacy_manifest.json` binding in snapshot mode, which the scheduled nightly uses; `score` does not). Alpha is
the shadow fill alpha 0.5 (`engine.fills.MID`, as `native_score_batch._SHADOW_FILL_ALPHA`); strategies are the default of
the `board_analog` mode (`strategies=()`: those present in the replay trades). Selection stays exact-key with no latest fallback: a
row whose cutoff is not the session cutoff still needs a release carrying it, which this does not provide.

Phases, each only after the previous succeeded: three `training` jobs (one per advancing member) bound to the session's
scoring manifest; one `phase5_state_stage` job staging a candidate release from their outputs; one `models_promote` job
guarded by `expected_previous_release_id` = the prior release. The scan for the latest scoring session covers a bounded window
of the newest nightly jobs, so a score older than that window is not found.

| Rule | Behaviour |
|---|---|
| R1 verified input | session and manifest id from the latest succeeded `finality`/`features` job (its `legacy_manifest.json` binding); prior release from the live pointer, classified with `classify_prior`: `USABLE` runs this carry cycle, `NONE`/`NO_POOL_CATALOG` start the bootstrap cycle (§2.2.2), `DAMAGED` submits nothing and reports `VALIDATION_FAILED` `PRIOR_MISMATCH` (deduped) |
| R7 serve lifetime | the scheduled serve (not the snapshot-import serve) also stays up while the latest session's pool cycle is pending: a candidate release for it is not yet deployed, or a phase job (including the promote) is still running, and no phase job has failed; a missing release root or a failed phase ends the wait, and the serve deadline still bounds it |
| R2 cache | durable job and pointer state is re-read each tick; no memo (reads are cheap indexed selects) |
| R3 retry | an existing phase key in any state is never resubmitted; a failed phase or stale-champion refusal stalls that cycle for operator recovery (#154); an incumbent changed before staging starts a new cycle under new keys; once a candidate is staged, the staged id or the guarded promote refuses and the cycle stalls |
| R4 transaction | phase N+1 is submitted only once every phase-N job has succeeded; a sidecar failure is caught and reported (deduped), never raised |
| R5 partial failure | succeeded jobs and staged objects may remain; no later phase or pointer change occurs until prerequisites succeed; before the pointer swap the prior release stays deployed, and a failure after the swap follows the §7.2 history rules |
| R6 idempotency | phase keys are stable per session, scoring manifest and prior release, so a rerun scored from a new manifest retrains; the candidate release id is `pools-{session}-{tag}` and does not include the prior release (the stage refusal carries `ID_CONFLICT` in `details` when the same candidate id is reused with different content; promotion separately checks `expected_previous_release_id`, R3); a deployed release already named so ends the cycle |

### 2.2.2 Bootstrap cycle (#1102; stage job, sidecar submission and promote guard built)

When the prior is `NONE` or `NO_POOL_CATALOG` (§2.2), the sidecar (§2.2.1) runs a bootstrap cycle and the stage job stages a full release instead of refusing: no store repair, and the cutoff is the session's data date. Later nightlies carry forward (§2.2) from it. Inputs: the `training` state `chooser_analog_pool` (one ISO cutoff; writes the `(features.chooser_analog_pool, cutoff)` artifact from the stored parquet, rows closed on/after the cutoff excluded; a missing or oversize source refuses; other bytes at an existing output are `CHECKPOINT_INCOMPATIBLE`) and the `mode: "bootstrap"` assembly below.

**Sidecar submission (slice C).** `submit_pool_nightly_refresh_if_ready` classifies the pointer's release with `classify_prior` only on the tick that would start a cycle (no `train-chooser` or `train-paired` job exists yet for the session, prior and manifest). Once a cycle's keys exist they fix its family, and the stage job re-derives the class itself, so the full prior is not re-verified every tick (a missing pointer is `NONE`; the prior id in phase keys is `none`). Phases, each only after every job of the previous one succeeded, keys `phase_key(session, phase, prior, tag)`:

1. `train-*` jobs, all bound to the session's scoring manifest, cutoff = session: `paired`, `board`, `trailing` (as §2.2.1); `driver-size`, `driver-implied_t1`, `driver-runup_move` (`driver_residual_pool:<role>` states; they take no cutoff); `chooser` (`chooser_analog_pool`); `payoff_line:STR-THRU` and `recalibration_map:STR-THRU` (the `:calibration` recipes, alpha `FILL_ALPHA`); `payoff_surface:STR-RUNUP`, `payoff_line:STR-RUNUP` and `recalibration_map:STR-RUNUP` also at the session: the native nightly enters every row at the session date, so its scored cutoff `min(as_of, entry_date)` is the session for STR-RUNUP too. Only the calibration artifacts (`payoff_artifact`, `recalibration_artifact`) are bound to the stage; a recipe job's bookkeeping files (`job.json`, `summary.json`, `COMPLETE.json`, the membership and label receipts) are not.
2. One `phase5_state_stage` with `mode: "bootstrap"`, `prior_release_id` the pointer id (`None` for `NONE`), `state_files` = every training output of phase 1 (plain `training/*.json`, none duplicated), and `bootstrap` = `{tier4_dir, panel_catalog, panel_snapshot_id, panel_artifact_root}` read from the environment variables `BOOTSTRAP_TIER4_DIR`, `BOOTSTRAP_PANEL_CATALOG`, `BOOTSTRAP_PANEL_SNAPSHOT_ID` and `BOOTSTRAP_PANEL_ARTIFACT_ROOT`. If any is unset or blank the cycle submits nothing and reports `VALIDATION_FAILED` ("bootstrap panel/tier4 inputs are not configured"), deduped, before phase 1 (no training is spent on a cycle that cannot stage).
3. One `models_promote`: `expected_previous_release_id` = the pointer id for `NO_POOL_CATALOG`; for `NONE` the plan sets `require_empty_store` true (new `PromoteParameters` field, default false; `deployment.promote(..., require_empty_store=False)`), and `ConcurrentPromote` is raised when the pointer exists, unless it already names the release (the same no-op rule as the expected-incumbent guard). The guard is rejected in a plan that also sets `expected_previous_release_id`.

| Sidecar condition | Outcome |
|---|---|
| Prior `DAMAGED`; pointer unreadable | nothing submitted; `VALIDATION_FAILED` reported (deduped), backoff as §2.2.1 |
| A phase-1 job fails or is cancelled/blocked | later phases never submitted; the cycle stalls for operator recovery (#154, R3) |
| Stage refuses (`BINDINGS_DIFFER`, `EMPTY_MEMBER`, registry drift, ...) | promote never submitted; stalls |
| Promote refuses `ConcurrentPromote` (incumbent changed, or the store was no longer empty) | cycle stalls; the next session starts new keys against the new prior |
| Phase-1 submit partly done when the tick fails | the next tick submits only the missing keys (keys are never resubmitted) |

The sidecar adds no cache and no retry beyond §2.2.1 R2/R3; it logs the prior class and reason once per cycle and each phase submit with counts to stderr. Cost: phase 1 trains every pool-type member, so it is expected to be heavy and runs through the scheduler's normal `experiment_heavy` admission (the supervisor runs no extra job); unmeasured until the first live cycle.

The pointer is re-read before each phase submit; if its class or id changed since the cycle's earlier jobs were keyed, the keys no longer match and a new cycle starts under the new prior id (an old cycle's jobs are never resubmitted; `pool_cycle_pending` counts the bootstrap cycle's jobs the same way).

**Standard tool check.** No third-party tool covers this (champion registry, content-addressed objects and catalog are repo-specific). The assembly reuses `tools/phase5_prepare_release.py` (`model_release`, `build_states`, `write_release`, the payload readers) from `tools/phase5_state_bootstrap.py`; the bound-file classification by schema, the `candidate_only` publication and the artifact loaders' `load_bytes` are new.

**Parameters (bootstrap).** The §2.2 parameters plus `bootstrap`: `{tier4_dir, panel_catalog, panel_snapshot_id, panel_artifact_root}`, all non-empty strings, present exactly when `mode` is `bootstrap` (otherwise plan-time `INVALID_REQUEST`; a carry plan's parameters are unchanged). `state_files` are plain `.json` names of every bound input: frozen states (driver residual pools, paired, board, trailing, chooser; classified by their own schema as `frozen_state_payloads` does) and payoff-line, payoff-surface and recalibration-map artifacts (classified by `schema_version`). The panel identity is the pinned panel copy (`verify_panel_copy`), never `auto`; `tier4_dir` is read, never written, and only folds whose snapshot matches that identity are taken. The admissible table is the static `default_admissible_table()`. The legacy chooser parquet is never read.

**Assembly invariants** (after `classify_prior`: `DAMAGED` refuses `PRIOR_MISMATCH`, `USABLE` refuses `PRIOR_USABLE`, both before any write):

- Binding equality (`NO_POOL_CATALOG` only): the candidate's bindings equal the incumbent's staged bindings (binding id, model id, role, strategy, clock, adapter, feature order, output names, member content hashes) and the inventory `deployment_id` equals the incumbent's; any difference refuses `BINDINGS_DIFFER` (a model change takes the §2.3 acceptance path). `NONE` has no incumbent to compare with.
- Legacy-trained Tier-4 serving folds enter as `tier4_folds:*` members, read only, never retrained or relabeled native. Every catalog state must be `STAGED`; a `MISSING` or `PENDING` one refuses `EMPTY_MEMBER` with the member ids. Paired, board and chooser objects carry `cutoff` as the last name segment and no trailing month starts after it; calibration folds keep their own key cutoff.
- The catalog is written under the candidate id only (`candidate_only`): no root `phase5_release.json` is written or preserved and the incumbent's directory is untouched. `sources` = `pool_cutoff` = `cutoff`, `carried_from` null (no prior catalog carried, not that the data is native), plus the pinned panel identity.
- The candidate must read back `USABLE` (`classify_prior`) and pass `light_checks` before `mark_staging_succeeded`. `DEPLOYED` is never touched.

| Condition | Outcome (`VALIDATION_FAILED`, fixed message) |
|---|---|
| Prior `DAMAGED` / `USABLE`; prior `NONE` while the pointer is unreadable or a release is deployed | `details.code` `PRIOR_MISMATCH` / `PRIOR_USABLE`; `PRIOR_MISMATCH` |
| Registry drift, inventory unreadable; every other refusal below marked "message only" | fixed message, `details` as §2.2 (`issues` for drift, `members`, `member`); nothing written unless stated |
| Bindings differ from the incumbent's | `BINDINGS_DIFFER`; nothing written |
| A catalog state not `STAGED` | `EMPTY_MEMBER`, `details.members`; nothing written |
| A bound file or model artifact unreadable, oversize, unclassifiable, duplicating an object, or naming a member outside the catalog's states; a pool object cutoff, or a trailing object month after the session cutoff | message only; nothing written |
| Champion registry or Tier-4 producer recipe unreadable or disagreeing with the registered recipe (`RegistryError`, `Tier4Error`) | message only; nothing written |
| The staging-success record cannot be written after a successful read-back | "staging success could not be recorded"; the candidate stays staged and unpromotable until a rerun |
| The staging-success record cannot be restored after an id conflict | "candidate release could not be published" (no `ID_CONFLICT` code) |
| Pinned panel unverifiable, or a fold or policy or preflight disagrees with it (the `PrepareRefused` text is not echoed) | message only; no catalog written |
| Candidate catalog present with different content or unreadable (`CatalogConflict`), or the candidate id already staged with different content (`RELEASE_ID_REUSED`, `MANIFEST_UNREADABLE`); the existing candidate and its staging-success record are preserved | `ID_CONFLICT` |
| Candidate does not read back `USABLE` | "candidate release failed the light checks" |

| Rule | Behaviour |
|---|---|
| R1 verified input | prior class re-derived in the job; registry inventory, panel copy, preflights and every bound file re-verified; the scoring manifest is pinned as in §2.2.1 |
| R2 cache | no stage-level cache; pointer, catalog, panel and bound files are re-read each call (the artifact loaders keep their per-loader content-hash cache, and `load_bytes` verifies the caller's bytes before consulting it) |
| R3 retry | one attempt; a refusal is terminal for the cycle until an operator acts (#154) |
| R4 transaction | objects and the staged manifest are content-addressed and idempotent; the candidate catalog is written last, atomically, under the pointer write lease; `DEPLOYED` is untouched |
| R5 partial failure | orphan objects, a staged manifest without a catalog, or (after a failed read-back or staging-success write) a published candidate catalog may remain, never promoted; the incumbent stays deployed, so STR-THRU rows without the required exact analog pool refuse with `MODEL_NOT_READY` (#681), while model simulation fields remain unavailable; STR-RUNUP stays `MODEL_NOT_READY` (#782) until a bootstrap promotes |
| R6 idempotency | same inputs give the same catalog (a rerun republishes the staging-success record); a different catalog under the same id is `ID_CONFLICT` |

**Progress logging.** Stage start and end with elapsed time, member and object counts and RSS go to stderr; the prior class and reason are logged once.

**Cost.** Unmeasured. The job writes the objects and verifies the pinned panel copy (it hashes the whole panel object) and reads every model artifact and Tier-4 fold, so it is not yet classified light or heavy; the supervisor runs the first live one.

### 2.3 Monthly retrain (design, cutover PR-13b, #805; slices 2 and 3 built: due decision, planned chain stub, sidecar gate, catalog and stage-time assembly; retrain execution remains unimplemented)

Binding user decisions D1-D10 (2026-10-25, recorded in #805). Once per calendar month the nightly refreshes the objects that change only on a retrain: the three driver residual pools and the chooser analog pool (and, later, champion/gate models). The retrain is a **planned early step of the nightly** (D8), not a `Service.tick()` sidecar: a chain of ordinary job kinds that the nightly plan includes when a month is due, ahead of scoring. Plan shape: `refresh` → `restage` (always planned) → train → stage → accept → check → promote → `score` (`score` depends on the last chain job); the chain jobs exist only when a month is due. Split 1 implements and registers the `retrain_check` worker for shadow/smoke jobs, but the nightly planner does not submit it yet; production has no caller until later chain wiring. The ops doc ("Current nightly orchestration boundary") points here.

| Decision | Contract consequence |
|---|---|
| D1 plumbing first | The first enabled retrain is pool-only: driver pools and chooser regenerated; champion/gate bindings stay equal to the base's. Model-changing retraining is a later slice, wired only after the acceptance check exists |
| D2 check first, auto-promote | The acceptance and non-regression check (slice 1) lands before any model-changing retrain is wired. A passing candidate is promoted automatically from day one, with no operator hold; a failing one fails the nightly |
| D3 no rollback | Nothing new is built on `rollback`; the primitive (§7.2) is unchanged. A bad live model is fixed forward by a new release, or its strategy is paused by the operator rather than served from an older release. The former forward-rollback slice is dropped |
| D4 qualification | The qualification count continues across a model-changing release: no counter is keyed on or reset by `release_id` or the model hashes (checked in slice 8) |
| D5 latest month only | Older due months get a durable `superseded:{M1}` marker naming `M` (live scoring only) |
| D6 failure fails the nightly | A failed retrain step fails the nightly and scoring does not run on the incumbent. No deadline: the step runs to completion |
| D7 no holdouts in production training | See "Isolation guard" |
| D8 orchestration | The nightly plans the retrain as an early step (plan shape above); no sidecar races it |
| D9 trigger | First nightly after month end, by date |
| D10 restage | See "Release contract restage" |

**Objects.**

| Object | Nightly (§2.2.1) | Monthly |
|---|---|---|
| Board analog, paired pools, trailing cutoff | Advance to the settled-event cutoff | Unchanged (the nightly sidecar carries them from the retrained release) |
| Three driver residual pools | Carried | Regenerated at `retrain_cutoff`; content changes only if inputs change |
| Chooser analog pool and row | Carried | Advances to the exclusive cutoff `M_end + 1 day` (keyed `(pool_id, cutoff)`) |
| Champion and gate models | Carried | Carried (D1); retrain is a later slice |

**Trigger (D9).** For session date `S`, `M` is the latest calendar month with `M_end < S` (so the first nightly after month end, deterministic, no wait for settlement lag); the live release is therefore predictable from the date. `M` is due when `M_end + 1 day` is strictly after the base release's `sources.retrain_cutoff` (a base without one: only the latest completed month is considered) and no durable record (success or `superseded`) exists. The decision is a pure function of `S`, the base catalog and durable job state, evaluated when the nightly plan is built; not due means no retrain jobs are planned and nothing is skipped silently: the plan records `retrain: not_due` with the reason. Every older due month gets a `superseded:{M1}` marker once.

**Retrained set and causality (D9, D7).** Training reads only the pinned snapshot. An event is admitted when its event date and its outcome-settlement date are both on or before `M_end` (data dates, never `computed_at`); an event whose outcome had not settled by `M_end` is excluded from `M` and enters the next month's retrain. The stored cutoff is exclusive, `M_end + 1 day` (rows dated on or before `M_end`, matching the pool builders' refusal of rows on or after their cutoff in `residual_artifact.py`); catalog `sources.retrain_cutoff` (new, one ISO date not after `pool_cutoff`) names it. Retrained members carry `retrain_cutoff`, nightly-advanced members the session cutoff, carried members their prior cutoff; this changes the §2.1 `CUTOFF_MISMATCH` rule. The catalog advances when `pool_cutoff` or `retrain_cutoff` advances; whenever a prior `retrain_cutoff` exists, the new one must itself strictly advance it (else `VALIDATION_FAILED`, `details.code` `CUTOFF_NOT_ADVANCED`). The first retrain has no prior `retrain_cutoff` and skips that comparison. The settled-by-`M_end` filter is a new admission rule in the retrain job, tested with an event settled one day after `M_end`.

**Isolation guard (D7).** No holdout is applied in production training, and no information flows from live scoring to experiments: no shared weights, pools, fitted objects or release artifacts. The guard is a static check (`checks/live_experiment_isolation.py`, a CI check; slice 4, my proposal for its shape) over import edges and path roots, in both directions: (a) production training/retrain/release modules (`engine/v2/models`, the retrain job, `tools/phase5_*`) may not import `engine.v2.foundation.experiment_holdouts`, `final_holdout` or `engine.v2.research`; (b) `engine.v2.research` and the experiment runners may not import `engine.v2.models` release, deployment or catalog modules, nor name the release store, `MODEL_RELEASE_ROOT`, the deployment pointer, the training output root of the retrain job, or live ledger rows. The retrain output records `sources.experiment_inputs: "none"`; the staged catalog is the only artifact and experiments cannot resolve it. Both lists live in the check as literals, so adding a path root to production without extending the check is a test failure (the check enumerates `production_release_root` and `production_deployment_root` callers).

**Release contract restage (D10).** Each nightly (not only monthly) the plan includes a `restage` job, first in the chain. It compares the live release with the current release contract (for example the `serving_fold` descriptors, #1208; the preflight is `serving_fold_preflight`, §3). A conforming release: success, no-op. A failing one: an automatic metadata-only restage under candidate id `restage-{contract_version}-{live_release_id}`: the same fitted objects (every member `content_hash` unchanged), new manifest metadata, then the same stage, light checks, release-scope `phase5_acceptance` and guarded promote as any candidate. A restage needing data the stored objects lack refuses `VALIDATION_FAILED`, `details.code` `RESTAGE_NEEDS_RETRAIN`: the nightly fails and a retrain (a code or data fix, then a release) is required; nothing is guessed.

**Promotion.** The monthly stage worker reads the deployed base when staging begins, with the catalog opened read-only; a missing, unreadable or malformed base refuses `VALIDATION_FAILED` (`details.code` `PRIOR_MISMATCH`) and never falls back to the plan-time base. It records that base as `sources.carried_from` and promotion guards against that same id. Because the retrain precedes the nightly pool sidecar (the sidecar submits nothing for a session while that session's retrain or restage job is nonterminal or failed) no routine promote can race it, so a `ConcurrentPromote` here is an unexpected failure: terminal for the session, no second candidate. A pool-only candidate passes light checks and release-scope `phase5_acceptance` (the stage job publishes the staging-success record after its light checks, §2.2; acceptance's `mark_staging_succeeded` is the other writer, §7.2). A model-changing candidate (bindings compared binding by binding; the semantic hash cannot say, since it includes `release_id`) also needs the non-regression check below; its receipt binds the check candidate, release candidate and incumbent hashes, and promotion requires that receipt naming the promoted hash plus the staging-success record. A failure before the pointer swap leaves the incumbent deployed, the candidate staged, and a failure receipt. `DEPLOYED` commits before the history append, so a failure there leaves the candidate live with the recorded failure (§7.2 history rules).

**Non-regression check (slice 1).** For a model-changing candidate: month `C = M - 6` (kept from the earlier design, where it was the newest month outside the experiment holdout; under D7 it only guarantees settled outcomes, my judgement). The check reads no holdout module (isolation guard (a)). A check candidate and check incumbent, each refit at `C_start - 1` with every pool and the trailing cutoff pinned to that date, score the same events of `C`. It measures rank and paired-difference consistency, not a magnitude cutoff (consistency over thresholds, as in the promotion-evidence standard). A pool-only candidate has equal bindings, so the check is a recorded no-op success. In this slice, `retrain_check` consumes a job-bound `phase5_acceptance.v1.0` report and verifies its evidence hash, candidate id and catalog hash; success requires release acceptance plus byte-identical model binding metadata and object bytes. A model change refuses `ACCEPTANCE_RULE_UNDEFINED`; its pass rule is an open user decision for split 9, with no inferred rank or paired-difference thresholds. R1 re-reads and verifies evidence, catalogs and bounded no-follow model files; R2 has no cache; R3 is one attempt (`RetryPolicy("none", 1)`); R4 writes only an attempt-private receipt; R5 refusal leaves the candidate staged and incumbent deployed and writes no receipt; R6 binds both release hashes and the evidence hash in a repeatable receipt.

**Failure semantics per step.** Every failed step makes its job `FAILED`, fails the nightly, and blocks `score` (D6). A key in any state is never resubmitted in the same session (#154). The next session retries: `M` stays due until a durable success or `superseded` marker, and step keys include session, `M` and the base release id, so a fix-forward is picked up with no operator edit. A later due month supersedes `M` (D5), but the nightly has been failing every night until then, so the failure is not hidden.

| Step | Failure | Outcome |
|---|---|---|
| `restage` | Contract check unreadable, `RESTAGE_NEEDS_RETRAIN`, stage or acceptance refusal | Job fails; nightly fails; live release unchanged |
| train (driver pools, chooser) | Input unverifiable, cutoff past the session's data, unsettled-event filter refuses, `INPUT_CHANGED` | Job fails; later steps never submitted |
| stage | `PRIOR_MISMATCH` (base drifted since training), `BINDINGS_DIFFER` on a pool-only retrain, `CUTOFF_NOT_ADVANCED`, `ID_CONFLICT`, light-check failure | Job fails; candidate never promotable (no staging-success record) |
| accept | Acceptance fails or receipt hash differs from the candidate | Fails; candidate staged, incumbent deployed |
| check (split 1) | Invalid worker parameters (`INVALID_REQUEST`, detail `RETRAIN_CHECK_PARAMETERS_INVALID`); missing staged manifest (`VALIDATION_FAILED`, detail `RELEASE_NOT_STAGED`); candidate acceptance-evidence mismatch (`INPUT_CHANGED`); accepted-evidence, staged-manifest-hash, or model-object verification failure (`VALIDATION_FAILED`); model-changing candidate has no decided pass rule (`ACCEPTANCE_RULE_UNDEFINED`) | Job fails without a check receipt; candidate stays staged and incumbent remains deployed |
| promote | `ConcurrentPromote`, receipt/guard mismatch | Fails; pointer and history unchanged |
| any step, crash | Orphan objects, staged manifest | Never promoted; a rerun with the same inputs gives the same catalog |
| nightly, slices not all built | See "Rollout" | Fails loudly with `RETRAIN_UNAVAILABLE` |

| R | Retrain training | Stage | Accept / check / promote |
|---|---|---|---|
| R1 verified input | Base release, `M_end`, snapshot pin and training inputs verified before compute; a cutoff past the session's data refuses `VALIDATION_FAILED` | Output hashes, `retrain_cutoff`, member ids and base payload hashes re-verified; drift refuses `PRIOR_MISMATCH` before any write | Parameters are validated before reads; evidence, catalog, staged manifest hashes, and requested manifest release IDs are verified before receipt; base equals the guard |
| R2 cache | None; durable state each plan | None | Pointer read per swap |
| R3 retry | Failed is terminal in the session; next session retries | One candidate id per (session, `M`); no second candidate | A failed key is terminal; no alternate target |
| R4 transaction | Outputs private to the attempt, hashed after a clean exit | New candidate id; `DEPLOYED` untouched | Existing atomic pointer swap |
| R5 partial failure | Orphans may remain, never promoted | Orphans unused | Refusal leaves pointer and history unchanged |
| R6 idempotency | Step keys `retrain-{step}:{session}:{M}:{base}` | Same inputs give the same catalog; reused id with other content refuses | Re-promoting the live id is a no-op |

**Rollout (first month end 2026-10-31, Saturday).** The first affected nightly is Monday 2026-11-02. Slice 2 (the due decision and the always-planned chain with a loud unavailable guard) must be merged and live before it. Before slice 7 the always-planned `restage` job is a stub that checks nothing and records `restage: unchecked`; it does not block Nov 2 (a contract-failing live release is separately visible as the existing `MODEL_NOT_READY`). If later slices are missing on Nov 2, the due month's chain fails closed: any planned step whose implementation is not built raises `VALIDATION_FAILED`, `details.code` `RETRAIN_UNAVAILABLE` (naming the missing step), the nightly fails, and scoring does not run on the incumbent. It never records a silent skip. If even slice 2 is not live on Nov 2, nothing is detected: the nightly would score on the carried release; the supervisor must then hold or run the nightly manually until it is. A failing nightly until the chain exists is the intended D6 outcome. A pool-only chain needs slices 1-6; `restage` (7), parity marking (8) and model retraining (9) are not needed for Nov 2.

**Standard tool check.** No standard library covers this (content-addressed release objects, repo catalog, nightly DAG); the `ast` import-edge check in slice 4 follows the existing `checks/import_layers.py`.

**Split list** (code, in order; each ≤ ~400 non-test lines, handed out one at a time):

1. Acceptance and non-regression check: `retrain-check` job kind, receipt binding candidate, release and incumbent hashes, `C = M - 6` fits; a pool-only candidate returns the recorded no-op. Nothing wires it into the nightly yet (D2).
2. Due-month decision, plan integration and the guard (built): `retrain_due.due_decision`, `superseded` marker files `retrain_superseded/{M}.json`, the `retrain` plan section (`not_due` reason logged and pinned; `base` is the deployed release id or `none`; the base `retrain_cutoff` is absent until slice 3, so only the latest completed month is considered), job kind `retrain_step` (`retrain_plan.py`; keys `retrain-restage:{session}:{plan}`, `retrain-{step}:{session}:{M}:{base}:{plan}` (`{plan}` = the plan generation token, 16 hex characters of the score job id, so every same-session replan gets fresh jobs)) with a `restage` stub (`restage: unchecked`) and `RETRAIN_UNAVAILABLE` for every chain step, `score` dependent on the chain, and the pool sidecar held while a `retrain-` job of the session is nonterminal or failed (cancelled jobs do not hold it). A plan without the section plans none of it. Must land before 2026-11-02.
3. **Built:** catalog `sources.retrain_cutoff`, stage-time base read, §2.1 advance rule, and the monthly accepted member set in `classify`/`prior_carry_set` (driver pools and chooser supplied).
4. Isolation guard `checks/live_experiment_isolation.py` (D7), before any training code lands.
5. Driver pool and chooser regeneration with the settled-by-`M_end` admission filter and `experiment_inputs: "none"`.
6. Chain wiring: stage, accept, check and promote steps behind the nightly node, replacing the stub; end-to-end pool-only retrain.
7. `restage` job (D10): contract comparison, metadata-only restage, `RESTAGE_NEEDS_RETRAIN`.
8. Parity release-id recording (`native_parity_report.py`) with a `release_changed` marker, and the D4 qualification check (the count is not keyed on release).
9. Later, post-D1: champion/gate retrain kind (`stage_release` of retrained models; inventory check accepts exactly the retrained bindings), wired only after slices 1 and 6 exist.

## 3. Inputs

**Forward serving-fold contract.** Offline descriptor authoring is described below; verified loading is defined below. Size-fold selection is implemented in `engine/v2/scoring/gate_fold_selection.py`; `engine.v2.ops.native_score_batch` selects the exact causal fold for each STR-THRU row and supplies it to the gate. A missing exact fold or selected-fold verification/load failure refuses that row with `MODEL_NOT_READY`; a malformed release catalog or descriptor fails release resolution before row assembly. Selection never falls back to another month or the champion. Monthly folds are separate from the full-refit champion bindings. A `ServingFoldRef` identifies one release-owned `tier4_folds:size` estimator object, its model, feature order, output, month start, full Tier-3 panel byte hash and declared interval policy.
Its held-out prediction/residual arrays belong to that same object. Offline release preparation records this metadata in the hashed state catalog; a read-only loader verifies it against bounded, hash-checked bytes. Existing catalogs without this declaration require offline restaging, not a nightly fit or a guessed interval policy. Verified loading covers size folds and the chooser producer artifacts below; this alone admits neither training sources nor a nightly chooser declaration.

Selection uses the exact earlier of the event month and decision month, and the registered size model and feature order. It never chooses the newest available fold or substitutes a full-refit model. The panel hash comes from the already-pinned snapshot's single original `feature_panel` object; an unsupported multi-fragment layout is refused. The scoring selector (`gate_fold_selection.py`) applies the month rule and the release-staged descriptor checks only; comparing the descriptor's panel hash with the pinned snapshot is not done by the selector.
Snapshot ID, dataset version and panel byte hash remain distinct identities. Selection returns a separate content-addressed, single-binding inference view linked to its parent release and descriptor; it does not mutate or promote the champion `ModelRelease`. The existing `tier4-serving-fold.v1` adapter executes that binding through `FrozenInference`; no second prediction engine is added.

| Forward fold condition | Outcome |
|---|---|
| Exact model/month match, verified metadata and pool | Return immutable selected fold and provenance |
| Missing, future or incompatible exact fold | Per-row `MODEL_NOT_READY`; no older-model or training fallback |
| Selected fold bytes are corrupt, oversized, escaped-path or fail the loader's header/descriptor checks | Per-row `MODEL_NOT_READY` before inference; redact filesystem details |
| Old cache lacks held-out pool or declared interval policy | Refuse; never rebuild the pool during scoring |

**Offline descriptor authoring.** Immutable `serving_folds.ServingFoldDescriptor`
records parent release/model-manifest identity, model, ordered features, output/clock,
exact month, full panel SHA-256, estimator member, embedded pool fields/count and
explicit interval policy. The enclosing catalog hash binds catalog identity
without a self-reference inside the descriptor.
`tools.phase5_prepare_release` obtains the registered producer's `SizeFoldPolicy`;
`serving_fold_preflight` (this package) validates cache bytes before catalog publication, without fitting/inference.
Serving-fold writers (`tier4.serving_model`, Phase-4 build/backfill) use binary handles and explicit `compress=0`; filename suffixes cannot enable compression.
Release preparation (`--tier4-dir` and raw-byte APIs) requires trusted uncompressed inputs from supported writers; it does not refuse arbitrary compressed inputs before decoding.
Hashes verify bytes, not provenance. The serialized byte cap is not a decoded-memory bound.
Champion registry artifacts retain their separate `compress=3` format.
Retained fold-parent checks reuse deployment's semantic release identity:
binding/member collection order and storage paths may differ on retry;
feature and output order, and every other identity field, remain significant.
All Tier-4 collector roles and chooser/pinned-size destination reads admit
regular files of at most 32 MiB with final-component `O_NOFOLLOW`; final symlinks
refuse. Parent directories remain operator-trusted, not traversed no-follow.
`read_bounded_state(path, limit, no_follow=False)` preserves frozen-loader defaults;
fold/publication callers opt in. It admits nonnegative limits below `sys.maxsize`;
invalid limits/nonregular/oversize raise coded `BoundedStateReadError`; I/O errors propagate.
Absent directories/unmatched files remain empty. Reader/chooser mismatches are unprefixed;
pinned COPY verification and pinned-size byte mismatches use `panel_copy_refused:`.

| Offline authoring condition | Outcome |
|---|---|
| Explicit policy and coherent size cache | Emit descriptor covered by the catalog self-hash; repeat inputs yield identical metadata |
| Header/model/features/full panel/month disagreement, malformed pool or oversized bytes | Refuse publication (before decode for oversize); never infer metadata from a filename prefix |
| No explicit policy in a programmatic legacy caller | Preserve the existing catalog shape; it cannot supply the future fold path |
| Other fold roles without their own explicit policy | Keep existing opaque catalog objects; never reinterpret them as size descriptors |
| Collector or post-write fold recheck encounters unreadable bytes/final symlink | Fixed read refusal with I/O cause; no new catalog/pointer; collector refuses before decode/output |
| Collector or post-write fold recheck encounters a special file | Fixed regular-file refusal before content reads; no new catalog/pointer |
| Collector or post-write fold recheck exceeds 32 MiB, initially or through growth | Fixed byte-limit refusal, before content reads if initially oversized; no new catalog/pointer |
| Shared writer finds retained bytes different from, or longer than, supplied payload | Verification refusal (role-specific mismatch for declared folds); never repair/relabel; no new catalog/pointer |
| Retry with an unchanged corrupt address | Refuse again without changing the object, catalog or pointer |
| Partial write | No pointer changes; staging may leave unreferenced files; no automatic store repair |
| Cache | Authoring validates each supplied object anew; no process-global descriptor cache |

**Chooser producer fold authoring.** `ChooserFoldPolicy` and `ChooserServingFoldDescriptor` declare `implied_t1`/`runup_move` under exact `tier4_folds:<role>` members. The descriptor binds the parent release, full panel hash, ordered features, clock, whole-cache estimator object (not the champion member), pool count and decile/floor policy. It records both the champion binding output and raw D14 chooser column; runup output is never day-scaled here. Standard release preparation obtains each present role's policy from its registered producer and validates the cache against that role's unique wildcard champion. Size descriptors keep their existing schema.
The chooser descriptor also has an optional `training_cutoff` day. The scoring consumer requires it to be canonical, strictly before the decision session, and equal to the cutoff embedded in the hash-verified estimator metadata. It refuses missing or mismatched values before using fold state. Current authoring only carries the value when the producer artifact supplies it; legacy folds without it are not consumed. The catalog and object hashes bind the declaration but do not themselves prove source finality or training membership.
The descriptor hash covers its canonical typed document, including parent, role and estimator identity.
Without pinned COPY, the panel digest is caller-asserted; preparation does not verify panel bytes.

| Chooser fold authoring condition | Outcome |
|---|---|
| Explicit coherent role policy and bounded cache | Catalog-hashed descriptor; no fitting or prediction |
| Unsupported role/output, mismatched policy/header/binding, malformed pool or oversized cache | Refuse preparation; no substitution by size or another role |
| Invalid chooser role/features/panel/interval policy/floor or descriptor month/count/parent hash/clock/output/object (constructor/semantic checks) | Typed `DocumentError`; decode preserves nested paths |
| Unsupported policy-map role or wrong policy type | Fixed `PrepareRefused`; distinguish unsupported role, invalid type and role/member disagreement |
| Registered chooser producer rejects recipe features/target | Fixed `PrepareRefused` with original cause; CLI exits 2 without a traceback |
| Pinned COPY context | Full panel hash must match; preflight precedes output effects and destination bytes must retain the validated hash |
| No policy on a programmatic caller | Preserve opaque catalog objects; no chooser fold declaration is invented |
| Shared fold-reader failure or unchanged corrupt destination | Same refusal outcomes as the offline-authoring table above |
| Destination bytes differ from preflight | Unprefixed chooser mismatch refusal; never repair or relabel an existing object |
| Cache/partial write | Validate anew; no descriptor cache or pointer changes; staging may leave unreferenced files; no automatic store repair |

Typed decoding may refuse earlier with `BAD_ENUM` for invalid role literals or `BAD_TYPE` for invalid field types.
Zero pool counts authorize no runtime selection or activation; COPY consistency does not prove training-source availability or finality.
The Phase-5 shared object writer owns durable create-only publication and typed
I/O failures ([contract](../../../checks/ARCHITECTURE.md)); fold consumers retain byte verification.

**Pinned offline COPY authoring.** `tools.phase5_prepare_release` accepts an
explicit committed snapshot/catalog and artifact-store context. Existing size
descriptors are checked against its one verified original `feature_panel`
object, full byte hash and registered release binding/policy. Snapshot, dataset
version, object and hash remain distinct identities. The manifest carries an
explicit pinned COPY claim; legacy unpinned callers cannot emit that claim.
Preflight and staging consume the same immutable bounded cache bytes. Publication
verifies stored cache bytes against the preflight hash and refuses disagreement;
panel identity is reverified at authoring. Snapshot, cache and policy checks
precede output creation, plan writing, incumbent copy and staging. No fitting,
pool recreation or live-panel fallback occurs in pinned mode.

| Pinned authoring condition | Outcome |
|---|---|
| Verified original panel and matching bounded size cache/policy | Author COPY-consistent descriptor with separate pinned identities |
| Missing/substituted snapshot/object, unsupported layout, corrupt input bytes or policy/header/model/features/output/clock/month disagreement | Refuse before output effects or staging |
| Published destination bytes disagree with the preflight hash | Refuse descriptor/catalog publication; never repair or relabel |
| Equivalent-row rewrite with different bytes | Cannot reuse the old full-hash fold declaration |
| Complete preparation or corrected-source REBUILD assertion without genuine producer evidence | Refuse stronger claim; COPY remains byte/policy consistency only |
| No pinned context | Preserve legacy behavior without a pinned claim |

Cache headers, self-hashed caller assertions and import receipts do not prove
actual training reads. Complete preparation evidence requires independently
verified producer/source-read/output causality; current imported caches lack it.
Training frame-value, membership and label receipts do not supply that proof.

**Verified serving-fold ownership.** `load_serving_fold` validates one
descriptor against a verified parent manifest and catalog identity, returning
a `ServingFoldRef` with immutable metadata/pools, its shared `FrozenInference`
owner and an ephemeral single-binding inference view. The view identity binds
parent, catalog and descriptor; it is never staged or promoted. Size-fold catalog
indexing and causal selection live in `engine/v2/scoring/gate_fold_selection.py`
(see the scoring ARCHITECTURE). Loaders are read-only:
no writes/transactions/retries. Refused calls return no reference but may retain a
hash-verified decoded artifact in the shared cache; each call still validates its
metadata and descriptor. Identical verified inputs produce identical references.

`load_chooser_serving_fold` prepares the separately typed implied-t1/runup-move
artifact from its exact `tier4_folds:<role>` catalog member. It rechecks catalog
schema/self-hash/release/deployment, unique membership, descriptor/object identity,
and the matching wildcard parent binding before decoding. The returned immutable
`ChooserServingFoldRef` preserves the raw D14 producer output and paired held-out
interval arrays; these are not the chooser k-NN population or driver residual state.
It makes no event-month selection or source-ready declaration. Genuine
training-source completion/finality, coordinator publication and per-request causal
bounds are required for admission. Artifact headers/descriptors alone do not
provide that proof. Existing nightly refusals remain.

`FrozenInference.prepare_serving_fold(binding, max_member_bytes=..., expected_bytes=None)`
returns detached immutable `ServingFoldMetadata`, never cached estimator/header objects.
Both fold loaders use it; chooser catalog size is checked against the already-read
buffers, including warm calls. The required byte limit is positive. The resolved
path is checked for containment; bounded reads open the original path with
final-component no-follow through the shared regular-file reader. Parent directories
remain operator-trusted. Missing, special, final-component symlinked, oversized or
changed objects refuse before decode, cold and warm. Direct preparation normalizes bounded I/O and regular-file/size
refusals to path-free `AdapterError`. During `infer()`, bindings without an established
cache-entry byte ceiling, including legacy fold bindings and champions, retain unbounded
verification; preparation always supplies a limit. An established ceiling applies to subsequent inference reads.

`FrozenInference` owns the only decoded-artifact cache, retained for its owner
lifetime without eviction. Its adapter retains fold metadata with the estimator,
so validation and prediction share
one decode. Each cache entry may also own a byte ceiling established by bounded
preparation; subsequent cold/warm verification cannot relax that ceiling.
Legacy entries without a ceiling retain their existing behavior. Prediction
still uses `Tier4ServingFoldAdapter` through the existing inference executors.

| Verified fold condition | Outcome |
|---|---|
| Parent/catalog/descriptor/header or embedded-pool disagreement | `MODEL_NOT_READY`; conditions share one fixed path-free refusal, no inference or fallback |
| Escaped path, missing/corrupt bytes or object above 32 MiB during fold preparation or prepared inference | Refuse before deserialization, including warm reads; unprepared inference with no established byte ceiling retains legacy unbounded verification |
| Repeated preparation and inference on the same owner/object | Reverify bytes; reuse the decoded artifact, never a second model cache |
| Missing/duplicate chooser catalog membership, wrong role/output or opaque undeclared fold | Fixed `MODEL_NOT_READY`; no champion/size/other-fold substitution |
| Source availability/finality or training-bound proof absent | No causal/source-ready claim; artifact preparation cannot bypass consuming admission |
| Different current panel bytes or multi-fragment layout | No equivalence is inferred here; selector availability/provenance contract remains a prerequisite |

- **`current_release_inventory` (`inventory.py`)**: the champion registry
  (`engine.models.registry`) and the real files under `data/models/*` —
  content hashes, never values; a symlink, non-regular or oversize file raises `ValueError`.
- **`deployment.stage_release`**: a caller-built `ModelRelease` +
  `ModelReleaseInventory` (already validated against each other by every
  training/preparation tool) plus a `{content_hash: bytes}` payload map for
  every member across every binding. The inventory is validation-only and
  is never persisted: `resolve_release` and every scoring reader return
  only the `ModelRelease`, so each `stage_release` call needs a freshly
  built, freshly validated inventory.
- **`deployment.promote`/`rollback`/`resolve_release`/`current_release`**:
  a `release_id` and this module's own `deployment/` root — they read only
  what `stage_release` and earlier promotions already wrote under that
  root, never a live panel, a request, or anything outside the store.
- **`deployment.production_release_root`**: the `MODEL_RELEASE_ROOT`
  environment variable — the store root, one level ABOVE the
  `deployment/` directory this module's own functions take as `root`.
  `production_deployment_root` (§7.4) is `production_release_root() /
  "deployment"` — what a caller of THIS module's own functions wants.
- **`deployment.restage_semantic_hash`**: the target `release_id`'s
  already-staged manifest (read once, verified under its own declared
  hash version before anything is trusted) — no payload bytes, no
  training/evaluation artifact.

## 4. Outputs

- **`stage_release`**: content-addressed objects under
  `<root>/objects/<hash>` (write-once; an existing object with the same
  hash is never rewritten) and one immutable `manifest.json` per
  `release_id` under `<root>/releases/<release_id>/`.
- **`promote`/`rollback`**: the single `DEPLOYED` pointer file (atomic
  temp-write + `fsync` + `os.replace` + `fsync(dir)`, so a crash leaves
  either the old pointer or the new one, never a partial file) and one new,
  append-only, sequence-numbered file under `<root>/history/`. Rollback
  refuses with `NoPriorRelease` before either write when history has no
  earlier incumbent.
- **`restage_semantic_hash`**: rewrites exactly one file,
  `<root>/releases/<release_id>/manifest.json`, atomically, in place. No
  other file under the release store is ever touched by this function.
- Every non-model frozen-state builder (`training/residuals.py`,
  `training/chooser_pool.py`, …) returns an immutable dataclass; the
  release/state catalog (`phase5_release.json`) is owned by
  `engine/v2/models/release_layout.py` and stored beside each release manifest.
  The release-root copy selects the last staged candidate for existing
  tooling; it is not authority for a different deployed release.

## 5. Dependencies

Imports only layers 0.0–2.0 (`contracts`, `foundation`, `data`, `features`)
per the root doc's layer table; `checks/import_layers.py` enforces this.
`deployment.py` additionally imports nothing outside this package and
`engine.v2.foundation` (`Clock`, `SystemClock`, `content_hash`,
`format_timestamp`, `from_document`, `fsync_directory`, `to_document`).
`release_layout.py` and `release_objects.py` import only this package and `engine.v2.foundation`
(`content_hash`, `ensure_directory`, `fsync_directory`); `release_objects.py` also uses
`frozen_state_io.read_bounded_state`.

**Callers** (real import edges, `from engine.v2.models import ...` /
`from engine.v2.models import deployment`, checked against `README.md`'s
`<!-- consumers: ... -->` directive):

- `engine.v2.scoring` — `gate_fold_selection.py` calls `load_serving_fold`,
  `validate_serving_fold_object` and `validate_serving_fold_descriptor` from
  `serving_fold_loader`; the verified frozen inference contract/loader for
  the model stage, the payoff-calibration artifact type, and
  `release_bindings.py`'s reads of `deployment.current_pointer`,
  `deployment._read_manifest`, `deployment._manifest_hash_matches`,
  `deployment.production_release_root`, and `deployment.MissingReleaseRoot`.
- `engine.v2.models.training` — the same artifact types and no-fit guards, plus
  closed ElasticNet and Ridge/HGB recipe validation; never imported *by* this package (one-way).
- `engine.v2.ops` — `engine/v2/ops/cli.py` (the no-fit guard for `ops
  rescore`) and `engine/v2/ops/training.py` (`deployment.promote` from the
  `models_promote` job worker, and `deployment.
  production_deployment_root`/`deployment.MissingReleaseRoot` from
  `promote_plan` — NOT `production_release_root` directly: `promote_plan`
  hands its resolved value straight to `deployment.promote`, which takes
  the `deployment/` directory itself as `root`); `engine/v2/ops/pool_refresh.py`
  (reads the `DEPLOYED` pointer through the bounded reader) and `supervisor.py`
  (`deployment.production_release_root`) for the §2.2.1 sidecar, and `nightly_trigger.py`
  (`deployment.production_release_root` for the scheduled-serve hold). Neither `nightly.py`,
  `worker.py` nor `stages.py` import this package's deployment surface
  directly.
- `engine.v2.serving` — `engine/v2/serving/operations.py` reads the
  deployment pointer read-only (`current_pointer`, `resolve_release`) to
  serve `/models/release.json`; never promotes.
- `tools/phase5_state_stage_job.py` — publishes the staging-success record
  (`mark_staging_succeeded`) for a nightly pool candidate (§2.2).
- `tools/phase5_state_bootstrap.py` — the stage job's `bootstrap` mode (§2.2.2) stages a full candidate
  through `write_release(candidate_only=True)`; `release_layout.CatalogConflict` is its `ID_CONFLICT`.
- `tools/phase5_prepare_release.py` and `tools/phase5_inventory.py` — release preparation and
  inventory; the former imports `serving_fold_preflight`, `serving_fold_checks` (fold metadata and
  its policy/header checks) and `pinned_panel` (the pinned panel check), which live in this package.
- `experiments/magnitude_search_trials.py` and `experiments/magnitude_search_execution.py`
  — `no_fit.forbid_fitting` before native trial claims and fixed-child fitting.
- `experiments/magnitude_candidate_recipe.py` consumes the closed ElasticNet/Ridge/HGB
  recipe constructors; `magnitude_candidate_fit.py` checks both legacy/v2 no-fit
  guards before recipe, column or array inspection, then delegates to the public fitter.

## 6. External systems and libraries

A local filesystem tree (the release store root —§7.4's env-configured
key names it for production). `pinned_panel.verify_panel_copy` also opens the
pinned snapshot's SQLite catalog read-only through `engine.v2.data`, and the
serving-fold preflight decodes joblib bytes (`joblib`, `numpy`). No network, no
third-party service. `hashlib.sha256` for every content hash; `os.fsync`/`os.replace`
for the pointer's crash-proof write.

## 7. Failure semantics

### 7.1 `stage_release`

| Condition | Outcome |
|---|---|
| Inventory itself is incomplete (`releases.require_complete_release`) | Refuses `ModelReleaseRefusal`, before anything else runs |
| Inventory is complete, but the inference release doesn't match it (role/strategy/clock binding, feature order, a required member kind missing) | Refuses `StagingRefused`, before any byte is written |
| Missing payload for a declared member, or a payload whose sha256 disagrees with its declared `content_hash` | Refuses `StagingRefused` (`MISSING_MEMBER_PAYLOAD` / `PAYLOAD_HASH_MISMATCH`) |
| Two inference bindings declare the same `(role, strategy_id)` | Refuses `StagingRefused` (`DUPLICATE_BINDING`) — the key is deliberately clock-independent, matching `scoring.release_bindings`'s own ambiguity key; without this a release could stage cleanly yet make every score for that role/strategy fail at read time |
| Caching | None: every call re-derives the release hash and re-checks every member from the caller's arguments |
| Same `release_id`, identical content, restaged | Idempotent no-op: returns the existing manifest byte-for-byte, including its original hash version (never silently upgraded) |
| Same `release_id`, different content | Refuses `RELEASE_ID_REUSED` |
| Same `release_id` already staged, but its manifest cannot be read or parsed | Refuses `StagingRefused` (`MANIFEST_UNREADABLE`) instead of a bare decode error (issue #207) |
| Crash mid-staging | Some content-addressed objects may be left on disk (harmless — addressed by their own hash, reused or ignored later) and no manifest, so the release is not staged |
| Every write | Atomic (temp + fsync + rename); never a half-written manifest or object |
| Idempotency | Same `(release, inventory, payloads)` always produces the same staged manifest |

### 7.2 `promote` / `rollback`

Both go through `_swap_pointer`, which re-validates the target release
independently of whatever staged it.

| Condition | Outcome |
|---|---|
| Unstaged `release_id` | Refuses `ReleaseNotStaged` |
| Staged manifest without a durable successful staging-completion record bound to that exact `release_id` and `release_hash` | Refuses typed, non-retryable `StagingNotSuccessful` before pointer/history writes; leaves the staged manifest and objects intact and creates no partial deployment state. A retry is permitted only after the staging workflow has completed all post-stage checks and published its matching success record |
| Target manifest cannot be read or parsed | Refuses `StagingRefused` (`MANIFEST_UNREADABLE`) instead of a bare decode error (issue #207), checked before the hash-version check below |
| Target's staged `release_hash_version` is not the current semantic version | Refuses `StaleReleaseHash`, checked before anything else — including re-promoting the currently live release, if it is itself stale. `restage_semantic_hash` (§7.5) is the only way to clear this; never automatic |
| Target manifest's recomputed content hash disagrees with its declared `release_hash` | Refuses `CorruptManifest`, before the pointer moves |
| Target manifest declares two bindings for the same `(role, strategy_id)` | Refuses `StagingRefused` (`DUPLICATE_BINDING`) — re-verified independent of `stage_release`'s own check, in case a release predates it or was staged by another path |
| `promote` receives `expected_previous_release_id`, its target is not already live, and the current deployed release is absent or has a different id | Refuses typed `ConcurrentPromote` (`CONCURRENT_PROMOTE`) before pointer/history writes; the expected id is compared with the current pointer read for this swap, with no retry or alternate target. If the target is already live, promotion is an idempotent no-op even when the supplied incumbent differs |
| `promote` receives `require_empty_store`, its target is not already live, and a pointer exists | Refuses typed `ConcurrentPromote` before pointer/history writes; passing both guards is a `ValueError` (plan and worker refuse it as `INVALID_REQUEST`) |
| Caching | None: `current_pointer`/the manifest are re-read from disk on every call |
| `promote` on the already-live `release_id` | No-op: returns the existing `PointerState`, never a new one |
| `rollback` with fewer than two ids left after replaying pointer history as a promote/rollback undo stack | Refuses `NoPriorRelease` |
| `rollback` target | The second-from-top id of that replayed stack (push on every `promote`, pop on every `rollback`) — never the live pointer's own `previous_release_id` field. This makes N chained rollbacks undo N chained promotions and never revisit a release just left |
| Recorded pointer history unreadable, its sequence numbers aren't the contiguous range `0..len(history)-1`, or its replayed top disagrees with what `DEPLOYED` names | `rollback` refuses `StagingRefused` (`HISTORY_UNREADABLE` / `HISTORY_SEQUENCE_GAP` / `HISTORY_INCONSISTENT`) rather than guess a target |
| Pointer write | Atomic; a crash leaves `DEPLOYED` as either the old value or the fully-written new one, never partial |
| Crash after the pointer write but before its history entry is appended | Self-healing: the next call re-derives the missing entry from the live pointer before acting, so the undo stack is never shifted |
| Repeated promote of the same id | Exactly one no-op, not two history entries |

`promote`'s optional `expected_previous_release_id` guard checks the current
pointer for callers promoting from a known incumbent. Omitting it preserves
the existing operator path; supplying it requires that the observed current
pointer still names that incumbent unless the target is already live, in which
case promotion is an idempotent no-op. This optimistic guard does not add a
cross-process lock for direct callers (#137). A refusal never changes the
pointer or history.

#### `mark_staging_succeeded`

This record is separate from the immutable model manifest. Staging alone never
writes it. `checks/phase5_acceptance.py` publishes it only after all release
subjects pass (`RELEASE_PASS` without a Phase 4 corpus, or `PASS` with one),
then writes successful evidence; the `phase5_state_stage` worker (§2.2)
publishes it for a nightly pool candidate after its light checks. A publication refusal leaves no evidence for
that attempt that claims acceptance succeeded. Preparation invalidates the
candidate record before replacing its content. An invalid, failed, or stale
record is never inferred from pointer/history entries or a manifest hash alone;
promotion and rollback refuse without matching staging success.

| Condition | Outcome |
|---|---|
| No prior status record for the candidate | `invalidate_staging_success` is a no-op; preparation continues |
| Removing a prior status record fails | Refuses `StagingRefused` (`STATUS_INVALIDATION_FAILED`) before staging the candidate manifest or writing its Phase 5 state; a deployment root or incumbent copy may remain |
| No manifest under `release_id` | `mark_staging_succeeded` refuses `ReleaseNotStaged` |
| Manifest unreadable, path identity differs, or its content hash is invalid | Refuses `StagingRefused` (`MANIFEST_UNREADABLE` / `RELEASE_ID_MISMATCH`) or `CorruptManifest`; writes no success record |
| Valid manifest; success record write fails | Refuses `StagingRefused` (`STATUS_UNWRITABLE`); no deployment pointer/history is changed and no partial success record is accepted |
| Valid manifest and atomic status write succeeds | Writes exactly `release_id`, `release_hash`, and `state: succeeded` to `staging-status.json` beside that manifest |
| Same manifest marked again | Idempotent: writes the same record |
| Manifest hash changes after marking, including `restage_semantic_hash` | Existing record is stale; complete acceptance again and record the new hash before promotion or rollback |
| Live incumbent staged before status records existed | Acceptance rehearses the candidate's promote on a scratch copy and does not rehearse rollback (#805 D3) |
| Release staged before status records existed, or incumbent with an existing invalid/failed/stale record | Remains ineligible; no automatic backfill or overwrite from a pointer/history entry or manifest hash alone, because neither proves successful staging |

### 7.3 `resolve_release` / `current_release` (read-only)

| Condition | Outcome |
|---|---|
| Unstaged `release_id` | `resolve_release` refuses `ReleaseNotStaged` |
| Staged manifest cannot be read or parsed, is a symlink or non-regular file, or exceeds 64 MiB | Both refuse `StagingRefused` (`MANIFEST_UNREADABLE`) instead of a bare decode error; a missing file is still "not staged" — `current_release` inherits this from its own call to `resolve_release` |
| Release staged under a superseded hash version (legacy) | Still resolves — replay of a score already recorded against it must never become unreplayable. Only the write path (§7.2) enforces the current hash version |
| No live pointer | `current_release` returns `None` |
| Caching, retry, transaction, partial write | None of these apply: read-only, no cache, nothing to retry, no write |
| Idempotency | Same `release_id` always resolves the same `ModelRelease` (a staged manifest is immutable once written) |

### 7.4 `production_release_root` / `production_deployment_root`

Both share one `MODEL_RELEASE_ROOT` environment variable;
`production_deployment_root` is `production_release_root() / "deployment"`.

| Condition | Outcome |
|---|---|
| `MODEL_RELEASE_ROOT` unset or blank | Refuses `MissingReleaseRoot` — no fallback to a repo-relative or other default path: this is the one config key naming "which release root is production" |
| Set | Read fresh from the environment on every call (never cached), then resolved via `Path.expanduser().resolve()` — this touches the filesystem to normalize the path (symlink resolution) and, for a relative `MODEL_RELEASE_ROOT` value, resolves it against the current working directory. The scheduled native nightly gets it from the unit's optional per-user `EnvironmentFile`; supervised runs can export it with `set -a; . <EnvironmentFile>; set +a`. Use single quotes around the complete assignment value when it contains spaces or shell metacharacters such as `$`; systemd and shell sourcing treat the quoted content literally. Values containing a single quote are unsupported in this shared-file convention |
| Retry, transaction, partial write | Not applicable: this is a read and a path computation, not a write |

### 7.5 `restage_semantic_hash`

| Condition | Outcome |
|---|---|
| Nothing staged under `release_id` | Refuses `ReleaseNotStaged` |
| Existing manifest cannot be read or parsed | Refuses `StagingRefused` (`MANIFEST_UNREADABLE`) |
| Existing manifest at this path declares a different `release_id` | Refuses `StagingRefused` (`RELEASE_ID_MISMATCH`) — checked before the hash check below |
| Existing manifest fails verification under its own declared hash version | Refuses `StagingRefused` (`RELEASE_ID_REUSED`, the same code `stage_release` uses) — a corrupt manifest is never a starting point for a rewrite |
| Already at the current semantic hash version | No-op: returns unchanged |
| Legacy (member-only) manifest | Its own verification covers `release_id` and every member's `content_hash`, not `adapter`/`feature_order`/`output_names`; the semantic hash this call writes covers all of them going forward, not retroactively |
| Write | One atomic rewrite of `manifest.json` only; nothing else under the release store is touched |
| Existing staging-success record | If the rewrite changes `release_hash`, the old record no longer matches and promotion/rollback refuse; rerun Phase 5 acceptance and record success for the new hash |
| Idempotency | Same starting manifest always produces the same rewritten manifest; a second call hits the "already at current version" no-op above |

### 7.6 Phase-5 state catalogs (staging-tool ownership)

Catalogs live at `<release_root>/deployment/releases/<release_id>/phase5_release.json`.
Release IDs are validated before path construction; preservation requires
a matching, hash-verified staged model manifest.
`release_layout.read_manifest` selects a candidate using the root copy;
production scoring instead selects the release named by `DEPLOYED`.

| Condition | Outcome |
|---|---|
| Selected release-local catalog exists | It is authoritative; schema, self-hash and release identity must validate |
| Cache | `release_layout.read_manifest` and `scoring.release_bindings._read_state_catalog` reload and revalidate catalogs on every call; no catalog data is cached |
| Release-local catalog absent | A valid legacy root catalog is usable only for its own matching release ID; reads do not migrate files |
| Local catalog corrupt or unreadable, a symlink or non-regular file, or over 64 MiB (`release_layout.read_manifest`; scoring `_read_state_catalog` likewise) | Refuse (`ReleaseLayoutError`, scoring `ModelNotReady`); never fall back to a valid root copy |
| Replacing the root candidate or copying a legacy incumbent | Preserve its valid catalog under its own release ID; validate and retain any existing local catalog over a stale root copy; refuse bad or unresolvable inputs; incumbent validation is read-only and precedes copying |
| Catalog publication interrupted | Individual files are replaced atomically; staged files may remain, but deployment pointers are untouched; retries may repeat completed writes; concurrent writers have no safety guarantee, except `carry_forward_release` calls, which serialize on `deployment/.carry_forward.lock` (§2.1) |
| Repeated staging under one release ID | Existing rewrite semantics remain; different release IDs retain separate catalogs |
| Historical catalog was already overwritten before migration | It cannot be recovered from a model manifest; rebuilding that catalog is required |

## 8. Invariants

- **A detected validation issue → typed refusal, never a silent default**
  (root doc §5), scoped to `deployment.py`'s own explicit checks: duplicate
  bindings, payload hash mismatches, a stale or corrupt manifest hash, an
  unparseable manifest, and a broken pointer history each raise a
  `DeploymentError` subclass, never a substituted value. Every call to the
  shared `_read_manifest` helper — from `stage_release`'s re-stage check,
  `_swap_pointer` (`promote`/`rollback`), `resolve_release`, and
  `restage_semantic_hash` — wraps the read in the same
  `except (OSError, ValueError)` and refuses `StagingRefused`
  (`MANIFEST_UNREADABLE`), so a corrupt or undecodable manifest.json on
  disk can never leak a bare `json.JSONDecodeError`/`DocumentError` out of
  any of them (issue #207, closed). An incomplete inventory raises
  `releases.require_complete_release`'s own `ModelReleaseRefusal`, not a
  `DeploymentError`; and `FrozenInference` (`loader.py`) reports every
  refusal it detects through a typed `InferenceResult` with
  `status=MODEL_NOT_READY`, never a raised exception.
- **A hash mismatch never falls back.** No branch anywhere in this package
  substitutes a different object, an older cached value, or a default when
  a content hash disagrees — including the pointer swap itself:
  `promote`/`rollback` recompute and check the manifest hash, not only its
  version tag, before `DEPLOYED` moves.
- **A rollback undoes exactly one promotion.** `rollback`'s target is
  derived by replaying the full pointer history as a stack, never by
  reading a single `previous_release_id` field off the live pointer — so
  chained rollbacks walk chained promotions backward and never revisit a
  release they just left.
- **A staged release covers every inventory binding exactly once.**
  `stage_release` refuses a second inference binding for a `(role,
  strategy_id)` pair already seen in the same release — the same key
  `scoring.release_bindings` resolves by, clock-independent — so it never
  accepts a binding set scoring itself would call ambiguous.
- **Never fits anything.** `RuntimeFitForbidden`/`ReadOnlyArtifact`
  (adapters/loader) and `no_fit.py`'s guard (reused by
  `engine.v2.scoring.native_payoff`) are this package's enforcement of the
  root doc's layering rule that only `engine/v2/models/training` (layer
  6.0) ever calls `.fit()`.
- **Staged content is immutable; only the pointer moves.** A staged
  manifest is written once and never mutated by a later promotion,
  rollback, or a `restage_semantic_hash` upgrade of a DIFFERENT release —
  `restage_semantic_hash` rewrites ONLY the one manifest it targets, and
  only its `release_hash`/`release_hash_version` fields.
- **Deployment requires the current hash version; replay does not.**
  `promote`/`rollback` refuse a stale-hashed release; `resolve_release`/
  `current_release` and every checker that calls `_manifest_hash_matches`
  continue to accept a verified legacy manifest — deliberately different
  rules for deliberately different questions ("is this safe to make live"
  vs. "is this the release a past score actually used").
- **Driver residual pool artifacts use `(role, model_id, fold)` keys and
  content-hashed payloads.**

## 9. Diagrams

```mermaid
stateDiagram-v2
    [*] --> Staged: stage_release()\n(RELEASE_HASH_SEMANTIC_V2 always)
    Staged --> Staged: restage_semantic_hash()\n(legacy manifest -> v2, no-op if already v2)
    Staged --> Deployed: promote()\nrefuses StaleReleaseHash\nunless hash_version == v2\nrefuses CorruptManifest\nif the manifest hash disagrees
    Deployed --> Deployed: promote() same id (no-op)
    Deployed --> Deployed: rollback()\ntarget = 2nd-from-top of the\nreplayed promote/rollback stack\nrefuses StaleReleaseHash/CorruptManifest\non the target release too
    Staged --> [*]: resolve_release()\n(read-only, by id; accepts legacy or current hash version)
    Deployed --> [*]: current_release()\n(follows DEPLOYED; None if unset)
```

```mermaid
flowchart LR
    ENV["MODEL_RELEASE_ROOT\n(environment variable)"] --> PRR["deployment.production_release_root()"]
    PRR -->|"set"| ROOT["release STORE root"]
    PRR -->|"unset/blank"| MRR["MissingReleaseRoot"]
    ROOT --> RB["release_bindings.resolve_production_release_binding()\n(adds deployment/ itself)"]
    ROOT --> PDR["deployment.production_deployment_root()\n(= root / \"deployment\")"]
    PDR --> PP["training.promote_plan()\n(only when --release-root is omitted)"]
    MRR --> RBERR["ModelNotReady('release_root', ...)"]
    MRR --> PPERR["OpsError INVALID_REQUEST"]
```

`experiments/magnitude_search_outer.py` consumes the legacy/v2 no-fit guards
before selected outer refit claims; completed evidence replay never fits.

Training `current_recipes` reads installed or alternate registry JSON through `read_bounded_state` with a 16 MiB ceiling before decoding; valid recipe identities are unchanged. `BoundedStateReadError` remains a `ValueError`; the SIZE adapter translates only this read refusal to a resource error, without fitting or writes.
