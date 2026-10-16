# Experiments architecture

## Native preregistration

`native_registration.register_native` explicitly registers one resolved arm
before evaluation; `require_native_registration` is its read-only admission
check. Both pin the current scope head once and read canonical event metadata
through the shared holdout population reader, never outcomes or prior results.
The stable registration key is experiment ID plus primary arm. Its immutable
binding includes the full resolved plan, detached spec hash, exact snapshot,
static source closure, numerical environment and declared single-thread policy,
scope, eligible event IDs and versioned holdout context. External input files and non-Python runner paths are refused.
The caller supplies the checkout root used for source fingerprinting.

Registration publishes canonical bytes through `ArtifactStore`, then reserves
the existing primary `hypotheses`/`experiment_runs` records in one catalog
transaction. Re-registering identical content returns the same identity.
The reserved arm is planned; registration does not mark a variant as tried.
Admission verifies the stored object and compares a freshly built binding;
returned documents are copies, not mutable views of registered bytes.
Registration and current-request admission do not fit models, publish reports, write the CSV ledger or emit outcome/refusal receipts.
The supervisor does not call them directly; its dispatched smoke worker reuses `verify_native_registration` for staged binding verification.
Public registration preparation/reservation and worker I/O/admission/error helpers remain in their owning modules; [README.md](README.md) lists their contracts and callers. Saved binding bytes and declared source roots are not rewritten.

Optional `final_read` metadata creates a versioned native registration that seals recipe, completeness-bar and frozen-model artifact references. Legacy registration bytes remain unchanged when omitted.
`prepare_final_holdout_request` verifies that original registration and derives an immutable request keyed by experiment ID; missing declarations or malformed request metadata refuse `INVALID_EXPERIMENT_SPEC`, and changed catalog/artifact identity refuses `EXPERIMENT_IDENTITY_CONFLICT`.
Preparation reads only registration metadata, performs no repository/head/population/result access or writes, and grants no authority. Foundation owns shared request decoding so ops and research need no peer import; later read admission must verify the referenced evidence and completeness before reading.

Optional `magnitude_training_start` binds v1.2 to a `computed_moves` spec;
years come only from `spec.folds`, validated by pure `magnitude_window`.
The metadata-only planner owns the sealed temporal plan, selected event IDs
and holdout context from one pinned snapshot; caller-supplied plans are refused.
Equivalent explicit/bulk membership gives identical bytes with other inputs fixed;
omission preserves v1.0/v1.1 bytes and identities.

Admission requires the catalog-owned original, even without `expected`, and a
complete fresh binding match. Stored integrity does not establish current
semantic admission: calendar arithmetic and fold reconstruction belong to the planner.
The closed stored schema requires canonical dates, nonempty ordered years,
fixed recipe/policy/availability declarations and snapshot-linked source identities.
Events follow `(event_date, event_id)` order; fold lists retain that order.
Registration itself has no feature/label reads, fitting, aggregate
materialization or final-read authority.

Optional `magnitude_labels_before`, together with `magnitude_training_start`,
creates v1.3 and seals a fixed `magnitude_execution.v1` contract. The explicit
evaluation-label cutoff must be a canonical day strictly after every evaluation
event; it is neither the forecast ceiling nor a training fit boundary.
The contract declares a fold-local arithmetic-mean baseline, constant intercept,
event-pooled RMSE with MAE, omitted skipped/unavailable/late training labels, and
complete finite evaluation labels available strictly before that cutoff.
Quantile calibration is explicitly unimplemented. Seed zero and empty economics
are required; this baseline has no hyperparameter or feature search declaration.
Omitting the new option preserves v1.0/v1.1/v1.2 bytes. Admission binds the original
v1.3 reservation before metadata scans; changing or omitting its execution policy
is an identity conflict. Registration reads metadata only. Only sealed v1.3
opens the intercept baseline; v1.4 below adds stored history. v1.2 remains metadata-only; none permits final reads.

`magnitude_inputs.admit_magnitude_inputs` requires this v1.3 registration with
the explicit training/evaluation clocks, then derives the union of all sealed
fold training candidates and evaluation IDs. It calls the research physical
preflight on exactly that union before any label read, returning the original
registration and detached preflight evidence. Unused selected-window events
are excluded. Admission/metadata/isolation refusals propagate with no writes,
retry or partial return. The magnitude CLI uses this admission before replay.

### Fixed magnitude baseline

`v2_prediction --target absolute_move` requires both magnitude clocks; the
default positive-move target refuses those options before store access.
`magnitude_report` validates the fixed seed-zero specification and sealed v1.3,
preflights every fold participant before labels, then qualifies each training
fold and freezes its arithmetic mean before opening that fold's evaluation labels.
All evaluation labels must be complete, finite and strictly cutoff-available;
empty training or any invalid evaluation refuses the whole assessment.
Each event retains all 5/3/1 full-session forecasts with the same intercept.
Primary RMSE pools equally weighted events, never fold RMSEs; MAE and per-horizon
metrics are also reported. Numerically scaled errors avoid square overflow.
The report binds every opened object, training omission, cutoff and sealed plan.
Schedule/source availability stays unverified reconstruction; quantile calibration,
market/history-feature model fitting, search and final evaluation are not implemented.
The existing native report/outcome/completion, ledger and replay lifecycle applies;
no-ledger runs still persist smoke evidence. Replay admits metadata but does not refit.

| Magnitude registration condition | Outcome |
|---|---|
| Invalid window, non-`computed_moves` source, or combined `final_read` | `INVALID_EXPERIMENT_SPEC` before repository access. |
| Execution cutoff without a magnitude window, malformed cutoff, nonzero seed or nonempty economics | `INVALID_EXPERIMENT_SPEC` before repository access. |
| Execution cutoff on/before any evaluation event | `INVALID_EXPERIMENT_SPEC` after metadata planning but before publication or reservation. |
| Missing admission reservation | `INVALID_EXPERIMENT_SPEC`; no metadata scan or implicit registration. |
| Changed/omitted mode, conflicting expected identity, or malformed/cross-context stored plan | `EXPERIMENT_IDENTITY_CONFLICT`; prior evidence remains unchanged. |
| On admission (`require_native_registration`): mode/request/source/environment/scope/snapshot/month drift or rehashed malformed/permuted metadata | Refusal before population/calendar scans. |
| Planner admission, calendar or empty-fold failure | Shared typed error propagates; no partial registration. |
| v1.2/v1.3 final-request preparation | `INVALID_EXPERIMENT_SPEC`; final declarations remain exclusive to v1.1. |

Existing publication/transaction/retry semantics above apply unchanged.

### Stored-history mean model (v1.4)

`--target absolute_move --magnitude-model history_mean` additionally requires
`--magnitude-feature-snapshot-id` and `--magnitude-history-start` on register and run.
Both magnitude clocks remain mandatory. Omitted model or explicit `pooled_mean`
uses v1.3; positive-move and pooled-mean modes reject history options before store access.
The library selects v1.4 by the same two history keyword arguments; both are required,
with seed zero, empty economics and runner `experiments/v2_prediction.py`.

The closed registration adds `magnitude_history_request` (exact feature snapshot ID,
canonical history start) and `magnitude_history` (binding version, full feature pin,
supported producer context). `magnitude_history_execution.v1` seals two model columns,
the empty-history fallback, evaluation cutoff and equal event/horizon RMSE policy.
The supported context is freshly derived from the original source, calendar, population,
holdout versions, forecast ceilings, history recipe and installed producer fingerprint;
its plan/source must exactly match the outer registration. Caller-supplied contexts fail.
Old wire policies remain unchanged for identical bound inputs; legitimate runner-source
changes still require new preregistration. Consumer wiring does not change producer code.

`magnitude_history_inputs` bridges research metadata admission into registration.
Admission checks mode/request drift before population scans, then derived context,
exact committed feature receipt/contract and complete object isolation before any
feature or outcome payload. Missing/changed feature identity refuses
`EXPERIMENT_IDENTITY_CONFLICT` before attempt/replay; protected isolation refusals retain
`HOLDOUT_ACCESS_DENIED`. There is no implicit build, repair, alternative pin or retry.

`magnitude_history_report` uses the original stored-only reader; it never recomputes history.
Each fold qualifies training labels and freezes their arithmetic mean, then fixes every
evaluation forecast before opening evaluation labels. Positive `n_prior` uses stored
`mean_prior_abs_move`, including zero; only empty history uses the fold mean.
All three forecasts per event are required. RMSE pools event/horizon squared errors,
giving each complete event equal weight; MAE and genuine per-horizon metrics accompany it.
The report distinguishes events from forecasts and binds both snapshots, stored input
digest, every decoded column/object/row, fallback inputs and actual model columns.
The native smoke/recorded/replay lifecycle applies; replay re-admits metadata without
reading feature payloads, labels or computing predictions. No search, quantiles,
holdout evaluation or observed point-in-time claim is added.

| Stored-history execution condition | Outcome |
|---|---|
| Missing/malformed history pair, incompatible runner/mode, unconsumed CLI options | `INVALID_EXPERIMENT_SPEC` (CLI parser errors exit 2 before store access). |
| Changed pin, context, receipt or registered source/environment | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT`; no attempt or ledger slot. |
| Empty qualified training, incomplete evaluation or invalid stored numeric/missingness | Whole assessment refuses; no partial report. Native failure-stage rules apply after attempt. |
| v1.4 final-request preparation | `INVALID_EXPERIMENT_SPEC`; final declarations remain exclusive to v1.1. |

### Fixed fitted-history candidates (v1.5)

`--magnitude-model history_ridge|history_hgb` uses the same two clocks and exact
stored-history pair. Library callers additionally pass `magnitude_fit_model`.
Each registration selects one closed recipe; absent fitting preserves v1.4.
The v1.5 execution binds all eleven stored columns in their original order,
the recipe, separate fold/horizon fits, original percentage-point targets,
nonnegative prediction projection, one numerical thread and equal event/horizon loss.
There are no caller-supplied hyperparameters, target transforms or search.

`models.training.magnitude` owns guarded fitting using maintained sklearn APIs;
the reporter passes recipe data only after ops validates the closed execution policy.
Ridge uses training-only median imputation (all-empty columns retain zero),
all-column missing indicators and StandardScaler before fixed alpha-one Ridge.
HGB uses squared-error loss, fixed seed/iterations and no automatic validation
split or warm start. Training-all-null columns are frozen to zero in train/predict
matrices; other nulls remain native. This avoids sklearn 1.9.0's empty-bin failure
without dropping columns or events and without learning from evaluation values.
Every fold/horizon uses a fresh estimator and only strictly qualified training
events. Empty history is handled by the declared preprocessing, not mean fallback.

The history reporter retains whole-run physical preflight and supported producer
admission. `magnitude_fitted` assembles each horizon's training/prediction matrices
from stored rows, binds their ordered keys/values and qualified labels by digest,
and freezes detached forecasts before that fold's evaluation labels open.
It never builds features. The result adds per-horizon fit evidence and a distinct
schema; existing fixed-history result bytes/policy remain unchanged for identical inputs.
Registration/replay never fit. Native smoke/recorded publication remains unchanged.

| Fitted-history condition | Outcome |
| --- | --- |
| Unknown recipe or missing magnitude/history inputs | `INVALID_EXPERIMENT_SPEC` before repository admission; CLI parser errors exit 2 before store access. |
| Changed recipe, parameter, feature order, preprocessing, source or pin | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT`; no implicit registration or fit. |
| Invalid fit arrays, estimator failure, wrong-shaped/nonfinite predictions | `EXPERIMENT_VARIANT_FAILED`; no partial successful report or baseline fallback. |
| A no-fit guard raises `RuntimeFitForbidden` (v2 or legacy type) during a fit | Propagates unchanged; it is never converted to `EXPERIMENT_VARIANT_FAILED`. |
| Missing/late evaluation labels or empty qualified training | Existing whole-assessment refusal; no row dropping or horizon omission. |
| v1.5 final-request preparation | `INVALID_EXPERIMENT_SPEC`; no final-read authority. |

| Condition | Outcome |
| --- | --- |
| Invalid plan, external inputs, unsupported runner path, or missing/malformed/indirect source | `INVALID_EXPERIMENT_SPEC` before registration. |
| Current scope head cannot be resolved | `SNAPSHOT_UNRESOLVED`; no fallback. |
| Invalid context or explicitly excluded/unknown event | Shared `HOLDOUT_ACCESS_DENIED`; no outcome read. |
| Admission has no native registration | `INVALID_EXPERIMENT_SPEC`; no implicit registration. |
| Existing key has changed code, environment, plan, snapshot, population or context, or corrupt stored evidence | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT`; existing catalog and artifact bytes stay unchanged. |
| Publication or catalog commit fails | No partial registration is admitted; a complete unreferenced artifact may remain. No automatic retry. |

## Sealed magnitude search plans

`magnitude_search_plan.seal_search_plan` / `read_search_plan` compile and verify
`magnitude_search_plan.v1` artifacts from a catalog-owned fitted v1.5 registration.
The standalone `python -m experiments.v2_magnitude_search prepare|verify` CLI has no
production caller. It consumes an existing native catalog/store, anchor spec and request
JSON; verify additionally takes a plan-ref JSON. The public research fold rule supplies
metadata boundaries. Standard-library Cartesian enumeration supplies candidate recipes.
Direct dependencies are `engine.v2.contracts`, `engine.v2.data.errors`, foundation,
ops and research, plus the native-registration experiment adapter. Planning code identity
uses this installed repository root; the supplied anchor must match that installation.
This metadata-only CLI accepts `--no-ledger` as a no-op for runner compatibility;
both flag states publish/verify the same plan and never write a ledger.

| Condition | Enforced outcome |
|---|---|
| Request shape | `magnitude_search_request.v1`: explicit `inner_years`, `space`, `max_candidates`, `max_fits`; Ridge `alpha` and/or HGB `learning_rate`, `max_iter`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization` axes. |
| Valid fitted anchor and closed request | Return detached canonical plan bytes binding request, anchor, installed sources/environment, full recipes/hashes, grouped event/horizon metadata, count limits and recorded selection policy. No estimator construction, fitting, labels, features, repository scans or results. |
| Missing/extra fields, invalid or unordered axes/years, or candidate count outside 2..`max_candidates` (itself 2..4096) | Nonretryable `INVALID_EXPERIMENT_SPEC` before search-plan artifact access/expansion; CLI registration metadata may already be read. Ridge alpha and all five HGB axes are explicit; no inferred knobs. |
| Reconstructed outer metadata differs from the anchor | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT`; no publication. |
| Inner-year map omits an outer year or lacks earlier training/evaluation candidates | Nonretryable `INVALID_EXPERIMENT_SPEC`; no fold skipping or publication. |
| Valid nested metadata | Inner events come only from outer training membership; each event remains grouped across horizons. The shared annual rule defines the conservative fit boundary. Recorded label/selection policies are metadata, not evidence of qualified labels or a selected winner. |
| Candidate/inner-fold/horizon fits plus selected outer-fold/horizon refits exceed the explicit fit cap | Nonretryable `INVALID_EXPERIMENT_SPEC` before Cartesian expansion/publication. No time, memory or accuracy guarantee. |
| Catalog ownership, fitted schema, installed anchor sources or environment differ | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT`; no publication. |
| Prepare succeeds or repeats unchanged inputs | Publish/return the same immutable content-addressed plan; no catalog reservation, attempt, ledger or final authority. Existing anchor/producer bytes remain unchanged. |
| Verify receives a reference differing from complete request reconstruction | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT` before that plan object opens. |
| Verified bytes fail integrity | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT`; no repair or automatic retry. |
| Non-integrity foundation `ArtifactError` at plan publication/read, including a missing plan object | `RESOURCE_UNAVAILABLE` with a sanitized `Problem`; no automatic retry or partial success. |
| Opened CLI spec/request/plan-ref descriptor is nonregular or exceeds 1 MiB, or opening refuses a leaf symlink | Nonretryable `INVALID_EXPERIMENT_SPEC` before catalog access; bounded descriptor reads reject growth beyond the cap. |
| CLI typed refusal | Preserve the full `Problem` (including native spec conversion), return exit 2 and close an opened catalog connection. Filesystem/artifact unavailability, including an unopenable socket path, maps to `RESOURCE_UNAVAILABLE`; malformed or excessively nested JSON maps to `INVALID_EXPERIMENT_SPEC`. |
| Unexpected builtin exception from catalog/registration/planning code | Propagate it after closing the catalog; do not misclassify it as malformed user JSON. |

## Recorded magnitude-search selection

`magnitude_search_trials.SearchTrials` is library-only, with no CLI or production caller. `magnitude_search_records` owns journal tables, immutability triggers and reference binding over SQLite/foundation; `magnitude_search_errors` tags artifact failures. Direct dependencies are contracts, foundation, ops migrations/catalog/fingerprints/errors, and research stored-history/target reads. `magnitude_history_inputs.history_binding` and `magnitude_search_plan.read_search_plan` own native/plan admission; public `magnitude_report.magnitude_metrics` owns pooled RMSE/MAE.

| Condition | Outcome |
|---|---|
| A submission violates multiple boundaries | Schema/plan/seal admission takes precedence over trial identity; missing start takes precedence over argument/content refusal; content refusal takes precedence over missing native-input binding. Later phase outcomes require preceding guards to pass. |
| `seal(clock=...)` | Verify the plan and installed sources before explicitly initializing the search-owned journal through ops migrations; bind one plan per anchor to those recorder sources. |
| `start`, `record` or `freeze` with missing/older, newer or checksum-mismatched journal migrations | `INTEGRITY_FAILED` with the migration reason before plan/evidence/native reads; no implicit migration. |
| `start(outer_year, inner_year, candidate_id)` for an unclaimed planned trial with valid native source/stored-feature bindings, inner participant preflight and training labels strictly before the sealed inner-fit cutoff | One immutable start claim predates native reads. Success returns stored feature rows and qualified training targets bound to the same immutable cohort for every candidate in the fold. |
| Trial outside the sealed plan; `record` before `start` | `INVALID_EXPERIMENT_SPEC` for an unplanned trial; `EXPERIMENT_IDENTITY_CONFLICT` for a missing start; no native reads. |
| An admitted, started trial receives both or neither of `forecasts`/`failure` | `INVALID_EXPERIMENT_SPEC` before native input/label reads or any record publication; a missing start takes precedence as `EXPERIMENT_IDENTITY_CONFLICT`. |
| Complete nonnegative finite real forecasts and integral horizons; complete inner validation labels before the sealed outer-fit cutoff | Immutable JSON float/int forecast evidence predates validation-label reads. Scoring and successful terminals require one immutable validation cohort binding native-derived targets and the full receipt to admitted preflight membership, cutoff and recipe; concurrent candidates must agree exactly. Nonempty future/holdout exclusions in an explicit-ID receipt refuse `EXPERIMENT_IDENTITY_CONFLICT`. |
| Non-JSON request; Boolean, malformed or incomplete forecasts | `INVALID_EXPERIMENT_SPEC` before any validation-label read. |
| Explicit `EXPERIMENT_VARIANT_FAILED` fit failure after input admission; defensive local scoring refusal | Record a candidate terminal. An initial scoring refusal raises its original error; a failed terminal write is chained as its cause, leaving an unfinished claim. No other stage creates automatic candidate refusals. |
| `record(failure=...)` receives a registered non-candidate Problem | Raise its sanitized registered policy without a terminal; retain the unfinished claim. Unregistered/non-Problem values refuse `INVALID_EXPERIMENT_SPEC`. |
| Any shared native-input/label admission failure | Propagate the original exception, including native `DataError` or binding `OpsError`; preserve the unfinished attempt and block selection regardless of error code. Non-replayable start and forecast writes refuse restart or forecast resubmission, even for a retryable resource Problem. |
| Durable journal write; raw SQLite error in migration checks, reference reads or transactions | Publish artifacts before catalog commits; a failed commit may leave an unreferenced object, never admitted evidence. Raw `sqlite3.Error` propagates unchanged, except as the chained cause of a refusal-persistence failure; failed transactions roll back. SQLite failure never becomes a candidate terminal. Records are capped at 16 MiB and cannot be updated, deleted or replaced. |
| `freeze()` with every planned candidate/inner-fold attempt terminal | Exclude a candidate's entire outer score after any failed/refused inner fold; pool all successful event/horizon errors, break exact RMSE ties by candidate ID, and freeze every outer choice plus the latest-outer prospective candidate. Admitted successful folds have nonempty evaluation rows. A pooled-scoring refusal propagates `EXPERIMENT_VARIANT_FAILED` for the whole freeze, with no selection or candidate exclusion. |
| Missing/started-only attempt or outer fold without a successful candidate | Refuse `EXPERIMENT_VARIANT_FAILED`; publish no selection. |
| Exact terminal/freeze replay | Verify submitted-forecast terminals against the separate validation cohort and successful targets/receipt against its exact bytes, without reopening native inputs/labels. Return the original result idempotently; refused replay returns its durable envelope without rerunning scoring or raising the initial error again. |
| Current migrated journal missing its seal; valid submission/evidence without admitted inputs, repeated start, changed submission, another plan for the anchor, source drift or corrupt evidence | Refuse `EXPERIMENT_IDENTITY_CONFLICT`; no retry authorization. Missing migrations instead refuse `INTEGRITY_FAILED` under schema admission above. |
| Unavailable artifact, raw `OSError`, or read-time `UNSAFE_PATH` after stored-path syntax admission | Refuse sanitized retryable `RESOURCE_UNAVAILABLE`, including open-time OS errors and symlink rejection; no journal association changes, symlink traversal or attempt-retry authorization. |
| Invalid stored-path syntax or reference; publication `UNSAFE_PATH` | Refuse `EXPERIMENT_IDENTITY_CONFLICT`; invalid reference syntax is rejected before artifact access. Content/hash mismatch during verified reads remains an identity conflict. |
| Journal payload exceeds 16 MiB | Refuse nonretryable `RESOURCE_LIMIT_EXCEEDED` before publishing that record; incomplete attempts remain unfinished and block selection. |
| Submitted-forecast selection | The source-sealed journal binds recorder-owned native admission, not independently authenticated execution; wholesale fabrication of the cohort and matching terminals is outside its consistency guarantee. Estimator execution stays unverified; no fitting, outer evaluation, deployment, ledger or final-read authority. |

## Stored research-history entrypoint

`python -m experiments.magnitude_history_store build|read` requires an existing catalog file.
Other inputs are an existing artifact-store directory, exact source snapshot, holdout month, evaluation years,
training start, history start (on/before every forecast ceiling) and optional repeatable `--event-id`.
`read` also requires the exact feature snapshot.
The entrypoint fingerprints its installed source closure and numerical environment;
callers cannot submit features, a context document or an alternative producer.
Research owns the native contract, protected build and stored-only reader; ops
supplies existing bootstrap/fingerprinting, with no peer-layer import added.
The CLI uses `engine.v2.data` Repository/errors and foundation ArtifactStore/errors/clock.
Build returns the source/context/output identities; read returns stored rows.
Both actions run shared catalog bootstrap/migrations; stored-only describes feature consumption.
Before bootstrap or ArtifactStore construction, a missing/non-directory store root refuses `UNSAFE_PATH`.
Native object admission then checks source contents; the CLI never initializes a missing root.
Both are explicit local operations, never implicit provider downloads or model
runs. Data/ops failures retain code/message; artifact failures retain only their
code. Raw local OS/SQLite failures produce `INPUT_CHANGED` without exception text.
Handled runtime failures are stdout JSON and exit 2; argument refusals use argparse stderr.
No partial successful output or ledger entry is returned.

## Native outcome publication

`native_outcomes.publish_native_outcome` records one admitted success or typed
refusal; `replay_native_outcome` completes its interrupted publication without
evaluation. Both verify the supplied native registration. Callers must also
perform current-request admission before replay or fitting. A success requires
an actual attempt; registration alone never counts as an attempted variant.
Immutable report and outcome objects use `ArtifactStore`; existing catalog evidence
owns separate smoke and recorded outcome slots, globally binding the
registration, result, original row date and hashed ledger destination. Snapshot
and holdout provenance derive from the verified registration, never duplicate fields. No new
catalog table or supervisor coordinator is introduced. File/CSV effects stay outside
short catalog transactions; completion follows durable append and never authorizes promotion.
`export_native_report` creates `REPORT.md` at the registration's fixed store-local
smoke/recorded address only after verified completion; existing bytes never change.
The export has an independent inode: a private copy is flushed and synced before
create-only installation, so editing the export cannot alter stored report bytes.
An existing export sharing the stored object's inode is an identity conflict.
Report-bearing publication, replay and export preflight the fixed destination
before new result artifacts, intent reservation, ledger append or completion.
Existing non-directory ancestors and other destination conflicts are R6 before
effects; preflight and CSV/export are not one atomic transaction.
Recorded destinations must be nonempty and bind the exact normalized path used
for append. Saved outcomes require the complete canonical schema, object-valued
failure details and a nonblank UTF-8 success report before any ledger effect.
Saved completion payloads must match canonical expected metadata, preserving JSON
scalar types, before any append or completion publication.
Outcome, report and completion references must match the exact identity rederived
from verified bytes; malformed metadata refuses before ledger or catalog effects.

| Condition | Outcome |
| --- | --- |
| R1 invalid specification, R2 unresolved snapshot, or R6 identity conflict | `publish_native_refusal` publishes private evidence only; no result or ledger row. |
| Empty, directory or symlinked recorded ledger destination | R1 before outcome reservation or ledger access; no default-ledger fallback or conflict evidence. |
| Publish, replay or export called within an active caller transaction | R1 before path access or artifact/CSV effects; leave the caller transaction unchanged. |
| Publish, replay or export with metadata-only magnitude v1.2 | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT` with private refusal evidence only, before outcome/report reads or catalog/ledger/export effects. Validated v1.0/v1.1/v1.3/v1.4/v1.5 are accepted; v1.2 remains registration-only. |
| Admitted R3 look-ahead or R5 holdout denial | Immutable refusal and `refused` row, unless smoke; no current-attempt report. |
| Admitted R4 variant failure | Immutable refusal and `failed` row, unless smoke; no current-attempt report. |
| Same canonical registration/result/destination repeated, including tuple/list-equivalent refusal details | Reuse the first outcome and date; reconcile one terminal CSV row per experiment/variant identity without refitting. |
| Changed result, status, binding or recording destination; corrupt, incomplete or semantically invalid saved evidence | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT` with private conflict evidence before ledger effects; preserve prior report, catalog slot and CSV bytes. |
| Publication/append/commit interruption | Keep complete published objects and any committed intent/row; no rollback or automatic retry. Explicit replay reconciles the same intent. |
| `no_ledger=True` | Do not resolve, read, create, lock or append any ledger path. The smoke slot cannot consume the recorded slot. |

Pre-admission receipts cannot invent admission; rejected-request ledger identities belong to the runner.

## Reconstructed gate evaluation

`python -m experiments.v2_gate register|run` evaluates one preregistered arm.
It composes native registration, `gate_report` and native outcome/replay/export;
the separate supervisor pricing/evaluation kinds share the fixed child entrypoint.
The fixed recipe requires this runner, seed zero, ordered unique year folds,
option-chain prices, and exactly `strategy` and finite `fill` economics.
Other declarations refuse before repository access. Each arm independently
resolves its immutable plan and reprices its own training and evaluation rows.
The registration binds recipe source, environment, snapshot, population and
holdouts; the report also binds actual history, cutoffs and replay inputs.

This is exploratory reconstructed-history research.
Calendar sessions come from the exact snapshot without projected extensions.
The feature ceiling is the completed session strictly before the earlier of
the strategy's decision and entry. Protected `load_event_history` supplies
`n_prior` and `mean_prior_abs_move` from the calendar's first session onward.
Empty history alone maps the missing mean to zero, retaining its zero count.
History/source dates are never converted into claimed observation timestamps;
schedule-as-known and source publication remain unverified.

Folds partition by gate-decision year. Training uses earlier decision years
whose exits precede both the year start and earliest evaluation feature ceiling.
Each fold fits a fresh depth-one sklearn DecisionTreeClassifier on profitable
training trades, requiring both classes. The shared training-score median
rule freezes its threshold before evaluation quotes/outcomes are read.
Per-event test histories may incorporate earlier completed events, but test
labels never fit the model or threshold. Selection uses score >= threshold.
Every requested event must have a complete native calendar plan; each consumed
pricing set must be complete and finite, otherwise the whole arm refuses.

Reports contain selected and all-candidate mark-based mean P&L (no commissions), row counts,
fold memberships, scores, cash flows, feature ceilings and complete provenance.
They make no observed-PIT, fill-validation, holdout, promotion or winner claim.
One arm counts as one attempt only when evaluation starts. Recorded success
uses its own report and ledger identity; smoke never accesses a ledger path.
Replay re-admits current inputs then reconciles existing evidence without fitting.
Changed inputs under the same arm identity refuse R6, preserving prior evidence.

| Condition | Outcome |
| --- | --- |
| Explicit excluded/unknown event or final-read registration | `HOLDOUT_ACCESS_DENIED` before features, pricing or metrics; no partial report. |
| Missing prior session, empty fold, incomplete/nonfinite inputs or fitting failure | `EXPERIMENT_VARIANT_FAILED`; existing native failure publication applies. |
| Admission, publication or replay failure | Existing native lifecycle's typed refusal and interruption semantics apply. |

## Native gate-variant pricing

The `native_gate_pricing` supervisor kind launches only the fixed unlayered
`native_gate_entrypoint` child. After the ops stdin/PID launch gate it accepts exactly
`native_gate_pricing` and `native_gate_evaluation`, importing their fixed callbacks.
For pricing, both entrypoint and adapter belong to the verified registration/source snapshot.
The adapter reads staged `spec.json` and
`native_registration.json`, verifies existing preregistration, and checks
the requested identity, resolved plan, source closure and single-thread
environment before pricing. Required `economic_params.strategy` selects the
native strategy independently of arm IDs; the fixed worker requires it before
registration reads or effects, and declared strategies must be supported. Snapshot,
event IDs and holdout context come only from registration.
It never re-resolves a scope head or accepts a supplied trades frame.
Submission uses the existing `SubmitRequest` API and artifact input bindings;
no production entrypoint submits this kind. The worker's catalog connection is read-only.
Each staged document itself must be a nonsymlink regular file of at most 16 MiB;
pathname replacement cannot redirect admitted bytes.
The worker installs one `native_gate_consumption.v1.0` receipt with the immutable plan,
stage inputs and mark-based cash flows; the supervisor publishes it only after worker success.
Installation is create-only and never assigns partial bytes; unsupported hard links
refuse `INPUT_CHANGED`. Existing destinations are never opened, overwritten or unlinked.
This receipt is not an evaluation report or promotion evidence.

| Native worker condition | Outcome |
| --- | --- |
| Recording requested or missing smoke opt-in | Submission `INVALID_REQUEST`; worker defense `INVALID_EXPERIMENT_SPEC` before catalog access. |
| Non-mapping or ill-typed worker arguments, including non-string purpose; unsupported recipe or malformed staging, including missing/null required spec fields or excessive JSON nesting; missing staged bytes | `INVALID_EXPERIMENT_SPEC` before catalog access for malformed arguments and before pricing for malformed staging; `INPUT_CHANGED` for missing bytes. |
| Changed staged spec, registration, expected identity, source (including missing, unreadable or malformed Python) or numerical environment | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT` before pricing; preserve registration and prior evidence. |
| Unresolved pin or non-sweep purpose | `SNAPSHOT_UNRESOLVED` or `HOLDOUT_ACCESS_DENIED`; no pricing receipt or partial report. |
| Incomplete/nonfinite pricing or caught pinned-data failure | `EXPERIMENT_VARIANT_FAILED` for pricing/validation failures; `DataError` uses the shared canonical ops mapping below, preserving ops policy for existing codes and data category/retryability for data-only codes. Underlying codes/types stay in private `data_failure_code`; no success receipt. The replay/catalog regressions define the catch boundary. |
| Unexpected programming failure | The existing worker classifier remains authoritative: unexpected replay exceptions such as `RuntimeError`, `IndexError` or `AttributeError` yield generic `WORKER_FAILED`, private type/location diagnostics and no successful output. This admission-only boundary does not reconcile full-sweep R4 outcomes. |
| Retry/cache/transaction | One attempt; cross-job cache reuse is disabled so admission is checked again. No worker catalog transaction or CSV access. |
| Existing receipt destination or receipt I/O failure | Non-retryable `INPUT_CHANGED`; preserve existing destinations and the original failure. Private partial bytes or a complete assigned receipt may remain, never a partial assigned receipt or successful output. |
| Successful smoke | Operations job/output artifact rows only; no reusable checkpoint is stored. Native registration/outcome and experiment ledger rows stay unchanged. |

This pricing-only path does not fit gates, record outcomes or orchestrate sweeps.

## Sealed gate sweep submission

`python -m experiments.v2_gate_sweep register|submit --no-ledger` accepts a bounded regular JSON roster, a catalog migrated through version 13, its store, explicit `--code-root` and a holdout month; read-only schema admission requires current checksums (`INTEGRITY_FAILED` otherwise), without migration. CLI success exits 0 with `{"roster": <hash>}` for registration or `{"state": "submitted_uncollected", "job_ids": [...]}` for submission; typed refusal exits 2 with only `{"refused": <code>}`. Untranslated foundation `ArtifactError` retains its code in CLI refusals; typed sweep/evaluation boundaries preserve registered codes, otherwise use `INTEGRITY_FAILED` with the original code in private `data_failure_code`. Existing data/registration translations remain authoritative. The closed `native_gate_sweep_request.v1.0` envelope contains `variants` and an ordered, two-member `primary_comparison`; other arms are exploratory. Arm order and explicit event-ID order are canonicalized. Every arm has one immutable `v2_gate.py` plan; experiment, hypothesis, strategy, folds, seed and price source must agree. Only supported fill economics may differ; other declarations refuse `INVALID_EXPERIMENT_SPEC` before population reads or reservation. No prior experiment output or shared prepriced frame is an input. `register_sweep` binds every arm to one verified scope-head `SnapshotRef` and eligible population; the roster seals requests, registrations (each with its environment), coordinator source and implementation ref. A shared SQL reader bounds stored text before Python materialization; the decoder caps JSON at 16 MiB UTF-8: malformed, non-object or oversized rows give `EXPERIMENT_IDENTITY_CONFLICT`; oversized new seals give `INVALID_EXPERIMENT_SPEC` before insertion. Missing seals give `INVALID_EXPERIMENT_SPEC` on submit; register may create one. Existing roster-arm/spec reservations are checked in the seal transaction before insertion. Every reservation call after sealing, including replay, must match the roster before artifact publication and again in its committing transaction; either writer ordering preserves roster-arm bindings. Earlier non-roster registrations remain independent and runnable, outside sweep submission/accounting. Catalogs without a roster table retain single-arm admission.

SQLite transactions and existing canonical JSON/content-hash helpers own identity and atomicity; no new serialization or scheduling framework is used. `submit_sweep` requires matching catalog evidence before staging; `submit_graph` inserts children atomically and replays their job IDs. The regular operations service owns execution; submission never starts a supervisor. Single-arm `v2_gate.py` bindings retain their runner-only source roots. The sealed worker implementation manifest separately binds the evaluation entrypoint and is verified by the launch gate; existing registration bytes are never rewritten. The roster is preregistration, and job success is durable child evidence only. Neither registration nor submission creates scientific outcomes, CSV rows, reports, aggregate attempted counts or a completed sweep index. A failed child cannot imply scientific completion. This interface has no scientific collection, recording, reconciliation or incomplete-run accounting authority. Non-sweep purpose refuses `HOLDOUT_ACCESS_DENIED` before catalog/input access; missing smoke opt-in refuses before effects. No final-read authority is accepted.

| Condition | Outcome |
| --- | --- |
| Register replay or interrupted arm reservation | Verify the saved pin and resume missing reservations; never resolve a replacement head. |
| Submit with partial registration | Refuse `EXPERIMENT_IDENTITY_CONFLICT` before staging/jobs; only register resumes missing reservations. |
| Changed request, source bytes, environment, binding or existing roster arm | `EXPERIMENT_IDENTITY_CONFLICT`; preserve sealed evidence and create no roster for a pre-existing roster-arm/spec conflict. Earlier non-roster arms do not prevent sealing. Both Python APIs translate filesystem/SQLite failures during preparation, sealing, reservation, staging or graph submission to `INPUT_CHANGED`, with private exception type; existing typed errors retain their policy. |
| Unresolved initial or saved pin | `SNAPSHOT_UNRESOLVED`, without fallback. |
| `code_root` that is not path-like | `INVALID_EXPERIMENT_SPEC`, never a leaked `TypeError`. |
| Changed explicit IDs | Saved-pin metadata admission gives R5 for excluded/unknown IDs; changed eligible membership gives R6. |
| Child admission | Exact staged identity, runner-source closure and single-thread environment are required before inputs; malformed staged `ValueError`/`TypeError`/`KeyError`/`RecursionError` gives `INVALID_EXPERIMENT_SPEC`. |
| Child success | Independent prior-session history/fitting/pricing; one create-only `native_gate_result.v1.0` supervisor-committed artifact. |
| Child failure | No successful output or sibling suppression; one attempt, no reusable checkpoint or cache. |
| Data refusal (pricing and evaluation) | Codes registered by ops retain ops category/retryability, including non-retryable `INPUT_CHANGED`. Data-only codes map to a canonical ops code with the data registry's category/retryability (for example `OBJECT_CORRUPT` → `INTEGRITY_FAILED`); the original code stays in private `data_failure_code`. |
| Evaluation input refusal | Registered `ArtifactError` codes retain ops semantics; foundation-only codes give `INTEGRITY_FAILED`. Caught evaluation validation failures give `EXPERIMENT_VARIANT_FAILED`, with private `data_failure_code`. Raw filesystem/SQLite failures give `INPUT_CHANGED` in admission or evaluation. Unexpected programming errors keep generic worker classification. |
| Evaluation receipt | Serialization `ValueError`/`TypeError`/`RecursionError` gives `EXPERIMENT_VARIANT_FAILED` with private exception type; receipt I/O remains `INPUT_CHANGED`. Neither publishes successful output. |

`v2_gate_variant.price_variant(repository, snapshot, spec, *, strategy,
as_of_month, event_ids=None)` resolves one immutable single-arm plan and
freshly prices its eligible canonical events from that snapshot's option
chains and calendar. It accepts a finite `fill` and optional `strategy`
economics declaration, which must match the explicit strategy argument.
The resolver permits that key only for `native-gate-replay`, the fixed native worker and `v2_gate.py`.
The admitted worker requires that declaration; manual fill-only calls retain
their explicit strategy input. Replay consumes the resolved fill. Returned
stage inputs bind the strategy, alpha, snapshot, event population and holdout
context. No prepriced trade frame or previous experiment output is accepted.
The source population uses `research.experiment_population.load_population`
with `purpose="sweep"`; raw quotes are read only after exclusion succeeds.
This is a pricing prerequisite, with no gate fitting or sweep lifecycle.

`python -m experiments.v2_gate_variant --no-ledger` is the smoke entrypoint.
It pins the current scope head once, opens the existing catalog read-only,
and prints execution evidence. It writes no report, artifact or ledger row.
This manual CLI is separate from the admitted smoke worker; neither authorizes promotion.

| Condition | Outcome |
| --- | --- |
| Unknown runner/strategy, external input files, invalid fill, or other economics | `INVALID_EXPERIMENT_SPEC` before population or quote reads. |
| Missing or invalid pin/context, or an explicit excluded event | Preserve the shared population reader's typed refusal; no pricing result or report. |
| Replay cannot price every eligible event | `EXPERIMENT_VARIANT_FAILED`; no partial result is returned. Native data integrity refusals propagate. |
| Repeated call or failure | No cache, internal retry, transaction or publication; recompute from the same pin and never retain a partial result. |
| CLI omits `--no-ledger` | Refuse before opening the catalog; recorded execution is unavailable. |

## Evaluation recording and promotion receipts

| Condition | Outcome |
| --- | --- |
| `record_evaluation_result` has no `run_dir` or its results are not a `Mapping` | Raise `LedgerError` before appending a `ran` row. |
| `publish_receipt=False` | Append the `ran` row; skip artifact validation, finalization, and receipt publication. |
| Receipt publication is enabled and the artifact matches the run and spec | Append the row, finalize the checklist and `recording_mode`, then publish a receipt for the exact final bytes. |
| Validation or receipt publication fails after the append | Keep the `ran` row; the row alone cannot authorize promotion. |
| Promotion consumes an artifact | Require matching artifact, requested spec, receipt, run ID, digest, and qualifying ledger row; otherwise refuse with `PROMOTION_LEDGER_RECEIPT_MISSING`. |
| An `EvalResult` is recorded | Use its own `run_dir` for its metrics artifact, including for grid arms. |

## Durable experiment CSV publication

`lib.ledger_append` and `ledger_ensure` serialize cooperating writers through
one stable `<ledger>.append.lock`, including first-file creation. Publication
preserves the exact previous byte prefix and exposes only complete CSV records.
Success guarantees file and directory durability, including interrupted parent
creation; unrelated ancestors need only traversal access. Existing access-mode
bits are retained; a newly created ledger is private.
This lock is distinct from the ops refusal identity lock, which may enclose an
append; the append helper never acquires that outer lock. Direct file editors
do not participate in this concurrency guarantee. No catalog transaction is
opened, and callers remain responsible for their outcome/receipt reconciliation.

| Condition | Outcome |
| --- | --- |
| Missing required row/header fields, invalid replay key or directory-creation marker, malformed CSV, exceeded CSV field limit, unterminated existing record, or symlinked destination | `LedgerError`; no CSV replacement. |
| `unique_by` omitted | Append every supplied row, preserving existing legacy duplicate-row behavior. |
| Optional `unique_by` names a previously stored key | An identical complete row is a no-op; contradictory or duplicate stored rows refuse with `LedgerError`. The return value counts newly appended rows. |
| Temporary write or file fsync fails before replacement | Prior CSV remains; a killed process may leave an unreferenced temporary file. No automatic retry. |
| Parent creation or its sync fails | No CSV publication. Persistent directory-creation markers let a retry finish only the required parent syncs. |
| Directory fsync fails after replacement | New complete CSV exists, durability is uncertain; no rollback. Exact keyed replay can complete directory sync without duplicating rows. |
| Existing CSV changes before replacement outside the shared lock | Refuse the observed conflict without restoring stale bytes. Concurrent uncooperative edits are unsupported. |
| Repeated `ledger_ensure`, or identical keyed replay | Existing CSV bytes remain unchanged. Neither operation emits a recording or promotion receipt. |

## Native prediction prior baseline

`prediction_report.validate_prediction_spec` admits only the Python runner
`experiments/v2_prediction.py`, `computed_moves`, empty economics and ordered,
unique four-digit year folds, with an integer seed in [0, 2**32) bound as run
identity. The prior estimator is deterministic and does not consume
randomness, so `prediction_result` supports only seed `0` and refuses other
valid seeds before computed-move outcome reads rather than implying they
change the computation. Registration may still bind another valid seed as
identity.
`prediction_result(repository, registration)`
reads only registered eligible targets from the exact registered snapshot/month.
Each fold fits `DummyClassifier(strategy="prior")` through the shared fold helper
on earlier-year labels available strictly before its January 1; constant
intercept features make no market-feature claim. Fitting limits numerical
thread pools to one within a restoring context. Empty train/test or single-class
training refuses `EXPERIMENT_VARIANT_FAILED`; unsupported declarations refuse
`INVALID_EXPERIMENT_SPEC` before target reads. No final-read option exists.
The deterministic JSON result includes OOF/per-fold Brier and clipped log-loss,
ten fixed reliability bins with weighted ECE, and score/event-ID rank deciles.
Empty bins use null rates. Random and rolling holdouts are separately excluded.
`render_prediction_report(..., spec=..., no_ledger=...)` binds the supplied
hypothesis by spec hash and renders these metrics, provenance and limitations.
These functions have no cache, retry, writes or partial-result return; the caller
owns admission/publication/recording. No return, P&L or promotion claim follows.

`python -m experiments.v2_prediction register|run` consumes a JSON spec and an
existing catalog/store and explicit holdout month. It fingerprints its executing checkout. Register is
metadata-only. `read_native_registration` verifies the stored identity; run performs current
admission before replay or evaluation; missing registration is R1. R3/R4/R5
refusals use that admitted identity and never export a current-attempt report.
R1/R2/R6 preserve private receipts only. Untyped evaluation/rendering errors use
retryable `WORKER_FAILED`, never terminal outcomes or CSV rows. `--no-ledger`
isolates smoke recording; otherwise `--ledger` is required. Completed `REPORT.md` is exported through the
shared immutable outcome lifecycle; retries reuse the original result/date.
Stored spec/source/snapshot/scope/environment identity is checked before population
admission, so a changed request cannot consume an original variant's refusal slot.
A current holdout denial preserves R5 and its private receipt even when an earlier
immutable completion prevents a second terminal CSV row; prior reports stay unchanged.
