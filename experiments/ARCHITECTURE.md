# Experiments architecture

## Native preregistration

`native_registration.register_native` explicitly registers one resolved arm
before evaluation; `require_native_registration` is its read-only admission
check. Both pin the current scope head once and read canonical event metadata
through the shared holdout population reader, never outcomes or prior results.
The stable registration key is experiment ID plus primary arm. Its immutable
binding includes the full resolved plan, detached spec hash, exact snapshot,
static source closure, numerical environment and declared single-thread policy,
scope, eligible event IDs and versioned holdout
context. External input files and non-Python runner paths are refused.
The caller supplies the checkout root used for source fingerprinting.

Registration publishes canonical bytes through `ArtifactStore`, then reserves
the existing primary `hypotheses`/`experiment_runs` records in one catalog
transaction. Re-registering identical content returns the same identity.
The reserved arm is planned; registration does not mark a variant as tried.
Admission verifies the stored object and compares a freshly built binding;
returned documents are copies, not mutable views of registered bytes.
Registration and current-request admission do not fit models, publish reports,
write the CSV ledger or emit outcome/refusal receipts. The supervisor module does
not call them directly; its dispatched smoke worker reuses
`verify_native_registration` to verify its staged binding.

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
opens magnitude outcomes; v1.2 remains metadata-only and neither permits final reads.

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
market/history features, search and final evaluation are not implemented.
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

| Condition | Outcome |
| --- | --- |
| Invalid plan, external inputs, unsupported runner path, or missing/malformed/indirect source | `INVALID_EXPERIMENT_SPEC` before registration. |
| Current scope head cannot be resolved | `SNAPSHOT_UNRESOLVED`; no fallback. |
| Invalid context or explicitly excluded/unknown event | Shared `HOLDOUT_ACCESS_DENIED`; no outcome read. |
| Admission has no native registration | `INVALID_EXPERIMENT_SPEC`; no implicit registration. |
| Existing key has changed code, environment, plan, snapshot, population or context, or corrupt stored evidence | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT`; existing catalog and artifact bytes stay unchanged. |
| Publication or catalog commit fails | No partial registration is admitted; a complete unreferenced artifact may remain. No automatic retry. |

## Stored research-history entrypoint

`python -m experiments.magnitude_history_store build|read` requires a catalog,
object store, exact source snapshot, holdout month, evaluation years, training
start and earlier history start. `read` also requires the exact feature snapshot.
The entrypoint fingerprints its installed source closure and numerical environment;
callers cannot submit features, a context document or an alternative producer.
Research owns the native contract, protected build and stored-only reader; ops
supplies existing bootstrap/fingerprinting, with no peer-layer import added.
Build returns the source/context/output identities; read returns stored rows.
Both are explicit local operations, never implicit provider downloads or model
runs. Typed failures produce a refusal and nonzero exit; no partial successful
output, report, registration or ledger entry is created. A future feature-model
registration must bind both snapshots and this context before consuming rows.

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
| Publish, replay or export with metadata-only magnitude v1.2 | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT` with private refusal evidence only, before outcome/report reads or catalog/ledger/export effects. Validated v1.0/v1.1/v1.3 are accepted; v1.2 remains registration-only. |
| Admitted R3 look-ahead or R5 holdout denial | Immutable refusal and `refused` row, unless smoke; no current-attempt report. |
| Admitted R4 variant failure | Immutable refusal and `failed` row, unless smoke; no current-attempt report. |
| Same canonical registration/result/destination repeated, including tuple/list-equivalent refusal details | Reuse the first outcome and date; reconcile one terminal CSV row per experiment/variant identity without refitting. |
| Changed result, status, binding or recording destination; corrupt, incomplete or semantically invalid saved evidence | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT` with private conflict evidence before ledger effects; preserve prior report, catalog slot and CSV bytes. |
| Publication/append/commit interruption | Keep complete published objects and any committed intent/row; no rollback or automatic retry. Explicit replay reconciles the same intent. |
| `no_ledger=True` | Do not resolve, read, create, lock or append any ledger path. The smoke slot cannot consume the recorded slot. |

Pre-admission receipts cannot invent admission; rejected-request ledger identities belong to the runner.

## Native gate-variant pricing

The `native_gate_pricing` supervisor kind launches only the fixed unlayered
`native_gate_entrypoint` child. Its callback imports native pricing after the
ops protocol's stdin/PID launch gate, refusing every other worker name. Both
entrypoint and adapter belong to the verified registration/source snapshot.
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
| Incomplete/nonfinite pricing or caught pinned-data failure | `EXPERIMENT_VARIANT_FAILED`; registered data refusal codes retain their code. Caught underlying failures retain their code, or uncoded type, only in private `data_failure_code` details; no success receipt. The real replay and catalog-failure regressions define the supported catch boundary. |
| Unexpected programming failure | The existing worker classifier remains authoritative: unexpected replay exceptions such as `RuntimeError`, `IndexError` or `AttributeError` yield generic `WORKER_FAILED`, private type/location diagnostics and no successful output. This admission-only boundary does not reconcile full-sweep R4 outcomes. |
| Retry/cache/transaction | One attempt; cross-job cache reuse is disabled so admission is checked again. No worker catalog transaction or CSV access. |
| Existing receipt destination or receipt I/O failure | Non-retryable `INPUT_CHANGED`; preserve existing destinations and the original failure. Private partial bytes or a complete assigned receipt may remain, never a partial assigned receipt or successful output. |
| Successful smoke | Operations job/output artifact rows only; no reusable checkpoint is stored. Native registration/outcome and experiment ledger rows stay unchanged. |

This admitted path does not fit gates, record outcomes or orchestrate sweeps.

`v2_gate_variant.price_variant(repository, snapshot, spec, *, strategy,
as_of_month, event_ids=None)` resolves one immutable single-arm plan and
freshly prices its eligible canonical events from that snapshot's option
chains and calendar. It accepts a finite `fill` and optional `strategy`
economics declaration, which must match the explicit strategy argument.
The resolver permits that key only for `native-gate-replay` and the fixed native worker.
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
