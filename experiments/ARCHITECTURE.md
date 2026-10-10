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
These library entrypoints do not fit models, publish reports, write the CSV
ledger or emit outcome/refusal receipts, and have no supervisor caller.

| Condition | Outcome |
| --- | --- |
| Invalid plan, external inputs, unsupported runner path, or missing/malformed/indirect source | `INVALID_EXPERIMENT_SPEC` before registration. |
| Current scope head cannot be resolved | `SNAPSHOT_UNRESOLVED`; no fallback. |
| Invalid context or explicitly excluded/unknown event | Shared `HOLDOUT_ACCESS_DENIED`; no outcome read. |
| Admission has no native registration | `INVALID_EXPERIMENT_SPEC`; no implicit registration. |
| Existing key has changed code, environment, plan, snapshot, population or context, or corrupt stored evidence | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT`; existing catalog and artifact bytes stay unchanged. |
| Publication or catalog commit fails | No partial registration is admitted; a complete unreferenced artifact may remain. No automatic retry. |

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
An existing conflicting export destination refuses before reconciliation effects;
this preflight and create-only export do not form an atomic transaction with CSV.
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
| Admitted R3 look-ahead or R5 holdout denial | Immutable refusal and `refused` row, unless smoke; no current-attempt report. |
| Admitted R4 variant failure | Immutable refusal and `failed` row, unless smoke; no current-attempt report. |
| Same canonical registration/result/destination repeated, including tuple/list-equivalent refusal details | Reuse the first outcome and date; reconcile one terminal CSV row per experiment/variant identity without refitting. |
| Changed result, status, binding or recording destination; corrupt, incomplete or semantically invalid saved evidence | Non-retryable `EXPERIMENT_IDENTITY_CONFLICT` with private conflict evidence before ledger effects; preserve prior report, catalog slot and CSV bytes. |
| Publication/append/commit interruption | Keep complete published objects and any committed intent/row; no rollback or automatic retry. Explicit replay reconciles the same intent. |
| `no_ledger=True` | Do not resolve, read, create, lock or append any ledger path. The smoke slot cannot consume the recorded slot. |

Pre-admission receipts cannot invent admission; rejected-request ledger identities belong to the runner.

## Native gate-variant pricing

`v2_gate_variant.price_variant(repository, snapshot, spec, *, strategy,
as_of_month, event_ids=None)` resolves one immutable single-arm plan and
freshly prices its eligible canonical events from that snapshot's option
chains and calendar. It accepts only a finite `fill` economics declaration;
the alpha passed to native replay is the resolved plan's own value. Returned
stage inputs bind the strategy, alpha, snapshot, event population and holdout
context. No prepriced trade frame or previous experiment output is accepted.
The source population uses `research.experiment_population.load_population`
with `purpose="sweep"`; raw quotes are read only after exclusion succeeds.
This is a pricing prerequisite, with no gate fitting or sweep lifecycle.

`python -m experiments.v2_gate_variant --no-ledger` is the smoke entrypoint.
It pins the current scope head once, opens the existing catalog read-only,
and prints execution evidence. It writes no report, artifact or ledger row.
It is not an admitted supervisor runner or a promotion-authorizing evaluation.

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
