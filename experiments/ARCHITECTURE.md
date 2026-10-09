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
