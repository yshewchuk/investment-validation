# Native experiment interfaces

These unlayered entrypoints compose the v2 platform. See [ARCHITECTURE.md](ARCHITECTURE.md)
for registration, single-arm evaluation and staged sweep contracts.

## Shared APIs

The owning modules expose these helpers to native experiment callers; no compatibility aliases are provided.

| Owner and public function | Inputs, result and effects |
| --- | --- |
| `magnitude_report.magnitude_metrics` | Complete scored rows yield stable pooled RMSE, MAE and count; empty rows or nonfinite metrics refuse `EXPERIMENT_VARIANT_FAILED`. Fixed and history reporters and the search recorder share this scorer. |
| `magnitude_search_errors.artifact_failure` / `SearchArtifactError` | Artifact error plus caller-owned messages yields a redacted `OpsError` with identity/resource policy; the subtype identifies artifact-origin failures without parsing messages. Journal reference syntax and publication `UNSAFE_PATH` are identity conflicts; read-time `UNSAFE_PATH` after syntax admission is a retryable resource failure, with no attempt-retry authorization. Plan publication retains its resource policy. |
| `magnitude_search_trials.SearchTrials` | Verified plan/anchor, catalog/store and native repository; `seal(clock=...)`, `start(outer_year, inner_year, candidate_id)`, `record(..., forecasts=... or failure=Problem)` and `freeze()` persist immutable all-attempt evidence and selections from submitted forecasts. Candidate terminals are guarded here by stage and canonical `EXPERIMENT_VARIANT_FAILED` admission. A native-derived validation cohort independently binds terminal targets and the complete receipt to admitted preflight/cutoff/recipe; replay never reopens labels. This detects inconsistent journal evidence, not wholesale fabrication of all evidence. Estimator execution is unverified; no fitting, outer evaluation or final authority. |
| `native_registration.prepare_native_registration` | A validated spec, repository, source root and holdout context produce `(NativeRegistration, normalized_spec)`; resolves one scope pin or verifies an explicit `SnapshotRef`, reads eligible metadata only, and writes nothing. |
| `native_registration.native_registration_exists` | An expected `NativeRegistration` gives false when absent, true for exact verified catalog/object evidence, or `EXPERIMENT_IDENTITY_CONFLICT`; read-only. Optional `max_evidence_bytes` bounds stored catalog/object JSON through the shared ops verifier; default None preserves prior policy. |
| `native_registration.reserve_native_registration` | Accepts the matching prepared registration/spec pair from the preparer; publishes complete binding bytes and reserves catalog rows, obeying any sweep seal. Exact replay is idempotent; conflicts refuse R6. Publication may leave an unreferenced complete object if reservation fails. |
| `native_registration.read_native_gate_sweep` | Catalog/experiment ID yields stored text or None for absence. SQL atomically caps returned bytes at 16 MiB (32 MiB for UTF-16 catalogs), then decoding enforces 16 MiB UTF-8; wrong type, invalid encoding or oversize gives `EXPERIMENT_IDENTITY_CONFLICT`. Preparation, sealing and reservation share it; SQLite internals/decoded-object overhead are not capped. |
| `native_registration.decode_native_gate_sweep` | Roster JSON text yields an object after a 16 MiB UTF-8 cap; malformed/oversized stored rows refuse `EXPERIMENT_IDENTITY_CONFLICT`. The new-seal writer requests `INVALID_EXPERIMENT_SPEC` before insertion. No writes or identity normalization. |
| `native_gate_worker.read_native_staged_input` | A path yields bounded bytes from a no-follow regular-file descriptor; malformed files refuse `INVALID_EXPERIMENT_SPEC` and I/O errors propagate to the caller's typed boundary. |
| `native_gate_worker.admit_native_gate_inputs` | Staged parameters/root, read-only catalog and store yield `(spec, registration)` only after exact identity, source and environment admission. No evaluation or writes; this does not replace supervisor launch verification. |
| `native_gate_worker.write_native_gate_receipt` | A destination and document install complete canonical bytes create-only; existing destinations survive. Serialization/I/O errors propagate; only private temporary names are cleaned. |
| `native_gate_worker.native_gate_data_failure` | A `DataError` yields a redacted `OpsError` for pricing/evaluation: existing ops codes retain their policy; data-only codes map by registered category/retryability, retaining the original code only in private details. |
| `v2_gate_sweep.register_sweep` | Catalog connection/store/repository, closed roster, source root, holdout month and optional scope/event IDs yield the sealed roster hash. Captures one common pin, seals immutable evidence and resumes matching arm reservations; no jobs or scientific outcomes. |
| `v2_gate_sweep.submit_sweep` | The same inputs require the existing exact seal and complete registrations; stages bound inputs and atomically returns independent job submission receipts, replaying their IDs. Does not start the supervisor or collect/record results; partial staging artifacts may precede a submission refusal. Optional `max_evidence_bytes` forwards a collector-selected bound to stored registration verification; default None leaves existing callers unchanged. |
| `native_sweep_collection.collect_sweep` | The same sealed inputs plus exactly one smoke/ledger mode yield an index and terminal artifact reference (otherwise None). Requires an existing directory for `code_root` (`INVALID_EXPERIMENT_SPEC` otherwise); root-stat I/O remains retryable. Rechecks the import-time installed-source snapshot and caller coordinator entry files before submission. Verifies exact attempts and bounded failure/result/native evidence bytes; recorded mode reconciles native outcomes/CSV/report exports, smoke only sweep artifacts. Identity/raw constraint conflicts refuse R6; raw boundary I/O is retryable `RESOURCE_UNAVAILABLE`. Delegated APIs retain their own error policies; unexpected programming errors reaching the collector boundary propagate. |
| `native_outcomes.publish_native_outcome`, `replay_native_outcome`, `export_native_report` | Existing native publication/reconciliation APIs accept optional `max_evidence_bytes` for catalog evidence and registration/outcome/completion JSON reads and matching writes. Default None preserves existing caller behavior; collection opts into 16 MiB. Reports are verified at their registered size. With the opt-in bound, existing REPORT.md comparisons open a no-follow regular descriptor, reject sizes above the expected report and read at most its byte length plus one; mismatches refuse R6, storage I/O remains retryable. This is a comparison bound, not the JSON size ceiling for reports. |
| `v2_gate_collect.run_sweep` | Collection inputs and a positive finite timeout observe submitted jobs until terminal/deadline, then collect. Returns the same index shape; never owns workers, retries jobs or stops the supervisor. Observer I/O/non-constraint SQLite failures are retryable `RESOURCE_UNAVAILABLE`; integrity constraints or malformed stored receipts refuse R6 (stored job failures retain `INTEGRITY_FAILED`). Library/CLI timeout defaults are 3600 seconds. |
| `v2_gate_collect.main` | CLI argument list selects collect/run, validates purpose/mode before opening inputs/catalog, prints the index or typed refusal JSON and returns 0 only for complete success, otherwise 2. Requires an existing migrated catalog. Roster path errors ENOENT/ENOTDIR/EACCES/EPERM/ELOOP/EINVAL/ENAMETOOLONG, nonregular input and malformed/recursive roster JSON refuse `INVALID_EXPERIMENT_SPEC` (non-retryable); other roster I/O (including EIO/EMFILE/ENOMEM) and storage I/O are retryable `RESOURCE_UNAVAILABLE`. Artifact/data refusals retain their original code. |

## Consumers and authority

`v2_gate_sweep` uses registration preparation/lookup/reservation, the bounded reader and `gate_report.validate_gate_spec`.
`native_sweep_collection` consumes `v2_gate_sweep.submit_sweep`, sealed registration, `native_outcomes`, `gate_report.render_gate_report`, ops transactions/registered artifacts/spec parsing/source closure/bounded native evidence readers and writers/job reads/`MAX_JOB_FAILURE_BYTES`/errors, contracts and foundation. `v2_gate_collect` consumes that collector, staged-input reading, ops catalog/job reads/errors, data `Repository`/errors and foundation artifacts.
It consumes ops catalog/transactions, checkpoints, errors, spec parsing, fingerprints, migration/schema admission, stage registry and atomic submission; data supplies `Repository`/typed errors, contracts supply jobs/snapshots, and foundation supplies artifacts, clock and canonical serialization.
`native_gate_evaluation` shares admission/receipt writing with the pricing worker;
its other edges are data errors/`Repository`, contracts' failure registry, foundation artifacts/decoding, ops errors/`NativeGateParameters`, and `gate_report` for the fixed runner and result. It shares the worker's data-error mapping. Registration reservation and sweep preparation use the bounded roster decoder.
The fixed entrypoint admits only those two worker names after supervisor launch checks.
The evaluation child maps receipt serialization failures to `EXPERIMENT_VARIANT_FAILED`;
receipt I/O remains `INPUT_CHANGED`. Public worker entrypoints retain purpose/smoke checks
before catalog access; shared admission repeats smoke validation for direct callers.
Submission creates jobs; supervised execution produces staged child evidence.
Neither grants scientific collection or ledger authority.

## RUNUP daily exit comparison

Run `python -m experiments.v2_runup_exits --catalog CATALOG --store-root STORE
--snapshot-id SNAPSHOT --as-of-month YYYY-MM --request REQUEST.json --no-ledger` on a pinned
native snapshot. The command prints one exploratory JSON report (or a refusal)
and exits 0 on success, 2 on refusal. It writes no catalog, ledger or report files.
`--no-ledger` is optional and has no effect; the request must be a regular file.

The closed request has `event_ids` (nonempty unique strings), `fill_alpha` (0–1),
`threshold_units` (exactly `quote_price_times_quantity`) and `arms` (1–8 objects).
Each arm has exactly `arm_id`, positive `target_pnl` and negative `stop_pnl`;
`fixed_day` is reserved for the control. IDs and arm names must be unique.
The existing STR-RUNUP D−14 entry and pre-print exit are fixed; this tool does not
choose a cohort or search thresholds. All events need all held-leg daily marks.

`runup_exit_report.compare_runup_exits(repository, snapshot, *, as_of_month,
document)` exposes the same read-only comparison. Reports include every arm's
entry cost, exit proceeds, mark-based P&L and paired difference from fixed-day,
with contracts, exit reason, ambiguity and reproducibility pins. Thresholds and
P&L are quote-price × quantity units, not percentages or contract dollars;
bid/ask spread is modeled but commissions, fees and ×100 are not. Daily marks
cannot establish intraday stop execution or a trading advantage.

## Scientific sweep collection

After sealing a roster, `python -m experiments.v2_gate_collect run` accepts the
`--roster`, `--catalog`, `--store-root`, `--code-root`, `--as-of-month` and either `--no-ledger` or
`--ledger PATH`, with optional `--scope` (default `shadow`), repeatable `--event-id`
and `--purpose` (only `sweep`). It submits the exact sealed jobs, waits for the ordinary running
ops supervisor (up to `--timeout-seconds`, default 3600), and collects terminal
attempts. `collect` submits/revalidates and collects immediately. Neither command
starts or stops a supervisor; timeout leaves workers/jobs untouched.
Both print an index plus its immutable artifact reference when terminal; exit 0
requires every arm to succeed, while pending/incomplete results and refusals exit 2.

`native_sweep_collection.collect_sweep` owns collection; `v2_gate_collect.run_sweep`
adds only waiting on the ordinary supervisor. The collection intent binds mode, ledger
address, sealed roster and collector sources before reading results. Recorded
mode publishes each supported native outcome and successful report, using the
existing replay-keyed CSV sink. Smoke publishes sweep metadata/report artifacts
only, never native outcome/smoke slots, exported reports or CSV rows. Modes have
distinct collection run IDs; their independent recording policy does not change
or resubmit the already sealed child jobs. Repeating either API resumes partial
publication without evaluating or counting any variant twice. A terminal failure
retains completed siblings and permanently leaves that mode's index incomplete.

Immutable roster/source/mode/destination conflicts refuse R6 with recovery context.
Restore the original installed sources and ledger address to replay, or seal a new
experiment identity; existing intent and failed `(id, spec_hash)` CSV rows never change to success.
