# Native experiment interfaces

These unlayered entrypoints compose the v2 platform. See [ARCHITECTURE.md](ARCHITECTURE.md)
for registration, single-arm evaluation and staged sweep contracts.

## Shared APIs

The owning modules expose these helpers to native experiment callers; no compatibility aliases are provided.

| Owner and public function | Inputs, result and effects |
| --- | --- |
| `magnitude_report.magnitude_metrics` | Complete scored rows yield stable pooled RMSE, MAE and count; empty rows or nonfinite metrics refuse `EXPERIMENT_VARIANT_FAILED`. Fixed and history reporters, the search recorder and the outer evaluator (fold and aggregate metrics) share this scorer. |
| `magnitude_search_evidence.admit_fold_inputs` | Repository, registered binding and fold yield protected preflight, stored feature identity/rows and qualified training targets without writes. History drift raises the caller's `history_failure`; native admission errors propagate. |
| `magnitude_search_evidence.fit_evidence` | Admitted inputs, fold, horizons and execution columns yield ordered per-horizon event/matrix identities without fitting or I/O. |
| `magnitude_search_evidence.normalize_forecasts` | Forecast rows, fold and horizons yield ordered complete finite nonnegative event/horizon predictions. Default Real/Integral admission supports inner submissions; `strict_json=True` requires JSON-native numbers. Malformed input raises structural/numeric exceptions for caller-owned refusal mapping. |
| `magnitude_search_evidence.validate_targets` | Preflight, fold, target receipt and rows must match protected membership, cutoff, recipe and finite nonnegative targets; returns None without I/O. Default Real support and optional strict-JSON validation preserve caller policies; malformed evidence raises structural/numeric exceptions. |
| `magnitude_search_errors.artifact_failure` / `SearchArtifactError` | Artifact error plus caller-owned messages yields a redacted `OpsError` with identity/resource policy; the subtype identifies artifact-origin failures without parsing messages. Journal reference syntax and publication `UNSAFE_PATH` are identity conflicts; read-time `UNSAFE_PATH` after syntax admission is a retryable resource failure, with no attempt-retry authorization. Plan publication retains its resource policy. |
| `magnitude_search_trials.SearchTrials` | Verified plan/anchor, catalog/store and native repository; `seal(clock=...)`, `start(outer_year, inner_year, candidate_id)`, `record(..., forecasts=... or failure=Problem)` and `freeze()` persist immutable all-attempt evidence and selections from submitted forecasts. Candidate terminals are guarded here by stage and canonical `EXPERIMENT_VARIANT_FAILED` admission. A native-derived validation cohort independently binds terminal targets and the complete receipt to admitted preflight/cutoff/recipe; replay never reopens labels. This detects inconsistent journal evidence, not wholesale fabrication of all evidence. These submitted-forecast methods do not verify execution or fit models; native `execute` is described below. No outer evaluation or final authority. |
| `SearchTrials.execute` | One declared candidate/inner fold plus explicit CPU-time, wall-time and address-space limits runs fresh native horizon fits in a fixed child. Immutable fit admission/completion counts and matrix/forecast digests feed the existing journal; completed replay launches nothing. Resource/shared failures stay unfinished, and no outer-evaluation or final-read authority is added. |
| `SearchTrials.reconcile` | The same trial and exact limits finish missing publication only from its immutable post-exit `execution-result` record. Verifies the original admission, fit markers, result and forecast bytes; never fits, reclaims a start or substitutes caller forecasts. Missing/incomplete results refuse. Existing validation is reused, otherwise native admission precedes label reads. Completed replay is read-only under both no-fit guards; original source/plan identity still applies. |
| `magnitude_search_execution.validate_execution_limits` / `execution_limits` | Admit exactly three positive platform-representable integer limits without effects; the latter also selects the lowest currently allowed Linux CPU or refuses `RESOURCE_UNAVAILABLE`. Neither claims a trial or reserves host resources. |
| `magnitude_search_execution.run_fitted_trial` | Admitted features/training targets, fold, horizons, recipe, resolved limits and a progress callback drive the fixed internal `magnitude_search_worker` child. Requires exactly the validated resource trio plus a non-Boolean integer `cpu_id` in `[0, 2**31 - 1)`; malformed limits refuse `INVALID_EXPERIMENT_SPEC` before input access or effects. Both no-fit guards apply; bounded I/O and child resource limits constrain execution, and each fit marker awaits the callback before acknowledgement. Returns the child outcome; always reaps the child and removes scratch inputs. The worker is an internal protocol entrypoint, not a public caller API. |
| `magnitude_fitted.fitted_forecasts` | Existing admitted matrices and recipe yield forecasts/evidence. Optional progress callbacks bracket each fit; bounded execution opts into propagating resource/infrastructure errors. Existing callers retain their previous failure mapping. |
| `native_registration.prepare_native_registration` | A validated spec, repository, source root and holdout context produce `(NativeRegistration, normalized_spec)`; resolves one scope pin or verifies an explicit `SnapshotRef`, reads eligible metadata only, and writes nothing. |
| `native_registration.native_registration_exists` | An expected `NativeRegistration` gives false when absent, true for exact verified catalog/object evidence, or `EXPERIMENT_IDENTITY_CONFLICT`; read-only. Optional `max_evidence_bytes` bounds stored catalog/object JSON through the shared ops verifier; default None preserves prior policy. |
| `native_registration.verify_native_registration` | Catalog/store and a `NativeRegistration` verify stored admission, returning None without writes or current-request revalidation. Wrong type, absent/conflicting registration or malformed/oversized bounded evidence refuses `EXPERIMENT_IDENTITY_CONFLICT`; optional `max_evidence_bytes` forwards the catalog/object JSON bound, while default None preserves existing limits. Foundation `ArtifactError` is converted to `EXPERIMENT_IDENTITY_CONFLICT`; raw `OSError`/`sqlite3.Error` propagate. |
| `native_registration.reserve_native_registration` | Accepts the matching prepared registration/spec pair from the preparer; publishes complete binding bytes and reserves catalog rows, obeying any sweep seal. Exact replay is idempotent; conflicts refuse R6. Publication may leave an unreferenced complete object if reservation fails. |
| `native_registration.read_native_gate_sweep` | Catalog/experiment ID yields stored text or None for absence. SQL atomically caps returned bytes at 16 MiB (32 MiB for UTF-16 catalogs), then decoding enforces 16 MiB UTF-8; wrong type, invalid encoding or oversize gives `EXPERIMENT_IDENTITY_CONFLICT`. Preparation, sealing and reservation share it; SQLite internals/decoded-object overhead are not capped. |
| `native_registration.decode_native_gate_sweep` | Roster JSON text yields an object after a 16 MiB UTF-8 cap; malformed/oversized stored rows refuse `EXPERIMENT_IDENTITY_CONFLICT`. The new-seal writer requests `INVALID_EXPERIMENT_SPEC` before insertion. No writes or identity normalization. |
| `native_gate_worker.read_native_staged_input` | A path yields bounded bytes from a no-follow regular-file descriptor; malformed files refuse `INVALID_EXPERIMENT_SPEC` and I/O errors propagate to the caller's typed boundary. |
| `native_gate_worker.admit_native_gate_inputs` | Staged parameters/root, read-only catalog and store yield `(spec, registration)` only after exact identity, source and environment admission. No evaluation or writes; this does not replace supervisor launch verification. |
| `native_gate_worker.write_native_gate_receipt` | A destination and document install complete canonical bytes create-only; existing destinations survive. Serialization/I/O errors propagate; only private temporary names are cleaned. |
| `native_gate_worker.native_gate_data_failure` | A `DataError` yields a redacted `OpsError` for pricing/evaluation: existing ops codes retain their policy; data-only codes map by registered category/retryability, retaining the original code only in private details. |
| `v2_gate_sweep.register_sweep` | Catalog connection/store/repository, closed roster, source root, holdout month and optional scope/event IDs yield the sealed roster hash. Captures one common pin, seals immutable evidence and resumes matching arm reservations; no jobs or scientific outcomes. |
| `v2_gate_sweep.submit_sweep` | The same inputs require the existing exact seal and complete registrations; stages bound inputs and atomically returns independent job submission receipts, replaying their IDs. Does not start the supervisor or collect/record results; partial staging artifacts may precede a submission refusal. Optional `max_evidence_bytes` forwards a collector-selected bound to stored registration verification; default None leaves existing callers unchanged. Optional `max_failure_bytes=None` and `receipt_decode_failure_code=None` validate receipt policy before staging and forward it to `submit_graph`. Omitted/None failure budget means 8 MiB; historical recovery can explicitly select a larger finite budget. Collection supplies fixed 1 MiB/R6; default queue/resource/progress decoder exceptions remain unchanged. |
| `native_sweep_collection.collect_sweep` | The same sealed inputs plus exactly one smoke/ledger mode yield an index and terminal artifact reference (otherwise None). Existing nonregular ledger entries refuse `INVALID_EXPERIMENT_SPEC`; absent destinations/parents remain valid. The no-follow leaf metadata check is preflight, not replacement-race protection; raw metadata I/O remains retryable. Requires an existing directory for `code_root` (`INVALID_EXPERIMENT_SPEC` otherwise); root-stat I/O remains retryable. Rechecks the import-time installed-source snapshot and caller coordinator entry files before submission. Recognized propagated source-read I/O interruptions are retryable `RESOURCE_UNAVAILABLE`; other unverifiable-source failures refuse R6, including a failed import baseline or known installed drift. Verifies exact attempts and bounded failure/result/native evidence bytes; recorded mode reconciles native outcomes/CSV/report exports, smoke only sweep artifacts. Identity/raw constraint conflicts refuse R6; raw boundary I/O is retryable `RESOURCE_UNAVAILABLE`. Delegated APIs retain their own error policies, including unchanged foundation `ArtifactError` from direct collector-owned intent/smoke-report/index publication (the CLI preserves its code); unexpected programming errors reaching the collector boundary propagate. |
| `native_outcomes.publish_native_outcome`, `replay_native_outcome`, `export_native_report` | Existing native publication/reconciliation APIs accept optional `max_evidence_bytes` for catalog evidence and registration/outcome/completion JSON reads and matching writes. Default None preserves existing caller behavior; collection opts into 16 MiB. Reports are verified at their registered size. With the opt-in bound, existing REPORT.md comparisons open a no-follow regular descriptor, reject sizes above the expected report and read at most its byte length plus one; mismatches refuse R6, storage I/O remains retryable. This is a comparison bound, not the JSON size ceiling for reports. Existing native policy maps structural KeyError/TypeError/ValueError to R6 and RuntimeError to retryable `RESOURCE_UNAVAILABLE`; the collector preserves those delegated refusals. |
| `v2_gate_collect.run_sweep` | Collection inputs and a positive finite timeout observe submitted jobs until terminal/deadline, then collect. Returns the same index shape; never owns workers, retries jobs or stops the supervisor. Observer I/O/non-constraint SQLite failures are retryable `RESOURCE_UNAVAILABLE`; integrity constraints or malformed stored receipts refuse R6 (malformed stored job failures retain `INTEGRITY_FAILED`, while collection's explicit 1 MiB size ceiling refuses R6). Library/CLI timeout defaults are 3600 seconds. |
| `v2_gate_collect.main` | CLI argument list selects collect/run. Non-sweep purpose refuses `HOLDOUT_ACCESS_DENIED`; missing/conflicting mode refuses `INVALID_EXPERIMENT_SPEC`, before opening inputs/catalog. Prints sorted index or typed refusal JSON: returns 0 for complete success, 2 for incomplete results/handled refusals; unexpected programming errors propagate. Requires an existing migrated catalog; missing/nonregular catalog admission refuses `INVALID_EXPERIMENT_SPEC`, other stat errors and raw SQLite failures reaching the CLI remain `RESOURCE_UNAVAILABLE` (the stat check is not race-proof). Rejected roster path/access or bounded-JSON admission refuses non-retryable `INVALID_EXPERIMENT_SPEC`; I/O outside that admission policy remains retryable `RESOURCE_UNAVAILABLE`. Detailed error classifications are covered by the collection boundary tests. Artifact/data refusals retain their original code. |

## Consumers and authority

`v2_magnitude_search_run` consumes `magnitude_search_plan.read_search_plan`, `magnitude_search_records.MIGRATIONS`/`OWNER`/`append_record`/`read_record`, `magnitude_search_trials.SearchTrials`, `magnitude_search_execution.validate_execution_limits`, and `magnitude_search_outer.execute_outer_fold`/`collect_outer_results` for source-sealed serial orchestration.
Its CLI also consumes `native_registration.read_native_registration` and `native_gate_worker.read_native_staged_input`; direct engine dependencies are contracts, data, foundation and ops. The [driver section](#serial-magnitude-search-driver) defines the public seal/run helpers and each CLI action's authority.
`v2_gate_sweep` uses registration preparation/lookup/reservation, the bounded reader and `gate_report.validate_gate_spec`.
`native_sweep_collection` consumes `v2_gate_sweep.submit_sweep`, sealed registration, `native_outcomes`, `gate_report.render_gate_report`, ops transactions/bounded SQL-text helpers/registered artifacts/spec parsing/source closure/bounded native evidence readers and writers/job reads/`MAX_JOB_FAILURE_BYTES`/errors, contracts and foundation. `v2_gate_collect` consumes that collector, staged-input reading, ops catalog/job reads/errors, data `Repository`/errors and foundation artifacts.
`v2_gate_sweep` consumes ops catalog/transactions, checkpoints, errors, spec parsing, fingerprints, migration/schema admission, stage registry and atomic submission; data supplies `Repository`/typed errors, contracts supply jobs/snapshots, and foundation supplies artifacts, clock and canonical serialization.
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
Never replace an existing collection intent. If import-time source capture failed, start a fresh process after restoring source access; the failed process has no trustworthy baseline.
The inherited shared `source_closure` currently refuses `.py` files over 8 MiB with `INPUT_CHANGED`; collector source admission maps this to `EXPERIMENT_IDENTITY_CONFLICT`. Catalog, artifact, result, failure and report comparison bounds remain enforced.

## Selected outer folds

After explicitly completing native inner trials and calling `SearchTrials.freeze`,
call `magnitude_search_outer.execute_outer_fold(conn, store, repository,
registration, request, plan_ref, outer_year, limits=...)`. Limits use the existing
explicit `memory_bytes`, `cpu_seconds`, `wall_seconds` contract. This library API
refits only that year's frozen inner winner on qualified prior labels. Submitted
forecasts alone cannot authorize it. All planned native inner completions,
including permitted candidate failures, are verified before outer execution.

The immutable outer journal owns source bindings, native inputs and full fit evidence.
Returned results contain the selected recipe, evidence IDs, training/evaluation
receipts, actual fit count, complete forecast/target pairs, event/forecast counts
and pooled/per-horizon RMSE/MAE; they do not inline all journal evidence. Forecasts
are durably fixed before that fold's evaluation targets open.
Repeating a completed call verifies the original evidence without fitting or
opening native payloads. Any incomplete outer attempt must remain unresolved;
there is no retry, replacement candidate or recovery API in this slice.

`collect_outer_results(conn, store, repository, registration, request, plan_ref)`
requires all outer folds complete and returns their verified results plus pooled
and per-horizon event-error metrics. After inventorying every outer terminal and limit,
it verifies the native inner inventory once per call, then independently replays each fold.
No admission is cached across calls. It never launches missing work or chooses a
winner using outer results: `prospective_candidate_id` is the latest fold's inner
selection. Collection returns a dictionary only; it does not persist an aggregate
artifact, export or ledger row. The serial CLI below composes these APIs.
No real-data execution, holdout/final access, ledger publication or release
authority is implied.

`magnitude_search_evidence` supplies shared fold input admission, matrix evidence,
forecast normalization and target-receipt validation. Inner wrappers retain submitted
Real/Integral support and their original refusal envelopes; outer wrappers enforce
JSON-native numeric types. The static dependency is included in source seals.

## Serial magnitude search driver

`python -m experiments.v2_magnitude_search_run` accepts `seal`, `run` or `collect`.
Every action requires `--spec SPEC.json --request REQUEST.json --plan-ref REF.json
--catalog CATALOG --store-root STORE`. Supply the original registered v1.5 spec,
the exact `magnitude_search_request.v1` request, and the `plan_ref` object returned
by `v2_magnitude_search prepare` (not that command's entire output envelope).
Inputs must be regular, nonsymlink-leaf JSON objects of at most 16 MiB; duplicate
keys and nonfinite JSON constants refuse. The catalog and store must already exist.

Run `seal` explicitly once after preparing the plan; identical seals are safe to
repeat. It binds driver sources in a separate immutable journal slot before any
trial or selection. An already sealed empty library journal is allowed; started
library journals cannot acquire a driver binding retroactively. `run` requires
that exact journal and additionally requires positive integer
`--memory-bytes`, `--cpu-seconds` and `--wall-seconds`. These are per-child limits,
not a reservation or total-search allowance. For example, after choosing limits
appropriate for the machine and experiment:

```bash
common=(--spec SPEC.json --request REQUEST.json --plan-ref REF.json \
        --catalog CATALOG --store-root STORE)
python -m experiments.v2_magnitude_search_run seal "${common[@]}"
python -m experiments.v2_magnitude_search_run run "${common[@]}" \
    --memory-bytes "$MEMORY_BYTES" --cpu-seconds "$CPU_SECONDS" \
    --wall-seconds "$WALL_SECONDS" > search-report.json
python -m experiments.v2_magnitude_search_run collect "${common[@]}"
```

`run_search(conn, store, repository, registration, request, plan_ref, *, limits)`
is the same orchestration API. It traverses outer folds, candidates and inner
folds in sealed plan order, one child at a time; then freezes all inner selections,
evaluates each selected outer winner, and collects every completed outer fold.
Valid candidate failures are excluded by the sealed policy; all-candidate failure
refuses. It never chooses a winner from outer scores.

Repeating `run` reconciles publication only when the original native start and
post-exit child result are both durable, using exactly the original limits.
Unfinished inner claims without that result, unfinished outer claims, or premature
selections refuse before fitting. All claimed inner results and completed outer
results are verified before fresh fitting. No reset, claim takeover, fit retry,
replacement candidate or source-binding migration is performed. The installation
must match registration, plan and journal; install the complete implementation
before registering/sealing, rather than reusing older source-bound evidence.

`run` and `collect` exit 0 with one canonical `magnitude_search_outer_report.v1` JSON
document, preserving its `result_id`, all fold predictions, metrics and provenance.
The stdout budget is 64 MiB of UTF-8 including the newline: excess refuses with
`RESOURCE_LIMIT_EXCEEDED` before any report bytes are printed, without truncation
or undoing completed records. Output staging uses disposable temporary storage.
The selected stdout stream is flushed before returning; write/flush errors propagate
without a second document, and staging closes without closing caller-owned stdout.
Typed refusals exit 2 as `{"refused": <Problem>}` with original fields; pre-emission
filesystem/SQLite errors are redacted as `RESOURCE_UNAVAILABLE`. Parser errors
exit 2 on stderr; incidental Python progress also goes to stderr.

`collect` verifies the driver seal, opens the catalog read-only and never starts
or repairs work. The existing report hash covers scientific results; the separate
driver seal, whose hash `seal` returns as `driver_id`, supplies driver provenance.
`seal_search_run(..., clock=...)` exposes the explicit seal as a library API.
`seal` and `collect` reject resource flags. Optional `--no-ledger` is a no-op:
all modes exclude ledger/outcome publication, final holdouts, releases and orders.
