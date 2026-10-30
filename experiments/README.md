# Native experiment interfaces

`iv_crush_search.prepare` / `validate_registration` are registration-only APIs (no CLI/search activation or
production caller), using shared `native_registration.prepare_regression_registration`.
It pins committed source, qualified-panel and IV-target snapshots plus explicit
signed target/clocks; only metadata is read. Both preparation APIs admit specification type/plan fields through the existing resolver before dependent reads; the IV wrapper validates both snapshot-reference shapes before either table-contract lookup. Malformed specs/plans retain `INVALID_EXPERIMENT_SPEC`; malformed references retain native `SNAPSHOT_UNRESOLVED`. The ordinary native reserve/read/verify
APIs persist/replay it; reserve/read apply registration-only `iv_crush_registration` policy by requested
spec runner even for downgraded documents, while generic verification stays target-agnostic. v1.8 requested reserve/read reuse preparation's supported-spec/no-external-input rule; prior size checks retain precedence. Preparation, default v1.8 reserve and low-level artifact reads use the existing 16 MiB cap;
full panel/label qualification remains the later fold-admission responsibility.

`iv_crush_plan.target` / `plan` provide metadata-only signed IV30 temporal planning via the shared admitted population/calendar path. No CLI, production caller, sealed registration, target read or fit is provided; invalid target metadata raises `ValueError`, native admission errors propagate.

These unlayered entrypoints compose the v2 platform. See [ARCHITECTURE.md](ARCHITECTURE.md)
for registration, single-arm evaluation and staged sweep contracts.

## Proposed research designs

[STR-THRU distribution forecasts](STR_THRU_DISTRIBUTION/DESIGN.md) proposes an
isolated conditional-distribution and net-profit comparison, reusing SIZE's causal
boundaries. Design only: no registered experiment, runnable recipe or read authority.

## IV-crush preparation

`iv_crush_prepare.prepare_iv_crush_targets` uses a pinned research workspace and
shared selection controls to reconstruct signed IV30 targets and skip coverage.
It streams daily rows with one-ticker quote lifetime, preserves full metadata,
writes only after successful scan exhaustion, and returns separate source
and target snapshots; no search registration, fitting or provider calls occur.
This opt-in preparation API has no production caller or CLI wiring; only synthetic
tests invoke it in this slice. It does not activate an IV search or experiment run.

## Shared APIs

The owning modules expose these helpers to native experiment callers; no compatibility aliases are provided.

| Owner and public function | Inputs, result and effects |
| --- | --- |
| `size_panel.materialize(repository, source, **request)` | Explicit frozen-policy/history/source/price snapshot request and finite limits publish qualified H/HM/HMO rows and return context plus a feature SnapshotRef. Publishes through the native artifact/catalog path; exact replay reuses the admitted receipt. No fitting or production caller. |
| `size_panel.read(repository, source, feature_snapshot, *, columns=None, **request)` | Same bound request re-admits the exact stored panel, returning serialized feature snapshot/context/rows with a closed projection. No rebuild or producer override; malformed shape/projection, changed source, incompatible artifact, protected access and exceeded limits retain typed data refusals. See the owning [architecture contract](ARCHITECTURE.md#qualified-size-panel-opt-in). |
| `eligible_history.iter_read(repository, source, feature_snapshot, *, columns=None, **request)` | Public installed-code-bound eligible-history admission returning context and a re-iterable lazy row stream. Every iteration enforces exact target/cutoff coverage and stored-query limits; used by SIZE composition, stored reads and receipt replay. |
| `size_search_binding.validate_panel_request` / `prepare` / `prepare_plan` / `verify_installation` / `validator_identity`, `size_preflight_receipt.input_bindings` / `issue` / `admit` | SIZE CLI/planner admit closed requests, installed source/environment and stored panel/folds into new v1.7 registrations or replayed v1.6/v1.7 anchors with v2 plans. Preflight validates once, publishes the bounded CAS shards and an immutable `size_preflight_receipt`; register (`--receipt-ref`) checks the receipt against its own inputs and the current validator identity instead of re-validating, then reserves. Compact bindings remain capped at 16 MiB and referenced v1.7 artifacts at 1 GiB total; invalid requests, drift and exceeded limits retain typed refusals. |
| `size_registration_artifacts.pack_size_registration` / `hydrate_size_registration` / `size_registration_artifact_bytes` / `native_registration.load_registration_document` | Registration preparation writes bounded CAS sections for large ordered SIZE values and stores content-hash descriptors; hydration checks the aggregate 1 GiB reference-byte cap, verifies each shard and recomputes logical artifact hashes before restoring planner fields. Missing, altered or malformed objects refuse `EXPERIMENT_IDENTITY_CONFLICT`; aggregate overflow refuses `RESOURCE_LIMIT_EXCEEDED`. |
| `size_search_binding.fold_inputs` / `execution_for_candidate` | Comparison/search evidence and `DistributionTrials.start` use the former to require an exact reconstructed inner/normalized outer fold before re-admitting stored rows and qualified training targets; trials/outer use the latter to bind a candidate execution dict. No hidden materialization or fitting; admission failures propagate. |
| `size_comparator.COLUMNS` / `native_recipe` / `recipe` / `fit_evidence` / `forecasts` | Binding/comparison/worker consume ordered input names, native/complete detached recipes and causal OLS+MLP forecasts from admitted arrays. Reads installed registry metadata, not market sources; fresh guarded fits or explicit abstention, unchanged raw predictions; invalid recipe/metadata raises `ValueError`, bounded registry refusal raises typed `OpsError`. `fit_evidence(inputs, fold, plan)` binds eligible ordered 14-column arrays, not estimator entry; the full admitted cohort remains in the input manifest. |
| `size_comparison_run.execute` | Outer execution passes sealed plan/fold/limits; returns validated forecasts and persists create-only input/fit/terminal receipts. `replay_only=True` reads exact evidence or refuses, never fits/writes; identity/resource/native failures retain their policy. |
| `size_fit_diagnostics.keys` / `evidence` | Trials/outer select versioned receipt keys and validate child diagnostics against execution recipe/horizons; pure set/dict results, no I/O; mismatches refuse `EXPERIMENT_IDENTITY_CONFLICT`. |
| `size_job_budget.remaining` / `resolved_limits` | CLI/trials/outer/comparison supply an admitted plan; persist the first immutable deadline, then return seconds/clamped child limits. No reset/retry; mismatched evidence/limits or exhausted/backward time refuse. |
| `size_paired_report.paired_report` | SIZE collector passes events, challenger/incumbent forecasts, targets and optional exclusions; returns matched metrics/coverage/bootstrap without I/O/fits. Structural/numeric errors propagate; missing incumbent inputs remain abstentions. |
| `size_search_report.collect` / `resource_report` | SIZE CLI uses verified read-only driver/outer/comparator replay for paired reports; the pure latter reports schedule-dependent maximum counts and configured bounds, not measured feasibility. No missing-work fallback; admission/receipt failures propagate. |
| `distribution_run_request.validate_request` / `distribution_book.validate_book_config` | Validate explicit research request/book policies; detached request and book mappings, no data access or execution. [Contract and units](#distribution-explicit-policy). |
| `distribution_book.funded_book(intents, quotes, *, config, report_end)` | Explicit research callers/tests supply immutable intents, total-USD quotes and a complete reporting horizon; returns detached event dispositions, settlement curve and summary without source/storage I/O. [Book contract](#distribution-premium-funded-book). |
| `distribution_report.paired_report` | Supplied common opportunities and book documents yield paired USD/opportunity summaries and declared calendar/ticker uncertainty; no execution or data access. [Contract](#distribution-paired-report). |
| `magnitude_candidate_recipe.candidate_recipe` | Exact family/parameter keys plus ordered columns and dictionary hash yield the existing closed family recipe; no training, source reads or admission proof. |
| `distribution_candidate_recipe.candidate_recipe` / `validate_candidate_recipe` | One explicit qualified-dictionary H/HM/HMO descriptor yields a detached versioned raw histogram/matched-point recipe and ID; validation refuses altered semantics. Static descriptors are not qualified data, preregistration or fitting authority. |
| `distribution_candidate_recipe.fit_budget` / `trial_identity` | Declared fold counts and explicit child limits yield conservative raw-pair fit/tree counts; opaque context/fold/cohort/budget references plus role/horizon yield an identity only. No source authentication, total-job containment or journal claim. |
| `distribution_class_support.training_support` / `class_histogram` | Exactly 81 bin counts yield support/tree metadata; explicit class labels/probabilities yield the numerical histogram with absent zeros and unchanged tail validation. Degenerate support refuses; no fitting, source reads or smoothing. |
| `distribution_plan.prepare_plan` | Verify distinct distribution v1.0 and source SIZE v1.7 registrations plus the exact native search-plan ref; return a compact distribution plan and the verified source plan. Cross-anchor population/clocks, installed code/environment, recipe and derived raw-pair limits are bound; no registration, source outcomes, fitting or SIZE search execution. |
| `distribution_trials.DistributionTrials` | `seal(clock=...)` binds that plan in the distribution run's existing journal; `start(role, outer_year, inner_year=None)` claims once before central native fold admission and publishes shared cohort shards. `read_admission(...)` verifies stored admission without reopening native outcomes. Detached receipts contain manifest/input IDs and three horizon identities, never fit completion. |
| `magnitude_candidate_fit.fit_candidate` | Both no-fit guards precede inspection; exact schema/model dispatch delegates one horizon to the existing numeric fitter and preserves its complete result and native errors. No source qualification, selection, retry or I/O. |
| `magnitude_report.magnitude_metrics` | Complete scored rows yield stable pooled RMSE, MAE and count; empty rows or nonfinite metrics refuse `EXPERIMENT_VARIANT_FAILED`. Fixed and history reporters, the search recorder and the outer evaluator (fold and aggregate metrics) share this scorer. |
| `magnitude_search_evidence.admit_fold_inputs` | Repository, registered binding and fold yield protected preflight, full stored-input identity, projected participant identity/numeric feature rows and qualified training targets without writes. Every call revalidates all stored rows; unselected corruption still refuses. Slim cohorts get new content hashes and source-bound seals; old seals are not migrated. History drift raises the caller's `history_failure`; native admission errors propagate. |
| `magnitude_search_evidence.fit_evidence` | Admitted inputs, fold, horizons and execution columns yield ordered per-horizon event/matrix identities without fitting or I/O. |
| `magnitude_search_evidence.normalize_forecasts` | Forecast rows, fold and horizons delegate to shared `research.regression_evidence` with fixed magnitude column/domain, yielding unchanged ordered complete finite nonnegative event/horizon predictions. Default Real/Integral admission supports inner submissions; `strict_json=True` requires JSON-native numbers. Malformed input raises structural/numeric exceptions for caller-owned refusal mapping. |
| `magnitude_search_evidence.validate_targets` | Preflight, fold, target receipt and rows must match protected membership, cutoff, recipe and finite nonnegative targets; returns None without I/O. Default Real support and optional strict-JSON validation preserve caller policies; malformed evidence raises structural/numeric exceptions. |
| `magnitude_search_shards.write_section` / `build_manifest` / `manifest_id` / `verify_manifest` / `iter_section` / `LIST_MEMBERS` | Called by `magnitude_search_inputs.load_inputs` and `load_fit_inputs` (`iter_section`) and `publish_inputs` (`write_section`, `build_manifest`, `manifest_id`). Rows with a strictly increasing key publish as canonical content-addressed shards through `ArtifactStore`; the manifest hash is the cohort `inputs_id`; `iter_section` streams one verified shard at a time after validating all shard metadata. Imports foundation, contracts, ops errors and `magnitude_search_errors`. Contract: [ARCHITECTURE.md](ARCHITECTURE.md#sharded-cohort-inputs). |
| `magnitude_search_inputs.publish_inputs` | The one cohort publisher: publishes the shards of an admitted inputs dict and returns `(manifest, inputs_id)`; the caller appends the manifest. Callers: `SearchTrials._start` (`cohort:`), the outer writer (`outer-inputs:`) and `DistributionTrials.start` (`distribution:admission:`). |
| `magnitude_search_inputs.load_inputs` | The one cohort loader: a v2 manifest is verified and streamed shard by shard into the inputs shape, a v1 whole-cohort record is bound by content hash. Consumers: `SearchTrials` (record, reconcile, freeze, `_validation`) and outer replay. Imports foundation, ops errors and `magnitude_search_shards`. |
| `magnitude_search_inputs.load_fit_inputs` | Store, manifest, expected `inputs_id`, evaluation events, horizons and feature columns yield projected `{features, training}` rows, streamed shard by shard from the `training` and `features` sections after the manifest hash check; any mismatch is `EXPERIMENT_IDENTITY_CONFLICT` (retryable store errors propagate; a projected row missing a requested column raises `KeyError`, which the child reports as unavailable). Consumers: `magnitude_search_worker` (the fit child) and `DistributionTrials.read_admission` (stored admission replay). |
| `magnitude_search_errors.artifact_failure` / `SearchArtifactError` | Artifact error plus caller-owned messages yields a redacted `OpsError` with identity/resource policy; the subtype identifies artifact-origin failures without parsing messages. Journal reference syntax and publication `UNSAFE_PATH` are identity conflicts; read-time `UNSAFE_PATH` after syntax admission is a retryable resource failure, with no attempt-retry authorization. Plan publication retains its resource policy. |
| `magnitude_search_trials.SearchTrials` | Verified plan/anchor, catalog/store and native repository; `seal(clock=...)`, `start(outer_year, inner_year, candidate_id)`, `record(..., forecasts=... or failure=Problem)` and `freeze()` persist immutable all-attempt evidence and selections from submitted forecasts. Candidate terminals are guarded here by stage and canonical `EXPERIMENT_VARIANT_FAILED` admission. A native-derived validation cohort independently binds terminal targets and the complete receipt to admitted preflight/cutoff/recipe; replay never reopens labels. This detects inconsistent journal evidence, not wholesale fabrication of all evidence. These methods never fit models. When native execution evidence exists, `record` checks completion status and the submitted forecast digest; terminal replay verifies the stored receipt and execution evidence. Native `execute` is described below. No outer evaluation or final authority. |
| `SearchTrials.execute` | One declared candidate/inner fold plus explicit CPU-time, wall-time and resident-memory (RSS watchdog) limits runs fresh native horizon fits in a fixed child. Immutable fit admission/completion counts and matrix/forecast digests feed the existing journal; completed replay launches nothing. Each native response verifies its execution receipt once, with fresh journal reads on every replay and exact caller limits checked. Resource/shared failures stay unfinished, and no outer-evaluation or final-read authority is added. |
| `SearchTrials.reconcile` | The same trial and exact limits finish missing publication only from its immutable post-exit `execution-result` record. Verifies the original admission, fit markers, result and forecast bytes; never fits, reclaims a start or substitutes caller forecasts. Missing/incomplete results refuse. Existing validation is reused, otherwise native admission precedes label reads. Completed replay is read-only under both no-fit guards and reuses only evidence verified within that same call; separate calls recheck current durable evidence. Original source/plan identity still applies. |
| `magnitude_search_execution.validate_execution_limits` / `execution_limits` | Admit exactly three positive platform-representable integer limits without effects; the latter also selects the lowest currently allowed Linux CPU or refuses `RESOURCE_UNAVAILABLE`. Neither claims a trial or reserves host resources. |
| `magnitude_search_execution.run_fitted_trial` | An admitted cohort source (`manifest`, `inputs_id`, `store_root`), fold, horizons, recipe, resolved limits and a progress callback drive the fixed internal `magnitude_search_worker` child. Requires exactly the validated resource trio plus a non-Boolean integer `cpu_id` in `[0, 2**31 - 1)`; malformed limits refuse `INVALID_EXPERIMENT_SPEC` before input access or effects. Both no-fit guards apply; the child streams the manifest's shards and projects only consumed identity/numeric rows and qualified targets, with no input byte cap (child output stays capped at 16 MiB); bounded I/O and child resource limits constrain execution, and each fit marker awaits the callback before acknowledgement. Returns the child outcome; always reaps the child and removes scratch inputs. The worker is an internal protocol entrypoint, not a public caller API. |
| `magnitude_fitted.fitted_forecasts` | Shared `research.horizon_fitted` runs fixed-nonnegative magnitude horizons; the wrapper preserves exact legacy error policy and evidence. Optional progress callbacks bracket each fit; bounded execution opts into propagating resource/infrastructure errors. Existing callers retain their previous failure mapping. |
| `native_registration.prepare_native_registration` | A validated spec, repository, source root and holdout context produce `(NativeRegistration, normalized_spec)`; resolves one scope pin or verifies an explicit `SnapshotRef`, reads eligible metadata only, and writes nothing. SIZE's wrapper publishes its versioned CAS artifacts on its preflight path (register only reads them via the receipt). Optional `max_evidence_bytes` bounds admission rereads; omitted/None retains this low-level helper's prior policy. |
| `native_registration.valid_registration_read_budget` | Pure predicate shared by read/register/require, optional prepare/reserve bounds, RUNUP registration and CLI parsers: true only for built-in integers from 1 through `2**62 - 1`. Returns false for booleans, None, other types and out-of-range values; no coercion, I/O or exceptions. Entry boundaries own their existing typed refusal or argparse diagnostic. |
| `native_registration.register_native`, `read_native_registration`, `require_native_registration` | Require a finite built-in integer `max_evidence_bytes` in `[1, 2**62 - 1]` (default 16 MiB UTF-8); invalid/None refuses `INVALID_EXPERIMENT_SPEC` before preparation/I/O. Register bounds prepared bytes before publication/reservation; read and require bound catalog transfer/reference size before each registration object load, including require's admission and final existence recheck. Binding overflow refuses nonretryable `RESOURCE_LIMIT_EXCEEDED` with `registration_bytes`/`max_registration_bytes`; oversized catalog evidence, corruption/mismatch remain `EXPERIMENT_IDENTITY_CONFLICT`, absence R1. Reads are write-free; raw storage errors propagate. Preparation allocation is not capped. Existing oversized bindings require an explicit finite override for each public call; seals are not rewritten. |
| `native_registration.native_registration_exists` | An expected `NativeRegistration` gives false when absent, true for exact verified catalog/object evidence, or `EXPERIMENT_IDENTITY_CONFLICT`; read-only. Optional `max_evidence_bytes` bounds stored catalog/object JSON through the shared ops verifier; registration reference overflow propagates `RESOURCE_LIMIT_EXCEEDED`. Default None preserves prior policy. |
| `native_registration.verify_native_registration` | Catalog/store and a `NativeRegistration` verify stored admission, returning None without writes or current-request revalidation. Wrong type, absent/conflicting registration or malformed/oversized catalog evidence refuses `EXPERIMENT_IDENTITY_CONFLICT`; optional `max_evidence_bytes` forwards the bound and registration reference overflow propagates `RESOURCE_LIMIT_EXCEEDED`, while default None preserves existing limits. Foundation `ArtifactError` is converted to `EXPERIMENT_IDENTITY_CONFLICT`; raw `OSError`/`sqlite3.Error` propagate. |
| `native_registration.reserve_native_registration` | Checks the prepared registration against the requested spec hash, canonical resolved execution plan and derived run ID before publishing binding bytes/reserving rows; conflicts refuse R6, including at read-side spec/plan verification. Obeys sweep seals. Optional `max_evidence_bytes` bounds prepared binding/catalog bytes before publication and both stored checks; invalid explicit budgets refuse R1, binding overflow `RESOURCE_LIMIT_EXCEEDED` with byte details, catalog overflow R6. Default None preserves low-level compatibility; public register cannot select it. Exact replay is idempotent. Publication may leave an unreferenced complete object if reservation fails. |
| `native_registration.read_native_gate_sweep` | Catalog/experiment ID yields stored text or None for absence. SQL atomically caps returned bytes at 16 MiB (32 MiB for UTF-16 catalogs), then decoding enforces 16 MiB UTF-8; wrong type, invalid encoding or oversize gives `EXPERIMENT_IDENTITY_CONFLICT`. Preparation, sealing and reservation share it; SQLite internals/decoded-object overhead are not capped. |
| `native_registration.decode_native_gate_sweep` | Roster JSON text yields an object after a 16 MiB UTF-8 cap; malformed/oversized stored rows refuse `EXPERIMENT_IDENTITY_CONFLICT`. The new-seal writer requests `INVALID_EXPERIMENT_SPEC` before insertion. No writes or identity normalization. |
| `native_gate_worker.read_native_staged_input` | A path yields bounded bytes from a no-follow regular-file descriptor; malformed files refuse `INVALID_EXPERIMENT_SPEC` and I/O errors propagate to the caller's typed boundary. |
| `native_gate_worker.admit_native_gate_inputs` | Staged parameters/root, read-only catalog and store yield `(spec, registration)` only after exact identity, source and environment admission. No evaluation or writes; this does not replace supervisor launch verification. |
| `native_gate_worker.write_native_gate_receipt` | A destination and document install complete canonical bytes create-only; existing destinations survive. Serialization/I/O errors propagate; only private temporary names are cleaned. |
| `native_gate_worker.native_gate_data_failure` | A `DataError` yields a redacted `OpsError` for pricing/evaluation: existing ops codes retain their policy; data-only codes map by registered category/retryability, retaining the original code only in private details. |
| `v2_gate_sweep.register_sweep` | Catalog connection/store/repository, closed roster, source root, holdout month and optional scope/event IDs yield the sealed roster hash. Captures one common pin, seals immutable evidence and resumes matching arm reservations; no jobs or scientific outcomes. The complete new roster, containing every binding, is capped at 16 MiB UTF-8 before sealing (`INVALID_EXPERIMENT_SPEC`); per-arm stored checks/reservation retain their low-level None policy. |
| `v2_gate_sweep.submit_sweep` | The same inputs require the existing exact seal and complete registrations; stages bound inputs and atomically returns independent job submission receipts, replaying their IDs. Does not start the supervisor or collect/record results; partial staging artifacts may precede a submission refusal. Optional `max_evidence_bytes` forwards a collector-selected bound to stored registration verification; default None leaves existing callers unchanged. Optional `max_failure_bytes=None` and `receipt_decode_failure_code=None` validate receipt policy before staging and forward it to `submit_graph`. Omitted/None failure budget means 8 MiB; historical recovery can explicitly select a larger finite budget. Collection supplies fixed 1 MiB/R6; default queue/resource/progress decoder exceptions remain unchanged. |
| `native_sweep_collection.collect_sweep` | The same sealed inputs plus exactly one smoke/ledger mode yield an index and terminal artifact reference (otherwise None). Existing nonregular ledger entries refuse `INVALID_EXPERIMENT_SPEC`; absent destinations/parents remain valid. The no-follow leaf metadata check is preflight, not replacement-race protection; raw metadata I/O remains retryable. Requires an existing directory for `code_root` (`INVALID_EXPERIMENT_SPEC` otherwise); root-stat I/O remains retryable. Rechecks the import-time installed-source snapshot and caller coordinator entry files before submission. Recognized propagated source-read I/O interruptions are retryable `RESOURCE_UNAVAILABLE`; other unverifiable-source failures refuse R6, including a failed import baseline or known installed drift. Verifies exact attempts and bounded failure/result/native evidence bytes; recorded mode reconciles native outcomes/CSV/report exports, smoke only sweep artifacts. Identity/raw constraint conflicts refuse R6; raw boundary I/O is retryable `RESOURCE_UNAVAILABLE`. Delegated APIs retain their own error policies, including unchanged foundation `ArtifactError` from direct collector-owned intent/smoke-report/index publication (the CLI preserves its code); unexpected programming errors reaching the collector boundary propagate. |
| `native_outcomes.publish_native_outcome`, `replay_native_outcome`, `export_native_report` | Existing native publication/reconciliation APIs accept optional `max_evidence_bytes` for catalog evidence and registration/outcome/completion JSON reads and matching writes. Default None preserves existing caller behavior; collection opts into 16 MiB. Reports are verified at their registered size. With the opt-in bound, existing REPORT.md comparisons open a no-follow regular descriptor, reject sizes above the expected report and read at most its byte length plus one; mismatches refuse R6, storage I/O remains retryable. This is a comparison bound, not the JSON size ceiling for reports. Existing native policy maps structural KeyError/TypeError/ValueError to R6 and RuntimeError to retryable `RESOURCE_UNAVAILABLE`; the collector preserves those delegated refusals. |
| `v2_gate_collect.run_sweep` | Collection inputs and a positive finite timeout observe submitted jobs until terminal/deadline, then collect. Returns the same index shape; never owns workers, retries jobs or stops the supervisor. Observer I/O/non-constraint SQLite failures are retryable `RESOURCE_UNAVAILABLE`; integrity constraints or malformed stored receipts refuse R6 (malformed stored job failures retain `INTEGRITY_FAILED`, while collection's explicit 1 MiB size ceiling refuses R6). Library/CLI timeout defaults are 3600 seconds. |
| `v2_gate_collect.main` | CLI argument list selects collect/run. Non-sweep purpose refuses `HOLDOUT_ACCESS_DENIED`; missing/conflicting mode refuses `INVALID_EXPERIMENT_SPEC`, before opening inputs/catalog. Prints sorted index or typed refusal JSON: returns 0 for complete success, 2 for incomplete results/handled refusals; unexpected programming errors propagate. Requires an existing migrated catalog; missing/nonregular catalog admission refuses `INVALID_EXPERIMENT_SPEC`, other stat errors and raw SQLite failures reaching the CLI remain `RESOURCE_UNAVAILABLE` (the stat check is not race-proof). Rejected roster path/access or bounded-JSON admission refuses non-retryable `INVALID_EXPERIMENT_SPEC`; I/O outside that admission policy remains retryable `RESOURCE_UNAVAILABLE`. Detailed error classifications are covered by the collection boundary tests. Artifact/data refusals retain their original code. |

`v2_gate`, `v2_prediction`, `v2_magnitude_search` and `v2_magnitude_search_run` expose `--max-registration-bytes`
(default 16777216) for initial catalog/object loading. Gate and prediction also
apply it to prepared registration bytes before publication/reservation and both
current-request registration rechecks (including magnitude admission). This flag
does not impose a new budget on later outcome/report APIs or worker paths. Larger
populations may need an explicit finite override; history binds its temporal plan twice.
Saved identity and later outcome/worker APIs' separate read policies do not change.

## Consumers and authority

`distribution_report.paired_report` is currently imported only by `tests/v2/research/test_distribution_report.py`; it is library-only and unreachable from a production entrypoint. D4 will compose its forecast report, with runner wiring in later D slices.

`record_export` (run-record export) is consumed by `v2_gate` and `v2_prediction` `run`; it depends on `lib` (EXP numbering) and ops errors only.

`v2_magnitude_search_run` consumes `magnitude_search_plan.read_search_plan`, `magnitude_search_records.MIGRATIONS`/`OWNER`/`append_record`/`read_record`, `magnitude_search_trials.SearchTrials`, `magnitude_search_execution.validate_execution_limits`, and `magnitude_search_outer.execute_outer_fold`/`collect_outer_results` for source-sealed serial orchestration.
Its CLI also consumes `native_registration.read_native_registration`/`valid_registration_read_budget`/`MAX_NATIVE_EVIDENCE_BYTES` and `native_gate_worker.read_native_staged_input`; direct engine dependencies are contracts, data, foundation and ops. The [driver section](#serial-magnitude-search-driver) defines the public seal/run helpers and each CLI action's authority.

`magnitude_search_request.compile_request` returns source-free stage-2 request
specifications and year/count previews for the full-HMO v1 dictionary.
`magnitude_search_request_versions.compile_request` dispatches versioned dictionaries via public validation/panel recipes; the qualified SIZE binding/CLI consume it while legacy previews retain their outputs/refusals.
It preserves the closed H/HM/HMO 33-candidate grid; compilation alone grants no source or fit authority.
`magnitude_candidate_recipe` and `magnitude_candidate_fit` are consumed by qualified plan construction and the bounded worker, as well as synthetic tests.
They reuse public model recipe/fitter APIs; admitted stored inputs and the sealed execution route, not compiler specifications alone, authorize fitting.

`distribution_candidate_recipe` is consumed by the distribution plan/journal adapters and synthetic tests; `distribution_class_support` currently has synthetic consumers only.
Bundle B uses a separate distribution-owned native metadata run and references the qualified SIZE registration/plan for source and folds only. It neither runs SIZE's 33 candidates nor writes distribution claims under the SIZE run. Existing v1.7/CAS/native admission own transport and central holdout policy; no panel/context payload is embedded in the distribution plan.
Their [contract](ARCHITECTURE.md#distribution-declarations) keeps raw uncalibrated policy, conservative tree counts and opaque-reference limitations explicit. Synthetic checks: `python -m pytest tests/v2/research/test_distribution_candidate_recipe.py tests/v2/research/test_distribution_class_support.py`.
The [admission contract](ARCHITECTURE.md#distribution-native-admission) describes immutable input claims, not completed trials or a runnable experiment. Bundle C's causal OOF controls and D's genuine runner/bounded distribution-output dispatch remain prerequisites. Focused synthetic catalog tests: `python -m pytest tests/v2/research/test_distribution_plan_trials.py tests/v2/research/test_distribution_native_admission.py`.

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
`threshold_units` (`quote_price_times_quantity` or `entry_debit_fraction`) and `arms` (1–8 objects).
Each arm has exactly `arm_id`, positive `target_pnl` and negative `stop_pnl`;
`fixed_day` is reserved for the control. IDs and arm names must be unique.
The existing STR-RUNUP D−14 entry and pre-print exit are fixed; this tool does not
choose a cohort or search thresholds. All events need all held-leg daily marks.

`runup_exit_report.compare_runup_exits(repository, snapshot, *, as_of_month,
document)` exposes the same read-only comparison. Reports include every arm's
entry cost, exit proceeds, mark-based P&L and paired difference from fixed-day,
with contracts, exit reason, ambiguity and reproducibility pins. Fractional thresholds
are signed ratios of each position's positive net opening debit at declared alpha,
not gross premium, percentage points or portfolio returns. They are converted once;
reports include that basis and effective thresholds. Zero/credit entries and unusable
conversions refuse. Absolute requests retain their existing report shape. P&L always
uses quote-price × quantity; bid/ask spread is modeled but commissions, fees and ×100 are not. Daily marks
cannot establish intraday stop execution or a trading advantage.

### Registered single-candidate RUNUP comparison

`python -m experiments.v2_runup_study register` seals one candidate before any
market-data read; `run` re-admits that seal, evaluates or replays, and publishes
immutable no-ledger JSON evidence. Both require `--spec SPEC.json --catalog CATALOG
--store-root STORE --snapshot-id SNAPSHOT --scope SCOPE --as-of-month YYYY-MM`
and repeatable `--event-id ID`. The catalog must already exist; no implicit migration
or current-head substitution is performed. Original spec and exact arguments are
required for replay, even after the scope head moves. The installed source closure
and numerical environment must still match; changed code needs a new registration.

Use the native `ExperimentSpec` document with runner `experiments/v2_runup_study.py`,
a nonblank hypothesis, exactly one named primary arm, seed 0, empty folds and no
input files. Economics are exactly `fill` and `exit`; exit requires `kind: target_stop`,
`trading_days: 14`, explicit `threshold_units` and signed `target_pnl`/`stop_pnl`.
The cohort and thresholds are caller choices. The fixed-day control is implicit
in this runner's contract, not a second attempted variant. No fitting is performed.

`registered_runup.comparison_request` validates without I/O; `register_runup` and
`run_runup` accept catalog/store/repository/spec plus explicit `snapshot` (SnapshotRef),
`scope`, `as_of_month` and `event_ids`. Registration reads canonical event metadata
only. Run rechecks the entire binding before outcome access; terminal replay reads
metadata/evidence but no market payloads. Missing common-horizon marks fail the whole
candidate, with no successful report or survivor subset. Supported scientific failures
become immutable terminals; resource interruptions leave publication resumable.

After argument parsing, the CLI emits one JSON document and exits 0 on success, 2 on refusal (including replayed scientific failure). Argparse errors exit 2 with stderr diagnostics and empty stdout. Run returns
`variant_id`, `run_id`, `replayed`, `outcome`, `receipt` and `receipt_ref`;
`receipt.report_ref` identifies canonical comparison JSON in the artifact store.
Its existing exploratory report schema/mode stays unchanged; the enclosing native
registration/outcome establishes the registered execution. No `REPORT.md` export or
CSV ledger is written; optional `--no-ledger` is an accepted no-op. Both actions forward
`--max-evidence-bytes` (default 16 MiB): registration preflights new binding/catalog bytes
and bounds both stored-evidence checks; run bounds registration/outcome JSON admission.
The public RUNUP helpers require a positive integer budget; None never disables it.
This does not cap market scans or report size; spec files are capped at 16 MiB.
This is reconstructed-history selection evidence, with the daily-mark economics above.
It does not establish OOS validation, as-known prices, threshold-search completeness
or release authority. Multi-arm selection and actual study design remain separate.

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

These three serial-search actions accept `--max-registration-bytes` (default 16777216); use an explicit larger finite budget for a larger saved registration. The shared loader validator requires an integer from 1 through `2**62 - 1` before storage access; here the option affects initial registration loading, not saved identity or later read policies. Prediction/gate registration also applies it before reservation, as described above.

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
The selected stdout stream consumes each chunk using reported progress, then flushes before returning.
Zero/unavailable/invalid progress and write/flush errors propagate without a second document; staging closes without closing caller-owned stdout.
Typed refusals exit 2 as `{"refused": <Problem>}` with original fields; pre-emission
filesystem/SQLite errors are redacted as `RESOURCE_UNAVAILABLE`. Parser errors
exit 2 on stderr; incidental Python progress also goes to stderr.

`collect` verifies the driver seal, opens the catalog read-only and never starts
or repairs work. The existing report hash covers scientific results; the separate
driver seal, whose hash `seal` returns as `driver_id`, supplies driver provenance.
`seal_search_run(..., clock=...)` exposes the explicit seal as a library API.
`seal` and `collect` reject resource flags. Optional `--no-ledger` is a no-op:
all modes exclude ledger/outcome publication, final holdouts, releases and orders.

Native inner recording normalizes each successful forecast submission once within
the recording call, reusing that local value for its completion digest and publication.
This adds no cross-call cache: fresh reads, validation order and receipt verification remain unchanged.

## Qualified SIZE v2 development entrypoint

The qualified-panel route is additive: old v1 search and source-free v1/v2
previews are not silently upgraded. It requires an explicitly built v3 panel,
its exact eligible-history artifact and a fresh native v1.7 experiment/arm.
This is a causal research refit, not byte-identical production reproduction.
Historical publication/event-schedule knowledge remains reconstructed/unverified.
No command below authorizes real data access, providers, registration, execution,
final holdouts, release or merging. The user separately authorizes local data runs.

**Developer synthetic checks (no private datasets):**

```sh
python -m pytest tests/v2/research/test_size_comparator.py tests/v2/research/test_size_paired_report.py
python -m pytest tests/v2/research/test_size_search_integration.py
```

The integration fixture creates synthetic native sources, materializes both
qualified artifacts, executes 33 candidates on one inner fold (102 search fits
including the outer refit), and checks deterministic no-refit replay. Its tiny
cohort correctly makes the 500-row incumbent unavailable; separate synthetic
500-row numerical tests establish native OLS+MLP parity, while report tests
establish exact matched metrics and clustered bootstrap arithmetic.

**User-local preparation:** inspect `SUPER_SIZE/request.v2.example.json` and
`SUPER_SIZE/spec.v2.example.json`. These concrete syntax fixtures use synthetic
identity tokens, not usable data references or recommended resource budgets.
Replace every snapshot/plan token with exact locally admitted artifacts, choose
an unused experiment/arm, and measure resource limits in a separately authorized
pilot. Keep fixed H/HM/HMO grids, 1/3/5 clocks and 30-day policy. The three-outer,
six-inner example requests 603 search fits, 9 comparator logical/18 component
fits, at most 612 logical/621 component fits total; another schedule differs.

Write `panel-request.json` with the exact original public `size_panel` request:
`feature_snapshot_id`, `history_snapshot` and `price_history_snapshot` are native
snapshot ID strings; retain `as_of_month`, `policy`, `event_ids`, `horizons`,
`history_start`, `source_start`, `limits` and `metadata_limits` from materialization.
Do not pass a producer ID. Both arms consume this same stored clean history.
No builder runs from the search worker. Preserve actual label anchors and frozen
membership; do not relabel annual evaluation as protected confirmation.

```sh
COMMON="--spec spec.json --request request.json --catalog /local/catalog.sqlite --store-root /local/store --no-ledger"
PREP="--panel-request panel-request.json --scope frozen-development --training-start 2010-01-01 --labels-before 2025-01-01"
python -m experiments.size_search preflight $COMMON $PREP > preflight.json
python -c 'import json; print(json.dumps(json.load(open("preflight.json"))["receipt_ref"]))' > receipt-ref.json
# Only after separate local registration/execution authorization:
python -m experiments.size_search register $COMMON $PREP --receipt-ref receipt-ref.json > registered.json
python -c 'import json; print(json.dumps(json.load(open("registered.json"))["plan_ref"]))' > plan-ref.json
python -m experiments.size_search seal $COMMON --plan-ref plan-ref.json
python -m experiments.size_search run $COMMON --plan-ref plan-ref.json > comparison.json
python -m experiments.size_search report $COMMON --plan-ref plan-ref.json > verified-comparison.json
```

Preflight checks actual stored admission, schedule feasibility and label windows,
not a memory/performance promise. CLI parent resident memory (RSS watchdog) and
parent/job wall ceilings are enforced; child CPU/resident memory/affinity/wall are
contained by the existing worker. Child memory must not exceed parent memory. Library callers
must provide parent containment themselves. Build limits are not consumed or
measured here because artifacts must already exist. Reports state these limits.
Progress/RSS goes to stderr; stdout is JSON. No `--ledger` mode exists.

Every candidate attempt precedes selection; infrastructure failure blocks it.
Completed runs replay exact evidence without refitting. An interrupted start
without durable result is not retry permission: preserve the journal and diagnose
it; never reset claims or raise caps automatically. Changed bindings require new
artifacts/experiment identity. Comparator masks can cause explicit abstention;
no history-mean substitution conceals an unavailable incumbent. Main pairing
requires all three horizons; broad coverage, exclusions and history-mean metrics
remain visible. Bootstrap intervals do not erase prior adaptive design reuse.

The first execution stores one immutable job deadline in the existing journal.
The legacy search CLI rejects qualified requests; use the SIZE entry point above.
Registration and replay share the native 16 MiB evidence cap; reports require the exact driver seal.
Resuming cannot reset it or enlarge child limits; completed read-only reports
remain available after expiry. Inner ElasticNet convergence, coefficients and
selected-column evidence are included in the report and retained in bounded
journal receipts; `magnitude_search_records.read_record(conn, store, run_id, slot)`
reads the reported execution slot with the existing 16 MiB record bound.

Start a fresh process after code changes: installed source identity is captured
at import and checked before preparation/replay, refusing subsequent tree drift.
This assumes stable process startup, not proof of arbitrary stale prior imports.

The panel-request wire envelope is closed and typed before source/catalog access.
Source seals include the dynamic fit-worker entrypoint and its estimator imports.
Child jobs count inner candidates + outer challengers + outer comparators:
the 603-search-fit example has 204 children, even if every comparator abstains.


## SIZE real-data preparation (`size_prep_inputs`)

Library-only helpers for preparing SIZE inputs inside a caller-supplied write root. They grant no data-run, registration, fitting or final-holdout authority; the user authorizes local runs separately. Holdout, embargo and fold-cutoff controls are never reimplemented here: they stay in `ExperimentHoldouts` (3% random plus rolling six months), `eligibility_policy` (`grouped_ticker_embargo.v1`, 30 days) and the registration's metadata folds, consumed by `eligible_history` and `size_panel`.

**`tier1_history_manifest.v1`** `{schema_version, session_ceiling, entries, manifest_id}`; each entry is `{ticker, key, fetched_at, body_sha256}` (sorted by ticker, one per ticker). `manifest_id` is `content_hash` of the other fields. `build_manifest(source_root, *, session_ceiling, tickers, allow_exclusions=False)` pins, per explicit ticker, the entry the existing Tier-1 rule would pick (newest successful `yfinance` `history period=max`); `allow_exclusions` must be a boolean. `pinned_history_fetcher(source_root, manifest)` returns a no-network fetcher with the `tier1_yfinance_history_fetcher` shape; it serves only pinned entries and re-verifies on construction that each key still exists with the same `fetched_at`, status 200 and body hash, and `fetch` re-hashes the bytes it serves (`INPUT_CHANGED` on mismatch). Outside the manifest the default fetcher returns `legitimate_empty` (never newest-wins); strict mode refuses unpinned refresh targets, except that v2 admits empty history only for its explicit exclusions. Legacy `generate_bounds` rejects v2, so exclusions cannot enter that write-root path.

**Workspace exclusion admission.** `build_manifest(..., allow_exclusions=True)` emits `tier1_history_manifest.v2` with a sorted `excluded_tickers` list for requested tickers without a Tier-1 entry. Before reading the workspace snapshot, `admit_manifest(workspace, scope, manifest, *, clock)` calls `resolve_size_scope_snapshot` to validate the workspace handle and scope; invalid inputs refuse `WORKSPACE_REFUSED` without source reads or writes, while unexpected snapshot-read failures map to `WORKSPACE_REFUSED` / `storage_failure`. It checks exclusions against the pinned snapshot's `earnings_events` table with a 64 MiB pre-decode object cap per fragment before any admission writes. After event checks, it records a workspace-local scope-head pin, the manifest object and the admission read pin. An excluded ticker with an event, or an exclusion list when that table is absent, refuses `PUBLICATION_REFUSED`; an oversized event fragment refuses `RESULT_LIMIT_EXCEEDED`. The legacy write-root `generate_bounds` rejects v2 with `WORKSPACE_REFUSED`.

**Two dates, never merged.** `session_ceiling` bounds the market sessions used: it must equal the refresh `as_of`, which truncates each series. `fetched_at` is the retrieval vintage of the pinned bytes and may be later than the ceiling. Close prices are split-adjusted as of retrieval, so a late `fetched_at` is reconstructed provenance, not a point-in-time vintage.

**`generate_bounds(...)`** runs `run_computed_moves_refresh(..., include_source_bounds=True, use_cached_receipts=False)` with the pinned fetcher against `<write_root>/catalog.sqlite` (must exist and resolve directly inside `write_root`, opened with the migrating `open_catalog`); `write_root` must be a literal non-symlink path and unrelated (equal, ancestor or descendant) to every path in the required `production_roots` argument, which must be non-empty; the object root is `write_root` itself; its input document and the supervisor lock are written there only. The refresh regenerates `computed_moves` for the snapshot's own refresh targets (each must be pinned; an unpinned target refuses before commit), adds the `computed_move_bounds.v1` sibling table, and commits one new snapshot that advances the selected scope's head in that write-root catalog (catalog rows and objects included); it writes nothing outside `write_root`.

**`size_prep_provenance.v1`** `{schema_version, label: "reconstructed", refresh_noop, session_ceiling, manifest_id, manifest_entry_count, entry_count, earliest_fetched_at, latest_fetched_at, entries_fetched_after_ceiling, source_snapshot_id, bounds_snapshot_id, record_id}` is written to `<write_root>/size_prep/provenance-<hash>.json` (atomic rename) and returned. `entry_count`, `earliest/latest_fetched_at` and `entries_fetched_after_ceiling` describe only the pinned entries the refresh actually served (`manifest_entry_count` is the whole manifest); a no-op rerun that re-fetches the same bytes reports them for that run. `source_snapshot_id` is the pre-refresh parent; `bounds_snapshot_id` is the refresh result (equal to the parent, with `refresh_noop: true`, if the refresh committed nothing; this is accepted only when the head already carries source bounds, so a rerun after an interrupted earlier run records the parent that already holds the bounds, not the original pre-bounds source).

| Failure | Code |
|---|---|
| Malformed manifest, ceiling, ticker list, non-boolean `allow_exclusions`, or `catalog.sqlite` not directly inside `write_root` | `INVALID_REQUEST` |
| Requested ticker has no eligible Tier-1 entry at build; or, in `generate_bounds` (strict fetcher), a refresh target not pinned in the manifest (refused during capture, before any commit) | `SOURCE_NOT_FOUND` |
| Empty manifest without an explicit nonempty v2 exclusion list | `INVALID_REQUEST` |
| Invalid workspace handle or scope during admission (before source reads or writes) | `WORKSPACE_REFUSED` |
| Unexpected workspace snapshot read failure during admission | `WORKSPACE_REFUSED`, `storage_failure` |
| Pinned event fragment exceeds the 64 MiB scan object cap (before decoding that fragment) | `RESULT_LIMIT_EXCEEDED` |
| Excluded ticker has an event, or exclusions cannot be checked because the pinned snapshot lacks `earnings_events` | `PUBLICATION_REFUSED`, `excluded_ticker_has_events` / `event_coverage_unavailable` |
| `source_root` not an existing directory (`build_manifest`, pinned fetcher, `generate_bounds`) | `INVALID_REQUEST` |
| Sidecar or cache body not a regular file (symlink, FIFO), sidecar above `MAX_META_BYTES` (1 MiB), an unreadable file, or a corrupt gzip body; each file is opened once with `O_NOFOLLOW|O_NONBLOCK`, checked with `fstat` and read through a byte cap (non-JSON sidecars, and sidecars whose `key` differs from the file name, are skipped as not entries). Scans and generation log bounded progress (every 100,000 sidecars: count, rate, ETA bound, RSS) to stderr | `SOURCE_INVALID` |
| `write_root` a symlink path, empty `production_roots`, or `write_root` equal to, containing or contained by a production root; unhashable or non-string ticker entries | `INVALID_REQUEST` |
| Refusals raised by the shared refresh (`computed_moves_command`: missing scope head, held supervisor lock, input or contract mismatches) | propagated unchanged |
| Refresh was a no-op and the head carries no source bounds | `PUBLICATION_REFUSED` |
| Decompressed cache body above `MAX_BODY_BYTES` (64 MiB), or more than `MAX_SIDECARS` (2,000,000) sidecars scanned | `RESOURCE_LIMIT_EXCEEDED` |
| Pinned entry missing, `fetched_at`, status or body hash changed | `INPUT_CHANGED` |
| `as_of` differs from `session_ceiling` | `INVALID_REQUEST` |
| Another supervisor holds the write-root catalog | `RESOURCE_UNAVAILABLE` |

## Distribution premium-funded book

`distribution_book.funded_book(intents, quotes, *, config, report_end)` consumes unique event/ticker/date intents with an immutable selected flag and separate total-USD entry/exit quotes. The caller owns native clocks, contract identity, multiplier/quantity, quote quality and all costs; the book checks the supplied quantity/multiplier and limits but does not price or authenticate quotes. It returns `distribution_book.v1` with per-event dispositions, settlement curve and summary, leaving inputs unchanged. Invalid supported inputs/arithmetic refuse `INVALID_EXPERIMENT_SPEC`; there is no source/storage I/O, source admission, retry or partial publication. After those checks it projects normalized opportunities into the [shared premium-settlement core](../engine/v2/evaluation/ARCHITECTURE.md), maps bucket refusals back to `ticker_cap`, and rejoins detached event/quote metadata. Core `CONTRACT_MISMATCH` retains the book’s existing fixed error envelope; other failure codes and unexpected errors propagate. The adapter still owns bounded stage/action progress.

Premium/capacity remain reserved through calendar-day settlement, not merely exit. Explicit same-time ordering and lexicographic event IDs resolve competition; no fractional contracts. Entry debit includes supplied entry costs. `report_end` must cover every supplied scheduled exit and its settlement date: either date beyond it refuses the whole input, even with missing exit cash. This is a complete-report-horizon contract, not an interim report of trades scheduled to close later. Missing entry is an execution refusal; missing exit cash with covered dates remains funded and unknown through report end. A trade's future exit availability does not change its own frozen selection or entry affordability; earlier realized settlements do affect cash available to later entries. Exact decimal-spelling Fraction arithmetic avoids comparison epsilons; O(n log n) time and O(n) workspace, bounded-cadence stderr progress.

Closing costs may produce negative cash, so this is not a fully fee-reserved executable allocator. Equity/drawdown use settled P&L with open positions displayed at debit cost, not mark-to-market or Sharpe. Full net profit per common opportunity is unavailable with unresolved positions; zero funded trades are not evidence of profitability/capacity. No production ledger, eligibility policy or holdout authority.

## Distribution explicit policy

These are additive policy contracts for the distribution experiment, not a runnable experiment. `validate_request(request)` returns a deep-detached closed `distribution_run_request.v1`; `validate_book_config(config)` returns a detached flat book policy. Pure helpers `number(value, positive=False)` and `require(condition)` serve this module family. Unknown, missing or unsupported inputs refuse `INVALID_EXPERIMENT_SPEC`; no repair, economic defaults, source reads, writes, retries or eligibility decisions.

Required request fields: quote snapshot ID; alpha grid (including 0/0.25/0.5) and primary alpha; existing bridge/benchmark ArtifactRefs by outer year; required arms and strict-benchmark flag; economics; book; selector policy/positive minimum net EV; report end; uncertainty; risk; parent/job limits; maximum opportunities; h1 decision horizon; reconstructed-snapshot availability policy. Required primary arms are histogram/same_mean/matched_point. Optional incumbent/gate coverage is enforced later by metadata preflight, not certified by this validator.

Economics are USD with standard multiplier100: commission/exchange fee per contract/leg/side, minimum fee per two-leg side, adverse per-share/per-leg slippage and maximum relative spread. Alpha already represents spread pricing. `unsupported_native_standard` and `not_established_no_depth` disclose that native adjustment identity and executable depth/volume capacity are not established; validation does not detect adjusted contracts.

Book fields are initial cash, positive integer contracts/trade, maximum trade/ticker debit, maximum concurrent positions, calendar-day settlement delay and explicit same-time settlement/entry order. These do not promise a production allocator or executable fee-reserved portfolio. Uncertainty requires seed, resamples, fixed block days/origin and confidence. Risk requires minimum USD uplift per common opportunity, maximum whole-book loss, maximum settled drawdown and funded-outcome completeness floor; later checks are diagnostic, not stops/promotion.

The initial selector is `fixed_preregistered_net_ev.v1`, with a caller-supplied positive USD threshold. It does not guarantee top-20% coverage or learn thresholds from outer outcomes. All numeric economic/resource values must be chosen before a real run; this PR authorizes none. Stdlib validation and the existing native limit/ArtifactRef contracts cover the need without a new dependency or framework.

## Distribution native market adapter

Public `distribution_market.read_bridge(store, entry, fold)` requires a matching entry/fold outer year before CAS, then validates an existing PayoffLineArtifact: STR-THRU/abs_move/exact alpha, finite coefficients/residuals, n>0 and canonical nonempty window with `window_end < cutoff <= outer training cutoff`. Both artifact readers enforce the existing 0<byte_size<=16MiB request limit before storage reads; referenced benchmark row shards retain the existing shared loader contract. It fits no bridge. `read_benchmark(store, entry, fold)` reads optional `distribution_benchmark.v1` with outer year, canonical training cutoff, h1/native-source-session policy and existing `distribution_benchmark` shard descriptor. Rows exactly cover the fold: event_id, finite nonnegative nonempty incumbent_magnitudes or null, Boolean gate_selected or null. Missing entry returns no benchmark; origin/declared causality are assumptions, not authenticated source proof or current-strategy parity.

`decision_quotes(repository, snapshot, events, alphas, economics, *, calendar=None)` reads only D−1 chains, derives sessions from native h1 forecasts, requires the planned decision date to equal the h1 source-session ceiling, and selects contracts once per alpha. One calendar/decision-chain read serves all alphas; a supplied run-local calendar avoids repeated construction. Unplannable events are absent from the quote map; callers must rejoin their original cohort and treat missing map entries as abstentions, not drop opportunities. Planned events with missing decision quotes retain explicit None values. `outcome_quotes(repository, snapshot, decisions, alpha, economics, contracts)` must be called only after intents are frozen; it prices the exact D−1 leg pins at D0/D+1, not newly ATM legs, retaining missing entry/exit values. This low-level adapter does not enforce orchestration order itself or apply `replay_one` future-survivor filtering.

`side_cost(economics, contracts)` returns USD per two-leg side: per-leg/contract commission+exchange fee subject to a side minimum, plus adverse per-share/per-leg slippage. It checks positive plain-integer quantity, USD/standard integer multiplier100, finite nonnegative consumed costs and the computed side cost. Outcome pricing checks these before quote reads; both quote readers check finite nonnegative relative-spread limits before native reads. Both quote readers reject nonfinite, Boolean or out-of-range alpha outside [0,1] before calendar/chain reads. The relative-spread limit gates decision/entry quotes only; pinned adverse exits remain priced to avoid hiding realized losses, not as proof of executable liquidity. Alpha already prices spread; no double spread surcharge. Native adjustment identity, depth/volume/quote-age capacity and production allocator parity are not established. The preceding request contract still owns the full closed policy schema.

Uses native plan_events/ChainSnapshot/price_structure/ChainIndex, existing ArtifactStore/shards and stdlib/Pandas; no new dependency, reader, registration, provider call, persistent cache or eligibility policy. The documented reference-size/identity/clock and consumed cost/quantity checks refuse `INVALID_EXPERIMENT_SPEC`; native artifact/storage and unexpected malformed-document errors propagate. Other policy fields and native event shape retain the preceding validated-request boundary rather than duplicate its full schema. Unpriceable/missing quotes return unavailable values, never fabricated zero proceeds. Read-only helpers publish nothing, offer no retry/partial transaction, and grant no fitting/holdout authority. Iteration logs bounded progress; memory scales with supplied event/cohort/quote reads and remains caller-contained.

## Distribution paired report

`paired_report(opportunities, books, *, seed, resamples, block_days, block_origin, confidence)` accepts a list of unique event_id/ticker/decision_date opportunities and arm→`distribution_book.v1` documents (or None). Every available arm must cover the exact same event set, book configuration, report end and event/ticker/decision clocks. Comparable pricing, fees, slippage and quote policies remain the caller’s responsibility; these economic identities are not present in the book document. It returns `distribution_paired_report.v1` with detached arm summaries, contrasts and explicit uncertainty settings. Invalid supplied shapes, status/value semantics, alignment or nonfinite numerics refuse `INVALID_EXPERIMENT_SPEC`; unexpected programming errors propagate. There are no source reads, writes, partial publication, retries or admission authority.

Event dates must agree between arms even when their decision dates match. For the producer's unplanned abstention (`decision_at=None`), the known `event_date` supplies only the calendar-grouping fallback; it is not an invented decision date or proof of historical availability.

| Paired-report condition | Outcome |
| --- | --- |
| Invalid book policy, shape, finite-value/status semantics, cohort/clocks or uncertainty parameters | `INVALID_EXPERIMENT_SPEC`; no report publication or source/storage writes. Book policy admission reuses `validate_book_config`. |
| Missing arm, unresolved paired outcome or fewer than two clusters | Explicit unavailable/incomplete contrast or interval, not a zero-return substitute. |

Histogram minus same_mean is primary; matched_point/incumbent/gate are secondary. Net deltas use total USD per common opportunity, not a selected-winner denominator. No-trades/refusals are zero; unresolved funded events are unknown. Any unresolved paired outcome disables the full-opportunity contrast, while scoreable-only diagnostics retain explicit exclusions. Missing arms are unavailable, not zero-return books.

Fixed occupied calendar blocks and ticker clusters resample whole-event paired deltas with a ratio-of-sums mean. Fewer than two clusters yields an unavailable interval. These intervals condition on the realized allocation and supplied fitted results; they do not refit models, replay competing cash allocations, correct multiple testing, establish executable capacity or support promotion/current-strategy superiority. Seed, resamples, confidence, block origin and block width are mandatory caller choices. Complexity is O(events log events + resamples × clusters) time and O(events + resamples) working memory per contrast; bounded-cadence stderr progress covers resampling.

`magnitude_candidate_fit.fit_candidate` is the guarded lazy compatibility wrapper of shared
`engine.v2.models.training.magnitude_candidate.fit_candidate`; SIZE recipes,
results, both no-fit guards and errors remain unchanged.

### Distribution declarations (Bundle B prerequisite)

`distribution_candidate_recipe.candidate_recipe(feature_set)` declares one explicit v3 H/HM/HMO feature subset, native seven-leaf HGB point recipe and matched log-loss classifier, the existing 81-bin output, train-only empirical tail policy and uncalibrated/no-smoothing/absent-zero policy. `validate_candidate_recipe` returns detached installed semantics with exact canonical identity; neither is preregistration, fitting or native admission.
`fit_budget` counts two estimator fits per declared inner/outer fold and each of horizons 5/3/1. Histogram trees use the conservative 81-per-iteration ceiling, not actual binary counts. Its child-limit syntax reuses the public executor validator; additional calibration/incumbent/payoff fits, preparation, memory feasibility and parent/job containment are excluded. `trial_identity` hashes recipe, opaque context/fold/input/budget references, role and horizon; references authenticate no rows, clocks, grouped eligibility or authority.
`distribution_class_support.training_support` requires 81 nonnegative integer counts and at least two observed bins, returning detached counts/classes/sample total and one tree per binary iteration or one per observed multiclass. `class_histogram` maps explicit distinct class labels positionally (including unsorted labels), pads absent bins with zero and delegates probability/tail validation to the numerical histogram owner without smoothing or repair.
These source-free helpers use public feature/native model recipes, foundation hashing, native errors and numerical histograms; the plan/journal below consumes recipe identities, while class-support consumers are synthetic tests. These declarations add no registration/CAS transport, journal, reader, eligibility or holdout policy and create no executable plan or resource reservation. Outputs have fixed feature/class dimensions, not row payloads; no input-byte/allocation cap, I/O, cache, writes, transaction or retry is provided. Omitted required arguments raise Python `TypeError`.

| Distribution declaration condition | Outcome |
| --- | --- |
| Malformed/changed recipe, references, class mapping, counts or resource syntax; insufficient fit/tree caps | Nonretryable `INVALID_EXPERIMENT_SPEC`; insufficient caps instead use `RESOURCE_LIMIT_EXCEEDED`. No partial plan or execution. |
| Fewer than two observed classes; malformed histogram probabilities/tail | `EXPERIMENT_VARIANT_FAILED` for degenerate support; numerical owner's `CONTRACT_MISMATCH` propagates unchanged for probabilities/tail. |

### Distribution native admission

`distribution_plan.prepare_plan(conn, store, own_registration, source_registration, source_request, source_plan_ref, recipe, budget_caps)` returns `(distribution_plan, verified_native_source_plan)`. It verifies a distinct native v1.0 distribution run and compact SIZE v1.7 source registration, matching snapshot/scope/holdout month/membership/event roster, seed-zero computed-move ownership and outer years. The registered runner must differ from SIZE's; its installed source closure/environment and the distribution implementation are bound. No placeholder runnable entrypoint is supplied: Bundle D owns genuine runner/activation checks.
The public native search-plan reader reconstructs and verifies exact source/fold metadata; its 33 candidates, RMSE selection and execution budget are not distribution policy. No SIZE execution is required. Both registration byte strings retain the 16 MiB ceiling before decoding; native v1.0's flat event roster must fit that existing bound. Oversize refuses, without copying v1.7 packing, raising caps or duplicating panel/metadata payloads.
`distribution_plan.v1` binds both registration refs/run/variant IDs, source plan ref/ID, qualified context ID, recipe, installed sources/environment and derived raw-pair budget. The identity contains no event/feature rows; actual inner/outer schedule counts replace caller fold counts. The returned native plan is invocation-local metadata, not copied into distribution evidence. These are raw-pair limits only: calibration/control fits and whole-job feasibility remain excluded until explicitly budgeted.
`DistributionTrials` takes the same arguments plus `repository`, with detached request/recipe/caps. `seal(clock=...)` revalidates dependencies and explicitly initializes the existing magnitude-search journal migrations, then appends the immutable plan in `distribution:seal` under the distribution run ID. Exact seal replay is idempotent; different plans refuse. No new store, catalog schema, registration version or holdout policy is introduced.
`start(role, outer_year, inner_year=None)` internally selects an exact native inner or normalized outer fold. Its identity includes the full fold, enclosing outer year and role; all three horizons remain together. Before claiming, it requires a callable repository `resolve` entrypoint without invoking it; this shape check does not pre-read or certify source availability/integrity. A create-only `distribution:start:<fold_id>` claim precedes public `fold_inputs` and any fresh training outcome read. Central qualification owns protected/future exclusions and authentic label anchors; the adapter does not filter eligibility itself.
Only after full native admission are feature rows projected to the recipe's installed H/HM/HMO columns. Existing `publish_inputs` writes immutable cohort shards; `distribution:admission:<fold_id>` commits their manifest, actual inputs ID, qualified-context/recipe/budget-bound horizon identities and status `admitted`. The receipt returns detached reference metadata, not labels, forecasts or a completed fit. No source outcome or private-data authority follows from calling recipe/plan constructors.
`read_admission(...)` revalidates registration/plan/claim identity and uses public `load_fit_inputs` to verify stored training/feature shards before returning the exact receipt. It does not reopen native outcomes, refit, repair or claim execution completion. It can read previously admitted training bytes, so it is not a metadata-only storage operation. Each call revalidates; no cross-call admission cache is added.

| Distribution admission condition | Outcome |
| --- | --- |
| Malformed recipe/caps/fold selector or missing/noncallable repository `resolve`; insufficient raw-pair caps; either registration over 16 MiB | `INVALID_EXPERIMENT_SPEC`; insufficient caps or oversized registration use `RESOURCE_LIMIT_EXCEEDED`, before a claim. Omitted arguments retain Python `TypeError`. |
| Verified registration/source-plan ownership, population, clock or installed digest mismatch; absent/changed seal or stored receipt; duplicate start or unfinished admission replay | Identity comparisons refuse `EXPERIMENT_IDENTITY_CONFLICT`; no new outcome read or automatic restart. Missing journal migrations retain native `INTEGRITY_FAILED`; dependency verification failures propagate their native errors. |
| Retryable `OpsError` or `DataError` during native fold admission, cohort publication or admission-record append after a successful start claim | Nonretryable `EXPERIMENT_IDENTITY_CONFLICT` with a fixed interruption message and no original details; the single-use claim cannot restart. Published unreferenced shards may remain. No release, retry or whole-operation rollback is introduced. |
| Failure before a successful start claim (including missing/oversized implementation source, `INPUT_CHANGED`), nonretryable `OpsError`/`DataError`, untyped error, or failure during `read_admission` | Original failure and retryability propagate. Existing append transactions own atomic associations; an already committed claim stays consumed, and replay verifies stored evidence only. |

Current consumers are synthetic unit/native-catalog tests only. Dependencies are native registration, public SIZE plan/fold admission, shared cohort publisher/loader and immutable journal/migrations, foundation hashing/CAS, installed fingerprints and bounded stderr stage logging. No estimator, provider, CLI, final read or production path is added. Native v1.7 hydration and cohort processing retain their existing bounds; this adapter does not establish memory feasibility or add a scheduler. C/D must supply causal OOF/result contracts and a narrow bounded distribution-worker dispatch rather than coercing histogram output into scalar `SearchTrials` forecasts.
