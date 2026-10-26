# `engine/v2/ops`

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for this component's contracts,
inputs/outputs, dependencies and invariants.

## Ownership

Implements the **Supervisor/catalog — transactions, leases, dependencies, capacity, retry history** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**7** of §4.1.

Replaces (§4.4): `new supervisor and catalog`, `dashboard/nightly.py becomes a job graph`, `tools/bounded_run.py becomes an executor adapter`.

## Responsibilities

- Durable job submission, leases, retry history and dependencies.
- Resource admission and per-job CPU placement.
- The nightly job graph and its release boundary.
- Explicit Massive acquisition via `python -m engine.v2.ops.massive_acquire` and daily/action snapshot or parent-bound refresh submission via `python -m engine.v2.ops.massive_snapshot`; [operator usage](stores/massive/README.md).

## Native monthly training

`ops plan training --training-mode recipe --recipe implied_t1:*:tier4_monthly`
accepts `--training-input-mode native_d14` with all four explicit pins:
`--training-catalog /absolute/catalog.sqlite`, `--training-artifact-root /absolute/store`,
`--training-snapshot-id ID`, and `--training-receipt-id ID`. Use the returned plan
with ordinary `ops submit`; `runup_move:*:tier4_monthly` uses the same contract.
Planning validates without reading the source. The worker verifies immutable
inputs on every attempt and binds their lineage into resume identity. No legacy
manifest, mutable-head discovery, schedule or promotion is involved.

## Non-responsibilities

- **Decide research conclusions** — `engine/v2/evaluation` does it instead.
- **Compute a score** — `engine/v2/scoring` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

The operator interface is the versioned command protocol exposed by
python3 -m engine.v2.ops. It supports immutable planning/submission, job and
attempt inspection, cancellation, recovery planning, health, read-only nightly
limit headroom (`headroom`) and supervisor execution. Except for the narrow
registration surface below, Python modules are internal to ops; production
packages consume versioned artifacts instead of importing the supervisor.

Native experiment registration also exposes these submodule APIs:
`submission.get_job` and `submit_graph` accept `max_failure_bytes=None` and `receipt_decode_failure_code=None`. An omitted/None budget means 8 MiB UTF-8; historical failures above that limit require an explicit larger finite budget. A positive exact-integer budget (at most `(2**63-1)//2`) caps failure text before SQL transfer/JSON decoding, with UTF-16 transfer allowance twice the UTF-8 budget and a UTF-8 recheck. Size refusal uses the explicit registered receipt code, otherwise `RESOURCE_LIMIT_EXCEEDED`; wrong-type, invalid-encoding or malformed failure JSON retains `INTEGRITY_FAILED`. The optional code also types only queue/resource/progress JSON decoder failures; other errors propagate. `validate_receipt_read_policy` rejects invalid code/budget with `INVALID_REQUEST` before reads, graph writes or sweep staging. Collection explicitly passes the fixed `MAX_JOB_FAILURE_BYTES` (1 MiB) and R6 throughout initial replay and observation.

All new job/attempt failure writes enforce a 1 MiB canonical UTF-8 ceiling. The supervisor validates the original `Problem` classification at every size before serialization or publication: registered code, matching category, exact boolean retryability and supported schema. It preserves an oversized original, including any prior diagnostic reference, as a content-addressed `problem.v1.0` artifact and writes a fixed-size summary with the original classification and a new diagnostic reference. Existing `failure_diagnostic.v1.0` artifacts still contain raw worker details. Internal `lifecycle.commit_attempt(failure_artifact=None)` registers a supplied, already-published `problem.v1.0` artifact and both failure receipts in the same fenced transaction; the summary must name that artifact. Storeless writers refuse oversized input with `RESOURCE_LIMIT_EXCEEDED` before persistence. Valid small failures retain their exact canonical bytes. Publication precedes the transaction: a crash, foreign fence or catalog refusal may leave a complete content-addressed orphan; publishing identical bytes is idempotent. Artifact publication failure raises typed storage/integrity failure with the original Problem chained in the exception, commits no completion, and cannot promise durability before the first successful publication. Historical rows are never migrated automatically.
`native_registration.read_native_evidence(conn, run_id, max_bytes=MAX_NATIVE_EVIDENCE_BYTES)` returns None or `(row, object)` from one capped SQL snapshot; `native_evidence_json(document, max_bytes=...)` returns matching bounded canonical object text. Default is 16 MiB UTF-8 (twice for UTF-16 SQL transfer before UTF-8 recheck); malformed/oversized catalog evidence remains R6, raw storage errors propagate. `verified_registration_bytes(max_evidence_bytes=None)` retains opt-in bounds: invalid schema/expected reference remains R6; an admitted reference over cap raises nonretryable `RESOURCE_LIMIT_EXCEEDED` before object read with `registration_bytes`/`max_registration_bytes` details. No writes, integrity claim or automatic retry. Public experiment read/require supply finite read caps; register caps prepared bytes and reservation rechecks. Existing low-level None callers keep their policy; this is not universal worker/outcome containment.
`checkpoints.registered_artifact(max_metadata_bytes=None)` optionally caps reference JSON before transfer/decoding, refusing `INTEGRITY_FAILED` on non-text, oversized or invalid-encoding text; malformed JSON/contract decoding still raises raw `ValueError` (including `DocumentError`) or `RecursionError`. Its default remains unchanged and it never reads artifact content.
`catalog.sql_text_budget` returns the catalog encoding and UTF-16-adjusted transfer allowance; `catalog.decode_bounded_text` decodes projected bytes and rechecks the caller's UTF-8 budget, raising `ValueError` on invalid evidence. Callers own SQL snapshots and typed refusal policy.
`catalog.register_artifact(conn, ref, attempt_id, clock)` owns canonical artifact registration inside the caller's transaction. Identical registration is a no-op preserving original provenance/time; conflicting metadata raises `INTEGRITY_FAILED` (`artifact identity conflict`). It neither publishes bytes nor verifies a fence. Lifecycle calls it after fence validation; `checkpoints.register_artifact` retains its exact public signature as a compatible wrapper, used by checkpoint and other existing registration callers.
The registration surface also includes `catalog.transaction`, `errors.OpsError`, `errors.fail`,
`experiments.ExperimentSpec`, `experiments.resolve_experiment_plan`,
`experiments.register_hypothesis_in_transaction`, `experiments.NATIVE_GATE_WORKER`, `experiments.NATIVE_GATE_ENTRYPOINT`, `experiments.NATIVE_GATE_RUNNERS`,
`fingerprints.environment_identity`, `fingerprints.source_closure` (reads each `.py` once and refuses
`INPUT_CHANGED` for one over 8 MiB) and
`native_registration.verified_registration_bytes` and
`native_registration.magnitude_execution_document` (the fixed mean-baseline
execution policy with an explicit evaluation-label cutoff) and
`native_registration.validate_magnitude_execution` (pure closed-envelope validation).
Stored history additionally exposes `native_registration.magnitude_history_request`;
`native_registration.magnitude_history_execution_document` and `native_registration.validate_magnitude_history_execution`
seal/check v1.4. `native_registration.magnitude_fitted_execution_document` and `native_registration.validate_magnitude_fitted_execution`
seal/check v1.5 recipes owned by registration, without importing training.
These constructors/validators do not grant read authority.
They compose one resolved plan, source/environment binding and primary catalog
reservation; the caller owns the transaction around the reservation helper.
The registration verifier reads only catalog-owned binding evidence and grants no final-read authority; internal `final_holdout.authorize_final_holdout` has only synthetic tests as consumers and no production entrypoint or read capability.
Native prediction and reconstructed gate reporting consume the public fold-local
`experiment_folds.TrainFoldRule` and `experiment_folds.fit_walk_forward_fold`.
The fixed native gate child consumes `worker.main(dispatcher=...)` for the unchanged launch/result protocol; ops never imports its pricing or evaluation adapters. Prediction and gate CLIs consume `catalog.connect` and `experiments.experiment_spec_from_document`
to open an existing catalog and decode the same strict experiment specification.

Native sweep coordination also exposes `submission.submit_graph`/`submission.NamespacePolicy`/`submission.validate_receipt_read_policy`,
`stages.registry`, `checkpoints.register_artifact`, `migrations.require_current`,
`schema.OWNER`/`schema.MIGRATIONS`, and `fingerprints.worker_source_manifest`
for current-schema admission, artifact registration, source binding and atomic job submission.
Ops exposes `migrations.Migration` and `migrations.migrate` to search recording for its
separate owner sequence; it does not add or reinterpret ops migration versions.

`native.input_decoding._load_native_score_inputs` is an internal decoder shared
by the CLI, worker and dashboard projection tool, separate from the versioned
command protocol.

The model rollback operator command is
`python3 -m engine.v2.ops plan rollback`. Use `--release-root ROOT` to select
the release store. The plan pins the live
incumbent and rollback target, refuses `--release-id`, and leaves the target
unset only for `NoPriorRelease`; that worker refusal is typed
`VALIDATION_FAILED`. A submitted plan with an incumbent but no pinned target
is also refused as typed `VALIDATION_FAILED` before pointer or history
mutation. Malformed pointer or history data refuses planning with typed
`VALIDATION_FAILED`. Submit the saved plan through
`python3 -m engine.v2.ops submit --plan PLAN --idempotency-key KEY`.

Internal staging API: `nightly_raw_rows.scan_calendar_row` returns
`CalendarRowInputs` from a pinned earnings event and explicit market context.
`nightly_calendar_inputs.scan_calendar_row_inputs` calls it; see the
component architecture contract.

`nightly_quote_rows.scan_quote_rows` is also an internal staging API. It
returns `QuoteRowInputs` from a pinned `option_chains` snapshot for an exact
`(ticker, decision_session)` request; an optional `chain_rows` supplies that
ticker's rows from the batched read. `nightly_raw_row_producer` is its
production caller; see the component architecture contract.

`nightly_chain_batch.scan_chain_slices`, `CHAIN_COLUMNS`, `CHAIN_TABLE`,
`require_chain_rows` and `require_usable_fields` are the internal batched
`option_chains` read: one scan per producer build, rows grouped by ticker,
handed to the quote and calendar readers as `chain_rows`.
`nightly_raw_row_producer` calls `scan_chain_slices`; `nightly_quote_rows` and
`nightly_calendar_inputs` call `require_chain_rows` to check the rows they are
given, and `nightly_calendar_inputs` calls `require_usable_fields` on every
candidate row.

`nightly_calendar_inputs.scan_decision_calendar`,
`nightly_calendar_inputs.scan_candidate_expiries`, and
`nightly_calendar_inputs.scan_calendar_row_inputs` (both of the latter accept
an optional `chain_rows`; calendar-row scanning also accepts `spot_reader`) are internal staging APIs
for pinned raw-row calendar inputs. They compose decision-calendar sessions,
strategy-eligible expiries and calendar-row inputs; `nightly_raw_row_producer` calls `scan_decision_calendar`, `make_spot_reader`, and `scan_calendar_row_inputs` (which calls `scan_candidate_expiries`). See the component architecture contract.

`nightly_calendar_inputs.make_spot_reader(repository, snapshot)` creates a frozen
`SpotReader` bound to those exact objects; at most 128 `(ticker, session)` spot rows or
missing values are retained. Full histories and exceptions are never cached.
Direct calls validate a nonempty string ticker and naive midnight session before cache access.
After existing key/date/chain validation, the calendar helper refuses a wrong reader
type or repository/snapshot binding, and wrong row type/session, as `INVALID_REQUEST`.

`nightly_raw_row_producer.build_native_score_batch_events` is an internal
producer that composes native raw-row event and refusal documents from pinned
inputs. `native_score_batch.run_native_score_batch_worker` is its production
caller: it resolves the pinned snapshot and invokes the producer in the worker
process. `Service._reconcile_native_score_batch_shadow` only submits producer
parameters for an eligible pinned-snapshot identity. See the component
architecture contract.
`nightly_read_summary.ReadTracker` is the producer's observation-only read
attribution (per-step `reads` deltas and the job-end `read_summary`);
`nightly_raw_row_producer` is its only caller, and its output rides on
`ProgressEvent.step_reads`.

The internal `native_score_batch._STRATEGY_REGISTRY` declares each admitted
row's ordered forecast outputs and exact/alias driver roles, gate, policy key,
source-bundle strategy and optional driver-name override. All resolved drivers
contribute features, recipes, identities and request artifact references.
STR-THRU, STR-RUNUP, TWIN-P and TWIN-P5 are registered (TWIN is gate-less: entry-rule gate over a frozen paired pool and trailing cutoff). RUNUP requires both verified champion
residual pools and a unique exact-key payoff surface; its model gate does not
request separate simulation. Missing/unusable model artifacts refuse its row.
There is no runtime registration API or chooser execution contract.

`carried_set.CarriedTickerExclusion`, `carried_set.CarriedSetResolution`,
`carried_set.resolve_carried_set`, and
`carried_set.build_uncarried_exclusions` expose slice-1 carried-set evidence.
The resolver returns the ticker sets held in a pinned snapshot; the exclusion
builder returns sorted `UNCARRIED_TICKER` evidence for candidates missing from
either required table. Their current consumer is
`tests/v2/ops/test_carried_set.py`; this slice has no production caller.

`nightly_session` (session/generation identity and compare-and-swap state, #564 slice 1) is internal to this package and used by completion persistence in `supervisor.Service`; tests include `tests/v2/ops/test_nightly_session.py` and `tests/v2/ops/test_nightly_completion.py`. `nightly_completion` (run-scoped reports, #564 slice 7; exports `Disposition`, `SCHEMA`, `build_completion_report`, `record_completion`) is internal and consumed through `supervisor.Service`; `tests/v2/ops/test_nightly_completion.py` covers its report contract and persistence. `nightly_receipts` (step receipts and effect reconciliation, #564 slice 2; exports `Effect`, `Reconciliation`, `StepReceipt`, `begin_step`, `complete_step`, `reconcile_step`) is imported by the production modules `nightly_chain`, `nightly_history`, and `nightly_repair`; `nightly_trigger` does not call it yet. `tests/v2/ops/test_nightly_receipts.py` covers its contract. `nightly_readiness` (legacy run-report readiness for a session, #564 slice 3; exports `LegacyReadiness`, `check_legacy_report`) is called by `nightly_trigger` as a temporary legacy-completion check (#875, removed with #876); `tests/v2/ops/test_nightly_readiness.py` covers its contract. `nightly_chain` (snapshot import/refresh/moves/capture step chain over step receipts, #564 slice 4; exports `ChainResult`, `ChainStep`, `run_snapshot_chain`) is imported by production module `nightly_history`; `nightly_trigger` does not call it directly. `tests/v2/ops/test_nightly_chain.py` covers its contract.

<!-- public-interface: Migration, migrate, magnitude_fitted_execution_document, validate_magnitude_fitted_execution, registered_artifact, register_artifact, verify_eod_availability, transaction, sql_text_budget, decode_bounded_text, OpsError, fail, ExperimentSpec, resolve_experiment_plan, register_hypothesis_in_transaction, verified_registration_bytes, magnitude_execution_document, validate_magnitude_execution, magnitude_history_request, magnitude_history_execution_document, validate_magnitude_history_execution, environment_identity, source_closure, worker_source_manifest, TrainFoldRule, fit_walk_forward_fold, connect, experiment_spec_from_document, READ_ONCE, reclaim, THREAD_VARIABLES, NativeGateParameters, NATIVE_GATE_WORKER, NATIVE_GATE_ENTRYPOINT, NATIVE_GATE_RUNNERS, main, submit_graph, validate_receipt_read_policy, NamespacePolicy, registry, require_current, OWNER, MIGRATIONS, MAX_JOB_FAILURE_BYTES, get_job, MAX_NATIVE_EVIDENCE_BYTES, read_native_evidence, native_evidence_json, scope_from_native_score_batch_key -->

## Consumers

`experiments/magnitude_search_request.py` consumes `errors.fail` for typed source-free declaration refusals.

`experiments/runup_exit_report.py` consumes `fingerprints.source_closure` /
`environment_identity` and `errors.OpsError` / `fail` for exploratory identity
and typed refusals; `experiments/v2_runup_exits.py` consumes those typed errors;
`experiments/record_export.py` consumes `errors.fail` / `errors.OpsError` for its typed refusals and failure causes; `experiments/v2_gate.py` and `experiments/v2_prediction.py` consume its `export_or_skip`.
`experiments/registered_runup.py` consumes specification resolution and typed refusals; `experiments/v2_runup_study.py` also consumes catalog connection and specification decoding for registered RUNUP execution.

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

`experiments/native_gate_worker.py` consumes `stages.NativeGateParameters`, `executor.THREAD_VARIABLES`, `experiments.experiment_spec_from_document`/`resolve_experiment_plan`, `fingerprints.environment_identity`/`source_closure` and `errors.OpsError`/`fail` for admission.
`experiments/native_gate_entrypoint.py` consumes `worker.main` and injects only fixed pricing/evaluation callbacks.
`experiments/native_gate_evaluation.py` consumes `stages.NativeGateParameters` and `errors.fail`; its admission/receipt helpers are owned by unlayered `experiments.native_gate_worker`, not ops.
`tools/phase7_candidate_manifest.py` consumes `stages.registry` to build the job-graph rows of the P7-1 manifest (operator-run, read-only).
`experiments/v2_gate_sweep.py` consumes `submission.submit_graph`/`NamespacePolicy`/`validate_receipt_read_policy`,
`stages.registry`, `checkpoints.register_artifact`, `catalog.connect`/`transaction`,
`migrations.require_current`, `schema.OWNER`/`MIGRATIONS`, `experiments.experiment_spec_from_document`, `errors.OpsError`/`fail` and `fingerprints.environment_identity`/`source_closure`/`worker_source_manifest`
to submit an immutable common-snapshot roster atomically. Child artifacts are
unrecorded evaluation evidence; operations success is not scientific sweep completion.
`experiments/native_sweep_collection.py` consumes `catalog.transaction`/`sql_text_budget`/`decode_bounded_text`,
`checkpoints.registered_artifact`, `experiments.experiment_spec_from_document`,
`fingerprints.source_closure`, `native_registration.read_native_evidence`/`native_evidence_json`/`MAX_NATIVE_EVIDENCE_BYTES`, `submission.get_job`/`MAX_JOB_FAILURE_BYTES` and `errors.OpsError`/`fail` for collection. It directly reads ops-owned `attempts` and `attempt_outputs` for the exact terminal attempt; `artifacts` remains accessed through `registered_artifact`.
`experiments/v2_gate_collect.py` consumes `catalog.connect`, `submission.get_job`/`MAX_JOB_FAILURE_BYTES` and `errors.OpsError`/`fail` for an observer-only CLI.

No other production package imports ops. The CLI starts the coordinator, and
serving reads a versioned health artifact without a peer-layer import. The one
exception is the layer-8 preview launcher's server composition
(`engine.v2.dashboard._server`), which imports `cli.refresh_action` lazily and
only to wire an optional `--ops-root` shadow nightly-plan submission onto the
operations server's `POST /actions/refresh`. That lazy import still runs
`engine.v2.ops.cli`'s own top-level imports, so the supervisor, executor and
submission write path all load, and `refresh_action` writes the queued jobs to
the catalog. What the route does NOT do is start the supervisor, execute the
submitted jobs inline, or switch production authority: it queues the plan and
returns the job ids, leaving execution to a separately started supervisor.

The top-level `experiments/native_registration.py` library consumes the narrow
registration surface above, including `native_registration.native_evidence_json` for bounded reservation serialization and `native_registration.MAX_NATIVE_EVIDENCE_BYTES` for its finite public default. Like other `experiments/` callers, it is outside
the engine import graph; it introduces no peer-layer engine dependency.
`experiments/native_outcomes.py` consumes `catalog.transaction`, `native_registration.read_native_evidence`/`native_evidence_json` and
`errors.OpsError`/`fail` for catalog-owned publication and typed refusal.
It owns short catalog transactions; artifact and CSV effects stay outside them.
`experiments/magnitude_inputs.py` consumes `native_registration.magnitude_execution_document`;
`experiments/magnitude_report.py` consumes `native_registration.validate_magnitude_execution`
and `errors.fail` before computing/reporting the admitted mean baseline.
The registrar also consumes the history/fitted constructors and matching validators;
`experiments/magnitude_history_report.py` consumes both history/fitted validators,
and the history admission bridge uses `errors.fail`.
`experiments/magnitude_search_plan.py` consumes the fitted constructor/validator,
`fingerprints.source_closure`/`environment_identity` and `errors.OpsError`/`fail`; its CLI uses
`catalog.connect`, `experiments.experiment_spec_from_document` and `errors.OpsError`/`fail`.
`experiments/magnitude_search_trials.py` consumes `migrations.migrate`/`require_current`,
`fingerprints.source_closure` and typed failures; `magnitude_search_records.py` consumes
`migrations.Migration`, `catalog.transaction` and typed failures for the search-owned immutable journal. `experiments/magnitude_search_errors.py` consumes `errors.OpsError`/`fail` for shared artifact-failure classification.
The trials caller uses legacy/v2 no-fit guards and `experiments.magnitude_search_execution`;
that fixed-child supervisor consumes `executor.THREAD_VARIABLES` and `errors.fail`, not the ops job/retry lifecycle.
`experiments/magnitude_search_worker.py` consumes `errors.OpsError` to distinguish estimator-local failures; `experiments/magnitude_fitted.py` consumes `errors.OpsError`/`fail` for strict propagation and legacy failure mapping.
`experiments/magnitude_search_outer.py` consumes `migrations.require_current`,
`fingerprints.source_closure` and `errors.OpsError`/`fail` for selected outer evaluation.
`experiments/v2_magnitude_search_run.py` consumes those same interfaces plus
`catalog.connect` and `experiments.experiment_spec_from_document` for serial CLI admission.

Within ops, `cli.rescore_command` and `worker._dispatch_adhoc_rescore` call
`native.input_decoding._load_native_score_inputs`. The dashboard projection
tool calls the same decoder from `tools/v2_dashboard_project.py`.

<!-- consumers: engine.v2.dashboard -->

## Usage

Read-only host inspection:

    python3 -m engine.v2.ops doctor --json

Initialize an isolated catalog:

    python3 -m engine.v2.ops init --root /tmp/operations-example

Initialization installs no timer and makes no production writer change. The
Phase 1 runbook documents planning, private verification and the separate
activation boundary.

Settling a stuck `recovery_pending` attempt by hand, when no supervisor is
ticking:

    python3 -m engine.v2.ops reconcile <job-id> --expected-attempt <attempt-id>

This runs the identical ownership proof a running supervisor applies every
tick (every recorded identity gone or a zombie; no live session still
carrying the launch pid; no live process's environ carrying the attempt's
staging marker) and refuses to release anything the proof cannot clear.
There is no force flag. It refuses with `RESOURCE_UNAVAILABLE` while a
supervisor holds the catalog's lock, since that supervisor already
reconciles every tick; a proof that fails prints its blocking processes as
pid plus start time only.

Capturing a `price_history` generation (task brief 2026-09-14; SEND-BACK
2026-09-14: a real Tier-2 catalog table, `engine.v2.ops.price_history_store`)
from the legacy yfinance sources plus the legacy `gspc_daily.csv` regime file
under a read-only checkout (the capture requires `CALENDAR_TICKER` and
`REGIME_TICKER` -- SPY and the ^GSPC `price_history` close), advancing
`--scope`'s head to a new snapshot that carries every other table's dataset
version forward unchanged alongside price_history's fresh one:

    python3 -m engine.v2.ops price-history capture \
        --source-root /path/to/legacy/checkout --scope shadow [--dry-run]

`--scope` must already have a committed head snapshot. `--dry-run` reports
counts without writing anything or committing a new generation.

## Testing

Recommended fast invocation, once `pytest-xdist` (pinned in
`requirements-dev.txt` -- test-runner tooling, kept out of `requirements.txt`
because that file is gated byte-for-byte against a frozen Phase 0 baseline)
is installed:

```text
python3 -m pytest -q -p no:cacheprovider -n auto --dist loadgroup tests/v2/ops/ tests/v2/integration/test_checks_phase2_gate.py
```

Measured on this host (12 cores, shared with other agents -- times vary with
load): serial 70-105s; three consecutive parallel runs at 39.95s (load
average 1.08), 33.78s (load 6.66), 33.26s (load 6.89) -- 567 passed, 1
skipped, every time, no parallel-only failures. Re-run three times after any
change to the grouping in tests/conftest.py to catch a new parallel-safety
bug before it lands.

`-n auto --dist loadgroup` is recommended ONLY for `tests/v2/ops/
tests/v2/integration/test_checks_phase2_gate.py` above. The LEGACY suite
(`tests/test_calendar.py`, `tests/v2/dashboard/test_dashboard.py`, `tests/test_features.py`;
274 tests, ~100s serial, per main's committed data) runs SERIALLY, full
stop -- it is not part of the v2 command above, needs the real data trees
(`earnings_predictions/`, `polygon_cache/`) a bare worktree checkout does
not have, and a spot check under real data found a genuine parallel-safety
bug: `python3 -m pytest -q -n auto --dist loadgroup tests/test_calendar.py
tests/test_features.py tests/v2/dashboard/test_dashboard.py`, from a real data checkout,
hung for 22+ minutes (vs ~100s serial) before being killed. Bisected to
`tests/test_features.py`: 2+ real xdist workers each independently loading
the real feature panel reliably crashed a worker on this RAM-constrained
shared host (`[gwN] node down: Not properly terminated`) and then hung
xdist's own crashed-worker replacement indefinitely; `test_calendar.py` and
`test_dashboard.py` were each independently parallel-safe alone. Grouping
`tests/test_features.py` onto one worker (`xdist_group("serial")`, kept as
a second guard) does NOT make the legacy suite parallel-safe as a whole --
it stops that file's own tests from piling up on each other, but does
nothing to stop a DIFFERENT worker from loading the same real panel again
at the same time from a different legacy test file, which is the same
memory pressure by another name. Run the legacy suite as
`python3 -m pytest -q tests/test_calendar.py tests/test_features.py
tests/v2/dashboard/test_dashboard.py` (plain serial; main's committed count is 274
passed) until someone actually re-verifies parallel safety end to end
against real data, not just against this data-less worktree.

`--dist loadgroup` is required, not optional: a handful of tests touch a
REAL, host-wide resource (a real child process's CPU affinity, a real
cgroup/watchdog probe, real process-group/session signaling, a real
Playwright browser) and are marked `@pytest.mark.xdist_group("serial")` (or
`pytestmark` for a whole file) so `loadgroup` pins each such group to one
worker instead of letting two copies collide on the same host resource. See
`tests/conftest.py` for the exact grouped files and the reason for each.
Without `pytest-xdist` installed, the plain serial command still works
unchanged: `python3 -m pytest -q tests/v2/ops/ tests/v2/integration/test_checks_phase2_gate.py`.

The one real end-to-end Phase 2 check — a fresh coverage measurement of the
REAL registered suite against the REAL tree, fed into the REAL gate, proving
it is red for exactly the rows lacking real evidence — is deliberately not
part of the unit suite (it needs a real `coverage run`, which the fast unit
tests must not pay for on every run). Run it directly instead:

```text
python3 checks/v2_coverage_ratchet.py --profile phase2 --measure --output /tmp/p2cov.json
python3 checks/rearchitecture_phase2_gate.py --coverage /tmp/p2cov.json --json
```

Today this is red with `MISSING_EVIDENCE` for D13/D14 (no real test file yet)
and D15/D16/D19 (tier-2 rows still needing a real receipt), and clean for
every other registered row. `tests/v2/integration/test_checks_phase2_gate.py`'s own smoke
test only checks the registry against the real tree cheaply (file existence
plus one `pytest --collect-only`, a few seconds) — it does not run this
measurement.

Both coverage profiles (`--profile phase1`, `--profile phase2` of
`checks/v2_coverage_ratchet.py`) accept an opt-in `--measure --parallel`
flag that runs the same fixed suite under `-n auto --dist loadgroup` and
combines every xdist worker's coverage data with coverage.py's own
multi-process support (`COVERAGE_PROCESS_START` plus the system
`coverage.process_startup` `.pth` hook, then `coverage combine`) rather than
adding pytest-cov. The default stays serial.

Measured on this host: Phase 1 (41 files) serial 74-85s, parallel 32-35s,
two runs each -- every per-package executed/executable count identical
across all four measurements. Phase 2 (18 files) serial ~44s, parallel
~27-33s; junit outcomes identical (361/361, same nodeids, same per-test
result) between one serial and one parallel run, but engine.v2.ops's
executed count came out 2 lines HIGHER under parallel (2709 vs 2707 of
4083) -- traced to `executor_watchdog.py:37-38`
(`process_table()`'s real `/proc` scan hitting its own documented TOCTOU
race more often under real concurrent process churn; see that function's
docstring and `measure()`'s docstring in both scripts for the full trace).
This is real host nondeterminism in production code, not a combine bug --
confirmed by Phase 1's 4-for-4 exact match and Phase 2's exact junit match
using the identical mechanism -- and it can only ADD lines under
contention, so it can never trip either script's DECREASE-only regression
check. `--parallel` is kept available on both scripts rather than refused.
If a `--parallel` measurement ever looks unexpectedly higher by a line or
two in engine.v2.ops, that TOCTOU race is the first thing to check, not
data loss.

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

The tests/test_v2_ops_*.py suite covers catalog uniqueness, lease fencing,
resource admission, worker identity, immutable checkpoints and effect recovery.
Fault tests must assert retained reservations and rejected stale effects after
process death, and exact immutable input identities before checkpoint reuse.

The copy-only `legacy_decisions` coordinator commits a candidate only when its
job admits immutable score, finality, decision-plan and decision-evidence
artifacts.  The evidence binds the population, causal cutoffs, selection and
replayed score content.  Candidate decisions, compatibility-export intent and
the decision watermark share the attempt-completion transaction.  Missing or
changed evidence fails closed.  The current general nightly planner does not
produce a decision-plan/decision-evidence bundle, so this coordinator path is
available for isolated authority rehearsals but does not make the full
nightly complete or activate a production writer.

A worker's `input_bindings` may name a parent job's committed output as
`job_<id>#<output_name>` instead of a direct artifact ID (P2-5/B1a).
`input_bindings.resolve_bindings` resolves every entry once, at launch,
against the parent's committed state — a direct ID must be in
`spec.input_refs`; a `job_` binding must name a declared dependency whose job
has succeeded, carrying an explicit `#output_name` that matches one of its
succeeded attempt's outputs — and `executor._materialize_inputs` records the
result in `attempt_input_bindings` (migration 7, append-only: update and
delete are refused by trigger) before staging any byte or launching the
worker process. The `legacy_decisions` coordinator (`decision_commit.py`)
resolves its four evidence bindings through that durable record rather than
re-querying a `job_` binding's parent at commit time, so a parent whose
output has since changed cannot silently swap what gets committed. Checkpoint
cache identity (`supervisor.py`, `checkpoints.commit_checkpoint`) folds the
same resolved `(name, artifact_id)` pairs into its `inputs` hash, so a job
whose parent output changed cannot reuse a stale checkpoint.

The four effect-receipt kinds (`ledger_export`, `engineering_gate`,
`publication`, `backup`) run a trivial pure worker (`worker.py`'s
`_dispatch_effect_receipt`) plus a coordinator effect
(`effects_graph.py`, called from `supervisor.Service._coordinator_effect`)
that does the real catalog/outbox/filesystem work and may publish its own
`extra_refs` artifact. Both land in `attempt_outputs`
(`PRIMARY KEY(attempt_id, name)`), so the worker's receipt is always named
`<kind>_receipt` — never the bare kind name a coordinator-published artifact
(e.g. `ledger_export`'s tar, `engineering_gate`'s gate document) uses, which
is also the name downstream `job_<id>#<name>` bindings expect. A duplicate
name refuses cleanly with `VALIDATION_FAILED`
(`supervisor._refuse_output_name_collisions`) rather than surfacing a raw
`sqlite3.IntegrityError`. `tests/test_v2_ops_effect_receipt_collision.py`
proves this end to end through a real `Service` with real subprocess
workers — the gap every direct-call test of the coordinator effect functions
leaves open — and `tests/v2/ops/test_v2_ops_render_parity.py`'s
`test_every_output_binding_names_an_output_its_producer_actually_registers`
checks every `#output_name` binding against the set its producer kind
actually registers.

`publication_effect` (P3-1c, rearchitecture phase-3 guide §5.4) binds a
Phase 3 serving projection candidate to this fenced publisher through one
more optional named `input_bindings` entry, `projection_binding.json` —
the smaller of the guide's two named operator-entry-point options ("a
coordinator step in `tools/v2_dashboard_project.py`, or a publication-
effect input"): no new job kind or DAG wiring, the same mechanism
`bundle.tar`/`finality.json`/`selfcheck.json`/`engineering_gate.json`
already use (`_publication_files`). Its content is an inert document
(`engine.v2.serving.projections.projection_binding` builds it; this
package never imports `engine.v2.serving`, keeping the layering intact) —
`tools/v2_dashboard_project.py` is what actually builds and emits it, and
an operator/submission script registers and binds it exactly like the
render bundle. Presence is the only branch: an ordinary bundle-only
publication (no projection candidate yet) behaves exactly as before.
Because `stage_release`'s `binding_hash` is computed over the exact `files`
dict passed to it, adding this file changes that hash, so every gate
`publication_effect` builds is freshly bound to the candidate this
publication now carries — gates minted for one generation's files never
validate a different generation's (`stage_release`'s `_gate_is_bound`
refuses on the `input_hash` mismatch, reported as `eligible: False`).
`engine.v2.serving.api`'s read-only current-resolver reads the published
`projection_binding.json` back (files only, no ops import) to answer
"current" for the Phase 3 board — see `engine/v2/serving/README.md`.
Tests: `tests/test_v2_serving_publication_binding.py` (ops+serving
together; `tests/` may compose both).

Writing the global `"shadow"` effect scope (`nightly.effect_scope_for`) must
be explicit: `ops plan nightly --full-run` records the declaration and
refuses unless `--tickers` equals `--context-tickers`; without it, every
plan gets a subset scope (`"shadow:" + hash(watchlist)`), even when the
watchlist happens to equal the context — equality alone is never inferred
as a full run. See the Phase 1 runbook's `plan nightly` row.

Settlement workers capture only newly appended legacy outcome bytes.  The
coordinator validates those observations against committed predictions and
imports them under the active fence; settlement is not checkpoint-shortcut
eligible because reuse must never skip its catalog effect.
Real scoring/parity runs are sequential private verification, outside the fast
commit hook; ops never imports their comparator.

`ops/snapshots.py` (P2-3) is the ops-side companion to the data catalog's
snapshot commit and exact resolve (phase-2 guide §7.3, §8.1).
`commit_snapshot_for_attempt` wraps `engine.v2.data.catalog.commit_snapshot`
with the real `lifecycle.verify_fence` as its injected `fence_check` — the
data package cannot import ops, so the fence check is supplied from here
instead of called there. `resolve_snapshot_head` reads one
`data_snapshot_heads` row, resolves it through
`engine.v2.data.repository.Repository` (never trusting the head row alone),
and publishes the verified `SnapshotRef` as a Phase 1 artifact with
`ops/checkpoints.py::register_artifact` — the same publish-then-register
shape `ops/plans.py::save_plan` already uses — so a job can pin
`JobSpec.input_refs` to one immutable document instead of a live, movable
head. Both are exercised end to end, with real SQLite and real Parquet
objects, by `tests/test_v2_data_commit.py`, `tests/test_v2_data_atomicity.py`
and `tests/test_v2_data_repository.py` rather than by an ops-local test file,
since every fault point and concurrency case they prove belongs to the data
catalog's own commit/resolve contract.

Snapshot-backed legacy stages (P2-6, phase-2 guide §9.3) replace the
mutable-store read-set barrier for exactly the kinds whose complete reads
`engine.v2.data.legacy_materialization.LEGACY_SCORE_READ_PLAN_V1` declares:
`legacy_score`, `legacy_score_requests` and `legacy_decision_replay`
(`stages.SNAPSHOT_BACKED_KINDS`). They are an input mode on the same kinds
(`parameters.input_mode == "snapshot"`), not new kind names; the kind validator
refuses the mode anywhere else. `legacy_finality`, `legacy_model_evidence` and
`legacy_selfcheck` stay on the barrier (`stages.BARRIER_ONLY_REASONS`), as do
decisions, settlement and render. One `legacy_materialize` job
(`materialization_worker.py`) writes a request's private read-only root once,
under `<ops_root>.materializations/<request hex>` — beside the operations root,
because `materialize` refuses any destination under the artifact store root —
and every later job for the same request re-hashes it and never rewrites it.
`snapshot_stages.py` is the supervisor side: before launch it validates the
bound `SnapshotRef`/request pair (request built for that snapshot, snapshot
still resolves, hash covers content, `read_plan_complete`), requires the bound
manifest to be one a `legacy_materialize` attempt committed for that request,
and re-verifies the root against it (`snapshot_roots.verify_root`: exact file
set, hashes, 0444/0555, no links). It skips `_pin_read_set`, the staging copy
and the legacy-store lease, and hands the worker that root as its only legacy
root. At finish it re-confirms the recorded bindings, the request hash and the
root's stat fingerprint. The checkpoint `inputs` fold in the snapshot manifest
hash, the request hash and the manifest artifact hash.
`ops plan nightly --input-mode snapshot --snapshot-scope <scope>` resolves the
head once (`snapshot_planning.py`) and takes the request's pinned reference
inputs (legacy SNAPSHOT, registry, structures, champion artifacts, Tier-4
serving caches, chooser analog pool, calendar) from the newest committed import
receipt for that snapshot (`engine.v2.data.reference_catalog`), refusing when
there is none;
the default `--input-mode legacy` graph is byte-identical to before. Tested in
`tests/v2/ops/test_v2_ops_snapshot_stages.py`.

The phase-1 engineering gate needs a coverage measurement passed in; run bare
it fails the coverage row with `COVERAGE_EVIDENCE_MISSING` by design (a missing
check is never green). The green-path command is:

```text
python3 checks/v2_coverage_ratchet.py --profile phase1 --measure --output /tmp/coverage.json
python3 checks/rearchitecture_phase1_gate.py --coverage /tmp/coverage.json
```
