# `engine/v2/ops`

## Ownership

Implements the **Supervisor/catalog — transactions, leases, dependencies, capacity, retry history** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**7** of §4.1.

Replaces (§4.4): `new supervisor and catalog`, `dashboard/nightly.py becomes a job graph`, `tools/bounded_run.py becomes an executor adapter`.

## Responsibilities

- Durable job submission, leases, retry history and dependencies.
- Resource admission and per-job CPU placement.
- The nightly job graph and its release boundary.

## Non-responsibilities

- **Decide research conclusions** — `engine/v2/evaluation` does it instead.
- **Compute a score** — `engine/v2/scoring` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

The operator interface is the versioned command protocol exposed by
python3 -m engine.v2.ops. It supports immutable planning/submission, job and
attempt inspection, cancellation, recovery planning, health and supervisor
execution. Python modules are internal to ops; other production packages consume
versioned artifacts instead of importing the supervisor.

<!-- public-interface: none -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

No other production package imports ops. The CLI starts the coordinator, and
serving reads a versioned health artifact without a peer-layer import.

<!-- consumers: none -->

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

## Testing

Recommended fast invocation, once `pytest-xdist` (pinned in
`requirements-dev.txt` -- test-runner tooling, kept out of `requirements.txt`
because that file is gated byte-for-byte against a frozen Phase 0 baseline)
is installed:

```text
python3 -m pytest -q -p no:cacheprovider -n auto --dist loadgroup tests/test_v2_*.py tests/test_checks_phase2_gate.py
```

Measured on this host (12 cores, shared with other agents -- times vary with
load): serial 70-105s; three consecutive parallel runs at 39.95s (load
average 1.08), 33.78s (load 6.66), 33.26s (load 6.89) -- 567 passed, 1
skipped, every time, no parallel-only failures. Re-run three times after any
change to the grouping in tests/conftest.py to catch a new parallel-safety
bug before it lands.

`-n auto --dist loadgroup` is recommended ONLY for `tests/test_v2_*.py
tests/test_checks_phase2_gate.py` above. The LEGACY suite
(`tests/test_calendar.py`, `tests/test_dashboard.py`, `tests/test_features.py`;
274 tests, ~100s serial, per main's committed data) runs SERIALLY, full
stop -- it is not part of the v2 command above, needs the real data trees
(`earnings_predictions/`, `polygon_cache/`) a bare worktree checkout does
not have, and a spot check under real data found a genuine parallel-safety
bug: `python3 -m pytest -q -n auto --dist loadgroup tests/test_calendar.py
tests/test_features.py tests/test_dashboard.py`, from a real data checkout,
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
tests/test_dashboard.py` (plain serial; main's committed count is 274
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
unchanged: `python3 -m pytest -q tests/test_v2_*.py tests/test_checks_phase2_gate.py`.

The one real end-to-end Phase 2 check — a fresh coverage measurement of the
REAL registered suite against the REAL tree, fed into the REAL gate, proving
it is red for exactly the rows lacking real evidence — is deliberately not
part of the unit suite (it needs a real `coverage run`, which the fast unit
tests must not pay for on every run). Run it directly instead:

```text
python3 checks/rearchitecture_phase2_coverage.py --measure --output /tmp/p2cov.json
python3 checks/rearchitecture_phase2_gate.py --coverage /tmp/p2cov.json --json
```

Today this is red with `MISSING_EVIDENCE` for D13/D14 (no real test file yet)
and D15/D16/D19 (tier-2 rows still needing a real receipt), and clean for
every other registered row. `tests/test_checks_phase2_gate.py`'s own smoke
test only checks the registry against the real tree cheaply (file existence
plus one `pytest --collect-only`, a few seconds) — it does not run this
measurement.

Both coverage scripts (`rearchitecture_phase1_coverage.py`,
`rearchitecture_phase2_coverage.py`) accept an opt-in `--measure --parallel`
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
leaves open — and `tests/test_v2_ops_render_parity.py`'s
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
`tests/test_v2_ops_snapshot_stages.py`.

The phase-1 engineering gate needs a coverage measurement passed in; run bare
it fails the coverage row with `COVERAGE_EVIDENCE_MISSING` by design (a missing
check is never green). The green-path command is:

```text
python3 checks/rearchitecture_phase1_coverage.py --measure --output /tmp/coverage.json
python3 checks/rearchitecture_phase1_gate.py --coverage /tmp/coverage.json
```
