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
python3 -m pytest -n auto --dist loadgroup -q tests/test_v2_*.py tests/test_checks_phase2_gate.py
```

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
adding pytest-cov. The default stays serial. `--parallel` was implemented but not yet run for
real: installing `pytest-xdist` here hit PEP 668 (`externally-managed-environment`)
and was left for a human decision rather than overridden with
`--break-system-packages`. Once installed, verify `--parallel`'s
executed/executable counts match the serial measurement on two consecutive
runs each before trusting it; if they differ, keep both scripts serial-only.

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

The phase-1 engineering gate needs a coverage measurement passed in; run bare
it fails the coverage row with `COVERAGE_EVIDENCE_MISSING` by design (a missing
check is never green). The green-path command is:

```text
python3 checks/rearchitecture_phase1_coverage.py --measure --output /tmp/coverage.json
python3 checks/rearchitecture_phase1_gate.py --coverage /tmp/coverage.json
```
