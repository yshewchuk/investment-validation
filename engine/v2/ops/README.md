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
produce this evidence bundle, so this coordinator path is available for
isolated authority rehearsals but does not make the full nightly complete or
activate a production writer.

Settlement workers capture only newly appended legacy outcome bytes.  The
coordinator validates those observations against committed predictions and
imports them under the active fence; settlement is not checkpoint-shortcut
eligible because reuse must never skip its catalog effect.
Real scoring/parity runs are sequential private verification, outside the fast
commit hook; ops never imports their comparator.

The phase-1 engineering gate needs a coverage measurement passed in; run bare
it fails the coverage row with `COVERAGE_EVIDENCE_MISSING` by design (a missing
check is never green). The green-path command is:

```text
python3 checks/rearchitecture_phase1_coverage.py --measure --output /tmp/coverage.json
python3 checks/rearchitecture_phase1_gate.py --coverage /tmp/coverage.json
```
