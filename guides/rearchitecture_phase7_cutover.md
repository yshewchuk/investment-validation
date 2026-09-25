# Rearchitecture Phase 7 — Official EOD cutover

Status: implementation plan, 2026-09-16. This is the authority-switch portion
of the former Phase 8. Live shadow moved to Phase 8D. Follow the
[delivery plan](rearchitecture_delivery_plan.md); this document is not a claim
that production activation has already been authorized or performed.

## Entry gate

The candidate has accepted incremental data (3B), native scoring (4), frozen
models/current training (5), and complete current consumer parity (6).
The 3A preview evidence is retained but cannot substitute for these gates.
Known accepted differences remain individually visible with original receipts
and user dispositions; new unexplained economic differences block cutover.

Legacy remains the only official prediction/settlement/publication writer until
the controlled switch. Shadow execution cannot promote champions, commit
official decisions or mutate research multiple-testing ledgers.

## Ordered implementation assignments

| Task | Deliverable | Acceptance / negative control |
|---|---|---|
| P7-1 Candidate and authority manifest | Exact code/config, snapshots, model deployment, job graph, consumer inventory, scheduler/credential ownership and old deployment refs. | No implicit latest dependency; no unaccounted scheduled writer or background job. |
| P7-2 Qualify completed sessions | Ten consecutive completed-session runs on the integrated cutover path under automatic resource admission. | Real dated evidence, full expected population, selfcheck/engineering/publication success; retries count once; missing sessions unknown; manual CPU placement disqualifies a run. |
| P7-3 Rehearse switch and rollback | Scratch/private-target drill using retained deployment, data/catalog backups, pointer and writer-lease/fence changes. | Interrupted switch never leaves two official writers; rollback reconciles new decisions instead of deleting history. |
| P7-4 Review concrete release | Candidate report and exact tested switch/backout commands, downtime expectation, snapshot/deployment refs and open dispositions. | All entry requirements and qualification records present; no untested placeholder command. |
| P7-5 Switch and observe | Stop/drain old official schedule, verify no in-flight writer, fence old authority, activate v2 schedule/consumers, verify release and next completed session. | Exactly one official writer; native result/ledger/export reconciliation; user access and last-good recovery work. |

Phase 6 can perform P7-1/P7-2 preparation and start the qualification window
before its report is finalized. Keep the ten-session requirement; avoid adding
another ten-session wait after a successful switch. Cosmetic/docs changes
need not reset a proven path. Any material change to data, model, score,
publication, authority or resource behavior requires a documented impact
decision and fresh qualification unless equivalence proves the path unchanged.
Do not grandfather compatibility-only preview nights.

## P7-1 manifest checker

`checks/phase7_candidate_manifest.py` validates one candidate/authority
manifest against schema `phase7_candidate_manifest.v1.0` (documented in the
module docstring) by reading and hash-checking the referenced files it can
reach, never trusting summary booleans: exact code/config identity, pinned
data snapshot and model deployment (the referenced snapshot and release
documents must be JSON objects declaring exactly that identity), native job
graph (rows strictly typed with string id/schedule and a real boolean
``writes_official``) and consumer inventory refs, old and proposed
schedule/writer/credential ownership validated as one pair per side (the
sides' values cannot be swapped), with retained old deployment refs, and
hash-verified 3B/4/5/6 phase evidence that must be JSON objects binding the
exact candidate commit and, for Phase 5, the deployed release (missing
bindings fail closed; 3A preview evidence cannot substitute). It rejects
implicit latest refs, duplicate or unaccounted scheduled writers/background
jobs, conflicting writer ownership, and stale or malformed evidence. It is
read-only: it never writes authority, production pointers, credentials or
release state. A green run validates manifest consistency only; it is not a
Phase 7 readiness claim, and the tests run on synthetic fixtures
exclusively.

    python3 checks/phase7_candidate_manifest.py MANIFEST.json [--root DIR] [--json]

## Switch and rollback protocol

Before P7-4, prepare all reversible work and a concrete reviewable result.
Apply the standing session authorization and deployment permission rules at
the actual activation step; this plan alone does not switch production.

The tested runbook must name real service/scheduler commands, not guesses:
capture current authority and ledger high-water marks; stop/drain the old
writer; take a consistent catalog/WAL backup with referenced objects; verify
the restore target; transfer the authority fence; start the new schedule;
switch read consumers to the validated release; check authentication,
freshness, predictions, settlement, book and export identities.

On failure, stop/fence the new writer first. Reconcile any committed v2
decisions and settlements before resuming the old deployment. Never restore
an old database blindly over new official facts, erase history, or let the old
scheduler repeat a completed decision. Rehearse rollback both before and after
one official commit; a pointer-only rollback is not proof of ledger recovery.

Rollback triggers include unexplained economic drift, missing population,
duplicate authority, unreconciled decisions, unrecoverable publication or
failure to meet the agreed session deadline. If facts cannot be reconciled,
hold writes and keep the last validated read release; do not guess.

## Acceptance record

A Phase 7 readiness checker/report is an implementation deliverable. It must
verify referenced candidate, phase evidence, qualification sessions, restore
and switch/rollback receipts, authority state and capability inventory rather
than trusting summary booleans. Add controls for duplicate session counting,
a missing session, stale candidate refs and competing official writers.

Cutover is complete after the controlled activation and a verified official
completed-session run, with rollback still available. Record exact activation
time, authority generation, released snapshot/deployment, ledger high-water
marks, run results and rollback retention owner. Retain the old deployment
for an explicit rollback window; its duration is set in the concrete release
runbook before switching.

Do not delete/rename code in the cutover change. Phase 8A handles it after the
rollback window. No intraday entitlement purchase, live model qualification
or trading-capital decision belongs to this EOD migration gate.
