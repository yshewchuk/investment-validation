# Rearchitecture Phase 6 — Consumer parity and operational rehearsal

Status: implementation plan, 2026-09-16. Authority:
[delivery plan](rearchitecture_delivery_plan.md), system design §§8–9, 11–13.
This phase completes existing workflows; it does not require rewriting every
screen in React.

## Outcome and entry

Every current consumer obtains native canonical outputs from the v2 pipeline.
The full current feature inventory works through new screens, compatibility
views, CLI or exports as appropriate to the existing access path. A local
board/detail screenshot is not sufficient.

Start the inventory now. Final integration uses accepted 3B data, Phase 4
scoring and Phase 5 deployments. Preserve the 3A API, immutable release binding,
pagination, authentication and lazy detail; do not build another serving stack.

## Required inventory

For every current capability record its old entrypoint, new entrypoint,
producer, snapshot/release identity, test and disposition. Required entries
include nightly refresh/score/selfcheck/publish/backup; predictions and
settlement; hypothetical/contrarian books and funding; replay/research and
generated reports; board filters/deep links; strategy/refusal details; model
evidence; history/analogs; flags/finality/health/quotas; and phone/offline export.
Use the Phase 0 screen/artifact inventory and actual source, not memory.

An existing interactive action needs an equivalent callable path. A frozen
screenshot cannot replace it. A CLI is suitable only where it preserves the
existing workflow or the user explicitly accepts that access change. New job
buttons, a strike explorer and new research views are post-cutover enhancements.

## Ordered implementation assignments

| Task | Deliverable | Acceptance / negative control |
|---|---|---|
| P6-1 Capability and adapter inventory | Complete mapping above; assign every adapter edge to a consumer/removal phase. | No unowned missing capability. Distinguish dormant historical readers from active official-path dependencies. |
| P6-2 Native nightly and command wiring | Existing supervisor graph consumes incremental snapshots, frozen deployment and native scorer; research/replay/evaluation use the same kernel. | Cold preparation handled by jobs; retry/cancel/resume idempotent; no request-time fitting; original generated report/ledger rules retained. |
| P6-3 Decision and accounting consumers | Prediction import/commit, settlement, portfolio/funding and compatible export against canonical records. | Duplicate/retry/same-session supersession tests; original decisions immutable; missing finality blocks settlement; totals agree with existing accounting. |
| P6-4 Serving and compatibility views | Complete Models/book/flags/operations/history/analog and other inventoried capabilities through existing/new views. | Same release end to end; current native outputs visible; null/refusal/stale semantics retained; old pinned legacy pages alone do not prove current capability. |
| P6-5 Existing access and recovery | Existing phone/offline workflow, private delivery/auth, consistent SQLite + referenced-object backup and restoration. | Offline opens without network; no secret/data in public assets; restored deployment replays a score and reconciles ledger/report. |
| P6-6 Full rehearsal and handoff | Real integrated EOD candidate, full population comparisons, runbook and Phase 7 session evidence collection. | Controlled failures keep last good release; measured runtime/memory with contention; every inventory row has evidence. |

## Compatibility is a presentation choice

Compatibility pages may remain after cutover, but they must consume the current
native release through an explicit maintained adapter. They cannot depend on
an old legacy nightly continuing to score or on mutable legacy forecasts.
Financial arithmetic moves to Phase 4; model evidence to Phase 5; accounting
to the existing semantics extracted into the evaluation/ledger boundary.
Presentation adapters may format values and render charts.

The Phase 3 bridge is temporary input plumbing. Replace its legacy-score input
with a canonical projection while retaining release IDs, provenance and tested
read contracts. Do not write financial formulas into React to fill missing fields.

All official-path dependencies must be inventoried. Compatibility rendering or
file-format adapters may remain if their behavior and read sets are frozen and
tested; their existence is not permission for a second authoritative scorer,
registry or ledger. Full physical adapter deletion is Phase 8A.

## Acceptance and qualified sessions

Create a Phase 6 capability matrix and acceptance entrypoint with subjects:
native nightly integration, research/replay/report parity, prediction/settlement
idempotency, book reconciliation, full screen inventory, browser release
consistency, offline/mobile access and restore faults. Reuse L01–L14 checks
for unchanged routes. A test suite for new components alone does not close
existing-consumer parity.

Measure a full current workload, not a six-ticker preview. Record workload,
provider/queue/compute time, cache state, available RAM, contention and deadline.
Agree the operating deadline from those measurements; do not fabricate a
latency gate from a historical contended run.

Start Phase 7 qualification when the integrated candidate satisfies its entry
requirements, even before writing the Phase 6 final report. Ten distinct
completed-session runs are required; old adapter-only nights are not eligible.
Missing sessions remain unknown and repeated attempts count once.

Handoff: capability matrix with evidence, active adapter/read-set inventory,
native job graph/deployment refs, backup/restore and authority-switch runbook,
measured resource policy, qualified session records and remaining Phase 8 work.
No production switch is performed merely by completing this phase.
