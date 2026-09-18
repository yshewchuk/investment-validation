# Rearchitecture Phase 3B — Incremental EOD data

Status: implementation plan, 2026-09-16; not implemented or accepted by this
document. The acceptance-evidence and documentation slice (R3B-1, R3B-2,
R3B-4, R3B-5, R3B-8) has a completed record as of 2026-09-18:
[`rearchitecture_phase3b_closeout.md`](rearchitecture_phase3b_closeout.md).
That record states plainly that it is bounded-scope evidence and that R3B-3,
R3B-6 and R3B-7 remain open — this plan is still not fully implemented or
accepted. Authority: [delivery plan](rearchitecture_delivery_plan.md) and
[system design §§5, 12](system_rearchitecture.md#5-data-storage-contracts-and-incremental-ingestion).
Phase 3A remains the [saved-score preview](rearchitecture_phase3_parity_launch.md).
Its L01–L14 gate does not cover this work.

## Outcome and boundaries

Make normal EOD refresh append and correct immutable data without rebuilding
all normalized history. Preserve the Phase 2 repository, snapshot identities,
bounded reads, knowledge modes and old releases. Prove the original gate:
no-op rewrites zero data partitions; append/correction/deletion produces the
same logical tables and downstream values as a clean rebuild.

This is required before cutover. Fine-grained downstream scheduling and stored
causal checkpoints are not: a conservatively complete suffix or full dependent
recipe rebuild is acceptable within measured resource limits. Never substitute
a short lookback for an expanding or recursive dependency.

## Inputs and ownership

Start from accepted Phase 2 manifests, query contracts and import/recovery
interfaces, plus the complete dataset/read inventory for current EOD consumers.
Reuse raw caches and provider adapters. Read system design §§5.2–5.5 and the
data/repository contracts; do not introduce another store or snapshot pointer.

| Owner | Files / responsibility |
|---|---|
| Data | `engine/v2/data/`: receipts, coverage, normalizer identities, revisions, manifests, changesets and query dependencies |
| Contracts | `engine/v2/contracts/`: strict versioned schemas, no I/O |
| Ops | `engine/v2/ops/`: supervised ingestion graph, quota admission, staging, retry and atomic promotion |
| Features/models | Phase 4/5 consume changesets; data declares affected inputs, never runs model training |
| Acceptance | New 3B checks/fixtures and private comparison report; existing Phase 2 evidence remains historical |

Read-only finality moves here, including per-ticker `covered_tickers`.
Model evidence belongs to Phase 5 and display projection to Phase 6. Keep
barriers around any remaining mutable adapter until its entire read set is
pinned; table migration alone is not permission to remove the barrier.

## Ordered implementation assignments

| Task | Deliverable | Exit evidence / negative control |
|---|---|---|
| P3B-1 Inventory and contracts | Per current EOD table: source, key, partition, units, coverage denominator, revision/finality priority, dependencies and consumer. Receipt, completed-coverage and changeset contracts. | Round trips and schema failures; a hole or partial symbol response cannot count as complete coverage. |
| P3B-2 Cache-first acquisition | Plan missing completed sessions and calendar discovery; persist raw hash and redacted receipt before normalization. Share provider budgets with existing jobs. | Same input does not refetch; legitimate empty distinguished from truncation, auth failure and unavailable symbol; failures do not advance watermark. |
| P3B-3 One table end to end | Start with `daily_market`: normalize by raw hash + normalizer version, merge affected keys/partitions, durable objects then atomic manifest + coverage commit. | Clean rebuild equals append and retry; interrupted write leaves old snapshot readable; no-op changes no data-fragment hashes or bytes. |
| P3B-4 Complete current dataset inventory | Apply the same mechanism to chains, price history, calendar/reference and other inventoried EOD inputs; preserve event IDs across supported revisions. | Corrections, tombstones, duplicate/conflicting revisions, missing contracts and BMO/AMC changes agree with clean rebuild; ambiguous event mappings refuse. |
| P3B-5 Dependencies and finality | Snapshot-bound chain dependency explanation; data change impact planner; native finality/coverage. | Advancing head cannot change an explained query. Include event/security lookup and quote fragments/columns/predicates for event-based and event-free requests. |
| P3B-6 Integrate and accept | Wire supervised refresh into candidate snapshots and complete invalidation requests for 4/5; operator append/correction/recovery runbook. | Real cached append/correction/no-op replay, injected publication failure, retained R1 readers during R2 promotion, measured resource profile, private report. |

Land each dataset slice before widening the inventory. The final scope is all
inputs needed by the current EOD workflow, not just the convenient first table.
Adapters can temporarily wrap unchanged normalization semantics, with explicit
read/write ownership and removal tasks; do not silently retain a nightly
full-table rebuild behind an incremental label.

## Required behavior

Coverage is an interval/set with an explicit denominator, not `max(date)`.
Failed/partial responses cannot promote coverage. Preserve ORATS requested vs
returned ticker checks, delayed EOD availability, quota headers on errors and
unsupported-symbol recording; preserve Polygon curl, shared pacing and legitimate
empty aggregates. Do not probe new entitlements or buy historical data for a
test that cached responses can support.

Revision resolution is deterministic from source/finality/revision policy,
never filesystem order. Retain raw revisions and tombstones. An event date
correction must not move an old prediction to another event/session. Do not
retroactively label reconstructed history observed.

Changesets carry keys, columns, time range, old/new hashes and revision kind.
They must cover residuals, analogs, target availability and later folds as well
as point features. Unknown dependencies invalidate conservatively or refuse;
they never imply an empty change. The first chain dependency consumer must
wait for P3B-5; `UNSUPPORTED_CONTRACT` remains honest until then.

A no-op may append an operational receipt; it normalizes zero already-known
payloads and rewrites zero data partitions. Append rewrites only affected
partitions, not twenty years of tables. Logical row equivalence is the rebuild
criterion when fragment layout differs. Keys, null masks, units and revisions
compare exactly; numerical tolerances remain the declared existing ones.

## Acceptance and handoff

Create a dedicated 3B acceptance entrypoint/registry and a fixed test inventory
when implementing P3B-1. Proposed test subjects: coverage receipts, incremental
merge, corrections/tombstones, snapshot dependency explanation, atomic commit
faults and rebuild equivalence. These are new deliverables, not runnable
commands already present in this checkout. Reuse Phase 2 repository/fault
tests instead of constructing another validation framework.

The final report includes each table population, raw/manifests used, no-op
normalization/write counters, changed partitions, full-rebuild comparisons,
negative controls, runtime/RSS/cache/contention, actual failures and retained
snapshot refs. Run frozen real data sequentially under the supervisor/resource
policy. Pure schema tests are insufficient.

Handoff to 4/5: snapshot + changeset contracts, query dependency APIs, coverage
and finality semantics, correction impact rules, and admitted resource profile.
Phase 3 as a whole closes only when 3A and 3B have separate completed records.
Defer compaction/GC, Arrow pushdown and narrower invalidation to Phase 8C unless
measurements make one necessary to meet the operating budget.
