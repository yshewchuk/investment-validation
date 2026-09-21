# Rearchitecture Phase 8 — Post-cutover completion and extensions

Status: implementation plan, 2026-09-16. Authority:
[delivery plan](rearchitecture_delivery_plan.md). These workstreams begin after
Phase 7, independently where their dependencies allow. They do not reopen the
EOD cutover gate. The target design remains the architectural destination.

## 8A — Retire compatibility dependencies and clean up the tree

Entry: cutover verified, agreed rollback window elapsed (Phase 7 switches the
official writer; 8A deletes the legacy tree only after that window -- the gap
is deliberate, since you cannot roll back to code you have already deleted),
restore/replay evidence retained and current consumers inventoried.

1. Remove remaining compatibility/import-only adapters one at a time, preserving
   historical artifact and CLI readability through supported formats.
2. Prove no scheduled, official, research, replay, export or restore consumer
   requires the old runtime. Drive the exact-symbol legacy adapter ledger to zero.
3. Preserve a recoverable archived deployment and private artifacts. Remove the
   legacy tree and mechanically rename `engine/v2/` to `engine/` in a separate
   change from numerical or storage behavior.
4. Update entrypoints, import map, READMEs, backup/restore and operator docs.

### How a retained legacy function leaves the legacy tree (user decision, 2026-09-20)

Every function retained from `engine/` leaves the legacy tree one of two ways,
decided by whether it is PURE or REACHES OUTSIDE ITSELF -- never by leaving it
in place:

- **PURE** = its result depends only on its arguments: no file, network or
  store access; no read of global or module-level mutable config; no mutation
  of shared state; deterministic. Pure functions are MOVED VERBATIM into
  `engine/v2/`, not reimplemented, and the move must be proven byte-identical
  on real inputs. Rewriting a working, tested calculation earns nothing and
  can only introduce defects: every writer defect found on 2026-09-20 was
  introduced BY the migration, not present in the legacy function beforehand.
  Stability argues for moving rather than rewriting; it never argues for
  leaving the function where it is.
  **Amendment (user decision, 2026-09-20):** non-determinism of any kind --
  clock, randomness, or mutable state read from anywhere -- makes a function
  impure, and therefore a REWRITE candidate, regardless of whether it touches
  the store, filesystem, network or global config. This closes a gap in the
  definition above: a function using `time.time()`, `datetime.now()`,
  `random`, or reading mutable class or instance state touches none of the
  listed things and would otherwise classify as PURE. Operational time
  leaking into results is a defect class this repo already guards against
  elsewhere -- replay identity and parity work exists precisely because a
  result that depends on when it ran cannot be replayed. A function moved
  verbatim despite being nondeterministic would reintroduce that quietly.
- **REACHES OUTSIDE ITSELF** = touches the mutable legacy store, the
  filesystem, the network, global config, or executes a legacy pipeline.
  These are REWRITTEN natively in `engine/v2/`, because a move cannot
  preserve behaviour when what the function reads has to move (or be
  retired) with it.
- **Classify from the call graph, not the import block.** A function-level
  (indented) `import` hides a legacy dependency from a top-of-file scan.
  `tools/phase5_training_job.py` pulls roughly ten legacy modules this way,
  at lines 96, 120, 121, 128, 136, 143, 150, 154, 159; a `^from engine\.`
  grep (anchored to line start) misses every one of them. Trace what a
  function actually calls, transitively, before classifying it as either
  pure or reaching.

Item 2's drive-to-zero above is this rule applied exhaustively: every
retained function is either moved (pure) or rewritten (reaching) so that
nothing legitimately needs the legacy runtime by the time 8A removes it.

Acceptance: tier-0 corpus and real replay unchanged by rename, zero legacy
adapter edges, import/budget/hygiene/coverage checks, current consumer smoke
checks and archived restore. A reference to an old private report is not a
reason to delete the report. Destructive cleanup follows the authorization
applicable when performed.

## 8B — Finish new UI and interactive capabilities

Migrate compatibility Models, book, health/flags, history/analog and research
views to React one screen at a time; preserve the Phase 6 capability matrix.
Then add new what-if/strike exploration, derivation views, job controls,
phone-install improvements or additional delivery surfaces when wanted.

Each screen consumes canonical projections with lazy fetches, release-aware
cache keys, auth and truthful loading/error/stale states. Expensive requests
return supervised job IDs; browser focus cannot initiate paid collection.
Acceptance: existing capability/browser parity, bounded initial requests,
mobile/offline regressions, access-control and cancellation tests. New financial
results must come from registered domain/scoring operations.

## 8C — Incremental efficiency and reusable research extensions

Use measured bottlenecks to order exact dependency invalidation, causal state
checkpoints, partition compaction/GC, filter pushdown, object verification cache
or storage changes. Correct invalidation already landed in 3B/4/5; this work
reduces recomputation, not the dependency domain.

Each optimization needs same-input equivalence, no-op/change counters,
crash/recovery controls and runtime/RSS measurements including contention.
GC additionally proves retention for releases, experiments, ledgers and active
readers. Consult the [debt registry](rearchitecture_tech_debt.md).

Generalized generators, scenarios, valuation, training backends and richer
dataset recipes follow the [reusable domain contracts](structure_generation_and_simulation.md).
Preserve existing selector-resolved behavior; new search/ranking or economics
requires a separately registered strategy/experiment with generated reports,
fill sensitivity and normal promotion rules. Do not implicitly promote a
different strategy through an infrastructure enhancement.

## 8D — Live shadow (former Phase 7)

Follow [live intraday scoring](live_intraday_scoring.md) and system design §10:
prove provider entitlements/schema/clock first; capture immutable raw live
responses with genuine receipt times; implement causal clock-specific features
and frozen deployments; run shadow/paper collection with quota and deadlines.
Keep provisional intraday objects separate from final EOD data.

Acceptance requires the live guide gates, timely publication, causal provenance,
no contamination, generated reports and shadow/paper evidence. An EOD model
or a Phase 7 infrastructure cutover is not live model qualification. Brokerage
execution or capital allocation retains its independent approval/go-live rules.

## Completion records

Each workstream has its own task inventory, acceptance report and runbook
update; there is no single giant Phase 8 release. Core rearchitecture completion
means parity, incremental normal operation, registered native scoring/model
workflows and physical cleanup are evidenced. Optional new UI/research/live
capabilities each remain separately labeled until their own gates pass.
