# Rearchitecture Phase 4 — Native scoring parity

Status: implementation plan, 2026-09-16. Follow the
[delivery plan](rearchitecture_delivery_plan.md) and system design §§3, 4, 6.
This migrates existing economics; it does not improve a strategy or promote one.

## Outcome and prerequisites

One registered application produces immutable canonical score records for
single, event, batch, replay and research requests. The renderer, UI and
experiment runners consume its financial values. All eleven structure factories
and DYN-SV, including disabled/historical definitions and critical refusals,
remain represented.

Start fixtures/contracts against accepted Phase 2 snapshots while 3B progresses.
Coordinate the frozen inference interface with Phase 5 immediately. Phase 4
may test stages against saved forecast artifacts; final integration requires
Phase 5 loaders and complete frozen model/residual state. Do not implement an
interim fit-on-request path or make Phase 5 wait for a complete scorer.

## Ownership

| Package | Scope |
|---|---|
| `contracts/`, `foundation/` | Canonical request/result identity, exact serialization, pure shared conventions |
| `features/` | Registered input recipes, cutoff/missing policy, separate context scopes |
| `domain/generation/`, `scenarios/`, `valuation/`, `simulation/` | Extract existing selection, pricing and simulation mechanics at existing layer boundaries |
| `scoring/` | Execution graph, gates, complete chooser menu, canonical record and diagnostics |
| `models/` | Phase 5 owns artifact loading/inference; Phase 4 consumes its frozen interface |
| `serving/`, `evaluation/`, `ledger/` | Consume records; workflow migration belongs to Phase 6 |

Do not move legacy files wholesale. Keep the comparator independent, package
READMEs current and budgets at zero exemptions. Remove adapter edges only when
their consumers have actually migrated; historical imports can survive until 8A.

## Ordered implementation assignments

| Task | Deliverable | Acceptance / negative control |
|---|---|---|
| P4-1 Fixed inventory and contracts | Generate strategy/model-role mapping from frozen source; pin strategy/deployment/request/record schemas and dependency identity. | Exact round trips, null/zero distinction, operational timestamps outside numerical hash; changing geometry, clock, fill, model or residual state changes identity. |
| P4-2 Feature and context recipes | Extract current features with explicit event, regime, analog and calibration populations. | Narrow visible watchlist preserves full required analog/residual population; planted cutoff leak and changed missing mask fail. |
| P4-3 First complete strategy | Migrate STR-THRU from frozen request through selection, pricing, forecast/analog gate and diagnostics using the Phase 5 inference seam. | Stage-named corpus comparison on same inputs; direct and batched scoring agree; no render-time economics. |
| P4-4 Remaining factories and chooser | STR-RUNUP, all short-vol members, CAL-P/CND-P refusals and direct DYN-SV complete-menu resolution. | Geometry/expiry/fill parity; irregular ladder, exact mirrors, zero-quantity reference legs, ties, missing competitors, fallback and no re-gating controls. |
| P4-5 Financial outputs | Move fair premium, ratios, risk summaries, payoff and current simulation outputs from renderer into domain/scoring results. | Full-precision engine and separately rounded display parity; terminal vs planned-exit payoff and multi-expiry refusal/labels preserved. |
| P4-6 Shared application and handoff | Single/event/batch/replay entrypoints, supervised score batches, canonical outputs and compatibility serialization for Phase 6. | All strategies and refusals compare over fixed corpus and a full saved release; no production import of an experiment runner; complete report. |

Preserve the registered divisors, thresholds, selector order, clocks, draw counts,
residual populations, fill policies, gates and chooser fallbacks. Preserve
monthly fold forecasts separately from champion diagnostics. Fixing an economic
defect is a separately identified change with its own evidence, never hidden
inside parity extraction.

## What is sufficient before cutover

Existing selector-resolved candidate domains and existing calibrated simulations
are sufficient. Generalized exhaustive strike search, arbitrary scenario/path
valuation, new structures and estimator backends are Phase 8C/research work.
Implement the contracts needed by current consumers, including truthful
unsupported outcomes; do not build unused numerical engines to satisfy every
future example in the target design.

Conservative dependency invalidation is allowed initially. Canonical identity
must still include all relevant dependencies. No mutable latest lookup, rounded
replay input, independent JavaScript arithmetic or watchlist-sized statistical
population is permitted.

## Tests and completion

At P4-1 define a Phase 4 acceptance registry and runnable gate with explicit
test file ownership. Required test groups are canonical identity, context scope,
each strategy/refusal, chooser menu, financial diagnostics, direct-vs-batch
parity and no experiment imports. Every new comparator needs a planted defect.
These future tests/gate are deliverables, not existing executable commands.

Use the existing tier-0 corpus for fast stage checks and sequential real-code
comparisons for final acceptance. Compare full populations, keys, contracts,
verdicts, flags and null masks; use only existing declared numeric tolerances.
Record any inherited disposition separately from new results.

Handoff: canonical API and fixtures, strategy/deployment inventory, native
score artifacts, legacy-format projection mapping, dependency declarations,
remaining adapter inventory and measured batch resource profile. Phase 4 is
complete only with Phase 5 inference integrated and existing economic behavior
accounted for. Phase 6 owns switching workflow consumers, not rebuilding this
kernel.
