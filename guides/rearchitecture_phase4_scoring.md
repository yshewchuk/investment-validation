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
comparisons for final acceptance. Compare the complete saved-release
population record by record, including refusals and missing outputs: request
membership, duplicates and missing rows; contracts, quantities, expiries and
execution dates; readiness, refusals, gates and chooser selections; nulls
versus zero; and full-precision forecasts, costs, simulation and financial
diagnostics using only existing declared tolerances. Check separately that
displayed rounding matches the canonical values. Legacy results belong only on
the expected-results side: native execution must calculate its own selection,
pricing, forecasts, simulation, gates and final records from the same frozen
request, data, models, residual/calibration state and configuration.

The saved release has one hash-verified manifest for shared frozen resources;
cases reference those resources rather than duplicating them. A fixed,
representative diagnostic corpus records only these hash-verified checkpoints:
model feature vector, missing mask and model identity; selected legs and entry
cost; simulation horizon, denominator, residual-population identity, draw
count and seed; gate inputs; and DYN-SV eligibility/ranking where applicable.
The diagnostic corpus explicitly covers every declared strategy and the
pre-expiry, multi-expiry, debit/credit, missing-input, override, tie,
fallback, and frozen-inference-through-canonical-application branches. Missing
checkpoint or executable inputs are an explicit incomparable disposition,
never a successful comparison. Each checkpoint comparator needs a deliberately
introduced defect that it rejects.

Exhaustive internal stage tracing is optional. It may help diagnose a mismatch,
but Phase 4 does not require serializing every internal operation or making
legacy and native internal layouts identical. It does require full saved-release
final-record parity, independent native calculation, the diagnostic corpus,
and focused regressions for planned-exit valuation, authoritative pricing in
gates, required forecasts, overrides, financial-output ownership, ladder
refusals, frozen inference, direct-versus-batch equality and shuffled input.
Record any inherited disposition separately from new results.

Handoff: canonical API and fixtures, strategy/deployment inventory, native
score artifacts, legacy-format projection mapping, dependency declarations,
remaining adapter inventory and measured batch resource profile. Phase 4 is
complete only when Phase 5 inference is integrated, the complete saved release
has no unexplained behavioral or numerical mismatch, the bounded diagnostic
corpus and its defect controls pass, and existing economic behavior is
accounted for. Phase 6 owns switching workflow consumers, not rebuilding this
kernel.
