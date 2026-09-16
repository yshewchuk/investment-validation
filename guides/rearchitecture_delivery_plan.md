# Rearchitecture delivery plan: parity to cutover

Updated 2026-09-16 after reviewing `main` at `9a23bf4`, the working Phase 3
handoff edits, source packages, acceptance producers, and the private Phase 2
closeout record. This is a documentation assessment; no fresh scoring,
provider pull, acceptance run, or deployment was performed.

## Authority and goal

[System design §12](system_rearchitecture.md#12-migration-sequence-and-rollback)
and this plan define the same delivery sequence. The system design and
[component contracts](component_contracts.md) own architecture and semantics;
this plan owns scheduling and scope; phase guides own implementation tasks;
status files record evidence; runbooks record commands. A status note or old
task brief cannot add a phase requirement. Change both this plan and §12 when
moving work between phases, and update the affected phase guide in that change.

The goal is to make v2 the official EOD pipeline while preserving current
strategies, model behavior, decisions, research/replay workflows, and usable
dashboard/export access. A preview is useful progress, but is not cutover.
Cutover does not require intraday trading, a redesigned version of every
screen, a universal training framework, or deleting the old deployment.

## One sequence, with stable identifiers

| Phase | What lands | Exit / handoff | Detailed plan |
|---|---|---|---|
| 0–2 | Baseline, supervisor, immutable snapshot repository and compatibility adapters | Existing recorded acceptance; keep explained differences visible | Existing Phase 0–2 guides |
| 3A | Finish the already implemented saved-score preview | Reproducible full-population release, usable board/detail, refresh/rollback, assembled launch evidence | [Preview plan](rearchitecture_phase3_parity_launch.md) |
| 3B | Minimum incremental EOD data path | No-op writes zero data partitions; append/correction/deletion equals clean rebuild; atomic coverage and snapshots | [Data plan](rearchitecture_phase3_incremental_data.md) |
| 4 | Native scoring and existing financial semantics | All strategies/refusals agree; one kernel for batch, replay, research and serving | [Scoring plan](rearchitecture_phase4_scoring.md) |
| 5 | Frozen model inference and current training lifecycle | No fitting or cache writes during scoring; exact fold/residual evidence; deployment rollback | [Models plan](rearchitecture_phase5_models.md) |
| 6 | Complete consumer parity and operational rehearsal | Every existing workflow accounted for; official nightly candidate, phone/offline access and restore drill | [Consumer plan](rearchitecture_phase6_consumer_parity.md) |
| 7 | Production cutover | Ten qualified completed-session runs, one official writer, verified switch and rollback | [Cutover plan](rearchitecture_phase7_cutover.md) |
| 8 | Post-cutover completion and extensions | Independently accepted cleanup, UI enhancements, efficiency work and live shadow | [Post-cutover plan](rearchitecture_phase8_post_cutover.md) |

Phase 3 is **3A plus 3B**. Existing P3-0…P3-4 task IDs, L01–L14 receipts,
`phase3_evidence.v1.0`, and `checks/rearchitecture_phase3_gate.py` keep their
names and cover **3A only**. Do not rename schemas or rewrite old receipts.
Passing that gate never proves incremental ingestion or all of Phase 3.

Numbering crosswalk: the original design Phase 3 is now 3B; its early UI
slice is 3A. Phases 4–6 retain their subjects. Original Phase 7 live shadow
moves to **8D**, after cutover. Original Phase 8 splits into **7** (authority
switch) and **8A** (physical deletion/rename). Guides named `phase3_dashboard`,
`phase4_implementation`, etc. without `rearchitecture_` belong to the older
research program and do not schedule this migration.

## Current position and immediate queue

| Observation | Evidence inspected | Consequence |
|---|---|---|
| Phase 2 accepted with explained D14/D15 differences; strict gate remains red | Private `reports/phase2_closeout/FINAL.md`, candidate `f0e631a` | Preserve the acceptance and the original red receipts. No repeated attempt to make stale legacy prices agree. |
| Preview, serving bridge/API, React board/detail and receipt producers merged | `b1bf750`; `engine/v2/serving/`, `ui/`, Phase 3 checks | Finish acceptance; do not rebuild these components. |
| Replay precision fixed; fresh bridge comparison succeeds | `f8b22d9`, `9a23bf4`, [status](rearchitecture_phase3_status.md) | Retain the full-population fix; an old subset is not acceptance. |
| Fresh D19 replay reports 11 mismatches in 20 sampled rows | Current Phase 3 status handoff | Repair and prove the frozen forecast/model boundary before accepting this release. Earlier D19 cannot attest it. |
| Coverage baseline and assembled Phase 3 evidence missing | `rearchitecture_phase3_quality.py`, current status | Close these once on the final candidate. |
| Phase 3 prerequisite validator requires clean Phase 2 validation | `rearchitecture_phase3_evidence.py::_check_phase2` | Represent the existing user disposition without changing the strict Phase 2 gate or fabricating agreement. |
| Native features/scoring/models are still skeletons | Package READMEs and `__init__.py` files | Preview readiness is not evidence that native migration is nearly complete. |

Next tasks, in order: (1) bound the replay defect and acceptance-disposition
handoff; (2) freeze one candidate, assemble 3A evidence and finish its runbook;
(3) implement 3B by dataset slice; (4) migrate one complete strategy through
4/5, then extend over the fixed inventory; (5) wire all consumers in 6;
(6) complete the session qualification and switch in 7. Continue safe
synthetic implementation while waiting for a scheduled real run.

## Critical path and scheduling rules

Phases specify ownership, not an instruction to keep everyone idle until the
previous report prints. Phase 4 contract/fixture work and Phase 5 artifact
inventory can start from Phase 2 snapshots while 3B lands. Phase 5 frozen
inference preparation must precede final Phase 4 integration; use saved model
outputs at intermediate seams, never introduce a fit-on-request dependency.
Phase 6 inventory and static UI work can start now. Final 6 acceptance needs
3B/4/5 outputs. One owner coordinates shared contracts and publication wiring.

The ten-session operational gate remains required, but **start collecting
qualifying sessions as soon as the actual cutover candidate is integrated**,
including during Phase 6. Count distinct completed market sessions, never
retries, synthetic dates, or the older compatibility preview. Unrelated docs
and presentation changes do not automatically reset the streak; a change to
scoring, data, models, publication, authority or resource behavior does unless
equivalence evidence establishes that the qualified path is unchanged.

Do not promise a completion date from unit-test counts. The fresh replay issue
is unbounded until diagnosed, native migration is substantive, and ten actual
sessions is a calendar constraint. Limit each implementation assignment to
one deliverable with owned files, dependencies, tests, negative controls and
an evidence handoff. Do not start broad redesigns while a bounded blocker is
open. Use saved artifacts for UI iteration; perform expensive comparisons
sequentially and only when their implementation or inputs changed.

## Acceptance without recurring closeout loops

Each phase produces a private report with source/environment/input identities,
population counts, comparison receipts, negative controls, actual gate status,
unresolved findings, accepted dispositions and the next phase handoff. New
checks proposed in phase guides are deliverables, not existing commands.
Update the runbook with commands actually tested when each lands.

Bind evidence to the producer it measures. A presentation change needs fresh
UI checks, not historical retraining. Do not relabel old receipts to a new
hash; if an existing gate has a broader code-hash rule, either satisfy it once
on the frozen candidate or explicitly implement and test a narrower dependency
binding. Never bypass it informally. Pin one integration candidate before the
heavy final sweep, rather than recapturing after every documentation iteration.

An explained difference is not agreement. The existing D14/D15 user disposition
applies only to its recorded candidate, populations and stale-price cause.
Phase 3A must preserve that provenance in its acceptance handoff and still
reject new unexplained differences, missing artifacts and the fresh D19
failure. No gate change may silently turn the strict Phase 2 report green.

## What waits until after cutover

| Deferred work | Owner | Interim guarantee |
|---|---|---|
| Delete legacy tree; rename `engine/v2`; remove archival/import-only adapters | 8A | Old deployment retained for rollback; official consumers use v2. |
| React replacement of every screen; new visualizations, what-if explorer and command UI | 8B | Existing capabilities remain accessible through parity-tested compatibility views/CLI/export. |
| Fine-grained downstream recomputation, state checkpoints, compaction/GC, alternate databases | 8C | Conservative complete invalidation and bounded replay preserve correctness and admitted runtime. |
| Generalized training backends, new generators/valuation models, search strategies | 8C / separate research | Existing algorithms, model roles and selectors retained. |
| Intraday provider entitlements, new clocks, live collection and paper/live qualification | 8D | EOD semantics and knowledge labels preserved; no live readiness claim. |

These are scheduled obligations or extensions, not hidden cutover gates.
Promote a deferred optimization into the critical path only when measurements
show parity or the agreed operating budget cannot otherwise be met.
