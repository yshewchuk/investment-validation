# Rearchitecture Phase 3 — First dashboard with real scores

Date: 2026-09-13. Status: implementation guide; Phase 2 is still in progress.

## 1. Objective and sequencing decision

Deliver a usable, locally accessible v2 dashboard with real saved scores as
soon as Phase 2 supplies a validated shadow release. The first screen is the
events/strategy board; selecting a score opens its details. Values must agree
with the legacy board built from the **same inputs**, including refusals,
nulls, selected structures, and forecast/gate distinctions.

This guide applies the user priority of 2026-09-13: parity and something
visible come first; work unnecessary for the initial dashboard is deferred.
It deliberately changes the build order in
[system_rearchitecture.md §12](system_rearchitecture.md#12-migration-sequence-and-rollback).
The original Phase 3 incremental-data work remains required for the completed
architecture, but is **not a prerequisite for this launch**. Pull the smallest
read API and UI slice forward from Phase 6. Use Phase 2 scoring and rendering
adapters until their native replacements pass parity in Phases 4 and 5.

This is not the older program guide `phase3_dashboard.md`.

| Milestone | User can inspect | What it proves |
|---|---|---|
| P3-0: immediate preview | Existing legacy views inside the v2 operations surface, pinned to a validated shadow release | Real output from the supervised snapshot-backed path is visible. This is a compatibility preview, not the new React UI. |
| P3-4: initial v2 dashboard | New paginated board and score detail, with a link to that same legacy release | Display/data parity for the shipped views, release consistency, and a repeatable refresh handoff. This is not full UI parity or native-scoring cutover. |

Complete P3-0 before building more infrastructure. Show the user the preview
and its as-of date immediately. Do not wait for incremental ingestion, native
scoring, training extraction, every legacy screen, or ten stable nights.

Initial launch means a private, read-only shadow dashboard. Legacy remains
the sole official prediction, settlement, and publication authority. Scores
produced by the unchanged legacy scorer through v2 are real scores; label the
producer honestly. Do not call them native v2 scores or execution fills.

## 2. Required context and starting state

Read [guides/README.md](README.md) and applicable `AGENTS.md` instructions.
For each task, read the relevant design sections and source below.

| Work | Read |
|---|---|
| Scope and ownership | System design §§3, 4.1–4.7, 6.2–6.4, 12–13 |
| Phase 2 handoff | [Phase 2 guide](rearchitecture_phase2_data_access.md) §§9.4, 12, 14; especially D14–D20 |
| Responses and release semantics | [Component contracts](component_contracts.md) §§2, 9.3–9.5, 13, 15 |
| Private fixture inventory | [Phase 0 guide](rearchitecture_phase0_baseline.md) §§5–9 |
| Serving and publishing | `engine/v2/serving/operations.py`, `engine/v2/ops/publication.py` |
| Saved payloads | `engine/v2/ops/legacy_adapter.py` scoring/render actions; `engine/dashboard/render.py` |

Source inspection for this guide used commit `704c007`. Phase 2 is actively
changing these seams: verify its completed public interfaces before wiring
them, and record the accepted commit in the handoff. Do not copy an old
signature from this guide over a newer accepted implementation.

Verified facts at that commit:

- `serving.operations.create_server(address, *, token, health_path,
  release_root, frozen_at="unknown")` already serves authenticated release
  files and health. Its shell embeds the legacy UI. Reuse it for P3-0.
- `ops.publication` already materializes immutable releases and uses a fenced
  `CURRENT` pointer with bound gate receipts. Do not invent another publisher.
- `_action_score` saves `score.json` with full-precision `rows`, `ladder`,
  planned/observed populations, and context metadata. It uses
  `json_safe(..., round_to=None)`. These rows, their job inputs, and Phase 2
  snapshot refs are the score evidence; `board.json` is a display projection.
- `render_bundle` writes `data/board.json`, `data/tickers/<ticker>.json`, and
  `meta.json`, `health.json`, `flags.json`, `strategies.json`, `book.json`, and
  `models.json` beneath `data/`. Ticker payloads contain events with rows,
  plus history and analog sections. Most display values are rounded to six
  places; replay input exceptions are separately preserved.
- `compact_row` computes display ratios and `payoff_curve` during rendering.
  Consume their **already rendered output** for now. Moving those calculations
  into the domain is Phase 4 work, not a reason to duplicate formulas.
- `dashboard/earnings_app.py` includes synchronous scoring/refresh routes.
  Do not mount that application as the new read API.
- A `ui/` React application did not yet exist. Some Phase 2 repository and
  nightly completion work was also outstanding. A guide is not evidence that
  those deliverables have landed.

Synthetic API/UI work can start while Phase 2 finishes. A real-data launch
requires its accepted snapshot, score/render parity evidence, and locally
published shadow release. A missing handoff is reported as missing; do not
bypass the Phase 2 gate or call a mock-data dashboard ready.

## 3. Scope boundary

### 3.1 Ship now

1. A launcher for a pinned compatibility preview using existing release files.
2. A strict, versioned bridge from one accepted score batch and its matching
   legacy bundle to a small serving projection.
3. A read-only FastAPI application with release, event, event-score, score
   detail, and operations-health routes.
4. One React/TypeScript board and one lazy score-detail view.
5. Parity, release-switch, pagination, failure, privacy, and browser checks,
   including a real saved-release demonstration.
6. A runbook to project the next completed Phase 2 shadow release and return
   to the previous release without changing legacy authority.

### 3.2 Defer explicitly

| Work | Later owner | Why initial launch does not need it | Trigger to resume |
|---|---|---|---|
| New provider ingestion, raw receipts, coverage watermarks, changed-key merges, automatic historical corrections/deletions | Original Phase 3 incremental-data continuation | Phase 2 can pin accepted legacy output; the UI reads saved scores | After initial dashboard acceptance; earlier only if no accepted data can be supplied |
| Fine-grained feature/model invalidation and causal state checkpoints | Phase 3 continuation with Phases 4/5 | Recompute a whole bounded shadow score batch when its inputs change | Refresh cost becomes the measured bottleneck |
| Native feature/scoring/chooser extraction and full canonical `ScoreRecord` production | Phase 4 | Existing validated scorer produces the required values | Replace one compatibility boundary at a time after preview |
| Training recipes, persisted fold/residual artifacts, promotion/rollback improvements | Phase 5 | Existing champion artifacts are pinned; no promotion is needed | Native inference migration or an independently requested model change |
| Models, book, explorer, derivation, flags pages as new React screens; portfolio aggregates; rich payoff interactions | Remaining Phase 6 | Existing views remain accessible in the pinned legacy release | Board/detail parity passes; migrate one screen at a time |
| Job submission, ad-hoc scoring, overrides, automatic refresh buttons, live collection | Later Phase 6/7 | Read-only scores meet initial use | Read surface accepted and command contracts implemented |
| New offline package, phone-install flow, remote delivery, host migration | Remaining Phase 6/operations | Local private preview plus existing exports suffice | Remote rollout requested or full UI parity being accepted |
| Generalized structure search, new valuation/scenarios, strategy changes | Domain/research work | These add capabilities beyond parity | Separate implementation/experiment request |
| Compaction, GC, alternative database engines, general frontend component framework | Post-launch optimization | Existing immutable storage and a small UI suffice | Measured constraint, not architectural completeness |
| Delete adapters/legacy tree, rename v2, switch official decisions | Phase 8 | Shadow launch retains rollback and comparison | All relevant migration gates pass |

Carried from Phase 2 into the incremental-data continuation: Phase 2 added
`covered_tickers` to legacy `engine/data/finality.py` so decision evidence
carries genuine per-ticker finality (P2-5 task 3). When finality moves into
`engine/v2/data`, move this per-ticker coverage computation with it and
retire the legacy helper and its adapter-ledger entry.

Do not put required later-phase work into
[rearchitecture_tech_debt.md](rearchitecture_tech_debt.md) as a nice-to-have.
Keep it assigned to its phase. No deferred task may silently become a launch
gate. Privacy, provenance, population checks, and atomic publication remain
launch requirements.

## 4. Implementation ownership

Paths below are proposed additions unless already present. Keep existing
package boundaries and zero budget exemptions.

| Path | Responsibility |
|---|---|
| `engine/v2/contracts/serving.py` | Schema-only preview release, event/page, score bridge, and projection receipt types |
| `engine/v2/serving/bridge.py` | Offline mapping of verified score/bundle artifacts; no legacy imports or financial arithmetic |
| `engine/v2/serving/projections.py` | Serving-owned schema statements, immutable release indexes, bounded queries |
| `engine/v2/serving/api.py` | FastAPI route/auth wiring over a read store; no scoring/provider imports |
| `engine/v2/serving/operations.py` | Preserve the existing compatibility surface and health behavior |
| `engine/v2/dashboard/preview.py` | Local launcher composing serving-layer interfaces only |
| `tools/v2_dashboard_project.py` | Coordinator CLI: validate inputs, build projection, prepare publication evidence |
| `ui/` | React/TypeScript client, API types, formatting, board/detail components |
| `checks/rearchitecture_phase3_gate.py` | Evidence validation using existing comparison/engineering controls |
| `tests/test_v2_serving_*.py`, `tests/test_v2_dashboard_*.py` | Synthetic, disk/HTTP, and browser checks |

Serving and ops are peers at layer 7; do not import either package from the
other. A coordinator under `tools/` may compose them. Pass verified refs,
catalog connections, and health paths explicitly. Serving declares plain
migration statement tuples; the coordinator applies them using the existing
migration machinery. No database opens or migrations happen at module import
or on a GET request.

Use a small indexed projection in the existing SQLite catalog. Large details
stay in immutable objects. API connections are read-only. Projections are
rebuildable; retain source releases and score artifacts. Do not parse an
entire board file for each paginated request.

Update exports, package README public interfaces/consumers, test inventory,
and coverage baselines as implementation lands. No new v2-to-legacy Python
imports should be necessary: Phase 2 emits the values being consumed. If an
artifact is missing, fix its handoff within Phase 2 ownership rather than
invoking `render_bundle` from an API.

## 5. Phase 2 handoff and compatibility bridge

### 5.1 Required immutable input set

The projection command accepts one private document, not arbitrary directories
to glob. Reuse actual Phase 2 handle types and persisted release manifests.

```text
PreviewInput (preview_input.v1.0):
  source_release_id, source_release_manifest_ref
  snapshot_ref, legacy_snapshot_object_ref
  score_batch_ref, score_job_input_refs, bundle_manifest_ref
  model_registry_artifact_refs, model_evidence_ref?
  finality_ref, expected_population_ref
  score_comparison_receipt_ref, render_comparison_receipt_ref
  source_code_hash, source_environment_hash
```

Refs must resolve to verified retained objects. Required refs cannot be null
or filled with fabricated IDs. Optional model evidence absence needs a reason
and must never be replaced with a current file from disk.

Require Phase 2 D14–D20 evidence appropriate to the release and the full
Phase 2 exit receipt before P3-4 completion. P3-0 can open an already validated
candidate while unrelated Phase 2 work finishes, provided it is marked
candidate and no phase-completion claim is made.

### 5.2 Separate engine evidence from display values

Use existing foundation hashing and strict decoding. Operational timestamps
live in envelopes, outside deterministic score/projection identity.

```text
PreviewRelease (preview_release.v1.0):
  release_id, source_release_id, projection_manifest_ref
  snapshot_ref, score_batch_ref, bundle_manifest_ref
  model_registry_artifact_refs, model_evidence_ref?
  comparison_receipt_refs, source_code_hash, projection_code_hash
  requested_as_of, resolved_as_of, clock_ids
  coverage_summary, stale_or_degraded_reasons
  score_format = "legacy_score_bridge.v1.0"
  producer = "legacy_via_v2"
  capabilities = {read: true, submit_jobs: false, collect_live: false}

LegacyScoreBridge (legacy_score_bridge.v1.0):
  score_id, event_ref, clock_id, legacy_row_id
  score_batch_ref, source_row_key, source_record_hash
  request_provenance_refs, snapshot_ref, model_registry_artifact_refs
  engine_record: full-precision saved row
  display_record: exact corresponding rendered row
  detail_refs, unavailable_detail_reasons
```

This bridge is **not** the completed Phase 4 `ScoreRecord`. Do not populate
missing strategy/deployment/recipe IDs with guesses to satisfy that future
schema. Advertise `score_format`; the client can later support the native
schema alongside this one.

`engine_record` retains saved precision, null/absent distinctions and ordered
geometry. `display_record` retains existing display values and units. Neither
the display digest nor rounded values become inference/replay inputs.
STR-RUNUP and absolute-move drivers retain distinct meanings; do not fill an
unavailable comparison ratio by dividing convenient fields.

### 5.3 Identity and mapping algorithm

1. Resolve the explicit source release once. Verify its complete manifest and
   required score, bundle, snapshot, and comparison refs before reading rows.
2. Load the planned event/strategy population from the score job. Do not derive
   expected population from whichever rows happened to render.
3. Map events through the Phase 2 event/calendar mapping. Preserve event ID and
   calendar revision together. Never invent a permanent ticker/date ID.
   Ambiguous or unresolved mappings refuse the candidate projection; they
   must not silently drop rows.
4. Join source to display rows through an explicit versioned legacy mapping.
   Scorer and renderer row IDs are not assumed identical. Use event revision,
   strategy, expiry, exact strike, and any additional discriminator needed by
   saved ladder/request data. Validate one-to-one matches; count main-board
   and ladder rows separately. DYN-SV stays its own result with its choice.
5. Hash the full source record and pinned score-batch/request dependencies to
   form an opaque score ID. Retain overrides and exact geometry; never use
   ticker/strategy alone. Use the existing canonical hash implementation.
6. Copy rendered fields through one checked mapping specification listing
   source JSON paths, units, nullable/absent behavior, and display formats.
   Every displayed financial field must have a source.
7. Emit all mismatches in one comparison receipt: missing/unplanned/duplicate
   rows, identity, null masks, values, model refs. A zero compared population
   cannot pass real launch.
8. Write immutable detail objects and insert the complete index transactionally
   under its release ID. Identical repeated input is a no-op; conflicting
   content under an existing ID fails. Incomplete releases cannot become current.

Do not invent stable identity from a ticker payload. If the Phase 2 mapping
is missing, report that handoff gap and keep the compatibility preview usable.

### 5.4 Indexing and publication

The minimal schema has release, event-summary, and score-summary tables.
Keys include release ID; event/score IDs are unique within a release. Index
event-date/ticker/event-ID ordering and event-to-score lookup. Store large
engine/detail payloads by immutable object ref. This index is not a prediction
or settlement ledger.

Build the candidate index and objects before requesting publication. Through
the coordinator, bind its manifest and parity/security evidence to the existing
fenced publisher. Adding files changes the publication binding: generate
receipts bound to that exact candidate, not copied green booleans or gates
bound to an earlier manifest. Retain the source release as a separate ref.

Keep one authoritative published pointer for this shadow dashboard. API current
resolution must name a fully committed projection of that release; do not
create an independently advancing UI-latest pointer. A crash after index
insertion but before publication leaves an unselected candidate. The previous
release remains readable.

**Implemented (P3-1c).** The binding is a plain document (`projection_
binding.v1.0`, `engine.v2.serving.projections.projection_binding`), built
purely from the committed serving index (no ops import): `projection_
release_id`, `source_release_id`, `projection_manifest_ref` plus its own
`projection_manifest_hash` (the manifest object's real content hash),
`bundle_manifest_ref` (P3-4's byte-bound render-bundle identity, restated
from the release), `serving_index_identity` (a fingerprint of the release's
own committed row — document plus findings — so a direct edit to the index,
not only a swapped manifest object, is caught), and `comparison_receipt_refs`
(the release's own parity evidence: score/render comparison plus the
findings receipt). Security evidence is not duplicated into it — the ops
release manifest already binds the security gate's receipt to the same
candidate via `binding_hash` (below).

The operator entry point is the smaller of this section's two options: a
`publication_effect` input, not a new coordinator stage.
`engine.v2.ops.effects_graph.publication_effect` accepts an optional named
`projection_binding.json` entry in the job's already-existing `input_
bindings`, exactly the mechanism `bundle.tar`/`finality.json`/`selfcheck.
json`/`engineering_gate.json` use — no new job kind or DAG wiring. Its
presence is the only branch: a bundle-only publication (no projection
candidate yet, e.g. still the P3-0 compatibility preview) behaves exactly
as before. Because `binding_hash = content_hash({release_id, occurrence,
files})` is computed over the exact `files` dict passed to `stage_release`,
adding this file changes it, so every gate `publication_effect` builds is
freshly bound to the candidate this publication now carries — never a
receipt computed for an earlier, projection-less binding, and gates minted
for one generation's files never validate a different generation's.
`tools/v2_dashboard_project.py` (the one place allowed to compose serving
and ops) is what actually BUILDS the document — it now emits `projection_
binding` alongside its existing `release`/`findings` output — so an
operator/submission script registers those bytes as an ops artifact and
binds them the same way it already binds the render bundle; the tool
itself stays offline and never publishes.

The read API's current resolution (`engine.v2.serving.api._publication_
resolver`) is the one pointer chain: the existing fenced ops publisher's
own `CURRENT` under its release root, that release's bound `projection_
binding.json`, reverified against the LIVE `serving.sqlite` index
(`projections.verify_projection_binding`) before being trusted. Files (plus
one bounded `serving.sqlite` connection) only — no `engine.v2.ops` import,
so the layering stays intact. A release with no bound projection at all
resolves to the ordinary "no current release" (503 `NO_CURRENT_RELEASE`,
retryable); a binding that names a release id which is not fully committed
or whose manifest/index hash no longer matches (a tampered binding
document or a changed index row) is a typed, non-retryable refusal (500
`CURRENT_BINDING_INVALID`) — never the same code, and never a silent
fallback to "latest". The temporary `serving_root/CURRENT` default this
module used before P3-1c is retired outright, not replaced by a new
fallback; `create_app`'s `resolver` seam is unchanged in shape (any zero-
argument `Callable[[], str | None]`, may also raise `ApiError`).

Crash safety: a crash after `build_candidate`'s index insertion but before
its operator entry point runs leaves that candidate committed-but-unbound
in `serving.sqlite` — the ops `CURRENT`/binding are untouched, so the API
keeps serving the previous release exactly as before, and a rerun (compute
the binding, bind it, publish) completes cleanly. Rollback goes through the
existing typed path: restage the prior generation's own content (here, its
bound `projection_binding.json`) under a fresh release id and publish it —
`publish_local`'s existing generation-aware watermarking (§5.5 item 1)
means a second-generation publish, and a rollback after it, never collide.
Both generations stay independently readable via `/api/v1/releases/{id}`.

Tests: `tests/test_v2_serving_publication_binding.py`.

### 5.5 Review follow-through for repeatable updates and live health

Owner: this Phase 3 launch, carried from the Sep-13 Phase 2 review of
`97e2a5c`; see [Phase 2 closeout §12.2](rearchitecture_phase2_data_access.md#122-sep-13-review-closeout-fixes-owned-by-phase-2).
These tasks do not delay opening an already validated P3-0 candidate, but
must finish before P3-4 claims repeatable updates or authoritative live health.

1. **Separate a retry from a new same-session plan.** **Done**, commit
   56d8709; see
   `tests/test_v2_ops_same_session_replan.py`, which reproduces the Sep-14
   operator trace (plan -> submit -> cancel all -> re-plan after a code/
   manifest change -> submit) and proves it now succeeds with fresh job ids
   while the cancelled rows are untouched.

   `nightly.py::_plan_identity` folds the legacy manifest, implementation
   (`plan["implementation_ref"]`), and pinned `decision_clock` into every
   stage's idempotency key (`_scope_hash`), so a genuinely new same-session
   plan (changed code/manifest, or simply a fresh `plan nightly` call, which
   always re-pins `decision_clock`) gets fresh job identities, while
   resubmitting the identical saved plan artifact reproduces the exact same
   keys. `cli.py`'s `--idempotency-key` semantics for nightly submission are
   now documented explicitly (`guides/rearchitecture_phase1_runbook.md`
   §2.1): accepted by the grammar, never read for job identity, which comes
   entirely from the plan. `effects_graph.py::_generation_ref` mirrors the
   same plan identity into the release id, so `publication_effect`'s
   `stage_release` call no longer raises `IDEMPOTENCY_CONFLICT` ("release
   manifest changed") for a genuinely new generation's bundle, and an old
   release's own row/manifest/files are never touched.

   **Done (2026-09-14, second pass — the item above's own "stop and
   report" finding, now resolved).** The gap left open by the first pass:
   even with a distinct release id, actually PUBLISHING a second
   same-session generation (moving `CURRENT`) or committing its decisions
   still conflicted. `outbox.watermark`'s one-receipt-per-occurrence rule —
   keyed on `(pipeline, scope, stage)` alone — fired inside
   `publish_local`'s own `_acknowledge` for the `publication`/`delivery`
   watermarks (and, separately, inside `engineering_gate_effect`/
   `ledger_export_effect`/`backup_effect`, which never carried a plan
   identity at all — the ACTUAL 2026-09-14 real-run failure,
   `run_sha256:4541ca27acdbeb04d`: `engineering_gate` FAILED with
   `IDEMPOTENCY_CONFLICT` because an earlier generation's receipt, keyed
   only by `(scope, session)`, was still there). Resolved with a
   generation-aware watermark: `watermarks` now keys on
   `(pipeline, scope, stage, generation)` (migration 8,
   `engine/v2/ops/schema.py`), `generation` defaulting to `""` for every
   caller that predates this (identical old behaviour). Engineering gate,
   ledger export and backup each pass their own `_generation_ref(claim)`
   (`nightly.py::_legacy_params` now pins `deployment`/`decision_clock` on
   all three, mirroring `decision_evidence`/`publication`); a genuine retry
   of the same generation stays idempotent, a same-generation content
   change still `IDEMPOTENCY_CONFLICT`s, and a new generation records its
   own receipt without touching an earlier one's (`backup`'s own enqueue
   key needed the same generation fold — see `effects_graph.backup_effect`
   — since `prepare_backup` is already content-idempotent but
   `run_backup`'s `claim()` finds nothing once an earlier generation's row
   already delivered).

   Publication ordering was also fixed: `publish_local` now checks
   `outbox.watermark_would_conflict` for both `publication` and `delivery`
   BEFORE the `CURRENT` pointer swap, so a foreseeable refusal never leaves
   `CURRENT` naming an unacknowledged release. A REAL crash between the
   swap and the acknowledgement (the `fault` hook) is a separate,
   unavoidable window; the existing O24 recoverable-on-retry design is
   unchanged (`tests/test_v2_ops_authority.py`). A newer generation may now
   become the published release; the prior release stays on disk and
   readable, and rollback is the same restage-under-a-fresh-id path proven
   in `tests/test_v2_ops_same_session_replan.py`.

   Decisions/predictions stay intentionally generation-INDEPENDENT: the
   first committed record for a scheduled occurrence is authoritative
   forever, and a later generation's differing content is recorded as a
   durable `DecisionDivergence` (component contracts §12.1) rather than
   failing the job or silently duplicating. See
   `engine.v2.ledger.decisions.record_divergence`,
   `engine.v2.ops.decision_commit.commit_decisions_in_transaction`, and
   `tests/test_v2_ops_same_session_replan.py`/
   `tests/test_v2_ops_generation_effects.py` for the full test coverage
   (two generations of engineering_gate/ledger_export/backup/publication;
   identical-retry idempotency; same-generation conflict; divergence
   recording; settlement reading generation 1's prediction; the
   health/streak hook counting one occurrence per night). True superseding
   decisions (an operator-approved `supersedes`/`supersede_reason`) remain
   ledger-phase scope, unchanged.
2. **Populate live engineering health from real observations.** **Done
   (2026-09-14).** `engineering_gate_effect` now calls `health.record_check`
   for `(session, "engineering")` on every attempt, in addition to (not
   instead of) its existing generation-aware watermark: `record_check`'s own
   `PRIMARY KEY(occurrence, kind)` (migration 9,
   `engine/v2/ops/schema.py`, adding a per-row `attempts` counter) collapses
   a retry or a later same-night generation onto that occurrence's one row,
   bumping `attempts` rather than adding a night. `health.trailing_occurrences`
   /`health.engineering_history` build the guide's per-night window
   (`{occurrence, status, retry_count, detail}`, `status` one of
   `pass`/`fail`/`unknown`) over the last N SCHEDULED TRADING sessions ending
   at the resolved session (2026-09-14 review fix: an earlier pass used
   calendar days, so every weekend/holiday read back as a fabricated
   `unknown` night, diluting the streak; `trailing_occurrences` now calls
   `engine.v2.ops.legacy_adapter.projected_trading_sessions` — pure weekday/
   US-market-holiday rule, no repo root needed — and REFUSES
   (`VALIDATION_FAILED`) rather than silently falling back to calendar days
   if the calendar cannot be resolved or does not yield enough sessions); a
   scheduled trading night with no recorded observation reads back as
   `unknown`, never a fabricated `pass`. `budget_streak` (unchanged) still
   feeds `/health.json`'s `code_budgets` from the same table, UNBOUNDED
   (every occurrence ever recorded); `OperationsStatus.engineering_streak`
   instead uses the new `health.engineering_streak_from_history`, derived
   ONLY from the same windowed `engineering_history` the document already
   carries, so the two fields of one document can never disagree — the
   review found `budget_streak`'s unbounded scan could report more
   consecutive failures than the visible window ever shows.
   `tests/test_v2_ops_engineering_history.py::test_engineering_streak_is_
   windowed_not_budget_streaks_unbounded_scan` asserts the two intentionally
   disagree on a fixture built to show it. See also
   `tests/test_v2_ops_engineering_history.py` (three nights: pass,
   retry-then-pass, unobserved; two generations in one night count as one
   night; trading-session window excludes weekends/holidays; a broken
   calendar refuses) and `tests/test_v2_ops_generation_effects.py`'s own
   health/streak hook test.
3. **Consume semantic status without reconstructing it.** **Done
   (2026-09-14).** `publication_effect` now writes a versioned
   `operations_status.json` sidecar
   (`engine.v2.contracts.OperationsStatus`/`OPERATIONS_STATUS_V1`,
   `engine/v2/contracts/operations.py`) into the fenced publisher's own scope
   root (a sibling of `CURRENT`, not inside any one release's immutable file
   set) on every attempt, success or failure — `_stage_and_publish` writes it
   both on success and from the `OpsError` except branch, so a FAILED update
   (a new generation that fails before publication) still leaves the prior
   release current and readable while the sidecar records this attempt's own
   `failed_update`/`failed_update_reason`. It carries, never recomputes: the
   pinned `requested_session`/`resolved_session`
   (`snapshot_stages.resolve_effective_session`'s own output), `conflicts`/
   `degraded_model_evidence` read back out of the already-rendered bundle's
   own `data/flags.json` (P2-C08), the bound `selfcheck.json` verbatim (or an
   explicit unknown state if none is bound), the engineering history window
   from item 2, and `stale`/`withheld` banners derived from the served
   release versus this attempt's candidate. `engine/v2/serving/api.py`'s
   `/api/v1/operations` reads this file directly off `publication_root`
   (files only, no `engine.v2.ops` import — the existing no-ops-import guard
   still holds) and returns a typed `OPERATIONS_UNAVAILABLE` Problem when it
   is missing or malformed; missing engineering history renders as
   `unknown`, never green. `/health.json` (`engine/v2/serving/operations.py`)
   is untouched and keeps its existing `operations_health.v1.0` fields. See
   `tests/test_v2_ops_engineering_history.py` (conflicts/degraded evidence/
   selfcheck carried; requested vs. resolved session; failed update keeps the
   old release and shows the reason; an unobserved window renders as
   `unknown` throughout) and `tests/test_v2_serving_api.py` (the route
   itself, including the no-history-is-not-green case). Response fields are
   documented for a future health/flags screen in `ui/README.md`; the
   dedicated screen itself remains Phase 6 work, unchanged.

   **2026-09-14 review fix.** The first pass only wrote the sidecar from
   inside `publication_effect`, so the REAL observed failure shape — an
   upstream dependency (`engineering_gate`, `legacy_finality`) fails or the
   run is cancelled, `lifecycle.block_descendants` blocks the dependent
   `publication` job BEFORE it is ever claimed — left `publication_effect`,
   and therefore the sidecar, never running at all; the old release kept
   being reported `failed_update=False`, exactly the silent-green outcome
   this item forbids. Fixed with a SECOND writer,
   `effects_graph.write_publication_terminal_status`, sharing the same
   `OperationsStatus`/`_write_status_document` construction (never
   duplicated) but built from the job row alone — `attempted_release_id`
   null (no attempt ever formed one), `failed_update_reason` naming the
   first failed/cancelled job's kind and code (`block_descendants`'s
   recursive CTE stamps every transitive descendant's `DEPENDENCY_FAILED`
   `Problem` with the SAME root job id, so one lookup names it). Called from
   `effects_graph.reconcile_publication_status`, hooked into
   `supervisor.Service.tick()` (`Service._reconcile_publication_status`,
   guarded against crashing the tick) rather than threaded through
   `lifecycle.py`'s kind-agnostic transition functions; idempotent and
   self-healing via a strict `generated_at > updated_at` comparison against
   the existing sidecar, so it can run every tick without ever clobbering a
   LATER successful publish's own fresher document. See
   `tests/test_v2_ops_engineering_history.py`
   (`test_upstream_engineering_gate_failure_blocks_publication_and_status_
   shows_it`, `test_cancelled_upstream_job_blocks_publication_and_status_
   shows_it`, `test_reconcile_does_not_clobber_a_later_successful_publish`),
   all driven through a real `supervisor.Service.tick()` with the
   failing/cancelled attempt committed directly (no live subprocess).

   Also: `/api/v1/operations` now takes an optional `?release_id=` query
   param — given and it disagrees with the sidecar's own `release_id`, a
   typed 409 `OPERATIONS_STATUS_NOT_FOR_RELEASE` (`details.
   requested_release_id`/`details.status_release_id`) replaces the 200,
   since the sidecar always describes the SCOPE's current state and a
   client that pinned an earlier release must never silently be handed a
   different one's status; omitted, behavior is unchanged. See
   `tests/test_v2_serving_api.py`'s `test_operations_route_release_id_
   param_*` (match/mismatch/absent). Documented in `ui/README.md`; `ui/src`
   untouched.

   **2026-09-14 SECOND review fix (ordering defect).** Reviewed against a
   real ops root (`/root/phase2-shadow-ops`, several historical blocked
   publications, several attempts, no sidecar yet): the first pass's
   `reconcile_publication_status` scanned every blocked/failed/cancelled
   `publication` row UNORDERED and wrote whichever one SQLite returned
   first, relying only on the `generated_at > updated_at` skip -- once that
   ONE row got a fresh sidecar, its `generated_at` outranked every OTHER
   (older) row's `updated_at`, so the actual LATEST failure was skipped
   forever; it also never considered a NEWER already-succeeded publication
   in the same scope unless that publish happened to write its own
   sidecar. Fixed by looking only at each scope's SINGLE latest
   `publication` job (any state) at all, in one grouped SQL query
   (`_LATEST_PUBLICATION_PER_SCOPE_SQL`: a `json_extract`-computed `scope`
   column — the same `effect_scope`-or-`output_namespace` fallback
   `effect_scope(claim)` uses — partitioned with `ROW_NUMBER() OVER
   (PARTITION BY scope ORDER BY updated_at DESC, job_id DESC)`, keeping only
   rank 1) rather than a scan plus per-row reads; a scope's latest job that
   already succeeded, or is still queued/running, gets no sidecar write at
   all. See `tests/test_v2_ops_engineering_history.py`'s
   `test_reconcile_names_the_newest_of_several_blocked_jobs_in_one_scope`
   (three blocked jobs, insertion order deliberately different from time
   order), `test_reconcile_skips_an_older_blocked_job_behind_a_newer_
   success` (both insertion orders), and
   `test_reconcile_handles_two_scopes_independently`. Minor, same pass: the
   `Service.tick()` failure print now fires only when the reconciliation
   problem's (code, message) changes from the last one actually printed
   (`Service._last_publication_status_problem`), not on every ~1s tick —
   `test_publication_status_reconcile_failure_prints_once_while_
   persisting`.

Extend the existing update/rollback and operations-health acceptance cases
with these controls. Consume saved synthetic/real artifacts sequentially;
do not start another full nightly solely to test UI plumbing.

## 6. Read API contract

Follow component contracts §13.2. Implement only these routes now:

| Route | Required behavior |
|---|---|
| `GET /api/v1/releases/current` | Resolve the published shadow release once; return `PreviewRelease` with immutable ID/readiness |
| `GET /api/v1/events` | Require release ID; bounded date/ticker filters and optional strategy/verdict; return an event page |
| `GET /api/v1/events/{id}/scores` | Require release ID and validate any clock filter; return main-board strategy summaries including refusals |
| `GET /api/v1/scores/{id}` | Return `LegacyScoreBridge` and available details; validate membership if a release ID is supplied |
| `GET /api/v1/operations` | Current versioned health sidecar, separately dated from frozen data |

No payoff computation, portfolio aggregation, job POSTs, training, or model
routes are required now. Saved payoff points can be lazy score detail. An
unavailable view returns its reason.

Event pages carry `schema_version`, `release_id`, `query_hash`, `items`,
`next_cursor`, and `total_matching`. An item includes event/calendar identity,
ticker, date/session, clocks, readiness, and the precomputed main-board score
summaries needed by the table. This avoids one request per event. Omit engine
records, full ladders, history, analogs, and model evidence from the first page.

Rules:

- Default 50 events, cap 200. Validate date/query bounds, filters, and sorts.
  Default order is event date, ticker, event ID. Every sort ends in a unique
  tie-breaker. Pages must have no missing/duplicate events.
- Strategy/verdict filters select matching events and their matching visible
  summaries consistently. Counts cover the complete filtered population, not
  the visible page. Document/test this policy.
- Integrity-protect opaque cursors with a server-held key; bind release,
  normalized filters/sort, and final key. A mismatched cursor returns
  `CURSOR_MISMATCH` (409), not an empty page or restart.
- No current release returns 503; unknown immutable ID 404; invalid input 422.
  Use `Problem` envelopes. Strategy refusals remain successful data responses.
- Authenticate data, details, compatibility files, and health. Reuse existing
  bearer/cookie behavior with same-origin access. Credentials never enter
  URLs, manifests, frontend source, or logs. Default launch is loopback.
- Immutable responses have ETags and private cache scope; current/health
  revalidate. Reject traversal/symlinks. Expose allowlisted serving objects,
  never arbitrary files from the artifact store.
- Neither GET nor API startup constructs/imports a scorer, loads model/feature
  arrays, calls providers, runs subprocesses, fits, or writes projections.

Compatibility links must name `/release/<source_release_id>/...` explicitly.
The existing shell uses `/release/current/index.html`, whose later requests
can follow a changing pointer. In P3-0 resolve current once and navigate the
frame to the immutable URL before loading its data. Preserve that ID on route
changes. Test a pointer switch between HTML and dependent data fetches.

## 7. Smallest useful React client

Use React + TypeScript, Vite, npm and its lockfile, with pinned installed
versions. Follow an existing frontend if one has landed by implementation
time. Two views do not need a router/state/chart/UI-kit framework; ordinary
components, fetch and a small SVG chart suffice.

Implement a typed `DataClient`: `getRelease`, `listEvents`, `getEventScores`,
`getScore`, `getOperations`. Components do not read raw bundles directly.
Cache by release plus event/score and normalized query. Abort/discard replies
whose release/query no longer matches the selected state.

Board requirements:

- Persistent shadow/producer label, resolved as-of date, quote/input freshness,
  coverage, and current failure/withheld banner.
- Event date/session, ticker, strategy, verdict/refusal, driver forecast,
  market implied move, entry premium, available expected-return fields, and
  DYN-SV choice. Use saved display fields and existing meanings.
- Date/ticker/strategy/verdict filters, sort, paging, count, and a link to the
  same frozen legacy view.
- Loading, no matches, no accepted release, unavailable score, unauthenticated,
  and detail-failure states. Null means unavailable, never zero.

Detail requirements:

- Strategy/event/clock, selected legs/contracts/geometry, entry/exit dates,
  quote freshness, fill assumption, model IDs, forecast bands, gate
  terms/threshold/verdict, and warnings wherever saved.
- Saved payoff points/convention, if present. Preserve terminal versus modeled
  exit-value labels. No new JavaScript valuation.
- Provenance IDs and producer. Missing provenance has an explicit unavailable
  state; do not invent source timestamps.
- Load only after selection; a detail failure leaves the board usable. A new
  strike explorer or interactive rescoring is deferred.

Preserve every strategy/refusal in the accepted population. Do not show only
passing/non-null rows. Presentation filters never become a strategy universe;
small browser fixtures must not narrow scorer analog context.

The UI may format units/dates/percentages and map stored chart values to pixels.
It cannot compute fair premium, ratios, gates, expected returns, strategy
selection or book totals. Those values are supplied by the accepted producer.

Add exact source/config/lockfile allowlist rules for `ui/` before existing hard
blocks. Do not allow JSON or TypeScript globally. Exclude `node_modules`,
`dist`, caches, screenshots, copied releases and licensed fixtures. Compiled
client assets contain code only; data arrives via authenticated requests.

## 8. Build order and task boundaries

Each task has one deliverable and named tests. If delegated under repository
instructions, give one task per agent, paste relevant signatures/facts, name
owned/prohibited files, and use isolated worktrees. Do not delegate this entire
guide as one task. Heavy/private scoring requires standing authorization and
resource checks; this guide primarily consumes saved artifacts.

### P3-0 — Open the compatibility preview first

1. Obtain the Phase 2 handoff and record accepted versus validated-candidate
   status. A missing handoff prevents a real preview claim.
2. Add the launcher and immutable frame pinning, preserving auth and health.
3. Run L01/L02, open one real release, and retain a private browser receipt and
   screenshot with release ID/as-of. Give the user its launch command/URL.
4. Continue with the new board after the compatibility view is usable.
   Synthetic work may proceed if Phase 2 is still completing the handoff.

Exit: `/usr/bin/python3 -m pytest -q tests/test_v2_dashboard_preview.py`
plus the real-release browser receipt. Do not rerun a whole nightly merely
for a screenshot if a validated release exists.

### P3-1 — Pin and index the bridge

1. Add §5 contracts, mapping and strict identity/population validation.
2. Add minimal serving schema, immutable detail objects and offline projection
   coordinator. Begin with synthetic score/bundle pairs.
3. Prove idempotency, candidate failure, old-release reads, and complete
   comparison receipts, then do one saved real-release disk round trip.
4. Record the actual accepted Phase 2 interfaces and unresolved mapping gaps.

Exit: `/usr/bin/python3 -m pytest -q tests/test_v2_serving_bridge.py`

### P3-2 — Read API

1. Add §6 routes, auth, bounded queries, cursors, ETags, and detail access.
2. Prove no startup/request path initiates scoring or provider work.
3. Use real SQLite/files and HTTP in tests, including release switches during
   open pagination/detail sequences.

Exit: `/usr/bin/python3 -m pytest -q tests/test_v2_serving_api.py`

### P3-3 — Board and detail

1. Scaffold client/build/lockfile and narrow public allowlist. Build the board
   first against P3-2, then lazy detail.
2. Carry release identity through requests, cache keys and comparison links.
3. Cover synthetic empty/error/null/zero/refusal/paging cases; open real scores
   as soon as the board renders.
4. Record shipped fields/views. Unshipped screens link to the compatibility
   surface and remain Phase 6 work.

Exit commands: `npm --prefix ui run typecheck`, `npm --prefix ui run build`,
and `/usr/bin/python3 -m pytest -q tests/test_v2_dashboard_browser.py`.
Define these npm scripts when scaffolding; these are required new commands,
not claims about existing tools. Use existing Python Playwright for browser
tests rather than adding a second test framework solely for this slice.

### P3-4 — Real parity, refresh handoff, and launch

1. Compare a nonempty real release: saved scores → bridge; matching rendered
   values → API → browser. Data comparison covers the entire shipped
   population, not just a screenshot.
2. Project a second accepted saved release or a second validated shadow
   generation from cached inputs. Never falsify its as-of date. Prove
   publication and rollback with both generations retained. Complete §5.5:
   distinguish identical-plan retries from new same-session plans and verify
   real engineering history, including unknown nights and retry counting.
3. Run Phase 3 tests and fresh engineering/coverage checks; validate Phase 2
   evidence against its producer commit. Generate the acceptance report and
   tested operator runbook.
4. Provide the working private command/URL, as-of, limitations, evidence refs,
   and deferred owners. No external deployment or production switch is implied.

Stop expanding the launch scope when this gate passes. Continue later work
under §3.2 rather than delaying the handoff.

## 9. Acceptance tests and negative controls

| ID | Tier | Required evidence |
|---|---:|---|
| L01 | 1 | Real compatibility preview opens; auth protects data/health. Invalid/missing token fails. |
| L02 | 1 | Switch `CURRENT` between HTML, data, navigation, paging and detail. R1 readers retain R1; a new session resolves R2. |
| L03 | 0 | Strict contract round trips; malformed refs/unsupported schemas fail. Operational timings do not change identity; geometry/model/snapshot changes do. |
| L04 | 0/1 | Mapping covers planned rows and separate ladder counts. Unmatched event, duplicate key and omitted refusal produce independent findings in one receipt. Empty comparison cannot pass. |
| L05 | 0/1 | Full-precision engine and display values match their respective sources. Rounded geometry, null-to-zero, wrong units and altered DYN-SV choice each fail locally. |
| L06 | 1 | Real disk/SQLite retry is idempotent; corrupt bytes/interrupted candidate cannot publish; the old release stays readable. Identical saved-plan retries reuse identity; a permitted new same-session input/plan receives distinct job and release identities (§5.5). |
| L07 | 1 | Multi-page filters yield exact full-filter counts and no duplicates/omissions; wrong release/filter cursor fails; tied date/ticker rows remain stable. |
| L08 | 1 | Unauthorized detail/health/legacy file, traversal and symlink requests fail. Logs/errors expose no secret or licensed payload. |
| L09 | 0/1 | Startup/GETs pass with scoring/provider constructors rigged to fail if invoked; read-only DB connections write no jobs/projections. |
| L10 | 1 | Browser initial load fetches current, health and one bounded event page; detail/evidence/ladder are not eagerly fetched. Old replies cannot overwrite new selection. |
| L11 | 1 | Loading/empty/stale/withheld/null/zero/refusal/auth/detail-error states work. Stale readiness does not erase the saved gate verdict. Real engineering observations count nights rather than retries; missing nights/history stay unknown, with prior selfcheck and nonempty conflict/degradation flags covered (§5.5). |
| L12 | 1/2 | Real saved-release comparison agrees on all shipped rows/fields. Browser samples STR-THRU, STR-RUNUP, DYN-SV and available refusals; frozen fixtures cover absent cases. |
| L13 | 1 | R1 → R2 → R1 uses existing fenced publication, retains histories, and leaves official legacy ledgers/registry/pointers unchanged. Failed R2 leaves usable R1 plus health failure. |
| L14 | 0/2 | Layers, budgets, READMEs, adapter ratchet, lint, hygiene, hook and coverage pass. UI typecheck/build pass; assets contain no market data/secrets. |

Suggested ownership: preview tests L01/L02; bridge L03–L06; API L02/L07–L09;
browser L10–L12; publication integration L06/L13. Reuse Phase 1/2 controls.
Every new comparator needs a negative control; do not mock every disk boundary.

Compare IDs, contracts, quantities, flags, null masks, verdicts, choices and
copied display payloads exactly. Independently recomputed floats use existing
declared field tolerances. UI formatting follows the documented unit/precision
mapping. Never widen tolerances or silently fix legacy economics for parity.

Measure API latency, first usable page, bytes, population, memory, cache state
and contention on a saved release. The design targets sub-second reads and a
usable page within two seconds on an agreed device; record actual results.
Fix eager whole-board downloads or visible stalls in this slice. Defer
speculative performance infrastructure.

## 10. Evidence, runbook, and completion

Implement a strict private gate input:

```text
Phase3Evidence (phase3_evidence.v1.0):
  implementation_code_hash, environment_hash, frontend_lock_hash
  phase2_acceptance_ref, source_code_hash, source_environment_hash
  preview_input_refs, accepted_release_refs
  population_manifest_ref, mapping_version
  comparison_receipt_refs, negative_control_receipt_refs
  browser_receipt_ref, refresh_rollback_receipt_ref
  engineering_receipt_ref, coverage_receipt_ref, performance_receipt_ref
  view_field_inventory_ref, deferred_work_ref
  authority_mode = "shadow"
```

Verify every referenced artifact, status, count and code hash; summary booleans
are insufficient. Phase 2/source evidence stays bound to its original producer
commit; Phase 3 tests/projection evidence binds to the final implementation.
These commits need not be identical. Do not rerun historical scoring merely
because frontend code changed; invalidate checks whose inputs/implementation
actually changed. Coverage uses the established per-package ratchet and a
committed fixed suite, not a newly invented percentage target.

Required new gate command after evidence producers finish:

```bash
/usr/bin/python3 -u checks/rearchitecture_phase3_gate.py --evidence-manifest /tmp/phase3-evidence.json --write-report
```

The gate generates a private `REPORT.md`: provenance, L01–L14 matrix,
population funnel, parity findings, measured timings, shipped/missing views,
and deferred owners. This is engineering acceptance, not a strategy experiment.
Do not insert invented metrics in `LEDGER.csv` or run `evaluate()` on UI test
rows. Any separate strategy experiment still follows all standing experiment
reporting and private-mirror rules.

During implementation add `guides/rearchitecture_phase3_runbook.md` with exact,
tested commands for:

1. Selecting accepted Phase 2 inputs and inspecting release/snapshot/as-of,
   populations, refusals and evidence.
2. Running `tools/v2_dashboard_project.py --input <private-input.json>` to
   prepare a candidate and receipt. This command must not fetch, rescore, or
   publish automatically.
3. Publishing through the existing supervisor/fenced workflow in a shadow
   target. Name actual completed Phase 2 interfaces, not guessed flags.
4. Starting `/usr/bin/python3 -m engine.v2.dashboard.preview` with explicit
   loopback host/port, shadow release root, catalog/health paths and a credential
   read from the environment. Document browser authentication without printing
   the value. Keep npm development access local too.
5. Projecting the next completed legacy-derived Phase 2 release and publishing
   after validation. Whole-batch projection/rescoring is acceptable temporarily.
   Never mix fresh quotes with stale forecasts by hand or claim `--no-refresh`
   rebuilt tiers. Retain original finality and knowledge-mode labels.
6. Returning to the prior release, stopping the preview process, and locating
   private reports/logs. Do not delete original artifacts or decisions.

Identify existing versus newly added commands. Verify the runbook against the
implementation before claiming completion. Use approved interpreter/resource
policies, supervised jobs for expensive work, and progress logging throughout.

Launch is complete when P3-0 and P3-4 are evidenced, the user can open the real
board/detail, L01–L14 pass, update/rollback works, and deferred work has owners.
Commit source/docs at green milestones; follow the standing instruction to
ask before pushing. Retain private evidence through the established backup.

Full rearchitecture parity remains pending: the original incremental-data
gate (no-op rewrites zero data; append/correction equals clean rebuild), native
scoring/model gates, remaining screen/offline parity and cutover are still
required at their later acceptance points. This launch marks none of them green.
