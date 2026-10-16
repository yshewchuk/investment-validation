# Nightly serving projection job (#717)

Design, not yet implemented. Decisions (owner, 2026-10-14): the projection is a new job after `publication`; a failure before the pointer swap fails that job loudly, leaves CURRENT on the previous release, and never fails publication or points CURRENT at a partial projection. Every failure after the swap (acknowledgement or step-4 read-back, below) leaves the new, pre-verified pointer visible; those are the stated exceptions. The read side is the last section of this guide; `engine/v2/serving/ARCHITECTURE.md` and `engine/v2/ops/ARCHITECTURE.md` change when each slice implements its part. `engine/v2/ops/ARCHITECTURE.md` is over its line ceiling (1,472 of 1,000 lines), so the job contract lives here until that doc is compressed.

## Job contract

A **new job kind `serving_projection`** (not the existing render stage `projection`) runs after `publication` in `_DAG_STAGES`/`plans.py`. It is a leaf: `delivery`, `backup` and the session's publication result do not wait on it, and it is not `OPTIONAL`, so its failure is loud. Publication's contract is unchanged; it never reads the serving index.


**Inputs.** Bound from the succeeded `publication` job's retained bindings and the same session's `score` job: `bundle.tar`, `finality.json`, the score artifact, the pinned snapshot id, the saved nightly plan, the source-provenance document and the native-score inputs the plan's `shadow_serving_scorer` needs. Every input is an artifact reference verified by `ArtifactStore.read_verified`; none is a local path or an unpinned latest.

**Execution.** The worker launches `tools/v2_dashboard_project.py` as a child process. `tools/` is the one place allowed to compose ops and serving (neither package imports the other). The tool's coordinator becomes a callable `run_projection(...)`; the CLI stays a thin wrapper over it. It unpacks the bundle into `/tmp` scratch (removed on success; kept as diagnostics on failure), then calls `build_candidate`. No provider calls, no scoring beyond the shadow row source the plan already names.

**Where it lives.** One configured **serving root**, `<ops_root>/serving` (overridable by ops configuration, never a checkout path), holds `serving.sqlite`, the content-addressed `objects/` store and `publisher/`, the serving pointer root (`CURRENT`, `releases/<ops_release_id>/projection_binding.json`). The API's `publication_root` and `serving_db` are that root's `publisher/` and `serving.sqlite`. The ops publication scope root (`<ops_root>/releases/<scope>`) keeps serving the bundle-only release that health and `delivery` read; the board no longer reads it.

**Binding and swap (order matters).**
1. `build_candidate` commits the projection release (objects first, then one index transaction).
2. The job builds `projection_binding.json` from the committed row, runs `verify_projection_binding` on it against the live index, and registers it as an artifact.
3. Proposed (slice 4 confirms `submit_retained_publication` can target the serving pointer root; today `publication_effect` builds its target from the ops releases scope, so a target parameter may be needed): it submits a retained publication (`publication_submit.submit_retained_publication`: the succeeded publication's retained bindings plus `projection_binding.json`, `operation_id` derived from the source publication job id) targeting the serving pointer root. That fenced `publish_local` materializes the release directory, then moves `CURRENT` with one `os.replace`.
4. The job re-reads `CURRENT` and the binding through the API's own resolver and `verify_projection_binding`; only then does it succeed.

A reader sees the previous release or the new one, never a mix: the new release directory is complete before `CURRENT` moves, the pointer swap is atomic, and index rows are append-only (an old binding keeps verifying while the swap happens). Each API request resolves once, and `/releases/current` carries a new ETag only when the response document changes: the existing validator hashes the projection identity and response, not the pointer, so republishing the same projection keeps the ETag. No new validator is designed here.

**Idempotency and retry.** `release_id` derives from content and index writes are `INSERT OR IGNORE`; the retained-publication `operation_id` is fixed per publication job, so a retry resubmits the identical job and `publish_local` accepts `observed == release_id`. A retry after a crash between steps 1 and 3 reuses the committed release; after step 3 it first completes or confirms the retained publication's durable acknowledgement (release row, outbox, watermarks; reading a valid pointer does not replace it) and only then runs the step-4 read-back. Retryable causes (store/index busy, `LEASE_LOST`, `RESOURCE_UNAVAILABLE`) go through the supervisor's normal retry; typed refusals are terminal until an operator reruns the job.

| Failure | Job outcome | `CURRENT` and board |
|---|---|---|
| A bound input missing or fails verification (`INPUT_CHANGED`/`INTEGRITY_FAILED`) | Fails, no index write | Previous release unchanged |
| Findings not ok (`PROJECTION_REFUSED`) | Fails; the findings artifact is kept as a diagnostic | Unchanged |
| Crash or error after the index commit, before step 3 | Fails or is retried; an unreferenced release row is inert | Unchanged |
| Retained publication refused (`STALE_EXPECTATION`, `PUBLICATION_REFUSED`, `IDEMPOTENCY_CONFLICT`) | Fails with the typed code | Unchanged |
| Crash between pointer swap and acknowledgement | Retry re-acknowledges (the existing O24 window) | New release, already complete |
| Step 4 re-verification fails (needs index damage after step 2's pre-swap check passed) | Fails loudly with the read-back's own code: `CURRENT_BINDING_INVALID` for a binding mismatch, `SERVING_INDEX_UNREADABLE` (retryable) for an unreadable index; a missing manifest object registration surfaces as `ServingIndexError(INTEGRITY_FAILED)`, which the resolver does not translate today (slice 4 assigns it a typed `SERVING_INDEX_UNREADABLE`-class refusal with a test) | Exception to "previous release": the new pointer is already visible. A binding mismatch makes the board refuse until the operator re-points with a rollback retained publication; an unreadable index is not repaired by a rollback and clears only when the index is readable again (retry). |
| Native plan present but native inputs missing (`INVALID_REQUEST`, raised before any index write) | Fails with that code; no index write | Previous release unchanged |
| Serving root unconfigured or unwritable | Fails `RESOURCE_UNAVAILABLE`; nothing is created | Unchanged |

A failure never fails `publication`, never marks `delivery` failed, and never points `CURRENT` at a partial projection: steps 1-2 are the only place a half-built projection can exist, and nothing reads it until step 3. "Previous release stays" holds for every failure before the swap; every post-swap failure (the acknowledgement-crash row and the step-4 row) leaves the new pointer visible. Acknowledgement recovery is a retry that completes the durable acknowledgement; a read-back failure is a verification fault and does not change the pointer. Completion role: slice 5 introduces a post-publication role for `serving_projection` that keeps existing required-job semantics (a failed job is `failed` for that job) but records the session as `completed_degraded` rather than failing the already-published session.

**Status.** The existing `operations_status.json` is written during publication and read from the ops scope root, so it cannot show a projection failure that happens before retained publication starts. Slice 5 adds a `serving_status.json` sidecar written by the `serving_projection` job on every attempt (success or typed failure) into the serving root (`<serving_root>/serving_status.json`, atomic replace): the session, the attempted ops release id, the served projection release id and the failure code. The API reads it from the configured serving root and exposes it with the current-release discovery, so clients show "tonight's projection failed, serving the previous release" instead of inferring it. Current-release discovery also takes the current attempt's id (the catalog attempt id of the latest `serving_projection` attempt, which changes on every retry, written into the sidecar too and required to match), session and attempted ops release id from an independent source (the catalog's latest `serving_projection` job, read by the ops-side status reader and handed to the API as configuration, not inferred from the sidecar). A missing, malformed, unreadable or identity-mismatched sidecar reads as unknown. Attempted and served ids need not match: a pre-swap failure legitimately has different ones.

**`/api/v1/releases/current` from a nightly.** The serving root's `CURRENT` -> that release's `projection_binding.json` -> projection release id, re-verified against the live index, exactly as the serving doc states; a night whose job failed before the pointer swap still resolves to the previous night's release (the step-4 row is the exception). Slice 1 also translates open-time index failures to `SERVING_INDEX_UNREADABLE`.

**Slices** (each at most about 400 code lines): (1) serving root configuration and API wiring (the API reads `operations_status.json` from `publication_root`, which becomes the serving `publisher/`; slice 1 states and tests which status document that is, since `serving_projection`'s own status is the slice 5 sidecar); (2) `run_projection` extracted from the tool, CLI unchanged; (3) `serving_projection` job kind, DAG/plan/profile wiring and the child-process worker; (4) serving-pointer retained publication, read-back verification and the failure table above; (5) completion disposition and the `serving_status.json` sidecar plus API exposure; (6) a real-browser test of the board against a real nightly's release.

Regression cases: slice 1 asserts `/releases/current` ETag changes when the response changes and stays constant only when the complete response is unchanged when the same projection is republished. Slice 5 also adds a same-job retry case: an earlier attempt's failure sidecar, while a new attempt runs or crashes before replacing it, reads unknown. Slice 5 adds the case that a changed status field (attempted id or failure code) with an unchanged projection changes the ETag.

Estimates (code / test / doc lines): 1: 120 / 100 / 15; 2: 150 / 120 / 10; 3: 250 / 200 / 20; 4: 300 / 250 / 25; 5: 200 / 150 / 20; 6: 40 / 150 / 10. Slice 4 must first confirm `submit_retained_publication` can target the separate serving pointer root; if it needs a parameter, that goes in slice 4.


## What #726 left open (#717 follow-up decisions)

The sections above settle ownership (new leaf job after `publication`), serving root, swap order, idempotency, failure table and slices. Three things were not stated; they are decided here and bind the slices.

**Inputs: one projection, native and legacy together.** There is no separate native or legacy projection path. The job projects the session's rendered legacy-format bundle (`bundle.tar`, unpacked, then read by `load_legacy_bundle`) and attaches the native rows the saved plan's `shadow_serving_scorer` names (`--nightly-plan`, `--native-inputs`), as `tools/v2_dashboard_project.py` does offline today. With no plan, the bundle is projected alone; a native plan whose native inputs are missing is refused, never a silent bundle-only release (the tool raises `INVALID_REQUEST` today, `tools/v2_dashboard_project.py:176`; slice 2's `run_projection` surfaces it unchanged and the job fails with that code, since `PROJECTION_REFUSED` is reserved for `build_candidate` findings not ok); slice 2 requires the release to record which row sources it carries (test: bundle-only and native-plus-bundle releases differ in that field), so the board never shows an unlabelled mix. Nothing is read from a checkout or a "latest" pointer.

**Scope and the P7-3 authority fence (`guides/authority_fence_switch_rollback.md`).** The swap is a retained publication carrying the session's effect scope, so the per-scope fence rules for `publish_local` apply (official scope fenced; `shadow` and drill scopes exempt, `guides/authority_fence_switch_rollback.md` slice 1a); the job adds no authority logic of its own. S2 (drain legacy and pause the shadow nightly), S5 (fence legacy, grant the catalog), S6 (start the official schedule), S7 (switch read consumers) and B1-B5 (backout) are that guide's cutover steps (its switch protocol and backout sections), not this guide's slices. The fence guide's single-writer invariant covers official releases; a `shadow`-scope release is non-official (slice 1 requires the board response to carry the release's scope), and a drill uses a scratch catalog and serving root, so it never moves the production serving `CURRENT`.

| Phase | Nightly scope | Serving swap | Board |
|---|---|---|---|
| Before S2 (shadow nightly runs) | `shadow` (`plans.nightly_plan` refuses other modes) | Planned (once slices 3-4 land): exempt from the fence; each nightly moves the serving `CURRENT` | The shadow nightly's release |
| S2 to S6 (the S2 drain pauses the shadow nightly and drills) | none runs | No nightly requests a swap | Last release swapped before S2 |
| Mid-S5 (legacy fenced, catalog not yet granted: zero writers) | none runs | Official-scope swaps refused `AUTHORITY_FENCED` | Last validated release |
| S5 granted, S6 not started (owner `v2`) | none runs | No nightly requests one; an operator official-scope retained publication is allowed at the current generation | Last validated release until an operator swap |
| After S6 (official, owner `v2`, current generation) | `official` | Allowed; a stale generation is refused `AUTHORITY_GENERATION_STALE` | Each night's release |
| After B1 to B5 (rollback) | none from v2 (on the pre-S5 abandonment path, `shadow` once B5 unpauses the nightly) | Official-scope swaps refused `AUTHORITY_FENCED` (owner `none` at B1, `legacy` from B4). This guide adds no rollback exemption; how B5 "switches consumers back" is the fence guide's to define, and this guide only requires that it use slice 4's serving-root target. On the pre-S5 abandonment path B5 also unpauses the shadow nightly, which then resumes shadow swaps | Last validated release, or the one B5 re-points |

A fence refusal (`AUTHORITY_FENCED`, `AUTHORITY_GENERATION_STALE`) is a typed retained-publication refusal, the same pre-swap row as `STALE_EXPECTATION`/`PUBLICATION_REFUSED` in the failure table (slice 4 adds both codes to that row): the job fails with the typed code, `CURRENT` and the board are unchanged, `publication` is untouched. S7 ("switch read consumers") is therefore no new mechanism: it is the first `serving_projection` swap after S6, or an operator retained publication of an already-indexed release, both through slice 4's serving-root target. S7 depends on slice 4 only, and `ops cutover consumers` reads the same serving `CURRENT`, binding and ETag the board does.

**First run, backfill and retention.** A fresh serving root has no `CURRENT`; the board reads `NO_CURRENT_RELEASE` until the first successful `serving_projection`. There is no automatic backfill of earlier nights: an operator reruns the job against a retained session's publication (idempotent, per the retry rules); anything older is rebuilt from providers, not migrated. Projection release rows and objects are append-only and never pruned by this job, because rollback needs old releases to stay verifiable. A retention policy is a separate design, to be filed as a follow-up issue by the supervisor, and on which no slice depends.

**Decisions for the user (defaults chosen; say if you disagree).** (1) Pre-cutover swap scope: default is the `shadow` scope so the board follows the shadow nightly until the S2 drain (the nightly is paused from S2 to S6); the alternative is an operator-only swap until S6, which keeps the board empty until the first operator swap and defeats #717's goal. (2) Completion role (also the role stated after the failure table; that statement is this default until the user decides): default `completed_degraded` for a session whose only failure is the post-publication `serving_projection`; the alternative is `failed`, which would re-trigger delivery and backup retry logic for a session whose publication already succeeded.

## Read side: serving root and nightly current release

Production reads use one configured serving root, `<ops_root>/serving`: `serving.sqlite`,
the immutable `objects/` store and `publisher/`, the serving pointer root the API takes as
`publication_root`. It is configuration, never a checkout path. The ops publication scope
root is a different pointer (bundle-only release, read by health and delivery); the API
does not resolve "current" from it.

`/releases/current` after a nightly resolves: serving `publisher/CURRENT` -> that release's
`projection_binding.json` -> projection release id -> `verify_projection_binding` against the
live index. Only the fenced ops publisher replaces that pointer, on request from the nightly
`serving_projection` job (after the projection is committed and its binding verified) or an
operator retained publication that the fence allows (see the fence guide for rollback); the swap is one atomic file replace into a
complete release directory. A request resolves once, so it sees the previous release or the
new one, never a mix. Index rows are append-only, so the old binding verifies during the swap.

| Condition | Read outcome |
|---|---|
| Nightly's `serving_projection` failed before the pointer swap, or has not run | `CURRENT` unchanged: the previous night's release is served |
| Serving root missing, or it has no `CURRENT` yet (first run, or no successful projection) | `NO_CURRENT_RELEASE` (503, retryable), as today |
| `serving.sqlite` fails to open, once `CURRENT` exists | An open-time `sqlite3.Error` (the resolver's connect or any route's open) returns `SERVING_INDEX_UNREADABLE` (503, retryable). The native-parity freshness check reads a failed re-open (`sqlite3.OperationalError`) as `available` |
| Pointer names a release whose binding fails verification | `CURRENT_BINDING_INVALID` (500); no fallback to an older release |

Serving gains no write path, retry or job start: the swap is the ops job's fenced
publisher, and the read contract above is otherwise unchanged. Clients compare the release
they show with the job's `serving_status.json` sidecar in the serving root (attempted versus
served release and failure code; unknown when unreadable) to detect a night whose projection
failed. The full proposed contract stays in the guide until implemented.

