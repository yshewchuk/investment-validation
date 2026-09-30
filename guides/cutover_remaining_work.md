# Cutover remaining work

Verified 2026-09-30 against `origin/main`, `gh pr list --state all`, and
`gh issue list --state open`. Companion to the
[delivery plan](rearchitecture_delivery_plan.md) and the
[Phase 7 cutover plan](rearchitecture_phase7_cutover.md): this tracks what is
actually merged versus open today, not the design. Where this file conflicts
with those, they win and this file is stale.

## Summary

The three core scoring-plumbing PRs (release reader, `SourceBundle`
assembler, batch dispatch — plan PR-1/2/3) are merged, and PR-7a's
shadow-submission check runs on every real supervisor tick. Since Cutover
PR-7b's `input_mode="snapshot"` flip, the real nightly's "score" stage
always pins a snapshot, so this no longer silently no-ops under a default
"legacy" mode: each tick attempt that clears its own bounded
backoff/attempt-limit gate reaches the still-missing raw-row producer and
raises `VALIDATION_FAILED` ("cutover PR-6 (raw-row producer) is not built
yet"), which the caller catches, reports and backs off like any other
transient failure. The raw-row producer that turns a real night's events
into per-event inputs ("cutover PR-6", tracked in #199, split into slices
6a-6f) has slice 6a merged and slice 6b open (#239); 6c-6f have no PR open.

The redone native_parity comparison job (plan PR-4, re-scoped 2026-09-28 to
run as a real supervisor-submitted job) is now fully merged, including the
tick-loop caller (#211): `Service._reconcile_native_parity` runs every real
tick. A separate, not-yet-scoped-as-required item, slice 2B(c) (a
nightly-graph node-width change), hasn't started. The dashboard's serving
projection and native_parity page (#189, #198) are both merged, but the page
renders empty (`no_report`) because no `native_score_batch` has ever
succeeded to produce a report from. The release-root binding is
environment-only; this checkout has it unconfigured, which fails the
release lookup closed here. Whether production's own environment has it set
is unverified from this checkout.

**Critical path to a populated side-by-side page:** finish raw-row producer
slices 6b-6f → confirm `native_score_batch` scores a real event → decide
slice 2B(c)'s scope → confirm the production release root is set. Both
native_parity and the dashboard side (projection, API route, page) are
already fully built and need no further PRs; they just have nothing to show
yet. Everything after that (LegacyScoreBridge replacement, native
decisions-predictions, the 10-session qualification) is Phase 7 work that
hasn't started. Slice 6b has a PR open (#239, in code review); every other
unstarted item below has no PR open.

## Critical path to the dual dashboard

1. **Production release reader** — Done. #47, #59. Resolves model identity/
   artifacts/analog/payoff/recalibration from the deployment pointer;
   confirmed called every tick.
2. **Per-night SourceBundle assembler** — Done, unfed. #48, #67. Builds
   context/raw_quotes/feature_vector/mask/recipes from staged rows; merged,
   but its only intended production caller (step 4) isn't built yet, so it's
   reached only by tests.
3. **Batch assembler + native_score_batch worker dispatch** — Done, wired,
   blocked upstream. #66. Turns per-event inputs into a submitted job; worker
   dispatch exists.
4. **Raw-row producer ("cutover PR-6", 6 slices)** — 6a done, 6b open, 6c-6f
   not started. #212 (merged), #239 (6b, open), #199 (tracking issue).
   Enumerates real board requests and stages the per-event rows the
   assembler needs. Slice 6a (events scan + enumeration) is merged; 6b
   (calendar row) is in code review at #239; 6c-6f have no PR open. Since
   the real nightly's "score" stage always pins a snapshot (Cutover PR-7b),
   each attempt that clears the shadow submission's own backoff/
   attempt-limit gate raises "not built yet" rather than quietly no-oping.
   *Note: this is a different "PR-6" than the plan's original S4C
   job-kind-wiring PR-6 — see the naming note below.*
5. **native_score_batch shadow submission (PR-7a)** — Code merged, active
   but blocked. #88, #126. Called every real tick; reaches step 4's gap
   (see above) until the raw-row producer lands.
6. **native_parity job (PR-4, redone 2026-09-28)** — All 5 planned slices
   merged. #132, #185, #191, #227, #211. Pairs a night's legacy and native
   records field-by-field. Re-scoped after the original design named a
   production caller that nothing actually calls; the real design is a
   supervisor-submitted job. `Service._reconcile_native_parity` runs every
   real tick, but has nothing to pair yet since no `native_score_batch` has
   succeeded (step 5). Separately, not counted in the 5: slice 2B(c), a
   nightly-graph node-width change, not started — scope not yet confirmed
   as required for Phase 7.
7. **Dashboard side-by-side view** — Page and authenticated JSON route
   merged. #189 (serving projection), #198 (page + route). Renders a nightly
   summary, per-field diff breakdown, and worst-rows drill-down, but shows
   `no_report` today because the production report path (steps 4-6) hasn't
   produced one yet. No further dashboard PRs are queued; this step is
   waiting on data, not code.
8. **Operator step: confirm the production release root is configured** —
   Unverified from this checkout. A model release was deployed 2026-09-28.
   The release-root binding is environment-only and unset in this checkout,
   which fails the release lookup closed here; whether production's own
   environment has it set is unverified. No downstream dependency, but not
   worth chasing until step 4 is closer to done.
9. **Native nightly timer/scheduling install** — Files checked in;
   production install unverified. `ops/systemd/` holds a service and timer
   unit that run `engine.v2.ops.nightly_trigger` on a 30-minute timer as a
   oneshot job. Whether this unit is actually installed and enabled on the
   production host is unverified from this checkout.

## Remaining work to the Phase 7 cutover

| Item | Status | Depends on | Notes |
|---|---|---|---|
| Plan PR-0 — inventory hygiene | Not started | — | No PR scoped as PR-0 found. |
| Plan PR-5/PR-6 (original) — S4C job-kind + nightly-graph wiring | Mostly done, one gap open | — | `computed_moves` fully wired (#39, #40, #50, #54). `forward_calendar_refresh` is a registered job kind (#83) but has no nightly graph node or tick-loop submitter — #206. |
| PR-7b-3 — pin_snapshot_inputs CAS guard | Half done | — | `calendar_version` shipped inside #212. `expected_snapshot_id` CAS check is open (#200). |
| Plan PR-8 — replace LegacyScoreBridge | Not started | PR-7 producing trustworthy native output | Bridge is unchanged. |
| Plan PR-9 — decisions-predictions from native ScoreRecords | Not started | PR-7/PR-8 | Prediction-row building is still legacy-sourced. |
| Plan PR-10 — native models-evidence-stage | Deferred | — | User deferred to post-cutover. |
| Plan PR-12 / UD-4 — research tools onto pinned v2 snapshot reads | Done | — | #68, merged. |
| PR-13a — native nightly pool/residual refresh (design) | Design done | — | #124 + fix-forward #158 merged. |
| PR-13a — implementation | Not started | PR-13a design | #192: no job kind, no sidecar functions exist yet. Related liveness gaps: #137, #149, #154. |
| PR-13b — monthly retrain cadence | Not designed | PR-13a | User approved a split cadence; no design PR yet. |
| PR-13c — native Tier-4 forecast table producer | Not designed | PR-13a | Named in #192 as a separate design; nothing started. |
| Cutover parity tolerance classification | Policy agreed, no data yet | step 6 above | User approved the classification approach; nothing to classify until native_parity runs on a real night. |
| Plan PR-11 — Phase 7 flip (default to native, run 10 qualified sessions) | Not started | everything above | Legacy board must keep publishing and stay tradeable throughout qualification. |
| Cloudflare Access check on the dashboard | Unverified | — | Flagged open previously; no closing PR/issue found, and no open issue tracks it either. |

Naming note: the plan's original PR-6 (S4C job-kind/graph wiring for
`computed_moves`/`forward_calendar`) and the later "cutover PR-6" (raw-row
producer for `native_score_batch`, #199, slices 6a-6f) are two different,
unrelated pieces of work that share a PR number from different planning
passes. Both are tracked separately above.

## Decisions waiting on the user

- **Native nightly timer/scheduling install.** The service/timer unit is
  checked in (step 9); whether it's installed and enabled on the production
  host is unconfirmed and not yet asked.
- **DYN-SV lineage rerun (EXP-160→169) on v2 trades.** Still undecided,
  separate from the side-by-side track — the recommendation on file is to
  redo it, but the user hasn't ruled.
- **Scope of slice 2B(c)** (the nightly-graph node-width change for
  native_parity). No ruling yet on whether it's required before Phase 7 or
  can stay a follow-up, now that #211 has merged without it.

## Related open issues

- **#136** — native_parity_handler tests derive their "legacy" fixture from
  the native fixture rather than an independent source; a real risk to
  trusting mismatch detection once the sidecar goes live.
- **#135** / **#214** — docs (native_parity_report.py diagram; ops
  ARCHITECTURE.md's failure-semantics table) predated #211; both remain
  open, so whether they still describe a stale state or were updated by
  #211 itself is worth a fresh look rather than assumed self-resolved.
- **#199** — parent tracking issue for the raw-row producer; 5 of 6 slices
  remain.
- **#200** — pin_snapshot_inputs's CAS check, half of PR-7b-3.
- **#206** — forward_calendar_refresh has no nightly graph node or tick-loop
  submitter.
- **#192** — PR-13a has a merged design but zero implementation.
- **#154** / **#137** / **#149** — PR-13a design gaps: no recovery for a
  hard training failure or stale champion; no lock against a direct
  non-job promote/rollback; can't tell a light-check failure from success.
- **#233** — a nightly.py schema-check helper misclassifies an error as
  retryable; affects native_score_batch/native_parity retry behavior.
- **#128** — the native refresh job's idempotency-key parser rejects every
  real key; affects the native refresh job's identity, not
  native_score_batch directly.
- **#220** — two modules still leak legacy filesystem paths into error
  details; a hygiene gap, not a functional blocker.
