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
shadow-submission check runs on every real supervisor tick, but the actual
`native_score_batch` submission it gates is conditional (a pinned snapshot
and no existing job for that identity). The chain is still functionally a
no-op in production, though: the raw-row producer
that turns a real night's events into per-event inputs ("cutover PR-6",
tracked in #199, split into slices 6a-6f) has only slice 6a merged, so the
shadow submission either no-ops (default legacy input mode) or explicitly
raises "raw-row producer is not built yet" when a snapshot is pinned.

The redone native_parity comparison job (plan PR-4, re-scoped 2026-09-28 to
run as a real supervisor-submitted job) has 4 of its 5 planned slices merged;
the 5th, the tick-loop caller (#211), is open in review. A separate,
not-yet-scoped-as-required item, slice 2B(c) (a nightly-graph node-width
change), hasn't started. The dashboard's serving
projection and native_parity page (#189, #198) are both merged, but the page
renders empty (`no_report`) because nothing upstream has produced a real
report yet. The production release binding also has no release root
configured in this checkout's environment, so it fails closed until an
operator sets one.

**Critical path to a populated side-by-side page:** finish raw-row producer
slices 6b-6f → confirm `native_score_batch` scores a real event → merge #211
and decide slice 2B(c)'s scope → configure the production release root. The
dashboard side itself (projection, API route, page) is already built and
needs no further PRs; it just has nothing to show yet. Everything after that
(LegacyScoreBridge replacement, native decisions-predictions, the 10-session
qualification) is Phase 7 work that hasn't started. Slice 6b has a PR open
(#239, in code review); every other unstarted item below has no PR open.

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
   (calendar row) is in code review at #239; 6c-6f have no PR open. The
   shadow-submission path currently either no-ops or raises a "not built
   yet" error, confirmed in code. *Note: this is a different "PR-6" than
   the plan's original S4C job-kind-wiring PR-6 — see the naming note
   below.*
5. **native_score_batch shadow submission (PR-7a)** — Code merged,
   functionally no-op. #88, #126. Called every real tick, gated by a
   release-availability check; produces nothing until step 4 lands.
6. **native_parity job (PR-4, redone 2026-09-28)** — 4 of 5 planned slices
   merged, the 5th open in review. #132, #185, #191, #227 merged; #211 open.
   Pairs a night's legacy and native records field-by-field. Re-scoped after
   the original design named a production caller that nothing actually
   calls; the real design is a supervisor-submitted job. Open: the
   tick-loop caller (#211, in review). Separately, not counted in the 5:
   slice 2B(c), a nightly-graph node-width change, not started — scope not
   yet confirmed as required for Phase 7.
7. **Dashboard side-by-side view** — Page and authenticated JSON route
   merged. #189 (serving projection), #198 (page + route). Renders a nightly
   summary, per-field diff breakdown, and worst-rows drill-down, but shows
   `no_report` today because the production report path (steps 4-6) hasn't
   produced one yet. No further dashboard PRs are queued; this step is
   waiting on data, not code.
8. **Operator step: configure the production release root** — Not done. A
   model release was deployed 2026-09-28, but the release-root binding is
   environment-only and unset in this checkout, so the release lookup fails
   closed every tick until an operator sets it. No downstream dependency,
   but not worth doing until step 4 is closer to done.
9. **Native nightly timer/scheduling install** — Not asked yet. Queued to
   ask the user once the raw-row producer lands. No scheduler entry for
   either the legacy or native nightly was found in this checkout; how the
   real nightly triggers in production is unverified here.

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

- **Native nightly timer/scheduling install.** Queued to ask once the
  raw-row producer (step 4) lands; not yet asked.
- **DYN-SV lineage rerun (EXP-160→169) on v2 trades.** Still undecided,
  separate from the side-by-side track — the recommendation on file is to
  redo it, but the user hasn't ruled.
- **Scope of slice 2B(c)** (the nightly-graph node-width change for
  native_parity). No ruling yet on whether it's required before Phase 7 or
  can stay a follow-up; worth confirming once #211 merges.

## Related open issues

- **#136** — native_parity_handler tests derive their "legacy" fixture from
  the native fixture rather than an independent source; a real risk to
  trusting mismatch detection once the sidecar goes live.
- **#135** / **#214** — docs (native_parity_report.py diagram; ops
  ARCHITECTURE.md's failure-semantics table) both currently claim a
  production caller/sidecar that doesn't exist yet; expected to self-resolve
  once #211 merges and docs are updated.
- **#199** — parent tracking issue for the raw-row producer; 5 of 6 slices
  remain.
- **#200** — pin_snapshot_inputs's CAS check, half of PR-7b-3.
- **#206** — forward_calendar_refresh has no nightly graph node or tick-loop
  submitter.
- **#192** — PR-13a has a merged design but zero implementation.
- **#154** / **#137** / **#149** — PR-13a design gaps: no recovery path for
  a hard training failure or a stale champion after a concurrent promote; no
  lock against a direct non-job promote/rollback caller; can't distinguish a
  light-check-failed staged release from a succeeded one.
- **#233** — a nightly.py schema-check helper misclassifies an error as
  retryable; affects native_score_batch/native_parity retry behavior.
- **#128** — the native refresh job's idempotency-key parser rejects every
  real key; affects the native refresh job's identity, not
  native_score_batch directly.
- **#220** — two modules still leak legacy filesystem paths into error
  details; a hygiene gap, not a functional blocker.
