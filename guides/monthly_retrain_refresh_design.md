# Monthly retrain + pool refresh inside the nightly (cutover PR-13b)

Design only. Refs #805, #192. Companion to the nightly pool refresh (#192
slices 3-4, `engine/v2/models/ARCHITECTURE.md` §2.1 and root
`ARCHITECTURE.md`'s "Native nightly pool/residual refresh"). Where this file
conflicts with a component `ARCHITECTURE.md`, that doc wins; the slices below
move the agreed contract into those docs in the same PR as the code.

User decision 2026-10-16: objects that change only on a retrain are refreshed
**monthly, inside the nightly, on the first nightly run after a calendar
month completes, covering the previous month**. Not every night, and not by a
separate manual job. §2 **changes** "first run after the month completes" to
"first run whose settled-event cutoff has reached the month end" (later by the
settlement lag); the user should confirm.

## 1. What changes and what does not

| Object | Nightly (slices 3-4) | Monthly retrain (this design) |
|---|---|---|
| Board analog pool, paired residual pool, trailing cutoff | Advanced every night to the settled-event cutoff | Unchanged (still nightly) |
| Three driver residual pools, chooser analog pool | Carried (same hash, recorded at the prior `pool_cutoff`) | Regenerated at the month-end cutoff |
| Champion and gate models | Carried (bindings unchanged) | Retrained (see D1) |
| Chooser row | Carried | Regenerated with the chooser pool |

The monthly refresh is one more reconcile in `Service.tick()`, the same shape
as the existing `_reconcile_computed_moves_refresh` and the planned
`_reconcile_pool_nightly_refresh` (#192 slice 4; never a `GRAPH`/`_DAG_STAGES`
node). It publishes through the planned `phase5_state_stage` job (#806, #192
slice 3; today only `run_state_job` in `tools/phase5_training_job.py` exists):
the retrained objects
arrive as non-carried `PoolOutput`s, everything else is carried or advanced as
on any night. There is no second release-assembly path.

## 2. Trigger rule and idempotency

**Due set.** For a refresh session dated `s`, the *completed* months are all
calendar months before `s`'s month. Evaluate every completed month with no
durable record (retrain or `superseded`; the first run, with no records, marks every month before the latest completed one `superseded`); the **target** is the latest due one,
`M`. Its **retrain cutoff** is the last calendar day of `M` (`M_end`), a data
date, fixed by `M` alone and independent of the night the job runs.

**Due.** Month `M` is due when all hold: (a) no retrain record exists for `M`
in any state; (b) the nightly's settled-event cutoff is on or after `M_end`
(events settle with a lag, so the first night of the new month is often not yet
due and the rule simply waits); (c) the nightly pool refresh for the session
has succeeded (the retrain starts from a verified, promoted incumbent); (d) no
other heavy job is running (§6).

**Exactly once.** The job key is `retrain:{M}`, durable like every ops job key.
An existing key in any state (queued, running, succeeded, failed, cancelled) is
never resubmitted; the reconcile derives "done" from durable job and release
state each tick, with no in-memory phase cache. A crash between "retrain
succeeded" and "stage submitted" is recovered by the next tick: the stage key
is `retrain-stage:{M}:{incumbent_release_id}` and the reconcile submits it when
the retrain job succeeded and either no stage key exists for `M`, or the only
one failed `ConcurrentPromote` (§4). Stage attempts are bounded at two per `M`;
any other stage failure is terminal for `M`, like `retrain:{M}`.

**Retry.** A hard failure is terminal for `retrain:{M}`. No automatic
replacement key. Recovery is operator-driven; the missing replacement-key path is the open gap
#154, which this design does not close.
A failed retrain costs stale driver pools/models, not a stale board: the
nightly advance and carry continue on the incumbent. A *running* retrain is
different: it delays the next session's heavy steps, including scoring (§6, D6).

**Missed month.** Training is cumulative to its cutoff, so a missed month is
subsumed by the next one. If months `M1 < M2` are both in the due set (the system was down,
or months were never evaluated), only the latest, `M2`, is retrained; earlier
due months get a durable `superseded:{M1}` marker naming `M2`, written once, so
they are not re-evaluated every tick. A failed `M1` followed by a due `M2`
therefore recovers by itself without reusing the failed key.

**Month boundary and holidays.** Month completion is the calendar month, not the
last trading day: `M_end` may be a weekend; the settled-event cutoff must reach
it, which holiday and weekend lag handle without a calendar special case.

## 3. What is retrained and regenerated

Inputs to every step are pinned: the incumbent release id, `M_end`, and the
refresh session's pinned snapshot. The snapshot may hold data after `M_end`; a
row filter on event and outcome data dates enforces the bound, so the retrain
admits no row dated after `M_end` and reads no live registry.

1. **Champion/gate/chooser retrain** (D1): the existing training kinds run with
   `cutoff = M_end` over rows whose event and outcome data dates are on or before `M_end`. Fold
   rules are the existing training contracts. Experiment holdouts are not
   applied: no training code reads them, and they scope experiment research, not
   the deployed model. Whether production training should exclude them is D7.
2. **Regenerate** the three `driver_residual_pool:{size,implied_t1,runup_move}`
   pools and the chooser analog pool at `M_end` from the incumbent models (retrained ones only under D1(b)).
3. **Stage** one candidate release: the six pool members plus chooser row via
   `phase5_state_stage`, retrained models bound, all other bindings carried.

**Causality (data dates, never `computed_at`).** Every retrained object records
`retrain_cutoff = M_end`. A training row is admitted only if its event and
outcome data dates are on or before `M_end`. The release then scores events
dated after `M_end`; the nightly keeps advancing the board pool and trailing
cutoff beyond `M_end` while the retrained members stay at `M_end` until the next
retrain.

**Per-member cutoffs (contract change).** `derive_catalog` today requires every
non-carried object to carry the session cutoff, and `sources.pool_cutoff` is a
single value. Retrained members are older than the session cutoff by design. The
catalog gains `sources.retrain_cutoff` (an ISO date not after `pool_cutoff`);
retrained members must carry `retrain_cutoff`, nightly-advanced members the
session cutoff, and carried members their prior recorded cutoff. A retrain cutoff
that does not advance past the prior `retrain_cutoff` refuses
`CUTOFF_NOT_ADVANCED`.

**Races with the nightly.** The retrain runs for hours; the nightly promotes
new releases meanwhile. The stage job therefore takes its base from the
**incumbent at stage time**, not from the release at retrain start, and carries
the nightly-advanced members from that incumbent. Retrained objects depend only
on the incumbent's model payload identity at retrain start; if the incumbent's
champion/gate bindings changed in between (only a retrain or an operator promote can do that),
stage refuses `PRIOR_MISMATCH` and the retrain is wasted, not applied.

## 4. Promotion, gating and rollback

Pool/chooser-only retrains (models unchanged) follow the nightly path: staged by
`phase5_state_stage`, light consistency checks, then release-scope
`phase5_acceptance` (the only writer of the staging-success record, models doc
§7.2), auto-promoted with `expected_previous_release_id`.

A release with **changed champion/gate bindings** has a new semantic hash and
needs the full staging gate before it can be promoted:

1. `phase5_acceptance` release-scope acceptance passes and
   `mark_staging_succeeded` records it (a failed or missing record refuses
   promotion, models doc §7.2).
2. A non-regression comparison on the **check month** `C = M − 6`. With
   `ExperimentHoldouts(as_of_month=M)` (`engine/v2/foundation/experiment_holdouts.py`)
   every event in `M−5 … M` is "rolling" and later ones "ambiguous", so `C` is
   the newest month outside the rolling holdout. A **check candidate** (same
   recipe, fit with training cutoff `C_start − 1`, where `C_start` is the first
   day of `C`) and a **check incumbent** (the incumbent's recipe refit the same
   way, since the incumbent's own fit post-dates `C`) both score events dated
   within `C`, with every pool and the trailing cutoff pinned to `C_start − 1`
   for both, so no scored event sits in a model or pool that scores it. Events labelled "random" (3%) or "ambiguous" are excluded (about 97% of
   `C` remains); the check uses no holdout outcomes. Slice 5b tests assert both
   pins and the exclusion. Both extra fits count in the heavy-job budget (§6). The check
   certifies the *recipe*, not the release fit: its receipt is bound to the
   check candidate hash, the release candidate hash and the incumbent id, and
   the promote step requires a receipt naming exactly the release hash being
   promoted, alongside the `mark_staging_succeeded` record. The check is **consistency of ranks and
   paired differences with the incumbent's, not a magnitude cutoff**; the
   measures and the pass rule are written into the slice that builds them and
   are labelled there as design judgement.
3. Promote with `expected_previous_release_id` = the incumbent. A concurrent
   nightly promotion refuses `ConcurrentPromote`; the next tick re-stages on the
   new incumbent (the retrained objects stay valid, §3) and submits the second
   and last stage attempt (§2) under a key naming the new incumbent.

Failure at 1-3 leaves the candidate staged and unpromoted, the incumbent
deployed, and a failure receipt (the job result naming the step and refusal code, e.g. `StaleReleaseHash`); no automatic next-tick action, operator recovery as in §2.

**Rollback.** `deployment.rollback` undoes exactly one promotion, and nightly
promotions follow the retrain promotion, so it cannot target the retrain itself
after the first night (D3). The designed path is a forward promotion of a
release re-derived from the pre-retrain models (explicit
`promote(release_id, expected_previous_release_id=…)`); the next nightly
re-advances the board pool from it.

## 5. Failure semantics (R1-R6)

| | Retrain job (`retrain:{M}`) | Stage (`retrain-stage:{M}:{incumbent}`) | Promotion |
|---|---|---|---|
| R1 verified input | Incumbent release id/hash, `M_end`, data bound and training inputs verified before any compute; a cutoff past the settled-event cutoff refuses | Retrained outputs' hashes, `retrain_cutoff`, member ids and the incumbent's model payload hashes re-verified; drift refuses before any write | Staging-success record matches the candidate hash; incumbent equals expected id |
| R2 cache | None: reads job, release and pointer state every tick | No cached catalog or artifact reads | Pointer read for this swap only |
| R3 retry | No resubmission of a key in any state; failed is terminal; the next due month supersedes | At most two stage keys per `M`: one per incumbent, the second only after `ConcurrentPromote`; other failures terminal | No retry or alternate target on refusal |
| R4 transaction | Training outputs are written to per-attempt staging and hashed only after a clean exit | New candidate release id; `DEPLOYED` untouched | Existing atomic pointer/history swap |
| R5 partial failure | Orphan outputs may remain; nothing is staged or promoted; incumbent keeps serving | Orphan objects or a manifest without a catalog may remain unused; a later stage attempt only under R3's `ConcurrentPromote` rule; no promotion | A refusal leaves pointer and history unchanged; the candidate stays staged |
| R6 idempotency | Same month, same key, never trains twice | Planned-job rule: identical inputs derive identical catalog content; a reused id with different content refuses | Re-promoting the live id is a no-op |

Reconcile-level rows:

| Condition | Outcome |
|---|---|
| Month not yet due (cutoff short of `M_end`, or session refresh failed) | No-op this tick; re-evaluated next tick |
| Another heavy job running (non-blocking admission probe says busy) | Not submitted; no key written; re-evaluated next tick |
| Reconcile itself raises | Degrades only this sidecar, like the other sidecars; required job dispatch is not blocked |
| Retrain running past the next session | That session's required-graph heavy steps wait for admission, non-heavy steps proceed (§6), so that session's scoring is delayed until the retrain ends (no bound until the deadline deliverable exists; D6); no second retrain is started (one key per month, one heavy job at a time) |
| Release store missing or incumbent unreadable | Refuse; no fallback to a stale release or a legacy rebuild |

## 6. Resource budget

The retrain is a **heavy job**: it runs only under the existing exclusive heavy
profile (`engine/v2/contracts/operations.py`; `tools/bounded_run.py --heavy`:
one at a time, bounded memory and cores). That tool has no job wall-clock limit
and its heavy wait is unbounded unless `--max-wait-s` is set, so a heavy submit
waits for admission rather than being denied. Consequences, stated once:

- The reconcile never blocks on admission: it uses a non-blocking probe (is a
  heavy job running?) and submits only when none is; otherwise it re-evaluates
  next tick (§5).
- It is submitted only when the session's required graph and the pool refresh
  have finished, never from agent sessions or CI.
- If the next session starts while it still runs, that session's heavy steps
  (including required-graph heavy steps) wait for admission until the retrain
  ends; non-heavy steps proceed. Accepted because a monthly cadence makes overlap
  rare.
- Neither a wall-clock deadline nor a non-blocking admission probe exists today
  (`bounded_run` only blocks). Slice 4 decides whether the retrain is enqueued
  through `ops/scheduler.py`'s admission decision or probed directly. The job
  envelope already has `deadline_at` (`contracts/jobs.py`) but nothing enforces
  it; enforcement (terminal failure) is a deliverable of each job's
  slice (3 for pools, 5a for models), sized from the receipt below; the probe
  and the D6(b) start window belong to slice 4.

Slice 3 records measured peak memory and wall time as a receipt; budgets in the
job spec come from that measurement, not from this document.

## 7. Parity and qualification

- Parity reports already pair native with legacy per session and record the
  release id used. A retrain (or any model-changing promotion) is a release id
  change; the report marks the first session scored under a new release
  `release_changed` and does not change its comparison rules (the bar-B
  per-field tolerances apply as before).
- The retrain is submitted only after the session's required graph (which holds
  the score job) and the pool refresh finished (§2 condition c), so promotion
  follows that session's scoring and no session scores half under old and half
  under new models; the release in force at score time is the one recorded.
- Legacy rebuilds its pools live every run and does not retrain monthly; a
  mismatch that first appears on a `release_changed` session and persists is a
  real divergence to investigate, not an expected retrain effect.
- Whether a model-changing retrain inside a qualification window resets or
  continues the session count is D4.

## 8. Open decisions (hand-back, not decided here)

- **D1 Scope of the first retrain.** (a) *Pools and chooser only first*, models
  unchanged: smallest, reuses the nightly path and its auto-promotion, closes
  the staleness the user cares about (pools at month end), no model-quality
  gate needed; champion/gate retrain follows as a later slice. (b) *Models and
  pools together from the start*: matches the PR-13b text, but needs the
  acceptance + non-regression gate (§4) before any slice can merge a usable
  path. Trade-off: (a) ships earlier and de-risks, (b) delivers the full intent
  in one step. Recommendation: (a), with (b) as slices 5a-6 below.
- **D2 Model-changing promotion.** (a) *Auto-promote after acceptance and the
  non-regression check* (matches "inside the nightly"). (b) *Stage and hold for
  an operator promote*. (a) removes manual work, (b) puts a human in front of an
  irreversible-in-practice model change. Recommendation: (a) once the check
  exists, (b) until then.
- **D3 Rollback.** (a) *Forward-promote a release re-derived from the
  pre-retrain models* (§4; keeps the undo-stack invariant, costs one night of
  re-advance). (b) *Allow `rollback` to target a named release*: simpler to use,
  breaks the documented "rollback undoes exactly one promotion" invariant.
  Recommendation: (a).
- **D4 Qualification across a retrain.** (a) *Count continues*, retrain
  sessions reported separately; (b) *count resets* at a model-changing release.
  (a) keeps momentum toward the cutover date; (b) makes the qualified release
  the one that goes live. Recommendation: (b) only for model-changing retrains
  (so with D1(a) nothing resets).
- **D6 Retrain vs next session's scoring.** A running retrain delays heavy
  scoring unboundedly until a deadline exists. (a) *Deadline only* (slices 3/5a
  fail the job at a limit sized from the receipt); (b) *start only in a window*
  with enough hours before the next session's heavy steps; (c) *let scoring
  preempt*. (a) is simplest, (b) avoids the delay, (c) needs executor work.
  Recommendation: (a)+(b).
- **D7 Experiment holdouts in production training.** (a) *Not applied* (§3):
  the deployed model sees all data to `M_end`; (b) *exclude* random and rolling
  holdouts from retrain rows so the holdouts stay clean for later experiments,
  at the cost of a model that ignores the newest six months. Recommendation: (a).
- **D5 Missed months.** Designed as latest-only with a `superseded` marker (§2);
  the alternative is to retrain each missed month in order. Recommendation:
  latest-only; ordering adds cost and no information.

## 9. Split list (code slices, one PR each, in order)

Sizes are non-test code / test / doc lines, estimated.

1. **Catalog `retrain_cutoff`.** `sources.retrain_cutoff` and per-member cutoff
   checks in `derive_catalog`/`carry_forward_release`; `CUTOFF_NOT_ADVANCED` for
   retrain members. 70-100 / 80-110 / 25-35. Depends on #806 and #192 slice 2.
2. **Due-month decision.** A pure function over session date, settled-event
   cutoff and durable job state returning `NOT_DUE | DUE(M) | SUPERSEDE(M…)`;
   the `retrain:{M}` and `superseded:{M}` key shapes. 80-110 / 100-140 / 20-30.
3. **Pool/chooser regeneration job** (D1 a). Worker producing the three driver
   pools and the chooser pool at `M_end`, bounded I/O, the heavy-job receipt and
   the job wall-clock deadline (§6); no model change. 150-200 / 120-160 / 30-40.
   Depends on 1.
4. **Monthly sidecar.** `_reconcile_monthly_retrain` submitting retrain then a
   `retrain-stage` job (retrained members non-carried), with the §2 crash and
   race rules. 140-180 / 150-200 / 35-50. Depends on 1-3 and #192 slice 4.
5. **Champion/gate retrain** (D1 b). Retrain kind, acceptance gate wiring, the
   non-regression check and receipt; 5a owns the model retrain deadline. Split: 5a retrain kind and check-candidate
   fit, 150-190 / 130-170 / 30-40; 5b acceptance wiring and the non-regression
   check with receipt, 120-160 / 120-160 / 30-40.
6. **Promotion and rollback path** (D2, D3). Guarded promotion of a
   model-changing candidate and the forward-rollback plan. 100-140 / 100-140 / 25-35.
7. **Parity marker** (§7). `release_changed` on the first session of a new
   release; qualification handling per D4. 60-90 / 70-100 / 15-25.

Each slice adds focused synthetic tests for its R1-R6 promises; none touches the
local model release store, runs the nightly, or runs a heavy job.

Library check: no standard tool covers a content-addressed release catalog or
this job-keyed reconcile; the design reuses the existing training, staging,
promotion and executor surfaces.
