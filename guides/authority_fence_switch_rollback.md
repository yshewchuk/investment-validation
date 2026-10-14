# Authority fence and switch/rollback protocol (P7-3, #739)

Design, not yet implemented. Source: `guides/rearchitecture_phase7_cutover.md` (P7-3, "Switch and rollback protocol"); `guides/rearchitecture_phase8_post_cutover.md` 8A entry. `engine/v2/ops/ARCHITECTURE.md` is at its ceiling, so the contract lives here until each slice folds its part in. No tool in this design switches production by itself: the activation step needs the standing session authorization and deployment permission (P7-5). Every drill below is supervisor-run against a scratch or private target (heavy jobs serialize; owners never run them).

**Decisions (owner, 2026-10-14, #739, not reopened).** The rollback window is **10 completed sessions** after the switch, mirroring P7-2. 8A may delete the legacy tree only after the tenth completed session has been verified. Legacy catch-up inside the window is specified in "Legacy catch-up".

**Standard tools.** Nothing here is a standard-format problem. The backup, restore, pointer and fence primitives already exist and are reused; the only new state is one authority record and one receipt format (JSON, hash-checked like the P7-1 manifest). A distributed lock service or external lease library is not needed: there is one host and one SQLite catalog, and the fence is a row in the same catalog transaction as the writes it guards.

## Invariant and fence

**Exactly one official writer.** At any instant at most one deployment (`legacy` or `v2`) may commit an official decision, settlement, ledger row, export or consumer-visible release. The fence is the existing `decision_authority` singleton in the v2 catalog (`engine/v2/ledger/decisions.py`: `owner`, `generation`, `switched_at`; `set_authority(conn, expected_owner, owner, stamp)` is compare-and-swap on `owner` and increments `generation`). It is extended, not replaced:

1. **Authority generation.** Every official commit path (`decision_commit`, outcome settlement, publication of a release that moves a consumer pointer) reads `decision_authority` inside its own write transaction and refuses unless `owner == "v2"` and the generation it was started with equals the current one. A writer started under generation N therefore cannot commit after generation N+1 exists, even if its process survived (a "zombie").
2. **Writer lease on the legacy side.** The legacy deployment has no catalog. Its writer lease is a lock file in the retained legacy data root, `authority.json {owner, generation}` plus an advisory `flock` taken by the legacy scheduler wrapper before any ledger append. Fencing legacy means writing `owner: "v2"` with the next generation and disabling the legacy scheduler entry; the wrapper refuses to run when `owner != "legacy"` or the generation differs from the one recorded in its crontab/unit environment. Slice 1 provides this wrapper and file; until then the legacy scheduler is fenced only by disabling it, which the receipt records but the drill treats as insufficient (a restarted scheduler must refuse).
3. **Transfer is two ordered writes, never one.** Fence order is always: (a) fence the *current* writer (it cannot commit any more), (b) verify it is fenced, (c) only then grant the other side. A crash between (a) and (c) leaves **zero** official writers (safe: writes held, last validated read release served), never two. The reverse order is forbidden.
4. **Switch receipt.** Each step appends one JSON receipt (step, authority generation before/after, high-water marks, input hashes, result) to `<ops_root>/cutover/<switch_id>/receipts.jsonl`, written with the existing artifact-store atomic write. The receipts are what the P7 readiness checker reads; summary booleans are never trusted.

Slice 1 provides `python3 -m engine.v2.ops cutover authority {status,fence,grant}`; `status` is read-only and is the only authority command the drills use for observation.

## Switch protocol (forward)

Each step is idempotent on `switch_id` (below, R6) and refuses to start if the previous step's receipt is absent or its hashes no longer verify.

| # | Step | Command (existing or slice) |
|---|---|---|
| S1 | Capture authority and high-water marks: `decision_authority` row, v2 `decisions`/`outcomes` max rowid and count, legacy ledger predictions/outcomes line counts and tail hash, current `DEPLOYED` and `CURRENT` pointers, last completed session id | `ops cutover capture` (slice 2); counts cross-checked with `ops ledger status` |
| S2 | Stop/drain the old writer: disable the legacy schedule, wait for any in-flight legacy run to exit, verify no process holds the legacy `flock` and no run directory is open | `ops cutover drain` (slice 3) |
| S3 | Consistent backup of catalog/WAL with referenced objects, from the quiescent state | `engine/v2/ops/backup.py` `run_backup` (via the existing backup job); legacy ledger copied with hash by `ops cutover capture --legacy-copy` (slice 2) |
| S4 | Verify the restore target: restore the S3 backup to an isolated root and prove the replay, ledger reconciliation and export identity | `python3 tools/v2_restore_drill.py --backup … --restore-root … --score-request … --native-inputs … --expected-score-hash … --original-export … --generation …` (existing); P6-5 checker `checks/rearchitecture_phase6_restore_drill.py` reads its receipt |
| S5 | Fence transfer: fence legacy (write `authority.json`), verify, then `set_authority(conn, "legacy", "v2", stamp)` in one catalog transaction | `ops cutover authority fence` then `grant` (slice 1) |
| S6 | Start the v2 official schedule (the non-shadow nightly plan) | `build_nightly_plan` today refuses every mode but `shadow`; slice 4 adds an `official` mode that additionally requires a fresh S5 receipt and the current generation in the plan |
| S7 | Switch read consumers to the validated release: dashboard/API/export read the v2 serving pointer, not the legacy files | `ops snapshot promote`/pointer commands (existing) and the serving root (`guides/serving_projection_job.md`); `ops cutover consumers` (slice 5) verifies pointer, ETag and freshness |
| S8 | Verify: authentication, freshness, predictions, settlement, book and export identities for the first official session, then the next completed session | `ops cutover verify` (slice 5) plus `ops ledger status` and the existing health report |

A model rollback is a separate, already-tested concern: `engine.v2.models.deployment.rollback` (#520) and the promotion expected-incumbent guard (#500) move the model pointer only. They are reused for the "model identity" part of S8 and the pointer-only rollback control below, and are never presented as ledger recovery.

## Backout (rollback)

**Triggers** are those in the Phase 7 guide: unexplained economic drift, missing population, duplicate authority, unreconciled decisions, unrecoverable publication, missed session deadline. If facts cannot be reconciled, hold writes (authority stays with nobody) and keep serving the last validated read release.

Backout never restores an old database over new official facts, never deletes a v2 row, and never lets the old scheduler repeat a completed decision.

| # | Step | Command |
|---|---|---|
| B1 | Fence v2 first: `set_authority(conn, "v2", "none", stamp)`; wait for in-flight v2 jobs to reach a terminal state; verify with `status` | `ops cutover authority fence` (slice 1) |
| B2 | Capture the v2 high-water marks again and diff against S1: the v2 official decisions/outcomes committed since the switch are the *reconciliation set* | `ops cutover capture --diff` (slice 2) |
| B3 | Reconcile: every committed v2 decision/outcome is exported in the legacy ledger format and appended to the legacy ledger (append-only, keyed by the deterministic decision id, skipping ids already present); a v2 decision is never marked void. A divergence between a legacy and a v2 row for the same occurrence is recorded in `decision_divergences` (existing) and the rollback stops for the operator | `ops cutover backout reconcile` (slice 6), built on `engine/v2/ledger/export.py` |
| B4 | Grant legacy: write `authority.json` with the next generation and `owner: "legacy"`, only after B3's receipt verifies | `ops cutover authority grant` (slice 1) |
| B5 | Re-enable the legacy schedule; switch consumers back to the legacy release; keep v2 read-only | `ops cutover backout resume` (slice 6) |
| B6 | Verify as S8 against legacy; the v2 catalog stays intact as evidence | `ops cutover verify --side legacy` (slice 6) |

## Legacy catch-up

Rollback at session `k` of the 10-session window means legacy missed sessions 1..k, during which v2 was the official writer. The old scheduler resuming is **not** a catch-up: it must not decide those sessions again.

1. **Decisions for missed sessions are taken from v2.** B3 appends them. Legacy treats them as already decided for those occurrences; its scheduler wrapper reads the appended ledger and refuses to emit a decision for an `as_of` that has one.
2. **Settlements**: positions opened by v2 decisions that mature after the rollback are settled by the legacy settlement run like any other open position; outcomes v2 already settled are carried by B3, so nothing settles twice (keyed by decision id).
3. **Sessions where v2 committed nothing** (a missed or held session) stay missed. Legacy does not retro-decide them with stale prices; they are listed in the receipt as `no_official_decision`, which is the same disposition a missed night has today.
4. **State and caches** the legacy deployment keeps (price history, calendar, model files): the retained legacy tree and data are not mutated during the window, so they are stale by exactly the missed sessions. The first legacy run after B5 runs its normal incremental data refresh; this is verified in B6 (freshness) and is the one place the window's cost shows. Anything legacy cannot refresh in one run is a rollback blocker the operator sees before B4, because B3's receipt records the data staleness.
5. **Rollback after the window** (session 11 and later) is not supported: 8A may have deleted the legacy tree. The checker reports window state (sessions completed, window closed) and `ops cutover backout` refuses once closed.

## Failure semantics (R1-R6)

Applied to every step above (S1-S8, B1-B6) unless a row says otherwise.

| | S1-S4, B2 (capture, drain, backup, restore verify) | S5, B1, B4 (fence moves) | S6-S8, B5, B6 (start, switch, verify) | B3 (reconcile) |
|---|---|---|---|---|
| **R1 missing input** | A missing or unverifiable prior receipt, backup, legacy copy or restore target refuses with a typed code; no state changed | Missing prior receipt or `status` mismatch refuses; no authority write | Missing release, pointer or receipt refuses | Missing export or legacy ledger refuses |
| **R2 cache** | None. Every read is from the live catalog or a hashed artifact; a cached high-water mark is never accepted | None; `status` is always read live | Consumers' ETags are compared, never trusted from memory | None |
| **R3 retry** | Safe: re-running recaptures and overwrites nothing (new receipt appended, earlier kept); a changed mark is reported, not merged | Retry resumes from the observed owner/generation; never grants if the other side is not verified fenced | Retried by `switch_id`; the schedule is started at most once per generation | Retry re-runs the diff and appends only missing ids |
| **R4 transaction** | Backup is atomic by its own contract; capture is one read transaction | `set_authority` is one catalog transaction (compare-and-swap); the legacy `authority.json` write is atomic rename | Pointer swap is atomic (`os.replace`); schedule start is a single enable | One transaction per batch; the receipt is written after the commit, with the batch hash |
| **R5 partial write** | A partial backup/restore is not accepted (the drill hashes the result); partial scratch is removed or kept as diagnostics | A crash between fence and grant leaves **no** writer; it is detected by `status` and finished or reversed by an operator command | A crash after the pointer swap but before the receipt: re-run `verify` observes and records; no second swap | A crash mid-batch leaves some ids appended; the diff finds the rest; rows are never deleted |
| **R6 idempotency** | `switch_id` + step name keys the receipt; a rerun is a no-op when the hashes match | Fencing an already-fenced writer or granting an already-granted side is a no-op, a grant to the wrong generation refuses | As S5 | Keyed by decision id; a duplicate append is a no-op, a differing payload is a divergence and stops |

Cross-cutting: no receipt, log or artifact carries a local path or raw exception text (`ARCHITECTURE.md` §6); the typed code and the operation id are enough to find the private diagnostic.

## Negative controls and expected outcomes

Each is a scenario for `tools/v2_controlled_failure_drill.py` (existing scenarios `after-good-release`, `before-any-release`; slice 7 adds the three below) or the P7-3 checker, with a `fault=` hook at the named point. The drill target is a scratch copy; #740's separate scratch catalog for native shadow decisions is the rehearsal target where it fits (it already holds native decisions apart from the official ledger), so the controls need no private data.

| Control | Fault | Expected outcome |
|---|---|---|
| **Interrupted switch** | Kill after S5(a) fences legacy, before S5(c) grants v2; then, separately, after the grant but before S6 | First: zero writers, no commit possible from either side, `status` reports `unowned`, consumers still on the last validated release; operator re-runs `grant` or `fence` and the switch completes or is abandoned. Second: v2 owns but no schedule; re-running S6 starts it once. In neither case do two writers exist; both writers attempting a commit are refused in the first case |
| **Rollback before any official commit** | B1-B6 immediately after S6 with no v2 official row | Reconcile set is empty; legacy resumes; the receipts show matching S1 and B2 marks; v2 catalog unchanged |
| **Rollback after one official commit** | Commit exactly one v2 decision and one outcome, then B1-B6 | B2 shows 1 decision, 1 outcome; B3 appends them to the legacy ledger; legacy does not re-decide that occurrence; v2 rows still present and identical; ledger counts equal S1 + 1; a pointer-only rollback in the same scenario fails the check (the legacy ledger lacks the row), which is the proof a pointer rollback is not ledger recovery |
| **Duplicate writer** | After S5, start a legacy scheduler run and a stale v2 job from generation N | Legacy wrapper refuses (owner mismatch); the stale v2 job's commit is refused by the generation check inside its transaction; zero rows written; refusals appear in the receipts |
| **Reconcile divergence** | Legacy ledger already holds a differing row for a v2-decided occurrence | B3 records a `decision_divergences` row, appends nothing for that id, stops with a typed refusal; authority stays unowned |
| **Backout past the window** | Mark 10 sessions complete, then request rollback | Refused: window closed |

## Slice list

Each slice is one PR of at most about 400 non-test code lines, with tests in `tests/v2/ops/` and no heavy run by the owner.

1. **Authority fence primitives**: generation check in the official commit paths, legacy `authority.json` + wrapper, `ops cutover authority {status,fence,grant}`.
2. **Capture and receipts**: `ops cutover capture [--diff] [--legacy-copy]`, the receipt writer/format, and the high-water-mark capture.
3. **Drain**: `ops cutover drain` (stop old writer, verify no in-flight run).
4. **Official nightly mode**: `build_nightly_plan` official mode, gated on a fresh S5 receipt and generation.
5. **Consumer switch and verify**: `ops cutover consumers`, `ops cutover verify` (S7-S8).
6. **Backout and legacy catch-up**: `ops cutover backout {reconcile,resume}`, `verify --side legacy`, window state.
7. **Drill scenarios and readiness input**: the three negative controls plus divergence and past-window cases in `tools/v2_controlled_failure_drill.py`, and the receipt reader the P7 readiness checker consumes. The supervisor-run rehearsal (scratch target, before and after one official commit) produces the P7-3 evidence.

Dependencies: names and refs come from the P7-1 manifest producer (separate issue); slices 1 and 2 may start in parallel; 3-6 depend on 1 and 2; 7 depends on all.
