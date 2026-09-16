# Phase 3A status: branches, worktrees, receipts and gaps

Written 2026-09-16. Phase 2 closeout is accepted and its integration branch
has been merged into `main`. Phase 3 is active, but is **not launch-complete**:
the fresh full-population bridge succeeds and the fresh render comparison
finds a snapshot-boundary replay failure described below. Producers and real
receipts exist for the earlier candidate; completion requires evidence for the
fresh release, not a relabelled earlier receipt.

Scope authority: [delivery plan](rearchitecture_delivery_plan.md) and
[3A closeout queue](rearchitecture_phase3_parity_launch.md#11-resume-here-bounded-closeout-queue).
This status describes the preview only. Incremental data is Phase 3B; a
preview launch cannot close that gate. Evidence below records earlier runs;
the Sep-16 documentation alignment did not rerun them.

## Branches and worktrees

| Path | Branch | Role |
| --- | --- | --- |
| `/root/investing-plan` | `main` | The working checkout. **Never switch its branch.** Phase 2 closeout and the Phase 3 implementation branch are merged here. |
| `/root/phase3-integration` | historical integration branch | Its accepted implementation was merged into `main` at `b1bf750`. |
| `.claude/worktrees/agent-<id>/` | `worktree-agent-<id>` | Historical task worktrees. New work starts from current local `main` under repository worktree rules; agents never push. |

Three agent worktrees still hold scratch worth keeping until Phase 3 resumes:

- `agent-a92cd9ef8fc683e25` — the preview release root, the preview token and
  `scratch/phase3/build_preview_input.py`. **Do not delete**: the running
  previews serve from it.
- `agent-aaf80f44c8fd7e36e` — bridge/publish scratch and the 44-row subset
  builder.
- `agent-ab8c96d6f8b7432e9` — browser scratch, the built `ui/dist` and
  `scratch/phase3/browser_receipt.py` (locked worktree).

To resume: the implementation is already merged into `main`; do not use the
old integration branch as the baseline. Preserve scratch artifacts needed by
the previews, integrate new task branches under the current repository rules,
and run hygiene with the real environment source when checking a worktree:

    python3 checks/repo_hygiene.py --repo-root <worktree-root> --all --env /root/investing-plan/.env

Phase 2 is already accepted and Phase 3 integration already merged. Publication
permissions follow the current session/repository rules, not this historical note.

## What exists

Historical integration at `65c5a09` contained three merges with real receipts
produced against the real attempt-20 release `relba732fb44d3a88dc2574cc99`
in the ops root `/root/phase2-shadow-ops`:

| Merge | Rows | Producers |
| --- | --- | --- |
| d47a4d1 | L01, L02, L07, L08, L09 | `checks/rearchitecture_phase3_{preview,current_switch,api_pagination,api_auth,startup}.py` |
| d2f1dde | L03, L04, L05, L06, L13 | bridge identity/mapping/value parity, publish idempotency, refresh rollback |
| 65c5a09 | L10, L11, L12, L14 | `checks/rearchitecture_phase3_{browser,quality}.py` |

Receipts, by the commit that produced them:

- `/root/phase3-evidence/77a489f594f024f0a0b13b4c3be280b07264d30d/` — preview
  open parity agree 31/31, current-switch parity agree 7/7, API pagination
  agree 10/10, plus auth, cursor and no-scoring-startup negative controls
  that all correctly differ. Includes a real Playwright screenshot.
- `/root/phase3-evidence/10037175ff0b21fd7c632abed0b2d50190a2c73e/` — bridge
  identity, mapping and value parity agree; publish idempotency agree; a
  refresh rollback receipt; malformed-ref, mapping, corruption and
  publish-failure negative controls all differ as intended.
- `/root/phase3-evidence/2bb5ad7f9ecbbc7781f9a5d912d2d1f9867a85d3/` — browser
  initial load, UI state and full-population parity agree; UI build and
  typecheck agree; the secret-scan negative control fires; plus browser,
  engineering, coverage and performance receipts.

Every acceptance row's tests pass: `tests/test_checks_phase3_gate.py` is 35
green, and the five preview/API producer test files are 47 green.

## The compatibility preview

Serves a real published release read-only, on loopback by default:

    cd /root/investing-plan/.claude/worktrees/agent-a92cd9ef8fc683e25
    V2_DASHBOARD_TOKEN=$(cat scratch/phase3/preview_token.txt) \
      python3 -m engine.v2.dashboard.preview --host 127.0.0.1 --port 8765 \
      --release-root scratch/phase3/release_root_l01 \
      --health-path scratch/phase3/release_root_l01/health.json

Add `--host 0.0.0.0 --allow-non-loopback` to share it outside the container;
that bind is refused by the permission classifier when an agent asks for it,
so a human runs that form. Authentication is a bearer token or an
`operations_token` cookie, and there is no sign-in route: a browser gets 401
until the cookie is set by hand
(`document.cookie = "operations_token=...; path=/"`). The page shell renders
without a token, so "health: unknown / release: unavailable" means
unauthenticated, not broken.

What it renders is the legacy renderer's own board — `render_version 2`, the
same HTML and `app.js` — published through the v2 pipeline from snapshot
`a99e5cac`, as of 2026-09-10, 10 events, 121 scored rows, 6 tickers. A
v2-looking board would mean Phase 2 parity had failed.

## Known gaps

1. **The ticker replay-input rounding defect is fixed.** `render.py` now
   preserves deep replay precision in ticker payloads. A fresh Phase 2 shadow
   release `rela47b76b6843078686cd278db`, from the same pinned snapshot and
   as-of date, passed bridge identity, mapping, and value parity across all
   121 compared fields. The bridge negative controls correctly differ.
2. **Fresh D19 is currently red for a real reason.** Its oracle rebuild agrees
   with the new rendered bundle, but its separate re-score finds 11 mismatches
   in 20 sampled rows, confined to forecast and simulation-derived fields.
   This shows the re-score still reads mutable forecast state outside the
   pinned snapshot. The earlier accepted D19 receipt cannot attest this new
   release. Resolve the snapshot/replay boundary and recapture before claiming
   P3-4 completion.
3. **No Phase 3 coverage ratchet baseline** is committed yet; the coverage
   receipt records this as a known gap (238 tests pass, `engine.v2.serving`
   at 97.8%).
4. **No assembled Phase 3 evidence document.** Each producer's receipts exist,
   but nothing has yet run the evidence builder over all of them, and the
   Phase 3 gate has not been run against an assembled document.
5. **Accepted Phase 2 disposition is not represented by Phase 3 validation.**
   The private closeout record accepts D14/D15 with the strict gate still red;
   `_check_phase2` currently requires clean validation. P3A-C2 owns a narrow,
   tested acceptance handoff that preserves those findings. This does not
   excuse fresh D19 or any new discrepancy.

Smaller notes: the UI ships no "withheld" state, so the stale-badge fixture is
the closest real analogue for that case; `PreviewInput`'s
`legacy_snapshot_object_ref` anchors to the real `snapshot_ref.json` artifact;
and the score/render comparison receipt refs are marked `not_yet_produced`
because no D14/D15 receipt exists for that generation.

## Remaining closeout order

Repair the replay boundary and accepted-disposition handoff, freeze one
candidate, enforce the coverage ratchet, assemble evidence and run the Phase 3
gate with actual strict/readiness status reported separately. Finish the real
update/rollback runbook. Then hand off to 3B; native 4/5 and consumer 6 remain
substantial work, as the delivery plan records. No reliable duration estimate
exists for the replay fix until its mutable read is bounded.
