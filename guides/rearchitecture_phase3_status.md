# Phase 3 status: branches, worktrees, receipts and gaps

Written 2026-09-16. Phase 3 is **paused by decision** until Phase 2 closes
(see `rearchitecture_phase2_closeout.md`). Producers and real receipts exist
for every acceptance row; what remains is assembly and three known gaps.

## Branches and worktrees

| Path | Branch | Role |
| --- | --- | --- |
| `/root/investing-plan` | `main` | The working checkout. **Never switch its branch.** Phase 3 commits stay off `main` until the Phase 2 final closeout. |
| `/root/phase3-integration` | `phase3-integration` | Where all Phase 3 work merges. Branched from `main` at 96e1de0, pushed to origin. Run merges, tests and pushes for Phase 3 from here. |
| `.claude/worktrees/agent-<id>/` | `worktree-agent-<id>` | Per-agent worktrees. Agents rebase onto `phase3-integration`, commit there, and never push; the supervisor merges. |

Three agent worktrees still hold scratch worth keeping until Phase 3 resumes:

- `agent-a92cd9ef8fc683e25` — the preview release root, the preview token and
  `scratch/phase3/build_preview_input.py`. **Do not delete**: the running
  previews serve from it.
- `agent-aaf80f44c8fd7e36e` — bridge/publish scratch and the 44-row subset
  builder.
- `agent-ab8c96d6f8b7432e9` — browser scratch, the built `ui/dist` and
  `scratch/phase3/browser_receipt.py` (locked worktree).

To resume: merge into `phase3-integration` from a worktree branch, then
`git push origin phase3-integration`. Run hygiene from the main checkout,
because a worktree has no `.env` and the secret check silently skips:

    python3 checks/repo_hygiene.py --repo-root /root/phase3-integration --all --env /root/investing-plan/.env

After Phase 2 closes, `phase3-integration` merges into `main`.

## What exists

`phase3-integration` is at 65c5a09. Three merges, each with real receipts
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

1. **Full population is blocked by a legacy defect.** `render.py:1162`/`:1434`
   round `structure_params` in `data/tickers/{T}.json`, so 77 of 121 rows fail
   bridge value parity and `build_candidate` refuses that population. L05's
   agree side, and L07-L09, therefore run on a real, honestly-labelled 44-row
   subset (CAL-P, CND-P, STR-RUNUP, STR-THRU — the rows with null or empty
   `structure_params`), and the real 77-row mismatch is used as L05's negative
   control. Rerun those rows on the full 121 once the legacy fix lands; that
   fix also forces a Phase 2 recapture, which is why it is deferred.
2. **No Phase 3 coverage ratchet baseline** is committed yet; the coverage
   receipt records this as a known gap (238 tests pass, `engine.v2.serving`
   at 97.8%).
3. **No assembled Phase 3 evidence document.** Each producer's receipts exist,
   but nothing has yet run the evidence builder over all of them, and the
   Phase 3 gate has not been run against an assembled document.

Smaller notes: the UI ships no "withheld" state, so the stale-badge fixture is
the closest real analogue for that case; `PreviewInput`'s
`legacy_snapshot_object_ref` anchors to the real `snapshot_ref.json` artifact;
and the score/render comparison receipt refs are marked `not_yet_produced`
because no D14/D15 receipt exists for that generation.

## Rough remaining effort

Assembling the evidence document, committing a coverage baseline and getting
the Phase 3 gate green is on the order of 3-5 hours of agent work if the
subset receipts are accepted as they stand. Fixing the legacy rounding defect
first, to get full-population receipts, adds a Phase 2 recapture and rerun on
top of that.
