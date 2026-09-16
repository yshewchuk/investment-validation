# Phase 2 closeout: status, blockers and what remains

## Current closeout execution (2026-09-16)

The handoff below is preserved as the starting record. Follow the final
execution record at `reports/phase2_closeout/FINAL.md` for the frozen candidate,
test results, receipt paths, gate findings and any remaining sign-off decision.
That private record is written after validation so recording the results does
not change the source hash to which the receipts bind.

The closeout fixes address immutable export identity, coordinator failure
diagnostics and materialization dependency evidence. Export generations follow
their captured contents while existing generations remain immutable; evidence
planning uses the existing whole-table materialization policy without relaxing
public query limits. Corpus receipt dependencies are packaged with the receipt.
The ordinary and seeded Tier-1 replays both pass all 20 cases; the seeded run
also detects all four planted defects.

D14 remains the previously accepted, historical explained difference. D15
requires a fresh comparison and an explicit disposition if the known legacy
price-source differences remain. Neither receipt is changed to claim agreement.
The final run also supplies the evidence artifact root explicitly to gate2.

Public pushes are pending explicit approval after automatic approval review
rejected a direct push to `origin/main`. Local validation can proceed.

## Original handoff

Written 2026-09-16 against `main` at 647ef5f. Companion to
`rearchitecture_phase2_data_access.md` (the D-item definitions) and
`rearchitecture_tech_debt.md` (post-Phase-6 nice-to-haves).

Ops root: `/root/phase2-shadow-ops`. Evidence for the latest run:
`/root/phase2-shadow-ops/evidence_647ef5f/`. Narrative log of every run:
`/root/phase2-shadow-ops/RUNLOG.md`.

## How a closeout run is driven

One script drives the whole sequence and is re-runnable per commit:

    SKIP_D14=1 scratchpad/closeout.sh <sha>

It builds a frozen worktree at `<sha>`, captures inputs, runs the
snapshot-mode and legacy-mode nightlies, then D15, D19, D11, D16, the
coverage ratchets, gates 0/1/2 and the evidence build. `SKIP_D14=1` skips
the corpus-parity rerun, which is closed (see below). The gates step runs
from the main checkout and refuses unless `HEAD` equals `<sha>` and the
tracked tree is clean.

## Where we are

The 647ef5f run took 47 minutes on a quiet box.

| Step | Result |
| --- | --- |
| capture | ok, 13 s |
| snapshot nightly | 11/14 jobs ok, 932 s. `legacy_model_evidence` now succeeds (peak 4861 MB against a 5 GiB reservation) |
| legacy nightly | same shape, 691 s |
| D15 score parity | `differ`, 847 findings over 117/117 rows — explained, see accepted issues |
| D19 render parity | **not produced** — render is blocked behind `ledger_export` |
| D11 fault matrix | pass |
| D16 rollback + roll-forward | both pass, receipts written |
| Phase 1 coverage + gate1 | green |
| Phase 2 coverage | green (`ok: true`, 0 findings) |
| gate0 | **red** — tier-1 replay receipts are stale |
| evidence build | **crashes** — row-cap mismatch |
| gate2 | **red**, 10 findings in 395 s: `PREREQUISITE_FAILED: phase0` plus nine `MISSING_EVIDENCE` (D02, D05, D09, D10, D11, D14, D15, D16, D19), every one `reason=evidence_manifest` — a single cause, the crashed evidence build, not nine separate gaps |

## Accepted issues: real defects in legacy, not in v2

Both are legacy-side, both are understood, and neither is being fixed
before cutover because changing legacy output invalidates every capture
and parity baseline the closeout evidence rests on.

1. **Stale price archive.** `add_runup_features` (`engine/dashboard/panel.py`,
   px-first read near line 533, Tier-1 fallback 444-475) reads a RAW_YF
   archive that ends 2026-08-27, while v2's `price_history` uses the latest
   retrieval. This explains **all** 336 D14 corpus findings (124 legacy
   finality drift + 212 price) and **all** 847 D15 findings: 9 driver rows
   differ on `dist_ema`/`abs_dist_ema`/`dist_high` and cascade through
   forecasts, structure widths and bootstrap confidence intervals.
   Magnitudes are not trivial — median relative difference 0.78%, p90 7.4%,
   30 rows above 10%, max 116% on a near-zero denominator — but
   `chosen_strategy`, `gate_pass` and `flags` are **identical on all 117
   rows**, so no trading decision changes. D14 is closed as explained by
   user decision; the Phase 2 gate stays red on D14 by design. D15 needs the
   same disposition.
2. **Per-ticker file rounding.** `engine/dashboard/render.py:1162` and `:1434`
   write `data/tickers/{T}.json` through the naive `_clean()`, which lacks
   the `REPLAY_INPUT_FIELDS` exemption that `board.json` received in 6b9d5cf.
   `structure_params` therefore arrives rounded to 6 dp where the score keeps
   full precision, so 77 of 121 rows fail the Phase 3 bridge value check.
   The v2 bridge reads the correct file — the per-ticker record is the
   designed richer source — so there is no v2-side fix. Phase 3 rows L07-L09
   and L05 run on an honest 44-row subset until this is fixed.

## Blockers to closing Phase 2

1. **`ledger_export` refuses, blocking render, selfcheck, publication and
   therefore D19.** The export generation directory from an earlier run still
   exists under the same release key holding 784 prediction rows, while the
   catalog now holds 3,278. Because a rerun commits no new decisions, the
   release key never changes, so `export_generation` takes its verify branch
   and `_verify_complete` raises `ValueError("export generation differs from
   catalog")` (`engine/v2/ledger/export.py:75-79`). Two fixes are needed:
   the staleness itself, and the fact that a plain `ValueError` is swallowed
   by `supervisor.py:576` into "worker output failed validation", which hides
   every non-`OpsError` cause and cost roughly an hour of diagnosis.
2. **Evidence build crashes on a real nightly.** `_dependency_plan_refs`
   (`checks/rearchitecture_phase2_evidence_build.py:97`) replays the recorded
   materialization queries through `Repository.explain_dependencies`, which
   enforces the table contract cap. The real `daily_market` query asks for
   9,123,661 rows against a 2,000,000 cap and is refused with
   `QUERY_NOT_BOUNDED`, even though the materializer accepted that same query
   during the run.
3. **gate0 is red on stale receipts**, which also fails gate2's phase0 prerequisite. `tier1_real_replay` and
   `tier1_seeded_controls` refuse with "code changed since the replay ran
   (code_hash differs)". A fresh tier-1 replay is needed
   (`--max-rss-gb 6.5`), not a code change.

## What remains

| Bucket | Estimate | Kind |
| --- | --- | --- |
| Fix the export-generation staleness and stop swallowing non-`OpsError` causes | 1-2 h | code |
| Fix the evidence builder's row-cap mismatch | 0.5-1 h | code |
| Rerun both nightlies plus D15, D19, D11, D16 | ~35 min | machine |
| Fresh tier-1 replay to clear gate0 | 30-60 min | machine |
| Gates and evidence build | ~20 min | machine |
| D15 disposition, as D14 was decided | minutes | decision |
| Assemble and report the final evidence | ~15 min | supervisor |

Roughly 3-4.5 hours, about 1.5 of it machine time.

## Resolved along the way, for context

- `legacy_model_evidence` used to fail at a 4 GiB reservation. Fixed by
  per-champion subprocess isolation, removing a duplicate `daily_market`
  load in `_dataset_for`'s gate branch, streaming trades per year and
  trimming freed pages; reservation now 5 GiB (policy v8, sized against a
  live `headroom_bytes` sample, never `capacity_bytes`).
- A rebuild is deterministic: repeated forced rebuilds are byte-identical.
  Differences against an older cached file are newer training data at
  unchanged champion fingerprints.
- Three tests asserted the repo root has no `.env`, so they passed in
  worktrees and failed in the real checkout, taking the coverage
  measurement down with them. They now build their own repo root.
- Both coverage ratchets are green in the real checkout, closed by covering
  12 lines with real tests rather than lowering a baseline.
