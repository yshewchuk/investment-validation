# Native board rebuild from the local Tier-1 cache

## Recommendation (amendment option A)

Use a runbook plus the existing commands, and add exactly one missing standard command:
`ops computed-moves capture` — the proposed CLI exposes/reuses the existing native `computed_moves.v3`
producer (`computed_moves_store.py`) but also requires a commit-path change to propagate the reference
pins, a requirement that is not implemented. Do not add a `--from-tier` sequencer, migration code, or a
readiness CLI: the runbook below already makes the order, head precondition, and serial heavy admission
explicit. No step has been shown unsafe or error-prone enough to justify sequencing code; the runbook is sufficient for supervised use; the remaining cross-step decisions fit a manual checklist. `python3 -m engine.data.pulls.computed_moves` is NOT that capture: it is a Tier-1 raw history pull (yfinance realized moves feeding the panel's event history, `engine/data/pulls/computed_moves.py:2-14`), never a native `computed_moves.v3` commit.

## Verified evidence and corrections (investigation report)

- Gap 3: `import_snapshot.plan_import` enumerates exactly the six `TIER2_DATASETS`
  (`legacy_mapping.py:107-109`), `feature_panel` + `tier4_forecasts` (`import_snapshot.py:175-183`),
  and reference inputs (`:185-189`) — eight legacy datasets, never the native
  `price_history.v3`/`computed_moves.v3`; and `nightly_trigger._ensure_shadow_snapshot`
  (`nightly_trigger.py:1011-1085`) reads the head, plans, submits, drives the import (`:1074-1083`) before nightly planning — no native capture (report step 7, gap 3).
- 2026-10-06 supervisor-provided fresh-root measurement: `snapshot plan-import` → `submit` → `serve --once` took 24.0 min under `bounded_run --heavy --cores 8 --max-rss-gb 6.5` — the `--max-rss-gb 6.5` flag was the historical RSS cap, the measured peak RSS was <1 GB, and the separately measured ~5.9 GiB was free headroom admission; neither number is a recommended override — and the run produced exactly eight legacy tables with no `price_history` or `computed_moves`; same-box `ledger import-history --dry-run` counted 2,374 predictions, 14,634 outcomes, and 7 divergent prediction rows — confirms gap 3 on a real store.
- Gap 4: `price_history_store._commit_generation` carries every other table's manifest forward
  (`price_history_store.py:593-604`) and copies the base receipt's reference inputs and lineage
  into its own receipt in the commit transaction (`:623-626`); `computed_moves_store` does the
  manifest carry (`computed_moves_store.py:438-452`) but its `record_references` inserts only
  capture audit rows (`:465`) — no reference pins. **Correction:** price-history copies pins
  from the head's newest committed receipt (`:577-578`; `reference_catalog.py:117-130`); under
  the current `reference_catalog` code, a capture whose base receipt pinned no references
  cannot complete: resolution refuses with typed `SNAPSHOT_NOT_READY` (`:142-144`), a verified consequence
  of that code, not inference. **Design requirement (not implemented yet):** the future computed-moves
  capture must copy the base receipt's reference pins into its new receipt (mirroring `:623-626`), else
  the subsequent price-history capture cannot complete; ordering stands: computed moves first, price
  history LAST (report step 7, gap 4).
- CLI: `engine/v2/ops/cli.py` defines `snapshot plan-import`/`submit` (`:127-134`),
  `price-history capture` (`:106-118`), `ledger import-history` (`:65-77`) — no computed-moves capture exists (report steps 8–10).
- Tier starts (report steps 3–4): `--table` is repeatable over `TABLE_ORDER` (`engine/data/rebuild.py:41-50,513-517`).
  Under this design, the selected tier is assumed valid; tier 2 — including `trades` — therefore remains
  untouched; only the panel (Tier 3) and Tier 4 rebuild. `engine.build_trades` reruns only after the full
  Tier 2–4 rebuild from tier 1: it replaces `trades` with only legacy-provenance rows (`n_trades.py:125,154-163`),
  dropping `engine.replay` rows, so it must rerun (`build_trades.py:16-17`); runs excluding `trades` keep replay rows.

## Preconditions and operator rules

- The Tier-1 raw cache is complete and usable; the implemented steps of this procedure never call providers or repair raw inputs. The future computed-moves capture (step 2, not implemented yet) must be held to the same cache-only rule — read only the frozen `SOURCE_ROOT` and refuse a cache miss before any provider/network I/O — while the current production binding instead uses yfinance for uncached units. One frozen legacy checkout is the absolute `SOURCE_ROOT` for the whole run.
- Only the supervisor runs heavy steps; agents never run them. The bounded example here is the legacy rebuild: `INVESTING_PLAN_ROOT="$SOURCE_ROOT" python3 tools/bounded_run.py --heavy --cores 8 -- python3 -m engine.data.rebuild` — heavy steps serially, one-heavy-job admission built into `--heavy`; never lower a limit to start sooner. The measured snapshot import job (~5.9 GiB free headroom) runs through the bounded `serve --once` invocation in step 1.
- Before `snapshot plan-import`, read the shadow head's (`snapshot_id`, `generation`) from `data_snapshot_heads` and pass both, or
  for an absent head follow step 1's no-head alternative; a stale head must stop the run, and never plan while the trades replay runs.

## Ordered procedure

Set `SOURCE_ROOT` to one absolute path — the frozen legacy checkout — plus `OPS_ROOT`, `AS_OF` (the legacy tree's selected session), and
`MODEL_RELEASE_ROOT`; make `OPS_ROOT` and `MODEL_RELEASE_ROOT` absolute too, so they remain valid after the `cd "$SOURCE_ROOT"` before the
tier table changes the directory. Run the Tier 2–4 rebuild commands with the current directory set to that checkout, binding their rebuild output
to the same tree later imported from `SOURCE_ROOT` (operator requirement, not locally verified). Complete the start-tier legacy work first:

| Start tier | Legacy rebuild work |
|---|---|
| 1 | `INVESTING_PLAN_ROOT="$SOURCE_ROOT" python3 tools/bounded_run.py --heavy --cores 8 -- python3 -m engine.data.rebuild`, then `INVESTING_PLAN_ROOT="$SOURCE_ROOT" python3 tools/bounded_run.py --heavy --cores 8 -- python3 -m engine.build_trades` |
| 2 | `INVESTING_PLAN_ROOT="$SOURCE_ROOT" python3 tools/bounded_run.py --heavy --cores 8 -- python3 -m engine.data.rebuild --table panel --table tier4` |
| 3 | `INVESTING_PLAN_ROOT="$SOURCE_ROOT" python3 tools/bounded_run.py --heavy --cores 8 -- python3 -m engine.data.rebuild --table tier4` |
| 4 | none — go straight to the import |

1. Open the operations root and import the rebuilt tree — before planning, read the catalog head state and set `HEAD_SNAPSHOT_ID` and `HEAD_GENERATION` from the
   shadow head row; with no head row, leave `HEAD_SNAPSHOT_ID` unset — the block passes the stored ID and generation for an existing head, or omits the snapshot-ID
   option entirely (never an empty string) with generation 0 for an absent head, and stops on any nonzero step while preserving its status. Wait for the trades replay
   to exit first; it parses `plan_ref` from the `plan-import` JSON into `PLAN_REF`, then defines the plan/source-bound stable key `IMPORT_KEY="rebuild-${PLAN_REF}-import"` before submitting:

   ```bash
   set -euo pipefail
   python3 -m engine.v2.ops --root "$OPS_ROOT" init
   if [ -n "${HEAD_SNAPSHOT_ID-}" ]; then
     PLAN_RESULT=$(python3 -m engine.v2.ops --root "$OPS_ROOT" snapshot plan-import --source-root "$SOURCE_ROOT" --scope shadow --expected-head-snapshot-id "$HEAD_SNAPSHOT_ID" --expected-head-generation "$HEAD_GENERATION")
   else
     PLAN_RESULT=$(python3 -m engine.v2.ops --root "$OPS_ROOT" snapshot plan-import --source-root "$SOURCE_ROOT" --scope shadow --expected-head-generation 0)
   fi
   PLAN_REF=$(printf '%s' "$PLAN_RESULT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["plan_ref"])')
   IMPORT_KEY="rebuild-${PLAN_REF}-import"
   python3 -m engine.v2.ops --root "$OPS_ROOT" snapshot submit "$PLAN_REF" --idempotency-key "$IMPORT_KEY"
   python3 tools/bounded_run.py --heavy --cores 8 -- python3 -m engine.v2.ops --root "$OPS_ROOT" serve --once --store-root "$SOURCE_ROOT"
   ```

2. Run the new `ops computed-moves capture` (not implemented yet — slice 2). Design requirement, not implemented in the current
   producer/CLI: under the preconditions' cache-only rule, drive the native producer, commit a new shadow generation, and copy
   the base receipt's reference pins into its new receipt (per the design requirement above).
   Proposed future CLI syntax, not an existing command: `python3 -m engine.v2.ops --root "$OPS_ROOT" computed-moves capture --source-root "$SOURCE_ROOT" --as-of "$AS_OF" --scope shadow`.

3. Capture price history LAST — the final generation then carries both native tables and pins:
   `python3 -m engine.v2.ops --root "$OPS_ROOT" price-history capture --source-root "$SOURCE_ROOT" --scope shadow`.

4. Import decision history through the rebuilt as-of session: `python3 -m engine.v2.ops --root "$OPS_ROOT" ledger import-history --source-root "$SOURCE_ROOT" --through "$AS_OF"`.

5. Complete the Phase 5 staged model release under `MODEL_RELEASE_ROOT` (inventory, calibration, training, preparation, acceptance, staging) — a separate heavy workflow (report step 12).

6. Verify every readiness item below before starting the nightly trigger; the population document
   is placed by hand at `reports/phase6/nightly_trigger/expected_population.json` (`nightly_trigger.py:652`) (report step 13).

## Proposed requirements R1–R6 (not current end-to-end guarantees)

A runbook, not one transaction: only snapshot-producing steps commit an atomic snapshot generation; ledger and model steps have
their own operation boundaries. There is NO whole-run rollback — earlier commits remain; never serve or score from an incomplete
head. Verified for existing commands only: snapshot import is fenced by the expected head, a fresh-root import was observed to
produce only the eight legacy tables, and price-history capture is the existing reference-pin copy-forward path — those observations do not prove the proposed R1–R6 contract, and computed-moves pin copy-forward remains proposed.

- R1: validate the source sits at `AS_OF` and required legacy inputs exist before submitting.
- R2: run steps serially; for a snapshot-producing step, failure before its commit leaves the prior head current.
  If any legacy rebuild exits nonzero or is interrupted, stop before the import and before continuing the tier table; treat its generated outputs and metadata as potentially inconsistent — this runbook has no verified safe in-place retry or restore procedure — and resume only after the supervisor independently establishes `SOURCE_ROOT` is complete and current to `AS_OF`.
- R3: plans carry the expected head snapshot ID and generation; a mismatch must meet a typed
  stale-head refusal — never bypass the head fence — then reread the latest pair and replan.
- R4: resubmit an interrupted import's saved plan under the SAME idempotency key only while its job is still active — same-key resubmission returns the existing job
  unchanged and never resumes a terminally failed job (`submission.py:298-307`); on a terminal failure, stop pending an
  established recovery procedure. A changed source or head requires a new plan and key. Operator instruction (requirement, not verified fact): rerun price-history capture and computed-moves capture for the same `as_of` after a failure.
- R5: a command success is step completion, never readiness; the checklist is manual sign-off, not a software refusal.
- R6: never start nightly work from an incomplete head; repair by rerunning the producer owning the missing item, then recheck everything.

## Readiness checklist (manual, against the operations catalog and release directory)

- [ ] Current `shadow` head holds all eight legacy tables, `price_history.v3`, `computed_moves.v3`.
- [ ] Newest head receipt carries the required reference pins; the price-history capture receipt has a `data_receipt_lineage` entry.
- [ ] Catalog `decisions` holds the imported prediction/outcome history the rebuilt tree needs.
- [ ] Legacy selected session is exactly `AS_OF` — wait for currency, never import later/partial.
- [ ] A staged model release exists under `MODEL_RELEASE_ROOT` and passed Phase 5 acceptance.
- [ ] The expected population document exists and matches the intended nightly population.
- [ ] Operations root, legacy store root, and timer target the same intended catalogs/tree.

## Retained derived inputs (supervisor recommendation; report step 5 / gap 2)

Retain, back up to the private mirror, restore before import, and verify by hash and Tier-4 fold
on the newest receipt the derived reference inputs that have no standard producer; the rebuild does not recreate them:
`data/features/pnl_sim_history.parquet` and `data/features/recalibration_pairs.parquet`; the chooser candidate pool built
from the EXP-137 `results/candidates.parquet`; champion artifacts from `train_all` plus `structures.json`; Tier-4 serving
caches for the imported panel hash — stored with a manifest of paths, hashes, and fold.
The named producer alternatives: pnl_sim history builder; recalibration-pairs CLI; chooser pool
builder from retained EXP-137 results; verified champion reproduction; Tier-4 cache builder for
the new panel hash — costing more to build and validate; listed for review, not chosen here.

## Rebuild invalidations and follow-ups (report step 15 / gap 6)

Regenerate snapshot IDs and preregistration hashes in affected experiment `spec.yaml` files by
hand; recapture the v17 corpus with `tools/capture_tier0_corpus.py` and rerun corpus parity; rerun
the D14, D15, and D19 evidence with their standard checks; old catalog attempts and receipts are
not current evidence. Follow-up issues: spec-hash rebinding, population-document automation, the
retention-vs-producers decision, a measured rebuild duration.

## Proposed slices

| Slice | Scope | Estimate |
|---|---|---:|
| 1 | This design guide plus the two one-line pointer updates. | 148 guide lines; 150 added doc lines including pointers |
| 2 | `ops computed-moves capture`: CLI + producer commit-path pin propagation in the native producer, committing a generation that copies reference pins forward. | est. ~70 code / ~35 tests / ~10 docs lines |
| — | No sequencer or readiness CLI proposed; readiness stays this checklist. | 0 |
| Later | Automate invalidation steps or retained-input producers, each after its own scope decision. | separate PRs |
