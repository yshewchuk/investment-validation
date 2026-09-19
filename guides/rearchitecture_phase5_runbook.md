# Rearchitecture Phase 5 — model release preparation, promotion and rollback

Status: scaffold, 2026-09-18 (P5-6). Authority: the
[Phase 5 plan](rearchitecture_phase5_models.md), P5-6 row. This runbook covers
one staged model release: build it, run the acceptance gate over it, promote
it, and roll it back. The gate is **red by design** until the pending members
listed in §5 land.

Rules that apply to every step:

- **Heavy steps are run by the supervisor, under `tools/bounded_run.py`,
  one at a time.** They are marked **[heavy, supervisor]** below. Before one,
  follow AGENTS.md *Sub-agent standing rules → Heavy steps*: no other bounded
  job running, `free -m` available ≥ 5.5 GB. Sub-agents run the synthetic
  tests only.
- Nothing here writes under `data/`. The preparer refuses an `--out` inside
  `data/`; the gate's report is refused inside the repo.
- Output is value-free: ids, statuses, codes, counts, byte sizes. Never paste
  a prediction, threshold or PnL value from any artifact.

Paths used below (pick your own; keep them outside the repo and `data/`):

    REL=/root/p5-6/release-A          # staged release root
    ACC=/root/p5-6/acceptance-A       # private gate report + evidence
    TRAIN=/root/p5-3-runs             # P5-3 training-job outputs
    LIVE=/root/p5-6/deployment-live   # the live deployment store, if any

## 1. Prepare the inputs

1. Registry drift must be zero. **[light]**

       INVESTING_PLAN_ROOT=/root/investing-plan python3 tools/phase5_inventory.py \
           --out /root/p5-6/inventory.json

   Any `release_issues` or real-file issue refuses the preparer (step 2).
2. Payoff calibration artifacts: `engine.v2.models.training.run_training_job`
   writes one `payoff_artifact.json` per calibration fold, but at `926cca9`
   **no real-data entry point builds a calibration dataset** —
   `tools/phase5_training_job.py::build_dataset` exits "not built by this job
   (P5-4 owns calibration fits)" for every `*:calibration` recipe (keys:
   `payoff_line:STR-THRU:calibration`, `payoff_line:STR-RUNUP:calibration`,
   `payoff_surface:STR-RUNUP:calibration`; `--list` shows them). Until that
   entry point exists, a real release has the three payoff members `MISSING`.
   When it lands, run each recipe **[heavy, supervisor]**:

       python3 tools/bounded_run.py --max-rss-gb 5.5 -- python3 -u \
           tools/phase5_training_job.py --recipe payoff_line:STR-THRU:calibration \
           --out $TRAIN/payoff-line-str-thru

   and pass `--training-root $TRAIN` to the preparer, which collects every
   `payoff_artifact.json` and every `recalibration_artifact.json` under it
   (the recalibration map, `engine/v2/models/recalibration_artifact.py`, is
   written by the same training job and has the same dataset gap; the
   STR-RUNUP map stays unconsumed: v2 refuses it as
   `UNSUPPORTED_RECALIBRATION`).
3. P5-4b frozen states (`52ef989`):
   - **Driver residual pools** (`size`, `implied_t1`, `runup_move`): the
     preparer freezes each driver champion's own embedded pool
     (`ModelArtifact.residuals` + `residual_buckets`) unchanged, keyed
     `(role, champion id, fold=None)`, lineage `tier3.panel` with no end
     bound. No flag needed.
   - **Paired residual pool**: no real-data builder calls
     `build_paired_residual_pool_artifact` yet, so pass a prebuilt JSON with
     `--frozen-state`; without one the member is `MISSING`.
   - **Admissible-depth table**: built by the preparer from
     `legacy_n_admissible_table()`.
   `--frozen-state` files are classified by `frozen_release.member_kind`
   (driver pools by their own `role`).

## 2. Assemble the staged release

Plan first. It prints one line per catalog member and writes `$REL/plan.json`.
**[heavy, supervisor]** — `--tier3-snapshot auto` streams a hash of
`panel.parquet`; the model and fold files are read into memory (~90 MB):

    INVESTING_PLAN_ROOT=/root/investing-plan python3 tools/bounded_run.py \
        --max-rss-gb 1.5 -- python3 -u tools/phase5_prepare_release.py \
        --release-id p5-6-2026-09-18a --out $REL \
        --training-root $TRAIN [--frozen-state <pool.json> ...] \
        [--incumbent $LIVE] --plan-only

Then the same command without `--plan-only` stages it. Staging goes through
`engine.v2.models.deployment.stage_release`, so a partial or
feature-order-incompatible model release refuses before anything is written.
Every catalog state is recorded in `$REL/phase5_release.json` as `STAGED`,
`MISSING` or `PENDING`; none is dropped.

`--incumbent` copies the live deployment store (manifests, objects,
`DEPLOYED`, history) into `$REL/deployment` first. Without it there is no
pointer to roll back to (see §4).

Useful flags: `--tier3-snapshot <sha256>` pins the Tier-4 fold snapshot
instead of hashing the panel; `--fold-month YYYYMM` stages one serving month
only; `--tier4-dir` points at a copy of the fold cache.

## 3. Run the acceptance gate

**[heavy, supervisor]** — loads every staged estimator once (a few MB each):

    INVESTING_PLAN_ROOT=/root/investing-plan python3 tools/bounded_run.py \
        --max-rss-gb 2 -- python3 -u checks/phase5_acceptance.py \
        --release-root $REL --artifact-root $ACC \
        [--phase4-corpus <corpus dir>]

Output: `$ACC/evidence.json` and the private `$ACC/phase5_report.md`
(refused inside the repo). Exit code 0 only when the release subjects pass.
Read the result without printing values:

    python3 -c "import json;e=json.load(open('$ACC/evidence.json'));print(e['status'], e['finding_codes'])"

Statuses: `FAIL` (any finding), `RELEASE_PASS` (every release subject passes,
no Phase 4 corpus supplied), `PASS` (release subjects pass and the Phase 4
corpus replays from the staged release).

**Phase 4 replay** (`checks/phase5_phase4_replay.py`). For each corpus pair
with an `input_trace`, Phase 4's own verifier
(`checks.phase4_real._verified_trace_bundle`, via
`checks.phase4_frozen_bridge.prepare_frozen_replay`) rebuilds the request,
native inputs and frozen plan. Each frozen binding's members are then looked
up **by content hash** among the staged model bindings and Tier-4 fold
objects and served from `$REL/deployment/objects/`, not from the corpus's
copy; the captured binding contract (release id, binding ids, role, feature
order, output names, adapter) is kept, since those ids enter stage inputs. The
pair is scored with `application.score_frozen` under both no-fit guards and
`checks.phase4_real._verify_runtime_execution` compares every runtime stage
receipt and final identity with the captured ones. Pairs with no trace, or a
trace with no frozen binding, never read the release: they are counted
(`untraced`, `not_frozen`), not findings — trace coverage belongs to the
Phase 4 gate (my judgement call). Dispositions are in
`evidence.json → phase4.dispositions`, and `phase4.members_exercised` lists
which staged members the corpus actually read.

What the gate checks, and its finding codes:

| Subject | Check | Codes |
|---|---|---|
| layout | `phase5_release.json` matches its own `manifest_hash`; staged model manifest's `release_hash` recomputes | `P5_RELEASE_LAYOUT`, `P5_MODEL_RELEASE_INVALID` |
| members | each of the 7 champion bindings and every catalog state is staged; object bytes hash to the recorded hash; typed loaders accept payoff, recalibration and frozen-state members; each loaded state is the kind/role/strategy its catalog row names | `P5_MEMBER_MISSING`, `P5_MEMBER_PENDING`, `P5_MEMBER_UNKNOWN`, `P5_MEMBER_OBJECT_ABSENT`, `P5_MEMBER_HASH_MISMATCH`, `P5_MEMBER_UNLOADABLE`, `P5_MEMBER_IDENTITY` |
| lineage | every staged frozen state declares lineage; `upstream` names another staged state (`<member_id>/<object name>`); no cycle (`lineage.propagate_corrections` with no changesets) | `P5_LINEAGE_INVALID` |
| consumers | each v2 score consumer resolves its members from the release, and refuses `MODEL_NOT_READY` with the member taken away | `P5_CONSUMER_UNRESOLVED`, `P5_CONSUMER_NO_REFUSAL`, `P5_CONSUMER_ERROR`, `P5_CONSUMER_PENDING`, `P5_CONSUMER_BLOCKED` |
| no fitting | the consumer pass runs under both no-fit guards; any `forbid_fitting` call (even a swallowed one), any `joblib.dump`, and any file change under `data/models` or the release root is a finding; a planted fit and a planted write must be detected | `P5_RUNTIME_FIT`, `P5_MODEL_CACHE_WRITE`, `P5_GUARD_CONTROL_FAILED` |
| rollback | on a scratch copy of the pointer state: promote, roll back, pointer resolves the incumbent again, manifests and earlier history byte-identical, exactly two history entries appended | `P5_ROLLBACK_NO_INCUMBENT`, `P5_ROLLBACK_CANDIDATE_ALREADY_DEPLOYED`, `P5_ROLLBACK_NOT_EXACT`, `P5_PROMOTE_REFUSED` |
| phase 4 | every traced frozen pair verifies, binds only staged models, and replays from the staged bytes with captured receipts under both no-fit guards; at least one pair replays | `P5_PHASE4_UNVERIFIED`, `P5_PHASE4_MEMBER_ABSENT`, `P5_PHASE4_MISMATCH`, `P5_PHASE4_ERROR`, `P5_PHASE4_EMPTY` (+ `P5_RUNTIME_FIT`/`P5_MODEL_CACHE_WRITE` with subject `phase4_replay`) |
| report | the written report carries every member, consumer, control and finding row | `P5_REPORT_INCOMPLETE` |

The gate never moves `$REL`'s own pointer: the round trip runs on a copy.

## 4. Promote and roll back

Only after the gate is green, and only by the supervisor with user approval.
The live store is whatever directory production resolves its release from
(`$LIVE`). Promotion copies the staged release in and swaps one pointer:

    python3 - <<'EOF'
    import shutil
    from pathlib import Path
    from engine.v2.models import deployment
    src, live, rid = Path("/root/p5-6/release-A/deployment"), Path("/root/p5-6/deployment-live"), "p5-6-2026-09-18a"
    for sub in ("objects", f"releases/{rid}"):
        shutil.copytree(src / sub, live / sub, dirs_exist_ok=True)
    shutil.copy2(src.parent / "phase5_release.json", live / "releases" / rid / "phase5_release.json")
    print(deployment.promote(live, rid))
    EOF

The state members travel in `phase5_release.json` beside the staged model
manifest; the deployment module never reads it, and no production consumer
resolves states from it yet (that is what the PENDING consumer probes track).

Rollback returns the pointer to the release the current one was promoted
from. It never deletes a staged release, so scores recorded against either
release stay replayable (`deployment.resolve_release`):

    python3 -c "from engine.v2.models import deployment; print(deployment.rollback('/root/p5-6/deployment-live'))"

Check afterwards: `deployment.current_pointer(live).release_id` is the
incumbent id and `deployment.pointer_history(live)` grew by one entry.

**First deployment.** With no incumbent pointer there is nothing to roll back
to, and the gate reports `P5_ROLLBACK_NO_INCUMBENT`. `--first-deployment`
accepts instead that promotion works and a rollback request is refused
(`NoPriorRelease`) without moving the pointer. The legacy path stays the
fallback until cutover; record that decision in the report handoff.

## 5. Why the gate is red today

At `52ef989` + P5-6 round 2, the full catalog run shows, for a release built
from real inputs:

- `P5_MEMBER_MISSING` for the three payoff members and
  `recalibration_map:STR-THRU` (no real-data calibration dataset entry point,
  §1.2), which blocks the payoff-line, payoff-surface, recalibration and
  driver-pool consumers (`P5_CONSUMER_BLOCKED`: the driver-pool probe scores
  through the staged payoff artifact of its strategy);
- `P5_MEMBER_MISSING` for `paired_residual_pool` unless `--frozen-state`
  supplies one (no real-data builder), blocking
  `simulation.paired_residual_pool`;
- `P5_MEMBER_PENDING` for the trailing `pnl_sim` cutoff and the board analog
  matcher (no frozen artifact type yet);
- `P5_CONSUMER_PENDING` for the consumers without a probe: admissible table
  (no v2 stage reads it), trailing cutoff, chooser analog pool, board analog
  matcher and Tier-4 serving folds (no v2 consumer resolves them from a
  release);
- `P5_ROLLBACK_NO_INCUMBENT` unless `--incumbent` or `--first-deployment`;
- with `--phase4-corpus`: whatever the replay finds. Expect
  `P5_PHASE4_MEMBER_ABSENT` for any pair captured against a model whose bytes
  are not the staged champion (e.g. a Tier-4 fold of another month), and
  `P5_PHASE4_EMPTY` if no pair in the corpus carries a frozen binding.

When a member's artifact type merges, add its builder to
`tools/phase5_prepare_release.py`, its probe to
`checks.phase5_consumers.CONSUMERS`, and a synthetic test; the catalog in
`checks/phase5_release.py` already names it.

## 6. Tests

Synthetic only, no real data:

    timeout 600 python3 -m pytest -q -n 2 tests/test_checks_phase5_acceptance.py \
        tests/test_checks_phase5_phase4_replay.py
