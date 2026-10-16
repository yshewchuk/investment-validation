# Native decisions in a scratch catalog (P6-3, #740)

Design, not yet implemented: everything below is target behaviour; "today" marks what the code does now. Source: `guides/rearchitecture_phase6_consumer_parity.md` (P6-3 row and the `decisions-*` capability rows); `guides/authority_fence_switch_rollback.md` (P7-3, #739/#749). No code changes in this PR.

**Decision (owner, 2026-10-14, #740, Option C; not reopened here).** Native decisions are written to a **separate scratch catalog**. They never enter the official ledger, the production ops catalog or the legacy ledger until cutover authority moves. The scratch catalog is also the P7-3 rehearsal target where it can be.

**Standard tools.** None applies: the catalog is the repo's own SQLite schema (`engine/v2/ledger/decisions.py`), and no standard format covers append-only decision facts. Reused as is: `decisions.insert`/`record_divergence`, `export_generation`, `population_key`/`population_difference` (`engine/v2/foundation/score_population.py`). Reused by shape only: `legacy_adapter._action_decisions` (it loads legacy action files, finality and a decision plan and writes `decisions.json`, so it cannot take native score records); the native candidate builder (slice 2) reuses its row serialization shape and pinned-timestamp handling. Reused with adaptation (slice 6, not assumed here): `backup.py` (needs its outbox infrastructure in the scratch root) and `tools/v2_restore_drill.py` (needs pinned score artifacts and a complete ops catalog, so a drill restores a copy into a scratch ops root). `import_lines` is **not** used: it inserts under owner `catalog`, and scratch never imports history.

## Today

- `legacy_adapter._action_decisions` builds rows with legacy `build_prediction_rows` and `decision_commit.commit_decisions` writes them to the **production** ops catalog under purpose `shadow` (`decisions-predictions`, `decisions-settlement`, `decisions-ledger-export` are `compatibility-adapter`).
- `decisions.insert` checks `decision_authority.owner == owner argument` (default `catalog`). That check is the only fence on inserts.
- The only consumer of `native_score_batch` output is `native_parity_report.py`. No night exercises a decision built from native scores.
- Blocked for **code** (not design) on the #681 driver_prediction class (#651 S4 measures it). Decisions built earlier would drift for a known defect, not an accepted difference.

## Contract

### Identity and location

1. **One scratch catalog per qualification line**, a SQLite file at `<native_root>/<line_id>/catalog.sqlite`, where `<native_root>` is a configured directory **outside** `<ops_root>` (default: a sibling `native_scratch/` of the ops root), with the unchanged `decisions` schema (`SCHEMA`, `SCHEMA_V2`). `line_id` is minted once by `ops native-decisions init` (UUID) and recorded in an immutable `scratch_meta` row (table added by slice 1) with `line_id`, creation time, the pinned snapshot id, `schema_version` and the **origin tag** `native-scratch`. These identity fields are immutable; lifecycle state (`closed_at`, `retired_at`) is a separate append-only `scratch_lifecycle` table.
2. **Never the official path.** The resolver refuses (`SCRATCH_PATH_OFFICIAL`) any path that is, or is a symlink or hardlink to, the official catalog file or its `-wal`/`-shm` companions, or that lies inside `<ops_root>`. A file whose `scratch_meta` is absent or whose origin is not `native-scratch` is refused with the distinct `SCRATCH_META_INVALID`. The official catalog never contains a `scratch_meta` table; the scratch catalog is never opened by a process holding the official fence.
3. **Authority owner is `scratch`**, a new catalog owner value alongside `drill`, `catalog`, `v2`, `none`. `decisions.insert(..., owner="scratch")` succeeds only on a catalog whose `decision_authority` row says `scratch`. Because the official catalog's owner is never `scratch`, a native writer pointed at the wrong file refuses (`DecisionConflict`) rather than writing. `scratch` is rejected by `set_authority` on the official catalog (slice 1 adds the guard on `scratch_meta` absence). It is not an official writer and is outside the P7-3 invariant count.
4. **Ids.** `logical_key` uses the existing `content_hash([purpose, event_id, strategy, deployment, clock, session])` derivation with `purpose = "native_scratch"`; it is **generation-scoped** (deployment and clock change across generations). The stable cross-generation identity is `decision_id`, which is `"native_scratch:" + row_id` (the row id names the occurrence); it locates the original decision for divergence evidence. It is never the legacy `"prediction:" + row_id`, and settlement references that same id. A scratch row and a legacy-origin row for the same event therefore have **different** keys and ids by construction; they are compared by `population_key`, never merged. No row is shared or copied between catalogs, and `generation_ref` is the native generation id.
5. **Lifetime.** While a line is live, its catalog, reconcile artifacts and exports are **durable qualification evidence**, not transient scratch: the workspace disposal rule does not apply. They are retained through the 10-session rollback window after cutover and removed only by an explicit `retire` after it (recorded in `scratch_lifecycle`). `reset` (operator, after a defect, only while no night of the line has counted toward qualification and the line is not closed) refuses with `RETENTION_ACTIVE` once the line has a counted night, `closed_at` or is inside the rollback window, so evidence cannot be deleted early; otherwise it deletes that line's catalog and exports, keeps its reconcile artifacts under a retired-line name, and never touches the official catalog. Transient files (temp exports, partial writes) are ordinary scratch, removed on success. No contents are migrated (rebuild-over-migration).

### Interfaces (new, slice-scoped)

| Interface | Role |
|---|---|
| `ops native-decisions init/status/reset/retire --line-id` | Create, inspect (read-only), reset or retire one line. `reset` and `retire` refuse while a run lock is held. |
| `native_decision_candidates(score_batch, population) -> rows` | Pure: native score record -> candidate row, population selected with `population_key` over the same planned set the legacy scorer uses. |
| `commit_native_decisions(scratch_conn, candidates, owner="scratch")` | The only writer; caller owns the transaction. **First**, inside that transaction, it checks `decision_authority.owner == scratch` and refuses `DecisionConflict` otherwise, before any lookup, so no branch below runs under a wrong owner (`record_divergence` has no authority check of its own). Then it looks up `decision_id`: identical payload hash (any generation) returns the existing row without calling `decisions.insert`, but only after `commit_native_decisions` itself checks that the stored `purpose`, `kind` and `supersedes` equal the candidate's (`logical_key` is excluded because it is generation-scoped, and `decision_id` is the lookup key); a mismatch refuses `IDEMPOTENCY_CONFLICT` with no write; a different payload from the **same** generation (`generation_ref` equal) refuses `IDEMPOTENCY_CONFLICT` with no write; a different payload from a **later** generation calls `record_divergence` and inserts nothing; only a missing id reaches `decisions.insert`. Divergence rows are written in the caller's transaction, so a rollback discards them with the rest of the session. This is needed because `logical_key` is generation-scoped and `insert` would otherwise raise `IDEMPOTENCY_CONFLICT` on an unchanged re-commit. |
| `settle_native(scratch_conn, outcomes)` | Settlement from native outcome observations into the same scratch catalog (kind `outcome`). |
| `export_native_generation(scratch_conn)` | Wraps `export_generation` into a **scratch export root**; never the legacy ledger directory. |
| `native_reconcile_report(line_id, session)` | Read-only nightly comparison against the official catalog opened read-only. |

Nightly: after the legacy shadow commit and `native_score_batch`, the new job set `native_decision_commit` -> `native_settlement` -> `native_reconcile` runs in the shadow nightly. It writes only scratch; the official jobs are unchanged and do not depend on it. A scratch failure never fails or blocks the official night.

### DYN-SV chooser replay

Two stages, kept separate. (1) Chooser scoring/ranking: the DYN-SV chooser is replayed from the frozen native score record plus the pinned model release; its inputs hash goes in the row's validation block. (2) Serialization: legacy `build_prediction_rows` only serializes already-scored rows, so the native path reuses its row shape, not its scoring. The exact chooser fields are slice 2's doc fold; the contract here is that the replay is deterministic on `(score_batch hash, model release, snapshot id)` and a second run produces byte-identical payloads.

### Relationship to the authority fence

- Scratch is **outside** the fence: S1 captures only the official catalog; S2's "pause non-official catalog writers" list gains `native_decision_commit` and `native_settlement` (they pause with the shadow nightly, and S2 verifies no holder of the scratch run lock). They cannot move official marks because they cannot open the official file for writing.
- **Rehearsal reuse.** A P7-3 drill may point its scratch target at a copy of a native scratch catalog (backup then restore through `backup.py` / `v2_restore_drill.py`) and write owner `drill` on the **copy**. The live scratch line stays `scratch`. Drills never write the live scratch line.
- **At cutover** (after S5 grants `v2` on the official catalog) the scratch line is **frozen**: `native-decisions status` shows `closed_at` and the generation; no scratch row is ever imported into the official catalog (official rows come from the S2b legacy import and the official `v2` nightly). The line is kept as durable qualification evidence for the 10-session rollback window and is removed only by explicit `retire` after it. Backout (B1-B3) does not touch scratch.
- Before the S5 switch legacy remains the only official writer (afterwards `v2` is, and scratch stays frozen); nothing here exports into the legacy ledger.

### Failure semantics (condition -> outcome)

| Condition | Outcome |
|---|---|
| Scratch path resolves to, or into, the official catalog (incl. via symlink) | Refuse `SCRATCH_PATH_OFFICIAL`; nothing opened for write |
| Catalog has no `scratch_meta`, wrong origin or wrong `line_id` | Refuse `SCRATCH_META_INVALID`; no write |
| `decision_authority.owner != scratch` on the target | `DecisionConflict` from the wrapper's initial authority check (before any lookup); no write |
| Same logical key, identical payload (retry, same session rerun) | Return the committed row; no new row |
| Same logical key, different payload within the same native generation | Refuse `IDEMPOTENCY_CONFLICT`; the first row stays; no write and no divergence row (same-generation drift is a defect, not evidence) |
| Later generation, same occurrence (same `decision_id`), identical content | Return the existing row; nothing written |
| Later generation, same occurrence, different content | First commit stays authoritative; divergence evidence only (existing rule); never a silent duplicate |
| `reset` with a counted night, `closed_at` set, or inside the rollback window | Refuse `RETENTION_ACTIVE`; nothing deleted (use a new `line_id` to start over) |
| Same-session supersession | Only by explicit operator supersede with a reason, against a native row; a supersede targeting a legacy-origin id is refused `SUPERSEDE_CROSS_ORIGIN` |
| Native score batch missing, partial or its population differs from planned | Commit nothing (all-or-nothing per session); report `population_difference`; official night unaffected |
| Missing finality for an event | Do not settle it; outcome not written; listed in the reconcile report as `awaiting_finality` |
| Crash mid-commit | Caller transaction rolls back; the retry re-derives identical ids; no partial session visible |
| Scratch catalog missing or corrupt | Job fails with `SCRATCH_UNAVAILABLE`; an eligible line (no counted night, not closed) is `reset` and rebuilt from the retained native score artifacts; a protected line is left intact and rebuilt under a **new `line_id`**; no repair in place |
| Concurrent native writers on one line | Single-run lock (`<line_dir>/run.lock`); second refuses `RUN_LOCK_HELD` |
| Native job holds a lock past the bound | Bounded wait (default 300 s) then `RUN_LOCK_TIMEOUT`; no authority change |
| Scratch disk full or write error | Job fails, transaction rolls back; the official night continues |
| Export target equals the legacy ledger directory | Refuse `EXPORT_TARGET_OFFICIAL`; no file written |

## Qualification evidence

Each night, `native_reconcile` writes one immutable JSON artifact (counts and hashes only) at `<line_dir>/reconcile/<session>.json`:

- **Population:** planned vs native committed vs legacy committed counts and `population_difference` both ways; accepted-difference classes named (e.g. #681 class while open).
- **Decision agreement:** per-field numeric tolerance (cutover parity bar B) between native and legacy decision payloads matched by `population_key`; decisions may drift, drift is counted by class, not hidden.
- **Settlement:** native outcome vs legacy outcome per settled event, with `awaiting_finality` counts.
- **Book/funding:** totals computed from the scratch catalog through the existing catalog accounting (`portfolio`) vs the legacy totals, per P6-3 acceptance ("totals agree with existing accounting").
- **Negative controls (once per release, by test and recorded):** duplicate commit, retry, same-session supersession and original-immutability each produce the table outcome above.
- **Isolation proof:** the official catalog's `decisions` max `sequence` and per-kind counts are identical before and after the native jobs; the artifact records both.

The P7 readiness checker may count a night toward the ten-session qualification only if its reconcile artifact exists, isolation holds and no unclassified drift remains.

## Slices (code blocked on #681/#651; each ends with R1-R6 semantics in its PR)

| # | Slice | Code | Test | Doc |
|---|---|---|---|---|
| 1 | Scratch identity: `scratch_meta`, owner `scratch`, path resolver, `init/status/reset` | ~180 | ~150 | ~30 |
| 2 | Candidate builder + DYN-SV replay, population rule | ~300 | ~220 | ~40 |
| 3 | `commit_native_decisions` + `settle_native`, negative controls | ~280 | ~250 | ~30 |
| 4 | Nightly jobs wiring + S2 pause-list entry | ~150 | ~120 | ~30 |
| 5 | `native_reconcile` report + scratch export + isolation proof | ~320 | ~220 | ~30 |
| 6 | P7-3 drill reuse (copy as target) | ~100 | ~100 | ~20 |
