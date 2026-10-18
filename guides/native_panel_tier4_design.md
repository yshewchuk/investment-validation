# Native feature panel and Tier-4 coverage (design, #876)

Design for removing the native nightly's dependency on the legacy panel and
Tier-4 outputs. Contract text lives in `engine/v2/data/ARCHITECTURE.md`
("Native feature panel producer") and `engine/v2/models/ARCHITECTURE.md`
("Native Tier-4 values"); this file holds the map, the options and the slices.
Decisions by the user, 2026-10-18: (1a) native batch producer for the
historical `feature_panel`; (2b) Tier-4 values derive from native release
models, only a coverage table is stored, `tier4.py` is not ported; (3) the
`legacy_manifest.json` binding is replaced by snapshot-id and dataset-version
bindings. Refs #876, #866, #782.

## 1. What native reads from legacy today

| Read | Where | Replacement |
| --- | --- | --- |
| `data/features/panel.parquet` -> snapshot `feature_panel` | `data/import_snapshot.py:20`, `data/legacy_mapping.py` | native panel producer (slice 2) |
| `data/features/tier4_forecasts.parquet` -> snapshot `tier4_forecasts` | same | native Tier-4 coverage table (slice 3) |
| Tier-4 serving caches (joblib named by panel sha) | `data/reference_inputs.py`, `data/tier4_coverage.py`, `models/adapters.py`, `models/serving_fold_loader.py` | release-model members (slice 3-4) |
| `feature_panel` history block | `models/training/d14_dataset.py` | the native table (slice 2) |
| `legacy_manifest.json` binding | `ops/training.py`, `ops/supervisor.py` (`_pin_read_set`), `ops/snapshot_import.py`, `ops/pool_refresh.py`, `ops/stages.py`, `ops/effects_graph.py`, `ops/capture_inputs.py`, models release stage | snapshot binding (slice 5) |
| `legacy_features` action: `engine.data.rebuild.build_panel_table` | `ops/legacy_adapter.py` | native producer (slice 2) |

Already native: `features/panel_math.py`, `scan_panel_row` (one forward row),
regime and run-up math, Massive daily, SUPER SIZE history, computed moves,
price history. Missing: the stored historical table (all events, ORATS and
pre-print block, outcome columns) and every Tier-4 column.

## 2. Native panel producer (decision 1a)

A batch job reads the pinned snapshot's `daily_market`, `earnings_events`,
`orats` and calendar tables in bounded batches (ticker groups, one year
partition at a time; peak memory is a function of the batch, never of the
table) and publishes `feature_panel` as a new dataset version of the same
snapshot lineage. Row math is `panel_math`, `regime` and `runup_math` only;
`scan_panel_row` and the batch share them, so a forward row equals the stored
row for the same event by construction, and a test pins that equality.
Qualification compares the native table with the legacy table per column under
the existing per-field numeric tolerance (parity bar B); a column that differs
is reported, never emulated. Lost-data rule: the table is always rebuildable
from the snapshot's source tables; no migration of the legacy file.

## 3. Native Tier-4 values (decision 2b)

A "Tier-4 value" (`pred_abs_move`, bands, `im_t1`, `runup_move`, `iv_crush`)
is the prediction of a Tier-4 member of a native model release for one event.
The stored artifact is `tier4_coverage`: one row per `(ticker, event_date)`
with the producing `release_id`, `model_id`, `retrain_cutoff` (stamp) and a
status; values themselves are computed at read time from the release (R2:
never cached across releases) or by an explicit materialization job whose
output is hashed to the release.

### Causality (real choice; default chosen, flagged)

A historical event's value may come only from a release whose
`retrain_cutoff` (exclusive, `M_end + 1 day`, models §2.3) is not after the
event's decision date, matching the existing `fold_start <= as_of` guard.
Options for history before the first eligible release:

- **A (default in this design): no invented history.** Events without an
  eligible release are `no_eligible_release` and their values are null, the
  legacy "no forecast" meaning, which consumers already handle. A backfill is
  an explicit operator job that runs the §2.3 monthly retrain at past
  cutoffs and keeps those releases; coverage then extends back to the
  earliest retained release. Start date = earliest retained eligible release.
- **B: mandatory per-cutoff backfill** of every month back to the first
  training month. Equivalent to re-implementing legacy monthly OOS folds; the
  user rejected that port in 2b, and it is heavy.

Training downstream of Tier-4 features (the `tier4_monthly_oos` lineage) can
only use events at or after the start date under A. This is the cost of A and
needs the user's confirmation before slice 3.

## 4. Failure semantics (R1-R6) and no-release behaviour

| R | Panel producer / coverage producer |
| --- | --- |
| R1 verified input | Snapshot id, table versions and (coverage) release manifest and hashes verified before any read; mismatch `IDENTITY_CONFLICT`; release not promoted/eligible `MODEL_NOT_READY` |
| R2 cache | No cache across snapshots or releases; coverage rows are keyed `(snapshot_id, release_id)` |
| R3 retry | A failed batch job is terminal for its key; a new key re-runs from the snapshot |
| R4 transaction | Batch outputs private until all batches finish, then one hashed publish; no partial dataset version is visible |
| R5 partial failure | Unpublished batch objects stay unused and are reclaimable by the owner of the store |
| R6 idempotency | Same snapshot, code version and release give the same dataset version hash; a reused key with other content refuses |

**No eligible release (#782).** When the release resolver finds no release
that is promoted and whose `retrain_cutoff` is not after the session's
decision date, the nightly neither guesses nor falls back to legacy: the
coverage row is `status=MODEL_NOT_READY`, `native_score_batch` refuses those
events with the existing `MODEL_NOT_READY` code and names the release check
that failed in `details`, and the parity report counts them as native
refusals (a visible, not hidden, class). The nightly's other jobs continue.
A refusal is correct only when the release truly is absent; #782 tracks
whether today's 101 refusals are that case.

## 5. Qualification replaces parity

Legacy Tier-4 is walk-forward fits of `tier4.py`; native values come from
different models, so equality is not expected and not tested. Qualification
instead checks: (1) causality: for sampled events the producing release's
`retrain_cutoff <= decision date` and no training row postdates it; (2)
coverage: share of the planned population with a non-null native value, by
month; (3) agreement as rank correlation and calibration of bands against
realised moves, on events both systems cover, reported as consistency
measures, not magnitude cutoffs; (4) downstream: native board decisions vs
legacy decisions under parity bar B, with drift attributed to Tier-4 only
when the rows' other inputs match.

## 6. Manifest replacement (decision 3)

Jobs bind `{snapshot_id, table_versions}` (and, for model-dependent jobs,
the release id) instead of a `legacy_manifest.json` artifact. The read-set
pinning that `_pin_read_set` does for legacy files becomes verification of
those ids against the catalog. Legacy-only job kinds (`legacy_*`) keep their
manifest until they are removed; native kinds stop accepting it. The model
training cutoff keys on the latest succeeded snapshot-bound `feature_panel`
version instead of the latest `finality`/`features` job.

## 7. Slices (one at a time, each add -> wire -> remove)

1. This design and the contract text (docs only).
2. Native panel producer: batch job, shared-math equality test, bounded memory
   test on a synthetic wide snapshot, publish as a dataset version (add).
3. Tier-4 coverage table and resolver, with `MODEL_NOT_READY` semantics (add).
4. Wire scoring and training to the native tables and the coverage resolver;
   the import stops copying the legacy files (wire).
5. Replace `legacy_manifest.json` bindings for native job kinds (wire).
6. Qualification report (section 5) in the nightly artifacts (add).
7. Remove `legacy_features`, legacy table import and serving-cache readers
   after qualification passes (remove).

Open decision for the user: causality option A vs B (section 3).
