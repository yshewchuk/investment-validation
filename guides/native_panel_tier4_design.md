# Native feature panel and Tier-4 coverage (design, #876)

Design for removing the native nightly's dependency on the legacy panel and
Tier-4 outputs. This file holds the map, the options and the slices; the
component `ARCHITECTURE.md` contracts (`engine/v2/data`, `engine/v2/models`
§2.4) are written by the slice that implements each producer, not before.
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

The batch job is owned by `engine.v2.ops` (a higher layer that may import
`engine.v2.features` math, so the import direction is preserved; `engine.v2.data`
only stores the published dataset version). It reads the pinned snapshot's `daily_market` (ORATS state lives
here; there is no `orats` table), `earnings_events`, `computed_moves`
(realised/implied history), `price_history` (adjusted-close run-up) and the
calendar in bounded batches (ticker groups, one year
partition at a time; peak memory is a function of the batch, never of the
table) and publishes `feature_panel` as a new dataset version of the same
snapshot lineage. Row math is `panel_math`, `regime` and `runup_math` only;
`scan_panel_row` and the batch share them, so a forward row equals the stored
row for the same event and the same `decision_session` and `history_start`
on the common causal feature projection (the columns `scan_panel_row`
returns); a test pins equality on that projection only. Outcome and pre-print
columns (`move`, `abs_move`, `or_iv30`, `pre_iv30`) exist only in the stored
table and carry no forward guarantee; another decision session legitimately
differs.
Qualification compares the native table with the legacy table per column under
the existing per-field numeric tolerance (parity bar B); a column that differs
is reported, never emulated. Lost-data rule: the table is always rebuildable
from the snapshot's source tables; no migration of the legacy file.

## 3. Native Tier-4 values (decision 2b)

A "Tier-4 value" (`pred_abs_move`, bands, `im_t1`, `runup_move`, `iv_crush`)
is the prediction of a Tier-4 member of a native model release for one event.
The only stored artifact is `tier4_coverage`: one row per
`(ticker, event_date, member)`, `member` in {`size`, `implied_t1`,
`runup_move`, `iv_crush`} (the four release feature roles), with that
member's producing `release_id`, `model_id`, `retrain_cutoff` (stamp) and
`status`, so partial availability is per member. Values are never persisted:
they are computed at read time from the release (R2: never cached across
releases).

### Causality (real choice; default chosen, flagged)

A historical event's value may come only from a release whose
member cutoff is not after the event's decision date. The member cutoff is
the producing model's own training cutoff (exclusive, `M_end + 1 day`):
a carried member keeps its prior cutoff (models §2.3), so it is read from the
member, not from the release-level `sources.retrain_cutoff`, and a pool-only
release does not re-stamp an unchanged model, matching the existing `fold_start <= as_of` guard.
Options for history before the first eligible release:

- **A (default in this design): no invented history.** Events without an
  eligible release are `no_eligible_release` and their values are null, the
  legacy "no forecast" meaning, which consumers already handle. A backfill is
  an explicit operator bootstrap job, separate from the monotonic §2.3
  monthly path (which only advances past the base release's cutoff). It fits
  each historical cutoff from an empty base into its own retained,
  never-promoted release; those releases are read only by the coverage
  resolver. Its contract is a slice 3 deliverable. Coverage then extends back
  to the earliest retained release. Start date = earliest retained eligible release.
- **B: mandatory per-cutoff backfill** of every month back to the first
  training month. Equivalent to re-implementing legacy monthly OOS folds; the
  user rejected that port in 2b, and it is heavy.

Training downstream of Tier-4 features (the `tier4_monthly_oos` lineage) can
only use events at or after the start date under A. This is the cost of A and
needs the user's confirmation before slice 3.

## 4. Failure semantics (R1-R6) and no-release behaviour

| R | Panel producer / coverage producer |
| --- | --- |
| R1 verified input | Snapshot id, table versions and (coverage) release manifest and hashes verified before any read; mismatch `IDENTITY_CONFLICT`; an absent or ineligible release is not a producer failure: it records `no_eligible_release` coverage (section 4) |
| R1 unusable input | A missing or unusable source table, column or row refuses `CONTRACT_MISMATCH` (registered data code); no dataset version is published. Distinct from `no_eligible_release` |
| R2 cache | No cache across snapshots, calendars or releases. The coverage-run identity is `(snapshot_id, calendar_hash, candidate release set)`; the row key within a run is `(ticker, event_date, member)` |
| R3 retry | A failed batch job is terminal for its key; a new key re-runs from the snapshot |
| R4 transaction | Batch outputs private until all batches finish, then one hashed publish; no partial dataset version is visible |
| R5 partial failure | Unpublished batch objects stay unused and are reclaimable by the owner of the store |
| R6 idempotency | Same snapshot, calendar hash, code version and release give the same dataset version hash; a reused key with other content refuses |

**No eligible release (#782).** The resolver considers the promoted releases
(live pointer) for forward events, and for historical events also the retained
never-promoted bootstrap releases (resolved by id, as `resolve_release` does).
A member is eligible when its own training cutoff (not the release stamp) is
not after the event's decision date. Selection is deterministic: among
eligible members the one with the latest own cutoff wins, ties broken by the
greatest `release_id`; the candidate release set is pinned in the run identity
(`resolve_release` only resolves a supplied id, it never chooses). When none is
eligible, the nightly neither guesses nor falls back to legacy. One
status, two consumers: the coverage `status` is `no_eligible_release` with a
`reason` (`none_promoted`, `cutoff_after_decision`, `member_missing`). A
historical event older than the earliest retained release is a null value
(legacy "no forecast"; section 3, option A). A forward event of the scored
session, whose release should exist, is a scoring refusal:
`native_score_batch` refuses it with the existing `MODEL_NOT_READY` code,
names the reason in `details`, and the parity report counts it as a native
refusal (a visible, not hidden, class). The nightly's other jobs continue.
A refusal is correct only when the release truly is absent; #782 tracks
whether today's 101 refusals are that case.

## 5. Qualification replaces parity

Legacy Tier-4 is walk-forward fits of `tier4.py`; native values come from
different models, so equality is not expected and not tested. Qualification
instead checks: (1) causality: for sampled events the producing member's
own training cutoff `<= decision date` and every training row is dated strictly
before the exclusive cutoff (a row dated exactly at it is rejected); (2)
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
those ids against the catalog. The trading-calendar pin is kept: the
binding also carries the committed calendar import receipt id and calendar
hash (as the D14 inputs do, models §2), because snapshot ids alone do not
identify calendar bytes. Legacy-only job kinds (`legacy_*`) keep their
manifest until they are removed; native kinds stop accepting it. The model
training cutoff keys on the latest succeeded snapshot-bound `feature_panel`
version instead of the latest `finality`/`features` job.

## 7. Slices (one at a time, each add -> wire -> remove)

Estimates are added lines (code / test / doc); each slice is its own PR.

1. This design and the contract text (docs only): 0 / 0 / ~190.
2. Native panel producer: batch job, shared-math equality test, bounded memory
   test on a synthetic wide snapshot, publish as a dataset version (add):
   ~380 / ~300 / ~30.
3. Tier-4 coverage table and resolver, per member, with the section 4
   semantics (add): ~300 / ~250 / ~30.
4. Wire scoring and training to the native tables and the coverage resolver;
   the import stops copying the legacy files (wire), split in two PRs
   (scoring; training): ~350 / ~250 / ~40 each.
5. Replace `legacy_manifest.json` bindings for native job kinds (wire),
   split by job kind if over limit: ~380 / ~300 / ~40.
6. Qualification report (section 5) in the nightly artifacts (add):
   ~350 / ~250 / ~30.
7. Remove `legacy_features`, legacy table import and serving-cache readers
   after qualification passes (remove): mostly deletions / ~50 / ~40.

Open decision for the user: causality option A vs B (section 3).
