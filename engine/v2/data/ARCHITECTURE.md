# `engine/v2/data` — architecture

Layer **1.0** in the root [`ARCHITECTURE.md`](../../../ARCHITECTURE.md)'s
layer table (§2). Replaces legacy's `data/sources/`, `store.py`, `fetch.py`,
`finality.py`, `rebuild.py`, and calendar sourcing from `calendar.py`. See
`engine/v2/data/README.md` for the checker-enforced Public interface /
Consumers lists; a caller reaching a name via a submodule-qualified import
(`from engine.v2.data.X import Y`) need not appear in that directive.

## Purpose

Ingestion, normalization, coverage/finality, and atomic snapshot commit: an
immutable, content-addressed object/fragment/dataset-version/snapshot
identity chain (`objects.py`, `manifests.py`); atomic snapshot commit with
idempotent inserts and a compare-and-swap head (`catalog.py`); the
incremental-refresh merge policy shared by every EOD table family
(`incremental_tables.py`, `incremental.py` for `daily_market`,
`generic_incremental.py` for any registered `TableContract`;
`generic_streaming.py` is the separate whole-table replacement route); bounded,
exact reads over a committed snapshot (`repository.py`, `query.py`,
`events.py`, `chains.py`, `price_history_query.py`); the one legacy-touching
seam (`legacy_adapter.py`) and what is built on it read-only (table
mapping, pinned reference-input resolution, snapshot-import planning,
legacy-tree materialization); legacy-free table families
(`computed_moves*`, bitemporal `price_history*`, pure `iv_crush*` targets); the data-owner catalog
schema (`schema.py`, colocated with ops/ledger under migration owner
`"data"`); pure coverage checks (`eod_inventory.py`, `event_revisions.py`);
and a Tier-4 serving-cache coverage check (`tier4_coverage.py` — not pure,
not legacy-free) gating a scoring launch from `engine.v2.ops`.

Non-responsibilities: never computes a trading verdict (`engine/v2/scoring`)
or promotes a champion (`engine/v2/models/training`); never decides *what* to
fetch or *when* to run a job (`engine/v2/ops`, a caller, owns that).

Repository scans merge one transitively overlapping full-primary-key range group at
a time. Disjoint inspected fragment ranges therefore keep one active fragment stream;
unknown bounds retain the full merge fallback. Bounds, timestamp predicates and decoded
rows share the same typed key ordering. Stored unpadded low-year spellings must round-trip exactly through
the shared formatter; admission never rewrites their content identities.
Shared bound admission precedes pruning in scan, population-bound,
dependency and pure query planning. Null-bearing keys prove no range: they
retain the full merge fallback and downstream row validation. Object verification,
footer counts, strict global order and result limits remain mandatory; batches are provisional until exhaustion.

`Repository.scan(..., max_object_bytes=None)` preserves path-based reads by default.
With a cap, `objects.read_verified_bytes` hashes bounded immutable bytes without cache
reuse; PyArrow checks all row-group uncompressed sizes before decoding. Thrift limits
are capped at the library's 32-bit maximum even for larger Python integers. Buffers
close on completion, errors or generator closure; existing row/integrity checks remain.

| Byte-read condition | Outcome |
|---|---|
| Positive integer cap (not bool); all declared sizes and total uncompressed bytes fit | Verified bounded scan; exact byte/expansion equality is admitted |
| None passed to scan / to read_verified_bytes | Default path scan / `QUERY_NOT_BOUNDED` |
| Other invalid cap (zero, negative, bool, non-integer) | `QUERY_NOT_BOUNDED` before object access |
| Any surviving declared object / total uncompressed row groups exceed cap | `RESULT_LIMIT_EXCEEDED` before any object read / before batch decode |
| Missing, unsafe, mutated or grown backing object | `OBJECT_CORRUPT`; no path/cache fallback |

## Primary contracts and public interfaces

**IV-crush target construction.** `iv_crush.build_rows` is a pure, bounded,
single-ticker builder for `iv_crush_table.IV_CRUSH_CONTRACT`
(`iv_crush_targets.v2`). It consumes caller-supplied event date/session and
daily date/IV30 projections, never a panel forecast or a store. Callers must
supply IV30 in vol points expressed as percent (50 for 50% IV); the builder
does not normalize or validate scale. Canonical and unique event/daily dates
are mandatory; unordered inputs sort deterministically. AMC selects the last
daily row on/before the event; BMO selects the last strictly before it. Post is
the immediately following row, including when its IV is missing. No search for
a later valid IV, forward fill, inferred holiday session or warm-up filter occurs.
The existing target is signed `100*(post_iv30/pre_iv30-1)`, a relative percentage
change, not a vol-point difference; anchor compatibility and skips are below.
For a computable label, `available_as_of_date` is post-close date plus one
calendar day, the native computed-moves day-clock convention. It is a modeled
availability floor, not evidence of actual source publication or finality.
Training/research consumers must separately admit holdouts and enforce their
label clock.
`input_hash` binds the actual typed projections (including invalid IV scalars),
while `source_id` is only a caller declaration, never verified provenance.
The builder neither reads/writes files nor registers/publishes a snapshot;
physical event isolation, source verification, feature reconstruction and
registered evaluation are outside the builder. There is no production caller.
Text columns carry no `null_policy`; null and provenance rules for them live in
the table-level `coverage_semantics` and `provenance_semantics`.
`RESULT_LIMIT_EXCEEDED` and `CONTRACT_MISMATCH` conditions are in the failure
table. No cache, retries or transactions.

Builder outcomes below are checked in order. Skips have `skipped=true` and null
target/availability, never imputed zeros; selected dates and finite IV values
remain diagnostic. Success has `skipped=false` and `skip_reason=null`.

| Condition | Outcome / skip reason |
|---|---|
| No supplied events, with otherwise valid bounded inputs | Empty list; daily input still validated |
| Null / other unrecognized session | `missing_session` / `unknown_session`; session and all anchors null |
| No pre row (including empty daily input) / no post row | `missing_pre` / `missing_post`; existing opposite anchor retained |
| Adjacent dates more than five calendar days apart | `gap_exceeds_limit`; gap exactly five is admitted |
| Pre-IV missing, nonnumeric, nonfinite or nonpositive | `invalid_pre_iv30`; nonpositive finite value retained diagnostically |
| Post-IV missing, nonnumeric or nonfinite | `invalid_post_iv30`; finite zero/negative post-IV is admitted as in the existing recipe, without endorsing its quality |
| Finite valid anchors overflow the ratio calculation | `nonfinite_target`; finite anchors retained |
| Computable signed label | Finite target, post-date plus one calendar day availability |

Both public refusals carry fixed redacted messages, empty details, no stage,
and the category/retryability in the failure table; no partial list is returned.

**Massive reconstruction.** `massive_daily` owns `massive_daily.v1`; `massive_actions`
owns `massive_splits.v1` and `massive_dividends.v1`, with original page provenance.
It never changes `price_history.v3` or imports ops/provider code. Each
normalization input contains one ticker and explicit response basis; no
dividend adjustment or cross-basis conversion is inferred. Eastern session dates,
finite float64-exact positive OHLC, consistent bounds, nonnegative volume and canonical UTC clocks are required;
duplicate ticker/basis/session keys refuse, including identical duplicates. Rows retain their
original page's raw receipt and observation time plus completion occurrence ID.
The dataset also pins the saved acquisition artifact. Knowledge is `reconstructed`,
without historical-availability, calendar completeness or finality evidence.
Only rows inside saved clipped bindings are included; an empty selected view
refuses. `Repository.scan(DataQuery(...))` supplies existing bounded pinned reads.
Ops performs live-fenced atomic snapshot publication; publishing object bytes
alone does not expose a dataset. `massive_refresh` composes explicit pinned parents.
Action keys are ticker/event ID, not date: same-day events remain distinct.
Only numeric action columns carry reported-unit and adjustment metadata;
text columns leave type-conditional metadata unset, as `ColumnContract` requires.
Registered definitions are immutable; a mismatched parent definition refuses
rather than being relabelled or overwritten by refresh.
Selected actions require ID, ticker, event date and split ratio or cash/currency;
optional reported dates, frequency, classifications and adjustment factors retain
null absence; supplied strings are nonblank and at most 256 characters. No ratio or return is computed.
Refresh replaces the selected ticker/basis's closed date interval, preserving
outside rows, other partitions and unrelated table versions. Duplicate keys refuse.
The affected ticker's prior object bytes are capped at 64 MiB and rows at 1,000;
the rewritten partition also has a 1,000-row cap. Original row receipts survive.
Updated manifests retain old coverage pins and add the acquisition artifact;
unrelated manifests and snapshot-wide policy/finality references carry forward.
Shared catalog identity reconciliation retains first-stored lineage/provenance.

The exhaustive name-by-name list is `README.md`'s checker-enforced Public
interface section; this names only the load-bearing entry points.

`documents.reject_duplicate_keys(pairs)` is the public, pure stdlib JSON hook
used by `loads_document`, `experiments.v2_runup_exits` and `experiments.v2_magnitude_search_run`; it returns an object
dictionary or raises `DocumentError` with `DUPLICATE_KEY`, with no I/O or retry.

- **Snapshot commit** — `catalog.commit_snapshot`/`.record_failed_import`/
  `.move_head`: the one place a snapshot's rows, membership and head move
  together inside one transaction (Failure semantics).
- **Contract-generic incremental commit** — `generic_incremental.
  build_generic_table_candidate`/`.commit_generic_table_candidate` wrap
  `catalog.commit_snapshot` with a head-fence expectation and an optional
  caller `fence_check` (Failure semantics).
- **`daily_market` incremental refresh** — `incremental.py`'s
  run/build/commit/merge functions and shared merge primitives in
  `incremental_tables.py` (also used by `engine.v2.research`/
  `forward_calendar_store`). See Invariants for revision identity/ordering,
  planned-unit key validation, `normalizer_id` versioning, coverage
  completeness, and mcap carry-forward.
- **Snapshot resolution and bounded reads** — `repository.Repository`:
  exact re-verifying `resolve`/`resolve_full` (+ `_pinned`; opt-in `pin_snapshots=True` memoises per instance `resolve` by snapshot id and the scan/bound path's table contract and membership by `(dataset_version_id, contract_id)`, not `fragment_records`, default off, a failure is never cached, each call returns an independent copy), a bounded Arrow `scan`, typed `get_event`/`get_chain`/`get_price_series`/`get_close`, and `explain_dependencies`.
  Metadata-only `scan_population_bound` sums recorded counts of surviving
  fragments of the supplied snapshot, with no head fallback or object reads,
  through the same membership-bound planner behind `scan` and
  `explain_dependencies`. That summed manifest-footer count is the
  authoritative selected-population bound: both paths check a request's
  `max_result_rows` against it before row streaming. A query-level limit
  stays explicit caller intent — legal below the bound, refused
  `QUERY_NOT_BOUNDED` above it. Retained caller guards must be prepared as
  `min(caller_guard, selected_population_bound)` before constructing a scan
  query; this preserves caller-side ceilings without weakening the direct-query
  refusal. Do not clamp a caller-supplied `DataQuery` inside the validator.
- **Pure primitives, no I/O** — `query.py` and `documents.py` (`manifests.py`
  and `objects.py` are identity builders, not pure: `manifests.
  verify_partition_hashes` calls `objects.partition_logical_hash`, which
  opens and streams object bytes through an `ArtifactStore`, and `objects.py`
  also holds `publish_legacy_file`/`inspect_fragment`, which do filesystem
  I/O). `query.compile_batch_matcher(contract, query)` is the entry point
  `repository.Repository._fragment_rows` calls once per fragment to
  vectorize `key_filter` equality/set-membership (pyarrow `compute.is_in`)
  on a `string`/`int64`/timestamp column (a timestamp column is floored
  and widened to microsecond resolution first, exactly like the row
  path's own wire form), returning a boolean mask `_fragment_rows` uses
  to drop non-matching rows via `RecordBatch.filter` *before* they are
  decoded to per-row Python dicts, so only surviving rows pay that cost.
  Any `time_interval`, any predicate on a `bool`/`float64` column
  (`is_in` compares a float's raw bit pattern, so `-0.0` never matches a
  `0` value_set entry even though the row path's plain Python equality
  treats them equal), or a predicate value not representable in its
  column's declared Arrow type makes the function return `None` at
  compile time, decided from `contract`/`query` alone, never from the
  data, and the whole query falls back to `compile_row_matcher`.
  `_fragment_rows` also falls back per fragment if the compiled mask
  itself raises when evaluated against a real batch — a case
  compile-time refusal cannot fully rule out. A null column value never
  matches, on both paths.

  `compile_row_matcher(contract, query)` is the one row-matching entry
  point for every query this file does not vectorize, and for any caller
  testing more than one row against the same `DataQuery`: it normalizes
  every `key_filter` predicate's `wanted` values and the `time_interval`'s
  bounds eagerly, once, at compile time; stored string-date observations are
  normalized per row. The returned closure is only valid
  for the `contract`/`query` it was compiled from and carries no state
  across scans. Both compiled forms must agree on every row the batch form
  accepts (the equivalence fixture in `tests/test_v2_data_query.py` pins
  this). `repository.Repository._fragment_rows` compiles the batch
  matcher for every fragment, and additionally compiles the row-matcher
  fallback when (and only when) batch compilation returns `None`, or the
  first time batch evaluation raises on that fragment (the scan's
  `DataQuery` does not change across fragments, so this is O(fragments ×
  predicate values), not O(rows × values)). `row_matches(row, contract,
  query)` stays available as the one-row form and is defined in terms of
  `compile_row_matcher` so the two can never diverge; it re-normalizes on
  every call and must not be used inside a per-row loop. Production
  reachability: `python3 -m engine.v2.ops serve` → `Service.tick()` →
  `_reconcile_computed_moves_refresh()` →
  `submit_computed_moves_refresh_if_ready()` →
  `_build_native_computed_moves_plan()` →
  `computed_moves_store.target_tickers_from_snapshot()` → `_scan_rows()` →
  `Repository.scan()` → `_fragment_rows()` → `query.compile_batch_matcher()`
  (row-path fallback: `query.compile_row_matcher()`).
  Time-interval comparisons normalize both stored fragment bounds and stored
  string-date observation values to the same timestamp form as query bounds.
  Invalid dates refuse `CONTRACT_MISMATCH`; pruning never drops a matching
  calendar day merely because its stored representation lacks a time suffix.
- **Read counters** — `read_counters.ReadCounters`, one per `Repository`
  instance (`Repository.read_counters`): per table scans, fragments opened,
  rows decoded, rows returned, bytes and seconds. Observational only; an
  error while counting is swallowed and never alters or fails a scan.
- **Legacy-touching seam** — `legacy_adapter.py`, the package's only module
  importing legacy `engine.*` code (17 declared, read-only entries). Built
  on it, read-only: `legacy_mapping.py` (table mapping);
  `legacy_materialization.py`/`legacy_nightly_read_plan.py` (materialization
  and nightly barrier plans, neither importing legacy `engine.*` directly);
  `reference_inputs.py`/`reference_catalog.py` (pinned reference-input
  resolution, the latter legacy-free); `import_snapshot.py`; `tier4_coverage.py`.
- **`computed_moves`/`price_history`** — pure math + each family's
  `TableContract`; `price_history_query.py` also exposed as
  `Repository.get_price_series`/`.get_close`. A price-history read
  reconstructs row versions through the latest retrieval present in its
  pinned snapshot; `retrieved_at` is provenance, not an observation ceiling.
  The reader exposes only bars dated
  on or before the decision session. This permits a session close fetched
  after midnight to be used for that session, while a later-dated bar remains
  unavailable. A vendor may revise historical values in a later retrieval,
  including split or dividend adjustments, or backfill a previously absent
  date; using the latest pinned retrieval accepts that vendor-revision risk
  rather than treating retrieval time as causal availability.
- **Errors** — `errors.DataError`, built only from a registered
  `DATA_FAILURE_CODES` entry.
- **Contract changes** — Recalculate data tables through the standard import
  and rebuild commands; stored identities are not migrated.

## Inputs

### EOD availability evidence

`DatasetManifest.availability_evidence_refs` pins content-addressed source
receipts; `SnapshotRef.finality_receipt_refs` pins the corresponding finality
evidence. These existing identity carriers are shared by quote and daily-state
admission. Import/reference receipt registration is not source availability.

An EOD source receipt names its table contract, exact session, exact output
`ObjectRef`/full byte hashes, source evidence, and producer attempt/fence. It
binds source finality validation and the coordinator-published checkpoint
whose artifact membership contains those outputs. The coordinator publication
clock is an availability upper bound only after genuine source completion
validation; a successful arbitrary worker is insufficient. Caller-provided
timestamps, file mtimes, legacy snapshot dates and fitted prices supply no proof.
Existing checkpoints have empty validation references and a pre-transaction
timestamp; neither field alone is an EOD availability/finality attestation.
The receipt is referenced by the resulting manifest and therefore contains no
resulting dataset/snapshot ID. Admission verifies that association through the
actual pinned manifest and complete relevant fragment membership instead.

Receipt bytes use `ArtifactStore` publication and full-hash verification.
Source objects retain their existing contract/fragment/object identity chain.
Producer/attempt/fence and decision-clock checks belong to operations;
this layer supplies immutable membership and object reads, with no dependency
on operations. Reconstructed quote sessions have no per-row availability
clock or finality receipt, and their synthesized midnight is never substituted.
The current operations preflight verifies pinned identities and candidate
receipt bytes, then refuses: no genuine source/finality validator is installed.

| Condition | Admission outcome |
|---|---|
| Exact pinned members, genuine source completion and finality, and verified publication at or before cutoff | Evidence can admit that session/domain |
| Exact pinned scope, genuine source completion and finality, and verified publication at or before cutoff, with successful exact-scope completion and no output objects | Admit the proven empty domain; missing, unavailable or unstarted source proof refuses |
| Missing proof, unsupported producer, incomplete coverage or ambiguous evidence | Refuse; do not substitute a later/stale session |
| Receipt/object hash, contract, session, membership or producer identity mismatch | Refuse before returning quote rows |
| Publication after cutoff or malformed/ambiguous clock | Refuse; import time cannot repair the evidence |
| Naive, non-canonical, future or contradictory evidence clocks | Refuse; accept only canonical timezone-aware UTC instants |

Legacy files, read only through `legacy_adapter.py` (Tier-2 curated tables,
`panel.parquet`/`tier4_forecasts.parquet`, the model registry, structure/
champion artifacts, the calendar CSV, the chooser pool, the legacy
`SNAPSHOT` file — every path an `engine.paths` constant); provider payloads
for `daily_market` (a caller-injected fetcher's raw bytes, staged as
`RawPayload`); already-committed catalog rows via a `sqlite3.Connection`
plus an `ArtifactStore` for fragment bytes; a `GenericTableCandidate`'s
`ResolvedSnapshot`/`ArtifactStore`/revisions/`CompletedCoverage` (must be
`state == "complete"`); pinned reference inputs from a `legacy_adapter`
accessor only.

## Outputs

`incremental.cache_raw_receipt` accepts an optional caller-owned `fence_check`.
It runs inside the insertion transaction, or a fresh write transaction after
object verification on existing-receipt replay, before returning. Omitted checks
preserve unfenced callers; object I/O remains outside SQL. A refused fence
can leave unreferenced immutable bytes but no new catalog receipt. Existing
identical content retains its original receipt metadata and acquisition time.
The optional `record_receipt(record)` callback runs after fencing and insertion
inside that same transaction, including verified replay. Its failure rolls back
both the new receipt and caller-owned occurrence; raw content identity is unchanged.

**Streaming table writer (`generic_streaming.py`).** A second,
additive route to a table's new dataset version for producers whose output must
not be held in memory. `StreamingTableWriter(store, parent, table_name, attempt_id,
coverage)` accepts rows in batches and `finish()` returns a
`GenericTableCandidate` that the unchanged `commit_generic_table_candidate`
commits.
- *Scope:* a whole-table contract only (`partition_columns == ()`); any other
  contract refuses `UNSUPPORTED_CONTRACT`. The result is one fragment that
  REPLACES the whole table: every partition the parent holds for the table (an
  imported table's single partition is `all`, a generic one `__whole__`) is
  removed, and the new fragment takes the parent's partition key when all prior
  fragments share one, else `__whole__`. No prior rows are read, no per-row
  revisions or `RowChange`s are recorded; the changeset is `changed` for the
  replaced and new partitions, `normalized_payloads` = row count, disposition
  `conservative_full` (no row-level changes recorded).
- *Provenance and coverage:* the new version carries the caller's `knowledge_mode`
  (default `reconstructed`) and `availability_evidence_refs` (default none), only
  the current coverage receipt and the parent version id as lineage, nothing
  inherited from the parent. Another table's coverage refuses `CONTRACT_MISMATCH`.
- *Order and validity:* each accepted row is copied (later caller mutation cannot
  change it) and must satisfy the contract: known columns only, every column present
  unless nullable, no null in a non-nullable column, and a value of the column's
  physical type (float64 finite, timestamps timezone-naive, a missing nullable value
  `None`); primary-key values must be present. The physical-type map
  `ARROW_TYPES` (including `date32`) is defined once in `query` and re-exported by
  `generic_incremental`. Rows arrive in strictly increasing
  primary-key order across and within batches (the caller orders batches). Shape,
  nulls, types and key order are checked at `append_rows`, Arrow conversion at the
  row-group write, `knowledge_mode` and evidence refs (observed and attested modes
  need evidence) at construction. Any violation or
  unconvertible value refuses `CONTRACT_MISMATCH`, never repaired. A staging-file
  I/O failure propagates as an `OSError` after cleanup. `append_rows` after
  `finish`/`abort`/failure refuses `INPUT_CHANGED`.
- *Memory:* one pending buffer of at most `ROW_GROUP_ROWS` (50,000) rows plus the
  caller's batch, and 65,536-row inspection batches at `finish`. Row groups are
  exactly `ROW_GROUP_ROWS` (the last shorter), so the hash depends only on the
  rows; the staged file is streamed into the store and inspected unmaterialised.
- *Failure semantics:*

  | R | Rule |
  | --- | --- |
  | R1 | Contract, coverage (`complete`) and parent table are checked at construction; rows at append; refusals above |
  | R2 | No cache; each writer owns one uniquely named staging file |
  | R3 | Any failure makes the writer terminal (`failed`); a new attempt starts over |
  | R4 | Nothing is visible to the catalog until the one `commit_generic_table_candidate` |
  | R5 | A failure, `abort()` or `__exit__` on an open writer removes the staging file; a crash leaves the staging file in the attempt's staging directory (reclaimed by the attempt-staging policy, not by this writer), and a published-but-uncommitted object stays unreferenced |
  | R6 | The same ordered rows, contract, parent snapshot and coverage id give the same object and dataset version hash; other rows give another |

  Zero rows at `finish` refuse `CONTRACT_MISMATCH` (an empty table is not
  published silently).

**Original panel COPY identity.** Offline size-fold authoring uses `Repository`
and the existing object verifier to bind the named committed snapshot's
`feature_panel` dataset version to one original object and full byte hash.
The application's committed catalog and configured content-addressed store are
the trust anchor; a caller hash is only a matching expectation. Equivalent rows
rewritten as different Parquet bytes do not preserve this identity.
Missing/substituted membership, unsupported panel layout or corrupt bytes refuse
before pinned authoring. No current-head or live legacy path supplies a fallback.
This proves membership and bytes, not the original producer read-set. Imported
completeness flags, empty evidence references and import registration time do not
prove that corrected upstream data reached the panel. No fitting belongs here.

A committed snapshot (new/reused `data_*` rows, an import receipt, and,
unless already at head, a compare-and-swapped `data_snapshot_heads` row —
`SnapshotImportReceipt`); a failed/conflict receipt in its own transaction,
never touching a head; an immutable object plus a `FragmentInspection`; a
re-verified `SnapshotRef`/`ResolvedSnapshot`, never read back from
`data_snapshot_heads` directly; bounded Arrow-batch scan results in one
global primary-key order; typed results (`EarningsEvent`, `ChainSnapshot`,
`PriceSeriesRow`, `DependencyPlan`); a `GenericTableCandidate`/commit result
(new manifest, changeset, receipt, coverage/revision rows — one transaction
with the head CAS; a reused fragment's changeset is reconciled against the
truly-stored dataset-version row — Invariants); a `computed_moves`/
`price_history` ticker partition (whole-partition rewrite — Invariants),
committed by `engine.v2.ops.computed_moves_store.py`/
`.price_history_store.py` (this package supplies only the math and the
`TableContract`); refusals created through `errors.fail` use a
`DataError`/`Problem` from `DATA_FAILURE_CODES`, intended to carry no local
path or row value — not fully enforced today (Invariants). Strict document
decoding can instead raise `DocumentError` directly.

New native `computed_moves` captures publish singleton event objects within each
ticker partition, with inspected key/time bounds and the existing streamed
logical-partition hash. Row values and table contracts are unchanged; object
layout changes physical version identity. Old pinned fragments remain unchanged.
Repository predicates filter decoded rows, not arbitrary bytes within a surviving
object: research must prove object isolation from metadata before reading labels.

**`computed_moves.v3` — point-in-time availability.** `available_as_of_date`
is the calendar day following the close that made `realized_move_pct`
knowable. It is null exactly when `realized_move_pct` is null; null means
unavailable to any decision. Readers must require
`available_as_of_date <= decision_session`.

**`computed_moves.v4` — event-history columns (design #651, slice S2).** Contract
`computed_moves.v4` / `4.0.0` replaces `computed_moves.v3`; v3 snapshots stay pinned
and readable, and there is no migration, dual read or backfill: the standard
refresh rewrites whole ticker partitions under v4 (rebuild over migration).
`quarter_ordinal` is dropped (it has no readers). New columns, stored at build time
and never computed per run by a reader:
- `event_ordinal` int64 not null: 0-based count of every earlier event of the ticker
  in the event list the builder was given (`event_sequence`, `warmup=4`, the legacy
  panel `MIN_HISTORY`), realised or skipped. `admitted` bool not null:
  `event_ordinal >= 4`.
- `implied_move_pct` (meaning changed, same name): the `AsofSeries` value
  `implied_move` at the last `daily_market` row strictly before `event_date` whose
  `src_iv` is non-null (`on_missing="null"`), filled for skipped events too. If that
  anchor's `implied_move` is null or non-finite the result is null; it never falls
  back to an older row. Null when no `src_iv` row precedes the event.
- `mean_prior_or_implied` float64 nullable and `n_prior_implied` int64 not null:
  `strict_prior_mean(population="admitted")` over the stored `implied_move_pct`
  of earlier admitted events, nulls skipped. A non-admitted row, or an admitted row
  with no earlier non-null implied value, is `(null, 0)`.
- The ordinal counts every confirmed event dated before `as_of`, including events
  that precede the first available close; those cannot be bracketed, so they are
  stored as skipped rows (null realised move and `available_as_of_date`) that still
  count in `event_ordinal`. v3 dropped them, so v4 has more rows than v3 for such
  tickers. The v3 admission rule stays: a ticker with fewer than five computable
  events emits no rows. For events inside the price history, realised moves,
  `skipped` and `available_as_of_date` are bit-identical to v3.

| Condition | Outcome |
|---|---|
| duplicate or unsorted event dates, or duplicate or unsorted `daily_market` dates, for a ticker | `build_rows` raises `CONTRACT_MISMATCH`; the refresh fails before any catalog commit, surfaced by the store as `OpsError` `INPUT_CHANGED` (`details.data_code` = the data code) |
| the `daily_market` frame, empty or not, lacks `date`, `implied_move` or `src_iv` | `build_rows` raises `CONTRACT_MISMATCH` (the store's own scan refuses a missing table column first, per the ops doc) |
| no `src_iv` row before an event | `implied_move_pct` null, no contribution to the aggregate |

A refresh over a parent that still holds v3 partitions refuses `VALIDATION_FAILED`
("full rebuild required", ops doc) when it neither rewrites nor attempts every non-v4
partition, or produces no v4 rows at all; v3 records are never carried.

`computed_at` and `retrieved_at` never order or filter anything here. The builder
imports `event_history` from this same package (no new cross-component dependency).

**`event_history.py` — event-history primitives (design #651).** Pure, no I/O,
no clock. A feature declares its parameters; none is hard-coded in the module.
- `event_sequence(event_dates, warmup=, first_known=)` takes one ticker's FULL
  ordered event list. `ordinal` is the 0-based count of every earlier event,
  realised or not; `admitted = ordinal >= warmup`.
- `AsofSeries(rows, eligible=, value=, on_missing=)` anchors on the last row
  with a data date strictly before the event date whose `eligible` fields are
  all non-null (None or float NaN); eligibility is separate from the value
  field, and the lookup never reaches back past that anchor. `max_staleness_days`
  optionally bounds the anchor's age. `on_missing` has no default.
- `strict_prior_mean(events, values, population=)` returns `(mean, count)` per
  event over earlier-ordinal events with a non-null value, within `population`
  (`"admitted"` or `"all"`; a non-admitted event gets `(None, 0)` under
  `"admitted"`). Each mean is the exact rational mean, rounded once to float.
- Dates are canonical `YYYY-MM-DD` data dates only; `retrieved_at` and
  `computed_at` are provenance and never filter or order.
- The module only computes from the inputs it is given; it stores nothing. Its
  one production caller is `computed_moves.build_rows` (`computed_moves.v4`).

| Condition | Outcome |
|---|---|
| no anchor, stale anchor, or null anchor value | `None` under `on_missing="null"`, `CONTRACT_MISMATCH` under `"refuse"` |
| any other malformed input: wrong type, or a container that is not a list or tuple (str, bytes, range, array, generator), non-canonical date, unsorted or duplicate dates, `first_known` not the first event, missing column, negative, non-integer or omitted declaration (`warmup`, `eligible`, `value`, `on_missing`, `population` have no default), unknown `on_missing`/`population`, non-numeric or non-finite value (every value is checked whatever the population or row eligibility; `AsofSeries` checks all rows at construction), length mismatch | `CONTRACT_MISMATCH`, never a bare exception |

## Dependencies

Layer 1.0 has no `only_imports` restriction (root doc §2), but in practice
this package imports only layer 0.0 (`engine.v2.contracts`) and 0.5
(`engine.v2.foundation`) — no module here imports `engine.v2.ops`,
`engine.v2.scoring`, or any other `engine.v2.*` package above its own layer.

`legacy_adapter.py` is the only declared adapter (17 entries): read-only
access to `engine.data.schemas.*`, `engine.data.features.panel/tier4.*`,
`engine.data.store._read_part`, `engine.models.registry.*`, and
`engine.paths.*`. No other module here imports legacy `engine.*`; legacy
never imports this package (root doc §3).

**Callers.** `engine.v2.ops` (7.0) — schema migrations, snapshot commit for
the Phase 1 fence, snapshot import/promotion, nightly barrier read plans,
`daily_market`/`computed_moves`/`price_history` refresh submission (`forward_calendar_store.py`
supplies a candidate `fence_check`; Massive acquisition fences raw receipts). `engine.v2.serving`
(7.0) imports `repository.Repository`. `engine.v2.research` (6.0/7.0) reads
pinned snapshots and calls `generic_incremental.commit_generic_table_candidate`
(no `fence_check`) and `incremental_tables.revision_hash`/`GenericRevision`.
`engine.v2.models.training` uses `errors`, `Repository`, `reference_catalog` and
`computed_moves` for snapshot/receipt-pinned D14 datasets and calendar sessions.
`experiments.v2_magnitude_search_run` consumes `repository.Repository`,
`errors.DataError` and `documents.reject_duplicate_keys` for serial CLI admission.
`tools/*`, `checks/*`, `tests/test_v2_data_*.py` exercise this package
directly.

`engine.v2.features.daily_state_inputs` reads bounded pinned `daily_market`
rows; its EOD source session does not establish receipt-time availability.

## External systems and libraries

`sqlite3` (the shared ops/ledger/data catalog file — `catalog.py` runs its
own local `BEGIN IMMEDIATE`); the local filesystem (legacy Parquet/CSV via
`legacy_adapter.py`, the artifact store's content-addressed backing store,
and one package-resource read for `legacy_mapping.py`'s bundled
annotations file); `pyarrow`/`pyarrow.parquet` (every fragment is Parquet;
commit/scan stream Arrow batches); `numpy`/`pandas` (`computed_moves.py`,
`price_history*.py`). No network access: every provider fetch is injected
by the caller as a plain callable.

## Failure semantics

Refusals created through `errors.fail` are `DataError`s wrapping `Problem`s
built only from a registered `engine.v2.contracts.data.DATA_FAILURE_CODES`
entry — category and retryability come from that table, never guessed at a
call site. Strict document decoding can instead raise `DocumentError`
directly, without conversion to `DataError`.

| Code | Category | Retryable | When |
|---|---|---|---|
| `SNAPSHOT_NOT_FOUND` | dependency | no | unknown `snapshot_id` |
| `SNAPSHOT_UNRESOLVED` | dependency | no | experiment input reader cannot verify the supplied exact committed snapshot |
| `SNAPSHOT_NOT_READY` | dependency | yes | scope has no committed head yet |
| `SNAPSHOT_CONFLICT` | dependency | yes | head-fence or compare-and-swap mismatch |
| `CONTRACT_MISMATCH` | validation | no | a table, pin or selection violates the snapshot's contract, including malformed timestamp key-predicate strings, non-string timestamp predicate values, or predicate scalars incompatible with fragment key bounds; IV-crush rejects malformed/missing projection fields, noncanonical dates, duplicate dates, ticker mismatch, unsupported scalar types, strings over 256 characters, invalid ticker/source text, nonpositive/noninteger row caps or unrepresentable next-day availability |
| `QUERY_NOT_BOUNDED` | validation | no | a malformed planning selection or an unbounded `DataQuery`/`ChainQuery` |
| `RESULT_LIMIT_EXCEEDED` | resource | no | a scan/materialization exceeds the query's `max_result_rows`; either IV-crush event or daily projection exceeds its independent `max_rows` cap (default 10,000), without truncation |
| `RESOURCE_UNAVAILABLE` | resource | yes | no fetcher configured for a refresh |
| `TRANSIENT_SOURCE` | source | yes | provider response neither complete nor a legitimate empty; in `daily_market` a response labeled `complete` that omits an expected key is refused before it is cached, while omissions on a `partial` response after the provider's bounded retry are committed as typed `missing` coverage outcomes |
| `SOURCE_NOT_FINAL` | source | yes | a `daily_market` refresh unit has expected keys but the response is a `legitimate_empty`; the data layer refuses before caching the response, leaving no receipt, coverage, or snapshot write |
| `INPUT_CHANGED` | integrity | yes | coverage incomplete, a candidate built from a now-stale input, expected keys missing/malformed/empty at the normalization boundary, or a planned unit whose `expected_keys` field is missing, malformed or empty refused at refresh acquisition (Invariants) |
| `OBJECT_CORRUPT` | integrity | no | a re-hashed object's bytes disagree with its recorded hash, or the file keeps changing while it is verified |
| `MANIFEST_CORRUPT` | integrity | no | a recomputed manifest/fragment id disagrees with the stored catalog row, or a fragment count is invalid or differs from its footer |
| `MANIFEST_CORRUPT` | integrity | no | a fragment key-bound pair has one missing endpoint, wrong arity/type, nonfinite values or reversed order; all planning paths refuse before pruning, even if the fragment would be excluded |
| `MANIFEST_CORRUPT` | integrity | no | a returned scan key lies outside its admitted full-primary-key range |
| `IDENTITY_CONFLICT` | validation | no | an existing row's payload disagrees with a new one under the same id; also a `daily_market` revision tie (Invariants) |
| `UNSUPPORTED_CONTRACT` | validation | no | an operation on a table contract this code path does not implement |
| `EVENT_NOT_FOUND` | dependency | no | `events.get_event` for an unknown key |
| `DEADLINE_EXCEEDED` | resource | yes | a scan/`explain_dependencies` exceeds its deadline |
| `POPULATION_COLLAPSED` | validation | no | a chain query's candidate population is empty |
| `DEST_ROOT_NOT_EMPTY` / `DEST_ROOT_UNSAFE` | validation | no | legacy materialization destination fails its pre-write check |
| `EVIDENCE_SCOPE_INCOMPLETE` | validation | no | a `trades` scan's real span escapes a too-narrow `evidence_scope` |
| `TIER4_CACHE_STALE` | validation | no | a pinned Tier-4 serving-cache ref's embedded panel hash disagrees with the actual panel object |
| `STALE_EXPECTATION` | validation | no | `explain_dependencies`'s chain-query path sees a stale caller expectation |
| `CALENDAR_UNAVAILABLE` | validation | no | registered here but raised only by `engine.v2.research`, never from inside this package |

**Legacy materialization row validation.** Rewritten curated parts and
single-file outputs are checked against a fresh scan in query order, comparing
the query's selected columns with exact value equality after the existing
null-like and timestamp normalization. A row-count mismatch keeps the
`CONTRACT_MISMATCH` refusal message `materialized row count {actual} != scanned
row count {expected}`; a value mismatch keeps
`materialized value for column {column!r} disagrees with the scanned row`.
Read path: row values are compared by reading the written Parquet part
directly with pyarrow (never through the legacy reader); a rewritten curated
part also reaches the legacy `coerce()` per batch; only byte-copied files, and
a rewritten part with zero rows, are opened with the legacy reader. Validation
compares bounded batches (8,192 rows per side, so peak memory does
not grow with rows per table-year), but no batch is accepted on its own: a
difference in any batch, including the last, refuses the whole
materialization, and a row-count mismatch takes precedence over a value
mismatch. Validation is read-only and idempotent (same parts and snapshot, same
verdict): it changes no materialized byte and no `SNAPSHOT`/receipt content,
keeps no cache of its own and no retry (every call re-scans the snapshot for
rewritten files and re-reads each part), and runs only after every part is
written, with no transaction around it. Byte-copied files are not re-scanned:
a copied curated table is reopened with the legacy reader and passed through
`coerce()` (failure: `CONTRACT_MISMATCH` "legacy coerce() refused ..."), and a
copied single file is only reopened with the legacy reader (failure: the
reader's own error). A `DataError` refusal makes `legacy_adapter.materialize` remove everything
that call wrote (except destination-root safety refusals, which leave the
pre-existing root alone). A missing or unreadable part raises the reader's own
error, never a match; that error is not a `DataError`, so it skips that cleanup
and leaves an unlocked, unreturned partial tree (pre-existing behaviour).

The v2 legacy-part adapter supplies an Arrow-owned, read-only file handle to
the unchanged legacy reader. It preserves that reader's projection, missing
columns, pandas metadata and coercion; no Python-owned file object is retained
by Arrow's asynchronous Parquet machinery during interpreter shutdown.
The handle closes on success or error. Opening a missing/unreadable part
raises the file reader's error; this adds no cache, retry, transaction or write,
and repeated reads retain the same row and materialization identities.

**`daily_market` missing-ticker outcome (R1–R6).** A non-empty 2xx ORATS
response that remains incomplete after the provider's single paired retry is
committable as partial coverage. It does not turn an absent ticker into a
revision or mark the response complete.

| Requirement | Outcome |
|---|---|
| R1 — typed result | Every expected ticker has a `CoverageOutcome`: observed tickers are `present`, each carrying a non-null `revision_id` under both `complete` and `partial` coverage; a `present` outcome without a revision makes coverage `incomplete`, so it cannot advance the snapshot. Omitted tickers are `missing`, keyed by ticker and session date and linked to the response's raw receipt id. |
| R2 — storage and query | The existing `data_snapshot_coverage.coverage_json` stores the typed outcomes and expected denominator. Consumers query by snapshot/table coverage, then select `missing` outcomes; no DDL or new migration is required. |
| R3 — retry/cache | The provider retries the summaries/cores pair once for missing keys; a partial gap stays receipt-backed and is never recorded as `complete`. Before native planning accepts a complete receipt as a cache hit it reconstructs the rows and verifies every expected key; a missing reconstructed key demotes the unit to `fetch_units` so the normal provider-call budget is reserved, and cache-only workers never issue provider calls. |
| R4 — transaction | Returned ticker revisions and the partial coverage record enter the same snapshot candidate and head-CAS commit. A commit refusal leaves the head unchanged. |
| R5 — visible residue | A successful commit contains all returned rows, no fabricated row for a missing ticker, and a queryable gap with session date and raw receipt identity. Empty/not-final responses still refuse under normal source retry semantics. |
| R6 — idempotency/downstream | Deterministic replay holds for a valid complete receipt: it reconstructs the same outcomes and coverage identity. An incomplete reconstruction is skipped at planning and followed by a budgeted fresh acquisition with a new receipt identity. Downstream consumers continue to read the stored coverage state and gap outcomes instead of inferring completeness from rows. |

A cache-only receipt is reusable only when `complete` and its recorded requested `keys`
exactly match the refresh unit's `expected_keys` after string normalization and sorting; a
mismatch or non-complete cached receipt refuses retryable integrity `INPUT_CHANGED` before
reading/staging ticker rows, leaving no coverage or snapshot write; re-running with the
matching receipt/unit is idempotent. Every planned `daily_market` unit must carry a nonempty
sequence of nonempty string ticker keys: an absent, malformed or empty planned key set refuses
retryable integrity `INPUT_CHANGED` before any provider call, cache/store read or write, or key
coercion. For such a valid keyed unit `legitimate_empty` is never an accepted answer — it refuses
retryable source `SOURCE_NOT_FINAL` before a raw receipt, coverage, or snapshot write; the caller's
source retry policy owns retry.

**Target contract: pinned scans and registration (R1–R6).**

| Requirement | Outcome |
|---|---|
| R1 — missing or unsupported input | Missing members raise their typed refusal codes above; unsupported `TableContract` schema versions raise `DocumentError` with code `UNSUPPORTED_VERSION` before rows are returned. |
| R2 — cache | Bounds come from the pinned fragment membership; no current-head fallback or cached bound from another snapshot. |
| R3 — retry | `RESULT_LIMIT_EXCEEDED`, `OBJECT_CORRUPT`, and `MANIFEST_CORRUPT` are non-retryable; `INPUT_CHANGED` is retryable. Registration retries retain the head fence. |
| R4 — transaction | Changed definitions never overwrite registered contracts. |
| R5 — partial result/write | Invalid surviving counts refuse `MANIFEST_CORRUPT` before streams open; footer/count mismatch refuses `MANIFEST_CORRUPT` before rows. Earlier batches are provisional until exhaustion. Failed registration leaves the head unchanged and staged objects unreferenced. |
| R6 — idempotency | Under a pinned snapshot, byte-identical reads return identical rows; scan completion requires exhaustion without an error. An identical registration reuses its receipt through the same head fence; conflicting identity refuses. |

**Snapshot commit (4c R1–R6).** Missing input: `INPUT_CHANGED`/
`CONTRACT_MISMATCH` before any write; every contract/fragment/manifest is
re-verified before the transaction opens. A caller-supplied `fence_check`
on `commit_generic_table_candidate` **composes with, never replaces**, the
built-in head-fence check — built-in first (`SNAPSHOT_CONFLICT` on
mismatch), then the caller's own (e.g. an attempt-lease check) — kept
active even on the replay shortcut below. A direct
`catalog.commit_snapshot` caller (`computed_moves_store.py`,
`price_history_store.py`, `snapshots.commit_snapshot_for_attempt`) gets no
such composition. No cache. No internal retry: a lost CAS or fence refusal
raises immediately. One `BEGIN IMMEDIATE` transaction covers the fence, the
replay lookup, every insert, the receipt, reference-table writes, and the
head CAS — any exception rolls it all back. No partial write: a fragment's
bytes publish *before* the transaction, so an uncommitted candidate's
object is merely unreferenced (content-addressed, reused by a retry).
Idempotency: (1) insert-level payload match (`IDENTITY_CONFLICT` otherwise
— a fragment match ignores its two non-identity provenance columns, so a
corrected re-ingest reuses the existing fragment); (2) a receipt-level
replay shortcut keyed on `request_hash`/`attempt_id`/`fence`/`scope`/
`snapshot_id` — through the composed fence, reachable only for a no-op or
that candidate's own already-applied effect; a direct caller with a
head-blind `fence_check` gets a weaker guarantee (a matching replay returns
the prior receipt even if the head moved since).

**`repository.py` reads (summary).** Missing input maps to the codes above;
a corrupt id is `MANIFEST_CORRUPT`; a re-hash mismatch is `OBJECT_CORRUPT`.
`resolve` rebuilds from catalog rows every call by default; `pin_snapshots=True` reuses per instance, with defensive copies, never caching failures. `objects.verify_object_path`
fully re-hashes an object on its first open in a process and skips the re-hash
on a later open only while the file's stat tuple (device, inode, size, mtime
and ctime, in nanoseconds) equals the one recorded when its hash last matched
and the store root and the object's parent directories are still real
directories (never symlinks);
any drift, a failed verify or a non-regular file forces a full verify, and a
file whose stat tuple is unavailable or changes while it is hashed is
re-verified a bounded number of times, then refused as `OBJECT_CORRUPT`. The
cache is in-memory and per process, bounded (emptied when full), and keyed by
store root, content hash and byte size. Integrity guarantee: every change visible in the stat tuple is
detected on the next open; a tamper that preserves all five fields within one
process lifetime is not (objects are immutable, read-only files, so that needs
out-of-band access to the store). No retry beyond that bounded re-verify, no
partial write (read-only). One read-only transaction covers a
whole `resolve` walk. Idempotent: every row is append-only.

**`data_normalizations` v13 recreate (contract; migration 13 defined by this PR).**
This PR defines `data`-owner migration 13 as a pending step: it recreates
`data_normalizations` through the migration framework's opt-in
foreign-key-off table-recreate procedure and removes exactly one thing —
v10's `UNIQUE (raw_hash, normalizer_id, contract_id)` — changing nothing
else. Existing normalization rows are carried over, `normalization_id`
stays the primary key, `contract_id` continues to reference
`data_contracts`, and the table's other columns, `CHECK`s and immutability
triggers are unchanged; committed
`data_daily_market_revisions.normalization_id` references to
`data_normalizations.normalization_id` survive the rebuild, which is
precisely what the enforcement-off procedure is for. This is a new numbered
step, never an edit of v10: applied steps stay checksum-protected and the
recreate flag itself joins the new step's checksum. This step removes only
that schema-level uniqueness rule, computes no cache identity and rekeys no
stored row. The current identity contract
(Invariants) folds the fetch unit's expected-key set into the writer's
`normalization_id`.

The refusal is the migration framework's typed `OpsError`/`INTEGRITY_FAILED`
integrity failure, not a `DATA_FAILURE_CODES` entry, and the concurrency and
rollback rules below are the framework's: see
[`engine/v2/ops/MIGRATIONS.md`](../ops/MIGRATIONS.md), which governs this step
and is not restated in full here.

| Requirement (framework numbering) | Outcome for migration 13 |
|---|---|
| R1 — ordinary migration | Not this step: it opts into the recreate procedure, so enforcement is off on the migration connection for its duration only. |
| R2 — recreate migration | `PRAGMA foreign_keys = OFF` before `BEGIN IMMEDIATE`; the rebuild statements and the post-rebuild `PRAGMA foreign_key_check` run inside that transaction, before commit. |
| R3 — violation | A dangling reference after the rebuild is a non-retryable integrity failure until the underlying data/schema problem is corrected; a failing rebuild or version-record statement is not retyped here — SQLite errors otherwise propagate. |
| R4 — rollback | Any statement, the FK check or the version record failing rolls back the whole migration: no rebuilt table and no advance of the `data` catalog's recorded versions. |
| R5 — restoration and concurrency | Enforcement returns to `ON` on every exit, including the refusal path. The pragma is per connection, so another connection's setting or open transaction is untouched, though `BEGIN IMMEDIATE` may serialize or reject a concurrent schema write. |
| R6 — retry | Retrying after the cause is corrected reattempts the still-pending step; no internal retry. |

## Invariants

Root doc §5 invariants this package is responsible for:

- **Scan population bound.** A stored `TableContract` carries no generic
  result-row cap; the query's `max_result_rows` is the scan's only result
  limit — explicit caller intent, enforced by a running counter that raises
  `RESULT_LIMIT_EXCEEDED` before yielding a batch that would exceed it, and
  legal below the selected-population bound so a caller can refuse a larger
  population for its own retained-memory or cardinality requirement. The
  manifest footer — the recorded row counts of the pinned fragments surviving
  the pruning predicates, summed by the one shared membership-bound planner —
  is the authoritative selected-population bound, checked before row
  streaming consistently by `scan` and `explain_dependencies`; a request
  limit above it is `QUERY_NOT_BOUNDED`. An empty selected population may
  produce a zero-row query with a positive, contract-bounded batch size.
  Invalid surviving fragment counts are `MANIFEST_CORRUPT` before any stream
  opens; fragment counts bound candidate rows, not the exact
  predicate-matching population or total process memory.
- **Missing input → typed refusal, never a silent default** — every failure
  path raises a `DataError`/`Problem` from the table above.
- **Snapshot/root isolation** — every store path resolves through
  `engine.paths`/`ArtifactStore`, never a locally computed project root
  (`legacy_mapping.py`'s one `Path(__file__)` use resolves a sibling
  package resource, not a project root).
- **Nothing published carries a local path or raw exception text — not
  fully enforced.** `errors.py` performs no automatic redaction; each call
  site is responsible for excluding legacy paths and row-derived values
  from a `Problem`'s `details` itself. `reference_catalog.py` (issue #202)
  and the refusal sites covered by issues #217 and #218 in
  `reference_inputs.py` now redact or drop legacy paths instead of echoing
  them. The `TIER4_CACHE_STALE` refusal in that same file is unchanged and
  still includes a registry-derived value in `details`. `import_snapshot.py`
  and `legacy_materialization.py` still put a raw legacy path in `details`
  for several refusals — tracked as issue
  [#220](https://github.com/yshewchuk/investment-validation/issues/220).
- **Atomic snapshot commit, compare-and-swap head, never last-writer-wins.**
  Zero rows changed on the head update is `SNAPSHOT_CONFLICT`. Only
  `data_snapshot_heads` is mutable; every other `data_*` table is
  append-only. A reused fragment/dataset-version's stored provenance is
  reconciled into any audit row already built from the caller's own,
  possibly stale, copy.
- **Idempotent insert, never overwrite** — an existing row is accepted under
  an id only when its full canonical payload matches.
- **The legacy-touching seam is confined to one module** — a legacy import
  elsewhere, or an undeclared one in `legacy_adapter.py`, fails
  `checks/import_layers.py`.
- **Native vs. legacy provenance** — this package never mints a "native"
  answer from a legacy-derived value; a native verdict is computed at
  `engine/v2/scoring` (layer 5), not here.
- **Whole-partition rewrite, no legacy append order** — a correction rewrites
  the entire ticker partition: `price_history` uses one fragment, while new
  `computed_moves` captures use ordered, non-overlapping singleton event fragments.
  Older snapshots may retain one whole-ticker fragment; neither layout appends
  corrections to the existing fragment membership.
- **The calendar ticker is a declared `price_history` dependency.** The native
  nightly derives its session calendar from the pinned `price_history` series of
  `price_history_table.CALENDAR_TICKER` (`SPY`), independent of any scoring
  scope or board universe. The source feeding capture must therefore carry it:
  `ops price-refresh` schedules it daily and skips it only when a cached fetch
  covers the session (refresh rule: the ops component doc), and
  `ops price-history capture` refuses `SOURCE_NOT_FOUND`
  (details name the ticker) before writing any fragment unless it has a live
  (non-tombstoned) latest-state row: for a ticker the source lists, in the
  capture's resulting state (the prior version is not consulted, so a listed
  but fully tombstoned ticker refuses); for a ticker the source does not list,
  in the prior dataset version. The gap surfaces at capture rather than as a failed native calendar scan
  (`CONTRACT_MISMATCH`, no `price_history` for the ticker) in the nightly.
- **A `price_history` retrieval never drops stored history.** A full-history
  retrieval that omits any currently live stored date (truncated end,
  mid-history gap, or empty), or starts later than the live history, is refused
  whole by `diff_retrieval` with `CONTRACT_MISMATCH`. A later start reports
  `stored_min_date` and `retrieval_min_date`; an omission reports
  `missing_count` plus the missing dates (or `first_missing_date`,
  `last_missing_date` past 20). Nothing from the refused response is written,
  the stored rows stay live, and the next refresh retries. Only live dates
  count: omitting a tombstoned date is not a refusal. `diff_retrieval` never
  emits a tombstone.
- **Registered contract definitions are immutable.** `catalog.commit_snapshot`
  refuses (`IDENTITY_CONFLICT`) a changed definition under an existing
  `contract_id`; a table picks its next `contract_id`/`semantic_version` per
  its own `schema_evolution_policy`.
- **`daily_market` revision identity/ordering.** A revision's id folds in
  its own content hash, so differing content never shares an id. Ranking
  picks the surviving group's highest ordinal (derived from `received_at`,
  floored and bumped past a process-local high-water mark, so two
  revisions from the same process never tie). Two revisions from separate
  processes CAN tie on ordinal; if their content also differs, that tie is
  `IDENTITY_CONFLICT` — an unresolvable ordering ambiguity, never silently
  picked either way.
- **`daily_market` coverage is measured against requested keys**, not what
  came back: after the provider's bounded retry, expected tickers omitted on
  a `partial` response are recorded as typed `missing` outcomes and coverage
  is `partial`; a response labeled `complete` that omits an expected key is
  refused with `TRANSIENT_SOURCE` before caching — never a tautological
  "complete."
- **`daily_market` planned refresh units are validated before
  acquisition.** A planned fetch unit's expected-key set must be a
  nonempty tuple/list of nonempty strings, checked before the provider is
  invoked, before any cache work, and before any coercion — for fetched
  units and for cached units reconstructed from a caller-supplied plan
  alike. A planned unit with an absent `expected_keys` field is invalid, and a
  missing, malformed or empty planned set refuses with the registered
  retryable `INPUT_CHANGED` before any provider call, cache/store work,
  receipt/cache write or key coercion — never a `str()`-coerced member, never a
  silent empty set. Valid
  keys are preserved as strings on the way to the receipt request and the
  normalization boundary below.
- **`daily_market` normalizer versioning.** `cache_normalization` keys
  `normalization_id` on a canonical hash of the fetch unit's expected-key set
  folded together with `raw_hash`, `normalizer_id` and `contract_id`; expected
  keys are the canonical set of nonempty string ticker keys: order and
  duplicates never change the hash. At the normalization boundary the
  expected-key sequence must be nonempty and every member must be a nonempty
  string ticker key; a raw receipt needed for normalization carries that
  sequence specifically as a nonempty `request.keys` list. A missing, null,
  non-list or malformed saved list, or an empty or malformed sequence passed
  directly, fails closed with the registered retryable `INPUT_CHANGED` before
  any normalization identity is derived or written — never a coerced member,
  never a silent empty set. The same raw payload and normalizer under different
  expected-key sets therefore produce distinct normalization identities and
  rows; repeated requests with the same set stay idempotent. A payload
  conflict under the same resulting identity still refuses with non-retryable
  `IDENTITY_CONFLICT`; distinct expected-key sets must never surface a raw
  SQLite uniqueness `IntegrityError`. `normalizer_id` must be bumped in the
  same PR as any change to what a normalized document contains for the same
  raw input, or an old-mapping session replays unchanged under a shared
  `raw_hash`. A row stored under the old triple-only formula carries no
  expected-key metadata, so it is never safely reused as a cache hit: it and
  its references stay unchanged and addressable by their stored legacy id,
  and the next request writes/uses a new expected-set-scoped id — no rekey or
  delete migration is performed. The v13 recreate (Failure semantics) dropped
  only that tuple's database-level `UNIQUE` and rewrote no row's id; this
  expected-key-scoped identity work is tracked as issue
  [#133](https://github.com/yshewchuk/investment-validation/issues/133).
- **`daily_market` mcap carry-forward is scoped to loaded partitions** — a
  winner row's null `mcap_usd` is backfilled only from an observation
  already loaded in this build, never by scanning further back; deliberate,
  not a defect (cross-year carry tracked as issue
  [#195](https://github.com/yshewchuk/investment-validation/issues/195)).
