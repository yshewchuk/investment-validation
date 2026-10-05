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
`generic_incremental.py` for any registered `TableContract`); bounded,
exact reads over a committed snapshot (`repository.py`, `query.py`,
`events.py`, `chains.py`, `price_history_query.py`); the one legacy-touching
seam (`legacy_adapter.py`) and what is built on it read-only (table
mapping, pinned reference-input resolution, snapshot-import planning,
legacy-tree materialization); two legacy-free table families
(`computed_moves*`, bitemporal `price_history*`); the data-owner catalog
schema (`schema.py`, colocated with ops/ledger under migration owner
`"data"`); pure coverage checks (`eod_inventory.py`, `event_revisions.py`);
and a Tier-4 serving-cache coverage check (`tier4_coverage.py` — not pure,
not legacy-free) gating a scoring launch from `engine.v2.ops`.

Non-responsibilities: never computes a trading verdict (`engine/v2/scoring`)
or promotes a champion (`engine/v2/models/training`); never decides *what* to
fetch or *when* to run a job (`engine/v2/ops`, a caller, owns that).

## Primary contracts and public interfaces

The exhaustive name-by-name list is `README.md`'s checker-enforced Public
interface section; this names only the load-bearing entry points.

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
  exact re-verifying `resolve`/`resolve_full` (+ `_pinned`), a bounded
  Arrow `scan`, typed `get_event`/`get_chain`/`get_price_series`/
  `get_close`, and `explain_dependencies`.
  Metadata-only `scan_population_bound` sums recorded counts of surviving
  fragments of the supplied snapshot, with no head fallback or object reads,
  through the same membership-bound planner behind `scan` and
  `explain_dependencies`. That summed manifest-footer count is the
  authoritative selected-population bound: both paths check a request's
  `max_result_rows` against it before row streaming. A query-level limit
  stays explicit caller intent — legal below the bound, refused
  `QUERY_NOT_BOUNDED` above it.
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
  bounds eagerly, once, at compile time, and returns a closure with no
  further normalization in its per-row path — the closure is only valid
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
- **Legacy-touching seam** — `legacy_adapter.py`, the package's only module
  importing legacy `engine.*` code (17 declared, read-only entries). Built
  on it, read-only: `legacy_mapping.py` (table mapping);
  `legacy_materialization.py`/`legacy_nightly_read_plan.py` (materialization
  and nightly barrier plans, neither importing legacy `engine.*` directly);
  `reference_inputs.py`/`reference_catalog.py` (pinned reference-input
  resolution, the latter legacy-free); `import_snapshot.py`; `tier4_coverage.py`.
- **`computed_moves`/`price_history`** — pure math + each family's
  `TableContract`; `price_history_query.py` also exposed as
  `Repository.get_price_series`/`.get_close`.
- **Errors** — `errors.DataError`, built only from a registered
  `DATA_FAILURE_CODES` entry.
- **Neutral snapshot inventory** — `tools.reregister_snapshot.neutral_inventory`
  accepts explicit source snapshot, receipt, scope and generation pins and returns
  deterministic table/object membership, partition/count/bound metadata,
  reference bindings and native capture history. It never parses contract
  documents or recomputes native values. Price-history captures follow the pinned
  receipt lineage; computed-moves captures retain their existing contract-wide
  scope. One read transaction pins the inventory; callers own supported source
  identity validation, object-byte verification, publication, and computing
  the canonical inventory hash (`engine.v2.foundation.content_hash` over the
  returned payload) and comparing that hash with the exported expectation to
  detect drift. Internally the tool builds every typed refusal through
  `engine.v2.data.errors` and `engine.v2.contracts.data.DATA_FAILURE_CODES`:
  its dependency on the data error catalog is part of this contract.

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
`price_history` fragment per ticker (whole-partition rewrite — Invariants),
committed by `engine.v2.ops.computed_moves_store.py`/
`.price_history_store.py` (this package supplies only the math and the
`TableContract`); refusals as a `DataError`/`Problem` from
`DATA_FAILURE_CODES`, intended to carry no local path or row value — not
fully enforced today (Invariants).

**`computed_moves.v3` — point-in-time availability.** `available_as_of_date`
is the calendar day following the close that made `realized_move_pct`
knowable. It is null exactly when `realized_move_pct` is null; null means
unavailable to any decision. Readers must require
`available_as_of_date <= decision_session`.

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
`daily_market`/`computed_moves`/`price_history` refresh submission (only
`forward_calendar_store.py` supplies a `fence_check`). `engine.v2.serving`
(7.0) imports `repository.Repository`. `engine.v2.research` (6.0/7.0) reads
pinned snapshots and calls `generic_incremental.commit_generic_table_candidate`
(no `fence_check`) and `incremental_tables.revision_hash`/`GenericRevision`.
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

Every refusal is a `DataError` wrapping a `Problem` built only from a
registered `engine.v2.contracts.data.DATA_FAILURE_CODES` entry — category
and retryability come from that table, never guessed at a call site.

| Code | Category | Retryable | When |
|---|---|---|---|
| `SNAPSHOT_NOT_FOUND` | dependency | no | unknown `snapshot_id` |
| `SNAPSHOT_NOT_READY` | dependency | yes | scope has no committed head yet |
| `SNAPSHOT_CONFLICT` | dependency | yes | head-fence or compare-and-swap mismatch |
| `CONTRACT_MISMATCH` | validation | no | a table, pin or selection violates the snapshot's contract, including malformed timestamp key-predicate strings, non-string timestamp predicate values, or predicate scalars incompatible with fragment key bounds |
| `QUERY_NOT_BOUNDED` | validation | no | a malformed planning selection or an unbounded `DataQuery`/`ChainQuery` |
| `RESULT_LIMIT_EXCEEDED` | resource | no | a scan/materialization exceeds the query's `max_result_rows` |
| `RESOURCE_UNAVAILABLE` | resource | yes | no fetcher configured for a refresh |
| `TRANSIENT_SOURCE` | source | yes | provider response neither complete nor a legitimate empty; in `daily_market` a response labeled `complete` that omits an expected key is refused before it is cached, while omissions on a `partial` response after the provider's bounded retry are committed as typed `missing` coverage outcomes |
| `SOURCE_NOT_FINAL` | source | yes | a `daily_market` refresh unit has expected keys but the response is a `legitimate_empty`; the data layer refuses before caching the response, leaving no receipt, coverage, or snapshot write |
| `INPUT_CHANGED` | integrity | yes | coverage incomplete, a candidate built from a now-stale input, expected keys missing/malformed/empty at the normalization boundary, or a planned unit whose `expected_keys` field is missing, malformed or empty refused at refresh acquisition (Invariants) |
| `OBJECT_CORRUPT` | integrity | no | a re-hashed object's bytes disagree with its recorded hash, or the file keeps changing while it is verified |
| `MANIFEST_CORRUPT` | integrity | no | a recomputed manifest/fragment id disagrees with the stored catalog row, or a fragment count is invalid or differs from its footer |
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
| R1 — missing or unsupported input | Missing members refuse with the codes above; snapshots containing an unsupported contract schema refuse `UNSUPPORTED_CONTRACT` before rows are returned. |
| R2 — cache | Bounds come from the pinned fragment membership; no current-head fallback or cached bound from another snapshot. |
| R3 — retry | No internal scan retry; integrity and result-limit refusals require corrected inputs. Registration retries retain the head fence. |
| R4 — transaction | Re-registration commits complete new identities and the head CAS atomically; changed definitions never overwrite registered contracts. |
| R5 — partial result/write | Invalid surviving counts refuse `MANIFEST_CORRUPT` before streams open. A fragment footer count differing from its recorded count refuses `MANIFEST_CORRUPT` before that fragment yields rows. Earlier streamed batches may already have been consumed; they are not a successful complete result. Failed registration leaves the head unchanged and staged objects unreferenced. |
| R6 — idempotency | An identical registration request reuses its committed receipt through the same head fence; a conflicting identity refuses. Scan completion requires exhaustion without an error. |

For neutral inventory reads, missing relational members or invalid receipt
lineage refuse `INPUT_CHANGED`; inconsistent fragment metadata or row counts
refuse `MANIFEST_CORRUPT`. Retryability follows the table above. There is no
automatic retry or cached inventory and no catalog writes or artifact output.
An active caller transaction refuses `INPUT_CHANGED` without altering it.
The same pins and metadata yield the same inventory; the caller-owned
canonical `content_hash` of the payload, compared with the exported
expectation, detects drift, including changed table membership.

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
`resolve` rebuilds from catalog rows every call (no cache). `objects.verify_object_path`
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
- **Whole-partition rewrite, no legacy append order** — `price_history_table.py`/
  `computed_moves_table.py` each cover one ticker's whole history in one
  fragment, so a correction rewrites it rather than appending a byte.
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
