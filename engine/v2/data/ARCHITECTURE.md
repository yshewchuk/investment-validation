# `engine/v2/data` — architecture

Layer **1.0** in the root [`ARCHITECTURE.md`](../../../ARCHITECTURE.md)'s
layer table (§2). Replaces legacy's `data/sources/`, `store.py`, `fetch.py`,
`finality.py`, `rebuild.py`, and calendar sourcing from `calendar.py`. See
`engine/v2/data/README.md` for the checker-enforced Public interface /
Consumers lists; a caller reaching a name via a submodule-qualified import
(`from engine.v2.data.X import Y`) need not appear in that directive, so it
is narrower than "everything this package exposes."

## Purpose

Ingestion, normalization, coverage/finality, and atomic snapshot commit. It
owns: turning a raw legacy file or provider payload into an immutable,
content-addressed object and a validated fragment/dataset-version/snapshot
identity chain (`objects.py`, `manifests.py`); committing a fully-built
snapshot atomically with idempotent inserts and a compare-and-swap head,
never last-writer-wins (`catalog.py`); the incremental-refresh merge policy
shared by every EOD table family (`incremental_tables.py`), `daily_market`'s
own adapter (`incremental.py`), and the same protocol generalized to any
registered `TableContract` (`generic_incremental.py`); bounded, exact reads
over an already-committed snapshot (`repository.py`, `query.py`,
`events.py`, `chains.py`, `price_history_query.py`); the one legacy-touching
seam (`legacy_adapter.py`) and what is built on it read-only — legacy→v2
table mapping, pinned reference-input resolution, snapshot-import planning,
and legacy-tree materialization for a nightly's barrier-only stages; two
natively-computed, legacy-free table families with no legacy Tier-2 backing
— `computed_moves*` (pure close-to-close move math) and the bitemporal
`price_history*` diff/as-of logic; the data-owner catalog schema
(`schema.py`, colocated with the ops/ledger schemas in one SQLite file under
migration owner `"data"`); pure completeness/coverage checks used by
incremental planning (`eod_inventory.py`, `event_revisions.py`); and a
Tier-4 serving-cache coverage check (`tier4_coverage.py` — not pure, not
legacy-free) gating a snapshot-backed scoring launch from `engine.v2.ops`.

Non-responsibilities: never computes a trading verdict (`engine/v2/scoring`)
or promotes a champion (`engine/v2/models/training`); never decides *what* to
fetch or *when* to run a job (`engine/v2/ops`, a caller, owns that).

## Primary contracts and public interfaces

- **Snapshot commit** — `catalog.commit_snapshot`/`.record_failed_import`/
  `.move_head`: the one place a snapshot's rows, membership and head move
  together inside one transaction (Failure semantics).
- **Contract-generic incremental commit** — `generic_incremental.
  build_generic_table_candidate`/`.commit_generic_table_candidate`/
  `.load_generic_revisions` wrap `catalog.commit_snapshot` with a head-fence
  expectation and an optional caller `fence_check` (Failure semantics).
- **`daily_market` incremental refresh** — `incremental.py`'s
  run/build/commit/merge functions, and shared merge primitives one layer
  down in `incremental_tables.py` (`merge_table_rows`, `revision_hash`,
  `GenericRevision`/`GenericMerge` — also used directly by
  `engine.v2.research` and `engine.v2.ops.forward_calendar_store`). See
  Invariants for revision identity/ordering, `normalizer_id` versioning, the
  coverage-completeness rule, and the mcap carry-forward scope limit.
- **Snapshot resolution and bounded reads** — `repository.Repository(conn,
  store=None)`: exact re-verifying `resolve`/`resolve_full` (+ `_pinned`
  variants), a bounded Arrow `scan`, typed `get_event`/`get_chain`/
  `get_price_series`/`get_close`, and `explain_dependencies` (names the
  exact snapshot/dataset-versions/fragments/columns/predicates a query
  would touch).
- **Pure primitives, no I/O** — `query.py`, `manifests.py`, `objects.py`,
  `documents.py`.
- **Legacy-touching seam** — `legacy_adapter.py` is the package's only
  module importing legacy `engine.*` code (17 declared, read-only entries in
  `checks/legacy_adapters.json`). Built on it, read-only: `legacy_mapping.py`
  (table mapping); `legacy_materialization.py`/`legacy_nightly_read_plan.py`
  (materialization and nightly barrier read plans — neither imports legacy
  `engine.*` directly); `reference_inputs.py`/`reference_catalog.py`
  (pinned reference-input resolution — the latter is itself legacy-free, so
  `catalog.commit_snapshot` never loads a legacy module through this path);
  `import_snapshot.py` (snapshot-import planning); `tier4_coverage.py`
  (Tier-4 serving-cache coverage — store-reading, not pure).
- **`computed_moves`/`price_history` table families** — pure math + each
  family's `TableContract`; `price_history_query.py` is also exposed as
  `Repository.get_price_series`/`.get_close`.
- **Errors** — `errors.DataError`, built only from a registered
  `DATA_FAILURE_CODES` entry (an unregistered code raises `ValueError` at
  construction, never mis-categorized at a call site).

The exhaustive name-by-name list (which functions each module above
actually exports) is `README.md`'s checker-enforced Public interface
section, not repeated here.

## Inputs

Legacy files, read only through `legacy_adapter.py` (Tier-2 curated tables,
`panel.parquet`/`tier4_forecasts.parquet`, the model registry, structure/
champion artifacts, the calendar CSV, the chooser pool, the legacy
`SNAPSHOT` file — every path an `engine.paths` constant, never computed
locally); provider payloads for `daily_market` (a caller-injected fetcher's
raw bytes, staged as `RawPayload`); already-committed catalog rows via a
`sqlite3.Connection` plus an `ArtifactStore` for fragment bytes; a
`GenericTableCandidate`'s `ResolvedSnapshot`/`ArtifactStore`/revisions/
`CompletedCoverage` (must be `state == "complete"`); pinned reference inputs,
every path from a `legacy_adapter` accessor.

## Outputs

A committed snapshot (new/reused `data_*` rows, an import receipt, and,
unless already at head, a compare-and-swapped `data_snapshot_heads` row —
`SnapshotImportReceipt`); a failed/conflict receipt in its own transaction,
never touching a head; an immutable object plus a `FragmentInspection`; a
re-verified `SnapshotRef`/`ResolvedSnapshot`, never read back from
`data_snapshot_heads` directly; bounded Arrow-batch scan results in one
global primary-key order (never a whole-table read); typed results
(`EarningsEvent`, `ChainSnapshot`, `PriceSeriesRow`, `DependencyPlan`); a
`GenericTableCandidate`/commit result (new manifest, changeset, receipt,
coverage/revision rows — one transaction with the head CAS; a reused
fragment's changeset is reconciled against the truly-stored dataset-version
row, never left citing stale provenance — Invariants); a `computed_moves`/
`price_history` fragment per ticker (whole-partition rewrite, never a byte
append — Invariants), committed by
`engine.v2.ops.computed_moves_store.py`/`.price_history_store.py` (this
package supplies only the math and the `TableContract`); refusals as a
`DataError`/`Problem` from `DATA_FAILURE_CODES` — never a local path or row
value, though a column/table name may appear.

## Dependencies

Layer 1.0 has no `only_imports` restriction (root doc §2), but in practice
this package imports only layer 0.0 (`engine.v2.contracts`) and 0.5
(`engine.v2.foundation`) — no module here imports `engine.v2.ops`,
`engine.v2.scoring`, or any other `engine.v2.*` package above its own layer.

`legacy_adapter.py` is the only declared adapter (17 entries): read-only
access to `engine.data.schemas.*`, `engine.data.features.panel.
PANEL_COLUMNS`, `engine.data.features.tier4.*`,
`engine.data.store._read_part`, `engine.models.registry.*`, and
`engine.paths.*`. No other module here imports legacy `engine.*`; legacy
never imports this package (root doc §3).

**Callers.** `engine.v2.ops` (7.0) — schema migrations, snapshot commit for
the Phase 1 fence, snapshot import/promotion, nightly barrier read plans,
`daily_market`/`computed_moves`/`price_history` refresh submission (only
`forward_calendar_store.py` supplies a `fence_check`). `engine.v2.serving`
(7.0) imports `repository.Repository` (an ordinary downward import).
`engine.v2.research` (6.0/7.0) reads pinned snapshots and calls
`generic_incremental.commit_generic_table_candidate` (no `fence_check`) and
`incremental_tables.revision_hash`/`GenericRevision`. `tools/*`, `checks/*`,
and `tests/test_v2_data_*.py` exercise this package's modules directly.

## External systems and libraries

`sqlite3` (the shared ops/ledger/data catalog file — `catalog.py` runs its
own local `BEGIN IMMEDIATE`, since it cannot import the `engine.v2.ops`
one); the local filesystem (legacy Parquet/CSV via `legacy_adapter.py`, the
artifact store's content-addressed backing store, and one package-resource
read for `legacy_mapping.py`'s bundled annotations file); `pyarrow`/
`pyarrow.parquet` (every fragment is Parquet; commit/scan stream Arrow
batches); `numpy`/`pandas` (`computed_moves.py`, `price_history*.py`). No
network access: every provider fetch is injected by the caller as a plain
callable.

## Failure semantics

Every refusal is a `DataError` wrapping a `Problem` built only from a
registered `engine.v2.contracts.data.DATA_FAILURE_CODES` entry — category
and retryability come from that table, never guessed at a call site.

| Code | Category | Retryable | When |
|---|---|---|---|
| `SNAPSHOT_NOT_FOUND` | dependency | no | unknown `snapshot_id` |
| `SNAPSHOT_NOT_READY` | dependency | yes | scope has no committed head yet |
| `SNAPSHOT_CONFLICT` | dependency | yes | head-fence or compare-and-swap mismatch |
| `CONTRACT_MISMATCH` | validation | no | table/column absent from a snapshot or contract |
| `QUERY_NOT_BOUNDED` | validation | no | an unbounded `DataQuery`/`ChainQuery` |
| `RESULT_LIMIT_EXCEEDED` | resource | no | a scan/materialization exceeds its row limit |
| `RESOURCE_UNAVAILABLE` | resource | yes | no fetcher configured for a refresh |
| `TRANSIENT_SOURCE` | source | yes | provider response neither complete nor a legitimate empty (includes a `daily_market` response missing an expected ticker — a genuine partial, never masked as complete) |
| `INPUT_CHANGED` | integrity | yes | coverage incomplete, or a candidate built from a now-stale input |
| `OBJECT_CORRUPT` | integrity | no | a re-hashed object's bytes disagree with its recorded hash |
| `MANIFEST_CORRUPT` | integrity | no | a recomputed manifest/fragment id disagrees with the stored catalog row |
| `IDENTITY_CONFLICT` | validation | no | an existing row's stored payload disagrees with the new one under the same id — for a `daily_market` revision, only two candidates from separate processes/attempts allocating the same content-addressed id (an unresolvable ordering ambiguity); a same-process correction gets a distinct id/ordinal instead (Invariants) |
| `UNSUPPORTED_CONTRACT` | validation | no | an operation on a table contract this code path does not implement |
| `EVENT_NOT_FOUND` | dependency | no | `events.get_event` for an unknown key |
| `DEADLINE_EXCEEDED` | resource | yes | a scan/`explain_dependencies` exceeds its deadline |
| `POPULATION_COLLAPSED` | validation | no | a chain query's candidate population is empty |
| `DEST_ROOT_NOT_EMPTY` / `DEST_ROOT_UNSAFE` | validation | no | legacy materialization destination fails its pre-write check |
| `EVIDENCE_SCOPE_INCOMPLETE` | validation | no | a `trades` scan's real span escapes a too-narrow `evidence_scope` |
| `TIER4_CACHE_STALE` | validation | no | a pinned Tier-4 serving-cache ref's embedded panel hash disagrees with the actual panel object |
| `STALE_EXPECTATION` | validation | no | `explain_dependencies`'s chain-query path sees a stale caller expectation |
| `CALENDAR_UNAVAILABLE` | validation | no | registered here but raised only by `engine.v2.research`, never from inside this package |

**Snapshot commit — fence, transaction, idempotency (4c R1–R6).** Missing
input: `INPUT_CHANGED`/`CONTRACT_MISMATCH` as above, before any write; every
contract/fragment/manifest is re-verified before the transaction opens. A
caller-supplied `fence_check` on `commit_generic_table_candidate` **composes
with, never replaces**, the built-in head-fence check: the built-in check
always runs first (`SNAPSHOT_CONFLICT` on mismatch), then the caller's own
check (e.g. an attempt-lease check) — kept active even on the replay
shortcut below, where the built-in check alone would otherwise be skipped. A
caller invoking `catalog.commit_snapshot` directly (bypassing
`commit_generic_table_candidate` — `computed_moves_store.py`,
`price_history_store.py`, `engine.v2.ops.snapshots.
commit_snapshot_for_attempt`) gets no such composition, only whatever
`fence_check` it supplies itself. No cache. No internal retry: a lost CAS or
fence refusal raises immediately; retry policy is the caller's. One `BEGIN
IMMEDIATE` transaction covers the fence, the replay lookup, every insert,
the receipt, any reference-table writes, and the head CAS — any exception
rolls it all back. No partial write: a fragment's bytes publish to the
artifact store *before* the transaction, so an uncommitted candidate's
object is merely unreferenced (content-addressed, so a retry reuses the
same id). Idempotency has two layers: an insert-level payload match
(`IDENTITY_CONFLICT` otherwise — a fragment match ignores the two
non-identity provenance columns `manifests.fragment_record` already
excludes, so a corrected re-ingest reuses the existing fragment and its
committed provenance), and a receipt-level replay shortcut keyed on
`request_hash`/`attempt_id`/`fence`/`scope`/resulting `snapshot_id` —
through the composed fence, reachable only for a no-op original commit or
that same candidate's own already-applied effect; a genuinely conflicting
retry is refused with `SNAPSHOT_CONFLICT` first. A direct
`catalog.commit_snapshot` caller with a head-blind `fence_check` gets a
weaker guarantee: a matching replay returns the prior receipt even if the
head has since moved.

**`repository.py` read paths (summary).** Missing input maps to the
dependency/validation codes above; a corrupt id (rebuilt through
`commit_snapshot`'s own builders) is `MANIFEST_CORRUPT`; a re-hash mismatch
is `OBJECT_CORRUPT`. No cache — `resolve` rebuilds from catalog rows every
call, and `objects.verify_object_path` re-hashes on every open
unconditionally (no stat-tuple cache yet: issue
[#194](https://github.com/yshewchuk/investment-validation/issues/194)). No
retry, no partial write (read-only). One read-only transaction covers a
whole `resolve` walk, so a concurrent commit can never hand back a torn
snapshot. Idempotent: every row is append-only, so resolving the same
`snapshot_id` twice returns identical results.

## Invariants

Root doc §5 invariants this package is responsible for:

- **Missing input → typed refusal, never a silent default.** Every failure
  path raises a `DataError`/`Problem` from the table above; nothing defaults
  a missing row, column or coverage state to `0`/`None`/an inferred value.
- **Snapshot/root isolation.** Every store path resolves through
  `engine.paths` (via `legacy_adapter.py`) or `ArtifactStore`, never a
  locally computed project root (`legacy_mapping.py`'s one `Path(__file__)`
  use resolves a sibling package resource, not a project root).
- **Nothing published carries a local path or raw exception text** —
  `errors.py` redacts messages on the way in; a column/table name may appear.
- **Atomic snapshot commit, compare-and-swap head, never last-writer-wins.**
  Zero rows changed on the head update is `SNAPSHOT_CONFLICT`, never a
  retried blind write. Only `data_snapshot_heads` is mutable; every other
  `data_*` table is append-only by schema trigger. A reused fragment/
  dataset-version's stored provenance is reconciled into any audit row
  (a `GenericTableCandidate`'s changeset) that had already been built from
  the caller's own, possibly stale, copy — never left citing provenance the
  catalog does not actually store.
- **Idempotent insert, never overwrite** — an existing row is accepted under
  an id only when its full canonical payload matches; `IDENTITY_CONFLICT`
  otherwise.
- **The legacy-touching seam is confined to one module** — adding a legacy
  import anywhere else, or to `legacy_adapter.py` without a matching
  declared entry, fails `checks/import_layers.py`.
- **Native vs. legacy provenance** — this package never mints a "native"
  answer from a legacy-derived value under a native label; a native verdict
  is computed at `engine/v2/scoring` (layer 5), not here.
- **Whole-partition rewrite for a table with no legacy append order** —
  `price_history_table.py`/`computed_moves_table.py` each cover one ticker's
  whole history in one fragment, so a correction rewrites that fragment
  rather than appending a byte, since the catalog's non-overlapping-
  fragment-range invariant cannot otherwise accept a correction to an
  already-committed key range.
- **`daily_market` revision identity is content-addressed, ordinal is
  process-monotonic.** A revision's id folds in a hash of its own content,
  so a corrected fetch gets a distinct id instead of colliding with the
  retained one; its ordinal derives from the fetch's `received_at`, treated
  as a floor and bumped past a process-local high-water mark whenever it
  would otherwise tie or go backward, so two revisions built by the same
  process always rank unambiguously. Two genuinely separate processes can
  still allocate the same ordinal for the same logical key — refused as
  `IDENTITY_CONFLICT`, an unresolvable ambiguity, not silently picked either
  way.
- **`daily_market` coverage completeness is measured against what was
  requested, never against what came back.** A unit's expected key set comes
  from its own request, independent of the returned rows, so a provider
  response missing an expected ticker is a genuine, detectable gap
  (`TRANSIENT_SOURCE`) rather than a tautological "complete" against
  whatever it happened to return.
- **`daily_market` normalizer versioning.** `cache_normalization` keys a
  cached normalized document on `(raw_hash, normalizer_id, contract_id)` and
  returns a hit unchanged; `normalizer_id` must be bumped in the same PR as
  any change to what a normalized document contains for the same raw input
  (a row-mapping change or a candidate-shape change alike), or a session
  cached under the old id replays unchanged under a `raw_hash` it shares
  with the new one. Normalization identity does not yet fold in a fetch
  unit's own expected-key set, so two fetches of the same raw payload under
  different context-ticker universes can still collide; closing that gap
  needs a migration-framework change and is tracked as issue
  [#133](https://github.com/yshewchuk/investment-validation/issues/133).
- **`daily_market` mcap carry-forward is scoped to loaded partitions** —
  `merge_daily_market` backfills a winner row's null `mcap_usd` only from an
  earlier observation already loaded in this build's own partitions, never
  by scanning one it did not load; a ticker outside that set keeps
  `mcap_usd = None` until a wider fetch loads it — deliberate, not a defect
  (cross-year carry tracked as issue
  [#195](https://github.com/yshewchuk/investment-validation/issues/195)).
