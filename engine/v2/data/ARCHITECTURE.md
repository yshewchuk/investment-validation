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
  `normalizer_id` versioning, coverage completeness, and mcap carry-forward.
- **Snapshot resolution and bounded reads** — `repository.Repository`:
  exact re-verifying `resolve`/`resolve_full` (+ `_pinned`), a bounded
  Arrow `scan`, typed `get_event`/`get_chain`/`get_price_series`/
  `get_close`, and `explain_dependencies`.
- **Pure primitives, no I/O** — `query.py` and `documents.py` (`manifests.py`
  and `objects.py` are identity builders, not pure: `manifests.
  verify_partition_hashes` calls `objects.partition_logical_hash`, which
  opens and streams object bytes through an `ArtifactStore`, and `objects.py`
  also holds `publish_legacy_file`/`inspect_fragment`, which do filesystem
  I/O).
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
The operations verifier owns producer/attempt/fence and decision-clock checks;
this layer supplies immutable membership and object reads, with no dependency
on operations. Reconstructed quote sessions have no per-row availability
clock or finality receipt, and their synthesized midnight is never substituted.

| Condition | Admission outcome |
|---|---|
| Exact pinned members, genuine source finality and verified publication before cutoff | Evidence can admit that session/domain |
| Exact pinned scope, genuine source finality and verified publication before cutoff, with successful exact-scope completion and no output objects | Admit the proven empty domain; missing, unavailable or unstarted source proof refuses |
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
| `CONTRACT_MISMATCH` | validation | no | table/column absent from a snapshot or contract |
| `QUERY_NOT_BOUNDED` | validation | no | an unbounded `DataQuery`/`ChainQuery` |
| `RESULT_LIMIT_EXCEEDED` | resource | no | a scan/materialization exceeds its row limit |
| `RESOURCE_UNAVAILABLE` | resource | yes | no fetcher configured for a refresh |
| `TRANSIENT_SOURCE` | source | yes | provider response neither complete nor a legitimate empty (a `daily_market` response missing an expected ticker counts as partial) |
| `INPUT_CHANGED` | integrity | yes | coverage incomplete, or a candidate built from a now-stale input |
| `OBJECT_CORRUPT` | integrity | no | a re-hashed object's bytes disagree with its recorded hash |
| `MANIFEST_CORRUPT` | integrity | no | a recomputed manifest/fragment id disagrees with the stored catalog row |
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
No cache — `resolve` rebuilds from catalog rows every call, and
`objects.verify_object_path` re-hashes on every open unconditionally (no
stat-tuple cache yet: issue
[#194](https://github.com/yshewchuk/investment-validation/issues/194)). No
retry, no partial write (read-only). One read-only transaction covers a
whole `resolve` walk. Idempotent: every row is append-only.

## Invariants

Root doc §5 invariants this package is responsible for:

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
- **`daily_market` revision identity/ordering.** A revision's id folds in
  its own content hash, so differing content never shares an id. Ranking
  picks the surviving group's highest ordinal (derived from `received_at`,
  floored and bumped past a process-local high-water mark, so two
  revisions from the same process never tie). Two revisions from separate
  processes CAN tie on ordinal; if their content also differs, that tie is
  `IDENTITY_CONFLICT` — an unresolvable ordering ambiguity, never silently
  picked either way.
- **`daily_market` coverage is measured against what was requested**, not
  what came back — a response missing an expected ticker is a genuine,
  detectable `TRANSIENT_SOURCE` gap, never a tautological "complete."
- **`daily_market` normalizer versioning.** `cache_normalization` keys on
  `(raw_hash, normalizer_id, contract_id)`; `normalizer_id` must be bumped
  in the same PR as any change to what a normalized document contains for
  the same raw input, or an old-mapping session replays unchanged under a
  shared `raw_hash`. It does not yet fold in a fetch unit's own
  expected-key set, so two fetches of the same payload under different
  context universes can still collide — tracked as issue
  [#133](https://github.com/yshewchuk/investment-validation/issues/133).
- **`daily_market` mcap carry-forward is scoped to loaded partitions** — a
  winner row's null `mcap_usd` is backfilled only from an observation
  already loaded in this build, never by scanning further back; deliberate,
  not a defect (cross-year carry tracked as issue
  [#195](https://github.com/yshewchuk/investment-validation/issues/195)).
