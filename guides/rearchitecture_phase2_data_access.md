# Rearchitecture Phase 2 — Data Access

## 1. Objective and scope

This is **Phase 2 of the rearchitecture: the third migration phase, after
Phase 0 — Baseline and Phase 1 — Operations**. It is not the original program
guide `phase2_experiment_framework.md`.

Build the immutable data boundary required by
[system_rearchitecture.md §12](system_rearchitecture.md#12-migration-sequence-and-rollback):
a snapshot repository, immutable manifests, bounded reads, and adapters that
let the unchanged legacy scorer consume one pinned snapshot. A reader must
resolve one snapshot before it starts and must see that same snapshot until it
finishes. A failed or interrupted replacement build must leave the active
snapshot readable.

This phase changes **how existing data is identified and read**. It does not
change the data, its economic meaning, the scoring formulas, or ordinary
production authority. The official legacy board remains authoritative. Phase 2
runs the replacement repository and adapter in shadow until the exit gate in
§14 is green.

### 1.1 Deliver in this phase

- Versioned data contracts for the existing six Tier-2 tables, feature panel,
  and Tier-4 forecasts, plus a separately pinned compatibility object carrying
  the exact legacy snapshot metadata required by scoring.
- Immutable object and fragment registration with byte hashes, logical content
  hashes, primary-key bounds, time bounds, row counts, and provenance.
- Catalog-owned `DatasetVersion` and `DataSnapshot` manifests, plus a
  compare-and-swap active-snapshot pointer.
- A repository that resolves an exact snapshot and returns projected,
  predicate-filtered Arrow `RecordBatch` streams under explicit row and batch
  limits.
- Exact event and option-chain lookup over the same repository contracts.
- A snapshot-to-legacy adapter that materializes a private, bounded compatibility
  view and invokes the unchanged scorer through the Phase 1 supervisor.
- Crash tests, corruption controls, read-consistency tests, and score parity
  evidence against the Phase 0 corpus and one real disk round trip.
- Completion of the three Phase 1 nightly stages a v1-parity shadow board still
  lacks (§9.4): a reachable decisions stage with a real plan/evidence producer,
  a render action that carries the full legacy renderer inputs, and graph
  consumers for the export, publication, and backup outbox effects.

### 1.2 Explicitly outside this phase

| Later owner/phase | Do not implement in Phase 2 |
|---|---|
| Phase 3 incremental data | Provider fetches, normalization changes, coverage watermarks, changed-key merges, append/correction planning, tombstones, dependency invalidation, compaction, or garbage collection. |
| Phase 4 scoring | Feature recipes, extracted scoring stages, new strategy or model registries, chooser extraction, structure generation, simulation, or financial calculations. |
| Phase 5 models | Training recipes, folds, residual construction, release candidates, promotion, or rollback of champions. |
| Phase 6 UI | Serving projections, read API, React application, or offline board export. |
| Phase 7 live shadow | Intraday objects, entitlement probes, pre-close clocks, or live collection. |
| Phase 8 cutover | Deleting legacy code, switching official prediction authority, or renaming `engine/v2`. |

Do not use a Phase 2 repository commit as a disguised data correction. The
initial snapshot is a byte-preserving registration of accepted legacy output.
If a value must change, stop and assign that work to Phase 3 or to a separately
approved defect correction with its own compatibility evidence.

## 2. Required design context

Read the named sections before implementing the corresponding work. The
contracts below narrow them into an executable Phase 2 slice; they do not
replace the parent documents.

| Work | Required reading | Constraint |
|---|---|---|
| Phase boundary | [System rearchitecture](system_rearchitecture.md) §§3–5, 11–12 | Preserve existing behavior; Phase 2 is snapshots and access, while Phase 3 owns updates and corrections. |
| Shared types | [Component contracts](component_contracts.md) §§2.1–2.5 | Canonical hashes, UTC timestamps, strict versions, object kinds, and failure envelopes. |
| Repository contracts | [Component contracts](component_contracts.md) §§4.2, 5.1–5.3 | Atomic publication, exact snapshot resolution, bounded reads, event/quote identity, finality references, and knowledge modes. |
| Data identities | [Data model](rearchitecture_data_model.md) §1 | `DataSnapshot`, `SnapshotTable`, `DatasetVersion`, `VersionFragment`, `Fragment`, and `FragmentInput`. |
| Diagnostics | [Component contracts](component_contracts.md) §15 | Stage-localized, complete `ComparisonReceipt`; empty populations are incomparable. |
| Operations seam | [Phase 1 operations](rearchitecture_phase1_operations.md) §§5.1, 9.2 | Replace `LegacyInputManifest` for migrated readers; retain the barrier for remaining mutable legacy paths. |
| Baseline evidence | [Phase 0 baseline](rearchitecture_phase0_baseline.md) §§5–9 | Compare through real public entrypoints using the frozen corpus and declared tolerances. |

The standing data rules remain in force: percent and fraction units stay
distinct; adjusted and unadjusted spot stay distinct; unavailable timestamps
remain unknown; reconstructed data never becomes observed merely because it
has a hash; and a ticker/date pair is not silently promoted into a permanent
security, event, or contract identity.

## 3. Starting point and prerequisites

### 3.1 Required green baseline

Before merging any repository implementation:

1. `checks/rearchitecture_phase0_gate.py` is green against the accepted corpus.
2. `checks/rearchitecture_phase1_gate.py` is green against fresh coverage
   evidence generated from the same commit.
3. `checks/import_layers.py`, `checks/code_budgets.py`,
   `checks/package_readmes.py`, repository hygiene, and the pinned linter are
   green.
4. The Phase 1 decision path remains shadow-only unless separately authorized.
5. No legacy writer is allowed to mutate a read set while the first snapshot
   is captured. The import stage declares all relevant store domains for read,
   acquires the Phase 1 shared read lease, and pins the complete read set. That
   lease blocks every writer until capture ends; an unavailable or dirty domain
   refuses the import.

Synthetic contract/catalog work may proceed when a large private corpus is not
available. Score parity, the initial production-data snapshot, and the Phase 2
exit claim may not.

The three Phase 1 completion items in §9.4 are not prerequisites for starting
Phase 2. They are Phase 2 deliverables, scheduled as P2-5 ahead of the
score-parity milestone, because Phase 4 changes scoring and a v1-versus-v2
board comparison must exist before it does.

### 3.2 Existing implementation to preserve

| Existing code | Phase 2 treatment |
|---|---|
| `engine/data/schemas.py` | Source of the six legacy table mappings and validated conversions. Import it only through `engine/v2/data/legacy_adapter.py`; do not copy its coercion logic into the repository. |
| `engine/data/store.py` | Its path layout and read results define adapter compatibility. Its destructive replacement writer is not used as the new repository writer. |
| `engine/data/manifest.py` | Preserve its `SNAPSHOT` bytes and Tier-2/Tier-4 hash distinction as compatibility metadata. It is not the new immutable manifest. |
| `engine/features.py`, `engine/score.py` | Remain unchanged. Run them only in a private root produced from a pinned repository snapshot. |
| `engine/v2/foundation/artifacts.py` | Reuse its durable, content-addressed publication and path defenses. Do not create another object store. |
| `engine/v2/ops/catalog.py`, `migrations.py`, `bootstrap.py` | Reuse the connection, transaction, and migration coordinator. The data package declares its schema without importing ops; ops bootstrap applies it as owner `data`. |
| `LegacyInputManifest`, `store_barrier.py` | Keep for legacy writers and unmigrated readers. Snapshot-backed score/finality readers stop using it after parity acceptance. Do not delete old contract versions. |

### 3.3 Decisions fixed by this plan

These decisions prevent lower-level implementations from selecting incompatible
shortcuts.

**One SQLite file, separate schema owner.** Data manifests live in the existing
local SQLite catalog, under migration owner `data`. `engine/v2/data/schema.py`
contains plain immutable statement tuples and imports only contracts/foundation.
`engine/v2/ops/bootstrap.py`, which is on a higher layer, wraps and applies
those statements with the existing migration machinery. Never import
`engine.v2.ops` from `engine.v2.data`.

**Objects are copied, not linked from the mutable legacy store.** An initial
import streams each accepted file through `ArtifactStore` into the immutable
object area. A manifest may not reference the mutable `data/curated` path.
Symlinks and hard links to legacy files are refused. Existing raw research
caches remain readable in place, but they are outside the Phase 2 curated
snapshot until Phase 3 gives them receipt contracts.

**Retain everything in Phase 2.** Do not implement deletion, garbage
collection, compaction, or retention leases. Retaining all registered objects
makes a reader safe for the full phase and avoids a partial GC design. Later
phases may add GC only after references from experiments, releases, ledgers,
and active readers are represented.

**Knowledge mode is per table version.** The abbreviated `SnapshotRef` example
in component contracts §5.1 shows one `knowledge_mode`, while its normative
prose and the system guide require the value per table. Implement
`knowledge_mode_by_table`. The imported legacy tables default to
`reconstructed` unless an accepted attestation or original availability
receipt is explicitly referenced. Never infer `observed` from a file timestamp.

**No implicit current snapshot.** Only an ops planner or coordinator may resolve
the mutable head. It resolves it once and writes the immutable `SnapshotRef`
artifact ID into the submitted job. Repository reads require that exact ID and
never fall back to the current head.

**Logical partitions are stable.** For the migration snapshot, a logical
partition is the legacy table year, or the single declared partition for panel,
Tier-4 forecasts, and compatibility metadata. Physical file boundaries do not
define table identity. A logical partition hash is computed over schema ID plus
rows in primary-key order; the dataset logical hash is computed over ordered
`(partition_key, logical_partition_hash)` pairs. This keeps later physical
compaction from changing logical identity.

## 4. Package and file ownership

Use the existing layer map. Proposed paths marked **new** do not exist at the
start of this phase.

| File | Owns | Must not own |
|---|---|---|
| `engine/v2/contracts/data.py` **new** | Schema-only data definitions and handles in §5. | Paths, SQLite, Arrow, clocks, hashes computed during construction, or legacy imports. |
| `engine/v2/data/schema.py` **new** | Data-owner catalog DDL as append-only migration statements. | Connection setup or importing ops. |
| `engine/v2/data/objects.py` **new** | Fragment inspection, byte/logical hashing, and durable publication through `ArtifactStore`. | Catalog head changes or data normalization. |
| `engine/v2/data/manifests.py` **new** | Deterministic dataset/snapshot manifest construction and validation. | Reading a mutable latest directory. |
| `engine/v2/data/repository.py` **new** | Exact snapshot resolution, bounded Arrow scans, event/chain lookup, and query dependency explanation. | pandas return values, provider access, feature computation, or implicit latest resolution. |
| `engine/v2/data/legacy_adapter.py` **new** | Exact mapping from accepted legacy tables/files to v2 contracts and private compatibility materialization. This is the package's only legacy import module. | New normalization rules or altered economic values. |
| `engine/v2/data/import_snapshot.py` **new** | Coordinator-facing import plan and commit functions for a completed legacy candidate. | Running a provider pull or rebuilding production in place. |
| `engine/v2/ops/bootstrap.py` | Applies the `data` migration after importing the lower-layer schema declaration. | Defining data semantics. |
| `engine/v2/ops/supervisor.py`, `plans.py`, `nightly.py` | Pin a `SnapshotRef` at submission and materialize it for approved legacy stages. | Resolving current again inside a worker. |
| `checks/rearchitecture_phase2_gate.py` **new** | One Phase 2 acceptance entrypoint and evidence validation. | Implementing repository behavior. |

Update `engine/v2/contracts/__init__.py`, `engine/v2/data/__init__.py`, both
package READMEs, `checks/legacy_adapters.json`, the coverage baseline, and the
guide index in the same milestones that add their public names. The only new
v2-to-legacy edge is `engine/v2/data/legacy_adapter.py` to the exact legacy
symbols it adapts. The adapter ledger ceiling may rise only by that reviewed
inventory, then resumes ratcheting downward.

## 5. Contracts to complete

Implement these as frozen, keyword-only dataclasses with strict document
decoding. Every constant uses the exact version shown. Reject unknown fields,
enum members, duplicate map keys, non-finite numbers, naive timestamps, and
hashes that are not `sha256:` plus 64 lowercase hexadecimal characters.

### 5.1 Table definitions

Add these definitions to `engine/v2/contracts/data.py`:

```text
COLUMN_CONTRACT_V1 = "column_contract.v1.0"
TABLE_CONTRACT_V1 = "table_contract.v1.0"
TABLE_CONTRACT_REF_V1 = "table_contract_ref.v1.0"

ColumnContract (Definition member):
  name, physical_type, nullable
  unit, scale, adjustment_basis, timezone
  null_policy, sentinel_policy, allowed_range
  observation_time_semantics

TableContract (Definition):
  contract_id, definition_hash, table_name, semantic_version
  columns: ordered tuple[ColumnContract]
  primary_key: ordered tuple[column]
  duplicate_policy, foreign_keys
  partition_columns, filterable_columns, orderable_columns
  observation_time_column?, publication_time_column?, receipt_time_column?
  finality_semantics, provenance_semantics, coverage_semantics
  schema_evolution_policy, maximum_batch_rows, maximum_result_rows
  legacy_mapping_ref?

TableContractRef (Handle):
  contract_id, definition_hash, schema_version
```

`definition_hash` is computed by the registration function from canonical
definition content excluding `definition_hash`; the dataclass does not compute
it. Column order is meaningful. Changed units, key meaning, time meaning, or
null policy require a new major contract. Nullable additions require a minor
version. Never edit a registered definition under the same ID.

The first migration maps exactly these accepted datasets:

| Dataset name | Primary key / logical partition | Source |
|---|---|---|
| `securities` | `(ticker, year)` / `year` | `engine.data.schemas.SECURITIES` |
| `earnings_events` | `(event_id)` / `year` | `EARNINGS_EVENTS` |
| `daily_market` | `(ticker, date)` / `year` | `DAILY_MARKET` |
| `option_chains` | `(ticker, obs_date, expiry, strike, right)` / `year` | `OPTION_CHAINS` |
| `option_daily` | `(contract_ticker, obs_date)` / `year` | `OPTION_DAILY` |
| `trades` | `(trade_id)` / `year` | `TRADES` |
| `feature_panel` | `(ticker, date)` / one declared logical partition | `engine.paths.PANEL` and `engine.data.features.panel.PANEL_COLUMNS` |
| `tier4_forecasts` | `(ticker, event_date)` / one declared logical partition | `engine.paths.TIER4` and `engine.data.features.tier4.KEY_COLUMNS/COLUMNS` |

Do not transcribe column definitions by hand in two places. The legacy adapter
produces a versioned mapping artifact from the authoritative legacy schemas and
the separately reviewed unit/time/provenance annotations required by system
guide §5.2. A test fails if any source column lacks a mapping or if an
unreviewed source column appears. Preserve the exact bytes from
`engine.paths.SNAPSHOT_FILE` as a separate compatibility `ObjectRef`; it is
not a queryable dataset or a member of `SnapshotRef.table_versions`.

### 5.2 Object, fragment, version, and snapshot handles

```text
OBJECT_REF_V1 = "object_ref.v1.0"
FRAGMENT_REF_V1 = "fragment_ref.v1.0"
FRAGMENT_RECORD_V1 = "fragment_record.v1.0"
DATASET_VERSION_REF_V1 = "dataset_version_ref.v1.0"
DATASET_MANIFEST_V1 = "dataset_manifest.v1.0"
SNAPSHOT_REF_V1 = "snapshot_ref.v1.0"

ObjectRef (Handle):
  kind, object_id, content_hash, byte_size, schema_version

FragmentRef (Handle):
  fragment_id, manifest_hash, schema_version

FragmentRecord (Record):
  fragment_id, manifest_hash, object_ref, table_contract_ref
  partition_key, row_count, byte_hash, logical_content_hash
  primary_key_min, primary_key_max, time_min?, time_max?
  input_receipt_refs, import_request_hash

DatasetVersionRef (Handle):
  dataset_version_id, table_contract_ref, manifest_hash, schema_version

DatasetManifest (Record):
  dataset_version_ref, logical_content_hash, row_count
  parent_dataset_version_id?
  fragment_refs: ordered tuple[FragmentRef]
  coverage_receipt_refs, knowledge_mode, availability_evidence_refs

SnapshotRef (Handle):
  snapshot_id, manifest_hash, parent_snapshot_id?
  table_versions: map[table_name, DatasetVersionRef]
  calendar_version, source_priority_version
  finality_receipt_refs, knowledge_mode_by_table, schema_version
```

`ObjectRef` deliberately omits a storage path. Only the repository resolves an
object ID to an internal location. Keep the Phase 1 `ArtifactRef` readable for
old jobs; do not rewrite it into `ObjectRef` or change old artifact identities.
`FragmentRef` and `DatasetVersionRef` are cheap pinned handles. The row
metadata and complete fragment membership live in immutable records instead of
turning a handle into an unbounded payload.

IDs are content-derived from their deterministic payloads:

- `fragment_id` covers the `FragmentRecord` payload: contract, logical
  partition, logical content hash, object hash, row count, and bounds;
- `dataset_version_id` covers the `DatasetManifest` payload: contract and
  ordered complete fragment membership;
- `snapshot_id` covers the ordered table-to-dataset-version map, calendar and
  source-priority versions, finality refs, and per-table knowledge modes.

Operational timestamps, attempt IDs, durations, and log refs live in import or
commit receipts and are excluded from these IDs.

### 5.3 Bounded query contracts

```text
KEY_PREDICATE_V1 = "key_predicate.v1.0"
TIME_INTERVAL_V1 = "time_interval.v1.0"
DATA_QUERY_V1 = "data_query.v1.0"

KeyPredicate (Command member):
  column, operator: eq | in, values: ordered tuple[canonical scalar]

TimeInterval (Command member):
  column, start_inclusive?, end_exclusive?

DataQuery (Command):
  snapshot_id, table_contract_ref, columns: ordered tuple[str]
  key_filter: ordered tuple[KeyPredicate]
  time_interval?, order_by: ordered tuple[str]
  max_batch_rows, max_result_rows, deadline?
```

Phase 2 requires both row limits to be finite positive integers. The contract
may later gain a supervised unbounded-streaming mode, but do not create one in
this phase. A full historical read is expressed with an explicit range and a
finite limit derived from the pinned dataset manifest. Limits may not exceed
the table contract. `deadline` is execution metadata and does not enter query
identity.

Only `eq` and `in` are supported now. Both time bounds are optional
individually, but every query must contain at least one key predicate or one
time bound. A full historical read names its full time interval explicitly.
`in` values are non-empty, unique, and sorted in canonical scalar order so the
same membership has one command identity. No caller-provided SQL, Arrow
expression, filesystem path, callable, regular expression, or arbitrary sort
expression enters the contract.

`columns` must be non-empty and known. Predicate and interval columns must be
declared filterable. In v1, `order_by` must equal the full primary key. The scan
result has that deterministic order. Missing historical nullable columns are
returned as typed nulls according to the registered contract; required missing
columns fail integrity validation.

### 5.4 Event, contract, and chain contracts

Complete the existing component-contract surface needed by repository
consumers:

```text
EVENT_REF_V1 = "event_ref.v1.0"
EARNINGS_EVENT_V1 = "earnings_event.v1.0"
CONTRACT_ID_V1 = "contract_id.v1.0"
CHAIN_QUERY_V1 = "chain_query.v1.0"
CHAIN_MEMBER_V1 = "chain_member.v1.0"
CHAIN_SNAPSHOT_V1 = "chain_snapshot.v1.0"
DEPENDENCY_PLAN_V1 = "dependency_plan.v1.0"

EventRef:
  event_id, calendar_revision

EarningsEvent:
  event_ref, security_id, ticker_at_event, scheduled_event_date
  session, actual_announcement_at?, session_source
  confidence, conflict_status, known_from?, supersedes_revision?

ContractId:
  contract_id, security_id, vendor_mappings
  expiry, right, exact_strike, multiplier, adjustment_identity

ChainQuery:
  event_ref?, security_id, observation_ceiling, session_date
  expiry_interval?, quote_policy_ref, max_contracts

ChainMember:
  contract_id, quote_observation_id?
  bid?, ask?, mid?, iv?, delta?, volume?, open_interest?
  bid_size?, ask_size?, source, source_row_ref
  availability_status, missing_reason?, quality_flags

ChainSnapshot:
  chain_id, source_snapshot_ref, security_id
  observed_at, available_at?, received_at?, session_date
  quote_policy_ref, spot fields, rows: ordered tuple[ChainMember]
  expected_contracts, supported_contracts, returned_contracts
  coverage_ref, knowledge_mode

DependencyPlan (Record):
  request_hash, snapshot_ref
  dependencies: ordered [table_name, DatasetVersionRef, FragmentRef,
                         columns, predicates, estimated_rows, maximum_rows]
```

The imported event ID is treated as an opaque stable identifier even if its
legacy spelling contains a date. Record the old ticker/date key in the mapping;
never regenerate the ID when a later revision moves the date. Ambiguous event
clusters remain conflicts.

For the compatibility chain mapping, preserve the current exact expiry/right/
strike tuple and its standard-contract multiplier assumption as an explicitly
versioned legacy mapping. Convert strike through its full stored value to a
decimal string; do not display-round it. Mark publication and receipt times
unknown when the legacy row lacks them, and mark the table reconstructed unless
accepted evidence says otherwise. `get_chain` may return reconstructed history;
it may not call it observed or certify decision-time availability.

Create compatibility identities mechanically and preserve their mapping:

- `security_id = sha256({scheme: legacy_ticker.v1, ticker: exact_ticker})`;
- `contract_id = sha256({scheme: legacy_option.v1, security_id, expiry, right,
  exact_strike, multiplier: "100", adjustment_identity:
  "legacy_standard.v1"})`.

These IDs preserve the current frozen snapshot; they do not assert that two
historical ticker aliases are one security. Phase 3 may add an explicit
symbology mapping supported by provider evidence. It must never rewrite IDs in
an old snapshot.

### 5.5 Repository interface

The public interface in `engine/v2/data/__init__.py` is exactly:

```text
Repository.resolve(snapshot_id: str) -> SnapshotRef
Repository.scan(query: DataQuery) -> Iterator[pyarrow.RecordBatch]
Repository.get_event(event_ref: EventRef, snapshot_ref: SnapshotRef) -> EarningsEvent
Repository.get_chain(query: ChainQuery, snapshot_ref: SnapshotRef) -> ChainSnapshot
Repository.explain_dependencies(query: DataQuery | ChainQuery) -> DependencyPlan
```

In Phase 2, `DependencyPlan` names the exact snapshot, dataset versions,
fragments, columns, predicates, and estimated/maximum rows used by a data or
chain query. Recipe dependency graphs and change invalidation are Phase 3/4 and
must return a versioned `UNSUPPORTED_CONTRACT`, not a guessed plan.

### 5.6 Import and compatibility contracts

These are the only Phase 2 operational contracts added to the data contract
module:

```text
SNAPSHOT_IMPORT_REQUEST_V1 = "snapshot_import_request.v1.0"
SNAPSHOT_IMPORT_RECEIPT_V1 = "snapshot_import_receipt.v1.0"
LEGACY_MATERIALIZATION_REQUEST_V1 = "legacy_materialization_request.v1.0"

SnapshotImportRequest (Command):
  scope, source_manifest_ref, source_manifest_hash
  table_sources: map[table_name, ordered tuple[LegacyFileRef]]
  table_contract_refs: map[table_name, TableContractRef]
  legacy_snapshot_source_ref: LegacyFileRef
  calendar_version, source_priority_version
  finality_receipt_refs, knowledge_mode_by_table
  expected_head_snapshot_id?, expected_head_generation

SnapshotImportReceipt (Receipt):
  receipt_id, request_hash, attempt_id, fence
  snapshot_ref?, legacy_snapshot_object_ref?
  prior_head_snapshot_id?, resulting_head_snapshot_id?
  resulting_head_generation?, status: committed | failed | conflict
  problem?, envelope

LegacyMaterializationRequest (Command):
  request_hash, snapshot_ref, legacy_snapshot_object_ref
  direct_scope, evidence_scope
  table_queries: map[legacy_path, DataQuery]
  registry_and_model_refs, calendar_refs
  legacy_layout_version, expected_population
```

`LegacyFileRef` remains the Phase 1 compatibility handle for source capture;
using it here does not make `LegacyInputManifest` the repository snapshot.
`direct_scope` contains requested events/tickers and `evidence_scope`
contains the broader analog/history population. Both participate in the
materialization request hash. A materialization contains concrete
`DataQuery` commands, so it is itself a Command rather than a Definition; this
is required by component contracts §2.5.

## 6. Catalog schema and invariants

Add one `data` owner migration. Never edit an applied ops or ledger migration.
Use `data_` table prefixes so ownership is obvious in a shared catalog.

| Table | Required columns and keys |
|---|---|
| `data_contracts` | `contract_id PK`, `schema_version`, `definition_hash UNIQUE`, `definition_json`, `registered_at` |
| `data_objects` | `object_id PK`, `kind`, `content_hash`, `byte_size`, internal `storage_key`, `registered_at`; unique `(kind, content_hash)` |
| `data_fragments` | `fragment_id PK`, `object_id FK`, `contract_id FK`, `partition_key`, row count, byte/logical hashes, key/time bounds JSON, import request hash |
| `data_dataset_versions` | `dataset_version_id PK`, `contract_id FK`, parent nullable FK, `manifest_hash UNIQUE`, logical hash, row count, knowledge mode, evidence JSON |
| `data_version_fragments` | `(dataset_version_id, ordinal) PK`, `fragment_id FK`, unique `(dataset_version_id, fragment_id)` |
| `data_snapshots` | `snapshot_id PK`, parent nullable FK, `manifest_hash UNIQUE`, calendar/source-priority versions, finality refs JSON, knowledge-mode JSON, commit receipt ref |
| `data_snapshot_tables` | `(snapshot_id, table_name) PK`, `dataset_version_id FK`; one version per named table |
| `data_snapshot_heads` | `scope PK`, `snapshot_id FK`, `generation`, `updated_at`, `update_receipt_ref` |
| `data_import_receipts` | `receipt_id PK`, attempt/fence, source-manifest hash, result snapshot nullable, status, problem JSON, timestamps |

Enforce these invariants in schema constraints or immutable-table triggers as
well as Python checks:

1. Contract, object, fragment, dataset-version, snapshot, membership, and
   receipt rows are append-only. Reject update/delete.
2. Only `data_snapshot_heads` is mutable. Update it with
   `WHERE scope=? AND snapshot_id=? AND generation=?`; zero changed rows is a
   conflict, never last-writer-wins.
3. A snapshot commit transaction inserts any new immutable metadata,
   complete membership, import receipt, and optional head update together.
4. Every dataset version is a complete logical view. Readers do not follow a
   parent chain to reconstruct membership.
5. A fragment object is durable and re-verified before its catalog row is
   inserted. A catalog row never points to staging.
6. Dataset row count equals the sum of its unique fragment row counts, and
   fragment key ranges do not overlap within a logical partition.
7. Snapshot table names are unique, known, and bound to the matching contract.
8. `knowledge_mode` is one of `observed`, `attested_stable`, or
   `reconstructed`. Observed/attested versions require non-empty evidence refs.

`registered_at` and receipt timestamps use the injected `Clock`. SQL and
manifest tests use `FrozenClock`; no data identity depends on wall time.

## 7. Initial snapshot import

### 7.1 Plan and freeze the source

The import coordinator performs these steps in order:

1. Acquire Phase 1 shared read leases for every declared legacy-store domain
   and pin the complete read set. Refuse an unavailable or dirty domain.
2. Resolve the accepted legacy `SNAPSHOT`, enumerate every declared table and
   compatibility object from known paths, and reject globs outside those roots.
3. Create `SnapshotImportRequest` containing the expected legacy snapshot
   hash, exact relative file list, expected contract refs, calendar and
   source-priority versions, knowledge modes, and expected current head.
4. Hash the request and submit it through the Phase 1 supervisor. The worker
   receives no mutable-current alias.
5. Recheck the legacy file byte manifest while the lease is held. A changed,
   missing, indirect, or newly appeared declared member fails with
   `INPUT_CHANGED`.

### 7.2 Publish and inspect objects

For each Parquet source file, in deterministic table/partition/path order:

1. Copy its bytes through `ArtifactStore`; never rename, symlink, or hard-link
   the legacy file into the object store.
2. Verify the byte hash and Parquet footer from the immutable copy.
3. Validate physical columns against the mapped table contract. Permit an
   absent column only when that contract version declares the historical gap
   nullable; synthesize it only at read time.
4. Stream rows in bounded Arrow batches. Verify primary-key monotonicity and
   uniqueness, types, finite-value rules, and declared partition membership.
5. Compute the logical partition hash from canonical typed values in primary-
   key order. Null, empty string, zero, false, NaN, and missing remain distinct;
   NaN/Infinity are refused unless mapped to contract null by the accepted
   legacy coercion.
6. Record row count, key bounds, time bounds, byte hash, logical hash, source
   file ref, and import request hash in a `FragmentRecord` candidate, then
   derive its cheap `FragmentRef`.

The logical-row hash algorithm is `logical_rows.v1`. Hash a canonical header
containing the algorithm version, table-contract ref, logical partition, and
ordered column/type list, followed by one newline and one JCS-encoded row array
per primary-key-sorted row. A row array follows contract column order. Dates
encode as `YYYY-MM-DD`; timestamps use canonical RFC 3339 UTC microseconds;
decimal values are exact strings; finite binary64 values use JCS number
encoding; null is JSON null. JCS escapes embedded newlines, so the record
delimiter is unambiguous. This algorithm must stream without retaining the
partition in memory.

Copy the legacy `SNAPSHOT` metadata through the same immutable object path,
validate its JSON shape and expected legacy hashes, and record its exact byte
hash. Do not run Parquet inspection or row hashing on that compatibility
object, and do not add it to a dataset version.

The migration importer may require one compacted, primary-key-sorted file per
legacy year. If it finds multiple or overlapping parts, it must either run the
existing accepted compaction in the private candidate root and compare its
logical rows, or fail with a precise remediation. It must not silently choose
filesystem order or load an unbounded year into pandas.

### 7.3 Build and commit manifests

After every object is durable:

1. Build each complete `DatasetManifest` from its ordered fragment members and
   derive its `DatasetVersionRef`.
2. Build the `SnapshotRef` from the exact named dataset versions and evidence
   refs. Validate all manifest hashes by recomputation.
3. Open one short `BEGIN IMMEDIATE` coordinator transaction.
4. Verify the attempt fence and expected current snapshot/generation.
5. Insert immutable data metadata and membership idempotently. An existing ID
   is accepted only if its full canonical payload matches.
6. Insert the success import receipt and compare-and-swap the requested head.
7. Commit. Only now is the snapshot active for that scope.

No file copy, Arrow scan, hash calculation, network call, or legacy scorer runs
inside the transaction. A failure before step 7 may leave an unreferenced
immutable object, which is retained. It may not expose a candidate snapshot.
The Phase 1 attempt receipt always records a failed attempt. Once the
coordinator regains control it also publishes a failed
`SnapshotImportReceipt` artifact and inserts it in a separate short
transaction; failure-receipt persistence never advances a snapshot head.

Required fault points are: before copy, during copy, after object fsync, after
object publication, during fragment inspection, before transaction, after each
catalog insert class, before head update, and before commit. Every injected
failure must leave either the old complete head or the new complete head; never
a mixed snapshot.

The import is one supervised, resumable job. It emits progress after each
logical partition and at least once per minute, estimates immutable-object and
scratch bytes before admission, and reuses an already verified object with the
same content hash on retry. It performs no provider call and does not run
concurrently with another legacy-store writer or memory-heavy data job.

## 8. Repository read behavior

### 8.1 Resolution

`Repository.resolve(snapshot_id)` loads the snapshot and all complete table and
fragment membership in one SQLite read transaction, verifies canonical
manifest hashes, and returns an immutable `SnapshotRef`. Unknown IDs return
`SNAPSHOT_NOT_FOUND`. Missing/corrupt membership returns `INTEGRITY_FAILED`.
Neither case consults `data_snapshot_heads`.

The ops helper `resolve_snapshot_head(scope)` is separate. It reads one head,
resolves it immediately, publishes the complete `SnapshotRef` as a Phase 1
artifact, and places that artifact ID in `JobSpec.input_refs`. Retries use the
same ref. A new head requires a new plan/job identity.

### 8.2 Scan validation and execution

Apply this order exactly:

1. Strictly decode `DataQuery` and validate limits/deadline.
2. Resolve `snapshot_id`; verify `table_contract_ref` equals the version pinned
   under that table name.
3. Validate selected/filter/order columns and scalar types against the
   contract. Reject duplicate columns and duplicate predicate columns.
4. Resolve the exact ordered fragment list from snapshot membership. Never
   glob a directory.
5. Prune fragments using manifest partition/key/time bounds.
6. Open only the surviving immutable objects and verify object identity. Cache
   a successful byte verification only for the same process and unchanged
   `(device, inode, size, mtime_ns)` tuple; a changed tuple forces rehashing.
7. Construct an Arrow dataset/scanner with column projection and filter
   expression before any pandas conversion. Include hidden primary-key columns
   needed to verify ordering, remove them before yielding when they were not
   requested, set `use_threads=False`, and set Arrow batch size no larger than
   `max_batch_rows`. Process fragments in manifest key-range order; never sort
   an unbounded result in memory.
8. Yield deterministic `RecordBatch` objects, checking cumulative rows before
   each yield. Exceeding `max_result_rows` fails; it never truncates and reports
   success.
9. Before completion, enforce deadline, total row count, ordering, and typed
   nullable-column synthesis. The containing job evidence records the query
   hash, resolved dependency plan, returned row count, and any shared
   `Problem`; do not invent a second repository receipt contract.

Do not expose a convenience `read_table()` that defaults to all columns/all
rows or returns a combined pandas frame. Callers that need pandas convert each
bounded batch themselves under their supervised resource profile.

### 8.3 Event and chain lookup

`get_event` builds one bounded query by exact `event_id` and verifies the row's
calendar revision against the requested `EventRef`. Zero rows is not found;
more than one is an integrity failure. It returns conflict state rather than
merging nearby events.

`get_chain` builds one bounded query by security/ticker compatibility mapping,
session, observation ceiling, and optional expiry interval. It must:

- exclude observations after the ceiling;
- apply the named quote policy deterministically;
- retain listed contracts without usable quotes with a null and reason;
- refuse a contract from another security or an ambiguous symbology mapping;
- preserve exact strikes, rights, expiries, quote repair flags, source fields,
  and unknown availability times;
- record expected, supported, and returned contract populations separately;
- fail rather than return a partial chain beyond `max_contracts`.

This method organizes existing rows; it does not select ATM legs, rank a
structure, repair new quote classes, or calculate PnL.

## 9. Legacy compatibility adapter

### 9.1 Adapter input and output

Use the versioned `LegacyMaterializationRequest` from §5.6. Its resolved content
contains:

- complete `SnapshotRef` artifact ID and manifest hash;
- planned tickers, years, events, table columns, and expected row ceilings;
- required feature-panel and Tier-4 columns;
- exact legacy registry/model/calendar artifact refs already pinned by Phase 1;
- target legacy layout version and the legacy snapshot metadata object ref.

The request hash is an input to the stage/checkpoint identity. A different
snapshot, ticker set, year range, column set, model ref, or calendar ref creates
a different materialization and scoring job.

The adapter creates a private attempt root with only the required legacy paths.
It scans the repository under finite limits, writes deterministic legacy-format
Parquet partitions, writes the exact preserved legacy `SNAPSHOT` bytes, and
copies the separately pinned model/calendar inputs. Every output is validated
against its source rows and made read-only before scoring starts.

Do not point `INVESTING_PLAN_ROOT` at the immutable object store and do not let
legacy code follow object-store paths directly. The compatibility root is a
derived attempt artifact: it can use scratch space and be deleted after the
attempt because the source snapshot remains immutable.

### 9.2 Required scoring population

For a score batch, the plan must include:

- requested/watchlist tickers and exact decision years;
- the broader historical analog population required by current behavior;
- earnings history through the correct event-history cutoff;
- option observations through the quote cutoff;
- panel/Tier-4 rows and model artifacts required by the registered current
  scorer;
- expected score population and legacy decision clock.

A watchlist-bounded daily-market query must not accidentally shrink the analog
pool. Record the direct event scope and broad evidence scope separately. If the
adapter cannot prove its read plan complete, it refuses checkpoint reuse and
score parity reports `incomparable`.

### 9.3 Phase 1 transition

After adapter parity passes:

1. Add snapshot-backed variants for read-only `legacy_finality`,
   `legacy_score`, `legacy_score_requests`, `legacy_model_evidence`, and
   `legacy_selfcheck` only where their complete reads are declared.
2. Keep effectful decision, settlement, render/publication, and remaining
   mutable legacy stages on their existing Phase 1 controls until their data
   reads are separately proven.
3. Supervisor validates the `SnapshotRef` artifact and materialization request
   before launch. It does not call `_pin_read_set` or copy mutable legacy data
   for a snapshot-backed stage.
4. The worker receives only the private compatibility root. The coordinator
   confirms the same snapshot artifact binding before admitting any output.
5. Checkpoint identity uses snapshot manifest hash, materialization request hash,
   implementation hash, parameters, environment, and output schema.

Do not delete `LegacyInputManifest`, `store_read_pins`, or the cooperative
barrier. Mark migrated stage uses in the adapter ledger; remove old paths only
when their live reference count reaches zero in a later phase.

### 9.4 Phase 1 nightly completion (carried into this phase)

Review of Phase 1 at `2e21e59` found that the supervised nightly cannot yet
produce a v1-parity board. Phase 2 closes the three gaps below at P2-5, before
its own score-parity milestone. None of them changes scoring, decision
semantics, or authority; all of it stays shadow-only under decisions D4/D5 of
the Phase 1 guide.

1. **Reachable decisions stage.** `build_legacy_job_requests` binds the
   `legacy_decisions` job's score and finality inputs to parent *job IDs* and
   binds no `decision_plan.json` or `decision_evidence.json`.
   `validated_decision_candidate` refuses job-ID bindings and requires both
   artifacts, and nothing outside the tests produces them, so `ops submit` can
   never commit a decision and projection/selfcheck never run. Add a
   coordinator-validated evidence stage between score and decisions that
   derives the `decision_plan.v1.0` and `decision_evidence.v1.0` artifacts
   from the committed score and finality artifacts: causality, coverage,
   finality, selection, and replay receipts bound to the exact artifact IDs
   and content hashes. Resolve job-ID bindings to the parent's committed
   output artifact at launch, and have the coordinator validate against those
   resolved IDs. Do not weaken `validate`. Receipts are derived from the
   artifacts, never hand-written; the canary selfcheck-receipt rule applies.

2. **Render at parity.** `_action_render` calls `render_bundle` with the score
   frame, session, and horizon only. The legacy nightly also concatenates the
   strike ladder rows into the scores and passes `fill_alpha`, `alt_strikes`,
   `panel`, `trades`, `build_meta` (freshness, quota, late as-ofs, and the
   execution clock with requested versus resolved session and finality),
   `build_health`, `flags`, and the registry. Carry the full argument set:
   the ladder and analog coverage already sit in the score artifact; the
   model-evidence artifact is already bound to the job but never read; the
   book view must read the exported ledger generation (item 3), not the
   staging copy of the mutable ledger. Meta and health are execution metadata:
   bind them by artifact and keep wall-clock fields out of the content hash
   the serialized selfcheck compares. Do not rewrite financial rendering; that
   remains Phase 4.

3. **Outbox consumers on the graph.** `commit_decisions_in_transaction`
   enqueues `export` and `release_intent` rows that nothing consumes;
   `export_generation`, `stage_release`, `publish_local`, and `run_backup`
   have test-only callers, so watermarks never advance and the
   budget-withholds-publication policy is never exercised on a real release.
   Wire export as a stage between decision commit and projection, with the
   projection reading the verified generation; publication as the fenced
   local pointer replace behind the selfcheck and engineering-gate receipts;
   and backup as its own optional branch. This is the §10.1 graph of the
   Phase 1 guide. Watermarks advance per effect and per scope; a subset shadow
   run cannot advance the global scope.

Stages not carried: refresh, the validate-refresh battery, the Tier 3/4
rebuild, missed-night backfill, and calibration flags stay outside the shadow
graph. Every shadow receipt lists them as absent so a shadow board is never
mistaken for a production one. Phase 3 owns refresh; the rebuild control is
D16.

## 10. Replacement rebuild and rollback behavior

Phase 2 does not rewrite normalization. It defines where a legacy rebuild is
allowed to write once a replacement run is needed:

1. The supervisor gives the existing rebuild command a private candidate root
   and the test proves every write stays beneath that root.
2. The rebuild writes there using its existing algorithms and validations.
3. A successful candidate is imported through §7 and compared with its
   expected parent/baseline before any head change.
4. A failed worker, validation, import, or comparison leaves the active head
   unchanged. The candidate and diagnostics may be retained privately.
5. Promotion of a candidate snapshot is an explicit coordinator action, not a
   side effect of worker success.

Rollback compare-and-swaps the head to a previously committed snapshot under a
new update receipt. It does not delete the failed/new snapshot, rewrite its
manifest, erase downstream decisions, or relabel old knowledge modes. Jobs
already pinned to either snapshot finish on that snapshot.

## 11. Failure codes

Use the shared `Problem` envelope and these stable codes. Do not throw raw
Arrow, pandas, SQLite, or filesystem errors across the public boundary.

| Code | Category / retryability | Meaning |
|---|---|---|
| `SNAPSHOT_NOT_FOUND` | dependency / false | Exact immutable snapshot ID is unknown. |
| `SNAPSHOT_NOT_READY` | dependency / true | Requested head/candidate has not been committed. |
| `SNAPSHOT_CONFLICT` | dependency / true | Expected parent/head generation lost compare-and-swap. |
| `CONTRACT_MISMATCH` | validation / false | Query, fragment, or dataset does not match its pinned contract. |
| `QUERY_NOT_BOUNDED` | validation / false | Missing/invalid projection, filter, row limit, or batch limit. |
| `RESULT_LIMIT_EXCEEDED` | resource / false | Actual rows exceed the declared maximum; no partial success. |
| `INPUT_CHANGED` | integrity / true | Mutable legacy source changed during capture. |
| `OBJECT_CORRUPT` | integrity / false | Immutable object bytes or size do not match its ref. |
| `MANIFEST_CORRUPT` | integrity / false | Membership or canonical manifest hash does not match. |
| `IDENTITY_CONFLICT` | validation / false | Event/security/contract mapping is ambiguous or inconsistent. |
| `UNSUPPORTED_CONTRACT` | validation / false | Later recipe/invalidation contract was requested in Phase 2. |

Public messages identify the stage and immutable refs but never include
licensed rows, credentials, source query tokens, or private filesystem paths.

## 12. Acceptance tests

Create focused tests with the following IDs. Each ID is a required behavior,
not a suggestion to assert an implementation detail.

| ID | Tier | Required proof and negative control |
|---|---:|---|
| D01 | 0 | Every data contract round-trips exactly; unknown fields/enums, naive time, malformed hash, duplicate column/key, NaN/Infinity, and incompatible version fail. |
| D02 | 0 | All six Tier-2 schemas plus panel, Tier-4, and legacy snapshot metadata have complete reviewed mappings. Removing or adding one source column fails. |
| D03 | 0 | IDs exclude operational timestamps but include every semantic/member field. Reordering meaningful columns or fragments changes identity; changing duration does not. |
| D04 | 0 | Catalog migrations are checksummed, idempotent, owner-scoped, and refuse edited/newer schemas. Immutable-table update/delete and duplicate conflicting IDs fail. |
| D05 | 0 | Query validator rejects implicit latest, unknown/mismatched snapshot or contract, empty projection, unsupported predicate/order, missing bounds, and limits above contract caps. |
| D06 | 0 | Synthetic scans return only projected/filter-matching rows in deterministic key order, synthesize only declared nullable historical columns, and fail rather than truncate. |
| D07 | 0 | Event revision mismatch, ambiguous event identity, cross-security chain row, post-ceiling quote, and collapsed expected population each fail with the correct code. |
| D08 | 0 | Logical hashes distinguish null/zero/empty/false and full-precision strikes, but remain unchanged when only Parquet physical encoding changes. Plant one row or key-order defect and catch it. |
| D09 | 1 | A real Parquet file is copied, fsynced, registered, read through Arrow, written to the private legacy layout, and read back with identical typed rows and byte/logical refs. Corrupt one object byte and prove the scan refuses it. |
| D10 | 1 | Snapshot resolution stays fixed while another connection publishes a new head. The first reader never mixes versions; a new resolver sees the new head. |
| D11 | 1 | Inject a crash at every §7 commit boundary. The head always resolves to the complete old or complete new snapshot and every referenced object verifies. |
| D12 | 1 | Repeating the same import is idempotent. Same ID with different content conflicts. Two concurrent expected-parent commits produce one winner and one clean conflict. |
| D13 | 1 | Private legacy materialization contains no undeclared file, mutable-store link, or writable input. Changing snapshot/tickers/years/columns changes job identity. |
| D14 | 0/1 | Phase 0 corpus requests scored through snapshot adapter match expected IDs, contracts, null masks, flags, forecasts, decisions, and full-precision values. A missing broad analog slice is caught as a stage-localized finding. |
| D15 | 2 | On a bounded cached current-data sample, with no provider calls, legacy direct scoring and snapshot-adapted scoring produce an `agree` `ComparisonReceipt` with nonzero expected/supported/compared populations. |
| D16 | 2 | The actual legacy rebuild entrypoint, run on a bounded representative private root, is failed during rebuild/import and leaves the active snapshot and legacy official board unchanged; successful rollback restores the prior head without deleting either snapshot. |
| D17 | 0 | Layer, README, code-budget, lint, hygiene, adapter-ratchet, and package coverage checks remain green with no v2 exemptions. |
| D18 | 1 | From `ops plan nightly` and `ops submit`, the finality → score → evidence → decisions chain runs through the real supervisor on frozen private inputs and commits decisions exactly once; resubmission returns the same receipts. A planted causality or population defect commits zero decisions and no release intent, while settlement still proceeds. |
| D19 | 2 | The v2 render stage's bundle, compared with `render_bundle` invoked the legacy way on the same scores, ladder, model evidence, ledger generation, meta, and health, is identical in every serialized view except declared execution-metadata fields; the serialized selfcheck passes in a separate bounded process. |
| D20 | 1 | Export produces a complete generation that the compatibility reader resolves before projection starts; publication cannot advance `current` without the selfcheck and engineering receipts, and a stale fence cannot advance it; a failed backup retries alone and no watermark other than its own moves. |

Suggested test files:

- `tests/test_v2_data_contracts.py` — D01–D03;
- `tests/test_v2_data_catalog.py` — D04, D12;
- `tests/test_v2_data_repository.py` — D05–D10;
- `tests/test_v2_data_atomicity.py` — D11, D12, D16 synthetic controls;
- `tests/test_v2_data_legacy_adapter.py` — D02, D13–D15;
- `tests/test_v2_ops_nightly_completion.py` — D18–D20 (ops-owned; the Phase 1
  coverage suite picks it up by its `test_v2_ops_` prefix);
- `checks/rearchitecture_phase2_coverage.py` and its committed baseline —
  run the fixed Phase 2 suite, bind results to the exact code hash, and enforce
  the per-package coverage ratchet;
- `checks/rearchitecture_phase2_gate.py` — validates fresh evidence for
  D01–D20 and reruns Phase 0/1 prerequisites.

Do not add tests that only restate dataclass assignments or mock every Arrow
and disk boundary. D09–D12 must touch real files and SQLite. D14 must invoke the
real legacy scoring public entrypoint in a fresh supervised process. D15, D16,
and D19 run sequentially on this host under the resource manager.

### 12.1 Gate evidence and invocation

The private gate input is one strict document:

```text
PHASE2_EVIDENCE_V1 = "phase2_evidence.v1.0"

Phase2Evidence:
  schema_version, code_hash, environment_hash
  snapshot_ref, legacy_snapshot_object_ref
  table_contract_mapping_hash
  import_receipt_refs, fault_matrix_ref
  dependency_plan_refs, comparison_receipt_ref, render_comparison_receipt_ref,
  rollback_receipt_ref
  expected_population, supported_population, compared_population
  authority_mode: shadow
```

`comparison_receipt_ref` and `render_comparison_receipt_ref` are two separate
fields, not one shared receipt: D15 is legacy-vs-adapter SCORE parity and D19
is v2-vs-legacy RENDER BUNDLE parity, over different populations and stage
graphs, so one field would let D15's receipt silently stand in for D19's.

The evidence validator verifies every referenced artifact, requires all D01–D20
rows, rejects a code/environment mismatch, rejects zero or collapsed
populations, and requires both comparison receipts present in the document to
carry verdict `agree`. It does not accept a summary boolean in place of the
referenced receipts.

Run the final gates from one unchanged commit. Long or memory-heavy checks are
submitted through the Phase 1 supervisor; the commands below are the
coordinator-side validation entrypoints after those receipts exist:

```bash
/usr/bin/python3 -u checks/rearchitecture_phase0_gate.py
/usr/bin/python3 -u checks/rearchitecture_phase1_coverage.py --measure --output /tmp/phase1-for-phase2-coverage.json
/usr/bin/python3 -u checks/rearchitecture_phase1_gate.py --coverage /tmp/phase1-for-phase2-coverage.json
/usr/bin/python3 -u checks/rearchitecture_phase2_coverage.py --measure --output /tmp/phase2-coverage.json
/usr/bin/python3 -u checks/rearchitecture_phase2_gate.py --coverage /tmp/phase2-coverage.json --evidence-manifest /tmp/phase2-evidence.json
```

Both Phase 2 inputs record the same code hash. The gate refuses stale coverage,
missing private artifacts, a different working-tree hash, or evidence produced
before the final implementation edit.

## 13. Implementation sequence

Complete and commit milestones in this order. Do not start supervisor wiring
or a full-data import before the synthetic repository is proven.

### P2-1 — Contracts and legacy mapping

1. Add the §5 contract schemas, exports, strict decoding examples, and D01/D03.
2. Add `data/legacy_adapter.py` mapping generation for the six tables, panel,
   Tier-4, and snapshot metadata; add the exact adapter-ledger entries.
3. Record missing legacy provenance/time fields as null and reconstructed.
4. Update contracts/data READMEs and public-interface markers.
5. Exit: D01–D03 and all Phase 0/1 engineering checks green.

### P2-2 — Catalog and immutable object inspection

1. Add data-owner schema statements and extend ops bootstrap without changing
   old migrations.
2. Implement fragment copy/verification, logical hashing, bounds, and manifest
   construction over small synthetic Parquet files.
3. Add immutable triggers, idempotent insert verification, and D04/D08/D09.
4. Exit: a corrupt object, edited migration, duplicate key, and conflicting
   existing ID all fail without a published snapshot.

### P2-3 — Snapshot commit and exact resolution

1. Implement complete dataset/snapshot manifests and one-transaction CAS head
   update.
2. Implement exact `resolve(snapshot_id)` and separate ops head resolution.
3. Add every fault point and concurrency case in D10–D12.
4. Exit: old/new atomicity and fixed-reader behavior proven on real SQLite and
   Parquet.

### P2-4 — Bounded repository reads

1. Implement query validation, manifest pruning, object verification cache,
   Arrow projection/filter pushdown, limits, ordering, and containing-job
   evidence.
2. Implement `get_event`, `get_chain`, and query-only dependency plans.
3. Add D05–D07 and the full D09 round trip.
4. Exit: no repository public method can perform an implicit or unbounded table
   read; planted identity/causality defects fail.

### P2-5 — Phase 1 nightly completion

Depends only on P2-1's green engineering checks; it may run alongside P2-2 to
P2-4 but must finish before P2-6.

1. Add the decision plan/evidence producer stage and launch-time resolution
   of job-ID bindings (§9.4 item 1).
2. Extend `_action_render` to the full legacy argument set, reading the bound
   model-evidence artifact and the exported ledger generation (§9.4 item 2).
3. Wire export, publication, and backup into the supervised graph as the
   Phase 1 guide's §10.1 shows (§9.4 item 3).
4. Add D18–D20; run D18 and D20 on synthetic inputs first, then on the frozen
   private root.
5. Exit: one shadow nightly submitted with `ops submit` yields a committed
   decision set, an exported generation, a rendered bundle that passes D19,
   and a locally published release under the fenced pointer, with no
   production ledger, registry, publication target, or paid pull touched.
   This is §14.2 step 3 of the Phase 1 guide; record it in the Phase 1 runbook
   activation table.

### P2-6 — Legacy materialization and score parity

1. Implement `LegacyMaterializationRequest` and private deterministic layout.
2. Wire snapshot refs into selected Phase 1 read-only stages; keep other stages
   on the barrier.
3. Run the full Tier-0 corpus and one Tier-1 real disk/scoring case. Produce
   complete `ComparisonReceipt` evidence.
4. Exit: D13–D15 green, with nonzero populations and no unexplained tolerance
   changes.

### P2-7 — Replacement-build atomicity and phase gate

1. Run the unchanged legacy rebuild entrypoint on a bounded representative
   private root under the supervisor; import its completed output and inject
   the D16 failure. Do not run a costly full rebuild solely to prove this
   control.
2. Prove injected failure and rollback behavior in D16.
3. Capture the first accepted full snapshot under a non-production shadow
   scope. Do not switch official consumers.
4. Generate fresh coverage evidence and run the Phase 0, Phase 1, and Phase 2
   gates from the exact candidate commit.
5. Exit: every item in §14 is evidenced and the working tree contains no
   private data or report artifact.

## 14. Phase exit criteria

Phase 2 is complete only when all of the following are true:

1. **Contracts complete:** `TableContract`, `TableContractRef`, `ObjectRef`,
   `FragmentRef`, `FragmentRecord`, `DatasetVersionRef`, `DatasetManifest`,
   `SnapshotRef`, `DataQuery`, `EventRef`, `EarningsEvent`, `ContractId`,
   `ChainQuery`, `ChainSnapshot`, the
   query-only `DependencyPlan`, `SnapshotImportRequest`,
   `SnapshotImportReceipt`, and `LegacyMaterializationRequest` are implemented
   at the versions in §5 and pass strict producer/consumer contract tests.
2. **All current score inputs represented:** the six Tier-2 tables, feature
   panel, Tier-4 forecasts, calendar/source-priority versions, finality refs,
   and per-table knowledge modes are pinned by one immutable shadow snapshot.
   Exact legacy snapshot metadata and model/registry artifacts remain
   separately pinned compatibility inputs.
3. **Immutable publication proven:** every committed fragment has verified byte
   and logical hashes; every dataset/snapshot manifest recomputes; no manifest
   points to mutable legacy or staging storage.
4. **Atomicity proven:** all required crash points and a competing-parent race
   expose only a complete old or complete new snapshot. A failed rebuild leaves
   the active snapshot readable.
5. **Reads bounded:** every public scan has explicit snapshot, projection,
   supported predicates, batch limit, result limit, and deterministic ordering;
   Arrow filtering/projection occurs before pandas conversion.
6. **Fixed reader proven:** a running reader keeps one snapshot while the head
   advances, and retries use the snapshot pinned at original job submission.
7. **Compatibility proven:** Phase 0 corpus parity and a nonempty real-data
   comparison both agree through the actual snapshot-to-legacy adapter. IDs,
   contract selections, null masks, flags, forecasts, decisions, and
   full-precision values have no unexplained differences.
8. **Rollback proven:** moving the shadow head back preserves both snapshot
   histories and does not mutate decisions, reports, or objects.
9. **No authority expansion:** official prediction/settlement/publication
   authority and the legacy board are unchanged; Phase 2 remains shadow-only.
10. **Engineering gates green:** fresh Phase 0, Phase 1, and Phase 2 gates,
    coverage ratchet, adapter ratchet, import layers, budgets, READMEs, lint,
    hygiene, and hook checks pass against one commit.
11. **Phase 1 nightly complete in shadow:** D18–D20 green, and one shadow
    nightly submitted through `ops submit` has committed decisions, exported a
    generation, rendered a bundle at parity with the legacy renderer, and
    published locally under the fenced pointer. Refresh, the rebuild, and
    backfill remain outside the shadow graph and are listed as absent.

The Phase 2 evidence bundle contains the snapshot manifest/ref, table-contract
mapping hashes, import/commit receipts, fault matrix, resolved query dependency
plans and job evidence, score `ComparisonReceipt`, rollback receipt, exact
code/environment hashes, and fresh gate outputs. Keep licensed rows and private
paths out of public code and reports. The public completion note records only
hashes, counts, contract versions, verdicts, and private evidence refs.

Passing this gate authorizes Phase 3 to build incremental ingestion on the
repository and permits later UI work against frozen projections. It does not
authorize production scoring cutover, strategy promotion, paid provider calls,
or deletion of any legacy path.
