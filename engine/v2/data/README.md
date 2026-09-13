# `engine/v2/data`

## Ownership

Implements the **Ingestion — fetch receipts, normalization, coverage, finality, source revisions** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**1** of §4.1.

Replaces (§4.4): `data/sources/`, `data/normalize/`, `data/pulls/`, `store.py`, `fetch.py`, `throttle.py`, `finality.py`, `rebuild.py`, `calendar sourcing from calendar.py`.

## Responsibilities

- Fetch receipts and raw object retention.
- Normalization into versioned datasets with declared contracts.
- Coverage watermarks, finality and source revisions.
- Atomic snapshot commit, per §5.5.

## Non-responsibilities

- **Compute a trading verdict** — `engine/v2/scoring` does it instead.
- **Change a champion** — `engine/v2/models/training` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

| Module | Names |
|---|---|
| `documents` | `loads_document`, `decode_document` — strict decoding of `engine.v2.contracts.data` documents, beyond what `engine.v2.foundation.typed` can express from annotations alone. A `TimeInterval`/`FragmentRecord` time bound accepts three mutually exclusive kinds — date, offset-required RFC 3339 timestamp, or the offset-less `naive_timestamp` `time_formats` recognizes — and refuses a mix of kinds or an out-of-order pair. |
| `time_formats` | `NAIVE_TIMESTAMP_FORMAT`, `NAIVE_TIMESTAMP_RE`, `format_naive_timestamp`, `is_naive_timestamp` — the one offset-less timestamp wire form (`objects.py` writes it for a legacy `datetime64[ns]`/`datetime64[us]` observation column with no UTC offset to trust; `documents.py` recognizes it back) so the two cannot drift apart. Stdlib only. |
| `manifests` | `table_contract_hash` — a `TableContract`'s `definition_hash` (phase-2 guide §5.1). `fragment_record`, `fragment_ref`, `dataset_manifest`, `snapshot_ref` — build a `FragmentRecord`/`DatasetManifest`/`SnapshotRef` with content-derived `frag_`/`dsv_`/`snap_` ids and a separate `manifest_hash` that additionally covers provenance/parent/evidence (phase-2 guide §5.2). `verify_fragment_record`, `verify_dataset_manifest`, `verify_snapshot_ref` — recompute every identity field and raise `MANIFEST_CORRUPT` on any mismatch. `fragment_identity_payload`, `dataset_version_identity_payload`, `snapshot_identity_payload` — the canonical pre-hash payload each id is derived from, exported for their own permutation tests (mirrors `objects.logical_partition_hash`). |
| `legacy_adapter` | `build_legacy_mapping`, `LegacyMappingError` — the versioned `legacy_table_mapping.v1.0` document mapping the six Tier-2 tables, the feature panel, and Tier-4 forecasts to `TableContract`s, plus the separately pinned (non-queryable) legacy `SNAPSHOT` compatibility metadata. This is the package's only module importing legacy code (phase-2 guide §4); its reviewed per-column facts live in `legacy_annotations.json` beside it. |
| `schema` | `OWNER`, `MIGRATIONS` — the data-owner catalog schema (phase-2 guide §6) as plain `(version, name, statements)` tuples, never `engine.v2.ops.migrations.Migration` objects. `engine/v2/ops/bootstrap.py` is the only consumer: it wraps these into `Migration`s and applies them as migration owner `"data"`, colocated in the same SQLite file as the `ops`/`ledger` owners (phase-2 guide §3.3). This module never imports `engine.v2.ops`. |
| `objects` | `publish_legacy_file`, `inspect_fragment`, `FragmentInspection` — publish one legacy Parquet file as an immutable, content-addressed object via `foundation.ArtifactStore` (never a rename, symlink or hard link), then stream it in bounded Arrow batches into a fragment candidate: row count, primary-key and time bounds, byte hash, and a streaming `logical_rows.v1` content hash, validated against a `TableContract`. No catalog insert (phase-2 guide §7.2); `manifests.fragment_record` (P2-2c) turns a `FragmentInspection` into a content-identified `FragmentRecord`. |
| `errors` | `DataError` — the `Problem` envelope every refusal in this package raises, built only from `contracts.data.DATA_FAILURE_CODES` (category and retryability come from that table, never guessed at a call site). Messages are redacted: no legacy filesystem path, no row value. |

<!-- public-interface: loads_document, decode_document, table_contract_hash, fragment_record, fragment_ref, dataset_manifest, snapshot_ref, verify_fragment_record, verify_dataset_manifest, verify_snapshot_ref, fragment_identity_payload, dataset_version_identity_payload, snapshot_identity_payload, build_legacy_mapping, LegacyMappingError, OWNER, MIGRATIONS, publish_legacy_file, inspect_fragment, FragmentInspection, DataError -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

`engine.v2.ops` imports `schema.OWNER`/`schema.MIGRATIONS` in `bootstrap.py` to
apply the data-owner catalog schema after the ops/ledger owners (phase-2 guide
§3.3, §4).

<!-- consumers: engine.v2.ops -->

## Usage

```python
from engine.v2.contracts import EventRef
from engine.v2.data import decode_document, loads_document
from engine.v2.foundation import to_document, canonical_json

ref = EventRef(event_id="evt_1", calendar_revision="cal.v1")
assert decode_document(EventRef, to_document(ref)) == ref
assert loads_document(EventRef, canonical_json(to_document(ref))) == ref
```

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
