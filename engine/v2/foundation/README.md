# `engine/v2/foundation`

## Ownership

Implements the **paths, env, canonical JSON, session/calendar arithmetic, causality primitives** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**0** of §4.1.

Replaces (§4.4): `paths.py`, `env.py`, `jsonio.py`, `audit.py`, `session arithmetic from calendar.py`.

## Responsibilities

- Canonical JSON (RFC 8785) and content hashing, per contracts §2.2.
- Path and environment resolution.
- Session arithmetic: BMO/AMC anchoring, trading-day offsets.
- Causality primitives — the cutoff comparison every feature respects.
- Strict decoding of contract documents: unknown fields, unknown enum values
  and unsupported schema versions are refused (contracts §2.3).
- Clocks and the RFC 3339 wire timestamp (contracts §2.1).
- Durable, content-addressed artifact publication and safe staged paths
  (rearchitecture phase 1 §7.1).

## Non-responsibilities

- **Fetch a calendar** — `engine/v2/data` does it instead.
- **Decide whether a difference is acceptable** — `engine/v2/diagnosis` does it instead.
- **Decide which artifact is authoritative, or read a job table** — `engine/v2/ops` does it instead.
- **Delete an orphaned artifact** — garbage collection needs catalog reachability and retention, which `engine/v2/ops` owns.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

| Name | What it is |
|---|---|
| `canonical_json`, `content_hash`, `CONTENT_HASH_PREFIX` | RFC 8785 canonical form and `sha256:` identity. Moved here from diagnosis unchanged; hashes pinned by `tests/test_v2_ops_foundation.py`. |
| `from_document`, `to_document`, `parse_schema_version`, `DocumentError` | Strict dataclass ⇄ JSON decoding driven by the contract's own annotations. Errors carry a code and a JSON path, never the offending value. |
| `Clock`, `SystemClock`, `format_timestamp`, `parse_timestamp` | Injected wall/monotonic clocks, and the one timestamp wire form. |
| `ArtifactStore`, `ArtifactError` | Attempt staging directories; copy-hash-fsync-link publication; re-verification before reuse. |
| `artifact_reference` | The identity `ArtifactStore.publish_bytes` would give some bytes, computed without touching storage — so a worker or coordinator that already holds the exact bytes of a published artifact can recompute its `ArtifactRef` and agree with the store by construction. |
| `safe_relative_path`, `ensure_directory`, `fsync_directory` | The path and durability primitives the store is built from. |

<!-- public-interface: canonical_json, content_hash, CONTENT_HASH_PREFIX, from_document, to_document, parse_schema_version, DocumentError, Clock, SystemClock, format_timestamp, parse_timestamp, ArtifactStore, ArtifactError, artifact_reference, safe_relative_path, ensure_directory, fsync_directory, artifacts, canonical, clock, typed -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

- `engine/v2/diagnosis` — re-exports `canonical_json` and `content_hash`, so the
  corpus, receipts and baseline hash through the one implementation.

- engine/v2/data — `DocumentError`, `from_document`, `parse_timestamp` for the strict document checks `engine.v2.foundation.typed` cannot express from annotations alone.
- engine/v2/ops — content identity, safe artifact storage, typed decoding and clocks.
- engine/v2/ledger — canonical append-only payloads and durable export paths.
- engine/v2/serving — safe immutable release paths and health timestamps.

<!-- consumers: engine.v2.diagnosis, engine.v2.data, engine.v2.ops, engine.v2.ledger, engine.v2.serving -->

## Usage

```python
from engine.v2.foundation import ArtifactStore, content_hash

store = ArtifactStore("/tmp/ops-example")
staging = store.staging_dir("att_1")
(staging / "scores.json").write_text('{"a": 1}')
ref = store.publish_candidate("att_1", "scores.json", schema_ref="scores.v1.0")
assert store.read_verified(ref) == b'{"a": 1}'
print(ref.content_hash, content_hash({"a": 1}))
```

## Testing

Tier 0: seconds, temporary directories, no data, no network.

- `tests/test_v2_ops_foundation.py` — golden hashes computed by the phase-0
  copy before the move; one canonicalizer in all of v2; strict decoding
  refusals by code and path.
- `tests/test_v2_ops_artifacts.py` — the unsafe inputs the store exists to
  refuse (traversal, symlinks at each component, hard links back to production,
  FIFOs), a stale writer mutating its file after publication, tampered objects,
  and named crash points before and after the link.

A negative control here looks like: plant a symlink from a staging directory
into a production directory and assert publication refuses it with
`UNSAFE_PATH` and leaves no object — not that "publish raised".
