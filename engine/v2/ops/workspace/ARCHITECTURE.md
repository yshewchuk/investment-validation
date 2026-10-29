# `engine/v2/ops/workspace` — architecture

An isolated experiment write sandbox over one **read-only** pinned snapshot: an overlay, never a copy of its objects (only catalog rows are copied, by `catalog_rows.py`). Library only: no CLI, no native heads, no supervisor. Modules: `closure.py` (read-only closure verification), `overlay.py` (create/open), `scope.py` (workspace-local scope binding), `catalog_rows.py` (bounded copy of the pinned snapshot's catalog rows), `guard.py` (execution refusal).

## Closure (`closure.verify_snapshot_closure(conn, store, snapshot_id, manifest_hash, *, deep=False) -> SnapshotClosure`)

Reads only: catalog queries on the caller's connection and one `lstat` per object (`deep` also re-hashes via `objects.verify_object_path`). The production catalog is trusted local data written by our own pipeline; every catalog read, including `resolve_full`'s, goes through one bounded cursor (over 2,000,000 rows or 256 MiB in one read refuses `closure_rows_over_cap` / `closure_bytes_over_cap`) that exists only to catch our own bugs, not hostile rows. The workspace's own `mode=ro` read connection also sets SQLite's `SQLITE_LIMIT_LENGTH` to 64 MiB, so a single value above it is refused (`closure_bytes_over_cap`) before it reaches Python. `Repository.resolve_full` re-derives dataset versions, fragments, objects and contracts; the module then checks the rest. Every defect is `SNAPSHOT_CLOSURE_BROKEN` (integrity, non-retryable) with `details.reason`; nothing is repaired.

| Defect | `reason` |
| --- | --- |
| `resolve_full` fails (unknown id, missing contract/version/fragment row, identity mismatch); resolved manifest hash differs from the pin | `snapshot_unresolved` (`cause` = data code); `manifest_mismatch` |
| a table's parent dataset-version row missing; an ancestor via `parent_snapshot_id` missing, cyclic or over 100,000 deep | `dataset_parent_missing`; `parent_missing`, `parent_chain_invalid` |
| no committed import receipt for the snapshot, or `commit_receipt_ref` naming none of them (`commit_receipt_ref_mismatch`); a `data_receipt_lineage` base missing, uncommitted or cyclic; a coverage row naming an uncommitted receipt | `commit_receipt_missing`; `lineage_receipt_missing`, `lineage_cycle`; `coverage_receipt_missing` |
| object file absent; not a regular file or wrong size; unreadable; (deep) bytes differ from `content_hash` | `object_missing`; `object_changed`; `object_unreadable`; `object_changed` |

`identity_digest` hashes catalog facts (only the snapshot's own `commit_receipt_ref` receipt counts, so a later same-data import adding another committed receipt does not change it; ancestors, receipts, lineage, coverage and the snapshot's own capture rows, every object id, hash and size): stable across restarts. `stat_digest` adds each object's inode and mtime, so a same-size rewrite after open is caught at seal without `deep`. Raw receipts and normalizations behind `daily_market` revisions are outside the closure: snapshot readers scan fragments only. Cost: one `resolve_full` plus about 250,000 `lstat`s for the full closure; progress to stderr every 30 s.

## Workspace (`overlay.create_workspace` / `open_workspace` -> `ResearchWorkspace`)

| Item | Contract |
| --- | --- |
| Layout | `.research_workspace` marker, `workspace.json`, `catalog.sqlite` (`open_catalog`), `store/`, `scratch/`, `output/` (0700). `workspace.json` (`research_workspace.v1.0`): `snapshot_id`, `manifest_hash`, `identity_digest`, resolved production catalog and store paths, `pin_artifact_id`; written last by temp file and `os.replace`. |
| Read pin | The resolved `SnapshotRef` published into the workspace store as `research_workspace_read_pin.v1.0` and registered in the workspace catalog; `open_workspace` requires it to equal a fresh production resolution byte for byte. |
| Read path | `ResearchWorkspace.source`: `SnapshotOnlyRepository` over a `mode=ro` + `query_only` production connection and store. An allowlist of read methods (`resolve`, `resolve_full`, `scan`, `table_contract`, `fragment_records`, `get_event`, `get_chain`, `get_price_series`, `get_close`, `scan_population_bound`, `explain_dependencies`), each refusing `SNAPSHOT_NOT_FOUND` for any snapshot id or `SnapshotRef` other than the pinned one; every other public method (head lookups included) refuses `WORKSPACE_REFUSED` `source_method_refused`. `ResearchWorkspace.source_snapshot()` resolves the workspace pin and checks its snapshot identity and manifest. |
| Read pins | `ResearchWorkspace.record_read_pin(name, document, *, clock)` is the hook PR 3's preflight calls: it publishes canonical JSON (`research_workspace_read_pin.v1.0`: name, pinned snapshot id and manifest hash, document) into the workspace store and registers it in the workspace catalog; never production. A bad name refuses `pin_name_invalid`. |
| Write path | `conn` (workspace catalog) and `store` only. Derived snapshots go to this local catalog; reading source and derived snapshots through one handle is the consumer's concern. |
| Seal | `verify_before_seal(deep=False)` re-runs the closure; `identity_digest` and `stat_digest` must equal open time. A same-size rewrite that restores mtime is detected only with `deep=True`; the seal path (PR 3 registration seal) must call `verify_before_seal(deep=True)`, while create/open and preflight use the default. |
| Execution | `guard.refuse_workspace_execution(*paths)` runs first in `supervisor.Service.__init__` and `executor.launch`; a path at or under a marked directory (literal or resolved) is refused. |

Create order: validate roots, open production read-only, verify the closure, then lay out; failure removes only what this call created. SQLite may create `-wal`/`-shm` beside the production catalog on a read-only open; the main file, store files and head rows are never written.

| Refusal (`WORKSPACE_REFUSED`, validation, non-retryable) | `reason` |
| --- | --- |
| root equals, contains or lies under the production store or the catalog's directory; reopened catalog is the production file | `production_root` |
| path crosses a symlink; parent missing; existing root not empty; production paths missing | `symlink_path`; `parent_missing`; `root_not_empty`; `production_missing` |
| reopen: marker, `workspace.json`, catalog or a subdirectory missing, a symlink, or the catalog hard-linked | `layout_invalid` |
| create/reopen: `root`, production paths, `snapshot_id` or `manifest_hash` of the wrong type | `argument_invalid` |
| create, reopen or seal: any other storage or metadata failure (filesystem, schema admission, malformed documents), with `cause` the exception class | `storage_failure` |
| reopen: `workspace.json` unreadable or incomplete; `schema_version` differs; recorded production paths differ | `identity_unreadable`; `schema_mismatch`; `production_mismatch` |
| execution root at or under a workspace | `execution_in_workspace` |

At open or seal the closure may also refuse `closure_changed` (digest differs), `pin_mismatch` or `manifest_mismatch`. Progress goes to stderr (`[workspace]`, `[closure]`); no stdout.

## SIZE workspace scope head

`validate_size_scope_request(workspace, scope)` performs the same local
workspace-handle and scope checks as `size_scope_repository` without reading
the pinned source or writing a read pin. `resolve_size_scope_snapshot(workspace,
scope)` applies those checks before reading the pinned source and uses the
shared typed storage-error mapping; it also writes no read pin. Admission uses
this helper so invalid requests fail before source access and snapshot read
failures retain the scope API's typed refusal.

`size_scope_repository(workspace, scope, *, clock)` adapts the workspace's
snapshot-only source to the existing research contract that resolves a scope
head. It publishes a `research_workspace_scope_head.v1.0` read pin in the
workspace catalog and store, binding the exact scope name to the workspace's
`snapshot_id` and `manifest_hash`. The binding identity hashes the schema
version, scope, snapshot id and manifest hash. The returned repository view delegates reads to `workspace.source`;
`resolve_pinned(scope)` returns `workspace.source_snapshot()` only for the
bound scope. It never reads a production head or accepts a different snapshot.

The local catalog does not contain production `data_snapshots`, so this is a
workspace-local scope binding, not a native `data_snapshot_heads` row. The
scope adapter itself copies nothing; `catalog_rows.copy_closure_rows` (below) copies the snapshot's catalog rows, never objects. A standard package or library does
not provide this repository-specific pin contract; the adapter reuses the
existing `ResearchWorkspace.record_read_pin`, `SnapshotOnlyRepository` and
`SnapshotRef` behavior.

| Failure | Outcome |
|---|---|
| Scope is empty, malformed, contains an unpaired surrogate or is over 100 characters | `WORKSPACE_REFUSED`, `scope_invalid`; no workspace access or read pin write. |
| Workspace handle lacks its snapshot reader, source view, pin writer or snapshot identity | `WORKSPACE_REFUSED`, `workspace_invalid`; no read pin write. |
| A different scope is resolved through the returned view | `WORKSPACE_REFUSED`, `scope_mismatch`. |
| Captured binding differs from a subsequently resolved workspace snapshot ID | `SNAPSHOT_CLOSURE_BROKEN`, `pin_mismatch`. If `ResearchWorkspace.source_snapshot()` detects a workspace manifest that no longer matches its pin, it propagates `SNAPSHOT_CLOSURE_BROKEN`, `manifest_mismatch`. |
| The pinned source snapshot is no longer resolvable | Existing `SNAPSHOT_NOT_FOUND` propagates; no fallback. A manifest mismatch propagates as `SNAPSHOT_CLOSURE_BROKEN`, `manifest_mismatch`. |
| Read-pin publication or catalog storage fails | Existing `OpsError`/`DataError` propagates; unexpected exceptions become `WORKSPACE_REFUSED`, `storage_failure`. A complete unreferenced CAS object may remain. |

R1–R6 follow the workspace failure contract: stable typed refusal; no partial
result; no repair or retry; stale scope fails closed; production remains
read-only and the read pin writes only inside the workspace; unknown states
refuse. Existing workspaces and native catalog heads are unchanged.

## SIZE workspace closure catalog rows (`catalog_rows.copy_closure_rows(workspace) -> dict`)

Copies the pinned snapshot S's catalog rows, including its `parent_snapshot_id` chain, from the read-only source connection into `workspace.conn`. A native scope head needs them: `manifest_hash` covers the parent and `data_snapshots.parent_snapshot_id` is a foreign key. No object file, head, receipt, lineage, coverage, capture or reference input is copied or written. The 7 GB catalog rules out a whole-file copy (`VACUUM INTO`, sqlite backup), hence a bounded row copy.

**Bound.** Exactly these rows; the workspace catalog must hold no other row in these tables:

| Table | Rows |
|---|---|
| `data_snapshots` | S and every ancestor through `parent_snapshot_id` |
| `data_snapshot_tables` | S's rows only |
| `data_dataset_versions` | the versions S names plus their `parent_dataset_version_id` chains |
| `data_version_fragments`, `data_fragments`, `data_objects` | those of the versions S names directly (not of ancestors or dataset-version parents) |
| `data_contracts` | contracts of the copied versions and fragments |

Reads are capped cumulatively, across the insert pass and both verification reads, at 8,000,000 rows and 3 GiB of rendered row text, and the foreign-key check reads at most one violation. All inserts are `INSERT OR IGNORE` in one workspace transaction with deferred foreign keys; per table, the workspace row count and an ordered content digest of the closure rows must equal the source's (equal digests make the manifest hash the pinned one; `resolve_full` cannot run inside the transaction). The return value has `snapshot_id`, `manifest_hash`, `chain` (ancestor ids, direct parent first), and per-table `rows`, computed before commit. Progress per table goes to stderr (`[workspace]`).

R1–R6, all `WORKSPACE_REFUSED` unless noted:

| Rule | Behaviour |
|---|---|
| R1 typed reason | `workspace_invalid` (workspace handle already in a transaction), `source_row_missing` (the source has no `data_snapshots`, `data_snapshot_tables`, `data_dataset_versions` or `data_contracts` row for S; fragment and object tables may legitimately be empty), `schema_mismatch` (column lists differ), `copy_rows_over_cap`, `copy_bytes_over_cap`, `unexpected_rows` (workspace holds rows beyond the closure), `row_mismatch` (a digest differs, including a conflicting existing row), `foreign_key_violation`, `storage_failure` (other exceptions, with `cause`). An unknown S propagates the existing `SNAPSHOT_NOT_FOUND`. |
| R2 no partial result | One transaction; any refusal rolls it back, so no row and no head remains. Nothing after commit can refuse. |
| R3 no repair | No overwrite, delete or retry; a conflicting row refuses. |
| R4 stale pin | `workspace.source_snapshot()` runs before the read and before commit; a changed pin propagates `SNAPSHOT_CLOSURE_BROKEN`. Production rows are immutable and the source is read inside one read transaction; a conflicting workspace row is caught by the digest comparison (`row_mismatch`). |
| R5 read-only production | Only the existing read-only source connection is read; all writes go to `workspace.conn`. |
| R6 unknown state | Malformed handle, unreadable rows or any unexpected exception refuses; nothing is guessed. |

Idempotent: a re-run over a complete copy inserts nothing and returns the same result; a re-run after a failure converges because failure leaves no rows.
