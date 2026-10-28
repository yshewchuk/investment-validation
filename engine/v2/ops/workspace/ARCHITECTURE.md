# `engine/v2/ops/workspace` — architecture

An isolated experiment write sandbox over one **read-only** pinned snapshot: an overlay, never a copy. Library only: no CLI, no heads, no scope, no supervisor. Modules: `closure.py` (read-only closure verification), `overlay.py` (create/open), `guard.py` (execution refusal).

## Closure (`closure.verify_snapshot_closure(conn, store, snapshot_id, manifest_hash, *, deep=False) -> SnapshotClosure`)

Reads only: catalog queries on the caller's connection and one `lstat` per object (`deep` also re-hashes via `objects.verify_object_path`). The production catalog is trusted local data written by our own pipeline; every catalog read, including `resolve_full`'s, goes through one bounded cursor (over 2,000,000 rows or 256 MiB in one read refuses `closure_rows_over_cap` / `closure_bytes_over_cap`) that exists only to catch our own bugs, not hostile rows. The workspace's own `mode=ro` read connection also sets SQLite's `SQLITE_LIMIT_LENGTH` to 64 MiB, so a single value above it is refused (`closure_bytes_over_cap`) before it reaches Python. `Repository.resolve_full` re-derives dataset versions, fragments, objects and contracts; the module then checks the rest. Every defect is `SNAPSHOT_CLOSURE_BROKEN` (integrity, non-retryable) with `details.reason`; nothing is repaired.

| Defect | `reason` |
| --- | --- |
| `resolve_full` fails (unknown id, missing contract/version/fragment row, identity mismatch); resolved manifest hash differs from the pin | `snapshot_unresolved` (`cause` = data code); `manifest_mismatch` |
| a table's parent dataset-version row missing; an ancestor via `parent_snapshot_id` missing, cyclic or over 100,000 deep | `dataset_parent_missing`; `parent_missing`, `parent_chain_invalid` |
| no committed import receipt for the snapshot, or `commit_receipt_ref` naming none of them (`commit_receipt_ref_mismatch`); a `data_receipt_lineage` base missing, uncommitted or cyclic; a coverage row naming an uncommitted receipt | `commit_receipt_missing`; `lineage_receipt_missing`, `lineage_cycle`; `coverage_receipt_missing` |
| object file absent; not a regular file or wrong size; unreadable; (deep) bytes differ from `content_hash` | `object_missing`; `object_changed`; `object_unreadable`; `object_changed` |

`identity_digest` hashes catalog facts (ancestors, receipts, lineage, coverage and the snapshot's own capture rows, every object id, hash and size): stable across restarts. `stat_digest` adds each object's inode and mtime, so a same-size rewrite after open is caught at seal without `deep`. Raw receipts and normalizations behind `daily_market` revisions are outside the closure: snapshot readers scan fragments only. Cost: one `resolve_full` plus about 250,000 `lstat`s for the full closure; progress to stderr every 30 s.

## Workspace (`overlay.create_workspace` / `open_workspace` -> `ResearchWorkspace`)

| Item | Contract |
| --- | --- |
| Layout | `.research_workspace` marker, `workspace.json`, `catalog.sqlite` (`open_catalog`), `store/`, `scratch/`, `output/` (0700). `workspace.json` (`research_workspace.v1.0`): `snapshot_id`, `manifest_hash`, `identity_digest`, resolved production catalog and store paths, `pin_artifact_id`; written last by temp file and `os.replace`. |
| Read pin | The resolved `SnapshotRef` published into the workspace store as `research_workspace_pin.v1.0` and registered in the workspace catalog; `open_workspace` requires it to equal a fresh production resolution byte for byte. |
| Read path | `ResearchWorkspace.source`: `SnapshotOnlyRepository` over a `mode=ro` + `query_only` production connection and store. An allowlist of read methods (`resolve`, `resolve_full`, `scan`, `table_contract`, `fragment_records`, `get_event`, `get_chain`, `get_price_series`, `get_close`, `scan_population_bound`, `explain_dependencies`), each refusing `SNAPSHOT_NOT_FOUND` for any snapshot id or `SnapshotRef` other than the pinned one; every other public method (head lookups included) refuses `WORKSPACE_REFUSED` `source_method_refused`. |
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
| create: `root`, `snapshot_id` or `manifest_hash` of the wrong type | `argument_invalid` |
| reopen: `workspace.json` unreadable or incomplete; `schema_version` differs; recorded production paths differ | `identity_unreadable`; `schema_mismatch`; `production_mismatch` |
| execution root at or under a workspace | `execution_in_workspace` |

At open or seal the closure may also refuse `closure_changed` (digest differs), `pin_mismatch` or `manifest_mismatch`. Progress goes to stderr (`[workspace]`, `[closure]`); no stdout.
