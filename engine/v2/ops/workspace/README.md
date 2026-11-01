# `engine/v2/ops/workspace`

## Ownership

Isolated experiment research workspaces under the planned dependency policy
(`checks/ops_dependencies.json`). Contract: [ARCHITECTURE.md](ARCHITECTURE.md).

## Responsibilities

- Verify a pinned snapshot's complete closure read-only (`closure`).
- Create and reopen a write sandbox over that snapshot with its own catalog, store, scratch and output (`overlay`).
- Bind a logical consumer scope to that immutable workspace snapshot with a read pin (`scope`).
- Copy the pinned snapshot's catalog rows, including its ancestor chain, into the workspace catalog (`catalog_rows`).
- Create the workspace catalog's native scope head over the pinned snapshot, with its receipt's reference inputs (`head`).
- Admit exactly the pinned snapshot's closure objects into the workspace store, on explicit request (`objects`).
- Refuse supervisor or job execution from inside a workspace (`guard`).

## Non-responsibilities

- **Copy production data implicitly** — the snapshot is read in place; catalog rows are copied by `catalog_rows`, object files only by the explicit `objects.admit_closure_objects`, limited to the pinned closure.
- **Build panels, bounds or registrations** — consumers do, inside the workspace.
- **Touch a production scope head** — the scope adapter only records a workspace-local read-pin binding; `head.commit_scope_head` writes a native head into the workspace catalog only, never production's `data_snapshot_heads`.

## Public interface

`overlay.create_workspace`, `overlay.open_workspace`, `overlay.ResearchWorkspace`,
`overlay.SnapshotOnlyRepository`, `closure.verify_snapshot_closure`, `closure.SnapshotClosure`
`scope.validate_size_scope_request`, `scope.resolve_size_scope_snapshot`,
`scope.size_scope_repository`, `scope.WorkspaceScopeRepository`,
`catalog_rows.copy_closure_rows`, `head.commit_scope_head`, `objects.admit_closure_objects`, and
`guard.refuse_workspace_execution` (the last is imported by `supervisor` and `executor`).
