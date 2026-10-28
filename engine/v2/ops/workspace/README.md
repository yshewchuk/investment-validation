# `engine/v2/ops/workspace`

## Ownership

Isolated experiment research workspaces under the planned dependency policy
(`checks/ops_dependencies.json`). Contract: [ARCHITECTURE.md](ARCHITECTURE.md).

## Responsibilities

- Verify a pinned snapshot's complete closure read-only (`closure`).
- Create and reopen a write sandbox over that snapshot with its own catalog, store, scratch and output (`overlay`).
- Bind a logical consumer scope to that immutable workspace snapshot with a read pin (`scope`).
- Refuse supervisor or job execution from inside a workspace (`guard`).

## Non-responsibilities

- **Copy production data** — the snapshot is read in place.
- **Build panels, bounds or registrations** — consumers do, inside the workspace.
- **Read or update a native scope head** — the scope adapter records a workspace-local binding to the pinned snapshot; it never consults or mutates `data_snapshot_heads`.

## Public interface

`overlay.create_workspace`, `overlay.open_workspace`, `overlay.ResearchWorkspace`,
`overlay.SnapshotOnlyRepository`, `closure.verify_snapshot_closure`, `closure.SnapshotClosure`
`scope.size_scope_repository`, `scope.WorkspaceScopeRepository`, and
`guard.refuse_workspace_execution` (the last is imported by `supervisor` and `executor`).
