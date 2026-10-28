# `engine/v2/ops/workspace`

## Ownership

Isolated experiment research workspaces under the planned dependency policy
(`checks/ops_dependencies.json`). Contract: [ARCHITECTURE.md](ARCHITECTURE.md).

## Responsibilities

- Verify a pinned snapshot's complete closure read-only (`closure`).
- Create and reopen a write sandbox over that snapshot with its own catalog, store, scratch and output (`overlay`).
- Refuse supervisor or job execution from inside a workspace (`guard`).

## Non-responsibilities

- **Copy production data** — the snapshot is read in place.
- **Build panels, bounds or registrations** — consumers do, inside the workspace.
- **Resolve a head or write one** — a workspace resolves by snapshot id only.

## Public interface

`overlay.create_workspace`, `overlay.open_workspace`, `overlay.ResearchWorkspace`,
`overlay.SnapshotOnlyRepository`, `closure.verify_snapshot_closure`, `closure.SnapshotClosure`
and `guard.refuse_workspace_execution` (the last is imported by `supervisor` and `executor`).
