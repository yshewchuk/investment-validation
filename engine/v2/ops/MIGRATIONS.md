# Schema migration contract

`migrations.py` applies each numbered step atomically and refuses changed
checksums for applied steps. A migration may opt into SQLite's table-recreate
procedure: foreign-key enforcement is disabled on that connection before
`BEGIN IMMEDIATE`, its statements and the post-migration
`PRAGMA foreign_key_check` run inside the transaction, and enforcement is
restored after commit or rollback. The flag is part of the migration's
checksum; historical unflagged checksums remain unchanged.

| Requirement | Outcome |
|---|---|
| R1 — ordinary migration | Foreign-key enforcement stays enabled; statements and the catalog version record commit together. |
| R2 — recreate migration | `PRAGMA foreign_keys = OFF` is issued before the transaction; changing it inside a transaction is ineffective. The check runs after the statements and before commit. |
| R3 — violation | A non-retryable `OpsError` with `INTEGRITY_FAILED` refuses the migration. |
| R4 — rollback | Any statement, check, or version-record failure rolls back the whole migration; no schema or catalog-version change is left behind. SQLite errors otherwise propagate. |
| R5 — restoration and concurrency | Foreign-key enforcement is restored to `ON` on every exit. The pragma is per connection; other connections retain their setting. `BEGIN IMMEDIATE` serializes catalog writers while readers may continue. |
| R6 — retry | The same pending migration may be retried after correcting its cause; applied steps remain checksum-protected and are never edited. No internal retry occurs. |
| R7 — additive schema | New durable tables are added in a new numbered migration; already-applied migration statements and checksums are immutable. |
| R8 — additive column | A new nullable column is added by a new numbered migration with its own byte bound; existing rows keep the additive default. Migration 15 adds nullable, bounded `resolved_at` to `native_stage_failures`, so resolved history stays in the catalog while health selects only unresolved rows. |
