# Massive supervised raw acquisition

## Purpose
Store-tier owner of one explicit US-stock daily OHLCV, split or dividend request under the versioned Basic policy. No research read, normalization, snapshot publication, pagination, refresh or fallback provider.

## Interfaces and dependencies
`acquisition.Parameters`, `scope_plan`, `job_kind` and `run` are internal ops interfaces. `ops.massive_acquire` submits; `stages` registers; `worker.dispatch` invokes `run`. `massive_plan` owns canonical request/policy/cache validation, `provider_budget` account admission/accounting, `lifecycle` fences, and `data.incremental.cache_raw_receipt` raw persistence. `providers.massive` owns the fixed HTTPS edge; it depends on Requests, stdlib, `ops.errors` and the shared 16 MiB limit in `providers.massive_policy`, which the planner also uses.

## Inputs and outputs
One strict ticker/dataset/date/adjustment scope, policy version and expected request hash; a fixed source account and one reserved call. Valid symbols are uppercased before deduplication and request identity; daily response symbols compare case-insensitively. Existing uppercase request identities are unchanged; old lowercase raw keys are not migrated or silently aliased. Trusted executor context supplies the catalog, object store and attempt/fence, never submission paths. The standalone submitter requires an initialized catalog and creates only a shadow job; `ops serve` executes it later.
Success writes `massive_acquisition.v1`: request hash, raw receipt/hash, first receipt time, this job's observation time, attempt/fence, cache-hit flag, HTTP-attempt count and policy identity. Supervisor completion publishes this output under its ordinary fence. A raw hit proves verified response bytes, not normalized coverage, entitlement, EOD finality or historical knowledge.

## Invariants
- Account admission serializes this producer's jobs. The worker rechecks the exact cache after admission; verified hits consume zero HTTP attempts/charges but still require an available account reservation. Cross-job checkpoint reuse is disabled; raw receipts are the only fetch cache.
- One job attempt, no client retries or redirects. The shared account is charged durably immediately before HTTP; crashes do not refund uncertain calls. Versioned policy paces charge admission and extends eligibility after observed responses/failures. This does not police other clients or prove socket-send spacing after arbitrary process stalls. No quota observation is supplied, so `remaining` is conservative local accounting and `uncertain=1` persists after success; it does not itself refuse admission.
- The edge requests identity encoding only, rejects other encodings, and bounds response bytes and whole-request duration. No token-bearing URL, provider error body, response headers or exception text is persisted. Credential echoes refuse before caching. Body-policy refusals preserve observed HTTP 200 accounting; only transport/deadline failures use transient accounting.
- Engine startup may load `.env`; executor launch may forward the declared `MASSIVE_API_KEY` even for a cache hit. The HTTP edge reads that worker variable only on misses. Credentials never enter canonical requests, catalog identity or result artifacts.
- Only valid, nonempty, terminal envelopes become complete raw receipts. Raw identities remain request-plus-content based; identical bytes retain their first receipt time. Successful job artifacts record reuse/acquisition, not an atomic history of every response before crashes.
- Raw bytes publish outside SQLite; the live attempt fence is checked inside the receipt insertion transaction and existing-receipt replay transaction. A rejected fence can leave unreferenced immutable bytes, but cannot expose a new raw catalog receipt. No nested transactions or network operations inside them.
- Supervisor resource admission and attempt-staging cleanup remain authoritative. No cache eviction, retention policy or disk-capacity claim beyond existing admission is supplied.

## Failure semantics
No refusal emits a success artifact or invents partial coverage. A failure after raw commit may leave a valid reusable receipt. Retryable codes retain their registry meaning, but this kind makes no automatic second attempt; a later explicit job is separate work.

| Condition | Outcome |
|---|---|
| Invalid scope, policy, identity, account, dependencies or parameters | `INVALID_REQUEST` before provider access; scope validation precedes storage. |
| Catalog lacks raw-receipt schema | `VALIDATION_FAILED`; no migration. |
| Missing/invalid credential or HTTP 401 | `CREDENTIAL_INVALID`; an observed 401 blocks the account pending operator action. |
| HTTP 403 | `SOURCE_INVALID` with `CAPABILITY_UNAVAILABLE`; does not mark the credential invalid. |
| Other rejected HTTP/envelope, unsupported compression or credential echo | `SOURCE_INVALID`. |
| HTTP 404 | `SOURCE_NOT_FOUND`. |
| Valid empty envelope | `SOURCE_EMPTY`; no negative cache. |
| Continuation present | `SOURCE_NOT_FINAL`; no pagination or complete receipt. |
| HTTP 429 or unelapsed account eligibility | `RATE_LIMITED`; observed 429 persists source backoff. |
| Transport error, deadline or server error | `TRANSIENT_SOURCE`; charged attempts remain spent. |
| Raw size policy exceeded | `RESOURCE_LIMIT_EXCEEDED`; no raw receipt. |
| Corrupt/missing raw evidence or omitted transport accounting | `INTEGRITY_FAILED`; no HTTP fallback over corrupt evidence. |
| Acquired receipt disappears | `INPUT_CHANGED`. |
| Cancellation / stale or expired attempt | `CANCELLED` / `LEASE_LOST`; no fenced publication. |
| Catalog/filesystem unavailable, exhausted reservation or conflicting worker timer | `RESOURCE_UNAVAILABLE`. |

## Diagrams
Operator submit → supervisor admission → verified raw lookup → optional charged HTTPS → fenced raw receipt → supervised result artifact. Research does not enter this flow.
