# Massive supervised raw acquisition

## Purpose
Store-tier owner of explicit US-stock daily OHLCV, split or dividend acquisition under the versioned Basic policy. Exact mode preserves the one-request contract; generation mode adds refresh, interval reuse and bounded continuation recovery. No research read, normalization, snapshot publication or fallback provider.

## Interfaces and dependencies
`acquisition` owns parameters, validation, both job kinds and `run`; `coverage` owns occurrence identity and bounded interval planning; `journal` owns immutable page progress; `pages` validates row scope, credential-free continuations and closed-local-day eligibility. `ops.massive_acquire` submits; `stages` registers; `worker.dispatch` invokes `run`. `massive_plan` owns canonical requests/policy/exact cache validation, `provider_budget` admission/accounting, `lifecycle` fences, and `data.incremental.cache_raw_receipt` raw persistence. `providers.massive` owns the fixed HTTPS edge and shared 16 MiB response limit.

## Inputs and outputs
One strict ticker/dataset/date/adjustment scope, policy version and expected request hash; a fixed source account and one reserved call. CLI and worker refuse `as_of` after the current UTC date before storage, preventing future-dated scopes from caching bars-so-far as complete. Valid symbols are uppercased before deduplication and request identity; daily response symbols compare case-insensitively. Existing uppercase request identities are unchanged; old lowercase raw keys are not migrated or silently aliased. Trusted executor context supplies the catalog, object store and attempt/fence, never submission paths. The standalone submitter requires an initialized catalog and creates only a shadow job; `ops serve` executes it later.
Success writes `massive_acquisition.v1`: request hash, raw receipt/hash, first receipt time, this job's observation time, attempt/fence, cache-hit flag, HTTP-attempt count and policy identity. Every success produces this job output, including cache hits, so `no_work` is false. Supervisor completion publishes this output under its ordinary fence. A raw hit proves verified response bytes, not normalized coverage, entitlement, EOD finality or historical knowledge.
Explicit `generation` selects `massive_history_acquisition`: a plain namespace, plus `max_http_calls` (1–100). Its result is `massive_history_acquisition.v1`, pinning the interval plan hash, clipped bindings and original occurrence/request/receipt evidence. `initial` may adopt an exact raw hit after stronger validation, using its original receipt time. Other generations bypass raw-only hits and fetch fresh responses; completed same-generation intervals remain reusable.

## Interval evidence and refresh
- Ops migration 12 adds append-only `ops_massive_occurrences`, indexed by semantic family/generation/date bounds and unique by job/request, plus `ops_massive_pages` keyed by job/original-gap/page-index. Content identity includes original request, receipt/hash, first receipt time, observed time, HTTP-versus-raw-reuse origin and producer job/attempt/fence. An identical HTTP refresh preserves raw identity/time and appends a distinct occurrence.
- Family identity includes the canonical non-date request, adjustment, policy and validator version. No ticker aliasing, cross-provider fallback or adjustment substitution. SQL selects at most 100 intersecting occurrences; a 101st refuses. Verification reads at most 64 MiB of declared raw bytes, rechecking original request/object identity and row scope before reuse.
- Half-open calendar intervals are split at coverage boundaries. Overlaps choose the greatest `(observed_at, occurrence_id)` deterministically; adjacent equal bindings coalesce, and only maximal uncovered gaps become provider requests. Original superset bytes are never rewritten into a synthetic smaller response.
- A plan describes current catalog evidence. Execution replans under account admission and between retries; the final immutable result pins the chosen mosaic. A later refresh/replan cannot mutate that result. Generation is a local cache namespace, not a provider revision snapshot or a coherent split-adjustment vintage.
- Coverage requires a valid nonempty terminal response or a verified contiguous, scope-consistent nonempty page chain ending in one. Completion pins ordered original page IDs; the final page and completion occurrence commit atomically. It does not prove a trading-session denominator, complete market data, source finality, permitted use or historical knowledge. Missing bars need not be missing trades; no target/holdout construction belongs here.
- Each job resumes only its own verified page prefix for the currently selected maximal gap. Different jobs may reuse complete same-generation intervals, never another job's incomplete prefix or HTTP allowance. If another completed acquisition changes the gap, the old prefix stays diagnostic and a new exact gap starts. A terminal journal without its matching completion is corrupt, never repair authority.
- Page chains permit at most 100 pages and reuse the same cumulative HTTP cap across retries and gaps. Every original request, predecessor, index, continuation, raw identity and observation is reverified; missing/reordered/cyclic or nonterminal chains supply no interval coverage. Persisted next requests come only from fixed-origin, same-family, bounded continuation validation; unknown/authentication query fields refuse before raw storage.
- Daily requests require `end` before the observation's `America/New_York` date. The worker checks before HTTP and when validating cached evidence. Legacy raw reuse uses first receipt time, so aging cannot rehabilitate bars-so-far. This versioned closed-local-day rule allows later provider corrections; it is not a finality SLA.

## Invariants
- Account admission serializes this producer's jobs. The worker rechecks the exact cache after admission; verified hits consume zero HTTP attempts/charges but still require an available account reservation. Cross-job checkpoint reuse is disabled; raw receipts are the only fetch cache.
- Exact mode has one attempt. Generation mode has at most 101 supervised attempts and acquires at most one page of one gap per attempt, without sleeping in the worker. Before each HTTP call, the fenced budget transaction sums this job/account's used calls across all reservations, including released attempts, and enforces its cumulative cap. No client retries/redirects; crashes do not refund uncertain calls. Versioned pacing/backoff remains account-wide; it does not police other clients or prove socket-send spacing after arbitrary stalls. Without quota observations, `remaining` is conservative local accounting and `uncertain=1` persists after success without itself refusing admission.
- The edge requests identity encoding only, rejects other encodings, and bounds response bytes and whole-request duration. No token-bearing URL, provider error body, response headers or exception text is persisted. Credential echoes refuse before caching. Body-policy refusals preserve observed HTTP 200 accounting; only transport/deadline failures use transient accounting.
- Engine startup may load `.env`; executor launch may forward the declared `MASSIVE_API_KEY` even for a cache hit. The HTTP edge reads that worker variable only on misses. Credentials never enter canonical requests, catalog identity or result artifacts.
- Only validated nonempty pages persist; incomplete pages use partial raw classification and cannot satisfy exact-cache or interval coverage. Generation mode refuses duplicate JSON keys, foreign tickers, invalid/out-of-range row dates and unsafe continuations before persistence. No negative cache or partial interval success.
- Raw bytes publish outside SQLite. The live fence and occurrence callback share the receipt insertion/replay transaction: rollback exposes neither a new receipt nor occurrence. Rejected fences may leave unreferenced immutable bytes. No network/object reads inside that transaction. Postcommit crashes retain verifiable progress; the next attempt reuses it without another HTTP call. Requests that crash before this transaction retain their charge but no successful occurrence.
- Supervisor resource admission and attempt-staging cleanup remain authoritative. No cache eviction, retention policy or disk-capacity claim beyond existing admission is supplied.

## Failure semantics
No refusal emits a success artifact or invents whole-scope coverage. Completed gaps survive a later failure. Generation mode retries under the supervisor with a 13-second delay even for uncharged failures, plus account backoff; attempt exhaustion can stop before the HTTP cap. Same-key submission only reattaches; generic `resume` does not reset terminal jobs. A new key is separately authorized work with a new cap, never continuation of the old budget.

| Condition | Outcome |
|---|---|
| Invalid scope, future `as_of`, policy, identity, account, dependencies, trusted context or parameters | `INVALID_REQUEST` before provider access; validation precedes storage. |
| Catalog lacks raw-receipt schema | `VALIDATION_FAILED`; no migration. |
| Missing/invalid credential or HTTP 401 | `CREDENTIAL_INVALID`; an observed 401 blocks the account pending operator action. |
| HTTP 403 | `SOURCE_INVALID` with `CAPABILITY_UNAVAILABLE`; does not mark the credential invalid. |
| Other rejected HTTP/envelope, non-bytes body, unsupported compression or credential echo | `SOURCE_INVALID`. |
| HTTP 404 | `SOURCE_NOT_FOUND`. |
| Valid empty envelope | `SOURCE_EMPTY`; no negative cache. |
| Valid continuation or remaining interval gaps; unclosed Eastern daily range | `SOURCE_NOT_FINAL`; generation retries retain fenced progress but no whole-scope result. Exact mode still refuses continuations without persistence. |
| HTTP 429 or unelapsed account eligibility | `RATE_LIMITED`; observed 429 persists source backoff. |
| Transport error, deadline or server error | `TRANSIENT_SOURCE`; charged attempts remain spent. |
| Raw size, interval candidate/verification-byte bound or cumulative HTTP cap exceeded | `RESOURCE_LIMIT_EXCEEDED`; no whole-scope result or additional charge. |
| Corrupt/missing raw or occurrence evidence, conflicting occurrence, or omitted accounting | `INTEGRITY_FAILED`; no HTTP fallback over corrupt evidence. |
| Acquired receipt disappears | `INPUT_CHANGED`. |
| Cancellation / stale or expired attempt | `CANCELLED` / `LEASE_LOST`; no fenced publication. |
| Catalog/filesystem unavailable, exhausted reservation or conflicting worker timer | `RESOURCE_UNAVAILABLE`. |

## Diagrams
Operator submit → supervisor admission → verified exact/interval lookup → optional charged gap HTTPS → atomic raw/occurrence receipt → replan/retry or pinned result. Research does not enter this flow.
