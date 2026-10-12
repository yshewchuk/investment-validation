# Massive acquisition

## Ownership
Native store-tier acquisition owner; see [ARCHITECTURE.md](ARCHITECTURE.md).

## Responsibilities
Validate requests, reuse verified raw bytes/intervals, coordinate a bounded provider edge and fenced receipt/occurrence publication.

## Non-responsibilities
Research admission, normalization, snapshots and provider entitlement verification.

## Public interface
Internal ops modules `acquisition`, `coverage`, `journal` and `pages`; no cross-package research API.

## Consumers
`engine.v2.ops.massive_acquire`, `engine.v2.ops.stages`, `engine.v2.ops.worker`, `tools.v2_massive_plan`.

## Usage
After verifying account access and data-use terms, initialize an ops root with the normal operator commands and configure its `massive-stocks` provider call budget. Keep the API key in the existing environment configuration, never command arguments. `as_of` cannot be after the current UTC date; both submission and worker enforce this before storage. Submission itself does not fetch:
```sh
python -m engine.v2.ops.massive_acquire --root ROOT --idempotency-key REQUEST_ID --ticker EXAMPLE --dataset daily --start 2026-08-01 --end 2026-08-31 --as-of 2026-10-12 --adjustment raw
python -m engine.v2.ops --root ROOT serve
```
Use `splits` or `dividends` for corporate actions; adjustment remains explicit. One job reserves one call even if cached. A new idempotency key authorizes a distinct job; the same key and request returns the existing job. The Basic policy is versioned, not a live subscription probe. This implementation supports only one-day stock bars: second, minute, hour, week, month, quarter and year aggregate intervals are excluded, as are S3 flat files; this is implementation scope, not a claim about live entitlement. Storage/throughput sizing requires a bounded authorized pilot, not extrapolation here.

For interval/superset reuse, add `--generation initial --max-http-calls 10` to submission. The HTTP cap covers the whole job across retries and continuation pages; each attempt reserves one call and fetches at most one page of an uncovered gap. A new named generation, e.g. `--generation refresh-2026-10-13`, deliberately bypasses other generations and raw-only hits. Reusing that generation reuses its completed intervals. No automatic TTL, eviction or destructive refresh is provided.
An initialized catalog must have the current normal ops migrations. Inspect semantic gaps without submitting or fetching:
```sh
python -m tools.v2_massive_plan --catalog ROOT/catalog.sqlite --store-root ROOT --ticker EXAMPLE --start 2026-08-01 --end 2026-08-31 --as-of 2026-10-12 --adjustment raw --max-calls-per-request 10 --generation initial
```
Planning is bounded/read-only; its HTTP count is a cap, not an exact page estimate. A missing occurrence may still be promotable from a verified exact raw hit under `initial`. The worker replans at execution; the completed result pins original receipts. New generations are acquisition namespaces, not coherent adjustment snapshots. Daily, split and dividend generation ranges must end before the observation's Eastern calendar day. Early cached observations fail once with `INTEGRITY_FAILED`; choose an explicit new generation to acquire fresh evidence. Later announcements/corrections still require explicit refresh.
Partial chains and empty responses never create interval coverage. Same-job retries recover verified page prefixes without refetching committed pages; only terminal complete chains become reusable intervals. Completed gaps survive a failed job, but resubmitting the same key does not restart it. Choose a distinct key only when authorizing a new bounded operation; it cannot inherit the failed job's partial chain or reset its allowance.

## Testing
Synthetic tests in `tests/v2/ops/test_massive_*.py` exercise real catalogs/objects, trusted subprocess launch, generation/gap reuse, cumulative accounting, atomic callbacks, continuation recovery and fake HTTP. No live provider run is required.
