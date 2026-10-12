# Massive acquisition

## Ownership
Native store-tier acquisition owner; see [ARCHITECTURE.md](ARCHITECTURE.md).

## Responsibilities
Validate one request, reuse verified raw bytes, coordinate a bounded provider edge and fenced receipt publication.

## Non-responsibilities
Research admission, normalization, snapshots, pagination, interval reuse and provider entitlement verification.

## Public interface
Internal ops module `acquisition`: `Parameters`, `scope_plan`, `job_kind`, `run`. No cross-package research API.

## Consumers
`engine.v2.ops.massive_acquire`, `engine.v2.ops.stages`, `engine.v2.ops.worker`.

## Usage
After verifying account access and data-use terms, initialize an ops root with the normal operator commands and configure its `massive-stocks` provider call budget. Keep the API key in the existing environment configuration, never command arguments. Submission itself does not fetch:
```sh
python -m engine.v2.ops.massive_acquire --root ROOT --idempotency-key REQUEST_ID --ticker EXAMPLE --dataset daily --start 2026-08-01 --end 2026-08-31 --as-of 2026-10-12 --adjustment raw
python -m engine.v2.ops --root ROOT serve
```
Use `splits` or `dividends` for corporate actions; adjustment remains explicit. One job reserves one call even if cached. A new idempotency key authorizes a distinct job; the same key and request returns the existing job. The Basic policy is versioned, not a live subscription probe. This implementation supports only one-day stock bars: second, minute, hour, week, month, quarter and year aggregate intervals are excluded, as are S3 flat files; this is implementation scope, not a claim about live entitlement. Storage/throughput sizing requires a bounded authorized pilot, not extrapolation here.

## Testing
Synthetic tests in `tests/v2/ops/test_massive_acquisition.py`, `test_massive_http.py` and `test_massive_submission.py` exercise real catalogs/objects, trusted subprocess launch and fake HTTP responses. No live provider run is required.
