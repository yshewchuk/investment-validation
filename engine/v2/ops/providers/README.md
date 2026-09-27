# `engine/v2/ops/providers`

## Ownership

Native provider edges for the ops layer's incremental refreshes: ORATS
(`daily_market`) and, as of spec s4c, Nasdaq's forward-calendar discovery
endpoint and the `yfinance` library's per-ticker earnings/history calls (the
forward-calendar and computed-moves stores). Each fetcher classifies its own
response into an ops-level acquisition code before anything is cached — the
ORATS fetcher raises its own codes directly (`CREDENTIAL_INVALID`,
`RATE_LIMITED`, `TRANSIENT_SOURCE`, `SOURCE_NOT_FINAL`); the Nasdaq/yfinance
fetchers instead return one of `engine.v2.ops.unit_receipts.RESPONSE_KINDS`
(`complete`/`legitimate_empty`/`not_final`/`transient`/`refused`/
`credential_invalid`), which the store maps to a failure code via
`provider_failure_code` (`SOURCE_NOT_FINAL`/`TRANSIENT_SOURCE`/
`SOURCE_INVALID`/`CREDENTIAL_INVALID`) — two different classification shapes
for the same purpose, not a shared one. Either way it belongs to
`engine.v2.ops`; the data layer's refresh wrappers never import these edges.

## Responsibilities

- Turn one planned `daily_market` fetch unit into one ORATS acquisition: the
  `hist/summaries` and `hist/cores` responses, their classification, and the
  ported ticker rows.
- Turn one planned forward-calendar date unit into one Nasdaq
  `calendar/earnings` acquisition (`nasdaq_calendar.py`), and one ticker into
  one yfinance earnings/history acquisition (`yfinance_edge.py`) — both
  keyless and unmetered. Unlike the ORATS fetcher (which raises its failure
  code itself), these two return their classification as a plain
  `response_kind` string and let the calling store decide the failure code
  via `provider_failure_code` — see "Ownership" above.

## Non-responsibilities

- **Commit a snapshot** — `engine.v2.data.incremental.run_daily_market_refresh`
  does it instead; the forward-calendar/computed-moves stores commit through
  `engine.v2.data.generic_incremental`, not this package.
- **Read credentials or touch the network at construction** — ORATS' key is
  read only when a returned fetcher is called; the Nasdaq/yfinance fetchers
  never read credentials at all (both accounts are keyless) and `yfinance`
  itself is imported lazily, only inside the default callables.
- **Cache or classify a raw receipt** — that is `engine.v2.ops.unit_receipts`'s
  job; a fetcher here only returns `(raw_bytes, response_kind, response_meta,
  rows)`.

## Public interface

`orats_daily_market` provides `orats_daily_market_fetcher` and its
`SUMMARY_FIELDS` mapping; `nasdaq_calendar` provides
`nasdaq_calendar_fetcher`; `yfinance_edge` provides
`yfinance_earnings_fetcher` and `yfinance_history_fetcher`; the package
exports `provider_credentials` and its `PROVIDER_CREDENTIAL_VARIABLES`
account table (three accounts: `orats-daily-market` keyed, `nasdaq`/
`yfinance` keyless).

<!-- public-interface: orats_daily_market_fetcher, SUMMARY_FIELDS, provider_credentials, PROVIDER_CREDENTIAL_VARIABLES, nasdaq_calendar_fetcher, yfinance_earnings_fetcher, yfinance_history_fetcher -->

## Consumers

`engine.v2.ops.incremental_data` injects the native ORATS fetcher through
`_load_data_refresh_callback`; `engine.v2.ops.executor` copies the job's
provider account credentials into the worker environment at launch. The
Nasdaq/yfinance fetchers are consumed as injected parameters —
`forward_calendar_store.run_forward_calendar_refresh(..., nasdaq_fetcher=,
earnings_fetcher=)` — rather than imported by the store itself; today
nothing in production constructs and passes them (spec s4c Parts 3-4 wire
that job kind into `nightly.py`/the worker's dispatch), so only
`tests/test_v2_ops_providers_nasdaq.py`/`_yfinance.py` import
`nasdaq_calendar_fetcher`/`yfinance_earnings_fetcher`/`yfinance_history_fetcher`
directly. No production package imports this subpackage for those two edges
yet.

<!-- consumers: engine.v2.ops -->

## Usage

```python
from engine.v2.ops.providers import orats_daily_market_fetcher

fetcher = orats_daily_market_fetcher()  # reads ORATS_API_KEY only when called
```

```python
from engine.v2.ops.providers import nasdaq_calendar_fetcher, yfinance_earnings_fetcher

nasdaq_fetcher = nasdaq_calendar_fetcher()      # keyless; no credential read
earnings_fetcher = yfinance_earnings_fetcher()  # keyless; imports yfinance lazily
```

## Testing

`tests/test_v2_ops_providers_orats.py` proves the response mapping and the
fail-closed wiring; every test injects `http_get`, so no network is touched.
`tests/test_v2_ops_providers_nasdaq.py`/`_yfinance.py` do the same for the
two s4c edges, injecting `http_get`/`history_fn`/`earnings_fn` respectively.