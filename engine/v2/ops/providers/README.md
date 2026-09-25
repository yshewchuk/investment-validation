# `engine/v2/ops/providers`

## Ownership

Native provider edges for the ops layer's incremental `daily_market` refresh.
The ORATS fetcher raises ops-level acquisition codes (`CREDENTIAL_INVALID`,
`RATE_LIMITED`, `TRANSIENT_SOURCE`, `SOURCE_NOT_FINAL`), so it belongs to
`engine.v2.ops`; the data layer's refresh wrapper never imports it.

## Responsibilities

- Turn one planned `daily_market` fetch unit into one ORATS acquisition: the
  `hist/summaries` and `hist/cores` responses, their classification, and the
  ported ticker rows.

## Non-responsibilities

- **Commit a snapshot** — `engine.v2.data.incremental.run_daily_market_refresh`
  does it instead.
- **Read credentials or touch the network at construction** — the key is read
  only when a returned fetcher is called.

## Public interface

`orats_daily_market` provides `orats_daily_market_fetcher` and its
`SUMMARY_FIELDS` mapping.

<!-- public-interface: orats_daily_market_fetcher, SUMMARY_FIELDS -->

## Consumers

`engine.v2.ops.incremental_data` injects the native ORATS fetcher through
`_load_data_refresh_callback`; no other production package imports this
subpackage.

<!-- consumers: engine.v2.ops -->

## Usage

```python
from engine.v2.ops.providers import orats_daily_market_fetcher

fetcher = orats_daily_market_fetcher()  # reads ORATS_API_KEY only when called
```

## Testing

`tests/test_v2_ops_providers_orats.py` proves the response mapping and the
fail-closed wiring; every test injects `http_get`, so no network is touched.