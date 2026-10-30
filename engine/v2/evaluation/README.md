# `engine/v2/evaluation`

## Ownership

Implements the **Evaluation/portfolio — realized outcomes, capital accounting, report generation** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**6** of §4.1.

Replaces (§4.4): `evaluate.py`, `report.py`, `build_trades.py`, `calibrate.py`, `recalibrate.py`.

## Responsibilities

- Realized outcomes against actual traded or quoted evidence.
- Capital accounting and report generation.

## Non-responsibilities

- **Recreate the selection logic used to choose trades** — `engine/v2/scoring` does it instead.
- **Use fitted vendor marks as a realized PnL source** — `engine/v2/ledger, from settlement evidence` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

`premium_settlement.settle_premiums(records, *, initial_cash, max_trade_debit, max_bucket_debit, max_concurrent, report_end, same_time_order, on_action=None)` computes a detached premium-funded settlement book from normalized opportunities. Explicit cash totals already include costs; it does not price or select trades. See the [input, output and refusal contract](ARCHITECTURE.md).

<!-- public-interface: settle_premiums -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

Only synthetic tests currently call the helper. No package or production entrypoint imports it; the existing distribution book remains unchanged. A later adapter can project its admitted records and preserve its own schema/errors without moving experiment admission into this package.

<!-- consumers: none -->

## Usage

```python
from datetime import date
from engine.v2.evaluation.premium_settlement import settle_premiums

result = settle_premiums([], initial_cash=1, max_trade_debit=1,
    max_bucket_debit=1, max_concurrent=1, report_end=date(2026, 1, 1),
    same_time_order="settlements_before_entries")
assert result["summary"]["funded"] == 0
```

For long calls the caller supplies bounded stage logging and an `on_action()` callback (for example an existing rate-limited progress tick). The core calls it once per scheduled action, including no-ops, and does not catch callback exceptions. There is no built-in logging, byte cap or process containment.

## Testing

`tests/v2/evaluation/test_premium_settlement.py` uses public synthetic records: complete hand-computed documents, exact money/cap boundaries, no-op curves, signed/missing credits, all explicit ordering choices, detached inputs and typed invalid-input/output-overflow refusals. A normalized differential oracle compares complete outputs with the landed distribution book across multi-event policies; this does not read licensed fixtures, fit models or establish real-market feasibility.
