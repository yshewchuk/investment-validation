# Ledger settlement policy

F01 remediation, September 5, 2026. This changes the ledger settlement path
only; the other project-assessment findings remain separate work.

## What a prediction means

Schema 2 records a selection rule to execute at the declared decision/entry
close. For a default straddle this means resolving ATM on the entry chain.
For an early-decision structure, replay selects on the declared decision
chain and pins those contracts through entry and exit. Explicit strike and
expiry requests remain fixed; dependent legs retain their geometry.

The scorer persists the complete structure specification, including leg
selectors, quantities, entry/exit/decision offsets, factory defaults,
structure parameters and variant label. Settlement reconstructs that
specification without calling the current strategy factory or forecast model.
The recorded alpha is passed exactly, including zero and values outside the
usual backtest grid. Predictions and canonical trade keys distinguish
parameter variants and explicit ladder requests.

The board quote is an estimate. It is retained with its quote date, legs and
cost, and is never substituted into the denominator of a different replayed
trade. P&L uses the simulated entry cost and exit value of the same contracts.
This is an ORATS quote-fill simulation, not brokerage execution evidence.
Independence from model/gate selection does not mean independent pricing.

## What outcomes disclose

- The frozen board structure and the realized entry legs, including quote
  bid/ask, alpha price, strike, expiry, side and quantity.
- Strike and expiry matches, plus a full-leg contract comparison when both
  sides have legs. A first-leg-only comparison is explicitly identified;
  unavailable comparisons are null, not successful matches.
- Original intended cost, simulated entry cost, and entry-cost drift.
  `entry_cost_drift` is simulated minus intended cost in premium units;
  `entry_cost_drift_fraction` divides that difference by absolute intended
  cost. A zero/missing intended cost has no fractional comparison.
- Calendar reconciliation against the current `earnings_events` table.
  Changed dates/sessions and missing or ambiguous event identities remain
  unresolvable pending reconciliation. IDs include dates, so a moved date
  usually removes the old ID: this is reported as missing, without guessing
  which nearby event replaced it. It is never reported as an unchanged date.

Replay output is assigned back to prediction row IDs within groups sharing
an identical recorded rule and alpha. Ambiguous replay results cannot
overwrite one another. An exit after the requested settlement cutoff cannot
be marked resolved.

## Existing records

Predictions and resolved outcomes remain append-only. Existing resolved P&L
is not recomputed. Original entry-cost drift can be measured from stored
prediction/outcome costs, but an old outcome without recorded legs cannot
establish its actual settled contract. A fresh replay would describe current
data, not recover that missing historical evidence.

Legacy pending rows without a complete selection specification are recorded
as unresolvable with a reproducibility reason. Current factory defaults are
not evidence of their original parameterization. Corrections need original
evidence and the existing explicit supersede workflow; never backfill a
current spec as though it had been recorded at prediction time.

`python3 -m engine.ledger status`, scored-pair exports, calibration reports
and health output expose settlement diagnostics. Summary counts use canonical
settled predictions. Legacy outcomes are labeled `legacy_unverified`;
contract mismatch counts always include the number actually compared.
Existing report files acquire these diagnostics on their next regeneration.
