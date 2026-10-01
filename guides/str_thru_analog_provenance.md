# STR-THRU analog population provenance

The STR-THRU forecast-plus-analog experiment uses replayed trades as its
analog population. Its population selector is an explicit caller input,
separate from the stored provenance column. Legacy and native v2 replay
identify distinct populations; combining both can duplicate events.

## Caller contract

`Scorer` accepts an optional single `trade_provenance` string. Omission
retains the existing `engine.replay` filter and empty-population behavior.
An explicit string selects exactly that tag only — it never infers a
population from row contents, selects multiple tags, or rewrites provenance
values. `gate_forecast_analog.build_dataset` forwards the same selector to
`Scorer` through its analog attachment. Call path: experiment runner ->
`build_dataset` -> `_attach_analogs` -> `Scorer` -> bucket enrichment ->
`match_frame`; feature definitions, model settings and walk-forward
selection are unchanged by population selection.

The pinned EXP-147/EXP-184 runner supplies `experiment_trades.PROVENANCE`
for its native replay input; this compatibility boundary covers only that
inherited runner — other runtime callers omit the selector (tests exercise
explicit selection directly), and legacy modules never import v2 for the
provenance constant.

## Refusals and repeatability

An explicit selector producing zero rows refuses before enrichment or
matching, including an already-empty injected frame or an explicit
`engine.replay` request; `build_dataset` likewise refuses an empty base
gate frame under an explicit selector before its early return can bypass
validation. Omission keeps the previous empty-frame behavior. Missing
history for one event stays the existing thin-analog outcome, distinct
from discarding the whole population via an incompatible filter.

Selection writes no files, cache, registry or ledger, has no retries or
transaction boundary, and is deterministic: the same selection over the
same input returns the same rows, same order, same source tags. The
experiment keeps trades and repricing pinned to one named snapshot; this
correction does not establish full v2 provenance for the remaining legacy
panel/forecast/market-state inputs, or prove that fills were executed at
quoted prices.
