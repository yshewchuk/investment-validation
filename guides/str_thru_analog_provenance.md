# STR-THRU analog population provenance

The STR-THRU forecast-plus-analog experiment uses replayed trades as its
analog population. Its population selector is an explicit caller input,
separate from the stored provenance column. Legacy replay and native v2
replay identify distinct populations; combining both can duplicate events.

## Caller contract

`Scorer` accepts an optional single `trade_provenance` string. Omission
retains the existing `engine.replay` filter and empty-population behavior.
An explicit string selects exactly that tag; it does not infer a population
from row contents, select multiple tags, or rewrite provenance values.

`gate_forecast_analog.build_dataset` forwards the same selector through
its analog attachment to `Scorer`. The pinned EXP-147 runner, also used by
EXP-184, supplies `experiment_trades.PROVENANCE` for its native replay
input. The loader keeps its source rows and their original native tag.
This compatibility boundary covers only the inherited research runner;
other callers omit the selector. Legacy modules do not import v2 for its
provenance constant: that import belongs only in the experiment runner.

The call path is the experiment runner, `build_dataset`, `_attach_analogs`,
`Scorer`, bucket enrichment, and `match_frame`. Analog matching retains its
existing per-event as-of date and considers only trades closed before that
date. Feature definitions, model settings and walk-forward selection are
unchanged by population selection.

## Refusals and repeatability

An explicit selector producing zero rows must fail before enrichment or
matching, including an already-empty injected frame and an explicit
request for `engine.replay`. The dataset builder also refuses an empty
base gate frame under an explicit selector, before its early return can
bypass population validation. Omission retains the previous empty-frame
behavior. Missing history for an individual event remains the
existing thin-analog outcome; it is distinct from discarding the entire
input population through an incompatible provenance filter.

Population selection writes no files, cache, registry or ledger. It has
no retries or transaction boundary. Repeating the same selection over the
same input returns the same rows in the same order, with source tags
unchanged. Report generation and preregistration remain owned by the
experiment harness.

The experiment keeps trades and repricing pinned to one explicitly named
snapshot. Panel, forecast and market-state dependencies on the remaining
legacy path require a recorded content manifest and unchanged before/after
hashes. This population correction does not establish full v2 provenance
for those inputs, or prove that quoted-price fill assumptions were executed.
