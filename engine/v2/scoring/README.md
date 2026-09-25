# `engine/v2/scoring`

## Ownership

Implements the **Scoring application — forecasts, shape, pricing, gate/chooser decisions, diagnostics** row of the §4 owner table of
[system rearchitecture](../../../guides/system_rearchitecture.md), at layer
**5** of §4.1.

Replaces (§4.4): `score.py split by the stages in §6.3`, `entry_rules.py`, `replay.py`, `trailing_cutoff from pnl_sim.py`.

## Responsibilities

- The §6.3 execution order, one module per stage.
- Gate and chooser decisions, including DYN-SV menu resolution.
- Financial diagnostics and a validated immutable ScoreRecord.

## Non-responsibilities

- **Read future outcomes** — `engine/v2/evaluation` does it instead.
- **Mutate a strategy or model registry** — `engine/v2/registry` does it instead.
- **Fit a model during a score request** — `engine/v2/models/training` does it instead.

## Public interface

The names other packages may import. Everything else is internal regardless of
underscore convention, and an import of a name absent from this list fails
`checks/package_readmes.py`.

`application` provides the shared scoring kernel; `compatibility` is the
temporary legacy adapter; `financial` owns diagnostics; `identity` owns
content-addressed request identities; `frozen_batch` is the Phase 6 batch
boundary that preflights a declared `ScoreBatch` and runs every request
through `application.score_frozen` under one pinned snapshot, release and
`FrozenInference`; `frozen_inputs` is the Phase 6 (P6-2) production frozen
inference input builder — `binding_feature_row` decides inclusion for one
binding from its captured feature vector (nonfinite-tag decoding, non-finite
omission, gate derived-column deferral, malformed/missing refusal) and
`build_inference_requests` turns a record's `NativeScoreInputs` and a
release's bindings into the `InferenceRequest` sequence, reading
`role_model_inputs` per-role rows or the merged `model_inputs` for
forecast-family roles, and `validate_answer_free` refuses a record whose
captured blocks smuggle a calculated answer in (the forbidden-field map) or
that arrives with a prebuilt geometry or pricing object.
`checks/phase4_frozen_bridge.py` and `tools/capture_tier0_corpus.py` are its
compatibility callers, so replay and future native workers share one
implementation.

<!-- public-interface: application, compatibility, financial, frozen_batch, frozen_executor, frozen_inputs, identity, source_inputs, stages, FrozenInputsError, binding_feature_row, build_inference_requests, validate_answer_free, canonical_request, dependency_hash, financial_diagnostics, FrozenBatchPreflightError, request_hash, replay, score_batch, score_event, score_frozen, score_frozen_batch, score_id, score_many, score_one, NativeScoreInputs, SourceBundle, FrozenStageExecutor, FrozenStageRefusal, FrozenStageResult, STAGE_NAMES, StageReceipt, build_native_score_inputs -->

## Consumers

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

`engine.v2.models.training` (P5-4, `training/payoff.py`) imports
`native_payoff`'s pure fitting math (`fit_payoff_line`,
`fit_runup_payoff_surface`) to build a payoff-calibration artifact from
causal source rows -- the one place besides this package's own
compatibility path that fitting math runs, and it is layer 6 importing
layer 5 (strictly lower), never the reverse. `engine.v2.ops.cli` imports
`score_one`, `NativeScoreInputs` and `StageReceipt` for the read-only
`ops rescore` command, which re-scores one already-captured event inside a
`no_fit_guard()` block.

<!-- consumers: engine.v2.models.training, engine.v2.ops -->

## Usage

The application takes a ScoreRequest and NativeScoreInputs. Every score must
carry context, feature, forecast, geometry, pricing, analog, simulation, gate,
chooser and serialization receipts. Legacy scoring remains available only
through the explicit compatibility module for comparison.

### Answer-free native inputs

Acceptance and parity callers start with `SourceBundle` and call
`build_native_score_inputs(bundle)` before `score_one`. Supply raw context and
quotes, the feature vector and missing mask, model identity and artifact
references, and recipes for forecasts, residual simulation, analogs and the
gate. The builder rejects calculated answer fields and leaves geometry,
pricing, simulation summaries and decisions for the native stages to produce.

The DYN-SV chooser's vector is derived the same way (`chooser_inputs.py`
builds the block, `native_chooser.py` assembles it, `native_chooser_features.py`
holds the ported `Scorer._chooser_frame` arithmetic): `chooser_recipe` names the
champion binding, its `producers` (the implied_t1/runup_move Tier-4 folds) and
the keys of the frozen `chooser_analog_pool` and `chooser_admissible_table`;
`chooser_fold_pools` carries the served folds' pools. Only the 17 primitive
features (event history, market/regime block, `dte_entry`) come from
`feature_vector`, which still wins for any derived column it declares.

The current builder is deliberately bounded to STR-THRU and its declared
recipes. To add another strategy or recipe, extend `source_inputs.py` and add
tests proving both execution from source inputs and rejection of injected
answers. Do not bypass the boundary by hand-constructing `NativeScoreInputs`
from legacy-selected contracts, forecasts, prices, simulation results, gate
decisions or diagnostics. Legacy records are expected values for the
comparator only.

## Feature-change contract

Before implementing a new or changed scoring stage, declare its authoritative
inputs, outputs, ownership, units, missing-data and refusal behavior, and
provenance. Emit the bounded Phase 4 checkpoint at the boundary that owns the
result:

- model feature vector, missing mask and model identity;
- selected legs and entry cost;
- simulation horizon, capital denominator, residual-population identity, draw
  count and seed;
- gate inputs; and
- DYN-SV candidate eligibility and ranking values when applicable.

Use the opt-in `engine.score.Phase4TraceCollector` for source-bound evidence:
construct it with `retain_full_trace=False` and
`engine.v2.diagnosis.content_hash`, pass it as `trace=` to
`engine.score.Scorer.score`, then persist `diagnostic_checkpoint()`. Set
`retain_full_trace=True` only for a small local diagnostic case; it is not an
acceptance artifact.

Real captures use `tools.phase4_checkpoint_sink.DiskCheckpointSink`. Place
shared resource files below the sink root and register each once with
`write_resource`. On restart, skip `completed_case_ids()`, write each finished
case immediately with `write_case`, and call `finalize` after all cases and
resources are present. The sink writes atomic, hash-bound per-case files and a
deterministic manifest; keep only case IDs and compact indexes in memory. See
`tools/capture_tier0_corpus.py` for the production wiring. Do not retain the
full corpus or full DataFrames for tracing.

Add a focused regression test and a planted-defect test for every checkpointed
output. Include direct, batch and shuffled-input cases where relevant. Legacy
outputs are expected results only: never supply them as native inputs or weaken
an acceptance check to fit a fixture. Exhaustive internal tracing is optional;
the acceptance evidence remains bounded checkpoints and full final-record
parity. Code review must verify that the default path is unchanged when tracing
is disabled.

## Testing

Tier 0 (`component_contracts.md` §15.3): seconds, from frozen fixtures, no
panel load, no network, no fitting. Fixtures live in the private
`fixtures/tier0/` corpus (`checks/tier0_corpus.py`), never in this repo — they
carry licensed quotes.

A negative control here looks like: corrupt one field of a frozen record, run
the comparator, and assert it names **this package's stage** and that field
path — not that "a row is red". A check that has never failed is not known to
work.
