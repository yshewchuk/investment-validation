# `engine/v2/research`

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for this component's contracts,
inputs/outputs, dependencies and invariants.

## Ownership

Implements **research tooling — offline analysis CLIs reading a pinned v2
snapshot** at layer **7.0** of [system rearchitecture](../../../guides/system_rearchitecture.md)
§4.1: a peer of `engine.v2.serving`/`engine.v2.ops` and below
`engine.v2.dashboard`, so nothing in the engine imports it back. It is not a
row of the §4 owner table; it was added by Phase 6 slice 6 (decision UD-4).

Also snapshot-pinned research tooling: the native replay and the trade table it
publishes, built on `engine.v2.data.Repository` over committed snapshots rather
than the legacy mutable store.

Replaces (the store-reaching halves of): `tools/signal_screen.py`,
`tools/fill_quality.py`, and the read path of
`engine/data/pulls/polygon_fills.py`. `tools/log_diagnostics.py` is
deliberately **not** moved: it reads only a local CSV transactions log (plus an optional sibling `metrics_*.json`) and needs no
snapshot read, so it has no v2 counterpart. It also replaces the replay-over-a
committed snapshot of `engine/replay.py` and `engine/build_trades.py`'s v2
write path.

## Responsibilities

- Pure analysis cores moved verbatim from the legacy research tools.
- One pinned snapshot identity per run, threaded through every read and output,
  including the experiment loader's explicit holdout context columns. Public
  readers may revalidate that identity: the eligibility planner resolves it at
  admission and again at the independently callable canonical reader boundary;
  repository scan planning/validation also resolves the pinned identity.
- Plan and price events against one resolved `SnapshotRef`, at a grid of fill
  alphas, with the same arithmetic the legacy replay uses.
- Publish replay output as the `trades` table through the generic incremental
  write path, pinned to the snapshot that produced it.
- Reconcile the published `trades` table to the canonical event universe,
  tombstoning only the non-canonical simulated rows.
- Move provenance onto every row (`engine.v2.research.replay`) so a v2 row is
  never mistaken for a legacy `engine.replay` row.

## Non-responsibilities

- **Mutate the trades ledger or fetch from a network provider** —
  `engine/build_trades.py` and `engine/data/pulls` (legacy, unchanged) do it
  instead. The Polygon pull's `build_plan`/`execute` network half stays there.
- **Serve live requests or render a dashboard** — `engine.v2.serving` and
  `engine.v2.dashboard` do it instead.
- **Decide a research conclusion or a verdict** — a person, from the written
  report, does that; these tools only produce the measured table.
- Fetch or normalize quotes (`engine.v2.data`).
- Decide whether to trade: selection lives in `engine.v2.scoring`.
- Mutate the legacy tree: `engine/replay.py` and `engine/build_trades.py` keep
  publishing exactly as before.

## Public interface

`magnitude_folds.plan_regression_folds` shares admitted population/calendar planning
with the unchanged magnitude wrapper, using an explicit validated target descriptor.
It returns `regression_temporal_plan.v1` with 5/3/1 clocks and annual candidate folds;
this is not native registration, target qualification, source publication proof or
nested-search execution. Its only current experiment caller is `iv_crush_plan.plan`.



`horizon_fitted.fitted_forecasts` shares guarded per-horizon fitting for closed
numeric routes and explicit target column/domain, without evaluation labels.
`HorizonFitFailure` retains the original fit exception as its cause so experiment
boundaries own compatibility error policy. Progress/preparation errors propagate;
Callbacks receive detached evidence copies; partial callbacks/diagnostics confer no completed result or experiment admission.
This offline API has no production entrypoint or provider/repository access.

`horizon_fit_inputs.prepare_horizon_inputs` materializes event-ID/column iterables once and shares ordered matrices and frozen
input hashes with an explicit training-target column. It consumes no evaluation
labels and grants no target-domain/source admission or fit authority. Current use
is offline experiment fitting, not a production entrypoint.

`magnitude_distribution.recipe`, `histogram`, `validate_distribution` and `bin_index`
provide a source-free absolute-magnitude histogram format and numerical refusals.
`histogram([1.0] + [0.0] * 80)` returns a detached document with an explicit unresolved tail;
`tail_excesses=[0.0, 2.0]` declares conditional excesses relative to 20 (zero means exactly 20).
No sample is fitted/admitted here. Consumers include the magnitude math/payoff helpers, `experiments.distribution_candidate_recipe`, `experiments.distribution_class_support`,
`experiments/distribution_candidate_fit`, `distribution_fitted`, `distribution_oof` and synthetic tests, including `tests/v2/research/test_magnitude_distribution.py`.

`magnitude_distribution_math` exposes `components`, `cdf`, `mean`, `crps`,
`empirical_crps`, `bin_scores` and `brier`: pure magnitude-family scores and
uniform/lower/upper finite-bin sensitivity. Full-law calls require a declared
empirical tail; bin scores alone allow unresolved tails. Arithmetic normalizes
only the admitted 1e-12 mass roundoff, without modifying the document.
`magnitude_distribution_payoff.bridge_summary`, `empirical_bridge_summary` and
`mean_matched_control` supply frozen-bridge/mean-control diagnostics, preserving
zero atoms and negative post-residual exit values. Fixed per-share spot/cost
inputs are decision-quote proxies, not realized net-profit or execution claims.
`experiments/distribution_oof` consumes `mean` and `mean_matched_control`; `experiments/distribution_scores` composes the magnitude format, proper scores and bridge summaries; matching synthetic tests cover these APIs.
The numerical helpers own no source admission, fit orchestration or production wiring.

`eligibility_policy.eligibility_policy` validates an explicit version/width;
`eligibility_policy.plan_eligibility` returns the opt-in metadata-only plan described below.
`experiment_population.canonical_events` reads the six canonical metadata columns
after exact committed-snapshot admission, and flags same-ticker/date conflicts. `experiment_trades.classify_events` accepts
that frame, snapshot and explicit as-of month; it returns detached canonical rows,
membership labels, structurally ambiguous IDs and pinned holdout context. Malformed
event IDs refuse; invalid scalar dates/sessions are classified as ambiguous.
`snapshot_inputs.snapshot_reference` checks native reference shape only;
`admit_snapshot` additionally resolves and compares the complete committed identity.
The pure classifier checks shape, not commitment of its caller-supplied context.
Neither metadata helper grants payload access or accepts caller-provided membership labels.

`progress.Progress`, `progress.stage` and `progress.decision` provide shared stderr telemetry:
bounded-cadence ticks, stage timing/RSS, optional native refusal codes (no message text) and aggregate decisions. Distribution scoring/reporting also consume these helpers.

`snapshot_rows.iter_snapshot_rows` streams native nullable/integer row projections
with explicit pinned predicates and row/decoded-byte/object/batch caps (default1024).
Rows are provisional until exhaustion; close abandoned streams. `read_snapshot_rows`
collects without partial returns. Caller owns admission; no pandas conversion or writes.

`event_target_store.publish_event_targets` publishes supplied complete earnings metadata
and selected event-keyed targets as a standalone two-table native snapshot. It
retains aliases/conflicts, writes singleton target objects, admits contracts through
native structural decoding, binds asserted source identity/strict JSON provenance
(recursive string keys), and enforces explicit row/object/total-byte caps. Returns the
resolved committed reference, including on reconciled-content replay. Caller owns
admission and output isolation; no source read, arithmetic or holdout policy occurs.
This opt-in primitive has no production caller and is not reachable from a production
entrypoint; callers are explicit `experiments.iv_crush_prepare` workspace preparation
and synthetic tests, not an activated IV search or production experiment.

`iv_crush_inputs.load_iv_crush_targets` reads admitted signed native IV-crush labels
with an explicit label cutoff; `preflight_iv_crush_targets` proves an explicit
participant union without opening target payloads. Both require event-isolated
objects and the exact current contract. Neither publishes targets or runs a study.

`magnitude_history_inputs.admit_stored` verifies stored-feature receipt, contract and
complete physical isolation using metadata only, without deriving a producer context.
It and both stored readers share `magnitude_history_store.preflight_stored`, which
returns ordered event/cutoff pairs after the same metadata-only admission.

`magnitude_history_store` owns build-time native history materialization and its
stored-only reader; `experiments.magnitude_history_store` provides the explicit
source-fingerprinted CLI. See the architecture's stored-history contract.

The names the `tools/v2_*` CLIs import. Those CLIs sit in `tools/`, outside
the `checks/import_layers.py` hook (which only parses `engine*` importers), so
no `engine` package is observed importing this one. The library APIs listed here
are also supported; other names are internal regardless of underscore convention.
`engine/v2/research/_scan.py` (`read_table`, `resolve_snapshot`, `DEFAULT_SCOPE`)
is internal shared machinery, not an interface. The replay path's split
modules (`_snapshot`, `_pricing`, `_plan`, `_chains`, `_trades_table`,
`_trades_revisions`, `_trades_publish`, `_replay_run`, `_build_run`) are
internal too.

`confirmed_metadata.reduce_sequences` is public pure metadata reduction,
not a public index builder. It returns detached per-ticker sequences from a complete
caller-supplied seven-column view, using data's public confirmation/sequence primitives.
`confirmed_metadata.validate_tickers` sorts canonical names without rewriting them;
`confirmed_metadata.metadata_rows(events, *, tickers)` validates/detaches rows/reasons,
not source completeness or cross-row conflicts. The full reducer refuses duplicate IDs or ticker/days, unconfirms
date-conflict and duplicated ticker/cluster rows, and `reduce_with_counts` also returns those unconfirmed counts.
It proves neither source completeness nor coverage/eligibility and performs no I/O.

The replay half: `replay.replay`, `replay.replay_one`, `replay.ALPHA_GRID`,
`_replay_run.run`, `_replay_run.events_frame`, `_plan.plan_events`,
`_chains.ChainIndex`, `_chains.load_chain_index`,
`_chains.filter_plan_by_availability`, `_chains.read_chain_keys`,
`_chains.read_chain_keys_for`, `_chains.read_chains_for_years`,
`_trades_table.to_trades_table`, `_build_run.run`, `reconcile_trades.run`,
`_trades_publish.publish`, `build_trades.coverage`, `_pricing.STRUCTURES`,
`_pricing.trading_calendar_from_snapshot`, `_pricing.execution_variant_label`.

`_chains.read_chains_for_years(repository, snapshot_ref, years, batch_filter=None,
key_filter=())` materializes the fixed chain projection from a pinned snapshot.
Years must be a nonempty, non-scalar-string iterable; empty or scalar str/bytes input
raises `ValueError` before repository access. String elements such as `["2024"]`
remain supported. A batch filter narrows rows or returns `None`; key predicates are forwarded
unchanged. Candidate-row/batch limits do not cap retained memory. This raw reader
performs no experiment holdout admission; errors propagate without retries.

The entry-selection surface also exposes `replay.clean_chain_rows` (drop invalid
quotes without mutating the input) and `_pricing.MIN_MEANINGFUL_COST`,
`ChainSnapshot`, `FillModel`, `StructureError`, `price_structure`: native
selection and spread pricing, with no I/O or alternate pricing implementation.

`experiment_trades.load_trades` is a second kind of entrypoint: a plain
library call (no `tools/v2_*.py` CLI of its own) for a caller that already
holds a `Repository` and a resolved `SnapshotRef` — today only
`experiments/common_v2.py`. `_pricing.trading_calendar_from_snapshot` is a
third, carved out of `_pricing`'s otherwise-internal contents the same way
`_pricing.STRUCTURES` already is, for the same caller.
`experiment_trades.PROVENANCE` is a fourth: the native replay tag constant,
imported only by the pinned
`experiments/EXP-147_str_thru_gate_promotion_confirmatory_val/run.py` runner
to select its analog population (see `guides/str_thru_analog_provenance.md`).
`_pricing.execution_variant_label` is a fifth, carved out for
`experiments/v2_candidate_grid.py`'s `price_candidate_grid` (issue #266
slice 2), which labels each priced grid-position step with the same
execution-variant string `replay()` itself uses.

`experiment_exits.walk_exit` consumes `EnteredPosition`/`PositionLeg` contracts
plus resolved fixed-day or target/stop economics and returns `ExitDecision`
values from one pinned snapshot. Target/stop thresholds are net mark-based
position P&L. Optional `exit.threshold_units` accepts `quote_price_times_quantity`
(the omitted-field behavior) or `entry_debit_fraction`; the latter scales signed
ratios by each position's positive net opening debit at declared alpha, excluding
commissions, fees and contract multipliers. Zero/credit debit and nonfinite or
zero converted thresholds refuse. If one day's alpha 0/1 aggregate mark range spans both
thresholds, the stop is selected, marked ambiguous, and priced at the worst
alpha (0.0) so its result cannot be better than the stop. `exit_report_frame`
projects decisions into per-trade `exit_reason`/`exit_day` columns with P&L
basis, mark provenance and the declared vs. applied exit alpha. Neither function
writes results or combines mark-based P&L with fill-validated performance.

`experiment_population.load_population` returns holdout-admitted canonical event
metadata, including for smoke-only `native_gate_pricing` and `native_gate_evaluation`
supervisor workers through `v2_gate_variant.price_variant` and `gate_report`.
`prediction_inputs.load_prediction_targets` reads eligible positive-move labels
and their availability dates from that same pinned snapshot; it has no supervisor
caller. Neither input reader writes reports or ledger rows.

`event_history.load_event_history` returns admitted prior-event features and
consumed-object/cutoff provenance; mixed outcome fragments refuse before access.

`magnitude_inputs.forecast_cutoff` defines full-session forecast ceilings;
`magnitude_inputs.load_magnitude_targets` reads isolated magnitude labels under
a separate strict label-availability cutoff. Neither is a runnable experiment.
`magnitude_inputs.preflight_magnitude_targets` proves an explicit population's
physical isolation; `qualify_magnitude_training_targets` returns strictly
fit-available training magnitudes with complete read/omission provenance.

`magnitude_features.load_magnitude_history` binds one admitted event's pinned
calendar, forecast ceiling and protected history features, explicitly retaining
unverified publication/schedule provenance and all decoded history inputs.

`magnitude_folds.plan_magnitude_folds` binds metadata-only expanding annual
groups, all three forecast horizons and conservative fitting boundaries.
Training IDs are candidates pending label-availability admission; no labels,
history aggregates, model fitting or experiment registration are read/created.
`magnitude_folds.magnitude_window` shares pure canonical window validation
with native preregistration without accessing a repository.
`magnitude_folds.annual_fold` shares that planner's unchanged
annual rule for already-admitted metadata, with native `POPULATION_COLLAPSED` refusals.

`magnitude_panel_table.CONTRACT` defines the stage-2 `research_magnitude_panel.v1`
raw table, ordered H/HM/HMO dictionaries and complete-cohort row validation.
`magnitude_panel_plan.build_context` and `preflight_context` bind the exact source
and stored-history snapshots, reconstruct population/clocks, and prove history
and prior-outcome fragment isolation using metadata only. Context hashes are
explicitly distinct from history hashes. `validate_context` is a pure structural
check, not a repository admission or permission to open payloads. It reconstructs
each horizon cutoff from the context-bound observed calendar, not supplied hashes alone.
It also checks calendar-source metadata against the history/snapshot bindings and
closed admission fragment shapes; only repository preflight authenticates their contents.
This foundation has no CLI, materializer, panel reader, model or search integration.
Raw market/price payload reads remain **not authorized** by its returned plan;
this API does not implement raw-source protected-event policy or grant read authority.
Session dates alone never establish publication evidence. See the
[stage-2 design](../../../experiments/SUPER_SIZE/STAGE2_DESIGN.md).

The manual no-ledger `experiments.v2_gate_variant` CLI consumes
`experiment_population.load_population`, `_pricing.STRUCTURES`, and
`replay.replay`/`replay.ReplayResult` for fresh per-arm fill pricing.
`experiments.gate_report` combines protected historical features, pinned
calendar plans and fresh per-arm pricing for exploratory fold reports.

`magnitude_panel_daily.build_daily_blocks` constructs only reconstructed partial
daily-market and `spy_*` regime blocks from an exact parent panel plan and the opt-in grouped
eligibility policy. The nine `spy_*` values read a pinned `price_history` snapshot's `REGIME_TICKER`
(`^GSPC`) close -- one regime source, never a `daily_market` SPY spot (the names are historical);
the snapshot's identity is bound into the context. It requires explicit source/event/physical-and-expanded byte bounds, retains all horizons
of each retained event, and shares one value scan per event ticker/year (the regime series is read
once as of the latest ceiling). Missing/late
market cap and zero implied quotes stay null with explicit reasons. Its separate
context binds producer-declared label bounds without verifying them; neither the
result nor the parent metadata-only context grants training, observed-PIT, holdout
or stored-panel admission. There is no writer, CLI, fit or automatic execution.

`magnitude_panel_v2_policies.policy_definitions()` owns closed logical policy
metadata for `magnitude_panel_v2_table`. The latter exports `CONTRACT`,
`feature_dictionary(subset)` and `recipe(subset)` with mandatory plain H/HM/HMO
subset strings: invalid values raise `CONTRACT_MISMATCH`, omission raises Python `TypeError`. Numeric widths are 11/28/42;
raw `model_columns` widths are 11/28/43, with only HMO's final quote indicator
outside the numeric table. Uniform ten-field descriptors inline source and
derivation policies; all v2 dictionary hashes differ, including H, while v1
remains unchanged. Recipe `missing_indicators` declares family-owned training
preprocessing, not stored missing flags. Containers are detached; native table
and dictionary hashes are distinct. Synthetic tests and the source-free versioned compiler consume recipes:
no v2 context, reader, source qualification, fit or search activation exists.
Resolved source snapshots must be bound separately by later native admission.

<!-- public-interface: snapshot_rows.iter_snapshot_rows, snapshot_rows.read_snapshot_rows, event_target_store.publish_event_targets, regression_evidence.normalize_forecasts, regression_evidence.error_metrics, magnitude_confirmed_metadata.build_index, confirmed_metadata.reduce_sequences, confirmed_metadata.reduce_with_counts, confirmed_metadata.validate_tickers, confirmed_metadata.metadata_rows, magnitude_distribution_math.components, magnitude_distribution_math.cdf, magnitude_distribution_math.mean, magnitude_distribution_math.crps, magnitude_distribution_math.empirical_crps, magnitude_distribution_math.bin_scores, magnitude_distribution_math.brier, magnitude_distribution_payoff.bridge_summary, magnitude_distribution_payoff.empirical_bridge_summary, magnitude_distribution_payoff.mean_matched_control, magnitude_distribution.recipe, magnitude_distribution.histogram, magnitude_distribution.validate_distribution, magnitude_distribution.bin_index, magnitude_panel_v2_table.CONTRACT, magnitude_panel_v2_table.feature_dictionary, magnitude_panel_v2_table.recipe, magnitude_panel_v2_policies.policy_definitions, magnitude_panel_daily.build_daily_blocks, eligibility_policy.eligibility_policy, eligibility_policy.plan_eligibility, experiment_population.canonical_events, experiment_trades.classify_events, snapshot_inputs.snapshot_reference, snapshot_inputs.admit_snapshot, progress.Progress, progress.stage, progress.decision, iv_crush_inputs.load_iv_crush_targets, iv_crush_inputs.preflight_iv_crush_targets, event_history.plan_event_targets, event_history.plan_with_reader, event_history.POPULATION_CONTEXT_COLUMNS, replay.clean_chain_rows, _pricing.MIN_MEANINGFUL_COST, _pricing.ChainSnapshot, _pricing.FillModel, _pricing.StructureError, _pricing.price_structure, signal_screen.run, fill_quality.run, polygon_fills.run, replay.replay, replay.replay_one, replay.ReplayResult, replay.ALPHA_GRID, _replay_run.run, _replay_run.events_frame, _plan.plan_events, _chains.ChainIndex, _chains.load_chain_index, _chains.filter_plan_by_availability, _chains.read_chain_keys, _chains.read_chain_keys_for, _chains.read_chains_for_years, _trades_table.to_trades_table, _build_run.run, reconcile_trades.run, _trades_publish.publish, build_trades.coverage, _pricing.STRUCTURES, experiment_trades.load_trades, _pricing.trading_calendar_from_snapshot, experiment_trades.PROVENANCE, _pricing.execution_variant_label, experiment_exits.PositionLeg, experiment_exits.EnteredPosition, experiment_exits.ExitDecision, experiment_exits.walk_fixed_day, experiment_exits.walk_exit, experiment_exits.exit_report_frame, experiment_population.load_population, prediction_inputs.load_prediction_targets, event_history.load_event_history, magnitude_inputs.magnitude_recipe, magnitude_inputs.forecast_cutoff, magnitude_inputs.load_magnitude_targets, magnitude_inputs.preflight_magnitude_targets, magnitude_inputs.qualify_magnitude_training_targets, magnitude_features.load_magnitude_history, magnitude_folds.plan_magnitude_folds, magnitude_folds.magnitude_window, magnitude_folds.annual_fold, magnitude_history_store.build_context, magnitude_history_store.materialize, magnitude_history_store.read_stored, magnitude_history_store.preflight_stored, magnitude_history_store.read_stored_for_events, magnitude_history_inputs.admit_stored, magnitude_panel_table.CONTRACT, magnitude_panel_table.feature_dictionary, magnitude_panel_table.recipe, magnitude_panel_table.validate_rows, magnitude_panel_plan.build_context, magnitude_panel_plan.preflight_context, magnitude_panel_plan.validate_context, magnitude_panel_table.canonical_day, magnitude_history_store.validate_context_identity, magnitude_history_store.validate_event_forecasts, magnitude_history_store.preflight_source, magnitude_history_store.validate_features, magnitude_features.calendar_input, magnitude_inputs.target_object_identity, horizon_fit_inputs.prepare_horizon_inputs, horizon_fitted.fitted_forecasts, horizon_fitted.HorizonFitFailure, magnitude_folds.plan_regression_folds -->

`regression_evidence.normalize_forecasts` checks exact event/horizon coverage
for an explicit prediction column and signed/nonnegative domain; `error_metrics`
returns pooled equal-row-weight count/RMSE/MAE from finite residuals. Both are
source-free numeric helpers, not target identity, fold or holdout admission.
Magnitude search/report wrappers retain their existing nonnegative contracts.

## Consumers

`experiments.iv_crush_prepare` consumes both native row APIs and the event-target publisher.

`experiments/magnitude_search_request.py` consumes `magnitude_panel_table.recipe`
and `eligibility_policy.eligibility_policy` for source-free dictionary/policy declarations, not data admission.
`experiments/magnitude_search_request_versions.py` consumes v1/v2/v3 panel `recipe` APIs for dictionary adaptation and request-version dispatch; it grants no source admission.
`experiments/distribution_candidate_recipe.py` consumes the qualified v3 feature recipe and distribution format; `distribution_class_support.py` consumes histogram validation for explicit class mapping. `distribution_trials.py` consumes bounded progress/stage logging during admission and projection; these helpers grant no data or fitting authority.

Which packages import this one, and for what. Checked against the import graph:
a claimed consumer that does not import, or an omitted one that does, is a
failure rather than a stale sentence.

_Nothing inside `engine/` imports this package. Its consumers are the CLI
leaves `tools/v2_signal_screen.py`, `tools/v2_fill_quality.py`,
`tools/v2_polygon_fills.py`, `tools/v2_replay.py`,
`tools/v2_build_trades.py` and `tools/v2_reconcile_trades.py`, plus
`experiments/common_v2.py` (for `experiment_trades.load_trades`,
`_pricing.trading_calendar_from_snapshot` and `_chains.load_chain_index`),
the pinned
`experiments/EXP-147_str_thru_gate_promotion_confirmatory_val/run.py` runner
(for `experiment_trades.PROVENANCE`), `experiments/v2_candidate_grid.py`
(`price_candidate_grid`, issue #266 slice 2 — for
`_chains.filter_plan_by_availability`/`read_chain_keys_for`,
`_plan.plan_events`, `_pricing.STRUCTURES`/`execution_variant_label`/
`trading_calendar_from_snapshot`, and `replay.ALPHA_GRID`/`replay_one`),
and `experiments/EXP-186_.../run.py` (for `_replay_run.events_frame`)
— none of which the layering hook parses (none is an `engine.*` module)._

`experiments/v2_gate_variant.py` also consumes the shared population loader,
registered structures and replay interfaces for read-only smoke pricing.
`experiments/runup_exit_report.py` consumes population admission, calendar/plan,
chain index, the entry-selection surface above and `experiment_exits.walk_exit`
for same-position, read-only fixed-day versus target/stop comparisons.
`experiments/gate_report.py` consumes `experiment_population.load_population`,
`_pricing.STRUCTURES`/`trading_calendar_from_snapshot`, `_plan.plan_events` and
`event_history.load_event_history` for independently priced fold-local reports.
`experiments/native_registration.py` consumes `_pricing.STRUCTURES`, population admission,
the pure magnitude window and metadata-only planner to bind research registrations.
`experiments/magnitude_search_plan.py` consumes `magnitude_folds.annual_fold`
to reconstruct outer metadata and plan nested folds with the same public rule.
`experiments/magnitude_search_evidence.py` consumes stored-history reads, target
preflight, training qualification and `magnitude_inputs.magnitude_recipe` for shared inner/outer evidence admission.
`experiments/magnitude_search_trials.py` and `experiments/magnitude_search_outer.py`
directly consume `load_magnitude_targets` for recorded inner and selected outer trials respectively.
`experiments/magnitude_inputs.py` consumes `magnitude_inputs.preflight_magnitude_targets`.
`experiments/magnitude_report.py` consumes that preflight, `load_magnitude_targets`
and `qualify_magnitude_training_targets` for the admitted mean baseline.
`experiments/magnitude_history_store.py` consumes `build_context`, `materialize`
and `read_stored` from `magnitude_history_store` for explicit stored-feature build/read.
`experiments.magnitude_search_evidence` uses `read_stored_for_events` to validate/hash
every full stored row while retaining only requested events and numeric columns.
Its `inputs_id` remains the full-envelope identity; projected rows are not that envelope.
Metadata precedes all numeric decoding, including unselected rows; no outcomes open.
`experiments/magnitude_history_inputs.py` consumes `magnitude_history_inputs.admit_stored`;
`experiments/magnitude_history_report.py` consumes `magnitude_history_store.read_stored`
and the same magnitude target APIs as the baseline.

The reporter reads the stored table named by `magnitude_history_table.TABLE`
(`research_magnitude_history`): the `METADATA` columns (`event_id`, `context_id`,
`horizon_sessions`, `event_date`, `cutoff_id`, `source_session_ceiling`,
`history_status`, `history_json`) plus the `FEATURES` columns
(`magnitude_features.HISTORY_FEATURE_COLUMNS`), under `magnitude_history_table.CONTRACT`.

<!-- consumers: none -->

## Usage

Every CLI takes a catalog, its object store root, and either a scope (whose
pinned head is resolved once) or an explicit `--snapshot-id` (to reproduce a
run after the head moves):

```bash
python3 tools/v2_signal_screen.py --catalog data/catalog.sqlite \
    --store-root data/artifacts [--snapshot-id snap_...]
python3 tools/v2_fill_quality.py --catalog data/catalog.sqlite \
    --store-root data/artifacts [--since 2026-07-30] [--csv out.csv]
python3 tools/v2_polygon_fills.py --catalog data/catalog.sqlite \
    --store-root data/artifacts [--min-date 2024-08-19]
```

Each writes a report (parquet/markdown for the first two, JSON for the plan)
that names the snapshot id it read.

The replay half is driven as a library:

```python
from engine.v2.research import _replay_run
result = _replay_run.run(repository, strategies=["STR-THRU"], events=events)
```

Reconciliation is its own library entrypoint and CLI:

```python
from engine.v2.research import reconcile_trades
report = reconcile_trades.run(repository, scope="shadow")
```

## Testing

`tests/test_v2_research_tools.py` (tier 0): a synthetic committed snapshot with
real Parquet fragments, no panel load, no network. Four cases per tool: the
moved function is byte-equal to its legacy counterpart on the same synthetic
frame; the tool's output carries the pinned snapshot id; an explicit
`--snapshot-id` still reads the first snapshot after the head has moved; and a
manifest with deleted fragment membership is refused (`MANIFEST_CORRUPT`)
rather than read.

`tests/test_v2_research_replay.py` and `tests/test_v2_research_build_trades.py`
cover the move-parity requirement against legacy `engine.replay` on synthetic
chains, snapshot pinning, and the refusal paths.
`tests/test_v2_research_reconcile_trades.py` covers the canonical-event prune:
non-canonical rows tombstoned, everything else byte-identical, the pinned
snapshot recorded, and an explicit `--snapshot-id` honoured.

## Opt-in grouped eligibility

`eligibility_policy.eligibility_policy(version=..., embargo_days=...)` declares
`grouped_ticker_embargo.v1`; whole calendar-day width is explicit, not globally 30.
`eligibility_policy.plan_eligibility` rederives random/rolling membership from
pinned canonical event metadata and returns a content-addressed development
exclusion plan with retained IDs, reasons, coverage, filtered evaluation IDs,
ambiguous-random event IDs and evaluation exclusion reasons. It merges
same-ticker random windows and uses inclusive interval overlap; one overlapping
horizon excludes the whole event, not individual price bars.

The closed `event_group_bounds.v1` input binds the full `source_snapshot`,
`holdout_as_of_month`, `policy_id`, sorted positive `horizons`, `groups`, and
`metadata_id` (native content hash of the other fields). Each group declares
`event_id`, `ticker`, `event_date`, `label_start`, `label_end`, and one forecast
per horizon with `horizon_sessions`, `decision_date`, `source_session_ceiling`.
Dates are canonical ISO calendar days. Producers must establish these bounds;
missing bounds refuse. A caller's digest is not evidence of their truth.

The eligible-history builder below consumes this metadata policy. Evaluation
IDs outside an ambiguous-random ticker are preserved, not authorized for reading;
IDs on that ticker are listed in `evaluation_exclusions`. Annual development
targets still need retained eligibility. The policy identity and dependent
request/dictionary hashes are unchanged; the embargo appears only in plan content,
and only when an ambiguous random event exists, so earlier plans keep their IDs.
Integration sequencing belongs to the [stage-2 design](../../../experiments/SUPER_SIZE/STAGE2_DESIGN.md).

## Confirmed metadata index (synthetic only)

`magnitude_confirmed_metadata.build_index(repository, snapshot, *, tickers, limits)`
opens a factory-owned bounded native earnings view, preflights all partitions,
exhausts one capped scan per year and composes the M1 sequence reducer (`reduce_with_counts`): duplicate IDs or ticker/days refuse, while date-conflict and duplicated ticker/cluster rows are unconfirmed and counted.
The detached `magnitude_confirmed_metadata.v1` proof retains complete native
source membership, selected/confirmed counts and a canonical `index_id`.
All 13 limits are mandatory; native failures remain intact and no partial proof
escapes. No coverage/PIT/numerical admission follows from dataset membership.
See [the frozen contract](../../../experiments/SUPER_SIZE/CONFIRMED_METADATA_DESIGN.md).

## Policy-qualified eligible history

<!-- public-interface: eligible_history.admit_source, eligible_history.iter_read, eligible_history.plan_targets, eligible_history_table.CONTRACT, eligible_history_table.FEATURES, eligible_history_table.admit_fields, eligible_history_table.recipe -->

These are intentionally manual, research-only Python APIs, not production CLI,
nightly or training entrypoints. A separately authorized research caller supplies
the pinned source and explicit policy; this PR neither activates a data run nor
wires production scoring. Synthetic tests and the library-only SIZE target derivation (`experiments.size_prep_targets`, itself uncalled until #1614 slice 4) are the only current invocations.
The sole public build/read boundary is `experiments.eligible_history` (plus the read-only, non-materializing `plan_targets` planner used by SIZE target derivation); it rejects
caller producer identities before source access and derives the installed identity.
An immutable import-time source/environment baseline refuses later disk drift with
`INPUT_CHANGED`; restart from a fixed code tree. This is not bytecode attestation.
Its backend `_materialize`/`_read` helpers are trusted internal orchestration, not
independent admitted APIs or a Python privacy/security boundary. The adapter consumes
backend source admission and these helpers;
`experiments.magnitude_eligible_history` consumes its qualified H11 field list.
Use pure `experiments.eligible_history.validate_request` for closed request shape only (not data admission).
Public malformed requests/SIZE artifact envelopes refuse `CONTRACT_MISMATCH` before
fingerprints or source access; a valid non-30-day SIZE policy still refuses `HOLDOUT_ACCESS_DENIED`.
Use `experiments.eligible_history.materialize/read` for complete list results, or
`experiments.eligible_history.iter_read` for incremental rows with the same
admission and coverage checks. The library-only `experiments.size_panel` wrapper
uses `iter_read`; no production or nightly entrypoint invokes it without separate
authorization. Supply an
explicit source snapshot, holdout month, grouped policy, target IDs, a nonempty
ascending subset of horizons `[1, 3, 5]`, history start and limits (`max_source_rows`, `max_fragments`,
`max_object_bytes`, `max_total_bytes`, `max_output_rows`). The separate
`research_eligible_history` table requires native
`computed_move_bounds`; missing/stale bindings refuse before outcomes.
Missing bounds for selected target/history tickers or events are `CONTRACT_MISMATCH`;
wrong outcome-object bindings/isolation remain `HOLDOUT_ACCESS_DENIED`. Unrelated
tickers need no sidecar; full canonical metadata still defines protected windows.
History selection uses each target ticker's own latest ceiling, retaining explicit
targets. Receipt lookup compares identity in SQL and returns only a bounded native
snapshot ID, never unused diagnostics; conflicting evidence is `IDENTITY_CONFLICT`.
Before object publication, output also obeys the native reader's summed uncompressed
Parquet row-group cap, alongside compressed-object/total caps; no peak-allocation guarantee.
`experiments.magnitude_eligible_history.build/comparison_inputs` supplies SIZE's
native folds and one shared clean-history input for challenger and comparator.
SIZE admits only its frozen 30-day grouped policy on both build and read.
This is a causal research refit, not production-byte parity. Annual evaluation
remains development and must pass embargo admission. No final-confirmation API.
Installed code/environment, source, policy/month, cutoffs and contributors bind
replay. Live-only aggregates, unknown fields and unapproved aliases/derivations
are not accepted experiment sources; v1 readers and production tables stay unchanged.
Each preparation revalidates clocks once per distinct date/session/horizon against
its fixed calendar, sharing a local memo across targets/contributors, never across reads.
The artifact contains native H11 means/EMAs only, not implied-quote history.
Historical publication and schedule knowledge remain unverified reconstruction.

## Complete qualified SIZE panel

<!-- public-interface: magnitude_panel_qualified_table.CONTRACT, magnitude_panel_qualified_table.FEATURES, magnitude_panel_qualified_table.METADATA, magnitude_panel_qualified_table.TABLE, magnitude_panel_qualified_table.admit_columns, magnitude_panel_qualified_table.feature_dictionary, magnitude_panel_qualified_table.recipe -->

`experiments.size_panel.materialize/read` is the installed-identity public boundary;
`magnitude_panel_qualified` and `_magnitude_panel_*` helpers are internal only.
Supply exact source/history/price snapshots, `as_of_month`, frozen 30-day `policy`,
`event_ids`, a nonempty ascending subset of `horizons` from `[1,3,5]`
(`horizons=[1]` for lead-1-only runs), `history_start`, `source_start`, eligible-history
`limits` and confirmed-index `metadata_limits`; no defaults imply unbounded reads.
Materialize returns a native feature_snapshot and context; read additionally takes
that snapshot and explicit columns, returning metadata plus projected features.
The v3 recipe preserves numeric order but has new eligible-history source hashes;
old v1/v2 helpers and partial daily artifacts are unchanged. H/HM/HMO are 11/28/42
numeric columns; only HMO uses the appended quote indicator. Rows include quote
status, cutoff/context IDs, per-block anchors/missingness and a canonical row hash.
The independent historical-quote population preserves full metadata ordinals even
when realized history is absent. Its price-date brackets are not computed-capture
attestations. Source-publication knowledge remains reconstructed/unverified.
This API publishes no labels, performs no fit and grants no final-holdout access.
Bounds cover explicit records/bytes/results, not total Python allocation or RSS.

Raw daily-market access explicitly permits selected-column decoding of yearly
fragments before date filtering; it never admits excluded event contributions.
Eligibility precedes decoding; one scan per required ticker/year preserves native
file/byte limits. Realized-outcome objects remain separately protected.
The SIZE wrapper compares a fresh source/environment fingerprint to its immutable
import-time baseline before admission; drift requires restart (`INPUT_CHANGED`).
SIZE obtains history through public `experiments.eligible_history.iter_read`, never
by treating the raw private history fingerprint as an admission capability. The
stream is exhausted for stored reads and receipt replay before the panel is returned.
Panel publication bounds compressed bytes and the native all-column uncompressed
row-group sum before storing each fragment, preserving same-budget bounded replay.
Lineage anchor indexes/caches are invocation-local and version-keyed; returned
anchors are detached. Receipt reuse projects only a syntax-admitted snapshot ID.
Malformed/missing/unknown request fields refuse before fingerprinting or source
access. Empty required source inventories refuse `CONTRACT_MISMATCH`, not an
unbounded query or builtin error; snapshot shape failures remain `SNAPSHOT_UNRESOLVED`.
Daily quote lineage uses native ticker/year/date fragment membership planning;
yearly files pruned for other tickers are not reported as consumed quote sources.

The experiment-owned `distribution_market` adapter also consumes the native calendar, chain, plan and pricing helpers for pinned STR-THRU research quotes; it adds no shared eligibility or source-reader policy.
