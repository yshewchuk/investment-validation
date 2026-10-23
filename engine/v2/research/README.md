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
deliberately **not** moved: it reads only a CSV transactions log and needs no
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
bounded-cadence ticks, stage timing/RSS, optional native refusal codes (no message text) and aggregate decisions.

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

The replay half: `replay.replay`, `replay.replay_one`, `replay.ALPHA_GRID`,
`_replay_run.run`, `_replay_run.events_frame`, `_plan.plan_events`,
`_chains.ChainIndex`, `_chains.load_chain_index`,
`_chains.filter_plan_by_availability`, `_chains.read_chain_keys`,
`_chains.read_chain_keys_for`, `_trades_table.to_trades_table`, `_build_run.run`, `reconcile_trades.run`,
`_trades_publish.publish`, `build_trades.coverage`, `_pricing.STRUCTURES`,
`_pricing.trading_calendar_from_snapshot`, `_pricing.execution_variant_label`.

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

The manual no-ledger `experiments.v2_gate_variant` CLI consumes
`experiment_population.load_population`, `_pricing.STRUCTURES`, and
`replay.replay`/`replay.ReplayResult` for fresh per-arm fill pricing.
`experiments.gate_report` combines protected historical features, pinned
calendar plans and fresh per-arm pricing for exploratory fold reports.

<!-- public-interface: eligibility_policy.eligibility_policy, eligibility_policy.plan_eligibility, experiment_population.canonical_events, experiment_trades.classify_events, snapshot_inputs.snapshot_reference, snapshot_inputs.admit_snapshot, progress.Progress, progress.stage, progress.decision, iv_crush_inputs.load_iv_crush_targets, iv_crush_inputs.preflight_iv_crush_targets, event_history.plan_event_targets, event_history.plan_with_reader, event_history.POPULATION_CONTEXT_COLUMNS, replay.clean_chain_rows, _pricing.MIN_MEANINGFUL_COST, _pricing.ChainSnapshot, _pricing.FillModel, _pricing.StructureError, _pricing.price_structure, signal_screen.run, fill_quality.run, polygon_fills.run, replay.replay, replay.replay_one, replay.ReplayResult, replay.ALPHA_GRID, _replay_run.run, _replay_run.events_frame, _plan.plan_events, _chains.ChainIndex, _chains.load_chain_index, _chains.filter_plan_by_availability, _chains.read_chain_keys, _chains.read_chain_keys_for, _trades_table.to_trades_table, _build_run.run, reconcile_trades.run, _trades_publish.publish, build_trades.coverage, _pricing.STRUCTURES, experiment_trades.load_trades, _pricing.trading_calendar_from_snapshot, experiment_trades.PROVENANCE, _pricing.execution_variant_label, experiment_exits.PositionLeg, experiment_exits.EnteredPosition, experiment_exits.ExitDecision, experiment_exits.walk_fixed_day, experiment_exits.walk_exit, experiment_exits.exit_report_frame, experiment_population.load_population, prediction_inputs.load_prediction_targets, event_history.load_event_history, magnitude_inputs.magnitude_recipe, magnitude_inputs.forecast_cutoff, magnitude_inputs.load_magnitude_targets, magnitude_inputs.preflight_magnitude_targets, magnitude_inputs.qualify_magnitude_training_targets, magnitude_features.load_magnitude_history, magnitude_folds.plan_magnitude_folds, magnitude_folds.magnitude_window, magnitude_folds.annual_fold, magnitude_history_store.build_context, magnitude_history_store.materialize, magnitude_history_store.read_stored, magnitude_history_store.preflight_stored, magnitude_history_store.read_stored_for_events, magnitude_history_inputs.admit_stored -->

## Consumers

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
exclusion plan with retained IDs, reasons, coverage and separate evaluation IDs.
It merges same-ticker random windows and uses inclusive interval overlap;
one overlapping horizon excludes the whole event, not individual price bars.

The closed `event_group_bounds.v1` input binds the full `source_snapshot`,
`holdout_as_of_month`, `policy_id`, sorted positive `horizons`, `groups`, and
`metadata_id` (native content hash of the other fields). Each group declares
`event_id`, `ticker`, `event_date`, `label_start`, `label_end`, and one forecast
per horizon with `horizon_sessions`, `decision_date`, `source_session_ceiling`.
Dates are canonical ISO calendar days. Producers must establish these bounds;
missing bounds refuse. A caller's digest is not evidence of their truth.

This is a metadata foundation only: no current runner uses it. Evaluation IDs
are preserved, not authorized for reading. Existing v1 behavior is unchanged.
Integration sequencing belongs to the [stage-2 design](../../../experiments/SUPER_SIZE/STAGE2_DESIGN.md).
