# `engine/v2/research` — architecture

Layer 7.0 in both `checks/layer_map.py` and the root `/ARCHITECTURE.md`
layer table. Native snapshot-reading counterparts to `tools/signal_screen.py`,
`tools/fill_quality.py`, `engine/data/pulls/polygon_fills.py`'s read path,
`engine/replay.py` (over a committed snapshot) and `engine/build_trades.py`'s
v2 write path. The legacy modules themselves are untouched and keep
publishing exactly as before; this package adds parallel `tools/v2_*` CLIs
that read a pinned v2 snapshot instead.

`tools/log_diagnostics.py` is deliberately **not** moved here: it reads only
a local transactions-log CSV, never the Tier-2 store, so it needs no
snapshot-pinned counterpart.

## Purpose

Offline research analysis CLIs — real-fill quality, a cross-sectional signal
screen, the real-trade pull's universe plan, deterministic as-of replay, and
the trade table that replay publishes — reading exactly one resolved,
named snapshot per run instead of the legacy mutable Tier-2 store. It
produces measured tables and reports for a person to read; it never decides
a research conclusion, never fetches from a network provider, and never
mutates the legacy trades ledger. `experiment_trades.load_trades` extends
this same one-snapshot-per-read contract to the v2 experiment platform:
read the committed `trades` version
`tools/v2_build_trades.py` published, never the legacy mutable store.
Completed experiments need not remain runnable;
their historical wrappers are not a supported experiment execution boundary.

## Primary contracts and public interfaces

CLI leaves (`tools/`, one process per run, each takes `--catalog`,
`--store-root`, `--scope` and `--snapshot-id`):

- `tools/v2_fill_quality.py` — real Polygon trades vs ORATS quotes.
- `tools/v2_polygon_fills.py` — the real-trade pull's contract universe
  (plan only; fetches nothing — the legacy `build_plan`/`execute` network
  half stays in `engine/data/pulls/`).
- `tools/v2_signal_screen.py` — the long-put cross-sectional screen.
- `tools/v2_replay.py` — deterministic as-of replay of selected structures.
- `tools/v2_build_trades.py` — replay a strategy set and publish a new
  `trades` dataset version.
- `tools/v2_reconcile_trades.py` — tombstone the `trades` table's
  non-canonical simulated rows as a new dataset version.

These CLI files sit outside `checks/import_layers.py`'s hook (it only parses
`engine*` importers), so no `engine` package is recorded as importing this
one; the library entrypoints below are this package's real public interface:

`eligibility_policy.eligibility_policy`, `eligibility_policy.plan_eligibility`,
`experiment_population.canonical_events`, `experiment_trades.classify_events`,
`snapshot_inputs.snapshot_reference`, `snapshot_inputs.admit_snapshot`,
`signal_screen.run`, `fill_quality.run`, `polygon_fills.run`, `replay.replay`,
`replay.replay_one`, `replay.ALPHA_GRID`, `_replay_run.run`,
`_replay_run.events_frame`, `_plan.plan_events`, `_chains.ChainIndex`,
`_chains.load_chain_index`, `_chains.filter_plan_by_availability`,
`_chains.read_chain_keys`, `_chains.read_chain_keys_for`,
`_trades_table.to_trades_table`, `_build_run.run`,
`reconcile_trades.run`, `_trades_publish.publish`, `build_trades.coverage`,
`_pricing.STRUCTURES`, `_pricing.trading_calendar_from_snapshot`,
`_pricing.execution_variant_label`.

The RUNUP comparison's public entry-selection seams are
`replay.clean_chain_rows` (returns valid quote rows without input mutation),
`_pricing.MIN_MEANINGFUL_COST`, `ChainSnapshot`, `FillModel`, `StructureError`
and `price_structure` (existing pure native selection/spread arithmetic).

`experiment_trades.load_trades(repository, snapshot, strategy, *, as_of_month,
purpose="training", event_ids=None)` is a second
kind of entrypoint: a plain library call (no `tools/v2_*.py` CLI of its own),
for a v2 platform caller that already holds a
`Repository` and a resolved `SnapshotRef` and wants one strategy's committed
`trades` rows, session-joined, with legacy-compatible trade columns plus
holdout context. `_pricing.trading_calendar_from_snapshot`
is a third: carved out of `_pricing.py`'s otherwise-internal contents the
same way `_pricing.STRUCTURES` already is, for the same caller — a repricer
built over a pinned snapshot needs the identical trading calendar `replay()`
itself derives, not a second implementation and not the legacy CSV fallback
(see Invariants). `_pricing.execution_variant_label` is a fourth, carved out
for `experiments/v2_candidate_grid.py`'s `price_candidate_grid` (issue #266
slice 2), which labels each priced grid-position step with the same
execution-variant string `replay()` itself uses, rather than reimplementing
that labeling.

Experiment training, selection-fold and sweep reads exclude the union of the
two holdouts specified in [#373](https://github.com/yshewchuk/investment-validation/pull/373).
Shared membership definitions live in `foundation.experiment_holdouts`.
The explicit rolling as-of month and both membership versions accompany the
snapshot id in returned columns; exclusion labels are in `holdout_exclusions`
frame attributes. Conflicting event IDs at one ticker/date/session or cluster,
and other ambiguous canonical identity/date matches, are excluded and
labelled `ambiguous`. Released eligible rows are labelled `post-release selection`.
A requested `event_ids` population containing any excluded event raises the
non-retryable `DataError(HOLDOUT_ACCESS_DENIED)` during population validation,
before returning a frame to metric/report writers. Missing/invalid context,
an as-of month later than the current UTC month,
unknown purposes and an entirely excluded population receive the same refusal.
There is no date fallback, partial returned frame, report write or retry here.
Final holdout reads are unavailable. The loaders write nothing themselves; the
recorded native prediction runner appends one plain `refused` row (no metrics;
replay adds no second) when a `run` on an admitted registration hits
`HOLDOUT_ACCESS_DENIED`, unless a different immutable outcome for that variant is
already saved, success or failure: identity conflict, ledger unchanged.
`register`, a run with no admitted registration, and `--no-ledger` smoke write
none. `ledger_context` excludes `refused*` rows from specs tried.

`experiment_population.load_population` validates an exact committed
`SnapshotRef` and returns canonical event metadata under that same holdout
contract, without reading outcomes. Explicit IDs are checked against the full
canonical calendar, including conflicting siblings, before any caller can
consume an eligible population. It shares the trade loader's membership and
ambiguity logic; neither source dates nor ticker names substitute for event IDs.
Multiple IDs on one ticker/day are ambiguous even when their sessions differ:
the prediction outcome table cannot distinguish those event identities.
`experiments.v2_gate_variant` consumes this population in sweep mode before
fresh native replay, directly or through the smoke-only pricing/evaluation
workers. Their no-report/no-ledger contracts live in `experiments/ARCHITECTURE.md`.
`experiments.gate_report` separately composes population, observed calendar sessions,
protected `event_history` and fresh replay for each fold's training/evaluation rows;
its reconstructed date-ceiling and native publication contracts live in that same doc.
`prediction_inputs.load_prediction_targets` consumes this population first,
then scans each eligible ticker/event-date pair in pinned `computed_moves`. The history
and magnitude readers (`event_history`, `magnitude_inputs`, `magnitude_history_store`)
accept the `computed_moves.v4` or `computed_moves.v5` contract and refuse any other
`CONTRACT_MISMATCH`; on v5 a row with a `not_computed_reason` other than `no_price_window`
is `skipped` and carries no outcome, so it never counts as a scoreable event. The
writer's reason is the only reason research reports for a skipped row: the history and
training-label and opt-in evaluation omission lists (`omitted_history`, `omitted_labels`) carry the row's
`not_computed_reason` as the omission `reason` (v4 rows, and v5 rows without a reason, keep
`skipped`; the magnitude search's `validate_targets` accepts exactly `skipped` and the writer's
closed reason set, so a receipt with any other reason is `EXPERIMENT_IDENTITY_CONFLICT`), never a research-side re-derivation. Non-skipped rows the readers omit for
availability keep the reader's own `unavailable` / `not_available_before_fit` reasons. The
isolation guard of the history and magnitude readers (`_plans`'s per-candidate checks, shared with `plan_with_reader`) is unchanged: every
candidate still needs exactly one isolated fragment, and a candidate with no fragment
refuses `HOLDOUT_ACCESS_DENIED`. `prediction_inputs` does not use that guard: it refuses a
table whose contract is neither native contract with `CONTRACT_MISMATCH` before any target
scan, and a v5 skipped row with `EXPERIMENT_VARIANT_FAILED` naming the reason (the v5 contract
requires one; a reason-less skipped row is malformed and gets the generic target message).
It returns the signed move, binary `positive_move` target, and
`target_available_on` date. The date is required and later than the event;
fold consumers must admit training labels only when available before their
decision cutoff. This input prerequisite performs no feature construction,
model fitting, or report/ledger publication, and has no supervisor caller.

| Prediction/population input condition | Outcome |
|---|---|
| Snapshot cannot be resolved or its identity does not match (design #372 R2) | Non-retryable `SNAPSHOT_UNRESOLVED`; no source fallback. |
| Integrity or object failure during a table scan | Repository's typed refusal, such as `MANIFEST_CORRUPT` or `OBJECT_CORRUPT`; no partial returned frame. |
| Missing table or malformed table contract | Repository's typed refusal; no partial returned frame. |
| Invalid holdout context, unknown requested ID, excluded/ambiguous requested event, or empty explicit population (R5) | `HOLDOUT_ACCESS_DENIED` before target scans; no metrics or report. |
| Well-formed `computed_moves` contract that is neither native v4 nor v5 | `CONTRACT_MISMATCH` before any target scan; no partial returned frame. |
| Empty bulk canonical calendar | `POPULATION_COLLAPSED`; no empty success. |
| Missing, skipped, non-finite target, non-date outcome key or invalid target availability (R4) | `EXPERIMENT_VARIANT_FAILED`; no partial returned frame. |
| `omit_not_computed=True` and a row is `skipped` | Event omitted, object still bound in `consumed_inputs`, listed in `omitted_labels`; other events returned. |
| `omit_not_computed=True` and every planned label is `skipped` | Empty frame whose attrs list every event in `omitted_labels`; scoring then refuses `EXPERIMENT_VARIANT_FAILED` ("require scored rows"), which the inner search records as a refused terminal. |
| `omit_not_computed` is not a `bool` | `CONTRACT_MISMATCH` before any repository access. |
| Cache, retry, transaction, partial write, replay | No cache or automatic retry; read-only, no writes; complete scans precede return; identical pinned inputs return identical rows (R6). |

These readers preserve no economic variant identity and cannot authorize the
separate user-only final holdout read.

`event_history.load_event_history(repository, snapshot, *, event_id, as_of_month,
decision_session, history_start, purpose="selection")` constructs prior-event
features for one admitted canonical target. Both dates are explicit calendar
days with `history_start <= decision_session <= target event day`. Shared
`load_population` admits the target and excludes random, rolling and ambiguous
history identities before any outcome access. Candidates are the same ticker's
events in `[history_start, min(target day, decision_session))`.

Every candidate query is planned before any outcome scan, using the same
fragment planner as Repository. Preflight indexes fragment metadata once by
partition and canonical time intervals after shared full-key bound admission.
Malformed keys remain in a global fallback; uncertain times stay partition-local.
Selected records retain manifest order and planner refusal precedence, including
malformed keys outside the selected issuer/day. Each bound is reused for query construction.
Its entire surviving membership must be one
inspected singleton with the exact ticker/day key, partition and time bounds.
The resolved committed catalog/manifest identity is the metadata trust anchor;
this trusts native producer inspection, not adversarially forged metadata.
Caller rows, implicit latest, and another experiment's outputs are not inputs.
Mixed, missing or unprovable membership refuses `HOLDOUT_ACCESS_DENIED` before
opening outcome objects. This intentionally refuses old whole-ticker layouts;
it neither migrates them nor treats row predicates as physical isolation.

Skipped or null-availability historical labels are explicitly omitted. A valid
availability day must follow its event and be on or before the decision day;
later labels are omitted before numeric validation or feature arithmetic. Their
admitted objects are decoded; protected event objects have a no-read guarantee.
Malformed availability or invalid eligible values refuse
`EXPERIMENT_VARIANT_FAILED`; repository failures propagate. The date bound is reconstructed source availability, not a claim of
intraday publication/finality evidence. Pure `panel_math.history_features` owns
the arithmetic: empty history and undersized EMAs retain `None`, never zero fill.
Results carry features, `available`/`empty` status, ordered consumed event/object
identities and availability, omitted-history reasons, snapshot/dataset/month/
membership-version/cutoff pins and their deterministic history digest.
The `research_event_history.v2` identity also binds `read_inputs`: every decoded
candidate object, including skipped/unavailable labels omitted from arithmetic.
No cache, automatic retry, transaction, write or partial return occurs.
The reader performs no supervisor dispatch, provider/market-price access or
publication; `experiments.gate_report` consumes it for fold-local exploratory
reporting. Production panel reads and final-read authority remain unchanged.

`iv_crush_inputs.load_iv_crush_targets(repository, snapshot, *, as_of_month,
labels_before, purpose="selection", event_ids=None)` reads signed relative IV30
change from the exact current `iv_crush_targets` contract, including its definition
hash (`iv_crush_targets.v2`, semantic version `2.0.0`). `preflight_iv_crush_targets(..., as_of_month, event_ids)` requires explicit
selection participants (`CONTRACT_MISMATCH` when absent) and returns metadata-only planned-object evidence; it is
not a capability, and the loader repeats admission. These offline library APIs
do not publish sources, register experiments or participate in production entrypoints.

| IV-crush boundary | Outcome |
|---|---|
| Snapshot/population | Shared canonical snapshot, holdout-union and ambiguity admission precedes targets; unsupported/final purposes or malformed holdout context refuse `HOLDOUT_ACCESS_DENIED`; a missing/malformed label cutoff refuses `CONTRACT_MISMATCH`. Bulk reads omit events on/after the explicit calendar-day label cutoff; explicit requests containing one refuse `CONTRACT_MISMATCH`. An empty eligible set refuses `POPULATION_COLLAPSED`. |
| Public planner candidates | `plan_event_targets` requires ticker and naive midnight event_date columns. Only candidate shape/identity checks translate to `CONTRACT_MISMATCH` before repository access; every error raised while planning, including native `DataError` envelopes, propagates unchanged. |
| Physical admission | Every candidate passes public `event_history.plan_event_targets` (metadata-only singleton admission for already admitted populations) before any target opens. Missing/mixed objects or unprovable exact ticker/date bounds refuse `HOLDOUT_ACCESS_DENIED`; nonmatching contract/hash refuses `CONTRACT_MISMATCH`. Every declared target object is capped at 1 MiB before any target opens (`RESULT_LIMIT_EXCEEDED`); bounded Repository scans also cap compressed and total uncompressed bytes. This resource ceiling accommodates singleton labels without authorizing arbitrary payloads. Computed-moves defaults remain unchanged. |
| Stored label | Exactly one matching ticker/date/session, boolean false skip and null reason are required. Canonical anchors span 1–5 calendar days: AMC requires pre ≤ event < post; BMO requires pre < event ≤ post. Availability equals post plus one calendar day, follows the event and precedes the cutoff before numeric use. These checks do not prove source-row adjacency or exchange sessions. Skipped, late or malformed rows refuse the whole read with `EXPERIMENT_VARIANT_FAILED`; finite signed values, including zero and values below -100, remain unchanged. |
| Provenance | Row identity includes target object; attributes bind full dataset/contract ref, cutoff, month/membership pins, purpose, future exclusions and ordered fragment/object/content hashes plus caller source_id/input_hash. Source marker/hash syntax is checked, not independently authenticated or recomputed from source quotes. Publication/finality remain unverified; availability is a modeled floor. |
| Progress | Admission, planning and read stages report start/end/failure, elapsed time and peak RSS to stderr. Completed-item ticks report count/rate/ETA at most every 30 seconds. Aggregate decisions include bulk/explicit selection, admitted/holdout/ambiguity/cutoff counts and planned batch/byte limits; IV refusals include the native refusal code only, never message text, event identities or error details. Logging cannot change results, identities, errors or read authority. |
| Effects/failures | No source-quote reads, target rebuilding, retries, persistent cache, writes, partial returns or experiment authority. Earlier admitted objects may decode availability before a late-label refusal. Native repository integrity/resource refusals propagate unchanged. |

This consumer accepts only the exact definition, without grandfathering earlier hashes.
`event_history.POPULATION_CONTEXT_COLUMNS` names the shared snapshot/holdout context.
No publisher, source-receipt validation, forecast recipe or model is supplied.

`magnitude_inputs.forecast_cutoff(event_date, session, horizon, calendar)` is a
pure contract over native `CalendarSessions`: horizons are five, three or one
full intervening sessions, counted backward from the shared last-pre-print
close. Wednesday AMC at one uses Tuesday close; Wednesday BMO uses Monday.
Invalid dates/sessions/horizons/calendars, projected anchors, gaps in the
required session window or insufficient history refuse `CONTRACT_MISMATCH`.
Its identity binds event/session/horizon, calendar contents, pre-print close
and source-session ceiling. It does not assert an intraday issue time, source
publication/finality or that the earnings schedule was known at that cutoff.
An experiment caller must bind the calendar to its pinned native snapshot;
this pure helper does not load a calendar or select a forecast horizon.

`magnitude_inputs.load_magnitude_targets(repository, snapshot, *, as_of_month,
labels_before, purpose="selection", event_ids=None)` is the separate label
view. The fixed target recipe is absolute split-neutral, non-dividend-adjusted
earnings close-to-close return in percentage points; the native signed
`computed_moves` value is retained, never rescaled, clipped or adjusted again.
Its primary prediction estimand is conditional mean, selected by pooled RMSE;
MAE and quantile calibration are secondary; the magnitude reporter implements
MAE while quantile calibration remains unimplemented.
Shared population admission precedes label access. Bulk reads exclude events
on/after `labels_before`; explicit requests containing one refuse. All
remaining objects pass `event_history`'s shared singleton preflight before
any is opened; mixed inherited snapshots refuse. Decoded label availability
must be strictly before `labels_before`; missing, skipped, malformed or
non-finite labels refuse the whole call, with zero remaining a valid value.
Exception, opt-in only: `omit_not_computed=True` (used by the magnitude search's inner and outer
evaluation reads) treats a row the writer marked `skipped` (v5 `not_computed_reason`, e.g. `not_orats`)
exactly as `qualify_magnitude_training_targets` does: it is omitted, never parsed, and listed in the
frame attrs and receipt as `omitted_labels` (`[{"event_id", "reason"}]`, sorted by event ID), where `reason` is the row's
`not_computed_reason` (`skipped` for a row without one), the same rule as the training qualifier.
`omit_not_computed` must be a `bool` (`CONTRACT_MISMATCH` otherwise). The receipt carries `omitted_labels` only when non-empty (receipts with none are byte-identical to before).
Every other refusal above is unchanged; only `skipped=true` is omitted, never null/late availability.
Forecasts for omitted events are made but not scored; `event_count` counts scored events.
Earlier events with late availability can be decoded to check their recorded
date; this is reconstructed availability, not verified publication evidence.
Output rows carry target/object identity; frame attributes bind recipe,
label cutoff, snapshot/dataset/month/membership versions and consumed inputs.
Forecast cutoff and training/evaluation label cutoff are distinct contracts;
this seam neither fits a model nor joins labels into forecast features.
Repository failures propagate; no cache, retry, write or partial return.
`experiments/magnitude_report.py` consumes the strict target loader through the
magnitude CLI; there is no supervisor or final-read caller.

`magnitude_inputs.preflight_magnitude_targets(repository, snapshot, *,
as_of_month, event_ids)` admits an explicit selection population and proves
every candidate object's singleton isolation without opening outcomes. Its
detached digest binds the exact population/context and planned object identities.
The magnitude runner preflights the union of actual train/evaluation participants
across all folds before its first label read; unused window events need not join.
The preflight is evidence, not a capability: target loaders repeat their own
admission and physical checks and accept no caller-supplied scan plan.

`qualify_magnitude_training_targets(repository, snapshot, *, as_of_month,
labels_before, event_ids)` admits explicit candidates whose event dates precede
the fit boundary, with whole-population physical preflight before outcome access.
Skipped, unavailable and cutoff-late labels are distinct omissions before numeric
parsing; eligible labels require valid identity/availability and finite signed values.
Zero and negative values retain their absolute percentage-point magnitude, without
clipping or zero fill. The training identity binds every candidate and decoded
object (including omissions), qualified IDs, omission reasons and the cutoff.
Empty qualification refuses `POPULATION_COLLAPSED`; malformed eligible labels
refuse `EXPERIMENT_VARIANT_FAILED`. Protected/ambiguous identities and mixed objects
refuse before outcome access. Late admitted objects may be decoded for qualification;
protected objects are never opened. Repository refusals propagate, with no cache,
retry, write or partial result. Existing strict target identities remain unchanged.
These library seams neither publish evidence nor compute historical feature aggregates.

`magnitude_features.load_magnitude_history(repository, snapshot, *, event_id,
as_of_month, horizon, history_start, purpose="selection")` binds one canonical
event's prior-history features to its full-session forecast ceiling. Exact
snapshot and target holdout/alias admission occur before calendar or outcomes.
The calendar is the sorted union of canonical `daily_market.date` values from
that snapshot, with no projected days or external calendar override. Only the
date column is requested; ordinary Repository object verification still applies.
Missing/empty calendars refuse through the shared reader/`CALENDAR_UNAVAILABLE`;
malformed dates, invalid horizons or incomplete anchor/session windows refuse
`CONTRACT_MISMATCH` before historical outcome access.

The shared `forecast_cutoff` computes the ceiling; `load_event_history` uses
that exact day, excluding protected/ambiguous/current/future outcomes before
its physical singleton preflight. This adapter never reads the target label.
Prior-label availability on the source-session day is admitted by the existing
history contract; later dates are omitted, not used as zero-filled features.
Outputs bind canonical event/session, snapshot, calendar version, calendar
dataset/object identities, horizon/cutoff, target and history-feature recipes,
all history reads/omissions and membership versions into one deterministic
input digest. Empty history retains explicit null features; non-finite derived
features refuse `EXPERIMENT_VARIANT_FAILED` rather than reaching a model.

Provenance explicitly says reconstructed date bounds, with event-schedule
as-known and source-publication evidence unverified. Calendar reconstruction
and source hashes do not establish that evidence; strict PIT experiment
admission remains a later prerequisite. The stored build uses `plan_magnitude_folds`,
`event_history._assemble_history` and `_feature_recipe`, not this adapter.
The adapter has no fitting, target-join, CLI, provider, report, ledger or final-read
caller, and no retry, write or partial result; its only cache is the
observed-session read (see its failure row); Repository failures propagate.

`magnitude_folds.plan_magnitude_folds(repository, snapshot, *, as_of_month,
evaluation_years, training_start, event_ids=None)` plans metadata-only expanding
annual evaluation folds from one admitted snapshot and its pinned calendar.
`magnitude_folds.magnitude_window(evaluation_years, training_start)` exposes the
same pure request validation to native preregistration and returns a detached
canonical window; invalid requests refuse `CONTRACT_MISMATCH` without reads.

`magnitude_folds.annual_fold(year, events)` is the public annual rule shared by
the native planner and `experiments.magnitude_search_plan` for admitted metadata.

| Shared annual-rule condition | Outcome |
|---|---|
| Admitted event metadata has evaluation events and strictly earlier candidates | Return detached year, conservative fit cutoff and training/evaluation IDs in input order; no I/O or label qualification. |
| No evaluation events or earlier candidates | Preserve the native `POPULATION_COLLAPSED` `DataError`; callers own boundary-specific translation. |

- Exact snapshot and random/rolling/ambiguous-union admission precede calendar
  reads. Bulk membership includes `training_start` and excludes January 1
  after the last evaluation year.
- Each event's 5/3/1 horizons share its event-year fold. Training candidates
  precede `min(year_start, all evaluation source-session ceilings)`; usable
  labels separately require availability strictly before that boundary.
- Earlier evaluation events may train later folds, including the same issuer.
  Annual membership need not cover a full year; no inner HPO split is defined.
- The deterministic digest binds request, target/fold rules, ordered canonical
  event/cutoff memberships, holdout versions, snapshot and metadata-source objects.
  With all other inputs fixed, equivalent admitted explicit/bulk memberships
  yield the same `plan_id`; selection mode is not part of identity.
- No computed-move, label or history-feature access, fitting or registration/
  final-read authority. Native non-session announcement anchoring is preserved;
  schedule/publication evidence remains unverified reconstruction, not PIT data.

| Magnitude fold condition | Outcome |
|---|---|
| Years not explicit ordered unique four-digit strings, or training start not canonical and before the first year | `CONTRACT_MISMATCH` before repository access. |
| Explicit event outside the declared window | `CONTRACT_MISMATCH` before calendar access. |
| Empty window or any year lacking evaluation events/training candidates | `POPULATION_COLLAPSED`; no empty success. |
| Missing calendar table, snapshot/admission or repository failure | Shared typed error propagates; no source fallback. |
| Empty calendar / malformed dates or incomplete session window | `CALENDAR_UNAVAILABLE` / `CONTRACT_MISMATCH`; no outcome access. |
| Catalog transactions | Short read-only SQLite transactions; no planner-owned write transaction or transaction spanning the full call. |
| Cache, retry, partial write, replay | The only cache is the observed-session read below; no retry, write or partial return; identical pinned inputs give identical plans. |
| Observed-session read | The `daily_market` observed days are read once per pinned `Repository` (`pins_snapshots`) and snapshot id; the cache lives and dies with that repository, a second repository or snapshot reads again, and an unpinned repository reads on every call. A failed read caches nothing. Output and `calendar_input` identities are unchanged. |

`experiment_exits.walk_exit(repository, snapshot, positions, economic_params=...)`
accepts entered `EnteredPosition`/`PositionLeg` contracts and resolved experiment
economics with `price_source="option_chains"` and explicit numeric `fill` in
the existing alpha ladder's range. It supports `exit={kind: "fixed_day",
trading_days: N}` and `exit={kind: "target_stop", trading_days: N,
target_pnl: T, stop_pnl: S}`; N is a positive integer, T is finite and positive,
and S is finite and negative. Declaration admission is shared with ops through
`foundation.experiment_exits.valid_exit_recipe`. Target/stop recipes may declare `threshold_units`:
`quote_price_times_quantity` (also the omitted-field behavior) or
`entry_debit_fraction`. Fractions multiply each position's positive net opening
debit, priced once at the declared alpha, before walking its daily closing marks.
This basis includes spread but excludes fees, commissions and contract multipliers;
it is not gross premium or a portfolio return. Zero/credit entry or converted
thresholds that overflow or underflow to zero refuse the whole call. Fractions
are signed ratios, never percentage-point inputs; no clipping or normalization.
The immutable resolved economics and canonical
plan identity retain every recipe field, so changing a threshold changes the
consumed plan identity. The walker aggregates net position P&L from each day's
held-leg quotes in the pinned `option_chains` snapshot and uses the pinned
`daily_market` calendar, `_pricing.FillModel`, and existing alpha ladder; it
adds no projected calendar, replacement contract or mark source. A target or
stop exits on its first observed crossing. If one day's alpha 0/1 aggregate
position mark range spans both thresholds, the stop is selected,
`ambiguous_exit` is true because the daily marks cannot order the two exits,
and its exit value and P&L are priced at the worst existing alpha (0.0) so the
ambiguous result cannot be better than the stop. No crossing by N sessions
uses the 7a fixed-day fallback; the existing expiry bound and typed refusal
still apply. Results are
`mark_based` net held-position P&L with exit reason/date, visited dates, source,
the declared recipe alpha, the alpha actually applied at exit
(`exit_fill_alpha`, equal to the recipe alpha except for an ambiguous stop) and
snapshot identity. `exit_report_frame` projects those
decisions into one row per trade with separate `mark_based_pnl`, `exit_reason`,
`exit_day`, ambiguity and provenance columns for report callers; it does not
write files or blend mark-based P&L with fill-validated results.

| Exit-walker failure condition | Outcome (design #372 R4) |
|---|---|
| Malformed position or leg record | Non-retryable `EXPERIMENT_VARIANT_FAILED`; whole call fails, no excluded trade or partial tuple. |
| Missing required leg mark or mark past held-leg expiry | Non-retryable `EXPERIMENT_VARIANT_FAILED`; whole call fails, no excluded trade or partial tuple. Failure details identify the trade and failing session. |
| Unusable quote reaching pricing | Non-retryable `EXPERIMENT_VARIANT_FAILED`; whole call fails, no excluded trade or partial tuple. |
| Insufficient calendar coverage | Non-retryable `EXPERIMENT_VARIANT_FAILED`; whole call fails, no excluded trade or partial tuple. |
| Invalid fractional debit basis or converted threshold | Non-retryable `EXPERIMENT_VARIANT_FAILED`; whole call fails, with entry trade/session context and no partial tuple. |

Other repository refusals propagate.
No cache, retry, transaction or write: identical inputs return identical results.
Other experiment report/ledger publication remains outside this library seam.

Internal (not interface, despite the non-underscore package norm elsewhere):
`_scan.py` and `_snapshot.py` (see Dependencies — two independent
snapshot-read helpers), `_pricing.py` (except the names carved out above),
`_trades_revisions.py`, `_final_population.py`, `_final_evaluator.py`.

### Proposed authorized final read (#490)

Request preparation, the private ops authorization/spend primitive, the private metadata-only final population reader and the private frozen evaluator are implemented; no production authorization entrypoint, final-read caller or result retrieval exists. The remaining lifecycle belongs to later #490 slices and preserves #445's union exclusion; #555 refusal/audit integration is dropped and handled separately in #489.
The [user decision](https://github.com/yshewchuk/investment-validation/issues/490#issuecomment-6072905454)
permits agents/supervisors to prepare requests, never to authorize them.

**Request.** Canonical versioned bytes use the existing `experiment_id` as the decision identity, without a question record or alias detection, per the [user decisions](https://github.com/yshewchuk/investment-validation/issues/490#issuecomment-6101284544).
They bind the selected registered `variant_id`/registration reference,
resolved plan/source/evaluation-recipe and applicable frozen-model hashes, exact committed
`snapshot_id`, `holdout_as_of_month`, `random_membership_version` and
`rolling_membership_version`. No latest-head resolution, caller population override,
alternate arm, fitting, threshold search or evaluator callback is accepted.
Membership stays in `foundation.experiment_holdouts`: random and rolling sets under
its versioned as-of-month contract, with ambiguous identities refused, never admitted.
Both sets are bound together; changing month, version, snapshot or winner cannot reset a decision's spend. The request digest identifies bytes, not a new decision.
`native_registration.prepare_final_holdout_request` verifies existing catalog/artifact registration metadata only; it never resolves a head, scans a population or reads results.
Native registration's optional `final_read` declaration seals recipe, completeness-bar and frozen-model artifact references before authorization; preparation derives them, never accepts replacements. Registrations without that declaration refuse preparation.
Request validity proves neither winner selection nor read eligibility. The later metadata-only preflight must verify the preregistered completeness artifact and refuse an unmet bar before any holdout outcome read; declared references alone do not prove completeness.

**Population reader.** Private `_final_population.load_final_population(repository, request)` is the only final-read reader that returns separate held-out event-ID sets. It accepts a validated `FinalHoldoutRequest` (never a caller population) and returns a frozen `FinalPopulation` with sorted random and rolling IDs, overlap and ambiguity counts, and bound snapshot/month/version context. It reads only canonical `earnings_events` metadata with the loaders' shared membership/ambiguity rules, excludes ambiguous events from both sets, performs no writes, and does not inspect ops state or hold a grant. It is not exported or reachable from loaders, training, selection or sweep.

| Population reader condition | Outcome |
|---|---|
| Invalid request object, unsupported membership version, or as-of month later than the current UTC month | Non-retryable `HOLDOUT_ACCESS_DENIED` before any scan. |
| Snapshot unresolved, or resolved identity differs from `request.snapshot_id` | `SNAPSHOT_UNRESOLVED`; no fallback. |
| `earnings_events` absent from the snapshot | `CONTRACT_MISMATCH`; no partial population. Present with no fragments, the scan yields no events and the empty-population row applies. |
| `resolve` raising a non-integrity `DataError` (such as not-found), `ValueError`, `TypeError` or `KeyError` | `SNAPSHOT_UNRESOLVED`. An integrity-category `DataError` (such as `MANIFEST_CORRUPT`) and any other exception propagate unchanged. Integrity failures during the scan propagate as the repository's typed refusal. |
| Both sets empty after ambiguity exclusion | `POPULATION_COLLAPSED`; never an empty success. Preregistered completeness remains the later preflight's refusal. |
| Retry, cache, write | None; read-only, identical pinned inputs return identical populations. |

**Evaluator.** Private `_final_evaluator.evaluate_final_holdout(repository, request, population)` takes only a validated `FinalHoldoutRequest` and the `FinalPopulation` that `load_final_population` returns; it has no other parameter, fit, alternate arm, threshold search or callback. It scores the committed `trades` rows of the pinned snapshot with this package's provenance whose `variant` equals `request.variant_id` (the frozen winner); the table is read whole, then filtered. It returns frozen `FinalMetrics` with the random and rolling results separately, each with event count, trade count, events without a trade, total and mean `ret` and win rate (`None` when a set has no trade), plus overlap and ambiguity counts; the sets are never averaged or merged. It does no writes, caching or retries, never imports ops, and checks no fence or revocation: the later leaf does that before invoking it.

| Evaluator condition | Outcome |
|---|---|
| `population` is not a `FinalPopulation` (event-ID list, training/selection/sweep frame), or differs from the one derived from `request` | `HOLDOUT_ACCESS_DENIED` before any trades read; an equal hand-built population is indistinguishable and accepted. |
| Invalid request, or snapshot mismatch | The population reader's refusal (`HOLDOUT_ACCESS_DENIED`, `SNAPSHOT_UNRESOLVED`), before any trades read. |
| Trades table has no fragments, or no winner trade falls in either set | `POPULATION_COLLAPSED`. |
| Retained winner trade's event date or ticker differs from its pinned canonical event (for example after a calendar correction that kept the event ID) | `CONTRACT_MISMATCH` listing the offending event IDs; never skipped or scored. |
| Non-finite `ret` | `CONTRACT_MISMATCH`; infinities are already refused at publication. |

**Authority/ownership.** A proposed `tools/v2_final_holdout.py` leaf composes ops'
private authorization/state owner with a private research final evaluator. It is
the sole supported evaluator caller; ops/research remain layer-7 peers with no imports between them. Foundation supplies membership and pure request validation, never grants or state.
Training's lower layer cannot import either owner. Selection/sweep/loaders gain no
grant argument, final purpose, environment bypass or returned holdout frame.
Import/call-site regression guards also cover unlayered tools/experiments, which
the layer checker alone does not cover. Only the dedicated leaf may compose these owners; registered training/selection/sweep entrypoints cannot reach that leaf.
The leaf checks live ops state/fence before its sole evaluator invocation and before publication; research checks the immutable bound request, never imports ops state.
No reusable grant escapes that invocation. This is procedural/audited, not a security boundary against arbitrary code, a forged terminal or catalog edits.

**Durability.** Ops owns the append-only authorization/spend ledger in its catalog;
one transaction records who, UTC when, explicit confirmation, request digest and
the permanent decision spend. An interactive terminal and fresh explicit user
confirmation of the displayed binding are required; flags/piped input/agents cannot assent.
`ops.final_holdout.authorize_final_holdout` is currently test-consumed only: it verifies the exact catalog-owned registration, displays canonical request bytes and requires a terminal-entered phrase containing the digest and recorded process OS identity. This is procedural confirmation, not authentication of the human at the terminal. Its persisted acknowledgement is not a read grant or completeness verdict; the later fenced reader must still verify winner/revocation and the preregistered completeness artifact before scans.
Unique `experiment_id` plus compare-and-swap state serialize authorization and reads.
States are `authorized_unread`, `read_started`, `result_staged`, `complete`, or terminal `revoked`/`indeterminate`; every transition appends evidence, never refunds. Authorization has no expiry; revocation or a changed winner ends eligibility with `HOLDOUT_ACCESS_DENIED`, retaining the spend. Interrupted reads stay spent with no continuation; no renewal resets spend.

| Proposed condition | Required outcome before any returned metric/report |
|---|---|
| Missing user record, noninteractive authorization, tuning/training/selection/sweep, new read after `read_started`, or revoked/changed winner | Non-retryable `HOLDOUT_ACCESS_DENIED`, before holdout outcome scans or metrics; no partial result. Recheck winner/revocation at the fenced start and publication. Revocation after start but before completion burns the read and withholds its result. |
| Duplicate authorization | Identical request returns the original acknowledgement only after the same interactive checks; never another grant/spend. Changed bytes under the same decision refuse `IDEMPOTENCY_CONFLICT`; changed registration refuses `EXPERIMENT_IDENTITY_CONFLICT`. |
| Authorization recorded, no read started | A first user-invoked read may atomically commit `read_started` with one owner/fence before any holdout outcome scan. This never continues an interrupted read; missing/corrupt state never means unspent. |
| Crash after `read_started`, before durable `result_staged` | Once ownership is proven gone, mark `indeterminate`; never reread, even if the crash might precede the first scan. No result is released and the spend remains. |
| Crash after durable result, before completion receipt | Only a catalog-bound `result_staged` reference to fully synced immutable result bytes permits artifact-only reconciliation. Verify hashes/pins and the current fence/winner, then commit the receipt reference; never evaluate again. Unbound files cannot prove completion. |
| Receipt/diagnostic/ledger publication failure | Preserve the original refusal and durable state; no success output. Retry storage reconciliation only. Missing/malformed evidence is `VALIDATION_FAILED`, conflicting identities `IDEMPOTENCY_CONFLICT`, transient storage `RESOURCE_UNAVAILABLE`; a visible file alone, including failed sync/rollback residue, is not committed evidence. |
| Missing/mismatched snapshot or failed bounded scan | `SNAPSHOT_UNRESOLVED` or the existing repository refusal; no fallback, cache, partial report or scan retry. Failure after `read_started` keeps the spend. |

Results remain private until completion binds the request, authorizing act, spend, both memberships, per-set counts/metrics and overlap count; never average the sets.
Random-set temporal-neighbour correlation and post-release-selection provenance remain disclosed (#373).
Only the authorizing user may view results through the dedicated tool, with fresh interactive confirmation. Results belong in a separate namespace excluded from agent-facing tools and covered by an agent read-deny rule; no encryption or OS containment. Other callers refuse `HOLDOUT_ACCESS_DENIED` before output.
Ops durably audits the viewing actor, UTC when, verified request/report references and checked revocation-history revision before output; failed checks/audit publication release nothing. Retrieval returns unchanged verified artifacts with that history alongside them; it never scans, recomputes, tunes, reauthorizes or spends again.
Agent reconciliation may verify private artifacts but cannot return their metrics/report. No loader, training, selection or sweep report interface or `--no-ledger` final read is available.
Completion is the release point: later revocation appends audit history and accompanies retrieval; it cannot erase the completed report or renew read access.

## Inputs

Target research reads cover each full pinned partition, including null and
non-midnight observation times, subject to caller predicates and batch
filtering. The manifest population bound limits candidate rows, not RSS or
process memory — the recorded row-count sum of pinned fragments surviving
pruning ([data scan population rule](../data/ARCHITECTURE.md#invariants)).
Each scan uses that same bound for its snapshot, contract and predicates;
research adds no second result limit. Empty selected membership
allows zero result rows with a positive batch size. Month/day retries and
null-overflow refusals are not part of this contract: a complete read either
returns its population or raises its typed refusal. Predicates, exact-pair
filtering, batch filtering, successful row ordering and deduplication remain
part of the read contract.

The internal `_scan.read_table` and `_snapshot.read_table` readers accept an
optional `batch_filter` callback. They invoke it immediately after each Arrow
batch becomes a pandas frame, before retaining frames; it returns narrowed
rows or `None` for no rows. Omitting it or passing `None` preserves existing
reads. `_chains.load_chain_index` uses it to keep exact requested ticker/date
pairs while reading one manifest year at a time; stored dates retain their
original timestamp semantics and requested dates use the existing normalization.
Independent ticker and date memberships may prune a scan batch, but only the
exact pair mask — never their cross product — defines the retained rows.
Successful frames retain ordering and duplicates.
This bounds retained unmatched rows, not total process RSS.

- `--catalog` (sqlite path) and `--store-root` (`ArtifactStore` root):
  required by every CLI; opened once per run via
  `engine.v2.ops.bootstrap.open_catalog` / `engine.v2.foundation.ArtifactStore`.
- `--scope` (default `"shadow"`, `_scan.DEFAULT_SCOPE`/`_snapshot.DEFAULT_SCOPE`)
  and `--snapshot-id` (default `None`): together they resolve the *one*
  `SnapshotRef` the run reads. An explicit id reproduces a run after the
  scope's head has moved; omitting it resolves the scope's current pinned
  head. A CLI that resolves before its first table read passes that id to
  downstream runners; any later lookup uses the explicit id rather than
  resolving the possibly advanced scope head again.
- Tier-2 tables, read through `Repository.scan` against that one snapshot:
  `option_chains` (ORATS EOD quotes — fill quality, replay's chain reads),
  `option_daily` (Polygon traded bars — fill quality), `daily_market`
  (signal screen; also the replay/build-trades planning calendar, see
  below), `trades` (build-trades' read-before-append, reconcile's
  read-before-prune) and `earnings_events` (the canonical event universe
  both trade-table tools filter to).
- Replay and build-trades derive the trading-day calendar `plan_events`
  needs from that same pinned snapshot's `daily_market` table
  (`_pricing.trading_calendar_from_snapshot`, reading only the `date`
  column through `_scan.read_table`) — never from a local file or
  `INVESTING_PLAN_ROOT`. A snapshot with no `daily_market` table refuses
  (`CONTRACT_MISMATCH`) rather than inventing a calendar; see Failure
  semantics.
- `--since`, `--min-date`, `--years`, `--strategy` (repeatable, from
  `_pricing.STRUCTURES`): per-tool read/filter narrowing; never widen a read
  past the resolved snapshot.
- `tools/v2_replay.py --alpha A` is repeatable. When supplied, the ordered
  values replace replay's default `ALPHA_GRID`; repeated values remain
  separate requests, and each priced event produces one row per requested
  fill alpha. Values must be finite and in `[0, 1]`. When omitted,
  `_replay_run.run` uses its default `ALPHA_GRID`. The CLI continues to
  derive its calendar from the pinned snapshot rather than accepting an
  external calendar.
- `--dry-run` (build-trades, reconcile-trades): compute the candidate
  changeset, skip `commit_generic_table_candidate`.

## Outputs

- Fill quality / signal screen: a parquet of joined/scored rows and a
  markdown summary under `reports/`, both stamped with the resolved
  `snapshot_id`; fill quality also accepts `--csv`.
- Polygon fills: a JSON contract plan under `reports/` (contract count,
  contract-days, ordered contracts), carrying `snapshot_id`. No network
  call and no write to the pull's raw cache.
- Replay: per-trade records (`_trades_table.to_trades_table`) with
  `snapshot_id` and `provenance` columns set to this package's own
  provenance tag, never the legacy `engine.replay` tag.
- Build-trades / reconcile-trades: a **new version** of the `trades` table,
  committed through `engine.v2.data.generic_incremental`, plus a JSON
  outcome summary (`committed`, `outcome`, `committed_snapshot_id`,
  `emitted_revisions`) — never a rewrite of the table version the run read.
- Every output that carries data also carries the `snapshot_id` it was
  read from, so a report is reproducible without re-resolving anything.
  `experiment_trades.load_trades` includes its explicit snapshot and holdout
  context columns alongside the legacy-compatible trade columns.

## Dependencies

`engine.v2.data` (`Repository`, `errors`, `generic_incremental`, `catalog`,
`objects`, `manifests`, `query`, `computed_moves_table`, `iv_crush_table`), `engine.v2.
contracts.data` (`DataQuery`, `KeyPredicate`, `SnapshotRef`, `TimeInterval`,
`ObjectRef`, `TableContractRef`, `ColumnContract`, `TableContract`),
`engine.v2.contracts.final_holdout` (`FinalHoldoutRequest`, for `_final_population`),
`engine.v2.features.panel_math` (`history_features`),
`engine.v2.foundation` (`ArtifactStore`, `SystemClock`, `canonical_json`, `iter_canonical_json`,
`content_hash`, `to_document`, `market_calendar`, `experiment_exits.valid_exit_recipe`,
`experiment_holdouts.calendar_date` (canonical days for IV input/anchor/clock validation), plus
`final_holdout.request_from_document` and `typed.to_document` for `_final_population`) — the package's own
lower-layer dependencies, consistent with the generic "may import anything
at a lower layer" rule (`only_imports` is unset for this package in
`checks/layer_map.py`, i.e. no narrower restriction than that generic
rule). No import of `engine.v2.scoring`, `engine.v2.evaluation`,
`engine.v2.ledger` or any sibling/higher package. `engine.v2.ops.
bootstrap.open_catalog` is a CLI dependency, not a package one: the
`tools/v2_*.py` leaves call it to open `--catalog` before handing the
connection to this package (see Inputs); nothing under
`engine/v2/research/` imports `engine.v2.ops` itself, even though layer
7.0 (ops) is not below this package's own enforced layer (also 7.0) and
so could not be imported under the generic rule regardless.
Stored history publishes new objects/manifests through `data.catalog.commit_snapshot`;
this native write path advances only its dedicated context scope, never source heads.
`_pricing.py` holds pure pricing primitives copied from legacy
`engine.structures`/`engine.fills`/`engine.calendar` rather than reached
through a legacy adapter (supervisor decision: no legacy adapter for this
package). `polygon_fills.option_ticker` is likewise a verbatim copy of
`engine.data.sources.polygon.option_ticker` rather than an import, for the
same reason: a v2 package may not import legacy code without a declared
adapter.

**Known duplication (issue #69):** `_scan.py` (used directly by
`fill_quality.py`, `polygon_fills.py`, `signal_screen.py`, and
`_pricing.trading_calendar_from_snapshot`) and
`_snapshot.py` (used by `_chains.py`'s replay reads and
`_trades_publish.py`'s build/reconcile reads) are still two independent
modules. `_snapshot.read_table` delegates its scan to `_scan.read_table`,
forwarding caller `key_filter` predicates; `_snapshot.py` keeps its partition-key
resolution, its own no-partition refusal, and its own empty-result frame
shape. Merging the two remains issue #69: that would change both
modules' callers, outside this PR's one concern.

Callers: no `engine/` modules import this package (checked by `checks/import_layers.py`).
`experiments/runup_exit_report.py` consumes population admission, calendar/plan,
chain index, the entry-selection seams above and exit walking for read-only comparisons.
`experiments/magnitude_inputs.py` consumes `magnitude_inputs.preflight_magnitude_targets`;
`experiments/magnitude_report.py` consumes `preflight_magnitude_targets`,
`load_magnitude_targets` and `qualify_magnitude_training_targets`.
`experiments/native_registration.py` consumes
`experiment_population.load_population` and `magnitude_folds`' window/planner.
`experiments/magnitude_search_plan.py` consumes `magnitude_folds.annual_fold`.
`experiments/magnitude_search_evidence.py` consumes `read_stored_for_events`, `preflight_magnitude_targets`,
`qualify_magnitude_training_targets` and `magnitude_recipe` for shared inner/outer admission.
`experiments/magnitude_search_trials.py` directly consumes `load_magnitude_targets`,
owns legacy/v2 no-fit guards and delegates bounded inner fitting to
`experiments.magnitude_search_execution` (using ops executor thread constants); it performs no outer evaluation.
`experiments/magnitude_search_outer.py` directly consumes `load_magnitude_targets` for selected outer folds.
`experiments/magnitude_history_store.py` consumes the stored-history context, build and read APIs.
Other consumers include `tools/v2_*.py` CLI leaves. `experiments/common_v2.py` reads
trade, calendar, and chain inputs from pinned v2 data. The archived
trade-read caller omits required holdout context and is refused.
Historical EXP-147 reads its native replay tag from `experiment_trades.PROVENANCE`.
`experiments/v2_candidate_grid.py` (`price_candidate_grid`, issue #266
slice 2) reads
`_chains.filter_plan_by_availability`/`read_chain_keys_for`, `_plan.plan_events`,
`_pricing.STRUCTURES`/`execution_variant_label`/`trading_calendar_from_snapshot`,
and `replay.ALPHA_GRID`/`replay_one` to price one strategy family across a
grid-position sweep on one pinned snapshot. `experiments/EXP-186_.../run.py`
reads `_replay_run.events_frame` for its known-session event universe before
handing it to `price_candidate_grid`. None of these research callers are
parsed by the layering hook (it only parses `engine*` importers, and
`experiments/` is outside it too).

### Stored magnitude-history materialization

`experiments.magnitude_history_store` derives the supported build/read context.
The engine accepts digest-consistent contexts; materialization independently guards target identity/windows.
It does not authenticate arbitrary rehashed calendars as builder-derived.

| Boundary | Contract |
|---|---|
| Inputs | Exact source snapshot, admitted temporal plan, history start, fixed 5/3/1 clocks, per-feature missingness and installed producer source/environment identity. No experiment outputs. |
| Protected reads | The complete eligible singleton candidate union is preflighted before any outcome opens. Protected/ambiguous objects never open; admitted late labels may decode availability and be omitted before arithmetic. |
| Build boundaries | Full candidate preflight completes before any outcome opens. An admitted candidate outcome is decoded once per build while its run stream stays open (up to 32 runs at once); a run evicted by later runs is re-scanned if asked for again. No cache persists across builds. Memory has no fixed byte cap; production scale is unmeasured. |
| Batched outcome reads | The build reads candidate outcomes through `event_history.plan_with_reader`: same preflight as `plan_event_targets`, plus unique non-null `event_id`s; malformed candidates refuse `CONTRACT_MISMATCH`. Adjacent candidates of a ticker share one scan, and a protected, mixed or unadmitted fragment is never opened. Membership that is not exactly the preflighted set, an unplanned event, or a mismatched day, ticker or request refuses `HOLDOUT_ACCESS_DENIED` before that run opens any outcome. A scan failure propagates its native code, keeps only rows already returned for completed candidates, and fails the build before any snapshot, receipt or head is published (see Publication). Rows equal per-candidate reads; scans per build: one per run when candidates are requested ticker by ticker (the build's event order), more if runs are evicted. |
| Catalog reads | The CLI opens its `Repository` with `pin_snapshots=True`, so each scan and bound reuses the pinned snapshot's cached table membership instead of re-decoding it from the catalog (decodes were quadratic in candidates). `fragment_records` and a table's first use still decode membership; `earnings_events` is still read once by fold planning and once by preflight. Output, receipts and identities are unchanged; the pin is per process and immutable-identity keyed. |
| Progress | Long passes write progress to stderr only (stdout stays the JSON result): `progress.stage` logs start/end with elapsed seconds and peak RSS for the build's `history preflight`, `history prepare` and `history materialize` stages (a full-universe `load_population` logs only its end, and only when it took 5 s or more); `progress.Progress` logs `done/total` (counted after an item finishes), rate, ETA and peak RSS at most once per interval (default 30 s, and not before the configured item count), never per item, and nothing for passes shorter than the interval. `progress.decision` emits aggregate control decisions; opt-in `stage(report_refusals=True)` adds the native DataError code only (no message, no details). The public helper is shared by package readers. Logging never changes results or errors. |
| Per-forecast features | Shared history arithmetic preserves availability and numeric-refusal order. Consumed objects, omissions and availability accompany all eleven columns. Exclusion rosters remain transient; persisted lineage omits `holdout_exclusions` and the stored reader refuses it. Empty `n_prior` is zero; other missing values are null with `on_missing=null`. No runtime imputation or production v4 aggregates. |
| Output | Features-only `research_magnitude_history.v1` snapshot, separate from unchanged source. Three ordered horizons per event-isolated object; primary key `(event_id, context_id, horizon_sessions)`, physical context field and context partition. |
| Publication | Inspected objects/manifests and success receipt commit atomically under an initially empty deterministic scope. Failure admits no snapshot; complete unreferenced objects remain for normal cleanup. |
| Replay/conflict | Matching success receipt returns the original artifact without moving a head. A failed/conflict receipt for the same context refuses with `IDENTITY_CONFLICT`; conflicting receipts/heads also refuse. Nothing retries automatically. |
| Stored read | Supported entrypoint reconstructs context; engine verifies exact receipt/output and complete object isolation, then all row metadata before feature-column decoding. Checksum verification reads bytes. No historical outcomes or arithmetic; malformed reads return no partial rows. |
| Trust/authority | Native inspected catalog metadata is trusted, not independent attestation against malicious catalog writers. Schedule/source-publication availability remain unverified reconstruction. No experiment/final-read authority; registration v1.0–v1.3 and fixed baseline are unchanged. |
| Refusals | Missing or changed context/receipt: `IDENTITY_CONFLICT`; unprovable isolation: `HOLDOUT_ACCESS_DENIED`; bad stored schema/metadata/missingness: `CONTRACT_MISMATCH`. |
| Invalid projection | Non-string selections, malformed selection containers, unknown event/feature or duplicate feature: `CONTRACT_MISMATCH` before any payload scan. |
| Source/publication failures | Unknown source/feature snapshot ID: `SNAPSHOT_NOT_FOUND`; resolved source-pin mismatch during public build/read context derivation: `SNAPSHOT_UNRESOLVED`; direct `materialize` source/context mismatch: `IDENTITY_CONFLICT`. Non-native computed-moves contract: `CONTRACT_MISMATCH`; empty fold: `POPULATION_COLLAPSED`; invalid eligible label/features: `EXPERIMENT_VARIANT_FAILED`; competing head: `SNAPSHOT_CONFLICT`. Other native catalog/object/source-admission errors propagate. |

`magnitude_history_inputs.admit_stored(repository, snapshot, context)` is the consumer-only
metadata boundary used by `experiments/magnitude_history_inputs.py`: exact receipt/output,
features-only contract and complete event isolation, without payload decoding or arithmetic.
It delegates to the readers' shared `magnitude_history_store.preflight_stored`
receipt/contract/isolation boundary and returns no feature values.
Missing/changed receipt or pin refuses `IDENTITY_CONFLICT`; bad contract refuses
`CONTRACT_MISMATCH`; unprovable isolation refuses `HOLDOUT_ACCESS_DENIED`.
It does not derive or authenticate a context; the unlayered caller owns supported producer
derivation and native registration. `experiments/magnitude_history_report.py` consumes
the existing `read_stored` plus magnitude target preflight/load/qualification APIs.
Neither consumer modifies the build closure, stored artifact identity or final-read authority.

`magnitude_history_store.read_stored_for_events` is the projected search reader.
Every call revalidates the complete receipt, isolation, metadata and feature content,
including events and columns not retained. Complete metadata admission precedes
numeric validation; corruption or inconsistent source content refuses the read.
Returned `inputs_id` names the exact original `read_stored` full-source envelope,
not the projection. Selected rows contain only primary-key identity and requested
numeric features, without lineage strings. Full-snapshot lineage is not retained;
memory still scales with metadata and selected rows, with no fixed overall byte cap.
A prior successful read never substitutes for fresh validation.
The original full-row `read_stored` API and its returned envelope remain unchanged.

## Opt-in eligibility policy boundary

`eligibility_policy.eligibility_policy` declares an explicit supported version and
nonnegative integer calendar-day width. `plan_eligibility` is metadata-only and
opt-in; existing consumers are unchanged. It uses `experiment_population.canonical_events`
for exact committed-snapshot admission and the six-column metadata projection and `experiment_trades.classify_events`
for detached membership/structural ambiguity labels and caller-pinned context.
Shared `snapshot_inputs` validates reference shape and exact committed identity;
the pure classifier checks shape only, without claiming repository commitment.

The closed producer envelope binds snapshot/month/policy and event groups with
explicit label bounds and every declared horizon's decision/source ceiling.
Bounds contain the event, with source ceiling <= decision <= event. These are
producer declarations, not proof of price-bracketing or publication times.
No calendar guessing, outcome/feature decoding, write, registration or fit occurs.

Random-held windows are merged per ticker. Inclusive event/label interval or
forecast-point overlap excludes an entire development group; shared random,
rolling and ambiguous labels also exclude it. Results bind sorted groups, policy,
metadata identity, merged windows, retained IDs, reasons, coverage and a separately
preserved evaluation manifest into `plan_id`. Supplied documents are detached.
There are no retries, persistence, cache or partial results. Bounded membership,
metadata, window and group progress plus stage timing/RSS go to stderr; admission
and planning refusals report only the native error code, not row values.

| Condition | Outcome |
| --- | --- |
| Unknown, malformed or mismatched snapshot reference | `SNAPSHOT_UNRESOLVED` |
| Repository integrity failure other than unknown snapshot | Original native error propagates |
| Unsupported policy, stale digest/envelope, malformed bounds, duplicate declared group IDs or malformed/duplicate evaluation IDs | `CONTRACT_MISMATCH` |
| Missing/duplicate classifier columns, malformed event IDs or metadata values that prevent classification | `CONTRACT_MISMATCH` |
| Invalid scalar dates/sessions or structural collisions handled by membership classification | Successful classification with `ambiguous` membership; structural ambiguity IDs returned |
| Invalid as-of month, unknown group/evaluation identity or colliding canonical population IDs | `HOLDOUT_ACCESS_DENIED` |
| Ambiguous random-event identity or overflowing window bounds | `HOLDOUT_ACCESS_DENIED` |
| Valid protected/overlapping development group | Excluded whole event with reasons; evaluation IDs preserved |

This is neither a physical-read capability nor a statistical isolation guarantee.
Long-memory/cross-ticker effects remain; chronological and random holdouts are
complementary, not interchangeable.

## External systems and libraries

Direct libraries include pandas/numpy for tabular/numerical work and pyarrow
for native stored-history Parquet. No network call (Polygon/ORATS pulls stay in
`engine/data/pulls/`, legacy and unmoved). Snapshot access uses
`ArtifactStore` and the sqlite catalog connection it is handed. Two
exceptions, both plain local files outside `ArtifactStore` and never
snapshot-store objects: report writers (`fill_quality.write_report`,
`signal_screen.write_report`, `polygon_fills.run`'s own write step) write
Markdown/parquet/CSV/JSON directly under the caller-provided
`reports_dir`/`out_dir`; and `_pricing.trading_calendar()` reads a local
CSV (`earnings_predictions/data/raw/polygon/gspc_daily.csv`, off
`INVESTING_PLAN_ROOT` or the worktree root). That CSV reader is kept only
for legacy-parity tests and direct library callers — no OTHER module in
this package imports or calls it (`_plan.py` no longer does; a guard test,
`test_trading_calendar_csv_fallback_is_not_reachable_from_the_package`,
statically checks every file in this package but `_pricing.py` itself for
an import or a call of the bare name). `_replay_run.run` and
`_build_run.run` resolve the replay/build-trades calendar from the pinned
snapshot's own `daily_market` table (`_pricing.trading_calendar_from_snapshot`)
before calling `replay()`; `replay()` itself also derives one this same
way, from its own `(repository, snapshot_ref)` arguments, for any OTHER
caller that leaves `calendar` unset — so every path to a calendar in this
package ends at a pinned snapshot or an explicit argument, never a local
file, and a run's calendar is fixed by its `snapshot_id` like everything
else it reads (see Failure semantics, Invariants). No third-party service.

## Failure semantics

The 4c R1–R6 template covers data-boundary refusals, generally `DataError`
(`engine.v2.data.errors.DataError`) carrying a registered `DATA_FAILURE_CODES` code.
CLI leaves catch these at `main()` and exit 2: stored-history runtime refusals
are JSON on stdout; other tools print code/message on stderr.
Argument validation is separate because it can refuse before that handler.
One pre-existing data-path exception, called out below
where it applies (`_snapshot.py`'s bare `ValueError`, issue #70), escapes as
an uncaught traceback instead.

- **CLI argument parsing (before R1–R6).**
  - **A supplied replay `--alpha` is non-finite or outside `[0, 1]`.**
    The CLI parser exits 2 before `open_catalog` or `ArtifactStore` is
    called, so no replay runs.
- **R1, missing input.**
  - **No committed scope head, and no explicit `--snapshot-id`.**
    `resolve_pinned(scope)` (or, for the `_snapshot.py`-backed tools,
    `head_generation`) raises `SNAPSHOT_NOT_READY` (`category="dependency"`,
    **retryable**) — this is the closest this package comes to "refuses
    rather than silently reading latest": a scope with nothing committed
    to it has no head to read, so the run stops rather than resolving
    nothing. A scope *with* a committed head and no explicit
    `--snapshot-id` resolves that head deterministically and once; the
    resolved id is always written into the output so the exact snapshot a
    report came from is never ambiguous after the fact, and `--snapshot-id`
    lets a later run reproduce it even after the head has since moved.
  - **An explicit `--snapshot-id` that does not exist.**
    `Repository.resolve`/`resolve_full` raises `SNAPSHOT_NOT_FOUND`
    (`category="dependency"`, not retryable) — an unknown id is refused,
    never silently substituted for the scope head.
  - **A resolved snapshot's manifest fails its own identity or hash
    check** (a dataset version, contract, or fragment that does not match
    its catalog identity, references an unknown object, or a scan result
    that is out of the manifest's declared primary-key order).
    `Repository.resolve`/`resolve_full`/`scan` raises `MANIFEST_CORRUPT`
    (`category="integrity"`, not retryable) — a tampered snapshot is
    refused rather than read.
  - **A table absent from the resolved snapshot** (`_scan.py`'s path).
    `CONTRACT_MISMATCH` (`category="validation"`, not retryable).
    `_scan.read_table` itself returns an *empty* frame, not a refusal, when
    a partition filter simply matches no fragment records. A replay or
    build-trades run receives the same refusal when its pinned snapshot
    has no `daily_market` table:
    `_pricing.trading_calendar_from_snapshot` reads `daily_market` through
    this same `_scan.read_table` path, so a snapshot that cannot supply a
    calendar is refused here rather than falling back to
    `trading_calendar()`'s local CSV.
  - **A `daily_market` table that IS part of the snapshot but has no valid
    dates in it** (present, zero fragments, or every date unparseable).
    `_pricing._calendar_from_dates` raises `CALENDAR_UNAVAILABLE`
    (`category="validation"`, not retryable) — distinct from the
    `CONTRACT_MISMATCH` case above, which is about the table's absence, not
    its contents. `plan_events` (`_plan.py`) raises the same code for a
    different reason: it has no repository/snapshot of its own, so a
    caller that reaches it with `calendar=None` (bypassing `replay()`'s own
    derivation, or calling `plan_events` directly) is refused rather than
    silently reading `trading_calendar()`'s CSV — the fallback this package
    used to have. Either way, exit 2 with a code, never an uncaught
    `ValueError`.
  - **No declared partition column, or no partition values available at
    all** (`_snapshot.py`'s path only — `replay.py`/`_chains.py`/
    `_trades_publish.py`, not `_scan.py`). `_snapshot.read_table` raises a
    bare `ValueError`, not a `DataError`, when it cannot build even one
    `KeyPredicate` — the table contract declares no partition column, or
    no partition values are available at all, i.e. the snapshot has no
    fragments for that table. An explicitly empty `partition_keys=[]` is
    falsy in Python, so it is treated the same as omitted and falls back
    to deriving values from the snapshot's own fragment records — it does
    not by itself trigger this path. Partition keys that are simply
    valid-but-non-matching still build a predicate and return an empty
    frame, same as `_scan.py`. The
    CLIs built on this path (`v2_replay.py`, `v2_build_trades.py`,
    `v2_reconcile_trades.py`) catch only `DataError` at `main()`, so this
    one condition escapes as an uncaught traceback rather than the typed
    refusal every other condition here gets. Pre-existing, not introduced
    or fixed by this doc; tracked as a follow-up (issue #70).
  - **No overlap rows after a join/filter** (fill quality's
    `since`-filtered join, or `experiment_trades.load_trades`'s
    strategy/provenance filter on the read `trades` table).
    `POPULATION_COLLAPSED` (`category="validation"`, not retryable) — an
    empty report, or an empty trades frame, is refused rather than returned
    as a zero-row "result".
  - **A resolved snapshot with no `trades` table at all**
    (`experiment_trades.load_trades`, via `_trades_publish.
    read_existing_trades`). `CONTRACT_MISMATCH`, the same code and the same
    helper `tools/v2_build_trades.py`'s own read-before-append already uses
    for this condition. `load_trades` raises the same `CONTRACT_MISMATCH`
    for a zero-fragment `earnings_events` table (the same issue #70 bare
    `ValueError`, converted the same way). Missing or ambiguous event joins
    are excluded under the holdout population contract above.
- **R2, cache.** None: every run resolves its snapshot and reads its tables
  fresh through `Repository.scan`; nothing is cached across runs or across
  processes. Within one run, the resolved `SnapshotRef` and, in the replay
  path, the loaded `ChainIndex` are held once in memory for that run's
  reads only (module-level caches the legacy code held for a mutable store
  are gone, on purpose — a pinned snapshot never changes under a run, so
  there is nothing to invalidate). The one exception is the magnitude
  observed-session day tuple, held per pinned `Repository` and snapshot id.
  With `index=None`, replay scopes chain availability and loaded chain
  data to the plan's own years, tickers and dates. The resulting
  `ChainIndex` contains only available plan keys. An explicit `index=`
  bypasses chain reads. `_build_run.run` creates an independent replay
  for each strategy.
- **R3, retry.** No scan automatically retries. `SNAPSHOT_NOT_READY` and
  `SNAPSHOT_CONFLICT` are the only two retryable codes this package can raise;
  an operator may rerun after the missing head appears or a conflicting
  writer finishes. Limit and integrity refusals, including
  `RESULT_LIMIT_EXCEEDED` and `MANIFEST_CORRUPT`, are not retried.
- **R4, transaction.** Build-trades and reconcile-trades are the only
  writers here; both go through the one shared path,
  `_trades_publish.publish` → `engine.v2.data.generic_incremental.
  build_generic_table_candidate` / `commit_generic_table_candidate`. The
  commit is fenced to the exact snapshot the run read
  (`expected_head_snapshot_id`, `expected_head_generation`): if another
  writer advanced the scope's head in between, the commit refuses
  `SNAPSHOT_CONFLICT` (`category="dependency"`, retryable) rather than
  silently overwriting or merging. `--dry-run` builds the same candidate
  and stops before the commit call, so the changeset can be inspected with
  no write at all.
- **R5, partial result/write.** Read batches stay provisional until scan
  exhaustion; a later failure returns no partial frame. A smaller explicit
  caller limit raises `RESULT_LIMIT_EXCEEDED`. Writes use the atomic
  `commit_generic_table_candidate`: a pre-call crash leaves the parent
  untouched, and commit atomicity is guaranteed by `generic_incremental`.
- **R6, idempotency.** A byte-identical read of the same pinned snapshot and
  request returns identical rows. Read-only tools (fill quality, polygon
  fills, signal screen, replay by itself) are naturally idempotent: the same
  resolved snapshot and arguments always read the same rows and produce a
  byte-identical report. Build-trades and reconcile-trades are NOT
  idempotent in the sense of "running twice changes nothing": each
  successful commit publishes a new `trades` version pinned to the
  snapshot it read, so re-running against the same parent snapshot after a
  successful commit reads a different (newer) `trades` version and can
  produce a different, non-empty changeset (or a `noop` outcome when the
  replay recomputes the same revisions the previous run already
  published). The commit call itself is not re-entrant/retried internally;
  a caller that wants "publish this exact changeset once" relies on the
  `SNAPSHOT_CONFLICT` fence, not on this package silently no-op-ing a
  duplicate call.

## Invariants

- Every run reads one selected snapshot. A CLI that resolves before its
  first table read passes that `snapshot_id` to downstream runners, whose
  later lookups use the explicit id; the id is threaded through every read
  and written into every output
  (root doc §5's "no silent default" invariant, applied to research reads) —
  including the experiment loader's returned context columns (see Outputs).
- Every experiment runner reads from its pinned v2 snapshot, applying shared random/rolling holdout enforcement through the reader appropriate to its inputs:
  `experiment_trades.load_trades` for trades; `experiment_population.load_population` for populations and its prediction/history consumers. Inadmissible reads refuse.
  Direct parquet/file reads bypass snapshot pinning and both holdout sets and are inadmissible.
- Never reads the legacy mutable Tier-2 store (`engine.data.store`) —
  every table read is a bounded `Repository.scan` against the one resolved
  snapshot.
- Every scan carries its own partition predicate alongside caller key
  predicates, so result rows cannot duplicate across partitions.
- No production entrypoint derives its calendar from a local file or
  environment variable: `_replay_run.run` and `_build_run.run` resolve the
  planning calendar from the pinned snapshot's own `daily_market` table
  (`_pricing.trading_calendar_from_snapshot`) and pass it explicitly into
  `replay()`; `replay()` itself does the same derivation, from its own
  `(repository, snapshot_ref)`, for any caller that leaves `calendar`
  unset, so the guarantee holds for every caller of `replay()`, not just
  these two. Either path refuses rather than inventing one — a missing
  `daily_market` table is `CONTRACT_MISMATCH`, one present but with no
  valid dates is `CALENDAR_UNAVAILABLE` — and `plan_events` itself refuses
  `CALENDAR_UNAVAILABLE` if it is ever reached with no calendar at all,
  never falling back to `trading_calendar()`'s CSV read (that fallback has
  been removed from `_plan.py`; `trading_calendar` is no longer imported
  or called anywhere in this package outside `_pricing.py`'s own
  definition, checked by
  `test_trading_calendar_csv_fallback_is_not_reachable_from_the_package`).
  Changing the CSV cannot change a replay or build-trades run's output;
  only a different pinned `snapshot_id` can. `trading_calendar()` itself
  stays, unchanged, for legacy-parity tests and direct library callers —
  it is never reached from anywhere in this package's own code.
- **A `--years`-scoped rebuild only ever tombstones rows inside its own
  scope** (issue #108). `_build_run.run` narrows the REPLAY to `--years` via
  `filter_events`, and passes the same `years` into
  `_trades_revisions.revisions_for_rebuild`, which narrows its tombstone set
  the same way: it tombstones an existing replay-provenance row of a
  `rebuilt_strategies` member only when the row's own `year` column is also
  inside the rebuilt `years` (when `years` is given) AND the rebuild no
  longer produces its `trade_id`. A year outside the rebuilt scope is never
  touched, so `tools/v2_build_trades.py --years Y` cannot delete another
  year's already-published rows for the same strategy — the requested
  year's own no-longer-produced rows are still tombstoned. Omitting
  `--years` continues to rebuild (and tombstone) every year, unchanged
  from before this fix.
- Never mutates the legacy trades ledger and never calls a network
  provider; `engine/build_trades.py` and `engine/data/pulls` (legacy,
  unchanged) keep doing both.
- A trade-table write always goes through
  `engine.v2.data.generic_incremental`'s fenced commit — never a
  read-modify-write of an existing table version.
- Pricing/planning/chain-indexing cores that moved from legacy
  (`replay.py`, `_plan.py`, `_chains.py`, `_trades_table.py`) stay
  byte-identical to their legacy counterparts; only the store-reaching
  edges were rewritten, so the v2 path and the legacy board cannot drift
  on arithmetic. `tests/test_v2_research_tools.py` /
  `test_v2_research_replay.py` / `test_v2_research_build_trades.py` /
  `test_v2_research_reconcile_trades.py` enforce this by comparing against
  the legacy function on the same synthetic frame.

## Diagram

```mermaid
flowchart LR
    CLI["tools/v2_*.py"] -->|"--scope / --snapshot-id"| Resolve["resolve_pinned / resolve\n(_scan.py or _snapshot.py)"]
    Resolve -->|"SnapshotRef, pinned for the run"| Read["Repository.scan\n(read_table)"]
    Read --> Core["pure analysis / pricing core\n(fill_quality, signal_screen,\nreplay, build_trades)"]
    Core -->|"read-only tools"| Report["reports/*.{md,parquet,json}\n+ snapshot_id"]
    Core -->|"build-trades / reconcile-trades"| Publish["_trades_publish.publish"]
    Publish --> Commit["generic_incremental.commit_generic_table_candidate\n(fenced: expected_head_snapshot_id)"]
    Commit -->|ok| NewVersion["new trades\ndataset version"]
    Commit -->|"head moved"| Conflict["SNAPSHOT_CONFLICT"]
```
