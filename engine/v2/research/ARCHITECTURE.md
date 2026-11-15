# `engine/v2/research` — architecture

Layer 7.0 in `checks/layer_map.py` and root `/ARCHITECTURE.md`. Native snapshot-reading counterparts to `tools/signal_screen.py`, `tools/fill_quality.py`, `engine/data/pulls/polygon_fills.py`'s read path, `engine/replay.py` (committed snapshots) and `engine/build_trades.py`'s v2 write path. Legacy modules remain untouched and publish as before; parallel `tools/v2_*` CLIs read pinned v2 snapshots. `tools/log_diagnostics.py` stays put: it reads a local transactions-log CSV and an optional sibling `metrics_*.json`, never Tier-2.

## Purpose

Offline analysis CLIs cover real-fill quality, cross-sectional screening, the real-trade pull's universe plan, deterministic as-of replay and its trade table, reading exactly one resolved named snapshot per run instead of mutable Tier-2. Measured tables/reports are for a person: no research verdict, provider fetch or legacy-ledger mutation. `experiment_trades.load_trades` extends it to v2 experiments (committed `trades` only). Completed experiments need not stay runnable; historical wrappers are unsupported execution boundaries.

## Source-free magnitude distributions

The source-free numerical APIs `magnitude_distribution` (`recipe`, `histogram`, `validate_distribution`, `bin_index`), `magnitude_distribution_math` (`components`, `cdf`, `mean`, `crps`, `bin_scores`, `brier`, `empirical_crps`, `prepare_empirical_crps`, `mean_matched_control`), `magnitude_distribution_payoff` (`bridge_summary`, `empirical_bridge_summary`) and the opt-in `magnitude_distribution_diagnostics` own no source admission; consumers are experiment distribution declarations/fitting/OOF adapters and synthetic tests. The closed plain-JSON `research_magnitude_distribution.v1` represents absolute simple-return percentage points: quarter-point half-open bins through 20, then uncapped overflow (exact boundaries go right), 81 finite masses in [0, 1] summing to one within absolute 1e-12 without repair or smoothing; tail policy is `unresolved` or `empirical_excess` (equal-weight nonnegative excesses relative to 20). Full-law APIs refuse unresolved tails even at zero overflow mass; `bin_scores` allows them. Inputs are plain `int`/`float` (not bool/subclasses), finite (magnitudes, tail excesses and empirical samples also nonnegative), probabilities in [0,1], empirical samples a nonempty plain list; documents are returned detached and no caller value is echoed in refusals. The payoff bridge returns `expected_profit`, `probability_profit`, `mean_exit_value` and `probability_negative_exit` as per-share dollar decision-quote proxies, not realized net-profit forecasts. `mean_matched_control` is named support repair (target mean must be positive); its original payoff-module export stays compatible, including its legacy refusal Problem. The caller owns sample eligibility, resource limits and sealed PIT randomization. No helper does I/O, writes, caching, retries or transactions, or provides source/fold/calibration envelope, fitting, holdout access or production wiring; `earnings_magnitude.v1` is unchanged. Per-function semantics, algorithms and costs: `guides/research_architecture_design_notes.md`.

| Condition | Outcome |
| --- | --- |
| Malformed histogram schema/types/support/masses/tail policy/counts | Non-retryable `CONTRACT_MISMATCH`; no partial result |
| Invalid `bin_index` numeric type, negative/nonfinite or unrepresentable magnitude | Non-retryable `CONTRACT_MISMATCH`; no index |
| Invalid math/payoff/diagnostic arguments (input rules above), unresolved full-law tail, nonfinite/unrepresentable arithmetic, nonpositive target mean or zero repaired mean/scale | Non-retryable `CONTRACT_MISMATCH`; no partial result |
| Zero observed-bin probability | `bin_scores.log_loss` is positive infinity; no smoothing |
| Omitted required argument to any numerical helper | Python `TypeError` |

## Stage-2 magnitude panel foundation

`magnitude_confirmed_metadata.build_index(repository, snapshot, *, tickers, limits)` is complete pinned-view admission (consumed by `magnitude_panel_qualified`, `experiments.size_panel`) returning one detached, content-hashed proof (`index_id` hashes the proof without that field; `limits` gate admission and retention but are not hashed; identical successful calls return the same proof). Catalog admission uses one read transaction released before the `bounded_snapshot` session is yielded; every limit is explicit (no defaults, retries, cache, fallback SQL or durable writes); inventory caps bind before any object read and retained-row/byte caps as data is read (not an RSS guarantee). Admission authenticates exact `snapshot_inputs` and the frozen earnings contract allowlist. Failure returns no partial proof; refusal codes are in the table below. Proof `counts` carry `unconfirmed_date_conflict` and `unconfirmed_cluster_ambiguous`; `policy.conflicts` is `unconfirm_date_and_cluster_conflicts.v2`; receipt references stay unopened. Complete membership proves neither expected-population coverage nor PIT/protected/numerical eligibility.

`confirmed_metadata.reduce_sequences` is a public pure seven-column DataFrame reducer of a complete caller-supplied view (not evidence of completeness; strict naive `datetime64[ns]` types precede `confirmed_event_metadata.classify_metadata`); `event_sequence(warmup=4)` counts all supplied confirmed rows from zero with no date cutoff. All rows, including unconfirmed siblings, participate in duplicate ID and ticker/day refusal. A `date_conflict` row is unconfirmed with reason `date_conflict`, and every row of a duplicated `(ticker, event_cluster_id)` with reason `cluster_ambiguous`; neither refuses the view. This is a deliberate modeling assumption (rationale: `experiments/SUPER_SIZE/CONFIRMED_METADATA_DESIGN.md`) matching live behaviour: both are disregarded for training and evaluation, a later-dated sibling may unconfirm an earlier row, the view applies no as-of cutoff, and equal clusters across tickers are independent. Such rows leave the confirmed sequence (ordinals, warm-up) but stay in `selected_rows`; a row already unconfirmed for another reason keeps that reason and is not counted. `reduce_with_counts` returns `(sequences, {unconfirmed_date_conflict, unconfirmed_cluster_ambiguous})`, `reduce_sequences` the sequences only. Shape/type refusals (table below) precede any conflict check; counts come from the same pass as the sequences, so a refusal records nothing. No index identity, source admission, coverage, holdout/value eligibility or runtime integration.

`magnitude_panel_table` (native `TableContract`/`ColumnContract`, JSON and canonical hashing) has an immutable v1 event/context/horizon key with ordered H/HM/HMO dictionaries, excluding learned Tier4 outputs/current-event outcomes; rows require exact cohort coverage, finite typed values, missing/quote status, row/fragment identity and a canonical SHA-256 `content_hash`; contexts are reconstructed-only, never relabeled observed; pure validation checks digest syntax/binding, not source membership or value equality. `magnitude_panel_plan` (no production caller; no raw-payload read authority) reconstructs protected population/calendar metadata and checks exact history receipts and isolated fragments without payloads; detached validation checks JSON types, closed schemas, native folds, fragments and calendar-source consistency, repository preflight authenticates membership/source truth, and horizon cutoffs require native calendar reconstruction. Public helpers `validate_context_identity`, `validate_event_forecasts`, `preflight_source`, `validate_features`, `magnitude_features.calendar_input` and `magnitude_inputs.target_object_identity` (detaches admitted `event_id`, `fragment_id`, `object_id` and object `content_hash` without reading target values) share existing validation without new authority; Panel boundaries require plain JSON; detached refusals use `CONTRACT_MISMATCH` (including `canonical_day` on non-canonical dates), native refusals propagate. `magnitude_panel_v2_policies.policy_definitions()` supplies detached source/derivation/quote/missing policies ([parity design](https://github.com/yshewchuk/investment-validation/blob/26b5958d868afc5189611aa77ec1774a9e246d71/experiments/SUPER_SIZE/STAGE2_DESIGN.md#incumbent-parity-amendment-october-24-design-only)) to the additive `magnitude_panel_v2_table` (`CONTRACT`, `feature_dictionary(subset)`, `recipe(subset)`; H/HM/HMO have 11/28/42 ordered numeric descriptors and 11/28/43 raw model columns, HMO alone appending `has_implied_quote`). Inline source/derivation policies change all dictionary hashes (including H); v1 APIs, contracts, hashes and the eleven H numeric values/order are unchanged; contract and dictionary hashes are distinct and admit no sources, snapshots or rows. There is no writer, materializer, CLI, model, v2 context, reader or runtime activation.

| Condition | Outcome |
| --- | --- |
| Index malformed request/row/partition, incomplete inventory, or unsupported holder/session | Nonretryable `CONTRACT_MISMATCH`; no partial index |
| Index missing/nonexact/unresolvable native snapshot | `SNAPSHOT_UNRESOLVED`; no object access or index |
| Index unsupported exact earnings contract; explicit cap exceeded; semantic conflict | `UNSUPPORTED_CONTRACT`; `RESULT_LIMIT_EXCEEDED`; `IDENTITY_CONFLICT`, respectively; no partial index |
| Index native catalog/query/object failure | Same exception/code/category/retryability/stage/message/details; no partial index |
| Index success after every partition exhausts | Detached plain-JSON proof; mutating it cannot alter admitted/caller state |
| Malformed reducer request/frame/row, including unrequested ticker | `CONTRACT_MISMATCH`; no partial sequence |
| Reducer duplicate ID or ticker/day | `IDENTITY_CONFLICT`; no partial sequence (date-conflict and duplicated ticker/cluster rows are unconfirmed, not refused) |
| Reducer native failure | Existing `DataError` code/retryability unchanged; no partial sequence |
| `policy_definitions()` | No arguments; detached metadata only, no I/O/fits/verification. Shared declaration exceptions propagate unchanged. |
| Invalid v2 subset (not a plain H/HM/HMO string); omitted argument | `CONTRACT_MISMATCH`; omitted argument raises Python `TypeError`. |
| Invalid detached panel identity/schema, nested digest/folds, snapshot, calendar, cutoff or row; non-plain builder producer/snapshot or history metadata, or invalid lane | `CONTRACT_MISMATCH` |
| Malformed or forged direct calendar snapshot | `SNAPSHOT_UNRESOLVED` |
| Empty native observed session calendar | `CALENDAR_UNAVAILABLE` |
| Native history identity/digest or committed binding differs | `IDENTITY_CONFLICT` |
| Builder history metadata exceeds structural depth | Existing native `IDENTITY_CONFLICT` |
| Structurally invalid direct history identity / source-admission input | `CONTRACT_MISMATCH` / `IDENTITY_CONFLICT`, respectively; native identity/safety refusals remain unchanged |
| Lineage is not plain UTF-8 text or exceeds 1 MiB UTF-8 per row | `CONTRACT_MISMATCH` before JSON decoding |
| Invalid `validate_event_forecasts`, `validate_features`, `canonical_day` or feature subset input | `CONTRACT_MISMATCH` |
| Protected requested event or unprovable physical isolation | `HOLDOUT_ACCESS_DENIED` |
| Snapshot ID is absent from the native catalog | `SNAPSHOT_NOT_FOUND` |
| Catalog snapshot/manifest identity is inconsistent | `MANIFEST_CORRUPT` |
| Repository-reconstructed temporal population lacks training/evaluation coverage | `POPULATION_COLLAPSED` |
| Cache / retry | Observed days are reused per weak-referenceable pinned repository and snapshot ID; failures are not cached. `preflight_context` reconstructs metadata through native repository APIs (including their catalog caches). |
| Transaction / partial write / idempotency | No enclosing/write transaction or durable writes; native repository resolution/table reads use read transactions. Identical inputs and metadata give the same `context_id`. |

## Partial daily-market blocks

`magnitude_panel_daily.build_daily_blocks` is an opt-in library-only reader (only synthetic tests call it). Inputs are pinned source/history/price_history snapshots, the metadata-only v1 plan, plain-JSON producer metadata, grouped eligibility declarations and explicit source/event/byte bounds; the output has a distinct versioned context and detached daily-market/regime blocks, never stored-panel admission, and the context binds parent, bounds, policy, exclusions, snapshot fragment inventory and producer (declarations, not installed-code attestation). The thirteen daily fields and nine historical `spy_*` values obey each forecast's source ceiling; regime closes come from pinned `price_history` `REGIME_TICKER` (`^GSPC`) via `get_price_series`, never daily SPY spot. Reads are row-capped (one ticker/year scan per event ticker plus one regime read at the latest ceiling) and a required context-bound `max_object_bytes` bounds new value scans; availability is reconstructed, not observed-PIT/fit-admissible. No history/runup block or outcomes are read; no CLI, fit, registration or runtime consumer; calls revalidate and reread pinned sources. Protected evaluation membership confers no reads. Detailed bounds and exclusion semantics: `guides/research_architecture_design_notes.md`.

| Condition | Outcome |
| --- | --- |
| Non-plain/malformed producer or detached parent, changed policy/clocks, unpinned repository, invalid limit/window or source contract | `CONTRACT_MISMATCH`; no partial result |
| Duplicate/malformed source date or value, negative implied quote, nonpositive/non-convertible `^GSPC` regime close or infinite regime result | `CONTRACT_MISMATCH`; no partial result |
| Invalid byte cap (including None/bool/nonpositive) | Native `QUERY_NOT_BOUNDED` before repository access |
| Declared event, cumulative row, physical-object or expanded-object limit exceeded | `RESULT_LIMIT_EXCEEDED`; no partial result |
| No events survive shared eligibility | `POPULATION_COLLAPSED`; no value reads |
| Snapshot, parent-binding, catalog or shared admission failure | Native refusal propagates unchanged; no partial result |
| Missing optional field/history, zero implied quote, absent/late market-cap anchor | Null with explicit missing/quote status; no cohort complete-case filtering |

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

The library entrypoints below are this package's real public interface:

`magnitude_panel_v2_table.CONTRACT`, `magnitude_panel_v2_table.feature_dictionary`, `magnitude_panel_v2_table.recipe`, `magnitude_panel_v2_policies.policy_definitions`, `eligibility_policy.eligibility_policy`, `eligibility_policy.plan_eligibility`,
`experiment_population.canonical_events`, `experiment_trades.classify_events`,
`snapshot_inputs.snapshot_reference`, `snapshot_inputs.admit_snapshot`,
`signal_screen.run`, `fill_quality.run`, `polygon_fills.run`, `replay.replay`,
`replay.replay_one`, `replay.ALPHA_GRID`, `_replay_run.run`,
`_replay_run.events_frame`, `_plan.plan_events`, `_chains.ChainIndex`,
`_chains.load_chain_index`, `_chains.filter_plan_by_availability`,
`_chains.read_chain_keys`, `_chains.read_chain_keys_for`, `_chains.read_chains_for_years`,
`_trades_table.to_trades_table`, `_build_run.run`,
`reconcile_trades.run`, `_trades_publish.publish`, `build_trades.coverage`,
`_pricing.STRUCTURES`, `_pricing.trading_calendar_from_snapshot`,
`_pricing.execution_variant_label`.

The RUNUP comparison's public entry-selection seams are
`replay.clean_chain_rows` (returns valid quote rows without input mutation),
`_pricing.MIN_MEANINGFUL_COST`, `ChainSnapshot`, `FillModel`, `StructureError`
and `price_structure` (existing pure native selection/spread arithmetic).

`experiment_trades.load_trades(repository, snapshot, strategy, *, as_of_month=None,
purpose="training", event_ids=None)` is a library call with no dedicated CLI. Given
a `Repository` and resolved `SnapshotRef`, it returns one strategy's committed,
session-joined trades with legacy-compatible columns and holdout context; an
omitted/`None` as-of month refuses `HOLDOUT_ACCESS_DENIED` (no implicit month).
`_pricing.trading_calendar_from_snapshot` lets pinned-snapshot repricers share
`replay()`'s calendar; `_pricing.execution_variant_label` gives replay's
execution-variant labels for each priced grid-position step.

Experiment training, selection-fold and sweep reads exclude the union of the
two holdouts specified in [#373](https://github.com/yshewchuk/investment-validation/pull/373).
Shared membership definitions live in `foundation.experiment_holdouts`.
The explicit rolling as-of month and both membership versions accompany the
snapshot id in returned columns; exclusion labels are in `holdout_exclusions`
frame attributes. Conflicting event IDs at one ticker/date/session or cluster,
and other ambiguous canonical identity/date matches, are excluded and labelled
`ambiguous`; released eligible rows are labelled `post-release selection`.
A requested `event_ids` population containing any excluded event raises the
non-retryable `DataError(HOLDOUT_ACCESS_DENIED)` during population validation,
before returning a frame to metric/report writers; so do missing/invalid
context, an as-of month later than the current UTC month, unknown purposes and an
entirely excluded population. There is no date fallback, partial frame, report
write or retry. Final holdout reads are unavailable. The loaders write nothing;
the recorded native prediction runner appends one plain `refused` row (no
metrics; replay adds none) when a `run` on an admitted registration hits
`HOLDOUT_ACCESS_DENIED`, unless a different immutable outcome for that variant is
already saved (identity conflict, ledger unchanged). `register`, a run with no
admitted registration, and `--no-ledger` smoke write none. `ledger_context`
excludes `refused*` rows from specs tried.

`experiment_population.load_population` validates an exact committed `SnapshotRef` and returns canonical event metadata under the same holdout contract, without reading outcomes; it shares the trade loader's membership and ambiguity logic. Explicit IDs are checked against the full canonical calendar, including conflicting siblings, before any caller consumes an eligible population; neither source dates nor ticker names substitute for event IDs, and multiple IDs on one ticker/day are ambiguous even when their sessions differ.

`prediction_inputs.load_prediction_targets` consumes this population first, then scans each eligible ticker/event-date pair in pinned `computed_moves`. The history and magnitude readers (`event_history`, `magnitude_inputs`, `magnitude_history_store`) accept the `computed_moves.v4` or `computed_moves.v5` contract and refuse any other `CONTRACT_MISMATCH`; on v5 a row with a `not_computed_reason` other than `no_price_window` is `skipped`, carries no outcome and never counts as a scoreable event. The omission lists (`omitted_history`, `omitted_labels`) carry the writer's `not_computed_reason` as `reason` (v4 rows and v5 rows without one keep `skipped`; the magnitude search's `validate_targets` accepts exactly `skipped` and the writer's closed reason set, any other is `EXPERIMENT_IDENTITY_CONFLICT`), never a research-side re-derivation; non-skipped rows omitted for availability keep `unavailable` / `not_available_before_fit`. The isolation guard of the history and magnitude readers (`_plans` per-candidate checks, shared with `plan_with_reader`) is unchanged: every candidate needs exactly one isolated fragment, else `HOLDOUT_ACCESS_DENIED`. `prediction_inputs` does not use that guard: it refuses a table whose contract is neither native contract with `CONTRACT_MISMATCH` before any target scan, and a v5 skipped row with `EXPERIMENT_VARIANT_FAILED` naming the reason (a reason-less skipped row is malformed and gets the generic target message). It returns the signed move, binary `positive_move` target and `target_available_on` date (required and later than the event); fold consumers admit training labels only when available before their decision cutoff.

| Prediction/population input condition | Outcome |
|---|---|
| Snapshot cannot be resolved or its identity does not match | Non-retryable `SNAPSHOT_UNRESOLVED`; no source fallback. |
| Integrity or object failure during a table scan | Repository's typed refusal, such as `MANIFEST_CORRUPT` or `OBJECT_CORRUPT`; no partial returned frame. |
| Missing table or malformed table contract | Repository's typed refusal; no partial returned frame. |
| Invalid holdout context, unknown requested ID, excluded/ambiguous requested event, or empty explicit population (R5) | `HOLDOUT_ACCESS_DENIED` before target scans; no metrics or report. |
| Well-formed `computed_moves` contract that is neither native v4 nor v5 | `CONTRACT_MISMATCH` before any target scan; no partial returned frame. |
| Empty bulk canonical calendar | `POPULATION_COLLAPSED`; no empty success. |
| Missing, skipped, non-finite target, non-date outcome key or invalid target availability (R4) | `EXPERIMENT_VARIANT_FAILED`; no partial returned frame. |
| `omit_not_computed=True` and a row is `skipped` | Event omitted, object still bound in `consumed_inputs`, listed in `omitted_labels`; other events returned. |
| `omit_not_computed=True` and every planned label is `skipped` | Empty frame whose attrs list every event in `omitted_labels`; scoring then refuses `EXPERIMENT_VARIANT_FAILED` ("require scored rows"), which the inner search records as a refused terminal. |
| `omit_not_computed` is not a `bool` | `CONTRACT_MISMATCH` before any repository access. |

These readers preserve no economic variant identity and cannot authorize the
separate user-only final holdout read.

`horizon_fit_inputs.prepare_horizon_inputs` assembles ordered selected-column training/prediction float arrays and freezes null-preserving input hashes under an explicit training-target column; reserved matrix keys refuse `ValueError` before event-ID/column iterables are materialized once. It reads no evaluation targets, preserves signed labels without a floor and returns detached arrays/evidence; the caller owns target/domain validation, source/membership admission and fitting.
`horizon_fitted.fitted_forecasts` shares the per-horizon fit loop over closed numeric routes with an explicit target column/domain, accepting raw fit inputs or prebuilt `HorizonInputs` from `build_fit_inputs` (consumed as assembled, not checked against positional inputs). The builder carries canonical `train_event_dates` only when the recipe declares `requires_event_dates`, so other recipe hashes are unchanged. Regression recipe/domain must match and magnitude routes require nonnegative outcomes; signed predictions need only be finite and nonnegative routes reject negative forecasts; no evaluation labels or repository are accepted. Only fit/forecast-validation exceptions become native `HorizonFitFailure` with the original `__cause__`; preparation/progress errors propagate; callbacks receive deep-copied evidence and are provisional. No partial predictions/evidence return, retries, writes or production entrypoint; the offline magnitude wrapper retains its legacy exception/strict-failure policy.
`event_history.load_event_history(repository, snapshot, *, event_id, as_of_month, decision_session, history_start, purpose="selection")` constructs prior-event features for one admitted target, with explicit calendar days satisfying `history_start <= decision_session <= target event day`; shared `load_population` excludes random, rolling and ambiguous history before outcome reads. Candidates are same-ticker events in `[history_start, min(target day, decision_session))`.

Every candidate query is planned before any outcome scan, using the same fragment planner as `Repository`; a candidate's entire surviving membership must be one inspected singleton with the exact ticker/day key, partition and time bounds. The resolved committed catalog/manifest identity is the metadata trust anchor (native producer inspection, not adversarially forged metadata). Caller rows, implicit latest and another experiment's outputs are not inputs. Mixed, missing or unprovable membership refuses `HOLDOUT_ACCESS_DENIED` before opening outcome objects; old whole-ticker layouts are refused, never migrated, and row predicates are not physical isolation. Preflight indexing and planner-ordering detail: `guides/research_architecture_design_notes.md`.

Skipped or null-availability historical labels are omitted. A valid availability
day must follow its event and be on or before the decision day; later labels are
omitted before numeric validation or arithmetic. Their admitted objects are
decoded; protected event objects have a no-read guarantee. Malformed
availability or invalid eligible values refuse `EXPERIMENT_VARIANT_FAILED`;
repository failures propagate. Pure
`panel_math.history_features` owns the arithmetic: empty history and undersized
EMAs retain `None`, never zero fill.
Results carry features/status, ordered consumed event/object identities and availability, omitted-history reasons, snapshot/dataset/month/membership/cutoff pins and a deterministic digest.
The `research_event_history.v2` identity also binds every decoded `read_inputs` object, including skipped/unavailable labels omitted from arithmetic.
No supervisor dispatch or provider access. `experiments.gate_report` consumes fold-local exploratory reporting; production and final authority are unchanged.

`iv_crush_inputs.load_iv_crush_targets(repository, snapshot, *, as_of_month, labels_before, purpose="selection", event_ids=None)` reads signed relative IV30 change from the exact `iv_crush_targets.v2` contract/hash (semantic version `2.0.0`).
`preflight_iv_crush_targets(..., as_of_month, event_ids)` requires explicit selection participants (`CONTRACT_MISMATCH` when absent) and returns metadata-only planned-object evidence, not a capability; the loader repeats admission.
These offline APIs do not publish sources, register experiments or participate in production entrypoints.

| IV-crush boundary | Outcome |
|---|---|
| Snapshot/population | Shared canonical snapshot, holdout-union and ambiguity admission precedes targets; unsupported/final purposes or malformed holdout context refuse `HOLDOUT_ACCESS_DENIED`; a missing/malformed label cutoff refuses `CONTRACT_MISMATCH`. Bulk reads omit events on/after the explicit calendar-day label cutoff; explicit requests containing one refuse `CONTRACT_MISMATCH`. An empty eligible set refuses `POPULATION_COLLAPSED`. |
| Public planner candidates | `plan_event_targets` requires ticker and naive midnight event_date columns. Only candidate shape/identity checks translate to `CONTRACT_MISMATCH` before repository access; every error raised while planning, including native `DataError` envelopes, propagates unchanged. |
| Physical admission | Every candidate passes public `event_history.plan_event_targets` (metadata-only singleton admission) before any target opens. Missing/mixed objects or unprovable exact ticker/date bounds refuse `HOLDOUT_ACCESS_DENIED`; nonmatching contract/hash refuses `CONTRACT_MISMATCH`. Every declared target object is capped at 1 MiB before opening (`RESULT_LIMIT_EXCEEDED`); bounded Repository scans also cap compressed and total uncompressed bytes. |
| Stored label | Exactly one matching ticker/date/session, boolean false skip and null reason are required. Canonical anchors span 1–5 calendar days: AMC requires pre ≤ event < post; BMO requires pre < event ≤ post. Availability equals post plus one calendar day, follows the event and precedes the cutoff before numeric use. These checks do not prove source-row adjacency or exchange sessions. Skipped, late or malformed rows refuse the whole read with `EXPERIMENT_VARIANT_FAILED`; finite signed values, including zero and values below -100, remain unchanged. |
| Provenance | `validate_iv_crush_receipt(preflight, fold, receipt, targets)` checks exact evaluation membership/order, finite signed rows, cutoff/context/object identities and the full receipt/hash without I/O; malformed evidence raises structural/numeric exceptions. Attributes bind dataset/contract ref, cutoff, month/membership pins, purpose, future exclusions and ordered fragment/object/content hashes plus caller source_id/input_hash; source marker/hash syntax is checked, not authenticated or recomputed. |
| Progress | Per "Progress logging"; refusal stages log the native code only (`report_refusals`). |
| Effects/failures | No source-quote reads, target rebuilding or experiment authority. Native repository integrity/resource refusals propagate unchanged. |

The reader accepts only the exact definition (no grandfathering), uses `event_history.POPULATION_CONTEXT_COLUMNS`, and supplies no source-receipt validation, forecast recipe or model.

`event_target_store.publish_event_targets` is a shared, source-free, opt-in publisher of explicit complete earnings metadata rows and a selected ticker/date target set, each with its exact contract (no production caller). Every target must match metadata, but metadata need not have a target; supplied aliases/conflicts and nullable fields are preserved. Missing/extra fields, null required values, duplicate/non-string keys, unsupported or structurally invalid native-decoder contracts, invalid caps, non-JSON values or cyclic provenance refuse `CONTRACT_MISMATCH` before hashing or publication; exceeded row/object/byte caps refuse `RESULT_LIMIT_EXCEEDED`; PyArrow/native conversion, inspection and catalog errors propagate with no row repair or dropped fields. Inputs are an asserted native source snapshot, nonempty JSON provenance (bound into the request hash, not authenticated), explicit positive per-table row/per-object/total-byte caps and an explicit output Repository; the caller owns admission and workspace isolation. A single native `commit_snapshot` transaction publishes only earnings metadata and the target table in reconstructed knowledge mode; exact reordered replay reuses the successful receipt without catalog changes; failure may leave unreferenced immutable objects but never a partial committed snapshot. It does no provider/source reads, target arithmetic, fitting, holdout policy, cache or retry. Writer batching and progress: `guides/research_architecture_design_notes.md`.

`snapshot_rows.iter_snapshot_rows` streams an exact committed snapshot/table projection with explicit key predicates as Arrow-native Python rows; `read_snapshot_rows` collects it without partial returns or pandas null/integer coercion.
Positive candidate-row, cumulative decoded Arrow-byte, per-object and batch-row caps (default1024, bounded by contract/population) are required; invalid caps refuse `CONTRACT_MISMATCH`, exceeded row/decoded-byte caps refuse `RESULT_LIMIT_EXCEEDED`; native scan/integrity refusals propagate. Byte caps are not Python heap limits.
Streaming rows remain provisional until exhaustion; abandoned streams must close. This layer releases each decoded batch before advancing; the collector retains its complete list. No outcome authority; callers own admission.

`magnitude_inputs.forecast_cutoff(event_date, session, horizon, calendar)` uses native `CalendarSessions`: five, three or one full intervening sessions backward from the shared last-pre-print close; Wednesday AMC at one uses Tuesday close, BMO uses Monday.
Invalid dates/sessions/horizons/calendars, projected anchors, required-session gaps or insufficient history refuse `CONTRACT_MISMATCH`.
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
Shared population admission precedes label access. Bulk reads
exclude events on/after `labels_before`; explicit requests containing one
refuse. All remaining objects pass `event_history`'s shared singleton preflight
before any is opened; mixed inherited snapshots refuse. Each scan uses the
`event_history.MAX_OUTCOME_OBJECT_BYTES` (1 MiB) ceiling: oversized stored bytes
refuse before object access and oversized summed Parquet row-group bytes before
row decoding (`RESULT_LIMIT_EXCEEDED`); this is not a population/heap cap. Label
availability must be strictly before `labels_before`; missing, skipped,
malformed or non-finite labels refuse the whole call, with zero a valid value.
Opt-in `omit_not_computed=True` (a `bool`, else `CONTRACT_MISMATCH`) omits, without parsing, a
row the writer marked `skipped` (v5 `not_computed_reason`). Omissions are listed in the frame
attrs and receipt as `omitted_labels` (`[{"event_id", "reason"}]`, sorted by
event ID, `reason` the row's `not_computed_reason` or `skipped`; the receipt
carries the key only when non-empty). Only `skipped=true` is omitted, never
null/late availability; every other refusal is unchanged; forecasts for omitted
events are made but not scored and `event_count` counts scored events. Earlier
events with late availability can be decoded to check their recorded date. Output rows carry
target/object identity; frame attributes bind recipe, label cutoff,
snapshot/dataset/month/membership versions and consumed inputs. Forecast cutoff
and label cutoff are distinct contracts; this seam neither fits a model nor
joins labels into features. It has no supervisor or final-read caller.

`magnitude_inputs.preflight_magnitude_targets(repository, snapshot, *,
as_of_month, event_ids)` admits an explicit selection population and proves
every candidate object's singleton isolation without opening outcomes; its
detached digest binds the population/context and planned object identities.
The preflight is evidence, not a capability: target loaders repeat their own
admission and physical checks and accept no caller-supplied scan plan.

`qualify_magnitude_training_targets(repository, snapshot, *, as_of_month,
labels_before, event_ids)` admits explicit candidates whose event dates precede
the fit boundary, with whole-population physical preflight before outcome access.
Skipped, unavailable and cutoff-late labels are distinct omissions before numeric
parsing; eligible labels require valid identity/availability and finite signed
values, keeping their absolute percentage-point magnitude without clipping or zero
fill. The training identity binds every candidate and decoded object (including
omissions), qualified IDs, omission reasons and the cutoff. Empty qualification
refuses `POPULATION_COLLAPSED`; malformed eligible labels `EXPERIMENT_VARIANT_FAILED`.
Protected/ambiguous identities and mixed objects refuse before outcome access;
late admitted objects may be decoded, protected objects are never opened.
Strict target identities are unchanged; these seams publish no evidence and compute no
historical feature aggregates.

`magnitude_features.load_magnitude_history(repository, snapshot, *, event_id, as_of_month, horizon, history_start, purpose="selection")` binds one canonical event's prior-history features to its full-session forecast ceiling. Exact snapshot and target holdout/alias admission precede calendar or outcomes. The calendar is the sorted union of canonical `daily_market.date` values from that snapshot (only the date column; no projected days or external override); missing/empty calendars refuse through the shared reader/`CALENDAR_UNAVAILABLE`, and malformed dates, invalid horizons or incomplete anchor/session windows refuse `CONTRACT_MISMATCH` before historical outcome access. The shared `forecast_cutoff` computes the ceiling and `load_event_history` uses that exact day, excluding protected/ambiguous/current/future outcomes before its physical singleton preflight; the adapter never reads the target label. Prior-label availability on the source-session day is admitted (as in `event_history`). Outputs bind canonical event/session, snapshot, calendar version, dataset/object identities, horizon/cutoff, target and history-feature recipes, all history reads/omissions and membership versions into one deterministic input digest. Empty history keeps explicit null features; non-finite derived features refuse `EXPERIMENT_VARIANT_FAILED`. The stored build uses `plan_magnitude_folds`, `event_history._assemble_history` and `_feature_recipe`, not this adapter, which has no fitting, target-join, CLI, provider, report, ledger or final-read caller, no retry, write or partial result, and only the observed-session cache (fold table "Cache, retry" row).

`magnitude_folds.plan_magnitude_folds(repository, snapshot, *, as_of_month, evaluation_years, training_start, event_ids=None)` plans metadata-only expanding annual evaluation folds from one admitted snapshot and its pinned calendar.
`plan_regression_folds` shares that population/calendar implementation but binds a validated explicit target descriptor into `regression_temporal_plan.v1`; legacy magnitude plans remain byte-identical. IV metadata planning uses its annual candidate folds and 5/3/1 clocks, not label qualification, native registration or nested search.
No outcomes or target-snapshot identity are introduced.
`magnitude_folds.magnitude_window(evaluation_years, training_start)` exposes the
same pure request validation to native preregistration and returns a detached
canonical window; invalid requests refuse `CONTRACT_MISMATCH` without reads.
`magnitude_folds.annual_fold(year, events)` is the public annual rule shared by
the native planner and `experiments.magnitude_search_plan` for admitted metadata.

| Shared annual-rule condition | Outcome |
|---|---|
| Admitted event metadata has evaluation events and strictly earlier candidates | Return detached year, conservative fit cutoff and training/evaluation IDs in input order; no I/O or label qualification. |
| No evaluation events or earlier candidates | Preserve the native `POPULATION_COLLAPSED` `DataError`; callers own boundary-specific translation. |

- Exact snapshot and random/rolling/ambiguous-union admission precede calendar reads; bulk membership includes `training_start` and excludes January 1 after the last evaluation year.
- Each event's 5/3/1 horizons share its event-year fold. Training candidates precede `min(year_start, all evaluation source-session ceilings)`; usable labels also need availability strictly before that boundary. Earlier evaluation events may train later folds, including the same issuer; annual membership need not cover a full year; no inner HPO split is defined.
- The digest binds request, target/fold rules, ordered canonical event/cutoff memberships, holdout versions, snapshot and metadata-source objects; equivalent explicit/bulk memberships give the same `plan_id`.
- No computed-move, label or history-feature access, fitting or registration/final-read authority.

| Magnitude fold condition | Outcome |
|---|---|
| Years not explicit ordered unique four-digit strings, or training start not canonical and before the first year | `CONTRACT_MISMATCH` before repository access. |
| Explicit event outside the declared window | `CONTRACT_MISMATCH` before calendar access. |
| Empty window or any year lacking evaluation events/training candidates | `POPULATION_COLLAPSED`; no empty success. |
| Missing calendar table, snapshot/admission or repository failure | Shared typed error propagates; no source fallback. |
| Empty calendar / malformed dates or incomplete session window | `CALENDAR_UNAVAILABLE` / `CONTRACT_MISMATCH`; no outcome access. |
| Catalog transactions | Short read-only SQLite transactions; no planner-owned write transaction or transaction spanning the full call. |
| Cache, retry, partial write, replay | No retry, write or partial return; identical pinned inputs give identical plans. The only cache: `daily_market` observed days are read once per pinned `Repository` (`pins_snapshots`) and snapshot id, living and dying with that repository (a second repository or snapshot, or an unpinned repository, reads again); a failed read caches nothing; output and `calendar_input` identities are unchanged. |

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
The resolved economics and canonical plan identity retain every recipe field, so
changing a threshold changes the consumed plan identity. The walker aggregates
net position P&L from each day's held-leg quotes in the pinned `option_chains`
snapshot, using the pinned `daily_market` calendar, `_pricing.FillModel` and the
alpha ladder; it adds no projected calendar, replacement contract or mark source.
A target or stop exits on its first observed crossing. If one day's alpha 0/1
aggregate mark range spans both thresholds, the stop is selected and
`ambiguous_exit` is true (daily marks cannot order the exits), with exit value
and P&L priced at the worst alpha (0.0) so it cannot beat the stop. No crossing
by N sessions uses the 7a fixed-day fallback; the expiry bound and typed refusal
still apply. Results are
`mark_based` net held-position P&L with exit reason/date, visited dates, source,
the declared recipe alpha, the alpha actually applied at exit
(`exit_fill_alpha`, equal to the recipe alpha except for an ambiguous stop) and
snapshot identity. `exit_report_frame` projects those
decisions into one row per trade with separate `mark_based_pnl`, `exit_reason`,
`exit_day`, ambiguity and provenance columns for report callers; it does not
write files or blend mark-based P&L with fill-validated results.

| Exit-walker failure condition | Outcome |
|---|---|
| Malformed position or leg record | Non-retryable `EXPERIMENT_VARIANT_FAILED`; whole call fails, no excluded trade or partial tuple. |
| Missing required leg mark or mark past held-leg expiry | Non-retryable `EXPERIMENT_VARIANT_FAILED`; whole call fails, no excluded trade or partial tuple. Failure details identify the trade and failing session. |
| Unusable quote reaching pricing | Non-retryable `EXPERIMENT_VARIANT_FAILED`; whole call fails, no excluded trade or partial tuple. |
| Insufficient calendar coverage | Non-retryable `EXPERIMENT_VARIANT_FAILED`; whole call fails, no excluded trade or partial tuple. |
| Invalid fractional debit basis or converted threshold | Same, with entry trade/session context. |

Other repository refusals propagate. Report/ledger publication is outside this seam.

Internal (not interface, despite the non-underscore package norm elsewhere):
`_scan.py` and `_snapshot.py` (see Dependencies — two independent
snapshot-read helpers), `_pricing.py` (except the names carved out above),
`_trades_revisions.py`, `_final_population.py`, `_final_evaluator.py`.

### Proposed authorized final read (#490)

Request preparation, the private ops authorization/spend primitive, the private metadata-only final population reader and the private frozen evaluator are implemented; no production authorization entrypoint, final-read caller or result retrieval exists. Agents and supervisors may prepare requests, never authorize them ([user decision](https://github.com/yshewchuk/investment-validation/issues/490#issuecomment-6072905454)); #445's union exclusion is preserved.

**Request.** Canonical versioned bytes use the existing `experiment_id` as the decision identity ([decisions](https://github.com/yshewchuk/investment-validation/issues/490#issuecomment-6101284544)) and bind the registered `variant_id`/registration reference, resolved plan/source/evaluation-recipe and frozen-model hashes, exact committed `snapshot_id`, `holdout_as_of_month` and both membership versions; no latest-head resolution, caller population, alternate arm, fitting, threshold search or evaluator callback is accepted. Membership stays in `foundation.experiment_holdouts` (ambiguous identities refused, random and rolling sets bound together); changing month, version, snapshot or winner cannot reset a decision's spend. `native_registration.prepare_final_holdout_request` verifies existing registration metadata only (never resolves a head, scans a population or reads results); the registration's optional `final_read` declaration seals recipe, completeness-bar and frozen-model references, and registrations without it refuse preparation. Request validity proves neither winner selection nor read eligibility: the later metadata-only preflight must verify the preregistered completeness artifact and refuse an unmet bar before any holdout outcome read.

**Population reader.** Private `_final_population.load_final_population(repository, request)` is the only final-read reader that returns separate held-out event-ID sets. It accepts a validated `FinalHoldoutRequest` (never a caller population) and returns a frozen `FinalPopulation` with sorted random and rolling IDs, overlap and ambiguity counts, and bound snapshot/month/version context. It reads only canonical `earnings_events` metadata with the loaders' shared membership/ambiguity rules, excludes ambiguous events from both sets, and does not inspect ops state or hold a grant. It is not reachable from loaders, training, selection or sweep.

| Population reader condition | Outcome |
|---|---|
| Invalid request object, unsupported membership version, or as-of month later than the current UTC month | Non-retryable `HOLDOUT_ACCESS_DENIED` before any scan. |
| Snapshot unresolved, or resolved identity differs from `request.snapshot_id` | `SNAPSHOT_UNRESOLVED`; no fallback. |
| `earnings_events` absent from the snapshot | `CONTRACT_MISMATCH`; no partial population. Present with no fragments, the scan yields no events and the empty-population row applies. |
| `resolve` raising a non-integrity `DataError` (such as not-found), `ValueError`, `TypeError` or `KeyError` | `SNAPSHOT_UNRESOLVED`. An integrity-category `DataError` (such as `MANIFEST_CORRUPT`) and any other exception propagate unchanged. Integrity failures during the scan propagate as the repository's typed refusal. |
| Both sets empty after ambiguity exclusion | `POPULATION_COLLAPSED`; never an empty success. Preregistered completeness remains the later preflight's refusal. |

**Evaluator.** Private `_final_evaluator.evaluate_final_holdout(repository, request, population)` takes only a validated `FinalHoldoutRequest` and the `FinalPopulation` that `load_final_population` returns; it has no other parameter, fit, alternate arm, threshold search or callback. It scores the committed `trades` rows of the pinned snapshot with this package's provenance whose `variant` equals `request.variant_id` (the frozen winner). It returns frozen `FinalMetrics` with the random and rolling results separately, each with event count, trade count, events without a trade, total and mean `ret` and win rate (`None` when a set has no trade), plus overlap and ambiguity counts; the sets are never averaged or merged. It never imports ops and checks no fence or revocation: the later leaf does that before invoking it.

| Evaluator condition | Outcome |
|---|---|
| `population` is not a `FinalPopulation` (event-ID list, training/selection/sweep frame), or differs from the one derived from `request` | `HOLDOUT_ACCESS_DENIED` before any trades read; an equal hand-built population is indistinguishable and accepted. |
| Invalid request, or snapshot mismatch | The population reader's refusal (`HOLDOUT_ACCESS_DENIED`, `SNAPSHOT_UNRESOLVED`), before any trades read. |
| Trades table has no fragments, or no winner trade falls in either set | `POPULATION_COLLAPSED`. |
| Retained winner trade's event date or ticker differs from its pinned canonical event (for example after a calendar correction that kept the event ID) | `CONTRACT_MISMATCH` listing the offending event IDs; never skipped or scored. |
| Non-finite `ret` | `CONTRACT_MISMATCH`; infinities are already refused at publication. |

**Authority and lifecycle.** Ops owns authorization, spend and state fencing
(`engine/v2/ops/ARCHITECTURE.md`, `ops.final_holdout`; currently test-consumed
only). The `tools/v2_final_holdout.py` leaf that would call this evaluator, and
result retrieval, are not built (design only:
`guides/final_holdout_authorized_read_design.md`). Training, selection, sweep and
the loaders accept no grant argument, final purpose or returned holdout frame.

## Inputs

Target research reads cover each full pinned partition, including null and
non-midnight observation times, subject to caller predicates and batch
filtering. The manifest population bound limits candidate rows, not RSS or
process memory: it is the recorded row-count sum of pinned fragments surviving
pruning ([data scan population rule](../data/ARCHITECTURE.md#invariants)).
Each scan uses that bound for its snapshot, contract and predicates; the partial
daily builder also enforces its explicit cumulative row bound. Empty selected
membership allows zero result rows with a positive batch size. A complete read
either returns its population or raises its typed refusal (no month/day retries
or null-overflow refusals).
Shared `_scan` reads report decoded rows against the candidate-row bound (an upper bound; filtered scans also report retained/skipped rows); no additional scan is used for telemetry.

Public `_chains.read_chains_for_years` delegates a nonempty iterable of years to `_snapshot.read_table`/`_scan.read_table` with the fixed chain projection, optional key predicates and `batch_filter`; it materializes retained rows and performs no experiment holdout admission. The callback runs on each batch's pandas frame before retention and returns narrowed rows or `None` (omitting it or `None` preserves existing reads). `_chains.load_chain_index` uses it to keep exactly the requested ticker/date pairs, one manifest year at a time: only the exact pair mask, never the cross product, defines retained rows; frames retain ordering and duplicates. This bounds retained unmatched rows, not process RSS.

- `--catalog` (sqlite path) and `--store-root` (`ArtifactStore` root): required by
  every CLI; opened once per run via `engine.v2.ops.bootstrap.open_catalog` /
  `engine.v2.foundation.ArtifactStore`.
- `--scope` (default `"shadow"`, `_scan.DEFAULT_SCOPE`/`_snapshot.DEFAULT_SCOPE`)
  and `--snapshot-id` (default `None`) resolve the *one* `SnapshotRef` the run
  reads. A CLI that resolves before its
  first table read passes that id downstream; later lookups use the explicit id,
  never the possibly advanced head.
- Tier-2 tables, read through `Repository.scan` against that snapshot:
  `option_chains` (ORATS EOD quotes: fill quality, replay chain reads),
  `option_daily` (Polygon traded bars: fill quality), `daily_market` (signal
  screen; also the replay/build-trades planning calendar), `trades`
  (build-trades' read-before-append, reconcile's read-before-prune) and
  `earnings_events` (the canonical event universe both trade-table tools filter to).
- Replay and build-trades derive `plan_events`' trading-day calendar from that
  snapshot's `daily_market` (`_pricing.trading_calendar_from_snapshot`, reading
  only `date` through `_scan.read_table`), never from a local file or
  `INVESTING_PLAN_ROOT`. A snapshot without it refuses `CONTRACT_MISMATCH`
  (see Failure semantics).
- `--since`, `--min-date`, `--years`, `--strategy` (repeatable, from
  `_pricing.STRUCTURES`): per-tool read/filter narrowing, never widening a read
  past the snapshot.
- `tools/v2_replay.py --alpha A` is repeatable: supplied values (finite, in
  `[0, 1]`) replace `ALPHA_GRID` in order, repeats stay separate requests, and
  each priced event produces one row per requested alpha; omitted, `_replay_run.run`
  uses `ALPHA_GRID`. The calendar still comes from the pinned snapshot.
- `--dry-run` (build-trades, reconcile-trades): compute the candidate changeset,
  skip `commit_generic_table_candidate`.

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
  outcome summary (`committed`, `outcome`, plus `emitted_revisions` for
  build-trades and `committed_snapshot_id` for reconcile-trades) — never a rewrite of the table version the run read.
- Every snapshot-reading output that carries data also carries its `snapshot_id`.

## Dependencies

`engine.v2.data` (`Repository`, `errors`, `generic_incremental`, `incremental_tables`, `bounded_snapshot`, `catalog`, `documents`, `objects`, `manifests`, `query`, `time_formats`, `computed_moves_table`, `computed_move_bounds`, `confirmed_event_metadata`, `event_history`, `iv_crush_table`, `price_history_query`, `price_history_table`), `engine.v2.contracts` and `engine.v2.contracts.data` (revision/coverage and data-query/snapshot/table contract types), `engine.v2.contracts.final_holdout` (`FinalHoldoutRequest`, for `_final_population`), `engine.v2.features` (`calendar_features`, `own_vol.compute_own_vol` (the G3 consumer edge), `panel_math`, `regime`), `engine.v2.models.training` (`FitInputs`, `fit_routes` for offline horizon fitting), `engine.v2.foundation` (`ArtifactStore`, `SystemClock`, canonical JSON/hash/document helpers, `format_timestamp`, `market_calendar`, `experiment_exits.valid_exit_recipe`, `experiment_holdouts.calendar_date` (canonical days for IV input/anchor/clock validation), `experiment_holdouts.ExperimentHoldouts` (panel membership-policy versions), plus
`final_holdout.request_from_document` and `typed.to_document` for `_final_population`); research itself owns `tail_history` and `iv_dynamics` (G1, G2).
Only lower-layer imports are allowed (`checks/layer_map.py`, `only_imports` unset):
no `engine.v2.scoring`, `engine.v2.evaluation`, `engine.v2.ledger` or sibling/higher
packages. Ops and research are both layer 7.0, so research cannot import ops.
`engine.v2.ops.bootstrap.open_catalog` belongs to the `tools/v2_*.py` CLI leaves,
which open `--catalog` and hand the connection to research (see Inputs).
Stored history publishes new objects/manifests through `data.catalog.commit_snapshot`;
this native write path advances only its dedicated context scope, never source heads.
`_pricing.py` copies pure primitives from legacy `engine.structures`,
`engine.fills` and `engine.calendar`; `polygon_fills.option_ticker` is copied
verbatim from `engine.data.sources.polygon.option_ticker`. Research has no legacy
adapter; v2 cannot import legacy without a declared adapter.

**Known duplication (issue #69):** `_scan.py` and `_snapshot.py` are two
independent snapshot-read modules. `_snapshot.read_table` delegates its scan to
`_scan.read_table`, forwarding caller `key_filter` predicates, and keeps its own
partition-key resolution, no-partition refusal and empty-result frame shape.

Callers: no `engine/` modules import this package (`checks/import_layers.py` parses only `engine*` importers, so the `tools/v2_*.py` CLI leaves and `experiments/` callers sit outside the layering hook). The `experiments/` consumers of each public seam are listed in `guides/research_architecture_design_notes.md` and their contracts live in `experiments/ARCHITECTURE.md`. `experiments/magnitude_search_plan.py` consumes `magnitude_folds.annual_fold` and owns native bounds/cutoff exclusions from the registration-pinned `binding["snapshot"]` (its seal and refusal rules are in `experiments/ARCHITECTURE.md`); the archived `experiments/common_v2.py` trade-read caller omits required holdout context and is refused.

### Stored magnitude-history materialization

`experiments.magnitude_history_store` derives the supported build/read context; the engine accepts digest-consistent contexts and materialization independently guards target identity/windows (it does not authenticate arbitrary rehashed calendars as builder-derived).

| Boundary | Contract |
| --- | --- |
| Protected reads | The complete eligible singleton candidate union is preflighted before any outcome opens; protected/ambiguous objects never open; admitted late labels may decode availability and are omitted before arithmetic. An admitted outcome is decoded once per build; no cache persists across builds; memory has no fixed byte cap. Outcomes are read through `event_history.plan_with_reader`: malformed candidates refuse `CONTRACT_MISMATCH`; membership that is not exactly the preflighted set, an unplanned event, or a mismatched day, ticker or request refuses `HOLDOUT_ACCESS_DENIED` before any outcome opens; each object is capped at 1 MiB (`MAX_OUTCOME_OBJECT_BYTES`, `RESULT_LIMIT_EXCEEDED`); at most 32 run streams are live. A scan failure propagates its native code and fails the build before any snapshot, receipt or head is published. |
| Catalog reads and progress | The CLI opens its `Repository` with `pin_snapshots=True`; canonical population scans follow the "Progress logging" invariant (native experiment CLIs retain JSON stdout). |
| Per-forecast features | Shared history arithmetic preserves availability and numeric-refusal order; consumed objects, omissions and availability accompany all eleven columns. Exclusion rosters stay transient: persisted lineage omits `holdout_exclusions` and the stored reader refuses it. Empty `n_prior` is zero; other missing values are null with `on_missing=null`; no runtime imputation or production v4 aggregates. |
| Output | Features-only `research_magnitude_history.v1` snapshot, separate from unchanged source: three ordered horizons per event-isolated object; primary key `(event_id, context_id, horizon_sessions)`, physical context field and context partition. |
| Publication | Inspected objects/manifests and success receipt commit atomically under an initially empty deterministic scope; failure admits no snapshot (complete unreferenced objects remain for normal cleanup). |
| Replay/conflict | A matching success receipt returns the original artifact without moving a head; a failed/conflict receipt for the same context, or conflicting receipts/heads, refuse `IDENTITY_CONFLICT`; nothing retries automatically. |
| Stored read | The supported entrypoint reconstructs context; the engine verifies exact receipt/output and complete object isolation, then all row metadata before feature-column decoding (checksum verification reads bytes); no historical outcomes or arithmetic; malformed reads return no partial rows. |
| Trust/authority | Native inspected catalog metadata is trusted, not attested against malicious catalog writers; no experiment/final-read authority; registration v1.0–v1.3 and the fixed baseline are unchanged. |
| Refusals | Missing or changed context/receipt `IDENTITY_CONFLICT`; unprovable isolation `HOLDOUT_ACCESS_DENIED`; bad stored schema/metadata/missingness `CONTRACT_MISMATCH`; non-string selections, malformed selection containers, unknown event/feature or duplicate feature `CONTRACT_MISMATCH` before any payload scan. |
| Source/publication failures | Unknown source/feature snapshot ID `SNAPSHOT_NOT_FOUND`; resolved source-pin mismatch during public build/read context derivation `SNAPSHOT_UNRESOLVED`; direct `materialize` source/context mismatch `IDENTITY_CONFLICT`; non-native computed-moves contract `CONTRACT_MISMATCH`; empty fold `POPULATION_COLLAPSED`; invalid eligible label/features `EXPERIMENT_VARIANT_FAILED`; competing head `SNAPSHOT_CONFLICT`; other native catalog/object/source-admission errors propagate. |

`magnitude_history_inputs.admit_stored` delegates to `magnitude_history_store.preflight_stored`: exact receipt/output, features-only contract and event isolation; no payloads or arithmetic.
Refusal codes are the table's Refusals row. Its unlayered caller owns context derivation/registration.

**Eligible history:** sole public build/read boundary `experiments.eligible_history` rejects producer overrides and fingerprints installed code/environment; backend `_materialize`/`_read`/`_iter_read` are trusted internal helpers, not admitted APIs or Python security. The backend `plan_targets` is a read-only planner (no materialization, no read capability) that SIZE target derivation calls to learn the retained set before `materialize`. SIZE's library-only wrapper consumes the public `iter_read` stream for bounded panel composition and validates it on stored reads and receipt replay. No production/nightly entrypoint invokes that wrapper; production wiring and data runs require separate authorization.

**SIZE panel feature sets.** Qualified-panel recipes extend `HMO` as `HMOS` (season heat; panel `research_magnitude_panel.v4`, request `v10`), `HMOSX` (groups G1 tail, G2 IV changes, G3 own vol, G4 calendar; panel and context `.v5`, request `v11`) and the derived `HMOSXL`/`HMOSXLP` (39 `log1p_<column>` and 28 pairwise `<a>_x_<b>` descriptors, `block: "derived"`; request `v12`). Every earlier recipe hash, `recipe()` JSON and request id is byte-identical; only derived recipes carry `base_subset: "HMOSX"` and a `derivation` marker; `panel_recipe(subset)` refuses a derived set `CONTRACT_MISMATCH` because nothing derived is stored. Panel values are computed once at build from data already in memory (no new source read, no raw read by the search) at each forecast clock's `source_session_ceiling`, by data date, never `computed_at`; shortage is null, never infinity; no clipping, imputation or winsorising. Sealed `.v4` panels rebuild with no compatibility reader; the registration validator accepts context v3, v4 and v5. `magnitude_panel_qualified` is not reachable from production or nightly entrypoints. Window, lag and column-level rules, lineage and original wording: `guides/research_architecture_design_notes.md`.

| Component | Contract |
| --- | --- |
| `season_heat.compute_season_heat`, `season_population`, `outcome_binding` | Source-free over caller rows; labels come from `load_population` under the context's `as_of_month` and purpose, so held-out events are never planned or opened; availability is anchor `post_close_date` + 1 calendar day; `plan_roster` opens no outcome and `read_labels` (once per build) does; excluded and counted in `season_heat_population.v1`: `missing_native_bounds`, `null_source_bounds`, `off_session_event_date`. Refusals: malformed shapes/keys/IDs/sessions/duplicates, invalid or duplicate bounds, session mismatch, skipped or mis-dated outcome under non-null bounds `CONTRACT_MISMATCH`; fragment or anchor binding mismatch, duplicate fragment table-wide `HOLDOUT_ACCESS_DENIED`; malformed availability, invalid `abs_move`, non-finite results `EXPERIMENT_VARIANT_FAILED`. Request `v10` requires the `HMOS` hash. |
| G1-G4 (`tail_history.compute_tail_history`, `iv_dynamics.compute_iv_dynamics`, `own_vol.compute_own_vol`, `calendar_features`) | Add 25 columns (72 dictionary fields, 73 model columns with `has_implied_quote`); float64 nullable (`on_missing=null`) except `cal_amc`, `cal_weekday`, `cal_month` (int64, non-null, `on_missing=refuse`) and nullable int64 `cal_days_since_prior`; the same-date event count is intentionally absent. Refusals: non-sequence, malformed or non-finite input, G1 negative move, G2/G3 dates not strictly ascending, G3 dates after the ceiling, session other than `AMC`/`BMO`, prior event date after the event, target event with no confirmed session, and non-positive or non-finite `close_adj` at or after `source_start` (earlier, in the price read) `CONTRACT_MISMATCH`; non-finite result from finite input `EXPERIMENT_VARIANT_FAILED`; X columns not admitted by `admit_columns(..., x_groups=True)` `HOLDOUT_ACCESS_DENIED`. Request `v11` requires the `HMOSX` hash and admits `H/HM/HMO/MO-/M/HMOS/HMOSX` in that order (candidate counts: `guides/size_feature_groups_design.md`). |
| Derived sets, request `v12`, `experiments.derived_rows.with_derived` | `v12` requires the `HMOSXLP` hash (any other `INVALID_EXPERIMENT_SPEC`) and admits `H/HM/HMO/MO-/M/HMOS/HMOSX/HMOSXL/HMOSXLP`; `v11` and earlier refuse `HMOSXL*` as an unknown set; the 1000-candidate ceiling applies to set count times `G`. `size_search_binding` checks the dictionary subset (`v12`: `HMOSXLP`) but reads the stored panel with the stored-panel subset (`v12`: `HMOSX`), so the stored fold input and sealed panel are the same for every arm. `with_derived` is the only fit-path caller of `derive_columns` (in `load_fit_inputs` and `fit_evidence`), acts only when `features` name a derived column, keeps nulls null and copies rows; a negative or non-null infinite source or missing source column refuses `CONTRACT_MISMATCH`, an overflowing product `EXPERIMENT_VARIANT_FAILED` (the fit child reports a `failed` event). |
| R1-R6 | R1 at most 61 closes, 21 surface rows and 252 label buckets per target, no per-target copies; R4 fragments are written before the snapshot commit, so failed publication leaves unreferenced objects and no partial snapshot; R2/R3/R5/R6 as in Failure semantics (equal pinned inputs give equal values, row ids and context id). |

The separate `research_eligible_history.v1` table binds exact source/bounds, policy/month/membership versions, history start, a nonempty ascending subset of the canonical 1/3/5 clocks, contributors and producer identity; exact receipts replay and live/v1 aggregates never substitute. Full canonical metadata establishes protected windows. Once the selected target/history population is known, missing required ticker/event bounds refuse `CONTRACT_MISMATCH` before outcomes (unrelated tickers need none); null historical bounds are omissions. All selection targets, including annual evaluation, require retained eligibility. `eligible_history_table.admit_fields` accepts only the exact research contract/recipe/policy and known H11 fields; live/unknown provenance and unsupported aliases/derivations refuse before feature reads. Limits cap decoded source rows, fragments and physical source bytes after native catalog resolution (output byte caps apply after per-target serialization, not peak allocations). Refusals: malformed dates `CONTRACT_MISMATCH`; fields/provenance/producer overrides, unisolated outcomes, excluded targets and wrong artifact `HOLDOUT_ACCESS_DENIED`; limits `RESULT_LIMIT_EXCEEDED`; changed receipt `IDENTITY_CONFLICT`. Protected outcomes stay unopened; failures publish no receipt (unreferenced objects use normal cleanup); no production/provider/fit/final-read authority.

`magnitude_history_store.read_stored_for_events` is the projected search reader.
Every call revalidates the complete receipt, isolation, metadata and feature content,
including events and columns not retained. Complete metadata admission precedes
numeric validation; corruption or inconsistent source content refuses the read.
Returned `inputs_id` names the exact original `read_stored` full-source envelope,
not the projection. Selected rows contain only primary-key identity and requested
numeric features, without lineage strings. Memory scales with metadata and selected rows, with no fixed overall byte cap.
The original full-row `read_stored` API and its returned envelope remain unchanged.

## Opt-in eligibility policy boundary

`eligibility_policy.eligibility_policy` declares an explicit supported version and
nonnegative integer calendar-day width. `plan_eligibility` is metadata-only and
opt-in. The policy identity (including its `preserve_membership` label, kept for replay) and plans without an ambiguous random event are unchanged; with one, evaluation IDs on that ticker are filtered by the policy. The policy does not apply the strict label cutoff; the SIZE fold binder retains only fold IDs surviving its ticker embargo and cutoff and records exclusions. Eligible-history planning passes no evaluation IDs and refuses development targets on such a ticker, so only the SIZE fold binder's defensive check filters fold IDs. It uses `experiment_population.canonical_events`
for exact committed-snapshot admission and the six-column metadata projection and `experiment_trades.classify_events`
for detached membership/structural ambiguity labels and caller-pinned context.
Shared `snapshot_inputs` validates reference shape and exact committed identity;
the pure classifier checks shape only, without claiming repository commitment.

The closed producer envelope binds snapshot/month/policy and event groups with
explicit label bounds and every declared horizon's decision/source ceiling.
Bounds contain the event, with source ceiling <= decision <= event. These are
producer declarations, not proof of price-bracketing or publication times.
No calendar guessing, outcome/feature decoding, write, registration or fit occurs.

Random-held windows merge per ticker. Inclusive event/label or forecast-point overlap
excludes a whole development group; shared random, rolling and ambiguous labels do too.
The plan ID binds groups/policy/metadata, windows, retained IDs, development reasons,
coverage, ambiguous IDs and ticker-filtered evaluation IDs/reasons. Inputs are detached.
Progress follows the "Progress logging" invariant.

| Condition | Outcome |
| --- | --- |
| Unknown, malformed or mismatched snapshot reference | `SNAPSHOT_UNRESOLVED` |
| Repository integrity failure other than unknown snapshot | Original native error propagates |
| Unsupported policy, stale digest/envelope, malformed bounds, duplicate declared group IDs or malformed/duplicate evaluation IDs | `CONTRACT_MISMATCH` |
| Missing/duplicate classifier columns, malformed event IDs or metadata values that prevent classification | `CONTRACT_MISMATCH` |
| Invalid scalar dates/sessions or structural collisions handled by membership classification | Successful classification with `ambiguous` membership and IDs; a random member with a missing, blank, padded or non-string ticker, or without usable event/canonical dates, refuses with `HOLDOUT_ACCESS_DENIED` |
| Invalid as-of month, unknown group/evaluation identity or colliding canonical population IDs | `HOLDOUT_ACCESS_DENIED` |
| Structurally ambiguous random event with usable identity | Keep membership unchanged, record its ID and add an inclusive `date.min`–`date.max` ticker window; exclude matching development groups and evaluation IDs, with `embargo_ticker` as the evaluation reason; all other IDs remain in `evaluation_event_ids` |
| Ordinary random-event window arithmetic exceeds the supported date range | `HOLDOUT_ACCESS_DENIED` |
| Valid protected/overlapping development group | Excluded whole event with development reasons |

This is neither a physical-read capability nor a statistical isolation guarantee.

## External systems and libraries

Libraries: pandas/numpy for tables/numerics, pyarrow for stored-history Parquet.
No network calls or third-party service; Polygon/ORATS pulls stay in legacy
`engine/data/pulls/`. Snapshot access uses `ArtifactStore` and the supplied sqlite
catalog connection. Two local-file exceptions are outside the snapshot store:

- `fill_quality.write_report`, `signal_screen.write_report`, `polygon_fills.run`,
  `_replay_run.run`, `_build_run.run` and `reconcile_trades.run` write
  Markdown/parquet/JSON under caller-provided `reports_dir`/`out_dir`; fill
  quality's optional CSV uses the independently supplied `csv` destination.
- `_pricing.trading_calendar()` reads the local legacy S&P 500 daily CSV, only
  for legacy-parity tests and direct library callers; no other research module
  calls it, and replay/build-trades never do (see Invariants).

## Failure semantics

The 4c R1–R6 template covers data-boundary refusals, generally `DataError`
(`engine.v2.data.errors.DataError`) carrying a registered `DATA_FAILURE_CODES` code.
CLI leaves catch these at `main()` and exit 2: stored-history runtime refusals
are JSON on stdout; other tools print code/message on stderr.
One pre-existing data-path exception, called out below
where it applies (`_snapshot.py`'s bare `ValueError`, issue #70), escapes as
an uncaught traceback instead.

| Shared row-scan condition | Outcome |
| --- | --- |
| `read_chains_for_years` receives an empty year iterable | Raise `ValueError("option_chains: years must not be empty")` before repository access or scan telemetry; never expand to all years. |
| `read_chains_for_years` receives scalar str/bytes | Raise `ValueError("option_chains: years must be a non-string iterable")` before repository access; string elements inside an iterable remain supported. |
| Scan iteration or batch filtering raises | Attempt the failed boundary on stderr with elapsed time/RSS; propagate the original exception unchanged, with no partial frame or retry. |
| Default stderr emission fails | Suppress the telemetry error; preserve the scan's rows or original failure. |

- **CLI argument parsing (before R1–R6).** A replay `--alpha` that is non-finite
  or outside `[0, 1]`: the parser exits 2 before `open_catalog`/`ArtifactStore`,
  so no replay runs.
- **R1, missing input.**
  - **No committed scope head and no `--snapshot-id`:** `resolve_pinned(scope)`
    (or `_snapshot.py`'s `head_generation`) raises `SNAPSHOT_NOT_READY`
    (`dependency`, **retryable**).
  - **Explicit `--snapshot-id` that does not exist:** `Repository.resolve`/
    `resolve_full` raises `SNAPSHOT_NOT_FOUND` (`dependency`, not retryable);
    never substituted for the scope head.
  - **Manifest fails its identity/hash check** (dataset version, contract or
    fragment not matching its catalog identity, unknown object, scan out of
    declared primary-key order): `MANIFEST_CORRUPT` (`integrity`, not retryable).
  - **Table absent from the resolved snapshot** (`_scan.py`): `CONTRACT_MISMATCH`
    (`validation`, not retryable). `_scan.read_table` returns an *empty* frame,
    not a refusal, when a partition filter matches no fragment records. Replay/
    build-trades get the same refusal for a snapshot without `daily_market`
    (see Inputs), never a fallback to `trading_calendar()`'s local CSV.
  - **`daily_market` present but with no valid dates** (zero fragments or all
    unparseable): `_pricing._calendar_from_dates` raises `CALENDAR_UNAVAILABLE`
    (`validation`, not retryable), distinct from the table-absent case.
    `plan_events` (`_plan.py`) raises it too when reached with `calendar=None`
    (it has no repository; no CSV fallback). Exit 2 with a code, never an
    uncaught `ValueError`.
  - **No declared partition column, or no partition values (no fragments):**
    `_snapshot.read_table` cannot build a `KeyPredicate` and raises bare
    `ValueError`, not `DataError`. Omitted or empty `partition_keys=[]` derives
    values from fragments and does not cause this; valid nonmatching keys return
    an empty frame. Affects `_replay_run.py`, `replay.py`, `_chains.py` and
    `_trades_publish.py`, not `_scan.py`; their CLIs (`v2_replay.py`,
    `v2_build_trades.py`, `v2_reconcile_trades.py`) catch only `DataError`, so it
    escapes as an uncaught traceback.
  - **No overlap rows after a join/filter** (fill quality's `since` join;
    `load_trades`'s strategy/provenance filter): `POPULATION_COLLAPSED`
    (`validation`, not retryable); never a zero-row result.
  - **No `trades` table in the snapshot** (`load_trades`, via
    `_trades_publish.read_existing_trades`): `CONTRACT_MISMATCH`, as in
    `tools/v2_build_trades.py`'s read-before-append. A zero-fragment
    `earnings_events` table gives the same code (the issue #70 `ValueError`,
    converted). Missing or ambiguous event joins are excluded under the holdout
    population contract above.
- **R2, cache.** None: every run resolves its snapshot and reads tables fresh
  through `Repository.scan`; nothing is cached across runs or processes. Within a
  run the resolved `SnapshotRef` and, in replay, the loaded `ChainIndex` are held
  in memory (a pinned snapshot never changes, so nothing invalidates). The one
  exception is the magnitude observed-session day tuple, held per pinned
  `Repository` and snapshot id.
- **R3, retry.** No scan automatically retries. `SNAPSHOT_NOT_READY`,
  `SNAPSHOT_CONFLICT` and propagated `RESOURCE_UNAVAILABLE` are retryable;
  limit and integrity refusals, including `RESULT_LIMIT_EXCEEDED` and
  `MANIFEST_CORRUPT`, are not.
- **R4, transaction.** Build-trades and reconcile-trades are the only writers of
  the `trades` table (stored-history materialization separately commits its own
  scope through `data.catalog.commit_snapshot`, see its Publication row); both
  use `_trades_publish.publish` →
  `generic_incremental.build_generic_table_candidate`/
  `commit_generic_table_candidate`, fenced to the snapshot the run read
  (`expected_head_snapshot_id`, `expected_head_generation`). If another writer
  advanced the head meanwhile, the commit refuses `SNAPSHOT_CONFLICT`
  (`dependency`, retryable), never overwriting or merging. `--dry-run` builds the
  same candidate and stops before the commit: no snapshot is committed, though
  building may leave unreferenced immutable objects in the store.
- **R5, partial result/write.** Read batches stay provisional until scan
  exhaustion; a later failure returns no partial frame. A smaller explicit caller
  limit raises `RESULT_LIMIT_EXCEEDED`. Writes use the atomic
  `commit_generic_table_candidate`: a pre-call crash leaves the parent untouched.
- **R6, idempotency.** A byte-identical read of the same pinned snapshot and
  request returns identical rows, so read-only tools give identical report rows; file names carry a date
  stamp (default today), and replay's JSON also embeds a `generated_at` UTC
  stamp, so its bytes differ per run. Build-trades and
  reconcile-trades are NOT idempotent: each commit publishes a new `trades`
  version. An explicit `--snapshot-id` stays pinned to its parent, so a rerun
  after a successful commit refuses `SNAPSHOT_CONFLICT` (stale parent); omitting
  it resolves the current head, which reads the newer version and may produce a
  different non-empty changeset (or `noop` when replay recomputes revisions
  already published). The commit is not re-entrant or
  retried internally; "publish this exact changeset once" relies on the
  `SNAPSHOT_CONFLICT` fence.

## Invariants

- Every run reads one selected snapshot: CLI-resolved IDs remain explicit in
  downstream lookups, every read and output, including experiment context columns
  (see Inputs/Outputs; root §5's "no silent default" invariant).
- Every experiment runner reads from its pinned v2 snapshot, applying shared random/rolling holdout enforcement through the reader appropriate to its inputs:
  `experiment_trades.load_trades` for trades; `experiment_population.load_population` for populations and its prediction/history consumers. Inadmissible reads refuse.
  Direct parquet/file reads bypass snapshot pinning and both holdout sets and are inadmissible.
- Never reads the legacy mutable Tier-2 store (`engine.data.store`): every table
  read is a bounded `Repository.scan` against the one resolved snapshot.
- Every scan carries its own partition predicate alongside caller key
  predicates, so result rows cannot duplicate across partitions.
- No production entrypoint derives calendars from local files/environment
  variables: `_replay_run.run` and `_build_run.run` pass the pinned snapshot's
  `daily_market` calendar into `replay()`, so CSV changes cannot affect
  replay/build-trades output (refusals: Inputs, Failure semantics).
- **Year-scoped rebuilds cannot tombstone other years.** `_build_run.run` passes
  the same `years` to `_trades_revisions.filter_events` and `revisions_for_rebuild`.
  A tombstone requires replay provenance, membership in `rebuilt_strategies`, a
  `trade_id` absent from the rebuild, and (when `years` is supplied) the existing
  row's own `year` in that scope. Omitting `--years` tombstones across all years.
- **Progress logging.** Stages that use the progress helper (not every planner) log start/end/failure with elapsed time and peak RSS to stderr, with count/rate/ETA ticks at most every 30 seconds and aggregate decisions; refusal stages log the native `DataError` code only, never message text, row values, identities, details or secrets. Logging cannot change results, errors, stdout or read authority.
- **Availability evidence.** Availability dates, forecast cutoffs and schedule knowledge are reconstructed bounds from stored data, never evidence of publication, finality or as-known event schedules, for every reader, planner and stored feature here.
- **Read-only by default.** Unless a section states otherwise (stored-history materialization, `event_target_store`, build-trades/reconcile-trades), readers, planners and validators write nothing, retry nothing, cache nothing across calls and return no partial result; native refusals propagate unchanged and identical pinned inputs give identical outputs.
- Never mutates the legacy trades ledger and never calls a network provider
  (`engine/build_trades.py` and `engine/data/pulls` keep doing both).
- A trade-table write always goes through `generic_incremental`'s fenced commit,
  never a read-modify-write of an existing table version.
- Pure pricing/planning/chain cores (`replay.py`, `_plan.py`, `_chains.py`, `_trades_table.py`) retain legacy arithmetic; only store edges changed. Synthetic legacy parity remains covered by the research tools/replay/build-trades/reconcile-trades tests.

## Spline ElasticNet recipe

`engine/v2/models/magnitude_elasticnet_spline_recipe` is closed metadata only (no sklearn, I/O or fitting). `SPLINE_COLUMNS` is the single literal list of 34 raw source columns; a test asserts it is a subset of the L-list sources.
`elasticnet_spline_recipe(columns, feature_dictionary_hash, alpha, l1_ratio, n_knots)` returns schema `magnitude_elasticnet_spline_recipe.v1`: the plain ElasticNet recipe with `precompute: True` (Gram solver, needed at about 500 inputs) and a `preprocessing.spline` block (`columns`, `n_knots`, `degree: 3`, `knot_rule: quantile`, `extrapolation: constant`, `include_bias: false`, `input: imputed`, `output: append`). `output_features` adds `34 * (n_knots + 1)`.
`validate_elasticnet_spline_recipe` rebuilds from `(features, hash, alpha, l1_ratio, n_knots)` and refuses any drift with `ValueError`. `n_knots` must be a plain int in 4..8, and every spline column must be among `features`. Learned knots are fit outputs, never identity.
`experiments.magnitude_candidate_recipe.candidate_recipe` accepts the elasticnet key `spline_knots` (exact key sets): `0` returns the unchanged `elasticnet_recipe` (byte-identical JSON, `precompute: False`); 4..8 returns the spline recipe. Request-version gating belongs to the request validator.

## Source-free regression evidence

`regression_evidence.normalize_forecasts` validates exact supplied event/horizon coverage, explicit prediction-column names and signed/nonnegative domains, normalizing complete finite Real/Integral rows to detached ordered JSON floats/ints (strict JSON mode rejects non-native numeric types; Booleans, duplicates, extra fields and nonfinite values refuse). `error_metrics` pools finite residuals at equal row weight (scaling before squaring) and returns count/RMSE/MAE; empty, nonnumeric or nonfinite residuals refuse. Both raise numeric/structural exceptions without partial results, I/O, fitting, cache, retries or publication; membership/domain arguments are caller declarations, not sealed target identity, source admission or holdout authority, and the caller owns horizon coverage and residual subtraction. Existing experiment magnitude normalization delegates with fixed nonnegative domain/column and keeps its typed failure envelope.

## Diagram


CLI data-flow diagram: `guides/research_architecture_design_notes.md`.

### Qualified SIZE panel composition

`experiments.size_panel` is the installed public boundary over internal composition/storage helpers. Its v3 dictionary binds eligible history without changing v1/v2 identities. Complete metadata, independent quote groups and daily/runup/^GSPC inputs preserve projection-independent source admission. Native clocks/outcome guards, exact versioned anchors and ticker/year/date fragment membership bind lineage; date-only quote windows do not attest computed-outcome capture. Native immutable catalog/Parquet storage enforces physical/expanded limits and exact bounded receipt/context replay. Fitting authority belongs to the execution/fold consumer. Quote eligibility records forecast clocks only for the horizon subset selected in the panel request.

| Qualified panel admission condition | Outcome before dependent reads |
| --- | --- |
| Request/admission refusals | Malformed request or empty required inventory: `CONTRACT_MISMATCH`; caller producer override or incompatible frozen SIZE policy: `HOLDOUT_ACCESS_DENIED` |
| Derived set passed as a stored-panel recipe | `panel_recipe("HMOSXL"/"HMOSXLP")`: `CONTRACT_MISMATCH`; `recipe()` itself declares columns and reads nothing |
| HMOSX feature-group inputs | Refusals are those listed in the "G1-G4" row of the SIZE panel feature sets table; legitimate shortage is null, never a refusal |
| Memory/bounds | Peak memory is bounded by explicit query/source limits plus selected targets, the unique contributor union and the active ticker's source window (no RSS cap claimed); stored-read batches are capped at the largest stored fragment's `row_count`, in no identity. Stored reads and receipt replays validate complete history coverage first (`CONTRACT_MISMATCH`); publication exposes no partial snapshot. Context `magnitude_panel_context.v5` carries `season_heat_population` (roster capped by `max_output_rows`, `RESULT_LIMIT_EXCEEDED`; calendar-source mismatch `IDENTITY_CONFLICT`) and `price_row_digests` instead of v3's inline `price_row_metadata`; a value build whose re-read count or digest differs refuses `IDENTITY_CONFLICT`, so any changed price-version row changes the `context_id`. The stored-panel read streams one scanner batch at a time; `read(..., lineage=False)` omits only `lineage` (default `True`, in no identity); a corrupt row refuses when its batch is reached with no rows returned. Anchor-cache and retention detail: `guides/research_architecture_design_notes.md`. |
| Non-native snapshot reference shape | `SNAPSHOT_UNRESOLVED` |
| Confirmed event dated strictly before its ticker's first non-deleted price date | No label bracket can exist, so `quote_plan` (returns `(plan, excluded)`) leaves it out of the quote groups with reason `before_price_coverage` instead of refusing. It is then absent from the retained quote ids, so implied-history marks it ineligible. The panel context carries `quote_precoverage_excluded` (`reason`, `count`, sorted `event_ids`) only when non-empty, so other context ids are unchanged. Any other bracket miss (inside the coverage range, after the last price date, or a `BMO` event on the first date, which has no prior session) still refuses `CONTRACT_MISMATCH`; no substitute bracket or retry. |
