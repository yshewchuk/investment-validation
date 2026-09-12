# System rearchitecture: one scoring contract, incremental data, reliable operation

Date: 2026-09-12. Status: proposed implementation and migration guide.
Review baseline: commit `68150a1`, with a clean worktree before this guide.

This proposal follows a source review of the current engine, dashboard,
storage, experiment harness, checks, and recent failure reports. It is not a
new backtest, a performance benchmark, or a certification that existing
strategies are ready for capital. No production configuration, model, strategy,
data, or ledger was changed for this review.

## 1. Recommendation

Keep one Python codebase with explicit domain boundaries, a thin FastAPI
application, and separately supervised worker processes. Keep the large data
in Parquet. Add a small SQLite catalog for transactions, ingestion progress,
job admission, immutable release manifests, and indexed serving projections.
Build a React/TypeScript interface that loads those projections on demand.

The central object should be an immutable **score record**: the resolved event,
exact strategy definition, decision clock, input snapshot, models, selected
contracts, forecasts, decision, financial diagnostics, and reasons. The board,
an experiment, a replay, and a live request must all obtain it from the same
scoring application. Neither a renderer nor an experiment runner should
reconstruct its financial meaning.

Three architectural changes do most of the work:

1. Turn a mutable collection of files into versioned datasets and validated,
   atomic releases. Readers keep using the last good release while new work
   happens in staging.
2. Turn implicit scoring dependencies into registered strategy, feature, and
   model recipes. Use those recipes for both historical and current inputs.
3. Turn independently launched scripts into resumable jobs admitted by one
   resource manager. CPU placement, memory reservations, and provider quotas
   become centrally managed policies.

The migration must preserve current economic definitions and model behavior
before adding new capabilities. A better result after a refactor is a
discrepancy to explain, not evidence that the refactor worked.

For review, start with sections 3-4 for preservation and ownership, sections
5-8 for storage/scoring/operations, then the separate
[component contracts](component_contracts.md) for concrete interfaces. The
[data model diagrams](rearchitecture_data_model.md) show entity identities and
relationships; the [structure generation and simulation contracts](structure_generation_and_simulation.md)
detail the reusable numerical components.

### Relationship to existing guides

Read this alongside [the program plan](../EARNINGS_VOL_PROGRAM_PLAN.md),
[serving parity](prompt_serving_parity.md),
[execution clocks](prompt_execution_clock.md), and
[live intraday scoring](live_intraday_scoring.md).
Some older guides describe defects as open that current code has fixed;
current source and pinned artifacts determine the migration baseline.

This proposal deliberately revises the architecture choices in
[guides/README.md](README.md), convention 7, which previously excluded
databases and frontend build tools. SQLite transactions and a frontend build
are justified by the requested capabilities. This document proposes those
changes; it does not change the current operating rules or authorize a
deployment, paid subscription change, or strategy promotion. Existing research,
fill, reporting, privacy, and ledger safeguards continue to apply.

## 2. What the review found

| Current implementation | Implication for the redesign |
|---|---|
| [score.py](../engine/score.py), `Scorer.score`, already orders forecasts, structure selection, pricing, features, model/analog/gate layers, and chooser inputs. It is about 3,400 lines. | Extract and register its existing stages. Replacing it with a new set of formulas would create another parity problem. |
| [render.py](../engine/dashboard/render.py), `compact_row`, `_model_fair_pct`, and `payoff_curve`, calculate premium ratios, fair values, and payoff geometry. | The browser has little financial arithmetic, but the renderer still owns financial logic. Move these calculations behind the scoring domain too. |
| [app.js](../engine/dashboard/static/assets/app.js) already loads ticker details and model evidence lazily. The board and other payloads remain generated JSON/JavaScript, with an all-in-one export. | Extend existing lazy loading to events, score details, evidence, and books. Preserve offline access as a separate export mode. |
| [earnings_app.py](../dashboard/earnings_app.py) reads bundle files; ad-hoc scoring constructs/uses a scorer; refresh runs a subprocess synchronously with a long timeout. | Reading a board should stay cheap. Expensive work needs a durable job receipt and must run outside an HTTP worker. |
| [store.py](../engine/data/store.py) atomically replaces individual files, but `write_table(replace=True)` and `PartitionedWriter(replace=True)` drop a table before rewriting it. | File atomicity does not make a multi-table rebuild atomic. An interrupted build can expose an incomplete dataset. |
| [rebuild.py](../engine/data/rebuild.py) scans normalizer inputs and rebuilds tables; Tier 3 is written as a whole panel. Tier 4 already supports `since` and causal folds. | Preserve the working incremental Tier-4 semantics and extend change tracking upstream; do not describe the existing store as wholly unversioned or wholly incremental. |
| [features.py](../engine/features.py), `FeatureContext.load`, reads daily partitions before filtering tickers; the scorer also carries historical trades and model state. | Bounded inputs must be planned before reading, with distinct scopes for event features and historical evidence. |
| [bounded_run.py](../tools/bounded_run.py) has `--cpu-set`, thread limits, and a process-tree watchdog. Its environment note records unavailable writable cgroups/systemd. | Keep it as a fallback executor. It is not a global scheduler, and its watchdog is not a kernel memory limit. |
| [nightly.py](../engine/dashboard/nightly.py) resolves finality, scores, writes the ledger, settles, renders, then self-checks before publishing. | Publication is guarded, but the final score replay check occurs after predictions have been frozen. Validate decisions before committing them. |
| [finality.py](../engine/data/finality.py) now requires positive session evidence and records coverage. Decision-aware serving is also present in current `score.py`. | These are existing protections to preserve and strengthen, not missing modules to reinvent from older prompts. |
| [training/chooser.py](../engine/models/training/chooser.py) imports EXP-169 by path; that dataset depends on earlier experiment code. | Production depends on an experiment directory. Extract the exact promoted recipe into the engine and make old runners call it. |
| [model_evidence.py](../engine/dashboard/model_evidence.py) dispatches dataset construction by role/feature set and caches evidence by champion artifact hashes. | Dataset recipes and evidence belong in model releases. A feature list alone does not identify feature construction. |
| Recent commits fix forecast suppression on replay, analog order dependence, and precision loss through serialization. [Serving parity](prompt_serving_parity.md) documents additional context and feature-definition failures. | Test training-versus-serving, serving-versus-replay, serialization, and causal source selection separately. A passing digest cannot prove all four. |

The existing pricing, audit, evaluation, and recovery code is valuable. The
weakness is that it allows independently correct pieces to be assembled with
different inputs or meanings.

## 3. Freeze the compatibility contract first

### 3.1 Inventory of strategies and models

There are eleven structure factories and a separate DYN-SV selector. Preserve
all of them, including disabled and historical definitions.

| Strategy | Behavior that must survive the migration |
|---|---|
| `STR-THRU` | ATM call/put straddle; current first-post-event expiry selection; entry offset 0, exit offset +1 relative to the last pre-print close. Preserve its forecast/analog gate and exact registered threshold. |
| `STR-RUNUP` | ATM straddle; entry offset -14, exit 0; current first expiry meeting the target DTE, default 30. Preserve its own gate, implied-move/runup models, and payoff surface. |
| `CAL-P` | Simultaneous short front put/long back put calendar, both closed after the event. Preserve exact selectors and parameters. Remain disabled with `UNVALIDATED_STRUCTURE`. |
| `CND-P` | Preserve its existing long put-condor factory and mechanics. Remain disabled pending the existing evidence/promotion requirements. |
| `TWIN-P` | Seven-strike twin peak; current forecast-to-width divisor 1.5; preserve the existing shared expected-PnL rule and its stated evidence limitation. |
| `TWIN-P5` | Five-strike twin peak; forecast-to-width divisor 1.0, not 1.5; preserve its registered rule. |
| `CND-PS`, `BFLY-P`, `BFLY-P5`, `RAMP7`, `CTR5` | Preserve factory geometry, quantities, anchors, and forecast width divisors 2, 1, 3, 3, 2 respectively. Preserve forward-tracking versus promoted status. |
| `DYN-SV` | Preserve the ordered seven-member menu: TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5. Rank using the existing chooser where available, preserve its partial/missing-score behavior and fallback resolver, and inherit the selected structure gate without re-gating. |

The current arithmetic short-vol rules include the trailing six-month top-20%
simulated-return bar, 25% relative-spread ceiling, and $10B market-cap floor.
These differ from the learned-gate domain floor. Do not consolidate apparently
similar constants. Width bounds, coarse-ladder refusals, strike tie resolution,
draw counts, residual pools, quote repairs, and fallback behavior are also
part of the definition.

The active model IDs at the review baseline are:

- `size_v1_4`, `opf_implied_t1_gbm`, `runup_move_d14_v1_gbm`, `iv_crush_v1_gbm`;
- `gate_midfill_str_runup`, `gate_midfill_str_thru_forecast_analog`;
- `dyn_sv_chooser_v1_1`.

Current champions have no separate decision offset registered. Preserve those
identities, weights, transforms, feature order, thresholds, folds, and execution
conventions. A D1 experiment existing on disk does not make it a live champion.
This table is an inventory; the frozen source and artifacts, not manually
transcribed prose, are the definitive specification.

### 3.2 Generate a baseline package

The first implementation deliverable should export an immutable compatibility
package containing:

- Resolved structure definitions, selectors, parameters, entry/exit/decision
  offsets, gate/rule definitions, DYN-SV menu order and fallback policy.
- All champion artifacts and fingerprints, feature recipes, preprocessing,
  model dependencies, fold artifacts, residual/calibration/payoff state, and
  current validation or tracking status.
- The calendar version, quote/fill conventions, data snapshot references,
  relevant source code and dependency lock, and exact score requests/results.
- A private reference corpus covering each strategy, BMO/AMC, year/month
  boundaries, missing data, non-ATM requests, pinned geometry, domain limits,
  coarse strikes, and chooser partial/tied/missing candidates.
- Existing report, book, prediction, and settlement examples with their
  provenance. Include old entry-cost estimates and actual later entry
  repricing as separate fields.

Fixtures must come through real public entry points. Do not invent a column
such as `event_id` in a fixture if the current serving row does not contain it.
Resolve and map legacy identities explicitly.

Use exact comparisons for IDs, selected contracts, integer quantities,
gate decisions, flags, null masks, and frozen canonical records. Use declared
per-field numeric tolerances for independently recomputed floating values;
never widen a global tolerance until a test passes. Unexplained changes block
cutover. A known defect requires a separately recorded correction and new
version, not a silent update to the baseline fixture.

## 4. Target architecture and ownership

```mermaid
flowchart TD
    V[Provider adapters] --> I[Ingestion and validation]
    I --> R[Immutable raw objects]
    R --> D[Versioned EOD datasets]
    I --> L[Separate live snapshots]
    D --> F[Feature and dataset recipes]
    F --> T[Model training and evaluation]
    T --> M[Immutable model releases]
    D --> S[Scoring application]
    L --> S
    F --> S
    M --> S
    G[Strategy registry] --> S
    S --> SG[Reusable structure generator]
    S --> SB[Reusable scenario builder]
    S --> PS[Reusable PnL simulator]
    SG --> PS
    SB --> PS
    PS --> PV[Shared position valuation and accounting]
    X[Experiment runner] --> S
    X --> T
    S --> C[Validated score records]
    C --> B[Prediction and position ledger]
    C --> P[Serving projections and export]
    P --> A[FastAPI read API]
    A --> U[Lazy-loading UI]
    J[Durable job supervisor] -. schedules .-> I
    J -. schedules .-> T
    J -. schedules .-> S
```

These are module and ownership boundaries, not eleven network services.
Initially deploy a small API process, one supervisor, and a limited number of
short-lived workers on the existing host. An interactive scoring worker may
stay warm during the decision window only when its memory is reserved.

| Owner | Owns | Must not do |
|---|---|---|
| Ingestion | Fetch receipts, normalization, coverage, finality, source revisions | Compute a trading verdict or change a champion |
| Feature engine | Registered transforms and their causal dependencies | Select an implicit latest dataset or silently change a missing-value policy |
| Model engine | Training recipes, folds, inference adapters, residuals, evidence | Fit during an ordinary score request |
| Structure generator | Template resolution, finite placement search, validity and completeness receipts | Rank by PnL or change strategy selection rules |
| Scenario builder | Causal historical/synthetic outcome populations, weights, mappings and RNG | Choose contracts or price a position |
| Valuation/simulation domain | Frozen-position revaluation, time/parameter shocks, cash flows and PnL distributions | Select a winning strategy or call model marks executable fills |
| Scoring application | Forecasts, shape, pricing, gate/chooser decisions, financial diagnostics | Read future outcomes or mutate strategy/model registries |
| Evaluation/portfolio | Realized outcomes, capital accounting, report generation | Recreate the selection logic used to choose trades |
| API/projection layer | Filter, paginate, authorize, serialize already computed records | Fit a model, simulate PnL, or fetch vendor data in a GET request |
| UI | Navigation, formatting, tables, charts, loading/error states | Compute gates, financial ratios, return estimates, or portfolio accounting |
| Supervisor/catalog | Transactions, leases, dependencies, capacity, retry history | Decide research conclusions |

Keep entry points such as `engine.score.score`, `score_calendar`, and existing
CLI commands as compatibility adapters while their implementations move.

See the [logical data model](rearchitecture_data_model.md) for three linked
entity-relationship views: market inputs; templates, positions and scenarios;
and registered scores, models, publication and the actual-position ledger.
The diagrams distinguish a generated position from a trade that was actually
opened, and a template from the strategy that selects its placement.

## 5. Data storage: contracts and incremental ingestion

### 5.1 Physical choices

Use three complementary stores:

1. **Immutable objects:** existing raw caches, versioned Parquet fragments,
   model artifacts, feature matrices, and score detail blobs. Keep existing
   raw research paths readable; register them in place.
2. **Transactional catalog:** SQLite on local disk, using WAL, foreign keys,
   short transactions, busy timeouts, schema migrations, and explicit durable
   commit settings. It tracks manifests, ingestion receipts, watermarks,
   jobs/leases, model releases, and score/ledger identities.
3. **Serving projections:** indexed event/score summaries in SQLite, with
   large detail/evidence objects referenced by hash. Rebuildable projections
   have a different retention policy from authoritative ledger events.

SQLite WAL supports readers alongside a writer and is intended for processes
on one host, not a shared network filesystem. Keep catalog writes serialized
and bounded. Move this catalog to PostgreSQL if multiple worker hosts become
necessary; do not put the SQLite file on shared storage. This is a deployment
boundary, not an invitation to move every option row into a database.
([SQLite WAL documentation](https://www.sqlite.org/wal.html))

Use the existing PyArrow dependency for projected, filtered, batched dataset
reads. Filter by symbol, time, and required columns before converting to
pandas. Arrow supports both filtering and batch iteration; selecting columns
alone does not bound the row count.
([Arrow dataset documentation](https://arrow.apache.org/docs/python/dataset.html#filtering-data))

Start with Arrow rather than adding a second analytical engine immediately.
DuckDB can later implement the same repository interface if measured joins
justify it. Neither pandas nor a SQL engine should leak into the domain API as
an unbounded global store.

### 5.2 Contracts must describe meaning

Extend [schemas.py](../engine/data/schemas.py) rather than replacing its
validated unit conversions. Each table contract must specify:

| Contract field | Required meaning |
|---|---|
| Identity | Primary key, duplicate policy, foreign keys, stable security/contract identifiers, and legacy-key mapping |
| Values | Type, unit, scale, adjustment basis, timezone, null/sentinel policy, allowed ranges |
| Time | Market observation time, vendor publication/update time when available, local receipt time, and final/provisional state |
| Provenance | Raw object hash, source, source priority, normalizer version, row revision and correction reason |
| Coverage | Expected symbols/contracts/sessions, explicit unavailable items, completeness rules and denominators |
| Evolution | Schema version, compatible nullable additions, explicit migration for semantic changes |
| Access | Supported bounded queries, partition/index choices, maximum batch size |

Store unavailable times as unknown. Do not manufacture an old publication time
for data downloaded today. Preserve percent-versus-fraction, adjusted-versus-
unadjusted spot, the three market-cap eras, and implied-move conventions.

Give earnings events stable IDs independent of the announced date. A calendar
revision changes event date/session while retaining the event identity when
the provider evidence supports that mapping. Preserve ambiguous mappings as
conflicts; do not merge adjacent quarterly events merely by ticker. Predictions
pin the calendar revision they used, so a later BMO/AMC correction cannot
silently change an old trade.

Keep the logical tiers: raw, normalized EOD, causal features, and causal model
forecasts. Keep provisional intraday snapshots outside those EOD tables.

### 5.3 Ingestion protocol

For each source/endpoint/partition, persist a **completed coverage watermark**,
not merely the largest date seen. It must distinguish a complete interval from
one containing a hole or a partially returned ticker batch.

The normal daily path is:

1. Plan missing sessions from coverage receipts and source publication rules.
   Check the raw cache first. A discovery refresh covers calendar frontiers
   even when there are currently no upcoming events.
2. Fetch through the existing provider adapters and shared quota control.
   Persist raw bytes and a redacted receipt before normalization. Track every
   response status, missing ticker, legitimate empty result, and quota header.
3. Normalize each new raw revision once, keyed by raw hash plus normalizer
   version. Validate expected coverage and units; quarantine failed rows or
   payloads with reasons.
4. Merge changed keys into new fragments for affected partitions. Choose the
   winning revision through explicit source-priority/finality/revision rules,
   not filesystem ordering or whichever batch arrived first.
5. Validate candidate table versions, referential integrity, and coverage.
   Commit the new manifest and watermark together only after referenced files
   are durable. Failed or partial batches never advance completed coverage.
6. Emit a change set: affected keys, symbols, sessions, columns, old/new hashes,
   and whether this was an append, correction, deletion/tombstone, or schema
   change. Downstream work consumes this change set.

An unchanged second run should normalize zero payloads and rewrite zero data
partitions. A newly acquired daily session should not scan and regenerate
twenty years of input. Retain full rebuild as an explicit recovery and
equivalence-check operation.

Keep the provider-specific defenses: ORATS requested/returned-symbol diffing,
slim field requests, no blind retries of unsupported symbols, delayed market-
wide publication, redacted query tokens, and quota headers from error
responses too. Retain Polygon curl authentication, legitimate empty bars,
shared pacing/backoff, and the current entitlement boundaries. Credential
rotation pauses the affected connector and dependent work.

### 5.4 Forward updates and historical corrections

"Only updating forward data" should mean that ordinary operation leaves
settled history untouched. It cannot mean ignoring a vendor correction, split,
late event, or source bug. Those require a separate correction path with a
visible impact plan. Old raw bytes, dataset versions, predictions, and reports
remain addressable.

Start with session partitions for chains and daily-market appends; compact
small fragments into immutable monthly files when measured size warrants it.
Sort/index for symbol and time. Feature matrices can partition by recipe and
decision month; forecasts by producer and fold. Do not create one file per
individual option quote.

Incremental computation requires dependency rules, not just `date > max(date)`:

| Change | Work that may be invalidated |
|---|---|
| New final daily session | Affected rolling state, newly eligible feature rows, quotes and current scores |
| Newly resolved earnings event | Subsequent history features for that symbol; eligible residual/analog state; future training folds |
| Corrected past daily value | Dependent rolling windows and any recursive state from that point; affected feature rows and model descendants |
| Corrected target or earlier event history | Training folds whose training set includes it, their later predictions/residuals, downstream gates/choosers |
| New quote on one contract | Scores/trades using that quote, related settlement, and affected historical decision training if applicable |
| Feature logic or model recipe change | Outputs of that recipe and its descendants, with a new version |

An expanding mean, an EMA, or a historical training set can have an unbounded
forward dependency. A 252-session window does not make every feature a
252-session dependency. Persist causal state checkpoints or replay the affected
suffix according to the actual formula. Tier-4 fold invalidation must also
include changes to residual pools and uncertainty bands, not just point-model
weights. Reuse the existing `since` equivalence guarantees.

### 5.5 Atomic snapshots and replay

A dataset version is a manifest of immutable fragments, their contracts and
hashes, and its parent version. A **research snapshot** pins the exact table,
feature, forecast, calendar, and reference-data versions consumed. Keep raw
data identity separate from model-derived forecast identity, preserving the
intent of the current separate Tier-4 hash.

Write files into staging, validate and hash them, flush durable files, then
commit manifest references in one catalog transaction. On a local filesystem,
temporary-file rename plus directory/file durability must precede publishing
the reference. A crash can leave an unreferenced object; it must never leave a
committed manifest pointing to a partial object. Readers resolve one snapshot
at the start and keep it for their entire operation.

No writer deletes the current dataset before building its replacement. No
reader follows a glob over both old and staged fragments. Garbage collection
uses manifest references and retention leases; it cannot remove an object
needed by an experiment, release, ledger record, or running reader.

Track two replay modes explicitly: **as actually known then**, which requires
original availability/receipt evidence, and **historical reconstruction**,
which uses a declared historical data vintage. A timestamp filter on a revised
2026 download does not prove what was available in 2018.

## 6. One scoring application

### 6.1 Registry objects

Register a versioned `StrategySpec` with these references:

```text
StrategySpec
  id, version, definition_hash, validation_status
  structure_recipe, parameters, contract_selection_recipe
  decision_clock, entry_policy, exit_policy
  universe_policy, domain_policy, quote_policy, fill_policy
  feature_recipe_ids, model_role_bindings
  forecast_sizing_recipe, analog_recipe, residual_and_payoff_recipes
  gate_recipe, chooser_recipe, fallback_policy
  risk_and_funding_policy_refs, evidence_refs
```

Model weights and strategies should not be one mutable object. A
`DeploymentSpec` pins a strategy version to exact model releases and policy
versions. A model promotion produces a new deployment; it does not rewrite
the meaning of an old strategy or prediction.

Feature recipes have namespaced identities such as `bucket_analogs.v1` and
`chooser_knn_analogs.v1`. They can map to the existing input names that trained
artifacts expect, but their meanings cannot collide merely because both output
`analog_mean`. Record each recipe implementation hash, inputs, universe,
lookbacks, missing/fallback rules, and source cutoff rules.

An experiment must register its candidate strategy/deployment and scoring
recipes in an **isolated research registry** before it runs. Registration
validates dependencies and makes the candidate reproducible; it does not
promote it or enable it on the board. Existing CAL-P/CND-P research remains
possible without removing their production refusal.

### 6.2 Request and result contracts

```text
ScoreRequest
  event_id, calendar_revision
  strategy_version or deployment_id
  decision_clock_id, requested_decision_at
  snapshot_id, mode: research | replay | shadow | serving
  fill_model, optional contract/geometry overrides

ScoreRecord
  schema_version, score_id, canonical_request, resolved_request
  event and clock provenance; observed/available/received timestamps
  snapshot and dependency hashes; exact model artifact IDs
  selected contracts, legs, quantities, entry/exit plan, quote provenance
  forecasts, uncertainty, residual/analog/payoff state references
  consumed feature values, null masks and per-feature provenance
  gate/rule terms, threshold, verdict, chooser candidates and selection
  financial diagnostics and requested payoff views
  validation_status, reason_codes, warnings, evidence references
```

Persist `computed_at` and other operational timings in an envelope separate
from the deterministic numerical payload. A replay does not have to pretend
it ran at the original wall-clock time to reproduce the score hash.

Use a complete canonical identity. Include event/session revision, decision
clock, fill alpha, strike/expiry/geometry overrides, source snapshot,
strategy/recipe versions, models, and residual/analog/calibration inputs.
Cache keys use the exact relevant dependencies; unrelated data changes should
not invalidate the entire board.

Preserve replay input precision. Never reconstruct a request from the rounded
number displayed on screen. Represent listed strikes with an explicit exact
contract representation; retain full precision for model-derived geometry.
Canonicalize ordering and numeric encoding once in the contract layer.
Formatting happens at the final display boundary.

### 6.3 Execution order

1. Resolve event, calendar, requested clock, allowed data ceiling, and pinned
   deployment. Validate clock and feature-contract compatibility.
2. Assemble pricing-independent features and infer the required feature models.
3. Resolve forecast-sized geometry, then select contracts using the existing
   factories and selectors. Pinned geometry still records its forecast.
4. Read bounded quote inputs and price through the current fill/structure code.
5. Assemble trade-dependent features and historical evidence; run the causal
   audit against actual source lineage.
6. Produce the model, analog, simulation, gate, and chooser inputs through the
   registered dependency graph. Preserve their distinct meanings.
7. Resolve any meta-strategy against the complete eligible candidate set.
8. Produce financial diagnostics and a validated immutable record.

Expose `score_event(request, strategies)` for the whole event and
`score_many(requests)` for batches. They share one kernel with single-strategy
scoring; batching reuses common feature work without changing results. A
direct DYN-SV request resolves its menu inside the application rather than
depending on the renderer or an experiment to have scored its competitors.

The context planner declares separate scopes for event state, market regime,
analog population, and training/calibration history. A request for AAPL may
need historical analogs from many symbols. Restricting all dependencies to the
visible watchlist would repeat the analog-context failure. Load only required
columns and intervals, but retain the entire population required by the recipe.

### 6.4 Move financial display calculations out of rendering

Move `model_vs_market`, fair premium, premium/fair, cost/width, risk summaries,
and payoff construction into tested domain functions. The API can request a
payoff view lazily by score ID, but it must use those same functions and frozen
legs. Distinguish terminal payoff from modeled value at the planned exit;
multi-expiry structures cannot be represented as if all legs expired together.

The projection layer may compute a documented display rank over a specified
score set. That rank must not become a hidden trading selector. Book totals
come from the portfolio/accounting module. The UI may format percentages and
draw chart coordinates, but it receives the authoritative financial values.

Preserve all existing refusal distinctions. Missing input, gate failure,
unvalidated strategy, superseded strategy, unsupported clock, extrapolation,
and explicitly sanctioned fallback are different outcomes. A model output of
zero is not a replacement for a missing model. The rule-level gate verdict
and operational readiness should be separate fields: an old passing forecast
can remain visible while being too stale for a current decision.

### 6.5 Reusable structure generation

Introduce a generator taking an event revision, frozen contract chain,
StructureTemplate and explicit finite PlacementDomain. It emits every valid
placement in that domain with exact contracts, signed quantities, geometry
resolution and separate structural/quote/strategy-admission statuses. Return
deterministic pages, rejection counts and a completeness receipt. A truncated
search is not an exhaustive candidate set.

This separates what can be built from what a strategy chooses. Current
strategies use their unchanged selector-resolved domains; searching all strikes
for the best simulated return requires a separately registered strategy.
Preserve exact listed mirrors, irregular-ladder dollar geometry, zero-quantity
reference legs, collision checks, pinned exits and existing quote refusals.
Extract the useful EXP-133 search patterns without making its experiment
constraints defaults for the entire engine.

### 6.6 Reusable scenarios, valuation and PnL simulation

Given a frozen proposed or actual position, build scenarios from eligible
similar historical positions, forecast/residual state, or a synthetic joint
distribution. A separate valuator reprices the same contracts under each
scenario at the requested time and parameter state. The simulator applies
signed cash-flow, fill, cost and return-denominator policies and reports the
resulting distribution. Selection/gates remain in the registered scoring graph.

The contract must distinguish elapsed sessions from calendar time, relative
IV changes from volatility-point changes, and each leg expiry from a shared
DTE. It must also separate model-to-model value change from economic PnL
against an opening fill. Historical returns without suitable factor/path data
cannot answer arbitrary horizon/IV changes; unsupported mappings refuse.
Preserve the current put-only simulation and calibrated payoff maps as named
legacy adapters. More general valuation is new capability, not an implicit
change to any existing strategy or its gate.

Reuse scenario sets and per-contract valuation blocks when dependencies agree,
under supervisor resource reservations. Combine per-leg values, not per-leg
quantiles or win probabilities. The [detailed reusable contracts](structure_generation_and_simulation.md)
specify input/output schemas, accounting signs and units, completeness,
capabilities, failure behavior, optimization limits and acceptance tests.

## 7. Reusable model training and serving

Build on [training/common.py](../engine/models/training/common.py),
[registry.py](../engine/models/registry.py), and
[tier4.py](../engine/data/features/tier4.py). Generalize their recipes and
artifact lifecycle without changing the numerical algorithms during migration.

| Concept | Responsibility |
|---|---|
| Dataset recipe | Causal inputs, target, eligibility, sample weights, groups, and label-availability times from a pinned snapshot |
| Model recipe | Role/target, feature order, preprocessing, estimator adapter, hyperparameters, seed, fold policy, calibration/residual policy |
| Model artifact | Fitted estimator, transforms, feature contracts, residuals, training fingerprint, environment, compatible clocks |
| Model release | Artifacts, OOS predictions, evaluation/evidence, dependencies, and promotion receipt in an immutable serving unit |

Use a small estimator protocol: fit, predict, save, load, and optional
distribution/interval outputs. Adapters support the current linear models,
GBMs, neural networks, and blends. Keep existing sklearn-compatible estimators;
add another NN runtime only when a concrete experiment requires it. Model
architecture and model role are independent.

Do not conflate predicted earnings-move size, forecast-based option geometry,
and portfolio position size. They belong to a feature model, a strategy recipe,
and a funding policy respectively. Preserve existing gates as return predictors
or arithmetic rules; a gate score is not a probability.

Feature-model predictions used to train gates or choosers must be causal OOS
predictions. Preserve the distinction between a monthly Tier-4 fold forecast
and the full refit champion forecast shown as another diagnostic. They can
legitimately differ. The dependency graph registers upstream producers and
rejects cycles. Training membership and label-availability receipts must prove
that an upstream fit did not see the downstream row or its future labels.

Preserve current expanding-year evaluation and monthly feature-model folds
where registered. Validate that overlapping targets were available before
each fold cutoff; new recipes declare any required purge/embargo. A discovered
violation blocks release and requires separate investigation, not a silent
split change. Fit preprocessing, target transforms, thresholds, calibrators,
residual buckets, and NN early stopping only on permitted training/validation
data. Preserve existing missing-row masks; a generic imputer changes the model.

Materialize fold models and residual/calibration state as supervised jobs.
Where the current scorer fits a monthly model into a local cache, persist the
identical fit keyed by recipe, training content, fold cutoff, seed and runtime.
Scoring only loads and infers. An absent artifact returns MODEL_NOT_READY and
can enqueue preparation; it does not trigger training inside an HTTP request.
New rows in a month use its already frozen model. Corrections invalidate the
dependent folds and residual states, which can include every later fold.

Generate model evidence from the exact frozen training dataset. Store its
recipe, coverage, feature explanations, input diagnostics and OOS metrics with
the release. Eliminate feature-list-based dataset guesses in
`model_evidence._dataset_for`. Promotion requires a complete compatible set of
artifacts, evidence and parity receipts, then atomically switches a deployment
pointer. Rollback restores that pointer without overwriting any weights.

## 8. Durable jobs for nightlies and experiments

### 8.1 One supervisor

Add one supervisor with a durable SQLite job table and an executor adapter.
Its scope is dependencies, leases, resource admission and subprocess execution;
research algorithms remain engine modules. A distributed broker or cluster
manager is unnecessary for the initial single-host workload.

Every nightly timer, experiment CLI, web action and heavy diagnostic submits
a JobSpec: immutable input references, implementation/spec/environment hashes,
dependencies, resource class, priority/deadline, provider budget, output
namespace, checkpoint contract and retry policy. Named resource classes own
conservative estimates, updated from measured peaks between runs. An
unprofiled heavy class starts alone. Users choose the job, not CPU ranges.

Use transactional claims, renewable leases and fencing tokens. A recovered
job must reject writes from its old orphaned worker. After host suspension,
reconcile processes and leases before starting another attempt. Execution is
at least once; unique logical output keys and atomic commits make effects
idempotent. Checkpoints include code, inputs, parameters and environment,
not merely a `.done` marker.

FastAPI returns a job ID immediately. Do not keep a long subprocess inside an
HTTP worker or use in-process BackgroundTasks as a durable scheduler. FastAPI
itself distinguishes small background tasks from heavy external work.
([FastAPI guidance](https://fastapi.tiangolo.com/tutorial/background-tasks/#caveat))

### 8.2 Automatic resource admission

Budget against the effective host/container memory ceiling and allowed CPU
affinity. Reserve room for the API, supervisor, OS and external workloads.
Admission must satisfy both the total reservation budget and current available
headroom, including unused reservations held by already running jobs.

On this roughly 7GB host, initially allow only one heavy scoring or training
process at a time. A 5.5GB rebuild and a 3GB scorer cannot coexist. If a stage
does not fit the available worker budget, keep it queued with a reason; split
or optimize it, or change host capacity. Reducing the declared reservation
does not reduce its memory use.

Allocate disjoint CPU sets centrally and set BLAS/OMP, estimator and loader
thread counts consistently. Budget disk scratch and disk-heavy work too.
Run normalization, feature generation, fitting and scoring in separate
processes so rebuild memory is released before the next stage. Bound caches
by bytes and snapshot. CPU separation alone is not memory isolation.

Provide two executors:

- Preferred: delegated Linux cgroups per job, with CPU/memory controls and
  whole-group ownership. `memory.high` applies pressure before `memory.max`;
  record OOM events. Kernel enforcement still needs global headroom.
  ([Linux cgroup documentation](https://cdn.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html))
- Current-host fallback: supervise `bounded_run` with strict global admission,
  disjoint placement, process-tree accounting and conservative headroom. Probe
  capability at installation: the current executor records unavailable
  writable cgroups/systemd. A polling watchdog cannot contain every allocation
  spike; describe this mode as best-effort containment.

Reserve live collection/scoring capacity ahead of the decision window.
Stop admitting experiments that could overlap it. Yield low-priority jobs at
safe checkpoints and exit their workers to release memory; SIGSTOP alone
retains RAM. A permanently warm scorer also needs a reservation.

All provider requests share account-level rate/quota state across jobs.
Preserve Polygon single-consumer pacing, ORATS live reserve and large-pull
approval rules. Reserve estimated calls and reconcile every available quota
header, including error responses. Missing headers retain uncertainty rather
than resetting usage. Distinguish 401, permanent 404, legitimate empty data,
429, transient failure and delayed publication in retry policy.

### 8.3 Nightly graph and release boundary

```text
plan -> ingest -> validate/normalize changed partitions
     -> commit candidate data snapshot -> resolve final session
     -> update affected features -> prepare model/state artifacts
     -> score batches -> validate features, decisions and replay
     -> commit validated predictions and score release
     -> build projections/export -> publish -> verify delivery
```

Settlement is an independent idempotent branch after final data ingestion.
A failed board must not prevent resolving an existing position when its
required data is final; a missing exit must remain visible and unresolved.

Checkpoint event batches and folds. Resume hash-matching shards and recompute
only unfinished work. This fixes the current loss of expensive computed scores
even when downloaded data survives a killed nightly.

Validate decisions before freezing new predictions. Keep the serialized
projection self-check before publication as a separate protection. Failed
candidates remain private diagnostics; they do not become validated
recommendations. Existing immutable bad records require a reasoned superseding
entry, never deletion or rewriting.

Commit validated predictions, release metadata and a publication outbox
transactionally. Deliver the remote release idempotently afterward. Publication
failure retains the frozen predictions and retries delivery; track validated,
published and delivered times separately. Computed does not mean seen by the
user. Keep the previous release readable with its true timestamp and a current
operations banner.

Replace one last-success timestamp with separate watermarks for ingestion,
scoring, prediction commit, settlement, publication and backup. A failed
publish cannot count as successful delivery.

### 8.4 Experiment lifecycle

```text
register candidate + preregister spec
  -> pin inputs and plan resources/quota
  -> build dataset -> fit causal folds -> score through engine
  -> replay selected trades on real prices
  -> evaluate -> REPORT.md -> testing ledger
  -> finalize artifacts -> private mirror sync once
```

Retain `engine.evaluate.evaluate(..., write_report=True)`: fill sensitivity,
sample funnel, stress, capacity, capital accounting, provenance and accuracy
checks remain mandatory. Model-only studies retain model metrics and evaluate
the relevant implied strategy/book rather than inventing trade headline values.
Promotion remains a separate evidence-based operation.

Keep every execution attempt in a run log, distinct from the multiple-testing
ledger of hypotheses/specifications. Failed evaluations remain visible.
Mandatory runner `--no-ledger` mode prevents subset pipeline tests occupying
real experiment slots. Changing a grid cell must regenerate its actual scores
and trades; changing only YAML while reusing one frame is not another test.

New runners call engine-owned registered components. Extract the exact
EXP-169/EXP-161 chooser recipe first, prove parity, and retain old runners as
adapters. Production must no longer import experiment runners by filesystem
path. Candidate registration does not mutate the production registry.

## 9. A richer UI with lazy data access

Use a shared component system for typography, spacing, status labels, tables,
forms and charts. Keep the event list focused, with an expandable evidence
drawer instead of placing every diagnostic in a permanent column.

| View | Preserved and added capability |
|---|---|
| Earnings | Date/session search, decision deadlines, strategy verdicts, source age, unavailable/stale states |
| Event detail | All strategies, gate terms, legs/order ticket, strike/expiry alternatives, payoff, historical prints and analog evidence |
| Portfolio | Hypothetical/contrarian books, predictions, entries/exits, unresolved items, funding and comparable summaries |
| Models | Active/historical releases, input/decile evidence, dependencies, OOS metrics and calibration |
| Research | Experiments, generated reports, primary/secondary arms, sample funnels and fill sensitivity |
| Operations | Job progress, failures, source coverage/finality, quota, publication and backup age |

Retain existing filters, sorting, deep links, disabled/out-of-domain visibility,
model health and offline downloads. Preserve evidence status: a tracked
structure and a promoted structure should not appear equally validated.
Use semantic controls, keyboard navigation, readable loading/error states and
mobile layouts. Put meaningful clocks/source age beside the decision and
technical hashes in the evidence drawer.

Recommend React + TypeScript + Vite and TanStack Query for independent screens,
typed contracts, pagination, cancellation and pending/error states. TanStack
supports paginated queries while retaining the previous page during loading;
never obscure the loading state when the release changes.
([TanStack pagination](https://tanstack.com/query/latest/docs/framework/react/guides/paginated-queries))

Generate client types from the versioned API schema. Pin tested dependencies
during implementation. The code build never reads market data or model
artifacts. Initial requests fetch release metadata and one bounded event page;
detail, analogs, history and model charts load when opened. Cursor/cache keys
include release, filters and clock. Immutable objects use ETags. Browser focus
must not initiate paid live collection automatically.

The [component contracts](component_contracts.md) specify the routes and
payloads. Read endpoints serve saved projections. Expensive what-if/live work
returns a job ID. Authenticate both HTML and data URLs; restrict stateful and
quota-consuming actions to operator roles, with CSRF protection where needed.
Use bounded filters and allowed sort fields. Provider credentials stay in
connectors. A new release triggers a visible refresh, never a mix of old lists
and new details under one timestamp.

Keep a common DataClient with HTTP and offline-snapshot implementations.
The ordinary UI loads separate data. An explicit offline export intentionally
packages a pinned release, retaining the single-file fallback and disabling
live actions. Test it with networking disabled.

The authenticated remote static snapshot remains available when the research
host sleeps. It can lazily serve detail objects and filter a bounded exported
index; new scoring requires the API. A later remote read service can provide
server pagination if that index outgrows static operation. Do not remove
remote access while adding a local API dependency.

Dependable intraday operation requires an awake collector/scorer during the
decision window. Recommend supervised always-on Linux for the live worker
when that capability is ready. A sleeping WSL host cannot meet a live deadline;
host provisioning is a separate deployment step.

## 10. Live scoring preserves structures but adds a new clock

Implement [live_intraday_scoring.md](live_intraday_scoring.md) through the same
scoring application. Economic structures remain unchanged; a new information
clock is a versioned deployment and research question. Do not silently replace
current close-clock champions with intraday inputs during migration.

Start with the proposed `d0_preclose_1545_et`. D0 identifies the strategy entry
session, not automatically the earnings date. A BMO print normally requires
the preceding trading session; 15:45 on its earnings date is already too late.
AMC normally permits a same-date pre-close decision. STR-RUNUP still enters
fourteen sessions earlier and exits before the print.

Use the exchange calendar and America/New_York, including DST, holidays,
early closes and event-session revisions. Initially refuse early-close days
for the 15:45 clock unless a separately validated shortened-session clock
exists. Automatically moving to 12:45 would change the definition.

Distinguish data cutoff, source availability, local receipt, snapshot sealing,
score completion, publication and execution deadline. A 15:44 observation
received at 15:46 was unavailable at 15:45. Prefetch before cutoff, seal only
eligible data, and record real completion/publication times. A late result is
late even when its feature timestamps pass causality.

ORATS currently documents live one-minute summaries/chains, CSV responses,
historical routes, source timestamps and parity-derived stockPrice. Coverage
differs by endpoint. Documentation is not proof of this account entitlement
or completeness for every symbol.
([ORATS live intraday API](https://orats.com/docs/live-intraday-api))

Begin implementation with a small redacted market-hours schema/entitlement
probe through existing quota controls. Confirm field units, timestamp and DST
semantics, coverage and cross-source consistency. Do not purchase a plan or
retry known-unauthorized Polygon tick endpoints by implication.

An append-only live snapshot stores raw responses, redacted receipts, source
and receipt timestamps, prior final-EOD snapshot, calendar revision, normalized
features, exact recipe/model dependencies, quote-quality results and score
records. All required legs must satisfy age/skew limits. Retain stockPrice and
spotPrice and explicitly identify the input each recipe uses. Missing live
provenance refuses the affected score. A close artifact cannot serve as a
fallback for a pre-close model.

Prior final EOD history plus permitted live fields is valid input. Same-day
provisional EOD tiers, request-time Tier-4 rebuilds and request-time model fits
are forbidden. Stored live decisions replay offline from their frozen inputs.

Train pre-close models on the actual pre-close contract, with causal folds.
Compare D0-close, D1-close and pre-close benchmarks on matched events; report
unmatched coverage separately. Historical snapshots without historical receipt
evidence constitute reconstructed timing experiments with declared latency
assumptions, not proof of what the account actually received then.

Preserve the live guide milestones: five valid collection sessions, causal
and replay tests, historical experiment, at least twenty valid shadow sessions,
then paper execution review. These do not replace the existing season-length
forward-test/go-live requirements. EOD-model-on-live-input diagnostics stay
shadow until the clock is validated and promoted.

Scoring never creates a fill. Record intended order/limit, submission time,
paper-versus-broker evidence, actual entry and actual exit separately. A 15:45
score does not establish execution at the official close. Daily close/VWAP or
a high/low crossing a limit does not prove a simultaneous multi-leg fill.
Retain measured-fill comparisons and the current data-entitlement limits.

## 11. Validation that catches and explains failures

Keep the unit suite and real-data acceptance checks, and add independent
boundary tests. A same-engine replay alone can agree with a shared bug.

| Check | Required proof | Trigger |
|---|---|---|
| Strategy compatibility | Same contracts, timing, nulls, forecasts, thresholds, flags and choices | Every migration step |
| Training/serving parity | Dataset rows and production context produce the same registered inputs | Model release; sampled nightly |
| Serving/replay parity | Single/batch, cache/fresh, pinned geometry, reordered inputs, API/export agree | Code change; nightly |
| Causal sources | Actual observation and label availability respect cutoff; future poisoning changes nothing | Feature/model changes; live snapshots |
| Incremental/full equality | Append/correction result matches rebuild; unaffected fragments unchanged | Data/recipe changes; bounded audits |
| Crash/retry atomicity | Last release readable; resume matches uninterrupted output; no duplicate effects | Storage/scheduler changes |
| Resource admission | Heavy jobs queue, live reservations hold, lease/worker recovery works | Scheduler and operational drills |
| Provider contracts | Partial batches, empty results, units, auth/rate errors and publication delay are handled correctly | Connector changes; ingestion |
| Ledger/accounting | Timely decisions, evidence-backed entry/exit, cash and positions reconcile | Commit/settlement |
| UI contracts | Bounded lazy requests; no mixed release; values/totals match saved records | Frontend changes |
| Evidence/recovery | Matching deployment/model page; restore replays score and ledger | Promotion; restore drill |

Name regression cases for the failures this program has encountered:

- BMO/AMC decision cutoffs and the separate event-history ceiling.
- Watchlist-bounded serving retaining the required broad analog population.
- Chooser kNN features versus STR-THRU bucket analog features.
- Pinned shape still recording forecast and gate inputs.
- Full-precision geometry surviving every serialization layer.
- Analog ordering and deterministic residual sampling.
- Partial/missing chooser menus and exact current tie/fallback behavior.
- Shrunk/empty universes not giving vacuous coverage passes; expected,
  supported and received populations recorded separately. Preserve baseline
  denominator behavior while making population collapse independently visible.
- Model promotion never serving old evidence under a new model identity.
- Late calendar/quote revisions leaving original decisions replayable.

Assert null masks, fallback rates, missing-row counts and eligible populations,
not just agreement on non-null pairs. Include negative controls that corrupt
a timestamp, feature builder, geometry, model hash or dataset membership and
prove the corresponding check fails. Use hand-calculated leg/cashflow fixtures
and independent source comparisons where they establish financial correctness.

Make documented serving approximations explicit too. For example, the current
chooser maps a grid-derived n_admissible input to a chain-depth-conditioned
surrogate. Preserve that behavior in baseline compatibility, register the
mapping and its evidence, and test the serving value against that mapping.
Report the difference from the original training input separately. Do not
silently declare exact training/serving equality for a known approximation,
or introduce new approximations to make parity pass.

A failure receipt names the first different stage, request/dependency hashes,
expected and actual fields, null masks, source rows/timestamps, recipe/model
versions and affected counts. Print a compact diagnosis immediately; retain
private detail even if the job exits. A recurring mismatch is not permission
to ignore it. Model calibration drift and poor OOS performance are distinct
from infrastructure failure, and process health does not prove causal validity.

## 12. Migration sequence and rollback

Run legacy/replacement paths on identical snapshots. Compare sequentially on
this host so a parity check does not need two multi-gigabyte scorers alive.

| Phase | Deliverable | Exit gate |
|---|---|---|
| 0. Baseline | Contract/artifact/screen inventory; private corpus and negative controls | Every strategy and critical refusal reproducible |
| 1. Operations | Catalog/supervisor wrapping existing commands; receipts and score checkpoints; pre-commit validation | Crash/resume and competing submissions pass; no per-job CPU selection |
| 2. Data access | Snapshot repository, immutable manifests, bounded reads, legacy adapters | Existing scores match; failed rebuild leaves active snapshot intact |
| 3. Incremental data | Coverage watermarks, changed-key merges, dependency invalidation, correction path | No-op rewrites zero data; append/correction matches clean rebuild |
| 4. Scoring | Registered extracted recipes, canonical score contract, engine-owned chooser data, financial logic moved from renderer | All strategies match; production imports no experiment runner |
| 5. Models | Persisted fold artifacts, transforms/residuals, dataset recipes and evidence; atomic promotions | Zero fitting on scoring requests; exact promotion/rollback |
| 6. UI | Read API then one screen at a time; old UI/export adapters retained | Feature inventory, lazy-load checks, phone/offline access pass |
| 7. Live shadow | Entitlement/schema proof, snapshots, causal live features, clock-specific experiments | Live guide gates pass; no contamination; timely publication |
| 8. Cutover | Switch consumers; remove duplicate implementations; update recovery/operations docs | Ten consecutive completed-session runs without manual resource placement; restore and compatibility evidence pass |

The UI can start against frozen projections after phase 2. Shadow collection
can begin before model readiness once its contract/quota are defined. Cutover
remains gated on relevant parity. Ten stable nights is an operational target,
not statistical evidence for a strategy.

Measure performance on a fixed workload with concurrent jobs, memory budget,
cache state and input size recorded. Proposed targets: sub-second read API;
usable first page within two seconds on the agreed device/network; only one
initial event page; recovery recomputes at most unfinished batches/folds.
Measure provider wait, queue wait and computation separately before setting
a nightly deadline. Live latency must fit the decision/execution deadline.
None of these timings were measured in this review.

The first implementation slice should deliver the baseline corpus, canonical
score schema, paginated read API over saved scores, and supervised legacy
scoring. Add immutable publication before switching rebuild behavior. This
creates useful UI/operations boundaries without simultaneously rewriting
storage, scoring and models.

Keep old research trees, specs/results, imports, predictions and reports intact.
Build new catalog/data state alongside the legacy store. During shadowing only
one path commits official predictions, registry changes and testing ledgers.
Rollback restores a prior deployment/release; it does not erase historical
facts. Necessary economic changes get separate experiments or explicit defect
corrections with evidence.

## 13. Privacy, observability and recovery

Extend the existing public-code/private-evidence policy with exact allowlist
rules for frontend source and lockfiles. Keep datasets, compiled data exports,
catalogs, models, live snapshots and decisions private. Preserve hygiene scans
and authenticated access to both pages and their data objects.

Structured job events record stages, counts, elapsed time, resources, input
versions and failure classes, with at least a minute heartbeat. Distinguish
OOM, watchdog kill, auth failure, stale data, validation failure and publish
failure. A private reproduction reference should make diagnosis possible
without repeating an hour of work.

Continue one private-mirror sync after an experiment is finalized. Extend
backup coverage to catalog transactions, snapshot manifests, model releases
and original live raw responses needed to reproduce decisions. Such data
cannot always be re-downloaded in its original vintage. Use a private artifact
backup for bulky immutable objects rather than storing every revision in Git.
Plan retention/storage costs explicitly; this extends the current accepted
raw-data backup gap.

Use a consistent SQLite backup mechanism plus all referenced immutable files;
copying only a live database file while ignoring its WAL is insufficient.
A restore drill must recover a deployment, replay an original score without
network access, reconcile the ledger and open the report.

## 14. Review and completion

The [component contract specification](component_contracts.md) makes the
boundaries in this guide concrete: schemas, method signatures, time semantics,
failure types, transactions and compatibility tests. Review those contracts,
the [data model diagrams](rearchitecture_data_model.md), and the
[generator/simulation specification](structure_generation_and_simulation.md)
before building the new components.

The rearchitecture is complete when the existing strategy corpus agrees,
the UI feature inventory passes, normal ingestion advances incrementally,
and all callers use registered scoring recipes. Jobs must survive interruption
without manual resource placement, and failures must leave prior releases
readable. Live readiness is an additional gate requiring a compatible clock,
causal inputs, timely publication, generated reports, shadow/paper evidence
and the existing go-live controls.
