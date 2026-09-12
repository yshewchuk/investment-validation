# Component contracts for the earnings research and scoring system

Date: 2026-09-12. Status: proposed v1 contracts for design review.
Companion: [system architecture and migration](system_rearchitecture.md).
Related: [data model diagrams](rearchitecture_data_model.md) and
[reusable structure generation and simulation](structure_generation_and_simulation.md).

This document specifies the boundaries to implement, not APIs that already
exist. Names are proposed. JSON examples are synthetic and hashes/IDs are
shortened placeholders; they contain no real recommendation or market data.
Required fields, invariants and failure behavior are part of the design.

The migration rule is unchanged: adapt the current strategies, artifacts and
calculation recipes to these contracts, and prove equivalence before cutover.
Introducing a contract does not authorize a different strategy definition,
fallback, model, threshold, fill convention, or decision clock.

## 1. Boundary map

| Producer -> consumer | Contract | Durable owner |
|---|---|---|
| Connector -> ingestion | RawReceipt and CoverageReceipt | Raw store/catalog |
| Normalizer -> repository | NormalizedBatch and ChangeSet | Dataset catalog |
| Repository -> all readers | SnapshotRef and bounded DataQuery | Dataset catalog |
| Feature engine -> training/scoring | FeatureRecipe and FeatureFrame | Feature store |
| Strategy registration -> scoring | StrategySpec and DeploymentSpec | Versioned registry |
| Model training -> inference | ModelRecipe, ArtifactManifest, ModelRelease | Model registry/artifact store |
| Scoring -> structure generator | StructureTemplate, GenerationRequest, CandidatePosition, GenerationReceipt | Candidate artifact store/catalog |
| Historical/synthetic source -> scenario builder | ScenarioBuildRequest and ScenarioSet | Scenario artifact store |
| Simulator -> position valuator | PositionDefinition, MarketState, ValuationPolicy, PositionValuation | Immutable valuation artifacts |
| Scoring -> PnL simulator | PositionState, SimulationRequest, SimulationResult | Simulation artifact store |
| Scoring -> experiment/API/ledger | ScoreRequest, ScoreRecord, ScoreBatch | Score store |
| Experiment -> evaluation | EvaluationInput and ExperimentManifest | Research artifact store |
| Entrypoints -> supervisor | JobSpec, JobReceipt, ProgressEvent | Job catalog |
| Scoring -> prediction ledger | ValidatedDecision | Append-only ledger |
| Execution/settlement -> ledger | PositionEvent | Append-only ledger |
| Projection/publisher -> UI | ServingRelease and paginated views | Release catalog |
| Any comparator -> operator or agent | ComparisonReceipt | Diagnosis store |

Only the named owner commits authoritative records. Other components submit
commands or immutable candidate objects. No renderer writes a score, no model
trainer writes a production promotion directly, and no experiment writes a
parallel decision formula outside its registered recipe.

The [entity-relationship diagrams](rearchitecture_data_model.md) show how these
contracts join through stable keys and immutable manifests, including the
difference between a simulated position and an actual ledger position. The
[generator and simulator specification](structure_generation_and_simulation.md)
extends this boundary map with full schemas, units, capability checks and tests.
Its components are shared engine modules, not additional microservices.

## 2. Shared types, identity and compatibility

### 2.1 Common types

| Type | Wire representation and semantics |
|---|---|
| Timestamp | RFC 3339 UTC with explicit Z, canonical microsecond precision; never a naive datetime |
| SessionDate | YYYY-MM-DD plus a referenced exchange-calendar version; never interpreted as midnight availability |
| ContentHash | `sha256:` followed by the complete 64-character lowercase hex digest |
| ObjectRef | `{kind, schema_version, object_id, content_hash}`; storage location resolved internally |
| DecimalAmount | Decimal string plus currency/unit; no loss through JavaScript integer/float limits |
| FeatureNumber | Finite binary64 value; unit defined by the feature contract; missing is null with a reason |
| ContractId | Stable internal instrument ID plus vendor symbology mapping, expiry, right, exact strike, multiplier and adjustment identity |
| EventRef | Stable event ID plus a calendar revision; ticker/date are attributes, not sufficient identity |
| SchemaVersion | Object family and major/minor, for example `score_request.v1.0` |

Exact strikes and money use decimal strings at the interface. Adapters preserve
the current calculation precision and rounding conventions internally; changing
to decimal arithmetic everywhere would itself be a numerical migration.
Percent points, return fractions, probabilities and dollars are different
units. For example, 5.0 percent-of-spot and 0.05 return-fraction are not
interchangeable model inputs.

Model features may retain legacy names/order through an explicit binding map.
Never infer the unit or recipe from a convenient column name. A missing value
cannot be sent as NaN, Infinity, zero, or an omitted required field.

### 2.2 Deterministic content versus execution metadata

Use canonical JSON for JSON content hashes, following JCS serialization and
ordering rules. Exact decimal values remain strings. Preserve array order
where meaningful, including feature order, menu order and leg definitions.
Do not use Python repr or display-rounded JSON for identity.
([RFC 8785](https://www.rfc-editor.org/rfc/rfc8785))

Each object has a deterministic payload and an operational envelope. The
payload hash excludes its own ID/hash and the envelope. The envelope contains
created/computed timestamps, worker/run IDs, duration and log references.
Replaying a score can reproduce its payload without reproducing elapsed time.

For Parquet, keep both a byte hash for integrity and a logical content hash
defined by schema, primary-key ordering and canonical values. Compaction can
change bytes without changing logical content. Pin the content-hash algorithm
version and use Merkle-style partition manifests so an append does not require
hashing every historical row again.

Identity and RNG are separate contracts. During migration preserve the legacy
RNG/seed recipe, including its request identity behavior. Switching hashing or
sampling algorithms is not permitted to silently change Monte Carlo results.
New recipes can adopt a new explicit deterministic RNG policy.

### 2.3 Version and evolution rules

- Commands and recipe registrations reject unknown fields and unknown enum
  values; silent typo/default behavior is forbidden.
- Compatible response additions require a minor schema version. Consumers
  can ignore documented optional fields but cannot guess new safety statuses.
- Changed units, field meaning, timing, null policy or required fields require
  a new major contract or explicitly versioned recipe.
- Old versions remain readable for retained experiments and ledger records.
  Migration adapters are explicit and tested; frozen payloads are not rewritten.
- A mutable name such as current or champion resolves to an immutable ID once
  at command submission. All subsequent work pins that resolution.

### 2.4 Shared failure envelope

```json
{
  "schema_version": "problem.v1.0",
  "code": "SNAPSHOT_NOT_READY",
  "category": "dependency",
  "retryable": true,
  "message": "The requested final session has not been committed.",
  "stage": "resolve_snapshot",
  "trace_id": "trace_example",
  "dependency_refs": ["snapshot_candidate_example"],
  "retry_after_seconds": 60,
  "diagnostic_ref": "private_diagnostic_example"
}
```

Categories are validation, dependency, source, resource, integrity and internal.
Provider credentials and raw licensed responses never appear in public errors.
A successful score may contain a refusal verdict; refusal is a business result,
not HTTP 500. Transient worker/provider failures are job failures, not a fake
zero-valued score. HTTP conventions: 422 invalid request; 409 incompatible
version/idempotency conflict; 404 unknown immutable ID; 429 application rate
limit; 503 temporary dependency failure; 202 accepted job.

### 2.5 Type kinds

This specification defines roughly seventy named types. They are grouped two
ways already — the boundary map groups them by durable owner, and sections 3-15
group them by pipeline stage. Neither says what KIND of thing a type is, which
is what determines its lifecycle. Every type belongs to exactly one kind, and
the suffix carries it:

| Kind | Suffix | Lifecycle | Examples |
|---|---|---|---|
| Definition | `*Spec`, `*Recipe`, `*Template`, `*Policy` | Registered once, versioned, immutable thereafter; identified by a definition hash | StrategySpec, FeatureRecipe, ValuationPolicy |
| Handle | `*Ref` | A pinned pointer to an immutable object; cheap to pass, never carries values | SnapshotRef, ObjectRef, EventRef |
| Command | `*Request` | Transient input to one operation; hashed to form the operation's identity | ScoreRequest, GenerationRequest |
| Record | `*Record`, `*Release`, `*Set`, `*Frame` | The durable numerical result; content-hashed, replayable, never rewritten | ScoreRecord, ModelRelease, ScenarioSet |
| Receipt | `*Receipt` | Evidence that an operation happened, and under what conditions; retained even when the operation failed | RawReceipt, PromotionReceipt, ComparisonReceipt |
| Event | `*Event` | An append-only fact in a sequence | PositionEvent |
| Envelope | `*Budget`, `*Resources`, `problem` | Operational metadata; explicitly excluded from every content hash | GenerationBudget, ResolvedResources |

Two rules follow from the table and are worth stating because they are the ones
most easily broken. A Definition may never reference a Command or an Envelope:
if a budget or a deadline can reach a definition hash, the same strategy gets
two identities on two machines. And a Record's hash covers only Definition,
Handle and Command inputs: an Envelope changing must never change a Record's
identity, which is what lets a replay reproduce a payload without reproducing
its elapsed time.

Types that do not fit their suffix are the outliers worth fixing before
implementation, not after: `EvaluationInput` is a Command, `ExperimentPlan` is
the resolved form of a Definition, and `DataQuery` is a Command. Rename or fold
them when the contract package is written.

## 3. Connector -> ingestion

### 3.1 Fetch contract

```text
FetchService.fetch(FetchRequest, ProviderLease) -> RawReceipt

FetchRequest:
  source_id, endpoint_id, canonical_params_without_credentials
  cache_policy_id, expected_coverage, publication_policy_id
  timeout_policy_id, request_idempotency_key

RawReceipt:
  receipt_id, request_hash, raw_object_ref
  source_id, endpoint_id, redacted_params, response_status, content_type
  requested_at, received_at, vendor_published_at?
  source_revision?, adapter_version
  cache_hit, quota_charge, quota_observation?
  parse_status, coverage_receipt_ref
```

Endpoint IDs resolve to vetted connector code. Callers cannot pass arbitrary
URLs or credential fields. Credentials are injected only in that adapter.
Raw objects remain immutable; the same request can have multiple receipts
and revisions. Cache hits do not pretend the original response arrived again
at a new market observation time.

CoverageReceipt records requested, returned, accepted, unsupported, genuinely
empty and failed keys. Its definition names the key space: symbol, symbol/day,
contract/day, or market session. A 200 status with omitted requested tickers is
partial coverage and cannot advance the completed watermark.

A capability record names supported endpoints, temporal resolution, observed
entitlement, source coverage window, verified_at and unavailable reason. Known
Polygon daily-only history constraints and connector-specific authentication
remain implemented in adapters, not rediscovered by each job.

**Acceptance:** repeated cached request spends no quota; partial ticker response
cannot be recorded complete; zero-trade Polygon contract is a legitimate empty
result; credential rotation pauses the connector without repeated retries.

## 4. Normalizer -> data repository

### 4.1 Normalized batches and corrections

```text
Normalizer.normalize(RawReceipt, TableContractRef) -> NormalizedBatch

NormalizedBatch:
  batch_id, input_receipt_ids, raw_hashes, normalizer_recipe_ref
  table_contract_ref, candidate_fragment_refs
  key_bounds, time_bounds, rows_in, rows_out
  inserted_keys_ref, revised_keys_ref, tombstoned_keys_ref
  validation_receipt, quarantine_refs, coverage_receipt_ref

Repository.commit_batch(batch, expected_parent_version, writer_fence)
  -> DatasetVersion + ChangeSet
```

The normalizer is offline and deterministic. Input identity is raw content
plus normalizer version and configuration. Units/source-priority/sentinel
handling are table-contract rules. Unknown required source fields fail
normalization; nullable historical schema gaps are represented explicitly.

`commit_batch` uses compare-and-swap on the parent version and a valid writer
fence. A concurrent commit causes a conflict and a new merge against the new
parent, not blind overwrite. Identical repeated batches return the original
receipt. Partial source success can commit validated partitions, but only
their completed coverage is advanced and the batch remains visibly partial.

ChangeSet fields: previous/new dataset versions, affected table/key references,
symbols, dates, columns, change_kind, causal_reason, and earliest affected
feature/label anchors. The dependency planner derives downstream work. It
must not assume every corrected rolling feature has a fixed short horizon.

Corrections create new revisions. A tombstone hides a fact in a new snapshot
but leaves its old revision accessible. A later correction does not modify
already committed score or ledger records.

### 4.2 Atomicity

Write fragments to staging, validate, hash and flush them, then atomically
commit manifest references and completed coverage. Files become durable before
catalog pointers reference them. The old manifest remains readable throughout.
Garbage collection considers retained releases, readers, experiments and
ledger references; unreferenced staging objects can be reclaimed separately.

**Acceptance:** injected crash at each boundary exposes either the previous or
new complete version; no partial table becomes current. Appending one session
does not rewrite unrelated historical fragments. Correction plus incremental
recompute matches a clean rebuild over the same raw revisions.

## 5. Repository -> features, training and scoring

### 5.1 Snapshot and read contracts

```text
SnapshotRef:
  snapshot_id, manifest_hash, parent_snapshot_id?
  table_versions: map[table_name, DatasetVersionRef]
  calendar_version, source_priority_version
  finality_receipt_refs, knowledge_mode

DataQuery:
  snapshot_id, table_contract_ref, columns
  key_filter, time_interval, order_by
  max_batch_rows, max_result_rows?, deadline?

Repository.scan(DataQuery) -> iterator[RecordBatch]
Repository.get_event(EventRef, SnapshotRef) -> EarningsEvent
Repository.get_chain(ChainQuery, SnapshotRef) -> ChainSnapshot
Repository.explain_dependencies(QueryOrRecipe) -> DependencyPlan
```

No default full-table read in serving. Training may request a full historical
range explicitly through streaming batches and a supervised resource profile.
Queries are bounded before pandas conversion. A missing version never falls
back to latest. Snapshot handles keep the same version through the whole call.

`knowledge_mode` is `observed`, `attested_stable` or `reconstructed`, and it is
recorded per table rather than per snapshot, because the risk it describes is a
property of the field.

- `observed` — an original availability/receipt record exists proving the data
  was held at the requested time. Reserved for decisions made under a live
  clock. No back-dated corpus can be promoted into it, and no future live
  snapshot may claim it without a real receipt.
- `attested_stable` — no contemporaneous receipt, but the values are attested
  not to have moved since, and the attestation carries a measured cross-source
  or cross-vintage agreement rate naming what was compared and when. Field
  class alone is never the attestation: an expired listed option chain is
  *expected* to be immutable, but expiration is not evidence — a vendor can
  restate settled history, and only measurement shows whether one does. Note
  what the measurement does and does not establish: agreement across vintages
  is revision confidence, and says nothing about WHEN a value first became
  obtainable. A field can be perfectly stable and still have been unknowable
  at the simulated decision time.
- `reconstructed` — a revisable field read from a single late vintage. Earnings
  dates, session classifications and split-adjusted spot stay here, because
  those are exactly the values a later download silently changes.

The two risks being separated are availability (could this have been obtained
then?) and vintage (has the value changed since?). They are separated because
their evidence and their repairs differ — NOT because only one of them binds a
simulation. Both do. Vintage threatens what a simulation consumed: a revised
value means the simulation ran on a number that never existed. Availability
threatens WHEN a simulation may consume it: reading a value that was not yet
published at the simulated decision time is look-ahead bias even if the value
was never afterwards revised, and cross-source or cross-vintage agreement
cannot catch it, because agreement measures drift, not publication lag.
Decision-time eligibility is therefore enforced against availability evidence
for every simulated decision, independently of the vintage classification, and
a knowledge_mode grades the strength of that evidence — it never grants
eligibility by itself. Collapsing the two into one flag forces most of an
existing research corpus to be labelled with the more alarming word for the
wrong reason; separating them must not demote availability into a live-clock
concern. The distinction survives into reports and scores, and a score states
the weakest mode among the tables it consumed.

### 5.2 Event and quote identity

EarningsEvent fields: event_id, security_id, ticker_at_event, calendar_revision,
scheduled_event_date, session classification, actual_announcement_at if known,
session_source, confidence/conflict status, known_from, supersedes_revision.
A date move updates a revision rather than renaming the event. Ambiguous
quarter/event mapping refuses automatic reconciliation.

ChainSnapshot fields: chain_id, security_id, source_snapshot_ref, observation
time, availability/receipt times, session_date, quote convention, spot fields,
and contract rows with explicit rights, quantities/multipliers, expiries,
strikes, bid/ask, volumes/sizes, provenance and quality flags. Use null with
reason for unavailable liquidity; do not turn never-collected size into zero.

Contract queries specify allowable observation ceiling and quote policy.
Exact selected contracts remain in the score; a later quote cannot silently
re-ATM an already decided or opened position.

### 5.3 Finality and coverage

FinalityReceipt includes requested/resolved session, policy version, required
sources, publication evidence, expected/supported/received populations by table,
exact-session coverage shares, quarantined keys, and pass/fail reasons.

Preserve the existing finality policy in the legacy adapter. In v1, make its
denominators explicit and separately monitor coverage shrinkage. A global
coverage pass does not imply every requested event has final quotes: each
score and settlement validates its own required keys. Staleness tolerances for
display are distinct from finality required for recording decisions/outcomes.

## 6. Feature engine -> training and scoring

```text
FeatureRecipe:
  recipe_id, version, implementation_hash
  input_contracts, dependency_recipe_ids
  output_columns: ordered list[name, type, unit, nullable, null_policy]
  source_scope, history_scope, lookback_and_invalidation_rule
  observation_cutoff_rule, label_availability_rule
  supported_clock_contracts, fallback_policy, determinism_policy

FeatureRequest:
  event_refs, decision_contexts, snapshot_ref, live_snapshot_ref?
  feature_recipe_refs, upstream_prediction_refs

FeatureEngine.build(FeatureRequest) -> FeatureFrame
FeatureFrame:
  frame_ref, schema_ref, row_keys_ref, ordered_columns
  recipe_refs, dependency_refs, values_hash, null_mask_hash
  per_feature_lineage_ref, coverage_receipt, causal_audit_receipt
```

A feature lineage entry records output row/column, recipe ID, source row IDs,
observation and availability ceilings, historical population, any upstream
model/forecast IDs, and a fallback reason if used. Large lineage can be stored
by shared block references; do not duplicate millions of source rows per score.

Serving and dataset construction call this interface with the same recipe.
There can be optimized batch and single-row implementations only if parity
tests prove values, nulls, provenance and eligible populations agree.

Namespaced recipe IDs distinguish `bucket_analogs.v1` from
`chooser_knn_analogs.v1`. An artifact binding can map either output to the
legacy column `analog_mean`, but it must declare which. Context planning
separates watchlist state from the broad historical analog/training universe.

A missing required feature returns a typed failure/refusal; an optional
diagnostic can be unavailable. Preserve any existing sanctioned fallback via
its named policy rather than silently installing a generic default. Feature
materialization is keyed by exact dependencies, clock and recipe versions.

For an existing intentional training/serving approximation, register a
ServingApproximation: training quantity, serving recipe, justification/evidence,
coverage, error characterization and allowed use. The chooser n_admissible
surrogate is one current example. Validate exact reproduction of that mapping
and report its difference from the training quantity. An undeclared mismatch
remains a failure; this is not a general tolerance exemption.

**Acceptance:** rebuild a champion training row and its serving row with the
production context planner; compare consumed values, masks and lineage. Poison
a future observation/label and verify that the score is unchanged or refused.

## 7. Registration -> scoring

### 7.1 StrategySpec

```json
{
  "schema_version": "strategy_spec.v1.0",
  "strategy_id": "STR-THRU",
  "strategy_version": "legacy-68150a1",
  "definition_hash": "sha256:example",
  "structure_recipe": "legacy.straddle_through.v1",
  "structure_parameters": {},
  "component_graph_ref": "legacy.str_thru_components.v1",
  "clock_contract": "legacy.entry_close.v1",
  "entry_policy": "legacy.str_thru_entry.v1",
  "exit_policy": "legacy.str_thru_exit.v1",
  "quote_policy": "legacy.str_thru_quotes.v1",
  "fill_policy": "legacy.fill_alpha.v1",
  "universe_policy": "legacy.str_thru_universe.v1",
  "domain_policy": "legacy.str_thru_domain.v1",
  "feature_recipes": ["legacy.str_thru_features.v1"],
  "forecast_sizing_recipe": null,
  "analog_recipe": "legacy.bucket_analogs.v1",
  "payoff_recipe": "legacy.str_thru_payoff.v1",
  "gate_recipe": "legacy.str_thru_gate.v1",
  "chooser_recipe": null,
  "risk_policy": "legacy.str_thru_risk.v1",
  "fallback_policy": "legacy.str_thru_fallbacks.v1",
  "legacy_status_ref": "baseline_status_example",
  "evidence_refs": ["baseline_evidence_example"]
}
```

Every referenced legacy policy is exported from the baseline source/config,
including constants and selectors. The example is not a substitute for those
resolved definitions. No empty policy identifier may mean use a convenient
current default. Unknown parameters or undeclared dependencies fail registration.

`component_graph_ref` resolves the template, placement domain/generator,
scenario source and mapping, valuation/accounting/execution policies and
candidate selector used by the strategy, with explicit nulls for unused
components. The graph and the existing policy fields must resolve to the same
canonical definition; contradictory bindings fail registration. Its detailed
[registration contract](structure_generation_and_simulation.md#7-integration-and-acceptance-tests)
keeps exhaustive placement searches and new simulation methods separate from
the legacy selector and scoring definitions.

Validation status is inherited from the baseline, including disabled versus
tracked versus promoted. Production refuses CAL-P/CND-P as before. Research
registration can allow mechanical replay without enabling their production
scoring. DYN-SV registers a meta-strategy with its ordered candidate menu,
ranking, missing-candidate, tie and fallback contracts, not a fictitious leg list.

### 7.2 DeploymentSpec

DeploymentSpec fields: deployment_id, strategy_spec_ref, clock_contract_ref,
model_role_bindings, feature_contract_bindings, evidence_state_refs,
validation_receipt_refs, effective_from, mode, promotion_receipt_ref.
Model bindings name exact releases/artifacts; wildcard event models resolve
explicitly before the request starts. Clock-qualified champions are unique per
strategy/role/clock. No close-model fallback for a missing pre-close champion.

```text
Registry.register_candidate(namespace, spec, implementation_refs)
  -> CandidateRegistrationReceipt
Registry.resolve_deployment(strategy_id, clock_id, at) -> DeploymentSpec
Registry.promote(candidate_release, expected_current, validation_receipts)
  -> PromotionReceipt
```

Registration uses trusted installed recipe identifiers; an API caller cannot
submit arbitrary import paths or executable code. Experiment namespaces are
isolated from production. Promotion uses compare-and-swap and complete evidence.
Old deployments remain addressable for replay and rollback.

## 8. Training -> model releases

```text
DatasetRecipe:
  recipe_ref, feature_recipe_refs, target_contract
  universe_policy, eligibility_mask, grouping_keys
  sample_weight_rule, label_availability_rule, split_policy

ModelRecipe:
  recipe_ref, model_role, target_contract, dataset_recipe_ref
  ordered_feature_bindings, preprocessing_recipe
  estimator_adapter, hyperparameters, seed_policy
  fold_policy, validation_policy, residual_and_calibration_policy
  compatible_clock_contracts, upstream_model_recipes

TrainingJob.run(ModelRecipe, SnapshotRef, FoldPlan) -> ModelReleaseCandidate
Inference.predict(ArtifactRef, FeatureFrameRef) -> PredictionFrame
```

Training artifacts carry weights, transforms, ordered feature contracts,
training membership/hash and label cutoff, fold identity, upstream artifacts,
runtime/dependency fingerprint, residual pools, calibration state and metrics.
Artifacts are immutable; a refit creates another ID.

PredictionFrame carries row keys, target/unit, point/interval outputs,
model artifact ID, input-frame hash, fold and residual-state IDs. It distinguishes
OOS fold outputs from final-refit diagnostic outputs. Training a downstream
gate requires OOS upstream forecasts, not a full-fit prediction column.

FoldPlan is explicit: training/validation/test membership, cutoffs, target
availability, event grouping and any purge/embargo. A model trained today can
be used in a historical research simulation if fitted only on allowed past
data, but it is not labeled as the model actually deployed back then.
Observed replay requires the original deployment and availability evidence.

ModelRelease requires ArtifactManifest, DatasetManifest, OOS predictions,
model evidence, generated evaluation/report references, parity and causal
receipts, and compatible strategy/clock bindings. Evidence is made from the
same frozen training frame, not rebuilt later from latest data.

Inference never fits. Missing required artifact yields MODEL_NOT_READY. NN,
linear, tree and blend adapters share the interface but preserve existing
normalization, missing policies, targets, folds and seeds. Geometry sizing
and capital allocation stay outside estimator implementations.

**Acceptance:** load/save prediction parity; bad hash/order/clock refused;
upstream leakage poison refused; promotion cannot expose an artifact without
matching evidence; rollback reproduces the former model and its diagnostics.

## 9. Scoring -> every consumer

### 9.1 DecisionContext and snapshot bundle

DecisionContext is resolved before feature access:

```text
DecisionContext:
  clock_contract_ref, event_ref, exchange_calendar_version
  data_cutoff_at, completed_eod_ceiling
  planned_entry_session, planned_exit_session
  decision_deadline_at, execution_deadline_at
  source_availability_policy, knowledge_mode

SnapshotBundle:
  bundle_id, eod_snapshot_ref, live_snapshot_ref?
  feature_state_refs, model_release_refs, analog_state_refs
  residual_state_refs, payoff_state_refs, calibration_state_refs
  dependency_manifest_hash
```

Cutoffs describe the information allowed into a score; operational timestamps
describe when collection and computation really occurred. Event-history
availability, source observation cutoff, and completed-session ceiling are
separate constraints. The stricter applicable constraint wins. Legacy EOD
research retains its historical convention without being mislabeled as an
executable pre-close decision.

### 9.2 ScoreRequest

This example requests a hypothetical pre-close shadow deployment, not the
current close-clock champion:

```json
{
  "schema_version": "score_request.v1.0",
  "event_ref": {
    "event_id": "event_example",
    "calendar_revision": "calendar_revision_example"
  },
  "deployment_id": "str_thru_preclose_shadow_example",
  "decision_context_ref": "decision_context_example",
  "snapshot_bundle_ref": "snapshot_bundle_example",
  "mode": "shadow",
  "fill": {
    "policy_id": "legacy.fill_alpha.v1",
    "alpha": 0.5
  },
  "overrides": {
    "strike": null,
    "expiry": null,
    "structure_parameters": null,
    "contract_ids": null
  }
}
```

Fields above are required, including explicit nulls for unused override
categories. Mutually incompatible overrides are rejected. A pinned contract
cannot simultaneously ask the selector to choose a different strike. Any
accepted override is carried in the resolved request and scored under the
existing extrapolation/domain rules.

```text
Scoring.score(ScoreRequest) -> ScoreRecord
Scoring.score_event(EventScoreRequest) -> ScoreBatch
Scoring.score_many(ordered_requests) -> ScoreBatch
Scoring.replay(score_id) -> ReplayReceipt
```

EventScoreRequest contains the common event/context/snapshot plus an explicit
ordered deployment list and fill/override policy. It resolves all required
DYN-SV candidates internally. ScoreBatch includes a manifest, request count,
score IDs, refusal counts and per-request failures. Every requested strategy
gets a score/refusal or explicit failed task, never a silently missing row.

The kernel performs no network call, training, registration, ledger write or
publication. It receives repositories and pinned artifacts through adapters.
An application job can collect inputs or commit outputs around it, but those
steps remain independently visible and retryable.

### 9.3 ScoreRecord fields

| Group | Required fields |
|---|---|
| Identity | schema_version, score_id, payload_hash, request_hash, canonical_request, resolved_request, dependency_manifest_ref |
| Event/time | event_ref, security/ticker, session, clock/context, planned entry/exit and evidence cutoff |
| Inputs | snapshot bundle, feature frame, consumed ordered values/masks, source/quote refs and per-feature lineage |
| Models/state | exact model roles (`model_role`) and artifacts, forecast folds, residual pools, analog population, payoff/calibration state and recipe versions |
| Geometry | requested/resolved parameters, selected contracts, legs/quantities/multipliers, forecast-sizing explanation |
| Reusable domain artifacts | generation request/candidate-set manifest and completeness, selected candidate/position, scenario-set and simulation-result refs, exact valuation/accounting policy refs; explicit nulls if unused |
| Prices | entry quote date/time, estimated entry cost, normalized spot and conventions, fill alpha, per-leg spread/quality |
| Estimates | named forecast/model/simulation/analog outputs, target/unit, uncertainty and sample sizes |
| Decision | gate type, score, threshold, terms, verdict, chooser candidate audit/selection, sanctioned fallback |
| Diagnostics | premium ratios, fair value, model/market convention, payoff references, domain flags and evidence links |
| Validity | validation status/receipts, eligibility/readiness at computation, reason codes, freshness expiry |

Money has explicit debit/credit conventions and multipliers; the legacy
positive-debit `entry_cost` mapping remains documented. Expected return fields
declare their denominator: premium, secured capital or another registered
base. No generic expected_pnl field may ambiguously switch between dollars
and return fraction.

Forecasts and analog estimates remain distinct outputs; disagreement is a
flag, not an averaging instruction. A simulated exit value can inform a gate,
but realized PnL must point to actual traded/quoted outcome evidence. Preserve
the ban on oquants fitted marks as a realized PnL source.

### 9.4 Verdict, readiness and failure are distinct

```json
{
  "gate": {
    "kind": "regression_threshold",
    "verdict": "unavailable",
    "score": null,
    "threshold": null,
    "score_unit": "return_fraction_on_premium",
    "reason_codes": ["MISSING_LIVE_FEATURE"]
  },
  "readiness": {
    "state": "shadow",
    "actionable": false,
    "valid_until": null,
    "reason_codes": ["CLOCK_NOT_PROMOTED", "MISSING_LIVE_FEATURE"]
  },
  "validation": {
    "status": "refused",
    "receipt_refs": ["validation_receipt_example"]
  }
}
```

This is a fragment illustrating refusal semantics, not a complete ScoreRecord.
Gate verdict is pass, fail or unavailable. Rule-based gates additionally
return each registered term and its pass/fail/missing status. Readiness states
are ready, preview, shadow, historical, stale, blocked or superseded. Validation
states are passed, refused or failed. A contract-invariant failure is failed
and cannot become a validated decision; an honest missing-input result is
refused and remains visible.

A passing gate does not override stale quotes, a missed deadline or an
unvalidated deployment. Conversely, a stale display does not rewrite the
original frozen gate verdict. Persist readiness at computation and valid_until;
current actions recheck freshness through the same readiness policy. The UI
can label a record expired but cannot turn it back into ready.

Important codes include UNVALIDATED_STRUCTURE, OUT_OF_DOMAIN, NO_CHAIN,
BAD_QUOTE, COARSE_LADDER, NO_FORECAST, MISSING_FEATURES,
MISSING_LIVE_FEATURE, CLOCK_MISMATCH, MODEL_NOT_READY,
SNAPSHOT_NOT_FINAL, SOURCE_TOO_OLD, SOURCE_AFTER_CUTOFF,
SOURCE_TIME_SKEW, DECISION_DEADLINE_MISSED and PARITY_FAILED.
Retain legacy flags and their meanings through a compatibility mapping;
these names do not replace already distinct fallback/refusal cases.

DYN-SV includes every offered candidate score/refusal, ranking eligibility,
chosen strategy, runner-up, margin and fallback reason. It inherits the
winner gate verdict. Preserve the actual baseline tie and missing-score
implementation even where old comments describe it differently.

### 9.5 Score identity and replay

The canonical input key hashes the resolved request, strategy/deployment,
snapshot dependencies, feature/model/state recipes, fill and RNG policy.
The payload hash additionally proves the resulting numerical content.
Generated timestamps and progress logs do not participate.

Persist the exact replay request independently of display projections.
Selecting contracts from a forecast and replaying with those parameters pinned
must still record the same forecast. A client must use the saved request or
score ID, never rounded values copied from a table.

ReplayReceipt compares exact IDs, selected contracts, quantities, flags,
verdicts and null masks; numerical deltas use registered per-field tolerances.
It reports stage/field/expected/actual/source/model differences, coverage and
hashes. Running twice in the same process is insufficient: include fresh
process, reordered inputs, batch/single and serialized round-trip cases.

## 10. Experiment -> model/scoring engine -> evaluation

```text
ExperimentSpec:
  experiment_id, immutable_spec_hash, preregistered_at
  hypothesis, primary_arm_id, secondary_arm_ids
  candidate_registration_refs, strategy_and_model_recipe_refs
  snapshot_refs, clock_contracts, universe_and_sample_policy
  fold_plan_ref, seeds, price_source_contract
  fill_sweep, stress_policy, funding_policy, promotion_target?

ExperimentPlan:
  spec_ref, resolved_recipes_and_inputs
  job_graph_ref, expected_coverage, estimated_provider_calls
  checkpoint_keys, output_namespace, ledger_mode

EvaluationInput:
  spec_ref, selected_score_refs, decision_trace_ref
  realized_trade_frame_ref, price_evidence_refs
  fold_membership_ref, selection_receipt, provenance_manifest

Evaluation.evaluate(EvaluationInput) -> EvaluationReceipt
```

All economic selection details resolve to registered recipes before execution.
An experiment can define a new candidate implementation, but that implementation
lives behind the scoring contract and must be accessible to both research and
serving. Registration is isolated and does not require promotion.

SelectionReceipt maps the full offered universe to selected/refused candidates,
with the exact score, gate and policy responsible. RealizedTradeFrame maps each
selected intent to named entry/exit contracts and actual quoted/traded prices,
with fill alpha, returns, costs and skip reasons. Evaluation cannot substitute
model prices for realized outcomes or reconstruct a different gate.

Separate two fill-sensitivity policies explicitly: fixed-selection repricing
and recomputing selection at each alpha. Both can be useful but answer different
questions. Preserve the current primary policy during migration and label each
reported sweep; no hidden reselection changes the evaluated sample.

EvaluationReceipt includes generated REPORT.md, figures/results, sample funnel,
fill sweep/breakeven alpha, stress, capacity/funding, calibration, OOS metrics,
accuracy checklist, spec/input/code fingerprints and multiple-testing context.
Call the existing evaluate/write_report path. Model-only studies retain their
model metrics and identify the implied book required for economic evaluation.

Unique hypothesis identity and retry identity are separate. A new primary/grid
configuration gets a distinct evaluated specification and actual new scores;
a retry of identical work does not add a hypothesis. `ledger_mode=smoke`
corresponds to --no-ledger and cannot commit a real test result. Failed
evaluations are retained as failures, not fabricated metric rows.

Experiment completion requires the report, artifact manifest and ledger receipt,
then one final private-mirror sync receipt. Backup failure is explicit and
retryable without rerunning the experiment or duplicating its ledger entry.

## 11. Entrypoints -> job supervisor

### 11.1 Commands and resources

```text
Supervisor.submit(JobSpec, idempotency_key) -> JobReceipt
Supervisor.get(job_id) -> JobReceipt
Supervisor.cancel(job_id, expected_attempt) -> CancellationReceipt

JobSpec:
  kind, implementation_ref, spec_hash, environment_ref
  input_refs, dependency_job_ids, output_namespace
  resource_class, priority, deadline_at?
  provider_budget_ref?, retry_policy_ref, checkpoint_contract_ref

ResolvedResources:
  effective_host_budget, reserved_memory_bytes
  assigned_cpu_ids, thread_count, scratch_limit_bytes
  executor_mode: cgroup | watchdog
  provider_leases, resource_profile_version
```

Ordinary callers select a resource class; the supervisor resolves memory and
CPU placement. Resource overrides require an operator policy and still cannot
exceed global admission. HTTP job kinds map to trusted entrypoints, not shell
commands supplied by the browser.

JobReceipt carries job/spec IDs, state, attempt ID, lease/fence, resolved inputs,
resources, completed checkpoints, output refs, progress and failure envelope.
Submitting the same idempotency key with the same payload returns the same
job. Reusing it with a different payload returns IDEMPOTENCY_CONFLICT.

### 11.2 State and retry rules

| State/event | Required transition or effect |
|---|---|
| submitted | queued; no worker until dependencies and reservations are satisfied |
| claim | running attempt with exclusive lease/fence and reserved resources |
| transient source failure | retry_wait with bounded backoff and retained checkpoints |
| credential/integrity failure | failed or blocked dependency; no automatic hammering/repeated publication |
| worker loss/expired lease | fence old attempt, reconcile process tree, then queue a resumable attempt |
| cancel | cancelling until worker exits; invalidate its fence before accepting cancellation completion |
| commit outputs | succeeded only after durable output/validation receipts |
| exhausted retries | failed with reason and diagnostic refs |

Workers heartbeat and emit ProgressEvent: job/attempt/stage, completed/total
units, elapsed time, current/peak memory, checkpoint, ETA when meaningful,
and latest error. At least one heartbeat per minute even during a long fit.
Events are structured, redacted and durably recorded, not only buffered stdout.

Output commits require a current fence. Fencing prevents an old worker from
publishing after lease takeover, but admission must also ensure that orphaned
process memory is released before admitting replacement heavy work.
No transaction remains open during network calls, fitting or long file writes.

**Acceptance:** two submitted heavy jobs queue safely, no overlapping CPU
allocation, resource limits include child processes, crash resumes exact
checkpoints, and duplicate attempts cannot commit duplicate effects.

## 12. Scoring and settlement -> append-only ledger

### 12.1 Prediction commit

```text
DecisionLedger.commit(
  ValidatedDecision, validation_receipts, expected_deployment, idempotency_key
) -> DecisionReceipt

ValidatedDecision:
  decision_id, score_id, event_ref, deployment_id, clock_context_ref
  purpose: production | shadow | research_reconstruction
  verdict, created_at, decision_deadline_at
  quote_and_feature_provenance_refs, validation_receipt_refs
  supersedes_decision_id?, supersession_reason?
```

A transaction checks the score/deployment/snapshot compatibility, required
validations, finality or live cutoff, deadline and unique decision identity.
Keep production, shadow and reconstructed research records distinct. Historical
backfill records its real creation time and cannot become an on-time decision.

Define unique logical identity from event, strategy/deployment, clock and
scheduled decision occurrence. A retry of the identical decision is idempotent.
A new snapshot during an explicit rescore is another immutable score; replacing
an official decision requires a superseding record and reason. Settled/opened
positions are not silently reselected by that supersession.

Commit validated predictions and release/outbox references in one short
catalog transaction. Compatibility JSONL exports use ledger sequence numbers
and are recoverable projections; they are not a second authoritative writer.
During migration choose one authority explicitly and reconcile exports.

### 12.2 Position lifecycle

```text
PositionEvent:
  ledger_sequence, event_type, position_id, decision_id
  event_idempotency_key, occurred_at, recorded_at
  selected_contracts_ref, quantity, order_policy_ref?
  evidence_kind, evidence_refs, cashflows?, supersedes_event_id?
```

Event types include intent_recorded, order_submitted, fill_recorded,
entry_completed, exit_fill_recorded, exit_completed, cancelled,
settlement_blocked, settlement_resolved and correction_recorded.
Partial fills remain explicit. Evidence_kind distinguishes hypothetical,
paper, quoted-replay and broker-confirmed activity. These must never merge
into an undifferentiated real-fill label.

Pricing a hypothetical open/close from later EOD data produces the existing
quoted/paper accounting result, not broker execution proof. Preserve actual
entry repricing separately from decision-time cost estimates. Freeze contract
identity according to the current execution policy; later spot movement is
not permission to change strikes.

The book is a deterministic projection of ledger events plus a named funding
policy. It must reconcile positions, cashflows, capital constraints, fees and
missing outcomes. Unresolved trades remain visible. Refuse invalid transitions
such as closing more than opened or settling from non-final required data.

**Acceptance:** kill between decision and publication, resume without duplicate
prediction; reopen/settle a saved book identically; revisions preserve old
records; paper fills never appear as broker fills.

## 13. Serving projections and publication -> UI

### 13.1 Release contract

```text
ServingRelease:
  release_id, schema_version, payload_manifest_hash
  score_batch_refs, event_index_ref, model_evidence_refs
  portfolio_projection_ref, portfolio_ledger_sequence
  data_snapshot_refs, deployment_refs, validation_receipt_refs
  requested_as_of, resolved_as_of, clock_ids
  completeness, coverage_summary, stale_or_degraded_reasons
  created_at, validated_at, published_at?
```

A release pins its event list, scores, models/evidence and book projection.
Optional diagnostics may be unavailable with a reason. A different champion
evidence table cannot be substituted under that release identity. Operations
health is a separate current feed, allowing a failed latest build to be shown
beside the last good data release without altering its provenance.

Publication writes immutable release objects, verifies hashes/access rules,
then switches the current pointer atomically. Remote delivery is an idempotent
outbox effect. Track validated and published watermarks separately. Retain the
prior release until rollback/retention rules permit removal.

### 13.2 Read API

| Method/route | Input | Output |
|---|---|---|
| GET /api/v1/releases/current | authorized principal | ServingRelease summary |
| GET /api/v1/events | release_id, date range, filters, sort, limit, cursor | Page of EventSummary |
| GET /api/v1/events/{id}/scores | release_id, clock filter | Strategy score summaries |
| GET /api/v1/scores/{id} | immutable score ID | ScoreRecord/detail refs |
| GET /api/v1/scores/{id}/payoff | named domain payoff view | Frozen chart values/convention |
| GET /api/v1/models/{release_id}/evidence | exact model release | Frozen model evidence |
| GET /api/v1/portfolio | release_id, filters, cursor | Position page and matching full-filter aggregates |
| GET /api/v1/jobs/{id} | job ID | JobReceipt |
| GET /api/v1/operations | authorized principal | Current operational health |
| POST /api/v1/score-jobs | request + idempotency key | 202 JobReceipt |
| POST /api/v1/live-snapshots | collection contract + idempotency key | 202 JobReceipt |

Read routes never initiate provider pulls or training. A missing expensive
payoff view returns its unavailability/preparation status; a separate command
can request domain computation. Ordinary chart formatting is not rescoring.
The current-release route resolves once; subsequent page/detail calls pin it.

```json
{
  "schema_version": "event_page.v1.0",
  "release_id": "release_example",
  "query_hash": "sha256:example",
  "items": [
    {
      "event_id": "event_example",
      "calendar_revision": "calendar_revision_example",
      "ticker": "EXAMPLE",
      "event_date": "2026-09-14",
      "session": "AMC",
      "score_summary_ref": "event_scores_example",
      "available_clocks": ["legacy.entry_close.v1"],
      "readiness": "preview"
    }
  ],
  "next_cursor": null,
  "total_matching": 1
}
```

Pagination defaults to 50 and caps at 200 items; these are proposed API limits,
not strategy filters. Cursor content is opaque, integrity-protected and bound
to release, filters, sort and a unique final tie-breaker. A cursor from another
release/query returns CURSOR_MISMATCH. Filtering and aggregate summaries use
the same complete population, not just the currently visible page.

Authenticated immutable objects return ETags. Cache scope includes principal
where visibility differs; protect underlying detail/object URLs too. UI caches
include release, event, clock and filters. A stale object can be displayed
with its timestamp, but cannot enable a live action past valid_until.

### 13.3 Offline adapter

```text
DataClient.get_release()
DataClient.list_events(query)
DataClient.get_event_scores(event_id, release_id)
DataClient.get_score(score_id)
DataClient.get_model_evidence(model_release_id)
DataClient.get_portfolio(query)
DataClient.capabilities() -> {read, offline, submit_jobs, collect_live}
```

HTTP and offline implementations return the same schemas. Offline packages
pin a release and list included objects/coverage, with read=true and live/job
capabilities=false. An all-in-one export is allowed to embed that pinned data;
the ordinary frontend build contains no market data. Browser tests exercise
network-disabled exports and ensure no action claims to refresh live data.

## 14. Live collector -> snapshot assembler -> scoring

```text
LiveCollectionRequest:
  event_refs, clock_contract_ref, completed_eod_snapshot_ref
  required_source_contracts, collect_window, data_cutoff_at
  decision_deadline_at, provider_budget_ref

LiveSnapshot:
  snapshot_id, event_refs, clock_context_ref, eod_snapshot_ref
  raw_receipt_refs, normalized_live_frame_ref
  per_source_and_leg_timestamps, quality_receipt
  field_provenance_matrix_ref, sealed_at
```

Snapshot assembler verifies every required input against the registered source
contract. Required fields include their source, unit, freshness ceiling,
observation/availability cutoff and maximum cross-source skew. Values such as
prior-event history and model predictions are not treated as fields returned
by a quote endpoint.

For observed live decisions, underlying source observations and receipts must
be at or before data_cutoff_at. Processing/sealing may finish later, but must
meet the registered decision deadline. Re-fetching after cutoff creates a
different observation or late diagnostic; it cannot repair the original clock
retroactively. A revised raw response creates another snapshot ID.

Early collection and a bounded skew window are required because summary and
chain responses are not one atomic vendor transaction. Refuse a mixed snapshot
outside the policy. Never insert provisional live values into final EOD tables.
The scorer receives the sealed feature/source snapshot and cannot make a
network call to improve it while computing.

Each deployment declares which live features its models support. Missing
provenance, incompatible clock, stale data or an early-close timing conflict
returns an explicit refusal. All existing strategies remain addressable, but
those without validated live deployments can only produce the appropriate
unavailable/shadow result. A UI toggle cannot create live model compatibility.

## 15. Comparators -> diagnosis

### 15.1 Why this is a contract and not a helper

At least six components compare two things and report whether they agree: the
nightly self-check, serving/replay parity, training/serving parity,
incremental-versus-full equality, the causal-source audit and the generator's
scalar-equivalence test. Each will otherwise invent its own report shape.

The current `engine/dashboard/selfcheck.py` returns `mismatches` as a list of
`{row_id, reason}` truncated to ten entries. On 2026-09-11 that surfaced one
red signal — ten of twenty sampled rows — with five independent causes behind
it: a forecast blanked when a pinned shape suppressed `_size_from_forecast`, an
explainer comparing 41 of the 70 fields the digest hashes, an analog bootstrap
sampling by index over an unordered set, `json_safe` rounding a replay input to
six places, and `_write_pair` re-rounding it after the exemption had already
been applied. Each fix was correct and the signal stayed red after every one,
because the report could express "different" but not "differently, in these
five independent ways".

Two properties are therefore required of every comparator, and they are what
this contract exists to enforce:

1. **Stage-localized.** A finding names the first stage at which inputs agreed
   and outputs did not, with per-stage input and output hashes. "Row 47 is red"
   is not a diagnosis; "serialization: `structure_params.width_moneyness`
   differs at the 7th significant figure" is.
2. **Complete, not first-wins.** A comparator reports every independent finding
   it can establish in one pass. Stopping at the first difference converts N
   causes into N runs, which is the failure this contract is written against.

### 15.2 ComparisonReceipt

```text
Comparator.compare(ComparisonRequest) -> ComparisonReceipt

ComparisonRequest:
  schema_version, comparison_kind, tier
  left_ref, right_ref, stage_plan_ref
  tolerance_policy_ref, population_expectation
  budget_ref

ComparisonReceipt:
  schema_version, receipt_id, comparison_kind, tier
  left_ref, right_ref, tolerance_policy_ref
  stage_hashes: ordered [stage_id, left_input_hash, left_output_hash,
                         right_input_hash, right_output_hash, agrees]
  findings: [Finding]
  population: expected, supported, compared, skipped_with_reasons
  verdict: agree | differ | incomparable
  envelope: started_at, duration, worker_ref, diagnostic_ref

Finding:
  finding_id, first_differing_stage, field_path
  left_value, right_value, delta, unit
  tolerance_applied, exceeded_by
  null_mask_left, null_mask_right
  source_rows_ref, recipe_and_artifact_versions
  affected_count, independent_of: [finding_id]
```

`findings` is a set, not a first-difference. `independent_of` records which
findings the comparator proved are not consequences of one another, so an
operator can fix several at once; findings it could not separate are reported
without that link rather than silently merged.

The compared field set is derived from the identity being checked, never
maintained by hand. A comparator whose digest covers 70 fields compares 70
fields. `population` prevents the reciprocal failure: a comparison over an
empty or collapsed set reports `incomparable`, never `agree`.

`verdict: incomparable` is a first-class outcome. A missing artifact, an
unresolvable snapshot or a zero-row population is not agreement.

### 15.3 Tiers and the inner loop

Every comparator declares a tier, and the tier is a latency commitment. A check
that can only run nightly cannot be part of the loop by which a fix is
confirmed, and the whole point of this section is that confirming a fix must
not cost a nightly.

| Tier | Budget | Inputs | Runs on |
|---|---|---|---|
| 0 | seconds | Frozen ScoreRequests to expected ScoreRecords, from fixtures. No panel load, no network, no fitting, no disk beyond the fixture | Every edit |
| 1 | ~a minute | One event end to end, real chain, including a real serialize-to-disk and read-back | Every commit |
| 2 | nightly | Full board, full parity matrix, incremental-versus-full equality | Nightly, and every migration step |

Tier 0 must be reachable without loading the panel or any model artifact
larger than the fixture pins, or it will not be run. Tier 1 must write to and
read from an actual file, because a round-trip loss through serialization is
invisible to any check that keeps the object in memory. Tier 2 remains the
authority; the lower tiers are a fast path to being wrong less often, not a
replacement for it.

Every check in the architecture guide's validation table names its tier. A new
check with no tier defaults to 2, which is a statement that it will not
participate in the inner loop, and should be treated as a gap to close rather
than a neutral choice.

**Acceptance:** a fixture corpus seeded with each of the five 2026-09-11 causes
produces five findings in one Tier-0 run, each naming its stage, and not one
finding that names only a row. Removing a field from the digest removes it from
the comparison automatically. An empty population reports `incomparable`. A
serialization round-trip regression is caught at Tier 1 and missed at Tier 0,
and the tier table says so rather than the test being deleted as flaky.

## 16. One end-to-end consistency example

For a synthetic Monday AMC event:

1. Ingestion commits final prior-session EOD snapshot S and calendar revision C.
2. A live collection job reserves quota/capacity and seals eligible pre-cutoff
   observations as L. The pre-close artifact release M was prepared earlier.
3. Scoring pins S, C, L, M and the strategy version, and creates score Q.
   Missing required fields produce a refusal instead of Q becoming ready.
4. Validation checks causal lineage and parity. A shadow deployment records a
   shadow decision only; a promoted production deployment can commit a
   validated decision if the deadline and other controls permit it.
5. A transaction creates decision D and release R plus publication outbox P.
   A network failure delays P without modifying D. The UI keeps prior release
   R0 and shows the delivery failure from operations health.
6. After retry, R becomes current. Its event page and every detail fetch use R.
7. A later vendor correction creates S2 or L2. Q and D still replay from their
   original references. A rescore creates Q2; it cannot change an open position.
8. Settlement records an outcome using its explicit price/fill evidence.
   Model and fill-quality evaluation joins that outcome to the frozen decision.

For a BMO event, the pre-print decision belongs to the prior trading session;
the same flow cannot be shifted to Monday afternoon without changing the
strategy timing. For an EOD historical replay, knowledge_mode and deployment
selection explicitly identify its reconstruction assumptions.

## 17. Contract implementation and review order

Review the ScoreRequest/ScoreRecord and time contracts first, then the
ComparisonReceipt and its tiers, then registry bindings, storage transactions,
jobs and ledger commits. These contain the choices that determine whether two
components can disagree silently — and the ComparisonReceipt is what determines
whether such a disagreement can be named when it happens.

Implement schema definitions in one small contract package, with Python
validation and generated OpenAPI/TypeScript types. Keep numerical functions
outside it. Version stored schemas and maintain both provider and consumer
contract tests. Test JSON examples, decimal/full-precision round trips, unknown
fields/enums and old-version adapters mechanically.

Before accepting the contracts, require these demonstrations:

- One existing event/strategy traverses legacy -> contract adapter -> API ->
  replay without changing its decision, contracts, inputs or displayed values.
- One experiment registers a candidate, trains if needed, scores through the
  engine and evaluates with a generated report; production stays unchanged.
- One appended session and one corrected row produce the same result as a
  clean rebuild while retaining the old snapshot.
- A crash at commit, a duplicate job, a stale lease and a failed publication
  preserve consistent data and idempotent ledger effects.
- A mismatched analog recipe, clock, artifact or rounded geometry produces a
  precise failure that identifies the differing component.
- Five independent seeded defects produce five findings in one Tier-0 pass,
  each naming its stage, in under the tier's declared budget.

No contract is accepted merely because a schema validator can parse its JSON.
Its meaning, causal inputs, compatibility and failure behavior must be tested.
