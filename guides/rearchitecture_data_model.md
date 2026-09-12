# Rearchitecture data model

Date: 2026-09-12. Status: proposed logical model, not a deployed schema.
Companions: [architecture](system_rearchitecture.md),
[component contracts](component_contracts.md), and
[structure generation and simulation](structure_generation_and_simulation.md).

The three entity-relationship views below describe one model. Repeated entity
names refer to the same entity, not a copied table. Boxes show selected identity
fields, not every required contract field. `PK` means primary key; `FK` means
foreign key. The end markers mean exactly one (`||`), zero or one (`o|`),
zero or many (`o{`), or one or many (`|{`).
([Mermaid ER notation](https://mermaid.js.org/syntax/entityRelationshipDiagram.html))

These are logical relationships. Small manifests and indices belong in the
SQLite catalog; market facts, scenarios, feature matrices and detailed audit
rows belong in immutable objects/Parquet. Do not put every quote or simulation
draw into SQLite. Relationships to object-backed data are validated when a
manifest is committed, not assumed to be SQL-enforced foreign keys.

## 1. Events, contracts and reproducible market inputs

```mermaid
erDiagram
    SECURITY ||--o{ EARNINGS_EVENT : has
    EARNINGS_EVENT ||--|{ EVENT_REVISION : revised_as
    SECURITY ||--o{ OPTION_CONTRACT : underlies
    OPTION_CONTRACT ||--o{ QUOTE_OBSERVATION : quoted_as
    RAW_OBJECT |o--o{ RAW_RECEIPT : recorded_by
    RAW_RECEIPT ||--o{ QUOTE_OBSERVATION : sources
    SNAPSHOT_BUNDLE ||--o{ CHAIN_SNAPSHOT : resolves
    CHAIN_SNAPSHOT ||--o{ CHAIN_MEMBER : contains
    OPTION_CONTRACT ||--o{ CHAIN_MEMBER : listed_in
    QUOTE_OBSERVATION |o--o{ CHAIN_MEMBER : selected_quote

    SECURITY {
        string security_id PK
        string symbology_history_ref
    }
    EARNINGS_EVENT {
        string event_id PK
        string security_id FK
    }
    EVENT_REVISION {
        string event_revision_id PK
        string event_id FK
        timestamp scheduled_at
        timestamp known_at
        string source_lineage_ref
    }
    OPTION_CONTRACT {
        string contract_id PK
        string security_id FK
        decimal strike
        string right
        timestamp expiry_at
        decimal multiplier
        string deliverable_ref
    }
    RAW_OBJECT {
        string content_hash PK
        string object_ref
    }
    RAW_RECEIPT {
        string receipt_id PK
        string content_hash FK
        timestamp received_at
        string coverage_receipt_ref
    }
    QUOTE_OBSERVATION {
        string quote_id PK
        string contract_id FK
        string receipt_id FK
        timestamp observed_at
        timestamp available_at
        string quote_kind
    }
    SNAPSHOT_BUNDLE {
        string bundle_id PK
        string eod_snapshot_ref
        string live_snapshot_ref
        string dependency_manifest_hash
    }
    CHAIN_SNAPSHOT {
        string chain_snapshot_id PK
        string bundle_id FK
        string security_id FK
        string decision_context_ref
        string quote_policy_ref
        string coverage_ref
    }
    CHAIN_MEMBER {
        string chain_snapshot_id PK, FK
        string contract_id PK, FK
        string quote_id FK
        string availability_status
    }
```

Important constraints:

- A ticker is an alias, not a security, event, or option-contract primary key.
  Event rescheduling creates an event revision, not a different decision
  history. Corporate actions require explicit symbology/deliverable mappings.
- A chain snapshot can be reused across events for the same security/context;
  a generation request binds it to one event revision. Every candidate leg
  must belong to that frozen chain and the requested security.
- A listed contract can have no usable quote. Its chain-member quote link is
  nullable, with a reason. A missing quote must not erase the distinction
  between a structurally valid placement and a priceable one.
- Source observations and later corrections are immutable versions. A quote
  identity includes source, observation identity and revision, not just the
  contract/date. An old snapshot keeps the old revision.
- RawReceipt is the [RawReceipt contract](component_contracts.md#3-connector---ingestion).
  Failed/empty fetches may have no raw object; several receipts may reference
  identical content. Quote observations require a successful source receipt.
  Derived marks instead carry all input lineage and a valuation-policy ref;
  they are not mislabeled provider quotes or real trades.
- The snapshot bundle pins EOD and live inputs separately, plus feature/model
  state as specified in the scoring contract. Chain snapshots are derived
  views with their own manifests, not replacements for those inputs.

The incremental storage relationships are also explicit:

| Relation | Key and meaning |
|---|---|
| DataSnapshot | `snapshot_id`; committed, immutable set of table versions |
| SnapshotTable | `(snapshot_id, table_name)` -> one `dataset_version_id` |
| DatasetVersion | `dataset_version_id`; schema/normalizer version, coverage, parent version and change-set refs |
| VersionFragment | `(dataset_version_id, fragment_id)`; exact active fragment membership |
| Fragment | `fragment_id`; logical key range/content hash and immutable object hash/location |
| FragmentInput | `(fragment_id, receipt_id)`; many-to-many normalization lineage |

Each table version is a full logical view, even if its manifest reuses almost
all old fragments. Appends or bounded corrections replace only affected
members; no reader reconstructs an unbounded delta chain. A parent reference
is lineage, not permission to mutate the parent. Compaction changes physical
fragments while preserving validated logical content. Watermarks advance in
the same catalog transaction that exposes the new snapshot.

## 2. Templates, placements and simulations

```mermaid
erDiagram
    STRUCTURE_TEMPLATE ||--o{ GENERATION_REQUEST : instantiated_by
    EVENT_REVISION ||--o{ GENERATION_REQUEST : binds
    CHAIN_SNAPSHOT ||--o{ GENERATION_REQUEST : supplies
    GENERATION_REQUEST ||--o| CANDIDATE_SET : produces
    CANDIDATE_SET ||--o{ CANDIDATE_POSITION : contains
    CANDIDATE_POSITION |o--|| POSITION_DEFINITION : resolves_to
    POSITION_DEFINITION ||--|{ POSITION_LEG : contains
    OPTION_CONTRACT ||--o{ POSITION_LEG : identifies
    POSITION_DEFINITION ||--o{ POSITION_STATE : valued_from
    POSITION_STATE ||--o{ SIMULATION_REQUEST : evaluated_by
    SCENARIO_SET ||--o{ SIMULATION_REQUEST : drives
    SCENARIO_SET ||--o{ SCENARIO_SOURCE : traces
    SCENARIO_SET ||--|{ SCENARIO : contains
    VALUATION_POLICY ||--o{ SIMULATION_REQUEST : prices
    SIMULATION_REQUEST ||--o| SIMULATION_RESULT : produces

    STRUCTURE_TEMPLATE {
        string template_version_id PK
        string leg_rules_ref
        string geometry_constraints_ref
    }
    GENERATION_REQUEST {
        string generation_request_hash PK
        string template_version_id FK
        string event_revision_id FK
        string chain_snapshot_id FK
        string search_domain_ref
        string generator_recipe_ref
    }
    CANDIDATE_SET {
        string candidate_set_id PK
        string generation_request_hash FK
        string manifest_ref
        string completeness
    }
    CANDIDATE_POSITION {
        string candidate_id PK
        string candidate_set_id FK
        string position_definition_id FK
        string resolution_trace_ref
    }
    POSITION_DEFINITION {
        string position_definition_id PK
        string economic_fingerprint
        string reference_geometry_ref
    }
    POSITION_LEG {
        string position_definition_id PK, FK
        string leg_name PK
        string contract_id FK
        decimal signed_quantity
    }
    POSITION_STATE {
        string position_state_id PK
        string position_definition_id FK
        string market_state_ref
        string entry_basis_ref
        string position_event_cursor
    }
    SCENARIO_SET {
        string scenario_set_id PK
        string scenario_recipe_ref
        string knowledge_cutoff_ref
        string probability_kind
        string rng_policy_ref
    }
    SCENARIO_SOURCE {
        string scenario_set_id PK, FK
        string source_ref PK
        string source_kind
        string outcome_mapping_ref
    }
    SCENARIO {
        string scenario_set_id PK, FK
        string scenario_id PK
        decimal weight
        string market_path_ref
    }
    VALUATION_POLICY {
        string valuation_policy_id PK
        string implementation_ref
        string capability_manifest_ref
    }
    SIMULATION_REQUEST {
        string simulation_request_hash PK
        string position_state_id FK
        string scenario_set_id FK
        string valuation_policy_id FK
        string horizon_and_shocks_ref
        string accounting_policy_ref
    }
    SIMULATION_RESULT {
        string simulation_result_id PK
        string simulation_request_hash FK
        string outcome_summary_ref
        string valuation_audit_ref
    }
```

A PositionDefinition here is an immutable proposed leg set, not an executed
trade. Each candidate owns its definition record; equivalent exposures across
requests can share an economic fingerprint without collapsing different
resolution traces. A manually supplied simulated position can have a definition
without a candidate. Implement that parent link as nullable with a unique
constraint for generated definitions.

Zero-quantity reference geometry is retained separately from economic legs.
It cannot be erased before the template collision/shape checks. PositionState
adds the valuation time, spot/surface/quotes, opening basis when relevant, and
any already realized cash flows. It can represent a hypothetical trade or a
reconstruction of a real position at a ledger cursor.

ScenarioSource references a typed immutable historical-position/outcome pool,
OOF residual artifact, or synthetic distribution specification. It is not an
untyped link to whichever DataFrame a caller happens to have. Large source
membership, draws and per-leg valuations live in object-backed manifests.
Multiple candidates can reuse the same scenario set. Valid probability sets
must have at least one scenario; a refused construction has no set.

Completed generation/simulation content is immutable. Work-in-progress pages
and retry attempts belong to job staging; completing a previously interrupted
generation creates a committed manifest, not edits to a published partial one.
The detailed [reusable contracts](structure_generation_and_simulation.md)
define completeness, weights, units, shocks and refusal behavior.

## 3. Registered decisions, evidence and actual positions

```mermaid
erDiagram
    STRATEGY_VERSION ||--o{ DEPLOYMENT : configured_as
    DEPLOYMENT ||--o{ DEPLOYMENT_MODEL : binds
    MODEL_RELEASE ||--o{ DEPLOYMENT_MODEL : supplies
    MODEL_RECIPE ||--o{ MODEL_RELEASE : trained_as
    DEPLOYMENT ||--o{ SCORE_RECORD : governs
    EVENT_REVISION ||--o{ SCORE_RECORD : scored_as
    SNAPSHOT_BUNDLE ||--o{ SCORE_RECORD : pins_inputs
    SCORE_RECORD ||--o{ SCORE_COMPONENT : records
    SCORE_RECORD ||--o{ RELEASE_SCORE : published_in
    SERVING_RELEASE ||--o{ RELEASE_SCORE : includes
    SCORE_RECORD ||--o| VALIDATED_DECISION : freezes
    VALIDATED_DECISION |o--o{ ACTUAL_POSITION : authorizes
    ACTUAL_POSITION ||--|{ POSITION_EVENT : evolves_by

    STRATEGY_VERSION {
        string strategy_version_id PK
        string definition_hash
        string registered_component_graph_ref
    }
    DEPLOYMENT {
        string deployment_id PK
        string strategy_version_id FK
        string clock_contract_ref
        string mode
    }
    MODEL_RECIPE {
        string model_recipe_id PK
        string dataset_and_training_recipe_ref
    }
    MODEL_RELEASE {
        string model_release_id PK
        string model_recipe_id FK
        string artifact_manifest_ref
        string training_snapshot_ref
        string evidence_ref
    }
    DEPLOYMENT_MODEL {
        string deployment_id PK, FK
        string role PK
        string model_release_id FK
    }
    SCORE_RECORD {
        string score_id PK
        string deployment_id FK
        string event_revision_id FK
        string bundle_id FK
        string feature_frame_ref
        string decision_context_ref
        string payload_hash
    }
    SCORE_COMPONENT {
        string score_id PK, FK
        string role PK
        string object_ref
    }
    SERVING_RELEASE {
        string serving_release_id PK
        string manifest_hash
    }
    RELEASE_SCORE {
        string serving_release_id PK, FK
        string score_id PK, FK
    }
    VALIDATED_DECISION {
        string decision_id PK
        string score_id FK
        string ledger_idempotency_key
    }
    ACTUAL_POSITION {
        string position_id PK
        string decision_id FK
        string opening_definition_ref
    }
    POSITION_EVENT {
        string position_event_id PK
        string position_id FK
        string event_type
        string execution_evidence_ref
        timestamp effective_at
    }
```

ScoreComponent is a typed role binding, such as `generation`, `selected_position`,
`scenario_set`, `simulation`, or a namespaced child score in the DYN-SV menu.
Multiple results for a role use a manifest or distinct qualified role keys.
Commit validation enforces the expected object kind and content hash. These
bindings connect this view to the previous one without copying calculations.
Model releases/folds actually consumed are likewise pinned in the score; the
deployment binding alone cannot hide a later wildcard or monthly resolution.

StrategyVersion pins template, generator, scenario, valuation, accounting and
selection recipes in its registered component graph. The same template/model
release can be referenced by many strategy versions. A template is not a
strategy, and an optimizer is not allowed to silently replace its selector.

ModelRecipe/ModelRelease work for regressions, neural networks, size forecasts,
gates and chooser roles. Model artifacts and evidence remain separate from a
strategy definition; promotion creates a new deployment, not a rewrite of
old scores. Training examples, labels and fitted preprocessing state reference
their own versioned datasets and temporal splits.

An actual position may represent a declared imported holding without a decision;
this requires an import provenance event, never a fabricated score. System-created
paper/live entries require a validated decision. Ledger uniqueness and release
outbox rules remain those in [the ledger contract](component_contracts.md#12-scoring-and-settlement---append-only-ledger).
Repeated job attempts reference the same decision idempotency key; they do not
create repeated fills. Simulation outputs never create execution evidence.

## 4. Cross-entity checks required before implementation cutover

- An event revision, chain snapshot and clock must agree about security and
  information availability. Historical corrections cannot retroactively change
  a frozen decision.
- Every candidate has all expected leg roles and exact contract identities.
  Every simulation has the matching frozen position, scenario manifest and
  valuation/accounting policies; another candidate cannot borrow its results.
- Every published score resolves every required dependency to a retained object.
  A missing model/quote/outcome is an explicit refusal, not a dangling success.
- Dataset publication, score/release publication and ledger effects use their
  own documented transactions. A successful file write is not a publication.
- Restore and retention tests walk these relationships from a serving score or
  ledger decision all the way to raw inputs, recipes, models and scenarios.
  Anything needed for replay must survive garbage collection.
