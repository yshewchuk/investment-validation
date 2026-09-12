# Reusable structure generation and position PnL simulation

Date: 2026-09-12. Status: proposed contracts for design review.
Companions: [architecture](system_rearchitecture.md),
[component contracts](component_contracts.md), and
[data model diagrams](rearchitecture_data_model.md).

These are domain components inside the shared engine, not independent network
services or alternate scoring engines. They have no vendor I/O, registry writes,
training, ledger writes or UI dependencies. The scoring application orchestrates
them. An experiment registers its component graph before producing strategy
scores; a UI what-if request uses a supervised domain job and returns a clearly
non-actionable analysis artifact, never an alternate trading decision.

## 1. The reusable boundaries

| Component | Input -> output | Does not own |
|---|---|---|
| Structure generator | Event, frozen chain, template, finite search domain -> candidate placements and completeness receipt | Forecast fitting, PnL ranking, gates, promotion |
| Scenario builder | Eligible historical positions/outcomes or a specified synthetic distribution -> weighted market scenarios and lineage | Contract selection, pricing a position, trading verdicts |
| Position valuator | Frozen legs, market state, time, valuation policy -> per-leg and net position value | Choosing a distribution or redefining the position |
| PnL simulator/accountant | Position state, scenarios, horizon/shocks, valuation and accounting policies -> value-change/PnL distribution | Choosing the winning candidate or treating a modeled mark as a fill |
| Registered selector/gate | Candidate scores and full offered-set audit -> strategy decision | Implementing its own pricing or simulation formulas |

Extract and adapt the current behavior first:

- [structures.py](../engine/structures.py) already defines templates, expiry
  and strike selectors, reference legs, pinned closing legs, fill pricing and
  collision checks. It is the compatibility source for current placements.
- [EXP-133 family enumeration](../experiments/EXP-133_every_symmetric_put_structure_the_ladder/family.py)
  and [candidate pricing](../experiments/EXP-133_every_symmetric_put_structure_the_ladder/build.py)
  contain useful bounded search and per-contract simulation reuse. They are
  experiment-specific today. In particular, listed ladder offsets are not
  dollar widths; floor/symmetry checks must use the actual resolved strikes.
- [pnl_sim.py](../engine/pnl_sim.py) implements causal paired move/crush
  residual sampling and put valuation. [payoff.py](../engine/payoff.py) provides
  calibrated payoff/runup mappings, while [analogs.py](../engine/analogs.py)
  provides a different historical matcher. Preserve these distinct recipes.

Names below are proposed. `ObjectRef`, event identity, units, canonical hashes,
cutoffs and missing-value rules inherit the [shared contracts](component_contracts.md#2-shared-types-identity-and-compatibility).
All recipe references resolve to trusted installed implementations plus pinned
configuration, not arbitrary executable code supplied by an API client.

## 2. Structure generation: all valid placements, within a declared domain

### 2.1 Template is shape; search domain is freedom; selector is policy

A StructureTemplate declares named legs, rights, signed quantities, expiry
relationships, strike relationships and shape invariants. Its parameters can
include an anchor, width and spacing. Zero-quantity anchor/reference geometry
is retained without pretending it is an economic holding.

A PlacementDomain declares which of those parameters may vary and over which
finite set: eligible expiries, listed anchors, widths/ladder offsets and explicit
contract overrides. Defaults must be resolved into the request. A domain of
all listed anchors/expiries is valid but potentially expensive. Quantities and
rights do not change simply because more strikes are available. Discovering
new quantity patterns or templates is a separate registered research step.

The generator answers which placements satisfy the template and domain. The
strategy decides which to select. Searching every placement for highest expected
PnL is a strategy change unless that search/selection rule is already registered.
Current strategies retain their exact factory parameters, expiry/strike
selectors, ordering, tie behavior and refusal/fallback rules.

```text
StructureTemplate:
  template_id, version, content_hash
  parameters_schema
  ordered_leg_rules: [name, right, signed_quantity, expiry_rule, strike_rule]
  reference_geometry_rules
  geometry_constraints, contract_compatibility_policy
  implementation_ref, compatibility_receipt_refs

GenerationRequest:
  schema_version, event_ref, decision_context_ref
  chain_snapshot_ref, template_ref, resolved_template_parameters
  placement_domain_ref, generator_recipe_ref
  quote_eligibility_policy_ref, strategy_admission_policy_ref
  output_order_policy_ref

PlacementDomain:
  mode: legacy_selectors | exhaustive_listed
  eligible_expiry_rule_and_bounds
  anchor_domain, width_or_offset_domains
  fixed_contract_bindings
  constraints_with_units_and_versions
  enumeration_order, equivalence_policy

GenerationBudget:                  # Operational, not a hidden domain change
  max_work_units, max_wall_seconds, page_size, reservation_ref

StructureGenerator.generate(GenerationRequest, GenerationBudget)
  -> CandidatePage
StructureGenerator.resume(request_hash, continuation_token, GenerationBudget)
  -> CandidatePage
StructureGenerator.resolve_pinned(GenerationRequest, ordered_contract_bindings)
  -> CandidatePosition | Refusal
```

All quotes, spot, delta inputs and contract metadata needed by a selector come
from the frozen chain/context. A caller cannot mix a current spot with a prior
snapshot without an explicit registered input policy. `legacy_selectors` is
complete only for its singleton selector-resolved domain; it never claims to
have enumerated all other listed placements. A legacy selector that chooses
an unusable quote must still refuse when the old path did, not silently try
the next better strike.

### 2.2 Candidate and completeness contract

```text
CandidatePosition:
  candidate_id, generation_request_hash, template_ref, event_ref
  chain_snapshot_ref, position_definition_ref, economic_fingerprint
  ordered_resolved_legs, resolved_reference_geometry
  requested_and_resolved_parameters, selector_resolution_trace
  structural_status, quote_status, strategy_admission_status
  rule_receipts, quote_refs, reason_codes

PositionDefinition:
  definition_id, currency, security_id
  ordered_legs: [name, contract_id, signed_quantity]
  reference_geometry, contract_metadata_refs

CandidatePage:
  request_hash, page_id, ordered_candidates, page_hash
  continuation_token, completeness, coverage_receipt_ref

GenerationReceipt:
  request_hash, domain_hash, chain_coverage_ref, ordered_page_manifest
  completeness: complete | budget_exhausted | cancelled | refused
  explored_count, structurally_valid_count, quote_eligible_count
  strategy_admissible_count, rejection_counts_by_rule
  total_count_if_proven, unexplored_domain_ref, constraint_boundary_counts
```

Candidate IDs include the request and full ordered resolution, not just a
rounded width. An economic fingerprint compares actual signed contract
exposures across candidates; it is not permission to net away colliding legs
or deduplicate different strategy explanations. The registered equivalence
policy determines whether identical exposures may be coalesced and records
every contributing placement. No deduplication rule changes during migration.

Keep validity layers separate:

1. **Structural:** listed contracts, legal expiry relationships, no forbidden
   collisions, exact required symmetry/spacing and declared payoff invariants.
2. **Quote eligibility:** quotes present, timely and usable under the registered
   policy. This can fail for an otherwise valid structure.
3. **Strategy admission:** spread, universe, capital or other registered rules.
   Forecast/PnL selection and gates are downstream, not geometry definitions.

Every structurally valid candidate is emitted, including those without usable
quotes, with status/reasons. Impossible combinations produce rejection counts
and inspectable diagnostic samples, not billions of full rejected rows.
Legacy fields such as `n_admissible` retain their exact old population/mask
through named adapters; do not replace them with a similarly named new count.

Completeness means every valid positioning in the declared finite domain and
the supplied chain has been emitted exactly once under the equivalence policy.
It does not assert that a vendor supplied every listed exchange contract:
chain coverage is a separate receipt and may be incomplete. A budget limit,
top-N request, timeout, missing chain page or sampled search cannot masquerade
as exhaustive. An exhaustive selector/gate refuses an incomplete candidate set;
preview can show partial results with the incomplete banner. A deliberately
approximate search needs a separately registered strategy/recipe.

For a synthetic same-expiry put butterfly with quantities `+1, -2, +1`,
listed strikes `{90, 95, 100, 105, 110}`, anchors `{95, 100, 105}` and allowed
half-widths `{5, 10}`, the complete structural output is:

| Anchor | Half-width | Ordered low/center/high strikes |
|---|---|---|
| 95 | 5 | 90, 95, 100 |
| 100 | 5 | 95, 100, 105 |
| 100 | 10 | 90, 100, 110 |
| 105 | 5 | 100, 105, 110 |

Six parameter combinations were explored; two failed the listed-strike rule.
If strike 95 has no usable quote, all four placements still exist but only two
are quote-eligible. Stopping after the first page cannot establish the total
or select the best placement across all four.

Continuation tokens bind the request, snapshot, generator version and ordered
traversal checkpoint. Stable order and page hashes allow deduplication across
retries. Changing page size cannot change the resulting set or winner. A
completed manifest is published atomically; budgets and attempts live in the
execution envelope, not the identity of a successful numerical result.

### 2.3 Preserve the difficult geometry cases

- Support current ATM, moneyness, delta, fixed, same-as, bracket, grid-step,
  offset-from and mirror semantics; dependent selectors resolve in their
  registered order. Delta selection uses the proper option right.
- Require a listed exact mirror when the template does. Never replace it with
  a nearest-strike approximation. Preserve current numerical tolerances.
- Validate dollar geometry after resolving an irregular strike ladder. A
  ladder index difference is not a dollar difference or a percent of spot.
- Retain zero-quantity reference legs for shape/collision checks. Current
  collision checks include references. Do not simplify the position before
  checking them; preserve pinned-entry/closing-leg invariants too.
- Distinguish equal strikes across different expiries/rights from duplicate
  contracts. Contract multiplier, deliverable, currency and exercise/settlement
  convention must satisfy the registered compatibility policy. Unknown metadata
  is not silently filled with standard-contract assumptions; legacy assumptions
  remain explicit in their compatibility adapter.
- A claim such as a nonnegative terminal payoff is template-specific. A
  same-expiry piecewise-linear proof must not be applied to a calendar spread
  with a still-live back leg. Funding requirements are separate from geometry.

## 3. Scenarios: historical analogs and synthetic outcomes share an interface

```text
ScenarioBuildRequest:
  schema_version, event_ref, decision_context_ref, conditioning_state_ref
  target_contract_ref, horizon_contract_ref
  source: HistoricalPositionSource | SyntheticDistributionSource
  scenario_recipe_ref, sampling_policy_ref, rng_policy_ref
  requested_draw_count, weighting_policy_ref

HistoricalPositionSource:
  historical_position_and_outcome_snapshot_ref
  eligible_population_recipe_ref, matcher_recipe_ref, query_features_ref
  outcome_mapping_ref, source_capability_ref
  selected_members_ref, weights_ref, selection_receipt_ref

SyntheticDistributionSource:
  distribution_spec_ref, calibrated_parameter_refs
  joint_dependence_spec_ref, support_and_tail_policy_ref

ScenarioSet:
  scenario_set_id, request_hash, schema_version
  source_manifest_ref, forecast_and_residual_release_refs
  target_and_horizon_contracts, knowledge_cutoff, lineage_audit_ref
  probability_kind: predictive | conditional_predictive | stress_only
  measure: empirical_predictive | assumed_physical | risk_neutral | unweighted_stress
  state_form: factors | final_market_states | market_paths
  factor_units, reference_state_ref, shock_composition_contract
  ordered_scenario_manifest, weights, sampling_and_rng_receipt
  raw_member_count, independent_event_count, effective_sample_size

ScenarioBuilder.build(ScenarioBuildRequest) -> ScenarioSet | Refusal
```

The caller may supply the similar historical positions directly, but they still
need immutable IDs, observation/availability times, complete position/outcome
definitions, weights and a selection receipt. An opaque list of historical
returns is not sufficient to establish what is being transferred to this
position. Match using information available at the target decision time; keep
later outcomes out of matching, hyperparameter selection and fitted transforms.
Outcome labels used in a historical scenario pool must already be available
by that decision cutoff. Multiple positions from one historical event must not
be counted as independent events.

Use an explicit mapping, for example normalized underlying return and IV-surface
changes over a comparable event horizon, or the current calibrated payoff
mapping. Record differences in moneyness, expiries, rights, entry convention,
liquidity and horizon; refuse or flag out-of-domain transfers by policy. Preserve
the existing bucket matcher and chooser kNN matcher as different recipes.

There are three supported recipe families, not one implicit mixture:

- **Historical-factor replay:** reconstruct observed joint spot/IV/time paths
  and reprice the target contracts under those factor changes.
- **Forecast plus causal residuals:** use pinned forecasts and OOF residual
  pools, preserving paired move/crush errors, conditioning and sample rules.
- **Pure synthetic:** caller supplies a versioned joint distribution or scenario
  grid, with units, dependence, support, weights and calibration lineage if any.

A direct empirical return/payoff adapter remains available for existing recipes,
but declares exactly which parameter/horizon changes it supports. Terminal
returns alone do not reveal the intermediate IV/path needed to answer a new
time/IV what-if. Return `UNSUPPORTED_OUTCOME_MAPPING` instead of fabricating that
information. Combining empirical and synthetic sources requires an explicit
mixture recipe and weights, not a fallback hidden inside the simulator.

Probability weights must be finite and nonnegative, with positive total mass
and an explicit normalization rule. Stress-only scenarios may have null weights.
Equal sampling weights are a declared choice. Effective
sample size from weights is reported separately from event clusters and any
uncertainty estimate. A stress grid without probability weights supplies
per-scenario changes/ranges but no expected return or win probability. A chosen
deterministic shock is a what-if, not an estimate of how likely it is to occur.

Scenario probabilities and conditional option valuation are separate objects.
A risk-neutral scenario measure is labeled as such and cannot silently become
a forecast of realized profit or a predictive gate input. An assumed physical
distribution carries its assumptions, not a claim of empirical calibration.
The registered gate declares which measure/estimator it consumes.

## 4. Position valuation, elapsed time and parameter changes

An option value depends on underlying price, strike, time remaining, volatility,
rates and dividends. Keep those inputs explicit instead of making one IV scalar
and DTE stand for every contract. The interface is deliberately wider than the
current put-only helper. ([OIC: options pricing inputs](https://www.optionseducation.org/optionsoverview/options-pricing))

```text
PositionState:
  position_state_id, position_definition_ref, state_at
  market_state_ref, per_leg_state_refs
  entry_basis_ref, realized_cashflow_refs, position_event_cursor

MarketState:
  observed_at, spot_by_security, iv_surface_ref
  rates_curve_ref, dividend_schedule_ref, borrow_and_fx_refs
  quote_refs, source_lineage_ref, assumption_flags

ValuationPolicy:
  policy_id, version, implementation_ref, calibration_artifact_refs
  capabilities: rights, exercise_styles, deliverables, expiries, paths
  input_contract, price_unit, day_count_and_settlement_rules
  vol_surface_interpolation_and_dynamics, missing_input_policy
  numerical_tolerances, approximation_flags, compatibility_receipts

HorizonSpec:
  valuation_at
  elapsed_time, elapsed_time_unit, calendar_ref, roll_rule
  action: mark_remaining_position | close_position
  expiry_exercise_assignment_policy_ref, carry_policy_ref

ParameterShock:
  target, scope, operation, value, unit
  reference: base_state | scenario_state
  application_time_or_path_rule

PositionValuator.value(PositionDefinition, MarketState, valuation_at, policy)
  -> PositionValuation | Refusal

PositionValuation:
  position_definition_ref, market_state_ref, valuation_policy_ref
  per_leg_values, net_value, currency, unit
  intervening_settlement_cashflows, residual_holdings_ref
  capability_checks, approximation_flags, lineage_ref
```

Required semantics:

- Resolve one absolute horizon timestamp; elapsed time is its audited input.
  Reject conflicts between them. Calendar days, trading sessions and elapsed
  seconds are different. Each leg uses its own expiry and day-count convention.
- IV units are annualized fractions unless another unit is declared. From
  0.40 IV, `add 0.05 volatility_fraction` yields 0.45; `relative_change 0.05`
  yields 0.42. A scalar parallel shift, expiry-node shift and skew change are
  different operations. Define sticky-strike/delta behavior and interpolation
  in the valuation policy; do not add an undisclosed surface-dynamics assumption.
- Specify whether scenarios represent factor changes or already-realized final
  states. Record shock order/reference. A sampled crush cannot also be applied
  as a deterministic adjustment accidentally. Reject ambiguous duplicate shocks.
- Revalue the same contract IDs/quantities. Do not reselect ATM strikes after
  spot moves. A rebalance is a separate registered path policy with trades,
  cash flows and execution assumptions; it is not part of a plain mark request.
- Do not carry an expired leg as an option while repricing its later-expiring
  companion. Settle/exercise/assign according to declared capabilities, record
  cash/underlying delivery and value residual holdings. Unsupported American
  exercise, adjusted deliverables, settlement or path requirements refuse.
- Full repricing, calibrated empirical payoff maps, and local Greek
  approximations are separate adapters. Greek approximations have an explicit
  validity domain and diagnostic flag; they cannot silently replace a gate
  estimator. Model/NN surrogates require the same registered release/evidence
  process as other models and cannot fit during a simulation.

For a path, an evolution adapter first applies settlement/corporate-action events
to PositionState and then calls the valuator on remaining holdings. A bare
`value` call cannot invent past assignment events. A mark-only request with no
entry basis can still estimate change in value; it cannot report trade PnL.

## 5. PnL simulation and accounting contract

```text
SimulationRequest:
  schema_version, position_state_ref, scenario_set_ref
  horizon_spec, ordered_parameter_shocks
  valuation_policy_ref, scenario_mapping_ref
  accounting_policy_ref, execution_assumption_ref
  return_denominator_spec, summary_spec

SimulationBudget:
  reservation_ref, contract_batch_size, scenario_batch_size
  max_wall_seconds, persist_draw_details

PnLSimulator.simulate(SimulationRequest, SimulationBudget)
  -> SimulationResult | Refusal
PnLSimulator.simulate_many(ordered_requests, SimulationBudget)
  -> SimulationBatchManifest

SimulationResult:
  result_id, request_hash, position_state_ref, scenario_set_ref
  resolved_horizon_and_shocks, recipe_and_artifact_refs
  baseline_model_value, baseline_market_value, baseline_entry_cashflow
  per_scenario_values_and_cashflows_ref
  value_change_summary, economic_pnl_summary, return_summary
  probability_of_profit, loss_quantiles, mean_estimation_error
  sample_counts, weighting_receipt, completion_status
  execution_basis, price_source_kind, approximation_and_domain_flags
  validation_receipts, refusal_reasons
```

Nullable fields are explicit with reasons. A batch emits a result or typed
refusal for every requested position; it never silently drops difficult legs,
scenarios or candidates. Partial computation is checkpointed but not published
as a complete distribution. Renormalizing only the scenarios that priced
successfully is forbidden unless a different conditioned population is an
explicitly registered recipe with its exclusion audit.

### 5.1 Do not conflate value change, trade PnL and return

For signed quantity `q_i` (long positive), contract multiplier `m_i` and a
per-underlying-unit price `p_i`, net marked value is:

```text
V(t) = sum_i q_i * m_i * p_i(t) + value_of_residual_delivered_holdings(t)

modeled_value_change_s = V_model(horizon, s) + intervening_cash_s - V_model(now)

economic_pnl_s = opening_cashflow + intervening_cash_s
                 + signed_close_proceeds_s - execution_costs_s

return_s = economic_pnl_s / explicitly_registered_positive_denominator
```

Cash-flow signs are positive for money received. Opening cash flow excludes
costs in these equations; fees must be subtracted exactly once. Intervention
cash flows exclude any opening/closing cash already counted. All values use a
common currency and evaluation numeraire under the carry policy. For a holding
opened before `now`, distinguish since-entry cash flows from the `now`-to-horizon
flows used in value change. At a mark-only horizon, substitute declared terminal
marks for close proceeds and label the result unrealized/model-mark PnL.
Adapters normalize provider per-share/per-contract price units before applying
the multiplier. Current premium-unit calculations map explicitly to currency
amounts; the return fraction must remain unchanged by that unit conversion.

Model value change and economic PnL answer different questions when the model
baseline differs from the opening fill or current market mark. Report the
baseline discrepancy; do not force the model to equal cost without a registered
calibration. A legacy adapter that only calculates terminal value may leave
baseline-model value/change unavailable while still reproducing its expected
trade return.

For example, synthetic long/short legs with quantities `+1/-1`, multipliers
100, opening premiums `3/1` and closing premiums `4/0.5` have opening cash flow
-200 and closing proceeds +350: trade PnL is +150 before costs. The difference
in net premium is 1.5, not 150%; return on the 200 debit is 75%. Reversing both
legs produces -150 PnL but a credit entry, so no positive-debit return is
defined. Return on secured capital requires a separately supplied funding
policy; margin is not the negative of premium.

### 5.2 Market evidence and execution assumptions

Theoretical prices are not fills. Return both the modeled value change and,
when requested, a separately labeled execution-adjusted PnL distribution.
An execution policy declares spread/slippage/fees, per-leg side, quote source,
fill alpha and liquidity assumptions. Future bid/ask cannot be inferred from
Black-Scholes alone. Missing executable evidence stays missing, not a zero-cost
fill. Preserve current worst/mid/best fill sensitivity and the measured-fill
checks; daily bars are real traded-price evidence, not proof a multi-leg order
could be filled simultaneously at those prices.

Predictive summaries use declared weights and quantile conventions. Outcome
quantiles describe the distribution of PnL; Monte Carlo standard error describes
estimation error in a statistic, not a confidence interval on profitability.
Estimate uncertainty with the appropriate weighting/dependence policy or return
it unavailable. Report historical sample size, independent event count and draw
count separately: 4,000 resamples do not mean 4,000 independent observations.
Each summary names its metric, unit and basis. `probability_of_profit` refers
to the declared economic PnL after specified costs being strictly positive;
probability of a positive model value change is a separate metric. Neither is
available as a predictive claim for an unweighted stress-only scenario set.

## 6. Compatibility and efficient implementation

### 6.1 Named legacy adapters, unchanged numbers

The first deployment uses adapters, not a new pricing methodology:

| Adapter | Behavior pinned from the current implementation |
|---|---|
| `legacy.structure_selectors.v1` | Factory parameters, ordered selectors, quote selection/refusals, reference/collision checks, pinned exits and fill cash flows |
| `legacy.paired_move_crush.v1` | Eligible prior residuals, conditioning, minimum pool, paired sampling, random sign draws, draw count and event/key seed |
| `legacy.put_exit_bs.v1` | Current put-only helper, aggregate exit DTE divided by 365, common IV, zero rate/dividend assumptions, volatility/spot floors and intrinsic-at-expiry behavior |
| `legacy.pnl_return.v1` | Positive-debit denominator, current expected return/win/quantile outputs and unavailable cases |
| `legacy.payoff_map.v1` / `legacy.runup_surface.v1` | Current calibrated maps, residual/noise rules, support/clamps and return bases |

The IDs are proposed names; exporting the baseline source/constants and fixture
receipts defines their actual content. In particular, `exp_pnl_sim` is currently
a return fraction on debit, not dollars. Map its legacy name explicitly. The
legacy put policy must not claim general call/calendar/American valuation
coverage; it remains an explicitly limited model-mark estimator where current
strategies already use it. A more realistic valuation policy is a new recipe
with new evidence, never an unannounced numerical correction in this refactor.

Preserve the actual existing RNG sequence, call ordering, clipping, quantile
calculation, numerical precision and missing-input semantics. Do not adopt
common random numbers, different residual pooling, per-leg IV, or new horizon
semantics for existing strategies without separate validation/versioning.
Adding capability to the engine does not enable disabled CAL-P/CND-P or change
any current selector, gate, DYN-SV ordering or deployment.

### 6.2 Reuse work without unbounded memory

For new registered searches, build one scenario set for the event/context and
use the same scenario IDs across candidate comparisons. Price each distinct
contract once per scenario/horizon/policy and compose position values with the
signed leg matrix. This generalizes the useful reuse in EXP-133; it is not a
second independent pricer. Prove the optimized path against the scalar valuator.

Linearity allows combining per-leg values/means under identical assumptions;
it does not allow adding quantiles, win probabilities, nonlinear fees, margin,
assignment or liquidation rules. Those require scenario-level position
accounting. No cross-candidate reuse is valid when scenario conditioning, quote
policy, horizon, contract metadata or valuation state differs.

Stream candidate pages and contract/scenario blocks under a supervisor
reservation. Never materialize all events x candidates x contracts x draws.
Cache keys include the actual consumed inputs, units, model/surface revisions,
shocks, RNG and valuation policies. Cached scenario values cannot cross a live
snapshot or model promotion implicitly. Persist compact summaries and a replayable
scenario manifest; retain detailed draws when required by the registered audit
policy. An RNG seed alone is not enough without the generator version and
immutable source population.

## 7. Integration and acceptance tests

Strategy registration adds a typed `component_graph_ref` resolving:

```text
structure_template + generator + placement_domain
scenario_source/matcher + outcome_mapping + forecast/residual bindings
valuation + accounting + execution + return_denominator
candidate_selector + gate + compatibility/evidence receipts
```

Dependencies are optional only when that strategy genuinely does not use the
component. No new root-level policy may disagree with the graph: registration
resolves both the existing StrategySpec fields and this graph to one canonical
definition hash and rejects conflicts. Legacy adapters preserve that definition
while the new wrapper/version identity remains separately recorded.

ScoreRecord retains candidate-set completeness, selected candidate/position,
scenario and simulation result refs, exact recipe versions, and the offered-set
audit used by its selector. Lazy event detail, payoff/what-if pages, replay and
experiments consume these same records/functions. A what-if with different
time/IV creates a new analysis request/result and cannot overwrite a frozen
score or enter the ledger as a recommendation.

Required implementation tests, in addition to the existing parity suite:

| Test | Required proof |
|---|---|
| Tiny-chain exhaustive oracle | Hand/brute-force solution set equals generator output, no omissions/duplicates, across calls/puts/multiple expiries and pinned domains |
| Irregular ladder and references | Exact mirrors, no nearest substitution, dollar-floor tests, zero-quantity references, collisions and selector ties match their registered rules |
| Missing quotes and budgets | Structural count survives missing quotes; unavailable input is not a negative gate; interrupted/resumed pages equal a full run; incomplete sets cannot be ranked as complete |
| Legacy golden corpus | Exact chosen contracts/quantities/refusals and existing forecasts, simulated returns, cutoff/gate and chooser outputs, with pinned numerical tolerances |
| Cash-flow arithmetic | Hand long/short example above, multiplier once, debit/credit/zero basis, fees once, same-state zero value change without costs, different model/market basis |
| Time and shocks | Zero shock/elapsed identity, relative versus additive IV, one remaining DTE per expiry, no double shock, no strike reselection, expiry settlement and unsupported-path refusal |
| Causal scenarios | Future outcome poisoning, late data receipt, OOF residual provenance, event clustering, horizon/shape mismatch and thin pools fail or leave the score unchanged as specified |
| Distribution accounting | Weights normalize by policy, stress-only grids have no predictive expectation, quantile/mean conventions match, failed scenarios cannot disappear through renormalization |
| Optimized/scalar equivalence | Batched/shared per-contract values reproduce scalar simulation under the same draws; position quantiles are not sums of leg quantiles |
| Operational integration | Fresh-process deterministic replay, cache invalidation, concurrent request isolation, bounded resource use and restart at every chunk boundary |

Migration sequence: freeze fixtures -> register/wrap existing components ->
extract generator and valuation kernels with scalar equivalence -> add the
exhaustive/what-if interfaces in research -> route callers through the shared
scoring graph -> independently validate any new method before promotion.
These are acceptance requirements, not claims that the tests were run by this
documentation-only PR.
