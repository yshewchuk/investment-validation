# SUPER SIZE stage 2: causal panel features and bounded magnitude search

Status: core scope and causal legacy-refit baseline accepted by the owner on
October 21, 2026. Implementation remains pending; nothing is preregistered.
Owner acceptance is not gate/review approval or authority for actual registration,
private-data reads, experiment runs or merging. Check this document in before code.
[Current executable contract](SPEC.md) remains authoritative. The accepted scope
retains the deferred work below and freezes neither a dataset nor a resource budget.

## Priority and evidence

Target roughly one engineering week for a bounded mean-magnitude comparison,
conditional on source provenance and review; estimates are not a delivery promise.
If causal input/baseline work exceeds budget, ship foundations and postpone search; never relax admission.

Main has eleven history inputs, Ridge/HGB, nested annual selection, sealing and serial execution, not a proven winner. The October 21 feedback reports 498 matched events: legacy MAE 2.79,
history Ridge 2.97, paired difference +0.19, 95% interval [+0.10, +0.28]. These
are separately reported rounded figures; raw estimates were not checked. It also
reports a 2,014-event single-arm resource/replay run and ~0.17 seconds/event history build
(118k events: about 5.6 hours). None was independently reproduced for this design;
none substitutes for the prospective nested comparison.

Retain the October 10 proposal: conditional mean/RMSE primary, MAE/quantiles secondary.
Target and 5/3/1 clocks stay unchanged; do not call an MAE-optimal output a conditional mean.

## 1. Stored causal research panel (core, highest priority)

**Do not copy final event-panel rows wholesale.** Their market state is the last
pre-print daily row, too late for some experiment horizons; persisted panel output
also drops intermediate anchors. See [legacy convention](../../engine/features.py),
[anchor math](../../engine/data/features/panel.py), and
[native panel design](../../guides/native_panel_tier4_design.md).
Reuse audited existing feature math, but reconstruct each block at the experiment's
source-session ceiling from qualified, pinned stored sources. Provider calls, new
source acquisition, production panel migration and production Tier4 rebuilds are
outside this design's scope; required stored-source availability must be established.

Proposed table: `research_magnitude_panel`, immutable versioned schema, primary key
`(event_id, context_id, horizon_sessions)`. Join to the exact history snapshot on
these semantics, with an explicit history-context binding (not equal context hashes).
Require exactly one row for each admitted event and each horizon, including rows
with unavailable optional inputs. Never silently complete-case filter the cohort.

The context binds source snapshot/closure, exact history snapshot, selected-event
manifest, holdout month/membership versions, eligibility policy ID, target/clock/availability policies,
training/history start, evaluation years, feature dictionary and producer/environment
hashes. Each row retains cutoff ID, source-session ceiling, block source date and
availability/knowledge mode, consumed source row/fragment identities, missingness
reason, and a digest. Preserve individual lag anchors, not only their maximum.
Refuse unknown columns/versions, duplicate keys, context mismatch or ambiguous lineage.
Optional missing values stay null; changed recipes/sources get new immutable identities, never overwrite.

### Feature dictionary and subsets

Freeze ordered column lists, units, aliases, missing rules and source dependencies
before registration. The proposed dictionary reuses these blocks' math, with the
new regime-source binding below:

| Set | Inputs | Eligibility |
| --- | --- | --- |
| H | Eleven current stored-history columns | Existing history contract unchanged |
| HM | H + signed_streak, ema12r_abs, dist_high, dist_ema, abs_dist_ema, ret5, ret10, ret20; ^GSPC-close ret21/63/252, dd252, vol5/20/60/252, vol20_rel252 (existing spy_* aliases) | Audited runup/regime math at each horizon; versioned source binding below |
| HMO | HM + implied_move, iv10, iv30, exern_iv10, exern_iv30, iee, skew, contango, fwd90_30, fexern90_30, rvol30, spot, mcap_log; mean_prior_or_implied | Daily-market mapping plus the protected implied-history prefix below |

The thirteen daily-market names denote research aliases from `DAILY_STATE_FIELDS`,
not permission to read a same-named realized-event field. Historical implied mean
has its separate prefix policy below. Add quote-present indicators
only to HMO, and missing indicators only for each recipe's selected columns; H
remains history-only. These are not extra search axes. A zero current implied
quote means unavailable to HMO, while a genuine zero target remains valid. Missing market
cap or late capture stays missing; do not use the earliest later retrieval as causal.

Defer daily-state lags, ratios/interactions and sector context pending exact source/availability maps.
Row lags are not calendar lags. No point-in-time sector mapping is established by this audit; do not join today's
sector classification backward. Existing options-derived daily fields are in core;
new option-chain surfaces, interpolation and liquidity engineering are not.

Exclude `or_exern_z252` (quarantined), current-event move/abs_move, post-event IV,
label-derived quality/completion flags, and stored learned Tier4 predictions.
Tier4 stacking requires fold-local/OOF regeneration and verified training cutoffs,
not merely an early forecast timestamp; defer it. The incumbent-parity amendment
below exposes its two previously omitted inputs to challenger selection as well.

### Incumbent-parity amendment (October 24; design only)

The owner approved this design-first lane before dictionary implementation. The
table above now proposes 11/28/42 numeric columns for H/HM/HMO; the checked-in
`research_magnitude_panel.v1` dictionary still has 11/27/40. No existing runtime
contract, snapshot or compiler declaration is changed by this document.
Do not mistake the obsolete ten-column `LEGACY_FEATURES` or registry prose for
the active [fourteen-input list](../../engine/models/training/size_model.py#L72-L87).
The following mapping is exhaustive; it adds no learned feature or search axis.

| Active incumbent input | Challenger subset / mapping (source caveat below) |
| --- | --- |
| has_implied_quote | HMO only; 1 for a positive current quote, otherwise 0 |
| mean_prior_abs_move | H; existing protected history value |
| abs_dist_ema | HM/HMO; abs(dist_ema), null iff its causal source is null |
| ema12r_abs | HM/HMO; ema12_prior_abs_move if n_prior >= 12, else mean_prior_abs_move |
| mean_prior_move | H; existing protected history value |
| signed_streak | HM/HMO; existing signed-run recursion over qualified prior outcomes |
| dist_high | HM/HMO; existing adjusted-price distance from trailing high |
| dist_ema | HM/HMO; existing adjusted-price distance from trailing EMA |
| spy_vol20 | HM/HMO; regime volatility; v2 selects ^GSPC close, current v1 differs |
| spy_dd252 | HM/HMO; regime drawdown; v2 selects ^GSPC close, current v1 differs |
| mean_prior_or_implied | HMO only; protected historical ORATS quote mean below |
| or_implied | HMO alias implied_move; comparator mask [0,60], HMO source-validity policy below |
| or_rvol30 | HMO alias rvol30; comparator mask [0,700], HMO source-validity policy below |
| mcap_log | HMO; missing/known-late capture remains null |

**Regime-source choice and current caveat ([#1276](https://github.com/yshewchuk/investment-validation/issues/1276)).**
Matching `spy_*` names and arithmetic does not establish source parity. The
[legacy panel](../../engine/data/features/panel.py#L367-L403) uses the ^GSPC
`gspc_daily.csv` close; [native inputs](../../engine/v2/features/panel_row_inputs.py#L349-L370)
and [research blocks](../../engine/v2/research/magnitude_panel_daily.py#L198-L207)
instead feed SPY `daily_market.spot` into the regime calculation's `close` column.
The provider maps `stockPrice` to `spot`; that mapping alone proves neither
official-close nor intraday-snapshot semantics. This verified mismatch concerns
regime inputs, not native runup, which uses `price_history.close_adj`.

Owner decision (October 24): use ^GSPC closing prices for the experiment's
market-context features, including HM/HMO and the causal legacy comparator.
The owner reports a prior comparison found no material difference and prefers
^GSPC to avoid ETF-specific fund drag; that comparison is not independently
reproduced here or asserted as measured equivalence. Retain existing `spy_*`
aliases/order, but bind the selected instrument, close definition and source policy
in the new v2 dictionary, with the stored snapshot and causal availability/provenance
in new recipe/context identities. Existing v1 SPY inputs and identities stay unchanged.

Implementation depends on a qualified stored ^GSPC-close source and a research
reader enforcing each 5/3/1 source ceiling and frozen admission/holdout policy.
Neither that source's availability nor its research reader is established here;
the legacy CSV path is code evidence, not permission to read it. Missing required
provenance refuses before values; missing qualified inputs follow the frozen
null/abstention policy. The source choice resolves the benchmark decision, not
storage/provenance qualification or exact source parity. No source switch or data
run is implemented or authorized by this design.

`ema12r_abs` is an EMA-with-mean fallback, not the ratio described in stale
feature-note prose. `abs_dist_high` is not an active incumbent input: it was
removed as a duplicate of -dist_high. No other explicit interaction belongs to
the active list. General ratios/lags/sector/stacking remain deferred. Source
math already exists in `runup_math`, `panel_math` and `event_history`; reuse it
where its admission contract matches, not production readers or stored predictions.

**Absolute distance.** Derive from the same horizon-admitted `dist_ema` used by
both arms; retain its adjusted-price anchors and missing reason. No extra read,
backfill, full-panel copy, rounding or independent availability claim is allowed.

**Historical implied mean.** Freeze one shared reconstruction for HMO and the
causal incumbent. For each target/horizon, establish the full pinned canonical
confirmed-event metadata prefix from its first known event, strictly before both
the target event date and source-session ceiling. This includes metadata before
history_start; that start bounds numeric contributions, not ordinal calculation.
Keep the ordered metadata sequence and its zero-based ordinal; warm-up is four events, as the native
[computed-move contract](../../engine/v2/data/computed_moves_table.py#L90-L102)
and `event_sequence(..., warmup=4)` define it. Protected or embargo-excluded events may count as metadata ordinals,
but never contribute quote values. Do not restart warm-up at history_start,
renumber after exclusions, infer an ordinal from H.n_prior, or reuse a production
`admitted`/completion flag. Prove the sequence/start against pinned metadata;
an incomplete prefix or conflicting event identity refuses before values.

A contributor must have ordinal >= 4, be on/after history_start, and be a
nonprotected development event admitted by the same frozen grouped-eligibility
policy. Resolve its quote from
already stored daily_market: the last src_iv row strictly before that historical
event date, never a newer row or an older nonmissing fallback. Its numerical
source prefix and any known availability must not exceed that historical
pre-print ceiling or the target forecast ceiling. Missing publication stamps
remain explicitly reconstructed, never observed-PIT; known-late values are null.
No realized value, skipped flag or label-completion status chooses quote inclusion.
Thus a missing realized move does not by itself exclude an available prior quote.

Compute the arithmetic mean over finite admitted quotes, including zero
sentinels only when the pinned stored source preserves them; skip nulls, count actual contributions, and return null when count
is zero. Do not apply the incumbent's current-quote [0,60] mask to historical
quotes or their mean. Observable stored source-contract violations refuse, not
silently winsorize; an ingest-masked null does not expose its original violation.
Reuse `strict_prior_mean(..., population="admitted")` on the complete ordinal sequence, masking every
noncontributor's quote to null without reading it. Append the target with a null
quote and take that target's returned mean/count, so the final eligible prior
event is included. The aggregate does not further mask source-preserved zeros;
normalized nulls remain null. A resulting
zero mean from preserved zeros is valid even though a zero *current* quote is unavailable
to HMO. Do not copy `computed_moves.mean_prior_or_implied`, resume its aggregate,
or substitute realized-move history: filtering its latest row cannot remove
protected contributions. Never open a mixed protected outcome fragment merely
to obtain an implied quote; reconstruction uses event metadata and daily sources.

Retain the complete prefix membership/exclusions, source and availability anchors,
consumed row/fragment identities, contribution count and missing reason in v2
lineage/context identity. These are provenance, not added predictors. A known
empty qualified prefix stays null; missing required provenance refuses rather
than manufacturing a trustworthy missing value. This is the declared research
population restriction, not byte-identical reproduction of old production inputs.

**Current quote and ordering.** Preserve source status `present`, `zero`, `absent`
or `unavailable`. HMO stores positive implied_move or null; its appended
has_implied_quote is 1 only for `present`. The comparator retains or_implied=0
only for a source-preserved zero, never for a normalized null. Both arms use the same
status/anchor and has_implied_quote rule; the comparator's existing masks and
abstentions stay unchanged. The source-derived indicator precedes comparator
value masks; masking or_implied causes abstention, not indicator recomputation.
HMO admits finite source-valid rvol30 and strictly
positive source-valid implied_move without adding the comparator's [0,700] and
[0,60] masks. Both inspected native and legacy normalizers enforce rvol30 in
[0,500] and implied_move in [0,100]; each admitted snapshot binds its actual source
contract/producer, not an assumed range. This preserves the designed arm-specific
preprocessing, not identical value admission; report mask-related abstentions.

**Lossy source encoding.** Both the [native normalizer](../../engine/v2/ops/providers/orats_daily_market.py#L346-L354)
and [legacy normalizer](../../engine/data/normalize/n_daily.py#L286-L304) already
replace nonpositive current quotes with null. Such stored nulls do not prove
whether the original value was zero or absent. Keep them null, record that loss
of original status, skip them in both arms' reconstructed historical mean, and
retain comparator abstention for missing current quotes. Set status `zero` and
retain zero in the baseline/mean only when the pinned stored source actually
preserves it; never reconstruct zero from null or fetch provider/raw data to
recover it in this scope. Synthetic preserved-zero cases test policy, not source
coverage. This prevents a claim that every historical production sentinel can
be reproduced from today's normalized store.

Add abs_dist_ema immediately after dist_ema in the
runup block; append an `implied_history` block containing mean_prior_or_implied
after daily_market. All other numeric order is unchanged. HMO's raw model matrix
is these 42 columns followed by has_implied_quote; H/HM have no quote column.
Recipe-specific missing flags are generated only from that recipe's columns
inside each training fold, never stored globally or used to preselect features.

**Immutable migration, not in-place widening.** Add research_magnitude_panel.v2
(semantic version 2.0.0) and magnitude_panel_context.v2; preserve the existing
v1 dictionary, helpers, row admission and identities. V2 dictionary entries bind
ordered names, units, missing policy, source dependencies, the regime-source policy
and the two derivation policies above; dictionary_hash remains content_hash of that ordered dictionary.
H's eleven values/order remain unchanged; an unchanged H dictionary hash alone
does not make the new versioned recipe/context identical. HM/HMO hashes change.
New producers bind v2 schema/policies, full HMO dictionary hash, source/history
snapshots, eligibility, prefix lineage and environment. Never relabel v1 or
partial daily blocks as v2 rows, overwrite old snapshots, or reuse a success
receipt under changed bindings. Shared history may be referenced only after
exact existing context and eligibility compatibility checks; never rebuild it
implicitly or reinterpret its membership to make a join succeed.

The source-free compiler landed in [#1236](https://github.com/yshewchuk/investment-validation/pull/1236)
with v1-only dictionary-hash support. P3a below separates source-free support for
the two exact dictionary versions from P3b execution wiring; no latest/default substitution.
Old request outputs must replay identically; the new hash/recipe yields new
request/candidate IDs. Keep the request schema v2 and closed H/HM/HMO grids;
dictionary version is explicit in the resolved recipe, not an extra search axis.
Any actual amended run needs a new experiment/arm registration and sealed
source/fold identities. Compiler declarations remain nonexecuting.

There are still 33 candidates and 603 search fits for the proposed six-inner /
three-outer fixture, plus 9 legacy logical / 18 component fits. Another schedule
has different counts. Wider matrices and prefix work can change cost; none of
these counts is a resource promise, real-data budget or performance claim.

**Acceptance and bounded sequence.** This design precedes dictionary code.
P1–P3 refine the original section-5 roadmap rather than replace it: P1 extends
core slice 1; P2 completes the parity-dependent part of slices 2/6; P3a extends
only the source-free slice 5a, while P3b wires versions through slices 3/4/5b/6.
Other core work and dependencies remain; P3a does not complete P3 or unblock a run.

| Slice | Dependency / acceptance | Estimated added code / tests / docs |
| --- | --- | --- |
| P1: add-only v2 dictionary/context contract | This reviewed design; explicit versions, closed lineage, no source reader | 180–300 / 250–400 / 35–60 |
| P2: protected implied-history and absolute derivation | P1 and remaining causal materializer; reuse pinned/batched sources, no per-candidate builds | 250–380 / 300–500 / 40–70 |
| P3a: source-free dictionary dispatch | Landed v2 definitions (#1293), compiler (#1236), and reviewed section 6.1; preserve old replay/closures | 50–80 / 180–240 / 20–35 |
| P3b1: stored reader/version admission wiring | P1/P2, complete qualified stored reader; no source rebuild | 70–110 / 140–220 / 20–35 |
| P3b2: full recipe/source binding | P3b1, qualified comparison and fitted-family recipes; bind all fixed knobs | 60–100 / 140–220 / 20–35 |
| P3b3: planner/executor version wiring | P3a/P3b2 and core 5b admission; no bypass of selection/holdout rules | 70–110 / 160–230 / 20–35 |

Split add/wire further at repository limits; these are estimates, not waivers.
P1 tests pin all 14 mappings, exact 11/28/42 order and HMO quote position,
derivation-policy identity changes, unchanged v1 outputs and cross-version refusal.
P2 uses independent numeric oracles for abs/nulls and prior means, zero/null quote
counts, full-prefix four-event warm-up without restart at history_start,
last-prior inclusion, protected/embargo holes without renumbering, skipped
realized outcomes with valid quotes, future/known-late poison and each 5/3/1 clock.
Include normalized-null versus preserved-zero fixtures and arm-specific mask boundaries.
It proves no protected payload read and bounded source scans/batch invariance.
P3a proves old request replay, new declaration identities and unchanged 33/603
fixture counts without qualifying a source. P3b requires the qualified ^GSPC-close
reader and pins its source-policy/snapshot identities, testing causal ceilings,
mismatched bindings, no partial-row admission and no global preprocessing or
outer-outcome selection. P3b estimates cover wiring only, not missing prerequisites;
freeze each concrete API in a reviewed design before implementation and split again
if its code/tests/docs would exceed 400 combined added lines.
No real source coverage, holdout opening, provider job, fit or registration is
authorized or established here. The parallel numeric adapters remain
column/hash-driven, with no panel dependency or hard-coded width.

### Admission before data and values

Owner clarification (October 21): the holdouts are complementary, not interchangeable.
Recent-month chronological holdout is primary forward confirmation: no training, tuning, selection or calibration.
Random events check historical event/regime robustness, not independence or standalone temporal isolation.
They cannot replace time holdouts. Report layers separately; preregister the joint decision before opening either.

Platform ownership: a reusable versioned eligibility policy in `engine/v2/research`,
not a SUPER SIZE filter. New consumers bind sealed policy IDs/parameters and exclusion
manifests into run identity/coverage. Existing v1 consumers remain unchanged; adoption
requires explicit versioned migration, never retroactive filtering. Cross-consumer tests
must prove identical grouped admission and refusal on missing/conflicting policy identities.
SUPER SIZE preregisters 30 days (not a platform constant): an inclusive ±30-calendar-day window for
each random-held event's ticker/date. Exclude a whole development event (all 1/3/5-session horizons) if its
label interval, event date or any forecast decision/source-session ceiling overlaps
any such window. Apply the same metadata-first exclusion before preprocessing,
selection, calibration and fits, including the legacy baseline; preserve protected
evaluation membership. Freeze the window/manifest before outcomes, with no tuning.
This excludes examples, not individual price bars; do not alter rolling formulas.

Extend protected event-outcome preflight, not production `scan_panel_row`: classify
the random/rolling union before outcome payloads; refuse mixed outcome fragments.
Protected labels/explicit event returns never enter development history or fitting.
The intended reconstructed loader may decode whole-ticker raw files containing later
rows, but numerical predictors use only causal prefixes through each source-session
ceiling and declared availability cutoff. Physical decoding proves neither numerical eligibility nor publication.
Long-memory price/IV features and cross-ticker correlations can retain event effects
beyond 30 days; the embargo does not erase those effects or prove random-layer isolation.
Chronological evidence does not guarantee regime independence. Enforce availability/strict prior-label cutoffs and frozen missing policies;
never backdate known-late inputs into an observed-PIT claim. This policy clarification
is no authority for private reads, opening either holdout, registration or execution.
Slice 1 metadata admission alone still grants no raw-payload access.

### Bounded construction, not another per-event history build

Use one metadata/holdout plan and pinned source resolution per build; batch by
bounded ticker/year groups, sort each source once, and share bounded ^GSPC/calendar
state. Advance causal prefixes/search indices across horizons and events. Reuse the
existing immutable history artifact; do not call its producer inside the panel loop
or repeat feature production per candidate/fold. Preserve prefix state across batch
boundaries without retaining the full universe in memory. Reuse an exact successful
receipt; verify bindings before reuse, with no silent source refresh. Publish immutable
objects first, then atomically commit the complete validated manifest/snapshot, success
receipt and context head via catalog.commit_snapshot. Failed batches/transactions leave
no admissible snapshot or success receipt; unreferenced objects are never reusable rows.
No automatic retry/resume or new context on failure; check the exact committed receipt
after an uncertain commit before any separately authorized rebuild of the same context.
Exact successful replay verifies identity without rebuilding; conflicts never overwrite.
Batch source processing, not blindly physical output: retain per-event fragments unless
whole-fragment protected-read admission is proven for coalesced output.

Acceptance uses synthetic access counters: bounded source scans per ticker/batch,
not per event, candidate or horizon; identical rows/digests across batch sizes,
no protected outcome reads or future numerical use, raw-policy compliance, and replay. A later authorized
same-cohort pilot records incremental wall/CPU/RSS/read counts separately from
history build, with explicit limits and bounded stderr progress/ETA. The 0.17-second
history figure is a reported reference, not an accepted panel budget. If panel cost
scales multiplicatively, block full-universe execution and fix batching first.

## 2. Models and inner-only selection (core)

Use installed sklearn Pipeline, SimpleImputer, MissingIndicator, StandardScaler, Ridge,
ElasticNet and HistGradientBoostingRegressor; reuse preprocessing, no new dependency/optimizer. Every fresh inner/outer fit learns
imputation and scaling only from its qualified training rows. Freeze all-null columns
and indicator layout deterministically; no global preprocessing or target transform.
Project finite point predictions to nonnegative values as the current recipes do.

The core adds ElasticNet: alpha [0.01, 0.1, 1], l1_ratio [0.25, 0.75], fit_intercept
true, selection cyclic, max_iter 10000, tol 0.0001. Record convergence; a convergence
failure is an explicit variant failure, not a reason to silently increase iterations.
Its sparse coefficients provide embedded selection; report selected columns per fold.
Do not use coefficients from all data to preselect inner-fold features.

Cross each of H/HM/HMO with Ridge alpha [0.1, 1, 10], ElasticNet's six combinations,
and two HGB configurations: current fixed defaults with max_leaf_nodes [7, 15].
All remaining HGB knobs stay fixed as in SPEC. This yields **33 candidates**.
Executable candidate identity binds family, ordered dictionary hash, all parameters, preprocessing, seed 0, projection and feature/source identities. Section 6's `candidate_spec_id` is a precursor, not this fit identity. No greedy subset
search, recursive elimination, global importance screen or outer-score tuning.

Inner years/complete labels precede the outer fit cutoff; expanding training uses earlier, strictly available labels;
keep all horizons of an event together and preserve existing label-window purging.
Select once per outer year by pooled inner event/horizon RMSE, exact ties by canonical
candidate ID. Fit the selected recipe on outer training only; outer outcomes never
choose subsets/parameters. Latest outer inner-selection is the prospective winner,
not the candidate retrospectively best across outer scores. Preserve current
attempt/failure/zero-retry and infrastructure-failure-blocks-selection policies.

## 3. Quantiles and optional scope (designed, deferred beyond core)

A separate follow-on recipe uses sklearn HGB quantile loss at q=[0.1, 0.5, 0.9],
with the same causal table and nested folds. Freeze its own small grid and fit cap
before execution. Select each quantile by inner pinball loss; report pinball,
80% interval coverage AND width, crossing frequency and nonnegative support.
Nonnegative projection is fixed; no post-hoc crossing repair or calibration on
outer labels. Any trained calibration belongs in inner training with its own budget.
A median/quantile is not entered as the primary mean challenger or scored as a mean.
Allow roughly 1–2 additional engineering days after core contracts, subject to review.

Defer RF/kNN, neural/distributional models, adaptive search/pruning, stacking,
sector provenance research, chain features and Tier4 OOF regeneration. These are
optional extensions, not prerequisites for a credible first magnitude search;
several together exceed the one-week target. No new sources or purchases are in scope.

## 4. Proposed preregistration and legacy comparison

This is a **request design**, not accepted v1 JSON, a sealed registration or a run.
Implement a closed versioned v2 request rather than reinterpret v1. Standard-library
JSON plus existing domain validation/canonical hashing remains sufficient; a separate
schema dependency does not replace temporal/identity/budget checks.

The proposed request has required keys:
`schema_version=magnitude_search_request.v2`, `feature_snapshot_id`,
`history_snapshot_id`, `feature_dictionary_hash`, `eligibility_policy_ref`, `feature_sets`, `space`,
`outer_years`, `inner_years`, `max_candidates`, `max_fits`, `comparison`, `limits`.
All snapshot/hash fields are immutable resolved references, never "latest". `space`
contains exactly the three family grids above; `feature_sets` contains H/HM/HMO.
The anchor registration separately binds population, target, holdouts and clocks.
Unknown keys, missing axes, duplicated configurations or disagreement with that
anchor refuse before outcome reads. Registration/search/comparison remain no-ledger,
using the existing native catalog, not legacy LEDGER.csv. Each arm reserves
run_id = "native_" + content_hash({experiment_id, arm}) without the sha256 prefix.
The transaction admits only exact canonical-binding replay; changed bindings, existing
spec reservations or key collisions refuse without overwrite. One plan per anchor and
(run_id, slot) journal keys bind the v2 request/plan hash and limits. Artifact publication
precedes catalog commit; orphan bytes confer no reservation. Replay never restarts fits.

Proposed development schedule: outer_years ["2022", "2023", "2024"]; inner_years
{"2022": ["2020", "2021"], "2023": ["2021", "2022"], "2024": ["2022", "2023"]}.
Training begins at the anchor's explicit training_start, before the first inner year.
These are prospective choices, not verified coverage. Metadata-only admission must
confirm source coverage and holdout eligibility before sealing; insufficient years
block rather than silently disappear. Any amended years/grid require a new request
and a new experiment ID, because the reservation identity omits the request hash,
before the new evaluation. The runtime PR must accept `--no-ledger` and refuse any
ledger write as an explicit acceptance criterion. Prior development reuse must be disclosed, not described
as untouched confirmation. Protected holdouts never become substitute years.

Set max_candidates=33 and search max_fits=603:
3 horizons × (33 candidates × 6 inner folds + 3 selected outer fits).
Separately cap the fixed legacy baseline at 9 logical outer fits (18 component fits
for its OLS+MLP blend); the nonfitted history mean adds no estimator fits.
Thus at most 612 logical fits / 621 component fits for search plus comparison.
Count all attempts, including failures. Limits explicitly bind positive child memory,
CPU and wall seconds, parent/build memory/wall ceilings and total job wall budget.
Actual numeric resource limits must be supplied from the authorized local pilot
before registration; no executable request or permission is implied by this template.
No automatic cap increase, retries, extra seeds or additional candidates.

`comparison` fixes reference `size_v1_4_causal_refit`, plus current history mean.
Reproduce the pinned incumbent feature list and equal-weight OLS/standardized-MLP
recipe with native estimator math, freshly fitted on each outer training fold with
the pinned legacy seed (currently 20260829), no tuning. Bind recipe hash and all
component parameters: MLP widths [64,32], max_iter 400, early_stopping true,
n_iter_no_change 15; its validation split stays within qualified training only.
Preserve n_prior>=4, complete-input training admission, minimum 500 training rows
and legacy training value masks (including abs_move in [0,200]). The target mask
never filters evaluation targets or hides large misses. Preserve predictor masks
or_implied [0,60] and or_rvol30 [0,700] at training and prediction. This preserves
historical-evaluation predictor masking (prepare precedes walk_forward's split),
not the live score path, which checks finiteness without these bounds. Under the
lossy-source policy above, retain a zero sentinel only if the pinned stored source
preserves it; normalized nulls stay null and cause baseline abstention. Preserve
has_implied_quote=(or_implied>0), computed from the source quote before masks,
not HMO's null-zero encoding. Unavailable-input/insufficient-history
predictions abstain explicitly. Do not apply the training target bound at prediction
time. Preserve finite raw blend predictions (including negatives), as the native
legacy estimator does; do not silently apply the challenger's zero projection. Reconstruct all fourteen
incumbent inputs at the same 5/3/1 cutoffs, including audited implied-history,
quote-presence and absolute aliases. Its historical-implied aggregate needs its own
protected/availability-qualified prefix, not the production reader's aggregate.
No old weights/panel forecasts/latest Tier4 release are reused. If causal inputs are unavailable,
report abstention; never relabel a different recipe or substitute history mean. This is a protocol-
matched reconstruction, not a claim to reproduce the reported 498-event result.
Matched event IDs do not repair the current v1 regime-source difference above. The legacy
arm must retain its causally qualified benchmark semantics; if those inputs are
unavailable, report the legacy comparison unavailable rather than substitute SPY
spot silently. A separately approved source-changed research comparator requires
a distinct pinned recipe and an explicit changed-input label; it cannot replace
the legacy arm or establish exact source parity on the matched set.

Freeze a common eligible event manifest before labels. Store forecasts before
opening evaluation outcomes through existing admission. For the main paired report,
match canonical event IDs with all three horizons and valid targets/predictions for
both selected challenger and legacy. No outcome-dependent matching. Separately report
full eligible population counts, coverage, abstentions and every exclusion reason by
arm/year/horizon/quote availability; do not claim universe superiority from a sparse
intersection. History mean remains the broad-coverage sanity reference.

Primary: paired difference in pooled original-unit RMSE (challenger minus legacy).
Secondary: paired MAE difference, per-year/horizon errors, signed bias and large-move
underprediction. Keep each event's three horizons together and give each event equal
total weight. Prespecify 2,000 paired calendar-quarter block bootstrap resamples,
seed 0, recomputing both metrics on identical sampled events; report percentile 95%
intervals, block/event counts and issuer-cluster bootstrap sensitivity. No independent
row bootstrap or claim that intervals account for every prior adaptive design choice.
A negative difference favors the challenger. No significance-driven stopping,
deployment, final holdout read or release follows automatically from this experiment.

The owner accepted the core and causal legacy-refit approach on October 21. Before
sealing, still confirm exact immutable inputs, coverage/year feasibility and numeric
resource limits, and verify the implementation matches the accepted baseline contract.
RMSE remains primary unless explicitly amended before outcomes. A practical minimum
gain/promotion rule is a separate future decision; this run reports evidence only.

## 5. Small PR sequence and acceptance

The [confirmed-event metadata proof checkpoint](CONFIRMED_METADATA_DESIGN.md)
proposes a metadata-first split of the historical-implied prefix work. It proves
only complete pinned-view membership; coverage and numerical admission remain separate.
Its 1.B*, 1.M*, P.M*, 1.C* and 2.M* labels refine rows 1, P and 2 (parity P1/P2).
This section owns scientific scope/dependencies; the checkpoint's mapped estimates
supersede these original coarse estimates for its named remaining work, not fit
inside them. Unlisted work is additional; the original one-week target is not reaffirmed.

This is the original core roadmap. The section-1 parity P1–P3 substeps refine
its named slices; row 0's estimate describes the original checkpoint, not the
size of this later amendment or the separate request-compiler checkpoint.
Each runtime PR targets <=400 added non-test code lines/<=120 doc lines; split add/wire further as needed with contract tests.
Estimates exclude full-universe jobs/review queues. Ranges count added code/tests/docs, not deletions.
Re-estimate before each PR and split at repository limits; these estimates grant no waiver.

| Order | PR / dependency | Acceptance | Added lines: code / tests / docs | Rough effort |
| --- | --- | --- | --- | --- |
| 0 | Initial design + SPEC link (historical estimate) | Design reviewed before code | 0 / 0 / 280–300 | Initial PR |
| 0C | Section 6 checkpoint | Publish before compiler code | 0 / 0 / 120–150 | This design refinement |
| P | Shared versioned eligibility policy / 0 | Grouped ticker/time exclusions, stable eval membership, cross-consumer identity/admission tests; no v1 migration | 200–320 / 250–400 / 35–60 | 0.5 day |
| 1 | Versioned panel schema + metadata-only plan | Closed dictionary, protected fragment refusal, identity tests | 220–340 / 250–400 / 30–60 | 0.5 day |
| 2 | Batched causal feature materializer / 1,P | Horizon/availability, poisoned future/protected rows, batch invariance, read-count tests | 300–400 / 350–550 / 40–70 | 1 day |
| 3 | Stored reader + snapshot join / 2 | Missingness/coverage and mismatch refusal; no runtime build | 200–320 / 250–400 / 30–60 | 0.5 day |
| 4 | ElasticNet recipe + subset identity / 3 | Train-only preprocessing, convergence/failure, sparse selection tests | 160–280 / 200–350 / 25–50 | 0.5 day |
| 5a | Pure request/schedule compiler / 0C,1,P | Closed specifications/counts, identity, no source reads or execution | 180–280 / 250–400 / 30–60 | 0.5 day |
| 5b | v2 bounded planner/executor wiring / 5a,4,P | Fixture-specific 33/603 counts, temporal rejection, immutable replay | 280–400 / 300–500 / 40–80 | 0.5–1 day |
| 6 | Causal legacy comparison adapter / 3 | Pinned recipe parity, same clocks, protected implied history | 250–380 / 300–500 / 35–70 | 0.5–1 day |
| 7 | Matched report + request fixture / 5b,6 | Paired membership/weights, bootstrap determinism, coverage, receipt checks | 220–350 / 250–400 / 40–80 | 0.5 day |

PRs P, 2 and 6 are schedule risks. If they exceed their slices, narrow optional
feature blocks before sealing; keep causality and the legacy comparison mandatory.
After code/review: authorized metadata/pilot checks, publish the concrete request for
review, then separately register and let the user run locally. No actual data read,
fit, registration or runtime implementation is part of this design PR.

## 6. Stage-2 request and schedule compiler checkpoint

Pre-code refinement of sections 2, 4 and 5 above.
This checkpoint specifies a pure, add-only precursor to slice 5, not a runnable
search, registration format migration or permission to read data. Existing v1
APIs, source closures, artifacts and identities remain unchanged.

### Boundary and reuse

Proposed public API: `experiments.magnitude_search_request.compile_request(request)`.
Input is a detached plain-JSON document, not JSON text, a file or a repository.
Only built-in dict/list/string/int/float/bool/null types are allowed; dictionary
keys are strings, numbers finite, and cycles/non-JSON aliases refuse. Integer
fields exclude booleans. Any future text/CLI parser must reject duplicate keys;
the current object-only API cannot recover duplicate keys already discarded.
Standard-library JSON/product and foundation canonical hashing suffice; use the
existing panel `recipe(subset)`, shared `eligibility_policy`, and public child
`validate_execution_limits`. Do not duplicate membership or source admission.

The compiler makes no estimator, reads no catalog/source/artifact, and writes
nothing. It does not qualify events/labels, enforce embargoes, learn preprocessing,
score/select candidates, reserve an experiment, or launch a worker. References
bind declared intent; syntax and a matching digest prove neither existence nor
provenance. No runtime consumer is introduced in this slice.

### Closed request encoding

Keep exactly the thirteen top-level keys named in stage-2 section 4:
`schema_version`, `feature_snapshot_id`, `history_snapshot_id`,
`feature_dictionary_hash`, `eligibility_policy_ref`, `feature_sets`, `space`,
`outer_years`, `inner_years`, `max_candidates`, `max_fits`, `comparison`, `limits`.
The schema literal is `magnitude_search_request.v2`. No defaults or extra keys.

| Field | Exact admission |
| --- | --- |
| Snapshot IDs | Strings matching native `snap_` plus 32 lowercase hex characters; no `latest`, path or resolver call. Full committed-reference comparison belongs to later admission. |
| Hashes | `sha256:` plus 64 lowercase hex characters. |
| `feature_dictionary_hash` | Existing compiler: exact full HMO v1 recipe hash. Proposed P3a wrapper (6.1): exact full HMO v1 or v2 recipe hash. Dictionary compatibility is not snapshot membership. |
| `eligibility_policy_ref` | Exactly `{"schema_ref":"group_eligibility_plan.v1","plan_id":<hash>,"policy_id":<hash>}`. Policy ID must equal the shared `grouped_ticker_embargo.v1` policy with 30 calendar days. This identity-only plan reference is not an ArtifactRef or read capability. |
| `feature_sets` | Exactly `["H","HM","HMO"]`, in that order; no arbitrary subsets/columns. The owning panel helper supplies each ordered dictionary and quote/missing policy. |
| `space` | Exactly the three grids below. Integer/float aliases for numerical grid values normalize to the shown numeric types. Object-key order (families and axes) is irrelevant; changed/missing/extra axes and reordered/duplicate/changed/missing values refuse, as do booleans. |
| `outer_years` | Nonempty strictly increasing unique four-digit ASCII year strings, from `0002` through `9998`. |
| `inner_years` | Object with exactly the outer-year keys; each value is a nonempty strictly increasing unique list of valid earlier year strings. No event coverage or cutoff availability is inferred. |
| `max_candidates` | Explicit positive integer cap, at least the calculated Cartesian-product count, checked before enumeration. The closed grid still contains only 33 candidates; a larger cap never adds candidates. |
| `max_fits` | Explicit positive integer cap, at least the calculated search fit count. It counts attempted logical horizon fits, including failures; this compiler performs none. |

```json
{
  "ridge": {"alpha": [0.1, 1.0, 10.0]},
  "elasticnet": {"alpha": [0.01, 0.1, 1.0], "l1_ratio": [0.25, 0.75]},
  "hgb": {"max_leaf_nodes": [7, 15]}
}
```

`comparison` has exactly `reference="size_v1_4_causal_refit"`,
`sanity_reference="history_mean"`, `max_logical_fits` and `max_component_fits`.
Both caps are explicit positive integers; they must cover respectively three and
six times the outer-year count. Each legacy horizon fit is one logical fit and
two OLS/MLP component fits. History mean consumes no estimator fits. These names
declare the accepted baseline protocol, not an implemented baseline recipe.

`limits` has exactly four objects, with exactly these keys:

```text
child:  memory_bytes, cpu_seconds, wall_seconds
parent: memory_bytes, wall_seconds
build:  memory_bytes, wall_seconds
job:    wall_seconds
```

Every value is a built-in positive integer below `sys.maxsize`; no null/unbounded
value, coercion or implicit default. Reuse the current public child-limit
validator `validate_execution_limits`; validate parent/build/job shapes separately with
the same integer domain, without calling the CPU-resolving `execution_limits`. Child limits
describe one bounded trial/outer worker invocation, potentially containing three
horizon fits, not three separately allocated budgets. Other limits describe the
parent orchestrator, panel build and complete job respectively. This stage only
binds declarations, not resource containment, feasibility or a measured pilot.
Synthetic fixture values are not approved real-data budgets.

### Deterministic preview, not a sealed plan

Return a detached document with `schema_version="magnitude_search_schedule.v2"`,
the normalized `request`, `request_id=content_hash(request)`, `horizons=[5,3,1]`,
`candidate_specs`, `schedule`, and `counts`. Each candidate specification contains
`family`, the owning `feature_recipe` for its subset, `parameters`, and
`candidate_spec_id=content_hash({request_id, family, feature_recipe, parameters})`.
Sort specifications by that ID. Parameters contain only the declared grid knobs;
these are not executable fit recipes or complete model/fold/source identities.
Fixed estimator/preprocessing knobs must be bound by later recipe assembly.

`schedule` is ordered by outer year, each entry containing `outer_year` and its
ordered `inner_years`. It carries no event IDs, target availability or fit cutoff.
Let C be the candidate count, I the sum of inner-year counts, O the outer count:
`counts` contains `candidates=C`, `inner_folds=I`, `outer_folds=O`,
`search_fits=3*(C*I+O)`, `baseline_logical_fits=3*O`,
`baseline_component_fits=6*O`, `total_logical_fits=search_fits+3*O`, and
`total_component_fits=search_fits+6*O`. Caps are checked before expansion.
The proposed schedule has `candidates=33`, `inner_folds=6`, `outer_folds=3`,
`search_fits=603`, `baseline_logical_fits=9`, `baseline_component_fits=18`,
`total_logical_fits=612`, `total_component_fits=621`; 603 is not universal.

Malformed declarations, incompatible dictionary/policy, year-order violations,
or insufficient caps raise nonretryable `INVALID_EXPERIMENT_SPEC`, with no partial
preview. Identical semantic requests replay identically; dictionary-key order is
irrelevant, list order is contractual. Changed valid references, years or limits
change request/spec IDs. The later registrar must still enforce a new experiment
ID on amendment and authenticate all references against the anchor.

### Acceptance and implementation split

This checkpoint's compiler, owning architecture contract and synthetic tests
landed in #1236. The original implementation estimate was 180–280 added code /
250–400 test / 30–60 documentation lines; no new dependency or framework.
Its tests cover exact grid/count arithmetic, all six ElasticNet pairs, smaller valid
schedules, aliases, malformed nested documents and cap boundaries, dictionary
and 30-day policy binding, deterministic non-mutating replay, identity changes,
and poisoned repository/store/fit hooks. Verify v1 source closures stay unchanged.
Native stored-panel admission/publication, fitted Ridge/HGB column generalization,
ElasticNet recipe dispatch, journaling/selection and resource-contained execution
remain later slices. No dummy runtime wiring, quantitative model recommendation,
actual registration, protected read or real-data job is authorized here.

### 6.1 P3a: additive dictionary-version compatibility checkpoint

This amendment changes dependency staging, not the search or its scientific rules.
The v2 dictionary/recipe definitions landed in #1293; the metadata prerequisites
in #1327/#1331 neither implement a stored panel reader nor qualify numerical inputs.
P3a needs only the definitions and existing pure compiler. P1/P2 and the qualified
reader remain prerequisites for P3b/execution, not for declaration-only dispatch.
The broader core/add/drop preview in draft #1290 is not the accepted initial search;
replace that scope rather than importing its ten-subset/110-candidate axis.

**Planned public boundary, not an implemented architecture contract:**
`experiments.magnitude_search_request_versions.compile_request(request)`.
Keep `magnitude_search_request.compile_request`, both owning panel definition
modules and existing v1 runtime source closures byte-for-byte unchanged. The new
module has synthetic callers only; its implementation PR must add its real owning
ARCHITECTURE/README contract. No CLI, native caller or registration wiring here.

Input/output schemas and all fields remain exactly section 6. The dispatch key is
only `feature_dictionary_hash`, compared with the full HMO hashes from public
`magnitude_panel_table.recipe("HMO")` and `magnitude_panel_v2_table.recipe("HMO")`.
No version field, default, `latest`, subset hash, opaque model ID or arbitrary
column list may substitute. A dictionary version is not another search axis.
Use the corresponding public `recipe(subset)` for every H/HM/HMO specification;
even H's unchanged values do not make a v2 recipe identical to v1. Preserve v2
11/28/42 raw columns and its HMO-only quote indicator as supplied by the recipe.

**Reuse without mutating the old compiler.** Standard-library JSON and foundation
canonical hashing already cover normalization and identities; no new dependency.
The wrapper recognizes the exact v2 hash only in a built-in dict with a built-in
string hash. All other inputs delegate unchanged to the public v1 compiler,
preserving its valid outputs and complete typed refusals. For a recognized v2
declaration, make a private shallow dict copy and substitute only the known v1
HMO hash for one call to that public compiler. This is a validation/normalization
adapter, not a source identity or admitted request; it must not escape.

That call must finish its existing plain-JSON, nested shape, grid, policy, year,
cap and limit checks before any v2 candidate expansion. It supplies a detached
normalized request and the existing schedule/counts. Restore the actual v2 hash
in that normalized request, recompute `request_id=content_hash(request)`, replace
each candidate's feature recipe via its declared subset, recompute every
`candidate_spec_id=content_hash({request_id,family,feature_recipe,parameters})`,
and re-sort candidates by the new IDs. Return a detached result; never retain a
surrogate request ID, v1 recipe or surrogate candidate ID in the v2 output. Do not
reach into private compiler helpers, monkeypatch recipe globals or duplicate the
closed-grid validator. Invalid input produces no partial preview or mutation.

| Condition | Required outcome |
| --- | --- |
| Valid full-HMO v1 hash | Exactly the existing compiler's canonical bytes, counts and IDs. |
| Valid full-HMO v2 hash | Same closed 33 candidates and fixture-specific 603 search fits; actual v2 recipes/request determine every ID. |
| Unknown/malformed/subset hash, malformed JSON shape, changed subsets/grid, invalid policy/years or insufficient caps | Existing nonretryable `INVALID_EXPERIMENT_SPEC` from public validation; no enumeration beyond that validator's ordering. |
| Numerically equivalent permitted grid aliases or object-key reordering | Same normalized version-specific result; list order remains contractual. |
| Valid snapshot/policy-reference syntax | Intent only: no source resolution, existence, membership, coverage, provenance, forecast or holdout qualification. |

The private normalized intermediate and final preview coexist: work/space scale
with the supplied declaration plus the fixed 33 metadata specifications. Existing
candidate/fit caps bound enumeration, not input bytes, construction memory, RSS,
CPU feasibility or real-data cost. No source I/O, estimator construction, fitting,
preprocessing, selection, artifact writes, registration or worker launch occurs.
Inner-fold selection and outer reporting are unchanged later execution rules.

**Independent acceptance oracles.** Pin the complete existing v1 canonical output
and source-file hashes; compare all valid v1 fixtures and complete refusal Problems
through both public APIs. For v2, assert literal H/HM/HMO descriptor/column order,
quote placement, the 33/603 and 9/18 comparison counts, and a smaller valid schedule.
Recompute every returned identity from the actual declarations independently and
assert known v1/surrogate IDs do not survive. Cover changed references, numeric
aliases, dictionary-key order, unsupported/subset hashes, malformed/cyclic/non-JSON
inputs, closed-grid/subset refusal, cap boundaries before expansion, detached
mutation tests, and poisoned source/store/fit hooks. No real data is needed.

P3a implementation estimate: 50–80 runtime / 180–240 tests / 20–35 docs, at most
355 combined added lines; split before exceeding 400, never weaken the oracles.
Publish this design amendment before implementation. A clean replacement for
#1290 preserves its old head/history and is preferable to rewriting unrelated
work. Actual execution still needs P3b, the admitted causal context/materializer,
protected-history and qualified ^GSPC-close evidence, full recipe/source/fold
bindings, matched comparison and separately approved real-data budgets/registration.
