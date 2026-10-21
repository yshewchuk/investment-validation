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
source-session ceiling from already stored, pinned sources. No provider calls,
new data source, production panel migration, or production Tier4 rebuild is needed.

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
before registration. The proposed first dictionary uses only these existing blocks:

| Set | Inputs | Eligibility |
| --- | --- | --- |
| H | Eleven current stored-history columns | Existing history contract unchanged |
| HM | H + signed_streak, ema12r_abs, dist_high, dist_ema, ret5, ret10, ret20; SPY ret21/63/252, dd252, vol5/20/60/252, vol20_rel252 | Audited runup/regime math at each horizon |
| HMO | HM + implied_move, iv10, iv30, exern_iv10, exern_iv30, iee, skew, contango, fwd90_30, fexern90_30, rvol30, spot, mcap_log | Existing daily_market fields via explicit native mapping |

Names in the last row denote research aliases from `DAILY_STATE_FIELDS`, not a
permission to read a same-named realized-event field. Add quote-present indicators
only to HMO, and missing indicators only for each recipe's selected columns; H
remains history-only. These are not extra search axes. A zero implied
quote means unavailable, while a genuine zero target remains valid. Missing market
cap or late capture stays missing; do not use the earliest later retrieval as causal.

Defer daily-state lags, ratios/interactions and sector context pending exact source/availability maps.
Row lags are not calendar lags. No point-in-time sector mapping is established by this audit; do not join today's
sector classification backward. Existing options-derived daily fields are in core;
new option-chain surfaces, interpolation and liquidity engineering are not.

Exclude `or_exern_z252` (quarantined), current-event move/abs_move, post-event IV,
label-derived quality/completion flags, and stored learned Tier4 predictions.
Tier4 stacking requires fold-local/OOF regeneration and verified training cutoffs,
not merely an early forecast timestamp; defer it. The separate legacy benchmark
below may add its audited input aliases without exposing them as search predictors.

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
bounded ticker/year groups, sort each source once, and share bounded SPY/calendar
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
Candidate identity binds family, ordered dictionary hash, all parameters, preprocessing, seed 0, projection and feature/source identities. No greedy subset
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

Proposed development schedule: outer_years [2022, 2023, 2024]; inner_years
{"2022": [2020, 2021], "2023": [2021, 2022], "2024": [2022, 2023]}.
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
or_implied [0,60] and or_rvol30 [0,700] at training and prediction. Keep source
zero-quote versus absent-quote status so the baseline retains its zero sentinel
and has_implied_quote=(or_implied>0), rather than inheriting HMO's null-zero encoding. Unavailable-input/insufficient-history
predictions abstain explicitly. Do not apply the training target bound at prediction
time. Preserve finite raw blend predictions (including negatives), as the native
legacy estimator does; do not silently apply the challenger's zero projection. Reconstruct all fourteen
incumbent inputs at the same 5/3/1 cutoffs, including audited implied-history,
quote-presence and absolute aliases. Its historical-implied aggregate needs its own
protected/availability-qualified prefix, not the production reader's aggregate.
No old weights/panel forecasts/latest Tier4 release are reused. If causal inputs are unavailable,
report abstention; never relabel a different recipe or substitute history mean. This is a protocol-
matched reproduction, not a claim to reproduce the reported 498-event result.

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

Each runtime PR targets <=400 added non-test code lines/<=120 doc lines; split add/wire further as needed with contract tests.
Estimates exclude full-universe jobs/review queues. Ranges count added code/tests/docs, not deletions.
Re-estimate before each PR and split at repository limits; these estimates grant no waiver.

| Order | PR / dependency | Acceptance | Added lines: code / tests / docs | Rough effort |
| --- | --- | --- | --- | --- |
| 0 | This design + SPEC link | Design reviewed before code | 0 / 0 / 280–300 | This PR |
| P | Shared versioned eligibility policy / 0 | Grouped ticker/time exclusions, stable eval membership, cross-consumer identity/admission tests; no v1 migration | 200–320 / 250–400 / 35–60 | 0.5 day |
| 1 | Versioned panel schema + metadata-only plan | Closed dictionary, protected fragment refusal, identity tests | 220–340 / 250–400 / 30–60 | 0.5 day |
| 2 | Batched causal feature materializer / 1,P | Horizon/availability, poisoned future/protected rows, batch invariance, read-count tests | 300–400 / 350–550 / 40–70 | 1 day |
| 3 | Stored reader + snapshot join / 2 | Missingness/coverage and mismatch refusal; no runtime build | 200–320 / 250–400 / 30–60 | 0.5 day |
| 4 | ElasticNet recipe + subset identity / 3 | Train-only preprocessing, convergence/failure, sparse selection tests | 160–280 / 200–350 / 25–50 | 0.5 day |
| 5 | v2 bounded planner/executor wiring / 4,P | Exact 33/603 counts, temporal rejection, immutable replay | 280–400 / 300–500 / 40–80 | 0.5–1 day |
| 6 | Causal legacy comparison adapter / 3 | Pinned recipe parity, same clocks, protected implied history | 250–380 / 300–500 / 35–70 | 0.5–1 day |
| 7 | Matched report + request fixture / 5,6 | Paired membership/weights, bootstrap determinism, coverage, receipt checks | 220–350 / 250–400 / 40–80 | 0.5 day |

PRs P, 2 and 6 are schedule risks. If they exceed their slices, narrow optional
feature blocks before sealing; keep causality and the legacy comparison mandatory.
After code/review: authorized metadata/pilot checks, publish the concrete request for
review, then separately register and let the user run locally. No actual data read,
fit, registration or runtime implementation is part of this design PR.
