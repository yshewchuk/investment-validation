# SUPER SIZE stage 2: causal panel features and bounded magnitude search

Status: proposed design, not implemented or preregistered. Check this document in
before runtime changes. [Current executable contract](SPEC.md) remains authoritative.
This design freezes neither a dataset nor permission to read protected outcomes.

## Priority and evidence

The first milestone is a useful, bounded mean-magnitude comparison, not an exhaustive
model zoo. Target roughly one engineering week for the core below, conditional on
source provenance and review; estimates are sizing guidance, not a delivery promise.
If causal input or baseline work exceeds that budget, ship the safe foundations and
postpone the search rather than relax admission or substitute a weaker comparison.

Main already has eleven stored history inputs, Ridge/HGB, nested annual selection,
sealed requests and serial execution. These are infrastructure, not evidence of a
winning model. The October 21 feedback reports 498 matched events: legacy MAE 2.79,
history Ridge 2.97, paired difference +0.19, 95% interval [+0.10, +0.28]. These
are separately reported rounded figures; raw estimates were not checked. It also
reports a 2,014-event
single-arm resource/replay run and approximately 0.17 seconds/event history build
(118k events: about 5.6 hours). None was independently reproduced for this design;
none substitutes for the prospective nested comparison.

The October 10 original proposal explicitly chose conditional mean and RMSE, with
MAE and quantiles secondary. Retain that decision: target and 5/3/1-session clocks
are unchanged; do not optimize MAE and call the output a conditional mean.

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
manifest, holdout month and membership versions, target/clock/availability policies,
training/history start, evaluation years, feature dictionary and producer/environment
hashes. Each row retains cutoff ID, source-session ceiling, block source date and
availability/knowledge mode, consumed source row/fragment identities, missingness
reason, and a digest. Preserve individual lag anchors, not only their maximum.
Unknown columns/versions, duplicate keys, context mismatch and ambiguous lineage
refuse. Missing optional values are null, not invented zeros or silently backfilled.
A changed recipe or source produces a new immutable identity, never an overwrite.

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

Daily-state 1/5/10-row differences, ratios/interactions and sector context are deferred
until their exact source/availability mappings are proven. Row lags are not calendar
lags. No point-in-time sector mapping is established by this audit; do not join today's
sector classification backward. Existing options-derived daily fields are in core;
new option-chain surfaces, interpolation and liquidity engineering are not.

Exclude `or_exern_z252` (quarantined), current-event move/abs_move, post-event IV,
label-derived quality/completion flags, and stored learned Tier4 predictions.
Tier4 stacking requires fold-local/OOF regeneration and verified training cutoffs,
not merely an early forecast timestamp; defer it. The separate legacy benchmark
below may add its audited input aliases without exposing them as search predictors.

### Admission before data and values

Extend the history store's metadata-first protected-population preflight, not the
production `scan_panel_row` reader unchanged. Classify the random/rolling union
before any historical outcomes or mixed-fragment payloads are decoded. Restrict
physical reads to admitted identities; fail closed if a fragment cannot be isolated.
Exclude protected outcomes from every prior-event statistic, fold, calibration and
selection. Price-derived history/runup must follow the existing protected-window
rules; do not reconstruct a forbidden event move from raw prices as a workaround.
Raw market state needs its own explicit permitted-read contract, not blanket access
because it is called a predictor. Current-event targets remain a separate reader.

For every contributing observation enforce the source-session ceiling and declared
availability cutoff, plus strictly prior event/label availability where applicable.
Retain schedule/availability reconstruction warnings: a source date is not proof of
historical publication. Never upgrade reconstructed data to observed point-in-time.
A later replacement/capture must not make an earlier row eligible in a strict lane.
If existing sources cannot establish the declared lane, refuse or mark unavailable
according to the frozen field policy; report coverage, never weaken policy mid-run.

### Bounded construction, not another per-event history build

Use one metadata/holdout plan and pinned source resolution per build; batch by
bounded ticker/year groups, sort each source once, and share bounded SPY/calendar
state. Advance causal prefixes/search indices across horizons and events. Reuse the
existing immutable history artifact; do not call its producer inside the panel loop
or repeat feature production per candidate/fold. Preserve prefix state across batch
boundaries without retaining the full universe in memory. Reuse an exact successful
receipt; verify bindings before reuse, with no silent source refresh. Batch source
processing, not blindly physical output: the current history store isolates each event
in its own fragment. Panel output may coalesce only where whole-fragment protected
read admission remains provable; otherwise retain per-event isolation.

Acceptance uses synthetic access counters: bounded source scans per ticker/batch,
not per event, candidate or horizon; identical rows/digests across batch sizes,
no future/protected payload reads, and replay without a build. A later authorized
same-cohort pilot records incremental wall/CPU/RSS/read counts separately from
history build, with explicit limits and bounded stderr progress/ETA. The 0.17-second
history figure is a reported reference, not an accepted panel budget. If panel cost
scales multiplicatively, block full-universe execution and fix batching first.

## 2. Models and inner-only selection (core)

Use installed sklearn Pipeline, SimpleImputer, MissingIndicator, StandardScaler,
Ridge, ElasticNet and HistGradientBoostingRegressor; no new ML dependency or custom
optimizer. Reuse current preprocessing contracts. Every fresh inner/outer fit learns
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
Candidate identity binds family, subset's ordered dictionary hash, all parameters,
preprocessing, seed 0, projection and feature/source identities. No greedy subset
search, recursive elimination, global importance screen or outer-score tuning.

Inner years must be earlier than the outer year and their labels complete before
its fit cutoff. Expanding training uses only earlier, strictly available labels;
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
`history_snapshot_id`, `feature_dictionary_hash`, `feature_sets`, `space`,
`outer_years`, `inner_years`, `max_candidates`, `max_fits`, `comparison`, `limits`.
All snapshot/hash fields are immutable resolved references, never "latest". `space`
contains exactly the three family grids above; `feature_sets` contains H/HM/HMO.
The anchor registration separately binds population, target, holdouts and clocks.
Unknown keys, missing axes, duplicated configurations or disagreement with that
anchor refuse before outcome reads. Each comparison arm has its own sealed identity.

Proposed development schedule: outer_years [2022, 2023, 2024]; inner_years
{"2022": [2020, 2021], "2023": [2021, 2022], "2024": [2022, 2023]}.
Training begins at the anchor's explicit training_start, before the first inner year.
These are prospective choices, not verified coverage. Metadata-only admission must
confirm source coverage and holdout eligibility before sealing; insufficient years
block rather than silently disappear. Any amended years/grid require a new request
before the new evaluation. Prior development reuse must be disclosed, not described
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
No old fitted weights, historical panel forecasts or latest Tier4 release are reused.
If exact causal inputs are unavailable, report abstention; never call a different
recipe "legacy reproduction" or replace it with history mean. This is a protocol-
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

Before sealing, owner confirms exact immutable inputs, coverage/year feasibility,
resource limits and the baseline reproduction contract. RMSE remains primary unless
an explicit scientific amendment is registered before outcomes. A practical minimum
gain/promotion rule is a separate future decision; this run reports evidence only.

## 5. Small PR sequence and acceptance

Each runtime PR targets <=400 added production lines and <=120 documentation lines;
split add/wire changes again if needed. Tests accompany the contract they exercise.
Estimates exclude real full-universe jobs and review queue time.

| Order | PR / dependency | Acceptance | Rough effort |
| --- | --- | --- | --- |
| 0 | This design + SPEC link | Design reviewed before code | This PR |
| 1 | Versioned panel schema + metadata-only plan | Closed dictionary, protected fragment refusal, identity tests | 0.5 day |
| 2 | Batched causal feature materializer / 1 | Horizon/availability, poisoned future/protected rows, batch invariance, read-count tests | 1 day |
| 3 | Stored reader + snapshot join / 2 | Missingness/coverage and mismatch refusal; no runtime build | 0.5 day |
| 4 | ElasticNet recipe + subset identity / 3 | Train-only preprocessing, convergence/failure, sparse selection tests | 0.5 day |
| 5 | v2 bounded planner/executor wiring / 4 | Exact 33/603 counts, temporal rejection, immutable replay | 0.5–1 day |
| 6 | Causal legacy comparison adapter / 3 | Pinned recipe parity, same clocks, protected implied history | 0.5–1 day |
| 7 | Matched report + request fixture / 5,6 | Paired membership/weights, bootstrap determinism, coverage, receipt checks | 0.5 day |

PRs 2 and 6 are the primary schedule risks. If either exceeds its slice, narrow optional
feature blocks before sealing; keep causality and the legacy comparison mandatory.
After code/review: authorized metadata/pilot checks, publish the concrete request for
review, then separately register and let the user run locally. No actual data read,
fit, registration or runtime implementation is part of this design PR.
