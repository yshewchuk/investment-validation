# Project assessment — September 5, 2026

## Scope and provenance

This report records the project assessment and subsequent remediation estimates requested by the user. The assessed revision is `6cdcfcd` (Every Tier-4 column names the champion that produces it). Source line references below refer to that revision and may move as code changes.

The assessment covered the research engine, replay and evaluation paths, feature construction and model training, data storage and publication, scorer, dashboard and forward ledger, experiment governance, and relevant tests and research artifacts. It is a broad review, not a claim that every legacy script or dataset row was exhaustively verified.

The assessment made no source or data edits and made no vendor API requests. A nightly build was active during part of the review; its mutable data counts are explicitly provisional. Its observed PID was no longer running at the final status check, but successful completion was not established. The worktree was clean before this report was added. Adding, committing, and publishing this document was separately authorized; no fixes are included.

The full unit suite run during the assessment completed with **1,544 passed, 53 warnings, in 274.79 seconds**. This report does not represent a fresh test run during publication.

## Assessment

The most serious problems affect the validity of the validation machinery itself. The forward ledger can settle a different trade from the one recorded; historical scorer calibration loads future-trained artifacts; production causality receipts use manufactured timestamps; and readers can observe partial or mixed dataset generations during publication.

The core project contains useful replay logic, walk-forward modeling, data validation, fill-sensitivity analysis, and candid research documentation. These findings do not establish that profitable strategies are absent. They establish which claims and measurements need correction before they can support that conclusion in either direction.

In particular, current ledger outcomes should be treated as simulated, scorer-level historical calibration should not be treated as fully out of time, and cashflow Monte Carlo outputs should not be treated as validated portfolio-risk estimates.

Severity indicates impact on experimental validity or operational correctness. A critical architectural risk does not mean that every stored dataset is currently corrupt; an invalid validation method does not mean that every underlying model metric is invalid.

## Findings

### F01 — Critical: forward settlement drops recorded contract identity and reprices entry

**Evidence.** [engine/ledger.py](engine/ledger.py), around lines 343–362, records strike, expiry, entry/exit information, intended alpha, and entry cost. In `score_outcomes`, around lines 472–560, settlement calls the same replay engine used for backtests. The constructed frame omits strike, expiry, variant, and structure parameters. Results are matched by `(strategy, event_id, fill_alpha)`, and the outcome uses the fresh replay return, entry cost, and exit value. The frozen intended entry price is not used to calculate the result.

**Impact.** Alternate strikes or expiries can settle against the default reconstructed trade or remain unresolved. Changes in raw data, normalization, or strategy defaults can alter entry pricing after the prediction was frozen. Parameterized structures cannot be reliably settled from this reduced identity.

The observed ledger contained 165 predictions and 139 event/strategy groups; 21 groups contained multiple distinct contracts. Examples included AMBA STR-THRU strikes 68, 67, and 63, and COST STR-RUNUP contracts with different strikes and expiries. These observations demonstrate why event/strategy identity is insufficient; they do not by themselves prove that every such row has already received an incorrect outcome.

The resulting P&L is a simulated ORATS quote-fill replay. It is not an actual brokerage fill or an independently observed Polygon transaction price. Polygon daily close/VWAP would provide independent market-price evidence, but would still not prove that a particular multi-leg trade could have executed simultaneously at those prices.

The event-date-change check around line 561 is ineffective in the real path: replay receives the frozen event date and echoes it, so comparing that value with the frozen row does not discover a moved canonical event. The test at [tests/test_ledger.py](tests/test_ledger.py), around lines 325–335, supplies a changed date through a mock without exercising actual calendar reconciliation.

**Remediation.** Settle every frozen leg and quantity, preserve original entry evidence, reconcile calendar changes explicitly, and distinguish simulated, independently observed, and actual executed outcomes. Corrections should retain an audit trail rather than silently rewrite original observations.

**Validation and reprocessing.** Test two contracts for the same strategy/event, a moved event, a revised vendor entry quote, and a changed strategy default. Reconcile existing ledger entries where original evidence is available. Missing historical execution evidence cannot be recreated reliably.

### F02 — Critical: historical scorer calibration uses artifacts trained on later years

**Evidence.** [engine/score.py](engine/score.py), around lines 582–592, loads the current champion. [engine/models/training/common.py](engine/models/training/common.py), around lines 225–234, fits final artifacts on all complete rows. [checks/phase1_calibration.py](checks/phase1_calibration.py) describes an out-of-sample exercise but constructs the current scorer and re-scores historical 2023+ events.

Historical cutoffs in payoff-map and analog selection do not make the loaded final model causal. The size/implied-move model, gate model, fitted transformations, and pooled residual distributions can contain information from years after the historical prediction date. The gate scorer also uses the current artifact.

**Impact.** End-to-end historical Brier scores, reliability curves, and scorer calibration claims are partially in sample. An `as_of` argument does not by itself select an artifact trained before that date. Independently computed model walk-forward metrics may remain valid; they need to be distinguished from this scorer-level evaluation.

**Remediation.** Select models, fitted transformations, thresholds, and residual pools by their true training cutoff. Historical evaluation should use the same information boundary that deployment would have had.

**Validation and reprocessing.** Assert that adding future data cannot change a past score under a pinned historical generation. Reconstruct or retrain missing historical folds and rerun affected scorer calibration and historical predictions.

### F03 — Critical: production causality auditing does not preserve actual feature provenance

**Evidence.** [engine/features.py](engine/features.py) contains source-stamp machinery, but the production scorer does not consistently use it. [engine/score.py](engine/score.py), around lines 1117–1132, builds features at the structure entry date. Around lines 1197–1214 it assigns the checked feature stamps that same entry date and audits against that date. This cannot detect a late join because the timestamps were assigned after feature construction rather than carried from their sources.

The audited set is also incomplete: the scorer focuses on event-history and daily-state fields, while other model inputs and derived values are not covered by a production completeness check. `FeatureVector.assert_complete` exists in [engine/audit.py](engine/audit.py), around lines 125–132, but was not called by the production scorer path inspected.

`score_calendar(as_of=...)` constructs requests with `as_of=None`, around lines 1948–1957 of the scorer. The result then adopts the structure decision/entry date. Explicit earlier request cutoffs also do not consistently constrain feature construction, which remains tied to entry date.

**Observed example.** An existing September 5 board had 128 of 184 scored driver rows with entry dates after the board cutoff. AVO STR-THRU for September 8 reported September 8 for `as_of`, evidence cutoff, and model-input cutoff despite being generated on September 5 using a September 3 quote.

**Impact.** Future preview rows misstate evidence availability. Re-running historical or backfilled requests can consume later data when that data is present. A preview generated before those later observations exist is not automatically proof of actual future-data consumption, but its provenance is still incorrect.

[engine/audit.py](engine/audit.py), around lines 312–353, permits future snapshot cutoffs relative to wall clock. [engine/ledger.py](engine/ledger.py), around lines 399–406, copies a board-level audit receipt onto entry rows. Such receipts do not establish each trade input was available at decision time. Nightly only freezing entry-dated rows limits one preview-related exposure, but does not repair the public historical scoring API or the audit mechanism.

**Remediation.** Propagate the caller cutoff consistently, carry true observation/availability stamps through feature construction, audit every consumed feature, and distinguish planned entry dates from evidence dates.

**Validation and reprocessing.** Test actual production scorer inputs with deliberately late source rows and an explicit cutoff earlier than entry. Regenerate affected historical scores and calibration after fixing F02 and F03 together.

### F04 — Critical architectural risk: dataset publication is not transactional

**Evidence.** [engine/data/store.py](engine/data/store.py), around lines 137–156, drops an existing table when opening a replacement writer. Individual file writes use a temporary file and rename, but this does not protect a whole table or generation. Finalization deletes old parts before replacements, around lines 221–239. `__exit__`, around lines 243–247, closes/finalizes even after an exception, allowing partial output to be published. `write_table`, around lines 254–290, also drops before replacement.

[engine/data/rebuild.py](engine/data/rebuild.py), around lines 340–347, directly overwrites the canonical panel with `to_parquet`. [engine/data/features/tier4.py](engine/data/features/tier4.py), around lines 968–979, writes Tier 4 directly. Rebuild updates multiple artifacts sequentially and publishes the snapshot only at the end. A comment around rebuild lines 368–372 records an earlier failure after Tier 4 changed but before snapshot completion.

**Impact.** Readers can encounter an empty or partial table, mixed table generations, or a new Tier 4 with an old snapshot. A failed build can leave a published partial state. The active nightly made this risk relevant, but the review did not establish that its output was corrupt.

**Remediation.** Build into an isolated generation, validate the full generation, atomically publish a generation pointer, and pin readers to one generation. Retain the previous valid generation for recovery.

**Validation and reprocessing.** Inject exceptions at table, panel, Tier-4, and manifest publication boundaries; verify concurrent readers observe either the old complete generation or the new complete generation. Initialize one complete generation from existing validated data. Renormalizing every vendor payload is not inherently required.

### F05 — Critical scientific-validity concern: live promotion bypassed its registered acceptance bar

**Evidence.** [EXP-131 spec](experiments/EXP-131_held_out_the_trailing_six_month_top_20_g/spec.yaml), around lines 80–101, requires a bootstrap interval excluding zero. The inspected primary-minus-incumbent bootstrap metric had median difference approximately -0.00396, 90% interval [-0.05883, +0.04939], probability positive approximately 0.451, and `distinguishable=false`.

[guides/pnl_gate_promotion.md](guides/pnl_gate_promotion.md), around lines 61–67, explicitly acknowledges that the rule did not clear its own bar. The guide reports a separate CAGR-difference statistic: +9.47 percentage points, 90% interval [-28.99, +43.71], probability positive 66%. These are different reported statistics and should not be conflated; neither establishes the registered distinguishability claim.

[engine/entry_rules.py](engine/entry_rules.py), around lines 173–233, nevertheless implements the live rule. This was documented as a discretionary decision, not concealed. The concern is what promotion status means scientifically, not an allegation that the decision lacked user authorization.

The inspected experiment ledger had 129 rows: 44 planned, 85 ran, one `promoted=true`, and 20 exact duplicate rows. EXP-129, EXP-130, and EXP-131 had planned records without corresponding ran records. EXP-126 results were recorded as not promoted while its structure was live. Structural and entry-rule changes bypass the centralized model-promotion workflow in [experiments/promote.py](experiments/promote.py). The guide also identifies EXP-127 structure confirmation as still unrun.

**Impact.** A live rule cannot be interpreted as having passed preregistered confirmation. The primary held-out result was approximately 29.8% CAGR, 1.54 Sharpe, and 22.8% maximum drawdown under its reported sizing and fill assumptions, but those point estimates do not resolve selection uncertainty.

The positive result is sensitive to sizing. The inspected one-contract premium-weighted return was approximately -2.03%, while fixed-fraction sizing was positive. Those are different portfolios. The negative unsized aggregate does not establish that the fixed-fraction portfolio lost money; it identifies a dependency on sizing cheaper structures up relative to expensive ones.

**Remediation.** Record discretionary exceptions explicitly, require the same evidence ledger for model, structure, and entry-rule changes, and keep the live research designation separate from confirmed promotion. Confirmation must come from additional evidence, not a governance patch.

**Important non-finding.** The custom EXP-129–131 simulator omitted a declared deployment cap, but the inspected held-out series peaked at 40% deployment with eight concurrent trades. A 100% cap would have constrained zero entries. This is a guardrail gap, not an explanation for the reported held-out profitability.

### F06 — High: gate calibration measures a different model from the one selecting trades

**Evidence.** [engine/evaluate.py](engine/evaluate.py), around lines 960–995, calls `gate.predict_proba(test)` before `gate.fit(train)`, then fits for selection. [experiments/common.py](experiments/common.py), around lines 150–169, acknowledges one-fold-stale probabilities in the registered gate state. The first eligible fold has missing probabilities.

**Impact.** Brier and reliability metrics evaluate the prior fitted model while the selected-trade metrics use the newly fitted model. This particular mismatch can be free of future leakage while still failing to measure the deployed selection rule.

[engine/models/training/gate.py](engine/models/training/gate.py), around lines 157–180, chooses one threshold from pooled out-of-sample scores across years, evaluates gated lift with it, and stores it alongside a final all-data model. The pooled score distribution was not available in each historical year. This is not direct target-label leakage, but it is not a threshold that could have been deployed causally in those folds, nor necessarily calibrated to the final model score scale.

**Remediation and reprocessing.** Fit, predict, and select with the same model within a fold; use a threshold fitted only on information available at the decision boundary. Rerun affected gate calibration, selection, and promotion metrics. Per-fold trained-gate paths should be assessed separately rather than assumed equally affected.

### F07 — High: caches and artifacts do not establish reproducible data provenance

**Evidence.** [experiments/common.py](experiments/common.py), around lines 122–142, reuses `gate_dataset_{strategy}.parquet` based on existence. It does not validate data snapshot, feature schema, trade identity, or code revision. Similar existence-only caches occur in custom experiments. Observed caches predated subsequent feature changes.

[experiments/lib.py](experiments/lib.py) records a spec containing `data_snapshot` but does not enforce equality with the active snapshot. [engine/models/registry.py](engine/models/registry.py), around lines 128–163 and 299–329, records model configuration and metrics without training dataset snapshot, feature-generation revision, or environment identity. Artifact hash and feature/role checks do not establish the provenance of the training inputs. No pinned dependency lock was identified for the model environment.

**Impact.** A rerun at the same seed can use stale features, new data, or different library behavior. A model artifact can be internally intact without being reproducible from its recorded metadata. Overwritten dataset generations also prevent reliable restoration of older inputs.

**Remediation and reprocessing.** Bind cache keys and artifacts to dataset generation, feature schema/code, and environment. Verify existing cache provenance where possible and regenerate where it cannot be established. Retrain affected artifacts as needed; a blanket retrain of every artifact is not justified without dependency tracing.

### F08 — High: cashflow Monte Carlo ignores dated capital deployment

**Evidence.** [engine/evaluate.py](engine/evaluate.py), around lines 1035–1134, accepts a Monte Carlo `mode` and reports it, but its compounding helper uses sequential `cumprod(1 + fraction * return)` for both modes. It receives bare returns without entry dates, exit dates, premium requirements, or deployment constraints. The deterministic equity builder in the same file does model dates and overlapping positions.

**Impact.** Outputs labeled cashflow Monte Carlo do not model overlap, committed capital, skipped entries, or cash. Probability of loss, terminal percentiles, drawdown distributions, and sizing curves can disagree with the deterministic cashflow model for structural reasons.

**Remediation and reprocessing.** Carry dated trades into simulations and reuse a consistent portfolio accounting model with an explicitly documented resampling unit. Recompute risk outputs from existing trade records; vendor downloads should not be necessary.

### F09 — High operational risk: conflicting earnings dates are scored and can enter the ledger

**Evidence.** [engine/calendar.py](engine/calendar.py), around lines 583–623, deliberately retains competing dates and flags the conflict. [engine/score.py](engine/score.py), around lines 1894–1901, does not ingest the conflict field into the scoring selection. [engine/dashboard/nightly.py](engine/dashboard/nightly.py), around lines 823–848 and 920–1000, adds warnings after scoring. Ledger snapshotting does not enforce a conflict block.

**Impact.** A row with an ambiguous event date can be recommended and frozen when its calculated entry date arrives. A display warning does not protect an automated trigger.

**Remediation and reprocessing.** Exclude unresolved conflicts from actionable entries while retaining them for inspection. Reconcile affected predictions with dated calendar evidence and regenerate the board. Do not silently choose whichever candidate produces a better outcome.

### F10 — Medium: uncertainty draws include impossible negative moves

**Evidence.** [engine/score.py](engine/score.py), around line 1293, constructs absolute-move and implied-move draws as unbounded point prediction plus residual. Current residual pools have substantial negative support. By contrast, [EXP-129 simulation](experiments/EXP-129_predicted_p_l_by_simulation_repricing_a/simulate.py), around line 222, clips absolute moves to nonnegative values.

In the inspected board, 110 absolute-move rows had a median impossible-negative share of 2.10%, with a maximum of 11.39%; 75 exceeded 1%. Among 74 implied-move rows, the median was 1.44%, maximum 5.25%, and 45 exceeded 1%.

**Impact.** Probabilities and intervals integrate outcomes outside the target domain. Flooring the eventual payoff does not make the driver distribution valid. The resulting bias may be conservative for some structures, but its sign and magnitude must be measured rather than assumed.

**Remediation and reprocessing.** Define and validate an appropriate nonnegative predictive distribution consistently across research and production. Simple clipping is one possible baseline, not automatically a calibrated final solution. Regenerate affected score distributions and calibration.

### F11 — Medium to high: score snapshots omit separate Tier-4 provenance

**Evidence.** [engine/data/manifest.py](engine/data/manifest.py), around lines 35–78, deliberately excludes Tier 4 from the main snapshot and records a separate hash. [engine/score.py](engine/score.py), around lines 1800–1805, returns only the main snapshot hash. The score records model/fold labels but not the complete Tier-4 table and serving-artifact identity needed to reproduce the input.

**Impact.** Forecast-sized trades can change after a Tier-4 rebuild or promotion while retaining the same reported data snapshot. A model name alone is not a content hash.

**Remediation and reprocessing.** Pin the Tier-4 generation and serving artifacts in scores and ledger entries. Backfill provenance only where the original identities can be established; do not assign current hashes to historical results without evidence.

### F12 — Evidence limitation: independently observed execution coverage is thin

The provisional manifest inspected during nightly showed approximately 15,618 `option_daily` rows, 23,051,840 `option_chains` rows, and 174,919 replay trades. These are different units, so their ratios are not a valid fill-coverage percentage. They nevertheless indicate that the independent traded-price evidence is much smaller than the quote/replay universe. An exact overlap join is needed to quantify supported trades.

Polygon daily bars provide independent traded-price observations, not guaranteed executable multi-leg fills or historical NBBO. F01 does not currently close this evidence gap because its settlement uses quote replay again.

[engine/data/validate.py](engine/data/validate.py), around lines 222–253, repairs crossed quotes to the lower quoted side and flags them. This is documented and has received sensitivity analysis; it is a modeling assumption, not automatically a hidden defect. Profitable slices should still report their dependence on repaired rows.

Re-evaluated strategy results should retain worst-case, mid, and best-case fill sensitivity, with measured spread and trade-price evidence where available. The point estimates discussed in F05 are references to existing artifacts, not a new execution-sensitivity study performed by this assessment. No strategy is declared nonviable by this report.

### F13 — Medium maintainability issue: status documentation overstates current guarantees

[README.md](README.md) contains phase/status descriptions that lag the active dashboard and ledger, and claims about empty or future components no longer match the observed system. Statements that leak auditing is asserted on every scoring path overstate the production guarantees described in F03.

Documentation should identify active versus legacy paths and state what an audit receipt, promoted rule, historical score, and realized outcome actually guarantees. Correcting those terms is necessary to prevent misuse while deeper fixes are implemented.

## Test-suite assessment

The passing suite is valuable evidence of implementation stability, but it does not establish the missing scientific and operational invariants.

Observed gaps include:

- Exact frozen-contract settlement and immutable entry-price evidence.
- Separation of simulated outcomes, independently observed prices, and actual fills.
- Historical selection of model artifacts, residuals, and thresholds.
- True source timestamps through the production scorer, including feature completeness.
- Failure recovery and concurrent reads during publication of a full data generation.
- Monte Carlo portfolio accounting under overlap and deployment caps.
- Consistent acceptance enforcement across model, structure, and entry-rule promotion.
- Cache invalidation after data, schema, code, and environment changes.

[tests/test_evaluate.py](tests/test_evaluate.py), around line 256, contains an assertion ending in `or True`, which cannot fail. An additional unconditional fallback was observed in [tests/test_store.py](tests/test_store.py), around line 207. The moved-event test described in F01 checks a mock response rather than the production reconciliation path. Feature-vector utility tests do not by themselves test the production timestamp assignment in F03.

Regression tests should exercise these invariants through the consuming production paths, not merely verify helper behavior or mirror implementation details.

## Remediation size

This is a substantial correctness project, but it does not require rewriting the research engine. Most replay, feature, and training infrastructure can remain. The two broad architectural changes are versioned dataset publication and historically correct scoring. Ledger settlement is important but more contained because much of the required contract information is already recorded.

The estimates below include implementation and meaningful regression tests. LOC means lines added or materially changed, not net repository growth. File counts overlap. Allow approximately plus or minus 40% until the larger changes have an implementation design.

| Area | Production LOC | Test LOC | Files touched | Engineering effort | Reprocessing |
| --- | ---: | ---: | ---: | --- | --- |
| Exact ledger settlement | 250–500 | 200–350 | 4–7 | 3–6 days | Reconcile frozen contracts and recompute recoverable outcomes |
| Historical scoring and causal audits | 500–1,000 | 400–700 | 8–14 | 1–3 weeks | Reconstruct historical folds/residuals; rerun scorer calibration and affected scores |
| Transactional dataset publication | 350–700 | 250–450 | 5–9 | 1–2 weeks | Initialize and validate one complete generation |
| Gate evaluation and thresholds | 100–200 | 150–250 | 3–5 | 2–4 days | Rerun affected calibration, selection, and promotion metrics |
| Cache/model provenance | 300–600 | 200–350 | 6–10 | 4–7 days | Regenerate unverifiable caches; retrain affected artifacts |
| Cashflow Monte Carlo | 250–500 | 200–350 | 3–5 | 3–6 days | Recompute risk reports from dated trade results |
| Promotion controls | 150–300 | 100–200 | 4–7 | 2–4 days | Reconcile experiment records; obtain confirmation separately |
| Calendar conflicts, uncertainty draws, Tier-4 hashes | 100–250 | 150–250 | 4–7 | 1–3 days | Regenerate affected scores, distributions, and boards |

Rounded total: **4,000–7,000 lines across 30–50 distinct files**, including tests. A planning allowance of **5–9 engineer-weeks** covers overlapping implementation work and targeted validation; per-row effort estimates are not additive because several fixes share infrastructure. This is not a measured runtime estimate or a delivery commitment. Vendor acquisition and elapsed forward-test time are additional dependencies.

A contained first package covering ledger correctness, gate evaluation ordering, calendar conflict blocking, and accurate outcome labeling is approximately **900–1,700 lines including tests across 10–16 files**, with roughly one week as an initial planning target. It would reduce immediate failure modes but would not repair historical causality or publication consistency.

## Reprocessing plan and limits

| Layer | Expected work | What can generally be reused |
| --- | --- | --- |
| Raw vendor cache | Pull only missing execution evidence or coverage gaps | Existing immutable payloads; no blanket redownload |
| Normalized tables | Establish a consistent initial generation; validate publication | Existing validated tables, unless a separate normalization defect is found |
| Feature matrices | Regenerate caches whose inputs/schema cannot be established | Matrices with verifiable matching provenance |
| Models and residual pools | Restore or retrain historical folds; identify affected final artifacts | Independently valid, reproducible walk-forward artifacts |
| Historical scoring/calibration | Rerun after artifact and feature cutoffs are corrected | Underlying raw observations and valid trade payoffs |
| Gate experiments | Rerun paths affected by stale probabilities or pooled thresholds | Experiments with separately verified causal fitting and thresholds |
| Portfolio-risk reports | Recompute simulations with dated deployment accounting | Existing trade returns, entry/exit dates, and costs |
| Ledger | Reconcile the 165 observed predictions where evidence permits | Original frozen contract and entry evidence; preserve it |
| Dashboard | Regenerate after corrected scoring and provenance are available | Confirmed calendar observations and valid source quotes |

Feature/model reprocessing is the largest uncertainty. Some old caches can be verified from preserved inputs; others cannot. Fixing provenance prospectively does not retroactively establish the origin of an old artifact. Likewise, code changes cannot recreate missing original fills or prove that a historical prediction was made before an event.

Existing experimental outputs should retain their original identities and receive superseding evaluations with explicit reasons. Corrected code should not silently confer validity on old reports. A complete repeat of every experiment is not justified without tracing which inputs and evaluation paths each one used.

No reprocessing runtime was benchmarked during this assessment. There is no evidence that the proposed fixes require another download of the full option-chain history. Additional independent traded-price acquisition, if pursued, remains subject to the documented Polygon entitlement and rate limits.

## Recommended order and completion criteria

1. Correct ledger identity and entry accounting; label outcome evidence accurately. Completion requires exact-leg settlement that survives changes in defaults and later vendor revisions.
2. Enforce causal artifact selection and real feature provenance together. Completion requires that future data and later model promotions cannot change a pinned past prediction.
3. Publish complete dataset generations transactionally. Completion requires interruption and concurrent-reader tests demonstrating no mixed or partial visible generation.
4. Correct gate calibration and thresholds, then rerun affected measurements. Completion requires that reported probabilities and selections come from the same causally fitted model.
5. Add artifact/cache provenance and regenerate unverifiable dependencies. Completion requires reproducibility from pinned data, code, schema, and environment.
6. Correct cashflow simulations and regenerate risk reports. Completion requires consistent dated deployment accounting with the deterministic portfolio path.
7. Reconcile promotion status and obtain the missing confirmation evidence. TWIN-P5 should remain identified as a provisional research deployment until the relevant confirmatory and execution evidence supports a stronger claim.

These changes make the research conclusions more dependable while preserving useful existing infrastructure and data. They do not prescribe a market verdict; they define the evidence needed to make one.
