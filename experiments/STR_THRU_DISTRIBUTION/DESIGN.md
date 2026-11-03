# Distribution prediction first; STR-THRU utility deferred

Original design, 26 October 2026; first-experiment scope amended 1 November 2026. No real experiment is preregistered or activated by this document.

## Current first experiment: prediction only

The user approved evaluating the distribution itself before any STR-THRU evaluation. This amendment supersedes the coupled first-experiment recommendation below: **do not require option quotes, payoff bridges, EV thresholds, trade selection or portfolio books to evaluate prediction quality.** Delivered strategy integration remains separate and unused by the prediction-only workflow. Its historical contracts and the downstream economic sections below are retained, not treated as prerequisites or authorization for this first experiment.

The question is whether the fixed learned distribution of absolute earnings-move magnitude predicts held-out development outcomes better than causal reference distributions. Reuse the admitted event cohort, 5/3/1 horizons, expanding annual outer folds and strictly earlier inner-OOF controls. Score every admitted outer event, not a trade-selected or quote-available subset; preserve explicit missing-label/control/tail exclusions. Keep the existing random/rolling holdout union, embargo and final-read authorization unchanged. Synthetic tests establish implementation correctness, not empirical forecast accuracy.

Compare the raw histogram with a **training-only empirical magnitude CDF**, the same-feature point forecast plus earlier-OOF residuals, and the same-histogram-mean residual shape control. The empirical baseline uses the same outer-training cohort as the fitted pair, never evaluation outcomes. Retain h1-primary CRPS and the existing shape contrast; report all prespecified baseline comparisons and h3/h5 robustness, rather than selecting the best outer result. RPS, common-bin log loss and fixed-threshold tail calibration supplement CRPS; no single coverage or accuracy number is sufficient.

Add predictive interval coverage **and width**, and PIT/reliability diagnostics with explicit discrete-atom/tail semantics, before calling the forecast-validation report complete. These are diagnostic measurements, not a fitted calibrator or new probability-smoothing policy. The first candidate remains fixed and explicitly uncalibrated; any later model search, calibration fit or changed bins must use earlier inner data and its own sealed recipe/budget. No new success cutoff, promotion rule or claim of validated calibration is introduced.

Implementation is incremental: first remove numerical strategy dependencies, then isolate forecast scores and the training empirical baseline, followed by a prediction-only report/diagnostic layer and separate contained lifecycle/CLI. Reuse native admission, bounded fitting, immutable forecast/target evidence and replay rather than making a new data or holdout platform. [Prediction implementation status](../DISTRIBUTION_PREDICTION/README.md) distinguishes delivered interfaces from remaining wiring. No private/provider study or protected-label access is authorized by this amendment. STR-THRU utility, if later requested, consumes frozen forecasts as a separate experiment; profitability is not required to learn whether the forecasts are useful.

## Original downstream recommendation (not the current first experiment)

Run this as a **conditional distribution experiment**, not as a replacement of a supposedly deterministic trading system. Start by changing only the forecast distribution, keeping the causal features, trade definition, payoff bridge and evaluation policy fixed. Test whether the new distribution improves calibrated probabilities and realized net results; do not assume it will.

The most important limitation: **where the payoff bridge is linear and its clip is inactive, changing distribution shape at a fixed mean leaves expected profit unchanged.** The existing `max(0, …)` clip is nonlinear when active, so shape can then affect EV. Shape can also improve profit probabilities and risk estimates. Attribute any economic gain to a changed mean, active clipping, another nonlinear payoff or a different risk-constrained selection rule, not merely to having a histogram.

This review delivers the design only. The first implementation slice is a source-free distribution/metric/payoff contract with synthetic tests. Do not train, open private data, consume holdouts, change legacy code or wire production scoring yet. Subsequent expensive runs remain the user's local runs after the required implementation merges and explicit run authorization.

## What the repository actually does

Source links pin the inspected contracts at `6aeae21e55920dd5771b2b7b5a94ddab6b033f69`; the design branch starts from main `699c25d93ada8bdecbe797f922388d426420f814`. The intervening run-up EMA warm-up fix does not change this proposal's target, payoff, holdout or activation contracts.

- **SIZE predicts magnitude.** Its incumbent target is `abs_move`; the native research target is `A = abs(R)`, where `R = 100 × (post_close / pre_close − 1)`, in percentage points, using the native split-neutral, non-dividend-adjusted price basis. This is a simple return, not a log return. Native source anchors depend on BMO/AMC and actual available bars. [Target and clocks][target], [move calculation][moves]
- **STR-THRU buys one ATM call and one put at the same strike**, with the earliest post-event expiry. Default entry is the last pre-print close (`e+0`); default exit is the first post-print close (`x+1`). The same contracts are pinned through the exit. The strategy normally sells options rather than waiting for expiration. [Structure and replay pricing][pricing]
- **The current model layer is already probabilistic.** It adds held-out SIZE residual draws, optionally conditioned on prediction decile, to the point estimate; then combines those with a causal fitted exit-value line and its residuals to produce `win_model` and `exp_pnl_model`. [Native payoff][payoff]
- **The actual gate is another model.** `gate_midfill_str_thru_forecast_analog` predicts realized per-trade return at mid fills. It includes the point forecast, p10/p90/SD, forecast edge and analog statistics. `gate_score` is not itself a probability of profit. EXP-112 studied prediction-conditioned residual distributions; EXP-173 studied surface-conditioned residual mean/scale. These are essential comparators, not new discoveries. [Gate][gate], [forecast features][gatefeatures], [EXP-112][exp112], [EXP-173][exp173]
- **Execution timing is not solved by forecasting.** The registry still describes the historical entry-close decision contract. The native SIZE horizons are 5/3/1 complete sessions before the pre-print anchor; horizon 1 is one session before the entry close, not the entry close itself. A source-session ceiling alone does not establish when a vendor published the data. [Registry][registry], [target and clocks][target]
- **Costs are not all in the replay return.** Native `structure_return` is exit value minus entry cost, divided by entry cost. Fill alpha models bid/ask execution; that function does not subtract a commission/fee schedule. The new experiment needs a separately frozen all-in net-cost overlay. [Pricing][pricing]

## Exact research question and target

**Question:** On the same causally available inputs and candidate events, does a learned conditional distribution of earnings-move magnitude improve (a) distribution/profitability calibration and (b) realized net STR-THRU performance relative to a point model with the existing residual-distribution approach?

Default first target: **absolute magnitude `A`**, retaining signed `R` for audits and later extensions. This matches SIZE and the current STR-THRU payoff bridge. A signed-return model is a separate experiment: direction matters for off-ATM contracts, asymmetric option repricing and entry drift, but doubling the state space is unnecessary for the first isolated test. Do not assume equiprobable signs and call that a learned signed distribution.

Use the native 5/3/1 row envelope, grouping all horizons of each event. **Horizon 1 is the prespecified primary decision horizon; 3 and 5 are secondary robustness results.** Do not add/remove horizon rows to bypass existing completeness contracts. For the actionable-economic arm, propose decision at `D−1`, entry at `D0`, exit at `D+1`, where `D0` is the last pre-print close. Every baseline must use that same decision cutoff. Keep entry-close replay only as a separately labelled historical diagnostic, not a tradability claim.

Pin contract selection at decision time for the early-decision variant, then price those same contracts at actual entry and exit. Record decision quote, entry fill and exit fill separately. If an entry-time contract re-selection rule is desired instead, it must be preregistered as a different executable policy; it cannot be selected retrospectively from the most profitable chain.

## Why probability and expected profit differ

For a frozen trade and information set `X`, calculate separately:

- `P_profit = P(net_profit > 0 | X)`.
- `EV = E[net_profit | X]`, in dollars per declared position and as a return on capital.
- Loss probability at prespecified severity levels, downside quantiles and expected shortfall; win probability alone is not a ranking objective.

**Illustration only, not measured results:** consider an ATM straddle on a $100 stock, with total premium plus costs of $4. Distribution A always moves 3% in magnitude. Distribution B moves 0% half the time and 6% half the time. Both predict an average absolute move of 3%. At expiry, net profit is `|move in dollars| − 4`: both have EV **−$1**, but A has a 0% profit probability and B a 50% profit probability. Choosing B because it wins more often does not turn its negative EV positive.

For a **toy pre-expiry** example, suppose the same fixed remaining-volatility assumption gives exit straddle marks of $1.60, $3.12 and $6.00 at 0%, 3% and 6% moves, and total entry premium/cost is $3.50. A has EV about **−$0.38**; B about **+$0.30**, with profit probabilities 0% and 50%. The difference comes from the nonlinear exit mark, not magic in a histogram. These rounded hypothetical marks are explanatory, not a fitted pricing model or evidence that the strategy is profitable.

### What the first payoff bridge can identify

Freeze the **same fold-specific, causally trained existing bridge** for every forecast arm: `V ≈ max(0, (intercept + slope × A) × spot) + payoff_residual × spot`. Reuse it; do not jointly improve its coefficients, residual pool, IV inputs or gate during the forecast comparison. “Fixed” means identical within each comparison fold, not a single fit on all years.

Outside the clipping region, EV depends on `E[A]`, the residual mean and entry cost, not higher moments. Add an analytical invariance test and report whether any apparent EV gain came from a different mean, clipping, or Monte Carlo noise. Use deterministic integration where feasible or common random numbers with a reported numerical error bound. With exactly mean-matched arms, inactive clipping and the same EV-only selector, economic delta is a predeclared zero/null control, not an economic-superiority test; do not manufacture power by changing the selector after results.

The existing bridge's residual is added after the nonnegative exit-value clip and can produce physically impossible negative exit values/returns below −100% before fees. Preserve an exact historical-parity baseline and quantify this diagnostic; do not silently fix only the challenger. Any support repair is a separately named, shared bridge version applied to all arms.

For D−1 decisions, neither the actual D0 spot nor paid premium is known yet. The minimal bridge produces a **decision-quote-conditioned proxy**, not a fully identified net-profit forecast. Evaluate it against actual pinned-contract replay and report entry-price drift and P&L calibration. Realized D0 entry prices may score outcomes, but must never become forecast-time inputs. A production-quality bridge would also need the conditional distribution of entry spot/premium and contract moneyness.

### When a richer target is necessary

Fixed unhedged D0-to-D+1 exits do not intrinsically need a full intraday price path: joint terminal state, held contracts and fills can suffice. But terminal underlying return alone does not determine pre-expiry option value. A richer bridge needs signed move/exit moneyness, residual IV surface or executable exit option marks, time remaining, entry-price uncertainty and their dependence. Independent move and IV marginals are not a valid substitute for the joint distribution.

If stops, targets, dynamic hedging or earlier exits enter the policy, terminal returns become insufficient; then model or resample joint price/IV/quote paths, specify observation frequency, gaps and ambiguous barrier ordering. The repo has separate target/stop exit machinery, but that is **not** part of default STR-THRU and is out of the first slice. A direct conditional distribution of realized net trade P&L is another useful later comparator, not a substitute for answering whether the move distribution helped.

## Minimal distribution representation and candidate

Propose an additive versioned distribution recipe, not a mutation of `earnings_magnitude.v1`'s conditional-mean semantics.

- Output a normalized conditional CDF/histogram with explicit units, bin edges, probabilities, support/tail policy, feature/source/fold identities and calibration identity. Expose mean, quantiles and threshold probabilities as derived values.
- Primary proposed grid: `[0, 0.25), [0.25, 0.50), …, [19.75, 20), [20, +infinity)` percentage points. Here 0.25 means a **quarter percentage point**, not 25% and not a relative 0.25% change in a prior estimate. Twenty is a proposed core-grid boundary, not an outcome cap.
- No clipping, dropping or winsorizing evaluation targets above the grid. The overflow probability must remain visible. A tail probability alone is insufficient for EV/CRPS/expected shortfall: freeze a train-only tail-magnitude rule, recommended pooled empirical excesses initially, with explicit tail-support counts and sensitivity, including stress beyond the largest training move. With no admissible tail sample, report bin-probability scores only and refuse full-CDF metrics/economics rather than invent an overflow midpoint.
- Within a finite bin, preregister interpolation (initially uniform) and report endpoint-based sensitivity where a break-even boundary crosses it. A 0.25-point display is not evidence of 0.25-point predictive precision. Preserve actual zero observations in the first bin; consider a separate zero atom only if preregistered.
- First fitted challenger: existing-library **HistGradientBoostingClassifier** with log-loss probabilities over the bins, paired with a same-feature HGB point regressor. Start with one fixed histogram configuration and the current native HGB settings, no new feature/hyperparameter search, random early stopping, class balancing or neural framework. Main pins sklearn 1.9.0; verify that exact environment/API before implementation rather than upgrading to match current online docs. Multiclass boosting builds one tree per class per iteration; count that cost, not just one nominal “fit.” [Official sklearn API][sklearn]
- Predeclare sparse/absent-class handling, common probability smoothing and any calibration using training/inner-OOF data only. Inspect training support before admitting the fixed recipe; inadequate support is a documented failure or a new preregistered coarser experiment, not permission to change bins after outer results.

Alternatives worth a later bounded comparison: coarser/adaptive bins, monotone ordinal-CDF models, a quantile grid, and a parametric location-scale distribution. Quantiles can be cheaper and respect ordering, but still need crossing/interpolation/tail rules before becoming a full CDF. A 0.25-point histogram is a useful output format, not necessarily the best training representation.

## Controls that isolate the effect

| Arm or control | Purpose |
| --- | --- |
| Empirical training CDF, plus no-trade and ungated books | Sanity and coverage controls; no sophisticated forecast should win by exclusions alone. |
| Causal incumbent SIZE plus its prediction-decile OOF residual distribution | Existing uncertainty baseline; never use a full-refit artifact or future residual pool as historical evidence. |
| Same-feature HGB point prediction plus causal OOF residual distribution | Learner-family/feature-matched baseline, not capacity-matched: multiclass boosting builds more trees. |
| Learned HGB distribution | Full distribution challenger. |
| Same fitted histogram's mean plus causal OOF residual envelope | Primary output-shape control: use the identical fitted histogram, features and folds, but retain only its mean plus earlier OOF mean-prediction residuals. Apply shared nonnegative support and rescale the envelope to that histogram's mean; refuse undefined zero-mean cases. |
| Cross-model mean-matched shape ablation | Rescale the challenger's nonnegative support to the HGB point baseline distribution's mean, without evaluation labels or refitting. This holds the forecast mean equal, not the independently trained models' learning capacity. |
| Causal incumbent STR-THRU return gate | Economic benchmark. Its broader features are acknowledged; it is not the architecture-isolation comparison. |

Keep rows, clocks, feature dictionary, preprocessing, shared learner settings, quote eligibility, payoff bridge, costs, candidate universe, position budget and selection-rule family fixed across matched arms. Output loss and effective tree capacity change between regression and multiclass fitting; that comparison cannot isolate the output head alone. The same-histogram-mean control isolates using distribution shape from that fitted learner, not whether distributional training improved its representation. If the causal incumbent's features are not admitted yet, report it as unavailable rather than substitute a different model under its name.

The primary matched residual baseline uses the same nonnegative magnitude support convention as the challenger: negative point-plus-residual draws become an explicit zero atom, with its actual resulting mean recorded. Keep the exact incumbent's unclipped distribution as a separately labelled parity diagnostic. Otherwise a seemingly learned-distribution gain could merely be a support correction. Do not clip the signed target itself.

Two evaluation stages prevent gate confounding:

1. **Distribution and bridge diagnostics:** preserve the incumbent gate's selections exactly; evaluate forecasts on all eligible opportunities and that fixed subset. Trade P&L on the frozen book must be identical across arms. This stage can establish better probabilities, not increased trading profit.
2. **Prespecified research-only selection:** apply the same EV-based selector and capital constraints to each arm. Thresholds use only inner-OOF predictions; do not take each outer year's realized top quintile. Use the existing top-20% training-quantile discipline as an initial coverage target, report realized coverage, and require estimated net EV above zero. A separate fixed risk-aware selector can test shape value; its loss constraint must be frozen before evaluation. Report matched-capital/coverage diagnostics, not just average return on different-sized books.

Do not retrain the existing gate on new distribution features in this first experiment. That is a later factorial experiment: forecast shape × gate/features × threshold policy. EXP-187 documents exactly why changing these together makes attribution unreliable. [EXP-187][exp187]

## Causality and holdout protocol

Use expanding annual outer folds with nested earlier-year selection/calibration. The proposed SIZE development years 2022–2024 and earlier inner years are a starting schedule only: metadata feasibility must establish enough admissible history. Do not reuse previously examined years as fresh confirmation or replace missing years with protected holdouts.

All preprocessing, bins if adaptive, residual pools, tail rules, probability calibration, hyperparameters, stopping decisions and selector thresholds are fitted/sealed inside outer training. OOF residuals come from genuinely earlier fits. Purge examples whose measured label/position intervals overlap evaluation decision boundaries; labels and closed trades must be available **strictly before** the fit/calibration cutoff. Payoff fits must not use the year being evaluated. For new postprocessing use chronological inner splits, not sklearn’s default non-temporal calibration folds.

Group all forecasts, quote versions, fills and duplicate/sibling records of one canonical event. Issuers can recur through time in the main temporal estimand, as existing native folds allow; do not pretend repeated issuer events are independent. Report a separate ticker-held-out sensitivity if useful, excluding those tickers’ learned outcome contributions during fitting while retaining only information genuinely available at inference. This is a robustness analysis, not a replacement for forward validation.

Preserve the repository's existing **3% hash-selected canonical-event holdout** and **six-calendar-month rolling holdout**, with the holdout month fixed in run identity. Exclude the union from development examples, fitting and calibration; protected labels/explicit event outcomes must not contribute to learned historical aggregates. Keep dataset-relative metadata ordinals and qualified causal raw-price/IV prefixes; do not remove individual bars or alter rolling formulas. Ambiguous membership refuses. The SIZE grouped policy adds an inclusive **±30-calendar-day same-ticker embargo** around random-held events, excluding a whole development event when any label interval, event date or forecast decision/source ceiling overlaps. Authenticate actual label anchors before using these intervals. Never invent them from a receipt hash. [Holdout definitions][holdouts], [SIZE grouped rules][size2]

Random holdouts are historical robustness, not temporal independence; long-memory features and market-wide effects survive this embargo. The rolling chronological holdout is primary forward confirmation. Report the two separately and their overlap, never a pooled average.

Final access remains a separate, authorized frozen single-winner process. Current final metadata requires one registered arm, and the present evaluator only reports basic held-out trade metrics; it is **not already a distribution-evaluation engine**. A fixed-winner-versus-locked-reference confirmation/reporting recipe requires explicit reviewed support, not a loop over contenders via final-read APIs. No tuning after opening either holdout. [Final contract][finalcontract], [final evaluator][finaleval]

## Measurements and decision rule

**Forecast primary:** paired event-weighted CRPS against the same-histogram-mean residual control. The family-matched point model and incumbent are secondary comparisons. Use the same distribution reconstruction/tail rules, with ranked probability score on common bin thresholds as an auditable companion. Secondary: common-bin log loss, Brier/log loss for `P_profit`, 50/80/90/95% interval coverage **and width**, randomized PIT/rank diagnostics for discrete distributions, mean RMSE, median MAE, and calibration by year/horizon/quote coverage/forecast scale.

**Tail checks:** calibration and observed counts at fixed magnitude exceedances (proposed 5/10/20 points), loss thresholds and expected shortfall; report uncertainty and avoid strong tail claims when counts are small. Include tail contribution to predicted EV and finite-bin/tail-sensitivity ranges. Do not use only interval coverage, overall accuracy or win rate as success.

**Economic primary:** paired net P&L per common opportunity under the fixed executable cash budget, with unselected opportunities contributing zero. Also report return on deployed capital and capital-days, net mean/median trade return, profit factor, annual equity/Sharpe/Sortino with cash idle periods retained, maximum drawdown, expected shortfall, losing streaks, and performance by regime/year/ticker. Preserve chronological simultaneous events and cash competition.

Use actual pinned entry/exit bid/ask replay for realized outcomes, a frozen commission and exchange-fee schedule, correct contract multipliers and counts, and adverse slippage assumptions. Avoid charging the quoted spread twice if the alpha fill already includes it. Show at least alpha 0/0.25/0.5 sensitivity; alpha 1 is an optimistic bound, not a credible execution default. Freeze liquidity/quote-age rules, per-ticker and concurrent-position caps, and any volume/open-interest capacity rule using information available at decision time. If depth/volume is absent, state that executable capacity is not established.

Never use future exit-quote availability to decide whether to enter. Report missing/unsettled outcomes and a preregistered completeness floor; an unavailable outcome is not a zero-return no-trade. Show the full eligible opportunity denominator, per-arm abstentions and the common scoreable subset so selection bias cannot masquerade as profitability.

Uncertainty: paired block resampling by calendar week/month to preserve earnings clusters and portfolio order, plus ticker-cluster sensitivity. Keep every event's horizons/fills together. Use blocks of economic time rather than treating all trade rows as iid; disclose instability from few years or sparse trades. Freeze block method/length, seeds and resample count before evaluation. Report confidence intervals for differences, not just separate arm intervals.

Preregister one forecast primary contrast and one economic primary contrast (full histogram versus its same-learner mean/residual control under the same selector). Record every attempted recipe, binning, tail, horizon and selector variant, including failures. Inner selection handles any later model search; adjust secondary family claims, for example with Holm, and do not present the best outer result as untouched evidence. EXP-112/173 and prior related experiments belong in the hypothesis history.

Advance only if distribution scoring improves without material tail-calibration degradation **and** the chosen economic policy improves realized after-cost results at matched capital with acceptable downside, across plausible fill/tail assumptions. Proposed statistical bar: a positive paired 95% lower bound for the economic delta; minimum economically useful uplift and risk limits must be set before registration. Better CRPS alone is a useful scientific result, not a profitability or deployment claim. A null result under the linear bridge is informative.

## Platform reuse and implementation slices

Reuse the native frozen snapshot, selected-event manifest, stored SIZE history/features, annual fold planning, strict target availability, search receipts, trial journals and bounded execution/replay boundaries. Do not build a parallel Parquet reader, new source-completeness project, or per-candidate feature materializer. Existing public research APIs own access; add versioned adapters rather than import legacy runtime code or mutate existing source closures.

Population means **eligible confirmed events within the selected frozen dataset**, not all provider or real-world events. Preserve complete manifest/partition inventory, dataset-relative history ordinals, pre-history-start metadata, conflict refusals and coverage diagnostics. #1403 removed the external-population-completeness claim, not clock qualification or holdout protection. [Merged population amendment][population]

Readiness at investigation time: #1375 foundations remained open at `cabfd4e8`; #1382 native identity/session/scanner remained open at `67d47c89`. Their synthetic tests/gates are not evidence of real-data capacity, causally authenticated label anchors or a runnable complete stage-two experiment. C0 availability/forecast/label qualification and dependent numerical integration remain prerequisites. HMO features are not assumed available; a first fit may use admitted H-only features in both matched arms, with a correspondingly limited conclusion. [SIZE status and boundaries][size2]

At the final status read, train #1390 was `d8ecc917` with both #1375 and #1396; earlier single-car checks are not evidence for this head. This plan neither claims current-head approval nor changes train ownership. [Train][train]

#1396 at `6cff2381` is separately owned native STR-THRU causal gate-forecast wiring. Reuse its exact-fold principle—earlier event/decision month; missing/incompatible fold refuses; no champion fallback—after its own merge/review lifecycle. Do not take it over or sneak distribution output into its parity scope. [#1396][gatepr]

**Frozen-gate baseline qualification:** open P1 #1423 reported that native `entry_cost_pct` used decision-session close while legacy gate training used the priced chain's spot. PR #1470 changes native scoring to use the staged ORATS chain spot, and PR #1873 uses that spot to select listed STR-THRU/STR-RUNUP strikes when present, falling back to calendar close only when absent. The owner reported 67 numeric mismatches plus 22 separate null/quote-availability differences; those private-artifact counts were not independently reproduced here. Before making gate-relative claims, require exact feature-source parity and account for quote-availability differences, or explicitly label the benchmark unqualified. This is a dependency to its existing owner, not authorization to inspect private artifacts. [#1423][spotissue]

### Platform ownership and reuse

This is not a platform rewrite or a new generic distribution framework. Reuse the existing snapshot/reader, fold, resource, trial, replay and holdout owners; add only minimal versioned integration where the new output contract requires it. No percentage split is estimated before those adapters are implemented.

- **A:** `engine/v2/research/magnitude_distribution.py` (#1430) and companion numerical utilities are reusable **magnitude-family research** code, not universal distributions: absolute moves and the fixed grid are deliberate family/recipe choices. STR-THRU payoff assumptions and comparison controls remain research-specific, not production scoring changes.
- **B:** reuse native `engine/v2/research/magnitude_*` readers/folds/stored panels and `experiments/magnitude_search_*` admission/resource/trial machinery; add the smallest distribution identity/admission adapters, not a replacement store, reader or scheduler.
- **C:** add the distribution candidate and causal calibration alongside existing `experiments/magnitude_candidate_fit.py`, `magnitude_candidate_recipe.py` and `magnitude_fitted.py`. The histogram-versus-point recipe and residual controls serve this experiment; reuse later does not justify global model infrastructure now.
- **D:** reuse bounded execution and native replay/cash-book/report boundaries; the distribution comparison runner, after-cost policy and additional report columns are experiment-specific. Existing holdout permissions and production entrypoints do not expand.

### Four-bundle implementation amendment, 26 October 2026

The approved packaging is **four implementation PRs**, not ten serial review units. Original slice IDs below identify retained substeps, not separate PRs. This changes review packaging and bounded size allowances only: scientific controls, tests, source qualification and run authority are unchanged. The design PR itself is not one of the four.

Estimates are **added source / test / documentation lines**, including owning README/ARCHITECTURE/inventory updates. User-approved source ceilings exclude tests; documentation has a separate ceiling. Bundle A uses the actual initial #1430 format slice (143 source / 227 tests / 20 docs) plus the remaining estimates. These are not delivery promises; stop and re-scope before exceeding an approved ceiling. Existing SIZE/clock/spot dependencies remain separately owned prerequisites, not hidden work in these bounds.

| Bundle and dependency | Retained deliverable and acceptance | Estimated source / tests / docs | Approved source / docs ceiling |
| --- | --- | --- | --- |
| A / design; expands #1430 (1a–1c) | Source-free bin/tail representation and refusal oracles; CRPS/RPS/log/Brier and reconstruction sensitivity; frozen payoff adapter and mean-matched controls, including linear invariance, clipping and nonlinear synthetic oracles. | 443–603 / 667–867 / 65–105 | 650 / 120 |
| B / stable A format interface and qualified native sources (2a–2b) | Versioned recipe/fold/feature admission, protected/future poisoning, identity/all-horizon tests; resource/fit-count envelope and trial identity, exact limits and failure/replay tests. | 340–520 / 460–680 / 65–115 | 550 / 120 |
| C / B (3a–3b) | Fixed histogram and matched point fitting, absent classes, deterministic fits/receipts; causal OOF residual/calibration controls, strict inner-only provenance and zero-mean refusal. | 360–560 / 500–740 / 70–120 | 600 / 120 |
| D / A and C, plus frozen economics (3c, 4a–4b) | Sealed bounded local runner/report, complete attempts/no default execution; net-cost overlay/common selector, units/multipliers/fees/cutoffs; native cash-book/report integration, paired capital/coverage and missing-outcome diagnostics. | 500–780 / 720–1080 / 105–180 | 800 / 180 |

A's scoring/payoff substeps and B's admission/resource work may proceed partly in parallel after the numerical interface stabilizes; B still waits for qualified SIZE/native inputs. C follows B; D joins A and C. This reduces review rounds, not causal prerequisites or testing. D contains the first bounded diagnostic runner and the first complete after-cost portfolio comparison; no runnable experiment is claimed before it and the independent data prerequisites are ready.

The following remain conditional **design-only** proposals, outside the four implementation PRs; they are not prerequisites for the first experiment and receive no blanket implementation size exemption:

| Later design and dependency | Deliverable | Added code / tests / docs |
| --- | --- | --- |
| 5a / useful development evidence only | Separate nonlinear/joint or direct-P&L follow-on design and its own implementation estimates. | 0 / 0 / 120–180 |
| 5b / frozen winner only | Separate final-confirmation design, fixed reference and protected-read acceptance/implementation estimates. | 0 / 0 / 120–180 |
| 5c / successful authorized confirmation only | Separate deployment design, including new release contracts and implementation estimates. | 0 / 0 / 120–180 |

Bundles A–D add research contracts/code only: no provider, private execution, production gate, strategy/exit optimization or automatic promotion. Metadata admission is not payload authority; no bundle may hide a feature rebuild. Designs 5a–5c budget designs only, not unspecified future implementations. Real development remains separately authorized user-run local work; no final holdout, model release or deployment is authorized here.

Only these four bundles receive the explicit ceilings above in place of the ordinary approximate 400-added-source/120-added-doc-line limits. Tests stay under `tests/v2/`; their exclusion from the source cap does not waive coverage. Standard sklearn/NumPy/stdlib cover the initial modeling and numerical work; no new framework is needed. Count classifier trees, calibration/residual fits, storage, memory and wall time, and retain no-automatic-retry semantics and bounded progress logging.

Synthetic acceptance includes exact bin boundaries/zero/tails, normalization/NaN/negative mass refusal, absent classes, deterministic replay, all-tail and no-tail cases, proper score oracles, identical-mean linear-EV invariance, nonlinear toy distinction, cost/multiplier accounting, no future fit/calibration, protected-event poisoning, all-horizon grouping, source identity mismatch, and no legacy runtime changes. Before any real run, require an explicit resource/coverage preflight and a sealed run budget; no expensive operation was performed for this plan.

## Decisions required before execution

1. **Recommended scope:** absolute magnitude first, fixed unhedged D0-to-D+1 exit, primary D−1 decision. Confirm this before registration, especially if the intended experiment is instead signed moves or the historical entry-close decision.
2. **Economics:** choose the actual commission/fee schedule, representative execution assumption, capital/concentration limits and minimum worthwhile net uplift. The plan can report a cost grid meanwhile, but cannot invent Yuri's risk tolerance or broker costs.
3. **Authority:** design publication and the four bounded research-code bundles are authorized; local private-data/training runs and any final holdout opening remain separate decisions. This review does not activate a recipe, allocate an experiment number or modify production behavior.

This document adds no runtime interface, dependency or data-access authority. Existing README/ARCHITECTURE runtime contracts remain unchanged; the indexing links identify a proposed follow-on only. No original private dataset, provider/training job, model release or final holdout is part of this review. Design acceptance is not preregistration or scientific approval.

## Source links

[target]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/v2/research/magnitude_inputs.py
[moves]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/v2/data/computed_moves.py#L75-L95
[pricing]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/v2/research/_pricing.py#L1015-L1167
[payoff]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/v2/scoring/native_payoff.py#L255-L351
[gate]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/models/training/gate.py#L1-L80
[gatefeatures]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/v2/scoring/native_gate_features.py
[registry]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/v2/registry/strategies.py
[exp112]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/experiments/EXP-112_size_v2_0_step_3_prediction_intervals_th/spec.yaml
[exp173]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/experiments/EXP-173_size_surface_residual_distribution/spec.yaml
[exp187]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/experiments/EXP-187_str_thru_gate_feature_threshold_decomposition/spec.yaml
[holdouts]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/v2/foundation/experiment_holdouts.py
[finalcontract]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/v2/foundation/final_holdout.py#L65-L89
[finaleval]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/engine/v2/research/_final_evaluator.py
[size2]: https://github.com/yshewchuk/investment-validation/blob/6aeae21e55920dd5771b2b7b5a94ddab6b033f69/experiments/SUPER_SIZE/STAGE2_DESIGN.md
[population]: https://github.com/yshewchuk/investment-validation/pull/1403
[gatepr]: https://github.com/yshewchuk/investment-validation/pull/1396
[train]: https://github.com/yshewchuk/investment-validation/pull/1390
[spotissue]: https://github.com/yshewchuk/investment-validation/issues/1423
[sklearn]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
