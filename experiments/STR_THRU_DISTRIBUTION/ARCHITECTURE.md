# Distribution fitting and causal controls

Research-only Bundle C extends the [experiment design](DESIGN.md) and the [B admission contract](../ARCHITECTURE.md#distribution-native-admission). There is no CLI, production path, provider read, final-holdout authority or bounded experiment runner; Bundle D owns execution containment and activation.

## Public boundary and dependencies

| API | Caller, input and output |
| --- | --- |
| `distribution_candidate_fit.fit_candidate(recipe, train_features, train_labels, prediction_features, *, columns)` | Numeric adapter/tests supply exact recipe, ordered columns and matrices; returns raw histogram rows, one shared training-only tail, matched points and diagnostics. |
| `DistributionTrials.read_fit_context(role, outer_year, inner_year=None)` | C obtains freshly verified plan, native fold, admission and projected stored training/features; no second outcome read or eligibility policy. |
| `DistributionFits(trials).fit(role, outer_year, horizon, inner_year=None)` | Explicit library caller fits one admitted horizon pair after a unique claim; returns its immutable completion receipt, forecast rows and shared tail. |
| `DistributionFits.read(...)` | Replay verifies the same admission, claim, matrices and stored result without refitting; not a metadata-only read. |
| `distribution_oof.residual_pool(fits, role, outer_year, horizon, inner_year=None)` | C/tests obtain exact earlier-inner residuals joined to destination training labels, with identity/provenance and explicit unavailable states. |
| `distribution_oof.control_for_prediction(pool, histogram_document, point_prediction)` | Numerical consumer applies the declared shared residual pool to one forecast; returns mean-anchored histogram and nonnegative point controls, not calibrated probabilities. |

Dependencies are existing public native admission, fit evidence, immutable journal and CAS shard primitives, foundation hashing, fixed HGB model APIs, magnitude distribution arithmetic/payoff helpers and bounded stderr progress. NumPy/sklearn/threadpoolctl provide numerical work; no new library, store, transport, scheduler or generic estimator framework is introduced. Current consumers are synthetic tests; D must explicitly connect a bounded worker.

## Numeric and fit contract

Both no-fit guards precede numeric inspection and the explicit fitted operation. Training-all-null columns become zero in both matrices; partial NaNs retain native routing. Nonnegative training labels alone determine class support and empirical excesses over 20 percentage points, including zero at 20. Two observed bins are required; absent classes keep zero mass, with no smoothing or weighting. Installed libraries may inspect runtime state (for example threadpoolctl reads `/proc/self/maps`); numerical fitting is not described as I/O-free.
The classifier's classes, iterations and tree multiplicity must match the recipe/support. Its counts are observed; point iterations remain explicitly declared because the existing public point fitter exposes predictions only. Missing tail samples remain unresolved even at zero overflow mass; no synthetic tail, CDF or mean is invented.
Ordered raw training/prediction matrix hashes preserve nulls and event axes before fitting. The claim binds the exact B trial, plan, admission and matrix evidence; installed C sources/environment are included in the plan, so old seals cannot silently acquire new code. A unique fit claim precedes estimators; an interrupted claim cannot restart. Completion means both numerical arms returned, not that a supervised process or complete experiment succeeded.
Forecast rows and the shared tail use existing verified CAS sections; the immutable completion commits their descriptors and diagnostics. Forecast rows restore native evaluation order on read. One tail is stored per fit, not copied into every row; no prediction-by-residual Cartesian product is persisted. A repeated `fit` with an existing completion performs verified replay, not another estimator attempt; its no-fit guards still run first, while `read` is available under those guards. Replay revalidates admitted input identity and stored output contracts, never opens evaluation labels or refits.
Inputs/matrices and projected result rows are materialized in memory. Existing shard/journal limits apply to stored evidence, not an end-to-end memory bound; estimator allocations and whole-job feasibility remain D/preflight responsibilities. Numerical pools use one thread. Stage start/end/elapsed/RSS and bounded progress use existing research logging; D must supply long-fit heartbeats and wall/CPU/memory enforcement.

## Causal residual contract

The versioned policy is pooled equal-weight earlier-OOF residuals per horizon, uniform finite-bin reconstruction, zero-clipped point support and the existing mean-anchored histogram control. `calibration_identity` remains `uncalibrated`: these empirical comparison controls do not calibrate classifier probabilities and perform no extra estimator fits. A learned probability calibrator requires an explicit later recipe and budget.
An outer destination requires its declared inner fits; an inner destination requires only its strictly earlier inner prefix in the same outer group. The first inner has no control pool. Missing or failed expected fits cannot silently disappear, and outer/self/future fits cannot become donors. All horizons retain their B identities rather than changing event completeness.
Only donor evaluation events present in the destination's centrally admitted training set contribute labels. Exclusions and coverage are explicit; no new label reader, threshold, holdout filter or residual-quality selection is added. Native admission owns protected membership and strict label availability. Each historical histogram mean uses that fit's own training-only tail, never the destination's later tail.
Pool provenance binds destination inputs/cutoff, plan/recipe/context, horizon and ordered donor result identities plus contributing keys/exclusions. Unresolved donor tails make the histogram pool unavailable without silently dropping rows; an otherwise usable point pool remains separate. Empty pools are unavailable, not fallback forecasts. The primary mean-matched control preserves zero atoms and refuses undefined zero/all-zero means; the point control records its actual post-clipping mean.

## Failure, storage and replay

| Condition | Outcome |
| --- | --- |
| No-fit guard active | Original guard exception before fitting/admission work. |
| Invalid recipe/order/numeric input or unsupported fit selector/horizon | Typed invalid-spec refusal; omitted arguments retain Python `TypeError`. OOF provenance/selector mismatch instead uses identity conflict. |
| Degenerate support or malformed estimator output | Typed candidate failure; no partial successful fit receipt. |
| Changed input/implementation/result identity, missing completion, duplicate unfinished fit or incompatible donor | Identity conflict; no automatic repair, retry or silently reduced donor set. Histogram validation retains its native `CONTRACT_MISMATCH` data error. |
| No earlier/eligible residual observations or unresolved tail | Explicit unavailable control state; never a fabricated distribution. |
| Undefined mean-matching arithmetic | Existing magnitude arithmetic refusal, not a zero-valued fallback. |
| Resource, storage or unexpected programming failure | Native failure propagates; a committed claim remains unfinished. Published unreferenced shards can remain after interruption. |

Existing append transactions own unique journal associations, not whole-fit rollback. Completed in-memory estimators cannot be rolled back after a later failure; unfinished claims reserve work without asserting how many arms completed. No claim here establishes real-data feasibility, tradability, calibrated profitability or scientific success. Synthetic acceptance covers exact identity/replay, all horizons, tiny fixed fits, protected/future source boundaries, chronological poisoning, unavailable pools/tails and mean/support oracles.
