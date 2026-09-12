# Live Intraday Scoring Guide

**Version:** 0.1 · **Date:** 2026-09-12 · **Status:** proposed build plan
**Depends on:** Phase 1 scorer, Phase 3 dashboard, Tier 4 feature models, and
`guides/prompt_execution_clock.md`

## 1. Decision and evidence

The research book, the final daily data tiers, and the live scoring surface are
three different things. This guide adds the third. It does not change the
meaning or storage contract of Tiers 1 through 4.

The execution-clock experiments establish that a gate is attached to the
information snapshot on which it was trained. For the current STR-THRU gate:

| scoring and execution book | mean return | Sharpe | CAGR |
|---|---:|---:|---:|
| D0-trained model on D0 data and trades | +3.84% | 2.06 | 269% |
| D1-trained model on D1 data and trades | +3.16% | 1.65 | 156% |
| D0-trained model on D1 data and trades | +2.74% | 1.50 | 133% |

The third row is EXP-183. It holds the D0 fold model and threshold fixed,
scores D1 features, and executes D1 trades. It is not a D1 retraining test.
The D1-trained comparison is EXP-182. Both use walk-forward folds.

The evidence supports these rules:

1. Do not expect a model trained at one information clock to transfer to
   another clock.
2. A model registry entry must identify its decision clock and source
   snapshot contract.
3. The current D0 close model is not automatically a D0 pre-close model.
   A real-time D0 candidate must be trained and validated on the inputs that
   existed at its actual decision time.
4. The dashboard should be able to display parallel model variants, but only
   a separately validated variant may produce a trade recommendation.

This is a plan to create a **D0 pre-close** clock. It is the executable
candidate that can eventually recover the value of a same-day decision without
waiting for the official closing files. It is not permission to serve an
end-of-day feature vector before the close.

## 2. Clock contract

Choose one named, fixed decision time before building anything. The initial
candidate is:

```
clock_id: d0_preclose_1545_et
decision timestamp: 15:45:00 America/New_York
entry target: same-session close or a separately recorded executable order rule
event timing: BMO and AMC remain resolved by the existing session-aware calendar
```

The time is an experiment parameter, not a dashboard setting. A later 15:30 or
15:55 model is a new clock, new dataset, and new registry entry.

Every score must carry:

- `clock_id`, requested decision timestamp, and actual source timestamps;
- the latest permitted observation time for each feature;
- a snapshot hash and raw-response hashes;
- model artifact fingerprint and feature-contract version;
- the planned entry and exit dates, plus BMO or AMC classification.

A score is invalid if a feature observation is after the declared decision
timestamp. It must return a visible refusal, not a stale or substituted value.

## 3. Data architecture

### 3.1 Keep Tiers 1 through 4 end-of-day

The existing tiers remain the canonical, immutable end-of-day research store:

```
Tier 1 -> Tier 2 -> Tier 3 -> Tier 4
```

They continue to use the completed-session finality rules in
`prompt_execution_clock.md`. A live fetch must not write a provisional
same-day value into `daily_market`, `option_chains`, the panel, or the
Tier-4 table. It also must not trigger a Tier-3 or Tier-4 rebuild during the
market session.

### 3.2 Add a separate live snapshot store

Introduce a private, ignored live store, for example:

```
data/live_snapshots/
  2026-09-14/
    154500_et/
      manifest.json
      orats_summary.csv
      orats_chain.csv
      feature_frame.parquet
      scores.json
```

The exact path can change, but these properties cannot:

- Snapshots are append-only and timestamped. A retry creates a new snapshot,
  it does not overwrite a recommendation record.
- `manifest.json` records request parameters, response hashes, vendor
  `quoteDate` and `updatedAt`, local receipt time, ticker coverage,
  failure reasons, and the source-schema version.
- The canonical feature frame is stored separately from raw vendor responses.
  Replaying a score uses this frame and makes no network call.
- Live snapshots, vendor data, scores, and recommendation records go only to
  the private mirror. They do not enter the public repository.

The dashboard renderer receives already-computed score rows. Browser code never
fetches ORATS or constructs a feature.

### 3.3 ORATS source contract

No single chain response contains all model inputs. The live collector combines
these endpoints per ticker:

| need | ORATS source | use |
|---|---|---|
| IV, ex-earnings IV, implied move, realized vol, skew, contango, forward IV and forward ex-earnings IV | `/datav2/live/one-minute/summaries` | current market-state vector |
| executable quotes, option IV, greeks, DTE, strike and current option cost | `/datav2/live/one-minute/strikes/chain` | structure selection and `entry_cost_pct` |
| optional surface detail by expiry | `/datav2/live/one-minute/monies/implied` | ATM IV, term surface, earnings effect |
| optional forecast surface detail | `/datav2/live/monies/forecast` | forecast-vol surface diagnostics |

The one-minute summaries schema includes `iv10d`, `iv30d`,
`exErnIv10d`, `exErnIv30d`, `impliedMove`, `rVol30`, `skewing`,
`contango`, `fwd90_30`, and `fexErn90_30`. The collector must map the
vendor names to the local feature names once in code, with a schema test.

ORATS documents live data at less than ten seconds of market delay. Its
`stockPrice` can be calculated from put-call parity, so the collector must
retain both `stockPrice` and `spotPrice`, their timestamps, and the field
selected for each model feature. It must reject an insufficiently fresh
snapshot rather than silently use a prior minute.

Official documentation:

- https://orats.com/docs/live-intraday-api
- https://orats.com/docs/live-data-api

Before implementation, make one authenticated, redacted schema probe during
market hours. It must confirm that the account is entitled to the live
one-minute endpoints and that required fields are populated for a liquid
sample. Do not print credentials or raw licensed data in logs.

## 4. Feature contract for the current STR-THRU gate

The registered STR-THRU champion currently has 50 inputs. They divide into
four provenance classes:

| class | examples | live scoring treatment |
|---|---|---|
| prior-event history | `n_prior`, prior moves, EMAs, streak | carry from the last completed Tier-3 history; no same-day event may enter |
| market state | `im`, `iv10`, `iv30`, `exern_iv10`, `iee`, `skew`, `contango`, `fwd90_30`, `fexern90_30`, `rvol30` | take from the timestamped live summary response |
| rolling changes and calendar state | `im_d1/d5/d10`, IV changes, `days_to_print`, `mcap_log` | calculate from the live value plus only completed-session history; use the documented earnings calendar |
| trade and model features | `entry_cost_pct`, `dte_entry`, `pred_abs_move` bands, `forecast_edge`, analog statistics | derive cost and DTE from the live chain; run feature-model inference and analog matching against frozen, causal state |

`pred_abs_move` and the analog fields are not supplied by ORATS. At live
time they are model outputs. They require a serving adapter that:

1. loads the model artifact valid for the current Tier-4 fold or registry
   version;
2. builds its inputs from prior completed state plus the declared live fields;
3. never reads a same-day EOD tier row; and
4. records the artifact and input-frame hashes in the live manifest.

This is a serving adapter, not a live Tier-4 rebuild. The historical Tier-4
table stays unchanged.

Any registered feature lacking a defined live provenance must cause
`MISSING_LIVE_FEATURE` and suppress that score. Filling missing fields with a
current close, a later quote, or a generic default is forbidden.

## 5. Model and registry changes

Add a clock-qualified model identity. Illustrative fields:

```json
{
  "id": "gate_midfill_str_thru_d0_preclose_1545_v1",
  "strategy": "STR-THRU",
  "role": "gate",
  "clock_id": "d0_preclose_1545_et",
  "feature_contract": "live_str_thru_v1",
  "artifact": "data/models/...",
  "source_contract": {
    "completed_eod_ceiling": "prior_session",
    "live_observation_cutoff": "15:45:00 America/New_York"
  }
}
```

The registry loader must reject a model if its clock or feature contract does
not exactly match the requested score. A close-clock artifact cannot be used
as a fallback for a pre-close request.

Train one candidate per meaningful execution clock, not one model per nominal
date. Initial clocks are:

| clock | status |
|---|---|
| D0 close | existing historical benchmark |
| D1 close | existing historical benchmark |
| D0 pre-close 15:45 ET | proposed, requires intraday dataset and experiment |

The D0 pre-close candidate is evaluated against the D0 close and D1 close
benchmarks on identical eligible events where possible. It must use
walk-forward training, the same fill sensitivity reporting, and a report per
arm. Its shorter intraday-history window must be stated prominently; it cannot
borrow pre-2022 EOD rows and call them 15:45 observations.

## 6. Build sequence

### Step 0: specify and probe

- Freeze the first clock and entry convention in an experiment spec.
- Confirm ORATS entitlement and inspect field completeness for liquid tickers.
- Define the local name, vendor source, unit, timestamp rule, and missing-value
  rule for every registered feature.
- Estimate request volume for the watchlist and use a quota-guarded dry run.

**Exit:** approved feature-source matrix and a redacted schema receipt.

### Step 1: collect immutable shadow snapshots

- Implement a bounded collector for the upcoming earnings universe at 15:45 ET.
- Fetch summary before chain so a chain retry cannot create an unbounded mixed
  snapshot. Store the actual receipt and vendor timestamps for both.
- Persist raw responses, manifest, and normalized live feature inputs.
- Do not score the dashboard or create recommendations yet.

**Exit:** five market sessions with one complete, timestamp-valid snapshot each,
or explicit coverage failures.

### Step 2: feature assembler and causal audit

- Build `LiveFeatureContext` from a named live snapshot plus the prior final
  EOD state. It must be separate from the EOD `FeatureContext`.
- Make the live assembler return per-feature provenance and observation time.
- Add `assert_live_causal`: every value must be at or before the decision
  timestamp, and every non-live value must be at or before the prior completed
  session.
- Add snapshot replay: feature frame and score hashes must be identical without
  network access.

**Exit:** a sampled live frame is complete for STR-THRU or refuses explicitly;
no field has ambiguous provenance.

### Step 3: historical intraday experiment

- Pull or reuse ORATS historical one-minute snapshots at exactly 15:45 ET for
  the eligible event universe. Respect the vendor historical coverage and
  request limits.
- Materialize the pre-close feature frame without accessing later same-day
  values.
- Train walk-forward D0-preclose variants, execute using the separately
  specified fill and entry convention, and generate reports for D0 close,
  D1 close, and D0 pre-close arms.
- Compare selection overlap, calibration, return, Sharpe, drawdown, and
  worst/mid/best fill sensitivity. Do not promote on a single headline return.

**Exit:** a preregistered experiment with generated reports and a recorded
promotion or non-promotion decision.

### Step 4: dashboard shadow mode

- Run live collection and pre-close scoring daily, but display a clear
  `SHADOW - NOT TRADEABLE` badge and create no portfolio open event.
- After the close, compare the frozen pre-close recommendation to the close
  score and record feature deltas, selected contracts, quote freshness, and
  later measured fills.
- Show the clock, score timestamp, source freshness, model version, and
  decision deadline on every row.

**Exit:** at least twenty valid shadow sessions and no timestamp, source, or
replay-equivalence failure.

### Step 5: paper execution, then promotion

- Paper trade only the selected clock-qualified model using a documented order
  policy and measured fills.
- Keep close-clock and pre-close recommendations visible side by side during
  calibration.
- Promote only through the normal registry and evidence workflow. Rebuild
  dashboard model evidence after a registry change.

## 7. Acceptance tests

1. **No EOD contamination:** poison a same-day Tier-2, Tier-3, or Tier-4
   fixture after a live snapshot; its feature frame and score must not change.
2. **Cutoff test:** a vendor row timestamped after 15:45 ET is rejected, even
   if it is fresher than the selected row.
3. **Source schema test:** all required ORATS fields map once, with expected
   units and non-null values on a fixture.
4. **Snapshot replay:** a stored manifest and raw responses reproduce the
   normalized feature frame and score byte-for-byte without network access.
5. **Clock isolation:** a D0 close model cannot score a D0 pre-close request,
   and vice versa.
6. **BMO and AMC timing:** session-aware event logic yields the expected
   decision, entry, and exit windows for both print types.
7. **Dashboard provenance:** every rendered live row names its clock, snapshot
   hash, receipt time, source freshness, and model artifact fingerprint.
8. **Failure behavior:** missing live quote, stale quote, incomplete chain, or
   missing feature produces a refusal and never a recommendation.

## 8. Non-goals and hazards

- This does not turn end-of-day Tiers 1 through 4 into intraday data.
- This does not assume the close price is executable at 15:45. The entry rule
  and fill measurement need their own experiment.
- A live vendor quote is a recommendation input, not proof of a tradable fill.
  Paper fills and the existing real-price checks remain required.
- ORATS intraday historical coverage is shorter than the EOD research history.
  Report the restricted sample and do not claim full-history equivalence.
- Do not expose raw ORATS live snapshots in the public dashboard bundle or
  public Git history.

## 9. Definition of done

The capability is complete only when a clock-qualified model has a generated
walk-forward report, its live feature contract has green acceptance tests, and
the dashboard can reproduce a timestamped shadow score from its saved
snapshot. A green collector alone is not a live trading signal.
