# Phase 6 Guide — AI-Correction Thesis Overlay (Long Puts)

**Objective:** express the macro thesis (unprofitable-AI failures, circular
lending unwind) with defined-risk put structures, timed by pre-registered
observable tripwires — while never pretending the timing is model-predicted
(our own research: regime→direction signals are null; weekly tone has zero
persistence).

---

## 1. Architecture

```
engine/thesis/
  universe.py        # vulnerability universe build + score
  tripwires.py       # tripwire evaluation from observable series
  staging.py         # armed-count -> stage -> sizing/tightening actions
  replay.py          # structure backtests through the Phase 2 harness
config/thesis/
  universe.yaml      # names + hand-curated exposure evidence (sourced)
  tripwires.yaml     # each tripwire: series, threshold, direction, weight — FROZEN before arming
  budget.yaml        # quarterly insurance budget cap + stage allocations
```

### universe.py — vulnerability map
- Start from STRATEGY.md's 51-name semis classification + the AI-compute
  list; extend with hyperscalers/AI-adjacent large caps as the user names
  them.
- Quantitative inputs (computed, point-in-time via the existing
  `bt/edgar_pit.py` machinery): capex/FCF, revenue growth vs valuation
  (P/S), cash runway for unprofitable names, 21d/63d run-up, index-flow
  attribution (return explained by SMH/QQQ beta vs idiosyncratic).
- Qualitative inputs (hand-curated in `universe.yaml`, each with a source
  URL and date): vendor-financing arrangements, customer concentration in
  other AI companies, circular-deal flags. **These are theses, not data —
  the YAML schema forces a `source:` and `as_of:` per claim, and the
  dashboard renders them as claims.**
- `vulnerability_score`: fixed weighted sum (weights in the YAML, frozen);
  output `data/features/thesis_universe.csv` ranked, rebuilt weekly.

### tripwires.py — observable, none predictive
Each tripwire: id, series source, computation, threshold, direction, weight.
Initial set (thresholds are placeholders — **the user signs off on final
values before the arming logic goes live**, then they freeze):
1. `hy_oas`: FRED series BAMLH0A0HYM2 (public CSV, no key). Armed when OAS
   widens ≥150bp off its trailing-90d low.
2. `guidance_cluster`: count of universe names cutting guidance in rolling
   30d (manual log entry with source; armed at ≥3). Manual-input tripwires
   are allowed but every entry is dated and sourced.
3. `breadth`: % of universe above its 50DMA (yfinance, cached). Armed <35%.
4. `index_iv_inversion`: SPY iv30 > iv90 for 5 consecutive sessions (ORATS
   summaries, cached).
5. `skew_steepening`: median put-skew z-score of the top-decile
   vulnerability names > +1.5 (ORATS skewing).
Evaluation is daily in the Phase 3 nightly job; state (armed booleans,
values, history) goes to `data/features/tripwires.json` and the dashboard
tripwire board. Every state CHANGE is ledgered (Phase 4 writer) — the record
of when tripwires armed is itself a forward test of the framework.

### staging.py — mechanical responses (from the plan, frozen in budget.yaml)
- Stage 1 (2 of 5 armed): deploy the stage-1 fraction of the quarterly
  insurance budget in put diagonals (long back put ~90–180 DTE financed by
  short front put — reuse `engine.structures`; the CAL-P pricing path prices
  it) on the top-ranked names.
- Stage 2 (3+ armed): deploy remainder; tighten core strategies — raise gate
  thresholds, suspend short-leg structures (CAL-P), halve sizing.
- De-escalation rules included (tripwires disarm → staged unwind), plus
  explicit invalidation: if the budget is spent and tripwires disarm, the
  loss IS the plan working (insurance expired worthless) — the report says
  so rather than reframing.
- **Budget cap enforced in code:** staging.py refuses orders past the
  quarterly cap regardless of how armed the board looks.

### replay.py — what the structures did in past unwinds
Through the Phase 2 harness (real chains, 2017+ coverage): put diagonals and
put spreads on the vulnerability list replayed over 2018Q4, 2020-02..04,
2022, plus the 2021 melt-up and 2023-24 rallies as the cost-of-carry control.
Index/ETF legs (SPY/QQQ/SMH) may need chain pulls — check cache coverage
first; budget within quota rules. Output: per-episode P&L, theta bleed per
quarter of being early, and the drawdown offset against the core strategies'
simulated P&L in the same windows (the hedge-quality number).

## 2. Constraints

- **No predictive claims anywhere.** Reports and dashboard copy must present
  tripwires as risk-management triggers. The one honest calibration check:
  the board should light up in KNOWN historical stress and stay quiet in
  melt-ups (§3.2) — that is a sanity check of the plumbing, not evidence of
  forecasting ability, and reports must say exactly that.
- Thresholds, weights, and budget are frozen in config before arming goes
  live; changes require a dated note and appear in the ledger.
- Long puts are bought as budgeted insurance, never sized by expected value.
- Data hygiene: FRED/yfinance pulls go through the Tier-1 fetch wrapper like
  everything else.

## 3. Acceptance tests (`checks/phase6_checks.py`)

1. **Tripwire correctness:** each computation reproduces hand-calculated
   values on fixed historical windows.
2. **Historical board replay:** evaluate tripwires over 2018–2026: the board
   reaches stage 1/2 during 2020-03 and 2022, and does NOT arm during the
   2021 melt-up or the 2023-24 rally. If it fails this, thresholds are
   miscalibrated — fix BEFORE freezing, document the iteration in the
   experiment report (threshold tuning on history is fitting; say so, and
   let the frozen thresholds be judged by the live ledger going forward).
3. **Budget cap:** synthetic stage-2 with an exhausted budget → orders
   refused.
4. **Staging state machine:** arm/disarm sequences step through stages and
   de-escalation exactly per config; every transition ledgered.
5. **Replay reports** render through the generator with the theta-bleed and
   hedge-offset numbers present.
6. **Schema enforcement:** a universe.yaml claim without source/as_of fails
   validation.

## 4. Failure modes

- Manual tripwires go stale (nobody logs guidance cuts) → the board shows
  per-tripwire last-updated ages; a stale manual input renders as UNKNOWN,
  not as disarmed.
- FRED/series discontinuity → fetch wrapper caches last good; freshness
  badge degrades visibly.
- Thesis names get acquired or restructure (the SWKS lesson) → universe.yaml
  supports an `excluded: reason` field; excluded names stay visible with the
  reason rather than disappearing.
