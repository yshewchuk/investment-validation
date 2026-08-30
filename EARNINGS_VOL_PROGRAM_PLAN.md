# Earnings-Vol Trading Program — Multi-Phase Implementation Plan

**Version:** 1.3 · **Date:** 2026-08-30 · **Owner:** YS + Claude
**v1.1:** CAL-P corrected to the simultaneous spec (both legs opened together
pre-print, closed together post-print); data layer expanded to an explicit
three-tier raw/normalized/feature architecture.
**v1.2:** source control & recovery added — all code in a public GitHub repo
(no secrets, no market data); irreplaceable non-code artifacts (ledger,
reports, research findings) mirrored to a private remote.
**v1.3:** the EXP-050 5%-sizing row is corrected everywhere below from
"2.4× / MC P(loss) 6%" to the reproducible **"2.83× / MC P(loss) 15%"**.
The Phase 2 harness regression proved the old row reproduces from nothing
(`reports/phase2_exp050_regression.md`); the 10% and 20% rows reproduce
exactly, which is what makes the diagnosis certain. **This is a risk-posture
change, not a typo fix:** 15% probability of final loss at the base sizing is
materially worse than 6%. The Phase 5 go-live rules are re-anchored to the
corrected numbers, and whether 5% sizing remains acceptable at P(loss) 15% is
an explicit decision to revisit at the Phase 5 go/no-go memo (options: keep
5% and accept 15%, drop to 2%, or require a fresh pre-registered gate run
through `engine.evaluate` before any capital decision).
**Scope:** the three core earnings-vol strategies, a continuous scoring/monitoring
system, a continuous-improvement experiment framework, and the AI-correction
long-put overlay.
**Foundation docs:** `earnings_predictions/HANDOFF.md`, `earnings_predictions/opf/VERDICT_2026-08-29.md`,
`earnings_predictions/strategies/VERDICT_2026-08-28.md`, `AGENTS.md` (research stance),
`STRATEGY.md` (semis dispersion / thesis universe).
**Build guides:** `guides/` — one detailed implementation guide per phase
(architecture, contracts, constraints, acceptance tests). Implementing agents
read `guides/README.md` first; the guides are the how, this plan is the what.

---

## 0. Where we stand (the inputs this plan builds on)

**Data moat (all local, never re-download):**
- 139,787 earnings events / 2,936 tickers / 2007–2026 (oquants moves + ORATS
  summaries/cores/earnings), leak-free causal panel in
  `earnings_predictions/data/processed/events_with_orats_sum.csv`.
- 17,679 real EOD option-chain files (~76.8k date×ticker chains, 2017–2026,
  ORATS `/hist/strikes` bid/ask, validated within ±2–3% vs Polygon real trades
  and live yfinance quotes).
- 35,167 true 50-delta straddle implied moves (`true_implied.csv`).
- ORATS quota resets **Sep 1** (20k calls/month) — the gating resource for the
  10× sample enlargements below.

**Champion models (current bests):**
- **Size model v1.3** — OLS+NN(64,32) blend, OOS r=0.459 on true implied
  rebuild (EXP-040).
- **OPF implied-move-change model** — walk-forward GBM predicting the change in
  quoted implied move from T−j to T−1; MAE 3.3–4.0pp, r 0.60–0.72, top-vs-bottom
  decile realized spread 15–22pp, positive every year 2015–2026 (EXP-043).
- **Mid-fill GBM gate** — trained on realized mid-fill returns; top-20% gate
  +4.6%/trade; walk-forward equity 2.83× at 5% sizing with MC P(loss)=15%
  (EXP-049/050; the original report's 2.4×/6% row was stale — see
  `reports/phase2_exp050_regression.md`).
- Direction remains unpredictable (AUC ceiling ~0.518 across 33 experiments) —
  no strategy in this program bets on direction at the event level.

**The two verdict-defining lessons (now codified in AGENTS.md):**
1. **Fill assumption is the verdict flipper.** Worst-case fills (buy ask / sell
   bid) kill every strategy; mid fills flip all three base exposures positive.
   Execution quality is not a detail — it is the program. Every result in every
   phase below is reported at worst/mid/best fills, side by side.
2. **The exposure is the asset; the gate is the optimization.** Base exposures
   are thin (+2–4%/trade); gates and slices (1–10B mcap +5.3%) are where the
   compounding comes from.

**Strategy name mapping** (user's three strategies → codebase):

| Code | User's description | Existing evidence | Status |
|---|---|---|---|
| **CAL-P** | Put calendar: short ~1 DTE put + long ~20 DTE put (back DTE subject to optimization), both legs opened together shortly before the print and closed together after it — profits from front-leg IV crush | EXP-046b is the nearest evidence (+2.0%/trade at mid, 58% win, 8/9 yrs) but tested a DIFFERENT spec: straddle legs, T−14 entry, unwound PRE-print. `s5_put_calendar/` experiments touch put structures. The exact spec has NOT been isolated | Promising direction; exact-spec backtest is Phase 2 backlog #1 |
| **STR-THRU** | Long straddle bought shortly before print, sold immediately after (mispriced expected move) | S2 exposure at mid fills +3.7%/trade 7/9 yrs (EXP-048); GBM gate +4.6% top-20% (EXP-049); equity 2.83×@5% sizing, MC P(loss) 15% (EXP-050, corrected by the Phase 2 regression) | Best-evidenced strategy |
| **STR-RUNUP** | Long straddle bought early, sold immediately before print (IV run-up harvest) | S3 exposure at mid fills +3.9%/trade 6/9 yrs (EXP-048); OPF timing model is the edge — enter on predicted run-up, exit T−1 (EXP-043) | Promising; timing model is the differentiator |
| **PUT-THESIS** | Long puts on AI-bubble casualties, timed | STRATEGY.md S1/S4 playbooks (semis dispersion, walk-forward validated); regime-timing research says event-level timing signals are null | Phase 6 overlay, thesis-driven with tripwires |

**Important structural note on CAL-P:** both legs open together and close
together — at no point is the short put naked. At the same strike this is a
defined-risk debit structure (max loss ≈ net debit paid), which changes the
risk profile vs the outright short-vol trades. Two things still need explicit
verification before trading it:
1. **The evidence gap.** EXP-046b (the +2.0% number) tested straddle legs,
   T−14 entry, and a pre-print unwind — none of which match this spec. The
   exact spec (put legs, entry shortly before the print with ~1 DTE front,
   held THROUGH the print, closed together after) has never been isolated;
   it is Phase 2 backlog #1, and the prior experiment's framing should be
   treated as a misunderstanding to be corrected, not as validation.
2. **Mechanics of the 1 DTE short leg through the event:** early assignment
   risk if the short put goes ITM on a big down move (American exercise), and
   pin risk at front expiry. The backtest must count assignment-eligible
   events, and the live protocol must close both legs promptly post-print.

---

## Phase 0 — Consolidation & data foundations

**Goal:** one coherent engine instead of scattered scripts; data layer ready
for daily operation; the Sep 1 quota spent deliberately.

**Deliverables**
1. **Data architecture — three tiers** (`engine/data/`). Policy: every byte
   ever fetched is kept raw forever (quotas and throttling make refetching
   expensive), and everything downstream reads from ONE normalized store
   instead of four per-source formats.
   - **Tier 1 — raw immutable cache** (`data/raw/`): every API response
     persisted verbatim before any parsing, keyed by
     (source, endpoint, canonical-params hash), append-only, never edited or
     deleted. One fetch wrapper for ALL sources (`engine/data/fetch.py`):
     cache-first (a repeated request never touches the network), per-source
     throttle pacing and backoff (the AGENTS.md playbook, centralized), and a
     fetch log (timestamp, quota cost, status) that feeds the quota ledger.
     Existing raw dirs (`earnings_predictions/data/raw/{orats,oquants,yfinance,polygon}`,
     `polygon_cache/`) are grandfathered in place and registered in the
     manifest; all NEW pulls go through the wrapper.
   - **Tier 2 — normalized store** (`data/curated/`, Parquet, partitioned by
     year): one coherent cross-source schema —
     `securities` (symbology, listing ranges, era-normalized mcap),
     `earnings_events` (ticker, date, BMO/AMC session, per-source agreement
     flags),
     `daily_market` (ticker × date: spot, IV terms, ex-earn IVs, implied
     move, rvol, skew, contango — with a source column per field),
     `option_chains` (ticker × obs_date × expiry × strike × right: bid/ask/
     mid, IV, delta, source),
     `trades` (every simulated, paper, and live trade in one schema).
     Normalizers are idempotent and can rebuild Tier 2 from Tier 1 from
     scratch (`rebuild.py`); every row carries provenance (raw file + fetch
     id). All known unit/convention traps are fixed HERE, once, instead of in
     every consumer: ORATS mktCap unit switch (2026-03-11), BMO/AMC move
     convention, oquants pre-2022 implied moves flagged as reconstructions,
     and a written source-priority rule where sources disagree (chains: ORATS;
     realized moves: oquants panel validated vs OHLCV; calendar: ORATS
     `anncTod`).
   - **Tier 3 — feature/serving layer** (`data/features/`): the causal panel,
     model feature matrices, analog trade sets — fully derived, rebuildable
     from Tier 2, versioned by snapshot hash (this hash is what every report's
     provenance block pins).
   - **Ingestion validation gate:** new raw data must pass the sanity battery
     (spot vs yfinance, straddle-mid cross-checks, schema checks) before its
     normalized output lands in Tier 2; failures quarantine the raw file and
     raise a flag — never a silent drop, never a blocked pull (raw is always
     kept regardless).
   - **Migration test:** the first Tier-3 build must reproduce the existing
     master panel (`events_with_orats_sum.csv`) row-for-row within tolerance.
     This one-time reconciliation doubles as the regression test proving the
     new pipeline didn't change any number the verdicts rest on.
   - `data/MANIFEST.md` becomes generated output of the tiers, not a
     hand-maintained doc.
2. `engine/` package at repo root consolidating the sanctioned pieces:
   - `engine/fills.py` — the fill model as a first-class object:
     `FillModel(alpha)` where alpha ∈ [0,1] interpolates worst→best
     (alpha=0.5 = mid). Every P&L computation in the program takes a FillModel;
     nothing hardcodes a convention.
   - `engine/calendar.py` — canonical forward earnings calendar (ORATS
     `/hist/earnings` + live confirmation), with BMO/AMC session from `anncTod`
     (99.52% validated, EXP-038) and date-change detection (dates move; stale
     dates are a known loss source).
   - `engine/structures.py` — trade-structure definitions (put calendar:
     short ~1 DTE front put + long back put, both legs opened and closed
     together; straddle-through; straddle-runup) that produce leg lists; one
     pricing path for all strategies.
3. **Sep 1 ORATS pull plan** (budget ~16k of 20k calls, logged in
   `quota_log.csv`; all pulls through the Tier-1 fetch wrapper):
   - T−14 + T−1 + post-print chains for the full 2017–2026 event universe in
     the 1–10B and >10B mcap slices → enlarges the calendar sample ~10×
     (n=359 → ~3.5k) and the gate sample similarly.
   - Put-side chains specifically (CAL-P needs puts; much of the cache was
     pulled straddle-centric).
   - **Liquidity fields on every new chain row** (added 2026-08-30): the pull
     now requests `callVolume, callOpenInterest, callBidSize, callAskSize` and
     the put equivalents. ORATS bills per CALL, not per field, so this costs
     nothing — and the existing 19,061-file cache has none of them, because the
     field list never asked. They are the first evidence in the program that
     bears directly on the mid-fill assumption every headline rests on: open
     interest, whether a contract traded at all, and the size actually resting
     at the touch. Back-filling them later costs the same 16k calls again, so
     the window is this pull. Landing nullable in Tier 2 (`volume`,
     `open_interest`, `bid_size`, `ask_size`); pre-Sep rows stay NULL, because
     "never asked" and "no size" are different facts.
   - Reserve ~3k calls/month for daily live operation (Phase 3).
4. Refresh of dailies/summaries to current date for live scoring.
5. **Source control & recovery (public GitHub repo) — set up FIRST, before
   new code accumulates.** All code and build docs live in a public GitHub
   repository so the work survives the laptop: `git init` at repo root with
   an ALLOWLIST `.gitignore` (everything ignored by default; code and docs
   explicitly allowed). Never in the public repo: `.env` or any credential
   (commit `.env.example` with variable NAMES only), all data tiers and
   caches (licensed and/or re-pullable), results/reports/figures and
   research verdict docs (data-derived findings), `ledger/`, and live
   watchlist configs (`config/thesis/universe.yaml` — position intent). A
   pre-commit hygiene check (`checks/repo_hygiene.py`) blocks staged content
   containing current secret values, data-file extensions, and files >1 MB.
   Accepted by design: the public code discloses the strategy *logic*;
   positions, findings, and data do not ship.
   **Private mirror for the irreplaceable non-code artifacts:** the nightly
   job syncs `ledger/`, reports, research findings docs, and
   `config/thesis/` to a private remote (private GitHub repo or the R2
   bucket) — these cannot be regenerated, unlike raw market data, which is
   deliberately NOT backed up (re-pullable at quota cost). A `RECOVERY.md`
   in the public repo documents the full restore path: clone, keys from the
   password manager, private-mirror restore, data re-pull scripts.

**Verification outputs (Phase 0 report: `reports/phase0_data_audit.*`)**
- Coverage heatmaps: events × year × mcap bucket with chain availability;
  put-vs-call chain coverage; DTE availability at entry/exit dates.
- Price-sanity battery re-run on the new pulls: ORATS spot vs yfinance close
  (tolerance 1.3%), straddle mids vs Polygon where 2024-08+ overlap exists.
- Row counts + checksums into `data/MANIFEST.md`; quota ledger reconciliation.

**Exit criteria:** engine imports cleanly and reproduces one known backtest
number per strategy from the verdict docs; Tier-2 rebuild from raw reproduces
the existing master panel (migration test green); new chain pulls pass the
sanity battery; coverage report shows ≥90% chain availability for the target
slices; public repo pushed with hygiene checks green and a recovery drill
(fresh clone → import smoke) passed.

---

## Phase 1 — Strategy scoring engine (expected PnL & win rate per ticker/strike)

**Goal:** `score(ticker, strategy, strike, expiry, as_of)` → expected PnL, win
rate, confidence interval, and the evidence behind them — computable for any
ticker/strike combination, powered by the current champion model per strategy.

**Deliverables**
1. **Model registry** — `engine/models/registry.json`: one entry per model
   (id, strategy, artifact path, feature list, training window, eval metrics,
   `champion: true/false`, promotion date, link to the experiment report that
   justified promotion). Champions load from here; nothing imports a model
   file directly. Initial champions: v1.3 size blend, OPF GBM T−1 implied
   predictor, mid-fill GBM gate.
2. **Scoring API** (`engine/score.py`), returning for each
   (ticker, strategy, strike, expiry):
   - `exp_pnl` and `win_rate` from TWO estimation layers, both always shown:
     - **Model layer:** champion-model prediction (predicted |move|, predicted
       T−1 implied, gate score) pushed through the structure's payoff at the
       chosen FillModel.
     - **Analog layer:** the empirical distribution of realized returns from
       matched historical trades (same strategy, matched by mcap bucket,
       implied-move level, DTE, moneyness) out of the trades_real datasets.
       Reports n_analogs, mean, median, win rate, p10/p90.
   - Disagreement between the layers is surfaced as a flag, not averaged away.
   - `ci`: bootstrap CI on the analog set; `gate`: pass/fail + score +
     threshold; `model_version`: registry id, so every number is traceable.
3. **Strike generalization:** existing evidence is ATM-centric. A dedicated
   experiment (runs through the Phase 2 framework) measures edge decay across
   moneyness (±1, ±2 strikes, delta buckets) per strategy before the scoring
   engine claims accuracy away from ATM. Until then, non-ATM scores are
   labeled EXTRAPOLATED in every output.

**Verification outputs (per scoring run)**
- **Backtest-replay equivalence test:** running the scoring engine on
  historical as-of dates must reproduce the trades and stats of the Phase 2
  backtests (same trades in → same P&L out). This is the proof that the live
  scorer and the research code cannot drift apart. Automated, run in CI-style
  before any registry change.
- **Calibration report:** predicted win rate vs realized win rate by decile,
  predicted E[PnL] vs realized mean, on held-out years — with plots
  (reliability curves) and Brier scores. Recomputed every time the ledger
  (Phase 5) accrues 50+ new scored events.
- Reproducibility: `score(..., as_of=frozen_date)` is deterministic given the
  data snapshot; snapshot hash embedded in output.

**Exit criteria:** replay equivalence passes on all three strategies; scorer
runs for the full upcoming-3-weeks calendar in <5 min from cache.

---

## Phase 2 — Experimentation framework (continuous improvement scaffolding)

**Goal:** make the EXP-000..050 discipline a reusable harness: any new model,
gate, structure variant, or parameter change runs through one standardized
evaluation suite and either beats the champion or doesn't — with the evidence
auto-generated.

**Deliverables**
1. **Experiment scaffolding** (`experiments/` at engine level):
   - `new_experiment.py` generates a numbered folder from a template:
     `spec.yaml` (hypothesis, primary spec pre-registered, feature list, train/
     test split, promotion target), `run.py`, auto-generated `REPORT.md`.
   - The registry table in PLAN.md becomes generated output, not hand-edited.
   - **Multiple-testing ledger:** every spec evaluated against a dataset is
     logged (including failures). Promotion decisions cite the ledger so we
     know how many tries preceded a winner — the guard against the overfitting
     that 50 experiments of iteration invites.
2. **Evaluation suite** (`engine/evaluate.py`) — every candidate passes through
   all of:
   - **Backtest** on real chains, all three fill conventions (worst/mid/best),
     plus an alpha sweep (fill-quality degradation curve: at what alpha does
     the strategy break even? This number is a headline stat — it is the
     margin of safety on the mid-fill assumption).
   - **Walk-forward** — expanding window by year, parameters frozen before
     each test year (the existing convention: train ≤Y−1, trade Y). Headline
     numbers come ONLY from walk-forward out-of-sample years.
   - **Monte Carlo** — block-bootstrap (block=20, preserving earnings-week
     clustering) on the walk-forward trade sequence: P(loss), drawdown
     percentiles, terminal-equity distribution, and a sizing curve
     (2%/5%/10%/20% per trade) so position size is chosen from MC, not vibes.
   - **Stress tests:**
     - Crisis replays: restrict to 2018Q4, 2020-03, 2022, and the worst
       realized earnings weeks; report per-regime P&L.
     - Tail injection for short legs (CAL-P legged, any short structure):
       re-run with the worst 1% of realized moves doubled; report ruin risk.
     - Entry/exit slippage days (±1 day), missing-chain degradation, and
       stale-earnings-date errors (simulate 1% wrong dates).
     - IV-regime split: high-VIX vs low-VIX years (the edge leans on
       2022/2024 — quantify how much).
   - **Metrics** (one common table so strategies are comparable): per-trade
     mean/median/std, win rate, profit factor, **Sharpe** (per-trade
     annualized via trades/year, and equity-curve annualized), Sortino, max
     drawdown, tail ratio (p95 win / p95 loss), capacity notes (spread width ×
     volume at the traded strikes), and by-year table.
3. **Champion/challenger protocol:** a challenger is promoted only if (a) it
   beats the champion on walk-forward OOS mean and Sharpe, (b) MC P(loss) at
   5% sizing does not worsen, (c) it survives the stress battery, (d) its spec
   was pre-registered before the OOS evaluation. Promotion = registry update +
   auto-generated promotion report diffing champion vs challenger.
4. **Backlog of first experiments to run through the new harness** (priority
   order):
   1. **CAL-P exact-spec backtest** at 10× sample (post-Sep-1 pull): put
      legs, both opened together shortly before the print (front DTE ~1),
      closed together post-print; back-DTE grid {7, 14, 20, 30, 45};
      same-strike vs diagonal; entry-day sweep (T−1 vs T−2). Compare against
      the EXP-046b variant (straddle legs, T−14, pre-print unwind) to isolate
      where the P&L actually accrues — this corrects the prior experiment's
      structure misunderstanding.
   2. **CAL-P risk mechanics:** early-assignment exposure (frequency of the
      short put trading ITM at the post-print open), pin risk at front
      expiry, and verification that realized max loss stays ≈ net debit
      across the full sample (the defined-risk claim, proven not assumed).
   3. STR-RUNUP entry-day optimization using the OPF T−j model (which j per
      predicted-run-up decile).
   4. Moneyness/strike edge-decay map (feeds Phase 1 strike generalization).
   5. Full-universe mid-fill gate test (EXP-047, still open).
   6. 1–10B mcap slice deep-dive with Polygon real-fill spot checks (the
      claimed +5.3% pocket; unverified at real fills for the smallest names).

**Verification outputs:** every `evaluate.py` run emits the standard report
(Phase 4 format) — no experiment result exists without one.

**Exit criteria:** one historical result (EXP-050 equity curve) reproduced
end-to-end through the new harness with matching numbers; backlog items 1–3
completed through it.

---

## Phase 3 — Continuous monitoring dashboard

**Goal:** the always-on view: for every upcoming earnings event, what does each
strategy expect to make, at what win rate, on which ticker/strike — refreshed
daily, with the evidence one click away.

**Deliverables** (extend `dashboard/` — server, refresh-job, and caching
patterns already exist for the semis scanner; add earnings-vol views):
1. **Upcoming prints board:** every confirmed event in the next 3 weeks × the
   three strategies: gate status, expected PnL (both estimation layers), win
   rate + CI, n_analogs, current premium levels vs model-fair, recommended
   entry/exit dates per strategy (STR-RUNUP entry day comes from the OPF
   model), and data-freshness badges. Sortable/filterable by gate score.
2. **Ticker/strike explorer:** pick a ticker → strike × expiry matrix
   heatmapped by expected PnL per strategy, with win rate on hover; ATM vs
   EXTRAPOLATED labeling per §P1.3; per-ticker evidence panel (its historical
   prints, implied-vs-realized history, past analog trades and their outcomes).
3. **Model health view:** rolling calibration (predicted vs realized win rate),
   live size-model MAE vs the implied-move baseline (the daily-scored ledger
   already started: model 5.8pp vs implied 8.1pp on 2026-08-28), data
   freshness/quota status, and champion registry versions.
4. **Ops:** daily post-close cron: refresh ORATS live summaries for calendar
   names (~budgeted from the 3k/month reserve), rescore the board, snapshot
   predictions to the ledger (Phase 5), publish the remote snapshot
   (deliverable 5), and flag: new gate triggers, earnings-date changes,
   calibration drift beyond threshold.
5. **Remote access (on-the-go):** two channels, both behind Cloudflare Access
   (free tier, email one-time-code login). Auth is not optional: the board
   discloses positions/intentions and redistributes ORATS-derived quotes
   (licensed data), so nothing ships publicly unauthenticated.
   - **Primary — published static snapshot.** The nightly cron renders the
     prints board + ticker/strike explorer as a self-contained static bundle
     (HTML + per-ticker JSON, rendered client-side) and pushes it to static
     hosting (Cloudflare Pages/R2 preferred for one-vendor auth; S3+CloudFront
     with signed access if AWS is preferred). Rationale: all data is EOD and
     the board changes once a day, so static loses nothing; it stays up when
     the WSL2 box sleeps (WSL2 suspends with the host); and it exposes zero
     server surface — no refresh endpoint that a stranger could hammer into
     the ORATS quota.
   - **Secondary (optional) — named cloudflared tunnel** to the local server
     for desk-time interactive use (manual refresh, ad-hoc rescoring).
     Outbound-only, so no WSL2 inbound port-forwarding; accepted limitation:
     down whenever the machine sleeps.
   - **Rejected:** public S3 bucket with an obscure URL (obscurity is not
     auth; licensing + position disclosure), and exposing the live server's
     refresh endpoint without auth.

**Verification outputs:** the board itself shows, per number, the model
version and n_analogs behind it; a nightly self-check compares the dashboard's
scores against a direct `engine/score.py` invocation (no silent divergence
between UI and engine).

**Exit criteria:** dashboard covers one full earnings week live with zero
manual intervention; nightly self-check green; remote snapshot reachable and
current from a phone (behind Access) for that full week.

---

## Phase 4 — Verification & reporting layer (cross-cutting)

**Goal:** the user-stated requirement that every piece outputs rich, auditable
reports. Built as one generator used by Phases 1–3 and 5, not per-phase
one-offs.

**Deliverables**
1. **Report generator** (`engine/report.py`): every evaluation, promotion,
   scoring calibration, and forward-test review emits a standard report
   (Markdown + figures, optionally HTML) containing:
   - Headline table (all metrics from §P2.2.5) at worst/mid/best fills.
   - Equity curve + drawdown chart; by-year bar chart; MC fan chart with
     percentile bands; stress-grid heatmap; calibration/reliability plots;
     fill-alpha breakeven curve.
   - **Provenance block:** data files used (+ row counts + manifest
     checksums), code git-style hash or file mtimes, spec hash, quota state,
     random seeds. A report that can't be regenerated from its provenance
     block is a bug.
2. **Accuracy-evidence standards** — the codified answer to "how do we know
   the results are actually accurate", printed as a checklist section in every
   report with pass/fail:
   1. Real traded/quoted prices only (ORATS chains, validated ±2–3%; oquants
      model-fitted marks banned from P&L — standing rule).
   2. Leak audit passed: automated check that every feature timestamp is
      strictly pre-decision (the existing causal-panel discipline, made
      mechanical: features carry as-of dates; the auditor asserts
      as_of < decision time).
   3. Headline numbers are walk-forward OOS only; in-sample numbers appear
      only in clearly-labeled diagnostics.
   4. Fill-sensitivity shown (worst/mid/best + breakeven alpha).
   5. Multiple-testing ledger cited (how many specs were tried).
   6. Survivorship caveat quantified where relevant (current-listed universe).
   7. For live claims: the prediction ledger (below) is the ultimate
      validator — frozen predictions scored later, no retroactive edits.
3. **Prediction ledger** (`ledger/`): append-only. Every daily dashboard
   snapshot writes its predictions (event, strategy, strike, exp_pnl,
   win_rate, model versions) BEFORE outcomes are known; a scorer joins
   outcomes after the event. This is the out-of-time, out-of-code-path test
   that catches anything the backtests miss. Cumulative ledger calibration is
   the first chart on the model-health view.

**Exit criteria:** all Phase 2 backlog reports and the Phase 1 calibration
report render through this generator; ledger accumulating daily.

---

## Phase 5 — Forward test → live protocol

**Goal:** prove the mid-fill assumption — the single biggest risk in the whole
program — before real capital scales.

**Deliverables**
1. **Paper-trading ledger with resting-limit simulation:** for every gate
   trigger, log the intended limit price (mid at decision time) and track
   whether the market traded through it (Polygon bars / next-day ORATS
   quotes). Output: **realized fill-quality alpha (α̂)** per strategy per
   liquidity bucket — the measured number that replaces the mid-fill
   assumption in all Phase 1 scoring and Phase 2 evaluation.
2. **Go-live rules** (pre-registered now, so the decision isn't made in the
   moment):
   - Minimum one full earnings season (Q3 2026, mid-Oct→mid-Nov) of paper
     trades with α̂ ≥ breakeven alpha + margin, before any real order.
   - Initial sizing from MC: 5% per trade (P(loss) 15% at current evidence —
     corrected from the stale 6% quote; see the v1.3 note and
     `reports/phase2_exp050_regression.md`; the 5%-at-15% posture is an
     explicit input to the go/no-go memo, not a settled fact);
     escalation to 10% only after a full season of live trades within MC
     bands; hard stop and post-mortem if drawdown exceeds the MC p95.
   - CAL-P trades (paper or real) only after its exact-spec backtest and
     risk-mechanics reports (Phase 2 backlog #1–2) pass — the current
     evidence is for a different structure and does not authorize this one.
3. **Broker reconciliation:** once live, every fill reconciled against the
   ledger's intended price; slippage report monthly; α̂ updated.

**Verification outputs:** weekly forward-test report (generator format):
paper P&L vs model-expected P&L with CIs, fill-quality distribution, gate
trigger log with hindsight outcomes.

**Exit criteria:** a season of ledger data; α̂ measured; go/no-go memo written
from the pre-registered rules.

---

## Phase 6 — AI-correction thesis overlay (long puts)

**Goal:** express the macro thesis (unprofitable-AI failures + circular-lending
unwind → broad downside) with defined-risk long-put structures, timed by
observable tripwires — while staying honest about what our own research says.

**The honesty constraint:** 33 experiments found event-level direction and
regime→direction timing statistically null; weekly market tone has zero
persistence. So this module does NOT pretend a model can time the correction.
It is thesis-driven, with pre-registered observable triggers and structures
chosen to survive being early — because the main way long-put theses die is
theta bleed while waiting.

**Deliverables**
1. **Exposure map:** extend the STRATEGY.md 51-name semis classification into
   an AI-bubble vulnerability universe: circular-financing exposure (vendor
   financing, customer-concentration in other AI companies, RPO quality),
   capex-to-FCF, cash runway for the unprofitable names, and index-flow
   inflation (the S1/S4 machinery already computes run-up + PIT fundamentals).
   Rank names by "falls hardest in the unwind".
2. **Tripwire board** (new dashboard view; all observable, none predictive):
   - Credit: HY/IG spread widening, AI-adjacent debt issues trading down.
   - Fundamentals: cluster of guidance cuts / capex-cut announcements among
     hyperscalers; vendor-financing writedowns.
   - Market internals: breadth breaks, index IV term-structure inversion,
     skew steepening on the vulnerability list.
   - Each tripwire pre-registered with a threshold and a staging rule (e.g.
     2 of 4 armed → stage 1 sizing; 3 of 4 → stage 2).
3. **Structures that survive being early:** put diagonals/calendars (long back
   put financed by short front put — reusing the exact CAL-P pricing
   machinery) and put spreads, instead of outright long puts; an explicit
   **insurance budget** (≤ a fixed % of equity per quarter, spent knowing the
   base case is it expires worthless) rather than an EV claim.
4. **Interaction rule with the core strategies:** in a correction, the short
   legs of CAL-P and the earnings-vol regime both change (market overprices
   event vol MORE in high-vol regimes per EXP-016 — but tails fatten). The
   stress-test battery (§P2) already includes crisis replays; the tripwire
   board at stage 2+ also tightens core-strategy gates (raise the gate
   threshold, suspend short-leg structures like CAL-P, halve sizing) —
   pre-registered here so the response is mechanical.
5. **Backtest of the playbook** through the Phase 2 harness where testable:
   put-diagonal P&L on the vulnerability list during 2018Q4/2020-03/2022
   replays (what would the structures have done in past unwinds, at real
   chain prices for the 2017+ window).

**Exit criteria:** exposure map + tripwire board live on the dashboard;
structure backtest report done; staging rules and insurance budget written and
frozen.

---

## Sequencing, dependencies, and calendar anchors

| Order | Phase | Depends on | Anchor |
|---|---|---|---|
| 1 | Phase 0 | Sep 1 ORATS quota reset (the 10× pulls) | Sep 1–7 |
| 2 | Phase 1 scoring engine | Phase 0 engine + registry | Sep, week 2 |
| 3 | Phase 2 harness + backlog 1–3 | Phase 0 data, Phase 1 replay test | Sep, weeks 2–3 |
| 4 | Phase 4 report generator | built alongside Phase 2 (its output format) | Sep, weeks 2–3 |
| 5 | Phase 3 dashboard | Phase 1 scorer + Phase 4 ledger | Sep, week 4 — live before Q3 season |
| 6 | Phase 5 forward test | everything above | **Q3 earnings season, mid-Oct → mid-Nov 2026** = the proving season |
| 7 | Phase 6 thesis overlay | Phase 2 harness (structure backtests), dashboard | parallel from Sep, week 3; tripwires live before Q3 season |

The hard calendar fact: Q3 earnings season is ~6 weeks out. The program's goal
is scorer + dashboard + ledger live before it starts, so the season produces a
full forward-test dataset.

---

## Top risks (ranked)

1. **Mid-fill achievability.** The entire positive verdict rests on limit
   fills at mid. Mitigation: breakeven-alpha reporting everywhere, Phase 5
   resting-limit measurement before capital, liquidity-bucket slicing.
2. **Lumpy, regime-dependent edge.** 2021 and 2025 were negative; 2022/2024
   carry the curve. Mitigation: MC sizing (5% ⇒ P(loss) 15%, corrected), regime stress
   splits, no escalation without a live season inside MC bands.
3. **Overfitting by iteration.** ~50 experiments have already touched this
   data. Mitigation: pre-registration, multiple-testing ledger, walk-forward
   only headlines, and the prediction ledger as the out-of-time backstop.
4. **Small samples on the best structures.** CAL-P n=359 until the Sep pulls;
   1–10B pocket unverified at real fills. Mitigation: Phase 0 pulls first,
   Polygon spot checks, EXTRAPOLATED labeling until verified.
5. **CAL-P short-leg mechanics.** The ~1 DTE short put held through the print
   carries early-assignment and pin risk even though the structure is
   defined-risk on paper. Mitigation: backlog #2 quantifies both before any
   trade; live protocol closes both legs promptly post-print; tail-injection
   stress still runs on every short-leg structure.
6. **Data/quota fragility.** 20k ORATS calls/month, oquants endpoints flaky,
   credentials rotate. Mitigation: budget ledger, local-cache-first rule,
   reserve for live ops.
7. **Laptop loss / secret leak.** Code survives via the public repo and the
   ledger/reports via the private mirror — but a secret pushed to a public
   remote is compromised permanently regardless of later removal.
   Mitigation: allowlist .gitignore, pre-commit secret/data scan, `.env`
   values grepped against staged content; if a secret ever reaches the
   remote, rotate it immediately — do not just delete the commit. Residual
   gap accepted: raw market data is not backed up (re-pullable at quota
   cost).

---

*Nothing here is financial advice. All results cited are historical
simulations with documented assumptions; live performance depends on execution
quality that Phase 5 exists to measure before capital scales.*
