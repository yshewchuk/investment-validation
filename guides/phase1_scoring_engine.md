# Phase 1 Guide — Model Registry & Scoring Engine

**Objective:** `score(ticker, strategy, strike, expiry, as_of)` → expected
PnL, win rate, CI, and evidence — deterministic, leak-audited, and provably
identical to the backtests.

---

## 1. Build order

1. `engine/models/registry.py` + `registry.json`; wrap the three existing
   champions as artifacts.
2. Feature builder (as-of feature vectors from Tier 3).
3. Analog matcher (empirical layer) — works before any model loads; useful alone.
4. Model layer + structure payoff simulation.
5. `score.py` public API + batch calendar scoring.
6. Replay-equivalence test, calibration report.

## 2. Model registry

`engine/models/registry.json` — list of entries:
```json
{"id": "size_v1_3", "strategy": "*", "role": "size",
 "artifact": "engine/models/artifacts/size_v1_3.joblib",
 "features": ["ema12r_abs", "dist_high", ...],
 "train_window": "<=2022", "eval": {"oos_r": 0.459, "source": "EXP-040"},
 "champion": true, "promoted": "2026-08-29", "evidence": "opf/VERDICT_2026-08-29.md"}
```
Roles: `size` (predicted |move|), `implied_t1` (OPF GBM predicted T−1 implied),
`gate` (mid-fill GBM gate). Loader enforces: valid schema, exactly one
champion per (strategy, role), artifact file exists and loads, feature list
matches the artifact's expected input names. **Initial task:** locate the
current champion artifacts in `earnings_predictions/` (retrain from their
scripts if only code exists — pin seeds and record the training run in the
registry entry). Registry edits happen ONLY via Phase 2's `promote.py` after
this phase.

## 3. Scoring API (`engine/score.py`)

```python
@dataclass
class ScoreRequest:  ticker; strategy; strike; expiry; as_of; fill=MID; variant=None
@dataclass
class ScoreResult:
    exp_pnl_model; win_model          # model layer
    exp_pnl_analog; win_analog        # empirical layer
    ci_low; ci_high; n_analogs; analog_widened
    gate_score; gate_pass
    extrapolated: bool                # non-ATM until edge-decay exp promoted
    flags: list[str]                  # e.g. LAYER_DISAGREE, WIDE_MARKET, THIN_HISTORY
    model_versions: dict; snapshot_hash: str; as_of: date
```
Plus `score_calendar(as_of, horizon_days=21, strategies=ALL) -> DataFrame`
(the dashboard's input): for each upcoming event × strategy, the default
structure (ATM, per-strategy expiry rules) and the top alternative strikes.

### Model layer
1. Build the as-of feature vector from Tier 3 (leak-audited — every feature
   carries `feature_as_of`; `engine.audit.assert_causal(features, as_of)`
   runs on EVERY call, not just in tests).
2. Champion models → predicted |move|, predicted T−1 implied, gate score.
3. Simulate the structure's payoff under the FillModel: predicted quantities
   → leg prices from the as-of chain snapshot → P&L distribution obtained by
   pushing the model's OOS **residual distribution** (empirical residuals
   from held-out years, stored with the artifact) through the payoff — NOT a
   normality assumption. `exp_pnl_model` = mean, `win_model` = P(P&L>0) of
   that distribution.

### Analog layer
Matched historical trades from the Tier-2 `trades` table (sim trades from the
Phase 2 backtests):
- Buckets: mcap {<1B, 1–10B, ≥10B} × implied-vs-own-history terciles ×
  DTE bands {1–3, 4–10, 11–25, 26–45} × |moneyness| {≤2% (ATM), 2–5%, >5%}.
- If n < 30: widen in FIXED order — drop moneyness, then DTE, then implied
  tercile — and set `analog_widened` to the number of dimensions dropped.
- Report mean, median, win rate, p10/p90 of matched returns at the requested
  FillModel alpha.
- CI: 2,000-draw bootstrap of the analog returns; seed derived from
  (snapshot_hash, request) so results are deterministic.

### Disagreement & flags
`LAYER_DISAGREE` when sign(exp_pnl_model) ≠ sign(exp_pnl_analog) or the model
mean falls outside the analog CI. Never average the layers — both are shown
everywhere downstream. `THIN_HISTORY` when the ticker has <4 prior events
(the known regime where models degrade, EXP-024 addendum 2).

## 4. Constraints

- Deterministic: (request, snapshot_hash) → identical result, byte-for-byte.
- No network: scoring reads Tier 2/3 + registry only. Fresh data arrives via
  the Phase 3 cron refreshing tiers first.
- Non-ATM strikes: `extrapolated=True` until the moneyness edge-decay
  experiment (Phase 2 backlog #4) is promoted; the flag must survive into
  every UI/report rendering.
- CAL-P scoring stays DISABLED (returns `flags=[UNVALIDATED_STRUCTURE]`, no
  numbers) until Phase 2 backlog #1–2 promote it — the current evidence is
  for a different structure.
- Speed: full 3-week calendar < 5 min from cache (vectorize the feature
  build; per-ticker loops only over the ~dozens of upcoming events).

## 5. Acceptance tests

1. **Replay equivalence (`checks/phase1_replay.py`) — the load-bearing test.**
   For each strategy, sample 10 historical as-of dates from the backtest
   period; run the scorer as-of each date with the backtest's fill alpha;
   assert (a) the gate-passing set of (ticker, structure) equals the Phase 2
   backtest's trades entered those days, (b) modeled entry cost matches the
   backtest's to 1e-6, (c) portfolio-level expected P&L consistent within
   tolerance. Any drift between scorer and research code fails loudly. Runs
   before every registry change.
2. **Determinism:** same request twice → identical ScoreResult (hash the
   serialized dataclass).
3. **Leak poison test:** shift one feature's `feature_as_of` past `as_of` →
   scorer must raise, not warn.
4. **Calibration report:** on held-out years, predicted win rate deciles vs
   realized (reliability curve), predicted vs realized mean P&L, Brier score
   — rendered through the Phase 4 generator. Sanity floor: monotone-ish
   reliability, Brier beats the base-rate predictor.
5. **Analog matcher:** hand-construct a synthetic trades table with known
   bucket means → matcher returns them; widening ladder triggers at n<30 in
   the specified order.
6. **Flag propagation:** a `LAYER_DISAGREE` and an `extrapolated` case appear
   correctly in `score_calendar` output.

## 6. Failure modes

- Registry artifact drift (model retrained but registry metrics stale) →
  loader compares artifact hash to registry hash; mismatch refuses to load.
- Missing chain for the requested expiry/strike → return flags=[NO_CHAIN]
  with analog layer only; never interpolate a price silently.
- Thin analogs after full widening (n still <30) → n_analogs reported as-is
  with `THIN_ANALOGS`; the dashboard renders it as low-confidence, the
  scorer does not invent a CI.
