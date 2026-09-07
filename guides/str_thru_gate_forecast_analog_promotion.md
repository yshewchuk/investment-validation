# STR-THRU gate: forecast + analog features replace the incumbent

*2026-09-07. Decided by the user. Rollback is one edit to `engine/models/registry.json`
(flip `champion` back to `gate_midfill_str_thru`, `false` on the new entry) or
`engine.models.registry.register()` with the old entry's `champion=True`.*

## What changed

`gate_midfill_str_thru` (41 features: event history + pre-print volatility
state) is replaced by `gate_midfill_str_thru_forecast_analog` (the same 41
plus six more):

* the size model's own forecast — `pred_abs_move`, its p10/p90/sd, and
  `forecast_edge = pred_abs_move - im` (predicted move minus the quoted
  implied move at entry, both percent of spot; positive means the model
  expects a bigger move than the market is pricing, STR-THRU's stated
  thesis);
* matched analog-trade statistics — `analog_mean`, `analog_win_rate`,
  `analog_n` from `engine.analogs.AnalogMatcher`, the same empirical layer
  the board already shows next to the model layer for every event.

Same GBM class, same hyperparameters, same top-20% selection discipline, same
mid-fill target. Both signals were already computed elsewhere in the engine
for other purposes (the forecast prices STR-THRU itself via
`engine.payoff.PAYOFF_DRIVER`; the analog stats are the board's own analog
layer) — this is the first gate to consume either as a **feature**, not just
display them separately.

## Why

`EXP-145` compared seven gating arms for STR-THRU: the incumbent alone, rule
gates on the forecast and the analog evidence separately and combined, and
three GBM variants adding each signal group (and both) to the incumbent's
feature set. All six challengers beat the incumbent on CAGR and Sharpe, even
isolating to the years the gate actually selects on (2020-2026, excluding the
~70% of trades that come from the identical ungated 2018-2019 cold start
every arm shares). The combined model — features 5+6 together — was the
strongest on every metric tested.

`EXP-147` re-ran that one configuration independently, as its own
pre-registered confirmatory spec, against a champion baseline that is itself
a fresh same-day run of the current registry champion
(`EXP-145`'s `arm1_incumbent_model` — not `EXP-105`, which pre-dates the
2026-09-06 registry promotion and is stale for this comparison).

**Held out nowhere — walk-forward OOS 2020-2026, gated selection only, mid
fills, on the full STR-THRU event universe:**

| | Challenger | Champion |
|---|---:|---:|
| CAGR | **235.5%** | 118.4% |
| Sharpe (trade) | **2.28** | 1.79 |
| Years positive | **9/9** | 6/9 |
| MC P(loss) @ 5% | 0.0% | 0.0% (tie) |
| Stress battery | no new red cell | — |
| Breakeven alpha | **0.41** | 0.43 |

`experiments/promote.py`'s mechanical `decide()` against this baseline:

```
PASS (a1) CAGR +235.48% > champion +118.40%
PASS (a2) sharpe_trade 2.279 > champion 1.789
PASS (a3) positive in 9/9 years >= champion 6/9
PASS (b)  MC P(loss)@5% 0.000 <= champion 0.000
PASS (c)  stress battery: no new red regime cell
PASS (d)  pre-registration valid
PASS (e)  accuracy checklist clean
FAIL (f)  Brier skill -0.0730 < -0.05
```

## What this promotion is NOT

**It did not clear its own pre-registered promotion bar.** Rule (f) — the
program's calibration floor, `MIN_BRIER_SKILL = -0.05` — is the one rule this
candidate fails, and `experiments/promote.py` has no override: "no partial
promotion... any red → print why and exit nonzero" is the literal design.
This entry was registered by calling `engine.models.registry.register()`
directly, not through `promote.py --apply-registry`, because `promote.py`
correctly refused to. **This is a deliberate exception, made by the user,
not something the mechanical gate approved.**

**The bar itself was never applied to the incumbent it replaces.**
`gate_midfill_str_thru` was registered by the bulk Phase-1 training script
(`train_all.train_gate`, `engine.models.registry.register()` unconditionally
on `champion=True`), which has never gone through `promote.py`'s `decide()`
at all — there is no historical instance of a STR-THRU gate clearing this
specific rule. Measuring the incumbent against its own replacement's
baseline: its Brier skill is **-0.0875**, worse than the challenger's
**-0.0730**. The challenger is not the first STR-THRU gate to ship below the
calibration floor; it is the first to ship a number for the floor to be
measured against, and that number is an improvement, not a regression.

**Neither gate's `win_model` probability should be read as a probability.**
Both are worse-calibrated than the base rate. The `gate_score`/`gate_pass`
selection decision does not use the isotonic-recalibrated probability at all
— it thresholds the raw regression prediction against a fold-fitted quantile
— so this failure does not touch the actual selection mechanism the CAGR and
Sharpe numbers above measure. It touches the DISPLAYED win-rate estimate a
board reader might use to judge one trade's odds. Until this is fixed, that
number should not be trusted for either gate, old or new. **This is the
single most important sentence in this document.**

**The reliability_monotonicity also degraded**: 0.695 (incumbent) → 0.537
(challenger) — the isotonic map's decile-bucket ranking is noisier with the
extra features, consistent with (though not proof of) the more likely
explanation: a per-fold isotonic fit on ~1,000-2,000 test-year rows, applied
to a 49-feature model's rank ordering, has less data per calibration bin than
the 41-feature model had, and the calibration curve moves more than the
ranking itself does. Not verified further before this promotion; see Open,
below.

**Multiple-testing debt is real.** `EXP-145` registered seven gate
configurations in one comparison; this is the second (arm 1, ungated) and
seventh (arm 7) of those seven, now re-run as `EXP-147`. The ledger records
all nine spec hashes (seven from EXP-145, two from EXP-147's re-plan) under
their own rows — a promotion decision citing "the strongest of seven tries"
should read that count, not a single clean discovery.

**Forecast coverage is not free.** `pred_abs_move` is null on the fraction of
STR-THRU events with insufficient training history at their fold (~2% on the
measured universe, much lower than the ~46% a stale spec estimate first
assumed — see `EXP-145`'s `COMPARISON.md`) — rows without a forecast are
never selected by this gate, same discipline as an incomplete-feature row
today. Small, but not zero, and not previously a dependency for this gate.

## How to roll back

```python
from engine.models.registry import RegistryEntry, load_registry, register

old = load_registry().get("gate_midfill_str_thru")
old.champion = True
register(old)  # demotes gate_midfill_str_thru_forecast_analog for (STR-THRU, gate)
```

The live-serving wiring in `engine/score.py` (`_forecast_for_gate`,
`_gate_feature_frame`) is a no-op for any gate whose feature list does not
name the new columns, so rolling back the registry entry alone is sufficient
— no code needs to be reverted.

## Open

- **Fix the calibration, not just note it.** A finer per-fold bucket count,
  a global (not per-fold) isotonic fit, or a different recalibration method
  entirely are all worth a dedicated experiment before this floor is trusted
  again for either gate.
- **The forward test (Phase 5)** is the only thing that can confirm any of
  this the way EXP-147's re-run confirmed EXP-145's exploratory result. Both
  are one universe of 19,161 events, seen at least twice now.
- **`gate_midfill_str_thru`'s own Brier skill was never checked against this
  floor at promotion time.** Whether that changes how the September 6
  promotion should be read is a separate, unopened question.
