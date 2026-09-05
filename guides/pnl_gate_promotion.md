# The P&L gate replaces `cost < peak/2` for TWIN-P5

*2026-09-05. Decided by the user. Rollback is one edit to `engine/entry_rules.py`.*

## What changed

TWIN-P5's reward term is now **simulated expected return against a trailing
top-20% bar**, recomputed monthly over six months. The other two terms are
untouched — spread ≤ 25%, market cap ≥ $10B.

```
was:  cost < peak / 2
now:  exp_pnl_sim >= 80th percentile of exp_pnl_sim over the trailing 6 months
```

TWIN-P is unaffected and still gates on arithmetic.

## Why

`cost < peak/2` asks whether max profit beats max loss and answers it without
reference to where the print is likely to land. EXP-129 replaced it with the
quantity it is a proxy for, and swept the incumbent's own constant so the
comparison could not be won by 0.5 merely being the wrong number. It is not:
`cost < 0.5·peak` is the best arithmetic cell on Sharpe and year-consistency,
and the gate still beats it at every matched selectivity.

**Held out on 2023-2026**, with the window and quantile fixed on 2018-2022
alone (EXP-131):

| | P&L gate | incumbent |
|---|---:|---:|
| gated events | 287 | 154 |
| final equity | **2.53x** | 1.68x |
| CAGR | **29.78%** | 15.83% |
| Sharpe per trade | **1.54** | 1.15 |
| breakeven alpha | **0.445** | 0.461 |
| years positive | 4/4 | 4/4 |
| max drawdown | 22.8% | **13.3%** |

The window choice is re-derivable from the selection period alone — a sweep on
2018-2022 picks six months on Sharpe — and the 6m > 9m > 12m ordering
replicated out of sample.

## Why six months, and it is not performance

**No window from 6 to 36 months is distinguishable from any other on returns.**
EXP-130's bootstrap: 6m minus annual is −0.05pp with a 90% interval of
[−4.17, +4.03]; the two share 62% of their gated events.

Six months was chosen for **volume stability**. The realized share of candidates
admitted, against a 20% target, has a yearly standard deviation of 0.027 at six
months against 0.051 at twelve and 0.074 for a calendar-year rule — a yearly
range of 0.18–0.28 against the annual rule's 0.10–0.31. The requirement it
serves is a predictable, bounded trade count, which is an operating constraint
rather than an edge claim.

Month-to-month lumpiness is NOT fixed by any window: monthly share SD is 0.169
at both six and twelve months, because what varies month to month is how many
earnings land in it — median 24, min 1, max 89.

## What this promotion is NOT

**It did not clear its own distinguishability bar.** EXP-131 registered that a
block bootstrap on the held-out difference must exclude zero. It does not: the
CAGR difference is +9.47pp with a 90% interval of [−28.99, +43.71], P(>0) = 66%.
Four years cannot separate these rules. **This gate is live on a decision the
statistics could not make**, and that is the single most important sentence here.

**It changes the KIND of the gate, not just its accuracy.** `cost < peak/2` is
an arithmetic guarantee — it cannot be wrong about an event, whatever any model
believes. A simulated expectation is a model output, so a model failure now
ADMITS trades rather than merely mis-ranking them, and it fails silently: an
over-optimistic exit price on a cheap wing reads as edge.

**The registered mechanism was wrong.** EXP-129's hypothesis was joint
(move, crush) integration. A post-hoc 2x2 found the move's variance is the whole
effect — holding it at its point forecast costs ~9pp of CAGR while the crush's
variance moves nothing. The crush model earns its place as a LEVEL, setting exit
vol, and its distribution does not. Nobody should describe this as joint
integration doing the work.

**Drawdown is worse at the incumbent's operating point** — 22.8% against 13.3%.
At MATCHED selectivity the gate is the safer of the two at three of four points,
but at the operating point each rule actually chooses, the incumbent is calmer.

**Selection-to-holdout degradation was large**: Sharpe 2.00 → 1.54, max drawdown
10.6% → 22.8%.

**The expensive quartile earns nothing.** On the holdout, structures in the top
cost quartile returned −0.33% per trade against +7.9% to +10.1% for the other
three. Under fixed-fraction sizing that is a quarter of the book contributing
nothing rather than a capital sink — an earlier reading that called the book
capital-negative was wrong, and read an unsized statistic. Still worth its own
experiment.

**The selection debt is real.** EXP-129 registered 21 gate configurations,
EXP-130 seven more, and this rule is a variant of a cell from the first. Only
the WINDOW was held out, not the decision to use a P&L gate at all.

## How to roll back

In `engine/entry_rules.py`, set `TWIN_P5_RULE.terms = TWIN_P_RULE.terms`. The
arithmetic term is unchanged and still live for TWIN-P, `structure_peak` is
still computed, and nothing else has to move.

## Open

- **The forward test (Phase 5)** is the only thing that can confirm any of this.
  Everything above is one universe of 2,802 events seen many times.
- **EXP-127**, the TWIN-P5 confirmatory, is still unrun. This gate now sits on
  top of an unconfirmed structure.
- **The expensive-quartile weakness** — gate on expected P&L in dollars rather
  than in return, or keep a weak cost bound underneath.
- `data/features/pnl_sim_history.parquet` is seeded from EXP-129's 2,802 events
  and must be extended by the replay as new events price, or the trailing bar
  goes stale.
