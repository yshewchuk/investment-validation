# EXP-141 — a smaller menu, chosen out of sample

*2026-09-06. Three of four registered criteria pass. Nothing promoted.
Artifacts: `experiments/EXP-141_a_smaller_menu_choosing_which_structures/`.*

## 1. The idea

The chooser ranks eight enumerated families by simulated expected P&L. Three of
them it cannot rank at all: on 4,641 common-universe events the notched seven,
the centre-five and the stepped-ramp seven are called best with precision
12.6%, 12.3% and 16.4% against a **12.5% chance baseline**. They are coin flips
inside an argmax, and every coin flip is a chance to displace a family the
model can actually rank.

So: don't improve the model — **shrink the menu**.

This matters because three prior attempts to improve the model all failed.
EXP-138 reshaped the simulated move distribution (hit rate 24.7% → 24.5%). A
stage-0 check eliminated Black-Scholes (1–2 cents against realized inputs).
EXP-139 conditioned the residual pool on market cap and subtracted a causal
per-family bias (25.2% and 24.3%, and **every** adjustment captured *less* of
the random-to-oracle spread than doing nothing).

## 2. The menu chose itself identically in and out of sample

Ranked by how often each family was the realized best:

```
2018-2022 (training only):  TWIN-P, TWIN-P5, wide-butterfly-5, condor, butterfly
full sample:                TWIN-P, TWIN-P5, wide-butterfly-5, condor, butterfly
```

Identical. `stable_menu` passes outright, which is the least interesting-looking
result here and arguably the most important: the thing being selected is stable,
so the selection is not the fragile step.

## 3. Holdout 2023–2026, top-20% gate

| | n | lift | Sharpe | win | mean | median | on capital | per collateral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| menu k=3 | 493 | 1.14 | 2.10 | 51.3% | +21.8% | +11.6% | +5.0% | 0.041% |
| menu k=4 | 485 | 1.34 | 3.17 | 57.1% | +31.6% | +40.0% | +17.1% | 0.143% |
| **menu k=5** | 490 | 1.56 | **3.22** | 56.9% | **+32.3%** | +40.5% | **+17.1%** | **0.145%** |
| menu k=6 | 489 | 1.69 | 3.22 | 57.1% | +32.0% | +41.0% | +16.1% | 0.137% |
| chooser, all 8 | 487 | 1.93 | 2.85 | 55.0% | +28.9% | +37.1% | +14.1% | 0.107% |
| always condor | 485 | — | **3.26** | **59.4%** | +28.3% | **+44.1%** | +12.0% | 0.122% |
| always TWIN-P | 472 | — | 2.36 | 57.8% | +20.0% | +32.6% | +13.4% | 0.090% |

At the top-30% gate the pattern repeats: k=5 Sharpe 3.65 against all-eight's
3.35, condor 3.80.

**Restricting the menu is worth +0.37 of Sharpe and +35% on profit per dollar
of collateral, out of sample.** That is more than EXP-138 and EXP-139 produced
combined, and it required no change to the model at all.

**It is a plateau, not a spike.** k=4, 5 and 6 sit within 0.06 of Sharpe; k=3
collapses to 2.10 and k=8 is the worst of the five. A result that only existed
at k=5 would be a fitted point; this one has a shape.

## 4. What it is not

**The chooser did not get better at ranking.** Lift over chance FALLS as the
menu shrinks — 1.93× at k=8 to 1.56× at k=5. The raw hit rate rises from 24.7%
to 31.9% only because chance rises from 12.5% to 20.0%. The gain is economic,
not skill: the chooser stopped being offered choices it could not make, and the
ones removed were worth least. Anyone extending the menu later should expect to
lose it again.

**And it still does not beat a single condor.** `beats_the_condor` FAILS:
Sharpe 3.22 against 3.26 at top-20%, 3.65 against 3.80 at top-30%, plus a worse
win rate and median. The same split that has held at every gate all programme:
**the chooser earns more per dollar, the condor earns it more smoothly.**

| | menu k=5 | always condor |
|---|---|---|
| better on | mean, return on capital, collateral efficiency | Sharpe, win rate, median |
| complexity | 8 families priced, argmax over 5, per-event width | one structure, per-event width |

**The holdout is thin.** 490 trades over four years in one regime. A 0.37
Sharpe difference is not separable from noise on that sample, and the spec said
so before the run.

## 5. Where this leaves the programme

Read together with EXP-137 and EXP-139, the ordering of what actually moves the
number is now clear, and it is the reverse of where the effort went:

| intervention | effect on the chooser |
|---|---|
| **shrink the menu to 4-6 families** | **+0.37 Sharpe, +35% collateral efficiency** |
| condition the pool on market cap | +0.5pp hit rate, worse capture |
| reshape the move distribution | −0.2pp hit rate |
| de-bias the expectation per family | −0.4pp hit rate |

And the collateral finding from EXP-137 still dominates all of it: a condor is
**2 short puts at a median $24,400** against a twin peak's **4 at $48,800**, so
under the cash-secured constraint a condor is 1.6–2× more capital efficient per
dollar posted before any ranking happens.

## 6. Recommendation

If the chooser runs, run it on these five. But on this evidence a single
four-strike condor with a per-event width remains the better risk-adjusted
book, needs no search, carries no max-statistic exposure, and posts half the
collateral. Nothing here is promotable: the holdout is four years, the gain is
inside noise, and there are no forward prints.
