# Plan — is `abs_move` correct, and can the panel's universe come from ORATS?

**Version:** 2.0 · **Date:** 2026-09-01 · **Owner:** YS + Claude
**Status:** plan only. Nothing registered. Stage 0 must close first — see §3.
**v2.0:** v1.0 made "reproduces oquants" the acceptance bar. That is circular —
it assumes the number we are trying to validate. Replaced with triangulation
against independent sources, which also changes what the experiment is *for*.

---

## 1. What this is really about

The starting question was narrow: the panel's event universe is bounded by
oquants (2,936 tickers), which leaves 53 board tickers permanently unscoreable,
so can it come from ORATS instead?

Answering it forced a bigger one. oquants supplies `realized_moves` and
`abs_realized_moves` **directly**, and `abs_realized_moves` becomes `abs_move` —
the size champion's training target, the STR-THRU payoff driver, and the
quantity every experiment from EXP-105 to EXP-116 is measured against. ORATS
`hist/earnings` supplies dates and `anncTod` only. So moving the universe means
*computing* that target rather than being handed it.

Which surfaces the question nobody has asked: **is the number we have been
training on correct?** It has never been checked against anything. It is a
vendor-supplied derived value, accepted as ground truth since the panel was
built.

That question is worth answering **whether or not the universe ever moves**, and
this plan is now ordered accordingly: validate the target first, decide about the
universe second.

## 2. Why "reproduce oquants" is the wrong bar

A first naive reproduction (event-date close → next close, one rule for all
events) gave correlation **0.7489**, with a median of **2.78** against oquants'
**3.93** — about 30% low. Session handling is one known cause: a BMO print runs
from the *prior* close to the event-date close, and that version used one rule.

The tempting next step is to fix the session logic and keep tightening until the
numbers match. That is exactly the wrong instinct. **A perfect reproduction of a
wrong number is a wrong number**, and it would ship with the false confidence of
a passed test. oquants' value has no more claim on being right than a computed
one does; it is simply the one we happened to start with.

## 3. Stage 0 — establish what each source measures

Desk work. No quota. Nothing else in this plan is meaningful until it closes.

1. **Define the quantity precisely.** "Realized earnings move" is which two
   prices, on which two dates, under which session rule, adjusted for what.
   Write it down. Every later stage measures against this definition, not
   against a vendor.
2. **What does oquants' `realized_move` measure?** Establish from the payloads
   and `earnings_predictions/HANDOFF.md`. If it cannot be established, that is a
   finding: an unspecified target is a poor thing to have trained on, and it
   raises the value of owning the definition.
3. **What is `daily_market.spot`?** Which close, split/dividend adjusted or not,
   and whether that differs by era. `src_spot` records provenance; the panel
   already carries a three-era mcap conversion, so era-dependence has precedent.
4. **What is `option_chains.stockPrice` / `spotPrice`?** Same questions. This is
   a *different ORATS pipeline* from summaries and therefore a partial
   independent check even before another vendor is involved.
5. **Session availability.** 227,456 of 228,223 events (99.7%) carry a session.
   The rest have no well-defined move and must be excluded, not guessed.
6. **Corporate actions and gaps.** Splits inside the two-day window; events with
   no next close (halt, delisting, IPO). Decide the rule before seeing its
   effect on the number.

**Exit condition:** a written definition, and a session-aware computation whose
disagreement with oquants is *explained*, not merely small.

## 4. Stage 1 — triangulate the target against independent sources

The acceptance bar is **agreement among independent measurements**, not
agreement with oquants. oquants becomes one input under test, not the referee.

Sources, in descending order of independence:

| source | on disk today | independence |
|---|---|---|
| **Polygon daily aggregates** | **no — option aggregates only** (8,327 responses are `O:...` contracts) | **different vendor** |
| `option_chains.stockPrice` | yes, 15.3M rows | different ORATS pipeline |
| `daily_market.spot` | yes, 8.99M rows | ORATS summaries |
| oquants `realized_moves` | yes, 2,944 files | the incumbent, under test |

**Polygon underlying bars must be fetched.** `v2/aggs/ticker/<T>/range/...`
returns a full date range in one call, so it is one call per ticker — but
Polygon is ~10 req/min on this plan with a 6.5s pacing gate, so ~200 tickers is
roughly 25 minutes of wall clock. Scope it to a validation sample, not the whole
universe: a few thousand events across all years and market-cap buckets is
enough to settle a definitional question, and a stratified sample is more
informative than a big convenience sample.

**How disagreement is adjudicated:**

- where **≥2 independent sources agree** and oquants differs → oquants is wrong
  on that event, and the size of that population is the headline finding;
- where the independent sources disagree with *each other* → the definition is
  ambiguous, not the data. Return to Stage 0;
- where all agree → the target is sound, and the universe question can proceed
  on its own merits.

**Report the disagreement rate, not just a pass/fail.** "oquants differs from
consensus on N% of events, concentrated in \<year / mcap bucket / session\>" is
the useful output whichever way it lands.

## 5. Stage 2 — the Greeks consistency check on the chain

Distinct instrument, distinct purpose. Backing an underlying move out of option
prices requires assuming a pricing model, so it is a poor *measurement* of the
stock move. It is a good **detector of bad quotes**.

For each event with a chain on both sides, check the observed option price
change against what delta, gamma, theta and vega imply for the observed move.
A straddle whose price change cannot be reconciled with any plausible move has a
bad quote on one side.

This has an immediate, independent payoff: the live board currently shows

```
CBAT  STR-THRU  cost% 166.666667  exp_pnl_model -0.953  win 0.0
```

A straddle priced at **166% of spot** is not a real quote, `WIDE_MARKET` did not
catch it, and it is driving a confident-looking −95% expected P&L. Whatever
happens to the universe question, that filter is worth having.

## 6. Stage 3 — only now, the universe

Meaningful only if Stage 1 establishes a trustworthy target.

- How many additional tickers reach **≥12 prior prints** — the count that
  decides scoreability, not the raw ticker count.
- Of the 53 currently-unreachable board tickers, how many clear it.
- Effect on panel rows and on `MISSING_FEATURES` for a live board.

The honest prior is that this is modest: many new names are recent listings, and
22 of the 35 in-universe blocked tickers already fail on print count alone.

## 7. Stage 4 — what it does to the champions

- Retrain `size_v1_4`; compare r / MAE / RMSE **on the original 2,936-ticker
  subset**, which is the like-for-like sample. Comparing on the extended panel
  measures the new rows, not the change. This confound has bitten the programme
  twice (EXP-110's probe, EXP-116's premise) and must not a third time.
- Report extended-panel metrics separately, never beside the incumbent's.
- Re-run gate invariance and `registry_current`: more training rows change the
  residual pools EXP-115 conditions on, and the recalibration map.

## 8. Promotion rule

Adopt only if **all** hold:

1. Stage 1 shows the computed target agrees with independent consensus, with the
   oquants disagreement rate quantified and explained;
2. Stage 3 shows a gain in scoreable rows that clears a threshold set **before**
   it is measured;
3. Stage 4 shows no degradation on the like-for-like subset;
4. The ledger records the commit at which the target's provenance changed, so
   later readers know which side of that boundary a result sits on.

Point 4 is not paperwork: without it, results either side are silently
incomparable and nothing says so.

## 9. Abandon criteria

- Independent sources disagree with **each other** and Stage 0 cannot resolve
  why — the definition is the problem, and no source choice fixes it.
- The computed target agrees with consensus but Stage 3 yields only a handful of
  extra scoreable rows — then the target validation was worth doing and the
  universe change was not. **Ship the first, drop the second.**
- Polygon coverage of the relevant names turns out to be as thin as oquants' —
  check this early, on a sample, before fetching at scale.

## 10. Cost

| | |
|---|---|
| ORATS quota | **0** — every ORATS input is already on disk |
| Polygon | ~1 call per ticker; ~25 min wall clock for a 200-ticker sample at the 6.5s gate |
| Stage 0 | desk work |
| Stages 1–2 | panel-scale comparisons, minutes |
| Stages 3–4 | panel rebuild (~10 min) + size retrain (~9 min) |

The scarce resource is care, not calls.

## 11. What this plan does not claim

- **It does not claim oquants is wrong.** It claims nobody has checked, and that
  the check is cheap relative to what depends on the answer.
- **It does not claim the switch is a good idea.** It specifies what would settle
  that, and §9 includes the outcome where the answer is no.
- **It does not claim ORATS coverage is better.** ORATS carries 6,025 tickers in
  `daily_market` against oquants' 4,205-name scan, but 25 board symbols returned
  404 from ORATS (`reports/orats_unknown_symbols.json`), some of which look like
  symbology mismatches. Neither source is a superset.
- **It does not treat the 0.7489 correlation as evidence against anything.** That
  came from a computation that ignored session. It shows the reproduction is
  non-trivial, nothing more.
- **It does not assume Polygon is right either.** Polygon is a second opinion,
  not an oracle. The bar is *consensus among independent measurements*, and a
  2-of-3 disagreement is a finding to investigate, not a verdict to apply.
