# Plan — can the gate predict trades on the small-name universe, and does a tighter threshold help?

**Version:** 1.0 · **Date:** 2026-09-02 · **Owner:** YS + Claude
**Status:** plan only. Nothing scaffolded or registered. §4 is pre-registration:
its numbers are frozen before any model runs.
**Experiment id (when scaffolded):** EXP-119.

---

## 1. What this is really about

EXP-118 retrained both mid-fill gates on the expanded replay universe and they
did not clear the champions: STR-THRU gated mean fell from +7.3%/trade to
+4.3%, STR-RUNUP from +2.2% to ~0 (OOS 2020-26). The champions stand.

The expansion is not uniformly bad, though. Measured 2026-09-02, the added
slice — names with market cap below $1B plus the computed-moves names oquants
does not carry — has the HIGHEST ungated base return in the STR-THRU book:

| STR-THRU bucket | n | base mean/trade | win | median premium |
|---|---|---|---|---|
| >10B | 8,626 | +1.4% | 38.5% | 5.1% of spot |
| 1-10B | 9,948 | +3.8% | 38.9% | 8.9% |
| <1B | 616 | **+10.4%** | 43.7% | 18.3% |

So the question splits in two, and this plan answers both:

1. **Can a gate make money on the small-name slice by itself?** Train and test
   on the slice alone. If yes, the slice deserves its own gate rather than the
   champion's or none.
2. **Does selecting fewer trades raise per-trade returns?** Evaluate a ladder
   of gate thresholds (top 10% / 20% / 30%) on identical out-of-sample scores,
   pre-registered, no picking the winner after looking.

The trading decision hangs on this: whether the `OUT_OF_DOMAIN` guard comes
off, whether the champions refresh on the full universe, or whether a dedicated
small-name gate gets built.

## 2. Universe definition (exact, measurable)

The **new universe** = replayed mid-fill trades (fill_alpha = 0.5, provenance
`engine.replay`) whose ticker satisfies EITHER:

- `bucket_mcap(exp(mcap_log)) == "<1B"` (mcap below $1e9, per
  `engine.data.coverage.MCAP_BUCKETS`), OR
- the ticker is one of the computed-moves names in `data/raw/computed_moves/`
  (the EXP-117 universe; 27 names).

Measured membership in the current gate datasets (2026-09-02):

| strategy | <1B | computed-moves | overlap | unique slice rows |
|---|---|---|---|---|
| STR-THRU | 616 | 37 | 34 | **619** |
| STR-RUNUP | 266 | 6 | 6 | **266** |

## 3. The data is thin, and the plan says so up front

Slice rows by year (STR-THRU / STR-RUNUP):

| year | THRU n | THRU base | RUNUP n | RUNUP base |
|---|---|---|---|---|
| 2018 | 6 | -13.2% | 10 | +12.9% |
| 2019 | 21 | -9.5% | 14 | +4.3% |
| 2020 | 11 | -1.1% | 10 | +19.4% |
| 2021 | 16 | -2.7% | 11 | -9.9% |
| 2022 | 34 | +5.7% | 24 | -6.5% |
| 2023 | 79 | +17.3% | 20 | +5.1% |
| 2024 | 124 | **+35.4%** | 26 | +12.2% |
| 2025 | 243 | -0.2% | 135 | -0.9% |
| 2026 | 85 | +14.0% | 16 | +15.4% |

Consequences the design must absorb:

- The slice only exists at scale from 2024 (the pulls that created it ran
  2026-08/09 and backfilled). Walk-forward evidence is structurally limited to
  the last ~2-3 years. This cannot be fixed by cleverness; it is the cost of
  the universe being new.
- The engine default `min_train_rows=500` makes RUNUP untestable at all (266
  rows total) and THRU testable only in 2025-26. **Arm A therefore pre-
  registers `min_train_rows=100`**, which opens THRU testing 2024-26 (452 OOS
  rows) and RUNUP testing 2025-26 (151 OOS rows). The relaxation is part of
  the spec, declared here, not a post-hoc salvage.
- At top-10% selection, THRU passes ~45 pooled OOS trades and RUNUP ~15.
  Standard errors at that n are ~4-6pp; **effects smaller than ~10pp are below
  the resolution of this dataset**, and the success criteria in §4 say so
  rather than squinting.
- 2024's +35.4% THRU base warns of fat tails: the run reports the single
  largest trade's contribution to every gated mean it prints. A gated mean
  carried by one trade is not a strategy.

## 4. Pre-registered protocol

All four arms use the champion gate's exact machinery — `gate_mod.FEATURES`,
`gate_mod.fit` (HistGBM, unchanged hyperparameters), walk-forward, threshold =
quantile of scored predictions (the `choose_threshold` rule) — varying only
what this section says. Nothing else moves.

### Arms

| arm | strategy | train set | test set | status |
|---|---|---|---|---|
| A-thru | STR-THRU | slice only | slice, 2024-26 | **primary** |
| A-runup | STR-RUNUP | slice only | slice, 2025-26 | secondary (exploratory power) |
| B-thru | STR-THRU | full universe (EXP-118 protocol) | slice rows only, 2020-26 | secondary |
| B-runup | STR-RUNUP | full universe | slice rows only, 2020-26 | secondary |

Arm A answers "can the model learn small names from small names". Arm B answers
"can a model trained on everything already pick within small names" — the
version relevant to keeping one universal gate. Both arms produce OOS scores;
the threshold ladder then applies to each arm's own scores.

### Threshold ladder

For each arm, cutoffs **top 10%, 20%, 30%** applied to that arm's pooled OOS
predictions (production rule `choose_threshold`, top fraction as varied). All
three thresholds are read off ONE walk-forward run per arm — same scores, three
cutoffs — so comparing them is not multiple fitting. The headline threshold is
20% (the production rule); 10% and 30% are secondary by label, not by looking.
A finer curve may be plotted for diagnostics but is never judged.

### Baselines (reported, not tested)

- Ungated slice mean (the +10.4% / +2.5% numbers above).
- Champion registry evals, for context only — they were measured on a
  different row set and are never placed in the same comparison column.

### Success criteria

Per arm, at the headline 20% cutoff, with 10,000 seeded bootstrap resamples of
the pooled OOS gated trades:

- **PASS** — gated mean > 0 AND bootstrap 95% CI excludes zero AND the CI
  half-width is <= 10pp (else the dataset cannot resolve the effect).
- **INCONCLUSIVE** — CI includes zero or is wider than 10pp. This is the
  expected outcome at these n's; it is a finding, not a failure to re-run away.
- **FAIL** — CI excludes zero from below.

Secondary, pre-registered shape test ("tighter is better"):
`gated_mean(10%) > gated_mean(20%) > gated_mean(30%)`, reported with CIs.
Monotone = the gate's ranking carries information at the top of the slice.
Non-monotone = the ranking is noise up there, and no cutoff rescues the arm.

Also reported per arm: n_passed per threshold, gated win rate, per-year gated
means (display only — no per-year selection), and the max-single-trade
contribution to each gated mean.

## 5. Decision rule — what each outcome does

Chosen now, before results, so the results cannot choose it:

1. **A-thru PASS at some cutoff** → build a dedicated small-name gate for
   STR-THRU (registered per (strategy, role, universe-slice)); the slice
   trades under its own gate, the `OUT_OF_DOMAIN` guard is replaced by it.
   RUNUP slice stays guarded unless A-runup also passes.
2. **A fails/inconclusive but B-thru PASS** → keep one gate, refresh the
   champion on the full universe (`train_all --role gate`, which re-measures
   and re-registers), guard comes off because the decision rule is then
   measured on the universe it trades.
3. **Neither passes** → the slice is not tradeable on gate decisions. Guard
   stays. The exposure (+10% ungated base) is real but unselectable with
   current features — the finding is recorded as a constraint on the search,
   and the route back in is features the slice actually needs (spread, quote
   quality, liquidity), not more threshold tuning.

No outcome triggers "try more cutoffs until one passes" — that is the
multiple-testing trap the ledger exists to prevent, and a new experiment with a
new pre-registration is the only legitimate next move.

## 6. Execution outline

1. Scaffold: `python3 experiments/new_experiment.py --title "gate on the
   small-name universe and the threshold ladder" --strategy STR-THRU` → EXP-119,
   spec.yaml stamped before any run (the enforceable pre-registration; this
   document is the design rationale).
2. `run.py` mirrors EXP-118's data path exactly (`_engine_trades` filter →
   `gate_mod.build_dataset(trades, panel=panel)`), adds the §2 slice mask and
   the `min_train_rows=100` relaxation for arm A, loops the threshold ladder
   over each arm's scored frame, bootstraps CIs, writes results/metrics.json +
   REPORT.md. No engine changes; the relaxation is a parameter, not a patch.
3. Runtime estimate: ~10-15 min local compute, zero API quota (all data is in
   Tier 2 already).
4. Results land in the ledger (PLANNED → RAN rows) and REPORT.md; the private
   mirror sync carries them.

## 7. Cost

No ORATS/Polygon spend — the trades table and features already exist. Compute
only. The expensive resource is the slice's own thinness, priced in §4.

## 8. What this plan does not claim

- It does not claim the slice's +10% ungated base survives costs beyond the
  mid-fill convention; the usual fill-sensitivity applies and is worse here
  (median premium 18-21% of spot means wide dollar spreads even at small
  percentages).
- It does not claim walk-forward on 2-3 years generalizes to a regime the
  slice has not lived through (a 2022-style vol shock hits microcaps first).
- It does not claim threshold tuning can manufacture signal the features do
  not contain — §5 outcome 3 exists precisely for that case.
