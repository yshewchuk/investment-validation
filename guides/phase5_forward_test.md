# Phase 5 Guide — Forward Test, Fill-Quality Measurement, Go-Live

**Objective:** measure α̂ — the fill quality actually achievable with resting
limit orders — because the entire positive verdict rests on the mid-fill
assumption. Everything here exists to turn that assumption into a number
before real capital scales.

---

## 1. Architecture

```
engine/forward/
  paper.py         # opens paper trades from gate triggers (nightly job step)
  fill_verify.py   # did the market trade through our limit?
  alpha_hat.py     # fill-quality estimation per liquidity bucket
  weekly_report.py # generator-format weekly review
  golive.py        # mechanical go/no-go check against preregistered rules
ledger/paper/YYYY-MM-DD.jsonl        # paper trade records (append-only, Phase 4 writer)
config/golive_rules.yaml             # the plan's rules, encoded — frozen now
```

### paper.py
Runs inside the Phase 3 nightly job after scoring: for each gate-passing
candidate (respecting per-strategy caps and the 5% sizing), record a paper
trade: decision_ts, structure legs, **intended limit = MID at decision time**
for each leg, size, and the full ScoreResult (so expectation vs outcome is
always joinable). Paper trades go through the Phase 4 append-only writer —
logged before outcomes, no retro edits, corrections via `supersedes`.

### fill_verify.py — the heart of the phase
For each paper leg, over the holding window, determine whether the limit
would plausibly have filled:
- **ORATS EOD path (all names):** on subsequent trade dates, compare the
  quoted bid/ask to our limit. Classification: `filled` (market crossed the
  limit: for a buy, ask ≤ limit at some EOD; for a sell, bid ≥ limit),
  `likely` (limit inside the quoted spread), `unfilled` (never touched).
  EOD-only granularity is a known limitation — state it in every output.
- **Polygon bars path (2024-08+ where option aggs exist):** intraday-daily
  high/low of the contract vs the limit — a strictly better signal; use it
  to VALIDATE the EOD classifier (agreement rate reported), budgeted within
  the polygon 10 req/min gate and single-process lock.
- Output per leg: classification, achievable price (best price our side
  could have gotten under the classification), days-to-fill.

### alpha_hat.py
α̂ = where the achievable price sits between worst and mid-or-better, i.e.
`(worst_side_price − achievable) / (worst_side_price − mid)` clipped to
[0, 1.5], aggregated per bucket: spread-width quartile × mcap band ×
strategy. Report mean, p25, and the fraction `unfilled` (an unfilled entry
is not a free pass — it's a missed trade; track missed-trade opportunity
cost separately). **α̂ (p25, not the mean) replaces the 0.5 assumption** in
Phase 1 scoring and Phase 2 evaluation defaults once ≥100 verified legs per
bucket exist — conservative by construction.

### weekly_report.py
Generator format, every Friday of an active season: paper P&L vs
model-expected with CIs (per strategy), fill-quality distribution + α̂ table,
gate trigger log with hindsight outcomes, missed-trade log, MC-band overlay
(is the season inside the backtest's Monte Carlo envelope?).

### golive.py
Reads `config/golive_rules.yaml` (encode the plan's §P5 rules verbatim: full
season of paper data; α̂_p25 ≥ breakeven_alpha + 0.1 margin per strategy;
CAL-P blocked until EXP-101/102 promoted; start sizing 5%; escalation and
drawdown-stop rules) and emits a go/no-go memo — every rule PASS/FAIL with
the number that decided it. The memo is generated, not written by hand, so
the go-live decision can't be quietly renegotiated in the moment. Changing
the rules file after the season starts requires a dated note in the memo.

## 2. Constraints

- Paper trades must be indistinguishable in schema from future live trades
  (`kind` field only) — the Tier-2 `trades` table is the single home.
- decision_ts discipline: a paper trade's inputs are the nightly snapshot's;
  fill verification only uses data dated AFTER decision_ts (leak auditor
  wired here too).
- Never mark a leg `filled` on the entry side and `unfilled` on the exit
  side silently — an entered-but-not-exited paper structure is carried and
  marked to EOD mids with a `STUCK` flag; stuck rates are a headline stat
  (they are what real trading would feel like).
- Q3 2026 season (mid-Oct → mid-Nov) is the proving window; the pipeline
  must be running dry by end of September (Phase 3 exit).

## 3. Acceptance tests (`checks/phase5_checks.py`)

1. **Historical dry-run:** replay a past earnings week (as_of overrides):
   paper trades open, verify, close, and the weekly report renders — end to
   end with zero manual steps.
2. **Synthetic fill truth:** construct quote series with known crossings →
   classifier reproduces filled/likely/unfilled exactly; α̂ on synthetic
   data with a planted true alpha recovers it within tolerance.
3. **EOD-vs-bars validation:** on the 2024-08+ overlap, EOD classifier
   agreement with Polygon bars ≥ target (report the number; if <80%, the
   EOD classifier's `likely` class must be treated as `unfilled` in α̂ —
   encode that fallback).
4. **Stuck handling:** a synthetic never-fillable exit → STUCK flag, carried
   marks, appears in the weekly report.
5. **Go/no-go mechanics:** synthetic season data passing/failing each rule →
   memo flips correctly rule by rule.
6. **Append-only:** same guarantees as the Phase 4 ledger (shared writer).

## 4. Failure modes

- Thin buckets (few triggers in a slice) → α̂ falls back to the pooled
  strategy level with a WIDENED flag; never invent per-bucket precision.
- The season is unusually quiet/volatile → MC-band overlay says whether the
  sample is informative; the go/no-go memo can conclude "extend one more
  season" — that outcome is a valid result, not a failure.
- Broker integration (real fills) is deliberately out of scope until after
  go-live; leave a `source: broker` enum stub in the trades schema.
