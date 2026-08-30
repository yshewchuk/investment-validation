# Phase 4 Implementation Guide — Completing the Verification & Reporting Layer

> **BUILT — 2026-08-30.** Every part of this guide is implemented and
> `python3 checks/phase4_checks.py` is green (17/17). The evidence report is
> `reports/phase4_reporting.md`; what remains open is listed there (chiefly:
> the ledger has no forward rows until the Sep-1 pull refreshes the calendar).
> The guide is kept as written — it is the specification the build was measured
> against.

**Companion to** `phase4_verification_reporting.md` (the *what*, written before
any of it existed). This guide is the *how*, written after the first three
phases shipped and against the reports they actually produced. Where the two
disagree, this one wins — it is the later document and it cites real output.

**Status when this guide was written (2026-08-30, commit `977568e`):** Phases
0, 1 and 2 are complete. Phase 4 is **~55% built** — the generator and the leak
auditor exist and are wired into `evaluate()`; the prediction ledger does not
exist at all; there is no `checks/phase4_checks.py`; and the reports the
generator emits are *complete but not legible* — the defect this guide spends
most of its length on.

---

## 0. Sequencing — confirmed

The plan's own sequencing table (`EARNINGS_VOL_PROGRAM_PLAN.md` §Sequencing)
orders the build **0 → 1 → 2 → 4 → 3 → 5 → 6**: report generator is order 4
("built alongside Phase 2, its output format"), dashboard is order 5 ("Phase 3
dashboard — depends on Phase 1 scorer + Phase 4 ledger"). The phases are
numbered in the order they were *specified*, not the order they are built.
Phase 3 was moved after Phase 4 because the dashboard's nightly job is the
thing that *writes* the ledger, and the ledger's schema is a Phase 4
deliverable. `guides/README.md` carries the same dependency ordering.

**So: Phase 4 is the next step, and it is a completion pass, not a new build.**

### What exists

| Deliverable | State | Evidence |
|---|---|---|
| §1 Report generator `engine/report.py` | **Built** (875 lines): provenance block, 7 auto-evaluated checklist items, 7 figure functions, fixed section order, promotion-report variant | 23 tests in `tests/test_report.py`; every EXP-10x report |
| §1 HTML output | Not built (markdown + PNG is the contract; optional) | — |
| §2 Prediction ledger `ledger/` | **Not built.** Directory is empty; `engine/paths.py:115` defines `LEDGER` and nothing writes to it; `checks/phase1_report.py:332` records it as *"DEFERRED — Phase 4"* | — |
| §3 Leak auditor `engine/audit.py` | **Built** (`assert_causal`, `assert_decision_causal`, `audit_frame`, `FeatureVector` stamps), wired into score and evaluate | `tests/test_audit.py` |
| §3 Audit **receipt** | **Not built.** The checklist's "Leak audit ran" evidence is a fold-year summary string, not a receipt with counts and max margin | EXP-105 §7: *"fits saw max year per fold: [2019, …]"* |
| §4 "the generator is the ONLY way phases emit results" | **Violated in 4 places** — see Part D | `checks/phase0_audit.py:130`, `checks/phase1_report.py:357`, `EXP-105/run.py:53`, `EXP-102/run.py:61` |
| §5 `checks/phase4_checks.py` | **Not built.** `checks/` has phase0, phase1, phase2 only | — |

### What is wrong with what exists

The generator emits everything the plan asked for and is *hard to read*. That
is not cosmetic: an unreadable report fails the plan's own stated purpose
("rich, auditable reports", "reports are the interface", `guides/README.md`
convention 9). Concretely, from `experiments/EXP-105_str_thru_validation_registered_mid_fill/REPORT.md`:

- The headline says `mean/trade 0.0404`. Nowhere does it say that is **+4.04%**.
  Three lines later the appended appendix writes the same number as `+4.04%`.
  Two conventions in one document.
- §6 Calibration is a **raw `json.dumps` blob**, 40 lines of it
  (`engine/report.py:815`). Its `brier_skill` is **−0.084** — the shipped win
  rates are *worse than always predicting the base rate* — and no sentence in
  the report says so. All seven checklist items still render **PASS**.
- §4 Monte Carlo reports `terminal p50 341,870×` at 5% sizing with
  `P(loss) 0.0000`, while the program plan's headline for the same strategy is
  `2.83× / P(loss) 15%`. The reconciliation exists (concurrency: MC ignores
  overlap) but is a prose caveat *below* the table, and the two sets of numbers
  are never printed side by side.
- `Stale dates (1% mis-dated): Δmean — on 76 events` — the `—` is a NaN. The
  stress ran with **0.1% chain coverage** (`Slippage -1d: coverage 0.0010`) and
  produced nothing, but renders as though it produced a result.
- `Tail injection: N/A — no short leg; tail injection N/A` — the note is
  concatenated with itself.
- The appendix ends with a **6-line raw JSON dict** of the by-year sweep
  (`engine/report.py:860`).
- Every figure's caption **collides with the x-axis label** (`figtext` at
  y=0.005 with no reserved margin). The equity curve plots a 25,000× series on
  a **linear axis**, so nine years of the curve are a flat line at zero; the MC
  fan tops out at `1e16` and shows p05/p50 as invisible.
- The report never states, in one place, **which universe each n belongs to**:
  §1 says `n=7620`, the anti-selection line says `n=17666`, the appendix says
  `n=11080`, and EXP-102's appended appendix reveals `543 events dropped`
  before any of them.
- There is **no verdict**. Nothing in the generated body says whether the
  experiment supported its hypothesis.

Part A fixes all of that. It is the largest piece of Phase 4 and the reason
this guide exists.

---

## 1. Build order

1. **A0–A2** — formatting contract, verdict header, glossary. Cheapest, biggest
   legibility win, and everything after builds on the formatters.
2. **A3–A6** — sample funnel, calibration honesty, dead-stress handling, MC
   reconciliation. These change *what the report says*, not only how it looks.
3. **A7** — figures.
4. **A8–A10** — extra-sections API (kills the `run.py` append hack), checklist
   upgrade, regeneration of the three existing EXP reports as the diff test.
5. **B** — prediction ledger + outcome scorer + calibration trigger.
6. **C** — audit receipts.
7. **D** — retire the bespoke writers.
8. **`checks/phase4_checks.py`** green; regenerate every report; commit.

B is on the **calendar critical path** — Q3 season starts mid-October and the
ledger has to be accruing rows before it, so if time gets tight, do A0–A6, then
all of B, then come back for A7–D.

---

## Part A — make the reports readable

The governing rule for this whole part:

> **Every number in a report answers three questions without leaving the page:
> what unit is it in, which sample is it computed on, and what would make it
> wrong.** A number that cannot answer all three is a bug in the generator.

Nothing in Part A may change a computed value. The acceptance test for the
whole part (A10) is that regenerating EXP-102/105/107 produces *identical
numbers* and a different document.

### A0. One formatting contract (`engine/report.py`, replace `_fmt` at :429)

```python
def pct(x, nd=2, signed=True)   -> "+4.04%"   | "—"      # ratios that are returns
def prob(x, nd=1)               -> "39.8%"    | "—"      # probabilities / win rates
def ratio(x, nd=2)              -> "1.24×"    | "—"      # multiples, profit factor
def num(x, nd=2)                -> "2.23"     | "—"      # unitless stats (Sharpe)
def count(n)                    -> "7,620"                # thousands separator always
def money_x(x)                  -> "341,871×" | ">1e6×"   # terminal equity, clamped
```

Rules:
- **Percent for anything that is a return, a rate, or a fraction.** `mean`,
  `median`, `std`, `win_rate`, `max_dd`, `p_loss`, `coverage`, `delta_mean`,
  spreads, ITM rates. No naked `0.0404` anywhere in a report body.
- **Signed** for returns (`+4.04%` / `−16.03%`), unsigned for rates.
- `—` only ever means *not available*; it must never be produced by a NaN that
  the caller thought was a value (see A5).
- Fixed decimals per formatter so golden-file comparison stays byte-stable.
- The canonical metrics table keeps its current key order but renders each key
  through its declared formatter, driven by one dict:

```python
METRIC_SPEC = {                       # key -> (formatter, label, one-line definition)
  "mean":          (pct,   "mean/trade",   "average return per trade on the trade's own premium"),
  "win_rate":      (prob,  "win rate",     "share of trades with return > 0"),
  "profit_factor": (ratio, "profit factor","gross wins / gross losses"),
  "sharpe_trade":  (num,   "Sharpe (trade)","mean/std × √(trades per year)"),
  "sharpe_equity": (num,   "Sharpe (equity)","daily equity-curve Sharpe, ×√252, at 5% sizing"),
  "sortino":       (num,   "Sortino",      "mean / downside deviation vs 0"),
  "max_dd":        (pct,   "max drawdown", "worst peak-to-trough on the 5%-sized equity curve"),
  "tail_ratio":    (ratio, "tail ratio",   "|p95 win| / |p95 loss|"),
  ...
}
```

`METRIC_SPEC` is the single source for labels, units **and** the glossary (A2),
so a metric can never appear in a table without a definition existing.

### A1. Section 0 — the verdict block (new, renders first)

Every report opens with a generated, rule-derived verdict. **No prose
invention**: each line is a template filled from the results dict, and the
overall call comes from a deterministic function.

```
## 0. Verdict — read this first

**SUPPORTED (with one warning).** The pre-registered hypothesis said the gate
lifts OOS mean materially above base; it did: +4.04% vs +2.70% ungated.

| Question | Answer | Where |
|---|---|---|
| Does it make money at mid fills? | **Yes** — +4.04%/trade over 7,620 OOS trades (2018–2026) | §1 |
| How much fill quality does it need? | Breakeven at **α = 0.42** — needs 42% of the spread; mid is 50%. Margin: 8 points | §1 |
| Is it positive every year? | **9 of 9** OOS years positive, weakest 2020 (+0.08%) | §3 |
| Does it survive the stress battery? | **Partly** — 4/4 crisis regimes positive; slippage and stale-date stresses INCONCLUSIVE (0.1% chain coverage) | §5 |
| Are the win rates trustworthy? | **No** — Brier skill −0.084, worse than the base rate. Treat win_rate as a ranking, not a probability | §6 |
| What sizing does the evidence support? | Undetermined here — MC ignores the 133-position overlap; see the deployment block | §4 |
| What would falsify it? | Live fill quality below α = 0.42, or the 2023–2024 concentration not repeating | §5, §3 |

**Warnings:** win-rate calibration below base rate · 2 of 5 stress stages inconclusive.
```

Implementation notes:
- `verdict(results, spec) -> ("SUPPORTED"|"NOT SUPPORTED"|"MIXED"|"DESCRIPTIVE", reasons[], warnings[])`.
  `DESCRIPTIVE` is for specs with `promotion_target: null` and a measurement
  hypothesis (EXP-102) — those get "what was measured" rows instead of a
  pass/fail.
- The hypothesis's falsification clause is already written in `spec.yaml`
  (`Falsified if …`); echo it verbatim in the last row rather than paraphrasing.
- **Warnings are computed, not passed in.** Minimum warning set: negative Brier
  skill; any stress stage with coverage < 5%; MC concurrency > 1 while sizing
  numbers are shown; any headline year with n < 30; breakeven alpha > 0.45;
  checklist FAIL (which also keeps the existing red banner).
- The verdict block is the *only* section allowed to editorialize, and it may
  only say things the templates can derive.

### A2. Glossary and "how to read this" (new §10, linked from every table)

- Generate a definitions table from `METRIC_SPEC` plus a fixed block for the
  program's own vocabulary: **fill alpha**, **breakeven alpha**, **walk-forward
  OOS**, **anti-selection guard**, **block bootstrap**, **deployment / marked
  equity**, **Brier skill**, **spec hash**, **snapshot hash**, **analog layer**
  vs **model layer**, **EXTRAPOLATED**.
- Each entry: one sentence of definition, one sentence of *why it is here*
  ("breakeven alpha is the margin of safety on the mid-fill assumption — the
  single biggest risk in the program").
- Section headings link to it: `## 1. Headline ([definitions](#10-glossary))`.
- A `checks` test asserts every key rendered in any table has a glossary entry.

### A3. §1.5 Sample funnel — which universe every n belongs to

The single largest source of confusion in the current reports. Add a mandatory
funnel table, sourced from `evaluate()` (extend it to record the counts it
already computes) and from `build_trades --dry-run`:

```
| stage | events | note |
|---|---:|---|
| calendar events in window | 98,705 | 2018–2026, all tickers with an earnings date |
| both legs' chains cached | 17,754 | chain availability is the binding constraint |
| priced into trades | 4,736 | 543 dropped: calendar priced at a credit at some α |
| scored (complete gate features) | 11,080 | rows the live scorer could also score |
| selected by the gate (headline) | 7,620 | ← every §1 number is on THIS row |
```

Rules:
- The row the headline is computed on is marked, in every report.
- Any drop > 1% carries a reason string. "Dropped" without a reason is a FAIL
  in the checklist (A9).
- EXP-102's "the priced universe is conditioned on surviving as a debit at the
  BEST fill — the cheapest calendars are systematically excluded from every
  number in this report" is exactly the kind of finding that must be
  structural, not appended by hand.

### A4. §6 Calibration — a section, not a JSON dump

Replace `json.dumps(cal, indent=1)` (`engine/report.py:815`) with:

```
## 6. Calibration — are the predicted win rates real probabilities?

**No.** Brier skill **−0.084** (Brier 0.2583 vs 0.2382 for always predicting the
base rate of 39.1%, n = 9,678). The model's win probabilities are worse than a
constant. Reliability monotonicity 0.43 — the deciles are only weakly ordered.

**What this does and does not invalidate:** the P&L numbers in §1–§5 are
realized returns and are unaffected. What is affected is any use of `win_rate`
as a probability — position sizing off it, or a dashboard reading it as
"58% chance of a win". Rank order is usable; the level is not.

| decile | predicted | realized | n | error |
|---|---:|---:|---:|---:|
| 1 | 15.4% | 36.4% | 955 | +21.0 pp |
| … |
```

- The verdict line is derived from `brier_skill` against a stated floor
  (−0.05, the Phase 1 decision record's threshold —
  `reports/phase1_decision_calibration_reclassification.md`).
- Link that decision record so a reader knows this is a *known, accepted,
  tracked* state and not a fresh surprise.
- When no calibration block exists, say *which* report will carry it and when
  ("the ledger calibration report, after 50 scored predictions") rather than
  the current bare note.

### A5. Dead stress stages must say they are dead

`available: True` with NaN results is the worst of both worlds. Introduce a
three-state render for every stress stage:

| state | condition | renders as |
|---|---|---|
| MEASURED | coverage ≥ 5% and the statistic is finite | the number, with coverage in parentheses |
| **INCONCLUSIVE** | ran but coverage < 5% or the statistic is NaN | `INCONCLUSIVE — only 0.1% of trades had an adjacent cached chain; this stress has not been performed` |
| N/A | structurally inapplicable (no short leg → tail injection) | one reason, printed once |

- Fix the doubled note (`stress.tail_injection.note` is concatenated with the
  literal "tail injection N/A" at `engine/report.py:790`).
- INCONCLUSIVE stages feed the A1 warning list and the A9 checklist advisory.
  They do **not** fail the report — they are honest gaps, and the plan's
  stress battery is explicitly best-effort where chains are missing.
- Coverage threshold (5%) is a module constant with a comment; it is a
  judgement call and should be visible as one.

### A6. §4 Monte Carlo — print the reconciliation, not the caveat

The MC table and the deployment block currently describe two different worlds.
Put them in one table and label the axis of disagreement:

```
| sizing | MC terminal p50 (overlap ignored) | Deterministic, deployment-capped | MC P(loss) | DD p95 |
|---|---:|---:|---:|---:|
| 2%  | 306×      | 213×    | 0.0% | 43.8% |
| 5%  | 341,871×  | 21,113× | 0.0% | 78.5% |
```

- Add a standing header sentence above the table:
  *"These are properties of the trade sequence, not forecasts. The MC column
  compounds trades one after another; this book ran up to 133 simultaneously,
  so its terminal column is an upper bound and its P(loss) a lower bound. The
  sizing decision belongs to the Phase 5 go/no-go memo, not to this table."*
- Clamp displayed terminal equity: anything > 1e6× renders `>1e6×` with the
  exact value available in `results/metrics_*.json`. A report that prints
  `45,528,139,812,307,560×` teaches the reader to distrust the document.
- Where the plan quotes a canonical figure for the same strategy (STR-THRU:
  2.83× at 5%, P(loss) 15%), print a **reconciliation line** naming the
  construction difference. `EXP-050` used `equity_mode="sequential"` on n=531;
  this run is `cashflow` on n=7,620. Different sample, different construction —
  say so where the numbers sit, not in a commit message.

### A7. Figures

All in `engine/report.py:243–428`. Fix once, in shared helpers:

1. **Caption collision** — every figure calls `fig.text(…, y=0.005)` on top of
   the x-label. Add `_finish(fig, caption)`: `fig.subplots_adjust(bottom=…)`
   reserving 0.12, caption at `y=0.02`, `fontsize=7`, `wrap=True`. One helper,
   used by all seven.
2. **Log scale where the span demands it** — `_auto_yscale(ax, series)`: if
   `max/min > 100`, `ax.set_yscale("log")` and note `(log scale)` in the axis
   label. Applies to `fig_equity`, `fig_mc_fan`, `fig_mc_fan_paths`. The
   current equity chart is unreadable for the first seven of nine years.
3. **`fig_mc_fan`** — plot p05/p50/p95 on a log axis so p05 and p50 are
   visible; keep P(loss) on the right axis; mark the 5% sizing point.
4. **`fig_equity`** — annotate max-drawdown span, mark the walk-forward year
   boundaries with light vlines, and label the ungated years (2018–2019 here)
   so the reader sees which part of the curve is not gated.
5. **`fig_reliability`** — add the base-rate horizontal line, bin counts as
   point sizes (already) *with* an n legend, and shade the region where
   predicted > realized. The reader should see "all points below the diagonal"
   as a statement, not infer it.
6. **`fig_stress_grid`** — diverging colormap centered at 0 (currently the
   colorbar range makes +3.1% look pale-yellow-bad), n labels inside the cells,
   caption below reserved space.
7. **`fig_alpha_curve`** — mark α = 0.5 (mid) as well as breakeven, and shade
   the margin between them; that gap is the headline safety number.
8. **`fig_by_year`** — colour bars by sign, print n above each bar.
9. Determinism is unchanged: fixed seeds, no timestamps inside PNGs. Keep
   figure data alongside the PNG (`figures/<name>.json`) so the golden test
   compares arrays, not pixels (the guide's own §6 failure-mode note).

### A8. `extra_sections` — end the append-after-the-fact pattern

`EXP-105/run.py:53` and `EXP-102/run.py:61` open `REPORT.md` in append mode and
write their own markdown *after* the generator finished. That is how EXP-102's
best content (the defined-risk falsification, the assignment exposure, the
zero-cost selection) ended up outside the generator's formatting, ordering,
checklist and provenance.

Contract:

```python
Report(context | {"extra_sections": [
    {"title": "Max-loss distribution vs net debit",
     "kind": "table", "columns": [...], "rows": [...],
     "note": "...", "falsifies": "..."},          # rendered via the A0 formatters
]})
```

- Extra sections render **before** the appendix, numbered continuously, and go
  through the same formatters, so `-386.7%` and `+4.04%` look the same
  everywhere.
- A section may declare `promote_to_verdict: true` to contribute a row to §0
  (EXP-102's "defined-risk claim: FALSIFIED, 80 of 4,736 exceed the debit"
  belongs in the verdict, not on page 4).
- After this lands, **appending to REPORT.md is forbidden**; the acceptance
  suite greps every `experiments/*/run.py` for `open(... "a")` on REPORT.md.

### A9. Checklist upgrade

Keep the 7 plan-mandated items (they are the accuracy contract) and add a
second, clearly separated **advisory** table that carries the things a reader
needs and the current 7 do not cover:

| advisory | example |
|---|---|
| Calibration state | `WARN — Brier skill −0.084 (below the −0.05 floor)` |
| Stress coverage | `WARN — 2 of 5 stages INCONCLUSIVE` |
| Sample funnel disclosed | `OK — 5 stages, all drops > 1% carry reasons` |
| Concurrency vs sizing | `WARN — max 133 concurrent; MC sizing is an upper bound` |
| Sample size per headline year | `OK — min year n = 114` |

Advisories never block; FAILs on the mandatory 7 still block promotion and
publication. The distinction matters: the mandatory list is about *whether the
evidence is admissible*, the advisories are about *how far it stretches*.

### A10. The regeneration diff — the acceptance test for all of Part A

Re-run the three existing experiments through the new generator from their
saved `results/metrics_*.json`, and assert:

1. **Every numeric value present in the old report is present in the new one**
   (parse both, compare the multiset of numbers modulo formatting). Part A is a
   presentation change; a changed number is a bug.
2. The new report contains **zero** `{` / `}` JSON blobs in the body.
3. Every table cell that is a return, rate or fraction carries `%`.
4. No `—` appears without an accompanying explanation token
   (`N/A`, `INCONCLUSIVE`, `not measured`).
5. Section order: 0 Verdict → 1 Headline → 1.5 Funnel → 2 Equity → 3 By-year →
   4 MC → 5 Stress → 6 Calibration → 7 Checklist → 8 Provenance → extras →
   9 Appendix → 10 Glossary.

Commit the regenerated reports in the same commit as the generator change, so
the diff *is* the review artifact.

---

## Part B — the prediction ledger (`engine/ledger.py`)

The out-of-time validator, and the one Phase 4 deliverable with a deadline: it
must be accruing rows before Q3 earnings season (mid-Oct). Phase 3's nightly
job will call it; until Phase 3 exists, a CLI entry point does.

### Layout

```
ledger/
  predictions/YYYY-MM-DD.jsonl     # one row per (event, strategy, structure)
  outcomes/YYYY-MM-DD.jsonl        # written after the event resolves
  calibration/REPORT.md + figures/ # regenerated every 50 newly scored rows
  health.json                      # what Phase 3's model-health view reads
```

### Prediction row

Everything needed to score it later without re-deriving anything:

```json
{"row_id": "2026-10-15|AAPL|STR-THRU|atm|20261023",
 "written_at": "2026-10-15T21:05:03Z",     // wall clock
 "as_of": "2026-10-15",                     // the decision date — file date follows THIS
 "decision_ts": "2026-10-15T20:00:00Z",     // what the leak auditor checks against
 "ticker": "AAPL", "event_date": "2026-10-16", "session": "AMC",
 "strategy": "STR-THRU", "structure": {"legs": [...]},
 "intended_prices": {"alpha": 0.5, "entry_cost": 8.42, "legs": [...]},
 "score": { ...ScoreResult.as_dict() },     // both layers, gate, CI, flags, extrapolated
 "model_versions": {...}, "snapshot_hash": "dce985…",
 "audit_receipt": {...},                    // Part C
 "supersedes": null, "supersede_reason": null}
```

`ScoreResult.as_dict()` (`engine/score.py:219`) already carries the model layer,
analog layer, gate, CI, flags and snapshot hash — do not invent a parallel
schema, embed it whole and version it with `schema_version`.

### Append-only, enforced in code

- Writer opens the date file with `"x"` on first write and `"a"` after, and
  **refuses to rewrite an existing `row_id`** — raises `LedgerError`, never
  overwrites, never dedupes silently.
- No delete path exists. Corrections are new rows with `supersedes: <row_id>`
  and a non-empty `supersede_reason`; readers resolve to the latest
  non-superseded row per `row_id`.
- File date = `as_of` date, **not wall clock** (a job that runs 00:20 must not
  write tomorrow's file — the guide's own failure-mode note).
- `ledger/` is in the private mirror, not the public repo
  (`tools/private_mirror.py:51` already lists it).

### Outcome scorer

```
python3 -m engine.ledger score --through 2026-11-30
```

- For every unresolved prediction whose event has passed and whose exit chain
  is cached: price the **same structure legs** through `engine.replay` /
  `engine.structures` at the same alpha — the same pricing path the backtests
  use, never a second implementation.
- Write `{row_id, resolved_at, realized_move, realized_pnl, realized_win,
  fill_alpha_used, exit_source, notes}`.
- **Idempotent:** running twice creates no duplicate outcome rows. Unresolvable
  rows (chain never arrived, event moved) get an explicit
  `status: "unresolvable"` outcome with a reason — never silent omission, which
  would quietly select the ledger toward resolvable (liquid) names.
- Earnings-date changes are a known loss source (`engine/calendar.py`): if the
  event moved after the prediction, the outcome row records
  `event_date_changed: true` and the calibration report reports those
  separately.

### Calibration recompute + health.json

- Trigger: every ≥50 newly scored outcomes → regenerate
  `ledger/calibration/REPORT.md` **through `engine.report`** (kind
  `"calibration"`, reusing `engine/calibrate.py`'s `reliability_table`,
  `decile_table`, `brier_skill`) and rewrite `health.json`.
- `health.json` is Phase 3's contract; freeze it now:
  `{generated_at, n_scored, per_strategy: {brier, brier_skill, base_rate,
  reliability_monotonicity, predicted_mean_pnl, realized_mean_pnl, n},
  champion_versions, snapshot_hash, data_freshness, quota_state}`.
- This is the first chart on the model-health view, and it is the only place in
  the program where predicted win rates get tested out of code path. Given the
  known −0.084 Brier skill, expect it to be the number that drives Phase 2's
  next round of work.

### Bootstrapping before Phase 3 exists

```
python3 -m engine.ledger snapshot --as-of 2026-09-02 --horizon 21
```

Scores the upcoming-3-weeks calendar through `engine/score.py` and writes the
prediction rows. Wire it to cron now (`service cron status` first — WSL2), so
the ledger has weeks of rows before the season rather than starting empty on
day one. Phase 3's nightly job later calls the same function.

---

## Part C — audit receipts

`engine/audit.py` does the checking; what is missing is the *proof it ran*,
which the checklist claims but does not have.

```python
@dataclass(frozen=True)
class AuditReceipt:
    n_features_checked: int
    n_rows_checked: int
    max_as_of: pd.Timestamp        # latest feature timestamp seen
    decision_ts: pd.Timestamp
    margin_seconds: float          # decision_ts - max_as_of; must be > 0
    paths: list[str]               # "score" | "evaluate.walk_forward" | "ledger.write"
    receipt_hash: str
```

- Emitted by `assert_causal` / `audit_frame`, collected by the caller, embedded
  in `results["audit_receipts"]` and in every ledger prediction row.
- Checklist item 2 renders from the receipt:
  `PASS — 3 paths, 146,220 rows, 12 features, min margin 18.5 h` and is **N/A
  only** when no features were fitted (the current gateless case).
- Poison tests in all three wired paths (score, evaluate, ledger write): a
  deliberately future-stamped feature must raise `LeakError`, and the receipt
  must be absent from the report when it does.

---

## Part D — retire the bespoke report writers

| File | Now | Change |
|---|---|---|
| `checks/phase0_audit.py:130` | hand-built markdown → `reports/phase0_data_audit.md` | `Report(kind="audit")` with tables as extra sections; keep the same path |
| `checks/phase1_report.py:357` | hand-built markdown → `reports/phase1_*.md` | `Report(kind="calibration")` |
| `experiments/EXP-105/run.py:53` | appends to REPORT.md | `extra_sections` (A8) |
| `experiments/EXP-102/run.py:61` | appends to REPORT.md | `extra_sections` (A8) |

The generator gains two lightweight kinds beside `evaluation` and `promotion`.
They share the provenance block, the formatters and the glossary; they skip the
sections that do not apply (an audit has no MC). Do **not** let a kind invent
its own section vocabulary — that is how the format drifts back apart.

Hand-written *decision records* (`reports/phase1_decision_calibration_reclassification.md`)
and *review* documents are not generator output and stay as they are; the rule
is about **results**, not about prose.

---

## Acceptance tests (`checks/phase4_checks.py`)

Plain script with asserts, runnable as `python3 checks/phase4_checks.py`,
`--only <name>` supported (match the Phase 1/2 suites' shape).

**From the original guide (keep all seven):**

1. `golden_report` — fixed synthetic context → REPORT.md byte-identical modulo
   the timestamp line; figure **data arrays** identical (compare
   `figures/*.json`, not pixels).
2. `regeneration` — a real experiment's report regenerates from its provenance
   block's pinned inputs → same numbers to tolerance.
3. `checklist_honesty` — context missing the fill sweep → item 4 FAILs, red
   banner renders, `promote.decide` refuses it.
4. `ledger_append_only` — rewriting an existing prediction row raises; a
   `supersedes` row round-trips through the outcome scorer.
5. `outcome_idempotent` — scorer run twice → no duplicate outcome rows;
   unresolvable rows carry a reason.
6. `leak_poison` — future-dated feature raises in all three wired paths.
7. `calibration_trigger` — 50 synthetic scored predictions → calibration report
   regenerates and `health.json` updates.

**New, for Part A (these are what stop the format regressing):**

8. `numbers_preserved` — A10.1: regenerated EXP-102/105/107 contain the same
   multiset of numeric values as the committed originals.
9. `no_raw_json` — no `{`-blob in any REPORT.md body outside fenced
   ```` ```json ```` appendix blocks.
10. `units_present` — every table cell parsed as a return/rate/fraction column
    carries `%` or `×`; no bare 4-decimal floats in the body.
11. `no_bare_dashes` — every `—` is adjacent to `N/A`, `INCONCLUSIVE`, or
    `not measured`.
12. `section_order` — the A10.5 order, on all four report kinds.
13. `glossary_complete` — every metric key rendered in any table has a
    `METRIC_SPEC` / glossary entry.
14. `verdict_derivable` — `verdict()` on three canned results dicts returns the
    expected call and warning set (supported / not supported / descriptive).
15. `no_report_append` — no `experiments/*/run.py` opens REPORT.md for append.
16. `figure_captions` — every emitted figure has a caption and the caption's
    bbox does not overlap the axes bbox (matplotlib renderer check).

---

## Constraints

- **No number changes in Part A.** If a formatting fix reveals a wrong number,
  fix it in a separate commit with its own note — never inside a presentation
  change, where a reviewer cannot see it.
- The generator stays dependency-light: string templates, matplotlib Agg,
  pandas. No jinja2, no HTML toolchain (optional single-file HTML only if it is
  a trivial md→html pass).
- Determinism holds: same context + seed → byte-identical markdown modulo the
  timestamp line, identical figure data.
- Reports and ledger stay plain files — greppable, diffable, no database.
- Ledger and reports are **private-mirror** artifacts; `checks/repo_hygiene.py`
  already blocks them from the public repo. Run it before every push.
- Don't touch `earnings_predictions/` or `bt/`.

## Failure modes to watch

- **Verdict-block overreach.** The moment `verdict()` starts writing sentences
  that aren't template-derived, the report stops being reproducible. Every
  clause maps to a results-dict field or it doesn't ship.
- **Formatter drift.** Two places computing percentages differently is exactly
  the bug being fixed; `METRIC_SPEC` must be the only mapping from key to unit.
- **Golden tests too tight.** Font metrics vary; compare figure *data*, and
  compare markdown modulo the one timestamp line.
- **Ledger clock skew.** File date follows `as_of`, never wall clock.
- **Silent ledger selection.** Unresolvable predictions dropped instead of
  recorded would bias the ledger toward liquid names — the one bias the ledger
  exists to be free of.
- **Extra-sections becoming a dumping ground.** If a section is worth writing
  every time, promote it into the fixed order instead.

## Definition of done

1. `python3 checks/phase4_checks.py` — 16/16 green.
2. Unit suite green (`tests/test_report.py` extended; new `tests/test_ledger.py`).
3. EXP-102/105/107 regenerated: numbers identical, documents legible, committed.
4. `reports/phase0_data_audit.md` and the Phase 1 calibration report render
   through the generator.
5. `ledger/predictions/` has ≥1 real snapshot written by the CLI, and the
   outcome scorer has resolved at least one past event end-to-end.
6. `health.json` exists and matches the frozen schema (Phase 3 depends on it).
7. A Phase 4 report (`reports/phase4_reporting.md`, generator format) documents
   the before/after and the audit receipt wiring — Phase 4's own evidence, in
   its own format.
8. Public repo pushed with hygiene checks green; ledger and reports to the
   private mirror.

**Then Phase 3** (dashboard) starts, with `score.py`, the report generator, the
ledger and `health.json` all in place — which is exactly why it was reordered
behind this one.
