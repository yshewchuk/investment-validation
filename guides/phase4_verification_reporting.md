# Phase 4 Guide — Report Generator, Prediction Ledger, Leak Auditor

**Objective:** one report generator used by every phase, an append-only
prediction ledger that makes live claims falsifiable, and a mechanical leak
auditor. Build this ALONGSIDE Phase 2 (evaluate.py's output format is this
generator's input format).

---

## 1. Report generator (`engine/report.py`)

```python
Report(context).write(out_dir)   # -> REPORT.md + figures/*.png [+ report.html]
# context: metrics dict (Phase 2 canonical keys), figures data, provenance
# inputs, accuracy-checklist evidence, free-text sections
```

- Markdown assembled from string templates (no jinja2 dependency); optional
  single-file HTML via a trivial md→html pass if pyarrow-era installs allow
  the dependency, else skip — markdown + PNGs is the contract.
- Figures: matplotlib with the Agg backend (headless), one function per
  standard figure so styles stay uniform: equity+drawdown, by-year bars, MC
  fan (percentile bands), stress grid heatmap, reliability/calibration
  curves, fill-alpha breakeven curve. Every figure gets a caption stating
  what would falsify the result it shows.
- **Fixed section order** (consumers rely on it): Headline table
  (worst/mid/best + breakeven alpha) → Equity/DD → By-year → Monte Carlo →
  Stress grid → Calibration → Accuracy checklist → Provenance → Appendix
  (grid/secondary results).

### Accuracy checklist — auto-evaluated
Each item renders PASS / FAIL / N/A **with an evidence pointer**, and the
generator computes them, it doesn't take the caller's word:
1. Real prices only (input trades' `src` fields all ORATS/Polygon).
2. Leak audit ran on this evaluation (audit receipt present, see §3).
3. Headline = walk-forward OOS only (headline stage tag == "wf_oos").
4. Fill sensitivity present (≥3 alphas + breakeven in metrics).
5. Multiple-testing ledger cited (LEDGER.csv rows for this spec_hash counted
   and printed: "spec N of M tried against this snapshot").
6. Survivorship caveat included.
7. Preregistration valid (timestamp ordering, from run_log).
A report with any FAIL renders a red banner at the top — it can exist as a
diagnostic, but promote.py and publish paths treat FAIL as blocking.

### Provenance block — the regeneration contract
- Input files: path + sha256 (files >100 MB: size+mtime+first-MB hash).
- Data snapshot hash (`data/features/SNAPSHOT`), spec hash, all seeds.
- Code state: sha256 of every `engine/` module imported during the run
  (collect via a small import hook or explicit list).
- Quota state and generator version.
**Rule: a report that cannot be regenerated from its provenance block is a
bug** — the acceptance suite proves this (§5.2).

## 2. Prediction ledger (`ledger/`)

```
ledger/predictions/YYYY-MM-DD.jsonl   # one row per event x strategy x structure
ledger/outcomes/YYYY-MM-DD.jsonl      # joined after the event resolves
ledger/calibration/                   # generated calibration reports
```
- Prediction row: everything needed to score it later — full ScoreResult
  fields + structure legs + intended prices at MID — written by the Phase 3
  nightly job BEFORE outcomes exist.
- **Append-only, enforced in code:** the writer opens date files with mode
  `"x"` semantics per row-batch and refuses to rewrite an existing file;
  corrections are NEW rows with `supersedes: <row_id>` and a reason. There
  is no delete path. (This is what makes the ledger the ultimate validator:
  nothing can be retroactively cleaned.)
- Outcome scorer (runs in the nightly job): for each unresolved prediction
  whose event has passed, join the realized move and realized structure P&L
  (priced from post-event chains through the SAME structures path), write the
  outcome row.
- Calibration recompute: every ≥50 newly scored predictions → regenerate the
  calibration report (predicted win deciles vs realized; predicted vs
  realized mean P&L; Brier; per-strategy) and update the dashboard's
  health.json. The cumulative ledger curve is the first chart on the model
  health view.

## 3. Leak auditor (`engine/audit.py`)

- Convention: every Tier-3 feature column has a companion as-of (or the
  table has a single `feature_as_of` when uniform).
  `assert_causal(features, decision_ts)` raises on any as_of ≥ decision_ts.
- Wired into: every `score()` call, every evaluate() feature build, the
  ledger writer (prediction decision_ts vs data as-of). It emits an **audit
  receipt** (counts checked, max as-of margin, timestamp) that the report
  checklist requires — "we ran it" is provable, not asserted.
- Poison tests are part of CI-style checks: a deliberately future-dated
  feature must raise in each wired path.

## 4. Constraints

- The generator is the ONLY way phases emit results (Phase 2 experiments,
  Phase 1 calibration, Phase 3 nightly summary, Phase 5 weekly reviews,
  promotions). Bespoke one-off result files are a review-blocking smell.
- Deterministic figures: fixed seeds, fixed axis policies, no timestamps
  inside PNGs (put timestamps in the markdown), so golden-file comparison
  works.
- Reports and ledger are plain files in the repo tree — greppable, diffable,
  no databases.

## 5. Acceptance tests (`checks/phase4_checks.py`)

1. **Golden report:** fixed synthetic context → REPORT.md byte-identical
   modulo the timestamp line; figures pixel-identical (or numerically
   identical via saved .npy figure data if font rendering varies).
2. **Regeneration:** take a real Phase 2 report, delete outputs, re-run from
   the provenance block's pinned inputs → same numbers to tolerance.
3. **Checklist honesty:** feed an evaluation missing the fill sweep → item 4
   FAILs and the red banner renders; promote.py refuses it.
4. **Ledger append-only:** rewriting an existing prediction file raises;
   a supersedes row round-trips correctly through the outcome scorer.
5. **Out-of-order scoring:** outcome scorer run twice → idempotent (no
   duplicate outcome rows).
6. **Leak poison** in all three wired paths (score, evaluate, ledger write).
7. **Calibration trigger:** 50 synthetic scored predictions → calibration
   report regenerates and health.json updates.

## 6. Failure modes

- Figure library drift breaking golden tests → compare underlying figure
  data arrays, not pixels, as the fallback assertion.
- Huge input lists bloating provenance → the >100 MB hashing shortcut; never
  skip a file silently (count must match inputs used).
- Ledger clock skew (job runs after midnight) → decision_ts is the scoring
  as_of, file date follows as_of, not wall clock.
