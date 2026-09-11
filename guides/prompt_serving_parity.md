# Prompt 2 — a nightly check that the board serves the strategy the experiments developed

**Version:** 1.0 · **Date:** 2026-09-10 · **Owner:** YS + Claude
**Status:** prompt only. Nothing here has been implemented. Every file:line and
every number was verified against `HEAD` on 2026-09-10.

Companion: `guides/prompt_execution_clock.md`. Independent of it; can run in
parallel. The third defect in §1 is that prompt's blocking item, found by hand
in an afternoon — this check is how it should have been found.

---

## 0. What you are being asked to build

A nightly step that fails the night when **the strategy the board describes and
serves is not the strategy the experiments developed**.

Concretely: for every champion in the registry, rebuild a sample of the rows it
was *trained* on, score the same events through the *serving* path, and require
every registered feature to agree. Anything that disagrees is a silent
substitution, and today nothing in the pipeline can see one.

This is not a new idea being proposed on principle. It is the check that would
have caught three real defects in one week, each of which reached the live board
and at least one of which reached the frozen ledger.

---

## 1. The evidence — three defects, one shape, one week

### 1.1 The analog context defect — served to the board for weeks

`Scorer._enrich` sourced `implied_at_entry` from `self.context.daily`. Training
passed a full daily frame; the nightly passes a context narrowed to the board's
own names, so **98.5% of analog matches fell back** in serving and roughly none
did in training. Same code, same column names, different context width.

Measured coverage: **1.52%** in the nightly against **95.18%** after the fix.
The standalone self-check had been reporting 7/20 mismatched rows for weeks, and
that was read as noise.

### 1.2 The chooser analog substitution — every DYN-SV row on every board

`dyn_sv_chooser_v1_1.features` lists `analog_mean`, `analog_win_rate`,
`analog_p10`, `analog_p90`, `analog_n`. Those five names were built two
completely different ways.

| | trained on | served (before the fix) |
|---|---|---|
| built by | `experiments/EXP-169_menu7prime_confirmation/run.py` → `base.add_causal_analogs`, where `base` is `experiments/EXP-161_.../run.py` loaded **by file path** | `engine/score.py`, `bucket_frame` |
| method | kNN, k=25, over `exp_pnl_sim`, `width_over_forecast`, `n_legs`, `anchor_over_spot`, `rel_spread` | exact bucket match on `(mcap_bucket, dte_band, moneyness_band, implied_tercile)` |
| rank correlation with the trade's own return, 47,017 menu trades | **+0.0587**, 9/9 years positive | **−0.0005**, 5/9 |

Rank agreement between the two constructions: **0.021**. The model was fed
uncorrelated noise where it had learned a signal, and the board rendered the
result as a confident number.

The same five names are also listed by
`gate_midfill_str_thru_forecast_analog`, where the **bucket** construction is
the correct and validated one (EXP-145 arm1→arm7: CAGR +118.4% → +235.5%, years
positive 6/9 → 9/9). Neither construction is wrong. The collision is.

### 1.3 The decision-date defect — currently open

Training picks its cutoff correctly:

```python
# engine/models/training/gate.py:111
as_of_column = "decision_date" if decided_early else "entry_date"
```

Serving does not:

```python
# engine/score.py:1455
built = entry_feature_frame(..., as_of_column="entry_date")   # hardcoded
```

Measured on 16 real 2025 STR-THRU events scored twice, with and without
`decision_offset=-1`: nine of the gate's features — `or_implied`, `dist_high`,
`dist_ema`, `spy_vol20`, `spy_dd252`, `mcap_log`, `rvol30`, `iv30`, `im` — are
**byte-identical on 16/16** between a D0 decision and a D−1 decision. Only
`dte_entry` and `entry_cost_pct` move, and only because `_price_entry` correctly
re-quotes off the decision chain.

This one has not shipped, because no structure sets a decision offset yet. It is
sitting in the code waiting for the first one to.

---

## 2. Why nothing in the pipeline can catch these

- **The self-check compares serving to serving.**
  `engine/dashboard/selfcheck.py` re-scores sampled board rows through
  `engine.score.score` and compares digests and displayed values. That proves
  the bundle matches the engine. It cannot, even in principle, prove the engine
  matches the training frame. Both defects in §1.1 and §1.2 were invisible to
  it, and §1.1 actually *did* make it print mismatches for weeks without anyone
  being able to say what they meant.
- **The registry pins names, not construction.** A registry entry carries
  `features: [...]` and `artifact_sha256`. It says which columns go in and in
  what order, and nothing at all about how any of them is computed. Two
  champions in the same `registry.json` list `analog_mean` and mean different
  things by it.
- **The dependency arrow points the wrong way.**
  `engine/models/training/chooser.py:22` loads
  `experiments/EXP-169_.../run.py` by file path to build the champion's training
  frame, and that runner loads `experiments/EXP-161_.../run.py` as `base`. The
  engine depends on an experiment; the serving implementation lives in
  `engine/score.py` and shares no code with any of it. Every promotion adds one
  more of these.
- **`check_feature_equivalence` asks a different question.** It compares the
  panel path against the live path for the *same* `as_of`
  (`checks/phase1_checks.py:278`). That is panel-vs-live, not
  training-vs-serving, and both sides of it sit inside the serving half.
- **1,567 passing tests said nothing.** Five defects have now hidden behind a
  green suite. All three above were found by looking at the board or by probing
  by hand.

---

## 3. What to build

### 3.1 P1 — the parity check itself

This is the deliverable. Everything else in §3 is supporting work.

For each champion in the registry with a `features` list:

1. Rebuild a sample of its training rows through its own
   `engine/models/training/<role>.py::build_dataset()`.
2. Score the same `(ticker, event_date, strategy)` through the serving
   `Scorer`, **using the nightly's bounded context, not a full one**.
3. Assert every registered feature agrees within tolerance. Report per feature:
   agreement rate, max delta, and the rows that disagree.

Three design notes, each learned the hard way:

- **The bounded context in step 2 is the whole point.** A full context is
  precisely what made §1.1 invisible. A parity check that builds its own
  convenient context reproduces the bug it exists to catch. Take the context the
  nightly actually constructs.
- **Tolerance is per feature, not global.** A float round-trip through parquet
  is not a defect; a 0.021 rank agreement is. Start at `1e-9` like
  `phase1_replay.FEATURE_TOLERANCE`, and where a feature genuinely cannot meet
  it, record *why* in the check rather than loosening the global bar.
- **Report coverage, not just agreement.** §1.1 was a *fallback rate* defect —
  the values that were present agreed fine; the problem was how few there were.
  A check that only compares non-null pairs would have passed it. Compare
  null-ness as a first-class assertion.

Cost budget: the ad-hoc version of this run during the chooser fix verified
64,910 rows bit-exact in a few minutes. The nightly's self-check spends 15.9s on
20 rows. Budget **~60s** for the nightly step by sampling per champion, and
leave the full sweep to the standalone CLI.

Ship it in two places:

- `checks/phase1_checks.py` — so it runs in the phase battery alongside
  `check_feature_equivalence`, with the full sample.
- A nightly step — so it runs against the real board, on the real context, every
  night. See §3.2.
- `python3 -m engine.dashboard.parity --champion <id>` — standalone, for
  debugging one model without a board.

### 3.2 Where it goes in the nightly, and what it does when it fails

The run's steps today, with timings from 2026-09-10:

```
universe → refresh → validate → tiers → score (1,019s) → ledger (2.4s)
→ settle → ladder → backfill → model_evidence → render → selfcheck (15.9s)
→ publish → flags → backup
```

Put parity **between `score` and `ledger`**. It needs the scored board, and it
must run before anything is frozen.

On failure, follow the nightly's existing philosophy, which is right:

| | on a parity failure |
|---|---|
| ledger | **defer** — write nothing, flag, exit clean. A frozen row recorded from a model fed the wrong features is append-only and unrecoverable. |
| board / publish | **degrade** — publish with a visible banner naming the champion and the features that disagreed. A dark board is worse and less visible than a flagged one. |

That mirrors the defer/degrade split in `guides/prompt_execution_clock.md` §3.3;
keep the two consistent, and if you build one first, factor the policy so the
second reuses it.

### 3.3 P2 — make feature identity explicit

Two options. Put both to YS; they are complementary rather than exclusive.

- **Namespace them.** Rename to `chooser_analog_*` and `bucket_analog_*` so the
  collision cannot be expressed. Cleanest, but it changes the registered feature
  lists, so both models must be re-registered — and **check whether the fitted
  artifacts key on feature name or on position before assuming a rename is
  free.**
- **Declare the builder.** Registry entries gain `feature_builder`: the import
  path plus a content hash of the function that produces the column.
  `Registry.validate` rejects any entry whose builder hash does not match what
  is on disk. Non-breaking, and it catches the *next* collision rather than only
  this one.

Recommendation: both, builder-declaration first — it is the general mechanism,
and the rename is a special case of what it protects.

### 3.4 P3 — invert the dependency

Move champion feature construction out of the experiment runners into `engine/`,
and have the experiments import it. An experiment is a frozen record of what was
run; it is the wrong thing to depend on, and the current arrangement guarantees
that a training frame and a serving frame can never share code.

Do this incrementally — one champion at a time, with P1 green before and after
each move as the proof the move changed nothing. `dyn_sv_chooser_v1_1` is the
worst case and the right one to do second, after a simpler champion has proved
the pattern.

### 3.5 P4 — make context-independence a standing rule

`tests/test_score.py::test_narrowing_the_live_context_does_not_move_the_analogs`
exists for the one column that broke. Generalise it: score a sample of rows
under the full context and under the nightly's bounded context and require
**identical digests**.

Any feature whose value depends on which other tickers happen to be on the board
tonight is a defect by construction. This test names that class rather than one
instance of it, and it is cheap enough to run in the unit suite.

### 3.6 P5 — close the pre-registration loop

`spec_hash_checked` is written and never read, and the guard no-ops without a
`planned` LEDGER.csv row — EXP-173–177 ran unregistered. A promotion protocol
that can be skipped silently is the upstream of everything in this document: it
is how a feature construction gets into a champion without anything recording
which construction it was.

---

## 4. Acceptance criteria

- [ ] The parity check runs in the phase battery **and** as a nightly step
      between `score` and `ledger`, and is green for every champion.
- [ ] A parity failure **defers the ledger** and **degrades the board** with a
      banner naming the champion and the disagreeing features.
- [ ] The nightly step costs ≤ 60s; the standalone CLI can sweep everything.
- [ ] Null-ness is compared as a first-class assertion, not only non-null pairs
      — demonstrate on the §1.1 fallback-rate shape.
- [ ] **Deliberately reverting each of the three defects in §1 makes it red.**
      Demonstrate this, one at a time, and record which features it named. A
      check that has never failed on a known defect is not evidence.
- [ ] The bounded-vs-full-context digest test covers every scored strategy.
- [ ] No engine module imports from `experiments/` — or a list of the remaining
      ones exists, with an owner and a date.

---

## 5. What not to do

- **Do not build the check on a convenient context.** If it constructs its own
  full context for step 2, it reproduces §1.1 and will report green on it.
- **Do not rename the analog columns without re-registering both models**, and
  check first whether the fitted artifacts key on feature name or on position.
- **Do not retrain the STR-THRU gate onto the chooser's kNN analog, or vice
  versa.** Both constructions are correct where they were fitted: the bucket
  analog is validated on STR-THRU and carries nothing on the menu population,
  where it was never fitted. The defect was the substitution, never the
  estimator.
- **Do not widen the tolerance to make a champion pass.** A feature that cannot
  meet `1e-9` has a reason; record the reason.
- **Do not extend `selfcheck.py` to do this.** It answers a different question —
  bundle versus engine — and it answers it well. Two checks, two names, two
  failure messages.
- **Do not treat a green pytest run as evidence the board is correct.** Run the
  phase 3 battery and look at the board.
