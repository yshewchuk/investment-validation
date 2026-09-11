# Prompt 1 — the execution clock, and the ledger's data restriction

**Version:** 2.0 · **Date:** 2026-09-10 · **Owner:** YS + Claude
**Status:** prompt only. Nothing here has been implemented. Every file:line and
every number was verified against `HEAD` on 2026-09-10; the measurement in §1.2
was run that evening.

Companion: `guides/prompt_serving_parity.md`, which builds the check that would
have caught the defect in §1.2. The two are independent and can run in parallel.

Execute in order. **§2 must land before §4** — that is the whole point of the
ordering, and §1.2 is why.

---

## 0. What you are being asked to change

The book must run on the clock the trade is actually executed on:

1. **A decision for whether to trade is baked into the book the day before the
   trade.** A gate that passes tonight is for tomorrow's trade.
2. **The position is opened the following day, at that day's close.**
3. **The position is closed on the exit date, at that day's close.**
4. **Nothing is recorded for a date — no decision, no open, no close — until
   the final closing data for that date has actually been retrieved.**

Requirements 1–3 are one change: move the **decision** one session earlier and
leave the trade exactly where it is. Mechanically that is
`decision_offset = entry_offset − 1`.

Requirement 4 is separate, larger, and has no existing machinery at all.

---

## 1. Ground truth — read before proposing anything

### 1.1 The timing machinery already exists and is unused

`decision_offset` was designed, built, tested and shipped on 2026-09-02 as
EXP-120 steps 1–6 (`guides/str_thru_t2_decision.md`). **Read that guide in full
before writing code.** It is the authority on this subject, it already answers
several questions you will otherwise re-derive, and its §4 documents a silent
leak that `assert_causal` cannot catch — §1.2 below is that leak, still open.

| what | where |
|---|---|
| `Structure.decision_offset`, `.decided_at`, `.decided_early`; rejects a decision later than its own entry | `engine/structures.py:465-521` |
| `resolve_offsets(..., decision_offset=)` → `PrintWindow.decision_date` | `engine/calendar.py:319-357` |
| `plan_events` carries `decision_date`; `chain_keys` unions the decision keys; a `no_decision_chain` skip reason is counted | `engine/replay.py:157` |
| `result.as_of` defaults to `window.decision_date` | `engine/score.py:954` |
| `_price_entry` quotes off the **decision-date** chain when `decided_early`, and records `quote_date` | `engine/score.py:1144-1168` |
| gate training picks its own cutoff: `"decision_date" if decided_early else "entry_date"` | `engine/models/training/gate.py:111` |
| `live_features` anchors all three market blocks on `as_of` and stamps the row each builder actually read | `engine/features.py:743-755` |
| D−1 chains bought and ingested — 3,628 calls, 0 failures, **99.9%** coverage over 26,746 replayable STR-THRU events, uniform by year and mcap | `reports/t2_pull_d1.json` |
| quote/fill drift measured on 26,702 events — median **0.00%**, cost-weighted **+1.48%** holding the named contract, **−1.31%** re-ATM'ing at D0 | `checks/t2_drift.py` |

**And it is all inert.** Every live structure ships `decision_offset=None`, so
decision date equals entry date for all eleven: CAL-P, STR-THRU, STR-RUNUP,
CND-P, TWIN-P, TWIN-P5, CND-PS, BFLY-P, BFLY-P5, RAMP7, CTR5.

### 1.2 The serving path is not decision-aware — measured, and this blocks everything

**`decision_offset` is not a config flag today.** Setting it would move the
decision and leave most of the features reading the entry close.

Measured 2026-09-10 on 16 real 2025 STR-THRU events (8 BMO, 8 AMC), each scored
twice — once as shipped, once with `decision_offset=-1`:

| | D0 → D−1 |
|---|---|
| `as_of`, `quote_date` | move to D−1 ✅ |
| `entry_date` | stays D0 ✅ |
| `dte_entry`, `entry_cost_pct` | move on **16/16** |
| `or_implied`, `dist_high`, `dist_ema`, `spy_vol20`, `spy_dd252`, `mcap_log`, `rvol30`, `iv30`, `im` | move on **0/16** |

Nine of the gate's features are byte-identical between a D0 decision and a D−1
decision. They must not be. Three sites cause it:

```python
# engine/score.py:1455  — the one feature row every layer reads
built = entry_feature_frame(
    frame, panel=self.context.panel, daily=self.context.daily,
    as_of_column="entry_date",          # hardcoded
)

# engine/score.py:1389-1401  — the forward path
key = (request.ticker, event_date, result.entry_date, result.session)
vector = live_features(..., as_of=result.entry_date, ...)

# engine/score.py:1487  — whether the panel market block is served at all
if result.entry_date is not None and result.entry_date >= window.last_pre_print:
    block = self._market_block(request, result)
```

And `_market_block` (`score.py:1416`) reads a historical event's block straight
off the panel row, whose anchor is the **event date** — with no rule that
withholds it when the decision precedes that anchor.

Note the asymmetry, because it decides what a regression test must assert. For
an **AMC** print, D0 is the event date and the panel anchors strictly before it,
i.e. at D−1 already — so the AMC half is correct by accident. For a **BMO**
print, D0 is E−1 and the panel anchors at E−1 too, so a D−1 decision is served a
full session of hindsight. The guide predicted exactly this (6/6 BMO, 0/6 AMC on
twelve events); the probe above reproduces it independently.

Training is decision-aware. Serving is pinned to the entry date. **That is the
same defect class as the analog-context bug and the chooser-analog substitution
— the third in one week** — and `assert_causal` passes through all of it,
because it compares stamps and every stamp would correctly read D−1.

### 1.3 Requirement 4 has no implementation anywhere

There is no predicate in the codebase for *"the close for date D exists and is
final"*. There are three loose, independent staleness heuristics:

| heuristic | what it actually allows | where |
|---|---|---|
| `daily_freshness` | newest `daily_market` row up to **4 days** old, and only **80%** of tickers need one. That is "Friday's data still serves Monday", not "today closed". | `nightly.py:57-61, 710` |
| `chain_anchor` | walks back to the newest session ORATS published and pulls chains for *that* date — then **leaves `as_of` alone**. The board keeps the requested stamp. | `nightly.py:665-672` |
| `quote_max_age_sessions` | prices off a chain up to **5 sessions** old and raises `STALE_QUOTE`. | `score.py:2924, 1178` |

The ledger file is named for `as_of` regardless — *"the file date follows
`as_of`, never the wall clock"* — so a run can freeze a row dated D whose
premium was quoted at D−1 or earlier, with nothing in the row required to say
so.

**Live proof.** The 2026-09-10 nightly ran with `--no-refresh`, so no
2026-09-10 data was fetched at all. It still wrote
`ledger/predictions/2026-09-10.jsonl` and published a board stamped
`as_of 2026-09-10`. Nothing objected.

Settlement has the same gap at the other end: `SETTLE_LAG_DAYS = 3` in
`engine/portfolio.py` is a calendar-day guess at when an exit chain *ought* to
exist, used to decide whether a trade is `awaiting_exit` or `unresolvable`. A
finality predicate replaces a guess with a fact.

### 1.4 The book has no position lifecycle

`engine/portfolio.py` builds the hypothetical book from frozen ledger rows and
derives `open` / `awaiting_exit` / `settled` / `unresolvable` after the fact.
`BOOK_COLUMNS` carries `as_of` and `event_date` but **not `entry_date` or
`exit_date`**, and the buy is taken "on the FIRST night the trade was
recommended" — which today is the same close as the entry, i.e. unactionable.
There is no record that says *this position was opened at this close* as an
event distinct from *this prediction was made*.

---

## 2. Step 1 — close the serving-side decision leak

**This lands first, and it is inert.** Because `decision_date == entry_date` for
all eleven live structures today, every change below is a no-op on the current
board — which is what makes it safe to land and verifiable before anything
moves.

### 2.1 The four sites

1. **`score.py:1455`** — `entry_feature_frame(..., as_of_column=...)` takes the
   decision date. The frame at `score.py:1444` needs a `decision_date` column to
   point at; `result.as_of` already carries it.
2. **`score.py:1389-1401`** — `_live_values` passes `as_of=result.as_of`, and
   **its cache key must gain the decision date**. Leaving the key on
   `entry_date` is worse than the original bug: two structures on the same event
   deciding on different dates would silently share one cached vector.
3. **`score.py:1487`** — re-key the guard from `result.entry_date` to the
   decision date, matching the docstring's own reasoning about STR-RUNUP.
4. **`_market_block` / `_panel_row` (`score.py:1369-1439`)** — withhold the
   panel block when the decision precedes the panel's own anchor, rather than
   serving a row baked at the event date. `panel_features` already has this rule
   from EXP-120 §5.3 step 5; reuse it, do not write a second one. A model that
   needs a withheld block reports `MISSING_FEATURES` and declines, which is the
   correct outcome and the refusal STR-RUNUP already gets.

Compose two ceilings rather than simply re-anchoring on the decision date — the
row read is `min(last row strictly before the event date, last row on or before
the decision date)`. EXP-120 §4 step 2 explains why each half is necessary and
why re-anchoring alone is wrong in both directions; do not re-derive it.

### 2.2 Prove it is inert, then prove it works

- **Inert:** re-score the current board and require **byte-identical** results.
  The 2026-09-10 board is 2,354 rows and a full score is ~17 minutes; a sampled
  digest comparison over a few hundred rows is enough if that is too slow, but
  say which you did.
- **Works:** re-run the §1.2 probe. The nine market and daily-state features
  must now move on the **BMO** half and stay put on the **AMC** half. Both
  halves are assertions — a test that only checks "something moved" would pass
  on a change that moved AMC too, which would be a different bug.
- **Regression tests**, per EXP-120 §5.8: the market block **differs** between
  `as_of=D0` and `as_of=D−1` on a BMO name (today it does not, and that identity
  is the bug); the event date stays a hard ceiling; and a stamp taken from an
  anchor *after* the decision makes `assert_causal` raise.

---

## 3. Step 2 — build the session-finality gate

This is requirement 4. Nothing downstream is trustworthy without it, and it is
independent of §2, so it can be built in parallel.

### 3.1 One predicate, one place

Add `engine/data/finality.py` — or the nearest existing home; check
`engine/data/manifest.py` and `engine/data/coverage.py` first and prefer
extending one over a new module. It answers, for a date and a ticker universe,
`session_finality(D, tickers) -> SessionFinality`, with a record that is
*positive and checkable* rather than a staleness bound:

| field | means |
|---|---|
| `market_wide` | ORATS `hist/summaries` and `hist/cores` for `tradeDate == D` returned 200, not 404 |
| `daily_share` | share of the universe with a `daily_market` row dated exactly D |
| `chain_share` | share of the universe with an `option_chains` row at `tradeDate == D` |
| `is_final` | all three clear their thresholds |
| `detail` | which failed, with the shares, so a flag can quote it |

Thresholds go in named module constants beside the existing
`MAX_STALENESS_DAYS`, with the reasoning in the docstring, not as inline
literals. Note the trap the nightly already documents: the ORATS market-wide EOD
file publishes hours after the per-ticker series, around 00:12, so `market_wide`
is the binding constraint most nights — and the walk-back in
`refresh_calendar_data` is the existing, working precedent for finding it at one
call per session.

### 3.2 Make it authoritative for `as_of`

`run_nightly` takes `as_of` from the CLI or the wall clock (`nightly.py:1043`).
Change it to *resolve* `as_of` to the newest date at or before the requested one
that is final, and record `requested_as_of`, `resolved_as_of` and the
`SessionFinality` that settled it on the run report. This generalises
`chain_anchor`, which already does exactly this for chains alone, and makes the
board's stamp mean something. `--as-of` becomes a ceiling, not an assertion; add
`--require-as-of` for a caller who genuinely means "this date or fail".

### 3.3 Gate the recorders — defer or degrade, per recorder

| recorder | on a non-final date | why |
|---|---|---|
| board / render / publish | **degrade** — publish, stamped with the resolved date, with a visible banner | a dark board is worse and less visible than a stale one; this is the nightly's existing philosophy and it is right |
| ledger decision row | **defer** — write nothing, flag, exit clean | a frozen row is append-only and un-editable. A decision recorded against data that did not exist is unrecoverable, and poisons every calibration that reads it |
| position open | **defer** | same |
| position close / settlement | **defer** | this is what `SETTLE_LAG_DAYS = 3` is guessing at; replace the guess with `is_final(exit_date)` |

The backfill path (`_nights_to_backfill`, `MAX_BACKFILL_DAYS = 7`) already exists
to cover missed nights, so "defer" has somewhere to defer *to*. Confirm it does
what you need before relying on it.

### 3.4 Carry it onto the row

Every ledger row gains the finality record for its own decision date, and every
settled outcome gains it for the exit date. A row that cannot say what data it
was computed from cannot be audited later, and that is the ledger's whole reason
to exist.

### 3.5 A known, separate ledger hazard — decide what to do about it

`row_id` for a DYN-SV row embeds the **chosen structure's** strike and expiry:

```
2026-09-10|KR|DYN-SV|56.0000|2026-09-11|2026-09-11|309d702f…
```

So a re-run of the same night that flips the chooser's pick does not dedupe — it
appends a second row for the same event. This happened for real on 2026-09-10
(KR: BFLY-P → CTR5, `rows: 1, skipped_existing: 204`), leaving two DYN-SV rows
for one event on one night. `ledger.snapshot` is correctly idempotent on
`row_id`; the question is whether a DYN-SV row's identity should include the
choice or only the event. Put it to YS rather than deciding it silently — and
note the ledger has a `supersedes` mechanism built for exactly this.

---

## 4. Step 3 — flip the decision offsets

Only after §2 and §3 are green.

### 4.1 Scope decisions for YS — bring the numbers, not the options

- **Which strategies move first?** STR-THRU's D−1 chains are bought and its arms
  are pre-registered, so it is nearly free. DYN-SV is not: `dyn_sv_chooser_v1_1`
  was trained on features cut at D0, and the D−1 pull targeted *STR-THRU's*
  replayable event set. The seven menu structures share `e+0_x+1`, so their
  entry and exit dates coincide — **measure the actual D−1 chain coverage over
  the menu candidate set before assuming it is covered.** Same 80% gate, same
  by-year and by-mcap breakdown EXP-120 §3 used.
- **Arm A or A′?** Name the strike at D−1 and buy that strike at D0 (A), or
  re-resolve ATM at D0 (A′). The drift measurement favours A — +1.48% against
  −1.31% cost-weighted — and A is the pre-registered primary. Confirm rather
  than re-litigate.
- **STR-RUNUP.** It enters at −14, so it already has a fourteen-session lead and
  its decision is not the thing that is late. Recommend leaving it at
  `decision_offset=None` and say why, rather than moving it for symmetry.

### 4.2 The change

- `decision_offset=-1` on STR-THRU and on whichever menu structures cleared.
  `straddle_through`, `put_calendar` and `straddle_runup` already take the
  argument through — check the menu factories do too.
- `_variant_label` appends `d{offset:+d}` only when `decided_early`, so the D−1
  book is a distinct variant (`e+0_x+1_d-1`) and no label already written
  changes meaning. Verify this holds for the menu factories: if two variants
  collide in `trades`, `e+0_x+1` silently means two different books.
- `dte_entry` at D−1 is `dte_entry@D0 + 1` **by construction**, and it is the
  gate's dominant feature (EXP-114, −0.353). Assert that identity in the dataset
  build. Leaving it on the entry date is a one-day leak in the single most
  important input.

### 4.3 Why this needs a retrain, and what kind

Not ceremony — two measured reasons. `dte_entry` shifts by exactly +1 on every
row, and `entry_cost_pct` re-quotes off a different session's chain. In the
§1.2 probe, gate scores moved materially on that alone: AAPL −0.068 → +0.025,
ABNB +0.037 → −0.041. And the champion's threshold, 0.0509, is the top-20%
**quantile of the D0 score distribution** — once the scores move, it is no
longer top-20% of anything.

Retrain per EXP-120 §6.2's frozen hypothesis: same architecture, same features,
same top-20% rule, `first_test_year=2020`, floor of `gate_lift ≥ +3.5pp` and
gated win ≥ 0.40. The floor sits deliberately below the champion's +4.42pp: a
D−1 decision is a strictly harder problem, what is being bought is
actionability, and the question is what it costs. Below the floor, the honest
conclusion is that STR-THRU does not survive a one-session lead — not that the
threshold needs rescuing.

### 4.4 Why promotion is the enforcement, not the ritual

The D−1 gate is a **new registry entry** (`gate_midfill_str_thru_t2`), never an
edit to the existing one, and `Scorer._score_gate` must select the champion by
`(strategy, role, decision_offset)`. That keying is the mechanism that makes it
impossible to serve a D0-trained gate against a D−1 decision — which, without
it, is precisely the substitution §1.2 documents. Until it promotes, the D−1
board runs **ungated and says so on every row**.

DYN-SV needs the same treatment: a D−1 variant of `dyn_sv_chooser_v1_1`,
selected by decision offset, not an in-place retrain.

---

## 5. Step 4 — make the lifecycle explicit in the book

- `PrintWindow` already has all three dates. Carry `decision_date`,
  `entry_date` and `exit_date` onto the ledger row — `structure` gains
  `decision_date`, `intended_prices` gains `quote_date` and `quoted_cost` — and
  bump `SCHEMA_VERSION` to 2. **Do not backfill v1 rows**: a v1 row means
  "decided at the entry close", which is true of it.
- `ScoreResult` gaining a field changes every `row_digest`. EXP-120 §5.2
  deliberately deferred that so the digest break is paid once, here. Expect the
  self-check to flag a wholesale digest change on the first run after this
  lands, and make sure that is distinguishable from a real mismatch.
- `engine/portfolio.py`: add `entry_date` and `exit_date` to `BOOK_COLUMNS`;
  replace "the buy is taken on the first night the trade was recommended" with
  "the buy is taken at the entry close following the decision that recommended
  it"; replace `SETTLE_LAG_DAYS` with `is_final(exit_date)`.
- The board needs a **Trade on** column beside the event date, a filter for
  "decision date == the resolved as-of" — the set actually actionable tomorrow
  — and the honest disclosure on every row: *premium quoted at the D−1 close;
  you will fill at the D0 close.* Put the measured number beside it: only
  **42.5%** of fills land within ±5% of the quoted premium, and median
  `|drift|` is **6.18%**. This is a rendering change and blocks nothing, but
  without it the board mixes tomorrow's trades with next week's and gives no way
  to tell them apart.

---

## 6. Acceptance criteria

**Step 1 — the leak**

- [ ] The four sites take the decision date; the `_live_values` cache key
      includes it.
- [ ] Re-scoring the current board is byte-identical (state whether full or
      sampled).
- [ ] The §1.2 probe re-run shows the nine features moving on BMO and **not**
      on AMC.
- [ ] Regression tests: BMO market block differs between D0 and D−1; event date
      stays a hard ceiling; a stamp later than the decision makes
      `assert_causal` raise.

**Step 2 — finality**

- [ ] `session_finality` exists, is unit-tested against a store with a missing
      day, and its thresholds are named constants with documented reasoning.
- [ ] A nightly run for a date whose close has not been retrieved **writes no
      ledger row**, flags, and exits clean; the following run covers the night.
- [ ] `--no-refresh --as-of <today>` on a day with no fetched close is refused
      by the ledger path. Reproduce the 2026-09-10 run first, so the
      before/after is on the record.
- [ ] The board and every ledger row carry the resolved as-of and the finality
      record that justified it.
- [ ] The DYN-SV `row_id` question in §3.5 has an answer from YS, applied.

**Steps 3–4 — the clock**

- [ ] `decision_offset=-1` on the agreed structures; decision date == entry
      date − 1 session for both BMO and AMC, asserted in `tests/test_calendar.py`.
- [ ] `dte_entry@D−1 == dte_entry@D0 + 1` asserted in the gate dataset build.
- [ ] A D−1 champion can never be served against a D0 decision, asserted in
      `tests/test_score.py`.
- [ ] `engine/portfolio.py` opens at the entry close following the decision and
      closes on `is_final(exit_date)`.
- [ ] The board shows the entry date, the quote date, and the drift disclosure.

---

## 7. What not to do

- **Do not flip `decision_offset` before §2 lands.** Measured: it would move the
  decision and leave nine of the gate's features reading the entry close, on the
  BMO half of the board, with `assert_causal` passing.
- **Do not backfill v1 ledger rows to schema v2.** The ledger is append-only by
  design and enforced in code; a v1 row already means something true.
- **Do not "fix" what is already solid.** `ledger.snapshot` is idempotent via
  `existing_row_ids()`; `_write_state` is tmp+replace; publish is atomic;
  settle, `model_evidence` and backup already degrade to flags rather than
  stopping the run.
- **Do not write a second withhold rule.** `panel_features` has one; reuse it.
- **Do not treat a green pytest run as evidence the board is correct.** Five
  defects have now hidden behind a green suite. Run the phase 3 battery and look
  at the board.
