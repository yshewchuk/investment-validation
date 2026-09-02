# Guide — moving STR-THRU to a T−2 decision

**Version:** 1.0 · **Date:** 2026-09-02 · **Owner:** YS + Claude
**Status:** plan only. No code changed, no quota spent, nothing registered.
§6 is pre-registration: its numbers are frozen before any model runs.
**Experiment id (when scaffolded):** EXP-120.

---

## 1. The problem, measured

The board cannot produce a STR-THRU number until the moment to trade has
already passed. This is not a bug in the nightly — it is the structure's own
definition, and it is visible in the frozen ledger.

`ledger/predictions/2026-08-31.jsonl`, 213 STR-THRU rows:

| outcome | rows | share |
|---|---|---|
| priced (has `entry_cost`) | 12 | 5.6% |
| `NO_CHAIN` | 201 | 94.4% |

The 12 that priced are the ones whose entry date *was* 2026-08-31 — the night
the board ran. Every other row is an event whose entry date is still in the
future, and `Scorer._price_entry` refuses to price it, correctly
(`engine/score.py:685`, `_note_chain_age`: substituting last quarter's chain
would be fiction frozen into the ledger).

### Why it is structural

`straddle_through()` ships `entry_offset=0` (`engine/structures.py:634`), and
offset 0 is the **last pre-print close** (`engine/calendar.py:275`). The
scorer then does two things that close a loop:

1. `result.as_of` defaults to `window.entry_date` (`engine/score.py:490`) —
   the decision date *is* the entry date.
2. `_price_entry` looks the chain up at `entry_date`.

The chain for a session only exists after that session's close. So the earliest
the number can exist is the evening of the entry close — hours after the close
you were supposed to buy into. Reading the same fact from the other side:
**any design where decision date == entry date is unactionable, at every
offset.** Picking a different `entry_offset` alone does not fix this.

The fix is therefore not "enter earlier". It is **split the decision date from
the entry date, and give the gap one trading session.**

---

## 2. What "T−2" means precisely

The codebase anchors offsets on the last pre-print close, not on the event
date, because BMO and AMC prints put that close on different calendar days.
Write it out once and use these names throughout:

| name | offset | BMO print on E | AMC print on E |
|---|---|---|---|
| `D0` | 0 | E−1 | E |
| `D−1` | −1 | E−2 | E−1 |
| `D−2` | −2 | E−3 | E−2 |

Today: decide at `D0`, enter at `D0`, exit at `+1`.

**Recommended target (Arm A):** decide at `D−1`, enter at `D0`, exit at `+1`.
The trade you place is the trade the champions were validated on. What moves is
the decision — features, gate, model, analogs, and the quoted premium all cut
off at `D−1` — which buys you a full session to act.

For a BMO name that is literally "decide E−2, buy during E−1". For an AMC name
it is "decide E−1, buy during E". If you meant E−2 for AMC names too, that is
**Arm B** below; it is pre-registered as a secondary arm so the data decides
rather than the wording.

---

## 3. The blocking data gap — read this before writing any code

The T−2 chains do not exist in the store. Measured 2026-09-02 over 100,696
planned STR-THRU events, 2017–2026:

| chain at | present | share |
|---|---|---|
| entry date (`D0`) | 52,477 | 52.1% |
| exit date (`+1`) | 26,784 | 26.6% |
| **`D−1`** | **298** | **0.3%** |

Events replayable end-to-end today (entry **and** exit present, 2018–2026):
**26,710**. Events replayable with a `D−1` decision as well: **43**.

The cache is event-centric by construction — every historical pull asked for
the entry date and the exit date and nothing else. So the T−2 transition
**cannot be backtested at all** until the `D−1` chains are bought. Forty-three
trades is not a backtest; it is a rounding error.

### What is *not* missing

`daily_market` is a genuine daily series — 9,029,678 rows, 2007–2026 — and
`daily_state_frame` reads "the last row on or before `as_of`"
(`engine/features.py:698`). Every daily-state feature in the gate (`im`,
`iv10`, `iv30`, `skew`, `contango`, `rvol30`, `spot`, `mcap_log`, and all the
`_d1/_d5/_d10` differences) is already available at `D−1` for free. Only the
**option chain** needs buying.

### The pull, costed

`hist/strikes` bills per `(tradeDate, ticker-batch≤10)`, so cost is driven by
dates × batches, not by event count. Against the 26,710 replayable events,
2018–2026:

| pull | calls | new (date, ticker) pairs |
|---|---|---|
| Arm A only (`D−1`) | 3,628 | 26,584 |
| Arm B extra (`D−2`) | 3,630 | 26,607 |
| **union, both arms** | **6,365** | 53,158 |

Quota at 2026-09-02 14:41Z: **16,528 remaining**, reserve floor 3,000 →
**13,528 spendable**. Both arms cost 47% of that, once. If you want to stage
it, `D−1` for events from 2023 onward is 1,954 calls and still gives 15,605
trades.

Pull parameters, deviating from `CHAIN_DTE` in exactly one place:

```
endpoint   hist/strikes
tradeDate  <the D−1 session>
ticker     <=10 per call, batches built in a STABLE sorted order
dte        1,46          # not 1,45 — see below
fields     engine.dashboard.nightly.CHAIN_FIELDS  (unchanged)
```

`dte=1,46`: the post-event expiry is exactly one day further out at `D−1` than
at `D0`, so a `1,45` window silently drops the events whose traded expiry sat
at the ceiling. Widening by one costs no extra calls.

Batch order matters for money: `Fetcher` caches on exact request params, so a
re-run with a differently-composed batch is a fresh call, not a cache hit. Reuse
`engine.data.pulls.sep2026_plan.execute` — it is already cache-first,
quota-guarded, resumable, and it distinguishes truncation from genuine absence
(`sep2026_plan.py:375`). Add a `build_t2_plan(...)` beside `build_plan`; do not
write a new puller.

### Step 3 result — the planner agrees with the estimate — **done 2026-09-02**

```
python3 -m engine.data.pulls.sep2026_plan --t2 --dry-run                     # arm A
python3 -m engine.data.pulls.sep2026_plan --t2 --dry-run --decision-offset -1 -2
```

(Space-separated, not `=`: `--decision-offset=-1 -2` leaves the `-2` dangling.)

`build_t2_plan` lives in `engine/data/pulls/sep2026_plan.py`, plans one or
several arms together, and every number it produced reproduces §3's estimate to
the call:

| | planner | §3 estimate |
|---|---|---|
| replayable events, 2018+ | 26,710 | 26,710 |
| `D−1` pairs / calls | 26,584 / **3,628** | 26,584 / 3,628 |
| `D−2` pairs / calls | 26,607 / **3,630** | 26,607 / 3,630 |
| both arms, planned together | 53,158 / **6,365** | 53,158 / 6,365 |

Existing decision-chain coverage over the replayable set: `D−1` **0.2%** (43
events), `D−2` **0.1%** (20). Neither arm has anything to build on.

**One thing the estimate did not say: staging is not free.** Pulling the two
arms in separate runs costs **7,258** calls against **6,365** for one run —
**+893**, 14% — because a call buys one *date* for up to ten tickers and the two
arms land on the same 1,643 dates, so each arm separately pays for its own
half-empty batches. That does not settle the staging question, which §6.3 does;
it prices it. Arm B's marginal cost is 2,737 if bought with Arm A and 3,630 if
bought after it.

Both arms together are 47% of the 13,528 spendable calls; Arm A alone is 27%.

The universe rule the planner enforces, which is not the same as `build_plan`'s:
it buys decision chains only for events that are **already replayable
end-to-end**. A decision chain on an event with no exit chain buys a trade that
still cannot be priced. It also defaults to every mcap slice including
`unknown`, where `build_plan` targets slices — this pull is not closing a
per-slice coverage gap, and dropping the 58 events whose panel row has no market
cap would restrict the retrain universe for a reason unrelated to timing.

Staging by year, if the budget ever gets tight: 2023-on is 15,589 of the 26,667
`D−1` events (58%) and 2020-on is 21,213 (80%).

Then ingest: `python3 -m engine.data.rebuild --table chains`. (The table is
named `option_chains` in the store; the rebuild flag calls it `chains` —
`--table option_chains` is rejected by argparse.)
`n_chains.normalize_fetch_rows` picks the new raw files up with no change, and
dedupes on the primary key.

**Coverage gate before anything downstream runs.** If fewer than 80% of the
26,710 replayable events come back with a usable `D−1` chain, stop and report
the shortfall by year and mcap bucket rather than retraining on whatever
survived — a T−2 book silently restricted to the liquid half of the universe
would beat the champion for reasons that have nothing to do with timing.

### Step 4 result — the pull ran and the gate passed — **done 2026-09-02**

Arm A only. `python3 -m engine.data.pulls.sep2026_plan --t2 --confirm`, then
`python3 -m engine.data.rebuild --table chains`.

```
planned 3,628   fetched 3,628   failed 0
cache_hits 0    absent_tickers 0   truncation_retries 0   elapsed 1,962s
```

Not one missing ticker across ~36,000 requested, so the verification-retry path
never fired and there was no overage. Report: `reports/t2_pull_d1.json`.
Chain table 15M → **22,763,172 rows**, snapshot `19d0afac…`.

**The gate, over the replayable set:**

| | |
|---|---|
| replayable events | 26,746 |
| with a `D−1` chain | 26,710 |
| **coverage** | **99.9% — PASS** (threshold 80%) |

Uniform, which is the point — the gate exists to catch a universe quietly
restricted to the liquid half:

```
by year   99.4%–100.0%, every year 2018–2026
by bucket 1-10B 100.0% | >10B 100.0% | <1B 99.9% | unknown 72.5% (80 events)
```

The only sub-gate slice is `unknown` — 22 events of 26,746 (0.08%), and
"unknown" means no market cap in the panel rather than no chain. 36 events lack
a `D−1` chain outright (PEP ×4, 32 singletons); genuine ORATS absences.

Two things the estimate did not predict:

- **The replayable set grew**, 26,710 → 26,746. Some `D−1` batches landed on
  dates that were also another event's entry or exit date, so 36 events became
  replayable for free.
- **The ORATS quota header is coarse.** The ledger logged 3,628 rows, one per
  call, but `quota_remaining` moved only **7 times**, in jumps of 530–578. The
  header updates in batches of ~550, so any single reading lags by up to that
  much — `throttle.check_quota` reads it, so the reserve floor carries a blind
  spot of the same size. The 3,000 floor absorbs it comfortably, which is a
  reason to keep the floor generous rather than trim it toward the true minimum.

Quota after: header **13,131**, true remaining ~**12,858** (16,486 at pull start
− 3,628), ~9,858 above the floor. Arm B (3,630) still fits.

---

## 4. The silent leak you must fix first

This one will not announce itself. `assert_causal` will pass and the number
will be wrong.

**Three** panel builders — not two — anchor on the **event date**, not on
`as_of`:

- `add_orats_features` (`engine/data/features/panel.py:421`) reads the last
  `daily_market` row *strictly before `out["date"]`* — the event date.
- `add_runup_features` (`panel.py:363`) computes `dist_high`, `dist_ema`,
  `ret5/10/20` at `searchsorted(pdates, event_date, "left") - 1` — again the
  event date.
- `add_regime_features` (`panel.py:254`) does the same for `spy_ret21/63/252`,
  `spy_dd252`, `spy_vol20`. This one is easy to miss and `spy_vol20` is a
  `size_v1_4` input.

`live_features` accepts an arbitrary `as_of` and passes it to
`assert_decision_causal`, but then calls all three builders, which ignore it.
It then computes `market_date` from `as_of` and hands that to `_stamps`
(`features.py:326`) as the observation date for the whole market block.

So with the decision moved to `D−1`, the market block would carry values read
at `D0` while stamped `D−1`. `assert_causal` compares stamps, not provenance —
it would pass. **One full session of hindsight, on exactly the features that
move most into a print, invisible to the audit that exists to catch it.**

**Measured scope: BMO names only.** For an AMC print `D0` is the event date
itself, and the panel's anchor is the last row *strictly before* it — already
`D−1`. So the inherited "one session stale on AMC" conservatism happens to put
the AMC anchor exactly where a `D−1` decision needs it. Verified on twelve real
events: at `D−1` the market block moves on 6/6 BMO names and 0/6 AMC names. The
magnitude on BMO is not marginal — `or_implied` moves 7.05 → 7.51 on LW
2024-01-04 and 12.19 → 15.54 on GBX 2024-01-05.

This bites the model layer hardest: `size_v1_4` — STR-THRU's driver — lists
`or_implied`, `or_rvol30`, `dist_high`, `dist_ema`, `abs_dist_ema`,
`spy_vol20`, `spy_dd252`, `has_implied_quote`, `mcap_log`. Nine of its
fourteen inputs come from this block.

Fix, in order:

**Status: done 2026-09-02.** All three builders take `as_of_column`, `_stamps`
takes real per-block anchors, `panel_features` withholds a stale block, 14
regression tests added, full suite 1182 passed.

Verified by rebuilding Tier 3 twice on today's data — once with the `HEAD` code,
once with the change — and asserting the two frames are exactly equal
(116,795 × 45). That comparison, not a hash against the stored panel, is the
one that means anything: `data/curated/daily_market` was rewritten 2026-09-02
10:52 and `data/features/panel.parquet` dates from 2026-09-01 13:18, so the
stored panel is one refresh stale and differs from *any* fresh build (3 rows of
ORATS values, 2 of market cap, 509 of `or_exern_z252`). Worth rebuilding Tier 3
before the retrain in step 7 for that reason alone.

1. Give all three builders an optional `as_of_column` (default `"date"`, so the
   panel build is unchanged).
2. **Compose two ceilings, do not simply re-anchor on `as_of`.** The row read is
   `min(last row strictly before the event date, last row on or before the
   decision date)` — `panel._anchor_index`. Re-anchoring on `as_of` alone is
   wrong in both directions: the builders use a *strictly-before* rule (right
   for an event date, since a BMO event-date close is post-print), but a
   decision date is a close we would trade at, and `daily_state_frame` already
   documents that such a close's own quotes are known to us. Anchoring
   strictly-before the decision would push every BMO name one session *staler*
   than the panel and break `check_feature_equivalence`; anchoring on-or-before
   the decision alone would hand every AMC name a session it never had, which
   is a modelling change smuggled in as a refactor. The `min` gives the panel's
   value whenever the decision is at or after its anchor, and the decision's own
   row when it is earlier.
3. Have `live_features` pass `as_of` down — only for the synthetic target row;
   prior rows keep their own event date, so the panel path is untouched. No
   block has a cross-row date dependency (`signed_streak` and `ema12r_abs` recur
   over events without reading a date; `or_exern_z252` walks the daily series).
4. Make the stamp honest: the builders report the row they actually read as
   `regime_asof` / `runup_asof` / `orats_asof`, and `_stamps` takes those
   instead of deriving one date from `as_of`. Three stamps, not one — the three
   series have gaps in different places, and collapsing them would stamp the
   stalest block later than it was observed, which is the direction that hides
   a leak.
5. Close the same hole on the panel path: `panel_features` serves a stored row
   whose market block was baked at the event date, so when `as_of` precedes that
   anchor it must **withhold** the block (NaN, `meta["market_block_withheld"]`)
   rather than serve it. A model that needs it then reports MISSING_FEATURES and
   declines — the refusal STR-RUNUP already gets. No current caller is affected:
   `last_pre_print` is at or after the panel anchor for both sessions.
6. Add the regression tests: the market block **differs** between `as_of=D0` and
   `as_of=D−1` (today it does not, and that identity is the bug); the event date
   stays a hard ceiling; and a stamp taken from an anchor *after* the decision
   makes `assert_causal` raise. That last one is the point — with the stamp
   derived from `as_of` it could never exceed `as_of`, so the audit could not
   fail however stale the value was.

### A separate defect found while fixing this one — guarded and quarantined

`add_orats_features` computed `or_exern_z252` **outside** the `j < 0` guard. For
an event with no prior daily row, `j == -1`, so `exern[j]` read the *last* row of
the ticker's series — often years after the print — and the window became
`exern[0:-1]`, the whole history. A future leak, on **507 of the 116,432
populated values (0.44%)**, spread over 1989–2025: every event that precedes its
own ticker's ORATS coverage.

`checks/phase0_migration.py::verify_z252_delta` — the reference implementation
this builder is meant to reproduce — has carried `if j < 0: continue` all along.
That is what makes this an oversight rather than a definition.

**Handled 2026-09-02, in two halves, because one is not enough.**

*The guard* stops new panels carrying the leaked values: the z-score block moved
inside the `else`, so those rows are now NaN. *The quarantine*
(`engine.features.QUARANTINED_FEATURES`) stops anything consuming the values
already on disk — a Tier-3 rebuild is not free, so old panels persist. The
column leaves `PANEL_FEATURE_COLUMNS`, `_MARKET_BLOCK` and the scorer's
`_PANEL_MARKET_BLOCK`, and `Registry.validate` now rejects any entry that lists
it, mirroring the existing `LIVE_UNAVAILABLE` guard.

It stays **in** `PANEL_COLUMNS`, so the panel's shape and column order are
untouched and the Phase 0 reconciliation keeps working. Quarantine is a read
rule, not a schema change. Nothing was consuming it — no registry entry,
champion or retired, lists it, and `model_inputs` is built from a champion's own
feature list, so it never reached the board or the ledger either.

**TODO(2026-Q4), tracked in code at all four sites:** delete the column
outright, and retire the `KnownDelta` and `verify_z252_delta` in
`checks/phase0_migration.py` that exist only to reconcile it. That needs a
Tier-3 rebuild, so it wants its own change rather than a ride-along.

The gate itself is clean — `gate_midfill_str_thru` reads only
`EVENT_HISTORY_FEATURES + DAILY_STATE_COLUMNS + days_to_print + entry_cost_pct
+ dte_entry`, none of which come from the panel market block. Note one
consequence anyway: `_features` currently overwrites `mcap_log` with the panel
block's value when `entry_date >= last_pre_print` (`score.py:813`). Withholding
that block at a `D−1` decision changes `mcap_log`'s source to `daily_state`.
Small, real, and it must be reflected in retraining rather than discovered in
serving.

---

## 5. Code changes, file by file

### 5.1 `engine/structures.py` — a decision offset — **done 2026-09-02**

`Structure` carries `decision_offset: int | None = None`. Two properties read
it, so "unset means the entry close" is decided in one place rather than at
every call site:

- `decided_at` → `entry_offset` when `decision_offset is None`
- `decided_early` → `decided_at < entry_offset`

`__post_init__` rejects `decision_offset > entry_offset`. `to_dict()` always
emits the key, including when it is `None`: a spec that omits it is
indistinguishable from one written before decision offsets existed, and those
two do not mean the same thing once any structure sets one.
`put_calendar`, `straddle_through` and `straddle_runup` all take it through.

`replay._variant_label` appends `d{offset:+d}` **only when `decided_early`**, so
every label already written still means what it meant — `STR-THRU` is still
`e+0_x+1`, and the T−2 book will be `e+0_x+1_d-1`.

Then, at step 6: `straddle_through(entry_offset=0, exit_offset=1, decision_offset=-1)`.

### 5.2 `engine/calendar.py` — resolve it — **done 2026-09-02**

`resolve_offsets` gained `decision_offset` and `PrintWindow` gained
`decision_date`. Same anchor arithmetic as the others; no new rules. A
`PrintWindow` built without one fills `decision_date` from `entry_date` in
`__post_init__`, so the field is never `None` downstream.

The calendar does **not** enforce `decision <= entry`: that is a property of the
structure, and lives in exactly one place, in `Structure.__post_init__`.

`engine/replay.py` also gained, in the same step:

- `plan_events` resolves and carries a `decision_date` column
- `ReplayPlan.chain_keys` unions the decision keys in — a set, so while decision
  and entry are the same close nothing extra is loaded
- `filter_plan_by_availability` checks the decision chain and counts what it
  drops under a new `no_decision_chain` skip reason, which stays zero until a
  structure is decided early. This is a hole plugged before step 6 can fall in
  it: without it, an event with no `D−1` chain would have been planned and then
  failed silently at pricing time.

**What step 2 deliberately did not do.** `ScoreResult` did *not* gain a
`decision_date` field, and the ledger was not touched. `ScoreResult.as_dict()`
is `asdict(self)`, and the dashboard's `row_digest` hashes it — a new key
changes every row digest. That change belongs with the ledger's `SCHEMA_VERSION
→ 2` in step 10, where the digest break is accounted for once instead of twice.
Until then `result.as_of` carries the decision date, which is what every audit
in the pipeline already keys on.

### 5.3 `engine/score.py` — the core change

- `result.as_of` defaults to `window.decision_date`, not `window.entry_date`.
  **Done 2026-09-02** — inert, because the two dates are the same close for
  every structure that has not set an offset. `Scorer._structure`, which
  rebuilds the spec to pin a caller's strike or expiry, passes `decision_offset`
  through: dropping it there would make asking for a specific contract silently
  revert the trade to deciding at its entry close.
- `_price_entry` prices on the **decision-date** chain and records
  `result.quote_date`. Everything it fills — `entry_cost`, `spot`, `strike`,
  `expiry`, `dte_entry` — is now an *estimate for* the entry, and the field
  names should stop pretending otherwise: add `entry_cost_quoted` alongside,
  or at minimum carry `quote_date` into the ledger row.
- The market-block guard at `score.py:813` re-keys from `entry_date >=
  last_pre_print` to `decision_date >= last_pre_print`. With `decision_offset
  = −1` the block is withheld, exactly as it is for STR-RUNUP today — that is
  the intended behaviour, not a regression.
- `evidence_cutoff = min(as_of, entry_date)` already resolves to `D−1`. No
  change.
- `assert_decision_causal(D−1 <= last_pre_print)` passes. No change.

### 5.4 `engine/replay.py` — price two dates

`plan_events` resolves `decision_date` and adds it to `chain_keys`.
`replay_one` needs both: the decision chain to pick the contract and record the
quoted premium, the entry chain to book the fill. The trade row carries
`decision_date`, `quoted_cost`, and `entry_cost` separately — the difference
between them is the number §6 exists to measure.

**Strike rule (a real decision, not a detail).** At `D−1` the board names a
strike. At `D0` the spot has moved and the ATM strike may differ. Two defensible
rules:

- **A (recommended):** name the strike at `D−1` and buy *that* strike at `D0`
  (`StrikeSelector(kind="fixed")`). The board's number and the trade are the
  same contract. Slight moneyness drift, zero ambiguity about what you bought.
- **A′:** re-resolve ATM at `D0`. Cleaner moneyness, but the board's quoted
  premium refers to a contract you did not buy.

A is primary; A′ is a pre-registered secondary arm because it is nearly free to
evaluate off the same pull.

### 5.5 `engine/models/training/gate.py`

`build_dataset` calls `entry_feature_frame(..., as_of_column="entry_date")`.
Change to `as_of_column="decision_date"`. `entry_cost_pct` must be built from
the **quoted** cost at `D−1` (what the gate can see), while `TARGET = "ret"`
stays the realized return on the `D0` fill. That asymmetry is the whole point:
the gate learns to select on what is knowable a session early, and is scored on
what is actually earned.

`dte_entry` at `D−1` is `dte_entry@D0 + 1` by construction — deterministic, and
it is the dominant feature in this gate (EXP-114: −0.353). Do not leave it
reading the entry-date value; that is a one-day leak in the top feature.

### 5.6 `engine/dashboard/nightly.py` and `render.py`

- No extra ORATS cost. `refresh_forward_chains` already pulls
  `tradeDate = as_of` for every board ticker (~18 calls/night); under Arm A that
  *is* the decision-date chain.
- The board needs a **Trade on** column — the entry date — next to the existing
  event date, and a row filter for "decision date == today", which is the set
  you can actually act on tomorrow.
- `_offset_note` (`render.py:618`) gains decision-offset wording.
- Say the staleness out loud on the row: *"premium quoted at the D−1 close;
  you will fill at the D0 close."* §6 gives you the measured drift to put
  beside it.

### 5.7 `engine/ledger.py`

`structure` gains `decision_date`; `intended_prices` gains `quote_date` and
`quoted_cost`. `SCHEMA_VERSION` → 2. Rows already written stay valid and stay
readable — do not backfill them, the ledger is append-only and a v1 row means
"decided at the entry close", which is true.

### 5.8 Checks and tests

| file | what changes |
|---|---|
| `checks/phase1_replay.py` | entry-pricing equivalence now runs against the decision chain; trade-selection set is keyed on decision date |
| `checks/phase1_checks.py::check_feature_equivalence` | must assert panel/live agreement *at a given `as_of`*, and that `as_of=D0` and `as_of=D−1` **differ** |
| `tests/test_calendar.py` | `resolve_offsets` with a decision offset; BMO/AMC boundary cases |
| `tests/test_structures.py` | `decision_offset <= entry_offset` validation; `as_dict` round-trip |
| `tests/test_score.py` | `as_of` defaults to decision date; market block withheld at `D−1` |
| `tests/test_replay.py` | two-chain replay; variant label distinguishes the T−2 book |
| `tests/test_ledger.py` | schema v2 |
| new | the leak regression from §4.4 |

---

## 6. EXP-120 — pre-registration

Scaffold before any model runs:

```
python3 experiments/new_experiment.py \
  --title "STR-THRU on a T-2 decision: does a one-session lead survive the edge" \
  --strategy STR-THRU \
  --hypothesis "<§6.2 verbatim>"
```

### 6.1 Arms

| arm | decide | enter | strike | status |
|---|---|---|---|---|
| **A** | `D−1` | `D0` | named at `D−1`, held | **primary** |
| A′ | `D−1` | `D0` | re-ATM at `D0` | secondary |
| B | `D−2` | `D−1` | named at `D−2`, held | secondary |
| C (control) | `D0` | `D0` | ATM at `D0` | champion, unchanged |

Arm C is the incumbent re-run on the identical event set, so the comparison is
not contaminated by whatever the `D−1` pull's coverage turns out to be.

### 6.2 Hypothesis, frozen

> Retraining `gate_midfill_str_thru` — same architecture (HistGBM, unchanged
> hyperparameters), same feature list, same top-20% rule, `first_test_year=2020`
> — on STR-THRU trades whose features and quoted premium are cut off one
> trading session before entry, and whose target remains the realized mid-fill
> return on the `D0` close, preserves selection quality: `gate_lift` ≥ **+3.5pp**
> and gated win rate ≥ **0.40** on the 2020–2026 OOS window. Champion reference,
> measured on the `D0` decision: n=11,080, threshold 0.06645, base mean
> **+2.92%**/trade, gated mean **+7.34%**, lift **+4.42pp**, base win 38.6%,
> gated win **41.2%** (promoted 2026-08-30).

The +3.5pp floor is set *below* the champion's +4.42pp on purpose. A T−2
decision is a strictly harder problem — one session less information, and a
premium quoted before the fill — so demanding parity would guarantee a null
result and teach nothing. What is being bought is actionability, and the
question is what it costs. If lift lands below +3.5pp or gated win below 0.40,
the honest conclusion is that STR-THRU does not survive a one-session lead and
the strategy needs a different execution model, not a rescued threshold.

### 6.3 The measurement that matters most

Once the `D−1` chains land you can measure the quote/fill gap directly on
~26k events, with no model and no extra quota:

```
drift = entry_cost@D0 / quoted_cost@D−1 - 1
```

Report median, p10, p90, and the split by mcap bucket and by `days_to_print`.
This single distribution tells you three things at once:

- how much premium you give up by committing a session early (the run-up is
  real and it is in this number),
- what to print on the board beside every quoted premium,
- whether Arm B's extra day makes it materially worse.

Run this **before** the retrain. If median drift is large enough to swamp the
+2.92% base return, the retrain is answering the wrong question and the design
needs revisiting first.

### 6.4 Promotion

Standard rules — `experiments/promote.py EXP-120 --role gate --strategy
STR-THRU`. All green or nothing. Consistency across OOS years, not a single
headline: a lift that comes from 2022 alone is not a lift, per the program's
promotion-evidence standard.

The T−2 gate is a **new registry entry**, not an edit to
`gate_midfill_str_thru`. It gates a different decision and must never be
served against a `D0` decision date. Register it as
`gate_midfill_str_thru_t2` and have `Scorer._score_gate` select the champion by
`(strategy, role, decision_offset)`. Until it promotes, the T−2 board runs
**ungated** and says so on every row.

---

## 7. Rollout

| # | step | spends quota | reversible |
|---|---|---|---|
| 1 | ~~Fix the §4 leak; add the regression tests; rebuild Tier 3 with and without the change and confirm the two agree~~ **done 2026-09-02** | no | yes |
| 2 | ~~Land the `decision_offset` plumbing with `decision_offset=None` everywhere — no behaviour change, full test suite green~~ **done 2026-09-02** | no | yes |
| 3 | ~~`build_t2_plan --dry-run`, review the call count~~ **done 2026-09-02** | no | yes |
| 4 | ~~Execute the `D−1` pull (~3,628 calls); ingest; run the §3 coverage gate~~ **done 2026-09-02 — gate PASSED at 99.9%** | **yes** | data is kept |
| 5 | Run §6.3 drift measurement. **Decision point.** | no | yes |
| 6 | Set `decision_offset=-1` on `straddle_through`; rebuild the T−2 trade set as a distinct variant | no | yes |
| 7 | Retrain the gate, walk-forward, EXP-120 report | no | yes |
| 8 | Promote or don't | no | registry write is last |
| 9 | Board: Trade-on column, quote-date disclosure, ungated-row labelling | no | yes |
| 10 | Ledger schema v2; first live T−2 board | no | append-only |

Rollback at any point before 8 is `decision_offset=None`. The pulled chains stay
useful regardless — they are the only T−2 surface data the program has ever had.

Arm B's extra 3,630 calls are worth spending only if §6.3 shows the one-session
drift is small. If a single session already costs meaningful premium, two will
cost more, and B can be dropped without running it.

---

## 8. What you actually do, once this ships

Tonight's board, run after the close of session `X`, lists the events whose
decision date is `X`. Those are the ones you act on **during session `X+1`**,
placing the straddle into that close (MOC or late session). The board names the
ticker, the strike, the expiry, the premium quoted at `X`, and the gate's
verdict.

For a BMO print you are buying two sessions before the announcement. For an AMC
print, one session plus the trading day of the print itself. Either way the
prediction exists before the market you have to trade in opens, which is the
thing that is not true today.

Two honesty rules to hold yourself to, because the ledger will measure them:

- **The premium on the board is a quote from yesterday's close, not your fill.**
  Expect drift; §6.3 tells you how much.
- **Fill into the close.** The whole book is priced close-to-close. An intraday
  fill is a different trade from the one that was backtested, and the
  calibration report will not be able to tell you which of the two it is
  scoring.

---

## 9. Known risks

| risk | why it bites | mitigation |
|---|---|---|
| The §4 leak ships unfixed | `assert_causal` passes; the T−2 model looks as good as the T−1 one for the wrong reason | step 1 gates everything; the regression test asserts the blocks differ |
| `D−1` coverage is thinner than `D0` | retrain runs on a quietly liquidity-selected universe | 80% coverage gate, reported by year and mcap bucket |
| Quote/fill drift eats the base return | +2.92%/trade base is not thick enough to absorb much run-up | §6.3 runs before the retrain and can stop the project there |
| `dte_entry` left on the entry date | one-day leak in the gate's dominant feature | §5.5; assert `dte@D−1 == dte@D0 + 1` in the dataset build |
| Two variants collide in `trades` | `e+0x+1` means two different books | `_variant_label` carries the decision offset |
| The T−2 gate is served on a `D0` decision | a model applied to a decision it was never trained for | champion lookup keyed on `decision_offset`; ungated + labelled until promotion |
