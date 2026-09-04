# Tier 4 Guide — Feature Models

**Version:** 1.0 · **Date:** 2026-09-04 · **Owner:** YS + Claude
**Status:** built, all six steps. Table on disk (116,795 rows / 85,618
forecasts / 164 folds), TWIN-P enabled on the live board behind an arithmetic
entry rule. `python3 checks/tier4_checks.py` — 10/10 GREEN, 2026-09-04:
causality and live/historical agreement both max |diff| 7.1e-15 over 200 rows,
`--since` equivalence exactly 0, a full-sample refit diverges by 1.75pp (so the
leak test can see a leak), and the Tier-3 snapshot hash is byte-identical
before and after Tier 4 exists — verified across a COMPLETE Tier-4 rebuild,
which is the strongest form of that test available. Bands on 85,312 of 85,618
forecasts, median width 10.51pp, median residual SD 4.64pp.
**Objective:** make a model's forecast available as a feature — causally, and
without making Tier 3 depend on a model.

---

## 1. Why this exists

EXP-125 sized a structure's strikes from the size model's predicted move. That
turned out to be the only lever in five experiments that moved the number that
mattered (share of prints landing beyond a wing, 37.5% → 16.7%). Putting it on
the live board hit a dependency cycle:

```
structure needs w  →  w = pred_abs_move/150  →  prediction needs features
   ↑                                                      │
   └────── pricing needs the structure ←── _features needs entry_cost, spot, dte_entry
```

The break is that the size model's fourteen features are **entirely
pricing-free** — all Tier-3 panel columns, none from a chain. So the forecast
was never downstream of pricing; it was only *computed* there. Materialising it
as data removes the cycle rather than reordering around it.

The general capability: **any structure whose shape depends on a forecast
becomes expressible on the live board.** TWIN-P is the first case, not the
reason.

## 2. Where it goes, and why not in Tier 3

The first design put `pred_abs_move` into Tier 3. That was wrong. Tier 3 is a
deterministic function of Tier 2, and `data_snapshot` pins it — so every
experiment's provenance rests on that hash. Adding a model would have made a
champion promotion silently invalidate the provenance of experiments that never
used a forecast.

```
TIER 1  raw vendor files, as fetched. Never edited.
   │
   ▼    engine/data/normalize/ — units and convention traps fixed once
TIER 2  five schema-enforced tables (securities, earnings_events,
   │    daily_market, option_chains, trades)
   ▼    engine/data/features/panel.py
TIER 3  the causal event panel. Deterministic from Tier 2.
   │    LEAK RULE: nothing in row k uses event k onward — enforced by DATES
   ▼    engine/data/features/tier4.py            (new)
TIER 4  Tier 3 + feature-model outputs, keyed on (ticker, event_date)
        LEAK RULE: fit on events strictly before the period — enforced by FOLDS
```

Two different causality disciplines with two different failure modes, each
auditable by its own machinery instead of conflated in one layer. An experiment
pins whichever tier it actually depends on.

**Tier 4 is NARROW.** Keys, model outputs, provenance — not a copy of Tier 3.
Tier 3 stays the single authority for its own columns, and the join at read
time makes the dependency visible rather than something a reader has to
remember.

## 3. The table

Grain: one row per `(ticker, event_date)` — the same grain as Tier 3.

| column | meaning |
|---|---|
| `ticker`, `event_date` | join key into Tier 3 |
| `pred_abs_move` | the forecast, in percent, walk-forward |
| `pred_abs_move_p10` / `_p90` | the 80% band around it |
| `pred_abs_move_sd` | SD of the held-out errors the band came from |
| `resid_n` | how many held-out errors that was |
| `model_id` | which registry entry produced it (`size_v1_4`) |
| `fold_start` | first date of the period this row's model was fit BEFORE |
| `tier3_snapshot` | the Tier-3 hash the training data came from |

### The interval, and why it is stored rather than computed

A point forecast without a width is half an answer, and the dashboard already
shows an 80% band — but it draws that band from the *registry artifact's*
residual pool, computed fresh at score time. That was fine while one model
served everything. It is not fine now: with TWIN-P sized by a monthly fold
model, the centre would come from one fit and the width from another.

`bucket_residuals` says the prediction↔residual pairing is "only available at
training time, because by the time an artifact is loaded the predictions are
gone." **The Tier-4 build is exactly that moment** — it holds every fold's
out-of-sample prediction next to the realized `abs_move`. So the band is
computed there, from the same fold model that produced the centre, and stored.

Three properties it inherits and one it adds:

* **Same fold causality.** At fold `F` the pool holds the errors of every
  EARLIER fold and nothing else. A band built from errors the model had not yet
  made is the same leak `fold_start` exists to prevent, wearing a different hat
  — and invisible to every check on the point forecast, which is why
  `checks/tier4_checks.py::interval` reproduces a fold's band by hand.
* **Same `--since` equivalence.** An incremental build seeds the pool from the
  carried prefix, whose stored forecasts *are* the earlier folds' predictions.
* **Conditioned, not flat.** Reuses EXP-115's decile buckets rather than
  reimplementing them, so a large forecast gets a wider band than a small one;
  a thin bucket falls back to the flat pool.
* **Floored at zero — BOTH bounds.** The target is a magnitude. Flooring only
  the lower bound was the first version and it inverts the band whenever the
  point estimate itself sits below zero. No real forecast does (0 of 85,618,
  minimum +0.39), but the synthetic fixture's linear target can, which is how
  it surfaced — and an inverted interval on the board would have been read as
  data corruption rather than as a clipping artefact.

Below `MIN_RESIDUALS` held-out errors there is no band at all. An 80% interval
from forty errors is not a distribution, and a number shaped like a confidence
interval is read as one.

`model_id` and `fold_start` are not decoration. **A partially rebuilt Tier 4
must be distinguishable from a complete one**, and a row that cannot say which
model and which fold produced it cannot be checked.

Rows before the first predictable period carry a NULL forecast. Consumers must
treat NULL as "no forecast", never as zero — the TWIN-P entry filters already
drop those rows, but that is a property to state rather than inherit by luck.

## 4. Causality: fit on strictly-before

For a row whose event falls in period `P`, the model is fit on events with
`date < P.start`. Not on `< event_date` — see §5 — and never on the full sample.

The failure this prevents is subtle and would survive the existing audit:
`assert_causal` compares FEATURE STAMPS against `as_of`, so a leak inside a
model's *training set* is invisible to it. A Tier-4 row built from the final
refit model would look perfectly causal and be worthless. Hence `fold_start`
in the row, and hence the acceptance test in §9.

## 5. Cadence: monthly

Strict causality needs *train on < D*. It does not need *retrain at every D*.

| cadence | fits | full rebuild | training data forgone at the margin |
|---|---:|---:|---|
| yearly (today's walk_forward) | 14 | ~1 min | up to 12 months |
| **monthly (chosen)** | **~168** | **~15 min** | **up to 1 month** |
| per-date | ~3,500 | 6–10 hours | none |

Monthly forgoes at most one month of new events — a few hundred out of a
~90,000-event training set — on a model whose OOS MAE moves by hundredths
between folds. Per-date's real risk is not compute: it is that **a 6–10 hour
rebuild does not get run**, and a panel that drifts from its own definition
costs more causality than the extra recency ever bought.

The cadence is a registered parameter, not a constant, so tightening it later
is a spec change rather than a rewrite.

### The property monthly buys

The *current* month's model — fit on everything before the month began — is
the same artifact the live scorer needs for an upcoming event. So the
historical Tier-4 row and the live board prediction for the same month come
from the **identical fitted model**, agreeing by construction rather than by a
test that hopes they agree.

Consequence for storage: only the current month's artifact needs persisting.
The historical ones are deterministic given the seed and the Tier-3 snapshot,
so they regenerate on rebuild instead of accumulating ~168 × 1.5MB of joblib.

## 6. Invalidation — the operational contract

| what changed | rebuild |
|---|---|
| Tier 2 data at date D | Tier 3 from D forward, then Tier 4 from D forward |
| Tier 3 build code | Tier 3 in full, then Tier 4 in full |
| feature-model code or seed | Tier 4 only |
| cadence | Tier 4 only |
| a decision model (gate) | neither — gates consume Tier 4, nothing produces into it |

**A backfill cascades forward.** Correcting data at date D changes the training
set of every period from D onward, so every forecast from D onward is stale —
not just the rows whose inputs changed. This is the expensive property and it
is not avoidable: it is what "fit on strictly-before" means.

The build therefore takes `--since` and is **idempotent and resumable**: rows
before `since` are untouched, rows from `since` forward are recomputed.

## 7. The registry gains a tier

The registry tracks `(id, role, strategy, features, target)` but cannot express
**which models feed which**. Adding two fields makes the dependency graph
answerable from data rather than memory:

```
tier:      "feature" | "decision"
produces:  the Tier-4 column, for feature models      (size_v1_4 → pred_abs_move)
consumes:  Tier-4 columns read, for decision models
```

So "what breaks if I re-promote the size model" stops being a question someone
has to remember the answer to.

Feature definitions are already shared (`EVENT_HISTORY_FEATURES`,
`DAILY_STATE_COLUMNS` in `engine/features.py`) as is the walk-forward machinery
(`training/common.py`). The per-model `FEATURES` tuples stay in their own
modules deliberately — that separation IS the record of what is used where.

## 8. Build order

1. **DONE** `engine/data/features/tier4.py` — build with `--since`, monthly
   folds, provenance columns. Wired into `engine.data.rebuild` after `panel`,
   with `--tier4-since` for the follow-up a Tier-2 correction requires.
2. **DONE** Registry `tier` / `produces` / `consumes`, plus `producers()`,
   `consumers()` and `tier4_graph()`. The Tier-4 hash lands in `SNAPSHOT` as
   `tier4_sha256`, deliberately *beside* `snapshot_hash` and not inside it —
   see §10.
3. **DONE** `load_panel(with_forecasts=True)` — the read-time join, opt-in.
4. **DONE** Scorer sizes the structure from the forecast, *before* pricing —
   `engine/forecast_sizing.py` plus `Scorer._size_from_forecast`. The forecast
   comes from `tier4.serving_model(serving_fold(...))`, the same `fit_fold`
   the table was built with.
5. **DONE** `engine/entry_rules.py` — arithmetic gates, applied in
   `Scorer._apply_entry_rule` only where the registry has no gate, so a
   promoted model always wins.
6. **DONE** TWIN-P removed from `DISABLED_STRATEGIES`. The readiness test in
   `tests/test_score.py` now accepts an entry rule alongside a promoted gate;
   that is a widening of "something decided it is ready", not a loophole — an
   arithmetic rule decides *every* row, which is more coverage than a gate.

Steps 1–3 are worth doing whether or not TWIN-P ever earns its place.

### What the build settled that the design left open

* **Scorable and trainable are different sets.** A prediction needs complete
  features; only training also needs a realized target. An event that has not
  printed yet has no `abs_move` and must still get a forecast — that is the
  entire point of materialising one. Conflating the two would have left every
  upcoming event, the only ones still tradeable, holding a NULL.
* **The table is TOTAL over Tier 3.** Every Tier-3 event gets a row, NULL where
  no forecast was possible. If Tier 4 held only the rows it could predict, a
  *missing* row would be ambiguous between "no forecast available" and "this
  table is stale", and only one of those is acceptable to a consumer.
* **The carry-over is guarded three ways** (`_carried_prefix`): a cadence
  change, a different `model_id`, and Tier-3 events appearing inside the
  retained prefix each refuse `--since` rather than stitch together two halves
  that answer to different definitions of causality.
* **`--since` rounds DOWN to its fold boundary**, because a fold is the unit of
  recomputation: half a month cannot be rebuilt without fitting the model its
  other half already used. The rounding recomputes a superset, which is
  identical to what was there, so the §9 equivalence holds either way.
* **A leak test needs a leak to catch.** `test_a_model_fit_on_everything_would_
  fail_that_test` refits on the full sample and asserts the stored values no
  longer reproduce. Without it, the causality test could pass by being unable
  to distinguish anything at all. `checks/tier4_checks.py::leak_is_detectable`
  is the same assertion against the real champion.
* **The width an entry rule tests is the LISTED one, not the target.**
  `_structure_width` reads `w` off the priced legs. This is not fussiness: on
  the first live board, IRS carried a 3.97% forecast — a $0.40 target width on
  a $15.23 stock — and its ladder lists $2.50 steps, so the tent came out 6×
  wider than asked. A rule comparing cost against the *requested* width would
  have been testing a trade that does not exist.
* **TWIN-P gets no payoff map, deliberately.** Its exit value is a function of
  the realized move, but a twin-peaked one, and `PAYOFF_DRIVER`'s linear
  `intercept + slope × driver` cannot represent a shape that rises, falls and
  rises again. It scores `NO_PAYOFF_MAP` and is decided by the entry rule
  alone. That is a real limitation of the model layer for this structure, and
  it is recorded in `engine/payoff.py` rather than papered over with a map
  that would happily fit.

## 9. Acceptance tests

- **A Tier-4 row never sees its own period.** For a sample of rows, refit on
  `< fold_start` and reproduce `pred_abs_move` exactly. This is the test that
  catches a final-refit leak, which `assert_causal` structurally cannot.
- **`--since` is equivalent to a full rebuild.** Build in full; build again
  from a mid date; the frames are identical. Non-negotiable — if incremental
  and full disagree, every backfill silently corrupts.
- **Live and historical agree.** The scorer's forecast for an event in month M
  equals the Tier-4 value for the same event. MEASURED, 2026-09-04: 400 events
  across 119 folds, max |live − table| = 3.55e-15, 230 of 400 bit-identical.
  Not byte-for-byte, and the reason is worth keeping: the MLP half of the blend
  is not associative across batch shapes, so the *same fitted model* returns
  values a few ULPs apart depending on how many rows it scores at once. The
  claim that holds is "one fitted estimator per fold, reached through
  `fit_fold` by both paths" — which is the property that matters, and which
  a bitwise assertion would have failed for a reason unrelated to causality.
- **The fold served live is never one that has not begun.** `serving_fold`
  returns the earlier of the event's fold and the decision's fold, so an event
  in next month is sized by THIS month's model rather than by one fit on events
  that have not happened. Agreement above is therefore exact for a trade
  decided inside the event's own month, which on a three-week board is the
  normal case, and the score records `forecast_fold` when it is not.
- **Tier 3 is unchanged.** Its hash before and after Tier 4 exists is
  identical. If Tier 4 can move Tier 3's hash, the layering has failed.
- **NULL is not zero.** A consumer given a row with no forecast declines to
  size a structure rather than sizing it at zero.
- **The band is ordered, in support, and causal.** `p10 <= p90`, both at or
  above zero, and one fold's band reproduces from the residuals of earlier
  folds alone.

## 10a. First live board, 2026-09-04

Measured on the real 21-day calendar the day the wiring landed. 153 rows, 76
forecasts produced, 24 priced, the entry rule decided on 23 — and **none
passed**.

That is not a defect; it is EXP-125's finding reproducing live. `cost / w` ran
1.07 to 1.98 across all 24 priced rows: the debit is above the tent width on
every one. Forecast-sizing fixes the geometry and collapses the universe, which
is what EXP-125 measured when `c < w` cut it to 90 events.

The mechanism is now visible in the data rather than inferred. Sizing to a
forecast pushes the wings to ±4w ≈ ±16-20% of spot, into strikes whose relative
spreads run 20-150%; and on a coarse ladder the realized `w` overshoots the
target several-fold. Both terms then fail together.

**What this buys is forward evidence at zero risk of a bad fill**: the ledger
records what the rule would have taken, on names it declines, every night. If
`c < w` is genuinely unreachable on tradeable names, that shows up as a long
run of empty books rather than as an argument — and that is a cheaper way to
learn it than trading it.

## 10. What this costs, stated plainly

- Backfills get materially more expensive: a corrected date rebuilds every
  forecast after it, not just its own row.
- The programme gains a second provenance hash. An experiment that reads
  forecasts pins Tier 4; one that does not pins Tier 3. Reports must say which.
- A champion promotion becomes a Tier-4 rebuild event, so promotion stops being
  free.

None of these is avoidable while forecasts are features. They are the price of
the capability, and they are cheaper than the alternative — recomputing a
forecast inside the scoring path and discovering, as EXP-125 did, that the
ordering makes it impossible.
