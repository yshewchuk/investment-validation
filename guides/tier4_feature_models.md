# Tier 4 Guide — Feature Models

**Version:** 1.0 · **Date:** 2026-09-04 · **Owner:** YS + Claude
**Status:** design, agreed on paper, no code written.
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
| `model_id` | which registry entry produced it (`size_v1_4`) |
| `fold_start` | first date of the period this row's model was fit BEFORE |
| `tier3_snapshot` | the Tier-3 hash the training data came from |

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

1. `engine/data/features/tier4.py` — build with `--since`, monthly folds,
   provenance columns. Standalone and testable before anything consumes it.
2. Registry `tier` / `produces` / `consumes`, and the snapshot covering Tier 4.
3. `load_panel(with_forecasts=True)` — the read-time join, so consumers opt in.
4. Scorer: pass `structure_params` from the Tier-4 forecast. The plumbing
   already exists (`ScoreRequest.structure_params`, commit `65037e0`).
5. An arithmetic gate for TWIN-P (`c < w`, spread, mcap) — the scorer currently
   assumes gates come from the registry, so this is its own small mechanism.
6. Enable TWIN-P; the ledger starts recording forward evidence.

Steps 1–3 are worth doing whether or not TWIN-P ever earns its place.

## 9. Acceptance tests

- **A Tier-4 row never sees its own period.** For a sample of rows, refit on
  `< fold_start` and reproduce `pred_abs_move` exactly. This is the test that
  catches a final-refit leak, which `assert_causal` structurally cannot.
- **`--since` is equivalent to a full rebuild.** Build in full; build again
  from a mid date; the frames are identical. Non-negotiable — if incremental
  and full disagree, every backfill silently corrupts.
- **Live and historical agree.** The scorer's forecast for an event in month M
  equals the Tier-4 value for the same event, byte for byte.
- **Tier 3 is unchanged.** Its hash before and after Tier 4 exists is
  identical. If Tier 4 can move Tier 3's hash, the layering has failed.
- **NULL is not zero.** A consumer given a row with no forecast declines to
  size a structure rather than sizing it at zero.

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
