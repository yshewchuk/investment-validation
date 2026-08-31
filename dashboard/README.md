# Semi Index-Flow Scanner Dashboard

Web UI for the semiconductor index-flow dispersion strategy (see ../STRATEGY.md).

## Run

    python3 refresh.py     # rebuild snapshot (calendar + prices + EDGAR PIT fundamentals)
    python3 server.py      # serves on http://localhost:8711

## Views

- **Weekly Planner** — upcoming earnings board: each candidate's PIT fundamentals,
  21d run-up, and live gate evaluation (TRIGGERED / conditional / blocked, with
  per-condition pass/fail detail). Full universe table below, sorted by run-up.
- **Ticker Deep-Dive** — per-ticker evidence trail: gate checklist, historical stats,
  run-up vs reaction scatter, earnings-return history, PIT valuation timeline, and
  every event in raw form.
- **Strategy Log** — walk-forward backtest results, equity curve, triggered trade log.

## Data sources

- Prices/earnings calendar/options: Yahoo Finance (yfinance)
- Point-in-time fundamentals: SEC EDGAR companyfacts (quarterly, filing-date aware,
  cached 7 days in dashboard/cache/facts_*.json)
- Historical events: bt/out/events_pit.csv (1,042 events, Jun 2021 – Aug 2026)
- Backtest trades: bt/out/strategy_trades.csv (walk-forward test period)

## Notes

- Foreign filers without quarterly XBRL (STM, TSM, UMC, etc.) show n/a fundamentals
  and never trigger gates.
- SWKS is merger-excluded by design.
- The Refresh button in the UI runs refresh.py synchronously (1–3 min).

---

# Earnings-vol monitoring board (Phase 3)

The second dashboard in this directory, on **port 8712** — the semis scanner
above keeps 8711 and is untouched by any of it. Different data, different
engine, deliberately separate process.

## Architecture: the bundle is the product

One renderer turns scorer output into a self-contained bundle; the local
FastAPI app serves that same bundle; the publisher ships that same bundle to
the remote host. Exactly one rendering path, so the phone view, the desk view
and the engine cannot drift.

    engine/dashboard/render.py      score_calendar() output -> bundle
    engine/dashboard/selfcheck.py   bundle vs a direct engine.score re-score
    engine/dashboard/publish.py     bundle -> target, atomic, access-checked
    engine/dashboard/nightly.py     the orchestrator (below)
    dashboard/earnings/             the bundle: index.html, assets/, data/
    dashboard/published/            local publish target (releases/ + current)
    dashboard/earnings_app.py       FastAPI, 127.0.0.1:8712

Every datum in `data/` travels twice — as `.json` (what the self-check, the API
and any other consumer read) and as a `.js` wrapper generated from the same
bytes, because `fetch()` is blocked on `file://` origins and the bundle has to
open from a filesystem with the network off. Nothing in the client computes a
number: it formats fields the renderer wrote, so the nightly self-check covers
every value on screen, including derived ones like rank and premium-vs-fair.

## Run

    # one night, no network, no publish — the safe way to look at it
    python3 -m engine.dashboard.nightly --no-refresh --no-publish

    # the real thing (spends ORATS quota from the 3k/month live reserve)
    python3 -m engine.dashboard.nightly

    # serve it at http://127.0.0.1:8712
    python3 dashboard/earnings_app.py

    # in a container, where a published port cannot reach the loopback bind
    DASHBOARD_HOST=0.0.0.0 DASHBOARD_PORT=8711 python3 dashboard/earnings_app.py

    # verify a bundle against the engine by hand
    python3 -m engine.dashboard.selfcheck --bundle dashboard/earnings

    # pack the whole board into ONE file that opens anywhere, no server
    python3 -c "from engine.dashboard.render import write_single_file as w; print(w('dashboard/earnings'))"

That last one inlines the bytes the renderer already wrote — same `app.js`,
same payloads, every ticker pre-loaded so the explorer still works — so it
cannot show a number the bundle does not. ~2 MB for a 148-event board, and it
is the fallback when the server is awkward to reach.

    # pack the whole board into ONE file you can open anywhere
    python3 -c "from engine.dashboard.render import write_single_file as w; print(w('dashboard/earnings'))"

The last one exists because the server is not always reachable — a container
without a published port, a phone, a machine that is asleep. It inlines the
bytes the renderer already wrote (same `app.js`, same payloads, every ticker
pre-loaded so the explorer still works), so it cannot show a number the bundle
does not. ~2 MB for a 148-event board.

Useful flags: `--as-of YYYY-MM-DD` (replay a past night), `--tickers A,B,C`
(bound the universe), `--alt-strikes N` (strikes either side of ATM, scored
for gate passers only), `--horizon N`, `--backup` (git push + private mirror).

**Timing.** Scoring is ~1s per event and the full calendar is ~2,200 events in
a three-week window, so a full unrestricted night is roughly 40 minutes. That
is why the strike ladder is priced only for rows the gate passed, and why the
job is a post-close cron rather than something you wait on.

## Two areas

**Trades** — upcoming prints, and the ticker/strike explorer. Finding something
to do.

**Models** — the model explorer (below), "how a number is made", and model
health. Deciding whether to believe it.

## Taking a number apart

Every board number has an audit trail, in two halves:

- **"How a number is made"** (a tab) reads `data/strategies.json`: the structure
  and its legs, when it opens and closes relative to the print, what the model
  predicts, the champion's registry id, training window and walk-forward OOS
  metrics, and **every input it consumes with a one-line explanation**. It is
  read off the registry and `STRUCTURES` at render time, so it cannot describe a
  pipeline the engine no longer runs.
- **Per-row inputs** live on the row itself as `model_inputs` — the exact values
  the champion consumed for THAT event, with the date they were read as of, plus
  the fitted payoff line (`exit/spot = intercept + slope x driver`) and the
  analog bucket the comparison set was matched on. Click a cell in the ticker
  explorer.

Inputs are recorded even when a row declines to score, because that is exactly
when someone needs to see which one was missing.

### The model explorer

    python3 -m engine.dashboard.model_evidence          # cached on the artifact hash

Per champion: what it is (a blend, a tree ensemble), what it predicts, how many
rows it learned from — and per input, the **Spearman and Pearson correlation
with the outcome**, the **decile shape** (the input cut into ten buckets with
the mean outcome in each), and coverage. Both correlations because a monotone
but curved relationship shows in Spearman and hides in Pearson; the decile table
because neither number can show whether a relationship is monotone, flat in the
middle, or driven by one tail.

**Non-monotone inputs are the reason two readings are shown.** A signed input
against a magnitude outcome runs high at both ends and low in the middle, and
BOTH correlations score that at approximately zero. `mean_prior_move` against
`abs_move` in the size model goes 8.35 → 4.60 → 7.82 across its deciles on a
Spearman of +0.013. So each input also carries **`magnitude_spearman`** — the
correlation of *distance from the middle* with the outcome, +0.185 for that
one — and **`decile_range`** (best-to-worst) beside `decile_spread`
(end-to-end). Inputs are ranked by the larger of the two readings, so a V no
longer sorts to the bottom, and the UI badges it `V` and says so in words.

The scatter shows the sampled rows, the straight line a linear model would fit
(red) and the mean outcome per decile (green) on the same axes. That pairing is
the point: where the line is flat and the decile curve is a V, the relationship
is real and a linear reading cannot see it.

Read as description, not attribution: these are marginal relationships in the
training set. A feature can correlate strongly and add nothing once the others
are present, or correlate near zero and matter through an interaction. The
caveat ships with the data and is rendered on the page.

Rebuilt only when a champion changes, not nightly — it rebuilds each model's own
training set. Two constraints found the hard way: `store.read_table("daily_market")`
is 8.9M rows and peaked at **6.9 GB** on a 7 GB box, so the daily table is
streamed per partition and filtered to the tickers a model needs; and the
`implied_t1` set is 577k rows built in a Python loop, so its EVENTS are sampled
before the build (recorded in the output, never silent).

Feature explanations live in `engine/features.py` (`FEATURE_NOTES` /
`feature_note`), next to the definitions. The `_dN` lag and `emaN_prior_*`
families are derived from the same rule that generates them, so a new lag
documents itself; an undocumented feature renders bare rather than getting an
invented explanation. An acceptance check fails if any champion input has no
note.

## Where the calendar comes from

The board scores confirmed events in the next three weeks, and getting that
list is its own problem: **ORATS `/hist/earnings` is a history endpoint.** Its
payloads stop at the last print that already happened, so no amount of
re-pulling it produces an upcoming event. A refresh built on it returns 200,
carries real rows, and leaves the board permanently empty.

Three sources, with different jobs:

| source | what it gives | cost |
|---|---|---|
| **Nasdaq** `calendar/earnings?date=` | who reports on a date — the whole market, keyless | 1 call per trading day (~15 for three weeks) |
| **yfinance** `get_earnings_dates` | the announcement TIME, hence BMO/AMC | ~0.9s per ticker, run only where the session is unknown |
| **ORATS** `anncTod` | the authority, once the print has happened | batched, only for names that printed in the last 10 days |

Session priority is **ORATS > yfinance > Nasdaq**, and every row records which
one supplied it in `session_src`. The order is by what each can be held to
account for, not convenience:

- ORATS agreed with the oquants panel on 99.52% of dates (EXP-038), but only
  ever sees the past.
- yfinance keeps the announcement time on *historical* rows, so its session is
  gradeable after the fact: **99.72% agreement with ORATS `anncTod`** across
  716 overlapping events (measured 2026-08-30).
- Nasdaq returns `time-not-supplied` for almost every past date (1,392 of 1,395
  sampled), so its session can never be graded retrospectively. It agreed with
  yfinance on 99.15% of the 117 forward events where both spoke — reassuring,
  but not the same evidence.

**Coverage is partial and self-healing.** Nasdaq states a session for ~52% of
forward rows overall but 75% across the 1–10B and >10B slices this program
trades, and the share rises as the print nears (9% at 2–3 weeks out, 54% inside
a week). yfinance lifts the combined figure to ~66%. An event with no session
is *skipped* by the scorer rather than guessed at — a wrong BMO/AMC shifts
every entry and exit by a day — and it gets picked up on a later night once a
source firms up.

**Disagreements are flagged, never resolved silently.** ~5% of forward tickers
get two different dates from the two sources. Both rows stay on the board with
`date_conflict` set, and the nightly raises `calendar_date_conflict` naming
them: one of the two is wrong, and the board says so rather than picking.

    python3 -m engine.data.pulls.forward_calendar --horizon 21   # standalone

### Why not Polygon

Probed 2026-08-30/31 on this plan: the earnings calendar is the Benzinga
add-on (`/benzinga/v1/earnings` → 403 not entitled), and for chains there are
**no quotes** — `/v3/quotes` is 403, and neither the chain snapshot nor the
unified snapshot carries `last_quote`. Polygon gives greeks, IV, open interest
and traded daily bars, which is real liquidity evidence and is what
`engine/data/pulls/polygon_fills.py` already harvests — but with no bid/ask
there is no spread for `FillModel(alpha)` to interpolate across, and the
worst/mid/best fill spread is the program's headline risk. Chains stay ORATS.

That is affordable: the board needs an entry/exit chain only for names whose
window is actually current, which is ~24 tickers a night (~509 a month) against
the 3,000-call live reserve.

### Two things that bite when running it locally

**A published Docker port cannot reach a loopback bind.** Docker forwards to
the container's bridge interface, so a server on `127.0.0.1` inside a container
is invisible from the host however the port was published. `DASHBOARD_HOST`
exists for that, and the default stays loopback so an all-interfaces bind is
never an accident — the board discloses position intent and redistributes
licensed ORATS-derived quotes, so whoever can reach the bind can read both.
Check what your publish is bound to (`-p 8711:8711` listens on every host
interface; `-p 127.0.0.1:8711:8711` does not).

**Do not leave the desk server warm while the nightly runs.** A `Scorer` is
~1.4 GB held for the process's life, and two at once exceeded a 7 GB box: the
nightly was OOM-killed with an empty log and exit 137, which looks exactly like
nothing happening. Serving the board needs no scorer at all — the bundle is
already on disk — so the warm-up is off unless `DASHBOARD_WARM=1`, and the
first `/api/score` pays the two minutes instead.

## The nightly job

Order matters; each step gates the next.

1. **Refresh**, cheapest and most load-bearing first: the forward calendar
   (unmetered, above), then one ORATS call each for market-wide summaries and
   cores, then batched ORATS confirmation for names that printed in the last 10
   days. A dead credential stops the run; any other failure degrades to cached
   data and raises `refresh_degraded` — staleness stays visible in `meta.json`
   instead of going silent.
2. **Validate** the store it is about to score from. Red stops the pipeline:
   yesterday's snapshot stays published and a flag is written.
3. **Score** the confirmed calendar, ATM, with one shared Scorer.
4. **Ledger** — the ATM board is frozen into `ledger/predictions/` *before*
   rendering. Missed nights backfill honestly (real `decision_ts`, flagged
   LATE), never fabricated as on-time.
5. **Render** the bundle, then **self-check**: re-score board rows directly
   through `engine.score` and diff both the row digest and every displayed
   field. Any mismatch stops the publish.
6. **Publish** atomically — stage a full release, then flip `current` with one
   `os.replace`. A kill mid-upload leaves the previous snapshot serving.
7. **Flags**: new gate triggers, earnings-date changes, calibration drift,
   quota below reserve. Written to `reports/phase3_flags/YYYY-MM-DD.json` and
   shown on the board.
8. **Backup** (`--backup`): git push + private mirror. A failure flags but does
   not block the publish — the snapshot and the backup are independent.

Re-running a night is safe: the fetch cache turns a same-day refresh into a
hit, and the ledger refuses duplicate row ids.

### Cron (WSL2)

    30 21 * * 1-5  cd /path/to/investing-plan && python3 -m engine.dashboard.nightly >> dashboard/nightly.log 2>&1

Check the daemon is actually running with `service cron status` — on WSL2 it
does not start by default, and a cron that was never up looks exactly like a
job that never had anything to say. If the machine was asleep, run the job by
hand: it backfills the missed nights and marks them LATE.

## Remote access

Primary channel: the published static snapshot. Secondary (optional): a named
cloudflared tunnel to :8712 for desk-time interactive use. The server binds
127.0.0.1 only, so the tunnel is the sole remote path to it, and the published
bundle has no mutating endpoint by construction — `POST /api/refresh` and
anything else that spends quota exists only on the local app.

### Checklist — steps only the account holder can do

The agent wires and verifies the publisher; it cannot create your Cloudflare
account or hold your credentials.

1. Create the Cloudflare account and a Pages project (or an R2 bucket) for the
   board.
2. Put **Cloudflare Access** in front of BOTH the Pages/R2 hostname and the
   tunnel hostname — an email one-time-code policy for your address is enough.
   This is not optional: the board discloses position intent and redistributes
   ORATS-derived quotes, which are licensed.
3. Add the publish credential to `.env` (never committed):

       DASHBOARD_PUBLISH_CMD="wrangler pages deploy {bundle} --project-name=earnings-board"
       DASHBOARD_PROBE_URL="https://earnings-board.example.workers.dev"

4. Run `python3 -m engine.dashboard.nightly --target "$DASHBOARD_PUBLISH_CMD"`
   once and confirm the release lands.
5. **Do the manual check and record it.** On a phone: load the URL, expect the
   Access login, then the board. From a machine with no session:
   `curl -sI https://<host>/data/meta.json` must NOT return 200 with data.
   Record the date in the Phase 3 report:

       python3 checks/phase3_report.py --checks-json reports/phase3_checks.json \
           --access-check-date 2026-09-05

`publish_bundle` refuses outright if its probe gets an unauthenticated 200 —
positions and licensed data never ship into the open. A 302 to an Access login
is not proof of publicness and is allowed (and recorded).

## Checks and the report

    python3 -m pytest tests/test_dashboard.py -q          # units
    python3 checks/phase3_checks.py --json reports/phase3_checks.json
    python3 checks/phase3_report.py --checks-json reports/phase3_checks.json

`--no-data` runs only the checks that need no store. The acceptance suite
leaves its working bundles in `reports/phase3_checks/` so a failure can be
opened and looked at.

## Failure modes, and what they look like

| Symptom | What happened | What to do |
|---|---|---|
| Board empty, `no_upcoming_events` flag | the forward calendar has not been pulled | `python3 -m engine.data.pulls.forward_calendar` — ORATS alone can never fill it |
| `calendar_date_conflict` flag | Nasdaq and yfinance disagree on a print date | both rows are on the board; one is wrong. Trust the one ORATS confirms once it lands |
| `validation_red`, previous snapshot still up | the store failed the freshness/coverage battery | read the flag file; do not publish around it |
| `selfcheck_red` | the bundle and the engine disagree | look at `mismatches`; a model or data change mid-run is the usual cause |
| `publish_failed` | target down, or the access probe saw a public 200 | the local bundle still rendered; fix the target, retry next night |
| `quota_below_reserve` | live ops is eating the 3k/month reserve | refresh degrades to cache until the month rolls |
| Rows badged `NO_CHAIN` | no option chain in the store for that event | expected for names outside the pulled slices |
| CAL-P rows unscored | the scorer disables CAL-P until EXP-101/102 | by design — the evidence is for a different structure |
