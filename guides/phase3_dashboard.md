# Phase 3 Guide — Monitoring Dashboard & Remote Snapshot

**Objective:** always-on view of expected PnL / win rate per upcoming event ×
strategy × strike, refreshed nightly, reachable from a phone — with zero
divergence between what the UI shows and what the engine computes.

---

## 1. Architecture: renderer-first

The core design decision: **the static bundle is the product; the server is a
convenience.** One renderer produces a self-contained bundle from scorer
output; the local FastAPI app serves that same bundle plus desk-only actions;
the publisher ships the same bundle to the remote host. There is exactly one
rendering path, so the phone view, the desk view, and the engine can't drift.

```
engine/dashboard/
  render.py        # score_calendar() output -> snapshot bundle
  selfcheck.py     # bundle vs direct engine.score diff
  publish.py       # bundle -> Cloudflare Pages/R2 (or S3), atomic
  nightly.py       # orchestrator: refresh -> rescore -> ledger -> render -> check -> publish
dashboard/earnings/           # served + published bundle
  index.html  assets/app.js  assets/app.css
  data/board.json  data/tickers/{T}.json  data/meta.json  data/health.json
dashboard/earnings_app.py     # FastAPI, port 8712 (leave the semis scanner on 8711 untouched)
```

- **Client:** vanilla JS + CSS, no npm, no CDN dependencies (must open from
  `file://` — that's also the offline test). Board table: sortable columns,
  filter by strategy/gate. Explorer: strike × expiry grid per ticker colored
  by expected PnL, hover for win rate/CI/n_analogs; render both layers
  (model + analog) side by side; `extrapolated` and `THIN_ANALOGS` visibly
  badged, `LAYER_DISAGREE` rows flagged. Views for model health and (Phase 6)
  the tripwire board hang off the same bundle.
- **board.json:** rows = event × strategy with the full ScoreResult fields —
  the UI displays, it never computes. **tickers/{T}.json:** the strike/expiry
  grid + the ticker's evidence panel (its print history, implied-vs-realized,
  past analog trades). **meta.json:** as_of, snapshot_hash, model_versions,
  quota remaining, per-source freshness. **health.json:** calibration drift,
  live-MAE-vs-implied series, last self-check result.

## 2. Nightly job (`nightly.py`, via cron)

Order matters; each step gates the next:
1. Refresh Tier 1/2 for calendar names (through the fetch wrapper; budget
   from the 3k/month reserve; ORATS quota guard active).
2. Validation battery on the fresh data — red stops the pipeline (yesterday's
   snapshot stays published; a flag email/file is raised).
3. `score_calendar(as_of=today)`.
4. Append predictions to the Phase 4 ledger (BEFORE rendering — the frozen
   record is the point).
5. Render bundle; run `selfcheck.py`: re-score 20 random board rows directly
   via `engine.score` and diff — any mismatch stops publish.
6. Publish atomically: upload to a staging prefix, then flip (or rely on
   Pages' atomic deploys). Never leave a half-written remote bundle.
7. Emit flags: new gate triggers, earnings-date changes, calibration drift
   past threshold, quota below reserve.
8. Backup sync: push code to the public repo (hygiene hook enforced) and
   mirror `ledger/`, reports, and `config/thesis/` to the private remote —
   the ledger is append-only and irreplaceable, so a dead laptop must lose
   at most one day of it. Sync failure raises a flag but does not block the
   publish (the snapshot and the backup are independent concerns).

Cron on WSL2: verify `service cron status` and document the entry in the
bundle's meta; the job must be idempotent (safe to re-run manually after a
missed night — it re-reads, re-renders, re-publishes).

## 3. Remote access

Per the plan: primary = published static snapshot; secondary = optional
cloudflared named tunnel to :8712 for desk-time interactive use.

- **User tasks (cannot be done by the agent; provide this checklist):**
  create the Cloudflare account/project, add the Pages/R2 target, configure
  Cloudflare Access (email one-time-code policy for the user's address) in
  front of BOTH the Pages site and the tunnel hostname, and put the publish
  token in `.env` (`CF_PAGES_TOKEN` or R2/S3 credentials). The agent wires
  `publish.py` to whichever target exists and verifies with a test upload.
- **Hard rule:** publish.py refuses to run if the target is confirmed
  publicly readable without Access (probe: unauthenticated GET of meta.json
  must NOT return 200 after setup). Positions + licensed ORATS-derived data
  never ship unauthenticated.
- The live server binds 127.0.0.1 only; the tunnel is the sole remote path
  to it. `POST /refresh` (and anything quota-spending) exists only on the
  local app — the published bundle has no mutating endpoints by
  construction.

## 4. Constraints

- The UI never computes a number — it renders ScoreResult fields. Any
  derived display value (e.g. rank) is computed in render.py so the
  self-check covers it.
- Bundle must stay small enough to load on mobile data: board.json < ~1 MB;
  ticker files loaded lazily per click.
- No secrets, `.env` values, or internal paths inside the bundle (grep the
  bundle for `token`, `key`, `/root/` as a publish precondition).
- CAL-P rows render as "unvalidated — pending EXP-101/102" until promotion,
  matching the scorer's disabled state.

## 5. Acceptance tests (`checks/phase3_checks.py`)

1. **Self-check:** render → selfcheck green; then poison one board.json row
   → selfcheck fails AND publish refuses.
2. **Offline bundle:** open index.html from file:// with network disabled —
   board and one ticker view render fully.
3. **Atomicity:** kill publish mid-upload → remote still serves the previous
   complete snapshot (verify meta.json as_of unchanged).
4. **Validation gate:** inject a red validation result → nightly stops at
   step 2, previous snapshot intact, flag raised.
5. **Historical dry-run:** run nightly.py replayed over 5 past dates
   (as_of override) → 5 coherent snapshots, ledger rows appended once each,
   idempotent on re-run.
6. **Secret scan:** bundle grep clean.
7. **Access check (user):** phone loads the URL through the Access login;
   unauthenticated curl gets a login page, not data. Recorded in the Phase 3
   report as a manual check with date.

## 6. Failure modes

- Machine asleep at cron time → job runs at next wake (anacron-style guard:
  nightly.py checks last successful as_of and backfills the ledger gap
  honestly — marked LATE, never fabricated as on-time).
- Publish target down → local bundle still renders; retry next night; flag.
- Quota reserve exhausted → refresh degrades to cached data; meta.json
  freshness makes the staleness visible instead of silent.
