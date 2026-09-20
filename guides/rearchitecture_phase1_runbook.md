# Rearchitecture Phase 1 — Operator Runbook

The operational companion to [Phase 1 — Operations](rearchitecture_phase1_operations.md)
(§8.1, §10.4, §14, §15) and [Phase 1 — decisions](rearchitecture_phase1_decisions.md)
(D1–D8). Every command, flag and path below was verified against the source on
2026-09-13. Where a step is not yet activated, it says so explicitly.

## 1. Scope and activation state

**What exists.** A shadow-only operations layer around the legacy research
engine: a durable SQLite job catalog with leases, dependencies, retries and
fenced cancellation; immutable planning and submission; a resource policy
(`ops_resources.2026-09-12.v1`, `engine/v2/ops/profiles.py`); a watchdog
executor (D7: `executor_mode=watchdog`, `containment=best_effort` — cgroup2 is
mounted read-only on this host, so kernel containment is unavailable); the
operator CLI `python3 -m engine.v2.ops`; the `operations_health.v1.0` document
(`engine/v2/ops/health.py`); the authenticated serving shell
(`engine/v2/serving/operations.py`); and a catalog-backed decision authority
**built and rehearsed against copies only** (D4).

**What is NOT activated.**

- Production decision authority. D4: switching the production writer is §14.2
  step 5 and needs separate explicit operator approval. Until then the legacy
  writer owns `ledger/`.
- `--mode` accepts only `shadow`; `nightly_plan` refuses anything else with
  "production cutover has not been activated".
- Timers. `init` installs none, and the host has no service manager (PID 1 is
  bash under WSL; no systemd unit, no crontab — decisions §3). Everything here
  is started by an operator.
- Remote publication. The serving shell is local and read-only; releases are
  staged through the transactional outbox and are not delivered anywhere
  remote in this phase.
- Registered experiment execution. `plan experiment` without `--no-ledger`
  refuses with "production experiment activation is disabled"; only the smoke
  plumbing check is submittable.

**The operations root.** `data/operations/` is the private production root. It
is gitignored by the `/data/` hard block in `.gitignore`, and `init` chmods it
`0700`. Tests never touch it — they use temporary roots. Refusals from the CLI
exit **2** and print a `problem.v1.0` JSON envelope (`code`, `category`,
`retryable`, `message`, ...).

## 2. Operator commands

All commands take the global `--root` (default `data/operations`); it may
appear before or after the subcommand. Output is always JSON on stdout.

`serve` additionally takes `--store-root PATH`: the legacy checkout a
snapshot-backed job's pinned read set and materialization roots resolve
against. Omit it and `Service` defaults to the code checkout `serve` itself
runs from (`Path(__file__).resolve().parents[3]`) — the pre-existing
behavior. Pass it explicitly whenever the supervisor runs from a frozen git
worktree or any checkout that is not itself the legacy tree with `data/`:
without it, a snapshot-backed launch refuses `INPUT_CHANGED` ("legacy input
is missing or indirect") before it ever starts, because it resolves the
pinned read set against a checkout with no `data/` to find it in. Validated
as an existing directory at parse time; never inferred from a plan or a
manifest.

| Command | Purpose | Example |
|---|---|---|
| `init` | Create the root (0700) and open the catalog. Installs no timer, starts nothing. | `python3 -m engine.v2.ops --root data/operations init` |
| `doctor` | Read-only host + catalog inspection: capacity sample, catalog existence, diagnostics (hook state, `unmanaged_processes`, environment, executor probe, profiles), `integrity_errors`, `schema_versions`, `activation`. Opens the catalog `mode=ro`; never starts jobs or changes quotas. | `python3 -m engine.v2.ops doctor --json` |
| `health` | Print the `operations_health.v1.0` document (§3). | `python3 -m engine.v2.ops health --json` |
| `serve` | Run the supervisor claim/execute loop; `--once` drains and exits when nothing is active. `--store-root PATH` points a snapshot-backed launch at the legacy checkout, when it is not the code checkout `serve` runs from (see above). | `python3 -m engine.v2.ops serve --once --store-root /path/to/legacy/checkout` |
| `capture-inputs` | The supported capture for `plan nightly --input-manifest` (2026-09-14). Enumerates and hashes, read-only, every file the six barrier-only nightly kinds (`legacy_finality`, `legacy_decisions`, `legacy_settlement`, `legacy_model_evidence`, `legacy_render`, `legacy_selfcheck`) actually read — see `engine/v2/data/legacy_nightly_read_plan.py` for the declared, cited families — and writes a `LegacyInputManifest` with `capture_implementation_ref="legacy_nightly_capture.v1"`. `--as-of`, `--tickers`, `--context-tickers` (defaults to `--tickers`), `--year-start/--year-end`, `--source-root` (the legacy checkout to scan), `--output` (where to write the manifest JSON) are all required except `--tickers`/`--context-tickers`. Refuses with a typed problem (`SOURCE_EMPTY`/`INPUT_CHANGED`) when a required family has no files, e.g. no cached ORATS market-wide files for any of the 15 finality lookback sessions. | `python3 -m engine.v2.ops capture-inputs --as-of 2026-09-12 --tickers AAA,BBB --year-start 2024 --year-end 2026 --source-root /path/to/legacy/checkout --output manifest.json` |
| `price-refresh` | Scheduled full-history yfinance price re-downloader feeding Tier-2 `price_history` (`engine.data.pulls.price_refresh`). `--session YYYY-MM-DD` (required): daily group is every ticker with a confirmed event in `[session, session+35 calendar days]` or within 5 trading sessions after one already printed; monthly group is every other price-universe ticker with no successful dated fetch this calendar month; delisted/failing tickers are never pruned, only reported. `--dry-run` reads Tier-2 `earnings_events` and the Tier-1 fetch store (disk only) and prints plan counts, making **zero** provider calls. A real run makes one yfinance call per planned ticker (~200-3,000 depending on the day) and writes `<root>/price_refresh/<session>.json`. **Not wired into the nightly DAG yet** (a later task) — run it by hand or by an external scheduler, and never from inside a heavy stage (capture-inputs, a nightly job, a parity replay): it makes real provider calls and must not share a run window with them. **Pre-nightly order**, until the DAG wiring lands: 1) `price-refresh --session <as-of>` (this command, real run, no `--dry-run`), 2) `price-history capture` (the other agent's Tier-2 normalizer, once landed), 3) the nightly (`plan nightly` / `submit`). | `python3 -m engine.v2.ops price-refresh --session 2026-09-14` |
| `ledger import-history` | One-time bootstrap for a FRESH catalog (2026-09-14): a brand-new `decisions` table starts empty, so `legacy_settlement` fails `VALIDATION_FAILED` ("settlement names no committed prediction") on its first night against any trade entered before the catalog existed. Reads `ledger/predictions/*.jsonl` and `ledger/outcomes/*.jsonl` under `--source-root` read-only, in deterministic file/line order, and imports them via `engine.v2.ledger.decisions.import_lines` (idempotent per `(source_hash, line_number)`; a changed byte or conflicting duplicate refuses typed, `IDEMPOTENCY_CONFLICT`, with that file's transaction rolled back). One `BEGIN IMMEDIATE` transaction per file, under the same `supervisor.lock` `serve` holds — refuses `RESOURCE_UNAVAILABLE` if a supervisor is running. `--through YYYY-MM-DD` excludes predictions (`as_of`) / outcomes (`resolved_at`) dated after it, so the bootstrap never imports rows the nightly under test is about to commit itself. `--dry-run` reports the same JSON summary (`files`/`lines`/`imported`/`already_present`/`conflicts`/`date_range` per family) without writing anything. | `python3 -m engine.v2.ops ledger import-history --source-root /path/to/legacy/checkout --through 2026-09-09` |
| `price-history capture` | Append-only point-in-time capture of yfinance price data into the normalized bitemporal `price_history` dataset (design confirmed 2026-09-14; see `engine/v2/ops/price_history_store.py`). Reads both legacy sources read-only under `--source-root` (a `px_<T>.csv`, when present, always wins over the Tier-1 fetch cache, per `panel.py`'s own read order) and, per ticker, appends one version row per new/changed date or tombstones a date missing from a full-history retrieval; refuses `INPUT_CHANGED` typed on anti-backdating (a capture's `retrieved_at` must be later than every one already recorded for that ticker — never floor-adjusted) and `VALIDATION_FAILED` on a partial-window retrieval (starts later than the ticker's live stored history). A byte-identical re-capture is a no-op (`duplicate_source_hash`, zero new rows/bytes). `--dry-run` runs every check and reports true counts (`by_outcome`, `rows_added`, `rows_tombstoned`, `estimated_compressed_bytes`, overlapping-ticker disagreement counts) without writing a fragment, a raw object, or committing the capture-log transaction. Materialization (`engine.v2.ops.price_history_store.materialize`, not yet a CLI command) writes `px_<T>.csv` at the legacy relpath from a caller's own pin — plan/replay code, not an operator command. | `python3 -m engine.v2.ops --root data/operations price-history capture --source-root /path/to/legacy/checkout --dry-run` |
| `plan nightly` | Immutable shadow nightly plan. `--as-of` (ISO date, required), `--mode shadow` (only choice), `--input-manifest` (frozen legacy inputs; without it the plan carries `blocked_prerequisites` and cannot be submitted — **always build this with `capture-inputs` above, never with `ops snapshot plan-import`'s manifest**: that manifest's `capture_implementation_ref` is `snapshot_import_plan.v1`, covers only `TIER2_DATASETS`/Tier-3/reference inputs, and carries no `data/raw/fetch/**` at all — passing it here is the exact 2026-09-14 operator mistake that failed `legacy_finality` with `SOURCE_NOT_FINAL`, and the CLI now refuses it outright with a typed `INPUT_CHANGED` problem naming the missing kind/family, whether the wrong manifest is caught at `plan` or at `submit`), `--tickers a,b`, `--context-tickers a,b` (historical evidence universe; defaults to `--tickers`), `--year-start/--year-end` (default 2024/2026), `--full-run` (declares this run scores its whole context — writes the global `shadow` effect scope instead of a subset hash; refused unless `--tickers` equals `--context-tickers` exactly). Without `--full-run` the effect scope is ALWAYS the subset hash, even when the watchlist happens to equal the context — writing the global watermark requires the explicit flag, so a small debugging run can no longer silently advance it. | `python3 -m engine.v2.ops plan nightly --as-of 2026-09-12 --input-manifest manifest.json --full-run` |
| `plan experiment` | Smoke plumbing plan from `--spec` JSON (must carry `experiment_id`). `--no-ledger` is required in practice: without it the plan refuses (production experiments disabled). | `python3 -m engine.v2.ops plan experiment --spec /tmp/spec.json --no-ledger` |
| `submit` | Submit a saved plan under `--idempotency-key` (both required). Operator namespace policy admits `shadow` and `smoke` only. A blocked plan is refused with `INVALID_REQUEST`, exit 2. For a `plan nightly` document, `--idempotency-key` is grammar-required but NOT used for job identity — see "`--idempotency-key` semantics for nightly submission" below. | `python3 -m engine.v2.ops submit --plan art_... --idempotency-key nightly-2026-09-12-01` |
| `get` | Job document + attempt receipts. | `python3 -m engine.v2.ops get job_... --json` |
| `logs` | Progress events; `--follow` re-polls every 2s until the job reaches a terminal state. | `python3 -m engine.v2.ops logs job_... --follow` |
| `cancel` | Fenced cancellation; pass `--expected-attempt` with the job's active attempt — the fence is invalidated first, and the job completes as cancelled only once nothing is running. Omit it only for a job with no active attempt (queued, never started, or `retry_wait`): omission means "expect none", so a job that does have an active attempt still refuses (`conflict`, `STALE_EXPECTATION`). | `python3 -m engine.v2.ops cancel job_... --expected-attempt att_...` |
| `resume` | Recovery inspection. `--dry-run` only: without it the command refuses ("use the saved immutable plan to submit a changed run"). Reports implementation/environment invalidation, checkpoint reuse, `new_effects_authorized: false`. | `python3 -m engine.v2.ops resume job_... --dry-run` |
| `explain` | State, `queue_reason` and `failure` for one job — the first stop for a stuck job (§8). | `python3 -m engine.v2.ops explain job_...` |

The `--json` flag on `init`/`doctor`/`health`/`get` is accepted but cosmetic:
the CLI prints JSON regardless.

`ops snapshot plan-import`/`submit` also pin two derived model artifacts the
legacy scorer reads (`data/features/pnl_sim_history.parquet`,
`data/features/recalibration_pairs.parquet`) as reference inputs, each
recording its sha256 and the Tier-4 monthly fold (`YYYYMM`) of the import's
session, and a snapshot-mode `legacy_score` plan refuses if either is missing
from the pinned set (2026-09-14).

### 2.1 `--idempotency-key` semantics for nightly submission

(Phase 3 launch §5.5 item 1, fixed 56d8709.) `submit` always
requires `--idempotency-key`, but what it MEANS depends on the plan kind:

- **Non-nightly plan** (`artifact_check`, `experiment`): the key you pass IS
  the job's identity — `job_id_for(namespace, key)`. Two submissions with the
  same key and the same plan return the same job; the same key with a
  DIFFERENT plan is `IDEMPOTENCY_CONFLICT`.
- **Nightly plan**: the key you pass is accepted (the CLI grammar requires
  one) but never read. Every stage's own idempotency key is derived
  entirely from the saved plan document — session, ticker/year/population/
  snapshot scope, AND the plan's pinned identity (implementation, legacy
  manifest, `decision_clock`; `nightly.py`'s `_plan_identity`). Concretely:
  - Two `submit --plan <same plan_ref>` calls, with the SAME or DIFFERENT
    `--idempotency-key`, are both retries of the identical saved plan: they
    resolve to the exact same jobs, no duplicates, regardless of the key.
  - A fresh `plan nightly` call — even with byte-identical
    `--tickers/--year-start/--year-end/--expected-population`/snapshot
    scope — always re-pins `decision_clock` from the current clock, so it
    always produces a plan with a new identity and therefore new job ids
    when submitted. A changed `--input-manifest` or a code change between
    two `plan nightly` calls changes the identity independently of the
    clock.
  - This is why a cancelled run's jobs are never silently reused: re-running
    `ops plan nightly` (never resubmitting the old `plan_ref`) always gets a
    fresh plan identity and therefore fresh job ids, leaving the old
    (cancelled) rows exactly where they were.

## 3. Health surface

`health(conn, clock=...)` returns `operations_health.v1.0`:
`generated_at`, `executor_mode` (`watchdog`), `containment` (`best_effort`),
`jobs` (everything not succeeded/cancelled, with queue reasons), `watermarks`,
`current_release` (latest delivered), `withheld_release` (latest `eligible=0`),
`code_budgets` (`ok`, `first_failed_on`, `consecutive_nights`,
`unknown_occurrences`, `latest`, `override` — always `null` while D5 stands)
and `activation: shadow_only`. The streak counts scheduled nightly
*occurrences*, not attempts (§10.4); missing checks land in
`unknown_occurrences`, never as green.

**No timer writes `health.json` yet.** The producer is the CLI:

```bash
python3 -m engine.v2.ops --root data/operations health --out data/operations/health.json
```

which writes the sidecar atomically (`write_health` in `engine/v2/ops/health.py`:
temp file, fsync, `os.replace`, parent fsync). Run it after any serve/session or
catalog change you want the shell to show; until the activation sequence lands a
timer, a stale sidecar is the operator's responsibility, and the shell displays
its `generated_at` age rather than hiding it.

`engine/v2/serving/operations.py` is a library, not a script:

```python
from engine.v2.serving.operations import create_server
server = create_server(("127.0.0.1", 8790), token=os.environ["OPS_TOKEN"],
                       health_path="data/operations/health.json", release_root="data/operations/release", frozen_at="2026-09-13T00:00:00Z")
server.serve_forever()
```

- Auth: `Authorization: Bearer <token>` header or an `operations_token` cookie,
  compared with `hmac.compare_digest`. No token configured = everything
  protected is refused. The shell page `/` itself is unauthenticated; it only
  frames authenticated routes.
- Routes: `/` and `/legacy/*` (shell), `/health.json` (auth),
  `/release/current` (auth; 302 to the `CURRENT` pointer's `index.html`),
  `/release/<id>/<path>` (auth; `current` resolves through the pointer).
  `release_root` must contain `CURRENT` (a single path component) and
  `releases/<id>/...`; symlinks anywhere in the path are refused.
- Banner states, polled every 30s: **withheld** when `withheld_release` is
  set; **degraded** when `code_budgets.consecutive_nights > 0`; **current**
  otherwise. On any health-fetch failure the banner shows **unknown / stale**
  with the `frozen_at` stamp — it must never fall back to green (§10.4).

## 4. Resource policy

`engine/v2/ops/profiles.py`, `POLICY_VERSION = "ops_resources.2026-09-12.v1"`.
Internal amounts are bytes; GiB below is 2**30. Policy constants: base reserve
**1 GiB** (OS/API/supervisor, excluded from worker capacity), free margin
**512 MiB** (spike buffer), reserved CPU count **1**, min free disk **5 GiB**,
`max_heavy_concurrency = 1`, `max_disk_heavy_concurrency = 1`.

| Profile | Memory (GiB) | CPUs | Scratch (GiB) | Heavy | Disk-heavy | Measured |
|---|---|---|---|---|---|---|
| `io_fetch` | 0.5 | 1 | 2 | no | no | no |
| `delivery` | 0.25 | 1 | 1 | no | no | no |
| `projection` | 2 | 2 | 2 | no | no | no |
| `validation` | 4 | 4 | 1 | yes | no | no |
| `legacy_score` | 4 | 5 | 2 | yes | no | no |
| `model_evidence` | 4 | 4 | 1 | yes | no | no |
| `legacy_rebuild` | 5.5 | 5 | 20 | yes | yes | no |
| `experiment_heavy` | 5.5 | 5 | 10 | yes | yes | no |

Every heavy profile is `measured=False` — the numbers are starting evidence
from AGENTS.md (3 GiB scorer, ~5.5 GiB rebuild peak), not measurements — so
each heavy profile runs **exclusively** until a reviewed measurement says
otherwise (`EXCLUSIVE_UNMEASURED` / `HEAVY_SLOT` queue reasons).

Admission rules (§8.1), both of which must pass before a claim:

```text
capacity = min(host_total, finite_container_limit) - base_reserve
sum(active_reservations) + new_reservation <= capacity

headroom = min(host_available, container_remaining) - free_margin
new_reservation + sum(max(0, reserved_i - measured_current_i)) <= headroom
```

Plus: one heavy worker at a time; one disk-heavy at a time; scratch bytes
reserved against the 5 GiB free-disk floor; CPUs allocated as disjoint IDs
from actual affinity minus the reserved one. A profile larger than capacity
queues forever under `PROFILE_EXCEEDS_CAPACITY` — that is a signal to optimize
or add capacity, **not** to lower the declared reservation until it fits.

Proposing a policy change: only **between runs**, as a new `POLICY_VERSION`
with reviewed measurement evidence (peak use by stage/input scale/profile),
conservative direction only. Never auto-lower a reservation after one cheap
cache-hit run; never retry an OOM forever by raising the cap. `policy_problems()`
validates the structure before a policy admits anything.

## 5. Verification runs

Run in order. (a) and (b) are fast; (c) and (d) are heavy — see §9 before
starting either. `/usr/bin/python3` is the interpreter (the venv lacks pandas).

### a. Fast gates

```text
/usr/bin/python3 -m pytest tests/test_v2_ops_*.py tests/test_diagnosis_comparator.py -q
/usr/bin/python3 checks/code_budgets.py --quiet
/usr/bin/python3 checks/package_readmes.py
/usr/bin/python3 checks/rearchitecture_phase1_lint.py --worktree
/usr/bin/python3 checks/install_hooks.py --check
```

Notes: bare `code_budgets.py` checks **staged blobs** (hook mode); use
`--all` for the whole tree, which is what the phase-0 gate runs. The lint
check pins ruff to the `ruff==0.16.7` line in `requirements.txt` and fails
closed with `LINTER_UNAVAILABLE` / `LINTER_VERSION_DRIFT` (D8). `install_hooks
--check` reports hook state and changes nothing; install/update is the same
command without `--check` (`--force` only over an unrecognized hook).

### b. Coverage evidence

```text
/usr/bin/python3 checks/v2_coverage_ratchet.py --profile phase1 --measure --output /tmp/coverage.json
/usr/bin/python3 checks/rearchitecture_phase1_gate.py --coverage /tmp/coverage.json
```

The fixed suite is `tests/test_v2_ops_*.py + tests/test_diagnosis_comparator.py`
under `coverage run --source=engine/v2`; the measurement records the sorted
test list and a source hash, and refuses if the tree changes mid-run. The
ratchet baseline is `checks/v2_coverage_ratchet_phase1_baseline.json`
(`--baseline` to point elsewhere), cut 2026-09-13 at commit `5fc0146`:
contracts 100%, diagnosis 96.8%, foundation 96.3%, serving 85.3%, ledger
80.8%, ops 74.1%, empty packages recorded as empty. Cutting a new baseline is
a reviewed evidence update committed by hand — measurement never writes the
baseline, and a baseline is never lowered automatically. If the tree changed
since the baseline was cut, `--measure` first and pass the fresh measurement
to the gate; `COVERAGE_SOURCE_DRIFT` means the measurement, not the baseline,
is stale.

### c. Tier-1 replay receipts (heavy)

~3 GiB resident, minutes each, **sequential, one at a time** (D6):

```text
/usr/bin/python3 tools/bounded_run.py --max-rss-gb 6.5 --cpu-set 0-5 -- /usr/bin/python3 tools/replay_tier1.py --json
/usr/bin/python3 tools/bounded_run.py --max-rss-gb 6.5 --cpu-set 0-5 -- /usr/bin/python3 tools/replay_tier1.py --seed-defects
```

(`--` separates bounded_run's flags from the command's own.) Each run writes a
receipt to `fixtures/tier0/receipts/<corpus-version>.json` (seeded:
`...<corpus-version>.seeded.json`), bound to the corpus hash, the code hash,
the frozen dependency hash, the **current baseline** (`baseline/CURRENT`) and
the store snapshot. Exit 0 only on verdict `AGREE`. `--limit` and
`--skip-deps-verify` are diagnostics-only: the phase-0 gate re-validates the
receipt and refuses partial or unverified runs.

### d. Private parity canary (heavy)

Read the two frozen-input pointers, then prepare a private root:

```text
BASELINE_VERSION=$(/usr/bin/python3 -c "import json;print(json.load(open('baseline/CURRENT'))['version'])")
CORPUS_VERSION=$(/usr/bin/python3 -c "import json;print(json.load(open('fixtures/tier0/CURRENT'))['version'])")
/usr/bin/python3 checks/rearchitecture_phase1_canary.py prepare --root . \
    --baseline baseline/$BASELINE_VERSION --corpus fixtures/tier0/$CORPUS_VERSION \
    --output-dir <private-dir>
/usr/bin/python3 checks/rearchitecture_phase1_canary.py reference --root <private-dir> --output-dir <private-dir>
/usr/bin/python3 checks/rearchitecture_phase1_canary.py adapted   --root <private-dir> --output-dir <private-dir>
/usr/bin/python3 checks/rearchitecture_phase1_canary.py compare   --root <private-dir> \
    --reference <private-dir>/reference.json --adapter <private-dir>/adapted.json \
    --selfcheck <selfcheck.json> --receipt <private-dir>/canary_receipt.json
```

`prepare` copies the frozen baseline package, the complete tier-0 corpus,
declared dependencies, `engine/models/registry.json`, `data/MANIFEST.md` and
current `engine/`, `tools/`, `checks/`, `data/curated` sources into the private
root (read-only, hash-manifested in `INPUT_MANIFEST.json`, plus
`score_requests.json`). `reference` and `adapted` each run a fresh-process
scoring runner inside that root, writing scored rows to
`<output-dir>/reference.json` / `<output-dir>/adapted.json` and a separate run
log to `<mode>.runlog.json` (the runner output is evidence; the log never
overwrites it). `compare` does per-row tier-1 comparison, keying real runs by
`request_id` over each row's `record`, and requires the selfcheck receipt to
be `{"ok": true, ...}`.

The selfcheck receipt must be **derived from real evidence — the fresh tier-1
receipts of §5c (both verdict `agree`, both bound to the current baseline and
corpus `CURRENT` versions, zero problems, full replayed population) are what
it rests on. Never hand-write it**: `compare` only checks the `ok` flag, so a
fabricated file silently launders the whole canary. The derivation used for
the 2026-09-13 certification run is recorded in its canary receipt.

### e. Phase gates

```text
/usr/bin/python3 checks/rearchitecture_phase0_gate.py
/usr/bin/python3 checks/v2_coverage_ratchet.py --profile phase1 --measure --output /tmp/coverage.json
/usr/bin/python3 checks/rearchitecture_phase1_gate.py --coverage /tmp/coverage.json --output reports/rearchitecture_phase1_gate.json
```

**Do not run the phase-1 gate bare.** Without `--coverage` its coverage row
fails with `COVERAGE_EVIDENCE_MISSING` and the top-level `ok` is `false`. That
is by design (a missing check is a failing check, never a green one), not a
regression: measure coverage first, then pass the measurement. A red gate seen
without `--coverage` says nothing about the code.

The phase-0 gate re-derives every row now (corpus integrity, both tier-1
receipts against CURRENT, negative-control suite, structural checks, baseline
reproducibility, hook state); `--json` and `--previous <file>` report which
rows changed since the last run. The phase-1 gate adds lint, coverage evidence
and budgets over the complete checkout; its `decision_correctness` and
`projection_security` rows stay `null` — passing it does not authorize a
prediction.

## 6. Activation sequence

§14.2 mapped to concrete state as of 2026-09-13:

| Step | State |
|---|---|
| 1. Unit/fault tests, synthetic data, isolated roots, no network | **Done** — `tests/test_v2_ops_*.py`, runnable via §5a. |
| 2. Read-only planning against current catalog/data metadata | **Done** — `doctor`, `plan nightly` (blocked without a manifest), `plan experiment --no-ledger`. Do not start a full nightly as a diagnostic. |
| 3. Approved small shadow nightly against private frozen inputs, then sequential parity | **Pending operator approval.** The mechanics exist (§5c–d); no production ledger, registry, publication or paid pull may be touched. |
| 4. Rehearse import/export and cutover in copies; measure stage peaks | **Pending.** Crash/restore and withheld-publication display rehearsals; production profile reservations wait on measured peaks (§4). |
| 5. Explicit operator approval (timer, data/provider ownership, decision authority, backup target, publication target) | **Pending.** D4: the production writer switch happens only here, only with separate explicit approval; until then the authority is exercised on copies only. |
| 6. Quiesce old writers, snapshot, switch one owner, one bounded production run | **Pending.** |
| 7. Recurring submission after canary evidence review | **Pending.** |

A catalog reaching step 3 for the first time must run `ops ledger
import-history --through <the day before the first nightly session>` first,
or that first `legacy_settlement` fails on any trade entered before the
catalog existed (§2's `ledger import-history` row).

D5 binds every step: an engineering-budget failure (code budgets over
`engine/v2`, the pinned linter, the coverage ratchet, hook drift) **withholds
publication with no override path**. It does not block ingestion, valid
predictions, settlement or backup, and it does not reset the streak.
Decision-correctness, projection/security and structural checks are never
overridable under any future policy.

## 7. Rollback (§14.3, in commands)

Order matters. Keep all committed artifacts, imported rows, decisions,
supersessions, unresolved outcomes and publication receipts; rolling back code
is not permission to restore an old ledger over new commits.

1. **Disable new submissions** — stop the supervisor (`serve`); start no other.
   There is no separate submission kill-switch: with no supervisor running,
   queued jobs simply are not claimed.
2. **Fence/cancel attempts** —
   `python3 -m engine.v2.ops cancel <job-id> --expected-attempt <attempt-id>`
   for anything in flight. The fence is invalidated first; stale workers
   cannot commit afterwards.
3. **Reconcile processes** — `python3 -m engine.v2.ops doctor --json`: check
   `diagnostics.unmanaged_processes` (competing jobs the catalog does not
   own), `integrity_errors`, and the executor probe.
4. **Pause outbox delivery** — delivery effects are claimed by the supervisor,
   so step 1 is the pause: queued outbox rows stay unclaimed and durable until
   a supervisor is deliberately restarted. There is no per-row pause flag.
5. **Restore the catalog only into a NEW root** —
   `engine.v2.ops.backup.restore_backup(source, destination)` (refuses an
   existing destination; verifies the database hash, `integrity_errors` and
   every artifact byte against the single backup manifest).
6. **Export decisions to a legacy generation** —
   `engine.v2.ledger.export.export_generation(conn, root, generation="<name>")`:
   sequence-ordered `predictions/`+`outcomes/` JSONL, durable write, atomic
   `CURRENT` switch; re-exporting an existing generation byte-compares instead
   of overwriting. Every catalog decision since cutover must be exported and
   reconciled **before** a legacy writer resumes.
7. **Switch writer ownership once** —
   `engine.v2.ledger.decisions.set_authority(conn, expected_owner, owner, stamp)`
   inside a caller-owned transaction (it raises otherwise); a current owner
   different from `expected_owner` raises `DecisionConflict`, and the
   generation counter increments. Never run both writers while comparing
   them. If new-authority records cannot be represented faithfully, stay
   read-only until reconciliation rather than dropping them.

The rehearsal is a test, not a hope:
`tests/test_v2_ops_authority.py::test_o32_authority_rollback_rehearsal_preserves_decisions`.

## 8. Failure triage

| Symptom | Where to look |
|---|---|
| Job sits queued | `python3 -m engine.v2.ops explain <job-id>` → `queue_reason.code`: `RESERVATION_BUDGET`, `MEMORY_HEADROOM`, `HEAVY_SLOT`, `HEAVY_SLOT_HELD`, `EXCLUSIVE_UNMEASURED`, `PROFILE_EXCEEDS_CAPACITY`, `DISK_SPACE`, `DISK_HEAVY_SLOT`, `CPU_UNAVAILABLE`, `PROVIDER_BUDGET`, `PROVIDER_UNAVAILABLE`, `LIVE_WINDOW`, `INVALID_LIVE_WINDOW`, `STORE_LEASE_HELD`, `STORE_RECOVERY` (each carries `needed`/`available` and a reconsideration reason). |
| Attempt failed | `python3 -m engine.v2.ops get <job-id> --json` → the attempt receipt's failure envelope, categories per operations §5.3: `RESOURCE_UNAVAILABLE`, `RESOURCE_LIMIT_EXCEEDED`, `TRANSIENT_SOURCE`, `RATE_LIMITED`, `CREDENTIAL_INVALID`, `SOURCE_NOT_FOUND`/`SOURCE_EMPTY`/`SOURCE_NOT_FINAL`, `INPUT_CHANGED`, `CHECKPOINT_INCOMPATIBLE`, `VALIDATION_FAILED`, `INTEGRITY_FAILED`, `LEASE_LOST`, `CANCELLED`, `PUBLICATION_REFUSED`, `DELIVERY_FAILED`, `BACKUP_FAILED`. `retryable` and `retry_after_seconds` say whether to wait or act. A failure the worker itself raised (not a bare `WORKER_FAILED`) carries a `diagnostic_ref` — `explain` prints it when present, and it names a verified artifact holding the raw `details` kept out of `failure_json`. |
| CLI printed a Problem and exited 2 | The JSON on stdout is the envelope (`problem.v1.0`). `INVALID_REQUEST` on submit usually means a blocked plan (nightly without `--input-manifest`) or a plan kind not enabled for submission. |
| Bounded job exited 137 | Read the tail: `CAP BREACH, killing tree` + a per-process memory breakdown = the watchdog did its job; raise the cap only with reviewed evidence. A bare `[bounded] exited -9` with the last heartbeat under cap = the **kernel** OOM killer — the cap was not the binding constraint; re-budget against `free -m` (AGENTS.md). |
| Replay/canary receipt refused by the gate | Receipts are content-bound: a changed corpus, code, dependency set or baseline invalidates them. Re-run §5c/§5d; never edit a receipt. |
| Health looks stale | The banner shows `unknown / stale` on any fetch failure — by design, never green. `code_budgets.unknown_occurrences` lists nights whose check is missing (absence is not health). For the legacy Models view inside the shell, AGENTS.md's `model_evidence_stale` flag applies: a failed evidence rebuild degrades to the cached table; check the flag before trusting the page. |
| Competing unknown processes | `doctor` → `diagnostics.unmanaged_processes`; reconcile before resuming (§7 step 3). |

## 9. Host discipline

Every heavy command in §5 is bound by the AGENTS.md rules for this box
(12 cores, 7.6 GiB, shared with other agents):

- **One heavy worker at a time.** The policy enforces it inside the catalog
  (`max_heavy_concurrency=1`, unmeasured heavy = exclusive); honor it for
  manual runs too — never overlap §5c with §5d or with a nightly.
- **`--cpu-set`, not `--cores`, for concurrent jobs.** `--cores` pins every
  job starting at core 0, so two bounded jobs collide by construction; give
  them disjoint ranges (e.g. `0-5` and `6-11`). bounded_run also sets
  BLAS/OMP/MKL/NUMEXPR threads to the set width and nices the tree to 19.
- **Budget the SUM of caps against `free -m` available**, not each cap alone.
  `--max-rss-gb` protects the box from your job; nothing protects your job
  from the box. Size caps against the PEAK step, not the average.
- **`/usr/bin/python3 -u` for anything long** — an OOM kill discards buffered
  stdout, and the evidence dies with the process.
- **Background + poll for long runs**, with progress at least once a minute
  (bounded_run's watchdog heartbeat satisfies this) and an operator check-in
  at least every 5 minutes.
- **Never two Polygon consumers at once** — they split the same ~10 req/min
  quota and both stall. Shadow verification here uses frozen inputs and makes
  no provider calls at all; keep it that way until step 5 of §6.
- **Credentials never in argv.** Source `.env` into the environment;
  bounded_run prints the full command line, so anything passed as an argument
  leaks into logs. The serving token likewise comes from the environment.
