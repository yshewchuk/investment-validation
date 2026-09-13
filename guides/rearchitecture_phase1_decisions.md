# Rearchitecture Phase 1 — decisions and starting inventory

Slice **P1-0** of [Phase 1 — Operations](rearchitecture_phase1_operations.md)
§15. That guide requires the §3.2 migration decisions to be recorded in a short
reviewed architecture decision before the affected code lands. This is that
record, plus the inventory P1-0 asks for: the baseline evidence, the host
environment, and the current nightly's commands and effects.

Each decision says who made it. **Operator** decisions were answered by the
repository owner on 2026-09-12. Decisions marked **implementer** are judgement
calls made while implementing, and are open to review like any other diff.

## 1. Baseline evidence (§3.1)

`python3 checks/rearchitecture_phase0_gate.py` was run on 2026-09-12 at
`e9a1f43`, before any Phase 1 code, and was **green, 9/9**. Its rows carry the
six prerequisites of §3.1:

| §3.1 prerequisite | Gate row that re-derives it |
|---|---|
| 1. Corpus membership, cases, hashes and manifest agree; nothing vacuous | `tier0_corpus` |
| 2. Every strategy scoreable where available, plus its refusals | `tier0_corpus` (required axes) |
| 3. Replay recomputes in a fresh process from frozen inputs | `tier1_real_replay` |
| 4. Typed, ordered, per-field comparison | `negative_controls`, `tier1_real_replay` |
| 5. Honest availability labels | `tier0_corpus` |
| 6. Planted defects through real stages | `tier1_seeded_controls` |

Catalog, executor and checkpoint work proceeds against synthetic fixtures
regardless (§3.1). Parity certification (O30) and production activation remain
gated on this row set being green **at the time they run**, not on this record.

## 2. Decisions

### D1 — Adapter budget bootstrap (operator: approved)

The adapter ledger's ceiling is zero and shrink-only (system rearchitecture
§4.6), while Phase 1 must call legacy code rather than copy it. Approved: one
reviewed inventory of the exact legacy symbols ops needs, after which the
ceiling only shrinks.

- The inventory lands **in the same commit as the first adapter** (slice P1-5),
  so the review sees each entry beside the code that uses it. Each entry names
  the callable or command, read set, write set, credentials, hidden
  subprocesses, retry behaviour, output contract and removal phase (§9.2).
- `committed_ceiling` is raised **once**, in that commit, to exactly that
  count. The import-layer check then fails any increase.
- Runtime entrypoints are entries too. A `subprocess` edge into a legacy module
  is a dependency exactly as an import is, and dynamic imports, filesystem
  imports and undeclared subprocess edges are not a way around the count.
- One adapter module per owning package remains the rule
  (`ops/legacy_adapter.py`, `ledger/legacy_adapter.py`).
- Until that commit, workers are synthetic.

### D2 — Shared canonical primitives (implementer, per the guide's recommendation)

The RFC 8785 implementation moved from `engine/v2/diagnosis/canonical.py` to
`engine/v2/foundation/canonical.py` byte-for-byte. Diagnosis re-exports the
same function objects, so there is still exactly one canonicalizer.
`tests/test_v2_ops_foundation.py` pins hashes computed by the phase-0 copy
before the move, and asserts no second definition exists anywhere in v2.
Production ops never imports diagnosis; the layer check already enforces it.

### D3 — Stage granularity (implementer, per the guide's recommendation)

- Legacy scoring is **one coarse stage** in Phase 1, not ten invented internal
  stages. Its implementation hash covers its declared dependency closure, so an
  edit to `analogs.py` invalidates scoring and everything downstream of it — not
  ingestion, and not only the analog block.
- A checkpoint is reusable only when inputs, implementation, parameters,
  environment and output schema all match. **The same rule applies after a
  crash.** The architecture's distinction between crash recovery and change
  recovery does not permit reusing an old-code checkpoint after a restart.
- Deferred to Phase 4, when extraction creates real boundaries: generation,
  analog/scenario construction, valuation, simulation and chooser as separately
  checkpointed stages. No Phase 1 receipt may claim localization finer than the
  worker boundary it actually observed (§13).

### D4 — Decision authority (operator: catalog authority, on copies only)

Approved: a minimal catalog-backed decision authority — ledger-owned tables in
the operations SQLite file, legacy row payloads unchanged, legacy JSONL
regenerated as a compatibility export. It is **built and rehearsed against
copies only** in this phase. Switching the production writer is §14.2 step 5,
which needs separate explicit approval.

The operator's condition: **no duplicated predictions.** It is carried by
mechanisms, each with a test (O19, O20, O32), not by a statement:

- a unique constraint on the logical decision key (event, strategy/deployment,
  clock, scheduled occurrence) in SQLite itself, not a Python precheck;
- an identical payload for an existing key returns the existing receipt;
- a different payload for an existing key is `IDEMPOTENCY_CONFLICT`, never a
  silent skip (today's `ledger.snapshot` filters on `row_id` and would drop a
  changed payload without saying so);
- the history import collapses byte-identical duplicate rows into one logical
  record, and stops with a reconciliation report on conflicting ones;
- one writer: the decision insert, its validation references and its outbox
  rows commit in one short transaction under a verified fence, and no worker
  appends to the exported JSONL.

### D5 — Publication overrides (operator: refused for now)

An engineering-budget failure — `checks/code_budgets.py` over `engine/v2`, the
pinned linter, the coverage ratchet, or a missing/drifted pre-commit hook —
withholds publication with **no override path**. It does not block ingestion,
valid predictions, settlement or backup, and it does not reset or pause the
failure streak. Decision-correctness, projection/security (secret scan, access
control, serialized selfcheck) and structural checks (import layers, READMEs,
adapter ledger) are never overridable under any future policy.

Still to reconcile before any override is enabled: system rearchitecture §4.7
refers to an override "in the same ledger the §4.6 exemptions use", while §4.6
says v2 has zero exemptions and no exemption file. A dedicated
publication-policy ledger (§3.2 recommendation) is the likely resolution, and
needs its own review.

### D6 — Tier-1 receipt code-hash scope (implementer)

`checks/replay_identity.code_hash` hashed all of `engine/**`, including every
future `engine/v2/ops` file. Legacy may not import v2 (enforced by §4.2 rule 3)
and the replay harness imports only `engine/v2/diagnosis`, whose closure is
`foundation` and `contracts`. So an operations edit could not change a replay's
answer, but would still turn the phase-0 gate red on every Phase 1 commit —
the failure mode the operator already rejected when choosing content-bound over
commit-bound receipts.

The scope is now legacy `engine/` plus `REPLAY_V2_PACKAGES` (contracts,
diagnosis, foundation). `tests/test_v2_ops_replay_scope.py` re-derives the
harness's v2 import closure from source and fails if it ever leaves that set;
a planted `engine.v2.ops` import in the harness is its negative control.

Changing the scope changes the hash, and the canonical move changed diagnosis
anyway, so both tier-1 receipts are re-run once after P1-1 under
`tools/bounded_run.py --max-rss-gb 6.5`, sequentially.

### D7 — Executor mode on this host (implementer, from the probe in §3)

The only honest executor here is `executor_mode=watchdog`,
`containment=best_effort`. cgroup2 is mounted read-only and the process sits in
the root cgroup, so a delegated cgroup cannot be created. The cgroup executor
is still implemented behind a capability probe (§9.1). The probe must report
it unavailable on this host, and a test proves that.

### D8 — Linter pin and the frozen environment lock (implementer; open until P1-7)

`ruff` is not installed; 0.16.7 is the newest version on the package index as
of 2026-09-12. §12 says to pin it in the existing environment lock, but the
phase-0 gate's `baseline_package` row requires the root `requirements.txt` to
equal the frozen baseline's copy. Adding a line therefore needs a new baseline
version cut through `tools/baseline_export.py` — a recorded decision plus a
new version, which is the phase-0 rule for changing a frozen input. That is
done in P1-7 together with the lint check, not as a side effect earlier.

## 3. Host capability inventory (probed 2026-09-12)

| Property | Observed | Consequence |
|---|---|---|
| Interpreter | `/usr/bin/python3` 3.14.4; pytest 9.1.1, coverage 7.16.0; ruff absent | Tests and coverage run; D8 |
| CPUs | 12; `sched_getaffinity` = 0–11 | Allocation uses affinity, never `os.cpu_count()` |
| Memory | MemTotal 8,162,779,136 B (7.60 GiB); swap 2 GiB | One heavy worker at a time (§8.1) |
| Container limit | none visible; root cgroup has no `memory.max` | Host values bound admission |
| cgroup2 | mounted `ro` at `/sys/fs/cgroup`; process in `/` | D7: watchdog only |
| Service manager | PID 1 is `bash` (WSL); `systemctl` present but offline; no `crontab` | Timer installation is a P1-9 operator decision; nothing may assume systemd or cron |
| Disk | overlay, ~906 GiB free | Scratch reservation still enforced |
| Kernel | 6.18 WSL2; boot ID readable at `/proc/sys/kernel/random/boot_id` | Process identity = boot ID + PID + start ticks |

## 4. Current nightly: commands and effects (§3.1, §9.2)

`python3 -m engine.dashboard.nightly` flags: `--as-of`, `--require-as-of`,
`--horizon`, `--alt-strikes`, `--tickers`, `--bundle`, `--target`,
`--no-refresh`, `--no-tiers`, `--no-publish`, `--no-backfill`, `--backup`,
`--chain-sessions`, `--max-staleness`, `--json`. The CLI wraps `run_nightly` in
`single_run_lock` (an `flock`); in-process callers are not locked.

`run_nightly` records these steps, in this order. The effect column comes from
the step order and the ledger's writer functions. The exact per-symbol read and
write sets are the P1-5 adapter inventory's job, and this table does not claim
them.

| Step | Effect class | Known side effects |
|---|---|---|
| `universe` | read | — |
| `refresh` | external + store write | ORATS / Polygon / Nasdaq requests (credentials); data store writes; `reports/orats_unknown_symbols.json` |
| `finality`, `validate` | read | — |
| `moves`, `tiers` | store write | computed moves; Tier 3 panel and Tier 4 forecasts rebuilt in place |
| `ledger` | **decision write** | `ledger.snapshot` appends `ledger/predictions/<date>.jsonl` |
| `settle` | **outcome write** | `ledger.score_outcomes` appends `ledger/outcomes/<date>.jsonl` |
| `score` | compute | ~3 GiB scorer in the same process |
| `backfill` | **decision write** | `ledger.snapshot` for missed nights |
| `model_evidence` | cache write | `data/features/model_evidence.json` |
| `render`, `selfcheck` | bundle write | the bundle directory (default `dashboard/earnings`) |
| `publish` | external delivery | the publish target |
| `flags` | report write | `reports/` flag report |
| `backup` | external delivery | git + private-mirror sync (only with `--backup`) |

Two hazards follow directly and shape the design. `run_nightly(publish=False)`
still writes predictions, outcomes and store state, so it is not a pure worker
(§9.2, O16). And the whole nightly is one process holding every stage's memory,
so no stage's reservation can be released before the next begins (§10.1).

`tools/bounded_run.py` is the executor's starting point. It offers
`--cpu-set`, thread-count environment variables and a proportional-RSS tree
watchdog. It does not: admit globally, persist anything, identify processes
across PID reuse, or kill descendants that left the session's process group.
It also prints the full command line, which would leak any argument carrying a
credential. The v2 watchdog executor fixes each of these rather than wrapping
it.
