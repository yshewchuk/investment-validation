# Rearchitecture Phase 6 — P6-6 rehearsal runbook

Status: operator-facing runbook, 2026-09-25. It packages already-decided
procedure for the [Phase 6 — Consumer parity](rearchitecture_phase6_consumer_parity.md)
P6-6 rehearsal and the [Phase 7 — Cutover](rearchitecture_phase7_cutover.md)
qualification window; it introduces no new capability, changes no authority and
makes no decision of its own. Commands below reuse existing entrypoints that
were verified against the source on 2026-09-25. This runbook is validated by
the supervisor actually following it during the real P6-6 HEAVY RUNS (the
same work that produces the evidence in §8), not by a test file.

## 1. Scope and entry

Entry conditions are the gates in
[Phase 7 — Cutover § Entry gate](rearchitecture_phase7_cutover.md#entry-gate)
and the qualification rules in
[Phase 6 — Consumer parity § Acceptance and qualified sessions](rearchitecture_phase6_consumer_parity.md#acceptance-and-qualified-sessions).
Read both first: this runbook assumes the integrated candidate is already
built, that complete consumer parity is accepted, and that the only remaining
work is running the candidate in shadow mode, measuring it, and collecting
real-session evidence.

In scope: operator setup, the shadow rehearsal runs, controlled-failure and
restore drills, resource measurement, evidence collection, and the definition
of a qualified session. Out of scope: scheduling the ten nights (Slice 12;
see §10), any production-authority change (§6), and any change to the legacy
publisher (§2).

The tools this runbook points at are built outside it:

- controlled-failure harness — `tools/v2_controlled_failure_drill.py` (v2a);
- resource-measurement recorder — `tools/v2_resource_measurement.py` and
  evidence completeness check — `tools/v2_session_evidence_check.py` (v2b
  Parts 2–3);
- backups and restore drill — existing `engine.v2.ops.backup` API and
  `tools/v2_restore_drill.py` (no new capability).

## 2. Legacy keeps publishing (user constraint)

Quoting [Phase 7 — Cutover § Entry gate](rearchitecture_phase7_cutover.md#entry-gate):

> Legacy remains the only official prediction/settlement/publication writer
> until the controlled switch.

This constraint holds for the whole qualification window. The legacy
schedule/cron and the legacy board run **UNCHANGED** throughout. Every v2 or
native run in this document is shadow mode only: `build_nightly_plan`
(`engine/v2/ops/nightly.py`) refuses any mode but `shadow` today
(`plans.py::nightly_plan` refuses with "production cutover has not been
activated"; the CLI's `--mode` accepts only `shadow`). No rehearsal step here
stops, pauses, reschedules, or shares credentials or publication targets with
legacy. The user keeps seeing predictions and can trade off the legacy board
exactly as today for the entire qualification window.

## 3. Provider account setup (one-time)

The native refresh reserves calls against a durable `provider_accounts`
budget row. Create or update it once per account with the S4B2 operator
command (`engine/v2/ops/cli.py::provider_account_command`):

    python3 -m engine.v2.ops provider-account \
        --account orats-daily-market --remaining <N> --live-reserve <N>

- The account name for the native `daily_market` refresh is
  `engine/v2/ops/nightly.py::NATIVE_DAILY_MARKET_ACCOUNT`
  (`"orats-daily-market"`); one fetch unit costs two ORATS calls
  (`hist/summaries` + `hist/cores`, `ORATS_CALLS_PER_DAILY_MARKET_UNIT`), so
  size `--remaining` in calls, not sessions.
- Absent row: created at generation 1. Present row: `remaining` and
  `live_reserve` are replaced and the generation increments in one
  transaction; operator/backoff state (`blocked_code`, `next_eligible_at`) is
  never touched by a budget edit.
- The scheduler reserves against exactly this row:
  `scheduler._reserve_provider` refuses `CREDENTIAL_INVALID` when the row is
  absent. Planning never creates one.
- This command takes and writes **only budget numbers — never a credential**.
  Never print or log the credential itself (the env var or wherever it is
  actually held). This restates AGENTS.md's standing "never print credential
  or env values" rule for this one-time setup step; it is not a new
  mechanism.

## 4. Native `daily_market` refresh timing (rehearsal constraint)

`engine/v2/ops/providers/orats_daily_market.py` does not walk back over recent
sessions. A market-wide date that is not published yet — 404 on both
`hist/summaries` and `hist/cores`, or a 2xx with no rows — is a
`SOURCE_NOT_FINAL` refusal, by design, with no automatic retry to an earlier
date (`_classify`/`_overall_kind`). The unused `lookback_days` parameter exists
only for a future caller that plans one unit per candidate date.

Consequence for rehearsal: run any native-refresh nightly
(`python3 -m engine.v2.ops plan nightly ... --refresh-mode native`) only AFTER
ORATS has published that date (observed ~midnight ET). Running earlier is an
expected refusal, not a defect — the `incremental_refresh` job's own retry
policy owns that decision. Record in the evidence that the run was started
after the publish window, not before: the v2b Part 2 recorder's `--cache-state`
plus `started_at`/`ended_at`/`wall_seconds` fields double as this record
(§7–§8). The default `--refresh-mode legacy` submits no refresh job at all and
leaves the DAG unchanged.

## 5. Backup and restore

Take a backup with the existing API — no new capability:

- `engine.v2.ops.backup.prepare_backup(conn, key, artifacts, clock=...)`
  enqueues one idempotent `backup` effect for `key`;
- `engine.v2.ops.backup.run_backup(conn, key=..., owner=..., target=...,
  clock=..., store=...)` writes a consistent SQLite backup plus referenced
  objects, then fsyncs and renames `<key>.manifest.json` before acknowledging.

A real nightly runs both through its `backup` stage (`backup_effect`) after
`decision_commit`; retain that backup directory for the session. Run the
restore drill after every qualified session:

    python3 tools/v2_restore_drill.py \
        --backup <backup-dir> --restore-root <new-dir> \
        --score-request <request.json> --native-inputs <native_inputs.json> \
        --expected-score-hash sha256:... \
        --original-export <live-export-root> --generation <generation> \
        --expected-decisions-count <N> \
        [--artifact-root <dir>]

`tools/v2_restore_drill.py` restores into an isolated root
(`engine.v2.ops.backup.restore_backup`, every database/artifact byte
reverified), replays one original score offline through the same no-fit,
no-provider-pull `ops rescore` entrypoint, and diffs a ledger generation
exported from the restored catalog against one exported from the live catalog
before the backup. It never mutates the backup or the original export
directory. Exit 0 = PASS, 1 = FAIL receipt, 2 = refused (unusable backup or
baseline).

## 6. Authority-switch drill (rehearsal only)

The switch and rollback procedure is rehearsed on copies only, per
[Phase 7 — Cutover](rearchitecture_phase7_cutover.md) P7-3: a
scratch/private-target drill using the retained deployment, data/catalog
backups, and pointer and writer-lease/fence changes — proving an interrupted
switch never leaves two official writers and that rollback reconciles new
decisions instead of deleting history. P7-5 (switch and observe: stop/drain
the old schedule, fence old authority, activate v2, verify exactly one writer)
is the production step that follows a separate, explicit user authorization.

State plainly: **Slice 11 performs NO real switch.** Nothing in this runbook
activates v2 authority, stops the legacy writer, or shares its credentials or
publication targets; §2 still governs.

## 7. Resource policy

Restating AGENTS.md "Running jobs on this box":

- **One heavy job at a time.** Before launch, all three probes must be empty:

      pgrep -f "[b]ounded_run.py"
      pgrep -f "[s]erve_monitor"
      pgrep -f "[c]orpus_parity.py run"

- **`free -m` available floor**: at least **5.5 GB** available, and at least
  **6.5 GB** for an uncapped measurement, is the hard stop *before* launch.
  The box's capacity has grown to ~10.7 G RAM / 20 G swap since that rule was
  measured (2026-09-11 at 7 GB), so these are conservative minimums, not the
  new ceiling.
- **Disjoint `--cpu-set`** for any concurrent jobs.
- **Standard rehearsal/qualification cap**:
  `--max-rss-gb 8 --max-swap-gb 6 --cores 8`.

Every rehearsal/qualification run must be wrapped in v2b's
`tools/v2_resource_measurement.py` (it launches `tools/bounded_run.py` — never
reimplement its watchdog — records pre/post `free -m`, contention, wall time
and the parsed peak, and exits with the child's own exit code). Every
controlled-failure rehearsal must use v2a's
`tools/v2_controlled_failure_drill.py` (real `fault=` hooks in production code,
scenario receipts, `--against-real-candidate` restores into an isolated copy
first).

## 8. Evidence

Receipts live under `reports/phase6_evidence/` in this layout:

    reports/phase6_evidence/
      resource_measurement/   # one JSON per measured run (v2b Part 2 recorder)
      route_probe/            # one <session>-route_probe.json per session (v2c)
      controlled_failure/     # v2a drill receipts (always copied here)
      qualified_session/      # real-EOD-candidate receipts from the HEAVY RUNS

Each capability row in `tools/phase6_capabilities.toml` declares its own
`evidence` field, and the completeness check resolves exactly that artifact —
it no longer unions `capabilities_covered` from every JSON under the evidence
root (that scan counted FAIL receipts as coverage). The check built in **v2b
Part 3** is now

    python3 tools/v2_session_evidence_check.py --session <id> \
        --window-start <iso> --window-end <iso> --catalog <ops.sqlite> --json

and reads three real sources, keyed to each row's `evidence` field:

- `job:<kind>` rows: a `succeeded` attempt of that job kind inside the window
  in the ops catalog (opened read-only), falling back to a `delivered` outbox
  effect of the same kind; the outbox has no timestamp column, so that
  fallback is reported with `window_checked: false` rather than silently
  accepted as in-window evidence;
- `route:<METHOD> <path>` rows: a 2xx row for that method/path in this
  session's own `route_probe/<session>-route_probe.json` — a receipt for
  another session is never cross-counted;
- `cli:<tool>` rows: a `resource_measurement/*.json` record whose `command`
  names that tool, with `exit_code == 0`, `killed == false` and `started_at`
  inside the window (the direct fix for "count only PASS / exit 0 & not
  killed").

`exempt` rows and `missing`/`dormant-historical` dispositions never need
evidence. Rows declared `evidence = "open"` are a known gap: they are
reported as OPEN and the check FAILs (exit 1) while any remain — open is never
counted as covered and never reported as uncovered. A row whose declaration
has no `evidence` key fails closed; a broken JSON receipt is named under
`unreadable_evidence_files` and the scan carries on.

## 9. Real-scale research-tools smoke

`tools/v2_fill_quality.py` and `tools/v2_signal_screen.py` (capability id
`research-tools`) must be smoke-tested at real scale — a real pinned snapshot,
not a synthetic fixture — as part of rehearsal:

    python3 tools/v2_fill_quality.py --catalog <catalog.sqlite> \
        --store-root <objects> --snapshot-id <snap_...> [--since <date>] \
        [--csv <out.csv>] --reports-dir <dir>
    python3 tools/v2_signal_screen.py --catalog <catalog.sqlite> \
        --store-root <objects> --snapshot-id <snap_...> --reports-dir <dir>

Both read bounded `Repository.scan` slices of one snapshot (explicit
`--snapshot-id`, or the scope's pinned head) and write the snapshot id into
their outputs. Note plainly: `nightly-features`/`legacy_features` (the
Tier-3/Tier-4 rebuild stage) cannot succeed in barrier mode — it globs a
data-dependent file set no static captured manifest can enumerate
(`engine/v2/ops/capture_inputs.py::UNCAPTURED_KINDS`). The smoke runs must use
real captured inputs the same way a real nightly does, never a bare
barrier-mode stub.

## 10. Qualified sessions

Define, do not schedule. A session is qualified per
[Phase 7 — Cutover](rearchitecture_phase7_cutover.md) P7-2 only when all of
the following hold:

- real dated evidence for a completed session;
- the full expected population was scored;
- selfcheck, engineering and publication all succeeded;
- repeated attempts count once, never as two sessions;
- missing sessions stay unknown — do not substitute or backfill;
- the session's route probe (`tools/v2_route_probe.py` against the running
  preview server) and the evidence-completeness check both use the SAME
  `--session` id and window as the qualified-session receipt; a qualified
  session with `rows_uncovered` non-empty is not fully evidenced even if
  scoring/selfcheck/publish all passed — report which rows, don't block the
  session's trading validity on it (evidence completeness and trading
  correctness are separate gates);
- **manual CPU placement disqualifies a run** — only `bounded_run.py`'s
  automatic admission counts.

Scheduling the ten nights — night count, start date, provider budget, who
watches — is Slice 12, the next step after this package; it is out of scope
here and is linked when that slice lands. This runbook only defines what makes
a session qualified and how each one is measured, backed up and evidenced
(§5–§9).
