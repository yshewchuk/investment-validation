# `engine/dashboard` — architecture (legacy)

Legacy tree (root `/ARCHITECTURE.md` §1): runs the production board today,
frozen until Phase 7/8 cutover. See the companion native doc,
`engine/v2/dashboard/ARCHITECTURE.md`, for the package that eventually
replaces its formatting half. This legacy package alone owns publication
to `dashboard/published/**` — `engine/v2/dashboard` never writes there, it
only composes and starts the serving preview — so this producer code is
the last review checkpoint before anything reaches that path (excluded
from CodeRabbit review by `.coderabbit.yaml`'s `path_filters`, and from the
secret scan by `checks/repo_hygiene.py`).

## Purpose

The legacy nightly orchestrator, board renderer, selfcheck re-scorer,
model-evidence page builder and atomic bundle publisher. `nightly.py`'s
own module docstring states the load-bearing order: refresh → validate →
warm Tier-4 serving models (subprocess) → score → ledger → render → selfcheck → publish → flags → backup, each step
gating the next. Idempotent by construction: re-running re-reads,
re-renders, re-publishes, and the ledger refuses duplicate `row_id` writes.

## Primary contracts and public interfaces

- `nightly.py` — the orchestrator; run as the legacy nightly entrypoint
  (root doc §4). Public helpers for the Tier-4 gap repair:
  `repair_tier4_gap`, `repair_tier4_gap_if_needed`, `mark_tiers_degraded`.
  `--fetch-only` (`run_nightly(fetch_only=True)`) runs only the steps whose
  output the v2 snapshot import reads (`engine/v2/data/import_snapshot.py`,
  `reference_inputs._tier4_caches`): universe, refresh (Tier 2
  `data/curated/*`), finality, validate, moves, tiers (panel, Tier 4,
  SNAPSHOT), the panel-staleness flags and warm_serving (the import refuses
  `TIER4_CACHE_STALE` without a serving cache for the current panel sha). It
  skips score, ledger, settle, ladder, backfill, model_evidence, render,
  selfcheck, publish and backup. Each skipped step is reported as
  `{"skipped": true, "reason": "fetch_only"}` and the report keeps the usual
  keys (`as_of`, `finality`, `steps.tiers` with an explicit `degraded`).
- `warm_serving.py` — `warm(event_dates, decision_start)`: before score
  (default scorer only; `--fetch-only` also runs it, with no score after), fits every Tier-4 serving model (all `PRODUCES`) whose
  cache file for the current panel sha is missing or not ready (bad metadata,
  or no embedded residual pool), in a SUBPROCESS via `tier4.serving_model`
  (same cache files). Folds are a superset: every month from
  `decision_start - 30 days` (STR-RUNUP enters 14 trading days early) to the latest event's month, because each
  strategy's own decision date picks the fold. If the panel sha changes, the
  child stops installing and discards its staged file (fits already installed
  are keyed by the old sha). Failure raises `WarmError`; the nightly
  stops with `WARM_SERVING_FAILED` and never refits in-process.
- `render.py` — `build_meta`, `build_health`, `build_strategies`,
  `compact_row`, `row_digest`, `freshness_summary`, `quota_state`,
  `size_model_mae_from_ledger`, `payoff_curve`: the board's row/meta/health
  documents. Several of these names are the ones
  `checks/legacy_adapters.json` records as reused by `engine/v2/ops` (see
  the root doc §3's adapter note).
- `nightly.py::history_backfill_tickers` — the per-ticker history backfill
  set: the tickers with an upcoming earnings event PLUS `nightly.CONTEXT_TICKERS`
  (the market-context tickers the scoring path reads from `daily_market`, e.g.
  `"SPY"`, which never has earnings). Legacy cannot import `engine.v2`, so this
  constant mirrors the owner, `engine.v2.features.panel_row_inputs.CONTEXT_TICKERS`;
  a v2 test asserts they are equal. `refresh_calendar_data` backfills this set, so a
  context ticker's full history is fetched once and an already-cached one
  costs no call.
- `selfcheck.py` — `SelfCheckReport`, `selfcheck()`, `reconstruct_request`,
  `scrub_mismatches`: re-scores board rows directly through the engine and
  stops the publish on a mismatch.
- `model_evidence.py` — `load_model_evidence`, `evidence_path`,
  `sanitize_reason`: the model-evidence page and its free-text
  sanitizer (see Invariants), also used by `nightly.py`.
- `publish.py` — `publish_bundle`, `LocalPublisher` (atomic
  release-symlink flip, the default target), `CommandPublisher` (a shell
  command target; see Failure semantics for what "atomic" does and does
  not cover there), `secret_scan`, `access_probe`: the publication path
  and its own pre-publish secret scan of the rendered bundle.

### Memory diagnostics (opt-in)

`engine/memprobe.py` — four env flags (`INVESTING_PLAN_MEMPROFILE`, `_TRIM`,
`_EVERY` default 250, `_TOP` default 12). Off = no output and no gc walk; on,
`[mem]` lines go to stdout. `_TRIM` runs `gc.collect()`, releases the pyarrow
pool (`release_unused`) when pyarrow is already imported, then libc
`malloc_trim(0)` before measuring; the before/after RSS pair (`rss=` plus the
`[trimmed from X]` suffix on the head line) is printed only under `_TRIM` and
only when the pyarrow release or malloc_trim actually ran — without them a
single RSS value is printed. Optional probes
are silently omitted when unavailable (pyarrow rows only if pyarrow is
already imported; map rows only if `/proc/self/smaps` is readable; rss
anon/file/shmem rows only if `/proc/self/status` is readable) and never print a failure line; only an
unexpected exception in the snapshot body prints `probe failed` — but the
before/after RSS pair obeys the `_TRIM` rule above. Hooks:
`NightlyReport.mark`, `score_calendar`, `load_chain_index`, `Scorer` caches.
Estimates are approximate; some paths sample values, and those samples may
transiently materialise part of a Series or object column. A `MemoryError` propagates into the run;
`estimate_size` returns 0 on any other failure, while `snapshot` prints a
probe-failure diagnostic and returns `None`. `NightlyReport.mark` restarts its
step clock after the probe, so profiling time is not charged to the next step. The native (`engine/v2`) nightly is NOT
covered: `engine/v2` must not import legacy `engine` (layer rule); legacy
adapters invoked from v2 run unprofiled. Each snapshot also logs RSS
anon/file/shmem, the pyarrow pool (backend, allocated, max; `_TRIM` also
calls `release_unused()`), the largest mappings, untracked numpy arrays
found via their holders, and gc object counts by type.

### Post-score memory release

| Condition | Outcome |
|---|---|
| `score_calendar` returns, before ledger replay | `Scorer.release_score_caches()` empties the `live_features` cache (pure memoization; a later call recomputes the same value) and calls the best-effort `_release_free_pages` (a platform without `malloc_trim` returns nothing to the OS) |
| Backfill of missed nights finishes | the same release runs again, because backfill scoring repopulates the cache |
| A stage among score, ledger, settle, ladder, backfill, selfcheck ends | `[nightly-mem] <stage> rss=<G> peak=<G>` goes to stderr, independent of the memprobe flags; other steps are not logged |
| Any stage | the trades frame and chain indexes stay (ladder, replay, selfcheck read them) |

## Inputs

Tier-1/2 fetch data (via the quota-guarded fetch wrapper), the shared
`Scorer`/chain index, the frozen ledger record, and (for selfcheck) the
just-rendered board rows re-read through the engine.

## Outputs

`dashboard/published/**` (via `publish.py`'s atomic release flip:
`os.symlink` a new release directory, then `os.replace` the `current`
symlink onto it — a process killed mid-copy leaves the previous release
intact), plus flags (new gate triggers, earnings-date changes, calibration
drift, quota below reserve) and a backup sync.

## Dependencies

Imports observed in this package, top-level and lazy (an `ast` walk over
every `.py` file reporting every `engine.*` import at any depth, including
inside function bodies):

- Top-level: `engine.paths` (all four of `model_evidence.py`/`nightly.py`/
  `publish.py`/`render.py`), `engine.data.throttle` (quota guard, also
  lazy in `nightly.py`), `engine.score` (`LADDER_STEP` and others, also
  lazy in several functions; `strike_ladder` builds ONE chain index via
  `board_chain_keys`/`load_chain_index` and passes it to every `score()`), `engine.jsonio` (`json_safe`), `engine.features` (`model_evidence.py`:
  `DAILY_STATE_FIELDS`, for the projected `daily_market` read; other uses lazy),
  `engine.memprobe` (`nightly.py`, also provided to `engine.score` and
  `engine.replay`, which are outside this package).
- Lazy, `warm_serving.py` and `nightly.py::run_nightly`: `engine.data.features.tier4`
  (`serving_model`,
  `fold_start_of`, `feature_model`, `PRODUCES`, panel hashing through
  `store.file_sha256` and `paths.PANEL`, private `_serving_path`; the child
  temporarily reassigns `SERVING_DIR` to stage writes; `clear_prepared_frames`
  from `run_nightly`) and `engine.features`
  (`load_panel` in the child; `clear_panel_caches` from `run_nightly`).
- Lazy (function-local) only, never at module top level: `engine.calendar`
  (`model_evidence.py::_dataset_for`, `nightly.py::_date_change_flag`/
  `_panel_staleness_flags`/`refresh_forward_chains`/`run_nightly`);
  `engine.data` (`model_evidence.py::_daily_subset`/`_replay_trades`,
  `nightly.py::_panel_staleness_flags`/`_recently_printed`/`run_nightly`/
  `validate_refresh`, `render.py::freshness_summary`); `engine.data.fetch`
  (`nightly.py::backfill_ticker_history`/`refresh_calendar_data`/
  `refresh_forward_chains`/`run_nightly`); `engine.data.finality`
  (`nightly.py::run_nightly`); `engine.data.pulls`
  (`nightly.py::run_nightly`); `engine.data.pulls.forward_calendar`
  (`nightly.py::refresh_calendar_data`); `engine.data.rebuild`
  (`nightly.py::refresh_calendar_data`/`run_nightly`/`mark_tiers_degraded`);
  `engine.data.manifest` (`nightly.py::_write_flag_report`);
  `engine.data.normalize.n_daily`
  (`nightly.py::run_nightly`); `engine.data.schemas`
  (`model_evidence.py::_replay_trades`); `engine.entry_rules`
  (`render.py::build_strategies`); `engine.features`
  (`model_evidence.py::_champion_block_impl`/`_dataset_for`,
  `nightly.py::_panel_staleness_flags`/`run_nightly`,
  `render.py::build_strategies`/`size_model_mae_from_ledger`);
  `engine.fills` (`nightly.py::strike_ladder`,
  `selfcheck.py::reconstruct_request`); `engine.forecast_sizing`
  (`nightly.py::strike_ladder`); `engine.ledger`
  (`nightly.py::_calibration_flag`/`run_nightly`,
  `render.py::build_health`/`size_model_mae_from_ledger`);
  `engine.models.registry` (`model_evidence.py::build_model_evidence`,
  `render.py::build_meta`/`build_strategies`); `engine.models.training`
  and `engine.models.training.train_all`
  (`model_evidence.py::_dataset_for`); `engine.payoff`
  (`render.py::build_strategies`); `engine.portfolio`
  (`render.py::build_book`); `engine.replay`
  (`nightly.py::refresh_forward_chains`); `engine.structure_registry`
  (`render.py::build_meta`/`build_strategies`); `engine.structures`
  (`nightly.py::strike_ladder`, `render.py::build_meta`/`build_strategies`).

**No file in this package imports `engine.v2`**, matching the root doc §3
rule 3 ("legacy never imports v2") — verified by the same mechanical scan
(zero `engine.v2*` hits), not assumed.

Callers: the legacy nightly cron/manual trigger; `engine/v2/ops`'s
declared adapter (`checks/legacy_adapters.json`) reuses several `render.py`
and `selfcheck.py`/`ledger`/`portfolio` names, listed by exact symbol, so
v2's supervised legacy actions call this package's functions directly
rather than reimplementing them.

## External systems and libraries

The Tier-1/2 data providers behind the fetch wrapper; the legacy ledger
and panel filesystem state; the publish target's filesystem (or a remote
publish command, per `publish.py`'s `CommandPublisher` variant);
`fcntl` for the nightly's own run lock.

## Failure semantics

- **CLI report** — after `run_nightly` returns a report, the CLI atomically
  writes `reports/nightly_<requested date>.json`; a distinct `--json`
  destination is also written atomically when requested. A standard-report
  write failure is logged to stderr and returns exit code 1 before the
  optional destination is attempted; optional-write failure is also logged
  and returns 1 after the standard report has been saved.
- **Missing input / refresh failure** — degrades to cached data (staleness
  stays visible in `meta.json`) — except a rotated credential, which stops
  the run rather than burning retries against a dead key.
- **Tier-4 gap repair** — with refresh and tiers on, at most one SNAPSHOT per run:
  on the repair path it is written after the Tier-4 gap check; a degraded
  tiers step writes its own (see table); a failed write keeps the previous
  file.
  A stop before the tiers step (after the store rebuild) leaves the previous
  SNAPSHOT in place; a later stop (selfcheck) keeps tonight's. The retained
  SNAPSHOT is the import gate, not a hash of the current working store: after
  a refused repair the store can be ahead of it (user decision, fail-closed
  readers tracked in #771). The flag report is written atomically (a failed
  write keeps the previous file); for a tiers refusal it is best-effort.

  | Condition | Outcome |
  | --- | --- |
  | Warm step (default scorer, or `--fetch-only`): subprocess exits non-zero, cannot start, times out (3 h), fails to plan/verify, leaves a fit missing or not ready, or the panel changed under it | `NightlyStop` `WARM_SERVING_FAILED`; exit 1; nothing scored; no in-process refit |
  | Incremental build leaves an out-of-window gap (`tier4_gap_partial`) | Full Tier-4 rebuild (no `--since`); no gap flag once repaired |
  | Repair raises / exceeds budget / leaves the gap / its finalize fails | `NightlyStop` with `TIER4_REPAIR_FAILED`, `TIER4_REPAIR_BUDGET` or `TIER4_GAP_REMAINS`; exit 1; nothing scored or published; no new SNAPSHOT (the previous one is kept), so the native import sees no new session for the night (`not_yet`, unless an earlier run that night already wrote one) |
  | With tiers on, run stops between refresh and tiers (red validation, as-of finality stop) | Store refreshed, no new SNAPSHOT; the previous SNAPSHOT is kept (with `--no-tiers` the refresh step has already written one) |
  | Tier-3/4 rebuild raises | `tiers_degraded` (error text sanitised); the run continues and scores the store as found; the degrade step writes a SNAPSHOT of that store (the deferred refresh snapshot, when refresh ran) so the scorer's digest describes the store scored; if that write fails the run stops with `NightlyStop` `TIERS_SNAPSHOT_FAILED` (exception type only), nothing scored |
  | No gap, finalize fails | `tiers_degraded` as above |
  | Repaired, or no gap | SNAPSHOT written once |

- **`--fetch-only`** — requires refresh and tiers on and no backup: with
  `--no-refresh`, `--no-tiers` or `--backup` the CLI exits 2 (argparse) before
  running anything. Gating failures behave as in a full run (red validate,
  non-final session, rotated credential, typed `TIER4_*` stop: `NightlyStop`,
  exit 1, no report), and so does a warm_serving failure
  (`WARM_SERVING_FAILED`, see the table). Before warm_serving the run releases
  its event frames, the Tier rebuild result and the process-global panel,
  Tier-4 prepared-frame and daily-index caches, so the warm subprocess does not
  overlap the parent's panel memory; the failure contract is unchanged.
  `run_nightly(fetch_only=True)` with
  refresh off, tiers off or backup on raises `ValueError` before any work
  (the CLI turns the same rule into the exit 2 above). Refresh and tiers degrade as in a full run and are
  recorded in `steps`/`flags`. The run writes the flag report and the
  standard `nightly_<date>.json`, never writes the nightly state file
  (`last_successful_as_of`, calendar), so a later full run still back-fills
  the nights this one did not score, and never touches the ledger, bundle or
  publish target. A degraded `tiers` step is visible to
  `nightly_readiness._require_ready`, which refuses it as for a full run.
- **Validate** — a red validation stops the pipeline; yesterday's snapshot
  stays published and a flag is raised.
- **Selfcheck mismatch** — stops the publish; the nightly records raw
  mismatches, v2's adapter persists `scrub_mismatches` row/field/reason only.
  Board and selfcheck use the same chains: no entry-date chain prices off the
  newest older one within the configured age limit (`STALE_QUOTE` +
  `quote_date`); if that one postdates the board night it is `NO_CHAIN`.
- **Publish** — `LocalPublisher` (the default target, a directory) is
  atomic: it stages the full bundle under `releases/{stamp}/`, then flips
  `current` with one `os.replace`, so a process killed mid-copy leaves the
  previous release serving. `CommandPublisher` (a shell-command target,
  e.g. `wrangler pages deploy {bundle}`) carries no atomicity of its own —
  it runs the configured command and checks its exit code; remote
  atomicity holds only when that command's own deployment contract is
  atomic, which this code neither verifies nor provides. Either way a
  down target never blocks — the local bundle still renders, and the
  retry is next night's. `secret_scan` runs over the rendered bundle
  before publish, independently of `checks/repo_hygiene.py` (which does
  not scan `dashboard/published/**` at all).
- **Backup** — a failure raises a flag but never blocks the publish; the
  snapshot and the backup are independent.
- **Idempotency** — re-running re-reads, re-renders, re-publishes; the
  ledger refuses a duplicate `row_id` write rather than silently
  overwriting or double-counting it.
- **Memory diagnostics** — condition → outcome (the `_EVERY`/`_TOP` sampling
  note above still applies):

  | Condition | Outcome |
  | --- | --- |
  | Probe disabled (flag unset) | Snapshot/tick returns before any work; no output, no gc walk |
  | pyarrow not imported | No arrow row; no pool release |
  | `/proc/self/smaps` unreadable | No map rows |
  | `/proc/self/status` unreadable | No rss anon/file/shmem row; the head line shows `rss=0.00G peak=0.00G` |
  | Ordinary probe failure (any non-`MemoryError` exception, including a broken stdout) | Prints `[mem] <label> probe failed: <ExcType>: <repr>` when stdout works; the run continues |
  | `MemoryError` anywhere in a probe | Propagates to the caller (nightly `MemoryError` handling unchanged) |
  | Trim flag set but libc lacks `malloc_trim` | `malloc_trim` skipped, but the pyarrow pool release still runs; RSS lines still printed |

## Invariants

- Never imports `engine.v2` (root doc §3 rule 3).
- No change lands here to support v2 work outside of a signed-off decision
  (root doc §5, "legacy board, nightly and ledger stay unchanged until
  cutover").
- Free-text fields that reach the bundle are sanitised before publish, not
  passed through raw: `model_evidence.py::sanitize_reason` and
  `selfcheck.py::scrub_mismatches` are the existing enforcement points for
  this package; a new free-text field added anywhere in this package
  (a new `degraded_reason`, a new flag detail, a caught exception's
  `str()`) must go through an equivalent sanitizer before it can reach
  `dashboard/published/**` (root doc §5).
- Nothing published carries a local path or raw exception text (root doc
  §5) — `publish.py`'s own `secret_scan` is this package's belt-and-suspenders
  check on top of the sanitizers above, since `checks/repo_hygiene.py`
  does not cover this path.
