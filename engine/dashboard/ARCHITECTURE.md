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
score → ledger → render → selfcheck → publish → flags → backup, each step
gating the next. Idempotent by construction: re-running re-reads,
re-renders, re-publishes, and the ledger refuses duplicate `row_id` writes.

## Primary contracts and public interfaces

- `nightly.py` — the orchestrator; run as the legacy nightly entrypoint
  (root doc §4).
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
  `_sanitize_reason`: the model-evidence page and its own free-text
  sanitizer (see Invariants).
- `publish.py` — `publish_bundle`, `LocalPublisher` (atomic
  release-symlink flip, the default target), `CommandPublisher` (a shell
  command target; see Failure semantics for what "atomic" does and does
  not cover there), `secret_scan`, `access_probe`: the publication path
  and its own pre-publish secret scan of the rendered bundle.

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

Imports observed in this package, top-level and lazy (mechanically walked
by `.oc_logs/import_scan.py`, an `ast` walk over every `.py` file that
reports every `engine.*` import at any depth, including inside function
bodies) — 24 distinct legacy `engine.*` modules in total, not 3:

- Top-level: `engine.paths` (all four of `model_evidence.py`/`nightly.py`/
  `publish.py`/`render.py`), `engine.data.throttle` (quota guard, also
  lazy in `nightly.py`), `engine.score` (`LADDER_STEP` and others, also
  lazy in several functions), `engine.jsonio` (`json_safe`).
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
  (`nightly.py::refresh_calendar_data`/`run_nightly`); `engine.data.schemas`
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

- **Missing input / refresh failure** — degrades to cached data (staleness
  stays visible in `meta.json`) — except a rotated credential, which stops
  the run rather than burning retries against a dead key.
- **Validate** — a red validation stops the pipeline; yesterday's snapshot
  stays published and a flag is raised.
- **Selfcheck mismatch** — stops the publish (`scrub_mismatches`: sanitised
  row/field/reason). Board and selfcheck use the same chains: no entry-date
  chain prices off the newest older one within 5 sessions (`STALE_QUOTE` +
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

## Invariants

- Never imports `engine.v2` (root doc §3 rule 3).
- No change lands here to support v2 work outside of a signed-off decision
  (root doc §5, "legacy board, nightly and ledger stay unchanged until
  cutover").
- Free-text fields that reach the bundle are sanitised before publish, not
  passed through raw: `model_evidence.py::_sanitize_reason` and
  `selfcheck.py::scrub_mismatches` are the existing enforcement points for
  this package; a new free-text field added anywhere in this package
  (a new `degraded_reason`, a new flag detail, a caught exception's
  `str()`) must go through an equivalent sanitizer before it can reach
  `dashboard/published/**` (root doc §5).
- Nothing published carries a local path or raw exception text (root doc
  §5) — `publish.py`'s own `secret_scan` is this package's belt-and-suspenders
  check on top of the sanitizers above, since `checks/repo_hygiene.py`
  does not cover this path.
