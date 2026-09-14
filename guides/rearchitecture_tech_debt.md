# Rearchitecture tech debt

Deferral rule (user decision, 2026-09-13): work that any later
rearchitecture phase needs is finished in its own phase. Only nice-to-haves
that can safely wait until **after Phase 6** are recorded here. Nothing on
this list may be a prerequisite of Phases 3–6. If a phase guide comes to need
an entry, move it into that phase and delete it here.

Each entry: what was deferred, why it can wait, the risk while it waits, and
what should trigger revisiting it.

| ID | Item | Origin | Why deferrable | Risk while deferred | Revisit when |
|---|---|---|---|---|---|
| TD-1 | Process-local object-verification cache keyed by `(device, inode, size, mtime_ns)` (Phase 2 guide §8.2 step 6). | P2-4 planning | The repository re-hashes every object it opens. Correctness is unchanged; only speed is lost. | Slower scans of large `option_chains` fragments. | A read path measurably exceeds its time budget, or live latency matters (Phase 7). |
| TD-2 | `TimeInterval` start < end compares canonical strings. | P2-1a | Correct for dates, naive timestamps, and UTC timestamps produced by `format_timestamp`. | A non-UTC offset written by hand would compare wrongly. | Any contract accepts caller-supplied offsets other than UTC. |
| TD-3 | `artifact_check`'s `CheckParameters` carries an optional `input_bindings` field used only by the cache-identity tests. | P2-5 B1a | A test seam on a production kind; no production caller sets it. | A production caller could start depending on it. | A non-legacy production kind exists that those tests can use instead. |
| TD-4 | Code shaped to fit the eight-module import fan-out budget. `engine/v2/data/legacy_adapter.py` builds contracts in two passes and hard-codes three legacy paths (tier-0 tested against `engine.paths`). `engine/v2/data/objects.py` uses `os`-only equivalents of `stat`/`uuid`/`math`. `objects.py` and `documents.py` import sibling modules as `from . import errors, time_formats`, which the budget counts as one import. | P2-1b, P2-2b | Behaviour is correct and tested; the cost is readability. | Future edits keep contorting around the budget. | The fan-out budget is revisited, or these modules are split. |
| TD-5 | Non-legacy subprocess helpers (`verify_export_generation`, `run_engineering_gate`, `run_security_scan`) live in `engine/v2/ops/legacy_adapter.py`, because `checks/import_layers.py` lets only `executor.py` and `legacy_adapter.py` call `subprocess`. | P2-5 task 5 | Behaviour is correct and audited in one place. | Blurs the legacy adapter's meaning: the helpers are not legacy bridges. | Before Phase 8 deletes `legacy_adapter.py` — move them to a dedicated audited subprocess module and update the import-layer allowlist. |
| TD-6 | `covered_tickers` was added to legacy `engine/data/finality.py` so decision evidence can carry per-ticker finality. | P2-5 B1c | It is additive and reuses the existing `session_finality` test. | Legacy code slated for deletion grew, and v2 depends on it through the adapter ledger. | Phase 3 moves finality into `engine/v2/data`, or before Phase 8 deletes legacy `engine/`. |
| TD-7 | Two coverage ratchets both measure `engine.v2.data`. The Phase 1 suite (`phase1_coverage_suite.v2`) includes `tests/test_v2_data_*.py` as an interim fix, and the Phase 2 script measures the package too. | P2-5 B1c, task 8 | Both baselines only rise, so nothing is hidden. | Every data change needs two baseline refreshes. | The Phase 2 gate closes: drop the data tests from the Phase 1 suite. |
| TD-8 | `export_generation(conn, root, *, generation, purposes=None)` exports every purpose by default. The nightly export stage passes `legacy_import` and `shadow`. | P2-5 task 5 | The only production caller filters correctly. | A new caller that forgets the filter mixes `research_reconstruction` rows into legacy ledger files. | A second caller appears; then make `purposes` required. |
| TD-9 | `tests/ops_support.py::TEST_POLICY` copies CPU and thread counts from `DEFAULT_POLICY`, because environment refs are derived from `DEFAULT_POLICY`. | P2-5 test fix | Tests stay memory-independent and env refs still match. | A production CPU change silently changes test env refs too. | Env refs become derived from the injected policy. |

## Tracked verification items (not debt)

These are scheduled Phase 2 work, recorded here so they are not lost. They
are not deferred.

- A Tier-4 serving-cache miss writes into the legacy root: `mkdir` and
  `joblib.dump` at `engine/data/features/tier4.py:1320-1321`, reached from
  `Scorer._serving` (`score.py:1967`). The materialized root is read-only, so
  a miss raises `PermissionError`. Caches are pinned only for the imported
  panel hash (currently `a67873b4eb95`). The heavy-run session must show
  that every (model, fold) the board touches hits a pinned cache; otherwise
  decide how shadow scoring handles a miss. A panel rebuild needs new caches
  before import.
- The `projection` profile reserves 2 GiB, but render loads the same panel
  context that peaked at 4.15 GiB in scoring. Measure it in the heavy-run session.
- Real replay identity, DYN-SV chooser rows included, is unproven until the
  frozen-data D18 run.
- The Phase 0 gate needs the private tier-0 corpus, so worktree agents cannot
  run it. The supervisor runs it on main.
- The Phase 0 gate is red on main since the Phase 2 merges
  (`tier1_real_replay`: "code changed since the replay ran"). Its tier-1
  receipt is bound to the engine code hash, by design, so any code change
  invalidates it. Re-run the tier-1 real replay and seeded controls on private
  data from the final Phase 2 commit (heavy run: needs `--max-rss-gb 6.5` and
  user approval), then run all three phase gates from that same commit.
