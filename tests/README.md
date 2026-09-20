# Testing strategy

**Tests mirror behaviour, not files.** There is deliberately no one-to-one
`engine/x.py` → `tests/test_x.py` mapping. What there *is* — and what
`checks/phase0_checks.py --only test_policy` enforces — is that every engine
module is covered by one of two layers, and that the assignment is declared
rather than assumed.

## The two layers

**Layer 1 — unit tests (`tests/`, pytest).** Pure logic, exercised with
fixtures, no network and no real data store. Fast enough to run on every edit
(~7s for 500+ tests). This is where the invariants live: fill arithmetic,
session-aware date math, schema enforcement, leak discipline, the standing bans.

**Layer 2 — acceptance checks (`checks/phase0_checks.py`, `checks/phase1_checks.py`,
`checks/phase2_checks.py`).**
Everything whose correctness is only meaningful against the real thing: 15.3M
real chain rows, the 115,500-row legacy panel, real trained artifacts, a real
git repo, a real clone. A fixture cannot prove that the rebuilt panel reproduces
the master panel; only the master panel can.

Some modules are covered by Layer 2 *by design*, and show 0% under `pytest`:

| Module | Covered by | Why not a unit test |
|---|---|---|
| `engine/data/rebuild.py` | `determinism`, `migration`, and the real rebuild | It is an orchestrator; mocking every stage would test the mock |
| `checks/phase0_migration.py` | `migration` (+ `tests/test_migration_logic.py` for its delta logic) | The claim is about 115,500 real rows |
| `checks/phase0_verdicts.py` | `verdicts` | The claim is that published numbers reproduce from real chains |
| `checks/phase0_audit.py` | `coverage_report` | Renders a report over the real store |
| `checks/phase0_checks.py` | it *is* the harness | — |
| `engine/build_trades.py` | phase 1 `replay_equivalence` | It drives the replay across 42k events; a fixture-scale run would test the argument parser |
| `engine/models/training/train_all.py` | phase 1 `registry` | Its output *is* the artifacts the registry check verifies |
| `checks/phase1_replay.py` | phase 1 `replay_equivalence` | It is a check |
| `checks/phase1_checks.py` | it *is* the Phase 1 harness | — |
| `checks/phase2_checks.py` | it *is* the Phase 2 harness | — |

That table is not documentation-by-good-intentions: `test_policy` reads it and
fails if a module appears in neither layer.

## What gets a test, and what kind

Not everything deserves equal weight. The ordering used here:

1. **Guards that enforce a standing rule** get tested first and hardest,
   because a guard with no test is a guard you do not have. Examples: the
   oquants model-fitted-marks ban (`test_sources.py`), the `exit_mode == "chain"`
   look-ahead filter (`test_normalize_trades.py`), ORATS token redaction, the
   quota reserve floor.
2. **Negative controls** for anything load-bearing. The migration test is the
   single check licensing "we changed no number that matters", so
   `test_migration_logic.py` mostly asserts that it *fails* when it should — a
   green check whose red state is unreachable proves nothing.
3. **Unit and convention traps**, each pinned to the evidence that established
   it: the three ORATS market-cap eras, FLT_MAX sentinels, `spy_vol20` being
   simple returns with `ddof=1`, the crossed-quote repair.
4. **Ordinary behaviour** — round trips, dtype handling, empty inputs.

Tests state *why* a behaviour matters where the reason is not obvious from the
assertion. A test that only says `assert f(x) == y` documents an implementation;
one that says why `y` is the right answer documents a decision.

## Running

```bash
python3 -m pytest tests -q                 # layer 1, ~20s
python3 checks/phase0_checks.py            # data foundations
python3 checks/phase1_checks.py            # scoring engine
python3 checks/phase1_checks.py --no-data  # layer 1 + the checks needing no store
python3 checks/phase2_checks.py            # experiment framework + EXP-050 regression
python3 -m coverage run --source=engine,checks,tools -m pytest tests \
  && python3 -m coverage report --sort=cover -m
```

### Running the whole suite on this box (12 cores, 7.7 GB)

Run it from the main checkout. A git worktree has no `data/` or
`fixtures/tier0/` (both gitignored), so the data-dependent tests fail there
with `FileNotFoundError`.

```bash
cd /root/investing-plan
free -m; ps -eo pid,rss,args --sort=-rss | head    # who else holds memory
python3 -m pytest -q -n 4 --dist loadgroup -rfEs --durations=15 tests/ 2>&1 | tail -60
```

- **`-n 4 --dist loadgroup`**: 4 workers when you are the only test runner.
  Use `-n 2` while a heavy job (nightly, D14, real scoring) or another agent's
  tests are running. `loadgroup` keeps the `xdist_group("serial")` files on
  one worker (see `conftest.py`).
- **Don't narrow the CPU affinity below 4 CPUs.** Don't run it under
  `taskset`, or `bounded_run --cores`/`--cpu-set`, with fewer than 4 CPUs.
  Real-`Service` tests admit jobs against this process's own affinity minus 1
  reserved CPU. `ops_support.TEST_POLICY` caps every profile at 3 worker CPUs
  (production `legacy_score`/`legacy_rebuild` need 5), so 4 CPUs is the floor,
  the size of a standard GitHub runner. Below that the heavier profiles can
  never be admitted.
- **`-rfEs`** lists every failure, error and skip with its reason in the
  summary, so a skip can't go unnoticed.

Nothing in the suite waits unboundedly any more:

| Guard | What it does | Knob |
|---|---|---|
| Per-test timeout (`conftest.py`) | Each setup/call/teardown phase gets 600 s; past that the test **fails** as `TIMEOUT: <node id> ...` with the stack it was waiting in, and the run continues. pytest-timeout is not installed; this is a SIGALRM equivalent that switches itself off if pytest-timeout is ever installed. The first `ui_dist_dir` setup (npm ci + build) gets a longer budget. | `--test-timeout N`, env `PYTEST_TEST_TIMEOUT`, `@pytest.mark.test_timeout(N)`; `0` disables |
| Admission waits (`ops_support.run_until` / `AdmissionWatch`) | A real-`Service` job left queued on a host-resource reason **fails** the test with `RESOURCE WAIT (environment, not a code failure): ...`, quoting the queue reason's numbers ("needs X GiB free, MemAvailable is Y"). `PROFILE_EXCEEDS_CAPACITY` (CPU affinity or host total too small) fails at once; `MEMORY_HEADROOM`, `CPU_UNAVAILABLE`, `DISK_SPACE` and `RESERVATION_BUDGET` fail after 60 s continuously queued. Reasons from the test's own catalog (dependencies, heavy slot, store lease) are left to the test. | env `OPS_TEST_ADMISSION_WAIT_S` |
| `ui/.npm-ci.lock` | Waits for another process's `npm ci` for at most 660 s, then fails, naming the lock. | none |

These are failures, not skips, on purpose: the check did not run, and a
skip is easy to miss when the rest of the summary is green. To separate
environment from code, `grep -E "RESOURCE WAIT|TIMEOUT:"` the output and rerun
those files alone on a quieter box. `TEST_POLICY` caps every profile at
256 MiB, so a memory `RESOURCE WAIT` means MemAvailable was below about
0.75 GiB for a whole minute.

## CI tests

GitHub Actions runs `.github/workflows/tests.yml` on every push to `main` and
on `workflow_dispatch`. The workflow deselects tests with four markers:
`needs_data`, `needs_corpus`, `heavy_host` and `browser`. Tests with these
markers require local resources (untracked data directories, real multi-GB
workers, or a browser environment) and run locally instead.

The four markers, and which tests carry them, are:

| Marker | Meaning | Tests |
|---|---|---|
| `needs_data` | reads the real `data/` root (gitignored, absent in CI and worktrees) | tests/test_calendar.py (10), TestChainRefreshCoversTheGap, TestPanelStalenessGuard, TestPanelFeaturesStaleMarketBlock, TestLiveFeaturesBoundedByTheDecision, test_checks_phase3b_real.py::test_table_run_reopens_persisted_clean_rebuild_and_downstream_scan, 3 incremental tests |
| `needs_corpus` | reads `fixtures/tier0` or another untracked fixture tree | tests/test_phase4_completion_review.py (whole file) |
| `heavy_host` | launches real multi-GB workers; run alone on a quiet box | tests/test_v2_ops_supervised_legacy.py, tests/test_v2_ops_nightly_completion.py (whole files) |
| `browser` | drives a real Playwright browser or needs node/npm (`ui/` build) | tests/test_v2_dashboard_browser.py, test_v2_dashboard_integration.py, test_v2_ops_serving_browser.py (whole files), test_l02_browser_frame_pins_r1_then_reload_shows_r2 in test_v2_dashboard_preview.py |

CI runs everything except these four categories:

```bash
python -m pytest -n auto --dist loadgroup -m "not needs_data and not needs_corpus and not heavy_host and not browser" -rfE --durations=25 --junitxml=junit.xml tests/
```

To run the local complement from the main checkout (not a worktree):

```bash
python3 tools/bounded_run.py --cores 8 -- python3 -m pytest -q -n 4 --dist loadgroup -m "needs_data or needs_corpus" -rfEs tests/
```

The `heavy_host` and `browser` tests keep their existing procedure (see above).

## Known thin spots

Honest list, so nobody has to rediscover it:

- **`engine/data/sources/*.request()`** — URL building, quota parsing, auth
  detection and the endpoint bans are all tested; the actual HTTP round trip is
  not, because the ORATS quota is spent. The Sep-1 pull is that seam's first
  real exercise.
- **`engine/data/coverage.py`** rendering helpers (~50%) — the coverage
  *arithmetic* is tested; the Markdown table formatting is exercised only via
  the `coverage_report` acceptance check.
- **`engine/calendar.py`** loaders (~78%) — the holiday rules, session mapping
  and day arithmetic are covered; `load_orats_earnings` / `load_oquants_event_dates`
  are I/O over the real cache and run in the acceptance layer.

## Mutation-testing pilot

A pilot, not monitoring: it answers whether the targeted tests would notice a
silent arithmetic or logic slip in six critical modules. Tool: mutmut 3.8.0,
installed per user (`pip install --user --break-system-packages mutmut==3.8.0`).
Config: `tools/mutation_pilot.toml` (modules, why each, their test files,
per-mutant timeout). Driver and report: `tools/mutation_pilot.py`.

- Each module mutates only its own files and runs only its own test files.
  Never the whole suite, never xdist. mutmut narrows each mutant further to the
  tests that executed the mutated function.
- All state lives outside the repo in
  `~/.cache/investing-plan-mutation-pilot/<module>/` (override with
  `MUTATION_PILOT_HOME`; the tool refuses a path inside a checkout). The work
  copy holds every tracked file and nothing else, like a CI checkout, so
  there's no `data/`.
- Runs are resumable. A rerun re-tests only functions whose source changed.
  `--fresh` wipes that module's state.

Run one module at a time, as a scheduled heavy job:

    python3 tools/mutation_pilot.py count            # mutant counts only, no tests
    python3 tools/bounded_run.py --max-rss-gb 2 --min-free-gb 0.5 --cores 4 -- \
        python3 -u tools/mutation_pilot.py run <module>
    python3 tools/mutation_pilot.py report [<module> ...] [--no-diffs]

Note: ops modules need 4 or more CPUs, because real-Service tests admit against affinity minus 1.

`report` prints, per module and per file: total, killed, survived, timeout,
no-tests, other and not-yet-run counts, plus the score
(killed + timeout) / checked. It then lists each survivor as `file:line` with
the mutation diff. It prints code only.

What to know before reading a score:

- mutmut 3 does not mutate decorated functions (`@property`,
  `@contextmanager`) or module-level constants. So `no_fit_guard` and the
  artifacts' `key` properties are outside the score. The key *builders*
  (`payoff_artifact_key` and similar) are in it.
- `process_isolation = forkserver` is load-bearing. mutmut's default `fork`
  forks every mutant from the process that already ran the tests, so state
  those tests left behind leaks into each mutant. In the smoke run, no_fit's
  thread-local flag made killable mutants read as survived. BLAS threads
  started before the fork hung others into false timeouts.
- A "no tests" mutant sits in a function that no targeted test executes. It
  counts against the score, the same as a survivor.

## Mutation CI

`.github/workflows/mutation.yml` runs the modules in
`tools/mutation_pilot.toml` on GitHub Actions, one matrix job per enabled
module, with `--max-children $(nproc)` (4 on a standard runner). It only
reports. Scores never fail a job. A job fails only when the tool does: mutmut's
clean test run fails, or the run hits its 330-minute step timeout.

| trigger | mode | state |
|---|---|---|
| push to main | incremental | restores the module's newest cached mutmut state |
| weekly (Sun 05:23 UTC) | full | no restore: every mutant from scratch |
| workflow_dispatch | full by default; untick `fresh` for incremental | `modules` picks a comma-separated subset |

- **Scope.** All of `engine/v2`, split into 23 modules plus the six pilot
  modules. The only legacy files are the pilot's `engine/pnl_sim.py` and
  `engine/models/no_fit.py`. `contracts` (with `engine/v2/__init__.py` and
  `evaluation/`) is listed as excluded: mutmut generates no mutants there.
  `data_legacy` (the legacy materialization/mapping/reference bridge into v2)
  is also excluded (2026-09-19): it is code the rearchitecture deletes on
  cutover, and it was the module whose 330-minute mutmut step timeout made
  the weekly full run time out at 5.5 hours (CI run 35458557499). Both stay
  listed, never run, so `tests/test_mutation_ci.py` still counts their files
  as owned. `tests/test_mutation_ci.py` fails if an `engine/v2` file is in no
  module, or in two.
- **Tests per module** are data-free files only, since CI has no `data/`. Each
  was run alone with no `data/` (2026-09-19). Tests are picked from those that
  import the module, most specific first, up to ~150 s of clean test time
  (40 s for `foundation`). Four tests that need `data/` or a browser are
  deselected in `[defaults] deselect`. Browser and npm test files are never
  selected.
- **Cache.** One `actions/cache` entry per module holds only mutmut's state:
  `mutants/**/*.meta`, `mutmut-stats.json` and the driver's
  `mutation-ci-state.json`. The key is
  `mutation-mutmut<ver>-py<ver>-<hash of mutation_pilot.toml>-<module>-<sha>-<run id>-<attempt>`.
  Restore uses the same key without the sha, so each run gets the newest
  state. The work copy and mutated files are rebuilt from the checkout. mutmut
  then keeps every verdict whose function hash is unchanged. Editing the toml,
  bumping mutmut or changing Python restarts every module.
- **What incremental re-tests.** mutmut re-tests a mutant only when its own
  function's source changed. The driver adds one rule: when a module's
  selected tests or the shared `tests/*.py` helpers change, it resets that
  module's survived and no-tests verdicts. Two things wait for the weekly full
  run: a killed verdict that a weakened test would now let survive, and a
  change in a helper function outside the module.
- **Partial runs.** State is saved even after a timeout. The next run resumes
  it, and unreached mutants show as `skipped`.

### Report files

Each job uploads `mutation-module-<module>` (90 days). It holds
`results.jsonl`, `summary.json` and `summary.md`, which is also the job
summary. The job summary shows the score table and each untriaged survivor in
a function the push changed (`git diff <before>..<sha>`), with its diff. Runs
that are not pushes list the survivors this run re-tested instead. The
`report` job merges every module into one `mutation-report` artifact
(90 days).

`results.jsonl` has one row per mutant, including mutants a run did not
re-test. Its fields (`schema_version` 1):

| field | meaning |
|---|---|
| `schema_version` | 1; bumped on any incompatible change |
| `run_id`, `sha`, `ref`, `trigger` | Actions run id, commit, ref, event (`local` for a local export) |
| `mode` | `full`, `incremental` or `local` |
| `module`, `file`, `function` | toml module, repo path, `func` or `Class.method` |
| `line` | file line of the first line the mutation changes |
| `mutant_name` | mutmut's name (`pkg.mod.x_func__mutmut_N`) |
| `status` | `killed`, `survived`, `no_tests`, `timeout`, `suspicious` or `skipped` |
| `mutmut_status` | mutmut's own label (`type check` maps to killed; `segfault` and `interrupted` map to suspicious; `not checked` maps to skipped) |
| `retested_this_run` | true if this run decided it; null when unknown (local) |
| `diff` | unified diff of the mutated function (code only) |
| `triage` | null, or `{verdict, note, stale}` from the triage file |

`summary.json` holds these counts for the module and for each file: `total`,
each status, `checked` (total minus skipped), `score` and
`survived_untriaged`. It also has `run_exit_code` (-1 means the step timeout
killed the run) and `elapsed_seconds`. `score` is (killed + timeout) / checked,
so `no_tests` counts against it. The merged `summary.json` adds overall totals
and a `modules` map.

### Triage file

`tools/mutation_triage.toml` has one `[[triage]]` table per reviewed
survivor. Each table has a `mutant` (its exact name), a `verdict` (`EQUIVALENT`
or `LOW-VALUE`), a `note`, and an optional `diff_contains` that pins the
mutated text. mutmut numbers mutants per function, so editing the function can
point an old name at a different mutation. When `diff_contains` stops
matching, the row's triage is marked `stale` and the survivor counts as
untriaged again. The file is empty for now; the header documents the format.

### Querying: `tools/mutation_report.py`

By default it reads the latest completed main run's `mutation-report`. It
fetches with `gh` into `~/.cache/investing-plan-mutation-report/<run id>/`
(`MUTATION_REPORT_CACHE`; paths inside the repo are refused). `--run ID`,
`--sha SHA` and `--dir PATH` choose another source. `--local [MODULE ...]` is
offline and reads the pilot's own work copies.

    # untriaged survivors in one module, with diffs
    python3 tools/mutation_report.py --module ops_decisions --untriaged --diff
    # survivors and no-tests mutants under scoring, as CSV
    python3 tools/mutation_report.py --file 'engine/v2/scoring/*' --status survived,no_tests --format csv
    # one function's mutants in one run, as JSON lines
    python3 tools/mutation_report.py --run 123456789 --function 'Scheduler.*' --format jsonl
    # survivors in functions changed since a commit (local git needs both commits)
    python3 tools/mutation_report.py --changed-since 43a3ae1 --untriaged
    # score trend over the last 10 main runs, with the change from the previous run
    python3 tools/mutation_report.py --history 10 --module canonical,pnl_sim
    # offline, from local pilot results
    python3 tools/mutation_report.py --local no_fit canonical --untriaged
    # a downloaded or locally exported directory
    python3 tools/mutation_results.py export canonical --out /tmp/mut/canonical
    python3 tools/mutation_report.py --dir /tmp/mut/canonical --status survived

Filters combine. `--file` and `--function` take exact names or globs. `--status`
takes a comma list. `--untriaged` means survived or no_tests with no
current triage entry. `--format` is `table` (the default), `jsonl` or `csv`.
In CSV, triage is flattened into `triage_verdict`, `triage_note` and
`triage_stale`. `--history N` prints one row per run and module, plus an `ALL`
row, with score and delta.
