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
- **Don't narrow the CPU affinity.** Don't run it under `taskset`, or
  `bounded_run --cores`/`--cpu-set` with fewer than 6 CPUs. Real-`Service`
  tests admit jobs against this process's own affinity minus 1 reserved CPU,
  and the `legacy_score`/`legacy_rebuild` profiles need 5 worker CPUs. Under a
  4-CPU affinity they can never be admitted.
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
  copy holds tracked `engine/ tests/ checks/ tools/` only, so there's no `data/`.
- Runs are resumable. A rerun re-tests only functions whose source changed.
  `--fresh` wipes that module's state.

Run one module at a time, as a scheduled heavy job:

    python3 tools/mutation_pilot.py count            # mutant counts only, no tests
    python3 tools/bounded_run.py --max-rss-gb 2 --min-free-gb 0.5 --cores 2 -- \
        python3 -u tools/mutation_pilot.py run <module>
    python3 tools/mutation_pilot.py report [<module> ...] [--no-diffs]

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
