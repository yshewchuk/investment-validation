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

**Layer 2 — acceptance checks (`checks/phase0_checks.py`).** Everything whose
correctness is only meaningful against the real thing: 15.4M real chain rows,
the 115,500-row legacy panel, a real git repo, a real clone. A fixture cannot
prove that the rebuilt panel reproduces the master panel; only the master panel
can.

Some modules are covered by Layer 2 *by design*, and show 0% under `pytest`:

| Module | Covered by | Why not a unit test |
|---|---|---|
| `engine/data/rebuild.py` | `determinism`, `migration`, and the real rebuild | It is an orchestrator; mocking every stage would test the mock |
| `checks/phase0_migration.py` | `migration` (+ `tests/test_migration_logic.py` for its delta logic) | The claim is about 115,500 real rows |
| `checks/phase0_verdicts.py` | `verdicts` | The claim is that published numbers reproduce from real chains |
| `checks/phase0_audit.py` | `coverage_report` | Renders a report over the real store |
| `checks/phase0_checks.py` | it *is* the harness | — |

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
python3 -m pytest tests -q                 # layer 1, ~7s
python3 checks/phase0_checks.py            # both layers
python3 checks/phase0_checks.py --no-data  # layer 1 + the checks needing no store
python3 -m coverage run --source=engine,checks,tools -m pytest tests \
  && python3 -m coverage report --sort=cover -m
```

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
