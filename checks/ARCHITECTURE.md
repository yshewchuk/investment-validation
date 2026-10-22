# Checks

## Purpose

`checks/` provides standalone verification tools for repository invariants, budgets, and phase evidence. The v2 release gate also invokes its bundle scanner in an isolated subprocess.

`check_bundle` applies `DECLARED_MAX_BYTES` to declared rendered-data files;
every other path, including undeclared paths under `data/`, uses ordinary
`MAX_BYTES`.

## Primary contracts and public interfaces

No package-wide runner/schema exists; invoke each CLI directly, e.g. `python3 checks/repo_hygiene.py`.
Some checks also expose callable report functions.
`phase5_modules.modules_available` probes optional modules (kept here: production code may not import dynamically).
`phase5_release_assembly` (`derive_catalog`, `carry_forward_release`) assembles a carried-forward
release catalog (plus `prior_carry_set`, which lists a prior release's objects as carried, and `classify_prior`, the read-only prior classification of §2.2.2);
its contract and refusal codes are `engine/v2/models/ARCHITECTURE.md` §2.1. The ops job
`phase5_state_stage` reaches it through `tools/phase5_state_stage_job.py` (§2.2).

The `write_object` contract (`engine.v2.models.release_objects`) is in `engine/v2/models/ARCHITECTURE.md` §2.1.

`tier0_support.py` owns `DEFAULT_CORPUS`, `round_params` and `finding_dicts`.
The default retains the corpus runner's checkout-relative `fixtures/tier0`
location. Rounding deep-copies the record and rounds only float values directly
inside a dictionary-valued `structure_params`; finding projection preserves
receipt order and returns fresh dictionaries with the same three fields.
`tier0_corpus.py` re-exports these names for existing callers. Its loading,
progress and subprocess orchestration remain there; Phase-2 evidence and the
synthetic Phase-0 controls import the support module directly.

The test selector maps changed paths through `checks/layer_map.py`, then
selects tests for those packages and their allowed importers. Unsafe or
unmapped input selects the full suite.
The test-layout ratchet blocks new tests outside `tests/v2/<package>/` and
`tests/v2/integration/`; the root-level unmoved test count stays the same or
decreases, and a root-level test cannot be modified in place (move it first).
`phase1_suite()` in `v2_coverage_ratchet.py` resolves baseline tests where they
now live and raises `FileNotFoundError` if one is missing.

Markdown is linted by `markdownlint-cli2` (rules in `.markdownlint-cli2.jsonc`),
run by the pre-commit hook on the working-tree copy of each staged `.md` file and by
the Tests workflow on all `.md`. In the hook, checks 1-4 and 6 are stdlib-only, check 5
runs Ruff, and check 7 needs Node. Default-mode inputs: checks 1, 2, 3 and 5 read staged
blobs (1 also reads the working-tree `.env`); check 4 layers staged blobs over the
working-tree copy of every tracked unstaged file; check 6 reads staged blobs against the
base ref; check 7 reads the working-tree copy of each staged `.md`.

The Phase 1 check binds the active test inventory to source and test
contents. It reports suite-version, inventory, or source drift when the
measurement differs from the active profile.

`import_layers.py --all` also checks the planned ops ownership/direction in
`ops_dependencies.json` against every tracked ops Python file. Static imports
in every scope, including lazy/function imports, form the module graph.
Forbidden directions and exact cyclic edges must match enumerated exceptions;
exceptions may only shrink against the base branch. Resolved exceptions must
be removed. Unmapped modules, syntax/JSON errors and read failures fail closed.
The existing always-run layer test invokes this check in CI; external layers
and dynamic-import restrictions are unchanged.

## Inputs

Inputs vary: source files/Git state, policy or baselines, application data, or prepared artifacts. Options select paths, scope, years, or outputs.

## Outputs

Commands write findings/progress to standard streams and may write reports or evidence artifacts; callable checks may return structured results.

## Dependencies

Checks use Python and repository metadata; evidence checks may import application packages or read built stores. `repo_hygiene.py` avoids `engine` imports.
`engine/v2/models/release_objects.py` uses foundation's public `ensure_directory`/`fsync_directory`
and `frozen_state_io.read_bounded_state`; `release_layout.py` uses models deployment for the existing store/catalog contracts.
Tier-0 support imports only the standard library and the diagnosis receipt
interface; it never imports the corpus runner or starts a process.

## External systems and libraries

Checks may use the filesystem or Git CLI; libraries are check-specific. `repo_hygiene.py` uses only the standard library.

## Failure semantics

No package-wide exit-code or artifact-write guarantee exists; the selected
check defines missing-input and failure outcomes:

| Condition | Outcome |
|---|---|
| Required input missing | Check-specific refusal, skip, or documented default |
| `.env` missing/empty (`check_files`) | CLI warns unless `--quiet` is set; current-value matching inactive; credential-pattern checks remain active; other checks continue |
| No secret needles (`check_bundle`) | Record `no-secrets-loaded`; still check bundle files |
| `repo_hygiene.py` scan | Clean: exit 0; violations: exit 1 |
| Staged hygiene read failure | Default returns empty bytes; `strict=True` raises `CalledProcessError` |
| Worktree hygiene read failure | Default returns empty bytes; `strict=True` raises `OSError`; missing non-symlink paths return empty bytes in either mode |
| The pre-commit hook finds no `npx` | One-line install hint on stderr; commit blocked (fails closed) |
| Cache | No shared cache contract; reuse is check-specific |
| Retry | No automatic package retry; check-specific if any |
| Transaction | No package-wide transaction contract |
| Partial write | No shared atomic-write or rollback guarantee |
| Repeat invocation | No package-wide idempotency guarantee; effects vary |
| Selector input or declaration is unsafe | Report the reason and select the full suite |
| Root test count grows, the budget increases from base, the budget differs from the current root-test count, or a root test is modified in place | Ratchet check fails |
| Budget decreases without root-test moves | Pass only when the new budget equals the current root-test count; increases remain refused |
| A phase-one baseline test cannot be found | `phase1_suite()` raises `FileNotFoundError` |
| Tier-0 support gets an invalid record/receipt | Existing copy, rounding or attribute errors propagate; no retry, cache, transaction or partial writes |

## Invariants

`checks/import_layers.py` bars v2 imports of `checks`/`tests` and legacy imports
of `engine/v2`; the bundle scanner runs in `legacy_adapter.py`'s isolated subprocess.
