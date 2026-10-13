# Checks

## Purpose

`checks/` provides standalone verification tools for repository invariants, budgets, and phase evidence. The v2 release gate also invokes its bundle scanner in an isolated subprocess.

`check_bundle` applies `DECLARED_MAX_BYTES` to declared rendered-data files;
every other path, including undeclared paths under `data/`, uses ordinary
`MAX_BYTES`.

## Primary contracts and public interfaces

No package-wide runner/schema exists; invoke each CLI directly, e.g. `python3 checks/repo_hygiene.py`.
Some checks also expose callable report functions.

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

`markdown_structure.py` is a thin wrapper over the pinned `markdownlint-cli2`
(`MDLINT`, run as `npx --yes`), configured by `.markdownlint-cli2.jsonc` at the
repo root: only MD022 (headings), MD031 (fences) and MD058 (tables), each a
blank-line rule. Inputs: default mode reads staged blobs; `--base-ref REF`
reads `HEAD` content for `REF...HEAD` (run in CI by the Tests workflow);
`--all` lints every tracked `.md` in full. The blobs are written to a temp
directory and linted there, so the worktree is never read. Output: one
`path:line: MDnnn message` row per violation on a line the change added or
modified (`git diff -U0` hunks), so existing debt never blocks. Unclosed fences
are not covered, since no markdownlint rule detects them. The pre-commit hook
runs the default mode.

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
| `markdown_structure.py` finds a violation on a changed line | Prints `path:line: MDnnn message`; exit 1 |
| `markdown_structure.py` git or file read fails, a file is not UTF-8, or `--base-ref` is empty | Message on stderr; exit 2; never read as clean, never staged mode |
| `markdown_structure.py` finds no `npx`, or the linter crashes, times out (300 s) or fails without findings | One-line message (install Node.js hint) on stderr; exit 2; fails closed, never passes |
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
