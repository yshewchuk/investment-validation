# Checks

## Purpose

`checks/` provides standalone verification tools for repository invariants, budgets, and phase evidence. The v2 release gate also invokes its bundle scanner in an isolated subprocess.

## Primary contracts and public interfaces

There is no package-wide runner or argument schema. Invoke a selected script,
such as `python3 checks/repo_hygiene.py` or `python3 checks/phase0_audit.py --report <path>`;
its CLI defines options. Some checks expose callable functions returning reports.

## Inputs

Inputs vary: source files/Git state, policy or baselines, application data, or prepared artifacts. Options select paths, scope, years, or outputs.

## Outputs

Commands write findings/progress to standard streams and may write reports or evidence artifacts; callable checks may return structured results.

## Dependencies

Checks use Python and repository metadata; evidence checks may import application packages or read built stores. `repo_hygiene.py` avoids `engine` imports.

## External systems and libraries

Checks may use the filesystem or Git CLI; libraries are check-specific. `repo_hygiene.py` uses only the standard library.

## Failure semantics

No package-wide exit-code or artifact-write guarantee exists; the selected
check defines missing-input and failure outcomes:

| Condition | Outcome |
|---|---|
| Required input missing | Check-specific refusal, skip, or documented default |
| `.env` missing/empty (`repo_hygiene.py`) | Warn; secret scan inactive; other checks continue |
| `repo_hygiene.py` scan | Clean: exit 0; violations: exit 1 |
| Staged hygiene read failure | Default returns empty bytes; `strict=True` raises `CalledProcessError` |
| Worktree hygiene read failure | Default returns empty bytes; `strict=True` raises `OSError`; missing non-symlink paths return empty bytes in either mode |
| Cache | No shared cache contract; reuse is check-specific |
| Retry | No automatic package retry; check-specific if any |
| Transaction | No package-wide transaction contract |
| Partial write | No shared atomic-write or rollback guarantee |
| Repeat invocation | No package-wide idempotency guarantee; effects vary |

## Invariants

`checks/import_layers.py` bars direct v2 imports of `checks`; the bundle-scanner
exception runs in `legacy_adapter.py`'s isolated subprocess; legacy `engine/**` imports are outside its scope.
