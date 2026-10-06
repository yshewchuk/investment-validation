# Checks

`checks/` contains standalone verification and development tools. They protect
repository and architecture invariants, enforce budgets, and evaluate phase
and rearchitecture evidence. They are consumers of the application and its
artifacts; production packages must not depend on this verification code.

## Interface and dependencies

There is no package-wide runner or shared argument schema. Invoke the selected
script directly with Python, for example `python3 checks/repo_hygiene.py` or
`python3 checks/phase0_audit.py --report <path>`. Each script documents its
options. CI, hooks, and owner tooling select the checks relevant to a change.

Inputs depend on the selected check: source files and repository state (often
the Git index or tracked paths), check-specific policy and baseline files, or
application data and prepared artifacts. Options may select paths, scope,
years, or report destinations. Static checks can expose callable functions
that return structured reports; command-line entrypoints adapt those results
to process output and status.

Dependencies are similarly specific. Static checks use Python and repository
metadata; some use Git to read staged or tracked content. Evidence checks may
import application packages and read built stores or prepared artifacts.
`repo_hygiene.py` is intentionally standard-library-only and does not import
`engine`, so it can run in a fresh clone as a pre-commit check.

## Results and failure semantics

Each command writes findings or progress to standard output/error and may
write a check-specific report or evidence artifact. There is no universal
exit-code contract: callers must use the selected script's documented status
and output. A failed invariant is reported as a nonzero command status or a
check-specific failed result; report generators may instead finish
successfully when they produce an audit report.

Missing and unreadable inputs follow each check's contract. For example,
`repo_hygiene.py` returns 0 for a clean scan and 1 for detected violations.
If its `.env` is missing or yields no secret patterns, it warns and continues
with value scanning inactive; path, size, and credential-shape checks still
run. Its staged and worktree blob readers are permissive by default and
represent read failures as empty bytes; callers that request strict reads get
an exception. Other checks may refuse missing required inputs or report them
as skipped/unavailable. Do not treat one check's missing-input or exit-status
behavior as a package-wide guarantee.
