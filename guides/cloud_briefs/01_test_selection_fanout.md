# PR test-selection fan-out

## Common rules (prepend to each task)
You are a PR owner on yshewchuk/investment-validation. You own ONE small PR to "ready for merge gate".
- Small PR: ≤~200 non-test code lines, ≤~60 doc lines. Strict scope: out-of-scope findings become new GitHub issues.
- Read root ARCHITECTURE.md and the component's ARCHITECTURE.md first; if you change interfaces/inputs/outputs/failure semantics, update that doc first in the same PR (contract-level, no history).
- Public repo: no local paths, credentials, strategy thresholds or clock values anywhere.
- CI's required `test` check is the test evidence; run only small targeted tests yourself.
- CodeRabbit reviews automatically: reply inside each thread (fix or explain), ONE push per round, then comment "@coderabbitai full review". Never "@coderabbitai approve", never tick "generate unit tests", never dismiss reviews. Max ~5 rounds.
- Never merge, never --admin, never push to main; push only your own branch. Don't run a merge gate — the supervisor gates it.
- Done when CI `test` is green on your head and CodeRabbit approved (or only answered comments). Final message ≤12 lines: PR URL, head SHA, change, CI + CodeRabbit state, issues filed.

## 1. PR test-selection fan-out
`tools/mutation_pilot.py select-tests` (CI's per-PR test picker) selects ~324 of ~361 test files whenever a PR touches any ARCHITECTURE.md or a common module, and a new test file returns `__ALL__`; 29 of the last 30 PRs touched an ARCHITECTURE.md, so nearly every PR runs nearly the full suite. File an issue with evidence (run select-tests on a few recent merged PRs' changed files), then fix: doc-only changes (*.md) select no tests or only doc-check tests; a new test file selects itself (plus what it imports that changed), not `__ALL__`; keep the full-suite fallback for genuinely global changes (CI config, conftest, shared fixtures, requirements). Add tests for the rules; confirm via .github/workflows that the required `test` check still runs and passes.
