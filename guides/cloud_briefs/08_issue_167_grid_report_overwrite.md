# Issue #167 — experiment grid overwrites the primary REPORT.md

## Common rules (prepend to each task)
You are a PR owner on yshewchuk/investment-validation. You own ONE small PR to "ready for merge gate".
- Small PR: ≤~200 non-test code lines, ≤~60 doc lines. Strict scope: out-of-scope findings become new GitHub issues.
- Read root ARCHITECTURE.md and the component's ARCHITECTURE.md first; if you change interfaces/inputs/outputs/failure semantics, update that doc first in the same PR (contract-level, no history).
- Public repo: no local paths, credentials, strategy thresholds or clock values anywhere.
- CI's required `test` check is the test evidence; run only small targeted tests yourself.
- CodeRabbit reviews automatically: reply inside each thread (fix or explain), ONE push per round, then comment "@coderabbitai full review". Never "@coderabbitai approve", never tick "generate unit tests", never dismiss reviews. Max ~5 rounds.
- Never merge, never --admin, never push to main; push only your own branch. Don't run a merge gate — the supervisor gates it.
- Done when CI `test` is green on your head and CodeRabbit approved (or only answered comments). Final message ≤12 lines: PR URL, head SHA, change, CI + CodeRabbit state, issues filed.

Read the issue first, confirm it still reproduces on current main, fix it in ONE small PR that closes the issue, with a test that fails before and passes after.

## 08. Issue #167 — experiment grid overwrites the primary REPORT.md
`experiments/new_experiment.py` passes the same run_dir to every grid cell, so the last secondary arm overwrites REPORT.md and shared figures. Give each secondary arm its own report path/figure dir while the primary keeps REPORT.md. Test that primary and secondary reports both survive.
