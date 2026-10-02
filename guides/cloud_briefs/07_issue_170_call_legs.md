# Issue #170 — planned-exit simulation values call legs as puts

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

## 07. Issue #170 — planned-exit simulation values call legs as puts
`engine/v2/scoring/stages.py` (planned-exit simulation) prices every leg with the put formula without checking `leg.right`. Price calls as calls (or refuse non-put legs explicitly if calls are unsupported — choose per the scoring ARCHITECTURE.md contract and state why). Test with the issue's synthetic reproduction.
