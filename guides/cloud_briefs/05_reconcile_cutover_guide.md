# Reconcile guides/cutover_remaining_work.md

## Common rules (prepend to each task)
You are a PR owner on yshewchuk/investment-validation. You own ONE small PR to "ready for merge gate".
- Small PR: ≤~200 non-test code lines, ≤~60 doc lines. Strict scope: out-of-scope findings become new GitHub issues.
- Read root ARCHITECTURE.md and the component's ARCHITECTURE.md first; if you change interfaces/inputs/outputs/failure semantics, update that doc first in the same PR (contract-level, no history).
- Public repo: no local paths, credentials, strategy thresholds or clock values anywhere.
- CI's required `test` check is the test evidence; run only small targeted tests yourself.
- CodeRabbit reviews automatically: reply inside each thread (fix or explain), ONE push per round, then comment "@coderabbitai full review". Never "@coderabbitai approve", never tick "generate unit tests", never dismiss reviews. Max ~5 rounds.
- Never merge, never --admin, never push to main; push only your own branch. Don't run a merge gate — the supervisor gates it.
- Done when CI `test` is green on your head and CodeRabbit approved (or only answered comments). Final message ≤12 lines: PR URL, head SHA, change, CI + CodeRabbit state, issues filed.

## 5. Reconcile guides/cutover_remaining_work.md
It is stale since PR #235 (2026-09-30). Check merges since (`gh pr list --state merged --search "merged:>=2026-09-30"`; e.g. #239/#248 raw-row slices, #249, #255 side-by-side JSON, #257, #258, #259, #261, issue #260). Mark done items with PR numbers, delete statements no longer true, keep a short accurate remaining checklist (React side-by-side screen not built; React /api/v1 vs comparison-JSON transport needs design; open raw-row slices; production release root + scheduler verification). Verify every claim; current state, not history. Prefer deleting stale text over adding.
