# Issue #194 — verify_object_path re-hashes every object on every open

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

## 12. Issue #194 — verify_object_path re-hashes every object on every open
`engine/v2/data/objects.verify_object_path` re-hashes the full object on each open. Add a stat-tuple (device, inode, size, mtime_ns) short-circuit cache so an unchanged file is not re-hashed, while any change forces a full verify. Keep the integrity guarantee explicit in the data ARCHITECTURE.md failure semantics. Do NOT edit engine/v2/data/repository.py or query.py (another PR is changing them). Test: second open skips hashing; modified bytes or mtime trigger a full verify and still detect corruption.
