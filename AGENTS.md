## Workspace artifacts

- Except for ops-store attempt staging assigned to the supervisor below, put disposable scratch, spills, transient staging copies, and intermediate outputs that no process reads after the producer exits under `/tmp` or a dedicated directory named `scratch`, `tmp`, or `disposable`, outside durable-data trees.
- Keep durable outputs, including catalogs, content-addressed objects, evidence, and the ledger, in their established locations.
- Tools that create scratch remove it on success; on failure, they remove all scratch contents except diagnostics and leave those in the disposable scratch location.
- Agents remove their worktrees when merged or abandoned; the supervisor also removes merged worktrees with no uncommitted changes.
- The supervisor manages and reclaims ops-store attempt staging as specified in [#609](https://github.com/yshewchuk/investment-validation/issues/609); this overrides the placement and cleanup rules above for that staging.
- Agents never hand-delete under an ops store and report cleanup needs to the supervisor.

## Pull requests, gates and owners

- The PR watcher runs the merge gate at `/root/agent-ops/gate/gate.sh` after CI `test` is green, CodeRabbit approves, and no thread is open on the head. It starts detached, once per head and at most five times per PR. Owners do not run the gate.
- Owners wake only for a BLOCK. A pass wakes nobody and auto-merge completes the PR. If a passing PR is still unmerged after 10 minutes, the owner is woken once.
- On a BLOCK, fix real findings and push once per round. Continue independently for up to five gate rounds, then stop and report. Deferring a finding to an issue requires the gate's agreement, as with a CodeRabbit comment. If the gate blocks on it again, fix it.
- Every PR description starts with `## Gate notes` (about 1,000 characters maximum; the gate reads the first 4,000 description characters). Keep concise round notes and any deferral's issue link and reason there. Update the notes before each fix push. The gate takes no notes file.
- Size limits count added lines only: code PRs, about 200 non-test code lines and 60 doc lines; design PRs, about 150 doc lines and a split list. Deleted lines do not count; a category with zero or negative net change is never size-blocked. A PR within limits at its first reviewed head is not size-blocked when the overage is review- or gate-found bug fixes. For scope growth over a limit mid-work, stop and split (add, then wire, then remove).
- A comment from the shared GitHub login that starts with `@pr-owner` force-wakes that PR's owner, bypassing the wake gap and hourly cap. This is the user's lever; owners never start their own comments with it.
- Owner tooling (watcher, gate, owner scripts and metrics) lives in `/root/agent-ops/`. See `/root/agent-ops/README.md` and `/root/agent-ops/pr_watch/README.md` for its workflow.
- Before the first push of a PR that edits an `ARCHITECTURE.md`, run `python3 tools/oc_check.py tests/test_architecture_doc_budgets.py`.
- To update a PR description, write a Markdown file and run `gh pr edit <n> --body-file <file>`. Do not use an inline script that writes files.
- New tests go in `tests/v2/<package>/` (package names from `checks/layer_map.py`) or `tests/v2/integration/`, never the root `tests/` folder. A PR that edits a root-level test should move it and lower `checks/test_layout_budget.txt` by exactly the number of files moved. Before pushing, run `python3 checks/test_layout_budget.py --base-ref origin/main`; PR CI enforces it.
