"""Real GitHub `workflow_run.pull_requests` entries (the pull-request-minimal
schema), fetched read-only via `gh api` on 2026-09-27. Field names and
values below are byte-for-byte from those live API responses -- nothing
invented, nothing renamed. In particular: `base.repo`/`head.repo` carry
ONLY `id`, `name` and `url` -- there is no `full_name` field, which is the
defect this fixture exists to catch (a filter on `.base.repo.full_name`
never matches anything real, so mutation CI silently never ran for any PR).

SAME_REPO_PR is
  gh api "repos/yshewchuk/investment-validation/actions/runs?event=pull_request&per_page=1" \
    --jq '.workflow_runs[0].pull_requests[0]'
-- this repo's own real PR #45 (this branch, ci/prioritize-tests), from
workflow_run id 36304279450 (the `Tests` run for commit d2d9bae).

FORK_PR is
  gh api repos/home-assistant/core/actions/runs/36303992058 --jq '.pull_requests[0]'
-- a real PR from a DIFFERENT public repository (home-assistant/core, a
large repo with many stale cross-fork PRs sharing common branch names),
included only to get a second genuine pull-request-minimal object whose
base.repo.id does not equal investment-validation's. It is unrelated to
investment-validation; that is exactly the point -- it must never match.
"""

SAME_REPO_PR = {
    "id": 4651898179,
    "number": 45,
    "url": "https://api.github.com/repos/yshewchuk/investment-validation/pulls/45",
    "base": {
        "ref": "main",
        "sha": "1a59ce86642af65a130af239442ebbf7d472488f",
        "repo": {
            "id": 1351028987,
            "name": "investment-validation",
            "url": "https://api.github.com/repos/yshewchuk/investment-validation",
        },
    },
    "head": {
        "ref": "ci/prioritize-tests",
        "sha": "d2d9bae845344d688ce274d19aa6b6d3f129d15e",
        "repo": {
            "id": 1351028987,
            "name": "investment-validation",
            "url": "https://api.github.com/repos/yshewchuk/investment-validation",
        },
    },
}

FORK_PR = {
    "id": 3960095091,
    "number": 12,
    "url": "https://api.github.com/repos/terafin/core/pulls/12",
    "base": {
        "ref": "dev",
        "sha": "e815c9f0ccdfacdab748abb0a1e79cd6831e8aee",
        "repo": {
            "id": 1281165613,
            "name": "core",
            "url": "https://api.github.com/repos/terafin/core",
        },
    },
    "head": {
        "ref": "dev",
        "sha": "a9196e0470add6a83de9a713499848d0c8e45a48",
        "repo": {
            "id": 12888993,
            "name": "core",
            "url": "https://api.github.com/repos/home-assistant/core",
        },
    },
}
