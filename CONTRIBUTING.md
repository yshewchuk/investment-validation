# Contributing

Before changing a component, read the root [`ARCHITECTURE.md`](ARCHITECTURE.md)
and, if it exists, that component's own `ARCHITECTURE.md` (the root doc's
"Component docs" index lists every one, and lists the rest as `(pending)`,
or, for a legacy `engine/**` component, `(legacy — removed at cutover; no
component doc)`). A cross-cutting change updates the root doc, in the same
pull request, before the code that implements it. A change to a
component's public interface, its dependencies, its inputs/outputs, or its
failure semantics updates that component's `ARCHITECTURE.md`, in the same
pull request, before the code that implements it. A legacy component
(`engine/**` outside `engine/v2/**`) gets no new component doc; a change
to it puts its design in the pull request body and, if durable, in the
operator guides instead.

A change under ~50 lines with no new interface or behaviour (a typo, a CI
flag, a one-line fix) is exempt from the doc-update rules above and says
so in its PR body instead (`ARCHITECTURE.md` §7: "Docs: n/a (trivial)"
plus the reason).

## Scope and deferred work

Keep each pull request inside the scope its description states. In scope:
the lines it adds or changes; anything that change makes wrong elsewhere
(a caller it breaks, a doc it makes false, a test it invalidates); code
the PR adds that cannot work against the current `main` as it stands
(bring in what's missing rather than deferring it); and any part of a
pre-existing defect that this PR's change makes worse or newly reachable.
Fix all of these in the PR — deferring one of them is not allowed.

A review comment that is real but falls outside that scope — a
pre-existing defect in code the PR doesn't touch and doesn't make worse or
newly reachable, a feature request, or a neighbouring refactor — never
blocks merge and is not fixed in the PR. File it as a GitHub issue instead
with `gh issue create`: the issue body starts with the line `Deferred from
#<PR> (<reviewer> review of <sha>, <file>:<line>).`, then states the
problem, what's wanted, and a test that would prove it — public-safe, no
strategy thresholds, edge figures, or local paths. Then reply on the
review thread: `Out of scope for this PR (<reason>); tracked in
#<issue>.`

## Small pull requests

One concern per pull request: something a reviewer can hold in their head
in one pass. If the description needs "and also", split it.

Aim for roughly 300 changed non-test lines and about 5 non-test files;
tests and generated fixtures don't count toward that target. A pull
request that goes over either target should say why it can't be split.
Every fix made in response to review feedback ships with a test that
fails without it, in the same push.
