# Contributing

Before changing a component, read the root [`ARCHITECTURE.md`](ARCHITECTURE.md)
and, if it exists, that component's own `ARCHITECTURE.md` (the root doc's
"Component docs" index lists every one, and lists the rest as `(pending)`).
A cross-cutting change updates the root doc, in the same pull request,
before the code that implements it. A change to a component's public
interface, its dependencies, its inputs/outputs, or its failure semantics
updates that component's `ARCHITECTURE.md`, in the same pull request,
before the code that implements it. Legacy code slated for removal has no
component doc of its own; its design goes in the pull request body
instead.

## Scope and deferred work

Keep each pull request inside the scope its description states: the lines
it adds or changes, plus anything that change makes wrong elsewhere (a
caller it breaks, a doc it makes false, a test it invalidates). Code the
PR adds that cannot work against the current `main` as it stands is in
scope too — bring in what's missing rather than deferring it.

A review comment that is real but falls outside that scope — a
pre-existing defect in code the PR doesn't touch, a feature request, or a
neighbouring refactor — is not fixed in the PR. File it as an issue
instead, referencing the PR and the review comment it came from, and
reply on the review thread noting that it is tracked separately.

## Small pull requests

One concern per pull request: something a reviewer can hold in their head
in one pass. If the description needs "and also", split it.

Aim for roughly 300 changed non-test lines and about 5 non-test files;
tests and generated fixtures don't count toward that target. A pull
request that goes over either target should say why it can't be split.
Every fix made in response to review feedback ships with a test that
fails without it, in the same push.
