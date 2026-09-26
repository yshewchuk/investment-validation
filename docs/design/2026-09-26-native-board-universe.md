# Native board universe (design)

## Context

The nightly board is enumerated once, by legacy `engine.score.score_calendar`:
for every confirmed earnings event inside a horizon window, for every
registered strategy the live board still scores, it builds one scoring
request. That enumeration reads the legacy option-chain index and constructs
a legacy `Scorer` up front, purely to know which chain keys the board will
need later — enumeration and legacy pricing are not currently separated.

Native scoring needs the *same* board — the same events, the same strategies
minus the ones native does not support — without pulling in the legacy chain
index or constructing a legacy `Scorer` at all. Doing that is the whole job
of Part 1 of the native-score-stage spec (embedded in full in this PR's
description): an answer-free function that reproduces `score_calendar`'s
event × strategy enumeration, and nothing else. This document designs only
that function. Later parts of the same spec (the nightly stage that consumes
this enumeration, the per-row scoring worker, and the shadow-projection
wiring) are each their own design doc and PR; they are listed under "Out of
scope" below and are not decided here.

This session produces the design doc only — no code. Implementation follows
in a later pass against whatever this doc settles, per the repo's rule that a
design doc must be approved before the corresponding code is written.

## Decisions

**1. A new, minimal `BoardRequest` key — not either existing `ScoreRequest`.**

Two classes named `ScoreRequest` already exist, and neither is right for an
enumeration step:

- Legacy `engine.score.ScoreRequest` carries `strike`, `expiry`, `fill`,
  `quote_max_age_sessions`, and other legacy-pricing fields alongside its
  identity fields. Building this class here would require importing the
  legacy pricing module the enumerator is explicitly not allowed to touch.
- Native `engine.v2.contracts.scoring.ScoreRequest` (the one `score_one`/
  `score_event` take) is a fully-resolved scoring identity: `event_id`,
  `calendar_revision`, `deployment_id`, `decision_clock_id`,
  `requested_decision_at`, `snapshot_id`, `model_artifact_refs`, and more.
  An enumerator over a raw events table has none of those — they come from
  resolving a specific snapshot, deployment and model binding, which is a
  scoring-time concern a later stage owns. Inventing placeholder values for
  them here would be exactly the silent-default pattern the native
  invariants forbid.

Chosen: a new frozen dataclass `BoardRequest(ticker, strategy, event_date,
session)` — a pure key, with no fill model, no strike, no snapshot binding.
Translating a `BoardRequest` into a full native `ScoreRequest` is later
work (see "Out of scope"), once a snapshot and deployment binding exist to
resolve it against.

**2. Strategy universe: the native-covered set, read from its one owner —
not the full `STRUCTURES` registry.**

- *Mirror legacy exactly (`STRUCTURES` minus `DISABLED_STRATEGIES`)* —
  rejected. That would enumerate strategies native scoring has no input
  builder for at all; every row of every such strategy would refuse forever,
  for a reason that belongs to the input builder, not to enumeration.
- *Hard-code the covered set (`STR-THRU`, `STR-RUNUP`, the seven dynamic-menu
  members) inside the new module* — rejected on its own. It would duplicate a
  decision the input-builder module already makes, and the two could drift.
- *Read the covered set from the input-builder module's own declaration* —
  chosen. `engine.v2.scoring.source_inputs` already keeps a
  strategy → forecast-outputs mapping that is the real definition of "native
  knows how to build inputs for this strategy," but it is a private
  (underscore-prefixed) name today, not part of that module's public
  interface. This design adds one small, additive export to that module —
  a public frozen-set alias of the existing mapping's keys — so the
  enumerator imports the real declaration instead of a second copy of it.
  This is the only interface change Part 1 makes to an existing module (see
  "Changed interfaces").

`DYN-SV` is a meta-row, not a tenth covered strategy: one `BoardRequest` per
event, never one per strategy, because at scoring time it fans out
internally to the seven dynamic-menu members and chooses among them — the
same shape legacy uses when it appends a chooser row once per event after
scoring the frame, rather than enumerating it as a per-alternative board row.

**3. A malformed events table is a whole-call refusal, not a per-row
status.**

- *Skip malformed rows silently* — rejected: would turn a schema defect into
  a silently smaller (or empty) board, which the native invariants forbid —
  a missing/unusable input must produce a typed status, never a silent
  empty result.
- *Per-row typed status* — rejected: a missing `ticker`/`event_date`/
  `session` column is a structural precondition of the whole table, not a
  fact about any individual row's content.
- *Whole-call typed refusal* — chosen: raise before reading any row, using
  the same typed-refusal envelope the rest of the ops layer already uses at
  whole-call boundaries for a malformed batch input.

**4. Filter exactly like `score_calendar`, no independent tightening or
loosening.**

The date window (`as_of <= event_date <= as_of + horizon_days`), the
`session` not-null filter, and the optional ticker filter are copied
byte-for-byte in spirit from `score_calendar`. A native board that covered a
different set of events than the legacy board would make every later
native-vs-legacy comparison meaningless — it would be comparing two
different boards, not the same board scored two ways.

## Layers and modules touched

New module: `engine/v2/ops/native_board_universe.py` — pure and
side-effect-free. It takes an already-loaded events table as a parameter; it
performs no I/O, no network call, no store write, and constructs no legacy
`Scorer` or chain index.

It reads (does not modify) four existing declarations, each from its actual
owner:
- `engine.structures.STRUCTURES` — the registered strategy keys (legacy
  owns this registry; native reads the key set only for the
  "not the full registry" comparison in decision 2).
- `engine.score.DISABLED_STRATEGIES` — the two strategies the board never
  scores by policy, for the same comparison.
- `engine.v2.registry.strategies.DYNAMIC_MENU` — the seven dynamic-menu
  strategy names.
- `engine.v2.scoring.source_inputs`'s new public export (decision 2) — the
  strategies native's own input builder declares support for.

Dependency direction: `engine/v2/ops` sits above `engine/v2/scoring` and
`engine/v2/registry` in the v2 layering already used elsewhere in this tree
(a v2 module may cite a legacy constant by name for a read-only comparison
without depending on legacy execution — the same pattern the existing
native shadow-serving module uses for its own legacy references). This
module never imports legacy `Scorer`, legacy pricing, or the legacy chain
index. `docs/ARCHITECTURE.md` does not exist on `main` yet, so this section
cites the real modules and their own docstrings directly; once that file
lands, its layer/import-direction sections should be checked against this
one.

## Production call path

Today: **none.** `board_requests()` has no caller anywhere in this PR or in
production. It becomes reachable only once a later stage adds a
`native_score` entry to the nightly stage graph and a job handler calls
`board_requests(...)` as the first step of that stage — both later parts of
the same spec, not built here. Until then this is a library function
exercised only by its own tests.

Worth being explicit about one existing constraint this doc does not change:
the nightly graph this future stage would join only ever runs in its
"shadow" mode — the plan-builder function that assembles the graph
explicitly refuses any other mode. That is a pre-existing property of the
whole nightly graph, not something Part 1 introduces or relies on beyond
noting it so the "production call path" claim above isn't overstated.

## Changed interfaces

One additive, backward-compatible change: `engine.v2.scoring.source_inputs`
gains one new public name — a frozen-set alias of the strategy keys its
existing (private) forecast-output mapping already declares. Its private
name is used internally in exactly two places in that module today (a
missing-forecast-output check, and a strategy-support guard); neither
changes behavior. No other module references the private name, so adding a
public alias cannot collide with an existing caller.

Everything else is new: the `BoardRequest` dataclass and the
`board_requests(as_of, horizon_days, tickers, events_table)` function have
no existing callers to enumerate, which is itself the fact this section
exists to state, not an omission. The PR that gives this function its first
caller (the later `native_score` stage) must re-run this grep and list that
caller.

## Failure semantics (4c R1–R6)

- **Missing input:** a required column absent from `events_table` (or the
  table otherwise not shaped as expected) is a whole-call typed refusal,
  raised before any row is read — never a partial or empty result.
- **Cache:** none. The function holds no cache of its own; it reads only the
  table its caller passes in.
- **Retry:** pure and deterministic for a given table snapshot; safe to call
  again with nothing to undo. "Retry" here just means re-execution, since
  there is no write to have partially happened.
- **Transaction:** none applies — this function makes no write. Whatever
  later commits the requests this function enumerates (a future part of the
  spec) owns its own commit-once boundary; this doc does not extend one to
  here.
- **Partial write:** not possible — the function either returns a complete
  tuple or raises; there is no partially-built output state to reason about.
- **Idempotency:** the same `(as_of, horizon_days, tickers, events_table)`
  always returns the same tuple in the same order. Idempotency of anything
  downstream (a scoring job's own commit key, built from a set of
  `BoardRequest`s) is explicitly a later part's concern, not this one's.

## Invariants touched

- **Native vs. legacy values:** this function reads two legacy-owned
  constants for a read-only set comparison; it never returns a legacy value
  — every field of every `BoardRequest` it emits (ticker, strategy,
  event date, session) comes from the shared events table, which neither
  side owns.
- **Typed status:** a malformed table refuses with a named code rather than
  degrading to a silent empty or partial result.
- **Isolation:** this function never loads the legacy option-chain index and
  never constructs a legacy `Scorer` — it needs no priced quotes at all,
  only the board's event dates and sessions.

`docs/ARCHITECTURE.md` does not exist on `main` yet; once it lands, this
section should cite its specific invariant bullets by number instead of
restating them here.

## Test plan

| Acceptance criterion | Test | How it could fail |
| --- | --- | --- |
| Enumerated `(ticker, strategy)` pairs match legacy's ATM-pass requests one-for-one for the native-covered strategies | New fixture-table test comparing `board_requests()`'s output against `score_calendar`'s own default enumeration on the same small fixture, read-only | A filter drifts out of sync with `score_calendar` (e.g. an off-by-one at the horizon boundary), or the native-covered strategy set silently gains or loses a member |
| Exactly one `DYN-SV` request per event, never per strategy | Fixture with one event asserts the count of `DYN-SV` requests is 1 regardless of how many dynamic-menu members exist | `DYN-SV` gets added inside the same per-strategy loop as the seven menu members instead of once per event |
| A malformed events table raises a named refusal, never a silent empty result | Fixture missing one of `ticker`/`event_date`/`session` asserts both the specific refusal code AND that no empty tuple is returned instead | A future refactor adds a broad `except` that swallows the schema error and returns an empty result |
| Determinism | Calling `board_requests()` twice on the same fixture returns equal tuples in the same order | Any dependency on unordered iteration (e.g. a set) leaking into the returned order |

## Out of scope

- The `native_score` job/stage itself, its stage-graph wiring, and its
  resource profile — a later part of the same spec, with its own design doc.
- Per-row scoring, `DYN-SV` fan-out execution, and the committed scored-row
  artifact this enumeration feeds — a later part.
- Automatic shadow-projection wiring that would read that artifact — a later
  part still.
- Translating a `BoardRequest` into a full native scoring identity (it needs
  a resolved snapshot, deployment and model binding this function does not
  have and should not invent) — a later part's job.
- Any change to legacy `score_calendar` itself; this function only reads its
  behavior as a reference, never modifies it.
