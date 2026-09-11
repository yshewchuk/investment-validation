# Guide — version the moves checkpoint on build logic, not just inputs

**Version:** 1.0 · **Date:** 2026-09-11 · **Owner:** YS + Claude
**Status:** ready to hand off. Small and self-contained: one constant, one
function, two or three tests. No data pull, no quota, no model.
**Touches:** `engine/data/pulls/computed_moves.py`, `tests/test_computed_moves.py`

---

## 1. The problem, observed live

`computed_moves` resumes an interrupted build from `.checkpoint.jsonl`, keyed
on a fingerprint of the build's **inputs**:

```python
def _build_fingerprint(all_scoreable: bool, events: pd.DataFrame) -> str:
    payload = json.dumps({
        "mode": "all_scoreable" if all_scoreable else "extension_only",
        "n_events": int(len(events)),
        "max_event_date": str(pd.to_datetime(events["event_date"]).max().date()),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
```

Nothing in there describes **how the build computes a move**. So a checkpoint
survives a change to the computation itself, and every ticker already recorded
is skipped by the new code.

Caught during the 2026-09-11 nightly, in the log:

```
resuming build ada32ba59317c5dc: 3 of 2,853 already finished
```

Those three (CAL, IBEX, ZUMZ) were written at 08:55 by a smoke test, **before**
`fe97514` added `live=True` to `fetch_history`. They therefore hold moves
computed from the stale 2026-09-04 price series, and the post-fix run skipped
them rather than rebuilding them.

Harmless in that instance — all three last printed in May/June 2026, so a
09-04 price cutoff could not affect their moves. That is luck, not design.

## 2. Why it is worth fixing

This pull is the source of record for realized moves, and the panel and Tier 4
are derived from it. The failure mode is the one this whole area has already
produced twice: **a fix that appears to land and silently does not apply to
part of the data.** A correction to `build_ticker`'s move arithmetic, to the
gap rule, or to the session handling would be skipped for whatever a prior
interrupted build had already written, with no message anywhere.

The 09-05..09-11 outage had exactly this shape — two independent causes, one
symptom, and fixing the first looked like a fix. Do not leave a third.

## 3. The change

Add a build-logic version to the fingerprint.

```python
#: Bumped whenever the COMPUTATION changes — fetch_history, build_ticker, the
#: gap rule, the session handling. A checkpoint from an earlier version is not
#: resumed, because the rows it recorded were produced by different code.
#: Input identity alone cannot see this: the same events rebuilt by different
#: arithmetic are a different build.
BUILD_VERSION = 2
```

and include it in the payload:

```python
payload = json.dumps({
    "build_version": BUILD_VERSION,
    "mode": ...,
    "n_events": ...,
    "max_event_date": ...,
}, sort_keys=True)
```

Start at `2`, not `1`: version 1 is everything written before this lands, and
those checkpoints should not be resumed into by the current code.

## 4. The decision to make — how the version gets bumped

Pick one. My recommendation is (c).

**(a) Manual constant.** A developer bumps it when they change the
computation. Minimal, explicit. The failure mode is forgetting, which
reproduces exactly the silent skip this is meant to prevent.

**(b) Hash the source of the computing functions.** Automatic, no discipline
required. But a comment or docstring edit invalidates every checkpoint, and on
a full-universe build that means re-fetching 2,853 tickers for a typo fix.
Too brittle for the cost involved.

**(c) Manual constant, with a test that notices.** Keep `BUILD_VERSION` manual,
and add a test holding a recorded hash of `inspect.getsource` for the functions
that define the computation (`fetch_history`, `build_ticker`, plus the
`MAX_GAP_CALENDAR_DAYS` / `MIN_SCOREABLE` constants). When the source changes,
the test fails with a message asking the one question that matters:

> the move computation changed — does this invalidate checkpoints written by
> the previous version? If yes bump `BUILD_VERSION`; if no (comment, rename,
> pure refactor) update the recorded hash.

That converts a silent skip into a deliberate decision, which is the property
worth buying, and it costs a one-line fixture update for cosmetic edits.

## 5. Migration — cheaper than it looks, check anyway

Bumping the version invalidates existing checkpoints. That sounds expensive and
mostly is not, because the checkpoint only tracks progress *within one build*
while the `.state.json` watermark controls *scope*:

* **Steady state** — the nightly asks only for names that printed since the
  watermark (~20–60 tickers). An invalidated checkpoint costs a redo of those.
  Negligible.
* **The expensive case** — a full-universe build (2,853) interrupted partway,
  with the version bumped before the resume. Then the whole thing restarts.
  Rare, and it is the correct answer anyway: the completed half was built by
  code we just decided produces different output.

Before landing, confirm no full build is mid-flight. If one is, either wait for
it or accept the restart knowingly.

## 6. Deliberately NOT in the fingerprint

`since`. Two runs with different `--since` values share a fingerprint on
purpose: the completed tickers are still validly built, so a wider run should
inherit them and do only the remainder. Adding `since` would throw away good
work every time the watermark moved. Leave it out.

## 7. Acceptance

- [ ] `BUILD_VERSION` exists, is in the fingerprint payload, and starts at 2.
- [ ] A checkpoint written under an older version is **not** resumed —
      `_load_checkpoint` returns `{}` for it.
- [ ] Two builds that differ only in `BUILD_VERSION` produce different
      fingerprints; two that differ in nothing produce the same one.
- [ ] The existing 16 tests in `tests/test_computed_moves.py` stay green —
      especially `test_a_checkpoint_from_another_build_is_not_inherited`, which
      already pins the general rule this extends.
- [ ] If option (c): the guard test fails when `build_ticker`'s source changes
      and its message names the decision, not just the mismatch.

## 8. What not to do

- **Do not hash the whole module.** The CLI, the docstrings and the checkpoint
  helpers all live there and none of them change what a move *is*. Hash the
  computation, or version it by hand.
- **Do not delete old checkpoint files** as part of this. Leaving them costs
  nothing — they simply stop matching — and a deletion path is a foot-gun in a
  directory that also holds the moves themselves.
- **Do not fold this into the watermark.** They answer different questions:
  the watermark is *how far the moves are built*, the fingerprint is *by which
  code*. Conflating them makes a logic change look like a date change.
