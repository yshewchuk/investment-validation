"""Goal 1 of the strict-trace RSS fix: nothing from candidate selection
onward may keep the Scorer (the panel + replayed trades) reachable.
`attach_strict_probe`/`chooser_trace`/`strict_trace_one`/`write` never
accept a Scorer at all (verified by reading their signatures), so the only
way it could still be alive when `write()` runs is if `_gather_candidates`
(or `main`) accidentally kept a reference to it -- directly, or nested
inside `chosen`/`index`. Both tests use weakref liveness rather than a
hand-rolled poison object: that catches ANY path that keeps the scorer
alive, not just an attribute a test author happened to anticipate.

Fully synthetic: no real Scorer, no panel, no chain table.
"""
from __future__ import annotations

import gc
import weakref
from argparse import Namespace

import pandas as pd

import tools.capture_tier0_corpus as capture


class _FakeCalendar:
    pass


class _FakeScorer:
    """Enough of `engine.score.Scorer` for `_gather_candidates`/`main` to
    run: `.calendar` (read by `_boundary_events`) and `.snapshot` (the one
    thing `write()` needs, captured before the scorer is dropped)."""

    def __init__(self) -> None:
        self.calendar = _FakeCalendar()
        self.snapshot = "snapshot-1"


def _args(**overrides) -> Namespace:
    base = dict(
        out=None, version=None, replace=False, as_of=None,
        forward_days=35, max_events=40, boundary_events=4,
        quote_max_age=5, strategies=None, strict_phase4_trace=False,
    )
    base.update(overrides)
    return Namespace(**base)


def _install_empty_pipeline(monkeypatch, chosen: list[dict]) -> None:
    """Every collaborator `_gather_candidates` calls, faked to synthetic,
    scorer-free data. `chosen` is exactly what `select()` returns as the
    covering subset."""
    empty = pd.DataFrame({"ticker": [], "event_date": []})
    monkeypatch.setattr(capture, "_events", lambda *a, **k: empty)
    monkeypatch.setattr(capture, "_boundary_events", lambda *a, **k: empty)
    monkeypatch.setattr(capture, "_forward_chain_keys", lambda *a, **k: set())
    monkeypatch.setattr(capture, "_boundary_chain_keys", lambda *a, **k: set())
    monkeypatch.setattr(capture, "_research_replay_chain_keys", lambda *a, **k: set())
    monkeypatch.setattr(capture, "forward_pass", lambda *a, **k: [])
    monkeypatch.setattr(capture, "boundary_pass", lambda *a, **k: [])
    monkeypatch.setattr(capture, "pinned_and_strike_pass", lambda *a, **k: [])
    monkeypatch.setattr(capture, "coarse_ladder_pass", lambda *a, **k: [])
    monkeypatch.setattr(capture, "dyn_sv_pass", lambda *a, **k: [])
    monkeypatch.setattr(capture, "research_replay_pass", lambda *a, **k: [])
    monkeypatch.setattr(capture, "select", lambda candidates: (chosen, {}))


def test_gather_candidates_drops_the_scorer_before_returning(monkeypatch):
    """The scorer must not be reachable from what `_gather_candidates`
    returns: once the caller (as `main` does) drops its own reference and
    collects, a weakref to it must resolve to None."""
    scorer = _FakeScorer()
    ref = weakref.ref(scorer)
    chosen_in = [{"fixture_id": "case-0", "kind": "score_result",
                 "request": {}, "record": {}, "duration": 0.1}]
    _install_empty_pipeline(monkeypatch, chosen_in)

    chosen, index, snapshot, audit = capture._gather_candidates(
        scorer, pd.Timestamp("2026-01-01"), _args(), None,
    )

    assert snapshot == "snapshot-1"
    # The tie audit rides back with the selection: it is computed over the
    # FULL candidate population, before `select()` drops any of it, so it
    # cannot be recovered from `chosen` afterwards.
    assert audit == {"examined": 0, "exercised": 0, "closest": None}
    assert chosen == chosen_in
    del scorer
    gc.collect()
    assert ref() is None, (
        "the scorer is still reachable after _gather_candidates returned; "
        "something in chosen/index (or a closure) is holding it"
    )


def test_main_releases_the_scorer_before_write_runs(monkeypatch, tmp_path):
    """End-to-end: by the time `main()` calls `write()` (which in turn calls
    `attach_strict_probe`/`chooser_trace`/`strict_trace_one`), the scorer
    must already be gone. `write` is faked here only to observe that moment
    -- everything upstream of it is the real `main()` control flow."""
    holder = {"scorer": _FakeScorer()}
    ref = weakref.ref(holder["scorer"])
    chosen_in = [{"fixture_id": "case-0", "kind": "score_result",
                 "request": {}, "record": {}, "duration": 0.1}]
    _install_empty_pipeline(monkeypatch, chosen_in)
    monkeypatch.setattr(capture.score_mod, "Scorer", lambda: holder.pop("scorer"))

    observed = {}

    def fake_write(out_dir, chosen, index, as_of, snapshot, **kwargs):
        gc.collect()
        observed["scorer_alive"] = ref() is not None
        return {"corpus_hash": "sha256:" + "0" * 64,
                "required_axes": [], "uncovered_axes": []}

    monkeypatch.setattr(capture, "write", fake_write)

    rc = capture.main(["--out", str(tmp_path / "out")])

    assert rc == 0
    assert observed == {"scorer_alive": False}