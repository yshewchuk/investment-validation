"""Regression tests for the 2026-09-18 boundary/rescore ChainIndex fix, and
the 2026-09-19 research_replay_pass follow-up.

An instrumented capture run (``scratch/diag_capture_serving.py``) measured
``boundary_pass``'s own ``_price_entry`` calls building a FRESH ``ChainIndex``
per boundary request — transients up to +673 MB/call, the run's net RSS
climbing in steps — because every boundary request reached ``Scorer.score``
with ``chain_index=None``: ``_price_entry``'s cache-miss branch streams its
own 1-2-key ``load_chain_index`` call against the whole ``option_chains``
table, every single time.

The fix mirrors ``forward_pass``'s own existing (already-correct) pattern:
collect every ``(ticker, date)`` key up front, from the plan rather than the
chain, and load ONE ``ChainIndex`` before any request is scored — applied to
``boundary_pass`` and, since a pinned/strike/coarse rescore of ANY candidate
always needs exactly the chain keys its source already resolved (see
``_rescore``'s own docstring), to ``pinned_and_strike_pass``/
``coarse_ladder_pass`` too.

These tests are synthetic throughout: no real ``Scorer``, no real chain
table read, no panel. Every collaborator (``_score``, ``STRUCTURES``,
``replay_mod.plan_events``/``load_chain_index``/``latest_chain_date``,
``score_mod.DISABLED_STRATEGIES``) is a fake recording what it was called
with — consistent with this repo's "tests are synthetic" rule for anything
touching ``data/``.
"""
from __future__ import annotations

import pandas as pd
import pytest

import tools.capture_tier0_corpus as capture


class _FakeCalendar:
    """A stand-in for ``engine.calendar.TradingCalendar`` — never read by
    anything under test here; only its identity matters (passed through to
    ``plan_events``, which is itself faked).
    """


class _FakeScorer:
    def __init__(self):
        self.calendar = _FakeCalendar()


class _FakeStructure:
    """A stand-in `Structure` — never introspected beyond `to_dict()`,
    which `research_replay_pass` calls to build its own request record.
    """

    def to_dict(self) -> dict:
        return {"kind": "fake"}


class _FakePlan:
    def __init__(self, frame: pd.DataFrame, keys: set[tuple[str, pd.Timestamp]]):
        self.frame = frame
        self._keys = keys

    @property
    def chain_keys(self) -> set[tuple[str, pd.Timestamp]]:
        return self._keys


def _key(ticker: str, date: str) -> tuple[str, pd.Timestamp]:
    return (ticker, pd.Timestamp(date))


@pytest.fixture
def fakes(monkeypatch):
    """Install fake `STRUCTURES`/`DISABLED_STRATEGIES`/`plan_events`/
    `load_chain_index`/`latest_chain_date`/`_score`/`_candidate`, and return
    a small namespace of call-recording lists the tests inspect.

    Two ordinary strategies ("S1", "S2") and one disabled one ("DIS"), each
    with its own distinct plan/chain-key set, so a test can tell whether a
    key set was unioned across strategies and whether the disabled one's
    plan was ever built at all.
    """
    calls = {
        "plan_events": [],
        "load_chain_index": [],
        "latest_chain_date": [],
        "score": [],
        "candidate": [],
    }

    monkeypatch.setattr(
        capture, "STRUCTURES",
        {"S1": _FakeStructure, "S2": _FakeStructure, "DIS": _FakeStructure},
    )
    monkeypatch.setattr(capture.score_mod, "DISABLED_STRATEGIES", {"DIS": "off"})

    plans_by_structure = {
        "S1": _FakePlan(
            pd.DataFrame([
                {"ticker": "AAA", "event_date": pd.Timestamp("2026-01-05"),
                 "session": "BMO", "decision_date": pd.Timestamp("2026-01-05"),
                 "event_id": "e1"},
            ]),
            {_key("AAA", "2026-01-05"), _key("AAA", "2026-01-07")},
        ),
        "S2": _FakePlan(
            pd.DataFrame([
                {"ticker": "BBB", "event_date": pd.Timestamp("2026-02-10"),
                 "session": "AMC", "decision_date": pd.Timestamp("2026-02-10"),
                 "event_id": "e2"},
            ]),
            {_key("BBB", "2026-02-10")},
        ),
        # A disabled strategy's plan must never even be requested -- see
        # `_boundary_chain_keys`'/`_forward_chain_keys`'s docstrings.
        "DIS": _FakePlan(pd.DataFrame(), {_key("ZZZ", "1999-01-01")}),
    }

    def fake_plan_events(structure, events, calendar=None):
        # Identify which structure by its slot in plans_by_structure via a
        # reverse lookup on STRUCTURES -- structure is literally
        # capture.STRUCTURES[name], i.e. the `_FakeStructure` CLASS itself
        # for every name here, so key on `events` identity/content instead:
        # each test passes a distinguishable `events` frame per call site.
        calls["plan_events"].append((structure, events, calendar))
        name = events.attrs["fake_strategy"]
        return plans_by_structure[name]

    def fake_load_chain_index(keys, progress_every=0):
        calls["load_chain_index"].append(frozenset(keys))
        return capture.replay_mod.ChainIndex(
            {k: pd.DataFrame({"marker": [str(k)]}) for k in keys}
        )

    def fake_latest_chain_date(ticker, on_or_before):
        calls["latest_chain_date"].append((ticker, on_or_before))
        return pd.Timestamp("2025-12-31")

    def fake_score(scorer, request, *, index=None):
        calls["score"].append((request, index))
        return ({"ok": True}, {"ok": True}, 0.01, {})

    def fake_candidate(request, raw, record, took, **kwargs):
        calls["candidate"].append((request, raw, record, took, kwargs))
        return {"request": request, "raw": raw, "record": record, "kind": "score_result"}

    monkeypatch.setattr(capture.replay_mod, "plan_events", fake_plan_events)
    monkeypatch.setattr(capture.replay_mod, "load_chain_index", fake_load_chain_index)
    monkeypatch.setattr(capture.replay_mod, "latest_chain_date", fake_latest_chain_date)
    monkeypatch.setattr(capture, "_score", fake_score)
    monkeypatch.setattr(capture, "_candidate", fake_candidate)

    return calls


def _events_for(name: str) -> pd.DataFrame:
    frame = pd.DataFrame({"ticker": ["ZZZ"], "event_date": [pd.Timestamp("2026-01-01")],
                          "session": ["BMO"], "event_id": ["e0"]})
    frame.attrs["fake_strategy"] = name
    return frame


# -- _boundary_chain_keys -----------------------------------------------------


def test_boundary_chain_keys_unions_across_strategies(fakes):
    """`_boundary_chain_keys` takes one events frame for every requested
    strategy; the fake `plan_events` above resolves a strategy's plan from a
    tag on that SAME frame, so this drives it once per single-strategy
    selection and unions by hand -- exactly what `_score_strategies` would
    hand a multi-strategy call internally.
    """
    keys_s1 = capture._boundary_chain_keys(_FakeScorer(), _events_for("S1"), ["S1"])
    keys_s2 = capture._boundary_chain_keys(_FakeScorer(), _events_for("S2"), ["S2"])

    assert keys_s1 == {_key("AAA", "2026-01-05"), _key("AAA", "2026-01-07")}
    assert keys_s2 == {_key("BBB", "2026-02-10")}
    assert keys_s1 | keys_s2 == {
        _key("AAA", "2026-01-05"), _key("AAA", "2026-01-07"), _key("BBB", "2026-02-10"),
    }


def test_boundary_chain_keys_skips_disabled_strategy_entirely(fakes):
    """A disabled strategy's plan must never even be requested -- it can
    never reach `_price_entry` (`Scorer.score`'s own early return), so
    planning its chain keys would load data no request ever reads.
    """
    keys_dis = capture._boundary_chain_keys(_FakeScorer(), _events_for("DIS"), ["DIS"])

    assert keys_dis == set()
    assert fakes["plan_events"] == []


def test_boundary_chain_keys_never_calls_latest_chain_date(fakes):
    """Unlike `_forward_chain_keys`, boundary requests never set
    `quote_max_age_sessions`, so the stale-quote fallback branch is
    unreachable and there is nothing to preload for it.
    """
    capture._boundary_chain_keys(_FakeScorer(), _events_for("S1"), ["S1"])
    assert fakes["latest_chain_date"] == []


def test_forward_chain_keys_still_calls_latest_chain_date(fakes):
    """The forward-only fallback preload is unchanged by this round's fix."""
    events = _events_for("S1")
    capture._forward_chain_keys(_FakeScorer(), events, pd.Timestamp("2026-01-01"), ["S1"])
    assert fakes["latest_chain_date"] == [("ZZZ", pd.Timestamp("2026-01-01"))]


# -- boundary_pass: builds ONE index, shares it across every request --------


def test_boundary_pass_builds_exactly_one_index_when_none_given(fakes):
    events = _events_for("S1")
    out = capture.boundary_pass(_FakeScorer(), events, ["S1"])

    assert len(fakes["load_chain_index"]) == 1
    assert len(out) == 1  # one row in S1's fake plan
    # Every _score call got the SAME index object -- not just an equal one.
    used_indexes = [index for _request, index in fakes["score"]]
    assert len(used_indexes) == 1
    assert used_indexes[0] is not None


def test_boundary_pass_reuses_a_caller_supplied_index_with_zero_new_loads(fakes):
    given_index = capture.replay_mod.ChainIndex({})
    events = _events_for("S1")
    capture.boundary_pass(_FakeScorer(), events, ["S1"], index=given_index)

    assert fakes["load_chain_index"] == []  # never builds its own
    used_indexes = [index for _request, index in fakes["score"]]
    assert all(index is given_index for index in used_indexes)


def test_boundary_pass_shares_one_index_across_many_requests_not_one_per_request(
    fakes, monkeypatch,
):
    """The actual regression this round fixes: N boundary requests must
    cost ONE `load_chain_index` call, not N (previously: N, one per
    `_price_entry` cache-miss -- see this module's own docstring).
    """
    many_rows = pd.DataFrame([
        {"ticker": "AAA", "event_date": pd.Timestamp("2026-01-05"),
         "session": "BMO", "decision_date": pd.Timestamp("2026-01-05"),
         "event_id": f"e{i}"}
        for i in range(25)
    ])
    plan = _FakePlan(many_rows, {_key("AAA", "2026-01-05")})
    monkeypatch.setattr(
        capture.replay_mod, "plan_events",
        lambda structure, events, calendar=None: plan,
    )

    out = capture.boundary_pass(_FakeScorer(), pd.DataFrame({"ticker": []}), ["S1"])

    assert len(out) == 25
    assert len(fakes["load_chain_index"]) == 1
    # And every one of the 25 requests got the SAME index object.
    used_indexes = {id(index) for _request, index in fakes["score"]}
    assert len(used_indexes) == 1


# -- _rescore / pinned_and_strike_pass / coarse_ladder_pass ------------------


def test_rescore_passes_index_straight_through_to_score(fakes):
    source = {
        "request": {
            "ticker": "AAA", "strategy": "S1", "as_of": "2026-01-05",
            "event_date": "2026-01-05", "session": "BMO",
        },
        "record": {}, "event_id": "e1",
    }
    given_index = capture.replay_mod.ChainIndex({})
    capture._rescore(_FakeScorer(), source, "pinned",
                     index=given_index, structure_params={"width_moneyness": 0.01})

    assert len(fakes["score"]) == 1
    _request, used_index = fakes["score"][0]
    assert used_index is given_index


def test_pinned_and_strike_pass_threads_index_into_rescore(monkeypatch):
    calls = []

    def fake_rescore(scorer, source, label, index=None, **changes):
        calls.append((label, index, changes))
        return None

    monkeypatch.setattr(capture, "_rescore", fake_rescore)
    monkeypatch.setattr(capture, "priced", lambda record: True)

    scored = [{
        "record": {"structure_params": {"width_moneyness": 0.02}, "legs": [
            {"name": "atm", "strike": 100.0},
        ]},
        "request": {"ticker": "AAA"},
    }]
    given_index = capture.replay_mod.ChainIndex({})
    capture.pinned_and_strike_pass(_FakeScorer(), scored, index=given_index)

    assert len(calls) == 2  # "pinned" and "strike"
    assert all(index is given_index for _label, index, _changes in calls)


def test_coarse_ladder_pass_threads_index_into_rescore(monkeypatch):
    calls = []

    def fake_rescore(scorer, source, label, index=None, **changes):
        calls.append((label, index, changes))
        return {"record": {"flags": ["COARSE_LADDER"]}}

    monkeypatch.setattr(capture, "_rescore", fake_rescore)

    scored = [{"record": {"spot": 100.0, "strategy": "TWIN-P5"}}]
    given_index = capture.replay_mod.ChainIndex({})
    result = capture.coarse_ladder_pass(_FakeScorer(), scored, index=given_index)

    assert len(result) == 1
    assert calls[0][1] is given_index


# -- _research_replay_chain_keys / research_replay_pass ---------------------
#
# 2026-09-19: research_replay_pass had the identical "fresh index per pass"
# gap as boundary_pass/_rescore, one level down -- it calls
# replay_mod.replay_one directly (never Scorer.score), so it is NOT covered
# by _forward_chain_keys'/_boundary_chain_keys' DISABLED_STRATEGIES skip
# (that skip is correct for THEM: a disabled strategy never reaches
# _price_entry through Scorer.score). _research_replay_chain_keys is the
# disabled-strategy counterpart those two must not provide, and
# research_replay_pass now accepts a caller-supplied index the same way
# boundary_pass does.


def test_research_replay_chain_keys_collects_disabled_strategies_only(fakes):
    """Unlike `_boundary_chain_keys`, this walks `DISABLED_STRATEGIES` --
    "DIS" here -- and does NOT skip it; its whole point is the keys the
    other two functions must not plan.
    """
    keys = capture._research_replay_chain_keys(_FakeScorer(), _events_for("DIS"), ["DIS"])

    assert keys == {_key("ZZZ", "1999-01-01")}


def test_research_replay_chain_keys_respects_strategy_filter(fakes):
    """A `strategies` filter that excludes the disabled strategy leaves
    nothing to plan -- no plan is built at all.
    """
    keys = capture._research_replay_chain_keys(_FakeScorer(), _events_for("DIS"), ["S1"])

    assert keys == set()
    assert fakes["plan_events"] == []


def test_research_replay_pass_builds_exactly_one_index_per_strategy_when_none_given(
    fakes, monkeypatch,
):
    many_rows = pd.DataFrame([{"row": i} for i in range(5)])
    plan = _FakePlan(many_rows, {_key("ZZZ", "1999-01-01")})
    monkeypatch.setattr(
        capture.replay_mod, "plan_events",
        lambda structure, events, calendar=None: plan,
    )
    replay_calls = []

    def fake_replay_one(structure, row, index, *, include_legs=True):
        replay_calls.append(index)
        return ([{"leg": "x"}], None)

    monkeypatch.setattr(capture.replay_mod, "replay_one", fake_replay_one)

    out = capture.research_replay_pass(_FakeScorer(), _events_for("DIS"), limit=8,
                                       strategies=["DIS"])

    assert len(fakes["load_chain_index"]) == 1  # one DISABLED_STRATEGIES entry
    assert len(out) == 5
    used_indexes = {id(index) for index in replay_calls}
    assert len(used_indexes) == 1  # every row shared the ONE index built


def test_research_replay_pass_reuses_a_caller_supplied_index_with_zero_new_loads(
    fakes, monkeypatch,
):
    many_rows = pd.DataFrame([{"row": i} for i in range(5)])
    plan = _FakePlan(many_rows, {_key("ZZZ", "1999-01-01")})
    monkeypatch.setattr(
        capture.replay_mod, "plan_events",
        lambda structure, events, calendar=None: plan,
    )
    replay_calls = []

    def fake_replay_one(structure, row, index, *, include_legs=True):
        replay_calls.append(index)
        return ([{"leg": "x"}], None)

    monkeypatch.setattr(capture.replay_mod, "replay_one", fake_replay_one)

    given_index = capture.replay_mod.ChainIndex({})
    out = capture.research_replay_pass(_FakeScorer(), _events_for("DIS"), limit=8,
                                       strategies=["DIS"], index=given_index)

    assert fakes["load_chain_index"] == []  # never builds its own
    assert len(out) == 5
    assert all(index is given_index for index in replay_calls)
