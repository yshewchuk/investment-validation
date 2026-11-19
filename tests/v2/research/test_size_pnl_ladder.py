"""EXP-188 S3: ladder-quote table. Synthetic fixtures only; no repository."""
import gc
import weakref

import numpy as np
import pandas as pd
import pytest

from engine.v2.ops.errors import OpsError
from experiments import size_pnl_ladder as lad
from experiments import size_pnl_readers as rd

CONFLICT = "EXPERIMENT_IDENTITY_CONFLICT"
DENIED = "HOLDOUT_ACCESS_DENIED"
E1, E2 = "2024-03-15", "2024-03-22"
PLAN_COLS = ["event_id", "ticker", "event_date", "session", "entry_date", "exit_date"]


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


class FakeIndex:
    def __init__(self, groups):
        self._groups = groups

    def get(self, ticker, obs_date):
        return self._groups.get((str(ticker), pd.Timestamp(obs_date).normalize()))


def _key(ticker, day):
    return (ticker, pd.Timestamp(day))


def _loader(chains, calls):
    def load(keys):
        calls.append(set(keys))
        return FakeIndex({k: v for k, v in chains.items() if k in keys})
    return load


def _admitted(rows):
    frame = pd.DataFrame(rows, columns=["event_id", "event_date"])
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    return rd.Admitted({"events": frame.sort_values("event_id").reset_index(drop=True), "excluded": {}})


def _quotes(expiry, dte, puts, calls, bid, ask):
    return [(expiry, dte, "P", k, bid, ask) for k in puts] + [(expiry, dte, "C", k, bid, ask) for k in calls]


def _entry_rows(expiry=E1, dte=2):
    return _quotes(expiry, dte, (90, 100, 110), (90, 100, 110), 1.0, 1.2)


def _exit_rows(expiry=E1, dte=0):
    return _quotes(expiry, dte, (90, 100, 110), (90, 100, 110), 0.8, 1.0)


def _chain(ticker, day, rows, spot=100.0):
    frame = pd.DataFrame(rows, columns=["expiry", "dte", "right", "strike", "bid", "ask"])
    frame["expiry"] = pd.to_datetime(frame["expiry"])
    frame.insert(0, "ticker", ticker)
    frame["obs_date"] = pd.Timestamp(day)
    frame["spot"] = spot
    return frame


def _event(eid, **over):
    """Defaults: BMO event 2024-03-14, entry 03-13, exit 03-15, spots 100. ``None`` for a chain means it is absent."""
    base = {"ticker": eid.upper(), "day": "2024-03-14", "session": "BMO", "entry": "2024-03-13",
            "exit": "2024-03-15", "entry_rows": _entry_rows(), "exit_rows": _exit_rows(),
            "spot_entry": 100.0, "spot_exit": 100.0}
    base.update(over)
    return {"eid": eid, **base}


def _world(events):
    admitted = _admitted([(e["eid"], e["day"]) for e in events])
    plan = pd.DataFrame([{"event_id": e["eid"], "ticker": e["ticker"], "event_date": e["day"],
                          "session": e["session"], "entry_date": e["entry"], "exit_date": e["exit"]}
                         for e in events], columns=PLAN_COLS)
    chains = {}
    for e in events:
        if e["entry_rows"] is not None:
            chains[_key(e["ticker"], e["entry"])] = _chain(e["ticker"], e["entry"], e["entry_rows"],
                                                           e["spot_entry"])
        if e["exit_rows"] is not None:
            chains[_key(e["ticker"], e["exit"])] = _chain(e["ticker"], e["exit"], e["exit_rows"],
                                                          e["spot_exit"])
    return admitted, plan, chains


def _run(events, **kw):
    admitted, plan, chains = _world(events)
    calls = []
    chunks = list(lad.ladder_chunks(admitted, plan, _loader(chains, calls), **kw))
    return chunks, calls


def test_non_admitted_refused():
    admitted, plan, chains = _world([_event("a1")])
    with pytest.raises(OpsError) as exc:
        list(lad.ladder_chunks(dict(admitted), plan, _loader(chains, [])))
    assert _refusal(exc) == (DENIED, "admission_not_issued")


def test_plan_outside_admission_refused_before_any_load():
    _, plan, chains = _world([_event("a1"), _event("b2")])
    admitted = _admitted([("a1", "2024-03-14")])
    calls = []
    with pytest.raises(OpsError) as exc:
        list(lad.ladder_chunks(admitted, plan, _loader(chains, calls)))
    assert _refusal(exc) == (DENIED, "key_not_admitted")
    assert calls == []


def test_date_mismatch_and_duplicate_ids_refused():
    admitted, plan, chains = _world([_event("a1")])
    with pytest.raises(OpsError) as exc:
        moved = plan.assign(event_date="2024-03-20", entry_date="2024-03-19", exit_date="2024-03-21")
        list(lad.ladder_chunks(admitted, moved, _loader(chains, [])))
    assert _refusal(exc) == (CONFLICT, "event_date_mismatch")
    calls = []
    doubled = pd.concat([plan, plan], ignore_index=True)
    with pytest.raises(OpsError) as exc:
        list(lad.ladder_chunks(admitted, doubled, _loader(chains, calls)))
    assert _refusal(exc) == (CONFLICT, "duplicate_event_id")
    assert calls == []


@pytest.mark.parametrize("broken, expect_reason", [
    (lambda p: p.drop(columns=["session"]), "plan_malformed"),                 # missing column
    (lambda p: p.assign(entry_date="not-a-date"), "plan_malformed"),          # unparsable date
    (lambda p: p.assign(exit_date=None), "plan_malformed"),                   # NaT
])
def test_plan_malformed_refused_before_any_load(broken, expect_reason):
    admitted, plan, chains = _world([_event("a1")])
    calls = []
    with pytest.raises(OpsError) as exc:
        list(lad.ladder_chunks(admitted, broken(plan), _loader(chains, calls)))
    assert _refusal(exc) == (CONFLICT, expect_reason)
    assert calls == []


@pytest.mark.parametrize("broken, expect_reason", [
    (lambda p: p.assign(ticker=""), "plan_malformed"),                                   # empty ticker
    (lambda p: p.assign(session="XYZ"), "plan_session_invalid"),                         # unknown session
    (lambda p: p.assign(entry_date="2024-03-16", exit_date="2024-03-17"),
     "plan_dates_inconsistent"),                                                         # entry after event
    (lambda p: p.assign(entry_date="2024-03-12", exit_date="2024-03-13"),
     "plan_dates_inconsistent"),                                                         # exit before event
    (lambda p: p.assign(entry_date="2024-03-14", exit_date="2024-03-14"),
     "plan_dates_inconsistent"),                                                         # entry >= exit
])
def test_plan_fields_refused_before_any_load(broken, expect_reason):
    admitted, plan, chains = _world([_event("a1")])
    calls = []
    with pytest.raises(OpsError) as exc:
        list(lad.ladder_chunks(admitted, broken(plan), _loader(chains, calls)))
    assert _refusal(exc) == (CONFLICT, expect_reason)
    assert calls == []


def test_null_session_is_skipped_as_no_session_not_refused():
    chunks, calls = _run([_event("a1", session=None)])
    [(_, chunk)] = chunks
    assert chunk["event_id"].tolist() == []
    assert chunk["skipped"]["no_session"] == 1
    assert len(calls) == 1


def test_entry_negative_bid_set_nan_zero_bid_kept():
    entry = [(E1, 2, "P", 85, -0.5, 1.0), (E1, 2, "P", 90, 0.0, 1.2),
             (E1, 2, "P", 95, 1.0, 1.2), (E1, 2, "P", 100, 1.0, 1.2), (E1, 2, "C", 100, 1.0, 1.2)]
    [(_, chunk)], _ = _run([_event("n1", entry_rows=entry)])
    assert chunk["event_id"].tolist() == ["n1"]
    assert chunk["right"].tolist() == [0, 0, 0, 0, 1]
    assert chunk["strike"].tolist() == [85, 90, 95, 100, 100]
    assert np.isnan(chunk["bid_e"][0]) and np.isnan(chunk["ask_e"][0])
    assert chunk["bid_e"][1] == 0.0 and chunk["ask_e"][1] == 1.2
    assert chunk["bid_e"][2:].tolist() == [1.0, 1.0, 1.0]


def test_one_load_per_quarter_ascending_keys_limited():
    events = [_event("q1a", day="2024-02-14", entry="2024-02-13", exit="2024-02-15"),
              _event("q1b"),
              _event("q2a", day="2024-05-10", entry="2024-05-09", exit="2024-05-13",
                     entry_rows=_entry_rows("2024-05-17", 4), exit_rows=_exit_rows("2024-05-17", 0))]
    chunks, calls = _run(events)
    assert [label for label, _ in chunks] == ["2024Q1", "2024Q2"]
    assert [chunk["event_id"].tolist() for _, chunk in chunks] == [["q1a", "q1b"], ["q2a"]]
    assert len(calls) == 2
    assert calls[0] == {_key("Q1A", "2024-02-13"), _key("Q1A", "2024-02-15"),
                        _key("Q1B", "2024-03-13"), _key("Q1B", "2024-03-15")}
    assert calls[1] == {_key("Q2A", "2024-05-09"), _key("Q2A", "2024-05-13")}


def test_csr_layout_and_exit_lookup_by_contract():
    exit_a1 = [(E1, 0, "P", 90, 0.5, 0.7), (E1, 0, "P", 100, 0.8, 1.0), (E1, 0, "P", 110, 0.9, 1.1),
               (E1, 0, "C", 90, 1.5, 1.7), (E1, 0, "C", 100, 1.8, 2.0), (E1, 0, "C", 110, 2.0, 2.2)]
    events = [_event("a1", exit_rows=exit_a1, spot_entry=101.5, spot_exit=97.25),
              _event("a2", entry_rows=_quotes(E1, 2, (95, 100, 105), (100,), 1.0, 1.2))]
    [(label, chunk)], _ = _run(events)
    assert label == "2024Q1"
    assert chunk["event_id"].tolist() == ["a1", "a2"]
    assert chunk["offsets"].tolist() == [0, 6, 10]
    assert chunk["right"][:6].tolist() == [0, 0, 0, 1, 1, 1]
    assert chunk["strike"][:6].tolist() == [90, 100, 110, 90, 100, 110]
    assert chunk["bid_e"][:6].tolist() == [1.0] * 6
    assert chunk["ask_e"][:6].tolist() == [1.2] * 6
    assert chunk["bid_x"][:6].tolist() == [0.5, 0.8, 0.9, 1.5, 1.8, 2.0]
    assert chunk["ask_x"][:6].tolist() == [0.7, 1.0, 1.1, 1.7, 2.0, 2.2]
    assert chunk["expiry"][0] == np.datetime64(E1)
    assert chunk["dte_entry"].tolist() == [2, 2]
    assert chunk["dte_exit"].tolist() == [0, 0]
    assert chunk["spot_entry"].tolist() == [101.5, 100.0]
    assert chunk["spot_exit"].tolist() == [97.25, 100.0]


def test_exit_rungs_one_per_case():
    exit_rows = [(E1, 0, "P", 90, 0.8, 1.0),    # normal
                 (E1, 0, "P", 100, 0.0, 0.5),   # zero bid
                 (E1, 0, "P", 110, 0.1, 0.5),   # wide: (ask-bid)/mid = 1.33
                 (E1, 0, "C", 90, 1.0, 1.0),    # locked: usable
                 (E1, 0, "C", 110, 1.0, 0.5)]   # crossed; C100 absent entirely (missing strike)
    [(_, chunk)], _ = _run([_event("r1", exit_rows=exit_rows)])
    assert chunk["exit_rung"].tolist() == [1, 2, 2, 1, 3, 3]
    assert chunk["bid_x"][:4].tolist() == [0.8, 0.0, 0.1, 1.0]
    assert np.isnan(chunk["bid_x"][4:]).all()
    assert np.isnan(chunk["ask_x"][4:]).all()


def test_skip_reasons_counted_never_raised():
    events = [_event("ok"),
              _event("noentry", entry_rows=None),
              _event("noexit", exit_rows=None),
              _event("nosess", session=None),
              _event("fewputs", entry_rows=_quotes(E1, 2, (90, 100), (90, 100, 110), 1.0, 1.2)),
              _event("noexp", entry_rows=_quotes("2024-01-05", 2, (90, 100, 110), (), 1.0, 1.2),
                     exit_rows=_exit_rows("2024-01-05", 0)),
              _event("gone", exit_rows=_exit_rows(E2, 7))]
    [(_, chunk)], _ = _run(events)
    # Exit gaps keep the event: noexit and gone are kept, counted in exit_missing, not skipped.
    assert chunk["event_id"].tolist() == ["fewputs", "gone", "noexit", "ok"]
    assert chunk["skipped"] == {"no_entry_chain": 1, "no_session": 1, "structure_unresolved": 1}
    assert chunk["exit_missing"] == {"no_exit_chain": 1, "expiry_gone_at_exit": 1}


def test_all_skipped_quarter_yields_empty_chunk():
    [(_, chunk)], _ = _run([_event("nobody", entry_rows=None)])
    assert chunk["event_id"].shape == (0,)
    assert chunk["offsets"].tolist() == [0]
    assert chunk["expiry"].shape == (0,)
    assert chunk["expiry"].dtype == np.dtype("datetime64[ns]")
    assert chunk["right"].dtype == np.int8 and chunk["right"].shape == (0,)
    assert chunk["skipped"]["no_entry_chain"] == 1


def test_expiry_choice_bmo_same_day_amc_next():
    both_entry = _entry_rows(E1, 0) + _entry_rows(E2, 7)
    both_exit = _exit_rows(E1, 0) + _exit_rows(E2, 7)
    common = {"day": "2024-03-15", "entry": "2024-03-14", "exit": "2024-03-18",
              "entry_rows": both_entry, "exit_rows": both_exit}
    [(_, chunk)], _ = _run([_event("bmo", session="BMO", **common),
                            _event("amc", session="AMC", **common)])
    expiry = dict(zip(chunk["event_id"].tolist(), chunk["expiry"]))
    assert expiry["bmo"] == np.datetime64(E1)
    assert expiry["amc"] == np.datetime64(E2)


def test_entry_crossed_and_nan_set_nan_zero_bid_kept():
    entry = [(E1, 2, "P", 85, 1.0, 1.2), (E1, 2, "P", 90, 1.0, 1.2), (E1, 2, "P", 95, 0.0, 0.5),
             (E1, 2, "P", 100, 1.5, 1.0), (E1, 2, "P", 105, 0.0, 0.0), (E1, 2, "P", 110, np.nan, 1.0),
             (E1, 2, "C", 100, 1.0, 1.2)]
    [(_, chunk)], _ = _run([_event("z", entry_rows=entry)])
    assert chunk["offsets"].tolist() == [0, 7]
    assert chunk["right"].tolist() == [0, 0, 0, 0, 0, 0, 1]
    assert chunk["strike"].tolist() == [85, 90, 95, 100, 105, 110, 100]
    assert chunk["bid_e"][:2].tolist() == [1.0, 1.0]
    assert chunk["bid_e"][2] == 0.0 and chunk["ask_e"][2] == 0.5
    assert np.isnan(chunk["bid_e"][3:6]).all() and np.isnan(chunk["ask_e"][3:6]).all()
    assert chunk["bid_e"][6] == 1.0 and chunk["ask_e"][6] == 1.2


def test_invalid_duplicate_before_valid_does_not_shadow_it():
    entry = [(E1, 2, "P", 90, 1.5, 1.0)] + _entry_rows()  # crossed P90 listed first
    [(_, chunk)], _ = _run([_event("d1", entry_rows=entry)])
    assert chunk["event_id"].tolist() == ["d1"]
    assert chunk["right"][:3].tolist() == [0, 0, 0]
    assert chunk["strike"][:3].tolist() == [90, 100, 110]
    assert chunk["bid_e"][0] == 1.0 and chunk["ask_e"][0] == 1.2


def test_invalid_exit_duplicate_before_valid_does_not_shadow_it():
    exit_rows = [(E1, 0, "P", 90, 1.5, 1.0)] + _exit_rows()  # crossed P90 listed first
    [(_, chunk)], _ = _run([_event("d2", exit_rows=exit_rows)])
    assert chunk["exit_rung"][0] == 1
    assert chunk["bid_x"][0] == 0.8 and chunk["ask_x"][0] == 1.0


def test_entry_gate_needs_a_put_row_and_a_call_row_not_valid_quotes():
    no_puts = _quotes(E1, 2, (), (90, 100, 110), 1.0, 1.2)
    [(_, chunk)], _ = _run([_event("np", entry_rows=no_puts)])
    assert chunk["event_id"].tolist() == []
    assert chunk["skipped"]["structure_unresolved"] == 1
    no_calls = _quotes(E1, 2, (90, 100, 110), (), 1.0, 1.2)
    [(_, chunk)], _ = _run([_event("nc", entry_rows=no_calls)])
    assert chunk["event_id"].tolist() == []
    assert chunk["skipped"]["structure_unresolved"] == 1
    # Puts whose quotes are all invalid still count as put rows: the event is kept.
    all_nan_puts = _quotes(E1, 2, (90, 100, 110), (90, 100, 110), np.nan, 1.2)
    [(_, chunk)], _ = _run([_event("g1", entry_rows=all_nan_puts)])
    assert chunk["event_id"].tolist() == ["g1"]
    assert chunk["skipped"]["structure_unresolved"] == 0


@pytest.mark.parametrize("exit_rows, gap", [
    (None, "no_exit_chain"),
    (_exit_rows(E2, 7), "expiry_gone_at_exit"),
    (_exit_rows(E1, np.nan), "expiry_gone_at_exit"),
])
def test_missing_exit_keeps_event_at_rung3(exit_rows, gap):
    [(_, chunk)], _ = _run([_event("mx", exit_rows=exit_rows)])
    assert chunk["event_id"].tolist() == ["mx"]
    assert chunk["exit_rung"].tolist() == [3] * 6
    assert np.isnan(chunk["bid_x"]).all() and np.isnan(chunk["ask_x"]).all()
    assert chunk["dte_exit"].tolist() == [-1]
    assert chunk["dte_entry"].tolist() == [2]
    if exit_rows is None:
        assert np.isnan(chunk["spot_exit"][0])
    else:
        assert chunk["spot_exit"][0] == 100.0  # spot from the exit chain's available rows
    assert chunk["exit_missing"] == {"no_exit_chain": int(gap == "no_exit_chain"),
                                     "expiry_gone_at_exit": int(gap == "expiry_gone_at_exit")}
    assert chunk["skipped"] == {"no_entry_chain": 0, "no_session": 0, "structure_unresolved": 0}


def test_progress_lines_on_stderr(capsys):
    events = [_event("a1"),
              _event("b2", day="2024-05-10", entry="2024-05-09", exit="2024-05-13",
                     entry_rows=_entry_rows("2024-05-17", 4), exit_rows=_exit_rows("2024-05-17", 0))]
    _run(events)
    err = capsys.readouterr().err
    assert "[pnl_ladder] start events=2 quarters=2" in err
    assert "[pnl_ladder] 2024Q1 1/2 events=1 kept=1" in err
    assert "[pnl_ladder] 2024Q2 2/2 events=1 kept=1" in err
    assert "exit_missing=" in err
    assert "rung1=" in err and "rung3=" in err
    assert "[pnl_ladder] end" in err


LIVE_INDEXES = weakref.WeakSet()


class CountedIndex(FakeIndex):
    """FakeIndex that registers each live instance so the test can count them."""

    def __init__(self, groups):
        super().__init__(groups)
        LIVE_INDEXES.add(self)


def test_only_one_chain_index_resident_across_quarters():
    events = [_event("q1a", day="2024-02-14", entry="2024-02-13", exit="2024-02-15"),
              _event("q1b"),
              _event("q2a", day="2024-05-10", entry="2024-05-09", exit="2024-05-13",
                     entry_rows=_entry_rows("2024-05-17", 4), exit_rows=_exit_rows("2024-05-17", 0))]
    admitted, plan, chains = _world(events)
    live_at_load = []

    def load(keys):
        gc.collect()
        live_at_load.append(len(LIVE_INDEXES))
        return CountedIndex({k: v for k, v in chains.items() if k in keys})

    gen = lad.ladder_chunks(admitted, plan, load)
    label, chunk = next(gen)  # generator now suspended at its first yield
    assert label == "2024Q1"
    assert chunk["event_id"].tolist() == ["q1a", "q1b"]
    gc.collect()
    assert len(LIVE_INDEXES) == 0  # first quarter's index already freed while suspended at yield

    rest = list(gen)
    assert [lbl for lbl, _ in rest] == ["2024Q2"]
    assert live_at_load == [0, 0]  # no index was alive when either quarter's load ran
    assert len(LIVE_INDEXES) == 0
