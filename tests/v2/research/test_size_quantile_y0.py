"""EXP-190 Q2c Y0 report: synthetic fixtures only (no store, no data/). Expected values are hand-written literals."""
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.data import objects
from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.data.repository import Repository
from engine.v2.ops.errors import OpsError
from engine.v2.research._pricing import TradingCalendar
from experiments.size_quantile_y0 import capped_chain_keys, y0_report
from tests.test_v2_research_chain_read_cap import _commit_chains, _rows

# (event_id, ticker, event_date, session, entry, exit, has_entry, has_exit). entry/exit are the hand-written calendar
# dates for the event's own session: AMC = (day, next weekday); BMO = (previous weekday, day). Weekdays only.
ROWS = [
    ("e19a", "T19a", "2019-03-12", "AMC", "2019-03-12", "2019-03-13", True, True),
    ("e19b", "T19b", "2019-04-10", "BMO", "2019-04-09", "2019-04-10", True, True),
    ("e19c", "T19c", "2019-06-13", "AMC", "2019-06-13", "2019-06-14", True, False),
    ("e19d", "T19d", "2019-09-17", "XX", None, None, False, False),
    ("e20a", "T20a", "2020-03-10", "AMC", "2020-03-10", "2020-03-11", True, True),
    ("e20b", "T20b", "2020-05-13", "BMO", "2020-05-12", "2020-05-13", True, True),
    ("e20c", "T20c", "2020-07-08", "AMC", "2020-07-08", "2020-07-09", False, False),
    ("e20d", "T20d", "2020-10-15", "XX", None, None, False, False),
    ("e21a", "T21a", "2021-02-09", "AMC", "2021-02-09", "2021-02-10", True, True),
    ("e21b", "T21b", "2021-05-11", "BMO", "2021-05-10", "2021-05-11", True, True),
    ("e21c", "T21c", "2021-08-10", "AMC", "2021-08-10", "2021-08-11", True, True),
    ("e21d", "T21d", "2021-10-13", "AMC", "2021-10-13", "2021-10-14", True, False),
    ("e22a", "T22a", "2022-03-15", "BMO", "2022-03-14", "2022-03-15", True, True),
    ("e22b", "T22b", "2022-07-12", "AMC", "2022-07-12", "2022-07-13", True, True),
    ("e22c", "T22c", "2022-10-18", "XX", None, None, False, False),
    ("e23a", "T23a", "2023-02-14", "AMC", "2023-02-14", "2023-02-15", True, True),
    ("e23b", "T23b", "2023-04-12", "BMO", "2023-04-11", "2023-04-12", True, True),
    ("e23c", "T23c", "2023-06-15", "AMC", "2023-06-15", "2023-06-16", True, True),
    ("e23d", "T23d", "2023-09-13", "BMO", "2023-09-12", "2023-09-13", False, False),
    ("e24a", "T24a", "2024-03-12", "BMO", "2024-03-11", "2024-03-12", True, True),
    ("e24b", "T24b", "2024-08-14", "AMC", "2024-08-14", "2024-08-15", True, False),
    ("e25a", "T25a", "2025-02-11", "AMC", "2025-02-11", "2025-02-12", True, True),
    ("e25b", "T25b", "2025-05-14", "BMO", "2025-05-13", "2025-05-14", False, False),
    ("e25c", "T25c", "2025-09-11", "AMC", "2025-09-11", "2025-09-12", True, True),
]
# Calendar (sorted) months of every window key, one chain_keys call each.
MONTHS = [(2019, 3), (2019, 4), (2019, 6), (2020, 3), (2020, 5), (2020, 7), (2021, 2), (2021, 5), (2021, 8),
          (2021, 10), (2022, 3), (2022, 7), (2023, 2), (2023, 4), (2023, 6), (2023, 9), (2024, 3), (2024, 8),
          (2025, 2), (2025, 5), (2025, 9)]
CAL = TradingCalendar(pd.bdate_range("2018-12-01", "2026-06-30"))
KEYS = ("year", "events", "no_window", "with_chains", "chain_share", "fold", "training_years", "passes")
HAPPY = [
    dict(zip(KEYS, ("2019", 4, 1, 2, 0.5, "population_collapsed", None, False))),
    dict(zip(KEYS, ("2020", 4, 1, 2, 0.5, "ok", 1, False))),
    dict(zip(KEYS, ("2021", 4, 0, 3, 0.75, "ok", 2, True))),
    dict(zip(KEYS, ("2022", 3, 1, 2, 2 / 3, "ok", 3, True))),
    dict(zip(KEYS, ("2023", 4, 0, 3, 0.75, "ok", 4, True))),
    dict(zip(KEYS, ("2024", 2, 0, 1, 0.5, "ok", 5, True))),
    dict(zip(KEYS, ("2025", 3, 0, 2, 2 / 3, "ok", 6, True))),
]


def _row(*values):
    return dict(zip(KEYS, values))


def _year(row):
    return row[2][:4]


def _ts(pairs):
    return {(t, pd.Timestamp(d)) for t, d in pairs}


def _present(rows=ROWS):
    return (_ts((r[1], r[4]) for r in rows if r[6]) | _ts((r[1], r[5]) for r in rows if r[7]))


def _spec(share=0.5, floor=2):
    return {"walk_forward": {"chain_share_min": share, "training_years_min": floor}}


SPEC = _spec()


class _FakeRepo:
    def scan(self, query, *, table_name, max_object_bytes=None):
        raise AssertionError("unexpected scan")


REPO = _FakeRepo()


def _sealed(rows=ROWS, admitted=None):
    return {
        "binding": "bind-y0", "repository": REPO, "snapshot": "snap",
        "events": [{"event_id": r[0], "ticker": r[1], "event_date": r[2], "session": r[3]}
                   for r in rows if admitted is None or r[0] in admitted],
        "context": {"target_events": [{"event_id": r[0], "ticker": r[1], "event_date": r[2],
                                       "label_end": f"{_year(r)}-11-30"} for r in rows],
                    "eligibility_plan": {"random_windows": []}},
        "fold_events": [{"event_id": r[0], "ticker": r[1], "event_date": r[2],
                         "forecasts": [{"source_session_ceiling": r[2]}]} for r in rows],
        "evaluation_labels_before": "2030-01-01",
        "plan": {"folds": [{"outer": {"year": "2025", "training_candidate_event_ids": [
                                r[0] for r in rows if r[2] < "2025-01-01"]},
                            "inner": [{"year": y, "training_candidate_event_ids": [
                                r[0] for r in rows if r[2] < f"{y}-01-01"]} for y in ("2023", "2024")]}]},
    }


class Recorder:
    def __init__(self, present=None, store=None):
        self.present = _present() if present is None else present
        self.store = {r[0]: f"{_year(r)}-03-01" for r in ROWS} if store is None else store
        self.calendar_calls, self.chain_calls, self.label_calls = [], [], []
        self.chain_caps = []
        self.chain_error = None

    def calendar(self, repository, snapshot, *, max_object_bytes):
        self.calendar_calls.append((repository, snapshot, max_object_bytes))
        return CAL

    def chain_keys(self, repository, snapshot, keys, *, max_object_bytes):
        self.chain_calls.append(list(keys))
        self.chain_caps.append(max_object_bytes)
        if self.chain_error is not None:
            raise self.chain_error
        return {k for k in keys if k in self.present}

    def labels(self, sealed, ids, before):
        self.label_calls.append((list(ids), before))
        rows = [(i, self.store[i]) for i in ids if i in self.store and self.store[i] < before]
        if not rows:
            raise data_fail("POPULATION_COLLAPSED", "no qualified training label")
        return pd.DataFrame(rows, columns=["event_id", "target_available_on"])


def _run(rec, sealed=None, spec=SPEC, **kwargs):
    return y0_report(_sealed() if sealed is None else sealed, spec, calendar=rec.calendar,
                     chain_keys=rec.chain_keys, labels=rec.labels, **kwargs)


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def test_happy_path_returns_the_hand_written_report_and_calls_each_source_once(capsys):
    rec = Recorder()
    out = _run(rec, max_object_bytes=1234, chain_max_object_bytes=5678)
    assert out == {"y0": "2021", "sealed_binding": "bind-y0", "chain_share_min": 0.5, "training_years_min": 2,
                   "years": HAPPY}
    assert rec.calendar_calls == [(REPO, "snap", 1234)]
    assert rec.chain_caps == [5678] * len(MONTHS)
    assert rec.label_calls == [(sorted(r[0] for r in ROWS if _year(r) <= "2024"), "2025-01-01")]
    assert [{(k[1].year, k[1].month) for k in call} for call in rec.chain_calls] == [{m} for m in MONTHS]
    err = capsys.readouterr().err
    for line in ("  [quantile y0] year=2019 events=4 no_window=1 with_chains=2 chain_share=0.500 "
                 "fold=population_collapsed training_years=na passes=no",
                 "  [quantile y0] year=2021 events=4 no_window=0 with_chains=3 chain_share=0.750 "
                 "fold=ok training_years=2 passes=yes",
                 "  [quantile y0] year=2024 events=2 no_window=0 with_chains=1 chain_share=0.500 "
                 "fold=ok training_years=5 passes=yes",
                 "  [quantile y0] y0=2021"):
        assert line in err


@pytest.mark.parametrize("remove, add", [
    # BMO event 2021-05-11: its own pair is (05-10, 05-11); the AMC pair (05-11, 05-12) does not cover it.
    ([("T21b", "2021-05-10"), ("T21b", "2021-05-11")], [("T21b", "2021-05-11"), ("T21b", "2021-05-12")]),
    # AMC event 2021-08-10: its own pair is (08-10, 08-11); the BMO pair (08-09, 08-10) does not cover it.
    ([("T21c", "2021-08-10"), ("T21c", "2021-08-11")], [("T21c", "2021-08-09"), ("T21c", "2021-08-10")]),
])
def test_only_the_session_correct_pair_covers_an_event(remove, add):
    present = (_present() - _ts(remove)) | _ts(add)
    out = _run(Recorder(present=present))
    assert out["years"][2] == _row("2021", 4, 0, 2, 0.5, "ok", 2, True)


def test_share_exactly_at_the_minimum_passes_and_just_below_fails():
    at = _run(Recorder(), spec=_spec(share=0.5))
    below = _run(Recorder(), spec=_spec(share=0.5 + 1e-9))
    assert at["years"][5]["passes"] is True and at["y0"] == "2021"
    assert below["years"][5]["passes"] is False and below["y0"] == "2025"


def test_a_later_failing_year_moves_y0_after_it():
    gone = _ts([("T23a", "2023-02-14"), ("T23a", "2023-02-15"), ("T23b", "2023-04-11"), ("T23b", "2023-04-12")])
    out = _run(Recorder(present=_present() - gone))
    assert out["years"][4] == _row("2023", 4, 0, 1, 0.25, "ok", 4, False)
    assert out["years"][2]["passes"] and out["years"][3]["passes"]
    assert out["y0"] == "2024"


@pytest.mark.parametrize("store, training", [
    # 2019 labels absent: fold 2020 counts 0 years, and each later fold counts one fewer than the happy path.
    pytest.param({r[0]: f"{_year(r)}-03-01" for r in ROWS if _year(r) != "2019"},
                 [None, 0, 1, 2, 3, 4, 5], id="absent"),
    # 2020 labels available exactly on the 2021 cutoff: strictly not before it, so fold 2021 does not count 2020.
    pytest.param({r[0]: ("2021-01-01" if _year(r) == "2020" else f"{_year(r)}-03-01") for r in ROWS},
                 [None, 1, 1, 3, 4, 5, 6], id="equal_boundary"),
])
def test_training_years_count_only_labels_qualified_for_that_fold(store, training):
    out = _run(Recorder(store=store))
    assert [r["training_years"] for r in out["years"]] == training
    assert [r["passes"] for r in out["years"]] == [False, False, False, True, True, True, True]
    assert out["y0"] == "2022"


def test_year_without_admitted_events_is_population_collapsed_not_a_refusal():
    sealed = _sealed(admitted=[r[0] for r in ROWS if _year(r) != "2022"])
    out = _run(Recorder(), sealed=sealed)
    assert out["years"][3] == _row("2022", 0, 0, 0, None, "population_collapsed", None, False)
    assert out["years"][4:] == [_row("2023", 4, 0, 3, 0.75, "ok", 3, True),
                                _row("2024", 2, 0, 1, 0.5, "ok", 4, True),
                                _row("2025", 3, 0, 2, 2 / 3, "ok", 5, True)]
    assert out["y0"] == "2023"


def test_unknown_session_is_no_window_and_session_is_normalised():
    sealed = _sealed()
    for ev in sealed["events"]:
        if ev["event_id"] == "e21a":
            ev["session"] = "Xyz"
        if ev["event_id"] == "e21b":
            ev["session"] = " bmo "
    out = _run(Recorder(), sealed=sealed)
    assert out["years"][2] == _row("2021", 4, 1, 2, 0.5, "ok", 2, True)


def test_no_y0_refuses_after_the_year_lines_are_printed(capsys):
    with pytest.raises(OpsError) as exc:
        _run(Recorder(), spec=_spec(share=1.0))
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "y0_none")
    err = capsys.readouterr().err
    assert ("  [quantile y0] year=2025 events=3 no_window=0 with_chains=2 chain_share=0.667 "
            "fold=ok training_years=6 passes=no") in err
    assert "  [quantile y0] y0=none" in err


def test_labels_with_no_qualified_row_leave_no_training_years_and_refuse(capsys):
    with pytest.raises(OpsError) as exc:
        _run(Recorder(store={}))
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "y0_none")
    assert "  [quantile y0] y0=none" in capsys.readouterr().err


BAD_SPECS = [
    pytest.param(None, id="no_spec"),
    pytest.param({}, id="no_walk_forward"),
    pytest.param(_spec(share=True), id="share_bool"),
    pytest.param(_spec(share=0), id="share_zero"),
    pytest.param(_spec(share=1.5), id="share_above_one"),
    pytest.param(_spec(share="0.5"), id="share_str"),
    pytest.param(_spec(share=float("nan")), id="share_nan"),
    pytest.param(_spec(floor=0), id="floor_zero"),
    pytest.param(_spec(floor=2.0), id="floor_float"),
    pytest.param(_spec(floor=True), id="floor_bool"),
]


@pytest.mark.parametrize("spec", BAD_SPECS)
def test_bad_thresholds_refuse_before_any_repository_calendar_or_label_access(spec):
    rec = Recorder()
    with pytest.raises(OpsError) as exc:
        _run(rec, spec=spec)
    assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "y0_inputs_invalid")
    assert rec.calendar_calls == [] and rec.chain_calls == [] and rec.label_calls == []


MALFORMED = [
    pytest.param(lambda ev: ev.update(event_date=None), id="null_date"),
    pytest.param(lambda ev: ev.update(event_date="not a date"), id="bad_date"),
    pytest.param(lambda ev: ev.pop("session"), id="missing_session"),
]


@pytest.mark.parametrize("break_event", MALFORMED)
def test_malformed_sealed_event_refuses(break_event):
    rec = Recorder()
    sealed = _sealed()
    break_event(sealed["events"][0])
    with pytest.raises(OpsError) as exc:
        _run(rec, sealed=sealed)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "sealed_malformed")
    assert rec.calendar_calls == []


def test_parity_mismatch_propagates_unchanged():
    sealed = _sealed()
    sealed["plan"]["folds"][0]["inner"][1]["training_candidate_event_ids"].pop()
    with pytest.raises(OpsError) as exc:
        _run(Recorder(), sealed=sealed)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "fold_parity_mismatch")


def test_memory_error_becomes_resource_limit():
    rec = Recorder()
    rec.chain_error = MemoryError()
    with pytest.raises(OpsError) as exc:
        _run(rec)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"


def test_events_after_2025_are_ignored():
    rows = ROWS + [("e26a", "T26a", "2026-02-10", "AMC", "2026-02-10", "2026-02-11", True, True)]
    rec = Recorder()
    out = _run(rec, sealed=_sealed(rows))
    assert out["years"] == HAPPY and out["y0"] == "2021"
    assert all(k[1].year <= 2025 for call in rec.chain_calls for k in call)


class _IntSub(int):
    pass


BAD_CAPS = [
    pytest.param(None, id="none"),
    pytest.param(0, id="zero"),
    pytest.param(True, id="bool"),
    pytest.param(1.5, id="float"),
    pytest.param(_IntSub(5), id="int_subclass"),
]


@pytest.mark.parametrize("keyword", ["max_object_bytes", "chain_max_object_bytes"])
@pytest.mark.parametrize("cap", BAD_CAPS)
def test_bad_byte_cap_refuses_before_any_calendar_chain_or_label_access(keyword, cap):
    rec = Recorder()
    with pytest.raises(DataError) as exc:
        _run(rec, **{keyword: cap})
    assert exc.value.code == "QUERY_NOT_BOUNDED"
    assert rec.calendar_calls == [] and rec.chain_calls == [] and rec.label_calls == []


@pytest.mark.parametrize("cap", BAD_CAPS)
def test_capped_chain_keys_refuses_a_bad_cap_before_any_repository_access(cap):
    with pytest.raises(DataError) as exc:
        capped_chain_keys(object(), "snap", [("TEST", pd.Timestamp("2024-06-15"))], max_object_bytes=cap)
    assert exc.value.code == "QUERY_NOT_BOUNDED"


def test_capped_chain_keys_returns_exact_present_pairs_and_propagates_the_cap_refusal(tmp_path):
    entries = [
        ("TEST", "2024-06-15", 10), ("TEST", "2024-06-15", 11),  # wanted pair, two rows: deduplicated
        ("ZEBRA", "2024-09-01", 12),                             # wanted pair
        ("TEST", "2024-09-01", 13),                              # wanted ticker and date, not wanted as a pair
    ] + [(f"AAA{i}", f"2024-{i:02d}-10", 100 + i) for i in range(1, 13)]
    conn, store, snap = _commit_chains(tmp_path, {"2024": _rows(2024, entries)}, maximum_batch_rows=2)
    repository = Repository(conn, store)
    # ZEBRA@2024-06-15: its ticker and its date both exist in the table, but never as this pair.
    wanted = [("TEST", pd.Timestamp("2024-06-15")), ("ZEBRA", pd.Timestamp("2024-09-01")),
              ("ZEBRA", pd.Timestamp("2024-06-15"))]
    try:
        found = capped_chain_keys(repository, snap, wanted, max_object_bytes=1 << 30)
        assert found == {("TEST", pd.Timestamp("2024-06-15")), ("ZEBRA", pd.Timestamp("2024-09-01"))}
        with pytest.raises(DataError) as exc:
            capped_chain_keys(repository, snap, wanted, max_object_bytes=1)
        assert exc.value.code == "RESULT_LIMIT_EXCEEDED"
    finally:
        conn.close()


class _ScanRepo:
    marker = "extra"

    def __init__(self):
        self.calls = []

    def scan(self, query, *, table_name, max_object_bytes=None):
        self.calls.append((table_name, max_object_bytes))
        return "rows"


def test_label_scans_are_byte_capped_and_the_caller_keeps_the_inner_repository():
    inner, rec, seen = _ScanRepo(), Recorder(), []

    def labels(sealed, ids, before):
        repo = sealed["repository"]
        repo.scan("q", table_name="computed_moves")
        repo.scan("q", table_name="earnings_events")
        repo.scan("q", table_name="computed_moves", max_object_bytes=7)
        seen.append(repo.marker)
        return rec.labels(sealed, ids, before)

    sealed = _sealed()
    sealed["repository"] = inner
    out = y0_report(sealed, SPEC, max_object_bytes=1234, calendar=rec.calendar, chain_keys=rec.chain_keys,
                    labels=labels)
    assert out["y0"] == "2021"
    assert inner.calls == [("computed_moves", 1048576), ("earnings_events", 1234), ("computed_moves", 7)]
    assert seen == ["extra"]
    assert sealed["repository"] is inner


class _EmptyStore:
    def read_verified(self, ref):
        return b""


class _SizedChainRecorder(Recorder):
    """Runs the real object size check on each chain read, with the cap that the report passes it."""

    def __init__(self, byte_size):
        super().__init__()
        self.object_store = _EmptyStore()
        self.chain_ref = SimpleNamespace(object_id="option_chains-2026", content_hash="sha256:" + "0" * 64,
                                         byte_size=byte_size)

    def chain_keys(self, repository, snapshot, keys, *, max_object_bytes):
        objects.read_verified_bytes(self.object_store, self.chain_ref, max_bytes=max_object_bytes)
        return super().chain_keys(repository, snapshot, keys, max_object_bytes=max_object_bytes)


@pytest.mark.parametrize("byte_size, admitted", [
    pytest.param(268435457, True, id="2_pow_28_plus_1"),
    pytest.param(362807296, True, id="346_mib"),
    pytest.param(536870912, True, id="2_pow_29"),
    pytest.param(536870913, False, id="2_pow_29_plus_1"),
])
def test_default_chain_cap_admits_the_real_2026_chain_object_and_refuses_past_512_mib(byte_size, admitted):
    rec = _SizedChainRecorder(byte_size)
    if admitted:
        out = _run(rec)
        assert out["y0"] == "2021"
    else:
        with pytest.raises(DataError) as exc:
            _run(rec)
        assert exc.value.code == "RESULT_LIMIT_EXCEEDED"
    assert rec.calendar_calls == [(REPO, "snap", 268435456)]


def test_default_caps_keep_the_calendar_and_label_scan_limits():
    inner, rec = _ScanRepo(), Recorder()

    def labels(sealed, ids, before):
        repo = sealed["repository"]
        repo.scan("q", table_name="computed_moves")
        repo.scan("q", table_name="earnings_events")
        return rec.labels(sealed, ids, before)

    sealed = _sealed()
    sealed["repository"] = inner
    out = y0_report(sealed, SPEC, calendar=rec.calendar, chain_keys=rec.chain_keys, labels=labels)
    assert out["y0"] == "2021"
    assert rec.calendar_calls == [(inner, "snap", 268435456)]
    assert inner.calls == [("computed_moves", 1048576), ("earnings_events", 268435456)]
