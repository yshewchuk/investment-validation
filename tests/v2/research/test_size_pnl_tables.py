"""EXP-188 real-store Sources adapter, slice 2: synthetic in-memory frames only (no store, no network)."""
import re
import types

import numpy as np
import pandas as pd
import pytest

from engine.v2.contracts import KeyPredicate
from engine.v2.data.errors import DataError
from engine.v2.data.errors import fail as data_fail
from engine.v2.ops.errors import OpsError
from experiments import size_pnl_ladder as ladder
from experiments import size_pnl_readers as readers
from experiments import size_pnl_tables as tbl

REPO = object()
MAX = 1 << 20
HASH = "tier4-manifest-9c2e41"  # literal manifest hash; deliberately unlike any other fake value
SNAP = types.SimpleNamespace(table_versions={"tier4_forecasts": types.SimpleNamespace(manifest_hash=HASH)})
D1, D2 = pd.Timestamp("2025-01-02"), pd.Timestamp("2025-02-03")
DAY = pd.Timedelta(days=1)
PROD = tuple(readers.PRODUCTION_COLUMNS)
LABEL = tuple(readers.LABEL_COLUMNS)
ADMITTED = readers.Admitted({"events": pd.DataFrame({"event_id": ["e1", "e2"], "event_date": [D1, D2]}), "excluded": {}})


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _events(sessions=("AMC", "BMO")):
    return pd.DataFrame({"event_id": ["e1", "e2"], "ticker": ["AAA", "BBB"], "event_date": [D1, D2], "session": list(sessions)})


def _frames(**over):
    frames = {
        "earnings_events": _events(),
        "tier4_forecasts": pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"], "event_date": [D1, D2, D1],
                                         "pred_abs_move": [4.0, 5.0, 6.0], "pred_iv_crush_30": [-1.0, -2.0, -3.0]}),
        "computed_moves": pd.DataFrame({"ticker": ["AAA", "BBB"], "event_date": [D1, D2],
                                        "realized_move_pct": [-3.0, 2.0], "skipped": [False, False]}),
        "iv_crush_targets": pd.DataFrame({"ticker": ["AAA", "BBB"], "event_date": [D1, D2],
                                          "crush_pct_iv30": [-10.0, -20.0], "skipped": [False, True]}),
        "feature_panel": pd.DataFrame({"ticker": ["AAA", "BBB"], "date": [D1, D2], "pre_iv30": [0.3, 0.4],
                                       "mcap_usd": [5e9, 2e9], "or_implied": [4.0, 5.0]}),
    }
    frames.update(over)
    return frames


class _Read:
    """In-memory stand-in for read_table. Takes no partition_keys (passing one raises TypeError), applies ticker `in`
    key_filters, then batch_filter, and records what the caller kept."""
    def __init__(self, frames):
        self.frames = frames
        self.kept = []
        self.calls = []
        self.labels = None

    def __call__(self, repository, snapshot, table, columns, *, key_filter=(), batch_filter=None, max_object_bytes=None):
        self.calls.append({"table": table, "key_filter": key_filter, "max_object_bytes": max_object_bytes})
        if table not in self.frames:
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot", details={"table_name": table})
        if any(c not in self.frames[table].columns for c in columns):
            raise data_fail("CONTRACT_MISMATCH", "a requested column is not in the table", details={"table_name": table})
        frame = self.frames[table]
        for pred in key_filter:
            assert pred.operator == "in" and pred.column == "ticker"
            frame = frame[frame["ticker"].isin(pred.values).to_numpy()]
        frame = frame[list(columns)].copy()
        if batch_filter is not None:
            frame = batch_filter(frame)
        self.kept.append((table, frame.copy()))
        return frame.reset_index(drop=True)


class _Labels:
    """Stand-in for label_rows: returns only the requested events' rows of the table frame, and records each call."""
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def __call__(self, repository, snapshot, table, columns, want, *, max_object_bytes):
        self.calls.append({"table": table, "events": list(want["event_id"]), "max_object_bytes": max_object_bytes})
        if table not in self.frames:
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot", details={"table_name": table})
        frame = self.frames[table]
        if any(c not in frame.columns for c in columns):
            raise data_fail("CONTRACT_MISMATCH", "a requested column is not in the table", details={"table_name": table})
        keys = set(zip(want["ticker"], want["event_date"]))
        hit = np.asarray([key in keys for key in zip(frame["ticker"], frame["event_date"])], dtype=bool)
        return frame.loc[hit, list(columns)].to_dict("records")


class _Cal:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = []

    def last_pre_print(self, day, session):
        self.calls.append(("last_pre_print", day, session))
        if day in self.missing:
            raise KeyError(f"no trading day before {day.date()}")
        return day - DAY

    def first_post_print(self, day, session):
        self.calls.append(("first_post_print", day, session))
        return day + DAY


def _build(frames=None, *, cal=None, admitted=ADMITTED, snapshot=SNAP, expected_hash=HASH):
    frames = _frames() if frames is None else frames
    read, labels = _Read(frames), _Labels(frames)
    cal = _Cal() if cal is None else cal
    ns = tbl.tables(REPO, snapshot, admitted, max_object_bytes=MAX, expected_hash=expected_hash, read=read,
                    label_rows=labels, calendar=lambda repository, snap, *, max_object_bytes: cal)
    read.labels = labels
    return ns, read


def _set(frame, column, value):
    out = frame.copy()
    out[column] = pd.Series([value, *out[column].iloc[1:]], dtype=object)
    return out


# --- table_hash and missing table --------------------------------------------

def test_table_hash_is_forecast_manifest_hash():
    ns, _ = _build()
    assert ns.table_hash == HASH


def test_hash_mismatch_refused_before_any_read():
    read = _Read(_frames())
    with pytest.raises(OpsError) as exc:
        tbl.tables(REPO, SNAP, ADMITTED, max_object_bytes=MAX, expected_hash="OTHER", read=read,
                   calendar=lambda r, s, *, max_object_bytes: _Cal())
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "production_hash_mismatch")
    assert read.calls == []


def test_calendar_receives_max_object_bytes():
    seen = {}

    def calendar(repository, snap, *, max_object_bytes):
        seen["max_object_bytes"] = max_object_bytes
        return _Cal()

    tbl.tables(REPO, SNAP, ADMITTED, max_object_bytes=MAX, expected_hash=HASH, read=_Read(_frames()), calendar=calendar)
    assert seen == {"max_object_bytes": MAX}


def test_calendar_contract_mismatch_refused_as_missing():
    def calendar(repository, snap, *, max_object_bytes):
        raise data_fail("CONTRACT_MISMATCH", "calendar table is absent")

    with pytest.raises(OpsError) as exc:
        tbl.tables(REPO, SNAP, ADMITTED, max_object_bytes=MAX, expected_hash=HASH, read=_Read(_frames()),
                   label_rows=_Labels(_frames()), calendar=calendar)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


def test_missing_forecast_table_refused_before_any_read():
    read = _Read(_frames())
    snap = types.SimpleNamespace(table_versions={})
    with pytest.raises(OpsError) as exc:
        tbl.tables(REPO, snap, ADMITTED, max_object_bytes=MAX, expected_hash=HASH, read=read,
                   calendar=lambda r, s, *, max_object_bytes: _Cal())
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")
    assert read.kept == [] and read.calls == []


def test_missing_earnings_events_refused_at_construction():
    frames = _frames()
    del frames["earnings_events"]
    with pytest.raises(OpsError) as exc:
        _build(frames)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


@pytest.mark.parametrize("events", [
    pd.concat([_events(), _events().iloc[[0]]], ignore_index=True),
], ids=["duplicate_event_id"])
def test_duplicate_event_id_refused(events):
    with pytest.raises(OpsError) as exc:
        _build(_frames(earnings_events=events))
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


def test_admitted_event_absent_from_earnings_events_refused():
    admitted = readers.Admitted({"events": pd.DataFrame({"event_id": ["e1", "e2", "e3"], "event_date": [D1, D2, D1]}), "excluded": {}})
    with pytest.raises(OpsError) as exc:
        _build(admitted=admitted)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


# --- production fetch --------------------------------------------------------

def test_production_fetch_joins_to_event_id_and_restricts_to_requested():
    ns, _ = _build()
    out = ns.fetch(PROD, ["e1"])
    assert list(out["event_id"]) == ["e1"]
    assert out.loc[0, "pred_abs_move"] == 4.0 and out.loc[0, "pred_iv_crush_30"] == -1.0
    assert out["horizon_sessions"].tolist() == [1]
    both = ns.fetch(PROD, ["e1", "e2"])
    assert sorted(both["event_id"]) == ["e1", "e2"]
    assert "CCC" not in set(both["event_id"])


def test_production_missing_forecast_refused():
    frames = _frames(tier4_forecasts=pd.DataFrame({"ticker": ["AAA"], "event_date": [D1],
                                                   "pred_abs_move": [4.0], "pred_iv_crush_30": [-1.0]}))
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(PROD, ["e1", "e2"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


def test_production_duplicate_key_refused():
    dup = pd.DataFrame({"ticker": ["AAA", "AAA"], "event_date": [D1, D1],
                        "pred_abs_move": [4.0, 5.0], "pred_iv_crush_30": [-1.0, -2.0]})
    ns, _ = _build(_frames(tier4_forecasts=dup))
    with pytest.raises(OpsError) as exc:
        ns.fetch(PROD, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


@pytest.mark.parametrize("column, value", [
    ("pred_abs_move", np.nan),
    ("pred_abs_move", np.inf),
    ("pred_abs_move", 3.4028235e38),
    ("pred_iv_crush_30", 3.0e38),
], ids=["nan", "inf", "flt_max", "at_bound"])
def test_production_sentinel_or_nonfinite_prediction_refused(column, value):
    frames = _frames()
    frames["tier4_forecasts"] = _set(frames["tier4_forecasts"], column, value)  # row 0 is AAA/D1, event e1
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(PROD, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


# --- label fetch -------------------------------------------------------------

def test_label_fetch_takes_abs_move_and_nans_skipped_crush():
    ns, _ = _build()
    out = ns.fetch(LABEL, ["e1", "e2"]).set_index("event_id")
    assert out.loc["e1", "abs_move"] == 3.0 and out.loc["e1", "crush_pct_iv30"] == -10.0
    assert out.loc["e2", "abs_move"] == 2.0 and np.isnan(out.loc["e2", "crush_pct_iv30"])


def test_label_skipped_blanks_move_too():
    frames = _frames()
    frames["computed_moves"] = _set(frames["computed_moves"], "skipped", True)
    ns, _ = _build(frames)
    out = ns.fetch(LABEL, ["e1"]).set_index("event_id")
    assert np.isnan(out.loc["e1", "abs_move"]) and out.loc["e1", "crush_pct_iv30"] == -10.0


def test_label_nan_value_allowed_and_comes_back_nan():
    frames = _frames()
    frames["computed_moves"] = _set(frames["computed_moves"], "realized_move_pct", np.nan)
    ns, _ = _build(frames)
    out = ns.fetch(LABEL, ["e1"]).set_index("event_id")
    assert np.isnan(out.loc["e1", "abs_move"])


def test_label_missing_row_refused():
    frames = _frames(
        computed_moves=pd.DataFrame({"ticker": ["AAA"], "event_date": [D1], "realized_move_pct": [-3.0], "skipped": [False]}),
        iv_crush_targets=pd.DataFrame({"ticker": ["AAA"], "event_date": [D1], "crush_pct_iv30": [-10.0], "skipped": [False]}))
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(LABEL, ["e2"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


def test_label_two_rows_for_one_event_refused():
    frames = _frames(computed_moves=pd.DataFrame({
        "ticker": ["AAA", "AAA"], "event_date": [D1, D1], "realized_move_pct": [-3.0, -4.0], "skipped": [False, False]}))
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


@pytest.mark.parametrize("table, column, value", [
    ("computed_moves", "realized_move_pct", np.inf),
    ("computed_moves", "realized_move_pct", 3.4028235e38),
    ("iv_crush_targets", "crush_pct_iv30", -np.inf),
    ("iv_crush_targets", "crush_pct_iv30", 3.0e38),
], ids=["move_inf", "move_flt_max", "crush_inf", "crush_at_bound"])
def test_label_sentinel_or_inf_refused(table, column, value):
    frames = _frames()
    frames[table] = _set(frames[table], column, value)  # row 0 is AAA/D1, event e1
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


def test_label_read_is_called_only_with_requested_events_once_per_table():
    frames = _frames(computed_moves=pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"], "event_date": [D1, pd.Timestamp("2024-06-03"), D2],
        "realized_move_pct": [-3.0, 9.0, 2.0], "skipped": [False, False, False]}))
    ns, read = _build(frames)
    out = ns.fetch(LABEL, ["e1"])
    assert [c["table"] for c in read.labels.calls] == ["computed_moves", "iv_crush_targets"]
    assert all(c["events"] == ["e1"] for c in read.labels.calls)
    assert out.set_index("event_id").loc["e1", "abs_move"] == 3.0


def test_plan_refusal_from_label_rows_propagates_unchanged():
    def refuse(repository, snapshot, table, columns, want, *, max_object_bytes):
        raise data_fail("HOLDOUT_ACCESS_DENIED", "event is not isolated")

    ns = tbl.tables(REPO, SNAP, ADMITTED, max_object_bytes=MAX, expected_hash=HASH, read=_Read(_frames()),
                    label_rows=refuse, calendar=lambda r, s, *, max_object_bytes: _Cal())
    with pytest.raises(DataError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert exc.value.code == "HOLDOUT_ACCESS_DENIED"


def test_contract_mismatch_from_label_read_maps_to_source_table_missing():
    frames = _frames()
    del frames["computed_moves"]
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


class _Batch:
    def __init__(self, rows):
        self._rows = rows

    def to_pylist(self):
        return list(self._rows)


class _ScanRepo:
    def table_contract(self, snapshot, table):
        return ("contract", table)

    def scan(self, request, *, table_name, max_object_bytes):
        return [_Batch([{"ticker": request[0], "event_date": request[1]}])]


LABEL_LOG = re.compile(r"^\[pnl_tables\] read (\w+) events=(\d+) rows=(\d+) elapsed=\d+\.\d+s rss=\d+MiB$", re.M)


def test_label_rows_plans_only_requested_events_and_logs_one_line(monkeypatch, capsys):
    seen = {}

    def plan(repository, snapshot, candidates, contract, *, columns):
        seen.update(event_ids=list(candidates["event_id"]), contract=contract, columns=columns)
        return [(eid, (t, d), None) for eid, t, d in zip(candidates["event_id"], candidates["ticker"], candidates["event_date"])]

    monkeypatch.setattr(tbl, "plan_event_targets", plan)
    want = pd.DataFrame({"event_id": ["e1"], "ticker": ["AAA"], "event_date": [D1]})
    rows = tbl.label_rows(_ScanRepo(), SNAP, "computed_moves", ["ticker", "event_date"], want, max_object_bytes=MAX)
    assert rows == [{"ticker": "AAA", "event_date": D1}]
    assert seen == {"event_ids": ["e1"], "contract": ("contract", "computed_moves"), "columns": ("ticker", "event_date")}
    assert LABEL_LOG.findall(capsys.readouterr().err) == [("computed_moves", "1", "1")]


class _PlanRepo:
    """Fake repository for the real label_rows: per-table contract, and one scan per request returning one batch."""
    def __init__(self):
        self.scans = []

    def table_contract(self, snapshot, table):
        return ("contract", table)

    def scan(self, request, *, table_name, max_object_bytes):
        self.scans.append((request, table_name, max_object_bytes))
        return [_Batch([{"ticker": request[0], "event_date": request[1]}])]


def test_label_rows_real_plan_passes_contract_tuple_columns_and_scans_each_event(monkeypatch):
    seen = []

    def plan(repository, snapshot, candidates, contract, *, columns):
        seen.append({"repository": repository, "snapshot": snapshot, "contract": contract, "columns": columns})
        return [("e1", ("AAA", D1), "rec1"), ("e2", ("BBB", D2), "rec2")]

    monkeypatch.setattr(tbl, "plan_event_targets", plan)
    repo = _PlanRepo()
    want = pd.DataFrame({"event_id": ["e1", "e2"], "ticker": ["AAA", "BBB"], "event_date": [D1, D2]})
    rows = tbl.label_rows(repo, SNAP, "iv_crush_targets", ["ticker", "event_date"], want, max_object_bytes=MAX)
    assert len(seen) == 1
    assert seen[0]["repository"] is repo and seen[0]["snapshot"] is SNAP
    assert seen[0]["contract"] == repo.table_contract(SNAP, "iv_crush_targets") == ("contract", "iv_crush_targets")
    assert isinstance(seen[0]["columns"], tuple) and seen[0]["columns"] == ("ticker", "event_date")
    assert repo.scans == [(("AAA", D1), "iv_crush_targets", MAX), (("BBB", D2), "iv_crush_targets", MAX)]
    assert rows == [{"ticker": "AAA", "event_date": D1}, {"ticker": "BBB", "event_date": D2}]


def test_label_rows_real_plan_refusal_propagates_unchanged_and_scans_nothing(monkeypatch):
    refusal = data_fail("HOLDOUT_ACCESS_DENIED", "event is not isolated")

    def plan(repository, snapshot, candidates, contract, *, columns):
        raise refusal

    monkeypatch.setattr(tbl, "plan_event_targets", plan)
    repo = _PlanRepo()
    want = pd.DataFrame({"event_id": ["e1"], "ticker": ["AAA"], "event_date": [D1]})
    with pytest.raises(DataError) as exc:
        tbl.label_rows(repo, SNAP, "computed_moves", ["ticker"], want, max_object_bytes=MAX)
    assert exc.value is refusal
    assert exc.value.code == "HOLDOUT_ACCESS_DENIED"
    assert repo.scans == []


def test_empty_keys_read_nothing():
    ns, read = _build()
    before = len(read.calls)
    assert ns.fetch(LABEL, []).empty
    assert ns.fetch(PROD, []).empty
    assert read.calls[before:] == []
    assert read.labels.calls == []


def test_reads_never_take_partition_keys_and_logical_tables_filter_by_ticker():
    ns, read = _build()
    ns.fetch(PROD, ["e1"])
    ns.facts(["e1"])
    logical = [c for c in read.calls if c["table"] in ("tier4_forecasts", "feature_panel")]
    assert {c["table"] for c in logical} == {"tier4_forecasts", "feature_panel"}
    for call in logical:
        assert call["key_filter"] == (KeyPredicate(column="ticker", operator="in", values=("AAA",)),)


READ_LOG = re.compile(r"^\[pnl_tables\] read (\w+) keys=(\d+) rows=(\d+) elapsed=\d+\.\d+s rss=\d+MiB$", re.M)


def test_read_progress_logged_once_per_read(capsys):
    ns, _ = _build()
    ns.fetch(PROD, ["e1"])
    logged = READ_LOG.findall(capsys.readouterr().err)
    assert logged == [("earnings_events", "2", "2"), ("tier4_forecasts", "1", "1")]


def test_unsupported_column_set_refused():
    ns, _ = _build()
    with pytest.raises(OpsError) as exc:
        ns.fetch(("abs_move",), ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


# --- facts -------------------------------------------------------------------

def test_facts_returns_indexed_sizing_inputs_with_or_implied_as_implied_move():
    ns, _ = _build()
    out = ns.facts(["e1", "e2"])
    assert list(out.index) == ["e1", "e2"]
    assert list(out.columns) == ["pre_iv30", "mcap_usd", "implied_move"]
    assert out["pre_iv30"].tolist() == [0.3, 0.4]
    assert out["mcap_usd"].tolist() == [5e9, 2e9]
    assert out["implied_move"].tolist() == [4.0, 5.0]


def test_facts_never_reads_labels_or_computed_moves():
    ns, read = _build()
    ns.facts(["e1", "e2"])
    assert all(c["table"] != "computed_moves" for c in read.calls)
    assert read.labels.calls == []


def test_facts_missing_feature_panel_row_refused():
    frames = _frames(feature_panel=pd.DataFrame({"ticker": ["AAA"], "date": [D1], "pre_iv30": [0.3],
                                                 "mcap_usd": [5e9], "or_implied": [4.0]}))
    with pytest.raises(OpsError) as exc:
        _build(frames)[0].facts(["e1", "e2"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


@pytest.mark.parametrize("column, value", [
    ("pre_iv30", np.nan),
    ("mcap_usd", np.inf),
    ("pre_iv30", 3.4028235e38),
    ("mcap_usd", 5e3),
    ("pre_iv30", "abc"),
    ("or_implied", 0.0),
], ids=["nan", "inf", "flt_max", "mcap_thousands", "non_numeric", "implied_not_positive"])
def test_facts_invalid_values_refused(column, value):
    frames = _frames()
    frames["feature_panel"] = _set(frames["feature_panel"], column, value)
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.facts(["e1", "e2"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


@pytest.mark.parametrize("column, value, passes", [
    ("mcap_usd", 1e6, True),
    ("mcap_usd", 999999.99, False),
    ("pre_iv30", 2.9e38, True),
    ("pre_iv30", 3.0e38, False),
    ("pre_iv30", 0.0, False),
], ids=["mcap_exactly_1e6", "mcap_just_below_1e6", "pre_iv30_below_flt_max", "pre_iv30_at_flt_max", "pre_iv30_zero"])
def test_facts_boundaries(column, value, passes):
    frames = _frames()
    frames["feature_panel"] = _set(frames["feature_panel"], column, value)
    ns, _ = _build(frames)
    if passes:
        assert ns.facts(["e1", "e2"]).loc["e1", column] == value
    else:
        with pytest.raises(OpsError) as exc:
            ns.facts(["e1", "e2"])
        assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


# --- plan_frame --------------------------------------------------------------

def test_plan_frame_uses_calendar_windows_per_session_and_normalises_session():
    cal = _Cal()
    ns, _ = _build(_frames(earnings_events=_events(sessions=("amc", "bmo"))), cal=cal)
    plan = ns.plan_frame
    assert list(plan.columns) == ["event_id", "ticker", "event_date", "session", "entry_date", "exit_date"]
    assert plan["event_id"].tolist() == ["e1", "e2"]
    assert plan["session"].tolist() == ["AMC", "BMO"]
    assert plan["entry_date"].tolist() == [D1 - DAY, D2 - DAY]
    assert plan["exit_date"].tolist() == [D1 + DAY, D2 + DAY]
    assert ("last_pre_print", D1, "AMC") in cal.calls and ("first_post_print", D2, "BMO") in cal.calls


def test_plan_frame_drops_bad_session_and_logs_counts(capsys):
    ns, _ = _build(_frames(earnings_events=_events(sessions=("AMC", "XYZ"))))
    assert ns.plan_frame["event_id"].tolist() == ["e1"]
    assert "[pnl_tables] plan_frame events=2 dropped=1" in capsys.readouterr().err


def test_plan_frame_drops_calendar_key_error():
    ns, _ = _build(cal=_Cal(missing=[D1]))
    assert ns.plan_frame["event_id"].tolist() == ["e2"]


def test_plan_frame_refuses_when_no_event_survives():
    with pytest.raises(OpsError) as exc:
        _build(_frames(earnings_events=_events(sessions=("XYZ", "XYZ"))))
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "plan_frame_empty")


def test_plan_frame_passes_check_plan():
    ns, _ = _build()
    ladder.check_plan(ADMITTED, ns.plan_frame)


# --- read errors -------------------------------------------------------------

def test_contract_mismatch_from_forecast_read_maps_to_source_table_missing():
    frames = _frames()
    del frames["tier4_forecasts"]
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(PROD, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


# --- end to end through the readers ------------------------------------------

def test_readers_accept_production_and_label_frames():
    ns, _ = _build()
    assert ns.table_hash == HASH
    prod = readers.read_production(ns.fetch, ADMITTED, expected_hash=HASH, table_hash=ns.table_hash)
    assert prod["table_hash"] == HASH
    assert prod["binding"] == readers._production_binding(HASH, prod["frame"])
    assert list(prod["frame"].index) == ["e1", "e2"]
    labels = readers.read_labels(ns.fetch, ADMITTED)
    assert list(labels.index) == ["e1"] and labels.loc["e1", "abs_move"] == 3.0
