"""EXP-188 real-store Sources adapter, slice 2: synthetic in-memory frames only (no store, no network)."""
import re
import types

import numpy as np
import pandas as pd
import pytest

from engine.v2.contracts import KeyPredicate
from engine.v2.data.errors import fail as data_fail
from engine.v2.ops.errors import OpsError
from experiments import size_pnl_ladder as ladder
from experiments import size_pnl_readers as readers
from experiments import size_pnl_tables as tbl

REPO = object()
MAX = 1 << 20
SNAP = types.SimpleNamespace(table_versions={"tier4_forecasts": types.SimpleNamespace(manifest_hash="H")})
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
                                        "realized_move_pct": [-3.0, 2.0], "implied_move_pct": [4.0, 5.0]}),
        "iv_crush_targets": pd.DataFrame({"ticker": ["AAA", "BBB"], "event_date": [D1, D2],
                                          "crush_pct_iv30": [-10.0, -20.0], "skipped": [False, True]}),
        "feature_panel": pd.DataFrame({"ticker": ["AAA", "BBB"], "date": [D1, D2],
                                       "pre_iv30": [0.3, 0.4], "mcap_usd": [5e9, 2e9]}),
    }
    frames.update(over)
    return frames


class _Read:
    """In-memory stand-in for read_table. Records every call's kwargs, applies partition_keys (by year of the date column)
    and ticker `in` key_filters, then batch_filter, and records what the caller kept."""
    def __init__(self, frames):
        self.frames = frames
        self.kept = []
        self.calls = []

    def __call__(self, repository, snapshot, table, columns, *, partition_keys=None, key_filter=(),
                 batch_filter=None, max_object_bytes=None):
        self.calls.append({"table": table, "partition_keys": partition_keys, "key_filter": key_filter,
                           "max_object_bytes": max_object_bytes})
        if table not in self.frames:
            raise data_fail("CONTRACT_MISMATCH", "table is not part of this snapshot", details={"table_name": table})
        if any(c not in self.frames[table].columns for c in columns):
            raise data_fail("CONTRACT_MISMATCH", "a requested column is not in the table", details={"table_name": table})
        frame = self.frames[table]
        if partition_keys is not None:
            date_col = "date" if "date" in frame.columns else "event_date"
            years = pd.to_datetime(frame[date_col]).dt.year.astype(str)
            frame = frame[years.isin(partition_keys).to_numpy()]
        for pred in key_filter:
            assert pred.operator == "in" and pred.column == "ticker"
            frame = frame[frame["ticker"].isin(pred.values).to_numpy()]
        frame = frame[list(columns)].copy()
        if batch_filter is not None:
            frame = batch_filter(frame)
        self.kept.append((table, frame.copy()))
        return frame.reset_index(drop=True)


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


def _build(frames=None, *, cal=None, admitted=ADMITTED, snapshot=SNAP, expected_hash="H"):
    read = _Read(_frames() if frames is None else frames)
    cal = _Cal() if cal is None else cal
    ns = tbl.tables(REPO, snapshot, admitted, max_object_bytes=MAX, expected_hash=expected_hash, read=read,
                    calendar=lambda repository, snap, *, max_object_bytes: cal)
    return ns, read


def _set(frame, column, value):
    out = frame.copy()
    out[column] = pd.Series([value, *out[column].iloc[1:]], dtype=object)
    return out


# --- table_hash and missing table --------------------------------------------

def test_table_hash_is_forecast_manifest_hash():
    ns, _ = _build()
    assert ns.table_hash == "H"


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

    tbl.tables(REPO, SNAP, ADMITTED, max_object_bytes=MAX, expected_hash="H", read=_Read(_frames()), calendar=calendar)
    assert seen == {"max_object_bytes": MAX}


def test_missing_forecast_table_refused_before_any_read():
    read = _Read(_frames())
    snap = types.SimpleNamespace(table_versions={})
    with pytest.raises(OpsError) as exc:
        tbl.tables(REPO, snap, ADMITTED, max_object_bytes=MAX, expected_hash="H", read=read,
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


def test_production_rows_without_forecast_are_omitted():
    frames = _frames(tier4_forecasts=pd.DataFrame({"ticker": ["AAA"], "event_date": [D1],
                                                   "pred_abs_move": [4.0], "pred_iv_crush_30": [-1.0]}))
    ns, _ = _build(frames)
    assert list(ns.fetch(PROD, ["e1", "e2"])["event_id"]) == ["e1"]


def test_production_duplicate_key_refused():
    dup = pd.DataFrame({"ticker": ["AAA", "AAA"], "event_date": [D1, D1],
                        "pred_abs_move": [4.0, 5.0], "pred_iv_crush_30": [-1.0, -2.0]})
    ns, _ = _build(_frames(tier4_forecasts=dup))
    with pytest.raises(OpsError) as exc:
        ns.fetch(PROD, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


# --- label fetch -------------------------------------------------------------

def test_label_fetch_takes_abs_move_and_nans_skipped_crush():
    ns, _ = _build()
    out = ns.fetch(LABEL, ["e1", "e2"]).set_index("event_id")
    assert out.loc["e1", "abs_move"] == 3.0 and out.loc["e1", "crush_pct_iv30"] == -10.0
    assert out.loc["e2", "abs_move"] == 2.0 and np.isnan(out.loc["e2", "crush_pct_iv30"])


def test_label_absent_rows_come_back_nan():
    frames = _frames(
        computed_moves=pd.DataFrame({"ticker": ["AAA"], "event_date": [D1], "realized_move_pct": [-3.0], "implied_move_pct": [4.0]}),
        iv_crush_targets=pd.DataFrame({"ticker": ["AAA"], "event_date": [D1], "crush_pct_iv30": [-10.0], "skipped": [False]}))
    ns, _ = _build(frames)
    out = ns.fetch(LABEL, ["e2"]).set_index("event_id")
    assert list(out.index) == ["e2"]
    assert np.isnan(out.loc["e2", "abs_move"]) and np.isnan(out.loc["e2", "crush_pct_iv30"])


def test_label_read_is_given_only_requested_keys():
    ns, read = _build()
    ns.fetch(LABEL, ["e1"])
    for table in ("computed_moves", "iv_crush_targets"):
        seen = [frame for name, frame in read.kept if name == table][-1]
        assert set(zip(seen["ticker"], seen["event_date"])) == {("AAA", D1)}


def test_label_read_passes_year_partition_and_ticker_filter():
    frames = _frames(computed_moves=pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"], "event_date": [D1, pd.Timestamp("2024-06-03"), D2],
        "realized_move_pct": [-3.0, 9.0, 2.0], "implied_move_pct": [4.0, 4.0, 5.0]}))
    ns, read = _build(frames)
    out = ns.fetch(LABEL, ["e1"])
    by_table = {c["table"]: c for c in read.calls if c["table"] in ("computed_moves", "iv_crush_targets")}
    for table in ("computed_moves", "iv_crush_targets"):
        assert by_table[table]["partition_keys"] == ["2025"]
        assert by_table[table]["key_filter"] == (KeyPredicate(column="ticker", operator="in", values=("AAA",)),)
    seen = [frame for name, frame in read.kept if name == "computed_moves"][-1]
    assert list(zip(seen["ticker"], seen["event_date"])) == [("AAA", D1)]
    assert out.set_index("event_id").loc["e1", "abs_move"] == 3.0


def test_empty_keys_read_nothing():
    ns, read = _build()
    before = len(read.calls)
    assert ns.fetch(LABEL, []).empty
    assert ns.fetch(PROD, []).empty
    assert read.calls[before:] == []


READ_LOG = re.compile(r"^\[pnl_tables\] read (\w+) keys=(\d+) rows=(\d+) elapsed=\d+\.\d+s rss=\d+MiB$", re.M)


def test_read_progress_logged_once_per_read(capsys):
    ns, _ = _build()
    ns.fetch(LABEL, ["e1"])
    logged = READ_LOG.findall(capsys.readouterr().err)
    assert logged == [("earnings_events", "2", "2"), ("computed_moves", "1", "1"), ("iv_crush_targets", "1", "1")]


def test_unsupported_column_set_refused():
    ns, _ = _build()
    with pytest.raises(OpsError) as exc:
        ns.fetch(("abs_move",), ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_value_invalid")


# --- facts -------------------------------------------------------------------

def test_facts_returns_indexed_sizing_inputs():
    ns, _ = _build()
    out = ns.facts(["e1", "e2"])
    assert list(out.index) == ["e1", "e2"]
    assert list(out.columns) == ["pre_iv30", "mcap_usd", "implied_move"]
    assert out["pre_iv30"].tolist() == [0.3, 0.4]
    assert out["mcap_usd"].tolist() == [5e9, 2e9]
    assert out["implied_move"].tolist() == [4.0, 5.0]


@pytest.mark.parametrize("table, frame", [
    ("computed_moves", pd.DataFrame({"ticker": ["AAA"], "event_date": [D1], "realized_move_pct": [-3.0], "implied_move_pct": [4.0]})),
    ("feature_panel", pd.DataFrame({"ticker": ["AAA"], "date": [D1], "pre_iv30": [0.3], "mcap_usd": [5e9]})),
], ids=["computed_moves_row_missing", "feature_panel_row_missing"])
def test_facts_missing_row_refused(table, frame):
    ns_frames = _frames(**{table: frame})
    with pytest.raises(OpsError) as exc:
        _build(ns_frames)[0].facts(["e1", "e2"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


@pytest.mark.parametrize("table, column, value", [
    ("feature_panel", "pre_iv30", np.nan),
    ("feature_panel", "mcap_usd", np.inf),
    ("feature_panel", "pre_iv30", 3.4028235e38),
    ("feature_panel", "mcap_usd", 5e3),
    ("feature_panel", "pre_iv30", "abc"),
    ("computed_moves", "implied_move_pct", 0.0),
], ids=["nan", "inf", "flt_max", "mcap_thousands", "non_numeric", "implied_not_positive"])
def test_facts_invalid_values_refused(table, column, value):
    frames = _frames()
    frames[table] = _set(frames[table], column, value)
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

def test_contract_mismatch_from_read_maps_to_source_table_missing():
    frames = _frames()
    del frames["computed_moves"]
    ns, _ = _build(frames)
    with pytest.raises(OpsError) as exc:
        ns.fetch(LABEL, ["e1"])
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "source_table_missing")


# --- end to end through the readers ------------------------------------------

def test_readers_accept_production_and_label_frames():
    ns, _ = _build()
    assert ns.table_hash == "H"
    prod = readers.read_production(ns.fetch, ADMITTED, expected_hash="H", table_hash=ns.table_hash)
    assert prod["table_hash"] == "H"
    assert prod["binding"] == readers._production_binding("H", prod["frame"])
    assert list(prod["frame"].index) == ["e1", "e2"]
    labels = readers.read_labels(ns.fetch, ADMITTED)
    assert list(labels.index) == ["e1"] and labels.loc["e1", "abs_move"] == 3.0
